"""Decide what to DO with a caller utterance, before any LLM call.

The problem this solves: the first version sent every single utterance through
the RAG and then the LLM. So "hello" cost an embedding + a vector search + a
Groq round-trip, and a question the brochure simply cannot answer (today's
price) produced a confident-sounding non-answer instead of a callback.

Five routes:

  SMALLTALK  greetings, thanks, "can you hear me" -> canned reply, ZERO network
             calls, answered in microseconds.
  CHAT       the caller said something conversational, or answered a question we
             asked ("this is Roy") -> LLM with the transcript and NO project
             knowledge. Without this route those turns fall through to ESCALATE,
             and the bot answers "let me have someone call you back" to a man
             telling it his name. That happened on the first live call.
  KNOWLEDGE  a project question the KB covers -> RAG context + LLM.
  ESCALATE   a question the brochure genuinely cannot answer (price, current
             availability, booking, possession) OR anything too far from the KB
             -> promise a human callback and capture the lead.
  HANDOFF    caller explicitly asked for a person -> dial the sales agent.

The escalation set is embedded in the same collection as the knowledge, so ONE
vector search answers both "do I know this?" and "should a human take it?".
"""
import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import settings
from rag import retriever

log = logging.getLogger("jarvis.router")


class Route(str, Enum):
    SMALLTALK = "smalltalk"
    CHAT = "chat"            # conversational turn -> LLM with history, no RAG
    KNOWLEDGE = "knowledge"
    ESCALATE = "escalate"
    HANDOFF = "handoff"


# The generic fallback when nothing in the KB is close enough. This is the
# "too deep for the data" case the brochure was never going to cover.
GENERIC_ESCALATION = (
    "That's a good question, and I'd rather get you an exact answer than guess. "
    "Let me have one of our sales experts call you back on this. "
    "May I have your name?"
)

# Repeating the full callback line every turn is what made the first live call
# unbearable. If we escalated on the previous turn, say less.
GENERIC_ESCALATION_REPEAT = (
    "I'll note that down for our team as well. "
    "Is there anything about the township itself I can help with?"
)

# "my name is Roy", "this is Roy", "I'm Roy", "Roy here", "myself Roy"
_SELF_INTRO = re.compile(
    r"^(?:my name(?:'s| is)?|myself|i am|i'm|im|this is|it's|its|here is)\s+(.{1,40})$",
    re.I)
_NAME_TRAILING = re.compile(r"^(.{2,30}?)\s+(?:here|speaking|this side)$", re.I)

# Callers open with a greeting and then introduce themselves in one breath:
# "Hi. This is Roy." Deepgram delivers that as a single utterance, so the
# self-intro patterns must be able to see past the greeting.
_LEAD_GREETING = re.compile(
    r"^(?:hi|hii|hello|hey|yeah|yes|ya|ok|okay|good\s+(?:morning|afternoon|evening)|"
    r"vanakkam|so|and|um|uh)\b[\s,.!-]*", re.I)

# Words that mean the "name" we captured is not a name.
_NOT_A_NAME = {
    "a", "an", "the", "good", "nice", "great", "fine", "ok", "okay", "not",
    "interested", "looking", "calling", "asking", "from", "for", "about",
    "price", "plot", "project", "township", "land", "house", "site", "sir",
    "madam", "yes", "no", "correct", "right", "true", "sure", "just",
}

# A turn that ends in a question mark, or opens with an interrogative, is the
# caller ASKING something. Anything else, mid-conversation, is usually them
# ANSWERING something - and answers must never be routed like questions.
_INTERROGATIVE = re.compile(
    r"^(what|where|when|which|who|whom|whose|why|how|is|are|was|were|do|does|"
    r"did|can|could|will|would|shall|should|may|might|have|has|had|tell|give|"
    r"explain|show|send)\b", re.I)

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")

# Filler the STT leaves in that shouldn't stop a smalltalk match.
_FILLER = {"um", "uh", "er", "ah", "hmm", "like", "just", "so", "well",
           "actually", "please", "sir", "madam", "ma'am"}


@dataclass
class Decision:
    route: Route
    reply: Optional[str] = None          # speak verbatim (smalltalk / escalation)
    context: str = ""                    # RAG grounding block (KNOWLEDGE)
    confident: bool = True               # False = weak match, hedge in the prompt
    intent: str = ""                     # smalltalk intent or escalation id
    capture: str = ""                    # what to log on the lead row
    distance: float = 2.0                # best cosine distance seen
    action: str = ""                     # e.g. "repeat_last", "end_call"
    name: str = ""                       # caller name captured from a self-intro
    meta: dict = field(default_factory=dict)


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", text).strip()


def _strip_filler(norm: str) -> str:
    words = [w for w in norm.split() if w not in _FILLER]
    return " ".join(words)


class Router:
    """Holds the compiled smalltalk table. One instance per process."""

    def __init__(self, smalltalk: dict):
        self.smalltalk = smalltalk or {}
        # exact-match lookup: normalized pattern -> intent name
        self._exact = {}
        for intent, spec in self.smalltalk.items():
            for pat in spec.get("patterns", []):
                self._exact[normalize(pat)] = intent

    # ------------------------------------------------------------ smalltalk
    def match_smalltalk(self, text: str):
        """Return (intent, spec) or None. Deterministic and instant.

        Only very short utterances are eligible: "hi" is a greeting, but
        "hi, what is the price of a plot" is a pricing question that happens to
        start with a greeting, and must never be swallowed here.
        """
        norm = normalize(text)
        if not norm:
            return None

        intent = self._exact.get(norm)
        if intent is None:
            stripped = _strip_filler(norm)
            intent = self._exact.get(stripped)
            norm = stripped or norm

        if intent is None:
            return None

        # Guard: a long utterance that merely *contains* a greeting is a real
        # question. Smalltalk only wins when it is essentially the whole turn.
        if len(norm.split()) > 6:
            return None

        return intent, self.smalltalk[intent]

    # ------------------------------------------------------------ self-intro
    @staticmethod
    def extract_name(text: str):
        """Return a caller name from a self-introduction, or None.

        Six of the ten turns in the first live call were the caller trying to
        give their name after the bot asked for it. Every one escalated, so the
        bot asked again. This is the fix for that loop.
        """
        t = (text or "").strip().rstrip(".!?,")
        if not t or len(t.split()) > 7:
            return None

        # Peel at most two leading greetings: "Hi. This is Roy." / "Hi hello, I'm Roy"
        for _ in range(2):
            stripped = _LEAD_GREETING.sub("", t, count=1)
            if stripped == t:
                break
            t = stripped.strip()
        if not t:
            return None

        m = _SELF_INTRO.match(t) or _NAME_TRAILING.match(t)
        if not m:
            return None

        cand = m.group(1).strip().rstrip(".!?,")
        words = [w for w in re.split(r"[\s.]+", cand) if w]
        if not words or len(words) > 3:
            return None
        if any(w.lower() in _NOT_A_NAME for w in words):
            return None
        if not all(re.fullmatch(r"[A-Za-z][A-Za-z'-]*", w) for w in words):
            return None
        return " ".join(w.capitalize() for w in words)

    @staticmethod
    def looks_like_question(text: str) -> bool:
        t = (text or "").strip()
        return t.endswith("?") or bool(_INTERROGATIVE.match(t))

    # ------------------------------------------------------------ main entry
    async def decide(self, text: str, ctx=None) -> Decision:
        """`ctx` carries the little conversational state routing needs:
            bot_asked_question  - the previous bot turn ended in a question
            asked_for_name      - that question was specifically for their name
            escalated_last_turn - we already promised a callback last turn
        Without it the router classifies every utterance as if it were the first,
        which is exactly how "This is Roy" became an escalation.
        """
        ctx = ctx or {}
        text = (text or "").strip()
        if not text:
            return Decision(Route.SMALLTALK, reply=None, intent="empty")

        # 0. The caller is telling us who they are. This must beat everything
        #    except an explicit request, because it is an ANSWER, not a query.
        name = self.extract_name(text)
        if name is None and ctx.get("asked_for_name") and not self.looks_like_question(text):
            # We just asked for their name and they replied with something short
            # that is not a question - e.g. bare "Denise."
            bare = text.strip().rstrip(".!?,")
            words = [w for w in re.split(r"[\s.]+", bare) if w]
            uniq = list(dict.fromkeys(w.lower() for w in words))
            # Deepgram repeats a name when the caller repeats it: "Denise. Denise."
            if 1 <= len(uniq) <= 2 and all(
                    re.fullmatch(r"[A-Za-z][A-Za-z'-]*", w) for w in words) \
                    and not any(w in _NOT_A_NAME for w in uniq):
                name = " ".join(w.capitalize() for w in uniq)

        if name:
            log.info("route=CHAT (self-intro) name=%r text=%r", name, text[:60])
            return Decision(route=Route.CHAT, intent="self_intro",
                            name=name, distance=0.0)

        # 1. Smalltalk: no embedding, no vector search, no LLM.
        hit = self.match_smalltalk(text)
        if hit:
            intent, spec = hit
            log.info("route=SMALLTALK intent=%s text=%r", intent, text[:60])
            return Decision(
                route=Route.SMALLTALK,
                reply=spec.get("reply"),
                intent=intent,
                action=spec.get("action", "end_call" if spec.get("end_call") else ""),
                distance=0.0,
            )

        # 1.5 Quick location intent detection - avoid RAG for obvious location queries
        q_lower = normalize(text)
        if ("where" in q_lower or "locat" in q_lower or "address" in q_lower or 
            "situat" in q_lower or "which area" in q_lower):
            log.info("route=KNOWLEDGE (location intent shortcut) text=%r", text[:60])
            # Will be handled through normal RAG flow below, but marked as probable
            # location query. This just ensures location document gets high priority.
            pass  # Continue to embedding step

        # 2. One embedding, one search, covering knowledge AND escalation.
        emb = await retriever.aembed(text)
        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(None, retriever.search, emb, 32, None)

        if not hits:
            log.warning("route=ESCALATE (empty index) text=%r", text[:60])
            return Decision(Route.ESCALATE, reply=GENERIC_ESCALATION,
                            intent="empty_index", capture="unanswered")

        kb = [h for h in hits if h["meta"].get("kind") == "knowledge"]
        esc = [h for h in hits if h["meta"].get("kind") == "escalation"]

        best_kb = kb[0]["distance"] if kb else 2.0
        best_esc = esc[0]["distance"] if esc else 2.0

        # 3. Explicit escalation topic wins when it is close AND at least as
        #    close as the best knowledge hit. The tie-break matters: "what is
        #    the price" is lexically near several knowledge entries, but the
        #    escalation entry should still take it.
        if best_esc <= settings.route_escalation_threshold and best_esc <= best_kb:
            m = esc[0]["meta"]
            route = Route.HANDOFF if m.get("entry_id") == "esc_human" else Route.ESCALATE
            log.info("route=%s topic=%s d=%.3f (kb d=%.3f) text=%r",
                     route.name, m.get("entry_id"), best_esc, best_kb, text[:60])
            reply = m.get("answer") or GENERIC_ESCALATION
            if ctx.get("escalated_last_turn") and route == Route.ESCALATE:
                # Same promise twice in a row sounds like a broken record; keep
                # the topic-specific first half, drop the second ask for a name.
                reply = reply.split(" May I")[0].split(" Shall I")[0].strip()
            return Decision(
                route=route,
                reply=reply,
                intent=m.get("entry_id", ""),
                capture=m.get("capture", ""),
                distance=best_esc,
                meta={"answer_hint": m.get("answer_hint", "")},
            )

        def build_ctx(hits):
            return retriever.build_context(
                retriever.dedupe_by_entry(hits, settings.rag_top_k))

        # 4. Knowledge, strong or weak.
        if best_kb <= settings.route_weak_threshold:
            top = retriever.dedupe_by_entry(kb, settings.rag_top_k)
            confident = best_kb <= settings.route_strong_threshold
            log.info("route=KNOWLEDGE entry=%s d=%.3f confident=%s text=%r",
                     top[0]["meta"].get("entry_id"), best_kb, confident, text[:60])
            return Decision(
                route=Route.KNOWLEDGE,
                context=retriever.build_context(top),
                confident=confident,
                intent=top[0]["meta"].get("entry_id", ""),
                distance=best_kb,
            )

        # 5. Nothing close enough. Two very different situations live here, and
        #    the first version treated them identically - which is why the bot
        #    kept promising a callback to someone who was just saying their name.
        #
        #    (a) The caller asked a real question we cannot answer -> ESCALATE.
        #    (b) The caller said something conversational, or answered a question
        #        we just asked -> hand it to the LLM WITH HISTORY and no project
        #        knowledge. It has the persona and the transcript; it can carry a
        #        normal exchange without inventing facts.
        is_question = self.looks_like_question(text)
        mid_conversation = ctx.get("bot_asked_question")

        if not is_question or (mid_conversation and len(text.split()) <= 6):
            log.info("route=CHAT (no KB match, conversational) kb_d=%.3f text=%r",
                     best_kb, text[:60])
            return Decision(Route.CHAT, intent="conversational", distance=best_kb)

        if settings.route_always_answer and kb:
            # Original always-answer behaviour: use the nearest context rather
            # than refusing. Explicit escalation topics still escalated above,
            # so price/availability/booking questions never reach here.
            log.info("route=KNOWLEDGE (always-answer) kb_d=%.3f text=%r",
                     best_kb, text[:60])
            return Decision(Route.KNOWLEDGE,
                            context=build_ctx(kb),
                            confident=False, distance=best_kb,
                            intent=kb[0]["meta"].get("entry_id", ""))

        if ctx.get("escalated_last_turn"):
            log.info("route=ESCALATE (repeat) kb_d=%.3f text=%r", best_kb, text[:60])
            return Decision(Route.ESCALATE, reply=GENERIC_ESCALATION_REPEAT,
                            intent="out_of_scope",
                            capture="unanswered: %s" % text[:120], distance=best_kb)

        log.info("route=ESCALATE (no match) kb_d=%.3f esc_d=%.3f text=%r",
                 best_kb, best_esc, text[:60])
        return Decision(Route.ESCALATE, reply=GENERIC_ESCALATION,
                        intent="out_of_scope", capture="unanswered question: %s" % text[:120],
                        distance=best_kb)
