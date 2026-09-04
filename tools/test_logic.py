"""Deterministic unit tests for the parts that must never regress.

No embedding model, no vector store, no network - the retriever is stubbed with
fixed distances so the ROUTING LOGIC itself (threshold ordering, escalation
tie-breaks, smalltalk precedence) is tested in isolation from embedding quality.
Embedding quality is what `tools.eval_router` measures, and that one needs the
real model.

    python -m tools.test_logic
"""
import asyncio
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print("  PASS  %s" % name)
    else:
        print("  FAIL  %s %s" % (name, detail))
        FAILS.append(name)


# --------------------------------------------------------------- stub the RAG
# router imports `from rag import retriever`, so install a fake before import.
_stub = types.ModuleType("rag.retriever")
_stub.PLAN = []            # list of hit dicts the next search() returns


async def _aembed(text):
    return [0.0]


def _search(emb, n_results=32, kind=None):
    hits = list(_stub.PLAN)
    if kind:
        hits = [h for h in hits if h["meta"].get("kind") == kind]
    return hits


def _dedupe_by_entry(hits, top_k):
    seen, out = set(), []
    for h in hits:
        k = h["meta"].get("entry_id")
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def _build_context(hits):
    return "\n".join("### %s\n%s" % (h["meta"].get("topic", ""),
                                     h["meta"].get("answer", "")) for h in hits)


_stub.aembed = _aembed
_stub.search = _search
_stub.dedupe_by_entry = _dedupe_by_entry
_stub.build_context = _build_context
_stub.count = lambda: len(_stub.PLAN)

_ragpkg = types.ModuleType("rag")
_ragpkg.retriever = _stub
sys.modules["rag"] = _ragpkg
sys.modules["rag.retriever"] = _stub

from config import settings           # noqa: E402
from router import Route, Router, normalize   # noqa: E402
import llm                            # noqa: E402


def hit(kind, entry_id, distance, topic="t", answer="a", **extra):
    meta = {"kind": kind, "entry_id": entry_id, "topic": topic,
            "answer": answer, "status": "confirmed"}
    meta.update(extra)
    return {"document": topic, "meta": meta, "distance": distance}


def load_router():
    with open(os.path.join(HERE, "data", "knowledge_base.json"), encoding="utf-8") as f:
        kb = json.load(f)
    return Router(kb.get("smalltalk", {}))


# ------------------------------------------------------------------ smalltalk
def test_smalltalk():
    print("\nsmalltalk matching")
    r = load_router()

    for text, intent in [("hello", "greeting"), ("Hi!", "greeting"),
                         ("thank you", "thanks"), ("Bye.", "farewell"),
                         ("can you hear me?", "audio_check"),
                         ("who are you", "identity"), ("okay", "acknowledge"),
                         ("um, hello", "greeting")]:
        m = r.match_smalltalk(text)
        check("%-22r -> %s" % (text, intent), m is not None and m[0] == intent,
              "got %s" % (m[0] if m else None))

    # The critical negative case: a greeting glued to a real question must NOT
    # be swallowed as smalltalk.
    for text in ["hi what is the price of a plot",
                 "hello can you tell me about the water supply",
                 "thanks and how many parks are there"]:
        check("not smalltalk: %r" % text[:34], r.match_smalltalk(text) is None)

    check("normalize strips punctuation", normalize("Hello!!  There?") == "hello there")


# --------------------------------------------------------------------- routing
def test_routing():
    print("\nrouting decisions")
    r = load_router()
    S = settings.route_strong_threshold
    W = settings.route_weak_threshold
    E = settings.route_escalation_threshold

    # Must READ as a question, or the new CHAT fallback correctly claims it.
    def decide(plan, text="what is the gst rate on this property?"):
        _stub.PLAN = plan
        return asyncio.run(r.decide(text))

    # close knowledge hit -> KNOWLEDGE, confident
    d = decide([hit("knowledge", "roads", S - 0.1)])
    check("close KB hit -> KNOWLEDGE", d.route == Route.KNOWLEDGE)
    check("close KB hit is confident", d.confident is True)

    # middling knowledge hit -> KNOWLEDGE but hedged
    d = decide([hit("knowledge", "roads", (S + W) / 2)])
    check("weak KB hit -> KNOWLEDGE", d.route == Route.KNOWLEDGE)
    check("weak KB hit is NOT confident", d.confident is False)

    # far knowledge hit -> ESCALATE (the "too deep for the data" case)
    d = decide([hit("knowledge", "roads", W + 0.2)])
    check("far KB hit -> ESCALATE", d.route == Route.ESCALATE)
    check("far KB hit captures the question", "unanswered" in d.capture)
    check("far KB hit has a spoken reply", bool(d.reply))

    # escalation topic closer than knowledge -> ESCALATE
    d = decide([hit("escalation", "esc_pricing", E - 0.1, answer="callback"),
                hit("knowledge", "roads", E + 0.05)])
    check("closer escalation wins", d.route == Route.ESCALATE)
    check("escalation reply is used", d.reply == "callback")
    check("escalation intent recorded", d.intent == "esc_pricing")

    # TIE-BREAK: escalation exactly ties knowledge -> escalation must still win.
    # "what is the price" sits near several knowledge entries lexically.
    d = decide([hit("escalation", "esc_pricing", 0.40, answer="callback"),
                hit("knowledge", "investment_case", 0.40)])
    check("tie goes to escalation", d.route == Route.ESCALATE)

    # but a clearly closer knowledge hit beats a marginal escalation
    d = decide([hit("knowledge", "water_supply", 0.20),
                hit("escalation", "esc_pricing", 0.54)])
    check("much closer KB beats marginal escalation", d.route == Route.KNOWLEDGE)

    # explicit human request -> HANDOFF, not plain ESCALATE
    d = decide([hit("escalation", "esc_human", 0.2, answer="connecting you")])
    check("esc_human -> HANDOFF", d.route == Route.HANDOFF)

    # empty index must not crash and must not fabricate
    d = decide([])
    check("empty index -> ESCALATE", d.route == Route.ESCALATE)
    check("empty index has a reply", bool(d.reply))

    # smalltalk beats everything, and must not consult the (stubbed) index
    _stub.PLAN = [hit("knowledge", "roads", 0.01)]
    d = asyncio.run(r.decide("hello"))
    check("smalltalk short-circuits the index", d.route == Route.SMALLTALK)
    check("smalltalk costs no search", d.distance == 0.0)

    # proposed status must reach the prompt context
    d = decide([hit("knowledge", "shopping_mall", 0.2, status="proposed",
                    topic="Shopping mall", answer="A mall is proposed")])
    check("KNOWLEDGE builds context", "proposed" in d.context.lower())


# ------------------------------------------------------------------------ LLM
class _FakeDelta:
    def __init__(self, c): self.content = c


class _FakeChoice:
    def __init__(self, c): self.delta = _FakeDelta(c)


class _FakeEvent:
    def __init__(self, c): self.choices = [_FakeChoice(c)]


class _FakeCompletions:
    def __init__(self, chunks): self._chunks = chunks

    def create(self, **kw):
        return (_FakeEvent(c) for c in self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(chunks))


def test_live_call_regression():
    """The exact turns from the first live call. Every one of these must now
    route somewhere the caller would find sane."""
    print("\nlive-call regression (real turns from call CA6f150a)")
    r = load_router()

    def decide(text, plan, ctx):
        _stub.PLAN = plan
        return asyncio.run(r.decide(text, ctx=ctx))

    # comfortably beyond ROUTE_WEAK_THRESHOLD so nothing matches
    far = [hit("knowledge", "why_neelambur", 0.95),
           hit("escalation", "esc_pricing", 0.92)]
    asked_name = {"bot_asked_question": True, "asked_for_name": True}
    mid = {"bot_asked_question": True}

    # self-introductions - these all escalated on the live call
    for text, want in [("Hi. This is Roy.", "Roy"), ("This is Roy.", "Roy"),
                       ("My name is Dennis", "Dennis"), ("I'm Karthi", "Karthi"),
                       ("Roy here", "Roy")]:
        d = decide(text, far, {})
        check("%-24r -> CHAT" % text, d.route == Route.CHAT, "got %s" % d.route.name)
        check("     captures name %r" % want, d.name == want, "got %r" % d.name)

    # bare name, only valid because we just asked for one
    d = decide("Denise.", far, asked_name)
    check("bare 'Denise.' after asking -> CHAT", d.route == Route.CHAT)
    check("     captures 'Denise'", d.name == "Denise", "got %r" % d.name)
    d = decide("Denise. Denise. Denise.", far, asked_name)
    check("repeated name de-duplicated", d.name == "Denise", "got %r" % d.name)

    # ...but a bare word when we did NOT ask for a name is not a name
    d = decide("Denise.", far, {})
    check("bare name without the ask -> not captured", d.name == "")

    # things that look like intros but are not names
    for text in ["This is a good project", "I am interested", "this is the price"]:
        d = decide(text, far, {})
        check("not a name: %-26r" % text, d.name == "", "got %r" % d.name)

    # conversational filler must not escalate
    # "got it" is in the smalltalk table; the others fall through to CHAT.
    # Either is fine - what matters is that NONE of them escalate.
    for text in ["Got it.", "Okay fine.", "Hmm alright then.", "Yeah sure."]:
        d = decide(text, far, mid)
        check("conversational %-20r -> not escalated" % text,
              d.route in (Route.CHAT, Route.SMALLTALK), "got %s" % d.route.name)

    # a REAL unanswerable question still escalates
    d = decide("What is the gst rate on this property?", far, {})
    check("real unanswerable question -> ESCALATE", d.route == Route.ESCALATE)

    # ...and does not repeat the full callback line twice
    d2 = decide("And what about stamp duty?", far,
                {"escalated_last_turn": True})
    check("second escalation is shortened",
          d2.reply != d.reply and bool(d2.reply), repr(d2.reply))


def test_llm():
    print("\nLLM streaming and metadata")

    brain = llm.GroqBrain.__new__(llm.GroqBrain)
    reply = ('We have eighty and sixty feet roads. They stretch over four '
             'kilometres. There are fifteen entrances.\n'
             'EXTRA: {"name": "Karthi", "requirement": "corner plot", '
             '"intent_score": 80, "whatsapp_wanted": true, "handoff": false}')
    # feed it in small chunks, like a real token stream
    chunks = [reply[i:i + 7] for i in range(0, len(reply), 7)]
    brain.client = _FakeClient(chunks)

    out = list(brain.stream_sentences([], "how wide are the roads", "ctx"))
    raw = next((o[1] for o in out if isinstance(o, tuple)), "")
    sentences = [o for o in out if isinstance(o, str)]

    check("streams multiple sentences", len(sentences) >= 2,
          "got %d: %r" % (len(sentences), sentences))
    check("first sentence is complete",
          sentences and sentences[0].endswith("."), repr(sentences[:1]))
    check("EXTRA never reaches the speaker",
          not any("EXTRA" in s for s in sentences), repr(sentences))
    check("respects MAX_REPLY_SENTENCES",
          len(sentences) <= settings.max_reply_sentences)

    extra = llm.GroqBrain.parse_extra(raw)
    check("parses name", extra.get("name") == "Karthi", repr(extra))
    check("parses requirement", extra.get("requirement") == "corner plot")
    check("parses intent_score", extra.get("intent_score") == 80)
    check("parses whatsapp_wanted", extra.get("whatsapp_wanted") is True)
    check("parses handoff", extra.get("handoff") is False)

    # robustness: malformed / missing EXTRA must never raise
    check("no EXTRA -> {}", llm.GroqBrain.parse_extra("just text") == {})
    check("broken EXTRA -> {}", llm.GroqBrain.parse_extra("EXTRA: {oops") == {})
    check("null name dropped",
          "name" not in llm.GroqBrain.parse_extra('EXTRA: {"name": null}'))
    check("string 'null' dropped",
          "name" not in llm.GroqBrain.parse_extra('EXTRA: {"name": "null"}'))
    check("score clamped",
          llm.GroqBrain.parse_extra('EXTRA: {"intent_score": 999}')["intent_score"] == 100)

    check("clean_for_speech strips markdown",
          "*" not in llm.clean_for_speech("**bold** text"))
    check("clean_for_speech strips EXTRA",
          llm.clean_for_speech('Hi there.\nEXTRA: {"a":1}') == "Hi there.")
    check("does not split on decimals",
          len(llm.split_sentences("The park is 9.5 acres wide")) == 1)
    check("splits on sentence end",
          len(llm.split_sentences("One thing. Another thing.")) == 2)

    # a stream with no EXTRA at all must still emit the text
    brain.client = _FakeClient(["Hello there. ", "All good."])
    out = [o for o in brain.stream_sentences([], "hi", "") if isinstance(o, str)]
    check("stream without EXTRA still speaks", len(out) == 2, repr(out))


# ------------------------------------------------------------------------ TTS
def test_tts_framing():
    print("\nTTS framing")
    import audioop
    FRAME = 160
    # 1 second of 8 kHz 16-bit silence -> mu-law -> frames
    pcm = b"\x00\x00" * 8000
    mulaw = audioop.lin2ulaw(pcm, 2)
    check("1s pcm -> 8000 mu-law bytes", len(mulaw) == 8000)
    frames = [mulaw[i:i + FRAME] for i in range(0, len(mulaw), FRAME)]
    check("1s -> 50 frames of 20ms", len(frames) == 50)
    check("every frame is 160 bytes", all(len(f) == FRAME for f in frames))

    # partial tail padding
    tail = mulaw[:8000 - 37]
    last = tail[len(tail) - (len(tail) % FRAME):]
    padded = last + b"\xff" * (FRAME - len(last))
    check("short tail pads to 160", len(padded) == FRAME)
    check("pads with mu-law silence 0xFF", padded[-1:] == b"\xff")


def main():
    print("=" * 62)
    print("Karthipuram Jarvis - logic tests (no model, no network)")
    print("=" * 62)
    test_smalltalk()
    test_routing()
    test_live_call_regression()
    test_llm()
    test_tts_framing()
    print("\n" + "=" * 62)
    if FAILS:
        print("%d FAILED: %s" % (len(FAILS), ", ".join(FAILS)))
        return 1
    print("All logic tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
