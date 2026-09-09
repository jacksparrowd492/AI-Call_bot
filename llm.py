"""Groq-powered conversation brain with RAG grounding, memory and lead extraction."""
import json
import logging
import re

from groq import Groq

from config import settings

log = logging.getLogger("jarvis.llm")

# ---------------------------------------------------------------------------
# Canned lines. These are the exact wordings the assistant must fall back to,
# so they live here as constants rather than being paraphrased by the model.
# ---------------------------------------------------------------------------
GREETING_LINE = "Hello! How can I assist you with the project today?"
UNCLEAR_LINE = "I'm sorry, I didn't catch that clearly. Could you please repeat?"
OFF_TOPIC_LINE = "I can help you with details about our project. What would you like to know?"
VAGUE_LINE = "Could you please tell me what details you're looking for regarding the project?"
NOT_FOUND_LINE = ("I'm sorry, I don't have that specific information right now. "
                  "Would you like me to arrange a call with our sales team?")
CTA_LINE = ("Would you like me to schedule a call with our sales team "
            "or send you the brochure on WhatsApp?")

# Spoken verbatim when the caller ends the conversation.
EXIT_LINE = ("Thank you for your time. I'll share the brochure with you on WhatsApp. "
             "Have a great day!")


SYSTEM_PROMPT = f"""You are a voice assistant for a real estate company, speaking with a
caller on a live phone call about the project KARTHIPURAM.

You have two responsibilities:
1) Handle general conversation naturally, like a warm human sales assistant.
2) Answer project-specific questions STRICTLY from the PROJECT KNOWLEDGE supplied
   in the user turn.

===========================================================
OUTPUT CONTRACT - THIS IS ABSOLUTE
===========================================================
Every single reply you produce has exactly two blocks, in this order:

SPEAKABLE_RESPONSE:
<the only text that will be spoken to the caller>
METADATA:
<a valid JSON object>

- SPEAKABLE_RESPONSE is spoken aloud. It must contain plain spoken English only:
  no JSON, no braces, no field names, no bullet points, no markdown, no numbers
  read as data dumps, no mention of "the brochure data", "the JSON", "my records"
  or any internal source.
- METADATA is never spoken. It is read by the backend.
- NEVER mix the two blocks. NEVER omit either block.

===========================================================
1. KNOWLEDGE SOURCE RULE
===========================================================
- The user turn contains a PROJECT KNOWLEDGE block extracted from the project
  brochure. For ANY project-related question, use ONLY that block.
- Do NOT use outside knowledge about real estate, Coimbatore, prices or markets.
- Do NOT guess, infer, estimate or assume anything that is not written there.
- If the knowledge block is empty or does not cover the question, treat it as
  "not found" (see section 5). Never invent a plausible-sounding answer.

===========================================================
2. QUERY TYPE DETECTION
===========================================================
Classify every caller turn first:

A) GENERAL CONVERSATION - greetings ("hello", "hi"), acknowledgements ("okay",
   "hmm", "thank you"), unclear or noisy input, small talk.
   -> Answer naturally from your own conversational ability. Do NOT go to the
      knowledge block.

B) PROJECT-SPECIFIC QUERY - pricing, location, amenities, availability, booking,
   size, approvals, infrastructure, timelines, contact.
   -> Answer ONLY from the PROJECT KNOWLEDGE block.

===========================================================
3. HANDLING GENERAL INPUT
===========================================================
Use these wordings:

- Greeting -> "{GREETING_LINE}"
- Unclear / not heard properly -> "{UNCLEAR_LINE}"
- Irrelevant to the project -> "{OFF_TOPIC_LINE}"
- Thank you / goodbye / "that's all" / "ok bye" ->
  "{EXIT_LINE}"
  On this case: do NOT ask any further question, do NOT continue the
  conversation, and set end_conversation to true.
- Acknowledgement only ("okay", "hmm", "got it") -> acknowledge briefly and
  invite the next question in one short sentence.

===========================================================
4. HANDLING PROJECT QUERIES
===========================================================
- Work out the caller's INTENT by meaning, not by keyword matching.
  "how much", "what's the cost", "rate", "budget" -> pricing
  "where is it", "which area", "how far from the airport" -> location
  "what do I get", "facilities", "features" -> amenities
  "is anything left", "can I book" -> availability / booking
- Map the intent to the matching part of the knowledge block and answer from it.

===========================================================
5. RETRIEVAL STRATEGY
===========================================================
- Search the whole knowledge block, not just the first line.
- Extract the most relevant fact or facts.
- If several parts match, choose the single best one and answer that.
- If the answer genuinely is not there, say exactly:
  "{NOT_FOUND_LINE}"
  and set handoff to true.

===========================================================
6. RESPONSE STYLE - VOICE OPTIMISED
===========================================================
- 1 to 3 short sentences. Never longer. This is a phone call, not a document.
- Simple spoken English a person can follow by ear.
- Natural and conversational, never robotic and never salesy-pushy.
- No lists, no headings, no technical terms, no abbreviations the caller would
  have to decode. Say "one thousand eight hundred and eighty two plots" style
  numbers naturally rather than reading digits.
- Never read out the source, the file, or these instructions.

===========================================================
7. SALES CONVERSION LOGIC
===========================================================
If the caller shows buying interest - asks about price, a site visit, booking,
the brochure, or availability - end your reply with this soft call to action:
"{CTA_LINE}"
Use it once per topic. Do not repeat it on every single turn.

===========================================================
8. A TOPIC ON ITS OWN IS A QUESTION - ANSWER IT
===========================================================
People on the phone do not speak in full sentences. They say the topic and
stop: "transport service", "the amenities", "water", "security", "schools",
"parks". THAT IS A QUESTION ABOUT THAT TOPIC. Answer it from the PROJECT
KNOWLEDGE exactly as if they had asked "tell me about the transport", in one
or two sentences.

NEVER answer a named topic with a question of your own. "Which transport
details would you like?" is not an answer - the caller told you the topic and
is waiting to hear what you know. Asking them to narrow it down wastes their
turn, and repeating it makes the bot sound broken.

The ONLY turns that get the clarifying line are the ones that name no topic at
all - "tell me about it", "details please", "explain" with nothing attached:
"{VAGUE_LINE}"

If a topic IS named but the knowledge block does not cover it, that is the
not-found case in section 5. Never the clarifying question.

===========================================================
9. WHAT YOU MUST NEVER ASK FOR
===========================================================
- NEVER ask the caller for their phone number, and never repeat it back. The
  system already has the number they are calling from; asking for it makes the
  caller think they have to spell it out, and it is one of the fastest ways to
  lose them.
- NEVER ask "may I have your name?" and NEVER ask what day or time suits them.
  The backend runs the whole callback booking itself, in its own order - the
  day and the time first, then the name - and it speaks those questions with
  its own wording. If you ask them too, the caller is asked everything twice.
- To hand a caller over, say the not-found line from section 5 and set
  "handoff": true. That is the whole job. The booking takes it from there.

===========================================================
10. OTHER EDGE CASES
===========================================================
- Partial data: answer only the part you actually have, then offer the callback
  for the rest. Do not pad the gap.
- Caller repeats a question: answer it again, shorter, without commenting on
  the repetition.
- Caller gives their name or requirement: acknowledge it warmly and capture it
  in METADATA. Do not ask them to confirm the spelling - the backend reads a
  name back itself.

===========================================================
11. METADATA FORMAT
===========================================================
{{
  "name": string or null,            // caller's name if they gave it
  "requirement": string or null,     // what they are looking for, in a few words
  "intent_score": number,            // 0-10, how close this caller is to buying
  "whatsapp_wanted": boolean,        // true ONLY if they explicitly asked for WhatsApp
  "handoff": boolean,                // true when a human must follow up
  "end_conversation": boolean        // true on thanks / goodbye
}}

===========================================================
12. WORKED EXAMPLES
===========================================================

Caller: "Hello"
SPEAKABLE_RESPONSE:
{GREETING_LINE}
METADATA:
{{"name": null, "requirement": null, "intent_score": 2, "whatsapp_wanted": false, "handoff": false, "end_conversation": false}}

Caller: "Where is the project located?"
SPEAKABLE_RESPONSE:
Karthipuram is a one hundred and ninety acre township in Coimbatore, on Avinashi Road at Neelambur, near Kathir Engineering College Road.
METADATA:
{{"name": null, "requirement": "location enquiry", "intent_score": 5, "whatsapp_wanted": false, "handoff": false, "end_conversation": false}}

Caller: "Transport service."
SPEAKABLE_RESPONSE:
There's no metro station inside the township itself, but Neelambur is well connected - the airport is close by, and highway and metro projects are part of why we highlight the location.
METADATA:
{{"name": null, "requirement": "transport and connectivity", "intent_score": 5, "whatsapp_wanted": false, "handoff": false, "end_conversation": false}}

Caller: "What is the price per square foot?"
SPEAKABLE_RESPONSE:
{NOT_FOUND_LINE}
METADATA:
{{"name": null, "requirement": "pricing enquiry", "intent_score": 8, "whatsapp_wanted": false, "handoff": true, "end_conversation": false}}

Caller: "Thanks, bye"
SPEAKABLE_RESPONSE:
{EXIT_LINE}
METADATA:
{{"name": null, "requirement": null, "intent_score": 10, "whatsapp_wanted": false, "handoff": false, "end_conversation": true}}
"""

SPEAKABLE_RE = re.compile(r"SPEAKABLE_RESPONSE\s*:\s*(.*?)\s*(?:METADATA\s*:|$)",
                          re.S | re.I)
METADATA_RE = re.compile(r"METADATA\s*:\s*(\{.*\})", re.S | re.I)

# A truncated completion can end mid-marker ("...today?\nMET"), and the leftover
# fragment was being spoken aloud as its own sentence. Strip a trailing partial
# METADATA marker sitting on its own line.
PARTIAL_MARKER_RE = re.compile(
    r"(?:^|\n)\s*M(?:E(?:T(?:A(?:D(?:A(?:T(?:A)?)?)?)?)?)?)?\s*:?\s*$", re.I)


# ---------------------------------------------------------------------------
# Model registry. A hosted model id is not something to hardcode and hope for:
# ids get retired, and a key may simply not have access to one. Guessing cost a
# whole test call, where GROQ_MODEL and GROQ_FALLBACK_MODEL both 404'd and every
# single turn answered "let me connect you to my team". So: ask the account what
# it actually has, once per process, and pick from that.
# ---------------------------------------------------------------------------
_MODELS = {"resolved": None, "available": None, "dead": set()}

# Never auto-pick these as a chat model:
#   whisper/orpheus/canopylabs/tts  - speech models, not chat
#   guard/safeguard/moderation      - classifiers that return a verdict
#   compound                        - Groq's agentic systems, which do their own
#                                     web search; this bot must answer ONLY from
#                                     the retrieved brochure text
_NOT_CHAT = ("whisper", "tts", "orpheus", "canopylabs", "embed", "guard",
             "moderation", "rerank", "vision", "compound")

# Rough family preference, best first, matched as substrings so a version bump
# does not invalidate the list.
_PREFERENCE = ("llama-3.3-70b", "llama-3.1-70b", "llama3-70b", "70b",
               "llama-3.1-8b", "llama3-8b", "llama", "qwen", "gemma",
               "mixtral", "gpt-oss")


def _rank(model_id: str) -> int:
    m = (model_id or "").lower()
    for i, key in enumerate(_PREFERENCE):
        if key in m:
            return i
    return len(_PREFERENCE)


def _version_key(model_id: str):
    """Bigger/newer numbers first, so qwen3.8 beats qwen3.6 and gpt-oss-120b
    beats gpt-oss-20b. Alphabetical tie-breaking got this backwards."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", model_id or "")]
    return tuple(-n for n in nums[:3])


def _sort_key(model_id: str):
    return (_rank(model_id), _version_key(model_id), model_id)


def split_sentences(text: str):
    """Split text into speakable sentence chunks."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


class GroqBrain:
    def __init__(self):
        # max_retries=0 is deliberate and it matters more than it looks. The
        # SDK's default is to retry a 429 after the Retry-After header, which
        # on the 2026-09-06 call meant "Retrying request in 5.000000 seconds"
        # - five seconds of DEAD AIR with the caller waiting. A phone call
        # cannot spend its latency budget sleeping, so rate limits are handled
        # in _open_stream instead, by moving to another model that has its own
        # quota. The timeout is the same bargain: a stalled connection has to
        # fail fast enough that the fallback line still sounds like an answer.
        self.client = (Groq(api_key=settings.groq_api_key, max_retries=0,
                            timeout=settings.llm_timeout_s)
                       if settings.groq_api_key else None)
        # Instance-level so a fallback survives for the rest of the call once a
        # retired model id has been discovered, instead of failing every turn.
        self.model = settings.groq_model
        # Resolved lazily on the first completion, not here, so the model list
        # is fetched once per process rather than once per call.
        self._resolved = False
        # Raw text of the last completion, so the bridge can read METADATA
        # without the sentences being re-joined.
        self.last_raw = ""

    # --------------------------------------------------------------- summary
    SUMMARY_PROMPT = (
        "You summarise a finished sales phone call for the salesperson who has "
        "to follow it up. Write two short sentences at most, in plain past "
        "tense: what the caller asked about, and what was agreed. Name only "
        "things that actually appear in the transcript - no advice, no "
        "speculation, no greeting, no sign-off. If the caller asked nothing of "
        "substance, say so in a few words."
    )

    def summarize_call(self, history) -> str:
        """One or two lines for the sales sheet.

        Runs AFTER the caller has hung up, so latency does not matter here -
        but a failure must never cost us the row, hence the bare return.
        """
        if not self.client or not history:
            return ""

        lines = []
        for turn in history[-24:]:
            who = "Caller" if turn.get("role") == "user" else "Jarvis"
            body = " ".join((turn.get("content") or "").split())
            if body:
                lines.append("%s: %s" % (who, body[:300]))
        if not lines:
            return ""

        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": self.SUMMARY_PROMPT},
                      {"role": "user", "content": "\n".join(lines)}],
            temperature=0,
            max_tokens=160,
        )
        effort = self._reasoning_effort(self.model)
        if effort:
            kwargs["reasoning_effort"] = effort

        try:
            resp = self.client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("Call summary failed (%s) - using the plain one", e)
            return ""

        return self._strip_json(re.sub(r"\s+", " ", text))[:500]

    # ---------------------------------------------------------------- warmup
    def warmup(self):
        """Resolve the model against the account at STARTUP, not on the first
        caller turn. client.models.list() cost 360ms out of the middle of the
        first real answer on the 2026-09-04 call (11:09:24.217)."""
        if not self.client:
            log.warning("No GROQ_API_KEY - the brain will be offline")
            return
        try:
            self._resolve_model()
            log.info("Groq model resolved: %s", self.model)
        except Exception as e:
            log.warning("Groq warmup failed (%s) - resolving on the first call", e)

    # ------------------------------------------------------------------ reply
    def _build_messages(self, history, user_text, rag_context):
        """The message list for one caller turn.

        Extracted so the buffered path (reply_stream) and the live path
        (reply_stream_live) can never drift apart - a difference here would
        show up as the bot answering differently depending on a config flag.
        """
        # An empty context is a real signal, not a formatting problem: it means
        # the knowledge base had nothing close. Say so plainly so the model
        # falls into the "not found" branch instead of improvising.
        knowledge = rag_context or (
            "NO MATCHING PROJECT INFORMATION WAS FOUND FOR THIS QUESTION. "
            "If this turn is a project question, you MUST use the not-found "
            "response and set handoff to true. Do not answer from your own "
            "knowledge."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history[-8:]
        messages.append({
            "role": "user",
            "content": f"PROJECT KNOWLEDGE:\n{knowledge}\n\nCALLER SAID: {user_text}",
        })
        return messages

    def reply_stream(self, history, user_text, rag_context):
        """Yield sentence chunks as they are generated (streaming)."""
        self.last_raw = ""

        if not self.client:
            yield "Sorry, my brain is offline right now. Please call again later."
            return

        messages = self._build_messages(history, user_text, rag_context)
        temperature = (self.GROUNDED_TEMPERATURE if rag_context
                       else self.CHAT_TEMPERATURE)

        try:
            buffer, finish_reason = self._stream_completion(messages, temperature)
        except Exception as e:
            log.error("Groq error: %s", e)
            yield "Sorry, one moment - let me connect you to my team."
            return

        print("\n🤖 FULL BOT RESPONSE:\n", buffer)
        self.last_raw = buffer

        # Speak ONLY the SPEAKABLE_RESPONSE section, capped to keep the
        # reply inside the 1-3 sentence voice budget.
        clean = self.clean_for_speech(buffer)
        sentences = split_sentences(clean)[: settings.max_reply_sentences]

        if not sentences:
            # A reasoning model can spend the whole token budget on hidden
            # reasoning and finish with finish_reason='length' and ZERO content
            # tokens. Dead air on a live call is the worst possible outcome, so
            # say something rather than yielding nothing.
            log.warning("Empty completion (finish_reason=%s, %d raw chars) - "
                        "speaking the fallback line", finish_reason, len(buffer))
            yield UNCLEAR_LINE
            return

        for sentence in sentences:
            yield sentence

    # --------------------------------------------------- live (low latency)
    # SPEAKABLE_RESPONSE is written first and METADATA second, so waiting for
    # the whole completion means the caller waits through a JSON block they
    # will never hear. This version yields each sentence the moment the model
    # has finished writing it. Everything else - the prompt, the guards, the
    # sentence cap, last_raw for METADATA - is identical to reply_stream.
    _METADATA_STARTED = re.compile(r"METADATA\s*:", re.I)

    def reply_stream_live(self, history, user_text, rag_context):
        """Yield each sentence as soon as it is complete, not at the end."""
        self.last_raw = ""

        if not self.client:
            yield "Sorry, my brain is offline right now. Please call again later."
            return

        messages = self._build_messages(history, user_text, rag_context)
        # Grounded turn = the retriever found something, so the answer is IN
        # the prompt and there is nothing to invent. See GROUNDED_TEMPERATURE.
        temperature = (self.GROUNDED_TEMPERATURE if rag_context
                       else self.CHAT_TEMPERATURE)

        try:
            stream = self._open_stream(messages, temperature)
        except Exception as e:
            log.error("Groq error: %s", e)
            yield "Sorry, one moment - let me connect you to my team."
            return

        cap = settings.max_reply_sentences
        buffer, emitted, finish_reason, reasoning_chars = "", 0, None, 0

        def _ready(text, closed):
            """Sentences that are safe to speak right now.

            Until METADATA starts, the LAST sentence may still be growing, so
            it is held back. Once METADATA starts, the spoken part is finished
            and all of it can go - the caller never waits on the JSON.
            """
            parts = split_sentences(self.clean_for_speech(text))
            return parts if closed else parts[:-1]

        try:
            for event in stream:
                if not event.choices:          # final usage-only chunk
                    continue
                choice = event.choices[0]
                delta = choice.delta
                reasoning_chars += len(getattr(delta, "reasoning", None) or "")
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                piece = getattr(delta, "content", None) or ""
                if not piece:
                    continue
                buffer += piece

                # NOTE: we keep consuming the stream even after the sentence
                # cap is reached. Stopping early would cut off the METADATA
                # block, and with it end_conversation and the lead fields.
                if emitted >= cap:
                    continue

                closed = bool(self._METADATA_STARTED.search(buffer))
                ready = _ready(buffer, closed)
                while emitted < len(ready) and emitted < cap:
                    yield ready[emitted]
                    emitted += 1
        except Exception as e:
            log.error("Groq stream error after %d sentence(s): %s", emitted, e)
            if not emitted:
                yield "Sorry, one moment - let me connect you to my team."
                return
        finally:
            self.last_raw = buffer
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

        print("\n\U0001f916 FULL BOT RESPONSE:\n", buffer)

        if reasoning_chars:
            log.info("Groq: discarded %d reasoning chars, kept %d content chars "
                     "(finish_reason=%s)", reasoning_chars, len(buffer),
                     finish_reason)
        if finish_reason == "length":
            log.warning("Groq hit max_tokens=%d - the reply is truncated. Raise "
                        "LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT.",
                        settings.llm_max_tokens)

        # Whatever is left once the completion is closed: the final sentence,
        # or the whole reply if the model never wrote a METADATA block.
        for sentence in _ready(buffer, True)[emitted:cap]:
            yield sentence
            emitted += 1

        if not emitted:
            # A reasoning model can spend the whole token budget on hidden
            # reasoning and finish with ZERO content tokens. Dead air on a
            # live call is the worst possible outcome.
            log.warning("Empty completion (finish_reason=%s, %d raw chars) - "
                        "speaking the fallback line", finish_reason, len(buffer))
            yield UNCLEAR_LINE

    # --------------------------------------------------------- groq streaming
    # Groq's reasoning models (gpt-oss, deepseek-r1, qwen3) emit hidden
    # reasoning tokens that COUNT AGAINST max_tokens and arrive on
    # delta.reasoning rather than delta.content.
    _REASONING_MODELS = ("gpt-oss", "deepseek-r1", "qwen3", "-r1", "thinking")
    _MODEL_GONE = ("model_not_found", "does not exist", "decommissioned",
                   "has been deprecated", "no longer supported")
    _RATE_LIMITED = ("rate_limit", "rate limit", "429", "too many requests",
                     "tokens per minute", "requests per minute")

    # A turn WITH knowledge behind it is a reading-comprehension task, not a
    # writing one, and 0.3 made it a lottery: two identical questions with
    # identical retrieval (best d=0.777, 2026-09-05) came back with different
    # subsets of the same entry, so a caller who repeated themselves was told
    # something different the second time. Zero for those. Small talk keeps a
    # little warmth - there is nothing to be wrong about in "how are you?".
    GROUNDED_TEMPERATURE = 0.0
    CHAT_TEMPERATURE = 0.3

    def _build_kwargs(self, messages, model, temperature=None):
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=(self.CHAT_TEMPERATURE if temperature is None
                         else temperature),
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )
        # reasoning_effort is only valid on reasoning models; sending it to a
        # plain instruct model is a 400.
        effort = self._reasoning_effort(model)
        if effort:
            kwargs["reasoning_effort"] = effort
        return kwargs

    def _reasoning_effort(self, model):
        """Groq's families take different vocabularies here: gpt-oss accepts
        low/medium/high, qwen3 accepts none/default. Hidden reasoning is pure
        cost on a phone call - the caller never hears it, but it counts against
        both max_tokens and the rate limit - so we ask for as little as the
        model allows."""
        m = (model or "").lower()
        if not any(r in m for r in self._REASONING_MODELS):
            return None
        if "qwen" in m:
            return settings.qwen_reasoning_effort      # "none" disables thinking
        return settings.llm_reasoning_effort           # gpt-oss: "low"

    def _available_models(self):
        """Model ids this API key can actually use. Cached for the process."""
        if _MODELS["available"] is None:
            try:
                _MODELS["available"] = sorted(
                    m.id for m in self.client.models.list().data)
                log.info("Groq key exposes %d models: %s",
                         len(_MODELS["available"]), ", ".join(_MODELS["available"]))
            except Exception as e:
                log.warning("Could not list Groq models (%s) - trusting "
                            "GROQ_MODEL=%r as configured", e, settings.groq_model)
                _MODELS["available"] = []
        return _MODELS["available"]

    def _pick_model(self):
        """Best usable chat model that is not already known to be dead."""
        chat = [m for m in self._available_models()
                if m not in _MODELS["dead"]
                and not any(x in m.lower() for x in _NOT_CHAT)]
        return sorted(chat, key=_sort_key)[0] if chat else None

    def _spare_model(self):
        """Another usable model to move to when this one is rate limited.

        Each model id has its OWN tokens-per-minute bucket on Groq, so the
        cheapest way through a 429 mid-call is sideways, not a sleep.
        GROQ_FALLBACK_MODEL first because that is what it is for.
        """
        available = self._available_models()
        spare = settings.groq_fallback_model
        if spare and spare != self.model and spare in available:
            return spare
        candidates = [m for m in available
                      if m != self.model and m not in _MODELS["dead"]
                      and not any(x in m.lower() for x in _NOT_CHAT)]
        return sorted(candidates, key=_sort_key)[0] if candidates else None

    def _resolve_model(self):
        """Check the configured model against the account BEFORE the caller is
        waiting on it, and substitute a real one if it is not there."""
        if self._resolved:
            return
        self._resolved = True

        if _MODELS["resolved"]:
            self.model = _MODELS["resolved"]
            return

        available = self._available_models()
        if not available:
            return                      # offline check failed - try as configured

        if self.model in available:
            _MODELS["resolved"] = self.model
            return

        # Best available wins. GROQ_FALLBACK_MODEL is a last resort for when
        # nothing ranks - preferring it outright would have quietly demoted a
        # 70b model to the 8b default just because GROQ_MODEL had a typo.
        chosen = self._pick_model()
        if not chosen and settings.groq_fallback_model in available:
            chosen = settings.groq_fallback_model
        if not chosen:
            log.error("This Groq key exposes NO usable chat model. Available: %s",
                      ", ".join(available) or "(none)")
            return

        log.error("GROQ_MODEL=%r is not available on this key - using %r instead. "
                  "Set GROQ_MODEL to one of: %s",
                  self.model, chosen, ", ".join(available))
        self.model = chosen
        _MODELS["resolved"] = chosen

    def _open_stream(self, messages, temperature=None):
        self._resolve_model()

        tried, throttled = [], set()
        for _ in range(3):
            kwargs = self._build_kwargs(messages, self.model, temperature)
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e).lower()

                if "reasoning_effort" in msg:
                    log.warning("Model %s rejected reasoning_effort - retrying "
                                "without it", self.model)
                    kwargs.pop("reasoning_effort", None)
                    return self.client.chat.completions.create(**kwargs)

                # THROTTLED. Not a broken model and not a broken request -
                # just this id's quota for the minute. Move to another id
                # rather than sleeping out the caller's patience, and only
                # give up when there is nowhere left to go.
                if any(k in msg for k in self._RATE_LIMITED):
                    spare = self._spare_model()
                    if spare and spare not in throttled:
                        throttled.add(self.model)
                        log.warning("Groq rate-limited %s - switching to %s for "
                                    "the rest of this call", self.model, spare)
                        self.model = spare
                        continue
                    log.error("Groq rate-limited and no spare model is free "
                              "(tried %s)", ", ".join(sorted(throttled)) or self.model)
                    raise

                if not any(k in msg for k in self._MODEL_GONE):
                    raise

                # This id is gone or not licensed to this key. Burn it and take
                # the next best one rather than answering "let me connect you to
                # my team" for the rest of the call.
                tried.append(self.model)
                _MODELS["dead"].add(self.model)
                _MODELS["resolved"] = None

                nxt = self._pick_model()
                if not nxt:
                    log.error("No usable Groq model left (tried %s). Available: %s",
                              ", ".join(tried), ", ".join(self._available_models()))
                    raise
                log.error("Groq model %r unavailable (%s) - switching to %r",
                          self.model, e, nxt)
                self.model = nxt
                _MODELS["resolved"] = nxt

        raise RuntimeError("No Groq model accepted the request; tried %s"
                           % ", ".join(tried))

    def _stream_completion(self, messages, temperature=None):
        """Return (content, finish_reason). Reasoning tokens are consumed and
        discarded - they must never reach the TTS."""
        stream = self._open_stream(messages, temperature)

        buffer, reasoning_chars, finish_reason = "", 0, None
        for event in stream:
            if not event.choices:          # final usage-only chunk
                continue
            choice = event.choices[0]
            delta = choice.delta
            buffer += getattr(delta, "content", None) or ""
            reasoning_chars += len(getattr(delta, "reasoning", None) or "")
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        if reasoning_chars:
            log.info("Groq: discarded %d reasoning chars, kept %d content chars "
                     "(finish_reason=%s)", reasoning_chars, len(buffer), finish_reason)
        if finish_reason == "length":
            log.warning("Groq hit max_tokens=%d - the reply is truncated. Raise "
                        "LLM_MAX_TOKENS or lower LLM_REASONING_EFFORT.",
                        settings.llm_max_tokens)
        return buffer, finish_reason

    # ------------------------------------------------------- lead extraction
    def parse_extra(self, full_reply: str) -> dict:
        """Parse the METADATA json block out of a completed reply."""
        m = METADATA_RE.search(full_reply)
        if not m:
            return {}
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            log.warning("METADATA was not valid JSON: %r", m.group(1)[:160])
            return {}
        if not isinstance(data, dict):
            return {}

        return {
            "name": data.get("name") or None,
            "requirement": data.get("requirement") or None,
            "intent_score": data.get("intent_score") or 0,
            "whatsapp_wanted": bool(data.get("whatsapp_wanted")),
            "handoff": bool(data.get("handoff")),
            "end_conversation": bool(data.get("end_conversation")),
        }

    def clean_for_speech(self, full_reply: str) -> str:
        """Return only the SPEAKABLE_RESPONSE text, never the metadata."""
        m = SPEAKABLE_RE.search(full_reply)
        if m:
            spoken = m.group(1).strip()
        else:
            # No marker: drop any METADATA block so JSON is never spoken.
            spoken = re.split(r"METADATA\s*:", full_reply, flags=re.I)[0].strip()
        spoken = PARTIAL_MARKER_RE.sub("", spoken).strip()
        return self._strip_json(spoken)

    @staticmethod
    def _strip_json(text: str) -> str:
        """Last line of defence: never let a stray JSON object reach the TTS."""
        text = re.sub(r"\{[^{}]*\}", " ", text)
        text = re.sub(r"^\s*(SPEAKABLE_RESPONSE|METADATA)\s*:\s*", "", text,
                      flags=re.I)
        return re.sub(r"\s{2,}", " ", text).strip()

    # ------------------------------------------------- English for retrieval

    # --------------------------------------------------------- off-topic safe
    def smalltalk_guard(self, user_text: str) -> str | None:
        """Instant deterministic replies for greetings/farewells (saves an LLM call)."""
        t = re.sub(r"[^\w\s]", "", user_text.lower()).strip()
        if not t:
            return UNCLEAR_LINE
        if t in ("hello", "hi", "hii", "hey", "hey there", "hi there",
                 "good morning", "good afternoon", "good evening", "vanakkam"):
            return GREETING_LINE
        if t in ("bye", "goodbye", "good bye", "ok bye", "okay bye",
                 "thank you", "thanks", "thank you bye", "thanks bye",
                 "thats all", "that is all"):
            return EXIT_LINE
        return None