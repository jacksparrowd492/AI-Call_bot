"""Tests for the two live-call failures seen in the 2026-09-03 call log:

  1. Groq returned HTTP 200 with an EMPTY completion three turns in a row, and
     the bridge spoke nothing at all - dead air on a live phone call.
  2. A truncated completion ended mid-marker ("...today?\nMET") and the bot
     read the fragment "MET" aloud as its own sentence.

The Groq client is replaced with a scripted stand-in, so this tests OUR
handling of reasoning tokens, empty content, usage-only chunks and truncated
markers - not the model.
"""
import os
import sys

os.environ.setdefault("GROQ_API_KEY", "gsk-test-not-a-real-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings          # noqa: E402
import llm as llm_mod                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


# ------------------------------------------------------------ scripted client
class _Delta:
    def __init__(self, content=None, reasoning=None):
        self.content, self.reasoning = content, reasoning


class _Choice:
    def __init__(self, delta, finish_reason=None):
        self.delta, self.finish_reason = delta, finish_reason


class _Event:
    def __init__(self, choices):
        self.choices = choices


class FakeCompletions:
    def __init__(self, script, reject_reasoning=False, dead_model=None):
        self.script, self.reject_reasoning = script, reject_reasoning
        self.dead_model = dead_model
        self.kwargs = []

    def create(self, **kw):
        self.kwargs.append(kw)
        if self.reject_reasoning and "reasoning_effort" in kw:
            raise TypeError("unsupported parameter: reasoning_effort")
        if self.dead_model and kw.get("model") == self.dead_model:
            raise RuntimeError(
                f"Error code: 404 - model_not_found: The model "
                f"`{self.dead_model}` does not exist or has been decommissioned")
        return iter(self.script)


class FakeModels:
    """Stands in for the account's model catalogue."""
    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def list(self):
        self.calls += 1
        return type("Resp", (), {
            "data": [type("M", (), {"id": i})() for i in self.ids]})()


class FakeClient:
    def __init__(self, script, reject_reasoning=False, dead_model=None, models=None):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(script, reject_reasoning, dead_model)
        if models is None:
            models = [settings.groq_model, settings.groq_fallback_model]
        self.models = FakeModels(models)


def reset_models():
    """The resolved-model cache is process-wide on purpose; clear it per test."""
    llm_mod._MODELS["resolved"] = None
    llm_mod._MODELS["available"] = None
    llm_mod._MODELS["dead"] = set()


def run(script, reject_reasoning=False, model=None, dead_model=None, models=None):
    reset_models()
    brain = llm_mod.GroqBrain()
    brain.client = FakeClient(script, reject_reasoning, dead_model, models)
    if model:
        brain.model = model
    out = list(brain.reply_stream([], "hello", ""))
    return brain, out


def text(content=None, reasoning=None, finish=None):
    return _Event([_Choice(_Delta(content, reasoning), finish)])


USAGE_ONLY = _Event([])          # Groq's final chunk carries no choices


print("\n=== 1. the empty-completion turn (reasoning ate the budget) ===")
brain, out = run([
    text(reasoning="Let me think about how to greet this caller. "),
    text(reasoning="They said hello, so the greeting line applies. "),
    text(content="", finish="length"),
    USAGE_ONLY,
])
check("no crash on a usage-only chunk", True)
check("something is always spoken", out == [llm_mod.UNCLEAR_LINE], str(out))
check("reasoning text never reaches TTS",
      not any("think" in s.lower() for s in out), str(out))

print("\n=== 2. the 'MET' leak (truncated mid-marker) ===")
brain, out = run([
    text(content="SPEAKABLE_RESPONSE:\nHello! How can I assist you with the "
                 "project today?\nMET", finish="length"),
    USAGE_ONLY,
])
check("truncated marker is not spoken", "MET" not in out, str(out))
check("the real sentences survive",
      out == ["Hello!", "How can I assist you with the project today?"], str(out))

print("\n=== 3. a well-formed completion still works ===")
brain, out = run([
    text(content="SPEAKABLE_RESPONSE:\n"),
    text(content="Karthipuram is a one hundred and ninety acre township."),
    text(content='\nMETADATA:\n{"name": "Karthi", "requirement": "location", '
                 '"intent_score": 5, "whatsapp_wanted": false, "handoff": false, '
                 '"end_conversation": false}', finish="stop"),
    USAGE_ONLY,
])
check("speaks only the speakable block",
      out == ["Karthipuram is a one hundred and ninety acre township."], str(out))
check("no JSON reaches TTS", not any("{" in s for s in out), str(out))
meta = brain.parse_extra(brain.last_raw)
check("metadata still parses", meta.get("name") == "Karthi" and
      meta.get("intent_score") == 5, str(meta))

print("\n=== 4. model selection and reasoning_effort ===")
OK = [text(content="SPEAKABLE_RESPONSE:\nHi there.", finish="stop"), USAGE_ONLY]

brain, out = run(list(OK))
kw = brain.client.chat.completions.kwargs[-1]
# The point is not "never a reasoning model" - this key only HAS reasoning
# models. The point is that the caller never pays for hidden reasoning, in
# tokens, latency or rate limit.
effort = kw.get("reasoning_effort")
check("the default model does not pay for hidden reasoning",
      effort in (None, "none"), f"{settings.groq_model} -> {effort}")
check("a plain instruct model gets no reasoning_effort at all",
      llm_mod.GroqBrain()._reasoning_effort("llama-3.3-70b-versatile") is None)
check("token budget comes from settings",
      kw.get("max_tokens") == settings.llm_max_tokens, str(kw.get("max_tokens")))

GPT_OSS = ["openai/gpt-oss-120b"]      # a key whose catalogue IS the reasoning model
brain, out = run(list(OK), model="openai/gpt-oss-120b", models=GPT_OSS)
kw = brain.client.chat.completions.kwargs[-1]
check("reasoning_effort IS sent to a reasoning model",
      kw.get("reasoning_effort") == settings.llm_reasoning_effort,
      str(kw.get("reasoning_effort")))

brain, out = run(list(OK), model="openai/gpt-oss-120b", models=GPT_OSS,
                 reject_reasoning=True)
calls = brain.client.chat.completions.kwargs
check("retries without reasoning_effort when rejected",
      len(calls) == 2 and "reasoning_effort" not in calls[1], str(len(calls)))
check("reply still produced after the retry", out == ["Hi there."], str(out))

print("\n=== 4b. a retired model id does not kill the call ===")
brain, out = run(list(OK), dead_model=settings.groq_model)
check("falls back to the backup model",
      brain.model == settings.groq_fallback_model, brain.model)
check("the caller still gets an answer", out == ["Hi there."], str(out))
calls = brain.client.chat.completions.kwargs
check("second attempt used the fallback model",
      calls[-1]["model"] == settings.groq_fallback_model, str(calls[-1]["model"]))

print("\n=== 4c. a model the key cannot access is caught BEFORE the caller waits ===")
# The real failure: GROQ_MODEL and GROQ_FALLBACK_MODEL both 404'd, so every
# turn of a live call answered "let me connect you to my team".
catalogue = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
             "whisper-large-v3", "meta-llama/llama-guard-4-12b"]
brain, out = run(list(OK), model="qwen-3.8-27B", models=catalogue)
check("swapped to a model the key really has",
      brain.model in catalogue, brain.model)
check("preferred the 70b chat model over the 8b",
      brain.model == "llama-3.3-70b-versatile", brain.model)
check("the request went out on the resolved model",
      brain.client.chat.completions.kwargs[0]["model"] == "llama-3.3-70b-versatile",
      str(brain.client.chat.completions.kwargs[0]["model"]))
check("caller gets a real answer, not the failure line",
      out == ["Hi there."], str(out))
check("the catalogue is fetched once, not per turn",
      brain.client.models.calls == 1, str(brain.client.models.calls))

print("\n=== 4d. non-chat models are never chosen as a fallback ===")
brain, out = run(list(OK), model="qwen-3.8-27B",
                 models=["whisper-large-v3", "playai-tts",
                         "meta-llama/llama-guard-4-12b", "gemma2-9b-it"])
check("skipped whisper/tts/guard, took the instruct model",
      brain.model == "gemma2-9b-it", brain.model)

# Nothing to substitute AND the configured id really is dead: exactly the
# 2026-09-04 call, where GROQ_MODEL and GROQ_FALLBACK_MODEL both 404'd. The
# bot must fail politely rather than raise into the media stream.
brain, out = run(list(OK), model="qwen-3.8-27B", models=["whisper-large-v3"],
                 dead_model="qwen-3.8-27B")
check("a key with no usable model degrades gracefully",
      out == ["Sorry, one moment - let me connect you to my team."], str(out))
check("the dead id is remembered, not retried forever",
      "qwen-3.8-27B" in llm_mod._MODELS["dead"], str(llm_mod._MODELS["dead"]))

print("\n=== 4e. the real catalogue on this Groq key (2026-09-04) ===")
REAL = ["allam-2-7b", "canopylabs/orpheus-arabic-saudi",
        "canopylabs/orpheus-v1-english", "groq/compound", "groq/compound-mini",
        "meta-llama/llama-prompt-guard-2-22m", "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3.6-27b", "qwen/qwen3.8-27b", "whisper-large-v3",
        "whisper-large-v3-turbo"]
usable = [m for m in REAL if not any(x in m.lower() for x in llm_mod._NOT_CHAT)]
for bad in ("canopylabs/orpheus-v1-english", "groq/compound",
            "whisper-large-v3", "meta-llama/llama-prompt-guard-2-22m",
            "openai/gpt-oss-safeguard-20b"):
    check(f"excluded {bad}", bad not in usable)
check("orpheus is a TTS model, not a chat model",
      not any("orpheus" in m for m in usable), str(usable))
check("compound does its own web search - never auto-picked",
      not any("compound" in m for m in usable), str(usable))

brain, out = run(list(OK), model="llama-3.3-70b-versatile", models=REAL)
check("picks the newest qwen, not qwen3.6",
      brain.model == "qwen/qwen3.8-27b", brain.model)
kw = brain.client.chat.completions.kwargs[-1]
check("qwen gets reasoning_effort=none, not 'low'",
      kw.get("reasoning_effort") == "none", str(kw.get("reasoning_effort")))
check("caller gets an answer on this key", out == ["Hi there."], str(out))

brain, out = run(list(OK), model="openai/gpt-oss-120b", models=REAL)
check("gpt-oss keeps low, not none",
      brain.client.chat.completions.kwargs[-1].get("reasoning_effort")
      == settings.llm_reasoning_effort,
      str(brain.client.chat.completions.kwargs[-1].get("reasoning_effort")))
check("bigger gpt-oss wins over the 20b",
      sorted(["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
             key=llm_mod._sort_key)[0] == "openai/gpt-oss-120b")

print("\n=== 5. clean_for_speech hardening ===")
b = llm_mod.GroqBrain()
cases = {
    "SPEAKABLE_RESPONSE:\nHello there.\nMET": "Hello there.",
    "SPEAKABLE_RESPONSE:\nHello there.\nMETADAT": "Hello there.",
    "SPEAKABLE_RESPONSE:\nHello there.\nMETADATA:": "Hello there.",
    'SPEAKABLE_RESPONSE:\nHello there.\nMETADATA:\n{"name": null}': "Hello there.",
    "Hello there.": "Hello there.",
}
for raw, want in cases.items():
    got = b.clean_for_speech(raw)
    check(f"clean_for_speech({raw[-14:]!r})", got == want, repr(got))
check("a sentence about metrics is not eaten",
      b.clean_for_speech("SPEAKABLE_RESPONSE:\nMetro station is nearby.")
      == "Metro station is nearby.")

print("\n" + "=" * 60)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("Failures: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
