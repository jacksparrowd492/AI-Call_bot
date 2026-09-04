"""Exercise every real dependency WITHOUT making a phone call.

    python tools/smoke_test.py

The unit tests replace Groq, Deepgram, Kokoro and Twilio with stand-ins, so
they cannot tell you whether this machine, this key and this database actually
work. This can. Each stage prints OK or FAIL and the reason.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, BAD = [], []


def stage(name, fn):
    print(f"\n--- {name} ---")
    t0 = time.monotonic()
    try:
        detail = fn()
        dt = time.monotonic() - t0
        print(f"  OK    ({dt:.2f}s) {detail or ''}")
        OK.append(name)
    except Exception as e:
        dt = time.monotonic() - t0
        print(f"  FAIL  ({dt:.2f}s) {type(e).__name__}: {e}")
        BAD.append(name)


# ------------------------------------------------------------------ 1. config
def check_config():
    from config import settings
    missing = settings.missing()
    if missing:
        raise RuntimeError("required settings are empty: " + ", ".join(missing))
    print(f"        groq model      : {settings.groq_model}")
    print(f"        groq fallback   : {settings.groq_fallback_model}")
    print(f"        embedding model : {settings.embedding_model}")
    print(f"        brochure url    : {settings.brochure_url}")
    print(f"        endpointing/turn: {settings.deepgram_endpointing_ms}ms / "
          f"{settings.deepgram_utterance_end_ms}ms")
    if "your-host" in settings.brochure_url or "example" in settings.brochure_url:
        raise RuntimeError("BROCHURE_URL is still a placeholder")
    return ""


# --------------------------------------------------------------- 2. embedding
def check_embeddings():
    from rag import retriever
    if retriever._embedder is None:
        raise RuntimeError("the local ONNX model failed to load - first run "
                           "needs internet to fetch it into ~/.cache/chroma")
    v = retriever.embed(["how many acres is karthipuram"])
    if not v or len(v[0]) != retriever.EMBED_DIMS:
        raise RuntimeError(f"expected a {retriever.EMBED_DIMS}-dim vector, got "
                           f"{len(v[0]) if v else 0}")
    return f"{retriever.EMBED_DIMS}-dim vectors"


# --------------------------------------------------------------- 3. knowledge
def check_rag():
    from rag import retriever
    n = retriever.count()
    if n == 0:
        raise RuntimeError("ChromaDB is empty - run `python -m rag.ingest`")

    probes = {
        "how many acres is karthipuram": ("acre", "190"),
        "what is the rera number": ("rera",),
        "tell me about the water supply": ("water",),
    }
    weak = []
    for q, expect in probes.items():
        ctx = retriever.retrieve(q)
        if not ctx:
            weak.append(f"{q!r} -> NO CONTEXT (dimension mismatch? re-ingest)")
        elif not any(w in ctx.lower() for w in expect):
            weak.append(f"{q!r} -> irrelevant match")
    if weak:
        raise RuntimeError("; ".join(weak))

    # And the opposite: smalltalk must NOT drag in a knowledge entry.
    if retriever.retrieve("Hello?"):
        raise RuntimeError("'Hello?' still returns context - lower RAG_MAX_DISTANCE")
    return f"{n} documents, 3/3 probes grounded, smalltalk correctly empty"


# --------------------------------------------------------------------- 4. LLM
def check_llm():
    from llm import GroqBrain
    brain = GroqBrain()
    if not brain.client:
        raise RuntimeError("no GROQ_API_KEY")

    from rag.retriever import retrieve
    q = "how many acres is karthipuram"
    sentences = list(brain.reply_stream([], q, retrieve(q)))

    if not sentences:
        raise RuntimeError("the model returned nothing at all")
    spoken = " ".join(sentences)
    print(f"        model in use : {brain.model}")
    print(f"        spoken       : {spoken}")

    meta = brain.parse_extra(brain.last_raw)
    if not meta:
        raise RuntimeError("METADATA block missing or not valid JSON - the "
                           "output contract is not holding on this model")
    print(f"        metadata     : {meta}")

    for leak in ("{", "METADATA", "SPEAKABLE_RESPONSE"):
        if leak in spoken:
            raise RuntimeError(f"{leak!r} leaked into the spoken text")
    if "sorry, one moment" in spoken.lower():
        raise RuntimeError("that is the failure line - the model call did not work")
    return f"{len(sentences)} sentence(s), metadata parsed, nothing leaked"


# --------------------------------------------------------------------- 5. TTS
def check_tts():
    from tts import _synthesize_sync
    pcm = _synthesize_sync("Karthipuram is a one hundred and ninety acre township.")
    if not pcm:
        raise RuntimeError("Kokoro produced no audio")
    return f"{len(pcm)} bytes of 8kHz PCM ({len(pcm) / 2 / 8000:.1f}s)"


# ---------------------------------------------------------------- 6. brochure
def check_brochure():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "brochure.pdf")
    if not os.path.exists(path):
        raise RuntimeError(f"missing {path} - /brochure.pdf will 404 and "
                           "WhatsApp will keep failing with error 21620")
    return f"{os.path.getsize(path) / 1e6:.1f} MB (served at /brochure.pdf)"


for name, fn in [
    ("1. configuration", check_config),
    ("2. local embeddings", check_embeddings),
    ("3. knowledge base", check_rag),
    ("4. groq completion", check_llm),
    ("5. kokoro tts", check_tts),
    ("6. brochure file", check_brochure),
]:
    stage(name, fn)

print("\n" + "=" * 62)
print(f"OK {len(OK)}   FAIL {len(BAD)}")
if BAD:
    print("Failed: " + ", ".join(BAD))
    print("\nThe bot is NOT ready for a call.")
else:
    print("\nEvery real dependency works. Safe to place a test call.")
sys.exit(1 if BAD else 0)
