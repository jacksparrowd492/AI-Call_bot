"""End-to-end test of the local ONNX-embedding RAG path.

The ONNX MiniLM embedder is replaced with a deterministic local stand-in
(hashed bag-of-words -> 384 dims, L2 normalised). That is NOT a quality test of
all-MiniLM-L6-v2; it is a test that OUR code around it is correct: lazy model
loading, batching, ordering, dimensions, caching, the Chroma round-trip,
dedupe by entry, and the dimension-mismatch branch.
"""
import hashlib
import os
import shutil
import sys

os.environ["CHROMA_DIR"] = os.path.join(os.path.dirname(__file__), "chroma_test")
os.environ["CHROMA_COLLECTION"] = "real_estate"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
# The stand-in embedder below is a hashed bag of words with no semantics, so
# its distances are nothing like MiniLM's. The production floor (1.5) is
# calibrated on real call-log distances; here the floor is opened up and the
# threshold LOGIC is tested explicitly in section 10 instead.
os.environ["RAG_MAX_DISTANCE"] = "2.0"

shutil.rmtree(os.environ["CHROMA_DIR"], ignore_errors=True)

DIMS = 384
CALLS = {"n": 0, "init": 0, "batches": [], "texts": []}


# ------------------------------------------------ fake ONNX embedding function
def _vec(text, dims=DIMS):
    """Deterministic hashed bag-of-words vector, L2 normalised."""
    v = [0.0] * dims
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dims] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class FakeONNX:
    def __init__(self, preferred_providers=None):
        CALLS["init"] += 1
        self.preferred_providers = preferred_providers

    def __call__(self, input):
        CALLS["n"] += 1
        CALLS["batches"].append(len(input))
        CALLS["texts"].extend(input)
        return [_vec(t) for t in input]


class Broken:
    """Stands in for an embedder whose ONNX session blows up at inference."""
    def __call__(self, input):
        raise RuntimeError("onnxruntime session failed")


from chromadb.utils import embedding_functions          # noqa: E402
embedding_functions.ONNXMiniLM_L6_V2 = FakeONNX

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import retriever            # noqa: E402
from rag import ingest               # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


print("\n=== 0. model loads once, at import ===")
check("embedder built at import (warm, not mid-call)", CALLS["init"] == 1,
      f"init={CALLS['init']}")
check("CPU execution provider requested",
      retriever._embedder.preferred_providers == ["CPUExecutionProvider"])
check("warm-up call made", CALLS["n"] == 1, f"{CALLS['n']} calls")

print("\n=== 1. embed() basics ===")
CALLS["n"], CALLS["batches"] = 0, []
vecs = retriever.embed(["hello world", "what is the price"])
check("returns one vector per input", len(vecs) == 2, f"got {len(vecs)}")
check("vector is 384-dim", len(vecs[0]) == DIMS, f"got {len(vecs[0])}")
check("EMBED_DIMS matches reality", retriever.EMBED_DIMS == DIMS)
check("order preserved", vecs[0] == _vec("hello world"))
check("plain python floats for chroma",
      all(isinstance(x, float) for x in vecs[0][:8]))
check("model not re-loaded per call", CALLS["init"] == 1, f"init={CALLS['init']}")

print("\n=== 2. embed() batching ===")
CALLS["n"], CALLS["batches"] = 0, []
big = [f"question number {i}" for i in range(300)]
vb = retriever.embed(big, batch_size=128)
check("300 docs -> 300 vectors", len(vb) == 300, f"got {len(vb)}")
check("300 docs -> 3 batches, not 300", CALLS["n"] == 3, f"{CALLS['n']} calls")
check("batch sizes 128/128/44", CALLS["batches"] == [128, 128, 44], str(CALLS["batches"]))

print("\n=== 3. embed() input hygiene ===")
CALLS["n"] = 0
check("empty list -> no inference", retriever.embed([]) == [] and CALLS["n"] == 0)
check("blank strings dropped", retriever.embed(["   ", ""]) == [])
CALLS["n"], CALLS["texts"] = 0, []
retriever.embed(["line one\nline two"])
check("newlines stripped before encoding",
      CALLS["texts"] == ["line one line two"], str(CALLS["texts"]))

print("\n=== 4. query cache ===")
retriever._embed_query_cached.cache_clear()
CALLS["n"] = 0
retriever.embed_query("What is the price?")
retriever.embed_query("what is the price?")      # different case
retriever.embed_query("  What is the price?  ")  # whitespace
check("3 equivalent queries -> 1 inference", CALLS["n"] == 1, f"{CALLS['n']} calls")
info = retriever._embed_query_cached.cache_info()
check("cache reports 2 hits", info.hits == 2, str(info))

print("\n=== 5. real ingest into Chroma ===")
_here = os.path.dirname(os.path.abspath(__file__))
_data = os.path.join(_here, "data", "realestatedata.json")
if not os.path.exists(_data):
    _data = os.path.join(_here, "..", "data", "realestatedata.json")
ingest.main(_data)
n = retriever.count()
check("collection is populated", n > 0, f"count={n}")

print("\n=== 6. real retrieval ===")
queries = {
    "where is the project located": ["coimbatore", "avinashi", "neelambur", "location"],
    "how many acres is karthipuram": ["190", "acre"],
    "what is the rera number": ["rera", "tn/", "3352"],
    "tell me about the water supply": ["water", "bhavani", "bore"],
    "what amenities do you have": ["park", "road", "amenit", "underground"],
}
for q, expect in queries.items():
    ctx = retriever.retrieve(q, top_k=3)
    hit = any(w in ctx.lower() for w in expect)
    check(f"retrieve({q!r}) is relevant", bool(ctx) and hit,
          (ctx[:70].replace("\n", " ") + "...") if ctx else "EMPTY")

print("\n=== 7. dedupe by entry ===")
ctx = retriever.retrieve("where is the project located", top_k=3)
blocks = [b for b in ctx.split("\n\n") if b.strip()]
check("top_k respected", len(blocks) <= 3, f"{len(blocks)} blocks")
qs = [b.split("Q: ")[1].split("\n")[0] for b in blocks if "Q: " in b]
check("no duplicate question phrasings", len(qs) == len(set(qs)), str(len(qs)))

print("\n=== 8. degraded paths ===")
check("empty query -> empty context", retriever.retrieve("") == "")
check("whitespace query -> empty context", retriever.retrieve("   ") == "")

_real = retriever._embedder
try:
    retriever._embedder = Broken()
    retriever._embed_query_cached.cache_clear()
    out = retriever.retrieve("what is the price")
    check("inference crash -> empty context, no exception", out == "", repr(out))

    retriever._embedder = None
    retriever._load_attempts = 99          # simulate 'gave up loading the model'
    retriever._embed_query_cached.cache_clear()
    out = retriever.retrieve("what is the price")
    check("no model available -> empty context, no crash", out == "", repr(out))
finally:
    retriever._embedder = _real
    retriever._load_attempts = 0
    retriever._embed_query_cached.cache_clear()

print("\n=== 9. dimension mismatch (the 3072 -> 384 trap) ===")
import chromadb
c2 = chromadb.PersistentClient(path=os.environ["CHROMA_DIR"])
try:
    c2.delete_collection("dim_test")
except Exception:
    pass
old = c2.get_or_create_collection("dim_test")
old.add(ids=["a"], documents=["legacy openai doc"], embeddings=[[0.1] * 3072])

saved_coll = retriever.collection
retriever.collection = old
try:
    out = retriever.retrieve("anything at all")
    check("3072-dim collection + 384-dim query -> empty, not a crash",
          out == "", repr(out))
finally:
    retriever.collection = saved_coll

print("\n=== 10. weak matches are not context (the 'Hello?' trap) ===")
# Nearest-neighbour search ALWAYS returns something. On the real call log
# "Hello?" came back at d=1.673 and the bot tried to answer a greeting out of
# the knowledge base. These queries share no tokens with any document, so they
# sit at the far end of the distance range and any sane floor must drop them.
for q in ("Hello?", "Hold on.", "zzz qqq"):
    ctx = retriever.retrieve(q, top_k=3, max_distance=1.5)
    check(f"smalltalk {q!r} -> empty context", ctx == "",
          (ctx[:60].replace("\n", " ") + "...") if ctx else "")

check("a real question still gets context",
      bool(retriever.retrieve("what is the rera number", top_k=3)))
check("an impossible threshold drops everything",
      retriever.retrieve("what is the rera number", max_distance=0.0) == "")
check("the floor defaults to settings.rag_max_distance",
      retriever.settings.rag_max_distance == 2.0,
      str(retriever.settings.rag_max_distance))

print("\n" + "=" * 60)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("Failures: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
