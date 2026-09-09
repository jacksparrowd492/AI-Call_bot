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
_data = os.path.join(_here, "data", "knowledge_base.json")
if not os.path.exists(_data):
    _data = os.path.join(_here, "..", "data", "knowledge_base.json")
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
    # Transport returned nothing for weeks, because the collection had been
    # built from a KB with no transport entry at all, and the bot asked what
    # the caller meant instead of answering.
    #
    # Bare noun phrases - "transport service", which is how callers actually
    # ask - are NOT asserted here on purpose. The stand-in embedder above is a
    # hashed bag of words with no semantics, so a two-word query says nothing
    # about whether the real MiniLM would match. That is what
    # `python -m tools.ask_kb --check` is for, against the real vectors.
    "how is the transport facility there": ["transport", "metro", "bus",
                                            "connectivity"],
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
# "Hello?" came back at d=1.673 against RAG_MAX_DISTANCE=1.2, and the bot tried
# to answer a greeting out of the knowledge base.
#
# What has to hold is that the FLOOR decides, and that is what is tested here.
# Whether a particular phrase lands above or below it is a property of
# all-MiniLM-L6-v2, and the stand-in embedder above is a hashed bag of words
# with no semantics - it cannot answer that question, so it is not asked to.
# (On a live call these never reach the retriever at all: bridge.py answers
# greetings from the smalltalk guard, with no RAG and no LLM.)
# Three words or more, so the short-turn allowance below is not in play here -
# this loop is about the floor itself.
for q in ("Hello there, can you hear me?", "Hold on a second.", "zzz qqq foo"):
    # n_results has to MATCH what retrieve() searches with internally. Asking
    # for 5 here and comparing against a call that asks for 30 made this flaky:
    # Chroma's HNSW index is approximate, so the wider search sometimes found a
    # nearer document than the narrow one had, and `best - 0.01` was then not
    # actually below the best. Measured over repeated runs the same phrase came
    # back at 1.626, 1.676, 1.704 and 1.720. (Pre-existing; it has nothing to do
    # with the lexical arm - it reproduces with that arm disabled.)
    best = retriever.search(retriever.embed_query(q),
                            n_results=max(3 * 10, 20))[0]["distance"]
    check(f"{q!r} is dropped by a floor below its best match",
          retriever.retrieve(q, top_k=3, max_distance=best - 0.01) == "",
          "best d=%.3f" % best)
    check(f"{q!r} is context only when the floor lets it be",
          bool(retriever.retrieve(q, top_k=3, max_distance=best + 0.01)),
          "best d=%.3f" % best)

# A one- or two-word turn gets a documented allowance on top of the floor,
# because the floor is calibrated on full questions and a bare noun is not one.
# Measured on the real vectors: "water" sits at 1.328 against a floor of 1.20,
# with the water supply entry as all three nearest documents.
short = "water"
best_short = retriever.search(retriever.embed_query(short),
                              n_results=max(3 * 10, 20))[0]["distance"]
floor = best_short - 0.01
check("a two-word turn is allowed past the sentence floor",
      bool(retriever.retrieve(short, top_k=3, max_distance=floor)),
      "best d=%.3f, floor %.3f + margin %.2f"
      % (best_short, floor, retriever.SHORT_QUERY_MARGIN))

# ---------------------------------------------------------------------------
# The three checks below measure a property of the DISTANCE FLOOR, so the
# lexical arm is held off for them.
#
# Since 2026-09-09 retrieval has two arms, and the lexical one answers on term
# overlap without consulting the distance at all - that is the entire point of
# it, and it is why "sports" is answerable now. Left switched on, it answers
# every one of these regardless of the threshold and the floor being measured
# becomes unobservable. Holding it off keeps the original assertions honest
# instead of deleting them; the hybrid contract is asserted separately below.
from rag import lexical                                          # noqa: E402
_real_search = lexical.search
lexical.search = lambda *a, **k: []
try:
    check("but not past the floor plus its margin",
          retriever.retrieve(short, top_k=3,
                             max_distance=best_short
                             - retriever.SHORT_QUERY_MARGIN - 0.01) == "",
          "best d=%.3f" % best_short)
    check("and a longer turn gets no such allowance",
          retriever.retrieve("tell me about the water supply here", top_k=3,
                             max_distance=0.01) == "")
    check("an impossible threshold drops the vector arm entirely",
          retriever.retrieve("what is the rera number", max_distance=0.0) == "")
finally:
    lexical.search = _real_search

# --------------------------------------------------------- the hybrid contract
print("\n=== 10b. hybrid: the lexical arm answers what the floor cannot ===")
check("a real question still gets context",
      bool(retriever.retrieve("what is the rera number", top_k=3)))
# The reported bug, as a test. "sports" appears nowhere in a question phrasing
# or a keyword list - only inside a long answer and its facts - so no distance
# threshold reaches it. Term overlap does.
check("'sports' is answered even with the vector floor shut to zero",
      bool(retriever.retrieve("sports", top_k=3, max_distance=0.0)))
check("'basketball' likewise", bool(retriever.retrieve("basketball", top_k=3,
                                                       max_distance=0.0)))
# ...and the guard on the other side, which matters more than any of it: an
# arm that answers without consulting distance must still refuse a greeting.
for noise in ("hello", "okay thank you", "hold on a second",
              "my name is karthi", "that sounds good", "zzz qqq foo"):
    check("hybrid still returns NOTHING for %r" % noise,
          retriever.retrieve(noise, top_k=3, max_distance=0.0) == "")
check("the floor defaults to settings.rag_max_distance",
      retriever.settings.rag_max_distance == 2.0,
      str(retriever.settings.rag_max_distance))

print("\n=== 11. the shipped knowledge base builds the documents we expect ===")
# The bug this guards: the live collection had been built from an older data
# file with no transport entry, so "transport service" had nothing to match
# and the bot answered it with a question, on every call, forever. These
# checks need no embedder - they are about WHAT gets embedded.
from rag import ingest                                      # noqa: E402

KB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "knowledge_base.json")
kb_entries, kb_project = ingest.load_payload(KB_FILE)
kb_docs, kb_ids, kb_metas = ingest.build_documents(kb_entries, kb_project)

check("the authored KB schema loads", len(kb_entries) >= 39, str(len(kb_entries)))
check("ids are unique", len(kb_ids) == len(set(kb_ids)),
      "%d ids, %d unique" % (len(kb_ids), len(set(kb_ids))))
check("chroma metadata stays scalar",
      all(isinstance(v, (str, int, float, bool))
          for m in kb_metas for v in m.values()))

by_entry = {}
for doc, _id, meta in zip(kb_docs, kb_ids, kb_metas):
    by_entry.setdefault(meta.get("entry_id"), []).append((_id, doc))

check("every entry is embedded under its own id",
      all(e["id"] in by_entry for e in kb_entries),
      str([e["id"] for e in kb_entries if e["id"] not in by_entry]))

transport = by_entry.get("transport_connectivity", [])
check("transport and connectivity is in the knowledge base", bool(transport))
topic_docs = [d for i, d in transport if i.startswith("topic_")]
check("it gets a TOPIC document, which is what a bare noun matches",
      len(topic_docs) == 1, str([i for i, _ in transport if i.startswith("topic_")]))
if topic_docs:
    t = topic_docs[0].lower()
    check("the topic document carries the caller's own words",
          all(w in t for w in ("transport", "metro", "bus", "commute")), t[:80])
    # Short ON PURPOSE. "water" against sixty words of answer lands past the
    # distance threshold and comes back empty - measured against the real
    # vectors, not guessed. The answer rides in the metadata instead.
    check("and NOT the answer, so a two-word turn stays close to it",
          "a: " not in t and len(t.split()) < 25, "%d words" % len(t.split()))

topic_meta = [m for m, i in zip(kb_metas, kb_ids)
              if i == "topic_%d" % next(n for n, e in enumerate(kb_entries)
                                        if e["id"] == "transport_connectivity")]
check("the answer travels in the metadata",
      bool(topic_meta) and "metro station" in topic_meta[0].get("answer", ""),
      str(topic_meta[0].get("answer", "")[:60]) if topic_meta else "")
expanded = retriever.build_context([{"document": topic_docs[0],
                                     "meta": topic_meta[0]}])
check("and build_context puts it back, so a topic hit is still an answer",
      "A: " in expanded and "metro station" in expanded
      and "Key facts:" in expanded, expanded[:70].replace("\n", " "))
check("an ordinary Q/A document is not double-answered",
      retriever.build_context([{"document": "Category: X\nQ: q\nA: real",
                                "meta": {"answer": "real"}}])
      == "Category: X\nQ: q\nA: real")

check("every entry with keywords gets exactly one topic document",
      all(len([i for i, _ in by_entry.get(e["id"], []) if i.startswith("topic_")]) == 1
          for e in kb_entries if e["keywords"] or e["topic"]))

# The intent metadata retrieve() filters on has to exist, or the filtered
# search returns nothing and the caller is told we have no information.
check("location entries are tagged location",
      by_entry["transport_connectivity"]
      and all(m["intent"] == "location" for m in kb_metas
              if m.get("entry_id") == "transport_connectivity"))
check("intent is only ever a bucket retrieve() knows",
      {m["intent"] for m in kb_metas} <= {"location", "contact", "price", "general"},
      str(sorted({m["intent"] for m in kb_metas})))

print("\n" + "=" * 60)
print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
if FAIL:
    print("Failures: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
