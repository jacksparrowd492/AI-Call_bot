"""ChromaDB retriever over the Karthipuram real estate knowledge base.

Embeddings run LOCALLY via ChromaDB's bundled ONNX all-MiniLM-L6-v2. No API
key, no per-call cost, no network on the hot path, and no torch/transformers
install - onnxruntime and tokenizers (already in requirements.txt) are all it
needs. The model is downloaded once, on first use, into ~/.cache/chroma.

Why not OpenAI: text-embedding-3-large gives better recall on paraphrased
questions, but it puts a paid, rate-limited network call in the middle of a
live phone call. A 429 there costs the caller ~1.5s of retries and then
silently strips the bot of all project knowledge.

IMPORTANT: this model is 384-dim. A chroma_db built with 3072-dim vectors is
NOT query-compatible - delete chroma_db/ and re-run `python -m rag.ingest`.

Public surface:
    embed(texts, batch_size=...) -> list[list[float]]   (used by rag.ingest)
    embed_query(text)            -> list[float] | None  (cached, per-query)
    aembed(text)                 -> awaitable of the above (used by router)
    search(emb, n_results, where)-> list[hit dicts]
    dedupe_by_entry(hits, k) / build_context(hits)
    retrieve(query, top_k)       -> one context block of text
"""
import asyncio
import logging
import threading
from functools import lru_cache

from config import settings

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    raise SystemExit(
        "Missing RAG dependencies. Install them with:\n"
        "  pip install chromadb onnxruntime tokenizers\n"
        f"(original error: {e})"
    )

log = logging.getLogger("jarvis.rag")

DB_PATH = settings.chroma_dir
COLLECTION = settings.collection_name

# Exported for rag.ingest so ingest and retrieval can never drift apart.
EMBED_MODEL = settings.embedding_model or "all-MiniLM-L6-v2"
EMBED_DIMS = 384                       # fixed by the model - not configurable
DEFAULT_BATCH = 128
_MAX_LOAD_ATTEMPTS = 3

if settings.embedding_dimensions and settings.embedding_dimensions != EMBED_DIMS:
    log.warning("EMBEDDING_DIMENSIONS=%s is ignored: the local MiniLM model is "
                "fixed at %d dimensions.",
                settings.embedding_dimensions, EMBED_DIMS)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION)

# Looked up through the module (embedding_functions.ONNXMiniLM_L6_V2) rather
# than imported by name so tests can swap in a stand-in before import.
_embedder = None
_embedder_lock = threading.Lock()
_load_attempts = 0


def _load_embedder():
    """Build the ONNX session once. First ever call downloads ~80MB; every call
    after that is local and takes single-digit milliseconds."""
    global _embedder, _load_attempts
    with _embedder_lock:
        if _embedder is not None:
            return _embedder
        if _load_attempts >= _MAX_LOAD_ATTEMPTS:
            return None
        _load_attempts += 1
        try:
            ef = embedding_functions.ONNXMiniLM_L6_V2(
                preferred_providers=["CPUExecutionProvider"])
            ef(["warm up"])            # force the download + session build NOW
            _embedder = ef
            log.info("Local embedding model ready: %s (%d dims)",
                     EMBED_MODEL, EMBED_DIMS)
        except Exception as e:
            log.error("Could not load the local embedding model (attempt %d/%d): "
                      "%s. The first run needs internet to fetch the ONNX model.",
                      _load_attempts, _MAX_LOAD_ATTEMPTS, e)
        return _embedder


# ------------------------------------------------------------------ embedding

def _clean(text) -> str:
    """One line per input; the tokenizer does not need our whitespace."""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def detect_intent(query: str):
    """Classify the query intent for targeted RAG filtering.
    
    Returns: "location", "contact", "price", or "general"
    This enables intent-based metadata filtering to boost accuracy."""
    q = query.lower()

    if "where" in q and ("locat" in q or "situat" in q or "address" in q):
        return "location"

    if "contact" in q or "phone" in q or "call" in q or "reach" in q:
        return "contact"

    if "price" in q or "cost" in q or "rate" in q or "afford" in q:
        return "price"

    return "general"


def embed(texts, batch_size: int = DEFAULT_BATCH):
    """Embed a list of strings. Returns one vector per NON-BLANK input, in the
    order given. Blank inputs are dropped; an empty result does no work."""
    if isinstance(texts, str):
        texts = [texts]

    cleaned = [c for c in (_clean(t) for t in texts) if c]
    if not cleaned:
        return []

    embedder = _embedder or _load_embedder()
    if embedder is None:
        log.error("embed() called with no embedding model available")
        return []

    step = max(1, batch_size)
    out = []
    for i in range(0, len(cleaned), step):
        chunk = cleaned[i:i + step]
        try:
            vecs = embedder(chunk)
        except Exception as e:
            log.exception("Local embedding failed (batch %d-%d): %s",
                          i, i + len(chunk), e)
            return []
        if len(vecs) != len(chunk):
            log.error("Embedding count mismatch: sent %d, got %d",
                      len(chunk), len(vecs))
            return []
        # ONNX returns numpy arrays; Chroma wants plain lists of floats.
        out.extend([float(x) for x in v] for v in vecs)

    return out


@lru_cache(maxsize=512)
def _embed_query_cached(normalized: str):
    """Cached on the NORMALIZED query, so 'What is the price?' and
    'what is the price?' share one embedding. Callers use embed_query()."""
    vecs = embed([normalized])
    return vecs[0] if vecs else None


def embed_query(query: str):
    """Embed one caller utterance. Returns None when embedding is unavailable."""
    normalized = " ".join(str(query or "").lower().split())
    if not normalized:
        return None
    return _embed_query_cached(normalized)


async def aembed(text: str):
    """Await an embedding without blocking the event loop. Local inference is
    fast but it is still CPU work, and the call bridge cannot afford to stall."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_query, text)


# ------------------------------------------------------------------ searching

def search(embedding, n_results: int = 32, where=None):
    """Vector search. Returns [{id, document, meta, distance}, ...] nearest
    first, or [] on any failure - retrieval must never take a call down."""
    if embedding is None:
        return []

    total = collection.count()
    if total == 0:
        log.warning("ChromaDB collection is EMPTY - run `python -m rag.ingest` first")
        return []

    try:
        res = collection.query(
            query_embeddings=[list(embedding)],
            n_results=max(1, min(n_results, total)),
            where=where or None,
        )
    except Exception as e:
        if "dimension" in str(e).lower():
            log.error("Embedding dimension mismatch: the collection at %s was "
                      "built with a different model than %s (%d dims). Delete it "
                      "and re-run `python -m rag.ingest`. (%s)",
                      DB_PATH, EMBED_MODEL, EMBED_DIMS, e)
        else:
            log.exception("ChromaDB query failed: %s", e)
        return []

    docs = (res.get("documents") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits = []
    for i, doc in enumerate(docs):
        hits.append({
            "id": ids[i] if i < len(ids) else None,
            "document": doc,
            "meta": (metas[i] if i < len(metas) else None) or {},
            "distance": dists[i] if i < len(dists) else 2.0,
        })
    return hits


def dedupe_by_entry(hits, top_k: int = 3):
    """The KB embeds one document PER QUESTION PHRASING, so the raw top-k is
    usually several rewordings of the SAME answer. Keep the best hit per
    knowledge entry so the LLM sees top_k DISTINCT facts."""
    seen, out = set(), []
    for h in hits:
        meta = h.get("meta") or {}
        key = meta.get("entry_id", meta.get("entry"))
        if key is None:
            key = h.get("id") or h.get("document")
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def build_context(hits) -> str:
    """Join hit documents into the single context block the LLM prompt takes."""
    return "\n\n".join(h["document"] for h in hits if h.get("document"))


# ------------------------------------------------------------------ retrieval

def retrieve(query: str, top_k: int = None, max_distance: float = None) -> str:
    """Return the top knowledge-base matches joined into one context block.
    
    IMPROVED: Uses intent detection for targeted filtering + relaxed distance for
    important queries + fallback logic for edge cases. This fixes the "no match for
    location queries" problem while keeping false matches out.
    """
    if top_k is None:
        top_k = settings.rag_top_k
    if max_distance is None:
        max_distance = settings.rag_max_distance

    if not query or not str(query).strip():
        return ""

    if collection.count() == 0:
        log.warning("ChromaDB collection is EMPTY - run `python -m rag.ingest` first")
        return ""

    # ✅ STEP 1: INTENT DETECTION
    intent = detect_intent(query)

    emb = embed_query(query)
    if emb is None:
        log.warning("retrieve: no embedding for query=%r - returning no context",
                    str(query)[:80])
        return ""

    # ✅ STEP 2: FILTER BY INTENT (BOOST ACCURACY)
    where_filter = None
    if intent != "general":
        where_filter = {"intent": intent}

    hits = search(emb, n_results=max(top_k * 10, 20), where=where_filter)

    if not hits:
        log.info("RAG: no documents for intent=%s query=%r - no context",
                 intent, str(query)[:80])
        return ""

    # ✅ STEP 3: RELAX DISTANCE FOR IMPORTANT QUERIES
    if intent in ["location", "contact"]:
        relaxed_distance = max_distance + 0.5
    else:
        relaxed_distance = max_distance

    close = [h for h in hits if h["distance"] <= relaxed_distance]

    # ✅ STEP 4: FALLBACK (IMPORTANT)
    # If no results pass the distance threshold, take the top 1 instead of returning
    # empty. This is critical for location queries where MiniLM is imperfect.
    if not close:
        log.info("RAG: nothing within %.2f for intent=%s query=%r (best d=%.3f) - "
                 "using fallback", relaxed_distance, intent, str(query)[:60],
                 hits[0]["distance"])
        close = hits[:1]

    top = dedupe_by_entry(close, top_k)
    
    log.info(
        "RAG FIXED: intent=%s | results=%d | best_d=%.3f | top=%s",
        intent,
        len(top),
        top[0]["distance"] if top else 2.0,
        (top[0]["document"][:80] + "...") if top else "<empty>"
    )

    return build_context(top)


def count() -> int:
    try:
        return collection.count()
    except Exception as e:
        log.warning("count() failed: %s", e)
        return 0


# Warm the model at import (server startup), never mid-call.
_load_embedder()

log.info("RAG ready: collection='%s' db='%s' count=%d model='%s' dims=%d",
         COLLECTION, DB_PATH, collection.count(), EMBED_MODEL, EMBED_DIMS)
