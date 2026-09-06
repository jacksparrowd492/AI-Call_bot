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
import hashlib
import logging
import os
import re
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
# How far past the distance threshold retrieve() will still accept the single
# nearest document for a location or contact question rather than hand the
# model nothing at all. It applies to those two intents only - see retrieve().
FALLBACK_MARGIN = 0.35

# Extra allowance for a turn of one or two words. RAG_MAX_DISTANCE is
# calibrated on full questions, and a bare noun is not one: with no sentence
# around it, its embedding sits further from EVERY document. Measured against
# the real vectors, "water" lands at 1.328 with a floor of 1.20 - and all
# three nearest documents are the water supply entry, so the floor was
# rejecting the right answer for being tersely asked. A greeting measured at
# 1.673 on the same store, which is why this allowance is 0.25 and not more.
SHORT_QUERY_MARGIN = 0.25
SHORT_QUERY_WORDS = 2
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

    Returns: "location", "contact", "price", or "general".

    This only NARROWS the search. It never decides the answer, and a wrong
    guess here must not cost the caller their context - see retrieve(), which
    repeats the search unfiltered whenever the narrowed one comes back empty.
    """
    q = " " + re.sub(r"[^a-z0-9]+", " ", query.lower()) + " "

    def has(*words):
        return any(" %s " % w in q for w in words)

    if has("where") and has("located", "location", "situated", "address",
                            "locality", "area"):
        return "location"

    # Whole words only. "call" as a substring also lives inside "called",
    # "recall" and "calling me back", and matching those routed ordinary
    # project questions into the contact bucket, where the metadata filter
    # then hid every entry that could have answered them.
    if has("contact", "phone", "reach", "email", "call"):
        return "contact"

    if has("price", "prices", "cost", "costs", "rate", "rates", "budget",
           "afford", "affordable", "emi", "payment"):
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
    """Join hits into the single context block the LLM prompt takes.

    A TOPIC document (rag.ingest) is deliberately short - a category, a topic
    line and the entry's keywords, and no answer at all - because a two-word
    caller turn like "water" has to land close to it, and every extra word of
    answer text pushes that distance up past the threshold. The answer travels
    in the metadata instead and is put back here, so what the model sees is
    always a complete answer whichever document matched.
    """
    blocks = []
    for h in hits:
        doc = (h.get("document") or "").rstrip()
        if not doc:
            continue
        meta = h.get("meta") or {}
        answer = meta.get("answer")
        if answer and "\nA: " not in doc:
            doc += "\nA: " + answer
            facts = meta.get("facts")
            if facts:
                doc += "\nKey facts: " + facts
        blocks.append(doc)
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ retrieval

# A caller very often puts two topics in one breath - "infrastructure and
# water supply", "location and price", "amenities, parks and security". One
# embedding of the whole sentence sits BETWEEN both topics and matches neither
# well: "Infrastructure and water supply" measured at 1.162 against the roads
# entry and never reached the water entry at all, so the caller was told about
# tar roads and then offered a callback for the half that was in the KB the
# whole time. Each clause is therefore searched on its own as well, and the
# results are merged.
_CLAUSE_SPLIT = re.compile(r"\s*(?:,|\band\b|\balso\b|\bplus\b|&)\s*", re.I)
# Words that carry no topic on their own, so a clause made only of these is
# not worth a search of its own.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "it", "its", "this", "that",
    "there", "here", "what", "whats", "how", "about", "of", "for", "to", "in",
    "on", "at", "me", "my", "you", "your", "we", "our", "us", "please", "tell",
    "say", "know", "want", "would", "like", "can", "could", "do", "does", "did",
    "give", "some", "any", "more", "just", "ok", "okay", "yes", "no", "and",
    "also", "then", "so", "well", "hmm", "um", "uh",
}


def _split_clauses(query: str):
    """Split a turn into the separate topics it actually asks about.

    Returns [] when the turn is a single topic - the caller then pays for one
    search, exactly as before.
    """
    parts = []
    for raw in _CLAUSE_SPLIT.split(str(query or "")):
        clause = raw.strip(" ?.!,;:")
        if not clause:
            continue
        words = [w for w in re.sub(r"[^a-z0-9\s]", " ", clause.lower()).split()
                 if w not in _STOPWORDS]
        if not words:
            continue
        parts.append(clause)
    if len(parts) < 2:
        return []
    # Two topics take more than two words to say. Below that it is one topic
    # with a stray connective in it.
    if len(str(query).split()) < 3:
        return []
    return parts


def _close_hits(query: str, top_k: int, max_distance: float,
                short_margin: bool = True):
    """The hits for ONE query that are close enough to be used. [] means the
    knowledge base does not cover it - never "the metadata filter missed"."""
    # STEP 1: what KIND of question is this? Used to narrow the search, never
    # to answer it.
    intent = detect_intent(query)

    emb = embed_query(query)
    if emb is None:
        log.warning("retrieve: no embedding for query=%r - returning no context",
                    str(query)[:80])
        return []

    # STEP 2: try the narrowed search first...
    hits = []
    if intent != "general":
        hits = search(emb, n_results=max(top_k * 10, 20),
                      where={"intent": intent})

    # ...and ALWAYS repeat it unfiltered when that comes back empty. The filter
    # matches metadata written at ingest time, so a collection built by a
    # different script - or from a KB whose entries carry no intent - returned
    # ZERO documents for every query containing "call", "price", "contact" or
    # "reach". No documents means no context, and no context means the bot told
    # callers it did not have information it was actually holding.
    if not hits:
        if intent != "general":
            log.info("RAG: intent=%s matched no metadata - searching unfiltered",
                     intent)
        hits = search(emb, n_results=max(top_k * 10, 20))

    if not hits:
        log.info("RAG: no documents at all for query=%r - no context",
                 str(query)[:80])
        return []

    # STEP 3: two kinds of question need more room than a full sentence about
    # the township does.
    #
    # Location and contact, because a place name carries little semantic
    # signal and MiniLM is weakest on them. And SHORT turns, because callers
    # say the topic and stop - "water", "transport service", "amenities" - and
    # the floor was calibrated on full questions. Both allowances are additive
    # on purpose: "the address?" is short AND a location question, and is
    # exactly the turn that used to come back empty twice over.
    relaxed_distance = (max_distance + 0.5 if intent in ("location", "contact")
                        else max_distance)
    if short_margin and len(str(query).split()) <= SHORT_QUERY_WORDS:
        relaxed_distance += SHORT_QUERY_MARGIN

    close = [h for h in hits if h["distance"] <= relaxed_distance]

    # STEP 4: a NARROW fallback, for location and contact questions only.
    #
    # Those are the two the embedder is worst at, and the two where an empty
    # context is most obviously wrong to a caller. Everything else gets no
    # fallback at all: nearest-neighbour search always returns SOMETHING, and
    # taking it regardless of distance is how "Hello?" (d=1.67 on the real call
    # log) pulled a random knowledge entry in for the model to answer from.
    # Past the threshold, an empty context is the honest answer and the
    # not-found line - which offers the callback - is the right reply.
    if (not close and intent in ("location", "contact")
            and hits[0]["distance"] <= relaxed_distance + FALLBACK_MARGIN):
        log.info("RAG: nothing within %.2f for intent=%s query=%r (best d=%.3f) - "
                 "using the nearest anyway", relaxed_distance, intent,
                 str(query)[:60], hits[0]["distance"])
        close = hits[:1]

    if not close:
        # The number quoted here is the one the hit was actually measured
        # against, including the fallback margin only where the fallback could
        # have applied. It read 0.35 too high on every other intent.
        ceiling = relaxed_distance + (FALLBACK_MARGIN
                                      if intent in ("location", "contact") else 0.0)
        log.info("RAG: best match for %r is %.3f, past %.2f - no context",
                 str(query)[:60], hits[0]["distance"], ceiling)
        return []

    return close


def retrieve(query: str, top_k: int = None, max_distance: float = None) -> str:
    """Return the top knowledge-base matches joined into one context block.

    An empty result means "the knowledge base does not cover this", which
    llm.py turns into the not-found line plus a callback offer. It must
    therefore mean exactly that, and never "the metadata filter did not match".
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

    close = list(_close_hits(query, top_k, max_distance))

    # A two-topic turn gets each topic searched on its own as well, so the
    # answer to the second half cannot be lost behind the first.
    #
    # ONLY once the turn as a whole has matched something, though. A clause is
    # shorter than the sentence it came from and therefore matches more
    # loosely, so splitting first would hand "Hello there, can you hear me?"
    # two more chances to drag a knowledge entry into a greeting. If the whole
    # turn is not a project question, neither are its halves. The short-turn
    # allowance is withheld here for the same reason: it exists for a caller
    # who SAID two words, not for two words we cut out of their sentence.
    clauses = _split_clauses(query) if close else []
    if clauses:
        log.info("RAG: multi-topic turn - also searching %s",
                 " | ".join(repr(c[:30]) for c in clauses))
        for clause in clauses:
            close.extend(_close_hits(clause, top_k, max_distance,
                                     short_margin=False))

    if not close:
        return ""

    close.sort(key=lambda h: h["distance"])
    top = dedupe_by_entry(close, top_k)

    log.info("RAG: results=%d | best_d=%.3f | top=%s",
             len(top), top[0]["distance"] if top else 2.0,
             (top[0]["document"][:80] + "...") if top else "<empty>")

    return build_context(top)


def count() -> int:
    try:
        return collection.count()
    except Exception as e:
        log.warning("count() failed: %s", e)
        return 0


# The KB file the vectors are SUPPOSED to have been built from. rag.ingest
# stamps the file it used into the collection's metadata; anything else means
# the bot is answering out of a stale index. That is not a hypothetical: the
# live collection was built from an older data file with no transport entry,
# so "how is the transport facility?" had nothing to match for weeks and the
# JSON on disk gave no hint, because nothing reads the JSON at runtime.
EXPECTED_SOURCE = "knowledge_base.json"
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", EXPECTED_SOURCE)


def kb_fingerprint(path=None):
    """SHA1 of the knowledge base file, so a rebuild can be checked against
    the file's CONTENT rather than only its name."""
    try:
        with open(path or KB_PATH, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return ""


def _check_source():
    try:
        meta = collection.metadata or {}
    except Exception:
        return
    source = meta.get("source")
    if source == EXPECTED_SOURCE:
        stamped = meta.get("kb_sha1")
        current = kb_fingerprint()
        if stamped and current and stamped != current:
            log.warning("STALE INDEX: data/%s has been edited since the vector "
                        "store was built (%s). The bot will keep answering "
                        "from the OLD text until you run: python -m rag.ingest",
                        EXPECTED_SOURCE, meta.get("built_at", "?"))
        elif not stamped:
            log.warning("The vector store predates content stamping - rebuild "
                        "it once with `python -m rag.ingest` so edits to "
                        "data/%s can be detected.", EXPECTED_SOURCE)
        else:
            log.info("KB source: %s (built %s)", source, meta.get("built_at", "?"))
    elif source:
        log.warning("The vector store was built from %r, not %r. Rebuild it: "
                    "python -m rag.ingest", source, EXPECTED_SOURCE)
    else:
        log.warning("The vector store carries no build stamp, so it predates "
                    "the current knowledge base. Rebuild it: python -m rag.ingest")


# Warm the model at import (server startup), never mid-call.
_load_embedder()

log.info("RAG ready: collection='%s' db='%s' count=%d model='%s' dims=%d",
         COLLECTION, DB_PATH, collection.count(), EMBED_MODEL, EMBED_DIMS)
_check_source()
