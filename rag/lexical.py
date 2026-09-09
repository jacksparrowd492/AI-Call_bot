"""BM25 keyword search over the SAME documents the vectors are built from.

Why this exists
---------------
MiniLM is a dense embedder, and dense embedders fail in one very specific way:
a rare, discriminating word buried in a long document. Measured on this KB, a
caller saying "sports" got nothing, while "cricket" got the right answer
instantly - because "cricket" is a keyword on the short topic document and
"sports" only ever appeared inside a sixty-word answer plus twenty-five facts.
Embedding distance grows with everything in a document the query does NOT
account for, so a one-word query against a long document lands out past the
noise floor no matter how well the meaning matches.

Raising the distance ceiling cannot fix that. On the real store a greeting
measured 1.673 and the ceiling for a short query is already 1.75: there is no
gap left to give away. The fix has to come from a different KIND of matching,
not a looser threshold on the same one.

BM25 is that other kind. It scores on term overlap weighted by how RARE the
term is, so "sports", "basketball", "1882", "Athikadavu" and "RERA" - every
token a dense model is worst at - are exactly what it is best at.

No new dependency, deliberately
-------------------------------
This is ~60 lines of arithmetic and it is written out rather than pulled from
PyPI. The deployment is a Windows box behind `pip install -r requirements.txt`,
and every dependency is one more thing that can fail there on the day of a
demo. It also lets the IDF table be read directly, which is what the noise gate
below needs and what a library would hide.

Alignment
---------
The index is built from `rag.ingest.build_documents()` - the same call that
produces what goes into Chroma - so the lexical arm and the vector arm can
never be searching different corpora. Build cost is ~20ms for 353 documents,
paid once, lazily, on the first query.
"""
import logging
import math
import os
import re
import threading

log = logging.getLogger("jarvis.lexical")

K1 = 1.5          # term-frequency saturation. Standard Okapi BM25.
B = 0.75          # length normalisation. Standard.

# Words that carry no discriminating signal.
#
# The second block is the one that earns its keep, and it was built from
# MEASUREMENT rather than taste. Scoring the real call-log phrasings against
# non-questions showed the two overlapping badly: "my name is Karthi" scored
# 7.93 and "that sounds good" 5.59, while the genuine caller turns "amenities"
# and "cricket" scored 2.59 and 2.76. No score floor can separate those.
#
# Neither can rarity, which is the tempting fix: "name" appears in exactly ONE
# of the 391 documents, giving it the HIGHEST idf in the corpus, while
# "amenities" appears in 99 and has one of the lowest. IDF measures how rare a
# word is in the brochure, not how conversational it is, and those are simply
# different things.
#
# So the words below are the ones that appear in ordinary phone conversation
# AND happen to collide with brochure prose - "fine" from "fine dining",
# "right" and "now" from the disclaimer, "evening" from the office hours. This
# is the same device `stt.FILLERS` already is, applied one layer further in.
STOPWORDS = {
    # --- ordinary English function words ---
    "a", "about", "all", "am", "an", "and", "any", "anything", "are", "as",
    "at", "be", "been", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "get", "give", "had", "has", "have", "he", "her", "here",
    "hi", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "know", "like", "many", "me", "more", "much", "my", "no", "not",
    "of", "off", "ok", "okay", "on", "one", "or", "other", "our", "out",
    "please", "she", "should", "so", "some", "tell", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "up",
    "us", "want", "was", "we", "well", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your", "yes",
    "yeah", "hello", "hey", "thanks", "thank", "sir", "madam", "second",
    "minute", "wait", "hold", "sorry", "again", "also", "very", "really",

    # --- conversational filler that collides with the brochure ---
    "name", "names", "call", "calls", "calling", "called", "back", "later",
    "send", "sending", "sounds", "sound", "good", "fine", "right", "now",
    "driving", "drive", "today", "tomorrow", "yesterday", "morning",
    "evening", "afternoon", "night", "moment", "interested", "interest",
    "details", "detail", "speak", "speaking", "spoke", "someone", "somebody",
    "think", "say", "said", "hear", "heard", "robot", "number", "numbers",
    "mean", "meant", "actually", "maybe", "sure", "alright", "great", "nice",
    "perfect", "understand", "understood", "help", "need", "looking", "look",
    "going", "got", "let", "make", "made", "take", "come", "coming", "see",
    "little", "bit", "thing", "things", "way", "time", "times",
    # Spelled-out digits: a caller reading out a phone number must never look
    # like a knowledge-base query. Real quantity questions use the noun
    # ("how many parks") or a numeral ("190 acres"), both of which survive.
    "zero", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    """Lowercase word tokens, stopwords dropped, numbers KEPT.

    Numbers stay because half the knowledge base is numbers - 1882 plots, 190
    acres, 650 street lights, DTCP 100/2023 - and a caller quoting one back is
    asking the single most answerable kind of question there is.
    """
    if not text:
        return []
    low = str(text).lower().replace("-", " ")
    return [t for t in _TOKEN_RE.findall(low)
            if t not in STOPWORDS and len(t) > 1]


class _Bm25:
    """Okapi BM25 over a fixed corpus."""

    def __init__(self, docs):
        self.n = len(docs)
        self.tokens = [tokenize(d) for d in docs]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0

        self.freqs = []                  # per doc: {term: count}
        doc_count = {}                   # term -> how many docs contain it
        for toks in self.tokens:
            f = {}
            for t in toks:
                f[t] = f.get(t, 0) + 1
            self.freqs.append(f)
            for t in f:
                doc_count[t] = doc_count.get(t, 0) + 1

        # Standard BM25 IDF with the +1 that keeps it positive for a term that
        # appears in more than half the corpus.
        self.idf = {
            t: math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))
            for t, c in doc_count.items()
        }

    def scores(self, query_tokens):
        """Raw BM25 score per document index."""
        out = [0.0] * self.n
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue                 # not in the corpus: contributes nothing
            for i, f in enumerate(self.freqs):
                tf = f.get(t)
                if not tf:
                    continue
                norm = 1.0 - B + B * (self.lengths[i] / self.avg_len
                                      if self.avg_len else 0.0)
                out[i] += idf * (tf * (K1 + 1.0)) / (tf + K1 * norm)
        return out


# ------------------------------------------------------------------ the index

_index = None
_metas = None
_lock = threading.Lock()
_failed = False

# A hit must clear this to count at all. Calibrated in tests/test_lexical.py
# against BOTH directions: every caller phrasing off the real call logs has to
# pass it, and every noise phrase - "hello sir", "okay thank you", "zzz qqq" -
# has to fail it. Like the vector floor, it can only be wrong one way at a
# time, so it is pinned by tests on both sides rather than by taste.
MIN_SCORE = 2.0

# ...and the query has to contain at least one reasonably rare word. This is
# the guard that a score floor alone cannot give: a long enough sentence of
# common words can accumulate past any fixed score, which is exactly how a
# rambling greeting would drag an entry in.
MIN_IDF = 1.0


def _build():
    global _index, _metas, _failed
    with _lock:
        if _index is not None or _failed:
            return _index
        try:
            # Imported HERE, not at module scope: rag.ingest imports
            # rag.retriever, rag.retriever imports this module, and a top-level
            # import would close that ring. By first query everything is loaded.
            from rag import ingest as _ingest

            entries, project = _ingest.load_payload(_ingest.DEFAULT_PATH)
            docs, ids, metadatas = _ingest.build_documents(entries, project)
            _index = _Bm25(docs)
            _metas = [dict(m, _id=i, _document=d)
                      for m, i, d in zip(metadatas, ids, docs)]
            log.info("Lexical index ready: %d documents, %d unique terms",
                     _index.n, len(_index.idf))
        except Exception as e:
            # A broken lexical arm must never take the call down. The vector
            # arm alone is exactly the behaviour this project had before.
            _failed = True
            log.error("Could not build the lexical index (%s) - falling back to "
                      "vector-only retrieval", e)
        return _index


def ready() -> bool:
    return _build() is not None


def search(query: str, n_results: int = 20):
    """Documents matching `query` lexically, best first.

    Returns [] when the turn has nothing worth searching for - which is the
    same contract the vector arm has, and means the not-found line stays the
    honest answer rather than becoming unreachable.
    """
    index = _build()
    if index is None:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    # THE NOISE GATE. "hello sir" and "okay thank you" tokenize to nothing or
    # to words that are in almost every document; without this a long enough
    # pleasantry accumulates score and pulls a knowledge entry into a greeting.
    # That regression is what tests/test_lexical.py exists to prevent.
    if not any(index.idf.get(t, 0.0) >= MIN_IDF for t in tokens):
        return []

    scored = index.scores(tokens)
    hits = [(s, i) for i, s in enumerate(scored) if s >= MIN_SCORE]
    if not hits:
        return []

    hits.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for score, i in hits[:n_results]:
        meta = _metas[i]
        out.append({
            "id": meta["_id"],
            "document": meta["_document"],
            "meta": {k: v for k, v in meta.items()
                     if k not in ("_id", "_document")},
            "score": score,
            # Not a real distance. Carried only so a lexical hit can travel
            # through the same plumbing as a vector hit; fusion ranks on
            # position, never on this number.
            "distance": None,
        })
    return out


def reset():
    """Drop the index. Tests rebuild against a stub KB; nothing else needs it."""
    global _index, _metas, _failed
    with _lock:
        _index, _metas, _failed = None, None, False
