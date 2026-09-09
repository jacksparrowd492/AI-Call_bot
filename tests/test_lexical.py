"""Tests for rag/lexical.py - the BM25 arm and the fusion that ranks it.

    python tests/test_lexical.py

These run WITHOUT the embedder, which is the point: the lexical arm is pure
arithmetic over the shipped knowledge base, so unlike the vector arm it can be
verified on any machine, offline, before a call is ever placed.

The gate is pinned from BOTH sides, exactly like the vector floor in
tools/ask_kb.py --check, and for the same reason: a floor can only ever be
wrong one way at a time.

  * every caller phrasing off the real call logs must come back with hits.
    An empty one is the bot telling a caller it has no information about
    something it is holding.
  * every noise phrase must come back EMPTY. A lexical arm loose enough to
    match "my name is Karthi" is one that hands the model a knowledge entry to
    answer a name with.

The second list is the one that matters. Measured while building this, the
overlap was severe before the filler lexicon went in: "my name is Karthi"
scored 7.99 and "that sounds good" 5.59, against 2.58 for the genuine caller
turn "amenities". A score floor alone cannot separate those, and neither can
rarity - "name" appears in ONE of the 391 documents and therefore has the
highest idf in the whole corpus.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import lexical                                       # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


# Off the real call logs, plus the turns this whole change exists for.
CALLER_PHRASINGS = [
    "transport service", "how is the transport facility there",
    "transport facilities", "what about transport", "how is the connectivity",
    "how is the water supply", "water supply", "infrastructure and water supply",
    "is there a bus stop nearby", "the amenities present in karthipuram",
    "amenities", "water", "what about security", "schools", "parks",
    "where is the project located", "what is the contact number",
    "how many acres", "what is the rera number", "what is the dtcp number",
    # The reported failures.
    "sports", "basketball", "tennis", "games", "play area",
    "what sports facilities are there", "karthipuram", "education",
    "hospitals", "shopping", "electricity", "internet", "gym", "cricket",
    "swimming pool", "street lights",
]

# Must return NOTHING. The first block is the existing noise list from
# tools/ask_kb.py; the second is non-questions that DO contain words the
# brochure happens to use, which is the harder half and the reason the filler
# lexicon exists at all.
NOISE = [
    "hello", "hello sir", "hold on a second", "okay thank you", "zzz qqq",
    "yes", "okay", "can you hear me", "sorry what", "thank you so much",
    "um well i mean",

    "my name is karthi",            # "karthi" is an ordinary first name
    "i will call you back later",
    "can you send me the details",
    "one moment please",
    "i am driving right now",       # "right"/"now" are in the disclaimer
    "my number is nine eight seven six",
    "yeah okay fine",               # "fine" is in "fine dining"
    "is that right",
    "no thank you not interested",
    "call me tomorrow evening",     # "evening" is in the office hours
    "i want to speak to someone",
    "hmm let me think",
    "actually hold on",
    "what did you say",
    "sorry i could not hear",
    "are you a robot",
    "who am i speaking to",
    "that sounds good",
]


def main():
    print("=== 1. the index builds from the shipped knowledge base ===")
    lexical.reset()
    check("the lexical index builds", lexical.ready())
    idx = lexical._index
    check("it indexed the whole corpus", idx.n > 300, "%d documents" % idx.n)
    check("it has a real vocabulary", len(idx.idf) > 500,
          "%d unique terms" % len(idx.idf))

    print("\n=== 2. tokenisation ===")
    check("stopwords are dropped", lexical.tokenize("what is the water supply")
          == ["water", "supply"], lexical.tokenize("what is the water supply"))
    check("numerals survive - half the KB is numbers",
          "190" in lexical.tokenize("how many acres is the 190 acre project"))
    check("spelled-out digits do not - a phone number is not a query",
          lexical.tokenize("nine eight seven six five") == [])
    check("a pure greeting tokenizes to nothing",
          lexical.tokenize("hello sir how are you") == [])
    check("hyphens split", "mart" in lexical.tokenize("is there a d-mart nearby"))

    print("\n=== 3. every caller phrasing returns something ===")
    empty = []
    for q in CALLER_PHRASINGS:
        if not lexical.search(q):
            empty.append(q)
    check("all %d caller phrasings match" % len(CALLER_PHRASINGS),
          not empty, "empty: %s" % empty if empty else "")

    print("\n=== 4. every noise phrase returns NOTHING ===")
    leaked = []
    for q in NOISE:
        hits = lexical.search(q)
        if hits:
            leaked.append((q, round(hits[0]["score"], 2),
                           (hits[0].get("meta") or {}).get("entry_id")))
    check("all %d noise phrases blocked" % len(NOISE),
          not leaked, "leaked: %s" % leaked if leaked else "")

    print("\n=== 5. the reported failure: 'sports' now reaches the sports entry ===")
    hits = lexical.search("sports")
    entries = [(h.get("meta") or {}).get("entry_id") for h in hits[:5]]
    check("'sports' returns hits", bool(hits), "%d hits" % len(hits))
    check("...and they are the sports/parks entries",
          any(e in ("sports_recreation", "parks", "amenities_full")
              for e in entries), str(entries[:3]))
    check("'cricket' still works - the case that always did",
          any((h.get("meta") or {}).get("entry_id") in ("parks", "sports_recreation")
              for h in lexical.search("cricket")))
    for word in ("basketball", "tennis", "gym", "games", "swimming"):
        check("'%s' reaches a sports or club entry" % word,
              any((h.get("meta") or {}).get("entry_id")
                  in ("sports_recreation", "parks", "recreational_club",
                      "amenities_full")
                  for h in lexical.search(word)))

    print("\n=== 6. the brochure content that had never been mined ===")
    # `want` is the set of entries any of which is a correct place to land.
    # "Prozone" is named by BOTH nearby_landmarks and shopping_overview, and
    # the shorter one winning is right, not a miss - the assertion was too
    # strict, not the retrieval.
    for word, want in (("prozone", {"nearby_landmarks", "shopping_overview"}),
                       ("railway", {"nearby_landmarks"}),
                       ("athikadavu", {"water_supply"}),
                       ("pillur", {"water_supply"}),
                       ("kmch", {"hospitals", "healthcare_nearby"}),
                       ("isha", {"nearby_landmarks"}),
                       ("basketball", {"sports_recreation"})):
        hits = lexical.search(word)
        got = [(h.get("meta") or {}).get("entry_id") for h in hits[:3]]
        ok = bool(hits) and bool(want & set(got))
        check("'%s' is findable" % word, ok, str(got[:2]))

    print("\n=== 7. fusion ranks both arms without either metric leaking ===")
    from rag import retriever
    vec = [{"id": "v1", "document": "d1", "meta": {"entry_id": "a"}, "distance": 1.10},
           {"id": "v2", "document": "d2", "meta": {"entry_id": "b"}, "distance": 1.40}]
    lex = [{"id": "v2", "document": "d2", "meta": {"entry_id": "b"}, "distance": None,
            "score": 9.0},
           {"id": "l3", "document": "d3", "meta": {"entry_id": "c"}, "distance": None,
            "score": 4.0}]
    fused = retriever._fuse(vec, lex)
    ids = [h["id"] for h in fused]
    check("every document from both arms survives", set(ids) == {"v1", "v2", "l3"}, str(ids))
    check("a document BOTH arms found ranks first", ids[0] == "v2", str(ids))
    check("the vector copy is kept when both arms match it",
          next(h for h in fused if h["id"] == "v2")["distance"] == 1.40)
    check("a lexical-only hit keeps distance=None rather than a fake number",
          next(h for h in fused if h["id"] == "l3")["distance"] is None)
    check("fusion of two empty arms is empty", retriever._fuse([], []) == [])
    check("fusion never invents a hit neither arm returned",
          len(retriever._fuse(vec, [])) == 2)

    print("\n" + "=" * 60)
    print("PASSED %-4d FAILED %d" % (len(PASS), len(FAIL)))
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
