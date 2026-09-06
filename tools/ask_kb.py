"""Ask the knowledge base a question from the command line, for real.

Unlike tests/test_rag.py - which swaps in a hashed stand-in so it can run
anywhere - this uses the REAL MiniLM embedder and the REAL vector store, which
is the only way to find out what a caller would actually get back.

    python -m tools.ask_kb "how is the transport facility there"
    python -m tools.ask_kb --check        # the phrasings callers really used

--check is the one to run after `python -m rag.ingest`. It checks BOTH
directions, because a distance floor can only ever be wrong one way at a time:

  * every phrasing in CALLER_PHRASINGS - taken off real call logs - must come
    back with context. An empty one is the bot telling that caller it has no
    information about something it knows.
  * every phrase in NOISE must come back EMPTY. These are the turns that are
    not questions at all, and a floor loose enough to let them through is a
    floor that hands the model a random knowledge entry to answer a greeting
    from. That happened on 2026-09-04.
"""
import sys

from rag import retriever

# Taken from leads.db transcripts and the call log. Short, ungrammatical and
# often just a noun - which is how people talk on the phone, and what the
# question-shaped knowledge base used to miss.
CALLER_PHRASINGS = [
    "transport service",
    "how is the transport facility there",
    # 2026-09-06: answered out of the metro entry, so the caller was told what
    # the township does NOT have instead of how well connected it is.
    "transport facilities",
    "what about transport",
    "how is the connectivity",
    # 2026-09-06: asked as one turn with infrastructure, and the water half
    # was never searched for at all.
    "how is the water supply",
    "water supply",
    "infrastructure and water supply",
    "tell me about the transport and the water supply",
    "is there a bus stop nearby",
    "the amenities present in karthipuram",
    "amenities",
    "water",
    "what about security",
    "schools",
    "parks",
    "where is the project located",
    "what is the contact number",
    "how many acres",
    "what is the rera number",
]

# Must return NOTHING. Nearest-neighbour search always returns something, so
# these are the guard on the other side of the floor: noise, acknowledgements
# and dead air, none of which the knowledge base has any business answering.
# (On a live call most never reach the retriever - bridge.py answers greetings
# from the smalltalk guard - but the floor has to hold on its own.)
NOISE = [
    "hello",
    "hello sir",
    "hold on a second",
    "okay thank you",
    "zzz qqq",
]


def why_empty(question: str):
    """Show how close the nearest documents were, and to what.

    "No context" on its own is not actionable: it can mean the knowledge base
    does not cover the question, or that it covers it and the distance floor
    cut it off by a hundredth. Those need opposite fixes, so print the numbers.
    """
    hits = retriever.search(retriever.embed_query(question), n_results=3)
    floor = retriever.settings.rag_max_distance
    if not hits:
        print("     nothing in the collection at all - is it ingested?")
        return
    print("     RAG_MAX_DISTANCE=%.2f. Nearest documents:" % floor)
    for h in hits:
        meta = h.get("meta") or {}
        first = (h.get("document") or "").splitlines()
        head = " / ".join(first[:2])
        print("       d=%.3f  %-26s %s" % (h["distance"],
                                           meta.get("entry_id", "?"), head[:70]))
    gap = hits[0]["distance"] - floor
    if gap <= 0.25:
        print("     -> the best match is only %.3f past the floor: this is a"
              " threshold problem, not a coverage one." % gap)
    else:
        print("     -> the best match is %.3f past the floor: nothing in the"
              " knowledge base is really about this." % gap)


def ask(question: str) -> str:
    ctx = retriever.retrieve(question)
    print("\n> %s" % question)
    if not ctx:
        print("  NO CONTEXT - the bot would say it does not have this and "
              "offer a callback")
        why_empty(question)
        return ctx
    for line in ctx.splitlines():
        print("  " + line)
    return ctx


def check() -> int:
    print("Collection %r at %s: %d documents"
          % (retriever.COLLECTION, retriever.DB_PATH, retriever.count()))

    empty = [q for q in CALLER_PHRASINGS if not ask(q)]

    print("\n--- and these must come back EMPTY ---")
    leaked = []
    for q in NOISE:
        ctx = retriever.retrieve(q)
        if ctx:
            leaked.append(q)
            print("\n> %s\n  LEAKED - the model would be handed:\n    %s"
                  % (q, ctx.splitlines()[0][:70]))
        else:
            print("> %-20s empty, as it should be" % q)

    print("\n" + "=" * 60)
    if empty:
        print("%d of %d caller phrasings get NO context:"
              % (len(empty), len(CALLER_PHRASINGS)))
        for q in empty:
            print("  - %s" % q)
        print("If the knowledge base covers these, the vector store is stale."
              "  Rebuild it:  python -m rag.ingest")
    if leaked:
        print("%d of %d noise phrases DID get context:" % (len(leaked),
                                                           len(NOISE)))
        for q in leaked:
            print("  - %s" % q)
        print("The floor is too loose. Lower RAG_MAX_DISTANCE, or narrow "
              "SHORT_QUERY_MARGIN in rag/retriever.py.")
    if empty or leaked:
        return 1
    print("All %d caller phrasings return context, and all %d noise phrases "
          "return none." % (len(CALLER_PHRASINGS), len(NOISE)))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a]
    if not args or args[0] == "--check":
        sys.exit(check())
    ask(" ".join(args))
