"""Calibrate and regression-test the router. Run this after ANY KB change.

    python -m tools.eval_router            # run the suite
    python -m tools.eval_router --tune     # sweep thresholds and suggest values
    python -m tools.eval_router "your question here"    # inspect one query

Why this exists: the routing thresholds are cosine distances, and the right
values depend on the KB you actually ingested. Guessing them means the bot
either escalates questions it could answer, or confidently answers questions it
cannot. This measures both.
"""
import argparse
import asyncio
import json
import sys

from config import settings
from rag import retriever
from router import Route, Router

# (utterance, expected route, note)
CASES = [
    # --- smalltalk: must never touch the LLM or the vector store -----------
    ("hello", Route.SMALLTALK, "greeting"),
    ("hi", Route.SMALLTALK, "greeting"),
    ("good morning", Route.SMALLTALK, "greeting"),
    ("thank you", Route.SMALLTALK, "thanks"),
    ("ok thanks", Route.SMALLTALK, "thanks"),
    ("bye", Route.SMALLTALK, "farewell"),
    ("can you hear me", Route.SMALLTALK, "audio check"),
    ("who are you", Route.SMALLTALK, "identity"),
    ("how are you", Route.SMALLTALK, "wellbeing"),
    ("sorry", Route.SMALLTALK, "repeat"),
    ("one minute", Route.SMALLTALK, "wait"),
    ("okay", Route.SMALLTALK, "acknowledge"),

    # --- knowledge: the brochure answers these ----------------------------
    ("what is karthipuram", Route.KNOWLEDGE, "overview"),
    ("how many acres is the project", Route.KNOWLEDGE, "overview"),
    ("where is the site located", Route.KNOWLEDGE, "location"),
    ("is it near neelambur", Route.KNOWLEDGE, "location"),
    ("how wide are the roads", Route.KNOWLEDGE, "roads"),
    ("do you have eighty feet roads", Route.KNOWLEDGE, "roads"),
    ("how many parks are there", Route.KNOWLEDGE, "parks"),
    ("is there a swimming pool", Route.KNOWLEDGE, "club"),
    ("what about water supply", Route.KNOWLEDGE, "water"),
    ("where does the water come from", Route.KNOWLEDGE, "water"),
    ("do i need my own borewell", Route.KNOWLEDGE, "water"),
    ("is the electricity underground", Route.KNOWLEDGE, "electricity"),
    ("is there internet connectivity", Route.KNOWLEDGE, "internet"),
    ("is there a sewage treatment plant", Route.KNOWLEDGE, "sewage"),
    ("will there be waterlogging", Route.KNOWLEDGE, "storm water"),
    ("is the project rera approved", Route.KNOWLEDGE, "approvals"),
    ("what is the dtcp number", Route.KNOWLEDGE, "approvals"),
    ("who is the builder", Route.KNOWLEDGE, "promoter"),
    ("what other projects have you done", Route.KNOWLEDGE, "track record"),
    ("which schools are nearby", Route.KNOWLEDGE, "nearby schools"),
    ("are there hospitals close by", Route.KNOWLEDGE, "hospitals"),
    ("is the airport nearby", Route.KNOWLEDGE, "landmarks"),
    ("how many trees are there", Route.KNOWLEDGE, "greenery"),
    ("is there cctv security", Route.KNOWLEDGE, "security"),
    ("tell me about the media tower", Route.KNOWLEDGE, "media tower"),
    ("how many plots are there", Route.KNOWLEDGE, "inventory"),
    ("why should i buy a plot instead of a flat", Route.KNOWLEDGE, "why plots"),
    ("what is your contact number", Route.KNOWLEDGE, "contact"),
    ("is there a school inside", Route.KNOWLEDGE, "school (proposed)"),
    ("is there a shopping mall", Route.KNOWLEDGE, "mall (proposed)"),

    # --- escalation: the brochure genuinely cannot answer ------------------
    ("what is the price", Route.ESCALATE, "pricing"),
    ("how much does a plot cost", Route.ESCALATE, "pricing"),
    ("what is the rate per square foot", Route.ESCALATE, "pricing"),
    ("what are the registration charges", Route.ESCALATE, "pricing"),
    ("what plot sizes do you have", Route.ESCALATE, "availability"),
    ("which plots are available right now", Route.ESCALATE, "availability"),
    ("do you have a corner plot", Route.ESCALATE, "availability"),
    ("how do i book a plot", Route.ESCALATE, "booking"),
    ("what is the booking amount", Route.ESCALATE, "booking"),
    ("can i get a home loan", Route.ESCALATE, "booking"),
    ("is there any discount", Route.ESCALATE, "offers"),
    ("when is the possession", Route.ESCALATE, "possession"),
    ("when will the mall be ready", Route.ESCALATE, "possession"),
    ("can i schedule a site visit", Route.ESCALATE, "site visit"),
    ("can you send me the layout", Route.ESCALATE, "site visit"),
    ("what is the maintenance charge", Route.ESCALATE, "maintenance"),
    ("what is the survey number", Route.ESCALATE, "legal detail"),
    ("how many floors can i build", Route.ESCALATE, "legal detail"),

    # --- handoff -----------------------------------------------------------
    ("i want to speak to a human", Route.HANDOFF, "human"),
    ("can you transfer me to someone", Route.HANDOFF, "human"),
    ("connect me to an agent", Route.HANDOFF, "human"),

    # --- genuinely out of scope -> must escalate, never hallucinate --------
    ("what is the weather in chennai", Route.ESCALATE, "off-topic"),
    ("do you sell cars", Route.ESCALATE, "off-topic"),
    ("what is the gst rate on property", Route.ESCALATE, "not in brochure"),
    # Answerable, and the honest answer has two halves: no station inside the
    # township, but metro projects ARE a listed Neelambur advantage.
    ("is there a metro station inside the township", Route.KNOWLEDGE, "transport"),
    ("is there public transport nearby", Route.KNOWLEDGE, "transport"),
    # Genuinely not in the brochure - the map shows stations but no distances.
    ("how far is the railway station in kilometres", Route.ESCALATE, "not in brochure"),
    ("what is the exact distance to the airport", Route.ESCALATE, "not in brochure"),
]


def _load_router():
    with open(settings.kb_path, encoding="utf-8") as f:
        kb = json.load(f)
    return Router(kb.get("smalltalk", {}))


async def inspect(query):
    router = _load_router()
    d = await router.decide(query)
    print("\nquery   : %s" % query)
    print("route   : %s" % d.route.name)
    print("intent  : %s" % d.intent)
    print("distance: %.4f  (strong<=%.2f  weak<=%.2f  esc<=%.2f)"
          % (d.distance, settings.route_strong_threshold,
             settings.route_weak_threshold, settings.route_escalation_threshold))
    print("confident: %s" % d.confident)
    if d.reply:
        print("reply   : %s" % d.reply)
    if d.context:
        print("context :\n%s" % d.context)

    if d.route not in (Route.SMALLTALK,):
        emb = await retriever.aembed(query)
        print("\ntop 5 raw hits:")
        for h in retriever.search(emb, 5)[:5]:
            m = h["meta"]
            print("  %.4f  %-11s %-24s %s"
                  % (h["distance"], m.get("kind"), m.get("entry_id"),
                     h["document"][:56]))


async def run_suite(verbose=False):
    router = _load_router()
    rows, fails = [], []

    for text, expected, note in CASES:
        d = await router.decide(text)
        ok = d.route == expected
        # HANDOFF is a strict subtype of ESCALATE; treat it as acceptable when
        # ESCALATE was expected, but not the reverse.
        if not ok and expected == Route.ESCALATE and d.route == Route.HANDOFF:
            ok = True
        rows.append((ok, text, expected, d, note))
        if not ok:
            fails.append((text, expected, d, note))
        if verbose or not ok:
            print("%s %-46s want=%-9s got=%-9s d=%.3f  %s"
                  % ("PASS" if ok else "FAIL", text[:46], expected.name,
                     d.route.name, d.distance, d.intent))

    total = len(rows)
    passed = sum(1 for r in rows if r[0])
    print("\n%d/%d passed (%.0f%%)" % (passed, total, 100.0 * passed / total))

    # distance distribution helps pick thresholds
    kb_d = [r[3].distance for r in rows if r[2] == Route.KNOWLEDGE and r[3].distance < 2]
    esc_d = [r[3].distance for r in rows
             if r[2] in (Route.ESCALATE, Route.HANDOFF) and r[3].distance < 2]
    if kb_d:
        print("knowledge-case distances : min=%.3f median=%.3f max=%.3f"
              % (min(kb_d), sorted(kb_d)[len(kb_d) // 2], max(kb_d)))
    if esc_d:
        print("escalation-case distances: min=%.3f median=%.3f max=%.3f"
              % (min(esc_d), sorted(esc_d)[len(esc_d) // 2], max(esc_d)))

    if fails:
        print("\n%d failing case(s) above. If a KNOWLEDGE case escalated, either "
              "raise ROUTE_WEAK_THRESHOLD or add that phrasing to the KB entry. "
              "If an ESCALATE case was answered, lower ROUTE_WEAK_THRESHOLD or add "
              "the phrasing to the escalation topic." % len(fails))
    return 0 if not fails else 1


async def tune():
    """Sweep the weak threshold and report accuracy, so the value is measured
    rather than guessed."""
    router = _load_router()
    precomputed = []
    for text, expected, note in CASES:
        d = await router.decide(text)
        precomputed.append((text, expected, d))

    print("weak_threshold  accuracy   knowledge_recall  escalation_precision")
    best = (0, None)
    for t in [x / 100 for x in range(40, 121, 5)]:
        ok = kb_ok = kb_n = esc_ok = esc_n = 0
        for text, expected, d in precomputed:
            if d.route == Route.SMALLTALK:
                got = Route.SMALLTALK
            elif d.intent and d.intent.startswith("esc_"):
                got = Route.HANDOFF if d.intent == "esc_human" else Route.ESCALATE
            else:
                got = Route.KNOWLEDGE if d.distance <= t else Route.ESCALATE
            hit = (got == expected) or (expected == Route.ESCALATE and got == Route.HANDOFF)
            ok += hit
            if expected == Route.KNOWLEDGE:
                kb_n += 1
                kb_ok += hit
            if expected in (Route.ESCALATE, Route.HANDOFF):
                esc_n += 1
                esc_ok += hit
        acc = ok / len(precomputed)
        print("  %.2f          %.0f%%        %.0f%%              %.0f%%"
              % (t, 100 * acc, 100 * kb_ok / max(kb_n, 1), 100 * esc_ok / max(esc_n, 1)))
        if acc > best[0]:
            best = (acc, t)
    print("\nBest ROUTE_WEAK_THRESHOLD = %.2f  (accuracy %.0f%%)"
          % (best[1], 100 * best[0]))
    print("Set it in .env as ROUTE_WEAK_THRESHOLD=%.2f" % best[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="inspect a single query")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.query:
        asyncio.run(inspect(" ".join(args.query)))
    elif args.tune:
        asyncio.run(tune())
    else:
        sys.exit(asyncio.run(run_suite(args.verbose)))
