"""Assert the knowledge base covers every fact in the 24-page brochure.

This is a content regression test, not a style check. Each string below is a
fact that appears in the brochure; if a KB edit ever drops one, this fails.
No embedding model or vector store is needed, so it runs anywhere.

    python -m tools.check_kb
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(HERE, "data", "knowledge_base.json")

# (brochure page, needle) - needle is matched case-insensitively against the
# whole KB text, with digits/spacing normalised.
BROCHURE_FACTS = [
    (2, "unnamalai group"), (2, "karthikeyan"),
    (3, "190"), (3, "manchester of south india"), (3, "4 k"),
    (3, "jewellery"), (3, "textile"), (3, "footwear"),
    (4, "1815"), (4, "1892"), (4, "1921"), (4, "1974"), (4, "race course"),
    (4, "town hall"), (4, "rs puram"), (4, "gandhipuram"),
    (5, "1989"), (5, "1993"), (5, "1996"), (5, "1997"), (5, "1999"),
    (5, "2002"), (5, "2009"), (5, "2011"), (5, "2019"), (5, "2020"),
    (5, "850"), (5, "kovilpatti"), (5, "dharapuram"), (5, "tirupur"),
    (5, "dindugul"), (5, "sulur"), (5, "gobi"), (5, "sathy"),
    (5, "unnamalai institute of technology"),
    (5, "unnamalai college of arts"),
    (5, "grow beyond limits"),
    (6, "full ownership"), (6, "expand anytime"),
    (6, "airport expansion"), (6, "logistics hub"), (6, "metro"),
    (6, "price appreciation"), (6, "economic momentum"),
    (7, "1,882"), (7, "213"), (7, "1,200"), (7, "15 "), (7, "10 park"),
    (7, "9-acre"), (7, "3 types of water"),
    (8, "school"), (8, "bus travel"),
    (9, "fine dining"), (9, "luxury retail"), (9, "entertainment"),
    (11, "central lawn"), (11, "walking track"), (11, "reflexology"),
    (11, "outdoor gym"), (11, "gazebo"), (11, "amphitheatre"),
    (11, "cricket net"), (11, "mega sculpture"), (11, "parents pavilion"),
    (11, "plaza"), (11, "fountain"), (11, "adult sports"),
    (12, "swimming pool"), (12, "fitness studio"), (12, "indoor games"),
    (12, "signature restaurant"), (12, "meeting space"),
    (13, "52"), (13, "led"), (13, "boulevard"), (13, "cultural event"),
    (15, "shopping street"),
    (16, "bhavani"), (16, "borewell"), (16, "athikadavu"),
    (16, "sewage line"), (16, "fibernet"), (16, "electricity cable"),
    (17, "concrete duct"), (17, "fiber optic"), (17, "odour-free"),
    (17, "mosquito-free"), (17, "recharge pit"), (17, "groundwater"),
    (17, "sewage treatment plant"),
    (18, "bhavani river"), (18, "pillur dam"), (18, "centralised bore"),
    (19, "nilgiri"), (19, "aanaikatti"), (19, "altitude"), (19, "cool"),
    (21, "650"), (21, "compound wall"), (21, "cctv"),
    (21, "tar road"), (21, "80"), (21, "60"), (21, "40"),
    (21, "proposed"),
    (22, "yellow train"), (22, "grd-cpf"), (22, "reeds world"),
    (22, "indian public school"), (22, "chandramari"), (22, "sri chaitanya"),
    (22, "kathir college"), (22, "kathir engineering"), (22, "adithya"),
    (22, "psg itech"), (22, "grd college"),
    (22, "royal care"), (22, "kmch"),
    (22, "broadway"), (22, "d-mart"), (22, "kpr tech park"),
    (22, "le-meridien"), (22, "gokulam park"), (22, "merlis"),
    (22, "audi"), (22, "mg "), (22, "haribhavanam"), (22, "anandas"),
    (22, "kannappa"), (22, "thalappakatti"), (22, "dominos"), (22, "kfc"),
    (22, "mariamman"), (22, "kaanaveda"), (22, "varadaraja"),
    (22, "it park"), (22, "10 km"),
    (24, "583/2b1"), (24, "poongothai"), (24, "goldwins"), (24, "641014"),
    (24, "9418 646464"), (24, "77780 77790"),
    (24, "info@karthipuram.com"), (24, "unnamalaipromoters.com"),
    (24, "100/2023"), (24, "tn/11/layout/3352/2025"), (24, "rera.tn.gov.in"),
    (24, "avinashi road"), (24, "neelambur"), (24, "kalapatti"), (24, "arasur"),
    (24, "indicative"), (24, "agreement for sale"),
]

# Topics the brochure does NOT cover; each must exist as an escalation.
REQUIRED_ESCALATIONS = [
    "esc_pricing", "esc_availability", "esc_booking", "esc_offers",
    "esc_possession", "esc_site_visit", "esc_maintenance",
    "esc_legal_detail", "esc_human",
]

# Amenities the brochure explicitly marks "Proposed". Selling these as built
# would be a misrepresentation, so the KB entry must carry status=proposed.
MUST_BE_PROPOSED = ["school_inside", "shopping_mall", "recreational_club"]

# Phrases that must NEVER appear in a spoken answer - they reveal the bot is
# reading a document, which the system prompt forbids.
FORBIDDEN_IN_ANSWERS = [
    "the brochure", "brochure says", "brochure states", "according to the",
    "knowledge base", "as an ai", "i am an ai", "the document",
    "source_pages", "not specified in",
]


def norm(s):
    return re.sub(r"\s+", " ", s.lower())


def main():
    if not os.path.exists(KB_PATH):
        print("knowledge_base.json not found - run: python build_kb.py")
        return 1
    with open(KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    entries = kb.get("entries", [])
    escalations = kb.get("escalations", [])
    haystack = norm(json.dumps(kb, ensure_ascii=False))

    failures = []

    # 1. brochure coverage
    missing = [(p, n) for p, n in BROCHURE_FACTS if norm(n) not in haystack]
    for p, n in missing:
        failures.append("brochure p%-2s fact not in KB: %r" % (p, n))

    # 2. escalation coverage
    have = {e["id"] for e in escalations}
    for eid in REQUIRED_ESCALATIONS:
        if eid not in have:
            failures.append("missing escalation topic: %s" % eid)

    # 3. proposed-status honesty
    by_id = {e["id"]: e for e in entries}
    for eid in MUST_BE_PROPOSED:
        e = by_id.get(eid)
        if not e:
            failures.append("missing entry: %s" % eid)
        elif e.get("status") != "proposed":
            failures.append("%s must have status=proposed (brochure says Proposed), "
                            "found %r" % (eid, e.get("status")))
        elif not re.search(r"propos|plan", norm(e.get("answer", ""))):
            failures.append("%s answer does not tell the caller it is planned" % eid)

    # 4. spoken answers must not leak the source
    for e in entries:
        a = norm(e.get("answer", ""))
        for bad in FORBIDDEN_IN_ANSWERS:
            if bad in a:
                failures.append("%s answer contains forbidden phrase %r"
                                % (e["id"], bad))
    for e in escalations:
        a = norm(e.get("reply", ""))
        for bad in FORBIDDEN_IN_ANSWERS:
            if bad in a:
                failures.append("%s reply contains forbidden phrase %r"
                                % (e["id"], bad))

    # 5. every escalation must actually offer a next step
    for e in escalations:
        r = norm(e.get("reply", ""))
        if not re.search(r"call|connect|team|expert|arrange|visit", r):
            failures.append("%s reply does not offer a human next step" % e["id"])

    # 6. answers must be speakable in one breath-ish
    for e in entries:
        n = len(e.get("answer", "").split())
        if n > 65:
            failures.append("%s answer is %d words - too long for a phone call"
                            % (e["id"], n))

    print("KB CHECK")
    print("  entries            : %d" % len(entries))
    print("  escalation topics  : %d" % len(escalations))
    print("  question phrasings : %d"
          % (sum(len(e["questions"]) for e in entries)
             + sum(len(e["questions"]) for e in escalations)))
    print("  brochure facts     : %d checked, %d missing"
          % (len(BROCHURE_FACTS), len(missing)))
    print("  proposed entries   : %s"
          % ", ".join(e["id"] for e in entries if e.get("status") == "proposed"))

    if failures:
        print("\n%d FAILURE(S):" % len(failures))
        for f in failures:
            print("  -", f)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
