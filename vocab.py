"""Repair domain proper nouns that speech recognition mangles.

Across two live calls Deepgram heard "Karthipuram" as Cartigram, Cartivaram,
Kartivaram, Kartipuram and Artipuram - five attempts, zero correct. Every bad
routing decision in those calls traces back to that, not to the router.

Two defences, and you want both:

  1. Tell Deepgram the words up front (keyword boosting + en-IN) - see stt.py.
     This is the real fix and it happens before a transcript exists.
  2. Repair what still comes through wrong - this module. Boosting raises the
     odds; it does not make an Indian place name a solved problem for an
     English model.

The matching is deliberately conservative. Turning a caller's real word into a
project term they never said is worse than leaving it alone, so a token must
clear BOTH a raw similarity bar and a consonant-skeleton bar before it is
replaced, and short or common English words are never touched.
"""
import logging
import re
from difflib import SequenceMatcher

from config import settings

log = logging.getLogger("jarvis.vocab")

# Canonical spellings the knowledge base and the caller both care about.
LEXICON = [
    "Karthipuram", "Unnamalai", "Neelambur", "Coimbatore", "Avinashi",
    "Kalapatti", "Arasur", "Bhavani", "Athikadavu", "Pillur", "Kathir",
    "Nilgiri", "Aanaikatti", "Serayampalayam", "Vellanaipatti", "Goldwins",
    "Poongothai", "Karthikeyan", "Kovilpatti", "Tirupur", "Dharapuram",
    "Dindugul", "Saravanampatti", "Singanallur", "Gandhipuram", "Sulur",
    "DTCP", "RERA", "Fibernet", "Kokoro",
]

# Multi-word phrases worth normalising as a unit.
PHRASES = {
    "karthi puram": "Karthipuram",
    "karthi param": "Karthipuram",
    "carti puram": "Karthipuram",
    "neelam bur": "Neelambur",
    "athi kadavu": "Athikadavu",
    "un namalai": "Unnamalai",
}

# Never rewrite these, however close they look.
PROTECTED = {
    "car", "cars", "card", "cart", "carts", "cost", "costly", "program",
    "gram", "grams", "art", "arts", "part", "party", "puram", "para",
    "karma", "cartoon", "carbon", "capital", "corner", "current", "certain",
    "curtain", "captain", "caption", "coating", "cutting", "rating",
    "about", "around", "amount", "apartment", "argument", "important",
}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Consonant equivalence classes: these are the substitutions an English ASR
# actually makes on Tamil place names.
_SKELETON_MAP = str.maketrans({
    "c": "k", "q": "k", "g": "k", "x": "k",
    "v": "w", "f": "w", "p": "b",
    "d": "t", "z": "s",
})
_VOWELS = re.compile(r"[aeiouyh]+")


def _skeleton(word: str) -> str:
    """Collapse a word to a rough consonant signature.

    Karthipuram -> krtbrm ; Cartivaram -> krtwrm ; Artipuram -> rtbrm
    Close enough to link them, different enough not to swallow 'program'.
    """
    w = word.lower().translate(_SKELETON_MAP)
    w = _VOWELS.sub("", w)
    return re.sub(r"(.)\1+", r"\1", w)


_LEX_SKELETONS = [(term, _skeleton(term)) for term in LEXICON]


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def best_match(token: str, raw_min=0.62, skel_min=0.72):
    """Return the canonical term for `token`, or None.

    Requires agreement from two independent measures - raw string similarity
    and consonant skeleton similarity - so a single fluke cannot trigger a
    rewrite.
    """
    low = token.lower()
    if len(low) < 5 or low in PROTECTED:
        return None

    tok_skel = _skeleton(token)
    if len(tok_skel) < 3:
        return None

    best, best_score = None, 0.0
    for term, term_skel in _LEX_SKELETONS:
        if low == term.lower():
            return None                      # already correct
        raw = _similar(token, term)
        skel = _similar(tok_skel, term_skel)
        if raw < raw_min or skel < skel_min:
            continue
        score = (raw + skel * 2) / 3         # weight the skeleton harder
        if score > best_score:
            best, best_score = term, score
    return best


def repair(text: str):
    """Return (repaired_text, [(heard, corrected), ...])."""
    if not text or not text.strip():
        return text, []

    out = text
    fixes = []

    # phrases first - they span the token boundaries the loop below works on
    low = out.lower()
    for phrase, canon in PHRASES.items():
        if phrase in low:
            out = re.sub(re.escape(phrase), canon, out, flags=re.I)
            fixes.append((phrase, canon))
            low = out.lower()

    def _sub(m):
        tok = m.group(0)
        canon = best_match(tok)
        if canon:
            fixes.append((tok, canon))
            return canon
        return tok

    out = _TOKEN.sub(_sub, out)
    if fixes:
        log.info("vocab repair: %s", ", ".join("%s->%s" % f for f in fixes))
    return out, fixes


# Ordinary English words, boosted but NEVER repaired. These are the topics
# callers actually ask about, and the recogniser mangles them exactly like any
# other unexpected word on a noisy phone line: "infrastructure and water
# supply" came back as "Infrastructure and, how many days?" on the 2026-09-06
# call, and the bot then answered half the question and offered a callback for
# a fact it was holding all along. They are kept out of LEXICON on purpose -
# repairing a caller's real word into one of these would be worse than the
# mishearing it fixes.
TOPIC_TERMS = [
    "transport", "transportation", "connectivity", "infrastructure",
    "water", "supply", "borewell", "sewage", "drainage", "electricity",
    "amenities",
]

# Boosting is not free in both directions. The first cut of TOPIC_TERMS also
# carried "security", "facilities", "layout", "plot" and friends - ordinary
# words the model already knows - and the very next test call answered "my
# name is Karthi" with "security". A boosted word is a word the recogniser
# reaches for when it is unsure, and the moment it is unsure most is a name it
# has never heard. Only terms that were actually mangled on a call stay in the
# list above, and names get a boost of their own to compete with them.
#
# NAME_TERMS is for BOOSTING ONLY and is deliberately not in LEXICON: repairing
# a caller's real word into one of these would invent a name they never said.
NAME_TERMS = [
    "Karthi", "Karthik", "Bharath", "Barath", "Vignesh", "Praveen", "Naveen",
    "Ramesh", "Suresh", "Dinesh", "Mahesh", "Arun", "Anand", "Ashok", "Balaji",
    "Deepak", "Gowtham", "Hari", "Jagan", "Kumar", "Manoj", "Muthu", "Prakash",
    "Prasad", "Priya", "Rajesh", "Ravi", "Sanjay", "Saravanan", "Senthil",
    "Siva", "Sundar", "Surya", "Vijay", "Vinoth", "Aravind", "Dhanush",
]


def deepgram_keywords():
    """Keyword-boost list for the Deepgram query string.

    nova-2 takes `keywords=Term:intensity`. Boosting the project's proper nouns
    is the single highest-leverage change available to this system - it fixes
    the problem before a transcript exists, rather than patching one after.
    """
    boosted = []
    for term in LEXICON:
        # The project name matters most; everything else gets a lighter nudge.
        intensity = 3 if term in ("Karthipuram", "Unnamalai", "Neelambur") else 2
        boosted.append("%s:%d" % (term, intensity))
    # A light nudge only. These are real English words the model already
    # knows, so they need a thumb on the scale, not a shove - boosting a
    # common word hard makes the recogniser hear it where it was not said.
    for term in TOPIC_TERMS:
        boosted.append("%s:1" % term)
    # Names get more weight than topics, because the one turn where the
    # recogniser has no context to fall back on is "may I have your name?".
    for term in NAME_TERMS:
        boosted.append("%s:2" % term)
    # Whatever this deployment adds in DEEPGRAM_EXTRA_KEYWORDS, hardest of
    # all - it is there because somebody watched it go wrong.
    for term in (settings.deepgram_extra_keywords or "").split(","):
        term = term.strip()
        if term:
            boosted.append("%s:3" % term)
    return boosted
