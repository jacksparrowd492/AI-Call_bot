"""Detect "put me through to a human" without relying on the LLM.

Two layers catch this, and both matter:

  1. This module - deterministic, instant, works even if Groq is slow, times
     out, or returns malformed METADATA.
  2. The METADATA "handoff": true field - catches phrasings not listed here.

A caller asking for a person is the highest-intent moment on the whole call.
Missing it because a JSON field did not parse is not acceptable, so the
keyword layer runs first and independently.
"""
import re

# Explicit, unambiguous asks.
_DIRECT = [
    r"\bconnect me\b", r"\btransfer me\b", r"\bput me (?:on|through)\b",
    r"\bcall me back\b", r"\bcall back\b", r"\bcallback\b",
    r"\bi want to (?:talk|speak)\b", r"\bcan i (?:talk|speak)\b",
    r"\bi need to (?:talk|speak)\b", r"\blet me (?:talk|speak)\b",
    # "not a bot", and the contraction form: "I don't want to talk to a bot"
    r"\bnot (?:a |the )?(?:bot|robot|machine|ai)\b",
    r"\b(?:to|with)\s+(?:a\s+|the\s+)?(?:bot|robot|machine|computer)\b",
    r"\b(?:don'?t|do not|dont|stop)\b[^.?!]{0,24}\b(?:bot|robot|machine|automated)\b",
    r"\breal person\b", r"\bactual person\b", r"\bhuman being\b",
]

# "talk / speak / connect / transfer" + "a person"
_VERB = r"(?:talk|speak|connect|transfer|reach|contact|meet|discuss)"
_PERSON = (r"(?:human|person|someone|somebody|agent|executive|manager|"
           r"representative|rep|advisor|adviser|specialist|expert|"
           r"sales(?:\s?person|\s?team|\s?guy)?|staff|officer|owner|builder)")
_COMBINED = re.compile(r"%s\b[^.?!]{0,30}?\b%s" % (_VERB, _PERSON), re.I)
_PERSON_ONLY = re.compile(r"\b%s\b" % _PERSON, re.I)

_DIRECT_RE = [re.compile(p, re.I) for p in _DIRECT]

# Phrases that mention a person but are NOT a request to be transferred.
_NEGATIVE = re.compile(
    r"\b(?:someone told|somebody told|a person told|i heard from|"
    r"my (?:friend|wife|husband|father|brother|sister|son|daughter)|"
    r"who is the (?:owner|builder|promoter|founder)|"
    r"how many (?:people|persons)|per person)\b", re.I)


def wants_human(text: str) -> bool:
    """True when the caller is asking to be put in touch with a person."""
    t = (text or "").strip()
    if not t:
        return False
    if _NEGATIVE.search(t):
        return False

    for rx in _DIRECT_RE:
        if rx.search(t):
            # "I want to speak" alone counts; "call me back" counts.
            return True

    if _COMBINED.search(t):
        return True

    # Bare "human?" / "agent" / "sales team please" - short turns only, so a
    # long sentence that merely contains the word does not trigger.
    if len(t.split()) <= 4 and _PERSON_ONLY.search(t):
        return True

    return False


# ------------------------------------------------------- preferred call time
_TIME_PATTERNS = [
    r"\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    r"\b(morning|afternoon|evening|night|tonight)\b",
    r"\b(today|tomorrow|day after tomorrow)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(after\s+\d{1,2}(?:\s*(?:am|pm))?)\b",
    r"\b(any\s?time|anytime|right now|now)\b",
    r"\b(next\s+week|this\s+week|weekend)\b",
]
_TIME_RE = [re.compile(p, re.I) for p in _TIME_PATTERNS]


def extract_time(text: str):
    """Pull a rough preferred time out of what the caller said, or None.

    Deliberately loose - this is a note for the salesperson, not a calendar
    entry, so "tomorrow evening" is a perfectly good answer.
    """
    t = (text or "").strip()
    if not t:
        return None
    spans = []
    for rx in _TIME_RE:
        for m in rx.finditer(t):
            val = (m.group(1) or "").strip()
            if val:
                spans.append((m.start(), m.end(), val))
    if not spans:
        return None

    # Keep source order, and drop any match contained inside a longer one
    # ("after 7pm" already covers "7pm").
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept = []
    for start, end, val in spans:
        if any(start >= ks and end <= ke for ks, ke, _ in kept):
            continue
        if any(val.lower() == kv.lower() for _, _, kv in kept):
            continue
        kept.append((start, end, val))
    return " ".join(v for _, _, v in kept[:3])
