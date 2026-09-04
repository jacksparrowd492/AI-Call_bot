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


# ------------------------------------------------------ yes / no answers
# The bot ASKS "shall I arrange a call with our sales team?" and must not act
# on it until the caller says yes. On the 2026-09-04 call the callback was
# recorded the moment the question was asked; the caller ignored it and moved
# on, and still got a text confirming an appointment he never agreed to.
_YES = re.compile(
    r"^(?:yes|yeah|yea|yep|yup|ya|sure|surely|ok|okay|okey|alright|all right|"
    r"please|definitely|absolutely|certainly|of course|why not|go ahead|"
    r"sounds good|good idea|that would be (?:good|great|nice|helpful)|"
    r"i (?:would|will|do|am))\b", re.I)

_NO = re.compile(
    r"^(?:no|nope|nah|not (?:now|really|yet|necessary|needed|interested|required)|"
    r"don'?t|do not|dont|never mind|nevermind|no need|"
    r"it'?s (?:ok|okay|fine|alright)|that'?s (?:ok|okay|fine|alright)|"
    r"maybe later|later|i'?m (?:ok|okay|fine|good))\b", re.I)

# A turn that says goodbye or thanks is closing the call, not booking one.
_FAREWELL = re.compile(
    r"\b(?:bye|goodbye|good bye|that'?s all|that is all|thank you|thanks|"
    r"thank u|thankyou)\b", re.I)

# "ok" / "okay" is a WEAK yes: people say it to acknowledge, not to agree, so
# "Okay. Thank you." must not book a callback - that exact turn ended the
# 2026-09-04 call. A turn that also says thanks or goodbye therefore only
# counts as consent when it OPENS with an unambiguous yes.
_STRONG_YES = re.compile(
    r"^(?:yes|yeah|yea|yep|yup|sure|please|definitely|absolutely|certainly|"
    r"of course|go ahead)\b", re.I)


def yes_no(text: str):
    """True / False for a direct answer, None when the turn is not one.

    None matters as much as the other two: a caller who answers the offer with
    a different question has not consented, and the turn must fall through to
    be answered normally rather than being read as either yes or no.
    """
    t = re.sub(r"[^\w\s']", " ", (text or "").lower())
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t:
        return None
    # An explicit no comes FIRST: "no thanks" is a decline, not a farewell.
    if _NO.match(t):
        return False
    if _FAREWELL.search(t):
        # "Yes please, thank you" is still a yes; "Okay, thank you" is not.
        return True if _STRONG_YES.match(t) else None
    if _YES.match(t):
        return True
    return None


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
