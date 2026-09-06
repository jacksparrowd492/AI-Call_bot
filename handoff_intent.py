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


# --------------------------------------------------- asking to SCHEDULE
# Separate from wants_human() on purpose, and both stay.
#
#   wants_human()    "put me through to a person"
#   wants_schedule() "let's fix a time"
#
# The second deserves its own answer: the caller has already asked, so the bot
# says yes and goes straight to which day and what time, instead of offering a
# callback they just requested.
#
# Three real knowledge-base questions sit in the blast radius and must NEVER
# trigger this - answering them with "sure, when suits you?" would be nonsense:
#   "What is the payment schedule?"
#   "How can I book a plot?"
#   "What is the booking amount?"
_SCHED_VERB = r"(?:schedule|arrange|set\s?up|fix|organi[sz]e)"
_SCHED_OBJECT = (r"(?:call|call\s?back|callback|appointment|meeting|visit|"
                 r"site\s?visit|tour|discussion|consultation|slot|time)")

_SCHED_COMBINED = re.compile(
    r"\b%s\b[^.?!]{0,20}?\b%s\b" % (_SCHED_VERB, _SCHED_OBJECT), re.I)

# "can I schedule", "I want to arrange", "let's fix" - no object needed. Note
# that "book" is deliberately absent here: "can I book" is about a plot.
_SCHED_BARE = re.compile(
    r"\b(?:can|could|may|shall)\s+(?:i|we|you)\s+(?:please\s+)?%s\b"
    r"|\bi\s+(?:want|would like|need)\s+to\s+%s\b"
    r"|\b(?:let'?s|let us)\s+%s\b" % (_SCHED_VERB, _SCHED_VERB, _SCHED_VERB), re.I)

# "book" counts only with a meeting-shaped object.
_BOOK_MEETING = re.compile(
    r"\bbook\b[^.?!]{0,15}?\b(?:call|call\s?back|callback|appointment|"
    r"meeting|visit|site\s?visit|slot|tour)\b", re.I)

_SCHED_NEG = re.compile(
    r"\b(?:payment|construction|possession|emi|instal?ment|delivery|handover|"
    r"work|project|price|cost)\s+schedule\b"
    r"|\bschedule\s+of\s+(?:payment|work|construction)\b"
    r"|\bbooking\s+(?:amount|fee|charge|process|advance|procedure)\b"
    r"|\bbook\s+(?:a|the|my|one)?\s*"
    r"(?:plot|flat|villa|unit|site(?!\s*visit)|land|apartment|house|property)\b",
    re.I)


def wants_schedule(text: str) -> bool:
    """True when the caller is asking to SET UP a call, visit or appointment."""
    t = (text or "").strip()
    if not t or _SCHED_NEG.search(t):
        return False
    return bool(_SCHED_COMBINED.search(t)
                or _SCHED_BARE.search(t)
                or _BOOK_MEETING.search(t))


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


# ------------------------------------------------------------ caller's name
# The bot asks "may I have your name?" before booking a callback, because a
# callback row with no name is next to useless to a salesperson. What comes
# back is usually just the name - Deepgram capitalises it - but people also
# say "my name is David" or "this is Karthi speaking".
_NAME_LEAD = re.compile(
    r"\b(?:my name is|my name's|i am|i'?m|this is|it'?s|it is|name is|"
    r"you can call me|call me)\s+"
    r"([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2})", re.I)

# "Karthi speaking", "David here" - the name is the part in front.
_NAME_TRAILERS = {"speaking", "here", "only", "side", "itself", "calling"}

# Words that are never the answer to "what is your name?", so a turn made only
# of these is a refusal, a greeting or an answer to some other question.
_NOT_A_NAME = {
    "yes", "yeah", "no", "nope", "not", "hello", "hi", "hey", "ok", "okay",
    "sure", "thanks", "thank", "you", "please", "sorry", "what", "who", "why",
    "how", "when", "where", "which", "tomorrow", "today", "tonight", "morning",
    "evening", "afternoon", "night", "call", "callback", "time", "name", "my",
    "is", "the", "a", "an", "i", "am", "it", "this", "that", "later", "now",
    "anytime", "speaking", "here", "fine", "good", "well", "rather", "prefer",
    "interested", "nothing", "none", "skip", "never", "mind", "mister", "sir",
    "madam", "maam",
    # Project vocabulary. When the recogniser mishears an answer to "may I
    # have your name?" it does not produce noise - it produces a word it was
    # expecting on this call, and "my name is Karthi" came back as "security".
    # Writing that into the sales sheet as somebody's name is worse than
    # asking them again, which is what bridge.py now does when this list
    # rejects the answer.
    "security", "water", "supply", "transport", "transportation", "project",
    "brochure", "price", "plot", "plots", "amenities", "infrastructure",
    "electricity", "connectivity", "parks", "park", "school", "schools",
    "metro", "road", "roads", "location", "address", "site", "layout",
    "booking", "township", "details", "detail", "information", "facility",
    "facilities", "sewage", "drainage", "borewell", "maintenance",
    # Contractions. A caller who starts to introduce themselves and is cut off
    # by the endpointer leaves a bare "I'm" behind - and "I'm" went into the
    # sales sheet as somebody's name on the 2026-09-06 call, because the plain
    # words "i" and "am" were listed here but the contraction was not.
    "i'm", "im", "i've", "ive", "i'll", "ill", "i'd", "it's", "its", "that's",
    "thats", "there's", "theres", "we're", "were", "you're", "youre", "let's",
    "lets", "don't", "dont", "can't", "cant", "won't", "wont", "isn't", "isnt",
    "he's", "she's", "they're", "what's", "whats",
    # Spoken clock words. These arrive on their own when a caller answers the
    # time question and the name question in the wrong order.
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "o'clock", "oclock", "am", "pm",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "week", "weekend", "next", "noon", "midnight", "around", "at",
    "after", "before", "by", "sometime", "half", "past", "quarter", "clock",
    "whenever", "any",
}

_NAME_WORD = re.compile(r"[A-Za-z][A-Za-z.'-]*$")


def extract_name(text: str):
    """Pull a caller's name out of their answer, or None.

    Conservative on purpose. Writing "Not Interested" into the sales sheet as
    somebody's name is worse than leaving the cell empty.
    """
    t = (text or "").strip()
    if not t:
        return None

    m = _NAME_LEAD.search(t)
    if m:
        candidate = m.group(1)
    else:
        # A bare answer: "David." / "Karthi Kumar" - and often the name plus
        # the time in one breath: "David, tomorrow at six". Take the leading
        # run of name-shaped words and stop at the first word that cannot be
        # part of a name, which is what ends "David, tomorrow ..." at "David".
        head = re.split(r"[,;]", t)[0]
        bare = re.sub(r"[^\w\s'-]", " ", head)
        bare = re.sub(r"\s{2,}", " ", bare).strip()
        words = bare.split()
        # More than three words before the comma is a sentence, not a name.
        # Without this, "can we schedule a call" becomes "Can We Schedule".
        if not words or len(words) > 3:
            return None
        lead = []
        for word in words:
            if word.lower() in _NOT_A_NAME or not _NAME_WORD.match(word):
                break
            lead.append(word)
        if not lead:
            return None
        candidate = " ".join(lead)

    words = [w.strip(".,'-") for w in candidate.split()]
    words = [w for w in words if w]
    while words and words[-1].lower() in _NAME_TRAILERS:
        words.pop()
    if not words or len(words) > 3:
        return None
    if any(w.lower() in _NOT_A_NAME for w in words):
        return None
    if any(not _NAME_WORD.match(w) for w in words):
        return None
    if len("".join(words)) < 2:
        return None
    return " ".join(w[:1].upper() + w[1:] for w in words)


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
