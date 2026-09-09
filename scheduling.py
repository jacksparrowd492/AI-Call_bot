"""Turn what a caller says about timing into a time the sales team can act on.

Most Karthipuram callers are NRIs. When one says "call me at six in the
evening" they mean six o'clock where THEY are, not six in Coimbatore. Writing
that down literally is how a caller gets rung at two in the morning, so every
requested time is anchored to the caller's own zone - worked out from the
number that rang us - and converted to IST for the person who has to make the
call.

Two jobs:
  zone_for(number)     -> which timezone that phone number lives in
  resolve(text, zone)  -> "tomorrow at six in the evening" -> a real datetime,
                          in the caller's zone AND in IST

resolve() deliberately reports when it could NOT pin down an hour. "Tomorrow
evening" is not a time; guessing 18:00 and writing it into the sheet would
invent a precision the caller never gave. The bridge asks one follow-up
question instead.
"""
import logging
import re
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:                                    # Python < 3.9
    ZoneInfo = None

log = logging.getLogger("jarvis.scheduling")

try:
    from config import settings
    IST_NAME = getattr(settings, "office_timezone", None) or "Asia/Kolkata"
except Exception:                                      # standalone use / tests
    IST_NAME = "Asia/Kolkata"

try:
    import phonenumbers
    from phonenumbers.timezone import time_zones_for_number
except ImportError:                                    # pragma: no cover
    phonenumbers = None
    log.warning("phonenumbers is not installed - falling back to a country-code "
                "table. US and Australian numbers will be approximate. "
                "Fix with: pip install phonenumbers")


# ---------------------------------------------------------------- timezones
# Used only when phonenumbers is unavailable. Longest prefix wins. Single-zone
# countries are exact here; +1 and +61 are a guess, which is the whole reason
# phonenumbers is the preferred path.
_PREFIX_ZONES = {
    "91": "Asia/Kolkata",      "971": "Asia/Dubai",     "966": "Asia/Riyadh",
    "974": "Asia/Qatar",       "968": "Asia/Muscat",    "965": "Asia/Kuwait",
    "973": "Asia/Bahrain",     "44": "Europe/London",   "65": "Asia/Singapore",
    "60": "Asia/Kuala_Lumpur", "1": "America/New_York", "61": "Australia/Sydney",
    "64": "Pacific/Auckland",  "27": "Africa/Johannesburg", "49": "Europe/Berlin",
    "33": "Europe/Paris",      "39": "Europe/Rome",     "41": "Europe/Zurich",
    "31": "Europe/Amsterdam",  "353": "Europe/Dublin",  "852": "Asia/Hong_Kong",
    "81": "Asia/Tokyo",        "82": "Asia/Seoul",      "86": "Asia/Shanghai",
    "254": "Africa/Nairobi",   "234": "Africa/Lagos",   "230": "Indian/Mauritius",
    "94": "Asia/Colombo",      "977": "Asia/Kathmandu", "880": "Asia/Dhaka",
    "92": "Asia/Karachi",      "20": "Africa/Cairo",    "962": "Asia/Amman",
    "961": "Asia/Beirut",      "90": "Europe/Istanbul", "7": "Europe/Moscow",
}

# A mobile number is not geographic, so phonenumbers returns EVERY zone in the
# country. Pick the commercial centre rather than the alphabetically first one
# (which for Australia is Adelaide, and for the US would be Adak).
_REGION_PREFERRED = {
    "AU": "Australia/Sydney",  "US": "America/New_York", "CA": "America/Toronto",
    "RU": "Europe/Moscow",     "BR": "America/Sao_Paulo", "ID": "Asia/Jakarta",
    "IN": "Asia/Kolkata",      "MY": "Asia/Kuala_Lumpur", "NZ": "Pacific/Auckland",
}

# phonenumbers still reports the pre-1993 spelling. The Crown Dependencies are
# here for a different reason: a UK mobile can come back as Europe/Guernsey,
# which keeps London's clock to the second but puts "Guernsey" in front of a
# sales team ringing a caller in Manchester.
_ALIASES = {"Asia/Calcutta": "Asia/Kolkata", "Asia/Katmandu": "Asia/Kathmandu",
            "Europe/Guernsey": "Europe/London", "Europe/Jersey": "Europe/London",
            "Europe/Isle_of_Man": "Europe/London", "Europe/Belfast": "Europe/London",
            "Europe/Busingen": "Europe/Zurich"}


def _digits(number):
    return re.sub(r"\D", "", number or "")


def _zone(name):
    if not name or ZoneInfo is None:
        return None
    try:
        return ZoneInfo(name)
    except Exception as e:
        # On Windows, zoneinfo has no database of its own: pip install tzdata.
        log.error("Unknown timezone %r (%s). On Windows: pip install tzdata", name, e)
        return None


def zone_for(number: str) -> dict:
    """Where the caller is, as best we can tell from their number.

    Returns {"name", "zone", "region", "ambiguous", "source"}. `zone` is None
    when the name cannot be loaded, and every caller of this module has to cope
    with that rather than assume IST.
    """
    out = {"name": None, "zone": None, "region": None,
           "ambiguous": False, "source": "unknown"}
    digits = _digits(number)
    if not digits:
        return out

    if phonenumbers is not None:
        try:
            parsed = phonenumbers.parse("+" + digits, None)
            out["region"] = phonenumbers.region_code_for_number(parsed)
            zones = [z for z in time_zones_for_number(parsed)
                     if z and not z.startswith("Etc/")]
            zones = [_ALIASES.get(z, z) for z in zones]
            if zones:
                out["ambiguous"] = len(zones) > 1
                preferred = _REGION_PREFERRED.get(out["region"])
                out["name"] = (preferred if preferred in zones else zones[0])
                out["source"] = "phonenumbers"
        except Exception as e:
            log.warning("Could not read a timezone off %r: %s", number, e)

    if not out["name"]:
        for length in (4, 3, 2, 1):                    # longest prefix wins
            hit = _PREFIX_ZONES.get(digits[:length])
            if hit:
                out["name"] = hit
                out["source"] = "prefix-table"
                out["ambiguous"] = digits[:length] in ("1", "61", "7")
                break

    out["zone"] = _zone(out["name"])
    return out


def to_ist(dt: datetime):
    """Convert an aware datetime to IST. None in, None out."""
    ist = _zone(IST_NAME)
    if dt is None or ist is None or dt.tzinfo is None:
        return None
    return dt.astimezone(ist)


# --------------------------------------------------------------- time words
_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday"]

# A period word is not an hour, but it does tell us which half of the clock a
# bare number belongs to: "six in the evening" is 18:00, not 06:00.
_PERIOD_PM = {"afternoon": True, "evening": True, "night": True, "tonight": True,
              "morning": False, "am": False, "pm": True, "noon": True}
_PERIOD_RE = re.compile(r"\b(morning|afternoon|evening|night|tonight|noon)\b", re.I)

_ANYTIME_RE = re.compile(r"\b(any\s?time|anytime|whenever|no preference|"
                         r"up to you|your convenience)\b", re.I)
_NOW_RE = re.compile(r"\b(right now|straight ?away|immediately|as soon as "
                     r"possible|asap)\b", re.I)

_HOUR_RES = [
    # A RANGE - "5 to 6 pm", "between 5 and 6". The caller is saying they are
    # free FROM the first hour, so that is the one to book; the am/pm sits on
    # the second half and applies to both.
    re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(?:to|till|until|-|and)\s*"
               r"\d{1,2}(?:[:.]\d{2})?\s*(a\.?m\.?|p\.?m\.?)", re.I),
    # 5:30 pm / 17:30 / 5.30pm
    re.compile(r"\b(\d{1,2})[:.](\d{2})\s*(a\.?m\.?|p\.?m\.?)?", re.I),
    # 5 pm / 5pm
    re.compile(r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)", re.I),
    # 5 o'clock
    re.compile(r"\b(\d{1,2})\s*o'?\s*clock\b", re.I),
    # five pm / five o'clock / five thirty
    re.compile(r"\b(%s)\s*(a\.?m\.?|p\.?m\.?|o'?\s*clock)?\b"
               % "|".join(_WORD_NUM), re.I),
    # "6 in the evening" / "6 evening" - the period word makes it a time
    re.compile(r"\b(\d{1,2})\s*(?:in the\s+)?(?:morning|afternoon|evening|night)\b",
               re.I),
    # "at 6", "around 6", "by 6" - a preposition makes a bare number a time
    # rather than a quantity, which keeps "3 BHK" and "1882 plots" out.
    re.compile(r"\b(?:at|around|by|after|before)\s+(\d{1,2})(?!\d)\b", re.I),
]


_FRACTION_RE = re.compile(r"\b(half|quarter)\s+(past|to)\b", re.I)


def _fraction_minutes(text):
    """(minute, hour_shift) for "half past six" / "quarter to seven"."""
    m = _FRACTION_RE.search(text)
    if not m:
        return 0, 0
    amount = 30 if m.group(1).lower() == "half" else 15
    if m.group(2).lower() == "to":
        return 60 - amount, -1
    return amount, 0


def _find_hour(text):
    """Return (hour_0_23 or None, minute, explicit_meridiem_bool)."""
    pm_hint = None
    m = _PERIOD_RE.search(text)
    if m:
        pm_hint = _PERIOD_PM.get(m.group(1).lower())

    for i, rx in enumerate(_HOUR_RES):
        m = rx.search(text)
        if not m:
            continue
        raw = m.group(1).lower()
        hour = _WORD_NUM.get(raw) if raw in _WORD_NUM else int(raw)
        minute = 0
        if m.re.groups >= 2 and m.group(2) and re.fullmatch(r"\d{1,2}", m.group(2)):
            minute = int(m.group(2))
        if hour is None or hour > 24 or minute > 59:
            continue

        if not minute:
            minute, shift = _fraction_minutes(text)
            hour += shift

        mer = None
        for g in m.groups()[1:]:
            if g and re.fullmatch(r"[ap]\.?m\.?", g.strip(), re.I):
                mer = g.strip().lower()[0]
        if mer == "p":
            hour = hour % 12 + 12
            return hour, minute, True
        if mer == "a":
            hour = hour % 12
            return hour, minute, True

        # No am/pm said. A period word settles it; otherwise assume the caller
        # means business hours, so 1-7 is afternoon/evening rather than dawn.
        if pm_hint is True and hour <= 12:
            hour = hour % 12 + 12
        elif pm_hint is False and hour <= 12:
            hour = hour % 12
        elif hour <= 7:
            hour += 12
        return hour, minute, False
    return None, 0, False


def _find_day_offset(text, now):
    """Days from today, or None when the caller named no day."""
    if re.search(r"\bday after tomorrow\b", text):
        return 2
    if re.search(r"\btomorrow\b", text):
        return 1
    if re.search(r"\b(today|tonight|this (?:morning|afternoon|evening))\b", text):
        return 0
    if re.search(r"\bnext week\b", text):
        return 7
    for i, day in enumerate(_WEEKDAYS):
        if re.search(r"\b%s\b" % day, text):
            ahead = (i - now.weekday()) % 7
            return ahead or 7                          # "monday" on a Monday = next one
    return None


def resolve(text: str, zone=None, now=None) -> dict:
    """Read a requested callback time out of what the caller said.

    Returns:
      phrase      what they said, trimmed to the time-ish part (for the notes)
      local       aware datetime in the caller's zone, or None
      ist         the same instant in IST, or None
      needs_hour  True when a DAY or PERIOD was given but no clock hour, so the
                  bridge should ask once rather than invent one
      anytime     True when the caller said they don't mind
    """
    out = {"phrase": None, "local": None, "ist": None,
           "needs_hour": False, "anytime": False}
    t = (text or "").strip().lower()
    if not t:
        return out

    if zone is None:
        zone = _zone(IST_NAME)
    if now is None:
        now = datetime.now(zone) if zone else datetime.now()

    if _ANYTIME_RE.search(t):
        out.update(phrase="anytime", anytime=True)
        return out

    if _NOW_RE.search(t):
        out.update(phrase="right now", local=now, ist=to_ist(now))
        return out

    hour, minute, _explicit = _find_hour(t)
    day_offset = _find_day_offset(t, now)
    period = _PERIOD_RE.search(t)

    if hour is None:
        # A day or a period with no hour is not a time. Say so.
        if day_offset is not None or period:
            bits = [b for b in (re.search(r"\b(today|tomorrow|day after tomorrow|"
                                          r"next week|%s)\b" % "|".join(_WEEKDAYS), t),
                                period) if b]
            out["phrase"] = " ".join(b.group(0) for b in bits)
            out["needs_hour"] = True
        return out

    if day_offset is None:
        day_offset = 0

    try:
        local = now.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
        local += timedelta(days=day_offset)
    except ValueError:
        return out

    # A time that has already passed today means tomorrow, unless they named
    # the day themselves.
    if local <= now and _find_day_offset(t, now) is None:
        local += timedelta(days=1)

    out["local"] = local
    out["ist"] = to_ist(local)
    out["phrase"] = human(local)
    return out


def describe_for_caller(local, ist, zone_name=None) -> str:
    """The same appointment, written for the person who asked for it.

    An NRI who says "six in the evening" and is texted "7:30 PM IST" has to do
    the arithmetic themselves to find out whether that is what they asked for -
    and the one who gets it wrong misses the call. Their own clock leads; IST
    follows in brackets so they know when the office will be dialling.
    """
    if local is None:
        return human(ist) + " IST" if ist is not None else ""
    line = human(local)
    if ist is not None and zone_name and zone_name != IST_NAME:
        line += " your time (%s IST)" % _clock(ist)
    return line


def _clock(dt) -> str:
    """5:30 PM - no leading zero, which reads badly out loud and in a sheet."""
    return "%s:%s %s" % (dt.strftime("%I").lstrip("0") or "12",
                         dt.strftime("%M"), dt.strftime("%p"))


def human(dt) -> str:
    """Sat 5 Sep, 10:30 AM"""
    return "%s %d %s, %s" % (dt.strftime("%a"), dt.day, dt.strftime("%b"),
                             _clock(dt))


def place(zone_name) -> str:
    """"Asia/Dubai" -> "Dubai". A tz identifier is not something to put in
    front of the sales team, or in a message to a caller."""
    if not zone_name:
        return ""
    return zone_name.split("/")[-1].replace("_", " ")


def describe(local, ist, zone_name=None) -> str:
    """One line for the sales team's SMS and the spreadsheet. IST first,
    because the person reading it is in Coimbatore."""
    if ist is None:
        return ""
    line = human(ist) + " IST"
    if local is not None and zone_name and zone_name != IST_NAME:
        line += " (%s in %s)" % (_clock(local), place(zone_name))
    return line
