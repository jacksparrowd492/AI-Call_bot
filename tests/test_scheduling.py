"""Timezone handling and the Excel call log.

Most Karthipuram callers are NRIs, so "call me at six in the evening" is six
o'clock where THEY are. Booking that as an Indian time is how a caller gets
rung at half past three in the morning - so every conversion below is checked
against a fixed "now", not the clock on whatever machine runs the suite.
"""
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="jarvis_calllog_")
os.environ["CALL_LOG_PATH"] = os.path.join(_TMP, "call_log.xlsx")

from zoneinfo import ZoneInfo                      # noqa: E402
import scheduling                                  # noqa: E402
from tools import call_log                         # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


DUBAI = ZoneInfo("Asia/Dubai")
NOW = datetime(2026, 9, 4, 15, 0, tzinfo=DUBAI)      # Friday afternoon in Dubai


def ist_of(text, zone=DUBAI, now=NOW):
    r = scheduling.resolve(text, zone, now)
    return r["ist"].strftime("%a %d %b %H:%M") if r["ist"] else None


def main():
    print("\n=== 1. the caller's timezone comes from their number ===")
    for number, region, zone in [
        ("+916382779352", "IN", "Asia/Kolkata"),
        ("+971501234567", "AE", "Asia/Dubai"),
        ("+966501234567", "SA", "Asia/Riyadh"),
        ("+442071838750", "GB", "Europe/London"),
        ("+6591234567",   "SG", "Asia/Singapore"),
        # +1 spans six zones; only the area code separates them.
        ("+14155552671",  "US", "America/Los_Angeles"),
        ("+12125551234",  "US", "America/New_York"),
    ]:
        got = scheduling.zone_for(number)
        check("%-15s -> %s" % (number, zone),
              got["name"] == zone and got["region"] == region,
              "%s / %s" % (got["region"], got["name"]))

    got = scheduling.zone_for("+61412345678")
    check("an Australian mobile picks Sydney, not alphabetical-first Adelaide",
          got["name"] == "Australia/Sydney" and got["ambiguous"] is True,
          "%s ambiguous=%s" % (got["name"], got["ambiguous"]))

    check("an unusable number yields no zone rather than a wrong one",
          scheduling.zone_for("garbage")["name"] is None)

    print("\n=== 2. what the caller says is in THEIR time, stored as IST ===")
    # Dubai is UTC+4, India UTC+5:30 -> +1h30m.
    for text, expected in [
        ("six in the evening",        "Fri 04 Sep 19:30"),
        ("at 6 pm",                   "Fri 04 Sep 19:30"),
        ("tomorrow at 9 am",          "Sat 05 Sep 10:30"),
        ("tomorrow evening around 6", "Sat 05 Sep 19:30"),
        ("10:30 am tomorrow",         "Sat 05 Sep 12:00"),
        ("half past nine tomorrow morning", "Sat 05 Sep 11:00"),
        ("quarter to seven",          "Fri 04 Sep 20:15"),
        ("8 in the morning",          "Sat 05 Sep 09:30"),
    ]:
        got = ist_of(text)
        check("%-34r -> %s IST" % (text, expected), got == expected, str(got))

    print("\n=== 3. the same words from New Jersey are a different IST time ===")
    ny = ZoneInfo("America/New_York")
    ny_now = datetime(2026, 9, 4, 9, 0, tzinfo=ny)
    got = ist_of("tomorrow at 8 pm", ny, ny_now)
    check("8pm in New York is 5:30am IST the NEXT day",
          got == "Sun 06 Sep 05:30", str(got))
    check("...which is exactly why it must not be booked as 8pm IST",
          got != "Sat 05 Sep 20:00")

    print("\n=== 4. a day without an hour is not a time ===")
    for text in ("tomorrow evening", "tomorrow", "monday morning", "evening"):
        r = scheduling.resolve(text, DUBAI, NOW)
        check("%-20r asks for an hour instead of guessing" % text,
              r["needs_hour"] is True and r["ist"] is None,
              "needs_hour=%s ist=%s" % (r["needs_hour"], r["ist"]))

    r = scheduling.resolve("anytime", DUBAI, NOW)
    check("'anytime' is an answer, not a missing hour",
          r["anytime"] is True and r["needs_hour"] is False)

    for text in ("I want a 3 BHK", "1882 plots", "yes please", ""):
        r = scheduling.resolve(text, DUBAI, NOW)
        check("%-18r is not read as a time" % text,
              r["ist"] is None and r["needs_hour"] is False,
              str(r["phrase"]))

    print("\n=== 5. without phonenumbers, the prefix table still answers ===")
    real, scheduling.phonenumbers = scheduling.phonenumbers, None
    try:
        for number, zone in [("+971501234567", "Asia/Dubai"),
                             ("+966501234567", "Asia/Riyadh"),
                             ("+916382779352", "Asia/Kolkata")]:
            got = scheduling.zone_for(number)
            check("fallback %-15s -> %s" % (number, zone),
                  got["name"] == zone and got["source"] == "prefix-table",
                  "%s via %s" % (got["name"], got["source"]))
    finally:
        scheduling.phonenumbers = real

    print("\n=== 6. the Excel call log ===")
    path = os.environ["CALL_LOG_PATH"]
    rec = call_log.build_record(
        started_at=datetime(2026, 9, 4, 15, 14, 46),
        ended_at=datetime(2026, 9, 4, 15, 18, 22),
        name="David", phone="+971501234567", country="AE",
        caller_timezone="Asia/Dubai", requirement="3BHK villa plot",
        intent_score=8, callback=True, asked_for="tomorrow evening around 6",
        callback_local="Sat 5 Sep, 6:00 PM", callback_ist="Sat 5 Sep, 7:30 PM",
        confirmed_via="whatsapp+sms", agent_alerted=True, brochure_sent=True,
        summary="Asked about schools and pricing. Callback booked for Sat 5 Sep.",
        call_sid="CAtest")

    check("duration is worked out from the call, in minutes",
          rec["duration_min"] == 3.6, str(rec["duration_min"]))
    check("append writes the sheet", call_log.append(rec) and os.path.exists(path))

    import openpyxl
    ws = openpyxl.load_workbook(path)["Calls"]
    check("header row is written once", ws.max_row == 2, "rows=%d" % ws.max_row)
    check("the header stays visible when scrolling", ws.freeze_panes == "A2")
    row = {h.value: c.value for h, c in zip(ws[1], ws[2])}
    check("both times are on the row",
          row["Callback (caller local)"] == "Sat 5 Sep, 6:00 PM"
          and row["Callback (IST)"] == "Sat 5 Sep, 7:30 PM", str(row))
    check("the caller's own words are kept too",
          row["Asked for"] == "tomorrow evening around 6", str(row["Asked for"]))
    check("the call summary has its own column",
          row["Call summary"].startswith("Asked about schools"),
          repr(row["Call summary"]))

    call_log.append(rec)
    ws = openpyxl.load_workbook(path)["Calls"]
    check("a second call appends rather than overwrites",
          ws.max_row == 3, "rows=%d" % ws.max_row)

    print("\n=== 7. a call is never lost because the sheet is open in Excel ===")
    real_append = call_log.append

    def locked(*a, **kw):
        raise PermissionError("[Errno 13] file is open in Excel")

    import openpyxl as _op
    orig_save = _op.Workbook.save
    _op.Workbook.save = locked
    try:
        ok = call_log.append(rec)
    finally:
        _op.Workbook.save = orig_save
    check("the row is parked instead of dropped", ok is True)
    check("...in a sidecar CSV next to the sheet",
          os.path.exists(path + ".pending.csv"))

    call_log.append(rec)
    ws = openpyxl.load_workbook(path)["Calls"]
    check("the parked row is merged back in on the next write",
          ws.max_row == 5, "rows=%d (want 3 + parked + new)" % ws.max_row)
    check("and the sidecar is cleared", not os.path.exists(path + ".pending.csv"))

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(main())
