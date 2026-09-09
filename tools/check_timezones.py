"""Prove the callback clock is right for a caller who is NOT in India.

    python -m tools.check_timezones            # the standard cases
    python -m tools.check_timezones +971501234567 "tomorrow at 6 in the evening"

Every test call so far has come from a +91 number, where the caller's zone and
the office's zone are the same instant - so the two columns in the sales sheet
(`callback_local` and `callback_ist`) always matched and the conversion was
never actually exercised. The bug that costs a lead only shows up on an ISD
call: an NRI in Dubai says "six in the evening", somebody in Coimbatore reads
"6:00 PM" off the sheet and rings them at half past four their time.

This drives the same three steps a live call does - scheduling.zone_for() on
the number that rang us, scheduling.resolve() on what the caller said,
scheduling.describe() for the sheet - and checks the result against an
independent conversion done straight from the tz database. No phone, no
Twilio, no minutes spent.
"""
import sys
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:                                    # pragma: no cover
    from backports.zoneinfo import ZoneInfo            # type: ignore

import scheduling

IST = ZoneInfo("Asia/Kolkata")

# number, what the caller said, and where they are - the last one only so a
# wrong zone lookup is obvious in the output.
CASES = [
    ("+916382779352",  "Tuesday around 5PM",             "India, the calls we have had"),
    ("+971501234567",  "tomorrow at 6 in the evening",   "Dubai, UTC+4"),
    ("+974 5512 3456", "tomorrow at 9 in the morning",   "Qatar, UTC+3"),
    ("+447700900123",  "tomorrow at 10 am",              "London, UTC+1 in September"),
    ("+12015550123",   "tomorrow at 9 am",               "New Jersey, UTC-4 in September"),
    ("+14155550123",   "tomorrow at 2 pm",               "San Francisco, UTC-7 in September"),
    ("+6591234567",    "Tuesday around 5 pm",            "Singapore, UTC+8"),
    ("+61412345678",   "tomorrow at 11 am",              "Australia, several zones"),
    ("+60123456789",   "day after tomorrow at 8 pm",     "Malaysia, UTC+8"),
    ("+96550123456",   "tomorrow evening",               "Kuwait - a period, no hour"),
]

failures = []


def check(label, ok, detail=""):
    print("   %s %s%s" % ("PASS" if ok else "FAIL", label,
                          ("  [%s]" % detail) if detail else ""))
    if not ok:
        failures.append(label)


def run(number, said, note=""):
    print("\n%s   %s" % (number, ("(%s)" % note) if note else ""))
    print('   caller said: "%s"' % said)

    where = scheduling.zone_for(number)
    print("   zone      : %s  region=%s  via %s%s"
          % (where["name"], where["region"], where["source"],
             "  AMBIGUOUS" if where["ambiguous"] else ""))
    check("a timezone was found", bool(where["zone"]), number)
    if not where["zone"]:
        return

    found = scheduling.resolve(said, where["zone"])

    if found["needs_hour"]:
        print("   -> a day with no clock hour (%r). The bridge asks once; "
              "nothing is written yet." % found["phrase"])
        check("no time is invented from a bare period", found["ist"] is None)
        return
    if found["anytime"]:
        print("   -> caller does not mind when.")
        return

    local, ist = found["local"], found["ist"]
    check("a time was understood", local is not None and ist is not None, said)
    if not (local and ist):
        return

    print("   local     : %s %s" % (scheduling.human(local), where["name"]))
    print("   sheet IST : %s" % scheduling.human(ist))
    print("   sales line: %s" % scheduling.describe(local, ist, where["name"]))
    print("   caller SMS: They will call you %s."
          % scheduling.describe_for_caller(local, ist, where["name"]))

    # The check that matters: the two columns are the SAME INSTANT, and the IST
    # column is what the tz database says that instant is in Coimbatore.
    expected = local.astimezone(IST)
    check("IST is the caller's instant, not their wall clock",
          ist == expected and ist.utcoffset() == expected.utcoffset(),
          "%s vs %s" % (ist.isoformat(), expected.isoformat()))
    check("the two sheet columns describe one moment",
          local.timestamp() == ist.timestamp())

    # And it must be in the future - a caller asking for "9 am" at 11 am their
    # time means tomorrow, worked out in THEIR morning, not ours.
    now_local = datetime.now(where["zone"])
    check("the booking is in the caller's future", local > now_local,
          "%s <= now %s" % (scheduling.human(local), scheduling.human(now_local)))

    shift = (ist.utcoffset() - local.utcoffset()).total_seconds() / 3600.0
    if abs(shift) > 0.01:
        print("   -> the sales team rings %+.1f hours off the caller's clock."
              % shift)


def main(argv):
    print("Office timezone: %s. Now: %s IST."
          % (scheduling.IST_NAME, scheduling.human(datetime.now(IST))))

    if len(argv) >= 2:
        run(argv[0], argv[1])
    else:
        for number, said, note in CASES:
            run(number, said, note)

    print("\n%s" % ("-" * 62))
    if failures:
        print("%d CHECK(S) FAILED:" % len(failures))
        for f in failures:
            print("  - %s" % f)
        return 1
    print("All conversions correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
