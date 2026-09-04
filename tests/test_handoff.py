"""Tests for handoff_intent.py - the deterministic layer that decides whether
the caller asked for a person, and whether they accepted an offered callback.

Both matter for the same reason: a callback books a salesperson's time and
sends the caller a text. Getting it wrong in either direction is expensive -
missing a real request loses the highest-intent moment on the call, and
inventing one texts a caller about an appointment they never agreed to
(which is exactly what happened on 2026-09-04 at 15:15:27).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from handoff_intent import wants_human, yes_no, extract_time   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def table(title, fn, cases):
    print("\n=== %s ===" % title)
    for text, expected in cases:
        got = fn(text)
        check("%-34r -> %s" % (text, expected), got == expected, "got %r" % (got,))


def main():
    table("asking for a person", wants_human, [
        ("can I speak to someone", True),
        ("connect me to a human", True),
        ("I want to talk to your sales team", True),
        ("please call me back", True),
        ("agent", True),
        ("I don't want to talk to a bot", True),
        # ...and the phrasings that merely MENTION a person
        ("who is the builder of this project", False),
        ("someone told me the plots are sold out", False),
        ("how many people can live there", False),
        ("is there any school nearby", False),
        ("what is the price per square foot", False),
        ("", False),
    ])

    table("answering 'shall I arrange a call?'", yes_no, [
        ("yes", True),
        ("yes please", True),
        ("yeah go ahead", True),
        ("sure", True),
        ("ok", True),
        ("okay", True),
        ("yes please, thank you", True),
        ("yes connect me", True),
        ("no", False),
        ("nope", False),
        ("no thanks", False),
        ("no thank you", False),
        ("not now", False),
        ("maybe later", False),
        ("never mind", False),
        # "okay" is a weak yes: acknowledgement, not consent. This exact turn
        # ended the 2026-09-04 call and must never book anything.
        ("okay thank you", None),
        ("Okay. Thank you.", None),
        ("thank you bye", None),
        # A caller who answers with a different question has not consented.
        ("Is there any school nearby?", None),
        ("what about the water supply", None),
        ("hmm", None),
        ("", None),
    ])

    table("preferred callback time", extract_time, [
        ("tomorrow evening", "tomorrow evening"),
        ("call me at 5pm", "5pm"),
        ("anytime", "anytime"),
        ("monday morning", "monday morning"),        # source order is kept
        ("yes please", None),
        ("", None),
    ])

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(main())
