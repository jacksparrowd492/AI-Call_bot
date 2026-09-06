"""Verify the Twilio credentials in .env actually work, before a call proves
they don't.

Swapping TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER is easy
to get half-right: the token belongs to a different account, the number is
formatted with spaces, or the number's Voice webhook still points at a dead
ngrok host. None of that surfaces until a caller hears silence.

    python -m tools.check_twilio
"""
import re
import sys

from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

from config import settings


def main() -> int:
    sid = settings.twilio_account_sid
    tok = settings.twilio_auth_token
    num = settings.twilio_phone_number
    base = settings.public_base_url.rstrip("/")
    expected_webhook = f"{base}/voice" if base else ""

    ok = True

    # --- shape checks (catch typos without spending an API call) -----------
    if not re.fullmatch(r"AC[0-9a-f]{32}", sid):
        print(f"✗ TWILIO_ACCOUNT_SID malformed (len={len(sid)}, want AC + 32 hex)")
        ok = False
    if not re.fullmatch(r"[0-9a-f]{32}", tok):
        print(f"✗ TWILIO_AUTH_TOKEN malformed (len={len(tok)}, want 32 hex)")
        ok = False
    if not re.fullmatch(r"\+[1-9]\d{7,14}", num):
        print(f"✗ TWILIO_PHONE_NUMBER {num!r} is not E.164 (want +17752786153, no spaces)")
        ok = False
    if not ok:
        return 1

    # --- does the token actually open this account? ------------------------
    client = Client(sid, tok)
    try:
        acct = client.api.accounts(sid).fetch()
    except TwilioRestException as e:
        print(f"✗ AUTH FAILED ({e.status}): {e.msg}")
        print("  The SID and token are from different accounts, or the token was rotated.")
        return 1
    print(f"✓ auth ok - account {acct.friendly_name!r} status={acct.status} type={acct.type}")
    if acct.status != "active":
        print(f"  ⚠ account is {acct.status}; calls will not connect")
        ok = False

    # --- is the number on this account, and where does it point? -----------
    owned = client.incoming_phone_numbers.list(phone_number=num)
    if not owned:
        print(f"✗ {num} is NOT on this account. Numbers that are:")
        for p in client.incoming_phone_numbers.list(limit=20):
            print(f"    {p.phone_number}")
        return 1

    p = owned[0]
    print(f"✓ {p.phone_number} owned  voice={p.capabilities.get('voice')} sms={p.capabilities.get('sms')}")
    print(f"  VoiceUrl: {p.voice_url or '<unset>'}  [{p.voice_method}]")

    # A Voice Application SID OVERRIDES voice_url completely. If one is
    # attached, Twilio ignores the URL printed above and asks the TwiML App
    # instead - which is how a number can look perfectly configured here and
    # still never hit this server.
    app_sid = getattr(p, "voice_application_sid", "") or ""
    trunk_sid = getattr(p, "trunk_sid", "") or ""
    if app_sid:
        print(f"  ✗ voice_application_sid={app_sid} is set - it OVERRIDES the VoiceUrl above.")
        print("    Console > this number > Configure With: 'Webhook, TwiML Bin...' (not TwiML App).")
        ok = False
    if trunk_sid:
        print(f"  ✗ trunk_sid={trunk_sid} is set - calls are routed to a SIP trunk, not this webhook.")
        ok = False

    if not p.capabilities.get("voice"):
        print("  ✗ this number has no VOICE capability - it can never take a call")
        ok = False
    if expected_webhook and p.voice_url != expected_webhook:
        print(f"  ✗ webhook mismatch - expected {expected_webhook}")
        print("    Fix in Console > Phone Numbers > this number > A Call Comes In,")
        print("    or rerun after updating PUBLIC_BASE_URL in .env.")
        ok = False
    elif p.voice_method != "POST":
        print("  ✗ webhook method must be POST")
        ok = False
    elif expected_webhook:
        print("  ✓ webhook matches PUBLIC_BASE_URL")

    if not base:
        print("✗ PUBLIC_BASE_URL is empty - the tunnel URL is what Twilio calls back on")
        ok = False

    # --- did the last few calls even REACH Twilio? -------------------------
    print("\nRecent inbound calls to this number:")
    calls = client.calls.list(to=num, limit=5)
    if not calls:
        print("  (none) - no call has ever arrived at Twilio for this number.")
        print("  That means the failure is on the DIALING side, not in this app:")
        print("    - the caller's carrier never connected the call to Twilio")
        print("    - dialing a US number from India needs ISD enabled on the caller's plan")
        print("    - a trial account cannot receive calls in some regions at all")
    for c in calls:
        # "from" is a Python keyword, so the SDK renames it - and which
        # spelling you get depends on the twilio version installed.
        caller = getattr(c, "_from", None) or getattr(c, "from_", None) or "?"
        print(f"  {c.start_time}  from={caller} to={c.to} status={c.status} "
              f"duration={c.duration}s direction={c.direction}")
        # Twilio stores the reason a call failed on the notification log, not
        # the call resource; surface the code so it can be looked up directly.
        for n in client.calls(c.sid).notifications.list(limit=3):
            print(f"      ⚠ error {n.error_code}: {n.message_text or ''} {n.more_info or ''}")

    print("\n" + ("ALL GOOD" if ok else "PROBLEMS ABOVE - fix before calling"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
