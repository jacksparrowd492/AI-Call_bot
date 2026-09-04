"""Outbound messaging to the caller: WhatsApp first, SMS if that is not available.

Every caller-facing message goes through deliver_message(), which tries WhatsApp
and falls back to SMS. That is the "according to their convenience" rule:
if the number is on WhatsApp they get it there, otherwise they get an SMS.

Development: with the Twilio sandbox, only numbers that have sent the join code
can receive WhatsApp, so most real callers will land on the SMS path.
Production : register a WhatsApp business sender; business-initiated messages
outside the 24h window need an approved template.
"""
import logging

from twilio.rest import Client

from config import settings

log = logging.getLogger("jarvis.whatsapp")

BROCHURE_BODY = ("Thank you for your interest in Karthipuram! Here is our project "
                 "brochure with the layout and details.")


def brochure_url() -> str:
    """Explicit BROCHURE_URL wins; otherwise serve it off PUBLIC_BASE_URL,
    which is the ngrok host Twilio already talks to."""
    if settings.brochure_url:
        return settings.brochure_url.strip()
    base = (settings.public_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if not base.startswith("http"):
        base = "https://" + base
    return base + "/brochure.pdf"


def _client():
    if not (settings.twilio_account_sid and settings.twilio_auth_token):
        log.warning("Twilio credentials missing - cannot send messages")
        return None
    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


# --------------------------------------------------------------- core sender

def _send_whatsapp(client, to_number, body, media_url=None):
    kwargs = {"from_": settings.whatsapp_from,
              "to": f"whatsapp:{to_number}",
              "body": body}
    if media_url:
        kwargs["media_url"] = [media_url]
    msg = client.messages.create(**kwargs)
    log.info("WhatsApp sent: sid=%s to=%s", msg.sid, to_number)
    return msg.sid


def _send_sms(client, to_number, body, media_url=None):
    if not settings.twilio_phone_number:
        raise RuntimeError("TWILIO_PHONE_NUMBER not set")
    # SMS cannot carry a PDF, so the link goes in the text.
    if media_url and media_url not in body:
        body = f"{body} {media_url}"
    msg = client.messages.create(
        from_=settings.twilio_phone_number, to=to_number, body=body)
    log.info("SMS sent: sid=%s to=%s", msg.sid, to_number)
    return msg.sid


def deliver_message(to_number: str, body: str, media_url: str = None) -> str:
    """WhatsApp, else SMS. Returns 'whatsapp', 'sms' or 'failed'."""
    if not to_number or to_number == "unknown":
        log.warning("No destination number - cannot send message")
        return "failed"

    client = _client()
    if not client:
        return "failed"

    try:
        _send_whatsapp(client, to_number, body, media_url)
        return "whatsapp"
    except Exception as e:
        log.warning("WhatsApp unavailable for %s (%s) - falling back to SMS",
                    to_number, e)

    try:
        _send_sms(client, to_number, body, media_url)
        return "sms"
    except Exception as e:
        log.error("SMS also failed for %s: %s", to_number, e)
        return "failed"


# ------------------------------------------------------------------ brochure

def deliver_brochure(to_number: str) -> str:
    url = brochure_url()
    if not url:
        log.warning("No brochure URL (set BROCHURE_URL or PUBLIC_BASE_URL)")
    return deliver_message(to_number, BROCHURE_BODY, url)


# ------------------------------------------------------------------ callback

def callback_confirmation(name=None, preferred_time=None,
                          include_brochure=True) -> str:
    """The message the caller receives after asking to speak to a person."""
    who = f"Hi {name}, " if name else "Hi, "
    when = (f"They will call you {preferred_time}."
            if preferred_time else "They will call you back shortly.")
    body = (f"{who}thank you for calling Karthipuram. "
            f"We have scheduled a callback with one of our sales specialists. "
            f"{when}")
    if include_brochure:
        body += " Meanwhile, here is our project brochure."
    return body


def deliver_callback_confirmation(to_number, name=None, preferred_time=None,
                                  include_brochure=True) -> str:
    body = callback_confirmation(name, preferred_time, include_brochure)
    url = brochure_url() if include_brochure else None
    return deliver_message(to_number, body, url)


# --------------------------------------------------------------- agent alert

def notify_agent(caller_number, name=None, requirement=None,
                 preferred_time=None) -> bool:
    """SMS the sales agent so a human actually makes the call."""
    if not settings.agent_dial_number:
        log.warning("AGENT_DIAL_NUMBER not set - the sales team was NOT alerted")
        return False

    client = _client()
    if not client or not settings.twilio_phone_number:
        log.warning("Cannot alert the agent - Twilio number or credentials missing")
        return False

    lines = ["Karthipuram callback request", f"Caller: {caller_number}"]
    if name:
        lines.append(f"Name: {name}")
    if requirement:
        lines.append(f"Interest: {requirement}")
    lines.append(f"Preferred time: {preferred_time or 'not specified'}")

    try:
        msg = client.messages.create(
            from_=settings.twilio_phone_number,
            to=settings.agent_dial_number,
            body="\n".join(lines),
        )
        log.info("Sales agent alerted: sid=%s to=%s", msg.sid,
                 settings.agent_dial_number)
        return True
    except Exception as e:
        log.error("Could not alert the sales agent: %s", e)
        return False


# Backwards-compatible names
send_brochure = deliver_brochure
