"""Warm human handoff: update live call to connect to a real agent"""

import logging

from twilio.rest import Client
from twilio.twiml.voice_response import Dial, VoiceResponse

from config import settings

log = logging.getLogger("jarvis.handoff")


def handoff_to_agent(call_sid: str) -> bool:
    """Redirect active call to human agent"""

    if not settings.agent_dial_number:
        log.warning("AGENT_DIAL_NUMBER not set - cannot hand off")
        return False

    # ✅ FIX: Use VoiceResponse instead of Response
    resp = VoiceResponse()

    # Inform user
    resp.say(
        "Sure, connecting you with our sales expert now. Please hold.",
        voice="alice"  # Twilio-supported voice
    )

    # Dial agent
    dial = Dial(answer_on_bridge=True, timeout=25)
    dial.number(settings.agent_dial_number)
    resp.append(dial)

    # Fallback if agent doesn't answer
    resp.say(
        "Our team is unavailable right now. We will call you back shortly.",
        voice="alice"
    )

    # Convert to TwiML
    twiml = str(resp)

    # Twilio client
    client = Client(
        settings.twilio_account_sid,
        settings.twilio_auth_token
    )

    try:
        client.calls(call_sid).update(twiml=twiml)
        log.info(
            "Handoff executed for call %s -> %s",
            call_sid,
            settings.agent_dial_number
        )
        return True

    except Exception as e:
        log.error("Handoff failed for %s: %s", call_sid, e)
        return False