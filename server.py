import json
import logging
import asyncio
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from twilio.twiml.voice_response import VoiceResponse, Connect

from config import settings
from bridge import MediaStreamBridge

# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("jarvis.server")


# ------------------------------------------------------------------ startup

async def _warm_models():
    """Pay for the model loads before the phone rings.

    Kokoro's ONNX model loaded lazily inside the FIRST caller's greeting (1.0s
    on 2026-09-04) and Groq's model list was fetched in the middle of the first
    real answer (0.36s). Neither belongs on a live call.
    """
    missing = settings.missing()
    if missing:
        log.warning("Missing required settings: %s", ", ".join(missing))

    def _warm():
        try:
            from tts import warmup as warm_tts
            warm_tts()
        except Exception as e:
            log.error("TTS warmup failed: %s", e)
        try:
            from llm import GroqBrain
            GroqBrain().warmup()
        except Exception as e:
            log.error("LLM warmup failed: %s", e)

    await asyncio.get_running_loop().run_in_executor(None, _warm)
    log.info("✅ Models warm - ready for calls")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _warm_models()
    yield


app = FastAPI(title="Jarvis Twilio Call Bot", lifespan=lifespan)

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", ""}

# Twilio has to FETCH the brochure over the public internet to attach it to a
# WhatsApp message, so it is served from this app through the same ngrok tunnel
# that Twilio already reaches for /voice.
BROCHURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "brochure.pdf")


# ------------------------------------------------------------------ helpers

def _public_base():
    return (getattr(settings, "public_base_url", "") or "").strip().rstrip("/")


def _host_from_request(request: Request):
    host = request.headers.get("host")
    if not host:
        return None
    h = host.split(":")[0].lower()
    if h in LOCAL_HOSTS:
        return None
    return h


def get_ws_url(request: Request):
    host = _host_from_request(request)
    base = host or _public_base().replace("https://", "")
    if not base:
        raise RuntimeError("No public URL configured")
    return f"wss://{base}/media"


# ----------------------------------------------------------------- brochure

@app.get("/brochure.pdf")
async def brochure_file():
    """Public URL for the brochure. Twilio fetches this when sending WhatsApp,
    and the caller opens it directly from the SMS link."""
    if not os.path.exists(BROCHURE_PATH):
        log.error("Brochure not found at %s", BROCHURE_PATH)
        return Response(status_code=404, content="brochure not found")
    return FileResponse(BROCHURE_PATH, media_type="application/pdf",
                        filename="Karthipuram-Brochure.pdf")


# ------------------------------------------------------------------ webhook

@app.api_route("/voice", methods=["GET", "POST"])
async def voice_webhook(request: Request):
    log.info("🔥 /voice webhook HIT")

    form = await request.form()
    caller = form.get("From", "unknown")

    log.info("📞 Incoming call from=%s", caller)

    response = VoiceResponse()

    # <Connect><Stream> is BIDIRECTIONAL: Twilio forks the caller's audio to us
    # AND plays back any media frames we send. <Start><Stream> is fork-only --
    # outbound media sent on it is silently discarded.
    connect = Connect()
    stream = connect.stream(url=get_ws_url(request))
    # The WebSocket start event does not carry the caller's number, and the
    # brochure has to be sent somewhere.
    stream.parameter(name="from", value=caller)
    response.append(connect)

    # No <Pause> needed: <Connect> blocks for the life of the stream.

    twiml = str(response)
    log.info("TwiML: %s", twiml)
    return Response(content=twiml, media_type="application/xml")


# ------------------------------------------------------------------ websocket

@app.websocket("/media")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info("🔌 WebSocket connected")

    bridge: MediaStreamBridge | None = None
    send_lock = asyncio.Lock()
    state = {"open": True}

    async def send_json(data: dict):
        # Once the socket is gone, stop trying. Without this one dead call
        # produces hundreds of identical warnings and drowns the real error.
        if not state["open"]:
            return
        async with send_lock:
            try:
                await ws.send_text(json.dumps(data))
            except Exception as e:
                state["open"] = False
                if bridge:
                    bridge.ws_closed = True
                log.warning("send_json failed, socket is closed: %s",
                            e or type(e).__name__)

    start_task: asyncio.Task | None = None

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            event = data.get("event")

            if event == "start":
                log.info("🚀 Stream started")

                bridge = MediaStreamBridge(send_json)

                # CRITICAL: on_start connects Deepgram and then SPEAKS THE
                # GREETING, which streams ~10 s of audio in real time. Awaiting
                # it here blocks this receive loop for that whole time, so
                # Twilio's 50 messages/second are never drained, backpressure
                # builds, and Twilio drops the call mid-greeting. Run it as a
                # task so the loop keeps reading.
                start_task = asyncio.create_task(
                    bridge.on_start(data.get("start", {})))

            elif event == "media":
                if bridge:
                    payload = data["media"]["payload"]
                    await bridge.on_media(payload)

                    # METADATA said end_conversation - the goodbye line has
                    # already been spoken, so close the call.
                    if bridge.should_end:
                        log.info("👋 Ending call (end_conversation)")
                        break

            elif event == "mark":
                # Twilio plays out our audio and echoes the mark back when the
                # caller has actually heard it. This is the ONLY reliable
                # "the bot has finished speaking" signal - the send loop
                # finishes seconds earlier, while audio is still queued.
                if bridge:
                    bridge.on_mark((data.get("mark") or {}).get("name", ""))

            elif event == "stop":
                log.info("🛑 Stream stopped")
                break

    except WebSocketDisconnect:
        log.info("❌ WebSocket disconnected")

    except Exception as e:
        log.exception("WebSocket error: %s\n%s", e, traceback.format_exc())

    finally:
        state["open"] = False
        if bridge:
            bridge.ws_closed = True
        if start_task and not start_task.done():
            start_task.cancel()
        if bridge:
            try:
                await bridge.on_stop("stream-closed")
            except Exception as e:
                log.warning("on_stop failed: %s", e)
        try:
            await ws.close()
        except Exception:
            pass


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
