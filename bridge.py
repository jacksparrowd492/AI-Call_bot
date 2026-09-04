import asyncio
import logging
import time

from config import settings
from stt import DeepgramStream
from tts import ElevenLabsStreamer
from llm import GroqBrain, EXIT_LINE
from rag.retriever import retrieve
from handoff_intent import wants_human, extract_time

log = logging.getLogger("jarvis.bridge")

# Keep the greeting SHORT. Every second of it is a second the caller cannot
# talk, and the old one was 10.8 seconds long.
GREETING = ("Hello! Thank you for calling Karthipuram. "
            "I'm Jarvis. How can I help you today?")
RECORDING_NOTICE = "This call is recorded for quality purposes."
FALLBACK_LINE = "Sorry, I did not catch that. Could you please say it again?"

# Spoken the moment the caller asks for a person. We do NOT transfer the live
# call - we schedule a callback and confirm it in writing.
HANDOFF_ASK_TIME = ("Of course. I'll arrange for one of our sales specialists to "
                    "call you back. What time works best for you?")
HANDOFF_CONFIRMED = ("Perfect. Our specialist will call you then, and I'll send "
                     "you a confirmation on WhatsApp or SMS right away.")


class MediaStreamBridge:
    def __init__(self, send_json):
        self.send_json = send_json
        self.stream_sid = None
        self.call_sid = None
        self.caller_number = "unknown"

        self.brain = GroqBrain()
        self.tts = ElevenLabsStreamer()
        self.stt = DeepgramStream(self._on_utterance)

        self.history = []
        self.speaking = False

        # One turn at a time. A second utterance must not start a second LLM
        # call and a second TTS stream on top of the first.
        self._reply_lock = asyncio.Lock()

        # Twilio echoes a mark back once the caller has actually HEARD the
        # audio we queued. That, not the end of our send loop, is when the bot
        # stops speaking and the mic may reopen.
        self._playback_done = asyncio.Event()
        self._mark_seq = 0
        self._expected_mark = None

        # Set by server.py when the Twilio socket dies, so the TTS loop stops
        # pushing frames into a closed connection.
        self.ws_closed = False
        self.stt_ready = False

        # Set when METADATA says end_conversation - server.py closes the call.
        self.should_end = False
        self.brochure_sent = False

        # Callback ("connect me to a human") state.
        self.callback_requested = False
        self.callback_time = None
        self.awaiting_callback_time = False
        self.callback_confirmed = False
        self.lead_name = None
        self.lead_requirement = None

    # ------------------------------------------------ lifecycle
    async def on_start(self, start: dict):
        self.stream_sid = start.get("streamSid")
        self.call_sid = start.get("callSid")

        params = start.get("customParameters") or {}
        self.caller_number = params.get("from") or self.caller_number

        log.info("Call started sid=%s from=%s", self.call_sid, self.caller_number)

        try:
            await self.stt.start()
            self.stt_ready = True
        except Exception as e:
            log.error("Deepgram failed to start: %s", e)

        greeting = GREETING
        if settings.announce_recording:
            greeting = f"{RECORDING_NOTICE} {GREETING}"

        print("\n🤖 BOT:", greeting)

        await self._clear_outbound_audio()
        await self._speak(greeting)

    async def on_media(self, payload_b64: str):
        # Audio can arrive before Deepgram finishes connecting. Dropping a few
        # frames of the caller clearing their throat is fine; crashing is not.
        if self.ws_closed or not self.stt_ready:
            return

        import base64
        audio = base64.b64decode(payload_b64)
        await self.stt.send_audio(audio)

    async def on_stop(self, reason=""):
        log.info("🛑 Stream stopped: %s", reason)
        self.ws_closed = True
        await self.stt.close()

        # The caller has hung up. A callback confirmation carries the brochure
        # with it, so send that when one was requested and the plain brochure
        # otherwise. Most callers hang up rather than saying goodbye, so this
        # has to happen here and not only on end_conversation.
        if self.callback_requested:
            await self._confirm_callback()
        if not self.brochure_sent:
            await self._deliver_brochure()

    # ------------------------------------------------ STT
    async def _on_utterance(self, text: str):
        # The mic is already shut while the bot talks; this is belt-and-braces
        # for a transcript that was in flight when playback started.
        if self.speaking or self.stt.ignoring:
            log.info("🔇 Ignored while speaking: %r", text[:60])
            return

        if self._reply_lock.locked():
            log.info("⏳ Still answering the previous turn - ignoring: %r", text[:60])
            return

        async with self._reply_lock:
            await self._handle_utterance(text)

    async def _handle_utterance(self, text: str):
        log.info("🧠 User: %s", text)

        if not text:
            await self._speak(FALLBACK_LINE)
            return

        # The caller asking for a person is the highest-intent moment on the
        # call. Handle it deterministically, before the LLM, so a slow or
        # malformed completion can never lose it.
        if self.awaiting_callback_time:
            when = extract_time(text)
            if when:
                self.callback_time = when
                log.info("📅 Preferred callback time: %s", when)
            self.awaiting_callback_time = False
            await self._speak(HANDOFF_CONFIRMED)
            return

        if wants_human(text):
            await self._request_callback(text)
            return

        # Greetings and acknowledgements need no knowledge base and no LLM.
        # Sending "Hello?" to Groq cost half a second and, worse, dragged a
        # random KB entry in as "context" for the model to answer from.
        canned = self.brain.smalltalk_guard(text)
        if canned:
            log.info("💬 Smalltalk (no RAG, no LLM): %r", text[:60])
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": canned})
            print("🤖 BOT:", canned)
            await self._clear_outbound_audio()
            await self._speak(canned)
            if canned == EXIT_LINE:
                await self._handle_metadata({"end_conversation": True})
            return

        await self._reply(text)

    # ------------------------------------------------ reply
    async def _reply(self, user_text: str):
        # Both of these are BLOCKING and used to run straight on the event
        # loop: Chroma's query plus MiniLM's ONNX embedding, and reply_stream,
        # which is a sync generator doing blocking HTTP to Groq. For the
        # 280-570ms they took, the server could not drain Twilio's 50 inbound
        # media messages a second, pace outbound audio frames, or service the
        # Deepgram receiver. They belong in a worker thread.
        loop = asyncio.get_running_loop()

        context = await loop.run_in_executor(None, retrieve, user_text)

        self.history.append({"role": "user", "content": user_text})

        def _generate():
            return [s for s in self.brain.reply_stream(self.history, user_text, context)
                    if s.strip()]

        sentences = await loop.run_in_executor(None, _generate)

        for sentence in sentences:
            print("🤖 BOT:", sentence)

        if not sentences:
            log.warning("Brain returned nothing - speaking the fallback line")
            sentences = [FALLBACK_LINE]

        await self._clear_outbound_audio()
        # Hand over the SENTENCES, not the joined string: tts.py pipelines them
        # so the caller hears sentence one while sentence two is synthesising.
        await self._speak(sentences)

        self.history.append({"role": "assistant", "content": " ".join(sentences)})

        # METADATA is parsed from the raw completion, never from the spoken text.
        await self._handle_metadata(self.brain.parse_extra(self.brain.last_raw))

    # ------------------------------------------------ callback to a human
    async def _request_callback(self, text: str):
        """Caller asked for a person. Schedule a callback instead of a live
        transfer, then confirm it over WhatsApp/SMS when the call ends."""
        log.info("🙋 Callback requested: %r", text)
        self.callback_requested = True

        when = extract_time(text)
        if when:
            self.callback_time = when
            log.info("📅 Preferred callback time: %s", when)
            await self._speak(HANDOFF_CONFIRMED)
        else:
            self.awaiting_callback_time = True
            await self._speak(HANDOFF_ASK_TIME)

        self.history.append({"role": "user", "content": text})
        await self._record_callback()

    async def _record_callback(self):
        """Persist first, notify second - a failed SMS must not lose the lead."""
        if not self.call_sid:
            return
        from tools import callbacks

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: callbacks.record(
                call_sid=self.call_sid,
                caller_number=self.caller_number,
                name=self.lead_name,
                requirement=self.lead_requirement,
                preferred_time=self.callback_time,
            ))

    async def _confirm_callback(self):
        """Send the caller their confirmation and alert the sales team."""
        if self.callback_confirmed or not self.callback_requested:
            return
        if not self.caller_number or self.caller_number == "unknown":
            log.warning("No caller number - cannot confirm the callback")
            return

        from tools import callbacks
        from tools.whatsapp import deliver_callback_confirmation, notify_agent

        loop = asyncio.get_running_loop()

        # One message to the caller: callback confirmation + the brochure.
        channel = await loop.run_in_executor(
            None,
            lambda: deliver_callback_confirmation(
                self.caller_number, self.lead_name, self.callback_time,
                include_brochure=not self.brochure_sent))

        if channel in ("whatsapp", "sms"):
            self.callback_confirmed = True
            self.brochure_sent = True      # it rode along with the confirmation
            log.info("📅 Callback confirmed to %s via %s",
                     self.caller_number, channel)
        else:
            log.error("📅 Callback confirmation FAILED for %s", self.caller_number)

        alerted = await loop.run_in_executor(
            None,
            lambda: notify_agent(self.caller_number, self.lead_name,
                                 self.lead_requirement, self.callback_time))

        await loop.run_in_executor(
            None,
            lambda: callbacks.record(
                call_sid=self.call_sid,
                caller_number=self.caller_number,
                name=self.lead_name,
                requirement=self.lead_requirement,
                preferred_time=self.callback_time,
                caller_notified=channel,
                agent_notified=alerted,
            ))

    # ------------------------------------------------ metadata
    async def _handle_metadata(self, meta: dict):
        if not meta:
            return

        log.info("METADATA: %s", meta)

        if meta.get("name"):
            self.lead_name = meta["name"]
        if meta.get("requirement"):
            self.lead_requirement = meta["requirement"]

        # Second layer: the model spotted a handoff the keywords missed.
        if meta.get("handoff") and not self.callback_requested:
            log.info("🙋 Callback requested via METADATA handoff=true")
            self.callback_requested = True
            await self._record_callback()

        if meta.get("whatsapp_wanted") and not self.brochure_sent:
            await self._deliver_brochure()

        if meta.get("end_conversation"):
            # The spoken exit line promises the brochure, so send it before
            # the call drops.
            if not self.brochure_sent:
                await self._deliver_brochure()
            log.info("end_conversation=true - closing the call")
            self.should_end = True

    async def _deliver_brochure(self):
        """WhatsApp first; SMS if the number is not on WhatsApp. Sent once."""
        if self.brochure_sent:
            return
        if not self.caller_number or self.caller_number == "unknown":
            log.warning("No caller number - cannot send the brochure")
            return

        from tools.whatsapp import deliver_brochure

        loop = asyncio.get_running_loop()
        channel = await loop.run_in_executor(
            None, deliver_brochure, self.caller_number)

        self.brochure_sent = channel in ("whatsapp", "sms")
        if self.brochure_sent:
            log.info("📄 Brochure sent to %s via %s", self.caller_number, channel)
        else:
            log.error("📄 Brochure delivery FAILED for %s", self.caller_number)

    # ------------------------------------------------ audio
    async def _speak(self, text):
        """`text` is a single line, or the list of sentences making up a reply.

        The list form is what lets tts.py pipeline synthesis; a bare string is
        still accepted, for the canned lines.
        """
        sentences = [text] if isinstance(text, str) else list(text or [])
        sentences = [s for s in sentences if s and s.strip()]

        if self.ws_closed or not sentences:
            return

        self.speaking = True
        self._mark_seq += 1
        self._expected_mark = f"jarvis-{self._mark_seq}"
        self._playback_done.clear()

        # Close the mic BEFORE the first frame goes out. Twilio echoes our own
        # audio back up the media stream and Deepgram cannot tell it from the
        # caller - that is how the bot ended up answering its own voice.
        self.stt.mute()

        sent = 0
        try:
            sent = await self.tts.stream_to_twilio(
                sentences, self._send_media, self._send_mark,
                self._is_cancelled) or 0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("TTS failed: %s", e)
        finally:
            # Frames are QUEUED at Twilio, not played. Wait for Twilio to say
            # the caller has heard them before reopening the mic.
            if sent and not self.ws_closed:
                try:
                    await asyncio.wait_for(
                        self._playback_done.wait(),
                        timeout=settings.mark_timeout_ms / 1000.0)
                except asyncio.TimeoutError:
                    # Do NOT scale this with the reply length. Doing so kept
                    # the mic shut for seconds after the bot had actually
                    # stopped talking, and every question asked in that window
                    # was silently discarded - dead air, from the caller's side.
                    log.warning("No playback mark from Twilio within %dms after "
                                "%.1fs of audio - reopening the mic anyway",
                                settings.mark_timeout_ms, sent * 0.02)
            self.speaking = False
            self.stt.unmute(settings.echo_tail_ms / 1000.0)

    def on_mark(self, name: str):
        """Twilio confirms the caller has finished HEARING a marked chunk."""
        if name and name == self._expected_mark:
            self._playback_done.set()

    async def _is_cancelled(self):
        """Stop mid-sentence if the caller hung up, instead of streaming the
        rest of the reply into a dead socket."""
        return self.ws_closed

    async def _send_media(self, frame: bytes):
        import base64

        if not self.stream_sid or self.ws_closed:
            return

        try:
            await self.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(frame).decode("ascii")},
            })
        except Exception as e:
            log.error("send_media failed: %s", e)

    async def _send_mark(self, name: str):
        if not self.stream_sid or self.ws_closed:
            return
        await self.send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": self._expected_mark or name}
        })

    async def _clear_outbound_audio(self):
        if self.stream_sid and not self.ws_closed:
            await self.send_json({
                "event": "clear",
                "streamSid": self.stream_sid
            })
