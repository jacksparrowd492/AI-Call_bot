"""Deepgram streaming STT with proper turn-taking and echo suppression.

Two failures this module exists to prevent, both seen on real calls:

1. THE BOT ANSWERING ITSELF. Twilio echoes the bot's own TTS back up the
   media stream. Deepgram happily transcribes it ("Hello?", "And", "Hold on.")
   and the bot replies to its own voice. Fixed by muting the audio feed while
   the bot speaks (plus an echo tail after playback ends) and by discarding any
   transcript that arrives inside that window.

2. CUTTING THE CALLER OFF. `endpointing` alone ends the turn after a short
   silence, so a caller who pauses to think mid-sentence gets answered
   half-way through. Fixed by treating endpointing as "a phrase ended", not
   "the turn ended": finalized phrases are accumulated and only flushed on
   Deepgram's UtteranceEnd (utterance_end_ms) or after a debounce with no new
   speech. SpeechStarted cancels a pending flush, so resuming mid-thought
   simply continues the same turn.
"""
import asyncio
import json
import logging
import re
import ssl
import time

import websockets

from config import settings

log = logging.getLogger("jarvis.stt")

# Deepgram closes a stream that receives no audio for ~10 s:
#   "No audio received recently on this stream; send audio or a KeepAlive"
# The bot speaks for longer than that - and while it speaks we deliberately
# send NO audio - so a KeepAlive is mandatory, not optional.
KEEPALIVE_SECONDS = 5

# Single words that are never a real turn: filler, breath noise and the tail of
# the bot's own audio. "yes"/"no"/"ok" are NOT here - those are real answers.
FILLERS = {
    "a", "ah", "aha", "and", "eh", "er", "erm", "hm", "hmm", "huh", "mm",
    "mmm", "mhm", "oh", "so", "the", "uh", "uhh", "um", "umm", "you",
}

DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=mulaw"
    "&sample_rate=8000"
    "&channels=1"
    f"&model={settings.deepgram_model}"
    "&interim_results=true"          # required for utterance_end_ms
    "&punctuate=true"
    "&smart_format=true"
    "&numerals=true"
    "&filler_words=false"
    "&vad_events=true"               # SpeechStarted, so we can cancel a flush
    f"&endpointing={settings.deepgram_endpointing_ms}"
    f"&utterance_end_ms={settings.deepgram_utterance_end_ms}"
)


class DeepgramStream:
    def __init__(self, on_utterance):
        self.on_utterance = on_utterance
        self._ws = None
        self._closed = False
        self._tasks = []

        # --- current turn being assembled ---
        self._parts = []
        self._confidence = 0.0
        self._flush_task = None
        self._lock = asyncio.Lock()

        # --- echo suppression ---
        self._muted = False
        self._mute_until = 0.0

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        try:
            log.info("🔌 Connecting to Deepgram... (endpointing=%sms, "
                     "utterance_end=%sms)", settings.deepgram_endpointing_ms,
                     settings.deepgram_utterance_end_ms)

            self._ws = await websockets.connect(
                DEEPGRAM_WS_URL,
                extra_headers={"Authorization": f"Token {settings.deepgram_api_key}"},
                ssl=ssl.create_default_context(),
                open_timeout=30,
                ping_interval=5,
                ping_timeout=20,
                close_timeout=10,
                max_size=None,
            )

            log.info("✅ Deepgram connected")

            self._tasks.append(asyncio.create_task(self._receiver()))
            self._tasks.append(asyncio.create_task(self._keepalive()))

        except Exception as e:
            log.exception("❌ Failed to connect to Deepgram: %s", e)
            raise

    async def close(self):
        if self._closed:
            return
        self._closed = True

        self._cancel_flush()
        for t in self._tasks:
            t.cancel()

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        log.info("🛑 Deepgram connection closed")

    # ------------------------------------------------------ echo suppression
    @property
    def ignoring(self) -> bool:
        """True while the bot is speaking, and for the echo tail after it."""
        return self._muted or time.monotonic() < self._mute_until

    def mute(self):
        """Bot is about to speak: stop listening and throw away any half-turn,
        so the caller's audio and the bot's echo can never be stitched together."""
        self._muted = True
        self._cancel_flush()
        if self._parts:
            log.debug("Dropping partial turn on mute: %r", " ".join(self._parts))
        self._parts, self._confidence = [], 0.0

    def unmute(self, tail_s: float = 0.0):
        """Bot has finished speaking. `tail_s` keeps the mic closed a little
        longer, because Twilio's jitter buffer is still draining the last of
        the bot's audio back at us."""
        self._muted = False
        self._mute_until = time.monotonic() + max(0.0, tail_s)

    async def send_audio(self, audio: bytes):
        # Not just "don't transcribe it" - don't send it at all. Deepgram never
        # hears the bot, so no echo can reach the endpointer. KeepAlive holds
        # the socket open through the silence.
        if self.ignoring or not self._ws or self._closed:
            return
        try:
            await self._ws.send(audio)
        except Exception as e:
            if not self._closed:
                log.error("❌ send_audio failed: %s", e)

    # ------------------------------------------------------------- internals
    async def _keepalive(self):
        try:
            while not self._closed:
                await asyncio.sleep(KEEPALIVE_SECONDS)
                if self._closed or not self._ws:
                    return
                try:
                    await self._ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    return
        except asyncio.CancelledError:
            pass

    def _cancel_flush(self):
        """Cancel a PENDING flush - never one that is already answering.

        The reply runs inside _flush_task (see _flush), and bridge._speak()
        opens with stt.mute(), which lands here. Cancelling unconditionally
        meant _speak cancelled the very task that was running it:
        CancelledError landed on the TTS await, unwound into
        _flush_after_debounce's `except CancelledError: pass`, and the caller
        heard nothing at all. On the 2026-09-04 call that silently killed
        8 of the 10 answers - every turn that ended via the debounce path.
        """
        t = self._flush_task
        self._flush_task = None
        if t and not t.done() and t is not asyncio.current_task():
            t.cancel()

    def _schedule_flush(self):
        """The caller stopped talking. Wait a beat before answering - people
        pause mid-sentence, and answering into that pause is what made the bot
        feel like it was interrupting."""
        self._cancel_flush()
        self._flush_task = asyncio.create_task(self._flush_after_debounce())

    async def _flush_after_debounce(self):
        # Cancelled during the WAIT is routine: the caller resumed talking and
        # this half-turn is being folded into the next one.
        try:
            await asyncio.sleep(settings.turn_debounce_ms / 1000.0)
        except asyncio.CancelledError:
            return

        # Cancelled during the ANSWER is a bug, and the symptom is dead air on
        # a live call. It must never be swallowed silently again.
        try:
            await self._flush("debounce")
        except asyncio.CancelledError:
            log.warning("Turn cancelled WHILE ANSWERING - the caller heard "
                        "nothing. Something cancelled the flush task mid-reply.")
            raise

    @staticmethod
    def _is_junk(text: str, confidence: float) -> str:
        """Return a reason to drop this turn, or '' to keep it."""
        bare = re.sub(r"[^\w\s]", "", text).strip().lower()
        if not bare:
            return "empty"
        if len(re.sub(r"[^a-z0-9]", "", bare)) < 2:
            return "too short"
        words = bare.split()
        if len(words) == 1 and words[0] in FILLERS:
            return f"filler word {words[0]!r}"
        if all(w in FILLERS for w in words):
            return "all filler words"
        if confidence and confidence < settings.stt_min_confidence:
            return f"low confidence {confidence:.2f}"
        return ""

    async def _flush(self, why: str):
        # Detach first. From here on this task is ANSWERING, not waiting, so
        # _cancel_flush() must not be able to reach it: mute() calls it at the
        # top of every _speak, and so does a trailing UtteranceEnd arriving
        # while the reply is still being spoken.
        self._flush_task = None

        async with self._lock:
            parts, confidence = self._parts, self._confidence
            self._parts, self._confidence = [], 0.0

        if not parts:
            return

        text = " ".join(p for p in parts if p).strip()
        text = re.sub(r"\s{2,}", " ", text)

        if self.ignoring:
            log.info("🔇 Ignored (bot speaking): %r", text[:80])
            return

        junk = self._is_junk(text, confidence)
        if junk:
            log.info("🗑 Ignored (%s): %r", junk, text[:80])
            return

        log.info("🗣 User said (%s, conf=%.2f): %s", why, confidence, text)
        try:
            await self.on_utterance(text)
        except Exception as e:
            log.exception("on_utterance failed: %s", e)

    async def _receiver(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "Results":
                    await self._on_results(msg)

                elif mtype == "UtteranceEnd":
                    # Deepgram is confident the caller has actually stopped.
                    # This is the real end of a turn.
                    self._cancel_flush()
                    await self._flush("utterance-end")

                elif mtype == "SpeechStarted":
                    # They started again - this is one turn, not two.
                    self._cancel_flush()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._closed:
                log.exception("❌ Deepgram receiver error: %s", e)

    async def _on_results(self, msg):
        alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
        text = (alt.get("transcript") or "").strip()
        confidence = float(alt.get("confidence") or 0.0)
        is_final = bool(msg.get("is_final"))
        speech_final = bool(msg.get("speech_final"))

        if self.ignoring:
            if text:
                log.debug("🔇 Echo suppressed: %r", text[:60])
            return

        if not text:
            return

        if not is_final:
            # Still talking. Hold off on any pending flush.
            self._cancel_flush()
            return

        async with self._lock:
            self._parts.append(text)
            self._confidence = max(self._confidence, confidence)

        if speech_final:
            # A phrase ended. That is NOT necessarily the end of the turn, so
            # wait out the debounce; UtteranceEnd or new speech may beat it.
            self._schedule_flush()
        else:
            self._cancel_flush()
