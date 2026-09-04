import asyncio
import logging
import time
import os
import audioop

import numpy as np
from kokoro_onnx import Kokoro

from config import settings

log = logging.getLogger("jarvis.tts")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ONNX_PATH = os.path.join(MODEL_DIR, "kokoro-v1_0.onnx")
VOICES_PATH = os.path.join(MODEL_DIR, "voices.bin")

KOKORO_VOICE = getattr(settings, "kokoro_voice", "af_heart")

_kokoro = None


# ------------------------------------------------ LOAD MODEL
def _get_kokoro():
    global _kokoro

    if _kokoro is None:
        if not os.path.exists(ONNX_PATH) or not os.path.exists(VOICES_PATH):
            raise FileNotFoundError("Kokoro model files missing")

        _kokoro = Kokoro(ONNX_PATH, VOICES_PATH)
        log.info("✅ Kokoro TTS loaded")

    return _kokoro


# ------------------------------------------------ SYNTHESIS
def _synthesize_sync(text: str) -> bytes:
    kokoro = _get_kokoro()

    samples, sr = kokoro.create(
        text,
        voice=KOKORO_VOICE,
        speed=1.0,
        lang="en-us"
    )

    print(f"🎧 samples={len(samples)} sr={sr}")

    if len(samples) == 0:
        return b""

    # 🔥 Convert float → int16 PCM
    pcm16 = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()

    # 🔥 Resample to 8kHz using audioop (CRITICAL)
    pcm8k, _ = audioop.ratecv(pcm16, 2, 1, sr, 8000, None)

    return pcm8k


# ------------------------------------------------ WARMUP
def warmup():
    """Load the ONNX model and prime espeak/phonemizer BEFORE the phone rings.

    Lazy loading put the 1.0s model load inside the FIRST caller's greeting on
    the 2026-09-04 call (11:08:44.886 -> 11:08:45.880). Call this at startup.
    """
    try:
        _get_kokoro()
        _synthesize_sync("Ready.")
        log.info("✅ Kokoro warmed up")
    except Exception as e:
        log.error("Kokoro warmup failed: %s", e)


# ------------------------------------------------ STREAMER
class KokoroStreamer:

    async def stream_to_twilio(self, text, send_media, send_mark=None, cancelled=None):
        """Speak `text` to Twilio. `text` is a string, or a list of sentences.

        Sentences are PIPELINED: sentence N+1 is synthesised in the executor
        while sentence N's frames are being paced out. The caller hears the
        first word after roughly one sentence of synthesis (~0.4s) instead of
        after the whole reply - measured at 1.56s for a 5.1s answer and 2.20s
        for a 7.3s one on the 2026-09-04 call, all of it dead air.
        """
        sentences = [text] if isinstance(text, str) else list(text or [])
        sentences = [s for s in sentences if s and s.strip()]
        if not sentences:
            return 0

        loop = asyncio.get_running_loop()

        frame_size = 160           # 20ms @ 8kHz μ-law
        # Deadline-based pacing. asyncio.sleep(0.02) is unreliable on Windows
        # (~15ms timer granularity), so a naive per-frame sleep drifts slower
        # than realtime and starves Twilio's jitter buffer -> choppy/no audio.
        # We also let a small burst run ahead so playback starts immediately.
        BURST_FRAMES = 20          # ~400ms of audio sent up front
        sent = 0
        # ONE clock for the whole utterance. Restarting it per sentence would
        # put a fresh 400ms burst - and an audible seam - at every full stop.
        start_t = None

        # Sentence 0 is already synthesising before we enter the loop.
        ahead = loop.run_in_executor(None, _synthesize_sync, sentences[0])
        try:
            for i, sentence in enumerate(sentences):
                pcm = await ahead
                # Start the NEXT one before sending this one, not after.
                ahead = (loop.run_in_executor(None, _synthesize_sync, sentences[i + 1])
                         if i + 1 < len(sentences) else None)

                if not pcm:
                    log.error("❌ No PCM generated for: %r", sentence[:60])
                    continue

                mulaw = audioop.lin2ulaw(pcm, 2)
                if start_t is None:
                    start_t = time.monotonic()

                for j in range(0, len(mulaw), frame_size):
                    if cancelled and await cancelled():
                        log.info("🛑 TTS cancelled after %d frames", sent)
                        return sent

                    frame = mulaw[j:j + frame_size]

                    if len(frame) < frame_size:
                        frame += b"\xff" * (frame_size - len(frame))  # μ-law silence

                    await send_media(frame)
                    sent += 1

                    if sent > BURST_FRAMES:
                        # Sleep until this frame's scheduled wall-clock slot.
                        target = start_t + (sent - BURST_FRAMES) * 0.02
                        delay = target - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
        finally:
            if ahead is not None:
                ahead.cancel()

        if not sent:
            log.error("❌ No PCM generated")
            return 0

        log.info("🔊 TTS sent %d frames (%.1fs audio, %d sentence(s))",
                 sent, sent * 0.02, len(sentences))

        if send_mark:
            await send_mark(f"tts-done-{sent}")

        return sent


# Alias
ElevenLabsStreamer = KokoroStreamer
