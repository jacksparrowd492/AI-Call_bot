import re
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
# Kokoro phonemises through espeak, and espeak is fussy about characters that
# are not plain ASCII. A curly apostrophe in "I’m sorry" is the whole reason
# every reply logged "words count mismatch on 100.0% of the lines" - espeak
# split the word and the phoneme count stopped matching. Left alone it also
# reads stray symbols aloud. Normalise once, here, so every spoken line goes
# out clean whether it came from the model or from a canned constant.
_SPEECH_MAP = {
    u"\u2019": u"'", u"\u2018": u"'", u"\u02bc": u"'",
    u"\u201c": u'"', u"\u201d": u'"',
    u"\u2013": u"-", u"\u2014": u"-", u"\u2212": u"-",
    u"\u2026": u"...", u"\u00a0": u" ", u"\u200b": u"",
    u"&": u" and ", u"\u20b9": u" rupees ", u"\u00b0": u" degrees ",
    u"%": u" percent ", u"\u00bd": u" half ",
}
_EMOJI_RE = re.compile(
    u"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\u2b00-\u2bff]")


def _normalise_for_speech(text: str) -> str:
    """Plain ASCII the phonemiser can handle, and nothing it would read aloud
    as a symbol."""
    out = str(text or "")
    for bad, good in _SPEECH_MAP.items():
        out = out.replace(bad, good)
    out = _EMOJI_RE.sub(" ", out)
    # Anything still outside ASCII would reach espeak as an unknown glyph.
    out = out.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s{2,}", " ", out).strip()


def _synthesize_sync(text: str) -> bytes:
    kokoro = _get_kokoro()

    text = _normalise_for_speech(text)
    if not text:
        return b""

    # speed comes from TTS_SPEED. 1.05 is imperceptibly faster to listen to and
    # takes 5% less wall-clock time to say - and on a phone call the length of
    # the reply is latency for whatever the caller wants to say next.
    speed = min(2.0, max(0.5, float(getattr(settings, "tts_speed", 1.0) or 1.0)))

    samples, sr = kokoro.create(
        text,
        voice=KOKORO_VOICE,
        speed=speed,
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


# ------------------------------------------------ SENTENCE SOURCES
# stream_to_twilio takes a string, a list of sentences, OR an async iterator of
# sentences. The last one is what lets the bot start speaking sentence one
# while Groq is still writing sentence two: bridge.py hands over a generator
# fed by the live completion instead of a finished list.


async def _aiter_list(items):
    for s in items:
        if s and s.strip():
            yield s


async def _afilter(source):
    """Same blank-dropping over an async source, and closes it when done."""
    try:
        async for s in source:
            if s and s.strip():
                yield s
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                pass


def _as_source(text):
    if text is None:
        return _aiter_list([])
    if isinstance(text, str):
        return _aiter_list([text])
    if hasattr(text, "__aiter__"):
        return _afilter(text)
    return _aiter_list(list(text))


async def _next_sentence(source):
    """The next sentence, or None when the source is finished."""
    try:
        return await source.__anext__()
    except StopAsyncIteration:
        return None


# ------------------------------------------------ STREAMER
class KokoroStreamer:

    async def stream_to_twilio(self, text, send_media, send_mark=None, cancelled=None):
        """Speak `text` to Twilio.

        `text` is a string, a list of sentences, or an ASYNC ITERATOR of
        sentences that are still being generated.

        Sentences are PIPELINED: sentence N+1 is synthesised in the executor
        while sentence N's frames are being paced out. The caller hears the
        first word after roughly one sentence of synthesis (~0.4s) instead of
        after the whole reply - measured at 1.56s for a 5.1s answer and 2.20s
        for a 7.3s one on the 2026-09-04 call, all of it dead air.

        With an async iterator the same pipeline reaches one step further back:
        a sentence is synthesised the moment the model finishes writing it, so
        the wait for the REST of the completion (including the METADATA block,
        which is never spoken) happens while the caller is already listening.
        """
        source = _as_source(text)
        loop = asyncio.get_running_loop()

        frame_size = 160           # 20ms @ 8kHz mu-law
        # Deadline-based pacing. asyncio.sleep(0.02) is unreliable on Windows
        # (~15ms timer granularity), so a naive per-frame sleep drifts slower
        # than realtime and starves Twilio's jitter buffer -> choppy/no audio.
        # We also let a small burst run ahead so playback starts immediately.
        BURST_FRAMES = 20          # ~400ms of audio sent up front
        sent = 0
        spoken = 0
        # ONE clock for the whole utterance. Restarting it per sentence would
        # put a fresh 400ms burst - and an audible seam - at every full stop.
        start_t = None

        ahead = None                                   # synthesis in flight
        pending = asyncio.ensure_future(_next_sentence(source))   # next sentence
        exhausted = False

        def pump():
            """Start synthesising the next sentence the moment it is available.

            Called between frames, so waiting on the model never stops audio
            going out - and synthesis of the next sentence overlaps with
            playback of this one exactly as it did with a plain list.
            """
            nonlocal ahead, pending, exhausted
            if ahead is not None or exhausted or pending is None:
                return
            if not pending.done() or pending.cancelled():
                return
            try:
                nxt = pending.result()
            except Exception as e:
                log.error("Sentence source failed: %s", e)
                pending, exhausted = None, True
                return
            if nxt is None:
                pending, exhausted = None, True
                return
            ahead = loop.run_in_executor(None, _synthesize_sync, nxt)
            pending = asyncio.ensure_future(_next_sentence(source))

        try:
            while True:
                if ahead is None:
                    if exhausted or pending is None:
                        break
                    # Nothing to send and nothing synthesising: the model is
                    # still writing. This is the only place we wait on it.
                    nxt = await pending
                    if nxt is None:
                        exhausted = True
                        break
                    ahead = loop.run_in_executor(None, _synthesize_sync, nxt)
                    pending = asyncio.ensure_future(_next_sentence(source))

                pcm = await ahead
                ahead = None
                spoken += 1
                pump()

                if not pcm:
                    log.error("❌ No PCM generated for one sentence")
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
                        frame += b"\xff" * (frame_size - len(frame))  # mu-law silence

                    await send_media(frame)
                    sent += 1
                    pump()

                    if sent > BURST_FRAMES:
                        # Sleep until this frame's scheduled wall-clock slot.
                        target = start_t + (sent - BURST_FRAMES) * 0.02
                        delay = target - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
        finally:
            if ahead is not None:
                ahead.cancel()
            if pending is not None:
                pending.cancel()
            try:
                await source.aclose()
            except Exception:
                pass

        if not sent:
            log.error("❌ No PCM generated")
            return 0

        log.info("🔊 TTS sent %d frames (%.1fs audio, %d sentence(s))",
                 sent, sent * 0.02, spoken)

        if send_mark:
            await send_mark(f"tts-done-{sent}")

        return sent


# Alias
ElevenLabsStreamer = KokoroStreamer
