"""Pipelining tests for tts.py.

The 2026-09-04 call synthesised the WHOLE reply before sending frame one:
1.56s of dead air for a 5.1s answer, 2.20s for a 7.3s one. stream_to_twilio
now takes a list of sentences and synthesises sentence N+1 while sentence N is
being paced out, so the caller hears the first word after one sentence.

Kokoro is replaced with a stub that sleeps for a known time and returns a known
number of PCM samples, so both the pipelining and the realtime pacing are
checked without the ONNX model.
"""
import asyncio
import os
import sys
import time
import types

_k = types.ModuleType("kokoro_onnx")
_k.Kokoro = object
sys.modules["kokoro_onnx"] = _k

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tts as tts_mod                  # noqa: E402

PASS, FAIL = [], []

SR = 8000
# Kokoro's measured real-time factor on the 2026-09-04 call was ~0.30
# (1.56s of synthesis for 5.1s of audio, 2.20s for 7.3s). The stub matches it,
# because the whole point of pipelining is that synthesis outruns playback.
SYNTH_S = 0.30          # how long the stub "thinks" per sentence
AUDIO_S = 1.00          # how much audio each sentence produces -> 50 frames
FPS = 50                # 20ms frames


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def make_stub(events):
    def fake_synth(text):
        time.sleep(SYNTH_S)                       # blocking, as Kokoro is
        events.append(("synth-done", text, time.monotonic()))
        return b"\x00\x00" * int(SR * AUDIO_S)    # PCM16 @ 8kHz
    return fake_synth


async def main():
    print("\n=== 1. sentence N+1 is synthesised WHILE sentence N is sending ===")
    events = []
    tts_mod._synthesize_sync = make_stub(events)
    streamer = tts_mod.KokoroStreamer()

    frames, marks = [], []

    async def send_media(f):
        frames.append((time.monotonic(), len(f)))

    async def send_mark(n):
        marks.append(n)

    t0 = time.monotonic()
    sent = await streamer.stream_to_twilio(["One.", "Two.", "Three."],
                                           send_media, send_mark)
    elapsed = time.monotonic() - t0

    check("every sentence is sent", sent == 3 * int(AUDIO_S * FPS), str(sent))
    check("all frames are a full 20ms", all(n == 160 for _, n in frames))

    first_audio = frames[0][0] - t0
    check("first audio after ONE synthesis, not three",
          first_audio < SYNTH_S * 2, f"{first_audio:.2f}s")

    # The structural claim, independent of machine speed: sentence 2 finished
    # synthesising before sentence 1 had finished going out.
    synth2 = [t for kind, txt, t in events if txt == "Two."][0]
    end_of_s1 = frames[int(AUDIO_S * FPS) - 1][0]
    check("sentence 2 is ready before sentence 1 ends",
          synth2 < end_of_s1, f"synth2 at {synth2 - t0:.2f}s, s1 ends {end_of_s1 - t0:.2f}s")

    print("\n=== 2. one continuous clock - no seam at the sentence boundary ===")
    gaps = [b - a for (a, _), (b, _) in zip(frames, frames[1:])]
    check("no gap longer than 150ms anywhere in the reply",
          max(gaps) < 0.15, f"max gap {max(gaps) * 1000:.0f}ms")
    boundary = gaps[int(AUDIO_S * FPS) - 1]
    check("the sentence-1/2 boundary is a normal frame gap",
          boundary < 0.15, f"{boundary * 1000:.0f}ms")

    print("\n=== 3. audio is still paced in realtime (Twilio needs this) ===")
    audio_s = sent / FPS
    check("does not dump the whole reply at once",
          elapsed > audio_s - 0.6, f"{elapsed:.2f}s for {audio_s:.2f}s of audio")
    check("does not run slower than realtime",
          elapsed < audio_s + SYNTH_S + 0.5, f"{elapsed:.2f}s for {audio_s:.2f}s")

    print("\n=== 4. the mark is sent once, after the last frame ===")
    check("exactly one mark", len(marks) == 1, str(marks))

    print("\n=== 5. a hung-up caller stops the stream mid-reply ===")
    events2 = []
    tts_mod._synthesize_sync = make_stub(events2)
    frames2 = []

    async def send_media2(f):
        frames2.append(f)

    async def cancelled():
        return len(frames2) >= 10

    sent2 = await streamer.stream_to_twilio(["One.", "Two.", "Three."],
                                            send_media2, None, cancelled)
    check("stops promptly when cancelled", sent2 == 10, str(sent2))
    check("does not keep synthesising the rest of the reply",
          len(events2) <= 2, str([e[1] for e in events2]))

    print("\n=== 6. a bare string still works (the canned lines) ===")
    frames3 = []

    async def send_media3(f):
        frames3.append(f)

    sent3 = await streamer.stream_to_twilio("Just one line.", send_media3)
    check("string input is accepted", sent3 == int(AUDIO_S * FPS), str(sent3))
    check("empty input is a no-op",
          await streamer.stream_to_twilio([], send_media3) == 0)
    check("blank sentences are dropped",
          await streamer.stream_to_twilio(["", "   "], send_media3) == 0)

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
