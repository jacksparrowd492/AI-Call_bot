"""Tests for the two changes made on 2026-09-05: answer sooner, and stop
answering the caller's background.

Nothing here needs Deepgram, Groq, Kokoro, Chroma or Twilio - the noise gate is
driven with generated audio, the model with a fake streaming client, and the
bridge with the same stand-ins tests/test_bridge.py uses.
"""
import asyncio
import math
import os
import struct
import sys
import time
import types

os.environ.setdefault("GROQ_API_KEY", "gsk-test")
os.environ.setdefault("DEEPGRAM_API_KEY", "dg-test")

# Kokoro's ONNX model, Chroma and the embedder are not needed to test control
# flow. rag.retriever is replaced BEFORE bridge imports it.
_k = types.ModuleType("kokoro_onnx")
_k.Kokoro = object
sys.modules["kokoro_onnx"] = _k

_rag = types.ModuleType("rag")
_rag.__path__ = []
_ret = types.ModuleType("rag.retriever")
_ret.retrieve = lambda query, top_k=None: "PROJECT KNOWLEDGE: none"
sys.modules["rag"] = _rag
sys.modules["rag.retriever"] = _ret

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings            # noqa: E402
from audio_utils import NoiseGate      # noqa: E402
import audioop                         # noqa: E402
import tts as tts_mod                  # noqa: E402
import llm as llm_mod                  # noqa: E402
import bridge as bridge_mod            # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------------- audio
FRAME = 160          # 20 ms of mu-law at 8 kHz


def tone(amplitude, frames=1, freq=300.0, phase=0.0):
    """`frames` x 20 ms of a sine wave, mu-law encoded, as Twilio sends it."""
    out = []
    n = FRAME * frames
    for i in range(n):
        out.append(int(amplitude * math.sin(2 * math.pi * freq * (i / 8000.0) + phase)))
    pcm = struct.pack("<%dh" % n, *out)
    return audioop.lin2ulaw(pcm, 2)


def rms_of(mulaw):
    return audioop.rms(audioop.ulaw2lin(mulaw, 2), 2)


def feed(gate, mulaw, frames):
    """Push `frames` 20 ms frames of the same audio through the gate."""
    out = []
    for i in range(frames):
        out.append(gate.process(mulaw[i * FRAME:(i + 1) * FRAME]
                                if len(mulaw) > FRAME else mulaw))
    return out


async def main():
    print("\n=== 1. the noise gate turns a noisy line down, not off ===")
    gate = NoiseGate(ratio=2.5, floor_rms=200, hangover_ms=320, attenuation=0.12)
    noise = tone(150)                      # steady background, RMS ~106
    quiet = feed(gate, noise, 40)
    before, after = rms_of(noise), rms_of(quiet[-1])
    check("background is attenuated, not deleted",
          0 < after < before * 0.3, f"{before} -> {after}")
    check("every frame keeps its 20ms length",
          all(len(f) == FRAME for f in quiet), str({len(f) for f in quiet}))

    print("\n=== 2. the caller's voice passes through untouched ===")
    speech = tone(9000)                    # normal telephone speech
    passed = feed(gate, speech, 10)
    check("speech is not attenuated at all",
          all(f == speech for f in passed[1:]),
          f"{rms_of(speech)} -> {rms_of(passed[-1])}")
    check("the gate reports itself open while speech is playing", gate.is_open)

    print("\n=== 3. the gate opens on the FIRST loud frame ===")
    gate2 = NoiseGate(ratio=2.5, floor_rms=200, hangover_ms=320, attenuation=0.12)
    feed(gate2, tone(150), 30)             # learn the floor
    first = gate2.process(speech[:FRAME])
    check("the first frame of speech is already at full gain",
          first == speech[:FRAME], f"{rms_of(speech[:FRAME])} -> {rms_of(first)}")

    print("\n=== 4. the hangover carries the gate over the gaps between words ===")
    gap = tone(150)
    kept = [gate2.process(gap[:FRAME]) for _ in range(8)]      # 160ms of gap
    check("a 160ms gap does not shut the gate", gate2.is_open)
    check("and the gap is still passed at full gain",
          kept[-1] == gap[:FRAME], str(rms_of(kept[-1])))
    for _ in range(20):                                        # 400ms of gap
        gate2.process(gap[:FRAME])
    check("but a 400ms silence does shut it", not gate2.is_open)

    print("\n=== 5. a loud room raises the bar; a quiet one lowers it ===")
    loud = NoiseGate(ratio=2.5, floor_rms=200, hangover_ms=320)
    feed(loud, tone(2000), 200)            # a genuinely noisy line
    still = NoiseGate(ratio=2.5, floor_rms=200, hangover_ms=320)
    feed(still, tone(80), 200)             # a quiet one
    check("the threshold follows the room",
          loud.threshold() > still.threshold(),
          f"loud {loud.threshold():.0f} vs quiet {still.threshold():.0f}")
    check("the quiet room never drops below the absolute floor",
          still.threshold() >= settings.noise_gate_floor_rms - 1,
          f"{still.threshold():.0f}")

    print("\n=== 6. the gate can be switched off entirely ===")
    off = NoiseGate(enabled=False)
    check("disabled is a pure passthrough",
          off.process(noise[:FRAME]) == noise[:FRAME])

    print("\n=== 7. sentences are spoken while the model is still writing ===")
    # The whole point of the change: the model writes SPEAKABLE_RESPONSE first
    # and METADATA second, so waiting for the completion means waiting through
    # a JSON block the caller never hears.
    RAW = ('SPEAKABLE_RESPONSE:\n'
           'Karthipuram is in Coimbatore. It is a one hundred and ninety acre '
           'township. Would you like the brochure?\n'
           'METADATA:\n'
           '{"name": null, "requirement": "location", "intent_score": 5, '
           '"whatsapp_wanted": false, "handoff": false, "end_conversation": false}')

    class FakeDelta:
        def __init__(self, content):
            self.content = content
            self.reasoning = None

    class FakeChoice:
        def __init__(self, content, finish=None):
            self.delta = FakeDelta(content)
            self.finish_reason = finish

    class FakeEvent:
        def __init__(self, content, finish=None):
            self.choices = [FakeChoice(content, finish)]

    def chunks(raw, size=12):
        return [raw[i:i + size] for i in range(0, len(raw), size)]

    class FakeStream:
        """Yields the completion a few characters at a time, like Groq does."""
        def __init__(self, raw):
            self.parts = chunks(raw)
            self.closed = False

        def __iter__(self):
            for i, p in enumerate(self.parts):
                yield FakeEvent(p, "stop" if i == len(self.parts) - 1 else None)

        def close(self):
            self.closed = True

    brain = llm_mod.GroqBrain()
    brain.client = object()                       # only truthiness is used
    stream = FakeStream(RAW)
    brain._open_stream = lambda messages: stream

    marker = RAW.index("METADATA")
    seen = []
    for sentence in brain.reply_stream_live([], "where is it", "ctx"):
        seen.append((sentence, len(brain.last_raw)))

    spoken = [s for s, _ in seen]
    check("every spoken sentence comes back",
          spoken == ["Karthipuram is in Coimbatore.",
                     "It is a one hundred and ninety acre township.",
                     "Would you like the brochure?"], str(spoken))
    check("the first sentence is spoken before the completion is finished",
          seen[0][1] == 0, "last_raw is only filled at the end")
    check("the METADATA block still arrives intact",
          brain.parse_extra(brain.last_raw).get("requirement") == "location",
          str(brain.parse_extra(brain.last_raw)))
    check("the stream is closed", stream.closed)

    print("\n=== 8. the reply cap still holds, and METADATA survives it ===")
    LONG = ('SPEAKABLE_RESPONSE:\nOne. Two. Three. Four. Five. Six.\n'
            'METADATA:\n{"end_conversation": true}')
    brain2 = llm_mod.GroqBrain()
    brain2.client = object()
    brain2._open_stream = lambda messages: FakeStream(LONG)
    said = list(brain2.reply_stream_live([], "hi", "ctx"))
    check("no more than MAX_REPLY_SENTENCES are spoken",
          len(said) == settings.max_reply_sentences, str(said))
    check("the stream is still drained, so end_conversation is not lost",
          brain2.parse_extra(brain2.last_raw).get("end_conversation") is True,
          brain2.last_raw[-40:])

    print("\n=== 9. an empty completion still says something ===")
    brain3 = llm_mod.GroqBrain()
    brain3.client = object()
    brain3._open_stream = lambda messages: FakeStream("")
    check("dead air is never the answer",
          list(brain3.reply_stream_live([], "hi", "ctx")) == [llm_mod.UNCLEAR_LINE])

    brain4 = llm_mod.GroqBrain()
    brain4.client = object()

    def _boom(messages):
        raise RuntimeError("groq is down")

    brain4._open_stream = _boom
    check("a dead API is spoken, not raised",
          list(brain4.reply_stream_live([], "hi", "ctx")) ==
          ["Sorry, one moment - let me connect you to my team."])

    print("\n=== 10. tts starts speaking sentence one before sentence two exists ===")
    synthed = []

    def fake_synth(text):
        time.sleep(0.05)
        synthed.append(text)
        return b"\x00\x00" * 4000          # 0.5s of PCM16 @ 8kHz

    tts_mod._synthesize_sync = fake_synth

    async def slow_source():
        yield "First sentence."
        await asyncio.sleep(0.40)          # the model is still writing
        yield "Second sentence."

    frames = []

    async def send_media(f):
        frames.append(time.monotonic())

    t0 = time.monotonic()
    sent = await tts_mod.KokoroStreamer().stream_to_twilio(slow_source(),
                                                           send_media)
    check("both sentences are spoken", sent == 50, str(sent))
    check("audio starts before the second sentence is even written",
          frames[0] - t0 < 0.30, f"{frames[0] - t0:.2f}s")
    check("the second sentence is synthesised, not dropped",
          synthed == ["First sentence.", "Second sentence."], str(synthed))

    print("\n=== 11. the bridge speaks a live stream and records it ===")
    class FakeSTT:
        def __init__(self):
            self.events, self.ignoring, self.audio = [], False, []

        def mute(self):
            self.ignoring = True

        def unmute(self, tail_s=0.0):
            self.ignoring = False

        async def start(self):
            pass

        async def close(self):
            pass

        async def send_audio(self, audio):
            self.audio.append(audio)

    class RecordingTTS:
        def __init__(self):
            self.spoken = []

        async def stream_to_twilio(self, text, send_media, send_mark=None,
                                   cancelled=None):
            if isinstance(text, str):
                self.spoken.append(text)
                return 1
            async for s in text:
                self.spoken.append(s)
            return 1

    async def send_json(data):
        pass

    b = bridge_mod.MediaStreamBridge(send_json)
    b.stream_sid = "MZtest"
    b.stt = FakeSTT()
    b.tts = RecordingTTS()
    b.brain.client = object()
    b.brain._open_stream = lambda messages: FakeStream(RAW)

    task = asyncio.create_task(b._reply("where is it"))
    for _ in range(200):
        await asyncio.sleep(0.01)
        if b.speaking:
            b.on_mark(b._expected_mark)
        if task.done():
            break
    await task

    check("the caller hears all three sentences",
          b.tts.spoken == ["Karthipuram is in Coimbatore.",
                           "It is a one hundred and ninety acre township.",
                           "Would you like the brochure?"], str(b.tts.spoken))
    check("history records the whole reply, as one assistant turn",
          [h["role"] for h in b.history] == ["user", "assistant"], str(b.history))
    check("and METADATA was read off the finished completion",
          b.lead_requirement == "location", str(b.lead_requirement))

    print("\n=== 12. inbound audio goes through the gate ===")
    import base64
    b2 = bridge_mod.MediaStreamBridge(send_json)
    b2.stt = FakeSTT()
    b2.stt_ready = True
    quiet_frame = tone(150)[:FRAME]
    for _ in range(40):
        await b2.on_media(base64.b64encode(quiet_frame).decode())
    check("background frames reach Deepgram quieter than they arrived",
          rms_of(b2.stt.audio[-1]) < rms_of(quiet_frame) * 0.3,
          f"{rms_of(quiet_frame)} -> {rms_of(b2.stt.audio[-1])}")
    loud_frame = tone(9000)[:FRAME]
    await b2.on_media(base64.b64encode(loud_frame).decode())
    check("the caller's voice reaches Deepgram unchanged",
          b2.stt.audio[-1] == loud_frame)

    b2.stt.ignoring = True
    n = len(b2.stt.audio)
    await b2.on_media(base64.b64encode(loud_frame).decode())
    check("nothing is forwarded while the bot is speaking",
          len(b2.stt.audio) == n)

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
