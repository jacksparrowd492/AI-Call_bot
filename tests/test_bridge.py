"""Bridge-level turn-taking tests: the mic, the mark, the lock, the smalltalk
shortcut. Deepgram, Kokoro, Groq, Twilio and the embedder are all replaced.
"""
import asyncio
import os
import sys
import types

os.environ.setdefault("GROQ_API_KEY", "gsk-test")
os.environ.setdefault("DEEPGRAM_API_KEY", "dg-test")
os.environ["CHROMA_DIR"] = os.path.join(os.path.dirname(__file__), "chroma_bridge_test")
# NOTE: config.py calls load_dotenv(override=True), so the project's .env
# BEATS anything set here. This is only a default for a checkout without one;
# the expected echo tail below is read from `settings`, never hardcoded.
os.environ.setdefault("ECHO_TAIL_MS", "150")

# Kokoro's ONNX model is not needed to test the bridge's control flow.
_k = types.ModuleType("kokoro_onnx")
_k.Kokoro = object
sys.modules["kokoro_onnx"] = _k

from chromadb.utils import embedding_functions        # noqa: E402
embedding_functions.ONNXMiniLM_L6_V2 = lambda **kw: (lambda texts: [[0.0] * 384
                                                                    for _ in texts])

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings           # noqa: E402
import bridge as bridge_mod           # noqa: E402
import llm as llm_mod                 # noqa: E402

# What FakeSTT records when _speak reopens the mic, at whatever ECHO_TAIL_MS
# is actually in force.
UNMUTE = "unmute(%s)" % (settings.echo_tail_ms / 1000.0)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


class FakeSTT:
    def __init__(self):
        self.events, self.ignoring = [], False

    def mute(self):
        self.events.append("mute")
        self.ignoring = True

    def unmute(self, tail_s=0.0):
        self.events.append(f"unmute({tail_s})")
        self.ignoring = False

    async def start(self):
        pass

    async def close(self):
        pass

    async def send_audio(self, audio):
        pass


class FakeTTS:
    def __init__(self, frames=5):
        self.spoken, self.frames = [], frames

    async def stream_to_twilio(self, text, send_media, send_mark=None, cancelled=None):
        # _speak now hands over a LIST of sentences so the real streamer can
        # pipeline synthesis. Record what the caller would actually hear.
        self.spoken.append(text if isinstance(text, str) else " ".join(text))
        if send_mark:
            await send_mark(f"tts-done-{self.frames}")
        return self.frames


def make_bridge(tts_frames=5):
    sent = []

    async def send_json(data):
        sent.append(data)

    b = bridge_mod.MediaStreamBridge(send_json)
    b.stream_sid = "MZtest"
    b.stt = FakeSTT()
    b.tts = FakeTTS(tts_frames)
    return b, sent


async def speak_and_confirm(b, text):
    """Run _speak, and play the part of Twilio: echo the mark back."""
    task = asyncio.create_task(b._speak(text))
    await asyncio.sleep(0.02)
    b.on_mark(b._expected_mark)
    await task


async def main():
    print("\n=== 1. the mic is shut for the whole time the bot speaks ===")
    b, sent = make_bridge()
    await speak_and_confirm(b, "Hello there.")
    check("muted before speaking, unmuted after",
          b.stt.events == ["mute", UNMUTE], str(b.stt.events))
    check("echo tail comes from settings (ECHO_TAIL_MS)",
          b.stt.events[-1] == UNMUTE, str(b.stt.events))
    check("speaking flag cleared", b.speaking is False)

    print("\n=== 2. the mic stays shut until TWILIO says playback finished ===")
    b, sent = make_bridge()
    task = asyncio.create_task(b._speak("A long answer about the township."))
    await asyncio.sleep(0.05)
    check("still muted while audio is queued at Twilio",
          b.stt.events == ["mute"] and b.speaking, str(b.stt.events))
    b.on_mark(b._expected_mark)
    await task
    check("unmuted once the mark comes back",
          b.stt.events == ["mute", UNMUTE], str(b.stt.events))

    print("\n=== 3. marks are per-utterance, not per-frame-count ===")
    b, sent = make_bridge()
    await speak_and_confirm(b, "First.")
    first = b._expected_mark
    await speak_and_confirm(b, "Second.")
    check("each utterance gets its own mark name", first != b._expected_mark,
          f"{first} vs {b._expected_mark}")
    marks = [m["mark"]["name"] for m in sent if m.get("event") == "mark"]
    check("the mark we send is the mark we wait for",
          marks == ["jarvis-1", "jarvis-2"], str(marks))
    b2, _ = make_bridge()
    b2._expected_mark = "jarvis-9"
    b2._playback_done.clear()
    b2.on_mark("jarvis-1")
    check("a stale mark does not reopen the mic", not b2._playback_done.is_set())

    print("\n=== 4. utterances arriving while the bot speaks are dropped ===")
    b, sent = make_bridge()
    b.speaking = True
    replied = []
    b._reply = lambda t: replied.append(t)
    await b._on_utterance("Hold on.")
    check("nothing happens while speaking", replied == [] and b.tts.spoken == [],
          str(replied))

    b.speaking = False
    b.stt.ignoring = True
    await b._on_utterance("And")
    check("nothing happens during the echo tail", replied == [], str(replied))

    print("\n=== 5. one turn at a time ===")
    b, sent = make_bridge()
    order = []

    async def slow_handle(text):
        order.append(f"start:{text}")
        await asyncio.sleep(0.1)
        order.append(f"end:{text}")

    b._handle_utterance = slow_handle
    await asyncio.gather(b._on_utterance("first question"),
                         b._on_utterance("second question"))
    check("the second turn is dropped, not interleaved",
          order == ["start:first question", "end:first question"], str(order))

    print("\n=== 6. greetings skip RAG and the LLM entirely ===")
    b, sent = make_bridge()
    calls = []
    b.brain.reply_stream = lambda *a, **kw: calls.append(a) or iter([])
    await b._on_utterance("Hello")
    check("no LLM call for a greeting", calls == [], str(calls))
    check("the canned greeting is spoken",
          b.tts.spoken == [llm_mod.GREETING_LINE], str(b.tts.spoken))
    check("history records both sides",
          [h["role"] for h in b.history] == ["user", "assistant"], str(b.history))

    print("\n=== 7. goodbye ends the call ===")
    b, sent = make_bridge()
    b._deliver_brochure = lambda: asyncio.sleep(0)
    await b._on_utterance("thank you bye")
    check("exit line spoken", b.tts.spoken == [llm_mod.EXIT_LINE], str(b.tts.spoken))
    check("call is marked for hangup", b.should_end is True)

    print("\n=== 8. a callback is booked ONLY when the caller says yes ===")
    # 2026-09-04, 15:15:27: the bot asked "shall I arrange a call with our
    # sales team?" and recorded the callback in the same breath. The caller
    # ignored the question and asked about schools - and still received an SMS
    # confirming an appointment he had never agreed to.

    async def utter(b, text):
        """One caller turn, playing Twilio's mark echo so _speak returns
        promptly instead of waiting out mark_timeout_ms."""
        task = asyncio.create_task(b._on_utterance(text))
        for _ in range(60):
            await asyncio.sleep(0.01)
            if b.speaking:
                b.on_mark(b._expected_mark)
            if task.done():
                break
        await task

    def bridge_with_spy():
        b, sent = make_bridge()
        recorded = []
        b._record_callback = lambda: recorded.append(1) or asyncio.sleep(0)
        b._deliver_brochure = lambda: asyncio.sleep(0)
        return b, recorded

    b, recorded = bridge_with_spy()
    await b._handle_metadata({"handoff": True})
    check("the offer alone books nothing",
          recorded == [] and b.callback_requested is False, str(recorded))
    check("the offer is held as pending", b.awaiting_handoff_consent is True)

    b, recorded = bridge_with_spy()
    await b._handle_metadata({"handoff": True})
    b.tts.spoken.clear()
    await utter(b, "Yes, please.")
    check("yes books the callback",
          b.callback_requested is True and recorded == [1], str(recorded))
    check("and the bot asks when to call",
          b.tts.spoken == [bridge_mod.HANDOFF_ASK_TIME], str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    await b._handle_metadata({"handoff": True})
    b.tts.spoken.clear()
    await utter(b, "No thanks.")
    check("no books nothing",
          recorded == [] and b.callback_requested is False, str(recorded))
    check("and the bot moves on politely",
          b.tts.spoken == [bridge_mod.HANDOFF_DECLINED], str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    answered = []

    async def fake_reply(t):
        answered.append(t)

    b._reply = fake_reply
    await b._handle_metadata({"handoff": True})
    await utter(b, "Is there any schools?")
    check("ignoring the offer books nothing",
          recorded == [] and b.callback_requested is False, str(recorded))
    check("and the ignored turn is answered as a normal question",
          answered == ["Is there any schools?"], str(answered))
    check("the offer does not stay pending", b.awaiting_handoff_consent is False)

    b, recorded = bridge_with_spy()
    await utter(b, "connect me to a human")
    check("an explicit ask still books immediately, no second question",
          b.callback_requested is True and recorded == [1], str(recorded))

    # "Okay" on its own is consent; "Okay, thank you" is a caller ending the
    # call. That exact turn closed the 2026-09-04 call at 15:18:12, and a weak
    # yes must never book a salesperson's time.
    b, recorded = bridge_with_spy()
    b._reply = fake_reply
    await b._handle_metadata({"handoff": True})
    await utter(b, "Okay. Thank you.")
    check("'okay thank you' ends the call, it does not book a callback",
          recorded == [] and b.callback_requested is False, str(recorded))

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
