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
import scheduling                     # noqa: E402

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
    check("and the bot asks for the day and time FIRST",
          b.tts.spoken == [bridge_mod.HANDOFF_ASK_TIME], str(b.tts.spoken))
    b.tts.spoken.clear()
    await utter(b, "tomorrow at 6 pm")
    check("the name comes after the time, not before",
          b.tts.spoken == [bridge_mod.ASK_NAME], str(b.tts.spoken))
    check("and the time is already on the row before the name is asked",
          b.callback_ist is not None and len(recorded) >= 2,
          "%s / %s" % (b.callback_ist, recorded))
    b.tts.spoken.clear()
    await utter(b, "Karthi")
    check("the name is read back before it is trusted",
          b.tts.spoken == [bridge_mod.CONFIRM_NAME.format(name="Karthi")],
          str(b.tts.spoken))
    b.tts.spoken.clear()
    await utter(b, "Yes")
    check("and the booking is confirmed once the name is settled",
          b.tts.spoken == [bridge_mod.HANDOFF_CONFIRMED], str(b.tts.spoken))

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

    print("\n=== 9. an NRI's callback time is asked for, then converted ===")
    # A Dubai caller saying "tomorrow evening" has named a day, not a time.
    # The bot asks once, keeps the day, and stores the result in IST - because
    # the person who has to make the call is in Coimbatore.
    b, recorded = bridge_with_spy()
    b.caller_number = "+971501234567"
    b.caller_zone = scheduling.zone_for(b.caller_number)
    b.callback_requested = True
    b.awaiting_callback_time = True
    b.tts.spoken.clear()

    await utter(b, "tomorrow evening")
    check("a day with no hour gets exactly one follow-up question",
          b.tts.spoken == [bridge_mod.HANDOFF_ASK_HOUR]
          and b.awaiting_callback_time is True, str(b.tts.spoken))
    check("and the day is carried into that question",
          b.callback_said == "tomorrow evening", repr(b.callback_said))

    b.tts.spoken.clear()
    await utter(b, "around 6")
    check("the hour completes the day given a turn earlier",
          b.callback_local is not None and b.callback_local.hour == 18,
          str(b.callback_local))
    check("6pm in Dubai is stored as 7:30pm IST",
          b.callback_ist is not None
          and (b.callback_ist.hour, b.callback_ist.minute) == (19, 30),
          str(b.callback_ist))
    check("the bot does not ask for an hour twice",
          b.awaiting_callback_time is False)
    check("the sales team's note carries both times",
          "IST" in (b.callback_time or "") and "Asia/Dubai" in (b.callback_time or ""),
          repr(b.callback_time))
    check("and only then does it ask for the name",
          b.tts.spoken == [bridge_mod.ASK_NAME], str(b.tts.spoken))

    print("\n=== 10. a caller who ASKS to schedule is answered with yes ===")
    # They have already asked, so offering them a callback would be a wasted
    # turn. Say yes and go straight for the day and time. The "speak to
    # someone" path is untouched and still uses its own wording.
    b, recorded = bridge_with_spy()
    b.tts.spoken.clear()
    await utter(b, "can we schedule a call")
    check("scheduling books without a consent round-trip",
          b.callback_requested is True and recorded == [1], str(recorded))
    check("the caller's own words become the sheet's 'Interested in'",
          b.lead_requirement == "can we schedule a call", repr(b.lead_requirement))
    check("and the day and time are asked straight away, in the schedule wording",
          b.tts.spoken == [bridge_mod.SCHEDULE_ASK_WHEN], str(b.tts.spoken))
    check("the name is NOT asked before the time",
          bridge_mod.ASK_NAME not in b.tts.spoken, str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    b.tts.spoken.clear()
    await utter(b, "can I speak to someone")
    check("the 'speak to someone' wording asks for the day and time too",
          b.tts.spoken == [bridge_mod.HANDOFF_ASK_TIME], str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    booking_q = []

    async def reply_booking(t):
        booking_q.append(t)

    b._reply = reply_booking
    await utter(b, "how can I book a plot")
    check("'how can I book a plot' stays a knowledge-base question",
          recorded == [] and booking_q == ["how can I book a plot"],
          str(booking_q))

    b, recorded = bridge_with_spy()
    b.caller_number = "+971501234567"
    b.caller_zone = scheduling.zone_for(b.caller_number)
    b.tts.spoken.clear()
    await utter(b, "schedule a call tomorrow at 6 pm")
    check("a time given in the same breath is converted, not stored raw",
          b.callback_ist is not None
          and (b.callback_ist.hour, b.callback_ist.minute) == (19, 30),
          str(b.callback_ist))
    check("and the bot goes straight to the name, without asking the time again",
          b.tts.spoken == [bridge_mod.ASK_NAME], str(b.tts.spoken))

    print("\n=== 11. the name is asked once, AFTER the time, and read back ===")
    b, recorded = bridge_with_spy()
    await utter(b, "can we schedule a call")
    await utter(b, "tomorrow at 10 am")
    b.tts.spoken.clear()
    await utter(b, "My name is David Kumar")
    check("a name given in any phrasing is captured and read back",
          b.lead_name == "David Kumar"
          and b.tts.spoken == [bridge_mod.CONFIRM_NAME.format(name="David Kumar")],
          "%r / %s" % (b.lead_name, b.tts.spoken))
    b.tts.spoken.clear()
    await utter(b, "Yes")
    check("and the booking is confirmed with the time already booked",
          b.tts.spoken == [bridge_mod.HANDOFF_CONFIRMED]
          and b.callback_local is not None and b.callback_local.hour == 10,
          "%s / %s" % (b.tts.spoken, b.callback_local))

    b, recorded = bridge_with_spy()
    await utter(b, "can we schedule a call")
    await utter(b, "tomorrow at 10 am")
    b.tts.spoken.clear()
    await utter(b, "I would rather not")
    check("a caller who will not give a name still gets their callback",
          b.lead_name is None and b.callback_requested is True
          and b.callback_local is not None, repr(b.lead_name))
    check("and is confirmed rather than asked again",
          b.tts.spoken == [bridge_mod.HANDOFF_CONFIRMED], str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    b.lead_name = "Priya"                      # already captured by METADATA
    await utter(b, "can we schedule a call")
    b.tts.spoken.clear()
    await utter(b, "tomorrow at 10 am")
    check("a name already known is not asked for again",
          b.tts.spoken == [bridge_mod.HANDOFF_CONFIRMED], str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    b.caller_number = "+971501234567"
    b.caller_zone = scheduling.zone_for(b.caller_number)
    await utter(b, "can we schedule a call")
    b.tts.spoken.clear()
    await utter(b, "David, tomorrow at 6 pm")
    check("a name AND a time in one answer are both taken",
          b.lead_name == "David" and b.callback_ist is not None
          and (b.callback_ist.hour, b.callback_ist.minute) == (19, 30),
          "%s / %s" % (b.lead_name, b.callback_ist))
    check("and the volunteered name is read back, not asked for again",
          b.tts.spoken == [bridge_mod.CONFIRM_NAME.format(name="David")],
          str(b.tts.spoken))

    b, recorded = bridge_with_spy()
    await utter(b, "can we schedule a call")
    b.tts.spoken.clear()
    await utter(b, "whenever you like")
    check("'whenever you like' is a time preference, never a name",
          b.lead_name is None and b.tts.spoken == [bridge_mod.ASK_NAME],
          "%r / %s" % (b.lead_name, b.tts.spoken))
    check("and 'anytime' is written down as what they actually said",
          b.callback_time == "anytime", repr(b.callback_time))

    print("\n=== 12. faults seen on real calls ===")
    # 2026-09-05 18:04:27 - the caller answered the offer by naming a time, and
    # the bot logged "offer went unanswered", asked again, and booked a
    # DIFFERENT time off a later turn.
    b, recorded = bridge_with_spy()
    await b._handle_metadata({"handoff": True})
    b.tts.spoken.clear()
    await utter(b, "Can you please schedule tomorrow at around 5PM?")
    check("naming a time IS accepting the offer",
          b.callback_requested is True and recorded, str(recorded))
    check("and the offer is not left pending",
          b.awaiting_handoff_consent is False)
    check("the time the caller first said is the one booked (5pm, not 6)",
          b.callback_ist is not None and b.callback_ist.hour == 17,
          str(b.callback_ist))

    # 2026-09-05 18:04:53 - "My name is Carty." was stored with the full stop.
    b, recorded = bridge_with_spy()
    await utter(b, "can we schedule a call")
    await utter(b, "tomorrow at 10 am")
    await utter(b, "My name is Carty.")
    check("a name never keeps its trailing full stop",
          b.lead_name == "Carty", repr(b.lead_name))
    b.tts.spoken.clear()
    await utter(b, "No, it is Karthi")
    check("a corrected name replaces the misheard one",
          b.lead_name == "Karthi", repr(b.lead_name))

    # 2026-09-06 10:52:36 - the booking went to the sales team with the name
    # "I'm" and no time at all: the name was asked first, "I'm" was taken as
    # the answer, and the time question never got a usable answer after it.
    b, recorded = bridge_with_spy()
    await utter(b, "can we schedule a call")
    await utter(b, "tomorrow at 10 am")
    check("the time is booked before the name is ever asked",
          b.callback_local is not None and b.callback_local.hour == 10,
          str(b.callback_local))
    b.tts.spoken.clear()
    await utter(b, "I'm")
    check("a bare \"I'm\" is not a name",
          b.lead_name is None, repr(b.lead_name))
    check("and the booking is confirmed anyway, time intact",
          b.tts.spoken == [bridge_mod.HANDOFF_CONFIRMED]
          and b.callback_local is not None, str(b.tts.spoken))

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
