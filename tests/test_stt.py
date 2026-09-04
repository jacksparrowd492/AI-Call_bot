"""Turn-taking and echo-suppression tests for stt.py.

These reproduce, as message sequences, the two things callers complained
about on the 2026-09-04 call:

  * the bot answering its own voice ("Hello?", "And", "Hold on." right after
    each TTS burst finished), and
  * the bot answering before the caller had finished the sentence.

No websocket and no Deepgram: the receiver's message handlers are driven
directly with the JSON Deepgram would have sent.
"""
import asyncio
import os
import sys

# NOTE: config.py calls load_dotenv(override=True), so the project's .env
# BEATS anything set here. These are defaults for a checkout without a .env;
# every wait below is derived from `settings`, never hardcoded, so the suite is
# correct either way.
os.environ.setdefault("TURN_DEBOUNCE_MS", "120")
os.environ["DEEPGRAM_ENDPOINTING_MS"] = "900"
os.environ["UTTERANCE_END_MS"] = "1400"
os.environ["STT_MIN_CONFIDENCE"] = "0.55"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings          # noqa: E402
import stt as stt_mod                # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def results(text, is_final=True, speech_final=False, confidence=0.95):
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text, "confidence": confidence}]},
    }


class Harness:
    """A DeepgramStream with the socket amputated."""
    def __init__(self):
        self.heard = []
        self.stream = stt_mod.DeepgramStream(self._on_utterance)

    async def _on_utterance(self, text):
        self.heard.append(text)

    async def feed(self, msg):
        if msg["type"] == "Results":
            await self.stream._on_results(msg)
        elif msg["type"] == "UtteranceEnd":
            self.stream._cancel_flush()
            await self.stream._flush("utterance-end")
        elif msg["type"] == "SpeechStarted":
            self.stream._cancel_flush()

    async def settle(self, ms=250):
        await asyncio.sleep(ms / 1000.0)


async def main():
    print("\n=== 1. a pause mid-sentence is NOT the end of the turn ===")
    h = Harness()
    await h.feed(results("Is there a school", speech_final=True))
    await h.settle(60)                       # caller pauses to think...
    check("no reply during the pause", h.heard == [], str(h.heard))
    await h.feed(results("near the project?", speech_final=True))
    await h.feed({"type": "UtteranceEnd"})
    check("both halves arrive as ONE turn",
          h.heard == ["Is there a school near the project?"], str(h.heard))

    print("\n=== 2. UtteranceEnd closes the turn ===")
    h = Harness()
    await h.feed(results("what is the rera number", speech_final=False))
    check("no reply before the turn ends", h.heard == [], str(h.heard))
    await h.feed({"type": "UtteranceEnd"})
    check("UtteranceEnd flushes", h.heard == ["what is the rera number"], str(h.heard))

    print("\n=== 3. debounce flushes when UtteranceEnd never comes ===")
    h = Harness()
    await h.feed(results("tell me about the water supply", speech_final=True))
    check("not flushed instantly", h.heard == [], str(h.heard))
    await h.settle(settings.turn_debounce_ms + 200)
    check("flushed after the debounce",
          h.heard == ["tell me about the water supply"], str(h.heard))

    print("\n=== 4. resuming speech cancels the pending flush ===")
    h = Harness()
    await h.feed(results("I wanted to ask", speech_final=True))
    await h.feed({"type": "SpeechStarted"})
    await h.settle(250)
    check("SpeechStarted holds the turn open", h.heard == [], str(h.heard))
    await h.feed(results("about the plot sizes.", speech_final=True))
    await h.feed({"type": "UtteranceEnd"})
    check("one turn, not two",
          h.heard == ["I wanted to ask about the plot sizes."], str(h.heard))

    print("\n=== 5. the echo problem: the bot must not hear itself ===")
    h = Harness()
    h.stream.mute()
    check("muted stream reports ignoring", h.stream.ignoring)
    await h.feed(results("Hello?", speech_final=True))
    await h.feed({"type": "UtteranceEnd"})
    await h.settle(250)
    check("echo during playback is dropped", h.heard == [], str(h.heard))

    sent = []

    class FakeWS:
        async def send(self, data):
            sent.append(data)
    h.stream._ws = FakeWS()
    await h.stream.send_audio(b"\xff" * 160)
    check("no caller audio is forwarded while muted", sent == [], str(len(sent)))

    print("\n=== 6. the echo TAIL after playback ends ===")
    h.stream.unmute(tail_s=0.25)
    check("still ignoring during the tail", h.stream.ignoring)
    await h.feed(results("Hold on.", speech_final=True))
    await h.feed({"type": "UtteranceEnd"})
    check("tail echo is dropped too", h.heard == [], str(h.heard))
    await h.settle(300)
    check("listening again after the tail", not h.stream.ignoring)
    await h.stream.send_audio(b"\xff" * 160)
    check("caller audio flows again once the tail expires", len(sent) == 1, str(len(sent)))

    print("\n=== 7. a half-finished turn is dropped when the bot starts talking ===")
    h = Harness()
    await h.feed(results("I was going to ask", speech_final=False))
    h.stream.mute()                          # bot starts speaking
    h.stream.unmute(0.0)
    await h.feed({"type": "UtteranceEnd"})
    check("stale half-turn never reaches the brain", h.heard == [], str(h.heard))

    print("\n=== 8. junk fragments are not turns ===")
    junk = [("And", 0.95), ("uh", 0.95), ("um um", 0.95), (".", 0.95),
            ("what is the price", 0.20)]
    for text, conf in junk:
        h = Harness()
        await h.feed(results(text, speech_final=True, confidence=conf))
        await h.feed({"type": "UtteranceEnd"})
        check(f"dropped {text!r} (conf={conf})", h.heard == [], str(h.heard))

    keep = [("No.", 0.9), ("Yes please.", 0.9), ("ok", 0.9),
            ("Is there a school near?", 0.62)]
    for text, conf in keep:
        h = Harness()
        await h.feed(results(text, speech_final=True, confidence=conf))
        await h.feed({"type": "UtteranceEnd"})
        check(f"kept {text!r} (conf={conf})", h.heard == [text], str(h.heard))

    print("\n=== 9. deepgram is asked for the right features ===")
    url = stt_mod.DEEPGRAM_WS_URL
    for param in ("interim_results=true", "vad_events=true",
                  f"utterance_end_ms={settings.deepgram_utterance_end_ms}",
                  f"endpointing={settings.deepgram_endpointing_ms}"):
        check(f"url has {param}", param in url)

    print("\n=== 10. muting the mic must not cancel the reply in progress ===")
    # The 2026-09-04 outage. bridge._speak() opens with stt.mute(); mute()
    # called _cancel_flush(); and on the debounce path _flush_task IS the task
    # running the reply - so the bot cancelled itself before a single frame of
    # audio went out, and _flush_after_debounce swallowed the CancelledError.
    # Eight of the ten answers on that call were lost exactly this way.
    h = Harness()
    spoke = []

    async def reply_like_the_bridge(text):
        h.stream.mute()                      # bridge._speak, first line
        await asyncio.sleep(0.05)            # ...the TTS await
        spoke.append(text)                   # frames actually reach the caller
        h.stream.unmute(0.0)

    h.stream.on_utterance = reply_like_the_bridge
    await h.feed(results("is there any school nearby?", speech_final=True))
    await h.settle(settings.turn_debounce_ms + 250)   # debounce + the reply
    check("the reply survives the bot muting the mic",
          spoke == ["is there any school nearby?"], str(spoke))

    print("\n=== 11. a trailing UtteranceEnd must not cancel the reply ===")
    # endpointing (900 ms) fires the debounce flush, then utterance_end_ms
    # (1400 ms) lands ~700 ms later - i.e. while the bot is already speaking.
    # That path calls _cancel_flush() too.
    h = Harness()
    spoke = []

    async def reply_then_utterance_end(text):
        h.stream.mute()
        await asyncio.sleep(0.01)
        await h.feed({"type": "UtteranceEnd"})   # arrives mid-reply
        await asyncio.sleep(0.05)
        spoke.append(text)
        h.stream.unmute(0.0)

    h.stream.on_utterance = reply_then_utterance_end
    await h.feed(results("what is the rera number?", speech_final=True))
    await h.settle(settings.turn_debounce_ms + 250)
    check("the reply survives a trailing UtteranceEnd",
          spoke == ["what is the rera number?"], str(spoke))

    print("\n" + "=" * 60)
    print(f"PASSED {len(PASS)}   FAILED {len(FAIL)}")
    if FAIL:
        print("Failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
