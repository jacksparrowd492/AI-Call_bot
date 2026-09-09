import asyncio
import logging
import re
import time
from datetime import datetime

from audio_utils import gate_from_settings, barge_in_from_settings
from config import settings
from stt import DeepgramStream
from tts import KokoroStreamer
from llm import GroqBrain, EXIT_LINE
from rag.retriever import retrieve
from handoff_intent import (wants_human, wants_schedule, extract_time,
                            extract_name, yes_no)
import scheduling

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
                    "call you back. Which day and what time would suit you?")
HANDOFF_CONFIRMED = ("Perfect. Our specialist will call you then, and I'll send "
                     "you a confirmation on WhatsApp and SMS right away.")
# Spoken when the caller turns the offer down. Nothing is recorded and no
# message goes out - declining must cost the caller nothing.
HANDOFF_DECLINED = "No problem. Is there anything else I can help you with?"
# "Tomorrow evening" is a day, not a time. Guessing an hour would write a
# precision into the sales sheet the caller never gave - and for an NRI that
# guess then gets converted into IST and someone acts on it. Ask once instead.
HANDOFF_ASK_HOUR = ("Sure. What time exactly - say, ten in the morning, "
                    "or six in the evening?")
# The caller ASKED to schedule, so there is nothing to offer them - say yes and
# get the day and the time. This is a separate line from HANDOFF_ASK_TIME on
# purpose: that one answers "I want to speak to someone", this one answers
# "can we schedule a call?".
SCHEDULE_ASK_WHEN = ("Yes, of course - I can arrange that for you. "
                     "Which day and what time would suit you?")
# Asked AFTER the day and time, and only on the callback path - a row in the
# sales sheet with a number and no name is next to useless to whoever has to
# ring it. Ordinary enquiry calls are never asked this.
#
# The order matters and it is the caller's, not ours. Someone who has just
# asked for a call back is thinking about WHEN. Asking their name first made
# them answer a question they had not asked yet, and the time then had to
# survive a name, a read-back and a yes before it was written down - which is
# how the 2026-09-06 10:52 booking ended up with a name and no time at all.
ASK_NAME = "Thank you. And may I have your name, please?"
# Deepgram heard "Karthi" as "Carty" at confidence 1.00 - certain and wrong.
# No recogniser gets an unknown personal name right every time, so the only
# reliable fix is to say it back and let the caller correct it.
CONFIRM_NAME = "Thank you. I have that as {name}. Is that right?"
ASK_NAME_AGAIN = ("Sorry, I didn't catch that. Could you say just your name, "
                  "slowly?")

# Offered ONCE, at the end of a call where the caller actually asked about the
# project. Everything before this point is the not-found path - the bot only
# offered a callback when it could not answer - so a caller whose questions
# were all answered was never asked whether they wanted to speak to anybody.
CLOSING_OFFER = ("One last thing before you go - would you like me to arrange "
                 "a call back from one of our sales specialists?")

# Words that are the START of a sentence, not a sentence. Deepgram closes a
# turn on silence, and a caller who pauses to think mid-question gets split
# across two or three turns: the 2026-09-06 call arrived as "Well," then
# "what" then "transport facilities?", and the bot answered the first two with
# "I didn't catch that" before the real question ever landed. A turn made only
# of these is HELD and joined to whatever the caller says next.
FRAGMENT_WORDS = {
    "well", "so", "and", "but", "um", "umm", "uh", "erm", "er", "hmm", "mm",
    "like", "actually", "see", "mean", "i", "what", "how", "about", "the",
    "a", "an", "is", "it", "its", "then", "also", "maybe", "just", "you",
    "know", "tell", "me", "my", "in", "of", "for", "can", "could",
}
# At most this many words can be held as a fragment...
FRAGMENT_MAX_WORDS = 2
# ...and this is how long the bot waits for the rest of the sentence before
# deciding the caller really did trail off. Long enough to think, short enough
# that the silence never feels like a dropped line.
FRAGMENT_HOLD_S = 3.0

# How long a turn may sit in the queue behind a reply before it stops being
# worth answering. Longer than the longest reply (15.2s was the record on the
# 2026-09-06 call), short enough that the bot never answers something the
# caller has moved on from.
QUEUED_TURN_MAX_AGE = 20.0


class MediaStreamBridge:
    def __init__(self, send_json):
        self.send_json = send_json
        self.stream_sid = None
        self.call_sid = None
        self.caller_number = "unknown"

        self.brain = GroqBrain()
        self.tts = KokoroStreamer()
        self.stt = DeepgramStream(self._on_utterance)
        # Turns the caller's background down before Deepgram ever hears it.
        # A confidence threshold can only reject a transcript that already
        # exists; this stops the transcript being made in the first place.
        self.gate = gate_from_settings()
        # Listens for the caller talking OVER the bot. Only fed while the bot
        # is speaking, and reset for every reply.
        self.barge = barge_in_from_settings()
        # Set when the caller has interrupted, and read by _is_cancelled so
        # the TTS loop stops mid-sentence instead of finishing the paragraph.
        self.interrupted = False

        self.history = []
        self.speaking = False

        # One turn at a time. A second utterance must not start a second LLM
        # call and a second TTS stream on top of the first - but it is not
        # thrown away either; see _drain_queued_turn.
        self._reply_lock = asyncio.Lock()
        self._queued_turn = None

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
        # The SAME appointment written for the caller: their own clock first.
        # callback_time is IST-first, because the sales team reads that one.
        self.callback_time_for_caller = None
        self.awaiting_callback_time = False
        # Waiting on "may I have your name?", asked once before the day/time.
        self.awaiting_name = False
        self.awaiting_name_confirm = False
        self.name_attempts = 0
        # Each asked once per call, and only once: an hour when they named a
        # day without one, a name when we still do not have theirs.
        self.asked_for_hour = False
        self.asked_for_name = False
        self.callback_said = None          # the day part, kept across that ask
        self.callback_raw = None           # what they said, for the sheet
        self.callback_local = None         # aware datetime in the caller's zone
        self.callback_ist = None           # the same instant, in IST
        # The bot has ASKED whether to arrange a callback and is waiting for
        # the answer. Nothing is recorded or sent until the caller says yes.
        self.awaiting_handoff_consent = False
        self.callback_confirmed = False
        # A callback has been offered once already - the closing offer is not
        # made twice, and never on top of the not-found one.
        self.callback_offered = False
        # The caller has asked at least one real question about the project,
        # which is what makes the closing offer worth making at all.
        self.asked_project_question = False
        # The closing offer is out and the call ends as soon as it is answered.
        self.pending_exit = False
        # Half a sentence, waiting for the other half. See FRAGMENT_WORDS.
        self._pending_fragment = None
        self._fragment_at = 0.0
        self._fragment_task = None
        self.lead_name = None
        self.lead_requirement = None
        self.lead_intent = None

        # Where the caller is, worked out from the number that rang us, and how
        # long they stayed on the phone.
        self.caller_zone = {"name": None, "zone": None, "region": None}
        self.started_at = None
        self.notified_via = None
        self.agent_alerted = False
        self.call_logged = False

    # ------------------------------------------------ lifecycle
    async def on_start(self, start: dict):
        self.stream_sid = start.get("streamSid")
        self.call_sid = start.get("callSid")

        params = start.get("customParameters") or {}
        self.caller_number = params.get("from") or self.caller_number

        self.started_at = datetime.now()
        self.caller_zone = scheduling.zone_for(self.caller_number)

        log.info("Call started sid=%s from=%s", self.call_sid, self.caller_number)
        if self.caller_zone.get("name"):
            log.info("🌍 Caller is in %s (%s)%s", self.caller_zone["name"],
                     self.caller_zone.get("region") or "?",
                     " - country spans several zones, using this one"
                     if self.caller_zone.get("ambiguous") else "")
        else:
            log.warning("🌍 No timezone for %s - a requested time will be "
                        "taken as IST", self.caller_number)

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

        # While the bot is talking this frame is MOSTLY our own echo, and
        # send_audio would discard it anyway - it stays out of the gate's
        # noise-floor measurement for that reason. But "mostly" is the point:
        # if the caller is talking over the top, this is the only place that
        # can tell, because nothing else is listening.
        if self.stt.ignoring:
            if self.speaking and not self.interrupted:
                import base64
                if self.barge.feed(base64.b64decode(payload_b64)):
                    await self._on_barge_in()
            return

        import base64
        audio = base64.b64decode(payload_b64)
        await self.stt.send_audio(self.gate.process(audio))

    async def on_stop(self, reason=""):
        self._cancel_fragment()
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

        # Last, so the row carries what actually happened above.
        await self._write_call_log()

    async def _write_call_log(self):
        """One row per call in the sales team's Excel sheet.

        Written at the end because that is the only point where talk time is
        known - and it runs after the notifications so the row records what
        really went out, not what was intended.
        """
        if self.call_logged or not self.started_at:
            return
        self.call_logged = True

        from tools import call_log

        loop = asyncio.get_running_loop()

        # The caller has gone, so a Groq round-trip costs nobody anything.
        summary = ""
        try:
            summary = await loop.run_in_executor(
                None, self.brain.summarize_call, list(self.history))
        except Exception as e:
            log.warning("Call summary failed: %s", e)
        if not summary:
            summary = self._plain_summary()

        rec = call_log.build_record(
            started_at=self.started_at,
            ended_at=datetime.now(),
            name=self.lead_name,
            phone=self.caller_number,
            country=self.caller_zone.get("region"),
            caller_timezone=self.caller_zone.get("name"),
            requirement=self.lead_requirement,
            summary=summary,
            intent_score=self.lead_intent,
            callback=self.callback_requested,
            asked_for=self.callback_raw,
            callback_local=(scheduling.human(self.callback_local)
                            if self.callback_local else ""),
            callback_ist=(scheduling.human(self.callback_ist)
                          if self.callback_ist else ""),
            confirmed_via=self.notified_via,
            agent_alerted=self.agent_alerted,
            brochure_sent=self.brochure_sent,
            call_sid=self.call_sid,
        )

        try:
            await loop.run_in_executor(None, call_log.append, rec)
        except Exception as e:
            log.error("Could not write the call log: %s", e)

    def _plain_summary(self) -> str:
        """Fallback when Groq is unavailable - built from the transcript, so
        the column is never empty just because the API was down."""
        asked = [" ".join((t.get("content") or "").split())
                 for t in self.history if t.get("role") == "user"]
        asked = [a for a in asked if a][:6]
        bits = []
        if asked:
            bits.append("Caller asked: " + "; ".join(asked))
        if self.callback_requested:
            when = self.callback_time or "time not given"
            bits.append("Callback booked for %s." % when)
        if self.brochure_sent:
            bits.append("Brochure sent.")
        return " ".join(bits)[:500] or "No questions asked."

    # ------------------------------------------------ STT
    async def _on_utterance(self, text: str):
        # The mic is already shut while the bot talks; this is belt-and-braces
        # for a transcript that was in flight when playback started.
        if self.speaking or self.stt.ignoring:
            log.info("🔇 Ignored while speaking: %r", text[:60])
            return

        if self._reply_lock.locked():
            # QUEUED, not dropped. A reply runs for several seconds, and a
            # caller who says something in that window - most often repeating
            # themselves because the bot has not answered yet - had the retry
            # thrown away and heard nothing back. One slot deep on purpose:
            # by the time the bot is free, the OLDEST thing the caller said is
            # the least worth answering, and answering three queued turns in a
            # row would talk over them for half a minute.
            dropped = self._queued_turn
            self._queued_turn = (text, time.time())
            log.info("⏳ Queued while answering: %r%s", text[:60],
                     " (replacing %r)" % dropped[0][:40] if dropped else "")
            return

        async with self._reply_lock:
            await self._handle_utterance(text)
            await self._drain_queued_turn()

    async def _drain_queued_turn(self):
        """Answer whatever arrived while the last reply was being spoken.

        Held to one turn and to QUEUED_TURN_MAX_AGE. A question the caller
        asked twenty seconds and two answers ago is not one they are still
        waiting for, and answering it then is worse than not answering it.
        """
        while self._queued_turn:
            text, queued_at = self._queued_turn
            self._queued_turn = None

            if time.time() - queued_at > QUEUED_TURN_MAX_AGE:
                log.info("⌛ Dropping a stale queued turn (%.1fs old): %r",
                         time.time() - queued_at, text[:60])
                return
            if self.ws_closed or self.should_end:
                return
            log.info("↩️  Answering the queued turn: %r", text[:60])
            await self._handle_utterance(text)

    # ------------------------------------------------ half-sentences
    @staticmethod
    def _is_fragment(text: str) -> bool:
        """True when this turn is the run-up to a question rather than one."""
        words = [w for w in re.sub(r"[^\w\s]", " ", text.lower()).split() if w]
        return (0 < len(words) <= FRAGMENT_MAX_WORDS
                and all(w in FRAGMENT_WORDS for w in words))

    def _cancel_fragment(self):
        task, self._fragment_task = self._fragment_task, None
        if task and not task.done():
            task.cancel()

    def _take_pending(self, text: str) -> str:
        """Put a held fragment back on the front of this turn."""
        self._cancel_fragment()
        pending, self._pending_fragment = self._pending_fragment, None
        if not pending:
            return text
        if time.time() - self._fragment_at > 12:
            log.info("🧩 Dropping stale fragment %r", pending)
            return text
        joined = ("%s %s" % (pending, text)).strip()
        log.info("🧩 Joined the held fragment: %r + %r -> %r",
                 pending, text, joined)
        return joined

    async def _hold_fragment(self, text: str):
        """Say nothing and wait for the rest of the sentence."""
        self._pending_fragment = text
        self._fragment_at = time.time()
        log.info("🧩 Holding %r - waiting for the rest of the sentence", text)
        self._cancel_fragment()
        self._fragment_task = asyncio.create_task(self._fragment_timeout())

    async def _fragment_timeout(self):
        """The caller really did trail off. Ask once, after the wait - not
        instantly, and not once per fragment."""
        try:
            await asyncio.sleep(FRAGMENT_HOLD_S)
        except asyncio.CancelledError:
            return
        async with self._reply_lock:
            pending, self._pending_fragment = self._pending_fragment, None
            self._fragment_task = None
            if not pending or self.speaking or self.ws_closed:
                return
            log.info("🧩 Nothing followed %r - asking the caller to repeat",
                     pending)
            await self._speak(FALLBACK_LINE)

    async def _handle_utterance(self, text: str):
        log.info("🧠 User: %s", text)

        if not text:
            await self._speak(FALLBACK_LINE)
            return

        # Half a sentence from the previous turn belongs on the front of this
        # one, before anything looks at what was said.
        text = self._take_pending(text)

        # A turn that is only a run-up word is held rather than answered. Not
        # while the bot is waiting for a name or a time, though: those answers
        # are short by nature and "my" or "just" could be part of one.
        if (self._is_fragment(text)
                and not (self.awaiting_name or self.awaiting_name_confirm
                         or self.awaiting_callback_time)):
            await self._hold_fragment(text)
            return

        # The caller asking for a person is the highest-intent moment on the
        # call. Handle it deterministically, before the LLM, so a slow or
        # malformed completion can never lose it.
        # Answering "may I have your name?". Whatever they say, the callback
        # goes ahead - a caller who would rather not give a name still gets
        # their call back.
        if self.awaiting_name:
            self.awaiting_name = False
            name = extract_name(text)

            if name:
                self.lead_name = name
                self.awaiting_name_confirm = True
                log.info("🙋 Heard the name as %r - reading it back", name)
                await self._speak(CONFIRM_NAME.format(name=name))
                return

            # Nothing name-shaped came back. That is almost always the
            # recogniser, not a caller refusing to answer - "my name is
            # Karthi" came back as "security" - so ask once more before
            # giving up on it. A caller who genuinely will not give a name
            # still gets their call back: the number rang us.
            if self.name_attempts < 2:
                self.name_attempts += 1
                self.awaiting_name = True
                log.info("🙋 Nothing name-like in %r - asking once more",
                         text[:60])
                await self._speak(ASK_NAME_AGAIN)
                return

            log.info("🙋 No name in %r - carrying on without one", text[:60])
            await self._finish_callback()
            return

        # Answering "I have that as Karthi. Is that right?"
        if self.awaiting_name_confirm:
            self.awaiting_name_confirm = False
            answer = yes_no(text)
            corrected = extract_name(text)

            # "No, it's Karthi" carries the correction in the same breath as
            # the no, so take the name before treating it as a rejection.
            if corrected and corrected != self.lead_name:
                log.info("🙋 Name corrected to %r", corrected)
                self.lead_name = corrected
            elif answer is False:
                self.lead_name = None
                if self.name_attempts < 2:
                    self.awaiting_name = True
                    self.name_attempts += 1
                    log.info("🙋 Name wrong - asking once more")
                    await self._speak(ASK_NAME_AGAIN)
                    return
                log.info("🙋 Name still wrong after two tries - going without")
                await self._finish_callback()
                return

            log.info("🙋 Caller's name: %s", self.lead_name)
            await self._finish_callback()
            return

        if self.awaiting_callback_time:
            await self._take_callback_time(text)
            return

        # The bot offered a callback on the previous turn. Only an actual
        # "yes" books it. A caller who ignores the offer and asks something
        # else has not consented - drop the offer and answer the new question.
        if self.awaiting_handoff_consent:
            self.awaiting_handoff_consent = False
            closing = self.pending_exit
            self.pending_exit = False
            answer = yes_no(text)
            if answer is True:
                log.info("🙋 Caller accepted the callback offer")
                await self._request_callback(text)
                return
            if answer is False:
                log.info("🙅 Caller declined the callback offer")
                # Declining the CLOSING offer is the end of the call: they were
                # already saying goodbye when it was made, so anything other
                # than "is there anything else?" keeps them on the line for a
                # question they have not got.
                if closing:
                    await self._end_call()
                    return
                await self._speak(HANDOFF_DECLINED)
                return
            # Not a plain yes or no - but "can you schedule tomorrow at
            # around five" is an acceptance with the time attached, and so is a
            # bare "tomorrow at five". The caller answered by naming when.
            # Dropping those made the bot ask again and then book a DIFFERENT
            # time off a later turn (2026-09-05, 18:04:27).
            found = scheduling.resolve(text, self.caller_zone.get("zone"))
            if found["local"] or found["anytime"] or found["needs_hour"]:
                log.info("🙋 Offer accepted by naming a time: %r", text[:60])
                await self._request_callback(text)
                return

            log.info("Callback offer went unanswered - dropping it: %r", text[:60])

        if wants_human(text):
            await self._request_callback(text)
            return

        # "Can we schedule a call?" - they have already asked, so do not offer
        # them a callback they just requested. Say yes, then get the day and
        # time. wants_schedule() is careful to leave the knowledge-base
        # questions alone: "how can I book a plot", "what is the payment
        # schedule", "what is the booking amount".
        if wants_schedule(text):
            log.info("📅 Caller asked to schedule: %r", text[:60])
            await self._request_callback(text, ask_line=SCHEDULE_ASK_WHEN)
            return

        # Greetings and acknowledgements need no knowledge base and no LLM.
        # Sending "Hello?" to Groq cost half a second and, worse, dragged a
        # random KB entry in as "context" for the model to answer from.
        canned = self.brain.smalltalk_guard(text)
        if canned:
            log.info("💬 Smalltalk (no RAG, no LLM): %r", text[:60])
            self.history.append({"role": "user", "content": text})

            # They are saying goodbye. Make the one closing offer FIRST, in
            # place of the farewell - saying "have a great day" and then
            # asking a question reads as the bot talking over its own exit.
            if canned == EXIT_LINE and await self._offer_before_goodbye():
                return

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

        # A turn that matched the knowledge base is a real enquiry about the
        # project, which is what earns the closing callback offer.
        if context:
            self.asked_project_question = True

        self.history.append({"role": "user", "content": user_text})

        # LATENCY: speak each sentence as the model finishes writing it. The
        # buffered path below is kept behind LLM_STREAM_SENTENCES=false.
        if settings.llm_stream_sentences:
            sentences = await self._speak_as_generated(user_text, context)
        else:
            def _generate():
                return [s for s in self.brain.reply_stream(self.history, user_text,
                                                          context)
                        if s.strip()]

            sentences = await loop.run_in_executor(None, _generate)

            for sentence in sentences:
                print("🤖 BOT:", sentence)

            if sentences:
                await self._clear_outbound_audio()
                # Hand over the SENTENCES, not the joined string: tts.py
                # pipelines them so the caller hears sentence one while
                # sentence two is synthesising.
                await self._speak(sentences)

        if not sentences:
            log.warning("Brain returned nothing - speaking the fallback line")
            sentences = [FALLBACK_LINE]
            await self._speak(FALLBACK_LINE)

        self.history.append({"role": "assistant", "content": " ".join(sentences)})

        # METADATA is parsed from the raw completion, never from the spoken text.
        await self._handle_metadata(self.brain.parse_extra(self.brain.last_raw))

    async def _speak_as_generated(self, user_text, context):
        """Speak the reply WHILE the model is still writing it.

        The old path waited for the whole completion - including the METADATA
        block, which is never spoken - before the first frame of audio went
        out. Groq streams, so there is no reason to: each finished sentence
        reaches tts.py the moment it exists, and synthesis of sentence one
        overlaps with generation of sentence two.

        Returns the sentences actually spoken, in order, so the reply is
        recorded in history exactly as it was before.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        DONE = object()
        spoken = []

        def _produce():
            # reply_stream_live is a blocking generator doing blocking HTTP,
            # so it lives in a worker thread and hands sentences back to the
            # loop one at a time. The queue is unbounded on purpose: the
            # producer must never block, or a cancelled reply would strand
            # this thread and lose last_raw - and with it the METADATA.
            try:
                for sentence in self.brain.reply_stream_live(
                        self.history, user_text, context):
                    if sentence and sentence.strip():
                        loop.call_soon_threadsafe(queue.put_nowait,
                                                  sentence.strip())
            except Exception as e:
                log.exception("Reply generation failed: %s", e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, DONE)

        producer = loop.run_in_executor(None, _produce)

        async def _sentences():
            while True:
                item = await queue.get()
                if item is DONE:
                    return
                spoken.append(item)
                print("🤖 BOT:", item)
                yield item

        await self._clear_outbound_audio()
        await self._speak(_sentences())

        # The completion has to be finished before METADATA is read off
        # last_raw. It normally is - the sentence stream ends with it - but a
        # hung-up call can get here first.
        try:
            await asyncio.wait_for(asyncio.shield(producer), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("The completion is still running after the reply was "
                        "spoken - METADATA may be incomplete for this turn")
        except Exception as e:
            log.warning("Reply producer ended badly: %s", e)

        return spoken

    async def _take_callback_time(self, text: str):
        """The caller is answering "which day and what time would suit you?".

        Whatever they say is in THEIR timezone, not ours. Most callers here are
        NRIs, so "six in the evening" from Dubai is half past seven in
        Coimbatore - and from New Jersey it is half past three the next
        morning. Resolve it against the zone their number belongs to and keep
        both, because the person who has to make the call is in India.
        """
        said = " ".join(p for p in (self.callback_said, text) if p).strip()
        found = scheduling.resolve(said, self.caller_zone.get("zone"))

        # A day with no hour is not a time. Ask once, keeping the day.
        if found["needs_hour"] and not self.asked_for_hour:
            self.asked_for_hour = True
            self.callback_said = said
            log.info("📅 %r names a day but no hour - asking once", said)
            await self._speak(HANDOFF_ASK_HOUR)
            return

        self.awaiting_callback_time = False
        self.callback_said = None
        self.callback_raw = said

        if found["local"]:
            self.callback_local = found["local"]
            self.callback_ist = found["ist"]
            self.callback_time = scheduling.describe(
                found["local"], found["ist"], self.caller_zone.get("name"))
            self.callback_time_for_caller = scheduling.describe_for_caller(
                found["local"], found["ist"], self.caller_zone.get("name"))
            log.info("📅 Callback: %s", self.callback_time)
        else:
            # "Anytime", or an answer with no time in it at all. Write down
            # what they said rather than invent a slot for them.
            self.callback_time = (found["phrase"] or extract_time(said)
                                  or "not specified")
            log.info("📅 Callback time left open: %r", self.callback_time)

        # Persist the time BEFORE asking anything else. Whatever the caller
        # does next - gives a name, refuses one, or simply hangs up - the slot
        # they asked for is already on the row.
        await self._record_callback()

        # "David, tomorrow at six" answers both questions in one breath, so
        # asking for a name they just gave is a wasted turn. Only the comma
        # form is trusted here: without one, "whenever you like" reads as a
        # name. It still gets read back, because Deepgram mishears names with
        # total confidence.
        if not self.lead_name and "," in text:
            volunteered = extract_name(text)
            if volunteered:
                self.lead_name = volunteered
                self.asked_for_name = True
                self.awaiting_name_confirm = True
                log.info("🙋 Name volunteered with the time: %r", volunteered)
                await self._speak(CONFIRM_NAME.format(name=volunteered))
                return

        await self._ask_for_name()

    # ------------------------------------------------ closing the call
    async def _offer_before_goodbye(self) -> bool:
        """Offer the callback once, on the way out. True if it was made.

        Until now the ONLY thing that offered a callback was the bot failing
        to answer something, so a caller whose questions were all answered -
        the interested one - hung up without ever being asked whether they
        wanted to hear from a person. This asks, once, and only when the call
        was a real enquiry: no offer after a wrong number, and none after a
        caller who already has a callback booked or has already turned one
        down.
        """
        if (self.callback_requested or self.callback_offered
                or not self.asked_project_question
                or self.ws_closed):
            return False

        self.callback_offered = True
        self.awaiting_handoff_consent = True
        self.pending_exit = True
        log.info("🙋 Closing callback offer - waiting for the caller's answer")
        self.history.append({"role": "assistant", "content": CLOSING_OFFER})
        print("🤖 BOT:", CLOSING_OFFER)
        await self._clear_outbound_audio()
        await self._speak(CLOSING_OFFER)
        return True

    async def _end_call(self):
        """Say goodbye, send the brochure, and let server.py hang up."""
        if not self.brochure_sent:
            await self._deliver_brochure()
        self.history.append({"role": "assistant", "content": EXIT_LINE})
        print("🤖 BOT:", EXIT_LINE)
        await self._speak(EXIT_LINE)
        log.info("👋 Closing the call")
        self.should_end = True

    # ------------------------------------------------ callback to a human
    async def _request_callback(self, text: str, ask_line: str = None):
        """Caller wants a callback. Schedule one instead of a live transfer,
        then confirm it over WhatsApp/SMS when the call ends.

        The order is the day and time first, then the name. `ask_line` is what
        to say when they have not given a time yet, so the two ways in can
        sound different: HANDOFF_ASK_TIME when they asked for a person,
        SCHEDULE_ASK_WHEN when they asked to schedule.
        """
        log.info("🙋 Callback requested: %r", text)
        self.callback_requested = True
        self.history.append({"role": "user", "content": text})

        # A scheduling-only call never reaches the LLM, so METADATA never fills
        # "requirement" in. Record what the caller actually said instead of
        # leaving the sales sheet blank. A later completion still wins.
        if not self.lead_requirement:
            self.lead_requirement = " ".join(text.split())[:120]

        await self._record_callback()

        self.awaiting_callback_time = True

        # They may have named the time in the same breath - "call me back
        # tomorrow at six". That has to go through the same resolver as any
        # other answer, or an NRI's time lands in the sheet as if it were IST.
        found = scheduling.resolve(text, self.caller_zone.get("zone"))
        if found["local"] or found["anytime"] or found["needs_hour"]:
            await self._take_callback_time(text)
            return

        await self._speak(ask_line or HANDOFF_ASK_TIME)

    async def _ask_for_name(self):
        """The time is settled - now the name, asked once and only once.

        A caller who will not give one still gets their call back: the number
        rang us, so the row is actionable either way. This is the last question
        of the booking, so anything that skips it goes straight to the
        confirmation.
        """
        if self.lead_name or self.asked_for_name:
            await self._finish_callback()
            return

        self.asked_for_name = True
        self.awaiting_name = True
        self.name_attempts += 1
        await self._speak(ASK_NAME)

    async def _finish_callback(self):
        """Day, time and name are settled - write the row and say so out loud.

        The WhatsApp and SMS confirmations do NOT go out here: they are sent
        from on_stop(), once the call is over and nothing more can change.
        """
        await self._record_callback()
        await self._speak(HANDOFF_CONFIRMED)

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

        # The appointment on its own. It used to carry the brochure with it,
        # and a single message that confirmed a callback and then said "here
        # is our project brochure" plus a URL was read as the brochure
        # message - callers said they never got the appointment at all. The
        # brochure follows as its own message, from on_stop().
        channel = await loop.run_in_executor(
            None,
            lambda: deliver_callback_confirmation(
                self.caller_number, self.lead_name,
                # Their clock, not the office's. An NRI told "7:30 PM IST" has
                # to work out for themselves whether that is the six o'clock
                # they asked for, and the one who gets it wrong misses the call.
                self.callback_time_for_caller or self.callback_time))

        self.notified_via = channel
        if channel and channel != "failed":
            self.callback_confirmed = True
            log.info("📅 Callback confirmed to %s via %s",
                     self.caller_number, channel)
        else:
            log.error("📅 Callback confirmation FAILED for %s", self.caller_number)

        alerted = await loop.run_in_executor(
            None,
            lambda: notify_agent(self.caller_number, self.lead_name,
                                 self.lead_requirement, self.callback_time))
        self.agent_alerted = bool(alerted)

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
        if meta.get("intent_score"):
            self.lead_intent = meta["intent_score"]

        # handoff=true means the bot has just ASKED "shall I arrange a call?".
        # That is a question, NOT consent. Recording it here booked a callback
        # for a caller who never answered, and he was sent a text confirming an
        # appointment he had not agreed to (2026-09-04, 15:15:27). Mark the
        # offer pending and let the next turn decide.
        if meta.get("handoff") and not self.callback_requested:
            log.info("🙋 Callback offered - waiting for the caller to accept")
            self.awaiting_handoff_consent = True
            self.callback_offered = True

        if meta.get("whatsapp_wanted") and not self.brochure_sent:
            await self._deliver_brochure()

        if meta.get("end_conversation") and self.interrupted:
            # The caller cut this reply off, so they never heard the goodbye
            # it ends with - and they are talking RIGHT NOW. Hanging up on
            # them mid-sentence because a completion they interrupted said
            # the call was over would be the worst possible reading of it.
            log.info("Ignoring end_conversation from an interrupted reply")
            return

        if meta.get("end_conversation"):
            # One offer on the way out, if this caller has not had one. The
            # model has already spoken its farewell by the time METADATA is
            # parsed, so the offer follows it rather than replacing it - and
            # the call stays open until they answer.
            if await self._offer_before_goodbye():
                return
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
        if isinstance(text, str):
            sentences = [text] if text.strip() else []
        elif hasattr(text, "__aiter__"):
            # A live sentence stream from the model. tts.py pulls from it, so
            # there is nothing to inspect here and nothing to wait for.
            sentences = text
        else:
            sentences = [s for s in (text or []) if s and s.strip()]

        streaming = hasattr(sentences, "__aiter__")

        if self.ws_closed or (not streaming and not sentences):
            if streaming:
                try:
                    await sentences.aclose()
                except Exception:
                    pass
            return

        self.speaking = True
        self.interrupted = False
        self.barge.reset()
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
            if sent and not self.ws_closed and not self.interrupted:
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
            # The line has been carrying our own audio, not the caller's
            # background. Measure the noise floor again from scratch.
            self.gate.reset()
            # An interruption gets the short tail: the queue was cleared, so
            # there is almost nothing draining, and the caller is ALREADY
            # talking - every millisecond of tail is a millisecond of what
            # they said that nobody hears.
            tail = (settings.barge_in_tail_ms if self.interrupted
                    else settings.echo_tail_ms)
            self.stt.unmute(tail / 1000.0)

    def on_mark(self, name: str):
        """Twilio confirms the caller has finished HEARING a marked chunk."""
        if name and name == self._expected_mark:
            self._playback_done.set()

    async def _on_barge_in(self):
        """The caller is talking over the bot. Stop talking.

        Order matters and all three steps are needed. `interrupted` stops the
        TTS loop at the next frame; clearing Twilio's queue drops the audio
        already sent but not yet played - without it the bot keeps talking for
        the length of whatever is buffered, which is the whole reason the
        first attempt at this "did not work"; and releasing the playback mark
        stops _speak() waiting for a mark that will never come for audio that
        has been thrown away.
        """
        if self.interrupted:
            return
        self.interrupted = True
        log.info("🙋 Caller interrupted (echo≈%.0f, threshold %.0f) - stopping",
                 self.barge.echo_level, self.barge.threshold())
        await self._clear_outbound_audio()
        self._playback_done.set()
        # Open the mic NOW. _speak's own unmute is still coming, with the
        # interrupted tail, but the caller is mid-sentence and the first word
        # of it is already on its way up the line.
        self.stt.unmute(settings.barge_in_tail_ms / 1000.0)

    async def _is_cancelled(self):
        """Stop mid-sentence if the caller hung up or interrupted, instead of
        streaming the rest of the reply into a dead socket or over the top of
        somebody who is trying to speak."""
        return self.ws_closed or self.interrupted

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
