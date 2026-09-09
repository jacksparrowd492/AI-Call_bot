# Changelog and open defects

Two markdown files live in this repo and no more: `README.md` says how the
service works, this one says what changed and what is still wrong. Dated
diagnosis notes get folded in here rather than kept as separate files.

---

## 2026-09-09 — back to one language, and the dead code is gone

Two things in one pass: the multilingual layer is **removed**, and every file
nothing imported is deleted. The pipeline is unchanged — Twilio -> Deepgram ->
Groq + Chroma -> Kokoro, with the barge-in, the booking order and the call log
exactly as they were.

### Why the languages went

Tamil, Hindi, French and Spanish worked. They were removed anyway, for reasons
that are about the deployment rather than the code:

- `language=multi` costs the **en-IN** model. Deepgram's multilingual stream has
  no Indian-English variant, and the whole en-IN + keyword-boost + `vocab.repair()`
  stack exists because Indian names and place names were the thing going wrong.
  Trading that away to gain French is the wrong trade for this project.
- Tamil has **no Kokoro voice**, so it put a paid ElevenLabs HTTP call per
  sentence on the hot path — in the one stage that was otherwise free, local and
  never a latency risk.
- The sales team is the constraint. A caller answered in French and then rung
  back by someone who speaks Tamil and English is worse served than a caller
  kept in English throughout.

Removed: `i18n.py`, `tools/check_languages.py`, and the language surface inside
`config.py` (`SUPPORTED_LANGUAGES`, `MULTILINGUAL`, `DEEPGRAM_MULTI_MODEL`,
every `ELEVEN_*`), `stt.py` (`_model_and_language`, `switch_language`,
`detected_language`, `_language_of` — the stream is the tuned en-IN nova-2 one
again), `tts.py` (`_synthesize_elevenlabs` and the whole ElevenLabs path, the
`language` argument, `KokoroStreamer.language`), `llm.py` (the "answer in the
caller's language" system turn, `to_english`) and `bridge.py`
(`_switch_language`, `_language_from_speech`, `i18n.localize` on every canned
line). `.env` lost the same keys, including a live ElevenLabs key that is now
unused — **revoke it at ElevenLabs**, since removing it from `.env` does not.

`tts.ElevenLabsStreamer` was only ever an alias for `KokoroStreamer`; the alias
is gone and `bridge.py` imports the real name.

`_boost_param()` stays, though the call now only ever runs on nova-2. It keys off
`DEEPGRAM_MODEL`, which is still an env var, and sending `keywords` to a nova-3
model is not an error — it is silently ignored, which would drop the entire
proper-noun boost with no symptom but the names going wrong again.

### Deleted because nothing imported it

- `tools/leads.py` — `upsert_lead()`, `init()` and `all_leads()` were called by
  nothing, and the `GET /leads` endpoint the README advertised does not exist in
  `server.py`. `tools/call_log.py` covers the same ground and *is* wired in.

  **`leads.db` itself stays, and so does `LEAD_STORE_PATH`.** The file is not
  orphaned with `leads.py` gone: `tools/callbacks.py` opens it through
  `settings.lead_store_path` and writes a `callbacks` row for every booking, on
  the path `bridge.py` takes when a caller accepts a callback. Deleting that
  setting broke the booking flow and the test suite did not catch it — the
  import is lazy and stubbed — so it is restored, with a comment saying who the
  real user is. The name is now misleading and was left alone anyway: renaming
  it earns nothing in a pass that is meant to be behaviour-neutral.
- `tools/handoff.py` — `handoff_to_agent()` did a live-call TwiML update to
  `<Dial>`. Superseded on 2026-09-04: the bot offers a callback and confirms it
  in writing, and never transfers a live call.
- `test_deepgram.py` — a scratch connectivity check from the first week.
  `tools/smoke_test.py` does this and the rest of the vendors properly.
- `data/realestatedata.json` — the pre-`build_kb.py` knowledge base, superseded
  by `data/knowledge_base.json` (40 entries, 310 phrasings).
- `data/brochure_extract.json` — referenced by no file in the repo.
- Nine orphaned HNSW directories under `chroma_db/` — segments left behind by
  past ingests, belonging to no row in the `segments` table.

All of it is in git history if it is ever wanted back.

### The vector store was NOT rebuilt, and did not need to be

`chroma_db/` holds two collections: `real_estate` (353 documents), which is the
one `CHROMA_COLLECTION` names and the service opens, and `karthipuram` (426
documents, cosine space) left over from an older KB shape and opened by nothing.

`real_estate`'s `kb_sha1` stamp is `f3e62398…`, which is the sha1 of the current
`data/knowledge_base.json` — the store is already current, so a rebuild would
have produced the same vectors. The nine orphaned segment directories were
removed; `chroma.sqlite3` was not touched.

Dropping the stale `karthipuram` collection needs Chroma itself, and therefore
the MiniLM ONNX model, which is what the next ingest will do anyway:

    python -m rag.ingest        # drops and rebuilds real_estate
    python -m tools.ask_kb --check

### Tests

`tests/test_bridge.py` **stopped crashing**: `FakeSTT` had no
`detected_language`, so the bridge's auto-detect path raised `AttributeError` on
the first turn and the whole suite died at check 5. With the language code gone
it runs: **59 passed, 4 failed**.

The four failures pre-date this pass — the same four fail on the previous commit
once `FakeSTT` is given the missing attribute — and two of them are real:

- **Stale.** "the second turn is dropped, not interleaved" still expects the
  pre-H behaviour, where a turn arriving mid-reply was discarded. H queues it, so
  both turns now run. The test is wrong, not the code.
- **Stale.** "the sales team's note carries both times" wants
  `(6:00 PM Asia/Dubai)`; the code writes `(6:00 PM in Dubai)`.
- **REAL.** "and is confirmed rather than asked again" and "and the booking is
  confirmed anyway, time intact" both end in *"Sorry, I didn't catch that. Could
  you say just your name, slowly?"* — a name that should have been extracted was
  not, so the caller is asked twice. This is on the booking path, which is the
  highest-intent moment on the call. Not fixed here; this pass was a cleanup and
  is deliberately behaviour-neutral.

Everything else is green, and unchanged from before the cleanup:
`test_handoff.py` 80/80, `test_scheduling.py` 43/43, `test_stt.py` 31/31,
`test_tts.py` 14/14, `test_llm.py` 43/43, `test_latency_noise.py` 45/45.
`test_rag.py` is 55/57 both before and after, and needs the real vector store.

The `language="en"` parameter was dropped from the `fake_synth` stubs in
`test_tts.py` and `test_latency_noise.py` to match `_synthesize_sync`.

### Still open

Defect **K** (the call log has no language column) is **void** — there is one
language. C, D, E, G and J below are unchanged, plus the two real test failures
above.

---

## 2026-09-06 (night) — the five open defects

A, B, F, H and I off the "Still open" list below, which is rewritten at the
bottom of this file to what is actually left.

### I. Barge-in — the caller can interrupt now

`stt.mute()` stopped forwarding audio for the whole time the bot spoke, so a
caller could not cut into a fifteen-second answer: they talked, nothing
happened, and they waited.

The hard part is not the interrupting, it is that **Twilio echoes the bot's own
audio back up the inbound stream**. The frames are never quiet while the bot
talks, so an absolute threshold cannot work: above the echo on a quiet line and
nobody on a loud one can ever interrupt; below it and the bot interrupts
itself on every reply, which is far worse.

`audio_utils.BargeInDetector` therefore MEASURES the echo, per call, and a
barge-in has to be loud relative to it — two voices on a line are louder than
one, and that is the whole signal. Three guards stop false positives: a 500ms
grace (the echo level is unknown at the start of a reply, and the caller's own
last word is still arriving), a 280ms sustain (one loud frame is a cough or a
door), and an absolute floor for lines so quiet the echo never registers.

Two things had to be got right, and the tests pin both:

- **The echo estimate tracks the PEAK, not the floor** — the opposite of
  `NoiseGate`, deliberately. Tracking the minimum put the estimate down in the
  gaps between the bot's own words, and its next syllable then cleared its own
  threshold: the bot interrupted itself.
- **The estimate freezes once a barge-in is in progress.** Letting those frames
  teach it is a race the caller loses — the estimate climbs towards their
  voice, the threshold climbs with it, and on a loud line it overtakes them
  before the sustain window is up. The louder they shouted, the less likely
  they were to be heard. This one was found by test 7 and not by reading.

On detection: `interrupted` stops the TTS loop at the next frame, Twilio's
queued audio is **cleared** (without this the bot keeps talking for the length
of whatever is buffered, which is why a naive version looks like it does not
work), the playback mark is released so `_speak` is not waiting for a mark for
audio that was thrown away, and the mic reopens on a 120ms tail instead of 550.

`BARGE_IN=false` restores the old behaviour exactly.

### H. A turn arriving mid-reply is queued, not dropped

`_reply_lock` discarded anything said while the previous answer was being
spoken — most often the caller repeating themselves *because* the bot had not
answered yet, so the retry was the thing thrown away. One slot deep and
20 seconds stale-dated: by the time the bot is free the oldest thing said is
the least worth answering, and replaying a backlog would talk over the caller
for half a minute.

### A. Same question, two different answers

`temperature=0.3` on a turn whose answer is already in the prompt made it a
lottery — two identical questions with identical retrieval (best d=0.777)
returned different subsets of the same entry. A grounded turn is
reading comprehension, not writing: `GROUNDED_TEMPERATURE = 0.0` when the
retriever found something, `0.3` when it did not and the turn is small talk.

### B. Dead code

- `vocab.repair()` is finally called — in `stt._flush()`, before the transcript
  goes anywhere. "Cartigram", "Kartivaram" and "Artipuram" become
  "Karthipuram" ahead of the routing and the search. English only: the matcher
  measures similarity against a Latin-script lexicon, so running it over Tamil
  or Devanagari could only damage the transcript.
- `router.py`, `tools/eval_router.py` and `tools/test_logic.py` are **deleted**.
  Everything the router decided, `bridge.py` decides inline now, and
  `test_logic.py` had been broken since `config.py` dropped
  `route_strong_threshold`. In git history if it is ever wanted back.

### F. phonemizer: words count mismatch

espeak aligns phonemes to WORDS, and that warning is it saying the two no
longer line up — which is not cosmetic, it risks **clipped audio**, and it fired
on 4 of 7 syntheses. Every trigger is a token that is one word to Python and
more (or fewer) to espeak. `tts._expand_for_espeak()` now handles them:

    D-Mart          -> D Mart          2400 sq.ft  -> 2400 square feet
    DTCP / RERA     -> D T C P ...     80/60 feet  -> 80 60 feet
    L&T             -> L and T         24x7        -> twenty four seven
    Rs. 5,500       -> rupees 5,500    3BHK        -> three b h k

`no.` keeps its full stop as a requirement on purpose — making it optional
turned every caller's "No, thank you" into "Number, thank you".

### Tests

`tests/test_latency_noise.py` gains 16 barge-in checks: four that the bot never
interrupts itself (steady echo, loud line, gaps between its own words, its own
audio getting louder), four that a real interruption IS caught (including on a
loud line and over a gappy reply, within a third of a second), three that
clicks, short bursts and the caller's own trailing word are not interruptions,
and four on the bridge acting on it. The stubs in that file were updated for
the new `temperature` and `language` parameters.

---

## 2026-09-06 (night) — Tamil, Hindi, French and Spanish

A call can now be held in English, Tamil, Hindi, French or Spanish. One new
module, `i18n.py`, holds everything that knows there is more than one language;
every other file keeps its shape and its English constants.

    python -m tools.check_languages

New, and it checks the four things that actually break: every canned line is
translated, every Kokoro voice id exists in `models/voices.bin`, the Deepgram
URL asks for the right model and the right boost parameter, and a caller asking
for a language by name is understood. All pass. Sound quality still needs a
real call — that is the part no script can check.

### What the vendors allow, which is what the design is

- **English, Hindi, French, Spanish are free and automatic.** Deepgram's
  `language=multi` picks all four out of one stream with no prompt and no
  reconnect, and Kokoro already has a voice for each (`hf_alpha`, `ff_siwis`,
  `ef_dora`). `stt._language_of()` reads the detected language off the
  per-word `language` fields — the majority of the words wins, so one English
  word inside a French sentence does not flip the call.
- **Tamil is neither.** Deepgram supports `ta` on **nova-3 only** and **not in
  the multi set**, so it cannot be detected — only asked for, and the stream
  has to be reopened on `language=ta` to get it. And Kokoro has no Tamil voice
  at all: its 54 voices are English, Spanish, French, Hindi, Italian,
  Portuguese, Japanese and Chinese, and that is the whole list. Tamil speech
  therefore goes out through **ElevenLabs** (`eleven_turbo_v2_5` supports it),
  which is why `ELEVENLABS_API_KEY` is now uncommented in `.env`. With no key
  Tamil is still understood — it is answered in English, and the caller is told
  so, rather than the line going silent.
- A caller enters Tamil by asking: *"can you speak Tamil"*, *"தமிழ்ல பேசுங்க"*,
  *"தமிழில் சொல்லுங்கள்"*. The trigger is the stem **தமிழ** with no pulli,
  because Tamil agglutinates and the citation form matches only one of those.
  Asking for a language also **locks** it — the detector stops second-guessing
  a caller who has stated a preference.

### The hooks, all small on purpose

- `bridge._speak()` runs `i18n.localize()` over every line on its way to the
  voice. That one hook is why `bridge.py` and `llm.py` still write their canned
  lines in English: the table is keyed by the English text. Anything the model
  wrote is not in the table and passes through, because the model was already
  told which language to answer in.
- `llm._build_messages()` adds a system turn naming the language — as a turn,
  not in `SYSTEM_PROMPT`, because the language can change mid-call and the
  prompt is a module constant built at import. METADATA stays English so the
  sales sheet stays readable.
- **The knowledge base stays in English.** `all-MiniLM-L6-v2` is an English
  embedder, so a Tamil or French question is translated for the *lookup only*
  (`llm.to_english()`) and the answer is written back in the caller's language.
  Translating the KB would mean four copies of every answer and four sets of
  vectors to keep in step.
- `tts.KokoroStreamer.language` is an instance attribute, not a module global:
  two callers can be on the line at once in two languages, and a global would
  give them each other's voice.
- `tts._normalise_for_speech()` now takes the language. Its ASCII fold — added
  to stop a curly apostrophe splitting a word in espeak — was deleting the
  reply in every other language: *français* lost its cedilla and Tamil was
  erased down to the punctuation. English only now.

### Two things this turned up

**nova-3 renamed `keywords` to `keyterm`.** Sending the old one is not an
error, it is silently ignored — so switching multilingual on would have thrown
away the entire Karthipuram and name boost, and the only symptom would have
been the proper nouns going wrong again. `stt._boost_param()` picks by model.

**`switch_language()` was cancelling the task it runs on.** The utterance-end
path runs the whole turn *inside* the receiver task (`_receiver` awaits
`_flush`, which awaits `on_utterance`), so cancelling `self._tasks` to reopen
the socket would have killed the reply that asked for the switch. It now skips
`asyncio.current_task()`; the old receiver ends on its own when its socket
closes.

### Worth knowing

`language=multi` has **no Indian-English variant** — the tuned `en-IN` stream
is nova-2 only. Indian names and place names will come back slightly worse than
they did. `MULTILINGUAL=false` in `.env` reverts to exactly the stream this
project has been running, in one line.

---

## 2026-09-06 (night) — the ISD clock, checked without an ISD call

Every call so far has come from +91, where the caller's zone and the office's
zone are the same instant, so `callback_local` and `callback_ist` always
matched and the conversion was never actually exercised. The bug that costs a
lead only appears on an international call.

### `python -m tools.check_timezones`

New. Drives the same three steps a live call does — `zone_for()` on the number
that rang us, `resolve()` on what the caller said, `describe()` for the sheet —
for Dubai, Qatar, London, New Jersey, San Francisco, Singapore, Sydney,
Malaysia and Kuwait, and checks each result against an independent conversion
straight from the tz database. No phone, no Twilio, no minutes.

    python -m tools.check_timezones
    python -m tools.check_timezones +971501234567 "tomorrow at 6 in the evening"

All ten cases pass: the two sheet columns are always the same instant, the
booking is always in the **caller's** future (a 9 am asked for at 11 am their
time is tomorrow, worked out in their morning, not ours), and a bare period
("tomorrow evening") still invents nothing.

### Two things it turned up

**A UK mobile came back as `Europe/Guernsey`.** Correct to the second — same
clock as London — but "Guernsey" in front of a sales team ringing Manchester
is noise. The Crown Dependencies, Belfast and Büsingen are now aliased to
their parent zone in `_ALIASES`.

**The caller was being texted the office's clock.** An NRI who says "six in
the evening" and receives *"They will call you Mon 7 Sep, 7:30 PM IST"* has to
do the arithmetic themselves to check it is the six o'clock they asked for,
and the one who gets it wrong misses the call.

- New `scheduling.describe_for_caller()` — their clock leads, IST follows:
  *"Mon 7 Sep, 6:00 PM your time (7:30 PM IST)"*.
- `bridge` keeps both: `callback_time` (IST first) for the sales sheet and the
  agent alert, `callback_time_for_caller` for the caller's own confirmation.
- `describe()` now says *"(6:00 PM in Dubai)"* rather than
  *"(6:00 PM Asia/Dubai)"* — new `scheduling.place()`. An IST caller is
  unaffected: their line is still just *"Tue 8 Sep, 5:00 PM IST"*.

---

## 2026-09-06 (late) — the appointment message, the WhatsApp sender, and the name

Three reports from the test call after the previous entry shipped.

### 1. The caller got a brochure and no appointment

They were sent one message that confirmed the callback *and* ended with
*"Meanwhile, here is our project brochure."* plus the URL — so it reads as a
brochure message with some words in front of it, and the appointment was
effectively invisible. The confirmation is now its **own** message.

- `whatsapp.callback_confirmation()` / `deliver_callback_confirmation()`
  default to `include_brochure=False`.
- `bridge._confirm_callback()` no longer marks the brochure as sent, so
  `on_stop()` sends it after, as a second message. Order on the wire: the
  appointment (WhatsApp **and** SMS, because an appointment should not depend
  on one channel), then the brochure (WhatsApp, SMS if not available).

### 2. Everything now goes out from the project's own WhatsApp number

`TWILIO_WHATSAPP_FROM` was still `whatsapp:+14155238886` — Twilio's shared
sandbox — which is why every message failed 63007 before falling back to SMS.

- `.env` now carries `whatsapp:+919418646464`.
- `Settings._as_whatsapp_sender()` accepts `9418646464`, `+919418646464` or the
  full `whatsapp:` form and normalises all three, with `DEFAULT_COUNTRY_CODE`
  (`+91`) filling in a bare local number. A number typed without the prefix
  used to be sent to Twilio verbatim and rejected in a way that looks exactly
  like a channel that was never registered.
- A 63007-class failure now switches WhatsApp **off for the rest of the run**
  (`whatsapp._whatsapp_off`) instead of costing a failed round trip — about
  0.7s — in front of every SMS. It says so once, loudly, with what to fix.
- `notify_agent()` goes through the same WhatsApp-then-SMS path, so the sales
  team's alert is not stuck on the voice number.

**If +91 94186 46464 is not registered as a WhatsApp sender on this Twilio
account, messages will still go out by SMS** — that is Twilio configuration,
not code.

### 3. "My name is Karthi" was written down as "Security"

Self-inflicted, by the previous entry. `TOPIC_TERMS` boosted `security`,
`facilities`, `layout`, `plot` and friends — ordinary words the recogniser
already knows — and a boosted word is one the model reaches for when it is
unsure. The moment it is least sure is a name it has never heard.

- `TOPIC_TERMS` is cut back to the words that were actually mangled on a call:
  transport, connectivity, infrastructure, water, supply, borewell, sewage,
  drainage, electricity, amenities.
- New `vocab.NAME_TERMS` — common first names, boosted at 2, **boost only**,
  deliberately not in `LEXICON` (repairing a caller's word into a name they
  never said would be worse than the mishearing).
- New `DEEPGRAM_EXTRA_KEYWORDS` in `.env`: comma-separated, boosted hardest,
  for whatever this deployment watches go wrong.
- `handoff_intent._NOT_A_NAME` now rejects the project vocabulary outright, so
  "Security" can never reach the sales sheet as a caller's name.
- `bridge` asks **once more** when nothing name-shaped comes back, instead of
  silently booking the callback with no name. `ASK_NAME_AGAIN` asks for just
  the name, slowly. A caller who still will not give one is not pestered - the
  number rang us, the row is actionable either way.

---

## 2026-09-06 (evening) — the transport answer, the half a question, and the offer nobody got

From the 21:17 call log (`CAe6790d3bbd9bded8bad34f52d237c991`, +91 63827 79352).

### 1. "transport facilities?" was answered with what the township does NOT have

Retrieval was fine — `best_d=0.916`, straight onto `transport_connectivity`.
The **answer** was the problem: it opened *"There's no metro station inside the
township itself"*, because the entry had been written to settle a metro
question and then reused for every transport question there is. A caller asking
how they will get around heard a denial first and the connectivity second.

- `transport_connectivity` now answers connectivity first — the State Highway
  road, fifteen entrances onto 80- and 60-foot roads, the airport nearby, the
  highway and metro projects — and is titled and phrased for transport, not
  metro. Sixteen question phrasings, including the bare *"transport
  facilities"* the caller actually said.
- The metro caveat moved to its own entry, **`metro_status`**, so *"is there a
  metro station?"* still gets the honest no and nothing else has to carry it.

### 2. "Infrastructure and water supply" was answered as if only the first half was asked

Deepgram returned it as *"Infrastructure and, how many days?"*, and one
embedding of a two-topic sentence sits between both topics and matches neither
well: `best_d=1.162` against the roads entry, and the water entry — which
answers it in full — was never reached. The bot read out the road widths, said
it did not have a timeline, and offered a callback for a fact it was holding.

- `rag/retriever.retrieve()` now splits a turn on `and` / `,` / `also` and
  searches each topic **as well as** the whole sentence, merging the hits.
  Only once the whole turn has matched something, though: a clause matches more
  loosely than the sentence it came from, and splitting first would give
  *"Hello there, can you hear me?"* two extra chances to drag a knowledge entry
  into a greeting. Clauses get no short-turn allowance for the same reason.
- `water_supply` gained the phrasings callers use (*"how is the water supply"*,
  *"water facility"*, *"what about water"*) and `transport`/`water`/`supply`/
  `infrastructure` and friends are now **boosted in Deepgram** (`vocab.TOPIC_TERMS`,
  intensity 1 — a thumb on the scale, not a shove, since they are ordinary
  English words that must never be *repaired* into project nouns).
- `RAG_MAX_DISTANCE` was **1.2** in `.env` while `config.py` documented 1.5.
  1.2 was rejecting real questions; it is now 1.45.
- `detect_intent()` matched `"call"`, `"cost"` and `"rate"` as substrings, so
  *"he called me"* and *"accurate"* were routed to buckets that then filtered
  the answer out of reach. Whole words only now.

### 3. The only caller ever offered a callback was one the bot had failed

`handoff: true` is set on the not-found path, so a caller whose questions were
all answered — the interested one — hung up without ever being asked whether
they wanted to hear from a person.

- `bridge.MediaStreamBridge._offer_before_goodbye()` makes **one** offer on the
  way out of any call where the caller asked at least one real project question
  (a turn that matched the knowledge base). Never twice, never after a booking,
  never after a caller has already turned one down, and never on a wrong
  number. Declining it ends the call; accepting it goes into the ordinary
  day/time/name booking.

### 4. Half a sentence was answered as if it were a whole one

The turn arrived as *"Well,"* → *"what"* → *"transport facilities?"*. The first
two each cost an LLM round-trip and each got *"I'm sorry, I didn't catch that."*
before the real question ever landed.

- A turn of one or two words that are **only** run-up words (`well`, `so`,
  `what`, `how about`, `tell me`, …) is now **held**, not answered, and joined
  to the front of the next turn. If nothing follows within 3 seconds the bot
  asks them to repeat — once, and only then. Never while it is waiting for a
  name or a time, where a one-word answer is the point.

### 5. Five seconds of dead air on a Groq 429

`Retrying request to /chat/completions in 5.000000 seconds` — the SDK's default
retry, with the caller listening to nothing.

- `max_retries=0` on the client, `LLM_TIMEOUT_S` (12s) as the whole budget, and
  a 429 now moves **sideways to another model id**, which has its own
  tokens-per-minute bucket, for the rest of the call. Sleeping is never the
  answer on a live call.

### 6. Smaller things in the same log

- *"words count mismatch on 100.0% of the lines"* on every reply: the model
  writes `I’m` with a curly apostrophe and espeak splits the word on it.
  `tts._normalise_for_speech()` now flattens smart quotes, dashes, ellipses,
  `%`, `₹` and emoji to plain ASCII before Kokoro sees them.
- One booking logged `Callback recorded` four times — it is an upsert, and only
  the insert is worth an INFO line now.
- A stale vector store was invisible: `rag/ingest.py` stamps the KB file's
  **SHA1** into the collection, and `rag/retriever` warns at startup when
  `data/knowledge_base.json` has been edited since the last build.
- Still open, and not code: WhatsApp fails 63007 (`TWILIO_WHATSAPP_FROM` is the
  shared sandbox number and this account has no channel on it) so every message
  costs a failed request before falling back to SMS, and the sales alert fails
  21608 because `AGENT_DIAL_NUMBER=+919800011122` is not a verified number on
  this trial account.

**After pulling this: `python -m rag.ingest`, then `python -m tools.ask_kb --check`.**
The vectors are a build artefact of the JSON; editing the JSON alone changes
nothing on a call.

---

## 2026-09-06 — transport answers, and the callback asks in the caller's order

Three faults, all reproduced from the call log and the `leads.db` rows for
`CA8a59f4b9...` (2026-09-06 10:52) and lead #9 (2026-09-02 18:58).

### 1. "Transport service." was answered with a question, every time

The caller said *"Transport service."* and got *"Could you let me know which
transport details you'd like?"*. Two independent causes, both fixed:

**The knowledge base being searched had no transport entry at all.** The live
Chroma collection (`real_estate`) had been built from `data/realestatedata.json`
— 36 entries, nothing about transport, metro, buses or commuting. The authored
`data/knowledge_base.json` has had a `transport_connectivity` entry all along,
with the answer the bot was supposed to give; it had simply never been the file
the vectors were built from, and nothing reads the JSON at runtime, so there
was no way to see that from a call.

- `rag/ingest.py` now reads `data/knowledge_base.json` by default and
  understands its schema (`entries` + `escalations`), on top of the two older
  shapes.
- Each entry now also gets a **topic document** carrying its topic line and its
  keywords. A caller who says a bare noun — *"transport service"*, *"water"*,
  *"amenities"* — is not asking anything question-shaped, and matching that
  against *"Is there a metro station inside the township?"* is exactly the miss
  that caused this. 39 entries now produce 328 documents.
- That topic document carries **no answer text**, and its shortness is the
  point. The first cut of it appended the answer and the key facts, and
  measured against the real vectors *"transport service"* and *"water"* still
  came back empty: embedding distance grows with how much of a document the
  query does not account for, so a two-word turn cannot stay close to sixty
  words of answer. The answer now travels in the document's metadata and
  `retriever.build_context()` puts it back on the way to the model — short to
  match, complete to answer.
- The distance floor (`RAG_MAX_DISTANCE=1.2`) was calibrated on full
  questions, and a bare noun is not one: with no sentence around it, a
  one-word turn sits further from every document. Measured on the real
  vectors, *"water"* landed at **1.328** — with the water supply entry as all
  three nearest documents, so the floor was rejecting the right answer for
  being tersely asked. Turns of one or two words now get a documented +0.25
  allowance (`SHORT_QUERY_MARGIN`), which is still well inside the 1.673 a
  greeting measured at. `tools/ask_kb.py` prints the distances behind any
  empty result, so this is a number next time rather than a guess.
- `rag/ingest.py` stamps the source file into the collection metadata and
  `rag/retriever.py` checks it at startup, so a stale index says so in the log
  instead of quietly answering out of the wrong knowledge base.
- Escalations are deliberately **not** embedded: their replies end in "may I
  have your name?", and the booking flow asks that itself, in its own order.

**The retriever's intent filter could return nothing at all.**
`detect_intent()` classified any query containing "call", "price", "contact" or
"reach" and then filtered the search on `{"intent": ...}` metadata. A collection
whose documents carry no such metadata matched **zero** documents — no context,
and the bot told callers it had no information it was in fact holding.
`retrieve()` now always repeats the search unfiltered when the narrowed one
comes back empty. The unconditional "take the nearest document anyway"
fallback that had been added alongside it is now limited to location and
contact questions, so *"Hello?"* (d=1.67 on the real call log) stops dragging a
random entry in for the model to answer from.

**And the prompt said to ask.** `llm.py` section 8 told the model to answer a
vague turn with a clarifying question, and a bare topic reads as vague. It now
says the opposite in as many words: a topic on its own **is** a question about
that topic, answer it; only a turn that names no topic at all gets the
clarifying line, and a named topic the knowledge block does not cover is the
not-found case, never the question.

### 2. The callback asked for the name first, and for the phone number

The 10:52 booking reached the sales team with the name **"I'm"** and no time at
all. Three things went wrong together and all three are fixed:

- **The order is now the caller's:** day and time first, then the name.
  Someone who has just asked for a call back is thinking about *when*. Asking
  the name first made them answer a question they had not asked yet, and it put
  the time behind a name, a read-back and a yes — so a caller who hung up, or
  drifted, lost it. The time is written to the row the moment it is understood,
  before anything else is asked. See `MediaStreamBridge._request_callback`,
  `_take_callback_time`, `_ask_for_name`, `_finish_callback`.
- **The bot no longer asks for a phone number.** It never had a canned line
  that did — the model improvised it. `llm.py` now forbids it outright, along
  with asking for a name or a time, because the backend owns that whole
  conversation and asking in parallel gets the caller asked everything twice.
- **`extract_name` no longer accepts `"I'm"`.** The plain words "i" and "am"
  were on the not-a-name list but the contraction was not, so a caller cut off
  mid-introduction had "I'm" written into the sales sheet as their name.
  Contractions, spoken clock words and weekday names are all excluded now.

A caller who answers both questions at once — *"David, tomorrow at six"* — is
not asked again; the name is read back and that is it. A caller who will not
give a name still gets their callback: the number that rang us is enough.

### 3. Housekeeping

`*.bak` files, `__pycache__`, a stray `-d` file, an Excel lock file and the
`Claude outputs/` folder are gone. `CALL_DIAGNOSIS_2026-09-04.md` was folded
into this file.

### After pulling this: rebuild the vector store

```bash
python -m rag.ingest        # reads data/knowledge_base.json
```

Nothing about the knowledge fix takes effect until this is run — the vectors
are a build artefact of the JSON, and the JSON is not read on a call. The
startup log tells you which file the current index came from.

Then check what a caller would actually get back, with the real embedder and
the real store:

```bash
python -m tools.ask_kb --check          # every phrasing off the call logs
python -m tools.ask_kb "transport service"
```

`--check` tests both directions and exits non-zero on either: every phrasing
off the call logs must come back with context, and every phrase in its noise
list — greetings, acknowledgements, gibberish — must come back with none. A
distance floor can only be wrong one way at a time, and both ways have already
cost a live call.

### Verified against the live vector store

`python -m tools.ask_kb --check`, on the rebuilt collection (328 documents):
all 13 caller phrasings return context — *"transport service"* and *"water"*
included, both of which returned nothing before this pass — and all 5 noise
phrases return none.

Still to confirm on a live call: that the spoken answer for a bare topic is the
answer and not a clarifying question (retrieval now supplies the facts; the
prompt change is what decides how they are used), and the callback order.

### Tests

`tests/test_bridge.py` 63/63 (the booking order, the volunteered name, the
`"I'm"` case, and the three real-call regressions), `tests/test_handoff.py`
80/80, `tests/test_stt.py` 31/31, `tests/test_tts.py` 14/14,
`tests/test_llm.py` 43/43, `tests/test_latency_noise.py` 28/28,
`tests/test_scheduling.py` 36/36, `python -m tools.check_kb` clean —
which now also asserts that every topic a caller has actually raised has an
entry with keywords behind it, transport included. `tests/test_rag.py` gained
a section that checks what `rag.ingest` builds out of the shipped KB.

---

## 2026-09-04 — the call that was silent, not slow

Call `CAb3dd76f6`, 11:08:43 → 11:12:04. The bot produced a correct answer for
every turn and the caller heard **three** of them. `Test3.aac` is 202.8 s long
and 152 s of it (75%) is below −45 dB.

**Root cause:** `bridge._speak()` opened with `self.stt.mute()`, which called
`_cancel_flush()`, which cancelled `_flush_task` — and on the debounce path
`_flush_task` *is the task currently running*, so the reply cancelled itself
before any audio went out.

Fixed in that pass, with regression tests that fail on the pre-fix code:

- the self-cancelling flush task (`stt.py`)
- per-sentence TTS pipelining (`tts.py`, `bridge.py`)
- Groq and Chroma moved off the event loop (`bridge.py`)
- Kokoro load and Groq model resolution moved to startup (`server.py`,
  `llm.py`, `tts.py`)

---

## Still open

**C. The recogniser can be confidently wrong.** "Is there any swapping mode?"
came back as "shopping mall" at confidence 0.97, and no confidence filter can
catch a confident error. Mitigated rather than fixed: the keyword boost is
wired in (`stt.py`), `vocab.repair()` now runs on every transcript, and
multilingual calls run on nova-3. `language=multi` has no Indian-English
variant, so this may get slightly worse before it gets better —
`MULTILINGUAL=false` reverts to the tuned en-IN stream.

**D. A false lead can still be recorded from an ASR error.** Largely defused:
`handoff: true` is now an OFFER that waits for the caller to accept, so a
mishearing can no longer book a callback on its own. What remains is the
`requirement` written into the sales sheet, which is still whatever was heard.

**E. The smalltalk guard is not turn-aware.** "Hello?" in the middle of a call
returns the cold-open greeting. It should branch on `self.history` being
non-empty and answer "yes, I'm here" instead.

**G. WhatsApp depends on Twilio configuration this repo cannot make.**
`TWILIO_WHATSAPP_FROM` is set to the project's own number now, but until that
number is registered as a WhatsApp sender on the Twilio account (or joined to
the sandbox), every message still falls back to SMS. The code says so once,
loudly, and stops retrying — it cannot fix it.

**J. An interrupted reply is still recorded in history as if it were spoken.**
The caller cut the bot off after one sentence of three; the model's next turn
believes all three were heard. Harmless today because the caller's own next
utterance dominates the context, but it will read oddly in a long call.

**K. The call log has no language column.** A call held in Tamil is
indistinguishable from one held in English in `data/call_log.xlsx`. Adding it
means changing the sheet's headers, which existing sheets do not have.
