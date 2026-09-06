# Changelog and open defects

Two markdown files live in this repo and no more: `README.md` says how the
service works, this one says what changed and what is still wrong. Dated
diagnosis notes get folded in here rather than kept as separate files.

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

**A. Same question, two different answers.** Two identical questions with
identical retrieval (`best d=0.777`) returned different subsets of the same KB
entry. `temperature=0.3` in `llm.py` plus "choose the single best one" means a
caller who repeats a question gets different facts. Use `temperature=0` for
grounded turns.

**B. `vocab.py` and `router.py` are dead code.** `deepgram_keywords()` and
`repair()` are written, documented as the highest-leverage change available,
and imported by nothing; `stt.py`'s Deepgram URL has no `keywords=` and no
`language=`. `router.py` (351 lines) is orphaned too, and its test
(`tools/test_logic.py`) no longer runs — it reads `settings.route_strong_threshold`,
which `config.py` has not had for some time. Either wire the router in or
delete all three.

**C. `nova-2` mishears domain words.** "Is there any swapping mode?" at
confidence 0.97 is "shopping mall". Confidence filtering cannot catch a
confident error. Fixes: the keyword boost in B, `language=en-IN`, and
`nova-3` / `nova-2-phonecall`.

**D. A false lead can be recorded from an ASR error.** That mishearing produced
`handoff: true, intent_score: 7` and a callback request for a question the
caller never asked. Confirm before escalating on a turn whose RAG best-distance
is above threshold.

**E. The smalltalk guard is not turn-aware.** "Hello?" mid-call returns the
cold-open line. It should branch on `self.history` being non-empty.

**F. `phonemizer: words count mismatch`** on 4 of 7 syntheses — espeak-ng
failing phoneme alignment, a real risk of clipped Kokoro output. Normalise text
before TTS (expand `D-Mart`, hyphens, abbreviations).

**G. WhatsApp is not configured (Twilio 63007).** `TWILIO_WHATSAPP_FROM` is the
shared sandbox number with no channel bound to this account, so every brochure
silently downgrades to SMS.

**H. `_reply_lock` drops turns silently.** An utterance arriving while the
previous one is still being answered is discarded. With a multi-second reply, a
caller who repeats themselves loses the retry.

**I. No barge-in.** `stt.mute()` stops forwarding audio for the whole time the
bot speaks, so a caller cannot interrupt a long reply.
