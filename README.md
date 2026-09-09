# Jarvis Call Bot for Twilio

Drop-in replacement for the FreeSWITCH telephony layer of the Karthipuram
Jarvis voice assistant. Twilio handles the phone call and streams the audio
to this service over a WebSocket; this service runs the AI conversation
(Deepgram STT -> Groq LLM + RAG -> Kokoro TTS) and the sales features
(call log, callback scheduling, WhatsApp brochure, call recording).

The call is held in **English**. There is one STT stream, one voice and one
prompt: see "One language, on purpose" below.

```
Caller (+91) --> Twilio Voice --WebSocket (mu-law 8kHz)--> this service
                  |                                        |-- Deepgram STT (en-IN, nova-2)
                  |                                        |-- Groq + ChromaDB RAG
                  |                                        |-- Kokoro TTS (local ONNX)
                  |<-------- mu-law audio back ------------|
                  |-- recording, call log, WhatsApp, callback (event-triggered)
```

## Features

- Inbound call answering with a natural greeting (recording notice included)
- Real-time streaming conversation, first audio in about one second
- RAG answers grounded strictly in `data/knowledge_base.json`
- One row per call in `data/call_log.xlsx` (name, requirement, intent score,
  talk time, callback time in both zones)
- Callback scheduling in the caller's own timezone, stored in IST
- WhatsApp brochure delivery during the call
- Dual-channel call recording with URL logging
- Echo suppression, and **barge-in**: the caller can cut into a long answer.
  The detector measures Twilio's echo of the bot's own audio per call and
  requires an interruption to be loud relative to it (`BARGE_IN=false` restores
  the old behaviour, where the mic stayed shut for the whole reply)

## What is not in this repo

Three things are gitignored and must be supplied locally before the service
will run:

| Missing | Why | How to get it |
|---|---|---|
| `models/kokoro-v1_0.onnx` (325 MB), `models/voices.bin` (28 MB) | Over GitHub's 100 MB per-file limit | Download from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) into `models/` |
| `data/` (brochure PDF + knowledge-base JSON) | Client material, not for a public repo | Supply your own; `rag/ingest.py` expects `data/knowledge_base.json` |
| `.env` | Live Twilio / Deepgram / Groq credentials | `cp .env.example .env` and fill in your own keys |

`chroma_db/` and `leads.db` are also ignored - the first is rebuilt by
`python -m rag.ingest`, the second is created on the first booking and holds real
callers' numbers.

## One language, on purpose

The call is held in English and nothing in the code knows about a second
language. That is a deliberate reversal - Tamil, Hindi, French and Spanish were
implemented and then removed - and the reasons are worth keeping:

- `language=multi` costs the **en-IN** model. Deepgram's multilingual stream has
  no Indian-English variant, so the Indian names and place names this project
  turns on came back measurably worse. The tuned en-IN nova-2 stream is the
  whole reason `vocab.repair()` and the keyword boost work.
- Tamil has **no Kokoro voice**, so it needed a paid ElevenLabs call per
  sentence on the hot path - the one part of the pipeline that was otherwise
  free and local.
- A caller answered in French who is then rung back by a salesperson who only
  speaks Tamil is worse served than one kept in English. The sales team is the
  constraint, not the bot.

The removal is one commit; `git log` has it if a language is ever wanted back.

## Callback scheduling across timezones

Most callers are NRIs, so a requested time is read in **their** zone and stored
in IST for whoever has to make the call.

There are three ways a callback gets booked, and all three end in the same
place:

| The caller says | What happens |
|---|---|
| "I want to speak to someone" (`wants_human`) | Books, then *"Which day and what time would suit you?"* |
| "Can we schedule a call?" (`wants_schedule`) | Says yes, then *"Which day and what time would suit you?"* |
| Nothing - the bot has no answer and offers one | **Asks first.** Nothing is booked until the caller says yes |

The third is the only one that needs consent, because it is the only one where
the caller has not asked. `wants_schedule` deliberately ignores the
knowledge-base questions that look like it: *"how can I book a plot"*, *"what
is the booking amount"*, *"what is the payment schedule"*.

**The order of the questions is the caller's, not ours: day and time first,
then the name.** Someone who has just asked for a call back is thinking about
*when*; asking their name first makes them answer a question they have not
asked yet, and it pushed the time to the end of the booking where it kept
getting lost. The caller's phone number is never asked for - the call itself
supplies it.

```
Caller: Can you schedule a call?
Jarvis: Yes, of course - I can arrange that for you.
        Which day and what time would suit you?
Caller: Tomorrow evening.
Jarvis: Sure. What time exactly - say, ten in the morning, or six in the evening?
Caller: Six.
        -> the time is written to the row HERE, before anything else is asked
Jarvis: Thank you. And may I have your name, please?
Caller: Karthi.
Jarvis: Thank you. I have that as Karthi. Is that right?
Caller: Yes.
Jarvis: Perfect. Our specialist will call you then, and I'll send you a
        confirmation on WhatsApp and SMS right away.
```

A caller who answers both at once - *"David, tomorrow at six"* - is not asked
again; the name is just read back. A caller who will not give a name still gets
their callback, because the number that rang us is enough to act on.

1. The caller's timezone comes from the number that rang in, via
   `phonenumbers` - which resolves +1 by area code (six US zones) rather than
   guessing. Mobile numbers are not geographic, so for countries that span
   several zones the commercial centre is used and the row is flagged.
2. `"tomorrow evening"` names a day, not a time. Rather than inventing 6pm and
   converting that, the bot asks once - `"What time exactly?"` - and carries
   the day into the answer.
3. Both times are kept: what the caller said, their local time, and the IST
   equivalent. The sales team's SMS reads
   `Sat 5 Sep, 7:30 PM IST (6:00 PM Asia/Dubai)`.

`8 pm tomorrow` from New Jersey is `5:30 AM IST` the following day - which is
the whole point of doing this rather than writing down "8 pm".

## The sales call log

Every call appends a row to `data/call_log.xlsx` (`CALL_LOG_PATH`): date, start
and end, **talk time in minutes**, caller name and number, country and
timezone, what they asked about, intent score, and - when a callback was
accepted - the requested time in both zones plus whether the caller and the
sales team were actually notified.

`data/` is gitignored, and this file holds callers' names and numbers: it must
not be committed.

If the sheet is open in Excel when a call ends, Windows locks it. The row is
parked in `call_log.xlsx.pending.csv` and merged back on the next successful
write, so no call is lost because someone was looking at the spreadsheet.

Open defects and the change history live in `CHANGELOG.md`.

## Known gaps

- The recogniser can be **confidently wrong** ("swapping mode" -> "shopping
  mall" at 0.97), which no confidence filter can catch. Mitigated by the keyword
  boost and `vocab.repair()`, not solved.
- The `requirement` written to the sales sheet is still whatever was heard, so
  an ASR error can reach the sheet (it can no longer book a callback on its own -
  that needs the caller to accept).
- The smalltalk guard is not turn-aware: "Hello?" mid-call returns the cold-open
  greeting instead of "yes, I'm here".
- An interrupted reply is recorded in history as if it were spoken in full.
- WhatsApp falls back to SMS unless a real WhatsApp sender is bound to the
  Twilio account (error 63007 with the shared sandbox number).
- Four checks in `tests/test_bridge.py` fail (59/63): two stale expectations
  left over from queueing mid-reply turns and from the timezone note's wording,
  and **two real ones** where a name is not extracted and the caller is asked
  again. See CHANGELOG.

## Quick start

```bash
# 1. Create the environment (Python 3.10-3.12)
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env               # then edit with your keys

# 3. Build the knowledge base (re-run after ANY edit to the KB JSON)
python -m rag.ingest             # reads data/knowledge_base.json
python -m tools.ask_kb --check   # confirm real callers' phrasings match

# 4. Expose the service publicly (development)
ngrok http 8000                    # copy the https URL into PUBLIC_BASE_URL, restart

# 5. Run
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Twilio console setup

1. Buy (or use a trial) voice number.
2. Number -> Voice Configuration -> "A call comes in" -> Webhook:
   `https://YOUR-HOST/voice` (HTTP POST).
3. India +91 numbers additionally require a Regulatory Bundle (business KYC):
   Console -> Regulatory Compliance -> India. Approval usually takes 1-3 days.

## Where each feature lives

| Feature | File | Notes |
|---|---|---|
| TwiML / webhook | `server.py` | `POST /voice`, `GET /voice`, `/media` (WebSocket), `/brochure.pdf` |
| Call conversation loop | `bridge.py` | turn-taking, echo suppression, barge-in, reply streaming, the booking flow |
| Speech-to-text | `stt.py` | Deepgram en-IN nova-2, native mu-law, endpointing, keyword boost |
| Proper-noun repair | `vocab.py` | the boost list, plus `repair()` on every transcript ("Cartigram" -> "Karthipuram") |
| LLM brain | `llm.py` | persona, RAG grounding, EXTRA lead metadata |
| Text-to-speech | `tts.py` | Kokoro ONNX 24 kHz -> 8 kHz mu-law, 20 ms frames, sentence-pipelined |
| Echo gate and barge-in | `audio_utils.py` | `NoiseGate`, `BargeInDetector` |
| Callback intent | `handoff_intent.py` | did they ask for a person, accept an offer, give a name or a time |
| Ask the KB directly | `tools/ask_kb.py` | `python -m tools.ask_kb --check` - real embedder, real store, the phrasings callers actually used |
| Knowledge base | `rag/` | ingest + retrieve (ChromaDB + MiniLM). **Editing `data/knowledge_base.json` changes nothing until `python -m rag.ingest` is re-run** - the vectors are a build artefact of that file |
| Rebuild the KB JSON | `build_kb.py` | assembles `data/knowledge_base.json` from `kb_part1/2` + `kb_escalation` |
| Sales call log | `tools/call_log.py` | Excel sheet, one row per call |
| Callback records | `tools/callbacks.py` | one `callbacks` row per booking, in `leads.db` (`LEAD_STORE_PATH`) |
| Caller timezone | `scheduling.py` | phone number -> zone -> callback time in IST |
| WhatsApp | `tools/whatsapp.py` | sandbox sender by default |
| Live dependency check | `tools/smoke_test.py` | every real vendor, no phone call |

## Testing the WhatsApp feature

1. Console -> Messaging -> Try it out -> Send a WhatsApp message.
2. From the test phone, send the sandbox join code to the sandbox number.
3. During a call say: "Please send the brochure on WhatsApp" - the bot sets
   `whatsapp_wanted=true` and the service delivers `BROCHURE_URL`.

For production, register a WhatsApp business sender and use an approved
template for business-initiated messages.

## Latency knobs

| Knob | Env var | Effect |
|---|---|---|
| End-of-turn silence | `DEEPGRAM_ENDPOINTING_MS` | Lower = snappier, but may cut pauses |
| Reply length | `MAX_REPLY_SENTENCES` | Cap sentences per turn |
| Handoff intent | `INTENT_HANDOFF_THRESHOLD` | Intent score that offers a callback |
| Barge-in | `BARGE_IN` | `false` keeps the mic shut for the whole reply |

## Production notes

- Run behind TLS (Caddy/nginx + Let's Encrypt) on a small VM in ap-south-1
  (Mumbai) for lowest round-trip latency with Indian callers.
- `data/call_log.xlsx` holds callers' names and numbers. It is gitignored;
  keep it that way, and treat the file as personal data.
- Set `ANNOUNCE_RECORDING=false` only where the law allows unannounced
  recording; in India, announce it.
- Pin your vendor keys in a secrets manager; never commit `.env`.

## Cost snapshot (list prices, verify current rates)

| Item | Approx. |
|---|---|
| Twilio India inbound voice | ~$0.055 / min |
| Deepgram nova-2 | ~$0.0059 / min |
| Kokoro TTS | $0 - runs locally on CPU |
| Groq llama-3.1-8b | free tier at pilot volume |
| Small VM hosting | ~$10-25 / month |

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Twilio connected" then silence | PUBLIC_BASE_URL mismatch | Restart ngrok -> update .env -> restart uvicorn |
| No transcripts in logs | Deepgram key/model | Check `DEEPGRAM_API_KEY`, model name |
| Bot answers but no audio | Kokoro model files missing | Check `models/kokoro-v1_0.onnx` and `models/voices.bin` exist |
| Handoff does nothing | AGENT_DIAL_NUMBER unset | Set it, restart |
| RAG answers generic | KB not ingested | `python -m rag.ingest` |
| Bot asks "which details would you like?" instead of answering | The vector store was built from an older KB file - the startup log says so | `python -m rag.ingest`, then restart |
