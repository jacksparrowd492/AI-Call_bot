# Jarvis Call Bot for Twilio

Drop-in replacement for the FreeSWITCH telephony layer of the Karthipuram
Jarvis voice assistant. Twilio handles the phone call and streams the audio
to this service over a WebSocket; this service runs the AI conversation
(Deepgram STT -> Groq LLM + RAG -> Kokoro TTS) and the sales features
(lead capture, WhatsApp brochure, call recording, human handoff).

```
Caller (+91) --> Twilio Voice --WebSocket (mu-law 8kHz)--> this service
                  |                                        |-- Deepgram STT
                  |                                        |-- Groq + ChromaDB RAG
                  |                                        |-- Kokoro TTS (local ONNX)
                  |<-------- mu-law audio back ------------|
                  |-- recording, leads, WhatsApp, handoff (event-triggered)
```

## Features

- Inbound call answering with a natural greeting (recording notice included)
- Real-time streaming conversation, first audio in about one second
- RAG answers grounded strictly in `data/knowledge_base.json`
- Lead capture into SQLite (name, requirement, intent score, transcript)
- WhatsApp brochure delivery during the call
- Dual-channel call recording with URL logging
- Warm human handoff to a sales agent via Twilio REST (replaces live TwiML)
- Echo suppression: the mic is closed while the bot speaks, so it never
  answers its own voice (barge-in is **not** implemented - see Known gaps)

## What is not in this repo

Three things are gitignored and must be supplied locally before the service
will run:

| Missing | Why | How to get it |
|---|---|---|
| `models/kokoro-v1_0.onnx` (325 MB), `models/voices.bin` (28 MB) | Over GitHub's 100 MB per-file limit | Download from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) into `models/` |
| `data/` (brochure PDF + knowledge-base JSON) | Client material, not for a public repo | Supply your own; `rag/ingest.py` expects `data/knowledge_base.json` |
| `.env` | Live Twilio / Deepgram / Groq credentials | `cp .env.example .env` and fill in your own keys |

`chroma_db/` and `leads.db` are also ignored - the first is rebuilt by
`python -m rag.ingest`, the second is created on the first call and holds real
callers' phone numbers.

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

- **No barge-in.** `stt.mute()` stops forwarding audio to Deepgram for the
  whole time the bot is speaking, so a caller cannot interrupt a long reply.
- Answers to the same question can differ between turns (`temperature=0.3` in
  `llm.py`).
- `vocab.py` (Deepgram keyword boosting and proper-noun repair) and `router.py`
  are written but imported by nothing.
- WhatsApp falls back to SMS unless a real WhatsApp sender is bound to the
  Twilio account (error 63007 with the shared sandbox number).

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
| TwiML / webhook | `server.py` | `/voice`, `/stream-result`, `/recording-complete` |
| Call conversation loop | `bridge.py` | turn-taking, echo suppression, reply streaming |
| Speech-to-text | `stt.py` | Deepgram, native mu-law, endpointing |
| LLM brain | `llm.py` | persona, RAG grounding, EXTRA lead metadata |
| Text-to-speech | `tts.py` | Kokoro ONNX 24 kHz -> 8 kHz mu-law, 20 ms frames, sentence-pipelined |
| Ask the KB directly | `tools/ask_kb.py` | `python -m tools.ask_kb --check` - real embedder, real store, the phrasings callers actually used |
| Knowledge base | `rag/` | ingest + retrieve (ChromaDB + MiniLM). **Editing `data/knowledge_base.json` changes nothing until `python -m rag.ingest` is re-run** - the vectors are a build artefact of that file |
| Leads | `tools/leads.py` | SQLite `leads.db`, one row per call |
| Sales call log | `tools/call_log.py` | Excel sheet, one row per call |
| Caller timezone | `scheduling.py` | phone number -> zone -> callback time in IST |
| WhatsApp | `tools/whatsapp.py` | sandbox sender by default |
| Handoff | `tools/handoff.py` | live-call TwiML update -> `<Dial>` |

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
| Handoff intent | `INTENT_HANDOFF_THRESHOLD` | Intent score that triggers dial-out |

## Production notes

- Run behind TLS (Caddy/nginx + Let's Encrypt) on a small VM in ap-south-1
  (Mumbai) for lowest round-trip latency with Indian callers.
- Protect or remove `GET /leads` before going public.
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
