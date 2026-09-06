"""Central configuration - all settings come from .env"""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    # override=True is deliberate. Without it a stale OS environment variable
    # silently beats the .env file - which is exactly what happened with
    # GROQ_MODEL: .env said llama-3.3-70b-versatile, the process used
    # qwen-3.8-27B from the shell, and every call 404'd. This file is the
    # single source of truth, as the docstring above says.
    load_dotenv(override=True)
except ImportError:
    pass


@dataclass
class Settings:
    # --- Twilio ---
    twilio_account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")   # your +91 number
    # Accepts any of "whatsapp:+919418646464", "+919418646464" or
    # "9418646464" - __post_init__ normalises it. A number typed without the
    # prefix used to be sent to Twilio verbatim and rejected as a From address
    # that is not a WhatsApp channel, which looks exactly like a channel that
    # was never registered.
    whatsapp_from: str = os.getenv("TWILIO_WHATSAPP_FROM", "")
    # Used only to complete a local number typed without one.
    default_country_code: str = os.getenv("DEFAULT_COUNTRY_CODE", "+91")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")            # https://your-host

    # --- Vendors ---
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-2")
    # Indian English. "Karthi" came back as "Carty" at confidence 1.00 on the
    # generic model - the recogniser was certain and wrong. Set DEEPGRAM_LANG
    # to "en" to fall back if a model ever rejects this.
    deepgram_language: str = os.getenv("DEEPGRAM_LANG", "en-IN")
    # Boost the project's proper nouns BEFORE a transcript exists. vocab.py has
    # carried this list all along; nothing imported it until now.
    deepgram_boost: bool = os.getenv("DEEPGRAM_BOOST", "true").lower() == "true"
    # Anything else worth boosting for THIS deployment - names the sales team
    # hears often, a road, a competitor. Comma separated, boosted hard, and
    # editable without touching vocab.py:
    #   DEEPGRAM_EXTRA_KEYWORDS=Karthi,Bharath,Sathyamangalam
    deepgram_extra_keywords: str = os.getenv("DEEPGRAM_EXTRA_KEYWORDS", "")
    # endpointing = "this PHRASE ended". utterance_end_ms = "the TURN ended".
    # Treating the first as the second is what made the bot answer callers
    # mid-sentence, so the turn is only closed by the second (or by the
    # debounce below, whichever comes first).
    #
    # LATENCY: these three numbers ARE the "why is it so slow to answer?"
    # budget. The turn closes after min(endpointing + debounce, utterance_end)
    # of silence, and nothing else can start until it does - so the old
    # 900/700/1400 meant every single answer began 1.4s after the caller
    # stopped talking. 550/400/1000 closes the turn in 0.95s, which is still
    # longer than a natural conversational pause (~0.6s) so a caller who
    # stops to think is not cut off. 1000 is Deepgram's floor for
    # utterance_end_ms; do not go below it, it is rejected.
    deepgram_endpointing_ms: str = os.getenv("DEEPGRAM_ENDPOINTING_MS", "550")
    deepgram_utterance_end_ms: str = os.getenv("UTTERANCE_END_MS", "1000")
    turn_debounce_ms: int = int(os.getenv("TURN_DEBOUNCE_MS", "400"))
    stt_min_confidence: float = float(os.getenv("STT_MIN_CONFIDENCE", "0.55"))
    # NOISE. Two extra confidence gates, both aimed at the caller's background
    # rather than at the caller:
    #  - a FRAGMENT below this never joins the turn at all. Without it a
    #    0.2-confidence scrap of background TV was concatenated onto real
    #    speech, and max(confidence) then hid it from the check at flush time.
    #  - a turn that is ONE word needs more confidence than a sentence does,
    #    because that is the shape almost every noise transcript takes
    #    ("Yeah.", "Right.", "Okay?" out of a passing conversation). Real
    #    one-word answers come back at 0.9+; noise rarely does.
    stt_noise_confidence: float = float(os.getenv("STT_NOISE_CONFIDENCE", "0.40"))
    stt_short_confidence: float = float(os.getenv("STT_SHORT_CONFIDENCE", "0.75"))

    # --- Noise gate (see audio_utils.NoiseGate) ---
    # Applied to the caller's audio BEFORE Deepgram sees it. Steady background
    # noise - a fan, road traffic, an office, line hiss - is what makes the
    # recogniser emit phantom turns, and no confidence threshold can undo a
    # transcript that was never speech. Frames that sit at the noise floor are
    # attenuated (not deleted: Deepgram needs continuous audio to detect the
    # end of a turn), so the floor drops far below the model's speech
    # threshold while the caller's own voice passes through untouched.
    noise_gate: bool = os.getenv("NOISE_GATE", "true").lower() == "true"
    # Open the gate when the frame is this many times louder than the measured
    # noise floor. Lower = more sensitive (more noise gets through).
    noise_gate_ratio: float = float(os.getenv("NOISE_GATE_RATIO", "2.5"))
    # ...but never open below this absolute RMS, whatever the floor says.
    # Telephone speech runs 1500-8000 RMS; background noise 50-500.
    noise_gate_floor_rms: int = int(os.getenv("NOISE_GATE_FLOOR_RMS", "200"))
    # How long the gate stays open after the last loud frame. This has to
    # comfortably outlast the gaps BETWEEN words or the gate chops speech up.
    noise_gate_hangover_ms: int = int(os.getenv("NOISE_GATE_HANGOVER_MS", "320"))
    # Gain applied while the gate is shut. 0.12 is about -18 dB: quiet enough
    # to stop the recogniser, loud enough that a soft-spoken caller who slips
    # under the threshold is still audible instead of cut off dead.
    noise_gate_attenuation: float = float(os.getenv("NOISE_GATE_ATTENUATION", "0.12"))
    # How long the mic stays shut AFTER Twilio confirms playback finished.
    # Twilio's jitter buffer is still draining the bot's own audio back at us.
    # Every millisecond here is a millisecond the caller can talk and not be
    # heard, so it is trimmed to 550 - the mark has already confirmed playback,
    # this only covers the drain - and the noise gate now attenuates whatever
    # tail does leak through.
    echo_tail_ms: int = int(os.getenv("ECHO_TAIL_MS", "550"))
    # tts.py paces frames in real time, so when the send loop ends only the
    # ~400ms burst-ahead is still queued at Twilio. Waiting on Twilio's mark
    # should therefore be a formality - if it never arrives, reopen the mic
    # promptly instead of staying deaf for the length of the reply.
    mark_timeout_ms: int = int(os.getenv("MARK_TIMEOUT_MS", "1500"))

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    # NOT a reasoning model, deliberately. gpt-oss-120b spent most of its token
    # budget on hidden reasoning the caller never hears, which is what pushed
    # this account over Groq's tokens-per-minute limit; the SDK then slept 9
    # seconds per 429 and the caller heard dead air.
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    # Used automatically if the model above is retired or unavailable, so a
    # renamed model id degrades the answer instead of killing the call.
    groq_fallback_model: str = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
    # No reasoning tokens now, so the reply itself is all this has to cover.
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "400"))
    # How long one completion may take before we give up on it. The SDK no
    # longer retries (see GroqBrain.__init__), so this is the whole budget,
    # and it has to stay well inside what a caller will sit through in silence.
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "12"))
    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "low")
    # qwen3 takes none/default rather than low/medium/high. "none" turns
    # thinking off outright, which is what a phone call wants - the caller
    # never hears reasoning, but it costs tokens, latency and rate limit.
    qwen_reasoning_effort: str = os.getenv("QWEN_REASONING_EFFORT", "none")

    # Speak a touch faster than 1.0. Kokoro at 1.05 is still perfectly natural
    # but takes ~5% less wall-clock time to say the same reply, and on a phone
    # call the reply IS the latency for whatever the caller wants to say next.
    tts_speed: float = float(os.getenv("TTS_SPEED", "1.05"))
    # Speak each sentence as soon as the model has finished writing it,
    # instead of waiting for the whole completion (including the METADATA
    # block, which the caller never hears). Set false to fall back to the
    # older buffer-the-lot path.
    llm_stream_sentences: bool = os.getenv("LLM_STREAM_SENTENCES", "true").lower() == "true"

    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    eleven_voice_id: str = os.getenv("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    eleven_model_id: str = os.getenv("ELEVEN_MODEL_ID", "eleven_turbo_v2_5")

    # --- Features ---
    agent_dial_number: str = os.getenv("AGENT_DIAL_NUMBER", "")        # sales agent +91...
    # Left blank (or pointing at the placeholder host) this is derived from
    # PUBLIC_BASE_URL in __post_init__ below - server.py already serves the PDF
    # at /brochure.pdf, and hardcoding an ngrok URL here goes stale every
    # restart. A bad URL is Twilio error 21620 and silently downgrades every
    # WhatsApp brochure to SMS.
    brochure_url: str = os.getenv("BROCHURE_URL", "")
    announce_recording: bool = os.getenv("ANNOUNCE_RECORDING", "true").lower() == "true"
    intent_handoff_threshold: int = int(os.getenv("INTENT_HANDOFF_THRESHOLD", "85"))

    # --- RAG ---
    chroma_dir: str = os.getenv("CHROMA_DIR",
                                os.path.join(os.path.dirname(__file__), "chroma_db"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "real_estate")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    # Chroma L2 distance over normalised MiniLM vectors. Observed on real
    # calls: a real question matches at 0.8-1.45, while "Hello?" / "Hold on."
    # bottom out around 1.6-1.75 and used to drag in a random KB entry that
    # the model then tried to answer from. Anything past this is NOT a match.
    rag_max_distance: float = float(os.getenv("RAG_MAX_DISTANCE", "1.5"))

    # --- Embeddings (local ONNX MiniLM, via ChromaDB) ---
    # all-MiniLM-L6-v2 runs on CPU through onnxruntime: no API key, no cost
    # and no network call on the hot path. It is fixed at 384 dimensions, so
    # EMBEDDING_DIMENSIONS is ignored. Changing the model means rebuilding
    # the Chroma collection (`python -m rag.ingest`).
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")   # unused by RAG now
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimensions: int | None = (
        int(os.environ["EMBEDDING_DIMENSIONS"])
        if os.getenv("EMBEDDING_DIMENSIONS") else None
    )
    embedding_timeout_s: float = float(os.getenv("EMBEDDING_TIMEOUT_S", "8"))

    # --- Behavior tuning ---
    end_turn_silence_ms: int = int(os.getenv("END_TURN_SILENCE_MS", "300"))
    max_reply_sentences: int = int(os.getenv("MAX_REPLY_SENTENCES", "4"))
    lead_store_path: str = os.getenv("LEAD_STORE_PATH",
                                     os.path.join(os.path.dirname(__file__), "leads.db"))
    # One row per call, in a sheet the sales team can just open. Lives under
    # data/ because it holds callers' names and numbers, and data/ is
    # gitignored - this must never reach the public repo.
    call_log_path: str = os.getenv("CALL_LOG_PATH",
                                   os.path.join(os.path.dirname(__file__),
                                                "data", "call_log.xlsx"))
    # Where the office is. Every callback the sales team sees is in this zone,
    # whatever time the caller quoted from Dubai, London or New Jersey.
    office_timezone: str = os.getenv("OFFICE_TIMEZONE", "Asia/Kolkata")

    def __post_init__(self):
        self.whatsapp_from = self._as_whatsapp_sender(self.whatsapp_from)

        placeholder = (not self.brochure_url
                       or "your-host" in self.brochure_url
                       or "example" in self.brochure_url)
        if placeholder and self.public_base_url:
            self.brochure_url = self.public_base_url.rstrip("/") + "/brochure.pdf"

    def _as_whatsapp_sender(self, value: str) -> str:
        """"9418646464" -> "whatsapp:+919418646464". Left alone if it already
        carries the prefix and a country code."""
        raw = (value or "").strip()
        if not raw:
            return ""
        number = raw[len("whatsapp:"):] if raw.lower().startswith("whatsapp:") else raw
        number = "".join(ch for ch in number if ch.isdigit() or ch == "+").strip()
        if not number:
            return ""
        if not number.startswith("+"):
            cc = (self.default_country_code or "+91").strip()
            if not cc.startswith("+"):
                cc = "+" + cc
            # A number already carrying its country code without the plus -
            # "919418646464" - must not get a second one.
            if number.startswith(cc.lstrip("+")) and len(number) > 10:
                number = "+" + number
            else:
                number = cc + number
        return "whatsapp:" + number

    def missing(self):
        """Return list of required keys that are not set."""
        required = [
            "twilio_account_sid", "twilio_auth_token", "twilio_phone_number",
            "deepgram_api_key", "groq_api_key",
        ]
        return [k for k in required if not getattr(self, k)]


settings = Settings()
