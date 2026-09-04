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
    whatsapp_from: str = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")            # https://your-host

    # --- Vendors ---
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = os.getenv("DEEPGRAM_MODEL", "nova-2")
    # endpointing = "this PHRASE ended". utterance_end_ms = "the TURN ended".
    # Treating the first as the second is what made the bot answer callers
    # mid-sentence, so the turn is only closed by the second (or by the
    # debounce below, whichever comes first).
    deepgram_endpointing_ms: str = os.getenv("DEEPGRAM_ENDPOINTING_MS", "900")
    deepgram_utterance_end_ms: str = os.getenv("UTTERANCE_END_MS", "1400")
    turn_debounce_ms: int = int(os.getenv("TURN_DEBOUNCE_MS", "700"))
    stt_min_confidence: float = float(os.getenv("STT_MIN_CONFIDENCE", "0.55"))
    # How long the mic stays shut AFTER Twilio confirms playback finished.
    # Twilio's jitter buffer is still draining the bot's own audio back at us.
    echo_tail_ms: int = int(os.getenv("ECHO_TAIL_MS", "700"))
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
    llm_reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "low")
    # qwen3 takes none/default rather than low/medium/high. "none" turns
    # thinking off outright, which is what a phone call wants - the caller
    # never hears reasoning, but it costs tokens, latency and rate limit.
    qwen_reasoning_effort: str = os.getenv("QWEN_REASONING_EFFORT", "none")

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

    def __post_init__(self):
        placeholder = (not self.brochure_url
                       or "your-host" in self.brochure_url
                       or "example" in self.brochure_url)
        if placeholder and self.public_base_url:
            self.brochure_url = self.public_base_url.rstrip("/") + "/brochure.pdf"

    def missing(self):
        """Return list of required keys that are not set."""
        required = [
            "twilio_account_sid", "twilio_auth_token", "twilio_phone_number",
            "deepgram_api_key", "groq_api_key",
        ]
        return [k for k in required if not getattr(self, k)]


settings = Settings()
