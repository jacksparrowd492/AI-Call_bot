"""Show which Groq models this API key can actually use.

    python tools/check_groq.py

Run this instead of guessing a model id. A wrong GROQ_MODEL does not fail
loudly at startup - it 404s on the first caller turn, and every reply for the
rest of the call becomes "Sorry, one moment - let me connect you to my team."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings                      # noqa: E402
from llm import _NOT_CHAT, _sort_key             # noqa: E402


def main():
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env")
        return 1

    from groq import Groq
    try:
        ids = sorted(m.id for m in Groq(api_key=settings.groq_api_key).models.list().data)
    except Exception as e:
        print(f"Could not reach Groq: {e}")
        return 1

    chat = [m for m in ids if not any(x in m.lower() for x in _NOT_CHAT)]
    chat.sort(key=_sort_key)

    print(f"\n{len(ids)} models on this key ({len(chat)} usable for chat):\n")
    for m in ids:
        tag = "  chat" if m in chat else "  ----"
        print(f"{tag}  {m}")

    print(f"\nGROQ_MODEL          = {settings.groq_model}"
          f"   {'OK' if settings.groq_model in ids else '** NOT AVAILABLE **'}")
    print(f"GROQ_FALLBACK_MODEL = {settings.groq_fallback_model}"
          f"   {'OK' if settings.groq_fallback_model in ids else '** NOT AVAILABLE **'}")

    if settings.groq_model not in ids:
        print(f"\nSet this in .env:\n    GROQ_MODEL={chat[0]}" if chat else
              "\nThis key has no chat models at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
