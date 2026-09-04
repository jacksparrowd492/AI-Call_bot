import asyncio
import websockets
import os
from dotenv import load_dotenv

load_dotenv()

async def test():
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        print("❌ DEEPGRAM_API_KEY is missing from .env!")
        return

    url = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1"
    
    print("Attempting to connect to Deepgram...")
    try:
        async with websockets.connect(
            url,
            extra_headers={"Authorization": f"Token {key}"}
        ) as ws:
            print("✅ Successfully connected to Deepgram!")
            await ws.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

asyncio.run(test())