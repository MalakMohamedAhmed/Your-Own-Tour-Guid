"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DIAGNOSTIC — test_voice.py                                             ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  A quick CLI script to verify Groq Whisper API connectivity and         ║
║  authentication using a dummy silent WAV buffer.                        ║
║                                                                          ║
║  Usage:                                                                  ║
║    python test_voice.py                                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
import os
import tempfile
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")

def test_transcription():
    client = Groq(api_key=GROQ_API_KEY)
    # Create a dummy tiny wav file (8-bit mono silence)
    # Actually, a valid small WAV file
    dummy_wav = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00D\xac\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(dummy_wav)
            temp_path = tmp.name
        
        print(f"Testing with key: {GROQ_API_KEY[:10]}...")
        with open(temp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=("test.wav", audio_file),
            )
        print("Success!")
        print(f"Transcript: {response.text}")
        os.unlink(temp_path)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_transcription()
