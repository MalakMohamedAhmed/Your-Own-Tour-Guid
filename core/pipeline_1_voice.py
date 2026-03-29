"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 1 — Voice Transcription                                       ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Converts raw audio bytes (WAV, MP3, WebM, etc.) into plain English     ║
║  text using Groq's hosted Whisper-large-v3 model.                       ║
║                                                                          ║
║  Flow:                                                                   ║
║    audio bytes  →  temp file on disk  →  Groq Whisper API               ║
║               →  transcript string  →  returned to main app             ║
║                                                                          ║
║  Why a temp file?                                                        ║
║    Groq's API expects a file-like object with a real filename so it can ║
║    detect the audio format from the extension. We write bytes to a      ║
║    named temp file, open it, stream it, then delete it.                 ║
║                                                                          ║
║  Output contract:                                                        ║
║    Always returns a plain string. On failure, returns an error string   ║
║    starting with "[Transcription failed:" so callers can detect it.     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import tempfile
from groq import Groq

from .config import GROQ_API_KEY


def transcribe_audio(audio_bytes: bytes, ext: str = "webm") -> str:
    """
    Send raw audio bytes to Groq Whisper and return the transcript.

    Parameters
    ----------
    audio_bytes : bytes
        The raw audio data recorded in the browser or uploaded by the user.
    ext : str
        File extension that identifies the audio format ("webm", "wav",
        "mp3", "m4a", "ogg", "flac"). Defaults to "webm" because that is
        what Streamlit's st.audio_input widget produces in most browsers.

    Returns
    -------
    str
        The transcribed text, stripped of leading/trailing whitespace.
        On any error, returns a string like "[Transcription failed: ...]".
    """
    temp_path = None
    try:
        client = Groq(api_key=GROQ_API_KEY)

        # Write bytes to a temporary file so Groq's SDK can stream it
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        with open(temp_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(f"audio.{ext}", audio_file, f"audio/{ext}"),
            )

        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)          # clean up the temp file
        return response.text.strip()

    except Exception as exc:
        # Clean up temp file even on failure
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        return f"[Transcription failed: {exc}]"
