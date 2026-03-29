import os

# 𓂀 KEMET RAG — Centralized Configuration
# This file holds all your API keys and shared settings in one place.

# ── API KEYS ──────────────────────────────────────────────────────────────────
# Locally, you can set these as environment variables or in a .env file.
# When running on Streamlit Cloud, add them to the "Secrets" section.
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "your_groq_api_key_here")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "your_elevenlabs_api_key_here")

# ── SHARED SETTINGS ───────────────────────────────────────────────────────────
# ElevenLabs specific settings
# TTS_VOICE_ID  = "JBFqnCBsd6RMkjVDRZzb"   # Derived from debug_tts.py
TTS_VOICE_ID  = os.environ.get("TTS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
TTS_MODEL     = "eleven_multilingual_v2"

# Groq specific models
GROQ_MODELS = {
    "LLaMA 3.3 · 70B":   "llama-3.3-70b-versatile",
    "LLaMA 3.1 · 8B ⚡": "llama-3.1-8b-instant",
    "Mixtral · 8×7B":    "mixtral-8x7b-32768",
    "Gemma 2 · 9B":      "gemma2-9b-it",
}
