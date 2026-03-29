"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 5 — LLM Generation                                            ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Sends the retrieved context + conversation history to a Groq-hosted    ║
║  LLM and returns the model's answer as a plain string.                  ║
║                                                                          ║
║  Three responsibilities live here:                                       ║
║                                                                          ║
║  1. groq_chat()                                                          ║
║     The core LLM call. Takes a model ID and a full messages list        ║
║     (OpenAI-format: [{role, content}, …]) and returns the response.     ║
║     Handles common error cases (bad API key, network timeout) with      ║
║     human-readable messages rather than raw stack traces.               ║
║                                                                          ║
║  2. transcribe_audio()                                                   ║
║     Re-exported here for convenience — see pipeline_1_voice.py for      ║
║     the full explanation. Uses Groq's Whisper endpoint.                 ║
║                                                                          ║
║  3. describe_image()                                                     ║
║     Sends a base64-encoded image + question to LLaMA-4-Scout (Groq's    ║
║     multimodal model) and returns a natural-language description.        ║
║     The image is encoded inline as a data URI so no file upload is      ║
║     needed — Groq accepts base64 directly in the messages payload.      ║
║                                                                          ║
║  Groq vs. OpenAI API format:                                             ║
║    Groq uses the exact same chat completions schema as OpenAI, so the   ║
║    messages list format (system / user / assistant roles) is identical.  ║
║    Only the base URL and model names differ.                             ║
║                                                                          ║
║  Available models (selectable in the sidebar):                           ║
║    llama-3.3-70b-versatile  — best quality, slightly slower             ║
║    llama-3.1-8b-instant     — fastest, good for quick answers           ║
║    mixtral-8x7b-32768       — long context window (32k tokens)          ║
║    gemma2-9b-it             — Google's Gemma 2, instruction-tuned        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import base64

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODELS

# ── The tour-guide system prompt ──────────────────────────────────────────────
# The {context} and {question} placeholders are filled by the main app
# just before the API call; the template is here for documentation clarity.
SYSTEM_PROMPT_TEMPLATE = """\
You are a professional Egyptian tour guide with deep knowledge of history and culture.
Use ONLY the context below to answer. If the answer is not in the context, say
"I don't have that information in my tour notes."

Context:
{context}

Tourist question: {question}

Answer as an immersive, storytelling tour guide using historical facts:"""


def groq_chat(model: str, messages: list[dict]) -> str:
    """
    Send a conversation to the Groq chat completions endpoint.

    Parameters
    ----------
    model : str
        One of the model IDs in GROQ_MODELS (e.g. "llama-3.3-70b-versatile").
    messages : list[dict]
        Full conversation in OpenAI format:
            [{"role": "system", "content": "…"},
             {"role": "user",   "content": "…"},
             {"role": "assistant", "content": "…"}, …]
        The last element should be the current user turn with the
        context block prepended to the question.

    Returns
    -------
    str
        The model's reply. On authentication failure or network error,
        returns a human-readable error string starting with "𓂀".
    """
    try:
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model      = model,
            messages   = messages,
            max_tokens = 1024,
        )
        return response.choices[0].message.content
    except Exception as exc:
        err = str(exc)
        if "401" in err or "invalid_api_key" in err.lower():
            return "𓂀 Invalid Groq API key — update the GROQ_API_KEY environment variable."
        return f"𓂀 LLM error: {err}"


def describe_image(image_bytes: bytes, question: str) -> str:
    """
    Ask LLaMA-4-Scout (Groq multimodal) to describe or analyse an image.

    The image is encoded as a base64 data URI and sent inline in the
    messages payload — no separate file upload step required.
    The model returns a natural-language description or answer.

    Parameters
    ----------
    image_bytes : bytes
        Raw JPEG / PNG / WebP bytes of the image.
    question : str
        What the user asked about the image. If empty, falls back to a
        generic "Describe this image in detail." prompt.

    Returns
    -------
    str
        The model's visual description, or an error string starting "𓂀".
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)
        b64    = base64.b64encode(image_bytes).decode()
        prompt = question.strip() if question.strip() else "Describe this image in detail."

        response = client.chat.completions.create(
            model    = "meta-llama/llama-4-scout-17b-16e-instruct",
            messages = [{
                "role":    "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }],
            max_tokens = 1024,
        )
        return response.choices[0].message.content

    except Exception as exc:
        return f"𓂀 Vision error: {exc}"
