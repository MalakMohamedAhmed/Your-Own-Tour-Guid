"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 6 — Text-to-Speech + Egyptian Music Mixing                    ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Converts the LLM's text answer into an audio file that combines a      ║
║  professional voice with a synthesised Egyptian ambient music bed.      ║
║                                                                          ║
║  Full audio pipeline (run_tts_pipeline):                                ║
║    1. Strip markdown symbols from the answer text (bold, headers, etc.) ║
║       because TTS engines read them aloud as literal characters.        ║
║    2. Send clean text to ElevenLabs → receive raw PCM audio.            ║
║       Format: signed int16, 24 kHz, mono (output_format="pcm_24000").  ║
║       Why PCM? Avoids needing an MP3 decoder at mix time.               ║
║    3. Synthesise (or load) an Egyptian music bed at the same sample rate.║
║    4. Mix voice over music: 3 s intro → overlay voice → 3 s fade-out.  ║
║    5. Encode the mixed float32 array to a standard WAV file using only  ║
║       Python's stdlib `wave` module — no ffmpeg or soundfile needed.    ║
║    6. Return WAV bytes → st.audio() plays them inline.                  ║
║                                                                          ║
║  Music bed synthesis (_synth_egyptian_music):                           ║
║    Four layered oscillators, all pure numpy:                             ║
║      • Deep 55 Hz drone + 2nd and 3rd harmonics (ancient lyre feel)     ║
║      • Oud partials at 82 / 165 / 247 Hz with 4.5 Hz vibrato           ║
║      • Tabla pulse: 180 Hz tone + exponential decay every 500 ms        ║
║      • C5 shimmer (523 Hz) with a slow 0.11 Hz amplitude swell          ║
║    The result is normalised to 82 % of full scale before mixing.        ║
║                                                                          ║
║  Music file fallback (_make_music_bed):                                  ║
║    If egyptian_crop.mp3 is present beside the script, it is loaded      ║
║    with librosa or pydub (tried in that order) and looped to length.    ║
║    If neither library is available, the numpy synth is used instead.    ║
║                                                                          ║
║  Caching:                                                                ║
║    WAV bytes are stored in st.session_state['tts_cache'] keyed by       ║
║    "{session_id}:{message_index}" so audio is generated once per       ║
║    message and survives Streamlit reruns without re-calling ElevenLabs. ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import io
import os
import re
import wave as _wave

import numpy as np
import streamlit as st

# ── Optional ElevenLabs ───────────────────────────────────────────────────────
TTS_AVAILABLE = False
try:
    from elevenlabs.client import ElevenLabs as ELabs
    TTS_AVAILABLE = True
except ImportError:
    pass

from .config import ELEVENLABS_API_KEY, TTS_VOICE_ID, TTS_MODEL
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSIC_FILE    = os.path.join(BASE_DIR, "assets", "egyptian_crop.mp3")

# Mix parameters
INTRO_SECONDS  = 3.0    # silence / music-only before voice starts
OUTRO_SECONDS  = 3.0    # fade-out duration at the end
MUSIC_DB       = -14    # music level under the voice in decibels
SAMPLE_RATE    = 24000  # must match ElevenLabs output_format


# ── Low-level audio helpers ───────────────────────────────────────────────────

def _synth_egyptian_music(n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Generate n_samples of Egyptian ambient music using pure numpy oscillators.

    Layers:
        Drone   — A1 (55 Hz) + 2nd harmonic (110 Hz) + 3rd (165 Hz).
                  Provides the deep, resonant foundation.
        Oud     — Three partials at 82 / 165 / 247 Hz with 4.5 Hz vibrato
                  (±0.3 % pitch deviation). Evokes the Arabic short-neck lute.
        Tabla   — 180 Hz sine burst with exponential decay, fired every 500 ms.
                  Mimics the bayan (bass drum head) stroke.
        Shimmer — C5 (523 Hz) with a very slow (0.11 Hz) amplitude swell.
                  Adds the high-frequency 'air' characteristic of ney flute.

    Returns float32 array normalised to [-0.82, 0.82].
    """
    t = np.linspace(0, n_samples / sr, n_samples, endpoint=False, dtype=np.float32)

    # Deep drone
    drone = (
        0.38 * np.sin(2 * np.pi * 55.0  * t) +
        0.14 * np.sin(2 * np.pi * 110.0 * t) +
        0.07 * np.sin(2 * np.pi * 165.0 * t)
    )

    # Oud with vibrato
    vib = 1.0 + 0.003 * np.sin(2 * np.pi * 4.5 * t)
    oud = (
        0.18 * np.sin(2 * np.pi * 82.4  * vib * t) +
        0.08 * np.sin(2 * np.pi * 164.8 * vib * t) +
        0.04 * np.sin(2 * np.pi * 247.0 * vib * t)
    )

    # Tabla pulses
    tabla = np.zeros(n_samples, dtype=np.float32)
    step  = int(sr * 0.5)    # fire every 500 ms
    plen  = int(sr * 0.07)   # 70 ms decay
    tt    = np.linspace(0, 0.07, plen, dtype=np.float32)
    pulse = 0.16 * np.sin(2 * np.pi * 180 * tt) * np.exp(-np.linspace(0, 8, plen, dtype=np.float32))
    for start in range(0, n_samples - plen, step):
        tabla[start : start + plen] += pulse

    # Shimmer
    swell   = (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t)).astype(np.float32)
    shimmer = 0.035 * swell * np.sin(2 * np.pi * 523.25 * t)

    mix  = drone + oud + tabla + shimmer
    peak = np.max(np.abs(mix))
    return (mix / peak * 0.82) if peak > 0 else mix


def _make_music_bed(total_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Produce a float32 array of exactly `total_samples` of Egyptian music.

    Tries three strategies in order:
      1. Load egyptian_crop.mp3 with librosa (best quality, correct resampling).
      2. Load it with pydub (no pip extras needed beyond pydub + ffmpeg).
      3. Synthesise with _synth_egyptian_music (zero dependencies, always works).

    In all cases the result is tiled / cropped to exactly total_samples.
    """
    if os.path.exists(MUSIC_FILE):
        try:
            import librosa
            y, _ = librosa.load(MUSIC_FILE, sr=sr, mono=True)
            reps = (total_samples // len(y)) + 2
            return np.tile(y, reps)[:total_samples].astype(np.float32)
        except Exception:
            pass
        try:
            from pydub import AudioSegment
            audio   = AudioSegment.from_file(MUSIC_FILE).set_frame_rate(sr).set_channels(1)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
            reps    = (total_samples // len(samples)) + 2
            return np.tile(samples, reps)[:total_samples].astype(np.float32)
        except Exception:
            pass

    # Fallback: pure-numpy synth
    tile = min(total_samples, int(30 * sr))
    bed  = _synth_egyptian_music(tile, sr)
    reps = (total_samples // tile) + 2
    return np.tile(bed, reps)[:total_samples]


def _pcm_to_wav(pcm_int16: np.ndarray, sr: int, channels: int = 1) -> bytes:
    """
    Encode a signed int16 numpy array into a standard WAV byte string.
    Uses only Python's stdlib `wave` module — no ffmpeg, soundfile, or scipy.

    WAV format details written:
        - PCM encoding (no compression)
        - 16-bit samples (2 bytes each)
        - Mono (1 channel), 24 000 Hz sample rate
    """
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)        # int16 = 2 bytes per sample
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def _mix_voice_with_music(voice_pcm_bytes: bytes, sr: int = SAMPLE_RATE) -> bytes:
    """
    Overlay ElevenLabs voice PCM onto an Egyptian music bed and return WAV bytes.

    Layout of the output audio:
        [0 s          ]  music only (INTRO_SECONDS)
        [3 s → end−3 s]  music + voice  (voice ducked above music at MUSIC_DB dB)
        [end−3 s → end]  music fades to silence (OUTRO_SECONDS)

    Steps:
        1. Interpret ElevenLabs raw bytes as int16, normalise to float32.
        2. Calculate total output length = intro + voice + outro.
        3. Fill a music buffer of that length using _make_music_bed.
        4. Scale music to MUSIC_DB dB below 0 dBFS.
        5. Add the voice samples starting at the intro offset.
        6. Apply a linear fade-out over the last OUTRO_SECONDS.
        7. Clip to [-1, 1], convert back to int16, encode as WAV.
    """
    voice = np.frombuffer(voice_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    intro_n = int(INTRO_SECONDS * sr)
    outro_n = int(OUTRO_SECONDS * sr)
    total_n = intro_n + len(voice) + outro_n

    music       = _make_music_bed(total_n, sr)
    music_gain  = 10 ** (MUSIC_DB / 20.0)
    music      *= music_gain

    # Overlay voice (additive mix — voice sits on top of music)
    end = intro_n + len(voice)
    music[intro_n : end] += voice

    # Linear fade-out on the outro
    fade = np.linspace(1.0, 0.0, outro_n, dtype=np.float32)
    music[total_n - outro_n :] *= fade

    out = np.clip(music, -1.0, 1.0)
    return _pcm_to_wav((out * 32767).astype(np.int16), sr)


# ── Public entry point ────────────────────────────────────────────────────────

def run_tts_pipeline(text: str) -> bytes | None:
    """
    Full TTS pipeline: clean text → ElevenLabs voice → mix with music → WAV.

    Parameters
    ----------
    text : str
        The LLM's answer, potentially containing markdown formatting.

    Returns
    -------
    bytes | None
        WAV bytes ready for st.audio(), or None if TTS failed.
        On failure, a diagnostic message is written to
        st.session_state['tts_err'] so the UI can display it.

    Fallback behaviour:
        If the music mixing step fails for any reason, the function retries
        with voice-only (no music) so the user always gets some audio.
    """
    st.session_state["tts_err"] = ""

    if not TTS_AVAILABLE:
        st.session_state["tts_err"] = "elevenlabs not installed — run: pip install elevenlabs"
        return None

    # Strip markdown symbols that would be read aloud literally
    clean = re.sub(r"[*_`#>~\[\]]+", "", text).strip()
    clean = re.sub(r"\s+", " ", clean)[:4000]   # ElevenLabs limit
    if not clean:
        return None

    try:
        client  = ELabs(api_key=ELEVENLABS_API_KEY)
        gen     = client.text_to_speech.convert(
            text          = clean,
            voice_id      = TTS_VOICE_ID,
            model_id      = TTS_MODEL,
            output_format = "pcm_24000",   # raw signed-int16 PCM, no MP3 decoder needed
        )
        raw_pcm = b"".join(gen)

        if not raw_pcm:
            st.session_state["tts_err"] = "ElevenLabs returned 0 bytes"
            return None

        return _mix_voice_with_music(raw_pcm, sr=SAMPLE_RATE)

    except Exception as exc:
        st.session_state["tts_err"] = str(exc)

        # Fallback: plain voice without music background
        try:
            client  = ELabs(api_key=ELEVENLABS_API_KEY)
            gen     = client.text_to_speech.convert(
                text          = clean,
                voice_id      = TTS_VOICE_ID,
                model_id      = TTS_MODEL,
                output_format = "pcm_24000",
            )
            raw = b"".join(gen)
            if raw:
                st.session_state["tts_err"] += " (playing voice only — music mix failed)"
                return _pcm_to_wav(np.frombuffer(raw, dtype=np.int16), sr=SAMPLE_RATE)
        except Exception:
            pass

        return None
