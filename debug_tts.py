"""
debug_tts.py  —  Run this SEPARATELY to diagnose TTS issues
Usage:  streamlit run debug_tts.py
"""
import streamlit as st, os, io, base64, re, traceback

st.set_page_config(page_title="TTS Debug", page_icon="🔊")
st.title("🔊 TTS Pipeline Debugger")

ELEVENLABS_API_KEY = st.text_input(
    "ElevenLabs API Key",
    value="sk_a280e81e4702d56c7f69dfea1c931775cfa9912cd1d066f3",
    type="password",
)
TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
TTS_MODEL    = "eleven_multilingual_v2"
TEST_TEXT    = st.text_area("Test text", value="Welcome to ancient Egypt. The pyramids stand before you.")

st.markdown("---")
st.subheader("Step-by-step checks")

# ── Step 1: elevenlabs ────────────────────────────────────────────────────────
with st.expander("① elevenlabs install", expanded=True):
    try:
        from elevenlabs.client import ElevenLabs as ELabs
        st.success("✅ elevenlabs imported OK")
    except Exception as e:
        st.error(f"❌ {e}")
        st.code("pip install elevenlabs")
        st.stop()

# ── Step 2: pydub ─────────────────────────────────────────────────────────────
with st.expander("② pydub install", expanded=True):
    try:
        from pydub import AudioSegment, effects as pydub_effects
        st.success("✅ pydub imported OK")
    except Exception as e:
        st.error(f"❌ {e}")
        st.code("pip install pydub")
        st.stop()

# ── Step 3: ffmpeg ────────────────────────────────────────────────────────────
with st.expander("③ ffmpeg available", expanded=True):
    try:
        import subprocess
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            st.success(f"✅ ffmpeg found: {r.stdout.splitlines()[0]}")
        else:
            raise RuntimeError(r.stderr)
    except FileNotFoundError:
        st.error("❌ ffmpeg NOT found in PATH")
        st.info("Windows: download from https://ffmpeg.org/download.html and add to PATH\n"
                "Linux: sudo apt-get install -y ffmpeg\n"
                "Conda: conda install -c conda-forge ffmpeg")
        st.stop()
    except Exception as e:
        st.warning(f"⚠ {e}")

# ── Step 4: noisereduce (optional) ────────────────────────────────────────────
with st.expander("④ noisereduce (optional)", expanded=True):
    try:
        import noisereduce as nr
        import numpy as np
        st.success("✅ noisereduce imported OK")
        NR_OK = True
    except Exception as e:
        st.warning(f"⚠ optional — {e}")
        NR_OK = False

# ── Step 5: ElevenLabs API call ───────────────────────────────────────────────
with st.expander("⑤ ElevenLabs API call", expanded=True):
    if st.button("▶ Call ElevenLabs now"):
        try:
            client    = ELabs(api_key=ELEVENLABS_API_KEY)
            audio_gen = client.text_to_speech.convert(
                text=TEST_TEXT[:500],
                voice_id=TTS_VOICE_ID,
                model_id=TTS_MODEL,
                output_format="mp3_44100_128",
            )
            raw_mp3 = b"".join(audio_gen)
            st.success(f"✅ Got {len(raw_mp3):,} bytes of MP3")
            st.session_state["_raw_mp3"] = raw_mp3
        except Exception as e:
            st.error(f"❌ ElevenLabs failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

# ── Step 6: pydub decode ─────────────────────────────────────────────────────
with st.expander("⑥ pydub decode + master", expanded=True):
    if "_raw_mp3" in st.session_state:
        try:
            seg = AudioSegment.from_file(io.BytesIO(st.session_state["_raw_mp3"]), format="mp3")
            st.success(f"✅ Decoded: {len(seg)} ms, {seg.frame_rate} Hz, {seg.channels}ch")
            seg = pydub_effects.normalize(seg)
            seg = pydub_effects.compress_dynamic_range(seg)
            seg = pydub_effects.normalize(seg)
            st.success("✅ Mastering chain OK")
            st.session_state["_voice_seg"] = seg
        except Exception as e:
            st.error(f"❌ pydub decode/master failed: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("Run step 5 first")

# ── Step 7: background music synthesis ───────────────────────────────────────
with st.expander("⑦ Background music synthesis", expanded=True):
    if "_voice_seg" in st.session_state:
        try:
            import numpy as np
            voice_len = len(st.session_state["_voice_seg"])
            total_ms  = voice_len + 6000
            sample_rate = 44100
            n_samples   = int(sample_rate * min(total_ms, 10000) / 1000)  # 10s test
            t = np.linspace(0, n_samples / sample_rate, n_samples, endpoint=False)
            drone = 0.4 * np.sin(2 * np.pi * 55 * t) + 0.15 * np.sin(2 * np.pi * 110 * t)
            pcm16 = (drone / np.max(np.abs(drone)) * 0.85 * 32767).astype("int16")
            import numpy as np
            stereo = np.column_stack([pcm16, pcm16])
            music  = AudioSegment(stereo.tobytes(), frame_rate=sample_rate, sample_width=2, channels=2)
            st.success(f"✅ Synth music: {len(music)} ms")
            st.session_state["_music_seg"] = music
        except Exception as e:
            st.error(f"❌ Music synth failed: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("Run step 5 first")

# ── Step 8: mix ───────────────────────────────────────────────────────────────
with st.expander("⑧ Mix voice + music", expanded=True):
    if "_voice_seg" in st.session_state and "_music_seg" in st.session_state:
        try:
            voice = st.session_state["_voice_seg"]
            music = st.session_state["_music_seg"] - 18
            total_ms = len(voice) + 6000
            # Loop music
            while len(music) < total_ms:
                music = music + music
            music = music[:total_ms]
            combined = music.overlay(voice, position=3000)
            final    = combined[:total_ms].fade_out(3000)
            buf = io.BytesIO()
            final.export(buf, format="mp3", bitrate="128k")
            mp3_bytes = buf.getvalue()
            st.success(f"✅ Final mix: {len(mp3_bytes):,} bytes")
            st.session_state["_final_b64"] = base64.b64encode(mp3_bytes).decode()
        except Exception as e:
            st.error(f"❌ Mix failed: {e}")
            st.code(traceback.format_exc())
    else:
        st.info("Run steps 5-7 first")

# ── Step 9: play it ───────────────────────────────────────────────────────────
with st.expander("⑨ Audio player output", expanded=True):
    if "_final_b64" in st.session_state:
        b64 = st.session_state["_final_b64"]
        st.success("✅ Playing final mixed audio:")
        # Use st.audio for guaranteed playback
        mp3_bytes = base64.b64decode(b64)
        st.audio(mp3_bytes, format="audio/mp3")
        # Also show the HTML player
        st.markdown(f"""
        <audio controls autoplay style="width:100%">
          <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>""", unsafe_allow_html=True)
        st.caption(f"Base64 length: {len(b64):,} chars")
    else:
        st.info("Complete steps 5-8 first")
