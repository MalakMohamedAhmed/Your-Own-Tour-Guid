"""
╔══════════════════════════════════════════════════════════════════════════╗
║  KEMET · RAG  —  Main Application                                       ║
║                                                                          ║
║  This file is the Streamlit entry point. It owns:                       ║
║    • Page configuration and all CSS styling                             ║
║    • Session state initialisation                                        ║
║    • Sidebar (pipeline status, library browser, session list)           ║
║    • Chat display loop                                                   ║
║    • Input bar (text + image upload + microphone)                       ║
║    • The "process send" block that calls all pipelines in order         ║
║                                                                          ║
║  Pipeline call order on each user message:                              ║
║    1. pipeline_1_voice     — transcribe audio if mic was used           ║
║    2. pipeline_7_classifier — classify + describe image if one attached ║
║    3. pipeline_2_nlp        — detect language, translate, clean text    ║
║    4. pipeline_3_indexing   — retrieve fingerprints / startup indexing  ║
║    5. pipeline_4_retrieval  — vector search for relevant chunks         ║
║    6. pipeline_5_llm        — build prompt and call Groq LLM            ║
║    7. pipeline_6_tts        — convert answer to speech + mix music      ║
║                                                                          ║
║  State keys used:                                                        ║
║    sessions          — {session_id: {title, messages, created}}         ║
║    active_id         — currently open session ID                        ║
║    show_chunks       — whether to show source citations in chat         ║
║    show_pipeline     — whether to show NLP trace per user message       ║
║    global_model      — currently selected Groq model ID                 ║
║    kb_docs           — list of indexed file metadata dicts              ║
║    kb_loaded         — bool, True after first startup scan              ║
║    pending_image     — bytes of attached image (cleared after send)     ║
║    pending_audio     — bytes of recorded audio (cleared after send)     ║
║    pending_audio_ext — file extension of audio ("webm", "wav", etc.)   ║
║    translation_cache — MD5 → translate result (session-level cache)    ║
║    tts_enabled       — bool toggle from sidebar                         ║
║    tts_cache         — {sid:idx → WAV bytes} for generated audio       ║
║    tts_err / tts_last_err / tts_flags — diagnostic strings for TTS     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import base64
import time
from datetime import datetime

import streamlit as st

# ── Import all pipeline modules ───────────────────────────────────────────────
from core.pipeline_1_voice     import transcribe_audio
from core.pipeline_2_nlp       import full_text_pipeline, TRANSLATION_AVAILABLE, NLTK_AVAILABLE
from core.pipeline_3_indexing  import (
    scan_and_index, chroma_count, chroma_clear,
    get_chroma_collection, PDF_BACKEND, CHROMA_AVAILABLE,
)
from core.pipeline_4_retrieval import retrieve, build_context
from core.pipeline_5_llm       import groq_chat, describe_image, GROQ_MODELS
from core.pipeline_6_tts       import run_tts_pipeline, TTS_AVAILABLE
from core.pipeline_7_classifier import (
    classify_image, CLASSIFIER_AVAILABLE, CLASSIFIER_CLASSES,
    CLASSIFIER_WEIGHTS as CLF_WEIGHTS,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Kemet · RAG",
    page_icon="𓂀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=Crimson+Pro:ital,wght@0,300;0,400;0,500;1,300;1,400&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --sand:#1a1408;--sand2:#221c0a;--sand3:#2c2310;--sand4:#352b14;
  --gold:#c8922a;--gold2:#e8b84b;
  --lapis:#1a3a5c;--lapis2:#2a5a8c;
  --teal:#3aaa8a;--red:#8b2020;--red2:#c23030;
  --text:#f0e6cc;--text2:#c8b890;--text3:#7a6840;
  --border:#3a2e14;--border2:#4a3c1c;
  --pdf:#c23030;--pdf2:#e05050;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{background:var(--sand)!important;}
.stApp{background:var(--sand)!important;font-family:'Crimson Pro',Georgia,serif!important;color:var(--text)!important;}
.stApp::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");opacity:0.6;}
#MainMenu,footer,.stDeployButton{display:none!important;}
header[data-testid="stHeader"]{background:transparent!important;}
.block-container{padding:0!important;max-width:100%!important;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:var(--sand2);}
::-webkit-scrollbar-thumb{background:var(--gold);border-radius:3px;}
section[data-testid="stSidebar"]{background:var(--sand2)!important;border-right:2px solid var(--border2)!important;}
section[data-testid="stSidebar"]>div{padding:0!important;background:var(--sand2)!important;}
.sb-logo{padding:1.5rem 1.25rem 0.5rem;border-bottom:1px solid var(--border2);margin-bottom:0.5rem;}
.sb-logo-glyph{font-size:1.8rem;color:var(--gold2);display:block;margin-bottom:0.3rem;text-shadow:0 0 20px rgba(200,146,42,0.5);}
.sb-logo-title{font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;letter-spacing:0.15em;color:var(--gold2);text-transform:uppercase;}
.sb-logo-title span{color:var(--teal);}
.sb-logo-sub{font-family:'Crimson Pro',serif;font-style:italic;font-size:0.75rem;color:var(--text3);margin-top:0.15rem;}
.hieroglyph-border{text-align:center;font-size:0.65rem;letter-spacing:0.3em;color:var(--gold);opacity:0.4;padding:0.2rem 0;overflow:hidden;white-space:nowrap;}
.sb-section{font-family:'Cinzel',serif;font-size:0.58rem;font-weight:600;letter-spacing:0.2em;text-transform:uppercase;color:var(--gold);padding:0.75rem 1.25rem 0.3rem;display:flex;align-items:center;gap:0.5rem;opacity:0.8;}
.sb-section::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--gold) 0%,transparent 100%);opacity:0.3;}
.file-pill{display:flex;align-items:center;gap:0.55rem;padding:0.5rem 1rem;margin:2px 0.75rem;border-radius:4px;font-family:'Crimson Pro',serif;font-size:0.85rem;color:var(--text2);background:var(--sand3);border:1px solid var(--border);border-left:3px solid var(--gold);}
.file-pill.pdf-pill{border-left-color:var(--pdf2);}
.fp-dot{width:6px;height:6px;border-radius:50%;background:var(--gold2);flex-shrink:0;}
.fp-dot.pdf{background:var(--pdf2);}
.fp-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.fp-size{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--text3);}
.fp-badge{font-family:'Cinzel',serif;font-size:0.5rem;letter-spacing:0.1em;padding:2px 5px;border-radius:2px;flex-shrink:0;font-weight:700;}
.fp-badge.pdf{background:rgba(194,48,48,0.2);color:var(--pdf2);border:1px solid rgba(194,48,48,0.3);}
.fp-badge.txt{background:rgba(200,146,42,0.15);color:var(--gold2);border:1px solid rgba(200,146,42,0.25);}
.stats-strip{display:flex;margin:0.5rem 0.75rem;background:var(--sand3);border:1px solid var(--border2);border-radius:4px;overflow:hidden;border-top:2px solid var(--gold);}
.stat-c{flex:1;text-align:center;padding:0.55rem 0;border-right:1px solid var(--border);}
.stat-c:last-child{border-right:none;}
.stat-n{font-family:'Cinzel',serif;font-size:1.1rem;font-weight:700;color:var(--gold2);display:block;line-height:1;}
.stat-l{font-family:'JetBrains Mono',monospace;font-size:0.52rem;color:var(--text3);text-transform:uppercase;letter-spacing:0.1em;display:block;margin-top:2px;}
.pipeline-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:3px;font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:600;letter-spacing:0.08em;margin:1px;}
.pb-on{background:rgba(58,170,138,0.2);color:var(--teal);border:1px solid rgba(58,170,138,0.4);}
.pb-off{background:rgba(122,104,64,0.2);color:var(--text3);border:1px solid var(--border);}
.pipeline-row{padding:0.4rem 0.75rem;display:flex;flex-wrap:wrap;gap:3px;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0.85rem 2rem;background:var(--sand2);border-bottom:2px solid var(--border2);position:relative;}
.topbar::after{content:'';position:absolute;bottom:-4px;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent 0%,var(--gold) 20%,var(--gold2) 50%,var(--gold) 80%,transparent 100%);opacity:0.35;}
.topbar-title{font-family:'Cinzel',serif;font-size:0.9rem;font-weight:600;letter-spacing:0.1em;color:var(--gold2);}
.topbar-badge{display:flex;align-items:center;gap:0.5rem;background:var(--sand3);border:1px solid var(--border2);border-top:1px solid var(--gold);border-radius:3px;padding:0.28rem 0.85rem;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--gold2);}
.topbar-eye{font-size:0.9rem;animation:glow-eye 3s ease-in-out infinite;}
@keyframes glow-eye{0%,100%{opacity:1;text-shadow:0 0 8px var(--gold2);}50%{opacity:0.5;text-shadow:none;}}
.step-card{background:var(--sand2);border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:0 4px 4px 0;padding:0.5rem 0.75rem;margin:0.3rem 0;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--text3);line-height:1.7;}
.step-label{color:var(--gold2);font-size:0.6rem;font-family:'Cinzel',serif;letter-spacing:0.1em;display:block;margin-bottom:2px;}
.step-val{color:var(--text2);}
.lang-badge{display:inline-block;background:rgba(42,90,140,0.3);color:#6ab0e0;border:1px solid rgba(42,90,140,0.5);border-radius:2px;padding:1px 6px;font-size:0.58rem;margin-right:4px;}
.translated-badge{display:inline-block;background:rgba(58,170,138,0.2);color:var(--teal);border:1px solid rgba(58,170,138,0.4);border-radius:2px;padding:1px 6px;font-size:0.58rem;}
div[data-testid="stChatMessage"]{background:transparent!important;border:none!important;padding:0.5rem 0!important;max-width:820px!important;margin:0 auto!important;}
div[data-testid="stChatMessage"] .stMarkdown p{font-family:'Crimson Pro',serif!important;font-size:1rem!important;line-height:1.8!important;color:var(--text2)!important;}
div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"]{background:linear-gradient(135deg,var(--red),var(--red2))!important;border-radius:4px!important;}
div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"]{background:linear-gradient(135deg,var(--lapis),var(--lapis2))!important;border-radius:4px!important;}
.src-box{background:var(--sand2);border:1px solid var(--border);border-left:3px solid var(--gold);border-radius:0 4px 4px 0;padding:0.7rem 0.9rem;margin:0.35rem 0;font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--text3);line-height:1.6;}
.src-box.pdf-src{border-left-color:var(--pdf2);}
.src-label{font-size:0.6rem;color:var(--gold2);margin-bottom:0.3rem;display:block;font-family:'Cinzel',serif;}
.src-label.pdf-lbl{color:var(--pdf2);}
.empty-wrap{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;gap:1.5rem;text-align:center;}
.empty-glyph{font-size:3.5rem;animation:float 4s ease-in-out infinite;}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}
.empty-title{font-family:'Cinzel',serif;font-size:1.8rem;font-weight:900;letter-spacing:0.2em;color:var(--gold2);}
.empty-title span{color:var(--teal);}
.empty-sub{font-family:'Crimson Pro',serif;font-style:italic;font-size:0.95rem;color:var(--text3);line-height:1.8;max-width:380px;}
.chip{display:inline-block;background:var(--sand3);border:1px solid var(--border2);border-top:1px solid var(--gold);border-radius:3px;padding:0.4rem 1rem;margin:0.25rem;font-family:'Cinzel',serif;font-size:0.68rem;letter-spacing:0.1em;color:var(--text2);text-transform:uppercase;}
.cartouche-top,.cartouche-bot{text-align:center;font-size:0.7rem;letter-spacing:0.5em;color:var(--gold);opacity:0.25;padding:0.3rem 0;user-select:none;}
.tts-player{background:var(--sand2);border:1px solid var(--border2);border-top:2px solid var(--gold);border-radius:8px;padding:0.7rem 1rem 0.6rem;margin:0.55rem 0 0.1rem;max-width:820px;}
.tts-label{font-family:'Cinzel',serif;font-size:0.58rem;letter-spacing:0.18em;color:var(--gold);margin-bottom:0.45rem;display:flex;align-items:center;gap:0.4rem;text-transform:uppercase;}
.tts-label .tts-glyph{font-size:0.88rem;animation:glow-eye 3s ease-in-out infinite;}
.tts-player audio{width:100%;height:34px;outline:none;border-radius:4px;accent-color:#c8922a;filter:sepia(0.15) brightness(0.9);}
[data-testid="stFileUploader"]{width:48px!important;min-width:48px!important;max-width:48px!important;height:48px!important;min-height:48px!important;overflow:hidden!important;border-radius:9px!important;border:1px solid var(--border2)!important;background:linear-gradient(160deg,var(--sand3),var(--sand2))!important;transition:border-color .18s,box-shadow .18s,transform .12s!important;position:relative!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--gold)!important;box-shadow:0 0 14px rgba(200,146,42,0.3)!important;transform:translateY(-1px)!important;}
[data-testid="stFileUploader"]>div{position:absolute!important;inset:0!important;width:48px!important;height:48px!important;opacity:0!important;z-index:2!important;cursor:pointer!important;overflow:hidden!important;}
[data-testid="stFileUploader"] section,[data-testid="stFileUploaderDropzone"]{position:absolute!important;inset:0!important;width:48px!important;height:48px!important;padding:0!important;margin:0!important;border:none!important;background:transparent!important;cursor:pointer!important;}
[data-testid="stFileUploader"] input[type="file"]{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;opacity:0!important;cursor:pointer!important;z-index:3!important;}
[data-testid="stAudioInput"]{border-radius:9px!important;border:1px solid var(--border2)!important;background:linear-gradient(160deg,var(--sand3),var(--sand2))!important;transition:border-color .18s,box-shadow .18s!important;padding:2px!important;}
[data-testid="stAudioInput"]:hover{border-color:var(--gold)!important;box-shadow:0 0 14px rgba(200,146,42,0.3)!important;}
[data-testid="stAudioInput"]>label{display:none!important;}
[data-testid="stAudioInput"] button{background:transparent!important;color:var(--gold2)!important;border:none!important;border-radius:6px!important;cursor:pointer!important;}
.kmt-bar{max-width:860px!important;margin:0 auto 0.75rem!important;background:var(--sand2)!important;border:1px solid var(--border2)!important;border-top:2px solid var(--gold)!important;border-radius:10px!important;padding:4px 8px 4px 4px!important;box-shadow:0 -4px 24px rgba(200,146,42,0.07),0 2px 8px rgba(0,0,0,0.4)!important;}
div[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--text)!important;font-family:'Crimson Pro',serif!important;font-size:1rem!important;}
div[data-testid="stChatInput"] textarea::placeholder{color:var(--text3)!important;font-style:italic!important;}
div[data-testid="stChatInput"] button{background:linear-gradient(135deg,#c8922a,#e8b84b)!important;border:none!important;border-radius:7px!important;color:#1a1408!important;}
div[data-testid="stButton"] button{background:var(--sand3)!important;border:1px solid var(--border2)!important;border-top:1px solid var(--gold)!important;color:var(--text2)!important;border-radius:3px!important;font-family:'Cinzel',serif!important;font-size:0.72rem!important;letter-spacing:0.08em!important;}
div[data-testid="stSelectbox"]>div>div{background:var(--sand3)!important;border:1px solid var(--border2)!important;border-radius:3px!important;color:var(--text)!important;font-family:'Crimson Pro',serif!important;}
div[data-testid="stExpander"]{background:var(--sand2)!important;border:1px solid var(--border2)!important;border-left:2px solid var(--gold)!important;border-radius:0 4px 4px 0!important;}
div[data-testid="stAlert"]{background:var(--sand2)!important;border:1px solid var(--border2)!important;border-left:3px solid var(--gold)!important;}
hr{border-color:var(--border2)!important;}
.stSpinner>div{border-top-color:var(--gold)!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [
    ("sessions", {}), ("active_id", None),
    ("show_chunks", False), ("show_pipeline", True),
    ("global_model", "llama-3.3-70b-versatile"),
    ("kb_docs", []), ("kb_loaded", False),
    ("pending_image", None), ("pending_audio", None), ("pending_audio_ext", "webm"),
    ("translation_cache", {}),
    ("tts_enabled", True),
    ("tts_cache", {}),
    ("tts_err", ""), ("tts_last_err", ""), ("tts_flags", {}),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — scan and index documents
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.kb_loaded:
    if not CHROMA_AVAILABLE:
        st.error("𓂀 ChromaDB not installed. Run: **pip install chromadb**")
    else:
        with st.spinner("𓂀 Consulting the sacred vault…"):
            scan_and_index()
    st.session_state.kb_loaded = True

# ══════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def new_session(title: str = "New Scroll") -> str:
    """
    Create a new chat session with a unique ID and initial metadata.
    Returns the new session's ID.
    """
    sid = str(int(time.time() * 1000))
    st.session_state.sessions[sid] = {
        "title": title, "messages": [],
        "created": datetime.now().strftime("%b %d"),
    }
    st.session_state.active_id = sid
    return sid


def active_sess():
    """
    Return the session dictionary for the currently active session ID.
    Returns None if no session is active or the ID is invalid.
    """
    if st.session_state.active_id and st.session_state.active_id in st.session_state.sessions:
        return st.session_state.sessions[st.session_state.active_id]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <span class="sb-logo-glyph">𓂀</span>
      <div class="sb-logo-title">KEMET <span>RAG</span></div>
      <div class="sb-logo-sub">Knowledge of the Ancient Scrolls</div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="hieroglyph-border">𓀀 𓁐 𓂋 𓃭 𓄿 𓅓 𓆙 𓇌 𓈖 𓉐</div>', unsafe_allow_html=True)

    if st.button("𓂋  Open New Scroll", use_container_width=True):
        new_session()
        st.rerun()

    st.markdown('<div class="sb-section">𓏏 Pipeline Status</div>', unsafe_allow_html=True)
    clf_ready = CLASSIFIER_AVAILABLE and os.path.exists(CLF_WEIGHTS)
    badges = {
        "🗄 ChromaDB":   CHROMA_AVAILABLE,
        "🔬 Swin-T":     clf_ready,
        "🌐 Translation":TRANSLATION_AVAILABLE,
        "📝 NLTK":       NLTK_AVAILABLE,
        "📄 PDF":        bool(PDF_BACKEND),
        "🔊 TTS":        TTS_AVAILABLE and st.session_state.tts_enabled,
    }
    badge_html = '<div class="pipeline-row">'
    for label, active in badges.items():
        cls  = "pb-on" if active else "pb-off"
        tag  = "ON" if active else "OFF"
        badge_html += f'<span class="pipeline-badge {cls}">{label} {tag}</span>'
    badge_html += "</div>"
    st.markdown(badge_html, unsafe_allow_html=True)

    with st.expander("⚙  Oracle Settings"):
        model_label = st.selectbox("Model", list(GROQ_MODELS.keys()))
        st.session_state.global_model  = GROQ_MODELS[model_label]
        st.session_state.show_chunks   = st.toggle("Reveal source papyri",  value=st.session_state.show_chunks)
        st.session_state.show_pipeline = st.toggle("Show pipeline steps",   value=st.session_state.show_pipeline)
        st.session_state.tts_enabled   = st.toggle(
            "🔊 Speak answers aloud" if TTS_AVAILABLE else "🔊 TTS (pip install elevenlabs)",
            value=st.session_state.tts_enabled,
            disabled=not TTS_AVAILABLE,
        )
        if st.session_state.tts_last_err:
            st.error(f"🔊 TTS Warning: {st.session_state.tts_last_err}")

    st.markdown('<div class="sb-section">𓏏 Sacred Library</div>', unsafe_allow_html=True)

    n_pdfs   = sum(1 for d in st.session_state.kb_docs if d.get("type") == "pdf"  and "error" not in d)
    n_texts  = sum(1 for d in st.session_state.kb_docs if d.get("type") == "text")
    n_chunks = chroma_count()
    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-c"><span class="stat-n">{n_pdfs}</span><span class="stat-l">Scrolls</span></div>
      <div class="stat-c"><span class="stat-n">{n_texts}</span><span class="stat-l">Tablets</span></div>
      <div class="stat-c"><span class="stat-n">{n_chunks}</span><span class="stat-l">Vectors</span></div>
    </div>""", unsafe_allow_html=True)

    if st.button("𓂋  Reload Sacred Texts", use_container_width=True):
        chroma_clear()
        st.session_state.kb_docs   = []
        st.session_state.kb_loaded = False
        with st.spinner("𓂀 Re-reading scrolls…"):
            scan_and_index()
        st.session_state.kb_loaded = True
        st.success(f"𓂀 {len(st.session_state.kb_docs)} sacred text(s) restored")
        st.rerun()

    for doc in st.session_state.kb_docs:
        is_pdf  = doc.get("type") == "pdf"
        has_err = "error" in doc
        sz      = f"{doc['size']//1000}k" if doc["size"] >= 1000 else f"{doc['size']}c"
        pages   = f" · {doc.get('pages','?')}pp" if is_pdf and not has_err else ""
        badge   = "pdf" if is_pdf else "txt"
        err_map = {"no_library": "no lib", "no_text": "no text / scanned"}
        err_note = f' <span style="color:#e05050;font-size:.6rem;">⚠ {err_map.get(doc.get("error",""),"⚠")}</span>' if has_err else ""
        st.markdown(f"""
        <div class="{'file-pill pdf-pill' if is_pdf else 'file-pill'}">
          <div class="{'fp-dot pdf' if is_pdf else 'fp-dot'}"></div>
          <span class="fp-name">{doc['name']}{err_note}</span>
          <span class="fp-badge {badge}">{"SCROLL" if is_pdf else "TABLET"}</span>
          <span class="fp-size">{sz}{pages}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-section">𓏏 Open Scrolls</div>', unsafe_allow_html=True)
    if not st.session_state.sessions:
        st.markdown('<p style="font-size:.8rem;color:#555;padding:.25rem 1rem;font-style:italic;">No scrolls opened yet.</p>', unsafe_allow_html=True)
    else:
        for sid, si in reversed(list(st.session_state.sessions.items())):
            label = si["title"][:26] + ("…" if len(si["title"]) > 26 else "")
            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(f"𓏏  {label}", key=f"sess_{sid}", use_container_width=True):
                    st.session_state.active_id = sid
                    st.rerun()
            with c2:
                if st.button("✕", key=f"del_{sid}"):
                    del st.session_state.sessions[sid]
                    if st.session_state.active_id == sid:
                        st.session_state.active_id = None
                    st.rerun()

    st.markdown("---")
    st.markdown('<div class="hieroglyph-border">𓀀 𓁐 𓂋 𓃭 𓄿 𓅓 𓆙 𓇌 𓈖 𓉐</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
sess          = active_sess()
model_display = st.session_state.global_model.split("-")[0].upper()
chat_title    = sess["title"] if sess else "Kemet · RAG"

st.markdown(f"""
<div class="topbar">
  <div class="topbar-title">𓏏 &nbsp;{chat_title}</div>
  <div class="topbar-badge"><span class="topbar-eye">𓂀</span> {model_display}</div>
</div>""", unsafe_allow_html=True)

if sess is None:
    st.markdown("""
    <div class="empty-wrap">
      <div class="empty-glyph">𓂀</div>
      <div class="empty-title">KEMET <span>RAG</span></div>
      <div class="empty-sub">The Eye of Horus sees all knowledge.<br>Open a new scroll and consult the sacred library.</div>
      <div style="margin-top:.5rem;">
        <span class="chip">𓏏 PDF Scrolls</span>
        <span class="chip">𓅓 Text Tablets</span>
        <span class="chip">𓂋 Semantic Search</span>
        <span class="chip">𓁐 Groq Oracle</span>
        <span class="chip">🗄 ChromaDB</span>
        <span class="chip">🌐 Multi-Language</span>
        <span class="chip">🎙 Voice Record</span>
        <span class="chip">🖼 Image Vision</span>
        <span class="chip">🔊 TTS Oracle</span>
      </div>
    </div>""", unsafe_allow_html=True)
    if st.button("𓂋  Open First Scroll"):
        new_session()
        st.rerun()
else:
    messages = sess["messages"]
    st.markdown('<div class="cartouche-top">── 𓀀 ── 𓁐 ── 𓂋 ── 𓃭 ── 𓄿 ──</div>', unsafe_allow_html=True)

    chat_area = st.container(height=460)
    with chat_area:
        if not messages:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:360px;gap:1rem;text-align:center;">
              <div style="font-size:2.5rem;opacity:.12;animation:float 4s ease-in-out infinite;">𓂀</div>
              <div style="font-family:'Crimson Pro',serif;font-style:italic;font-size:.95rem;color:#4a3c1c;line-height:2;">
                Type · press 🖼 to attach an image · press 🎙 to record voice.<br>
                <span style="color:#3a2e14;">Write in any language — the oracle will understand.</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in messages:
                with st.chat_message(msg["role"]):
                    if msg.get("image_bytes"):
                        st.image(msg["image_bytes"], width=220)
                    if msg.get("classification_html"):
                        st.markdown(msg["classification_html"], unsafe_allow_html=True)
                    if msg.get("audio_label"):
                        st.caption(f"🎙 {msg['audio_label']}")

                    # Pipeline trace (user messages only)
                    if msg["role"] == "user" and st.session_state.show_pipeline and msg.get("pipeline"):
                        p = msg["pipeline"]
                        lang_html = f'<span class="lang-badge">{p["detected_lang"].upper()}</span>'
                        if p["was_translated"]:
                            lang_html += '<span class="translated-badge">→ EN translated</span>'
                        st.markdown(f"""
                        <div class="step-card">
                          <span class="step-label">𓂋 PIPELINE TRACE</span>
                          <span class="step-val">{lang_html}</span><br>
                          <span style="opacity:.6">cleaned: </span><span class="step-val">{p["cleaned_text"][:120]}{"…" if len(p["cleaned_text"])>120 else ""}</span>
                        </div>""", unsafe_allow_html=True)

                    st.markdown(msg["content"])

                    # Audio player (assistant messages)
                    if msg["role"] == "assistant":
                        mp3_data = st.session_state.tts_cache.get(msg.get("tts_key", ""))
                        if mp3_data:
                            fmt = "audio/wav" if mp3_data[:4] == b"RIFF" else "audio/mp3"
                            st.caption("🔊 Oracle Voice  ·  🎵 Egyptian Music")
                            st.audio(mp3_data, format=fmt)

                    # Source papyri
                    if msg["role"] == "assistant" and st.session_state.show_chunks and msg.get("sources"):
                        with st.expander(f"𓏏 {len(msg['sources'])} source papyri"):
                            for c in msg["sources"]:
                                pct    = int(c.get("score", 0) * 100)
                                is_pdf = c.get("type") == "pdf"
                                pg_info= f" · p.{c['page']}" if c.get("page") else ""
                                st.markdown(f"""
                                <div class="{'src-box pdf-src' if is_pdf else 'src-box'}">
                                  <span class="{'src-label pdf-lbl' if is_pdf else 'src-label'}">
                                    {'𓏏' if is_pdf else '𓅓'} {c['source']}{pg_info} · glyph #{c['chunk_id']} · {pct}% resonance
                                  </span>
                                  {c['text'][:300]}{"…" if len(c['text'])>300 else ""}
                                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="cartouche-bot">── 𓀀 ── 𓁐 ── 𓂋 ── 𓃭 ── 𓄿 ──</div>', unsafe_allow_html=True)

    # ── Attachment preview ────────────────────────────────────────────────────
    if st.session_state.pending_image or st.session_state.pending_audio:
        parts = []
        if st.session_state.pending_image:
            b64img = base64.b64encode(st.session_state.pending_image).decode()
            parts.append(f'<img src="data:image/jpeg;base64,{b64img}" style="height:36px;border-radius:3px;">')
        if st.session_state.pending_audio:
            parts.append('<span style="background:rgba(194,48,48,0.18);border:1px solid rgba(194,48,48,0.4);color:#e05050;border-radius:3px;padding:2px 8px;font-family:Cinzel,serif;font-size:0.58rem;">● AUDIO READY</span>')
        c_prev, c_x = st.columns([20, 1])
        with c_prev:
            st.markdown(f'<div style="max-width:860px;margin:.4rem auto 0;padding:0 .75rem;"><div style="background:var(--sand3);border:1px solid var(--border2);border-bottom:none;border-radius:6px 6px 0 0;padding:.38rem .9rem;display:flex;align-items:center;gap:.6rem;">{"".join(parts)}</div></div>', unsafe_allow_html=True)
        with c_x:
            if st.button("✕", key="clr_attach"):
                st.session_state.pending_image = None
                st.session_state.pending_audio = None
                st.rerun()

    # ── Input bar ─────────────────────────────────────────────────────────────
    col_img, col_mic, col_chat = st.columns([1, 3, 14])

    with col_img:
        img_file = st.file_uploader(
            "🖼✔" if st.session_state.pending_image else "🖼",
            type=["png","jpg","jpeg","webp","gif"],
            key="img_up", label_visibility="visible",
        )
        if img_file is not None:
            new_bytes = img_file.getvalue()
            if new_bytes != st.session_state.pending_image:
                st.session_state.pending_image = new_bytes
                st.rerun()

    with col_mic:
        try:
            audio_val = st.audio_input("🎙", key="mic_input", label_visibility="visible")
            if audio_val is not None:
                new_audio = audio_val.getvalue()
                if new_audio != st.session_state.pending_audio:
                    st.session_state.pending_audio     = new_audio
                    st.session_state.pending_audio_ext = "wav"
                    st.rerun()
        except AttributeError:
            aud_file = st.file_uploader(
                "🎙", type=["wav","mp3","m4a","ogg","webm","flac"],
                key="aud_up", label_visibility="visible",
            )
            if aud_file is not None:
                new_bytes = aud_file.getvalue()
                if new_bytes != st.session_state.pending_audio:
                    st.session_state.pending_audio     = new_bytes
                    st.session_state.pending_audio_ext = aud_file.name.rsplit(".", 1)[-1].lower()
                    st.rerun()

    with col_chat:
        user_text = st.chat_input("Speak your question to the oracle… (any language)", key="oracle_chat")

    # ══════════════════════════════════════════════════════════════════════════
    # PROCESS SEND — orchestrate all pipelines
    # ══════════════════════════════════════════════════════════════════════════
    if user_text is not None:
        pending_img   = st.session_state.pending_image
        pending_audio = st.session_state.pending_audio
        audio_ext     = st.session_state.pending_audio_ext
        raw_text      = (user_text or "").strip()

        if not raw_text and not pending_img and not pending_audio:
            st.stop()

        # ── PIPELINE 1: Transcribe audio ──────────────────────────────────
        audio_label = None
        if pending_audio:
            with st.spinner("𓂀 Deciphering the voice offering…"):
                transcript = transcribe_audio(pending_audio, audio_ext)
            audio_label = f'Voice: "{transcript[:70]}{"…" if len(transcript)>70 else ""}"'
            raw_text    = (transcript + (" · " + raw_text if raw_text else "")).strip()

        # ── PIPELINE 7: Classify and describe image ───────────────────────
        image_answer        = None
        classification_html = ""
        if pending_img:
            if CLASSIFIER_AVAILABLE and os.path.exists(CLF_WEIGHTS):
                with st.spinner("𓂀 The Eye of Horus classifies the image…"):
                    clf_result = classify_image(pending_img)
                if "predictions" in clf_result:
                    preds = clf_result["predictions"]
                    top   = preds[0]
                    # If confidence is too low, we don't 'know' the landmark
                    if top['confidence'] < 45.0:
                        classification_html = f"""<div class="step-card" style="margin-bottom:6px; border-left-color: var(--red);">
                          <span class="step-label" style="color:var(--red2);">𓂀 UNKNOWN SUBJECT</span>
                          <span style="color:var(--text3);font-size:0.75rem;">This is not from my knowledge.</span>
                        </div>"""
                        raw_text = ("[Unknown landmark - not from knowledge]"
                                    + (" — " + raw_text if raw_text else "")).strip()
                    else:
                        bars = ""
                        for p in preds:
                            bars += f"""<div style="margin:3px 0;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">
                              <span style="color:var(--gold2);min-width:220px;display:inline-block;">{p['label']}</span>
                              <span style="background:var(--sand3);border:1px solid var(--border2);border-radius:3px;display:inline-block;width:120px;height:10px;vertical-align:middle;">
                                <span style="display:block;width:{int(p['confidence'])}%;height:100%;background:linear-gradient(90deg,var(--gold),var(--gold2));border-radius:3px;"></span>
                              </span>
                              <span style="color:var(--text3);margin-left:6px;">{p['confidence']}%</span>
                            </div>"""
                        classification_html = f"""<div class="step-card" style="margin-bottom:6px;">
                          <span class="step-label">𓂀 SWIN-T CLASSIFICATION</span>
                          <span style="color:var(--teal);font-size:0.75rem;">▶ {top['label']} ({top['confidence']}%)</span>
                          {bars}</div>"""
                        raw_text = (f"[Image classified as: {top['label']} ({top['confidence']}% confidence)]"
                                    + (" — " + raw_text if raw_text else "")).strip()

            with st.spinner("𓂀 The Eye of Horus examines the image…"):
                image_answer = describe_image(pending_img, raw_text or "Describe this image in detail.")

        # ── PIPELINE 2: NLP preprocessing ────────────────────────────────
        pipeline_info = {}
        query_for_rag = raw_text or "(Image submitted)"
        
        # RAG Boost: If we have a very high confidence classification, add it to the query
        if pending_img and CLASSIFIER_AVAILABLE and os.path.exists(CLF_WEIGHTS):
            try:
                clf_result = classify_image(pending_img)
                if "predictions" in clf_result:
                    top = clf_result["predictions"][0]
                    if top['confidence'] > 85.0:
                        query_for_rag = (f"{top['label']} " + (raw_text or "")).strip()
            except Exception:
                pass

        if raw_text:
            with st.spinner("𓂀 Processing through the pipeline…"):
                pipeline_info = full_text_pipeline(raw_text)
                query_for_rag = pipeline_info.get("query_for_rag", query_for_rag)

        # ── Session title (set on first message) ──────────────────────────
        if not sess["messages"]:
            if pending_img and not raw_text:
                sess["title"] = "Image query"
            elif pending_audio and not raw_text:
                sess["title"] = "Voice query"
            else:
                sess["title"] = raw_text[:40]

        # ── PIPELINE 4: Retrieve relevant chunks ──────────────────────────
        retrieved = []
        if query_for_rag not in ("(Image submitted)", "(Audio submitted)"):
            with st.spinner("𓂀 Searching the sacred vault…"):
                retrieved = retrieve(query_for_rag, top_k=5)

        # Store user message
        sess["messages"].append({
            "role":                "user",
            "content":             raw_text or ("🖼 Image" if pending_img else "🎙 Voice"),
            "image_bytes":         pending_img,
            "audio_label":         audio_label,
            "pipeline":            pipeline_info,
            "classification_html": classification_html,
        })

        # ── PIPELINE 5: LLM generation ────────────────────────────────────
        ctx      = build_context(retrieved) if retrieved else "No relevant scrolls found in the library."
        msgs_llm = [{"role": "system", "content": (
            "You are a professional Egyptian tour guide with deep knowledge of history and culture.\n"
            "Use the library context below AND the image analysis to answer the tourist's question.\n"
            "If the answer is not in either, say 'I don't have that information in my tour notes.'\n\nContext:\n" + ctx
        )}]
        for m in sess["messages"][-8:]:
            msgs_llm.append({"role": m["role"], "content": m["content"]})

        combined = (f"[Image Analysis]\n{image_answer}\n\n" if image_answer else "") + f"Context:\n{ctx}"
        msgs_llm[-1]["content"] = (
            f"{combined}\n\nQuestion: {raw_text or '(Describe the image as a tour guide)'}"
        )

        with st.spinner("𓂀 The oracle consults the scrolls…"):
            answer = groq_chat(st.session_state.global_model, msgs_llm)

        # ── PIPELINE 6: TTS ────────────────────────────────────────────────
        st.session_state["tts_flags"] = {
            "TTS_AVAILABLE": TTS_AVAILABLE,
            "tts_enabled":   st.session_state.get("tts_enabled", True),
        }
        tts_wav = None
        if TTS_AVAILABLE and st.session_state.get("tts_enabled", True):
            with st.spinner("𓂀 The oracle speaks…"):
                tts_wav = run_tts_pipeline(answer)
            st.session_state["tts_last_err"] = st.session_state.get("tts_err", "")
        else:
            reasons = []
            if not TTS_AVAILABLE:
                reasons.append("elevenlabs not installed")
            if not st.session_state.get("tts_enabled", True):
                reasons.append("TTS toggled off")
            st.session_state["tts_last_err"] = " | ".join(reasons)

        # Cache WAV bytes keyed by session + message index
        msg_idx   = len(sess["messages"])
        cache_key = f"{st.session_state.active_id}:{msg_idx}"
        if tts_wav:
            st.session_state.tts_cache[cache_key] = tts_wav

        sess["messages"].append({
            "role":    "assistant",
            "content": answer,
            "sources": retrieved,
            "tts_key": cache_key if tts_wav else None,
        })

        # Clear pending attachments
        st.session_state.pending_image = None
        st.session_state.pending_audio = None
        st.rerun()
