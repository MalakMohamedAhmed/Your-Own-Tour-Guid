"""
╔══════════════════════════════════════════════════════════════════════╗
║  KEMET · RAG  —  Unified Application  (ChromaDB + TTS Edition)      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, glob, hashlib, re, tempfile, base64, time, io
from datetime import datetime
import numpy as np
import streamlit as st

# ── PDF backend ───────────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except ImportError:
        PDF_BACKEND = None

# ── Translation ───────────────────────────────────────────────────────────────
TRANSLATION_AVAILABLE = False
_lang_detector = None
_translators   = {}
try:
    from transformers import pipeline as hf_pipeline, MarianMTModel, MarianTokenizer
    TRANSLATION_AVAILABLE = True
except ImportError:
    pass

# ── NLTK ──────────────────────────────────────────────────────────────────────
NLTK_AVAILABLE = False
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet",   quiet=True)
    nltk.download("omw-1.4",  quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem  import WordNetLemmatizer
    NLTK_AVAILABLE = True
except Exception:
    pass

# ── spaCy ─────────────────────────────────────────────────────────────────────
SPACY_AVAILABLE = False
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    pass

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    pass

from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Swin-T Image Classifier ───────────────────────────────────────────────────
CLASSIFIER_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import swin_t
    from PIL import Image as PILImage
    import io as _io
    CLASSIFIER_AVAILABLE = True
except ImportError:
    pass

# ── ElevenLabs TTS ────────────────────────────────────────────────────────────
TTS_AVAILABLE = False
try:
    from elevenlabs.client import ElevenLabs as ELabs
    from elevenlabs import save as el_save
    TTS_AVAILABLE = True
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "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")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "sk_a280e81e4702d56c7f69dfea1c931775cfa9912cd1d066f3")
DOCS_FOLDER     = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR      = os.path.join(DOCS_FOLDER, ".kemet_chroma_db")
COLLECTION_NAME = "kemet_scrolls"
CLASSIFIER_WEIGHTS = os.path.join(DOCS_FOLDER, "model.pth")

# TTS settings
TTS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"   # Adam — crystal-clear narrator
TTS_MODEL    = "eleven_multilingual_v2"

# ── 20 Class names ────────────────────────────────────────────────────────────
CLASSIFIER_CLASSES = [
    "Abu Simbel", "Alexandria Library", "Bent Pyramid", "Citadel of Qaitbay",
    "Colossi of Memnon", "Egyptian Museum", "Great Pyramid of Giza",
    "Great Sphinx", "Karnak Temple", "Khan el-Khalili", "Luxor Temple",
    "Medinet Habu", "Mortuary Temple of Hatshepsut", "Philae Temple",
    "Pyramid of Khafre", "Pyramid of Menkaure", "Red Pyramid",
    "Step Pyramid of Djoser", "Temple of Edfu", "Valley of the Kings",
]

GROQ_MODELS = {
    "LLaMA 3.3 · 70B":   "llama-3.3-70b-versatile",
    "LLaMA 3.1 · 8B ⚡": "llama-3.1-8b-instant",
    "Mixtral · 8×7B":    "mixtral-8x7b-32768",
    "Gemma 2 · 9B":      "gemma2-9b-it",
}

TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
}

IRREGULAR_VERBS = {
    "went":"go","gone":"go","goes":"go","was":"be","were":"be","been":"be",
    "is":"be","are":"be","had":"have","has":"have","did":"do","done":"do",
    "does":"do","saw":"see","seen":"see","came":"come","comes":"come",
    "took":"take","taken":"take","takes":"take","made":"make","makes":"make",
    "said":"say","says":"say","got":"get","gotten":"get","gets":"get",
    "knew":"know","known":"know","knows":"know","thought":"think",
    "thinks":"think","brought":"bring","brings":"bring","bought":"buy",
    "buys":"buy","taught":"teach","teaches":"teach","caught":"catch",
    "catches":"catch","left":"leave","leaves":"leave","kept":"keep",
    "keeps":"keep","felt":"feel","feels":"feel","met":"meet","meets":"meet",
    "ran":"run","runs":"run","gave":"give","given":"give","gives":"give",
    "found":"find","finds":"find","told":"tell","tells":"tell",
    "stood":"stand","stands":"stand","spent":"spend","spends":"spend",
    "began":"begin","begun":"begin","begins":"begin","going":"go",
    "exploring":"explore","wandering":"wander","visiting":"visit",
    "admiring":"admire","crossed":"cross","sailed":"sail","climbed":"climb",
}

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
/* ── TTS Audio Player ─────────────────────────────────────────────── */
.tts-player{
  background:var(--sand2);
  border:1px solid var(--border2);
  border-top:2px solid var(--gold);
  border-radius:8px;
  padding:0.7rem 1rem 0.6rem;
  margin:0.55rem 0 0.1rem;
  max-width:820px;
}
.tts-label{
  font-family:'Cinzel',serif;
  font-size:0.58rem;
  letter-spacing:0.18em;
  color:var(--gold);
  margin-bottom:0.45rem;
  display:flex;
  align-items:center;
  gap:0.4rem;
  text-transform:uppercase;
}
.tts-label .tts-glyph{
  font-size:0.88rem;
  animation:glow-eye 3s ease-in-out infinite;
}
.tts-player audio{
  width:100%;
  height:34px;
  outline:none;
  border-radius:4px;
  accent-color:#c8922a;
  filter:sepia(0.15) brightness(0.9);
}
[data-testid="stFileUploader"]{width:48px!important;min-width:48px!important;max-width:48px!important;height:48px!important;min-height:48px!important;overflow:hidden!important;border-radius:9px!important;border:1px solid var(--border2)!important;background:linear-gradient(160deg,var(--sand3),var(--sand2))!important;transition:border-color .18s,box-shadow .18s,transform .12s!important;position:relative!important;}
[data-testid="stFileUploader"]:hover{border-color:var(--gold)!important;box-shadow:0 0 14px rgba(200,146,42,0.3)!important;transform:translateY(-1px)!important;}
[data-testid="stFileUploader"]>div{position:absolute!important;inset:0!important;width:48px!important;height:48px!important;opacity:0!important;z-index:2!important;cursor:pointer!important;overflow:hidden!important;}
[data-testid="stFileUploader"] section,[data-testid="stFileUploaderDropzone"]{position:absolute!important;inset:0!important;width:48px!important;height:48px!important;padding:0!important;margin:0!important;border:none!important;background:transparent!important;cursor:pointer!important;}
[data-testid="stFileUploader"] input[type="file"]{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;opacity:0!important;cursor:pointer!important;z-index:3!important;}
[data-testid="stFileUploader"] button{position:absolute!important;inset:0!important;width:100%!important;height:100%!important;opacity:0!important;cursor:pointer!important;z-index:3!important;}
[data-testid="stFileUploader"]>label{position:absolute!important;inset:0!important;z-index:1!important;display:flex!important;align-items:center!important;justify-content:center!important;width:48px!important;height:48px!important;cursor:pointer!important;margin:0!important;padding:0!important;font-size:1.35rem!important;line-height:1!important;color:var(--text2)!important;font-family:inherit!important;letter-spacing:0!important;text-transform:none!important;background:transparent!important;border:none!important;pointer-events:none!important;}
[data-testid="stFileUploader"]:hover>label{color:var(--gold2)!important;}
[data-testid="stAudioInput"]{border-radius:9px!important;border:1px solid var(--border2)!important;background:linear-gradient(160deg,var(--sand3),var(--sand2))!important;transition:border-color .18s,box-shadow .18s!important;padding:2px!important;}
[data-testid="stAudioInput"]:hover{border-color:var(--gold)!important;box-shadow:0 0 14px rgba(200,146,42,0.3)!important;}
[data-testid="stAudioInput"]>label{display:none!important;}
[data-testid="stAudioInput"] button{background:transparent!important;color:var(--gold2)!important;border:none!important;border-radius:6px!important;cursor:pointer!important;}
[data-testid="stAudioInput"] button:hover{color:var(--gold)!important;background:rgba(200,146,42,0.1)!important;}
[data-testid="stAudioInput"] audio{height:28px!important;max-width:100%!important;filter:invert(0.8) sepia(0.3)!important;}
.kmt-bar{max-width:860px!important;margin:0 auto 0.75rem!important;background:var(--sand2)!important;border:1px solid var(--border2)!important;border-top:2px solid var(--gold)!important;border-radius:10px!important;padding:4px 8px 4px 4px!important;box-shadow:0 -4px 24px rgba(200,146,42,0.07),0 2px 8px rgba(0,0,0,0.4)!important;}
.kmt-bar [data-testid="stHorizontalBlock"]{gap:6px!important;align-items:center!important;flex-wrap:nowrap!important;}
.kmt-bar [data-testid="stHorizontalBlock"]>div:first-child{flex:0 0 48px!important;min-width:48px!important;max-width:48px!important;padding:0!important;}
div[data-testid="stChatInput"]{flex:1!important;margin:0!important;padding:0!important;background:transparent!important;border:none!important;box-shadow:none!important;border-radius:0!important;}
div[data-testid="stChatInput"]>div{background:transparent!important;border:none!important;box-shadow:none!important;}
div[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--text)!important;font-family:'Crimson Pro',serif!important;font-size:1rem!important;}
div[data-testid="stChatInput"] textarea::placeholder{color:var(--text3)!important;font-style:italic!important;}
div[data-testid="stChatInput"] button{background:linear-gradient(135deg,#c8922a,#e8b84b)!important;border:none!important;border-radius:7px!important;color:#1a1408!important;}
div[data-testid="stChatInput"] button:hover{box-shadow:0 0 16px rgba(200,146,42,0.5)!important;}
.attach-strip{max-width:860px;margin:0.4rem auto 0;padding:0 0.75rem;}
.attach-inner{background:var(--sand3);border:1px solid var(--border2);border-bottom:none;border-radius:6px 6px 0 0;padding:0.38rem 0.9rem;display:flex;align-items:center;gap:0.6rem;font-family:'Crimson Pro',serif;font-size:0.78rem;color:var(--text2);}
.attach-inner img{height:36px;border-radius:3px;border:1px solid var(--border2);}
.rec-pill{background:rgba(194,48,48,0.18);border:1px solid rgba(194,48,48,0.4);color:var(--pdf2);border-radius:3px;padding:2px 8px;font-family:'Cinzel',serif;font-size:0.58rem;letter-spacing:0.1em;animation:pulse-r 1.1s ease-in-out infinite;}
@keyframes pulse-r{0%,100%{opacity:1;}50%{opacity:0.3;}}
div[data-testid="stButton"] button{background:var(--sand3)!important;border:1px solid var(--border2)!important;border-top:1px solid var(--gold)!important;color:var(--text2)!important;border-radius:3px!important;font-family:'Cinzel',serif!important;font-size:0.72rem!important;letter-spacing:0.08em!important;font-weight:500!important;transition:all 0.2s!important;}
div[data-testid="stButton"] button:hover{background:var(--sand4)!important;border-color:var(--gold2)!important;color:var(--gold2)!important;}
div[data-testid="stSelectbox"]>div>div{background:var(--sand3)!important;border:1px solid var(--border2)!important;border-radius:3px!important;color:var(--text)!important;font-family:'Crimson Pro',serif!important;font-size:0.9rem!important;}
div[data-testid="stExpander"]{background:var(--sand2)!important;border:1px solid var(--border2)!important;border-left:2px solid var(--gold)!important;border-radius:0 4px 4px 0!important;}
div[data-testid="stExpander"] summary{color:var(--text3)!important;font-family:'Cinzel',serif!important;font-size:0.68rem!important;}
div[data-testid="stAlert"]{background:var(--sand2)!important;border:1px solid var(--border2)!important;border-left:3px solid var(--gold)!important;border-radius:0 4px 4px 0!important;font-size:0.85rem!important;font-family:'Crimson Pro',serif!important;}
div[data-testid="stWidgetLabel"] p,label{color:var(--text3)!important;font-size:0.75rem!important;font-family:'Cinzel',serif!important;letter-spacing:0.08em!important;}
hr{border-color:var(--border2)!important;margin:0.75rem 0!important;}
.stCaption{color:var(--text3)!important;font-family:'Crimson Pro',serif!important;font-style:italic!important;font-size:0.78rem!important;}
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
    ("tts_cache", {}),       # key="sid:msg_idx" → MP3 bytes
    ("tts_err", ""),             # last error string
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# CHROMADB SETUP
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="𓂀 Awakening the scribe…")
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="𓂀 Opening the sacred vault…")
def get_chroma_collection():
    if not CHROMA_AVAILABLE:
        return None
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chroma_count() -> int:
    col = get_chroma_collection()
    if col is None:
        return 0
    try:
        return col.count()
    except Exception:
        return 0


def chroma_clear():
    if not CHROMA_AVAILABLE:
        return
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    get_chroma_collection.clear()


def _file_fingerprint(fpath: str) -> str:
    s = os.stat(fpath)
    return hashlib.sha256(f"{fpath}|{s.st_mtime}|{s.st_size}".encode()).hexdigest()


def _indexed_fingerprints() -> dict:
    col = get_chroma_collection()
    if col is None or col.count() == 0:
        return {}
    try:
        results = col.get(include=["metadatas"])
        fps = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "")
            fp  = meta.get("fingerprint", "")
            if src and fp:
                fps[src] = fp
        return fps
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 1: Language Detection + Translation
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="𓂀 Loading language oracle…")
def _load_lang_detector():
    if not TRANSLATION_AVAILABLE:
        return None
    try:
        return hf_pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
        )
    except Exception:
        return None


def _load_translator(lang_code: str):
    if lang_code in _translators:
        return _translators[lang_code]
    model_name = TRANSLATION_MODELS.get(lang_code)
    if not model_name:
        return None, None
    try:
        tok   = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _translators[lang_code] = (tok, model)
        return tok, model
    except Exception:
        return None, None


def detect_and_translate(text: str) -> dict:
    result = {
        "original": text, "detected_lang": "en",
        "confidence": 1.0, "was_translated": False, "english_text": text,
    }
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key]
    if not TRANSLATION_AVAILABLE:
        return result
    detector = _load_lang_detector()
    if not detector:
        return result
    try:
        detection = detector(text[:512])[0]
        lang  = detection["label"]
        score = detection["score"]
        result["detected_lang"] = lang
        result["confidence"]    = score
        if lang != "en" and lang in TRANSLATION_MODELS:
            tok, model = _load_translator(lang)
            if tok and model:
                tokens     = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = model.generate(**tokens)
                english    = tok.decode(translated[0], skip_special_tokens=True)
                result["was_translated"] = True
                result["english_text"]   = english
    except Exception:
        pass
    st.session_state.translation_cache[cache_key] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 2: NLP Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


@st.cache_resource(show_spinner=False)
def _get_nltk_tools():
    if not NLTK_AVAILABLE:
        return None, None
    return set(stopwords.words("english")), WordNetLemmatizer()


def safe_lemmatize(word: str) -> str:
    sw, lem = _get_nltk_tools()
    if not lem:
        return word
    w = word.lower()
    if w in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[w]
    candidates = []
    for pos in ('v', 'a', 'n', 'r'):
        lemma = lem.lemmatize(w, pos=pos)
        if lemma != w and _edit_distance(w, lemma) > len(w) * 0.5:
            continue
        candidates.append(lemma)
    return candidates[0] if candidates else w


def advanced_preprocess(text: str) -> str:
    sw, _ = _get_nltk_tools()
    text = text.lower()
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not sw:
        return text
    return " ".join(
        word if "_" in word else safe_lemmatize(word)
        for word in text.split() if word not in sw
    )


def full_text_pipeline(raw_text: str) -> dict:
    trans   = detect_and_translate(raw_text)
    eng     = trans["english_text"]
    cleaned = advanced_preprocess(eng)
    return {
        "original":       raw_text,
        "detected_lang":  trans["detected_lang"],
        "confidence":     trans["confidence"],
        "was_translated": trans["was_translated"],
        "english_text":   eng,
        "cleaned_text":   cleaned,
        "query_for_rag":  cleaned if cleaned.strip() else eng,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 3: Chunking + ChromaDB Indexing
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list:
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        c = " ".join(words[i:i + size])
        if c.strip():
            chunks.append(c)
        i += size - overlap
    return chunks


def extract_pdf_text(fpath: str) -> list:
    pages = []
    try:
        if PDF_BACKEND == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(fpath) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        text = (page.extract_text() or "").strip()
                    except Exception:
                        text = ""
                    if text:
                        pages.append({"page": i, "text": text})
        elif PDF_BACKEND == "pypdf":
            import pypdf
            reader = pypdf.PdfReader(fpath)
            for i, page in enumerate(reader.pages, 1):
                try:
                    try:
                        text = page.extract_text(extraction_mode="layout") or ""
                    except TypeError:
                        text = page.extract_text() or ""
                    text = text.strip()
                except Exception:
                    text = ""
                if text:
                    pages.append({"page": i, "text": text})
    except Exception as e:
        st.error(f"𓂀 Failed to read '{os.path.basename(fpath)}': {e}")
    return pages


def _upsert_chunks(col, ids, embeddings, documents, metadatas, batch=100):
    for b in range(0, len(ids), batch):
        col.upsert(
            ids=ids[b:b+batch],
            embeddings=embeddings[b:b+batch],
            documents=documents[b:b+batch],
            metadatas=metadatas[b:b+batch],
        )


def scan_and_index() -> int:
    if not CHROMA_AVAILABLE:
        st.error("𓂀 ChromaDB not installed. Run: **pip install chromadb**")
        return 0

    all_files = sorted(set(
        f for pat in [os.path.join(DOCS_FOLDER, p) for p in ("*.txt", "*.md", "*.pdf")]
        for f in glob.glob(pat)
    ))
    sn = os.path.basename(os.path.abspath(__file__))
    all_files = [
        f for f in all_files
        if os.path.basename(f) != sn and not os.path.basename(f).startswith(".")
    ]

    indexed_fps  = _indexed_fingerprints()
    loaded_names = {d["name"] for d in st.session_state.kb_docs}
    em           = load_embed_model()
    col          = get_chroma_collection()
    new_count    = 0

    for fpath in all_files:
        fname = os.path.basename(fpath)
        fp    = _file_fingerprint(fpath)
        ext   = os.path.splitext(fname)[1].lower()

        if fname in indexed_fps and indexed_fps[fname] == fp:
            if fname not in loaded_names:
                st.session_state.kb_docs.append({
                    "name": fname, "path": fpath,
                    "size": os.path.getsize(fpath),
                    "type": "pdf" if ext == ".pdf" else "text",
                })
            continue

        if fname in indexed_fps:
            try:
                existing = col.get(where={"source": fname})
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
            except Exception:
                pass
            st.session_state.kb_docs = [d for d in st.session_state.kb_docs if d["name"] != fname]

        if ext == ".pdf":
            if PDF_BACKEND is None:
                st.session_state.kb_docs.append({
                    "name": fname, "path": fpath,
                    "size": os.path.getsize(fpath), "type": "pdf", "error": "no_library"
                })
                continue
            pages = extract_pdf_text(fpath)
            if not pages:
                st.session_state.kb_docs.append({
                    "name": fname, "path": fpath,
                    "size": os.path.getsize(fpath), "type": "pdf", "error": "no_text"
                })
                continue
            ids, embeddings, documents, metadatas = [], [], [], []
            for pg in pages:
                for idx, chunk in enumerate(chunk_text(pg["text"])):
                    ids.append(f"{fname}::p{pg['page']}::c{idx}")
                    embeddings.append(em.encode([chunk], convert_to_numpy=True)[0].tolist())
                    documents.append(chunk)
                    metadatas.append({
                        "source": fname, "page": pg["page"],
                        "chunk_id": idx, "type": "pdf", "fingerprint": fp,
                    })
            if ids:
                _upsert_chunks(col, ids, embeddings, documents, metadatas)
            st.session_state.kb_docs.append({
                "name": fname, "path": fpath,
                "size": os.path.getsize(fpath), "type": "pdf", "pages": len(pages)
            })
        else:
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
            except Exception:
                continue
            chunks = chunk_text(raw)
            if not chunks:
                continue
            ids, embeddings, metadatas = [], [], []
            for idx, chunk in enumerate(chunks):
                ids.append(f"{fname}::c{idx}")
                embeddings.append(em.encode([chunk], convert_to_numpy=True)[0].tolist())
                metadatas.append({
                    "source": fname, "page": -1,
                    "chunk_id": idx, "type": "text", "fingerprint": fp,
                })
            _upsert_chunks(col, ids, embeddings, chunks, metadatas)
            st.session_state.kb_docs.append({
                "name": fname, "path": fpath, "size": len(raw), "type": "text"
            })

        new_count += 1

    return new_count


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 4: ChromaDB Retrieval
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(query: str, top_k: int = 5) -> list:
    col = get_chroma_collection()
    if col is None or col.count() == 0:
        return []
    em    = load_embed_model()
    q_emb = em.encode([query], convert_to_numpy=True)[0].tolist()
    try:
        results = col.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - dist
        if score < 0.1:
            continue
        retrieved.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", 0),
            "page":     meta.get("page") if meta.get("page", -1) != -1 else None,
            "type":     meta.get("type", "text"),
            "score":    score,
        })
    return retrieved


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 5: Groq LLM Generation
# ══════════════════════════════════════════════════════════════════════════════

def build_context(retrieved: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source {i} – {c['source']} p.{c.get('page','?')}]\n{c['text']}"
        for i, c in enumerate(retrieved, 1)
    )


def groq_chat(model: str, messages: list) -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        r = client.chat.completions.create(model=model, messages=messages, max_tokens=1024)
        return r.choices[0].message.content
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            return "𓂀 Invalid Groq key — update GROQ_API_KEY."
        return f"𓂀 {err}"


def transcribe_audio(audio_bytes: bytes, ext: str = "webm") -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tp = tmp.name
        with open(tp, "rb") as f:
            r = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(f"audio.{ext}", f, f"audio/{ext}"),
            )
        os.unlink(tp)
        return r.text.strip()
    except Exception as e:
        return f"[Transcription failed: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE 6 — TTS  (ElevenLabs voice only, no music mixing)
# ══════════════════════════════════════════════════════════════════════════════

def run_tts_pipeline(text: str) -> bytes | None:
    """Call ElevenLabs and return raw MP3 bytes. Stores error in session state."""
    st.session_state["tts_err"] = ""
    if not TTS_AVAILABLE:
        st.session_state["tts_err"] = "pip install elevenlabs"
        return None

    clean = re.sub(r"[*_`#>~\[\]]+", "", text).strip()
    clean = re.sub(r"\s+", " ", clean)[:4000]
    if not clean:
        return None
    try:
        client = ELabs(api_key=ELEVENLABS_API_KEY)
        gen    = client.text_to_speech.convert(
            text=clean,
            voice_id=TTS_VOICE_ID,
            model_id=TTS_MODEL,
            output_format="mp3_44100_128",
        )
        data = b"".join(gen)
        if not data:
            st.session_state["tts_err"] = "ElevenLabs returned 0 bytes"
            return None
        return data
    except Exception as e:
        st.session_state["tts_err"] = str(e)
        return None


# PIPELINE 7: Image Classifier
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="𓂀 Loading the Eye of Horus classifier…")
def load_classifier():
    if not CLASSIFIER_AVAILABLE:
        return None
    if not os.path.exists(CLASSIFIER_WEIGHTS):
        return None
    try:
        import torch.nn as nn
        model = swin_t()
        model.head = nn.Linear(model.head.in_features, len(CLASSIFIER_CLASSES))
        state = torch.load(CLASSIFIER_WEIGHTS, map_location="cpu", weights_only=False)
        for key in ("model", "state_dict", "model_state_dict", "net", "weights"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            st.warning(f"𓂀 Classifier: {len(missing)} missing keys")
        model.eval()
        return model
    except Exception as e:
        st.warning(f"𓂀 Could not load classifier: {e}")
        return None


_CLASSIFY_TRANSFORM = None
def _get_transform():
    global _CLASSIFY_TRANSFORM
    if _CLASSIFY_TRANSFORM is None and CLASSIFIER_AVAILABLE:
        _CLASSIFY_TRANSFORM = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return _CLASSIFY_TRANSFORM


def classify_image(image_bytes: bytes) -> dict:
    if not CLASSIFIER_AVAILABLE:
        return {"error": "torch / torchvision not installed"}
    model = load_classifier()
    if model is None:
        return {"error": "Classifier weights not found — place model.pth next to the app"}
    try:
        img = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
        transform = _get_transform()
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        top3 = torch.topk(probs, k=min(3, len(CLASSIFIER_CLASSES)))
        predictions = [
            {
                "label":      CLASSIFIER_CLASSES[idx.item()],
                "confidence": round(prob.item() * 100, 1),
            }
            for prob, idx in zip(top3.values, top3.indices)
        ]
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}


def describe_image(image_bytes: bytes, question: str) -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        b64    = base64.b64encode(image_bytes).decode()
        prompt = question if question.strip() else "Describe this image in detail."
        r = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text",      "text": prompt},
            ]}],
            max_tokens=1024,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"𓂀 Vision error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
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
    sid = str(int(time.time() * 1000))
    st.session_state.sessions[sid] = {
        "title": title, "messages": [],
        "created": datetime.now().strftime("%b %d"),
    }
    st.session_state.active_id = sid
    return sid


def active_sess():
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

    if st.button("𓂋  Open New Scroll", use_container_width=True, key="new_chat_btn"):
        new_session()
        st.rerun()

    st.markdown('<div class="sb-section">𓏏 Pipeline Status</div>', unsafe_allow_html=True)
    chroma_cls = "pb-on" if CHROMA_AVAILABLE else "pb-off"
    trans_cls  = "pb-on" if TRANSLATION_AVAILABLE else "pb-off"
    nltk_cls   = "pb-on" if NLTK_AVAILABLE        else "pb-off"
    pdf_cls    = "pb-on" if PDF_BACKEND            else "pb-off"
    clf_cls    = "pb-on" if (CLASSIFIER_AVAILABLE and os.path.exists(CLASSIFIER_WEIGHTS)) else "pb-off"
    tts_cls    = "pb-on" if (TTS_AVAILABLE and st.session_state.tts_enabled) else "pb-off"
    st.markdown(f"""
    <div class="pipeline-row">
      <span class="pipeline-badge {chroma_cls}">🗄 ChromaDB {'ON' if CHROMA_AVAILABLE else 'OFF'}</span>
      <span class="pipeline-badge {clf_cls}">🔬 Swin-T {'ON' if (CLASSIFIER_AVAILABLE and os.path.exists(CLASSIFIER_WEIGHTS)) else 'OFF'}</span>
      <span class="pipeline-badge {trans_cls}">🌐 Translation {'ON' if TRANSLATION_AVAILABLE else 'OFF'}</span>
      <span class="pipeline-badge {nltk_cls}">📝 NLTK {'ON' if NLTK_AVAILABLE else 'OFF'}</span>
      <span class="pipeline-badge {pdf_cls}">📄 PDF {'ON' if PDF_BACKEND else 'OFF'}</span>
      <span class="pipeline-badge {tts_cls}">🔊 TTS {'ON' if (TTS_AVAILABLE and st.session_state.tts_enabled) else 'OFF'}</span>
    </div>""", unsafe_allow_html=True)

    with st.expander("⚙  Oracle Settings"):
        model_label = st.selectbox("Model", list(GROQ_MODELS.keys()))
        st.session_state.global_model  = GROQ_MODELS[model_label]
        st.session_state.show_chunks   = st.toggle("Reveal source papyri",  value=st.session_state.show_chunks)
        st.session_state.show_pipeline = st.toggle("Show pipeline steps",   value=st.session_state.show_pipeline)
        # ── TTS toggle ────────────────────────────────────────────────────
        st.session_state.tts_enabled = st.toggle(
            "🔊 Speak answers aloud" if TTS_AVAILABLE else "🔊 TTS (pip install elevenlabs)",
            value=st.session_state.tts_enabled,
            disabled=not TTS_AVAILABLE,
        )

    st.markdown('<div class="sb-section">𓏏 Sacred Library</div>', unsafe_allow_html=True)

    if PDF_BACKEND is None and glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")):
        st.markdown('<div style="background:rgba(139,32,32,.15);border:1px solid rgba(194,48,48,.3);border-radius:4px;padding:.6rem .9rem;margin:.4rem .75rem;font-size:.78rem;color:#e05050;">⚠ PDF files found but no reader.<br>Run: <b>pip install pdfplumber</b></div>', unsafe_allow_html=True)

    n_pdfs   = sum(1 for d in st.session_state.kb_docs if d.get("type") == "pdf"  and "error" not in d)
    n_texts  = sum(1 for d in st.session_state.kb_docs if d.get("type") == "text")
    n_chunks = chroma_count()

    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-c"><span class="stat-n">{n_pdfs}</span><span class="stat-l">Scrolls</span></div>
      <div class="stat-c"><span class="stat-n">{n_texts}</span><span class="stat-l">Tablets</span></div>
      <div class="stat-c"><span class="stat-n">{n_chunks}</span><span class="stat-l">Vectors</span></div>
    </div>""", unsafe_allow_html=True)

    if st.button("𓂋  Reload Sacred Texts", use_container_width=True, key="reload_btn"):
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
        err_msg = {"no_library":"no lib","no_text":"no text / scanned"}.get(doc.get("error",""),"⚠")
        err_note = f' <span style="color:#e05050;font-size:.6rem;">⚠ {err_msg}</span>' if has_err else ""
        st.markdown(f"""
        <div class="{'file-pill pdf-pill' if is_pdf else 'file-pill'}">
          <div class="{'fp-dot pdf' if is_pdf else 'fp-dot'}"></div>
          <span class="fp-name">{doc['name']}{err_note}</span>
          <span class="fp-badge {badge}">{"SCROLL" if is_pdf else "TABLET"}</span>
          <span class="fp-size">{sz}{pages}</span>
        </div>""", unsafe_allow_html=True)

    if not st.session_state.kb_docs:
        st.markdown('<p style="font-size:.8rem;color:#555;padding:.4rem 1rem;font-style:italic;">The library is empty…</p>', unsafe_allow_html=True)

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
    st.markdown("""
    <div style="display:flex;align-items:center;gap:.65rem;padding:.5rem .75rem;">
      <div style="width:30px;height:30px;border-radius:3px;background:linear-gradient(135deg,#1a3a5c,#2a5a8c);
      border:1px solid rgba(200,146,42,.3);display:flex;align-items:center;justify-content:center;font-size:1rem;">𓀀</div>
      <span style="font-family:'Cinzel',serif;font-size:.72rem;color:#7a6840;letter-spacing:.08em;">THE SCRIBE</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="hieroglyph-border">𓀀 𓁐 𓂋 𓃭 𓄿 𓅓 𓆙 𓇌 𓈖 𓉐</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
sess = active_sess()
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

                    # ── Text answer (always shown) ────────────────────────
                    st.markdown(msg["content"])

                    # ── Audio player (assistant only) ─────────────────────
                    if msg["role"] == "assistant":
                        mp3_data = None
                        if msg.get("tts_key"):
                            mp3_data = st.session_state.tts_cache.get(msg["tts_key"])

                        if mp3_data:
                            st.caption("🔊 Oracle Voice")
                            st.audio(mp3_data, format="audio/mp3")
                        else:
                            # Show exactly why there is no audio
                            err = st.session_state.get("tts_last_err", "")
                            flags = st.session_state.get("tts_flags", {})
                            if flags:
                                flag_str = (
                                    f"elevenlabs={'✅' if flags.get('TTS_AVAILABLE') else '❌'}  "
                                    f"enabled={'✅' if flags.get('tts_enabled') else '❌'}"
                                )
                                st.caption(f"🔇 No audio — {flag_str}" + (f"  |  {err}" if err else ""))

                    # ── Source papyri ──────────────────────────────────────
                    if msg["role"] == "assistant" and st.session_state.show_chunks and msg.get("sources"):
                        with st.expander(f"𓏏 {len(msg['sources'])} source papyri"):
                            for c in msg["sources"]:
                                pct     = int(c.get("score", 0) * 100)
                                is_pdf  = c.get("type") == "pdf"
                                pg_info = f" · p.{c['page']}" if c.get("page") else ""
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
            parts.append(f'<img src="data:image/jpeg;base64,{b64img}">')
        if st.session_state.pending_audio:
            parts.append('<span class="rec-pill">● AUDIO READY</span>')
        c_prev, c_x = st.columns([20, 1])
        with c_prev:
            st.markdown(f'<div class="attach-strip"><div class="attach-inner">{"".join(parts)}<span style="margin-left:auto;font-size:.7rem;color:#7a6840;font-style:italic;">ready — type a question or send now</span></div></div>', unsafe_allow_html=True)
        with c_x:
            st.markdown("<div style='padding-top:6px'>", unsafe_allow_html=True)
            if st.button("✕", key="clr_attach", help="Clear attachments"):
                st.session_state.pending_image = None
                st.session_state.pending_audio = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Input bar ─────────────────────────────────────────────────────────────
    col_img, col_mic, col_chat = st.columns([1, 3, 14])

    with col_img:
        img_label = "🖼✔" if st.session_state.pending_image else "🖼"
        img_file = st.file_uploader(
            img_label, type=["png","jpg","jpeg","webp","gif"],
            key="img_up", label_visibility="visible",
        )
        if img_file is not None:
            new_bytes = img_file.read()
            if new_bytes != st.session_state.pending_image:
                st.session_state.pending_image = new_bytes
                st.rerun()

    with col_mic:
        try:
            audio_val = st.audio_input("🎙", key="mic_input", label_visibility="visible")
            if audio_val is not None:
                new_audio = audio_val.read()
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
                new_bytes = aud_file.read()
                if new_bytes != st.session_state.pending_audio:
                    st.session_state.pending_audio     = new_bytes
                    st.session_state.pending_audio_ext = aud_file.name.rsplit(".", 1)[-1].lower()
                    st.rerun()

    with col_chat:
        user_text = st.chat_input("Speak your question to the oracle… (any language)", key="oracle_chat")

    st.markdown("""
    <script>
    (function(){
      function tagBar(){
        var up=document.querySelectorAll('[data-testid="stFileUploader"]');
        if(!up.length){setTimeout(tagBar,200);return;}
        var node=up[0];
        for(var i=0;i<10;i++){
          node=node.parentElement;
          if(!node) break;
          if(node.getAttribute&&node.getAttribute('data-testid')==='stHorizontalBlock'){
            if(node.parentElement) node.parentElement.classList.add('kmt-bar');
            return;
          }
        }
        setTimeout(tagBar,200);
      }
      tagBar();
    })();
    </script>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PROCESS SEND
    # ══════════════════════════════════════════════════════════════════════════
    if user_text is not None:
        pending_img   = st.session_state.pending_image
        pending_audio = st.session_state.pending_audio
        audio_ext     = st.session_state.pending_audio_ext
        raw_text      = (user_text or "").strip()

        if not raw_text and not pending_img and not pending_audio:
            st.stop()

        # ── Audio transcription ──
        audio_label         = None
        classification_html = ""
        if pending_audio:
            with st.spinner("𓂀 Deciphering the voice offering…"):
                transcript = transcribe_audio(pending_audio, audio_ext)
            audio_label = f'Voice: "{transcript[:70]}{"…" if len(transcript)>70 else ""}"'
            raw_text    = (transcript + (" · " + raw_text if raw_text else "")).strip()

        # ── Image classification + vision ──
        image_answer        = None
        classification_html = ""
        if pending_img:
            if CLASSIFIER_AVAILABLE and os.path.exists(CLASSIFIER_WEIGHTS):
                with st.spinner("𓂀 The Eye of Horus classifies the image…"):
                    clf_result = classify_image(pending_img)
                if "predictions" in clf_result:
                    preds = clf_result["predictions"]
                    top   = preds[0]
                    badges = ""
                    for p in preds:
                        bar_w = int(p["confidence"])
                        badges += f"""
                        <div style="margin:3px 0;font-family:'JetBrains Mono',monospace;font-size:0.7rem;">
                          <span style="color:var(--gold2);min-width:220px;display:inline-block;">{p['label']}</span>
                          <span style="background:var(--sand3);border:1px solid var(--border2);border-radius:3px;display:inline-block;width:120px;height:10px;vertical-align:middle;">
                            <span style="display:block;width:{bar_w}%;height:100%;background:linear-gradient(90deg,var(--gold),var(--gold2));border-radius:3px;"></span>
                          </span>
                          <span style="color:var(--text3);margin-left:6px;">{p['confidence']}%</span>
                        </div>"""
                    classification_html = f"""
                    <div class="step-card" style="margin-bottom:6px;">
                      <span class="step-label">𓂀 SWIN-T CLASSIFICATION</span>
                      <span style="color:var(--teal);font-size:0.75rem;">▶ {top['label']} ({top['confidence']}%)</span>
                      {badges}
                    </div>"""
                    clf_context = f"[Image classified as: {top['label']} ({top['confidence']}% confidence)]"
                    raw_text = (clf_context + (" — " + raw_text if raw_text else "")).strip()
            vision_q = raw_text or "Describe this image in detail."
            with st.spinner("𓂀 The Eye of Horus examines the image…"):
                image_answer = describe_image(pending_img, vision_q)

        # ── NLP pipeline ──
        pipeline_info = {}
        query_for_rag = raw_text or "(Image submitted)"
        if raw_text:
            with st.spinner("𓂀 Processing through the pipeline…"):
                pipeline_info = full_text_pipeline(raw_text)
                query_for_rag = pipeline_info.get("query_for_rag", raw_text)

        # ── Session title ──
        if not sess["messages"]:
            if pending_img and not raw_text:
                sess["title"] = "Image query"
            elif pending_audio and not raw_text:
                sess["title"] = "Voice query"
            else:
                sess["title"] = raw_text[:40]

        # ── ChromaDB Retrieval ──
        retrieved = []
        if query_for_rag and query_for_rag not in ("(Image submitted)", "(Audio submitted)"):
            with st.spinner("𓂀 Searching the sacred vault…"):
                retrieved = retrieve(query_for_rag, top_k=5)

        sess["messages"].append({
            "role":                "user",
            "content":             raw_text or ("🖼 Image" if pending_img else "🎙 Voice"),
            "image_bytes":         pending_img,
            "audio_label":         audio_label,
            "pipeline":            pipeline_info,
            "classification_html": classification_html,
        })

        ctx = build_context(retrieved) if retrieved else "No relevant scrolls found in the library."

        system_prompt = """\
You are a professional Egyptian tour guide with deep knowledge of history and culture.
Use ONLY the context below to answer. If the answer is not in the context, say
"I don't have that information in my tour notes."

Context:
{context}

Tourist question: {question}

Answer as an immersive, storytelling tour guide using historical facts:"""

        msgs_llm = [{"role": "system", "content": system_prompt}]
        for m in sess["messages"][-8:]:
            msgs_llm.append({"role": m["role"], "content": m["content"]})

        combined = (f"[Image Analysis]\n{image_answer}\n\n" if image_answer else "")
        combined += f"Context:\n{ctx}"
        msgs_llm[-1]["content"] = f"{combined}\n\nQuestion: {raw_text or '(Describe the image as a tour guide)'}"

        with st.spinner("𓂀 The oracle consults the scrolls…"):
            answer = groq_chat(st.session_state.global_model, msgs_llm)

        # ── TTS: ElevenLabs → master → mix with Egyptian music ──────────
        # Diagnose availability first so user sees exact reason if silent
        st.session_state["tts_flags"] = {
            "TTS_AVAILABLE":  TTS_AVAILABLE,
            "tts_enabled":    st.session_state.get("tts_enabled", True),
        }

        tts_mp3 = None
        if TTS_AVAILABLE and st.session_state.get("tts_enabled", True):
            with st.spinner("𓂀 The oracle speaks…"):
                tts_mp3 = run_tts_pipeline(answer)
            # Store error BEFORE rerun so it survives
            st.session_state["tts_last_err"] = st.session_state.get("tts_err", "")
        else:
            # Record why we skipped
            reasons = []
            if not TTS_AVAILABLE:  reasons.append("elevenlabs not installed")
            if not st.session_state.get("tts_enabled", True): reasons.append("TTS toggled off")
            st.session_state["tts_last_err"] = " | ".join(reasons)

        # Cache MP3 bytes keyed by session+message-index (BEFORE rerun)
        msg_idx   = len(sess["messages"])
        cache_key = f"{st.session_state.active_id}:{msg_idx}"
        if tts_mp3:
            st.session_state.tts_cache[cache_key] = tts_mp3

        sess["messages"].append({
            "role":    "assistant",
            "content": answer,
            "sources": retrieved,
            "tts_key": cache_key if tts_mp3 else None,
        })

        st.session_state.pending_image = None
        st.session_state.pending_audio = None
        st.rerun()