"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 2 — NLP Preprocessing                                         ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Transforms raw user text into a clean, normalised English query        ║
║  that gives the vector search engine the best possible signal.          ║
║                                                                          ║
║  Step-by-step inside full_text_pipeline():                              ║
║                                                                          ║
║    1. Language detection                                                 ║
║       xlm-roberta checks whether the text is English. It returns a      ║
║       language code ("en", "ar", "fr", "de", …) and a confidence score. ║
║                                                                          ║
║    2. Translation (if non-English)                                       ║
║       A Helsinki-NLP MarianMT model translates the text to English.     ║
║       Only the languages listed in TRANSLATION_MODELS are supported;    ║
║       anything else is passed through as-is.                            ║
║                                                                          ║
║    3. Stopword removal                                                   ║
║       Common words that add no meaning ("the", "is", "of") are dropped ║
║       using NLTK's English stopword list.                               ║
║                                                                          ║
║    4. Lemmatisation                                                      ║
║       Inflected forms are reduced to their dictionary base:             ║
║         "visiting" → "visit", "pyramids" → "pyramid"                   ║
║       NLTK's WordNetLemmatizer tries four POS tags (verb, adjective,    ║
║       noun, adverb) and picks the first result that looks plausible.    ║
║       An edit-distance guard prevents over-aggressive stemming.         ║
║                                                                          ║
║    5. Irregular-verb lookup                                              ║
║       A 60-entry hand-coded dictionary catches the common irregular     ║
║       verbs that the lemmatiser gets wrong ("went"→"go", "was"→"be").   ║
║                                                                          ║
║  Caching strategy:                                                       ║
║    Translation results are MD5-keyed inside st.session_state so the     ║
║    same phrase never hits the model twice in one session.               ║
║                                                                          ║
║  Output contract (full_text_pipeline):                                  ║
║    Returns a dict with keys:                                             ║
║      original, detected_lang, confidence, was_translated,               ║
║      english_text, cleaned_text, query_for_rag                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import re
import hashlib

import streamlit as st

# ── Optional heavy dependencies ───────────────────────────────────────────────
TRANSLATION_AVAILABLE = False
_lang_detector = None
_translators: dict = {}

try:
    from transformers import pipeline as hf_pipeline, MarianMTModel, MarianTokenizer
    TRANSLATION_AVAILABLE = True
except ImportError:
    pass

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

# ── Language → translation model mapping ──────────────────────────────────────
TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
}

# ── Irregular English verbs the lemmatiser gets wrong ─────────────────────────
# Maps surface form → base form so "went" becomes "go", etc.
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


# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="𓂀 Loading language oracle…")
def _load_lang_detector():
    """Load the XLM-RoBERTa language-detection classifier once per process."""
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
    """
    Lazily load the MarianMT tokeniser + model for `lang_code → English`.
    Results are cached in the module-level dict so each model loads once.
    Returns (tokeniser, model) or (None, None) if the language is unsupported.
    """
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


@st.cache_resource(show_spinner=False)
def _get_nltk_tools():
    """Return the NLTK stopword set and lemmatiser, or (None, None) if NLTK unavailable."""
    if not NLTK_AVAILABLE:
        return None, None
    return set(stopwords.words("english")), WordNetLemmatizer()


# ── Helper: edit distance guard ───────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """
    Classic DP Levenshtein distance.
    Used to reject lemmatisations that change the word beyond recognition
    (e.g. would prevent "bus" → "bu" if the model were that aggressive).
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
    return dp[n]


# ── Core helpers ──────────────────────────────────────────────────────────────

def safe_lemmatize(word: str) -> str:
    """
    Lemmatise a single word with three safeguards:
      1. Irregular-verb dictionary lookup (fastest, most accurate).
      2. Try all four POS tags and pick the first candidate.
      3. Edit-distance guard: reject any lemma that removes more than
         50 % of the original characters (prevents over-stemming).
    """
    _, lem = _get_nltk_tools()
    if not lem:
        return word
    w = word.lower()
    if w in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[w]
    candidates = []
    for pos in ("v", "a", "n", "r"):
        lemma = lem.lemmatize(w, pos=pos)
        if lemma != w and _edit_distance(w, lemma) > len(w) * 0.5:
            continue          # too aggressive — skip
        candidates.append(lemma)
    return candidates[0] if candidates else w


def detect_and_translate(text: str) -> dict:
    """
    Detect the language of `text` and, if non-English, translate it.

    Caches results in st.session_state['translation_cache'] keyed by
    the MD5 of the input text, so repeat queries are instant.

    Returns a dict with keys:
        original, detected_lang, confidence, was_translated, english_text
    """
    result = {
        "original":       text,
        "detected_lang":  "en",
        "confidence":     1.0,
        "was_translated": False,
        "english_text":   text,
    }
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in st.session_state.get("translation_cache", {}):
        return st.session_state["translation_cache"][cache_key]

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
                tokens     = tok(text, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512)
                translated = model.generate(**tokens)
                english    = tok.decode(translated[0], skip_special_tokens=True)
                result["was_translated"] = True
                result["english_text"]   = english
    except Exception:
        pass

    if "translation_cache" not in st.session_state:
        st.session_state["translation_cache"] = {}
    st.session_state["translation_cache"][cache_key] = result
    return result


def advanced_preprocess(text: str) -> str:
    """
    Clean and normalise an English string for vector search:
      1. Lowercase everything.
      2. Replace hyphens with spaces ("step-pyramid" → "step pyramid").
      3. Strip punctuation except apostrophes.
      4. Collapse multiple whitespace.
      5. Remove stopwords.
      6. Lemmatise remaining tokens.
      7. Preserve compound tokens (words containing "_") unchanged.
    """
    sw, _ = _get_nltk_tools()
    text = text.lower()
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not sw:
        return text
    return " ".join(
        word if "_" in word else safe_lemmatize(word)
        for word in text.split()
        if word not in sw
    )


def full_text_pipeline(raw_text: str) -> dict:
    """
    Master entry point for Pipeline 2.

    Runs the complete chain:
        raw_text → detect language → translate if needed
                 → remove stopwords → lemmatise
                 → return enriched dict for the rest of the app.

    The 'query_for_rag' key is what Pipeline 4 (retrieval) actually uses.
    It falls back to the English text if cleaning produces an empty string
    (e.g. the query was a single stopword like "the").
    """
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
