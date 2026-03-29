"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 3 — Document Indexing                                         ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Reads every .txt / .md / .pdf file in the app folder, splits each one  ║
║  into overlapping text chunks, embeds those chunks as 384-dimensional   ║
║  vectors, and stores everything in ChromaDB on disk.                    ║
║                                                                          ║
║  Incremental indexing (the fingerprint system):                         ║
║    Each file gets a SHA-256 fingerprint derived from its path,          ║
║    modification time, and byte size. On startup the app reads all       ║
║    fingerprints already stored in ChromaDB. A file is only re-indexed   ║
║    if its fingerprint has changed — so adding one new PDF doesn't       ║
║    rebuild the entire database.                                         ║
║                                                                          ║
║  Chunking strategy:                                                      ║
║    400-word window with 50-word overlap. The overlap means that a       ║
║    sentence straddling a chunk boundary will appear in both chunks,     ║
║    so a retrieval query can always find its full context.               ║
║                                                                          ║
║  PDF handling:                                                           ║
║    Tries pdfplumber first (better layout parsing), falls back to pypdf. ║
║    Scanned-only PDFs (no embedded text) are noted in kb_docs with       ║
║    error="no_text" so the sidebar can display a warning.                ║
║                                                                          ║
║  Embedding model:                                                        ║
║    all-MiniLM-L6-v2 — a fast, lightweight 384-d sentence transformer.   ║
║    It is loaded once via @st.cache_resource and reused everywhere.      ║
║                                                                          ║
║  ChromaDB collection settings:                                           ║
║    Cosine similarity space, persistent storage in .kemet_chroma_db/.    ║
║    Upsert is used (not insert) so re-indexing a changed file is safe.   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import glob
import hashlib

import streamlit as st
from sentence_transformers import SentenceTransformer

# ── Optional PDF backends ─────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_BACKEND = "pdfplumber"
except ImportError:
    try:
        import pypdf
        PDF_BACKEND = "pypdf"
    except ImportError:
        PDF_BACKEND = None

# ── Optional ChromaDB ─────────────────────────────────────────────────────────
CHROMA_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_FOLDER     = os.path.join(BASE_DIR, "data")
CHROMA_DIR      = os.path.join(BASE_DIR, ".kemet_chroma_db")
COLLECTION_NAME = "kemet_scrolls"


# ── Model loaders (cached so they load once per Streamlit process) ────────────

@st.cache_resource(show_spinner="𓂀 Awakening the scribe…")
def load_embed_model() -> SentenceTransformer:
    """
    Load all-MiniLM-L6-v2 from HuggingFace (or local cache).
    This model maps any sentence to a 384-dimensional float vector.
    The same model instance is used for both indexing (here) and
    retrieval (Pipeline 4) so embeddings are always comparable.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="𓂀 Opening the sacred vault…")
def get_chroma_collection():
    """
    Open (or create) the persistent ChromaDB collection.
    The collection uses cosine similarity so scores are normalised to [0, 1].
    Returns None if ChromaDB is not installed.
    """
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


# ── Utility functions ─────────────────────────────────────────────────────────

def chroma_count() -> int:
    """Return the number of vectors currently stored in ChromaDB."""
    col = get_chroma_collection()
    if col is None:
        return 0
    try:
        return col.count()
    except Exception:
        return 0


def chroma_clear():
    """
    Delete and recreate the ChromaDB collection, effectively wiping all vectors.
    Called when the user clicks "Reload Sacred Texts" in the sidebar.
    Also clears the Streamlit cache so the collection object is re-fetched.
    """
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
    get_chroma_collection.clear()   # invalidate Streamlit cache


def _file_fingerprint(fpath: str) -> str:
    """
    Produce a SHA-256 hash from a file's path + mtime + size.
    This uniquely identifies a specific version of a file without reading it.
    If any of the three values change, the fingerprint changes.
    """
    s = os.stat(fpath)
    return hashlib.sha256(f"{fpath}|{s.st_mtime}|{s.st_size}".encode()).hexdigest()


def _indexed_fingerprints() -> dict:
    """
    Read the 'fingerprint' metadata field from every vector already in ChromaDB.
    Returns {filename: fingerprint_hash} so scan_and_index() can compare
    against the current files on disk without re-reading any content.
    """
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


# ── Text processing ───────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """
    Split `text` into word-count windows of `size` words with `overlap` words
    of overlap between adjacent chunks.

    Example with size=5, overlap=2:
        "a b c d e f g h"  →  ["a b c d e", "d e f g h"]

    Why 400 / 50?
        400 words ≈ one or two paragraphs — enough context for the LLM.
        50-word overlap ensures sentence-spanning context isn't lost at seams.
    """
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def extract_pdf_text(fpath: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF using pdfplumber (preferred) or pypdf.
    Returns a list of {"page": int, "text": str} dicts for non-empty pages.
    Empty pages (cover images, blank separators) are silently skipped.

    pdfplumber is preferred because it preserves column layout better.
    pypdf's layout mode is tried first; it falls back to basic extraction
    if the newer API is not available in the installed version.
    """
    pages = []
    try:
        if PDF_BACKEND == "pdfplumber":
            with pdfplumber.open(fpath) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    try:
                        text = (page.extract_text() or "").strip()
                    except Exception:
                        text = ""
                    if text:
                        pages.append({"page": i, "text": text})

        elif PDF_BACKEND == "pypdf":
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


def _upsert_chunks(col, ids, embeddings, documents, metadatas, batch: int = 100):
    """
    Batch-upsert vectors into ChromaDB in groups of `batch`.
    Upsert (not insert) means re-running on the same IDs is safe —
    it overwrites rather than duplicates.
    Batching avoids exceeding ChromaDB's per-request size limit.
    """
    for b in range(0, len(ids), batch):
        col.upsert(
            ids        = ids[b : b + batch],
            embeddings = embeddings[b : b + batch],
            documents  = documents[b : b + batch],
            metadatas  = metadatas[b : b + batch],
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def scan_and_index() -> int:
    """
    Scan DOCS_FOLDER for .txt / .md / .pdf files and index any that are
    new or changed since the last run.

    Algorithm:
        1. Collect all eligible files (excluding this script itself).
        2. Load fingerprints already stored in ChromaDB.
        3. For each file:
           - If fingerprint matches → skip (already current).
           - If fingerprint differs → delete old vectors, re-index.
           - If not in ChromaDB at all → index fresh.
        4. Embed each chunk and upsert into ChromaDB with metadata.

    Metadata stored per chunk:
        source      — filename  (used for source citations)
        page        — page number for PDFs, -1 for plain text
        chunk_id    — position of this chunk within its source
        type        — "pdf" or "text"
        fingerprint — used to detect file changes on next startup

    Returns the number of files actually (re-)indexed this run.
    Updates st.session_state.kb_docs so the sidebar can list them.
    """
    if not CHROMA_AVAILABLE:
        st.error("𓂀 ChromaDB not installed. Run: **pip install chromadb**")
        return 0

    # ── Collect files ──────────────────────────────────────────────────────
    all_files = sorted(set(
        f
        for pat in [os.path.join(DOCS_FOLDER, p) for p in ("*.txt", "*.md", "*.pdf")]
        for f in glob.glob(pat)
    ))
    self_name = os.path.basename(os.path.abspath(__file__))
    all_files = [
        f for f in all_files
        if os.path.basename(f) != self_name and not os.path.basename(f).startswith(".")
    ]

    indexed_fps  = _indexed_fingerprints()
    loaded_names = {d["name"] for d in st.session_state.get("kb_docs", [])}
    em           = load_embed_model()
    col          = get_chroma_collection()
    new_count    = 0

    for fpath in all_files:
        fname = os.path.basename(fpath)
        fp    = _file_fingerprint(fpath)
        ext   = os.path.splitext(fname)[1].lower()

        # ── Already indexed and unchanged ─────────────────────────────────
        if fname in indexed_fps and indexed_fps[fname] == fp:
            if fname not in loaded_names:
                st.session_state["kb_docs"].append({
                    "name": fname, "path": fpath,
                    "size": os.path.getsize(fpath),
                    "type": "pdf" if ext == ".pdf" else "text",
                })
            continue

        # ── File changed — remove old vectors first ────────────────────────
        if fname in indexed_fps:
            try:
                existing = col.get(where={"source": fname})
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
            except Exception:
                pass
            st.session_state["kb_docs"] = [
                d for d in st.session_state["kb_docs"] if d["name"] != fname
            ]

        # ── Index PDF ─────────────────────────────────────────────────────
        if ext == ".pdf":
            if PDF_BACKEND is None:
                st.session_state["kb_docs"].append({
                    "name": fname, "path": fpath,
                    "size": os.path.getsize(fpath), "type": "pdf", "error": "no_library"
                })
                continue

            pages = extract_pdf_text(fpath)
            if not pages:
                st.session_state["kb_docs"].append({
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
                        "source":      fname,
                        "page":        pg["page"],
                        "chunk_id":    idx,
                        "type":        "pdf",
                        "fingerprint": fp,
                    })

            if ids:
                _upsert_chunks(col, ids, embeddings, documents, metadatas)

            st.session_state["kb_docs"].append({
                "name": fname, "path": fpath,
                "size": os.path.getsize(fpath),
                "type": "pdf", "pages": len(pages),
            })

        # ── Index plain text / markdown ────────────────────────────────────
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
                    "source":      fname,
                    "page":        -1,
                    "chunk_id":    idx,
                    "type":        "text",
                    "fingerprint": fp,
                })
            _upsert_chunks(col, ids, embeddings, chunks, metadatas)
            st.session_state["kb_docs"].append({
                "name": fname, "path": fpath,
                "size": len(raw), "type": "text",
            })

        new_count += 1

    return new_count
