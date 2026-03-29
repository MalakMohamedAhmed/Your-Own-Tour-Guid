"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PIPELINE 4 — Semantic Retrieval                                        ║
║                                                                          ║
║  What this file does:                                                    ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Takes the cleaned query from Pipeline 2 and finds the most relevant    ║
║  text chunks from the ChromaDB vector store built by Pipeline 3.        ║
║                                                                          ║
║  How cosine similarity search works here:                               ║
║    1. The query string is embedded with the same all-MiniLM-L6-v2 model ║
║       that was used to embed the document chunks at index time.          ║
║    2. ChromaDB computes cosine similarity between the query vector and   ║
║       every stored chunk vector (via its HNSW index — Hierarchical      ║
║       Navigable Small World graph — which makes this fast even with     ║
║       millions of vectors).                                              ║
║    3. The top-k closest chunks are returned with their similarity score. ║
║    4. Any chunk scoring below 10 % similarity is discarded to prevent   ║
║       irrelevant "filler" context from confusing the LLM.               ║
║                                                                          ║
║  Why the same embedding model matters:                                   ║
║    If you embedded documents with model A but query with model B, the    ║
║    vector spaces would be incompatible and similarity scores would be    ║
║    meaningless. Pipeline 3 and Pipeline 4 both call load_embed_model()  ║
║    from pipeline_3_indexing so they share the same cached instance.     ║
║                                                                          ║
║  Output contract (retrieve):                                             ║
║    Returns a list of dicts, each containing:                             ║
║      text     — the raw chunk text                                       ║
║      source   — filename it came from                                    ║
║      chunk_id — position within that file                                ║
║      page     — PDF page number, or None for plain text files            ║
║      type     — "pdf" or "text"                                          ║
║      score    — cosine similarity in [0.10, 1.0]                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from .pipeline_3_indexing import get_chroma_collection, load_embed_model


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed `query` and return the top-k most similar chunks from ChromaDB.

    Parameters
    ----------
    query : str
        The preprocessed English query from Pipeline 2
        (stopwords removed, lemmatised).
    top_k : int
        Maximum number of chunks to return. The actual number returned may
        be lower if fewer chunks clear the 10 % relevance threshold, or if
        the database contains fewer than top_k vectors in total.

    Returns
    -------
    list[dict]
        Sorted by descending similarity score. Empty list if the database
        is empty, ChromaDB is unavailable, or an exception occurs.

    Similarity threshold:
        score = 1.0 − cosine_distance.
        Chunks with score < 0.10 are dropped. This cutoff was chosen
        empirically: below 10 % the content is usually from a completely
        different topic and adds noise rather than context.
    """
    col = get_chroma_collection()
    if col is None or col.count() == 0:
        return []

    em    = load_embed_model()
    q_emb = em.encode([query], convert_to_numpy=True)[0].tolist()

    try:
        results = col.query(
            query_embeddings = [q_emb],
            n_results        = min(top_k, col.count()),
            include          = ["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
        # Convert to similarity: 1 − distance puts it in [−1, 1] but with
        # cosine space the range is effectively [0, 1] for real content.
        score = 1.0 - dist
        if score < 0.10:
            continue        # too dissimilar — discard

        retrieved.append({
            "text":     doc,
            "source":   meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", 0),
            "page":     meta.get("page") if meta.get("page", -1) != -1 else None,
            "type":     meta.get("type", "text"),
            "score":    score,
        })

    return retrieved


def build_context(retrieved: list[dict]) -> str:
    """
    Format the retrieved chunks into a single context block for the LLM prompt.

    Each chunk is labelled with its source file and page number so the LLM
    can (and should) cite its sources. Chunks are separated by a horizontal
    rule so boundaries are clear even in long contexts.

    Example output:
        [Source 1 – great_pyramid.pdf p.3]
        The Great Pyramid of Giza was built around 2560 BCE …

        ---

        [Source 2 – karnak.txt p.?]
        Karnak Temple complex covers more than 100 hectares …
    """
    return "\n\n---\n\n".join(
        f"[Source {i} – {c['source']} p.{c.get('page', '?')}]\n{c['text']}"
        for i, c in enumerate(retrieved, 1)
    )
