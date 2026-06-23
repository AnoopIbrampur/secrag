"""Local embeddings via sentence-transformers — no API key, no per-call cost.

Uses BAAI/bge-small-en-v1.5 by default. BGE models are trained to prepend a
short instruction to *queries* (but not to passages), and to compare with
cosine similarity on normalized vectors — both handled here.
"""

from __future__ import annotations

from functools import lru_cache

from . import config

# BGE's recommended retrieval instruction for the query side only.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so that modules which only need config/chunking don't pay
    # the (heavy) torch import cost.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for indexing (no instruction prefix)."""
    vecs = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 256
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query (with BGE's retrieval instruction)."""
    vec = _model().encode(
        _QUERY_INSTRUCTION + text, normalize_embeddings=True
    )
    return vec.tolist()


if __name__ == "__main__":
    import numpy as np

    docs = ["Net sales increased 5% year over year.", "The board declared a dividend."]
    q = "How did revenue change?"
    dvecs = np.array(embed_documents(docs))
    qvec = np.array(embed_query(q))
    sims = dvecs @ qvec
    print(f"Model: {config.EMBED_MODEL}  dim={dvecs.shape[1]}")
    for doc, sim in zip(docs, sims):
        print(f"  {sim:.3f}  {doc}")
