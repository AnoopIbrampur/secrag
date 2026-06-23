"""Build and query the Chroma vector index.

We supply our own (local) embeddings rather than letting Chroma download its
own model, so indexing and querying use the exact same bge model. Cosine space
matches the normalized BGE vectors.

Build the index:

    python -m secrag.index build

Query it:

    python -m secrag.index query "What are the main risk factors?"
"""

from __future__ import annotations

import sys

import chromadb

from . import config, embed
from .chunk import Chunk, chunk_all


def _client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(create: bool = False):
    client = _client()
    if create:
        # Start clean so rebuilds are deterministic.
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return client.get_collection(config.COLLECTION_NAME)


def build(chunks: list[Chunk] | None = None, batch_size: int = 256) -> int:
    """Embed all chunks and write them to a fresh Chroma collection."""
    if chunks is None:
        chunks = chunk_all()
    collection = get_collection(create=True)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
            embeddings=embed.embed_documents([c.text for c in batch]),
        )
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")
    return collection.count()


def query(text: str, top_k: int | None = None, where: dict | None = None) -> list[dict]:
    """Return the top-k chunks for a query as dicts with text + metadata + score."""
    top_k = top_k or config.TOP_K
    collection = get_collection()
    res = collection.query(
        query_embeddings=[embed.embed_query(text)],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({"id": cid, "text": doc, "metadata": meta, "score": 1.0 - dist})
    return hits


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        n = build()
        print(f"\nIndexed {n} chunks into '{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")
    elif cmd == "query":
        q = " ".join(sys.argv[2:]) or "What are the main risk factors?"
        print(f"Query: {q}\n")
        for h in query(q):
            m = h["metadata"]
            print(f"[{h['score']:.3f}] {m['ticker']} {m['filing_date']} — {m['section']}")
            print(f"    {h['text'][:200].strip()}\n")
    else:
        print(f"Unknown command {cmd!r}. Use 'build' or 'query'.")
