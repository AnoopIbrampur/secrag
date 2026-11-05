"""Export the index to a static, in-browser search demo (site/), for GitHub Pages.

    python -m secrag.site          # needs a built index (python -m secrag.index build)

Writes every chunk's text and citation metadata to site/chunks.json and its embedding to
site/vectors.bin as int8 (one global scale; the vectors are already normalized, so
ranking by dot product is unchanged in practice). The page embeds the query in the
browser with the same model (bge-small-en-v1.5 via transformers.js), so retrieval needs
no server and no key. Answer generation calls Gemini, so the page shows the answers from
the latest eval run instead of generating new ones.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config

SITE_DIR = config.ROOT / "site"


def quantize(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.abs(vectors).max()) or 1.0
    return np.round(vectors * (127.0 / scale)).astype(np.int8), scale


def load_eval(eval_dir: Path = config.EVAL_DIR) -> dict | None:
    results = eval_dir / "results.json"
    if not results.exists():
        return None
    res = json.loads(results.read_text())
    gold = {}
    qfile = eval_dir / "questions.jsonl"
    if qfile.exists():
        for line in qfile.read_text().splitlines():
            if line.strip():
                q = json.loads(line)
                gold[q["id"]] = q
    questions = []
    for p in res.get("per_question", []):
        g = gold.get(p["id"], {})
        questions.append({
            "id": p["id"], "question": p["question"], "answer": p.get("answer"),
            "gold_answer": g.get("gold_answer"), "gold_ticker": g.get("gold_ticker"),
            "gold_section": g.get("gold_section"), "gold_chunk_ids": g.get("gold_chunk_ids", []),
            "retrieval": p.get("retrieval", {}), "faithful": p.get("faithful"), "correct": p.get("correct"),
            "unsupported_claims": p.get("unsupported_claims", []),
        })
    return {"summary": res.get("summary", {}), "questions": questions}


def export(out: Path = SITE_DIR, collection=None) -> dict:
    if collection is None:
        from .index import get_collection  # imported here so the helpers don't need chromadb
        collection = get_collection()
    col = collection
    got = col.get(include=["embeddings", "documents", "metadatas"])
    order = sorted(range(len(got["ids"])), key=lambda i: got["ids"][i])
    vectors = np.asarray([got["embeddings"][i] for i in order], dtype=np.float32)
    q, scale = quantize(vectors)

    filings: dict[str, dict] = {}
    chunks = []
    for i in order:
        m = got["metadatas"][i]
        key = f"{m['ticker']}_{m['filing_date']}"
        filings.setdefault(key, {"ticker": m["ticker"], "company": m["company"], "filing_date": m["filing_date"],
                                 "form": m.get("form", "10-K"), "url": m["url"]})
        chunks.append({"id": got["ids"][i], "f": key, "s": m["section"], "t": got["documents"][i]})

    out.mkdir(parents=True, exist_ok=True)
    (out / "vectors.bin").write_bytes(q.tobytes())
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.EMBED_MODEL, "dim": int(vectors.shape[1]), "count": len(chunks), "scale": scale,
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "filings": sorted(filings.values(), key=lambda f: f["ticker"]),
        "eval": load_eval(),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, separators=(",", ":")))
    return {"chunks": len(chunks), "filings": len(filings), "dim": meta["dim"]}


if __name__ == "__main__":
    r = export()
    print(f"site/: {r['chunks']} chunks from {r['filings']} filings, dim {r['dim']}")
