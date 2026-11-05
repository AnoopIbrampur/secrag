import json

import numpy as np

from secrag.site import export, load_eval, quantize


class FakeCollection:
    def __init__(self, n=4, dim=8):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(n, dim)).astype(np.float32)
        self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.ids = [f"NVDA_2026-02-25_{i}" for i in range(n)]

    def get(self, include):
        meta = {"ticker": "NVDA", "company": "NVIDIA CORP", "filing_date": "2026-02-25", "form": "10-K",
                "url": "https://www.sec.gov/x.htm", "section": "Item 1A — Risk Factors"}
        # reversed, to check the export sorts by id
        return {"ids": self.ids[::-1], "embeddings": self.vectors[::-1].tolist(),
                "documents": [f"text {i}" for i in range(len(self.ids))][::-1], "metadatas": [meta] * len(self.ids)}


def test_quantize_keeps_ranking():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q, scale = quantize(v)
    assert q.dtype == np.int8 and np.abs(q).max() == 127
    query = v[3]
    assert np.argmax(v @ query) == np.argmax(q.astype(np.float32) @ query) == 3
    assert np.allclose(q.astype(np.float32) * scale / 127, v, atol=scale / 127)


def test_export_writes_vectors_chunks_and_meta(tmp_path):
    col = FakeCollection()
    r = export(tmp_path, collection=col)
    assert r == {"chunks": 4, "filings": 1, "dim": 8}
    q = np.frombuffer((tmp_path / "vectors.bin").read_bytes(), dtype=np.int8).reshape(4, 8)
    chunks = json.loads((tmp_path / "chunks.json").read_text())
    assert [c["id"] for c in chunks] == sorted(col.ids)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["count"] == 4 and meta["filings"][0]["ticker"] == "NVDA"
    first = np.asarray(col.vectors[0])  # id ..._0 sorts first
    assert np.argmax(q.astype(np.float32) @ first) == 0


def test_load_eval_joins_gold(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"summary": {"n_questions": 1}, "per_question": [
        {"id": "q1", "question": "Q?", "answer": "A [1]", "retrieval": {"hit": 1.0, "soft_hit": 1.0},
         "faithful": True, "correct": True}]}))
    (tmp_path / "questions.jsonl").write_text(json.dumps(
        {"id": "q1", "gold_answer": "A", "gold_ticker": "NVDA", "gold_section": "Item 7", "gold_chunk_ids": ["c1"]}) + "\n")
    ev = load_eval(tmp_path)
    assert ev["summary"]["n_questions"] == 1
    q = ev["questions"][0]
    assert q["gold_chunk_ids"] == ["c1"] and q["gold_answer"] == "A" and q["correct"] is True
    assert load_eval(tmp_path / "missing") is None
