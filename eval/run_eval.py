"""Evaluation harness: retrieval quality, answer faithfulness/correctness, latency.

    PYTHONPATH=../src python run_eval.py --questions questions.jsonl --k 5

Metrics
-------
Retrieval (vs. gold_chunk_ids):
  hit@k        fraction of questions where a gold chunk appears in the top-k
  precision@k  mean (relevant retrieved / k)
  recall@k     mean (relevant retrieved / #gold)
  MRR          mean reciprocal rank of the first gold chunk
  soft hit@k   gold chunk OR same ticker+section in top-k (overlap-tolerant)

Answer (LLM-as-judge, grounded in the retrieved context):
  faithfulness  fraction of answers fully supported by retrieved context
  correctness   fraction of answers matching the gold answer

Latency:
  p50 / p95 for retrieval, generation, and total (ms)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secrag import config, index, llm, rag  # noqa: E402


# ---------- LLM judges (structured output) ----------

class Faithfulness(BaseModel):
    faithful: bool
    unsupported_claims: list[str]

class Correctness(BaseModel):
    correct: bool
    reason: str


_FAITH_SYSTEM = """You check whether an answer is fully supported by the provided \
source context. Return faithful=true only if every factual claim in the answer is \
directly supported by the context. List any claims that are not supported."""

_CORRECT_SYSTEM = """You grade whether a candidate answer matches the reference \
(gold) answer for a question about an SEC filing. Return correct=true if the \
candidate conveys the same key facts/figures as the gold answer, even if worded \
differently. Minor extra detail is fine; contradictions or missing key facts are not."""


def judge_faithfulness(question, answer, context, model=None) -> Faithfulness:
    return llm.generate_structured(
        _FAITH_SYSTEM,
        f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
        schema=Faithfulness,
        model=model,
    )


def judge_correctness(question, answer, gold, model=None) -> Correctness:
    return llm.generate_structured(
        _CORRECT_SYSTEM,
        f"Question: {question}\n\nGold answer: {gold}\n\nCandidate answer: {answer}",
        schema=Correctness,
        model=model,
    )


# ---------- retrieval metrics ----------

def retrieval_metrics(hits: list[dict], gold_ids: set[str], gold_ticker: str,
                      gold_section: str, k: int) -> dict:
    ret_ids = [h["id"] for h in hits[:k]]
    relevant = [i for i, cid in enumerate(ret_ids) if cid in gold_ids]
    # First rank (1-indexed) of a gold chunk, for MRR.
    rr = 1.0 / (relevant[0] + 1) if relevant else 0.0
    # Overlap-tolerant: a hit in the same filing section counts as soft-relevant.
    soft_hit = any(
        h["id"] in gold_ids
        or (h["metadata"]["ticker"] == gold_ticker
            and h["metadata"]["section"] == gold_section)
        for h in hits[:k]
    )
    return {
        "hit": 1.0 if relevant else 0.0,
        "precision": len(relevant) / k,
        "recall": len(relevant) / max(len(gold_ids), 1),
        "rr": rr,
        "soft_hit": 1.0 if soft_hit else 0.0,
    }


def pct(values: list[float], p: float) -> float:
    return round(float(np.percentile(values, p)), 1) if values else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions.jsonl")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N (0=all)")
    # Optional per-call-type models. Spreading answer + the two judges across
    # different models keeps each under the free tier's 20-requests/day/model cap.
    ap.add_argument("--answer-model", default=None)
    ap.add_argument("--faith-model", default=None)
    ap.add_argument("--correct-model", default=None)
    args = ap.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        raise SystemExit(f"{qpath} not found. Build it first with build_questions.py "
                         "(then verify the rows).")

    rows = [json.loads(line) for line in qpath.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    if not llm.has_key():
        raise SystemExit("GEMINI_API_KEY not set — needed for generation + the LLM judges.")

    # Warm the embedding model so the first question's latency isn't dominated by
    # the one-time model load (it loads once at server startup in production).
    index.query("warmup", top_k=1)

    per_q, faith_flags, correct_flags = [], [], []
    ret_acc = {"hit": [], "precision": [], "recall": [], "rr": [], "soft_hit": []}
    lat = {"retrieval_ms": [], "generation_ms": [], "total_ms": []}
    failed = 0

    for i, row in enumerate(rows, 1):
        try:
            result = rag.answer(row["question"], top_k=args.k, model=args.answer_model)
            hits = index.query(row["question"], top_k=args.k)  # same retrieval, ids included

            rm = retrieval_metrics(
                hits, set(row.get("gold_chunk_ids", [])),
                row.get("gold_ticker", ""), row.get("gold_section", ""), args.k,
            )
            context = "\n\n".join(f"[{s.n}] {s.text}" for s in result.sources)
            faith = judge_faithfulness(row["question"], result.answer, context, model=args.faith_model)
            correct = judge_correctness(row["question"], result.answer, row["gold_answer"],
                                        model=args.correct_model)
        except Exception as e:  # don't let one bad call (e.g. quota) sink the run
            failed += 1
            print(f"  [{i}/{len(rows)}] {row['id']}  SKIPPED: {repr(e)[:120]}")
            continue

        for key in ret_acc:
            ret_acc[key].append(rm[key])
        for key in lat:
            lat[key].append(result.latency_ms.get(key, 0.0))
        faith_flags.append(1.0 if faith.faithful else 0.0)
        correct_flags.append(1.0 if correct.correct else 0.0)

        per_q.append({
            "id": row["id"], "question": row["question"],
            "answer": result.answer, "retrieval": rm,
            "faithful": faith.faithful, "unsupported_claims": faith.unsupported_claims,
            "correct": correct.correct, "latency_ms": result.latency_ms,
        })
        print(f"  [{i}/{len(rows)}] {row['id']}  hit={rm['hit']:.0f} "
              f"faithful={faith.faithful} correct={correct.correct}")

    n = len(per_q)
    if n == 0:
        raise SystemExit("All questions failed — check the API key / quota.")
    summary = {
        "n_questions": n,
        "n_failed": failed,
        "k": args.k,
        "retrieval": {key: round(float(np.mean(vals)), 3) for key, vals in ret_acc.items()},
        "faithfulness": round(float(np.mean(faith_flags)), 3),
        "correctness": round(float(np.mean(correct_flags)), 3),
        "latency_ms": {
            "retrieval_p50": pct(lat["retrieval_ms"], 50), "retrieval_p95": pct(lat["retrieval_ms"], 95),
            "generation_p50": pct(lat["generation_ms"], 50), "generation_p95": pct(lat["generation_ms"], 95),
            "total_p50": pct(lat["total_ms"], 50), "total_p95": pct(lat["total_ms"], 95),
        },
    }

    Path(args.out).write_text(json.dumps({"summary": summary, "per_question": per_q}, indent=2))

    r = summary["retrieval"]
    print("\n" + "=" * 56)
    print(f"  EVAL SUMMARY  (n={n}, k={args.k})")
    print("=" * 56)
    print(f"  Retrieval   hit@k {r['hit']:.3f} | soft {r['soft_hit']:.3f} | "
          f"P@k {r['precision']:.3f} | R@k {r['recall']:.3f} | MRR {r['rr']:.3f}")
    print(f"  Answer      faithfulness {summary['faithfulness']:.3f} | "
          f"correctness {summary['correctness']:.3f}")
    L = summary["latency_ms"]
    print(f"  Latency     total p50 {L['total_p50']}ms / p95 {L['total_p95']}ms "
          f"(retrieval p50 {L['retrieval_p50']}ms)")
    print("=" * 56)
    print(f"  Full results -> {args.out}")


if __name__ == "__main__":
    main()
