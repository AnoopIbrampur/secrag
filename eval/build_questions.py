"""Bootstrap a labeled eval set from indexed chunks (then HUMAN-VERIFY the output).

Hand-writing 75 grounded Q&A is slow; this drafts candidates by sampling
substantive chunks and asking the LLM to write a self-contained question + gold
answer for each. The sampled chunk becomes the gold retrieval target, so we can
measure retrieval precision/recall later.

    PYTHONPATH=../src python build_questions.py --n 75 --out questions.candidates.jsonl

IMPORTANT: the output is a *draft*. Review every row, fix bad questions, and
confirm the gold answer before renaming to questions.jsonl. The verification is
what makes the eval credible — don't skip it.
"""

from __future__ import annotations

import argparse
import json
import random

from pydantic import BaseModel

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secrag import config, llm  # noqa: E402
from secrag.chunk import chunk_all  # noqa: E402

# Sections worth asking about (skip cover pages, exhibit lists, boilerplate).
_GOOD_SECTIONS = ("Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8")

_GEN_SYSTEM = """You write evaluation questions for a SEC 10-K question-answering \
system. Given one excerpt from a filing, write ONE specific, self-contained \
question that can be answered from this excerpt alone, plus the correct answer.

Requirements:
- The question must name the company (so it's answerable without seeing the excerpt).
- The question must be answerable ONLY from the excerpt — prefer concrete facts, \
figures, or named risks.
- The gold answer must be short and directly supported by the excerpt.
- Rate difficulty: "easy" (direct lookup), "medium" (needs light synthesis), \
"hard" (specific detail or figure).
- If the excerpt is boilerplate with no answerable content, set question to "" \
(empty)."""


class GenQA(BaseModel):
    question: str
    gold_answer: str
    difficulty: str


def _eligible(chunks):
    out = [
        c for c in chunks
        if any(c.metadata["section"].startswith(s) for s in _GOOD_SECTIONS)
        and len(c.text) > 400
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--out", default="questions.candidates.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not llm.has_key():
        raise SystemExit("GEMINI_API_KEY not set — needed to draft questions.")

    random.seed(args.seed)
    chunks = _eligible(chunk_all())
    random.shuffle(chunks)

    out_path = Path(args.out)
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            if written >= args.n:
                break
            m = chunk.metadata
            try:
                qa = llm.generate_structured(
                    _GEN_SYSTEM,
                    f"{m['company']} ({m['ticker']}), {m['section']}:\n\n{chunk.text}",
                    schema=GenQA,
                    max_tokens=512,
                )
            except Exception as e:  # skip a bad row rather than abort the run
                print(f"  ! skipped {chunk.id}: {e}")
                continue

            if not qa.question.strip():
                continue

            row = {
                "id": f"q{written:03d}",
                "question": qa.question.strip(),
                "gold_answer": qa.gold_answer.strip(),
                "difficulty": qa.difficulty,
                "gold_ticker": m["ticker"],
                "gold_filing_date": m["filing_date"],
                "gold_section": m["section"],
                "gold_chunk_ids": [chunk.id],
                "verified": False,  # <-- flip to true after human review
            }
            f.write(json.dumps(row) + "\n")
            written += 1
            print(f"  [{written}/{args.n}] {m['ticker']} {qa.difficulty}: {qa.question[:70]}")

    print(f"\nDrafted {written} candidate questions -> {out_path}")
    print("NEXT: review each row, fix questions/answers, set verified=true, "
          "then rename to questions.jsonl")


if __name__ == "__main__":
    main()
