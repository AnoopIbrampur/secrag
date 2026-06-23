"""Retrieval-augmented generation: retrieve chunks, answer with an LLM + citations.

The answer is grounded strictly in retrieved 10-K text. Each source chunk is
numbered; the model is instructed to cite with inline [n] markers, and we return
the source list so the UI/eval can map [n] back to the exact filing + section.

    python -m secrag.rag "How much did Apple spend on R&D?"
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from . import config, index, llm

_SYSTEM = """You are a financial-research assistant answering questions about \
SEC 10-K filings. Answer ONLY using the numbered sources provided. Rules:

- Ground every claim in the sources. Cite them inline with [n] markers, e.g. \
"R&D expense rose to $31.4B [2]".
- If the sources do not contain the answer, say so plainly. Do not use outside \
knowledge or guess.
- Be concise and specific; prefer exact figures and quotes from the filings.
- When a figure could come from multiple companies or years, name the company \
and fiscal period."""


@dataclass
class Source:
    n: int
    ticker: str
    company: str
    filing_date: str
    section: str
    url: str
    score: float
    text: str


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)  # retrieval_ms, generation_ms, total_ms


def _format_context(hits: list[dict]) -> tuple[str, list[Source]]:
    blocks, sources = [], []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        blocks.append(
            f"[{i}] {m['company']} ({m['ticker']}), {m['form']} filed "
            f"{m['filing_date']}, {m['section']}:\n{h['text']}"
        )
        sources.append(
            Source(
                n=i,
                ticker=m["ticker"],
                company=m["company"],
                filing_date=m["filing_date"],
                section=m["section"],
                url=m["url"],
                score=round(h["score"], 4),
                text=h["text"],
            )
        )
    return "\n\n".join(blocks), sources


def answer(
    question: str,
    top_k: int | None = None,
    where: dict | None = None,
    model: str | None = None,
) -> RAGResult:
    """Retrieve, then generate a grounded, cited answer."""
    t0 = time.perf_counter()
    hits = index.query(question, top_k=top_k, where=where)
    t1 = time.perf_counter()

    if not hits:
        return RAGResult(
            question=question,
            answer="No indexed filings matched this question.",
            latency_ms={"retrieval_ms": round((t1 - t0) * 1000, 1),
                        "generation_ms": 0.0, "total_ms": round((t1 - t0) * 1000, 1)},
        )

    context, sources = _format_context(hits)
    user_msg = f"Sources:\n\n{context}\n\nQuestion: {question}"

    text = llm.generate(_SYSTEM, user_msg, max_tokens=1024, model=model)
    t2 = time.perf_counter()

    return RAGResult(
        question=question,
        answer=text,
        sources=sources,
        latency_ms={
            "retrieval_ms": round((t1 - t0) * 1000, 1),
            "generation_ms": round((t2 - t1) * 1000, 1),
            "total_ms": round((t2 - t0) * 1000, 1),
        },
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are Apple's main risk factors?"
    result = answer(q)
    print(f"Q: {result.question}\n")
    print(result.answer)
    print("\nSources:")
    for s in result.sources:
        print(f"  [{s.n}] {s.ticker} {s.filing_date} — {s.section}  (score {s.score})")
    print(f"\nLatency: {result.latency_ms}")
