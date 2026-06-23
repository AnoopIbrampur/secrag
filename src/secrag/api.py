"""FastAPI backend: POST a question, get a grounded answer + citations + latency.

    uvicorn secrag.api:app --reload    # from the src/ dir, or with PYTHONPATH=src

Serves the single-page frontend at / and the JSON API at /api/query.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import config, rag

app = FastAPI(title="secrag", description="RAG QA over SEC 10-K filings")

_FRONTEND = config.ROOT / "frontend" / "index.html"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)
    ticker: str | None = Field(default=None, description="Optional ticker filter, e.g. AAPL")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    latency_ms: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "has_api_key": bool(config.GEMINI_API_KEY)}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    where = {"ticker": req.ticker.upper()} if req.ticker else None
    try:
        result = rag.answer(req.question, top_k=req.top_k, where=where)
    except RuntimeError as e:  # missing API key
        raise HTTPException(status_code=503, detail=str(e))
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[asdict(s) for s in result.sources],
        latency_ms=result.latency_ms,
    )


@app.get("/")
def index_page() -> FileResponse:
    if not _FRONTEND.exists():
        raise HTTPException(status_code=404, detail="frontend not found")
    return FileResponse(_FRONTEND)
