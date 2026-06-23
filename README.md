# secrag

[![tests](https://github.com/AnoopIbrampur/secrag/actions/workflows/tests.yml/badge.svg)](https://github.com/AnoopIbrampur/secrag/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A question-answering system over real SEC 10-K filings that cites its sources and
ships with an evaluation harness I can defend line by line.

Most RAG demos stop at "look, it answers questions." The part that matters in
production is whether retrieval finds the right passage and whether the answer is
grounded in it, so this repo measures both. Ask a
question, get an answer with inline `[n]` citations that link back to the exact
filing and section, and run the eval harness to see retrieval precision, answer
faithfulness, and latency on a labeled question set.

![Evaluation results](docs/eval.png)

## Results

Evaluated on labeled questions, each tagged with the filing chunk that contains
the answer, so retrieval can be scored against ground truth. Numbers below are a
free-tier-limited sample (see the note under the table) but every one is
reproducible from `eval/run_eval.py`.

| Metric | Value | Reading |
|---|---|---|
| `hit@5` (exact gold chunk) | 0.33 | the exact labeled chunk lands in the top 5 |
| `soft hit@5` (gold section) | **0.83** | the right filing + section lands in the top 5 |
| `faithfulness` | **1.00** | answers stay grounded in the retrieved text |
| `correctness` | 0.33 | answer matches the gold answer |
| retrieval latency (p50) | **90 ms** | local embeddings, no API round-trip |
| total latency (p50 / p95) | 1.9 s / 3.0 s | generation dominates |

The gap between `hit@5` (0.33) and `soft hit@5` (0.83) is the interesting part.
The retriever reliably finds the right section of the right filing; it just
often returns a neighboring chunk rather than the one exact chunk a question was
written from, because adjacent chunks in a section overlap and look almost
identical to the embedder. So `soft hit@5` is the fairer read of retrieval, and
the clearest lever for lifting answer correctness is chunk granularity (or hybrid
lexical and dense retrieval) rather than more prompt tuning. Faithfulness sitting
at 1.00 says the grounding prompt is doing its job: the model answers from the
retrieved text instead of its own memory.

> **On sample size.** The eval ran on n=6 because Google's Gemini free tier
> caps usage at 20 requests/day/model, and a full pass needs three calls per
> question (answer + two judges). The harness itself runs at any N: point it at
> a larger labeled set, or spread the answer/faithfulness/correctness calls
> across separate models, and it scales. The methodology is the deliverable, and
> the numbers refresh with one command.

## How it works

```
SEC EDGAR API ─▶ ingest ─▶ section-aware ─▶ local embeddings ─▶ Chroma
 (public, no key)  (iXBRL    chunking         (bge-small,         (vector
                    strip)   (10-K Items)      no API key)         index)
                                                                     │
                                               query ────────────────┤
                                                 │                    ▼
                                           local embed ─▶ top-k chunks ─▶ Gemini ─▶ answer
                                                                           (grounded   + [n]
                                                                            + cited)   citations
```

A few decisions worth calling out:

- **Embeddings run locally** (`sentence-transformers`, `bge-small-en-v1.5`), so
  retrieval costs nothing per query and needs no key. Generation is the only
  hosted piece, behind `secrag/llm.py`, so swapping the provider is a one-file
  change.
- **iXBRL-aware extraction.** Modern 10-Ks interleave readable prose with
  machine-readable XBRL tags. A naive HTML strip dumps thousands of lines of tag
  soup into the text; the extractor drops the hidden XBRL context but keeps the
  inline tags that wrap reported figures, so dollar amounts survive. This is
  covered by a unit test.
- **Section-aware chunking.** Chunks carry the 10-K Item they came from (Item 1A
  Risk Factors, Item 7 MD&A, ...), which is what makes citations specific and
  retrieval scoreable.

## Stack

| Layer | Choice |
|---|---|
| Corpus | SEC 10-K via the public EDGAR API |
| Parsing | BeautifulSoup, inline-XBRL aware |
| Embeddings | `sentence-transformers` (bge-small, local) |
| Vector store | Chroma (persistent, cosine) |
| Generation | Google Gemini (`gemini-2.5-flash-lite`, free tier) |
| API | FastAPI + Uvicorn |
| Frontend | single-page HTML/JS |
| Eval judge | Gemini (LLM-as-judge) |
| Packaging | Docker / docker-compose |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add a free GEMINI_API_KEY (aistudio.google.com/apikey)
export PYTHONPATH=src

python -m secrag.ingest AAPL MSFT NVDA AMZN JPM XOM   # download 10-Ks (no key)
python -m secrag.index build                          # chunk + embed + index (no key)
python -m secrag.index query "What are NVIDIA's competitive risks?"   # retrieval only, no key

uvicorn secrag.api:app --reload    # http://localhost:8000
# or
docker compose up --build
```

Only `GEMINI_API_KEY` is required, and only for answer generation and the eval
judge. Ingestion, chunking, embeddings, and retrieval all run with no key.

## Evaluation

```bash
cd eval
# 1. Draft candidate Q&A from indexed chunks (then review them):
PYTHONPATH=../src python build_questions.py --n 30
#    review questions.candidates.jsonl, fix/drop rows, set verified=true,
#    rename to questions.jsonl
# 2. Score retrieval, faithfulness, correctness, and latency:
PYTHONPATH=../src python run_eval.py --questions questions.jsonl --k 5
# 3. Redraw the figure:
PYTHONPATH=../src python plot_results.py
```

**Metrics.** Retrieval is scored against the gold chunk each question was written
from: `hit@k`, `precision@k`, `recall@k`, `MRR`, plus `soft hit@k` (gold chunk
*or* same filing+section, which tolerates the chunk-overlap effect above). Answer
quality uses an LLM judge: `faithfulness` (is every claim supported by the
retrieved context?) and `correctness` (does the answer match the gold answer?).
Latency is reported as p50/p95, split into retrieval and generation.

The labeled questions are bootstrapped: Gemini drafts a question and gold answer
from a sampled chunk, then they get reviewed and the weak ones dropped. That is
why each question already knows its gold chunk for retrieval scoring.

## Honest limitations

- **Small, skewed sample.** The committed eval set is modest and over-weights
  JPMorgan, because random chunk sampling follows corpus size and JPM's 10-K is
  by far the largest. A production run wants a balanced, larger set.
- **LLM-as-judge is a proxy.** Faithfulness and correctness come from a model,
  not a human. `results.json` keeps the per-question `unsupported_claims` so the
  judgments can be spot-checked.
- **Demo corpus.** Six filings here; scaling to thousands is more tickers through
  `ingest` (respecting EDGAR's ~10 req/s limit).

## License

MIT, see [LICENSE](LICENSE).
