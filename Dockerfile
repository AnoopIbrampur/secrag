FROM python:3.12-slim

WORKDIR /app

# System deps for lxml etc. are wheels-only on slim; keep it lean.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container doesn't fetch it on first
# request (faster, predictable cold start, works in no-egress deploys).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY src/ ./src/
COPY frontend/ ./frontend/
# Ship the prebuilt Chroma index (rebuild locally with `python -m secrag.index build`).
COPY data/chroma/ ./data/chroma/

ENV PYTHONPATH=/app/src
EXPOSE 8000

# GEMINI_API_KEY is provided at runtime (-e GEMINI_API_KEY=...).
CMD ["uvicorn", "secrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
