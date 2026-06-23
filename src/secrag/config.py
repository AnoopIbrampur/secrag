"""Central configuration. Reads .env if present; everything has a sane default."""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
# Load .env from the project root explicitly, so it works no matter which
# directory the server/CLI is launched from.
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
FILINGS_DIR = DATA_DIR / "filings"
CHROMA_DIR = DATA_DIR / "chroma"
EVAL_DIR = ROOT / "eval"

# --- SEC EDGAR ---
# SEC mandates a descriptive User-Agent with contact info and ~10 req/s max.
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "secrag-research example@example.com")
SEC_RATE_LIMIT_SLEEP = 0.15  # seconds between requests (well under 10/s)

# --- Embeddings (local) ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking ---
CHUNK_SIZE = 1000        # target characters per chunk
CHUNK_OVERLAP = 150      # character overlap between adjacent chunks

# --- Retrieval ---
TOP_K = 5                # chunks retrieved per query

# --- Generation (Google Gemini, free tier) ---
def _real_key(name: str) -> str | None:
    """Return the env var unless it's empty or an obvious placeholder."""
    val = (os.getenv(name) or "").strip()
    if not val or val.startswith("your-") or val == "sk-ant-...":
        return None
    return val


GEMINI_API_KEY = _real_key("GEMINI_API_KEY")
# flash-lite is the free-tier-friendly default (generous quota); override with
# gemini-2.5-flash for higher answer quality when you have the quota.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-2.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-flash-lite")

# --- Chroma ---
COLLECTION_NAME = "sec_10k"
