"""Download 10-K filings from SEC EDGAR and extract clean text.

EDGAR is fully public — no API key, just a descriptive User-Agent header and a
polite request rate. Flow:

    ticker --> CIK            (company_tickers.json)
    CIK    --> filing list    (data.sec.gov submissions API)
    filing --> primary doc    (Archives/edgar/data/...) --> stripped text

Run directly to pull a few filings:

    python -m secrag.ingest AAPL MSFT NVDA
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from . import config

# Some filings are served as XBRL/XML; the HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_HIDDEN_STYLE = re.compile(r"display:\s*none", re.I)

_HEADERS = {"User-Agent": config.SEC_USER_AGENT}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _get(url: str) -> requests.Response:
    """GET with the required SEC header and a polite delay."""
    time.sleep(config.SEC_RATE_LIMIT_SLEEP)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


@dataclass
class FilingMeta:
    ticker: str
    cik: str            # zero-padded 10-digit
    company: str
    form: str           # "10-K"
    filing_date: str    # YYYY-MM-DD
    accession: str      # with dashes
    primary_doc: str    # filename of the main document
    url: str            # full URL to the primary document
    text_path: str = "" # local path once extracted
    n_chars: int = 0    # length of extracted text


def resolve_cik(ticker: str) -> tuple[str, str]:
    """Map a ticker symbol to (zero-padded CIK, company name)."""
    data = _get(_TICKERS_URL).json()
    ticker = ticker.upper()
    for row in data.values():
        if row["ticker"].upper() == ticker:
            return str(row["cik_str"]).zfill(10), row["title"]
    raise ValueError(f"Ticker {ticker!r} not found in EDGAR ticker map")


def list_10k_filings(cik: str, ticker: str, company: str, limit: int = 1) -> list[FilingMeta]:
    """Return the most recent `limit` 10-K filings for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    recent = _get(url).json()["filings"]["recent"]

    filings: list[FilingMeta] = []
    for form, date, accession, doc in zip(
        recent["form"],
        recent["filingDate"],
        recent["accessionNumber"],
        recent["primaryDocument"],
    ):
        if form != "10-K":
            continue
        acc_nodash = accession.replace("-", "")
        cik_int = int(cik)  # archive path uses the un-padded CIK
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
        )
        filings.append(
            FilingMeta(
                ticker=ticker.upper(),
                cik=cik,
                company=company,
                form=form,
                filing_date=date,
                accession=accession,
                primary_doc=doc,
                url=doc_url,
            )
        )
        if len(filings) >= limit:
            break
    return filings


def extract_text(html: str) -> str:
    """Strip an EDGAR HTML/iXBRL filing down to readable plain text.

    Modern 10-Ks are inline XBRL: the readable prose is interleaved with
    machine-readable tags. We remove the non-displayed XBRL context
    (``ix:hidden``, ``ix:header``) and ``display:none`` blocks — which would
    otherwise dump thousands of lines of tag soup into the text — but keep the
    inline ``ix:nonFraction`` tags that wrap displayed financial figures.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name in ("ix:hidden", "ix:header")):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse the heavy whitespace EDGAR filings are full of.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_filing(meta: FilingMeta) -> FilingMeta:
    """Download a filing's primary document, extract text, save to disk."""
    config.FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    html = _get(meta.url).text
    text = extract_text(html)

    stem = f"{meta.ticker}_{meta.filing_date}"
    text_path = config.FILINGS_DIR / f"{stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    meta.text_path = str(text_path)
    meta.n_chars = len(text)

    (config.FILINGS_DIR / f"{stem}.meta.json").write_text(
        json.dumps(asdict(meta), indent=2), encoding="utf-8"
    )
    return meta


def ingest_tickers(tickers: list[str], per_ticker: int = 1) -> list[FilingMeta]:
    """End-to-end: resolve, list, and download 10-Ks for each ticker."""
    results: list[FilingMeta] = []
    for ticker in tickers:
        cik, company = resolve_cik(ticker)
        filings = list_10k_filings(cik, ticker, company, limit=per_ticker)
        if not filings:
            print(f"  ! no 10-K found for {ticker}")
            continue
        for meta in filings:
            meta = fetch_filing(meta)
            print(f"  ✓ {meta.ticker} {meta.filing_date}  ({meta.n_chars:,} chars)  {meta.company}")
            results.append(meta)
    return results


if __name__ == "__main__":
    tickers = sys.argv[1:] or ["AAPL", "MSFT", "NVDA"]
    print(f"Ingesting 10-Ks for: {', '.join(t.upper() for t in tickers)}")
    ingest_tickers(tickers)
    print(f"\nDone. Filings saved to {config.FILINGS_DIR}")
