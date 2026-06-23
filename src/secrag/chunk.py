"""Split extracted 10-K text into retrieval chunks with citation metadata.

10-Ks are organized into numbered "Items" (Item 1 Business, Item 1A Risk
Factors, Item 7 MD&A, ...). We detect those headers so every chunk knows which
section it came from, then sliding-window within each section. The section label
plus company/date/URL travel with each chunk — that metadata is what powers
citations and the retrieval-precision metric in the eval harness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import config

# Matches "Item 1.", "Item 1A.", "Item 7A" etc. at the start of a line.
_ITEM_RE = re.compile(r"(?im)^\s*item\s+(\d{1,2}[A-Z]?)\.?\s*([^\n]{0,80})")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_label, section_text) on 10-K Item headers.

    Falls back to a single "full" section if no Item headers are found.
    """
    matches = list(_ITEM_RE.finditer(text))
    if not matches:
        return [("Document", text)]

    sections: list[tuple[str, str]] = []
    # Preamble before the first Item header (cover page, etc.).
    if matches[0].start() > 0:
        sections.append(("Cover", text[: matches[0].start()]))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item_num = m.group(1).upper()
        title = re.sub(r"\s+", " ", m.group(2)).strip(" .")
        label = f"Item {item_num}" + (f" — {title}" if title else "")
        sections.append((label, text[m.start() : end]))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Sliding character window, preferring to break on whitespace."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        end = start + size
        piece = text[start:end]
        # Try not to cut mid-word: back up to the last newline/space.
        if end < len(text):
            brk = max(piece.rfind("\n"), piece.rfind(". "))
            if brk > size // 2:
                piece = piece[: brk + 1]
                end = start + brk + 1
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        start = end if end > start else start + step
    return chunks


def chunk_filing(meta: dict, text: str) -> list[Chunk]:
    """Produce chunks for one filing, carrying citation metadata on each."""
    chunks: list[Chunk] = []
    idx = 0
    for label, section_text in _split_sections(text):
        for piece in _window(section_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunk_id = f"{meta['ticker']}_{meta['filing_date']}_{idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=piece,
                    metadata={
                        "ticker": meta["ticker"],
                        "company": meta["company"],
                        "filing_date": meta["filing_date"],
                        "form": meta["form"],
                        "section": label,
                        "url": meta["url"],
                        "accession": meta["accession"],
                    },
                )
            )
            idx += 1
    return chunks


def chunk_all() -> list[Chunk]:
    """Chunk every filing in the filings directory."""
    all_chunks: list[Chunk] = []
    for meta_path in sorted(config.FILINGS_DIR.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text = (config.FILINGS_DIR / f"{meta['ticker']}_{meta['filing_date']}.txt").read_text(
            encoding="utf-8"
        )
        filing_chunks = chunk_filing(meta, text)
        all_chunks.extend(filing_chunks)
        print(f"  {meta['ticker']} {meta['filing_date']}: {len(filing_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all()
    print(f"\nTotal: {len(chunks)} chunks from filings in {config.FILINGS_DIR}")
    if chunks:
        sample = chunks[len(chunks) // 2]
        print(f"\nSample chunk [{sample.id}] section={sample.metadata['section']!r}:")
        print(sample.text[:300])
