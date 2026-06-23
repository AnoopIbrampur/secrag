"""Chunking tests: section detection, windowing, and citation metadata."""

from secrag import config
from secrag.chunk import _split_sections, _window, chunk_filing

_META = {
    "ticker": "TEST", "company": "Test Corp", "filing_date": "2025-01-01",
    "form": "10-K", "url": "http://example.com", "accession": "0000-00",
}


def test_split_sections_detects_items():
    text = (
        "cover page text\n"
        "Item 1. Business\nWe make things.\n"
        "Item 1A. Risk Factors\nThings could go wrong.\n"
        "Item 7. Management's Discussion\nResults were fine.\n"
    )
    sections = dict(_split_sections(text))
    labels = list(sections)
    assert any(l.startswith("Cover") for l in labels)
    assert any(l.startswith("Item 1A") for l in labels)
    assert any(l.startswith("Item 7") for l in labels)
    # Item 1A content stays under the 1A label, not Item 1.
    assert "go wrong" in next(v for k, v in sections.items() if k.startswith("Item 1A"))


def test_split_sections_falls_back_without_items():
    sections = _split_sections("just some prose with no item headers")
    assert len(sections) == 1
    assert sections[0][0] == "Document"


def test_window_respects_size_and_overlap():
    text = "x" * 2500
    chunks = _window(text, size=1000, overlap=150)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_filing_attaches_metadata():
    text = "Item 1. Business\n" + ("revenue grew. " * 200)
    chunks = chunk_filing(_META, text)
    assert chunks
    first = chunks[0]
    assert first.id.startswith("TEST_2025-01-01_")
    assert first.metadata["ticker"] == "TEST"
    assert first.metadata["section"].startswith("Item 1")
    assert first.metadata["url"] == "http://example.com"
