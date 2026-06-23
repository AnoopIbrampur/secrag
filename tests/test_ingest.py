"""Extraction tests — the part most likely to silently regress on real filings."""

from secrag.ingest import extract_text

# Minimal inline-XBRL snippet: a hidden context block (must be dropped) plus a
# displayed financial figure wrapped in ix:nonFraction (must be kept).
_IXBRL = """
<html>
  <body>
    <ix:header><ix:hidden>
      us-gaap:Revenue 0000320193 2025-09-27 contextRef-junk
    </ix:hidden></ix:header>
    <div style="display:none">SHOULD_NOT_APPEAR boilerplate</div>
    <p>Total net sales were
       <ix:nonfraction>416,161</ix:nonfraction> million.</p>
    <script>var x = 1;</script>
  </body>
</html>
"""


def test_extract_drops_hidden_xbrl_and_keeps_figures():
    text = extract_text(_IXBRL)
    assert "416,161" in text           # displayed figure survives
    assert "Total net sales" in text   # prose survives
    assert "contextRef-junk" not in text   # hidden XBRL context removed
    assert "SHOULD_NOT_APPEAR" not in text  # display:none removed
    assert "var x" not in text             # script removed


def test_extract_collapses_whitespace():
    text = extract_text("<p>a   b\n\n\n\n\nc</p>")
    assert "a b" in text          # runs of spaces collapsed to one
    assert "\n\n\n" not in text   # runs of blank lines collapsed
