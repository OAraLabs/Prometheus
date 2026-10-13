"""PDF text extraction without PyMuPDF (AGPL): pypdfium2, plus the form fields PDFium's text layer leaves out.

The comparison behind the swap, on ten real PDFs: docs/audits/pdf-extraction/. What it found, pinned here:

* the text must come out with PyMuPDF absent (the Mac app ships without it; a placeholder that tells an app
  user to "install PyMuPDF" is advice they cannot take);
* PDFium's text layer never contains what was typed into a form, so the extractor reads the filled fields
  and appends them, or a filled form loses exactly the part someone wrote;
* PDFium marks a hyphen it removed at a line break with U+0002 and joins the halves; the model must never
  see that control character mid-word;
* PDFium is not thread-safe, and uploads arrive concurrently.

The fixtures were made once with PyMuPDF 1.27.2.3 (PDFium cannot create form fields) and are committed:
tests/fixtures/pdf/two_pages.pdf (two pages of plain text) and filled_form.pdf (three filled text fields,
one empty field, one unchecked box).
"""

from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path

import pytest

from prometheus.utils import file_extract

FIXTURES = Path(__file__).parent / "fixtures" / "pdf"


@pytest.fixture
def no_pymupdf(monkeypatch):
    """The Mac app's environment: importing fitz (or pymupdf) fails."""
    monkeypatch.setitem(sys.modules, "fitz", None)
    monkeypatch.setitem(sys.modules, "pymupdf", None)


def _copy(name: str, tmp_path: Path) -> str:
    dest = tmp_path / name
    shutil.copy(FIXTURES / name, dest)
    return str(dest)


def test_text_is_extracted_without_pymupdf(no_pymupdf, tmp_path):
    text = file_extract.extract_text(_copy("two_pages.pdf", tmp_path))
    assert text is not None
    assert "install PyMuPDF" not in text
    assert "Prometheus reads this first page." in text
    assert "The second page says 1,234 and Zurich." in text
    assert text.index("first page") < text.index("second page"), "pages out of order"


def test_filled_form_values_are_in_the_text(no_pymupdf, tmp_path):
    text = file_extract.extract_text(_copy("filled_form.pdf", tmp_path))
    assert "Application form" in text, "the page's own text must still come first"
    assert "[Form fields]" in text
    for line in ("full_name: Ada Lovelace", "city: Zurich", "account: 12-3456"):
        assert line in text, f"{line!r} missing: a filled form must not lose what was typed"
    assert "empty_field" not in text, "an empty field is noise, not content"
    assert "agree" not in text, "an unchecked box ('Off') is noise, not content"


def test_pdfium_line_break_marker_never_reaches_the_model():
    raw = "residual func-\r\ntions are learn\x02ing\r\n"
    cleaned = file_extract._clean_pdfium_text(raw)
    assert "\x02" not in cleaned
    assert "\r" not in cleaned
    assert "learning" in cleaned


def test_the_cap_still_applies(no_pymupdf, tmp_path, monkeypatch):
    monkeypatch.setattr(file_extract, "MAX_EXTRACT_BYTES", 40)
    text = file_extract.extract_text(_copy("two_pages.pdf", tmp_path))
    assert "[... truncated at page 2/2]" in text
    assert "second page" not in text


def test_not_a_pdf_says_so_instead_of_raising(no_pymupdf, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    text = file_extract.extract_text(str(bad))
    assert text is not None and text.startswith("[PDF file: broken.pdf")


def test_concurrent_extractions_all_come_out_whole(no_pymupdf, tmp_path):
    """PDFium holds process-global state. Eight threads at once must each get the whole, right text."""
    paths = [_copy("two_pages.pdf", tmp_path / str(i)) for i in range(8) if (tmp_path / str(i)).mkdir() is None]
    results: list[str | None] = [None] * len(paths)

    def run(i: int) -> None:
        results[i] = file_extract.extract_text(paths[i])

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(paths))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    for text in results:
        assert text is not None and "first page" in text and "second page" in text


def test_nothing_in_the_package_imports_pymupdf():
    """The swap is the whole point: no module may reach for fitz again, guarded or not."""
    src = Path(file_extract.__file__).resolve().parents[1]
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("import fitz", "from fitz", "import pymupdf", "from pymupdf"))
    ]
    assert offenders == []
