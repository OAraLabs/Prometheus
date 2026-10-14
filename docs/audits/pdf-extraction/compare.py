#!/usr/bin/env python3
"""Compare PDF text extraction and image downscaling: PyMuPDF (today) against pypdfium2 + Pillow (proposed).

    python docs/audits/pdf-extraction/compare.py <dir of PDFs> [--json out.json] [--excerpts out.md]
        [--image-from some.pdf]

Run it in an environment that has all four libraries (PyMuPDF, pypdfium2, pypdf, Pillow); it is not part of
the test suite and imports nothing from Prometheus. Each extractor mirrors utils/file_extract.py's
_extract_pdf: one page's text after another, joined with a newline, stripped. "What the model sees" applies
the same 30,000-character cap the daemon applies (MAX_EXTRACT_BYTES), with the same truncation note.

Similarity is measured against PyMuPDF, because the question is "would anyone notice the swap", not "which
is right": 1.0 means the same words in the same order. Two numbers, because they fail differently:

* order: difflib's ratio over the word sequences (reading order matters; a two-column page read across the
  columns scores low here even if every word is present);
* words: the multiset F1 over the words (order ignored; a missing or garbled word scores low here).
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import statistics
import time
from collections import Counter
from pathlib import Path

MAX_EXTRACT_BYTES = 30_000  # utils/file_extract.py:18
MAX_LONG_EDGE = 2576        # gateway/image_prep.py:38
SHRINK_ABOVE = MAX_LONG_EDGE * 2  # gateway/image_prep.py:54
WORD_CAP = 20_000           # difflib is quadratic in the worst case; compare the first 20k words


# ── extractors: page texts, in order ─────────────────────────────────────────

def pages_pymupdf(path: Path) -> list[str]:
    import fitz
    with fitz.open(str(path)) as doc:
        return [page.get_text() for page in doc]


def _clean_pdfium(text: str) -> str:
    """PDFium marks a hyphen it removed at a line break with U+0002 and joins the halves ("learn\x02ing"),
    and ends lines with CRLF. Dropping the marker gives the joined word; CRLF becomes LF like the others."""
    return text.replace("\r\n", "\n").replace("\x02", "")


def pages_pdfium(path: Path) -> list[str]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(path))
    try:
        out = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            out.append(_clean_pdfium(textpage.get_text_bounded()))
            textpage.close()
            page.close()
        return out
    finally:
        pdf.close()


def _utf16(fn, *args) -> str:
    import ctypes
    size = fn(*args, None, 0)
    if size <= 2:
        return ""
    buf = (ctypes.c_ushort * (size // 2))()
    fn(*args, buf, size)
    return bytes(buf).decode("utf-16-le").rstrip("\x00")


def pages_pdfium_fields(path: Path) -> list[str]:
    """Page text, then each filled form field on that page as "name: value". PDFium's text layer never
    includes what was typed into a form (PyMuPDF's includes some of it), so a filled form needs this."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    pdf = pdfium.PdfDocument(str(path))
    try:
        has_form = pdf.init_forms() is not False and pdf.formenv is not None
        env = getattr(pdf.formenv, "raw", None) if has_form else None
        out = []
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = _clean_pdfium(textpage.get_text_bounded())
            textpage.close()
            fields = []
            if env is not None:
                for k in range(raw.FPDFPage_GetAnnotCount(page.raw)):
                    annot = raw.FPDFPage_GetAnnot(page.raw, k)
                    try:
                        if raw.FPDFAnnot_GetSubtype(annot) != raw.FPDF_ANNOT_WIDGET:
                            continue
                        value = _utf16(raw.FPDFAnnot_GetFormFieldValue, env, annot).strip()
                        if value and value != "Off":
                            name = _utf16(raw.FPDFAnnot_GetFormFieldName, env, annot).rsplit(".", 1)[-1]
                            fields.append(f"{name}: {value}")
                    finally:
                        raw.FPDFPage_CloseAnnot(annot)
            if fields:
                text += "\n[Form fields]\n" + "\n".join(fields)
            out.append(text)
            page.close()
        return out
    finally:
        pdf.close()


def pages_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


EXTRACTORS = {"pymupdf": pages_pymupdf, "pypdfium2": pages_pdfium, "pypdfium2+fields": pages_pdfium_fields,
              "pypdf": pages_pypdf}


def model_sees(pages: list[str], name: str) -> str:
    """utils/file_extract.py:_extract_pdf's assembly and cap, applied to any extractor's pages."""
    out, total = [], 0
    for number, text in enumerate(pages, 1):
        if total + len(text) > MAX_EXTRACT_BYTES:
            out.append(text[: MAX_EXTRACT_BYTES - total])
            out.append(f"\n\n[... truncated at page {number}/{len(pages)}]")
            break
        out.append(text)
        total += len(text)
    result = "\n".join(out).strip()
    return result if result else f"[PDF file: {name} — no extractable text (may be image-based)]"


# ── similarity ───────────────────────────────────────────────────────────────

def words(text: str) -> list[str]:
    return text.split()


def normalise(text: str) -> str:
    """What a reader would call the same text: NFKC (ligatures like "ﬁ" become "fi") and a word hyphenated
    across a line break joined back ("learn-\ning" becomes "learning"), on both sides alike."""
    import re
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Table-of-contents dot leaders: ". . . . ." in one library, "....." in another, the same thing.
    return re.sub(r"(?:\. ?){3,}", " … ", text)


def char_ratio(a: str, b: str, cap: int = 12_000) -> float:
    """Over the non-space characters: the measure that means something for Chinese or Japanese, which a
    whitespace split turns into a handful of very long "words"."""
    a2, b2 = "".join(a.split())[:cap], "".join(b.split())[:cap]
    if not a2 and not b2:
        return 1.0
    return difflib.SequenceMatcher(None, a2, b2, autojunk=False).ratio()


def order_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a[:WORD_CAP], b[:WORD_CAP], autojunk=False).ratio()


def bag_f1(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    precision, recall = common / sum(cb.values()), common / sum(ca.values())
    return 2 * precision * recall / (precision + recall)


def timed(fn, arg, runs: int = 3):
    times, result = [], None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(arg)
        times.append(time.perf_counter() - start)
    return result, statistics.median(times) * 1000


# ── images ───────────────────────────────────────────────────────────────────

def downscale_pymupdf(data: bytes) -> tuple[int, int, int]:
    """gateway/image_prep.py:_downscale_if_needed, as it is today: powers of two above SHRINK_ABOVE."""
    import fitz
    pix = fitz.Pixmap(data)
    long_edge, shrink = max(pix.width, pix.height), 0
    while long_edge > SHRINK_ABOVE:
        long_edge //= 2
        shrink += 1
    if shrink:
        pix.shrink(shrink)
    return pix.width, pix.height, len(pix.tobytes(output="png"))


def downscale_pillow(data: bytes) -> tuple[int, int, int]:
    """The proposal: above the same trigger, resample to exactly MAX_LONG_EDGE on the long edge."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        if max(img.size) > SHRINK_ABOVE:
            img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return img.width, img.height, len(buf.getvalue())


def big_png(pdf: Path, scale: float) -> bytes:
    """A large test image: one page rendered by PDFium, the size a phone photo or 5K screenshot arrives at."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    try:
        image = doc[0].render(scale=scale).to_pil()
    finally:
        doc.close()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--excerpts", type=Path)
    parser.add_argument("--image-from", type=Path, help="a PDF whose first page is rendered as the large image")
    args = parser.parse_args()

    rows, excerpts = [], []
    for pdf in sorted(args.pdf_dir.glob("*.pdf")):
        row = {"file": pdf.name, "bytes": pdf.stat().st_size}
        texts = {}
        for name, fn in EXTRACTORS.items():
            try:
                pages, ms = timed(fn, pdf)
                full = "\n".join(pages).strip()
                seen = model_sees(pages, pdf.name)
                texts[name] = (full, seen)
                row[name] = {"pages": len(pages), "ms": round(ms, 1), "chars": len(full),
                             "empty_pages": sum(1 for p in pages if not p.strip()), "seen_chars": len(seen)}
            except Exception as exc:  # one library's failure is a result, not a crash
                texts[name] = ("", "")
                row[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        base_full, base_seen = (words(t) for t in texts["pymupdf"])
        base_norm = words(normalise(texts["pymupdf"][0]))
        for name in ("pypdfium2", "pypdfium2+fields", "pypdf"):
            if "error" in row[name]:
                continue
            full, seen = (words(t) for t in texts[name])
            row[name].update({
                "order_full": round(order_ratio(base_full, full), 3),
                "words_full": round(bag_f1(base_full, full), 3),
                "order_seen": round(order_ratio(base_seen, seen), 3),
                "words_seen": round(bag_f1(base_seen, seen), 3),
                "order_normalised": round(order_ratio(base_norm, words(normalise(texts[name][0]))), 3),
                "char_ratio": round(char_ratio(normalise(texts["pymupdf"][0]), normalise(texts[name][0])), 3),
            })
        if "filled" in pdf.name:
            for name in EXTRACTORS:
                row[name]["filled_values_found"] = texts[name][0].count("FILLEDVALUE")
        rows.append(row)
        excerpts.append((pdf.name, {name: texts[name][1][:700] for name in EXTRACTORS}))

    images = []
    if args.image_from:
        from PIL import Image
        for scale in (8.0, 4.0):
            data = big_png(args.image_from, scale)
            with Image.open(io.BytesIO(data)) as img:
                size = img.size
            (mw, mh, mb), mms = timed(downscale_pymupdf, data, runs=1)
            (pw, ph, pb), pms = timed(downscale_pillow, data, runs=1)
            images.append({"input": f"{size[0]}x{size[1]}", "input_bytes": len(data),
                           "pymupdf": {"out": f"{mw}x{mh}", "bytes": mb, "ms": round(mms)},
                           "pillow": {"out": f"{pw}x{ph}", "bytes": pb, "ms": round(pms)}})

    result = {"pdfs": rows, "images": images}
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.excerpts:
        lines = ["# What the model sees: the first 700 characters, per library", ""]
        for name, by_lib in excerpts:
            lines += [f"## {name}", ""]
            for lib, text in by_lib.items():
                lines += [f"### {lib}", "", "```", text, "```", ""]
        args.excerpts.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
