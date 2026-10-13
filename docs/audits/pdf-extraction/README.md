# PDF extraction: PyMuPDF against pypdfium2 + Pillow

**Run 2026-10-13 on this Mac (Apple silicon, Python 3.12):** PyMuPDF 1.27.2.3, pypdfium2 5.14.0, pypdf
6.20.1, Pillow 12.3.0. The script is [`compare.py`](compare.py), and the raw numbers are in
[`results.json`](results.json). The PDFs are not committed; their sources are listed below, so anyone can
fetch the same files.

**Why.** PyMuPDF is AGPL-3.0, so the Mac app ships without it, and PDF text extraction and image downscaling
are missing there (`docs/design/plugins.md`, section 2). The question is whether pypdfium2 (text) and Pillow
(images) can replace it in every install without anyone noticing a loss.

**Result: yes.**

- **Text matches.** On every PDF the text is the same as PyMuPDF's once both sides are normalised the same
  way. Normalising means ligatures expanded, a word hyphenated across a line break joined back, and dot
  leaders collapsed.
- **Faster.** Extraction is 1.3 to 2 times faster.
- **Filled forms improve.** PDFium recovers all 8 typed values on a filled form, where PyMuPDF recovers 6.
- **Images improve.** Pillow downscales to exactly the cap, where PyMuPDF's powers of two land 23% over it.

## The PDFs

| File | What it tests | Source |
|---|---|---|
| `resnet-2col.pdf` | two-column paper, ligatures, hyphenation, 12 pages | arxiv.org/pdf/1512.03385v1 |
| `irs-p15-long-tables.pdf` | 59 pages, tables, two columns, a table of contents; past the 30,000-character cap | irs.gov/pub/irs-pdf/p15.pdf |
| `irs-w9-form.pdf` | a fillable form (AcroForm, XFA hybrid), blank | irs.gov/pub/irs-pdf/fw9.pdf |
| `irs-w9-filled.pdf` | the same form with 8 text fields filled programmatically | made from the above with PyMuPDF |
| `scan-skew.pdf` | a real scan, no text layer | OCRmyPDF test resources, `skew.pdf` |
| `scan-francais.pdf` | a real scan, French, no text layer | OCRmyPDF test resources, `francais.pdf` |
| `scan-multipage.pdf` | a real multi-page scan | OCRmyPDF test resources, `multipage.pdf` |
| `scan-with-ocr-layer.pdf` | a scan with an OCR text layer | OCRmyPDF test resources, `graph_ocred.pdf` |
| `udhr-chinese.pdf` | Chinese, 7 pages | ohchr.org, UDHR translation `chn.pdf` |
| `udhr-japanese.pdf` | Japanese, 9 pages | ohchr.org, UDHR translation `jpn.pdf` |

The OCRmyPDF resources are at `github.com/ocrmypdf/OCRmyPDF/tree/main/tests/resources`.

## Text

The **pypdfium2** columns are the proposal: page text, then the page's filled form fields (below).

| File | Pages | PyMuPDF chars | ms | pypdfium2 chars | ms | Same order | Same characters | pypdf ms |
|---|---|---|---|---|---|---|---|---|
| irs-p15-long-tables | 59 | 307,407 | 270 | 304,046 | 190 | 1.000 | 1.000 | 1,650 |
| irs-w9-filled | 6 | 38,291 | 34 | 38,324 | 18 | 0.996 | 0.992 | 63 |
| irs-w9-form | 6 | 38,187 | 34 | 38,113 | 16 | 0.997 | 0.999 | 54 |
| resnet-2col | 12 | 59,435 | 65 | 59,164 | 44 | 0.998 | 1.000 | 264 |
| scan-skew | 1 | 0 | 0.5 | 0 | 0.3 | n/a | n/a | 0.6 |
| scan-francais | 1 | 0 | 4.4 | 0 | 0.5 | n/a | n/a | 0.7 |
| scan-multipage | 6 | 111 | 2.1 | 108 | 0.7 | 1.000 | 1.000 | 3.2 |
| scan-with-ocr-layer | 1 | 111 | 1.3 | 108 | 0.4 | 1.000 | 1.000 | 1.5 |
| udhr-chinese | 7 | 5,920 | 15 | 5,763 | 12 | 1.000 | 1.000 | 102 |
| udhr-japanese | 9 | 4,757 | 12 | 4,471 | 8 | 1.000 | 1.000 | 39 |

The columns:

- **ms** is the median of three runs.
- **Same order** is difflib's ratio over the two word sequences, after the same normalisation on both sides.
  A two-column page read across its columns would score low here.
- **Same characters** is the same ratio over the first 12,000 non-space characters, which is the measure that
  means something for Chinese and Japanese.
- The character counts differ mainly in whitespace (CRLF, spaces between CJK glyphs) and dot leaders.

**What the raw differences turned out to be**, from reading every page that scored below 0.99 before
normalisation:

- **Ligatures.** PDFium returns "classification" where PyMuPDF returns "classiﬁcation", with the U+FB01
  ligature. PDFium's form is the one a search or a model expects.
- **Hyphenation.** PDFium joins a word hyphenated at a line break and marks the join with U+0002
  ("learn\x02ing"). PyMuPDF keeps "learn- ing". Removing the U+0002 gives "learning". The extractor has to
  strip it; without that, the model would see a control character mid-word.
- **Dot leaders.** On the IRS table of contents, PDFium gives "....." and PyMuPDF ". . . . .". This was the
  whole of the worst page's difference, 0.51 before normalisation.
- **Reading order.** No reading-order difference was found on any page, including both two-column documents.

**Scans.**

- Neither library does OCR, and neither did before.
- A scan with no text layer gives no text in either. The daemon then answers with its existing image-based
  placeholder (`utils/file_extract.py:123`), unchanged.
- A scan with an OCR layer gives the same text in both.

**The 30,000-character cap.** Where a document runs past it (the IRS publication), the cut lands a little
differently, because the two libraries' text differs by a few characters per page (dot leaders, whitespace).
What the model sees before the cut is the same text.

## Filled forms: the one place the plain swap lost, and the fix

PDFium's text layer never contains what was typed into a form field, and neither does pypdf's `extract_text`.
PyMuPDF's `get_text` contains some of it.

| Filled values found, of 8 | PyMuPDF | pypdfium2, text only | pypdfium2, text + fields | pypdf |
|---|---|---|---|---|
| `irs-w9-filled.pdf` | 6 | 0 | **8** | 0 |

The proposal reads each page's widget annotations through pypdfium2's own PDFium API
(`FPDFAnnot_GetFormFieldName`, `FPDFAnnot_GetFormFieldValue`, after `init_forms()`). It then appends the filled
ones to that page's text, as a `[Form fields]` block of `name: value` lines, skipping empty and unchecked
(`Off`) fields. That recovers every value, with no third library.

`init_forms()` logs that this PDFium build has no XFA support. The W-9 is an XFA/AcroForm hybrid, and its
AcroForm fields are what was read. A pure-XFA form, which is rare outside some government forms, would show
its static text and no field values, which is what PyMuPDF does with one too.

## Images

`gateway/image_prep.py` shrinks above 5,152 px on the long edge (2 × 2,576), by powers of two. The test image
is a page of the IRS publication rendered as a PNG.

| Input | PyMuPDF today | Pillow (proposal) |
|---|---|---|
| 4,896 × 6,336 px, 7.1 MB | 2,448 × 3,168 px, 2.8 MB, 643 ms | **1,991 × 2,576 px**, 2.5 MB, 609 ms |
| 2,448 × 3,168 px, 2.9 MB (below the trigger) | sent as captured | sent as captured |

Halving cannot hit the cap, so PyMuPDF's output is 23% over the 2,576 px long edge the code says it enforces.
Pillow lands on it exactly, with Lanczos resampling. Below the trigger nothing changes: the original bytes go
untouched, as today (`image_prep.py:125-132`).

## Size and licences

Installed on this Mac, arm64 (`du`):

| | Installed | Licence (package metadata) |
|---|---|---|
| PyMuPDF (today) | 51.4 MiB | AGPL-3.0 or Artifex commercial |
| **pypdfium2** | 7.6 MiB | `BSD-3-Clause, Apache-2.0, dependency licenses` |
| **Pillow** | 12.9 MiB | `MIT-CMU` |
| pypdf (not proposed) | 1.8 MiB | BSD-3-Clause |

The swap is about 31 MiB smaller installed.

**pypdfium2's dependency licences.** The pypdfium2 wheel lists each bundled component's licence as a
`License-File` in its metadata, under `BUILD_LICENSES/`:

- abseil, agg23, fast_float, freetype, icu, lcms, libjpeg-turbo (IJG and its own file), libopenjpeg, libpng,
  llvm-libc, pdfium, pdfium-binaries, simdutf, zlib.

The FreeType file shipped is the FreeType Project License (BSD-style, with an attribution clause), not
the GPL.

**Pillow.** Pillow bundles its image libraries as dylibs: libjpeg, libpng, libtiff, libwebp, libavif,
freetype, harfbuzz, lcms2, openjpeg, brotli, xz, zlib-ng, libxcb.

**This is a reading of licence labels, not legal advice.**

## Before it ships

- **Threading.** pypdfium2's own documentation says: "PDFium is inherently not thread-safe" (its README,
  "Incompatibility with Threading"). The extractor must hold a process-wide lock around every PDFium call,
  and keep the document, page and text-page lifetimes inside it.
- **App build.** The app build must carry pypdfium2's `BUILD_LICENSES` into the bundle. Its metadata licence
  string has no copyleft word in it, so `build_app.py`'s copyleft scan will not copy them on its own. The
  `--without pymupdf` default goes.
- **Not a corpus.** Ten real PDFs is a spot check. It found the one real gap (filled forms) and nothing else.
  A wider run is cheap: point `compare.py` at a folder.
