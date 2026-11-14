"""What is in a project's folder: its instructions file and its reference files (docs/design/chat-projects.md, 3.4-3.5).

Read without following links: ``PROMETHEUS.md`` counts only as a regular file at the folder's top level, and a
reference file only as a regular file directly in ``reference/`` whose name passes :func:`valid_reference_name`. A
symlinked ``reference/`` lists nothing. This module only reads; the write routes come with the instructions and files
routes.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
import unicodedata
from pathlib import Path

INSTRUCTIONS_FILE = "PROMETHEUS.md"
REFERENCE_DIR = "reference"
#: The instruction loader's per-file cap (context/prometheusmd.py), so what is saved is what the model sees.
MAX_INSTRUCTIONS_CHARS = 12_000
NAME_MAX = 120
_CHUNK = 256 * 1024


def instructions_info(folder: str | Path) -> tuple[int, str]:
    """``(characters, rev)`` of the folder's ``PROMETHEUS.md``; ``(0, "")`` when there is no regular file there.

    ``rev`` is the first 16 hex of the SHA-256 of the whole file's bytes, read in chunks.
    """
    path = Path(folder) / INSTRUCTIONS_FILE
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return 0, ""
        # O_NONBLOCK so a FIFO swapped in after the lstat cannot hang the read; fstat confirms what was opened.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as fh:
            if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                return 0, ""
            digest = hashlib.sha256()
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            chars = 0
            while chunk := fh.read(_CHUNK):
                digest.update(chunk)
                chars += len(decoder.decode(chunk))
            chars += len(decoder.decode(b"", final=True))
    except OSError:
        return 0, ""
    return chars, digest.hexdigest()[:16]


def valid_reference_name(name: str) -> bool:
    """One path segment: no separator, no leading dot, no control or format character, at most 120 characters."""
    if not name or len(name) > NAME_MAX or name.startswith(".") or "/" in name or "\\" in name:
        return False
    return not any(unicodedata.category(ch) in ("Cc", "Cf") for ch in name)


def reference_names(folder: str | Path) -> list[str]:
    """The reference files listed: regular files directly in ``reference/`` with valid names, sorted."""
    ref = Path(folder) / REFERENCE_DIR
    try:
        if not stat.S_ISDIR(os.lstat(ref).st_mode):
            return []
        with os.scandir(ref) as entries:
            return sorted(e.name for e in entries
                          if valid_reference_name(e.name) and e.is_file(follow_symlinks=False))
    except OSError:
        return []
