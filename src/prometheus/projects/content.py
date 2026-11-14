"""What is in a project's folder: its instructions file and its reference files (docs/design/chat-projects.md, 3.4-3.5).

Read without following links: ``PROMETHEUS.md`` counts only as a regular file at the folder's top level, and a
reference file only as a regular file directly in ``reference/`` whose name passes :func:`valid_reference_name`. A
symlinked ``reference/`` lists nothing. This module only reads; the write routes come with the instructions and files
routes.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

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
_READ_CAP = 1024 * 1024


def instructions_info(folder: str | Path) -> tuple[int, str]:
    """``(characters, rev)`` of the folder's ``PROMETHEUS.md``; ``(0, "")`` when there is no regular file there.

    ``rev`` is the first 16 hex of the SHA-256 of the file's bytes.
    """
    path = Path(folder) / INSTRUCTIONS_FILE
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            return 0, ""
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as fh:
            data = fh.read(_READ_CAP + 1)
    except OSError:
        return 0, ""
    return len(data.decode("utf-8", "replace")), hashlib.sha256(data).hexdigest()[:16]


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
