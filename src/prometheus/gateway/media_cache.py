"""Media cache — disk-backed cache for Telegram photos, audio, and documents.

Donor pattern: NousResearch/hermes-agent gateway/platforms/base.py (module-level cache functions).
Adapted for Prometheus: paths use ~/.prometheus/cache/{type}/, UUID-based filenames.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from prometheus.config.paths import get_config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported document types (extension -> MIME)
# ---------------------------------------------------------------------------

SUPPORTED_DOCUMENT_TYPES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".html": "text/html",
    ".xml": "application/xml",
    ".log": "text/plain",
    ".sh": "text/x-shellscript",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".env": "text/plain",
    ".sql": "application/sql",
    ".dockerfile": "text/plain",
}

# Max inline text injection size for text documents
MAX_INLINE_TEXT_BYTES = 100_000  # 100 KB


# ---------------------------------------------------------------------------
# Cache directories
# ---------------------------------------------------------------------------

def _cache_dir(subdir: str) -> Path:
    """Return (and create) a cache subdirectory under ~/.prometheus/cache/."""
    d = get_config_dir() / "cache" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_cache_dir() -> Path:
    return _cache_dir("images")


def audio_cache_dir() -> Path:
    return _cache_dir("audio")


def document_cache_dir() -> Path:
    return _cache_dir("documents")


# ---------------------------------------------------------------------------
# Cache functions (following Hermes module-level pattern)
# ---------------------------------------------------------------------------

def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """Write image bytes to cache, return absolute path."""
    name = f"img_{uuid4().hex[:12]}{ext}"
    path = image_cache_dir() / name
    path.write_bytes(data)
    logger.debug("Cached image: %s (%d bytes)", path, len(data))
    return str(path)


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """Write audio bytes to cache, return absolute path."""
    name = f"audio_{uuid4().hex[:12]}{ext}"
    path = audio_cache_dir() / name
    path.write_bytes(data)
    logger.debug("Cached audio: %s (%d bytes)", path, len(data))
    return str(path)


def cache_document_from_bytes(data: bytes, original_filename: str) -> str:
    """Write document bytes to cache, return absolute path."""
    safe_name = original_filename.replace("/", "_").replace("\\", "_")
    name = f"doc_{uuid4().hex[:12]}_{safe_name}"
    path = document_cache_dir() / name
    path.write_bytes(data)
    logger.debug("Cached document: %s (%d bytes)", path, len(data))
    return str(path)


def extract_text_from_document(path: str) -> str | None:
    """Extract text content from a cached document if it's a text format.

    Returns the text content (up to MAX_INLINE_TEXT_BYTES), or None if
    the file is binary or too large for inline injection.
    """
    p = Path(path)
    ext = p.suffix.lower()
    mime = SUPPORTED_DOCUMENT_TYPES.get(ext, "")

    # Only inline text-based formats
    if not mime.startswith("text/") and ext not in (".json", ".yaml", ".yml", ".toml", ".sql"):
        return None

    if p.stat().st_size > MAX_INLINE_TEXT_BYTES:
        return None

    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def sniff_image_extension(file_path: str | None) -> str:
    """Guess image extension from a Telegram file_path string."""
    if file_path:
        for ext in (".png", ".webp", ".gif", ".jpeg", ".jpg"):
            if file_path.lower().endswith(ext):
                return ext
    return ".jpg"


def cleanup_cache(subdir: str, max_age_hours: int = 24) -> int:
    """Remove files older than max_age_hours from a cache subdirectory."""
    cache_dir = _cache_dir(subdir)
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    for f in cache_dir.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    return removed
