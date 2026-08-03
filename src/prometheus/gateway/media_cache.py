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


def video_cache_dir() -> Path:
    return _cache_dir("videos")


# ---------------------------------------------------------------------------
# Cache functions (following Hermes module-level pattern)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Quota + eviction (2026-08-03). Previously the cache grew without bound and
# ``cleanup_cache`` below was defined but NEVER CALLED — an orphan guard.
#
# Precedent, not hypothetical: the mini hit 100% disk on 2026-08-01 from
# unpruned nightly copies. An unbounded media cache on the same box is the
# same failure waiting for a different trigger.
#
# Both are CONVENIENCE guards (see gateway.guards): they fail OPEN. A cache
# that cannot evict, or a disk below the floor, must never drop a message —
# refuse to CACHE, never refuse to SERVE.
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_MAX_MB = 512
_DEFAULT_FREE_DISK_FLOOR_MB = 1024

_cache_max_bytes: int = _DEFAULT_CACHE_MAX_MB * 1024 * 1024
_free_disk_floor_bytes: int = _DEFAULT_FREE_DISK_FLOOR_MB * 1024 * 1024


def configure_cache(*, max_mb: int | None = None, free_disk_floor_mb: int | None = None) -> None:
    """Set the cache quota and the free-disk floor (called once at daemon start)."""
    global _cache_max_bytes, _free_disk_floor_bytes
    if max_mb is not None:
        _cache_max_bytes = int(max_mb) * 1024 * 1024
    if free_disk_floor_mb is not None:
        _free_disk_floor_bytes = int(free_disk_floor_mb) * 1024 * 1024


def cache_root() -> Path:
    return get_config_dir() / "cache"


def cache_size_bytes() -> int:
    root = cache_root()
    if not root.exists():
        return 0
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def free_disk_bytes() -> int:
    import shutil

    try:
        return shutil.disk_usage(str(cache_root())).free
    except Exception:
        return _free_disk_floor_bytes + 1  # unknown -> do not block the write


def _below_free_disk_floor() -> bool:
    return free_disk_bytes() < _free_disk_floor_bytes


def evict_lru(target_bytes: int | None = None) -> int:
    """Evict oldest-first until the cache fits. Returns bytes freed.

    CONVENIENCE: any failure is swallowed. Failing to evict must not block the
    write that triggered it.
    """
    limit = _cache_max_bytes if target_bytes is None else target_bytes
    freed = 0
    try:
        root = cache_root()
        if not root.exists():
            return 0
        files = sorted(
            (f for f in root.rglob("*") if f.is_file()),
            key=lambda f: f.stat().st_mtime,
        )
        total = sum(f.stat().st_size for f in files)
        for f in files:
            if total <= limit:
                break
            size = f.stat().st_size
            f.unlink()
            total -= size
            freed += size
            logger.debug("Cache eviction: removed %s (%d bytes)", f, size)
    except Exception as exc:
        logger.warning("Cache eviction failed (continuing): %s", exc)
    return freed


def _cache_write(path: Path, data: bytes) -> str | None:
    """Write through the quota + floor guards. Returns the path, or None.

    None means "not cached" — the caller must still process the media. This is
    the fail-open half of the CONVENIENCE classification.
    """
    if _below_free_disk_floor():
        logger.warning(
            "Free disk below floor (%d MB) — not caching %s; message still processed",
            _free_disk_floor_bytes // (1024 * 1024), path.name,
        )
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except Exception as exc:
        logger.warning("Cache write failed (continuing): %s", exc)
        return None
    if cache_size_bytes() > _cache_max_bytes:
        evict_lru()
    return str(path)


def cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
    """Write image bytes to cache, return absolute path."""
    name = f"img_{uuid4().hex[:12]}{ext}"
    path = image_cache_dir() / name
    written = _cache_write(path, data)
    logger.debug("Cached image: %s (%d bytes)", written or "<not cached>", len(data))
    return written or str(path)


def cache_video_from_bytes(data: bytes, ext: str = ".mp4") -> str:
    """Write video bytes to cache, return absolute path.

    CLOUD EXPANSION (2026-07): sink for the video_generate tool — mirrors
    cache_image_from_bytes so generated videos land next to generated
    images under ~/.prometheus/cache/ and shared cleanup covers both.
    """
    name = f"vid_{uuid4().hex[:12]}{ext}"
    path = video_cache_dir() / name
    written = _cache_write(path, data)
    logger.debug("Cached video: %s (%d bytes)", written or "<not cached>", len(data))
    return written or str(path)


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    """Write audio bytes to cache, return absolute path."""
    name = f"audio_{uuid4().hex[:12]}{ext}"
    path = audio_cache_dir() / name
    written = _cache_write(path, data)
    logger.debug("Cached audio: %s (%d bytes)", written or "<not cached>", len(data))
    return written or str(path)


def cache_document_from_bytes(data: bytes, original_filename: str) -> str:
    """Write document bytes to cache, return absolute path."""
    safe_name = original_filename.replace("/", "_").replace("\\", "_")
    name = f"doc_{uuid4().hex[:12]}_{safe_name}"
    path = document_cache_dir() / name
    written = _cache_write(path, data)
    logger.debug("Cached document: %s (%d bytes)", written or "<not cached>", len(data))
    return written or str(path)


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
