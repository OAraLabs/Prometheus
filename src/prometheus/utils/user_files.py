"""User file storage — ~/.prometheus/files/

Manages intentionally saved files (vs cache/documents/ which is temporary).
Both Telegram and Beacon handlers call save_user_file() when the user
asks to keep a document. The agent can list saved files via list_user_files()
for context injection at session start.

Cache dir: ~/.prometheus/cache/documents/ → temporary extraction, auto-cleaned
Files dir: ~/.prometheus/files/ → persistent, user-requested saves
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

from prometheus.config.paths import get_config_dir

logger = logging.getLogger(__name__)


def _files_dir() -> Path:
    """Return (and create) the user files directory."""
    d = get_config_dir() / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_user_file(source_path: str, original_name: str | None = None) -> str:
    """Copy a file from cache (or any path) to ~/.prometheus/files/.

    If a file with the same name exists, appends a timestamp to avoid collision.
    Returns the destination path.
    """
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    name = original_name or src.name
    # Strip UUID-hash prefixes from cache filenames (doc_abc123_report.pdf → report.pdf)
    if name.startswith("doc_") and "_" in name[4:]:
        name = name.split("_", 2)[-1]  # Remove doc_{hash}_ prefix

    dest_dir = _files_dir()
    dest = dest_dir / name

    if dest.exists():
        stem = Path(name).stem
        ext = Path(name).suffix
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = dest_dir / f"{stem}_{ts}{ext}"

    shutil.copy2(str(src), str(dest))
    logger.info("Saved user file: %s → %s", src.name, dest)
    return str(dest)


def list_user_files() -> list[dict[str, str | int]]:
    """List all files in ~/.prometheus/files/ with metadata.

    Returns list of { name, path, size, size_formatted, modified }.
    """
    d = _files_dir()
    files = []
    for entry in sorted(d.iterdir()):
        if entry.is_file() and not entry.name.startswith("."):
            stat = entry.stat()
            size = stat.st_size
            if size < 1024:
                size_fmt = f"{size} B"
            elif size < 1024 * 1024:
                size_fmt = f"{size / 1024:.1f} KB"
            else:
                size_fmt = f"{size / (1024 * 1024):.1f} MB"

            files.append({
                "name": entry.name,
                "path": str(entry),
                "size": size,
                "size_formatted": size_fmt,
                "modified": time.strftime("%b %d %H:%M", time.localtime(stat.st_mtime)),
            })
    return files


def files_context_block() -> str | None:
    """Generate a context block listing saved files for system prompt injection.

    Returns None if no files are saved.
    """
    files = list_user_files()
    if not files:
        return None

    lines = ["User's saved files (~/.prometheus/files/):"]
    for f in files:
        lines.append(f"  - {f['name']} ({f['size_formatted']}, {f['modified']})")

    return "\n".join(lines)
