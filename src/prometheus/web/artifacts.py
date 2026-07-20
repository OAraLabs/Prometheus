"""Artifact outbox index — content-addressed downloads for remote clients (Beacon).

The agent saves deliverables into the outbox (``get_artifacts_dir()``, default
``~/.prometheus/files``); this module indexes that tree and serves it by CONTENT
ID (sha256 prefix) instead of by path. Clients never send a path string, so the
whole traversal/symlink-escape class stays out of the wire contract; ids survive
renames and moves; identical bytes dedup to one id. Scan-on-read with an
(size, mtime) hash cache — no database, no watcher: the outbox directory IS the
registry. Symlinks and dotfiles are never indexed; oversized files are listed
but flagged unhashable-safe via the size cap below.
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

# Content ids are sha256[:16] — 64 bits of prefix, ample for a personal outbox.
_ID_LEN = 16
# Streaming-hash chunk; also the FileResponse default is fine for delivery.
_CHUNK = 1024 * 1024
# Refuse to hash (and therefore to index) anything over 1 GiB — the outbox is a
# delivery tray for documents, not a bulk store; a cap keeps scan-on-read honest.
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024

# abs path → (size, mtime_ns, id): re-hash only when the file actually changed.
_hash_cache: dict[str, tuple[int, int, str]] = {}


def _content_id(path: Path, size: int, mtime_ns: int) -> str:
    key = str(path)
    cached = _hash_cache.get(key)
    if cached and cached[0] == size and cached[1] == mtime_ns:
        return cached[2]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    digest = h.hexdigest()[:_ID_LEN]
    _hash_cache[key] = (size, mtime_ns, digest)
    return digest


def _iter_regular_files(root: Path) -> list[Path]:
    """Regular, non-hidden, non-symlink files under root (recursive), containment-checked.

    Symlinks are skipped OUTRIGHT (not resolved-and-checked): the outbox is a
    publish boundary, and a link is a pointer to something that was NOT placed
    here. The belt-and-suspenders resolve() check guards odd mounts.
    """
    out: list[Path] = []
    resolved_root = root.resolve()
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        rp = p.resolve()
        if rp != resolved_root and resolved_root not in rp.parents:
            continue
        out.append(p)
    return out


def scan_artifacts(root: Path) -> list[dict[str, object]]:
    """The manifest: every deliverable in the outbox, newest first.

    Entry: ``{id, name, path, size, mtime, mime}`` — ``path`` is outbox-relative
    (display only; the wire contract for download is the id), ``name`` is the
    basename clients match in chat text, ``mime`` is a best-effort guess.
    """
    entries: list[dict[str, object]] = []
    for p in _iter_regular_files(root):
        try:
            st = p.stat()
        except OSError:
            continue
        if st.st_size > MAX_ARTIFACT_BYTES:
            continue
        try:
            digest = _content_id(p, st.st_size, st.st_mtime_ns)
        except OSError:
            continue
        entries.append(
            {
                "id": digest,
                "name": p.name,
                "path": str(p.relative_to(root)),
                "size": st.st_size,
                "mtime": st.st_mtime,
                "mime": mimetypes.guess_type(p.name)[0] or "application/octet-stream",
            }
        )
    entries.sort(key=lambda e: float(e["mtime"]), reverse=True)  # type: ignore[arg-type]
    return entries


def resolve_artifact(root: Path, artifact_id: str) -> Path | None:
    """id → path, via a fresh scan (the id is only ever matched against files the
    scan itself found — no client string touches the filesystem). None if the id
    is unknown, malformed, or the file has since left the outbox."""
    if not isinstance(artifact_id, str) or len(artifact_id) != _ID_LEN or not all(c in "0123456789abcdef" for c in artifact_id):
        return None
    for entry in scan_artifacts(root):
        if entry["id"] == artifact_id:
            return root / str(entry["path"])
    return None
