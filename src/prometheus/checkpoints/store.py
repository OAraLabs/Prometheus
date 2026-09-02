"""Per-turn file checkpoints — the undo an agent that edits your files needs.

Item 4 of the 2026-09-01 roadmap. Everything that looked like this before
said, in its own docstring, that it was not: the file-mutation verifier
holds size/mtime/mode and no content ("detection, never containment"),
the divergence CheckpointStore snapshots messages and is forensic by
design, and BackupVault snapshots Prometheus's own tree. Nothing held the
bytes a turn overwrote.

This store does, for sessions that have a workspace (item W):

* **When**: once per turn, at turn start, before any tool runs — "restore
  the files to before this message". Bash writes are covered because the
  whole workspace is captured, not the paths a tool declared; the bwrap
  write floor is what makes the workspace the complete write domain.
* **What**: every regular file under the workspace, skipping the usual
  generated trees (``.git``, ``node_modules``, ``.venv``, …) and files over
  ``max_file_bytes``. Skipped files are RECORDED per checkpoint, so a
  restore can say "not restored: too large" instead of pretending.
* **How**: content-addressed blobs (sha256) under the store root, so an
  unchanged file costs one hash per turn and no bytes; a SQLite manifest
  keyed by checkpoint id. Caps (``max_files``, ``max_total_bytes``) refuse
  to checkpoint a workspace that is too big — logged at ERROR, the turn
  proceeds, and the list shows the gap rather than a checkpoint that lies.
* **Restore**: writes changed/missing files back from their blobs, deletes
  files created since the checkpoint (under the workspace, outside skip
  dirs), reports every path per category. ``dry_run`` reports without
  touching anything. Naming the checkpoint twice is the confirmation.
* **Retention**: the newest ``keep_per_session`` per session; blobs no
  checkpoint references are garbage-collected on prune.

Deliberately NOT here: sessions without a workspace. The daemon's own cwd
is the deploy clone, and checkpointing it on every turn of every session
would be the wrong domain and the wrong cost. Point a conversation at a
directory (``/workspace``), and its turns become undoable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

DEFAULT_SKIP_DIRS: tuple[str, ...] = (
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox",
    ".prometheus",
)
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "max_file_bytes": 2_000_000,
    "max_total_bytes": 200_000_000,
    "max_files": 5000,
    "keep_per_session": 20,
    "skip_dirs": list(DEFAULT_SKIP_DIRS),
}


def resolve_checkpoints_config(config: dict | None) -> dict[str, Any]:
    """The ``checkpoints:`` section with every key present. Absent keys take
    the shipped default; ``enabled`` absent means ON — undo is not a feature
    a fresh install should have to discover.

    Each read carries its fallback as a LITERAL equal to the template value
    (tests/test_config_defaults_equality pins the two), so a default change
    is a diff on both sides or a red test, never a silent drift.
    """
    raw = (config or {}).get("checkpoints") or {}
    skip = raw.get("skip_dirs", [".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
                                 ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox",
                                 ".prometheus"])
    return {
        "enabled": bool(raw.get("enabled", True)),
        "max_file_bytes": int(raw.get("max_file_bytes", 2000000)),
        "max_total_bytes": int(raw.get("max_total_bytes", 200000000)),
        "max_files": int(raw.get("max_files", 5000)),
        "keep_per_session": int(raw.get("keep_per_session", 20)),
        "skip_dirs": [str(d) for d in skip] if isinstance(skip, list) else list(DEFAULT_SKIP_DIRS),
    }


@dataclass
class CheckpointRecord:
    id: str
    session_id: str
    workspace: str
    label: str
    created_at: float
    files: int
    bytes: int
    skipped: int
    files_detail: list[dict[str, Any]] = field(default_factory=list)
    skipped_detail: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "session_id": self.session_id, "workspace": self.workspace,
            "label": self.label, "created_at": self.created_at,
            "files": self.files, "bytes": self.bytes, "skipped": self.skipped,
        }


class CheckpointRefused(RuntimeError):
    """The workspace could not be checkpointed; the message says why."""


class FileCheckpointStore:
    def __init__(self, root: str | Path, *, config: dict | None = None) -> None:
        self.root = Path(root).expanduser()
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.cfg = resolve_checkpoints_config({"checkpoints": config} if config else None)
        self._db_path = self.root / "checkpoints.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                workspace   TEXT NOT NULL,
                label       TEXT NOT NULL,
                created_at  REAL NOT NULL,
                files       INTEGER NOT NULL,
                bytes       INTEGER NOT NULL,
                skipped     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id, created_at);
            CREATE TABLE IF NOT EXISTS checkpoint_files (
                checkpoint_id TEXT NOT NULL,
                rel_path      TEXT NOT NULL,
                sha256        TEXT NOT NULL,
                size          INTEGER NOT NULL,
                mtime         REAL NOT NULL,
                mode          INTEGER NOT NULL,
                PRIMARY KEY (checkpoint_id, rel_path)
            );
            CREATE TABLE IF NOT EXISTS checkpoint_skipped (
                checkpoint_id TEXT NOT NULL,
                rel_path      TEXT NOT NULL,
                reason        TEXT NOT NULL,
                PRIMARY KEY (checkpoint_id, rel_path)
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------ walk

    def _walk(self, workspace: Path) -> Iterator[tuple[Path, str]]:
        skip = set(self.cfg["skip_dirs"])
        for dirpath, dirnames, filenames in os.walk(workspace):
            dirnames[:] = sorted(d for d in dirnames if d not in skip)
            for name in sorted(filenames):
                full = Path(dirpath) / name
                if full.is_symlink() or not full.is_file():
                    continue
                yield full, str(full.relative_to(workspace))

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _blob_path(self, sha: str) -> Path:
        return self.blobs / sha[:2] / sha

    def _store_blob(self, path: Path, sha: str) -> None:
        dest = self._blob_path(sha)
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(path.read_bytes())
        os.replace(tmp, dest)

    # ---------------------------------------------------------------- create

    def create(self, session_id: str, workspace: str | Path, label: str = "") -> CheckpointRecord:
        """Snapshot *workspace* for *session_id*. Raises CheckpointRefused
        (nothing written) when a cap is exceeded or the directory is gone."""
        ws = Path(workspace).expanduser().resolve()
        if not ws.is_dir():
            raise CheckpointRefused(f"workspace {ws} is not a directory")
        cid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
        files: list[tuple[str, str, int, float, int, Path]] = []
        skipped: list[tuple[str, str]] = []
        total = 0
        for full, rel in self._walk(ws):
            st = full.stat()
            if st.st_size > int(self.cfg["max_file_bytes"]):
                skipped.append((rel, f"larger than max_file_bytes ({st.st_size} > {self.cfg['max_file_bytes']})"))
                continue
            total += st.st_size
            if total > int(self.cfg["max_total_bytes"]):
                raise CheckpointRefused(
                    f"workspace exceeds max_total_bytes ({self.cfg['max_total_bytes']}) — not checkpointed"
                )
            files.append((rel, "", st.st_size, st.st_mtime, st.st_mode, full))
            if len(files) > int(self.cfg["max_files"]):
                raise CheckpointRefused(
                    f"workspace has more than max_files ({self.cfg['max_files']}) files — not checkpointed"
                )
        rows = []
        for rel, _sha, size, mtime, mode, full in files:
            sha = self._sha256(full)
            self._store_blob(full, sha)
            rows.append((cid, rel, sha, size, mtime, mode))
        with self._conn:
            self._conn.execute(
                "INSERT INTO checkpoints (id, session_id, workspace, label, created_at, files, bytes, skipped)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, session_id, str(ws), (label or "")[:200], time.time(), len(rows), total, len(skipped)),
            )
            self._conn.executemany(
                "INSERT INTO checkpoint_files (checkpoint_id, rel_path, sha256, size, mtime, mode) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.executemany(
                "INSERT INTO checkpoint_skipped (checkpoint_id, rel_path, reason) VALUES (?, ?, ?)",
                [(cid, rel, why) for rel, why in skipped],
            )
        self.prune(session_id)
        return self.get(cid)  # type: ignore[return-value]

    # ------------------------------------------------------------------ read

    def list(self, session_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id, session_id, workspace, label, created_at, files, bytes, skipped FROM checkpoints"
            " WHERE session_id = ? ORDER BY created_at DESC", (session_id,),
        )
        return [
            {"id": r[0], "session_id": r[1], "workspace": r[2], "label": r[3],
             "created_at": r[4], "files": r[5], "bytes": r[6], "skipped": r[7]}
            for r in cur.fetchall()
        ]

    def get(self, cid: str) -> CheckpointRecord | None:
        r = self._conn.execute(
            "SELECT id, session_id, workspace, label, created_at, files, bytes, skipped FROM checkpoints WHERE id = ?",
            (cid,),
        ).fetchone()
        if r is None:
            return None
        files = [
            {"rel_path": f[0], "sha256": f[1], "size": f[2], "mtime": f[3], "mode": f[4]}
            for f in self._conn.execute(
                "SELECT rel_path, sha256, size, mtime, mode FROM checkpoint_files WHERE checkpoint_id = ? ORDER BY rel_path",
                (cid,),
            ).fetchall()
        ]
        skipped = [
            {"rel_path": s[0], "reason": s[1]}
            for s in self._conn.execute(
                "SELECT rel_path, reason FROM checkpoint_skipped WHERE checkpoint_id = ? ORDER BY rel_path", (cid,),
            ).fetchall()
        ]
        return CheckpointRecord(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], files, skipped)

    def diff(self, cid: str) -> dict[str, Any]:
        """The workspace now vs the checkpoint: what a restore would touch."""
        rec = self.get(cid)
        if rec is None:
            raise KeyError(cid)
        ws = Path(rec.workspace)
        if not ws.is_dir():
            return {"workspace_missing": True, "changed": [], "deleted": [], "added": [], "unchanged": 0}
        recorded = {f["rel_path"]: f for f in rec.files_detail}
        skipped = {s["rel_path"] for s in rec.skipped_detail}
        current: dict[str, Path] = {rel: full for full, rel in self._walk(ws)}
        changed, deleted, added = [], [], []
        unchanged = 0
        for rel, meta in recorded.items():
            full = current.get(rel)
            if full is None:
                deleted.append(rel)
            elif full.stat().st_size != meta["size"] or self._sha256(full) != meta["sha256"]:
                changed.append(rel)
            else:
                unchanged += 1
        for rel in current:
            if rel not in recorded and rel not in skipped:
                added.append(rel)
        return {"workspace_missing": False, "changed": sorted(changed), "deleted": sorted(deleted),
                "added": sorted(added), "unchanged": unchanged, "uncaptured": sorted(skipped)}

    # --------------------------------------------------------------- restore

    def restore(self, cid: str, *, dry_run: bool = False) -> dict[str, Any]:
        """Put the workspace back to the checkpoint. Returns what was (or
        would be) restored, deleted and left alone — every path named."""
        rec = self.get(cid)
        if rec is None:
            raise KeyError(cid)
        d = self.diff(cid)
        if d.get("workspace_missing"):
            raise CheckpointRefused(f"workspace {rec.workspace} no longer exists")
        ws = Path(rec.workspace)
        meta = {f["rel_path"]: f for f in rec.files_detail}
        result = {"checkpoint": cid, "dry_run": dry_run,
                  "restored": [], "deleted": [], "unchanged": d["unchanged"],
                  "uncaptured": d["uncaptured"], "errors": []}
        for rel in d["changed"] + d["deleted"]:
            m = meta[rel]
            blob = self._blob_path(m["sha256"])
            if not blob.exists():
                result["errors"].append({"path": rel, "error": "blob missing from the store"})
                continue
            result["restored"].append(rel)
            if dry_run:
                continue
            target = ws / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".prometheus-restore.tmp")
            tmp.write_bytes(blob.read_bytes())
            os.replace(tmp, target)
            try:
                os.chmod(target, m["mode"] & 0o7777)
            except OSError:
                pass
        for rel in d["added"]:
            result["deleted"].append(rel)
            if not dry_run:
                try:
                    (ws / rel).unlink()
                except FileNotFoundError:
                    pass
        return result

    # ------------------------------------------------------------- retention

    def prune(self, session_id: str) -> int:
        keep = int(self.cfg["keep_per_session"])
        ids = [r[0] for r in self._conn.execute(
            "SELECT id FROM checkpoints WHERE session_id = ? ORDER BY created_at DESC", (session_id,),
        ).fetchall()]
        stale = ids[keep:]
        if not stale:
            return 0
        with self._conn:
            for cid in stale:
                self._conn.execute("DELETE FROM checkpoint_files WHERE checkpoint_id = ?", (cid,))
                self._conn.execute("DELETE FROM checkpoint_skipped WHERE checkpoint_id = ?", (cid,))
                self._conn.execute("DELETE FROM checkpoints WHERE id = ?", (cid,))
        self.gc_blobs()
        return len(stale)

    def gc_blobs(self) -> int:
        live = {r[0] for r in self._conn.execute("SELECT DISTINCT sha256 FROM checkpoint_files").fetchall()}
        removed = 0
        for sub in self.blobs.iterdir() if self.blobs.exists() else []:
            if not sub.is_dir():
                continue
            for blob in sub.iterdir():
                if blob.name not in live and not blob.name.endswith(".tmp"):
                    blob.unlink()
                    removed += 1
        return removed
