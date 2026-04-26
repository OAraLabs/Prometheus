"""BackupVault — versioned snapshot store for Prometheus source + state.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session B.
License: MIT

Creates timestamped, versioned tarballs of:
  • src/prometheus/            production source code
  • tests/                     test files
  • config/prometheus.yaml     active config (secrets included — vault is local-only)
  • PROMETHEUS.md              project documentation
  • ~/.prometheus/SOUL.md      identity (optional, ``include_identity=True``)
  • ~/.prometheus/AGENTS.md    agent registry (optional)
  • ~/.prometheus/ANATOMY.md   hardware awareness (optional)

Excludes: ``.git/``, ``__pycache__/``, ``.venv/``, ``node_modules/``,
``dist/``, ``build/``, ``*.pyc``, plus the vault's own directory tree
(``~/.prometheus/symbiote/``) and stateful subsystems
(``~/.prometheus/wiki/``, ``~/.prometheus/memory/``).

Manifest lives in SQLite at ``~/.prometheus/symbiote/backups/manifest.db``
so listing/restoring across daemon restarts is cheap.

Retention policy: keep ``max_backups`` newest snapshots. Today's backups,
manual backups, ``symbiote_morph`` backups, and ``pre_restore`` backups
are exempt from auto-cleanup.

This is a SHARED safety utility — independently useful any time a caller
wants a "backup before doing something risky" snapshot, not just for
SYMBIOTE MORPH.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _vault_root_default() -> Path:
    return Path.home() / ".prometheus" / "symbiote" / "backups"


def _project_root_default() -> Path:
    """Best-effort project root — three parents up from this file
    (``…/Prometheus/src/prometheus/symbiote/backup_vault.py`` →
    ``…/Prometheus``)."""
    return Path(__file__).resolve().parents[3]


# Top-level project files/directories to back up. Paths are relative to
# ``project_root`` and resolved at backup time.
_PROJECT_INCLUDES: tuple[str, ...] = (
    "src/prometheus",
    "tests",
    "config/prometheus.yaml",
    "config/prometheus.yaml.default",
    "PROMETHEUS.md",
    "scripts/daemon.py",
    "pyproject.toml",
)

# Identity files in the user's home, optionally backed up.
_IDENTITY_FILES: tuple[str, ...] = (
    "SOUL.md",
    "AGENTS.md",
    "ANATOMY.md",
)

# Names that are skipped wherever they appear in the tree.
_EXCLUDED_NAMES: frozenset[str] = frozenset({
    ".git", ".github", ".gitlab", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "node_modules",
    "dist", "build", ".eggs", "htmlcov",
    ".idea", ".vscode",
})

# Suffixes never archived.
_EXCLUDED_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".swp", ".swo")

# Retention sources that are NEVER auto-cleaned.
_DEFAULT_EXEMPT_SOURCES: frozenset[str] = frozenset({
    "symbiote_morph",
    "manual",
    "pre_restore",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BackupSnapshot:
    """Metadata for one backup snapshot."""

    backup_id: str
    version_number: int
    timestamp: str
    description: str
    source: str
    tarball_path: str
    size_bytes: int
    file_count: int
    manifest: list[str] = field(default_factory=list)
    prometheus_md_hash: str = ""
    test_status: str = "unknown"   # "passing" | "failing" | "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RestoreResult:
    """Result of a restore (or dry-run restore)."""

    backup_id: str
    files_restored: int
    files_added: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    pre_restore_backup_id: str = ""
    tests_passed: bool = False
    tests_output: str = ""
    dry_run: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# BackupVault
# ---------------------------------------------------------------------------


class BackupVault:
    """Versioned backup store. Thread- and async-safe (uses an internal mutex)."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        vault_root: Path | None = None,
        max_backups: int = 10,
        include_identity: bool = True,
        include_config: bool = True,
        exempt_from_retention: set[str] | None = None,
    ) -> None:
        self._project_root = (project_root or _project_root_default()).resolve()
        self._vault_root = (vault_root or _vault_root_default()).expanduser().resolve()
        self._vault_root.mkdir(parents=True, exist_ok=True)
        self._max_backups = max(1, int(max_backups))
        self._include_identity = bool(include_identity)
        self._include_config = bool(include_config)
        self._exempt_sources = frozenset(
            exempt_from_retention or _DEFAULT_EXEMPT_SOURCES
        )

        self._db_path = self._vault_root / "manifest.db"
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_snapshot(
        self,
        description: str = "",
        *,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
        capture_test_status: bool = True,
    ) -> BackupSnapshot:
        """Create a new backup snapshot.

        ``capture_test_status`` runs the pytest suite synchronously with a
        60s timeout to record pass/fail for diagnostics. Tests that exercise
        ``BackupVault`` itself should pass ``capture_test_status=False`` to
        keep the suite fast.
        """
        version = self._get_next_version()
        ts_iso = _now_iso()
        ts_slug = ts_iso.replace(":", "").replace("-", "").rstrip("Z")
        backup_id = f"v{version}_{ts_slug}"
        tarball_path = self._vault_root / f"{backup_id}.tar.gz"

        files = self._collect_files()
        manifest_paths: list[str] = []
        with tarfile.open(tarball_path, "w:gz") as tar:
            for src_path, archive_name in files:
                try:
                    tar.add(src_path, arcname=archive_name, recursive=False)
                except (OSError, tarfile.TarError):
                    log.debug(
                        "BackupVault: could not add %s to tarball",
                        src_path, exc_info=True,
                    )
                    continue
                manifest_paths.append(archive_name)

        size_bytes = tarball_path.stat().st_size if tarball_path.exists() else 0
        prom_md = self._project_root / "PROMETHEUS.md"
        prom_md_hash = _sha256_file(prom_md) if prom_md.exists() else ""

        test_status = "unknown"
        if capture_test_status:
            test_status = await self._run_tests_quick()

        snapshot = BackupSnapshot(
            backup_id=backup_id,
            version_number=version,
            timestamp=ts_iso,
            description=description or f"{source} snapshot",
            source=source,
            tarball_path=str(tarball_path),
            size_bytes=size_bytes,
            file_count=len(manifest_paths),
            manifest=manifest_paths,
            prometheus_md_hash=prom_md_hash,
            test_status=test_status,
            metadata=metadata or {},
        )
        self._persist_manifest(snapshot)
        self._enforce_retention()
        log.info(
            "BackupVault: created %s (%d files, %.1fKB, source=%s)",
            backup_id, len(manifest_paths), size_bytes / 1024.0, source,
        )
        return snapshot

    async def restore_snapshot(
        self,
        backup_id: str,
        *,
        dry_run: bool = False,
        capture_test_status: bool = True,
    ) -> RestoreResult:
        """Restore the project to a snapshot.

        Always creates a ``pre_restore`` backup first (unless ``dry_run``).
        Diffs the staging tree vs live, then replaces live files with the
        backup contents.
        """
        snapshot = self.get_snapshot(backup_id)
        if snapshot is None:
            return RestoreResult(
                backup_id=backup_id,
                files_restored=0,
                error=f"Unknown backup_id: {backup_id!r}",
            )
        tarball_path = Path(snapshot.tarball_path)
        if not tarball_path.exists():
            return RestoreResult(
                backup_id=backup_id,
                files_restored=0,
                error=f"Tarball missing on disk: {tarball_path}",
            )

        # Stage extraction in a temp dir so we can diff before clobbering live.
        with tempfile.TemporaryDirectory(prefix="prometheus_restore_") as tmp:
            staging = Path(tmp) / "staging"
            staging.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(tarball_path, "r:gz") as tar:
                    self._safe_extract_all(tar, staging)
            except (tarfile.TarError, OSError) as exc:
                return RestoreResult(
                    backup_id=backup_id,
                    files_restored=0,
                    error=f"Could not extract tarball: {exc}",
                )

            files_added, files_removed, files_changed = self._diff_staging_vs_live(
                staging, snapshot.manifest,
            )

            if dry_run:
                return RestoreResult(
                    backup_id=backup_id,
                    files_restored=0,
                    files_added=files_added,
                    files_removed=files_removed,
                    files_changed=files_changed,
                    dry_run=True,
                )

            pre_restore = await self.create_snapshot(
                description=f"pre-restore safety net for {backup_id}",
                source="pre_restore",
                metadata={"restoring_to": backup_id},
                capture_test_status=False,  # avoid recursion-style slowness
            )

            files_restored = self._copy_staging_to_live(staging)

        tests_passed = False
        tests_output = ""
        if capture_test_status:
            tests_passed, tests_output = await self._run_tests_full()

        self._log_restore(
            backup_id=backup_id,
            pre_restore_backup_id=pre_restore.backup_id,
            files_restored=files_restored,
            tests_passed=tests_passed,
        )
        return RestoreResult(
            backup_id=backup_id,
            files_restored=files_restored,
            files_added=files_added,
            files_removed=files_removed,
            files_changed=files_changed,
            pre_restore_backup_id=pre_restore.backup_id,
            tests_passed=tests_passed,
            tests_output=tests_output,
        )

    def list_snapshots(self, limit: int = 20) -> list[BackupSnapshot]:
        with self._lock:
            cur = self._db.execute(
                "SELECT data FROM backup_manifest ORDER BY version_number DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = cur.fetchall()
        out: list[BackupSnapshot] = []
        for (data,) in rows:
            try:
                out.append(self._deserialize(data))
            except Exception:
                log.debug("BackupVault: bad manifest row", exc_info=True)
        return out

    def get_snapshot(self, backup_id: str) -> BackupSnapshot | None:
        with self._lock:
            cur = self._db.execute(
                "SELECT data FROM backup_manifest WHERE backup_id = ?",
                (backup_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        try:
            return self._deserialize(row[0])
        except Exception:
            return None

    def get_latest(self) -> BackupSnapshot | None:
        snapshots = self.list_snapshots(limit=1)
        return snapshots[0] if snapshots else None

    # ------------------------------------------------------------------
    # File collection
    # ------------------------------------------------------------------

    def _collect_files(self) -> list[tuple[Path, str]]:
        """Return (absolute_source_path, relative_archive_name) tuples."""
        out: list[tuple[Path, str]] = []

        # Project files.
        for rel in _PROJECT_INCLUDES:
            if rel == "config/prometheus.yaml" and not self._include_config:
                continue
            full = self._project_root / rel
            if not full.exists():
                continue
            if full.is_file():
                out.append((full, rel))
                continue
            # It's a directory — walk it.
            for path in self._walk_files(full):
                rel_path = path.relative_to(self._project_root).as_posix()
                out.append((path, rel_path))

        # Identity files (under ~/.prometheus/, archived under "identity/").
        if self._include_identity:
            identity_root = Path.home() / ".prometheus"
            for name in _IDENTITY_FILES:
                full = identity_root / name
                if full.exists() and full.is_file():
                    out.append((full, f"identity/{name}"))

        return out

    @staticmethod
    def _walk_files(root: Path) -> list[Path]:
        """Walk a tree, skipping excluded dirs and suffixes."""
        out: list[Path] = []
        for path in root.rglob("*"):
            # Skip excluded directories anywhere in the path.
            if any(part in _EXCLUDED_NAMES for part in path.parts):
                continue
            if not path.is_file():
                continue
            if path.suffix in _EXCLUDED_SUFFIXES:
                continue
            out.append(path)
        return out

    # ------------------------------------------------------------------
    # Restore helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_extract_all(tar: tarfile.TarFile, dest: Path) -> None:
        """Extract all members, refusing absolute paths and `..` traversal."""
        dest_resolved = dest.resolve()
        members = []
        for member in tar.getmembers():
            name = member.name.lstrip("/")
            target = (dest / name).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError:
                log.warning(
                    "BackupVault: refusing to extract path-traversal member %r",
                    member.name,
                )
                continue
            members.append(member)
        tar.extractall(path=str(dest), members=members)  # noqa: S202 — guarded above

    def _diff_staging_vs_live(
        self,
        staging: Path,
        manifest_paths: list[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """Compute (added, removed, changed) sets vs the live tree.

        ``added`` = present in staging, missing in live
        ``removed`` = currently in live (within the project includes), absent
                     from the snapshot manifest
        ``changed`` = present in both but bytes differ
        """
        added: list[str] = []
        changed: list[str] = []
        for rel in manifest_paths:
            if rel.startswith("identity/"):
                continue  # don't restore identity over live identity by default
            staging_path = staging / rel
            live_path = self._project_root / rel
            if not staging_path.exists():
                continue
            if not live_path.exists():
                added.append(rel)
                continue
            if staging_path.is_file() and live_path.is_file():
                if _sha256_file(staging_path) != _sha256_file(live_path):
                    changed.append(rel)
        # Removed: in live project includes but not in manifest.
        manifest_set = {p for p in manifest_paths if not p.startswith("identity/")}
        for rel_root in _PROJECT_INCLUDES:
            full = self._project_root / rel_root
            if not full.exists() or full.is_file():
                continue
            for path in self._walk_files(full):
                rel = path.relative_to(self._project_root).as_posix()
                if rel not in manifest_set:
                    removed_check_path = self._project_root / rel
                    if removed_check_path.exists():
                        # Only count file as "removed" if it would no longer
                        # exist in the snapshot tree.
                        if not (staging / rel).exists():
                            # avoid duplicates
                            if rel not in added and rel not in changed:
                                pass  # keep "removed" as info only
        # We don't aggressively delete on restore — `removed` stays empty
        # for the V1 restore behaviour. (The diff above is informational.)
        removed: list[str] = []
        return added, removed, changed

    def _copy_staging_to_live(self, staging: Path) -> int:
        """Copy staged files into the live project tree, preserving structure."""
        count = 0
        for staged in self._walk_files(staging):
            rel = staged.relative_to(staging).as_posix()
            if rel.startswith("identity/"):
                continue
            target = self._project_root / rel
            try:
                target.relative_to(self._project_root)
            except ValueError:
                continue  # path-traversal guard
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(staged, target)
                count += 1
            except OSError:
                log.debug("BackupVault: failed to copy %s", staged, exc_info=True)
        return count

    # ------------------------------------------------------------------
    # SQLite manifest
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS backup_manifest (
                    backup_id TEXT PRIMARY KEY,
                    version_number INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source TEXT NOT NULL,
                    tarball_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    file_count INTEGER NOT NULL,
                    prometheus_md_hash TEXT,
                    test_status TEXT,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS restore_log (
                    restore_id TEXT PRIMARY KEY,
                    backup_id TEXT NOT NULL,
                    pre_restore_backup_id TEXT NOT NULL,
                    restored_at TEXT NOT NULL,
                    files_restored INTEGER NOT NULL,
                    tests_passed INTEGER
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_backup_version "
                "ON backup_manifest(version_number)"
            )
            self._db.commit()

    def _persist_manifest(self, snapshot: BackupSnapshot) -> None:
        payload = json.dumps(snapshot.to_dict())
        with self._lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO backup_manifest
                    (backup_id, version_number, timestamp, description, source,
                     tarball_path, size_bytes, file_count, prometheus_md_hash,
                     test_status, created_at, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.backup_id,
                    snapshot.version_number,
                    snapshot.timestamp,
                    snapshot.description,
                    snapshot.source,
                    snapshot.tarball_path,
                    snapshot.size_bytes,
                    snapshot.file_count,
                    snapshot.prometheus_md_hash,
                    snapshot.test_status,
                    snapshot.timestamp,
                    payload,
                ),
            )
            self._db.commit()

    def _get_next_version(self) -> int:
        with self._lock:
            cur = self._db.execute(
                "SELECT MAX(version_number) FROM backup_manifest"
            )
            row = cur.fetchone()
        current_max = row[0] if row and row[0] is not None else 0
        return int(current_max) + 1

    def _enforce_retention(self) -> None:
        """Delete oldest non-exempt backups beyond the retention limit."""
        with self._lock:
            cur = self._db.execute(
                "SELECT backup_id, source, timestamp, tarball_path "
                "FROM backup_manifest ORDER BY version_number ASC"
            )
            rows = cur.fetchall()
        # Identify candidates that COULD be deleted (non-exempt, not today).
        today_str = date.today().isoformat()
        deletable = []
        retained = 0
        for backup_id, source, ts, tarball in rows:
            is_today = ts.startswith(today_str)
            is_exempt = source in self._exempt_sources
            if is_today or is_exempt:
                continue
            deletable.append((backup_id, tarball))
            retained += 1

        # Total non-exempt count: keep the newest ``max_backups`` of them.
        if retained <= self._max_backups:
            return
        to_remove = deletable[: retained - self._max_backups]
        for backup_id, tarball in to_remove:
            try:
                p = Path(tarball)
                if p.exists():
                    p.unlink()
            except OSError:
                log.debug("BackupVault: could not unlink %s", tarball, exc_info=True)
            with self._lock:
                self._db.execute(
                    "DELETE FROM backup_manifest WHERE backup_id = ?",
                    (backup_id,),
                )
                self._db.commit()
            log.info("BackupVault: pruned old snapshot %s", backup_id)

    def _log_restore(
        self,
        *,
        backup_id: str,
        pre_restore_backup_id: str,
        files_restored: int,
        tests_passed: bool,
    ) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO restore_log
                    (restore_id, backup_id, pre_restore_backup_id,
                     restored_at, files_restored, tests_passed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    backup_id,
                    pre_restore_backup_id,
                    _now_iso(),
                    files_restored,
                    1 if tests_passed else 0,
                ),
            )
            self._db.commit()

    @staticmethod
    def _deserialize(data: str) -> BackupSnapshot:
        obj = json.loads(data)
        return BackupSnapshot(
            backup_id=obj["backup_id"],
            version_number=int(obj.get("version_number", 0)),
            timestamp=obj.get("timestamp", ""),
            description=obj.get("description", ""),
            source=obj.get("source", ""),
            tarball_path=obj.get("tarball_path", ""),
            size_bytes=int(obj.get("size_bytes", 0)),
            file_count=int(obj.get("file_count", 0)),
            manifest=list(obj.get("manifest") or []),
            prometheus_md_hash=obj.get("prometheus_md_hash", ""),
            test_status=obj.get("test_status", "unknown"),
            metadata=obj.get("metadata") or {},
        )

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    async def _run_tests_quick(self) -> str:
        """Return ``"passing"`` | ``"failing"`` | ``"unknown"`` for diagnostics."""
        passed, _ = await self._run_pytest(timeout=60, full_output=False)
        if passed is None:
            return "unknown"
        return "passing" if passed else "failing"

    async def _run_tests_full(self) -> tuple[bool, str]:
        passed, output = await self._run_pytest(timeout=300, full_output=True)
        return bool(passed), output or ""

    async def _run_pytest(
        self,
        *,
        timeout: int,
        full_output: bool,
    ) -> tuple[bool | None, str]:
        cmd = ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except (FileNotFoundError, OSError):
            return None, ""
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None, "timeout"
        text = stdout.decode("utf-8", errors="replace") if full_output else ""
        return proc.returncode == 0, text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _sha256_file(path: Path, *, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(block_size), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
