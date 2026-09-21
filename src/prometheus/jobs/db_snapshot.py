#!/usr/bin/env python3
"""Point-in-time-consistent SQLite snapshots for the nightly backup (#471).

Run::

    python3 -m prometheus.jobs.db_snapshot

WHY THIS EXISTS
---------------
The nightly backup tars ``~/.prometheus`` while the daemon is writing. It does
include ``*.db``, ``*.db-wal`` and ``*.db-shm`` — the common WAL mistake is not
what was happening — but ``tar`` reads those three files **sequentially**. The
main file is read at one instant and the ``-wal`` at a later one, so a checkpoint
landing in between yields a pair that never coexisted. The script suppresses the
one signal that would reveal it (``--warning=no-file-changed``, deliberately: a
live writer must not fail the run) and points the reader at an alternative:

    For a point-in-time-consistent database copy, prefer the daemon's own
    db-snapshots.

**That alternative did not exist.** ``~/.prometheus/db-snapshots`` held three
directories, newest content 2026-06-21, made by hand during a memory-hygiene
pass. Nothing in this repository has ever written there. So the comment named a
mechanism rather than describing one, and anyone who followed the pointer stopped
looking. This module is that mechanism, built rather than deleted — the pointer
becomes true instead of becoming a removed comment.

WHY A SNAPSHOT FIXES THE RACE BY CONSTRUCTION
---------------------------------------------
``VACUUM INTO`` reads through the WAL under a read transaction and writes ONE
self-contained file with no ``-wal``/``-shm`` sidecars. Once written, that file
is **static**. Tar cannot tear a file nothing is writing, so the archive's copy
of a snapshot is sound for the same reason the live copy is not.

Measured here rather than assumed (sqlite 3.53.4, writer committing throughout):

    read-only VACUUM INTO   OK, integrity_check=ok, sidecars=[]
    read-write VACUUM INTO  OK, integrity_check=ok, sidecars=[]

The source is opened ``mode=ro``, so this job cannot write a live database's
DATA even through a bug — a property rather than a promise (CROSS-CUTTING §5).

Stated precisely, because the loose version is wrong and a test pins the exact
claim: ``.db`` and ``-wal`` come out byte-identical; the ``-shm`` does not. That
file is the wal-index, SQLite rebuilds it from the ``-wal`` whenever it is
absent, and **every** reader updates it — a read-only connection included. It
carries no database content, so "the source is unchanged" is true of the two
files that hold data and false of the one that does not.

WHY THE SNAPSHOTS ARE KEPT ON DISK RATHER THAN DELETED AFTER THE TAR
--------------------------------------------------------------------
They live under ``~/.prometheus/db-snapshots/<UTC timestamp>/`` which is INSIDE
the tree the backup already archives, so the nightly tar picks them up with no
change to ``backup.sh`` at all.

⚠ THE 2026-08-01 REWRITE REMOVED STAGING COPIES BECAUSE THEY FILLED THE DISK.
This is not that. That incident was a ~2.8 GB/night copy of the whole tree; these
are SQLite files totalling a few tens of MB, pruned to ``--keep`` sets. If you are
reading this while reclaiming disk, prune harder or lower ``--keep`` — do not
delete the mechanism, or the backup silently returns to capturing torn pairs.

Retention only ever deletes sets this job wrote. Anything else it finds in
``db-snapshots/`` is moved to ``unrecognized/`` and reported separately, never
removed — see :func:`prune_snapshots`.

WIRING (the operator half — this does nothing until it is scheduled)
--------------------------------------------------------------------
Run it shortly BEFORE the nightly tar, so the archive carries a snapshot minutes
old rather than a day old::

    55 2 * * *  cd /path/to/Prometheus && /usr/bin/python3 -m prometheus.jobs.db_snapshot >> /path/to/home/.prometheus/logs/db_snapshot.log 2>&1
    0  3 * * *  /path/to/home/backups/backup.sh

``PYTHONPATH`` is NOT needed when the package is installed. An editable install
leaves a ``.pth`` in site-packages that resolves with no environment at all —
``env -i /usr/bin/python3 -c "import prometheus.jobs.db_snapshot"`` still finds
it. Set ``PYTHONPATH`` only for a checkout that was never installed, and note
that if you set it wrongly you will silently test a DIFFERENT tree.

Use an ABSOLUTE interpreter path: cron's ``PATH`` is minimal, so a bare
``python3`` may not resolve at all. Absolute log paths are steadier than ``~``
for the same reason — depend on as little of the environment as you can.

``backup.sh`` needs NO change to include them: the snapshots land under
``~/.prometheus/db-snapshots/``, inside the tree it already archives. Its comment
does want one word — it currently says to *prefer* the daemon's db-snapshots, and
the honest version now names where they come from and that they are the copy to
restore from::

    # Live *.db/-wal/-shm in this archive are captured sequentially and may be a
    # torn pair. Restore from db-snapshots/<newest>/ instead: written by
    # `python3 -m prometheus.jobs.db_snapshot` (VACUUM INTO), verified at capture,
    # and static by the time tar reads them. db-snapshots/LAST_RUN.json says
    # whether the most recent run succeeded.

FAIL-LOUD CONTRACT
------------------
Any database that cannot be snapshotted or does not verify raises
:class:`SnapshotError` and ``main()`` exits non-zero. The run ALSO writes
``db-snapshots/LAST_RUN.json`` on both success and failure, because "the failure
is silent on both ends" is the actual finding in #471: an exit code is only loud
if something reads it, and cron here reports to nobody. The status file is the
channel that does not depend on anyone watching the run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prometheus.security import install_log_redaction

logger = logging.getLogger("prometheus.jobs.db_snapshot")

# VACUUM INTO landed in SQLite 3.27 (2019). Below that there is no one-statement
# consistent copy and this job must refuse rather than fall back to a file copy —
# a file copy is precisely the torn read it exists to replace.
_MIN_SQLITE = (3, 27, 0)

# Directories never descended into. `logs` and `cache` are what the backup itself
# excludes; `db-snapshots` is this job's own output (snapshotting a snapshot is
# both pointless and how a run becomes quadratic).
_SKIP_DIRS = frozenset({"logs", "cache", "db-snapshots", "__pycache__", ".git"})

_SNAPSHOT_DIRNAME = "db-snapshots"
_MANIFEST_NAME = "MANIFEST.json"
_STATUS_NAME = "LAST_RUN.json"

# Written into a set's directory the moment it is created, BEFORE any database is
# copied. It is how retention tells an interrupted run of THIS job (safe to
# delete) from a directory this job did not write (never safe to delete). The
# manifest cannot carry that meaning: it is written last, so it is absent in both
# cases, and reading its absence as "interrupted" is what made a first run
# destructive.
_OWNED_MARKER = ".written-by-db-snapshot"

# Unrecognised directories are moved here instead of being deleted. It lives
# inside db-snapshots/, so the nightly tar still archives whatever lands in it.
_UNRECOGNIZED_DIRNAME = "unrecognized"


class SnapshotError(RuntimeError):
    """A database could not be snapshotted or did not verify. Exits non-zero."""


@dataclass
class DbResult:
    """One database's outcome, as the manifest records it."""

    source: str
    snapshot: str
    source_bytes: int
    snapshot_bytes: int
    integrity: str
    user_version: int
    tables: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.integrity == "ok"


@dataclass
class RunReport:
    started_at: float
    finished_at: float = 0.0
    snapshot_dir: str = ""
    sqlite_version: str = sqlite3.sqlite_version
    databases: list[DbResult] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    set_aside: list[str] = field(default_factory=list)
    ok: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_databases(root: Path) -> list[Path]:
    """Every ``*.db`` under *root*, skipping :data:`_SKIP_DIRS`, sorted.

    Only the ``.db`` suffix. ``-wal`` and ``-shm`` sidecars do not match it and
    must not be snapshotted separately — ``VACUUM INTO`` folds them in. Neither
    do ``memory.db.backup-<ts>`` files written by ``MemoryStore._snapshot_db``,
    which are already static copies and would double the work.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".db"):
                found.append(Path(dirpath) / name)
    return sorted(found)


# ---------------------------------------------------------------------------
# Snapshot + verify
# ---------------------------------------------------------------------------


def snapshot_database(src: Path, dst: Path) -> DbResult:
    """``VACUUM INTO`` *src* -> *dst*, then verify what was written.

    Verification reads the SNAPSHOT, never the source: the question is whether
    the artefact the backup will carry is sound, and asking the live database
    instead would answer an adjacent one (RECURRING 4j).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():                      # VACUUM INTO refuses an existing target
        dst.unlink()

    # mode=ro: this job cannot write a live database's DATA even through a bug.
    # (The -shm wal-index is still updated, as it is by any reader; it holds no
    # content and SQLite rebuilds it from the -wal. See the module docstring.)
    # immutable=0 is the default and is correct — the file IS changing under us,
    # which is the whole point; `immutable=1` would license SQLite to cache pages
    # it is not entitled to cache and is how a "consistent" copy becomes a lie.
    uri = f"file:{src}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    except sqlite3.Error as exc:
        raise SnapshotError(f"{src}: cannot open read-only: {exc}") from exc
    try:
        # Parameters are not permitted in VACUUM INTO, so the path is quoted as a
        # SQL string literal with '' escaping. Paths here come from os.walk of a
        # directory we own, not from input, but the escaping is not optional.
        conn.execute(f"VACUUM INTO '{str(dst).replace(chr(39), chr(39) * 2)}'")
    except sqlite3.Error as exc:
        raise SnapshotError(f"{src}: VACUUM INTO failed: {exc}") from exc
    finally:
        conn.close()

    if not dst.exists():
        raise SnapshotError(f"{src}: VACUUM INTO reported success but wrote no file")

    sidecars = sorted(p.name for p in dst.parent.glob(dst.name + "-*"))
    if sidecars:
        # If this ever fires, the copy is not self-contained and the archive is
        # back to capturing a multi-file set sequentially.
        raise SnapshotError(f"{dst}: snapshot left sidecars {sidecars}")

    try:
        check = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        integrity = check.execute("PRAGMA integrity_check").fetchone()[0]
        user_version = check.execute("PRAGMA user_version").fetchone()[0]
        tables = check.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()[0]
        check.close()
    except sqlite3.Error as exc:
        raise SnapshotError(f"{dst}: snapshot is unreadable: {exc}") from exc

    result = DbResult(
        source=str(src),
        snapshot=str(dst),
        source_bytes=src.stat().st_size,
        snapshot_bytes=dst.stat().st_size,
        integrity=integrity,
        user_version=int(user_version),
        tables=int(tables),
    )
    if integrity != "ok":
        raise SnapshotError(f"{dst}: integrity_check returned {integrity!r}")
    return result


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


@dataclass
class PruneResult:
    """What retention did, split by whether it DELETED or merely MOVED.

    Two lists rather than one count, because ``(pruned 3)`` reading identically
    for "three of my old sets aged out" and "three directories I did not
    recognise were destroyed" is how the first run of this job deleted three
    hand-made snapshots without anyone noticing.
    """

    pruned: list[str] = field(default_factory=list)
    set_aside: list[str] = field(default_factory=list)


def _is_own_set(path: Path) -> bool:
    """True when this job wrote *path*.

    A complete set has a manifest; an interrupted one has at least the marker,
    written before any copying starts. Neither present means it predates this job
    or belongs to something else, and it is not ours to delete.
    """
    return (path / _MANIFEST_NAME).exists() or (path / _OWNED_MARKER).exists()


def _set_aside(path: Path, aside_root: Path) -> str | None:
    """Move an unrecognised directory out of the retention path. Never deletes.

    Returns the destination name, or ``None`` if it could not be moved — in which
    case it is left exactly where it is, which is the safe failure.
    """
    try:
        aside_root.mkdir(parents=True, exist_ok=True)
        dest = aside_root / path.name
        suffix = 1
        while dest.exists():
            dest = aside_root / f"{path.name}.{suffix}"
            suffix += 1
        shutil.move(str(path), str(dest))
    except OSError:
        logger.warning(
            "could not set aside %s — leaving it in place", path, exc_info=True
        )
        return None
    logger.warning(
        "SET ASIDE %s -> %s: no %s and no %s, so this job did not write it. "
        "Retention does not delete what it did not create.",
        path.name, dest, _MANIFEST_NAME, _OWNED_MARKER,
    )
    return dest.name


def prune_snapshots(snapshot_root: Path, keep: int) -> PruneResult:
    """Delete all but the *keep* newest complete sets THIS JOB WROTE.

    Three classes, and the distinction between the last two is the whole point:

    * **complete** (has a manifest) — ordinary retention; oldest beyond *keep* go.
    * **interrupted** (ours: marker, no manifest) — deleted first, costing no
      ``keep`` slot. Retaining a half-written set over a good one is the opposite
      of what retention is for.
    * **unrecognised** (neither) — **never deleted.** Moved into
      ``db-snapshots/unrecognized/`` and reported separately.

    The third class exists because the first version had only the first two: it
    read "no manifest" as "interrupted run" and deleted on sight, *before* ``keep``
    was applied. A first run therefore destroyed whatever was already in
    ``db-snapshots/`` at any ``keep`` value, and reported it as ordinary
    retention. On 2026-09-20 that deleted three hand-made directories.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    result = PruneResult()
    aside_root = snapshot_root / _UNRECOGNIZED_DIRNAME
    candidates = sorted(
        (
            p for p in snapshot_root.iterdir()
            if p.is_dir() and p.name != _UNRECOGNIZED_DIRNAME
        ),
        key=lambda p: p.name,
        reverse=True,
    )

    for path in [p for p in candidates if not _is_own_set(p)]:
        if _set_aside(path, aside_root) is not None:
            result.set_aside.append(path.name)

    ours = [p for p in candidates if _is_own_set(p)]
    complete = [p for p in ours if (p / _MANIFEST_NAME).exists()]
    interrupted = [p for p in ours if not (p / _MANIFEST_NAME).exists()]
    for path in interrupted + complete[keep:]:
        try:
            shutil.rmtree(path)
            result.pruned.append(path.name)
        except OSError:
            # Retention failing is not a reason to fail the backup: the snapshot
            # that matters was already written and verified above.
            logger.warning("could not prune %s", path, exc_info=True)
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_snapshot(root: Path, *, keep: int = 3) -> RunReport:
    """Snapshot every database under *root*. Raises on the first failure."""
    if sqlite3.sqlite_version_info < _MIN_SQLITE:
        raise SnapshotError(
            f"sqlite {sqlite3.sqlite_version} has no VACUUM INTO "
            f"(needs {'.'.join(map(str, _MIN_SQLITE))}). Refusing: a plain file "
            "copy is the torn read this job exists to replace."
        )

    snapshot_root = root / _SNAPSHOT_DIRNAME
    snapshot_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = snapshot_root / stamp
    # Claim the directory before copying anything. If this run dies partway, the
    # marker is what lets the NEXT run recognise the wreckage as its own and clear
    # it; without it, retention cannot tell our debris from somebody else's data.
    target.mkdir(parents=True, exist_ok=True)
    (target / _OWNED_MARKER).write_text(
        "Written by prometheus.jobs.db_snapshot when this set was created.\n"
        "Retention uses it to tell an interrupted run of this job from a\n"
        "directory the job did not write. Do not remove it.\n",
        encoding="utf-8",
    )

    report = RunReport(started_at=time.time(), snapshot_dir=str(target))
    try:
        databases = discover_databases(root)
        if not databases:
            raise SnapshotError(f"no *.db found under {root} — nothing was captured")
        logger.info("snapshotting %d database(s) from %s", len(databases), root)
        for src in databases:
            rel = src.relative_to(root)
            result = snapshot_database(src, target / rel)
            report.databases.append(result)
            logger.info(
                "  %-40s %8.1f KB -> %8.1f KB  integrity=%s",
                str(rel), result.source_bytes / 1024,
                result.snapshot_bytes / 1024, result.integrity,
            )
        # The manifest is what makes a set COMPLETE, so it is written last and
        # only after every database verified. An interrupted run leaves a
        # directory carrying the marker but no manifest, which the next run's
        # retention recognises as its own and clears.
        (target / _MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "created_at": stamp,
                    "root": str(root),
                    "sqlite_version": sqlite3.sqlite_version,
                    "method": "VACUUM INTO (read-only source)",
                    "databases": [asdict(d) for d in report.databases],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        prune = prune_snapshots(snapshot_root, keep)
        report.pruned = prune.pruned
        report.set_aside = prune.set_aside
        report.ok = True
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report.finished_at = time.time()
        _write_status(snapshot_root, report)
    return report


def _write_status(snapshot_root: Path, report: RunReport) -> None:
    """Record the outcome where something other than cron can read it.

    Written on success AND failure. #471's finding is that this failure mode is
    silent on both ends; a non-zero exit is only loud if something is watching,
    and nothing watches this cron. A status file is readable after the fact by
    anyone — a health probe, the next run, or a person with a question.
    """
    try:
        (snapshot_root / _STATUS_NAME).write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
    except OSError:
        logger.warning("could not write %s", _STATUS_NAME, exc_info=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on failure."""
    parser = argparse.ArgumentParser(
        prog="python3 -m prometheus.jobs.db_snapshot",
        description="Write point-in-time-consistent copies of every Prometheus "
                    "SQLite database, for the nightly backup to archive.",
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Tree to scan (default: the Prometheus config dir, ~/.prometheus).",
    )
    parser.add_argument(
        "--keep", type=int, default=3,
        help="Complete snapshot sets to retain (default: 3).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Whoever configures logging arms redaction — the rule is uniform because it
    # was missed at three of four entry points for months when it was per-site.
    # It is not decorative here: this job logs sqlite error text, and a failing
    # statement can carry row content into the message.
    install_log_redaction()

    root = args.root
    if root is None:
        from prometheus.config.paths import get_config_dir

        root = get_config_dir()

    try:
        report = run_snapshot(Path(root), keep=args.keep)
    except (SnapshotError, ValueError) as exc:
        logger.error("db snapshot FAILED: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — the job must not exit 0 on anything
        logger.error("db snapshot FAILED unexpectedly: %s", exc, exc_info=True)
        return 1

    total = sum(d.snapshot_bytes for d in report.databases)
    # Deleting and setting aside are reported as different things. Collapsing them
    # into one "(pruned N)" is precisely how three directories went quietly.
    notes: list[str] = []
    if report.pruned:
        notes.append(f"pruned {len(report.pruned)}")
    if report.set_aside:
        notes.append(
            f"SET ASIDE {len(report.set_aside)} unrecognised -> "
            f"{_UNRECOGNIZED_DIRNAME}/ (NOT deleted)"
        )
    logger.info(
        "db snapshot OK: %d database(s), %.1f MB, -> %s%s",
        len(report.databases), total / 1_048_576, report.snapshot_dir,
        f" ({'; '.join(notes)})" if notes else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
