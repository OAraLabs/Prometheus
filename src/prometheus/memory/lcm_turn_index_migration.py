"""One-time migration: make ``(session_id, turn_index)`` UNIQUE in ``lcm.db``.

``turn_index`` is a message's durable prompt position, and the LCM readers order
by it. For most of this store's life nothing kept it unique: every restart of a
session's numbering (a daemon restart that did not rehydrate, a rehydrate that
numbered from its window instead of the session, ``/reset``, a rollback of a
durable row) re-used indices that older rows already held, and
``ORDER BY turn_index`` then zipped two conversations together. The numbering
fix is in :class:`~prometheus.engine.session.ChatSession`. This module repairs
the rows already written and installs the UNIQUE index that guards the fix.
Evidence and design: docs/audits/LCM-TURN-INDEX-DUPLICATES.md §5.3.

The daemon calls :func:`migrate_turn_index` once at start, after the LCM engine
exists and BEFORE any writer (gateways, jobs, the compactor) is wired to it. It
never raises: on any failure it rolls back, says so loudly, and the daemon runs
on without the unique index, exactly as before this migration existed.

**Gated** by ``PRAGMA user_version`` 0 → 1 on ``lcm.db``. Neither the version nor
an index is a table, so a migrated DB keeps exactly the tables and columns it
had. That is also why the renumbering map lives in a TEMP table: a new table or
column in ``lcm.db`` would change every parity golden's store dump.

**No duplicates** (every fresh install, every parity run): the legacy non-unique
index is swapped for the UNIQUE one and ``user_version`` is set. Nothing else is
written, and no backup file is made.

**Duplicates**: first a backup with SQLite's backup API to
``<lcm.db>.pre-turn-index-<UTC time>.bak``, checked (``integrity_check`` and row
count) before anything is touched. Then ONE ``BEGIN IMMEDIATE`` transaction:

1. each affected session's rows (rowid order) are split into numbering *runs*.
   A run ends where the numbering restarted: an index repeats inside the run, or
   the index drops below everything the run has used. A drop into an unused gap
   of the run stays in the run: that is a message sent mid-turn and persisted
   ahead of the tail below it, whose ``turn_index`` is already its true place;
2. run 0 keeps its values; each later run is shifted up to start above every
   value before it, keeping its internal gaps; within a run the order is
   ``(turn_index, rowid)``;
3. the new values go in through a TEMP map, the legacy index is dropped, the
   UNIQUE index is created and ``user_version`` is set;
4. before COMMIT: the row count is unchanged, a digest of every other column (by
   rowid) is unchanged, no ``(session_id, turn_index)`` repeats, and every
   affected session reads back in exactly the planned order. Any mismatch rolls
   the whole transaction back.

⚠ Pre-fix code must not write a migrated DB. It inserts with
``INSERT OR REPLACE``, which under the UNIQUE index resolves a collision by
DELETING the older row. Roll back to such a version only after undoing the
index (see the migration PR's deploy note), or restore the ``.bak``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from prometheus.memory.lcm_conversation_store import (
    LEGACY_TURN_INDEX_INDEX,
    TURN_INDEX_UNIQUE_INDEX,
)

log = logging.getLogger(__name__)

#: ``PRAGMA user_version`` of an ``lcm.db`` whose ``(session_id, turn_index)`` is UNIQUE.
TURN_INDEX_UNIQUE_VERSION = 1

_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6
# Headroom demanded on the backup's filesystem, beyond the DB's own size.
_BACKUP_HEADROOM_BYTES = 64 * 1024 * 1024


class TurnIndexMigrationError(RuntimeError):
    """The migration refused to proceed, or a check failed. Nothing was committed."""


@dataclass
class RenumberPlan:
    """What the migration will change. Pure data, built by :func:`plan_renumbering`."""

    #: ``(rowid, new turn_index)`` for every row whose value changes.
    mapping: list[tuple[int, int]] = field(default_factory=list)
    #: For each affected session, its rowids in the intended read order.
    order: dict[str, list[int]] = field(default_factory=dict)
    runs: int = 0
    max_shift: int = 0
    #: Rows whose place in the intended order differs from their rowid order
    #: (the mid-turn rows that ``turn_index`` puts back where the model saw them).
    reordered: int = 0

    @property
    def sessions(self) -> int:
        return len(self.order)


@dataclass
class TurnIndexMigrationResult:
    #: ``migrated`` | ``indexed`` (no duplicates) | ``done_before`` | ``no_table`` | ``failed``
    status: str
    sessions: int = 0
    runs: int = 0
    rows_renumbered: int = 0
    max_shift: int = 0
    rows_reordered: int = 0
    backup_path: str | None = None
    backup_bytes: int = 0
    seconds: float = 0.0
    error: str | None = None


def plan_renumbering(rows: list[tuple[int, str, int]]) -> RenumberPlan:
    """Plan the renumbering of ``rows`` = ``(rowid, session_id, turn_index)``.

    Only sessions that repeat a ``turn_index`` are touched. The rows need not be
    sorted; each session is walked in rowid order. See the module docstring for
    the run rule. The result is unique per session by construction: a run never
    repeats an index, and every later run is lifted above every earlier value.
    """
    by_session: dict[str, list[tuple[int, int]]] = {}
    for rowid, session_id, turn_index in rows:
        by_session.setdefault(session_id, []).append((rowid, turn_index))

    plan = RenumberPlan()
    for session_id, session_rows in by_session.items():
        session_rows.sort()
        if len({ti for _, ti in session_rows}) == len(session_rows):
            continue  # no repeated index: untouched, whatever its order

        runs: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        used: set[int] = set()
        low = 0
        for rowid, ti in session_rows:
            if used and (ti in used or ti < low):
                runs.append(current)
                current, used = [], set()
            if not used:
                low = ti
            current.append((rowid, ti))
            used.add(ti)
            low = min(low, ti)
        runs.append(current)

        order: list[int] = []
        top = -1
        for k, run in enumerate(runs):
            shift = 0 if k == 0 else max(0, top + 1 - min(ti for _, ti in run))
            for rowid, ti in sorted(run, key=lambda r: (r[1], r[0])):
                if shift:
                    plan.mapping.append((rowid, ti + shift))
                    plan.max_shift = max(plan.max_shift, shift)
                order.append(rowid)
            top = max(top, max(ti for _, ti in run) + shift)
        plan.runs += len(runs)
        plan.order[session_id] = order
        plan.reordered += sum(
            1 for (rowid, _), planned in zip(session_rows, order) if rowid != planned
        )
    return plan


def default_backup_path(db_path: str | Path) -> Path:
    """``<lcm.db>.pre-turn-index-<UTC time>.bak``, beside the DB.

    Not ``*.db``, so the nightly ``db_snapshot`` job does not treat it as a live
    database to capture.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = Path(db_path)
    return path.with_name(f"{path.name}.pre-turn-index-{stamp}.bak")


def migrate_turn_index(
    db_path: str | Path, *, backup_path: str | Path | None = None
) -> TurnIndexMigrationResult:
    """Daemon-start entry point. Never raises; see the module docstring."""
    started = time.perf_counter()
    if not Path(db_path).exists():
        # Connecting would CREATE an empty file; the store makes the DB, not this.
        return TurnIndexMigrationResult("no_table", seconds=_since(started))
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=30.0)
    except sqlite3.Error as exc:
        return _failed(exc, started, None)
    try:
        return run_turn_index_migration(
            conn,
            backup_path=str(backup_path or default_backup_path(db_path)),
        )
    finally:
        conn.close()


def run_turn_index_migration(
    conn: sqlite3.Connection, *, backup_path: str
) -> TurnIndexMigrationResult:
    """Migrate the database open on *conn*. Never raises.

    *conn* must be in autocommit mode (``isolation_level=None``) and must not be
    inside a transaction. *backup_path* is where the pre-migration backup goes
    when there is something to renumber; ``":memory:"`` keeps it in memory,
    for a dry run that must not write a file.
    """
    started = time.perf_counter()
    backup: tuple[str, int] | None = None
    try:
        if conn.in_transaction or conn.isolation_level is not None:
            raise TurnIndexMigrationError(
                "the migration needs an autocommit connection outside any transaction"
            )
        if _user_version(conn) >= TURN_INDEX_UNIQUE_VERSION:
            return TurnIndexMigrationResult("done_before", seconds=_since(started))
        if not _has_messages_table(conn):
            return TurnIndexMigrationResult("no_table", seconds=_since(started))

        renumber = _has_duplicates(conn)
        seen_version = None
        if renumber:
            # Backup BEFORE the write lock: SQLite cannot back up from a
            # connection that holds a write transaction (the step reports BUSY
            # for as long as the transaction lasts). data_version then proves
            # that no other connection committed between the backup and the lock.
            seen_version = _data_version(conn)
            backup = _backup(conn, backup_path)

        conn.execute("BEGIN IMMEDIATE")
        try:
            if _user_version(conn) >= TURN_INDEX_UNIQUE_VERSION:
                conn.execute("ROLLBACK")
                return TurnIndexMigrationResult("done_before", seconds=_since(started))
            plan = RenumberPlan()
            count, digest = 0, ""
            if renumber:
                if _data_version(conn) != seen_version:
                    raise TurnIndexMigrationError(
                        "another connection wrote lcm.db between the backup and the "
                        "write lock; nothing was changed, the next start will retry"
                    )
                count = _row_count(conn)
                if backup is None or backup[1] != count:
                    raise TurnIndexMigrationError(
                        f"the backup does not hold the {count} rows being migrated"
                    )
                digest = _digest_except_turn_index(conn)
                plan = plan_renumbering(
                    conn.execute(
                        "SELECT rowid, session_id, turn_index FROM lcm_messages"
                    ).fetchall()
                )
                _apply(conn, plan)
            conn.execute(f"DROP INDEX IF EXISTS {LEGACY_TURN_INDEX_INDEX}")
            # Creating the UNIQUE index is itself the proof that no pair repeats.
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {TURN_INDEX_UNIQUE_INDEX}"
                " ON lcm_messages (session_id, turn_index)"
            )
            conn.execute(f"PRAGMA user_version = {TURN_INDEX_UNIQUE_VERSION}")
            if renumber:
                _verify(conn, plan, count, digest)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    except Exception as exc:
        return _failed(exc, started, backup)

    result = TurnIndexMigrationResult(
        "migrated" if renumber else "indexed",
        sessions=plan.sessions,
        runs=plan.runs,
        rows_renumbered=len(plan.mapping),
        max_shift=plan.max_shift,
        rows_reordered=plan.reordered,
        backup_path=backup[0] if backup else None,
        backup_bytes=_file_size(backup[0]) if backup else 0,
        seconds=_since(started),
    )
    if renumber:
        log.warning(
            "LCM turn_index migration: renumbered %d rows in %d sessions (%d runs, "
            "largest shift %d); (session_id, turn_index) is now UNIQUE. Backup of the "
            "previous state: %s (%d bytes). %.2fs",
            result.rows_renumbered, result.sessions, result.runs, result.max_shift,
            result.backup_path, result.backup_bytes, result.seconds,
        )
    else:
        log.info(
            "LCM turn_index migration: no duplicates; (session_id, turn_index) is now "
            "UNIQUE (%.2fs)", result.seconds,
        )
    return result


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def _backup(conn: sqlite3.Connection, backup_path: str) -> tuple[str, int]:
    """Back up *conn* to *backup_path* and check the copy. Returns (path, rows)."""
    in_memory = backup_path == ":memory:"
    target = Path(backup_path)
    if not in_memory:
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        needed = page_count * page_size + _BACKUP_HEADROOM_BYTES
        free = shutil.disk_usage(target.parent).free
        if free < needed:
            raise TurnIndexMigrationError(
                f"not enough free space for the backup: {free} bytes free, "
                f"{needed} needed in {target.parent}"
            )
        # Created up front, owner-only and exclusively: the copy is private data
        # from its first byte, and an existing file is never overwritten.
        try:
            os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            raise TurnIndexMigrationError(
                f"backup target already exists: {target}"
            ) from None

    def _refuse_to_spin(status: int, remaining: int, total: int) -> None:
        # Python's backup retries BUSY/LOCKED forever; a migration must not hang
        # the daemon's start.
        if status in (_SQLITE_BUSY, _SQLITE_LOCKED):
            raise TurnIndexMigrationError("lcm.db was locked during the backup")

    try:
        copy = sqlite3.connect(backup_path)
        try:
            conn.backup(copy, progress=_refuse_to_spin)
            integrity = copy.execute("PRAGMA integrity_check").fetchone()[0]
            rows = copy.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0]
        finally:
            copy.close()
        if integrity != "ok":
            raise TurnIndexMigrationError(f"backup failed integrity_check: {integrity}")
    except BaseException:
        # A partial or corrupt copy is not a backup: never leave one looking like one.
        if not in_memory:
            target.unlink(missing_ok=True)
        raise
    return backup_path, int(rows)


def _apply(conn: sqlite3.Connection, plan: RenumberPlan) -> None:
    """Write the new values through a TEMP map (never a table in lcm.db)."""
    conn.execute(
        "CREATE TEMP TABLE turn_index_map "
        "(rid INTEGER PRIMARY KEY, new_turn_index INTEGER NOT NULL)"
    )
    conn.executemany("INSERT INTO temp.turn_index_map VALUES (?, ?)", plan.mapping)
    conn.execute(
        "UPDATE lcm_messages SET turn_index ="
        " (SELECT new_turn_index FROM temp.turn_index_map"
        "  WHERE rid = lcm_messages.rowid)"
        " WHERE rowid IN (SELECT rid FROM temp.turn_index_map)"
    )
    conn.execute("DROP TABLE temp.turn_index_map")


def _verify(
    conn: sqlite3.Connection, plan: RenumberPlan, count: int, digest: str
) -> None:
    if _row_count(conn) != count:
        raise TurnIndexMigrationError("row count changed during the migration")
    if _digest_except_turn_index(conn) != digest:
        raise TurnIndexMigrationError("a column other than turn_index changed")
    if _has_duplicates(conn):
        raise TurnIndexMigrationError("a (session_id, turn_index) pair still repeats")
    for session_id, expected in plan.order.items():
        got = [
            r[0]
            for r in conn.execute(
                "SELECT rowid FROM lcm_messages WHERE session_id = ?"
                " ORDER BY turn_index, rowid",
                (session_id,),
            )
        ]
        if got != expected:
            raise TurnIndexMigrationError(
                f"a session does not read back in the planned order ({len(got)} rows)"
            )


def _failed(
    exc: BaseException, started: float, backup: tuple[str, int] | None
) -> TurnIndexMigrationResult:
    log.error(
        "LCM TURN_INDEX MIGRATION FAILED — nothing was committed; the daemon runs on "
        "WITHOUT the unique index, as before. %s%s",
        exc,
        f" (backup kept at {backup[0]})" if backup and backup[0] != ":memory:" else "",
        exc_info=True,
    )
    try:
        from prometheus.telemetry.tracker import get_telemetry_handle

        tel = get_telemetry_handle()
        if tel is not None:
            tel.record_silent_failure(
                subsystem="lcm",
                operation="turn_index_migration",
                exc=exc,
                context={"backup_path": backup[0] if backup else None},
            )
    except Exception:
        log.warning("telemetry unavailable to record the migration failure", exc_info=True)
    return TurnIndexMigrationResult(
        "failed",
        backup_path=backup[0] if backup else None,
        backup_bytes=_file_size(backup[0]) if backup else 0,
        seconds=_since(started),
        error=f"{type(exc).__name__}: {exc}",
    )


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _data_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA data_version").fetchone()[0])


def _has_messages_table(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lcm_messages'"
    ).fetchone() is not None


def _has_duplicates(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM lcm_messages GROUP BY session_id, turn_index"
        " HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone() is not None


def _row_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0])


def _digest_except_turn_index(conn: sqlite3.Connection) -> str:
    """SHA-256 over every column but turn_index, row by row in rowid order."""
    columns = [
        r[1]
        for r in conn.execute("PRAGMA table_info(lcm_messages)")
        if r[1] != "turn_index"
    ]
    select = ", ".join(f'"{c}"' for c in columns)
    digest = hashlib.sha256()
    for row in conn.execute(f"SELECT rowid, {select} FROM lcm_messages ORDER BY rowid"):
        digest.update(repr(tuple(row)).encode())
    return digest.hexdigest()


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path) if path != ":memory:" else 0
    except OSError:
        return 0


def _since(started: float) -> float:
    return round(time.perf_counter() - started, 3)
