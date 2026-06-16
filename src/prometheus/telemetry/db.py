"""Shared SQLite connection setup for ``telemetry.db`` — WAL + busy_timeout.

The whole daemon writes to ``telemetry.db`` (the ``ToolCallTelemetry`` writer
at ``tracker.py``), the dashboard reads it (``dashboard.py``), and during a
coding run a SECOND process — the ``prometheus code`` subprocess — opens its own
writer and commits a row every round. The telemetry-write-timing survey
(``audits/20260616T014041Z-telemetry-write-timing-survey.md``) caught a real
``database is locked (5)`` when a concurrent reader polled that db at the default
``busy_timeout=0`` under ``journal_mode=delete``.

This module is the single place that configures a telemetry connection so every
opener — writer, dashboard reader, and the forthcoming live-stream tail — shares
one concurrency-safe substrate:

- ``journal_mode=WAL`` — readers never block the per-round writer and vice-versa.
  WAL is a persistent property of the database FILE once set; re-issuing the
  pragma on later opens is a harmless no-op (it just reports ``wal``).
- ``busy_timeout`` — WAL still serialises the residual contention (writer-vs-writer,
  since the daemon and the coding subprocess each hold a writer; and checkpoint).
  ``busy_timeout`` makes a contending connection WAIT rather than raise
  ``SQLITE_BUSY``. It is a per-CONNECTION setting, so it must be re-applied on
  every open — hence a shared helper rather than a one-time migration.

This module configures connections only. It does NOT change what or when anything
writes; the per-round ``INSERT``+immediate-``commit`` in ``tracker.py`` is
untouched (that incrementality is the property the live-stream depends on).
Sprint ``feat/telemetry-wal``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# 5s — comfortably longer than the sub-second single-INSERT+commit write window
# the survey measured, so a contending opener waits through a commit rather than
# erroring. Not so long that a genuinely wedged db hangs a caller indefinitely.
TELEMETRY_BUSY_TIMEOUT_MS = 5000


def apply_telemetry_pragmas(
    conn: sqlite3.Connection,
    *,
    busy_timeout_ms: int = TELEMETRY_BUSY_TIMEOUT_MS,
) -> None:
    """Set ``journal_mode=WAL`` + ``busy_timeout`` on an open telemetry connection.

    Idempotent: safe to call on a connection whose db file is already WAL.
    Must be issued before any transaction is open on *conn* (WAL cannot be
    entered from within a transaction); callers apply it immediately after
    ``sqlite3.connect``.
    """
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")


def connect_telemetry_db(
    db_path: str | Path,
    *,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Open ``telemetry.db`` with the shared WAL + busy_timeout setup applied.

    The single front door every in-process telemetry opener should use so none
    is surprised by the journal mode and all share the busy_timeout. Mirrors the
    historical ``sqlite3.connect(str(path), check_same_thread=False)`` call the
    openers used, plus the pragmas.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    apply_telemetry_pragmas(conn)
    return conn
