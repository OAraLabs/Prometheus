"""Queued writer for the telemetry v2 tables (WP-X.54 T-1).

WHY A QUEUE. Every telemetry write today is a plain INSERT plus an immediate
commit on the caller's thread, which on the turn path is the event loop
(audit 2026-09-30, "What changes the plan" item 2). The insert itself is
sub-millisecond, but it sits behind a 5 s ``busy_timeout``: whenever the
coding subprocess or a checkpoint holds the write lock, the turn waits. The v2
tables add a write per model response, so they go through this instead: a
bounded queue drained by ONE thread with its OWN WAL connection. The caller
puts a row on the queue and returns.

WHAT IT WRITES. The new tables only (``V2_TABLES``), plus the write-once
``telemetry_v2_capture_since`` stamp in ``schema_meta`` (``stamp_meta``), plus
``call``: a function run on the writer thread, in queue order, for the
conditional updates T-4's outcomes need. The existing INSERTs stay
synchronous (ruling 6): the Beacon coding live stream, the context meter and
the parity runner read those rows back immediately, and a queue would make
them late.

ITS PROMISES.

* The time on a row is when the write was QUEUED. ``stamp=<column>`` fills
  that column from the clock on the caller's thread, so a backed-up queue
  delays the row, never its timestamp.
* Shutdown drains. ``close()`` writes everything queued before it, then stops
  the thread. It is registered with ``atexit`` too, because the daemon never
  closes its tracker; a SIGKILL still loses the queue, as it would lose an
  in-flight synchronous write.
* Nothing is lost silently. A full queue drops the row and counts it; the
  writer thread records the count as one ``silent_failures`` row
  (``telemetry_writer`` / ``queue_full``) once it catches up. A row the
  database refuses is a WARNING plus a ``silent_failures`` row
  (``telemetry_writer`` / ``write``), and the rows queued beside it still land.
* Nothing raises into the caller, except a programming error caught at the
  call site: a table outside ``V2_TABLES`` or a column name that is not an
  identifier is a ``ValueError``, because it would otherwise be interpolated
  into SQL.

Free-text columns (``REDACTED_COLUMNS``) are redacted on the writer thread,
before the INSERT, through the same redactor every capture store uses.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import queue
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.security.log_redaction import redact_json_text
from prometheus.telemetry.db import connect_telemetry_db

log = logging.getLogger(__name__)

#: The only tables this writer will touch.
V2_TABLES: frozenset[str] = frozenset({"turns", "responses", "tool_sets"})

#: The one other table the writer touches, through ``stamp_meta`` only.
_META_TABLE = "schema_meta"

#: Columns that can carry conversation text, redacted before they are stored.
#: tests/test_scrub_covers_every_text_column.py holds the scrub to the same list.
REDACTED_COLUMNS: dict[str, frozenset[str]] = {
    "responses": frozenset({"prose"}),
}

#: Queue bound. At ~266 rounds a day this is hours of backlog, so a full queue
#: means the writer is wedged, not busy.
DEFAULT_MAXSIZE = 1000

#: Rows written per transaction when the queue has backed up.
_BATCH = 200

_IDENT = re.compile(r"^[a-z_][a-z0-9_]*$")


def tool_set_row(names: Iterable[str]) -> dict[str, Any]:
    """The ``tool_sets`` row for the tools offered in one round.

    The set, not the list: order and repeats do not make a different set, so
    they do not make a different hash. The names are stored once per distinct
    set and each ``responses`` row carries only ``tool_set_hash`` (Will's
    addition to the rulings, 2026-09-30). 16 hex characters: a 64-bit key is
    ample for the handful of distinct sets a deployment offers, and a full
    sha256 hex string is the shape the repo's secret scanner refuses.
    """
    unique = sorted(set(names))
    names_json = json.dumps(unique, separators=(",", ":"))
    return {
        "tool_set_hash": hashlib.sha256(names_json.encode("utf-8")).hexdigest()[:16],
        "tool_names": names_json,
        "tool_count": len(unique),
    }


@dataclass
class _Write:
    table: str
    row: dict[str, Any]
    verb: str                     # "insert" | "insert_or_ignore" | "upsert"
    key: str | None = None


@dataclass
class _Call:
    label: str                    # names it in a silent_failures row
    fn: Callable[[sqlite3.Connection], Any]


@dataclass
class _Marker:
    done: threading.Event = field(default_factory=threading.Event)


_STOP = object()


class TelemetryV2Writer:
    """Bounded queue + one writer thread + its own WAL connection."""

    THREAD_NAME = "telemetry-v2-writer"

    def __init__(
        self,
        db_path: str | Path,
        *,
        maxsize: int = DEFAULT_MAXSIZE,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._db_path = Path(db_path)
        self._clock = clock
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._closed = False
        self._dropped = 0
        self._unreported: Counter[str] = Counter()
        self._thread = threading.Thread(target=self._run, name=self.THREAD_NAME, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # -- caller side ---------------------------------------------------------

    @property
    def dropped(self) -> int:
        """Rows refused because the queue was full or the writer was closed."""
        return self._dropped

    def insert(
        self,
        table: str,
        row: dict[str, Any],
        *,
        stamp: str | None = None,
        or_ignore: bool = False,
    ) -> bool:
        """Queue one INSERT. Returns False if it was dropped (and counted)."""
        return self._submit(_Write(table, dict(row), "insert_or_ignore" if or_ignore else "insert"),
                            stamp)

    def upsert(
        self,
        table: str,
        row: dict[str, Any],
        *,
        key: str,
        stamp: str | None = None,
    ) -> bool:
        """Queue an INSERT that, on a ``key`` conflict, updates the given columns.

        Only the columns in ``row`` are written, so a turn's start and its end
        can be two upserts that each fill their own half.
        """
        if key not in row:
            raise ValueError(f"upsert row has no value for its key {key!r}")
        return self._submit(_Write(table, dict(row), "upsert", key), stamp)

    def stamp_meta(self, key: str) -> bool:
        """Queue a ``schema_meta`` boundary stamp: ``key`` = now, set ONCE.

        INSERT OR IGNORE, so a later process never moves it. The value is the
        clock when the stamp was QUEUED, like every row here, and as a string,
        like the tracker's other boundary keys. The one write outside
        ``V2_TABLES``, and only to ``schema_meta``: T-3 stamps
        ``telemetry_v2_capture_since`` through the queue rather than with a
        synchronous write on the turn path.
        """
        if not _IDENT.match(key):
            raise ValueError(f"not a schema_meta key: {key!r}")
        return self._enqueue(_Write(_META_TABLE, {"key": key, "value": str(self._clock())},
                                    "insert_or_ignore"))

    def call(self, label: str, fn: Callable[[sqlite3.Connection], Any]) -> bool:
        """Queue ``fn(conn)`` to run on the writer thread, in its own transaction.

        For writes an INSERT or an upsert cannot say: T-4's outcomes are
        conditional UPDATEs ("only while ``outcome`` IS NULL") that must run
        AFTER the turn rows queued before them, which the one queue gives for
        free. ``fn`` gets the writer's connection and must touch only
        ``V2_TABLES``; it is committed when it returns. One that raises is
        rolled back and becomes a WARNING plus a ``silent_failures`` row named
        ``label``, and the rows queued after it still land. Returns False if it
        was dropped (and counted), like a row.
        """
        if not _IDENT.match(label):
            raise ValueError(f"not a call label: {label!r}")
        return self._enqueue(_Call(label, fn))

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until everything queued so far is written. True if it was."""
        if self._closed:
            return not self._thread.is_alive()
        marker = _Marker()
        try:
            self._queue.put(marker, timeout=timeout)
        except queue.Full:
            return False
        return marker.done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        """Drain the queue, then stop the thread. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            atexit.unregister(self.close)
        except Exception:  # pragma: no cover - interpreter shutdown
            pass
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            log.warning("telemetry v2 writer: queue still full after %.1fs; "
                        "%d queued row(s) may be lost", timeout, self._queue.qsize())
        self._thread.join(timeout)
        if self._thread.is_alive():
            log.warning("telemetry v2 writer: did not drain within %.1fs; "
                        "%d row(s) still queued", timeout, self._queue.qsize())

    def _submit(self, write: _Write, stamp: str | None) -> bool:
        if write.table not in V2_TABLES:
            raise ValueError(f"the v2 writer writes only {sorted(V2_TABLES)}, not {write.table!r}")
        for col in [*write.row, *([stamp] if stamp else [])]:
            if not _IDENT.match(col):
                raise ValueError(f"not a column name: {col!r}")
        if stamp and write.row.get(stamp) is None:
            write.row[stamp] = self._clock()
        return self._enqueue(write)

    def _enqueue(self, write: _Write | _Call) -> bool:
        # Check and put under the lock close() takes, so no row can land
        # behind the stop sentinel and be lost without being counted.
        with self._lock:
            why = "writer closed" if self._closed else None
            if why is None:
                try:
                    self._queue.put_nowait(write)
                except queue.Full:
                    why = "queue full"
        if why is not None:
            self._count_drop(write.table if isinstance(write, _Write) else write.label, why)
            return False
        return True

    def _count_drop(self, table: str, why: str) -> None:
        with self._lock:
            self._dropped += 1
            self._unreported[table] += 1
            first = self._dropped == 1
        if first or self._dropped % 100 == 0:
            log.warning("telemetry v2 writer: dropped a %s row (%s); %d dropped so far",
                        table, why, self._dropped)

    # -- writer thread -------------------------------------------------------

    def _run(self) -> None:
        try:
            conn = connect_telemetry_db(self._db_path)
        except Exception:
            log.warning("telemetry v2 writer: could not open %s; nothing will be written",
                        self._db_path, exc_info=True)
            self._drain_without_db()
            return
        try:
            while True:
                item = self._queue.get()
                batch = [item]
                while len(batch) < _BATCH and item is not _STOP:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    batch.append(item)
                stop = False
                writes: list[_Write] = []
                for it in batch:
                    if it is _STOP:
                        stop = True
                    elif isinstance(it, _Marker):
                        self._write_all(conn, writes)
                        writes = []
                        it.done.set()
                    elif isinstance(it, _Call):
                        self._write_all(conn, writes)
                        writes = []
                        self._run_call(conn, it)
                    else:
                        writes.append(it)
                self._write_all(conn, writes)
                self._report_drops(conn)
                if stop:
                    break
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _drain_without_db(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if isinstance(item, _Marker):
                item.done.set()

    def _write_all(self, conn: sqlite3.Connection, writes: list[_Write]) -> None:
        if not writes:
            return
        try:
            for w in writes:
                conn.execute(*_statement(w))
            conn.commit()
            return
        except Exception:
            _rollback(conn)
        # One row was refused. Write them one by one so its neighbours land.
        for w in writes:
            try:
                conn.execute(*_statement(w))
                conn.commit()
            except Exception as exc:
                _rollback(conn)
                log.warning("telemetry v2 writer: %s into %s failed", w.verb, w.table,
                            exc_info=True)
                self._silent_failure(conn, "write", exc, {"table": w.table, "verb": w.verb})

    def _run_call(self, conn: sqlite3.Connection, call: _Call) -> None:
        try:
            call.fn(conn)
            conn.commit()
        except Exception as exc:
            _rollback(conn)
            log.warning("telemetry v2 writer: %s failed", call.label, exc_info=True)
            self._silent_failure(conn, call.label, exc, {})

    def _report_drops(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            if not self._unreported:
                return
            tables = dict(self._unreported)
            self._unreported.clear()
        self._silent_failure(conn, "queue_full", QueueFull(f"dropped {sum(tables.values())} row(s)"),
                             {"dropped": sum(tables.values()), "tables": tables})

    def _silent_failure(self, conn: sqlite3.Connection, operation: str,
                        exc: BaseException, context: dict[str, Any]) -> None:
        from prometheus.telemetry.tracker import insert_silent_failure

        try:
            insert_silent_failure(conn, "telemetry_writer", operation, exc, context)
            conn.commit()
        except Exception:
            _rollback(conn)
            log.warning("telemetry v2 writer: could not record a %s failure", operation,
                        exc_info=True)


class QueueFull(Exception):
    """What a ``queue_full`` row in ``silent_failures`` names as its exception."""


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except Exception:
        pass


def _statement(w: _Write) -> tuple[str, list[Any]]:
    redact = REDACTED_COLUMNS.get(w.table, frozenset())
    cols = list(w.row)
    values = [redact_json_text(v) if c in redact and isinstance(v, str) else v
              for c, v in w.row.items()]
    names = ", ".join(cols)
    marks = ", ".join("?" for _ in cols)
    if w.verb == "upsert":
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != w.key)
        sql = (f"INSERT INTO {w.table} ({names}) VALUES ({marks}) "
               f"ON CONFLICT({w.key}) DO " + (f"UPDATE SET {updates}" if updates else "NOTHING"))
    elif w.verb == "insert_or_ignore":
        sql = f"INSERT OR IGNORE INTO {w.table} ({names}) VALUES ({marks})"
    else:
        sql = f"INSERT INTO {w.table} ({names}) VALUES ({marks})"
    return sql, values
