"""The telemetry v2 queued writer (WP-X.54 T-1).

The audit found every telemetry write synchronous on the turn path, with a
5 s ``busy_timeout`` behind it. The new tables are written through a bounded
queue drained by one thread with its own WAL connection, so the caller never
waits on the database. These tests pin the four promises that design makes:

* the caller does not wait, even when the database is locked;
* the time on a row is when the write was QUEUED, not when it landed;
* shutdown drains the queue, so nothing queued is lost on a clean exit;
* a write that cannot happen (queue full, insert refused) is a WARNING plus a
  ``silent_failures`` row, never an exception and never silence.

And one restriction: it writes the new tables only. Existing INSERTs stay
synchronous (ruling 6), because readers depend on them appearing at once.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.telemetry.writer import V2_TABLES, TelemetryV2Writer, tool_set_row


def _rows(db: Path, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "telemetry.db"
    ToolCallTelemetry(path).close()  # the schema comes from the tracker
    return path


def _response(i: int = 0, **extra) -> dict:
    return {"session_id": "desktop:s", "turn_id": "desktop:s:t", "round_index": i,
            "response_kind": "prose", **extra}


class _Lock:
    """Hold the database's write lock from another connection."""

    def __init__(self, db: Path) -> None:
        self.conn = sqlite3.connect(str(db), isolation_level=None)
        self.conn.execute("BEGIN IMMEDIATE")

    def release(self) -> None:
        self.conn.execute("COMMIT")
        self.conn.close()


def test_the_new_tables_are_the_only_ones_it_writes(db):
    assert V2_TABLES == {"turns", "responses", "tool_sets"}
    w = TelemetryV2Writer(db)
    try:
        for table in ("tool_calls", "subsystem_runs", "silent_failures", "signal_events"):
            with pytest.raises(ValueError):
                w.insert(table, {"id": "x"})
        with pytest.raises(ValueError):
            w.insert("responses", {"session_id; DROP TABLE turns": "x"})
    finally:
        w.close()


def test_the_caller_never_waits_on_a_locked_database(db):
    lock = _Lock(db)
    w = TelemetryV2Writer(db)
    try:
        start = time.monotonic()
        for i in range(5):
            assert w.insert("responses", _response(i), stamp="ts") is True
        assert time.monotonic() - start < 0.5   # the lock would cost 5 s per write
    finally:
        lock.release()
        w.close()
    assert _rows(db, "SELECT round_index FROM responses ORDER BY id") == [(i,) for i in range(5)]


def test_the_time_is_stamped_when_the_write_is_queued(db):
    calls: list[str] = []

    def clock() -> float:
        calls.append(threading.current_thread().name)
        return 1234.5

    lock = _Lock(db)
    w = TelemetryV2Writer(db, clock=clock)
    try:
        w.insert("responses", _response(), stamp="ts")
        w.insert("responses", _response(1, ts=99.0), stamp="ts")  # a caller's own time wins
    finally:
        lock.release()
        w.close()
    # Read on the CALLER's thread, once per queued write that needed it.
    assert calls == [threading.current_thread().name]
    assert _rows(db, "SELECT ts FROM responses ORDER BY id") == [(1234.5,), (99.0,)]


def test_close_drains_everything_that_was_queued(db):
    w = TelemetryV2Writer(db)
    for i in range(200):
        assert w.insert("responses", _response(i), stamp="ts")
    w.close()
    assert _rows(db, "SELECT COUNT(*) FROM responses") == [(200,)]
    assert _rows(db, "SELECT COUNT(*) FROM silent_failures") == [(0,)]


def test_flush_waits_for_the_queue_without_closing(db):
    w = TelemetryV2Writer(db)
    try:
        w.insert("responses", _response(), stamp="ts")
        assert w.flush(timeout=5.0) is True
        assert _rows(db, "SELECT COUNT(*) FROM responses") == [(1,)]
        w.insert("responses", _response(1), stamp="ts")   # still open
    finally:
        w.close()
    assert _rows(db, "SELECT COUNT(*) FROM responses") == [(2,)]


def test_a_full_queue_drops_and_counts_the_drops_into_silent_failures(db, caplog):
    lock = _Lock(db)
    w = TelemetryV2Writer(db, maxsize=1)
    with caplog.at_level(logging.WARNING, logger="prometheus.telemetry.writer"):
        results = [w.insert("responses", _response(i), stamp="ts") for i in range(6)]
        lock.release()
        w.close()
    kept, dropped = results.count(True), results.count(False)
    assert dropped >= 3          # one in flight, one queued, the rest refused
    assert w.dropped == dropped
    assert _rows(db, "SELECT COUNT(*) FROM responses") == [(kept,)]
    rows = _rows(db, "SELECT subsystem, operation, exception_type, context FROM silent_failures")
    assert len(rows) == 1
    subsystem, operation, exc_type, context = rows[0]
    assert (subsystem, operation, exc_type) == ("telemetry_writer", "queue_full", "QueueFull")
    assert json.loads(context) == {"dropped": dropped, "tables": {"responses": dropped}}
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_a_refused_write_is_a_warning_and_a_silent_failure_and_its_neighbours_land(db, caplog):
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TRIGGER refuse BEFORE INSERT ON responses WHEN NEW.round_index = 1 "
                 "BEGIN SELECT RAISE(ABORT, 'synthetic refusal'); END")
    conn.commit()
    conn.close()
    w = TelemetryV2Writer(db)
    with caplog.at_level(logging.WARNING, logger="prometheus.telemetry.writer"):
        for i in range(3):
            w.insert("responses", _response(i), stamp="ts")
        w.close()
    assert _rows(db, "SELECT round_index FROM responses ORDER BY id") == [(0,), (2,)]
    rows = _rows(db, "SELECT subsystem, operation, exception_msg, context FROM silent_failures")
    assert len(rows) == 1
    assert rows[0][:2] == ("telemetry_writer", "write")
    assert "synthetic refusal" in rows[0][2]
    assert json.loads(rows[0][3])["table"] == "responses"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_a_write_after_close_is_counted_not_raised(db):
    w = TelemetryV2Writer(db)
    w.close()
    assert w.insert("responses", _response(), stamp="ts") is False
    assert w.dropped == 1
    w.close()  # idempotent


def test_upsert_merges_the_end_of_a_turn_into_its_start(db):
    w = TelemetryV2Writer(db)
    w.upsert("turns", {"turn_id": "desktop:s:t", "session_id": "desktop:s", "surface": "beacon"},
             key="turn_id", stamp="started_at")
    w.upsert("turns", {"turn_id": "desktop:s:t", "session_id": "desktop:s", "rounds": 3,
                       "forced_stop_reason": "iteration_limit"},
             key="turn_id", stamp="ended_at")
    w.close()
    rows = _rows(db, "SELECT surface, rounds, forced_stop_reason, started_at IS NOT NULL, "
                     "ended_at IS NOT NULL FROM turns")
    assert rows == [("beacon", 3, "iteration_limit", 1, 1)]


def test_a_tool_set_is_stored_once_and_keyed_by_its_names_in_any_order(db):
    a = tool_set_row(["bash", "read_file", "grep"])
    b = tool_set_row(["grep", "bash", "read_file", "bash"])
    assert a == b
    assert json.loads(a["tool_names"]) == ["bash", "grep", "read_file"]
    assert a["tool_count"] == 3
    assert tool_set_row(["bash"])["tool_set_hash"] != a["tool_set_hash"]
    w = TelemetryV2Writer(db)
    w.insert("tool_sets", a, stamp="created_at", or_ignore=True)
    w.insert("tool_sets", b, stamp="created_at", or_ignore=True)
    w.close()
    assert _rows(db, "SELECT tool_set_hash, tool_count FROM tool_sets") == [
        (a["tool_set_hash"], 3)]


def test_the_prose_column_is_redacted_before_it_is_stored(db):
    token = "ghp_" + "Abcdefghij" * 4
    w = TelemetryV2Writer(db)
    w.insert("responses", _response(prose="here: " + token), stamp="ts")
    w.close()
    (prose,) = _rows(db, "SELECT prose FROM responses")[0]
    assert token not in prose


def test_the_tracker_starts_no_writer_until_one_is_asked_for(tmp_path):
    before = {t.name for t in threading.enumerate()}
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    tel.record(model="m", tool_name="bash", success=True)
    assert {t.name for t in threading.enumerate()} == before
    w = tel.v2_writer()
    assert w is tel.v2_writer()
    w.insert("responses", _response(), stamp="ts")
    tel.close()   # closing the tracker drains its writer
    assert _rows(tmp_path / "telemetry.db", "SELECT COUNT(*) FROM responses") == [(1,)]
    assert not any(t.name == TelemetryV2Writer.THREAD_NAME for t in threading.enumerate())
