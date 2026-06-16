"""feat/telemetry-wal — WAL + busy_timeout on telemetry.db, and the proof.

These tests pin the concurrency substrate the coding live-stream is built on.
Origin: the telemetry-write-timing survey (audits/20260616T014041Z-...) hit a
real ``database is locked (5)`` when a separate connection polled telemetry.db
while the per-round writer committed, under ``journal_mode=delete`` /
``busy_timeout=0``. This sprint flips the db to WAL and adds a busy_timeout.

All assertions run against REAL sqlite connections (no mocks) — a pragma/journal
mode is a property of an actually-opened connection, and the load-bearing test is
inherently about two real connections contending.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from prometheus.telemetry.db import (
    TELEMETRY_BUSY_TIMEOUT_MS,
    apply_telemetry_pragmas,
    connect_telemetry_db,
)
from prometheus.telemetry.dashboard import ToolDashboard
from prometheus.telemetry.tracker import ToolCallTelemetry


# ---------------------------------------------------------------------------
# Pragma side-effects — assert against a real opened connection
# ---------------------------------------------------------------------------

def _journal_mode(conn: sqlite3.Connection) -> str:
    return conn.execute("PRAGMA journal_mode").fetchone()[0]


def _busy_timeout(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA busy_timeout").fetchone()[0]


def test_connect_helper_sets_wal_and_busy_timeout(tmp_path: Path) -> None:
    conn = connect_telemetry_db(tmp_path / "telemetry.db")
    try:
        assert _journal_mode(conn) == "wal"
        assert _busy_timeout(conn) == TELEMETRY_BUSY_TIMEOUT_MS == 5000
    finally:
        conn.close()


def test_apply_pragmas_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    conn = connect_telemetry_db(db)
    try:
        # Re-applying on an already-WAL file is a harmless no-op.
        apply_telemetry_pragmas(conn)
        assert _journal_mode(conn) == "wal"
        # A second, independent connection inherits the file's WAL mode.
        other = connect_telemetry_db(db)
        try:
            assert _journal_mode(other) == "wal"
            assert _busy_timeout(other) == 5000
        finally:
            other.close()
    finally:
        conn.close()


def test_tracker_writer_connection_is_wal(tmp_path: Path) -> None:
    tel = ToolCallTelemetry(db_path=str(tmp_path / "telemetry.db"))
    try:
        assert _journal_mode(tel._conn) == "wal"
        assert _busy_timeout(tel._conn) == 5000
    finally:
        tel.close()


def test_dashboard_reader_connection_is_wal(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    # Create the schema first via the writer, then open the reader.
    ToolCallTelemetry(db_path=str(db)).close()
    dash = ToolDashboard(db_path=str(db))
    try:
        assert _journal_mode(dash._conn) == "wal"
        assert _busy_timeout(dash._conn) == 5000
    finally:
        dash._conn.close()


def test_wal_sidecars_created(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    try:
        tel.record_run(
            subsystem="agent_loop", operation="loop_round", outcome="success",
            round_index=0, session_id="coding:probe",
        )
        names = {p.name for p in tmp_path.iterdir()}
        # WAL keeps writes in the -wal sidecar (with the -shm index) until a
        # checkpoint — their presence is the observable signature of WAL mode.
        assert "telemetry.db-wal" in names
        assert "telemetry.db-shm" in names
    finally:
        tel.close()


# ---------------------------------------------------------------------------
# Behavioural no-change — a written row still reads back identically
# ---------------------------------------------------------------------------

def test_writes_are_unchanged_under_wal(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    try:
        tel.record_run(
            subsystem="agent_loop", operation="loop_round", outcome="success",
            duration_ms=12.5, input_tokens=100, output_tokens=20,
            round_index=3, session_id="coding:x", model="gemma4-26b",
            thinking=True,
        )
        tel.record(model="gemma4-26b", tool_name="code_view", success=True)
        # Read back through a fresh connection (cross-connection visibility).
        conn = connect_telemetry_db(db)
        try:
            row = conn.execute(
                "SELECT subsystem, round_index, session_id, input_tokens, "
                "output_tokens, thinking FROM subsystem_runs "
                "WHERE session_id='coding:x'"
            ).fetchone()
            assert row == ("agent_loop", 3, "coding:x", 100, 20, 1)
            tc = conn.execute(
                "SELECT model, tool_name, success FROM tool_calls"
            ).fetchone()
            assert tc == ("gemma4-26b", "code_view", 1)
        finally:
            conn.close()
    finally:
        tel.close()


# ---------------------------------------------------------------------------
# THE load-bearing test — concurrent read-during-write, no "database is locked"
# Reproduces the survey's failure scenario and proves it is fixed.
# ---------------------------------------------------------------------------

def test_concurrent_read_during_write_no_lock(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    n_rows = 40
    sid = "coding:concurrency-probe"

    # Writer's ToolCallTelemetry: created here so the schema exists before the
    # reader queries it. check_same_thread=False lets the writer thread use it.
    tel = ToolCallTelemetry(db_path=str(db))

    errors: list[tuple[str, BaseException]] = []
    observed: list[int] = []
    start = threading.Barrier(2)

    def writer() -> None:
        try:
            start.wait(timeout=5)
            for i in range(n_rows):
                tel.record_run(
                    subsystem="agent_loop", operation="loop_round",
                    outcome="success", round_index=i, session_id=sid,
                )
                time.sleep(0.005)  # ~200ms total — wide enough to interleave
        except BaseException as exc:  # noqa: BLE001 — captured for assertion
            errors.append(("writer", exc))

    def reader() -> None:
        try:
            conn = connect_telemetry_db(db)
            try:
                start.wait(timeout=5)
                deadline = time.monotonic() + 15.0
                last = -1
                while time.monotonic() < deadline:
                    cnt = conn.execute(
                        "SELECT COUNT(*) FROM subsystem_runs WHERE session_id=?",
                        (sid,),
                    ).fetchone()[0]
                    if cnt != last:
                        observed.append(cnt)
                        last = cnt
                    if cnt >= n_rows:
                        break
                    time.sleep(0.002)
            finally:
                conn.close()
        except BaseException as exc:  # noqa: BLE001 — captured for assertion
            errors.append(("reader", exc))

    tw = threading.Thread(target=writer)
    tr = threading.Thread(target=reader)
    tr.start()
    tw.start()
    tw.join(timeout=30)
    tr.join(timeout=30)
    tel.close()

    # 1. Neither side raised — critically, no sqlite3.OperationalError
    #    "database is locked" (the survey's failure).
    assert not errors, f"concurrent access raised: {errors!r}"
    assert not tw.is_alive() and not tr.is_alive(), "threads did not finish"

    # 2. The reader (a SEPARATE connection) observed every committed row —
    #    cross-connection mid-write visibility.
    assert observed, "reader observed nothing"
    assert observed[-1] == n_rows, f"reader saw {observed[-1]}, expected {n_rows}"

    # 3. The count CLIMBED rather than appearing in one jump — proves the reader
    #    saw rows accrue incrementally while the writer was committing.
    assert observed == sorted(observed), f"counts not monotonic: {observed}"
    assert len(observed) > 1, f"reader never saw an intermediate count: {observed}"
