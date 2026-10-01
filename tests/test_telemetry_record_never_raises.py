"""``ToolCallTelemetry.record()`` never raises into its caller (WP-X.54 T-1).

The agent loop calls ``record()`` inside ``_log_iteration`` and on every tool
execution path. A raise there ended the turn, or reported a tool call that
succeeded as a failure (audit 2026-09-30, "What changes the plan" item 3). A
failed write is now what it already was for ``record_signal_event``: a WARNING
plus a ``silent_failures`` row.

The failures are made by the database itself (a trigger that refuses the
insert), not by patching the tracker, so the test sees what a real refusal does.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from prometheus.telemetry.tracker import ToolCallTelemetry


def _refuse(db: Path, table: str, message: str) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute(
        f"CREATE TRIGGER refuse_{table} BEFORE INSERT ON {table} "
        f"BEGIN SELECT RAISE(ABORT, '{message}'); END"
    )
    conn.commit()
    conn.close()


def _rows(db: Path, sql: str) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def test_a_refused_insert_is_a_warning_and_a_silent_failure_row(tmp_path, caplog):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    _refuse(db, "tool_calls", "synthetic refusal")

    with caplog.at_level(logging.WARNING, logger="prometheus.telemetry.tracker"):
        tel.record(model="m", tool_name="bash", success=True, session_id="desktop:s")

    assert any("record" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)
    rows = _rows(db, "SELECT subsystem, operation, exception_type, exception_msg, context "
                     "FROM silent_failures")
    assert len(rows) == 1
    subsystem, operation, exc_type, msg, context = rows[0]
    assert (subsystem, operation) == ("telemetry", "record")
    assert exc_type == "IntegrityError"
    assert "synthetic refusal" in msg
    assert "bash" in context
    assert _rows(db, "SELECT COUNT(*) FROM tool_calls") == [(0,)]
    tel.close()


def test_when_the_silent_failure_row_is_refused_too_it_still_does_not_raise(tmp_path, caplog):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    _refuse(db, "tool_calls", "synthetic refusal")
    _refuse(db, "silent_failures", "also refused")

    with caplog.at_level(logging.WARNING, logger="prometheus.telemetry.tracker"):
        tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")

    assert sum(r.levelno == logging.WARNING for r in caplog.records) >= 2
    assert _rows(db, "SELECT COUNT(*) FROM silent_failures") == [(0,)]
    tel.close()


def test_a_failed_record_leaves_no_half_written_transaction(tmp_path):
    # After a refusal the connection must be usable and must not later commit
    # the refused row along with the next good one.
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    _refuse(db, "tool_calls", "synthetic refusal")
    tel.record(model="m", tool_name="first", success=True)
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TRIGGER refuse_tool_calls")
    conn.commit()
    conn.close()
    tel.record(model="m", tool_name="second", success=True)
    assert _rows(db, "SELECT tool_name FROM tool_calls") == [("second",)]
    tel.close()


def test_record_stores_the_new_v2_fields(tmp_path):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    tel.record(
        model="m", tool_name="bash", success=False, error_type="nonzero_exit",
        session_id="desktop:s", turn_id="desktop:s:abc", round_index=3,
        tool_use_id="toolu_x1", repair_kind="type_coerce",
        raw_before_repair='{"name": "bash", "input": {"command": 1}}', retry_index=1,
        result_summary="exit 2",
    )
    row = _rows(db, "SELECT turn_id, round_index, tool_use_id, repair_kind, "
                    "raw_before_repair, retry_index, result_summary FROM tool_calls")[0]
    assert row == ("desktop:s:abc", 3, "toolu_x1", "type_coerce",
                   '{"name": "bash", "input": {"command": 1}}', 1, "exit 2")
    tel.close()


def test_the_new_text_fields_are_redacted_and_the_summary_is_capped(tmp_path):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    # A GitHub-shaped token, built by concatenation so no literal secret sits
    # in the source. Letters only after the prefix.
    token = "ghp_" + "Abcdefghij" * 4
    tel.record(
        model="m", tool_name="bash", success=True,
        raw_before_repair='{"name": "bash", "input": {"command": "echo ' + token + '"}}',
        result_summary=("x" * 600) + token,
    )
    raw, summary = _rows(db, "SELECT raw_before_repair, result_summary FROM tool_calls")[0]
    assert token not in raw
    assert len(summary) == 500
    tel.close()
    # Redacted BEFORE the cut: a token straddling position 500 is not left as
    # a matchable-looking stub, and a token wholly inside the kept part is gone.
    db2 = tmp_path / "t2.db"
    tel2 = ToolCallTelemetry(db2)
    tel2.record(model="m", tool_name="bash", success=True, result_summary="out " + token)
    (summary2,) = _rows(db2, "SELECT result_summary FROM tool_calls")[0]
    assert token not in summary2
    tel2.close()


def test_record_run_returns_the_id_of_the_row_it_wrote(tmp_path):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    run_id = tel.record_run("agent_loop", "loop_round", "success", 10.0,
                            round_index=0, session_id="desktop:s")
    assert run_id is not None
    assert _rows(db, "SELECT id FROM subsystem_runs") == [(run_id,)]
    _refuse(db, "subsystem_runs", "synthetic refusal")
    assert tel.record_run("agent_loop", "loop_round", "success") is None
    tel.close()
