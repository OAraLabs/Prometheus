"""Trace labels: tool_ok, turn_outcome and judged_ok, kept apart (telemetry/trace_labels.py).

"Pass" on a stored trace meant the tool call ran without error and was read
as task success (docs/sprints/TELEMETRY-ACCURACY-AUDIT.md). Pinned here: the
additive migration on both trace stores leaves every existing value alone;
the backfill derives turn_outcome from the turn's end state, says unknown
where it cannot link, never overwrites a label, and writes nothing on a dry
run; a turn's end stamps its calls and pairs; a turn that never ended is
called abandoned only after the horizon; a judgement is stored with who made it.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from prometheus.learning.pair_capture import PairStore
from prometheus.telemetry import trace_labels as tl
from prometheus.telemetry.tracker import ToolCallTelemetry

NOW = 2_000_000_000.0
OLD = NOW - tl.ABANDONED_AFTER_SECONDS - 60


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)


# --------------------------------------------------------------------------- #
# the end-state rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("ended", "reason", "kind", "want"), [
    (NOW, None, "prose", tl.COMPLETED),
    (NOW, None, None, tl.COMPLETED),
    (NOW, "circuit_breaker_trip", "tool_call", tl.STOPPED),
    (NOW, "cancelled", "tool_call", tl.STOPPED),
    (NOW, "max_iterations_hit", "tool_call", tl.STOPPED),
    (NOW, "provider_error", None, tl.ERRORED),
    (NOW, "degenerate_output", "prose", tl.ERRORED),
    (NOW, None, "error", tl.ERRORED),
])
def test_an_ended_turns_outcome_is_its_end_state(ended, reason, kind, want):
    assert tl.turn_outcome(started_at=NOW - 5, ended_at=ended, forced_stop_reason=reason,
                           terminal_kind=kind, now=NOW) == want


def test_a_turn_with_no_end_is_running_until_the_horizon_then_abandoned():
    assert tl.turn_outcome(started_at=NOW - 60, ended_at=None, forced_stop_reason=None,
                           terminal_kind=None, now=NOW) is None
    assert tl.turn_outcome(started_at=OLD, ended_at=None, forced_stop_reason=None,
                           terminal_kind=None, now=NOW) == tl.ABANDONED


def test_the_sql_rule_matches_the_python_rule(tmp_path):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE turns (turn_id TEXT, forced_stop_reason TEXT, terminal_kind TEXT)")
    from prometheus.telemetry.outcomes import ERROR_TERMINAL_REASONS, FORCED_STOP_REASONS

    cases = [(r, k) for r in [None, *FORCED_STOP_REASONS, *ERROR_TERMINAL_REASONS]
             for k in (None, "prose", "tool_call", "error")]
    for i, (r, k) in enumerate(cases):
        con.execute("INSERT INTO turns VALUES (?, ?, ?)", (str(i), r, k))
    sql, params = tl._turn_outcome_sql("t")
    got = dict(con.execute(f"SELECT t.turn_id, {sql} FROM turns t", params).fetchall())
    for i, (r, k) in enumerate(cases):
        assert got[str(i)] == tl.turn_outcome(
            started_at=NOW, ended_at=NOW, forced_stop_reason=r, terminal_kind=k), (r, k)


def test_the_user_signals_abandoned_is_not_the_end_states_abandoned():
    """turns.outcome = 'abandoned' (no reply within 30 min) is neutral; a turn
    that ended on its own is completed here whatever the next message did."""
    assert tl.turn_outcome(started_at=NOW - 5, ended_at=NOW, forced_stop_reason=None,
                           terminal_kind="prose") == tl.COMPLETED


# --------------------------------------------------------------------------- #
# migration: additive, never rewrites
# --------------------------------------------------------------------------- #


def _old_telemetry(path: Path) -> None:
    """A tool_calls table as it was before the labels (and before v2)."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE tool_calls (
            id TEXT PRIMARY KEY, timestamp REAL NOT NULL, model TEXT NOT NULL,
            tool_name TEXT NOT NULL, success INTEGER NOT NULL,
            retries INTEGER NOT NULL DEFAULT 0, latency_ms REAL, error_type TEXT,
            error_detail TEXT);
        INSERT INTO tool_calls VALUES ('a', 1.0, 'm', 'bash', 1, 0, 3.0, NULL, NULL);
        INSERT INTO tool_calls VALUES ('b', 2.0, 'm', 'bash', 0, 0, 4.0, 'tool_error', 'boom');
    """)
    con.commit()
    con.close()


def test_the_telemetry_migration_adds_the_label_columns_and_rewrites_nothing(tmp_path):
    db = tmp_path / "telemetry.db"
    _old_telemetry(db)
    before = sqlite3.connect(db).execute(
        "SELECT id, timestamp, model, tool_name, success, retries, latency_ms, error_type,"
        " error_detail FROM tool_calls ORDER BY id").fetchall()
    ToolCallTelemetry(db).close()
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(tool_calls)")}
    assert {"tool_ok", "turn_outcome", "judged_ok", "judged_by", "judged_at", "request_id"} <= cols
    after = con.execute(
        "SELECT id, timestamp, model, tool_name, success, retries, latency_ms, error_type,"
        " error_detail FROM tool_calls ORDER BY id").fetchall()
    assert after == before
    assert con.execute("SELECT tool_ok, turn_outcome, judged_ok FROM tool_calls").fetchall() \
        == [(None, None, None)] * 2, "opening must not backfill: that is `oara trace-labels`"
    # Twice is a no-op.
    ToolCallTelemetry(db).close()


def test_the_training_migration_adds_the_label_columns_and_rewrites_nothing(tmp_path):
    db = tmp_path / "training.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE training_pairs (
            id TEXT PRIMARY KEY, timestamp REAL NOT NULL, pair_source TEXT NOT NULL,
            model_id TEXT NOT NULL, tool_name TEXT NOT NULL, context TEXT, rejected TEXT,
            chosen TEXT NOT NULL, meta TEXT, context_hash TEXT NOT NULL UNIQUE);
        INSERT INTO training_pairs VALUES ('p', 1.0, 'cloud_golden', 'm', 'bash', '{}', NULL,
            '{"name":"bash","input":{}}', '{}', 'h1');
    """)
    con.commit()
    con.close()
    PairStore(db).close()
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(training_pairs)")}
    assert {"tool_ok", "turn_outcome", "judged_ok", "judged_by", "judged_at"} <= cols
    assert con.execute("SELECT id, pair_source, chosen, context_hash, tool_ok, turn_outcome"
                       " FROM training_pairs").fetchall() == [
        ("p", "cloud_golden", '{"name":"bash","input":{}}', "h1", None, None)]


def test_a_new_call_row_carries_tool_ok(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    tel.record(model="m", tool_name="bash", success=True)
    tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")
    rows = tel._conn.execute("SELECT success, tool_ok, turn_outcome FROM tool_calls"
                             " ORDER BY rowid").fetchall()
    tel.close()
    assert rows == [(1, 1, None), (0, 0, None)]


# --------------------------------------------------------------------------- #
# backfill
# --------------------------------------------------------------------------- #


def _seed_backfill(tmp_path: Path) -> tuple[Path, Path]:
    tdb = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(tdb)
    c = tel._conn
    turns = [
        ("s:done", NOW - 100, NOW - 90, None, "prose", "abandoned"),     # user signal: neutral
        ("s:stop", NOW - 100, NOW - 90, "circuit_breaker_trip", "tool_call", "forced_stop"),
        ("s:err", NOW - 100, NOW - 90, "provider_error", None, "error_terminal"),
        ("s:cut", OLD, None, None, None, None),                           # never ended
        ("s:run", NOW - 30, None, None, None, None),                      # still running
    ]
    for tid, st, en, reason, kind, outcome in turns:
        c.execute("INSERT INTO turns (turn_id, session_id, started_at, ended_at,"
                  " forced_stop_reason, terminal_kind, outcome) VALUES (?, 's', ?, ?, ?, ?, ?)",
                  (tid, st, en, reason, kind, outcome))
    calls = [
        ("c1", "s:done", NOW - 95, 1), ("c2", "s:stop", NOW - 95, 1), ("c3", "s:err", NOW - 95, 0),
        ("c4", "s:cut", OLD + 1, 1), ("c5", "s:run", NOW - 20, 1),
        ("c6", None, 5.0, 1),                    # pre-v2: no turn id
        ("c7", "s:gone", OLD, 1),                # turn row never written, long ago
        ("c8", "s:new", NOW - 1, 1),             # turn row not landed yet
    ]
    for cid, tid, ts, ok in calls:
        c.execute("INSERT INTO tool_calls (id, timestamp, model, tool_name, success, turn_id)"
                  " VALUES (?, ?, 'm', 'bash', ?, ?)", (cid, ts, ok, tid))
    # Already labelled: must survive.
    c.execute("INSERT INTO tool_calls (id, timestamp, model, tool_name, success, turn_id,"
              " tool_ok, turn_outcome) VALUES ('c9', ?, 'm', 'bash', 1, 's:stop', 1, 'completed')",
              (NOW - 95,))
    c.commit()
    tel.close()

    pdb = tmp_path / "training.db"
    store = PairStore(pdb)
    for i, (source, tid, meta) in enumerate([
        ("cloud_golden", "s:stop", None),
        ("retry_success", "s:done", None),
        ("schema_repair", "s:done", None),               # banked before it ran
        ("gym_harvest", None, {"origin_source": "self_correction"}),
        ("gym_harvest", None, {"origin_source": "levenshtein_repair"}),
    ]):
        store.add_pair(pair_source=source, model_id="m", tool_name="bash",
                       context={"i": i}, rejected=None if source == "cloud_golden" else {
                           "name": "bash", "input": {"i": i}},
                       chosen={"name": "bash", "input": {}}, meta=meta, turn_id=tid)
    store._conn.execute("UPDATE training_pairs SET timestamp = ?", (OLD,))
    store._conn.commit()
    store.close()
    return tdb, pdb


def _labels(db: Path, table: str, key: str = "id") -> dict:
    con = sqlite3.connect(db)
    return {r[0]: (r[1], r[2]) for r in con.execute(
        f"SELECT {key}, tool_ok, turn_outcome FROM {table}")}


def test_the_telemetry_backfill_labels_from_the_turns_end_state(tmp_path):
    tdb, _ = _seed_backfill(tmp_path)
    con = sqlite3.connect(tdb)
    report = tl.backfill_telemetry(con, apply=True, now=NOW)
    con.close()
    got = _labels(tdb, "tool_calls")
    assert got == {
        "c1": (1, "completed"), "c2": (1, "stopped"), "c3": (0, "errored"),
        "c4": (1, "abandoned"), "c5": (1, None), "c6": (1, "unknown"),
        "c7": (1, "unknown"), "c8": (1, None), "c9": (1, "completed"),
    }
    assert report.still_running == 2
    assert report.turn_outcome == {"completed": 1, "stopped": 1, "errored": 1,
                                   "abandoned": 1, "unknown": 2}


def test_a_dry_run_writes_nothing(tmp_path):
    tdb, pdb = _seed_backfill(tmp_path)
    before = (tdb.read_bytes(), pdb.read_bytes())
    tcon = sqlite3.connect(f"file:{tdb}?mode=ro", uri=True)
    pcon = sqlite3.connect(f"file:{pdb}?mode=ro", uri=True)
    r1 = tl.backfill_telemetry(tcon, apply=False, now=NOW)
    r2 = tl.backfill_training(pcon, tcon, apply=False, now=NOW)
    tcon.close()
    pcon.close()
    assert r1.rows == 8 and r2.rows == 5
    assert (tdb.read_bytes(), pdb.read_bytes()) == before


def test_the_backfill_is_idempotent_and_never_overwrites(tmp_path):
    tdb, _ = _seed_backfill(tmp_path)
    con = sqlite3.connect(tdb)
    tl.backfill_telemetry(con, apply=True, now=NOW)
    first = _labels(tdb, "tool_calls")
    again = tl.backfill_telemetry(con, apply=True, now=NOW)
    con.close()
    assert _labels(tdb, "tool_calls") == first
    # Only the two still-running rows are re-read; both stay NULL.
    assert again.rows == 2 and again.still_running == 2


def test_the_training_backfill_labels_pairs_from_telemetry(tmp_path):
    tdb, pdb = _seed_backfill(tmp_path)
    tcon, pcon = sqlite3.connect(tdb), sqlite3.connect(pdb)
    tl.backfill_training(pcon, tcon, apply=True, now=NOW)
    tcon.close()
    rows = pcon.execute("SELECT pair_source, meta, tool_ok, turn_outcome FROM training_pairs"
                        " ORDER BY rowid").fetchall()
    pcon.close()
    assert [(s, ok, o) for s, _m, ok, o in rows] == [
        ("cloud_golden", 1, "stopped"),
        ("retry_success", 1, "completed"),
        ("schema_repair", None, "completed"),      # banked before the call ran: not known
        ("gym_harvest", 1, "unknown"),             # origin self_correction ran; no turn id
        ("gym_harvest", None, "unknown"),
    ]


def test_without_telemetry_every_pair_is_unknown(tmp_path):
    _, pdb = _seed_backfill(tmp_path)
    pcon = sqlite3.connect(pdb)
    report = tl.backfill_training(pcon, None, apply=True, now=NOW)
    pcon.close()
    assert report.turn_outcome == {"unknown": 5}


def test_the_backfill_refuses_a_store_without_the_columns(tmp_path):
    db = tmp_path / "telemetry.db"
    _old_telemetry(db)
    with pytest.raises(RuntimeError, match="no label columns"):
        tl.backfill_telemetry(sqlite3.connect(db), apply=False)


# --------------------------------------------------------------------------- #
# oara trace-labels
# --------------------------------------------------------------------------- #


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "prometheus", "trace-labels", *args],
                          capture_output=True, text=True, timeout=120)


def test_the_cli_dry_run_counts_and_touches_nothing(tmp_path):
    tdb, pdb = _seed_backfill(tmp_path)
    before = (tdb.read_bytes(), pdb.read_bytes())
    r = _cli("--telemetry", str(tdb), "--training", str(pdb), "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout[r.stdout.index("{"):])
    assert out["apply"] is False
    assert {s["store"] for s in out["stores"]} == {
        "telemetry.db/tool_calls", "training.db/training_pairs"}
    assert (tdb.read_bytes(), pdb.read_bytes()) == before
    assert not list(tmp_path.glob("*.pre-trace-labels-*"))


def test_the_cli_apply_backs_up_then_labels(tmp_path):
    tdb, pdb = _seed_backfill(tmp_path)
    r = _cli("--telemetry", str(tdb), "--training", str(pdb), "--apply")
    assert r.returncode == 0, r.stderr
    assert len(list(tmp_path.glob("telemetry.db.pre-trace-labels-*"))) == 1
    assert len(list(tmp_path.glob("training.db.pre-trace-labels-*"))) == 1
    # The labels are written; with the real clock every unended turn is past the horizon or not.
    labels = _labels(tdb, "tool_calls")
    assert labels["c6"] == (1, "unknown") and labels["c2"] == (1, "stopped")
    backup = next(tmp_path.glob("telemetry.db.pre-trace-labels-*"))
    assert sqlite3.connect(backup).execute(
        "SELECT COUNT(*) FROM tool_calls WHERE turn_outcome IS NOT NULL").fetchone()[0] == 1


# --------------------------------------------------------------------------- #
# ongoing labelling: turn end, the sweep, judgements
# --------------------------------------------------------------------------- #


def test_a_turns_end_labels_its_calls_and_pairs(tmp_path):
    from prometheus.engine.agent_loop import _TurnRecord
    from prometheus.learning import pair_capture

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    pair_capture.configure({"db_path": str(tmp_path / "training.db")})
    try:
        writer = tel.v2_capture_writer()
        rec = _TurnRecord(turn_id="s:t1", session_id="s", surface="beacon", mode="agent",
                          coding_run_id=None, ephemeral=False, writer=writer)
        rec.start(None)
        tel.record(model="m", tool_name="bash", success=True, turn_id="s:t1")
        tel.record(model="m", tool_name="bash", success=True, turn_id="s:other")
        pair_capture.capture_pair(pair_source="retry_success", model_id="m", tool_name="bash",
                                  context={}, rejected={"name": "bash", "input": {"x": 1}},
                                  chosen={"name": "bash", "input": {}}, turn_id="s:t1",
                                  tool_ok=True)
        rec.stop("circuit_breaker_trip")
        rec.end()
        assert writer.flush()
        rows = tel._conn.execute("SELECT turn_id, turn_outcome FROM tool_calls ORDER BY rowid"
                                 ).fetchall()
        assert rows == [("s:t1", "stopped"), ("s:other", None)]
        pair = pair_capture.get_store()._conn.execute(
            "SELECT tool_ok, turn_outcome FROM training_pairs").fetchone()
        assert tuple(pair) == (1, "stopped")
    finally:
        pair_capture.configure({"capture_enabled": False})
        tel.close()


def test_the_sweep_calls_a_long_unended_turn_abandoned(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    c = tel._conn
    c.execute("INSERT INTO turns (turn_id, session_id, started_at) VALUES ('s:old', 's', ?)",
              (time.time() - tl.ABANDONED_AFTER_SECONDS - 10,))
    c.execute("INSERT INTO turns (turn_id, session_id, started_at) VALUES ('s:new', 's', ?)",
              (time.time() - 10,))
    for tid in ("s:old", "s:new"):
        tel.record(model="m", tool_name="bash", success=True, turn_id=tid)
    c.commit()
    tl.queue_sweep_unended(tel)
    assert tel.v2_writer().flush()
    got = dict(c.execute("SELECT turn_id, turn_outcome FROM tool_calls").fetchall())
    tel.close()
    assert got == {"s:old": "abandoned", "s:new": None}


def test_a_judgement_is_stored_with_its_judge_and_replaced_by_a_rejudge(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    tel.record(model="m", tool_name="bash", success=True)
    (row_id,) = tel._conn.execute("SELECT id FROM tool_calls").fetchone()
    assert tl.record_judgement(tel._conn, "tool_calls", row_id, ok=False, judged_by="rubric-v2",
                               at=10.0)
    assert tl.record_judgement(tel._conn, "tool_calls", row_id, ok=True, judged_by="rubric-v3",
                               at=20.0)
    assert not tl.record_judgement(tel._conn, "tool_calls", "missing", ok=True, judged_by="x")
    got = tel._conn.execute("SELECT tool_ok, judged_ok, judged_by, judged_at FROM tool_calls"
                            ).fetchone()
    tel.close()
    assert tuple(got) == (1, 1, "rubric-v3", 20.0)
    with pytest.raises(ValueError):
        tl.record_judgement(sqlite3.connect(":memory:"), "turns", "x", ok=True, judged_by="j")
    with pytest.raises(ValueError):
        tl.record_judgement(sqlite3.connect(":memory:"), "tool_calls", "x", ok=True, judged_by="")


def test_outcome_rates_keep_unknown_and_pending_out_of_the_denominator():
    rates = tl.outcome_rates({"completed": 6, "stopped": 2, "errored": 1, "abandoned": 1,
                              "unknown": 90, None: 3})
    assert rates["linked"] == 10 and rates["completed_rate"] == 0.6
    assert rates["unknown"] == 90 and rates["pending"] == 3
    assert tl.outcome_rates({"unknown": 5})["completed_rate"] is None


def test_a_full_turn_through_the_loop_labels_its_call(tmp_path):
    """End to end: run_loop → the call row's turn_outcome is the turn's end state."""
    from tests.test_fallback_attribution import _run, _Scripted

    con = _run(tmp_path, primary=_Scripted("bash", provider_name="qwen"))
    rows = con.execute("SELECT tool_ok, turn_outcome FROM tool_calls"
                       " WHERE tool_name = 'bash'").fetchall()
    assert [tuple(r) for r in rows] == [(1, "completed")]
