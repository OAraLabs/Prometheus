"""Every consumer the accuracy audit flagged reads the right field and counts once.

docs/sprints/TELEMETRY-ACCURACY-AUDIT.md lists them. Pinned here, per caller:
a tool-call rate is named as one and a turn-level number (how turns ended)
sits beside it; /tools judges calls over the same denominator as /health;
the digest's baseline ends where its window starts; /pairs never presents
chosen-only cloud copies as pairs; the doctor reads the tracker's DB and
counts only the incremental exports; a gym run says when its pass is
tool-only; the two script pass checks measure something.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry, set_telemetry_handle


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)


def _turn(tel, tid, *, started, ended=None, reason=None, kind="prose", outcome=None):
    tel._conn.execute(
        "INSERT INTO turns (turn_id, session_id, started_at, ended_at, forced_stop_reason,"
        " terminal_kind, outcome) VALUES (?, 's', ?, ?, ?, ?, ?)",
        (tid, started, ended, reason, kind, outcome))
    tel._conn.commit()


def _seed_turns(tel, now):
    _turn(tel, "a", started=now - 50, ended=now - 40, outcome="accepted_user")
    _turn(tel, "b", started=now - 50, ended=now - 40, outcome="abandoned")
    _turn(tel, "c", started=now - 50, ended=now - 40, reason="circuit_breaker_trip",
          kind="tool_call", outcome="forced_stop")
    _turn(tel, "d", started=now - 50, ended=now - 40, reason="provider_error", kind=None,
          outcome="error_terminal")
    _turn(tel, "e", started=now - 10)                       # running


# --------------------------------------------------------------------------- #
# report() / /api/telemetry
# --------------------------------------------------------------------------- #


def test_report_names_its_rates_as_tool_call_rates_and_adds_turns(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    now = time.time()
    _seed_turns(tel, now)
    tel.record(model="m", tool_name="bash", success=True)
    tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")
    rep = tel.report()
    tel.close()
    assert rep["overall_tool_ok_rate"] == rep["overall_success_rate"] == 0.5
    assert rep["tools"]["bash"]["tool_ok_rate"] == 0.5
    assert rep["models"]["m"]["bash"]["tool_ok_rate"] == 0.5
    turns = rep["turns"]
    assert (turns["completed"], turns["stopped"], turns["errored"], turns["pending"]) == (2, 1, 1, 1)
    assert turns["completed_rate"] == 0.5
    # The user signal rides along, abandoned (neutral) included, apart from the end state.
    assert turns["user_signal"]["abandoned"] == 1 and turns["user_signal"]["none"] == 1


def test_report_until_bounds_the_window(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "t.db")
    tel.record(model="m", tool_name="bash", success=True)
    cut = time.time() + 0.001
    time.sleep(0.01)
    tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")
    assert tel.report(until=cut)["total_calls"] == 1
    assert tel.report(since=cut)["total_calls"] == 1
    tel.close()


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #


def test_health_says_calls_ran_and_shows_how_turns_ended(tmp_path):
    from prometheus.gateway.commands import cmd_health

    tel = ToolCallTelemetry(tmp_path / "t.db")
    set_telemetry_handle(tel)
    try:
        now = time.time()
        _seed_turns(tel, now)
        tel.record(model="m", tool_name="bash", success=True)
        tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")
        text = cmd_health()
    finally:
        set_telemetry_handle(None)
        tel.close()
    calls = next(ln for ln in text.splitlines() if "Tool calls:" in ln)
    assert "50.0% ran OK" in calls and "success" not in calls
    turns = next(ln for ln in text.splitlines() if "Turns ended:" in ln)
    assert "2 completed, 1 stopped, 1 errored, 0 cut off" in turns


# --------------------------------------------------------------------------- #
# /tools
# --------------------------------------------------------------------------- #


def test_tools_judges_calls_over_the_same_denominator_as_health(tmp_path):
    from prometheus.telemetry.dashboard import ToolDashboard

    db = tmp_path / "t.db"
    tel = ToolCallTelemetry(db)
    tel.record(model="m", tool_name="bash", success=True)
    tel.record(model="m", tool_name="bash", success=False, error_type="nonzero_exit")
    tel.record(model="m", tool_name="bash", success=False, error_type="permission_denied")
    health = tel.health_summary(since=0)["tool_calls"]["success_rate"]
    tel.close()
    stats = ToolDashboard(db).get_stats(hours=1)
    assert stats["overall_success_rate"] == health == 1.0
    assert stats["success_rate_by_tool"]["bash"] == 1.0
    assert stats["total_calls"] == 3 and stats["total_denials"] == 1


def test_tools_reads_the_trackers_db_by_default(tmp_path, monkeypatch):
    from prometheus.telemetry.dashboard import ToolDashboard

    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    dash = ToolDashboard()
    try:
        assert dash._db_path == (tmp_path / "cfg" / "telemetry.db").resolve()
    finally:
        dash.close()


# --------------------------------------------------------------------------- #
# the digest
# --------------------------------------------------------------------------- #


def test_the_digest_baseline_ends_where_its_window_starts(tmp_path):
    from prometheus.sentinel.telemetry_digest import TelemetryDigest

    tel = ToolCallTelemetry(tmp_path / "t.db")
    old = time.time() - 3 * 86400
    # Baseline: 10 good calls three days ago. Window: 10 bad calls now.
    for _ in range(10):
        tel.record(model="m", tool_name="bash", success=True)
    tel._conn.execute("UPDATE tool_calls SET timestamp = ?", (old,))
    tel._conn.commit()
    for _ in range(10):
        tel.record(model="m", tool_name="bash", success=False, error_type="tool_error")
    result = TelemetryDigest(tel).generate()
    tel.close()
    [drop] = [a for a in result.anomalies if a.metric == "success_rate_drop"]
    # Overlapping, the baseline would read 50% (10 good + the 10 bad it is compared with).
    assert drop.baseline_value == 1.0 and drop.current_value == 0.0
    assert "ran without error" in result.summary and "overall success rate" not in result.summary


# --------------------------------------------------------------------------- #
# /pairs
# --------------------------------------------------------------------------- #


def test_pair_stats_split_pairs_from_cloud_copies(tmp_path):
    from prometheus.learning.pair_capture import PairStore

    store = PairStore(tmp_path / "training.db")
    store.add_pair(pair_source="retry_success", model_id="m", tool_name="bash", context={"a": 1},
                   rejected={"name": "bash", "input": {"x": 1}}, chosen={"name": "bash", "input": {}})
    for i in range(3):
        store.add_pair(pair_source="cloud_golden", model_id="m", tool_name="bash",
                       context={"i": i}, rejected=None, chosen={"name": "bash", "input": {}})
    stats = store.stats()
    store.close()
    assert (stats["total"], stats["pairs"], stats["exemplars"]) == (4, 1, 3)
    assert "telemetry" in stats["exemplars_note"]
    assert stats["by_turn_outcome"] == {"unlabelled": 4}


def test_the_pairs_command_never_calls_cloud_copies_pairs(monkeypatch):
    from unittest.mock import MagicMock

    import prometheus.learning.pair_capture as pc
    from prometheus.gateway.commands import cmd_pairs

    store = MagicMock()
    store.stats.return_value = {"total": 4, "pairs": 1, "exemplars": 3, "last_7d": 4,
                                "per_day_7d": 0.57, "by_source": {}, "by_tool": {}}
    monkeypatch.setattr(pc, "get_store", lambda: store)
    text = cmd_pairs()
    assert "Pairs: 1" in text and "Total" not in text
    assert "Cloud exemplars: 3 (copies of golden calls; not a separate corpus)" in text


# --------------------------------------------------------------------------- #
# oara doctor
# --------------------------------------------------------------------------- #


def test_the_doctor_counts_only_incremental_exports_from_the_trackers_db(tmp_path, monkeypatch):
    from prometheus.cli.doctor import check_trajectory_export

    cfg = tmp_path / "cfg"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(cfg))
    tel = ToolCallTelemetry(cfg / "telemetry.db")
    tel.record(model="m", tool_name="bash", success=True, raw_model_output="x", provider="qwen")
    tel.close()
    out = tmp_path / "traj"
    out.mkdir()
    (out / "golden_traces_1_1.jsonl").write_text("{}\n")
    (out / "golden_dump_2.jsonl").write_text("{}\n")          # a CLI dump: same rows again
    check = check_trajectory_export({"trajectory_export": {"enabled": True,
                                                           "output_dir": str(out)}})
    assert "1 export file" in check.message and "1 trace(s) awaiting" in check.message


def test_the_doctor_reads_a_missing_flag_as_off_like_the_daemon(tmp_path, monkeypatch):
    from prometheus.cli.doctor import check_trajectory_export

    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    check = check_trajectory_export({"trajectory_export": {"output_dir": str(tmp_path / "t")}})
    assert "DISABLED" in check.message


# --------------------------------------------------------------------------- #
# gym
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("score", "basis"), [
    ({"expect_tool": "bash"}, "tool_only"),
    ({"expect_tool": "task_create", "expect_tool_args_present": ["command"],
      "require_graceful": True}, "tool_only"),
    ({"expect_tool": "write_file", "expect_file": "out.txt"}, "outcome"),
    ({"expect_text_contains": "42"}, "outcome"),
])
def test_a_gym_task_says_whether_its_pass_is_tool_only(score, basis):
    from prometheus.gym.scoring import pass_basis

    assert pass_basis(score) == basis


def test_the_gym_report_says_how_many_runs_are_tool_only(tmp_path):
    from prometheus.gym.report import render_report
    from prometheus.gym.store import GymStore

    store = GymStore(tmp_path / "gym.db")
    for i, basis in enumerate(["tool_only", "tool_only", "outcome", None]):
        store.record_run(series="s", experiment="e", task_id=f"t{i}", run_idx=0, model="m",
                         category="c", success=1, emission_pass=1, execution_pass=1,
                         manifest_sha="x", taskset_sha="y", pass_basis=basis)
    rows = store._conn.execute("SELECT * FROM gym_runs").fetchall()
    text = render_report("s", "e", rows)
    assert "Scored on tool-call checks only (no file/text predicate): 2/4 runs" in text
    assert "Pass basis not recorded (runs before it existed): 1" in text


def test_an_old_gym_db_gains_the_pass_basis_column(tmp_path):
    from prometheus.gym.store import GymStore

    db = tmp_path / "gym.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE gym_runs (series TEXT, experiment TEXT, task_id TEXT, run_idx INT,"
                " timestamp REAL, model TEXT, category TEXT, success INT, fail_reasons TEXT,"
                " tools_called TEXT, latency_ms REAL, retries INT, repairs INT,"
                " dropped_malformed INT, feedback_retries INT, breaker_tripped INT, error TEXT,"
                " manifest_sha TEXT, taskset_sha TEXT)")
    con.commit()
    con.close()
    GymStore(db)
    assert "pass_basis" in {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(gym_runs)")}


# --------------------------------------------------------------------------- #
# scripts
# --------------------------------------------------------------------------- #


def _load_script(name: str, monkeypatch):
    import importlib.util
    import sys

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(f"_test_{name}", scripts / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_the_web_audit_reads_the_tools_a_test_called_from_the_real_key(tmp_path, monkeypatch):
    src = (Path(__file__).resolve().parents[1] / "scripts" / "web_capability_audit.py").read_text()
    assert 'report.get("by_tool"' not in src
    tel = ToolCallTelemetry(tmp_path / "t.db")
    tel.record(model="m", tool_name="bash", success=True)

    class _Holder:
        telemetry = tel

    mod = _load_script("web_capability_audit", monkeypatch)
    counts = mod.WebAuditRunner._tool_call_counts(_Holder())  # type: ignore[arg-type]
    tel.close()
    assert counts == {"bash": 1}


def test_the_smoke_tests_telemetry_check_needs_a_recorded_call():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_tool_calling.py").read_text()
    assert "has_records = bool(stats)\n" not in src
    assert 'stats.get("total_calls")' in src
