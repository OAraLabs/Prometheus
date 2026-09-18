"""POST /api/code + GET /api/code/{id} — coding runs over the API (v2 scope 5).

The BackgroundTaskManager singleton is stubbed so these never spawn a real
subprocess; what's asserted is the wiring — validation, the command the
manager is asked to run, SecurityGate-rejection surfacing, and the status
read including the output tail.
"""

from __future__ import annotations

import json as _json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.tasks.types import TaskRecord  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _StubManager:
    def __init__(self, tmp_path: Path, *, reject: bool = False, stuck_stop: bool = False) -> None:
        self._tmp = tmp_path
        self._reject = reject
        self._stuck_stop = stuck_stop
        self.created_with: dict | None = None
        self._records: dict[str, TaskRecord] = {}

    def _record(self, status: str, error: str | None = None) -> TaskRecord:
        out = self._tmp / "task-output.log"
        # A realistic report: acceptance_exit/diff_stat/sandbox_root are REQUIRED fields on
        # CodingRunReport (no defaults), so every real run prints them. parse_coding_report keys on
        # those markers to tell the report apart from other JSON a subprocess prints — a stub
        # without them would be testing against a shape that never occurs.
        out.write_text(_json.dumps({
            "status": "success",
            "branch": "coding/x",
            "acceptance_exit": 0,
            "diff_stat": " src/a.py | 2 +-",
            "sandbox_root": "/tmp/sandbox/a12345678",
        }))
        rec = TaskRecord(
            id="a12345678",
            type="local_agent",
            status=status,
            description="coding task",
            cwd=str(self._tmp),
            output_file=out,
            command="…",
            created_at=1.0,
            error=error,
        )
        self._records[rec.id] = rec
        return rec

    async def create_shell_task(self, **kwargs) -> TaskRecord:
        self.created_with = kwargs
        if self._reject:
            return self._record("failed", error="blocked: rm -rf is not allowed")
        return self._record("running")

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    async def stop_task(self, task_id: str) -> TaskRecord:
        # Mirrors the REAL BackgroundTaskManager.stop_task: a terminal task is
        # returned as-is (idempotent — a process-less terminal run reaches the
        # `status in TERMINAL_STATUSES → return task` branch), a running one is
        # killed. The genuine ValueError("not running") case is a non-terminal,
        # process-less, waiter-less task — modelled by `stuck_stop`.
        rec = self._records[task_id]
        if self._stuck_stop:
            raise ValueError("not running")
        if rec.status not in ("completed", "failed", "killed"):
            rec.status = "killed"
        return rec


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "target"
    r.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=r, check=True)
    return r


def _client_with(monkeypatch, manager: _StubManager) -> TestClient:
    monkeypatch.setattr(
        "prometheus.tasks.manager.get_task_manager", lambda: manager
    )
    # coding.enabled is a MASTER SWITCH: POST /api/code 403s without it.
    # These tests exercise the route, so they enable it the way an
    # operator would.
    return TestClient(create_app({"coding": {"enabled": True}}))


def test_create_launches_managed_task(monkeypatch, tmp_path, repo):
    mgr = _StubManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    r = c.post("/api/code", json={
        "repo": str(repo),
        "description": "fix the bug",
        "acceptance_command": "python3 -m pytest tests/ -q",
        "task_id": "t42",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "a12345678"
    assert body["coding_task_id"] == "t42"
    assert body["status"] == "running"
    # The manager was asked to run the real CLI with everything quoted in.
    cmd = mgr.created_with["command"]
    assert "-m prometheus code" in cmd
    assert "--task-id t42" in cmd
    assert "'python3 -m pytest tests/ -q'" in cmd
    assert mgr.created_with["task_type"] == "local_agent"
    assert mgr.created_with["timeout_seconds"] == 1500  # wall cap + headroom


def test_validation_rejects_missing_fields_and_non_repo(monkeypatch, tmp_path, repo):
    c = _client_with(monkeypatch, _StubManager(tmp_path))
    assert c.post("/api/code", json={"repo": str(repo)}).status_code == 400
    assert c.post("/api/code", json={
        "repo": str(tmp_path / "not-a-repo"),
        "description": "x", "acceptance_command": "y",
    }).status_code == 400


def test_security_gate_rejection_surfaces_as_400(monkeypatch, tmp_path, repo):
    c = _client_with(monkeypatch, _StubManager(tmp_path, reject=True))
    r = c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    assert r.status_code == 400
    assert "blocked" in r.json()["error"]


def test_status_read_includes_output_tail(monkeypatch, tmp_path, repo):
    mgr = _StubManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    r = c.get("/api/code/a12345678")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    assert '"branch": "coding/x"' in body["output_tail"]
    assert c.get("/api/code/nope").status_code == 404


# --------------------------------------------------------------------------- #
# The report is returned as its own field — Beacon#130 / audit P9.10
# --------------------------------------------------------------------------- #


def test_report_field_parsed_from_full_output(monkeypatch, tmp_path, repo):
    """`report` arrives pre-parsed, so a client never digs it out of a tail.

    The stub's output is a valid report-shaped object; the endpoint must return
    it structured. Without parse_coding_report wired in, `report` would not be
    in the body at all and this fails on the KeyError below.
    """
    mgr = _StubManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    body = c.get("/api/code/a12345678").json()
    assert body["report"]["status"] == "success"
    assert body["report"]["branch"] == "coding/x"
    # output_tail is still there and still a tail — additive, nothing lost.
    assert '"branch": "coding/x"' in body["output_tail"]


def test_report_larger_than_the_tail_still_arrives(monkeypatch, tmp_path, repo):
    """THE BUG. A report bigger than the 4000-byte tail used to be unrecoverable.

    The endpoint sliced `output[-4_000:]`, and the report is the LAST JSON
    object — so a report larger than the cap started mid-object, no client-side
    parser could recover it, and a FINISHED run rendered as "No report". This is
    reachable, not theoretical: `acceptance_output_tail` is capped at 3000 chars
    and `diff_stat` is uncapped (one line per changed file), so a run touching
    many files overflows 4000.

    Measured against 124 real task logs before writing this: the largest report
    was 2.1 KB, so the ceiling had not been hit in practice — which is exactly
    why it needed a structural fix rather than a bigger constant.

    Asserts BOTH halves: the report arrives whole, AND the tail genuinely does
    not contain it (so the test would fail if someone "fixed" it by raising the
    cap instead of parsing the full output).
    """
    big_report = {
        "task_id": "a12345678",
        "status": "success",
        "reason": "acceptance command exited 0",
        "rounds_used": 7,
        "episodes": 2,
        "wall_seconds": 210.5,
        "acceptance_exit": 0,
        # 3000 is the producer's own cap (_finalize), so this is a realistic shape.
        "acceptance_output_tail": "z" * 3000,
        "branch": "coding/feature-x",
        # Uncapped in the producer: one line per changed file. 400 files is an
        # ordinary refactor and is what pushes the report past the tail.
        "diff_stat": "\n".join(f" src/module_{i}.py | {i} +-" for i in range(400)),
        "sandbox_root": "/tmp/sandbox/a12345678",
    }
    blob = _json.dumps(big_report)
    assert len(blob) > 4000, f"test premise: report must exceed the tail (got {len(blob)})"

    class _BigReportManager(_StubManager):
        def _record(self, status, error=None):
            rec = super()._record(status, error)
            # Log noise first, then the report LAST — the real output shape.
            rec.output_file.write_text("round 1 output\n" * 400 + blob + "\n")
            return rec

    mgr = _BigReportManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    body = c.get("/api/code/a12345678").json()

    # The whole report, parsed server-side from the FULL output.
    assert body["report"]["status"] == "success"
    assert body["report"]["rounds_used"] == 7
    assert body["report"]["diff_stat"] == big_report["diff_stat"]
    assert body["report"]["acceptance_output_tail"] == "z" * 3000

    # And the tail does NOT contain the report — proving the old client-side
    # path was genuinely broken here, and that raising the cap was not the fix.
    # Precisely: the tail holds the report's LAST 4000 bytes (so its trailing
    # keys — sandbox_root, the end of diff_stat — are visible), but NOT its
    # opening brace or leading keys. A fragment with no opening brace is
    # unparseable by construction, which is what made the finished run render as
    # "No report".
    assert len(body["output_tail"]) == 4000
    assert not body["output_tail"].lstrip().startswith("{")
    for leading_key in ("task_id", "status", "rounds_used", "acceptance_output_tail"):
        assert f'"{leading_key}"' not in body["output_tail"], (
            f"{leading_key} leaked into the tail — the report would be smaller than the cap "
            f"and this test would not be exercising the overflow case"
        )
    # The fragment the tail DOES hold is not valid JSON on its own.
    with pytest.raises(ValueError):
        _json.loads(body["output_tail"])


def test_report_is_null_when_no_report_printed_yet(monkeypatch, tmp_path, repo):
    """A run still in progress has no report — null, not a fabricated object.

    This is the state the client previously could not distinguish from a
    truncated report (both rendered "No report"). Now: null means "not printed
    yet", a present object means "here it is".
    """
    class _NoReportManager(_StubManager):
        def _record(self, status, error=None):
            rec = super()._record(status, error)
            rec.output_file.write_text("cloning repo...\nround 1: thinking\n")
            return rec

    mgr = _NoReportManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    body = c.get("/api/code/a12345678").json()
    assert body["report"] is None
    # The tail still carries the diagnostic end of the log.
    assert "round 1: thinking" in body["output_tail"]


def test_report_not_confused_by_other_json_in_output(monkeypatch, tmp_path, repo):
    """Only a report-shaped object is returned — not any JSON the run printed.

    A subprocess prints plenty of JSON (tool results, pytest --json, a config
    dump). Returning the last one of those as "the report" would be worse than
    returning null, because it would render as a finished run with invented
    fields. The markers + a required `status` are what prevent it.
    """
    class _DecoyManager(_StubManager):
        def _record(self, status, error=None):
            rec = super()._record(status, error)
            rec.output_file.write_text(
                '{"status": "ok", "unrelated": true}\n'          # has status, no markers
                '[{"status": "passed"}]\n'                        # array, not object
                '{"tool": "bash", "output": "{not json"}\n'       # tool result with braces
                "Traceback (most recent call last):\n  File x\n"  # a traceback, the common tail
            )
            return rec

    mgr = _DecoyManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    body = c.get("/api/code/a12345678").json()
    assert body["report"] is None, "a JSON object with a status but no report markers is not the report"


def test_report_missing_output_file_is_null_not_500(monkeypatch, tmp_path, repo):
    """An unreadable output file degrades to report:null + empty tail, never a 500.

    The old code wrapped the read in `except OSError: pass` for exactly this
    reason; the parse must inherit that rather than reintroduce the failure.
    """
    mgr = _StubManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={
        "repo": str(repo), "description": "x", "acceptance_command": "y",
    })
    rec = mgr.get_task("a12345678")
    rec.output_file.unlink()  # gone between launch and read
    r = c.get("/api/code/a12345678")
    assert r.status_code == 200
    body = r.json()
    assert body["report"] is None
    assert body["output_tail"] == ""


# --------------------------------------------------------------------------- #
# CLI safety net — run_coding_task always emits a report (never bare exit 1)
# --------------------------------------------------------------------------- #


def test_run_coding_task_emits_report_on_uncaught_exception(monkeypatch, tmp_path, capsys):
    """A mid-run crash must still print a JSON report + return 1, never leave
    a caller with an exit code and no report (the matrix's status=None case)."""
    import json as _json
    import subprocess as _sp

    from prometheus import __main__ as m

    # Explicit config: coding.enabled is a MASTER SWITCH now, so a test that
    # leaves it to whatever config happens to exist on the machine would pass
    # or fail by accident. See TestCodingEnabledMasterSwitch below.
    _cfg = tmp_path / "cfg.yaml"
    _cfg.write_text("coding:\n  enabled: true\n", encoding="utf-8")

    class _Args:
        config = str(_cfg)
        repo = str(tmp_path / "r")
        task_description = "x"
        acceptance_command = "true"
        task_id = "crashy"
        max_rounds = 5
        max_wall_seconds = 60
        sandbox_parent = str(tmp_path / "sb")
        suppress_thinking = False
        control_dir = None  # Loop Manager Sprint 2 — mirrors the new --control-dir CLI arg

    # Isolate the crash-handling path: stub provider/adapter/clone so the
    # test exercises ONLY "session.run() raises → structured failed_error
    # report + exit 1" without a real model, git, or filesystem clone.
    class _StubSandbox:
        root = tmp_path / "clone"

    from tests.support.doubles import register_double

    monkeypatch.setattr(
        m, "create_provider",
        register_double("api_code.create_provider", replaces="prometheus coding create_provider")(lambda cfg: (object(), "m")),
    )
    monkeypatch.setattr(
        m, "create_adapter",
        register_double("api_code.create_adapter", replaces="prometheus coding create_adapter")(lambda *a, **k: None),
    )
    monkeypatch.setattr(
        "prometheus.coding.sandbox.clone_repo_for_sandbox",
        register_double("api_code.clone_repo_for_sandbox", replaces="prometheus.coding.sandbox.clone_repo_for_sandbox")(lambda *a, **k: _StubSandbox()),
    )

    def _boom(self):
        raise RuntimeError("simulated mid-run crash")

    monkeypatch.setattr(
        "prometheus.coding.session.CodingSession.run", _boom
    )

    rc = m.run_coding_task(_Args())
    out = capsys.readouterr().out
    assert rc == 1
    payload = _json.loads(out[out.index("{"):])
    assert payload["status"] == "failed_error"
    assert "simulated mid-run crash" in payload["reason"]


# --------------------------------------------------------------------------- #
# POST /api/code/{id}/stop — cancel a running run (Phase-C follow-up)
# --------------------------------------------------------------------------- #


def test_stop_running_task_kills_it(monkeypatch, tmp_path, repo):
    mgr = _StubManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={"repo": str(repo), "description": "x", "acceptance_command": "y"})
    r = c.post("/api/code/a12345678/stop")
    assert r.status_code == 200
    assert r.json() == {"task_id": "a12345678", "status": "killed"}


def test_stop_unknown_task_404(monkeypatch, tmp_path):
    c = _client_with(monkeypatch, _StubManager(tmp_path))
    assert c.post("/api/code/nope/stop").status_code == 404


def test_stop_completed_is_idempotent_terminal(monkeypatch, tmp_path, repo):
    # Spec: /stop on a completed run returns the terminal state, not 404/409 —
    # the real manager returns the record (process-less terminal branch).
    mgr = _StubManager(tmp_path)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={"repo": str(repo), "description": "x", "acceptance_command": "y"})
    mgr._records["a12345678"].status = "completed"  # already done
    r = c.post("/api/code/a12345678/stop")
    assert r.status_code == 200
    assert r.json() == {"task_id": "a12345678", "status": "completed"}


def test_stop_stuck_not_running_task_409(monkeypatch, tmp_path, repo):
    # The genuine 409 path: stop_task raises ValueError (non-terminal task with
    # no process/waiter) → the route maps it to 409.
    mgr = _StubManager(tmp_path, stuck_stop=True)
    c = _client_with(monkeypatch, mgr)
    c.post("/api/code", json={"repo": str(repo), "description": "x", "acceptance_command": "y"})
    r = c.post("/api/code/a12345678/stop")
    assert r.status_code == 409
    assert "not running" in r.json()["error"]


# --------------------------------------------------------------------------- #
# Mid-run supervision — POST /api/code/{id}/pause|inject|resume (Sprint 2)
# --------------------------------------------------------------------------- #


def _supervisable(tmp_path, monkeypatch, *, status="running"):
    """A stub manager holding one run with a coding session_id, with the control
    dir redirected under tmp_path so the endpoints write there, not ~/.prometheus."""
    monkeypatch.setattr(
        "prometheus.coding.managed.coding_control_dir",
        lambda coding_id: tmp_path / "ctl" / coding_id,
    )
    mgr = _StubManager(tmp_path)
    mgr._records["a12345678"] = TaskRecord(
        id="a12345678", type="local_agent", status=status, description="coding",
        cwd=str(tmp_path), output_file=tmp_path / "o.log", session_id="coding:t42",
    )
    return mgr


def _ctl_state(tmp_path):
    from prometheus.coding.control import control_path, parse_control
    p = control_path(tmp_path / "ctl" / "t42")
    return parse_control(p.read_text()) if p.exists() else None


def test_pause_inject_resume_round_trip(monkeypatch, tmp_path):
    c = _client_with(monkeypatch, _supervisable(tmp_path, monkeypatch))

    r = c.post("/api/code/a12345678/pause")
    assert r.status_code == 200 and r.json()["status"] == "paused"
    assert _ctl_state(tmp_path).paused is True
    # idempotent + non-destructive: pausing an already-paused run
    assert c.post("/api/code/a12345678/pause").json()["status"] == "already-paused"

    r = c.post("/api/code/a12345678/inject", json={"text": "FOCUS the parser"})
    assert r.status_code == 200 and r.json()["status"] == "injected" and r.json()["injection_id"]
    injs = _ctl_state(tmp_path).injections
    assert len(injs) == 1 and injs[0].text == "FOCUS the parser"

    r = c.post("/api/code/a12345678/resume")
    assert r.status_code == 200 and r.json()["status"] == "resumed"
    assert _ctl_state(tmp_path).paused is False
    assert len(_ctl_state(tmp_path).injections) == 1  # queued steer survives resume (consumed by id)
    # idempotent + non-destructive: resuming a non-paused run
    assert c.post("/api/code/a12345678/resume").json()["status"] == "not-paused"


def test_inject_requires_nonblank_text(monkeypatch, tmp_path):
    c = _client_with(monkeypatch, _supervisable(tmp_path, monkeypatch))
    assert c.post("/api/code/a12345678/inject", json={"text": "   "}).status_code == 400
    assert c.post("/api/code/a12345678/inject", json={}).status_code == 400


def test_supervise_unknown_id_404(monkeypatch, tmp_path):
    c = _client_with(monkeypatch, _StubManager(tmp_path))  # no records
    for ep in ("pause", "resume", "inject"):
        assert c.post(f"/api/code/nope/{ep}", json={"text": "x"}).status_code == 404, ep


def test_supervise_terminal_run_409(monkeypatch, tmp_path):
    c = _client_with(monkeypatch, _supervisable(tmp_path, monkeypatch, status="completed"))
    for ep in ("pause", "resume", "inject"):
        assert c.post(f"/api/code/a12345678/{ep}", json={"text": "x"}).status_code == 409, ep


# --------------------------------------------------------------------------- #
# GET /api/code/{id}/diff — full artifact diff (Phase-C follow-up)
# --------------------------------------------------------------------------- #


def _make_coding_clone(coding_root: Path, name: str) -> Path:
    """A git repo with a base commit + one artifact commit on a coding/ branch —
    exactly what session._commit_artifact leaves behind (diff range HEAD~1..HEAD)."""
    root = coding_root / name
    root.mkdir(parents=True)
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=root, check=True,
        capture_output=True,
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    git("add", "."); git("commit", "-qm", "base")
    git("checkout", "-q", "-b", "coding/run1")
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git("add", "."); git("commit", "-qm", "artifact")
    return root


class _DiffManager:
    """Stub manager whose task points its output file at a report we control."""

    def __init__(self, output_file: Path) -> None:
        self._rec = TaskRecord(
            id="d1", type="local_agent", status="completed",
            description="coding", cwd="/tmp", output_file=output_file,
            command="…", created_at=1.0,
        )

    def get_task(self, task_id: str):
        return self._rec if task_id == "d1" else None


def test_diff_returns_full_unified_diff(monkeypatch, tmp_path):
    coding_root = tmp_path / "coding"
    clone = _make_coding_clone(coding_root, "run1")
    out = tmp_path / "out.log"
    out.write_text(_json.dumps({"status": "success", "branch": "coding/run1", "sandbox_root": str(clone)}))

    monkeypatch.setattr("prometheus.web.server._coding_sandbox_root", lambda: coding_root.resolve())
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: _DiffManager(out))
    c = TestClient(create_app({"coding": {"enabled": True}}))

    r = c.get("/api/code/d1/diff")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["branch"] == "coding/run1"
    assert "-    return a - b" in body["diff"] and "+    return a + b" in body["diff"]


def test_diff_rejects_sandbox_outside_coding_dir(monkeypatch, tmp_path):
    # A report pointing OUTSIDE ~/.prometheus/coding must be refused (path guard).
    evil = tmp_path / "evil"
    _make_coding_clone(tmp_path, "evil")  # a real repo, but not under coding_root
    out = tmp_path / "out.log"
    out.write_text(_json.dumps({"status": "success", "branch": "coding/x", "sandbox_root": str(evil)}))

    monkeypatch.setattr("prometheus.web.server._coding_sandbox_root", lambda: (tmp_path / "coding").resolve())
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: _DiffManager(out))
    c = TestClient(create_app({"coding": {"enabled": True}}))
    assert c.get("/api/code/d1/diff").status_code == 422


def test_diff_not_ready_while_running(monkeypatch, tmp_path):
    out = tmp_path / "out.log"
    out.write_text("starting…\nworking, no report yet\n")  # no JSON report
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: _DiffManager(out))
    c = TestClient(create_app({"coding": {"enabled": True}}))
    r = c.get("/api/code/d1/diff")
    assert r.status_code == 200
    assert r.json()["ready"] is False


def _make_failed_clone(coding_root: Path, name: str) -> Path:
    """A clone as a run that died BEFORE creating its branch leaves it.

    Two commits from the SOURCE repo and no coding/ branch — so HEAD~1..HEAD resolves to the
    source repo's own last commit, which is the whole defect.
    """
    root = coding_root / name
    root.mkdir(parents=True)
    git = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=root, check=True,
        capture_output=True,
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    git("add", "."); git("commit", "-qm", "base")
    # The human's own last commit on the source repo — what the buggy range returned.
    (root / "NOTES.md").write_text("HUMAN_COMMIT_CANARY\n")
    git("add", "."); git("commit", "-qm", "loop contract: acceptance line format")
    return root


def test_diff_refuses_a_report_with_no_branch(monkeypatch, tmp_path):
    """A run that never made an artifact branch must not be handed somebody else's commit.

    Observed 2026-09-18 (task aa0db5661 / run c16ac96c9, which failed at `git checkout -b` because
    the sandbox image had no git): the report carried a sandbox_root and no branch, the clone sat
    on the source repo's tip, and HEAD~1..HEAD returned ` LOOP.md | 3 +--` — a human's commit from
    2026-07-10, delivered as the run's artifact with ready:true.

    The canary assertion is the one that does the work. Asserting only `ready is False` would pass
    against a fix that returned not-ready for the wrong reason; asserting the source commit's
    content is ABSENT pins that no foreign diff escapes.
    """
    coding_root = tmp_path / "coding"
    clone = _make_failed_clone(coding_root, "failedrun")
    out = tmp_path / "out.log"
    out.write_text(_json.dumps({
        "status": "failed_error",
        "reason": "uncaught CodingGitError: git checkout -q -b coding/x failed (exit 127)",
        "sandbox_root": str(clone),
    }))

    monkeypatch.setattr("prometheus.web.server._coding_sandbox_root", lambda: coding_root.resolve())
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: _DiffManager(out))
    c = TestClient(create_app({"coding": {"enabled": True}}))

    r = c.get("/api/code/d1/diff")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is False, "a run with no artifact branch is not a ready diff"
    assert body["branch"] is None
    assert "HUMAN_COMMIT_CANARY" not in _json.dumps(body), (
        "the source repo's own commit escaped as this run's artifact diff"
    )
    assert "NOTES.md" not in _json.dumps(body)
    assert body["reason"] == "the run produced no artifact branch"


def test_diff_not_ready_reasons_distinguish_running_from_no_artifact(monkeypatch, tmp_path):
    """Both not-ready cases carry a reason, because they are different facts.

    "still running" and "finished, produced nothing" both render as ready:false, and a client that
    cannot tell them apart says "no diff yet" about a run that will never have one.
    """
    out = tmp_path / "out.log"
    out.write_text("starting…\nno report yet\n")
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: _DiffManager(out))
    c = TestClient(create_app({"coding": {"enabled": True}}))
    assert c.get("/api/code/d1/diff").json()["reason"] == "the run has not reported yet"


def test_diff_unknown_task_404(monkeypatch, tmp_path):
    out = tmp_path / "out.log"
    out.write_text("{}")
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: _DiffManager(out))
    c = TestClient(create_app({"coding": {"enabled": True}}))
    assert c.get("/api/code/nope/diff").status_code == 404


# --------------------------------------------------------------------------- #
# coding.enabled — the master switch that was not one
#
# It shipped in prometheus.yaml.default documented as `enabled: false` and was
# read by NOTHING: not run_coding_task, not POST /api/code. Setting it false —
# or taking the shipped default — produced a coding mode that ran anyway. A
# documented key with a false default that gates nothing is worse than no key,
# because it answers "is this off?" with a confident yes.
# --------------------------------------------------------------------------- #


class TestCodingEnabledMasterSwitch:

    def _args(self, tmp_path, enabled: bool):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"coding:\n  enabled: {str(enabled).lower()}\n", encoding="utf-8")

        class _Args:
            config = str(cfg)
            repo = str(tmp_path / "r")
            task_description = "x"
            acceptance_command = "true"
            task_id = "gated"
            max_rounds = 5
            max_wall_seconds = 60
            sandbox_parent = str(tmp_path / "sb")
            suppress_thinking = False
            control_dir = None

        return _Args()

    def test_cli_refuses_when_disabled(self, tmp_path, capsys):
        import json as _json

        from prometheus import __main__ as m

        rc = m.run_coding_task(self._args(tmp_path, enabled=False))
        out = capsys.readouterr().out
        assert rc == 2
        payload = _json.loads(out[out.index("{"):])
        assert payload["ok"] is False
        assert "disabled" in payload["error"]
        assert "coding.enabled" in payload["detail"]

    def test_cli_refusal_happens_before_any_provider_is_built(
        self, tmp_path, capsys, monkeypatch
    ):
        """The gate must short-circuit BEFORE provider/adapter construction —
        otherwise a disabled coding mode still dials a cloud API."""
        from prometheus import __main__ as m

        def _explode(*a, **k):
            raise AssertionError("provider built despite coding being disabled")

        monkeypatch.setattr(m, "create_provider", _explode)
        assert m.run_coding_task(self._args(tmp_path, enabled=False)) == 2

    def test_cli_default_is_disabled(self, tmp_path, capsys):
        """An ABSENT key means OFF, matching the shipped default. Defaulting
        to True here would have preserved the old behaviour silently and
        left every existing config still ungated."""
        import json as _json

        from prometheus import __main__ as m

        cfg = tmp_path / "no-coding-block.yaml"
        cfg.write_text("model: {}\n", encoding="utf-8")
        args = self._args(tmp_path, enabled=True)
        args.config = str(cfg)

        rc = m.run_coding_task(args)
        out = capsys.readouterr().out
        assert rc == 2
        assert _json.loads(out[out.index("{"):])["ok"] is False

    def test_api_route_refuses_when_disabled(self):
        """POST /api/code is the reason this key matters more than a local
        flag: it launches model-authored command execution over the network."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.web.server import create_app

        app = create_app({"coding": {"enabled": False}, "web": {}})
        client = TestClient(app)
        r = client.post("/api/code", json={
            "repo": "/tmp/x", "description": "d", "acceptance_command": "true",
        })
        assert r.status_code == 403
        assert "disabled" in r.json()["error"]

    def test_api_route_refuses_before_validating_the_body(self):
        """A disabled route must not leak input-validation behaviour — the
        403 comes first, so an empty body still reads 'disabled', not 400."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.web.server import create_app

        client = TestClient(create_app({"coding": {"enabled": False}, "web": {}}))
        r = client.post("/api/code", json={})
        assert r.status_code == 403
