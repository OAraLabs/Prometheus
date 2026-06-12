"""POST /api/code + GET /api/code/{id} — coding runs over the API (v2 scope 5).

The BackgroundTaskManager singleton is stubbed so these never spawn a real
subprocess; what's asserted is the wiring — validation, the command the
manager is asked to run, SecurityGate-rejection surfacing, and the status
read including the output tail.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.tasks.types import TaskRecord  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _StubManager:
    def __init__(self, tmp_path: Path, *, reject: bool = False) -> None:
        self._tmp = tmp_path
        self._reject = reject
        self.created_with: dict | None = None
        self._records: dict[str, TaskRecord] = {}

    def _record(self, status: str, error: str | None = None) -> TaskRecord:
        out = self._tmp / "task-output.log"
        out.write_text('{"status": "success", "branch": "coding/x"}')
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
    return TestClient(create_app({}))


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
# CLI safety net — run_coding_task always emits a report (never bare exit 1)
# --------------------------------------------------------------------------- #


def test_run_coding_task_emits_report_on_uncaught_exception(monkeypatch, tmp_path, capsys):
    """A mid-run crash must still print a JSON report + return 1, never leave
    a caller with an exit code and no report (the matrix's status=None case)."""
    import json as _json
    import subprocess as _sp

    from prometheus import __main__ as m

    class _Args:
        config = None
        repo = str(tmp_path / "r")
        task_description = "x"
        acceptance_command = "true"
        task_id = "crashy"
        max_rounds = 5
        max_wall_seconds = 60
        sandbox_parent = str(tmp_path / "sb")
        suppress_thinking = False

    # Isolate the crash-handling path: stub provider/adapter/clone so the
    # test exercises ONLY "session.run() raises → structured failed_error
    # report + exit 1" without a real model, git, or filesystem clone.
    class _StubSandbox:
        root = tmp_path / "clone"

    monkeypatch.setattr(m, "create_provider", lambda cfg: (object(), "m"))
    monkeypatch.setattr(m, "create_adapter", lambda *a, **k: None)
    monkeypatch.setattr(
        "prometheus.coding.sandbox.clone_repo_for_sandbox",
        lambda *a, **k: _StubSandbox(),
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
