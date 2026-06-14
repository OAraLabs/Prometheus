"""Completed coding tasks survive a daemon restart (chore/coding-task-restart-survival).

Before this fix, ``resume_running`` rehydrated only RUNNING tasks, and ``get_task``
read only the in-memory ``_tasks`` map — so a coding run that FINISHED before the
current daemon lifetime was durable in tasks.db but invisible to the API, and
``GET /api/code/{id}``, ``/diff`` and ``/stop`` all 404'd even though the sandbox
clone and the report still existed on disk.

These tests assert the SIDE EFFECTS of the fix: a fresh manager built on the same
durable store (the in-process stand-in for "after restart") resolves the completed
task, returns its real diff bytes and status, and stops it idempotently — and a
genuinely-unknown id still 404s (no false rehydration).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.tasks.manager import BackgroundTaskManager  # noqa: E402
from prometheus.tasks.store import TaskStore  # noqa: E402
from prometheus.tasks.types import TaskRecord  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=root, check=True, capture_output=True,
    )


def _make_coding_clone(coding_root: Path, name: str) -> Path:
    """A repo with a base commit + one artifact commit on a coding/ branch —
    what session._commit_artifact leaves behind (diff range HEAD~1..HEAD)."""
    root = coding_root / name
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(root, "add", "."); _git(root, "commit", "-qm", "base")
    _git(root, "checkout", "-q", "-b", "coding/run1")
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    _git(root, "add", "."); _git(root, "commit", "-qm", "artifact")
    return root


@pytest.fixture()
def finished_run(tmp_path: Path):
    """A completed coding task as it would be AFTER the run, persisted durably:
    a real sandbox clone, a report file, and a 'completed' row in a TaskStore."""
    coding_root = tmp_path / "coding"
    clone = _make_coding_clone(coding_root, "run1")
    report = {
        "task_id": "run1", "status": "success", "branch": "coding/run1",
        "sandbox_root": str(clone), "acceptance_exit": 0,
        "diff_stat": " calc.py | 2 +-", "rounds_used": 8,
    }
    out = tmp_path / "task-output.log"
    out.write_text("…logs…\n" + json.dumps(report))

    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.upsert(TaskRecord(
        id="t-done", type="local_agent", status="completed",
        description="coding task", cwd=str(tmp_path), output_file=out,
        command="…", created_at=1.0, ended_at=2.0, return_code=0,
    ))
    return {"store": store, "coding_root": coding_root.resolve(), "clone": clone}


def _restarted_client(monkeypatch, finished_run) -> TestClient:
    """A FRESH manager on the SAME store (empty _tasks) — the in-process
    equivalent of a daemon restart — wired into the API."""
    mgr = BackgroundTaskManager(store=finished_run["store"])
    assert mgr.get_task("t-done") is not None  # rehydrated from the store…
    # …but it started with an empty live map (proving the fallback did it):
    assert "t-done" in mgr._tasks  # cached after the lookup above
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: mgr)
    monkeypatch.setattr(
        "prometheus.web.server._coding_sandbox_root", lambda: finished_run["coding_root"]
    )
    return TestClient(create_app({}))


# ---------------------------------------------------------------------------
# Manager level — the lazy store fallback
# ---------------------------------------------------------------------------


def test_fresh_manager_rehydrates_completed_task_from_store(finished_run):
    mgr = BackgroundTaskManager(store=finished_run["store"])
    assert mgr._tasks == {}  # nothing in memory yet (post-restart)
    rec = mgr.get_task("t-done")
    assert rec is not None
    assert rec.status == "completed"
    assert mgr._tasks["t-done"] is rec  # cached so stop_task resolves too


def test_unknown_id_is_not_falsely_rehydrated(finished_run):
    mgr = BackgroundTaskManager(store=finished_run["store"])
    assert mgr.get_task("does-not-exist") is None
    assert "does-not-exist" not in mgr._tasks


@pytest.mark.asyncio
async def test_stop_rehydrated_completed_returns_terminal(finished_run):
    mgr = BackgroundTaskManager(store=finished_run["store"])
    rec = await mgr.stop_task("t-done")  # resolves via _require_task → store
    assert rec.status == "completed"  # idempotent: terminal, not raised


# ---------------------------------------------------------------------------
# Route level — all three /api/code reads resolve post-"restart"
# ---------------------------------------------------------------------------


def test_status_resolves_after_restart(monkeypatch, finished_run):
    c = _restarted_client(monkeypatch, finished_run)
    r = c.get("/api/code/t-done")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert '"branch": "coding/run1"' in body["output_tail"]


def test_diff_returns_real_bytes_after_restart(monkeypatch, finished_run):
    c = _restarted_client(monkeypatch, finished_run)
    r = c.get("/api/code/t-done/diff")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["branch"] == "coding/run1"
    # The ACTUAL diff bytes — not just non-404.
    assert "-    return a - b" in body["diff"]
    assert "+    return a + b" in body["diff"]


def test_stop_resolves_terminal_after_restart(monkeypatch, finished_run):
    c = _restarted_client(monkeypatch, finished_run)
    r = c.post("/api/code/t-done/stop")
    assert r.status_code == 200
    assert r.json() == {"task_id": "t-done", "status": "completed"}
    # Idempotent — a second stop is still the terminal state.
    r2 = c.post("/api/code/t-done/stop")
    assert r2.status_code == 200 and r2.json()["status"] == "completed"


def test_unknown_run_still_404s_after_restart(monkeypatch, finished_run):
    c = _restarted_client(monkeypatch, finished_run)
    assert c.get("/api/code/ghost").status_code == 404
    assert c.get("/api/code/ghost/diff").status_code == 404
    assert c.post("/api/code/ghost/stop").status_code == 404
