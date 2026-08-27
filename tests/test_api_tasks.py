"""GET /api/tasks — background task cockpit (branch feat/api-tasks).

The tasks table has been durable since June and nothing ever served it. The task_* WS frames are
fan-out only and are not persisted to signal_events either, so a client that was not running when
a task started AND finished had no way to learn it happened — which is exactly the "what was it
doing while I was away" question this exists to answer.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


def _rec(**kw):
    base = dict(
        id="t1", type="local_agent", status="completed", description="", command=None,
        cwd="/tmp", session_id=None, created_at=1000.0, started_at=1000.0, ended_at=1010.0,
        return_code=0, error=None, on_complete="notify",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Mgr:
    def __init__(self, records, output="hello from the task"):
        self._records = records
        self._output = output
        self.stopped: list[str] = []

    def list_tasks(self, *, status=None):
        return [r for r in self._records if status is None or r.status == status]

    def get_task(self, task_id):
        return next((r for r in self._records if r.id == task_id), None)

    def read_task_output(self, task_id, *, max_bytes=12000):
        return self._output[:max_bytes]

    async def stop_task(self, task_id):
        self.stopped.append(task_id)
        rec = self.get_task(task_id)
        rec.status = "killed"
        rec.ended_at = 2000.0
        return rec


@pytest.fixture
def client(monkeypatch):
    def _make(records, **kw):
        mgr = _Mgr(records, **kw)
        monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: mgr)
        return TestClient(create_app({})), mgr
    return _make


def test_running_task_reports_elapsed_not_a_missing_duration(client):
    started = time.time() - 42
    c, _ = client([_rec(id="live", status="running", started_at=started, ended_at=None, return_code=None)])
    row = c.get("/api/tasks").json()["tasks"][0]
    assert row["running"] is True
    # A task that has not ended still has a duration — how long it has been going. Returning null
    # here would make the cockpit unable to say anything about the thing it exists to show.
    assert row["duration_s"] == pytest.approx(42, abs=2)


def test_finished_task_reports_how_long_it_took(client):
    c, _ = client([_rec(started_at=1000.0, ended_at=1010.0)])
    row = c.get("/api/tasks").json()["tasks"][0]
    assert row["duration_s"] == 10.0
    assert row["running"] is False


def test_a_task_that_never_started_has_no_duration(client):
    c, _ = client([_rec(status="pending", started_at=None, ended_at=None)])
    row = c.get("/api/tasks").json()["tasks"][0]
    assert row["duration_s"] is None       # null, not 0 — it has not run for zero seconds
    assert row["running"] is True          # pending still counts as live work


def test_counts_are_the_full_histogram_regardless_of_filter(client):
    c, _ = client([
        _rec(id="a", status="completed"), _rec(id="b", status="failed"),
        _rec(id="c", status="running", ended_at=None),
    ])
    body = c.get("/api/tasks?status=running").json()
    assert [t["id"] for t in body["tasks"]] == ["c"]
    # The point: an empty or single-row list must still be able to say what else exists, so
    # "running: 0" reads as "nothing running" rather than as a broken panel.
    assert body["counts"] == {"completed": 1, "failed": 1, "running": 1}


def test_newest_first_and_capped(client):
    recs = [_rec(id=f"t{i}", created_at=float(i)) for i in range(10)]
    c, _ = client(recs)
    ids = [t["id"] for t in c.get("/api/tasks?limit=3").json()["tasks"]]
    assert ids == ["t9", "t8", "t7"]


def test_detail_carries_the_output_tail(client):
    c, _ = client([_rec(id="t1")], output="line one\nline two")
    body = c.get("/api/tasks/t1").json()
    assert body["output"] == "line one\nline two"
    assert body["id"] == "t1"


def test_unreadable_output_is_a_fact_not_a_500(client, monkeypatch):
    c, mgr = client([_rec(id="t1")])
    def boom(*a, **k):
        raise OSError("output file vanished")
    monkeypatch.setattr(mgr, "read_task_output", boom)
    r = c.get("/api/tasks/t1")
    assert r.status_code == 200
    assert "output unavailable" in r.json()["output"]
    assert "vanished" in r.json()["output"]  # says WHY, not just that


def test_missing_task_is_404_not_an_empty_row(client):
    c, _ = client([_rec(id="t1")])
    assert c.get("/api/tasks/nope").status_code == 404
    assert c.post("/api/tasks/nope/stop").status_code == 404


def test_stop_returns_the_task_as_it_stands_after(client):
    c, mgr = client([_rec(id="live", status="running", ended_at=None, return_code=None)])
    body = c.post("/api/tasks/live/stop").json()
    assert mgr.stopped == ["live"]
    assert body["status"] == "killed"
    assert body["running"] is False  # the caller can render the result without a second GET


def test_stopping_a_finished_task_is_a_no_op_not_an_error(client):
    """Otherwise a client must race the task to know whether its own button was allowed."""
    c, _ = client([_rec(id="done", status="completed")])
    r = c.post("/api/tasks/done/stop")
    assert r.status_code == 200


def test_no_task_manager_degrades_instead_of_500(client, monkeypatch):
    monkeypatch.setattr("prometheus.tasks.manager.get_task_manager", lambda: None)
    c = TestClient(create_app({}))
    assert c.get("/api/tasks").json() == {"tasks": [], "counts": {}}
    assert c.get("/api/tasks/x").status_code == 503


# ── the defect the fake-manager tests could not see ──────────────────────────────────────────
# Every test above uses a stub manager, so none of them could catch that the REAL
# BackgroundTaskManager.list_tasks read only this process's memory. On the live box that meant
# /api/tasks answered "0 tasks" while tasks.db held 24 — and the agent's own task_list tool had
# been doing the same after every restart since June. A live call caught it; these pin it.

def test_manager_lists_tasks_it_did_not_create(tmp_path):
    """The restart case: a fresh manager, an empty memory, a populated store."""
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore
    from prometheus.tasks.types import TaskRecord

    store = TaskStore(db_path=tmp_path / "tasks.db")
    mk = lambda **kw: TaskRecord(description="", cwd="/tmp", output_file=tmp_path / "out.log", **kw)
    store.upsert(mk(id="old-1", type="local_bash", status="completed", created_at=1.0))
    store.upsert(mk(id="old-2", type="local_agent", status="failed", created_at=2.0))

    mgr = BackgroundTaskManager(store=store)
    assert mgr._tasks == {}, "precondition: this process created nothing"
    listed = [t.id for t in mgr.list_tasks()]
    assert listed == ["old-2", "old-1"], f"durable tasks must be listed, newest first (got {listed})"
    assert [t.id for t in mgr.list_tasks(status="failed")] == ["old-2"]


def test_in_memory_record_wins_over_a_staler_stored_copy(tmp_path):
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore
    from prometheus.tasks.types import TaskRecord

    store = TaskStore(db_path=tmp_path / "tasks.db")
    mk = lambda **kw: TaskRecord(description="", cwd="/tmp", output_file=tmp_path / "out.log", **kw)
    store.upsert(mk(id="t", type="local_bash", status="running", created_at=1.0))
    mgr = BackgroundTaskManager(store=store)
    # an in-flight update this process knows about and the store has not caught up on
    mgr._tasks["t"] = mk(id="t", type="local_bash", status="completed", created_at=1.0)
    assert mgr.list_tasks()[0].status == "completed"
    assert mgr.list_tasks(status="running") == []
