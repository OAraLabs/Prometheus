"""POST/PUT/DELETE /api/cron + POST /api/cron/{name}/run — cron registry writes
(Beacon Desktop Tier 3).

The cron registry path is monkeypatched to a tmp file so these never touch the
real ~/.prometheus/data/cron_jobs.json. Run-now's executor is stubbed (it is
fire-and-forget, so the route only acknowledges dispatch).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    reg = tmp_path / "cron_jobs.json"
    # cron_service binds get_cron_registry_path at import; patch it in that namespace.
    monkeypatch.setattr(
        "prometheus.gateway.cron_service.get_cron_registry_path", lambda: reg
    )
    return TestClient(create_app({})), reg


def test_create_lists_and_computes_next_run(client):
    c, _ = client
    r = c.post("/api/cron", json={"name": "nightly", "schedule": "30 7 * * *", "command": "echo hi"})
    assert r.status_code == 201
    job = r.json()["job"]
    assert job["name"] == "nightly" and job["command"] == "echo hi"
    assert job["enabled"] is True
    assert job["next_run"]  # upsert computed the next run from the schedule
    # now visible via the read route
    assert [j["name"] for j in c.get("/api/cron").json()] == ["nightly"]


def test_create_rejects_bad_schedule_and_missing_fields(client):
    c, _ = client
    assert c.post("/api/cron", json={"name": "x", "schedule": "not a cron", "command": "echo"}).status_code == 400
    assert c.post("/api/cron", json={"name": "x", "schedule": "* * * * *"}).status_code == 400  # no command
    assert c.post("/api/cron", json={"schedule": "* * * * *", "command": "echo"}).status_code == 400  # no name


def test_create_conflict_on_duplicate_name(client):
    c, _ = client
    assert c.post("/api/cron", json={"name": "dup", "schedule": "* * * * *", "command": "echo"}).status_code == 201
    assert c.post("/api/cron", json={"name": "dup", "schedule": "* * * * *", "command": "echo"}).status_code == 409


def test_update_merges_fields_and_preserves_untouched(client):
    c, _ = client
    c.post("/api/cron", json={"name": "job", "schedule": "* * * * *", "command": "echo old", "enabled": True})
    r = c.put("/api/cron/job", json={"command": "echo new", "enabled": False})
    assert r.status_code == 200
    job = r.json()["job"]
    assert job["command"] == "echo new" and job["enabled"] is False
    assert job["schedule"] == "* * * * *"  # untouched field preserved


def test_update_validates_schedule_and_404(client):
    c, _ = client
    c.post("/api/cron", json={"name": "job", "schedule": "* * * * *", "command": "echo"})
    assert c.put("/api/cron/job", json={"schedule": "bogus"}).status_code == 400
    assert c.put("/api/cron/missing", json={"command": "echo"}).status_code == 404


def test_delete_and_404(client):
    c, _ = client
    c.post("/api/cron", json={"name": "gone", "schedule": "* * * * *", "command": "echo"})
    assert c.delete("/api/cron/gone").status_code == 200
    assert c.get("/api/cron").json() == []
    assert c.delete("/api/cron/gone").status_code == 404


def test_run_now_dispatches_and_404(client, monkeypatch):
    c, _ = client
    c.post("/api/cron", json={"name": "runme", "schedule": "* * * * *", "command": "echo hi"})

    # Stub the executor so the fire-and-forget task has no shell/history side effects.
    async def fake_execute(job):
        return {"name": job["name"], "status": "success"}

    monkeypatch.setattr("prometheus.gateway.cron_scheduler.execute_job", fake_execute)

    r = c.post("/api/cron/runme/run")
    assert r.status_code == 200 and r.json()["started"] is True
    assert c.post("/api/cron/missing/run").status_code == 404
