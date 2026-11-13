"""A cron job due while everything is paused is SKIPPED and recorded as skipped, not saved up (Piece 17, PR 2).

The job's command never runs; its next run is moved on to the next time on its schedule (so it is not run on
resume, and not re-dispatched every tick); its last status reads ``skipped``; its history row says why; and no
failure notification goes out, because nothing failed.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine import pause
from prometheus.gateway import cron_scheduler as cs
from prometheus.gateway import cron_service


@pytest.fixture
def world(tmp_path, monkeypatch):
    reg = tmp_path / "cron_jobs.json"
    monkeypatch.setattr(cron_service, "get_cron_registry_path", lambda: reg)
    history: list[dict] = []
    monkeypatch.setattr(cs, "append_history", lambda entry: history.append(entry))
    notified: list[dict] = []

    async def _notify(entry):
        notified.append(entry)

    monkeypatch.setattr(cs, "_maybe_notify_failure", _notify)
    ran: list[str] = []

    async def _spawn(*a, **kw):
        ran.append("spawned")
        raise AssertionError("a paused cron job must not spawn anything")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    cron_service.save_cron_jobs([{"name": "nightly", "command": "echo hi", "schedule": "0 3 * * *",
                                  "next_run": "2026-01-01T03:00:00+00:00", "enabled": True}])
    state = pause.PauseState(tmp_path / "pause.json")
    pause.install(state)
    yield history, notified, ran, state
    pause.install(None)


def test_a_due_job_is_skipped_recorded_and_moved_on(world):
    history, notified, ran, state = world
    state.pause(by="owner", reason="away")
    entry = asyncio.run(cs.execute_job({"name": "nightly", "command": "echo hi", "schedule": "0 3 * * *"}))
    assert ran == [] and entry["status"] == "skipped" and "paused" in entry["stderr"].lower()
    job = cron_service.load_cron_jobs()[0]
    assert job["last_status"] == "skipped"
    assert job["next_run"] != "2026-01-01T03:00:00+00:00"          # moved on, not saved up
    assert history and history[-1]["status"] == "skipped" and notified == []


def test_a_job_runs_as_before_when_not_paused(world, monkeypatch):
    history, notified, ran, state = world
    monkeypatch.setattr(cs, "vet_cron_command", lambda command, cwd=None: (False, "test stops here"))
    entry = asyncio.run(cs.execute_job({"name": "nightly", "command": "echo hi", "schedule": "0 3 * * *"}))
    assert entry["status"] == "blocked"                               # it reached the gate: not skipped
