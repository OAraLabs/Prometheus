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


# ── side model calls (compaction, titles, extraction, curator, synthesis) ────

@pytest.mark.parametrize("mode", ["raise", "return_none", "log_only"])
def test_a_side_call_while_paused_reaches_no_model_and_fails_by_its_policy(tmp_path, mode):
    from prometheus.learning.llm_envelope import LLMCallEnvelope

    state = pause.PauseState(tmp_path / "pause.json")
    pause.install(state)
    state.pause(by="owner", reason="")
    called: list[str] = []

    class _Provider:
        async def stream_message(self, request):
            called.append("model")
            if False:
                yield None

    env = LLMCallEnvelope("curator", telemetry=None, on_failure=mode)
    try:
        if mode == "raise":
            with pytest.raises(pause.Paused):
                asyncio.run(env.call(provider=_Provider(), model="m", prompt="p"))
        elif mode == "return_none":
            assert asyncio.run(env.call(provider=_Provider(), model="m", prompt="p")) is None
        else:
            result = asyncio.run(env.call(provider=_Provider(), model="m", prompt="p"))
            assert isinstance(result.error, pause.Paused) and result.text == ""
    finally:
        pause.install(None)
    assert called == []


# ── no new coding run or desktop task while paused ───────────────────────────

@pytest.mark.parametrize(("path", "body"), [
    ("/api/code", {"description": "fix it", "repo": "/tmp/x"}),
    ("/api/computer/tasks", {"goal": "open notes", "app": "Notes", "session_id": "web:s1"}),
])
def test_starting_new_work_while_paused_is_423_and_starts_nothing(tmp_path, monkeypatch, path, body):
    pytest.importorskip("fastapi")
    from tests.support.device_world import World

    state = pause.PauseState(tmp_path / "pause.json")
    pause.install(state)
    try:
        w = World(tmp_path)
        from prometheus.coding import managed

        monkeypatch.setattr(managed, "create_coding_managed_task",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("started a coding run")))
        state.pause(by="owner", reason="")
        r = w.call("op", "POST", path, json=body)
        assert r.status_code == 423, (path, r.status_code, r.text)
        assert r.json()["error"] == "paused"
    finally:
        pause.install(None)
