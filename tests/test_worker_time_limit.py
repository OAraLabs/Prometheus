"""A worker's time limit is the owner's setting, counts only the time it works, and is reported plainly.

WHAT HAPPENED
-------------
Six research workers in one conversation were killed at 15 minutes (rc -9). There is no 15-minute constant in the
code: a worker took whatever ``timeout_seconds`` the spawning MODEL put on ``task_create`` (or
``tasks.default_timeout_seconds``), one wall clock that kept running while the worker sat waiting for a person to
approve something, and the parent conversation was told only "it ran past its time limit (timeout)", with no limit
and no time, so the agent could not tell whether to retry, narrow the task or give up.

THE RULE THESE TESTS PIN
------------------------
* ``workers.timeout_minutes`` (default 60) is a worker's limit. The model no longer sets it: ``task_create`` passes
  no ``timeout_seconds`` for an agent worker. (A caller of the task manager that names one still gets it; shell,
  watch and poll tasks keep ``timeout_seconds`` and ``tasks.default_timeout_seconds``.)
* Time spent waiting for an approval does not count. The daemon brackets each wait it files for a worker
  (overlapping waits count once), and the limit is measured in working time.
* A worker stopped for time reports it to the parent conversation in so many words: "stopped after N minutes — time
  limit", the limit and where it is set, how long it waited for approval, and what to do next.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import pytest
import yaml

from tests.test_worker_approvals import _app, _ask, _client, _until

ROOT = Path(__file__).resolve().parents[1]
_LOCAL_TURN = {
    "session_id": "web", "effective_session_id": "beacon:parent", "ephemeral": False,
    "serving_route": {"key": "local", "provider": "llama_cpp", "model": "fake-local-27b", "backend": None},
}


class _Gate:
    """The queue requires a gate reference; nothing here records a grant."""


class _Bus:
    def __init__(self) -> None:
        self.signals: list = []

    async def emit(self, signal) -> None:
        self.signals.append(signal)


def _sleeping_workers(monkeypatch, command: str = "sleep 30"):
    from prometheus.tasks import manager as mgr_mod

    monkeypatch.setattr(mgr_mod, "prometheus_argv", lambda *a: ["/bin/sh", "-c", command, "standin", *a])


async def _terminal(mgr, task_id: str, seconds: float = 20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rec = mgr.get_task(task_id)
        if rec.status not in ("running", "pending"):
            return rec
        await asyncio.sleep(0.05)
    return mgr.get_task(task_id)


# ── the setting ──────────────────────────────────────────────────────────────

def test_the_template_ships_workers_timeout_minutes_60():
    template = yaml.safe_load((ROOT / "config" / "prometheus.yaml.default").read_text())
    assert template["workers"]["timeout_minutes"] == 60


@pytest.mark.parametrize("config, seconds", [
    ({"workers": {"timeout_minutes": 30}}, 1800),
    ({"workers": {"timeout_minutes": 240}}, 14400),
    ({"workers": {}}, 3600),
    ({}, 3600),
    (None, 3600),
])
def test_the_setting_is_read_in_minutes(config, seconds):
    from prometheus.tasks.worker_settings import worker_timeout_seconds

    assert worker_timeout_seconds(config) == seconds


@pytest.mark.parametrize("bad", [0, -5, "abc", True, 1.5, [60]])
def test_a_setting_that_is_not_a_whole_number_of_minutes_warns_and_uses_60(bad, caplog):
    from prometheus.tasks.worker_settings import worker_timeout_seconds

    caplog.set_level(logging.WARNING)
    assert worker_timeout_seconds({"workers": {"timeout_minutes": bad}}) == 3600
    assert any("workers.timeout_minutes" in r.getMessage() for r in caplog.records)


def test_the_daemon_hands_the_setting_to_the_task_manager():
    text = (ROOT / "src" / "prometheus" / "daemon.py").read_text()
    assert "task_manager.worker_timeout_seconds = worker_timeout_seconds(config)" in text


@pytest.mark.asyncio
async def test_a_worker_gets_the_worker_limit_and_other_tasks_keep_theirs(tmp_path, monkeypatch):
    from prometheus.tasks.manager import BackgroundTaskManager

    _sleeping_workers(monkeypatch)
    mgr = BackgroundTaskManager(default_timeout_seconds=900)
    mgr.worker_timeout_seconds = 5400
    try:
        worker = await mgr.create_agent_task(prompt="p", description="w", cwd=str(tmp_path), route="local",
                                             worker=True, session_id="beacon:abc")
        shell = await mgr.create_shell_task(command="sleep 30", description="s", cwd=str(tmp_path))
        named = await mgr.create_agent_task(prompt="p", description="n", cwd=str(tmp_path), route="local",
                                            worker=True, session_id="beacon:abc", timeout_seconds=7)
        assert worker.timeout_seconds == 5400
        assert shell.timeout_seconds == 900
        assert named.timeout_seconds == 7, "a caller of the manager that names a limit still gets it"
    finally:
        await mgr.shutdown()


def test_the_manager_default_worker_limit_is_an_hour():
    from prometheus.tasks.manager import BackgroundTaskManager

    assert BackgroundTaskManager().worker_timeout_seconds == 3600


@pytest.mark.asyncio
async def test_the_model_cannot_set_a_workers_time_limit(tmp_path, monkeypatch):
    """The likely source of the 15 minutes: task_create passed the model's timeout_seconds=900 straight through."""
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin import task_create as tc

    seen: dict = {}

    class Recorder:
        """Records what the tool asked for, then stops it (the tool answers a ValueError as an error result)."""

        async def create_agent_task(self, **kw):
            seen["agent"] = kw
            raise ValueError("recorded")

        async def create_shell_task(self, **kw):
            seen["shell"] = kw
            raise ValueError("recorded")

    monkeypatch.setattr(tc, "get_task_manager", lambda: Recorder())
    await tc.TaskCreateTool().execute(
        tc.TaskCreateToolInput(type="local_agent", description="w", prompt="research x", timeout_seconds=900),
        ToolExecutionContext(cwd=tmp_path, metadata=dict(_LOCAL_TURN)))
    await tc.TaskCreateTool().execute(
        tc.TaskCreateToolInput(type="local_bash", description="s", command="sleep 1", timeout_seconds=900),
        ToolExecutionContext(cwd=tmp_path, metadata=dict(_LOCAL_TURN)))
    assert seen["agent"].get("timeout_seconds") is None, seen["agent"]
    assert seen["shell"]["timeout_seconds"] == 900


def test_the_tool_tells_the_model_its_timeout_does_not_apply_to_workers():
    from prometheus.tools.builtin.task_create import TaskCreateToolInput

    text = TaskCreateToolInput.model_fields["timeout_seconds"].description
    assert "local_agent" in text and "workers.timeout_minutes" in text


# ── waiting for approval does not count ──────────────────────────────────────

@pytest.mark.asyncio
async def test_time_spent_waiting_for_approval_is_not_counted(tmp_path, monkeypatch):
    from prometheus.tasks.manager import BackgroundTaskManager

    _sleeping_workers(monkeypatch)
    mgr = BackgroundTaskManager()
    try:
        rec = await mgr.create_agent_task(prompt="p", description="w", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc", timeout_seconds=2)
        async with mgr.waiting_for_approval(rec.id):
            await asyncio.sleep(3)
            assert mgr.get_task(rec.id).status == "running", "the clock ran while it waited for a person"
        started = time.monotonic()
        done = await _terminal(mgr, rec.id)
        assert done.status == "failed" and done.error == "timeout"
        assert time.monotonic() - started < 2.5, "its own two seconds should have been nearly all left"
        assert float(done.metadata["approval_wait_seconds"]) >= 2.9
        assert 1.5 <= float(done.metadata["active_seconds"]) <= 2.6
        assert done.metadata["time_limit_seconds"] == "2"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_overlapping_waits_count_once(tmp_path, monkeypatch):
    from prometheus.tasks.manager import BackgroundTaskManager

    _sleeping_workers(monkeypatch)
    mgr = BackgroundTaskManager()
    try:
        rec = await mgr.create_agent_task(prompt="p", description="w", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc", timeout_seconds=2)

        async def wait(after: float, seconds: float) -> None:
            await asyncio.sleep(after)
            async with mgr.waiting_for_approval(rec.id):
                await asyncio.sleep(seconds)

        # One wait from 0 to 2 s, another from 1 to 2.5 s: the clock is stopped from 0 to 2.5 s, once.
        await asyncio.gather(wait(0, 2.0), wait(1.0, 1.5))
        done = await _terminal(mgr, rec.id)
        assert done.error == "timeout"
        assert 2.4 <= float(done.metadata["approval_wait_seconds"]) <= 3.0, done.metadata
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_worker_that_never_waits_is_stopped_on_time(tmp_path, monkeypatch):
    from prometheus.tasks.manager import BackgroundTaskManager

    _sleeping_workers(monkeypatch)
    mgr = BackgroundTaskManager()
    try:
        started = time.monotonic()
        rec = await mgr.create_agent_task(prompt="p", description="w", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc", timeout_seconds=1)
        done = await _terminal(mgr, rec.id)
        assert done.error == "timeout" and done.return_code == -9
        assert time.monotonic() - started < 3
        assert done.metadata["approval_wait_seconds"] == "0"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_the_daemon_stops_the_clock_while_a_workers_request_waits_for_a_person(tmp_path, monkeypatch):
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.permissions.approver import in_process
    from prometheus.tasks.manager import BackgroundTaskManager

    _sleeping_workers(monkeypatch)
    mgr = BackgroundTaskManager()
    mgr.worker_approval_url = "http://127.0.0.1:8005/api/workers/approvals"
    queue = ApprovalQueue(security_gate=_Gate())
    try:
        rec = await mgr.create_agent_task(prompt="p", description="w", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc", timeout_seconds=2)
        token = mgr._worker_tokens[rec.id]
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            req = asyncio.create_task(c.post("/api/workers/approvals", **_ask(rec.id, token)))
            pending = await _until(queue.list_pending)
            assert pending
            await asyncio.sleep(3)
            assert mgr.get_task(rec.id).status == "running", "the worker was killed while a person decided"
            await queue.approve(pending[0].request_id, by=in_process("a person"))
            assert (await req).json()["result"] == "approved"
        done = await _terminal(mgr, rec.id)
        assert done.error == "timeout" and float(done.metadata["approval_wait_seconds"]) >= 2.9
    finally:
        await mgr.shutdown()


# ── the report to the parent conversation ────────────────────────────────────

@pytest.mark.asyncio
async def test_the_completion_carries_the_limit_the_working_time_and_the_wait(tmp_path, monkeypatch):
    from prometheus.tasks.manager import BackgroundTaskManager

    _sleeping_workers(monkeypatch)
    mgr = BackgroundTaskManager()
    bus = _Bus()
    mgr.signal_bus = bus
    try:
        rec = await mgr.create_agent_task(prompt="p", description="w", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc", on_complete="reengage",
                                          timeout_seconds=1)
        await _terminal(mgr, rec.id)
        await _until(lambda: bus.signals)
        payload = bus.signals[-1].payload
        limit = payload["worker"]["time_limit"]
        assert limit["limit_seconds"] == 1 and limit["waited_seconds"] == 0
        assert 0.5 <= limit["active_seconds"] <= 2
    finally:
        await mgr.shutdown()


def _timeout_payload(**time_limit) -> dict:
    return {"task_id": "a0000test", "description": "research the EU AI Act", "status": "failed",
            "error": "timeout", "return_code": -9, "output_tail": "partial notes",
            "worker": {"route": "local", "time_limit": {"limit_seconds": 3600, "active_seconds": 3600.4,
                                                        "waited_seconds": 0, **time_limit}}}


def test_the_parent_is_told_plainly_that_the_worker_was_stopped_for_time():
    from prometheus.tasks.completion_handler import _worker_report

    text = _worker_report(_timeout_payload())
    assert "stopped after 60 minutes — time limit" in text
    assert "workers.timeout_minutes" in text
    assert "narrower" in text, "the agent is told what it can do about it"
    assert "partial notes" in text


def test_the_report_says_how_long_it_waited_for_approval_when_it_did():
    from prometheus.tasks.completion_handler import _worker_report

    text = _worker_report(_timeout_payload(active_seconds=900, limit_seconds=900, waited_seconds=600))
    assert "stopped after 15 minutes — time limit" in text
    assert "10 minutes waiting for approval did not count" in text


def test_a_short_limit_is_reported_in_seconds():
    from prometheus.tasks.completion_handler import _worker_report

    text = _worker_report(_timeout_payload(active_seconds=2.1, limit_seconds=2))
    assert "stopped after 2 seconds — time limit" in text


@pytest.mark.parametrize("active, words", [(60, "60 seconds"), (89, "89 seconds"), (120, "2 minutes"),
                                           (1, "1 second")])
def test_short_times_read_in_seconds_and_longer_ones_in_minutes(active, words):
    from prometheus.tasks.completion_handler import _worker_report

    assert f"stopped after {words} — time limit" in _worker_report(_timeout_payload(active_seconds=active))


def test_a_timeout_from_before_this_change_still_reads_sensibly():
    """A payload with no time_limit (an older record, a resumed task) keeps the old words."""
    from prometheus.tasks.completion_handler import _worker_report

    payload = _timeout_payload()
    del payload["worker"]["time_limit"]
    text = _worker_report(payload)
    assert "time limit" in text and "None" not in text
