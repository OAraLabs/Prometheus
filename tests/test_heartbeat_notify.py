"""Tests for Heartbeat proactive background-task notifications (audit fix #3).

Stubs the gateway and task manager; drives _check_task_transitions() with
asyncio.run() so there's no pytest-asyncio mode coupling and no network/daemon.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from prometheus.gateway.heartbeat import Heartbeat
from prometheus.tasks.types import TaskRecord


class _FakeGateway:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []
        self.running = True

    async def send(self, chat_id, text, **kw):  # noqa: ANN001
        self.sent.append((chat_id, text))
        return None


class _FakeManager:
    def __init__(self, tasks: list[TaskRecord]) -> None:
        self._tasks = tasks

    def set(self, tasks: list[TaskRecord]) -> None:
        self._tasks = tasks

    def list_tasks(self, *, status=None):  # noqa: ANN001
        if status is None:
            return list(self._tasks)
        return [t for t in self._tasks if t.status == status]


def _task(tid, status, *, started_at=None, ended_at=None, return_code=None, desc="demo"):
    return TaskRecord(
        id=tid,
        type="local_bash",
        status=status,
        description=desc,
        cwd="/tmp",
        output_file=Path("/tmp/out.log"),
        command="echo hi",
        created_at=started_at or time.time(),
        started_at=started_at,
        ended_at=ended_at,
        return_code=return_code,
    )


def _hb(mgr, gw, **kw):
    return Heartbeat(gateway=gw, task_manager=mgr, notify_chat_id=8139235390, **kw)


def test_notifies_on_completion_transition():
    gw, mgr = _FakeGateway(), _FakeManager([_task("t1", "running", started_at=time.time())])
    hb = _hb(mgr, gw)
    asyncio.run(hb._check_task_transitions())  # tick 1: seed running, no msg
    assert gw.sent == []
    mgr.set([_task("t1", "completed", started_at=time.time() - 5,
                   ended_at=time.time(), return_code=0)])
    asyncio.run(hb._check_task_transitions())  # tick 2: running -> completed
    assert len(gw.sent) == 1
    chat, text = gw.sent[0]
    assert chat == 8139235390
    assert "Task done" in text and "t1" in text


def test_notifies_on_failure_with_returncode():
    gw, mgr = _FakeGateway(), _FakeManager([_task("t2", "running", started_at=time.time())])
    hb = _hb(mgr, gw)
    asyncio.run(hb._check_task_transitions())
    mgr.set([_task("t2", "failed", started_at=time.time() - 3,
                   ended_at=time.time(), return_code=1)])
    asyncio.run(hb._check_task_transitions())
    assert len(gw.sent) == 1
    assert "failed" in gw.sent[0][1] and "rc=1" in gw.sent[0][1]


def test_no_spam_for_preexisting_terminal_task():
    # prev status is unknown (None) on first sight -> must NOT notify.
    gw = _FakeGateway()
    mgr = _FakeManager([_task("old", "completed", started_at=time.time() - 100,
                              ended_at=time.time() - 90, return_code=0)])
    hb = _hb(mgr, gw)
    asyncio.run(hb._check_task_transitions())
    assert gw.sent == []


def test_progress_ping_for_long_runner_then_throttled():
    gw = _FakeGateway()
    mgr = _FakeManager([_task("lr", "running", started_at=time.time() - 10_000)])
    hb = _hb(mgr, gw, task_progress_interval=600)
    asyncio.run(hb._check_task_transitions())  # baseline=start (10000s ago) -> due
    assert len(gw.sent) == 1 and "Still running" in gw.sent[0][1]
    asyncio.run(hb._check_task_transitions())  # immediately again -> throttled
    assert len(gw.sent) == 1


def test_fresh_running_task_not_pinged_immediately():
    gw = _FakeGateway()
    mgr = _FakeManager([_task("fresh", "running", started_at=time.time())])
    hb = _hb(mgr, gw, task_progress_interval=600)
    asyncio.run(hb._check_task_transitions())
    assert gw.sent == []


def test_no_notify_without_chat_id():
    gw = _FakeGateway()
    mgr = _FakeManager([_task("t", "running", started_at=time.time())])
    hb = Heartbeat(gateway=gw, task_manager=mgr, notify_chat_id=None)
    asyncio.run(hb._check_task_transitions())
    mgr.set([_task("t", "completed", started_at=time.time() - 1,
                   ended_at=time.time(), return_code=0)])
    asyncio.run(hb._check_task_transitions())
    assert gw.sent == []
