"""A daemon restart stops the task processes it can no longer supervise.

WHAT WAS WRONG (diagnosed 2026-09-20, wiki/log.md)
--------------------------------------------------
``resume_running()`` reaped a restarted daemon's ``running`` process tasks
to ``failed: daemon_restart`` — the RECORD only. The process itself was
started with ``start_new_session=True`` and so outlived the daemon, alive
and untracked: a quiet one (a worker waiting on its model, ``sleep``) ran on
indefinitely, a chatty one died of SIGPIPE the next time it wrote. And no
later daemon could find it, because nothing recorded its pid.

WILL'S RULING (2026-09-20, restated 2026-10-11)
-----------------------------------------------
"Kill on reap, record the pid regardless, since you can't kill what you
can't find."

THE RULE THESE TESTS PIN
------------------------
* Every task process records its pid and a start marker (its kernel start
  time) when it is launched, in the durable record.
* On restart, a ``running`` process task whose recorded pid is still THAT
  process (same start marker) is stopped with its whole tree — its process
  group and every descendant, including ones that started their own session
  (a worker's own tool calls) — SIGTERM, then SIGKILL after a grace period.
* A pid that is gone, or that now belongs to some other process (pid
  reuse: the start marker differs), is left alone. Nothing is signalled
  that cannot be shown to be ours.
* The record says what happened, and a worker's parent conversation is told
  plainly that the daemon restarted under it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")

_HELPER = r'''
import asyncio, json, os, sys
from pathlib import Path
from prometheus.tasks.manager import BackgroundTaskManager
from prometheus.tasks.store import TaskStore

async def main():
    mgr = BackgroundTaskManager(store=TaskStore(Path(sys.argv[1])))
    rec = await mgr.create_shell_task(command=sys.argv[2], description="orphan",
                                      cwd=sys.argv[3], floored=False)
    await asyncio.sleep(1.0)  # let the command fork its children
    print(json.dumps({"id": rec.id, "pid": mgr._processes[rec.id].pid}), flush=True)
    os._exit(0)  # the daemon dies: no shutdown, nothing killed

asyncio.run(main())
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _tree(root: int) -> list[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True)
    kids: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen, todo = [root], [root]
    while todo:
        for child in kids.get(todo.pop(), []):
            seen.append(child)
            todo.append(child)
    return seen


def _orphan(tmp_path: Path, command: str) -> tuple[Path, str, list[int]]:
    """A task started by a 'daemon' process that then died. Returns the store,
    the task id and every pid in the orphan's tree."""
    db = tmp_path / "tasks.db"
    helper = tmp_path / "helper.py"
    helper.write_text(_HELPER, encoding="utf-8")
    out = subprocess.run([sys.executable, str(helper), str(db), command, str(tmp_path)],
                         capture_output=True, text=True, timeout=60, env=os.environ.copy())
    assert out.returncode == 0, out.stderr[-800:]
    info = json.loads(out.stdout.strip().splitlines()[-1])
    pids = _tree(int(info["pid"]))
    assert all(_alive(p) for p in pids), "the orphan should outlive its daemon"
    return db, info["id"], pids


def _kill_quietly(pids: list[int]) -> None:
    for p in pids:
        try:
            os.kill(p, 9)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_a_task_records_its_pid_and_start_marker(tmp_path):
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore

    store = TaskStore(tmp_path / "tasks.db")
    mgr = BackgroundTaskManager(store=store)
    rec = await mgr.create_shell_task(command="sleep 30", description="d",
                                      cwd=str(tmp_path), floored=False)
    try:
        pid = mgr._processes[rec.id].pid
        stored = store.get(rec.id)
        assert stored is not None
        assert stored.metadata.get("pid") == str(pid), stored.metadata
        assert stored.metadata.get("pid_started"), stored.metadata
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_restart_stops_the_orphaned_process_and_its_whole_tree(tmp_path):
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore

    # Two children in the task's process group, and one in a session of its
    # own — what a worker's own tool calls look like.
    command = (f"sleep 120 & {sys.executable} -c 'import os, time; os.setsid(); "
               f"time.sleep(120)' & sleep 120")
    db, task_id, pids = _orphan(tmp_path, command)
    try:
        mgr = BackgroundTaskManager(store=TaskStore(db))
        await mgr.resume_running()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and any(_alive(p) for p in pids):
            await asyncio.sleep(0.2)
        survivors = [p for p in pids if _alive(p)]
        rec = mgr.get_task(task_id)
    finally:
        _kill_quietly(pids)
    assert not survivors, f"{len(survivors)} of {len(pids)} orphaned process(es) still running"
    assert rec.status == "failed" and rec.error == "daemon_restart", (rec.status, rec.error)
    assert rec.metadata.get("restart_reap") == "stopped", rec.metadata


@pytest.mark.asyncio
async def test_a_pid_that_is_no_longer_ours_is_left_alone(tmp_path):
    """Pid reuse: the recorded pid now belongs to an unrelated process."""
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore
    from prometheus.tasks.types import TaskRecord

    stranger = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        store = TaskStore(tmp_path / "tasks.db")
        store.upsert(TaskRecord(
            id="b00000001", type="local_bash", status="running", description="d",
            cwd=str(tmp_path), output_file=tmp_path / "b00000001.log", command="sleep 60",
            metadata={"pid": str(stranger.pid), "pid_started": "not-this-process"}))
        mgr = BackgroundTaskManager(store=store)
        await mgr.resume_running()
        await asyncio.sleep(1.0)
        assert stranger.poll() is None, "an unrelated process was signalled"
        rec = mgr.get_task("b00000001")
        assert rec.status == "failed" and rec.error == "daemon_restart"
        assert "left alone" in rec.metadata.get("restart_reap", ""), rec.metadata
    finally:
        stranger.kill()
        stranger.wait()


@pytest.mark.asyncio
async def test_a_task_whose_process_is_already_gone_is_reaped_as_before(tmp_path):
    from prometheus.tasks.manager import BackgroundTaskManager
    from prometheus.tasks.store import TaskStore
    from prometheus.tasks.types import TaskRecord

    store = TaskStore(tmp_path / "tasks.db")
    store.upsert(TaskRecord(
        id="b00000002", type="local_bash", status="running", description="d",
        cwd=str(tmp_path), output_file=tmp_path / "b00000002.log", command="true",
        metadata={"pid": "999999999", "pid_started": "x"}))
    mgr = BackgroundTaskManager(store=store)
    await mgr.resume_running()
    rec = mgr.get_task("b00000002")
    assert rec.status == "failed" and rec.error == "daemon_restart"
    assert rec.metadata.get("restart_reap") == "already gone", rec.metadata


def test_a_reaped_workers_report_says_the_daemon_restarted():
    from prometheus.tasks.completion_handler import _build_reengage_content

    text = _build_reengage_content({
        "task_id": "a1", "description": "worker", "status": "failed",
        "error": "daemon_restart", "output_tail": "",
        "worker": {"route": "local", "parent_route": "local", "retry": None,
                   "retry_skipped": None},
        "worker_result": None,
    })
    assert "did not finish" in text and "restarted" in text, text
