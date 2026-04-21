"""Tests for the tasks module."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from prometheus.tasks.manager import BackgroundTaskManager, _task_id
from prometheus.tasks.types import TaskRecord, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manager_with_tmp(tmp: str) -> BackgroundTaskManager:
    """Return a BackgroundTaskManager with tasks dir overridden to tmp."""
    mgr = BackgroundTaskManager()
    return mgr


# ---------------------------------------------------------------------------
# TaskRecord
# ---------------------------------------------------------------------------


def test_task_record_defaults():
    rec = TaskRecord(
        id="b12345678",
        type="local_bash",
        status="pending",
        description="test task",
        cwd="/tmp",
        output_file=Path("/tmp/b12345678.log"),
    )
    assert rec.command is None
    assert rec.return_code is None
    assert rec.metadata == {}


# ---------------------------------------------------------------------------
# _task_id
# ---------------------------------------------------------------------------


def test_task_id_prefix_bash():
    tid = _task_id("local_bash")
    assert tid.startswith("b")
    assert len(tid) == 9  # "b" + 8 hex chars


def test_task_id_prefix_agent():
    assert _task_id("local_agent").startswith("a")


def test_task_id_prefix_remote():
    assert _task_id("remote_agent").startswith("r")


def test_task_id_prefix_teammate():
    assert _task_id("in_process_teammate").startswith("t")


def test_task_ids_are_unique():
    ids = {_task_id("local_bash") for _ in range(100)}
    assert len(ids) == 100


# ---------------------------------------------------------------------------
# BackgroundTaskManager — create_shell_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_shell_task():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(
                command="echo hello",
                description="Say hello",
                cwd=tmp,
            )
            assert task.type == "local_bash"
            assert task.status == "running"
            assert task.description == "Say hello"
            assert task.id.startswith("b")
            # Wait for completion
            await asyncio.sleep(0.3)
            task = mgr.get_task(task.id)
            assert task.status == "completed"
            assert task.return_code == 0


@pytest.mark.asyncio
async def test_shell_task_output():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(
                command="echo 'output text'",
                description="Echo test",
                cwd=tmp,
            )
            await asyncio.sleep(0.3)
            output = mgr.read_task_output(task.id)
            assert "output text" in output


@pytest.mark.asyncio
async def test_shell_task_failing_command():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(
                command="exit 1",
                description="Failing task",
                cwd=tmp,
            )
            await asyncio.sleep(0.3)
            task = mgr.get_task(task.id)
            assert task.status == "failed"
            assert task.return_code == 1


# ---------------------------------------------------------------------------
# BackgroundTaskManager — list / get / update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_tasks():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            await mgr.create_shell_task(command="echo a", description="A", cwd=tmp)
            await mgr.create_shell_task(command="echo b", description="B", cwd=tmp)
            tasks = mgr.list_tasks()
            assert len(tasks) == 2


@pytest.mark.asyncio
async def test_list_tasks_filtered_by_status():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(command="echo x", description="X", cwd=tmp)
            await asyncio.sleep(0.3)
            completed = mgr.list_tasks(status="completed")
            assert any(t.id == task.id for t in completed)
            running = mgr.list_tasks(status="running")
            assert all(t.id != task.id for t in running)


@pytest.mark.asyncio
async def test_get_task_returns_none_for_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            assert mgr.get_task("nonexistent") is None


@pytest.mark.asyncio
async def test_update_task_description():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(
                command="sleep 10",
                description="Original",
                cwd=tmp,
            )
            updated = mgr.update_task(task.id, description="Updated")
            assert updated.description == "Updated"
            # cleanup
            await mgr.stop_task(task.id)


@pytest.mark.asyncio
async def test_update_task_progress():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(
                command="sleep 10",
                description="Progress test",
                cwd=tmp,
            )
            mgr.update_task(task.id, progress=42)
            assert mgr.get_task(task.id).metadata["progress"] == "42"
            await mgr.stop_task(task.id)


# ---------------------------------------------------------------------------
# BackgroundTaskManager — stop_task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_task():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            task = await mgr.create_shell_task(
                command="sleep 60",
                description="Long task",
                cwd=tmp,
            )
            assert task.status == "running"
            stopped = await mgr.stop_task(task.id)
            assert stopped.status == "killed"


@pytest.mark.asyncio
async def test_stop_nonexistent_task_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with patch("prometheus.tasks.manager.get_tasks_dir", return_value=Path(tmp)):
            mgr = BackgroundTaskManager()
            with pytest.raises(ValueError, match="No task found"):
                await mgr.stop_task("nonexistent-id")
