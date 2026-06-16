"""Coding runs as managed tasks (SPRINT-coding-mode v2, scope item 5).

A coding run is a ``tasks.db`` managed task: the ``prometheus code`` CLI
subprocess IS the run, registered through the existing
:class:`~prometheus.tasks.manager.BackgroundTaskManager` — durability
(TaskStore), SecurityGate vetting at launch, ``task_completed`` /
``task_failed`` SignalBus emission, and the Telegram/Beacon completion
notification all come along for free. The run's JSON report lands in the
task's output file; the process exit code is the verdict (0 = acceptance
green).
"""

from __future__ import annotations

import shlex
import sys
from typing import Any

from prometheus.tasks.types import TaskRecord


def build_coding_command(
    *,
    repo: str,
    description: str,
    acceptance_command: str,
    task_id: str,
    max_rounds: int = 30,
    max_wall_seconds: int = 1_200,
    sandbox_parent: str | None = None,
    config_path: str | None = None,
    suppress_thinking: bool = False,
) -> str:
    """Build the shell command for one coding run, fully quoted.

    Uses the running interpreter so the managed subprocess gets the same
    environment (venv) the daemon runs under.
    """
    parts: list[str] = [
        sys.executable, "-m", "prometheus", "code",
        "--repo", repo,
        "--task", description,
        "--acceptance", acceptance_command,
        "--task-id", task_id,
        "--max-rounds", str(max_rounds),
        "--max-wall-seconds", str(max_wall_seconds),
    ]
    if sandbox_parent:
        parts += ["--sandbox-parent", sandbox_parent]
    if config_path:
        parts += ["--config", config_path]
    if suppress_thinking:
        parts += ["--suppress-thinking"]
    return " ".join(shlex.quote(p) for p in parts)


async def create_coding_managed_task(
    manager: Any,
    *,
    repo: str,
    description: str,
    acceptance_command: str,
    task_id: str,
    cwd: str,
    notify_target: str | None = None,
    max_rounds: int = 30,
    max_wall_seconds: int = 1_200,
    sandbox_parent: str | None = None,
    config_path: str | None = None,
    timeout_seconds: int | None = None,
) -> TaskRecord:
    """Register one coding run with the BackgroundTaskManager.

    The manager vets the command through the SecurityGate at system trust
    before spawning (same path as cron). ``timeout_seconds`` defaults to
    the wall cap plus headroom so the manager's own watchdog backstops a
    hung run without racing the session's honest abandonment.
    """
    command = build_coding_command(
        repo=repo,
        description=description,
        acceptance_command=acceptance_command,
        task_id=task_id,
        max_rounds=max_rounds,
        max_wall_seconds=max_wall_seconds,
        sandbox_parent=sandbox_parent,
        config_path=config_path,
    )
    return await manager.create_shell_task(
        command=command,
        description=f"coding task {task_id}: {description[:120]}",
        cwd=cwd,
        task_type="local_agent",
        # The coding telemetry session key — so the manager's terminal
        # task_completed/task_failed event self-identifies the run to the
        # CodingLiveStream tailer (feat/coding-livestream), no side map.
        session_id=f"coding:{task_id}",
        notify_target=notify_target,
        on_complete="notify",
        timeout_seconds=timeout_seconds or (max_wall_seconds + 300),
    )
