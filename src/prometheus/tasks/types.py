"""Task data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# "process" tasks are spawned as local_bash; "local_agent"/"remote_agent"/
# "in_process_teammate" are agent subprocesses. Managed-tasks sprint adds the
# two non-process detection kinds: "file_watch" (watchdog) and "poll" (fallback).
TaskType = Literal[
    "local_bash",
    "local_agent",
    "remote_agent",
    "in_process_teammate",
    "file_watch",
    "poll",
]
# "blocked" is a REFUSAL, not a failure: the SecurityGate denied the command and
# no process was ever launched. It is terminal exactly like "failed" and is
# deliberately distinct from it — "the gate did its job" and "the command ran and
# broke" are different states, repaired by different actions (grant approval or
# narrow the command, versus debug), so a tally that merges them answers neither.
TaskStatus = Literal[
    "pending", "running", "completed", "failed", "killed", "blocked"
]

# What to do when a managed task resolves. Notification ALWAYS fires (heartbeat);
# ``on_complete`` only gates re-engagement of the agent.
OnComplete = Literal["notify", "reengage", "both"]

# Statuses that mean the task has stopped running. "blocked" belongs here because
# a refused task is as finished as a failed one: the supervisor must not resume or
# reap it on restart, and a reader asking "is this still going?" must get "no".
TERMINAL_STATUSES = frozenset({"completed", "failed", "killed", "blocked"})


@dataclass
class TaskRecord:
    """Runtime representation of a background task."""

    id: str
    type: TaskType
    status: TaskStatus
    description: str
    cwd: str
    output_file: Path
    command: str | None = None
    prompt: str | None = None
    created_at: float = 0.0
    started_at: float | None = None
    ended_at: float | None = None
    return_code: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    # ── Managed-tasks sprint additions (all defaulted; backward-compatible) ──
    # Who spawned the task — used to route notification + re-engagement. Resolved
    # from the session context AT CREATION, never from job output/arguments.
    session_id: str | None = None
    # Resolved notify destination (e.g. a Telegram chat id) captured at creation.
    notify_target: str | None = None
    on_complete: OnComplete = "notify"
    # Optional template for the synthetic re-engagement turn.
    reengage_prompt: str | None = None
    # Hard ceiling; on expiry the task is marked failed with error "timeout".
    timeout_seconds: int | None = None
    # For file_watch: the matched file. For process: an optional declared artifact.
    artifact_path: str | None = None
    # Why the task ended without succeeding. For status "blocked" this is the
    # gate's own "blocked: <reason>" text; for "failed" it is "timeout",
    # "daemon_restart", or an exception string.
    error: str | None = None
    # Detector parameters for non-process kinds:
    #   file_watch: {"dir": str, "pattern": str}
    #   poll:       {"predicate_cmd": str}
    spec: dict[str, Any] = field(default_factory=dict)
