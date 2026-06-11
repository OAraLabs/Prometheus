"""Tool for creating managed background tasks.

Agent-facing replacement for raw ``nohup … &``. A task is registered, its id is
returned immediately (the turn can end), and the daemon's non-LLM supervisor
detects completion and (a) always notifies via Telegram and (b) optionally
re-engages the agent with the result.

The notify target and creating-session id are resolved from the TRUSTED
execution context (threaded from the agent loop), never from tool arguments —
arguments could originate in observed/injected content.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TaskCreateToolInput(BaseModel):
    """Arguments for task creation."""

    type: str = Field(
        default="local_bash",
        description="Task type: 'local_bash', 'local_agent', 'file_watch', or 'poll'.",
    )
    description: str = Field(description="Short human-readable task description.")
    command: str | None = Field(default=None, description="Shell command (local_bash only).")
    prompt: str | None = Field(default=None, description="Agent prompt (local_agent only).")
    model: str | None = Field(default=None, description="Model override for agent tasks.")
    watch_dir: str | None = Field(
        default=None, description="Directory to watch (file_watch only)."
    )
    watch_pattern: str | None = Field(
        default=None,
        description="Glob pattern for the file to wait for, e.g. '*.done' (file_watch only).",
    )
    poll_predicate: str | None = Field(
        default=None,
        description="Shell command polled until it exits 0 (poll only; fallback detector).",
    )
    on_complete: str = Field(
        default="notify",
        description=(
            "What to do on completion: 'notify' (Telegram only), 'reengage' "
            "(also inject the result back to the agent), or 'both'."
        ),
    )
    reengage_prompt: str | None = Field(
        default=None,
        description="Optional template for the synthetic re-engagement turn.",
    )
    timeout_seconds: int | None = Field(
        default=None,
        description="Hard ceiling; on expiry the task is marked failed (timeout).",
    )


# Phase 4 experiment gate (default OFF — model-facing error feedback ships
# config-gated per sprint rule 7; the gym arbitrates). When ON, mode-misuse
# errors name the ACTUAL mistake: live history (D2) shows the model supplies
# `prompt` but omits `type`, gets "'command' is required for local_bash
# tasks" (the silent default's complaint), and flails — JSON-stuffing the
# prompt, dict-wrapping params — without ever learning about `type`.
_HONEST_MODE_ERRORS = False


def set_honest_mode_errors(enabled: bool) -> None:
    global _HONEST_MODE_ERRORS
    _HONEST_MODE_ERRORS = bool(enabled)


def _mode_error(arguments: "TaskCreateToolInput", missing: str, mode: str) -> str:
    """Build the mode-misuse error message (honest variant behind the gate)."""
    if not _HONEST_MODE_ERRORS:
        if missing == "watch_dir and watch_pattern":
            return f"'watch_dir' and 'watch_pattern' are required for {mode} tasks"
        return f"'{missing}' is required for {mode} tasks"
    hints = [
        f"'{missing}' is required for type='{mode}'"
        + (" (the default)" if mode == "local_bash" else "")
        + "."
    ]
    if mode == "local_bash" and arguments.prompt:
        hints.append(
            "You supplied 'prompt' — for an agent task pass type='local_agent'."
        )
    hints.append(
        "Valid types: local_bash (needs command), local_agent (needs prompt), "
        "file_watch (needs watch_dir + watch_pattern), poll (needs poll_predicate)."
    )
    hints.append(
        'Example: {"name": "task_create", "arguments": {"type": "local_agent", '
        '"description": "Summarize report", "prompt": "Read /data/q3.csv and '
        'summarize revenue trends."}}'
    )
    return " ".join(hints)


def _notify_target_from_session(session_id: str | None) -> str | None:
    """Derive a Telegram chat id from the creating session id.

    Sessions are keyed like ``telegram:<chat_id>``. Only Telegram targets are
    resolved here; other platforms return None and notification falls back to the
    heartbeat's globally-configured chat.
    """
    if not session_id:
        return None
    if session_id.startswith("telegram:"):
        return session_id.split(":", 1)[1] or None
    return None


class TaskCreateTool(BaseTool):
    """Create a managed background task (shell, agent, file-watch, or poll)."""

    name = "task_create"
    description = (
        "Create a managed background task instead of running a long job with "
        "'nohup … &'. Types: local_bash (shell command), local_agent (sub-agent "
        "prompt), file_watch (wait for a file matching a glob), poll (run a "
        "predicate command until it succeeds). The daemon detects completion and "
        "sends a Telegram notification; set on_complete='reengage'/'both' to also "
        "have the agent act on the result. Returns a task id immediately."
    )
    input_model = TaskCreateToolInput

    async def execute(
        self, arguments: TaskCreateToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        manager = get_task_manager()

        # Trusted context — NOT from arguments.
        session_id = context.metadata.get("session_id") if context.metadata else None
        notify_target = _notify_target_from_session(session_id)

        on_complete = arguments.on_complete if arguments.on_complete in (
            "notify", "reengage", "both",
        ) else "notify"

        common = dict(
            description=arguments.description,
            cwd=context.cwd,
            session_id=session_id,
            notify_target=notify_target,
            on_complete=on_complete,
            reengage_prompt=arguments.reengage_prompt,
            timeout_seconds=arguments.timeout_seconds,
        )

        try:
            if arguments.type == "local_bash":
                if not arguments.command:
                    return ToolResult(output=_mode_error(arguments, "command", "local_bash"), is_error=True)
                task = await manager.create_shell_task(command=arguments.command, **common)
            elif arguments.type == "local_agent":
                if not arguments.prompt:
                    return ToolResult(output=_mode_error(arguments, "prompt", "local_agent"), is_error=True)
                task = await manager.create_agent_task(
                    prompt=arguments.prompt,
                    model=arguments.model,
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    **common,
                )
            elif arguments.type == "file_watch":
                if not arguments.watch_dir or not arguments.watch_pattern:
                    return ToolResult(
                        output=_mode_error(arguments, "watch_dir and watch_pattern", "file_watch"),
                        is_error=True,
                    )
                task = await manager.create_file_watch_task(
                    watch_dir=arguments.watch_dir,
                    watch_pattern=arguments.watch_pattern,
                    **common,
                )
            elif arguments.type == "poll":
                if not arguments.poll_predicate:
                    return ToolResult(output=_mode_error(arguments, "poll_predicate", "poll"), is_error=True)
                task = await manager.create_poll_task(
                    poll_predicate=arguments.poll_predicate, **common
                )
            else:
                return ToolResult(output=f"unsupported task type: {arguments.type}", is_error=True)
        except ValueError as exc:
            return ToolResult(output=str(exc), is_error=True)

        if task.status == "failed":
            return ToolResult(
                output=f"Task rejected ({task.id}): {task.error or 'unknown reason'}",
                is_error=True,
            )

        notify_note = (
            "I'll notify you on Telegram when it finishes"
            if notify_target
            else "completion will be reported to the configured chat"
        )
        reengage_note = (
            " and re-engage with the result" if on_complete in ("reengage", "both") else ""
        )
        return ToolResult(
            output=(
                f"Started managed task {task.id} ({task.type}): {task.description}. "
                f"{notify_note}{reengage_note}."
            ),
            metadata={"task_id": task.id, "task_type": task.type, "on_complete": on_complete},
        )
