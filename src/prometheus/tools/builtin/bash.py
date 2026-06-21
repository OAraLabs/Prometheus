# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/bash_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus);
#           added workspace_root locking (refuses commands outside allowed dir);
#           added configurable timeout (default 30s);
#           added output truncation (default 10000 chars)

"""Shell command execution tool."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 10_000


class BashToolInput(BaseModel):
    """Arguments for the bash tool."""

    command: str = Field(description="Shell command to execute")
    cwd: str | None = Field(default=None, description="Working directory override")
    timeout_seconds: int = Field(default=_DEFAULT_TIMEOUT, ge=1, le=600)


class BashTool(BaseTool):
    """Execute a shell command with stdout/stderr capture.

    Optionally locked to a workspace_root: commands whose resolved cwd falls
    outside the workspace are refused before execution.
    """

    name = "bash"
    description = (
        "Run a short shell command in the local repository. For work that "
        "outlives this turn (long builds, downloads, file watches) use "
        "task_create instead of 'nohup … &' — a backgrounded Bash job is killed "
        "at the timeout and left untracked."
    )
    input_model = BashToolInput
    example_call = {"command": "ls -la"}

    def __init__(
        self,
        workspace: str | Path | None = None,
        max_output: int = _DEFAULT_MAX_OUTPUT,
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve() if workspace else None
        self._max_output = max_output

    async def execute(self, arguments: BashToolInput, context: ToolExecutionContext) -> ToolResult:
        cwd = Path(arguments.cwd).expanduser().resolve() if arguments.cwd else context.cwd.resolve()

        if self._workspace is not None:
            try:
                cwd.relative_to(self._workspace)
            except ValueError:
                if not arguments.cwd:
                    # No explicit cwd requested — fall back to workspace root
                    # instead of blocking (daemon often runs from repo dir)
                    cwd = self._workspace
                else:
                    return ToolResult(
                        output=(
                            f"Workspace lock violation: {cwd} is outside "
                            f"allowed workspace {self._workspace}"
                        ),
                        is_error=True,
                    )

        # start_new_session=True puts the shell in its own process group, so the
        # ENTIRE pipeline (e.g. ``find … | grep …``) can be killed as a unit. A
        # bare process.kill() only signals /bin/bash; its children get reparented
        # to init and keep running — that is how a timed-out ``find`` orphaned
        # itself and thrashed the disk for minutes after the turn moved on.
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            arguments.command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=arguments.timeout_seconds,
            )
        except asyncio.TimeoutError:
            await self._kill_process_group(process)
            return ToolResult(
                output=f"Command timed out after {arguments.timeout_seconds} seconds",
                is_error=True,
            )
        except asyncio.CancelledError:
            # The agent loop wraps tool.execute() in its own (longer) timeout;
            # when THAT fires it cancels us instead of raising TimeoutError here.
            # Without this handler the subprocess and its children would keep
            # running after the turn was abandoned (the original freeze: an
            # orphaned ``find`` still scanning $HOME). Kill the whole group,
            # then let the cancellation propagate.
            await self._kill_process_group(process)
            raise

        parts = []
        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace").rstrip())
        if stderr:
            parts.append(stderr.decode("utf-8", errors="replace").rstrip())

        text = "\n".join(part for part in parts if part).strip()
        if not text:
            text = "(no output)"

        if len(text) > self._max_output:
            text = f"{text[:self._max_output]}\n...[truncated]..."

        return ToolResult(
            output=text,
            is_error=process.returncode != 0,
            metadata={"returncode": process.returncode},
        )

    @staticmethod
    async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        """SIGKILL the shell's whole process group, then reap it.

        The shell is launched with ``start_new_session=True`` so it leads its own
        group; killing the group takes the entire pipeline (``find``, ``grep``,
        …) with it rather than leaving orphans. Best-effort and idempotent: the
        process or group may already be gone.
        """
        if process.returncode is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Already dead, or we couldn't address the group — fall back to
            # signalling the shell directly so we don't leak it.
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 — reaping must never raise
            pass
