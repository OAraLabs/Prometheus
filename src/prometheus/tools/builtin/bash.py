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
    description = "Run a shell command in the local repository."
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

        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            arguments.command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=arguments.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ToolResult(
                output=f"Command timed out after {arguments.timeout_seconds} seconds",
                is_error=True,
            )

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
