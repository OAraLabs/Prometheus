# Provenance: openclaw/openclaw (https://github.com/openclaw/openclaw)
# Original: src/agents/tools/sessions-spawn-tool.ts
# License: MIT
# Modified: Rewritten as Prometheus BaseTool wrapping task manager

"""Spawn a new agent session."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class SessionsSpawnInput(BaseModel):
    """Arguments for spawning a new session."""

    prompt: str = Field(description="Initial prompt for the new agent session")
    description: str = Field(
        default="agent session", description="Short description of the session"
    )
    model: str | None = Field(
        default=None, description="Model override for this session"
    )


class SessionsSpawnTool(BaseTool):
    """Spawn a new agent session as a background task."""

    name = "sessions_spawn"
    description = (
        "Create a new background agent session with the given prompt. "
        "Returns the session/task ID for tracking."
    )
    input_model = SessionsSpawnInput

    async def execute(
        self, arguments: SessionsSpawnInput, context: ToolExecutionContext
    ) -> ToolResult:
        manager = get_task_manager()
        try:
            task = await manager.create_agent_task(
                prompt=arguments.prompt,
                description=arguments.description,
                cwd=context.cwd,
                model=arguments.model,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        except (ValueError, RuntimeError) as exc:
            return ToolResult(output=f"sessions_spawn failed: {exc}", is_error=True)
        return ToolResult(
            output=f"Session spawned: {task.id} — {task.description}",
            metadata={"task_id": task.id},
        )
