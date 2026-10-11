# Provenance: pattern-inspired by openclaw/openclaw sessions-spawn-tool.ts
#             (concept/tool-name only; no code derived)
# Implementation: original Prometheus BaseTool wrapping the task manager

"""Spawn a new agent session."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tasks.worker_route import WorkerRouteRefused, plan_spawn
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class SessionsSpawnInput(BaseModel):
    """Arguments for spawning a new session."""

    prompt: str = Field(description="Initial prompt for the new agent session")
    description: str = Field(
        default="agent session", description="Short description of the session"
    )
    # The advertised text is pinned by the parity recording; the RULE is in
    # tasks/worker_route.py: a worker runs on its conversation's model or on
    # workers.model, and any other value here is refused with that reason.
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
        # The worker runs on this turn's route, or on the one the operator set;
        # never on one the model picked, and never on a substitute.
        try:
            plan = plan_spawn(context.metadata, requested_model=arguments.model)
        except WorkerRouteRefused as exc:
            return ToolResult(output=f"sessions_spawn refused: {exc}", is_error=True)
        try:
            task = await manager.create_agent_task(
                prompt=arguments.prompt,
                description=arguments.description,
                cwd=context.cwd,
                route=plan.key,
                metadata=plan.record(),
            )
        except (ValueError, RuntimeError) as exc:
            return ToolResult(output=f"sessions_spawn failed: {exc}", is_error=True)
        return ToolResult(
            output=f"Session spawned: {task.id} — {task.description}. {plan.note()}",
            metadata={"task_id": task.id, "worker_route": plan.key},
        )
