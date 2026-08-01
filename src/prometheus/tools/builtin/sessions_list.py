# Provenance: pattern-inspired by openclaw/openclaw sessions-list-tool.ts
#             (concept/tool-name only; no code derived)
# Implementation: original Prometheus BaseTool wrapping the task manager

"""List active agent sessions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class SessionsListInput(BaseModel):
    """Arguments for listing sessions."""

    status: str | None = Field(
        default=None,
        description="Filter by status: running, completed, failed. Omit for all.",
    )


class SessionsListTool(BaseTool):
    """List active agent sessions and their status."""

    name = "sessions_list"
    description = (
        "List agent sessions (background tasks) with their IDs, status, "
        "and descriptions. Use to discover running agents."
    )
    input_model = SessionsListInput

    def is_read_only(self, arguments: SessionsListInput) -> bool:
        return True

    async def execute(
        self, arguments: SessionsListInput, context: ToolExecutionContext
    ) -> ToolResult:
        tasks = get_task_manager().list_tasks(status=arguments.status)
        if not tasks:
            return ToolResult(output="No active sessions.")

        lines = ["ID | Type | Status | Description"]
        lines.append("-" * 60)
        for t in tasks:
            lines.append(f"{t.id} | {t.type} | {t.status} | {t.description}")
        return ToolResult(output="\n".join(lines))
