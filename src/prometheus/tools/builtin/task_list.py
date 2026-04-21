"""Tool for listing background tasks."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TaskListToolInput(BaseModel):
    """Arguments for task listing."""

    status: str | None = Field(
        default=None,
        description="Optional status filter: running, completed, failed, killed.",
    )


class TaskListTool(BaseTool):
    """List background tasks."""

    name = "task_list"
    description = "List all background tasks, optionally filtered by status."
    input_model = TaskListToolInput

    def is_read_only(self, arguments: TaskListToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: TaskListToolInput, context: ToolExecutionContext) -> ToolResult:
        del context
        tasks = get_task_manager().list_tasks(status=arguments.status)  # type: ignore[arg-type]
        if not tasks:
            return ToolResult(output="(no tasks)")
        lines = [
            f"{t.id}  {t.type:<20}  {t.status:<10}  {t.description}"
            for t in tasks
        ]
        return ToolResult(output="\n".join(lines))
