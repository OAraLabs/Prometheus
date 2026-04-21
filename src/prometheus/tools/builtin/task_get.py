"""Tool for retrieving task details."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TaskGetToolInput(BaseModel):
    """Arguments for task lookup."""

    task_id: str = Field(description="Task identifier.")


class TaskGetTool(BaseTool):
    """Return full details for a background task."""

    name = "task_get"
    description = "Get the status and details for a background task."
    input_model = TaskGetToolInput

    def is_read_only(self, arguments: TaskGetToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: TaskGetToolInput, context: ToolExecutionContext) -> ToolResult:
        del context
        task = get_task_manager().get_task(arguments.task_id)
        if task is None:
            return ToolResult(output=f"No task found with ID: {arguments.task_id}", is_error=True)
        lines = [
            f"id:          {task.id}",
            f"type:        {task.type}",
            f"status:      {task.status}",
            f"description: {task.description}",
            f"cwd:         {task.cwd}",
        ]
        if task.command:
            lines.append(f"command:     {task.command}")
        if task.return_code is not None:
            lines.append(f"return_code: {task.return_code}")
        if task.metadata:
            for k, v in task.metadata.items():
                lines.append(f"{k}: {v}")
        return ToolResult(output="\n".join(lines))
