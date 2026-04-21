"""Tool for maintaining a project TODO checklist."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TodoWriteToolInput(BaseModel):
    """Arguments for TODO writes."""

    item: str = Field(description="TODO item text.")
    checked: bool = Field(default=False, description="Mark as already completed.")
    path: str = Field(default="TODO.md", description="Relative path to TODO file.")


class TodoWriteTool(BaseTool):
    """Append an item to a TODO markdown checklist file."""

    name = "todo_write"
    description = "Append a TODO item to a project markdown checklist."
    input_model = TodoWriteToolInput

    async def execute(self, arguments: TodoWriteToolInput, context: ToolExecutionContext) -> ToolResult:
        todo_path = Path(context.cwd) / arguments.path
        prefix = "- [x]" if arguments.checked else "- [ ]"
        existing = todo_path.read_text(encoding="utf-8") if todo_path.exists() else "# TODO\n"
        updated = existing.rstrip() + f"\n{prefix} {arguments.item}\n"
        todo_path.write_text(updated, encoding="utf-8")
        return ToolResult(output=f"Updated {todo_path}")
