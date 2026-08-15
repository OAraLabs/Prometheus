"""Tool for maintaining a project TODO checklist."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import DIR_FIELD, PATH_FIELD
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TodoWriteToolInput(BaseModel):
    """Arguments for TODO writes."""

    item: str = Field(description="TODO item text.")
    checked: bool = Field(default=False, description="Mark as already completed.")
    path: str = Field(
        json_schema_extra=PATH_FIELD,
        default=".prometheus/TODO.md",
        description=(
            "Relative path to TODO file. Defaults to a runtime scratchpad "
            "inside the (gitignored) .prometheus/ directory so automated "
            "tools (evals, benchmarks, hooks) don't pollute the project "
            "TODO.md. Pass 'TODO.md' explicitly to write to the repo root."
        ),
    )


class TodoWriteTool(BaseTool):
    """Append an item to a TODO markdown checklist file."""

    name = "todo_write"
    description = "Append a TODO item to a project markdown checklist."
    input_model = TodoWriteToolInput

    async def execute(self, arguments: TodoWriteToolInput, context: ToolExecutionContext) -> ToolResult:
        # CONFINEMENT. `path` is documented as RELATIVE to the working
        # directory, and this used to be a bare `Path(context.cwd) / path`:
        # an absolute path, or `../../../..`, escaped freely and the tool then
        # mkdir'd and wrote there — a write-anywhere primitive.
        #
        # It matters more than it looks: this tool is EXEMPT from the
        # SecurityGate's path check (permissions/tool_paths.PATH_PARAM_EXEMPT)
        # precisely because its `path` is a scratchpad-relative identifier
        # rather than a filesystem path. An exemption without its own guard is
        # a hole, so the guard lives here. Pinned by
        # tests/test_gate_sees_the_path.py::test_todo_write_confines_itself.
        root = Path(context.cwd).resolve()
        todo_path = (root / arguments.path).resolve()
        try:
            todo_path.relative_to(root)
        except ValueError:
            return ToolResult(
                output=(
                    f"path must stay under the working directory: "
                    f"{arguments.path!r} resolves outside {root}"
                ),
                is_error=True,
            )
        todo_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "- [x]" if arguments.checked else "- [ ]"
        existing = todo_path.read_text(encoding="utf-8") if todo_path.exists() else "# TODO\n"
        updated = existing.rstrip() + f"\n{prefix} {arguments.item}\n"
        todo_path.write_text(updated, encoding="utf-8")
        return ToolResult(output=f"Updated {todo_path}")
