# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/file_edit_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""String-based file editing tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import PATH_FIELD_WRITE
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class FileEditToolInput(BaseModel):
    """Arguments for the file edit tool."""

    path: str = Field(
        json_schema_extra=PATH_FIELD_WRITE,description="Path of the file to edit")
    old_str: str = Field(description="Existing text to replace")
    new_str: str = Field(description="Replacement text")
    replace_all: bool = Field(
        default=False,
        description=(
            "By default ONLY THE FIRST occurrence of old_str is replaced, and "
            "the edit fails if old_str is not unique. Set true to replace "
            "every occurrence."
        ),
    )


class FileEditTool(BaseTool):
    """Replace text in an existing file."""

    name = "edit_file"
    description = "Edit an existing file by replacing a string."
    input_model = FileEditToolInput
    example_call = {"path": "/path/to/file", "old_str": "before", "new_str": "after"}

    async def execute(
        self,
        arguments: FileEditToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        path = _resolve_path(context.cwd, arguments.path)
        if not path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        original = path.read_text(encoding="utf-8")
        # The advertised schema has long promised that "the edit fails if old_str is not
        # unique". It never did: the code replaced the FIRST match and reported success, so an
        # ambiguous edit silently changed the wrong place. The description was added deliberately
        # — its own test calls replace_all "a correctness question, not merely an efficiency one"
        # — which made the advertisement true-sounding and left the behaviour wrong.
        #
        # The caller that makes this urgent is an agent ticking its own checklist: plan steps share
        # phrasing constantly ("- [ ] Add smoke tests"), so first-match-wins ticks the wrong box
        # with no signal to anyone. Semantics and wording match code_str_replace, the primitive in
        # this repo that has always got it right.
        count = original.count(arguments.old_str)
        if count == 0:
            return ToolResult(output="old_str was not found in the file", is_error=True)
        if count > 1 and not arguments.replace_all:
            return ToolResult(
                output=(
                    f"{count} MATCHES in {arguments.path}: old_str is ambiguous. "
                    f"Nothing was changed. Include more surrounding lines in old_str "
                    f"until it is unique, or set replace_all to change every occurrence."
                ),
                is_error=True,
            )

        if arguments.replace_all:
            updated = original.replace(arguments.old_str, arguments.new_str)
        else:
            updated = original.replace(arguments.old_str, arguments.new_str, 1)

        path.write_text(updated, encoding="utf-8")
        return ToolResult(output=f"Updated {path}")


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
