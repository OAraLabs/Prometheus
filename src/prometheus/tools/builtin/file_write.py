# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/file_write_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""File writing tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import DIR_FIELD, PATH_FIELD
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class FileWriteToolInput(BaseModel):
    """Arguments for the file write tool."""

    path: str = Field(
        json_schema_extra=PATH_FIELD,description="Path of the file to write")
    content: str = Field(description="Full file contents")
    create_directories: bool = Field(
        default=True,
        description=(
            "Missing parent directories are created by default, so a write to "
            "a new folder succeeds without a preceding mkdir."
        ),
    )


class FileWriteTool(BaseTool):
    """Write complete file contents."""

    name = "write_file"
    description = (
        "Create or overwrite a file with UTF-8 text content, written verbatim. "
        "Handles any text-based format (.md, .py, .json, .csv, .svg, .html). "
        "For binary or converted formats (.pdf, .docx, .xlsx, images), write "
        "the source as text with this tool, then convert it via bash "
        "(e.g. pandoc, python-docx, libreoffice) — do not refuse these formats."
    )
    input_model = FileWriteToolInput
    example_call = {"file_path": "/path/to/file", "content": "file contents"}

    async def execute(
        self,
        arguments: FileWriteToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        path = _resolve_path(context.cwd, arguments.path)
        if arguments.create_directories:
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments.content, encoding="utf-8")
        return ToolResult(output=f"Wrote {path}")


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
