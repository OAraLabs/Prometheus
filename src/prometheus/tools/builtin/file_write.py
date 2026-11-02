# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/file_write_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""File writing tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import PATH_FIELD_WRITE
from prometheus.security import safe_fs
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class FileWriteToolInput(BaseModel):
    """Arguments for the file write tool."""

    path: str = Field(
        json_schema_extra=PATH_FIELD_WRITE,description="Path of the file to write")
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
    find_me = ("write a new file", "create or overwrite a file")
    example_call = {"path": "/path/to/file", "content": "file contents"}

    async def execute(
        self,
        arguments: FileWriteToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        # The path the gate checked, opened through no link (security/safe_fs.py, GHSA-wgcv-xw9m-9q9q).
        try:
            path = safe_fs.resolve_tool_path(arguments.path, context.cwd, context.metadata)
            safe_fs.write_text(path, arguments.content, create_parents=arguments.create_directories)
        except safe_fs.PathChanged as exc:
            return ToolResult(output=str(exc), is_error=True)
        except FileNotFoundError:
            return ToolResult(output=f"Folder not found: {Path(arguments.path).parent} (create_directories is off)",
                              is_error=True)
        except OSError as exc:
            return ToolResult(output=f"Could not write {arguments.path}: {exc.strerror or exc}", is_error=True)
        return ToolResult(output=f"Wrote {path}")


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
