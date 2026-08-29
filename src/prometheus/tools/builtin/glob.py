# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/glob_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""Filesystem globbing tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import DIR_FIELD, PATH_FIELD
from prometheus.tools.denied_prune import (
    is_denied,
    resolve_denied,
    withheld_note,
)
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult
# The one owner of absolute-glob splitting (#134) — imported rather than
# copied, so the two sibling tools cannot drift apart again.
from prometheus.tools.builtin.grep import _split_absolute_glob


class GlobToolInput(BaseModel):
    """Arguments for the glob tool."""

    pattern: str = Field(description="Glob pattern relative to the working directory")
    root: str | None = Field(
        json_schema_extra=DIR_FIELD,default=None, description="Optional search root")
    limit: int = Field(
        default=200, ge=1, le=5000,
        description="Maximum paths to return.",
    )


class GlobTool(BaseTool):
    """List files matching a glob pattern."""

    name = "glob"
    description = "List files matching a glob pattern."
    input_model = GlobToolInput
    example_call = {"pattern": "**/*.py"}

    def __init__(self, denied_paths=None) -> None:
        self._denied = resolve_denied(denied_paths)

    def is_read_only(self, arguments: GlobToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: GlobToolInput, context: ToolExecutionContext) -> ToolResult:
        root = _resolve_path(context.cwd, arguments.root) if arguments.root else context.cwd
        # Same defect grep fixed in #134, unfixed in this sibling until the
        # 2026-08-18 handoff sweep: models frequently pass an ABSOLUTE
        # pattern ("/tmp/x/*.py"), and pathlib raises NotImplementedError on
        # non-relative patterns — killing the call and teaching the model to
        # re-encode. Split it into (anchor dir, relative pattern) with the
        # one helper that already owns this logic.
        root, pattern = _split_absolute_glob(root, arguments.pattern)
        found = list(root.glob(pattern))
        allowed = [p for p in found if not is_denied(p, self._denied)]
        withheld = len(found) - len(allowed)
        matches = sorted(str(p.relative_to(root)) for p in allowed)
        note = withheld_note(withheld)
        if not matches:
            return ToolResult(output="(no matches)" + note)
        return ToolResult(output="\n".join(matches[: arguments.limit]) + note)


def _resolve_path(base: Path, candidate: str | None) -> Path:
    path = Path(candidate or ".").expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
