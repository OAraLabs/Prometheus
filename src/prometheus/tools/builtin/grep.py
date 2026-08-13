# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/tools/grep_tool.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""Content search tool with a pure-Python fallback."""

from __future__ import annotations

import re
from glob import has_magic
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class GrepToolInput(BaseModel):
    """Arguments for the grep tool."""

    pattern: str = Field(description="Regular expression to search for")
    root: str | None = Field(default=None, description="Search root directory")
    file_glob: str = Field(
        default="**/*",
        description=(
            "Which files to search, relative to root. Default searches every "
            "file recursively. Must be RELATIVE — an absolute pattern raises."
        ),
    )
    case_sensitive: bool = Field(
        default=True,
        description=(
            "Matching is case-SENSITIVE by default. To match any casing, set "
            "this to false — do NOT retry the search with different casings "
            "of the pattern. (A live turn burned 7 of its 25 tool calls "
            "permuting SYMBIOTE/symbiote/Symbiote/symbi before falling back "
            "to `bash grep -i`, and hit the iteration cap.)"
        ),
    )
    limit: int = Field(
        default=200, ge=1, le=2000,
        description="Maximum matching lines to return.",
    )


class GrepTool(BaseTool):
    """Search text files for a regex pattern."""

    name = "grep"
    description = "Search file contents with a regular expression."
    input_model = GrepToolInput
    example_call = {"pattern": "TODO", "root": "."}

    def is_read_only(self, arguments: GrepToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: GrepToolInput, context: ToolExecutionContext) -> ToolResult:
        root = _resolve_path(context.cwd, arguments.root) if arguments.root else context.cwd
        # Models frequently pass an ABSOLUTE file_glob (e.g. "/tmp/x/*.py");
        # pathlib raises NotImplementedError on non-relative patterns, which
        # killed the call (24 occurrences in one day of gym runs). Split an
        # absolute glob into (anchor dir, relative pattern) instead.
        root, file_glob = _split_absolute_glob(root, arguments.file_glob)
        flags = 0 if arguments.case_sensitive else re.IGNORECASE
        pattern = re.compile(arguments.pattern, flags)
        matches: list[str] = []

        for path in sorted(root.glob(file_glob)):
            if len(matches) >= arguments.limit:
                break
            if not path.is_file():
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    try:
                        shown = path.relative_to(root)
                    except ValueError:
                        shown = path
                    matches.append(f"{shown}:{line_no}:{line}")
                    if len(matches) >= arguments.limit:
                        break

        if not matches:
            return ToolResult(output="(no matches)")
        return ToolResult(output="\n".join(matches))


def _split_absolute_glob(root: Path, file_glob: str) -> tuple[Path, str]:
    """Turn an absolute glob into (anchor directory, relative pattern).

    "/tmp/x/src/*.py" → (Path("/tmp/x/src"), "*.py"); a concrete absolute
    path (no wildcards) anchors at its parent and globs its name. Relative
    patterns pass through untouched.
    """
    glob_path = Path(file_glob)
    if not glob_path.is_absolute():
        return root, file_glob
    parts = glob_path.parts
    anchor_end = len(parts)
    for i, part in enumerate(parts):
        if has_magic(part):
            anchor_end = i
            break
    anchor = Path(*parts[:anchor_end]) if anchor_end else Path(glob_path.anchor)
    rest = parts[anchor_end:]
    if not rest:
        # concrete path, no wildcard — glob for exactly its final component
        return anchor.parent, anchor.name
    return anchor, str(Path(*rest))


def _resolve_path(base: Path, candidate: str | None) -> Path:
    path = Path(candidate or ".").expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
