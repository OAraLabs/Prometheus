"""Coding-mode editor + execution tools (SPRINT-coding-mode v2, scope item 2).

Native Tool classes with strict pydantic schemas — deliberately NO extra
client-side GBNF layer (bakeoff Q2: the server-side --jinja grammar already
drives malformed calls to zero; strict schemas suffice). Semantics mirror the
str_replace-editor convention (view-before-edit, exactly-once match) because
the local model has seen that convention in training data — familiarity is
reliability.

Every tool is constructed AROUND a :class:`~prometheus.coding.sandbox.Sandbox`
and registered into a per-coding-session registry by
:func:`build_coding_registry` — these are session-scoped tools, never part of
the global builtin registry (a coding run's jail is meaningless outside it).

The load-bearing semantic for iterate-to-green: ``code_run`` returns a
FAILING test run as ``is_error=False`` with the exit code in the output —
command-ran-and-reported is tool success. ``is_error=True`` is reserved for
timeout / sandbox violation / spawn failure. This keeps repeated red test
runs from feeding the agent loop's circuit breaker (3-identical/5-any would
otherwise kill the edit→test→re-edit cycle the whole sprint exists to build).
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.coding.sandbox import Sandbox, SandboxViolation
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult

log = logging.getLogger(__name__)

_VIEW_MAX_LINES = 2_000
_GREP_MAX_HITS = 200
_GLOB_MAX_PATHS = 500
_GREP_FILE_SIZE_CAP = 1_000_000  # bytes — skip blobs
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".tox", ".eggs"}


def _numbered(text: str, start: int = 1) -> str:
    return "\n".join(
        f"{i:6d}\t{line}" for i, line in enumerate(text.splitlines(), start)
    )


def _violation(exc: SandboxViolation) -> ToolResult:
    return ToolResult(output=f"SANDBOX VIOLATION: {exc}", is_error=True)


def _iter_files(root: Path):
    """Walk *root* skipping VCS/cache dirs, yielding files only."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file():
                yield entry


# --------------------------------------------------------------------------- #
# code_view
# --------------------------------------------------------------------------- #


class CodeViewArgs(BaseModel):
    path: str = Field(description="File or directory path, relative to the repo root.")
    view_range: list[int] | None = Field(
        default=None,
        description="Optional [start_line, end_line] (1-based, inclusive) for files.",
    )


class CodeViewTool(BaseTool):
    name = "code_view"
    description = (
        "View a file with line numbers, or list a directory. Always view a "
        "file before editing it — code_str_replace requires the exact "
        "current text. Optional view_range narrows to [start, end] lines."
    )
    input_model = CodeViewArgs
    example_call = {"path": "src/app.py", "view_range": [1, 60]}

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(self, arguments: CodeViewArgs, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            real = self._sandbox.resolve(arguments.path)
        except SandboxViolation as exc:
            return _violation(exc)

        if real.is_dir():
            entries = []
            for entry in sorted(real.iterdir()):
                if entry.name in _SKIP_DIRS:
                    continue
                entries.append(entry.name + ("/" if entry.is_dir() else ""))
            return ToolResult(
                output=f"{arguments.path or '.'}/ ({len(entries)} entries)\n"
                + "\n".join(entries)
            )
        if not real.is_file():
            return ToolResult(
                output=f"NOT FOUND: {arguments.path!r} does not exist. "
                f"Use code_glob or code_view on a directory to locate files.",
                is_error=True,
            )

        try:
            text = real.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(output=f"READ FAILED: {exc}", is_error=True)

        lines = text.splitlines()
        start = 1
        if arguments.view_range:
            if len(arguments.view_range) != 2:
                return ToolResult(
                    output="BAD RANGE: view_range must be [start_line, end_line].",
                    is_error=True,
                )
            lo, hi = arguments.view_range
            lo = max(1, lo)
            hi = min(len(lines), hi) if hi != -1 else len(lines)
            lines = lines[lo - 1 : hi]
            start = lo

        truncated = ""
        if len(lines) > _VIEW_MAX_LINES:
            lines = lines[:_VIEW_MAX_LINES]
            truncated = (
                f"\n… [view truncated at {_VIEW_MAX_LINES} lines — "
                f"use view_range to see more]"
            )
        return ToolResult(
            output=_numbered("\n".join(lines), start) + truncated
        )


# --------------------------------------------------------------------------- #
# code_str_replace
# --------------------------------------------------------------------------- #


class CodeStrReplaceArgs(BaseModel):
    path: str = Field(description="File to edit, relative to the repo root.")
    old_str: str = Field(
        description="Exact text to replace — must appear EXACTLY ONCE in the file, "
        "including whitespace and indentation."
    )
    new_str: str = Field(description="Replacement text.")


class CodeStrReplaceTool(BaseTool):
    name = "code_str_replace"
    description = (
        "Replace text in a file. old_str must match exactly once (whitespace "
        "included). If it matches zero or multiple times nothing is changed "
        "and the error tells you which — re-view the file and widen old_str "
        "with surrounding lines until it is unique."
    )
    input_model = CodeStrReplaceArgs
    example_call = {
        "path": "src/app.py",
        "old_str": "def total(x):\n    return x",
        "new_str": "def total(x):\n    return x + 1",
    }

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, arguments: CodeStrReplaceArgs, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            real = self._sandbox.resolve(arguments.path)
        except SandboxViolation as exc:
            return _violation(exc)
        if not real.is_file():
            return ToolResult(
                output=f"NOT FOUND: {arguments.path!r} — view the directory first.",
                is_error=True,
            )
        if not arguments.old_str:
            return ToolResult(
                output="EMPTY old_str: to create content use code_create; to "
                "edit, view the file and copy the exact text to replace.",
                is_error=True,
            )

        text = real.read_text(encoding="utf-8", errors="replace")
        count = text.count(arguments.old_str)
        if count == 0:
            return ToolResult(
                output=(
                    f"NO MATCH in {arguments.path}: old_str was not found. "
                    f"Nothing was changed. View the file (code_view) and copy "
                    f"the exact current text — whitespace and indentation must "
                    f"match exactly."
                ),
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                output=(
                    f"{count} MATCHES in {arguments.path}: old_str is ambiguous. "
                    f"Nothing was changed. Re-view the region and include more "
                    f"surrounding lines in old_str until it is unique."
                ),
                is_error=True,
            )

        new_text = text.replace(arguments.old_str, arguments.new_str, 1)
        real.write_text(new_text, encoding="utf-8")

        # Show the edited region so the model can verify without re-viewing.
        offset = new_text.index(arguments.new_str) if arguments.new_str else 0
        line_no = new_text[:offset].count("\n") + 1
        snippet_lines = new_text.splitlines()[
            max(0, line_no - 3) : line_no + arguments.new_str.count("\n") + 3
        ]
        return ToolResult(
            output=(
                f"OK — replaced 1 occurrence in {arguments.path} "
                f"(now at line {line_no}):\n"
                + _numbered("\n".join(snippet_lines), max(1, line_no - 2))
            )
        )


# --------------------------------------------------------------------------- #
# code_create
# --------------------------------------------------------------------------- #


class CodeCreateArgs(BaseModel):
    path: str = Field(description="New file path, relative to the repo root.")
    content: str = Field(description="Full file content.")


class CodeCreateTool(BaseTool):
    name = "code_create"
    description = (
        "Create a new file with the given content. Fails if the file already "
        "exists — use code_str_replace to modify existing files."
    )
    input_model = CodeCreateArgs
    example_call = {"path": "tests/test_new.py", "content": "def test_ok():\n    assert True\n"}

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def execute(self, arguments: CodeCreateArgs, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            real = self._sandbox.resolve(arguments.path)
        except SandboxViolation as exc:
            return _violation(exc)
        if real.exists():
            return ToolResult(
                output=(
                    f"ALREADY EXISTS: {arguments.path!r}. Nothing was written. "
                    f"Use code_view + code_str_replace to modify it."
                ),
                is_error=True,
            )
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text(arguments.content, encoding="utf-8")
        return ToolResult(
            output=f"OK — created {arguments.path} "
            f"({len(arguments.content)} chars, "
            f"{arguments.content.count(chr(10)) + 1} lines)."
        )


# --------------------------------------------------------------------------- #
# code_grep
# --------------------------------------------------------------------------- #


class CodeGrepArgs(BaseModel):
    pattern: str = Field(description="Python regular expression to search for.")
    path_glob: str = Field(
        default="**/*",
        description="Glob filter for files to search (e.g. 'src/**/*.py').",
    )


class CodeGrepTool(BaseTool):
    name = "code_grep"
    description = (
        "Search file contents with a regex across the repo (skips .git, "
        "caches, binaries). Returns path:line: text matches."
    )
    input_model = CodeGrepArgs
    example_call = {"pattern": r"def total\(", "path_glob": "src/**/*.py"}

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(self, arguments: CodeGrepArgs, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            rx = re.compile(arguments.pattern)
        except re.error as exc:
            return ToolResult(
                output=f"BAD REGEX: {exc}. Pattern is a Python regular expression.",
                is_error=True,
            )

        root = self._sandbox.root
        hits: list[str] = []
        for f in _iter_files(root):
            rel = f.relative_to(root).as_posix()
            if not fnmatch.fnmatch(rel, arguments.path_glob):
                continue
            try:
                if f.stat().st_size > _GREP_FILE_SIZE_CAP:
                    continue
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()[:300]}")
                    if len(hits) >= _GREP_MAX_HITS:
                        return ToolResult(
                            output="\n".join(hits)
                            + f"\n… [stopped at {_GREP_MAX_HITS} matches — narrow the pattern or glob]"
                        )
        if not hits:
            return ToolResult(
                output=f"NO MATCHES for /{arguments.pattern}/ in {arguments.path_glob}"
            )
        return ToolResult(output="\n".join(hits))


# --------------------------------------------------------------------------- #
# code_glob
# --------------------------------------------------------------------------- #


class CodeGlobArgs(BaseModel):
    pattern: str = Field(description="Glob pattern relative to the repo root, e.g. 'src/**/*.py'.")


class CodeGlobTool(BaseTool):
    name = "code_glob"
    description = "List repo files matching a glob pattern (sorted; skips .git and caches)."
    input_model = CodeGlobArgs
    example_call = {"pattern": "tests/**/*.py"}

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(self, arguments: CodeGlobArgs, context: ToolExecutionContext) -> ToolResult:
        del context
        root = self._sandbox.root
        matches = sorted(
            f.relative_to(root).as_posix()
            for f in _iter_files(root)
            if fnmatch.fnmatch(f.relative_to(root).as_posix(), arguments.pattern)
        )
        if not matches:
            return ToolResult(output=f"NO FILES match {arguments.pattern!r}")
        truncated = ""
        if len(matches) > _GLOB_MAX_PATHS:
            truncated = f"\n… [{len(matches) - _GLOB_MAX_PATHS} more — narrow the pattern]"
            matches = matches[:_GLOB_MAX_PATHS]
        return ToolResult(output="\n".join(matches) + truncated)


# --------------------------------------------------------------------------- #
# code_run
# --------------------------------------------------------------------------- #


class CodeRunArgs(BaseModel):
    command: str = Field(
        description="Shell command to run in the repo root (tests, builds, git)."
    )
    timeout_seconds: float | None = Field(
        default=None,
        description="Optional wall-clock cap for this command (default: sandbox default).",
    )


class CodeRunTool(BaseTool):
    name = "code_run"
    description = (
        "Run a shell command in the repo root (sandboxed). Use this to run "
        "the test suite and the task's acceptance command — a coding task is "
        "only done when its acceptance command exits 0 here. The exit code "
        "is reported in the output; a failing command is a normal result to "
        "read and act on, not a tool error."
    )
    input_model = CodeRunArgs
    example_call = {"command": "python -m pytest tests/ -x -q"}
    # Test runs legitimately outlast the loop's 300 s default; the sandbox
    # enforces the real per-command wall inside this allowance.
    execution_timeout_seconds = 360.0

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return False  # truthfully: it can mutate anything inside the jail

    async def execute(self, arguments: CodeRunArgs, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            result = await self._sandbox.run(
                arguments.command, timeout_seconds=arguments.timeout_seconds
            )
        except SandboxViolation as exc:
            return _violation(exc)
        except OSError as exc:
            return ToolResult(output=f"SPAWN FAILED: {exc}", is_error=True)

        if result.timed_out:
            return ToolResult(
                output=(
                    f"TIMED OUT after {result.duration_seconds:.0f}s — the "
                    f"process tree was killed. Partial output:\n{result.output}"
                ),
                is_error=True,
                metadata={"timed_out": True},
            )
        # Failing commands are REPORTED, not errored — iterate-to-green reads
        # the exit code from here, and the circuit breaker stays out of it.
        return ToolResult(
            output=(
                f"exit code: {result.exit_code} "
                f"(in {result.duration_seconds:.1f}s)\n{result.output}"
            ),
            metadata={"exit_code": result.exit_code},
        )


# --------------------------------------------------------------------------- #
# Registry factory — session-scoped, never global
# --------------------------------------------------------------------------- #

CODING_TOOL_CLASSES = (
    CodeViewTool,
    CodeStrReplaceTool,
    CodeCreateTool,
    CodeGrepTool,
    CodeGlobTool,
    CodeRunTool,
)


def build_coding_registry(sandbox: Sandbox) -> ToolRegistry:
    """Build the per-coding-session registry around one sandbox.

    Deliberately NOT registered into the global builtin registry: these
    tools are meaningless outside their jail, and a coding run gets a
    fresh registry per task (the orphan-tool audit looks for tools
    produced-but-unregistered in the GLOBAL registry; session-scoped
    construction here is by design, asserted by the registration test).
    """
    registry = ToolRegistry()
    for cls in CODING_TOOL_CLASSES:
        registry.register(cls(sandbox))
    return registry
