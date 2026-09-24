"""LSP tool — compiler-grade code intelligence for the agent.

Single tool with multiple actions. The ``context`` action is the most important
one — it packages definition + references + type info into one call instead of
three separate round trips.

Modeled after Claude Code's ``LSPTool`` pattern.

Every answer is either what the server said or an error saying why it could
not be asked. "none found" / "No diagnostics" only ever come from a server
that answered; no server, a failed start, or a failed request is an error.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.lsp.orchestrator import LSPUnavailable
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

# Module-level orchestrator — set by daemon at startup
_orchestrator = None

# How long the diagnostics action waits for the server to publish for the
# file's current content. A first open of a large project can take seconds.
_DIAGNOSTICS_WAIT_S = 15.0


def set_lsp_orchestrator(orch: object) -> None:
    """Wire the LSP orchestrator at daemon startup."""
    global _orchestrator
    _orchestrator = orch


class LSPToolInput(BaseModel):
    """Arguments for the LSP tool."""

    action: str = Field(
        description=(
            "LSP action: definition, references, hover, diagnostics, "
            "symbols, rename, or context (recommended — returns definition + "
            "references + type info in one call)"
        ),
    )
    file: str = Field(description="Path to the source file")
    line: int | None = Field(
        default=None,
        description="Line number (1-indexed). Not needed for diagnostics/symbols.",
    )
    column: int | None = Field(
        default=None,
        description="Column number (1-indexed). Defaults to first occurrence if symbol is given.",
    )
    symbol: str | None = Field(
        default=None,
        description="Symbol name to look up. If given without line/column, the file is searched for it.",
    )
    new_name: str | None = Field(
        default=None,
        description="New name for the symbol (rename action only). Rename returns the server's proposed edits; it does not write files.",
    )


class LSPTool(BaseTool):
    """Compiler-grade code intelligence via Language Server Protocol."""

    name = "lsp"
    description = (
        "Code intelligence: get definitions, references, type info, diagnostics, "
        "and document symbols via LSP. Use the 'context' action to get definition + "
        "references + type info in a single call (recommended)."
    )
    input_model = LSPToolInput

    def is_read_only(self, arguments: LSPToolInput) -> bool:
        return arguments.action != "rename"

    async def execute(
        self,
        arguments: LSPToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        orch = _orchestrator or context.metadata.get("lsp_orchestrator")
        if orch is None:
            return ToolResult(
                output="LSP is not available. Language server not configured.",
                is_error=True,
            )

        filepath = _resolve_path(context.cwd, arguments.file)
        if not Path(filepath).exists():
            return ToolResult(output=f"File not found: {filepath}", is_error=True)

        action = arguments.action.lower()

        # Resolve symbol name to line/column if needed
        line = arguments.line
        col = arguments.column or 1
        if line is None and arguments.symbol:
            try:
                resolved = _find_symbol_in_file(filepath, arguments.symbol)
            except OSError as exc:
                return ToolResult(output=f"Couldn't check: could not read {filepath}: {exc}", is_error=True)
            if resolved is None:
                return ToolResult(
                    output=f"Symbol '{arguments.symbol}' not found in {filepath}",
                    is_error=True,
                )
            line, col = resolved

        try:
            if action == "definition":
                if line is None:
                    return ToolResult(output="Line number or symbol name required for definition", is_error=True)
                locations = await orch.get_definition(filepath, line, col)
                return ToolResult(output=_format_locations("Definition", locations))

            elif action == "references":
                if line is None:
                    return ToolResult(output="Line number or symbol name required for references", is_error=True)
                locations = await orch.get_references(filepath, line, col)
                return ToolResult(output=_format_locations("References", locations))

            elif action == "hover":
                if line is None:
                    return ToolResult(output="Line number or symbol name required for hover", is_error=True)
                info = await orch.get_hover(filepath, line, col)
                if info is None:
                    return ToolResult(output="No hover information available.")
                return ToolResult(output=str(info))

            elif action == "diagnostics":
                diags = await orch.get_diagnostics(filepath, wait_s=_DIAGNOSTICS_WAIT_S)
                return ToolResult(output=_format_diagnostics(filepath, diags))

            elif action == "symbols":
                symbols = await orch.get_symbols(filepath)
                return ToolResult(output=_format_symbols(symbols))

            elif action == "rename":
                if line is None:
                    return ToolResult(output="Line number or symbol name required for rename", is_error=True)
                if not arguments.new_name:
                    return ToolResult(output="new_name is required for rename action", is_error=True)
                edits = await orch.rename(filepath, line, col, arguments.new_name)
                return ToolResult(output=_format_rename(arguments.new_name, edits))

            elif action == "context":
                if line is None:
                    return ToolResult(output="Line number or symbol name required for context", is_error=True)
                text = await orch.get_symbol_context(filepath, line, col)
                return ToolResult(output=text)

            else:
                return ToolResult(
                    output=f"Unknown action: {action}. Use: definition, references, hover, diagnostics, symbols, rename, or context",
                    is_error=True,
                )
        except LSPUnavailable as exc:
            return ToolResult(output=str(exc), is_error=True)
        except Exception as exc:
            return ToolResult(output=f"LSP error: {exc}", is_error=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_path(base: Path, candidate: str) -> str:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _find_symbol_in_file(filepath: str, symbol: str) -> tuple[int, int] | None:
    """Search a file for a symbol name, return (line, col) or None.

    A file that cannot be read raises OSError: "not in the file" and "could
    not read the file" are different answers.
    """
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
    for i, line in enumerate(text.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            return i, m.start() + 1
    return None


def _format_locations(label: str, locations: list) -> str:
    if not locations:
        return f"{label}: none found"
    lines = [f"{label} ({len(locations)}):"]
    for loc in locations:
        lines.append(f"  {loc}")
    return "\n".join(lines)


def _format_diagnostics(filepath: str, diags: list) -> str:
    if not diags:
        return f"No diagnostics for {Path(filepath).name}"
    lines = [f"Diagnostics for {Path(filepath).name} ({len(diags)}):"]
    for d in diags:
        lines.append(f"  {d}")
    return "\n".join(lines)


def _format_symbols(symbols: list, indent: int = 0) -> str:
    if not symbols:
        return "No symbols found"
    lines: list[str] = []
    for s in symbols:
        lines.append(f"{'  ' * indent}{s}")
        if s.children:
            lines.append(_format_symbols(s.children, indent + 1))
    return "\n".join(lines)


def _format_rename(new_name: str, edits: dict) -> str:
    # Nothing here writes files: the server returns a workspace edit and this
    # reports it. It used to say "Renamed to", which read as done.
    if not edits:
        return f"Rename to '{new_name}': the server proposed no changes (the symbol may not support rename)"
    total = sum(len(v) for v in edits.values())
    lines = [
        f"Rename to '{new_name}' NOT APPLIED. The server proposes {total} edit(s) "
        f"across {len(edits)} file(s); nothing was written. Apply them with edit_file:",
    ]
    for path, file_edits in edits.items():
        lines.append(f"  {path}: {len(file_edits)} edit(s)")
        for e in file_edits:
            lines.append(f"    line {e.get('start_line')}: -> {e.get('newText', '')!r}")
    return "\n".join(lines)
