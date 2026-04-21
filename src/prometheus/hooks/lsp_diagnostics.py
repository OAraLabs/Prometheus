"""LSP diagnostics hook — auto-checks for type errors after file mutations.

PostToolUse hook that fires after ``write_file`` and ``edit_file``. Notifies
the LSP server of the change, waits briefly for diagnostics, and appends
any errors to the tool result so the model sees them in the same turn.

This is the novel integration — no other agent does this.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Tool names that trigger diagnostics checking
_MUTATION_TOOLS = {"write_file", "edit_file"}


class LSPDiagnosticsHook:
    """Async callable that appends LSP diagnostics to file-mutation results."""

    def __init__(
        self,
        orchestrator: object,
        delay_ms: int = 500,
        enabled: bool = True,
    ) -> None:
        self._orchestrator = orchestrator
        self._delay_s = delay_ms / 1000.0
        self._enabled = enabled

    async def __call__(self, tool_name: str, tool_input: dict, tool_result: object) -> object:
        """Check for diagnostics after file mutations.

        Returns the tool_result, possibly with appended diagnostic text.
        Designed to be called from agent_loop as a post-result hook.
        """
        if not self._enabled:
            return tool_result
        if tool_name not in _MUTATION_TOOLS:
            return tool_result
        if tool_result.is_error:
            return tool_result

        # Extract the file path from tool input
        filepath = tool_input.get("path")
        if not filepath:
            return tool_result

        try:
            filepath = str(Path(filepath).expanduser().resolve())
            if not Path(filepath).exists():
                return tool_result

            # Notify LSP of the change
            await self._orchestrator.notify_file_changed(filepath)

            # Wait for diagnostics to settle
            await asyncio.sleep(self._delay_s)

            # Fetch diagnostics
            diags = await self._orchestrator.get_diagnostics(filepath)
            errors = [d for d in diags if d.severity <= 2]  # ERROR + WARNING

            if not errors:
                return tool_result

            # Append diagnostics to the tool result
            diag_lines = [f"\n\u26a0\ufe0f LSP detected {len(errors)} issue(s):"]
            for d in errors[:10]:
                diag_lines.append(f"  {d}")
            if len(errors) > 10:
                diag_lines.append(f"  ... and {len(errors) - 10} more")

            # Create new ToolResultBlock with appended text
            from prometheus.engine.messages import ToolResultBlock
            return ToolResultBlock(
                tool_use_id=tool_result.tool_use_id,
                content=tool_result.content + "\n".join(diag_lines),
                is_error=tool_result.is_error,
            )
        except Exception:
            log.debug("LSP diagnostics hook failed", exc_info=True)
            return tool_result
