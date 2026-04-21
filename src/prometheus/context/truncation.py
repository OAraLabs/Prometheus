"""ToolResultTruncator — PostToolUseHook-compatible tool output truncation.

Sprint 4: trims oversized tool results before they consume context budget.
Different truncation strategies per tool type.

Usage (standalone):
    truncator = ToolResultTruncator(max_tokens=4000)
    trimmed = truncator.truncate("bash", long_output)

Usage (wired into agent_loop via post_tool hook — Sprint 5):
    The truncator exposes __call__(tool_name, output) -> str so it can be
    passed as a lightweight callable hook in a future HookDefinition wrapper.
"""

from __future__ import annotations

from prometheus.context.token_estimation import estimate_tokens

_DEFAULT_MAX_TOKENS = 4000


class ToolResultTruncator:
    """Truncate tool output that exceeds the configured token budget.

    Truncation strategies:
    - bash       : keep last 100 lines
    - read_file  : first 50 lines + last 50 lines with a gap marker
    - grep       : top 20 results
    - default    : hard-truncate with a token-count trailer
    """

    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        self._max_tokens = max_tokens

    @classmethod
    def from_config(cls, config_path: str | None = None) -> ToolResultTruncator:
        """Build from prometheus.yaml context.tool_result_max."""
        import yaml
        from pathlib import Path

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            max_tokens = data.get("context", {}).get("tool_result_max", _DEFAULT_MAX_TOKENS)
        except (OSError, Exception):
            max_tokens = _DEFAULT_MAX_TOKENS

        return cls(max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def truncate(self, tool_name: str, output: str) -> str:
        """Truncate *output* if it exceeds the token budget.

        Args:
            tool_name: Name of the tool that produced the output.
            output:    Raw tool output string.

        Returns:
            Possibly-truncated string.
        """
        if estimate_tokens(output) <= self._max_tokens:
            return output

        if tool_name == "bash":
            return self._truncate_bash(output)
        if tool_name == "read_file":
            return self._truncate_file_read(output)
        if tool_name == "grep":
            return self._truncate_grep(output)
        return self._truncate_default(output)

    def __call__(self, tool_name: str, output: str) -> str:
        """Allow the truncator to be used as a callable."""
        return self.truncate(tool_name, output)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _truncate_bash(self, output: str) -> str:
        """Keep the last 100 lines — bash output tail is most relevant."""
        lines = output.splitlines()
        kept = lines[-100:]
        dropped = len(lines) - len(kept)
        header = f"[... {dropped} lines truncated ...]\n" if dropped else ""
        return header + "\n".join(kept)

    def _truncate_file_read(self, output: str) -> str:
        """Keep first 50 + last 50 lines with a gap marker."""
        lines = output.splitlines()
        if len(lines) <= 100:
            return self._truncate_default(output)
        head = lines[:50]
        tail = lines[-50:]
        gap = len(lines) - 100
        return "\n".join(head) + f"\n[... {gap} lines truncated ...]\n" + "\n".join(tail)

    def _truncate_grep(self, output: str) -> str:
        """Keep top 20 grep results."""
        lines = [l for l in output.splitlines() if l.strip()]
        kept = lines[:20]
        dropped = len(lines) - len(kept)
        result = "\n".join(kept)
        if dropped:
            result += f"\n[... {dropped} more results truncated ...]"
        return result

    def _truncate_default(self, output: str) -> str:
        """Hard-truncate to approximately max_tokens, append a trailer."""
        char_limit = self._max_tokens * 4
        if len(output) <= char_limit:
            return output
        truncated = output[:char_limit]
        token_count = estimate_tokens(output)
        return truncated + f"\n[truncated at {token_count} tokens]"
