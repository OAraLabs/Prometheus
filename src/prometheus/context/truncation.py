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

    Truncation strategies (``_STRATEGIES`` — name-keyed; the names are pinned
    to the real registry by tests/test_truncation_contract.py, because a tool
    rename would otherwise silently demote it to the default strategy):
    - bash       : keep last 100 lines
    - read_file  : first 50 lines + last 50 lines with a gap marker
    - grep       : top 20 results
    - default    : head + tail window + a trailer that is true at this layer

    THE NOTICE CONTRACT (selector survey, 2026-08-11): whatever this class
    emits becomes part of the tool result the model reasons over. A notice
    must therefore be TRUE (state what was kept and what was received — never
    imply the payload it was handed is the whole artifact) and ACTIONABLE
    (name a recovery that actually works; the full output is NOT retained
    anywhere, so the only honest advice is re-running the tool narrower).
    tests/test_truncation_contract.py holds every strategy to that contract.
    """

    # Tail window the DEFAULT strategy preserves. Tools put their own state at
    # the END of a result — wiki_query's budget marker, web_fetch's cut
    # notice, vault_read's continue offset — and a head-only cut beheaded all
    # of them: the model lost the content AND the notice saying how to get it.
    _DEFAULT_TAIL_CHARS = 400

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

        strategy = self._STRATEGIES.get(tool_name)
        if strategy is not None:
            return strategy(self, output)
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
        """Head + tail window + a trailer that is true at this layer.

        Replaces a head-only cut whose trailer said "[truncated at N tokens]"
        with N = the size of the payload it was HANDED. Both halves were dead
        wrong in live use (2026-08-11 vault audit): the head-only cut beheaded
        every tool's own tail-positioned notice, and the trailer read as the
        artifact's size — a 72k-char page whose upstream cap had already cut
        it to 48k was reported as "12041 tokens". This layer knows exactly two
        true things, what it received and what it kept; the trailer states
        those, says the remainder was NOT retained, and names the one recovery
        that actually works.
        """
        char_limit = self._max_tokens * 4
        if len(output) <= char_limit:
            return output
        tail_keep = min(self._DEFAULT_TAIL_CHARS, char_limit // 4)
        head_keep = char_limit - tail_keep
        dropped = len(output) - head_keep - tail_keep
        return (
            output[:head_keep]
            + f"\n[... {dropped} chars omitted ...]\n"
            + output[-tail_keep:]
            + f"\n[truncated by the per-result budget: kept the first "
            f"{head_keep} and last {tail_keep} chars of the {len(output)} "
            f"received — the rest was not retained; re-run the tool with "
            f"narrower arguments if more is needed]"
        )

    # Name-keyed strategy table. Introspectable on purpose:
    # tests/test_truncation_contract.py asserts every key names a tool in the
    # registry the daemon actually builds, so a rename cannot silently demote
    # a tool to the default strategy.
    _STRATEGIES = {
        "bash": _truncate_bash,
        "read_file": _truncate_file_read,
        "grep": _truncate_grep,
    }
