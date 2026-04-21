"""RetryEngine — manages validation-failure retries for tool calls.

When a tool call fails validation and cannot be auto-repaired, the retry
engine builds a targeted error prompt that tells the model exactly what
went wrong and how to fix it. It tracks per-tool retry counts to prevent
infinite loops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetryAction(str, Enum):
    RETRY = "RETRY"       # Send retry prompt to model
    ABORT = "ABORT"       # Max retries exceeded — give up
    ESCALATE = "ESCALATE" # Max retries exceeded — escalate to stronger model via router


@dataclass
class RetryState:
    """Per-tool retry state within a single session."""
    count: int = 0
    last_error: str = ""


class RetryEngine:
    """Build retry prompts and track per-tool retry counts.

    Usage:
        engine = RetryEngine(max_retries=3)
        action, prompt = engine.handle_failure(
            tool_name="bash",
            error="missing required param: command",
            tool_registry=registry,
            session_key="bash",
        )
        if action == RetryAction.RETRY:
            # inject `prompt` as a user message and continue the loop
    """

    def __init__(self, max_retries: int = 3, router: Any = None) -> None:
        self.max_retries = max_retries
        self.router = router
        self._state: dict[str, RetryState] = {}

    def handle_failure(
        self,
        tool_name: str,
        error: str,
        tool_registry: Any,
        session_key: str | None = None,
    ) -> tuple[RetryAction, str]:
        """Handle a tool-call validation failure.

        Args:
            tool_name:   The name of the failing tool.
            error:       Human-readable error description.
            tool_registry: Registry to look up the tool schema.
            session_key: Key for tracking retries; defaults to tool_name.

        Returns:
            (RetryAction.RETRY, retry_prompt) or (RetryAction.ABORT, reason)
        """
        key = session_key or tool_name
        state = self._state.setdefault(key, RetryState())
        state.count += 1
        state.last_error = error

        if state.count > self.max_retries:
            # Check if router supports escalation to a stronger model
            if self.router is not None:
                escalation_enabled = getattr(
                    getattr(self.router, "config", None),
                    "escalation_enabled",
                    False,
                )
                if escalation_enabled:
                    return (
                        RetryAction.ESCALATE,
                        f"Escalating {tool_name!r} to stronger model after "
                        f"{self.max_retries} retries. Last error: {error}",
                    )
            return (
                RetryAction.ABORT,
                f"Giving up on {tool_name!r} after {self.max_retries} retries. Last error: {error}",
            )

        prompt = self.build_retry_prompt(tool_name, error, tool_registry)
        return RetryAction.RETRY, prompt

    def build_retry_prompt(
        self,
        tool_name: str,
        error: str,
        tool_registry: Any,
    ) -> str:
        """Build a targeted retry prompt for the model.

        The prompt shows the error, the expected schema, and asks the model
        to try again with a corrected call.
        """
        schema_str = ""
        if tool_registry is not None:
            tool = tool_registry.get(tool_name)
            if tool is not None:
                try:
                    schema = tool.input_model.model_json_schema()
                    schema_str = json.dumps(schema, indent=2)
                except Exception:
                    pass

        lines = [
            f"Your tool call for '{tool_name}' failed with this error:",
            f"  {error}",
        ]
        if schema_str:
            lines += [
                f"\nThe tool '{tool_name}' expects these parameters:",
                schema_str,
            ]
        lines += [
            "\nPlease try again with a corrected tool call.",
        ]
        return "\n".join(lines)

    def retry_count(self, session_key: str) -> int:
        """Return the number of retries for the given session key."""
        return self._state.get(session_key, RetryState()).count

    def reset(self, session_key: str | None = None) -> None:
        """Reset retry state. Resets all keys if session_key is None."""
        if session_key is None:
            self._state.clear()
        else:
            self._state.pop(session_key, None)
