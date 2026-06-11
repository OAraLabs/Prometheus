"""Deterministic scoring over a finished run's transcript. No LLM judging.

The transcript is the loop's final ``messages`` list plus run metadata the
runner extracts (drops, breaker state). Every predicate present on the task
must pass (AND).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


@dataclass
class ToolEvent:
    name: str
    input: dict[str, Any]
    is_error: bool | None      # None = no result block found (shouldn't happen)
    result_content: str = ""


@dataclass
class RunTranscript:
    messages: list[ConversationMessage]
    dropped_malformed: int = 0          # provider-dropped empty envelopes
    orchestrator_feedback: int = 0      # loop's malformed-retry injections
    tool_events: list[ToolEvent] = field(default_factory=list)
    final_text: str = ""
    breaker_tripped: bool = False

    @classmethod
    def from_messages(cls, messages: list[ConversationMessage]) -> "RunTranscript":
        events: list[ToolEvent] = []
        by_id: dict[str, ToolEvent] = {}
        feedback = 0
        final_text = ""
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        ev = ToolEvent(name=block.name, input=dict(block.input), is_error=None)
                        events.append(ev)
                        by_id[block.id] = ev
                    elif isinstance(block, TextBlock) and block.text.strip():
                        final_text = block.text
            else:
                if msg.provenance == "orchestrator":
                    feedback += 1
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.tool_use_id in by_id:
                        by_id[block.tool_use_id].is_error = block.is_error
                        by_id[block.tool_use_id].result_content = block.content

        breaker = "circuit breaker tripped" in final_text.lower()
        return cls(
            messages=messages,
            orchestrator_feedback=feedback,
            tool_events=events,
            final_text=final_text,
            breaker_tripped=breaker,
        )


def _successful_calls(t: RunTranscript, tool: str) -> list[ToolEvent]:
    return [e for e in t.tool_events if e.name == tool and e.is_error is False]


def _looks_like_json_blob(value: Any) -> bool:
    if not isinstance(value, str):
        return True  # not even a string — worse than a blob
    s = value.strip()
    if not s.startswith("{"):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return s.startswith('{"')  # truncated serialization still counts


def score(task_score: dict[str, Any], t: RunTranscript, workspace: Path) -> tuple[bool, list[str]]:
    """Apply every predicate; return (passed, failure_reasons)."""
    reasons: list[str] = []
    expect_tool = task_score.get("expect_tool")
    ok_calls = _successful_calls(t, expect_tool) if expect_tool else []

    if expect_tool:
        if not ok_calls:
            attempted = [e.name for e in t.tool_events]
            reasons.append(
                f"no successful {expect_tool!r} call (tools attempted: {attempted or 'none'})"
            )

    if "expect_tool_args_string" in task_score and ok_calls:
        for param in task_score["expect_tool_args_string"]:
            for call in ok_calls:
                if param in call.input and not isinstance(call.input[param], str):
                    reasons.append(
                        f"{expect_tool}.{param} should be a string, got "
                        f"{type(call.input[param]).__name__}: {call.input[param]!r:.120}"
                    )

    if "expect_tool_args_require" in task_score:
        required = task_score["expect_tool_args_require"]
        if ok_calls and not any(
            all(c.input.get(k) == v for k, v in required.items()) for c in ok_calls
        ):
            reasons.append(
                f"no {expect_tool} call matched required args {required!r} "
                f"(got: {[c.input for c in ok_calls]!r:.300})"
            )

    if task_score.get("prompt_not_json_blob") and ok_calls:
        for call in ok_calls:
            if "prompt" in call.input and _looks_like_json_blob(call.input["prompt"]):
                reasons.append(
                    f"{expect_tool}.prompt is a serialized JSON blob, not plain text: "
                    f"{str(call.input['prompt'])[:120]!r}"
                )

    if "expect_file" in task_score:
        p = Path(task_score["expect_file"])
        if not p.is_absolute():
            p = workspace / p
        if not p.exists():
            reasons.append(f"expected file {p} to exist")
        elif "expect_file_contains" in task_score:
            content = p.read_text(errors="replace")
            if task_score["expect_file_contains"] not in content:
                reasons.append(
                    f"file {p} lacks {task_score['expect_file_contains']!r} "
                    f"(got: {content[:120]!r})"
                )

    if "expect_text_contains" in task_score:
        want = task_score["expect_text_contains"].lower()
        if want not in t.final_text.lower():
            reasons.append(
                f"final text lacks {task_score['expect_text_contains']!r} "
                f"(got: {t.final_text[:160]!r})"
            )

    if "expect_text_min_len" in task_score:
        if len(t.final_text.strip()) < int(task_score["expect_text_min_len"]):
            reasons.append(
                f"final text too short ({len(t.final_text.strip())} chars "
                f"< {task_score['expect_text_min_len']})"
            )

    if "forbid_bash_containing" in task_score:
        needle = task_score["forbid_bash_containing"]
        for e in t.tool_events:
            if e.name == "bash" and needle in str(e.input.get("command", "")):
                reasons.append(f"bash command contained forbidden {needle!r}")

    if task_score.get("forbid_breaker_trip") and t.breaker_tripped:
        reasons.append("circuit breaker tripped")

    if task_score.get("forbid_malformed") and t.dropped_malformed > 0:
        reasons.append(f"{t.dropped_malformed} malformed-empty call(s) dropped")

    if task_score.get("require_graceful"):
        if t.breaker_tripped:
            reasons.append("not graceful: circuit breaker tripped")
        elif not t.final_text.strip():
            reasons.append("not graceful: empty final text")

    return (not reasons, reasons)
