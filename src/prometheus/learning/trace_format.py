"""How a task's tool trace is shown to the learning loop's models.

SkillCreator and SkillRefiner both render the trace as one line per call::

    3. bash({"command": "docker compose up -d"}) → started

Until option C3 each call rendered as ``bash({})``: the formatters read an
``arguments`` key that ``AgentLoop.run_async`` never sets, so a skill was
written from tool names and result snippets alone. The input now comes from
``tool_input`` (set since C1), with ``arguments`` still read for callers
that build their own trace in that older shape.

An input is redacted before it is cut to :data:`INPUT_CHARS`: cutting first
can leave the head of a token that is too short to match a token pattern.
Results are redacted and cut the same way, to :data:`RESULT_CHARS`.
"""

from __future__ import annotations

import json
from typing import Any

from prometheus.security.log_redaction import redact_capture, redact_secrets

# Per call. A 50-call trace (SkillCreator's ceiling) adds about 3K tokens.
INPUT_CHARS = 240
RESULT_CHARS = 200


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_input(call: dict[str, Any]) -> str:
    """The call's input, redacted, as compact JSON (or as given, if a string), cut."""
    raw = call.get("tool_input")
    if raw is None:
        raw = call.get("arguments") or {}
    if isinstance(raw, str):
        text = redact_secrets(raw)
    else:
        text = json.dumps(redact_capture(raw), ensure_ascii=False, default=str)
    return _cut(text, INPUT_CHARS)


def format_trace(trace: list[dict[str, Any]], *, mark_errors: bool = False) -> str:
    """One ``n. tool(input) → result`` line per call; ``[ERROR]`` on failed calls if asked."""
    lines: list[str] = []
    for i, call in enumerate(trace, 1):
        tool = call.get("tool_name", "unknown")
        result = redact_secrets(str(call.get("result", "")))[:RESULT_CHARS]
        flag = " [ERROR]" if mark_errors and call.get("is_error") else ""
        lines.append(f"{i}. {tool}({format_input(call)}) → {result}{flag}")
    return "\n".join(lines)
