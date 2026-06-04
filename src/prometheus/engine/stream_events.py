# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/engine/stream_events.py
# License: MIT
# Modified: renamed imports (openharness → prometheus)

"""Events yielded by the query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prometheus.engine.usage import UsageSnapshot
from prometheus.engine.messages import ConversationMessage


@dataclass(frozen=True)
class AssistantTextDelta:
    """Incremental assistant text."""

    text: str


@dataclass(frozen=True)
class AssistantTurnComplete:
    """Completed assistant turn."""

    message: ConversationMessage
    usage: UsageSnapshot


@dataclass(frozen=True)
class ToolExecutionStarted:
    """The engine is about to execute a tool."""

    tool_name: str
    tool_input: dict[str, Any]
    # Stable id correlating this start to its ToolExecutionCompleted — the
    # ToolUseBlock id (``toolu_<hex>``) the model assigned. Defaulted so older
    # construction sites keep working.
    tool_use_id: str = ""


@dataclass(frozen=True)
class ToolExecutionCompleted:
    """A tool has finished executing."""

    tool_name: str
    output: str
    is_error: bool = False
    # Same id as the matching ToolExecutionStarted (see above).
    tool_use_id: str = ""


StreamEvent = (
    AssistantTextDelta
    | AssistantTurnComplete
    | ToolExecutionStarted
    | ToolExecutionCompleted
)
