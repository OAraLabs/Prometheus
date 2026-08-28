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
class ProviderDegraded:
    """A terminal provider failure was served by the fallback instead (SPRINT-provider-fallback).

    Yielded so NON-CHAT clients learn the fact too. The chat surface already sees the degrade as
    text — the wrapper emits a notice delta and it lands in stored history — but a client that
    renders tool activity or status without reading reply prose would otherwise show a normal
    answer from a model nobody chose.

    `requested_model` is what the caller asked for and `served_model` is what actually answered;
    they are separate fields on purpose. Collapsing them is what made "why did my model change?"
    unanswerable and is the reason fallback was kept out of the router's decision path.
    """

    requested_model: str
    served_model: str
    provider_name: str
    reason: str


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
