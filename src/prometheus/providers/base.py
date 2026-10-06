"""Abstract model provider interface.

Replaces OpenHarness's SupportsStreamingMessages Protocol (which was coupled to
anthropic.AsyncAnthropic) with a proper ABC that any provider can implement.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from prometheus.engine.messages import ConversationMessage
from prometheus.engine.usage import UsageSnapshot


@dataclass(frozen=True)
class ApiMessageRequest:
    """Input parameters for a model invocation."""

    model: str
    messages: list[ConversationMessage]
    system_prompt: str | None = None
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)
    # Per-call override for the provider's thinking-suppression default.
    # ``None`` (the default) means "use provider's configured default" —
    # currently True for LlamaCppProvider. Set to ``False`` to opt a call
    # back INTO chain-of-thought (e.g. a task that wants reasoning_content).
    suppress_thinking: bool | None = None
    # Per-call no-tools turn (Sprint B / Piece 2 — chat mode): when True the model is offered
    # NO tools. run_loop empties the tool schema (prompt + payload) AND providers drop any
    # tool-calling grammar (the llama.cpp GBNF) so the turn is structurally tool-free at every
    # tier, not just tool-free-in-practice. Default False = today's behavior, untouched.
    suppress_tools: bool = False
    # Per-call tool_choice (force-search): "auto" | "none" | "required" | {"tool": "<name>"}.
    # Generalizes suppress_tools (none == suppress) into a four-state lever the provider
    # translates per tier — local GBNF grammar SELECTION, cloud native tool_choice. Default
    # "auto" = today's agent path. See prometheus.api.tool_choice.
    tool_choice: Any = "auto"


@dataclass(frozen=True)
class ApiTextDeltaEvent:
    """Incremental text produced by the model."""

    text: str


@dataclass(frozen=True)
class ApiMessageCompleteEvent:
    """Terminal event containing the full assistant message."""

    message: ConversationMessage
    usage: UsageSnapshot
    stop_reason: str | None = None
    # Count of tool-call entries the provider DROPPED because they arrived
    # structurally empty (no function name) — the malformed_empty guard at
    # the parse boundary. The agent loop uses this to give the model
    # structured feedback instead of silently ending the turn.
    dropped_malformed: int = 0
    # What the server said actually served this call, echoed in its own
    # response (`{"model": ...}` on both the buffered and streaming shapes).
    # Ground truth per call, as opposed to the name the caller requested at
    # construction time — which for six out-of-daemon harnesses is a config
    # string that stopped matching reality when the server was swapped.
    # None for providers that do not echo one; callers must tolerate that.
    served_model: str | None = None
    # Ids of the tool_use blocks the output limit cut off: the reply stopped
    # at max_tokens while that call's arguments were still streaming, so they
    # could not be read and the block carries ``{}`` in their place. The loop
    # never runs these; it answers each with an error that says the call was
    # cut (see :func:`cut_by_output_limit`).
    truncated_tool_calls: tuple[str, ...] = ()


ApiStreamEvent = ApiTextDeltaEvent | ApiMessageCompleteEvent

# What each wire calls a reply the output limit stopped: OpenAI-shape servers
# (the compatible clouds, llama.cpp, Ollama) say "length", Anthropic says
# "max_tokens".
OUTPUT_LIMIT_STOPS = frozenset({"length", "max_tokens"})


def cut_by_output_limit(stop_reason: str | None, raw_arguments: str | None) -> bool:
    """Did the output limit cut off the call whose arguments streamed as
    ``raw_arguments``?

    Ask it about a reply's LAST tool call only: calls stream in order, so no
    other can be in progress when the limit stops the reply. True when the
    reply stopped at the limit and the arguments do not read as a JSON object.
    Empty counts, because a finished call with no arguments still carries
    ``{}``. Unreadable arguments in a reply the model ended itself are a
    malformed call, not a cut one, and stay out of this.
    """
    if stop_reason not in OUTPUT_LIMIT_STOPS:
        return False
    try:
        return not isinstance(json.loads(raw_arguments or ""), dict)
    except json.JSONDecodeError:
        return True


class ModelProvider(ABC):
    """Abstract base class for all model providers.

    Concrete implementations: StubProvider (llama.cpp/OpenAI-compatible),
    OllamaProvider, etc.
    """

    supports_vision: bool = False
    api_enforced_structure: bool = False
    # The registry key that builds this provider ("anthropic", "ollama", ...).
    # The loop reads it to tell the router which provider just failed, so every
    # concrete class sets it; the OpenAI-compatible class, which serves many
    # keys, sets it per instance.
    provider_name: str = ""
    # What the served chat template does with tools — DETECTED by
    # ``detect_tool_template`` on the local providers (llama.cpp ``/props``,
    # ollama ``/api/show``), an ``adapter.tier.ToolTemplate``; None until a
    # probe ran. It decides the adapter tier for a model the registry does not
    # list (WP-X.28). Same posture as ``supports_vision``: recorded, not assumed.
    tool_template: Any = None

    async def detect_vision(self) -> bool:
        """Probe whether the provider supports vision. Override in subclasses."""
        return False

    @abstractmethod
    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a model response, yielding text deltas then a final complete event."""
