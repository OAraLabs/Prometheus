"""Abstract model provider interface.

Replaces OpenHarness's SupportsStreamingMessages Protocol (which was coupled to
anthropic.AsyncAnthropic) with a proper ABC that any provider can implement.
"""

from __future__ import annotations

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


ApiStreamEvent = ApiTextDeltaEvent | ApiMessageCompleteEvent


class ModelProvider(ABC):
    """Abstract base class for all model providers.

    Concrete implementations: StubProvider (llama.cpp/OpenAI-compatible),
    OllamaProvider, etc.
    """

    supports_vision: bool = False
    api_enforced_structure: bool = False

    async def detect_vision(self) -> bool:
        """Probe whether the provider supports vision. Override in subclasses."""
        return False

    @abstractmethod
    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        """Stream a model response, yielding text deltas then a final complete event."""
