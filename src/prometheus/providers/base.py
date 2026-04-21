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
