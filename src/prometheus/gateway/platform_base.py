"""Base platform adapter — ABC, message event, send result.

Source: Novel code for Prometheus Sprint 6 (architecture inspired by Hermes gateway).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from prometheus.gateway.config import Platform, PlatformConfig


class MessageType(str, Enum):
    """Types of incoming messages."""

    TEXT = "text"
    COMMAND = "command"
    CALLBACK = "callback"
    EDITED = "edited"
    PHOTO = "photo"
    DOCUMENT = "document"
    VOICE = "voice"
    STICKER = "sticker"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MessageEvent:
    """Normalised incoming message from any platform."""

    chat_id: int
    user_id: int
    text: str
    message_id: int
    platform: Platform
    message_type: MessageType = MessageType.TEXT
    username: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict[str, Any] = field(default_factory=dict)
    # Sprint 15 GRAFT: media fields (additive — Hermes parity)
    media_urls: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    caption: str | None = None

    def session_key(self) -> str:
        """Return a unique key for this chat's session."""
        return f"{self.platform.value}:{self.chat_id}"


@dataclass(frozen=True)
class SendResult:
    """Result of sending a message back to the platform."""

    success: bool
    message_id: int | None = None
    error: str | None = None


class BasePlatformAdapter(ABC):
    """Abstract base for all platform adapters."""

    def __init__(self, config: PlatformConfig) -> None:
        self.config = config
        self._running = False

    @property
    def platform(self) -> Platform:
        return self.config.platform

    @property
    def running(self) -> bool:
        return self._running

    @abstractmethod
    async def start(self) -> None:
        """Start the adapter (polling, webhook, etc.)."""

    @abstractmethod
    async def stop(self) -> None:
        """Graceful shutdown."""

    @abstractmethod
    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> SendResult:
        """Send a message to a chat."""

    @abstractmethod
    async def on_message(self, event: MessageEvent) -> None:
        """Handle an incoming message event."""
