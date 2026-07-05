"""Base platform adapter — ABC, message event, send result.

Source: Novel code for Prometheus Sprint 6 (architecture inspired by Hermes gateway).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from prometheus.gateway.config import Platform, PlatformConfig

logger = logging.getLogger(__name__)


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
        # SPRINT G1 — gateway-generic subsystem slots. daemon.py attaches
        # each subsystem to EVERY registered adapter through the
        # GatewaySubsystemRegistry below (never to one adapter by name), so
        # any gateway — Telegram, Slack, or a future Discord — inherits the
        # full set for free. All slots default to None ("not wired"); the
        # shared command layer treats None as "subsystem not active".
        self.cost_tracker: Any = None            # CostTracker (cloud spend)
        self.escalation_engine: Any = None       # TeacherEscalation
        self._approval_queue: Any = None         # ApprovalQueue (LEVEL 1 gates)
        self._gepa_engine: Any = None            # GEPAEngine (/gepa)
        self._printing_press: Any = None         # PrintingPressRegistry (/press)
        self._backup_vault: Any = None           # BackupVault (/symbiote backup*)
        self._morph_engine: Any = None           # MorphEngine (/symbiote morph/swap)

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


class GatewaySubsystemRegistry:
    """Broadcast daemon subsystems to every gateway adapter (SPRINT G1).

    daemon.py constructs adapters and subsystems at different points during
    startup (the approval queue exists before Slack does; GEPA long after
    both). This registry makes ordering irrelevant:

      * ``register_adapter(adapter)`` — adds a gateway and immediately
        replays every subsystem attached so far onto it.
      * ``attach(name, value)`` — records a subsystem and sets it on every
        adapter registered so far (and on all future ones).

    Attachment is a plain ``setattr``, so adapters that expose a property
    (e.g. ``signal_bus`` with its subscribe-on-set side effect) get their
    setter invoked, and adapters that only define the base-class slot get a
    plain attribute. A failing setter is logged loudly but never blocks the
    other adapters or daemon startup.

    Design goal: adding a gateway (Discord — Sprint G2) means constructing
    the adapter and calling ``register_adapter`` once; it inherits ALL
    subsystems with no per-subsystem wiring.
    """

    def __init__(self) -> None:
        self._adapters: list[BasePlatformAdapter] = []
        self._subsystems: dict[str, Any] = {}

    @property
    def adapters(self) -> list[BasePlatformAdapter]:
        """The registered adapters (live list copy)."""
        return list(self._adapters)

    @property
    def subsystems(self) -> dict[str, Any]:
        """The attached subsystems by slot name (copy)."""
        return dict(self._subsystems)

    def register_adapter(self, adapter: BasePlatformAdapter | None) -> None:
        """Add *adapter* and replay every already-attached subsystem onto it."""
        if adapter is None or adapter in self._adapters:
            return
        self._adapters.append(adapter)
        for name, value in self._subsystems.items():
            self._set(adapter, name, value)

    def attach(self, name: str, value: Any) -> None:
        """Attach subsystem *value* under slot *name* on all adapters.

        Also recorded for adapters registered later.
        """
        self._subsystems[name] = value
        for adapter in self._adapters:
            self._set(adapter, name, value)

    @staticmethod
    def _set(adapter: BasePlatformAdapter, name: str, value: Any) -> None:
        try:
            setattr(adapter, name, value)
        except Exception:
            logger.warning(
                "failed to attach subsystem %r to %s",
                name, type(adapter).__name__, exc_info=True,
            )
