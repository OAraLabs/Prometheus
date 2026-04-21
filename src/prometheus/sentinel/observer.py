"""ActivityObserver — watches signals, detects patterns, sends nudges.

Source: Novel code for Prometheus Sprint 9.
Subscribes to all signals on the bus. When it detects interesting patterns
(extraction spikes, tool errors, long idle), it sends proactive nudges
via Telegram. Never auto-executes — always asks permission.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prometheus.sentinel.signals import ActivitySignal, SignalBus

if TYPE_CHECKING:
    from prometheus.gateway.platform_base import BasePlatformAdapter

log = logging.getLogger(__name__)


@dataclass
class PendingNudge:
    """A nudge queued for delivery."""

    nudge_type: str
    message: str
    timestamp: float = field(default_factory=time.time)


class ActivityObserver:
    """Watches activity signals and sends proactive nudges.

    Subscribes to ``"*"`` on the signal bus. Pattern detectors check
    for interesting or concerning trends. Nudges are sent via Telegram
    with a configurable cooldown per nudge type.
    """

    def __init__(
        self,
        bus: SignalBus,
        gateway: BasePlatformAdapter | None = None,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._bus = bus
        self._gateway = gateway

        cfg = config or {}
        self._nudge_cooldown = cfg.get("nudge_cooldown_minutes", 60) * 60
        self._extraction_spike_threshold = cfg.get("extraction_spike_threshold", 20)
        self._error_streak_threshold = cfg.get("error_streak_threshold", 5)
        self._chat_id = cfg.get("nudge_chat_id")

        self._last_nudge: dict[str, float] = {}
        self._pending_nudges: list[PendingNudge] = []
        self._last_activity: float = time.time()
        self._error_streak: int = 0
        self._started = False

    async def start(self) -> None:
        """Subscribe to all signals on the bus."""
        self._bus.subscribe("*", self._on_signal)
        self._started = True
        log.info("ActivityObserver: watching for patterns")

    async def _on_signal(self, signal: ActivitySignal) -> None:
        """Update tracking state and run pattern detectors."""
        self._last_activity = signal.timestamp

        if signal.kind == "extraction_complete":
            await self._check_extraction_insights(signal)
        elif signal.kind == "tool_executed":
            await self._check_tool_patterns(signal)
        elif signal.kind == "error_occurred":
            await self._check_error_patterns(signal)
        elif signal.kind == "dream_insight":
            # Forward dream insights as nudges
            digest = signal.payload.get("digest", "")
            if digest:
                await self._send_nudge(
                    "dream_insight",
                    f"\U0001f4ad AutoDream insight:\n{digest}",
                )

    async def _check_extraction_insights(self, signal: ActivitySignal) -> None:
        """After extraction, check if anything interesting was found."""
        count = signal.payload.get("count", 0)
        if count >= self._extraction_spike_threshold:
            await self._send_nudge(
                "extraction_spike",
                f"\U0001f4ca SENTINEL noticed:\n"
                f"Memory extraction found {count} new facts in one pass.\n"
                f"This is unusually high — might indicate a knowledge-dense "
                f"conversation.\n\nWant me to review the new entities?",
            )

    async def _check_tool_patterns(self, signal: ActivitySignal) -> None:
        """Detect concerning tool patterns."""
        success = signal.payload.get("success", True)
        tool_name = signal.payload.get("tool_name", "unknown")

        if not success:
            self._error_streak += 1
            if self._error_streak >= self._error_streak_threshold:
                await self._send_nudge(
                    "error_streak",
                    f"\u26a0\ufe0f SENTINEL noticed:\n"
                    f"Tool '{tool_name}' has failed {self._error_streak} "
                    f"times in a row.\n"
                    f"This might indicate a model regression or configuration "
                    f"issue.\n\nWant me to run diagnostics?",
                )
                self._error_streak = 0
        else:
            self._error_streak = 0

    async def _check_error_patterns(self, signal: ActivitySignal) -> None:
        """Check for error clusters."""
        error_type = signal.payload.get("error_type", "unknown")
        await self._send_nudge(
            "error_cluster",
            f"\U0001f6a8 SENTINEL noticed:\n"
            f"Error occurred: {error_type}\n"
            f"Detail: {signal.payload.get('detail', 'N/A')}\n\n"
            f"Want me to investigate?",
        )

    async def _send_nudge(self, nudge_type: str, message: str) -> None:
        """Send a proactive nudge via Telegram. Respects cooldown."""
        now = time.time()
        last = self._last_nudge.get(nudge_type, 0)

        if now - last < self._nudge_cooldown:
            self._pending_nudges.append(PendingNudge(
                nudge_type=nudge_type, message=message
            ))
            log.debug(
                "ActivityObserver: nudge '%s' queued (cooldown active)", nudge_type
            )
            return

        if self._gateway and self._chat_id:
            try:
                await self._gateway.send(self._chat_id, message)
                self._last_nudge[nudge_type] = now
                log.info("ActivityObserver: sent nudge '%s'", nudge_type)
            except Exception:
                log.exception("ActivityObserver: failed to send nudge")
                self._pending_nudges.append(PendingNudge(
                    nudge_type=nudge_type, message=message
                ))
        else:
            # No gateway — just log it
            log.info("ActivityObserver: nudge (no gateway): %s", message[:100])
            self._last_nudge[nudge_type] = now

    # ------------------------------------------------------------------
    # Status (used by SentinelStatusTool)
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_activity(self) -> float:
        return self._last_activity

    @property
    def pending_nudges(self) -> list[PendingNudge]:
        return list(self._pending_nudges)

    @property
    def nudge_history(self) -> dict[str, float]:
        return dict(self._last_nudge)
