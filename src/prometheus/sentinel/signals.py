"""SignalBus — async pub/sub for activity signals.

Source: Novel code for Prometheus Sprint 9.
The connective tissue of SENTINEL. All components communicate via signals
emitted on a shared bus. Subscribers receive signals asynchronously.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)


@dataclass
class ActivitySignal:
    """A single activity event flowing through the SENTINEL bus."""

    kind: str  # "idle_start", "idle_end", "extraction_complete", etc.
    timestamp: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = ""


# Type alias for signal callbacks
SignalCallback = Callable[[ActivitySignal], Awaitable[None]]


class SignalBus:
    """Simple async pub/sub for ActivitySignals.

    Subscribe to a specific signal kind, or use ``"*"`` to receive all signals.
    Subscriber exceptions are caught and logged — they never propagate.
    """

    def __init__(self, *, history_limit: int = 500) -> None:
        self._subscribers: dict[str, list[SignalCallback]] = defaultdict(list)
        self._history: deque[ActivitySignal] = deque(maxlen=history_limit)

    def subscribe(self, kind: str, callback: SignalCallback) -> None:
        """Register *callback* for signals of *kind* (or ``"*"`` for all)."""
        self._subscribers[kind].append(callback)

    async def emit(self, signal: ActivitySignal) -> None:
        """Broadcast *signal* to matching subscribers and wildcards."""
        self._history.append(signal)

        targets = list(self._subscribers.get(signal.kind, []))
        targets.extend(self._subscribers.get("*", []))

        for cb in targets:
            try:
                await cb(signal)
            except Exception:
                log.exception(
                    "SignalBus: subscriber %s failed on %s", cb, signal.kind
                )

    def recent(
        self, kind: str | None = None, *, limit: int = 50
    ) -> list[ActivitySignal]:
        """Return recent signals, optionally filtered by *kind*."""
        if kind is None:
            return list(self._history)[-limit:]
        return [s for s in self._history if s.kind == kind][-limit:]

    @property
    def subscriber_count(self) -> int:
        """Total number of registered callbacks across all kinds."""
        return sum(len(cbs) for cbs in self._subscribers.values())

    @property
    def signal_count(self) -> int:
        """Number of signals in history."""
        return len(self._history)
