"""SignalBus — async pub/sub for activity signals.

Source: Novel code for Prometheus Sprint 9.
The connective tissue of SENTINEL. All components communicate via signals
emitted on a shared bus. Subscribers receive signals asynchronously.

SignalBus Persistence sprint additions:
  - ``emit`` now writes to ``telemetry.signal_events`` *synchronously,
    before* broadcasting. Consumers can trust that persisted == broadcast.
  - The in-memory ``deque(maxlen=500)`` stays as the hot cache for the
    existing ``recent(limit=N)`` callers (sentinel_status, Beacon's
    ``/api/sentinel``, Telegram /sentinel). No behaviour change for them.
  - ``history(signal_type=None, signal_types=None, since=None, limit=100)``
    reads from the DB — the durable tail.
  - ``recent(limit=N)`` backfills from DB only when the in-memory deque
    has fewer than N matching items, keeping the hot path zero-DB.
  - Persistence failures never block the broadcast: ``record_signal_event``
    catches and writes a ``silent_failures`` row, then ``emit`` continues.

The public ``emit(signal: ActivitySignal)`` signature is unchanged.
``ActivitySignal`` is unchanged. Existing emitters and subscribers work
without modification.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from prometheus.telemetry.tracker import ToolCallTelemetry

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


def _signal_to_iso(ts: float) -> str:
    """Convert ActivitySignal.timestamp (unix float) to ISO8601 UTC."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class SignalBus:
    """Simple async pub/sub for ActivitySignals.

    Subscribe to a specific signal kind, or use ``"*"`` to receive all signals.
    Subscriber exceptions are caught and logged — they never propagate.

    Persistence: when a ``ToolCallTelemetry`` handle is wired (either via
    constructor or by ``get_telemetry_handle()`` at emit time), every
    emission writes a row to ``signal_events`` synchronously before the
    broadcast. The in-memory deque continues to serve hot reads. See the
    module docstring for the full contract.
    """

    def __init__(
        self,
        *,
        history_limit: int = 500,
        telemetry: "ToolCallTelemetry | None" = None,
    ) -> None:
        self._subscribers: dict[str, list[SignalCallback]] = defaultdict(list)
        self._history: deque[ActivitySignal] = deque(maxlen=history_limit)
        self._telemetry = telemetry

    def subscribe(self, kind: str, callback: SignalCallback) -> None:
        """Register *callback* for signals of *kind* (or ``"*"`` for all)."""
        self._subscribers[kind].append(callback)

    def _resolve_telemetry(self) -> "ToolCallTelemetry | None":
        """Return the bound telemetry handle, or fall back to the singleton.

        The daemon may construct the bus before telemetry is wired (rare
        but possible during ordering refactors); the singleton fallback
        keeps the contract resilient. Mirrors the pattern from
        prometheus.tools.registration.try_register.
        """
        if self._telemetry is not None:
            return self._telemetry
        try:
            from prometheus.telemetry.tracker import get_telemetry_handle
            return get_telemetry_handle()
        except Exception:
            return None

    async def emit(self, signal: ActivitySignal) -> None:
        """Broadcast *signal* to matching subscribers and wildcards.

        Persistence runs BEFORE the broadcast. If it fails, the failure is
        captured in ``silent_failures`` (via record_signal_event) and the
        broadcast continues — persistence must never block the live stream.
        """
        # ── Persist first (cold tail / restart-durable) ──────────────
        tel = self._resolve_telemetry()
        if tel is not None:
            tel.record_signal_event(
                signal_type=signal.kind,
                payload=signal.payload,
                source_subsystem=signal.source,
                timestamp_iso=_signal_to_iso(signal.timestamp),
            )

        # ── Then hot cache + broadcast (existing behaviour) ──────────
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
        """Return recent signals, optionally filtered by *kind*.

        Hot path: serves from the in-memory deque. Backfills from the
        ``signal_events`` table only when the deque holds fewer than
        ``limit`` matching items — typical after a daemon restart, when
        the deque is empty but the DB still has history.
        """
        if kind is None:
            in_mem = list(self._history)[-limit:]
        else:
            in_mem = [s for s in self._history if s.kind == kind][-limit:]

        if len(in_mem) >= limit:
            return in_mem

        # Backfill from DB. If the bus has no telemetry, hot cache is the
        # only source — return what we have.
        tel = self._resolve_telemetry()
        if tel is None:
            return in_mem

        need = limit - len(in_mem)
        # Pull a comfortable buffer so de-duping against in_mem still
        # leaves us with `limit` rows.
        db_rows = tel.signal_events_since(
            signal_type=kind, limit=max(need * 2, need + 5),
        )
        # De-dupe key: (kind, source, ms-rounded-timestamp, payload JSON).
        # Why ms-rounding: ActivitySignal.timestamp is a float with sub-µs
        # precision, but the DB roundtrips it through ISO8601 which drops
        # the sub-µs tail. A strict (kind, ts_float) tuple would mis-match
        # the same event between in-mem and DB.
        import json as _json

        def _key(kind_: str, ts: float, payload: dict[str, Any], src: str) -> tuple:
            try:
                payload_json = _json.dumps(payload, sort_keys=True, default=str)
            except Exception:
                payload_json = ""
            return (kind_, src, round(ts * 1000), payload_json)

        in_mem_keys = {
            _key(s.kind, s.timestamp, s.payload, s.source) for s in in_mem
        }
        backfill: list[ActivitySignal] = []
        for row in db_rows:
            try:
                ts_iso = row["timestamp"]
                ts_float = datetime.fromisoformat(ts_iso).timestamp()
            except Exception:
                continue
            key = _key(row["signal_type"], ts_float, row["payload"], row["source_subsystem"])
            if key in in_mem_keys:
                continue
            backfill.append(ActivitySignal(
                kind=row["signal_type"],
                timestamp=ts_float,
                payload=row["payload"],
                source=row["source_subsystem"],
            ))
            if len(backfill) >= need:
                break
        # DB rows are newest-first; in_mem is oldest-first (deque order).
        # Return chronological: DB tail (older) … in_mem (newer).
        return list(reversed(backfill)) + in_mem

    def history(
        self,
        signal_type: str | None = None,
        *,
        signal_types: list[str] | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[ActivitySignal]:
        """Query the durable ``signal_events`` table directly.

        Returns ``ActivitySignal`` instances reconstructed from DB rows,
        newest-first. ``since`` is an ISO8601 lower bound on ``timestamp``.

        For the hot path (recent events likely in the in-memory deque),
        prefer :meth:`recent`; this method exists for explicit DB queries
        like ``/events recent`` and the Beacon hydration endpoint, where
        the caller wants the durable view by design.
        """
        tel = self._resolve_telemetry()
        if tel is None:
            return []
        rows = tel.signal_events_since(
            since=since,
            signal_type=signal_type,
            signal_types=signal_types,
            limit=limit,
        )
        out: list[ActivitySignal] = []
        for row in rows:
            try:
                ts_float = datetime.fromisoformat(row["timestamp"]).timestamp()
            except Exception:
                ts_float = 0.0
            out.append(ActivitySignal(
                kind=row["signal_type"],
                timestamp=ts_float,
                payload=row["payload"],
                source=row["source_subsystem"],
            ))
        return out

    @property
    def subscriber_count(self) -> int:
        """Total number of registered callbacks across all kinds."""
        return sum(len(cbs) for cbs in self._subscribers.values())

    @property
    def signal_count(self) -> int:
        """Number of signals in history."""
        return len(self._history)
