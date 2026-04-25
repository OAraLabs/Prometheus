"""GEPAEngine — SENTINEL idle-task wrapper around GEPAOptimizer.

Source: Novel code for SUNRISE Sprint Session B.
Mirrors AutoDreamEngine's bus-subscription + spawn-task pattern. Subscribes
to ``idle_start`` and ``idle_end`` on the SignalBus; when continuous idle
exceeds ``gepa_min_idle_minutes`` and the rate-limit window allows, runs
one full GEPA optimization cycle.

There is no ``register_idle_task`` API on SENTINEL — this engine is the
canonical pattern for new idle-driven workloads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from prometheus.sentinel.signals import ActivitySignal, SignalBus

if TYPE_CHECKING:
    from prometheus.learning.gepa import GEPAOptimizer, GEPAReport

log = logging.getLogger(__name__)


class GEPAEngine:
    """Idle-triggered GEPA optimization cycles.

    Args:
        optimizer: GEPAOptimizer instance.
        signal_bus: SignalBus to subscribe on.
        config: ``learning`` section dict. Recognised keys:
            ``gepa_min_idle_minutes`` (int, default 10)
            ``gepa_max_frequency_hours`` (float, default 24)
    """

    def __init__(
        self,
        optimizer: GEPAOptimizer,
        signal_bus: SignalBus,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._opt = optimizer
        self._bus = signal_bus
        cfg = config or {}
        self._min_idle_s = max(0, int(cfg.get("gepa_min_idle_minutes", 10))) * 60
        self._max_freq_s = max(60, int(
            float(cfg.get("gepa_max_frequency_hours", 24)) * 3600
        ))
        self._last_run: float = 0.0
        self._idle_since: float | None = None
        self._running = False
        self._last_report: "GEPAReport | None" = None
        self._pending_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Subscribe to idle signals."""
        self._bus.subscribe("idle_start", self._on_idle_start)
        self._bus.subscribe("idle_end", self._on_idle_end)
        log.info(
            "GEPAEngine: subscribed (min_idle=%ds, max_freq=%ds)",
            self._min_idle_s, self._max_freq_s,
        )

    async def _on_idle_start(self, signal: ActivitySignal) -> None:
        """Begin tracking idle. Spawn the run-after-wait task."""
        self._idle_since = time.time()
        if self._pending_task is not None and not self._pending_task.done():
            return  # already waiting
        self._pending_task = asyncio.create_task(
            self._run_after_idle(),
            name="gepa_idle_wait",
        )

    async def _on_idle_end(self, signal: ActivitySignal) -> None:
        """User activity resumed — cancel any pending idle wait."""
        self._idle_since = None
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None

    async def _run_after_idle(self) -> None:
        """Wait min_idle_seconds; if still idle and rate-limit allows, run a cycle."""
        try:
            await asyncio.sleep(self._min_idle_s)
        except asyncio.CancelledError:
            return
        if self._idle_since is None:
            return  # idle ended during wait
        now = time.time()
        if (now - self._last_run) < self._max_freq_s:
            log.debug(
                "GEPAEngine: rate-limited (last=%.0fs ago, min=%ds)",
                now - self._last_run, self._max_freq_s,
            )
            return
        if self._running:
            return
        await self._run_cycle()

    async def run_now(self) -> "GEPAReport | None":
        """Manually trigger a cycle (e.g. from /gepa run).

        Bypasses the idle gate but still respects ``_running`` so two cycles
        can't overlap.
        """
        if self._running:
            log.info("GEPAEngine: cycle already running, ignoring run_now")
            return None
        return await self._run_cycle()

    async def _run_cycle(self) -> "GEPAReport | None":
        """Execute one optimization cycle and emit a completion signal."""
        self._running = True
        try:
            report = await self._opt.run_optimization_cycle()
            self._last_run = time.time()
            self._last_report = report
            try:
                await self._bus.emit(ActivitySignal(
                    kind="gepa_cycle_complete",
                    payload={
                        "summary": report.to_telegram_summary(),
                        "skills_evaluated": report.skills_evaluated,
                        "skills_promoted": report.skills_promoted,
                    },
                    source="gepa_engine",
                ))
            except Exception:
                log.debug("GEPAEngine: signal emit failed", exc_info=True)
            return report
        except Exception:
            log.exception("GEPAEngine: cycle failed")
            return None
        finally:
            self._running = False

    @property
    def last_report(self) -> "GEPAReport | None":
        return self._last_report

    @property
    def running(self) -> bool:
        return self._running
