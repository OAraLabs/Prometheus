"""Heartbeat — periodic health check for Prometheus subsystems.

Source: Novel code for Prometheus Sprint 6.
Checks every 30 seconds for due cron jobs, pending tasks, and gateway health.
Sprint 9: Added idle detection for SENTINEL signal bus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from prometheus.gateway.cron_service import load_cron_jobs, validate_cron_expression

if TYPE_CHECKING:
    from prometheus.gateway.platform_base import BasePlatformAdapter
    from prometheus.sentinel.signals import SignalBus
    from prometheus.tasks.manager import BackgroundTaskManager

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 30  # seconds
DEFAULT_IDLE_THRESHOLD = 900  # 15 minutes


class Heartbeat:
    """Periodic subsystem health checker."""

    def __init__(
        self,
        *,
        interval: int = DEFAULT_INTERVAL,
        gateway: BasePlatformAdapter | None = None,
        task_manager: BackgroundTaskManager | None = None,
        signal_bus: SignalBus | None = None,
        idle_threshold: int = DEFAULT_IDLE_THRESHOLD,
    ) -> None:
        self.interval = interval
        self.gateway = gateway
        self.task_manager = task_manager
        self._running = False

        # SENTINEL idle detection (Sprint 9)
        self._signal_bus: SignalBus | None = signal_bus
        self._idle_threshold = idle_threshold
        self._last_activity: float = time.time()
        self._idle_emitted = False

    @property
    def signal_bus(self) -> SignalBus | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: SignalBus) -> None:
        """Set signal bus after construction (used by daemon.py wiring)."""
        self._signal_bus = bus
        # Subscribe to message_received to track activity
        bus.subscribe("message_received", self._on_activity)

    async def _on_activity(self, signal: Any) -> None:
        """Update last activity timestamp when a message is received."""
        self._last_activity = time.time()

    def record_activity(self) -> None:
        """Record that user activity occurred (called by gateway)."""
        self._last_activity = time.time()

    async def check(self) -> dict[str, Any]:
        """Run one health check cycle. Returns status dict."""
        now = datetime.now(timezone.utc)
        status: dict[str, Any] = {"timestamp": now.isoformat()}

        # Cron: count due jobs
        jobs = load_cron_jobs()
        due_count = 0
        for job in jobs:
            if not job.get("enabled", True):
                continue
            schedule = job.get("schedule", "")
            if not validate_cron_expression(schedule):
                continue
            next_run_str = job.get("next_run")
            if not next_run_str:
                continue
            try:
                next_run = datetime.fromisoformat(next_run_str)
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)
                if next_run <= now:
                    due_count += 1
            except (ValueError, TypeError):
                continue
        status["cron_jobs_due"] = due_count
        status["cron_jobs_total"] = len(jobs)

        # Gateway health
        if self.gateway:
            status["gateway_running"] = self.gateway.running
            status["gateway_platform"] = self.gateway.platform.value
        else:
            status["gateway_running"] = None

        # Pending tasks
        if self.task_manager:
            pending = self.task_manager.list_tasks(status="running")
            status["tasks_running"] = len(pending)
        else:
            status["tasks_running"] = None

        return status

    async def run_forever(self) -> None:
        """Run the heartbeat loop until cancelled."""
        self._running = True
        logger.info("Heartbeat started (interval=%ds)", self.interval)
        try:
            while self._running:
                try:
                    status = await self.check()
                    logger.debug("Heartbeat: %s", status)

                    # Log warnings for noteworthy states
                    if status.get("gateway_running") is False:
                        logger.warning("Heartbeat: gateway is not running")
                    if status.get("cron_jobs_due", 0) > 0:
                        logger.info(
                            "Heartbeat: %d cron job(s) due",
                            status["cron_jobs_due"],
                        )

                    # SENTINEL idle detection (Sprint 9)
                    await self._check_idle()

                except Exception as exc:
                    logger.error("Heartbeat check failed: %s", exc)

                await asyncio.sleep(self.interval)
        finally:
            self._running = False
            logger.info("Heartbeat stopped")

    async def _check_idle(self) -> None:
        """Emit idle_start / idle_end signals on the bus."""
        if self._signal_bus is None:
            return

        from prometheus.sentinel.signals import ActivitySignal

        idle_seconds = time.time() - self._last_activity

        if idle_seconds > self._idle_threshold and not self._idle_emitted:
            self._idle_emitted = True
            await self._signal_bus.emit(ActivitySignal(
                kind="idle_start",
                payload={"idle_since": self._last_activity},
                source="heartbeat",
            ))
            logger.info("Heartbeat: idle detected (%.0fs)", idle_seconds)

        elif idle_seconds <= self._idle_threshold and self._idle_emitted:
            self._idle_emitted = False
            await self._signal_bus.emit(ActivitySignal(
                kind="idle_end",
                payload={},
                source="heartbeat",
            ))
            logger.info("Heartbeat: activity resumed")

    def stop(self) -> None:
        """Signal the heartbeat to stop on the next iteration."""
        self._running = False
