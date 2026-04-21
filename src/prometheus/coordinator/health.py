# Source: Novel code for Prometheus Sprint 8.
# Extends Sprint 6's Heartbeat with infrastructure-level health checks.

"""HealthMonitor — periodic infrastructure health checks for Prometheus."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 60  # seconds


class HealthState(str, Enum):
    """Overall health state."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Health of a single component."""

    name: str
    healthy: bool
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthStatus:
    """Aggregate health report."""

    state: HealthState
    components: list[ComponentHealth] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def degraded_components(self) -> list[ComponentHealth]:
        return [c for c in self.components if not c.healthy]

    def summary(self) -> str:
        lines = [f"Health: {self.state.value}"]
        for c in self.components:
            status = "OK" if c.healthy else "FAIL"
            lines.append(f"  {c.name}: {status} — {c.detail}")
        return "\n".join(lines)


# Individual check functions

def check_llama_cpp(base_url: str = "http://127.0.0.1:8080") -> ComponentHealth:
    """Check if llama.cpp server is responding."""
    import httpx

    t0 = time.monotonic()
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5.0)
        latency = (time.monotonic() - t0) * 1000
        ok = resp.status_code == 200
        return ComponentHealth(
            name="llama.cpp",
            healthy=ok,
            detail=f"status={resp.status_code}" if not ok else "responding",
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return ComponentHealth(
            name="llama.cpp",
            healthy=False,
            detail=str(exc),
            latency_ms=latency,
        )


def check_sqlite(db_path: str | None = None) -> ComponentHealth:
    """Check SQLite database accessibility."""
    import sqlite3

    path = db_path or os.environ.get("PROMETHEUS_DB", "prometheus.db")
    t0 = time.monotonic()
    try:
        conn = sqlite3.connect(path, timeout=5)
        conn.execute("SELECT 1")
        conn.close()
        latency = (time.monotonic() - t0) * 1000
        return ComponentHealth(
            name="sqlite",
            healthy=True,
            detail=f"path={path}",
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return ComponentHealth(
            name="sqlite",
            healthy=False,
            detail=str(exc),
            latency_ms=latency,
        )


def check_tailscale() -> ComponentHealth:
    """Check Tailscale connectivity status."""
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        latency = (time.monotonic() - t0) * 1000
        ok = result.returncode == 0
        return ComponentHealth(
            name="tailscale",
            healthy=ok,
            detail="connected" if ok else f"exit={result.returncode}",
            latency_ms=latency,
        )
    except FileNotFoundError:
        return ComponentHealth(
            name="tailscale",
            healthy=True,
            detail="not installed (skipped)",
            latency_ms=0,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return ComponentHealth(
            name="tailscale",
            healthy=False,
            detail=str(exc),
            latency_ms=latency,
        )


def check_gpu_memory() -> ComponentHealth:
    """Check GPU memory via nvidia-smi."""
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        latency = (time.monotonic() - t0) * 1000
        if result.returncode != 0:
            return ComponentHealth(
                name="gpu_memory",
                healthy=False,
                detail=f"nvidia-smi exit={result.returncode}",
                latency_ms=latency,
            )
        line = result.stdout.strip().split("\n")[0]
        used_str, total_str = line.split(",")
        used = int(used_str.strip())
        total = int(total_str.strip())
        pct = (used / total * 100) if total > 0 else 0
        return ComponentHealth(
            name="gpu_memory",
            healthy=pct < 95,
            detail=f"{used}/{total} MiB ({pct:.0f}%)",
            latency_ms=latency,
        )
    except FileNotFoundError:
        return ComponentHealth(
            name="gpu_memory",
            healthy=True,
            detail="nvidia-smi not found (skipped)",
            latency_ms=0,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return ComponentHealth(
            name="gpu_memory",
            healthy=False,
            detail=str(exc),
            latency_ms=latency,
        )


def check_disk(path: str = "/") -> ComponentHealth:
    """Check disk space on the given mount."""
    t0 = time.monotonic()
    try:
        usage = shutil.disk_usage(path)
        latency = (time.monotonic() - t0) * 1000
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        pct_used = (usage.used / usage.total * 100) if usage.total > 0 else 0
        return ComponentHealth(
            name="disk",
            healthy=free_gb > 1.0,
            detail=f"{free_gb:.1f}/{total_gb:.1f} GB free ({pct_used:.0f}% used)",
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return ComponentHealth(
            name="disk",
            healthy=False,
            detail=str(exc),
            latency_ms=latency,
        )


# Alert callback type
AlertCallback = Callable[[HealthStatus], Awaitable[None]]


class HealthMonitor:
    """Periodic infrastructure health monitor.

    Checks: llama.cpp, SQLite, Tailscale, GPU memory, disk space.
    Reports via callback (e.g., Telegram) when degraded.
    """

    def __init__(
        self,
        *,
        interval: int = DEFAULT_INTERVAL,
        llama_url: str = "http://127.0.0.1:8080",
        db_path: str | None = None,
        disk_path: str = "/",
        alert_callback: AlertCallback | None = None,
        checks: list[str] | None = None,
    ) -> None:
        self.interval = interval
        self.llama_url = llama_url
        self.db_path = db_path
        self.disk_path = disk_path
        self.alert_callback = alert_callback
        self._enabled_checks = checks or [
            "llama_cpp",
            "sqlite",
            "tailscale",
            "gpu_memory",
            "disk",
        ]
        self._running = False
        self._last_state: HealthState | None = None

    def _run_checks(self) -> list[ComponentHealth]:
        """Run all enabled checks synchronously."""
        results: list[ComponentHealth] = []
        dispatch = {
            "llama_cpp": lambda: check_llama_cpp(self.llama_url),
            "sqlite": lambda: check_sqlite(self.db_path),
            "tailscale": check_tailscale,
            "gpu_memory": check_gpu_memory,
            "disk": lambda: check_disk(self.disk_path),
        }
        for name in self._enabled_checks:
            fn = dispatch.get(name)
            if fn is not None:
                results.append(fn())
        return results

    async def check(self) -> HealthStatus:
        """Run one health check cycle."""
        loop = asyncio.get_event_loop()
        components = await loop.run_in_executor(None, self._run_checks)
        degraded = [c for c in components if not c.healthy]

        if not degraded:
            state = HealthState.HEALTHY
        elif len(degraded) >= 3:
            state = HealthState.CRITICAL
        else:
            state = HealthState.DEGRADED

        return HealthStatus(state=state, components=components)

    async def run_forever(self) -> None:
        """Run the health monitor loop until cancelled."""
        self._running = True
        logger.info("HealthMonitor started (interval=%ds)", self.interval)
        try:
            while self._running:
                try:
                    status = await self.check()
                    logger.debug("Health: %s", status.summary())

                    # Alert on state transitions to degraded/critical
                    if (
                        status.state != HealthState.HEALTHY
                        and status.state != self._last_state
                        and self.alert_callback is not None
                    ):
                        await self.alert_callback(status)

                    self._last_state = status.state
                except Exception as exc:
                    logger.error("HealthMonitor check failed: %s", exc)

                await asyncio.sleep(self.interval)
        finally:
            self._running = False
            logger.info("HealthMonitor stopped")

    def stop(self) -> None:
        """Signal the monitor to stop."""
        self._running = False
