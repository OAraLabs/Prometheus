"""SentinelStatusTool — agent can check what SENTINEL has been doing.

Source: Novel code for Prometheus Sprint 9.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

if TYPE_CHECKING:
    from prometheus.sentinel.autodream import AutoDreamEngine
    from prometheus.sentinel.observer import ActivityObserver
    from prometheus.sentinel.signals import SignalBus

# Module-level singletons (set by daemon.py at startup)
_signal_bus: SignalBus | None = None
_observer: ActivityObserver | None = None
_autodream: AutoDreamEngine | None = None


def set_sentinel_components(
    bus: SignalBus,
    observer: ActivityObserver,
    autodream: AutoDreamEngine,
) -> None:
    """Register SENTINEL components so the tool can access them."""
    global _signal_bus, _observer, _autodream  # noqa: PLW0603
    _signal_bus = bus
    _observer = observer
    _autodream = autodream


class SentinelStatusInput(BaseModel):
    """Arguments for sentinel_status."""

    verbose: bool = Field(
        default=False,
        description="Include detailed signal history and dream phase results.",
    )


class SentinelStatusTool(BaseTool):
    """Check what the SENTINEL proactive subsystem has been doing."""

    name = "sentinel_status"
    description = (
        "Returns the current state of the SENTINEL proactive subsystem: "
        "observer state, dream engine state, recent signals, pending nudges, "
        "and last dream cycle results."
    )
    input_model = SentinelStatusInput

    def is_read_only(self, arguments: SentinelStatusInput) -> bool:
        return True

    async def execute(
        self, arguments: SentinelStatusInput, context: ToolExecutionContext
    ) -> ToolResult:
        if _signal_bus is None or _observer is None or _autodream is None:
            return ToolResult(
                output="SENTINEL not initialised. Is the daemon running with sentinel enabled?",
                is_error=True,
            )

        lines: list[str] = ["# SENTINEL Status\n"]

        # Observer
        idle_secs = int(time.time() - _observer.last_activity)
        lines.append("## Observer")
        lines.append(f"- Active: {_observer.started}")
        lines.append(f"- Last activity: {idle_secs}s ago")
        lines.append(f"- Pending nudges: {len(_observer.pending_nudges)}")

        # AutoDream
        lines.append("\n## AutoDream Engine")
        lines.append(f"- Dreaming: {_autodream.dreaming}")
        lines.append(f"- Dream cycles completed: {_autodream.cycle_count}")
        if _autodream.last_cycle_time:
            ago = int(time.time() - _autodream.last_cycle_time)
            lines.append(f"- Last cycle: {ago}s ago")

        # Signal bus
        lines.append("\n## Signal Bus")
        lines.append(f"- Total signals: {_signal_bus.signal_count}")
        lines.append(f"- Subscribers: {_signal_bus.subscriber_count}")

        if arguments.verbose:
            # Recent signals
            recent = _signal_bus.recent(limit=10)
            if recent:
                lines.append("\n## Recent Signals (last 10)")
                for sig in recent:
                    ago = int(time.time() - sig.timestamp)
                    lines.append(f"- [{sig.kind}] from {sig.source} ({ago}s ago)")

            # Last dream results
            if _autodream.last_results:
                lines.append("\n## Last Dream Cycle Results")
                for r in _autodream.last_results:
                    status = "OK" if not r.error else f"FAILED: {r.error}"
                    lines.append(f"- {r.phase}: {status} ({r.duration_seconds:.1f}s)")
                    for k, v in r.summary.items():
                        lines.append(f"  - {k}: {v}")

            # Pending nudges
            if _observer.pending_nudges:
                lines.append("\n## Pending Nudges")
                for nudge in _observer.pending_nudges[:5]:
                    lines.append(f"- [{nudge.nudge_type}] {nudge.message[:80]}...")

        return ToolResult(output="\n".join(lines))
