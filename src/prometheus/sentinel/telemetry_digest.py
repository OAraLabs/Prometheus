"""TelemetryDigest — periodic health report from tool call data.

Source: Novel code for Prometheus Sprint 9.
Compares current period against baseline to flag anomalies. No LLM needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prometheus.telemetry.tracker import ToolCallTelemetry

log = logging.getLogger(__name__)


@dataclass
class DigestAnomaly:
    """A single telemetry anomaly."""

    tool_name: str
    metric: str  # "success_rate_drop", "latency_spike", "retry_increase"
    current_value: float
    baseline_value: float
    severity: str = "warning"  # "warning" or "error"


@dataclass
class DigestResult:
    """Output of a telemetry digest."""

    period_hours: int = 24
    total_calls: int = 0
    anomalies: list[DigestAnomaly] = field(default_factory=list)
    summary: str = ""

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0


class TelemetryDigest:
    """Generate periodic health reports from tool call telemetry."""

    def __init__(
        self,
        telemetry: ToolCallTelemetry,
        *,
        period_hours: int = 24,
        baseline_hours: int = 168,  # 7 days
    ) -> None:
        self._telemetry = telemetry
        self._period_hours = period_hours
        self._baseline_hours = baseline_hours

    def generate(self) -> DigestResult:
        """Compare current period against baseline. No LLM needed."""
        now = time.time()
        current_since = now - (self._period_hours * 3600)
        baseline_since = now - (self._baseline_hours * 3600)

        current = self._telemetry.report(since=current_since)
        baseline = self._telemetry.report(since=baseline_since)

        anomalies = self._compare(current, baseline)

        result = DigestResult(
            period_hours=self._period_hours,
            total_calls=current.get("total_calls", 0),
            anomalies=anomalies,
            summary=self._format_summary(current, anomalies),
        )
        return result

    def _compare(
        self,
        current: dict[str, Any],
        baseline: dict[str, Any],
    ) -> list[DigestAnomaly]:
        """Flag anomalies: success drop >5%, retry increase >10%, latency >50%."""
        anomalies: list[DigestAnomaly] = []

        current_tools = current.get("tools", {})
        baseline_tools = baseline.get("tools", {})

        for tool_name, ct in current_tools.items():
            bt = baseline_tools.get(tool_name)
            if not bt or bt.get("calls", 0) < 3:
                continue  # Not enough baseline data

            # Success rate drop > 5%
            c_rate = ct.get("success_rate", 1.0)
            b_rate = bt.get("success_rate", 1.0)
            if b_rate > 0 and (b_rate - c_rate) > 0.05:
                anomalies.append(DigestAnomaly(
                    tool_name=tool_name,
                    metric="success_rate_drop",
                    current_value=c_rate,
                    baseline_value=b_rate,
                    severity="error" if (b_rate - c_rate) > 0.20 else "warning",
                ))

            # Retry rate increase > 10%
            c_retry = ct.get("avg_retries", 0.0)
            b_retry = bt.get("avg_retries", 0.0)
            if b_retry > 0 and c_retry > 0 and (c_retry - b_retry) / b_retry > 0.10:
                anomalies.append(DigestAnomaly(
                    tool_name=tool_name,
                    metric="retry_increase",
                    current_value=c_retry,
                    baseline_value=b_retry,
                ))

            # Latency increase > 50%
            c_lat = ct.get("avg_latency_ms", 0.0)
            b_lat = bt.get("avg_latency_ms", 0.0)
            if b_lat > 0 and c_lat > 0 and (c_lat - b_lat) / b_lat > 0.50:
                anomalies.append(DigestAnomaly(
                    tool_name=tool_name,
                    metric="latency_spike",
                    current_value=c_lat,
                    baseline_value=b_lat,
                ))

        return anomalies

    @staticmethod
    def _format_summary(
        report: dict[str, Any], anomalies: list[DigestAnomaly]
    ) -> str:
        """Human-readable digest summary."""
        total = report.get("total_calls", 0)
        rate = report.get("overall_success_rate", 0.0)
        lines = [
            f"Telemetry digest: {total} calls, {rate:.1%} overall success rate",
        ]
        if anomalies:
            lines.append(f"  {len(anomalies)} anomaly(ies) detected:")
            for a in anomalies:
                lines.append(
                    f"    [{a.severity}] {a.tool_name}: {a.metric} "
                    f"(current={a.current_value:.2f}, baseline={a.baseline_value:.2f})"
                )
        else:
            lines.append("  No anomalies detected.")
        return "\n".join(lines)
