"""Audit Query Tool — inspect security decisions from within Prometheus.

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from prometheus.permissions.audit import AuditDecision, AuditLogger
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class AuditQueryInput(BaseModel):
    """Arguments for the audit_query tool."""

    limit: int = Field(default=20, ge=1, le=100, description="Number of entries to return")
    decision: str = Field(
        default="all",
        description="Filter by decision type: 'allow', 'deny', or 'all'",
    )
    tool: str | None = Field(default=None, description="Filter by tool name (e.g., 'bash')")


class AuditQueryTool(BaseTool):
    """Query the permission audit log to debug blocked actions."""

    name = "audit_query"
    description = "Query recent security gate decisions. Use to debug why something was blocked."
    input_model = AuditQueryInput

    def __init__(self, audit_logger: AuditLogger) -> None:
        self._audit = audit_logger

    def is_read_only(self, arguments: AuditQueryInput) -> bool:
        return True

    async def execute(
        self,
        arguments: AuditQueryInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        decision_filter = None
        if arguments.decision == "deny":
            decision_filter = AuditDecision.DENY
        elif arguments.decision == "allow":
            decision_filter = AuditDecision.ALLOW

        entries = self._audit.query_recent(
            limit=arguments.limit,
            decision=decision_filter,
            tool_name=arguments.tool,
        )

        if not entries:
            return ToolResult(output="No audit entries found.")

        lines = [f"Recent security decisions ({len(entries)}):"]
        lines.append("")

        for e in entries:
            ts = datetime.fromtimestamp(e.timestamp).strftime("%m-%d %H:%M:%S")
            icon = "ok" if e.decision == AuditDecision.ALLOW else "DENY"
            lines.append(f"[{ts}] {icon:4} {e.decision.value:16} {e.tool_name:15} {e.reason}")

        # Stats summary
        stats = self._audit.stats(hours=24)
        lines.append("")
        parts = [f"{k}={v}" for k, v in sorted(stats.items())]
        lines.append(f"Last 24h: {', '.join(parts) if parts else 'no entries'}")

        return ToolResult(output="\n".join(lines))
