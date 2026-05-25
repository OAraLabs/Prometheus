"""Tool for creating local cron-style jobs.

Source: Adapted from OpenHarness tools/cron_create_tool.py (MIT).
Original path: OpenHarness/src/openharness/tools/cron_create_tool.py
Modified: Import paths changed to prometheus.*; Polish-sprint added
natural-language schedule parsing on top of the original cron-syntax path.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.gateway.cron_service import upsert_cron_job, validate_cron_expression
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult
from prometheus.tools.builtin.cron_nl import (
    ParsedSchedule,
    parse_natural_schedule,
    try_llm_fallback,
)


class CronCreateToolInput(BaseModel):
    """Arguments for cron job creation."""

    name: str = Field(description="Unique cron job name")
    schedule: str = Field(
        description=(
            "Schedule for the job. Either a standard 5-field cron expression "
            "(e.g. '*/5 * * * *', '0 9 * * 1-5') or a natural-language phrase "
            "(e.g. 'every Monday at 9am', 'in 30 minutes', 'tomorrow at 3pm', "
            "'every weekday at noon')."
        ),
    )
    command: str = Field(description="Shell command to run when triggered")
    cwd: str | None = Field(
        default=None, description="Optional working directory override"
    )
    enabled: bool = Field(default=True, description="Whether the job is active")


class CronCreateTool(BaseTool):
    """Create or replace a local cron job."""

    name = "cron_create"
    description = (
        "Create or replace a local cron job. Accepts standard 5-field cron "
        "expressions OR natural-language phrases like 'every Monday at 9am' "
        "or 'in 30 minutes'. Jobs are executed by the Prometheus cron scheduler."
    )
    input_model = CronCreateToolInput

    async def execute(
        self,
        arguments: CronCreateToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        cron_expr = arguments.schedule
        parsed: ParsedSchedule | None = None

        if not validate_cron_expression(cron_expr):
            parsed = parse_natural_schedule(arguments.schedule)
            if parsed is None:
                parsed = try_llm_fallback(arguments.schedule)
            if parsed is None or not validate_cron_expression(parsed.cron):
                return ToolResult(
                    output=(
                        f"Couldn't parse schedule: {arguments.schedule!r}\n"
                        "Provide either a standard 5-field cron expression "
                        "('0 9 * * 1', '*/5 * * * *') or a natural phrase "
                        "('every Monday at 9am', 'in 30 minutes', "
                        "'tomorrow at 3pm', 'every weekday at noon')."
                    ),
                    is_error=True,
                )
            cron_expr = parsed.cron

        upsert_cron_job(
            {
                "name": arguments.name,
                "schedule": cron_expr,
                "command": arguments.command,
                "cwd": arguments.cwd or str(context.cwd),
                "enabled": arguments.enabled,
            }
        )
        status = "enabled" if arguments.enabled else "disabled"
        lines = [
            f"Created cron job '{arguments.name}' [{cron_expr}] ({status})"
        ]
        if parsed is not None and parsed.cron != arguments.schedule:
            lines.append(f"Interpreted {arguments.schedule!r} → {cron_expr}")
        if parsed is not None and parsed.one_shot:
            lines.append(
                "Note: one-shot schedule. The cron daemon will re-fire this "
                "exact minute annually unless you delete the job after it runs."
            )
        return ToolResult(output="\n".join(lines))
