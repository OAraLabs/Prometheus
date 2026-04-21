"""Tool for listing local cron-style jobs.

Source: Adapted from OpenHarness tools/cron_list_tool.py (MIT).
Original path: OpenHarness/src/openharness/tools/cron_list_tool.py
Modified: Import paths changed to prometheus.*.
"""

from __future__ import annotations

from pydantic import BaseModel

from prometheus.gateway.cron_scheduler import is_scheduler_running
from prometheus.gateway.cron_service import load_cron_jobs
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class CronListToolInput(BaseModel):
    """Arguments for cron job listing (none required)."""


class CronListTool(BaseTool):
    """List all local cron jobs with their status."""

    name = "cron_list"
    description = "List all registered cron jobs with schedule, status, and next run time."
    input_model = CronListToolInput

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(
        self,
        arguments: CronListToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        jobs = load_cron_jobs()
        scheduler_up = is_scheduler_running()

        if not jobs:
            status = "running" if scheduler_up else "stopped"
            return ToolResult(
                output=f"No cron jobs registered. Scheduler: {status}"
            )

        lines: list[str] = []
        lines.append(f"Scheduler: {'running' if scheduler_up else 'stopped'}")
        lines.append(f"Total jobs: {len(jobs)}")
        lines.append("")

        for job in jobs:
            enabled = job.get("enabled", True)
            state = "enabled" if enabled else "disabled"
            name = job.get("name", "?")
            schedule = job.get("schedule", "?")
            next_run = job.get("next_run", "—")
            last_run = job.get("last_run", "never")
            last_status = job.get("last_status", "—")
            command = job.get("command", "")

            lines.append(f"  {name} [{state}]")
            lines.append(f"    Schedule:    {schedule}")
            lines.append(f"    Command:     {command}")
            lines.append(f"    Next run:    {next_run}")
            lines.append(f"    Last run:    {last_run} ({last_status})")
            lines.append("")

        return ToolResult(output="\n".join(lines))
