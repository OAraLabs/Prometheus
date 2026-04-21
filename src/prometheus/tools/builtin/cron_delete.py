"""Tool for deleting local cron-style jobs.

Source: Adapted from OpenHarness tools/cron_delete_tool.py (MIT).
Original path: OpenHarness/src/openharness/tools/cron_delete_tool.py
Modified: Import paths changed to prometheus.*.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.gateway.cron_service import delete_cron_job
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class CronDeleteToolInput(BaseModel):
    """Arguments for cron job deletion."""

    name: str = Field(description="Name of the cron job to delete")


class CronDeleteTool(BaseTool):
    """Delete a local cron job by name."""

    name = "cron_delete"
    description = "Delete a local cron job by its unique name."
    input_model = CronDeleteToolInput

    async def execute(
        self,
        arguments: CronDeleteToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        deleted = delete_cron_job(arguments.name)
        if deleted:
            return ToolResult(output=f"Deleted cron job '{arguments.name}'")
        return ToolResult(
            output=f"Cron job '{arguments.name}' not found",
            is_error=True,
        )
