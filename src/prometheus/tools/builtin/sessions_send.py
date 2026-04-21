# Provenance: openclaw/openclaw (https://github.com/openclaw/openclaw)
# Original: src/agents/tools/sessions-send-tool.ts
# License: MIT
# Modified: Rewritten as Prometheus BaseTool wrapping task manager

"""Send a message to a running agent session."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class SessionsSendInput(BaseModel):
    """Arguments for sending a message to a session."""

    task_id: str = Field(description="ID of the target task/session")
    message: str = Field(description="Message to send to the session's stdin")


class SessionsSendTool(BaseTool):
    """Send a message to a running background task or agent session."""

    name = "sessions_send"
    description = (
        "Send a text message to a running agent session (by task ID). "
        "The message is written to the session's stdin."
    )
    input_model = SessionsSendInput

    async def execute(
        self, arguments: SessionsSendInput, context: ToolExecutionContext
    ) -> ToolResult:
        manager = get_task_manager()
        try:
            await manager.write_to_task(arguments.task_id, arguments.message)
        except (KeyError, ValueError) as exc:
            return ToolResult(output=f"sessions_send failed: {exc}", is_error=True)
        return ToolResult(output=f"Message sent to session {arguments.task_id}.")
