"""SymbioteStatusTool — read-only status of an active or past SYMBIOTE session.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Trust level: 2 (read-only).
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from prometheus.symbiote import get_coordinator
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SymbioteStatusInput(BaseModel):
    session_id: str | None = Field(
        None, description="Session to query (None = active session)"
    )
    history_limit: int = Field(0, ge=0, le=20, description="Also return last N sessions")


class SymbioteStatusTool(BaseTool):
    name = "symbiote_status"
    description = (
        "Return the current SYMBIOTE session state (or a named session). "
        "Optionally include the most recent N past sessions."
    )
    input_model = SymbioteStatusInput
    example_call = {}

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = arguments
        assert isinstance(args, SymbioteStatusInput)
        coordinator = get_coordinator()
        if coordinator is None:
            return ToolResult(
                output=json.dumps({"active": None, "history": [], "enabled": False}),
            )
        active = coordinator.get_status(args.session_id)
        history = coordinator.get_history(args.history_limit) if args.history_limit > 0 else []
        payload = {
            "enabled": True,
            "active": active.to_dict() if active else None,
            "history": [s.to_dict() for s in history],
        }
        return ToolResult(output=json.dumps(payload, indent=2, ensure_ascii=False))
