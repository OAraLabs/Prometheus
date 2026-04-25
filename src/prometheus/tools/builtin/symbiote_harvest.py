"""SymbioteHarvestTool — agent-facing trigger for the SYMBIOTE Harvest phase.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Trust level: 1 (clones a remote repo to disk and reads its files; should
be approval-gated by the calling layer — e.g. the /symbiote Telegram flow
uses ApprovalQueue.request_approval before invoking this tool).
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from prometheus.symbiote import get_coordinator
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SymbioteHarvestInput(BaseModel):
    candidate: str = Field(..., description="full_name of an approved scout candidate")
    session_id: str | None = Field(
        None, description="Session to attach to (defaults to the active one)"
    )


class SymbioteHarvestTool(BaseTool):
    name = "symbiote_harvest"
    description = (
        "Clone an approved GitHub repository, scan its source, extract "
        "relevant modules. Requires Trust Level 1 approval upstream."
    )
    input_model = SymbioteHarvestInput
    example_call = {"candidate": "owner/repo"}

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return False

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = arguments
        assert isinstance(args, SymbioteHarvestInput)
        coordinator = get_coordinator()
        if coordinator is None:
            return ToolResult(
                output="SYMBIOTE is not active. Set symbiote.enabled in config.",
                is_error=True,
            )
        session_id = args.session_id
        if session_id is None:
            active = coordinator.get_status()
            if active is None:
                return ToolResult(
                    output="No active SYMBIOTE session. Run symbiote_scout first.",
                    is_error=True,
                )
            session_id = active.session_id
        try:
            session = await coordinator.approve_scout(session_id, args.candidate)
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)
        except Exception as exc:
            log.exception("SymbioteHarvestTool: harvest failed")
            return ToolResult(output=f"Harvest failed: {exc}", is_error=True)
        payload = {
            "session_id": session.session_id,
            "phase": session.phase.value,
            "harvest_report": session.harvest_report,
            "error": session.error or None,
        }
        return ToolResult(output=json.dumps(payload, indent=2, ensure_ascii=False))
