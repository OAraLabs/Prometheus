"""SymbioteGraftTool — agent-facing trigger for the SYMBIOTE Graft phase.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Trust level: 1 (writes Prometheus source files; approval-gated upstream).
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from prometheus.symbiote import get_coordinator
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SymbioteGraftInput(BaseModel):
    session_id: str | None = Field(
        None, description="Session to graft (defaults to the active one)"
    )


class SymbioteGraftTool(BaseTool):
    name = "symbiote_graft"
    description = (
        "Apply the harvest's adaptation plan: write adapted files with "
        "provenance headers, generate tests, run the suite, update "
        "PROMETHEUS.md. Requires Trust Level 1 approval upstream."
    )
    input_model = SymbioteGraftInput
    example_call = {}

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
        assert isinstance(args, SymbioteGraftInput)
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
                    output="No active SYMBIOTE session.",
                    is_error=True,
                )
            session_id = active.session_id
        try:
            session = await coordinator.approve_harvest(session_id)
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)
        except Exception as exc:
            log.exception("SymbioteGraftTool: graft failed")
            return ToolResult(output=f"Graft failed: {exc}", is_error=True)
        payload = {
            "session_id": session.session_id,
            "phase": session.phase.value,
            "graft_report": session.graft_report,
            "error": session.error or None,
        }
        return ToolResult(output=json.dumps(payload, indent=2, ensure_ascii=False))
