"""SymbioteScoutTool — agent-facing trigger for the SYMBIOTE Scout phase.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Trust level: 2 (read-only — GitHub search + LLM scoring, no side effects).
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from prometheus.symbiote import get_coordinator
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SymbioteScoutInput(BaseModel):
    problem_statement: str = Field(..., description="Capability need or problem to solve")
    language: str = Field("Python", description="Filter by language")
    min_stars: int = Field(10, ge=0, description="Minimum star count")


class SymbioteScoutTool(BaseTool):
    name = "symbiote_scout"
    description = (
        "Search GitHub for open-source solutions to a capability gap. "
        "Returns a ranked candidate list. Read-only; no clones, no writes."
    )
    input_model = SymbioteScoutInput
    example_call = {
        "problem_statement": "I need YAML schema validation",
        "language": "Python",
        "min_stars": 50,
    }

    def is_read_only(self, arguments: BaseModel) -> bool:  # noqa: D401
        del arguments
        return True

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = arguments
        assert isinstance(args, SymbioteScoutInput)
        coordinator = get_coordinator()
        if coordinator is None:
            return ToolResult(
                output="SYMBIOTE is not active. Set symbiote.enabled in config.",
                is_error=True,
            )
        try:
            session = await coordinator.start_scout(args.problem_statement)
        except RuntimeError as exc:
            return ToolResult(output=f"Scout could not start: {exc}", is_error=True)
        except Exception as exc:
            log.exception("SymbioteScoutTool: scout failed")
            return ToolResult(output=f"Scout failed: {exc}", is_error=True)
        payload = {
            "session_id": session.session_id,
            "phase": session.phase.value,
            "scout_report": session.scout_report,
            "error": session.error or None,
        }
        return ToolResult(output=json.dumps(payload, indent=2, ensure_ascii=False))
