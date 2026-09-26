"""Tool for reading skill contents."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.skills.loader import load_skill_registry
from prometheus.skills.types import SkillDefinition
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)


class SkillToolInput(BaseModel):
    """Arguments for skill lookup."""

    name: str = Field(description="Skill name to retrieve.")


class SkillTool(BaseTool):
    """Return the content of a loaded skill by name."""

    name = "skill"
    description = "Read a builtin or user-defined skill by name."
    input_model = SkillToolInput

    def is_read_only(self, arguments: SkillToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: SkillToolInput, context: ToolExecutionContext) -> ToolResult:
        registry = load_skill_registry(context.cwd)
        skill = registry.get(arguments.name)
        _record_load(context, requested=arguments.name, skill=skill)
        if skill is None:
            available = ", ".join(s.name for s in registry.list_skills())
            msg = f"Skill not found: {arguments.name}"
            if available:
                msg += f". Available: {available}"
            return ToolResult(output=msg, is_error=True)
        return ToolResult(output=skill.content)


def _record_load(
    context: ToolExecutionContext, *, requested: str, skill: SkillDefinition | None
) -> None:
    """One telemetry row per load, so skill use is measurable (the Curator reads it).

    The session is the turn's own (``effective_session_id`` — on the web path
    ``session_id`` is the shared routing namespace), and none for an ephemeral
    session, whose content columns telemetry nulls elsewhere too. The row
    itself always lands: a load is a use whoever made it. Best-effort — a
    telemetry failure must never cost the model its skill.
    """
    from prometheus.telemetry.tracker import (
        SKILL_LOAD_OPERATION,
        SKILL_LOAD_SUBSYSTEM,
        get_telemetry_handle,
    )

    telemetry = get_telemetry_handle()
    if telemetry is None:
        return
    meta = context.metadata or {}
    session_id = None if meta.get("ephemeral") else (
        meta.get("effective_session_id") or meta.get("session_id")
    )
    if skill is None:
        outcome = "failed"
        summary: dict[str, object] = {"skill": requested, "reason": "not_found"}
    else:
        outcome = "success"
        summary = {"skill": skill.name, "source": skill.source,
                   "file": Path(skill.path).stem if skill.path else None}
    try:
        telemetry.record_run(
            SKILL_LOAD_SUBSYSTEM, SKILL_LOAD_OPERATION, outcome,
            summary=summary, session_id=session_id,
        )
    except Exception:
        log.debug("skill load telemetry write failed", exc_info=True)
