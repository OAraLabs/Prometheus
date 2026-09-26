"""Tool for reading skill contents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.skills.loader import load_skill_registry
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class SkillToolInput(BaseModel):
    """Arguments for skill lookup."""

    name: str = Field(description="Skill name to retrieve.")


class SkillTool(BaseTool):
    """Return the content of a loaded skill by name."""

    name = "skill"
    description = (
        "Load a skill: saved step-by-step instructions for one kind of task (a "
        "procedure, a house convention, a known fix). Call it with the skill's name "
        "before starting a task that a skill covers, then follow what it returns. The "
        "system prompt names the core skills; find others with tool_search."
    )
    input_model = SkillToolInput

    def is_read_only(self, arguments: SkillToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: SkillToolInput, context: ToolExecutionContext) -> ToolResult:
        registry = load_skill_registry(context.cwd)
        skill = registry.get(arguments.name)
        if skill is None:
            available = ", ".join(s.name for s in registry.list_skills())
            msg = f"Skill not found: {arguments.name}"
            if available:
                msg += f". Available: {available}"
            return ToolResult(output=msg, is_error=True)
        return ToolResult(output=skill.content)
