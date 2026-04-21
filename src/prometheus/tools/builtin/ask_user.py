# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/ask_user_question_tool.py
# License: Apache-2.0
# Modified: Rewritten as Prometheus BaseTool

"""Ask the user a clarifying question and wait for their response."""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class AskUserInput(BaseModel):
    """Arguments for asking the user a question."""

    question: str = Field(description="The question to ask the user")


class AskUserTool(BaseTool):
    """Ask the user a clarifying question during task execution."""

    name = "ask_user"
    description = (
        "Ask the user a question and wait for their response. "
        "Use when you need clarification before proceeding."
    )
    input_model = AskUserInput

    async def execute(
        self, arguments: AskUserInput, context: ToolExecutionContext
    ) -> ToolResult:
        # The actual user interaction is handled by the agent loop.
        # This tool returns the question as output, and the agent loop
        # intercepts it to prompt the user.
        return ToolResult(
            output=arguments.question,
            metadata={"requires_user_input": True},
        )
