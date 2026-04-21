# Source: OpenHarness (HKUDS/OpenHarness) tools/agent_tool.py
# License: MIT
# Modified: adapted to Prometheus's BaseTool interface, uses SubagentSpawner

"""Agent tool — spawn a subagent as a tool call."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class AgentToolInput(BaseModel):
    """Arguments for the Agent tool."""

    description: str = Field(description="Short description (3-5 words) of the subagent task.")
    prompt: str = Field(description="Full task prompt for the subagent.")
    subagent_type: str = Field(
        default="general-purpose",
        description="Agent type: general-purpose, explorer, planner, worker, verification.",
    )
    model: str | None = Field(default=None, description="Override model for this subagent.")


class AgentTool(BaseTool):
    """Spawn a subagent to handle a task in isolated context."""

    name = "Agent"
    description = (
        "Launch a subagent to handle a complex task autonomously. "
        "The subagent runs with isolated context and returns its result."
    )
    input_model = AgentToolInput

    async def execute(
        self,
        arguments: AgentToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        spawner = context.metadata.get("subagent_spawner")
        if spawner is None:
            return ToolResult(
                output="SubagentSpawner not configured — cannot spawn subagents.",
                is_error=True,
            )

        result = await spawner.spawn(
            task=arguments.prompt,
            agent_type=arguments.subagent_type,
            model=arguments.model,
        )

        if not result.success:
            return ToolResult(
                output=f"Subagent {result.agent_id} failed: {result.error}",
                is_error=True,
            )

        return ToolResult(
            output=result.text,
            metadata={
                "agent_id": result.agent_id,
                "agent_type": result.agent_type,
                "turns": result.turns,
            },
        )
