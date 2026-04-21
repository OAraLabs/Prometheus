# Source: Novel code for Prometheus Sprint 8, inspired by OpenHarness agent_tool.py
# and analysis-2 s04_subagent pattern.

"""SubagentSpawner — spawn isolated AgentLoop instances as subagents."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.coordinator.agent_definitions import AgentDefinition, get_agent_definition
from prometheus.engine.agent_loop import AgentLoop, RunResult
from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ModelProvider
from prometheus.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubagentResult:
    """Result returned from a subagent execution."""

    agent_id: str
    agent_type: str
    text: str
    turns: int = 0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SubagentSpawner:
    """Spawn isolated subagent instances using Prometheus's AgentLoop.

    Each subagent gets:
    - A fresh messages list (isolated context)
    - An optional tool subset (from parent registry)
    - Its own AgentLoop instance
    Results are returned without polluting the parent's conversation.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        parent_tool_registry: ToolRegistry | None = None,
        model: str = "qwen3.5-32b",
        max_tokens: int = 4096,
        cwd: Path | None = None,
        adapter: Any | None = None,
        telemetry: Any | None = None,
    ) -> None:
        self._provider = provider
        self._parent_registry = parent_tool_registry
        self._model = model
        self._max_tokens = max_tokens
        self._cwd = cwd or Path.cwd()
        self._adapter = adapter
        self._telemetry = telemetry
        self._active: dict[str, asyncio.Task] = {}

    def _build_tool_registry(self, tool_names: list[str] | None) -> ToolRegistry | None:
        """Build a subset registry from the parent, or return the full parent."""
        if self._parent_registry is None:
            return None
        if tool_names is None:
            return self._parent_registry

        subset = ToolRegistry()
        for name in tool_names:
            tool = self._parent_registry.get(name)
            if tool is not None:
                subset.register(tool)
        return subset

    async def spawn(
        self,
        task: str,
        *,
        agent_type: str = "general-purpose",
        tools_subset: list[str] | None = None,
        model: str | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ) -> SubagentResult:
        """Spawn a subagent, run to completion, return result.

        Args:
            task: The user message / task prompt for the subagent.
            agent_type: Name of a registered AgentDefinition.
            tools_subset: Tool names to make available (None = all parent tools).
            model: Override model for this subagent.
            system_prompt: Override system prompt (otherwise uses agent definition).
            max_turns: Override max turns.
        """
        agent_id = f"sub_{uuid.uuid4().hex[:8]}"
        defn = get_agent_definition(agent_type)

        effective_prompt = system_prompt
        effective_tools = tools_subset
        effective_max_turns = max_turns or 50

        if defn is not None:
            effective_prompt = effective_prompt or defn.system_prompt
            effective_tools = effective_tools if tools_subset is not None else (defn.tools or None)
            effective_max_turns = max_turns or defn.max_turns
        else:
            effective_prompt = effective_prompt or "You are a helpful assistant."

        registry = self._build_tool_registry(effective_tools)

        loop = AgentLoop(
            provider=self._provider,
            model=model or self._model,
            max_tokens=self._max_tokens,
            max_turns=effective_max_turns,
            tool_registry=registry,
            adapter=self._adapter,
            telemetry=self._telemetry,
            cwd=self._cwd,
        )

        logger.info(
            "Spawning subagent %s (type=%s, tools=%s)",
            agent_id,
            agent_type,
            effective_tools,
        )

        try:
            result: RunResult = await loop.run_async(
                system_prompt=effective_prompt,
                user_message=task,
            )
            return SubagentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                text=result.text,
                turns=result.turns,
                success=True,
            )
        except Exception as exc:
            logger.error("Subagent %s failed: %s", agent_id, exc)
            return SubagentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                text="",
                turns=0,
                success=False,
                error=str(exc),
            )

    async def spawn_parallel(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[SubagentResult]:
        """Spawn multiple subagents in parallel.

        Each dict in tasks should have 'task' (str) and optionally
        'agent_type', 'tools_subset', 'model', 'system_prompt', 'max_turns'.
        """
        coros = [self.spawn(**t) for t in tasks]
        return list(await asyncio.gather(*coros))
