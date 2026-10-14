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

from prometheus.coordinator.agent_definitions import get_agent_definition, get_all_agent_definitions
from prometheus.engine.agent_loop import AgentLoop, RunResult
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
        permission_checker: Any | None = None,
    ) -> None:
        self._provider = provider
        self._parent_registry = parent_tool_registry
        self._model = model
        self._max_tokens = max_tokens
        self._cwd = cwd or Path.cwd()
        self._adapter = adapter
        self._telemetry = telemetry
        # The gate each subagent's loop runs under. None keeps the old behaviour (no gate at all); see
        # attach_subagent_spawner for why the daemon always passes one.
        self._permission_checker = permission_checker
        self._active: dict[str, asyncio.Task] = {}

    def _build_tool_registry(self, tool_names: list[str] | None) -> ToolRegistry | None:
        """Build a subset registry from the parent, or return the full parent."""
        if self._parent_registry is None:
            return None
        if tool_names is None:
            return self._parent_registry

        subset = ToolRegistry()
        unresolved: list[str] = []
        for name in tool_names:
            tool = self._parent_registry.get(name)
            if tool is not None:
                subset.register(tool)
            else:
                unresolved.append(name)
        if unresolved:
            # Lookup is exact (the convention in tests/test_tool_name_convention.py), so a miscased or misspelt
            # name resolves to nothing. Silently dropping it is how every built-in subagent ended up with no
            # tools at all (#790): say so, once per spawn.
            logger.warning(
                "Subagent tool list names tools the parent registry does not have, so they are dropped: %s",
                ", ".join(unresolved),
            )
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
        parent_session_id: str | None = None,
    ) -> SubagentResult:
        """Spawn a subagent, run to completion, return result.

        Args:
            task: The user message / task prompt for the subagent.
            agent_type: Name of a registered AgentDefinition.
            tools_subset: Tool names to make available (None = all parent tools).
            model: Override model for this subagent.
            system_prompt: Override system prompt (otherwise uses agent definition).
            max_turns: Override max turns.
            parent_session_id: The conversation that spawned it, if any. Names
                the telemetry rows only (see below); None when there is none,
                or when that turn is ephemeral.
        """
        agent_id = f"sub_{uuid.uuid4().hex[:8]}"
        # The id this subagent's telemetry rows are filed under (WP-X.21 T4):
        # its OWN, derived from the parent's — never the parent's itself, or
        # the golden-trace exporter would pair the subagent's calls with the
        # parent's conversation. RECORD-ONLY: the loop runs with no session of
        # its own, as before, so its permission origin stays 'system'. The
        # parent is named in the run's advertisement summary too.
        record_session_id = f"subagent:{parent_session_id or 'none'}:{agent_id}"
        defn = get_agent_definition(agent_type)

        # A subagent only ever gets tools something NAMED. An agent type with no definition and no explicit tool
        # list used to fall through to the parent's WHOLE registry (``tools_subset`` None), and so did a
        # definition that listed no tools (``defn.tools or None``): two silent ways to hand a model every tool
        # the daemon has (#790). The first is refused with the valid types named, so the model can correct
        # itself; the second is now simply no tools.
        if defn is None and tools_subset is None:
            known = ", ".join(sorted(get_all_agent_definitions()))
            logger.warning("Refusing to spawn a subagent of unlisted type %r (known types: %s)", agent_type, known)
            return SubagentResult(
                agent_id=agent_id,
                agent_type=agent_type,
                text="",
                turns=0,
                success=False,
                error=f"unknown agent type {agent_type!r}; known types: {known}",
            )

        effective_prompt = system_prompt
        effective_tools = tools_subset
        effective_max_turns = max_turns or 50

        if defn is not None:
            effective_prompt = effective_prompt or defn.system_prompt
            effective_tools = effective_tools if tools_subset is not None else list(defn.tools)
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
            permission_checker=self._permission_checker,
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
                record_session_id=record_session_id,
                record_summary={"parent_session": parent_session_id},
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


def attach_subagent_spawner(agent_loop: AgentLoop) -> SubagentSpawner:
    """Give ``agent_loop``'s tools a spawner, so the ``agent`` tool can run (#761).

    ``AgentTool`` reads ``context.metadata["subagent_spawner"]``. The loop builds that metadata for every
    tool call from its ``tool_metadata``, so the spawner is attached there once the loop exists, the way
    ``run_daemon()`` attaches the printing-press registry. Nothing set the key before, so the tool was
    offered to the model and always answered "SubagentSpawner not configured".

    Everything the spawner needs is read from the loop it serves, so a subagent uses the same provider,
    model, tools, adapter and telemetry as the turn that spawned it.

    The spawner is also handed the loop's ``permission_checker``. This is a safety property, not a
    convenience: a loop with no checker skips the permission block altogether, so a subagent built without
    one would run every tool it is given (an unknown ``subagent_type`` gets the parent's whole registry)
    with no gate, no approvals, no denied paths and no audit rows. A subagent's loop has no session of
    its own, so the gate sees it as the stricter ``system`` origin.

    Not covered here: a subagent gets no hooks, and it carries no spawner of its own, so a subagent
    cannot spawn another (``agent`` inside one answers "not configured").
    """
    spawner = SubagentSpawner(
        provider=agent_loop._provider,
        parent_tool_registry=agent_loop._tool_registry,
        model=agent_loop._model,
        max_tokens=agent_loop._max_tokens,
        cwd=agent_loop._cwd,
        adapter=agent_loop._adapter,
        telemetry=agent_loop._telemetry,
        permission_checker=agent_loop._permission_checker,
    )
    if agent_loop._tool_metadata is None:
        agent_loop._tool_metadata = {}
    agent_loop._tool_metadata["subagent_spawner"] = spawner
    return spawner
