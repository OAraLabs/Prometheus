"""#790 — the tools a built-in subagent lists are the daemon's real tools, and a subagent never silently gets all of them.

Every built-in agent definition named its tools the OpenHarness way (``Bash``, ``FileRead``, ``FileWrite``,
``FileEdit``, ``Glob``, ``Grep``). This daemon's tools are ``bash``, ``read_file``, ``write_file``, ``edit_file``,
``glob`` and ``grep``, and ``ToolRegistry.get`` is an exact match (``tests/test_tool_name_convention.py`` pins that on
purpose: the fix belongs at the name). So ``SubagentSpawner._build_tool_registry`` found nothing and every built-in
subagent ran with no tools at all.

The same spawner had two ways to hand a subagent the parent's WHOLE registry by accident: an agent type with no
definition (``tools_subset`` None), and a definition that lists no tools (``defn.tools or None`` turned ``[]`` into
None). Both are closed here: a subagent only gets tools something named, and an unlisted type is refused.

These tests use the registry the daemon really builds (``build_tool_registry``) and the real spawner, not a
hand-built tool set.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import prometheus.coordinator.agent_definitions as agent_definitions
from prometheus.coordinator.agent_definitions import AgentDefinition, get_all_agent_definitions
from prometheus.coordinator.subagent import SubagentSpawner, attach_subagent_spawner
from prometheus.daemon import build_tool_registry
from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from tests.support.doubles import register_double

#: Taken at import, before any test can register a custom definition into the same dict.
BUILTINS = dict(get_all_agent_definitions())
assert BUILTINS, "no built-in agent definitions"

PARENT_MARK = "PARENT-TURN"


def _reply(blocks):  # noqa: ANN001
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=blocks),
        usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        stop_reason="stop",
    )


@register_double("builtin_agent_tools.Recorder", replaces="prometheus.providers.base.ModelProvider")
class _Recorder(ModelProvider):
    """Answers every request with one word and records the tool names it was offered."""

    def __init__(self) -> None:
        self.offered: list[set[str]] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.offered.append({t.get("name") for t in request.tools})
        yield _reply([TextBlock(text="done")])


@pytest.fixture(scope="module")
def registry():
    return build_tool_registry({})


@pytest.fixture(autouse=True)
def _restore_definitions(monkeypatch):
    """A test that registers a custom definition must not leak it into the others."""
    monkeypatch.setattr(agent_definitions, "_BUILTIN_AGENTS", dict(agent_definitions._BUILTIN_AGENTS))


def _spawn(provider, registry, **kw):  # noqa: ANN001
    return asyncio.run(SubagentSpawner(provider, parent_tool_registry=registry).spawn(task="say done", **kw))


@pytest.mark.parametrize("name", sorted(BUILTINS))
def test_every_tool_a_builtin_agent_lists_is_a_real_tool(name, registry):
    missing = [t for t in BUILTINS[name].tools if registry.get(t) is None]
    assert not missing, f"{name!r} lists tools the daemon does not register: {missing}"


@pytest.mark.parametrize("name", sorted(BUILTINS))
def test_a_builtin_subagent_is_offered_exactly_the_tools_it_lists(name, registry):
    provider = _Recorder()
    result = _spawn(provider, registry, agent_type=name)
    assert result.success, result.error
    assert provider.offered[0] == set(BUILTINS[name].tools), (
        f"{name!r} should be offered its listed tools and nothing else")


def test_an_unlisted_agent_type_is_refused_and_the_model_is_never_called(registry):
    provider = _Recorder()
    result = _spawn(provider, registry, agent_type="no-such-agent")
    assert result.success is False
    assert "no-such-agent" in (result.error or "") and "explorer" in (result.error or ""), result.error
    assert provider.offered == [], "an unlisted type must not run at all, let alone with every tool"


def test_an_unlisted_agent_type_with_an_explicit_tool_list_runs_with_only_those(registry):
    provider = _Recorder()
    result = _spawn(provider, registry, agent_type="adhoc", tools_subset=["glob"])
    assert result.success, result.error
    assert provider.offered == [{"glob"}]


def test_a_definition_that_lists_no_tools_gets_none(registry):
    agent_definitions.register_agent_definition(AgentDefinition(name="bare", description="lists no tools"))
    provider = _Recorder()
    result = _spawn(provider, registry, agent_type="bare")
    assert result.success, result.error
    assert provider.offered == [set()], "an empty tool list means no tools, not the parent's whole registry"


def test_a_tool_name_that_does_not_resolve_is_reported(registry, caplog):
    agent_definitions.register_agent_definition(
        AgentDefinition(name="legacy", description="old names", tools=["Bash", "read_file"]))
    provider = _Recorder()
    with caplog.at_level(logging.WARNING):
        result = _spawn(provider, registry, agent_type="legacy")
    assert result.success, result.error
    assert provider.offered == [{"read_file"}], "the name that resolves is kept"
    assert any("Bash" in r.getMessage() for r in caplog.records), "the name that did not resolve must be said"


@register_double("builtin_agent_tools.TwoLoopProvider", replaces="prometheus.providers.base.ModelProvider")
class _TwoLoop(ModelProvider):
    """The parent turn calls ``agent`` for an explorer; the subagent turn records the tools it is offered."""

    def __init__(self) -> None:
        self.sub_offered: list[set[str]] = []

    async def stream_message(self, request):  # noqa: ANN001
        results = [b for b in request.messages[-1].content if isinstance(b, ToolResultBlock)]
        if PARENT_MARK in (request.system_prompt or ""):
            if results:
                yield _reply([TextBlock(text="parent done")])
            else:
                yield _reply([ToolUseBlock(id="p1", name="agent", input={
                    "description": "look around", "prompt": "list files", "subagent_type": "explorer"})])
        else:
            self.sub_offered.append({t.get("name") for t in request.tools})
            yield _reply([TextBlock(text="explored")])


def test_through_the_agent_tool_an_explorer_is_offered_the_read_tools(registry):
    """The path a model takes: ``agent`` with ``subagent_type="explorer"`` on a loop the daemon's way."""
    provider = _TwoLoop()
    loop = AgentLoop(provider=provider, model="m", tool_registry=registry)
    attach_subagent_spawner(loop)
    asyncio.run(loop.run_async(system_prompt=PARENT_MARK, user_message="explore"))
    assert provider.sub_offered == [{"read_file", "glob", "grep"}]
