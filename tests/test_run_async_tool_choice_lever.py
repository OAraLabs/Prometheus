"""#454 — run_async/run: the dead `tools=` parameter is gone, and the lever
that actually restricts a turn (`tool_choice`) is exposed and forwarded.

Two defects, both pinned:

1. `run_async` accepted `tools=` and NEVER READ IT — a caller handing over a
   restricted list silently got the full registry. The parameter is deleted,
   so the silent wrong answer is now a loud TypeError.

2. There was no restriction API to accept: no `tool_choice`, no `mode` on
   either entry point, so a caller wanting a tool-free turn (the three
   /benchmark diagnostics, which passed `tools=[]`) had nothing correct to
   call. `tool_choice` is now accepted and threaded into `run_loop`, where
   `"none"` empties the schema AND sets suppress_tools.

The regression these guard against: someone "helpfully" re-adds `tools=` as a
no-op because old call sites still pass it, or accepts it and still doesn't
read it — the defect wearing a compatibility shim.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok", is_error=False)

    return _T()


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    for n in ("bash", "read_file", "write_file"):
        reg.register(_tool(n))
    return reg


class _QuietProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text="4")]
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _loop(registry, provider=None) -> AgentLoop:
    return AgentLoop(
        provider=provider or _QuietProvider(),
        model="stub",
        tool_registry=registry,
        max_tool_iterations=4,
    )


# ---------------------------------------------------------------------------
# The signature contract
# ---------------------------------------------------------------------------


def test_run_async_no_longer_accepts_tools():
    """Defect 1, pinned structurally: the dead parameter is GONE. A caller
    still passing tools=[] gets a TypeError, not a silent full catalog."""
    sig = inspect.signature(AgentLoop.run_async)
    assert "tools" not in sig.parameters, (
        "run_async grew a tools= parameter back — it was never read; re-adding "
        "it re-arms the silent-wrong-answer defect (#454)"
    )


def test_run_no_longer_accepts_tools():
    sig = inspect.signature(AgentLoop.run)
    assert "tools" not in sig.parameters


def test_passing_tools_raises_typeerror(registry):
    """Behavioural twin of the structural check above — the loud failure the
    issue says is 'strictly better' than the silent one."""
    loop = _loop(registry)
    with pytest.raises(TypeError):
        loop.run("sys", "hi", tools=[])


def test_run_async_exposes_tool_choice():
    """Defect 2, pinned structurally: the real lever exists on both entry
    points."""
    assert "tool_choice" in inspect.signature(AgentLoop.run_async).parameters
    assert "tool_choice" in inspect.signature(AgentLoop.run).parameters


# ---------------------------------------------------------------------------
# The behaviour: tool_choice="none" actually suppresses the catalog
# ---------------------------------------------------------------------------


def test_tool_choice_none_sends_no_tools(registry):
    """The /benchmark intent, finally true: a tool-free turn sends no tool
    schemas to the provider. Before the fix, tools=[] was dropped and all 3
    registered tools were advertised."""
    provider = _QuietProvider()
    loop = _loop(registry, provider)
    loop.run("You are a helpful assistant.", "What is 2+2?", tool_choice="none")

    assert provider.requests, "the turn never reached the provider"
    assert not provider.requests[0].tools, (
        f"tool_choice='none' still advertised "
        f"{[t['name'] for t in provider.requests[0].tools]} — the lever is not "
        "threaded into run_loop"
    )


def test_tool_choice_none_sets_suppress_tools(registry):
    """The stronger half of "none": the provider is TOLD to drop the grammar,
    not just handed an empty list (run_loop's documented contract at
    :1645 — suppress_tools=not tools_enabled)."""
    provider = _QuietProvider()
    loop = _loop(registry, provider)
    loop.run("s", "What is 2+2?", tool_choice="none")
    assert getattr(provider.requests[0], "suppress_tools", False) is True


def test_default_advertises_the_full_catalog(registry):
    """Control: NOT passing tool_choice is byte-identical to today — every
    registered tool is advertised. The new parameter must be dormant by
    default, or the fix changes behaviour for every caller."""
    provider = _QuietProvider()
    loop = _loop(registry, provider)
    loop.run("s", "hi")
    names = {t["name"] for t in provider.requests[0].tools}
    assert names == {"bash", "read_file", "write_file"}


def test_async_and_sync_entry_points_agree(registry):
    """run() wraps run_async(); both must forward the lever identically."""
    p1, p2 = _QuietProvider(), _QuietProvider()
    _loop(registry, p1).run("s", "hi", tool_choice="none")
    asyncio.run(_loop(registry, p2).run_async("s", "hi", tool_choice="none"))
    assert not p1.requests[0].tools
    assert not p2.requests[0].tools
