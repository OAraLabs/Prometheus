"""Tier-2 wire/fix batch (audit H1, H2, H3).

- H1: per-result truncation (tool_result_max) is wired into _execute_tool_call,
  caps every result including errors.
- H2: the GBNF generator emits valid JSON (no forced/double/trailing commas) and
  a tool-OR-text root (won't force a tool call); tier-full withholds native
  tools so the grammar actually engages. Live tier-light path is untouched.
- H3: the hook registry is built from config (the loader is wired) instead of an
  always-empty registry.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from prometheus.adapter.enforcer import StructuredOutputEnforcer
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


# ---------------------------------------------------------------------------
# H1 — per-result truncation wired into the loop
# ---------------------------------------------------------------------------


class _BigTool(BaseTool):
    name = "big"
    description = "returns a large output"
    input_model = _EmptyInput

    def __init__(self, *, is_err: bool = False) -> None:
        self._is_err = is_err

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="x" * 100_000, is_error=self._is_err)


def _ctx(reg, **kw):
    return LoopContext(
        provider=None, model="stub", system_prompt="", max_tokens=1024,
        tool_registry=reg, **kw,
    )


class TestPerResultTruncationH1:
    def test_large_result_truncated_when_max_set(self):
        reg = ToolRegistry()
        reg.register(_BigTool())
        block = asyncio.run(
            _execute_tool_call(_ctx(reg, tool_result_max=100), "big", "t1", {})
        )
        assert len(block.content) < 100_000
        assert "truncated" in block.content.lower()

    def test_no_truncation_when_max_zero(self):
        reg = ToolRegistry()
        reg.register(_BigTool())
        # tool_result_max defaults to 0 (disabled) — back-compat.
        block = asyncio.run(_execute_tool_call(_ctx(reg), "big", "t1", {}))
        assert len(block.content) == 100_000

    def test_error_result_is_also_truncated(self):
        # The cross-result turn budget skips errors; per-result truncation must
        # NOT, or a giant error payload stays unbounded.
        reg = ToolRegistry()
        reg.register(_BigTool(is_err=True))
        block = asyncio.run(
            _execute_tool_call(_ctx(reg, tool_result_max=100), "big", "t1", {})
        )
        assert block.is_error
        assert len(block.content) < 100_000


# ---------------------------------------------------------------------------
# H2 — grammar generator produces valid JSON + tool-or-text root
# ---------------------------------------------------------------------------


class TestGrammarGeneratorH2:
    def _args_rule(self, grammar: str, prefix: str) -> str:
        line = next(l for l in grammar.splitlines() if l.startswith(f"{prefix}-args ::="))
        return line

    def test_required_plus_optional_has_no_double_or_trailing_comma(self):
        e = StructuredOutputEnforcer()
        g = e.generate_grammar([{
            "name": "bash",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        }])
        rule = self._args_rule(g, "bash")
        # required first, optional as a leading-comma group:
        assert '"\\"command\\"" ws ":" ws string' in rule
        assert '(ws "," ws "\\"timeout\\"" ws ":" ws integer)?' in rule
        # the old bug: a mandatory comma AND a second comma inside the optional.
        assert 'ws string ws "," ws (ws "," ws' not in rule

    def test_all_optional_has_no_trailing_comma(self):
        e = StructuredOutputEnforcer()
        g = e.generate_grammar([{
            "name": "task_list",
            "input_schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
                "required": [],
            },
        }])
        rule = self._args_rule(g, "task-list")
        assert "( task-list-member (ws \",\" ws task-list-member)* )?" in rule
        # old bug emitted a per-member trailing comma: (... ws "," ws)?
        assert 'ws "," ws)?' not in g

    def test_root_is_tool_or_text(self):
        e = StructuredOutputEnforcer()
        g = e.generate_grammar([{
            "name": "bash",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }])
        # Must allow prose, not force a tool call every turn.
        assert "root ::= tool-call | prose" in g
        assert "prose ::=" in g


# ---------------------------------------------------------------------------
# H2 — tier-full withholds native tools so the grammar engages
# ---------------------------------------------------------------------------


class _CapturingProvider(ModelProvider):
    def __init__(self) -> None:
        self.last_tools = "unset"

    async def stream_message(self, request):  # noqa: ANN001
        self.last_tools = request.tools
        msg = ConversationMessage(role="assistant", content=[TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _reg_with_tool():
    class _Echo(BaseTool):
        name = "echo"
        description = "echoes"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok")

    reg = ToolRegistry()
    reg.register(_Echo())
    return reg


def _drive(ctx):
    async def go():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")]):
            pass
    asyncio.run(go())


class TestTierFullWithholdsToolsH2:
    def test_tier_full_sends_no_tools(self):
        from prometheus.adapter import ModelAdapter

        prov = _CapturingProvider()
        ctx = LoopContext(
            provider=prov, model="m", system_prompt="", max_tokens=64,
            tool_registry=_reg_with_tool(),
            adapter=ModelAdapter(strictness="MEDIUM"),  # tier full
        )
        _drive(ctx)
        assert prov.last_tools == []  # withheld so the grammar isn't dropped

    def test_tier_light_still_sends_tools(self):
        from prometheus.adapter import ModelAdapter
        from prometheus.adapter.formatter import QwenFormatter

        prov = _CapturingProvider()
        ctx = LoopContext(
            provider=prov, model="qwen", system_prompt="", max_tokens=64,
            tool_registry=_reg_with_tool(),
            adapter=ModelAdapter(formatter=QwenFormatter(), tier="light"),
        )
        _drive(ctx)
        assert prov.last_tools  # native tool calling — tools still sent


# ---------------------------------------------------------------------------
# H3 — hook registry is built from config (loader wired)
# ---------------------------------------------------------------------------


class TestHooksLoaderWiredH3:
    def test_empty_config_yields_no_hooks(self):
        from prometheus.hooks.loader import load_hook_registry
        from prometheus.hooks.events import HookEvent

        reg = load_hook_registry({})
        assert sum(len(reg.get(ev)) for ev in HookEvent) == 0

    def test_configured_command_hook_loads_and_fires(self):
        # End-to-end over the path the daemon now wires: config -> registry ->
        # executor -> PRE_TOOL_USE fires the hook.
        from prometheus.hooks.loader import load_hook_registry
        from prometheus.hooks.executor import HookExecutor, HookExecutionContext
        from prometheus.hooks.events import HookEvent

        cfg = {
            "pre_tool_use": [
                {"type": "command", "command": "echo hooked", "block_on_failure": False},
            ],
        }
        reg = load_hook_registry(cfg)
        assert len(reg.get(HookEvent.PRE_TOOL_USE)) == 1

        from pathlib import Path
        from unittest.mock import MagicMock

        executor = HookExecutor(
            registry=reg,
            context=HookExecutionContext(
                cwd=Path.cwd(), provider=MagicMock(), default_model="m",
            ),
        )
        result = asyncio.run(
            executor.execute(
                HookEvent.PRE_TOOL_USE,
                {"tool_name": "bash", "event": HookEvent.PRE_TOOL_USE.value},
            )
        )
        assert any("hooked" in (r.output or "") for r in result.results)

    def test_daemon_wires_loader_not_empty_registry(self):
        # Guard the wiring itself: the daemon must build the registry from
        # config via load_hook_registry, not an always-empty HookRegistry().
        import inspect
        import prometheus.daemon as daemon

        src = inspect.getsource(daemon)
        assert "load_hook_registry(config.get(\"hooks\"" in src
