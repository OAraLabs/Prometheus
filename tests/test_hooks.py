"""Tests for Sprint 2: hook pipeline (registry, executor, block/allow behavior)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.hooks.events import HookEvent
from prometheus.hooks.executor import HookExecutionContext, HookExecutor
from prometheus.hooks.registry import HookRegistry
from prometheus.hooks.schemas import CommandHookDefinition
from prometheus.hooks.types import AggregatedHookResult, HookResult
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)


# ---------------------------------------------------------------------------
# Minimal stub provider for prompt/agent hooks
# ---------------------------------------------------------------------------

class StubProvider(ModelProvider):
    """Returns a scripted text response."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        msg = ConversationMessage(role="assistant", content=[TextBlock(text=self._text)])
        yield ApiTextDeltaEvent(text=self._text)
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=5, output_tokens=5),
            stop_reason="stop",
        )


def _ctx(tmp_path: Path) -> HookExecutionContext:
    return HookExecutionContext(
        cwd=tmp_path,
        provider=StubProvider('{"ok": true}'),
        default_model="stub",
    )


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------

def test_registry_get_empty():
    reg = HookRegistry()
    assert reg.get(HookEvent.PRE_TOOL_USE) == []


def test_registry_add_and_get():
    reg = HookRegistry()
    hook = CommandHookDefinition(command="echo hi")
    reg.add(HookEvent.PRE_TOOL_USE, hook)
    hooks = reg.get(HookEvent.PRE_TOOL_USE)
    assert len(hooks) == 1
    assert hooks[0] is hook


def test_registry_clear_single_event():
    reg = HookRegistry()
    reg.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="echo a"))
    reg.add(HookEvent.POST_TOOL_USE, CommandHookDefinition(command="echo b"))
    reg.clear(HookEvent.PRE_TOOL_USE)
    assert reg.get(HookEvent.PRE_TOOL_USE) == []
    assert len(reg.get(HookEvent.POST_TOOL_USE)) == 1


def test_registry_clear_all():
    reg = HookRegistry()
    reg.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="echo a"))
    reg.add(HookEvent.POST_TOOL_USE, CommandHookDefinition(command="echo b"))
    reg.clear()
    assert reg.get(HookEvent.PRE_TOOL_USE) == []
    assert reg.get(HookEvent.POST_TOOL_USE) == []


# ---------------------------------------------------------------------------
# HookResult / AggregatedHookResult
# ---------------------------------------------------------------------------

def test_aggregated_result_not_blocked_by_default():
    agg = AggregatedHookResult(results=[
        HookResult(hook_type="command", success=True),
    ])
    assert not agg.blocked


def test_aggregated_result_blocked_when_any_blocked():
    agg = AggregatedHookResult(results=[
        HookResult(hook_type="command", success=True),
        HookResult(hook_type="command", success=False, blocked=True, reason="denied"),
    ])
    assert agg.blocked
    assert agg.reason == "denied"


def test_aggregated_result_empty_not_blocked():
    agg = AggregatedHookResult()
    assert not agg.blocked
    assert agg.reason == ""


# ---------------------------------------------------------------------------
# HookExecutor — command hooks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_no_hooks_returns_empty(tmp_path):
    reg = HookRegistry()
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert isinstance(result, AggregatedHookResult)
    assert not result.blocked
    assert result.results == []


@pytest.mark.asyncio
async def test_executor_command_hook_allow(tmp_path):
    reg = HookRegistry()
    reg.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="exit 0"))
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert not result.blocked
    assert result.results[0].success


@pytest.mark.asyncio
async def test_executor_command_hook_deny_when_block_on_failure(tmp_path):
    reg = HookRegistry()
    reg.add(
        HookEvent.PRE_TOOL_USE,
        CommandHookDefinition(command="exit 1", block_on_failure=True),
    )
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert result.blocked


@pytest.mark.asyncio
async def test_executor_command_hook_fail_without_block(tmp_path):
    reg = HookRegistry()
    reg.add(
        HookEvent.PRE_TOOL_USE,
        CommandHookDefinition(command="exit 1", block_on_failure=False),
    )
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert not result.blocked
    assert not result.results[0].success


@pytest.mark.asyncio
async def test_executor_command_hook_captures_output(tmp_path):
    reg = HookRegistry()
    reg.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="echo sentinel"))
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert "sentinel" in result.results[0].output


@pytest.mark.asyncio
async def test_executor_command_hook_timeout(tmp_path):
    reg = HookRegistry()
    reg.add(
        HookEvent.PRE_TOOL_USE,
        CommandHookDefinition(command="sleep 10", timeout_seconds=1, block_on_failure=True),
    )
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert result.blocked
    assert "timed out" in result.results[0].reason


# ---------------------------------------------------------------------------
# HookExecutor — matcher filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_matcher_skips_non_matching(tmp_path):
    """Hook with matcher='write_*' should not run for tool_name='bash'."""
    reg = HookRegistry()
    reg.add(
        HookEvent.PRE_TOOL_USE,
        CommandHookDefinition(command="exit 1", block_on_failure=True, matcher="write_*"),
    )
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert not result.blocked
    assert result.results == []


@pytest.mark.asyncio
async def test_executor_matcher_runs_matching(tmp_path):
    """Hook with matcher='bash' should run for tool_name='bash'."""
    reg = HookRegistry()
    reg.add(
        HookEvent.PRE_TOOL_USE,
        CommandHookDefinition(command="exit 0", matcher="bash"),
    )
    executor = HookExecutor(reg, _ctx(tmp_path))
    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert len(result.results) == 1


# ---------------------------------------------------------------------------
# update_registry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_update_registry(tmp_path):
    reg1 = HookRegistry()
    reg1.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="exit 1", block_on_failure=True))
    executor = HookExecutor(reg1, _ctx(tmp_path))

    reg2 = HookRegistry()  # empty registry
    executor.update_registry(reg2)

    result = await executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert not result.blocked
