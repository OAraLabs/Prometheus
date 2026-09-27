"""A round served by the provider fallback is recorded under the model that served it
(WP-X.21 T14, docs/audits/TELEMETRY-GAPS.md).

On a terminal provider failure (an expired key, an exhausted plan) the loop
serves the round from ``context.fallback``, a local model. ``_on_degrade``
rewrote the prompt's identity line and nothing else, so every row the round
wrote named the model that FAILED: its tool calls and transitions carried the
cloud model, and a successful zero-retry call was flagged ``is_golden`` because
the provider name was the cloud one. That files the local model's output as a
teacher example — the direction ``telemetry/tracker.py`` calls the worse one.

Pinned here, against the rows actually written: in a degraded round, the tool
rows (success and failure alike) and the transitions name the fallback and are
never golden; a healthy cloud round still records the cloud model and is still
golden. Only what is recorded changes: the next round still asks the primary.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import AsyncIterator

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.fallback import FallbackTarget
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import PermissionDecision
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

CLOUD, LOCAL = "qwen3.8-max", "Qwen3.8-27B"


class _EmptyInput(BaseModel):
    pass


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = "test tool"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok", is_error=False)

    return _T()


class _ExpiredCloud(ModelProvider):
    """A cloud provider whose key has expired: 401 on every call."""

    provider_name = "qwen"

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.calls += 1

        class R:
            status_code = 401

        class AuthError(Exception):
            response = R()

        raise AuthError("expired credential")
        yield  # pragma: no cover — makes this an async generator


class _Scripted(ModelProvider):
    """Round 1 calls *tool*; round 2 answers."""

    def __init__(self, tool: str, provider_name: str = "llama_cpp") -> None:
        self.tool = tool
        self.provider_name = provider_name
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.calls += 1
        content = ([ToolUseBlock(id=f"t{self.calls}", name=self.tool, input={})]
                   if self.calls == 1 else [TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=10, output_tokens=4),
            stop_reason="stop",
        )


class _DenyAll:
    def evaluate(self, tool_name, **kw):  # noqa: ANN001
        return PermissionDecision.deny("denied by test")


def _run(tmp_path, *, primary, fallback_provider=None, tool="bash", gate=None) -> sqlite3.Connection:
    registry = ToolRegistry()
    registry.register(_tool(tool))
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(
        provider=primary,
        model=CLOUD,
        system_prompt="- Model: qwen3.8-max (provider: qwen)",
        max_tokens=512,
        tool_registry=registry,
        telemetry=tel,
        permission_checker=gate,
        fallback=(FallbackTarget(model=LOCAL, provider_name="llama_cpp",
                                 provider=fallback_provider, is_local_backend=True)
                  if fallback_provider is not None else None),
    )

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="desktop:fb"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def _tool_rows(con):
    return con.execute("SELECT tool_name, model, is_golden, error_type FROM tool_calls"
                       " WHERE tool_name != '_loop_transition' ORDER BY rowid").fetchall()


def _transitions(con):
    return con.execute("SELECT model, error_type FROM tool_calls"
                       " WHERE tool_name = '_loop_transition' ORDER BY rowid").fetchall()


def test_a_degraded_rounds_call_is_recorded_under_the_fallback_and_is_not_golden(tmp_path):
    con = _run(tmp_path, primary=_ExpiredCloud(), fallback_provider=_Scripted("bash"))

    [row] = _tool_rows(con)
    assert (row["tool_name"], row["model"], row["error_type"]) == ("bash", LOCAL, None), (
        "the local fallback made this call; it was recorded under the cloud model that failed"
    )
    assert row["is_golden"] == 0, "a local model's output must never be filed as a teacher example"


def test_a_degraded_rounds_transition_names_the_fallback(tmp_path):
    con = _run(tmp_path, primary=_ExpiredCloud(), fallback_provider=_Scripted("bash"))
    transitions = _transitions(con)
    assert transitions, "the tool round must leave a transition row"
    assert {t["model"] for t in transitions} == {LOCAL}


def test_a_degraded_rounds_failed_call_names_the_fallback(tmp_path):
    con = _run(tmp_path, primary=_ExpiredCloud(), fallback_provider=_Scripted("bash"), gate=_DenyAll())
    [row] = _tool_rows(con)
    assert (row["model"], row["error_type"]) == (LOCAL, "permission_denied")


def test_every_round_still_asks_the_primary_first(tmp_path):
    """Only what is recorded changes: the second round tries the primary again."""
    primary = _ExpiredCloud()
    con = _run(tmp_path, primary=primary, fallback_provider=_Scripted("bash"))
    assert primary.calls == 2
    rounds = con.execute("SELECT model, outcome FROM subsystem_runs WHERE subsystem='agent_loop'"
                         " AND operation='loop_round' ORDER BY rowid").fetchall()
    assert [(r["model"], r["outcome"]) for r in rounds] == [
        (CLOUD, "failed"), (LOCAL, "success"), (CLOUD, "failed"), (LOCAL, "success")]


def test_a_healthy_cloud_round_is_still_recorded_as_the_cloud_and_golden(tmp_path):
    con = _run(tmp_path, primary=_Scripted("bash", provider_name="qwen"))
    [row] = _tool_rows(con)
    assert (row["model"], row["is_golden"]) == (CLOUD, 1)
    assert {t["model"] for t in _transitions(con)} == {CLOUD}


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)


def test_a_degraded_rounds_lucky_guess_names_the_fallback(tmp_path):
    """The lucky-guess marker (#605) belongs to the round that made the call, so it
    names the model that served that round: the fallback's, when it did."""
    from prometheus.adapter import ModelAdapter
    from prometheus.context.dynamic_tools import DynamicToolLoader

    registry = ToolRegistry()
    for name in ("bash", "read_file", "image_generate"):
        registry.register(_tool(name))
    # Deferred loading on, advertising bash + read_file only: image_generate is
    # registered but not advertised, so the fallback's call to it is a lucky guess.
    loader = DynamicToolLoader(registry, {"enabled": True, "always_loaded": ["bash", "read_file"]})
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(
        provider=_ExpiredCloud(),
        model=CLOUD,
        system_prompt="- Model: qwen3.8-max (provider: qwen)",
        max_tokens=512,
        tool_registry=registry,
        tool_loader=loader,
        adapter=ModelAdapter(tier="light"),
        telemetry=tel,
        fallback=FallbackTarget(model=LOCAL, provider_name="llama_cpp",
                                provider=_Scripted("image_generate"), is_local_backend=True),
    )

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="desktop:fb"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    [marker] = con.execute("SELECT model, session_id FROM subsystem_runs WHERE subsystem='agent_loop'"
                           " AND operation='lucky_guess'").fetchall()
    assert (marker["model"], marker["session_id"]) == (LOCAL, "desktop:fb"), (
        "the fallback served the round that guessed; the marker named the model that failed"
    )
    [row] = _tool_rows(con)
    assert (row["tool_name"], row["model"]) == ("image_generate", LOCAL), "and so does the call's own row"
