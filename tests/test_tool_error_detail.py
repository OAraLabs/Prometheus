"""Audit fix #4: tool_error telemetry rows must capture the tool's own message.

Before the fix, _execute_tool_call recorded error_type="tool_error" but never
passed error_detail, so every tool failure (e.g. symbiote) was a blank black box.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


class _FailTool(BaseTool):
    name = "failtool"
    description = "always fails"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="boom: the real reason (rate limited)", is_error=True)


class _OkTool(BaseTool):
    name = "oktool"
    description = "always ok"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="all good")


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(self, **kw):  # noqa: ANN003
        self.calls.append(kw)


def _ctx(reg: ToolRegistry, tel: _RecordingTelemetry) -> LoopContext:
    # provider=None and permission_checker=None are fine: the failure path only
    # calls _provider_name_for_telemetry(None) (-> "") and the permission block
    # is None-guarded.
    return LoopContext(
        provider=None,
        model="stub",
        system_prompt="",
        max_tokens=1024,
        tool_registry=reg,
        telemetry=tel,
    )


def test_tool_error_records_error_detail_from_output():
    reg = ToolRegistry()
    reg.register(_FailTool())
    tel = _RecordingTelemetry()
    block = asyncio.run(_execute_tool_call(_ctx(reg, tel), "failtool", "tu1", {}))
    assert block.is_error
    rec = next(c for c in tel.calls if c.get("tool_name") == "failtool")
    assert rec["error_type"] == "tool_error"
    # The whole point of the fix: this was None before.
    assert "the real reason" in (rec.get("error_detail") or "")


def test_successful_tool_has_no_error_detail():
    reg = ToolRegistry()
    reg.register(_OkTool())
    tel = _RecordingTelemetry()
    asyncio.run(_execute_tool_call(_ctx(reg, tel), "oktool", "tu2", {}))
    rec = next(c for c in tel.calls if c.get("tool_name") == "oktool")
    assert rec["error_type"] is None
    assert rec.get("error_detail") is None
