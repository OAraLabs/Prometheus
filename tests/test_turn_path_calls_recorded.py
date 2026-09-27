"""Model calls on the turn path leave a usage row (WP-X.21 T11, docs/audits/TELEMETRY-GAPS.md).

LCM summaries, the vision tool and prompt hooks called their provider directly
and wrote no row at all: 832 / 1,503 LCM summaries in the audit's windows spent
tokens on the 4090's one slot that no reader could see. Each now goes through
``LLMCallEnvelope.stream()``, which passes the caller's request through unchanged
and writes one usage row, under the caller's own subsystem name.

Pinned here against the rows actually written, and against the request the
provider received, which must be the one it received before: the bundle this
lands in re-derives its goldens from the committed exchanges, so a changed
request would be refused there anyway.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ApiTextDeltaEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry


class _Answers(ModelProvider):
    """Answers every request with *text*; keeps the requests it received."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiTextDeltaEvent(text=self.text)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self.text)]),
            usage=UsageSnapshot(input_tokens=300, output_tokens=20), stop_reason="stop")


@pytest.fixture
def tel(tmp_path, monkeypatch):
    import prometheus.telemetry.tracker as tracker

    handle = ToolCallTelemetry(tmp_path / "telemetry.db")
    monkeypatch.setattr(tracker, "_telemetry_singleton", handle)
    yield handle
    handle.close()


def _rows(tel, subsystem: str) -> list[tuple]:
    return tel._conn.execute(
        "SELECT operation, outcome, model, input_tokens, output_tokens FROM subsystem_runs"
        " WHERE subsystem = ? ORDER BY rowid", (subsystem,)).fetchall()


def test_an_lcm_summary_leaves_a_usage_row(tel):
    from prometheus.memory.lcm_summarize import LCMSummarizer
    from prometheus.memory.lcm_types import MessagePart

    provider = _Answers("a summary")
    summarizer = LCMSummarizer(provider, model="Qwen3.8-27B")
    assert asyncio.run(summarizer.summarize_messages([MessagePart(role="user", content="hello")]))
    assert _rows(tel, "lcm_summarizer") == [("summarize", "success", "Qwen3.8-27B", 300, 20)]
    [request] = provider.requests
    assert request.system_prompt.startswith("You are a precise summarization engine"), (
        "the summarizer's own request, unchanged")


def test_a_vision_call_leaves_a_usage_row(tel, tmp_path):
    from pathlib import Path

    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.vision import VisionInput, VisionTool

    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    provider = _Answers("a red square")
    result = asyncio.run(VisionTool().execute(
        VisionInput(image_path=str(img)),
        ToolExecutionContext(cwd=Path(tmp_path), metadata={"provider": provider})))
    assert not result.is_error and "red square" in result.output
    assert _rows(tel, "vision") == [("describe_image", "success", "", 300, 20)]
    [request] = provider.requests
    assert request.system_prompt == "You are analyzing an image. Be detailed and accurate."


@pytest.mark.parametrize("kind", ["prompt", "agent"])
def test_a_prompt_hook_leaves_a_usage_row(tel, tmp_path, kind):
    from prometheus.hooks.events import HookEvent
    from prometheus.hooks.executor import HookExecutionContext, HookExecutor
    from prometheus.hooks.registry import HookRegistry
    from prometheus.hooks.schemas import AgentHookDefinition, PromptHookDefinition

    cls = PromptHookDefinition if kind == "prompt" else AgentHookDefinition
    registry = HookRegistry()
    registry.add(HookEvent.PRE_TOOL_USE, cls(prompt="Is this call safe? $ARGUMENTS"))
    provider = _Answers('{"ok": true}')
    executor = HookExecutor(registry, HookExecutionContext(cwd=tmp_path, provider=provider,
                                                           default_model="stub"))
    asyncio.run(executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"}))
    assert _rows(tel, "prompt_hooks") == [(kind, "success", "stub", 300, 20)]
    [request] = provider.requests
    assert request.system_prompt.startswith("You are validating whether a hook condition passes")
