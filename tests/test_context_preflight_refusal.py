"""#356 — don't send a prompt the local backend cannot serve, and say what
happened when one cannot be sent.

The live failure: a turn lost to a generic provider 400 because the assembled
prompt was 31 tokens over the backend's 32768 window — 0.09% over, the failure
mode of an UNCHECKED limit, not of a genuinely oversized conversation. The
user saw "rejected the request (400)" and lost 26 streamed frames; the daemon
knew the window (llama.cpp publishes n_ctx at /props and the daemon detects it
at boot) and never compared against it.

Pinned here:
  * an over-window turn on a MEASURED (local) backend is refused LOCALLY —
    nothing reaches the provider;
  * the user-facing message names both numbers and says retrying will fail;
  * a measurable telemetry row records the refusal;
  * an UNMEASURED window (cloud, configured floor) never refuses — refusing a
    turn that would have served is the same defect wearing the other face;
  * an under-window turn is unaffected.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


class _QuietProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text="ok")]
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _StubCompactor:
    """Minimal compactor surface: limit_for() (the window) + estimate_total()
    (the size) + apply() passthrough (no compaction available)."""

    def __init__(self, window: int, estimate: int) -> None:
        self._window = window
        self._estimate = estimate
        self.applied = False

    def limit_for(self, model=None, backend=None) -> int:
        return self._window

    def estimate_total(self, system_prompt, messages, tools_chars=0) -> int:
        return self._estimate

    async def apply(self, messages, **kw):
        self.applied = True
        return messages  # compaction could not help — the refusal path's precondition


class _MeasuringFallback:
    """`is_local_backend` is what makes the window MEASURED (vs a configured
    cloud floor)."""

    def __init__(self, local: bool) -> None:
        self.is_local_backend = local


def _ctx(*, window: int, estimate: int, measured: bool, telemetry=None):
    provider = _QuietProvider()
    compactor = _StubCompactor(window, estimate)
    context = LoopContext(
        provider=provider,
        model="Qwen3.8-27B.gguf",
        system_prompt="s",
        max_tokens=128,
        tool_registry=ToolRegistry(),
        compactor=compactor,
        fallback=_MeasuringFallback(measured),
        telemetry=telemetry,
    )
    return context, provider, compactor


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def record_run(self, **kw) -> None:
        self.runs.append(kw)

    def record(self, **kw) -> None:
        pass


async def _drain(ctx):
    finals = []
    async for event, _usage in run_loop(ctx, [ConversationMessage.from_user_text("hi")]):
        from prometheus.engine.agent_loop import AssistantTurnComplete
        if isinstance(event, AssistantTurnComplete):
            finals.append(event.message.text)
    return finals


def test_an_over_window_turn_is_refused_without_reaching_the_provider():
    """31 tokens over, the live shape: nothing is sent, the turn ends with an
    explanation instead of a 400."""
    ctx, provider, _ = _ctx(window=32768, estimate=32799, measured=True)
    finals = asyncio.run(_drain(ctx))

    assert provider.requests == [], (
        "the over-window request was sent to the backend anyway — the whole "
        "point is not discovering the limit from the far side"
    )
    assert len(finals) == 1
    text = finals[0]
    assert "32,799" in text and "32,768" in text, (
        f"the message must name BOTH numbers (actionable), got: {text!r}"
    )
    assert "retry" in text.lower() or "retrying" in text.lower(), (
        "the message must say retrying unchanged fails the same way"
    )


def test_the_refusal_is_recorded_in_telemetry():
    tel = _RecordingTelemetry()
    ctx, _, _ = _ctx(window=32768, estimate=34020, measured=True, telemetry=tel)
    asyncio.run(_drain(ctx))
    rows = [r for r in tel.runs if r.get("operation") == "context_preflight_refusal"]
    assert len(rows) == 1
    s = rows[0]["summary"]
    assert s["estimated_tokens"] == 34020
    assert s["window"] == 32768
    assert s["overshoot"] == 1252
    assert rows[0]["outcome"] == "failed"


def test_an_unmeasured_window_never_refuses():
    """Cloud path: the 'window' is a configured floor, not a published n_ctx.
    Refusing on a guess trades a 400 for a lost turn that would have served."""
    ctx, provider, _ = _ctx(window=32768, estimate=999_999, measured=False)
    finals = asyncio.run(_drain(ctx))

    assert provider.requests, "a cloud-sized turn was refused on an unmeasured window"
    assert finals == ["ok"]


def test_an_under_window_turn_is_unaffected():
    ctx, provider, _ = _ctx(window=32768, estimate=5000, measured=True)
    finals = asyncio.run(_drain(ctx))
    assert len(provider.requests) == 1
    assert finals == ["ok"]


def test_the_refusal_is_loud_in_the_log(caplog):
    with caplog.at_level(logging.ERROR, logger="prometheus.engine.agent_loop"):
        ctx, _, _ = _ctx(window=32768, estimate=32799, measured=True)
        asyncio.run(_drain(ctx))
    assert any(
        "#356" in r.getMessage() and r.levelno == logging.ERROR
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]
