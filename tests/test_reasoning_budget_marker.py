"""Reasoning-budget exhaustion is a telemetry-recorded event (coding-mode pre-step).

The model box's llama-server now runs ``--reasoning-budget 2048
--reasoning-budget-message "[REASONING_BUDGET_EXHAUSTED]"`` — when the budget
runs out the server injects the marker into the reasoning stream and forces
the end-of-thinking tag, after which content generation continues normally.
That makes exhaustion DETECTABLE in-band but invisible to the pre-existing
empty-content handling (content is non-empty on this shape). These tests
drive a full ``stream_message`` fixture turn through a mocked httpx SSE
stream — the call-site wiring is asserted, not just the helper — and assert
the side effect: one ``subsystem_runs`` row, outcome ``"partial"``,
operation ``"reasoning_budget_exhausted"``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
)
from prometheus.providers.llama_cpp import (
    REASONING_BUDGET_MARKER,
    LlamaCppProvider,
)


# --------------------------------------------------------------------------- #
# Mock httpx plumbing — scripted SSE lines through the real _call_once path
# --------------------------------------------------------------------------- #


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}"


def _delta_chunk(**delta: Any) -> str:
    return _sse({"choices": [{"delta": delta, "finish_reason": None}]})


def _final_chunk(finish_reason: str = "stop") -> str:
    return _sse({"choices": [{"delta": {}, "finish_reason": finish_reason}]})


def _usage_chunk(prompt_tokens: int = 50, completion_tokens: int = 20) -> str:
    return _sse({
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    })


class _FakeResponse:
    status_code = 200

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCM:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _FakeResponse:
        return _FakeResponse(self._lines)

    async def __aexit__(self, *exc) -> bool:
        return False


from tests.support.doubles import register_double


@register_double("reasoning_budget._FakeClient", replaces="httpx.AsyncClient (llama.cpp HTTP boundary)")
class _FakeClient:
    lines: list[str] = []

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def stream(self, *a: Any, **k: Any) -> _FakeStreamCM:
        return _FakeStreamCM(type(self).lines)

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _RecordingTelemetry:
    """record_run + record_silent_failure recorder (matches tracker surface)."""

    def __init__(self) -> None:
        self.runs: list[dict] = []
        self.failures: list[dict] = []

    def record_run(self, subsystem, operation, outcome, duration_ms=0.0,
                   summary=None, **usage):  # noqa: ANN001
        self.runs.append({
            "subsystem": subsystem, "operation": operation,
            "outcome": outcome, "summary": summary or {}, **usage,
        })

    def record_silent_failure(self, subsystem, operation, exc, context=None):  # noqa: ANN001
        self.failures.append({
            "subsystem": subsystem, "operation": operation,
            "exc_type": type(exc).__name__, "context": context or {},
        })


def _install(monkeypatch, lines: list[str], tel: _RecordingTelemetry) -> None:
    import httpx

    from prometheus.telemetry import tracker as t

    _FakeClient.lines = lines
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(t, "get_telemetry_handle", lambda: tel)


def _run_turn(suppress: bool | None = False) -> tuple[list, ApiMessageRequest]:
    provider = LlamaCppProvider(base_url="http://unit.test:1")
    request = ApiMessageRequest(
        model="gemma-fixture",
        messages=[ConversationMessage.from_user_text("think hard")],
        suppress_thinking=suppress,
    )

    async def collect():
        return [e async for e in provider.stream_message(request)]

    return asyncio.run(collect()), request


# --------------------------------------------------------------------------- #
# The fixture turn — budget exhausts mid-thought, content still arrives
# --------------------------------------------------------------------------- #


def test_marker_in_reasoning_records_partial_run_row(monkeypatch) -> None:
    tel = _RecordingTelemetry()
    _install(monkeypatch, [
        _delta_chunk(reasoning_content="Let me think step by step... "),
        _delta_chunk(reasoning_content=f"hmm {REASONING_BUDGET_MARKER}"),
        _delta_chunk(content="The answer is 4."),
        _final_chunk("stop"),
        _usage_chunk(),
        "data: [DONE]",
    ], tel)

    events, _req = _run_turn()

    # The call itself is healthy: content delta + complete event with usage.
    complete = [e for e in events if isinstance(e, ApiMessageCompleteEvent)]
    assert len(complete) == 1
    assert complete[0].message.text == "The answer is 4."
    assert complete[0].usage.input_tokens == 50

    # THE side effect: one partial run row tagged to the operation.
    assert len(tel.runs) == 1
    row = tel.runs[0]
    assert row["subsystem"] == "llama_cpp_provider"
    assert row["operation"] == "reasoning_budget_exhausted"
    assert row["outcome"] == "partial"
    assert row["model"] == "gemma-fixture"
    assert row["summary"]["marker"] == REASONING_BUDGET_MARKER
    assert row["summary"]["content_chars"] == len("The answer is 4.")
    assert row["summary"]["finish_reason"] == "stop"
    # Not the empty-content path — no silent failure for a healthy reply.
    assert tel.failures == []


def test_marker_split_across_deltas_still_detected(monkeypatch) -> None:
    # The server streams tokens; the marker can arrive fragmented. Detection
    # runs on the ACCUMULATED reasoning, so fragmentation must not hide it.
    tel = _RecordingTelemetry()
    half = len(REASONING_BUDGET_MARKER) // 2
    _install(monkeypatch, [
        _delta_chunk(reasoning_content="thinking... " + REASONING_BUDGET_MARKER[:half]),
        _delta_chunk(reasoning_content=REASONING_BUDGET_MARKER[half:]),
        _delta_chunk(content="ok"),
        _final_chunk("stop"),
        "data: [DONE]",
    ], tel)

    _run_turn()
    assert [r["operation"] for r in tel.runs] == ["reasoning_budget_exhausted"]


def test_no_marker_no_row(monkeypatch) -> None:
    tel = _RecordingTelemetry()
    _install(monkeypatch, [
        _delta_chunk(reasoning_content="brief thought"),
        _delta_chunk(content="answer"),
        _final_chunk("stop"),
        "data: [DONE]",
    ], tel)

    _run_turn()
    assert tel.runs == []
    assert tel.failures == []


def test_marker_with_empty_content_fires_both_paths(monkeypatch) -> None:
    # Budget exhausted AND the model never produced content afterwards:
    # the partial row fires AND the pre-existing empty-content fallback
    # (reasoning recovery + silent_failures) keeps working unchanged.
    tel = _RecordingTelemetry()
    _install(monkeypatch, [
        _delta_chunk(reasoning_content=f"deep thought {REASONING_BUDGET_MARKER}"),
        _final_chunk("length"),
        "data: [DONE]",
    ], tel)

    events, _ = _run_turn()

    assert [r["operation"] for r in tel.runs] == ["reasoning_budget_exhausted"]
    assert len(tel.failures) == 1  # empty-content path, unchanged
    assert tel.failures[0]["context"]["used_reasoning_fallback"] is True


def test_no_telemetry_handle_is_safe(monkeypatch) -> None:
    import httpx

    from prometheus.telemetry import tracker as t

    _FakeClient.lines = [
        _delta_chunk(reasoning_content=REASONING_BUDGET_MARKER),
        _delta_chunk(content="x"),
        _final_chunk("stop"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(t, "get_telemetry_handle", lambda: None)

    events, _ = _run_turn()
    assert any(isinstance(e, ApiMessageCompleteEvent) for e in events)
