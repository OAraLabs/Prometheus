"""Layer b — honest async-promise validator (engine/honesty.py).

Side-effect tests on the pure helper + the call-site wrapper: true positive,
true negative (qualifying task), conservative false-positive guards, the
provenance gate, and the telemetry side effect on detection.
"""

from __future__ import annotations

from typing import Any

from prometheus.engine.honesty import (
    CORRECTION,
    check_async_promise,
    evaluate_and_record,
)
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock


def _assistant_with_task(on_complete: str = "notify") -> ConversationMessage:
    """An assistant turn that called task_create with the given on_complete."""
    return ConversationMessage(
        role="assistant",
        content=[
            TextBlock(text="Starting that now."),
            ToolUseBlock(
                name="task_create",
                input={
                    "type": "local_bash",
                    "command": "sleep 30 && touch /tmp/x.done",
                    "description": "demo",
                    "on_complete": on_complete,
                },
            ),
        ],
    )


_PROMISE = "Sure — I'll let you know when it's done."


# ── True positive ───────────────────────────────────────────────────────


def test_true_positive_promise_without_task_returns_correction():
    out = check_async_promise(_PROMISE, [])
    assert out == CORRECTION


def test_true_positive_variants():
    for text in (
        "I'll update you once the build finishes.",
        "I will notify you after the run completes.",
        "I'll ping you when the download is finished.",
        "Okay, I'll tell you once it's ready.",
    ):
        assert check_async_promise(text, []) == CORRECTION, text


# ── True negative: a qualifying task was registered this turn ────────────


def test_true_negative_promise_with_qualifying_task_returns_none():
    new_messages = [_assistant_with_task(on_complete="notify")]
    assert check_async_promise(_PROMISE, new_messages) is None


def test_true_negative_reengage_and_both_also_qualify():
    assert check_async_promise(_PROMISE, [_assistant_with_task("reengage")]) is None
    assert check_async_promise(_PROMISE, [_assistant_with_task("both")]) is None


# ── Conservative false-positive guards ──────────────────────────────────


def test_false_positive_guards_return_none():
    for text in (
        "I'll let you know if I find anything.",          # conditional, not completion
        "Let me know if you want X.",                      # reversed direction
        "I'll keep you posted as I work through this.",     # within-turn progress
        "I'll let you know the result.",                   # no completion connector
        "Done — here's the summary.",                      # no promise at all
    ):
        assert check_async_promise(text, []) is None, text


# ── Provenance gate (call-site wrapper) ─────────────────────────────────


def test_provenance_non_user_is_not_corrected():
    # A task_supervisor re-engagement turn must never be corrected.
    out = evaluate_and_record(
        _PROMISE, [], provenance="task_supervisor", session_id="telegram:1",
    )
    assert out is None


def test_provenance_user_promise_without_task_is_corrected():
    out = evaluate_and_record(
        _PROMISE, [], provenance="user", session_id="telegram:1",
    )
    assert out == CORRECTION


# ── Telemetry side effect ───────────────────────────────────────────────


class _SpyTelemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record_signal_event(self, signal_type, payload, source_subsystem, **kw):  # noqa: ANN001
        self.events.append(
            {"signal_type": signal_type, "payload": payload, "source": source_subsystem}
        )
        return 1


def test_true_positive_records_telemetry_event():
    spy = _SpyTelemetry()
    out = evaluate_and_record(
        _PROMISE, [], provenance="user", session_id="telegram:99", telemetry=spy,
    )
    assert out == CORRECTION
    assert len(spy.events) == 1
    ev = spy.events[0]
    assert ev["signal_type"] == "async_promise_uncovered"
    assert ev["source"] == "honesty_validator"
    assert ev["payload"]["session_id"] == "telegram:99"


def test_no_detection_records_no_telemetry():
    spy = _SpyTelemetry()
    # qualifying task → no correction, no event
    out = evaluate_and_record(
        _PROMISE,
        [_assistant_with_task("notify")],
        provenance="user",
        session_id="telegram:1",
        telemetry=spy,
    )
    assert out is None
    assert spy.events == []
