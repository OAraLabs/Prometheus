"""History must not shrink mid-run — and a non-zero exit is not a failed call.

TWO defects, both found from the EMBERFALL baseline telemetry:

1. **Mid-run history shrink.** Round 4's prompt dropped 3,595 tokens while every
   other round grew. Cause: ``_microcompact_old_results`` rewrites older
   ToolResultBlocks IN PLACE once ``turn >= microcompact_after_turns`` (default
   3). That is intentional context-saving for small local windows — but on a
   cloud provider it is a net loss: it saves a few thousand tokens while
   invalidating the *entire* cached prompt prefix, which then gets re-billed at
   full rate. ~92% of that run's 535k input tokens were cacheable prefix.

2. **Non-zero exit recorded as a failed call.** ``pytest`` exiting 1 on failing
   tests is the tool working correctly, but it was logged identically to "the
   call blew up", dragging bash's success rate to 82% and hiding real failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prometheus.engine.agent_loop import _microcompact_old_results
from prometheus.engine.messages import ConversationMessage, ToolResultBlock


@dataclass
class _Adapter:
    """Stand-in for the strictness adapter. tier 'off' == cloud provider."""

    tier: str


@dataclass
class _Ctx:
    """Minimal LoopContext surface used by _microcompact_old_results."""

    microcompact_after_turns: int = 3
    microcompact_keep_chars: int = 200
    microcompact_keep_chars_no_lcm: int = 500
    lcm_engine: Any = None
    adapter: Any = None
    telemetry: Any = None
    tool_registry: Any = None
    session_id: str = "test"
    model: str = "grok-4.5"
    microcompact_on_cloud: bool = False


def _history(n_results: int = 3, size: int = 4000) -> list[ConversationMessage]:
    """A conversation with several large, already-appended tool results."""
    msgs: list[ConversationMessage] = []
    for i in range(n_results):
        msgs.append(ConversationMessage.from_user_text(f"do step {i}"))
        msgs.append(
            ConversationMessage(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id=f"call-{i}",
                        content=("X" * size),
                        is_error=False,
                    )
                ],
            )
        )
    # A fresh tail so the "recent window" logic has something to protect.
    msgs.append(ConversationMessage.from_user_text("continue"))
    return msgs


def _total_chars(msgs: list[ConversationMessage]) -> int:
    total = 0
    for m in msgs:
        for b in m.content:
            total += len(getattr(b, "content", "") or getattr(b, "text", "") or "")
    return total


# ---------------------------------------------------------------------------
# 1. The shrink
# ---------------------------------------------------------------------------


def test_cloud_run_history_is_append_only():
    """THE REGRESSION: on a cloud provider, a later round must not rewrite
    earlier tool results — that is what silently invalidated the cache prefix
    at round 4 of the EMBERFALL run."""
    ctx = _Ctx(adapter=_Adapter(tier="off"))  # cloud
    msgs = _history()
    before = _total_chars(msgs)

    _microcompact_old_results(ctx, msgs, current_turn=4)

    assert _total_chars(msgs) == before, (
        "history shrank mid-run on a cloud provider — this invalidates the "
        "cached prompt prefix and re-bills the whole context at full rate"
    )
    assert not any(
        "[microcompacted]" in getattr(b, "content", "")
        for m in msgs
        for b in m.content
    )


def test_local_run_still_compacts():
    """Intent preserved: small local windows still get the context savings."""
    ctx = _Ctx(adapter=_Adapter(tier="strict"))  # local model
    msgs = _history()
    before = _total_chars(msgs)

    _microcompact_old_results(ctx, msgs, current_turn=4)

    assert _total_chars(msgs) < before, "local runs should still microcompact"
    assert any(
        "[microcompacted]" in getattr(b, "content", "")
        for m in msgs
        for b in m.content
    )


def test_cloud_opt_in_restores_compaction():
    """The cloud skip is a documented default, not a hard block."""
    ctx = _Ctx(adapter=_Adapter(tier="off"), microcompact_on_cloud=True)
    msgs = _history()
    before = _total_chars(msgs)

    _microcompact_old_results(ctx, msgs, current_turn=4)

    assert _total_chars(msgs) < before


def test_no_adapter_defaults_to_compacting():
    """Unknown provider tier must not silently disable a context safeguard."""
    ctx = _Ctx(adapter=None)
    msgs = _history()
    before = _total_chars(msgs)

    _microcompact_old_results(ctx, msgs, current_turn=4)

    assert _total_chars(msgs) < before


def test_below_turn_threshold_never_touches_history():
    ctx = _Ctx(adapter=_Adapter(tier="strict"))
    msgs = _history()
    before = _total_chars(msgs)
    _microcompact_old_results(ctx, msgs, current_turn=1)
    assert _total_chars(msgs) == before


def test_compaction_is_reported_not_silent():
    """A history rewrite must leave a trace. It previously only log.debug'd, so
    the shrink was invisible until someone diffed token counts by hand."""
    recorded: list[dict] = []

    class _Tel:
        def record_run(self, **kw):
            recorded.append(kw)

    ctx = _Ctx(adapter=_Adapter(tier="strict"), telemetry=_Tel())
    _microcompact_old_results(ctx, _history(), current_turn=4)

    assert recorded, "microcompaction must record telemetry when it fires"
    row = recorded[0]
    assert row.get("operation") == "microcompact"
    summary = row.get("summary") or {}
    assert summary.get("results_compacted", 0) > 0
    assert summary.get("chars_dropped", 0) > 0


def test_no_telemetry_row_when_nothing_compacted():
    recorded: list[dict] = []

    class _Tel:
        def record_run(self, **kw):
            recorded.append(kw)

    ctx = _Ctx(adapter=_Adapter(tier="off"), telemetry=_Tel())  # cloud → skipped
    _microcompact_old_results(ctx, _history(), current_turn=4)
    assert recorded == []


# ---------------------------------------------------------------------------
# 2. non-zero exit != failed call
# ---------------------------------------------------------------------------


def test_nonzero_exit_is_classified_separately():
    from prometheus.engine.agent_loop import _classify_tool_error

    # pytest exiting 1 on failing tests: the tool RAN and did its job.
    assert _classify_tool_error(is_error=True, metadata={"returncode": 1}) == "nonzero_exit"
    assert _classify_tool_error(is_error=True, metadata={"returncode": 127}) == "nonzero_exit"


def test_genuine_call_failure_still_reads_as_tool_error():
    from prometheus.engine.agent_loop import _classify_tool_error

    # No returncode => the call itself failed (exception, bad args, missing tool).
    assert _classify_tool_error(is_error=True, metadata={}) == "tool_error"
    assert _classify_tool_error(is_error=True, metadata=None) == "tool_error"
    # returncode 0 with is_error set is a tool asserting failure on its own terms.
    assert _classify_tool_error(is_error=True, metadata={"returncode": 0}) == "tool_error"


def test_success_has_no_error_type():
    from prometheus.engine.agent_loop import _classify_tool_error

    assert _classify_tool_error(is_error=False, metadata={"returncode": 0}) is None
    assert _classify_tool_error(is_error=False, metadata={}) is None


# ---------------------------------------------------------------------------
# 3. Reporting: a non-zero exit must not drag the success rate
# ---------------------------------------------------------------------------


def test_nonzero_exit_excluded_from_success_rate(tmp_path):
    """The EMBERFALL complaint end-to-end: pytest exiting 1 on failing tests
    reported bash at 82% when every 'failure' was a correct execution."""
    from prometheus.telemetry.tracker import ToolCallTelemetry

    t = ToolCallTelemetry(db_path=tmp_path / "t.db")
    for _ in range(9):
        t.record(model="grok-4.5", tool_name="bash", success=True)
    # Two runs that executed fine but exited non-zero.
    for _ in range(2):
        t.record(model="grok-4.5", tool_name="bash", success=False,
                 error_type="nonzero_exit", error_detail="tests failed")

    rep = t.report()
    bash = rep["tools"]["bash"]
    assert bash["nonzero_exits"] == 2
    assert bash["calls"] == 9, "executed-but-nonzero must leave the denominator"
    assert bash["success_rate"] == 1.0
    assert rep["overall_success_rate"] == 1.0, (
        "a correct pytest failure must not read as a tool-calling failure"
    )


def test_real_tool_failure_still_counts_against_success_rate(tmp_path):
    from prometheus.telemetry.tracker import ToolCallTelemetry

    t = ToolCallTelemetry(db_path=tmp_path / "t.db")
    for _ in range(9):
        t.record(model="grok-4.5", tool_name="bash", success=True)
    t.record(model="grok-4.5", tool_name="bash", success=False,
             error_type="tool_error", error_detail="command not found")

    rep = t.report()
    assert rep["tools"]["bash"]["calls"] == 10
    assert rep["overall_success_rate"] < 1.0, "genuine failures must still show"
