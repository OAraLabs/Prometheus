"""Per-tool latency averages must exclude rows that were never measured.

`latency_ms` is `REAL NOT NULL DEFAULT 0.0`, so "nobody measured this" and "measured as zero" are
the same stored value. Real execution is timed with `time.monotonic()`, which does not return an
exact 0.0 — so 0.0 is always a placeholder written by the record sites for calls that never ran.

The dashboard already filtered two causes by name (`_loop_transition`, POLICY_ERROR_TYPES) — and
that enumeration had fallen out of sync: `validation_failed`, `unknown_tool` and
`input_validation` all mean "never ran" and none were listed, so they were averaged in as
zero-duration executions. Measured on the live 16,370-row table: lcm_grep 0.64 -> 0.85ms.

The fix is structural rather than another name on the list: NULLIF drops every unmeasured row
without anyone maintaining a taxonomy of failure modes.
"""

from __future__ import annotations

import time

import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry


@pytest.fixture
def tel(tmp_path):
    """A real ToolCallTelemetry writing a real DB — not a stub.

    Deliberate: the bug lives in the interaction between what `record()` STORES for an unmeasured
    call (0.0, via its default) and what the dashboard's SQL then AVERAGES. A double for either
    side could not exhibit it.
    """
    return ToolCallTelemetry(db_path=str(tmp_path / "t.db"))


def _avg(tel):
    from prometheus.telemetry.dashboard import ToolDashboard

    dash = ToolDashboard(db_path=tel.db_path)
    try:
        return dash.get_stats(hours=24)["avg_latency_by_tool"]
    finally:
        dash.close()


def test_a_never_ran_row_does_not_drag_the_average(tel):
    """The regression, in the shape the live data actually had."""
    tel.record(model="m", tool_name="grep", success=True, latency_ms=1.0)
    tel.record(model="m", tool_name="grep", success=True, latency_ms=3.0)
    # never executed — validation rejected it, so record() got no latency at all
    tel.record(model="m", tool_name="grep", success=False, error_type="validation_failed")

    avg = _avg(tel)
    assert avg["grep"] == pytest.approx(2.0), (
        f"expected the mean of the two MEASURED calls, got {avg['grep']} "
        "— the unmeasured row was averaged in as a zero-duration execution"
    )


@pytest.mark.parametrize("error_type", ["validation_failed", "unknown_tool", "input_validation"])
def test_the_failure_modes_the_enumerated_list_MISSED(tel, error_type):
    """Each of these means "never ran" and none were in POLICY_ERROR_TYPES."""
    tel.record(model="m", tool_name="edit_file", success=True, latency_ms=4.0)
    tel.record(model="m", tool_name="edit_file", success=False, error_type=error_type)
    avg = _avg(tel)
    assert avg["edit_file"] == pytest.approx(4.0), f"{error_type} dragged the average"


def test_a_tool_with_NO_measured_rows_is_omitted_not_reported_as_zero(tel):
    """"We never timed this" is not "this is instant"."""
    tel.record(model="m", tool_name="ghost", success=False, error_type="unknown_tool")
    avg = _avg(tel)
    assert "ghost" not in avg, f"a tool with no measurement reported {avg.get('ghost')!r}"


def test_measured_values_are_untouched(tel):
    """The guard must not become a way to lose real data."""
    for ms in (0.5, 1.5, 10.0):
        tel.record(model="m", tool_name="read_file", success=True, latency_ms=ms)
    avg = _avg(tel)
    assert avg["read_file"] == pytest.approx(4.0), "sub-ms and whole-ms values all survive"


def test_sub_millisecond_measurements_still_count(tel):
    """The live floor was 0.14ms — these are real and must not be mistaken for placeholders."""
    tel.record(model="m", tool_name="glob", success=True, latency_ms=0.14)
    tel.record(model="m", tool_name="glob", success=True, latency_ms=0.26)
    avg = _avg(tel)
    assert avg["glob"] == pytest.approx(0.2), "a fast call is measured, not a placeholder"


def test_loop_transition_is_still_not_a_row_in_a_per_tool_table(tel):
    """Recorded WITH a latency on purpose — otherwise this test cannot fail.

    Every real `_loop_transition` row is unmeasured, so NULLIF already drops it and removing the
    name filter changes nothing observable. Mutation testing caught that: deleting the filter left
    this test green. Giving the row a latency is the only way to prove the name filter is
    load-bearing rather than decorative — and it is: that pseudo-tool is loop bookkeeping, not a
    tool, so it must not be a row in a per-TOOL latency table even if something starts timing it.
    """
    tel.record(model="m", tool_name="_loop_transition", success=True, latency_ms=5.0)
    tel.record(model="m", tool_name="grep", success=True, latency_ms=2.0)
    avg = _avg(tel)
    assert "_loop_transition" not in avg, "loop bookkeeping is not a tool"
    assert avg["grep"] == pytest.approx(2.0), "and real tools are unaffected"
