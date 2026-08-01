"""CostTracker wiring (audit): the telemetry usage seam feeds the cost handle.

The tracker used to be instantiated + reported on (telegram /status) but never
``.record()``'d, so cost always read $0. Now ``record_run`` feeds the
process-wide handle from the one seam all LLM-usage rows flow through. These pin:
a registered handle accumulates, no handle is a clean no-op, unknown/local models
cost 0 but still count tokens, and non-LLM rows never create a cost record.
"""

from __future__ import annotations

import pytest

from prometheus.telemetry.cost import (
    CostTracker,
    get_cost_tracker_handle,
    set_cost_tracker_handle,
)
from prometheus.telemetry.tracker import ToolCallTelemetry


@pytest.fixture
def tele(tmp_path):
    t = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    yield t
    set_cost_tracker_handle(None)  # never leak the handle across tests


def test_registered_handle_accumulates_cloud_cost(tele):
    ct = CostTracker()
    set_cost_tracker_handle(ct)
    tele.record_run("agent_loop", "round", "success",
                    input_tokens=1000, output_tokens=500, model="claude-sonnet-4-6")
    assert ct.total_input_tokens == 1000
    assert ct.total_output_tokens == 500
    assert ct.total_cost > 0  # priced cloud model


def test_no_handle_is_a_noop(tele):
    set_cost_tracker_handle(None)
    tele.record_run("agent_loop", "round", "success",
                    input_tokens=10, output_tokens=5, model="claude-sonnet-4-6")
    assert get_cost_tracker_handle() is None  # no crash, nothing registered


def test_local_model_counts_tokens_at_zero_cost(tele):
    ct = CostTracker()
    set_cost_tracker_handle(ct)
    tele.record_run("agent_loop", "round", "success",
                    input_tokens=2000, output_tokens=800, model="gemma4-26b")
    assert ct.total_tokens == 2800
    assert ct.total_cost == 0.0  # unknown/local model → no price, no crash


def test_rows_without_tokens_dont_feed(tele):
    ct = CostTracker()
    set_cost_tracker_handle(ct)
    tele.record_run("curator", "pass", "success")  # a non-LLM subsystem cycle
    assert ct.total_tokens == 0
    assert len(ct._records) == 0
