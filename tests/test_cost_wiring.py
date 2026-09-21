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


def test_unpriced_model_counts_tokens_but_is_not_called_free(tele):
    """#533: this was `test_local_model_counts_tokens_at_zero_cost`, and the
    comment said "unknown/local model" — conflating the two facts the billing
    vocabulary exists to separate. `gemma4-26b` is a bare name with no path and
    no .gguf, so the classifier calls it UNKNOWN, not local: we cannot price it,
    which is different from it being free. The tokens must still be counted (a
    dropped row is worse than an unpriced one) and the gap must be visible."""
    ct = CostTracker()
    set_cost_tracker_handle(ct)
    tele.record_run("agent_loop", "round", "success",
                    input_tokens=2000, output_tokens=800, model="gemma4-26b")
    assert ct.total_tokens == 2800
    assert ct.total_cost == 0.0          # no price → no dollars invented
    assert ct.cost_is_complete is False  # ...and the total says it is partial
    assert "gemma4-26b" in ct.to_dict()["unpriced_models"]


def test_a_real_local_path_is_free_not_unpriced(tele):
    """The contrast case. A model served from a file on our own disk has no
    bill — that $0 is a FACT, and folding it into the unpriced warning would
    make the warning fire on every local box and stop meaning anything."""
    ct = CostTracker()
    set_cost_tracker_handle(ct)
    tele.record_run("agent_loop", "round", "success", input_tokens=2000,
                    output_tokens=800, model="/models/gemma4-26b-Q4_K_M.gguf")
    assert ct.total_tokens == 2800
    assert ct.total_cost == 0.0
    assert ct.cost_is_complete is True
    assert ct.to_dict()["unpriced_models"] == []


def test_rows_without_tokens_dont_feed(tele):
    ct = CostTracker()
    set_cost_tracker_handle(ct)
    tele.record_run("curator", "pass", "success")  # a non-LLM subsystem cycle
    assert ct.total_tokens == 0
    assert len(ct._records) == 0
