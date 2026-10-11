"""OpenRouter spend must read as spend — never as free local traffic.

OpenRouter model ids are ``vendor/model`` (``anthropic/claude-sonnet-5``). ``billing_for`` called any name containing
a slash "served from a local model file", so, with no other change, a person paying OpenRouter would have seen their
usage labelled ``local`` and counted at $0.00, with no warning: the silent zero ``telemetry/cost.py`` is built to
forbid. The telemetry seam already computes the right stamp from the live provider (``billing_stamp_full``); the
in-memory tracker ignored it and re-derived the mode from the name alone.

Pinned here:

* the stamp the seam writes for an OpenRouter call is ``metered`` when the model is priced and ``unknown`` (a gap
  that warns) when it is not — never ``local``;
* a slash in the name still means local on every other host (LM Studio / vLLM ids are ``publisher/model``), and a
  real file path is local wherever it is served from;
* the default model has a price row, taken from the recorded OpenRouter catalog entry;
* the cost tracker uses the stamp the seam computed, so ``cost_is_complete`` is False when OpenRouter traffic is
  unpriced instead of reporting a confident $0.00.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from prometheus.providers.registry import CLOUD_DEFAULTS, ProviderRegistry
from prometheus.telemetry import cost
from prometheus.telemetry.cost import CostTracker, billing_for, billing_stamp_full, price_for

DEFAULT = CLOUD_DEFAULTS["openrouter"]["model"]
CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "provider_key_checks" / "openrouter_default_model.json")
    .read_text(encoding="utf-8")
)["model"]
UNPRICED = "meta-llama/llama-4-maverick"


@pytest.fixture
def openrouter_provider():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-v1-" + "ab" * 32}):
        return ProviderRegistry.create({"provider": "openrouter", "api_key_env": "OPENROUTER_API_KEY"})


@pytest.fixture(autouse=True)
def fresh_unpriced_memory():
    cost._unpriced_seen.discard(UNPRICED)
    yield
    cost.set_cost_tracker_handle(None)


class TestTheStampTheSeamWrites:
    def test_a_priced_openrouter_model_is_metered(self, openrouter_provider):
        assert billing_stamp_full(DEFAULT, openrouter_provider) == ("metered", None)

    def test_an_unpriced_openrouter_model_is_unknown_never_local(self, openrouter_provider):
        mode, marker = billing_stamp_full(UNPRICED, openrouter_provider)
        assert (mode, marker) == ("unknown", None)

    def test_billing_for_with_the_openrouter_host(self):
        assert billing_for(DEFAULT, "//openrouter.ai")[0] == "metered"
        assert billing_for(UNPRICED, "//openrouter.ai")[0] == "unknown"

    def test_a_file_path_is_local_wherever_it_is_served_from(self):
        assert billing_for("/models/qwen.gguf", "//openrouter.ai")[0] == "local"
        assert billing_for("qwen.gguf", "//openrouter.ai")[0] == "local"

    def test_a_slash_is_still_local_on_every_other_host(self):
        """LM Studio and vLLM name models publisher/model; nothing about that changed."""
        assert billing_for("qwen/qwen3-coder-30b", "//localhost:1234")[0] == "local"
        assert billing_for("qwen/qwen3-coder-30b", None)[0] == "local"
        assert billing_for("unsloth/Qwen3.8-27B-GGUF", "//192.168.1.5:8080")[0] == "local"


class TestThePrice:
    def test_the_default_model_is_priced_from_the_recorded_catalog(self):
        per_mtok = (float(CATALOG["pricing"]["prompt"]) * 1e6, float(CATALOG["pricing"]["completion"]) * 1e6)
        assert price_for(DEFAULT) == pytest.approx(per_mtok)

    def test_the_vendor_prefix_does_not_borrow_another_models_price(self):
        """OpenRouter's catalog is the source for its own ids; an unlisted one is unpriced, not guessed."""
        assert price_for(UNPRICED) is None


class TestTheTrackerUsesTheStamp:
    def test_a_metered_call_costs_what_the_catalog_says(self):
        tracker = CostTracker()
        spent = tracker.record(DEFAULT, 1_000_000, 100_000, billing_mode="metered")
        assert spent == pytest.approx(2.00 + 1.00)
        assert tracker.records[-1].billing_mode == "metered"
        assert tracker.cost_is_complete

    def test_an_unpriced_call_is_a_stated_gap_not_a_zero(self, caplog):
        caplog.set_level(logging.WARNING)
        tracker = CostTracker()
        assert tracker.record(UNPRICED, 1000, 100, billing_mode="unknown") == 0.0
        assert tracker.records[-1].billing_mode == "unknown"
        assert not tracker.cost_is_complete and tracker.unpriced_tokens == 1100
        assert any(UNPRICED in r.getMessage() for r in caplog.records), "the gap must be said, once"

    def test_with_no_stamp_it_derives_the_mode_exactly_as_before(self):
        tracker = CostTracker()
        tracker.record("claude-sonnet-5", 1000, 100)
        tracker.record("/models/x.gguf", 1000, 100)
        assert [r.billing_mode for r in tracker.records] == ["metered", "local"]

    def test_the_telemetry_seam_hands_the_tracker_its_stamp(self, tmp_path, openrouter_provider):
        """The consumed layer: record_run is what feeds the tracker in the daemon."""
        from prometheus.telemetry.tracker import ToolCallTelemetry

        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        tracker = CostTracker()
        cost.set_cost_tracker_handle(tracker)
        for model in (DEFAULT, UNPRICED):
            mode, marker = billing_stamp_full(model, openrouter_provider)
            tel.record_run("agent_loop", "llm_call", "ok", 1.0, input_tokens=1000, output_tokens=100,
                           model=model, billing_mode=mode, billing_marker=marker)
        assert [r.billing_mode for r in tracker.records] == ["metered", "unknown"]
        assert tracker.total_cost > 0 and not tracker.cost_is_complete


class TestTheRouterKnowsItIsACloudAPI:
    def test_a_router_entry_naming_openrouter_gets_the_pass_through_tier(self):
        """Not the local Qwen formatter its name-less default would give it (the trap `qwen` documents)."""
        from prometheus.adapter import ModelAdapter
        from prometheus.adapter.formatter import PassthroughFormatter
        from prometheus.router.model_router import _build_adapter_for

        adapter = _build_adapter_for("openrouter")
        assert isinstance(adapter.formatter, PassthroughFormatter)
        assert adapter.tier == ModelAdapter.TIER_OFF
