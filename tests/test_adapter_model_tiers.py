"""adapter.model_tiers: an operator's word for a model's adapter tier (WP-X.28, PR 3).

The tier is decided in a fixed order — override > provider class > registry >
served template > fallback (#583). This is the override: a map in
prometheus.yaml from a substring of the served model's name to
``auto | off | light | full``. It exists for the model the registry does not
list and the template cannot decide, and for the operator who knows better:
a Bonsai pinned to light before its template is trusted, a model pinned to
full to keep the schema-validating path, a parity scenario that must run a
local backend at full.

On main the key does not exist: ``create_adapter`` takes no override and the
template does not document one; the tests that read the map fail there.
"""

from __future__ import annotations

import logging

import pytest

from prometheus import __main__ as m
from prometheus.adapter.formatter import PassthroughFormatter, QwenFormatter
from prometheus.adapter.tier import ToolTemplate
from prometheus.adapter.validator import Strictness

BONSAI = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
ORNITH = "Ornith-1.5-9B-Q4_K_M.gguf"
PRODUCTION = "/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"   # listed: qwen-3 → light
NATIVE = ToolTemplate(native=True, call_format="qwen-xml", evidence="a template that renders tools")


def _cfg(**tiers):
    return {"model_tiers": tiers}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

class TestMatching:
    def test_a_substring_key_names_the_model_case_insensitively(self):
        assert m._model_tier_override(BONSAI, _cfg(**{"ternary-bonsai-2": "light"})) == ("light", "ternary-bonsai-2")
        assert m._model_tier_override(BONSAI, _cfg(**{"TERNARY-BONSAI": "full"})) == ("full", "TERNARY-BONSAI")
        assert m._model_tier_override(ORNITH, _cfg(ornith="light")) == ("light", "ornith")

    def test_the_longest_matching_key_wins(self):
        cfg = _cfg(**{"qwen": "full", "qwen3.8-27b": "light", "27b": "off"})
        assert m._model_tier_override(PRODUCTION, cfg) == ("light", "qwen3.8-27b")

    def test_auto_decides_nothing_but_names_its_key(self):
        assert m._model_tier_override(BONSAI, _cfg(bonsai="auto")) == (None, "bonsai")

    def test_no_entry_and_no_map_decide_nothing(self):
        assert m._model_tier_override(BONSAI, None) == (None, None)
        assert m._model_tier_override(BONSAI, {}) == (None, None)
        assert m._model_tier_override(BONSAI, _cfg()) == (None, None)
        assert m._model_tier_override(BONSAI, _cfg(gemma="light")) == (None, None)
        assert m._model_tier_override("", _cfg(**{"": "light"})) == (None, None)

    def test_a_map_that_matches_nothing_is_said_once(self, caplog):
        with caplog.at_level(logging.INFO):
            m._model_tier_override(BONSAI, _cfg(gemma="light", ornith="full"))
        lines = [r.getMessage() for r in caplog.records if "adapter.model_tiers" in r.getMessage()]
        assert len(lines) == 1 and "gemma" in lines[0] and "ornith" in lines[0] and "Bonsai" in lines[0]

    def test_a_bad_value_is_refused_with_its_key_named_and_treated_as_absent(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert m._model_tier_override(BONSAI, _cfg(bonsai="medium")) == (None, None)
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and "'bonsai'" in errors[0] and "'medium'" in errors[0] and "refused" in errors[0]
        # A bad entry beside a good one: the good one still decides.
        assert m._model_tier_override(BONSAI, _cfg(bonsai="medium", ternary="light")) == ("light", "ternary")

    def test_a_map_that_is_not_a_map_is_refused_whole(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert m._model_tier_override(BONSAI, {"model_tiers": ["bonsai"]}) == (None, None)
        assert any("must be a map" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The override in the decision: it wins over every other source
# ---------------------------------------------------------------------------

class TestOverrideWins:
    def test_over_the_registry(self):
        adapter = m.create_adapter({"provider": "llama_cpp", "model": PRODUCTION}, _cfg(**{"qwen3.8": "full"}))
        assert adapter.tier == "full"
        assert adapter.validator.strictness == Strictness.MEDIUM
        assert adapter.tier_decision.source == "override"
        assert "adapter.model_tiers names" in adapter.tier_decision.detail

    def test_over_the_served_template(self):
        adapter = m.create_adapter({"provider": "llama_cpp", "model": BONSAI}, _cfg(**{"ternary-bonsai-2": "full"}),
                                   template=NATIVE)
        assert (adapter.tier, adapter.tier_decision.source) == ("full", "override")

    def test_over_the_fallback(self):
        adapter = m.create_adapter({"provider": "llama_cpp", "model": ORNITH}, _cfg(ornith="light"))
        assert (adapter.tier, adapter.tier_decision.source) == ("light", "override")
        assert isinstance(adapter.formatter, QwenFormatter)
        assert adapter.retry.max_retries == 1

    def test_a_cloud_provider_can_be_pinned_too(self):
        adapter = m.create_adapter({"provider": "openai", "model": "gpt-4o"}, _cfg(**{"gpt-4o": "light"}))
        assert (adapter.tier, adapter.tier_decision.source) == ("light", "override")

    def test_auto_is_the_same_as_no_entry(self):
        with_auto = m.create_adapter({"provider": "llama_cpp", "model": PRODUCTION}, _cfg(**{"qwen3.8": "auto"}))
        without = m.create_adapter({"provider": "llama_cpp", "model": PRODUCTION}, {})
        assert (with_auto.tier, with_auto.tier_decision.source) == (without.tier, without.tier_decision.source) == ("light", "registry")

    def test_the_seam_agrees_so_the_record_is_not_forced(self):
        adapter = m.create_adapter({"provider": "llama_cpp", "model": BONSAI}, _cfg(bonsai="light"))
        assert adapter.tier_decision.source == "override"
        assert m._get_adapter_tier("llama_cpp", BONSAI, None, "light") == "light"
        assert m._get_adapter_tier("llama_cpp", BONSAI) == "full"

    def test_a_forcing_seam_still_wins_and_is_recorded_as_forced(self, monkeypatch):
        monkeypatch.setattr(m, "_get_adapter_tier", lambda provider, model, template=None, override=None: "full")
        adapter = m.create_adapter({"provider": "llama_cpp", "model": BONSAI}, _cfg(bonsai="light"))
        assert (adapter.tier, adapter.tier_decision.source) == ("full", "forced")


# ---------------------------------------------------------------------------
# off on a local backend: allowed, and warned about
# ---------------------------------------------------------------------------

class TestOffOnLocal:
    def test_off_on_a_local_backend_warns_and_builds_a_passthrough(self, caplog):
        with caplog.at_level(logging.WARNING):
            adapter = m.create_adapter({"provider": "llama_cpp", "model": BONSAI}, _cfg(bonsai="off"))
        assert adapter.tier == "off"
        assert isinstance(adapter.formatter, PassthroughFormatter)
        assert adapter.retry.max_retries == 0
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "LOCAL backend" in r.getMessage()]
        assert warned and "'bonsai'" in warned[0] and "lost" in warned[0]

    def test_off_on_a_cloud_provider_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            m.create_adapter({"provider": "openai", "model": "gpt-4o"}, _cfg(**{"gpt-4o": "off"}))
        assert not [r for r in caplog.records if "LOCAL backend" in r.getMessage()]


# ---------------------------------------------------------------------------
# The key is documented where the code reads it
# ---------------------------------------------------------------------------

def test_the_template_documents_the_key_as_an_open_map():
    import yaml
    from pathlib import Path

    tmpl = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml.default").read_text())
    assert tmpl["adapter"]["model_tiers"] == {}


@pytest.mark.parametrize("value", ["off", "light", "full"])
def test_every_documented_value_is_accepted(value):
    tier, key = m._model_tier_override(BONSAI, _cfg(bonsai=value))
    assert (tier, key) == (value, "bonsai")
