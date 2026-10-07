"""No retired model is offered or set as a default, and the catalog says what a model cannot do.

Xiaomi retires ``mimo-v2.5-pro`` and ``mimo-v2.5`` at 2026-10-21 10:00 Beijing
time with "no system replacement model": after it, a request naming either one
gets an error. ``mimo-v2.5-pro`` was the /mimo preset's model in five places
(router preset, registry default, the CLI's fast path, the setup wizard, the
shipped config) and both were offered in the picker. /mimo now points at
``mimo-v2.6-pro``, which Xiaomi's model list recommends, at the same price and
with function calling.

OpenAI documents that GPT-6 Astra takes no tool calls over Chat Completions
("tool calling requires Responses"), the API every OpenAI route here speaks.
The picker offers it, so the wizard says so where the model is chosen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Model -> the date its vendor stops serving it.
RETIRED = {
    "mimo-v2.5-pro": "2026-10-21",
    "mimo-v2.5": "2026-10-21",
}


def _offered_and_defaults() -> dict[str, set[str]]:
    """Every model name a user can pick or gets by default, by where it is."""
    from prometheus.cli.init import _CLOUD_FAST_PROVIDERS
    from prometheus.providers.registry import CLOUD_DEFAULTS
    from prometheus.router.model_router import OVERRIDE_PRESETS, PRESET_MODEL_CHOICES
    from prometheus.setup_wizard import CLOUD_PROVIDER_MODELS

    template = yaml.safe_load((ROOT / "config" / "prometheus.yaml.default").read_text())
    return {
        "router presets": {p.get("model", "") for p in OVERRIDE_PRESETS.values()},
        "picker choices": {m for models in PRESET_MODEL_CHOICES.values() for m in models},
        "registry defaults": {d.get("model", "") for d in CLOUD_DEFAULTS.values()},
        "CLI fast path": {model for _env, model, _limit in _CLOUD_FAST_PROVIDERS.values()},
        "setup wizard": {m for models in CLOUD_PROVIDER_MODELS.values() for m, _d, _p in models},
        "shipped config": {(e or {}).get("model", "")
                           for e in (template.get("slash_commands") or {}).values()},
    }


@pytest.mark.parametrize("model", sorted(RETIRED))
def test_a_retired_model_is_neither_offered_nor_a_default(model):
    places = [where for where, names in _offered_and_defaults().items() if model in names]
    assert places == [], f"{model} (retired {RETIRED[model]}) is still in: {places}"


def test_mimo_points_at_a_current_model_with_tool_calls():
    from prometheus.router.model_router import OVERRIDE_PRESETS, PRESET_MODEL_CHOICES

    assert OVERRIDE_PRESETS["mimo"]["model"] == "mimo-v2.6-pro"
    assert PRESET_MODEL_CHOICES["mimo"][0] == "mimo-v2.6-pro"
    registry = yaml.safe_load((ROOT / "config" / "model_registry.yaml").read_text())
    assert registry["models"]["mimo"]["capabilities"]["function_calling"]["supported"] is True


def test_the_wizard_says_gpt_6_astra_cannot_call_tools():
    from prometheus.setup_wizard import CLOUD_PROVIDER_MODELS

    [desc] = [d for name, d, _p in CLOUD_PROVIDER_MODELS["openai"] if name == "gpt-6-astra"]
    assert "no tool calls" in desc
