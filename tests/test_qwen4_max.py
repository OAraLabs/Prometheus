"""qwen4-max is a Qwen choice, priced and capped from Alibaba's own pages (read 2026-11-23).

Model page (help/en/model-studio/qwen4-max): id ``qwen4-max``, 1,000,000 context, 131,072 output (262,144 in
thinking mode, which has its own budget), tool calls and image input supported, $1.5 / $4.5 per 1M tokens
international. The Token Plan (Personal) model list carries it too. It is offered, not the default.
"""

from __future__ import annotations

from prometheus.providers.output_caps import MODEL_OUTPUT_LIMITS
from prometheus.router.model_router import OVERRIDE_PRESETS, resolve_model_choices, resolve_model_target
from prometheus.setup_wizard import CLOUD_PROVIDER_MODELS
from prometheus.telemetry.cost import price_for


def test_qwen4_max_is_offered_but_not_the_default():
    choices = resolve_model_choices("qwen", {})
    assert "qwen4-max" in choices
    assert choices[0] == OVERRIDE_PRESETS["qwen"]["model"] == "qwen3.8-max"
    assert resolve_model_target("qwen:qwen4-max", {})["model"] == "qwen4-max"


def test_qwen4_max_is_priced_and_capped():
    assert price_for("qwen4-max") == (1.50, 4.50)
    assert MODEL_OUTPUT_LIMITS["qwen4-max"].maximum == 131072


def test_setup_offers_qwen4_max():
    assert "qwen4-max" in [row[0] for row in CLOUD_PROVIDER_MODELS["qwen"]]
