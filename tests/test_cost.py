"""Tests for the per-model PRICING table in telemetry/cost.py.

The primary intent (PR #19) is to catch the failure mode that motivated
this PR: a future ``/<command>`` slash-handler PR adds a new model to the
``slash_commands`` config or to ``OVERRIDE_PRESETS``, but doesn't add the
corresponding row to PRICING, and CostTracker.record() silently prices it
at $0.

By naming every slash-command target as a required PRICING entry here,
that omission becomes a test failure instead of a silent telemetry hole.

PR #18 default slash_commands models (the four checked below) are pulled
from ``config/prometheus.yaml.default`` and from
``OVERRIDE_PRESETS`` in ``src/prometheus/router/model_router.py``.
"""

from __future__ import annotations

import pytest

from prometheus.telemetry.cost import PRICING, CostTracker


# Models that the project's default slash-command config will hit. Each
# must have a non-zero entry in PRICING. Adding a new /<command> with a
# new default model means adding both an OVERRIDE_PRESETS entry AND a
# row here.
SLASH_COMMAND_DEFAULT_MODELS: tuple[str, ...] = (
    "claude-sonnet-4-5",   # /claude default per PR #18
    "gpt-4o",              # /gpt default
    "gemini-2.5-pro",      # /gemini default
    "grok-3",              # /xai default
    # CLOUD EXPANSION (2026-07) — pricing entered from the 2026-07-05
    # research pass, verify at first live use
    "deepseek-v4-flash",   # /deepseek default
    "deepseek-v4-pro",     # DeepSeek reasoning flagship (documented pin target)
    "kimi-k2.6",           # /kimi default
    "glm-5.2",             # /glm default
    "mimo-v2.5-pro",       # /mimo default
)


class TestSlashCommandPricingCoverage:
    @pytest.mark.parametrize("model", SLASH_COMMAND_DEFAULT_MODELS)
    def test_slash_command_model_has_pricing(self, model: str) -> None:
        """Every PR #18 slash-command default must have a PRICING entry."""
        assert model in PRICING, (
            f"{model} is a slash-command default but has no entry in "
            f"prometheus.telemetry.cost.PRICING — CostTracker.record() will "
            f"price it at $0. Add the (input, output) tuple to PRICING and "
            f"verify against the provider's current pricing page."
        )

    @pytest.mark.parametrize("model", SLASH_COMMAND_DEFAULT_MODELS)
    def test_slash_command_pricing_is_nonzero(self, model: str) -> None:
        """A $0 entry would defeat the purpose — pricing must be positive."""
        input_price, output_price = PRICING[model]
        assert input_price > 0, f"{model} has $0 input pricing — looks unset"
        assert output_price > 0, f"{model} has $0 output pricing — looks unset"

    @pytest.mark.parametrize("model", SLASH_COMMAND_DEFAULT_MODELS)
    def test_slash_command_pricing_records_nonzero_cost(
        self, model: str
    ) -> None:
        """End-to-end: a real CostTracker.record() call returns >0 USD."""
        tracker = CostTracker()
        cost = tracker.record(model, input_tokens=1000, output_tokens=500)
        assert cost > 0, (
            f"CostTracker.record({model!r}, ...) returned $0 — PRICING "
            f"entry exists but the lookup or math is broken."
        )


class TestSonnet45AliasAndSnapshotMatch:
    """Anthropic aliases like claude-sonnet-4-5 resolve to a dated snapshot
    server-side. Both forms should price identically — if they diverge, a
    user editing slash_commands with one form vs the other would see
    different costs for the same calls.
    """

    def test_sonnet_4_5_alias_and_snapshot_price_identically(self) -> None:
        alias = PRICING["claude-sonnet-4-5"]
        snapshot = PRICING["claude-sonnet-4-5-20250929"]
        assert alias == snapshot, (
            f"claude-sonnet-4-5 ({alias}) and claude-sonnet-4-5-20250929 "
            f"({snapshot}) should price identically — they're the same model."
        )

    def test_haiku_4_5_alias_and_snapshot_price_identically(self) -> None:
        alias = PRICING["claude-haiku-4-5"]
        snapshot = PRICING["claude-haiku-4-5-20251001"]
        assert alias == snapshot, (
            f"claude-haiku-4-5 ({alias}) and claude-haiku-4-5-20251001 "
            f"({snapshot}) should price identically — they're the same model."
        )


class TestCostTrackerHandlesUnknownModel:
    """CostTracker.record() should fail gracefully (cost=0) for unknown
    models rather than raise — this is the silent-degradation we want
    PRICING coverage to prevent for known models."""

    def test_unknown_model_returns_zero(self) -> None:
        tracker = CostTracker()
        cost = tracker.record("not-a-real-model-zzz", 1000, 500)
        assert cost == 0.0
