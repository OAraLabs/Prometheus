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

from prometheus.router.model_router import OVERRIDE_PRESETS
from prometheus.telemetry.cost import PRICING, CostTracker


# DERIVED, not re-typed (#533). This tuple used to be a hand-maintained copy of
# the slash-command defaults, and it had drifted from the thing it claimed to
# mirror: it named `claude-sonnet-4-5` and `gemini-2.5-pro` as the /claude and
# /gemini defaults when the presets actually said `claude-haiku-4-5-20251001`
# and `gemini-2.5-flash`. A coverage test over a stale copy of the list is
# coverage of the wrong list — it passed while the real defaults went unchecked.
#
# Membership and positivity now live in tests/test_model_catalog_consistency.py,
# which checks them across all nine sites. What stays HERE is the end-to-end
# arithmetic, which that file does not exercise.
SLASH_COMMAND_DEFAULT_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [str(spec["model"]) for spec in OVERRIDE_PRESETS.values()]
        # A documented pin target that is not any preset's default, so it would
        # otherwise go unpriced-checked.
        + ["deepseek-v4-pro"]
    )
)


class TestSlashCommandPricingCoverage:
    @pytest.mark.parametrize("model", SLASH_COMMAND_DEFAULT_MODELS)
    def test_slash_command_pricing_records_nonzero_cost(
        self, model: str
    ) -> None:
        """End-to-end: a real CostTracker.record() call returns >0 USD.

        VALUE-PINNED ON PURPOSE — this is arithmetic, and arithmetic needs a
        known price. The models come from the presets, but the assertion is
        that the lookup and the multiply actually happen, which a shape
        assertion cannot express.
        """
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


# ─────────────────────────────────────────────────────────────────────────────
# THE CLASS FIX: "no price on file" and "free" are different facts.
#
# The rows for a given model are a fix with a shelf life — the next model
# upstream reintroduces the hole on the day it ships. What does not expire is
# the tracker being able to SAY it could not price something. Until it can,
# every unpriced model is indistinguishable from genuinely-free usage at the
# only surfaces a human reads: `/status` (report) and the structured dict.
#
# The vocabulary already exists — billing_for()'s four modes — so this reuses
# it rather than inventing a second, drifting notion of "free".
# ─────────────────────────────────────────────────────────────────────────────


class TestUnpricedIsVisiblyDistinctFromZero:
    """An unpriced model must never be rendered as $0.00 with no qualifier."""

    def test_report_does_not_claim_a_bare_dollar_total_when_unpriced(self) -> None:
        tracker = CostTracker()
        tracker.record("not-a-real-model-zzz", 1_000_000, 500_000)
        report = tracker.report()
        assert "no price on file" in report.lower(), (
            "CostTracker.report() rendered a session containing 1.5M unpriced "
            f"tokens without saying so. Got: {report!r}. A reader of /status "
            "cannot tell this from genuinely-free usage — which is exactly how "
            "qwen3.8-max ran 57% of this box's tokens at $0.00 for months."
        )

    def test_report_names_the_unpriced_model(self) -> None:
        tracker = CostTracker()
        tracker.record("not-a-real-model-zzz", 1000, 500)
        report = tracker.report()
        assert "not-a-real-model-zzz" in report, (
            "report() said tokens were unpriced but not WHICH model, so the "
            f"reader cannot go add the row. Got: {report!r}"
        )

    def test_to_dict_exposes_unpriced_tokens_separately(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("not-a-real-model-zzz", 2000, 1000)
        data = tracker.to_dict()
        assert data.get("unpriced_input_tokens") == 2000, (
            "to_dict() must report unpriced input tokens separately from the "
            f"priced total, so a client can state coverage. Got: {data!r}"
        )
        assert data.get("unpriced_output_tokens") == 1000, (
            f"unpriced output tokens missing or wrong. Got: {data!r}"
        )
        assert "not-a-real-model-zzz" in (data.get("unpriced_models") or []), (
            f"to_dict() must name the unpriced models. Got: {data!r}"
        )

    def test_to_dict_flags_the_total_as_incomplete(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        tracker.record("not-a-real-model-zzz", 2000, 1000)
        data = tracker.to_dict()
        assert data.get("cost_is_complete") is False, (
            "A dollar total that covers only some of the traffic must say so. "
            f"Got: {data!r}"
        )

    def test_a_fully_priced_session_is_complete_and_says_nothing_extra(self) -> None:
        tracker = CostTracker()
        tracker.record("gpt-4o", 1000, 500)
        data = tracker.to_dict()
        assert data.get("cost_is_complete") is True, (
            f"A fully-priced session must report complete coverage. Got: {data!r}"
        )
        assert "no price on file" not in tracker.report().lower(), (
            "A fully-priced session must not carry an unpriced warning — the "
            "warning has to stay rare enough to mean something."
        )

    def test_local_model_zero_is_a_fact_not_a_gap(self) -> None:
        """$0 for a local .gguf is TRUE. It must not be counted as unpriced."""
        tracker = CostTracker()
        tracker.record("/models/qwen3-27b-instruct.gguf", 50_000, 10_000)
        data = tracker.to_dict()
        assert data.get("unpriced_input_tokens") == 0, (
            "A local model served from our own hardware has no bill — that $0 "
            "is a fact, not a missing price row. Counting it as unpriced would "
            f"make the warning meaningless on the local box. Got: {data!r}"
        )
        assert data.get("cost_is_complete") is True, (
            f"A local-only session is fully accounted for. Got: {data!r}"
        )

    def test_unpriced_record_is_marked_on_the_row(self) -> None:
        tracker = CostTracker()
        tracker.record("not-a-real-model-zzz", 1000, 500)
        (record,) = tracker.records
        assert record.billing_mode == "unknown", (
            "The UsageRecord itself must carry WHY its cost is what it is, so "
            "the distinction survives into anything that reads the rows rather "
            f"than the rollup. Got: {record!r}"
        )


class TestPrefixMatchPicksTheLongestKey:
    """Overlapping model families make iteration order a pricing bug.

    The old lookup took the FIRST key in dict order that the name started with.
    That was harmless while the table held no overlapping families; it is not
    harmless now. `glm-5.3` is a prefix of `glm-5.3-flash` at 9x the price, and
    `grok-3` is a prefix of `grok-3-mini` at 10x — so an unrecognised variant of
    either could bill at the wrong rate depending on nothing more than where its
    family landed in the dict.
    """

    def test_longest_prefix_wins_over_a_shorter_family_name(self) -> None:
        from prometheus.telemetry.cost import PRICING, price_for

        # A variant name that is NOT in the table and must fall back by prefix.
        assert price_for("glm-5.3-flash-preview") == PRICING["glm-5.3-flash"], (
            "glm-5.3-flash-preview matched a shorter key — it would bill at "
            "glm-5.3's $1.40/$4.40 instead of glm-5.3-flash's $0.15/$0.50."
        )
        assert price_for("grok-3-mini-fast") == PRICING["grok-3-mini"], (
            "grok-3-mini-fast matched 'grok-3' — a 10x overbill."
        )

    def test_exact_match_still_beats_any_prefix(self) -> None:
        from prometheus.telemetry.cost import PRICING, price_for

        assert price_for("glm-5.3") == PRICING["glm-5.3"]
        assert price_for("grok-3") == PRICING["grok-3"]

    def test_unknown_family_is_still_none(self) -> None:
        from prometheus.telemetry.cost import price_for

        assert price_for("a-model-from-nowhere") is None
