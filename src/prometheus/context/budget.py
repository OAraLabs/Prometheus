"""TokenBudget — context window tracking for Sprint 4.

Tracks estimated token usage by category (system_prompt, messages, tool_results)
and signals when the context is approaching its limit.

Usage:
    budget = TokenBudget.from_config(model="qwen3.5-32b")
    budget.add("system", system_prompt)
    budget.add("messages", message_text)
    if budget.is_approaching_limit():
        # trigger ContextCompressor
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prometheus.context.token_estimation import estimate_tokens


@dataclass
class TokenBudget:
    """Tracks estimated token usage across context categories.

    Args:
        effective_limit:  Total token budget for this session (model context window).
        reserved_output:  Tokens reserved for model output (subtracted from headroom).
        model_overrides:  Per-model effective_limit overrides (from prometheus.yaml).
    """

    effective_limit: int
    reserved_output: int = 2000
    model_overrides: dict[str, int] = field(default_factory=dict)

    # Internal usage tracking by category
    _usage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._usage = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: str | None = None,
        config_path: str | None = None,
        *,
        local_model: str | None = None,
        detected_limit: int | None = None,
    ) -> TokenBudget:
        """Build a TokenBudget from prometheus.yaml context section.

        Args:
            model: Active model name — the one serving this session.
            config_path: Path to prometheus.yaml; omit and it resolves via
                ``config.defaults.resolve_config_path()`` (the same search
                order the CLI and daemon use).
            local_model: The model the local inference server has loaded.
                Needed to tell "this session is on the local model" from "this
                session was routed to a cloud provider", which get different
                budgets. Omit and resolution falls back to exact-match only.
            detected_limit: Context size the local server reported. Used only
                when *model* is the local one.

        This mirrors :meth:`ContextCompactor.limit_for` deliberately — a
        reported budget that disagrees with the enforced one is worse than no
        report, because it is consulted precisely when something looks wrong.
        """
        import yaml
        from pathlib import Path

        if config_path is None:
            from prometheus.config.defaults import resolve_config_path
            config_path = str(resolve_config_path())

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            ctx = data.get("context", {})
        except (OSError, Exception):
            ctx = {}

        effective_limit = ctx.get("effective_limit", 24000)
        reserved_output = ctx.get("reserved_output", 2000)
        model_overrides: dict[str, int] = {}
        for m, overrides in (ctx.get("model_overrides") or {}).items():
            if isinstance(overrides, dict) and "effective_limit" in overrides:
                model_overrides[m] = overrides["effective_limit"]

        # Resolution must MATCH ContextCompactor.limit_for(), or the number
        # reported to the operator is not the number in force. Before this,
        # exact-match was the only rule, so `/context` answered 72000 on a
        # local session actually budgeted 32768 (detected) and on a cloud
        # session actually budgeted 1000000 — wrong in both directions, and
        # wrong precisely where someone would look to check.
        #
        # Precedence, most specific first:
        #   1. explicit per-model override — an operator said so
        #   2. detected local window, when this is the local model
        #   3. cloud default, for any other model (a per-session override)
        #   4. the configured global
        if model and model in model_overrides:
            effective_limit = model_overrides[model]
        elif model and local_model and model == local_model and detected_limit:
            effective_limit = detected_limit
        elif model and local_model and model != local_model:
            from prometheus.context.compactor import DEFAULT_CLOUD_LIMIT

            effective_limit = int(
                ctx.get("cloud_default_limit", DEFAULT_CLOUD_LIMIT)
            )

        return cls(
            effective_limit=effective_limit,
            reserved_output=reserved_output,
            model_overrides=model_overrides,
        )

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, category: str, text: str) -> None:
        """Add estimated tokens for *text* under *category*.

        Common categories: "system", "messages", "tool_results".
        Categories are cumulative — call add() each time new content arrives.
        """
        self._usage[category] = self._usage.get(category, 0) + estimate_tokens(text)

    def reset(self) -> None:
        """Clear all tracked usage."""
        self._usage.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def used(self) -> int:
        """Total estimated tokens used across all categories."""
        return sum(self._usage.values())

    def usage_by_category(self) -> dict[str, int]:
        """Return a copy of the per-category usage dict."""
        return dict(self._usage)

    def headroom(self) -> int:
        """Tokens available before hitting the limit (after reserving output space)."""
        available = self.effective_limit - self.reserved_output
        return max(0, available - self.used)

    def is_approaching_limit(self, threshold: float = 0.75) -> bool:
        """Return True when usage has consumed *threshold* of the available budget.

        Args:
            threshold: Fraction of (effective_limit - reserved_output) at which
                       to trigger compression. Default 0.75 (75%).
        """
        available = self.effective_limit - self.reserved_output
        if available <= 0:
            return True
        return self.used >= available * threshold
