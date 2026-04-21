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
    ) -> TokenBudget:
        """Build a TokenBudget from prometheus.yaml context section.

        Args:
            model: Active model name — used to apply model_overrides.
            config_path: Path to prometheus.yaml; defaults to DEFAULTS_PATH.
        """
        import yaml
        from pathlib import Path

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

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

        # Apply model-specific override
        if model and model in model_overrides:
            effective_limit = model_overrides[model]

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
