"""CostTracker — per-model token cost tracking for cloud API providers.

Tracks input/output tokens and calculates costs based on per-model
pricing tables. Reports session and cumulative costs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Pricing per million tokens (input, output) — USD.
#
# Standard tier (non-batch, non-cached, non-fast-mode). For batch discounts
# (50% off both sides on Anthropic) and prompt-caching multipliers (0.1×–2×
# on Anthropic), see the provider's pricing page. The values below are the
# "default" rate a per-call CostTracker.record() should multiply tokens by.
#
# Anthropic entries verified 2026-05-25 against
# docs.anthropic.com/en/docs/about-claude/pricing. Pre-4.6 aliases resolve
# server-side to the dated snapshot; both forms are listed for clarity so
# whichever string the caller passes through resolves to the same price.
#
# Gemini 2.5 Pro: $1.25 / $10 below 200k context, $2.50 / $15 above. The
# entry here is the ≤200k tier; long-context calls will under-bill by 2×
# until tiered pricing is wired. Verified 2026-05-25 against
# ai.google.dev/gemini-api/docs/pricing.
#
# OpenAI and xAI entries pre-date this verification pass. The OpenAI
# (openai.com/api/pricing) and xAI (docs.x.ai/docs/models, x.ai/api)
# pricing pages were unreachable (HTTP 403) or no longer list grok-3,
# so existing values are left intact rather than re-asserted with
# guesswork. See PR #19's "Drive-by findings" for follow-up.
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI (not re-verified in PR #19 — pages 403)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    # Anthropic — verified 2026-05-25 from docs.anthropic.com
    "claude-opus-4-7": (5.00, 25.00),               # current flagship (added PR #19)
    "claude-opus-4-6": (5.00, 25.00),               # PR #19: was (15.00, 75.00) — pricing dropped to match Opus 4.5+
    "claude-opus-4-5": (5.00, 25.00),               # added PR #19
    "claude-opus-4-1-20250805": (15.00, 75.00),     # legacy, still on the table
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),             # alias — added PR #19 (was missing despite being the user's /claude target)
    "claude-sonnet-4-5-20250929": (3.00, 15.00),    # dated snapshot of Sonnet 4.5 — added PR #19
    "claude-sonnet-4-20250514": (3.00, 15.00),     # legacy Sonnet 4 (deprecated June 2026)
    "claude-haiku-4-5": (1.00, 5.00),               # alias — added PR #19
    "claude-haiku-4-5-20251001": (1.00, 5.00),      # PR #19: was (0.80, 4.00) — pricing increased on GA
    # Gemini — verified 2026-05-25 from ai.google.dev (≤200k tier)
    "gemini-2.5-flash": (0.15, 0.60),               # NOTE: ai.google.dev now lists $0.30/$2.50 for text — pricing diverges,
                                                    # left as-is in PR #19 pending a follow-up that handles tiered modality pricing
    "gemini-2.5-pro": (1.25, 10.00),
    # xAI (not re-verified in PR #19 — docs.x.ai no longer lists grok-3)
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    # -- CLOUD EXPANSION (2026-07) — values from the 2026-07-05 web research
    # pass; verify each against the provider's pricing console at first live
    # use (no keys existed on this box when they were entered).
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),      # 2026-07 research, verify at first live use
    "kimi-k2.6": (0.95, 4.00),             # 2026-07 research, verify at first live use
    "glm-5.2": (1.40, 4.40),               # 2026-07 research, verify at first live use
    "mimo-v2.5-pro": (0.435, 0.87),        # 2026-07 research, verify at first live use
}


@dataclass
class UsageRecord:
    """A single token usage entry."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float


class CostTracker:
    """Track token usage and costs across a session."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_input: int = 0
        self._total_output: int = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record a usage event. Returns the cost in USD."""
        pricing = PRICING.get(model)
        if pricing is None:
            # Try prefix match (e.g. "gpt-4o-2024-05-13" -> "gpt-4o")
            for key in PRICING:
                if model.startswith(key):
                    pricing = PRICING[key]
                    break

        if pricing is None:
            cost = 0.0
        else:
            input_price, output_price = pricing
            cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=time.time(),
        )
        self._records.append(record)
        self._total_cost += cost
        self._total_input += input_tokens
        self._total_output += output_tokens
        return cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    @property
    def total_tokens(self) -> int:
        return self._total_input + self._total_output

    def report(self) -> str:
        """Human-readable cost report for /status command."""
        if not self._records:
            return "Cost: $0.00 (no cloud API usage)"

        return (
            f"Session cost: ${self._total_cost:.4f} "
            f"({self._total_input:,} input + {self._total_output:,} output tokens)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Structured cost data."""
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "records": len(self._records),
        }
