"""CostTracker — per-model token cost tracking for cloud API providers.

Tracks input/output tokens and calculates costs based on per-model
pricing tables. Reports session and cumulative costs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Pricing per million tokens (input, output) — USD
PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-opus-4-6": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    # Gemini
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    # xAI
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
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
