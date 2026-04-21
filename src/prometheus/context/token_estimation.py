"""Token estimation utility for Sprint 4 context management.

Provides a fast, dependency-free approximation of token count.
Rule of thumb: 1 token ≈ 4 characters (works across most English LLM tokenizers).
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in *text*.

    Uses the 4-chars-per-token heuristic — fast and good enough for
    budget tracking; not a substitute for exact tokenizer counts.

    Args:
        text: Input string to estimate.

    Returns:
        Estimated token count (minimum 0).
    """
    if not text:
        return 0
    return max(0, len(text) // 4)
