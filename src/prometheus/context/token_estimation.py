"""Token estimation utility for Sprint 4 context management.

Provides a fast, dependency-free approximation of token count.
Rule of thumb: 1 token ≈ 4 characters (works across most English LLM tokenizers).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


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


def estimate_message_tokens(msg: Any) -> int:
    """Estimate tokens for one conversation message.

    ``content_json`` covers every block type (text, tool_use, tool_result);
    ``.text`` does not, and a message whose payload is a tool result would
    otherwise measure as zero.
    """
    try:
        return estimate_tokens(msg.content_json)  # type: ignore[attr-defined]
    except Exception:
        return estimate_tokens(getattr(msg, "text", "") or "")


def estimate_messages(messages: Iterable[Any] | None) -> int:
    """Estimate tokens for a whole conversation.

    One implementation, shared by the compactor (which decides when to act on
    the number) and /context (which reports it). They disagreed before: the
    command measured the system prompt alone and called the result "used", so
    it read the same percentage on turn 1 and turn 100 while the compactor was
    firing every turn on a figure the command never showed.
    """
    if not messages:
        return 0
    return sum(estimate_message_tokens(m) for m in messages)
