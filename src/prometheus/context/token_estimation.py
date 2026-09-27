"""Token estimation utility for Sprint 4 context management.

Provides a fast, dependency-free approximation of token count.
Rule of thumb: 1 token ≈ 4 characters (works across most English LLM tokenizers).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from prometheus.engine.messages import ThinkingBlock


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

    Thinking that no request builder sends back is left out: a ThinkingBlock
    without a signature — Ollama, OpenAI-compatible and stub reasoning; only
    Anthropic signs — is skipped by the OpenAI-shape builder
    (``stub._build_openai_messages``) and by the Anthropic one, which returns
    only signed thinking. Counting it budgeted the compactor, the context
    pre-flight and the fallback window for tokens never in the prompt. Signed
    and redacted thinking go back to Anthropic and still count. The message
    itself keeps every block; only the estimate leaves them out.
    """
    try:
        return estimate_tokens(_sent_content_json(msg))
    except Exception:
        return estimate_tokens(getattr(msg, "text", "") or "")


def _never_sent(block: Any) -> bool:
    return isinstance(block, ThinkingBlock) and not block.signature


def _sent_content_json(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if not content or not any(_never_sent(b) for b in content):
        # The same string as always, so the same figure: a message without
        # unsent thinking is estimated exactly as before.
        return str(msg.content_json)
    kept = [b for b in content if not _never_sent(b)]
    # Serialized by content_json itself, so what remains measures exactly
    # as it would without the thought.
    return str(msg.model_copy(update={"content": kept}).content_json)


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
