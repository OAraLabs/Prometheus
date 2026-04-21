"""Tests for Sprint 15b GRAFT: context compression upgrade (Tier 2)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from prometheus.context.budget import TokenBudget
from prometheus.context.compression import ContextCompressor
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
)
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.engine.usage import UsageSnapshot


class SummarizerProvider(ModelProvider):
    """Mock provider that returns a canned summary."""

    def __init__(self, summary: str = "Summary of older conversation.") -> None:
        self._summary = summary

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        msg = ConversationMessage(role="assistant", content=[TextBlock(text=self._summary)])
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=10, output_tokens=10),
            stop_reason="stop",
        )


def _make_messages(n: int) -> list[ConversationMessage]:
    """Generate n user/assistant turn pairs."""
    msgs = []
    for i in range(n):
        msgs.append(ConversationMessage.from_user_text(f"User message {i}"))
        msgs.append(ConversationMessage(
            role="assistant",
            content=[TextBlock(text=f"Assistant response {i}")],
        ))
    return msgs


def _make_messages_with_tools(n: int) -> list[ConversationMessage]:
    """Generate n turns with tool results in user messages."""
    msgs = []
    for i in range(n):
        msgs.append(ConversationMessage(
            role="user",
            content=[ToolResultBlock(tool_use_id=f"t{i}", content=f"result data {i}" * 50)],
        ))
        msgs.append(ConversationMessage(
            role="assistant",
            content=[TextBlock(text=f"Processed tool result {i}")],
        ))
    return msgs


class TestTier1Pruning:
    """Existing pruning behavior must be unchanged."""

    def test_no_compression_when_under_budget(self):
        budget = TokenBudget(effective_limit=100000)
        compressor = ContextCompressor(budget, fresh_tail_count=4)
        msgs = _make_messages(5)
        result = compressor.maybe_compress(msgs)
        assert result == msgs  # unchanged

    def test_prunes_old_tool_results(self):
        budget = TokenBudget(effective_limit=100, reserved_output=10)
        budget.add("test", "x" * 400)  # 400 chars ÷ 4 = 100 tokens, well over 75% of 90
        compressor = ContextCompressor(budget, fresh_tail_count=2)
        msgs = _make_messages_with_tools(10)
        result = compressor.maybe_compress(msgs)
        # Old tool results should be pruned
        pruned = [
            m for m in result if m.role == "user"
            and any(
                isinstance(b, ToolResultBlock) and "pruned" in b.content
                for b in m.content
            )
        ]
        assert len(pruned) > 0


class TestTier2Summarization:

    def test_tier2_activates_when_pruning_insufficient(self):
        budget = TokenBudget(effective_limit=100, reserved_output=5)
        budget.add("test", "x" * 400)  # 400 chars ÷ 4 = 100 tokens, well over 90% of 95
        compressor = ContextCompressor(budget, fresh_tail_count=4)

        msgs = _make_messages(20)
        provider = SummarizerProvider("Older conversation summary.")

        result = asyncio.run(compressor.maybe_compress_async(msgs, provider=provider))
        # Should have fewer messages (batches replaced by summaries)
        assert len(result) < len(msgs)
        # At least one summary marker should exist
        summaries = [m for m in result if m.text and "summarized" in m.text.lower()]
        assert len(summaries) >= 1

    def test_tier2_skipped_when_no_provider(self):
        budget = TokenBudget(effective_limit=100, reserved_output=5)
        budget.add("test", "x" * 400)
        compressor = ContextCompressor(budget, fresh_tail_count=4)
        msgs = _make_messages(20)

        result = asyncio.run(compressor.maybe_compress_async(msgs, provider=None))
        # Without provider, only Tier 1 runs — no summarization
        summaries = [m for m in result if m.text and "summarized" in m.text.lower()]
        assert len(summaries) == 0

    def test_recent_messages_preserved(self):
        budget = TokenBudget(effective_limit=100, reserved_output=5)
        budget.add("test", "x" * 400)
        compressor = ContextCompressor(budget, fresh_tail_count=4)

        msgs = _make_messages(20)
        provider = SummarizerProvider("Summary.")

        result = asyncio.run(compressor.maybe_compress_async(msgs, provider=provider))
        # Last 4 user messages should still be intact (not summarized)
        user_msgs = [m for m in result if m.role == "user"]
        last_users = user_msgs[-4:]
        for m in last_users:
            assert "summarized" not in (m.text or "").lower()
