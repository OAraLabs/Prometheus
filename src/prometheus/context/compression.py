"""ContextCompressor — two-tier context compression.

Tier 1 (pruning): strip tool_result content from old messages. Free, no LLM call.
Tier 2 (summarization): summarize old message batches via ModelProvider when
    pruning alone isn't enough. Added in Sprint 15b GRAFT.

Usage:
    compressor = ContextCompressor(budget, fresh_tail_count=32)
    messages = compressor.maybe_compress(messages)
    # or with Tier 2:
    messages = await compressor.maybe_compress_async(messages, provider=provider)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus.context.budget import TokenBudget

if TYPE_CHECKING:
    from prometheus.engine.messages import ConversationMessage
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

_PRUNED_MARKER = "[content pruned — context compression]"
_SUMMARY_MARKER = "[summarized — context compression]"
_SUMMARY_BATCH_SIZE = 8  # messages per summary batch
_SUMMARY_PROMPT = (
    "Summarize the following conversation excerpt in 3-5 sentences. "
    "Capture: what was discussed, what was decided, what tools were called "
    "and their key results. Be concise and factual.\n\n{text}"
)


class ContextCompressor:
    """Prune old tool results when the context budget is approaching its limit.

    Args:
        budget:           TokenBudget to check before deciding to compress.
        fresh_tail_count: Number of most-recent *user* messages to preserve intact.
                          Tool results in older messages are pruned.
    """

    def __init__(
        self,
        budget: TokenBudget,
        fresh_tail_count: int = 32,
    ) -> None:
        self._budget = budget
        self._fresh_tail_count = fresh_tail_count

    @classmethod
    def from_config(
        cls,
        budget: TokenBudget,
        config_path: str | None = None,
    ) -> ContextCompressor:
        """Build from prometheus.yaml context.fresh_tail_count."""
        import yaml
        from pathlib import Path

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            fresh_tail_count = data.get("context", {}).get("fresh_tail_count", 32)
        except (OSError, Exception):
            fresh_tail_count = 32

        return cls(budget=budget, fresh_tail_count=fresh_tail_count)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_compress(
        self,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        """Compress *messages* if budget is approaching limit (Tier 1 only).

        Returns the (possibly pruned) message list.  Modifies nothing in place;
        returns a new list with pruned message objects.
        """
        if not self._budget.is_approaching_limit():
            return messages

        compressed, pruned_count = self._prune_old_tool_results(messages)
        if pruned_count:
            log.info(
                "ContextCompressor: pruned %d tool_result blocks from older turns "
                "(fresh_tail_count=%d, budget_used=%d/%d)",
                pruned_count,
                self._fresh_tail_count,
                self._budget.used,
                self._budget.effective_limit,
            )
        return compressed

    async def maybe_compress_async(
        self,
        messages: list[ConversationMessage],
        provider: ModelProvider | None = None,
    ) -> list[ConversationMessage]:
        """Compress with Tier 1 (pruning) + Tier 2 (summarization) if needed.

        Tier 1 runs first. If the budget is still over threshold after pruning
        and a provider is available, Tier 2 summarizes older message batches.
        """
        # Tier 1: pruning
        compressed = self.maybe_compress(messages)

        # Check if still over budget after pruning
        if not self._budget.is_approaching_limit(threshold=0.90):
            return compressed
        if provider is None:
            return compressed

        # Tier 2: summarization of old messages
        summarized, summary_count = await self._summarize_old_messages(
            compressed, provider
        )
        if summary_count:
            log.info(
                "ContextCompressor Tier 2: summarized %d message batches",
                summary_count,
            )
        return summarized

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self,
        messages: list[ConversationMessage],
    ) -> tuple[list[ConversationMessage], int]:
        """Return (pruned_messages, count_of_pruned_blocks).

        Identifies the most-recent `fresh_tail_count` user-role messages and
        prunes tool_result content from all earlier user messages.
        """
        from prometheus.engine.messages import ConversationMessage, ToolResultBlock

        # Collect indices of user messages (tool results come in user turns)
        user_indices = [i for i, m in enumerate(messages) if m.role == "user"]

        # The last fresh_tail_count user messages are protected
        protected = set(user_indices[-self._fresh_tail_count :])

        pruned_count = 0
        result: list[ConversationMessage] = []

        for idx, msg in enumerate(messages):
            if idx in protected or msg.role != "user":
                result.append(msg)
                continue

            # Check if any content block is a ToolResultBlock
            has_tool_results = any(
                isinstance(block, ToolResultBlock) for block in msg.content
            )
            if not has_tool_results:
                result.append(msg)
                continue

            # Rebuild message with pruned ToolResultBlocks
            new_content = []
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    pruned_count += 1
                    new_content.append(
                        ToolResultBlock(
                            tool_use_id=block.tool_use_id,
                            content=_PRUNED_MARKER,
                            is_error=block.is_error,
                        )
                    )
                else:
                    new_content.append(block)

            result.append(
                ConversationMessage(role=msg.role, content=new_content)
            )

        return result, pruned_count

    async def _summarize_old_messages(
        self,
        messages: list[ConversationMessage],
        provider: ModelProvider,
    ) -> tuple[list[ConversationMessage], int]:
        """Summarize batches of older messages into compact summaries (Tier 2).

        Protected messages (last fresh_tail_count user turns) are never summarized.
        Returns (compressed_messages, number_of_batches_summarized).
        """
        from prometheus.engine.messages import ConversationMessage, TextBlock
        from prometheus.providers.base import (
            ApiMessageCompleteEvent,
            ApiMessageRequest,
        )

        user_indices = [i for i, m in enumerate(messages) if m.role == "user"]
        protected = set(user_indices[-self._fresh_tail_count:])

        # Collect compressible (unprotected) message indices
        compressible = [
            i for i in range(len(messages))
            if i not in protected and messages[i].role != "system"
        ]
        if len(compressible) < _SUMMARY_BATCH_SIZE:
            return messages, 0

        # Batch compressible messages
        batches: list[list[int]] = []
        for start in range(0, len(compressible), _SUMMARY_BATCH_SIZE):
            batch = compressible[start : start + _SUMMARY_BATCH_SIZE]
            if len(batch) >= 2:
                batches.append(batch)

        if not batches:
            return messages, 0

        # Summarize each batch
        summaries: dict[int, str] = {}  # first_index_of_batch -> summary text
        indices_to_remove: set[int] = set()

        for batch_indices in batches:
            text_parts = []
            for idx in batch_indices:
                msg = messages[idx]
                text_parts.append(f"[{msg.role}]: {msg.text or ''}")
            batch_text = "\n".join(text_parts)

            try:
                summary = await self._call_summarize(provider, batch_text)
            except Exception as exc:
                log.debug("Tier 2 summarization failed for batch: %s", exc)
                continue

            summaries[batch_indices[0]] = summary
            indices_to_remove.update(batch_indices[1:])

        if not summaries:
            return messages, 0

        # Rebuild message list
        result: list[ConversationMessage] = []
        for idx, msg in enumerate(messages):
            if idx in indices_to_remove:
                continue
            if idx in summaries:
                result.append(
                    ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text=f"{_SUMMARY_MARKER}\n{summaries[idx]}")],
                    )
                )
            else:
                result.append(msg)

        return result, len(summaries)

    @staticmethod
    async def _call_summarize(provider: ModelProvider, text: str) -> str:
        """Call the provider to summarize a batch of messages."""
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import (
            ApiMessageCompleteEvent,
            ApiMessageRequest,
        )

        request = ApiMessageRequest(
            model="",
            messages=[ConversationMessage.from_user_text(_SUMMARY_PROMPT.format(text=text))],
            system_prompt="You are a concise summarizer.",
            max_tokens=300,
        )

        parts: list[str] = []
        async for event in provider.stream_message(request):
            if isinstance(event, ApiMessageCompleteEvent) and event.message.text:
                parts.append(event.message.text)
            elif hasattr(event, "text"):
                parts.append(event.text)

        return "".join(parts).strip()
