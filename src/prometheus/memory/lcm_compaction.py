"""LCM DAG Compaction Engine.

Implements incremental DAG-based compaction for the Lossless Context Management
system.  Raw conversation messages are batched, summarized into depth-0
:class:`SummaryNode` instances, and cascaded into higher-depth nodes when the
number of leaf summaries at any level exceeds a threshold.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus.context.token_estimation import estimate_tokens
from prometheus.memory.lcm_summarize import LCMCircuitBreakerOpen, LCMSummarizer
from prometheus.memory.lcm_types import (
    CompactionConfig,
    CompactionResult,
    MessagePart,
    SummaryNode,
)

if TYPE_CHECKING:
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_summary_store import LCMSummaryStore
    from prometheus.providers.base import ModelProvider

logger = logging.getLogger(__name__)

# The maximum number of leaf nodes at any single depth before a cascade is
# triggered.  This keeps the summary DAG balanced.
_CASCADE_LEAF_THRESHOLD = 6


class LCMCompactor:
    """DAG-based compaction engine for LCM.

    The compactor operates in two phases:

    1. **Message compaction** -- uncompacted messages (excluding a fresh tail)
       are batched and summarized into depth-0 :class:`SummaryNode` leaves.
    2. **Cascade** -- if the number of leaf nodes at any depth level exceeds
       ``_CASCADE_LEAF_THRESHOLD``, groups are merged into depth+1 nodes,
       up to ``config.max_summary_depth``.
    """

    def __init__(
        self,
        conversation_store: LCMConversationStore,
        summary_store: LCMSummaryStore,
        summarizer: LCMSummarizer,
        config: CompactionConfig,
    ) -> None:
        self._conv_store = conversation_store
        self._sum_store = summary_store
        self._summarizer = summarizer
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compact(self, session_id: str) -> CompactionResult:
        """Run a full compaction pass for *session_id*.

        Steps:
            1. Fetch uncompacted messages, excluding the fresh tail.
            2. Batch them into groups of ``compaction_batch_size``.
            3. Summarize each batch into a depth-0 leaf node.
            4. Mark source messages as compacted.
            5. Cascade if any depth level has too many leaves.

        Returns:
            A :class:`CompactionResult` with statistics about the pass.
        """
        result = CompactionResult()

        # 1. Get compactable messages (everything except the fresh tail).
        all_uncompacted = self._conv_store.get_uncompacted_messages(session_id)
        if len(all_uncompacted) <= self._config.fresh_tail_count:
            logger.debug(
                "Session %s: only %d uncompacted messages (tail=%d), nothing to compact",
                session_id,
                len(all_uncompacted),
                self._config.fresh_tail_count,
            )
            return result

        # The messages to actually compact (everything before the fresh tail).
        # Messages are assumed to be in chronological order (oldest first).
        compactable = all_uncompacted[: -self._config.fresh_tail_count]
        if not compactable:
            return result

        # 2. Batch and summarize.
        batch_size = self._config.compaction_batch_size
        batches = [
            compactable[i : i + batch_size]
            for i in range(0, len(compactable), batch_size)
        ]

        tokens_before = sum(m.token_count or estimate_tokens(m.content) for m in compactable)

        for batch in batches:
            try:
                node = await self._summarize_messages(batch, session_id)
                result.summaries_created += 1
                result.messages_compacted += len(batch)

                # 4. Mark source messages as compacted.
                msg_ids = [m.message_id for m in batch]
                self._conv_store.mark_compacted(msg_ids)

            except LCMCircuitBreakerOpen:
                logger.warning(
                    "Circuit breaker open -- aborting compaction for session %s",
                    session_id,
                )
                break
            except Exception:
                logger.exception(
                    "Failed to summarize batch of %d messages in session %s",
                    len(batch),
                    session_id,
                )
                # Circuit breaker in the summarizer tracks consecutive failures.
                # If we hit 3, the next iteration will trip the breaker above.
                continue

        # 5. Cascade summaries if needed.
        try:
            cascade_count = await self._cascade_summaries(session_id)
            if cascade_count:
                result.summaries_created += cascade_count
        except LCMCircuitBreakerOpen:
            logger.warning("Circuit breaker open during cascade -- skipping")

        # Compute tokens saved.
        tokens_after = sum(
            node.token_count or estimate_tokens(node.summary_text)
            for node in self._sum_store.get_leaf_summaries(session_id)
        )
        result.tokens_saved = max(0, tokens_before - tokens_after)
        result.new_depth = self._sum_store.get_max_depth(session_id)

        return result

    def should_compact(self, session_id: str) -> bool:
        """Return ``True`` if the session has enough uncompacted messages."""
        count = self._conv_store.count_uncompacted(session_id)
        # Compact when we have more uncompacted messages than the fresh tail
        # plus one full batch (so there is actually something to compact).
        return count > self._config.fresh_tail_count + self._config.compaction_batch_size

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _summarize_messages(
        self, messages: list[MessagePart], session_id: str
    ) -> SummaryNode:
        """Create a depth-0 summary node from raw messages."""
        summary_text = await self._summarizer.summarize_messages(messages)

        node = SummaryNode(
            source_message_ids=[m.message_id for m in messages],
            summary_text=summary_text,
            depth=0,
            token_count=estimate_tokens(summary_text),
            is_leaf=True,
        )
        self._sum_store.add_summary(session_id, node)
        return node

    async def _cascade_summaries(self, session_id: str) -> int:
        """Check each depth level and merge when too many leaves exist.

        Returns the number of new summary nodes created by cascading.
        """
        created = 0
        for depth in range(self._config.max_summary_depth):
            leaves = self._sum_store.get_leaf_summaries_at_depth(session_id, depth)
            if len(leaves) < _CASCADE_LEAF_THRESHOLD:
                continue

            # Group leaves into batches for merging.
            batch_size = self._config.compaction_batch_size
            batches = [
                leaves[i : i + batch_size]
                for i in range(0, len(leaves), batch_size)
            ]
            # Only merge if we have a full batch (avoid single-node merges).
            for batch in batches:
                if len(batch) < 2:
                    continue
                new_node = await self._summarize_nodes(batch, depth + 1, session_id)
                created += 1

                # Mark source nodes as non-leaf.
                self._sum_store.mark_non_leaf([n.id for n in batch])

        return created

    async def _summarize_nodes(
        self, nodes: list[SummaryNode], new_depth: int, session_id: str
    ) -> SummaryNode:
        """Create a higher-depth summary node from child nodes."""
        summary_text = await self._summarizer.summarize_summaries(nodes)

        node = SummaryNode(
            parent_ids=[n.id for n in nodes],
            summary_text=summary_text,
            depth=new_depth,
            token_count=estimate_tokens(summary_text),
            is_leaf=True,
        )
        self._sum_store.add_summary(session_id, node)
        return node
