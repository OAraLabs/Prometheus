"""LCM Context Assembler.

Builds the message array that gets sent to the LLM by combining summary
preamble nodes with the fresh (uncompacted) tail of the conversation.
Token budgets are enforced by trimming the oldest/deepest summaries first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus.context.token_estimation import estimate_tokens
from prometheus.memory.lcm_types import (
    AssemblyResult,
    CompactionConfig,
    MessagePart,
    SummaryNode,
)

if TYPE_CHECKING:
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.lcm_summary_store import LCMSummaryStore

logger = logging.getLogger(__name__)


class LCMAssembler:
    """Assemble LLM context from the LCM summary DAG and fresh message tail.

    The assembled context has two logical regions:

    1. **Summary preamble** -- a formatted block of leaf summary nodes,
       ordered deepest-first (i.e. highest compression first) so that the
       most condensed context appears at the top.
    2. **Fresh tail** -- the last ``fresh_tail_count`` uncompacted messages,
       presented in chronological order.

    If the combined token count exceeds the *token_budget*, summaries are
    dropped oldest-and-deepest first until the budget is met.
    """

    def __init__(
        self,
        conversation_store: LCMConversationStore,
        summary_store: LCMSummaryStore,
        config: CompactionConfig,
    ) -> None:
        self._conv_store = conversation_store
        self._sum_store = summary_store
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(self, session_id: str, token_budget: int) -> AssemblyResult:
        """Build an :class:`AssemblyResult` for the given session.

        Args:
            session_id: The conversation session to assemble context for.
            token_budget: Maximum number of tokens the assembled context
                should occupy.

        Returns:
            An :class:`AssemblyResult` containing the selected summaries,
            fresh messages, total estimated tokens, and compression ratio.
        """
        # 1. Get fresh tail messages (uncompacted, most recent).
        all_uncompacted = self._conv_store.get_uncompacted_messages(session_id)
        fresh_messages = all_uncompacted[-self._config.fresh_tail_count :]

        # 2. Get leaf summary nodes, ordered deepest-first then by created_at.
        leaf_summaries = self._sum_store.get_leaf_summaries(session_id)
        leaf_summaries.sort(key=lambda s: (-s.depth, s.created_at))

        # 3. Estimate fresh tail tokens (non-negotiable -- always included).
        fresh_tokens = sum(
            m.token_count or estimate_tokens(m.content)
            for m in fresh_messages
        )

        # 4. Fit summaries within remaining budget.
        remaining_budget = max(0, token_budget - fresh_tokens)
        selected_summaries: list[SummaryNode] = []
        summary_tokens = 0

        for node in leaf_summaries:
            node_tokens = node.token_count or estimate_tokens(node.summary_text)
            if summary_tokens + node_tokens > remaining_budget:
                # Drop this summary (oldest/deepest are tried first, so we
                # naturally shed the least-relevant ones).
                logger.debug(
                    "Dropping summary %s (depth=%d, tokens=%d) -- over budget",
                    node.id[:8],
                    node.depth,
                    node_tokens,
                )
                continue
            selected_summaries.append(node)
            summary_tokens += node_tokens

        total_tokens = fresh_tokens + summary_tokens

        # 5. Compute compression ratio.
        total_messages = self._conv_store.count_all(session_id)
        if total_messages > 0 and total_tokens > 0:
            # Rough estimate: what the full history would cost vs what we use.
            full_token_estimate = sum(
                m.token_count or estimate_tokens(m.content)
                for m in self._conv_store.get_all_messages(session_id)
            )
            compression_ratio = (
                full_token_estimate / total_tokens if total_tokens else 1.0
            )
        else:
            compression_ratio = 1.0

        return AssemblyResult(
            summaries=selected_summaries,
            fresh_messages=fresh_messages,
            total_tokens=total_tokens,
            compression_ratio=compression_ratio,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_summary_preamble(self, summaries: list[SummaryNode]) -> str:
        """Format summary nodes as a readable preamble for the model.

        The preamble is a labeled block that the model can use as compressed
        context about earlier parts of the conversation.

        Args:
            summaries: Ordered list of :class:`SummaryNode` to include.

        Returns:
            A formatted string suitable for prepending to the message list.
        """
        if not summaries:
            return ""

        lines = ["[Earlier conversation context (compressed)]", ""]
        for i, node in enumerate(summaries, 1):
            depth_label = f"L{node.depth}" if node.depth > 0 else "base"
            lines.append(f"--- Summary {i} ({depth_label}) ---")
            lines.append(node.summary_text.strip())
            lines.append("")

        lines.append("[End of compressed context]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Convenience wrapper around the shared token estimator."""
        return estimate_tokens(text)
