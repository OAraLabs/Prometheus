"""LCM Describe Tool — inspect metadata about a summary node or overall LCM stats.

Provides introspection into the LCM DAG: individual node metadata or
aggregate statistics across the entire memory system.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

# Re-use the same engine wiring as lcm_grep
from prometheus.tools.builtin.lcm_grep import _get_engine


class LCMDescribeInput(BaseModel):
    """Arguments for the LCM describe tool."""

    summary_id: str | None = Field(
        default=None,
        description="ID of a specific summary node to inspect. Omit for overall stats.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID to scope stats to (only used when summary_id is None).",
    )


class LCMDescribeTool(BaseTool):
    """Inspect metadata about an LCM summary node or show overall stats."""

    name = "lcm_describe"
    description = (
        "Show metadata for a specific LCM summary node (depth, parents, "
        "source count, token count, etc.) or overall LCM statistics "
        "(total messages, summaries, max depth, compression ratio)."
    )
    input_model = LCMDescribeInput

    def is_read_only(self, arguments: LCMDescribeInput) -> bool:
        return True

    async def execute(
        self, arguments: LCMDescribeInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            engine = _get_engine()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        # --- Describe a specific summary node ---
        if arguments.summary_id is not None:
            return await self._describe_node(engine, arguments.summary_id)

        # --- Overall LCM stats ---
        return await self._describe_stats(engine, arguments.session_id)

    async def _describe_node(self, engine: object, summary_id: str) -> ToolResult:
        """Return metadata for a single summary node."""
        summary_store = engine.summary_store

        try:
            node = summary_store.get(summary_id)
        except Exception as exc:
            return ToolResult(
                output=f"Failed to retrieve summary {summary_id}: {exc}",
                is_error=True,
            )

        if node is None:
            return ToolResult(
                output=f"Summary node {summary_id!r} not found.",
                is_error=True,
            )

        created_str = time.strftime(
            "%Y-%m-%d %H:%M:%S UTC", time.gmtime(node.created_at)
        )

        lines = [
            f"Summary Node: {node.id}",
            f"  depth:              {node.depth}",
            f"  is_leaf:            {node.is_leaf}",
            f"  parent_ids:         {node.parent_ids}",
            f"  source_message_ids: {len(node.source_message_ids)} message(s)",
            f"  token_count:        {node.token_count}",
            f"  created_at:         {created_str}",
            f"  summary_text:       {node.summary_text[:200]}{'...' if len(node.summary_text) > 200 else ''}",
        ]
        return ToolResult(output="\n".join(lines))

    async def _describe_stats(
        self, engine: object, session_id: str | None
    ) -> ToolResult:
        """Return aggregate LCM statistics."""
        lines: list[str] = ["LCM Statistics"]

        # Message stats from conversation store
        conv_store = engine.conversation_store
        try:
            if hasattr(conv_store, "_conn"):
                conn = conv_store._conn
                if session_id:
                    row = conn.execute(
                        "SELECT COUNT(*) as cnt, SUM(token_count) as tokens "
                        "FROM lcm_messages WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) as cnt, SUM(token_count) as tokens "
                        "FROM lcm_messages",
                    ).fetchone()
                total_messages = row["cnt"] if row else 0
                total_msg_tokens = row["tokens"] or 0 if row else 0
                lines.append(f"  total_messages:     {total_messages}")
                lines.append(f"  message_tokens:     {total_msg_tokens}")
            else:
                lines.append("  total_messages:     (unavailable)")
        except Exception as exc:
            lines.append(f"  total_messages:     (error: {exc})")

        # Summary stats from summary store
        summary_store = engine.summary_store
        try:
            if hasattr(summary_store, "_conn"):
                conn = summary_store._conn
                if session_id:
                    row = conn.execute(
                        "SELECT COUNT(*) as cnt, MAX(depth) as max_d, "
                        "SUM(token_count) as tokens "
                        "FROM lcm_summaries WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) as cnt, MAX(depth) as max_d, "
                        "SUM(token_count) as tokens "
                        "FROM lcm_summaries",
                    ).fetchone()
                total_summaries = row["cnt"] if row else 0
                max_depth = row["max_d"] or 0 if row else 0
                total_sum_tokens = row["tokens"] or 0 if row else 0
                lines.append(f"  total_summaries:    {total_summaries}")
                lines.append(f"  max_depth:          {max_depth}")
                lines.append(f"  summary_tokens:     {total_sum_tokens}")

                # Compression ratio
                if total_msg_tokens and total_sum_tokens:
                    ratio = total_msg_tokens / total_sum_tokens
                    lines.append(f"  compression_ratio:  {ratio:.2f}x")
                else:
                    lines.append("  compression_ratio:  N/A")
            else:
                lines.append("  total_summaries:    (unavailable)")
        except Exception as exc:
            lines.append(f"  total_summaries:    (error: {exc})")

        # Engine-level stats if available
        if hasattr(engine, "stats"):
            stats = engine.stats
            lines.append(f"  total_compactions:  {stats.total_compactions}")
            if stats.last_compaction_at:
                last_str = time.strftime(
                    "%Y-%m-%d %H:%M:%S UTC",
                    time.gmtime(stats.last_compaction_at),
                )
                lines.append(f"  last_compaction:    {last_str}")

        if session_id:
            lines.insert(1, f"  session:            {session_id}")

        return ToolResult(output="\n".join(lines))
