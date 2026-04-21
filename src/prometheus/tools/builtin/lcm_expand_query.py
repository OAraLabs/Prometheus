"""LCM Expand Query Tool — answer questions by expanding compressed history.

Given a natural-language question, searches LCM summaries, expands the most
relevant nodes back to source messages, and returns the expanded context so
the agent can answer the question with full fidelity.

This is the 4th LCM tool (alongside lcm_grep, lcm_describe, lcm_expand).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult
from prometheus.tools.builtin.lcm_grep import _get_engine


class LCMExpandQueryInput(BaseModel):
    """Arguments for the LCM expand query tool."""

    query: str = Field(
        description=(
            "Natural-language question to answer from compressed history. "
            "Example: 'What did we decide about the database schema?'"
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Restrict search to a specific session (omit for all sessions)",
    )
    max_expansions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of summary nodes to expand",
    )
    max_messages_per_expansion: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum source messages to include per expanded node",
    )


class LCMExpandQueryTool(BaseTool):
    """Answer a question by searching and expanding compressed conversation history."""

    name = "lcm_expand_query"
    description = (
        "Search compressed conversation history for context relevant to a question, "
        "then expand the most relevant summary nodes back to their original messages. "
        "Use this when you need to recall details from earlier in the conversation "
        "that may have been compressed by LCM."
    )
    input_model = LCMExpandQueryInput

    def is_read_only(self, arguments: LCMExpandQueryInput) -> bool:
        return True

    async def execute(
        self, arguments: LCMExpandQueryInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            engine = _get_engine()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        summary_store = engine.summary_store
        conv_store = engine.conversation_store

        # Step 1: Search summaries for the query
        matching_summaries = []
        try:
            if hasattr(summary_store, "search"):
                matching_summaries = summary_store.search(
                    arguments.query,
                    session_id=arguments.session_id,
                    limit=arguments.max_expansions,
                )
        except Exception as exc:
            return ToolResult(
                output=f"Summary search failed: {exc}", is_error=True
            )

        # Step 2: Also search raw messages for additional context
        matching_messages = []
        try:
            matching_messages = conv_store.search(
                arguments.query,
                session_id=arguments.session_id,
                limit=5,
            )
        except Exception:
            pass  # non-fatal — summaries are the primary source

        if not matching_summaries and not matching_messages:
            return ToolResult(
                output=f"No relevant history found for query: {arguments.query!r}"
            )

        lines: list[str] = [
            f"Expanded context for query: {arguments.query!r}",
            f"Found {len(matching_summaries)} summary node(s) and "
            f"{len(matching_messages)} direct message match(es).",
            "",
        ]

        # Step 3: Expand each matching summary to source messages
        for i, node in enumerate(matching_summaries):
            lines.append(f"--- Expanded Summary {i + 1}/{len(matching_summaries)} ---")
            lines.append(
                f"Summary (depth={node.depth}, sources={len(node.source_message_ids)}): "
                f"{node.summary_text[:300]}"
            )
            if len(node.summary_text) > 300:
                lines.append("  ...")
            lines.append("")

            # Expand to source messages
            if node.source_message_ids:
                expanded_count = 0
                for mid in node.source_message_ids:
                    if expanded_count >= arguments.max_messages_per_expansion:
                        remaining = len(node.source_message_ids) - expanded_count
                        lines.append(
                            f"  ... and {remaining} more source message(s) "
                            f"(use lcm_expand for full expansion)"
                        )
                        break
                    try:
                        msg = None
                        if hasattr(conv_store, "get_by_id"):
                            msg = conv_store.get_by_id(mid)
                        if msg is not None:
                            lines.append(
                                f"  [{msg.role}] turn={msg.turn_index}: "
                                f"{msg.content[:500]}"
                            )
                            if len(msg.content) > 500:
                                lines.append("    ...")
                            expanded_count += 1
                        else:
                            lines.append(f"  [message {mid} — not found in store]")
                    except Exception as exc:
                        lines.append(f"  [message {mid} — error: {exc}]")
            lines.append("")

        # Step 4: Include direct message matches (not from summaries)
        if matching_messages:
            lines.append("--- Direct Message Matches ---")
            for msg in matching_messages:
                lines.append(
                    f"  [{msg.role}] session={msg.session_id} turn={msg.turn_index}: "
                    f"{msg.content[:400]}"
                )
                if len(msg.content) > 400:
                    lines.append("    ...")
            lines.append("")

        return ToolResult(output="\n".join(lines))
