"""LCM Expand Tool — expand a summary node back to its source content.

Given a summary node ID, retrieves either the original messages (for
depth-0 summaries) or the child summary nodes (for higher-depth summaries).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

# Re-use the same engine wiring as lcm_grep
from prometheus.tools.builtin.lcm_grep import _get_engine


class LCMExpandInput(BaseModel):
    """Arguments for the LCM expand tool."""

    summary_id: str = Field(description="ID of the summary node to expand")
    depth: int | None = Field(
        default=None,
        description=(
            "Expand to a specific depth. None = auto (depth-0 shows source "
            "messages, higher depths show child summaries)"
        ),
    )


class LCMExpandTool(BaseTool):
    """Expand a summary node back to its original messages or child summaries."""

    name = "lcm_expand"
    description = (
        "Expand an LCM summary node to see its source content. "
        "Depth-0 summaries expand to original messages; "
        "higher-depth summaries expand to child summary nodes."
    )
    input_model = LCMExpandInput

    def is_read_only(self, arguments: LCMExpandInput) -> bool:
        return True

    async def execute(
        self, arguments: LCMExpandInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            engine = _get_engine()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        # Look up the summary node
        summary_store = engine.summary_store
        try:
            node = summary_store.get(arguments.summary_id)
        except Exception as exc:
            return ToolResult(
                output=f"Failed to retrieve summary {arguments.summary_id}: {exc}",
                is_error=True,
            )

        if node is None:
            return ToolResult(
                output=f"Summary node {arguments.summary_id!r} not found.",
                is_error=True,
            )

        # Determine effective depth to expand
        target_depth = arguments.depth if arguments.depth is not None else node.depth

        lines: list[str] = [
            f"Expanding summary {node.id} (depth={node.depth}, leaf={node.is_leaf})",
            f"Summary text: {node.summary_text[:200]}{'...' if len(node.summary_text) > 200 else ''}",
            "",
        ]

        # Depth-0: retrieve original source messages
        if target_depth == 0 or node.depth == 0:
            conv_store = engine.conversation_store
            if node.source_message_ids:
                lines.append(f"Source messages ({len(node.source_message_ids)}):")
                for mid in node.source_message_ids:
                    try:
                        # Try to fetch individual message by ID
                        if hasattr(conv_store, "get_by_id"):
                            msg = conv_store.get_by_id(mid)
                        else:
                            # Fallback: search within the store
                            msg = None
                        if msg is not None:
                            lines.append(
                                f"  [{msg.role}] turn={msg.turn_index} "
                                f"id={msg.message_id}"
                            )
                            lines.append(f"    {msg.content[:400]}")
                            if len(msg.content) > 400:
                                lines.append("    ...")
                        else:
                            lines.append(f"  [message {mid} — not found in store]")
                    except Exception as exc:
                        lines.append(f"  [message {mid} — error: {exc}]")
            else:
                lines.append("(no source message IDs recorded)")

        # Higher depth: retrieve child summary nodes
        else:
            if node.parent_ids:
                lines.append(f"Child summaries ({len(node.parent_ids)}):")
                for child_id in node.parent_ids:
                    try:
                        child = summary_store.get(child_id)
                        if child is not None:
                            lines.append(
                                f"  [summary] id={child.id} depth={child.depth} "
                                f"sources={len(child.source_message_ids)} "
                                f"leaf={child.is_leaf}"
                            )
                            snippet = child.summary_text[:200]
                            if len(child.summary_text) > 200:
                                snippet += "..."
                            lines.append(f"    {snippet}")
                        else:
                            lines.append(f"  [summary {child_id} — not found]")
                    except Exception as exc:
                        lines.append(f"  [summary {child_id} — error: {exc}]")
            else:
                lines.append("(no child summary IDs recorded)")

        return ToolResult(output="\n".join(lines))
