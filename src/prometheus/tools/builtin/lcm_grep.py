"""LCM Grep Tool — FTS5 search over conversation messages and summaries.

Provides full-text search into the Lossless Context Management stores,
allowing the agent to recall prior conversation content and summaries
by keyword.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

# ---------------------------------------------------------------------------
# Engine wiring — call set_lcm_engine() at startup to connect stores
# ---------------------------------------------------------------------------
_engine: object | None = None


def set_lcm_engine(engine: object) -> None:
    """Register the global LCM engine so the tool can access stores."""
    global _engine
    _engine = engine


def _get_engine():
    """Return the wired LCM engine or raise."""
    if _engine is None:
        raise RuntimeError(
            "LCM engine has not been initialised. "
            "Call set_lcm_engine() before using the lcm_grep tool."
        )
    return _engine


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class LCMGrepInput(BaseModel):
    """Arguments for the LCM grep tool."""

    query: str = Field(description="Full-text search query")
    session_id: str | None = Field(
        default=None,
        description="Restrict search to a specific session (omit for all sessions)",
    )
    search_target: Literal["messages", "summaries", "both"] = Field(
        default="both",
        description="Search messages, summaries, or both",
    )
    limit: int = Field(default=20, ge=1, le=200, description="Max results to return")


class LCMGrepTool(BaseTool):
    """Full-text search across LCM conversation messages and summaries."""

    name = "lcm_grep"
    description = (
        "Search conversation history and summaries using FTS5 full-text search. "
        "Returns matching messages and/or summary nodes ranked by relevance."
    )
    input_model = LCMGrepInput

    def is_read_only(self, arguments: LCMGrepInput) -> bool:
        return True

    async def execute(
        self, arguments: LCMGrepInput, context: ToolExecutionContext
    ) -> ToolResult:
        try:
            engine = _get_engine()
        except RuntimeError as exc:
            return ToolResult(output=str(exc), is_error=True)

        results: list[str] = []

        # --- Search messages ---
        if arguments.search_target in ("messages", "both"):
            try:
                conv_store = engine.conversation_store
                messages = conv_store.search(
                    arguments.query,
                    session_id=arguments.session_id,
                    limit=arguments.limit,
                )
                for msg in messages:
                    header = (
                        f"[message] id={msg.message_id} "
                        f"session={msg.session_id} "
                        f"role={msg.role} turn={msg.turn_index}"
                    )
                    snippet = msg.content[:300]
                    if len(msg.content) > 300:
                        snippet += "..."
                    results.append(f"{header}\n  {snippet}")
            except Exception as exc:
                results.append(f"[message search error] {exc}")

        # --- Search summaries ---
        if arguments.search_target in ("summaries", "both"):
            try:
                summary_store = engine.summary_store
                if hasattr(summary_store, "search"):
                    summaries = summary_store.search(
                        arguments.query,
                        session_id=arguments.session_id,
                        limit=arguments.limit,
                    )
                    for node in summaries:
                        header = (
                            f"[summary] id={node.id} "
                            f"depth={node.depth} "
                            f"sources={len(node.source_message_ids)} "
                            f"leaf={node.is_leaf}"
                        )
                        snippet = node.summary_text[:300]
                        if len(node.summary_text) > 300:
                            snippet += "..."
                        results.append(f"{header}\n  {snippet}")
                else:
                    results.append("[summary search] summary store does not support FTS yet")
            except Exception as exc:
                results.append(f"[summary search error] {exc}")

        if not results:
            return ToolResult(output="(no matches)")

        header = f"Found {len(results)} result(s) for query: {arguments.query!r}\n"
        return ToolResult(output=header + "\n\n".join(results))
