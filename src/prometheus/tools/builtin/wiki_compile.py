"""Tool for on-demand wiki compilation from extracted memory facts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

if TYPE_CHECKING:
    from prometheus.memory.store import MemoryStore
    from prometheus.memory.wiki_compiler import WikiCompiler

log = logging.getLogger(__name__)

# Module-level singletons (set by daemon.py at startup)
_wiki_compiler: WikiCompiler | None = None
_memory_store: MemoryStore | None = None


def set_wiki_compiler(compiler: WikiCompiler, store: MemoryStore) -> None:
    """Register the global WikiCompiler so the tool can access it."""
    global _wiki_compiler, _memory_store  # noqa: PLW0603
    _wiki_compiler = compiler
    _memory_store = store


class WikiCompileInput(BaseModel):
    """Arguments for wiki_compile."""

    entity_name: str | None = Field(
        default=None,
        description="Compile only facts about this entity. Omit to compile all uncompiled facts.",
    )


class WikiCompileTool(BaseTool):
    """Compile recent memory facts into wiki pages on demand."""

    name = "wiki_compile"
    description = (
        "Compile extracted memory facts into the Prometheus wiki. "
        "Pass an entity_name to compile facts for a single entity, "
        "or omit it to compile all facts since the last compilation."
    )
    input_model = WikiCompileInput

    async def execute(
        self, arguments: WikiCompileInput, context: ToolExecutionContext
    ) -> ToolResult:
        if _wiki_compiler is None or _memory_store is None:
            return ToolResult(
                output="WikiCompiler not initialised. Is the daemon running?",
                is_error=True,
            )

        watermark = _wiki_compiler.get_watermark()

        # Query MemoryStore for facts newer than the watermark
        if arguments.entity_name:
            facts = _memory_store.search_memories(
                entity=arguments.entity_name, limit=200
            )
            # Filter to only facts after watermark
            facts = [f for f in facts if f.get("timestamp", 0) > watermark]
        else:
            # Get all memories after watermark — use a broad search
            facts = _memory_store.search_memories(limit=500)
            facts = [f for f in facts if f.get("timestamp", 0) > watermark]

        if not facts:
            return ToolResult(output="No new facts to compile.")

        _wiki_compiler.compile(facts)

        return ToolResult(
            output=f"Compiled {len(facts)} facts into the wiki."
        )
