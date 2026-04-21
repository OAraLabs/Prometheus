"""Tool for querying the Prometheus wiki to answer knowledge questions."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.config.paths import get_config_dir
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

# Minimum thresholds for filing a query result back to the wiki
_MIN_PAGES_FOR_FILEBACK = 2
_MIN_CONTENT_LEN_FOR_FILEBACK = 200


def _safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "_", name)[:80]


class WikiQueryInput(BaseModel):
    """Arguments for wiki_query."""

    query: str = Field(description="Knowledge question to answer from the wiki.")


class WikiQueryTool(BaseTool):
    """Search the Prometheus wiki to answer knowledge questions.

    Reads index.md, finds relevant entity pages, and returns their
    content.  If the answer spans multiple pages and is substantial,
    it is filed back to ``wiki/queries/`` so future queries can find it
    directly (compounding knowledge loop).
    """

    name = "wiki_query"
    description = (
        "Search the Prometheus wiki for answers to knowledge questions. "
        "Returns relevant wiki page content. Substantial multi-page "
        "answers are saved to wiki/queries/ for future reference."
    )
    input_model = WikiQueryInput

    def is_read_only(self, arguments: WikiQueryInput) -> bool:
        del arguments
        return False  # may write query result pages

    async def execute(
        self, arguments: WikiQueryInput, context: ToolExecutionContext
    ) -> ToolResult:
        wiki_root = get_config_dir() / "wiki"
        index_path = wiki_root / "index.md"

        if not index_path.exists():
            return ToolResult(
                output="Wiki not found. Run wiki_compile first to build the wiki.",
                is_error=True,
            )

        index_text = index_path.read_text(encoding="utf-8")
        query_words = set(arguments.query.lower().split())

        # Score each entry by keyword overlap
        scored: list[tuple[int, str, str]] = []
        for line in index_text.splitlines():
            m = re.match(r"^- \[(.+?)\]\((.+?)\)\s*(?:—\s*(.*))?$", line)
            if not m:
                continue
            name, rel_path, summary = m.group(1), m.group(2), m.group(3) or ""
            entry_words = set((name + " " + summary).lower().split())
            overlap = len(query_words & entry_words)
            if overlap > 0:
                scored.append((overlap, name, rel_path))

        if not scored:
            return ToolResult(output="No relevant wiki pages found for this query.")

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:5]

        # Read the top pages
        pages_read = 0
        content_parts: list[str] = []
        for _score, name, rel_path in top:
            page_path = wiki_root / rel_path
            if not page_path.exists():
                continue
            text = page_path.read_text(encoding="utf-8")
            content_parts.append(f"### {name}\n{text}")
            pages_read += 1

        if not content_parts:
            return ToolResult(output="Found index entries but page files are missing.")

        combined = "\n\n---\n\n".join(content_parts)

        # File-back: save substantial multi-page results to queries/
        if (
            pages_read >= _MIN_PAGES_FOR_FILEBACK
            and len(combined) >= _MIN_CONTENT_LEN_FOR_FILEBACK
        ):
            self._file_back(wiki_root, arguments.query, combined, index_path)

        return ToolResult(output=combined)

    @staticmethod
    def _file_back(
        wiki_root: Path, query: str, content: str, index_path: Path
    ) -> None:
        """Write the query result to wiki/queries/ and update the index."""
        queries_dir = wiki_root / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)

        safe = _safe_filename(query)
        query_path = queries_dir / f"{safe}.md"

        today = time.strftime("%Y-%m-%d", time.localtime())
        page_text = (
            f"---\ntype: query\ndate: {today}\nquery: \"{query}\"\n---\n\n"
            f"# {query}\n\n{content}\n"
        )
        query_path.write_text(page_text, encoding="utf-8")

        # Append to index if not already present
        index_text = index_path.read_text(encoding="utf-8")
        rel = f"queries/{query_path.name}"
        if rel not in index_text:
            if "## Queries" not in index_text:
                index_text = index_text.rstrip() + "\n\n## Queries\n"
            index_text += f"- [{query}]({rel}) — auto-filed query result\n"
            index_path.write_text(index_text, encoding="utf-8")

        log.info("WikiQueryTool: filed query result to %s", query_path)
