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

# Bound the returned payload by SIZE, not just page count, so a few large pages
# can't blow past the model's working window. Expressed in tokens (the
# meaningful unit) with a coarse char proxy — no tokenizer dependency. Survey
# range: 24-32K tokens. (The index.md write-path growth is carved out to Phase 4.)
_MAX_RESULT_TOKENS = 28_000
_CHARS_PER_TOKEN = 4  # coarse English/markdown proxy
_MAX_RESULT_CHARS = _MAX_RESULT_TOKENS * _CHARS_PER_TOKEN
_PAYLOAD_NOTICE_RESERVE = 200  # chars reserved for the bounded-view notice


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

        # Resolve which of the top-ranked entries actually have page files.
        candidates: list[tuple[str, Path]] = []
        for _score, name, rel_path in top:
            page_path = wiki_root / rel_path
            if page_path.exists():
                candidates.append((name, page_path))

        if not candidates:
            return ToolResult(output="Found index entries but page files are missing.")

        # Bound the payload by size. Include pages in rank order until the next
        # would exceed the budget; if the highest-ranked page alone exceeds it,
        # include it truncated with an explicit marker. A small reserve keeps
        # the prepended notice within the total budget.
        page_budget = _MAX_RESULT_CHARS - _PAYLOAD_NOTICE_RESERVE
        separator = "\n\n---\n\n"
        content_parts: list[str] = []
        used = 0
        truncated = False
        for name, page_path in candidates:
            block = f"### {name}\n{page_path.read_text(encoding='utf-8')}"
            sep_cost = len(separator) if content_parts else 0
            if used + sep_cost + len(block) <= page_budget:
                content_parts.append(block)
                used += sep_cost + len(block)
            elif not content_parts:
                marker = "\n\n[truncated: page exceeds the wiki_query size budget]"
                keep = max(0, page_budget - len(marker))
                content_parts.append(block[:keep] + marker)
                truncated = True
                break
            else:
                break

        pages_read = len(content_parts)
        pages_available = len(candidates)
        combined = separator.join(content_parts)

        # Tell the model this is a bounded view — it never sees the dropped
        # pages or the truncated tail otherwise.
        notice = (
            f"[wiki_query: {pages_read} of {pages_available} relevant page(s), "
            f"bounded to ~{_MAX_RESULT_TOKENS // 1000}K tokens"
            + ("; top page truncated" if truncated else "")
            + "]"
        )
        output = f"{notice}\n\n{combined}"

        # File-back: save substantial multi-page results to queries/. Files the
        # bounded page content (not the notice). The index.md write here is the
        # Phase-4 carve-out — left untouched.
        if (
            pages_read >= _MIN_PAGES_FOR_FILEBACK
            and len(combined) >= _MIN_CONTENT_LEN_FOR_FILEBACK
        ):
            self._file_back(wiki_root, arguments.query, combined)

        return ToolResult(output=output)

    @staticmethod
    def _file_back(wiki_root: Path, query: str, content: str) -> None:
        """Write the query-result note to ``wiki/queries/``.

        Does NOT touch ``index.md``. The compiler is the sole writer of the
        index and enumerates ``queries/`` into the ``## Queries`` section on its
        next regenerate (``WikiCompiler._regenerate_index``). A query-time index
        append made index.md a two-writer artifact that compile then fought and
        overwrote (the same smell #53 fixed for links).
        """
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

        log.info("WikiQueryTool: filed query result to %s", query_path)
