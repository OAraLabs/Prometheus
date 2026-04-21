"""Tests for the wiki compiler and wiki tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import sys
import types

# The prometheus.memory package __init__ has a circular import chain
# (LCMEngine → providers → engine → providers). Bypass it by injecting
# a stub package with __path__ so submodule imports resolve directly.
if "prometheus.memory" not in sys.modules:
    _pkg = types.ModuleType("prometheus.memory")
    _pkg.__path__ = ["src/prometheus/memory"]
    _pkg.__package__ = "prometheus.memory"
    sys.modules["prometheus.memory"] = _pkg

import pytest  # noqa: E402
import yaml  # noqa: E402

from prometheus.memory.store import MemoryStore  # noqa: E402
from prometheus.memory.wiki_compiler import WikiCompiler  # noqa: E402
from prometheus.tools.base import ToolExecutionContext, ToolResult  # noqa: E402
from prometheus.tools.builtin.wiki_query import WikiQueryTool  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp: str) -> MemoryStore:
    return MemoryStore(db_path=Path(tmp) / "memory.db")


def _make_fact(
    entity_name: str,
    fact: str,
    entity_type: str = "person",
    confidence: float = 0.9,
    source_event_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "entity_type": entity_type,
        "entity_name": entity_name,
        "fact": fact,
        "confidence": confidence,
        "source_event_ids": source_event_ids or ["abc12345"],
        "tags": tags or [],
    }


def _read_frontmatter(page_path: Path) -> dict:
    text = page_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}
    return {}


# ---------------------------------------------------------------------------
# WikiCompiler — creates a new entity page from facts
# ---------------------------------------------------------------------------


def test_compiler_creates_entity_page():
    """An entity with 2+ mentions gets a wiki page with correct frontmatter."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        # Persist memory twice so mention_count >= 2
        store.persist_memory("person", "Dr. Pham", "nephrologist", 0.95)
        store.persist_memory("person", "Dr. Pham", "based in Houston", 0.8)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        facts = [
            _make_fact("Dr. Pham", "nephrologist"),
            _make_fact("Dr. Pham", "based in Houston"),
        ]
        compiler.compile(facts)

        page = wiki_root / "people" / "Dr. Pham.md"
        assert page.exists(), "Page should be created for entity with 2+ mentions"

        text = page.read_text(encoding="utf-8")
        assert "# Dr. Pham" in text
        assert "nephrologist" in text
        assert "based in Houston" in text

        fm = _read_frontmatter(page)
        assert fm["type"] == "person"
        assert fm["source_count"] == 2

        store.close()


# ---------------------------------------------------------------------------
# WikiCompiler — updates existing page with new facts
# ---------------------------------------------------------------------------


def test_compiler_updates_existing_page():
    """New facts are appended and last_updated is bumped."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        store.persist_memory("person", "Dr. Pham", "nephrologist", 0.95)
        store.persist_memory("person", "Dr. Pham", "based in Houston", 0.8)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)

        # Initial compile
        compiler.compile([_make_fact("Dr. Pham", "nephrologist"),
                          _make_fact("Dr. Pham", "based in Houston")])

        page = wiki_root / "people" / "Dr. Pham.md"
        fm_before = _read_frontmatter(page)
        initial_count = fm_before["source_count"]

        # Update with new fact
        compiler.compile([_make_fact("Dr. Pham", "speaks Mandarin")])

        text = page.read_text(encoding="utf-8")
        assert "speaks Mandarin" in text

        fm_after = _read_frontmatter(page)
        assert fm_after["source_count"] == initial_count + 1

        store.close()


# ---------------------------------------------------------------------------
# WikiCompiler — adds cross-references between related entities
# ---------------------------------------------------------------------------


def test_compiler_adds_cross_references():
    """When one entity's fact mentions another, a [[wiki-link]] is added."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        # Both entities need 2+ mentions
        store.persist_memory("person", "Dr. Pham", "nephrologist", 0.95)
        store.persist_memory("person", "Dr. Pham", "works at Mercy Hospital", 0.9)
        store.persist_memory("organization", "Mercy Hospital", "healthcare org", 0.9)
        store.persist_memory("organization", "Mercy Hospital", "in Houston", 0.8)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)

        facts = [
            _make_fact("Dr. Pham", "works at Mercy Hospital"),
            _make_fact("Mercy Hospital", "healthcare org", entity_type="organization"),
        ]
        compiler.compile(facts)

        pham_page = wiki_root / "people" / "Dr. Pham.md"
        assert pham_page.exists()

        text = pham_page.read_text(encoding="utf-8")
        assert "[[Mercy Hospital]]" in text, "Cross-reference should link to Mercy Hospital"

        store.close()


# ---------------------------------------------------------------------------
# WikiCompiler — regenerates index.md correctly
# ---------------------------------------------------------------------------


def test_compiler_regenerates_index():
    """Index.md should list all pages organized by category."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        # Create entities across different types
        store.persist_memory("person", "Dr. Pham", "nephrologist", 0.95)
        store.persist_memory("person", "Dr. Pham", "candidate", 0.9)
        store.persist_memory("concept", "Kubernetes", "container orchestration", 0.9)
        store.persist_memory("concept", "Kubernetes", "used in production", 0.85)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        facts = [
            _make_fact("Dr. Pham", "nephrologist"),
            _make_fact("Dr. Pham", "candidate"),
            _make_fact("Kubernetes", "container orchestration", entity_type="concept"),
            _make_fact("Kubernetes", "used in production", entity_type="concept"),
        ]
        compiler.compile(facts)

        index = wiki_root / "index.md"
        assert index.exists()

        text = index.read_text(encoding="utf-8")
        assert "## People" in text
        assert "## Topics" in text
        assert "Dr. Pham" in text
        assert "Kubernetes" in text

        store.close()


# ---------------------------------------------------------------------------
# WikiCompiler — skips single-mention entities
# ---------------------------------------------------------------------------


def test_compiler_skips_single_mention():
    """An entity with only 1 mention should NOT get a page."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        # Only one memory — mention_count = 1
        store.persist_memory("person", "Jane Doe", "recruiter", 0.7)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        compiler.compile([_make_fact("Jane Doe", "recruiter")])

        page = wiki_root / "people" / "Jane Doe.md"
        assert not page.exists(), "Should not create page for single-mention entity"

        store.close()


# ---------------------------------------------------------------------------
# WikiQueryTool — reads index, finds relevant page, returns content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_tool_finds_page():
    """WikiQueryTool should find a relevant page and return its content."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        store.persist_memory("person", "Dr. Pham", "nephrologist", 0.95)
        store.persist_memory("person", "Dr. Pham", "based in Houston", 0.8)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        compiler.compile([
            _make_fact("Dr. Pham", "nephrologist"),
            _make_fact("Dr. Pham", "based in Houston"),
        ])

        # Patch get_config_dir to point to our tmp wiki
        import prometheus.tools.builtin.wiki_query as wq_mod
        original = wq_mod.get_config_dir
        wq_mod.get_config_dir = lambda: Path(tmp)

        try:
            tool = WikiQueryTool()
            ctx = ToolExecutionContext(cwd=Path(tmp))
            result = await tool.execute(
                wq_mod.WikiQueryInput(query="Dr. Pham nephrologist"),
                ctx,
            )
            assert not result.is_error
            assert "Dr. Pham" in result.output
            assert "nephrologist" in result.output
        finally:
            wq_mod.get_config_dir = original

        store.close()


# ---------------------------------------------------------------------------
# WikiQueryTool — files query result back to wiki/queries/
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_tool_writes_to_queries():
    """When a query spans 2+ pages and is substantial, it gets filed back."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        # Create two entities that will both match the query
        store.persist_memory("person", "Dr. Pham", "nephrologist in Houston", 0.95)
        store.persist_memory("person", "Dr. Pham", "speaks Mandarin", 0.8)
        store.persist_memory("organization", "Houston Medical", "hospital in Houston", 0.9)
        store.persist_memory("organization", "Houston Medical", "employs nephrologists", 0.85)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        compiler.compile([
            _make_fact("Dr. Pham", "nephrologist in Houston"),
            _make_fact("Dr. Pham", "speaks Mandarin"),
            _make_fact("Houston Medical", "hospital in Houston", entity_type="organization"),
            _make_fact("Houston Medical", "employs nephrologists", entity_type="organization"),
        ])

        import prometheus.tools.builtin.wiki_query as wq_mod
        original = wq_mod.get_config_dir
        wq_mod.get_config_dir = lambda: Path(tmp)

        try:
            tool = WikiQueryTool()
            ctx = ToolExecutionContext(cwd=Path(tmp))
            result = await tool.execute(
                wq_mod.WikiQueryInput(query="Houston nephrologist hospital"),
                ctx,
            )
            assert not result.is_error

            # Check that a query page was filed
            queries_dir = wiki_root / "queries"
            query_files = list(queries_dir.glob("*.md")) if queries_dir.exists() else []
            assert len(query_files) > 0, "Query result should be filed to queries/"

            # Check that index was updated
            index_text = (wiki_root / "index.md").read_text(encoding="utf-8")
            assert "queries/" in index_text
        finally:
            wq_mod.get_config_dir = original

        store.close()
