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


def _seed(store: MemoryStore, entity_type: str, name: str, fact: str, conf: float) -> None:
    """Persist a fact with a placeholder source (provenance is mandatory)."""
    store.persist_memory(entity_type, name, fact, conf, source_event_ids=["seed"])


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
        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "based in Houston", 0.8)

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
    """A newly-persisted fact appears after recompile; the page is rebuilt
    from the DB (no blind append → no duplication)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "based in Houston", 0.8)

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        compiler.compile([_make_fact("Dr. Pham", "nephrologist"),
                          _make_fact("Dr. Pham", "based in Houston")])

        page = wiki_root / "people" / "Dr. Pham.md"
        assert _read_frontmatter(page)["source_count"] == 2

        # Persist a third fact, then recompile — the page reflects the DB.
        _seed(store, "person", "Dr. Pham", "speaks Mandarin", 0.7)
        compiler.compile([_make_fact("Dr. Pham", "speaks Mandarin")])

        text = page.read_text(encoding="utf-8")
        assert "speaks Mandarin" in text
        assert _read_frontmatter(page)["source_count"] == 3
        # The original fact appears exactly once — not re-appended.
        assert text.count("based in Houston") == 1

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
        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "works at Mercy Hospital", 0.9)
        _seed(store, "organization", "Mercy Hospital", "healthcare org", 0.9)
        _seed(store, "organization", "Mercy Hospital", "in Houston", 0.8)

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
        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "candidate", 0.9)
        _seed(store, "concept", "Kubernetes", "container orchestration", 0.9)
        _seed(store, "concept", "Kubernetes", "used in production", 0.85)

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
        _seed(store, "person", "Jane Doe", "recruiter", 0.7)

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

        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "based in Houston", 0.8)

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
        _seed(store, "person", "Dr. Pham", "nephrologist in Houston", 0.95)
        _seed(store, "person", "Dr. Pham", "speaks Mandarin", 0.8)
        _seed(store, "organization", "Houston Medical", "hospital in Houston", 0.9)
        _seed(store, "organization", "Houston Medical", "employs nephrologists", 0.85)

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

            # index.md as compile left it, before the query files back
            index_before = (wiki_root / "index.md").read_text(encoding="utf-8")
            result = await tool.execute(
                wq_mod.WikiQueryInput(query="Houston nephrologist hospital"),
                ctx,
            )
            assert not result.is_error

            # Still files the synthesis note to queries/ (the compounding loop)
            queries_dir = wiki_root / "queries"
            query_files = list(queries_dir.glob("*.md")) if queries_dir.exists() else []
            assert len(query_files) > 0, "Query result should be filed to queries/"

            # ...but must NOT mutate index.md — compile is the sole index writer.
            index_after = (wiki_root / "index.md").read_text(encoding="utf-8")
            assert index_after == index_before, "wiki_query must not write index.md"
        finally:
            wq_mod.get_config_dir = original

        store.close()


# ---------------------------------------------------------------------------
# 2d — compile is idempotent (no blind append)
# ---------------------------------------------------------------------------


def test_compile_is_idempotent():
    """Recompiling against an unchanged store yields a byte-identical page."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"
        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "based in Houston", 0.8)
        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        facts = [_make_fact("Dr. Pham", "nephrologist"),
                 _make_fact("Dr. Pham", "based in Houston")]

        compiler.compile(facts)
        page = wiki_root / "people" / "Dr. Pham.md"
        first = page.read_text(encoding="utf-8")

        compiler.compile(facts)
        assert page.read_text(encoding="utf-8") == first, "compile must be idempotent"

        store.close()


# ---------------------------------------------------------------------------
# 2e — regenerate_all: deterministic, junk-free, queries preserved, no broken links
# ---------------------------------------------------------------------------


def test_regenerate_all_is_deterministic_and_clean():
    import re as _re
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"

        # Two real entities (>=2 mentions) with a cross-reference, plus a junk
        # (path) entity that must not get a page.
        _seed(store, "person", "Dr. Pham", "works at Mercy Hospital", 0.95)
        _seed(store, "person", "Dr. Pham", "nephrologist", 0.9)
        _seed(store, "organization", "Mercy Hospital", "healthcare org", 0.9)
        _seed(store, "organization", "Mercy Hospital", "in Houston", 0.8)
        _seed(store, "tool", "src/marshmallow/utils.py", "a file path", 0.9)
        _seed(store, "tool", "src/marshmallow/utils.py", "still a path", 0.9)

        # A filed-back query note must survive regen verbatim.
        (wiki_root / "queries").mkdir(parents=True)
        qnote = wiki_root / "queries" / "insight-test.md"
        QBODY = "# Insight: keep me\n\nsynthesized, no DB source\n"
        qnote.write_text(QBODY, encoding="utf-8")

        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        compiler.regenerate_all()

        pham = wiki_root / "people" / "Dr. Pham.md"
        mercy = wiki_root / "clients" / "Mercy Hospital.md"
        assert pham.exists() and mercy.exists()
        # junk path entity got no page anywhere
        assert not list(wiki_root.glob("*/*utils*"))
        # cross-link resolves to a real page
        assert "[[Mercy Hospital]]" in pham.read_text(encoding="utf-8")
        # queries/ preserved verbatim
        assert qnote.read_text(encoding="utf-8") == QBODY

        # zero broken links across regenerated entity pages
        pages, links = set(), []
        for sub in ("people", "clients", "projects", "topics"):
            for fp in (wiki_root / sub).glob("*.md"):
                pages.add(fp.stem.replace("_", " ").lower())
                links += [m.lower() for m in _re.findall(r"\[\[([^\]]+)\]\]",
                                                         fp.read_text(encoding="utf-8"))]
        assert [l for l in links if l not in pages] == [], "regen must produce no broken links"

        # idempotent: a second regen reproduces the page byte-for-byte
        before = pham.read_text(encoding="utf-8")
        compiler.regenerate_all()
        assert pham.read_text(encoding="utf-8") == before

        store.close()


def test_no_filename_collisions_among_page_having_entities():
    """Every page-having entity maps to a unique file. The entity gate rejects
    the path/filename-shaped names that would otherwise collide in a shared
    subdir, so the projection is collision-free (SPRINT MEMORY-2 guard)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"
        _seed(store, "person", "Dr. Pham", "nephrologist", 0.95)
        _seed(store, "person", "Dr. Pham", "in Houston", 0.9)
        _seed(store, "organization", "Mercy Hospital", "an org", 0.9)
        _seed(store, "organization", "Mercy Hospital", "in Houston", 0.8)
        # would-be colliders: a path and an underscore form that sanitize toward
        # the same file — both must be gate-rejected, so neither gets a page.
        _seed(store, "tool", "src/foo/utils.py", "x", 0.9)
        _seed(store, "tool", "src/foo/utils.py", "y", 0.9)
        _seed(store, "tool", "src_foo_utils.py", "x", 0.9)
        _seed(store, "tool", "src_foo_utils.py", "y", 0.9)

        WikiCompiler(store=store, wiki_root=wiki_root).regenerate_all()

        files = []
        for sub in ("people", "clients", "projects", "topics"):
            files += [(sub, fp.name) for fp in (wiki_root / sub).glob("*.md")]
        assert len(files) == len(set(files)), f"filename collision among pages: {files}"
        assert not list(wiki_root.glob("*/*utils*")), "path/filename entities must get no page"
        store.close()


# ---------------------------------------------------------------------------
# WikiCompiler — sole writer of index.md, enumerates queries/ (Phase 4)
# ---------------------------------------------------------------------------


def test_compiler_enumerates_queries_into_index():
    """A note in wiki/queries/ is picked up into ## Queries on a compile run.

    This is how filed query results reach the index now that wiki_query no
    longer appends to it (compile is the sole index writer).
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        wiki_root = Path(tmp) / "wiki"
        compiler = WikiCompiler(store=store, wiki_root=wiki_root)
        compiler.compile([
            _make_fact("Dr. Pham", "nephrologist"),
            _make_fact("Dr. Pham", "based in Houston"),
        ])

        # Drop a query note straight into queries/, as wiki_query._file_back does.
        (wiki_root / "queries").mkdir(parents=True, exist_ok=True)
        (wiki_root / "queries" / "houston-care.md").write_text(
            "---\ntype: query\ndate: 2026-06-18\nquery: \"houston care\"\n---\n\n"
            "# Houston care\n\nDr. Pham provides nephrology care in Houston.\n",
            encoding="utf-8",
        )

        # A compile run regenerates the index, which enumerates queries/.
        compiler.regenerate_all()
        index_text = (wiki_root / "index.md").read_text(encoding="utf-8")
        assert "## Queries" in index_text
        assert "queries/houston-care.md" in index_text
        store.close()
