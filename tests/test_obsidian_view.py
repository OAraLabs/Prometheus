"""Phase 5 — Obsidian read-only view: graph color group + manual predicate.

Validates config/obsidian/graph.json and proves the data its color-group search
relies on: a compiled manual (/note) page carries `manual: true` frontmatter
(so Obsidian's ["manual":true] search matches it) and an ambient page does not.
The final visual confirmation happens in Obsidian on the Mac.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

# Bypass the prometheus.memory circular-import chain (same shim as test_wiki).
if "prometheus.memory" not in sys.modules:
    _pkg = types.ModuleType("prometheus.memory")
    _pkg.__path__ = ["src/prometheus/memory"]
    _pkg.__package__ = "prometheus.memory"
    sys.modules["prometheus.memory"] = _pkg

import yaml  # noqa: E402

from prometheus.memory.store import MemoryStore  # noqa: E402
from prometheus.memory.wiki_compiler import WikiCompiler  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_GRAPH = _REPO / "config" / "obsidian" / "graph.json"
_MANUAL_QUERY = '["manual":true]'


def _read_frontmatter(page: Path) -> dict:
    text = page.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}
    return {}


def _obsidian_manual_predicate(fm: dict) -> bool:
    """Proxy for Obsidian's graph color-group search ``["manual":true]``:
    a frontmatter property ``manual`` equal to boolean true."""
    return fm.get("manual") is True


def test_graph_json_valid_with_manual_color_group():
    cfg = json.loads(_GRAPH.read_text(encoding="utf-8"))
    groups = cfg.get("colorGroups", [])
    queries = [g.get("query") for g in groups]
    assert _MANUAL_QUERY in queries, f"manual color-group query missing; got {queries}"
    group = next(g for g in groups if g.get("query") == _MANUAL_QUERY)
    assert "rgb" in group.get("color", {}), "color group must carry an rgb color"


def test_manual_predicate_matches_manual_not_ambient():
    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(db_path=Path(tmp) / "memory.db")
        wiki = Path(tmp) / "wiki"

        # Ambient entity: 2 mentions, no manual flag -> page, but no manual marker.
        store.persist_memory("person", "AmbientAlice", "a recruiter", 0.8, source_event_ids=["e1"])
        store.persist_memory("person", "AmbientAlice", "based in Austin", 0.8, source_event_ids=["e2"])
        # Manual entity: one /note fact -> manual: true marker.
        store.persist_memory(
            "note", "ManualBob", "pinned by hand", 1.0,
            source_event_ids=["manual"], manual=True,
        )
        WikiCompiler(store=store, wiki_root=wiki).regenerate_all()

        manual_page = next(iter(wiki.glob("*/ManualBob.md")), None)
        ambient_page = next(iter(wiki.glob("*/AmbientAlice.md")), None)
        assert manual_page is not None, "manual fact should have earned a page"
        assert ambient_page is not None, "ambient entity should have a page"

        # The color-group search matches the manual page and NOT the ambient one.
        assert _obsidian_manual_predicate(_read_frontmatter(manual_page)) is True
        assert _obsidian_manual_predicate(_read_frontmatter(ambient_page)) is False

        store.close()
