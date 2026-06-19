"""Wiki Compiler — transforms extracted memory facts into a cross-linked Markdown wiki.

Reads from MemoryStore, writes to ~/.prometheus/wiki/ with entity pages organized
by type (people/, clients/, projects/, topics/) and an auto-generated index.md.

Source: Sprint 5 extension for Prometheus.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import yaml

from prometheus.config.paths import get_config_dir
from prometheus.memory.entity_validation import classify_entity, quarantine
from prometheus.memory.store import MemoryStore

log = logging.getLogger(__name__)

# Entity type → wiki subdirectory
_TYPE_TO_SUBDIR: dict[str, str] = {
    "person": "people",
    "organization": "clients",
    "task": "projects",
    "tool": "projects",
    "concept": "topics",
    "place": "topics",
    "preference": "topics",
}

_DEFAULT_SUBDIR = "topics"

_SUBDIRS = ("people", "clients", "projects", "topics", "queries")


def _safe_filename(name: str) -> str:
    """Sanitise an entity name for use as a filename (no extension)."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


class WikiCompiler:
    """Compile extracted memory facts into a navigable Markdown wiki."""

    def __init__(
        self,
        store: MemoryStore,
        wiki_root: Path | None = None,
    ) -> None:
        self._store = store
        self._wiki = (
            Path(wiki_root) if wiki_root else get_config_dir() / "wiki"
        )
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, new_facts: list[dict]) -> None:
        """Compile *new_facts* into wiki pages.

        Each element of *new_facts* is a dict with at least:
        ``entity_type``, ``entity_name``, ``fact``, ``confidence``.
        """
        if not new_facts:
            return

        with self._lock:
            self._compile_locked(new_facts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compile_locked(self, new_facts: list[dict]) -> None:
        self._ensure_dirs()

        # The page-having set is derived from the store, so compile and
        # regenerate make identical create/link decisions. Each touched
        # entity's page is rebuilt in full from the DB (no blind append) —
        # re-running with an unchanged store is a byte-identical no-op.
        linkable = self._page_having_entities()
        touched = {f.get("entity_name", "Unknown") for f in new_facts}

        pages_created = 0
        pages_updated = 0
        for entity_name in sorted(touched):
            reason = classify_entity(entity_name)
            if reason is not None:
                quarantine(str(entity_name), reason, context="wiki_compile")
                continue
            if entity_name not in linkable:
                continue  # fewer than 2 mentions — not yet page-worthy
            entity_type = linkable[entity_name]
            page_path = self._entity_page_path(entity_type, entity_name)
            existed = page_path.exists()
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_text(
                self._render_page(entity_name, entity_type, linkable),
                encoding="utf-8",
            )
            pages_updated += int(existed)
            pages_created += int(not existed)

        self._regenerate_index()
        self._append_log(pages_updated, pages_created)
        self._update_watermark()

        log.info(
            "WikiCompiler: %d pages updated, %d created from %d facts",
            pages_updated,
            pages_created,
            len(new_facts),
        )

    def regenerate_all(self) -> dict[str, int]:
        """Deterministically rebuild ALL entity pages from the store.

        Wipes the entity subdirs (people/clients/projects/topics) and rewrites
        one page per page-having entity straight from memory.db. ``queries/``
        is never touched — those are filed-back synthesized answers with no DB
        source. Idempotent: an unchanged store yields byte-identical entity
        pages + index (``log.md`` / ``.last_compile_ts`` aside — append/stamp).
        """
        with self._lock:
            self._ensure_dirs()
            for subdir in ("people", "clients", "projects", "topics"):
                d = self._wiki / subdir
                if d.exists():
                    for page in d.glob("*.md"):
                        page.unlink()

            linkable = self._page_having_entities()
            for entity_name, entity_type in sorted(linkable.items()):
                page_path = self._entity_page_path(entity_type, entity_name)
                page_path.parent.mkdir(parents=True, exist_ok=True)
                page_path.write_text(
                    self._render_page(entity_name, entity_type, linkable),
                    encoding="utf-8",
                )

            self._regenerate_index()
            self._log_line(
                f"regenerate | {len(linkable)} entity pages rebuilt from memory.db"
            )
            self._update_watermark()
            log.info("WikiCompiler: regenerated %d entity pages", len(linkable))
            return {"pages": len(linkable)}

    # -- Directory setup ------------------------------------------------

    def _ensure_dirs(self) -> None:
        for subdir in _SUBDIRS:
            (self._wiki / subdir).mkdir(parents=True, exist_ok=True)

    # -- Index ----------------------------------------------------------

    def _regenerate_index(self) -> None:
        """Scan all subdirs and rebuild index.md organized by category."""
        sections: dict[str, list[str]] = {s: [] for s in _SUBDIRS if s != "queries"}
        sections["queries"] = []

        for subdir in _SUBDIRS:
            subdir_path = self._wiki / subdir
            if not subdir_path.exists():
                continue
            for page in sorted(subdir_path.glob("*.md")):
                name, summary, etype = self._read_page_meta(page)
                rel = f"{subdir}/{page.name}"
                sections[subdir].append(f"- [{name}]({rel}) — {summary}")

        lines = ["# Prometheus Wiki Index", ""]
        category_titles = {
            "people": "People",
            "clients": "Clients",
            "projects": "Projects",
            "topics": "Topics",
            "queries": "Queries",
        }
        for subdir in _SUBDIRS:
            entries = sections[subdir]
            if not entries:
                continue
            lines.append(f"## {category_titles.get(subdir, subdir.title())}")
            lines.extend(entries)
            lines.append("")

        (self._wiki / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _read_page_meta(page_path: Path) -> tuple[str, str, str]:
        """Read entity name, summary, and type from a page's frontmatter."""
        text = page_path.read_text(encoding="utf-8")
        name = page_path.stem.replace("_", " ")
        summary = ""
        etype = "unknown"

        # Parse YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1])
                    if isinstance(fm, dict):
                        etype = fm.get("type", etype)
                except yaml.YAMLError:
                    pass
                body = parts[2].strip()
            else:
                body = text
        else:
            body = text

        # Extract heading as name
        for line in body.splitlines():
            if line.startswith("# "):
                name = line[2:].strip()
                break

        # First non-heading, non-empty line as summary (paragraph or bullet)
        in_body = False
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                in_body = True
                continue
            if in_body and stripped:
                # Strip leading bullet markers for cleaner summary
                clean = stripped.lstrip("- ").split("(source:")[0].strip()
                if clean:
                    summary = clean[:120]
                    break

        return name, summary, etype

    # -- Page creation / update -----------------------------------------

    def _entity_page_path(self, entity_type: str, entity_name: str) -> Path:
        subdir = _TYPE_TO_SUBDIR.get(entity_type, _DEFAULT_SUBDIR)
        return self._wiki / subdir / f"{_safe_filename(entity_name)}.md"

    def _facts_for_entity(self, entity_name: str) -> list[dict]:
        """All stored facts whose entity_name matches *entity_name* exactly
        (case-insensitive). ``search_memories`` does a ``LIKE %name%`` match,
        so the exact filter happens here."""
        results = self._store.search_memories(entity=entity_name, limit=500)
        el = entity_name.lower()
        return [r for r in results if (r.get("entity_name") or "").lower() == el]

    def _page_having_entities(self) -> dict[str, str]:
        """Map ``entity_name -> entity_type`` for entities that qualify for a
        page: structurally valid AND (>= 2 total mentions OR any manual fact).
        Manual facts (``/note``) earn a page on first mention — you asserting it
        explicitly *is* the signal the >= 2 threshold approximates. Derived from
        the store so compile and regenerate agree on which links resolve."""
        agg: dict[str, dict] = {}
        for r in self._store.get_all_memories(limit=1_000_000):
            name = r.get("entity_name") or "Unknown"
            entry = agg.setdefault(
                name,
                {"mentions": 0, "type": r.get("entity_type", "concept"), "manual": False},
            )
            entry["mentions"] += r.get("mention_count", 1) or 1
            if r.get("manual"):
                entry["manual"] = True
        return {
            name: entry["type"]
            for name, entry in agg.items()
            if classify_entity(name) is None
            and (entry["mentions"] >= 2 or entry["manual"])
        }

    @staticmethod
    def _fact_date(r: dict) -> str:
        ts = r.get("last_mentioned") or r.get("timestamp") or 0.0
        return time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "unknown"

    def _render_page(
        self, entity_name: str, entity_type: str, linkable: dict[str, str]
    ) -> str:
        """Render the full entity page deterministically from the store.

        Pages are a pure projection of memory.db: re-rendering an unchanged
        store yields byte-identical output (no blind append). Cross-links point
        only to entities that have a page, so the output has no broken links.
        """
        facts = self._facts_for_entity(entity_name)
        facts.sort(key=lambda r: (r.get("timestamp") or 0.0, r.get("fact") or ""))

        dates = [self._fact_date(r) for r in facts]
        first_seen = min(dates) if dates else _today()
        last_updated = max(dates) if dates else _today()

        fm: dict[str, object] = {
            "type": entity_type,
            "first_seen": first_seen,
            "last_updated": last_updated,
            "source_count": len(facts),
        }
        # Render marker (Phase 4b): a page with any manual (/note) fact is
        # flagged so retrieval ranks it first, lint exempts it, and Phase 5
        # styles the graph node distinctly.
        if any(r.get("manual") for r in facts):
            fm["manual"] = True
        frontmatter = yaml.dump(
            fm, default_flow_style=False, sort_keys=False
        ).strip()

        lines = [f"---\n{frontmatter}\n---", "", f"# {entity_name}", "", "## Key Facts"]
        related: set[str] = set()
        for r in facts:
            src = sorted(r.get("source_event_ids") or [])
            tag = src[0][:8] if src else "unknown"
            lines.append(f"- {r['fact']} (source: {tag}, {self._fact_date(r)})")
            related.update(
                self._detect_related_entities(
                    r.get("fact", ""), entity_name, set(linkable)
                )
            )

        lines.extend(["", "## Related", ""])
        for name in sorted(e for e in related if e in linkable):
            lines.append(f"- [[{name}]]")

        return "\n".join(lines) + "\n"

    # -- Cross-references -----------------------------------------------

    @staticmethod
    def _detect_related_entities(
        fact_text: str,
        self_entity: str,
        known_entities: set[str],
    ) -> list[str]:
        """Find known entity names mentioned in *fact_text* as whole words.

        Word-boundary matching (not naive substring) so short names like
        "sed" no longer match inside "used" / "passed". Structurally-invalid
        candidates are never linked.
        """
        related: list[str] = []
        lower_text = fact_text.lower()
        for entity in known_entities:
            if entity == self_entity:
                continue
            if len(entity) < 3:
                continue
            if classify_entity(entity) is not None:
                continue
            if re.search(rf"\b{re.escape(entity.lower())}\b", lower_text):
                related.append(entity)
        return related

    # -- Log / watermark ------------------------------------------------

    def _log_line(self, message: str) -> None:
        log_path = self._wiki / "log.md"
        entry = f"## [{_today()}] {message}\n\n"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8") + entry
        else:
            text = "# Wiki Compile Log\n\n" + entry
        log_path.write_text(text, encoding="utf-8")

    def _append_log(self, updated: int, created: int) -> None:
        self._log_line(f"compile | {updated} pages updated, {created} created")

    def _update_watermark(self) -> None:
        ts_path = self._wiki / ".last_compile_ts"
        ts_path.write_text(str(time.time()), encoding="utf-8")

    def get_watermark(self) -> float:
        """Return the timestamp of the last compilation (0.0 if never run)."""
        ts_path = self._wiki / ".last_compile_ts"
        if ts_path.exists():
            try:
                return float(ts_path.read_text(encoding="utf-8").strip())
            except ValueError:
                return 0.0
        return 0.0

    @property
    def wiki_root(self) -> Path:
        return self._wiki
