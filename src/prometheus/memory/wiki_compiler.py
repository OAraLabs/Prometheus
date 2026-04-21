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
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from prometheus.config.paths import get_config_dir
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
        index = self._load_index()

        # Group facts by entity name
        by_entity: dict[str, list[dict]] = defaultdict(list)
        for fact in new_facts:
            name = fact.get("entity_name", "Unknown")
            by_entity[name].append(fact)

        # Collect all known entity names for cross-reference detection
        known_entities: set[str] = set(index.keys()) | set(by_entity.keys())

        pages_created = 0
        pages_updated = 0

        for entity_name, facts in by_entity.items():
            entity_type = facts[0].get("entity_type", "concept")
            page_path = self._entity_page_path(entity_type, entity_name)

            if page_path.exists():
                self._update_page(page_path, facts)
                pages_updated += 1
            elif self._should_create_page(entity_name):
                self._create_page(page_path, entity_name, entity_type, facts)
                pages_created += 1
            else:
                continue

            # Cross-references
            related: set[str] = set()
            for fact in facts:
                related.update(
                    self._detect_related_entities(
                        fact.get("fact", ""), entity_name, known_entities
                    )
                )
            if related:
                self._add_wiki_links(page_path, related)

        self._regenerate_index()
        self._append_log(pages_updated, pages_created)
        self._update_watermark()

        log.info(
            "WikiCompiler: %d pages updated, %d created from %d facts",
            pages_updated,
            pages_created,
            len(new_facts),
        )

    # -- Directory setup ------------------------------------------------

    def _ensure_dirs(self) -> None:
        for subdir in _SUBDIRS:
            (self._wiki / subdir).mkdir(parents=True, exist_ok=True)

    # -- Index ----------------------------------------------------------

    def _load_index(self) -> dict[str, dict[str, str]]:
        """Parse index.md into ``{entity_name: {path, type, summary}}``."""
        index_path = self._wiki / "index.md"
        if not index_path.exists():
            return {}

        entries: dict[str, dict[str, str]] = {}
        for line in index_path.read_text(encoding="utf-8").splitlines():
            # Format: - [Entity Name](people/Entity_Name.md) — summary
            m = re.match(
                r"^- \[(.+?)\]\((.+?)\)\s*(?:—\s*(.*))?$", line
            )
            if m:
                name, path, summary = m.group(1), m.group(2), m.group(3) or ""
                entries[name] = {"path": path, "summary": summary.strip()}
        return entries

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

    def _should_create_page(self, entity_name: str) -> bool:
        """True if the entity has 2+ total mentions across all memories."""
        results = self._store.search_memories(entity=entity_name, limit=50)
        total_mentions = sum(r.get("mention_count", 1) for r in results)
        return total_mentions >= 2

    def _create_page(
        self,
        page_path: Path,
        entity_name: str,
        entity_type: str,
        facts: list[dict],
    ) -> None:
        today = _today()
        source_count = len(facts)

        frontmatter = yaml.dump(
            {
                "type": entity_type,
                "first_seen": today,
                "last_updated": today,
                "source_count": source_count,
            },
            default_flow_style=False,
            sort_keys=False,
        ).strip()

        lines = [
            f"---\n{frontmatter}\n---",
            "",
            f"# {entity_name}",
            "",
            "## Key Facts",
        ]
        for f in facts:
            source_ids = f.get("source_event_ids", [])
            source_tag = source_ids[0][:8] if source_ids else "unknown"
            lines.append(f"- {f['fact']} (source: {source_tag}, {today})")

        lines.extend(["", "## Related", ""])

        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _update_page(self, page_path: Path, facts: list[dict]) -> None:
        text = page_path.read_text(encoding="utf-8")
        today = _today()

        # Update frontmatter
        text = self._update_frontmatter(text, len(facts), today)

        # Append facts before the Related section
        new_fact_lines = []
        for f in facts:
            source_ids = f.get("source_event_ids", [])
            source_tag = source_ids[0][:8] if source_ids else "unknown"
            new_fact_lines.append(f"- {f['fact']} (source: {source_tag}, {today})")

        insertion = "\n".join(new_fact_lines) + "\n"

        # Insert before "## Related" if it exists, otherwise append
        if "## Related" in text:
            text = text.replace("## Related", insertion + "\n## Related", 1)
        else:
            text = text.rstrip() + "\n" + insertion

        page_path.write_text(text, encoding="utf-8")

    @staticmethod
    def _update_frontmatter(text: str, new_fact_count: int, today: str) -> str:
        """Update last_updated and source_count in YAML frontmatter."""
        if not text.startswith("---"):
            return text

        parts = text.split("---", 2)
        if len(parts) < 3:
            return text

        try:
            fm = yaml.safe_load(parts[1])
            if not isinstance(fm, dict):
                return text
        except yaml.YAMLError:
            return text

        fm["last_updated"] = today
        fm["source_count"] = fm.get("source_count", 0) + new_fact_count

        new_fm = yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
        return f"---\n{new_fm}\n---{parts[2]}"

    # -- Cross-references -----------------------------------------------

    @staticmethod
    def _detect_related_entities(
        fact_text: str,
        self_entity: str,
        known_entities: set[str],
    ) -> list[str]:
        """Find known entity names mentioned in *fact_text* (3+ chars, case-insensitive)."""
        related: list[str] = []
        lower_text = fact_text.lower()
        for entity in known_entities:
            if entity == self_entity:
                continue
            if len(entity) < 3:
                continue
            if entity.lower() in lower_text:
                related.append(entity)
        return related

    @staticmethod
    def _add_wiki_links(page_path: Path, related: set[str]) -> None:
        """Add ``[[Entity]]`` links to the Related section of a page."""
        text = page_path.read_text(encoding="utf-8")

        # Find existing links to avoid duplicates
        existing_links: set[str] = set()
        for m in re.finditer(r"\[\[(.+?)\]\]", text):
            existing_links.add(m.group(1))

        new_links = sorted(related - existing_links)
        if not new_links:
            return

        link_lines = "\n".join(f"- [[{name}]]" for name in new_links) + "\n"

        if "## Related" in text:
            text = text.replace("## Related\n", f"## Related\n{link_lines}", 1)
        else:
            text = text.rstrip() + f"\n\n## Related\n{link_lines}"

        page_path.write_text(text, encoding="utf-8")

    # -- Log / watermark ------------------------------------------------

    def _append_log(self, updated: int, created: int) -> None:
        log_path = self._wiki / "log.md"
        today = _today()
        entry = f"## [{today}] compile | {updated} pages updated, {created} created\n\n"
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8")
            text += entry
        else:
            text = "# Wiki Compile Log\n\n" + entry
        log_path.write_text(text, encoding="utf-8")

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
