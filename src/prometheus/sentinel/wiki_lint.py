"""WikiLinter — health checks on the Prometheus wiki.

Source: Novel code for Prometheus Sprint 9.
Scans for orphan pages, broken links, stale pages, potential duplicates,
missing cross-references, and category imbalance. No LLM needed.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

_SUBDIRS = ("people", "clients", "projects", "topics", "queries")


@dataclass
class LintIssue:
    """A single wiki health issue."""

    severity: str  # "error", "warning", "info"
    category: str  # "orphan", "broken_link", "stale", "duplicate", "missing_crossref", "imbalance"
    page: str
    detail: str
    fixable: bool = False


@dataclass
class LintResult:
    """Aggregate lint output."""

    issues: list[LintIssue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


class WikiLinter:
    """Scan the Prometheus wiki for health issues."""

    def __init__(self, wiki_root: Path | None = None) -> None:
        self.wiki_root = Path(wiki_root) if wiki_root else get_config_dir() / "wiki"

    def lint(self) -> LintResult:
        """Run all lint checks. No LLM needed."""
        if not self.wiki_root.exists():
            return LintResult()

        issues: list[LintIssue] = []
        pages = self._scan_pages()

        if not pages:
            return LintResult()

        issues.extend(self._find_orphans(pages))
        issues.extend(self._find_broken_links(pages))
        issues.extend(self._find_stale_pages(pages))
        issues.extend(self._find_potential_duplicates(pages))
        issues.extend(self._find_missing_crossrefs(pages))
        issues.extend(self._check_category_balance(pages))

        return LintResult(issues=issues)

    def auto_fix(self, result: LintResult) -> int:
        """Fix safe issues. Returns count of fixes applied."""
        fixed = 0
        for issue in result.issues:
            if not issue.fixable:
                continue
            try:
                if issue.category == "broken_link":
                    self._fix_broken_link(issue)
                    fixed += 1
                elif issue.category == "missing_crossref":
                    self._fix_missing_crossref(issue)
                    fixed += 1
            except Exception:
                log.exception("WikiLinter: failed to fix %s", issue)

        if fixed:
            self._append_log(f"Auto-fixed {fixed} issues")
        return fixed

    def summary(self, issues: list[LintIssue] | None = None) -> str:
        """Human-readable summary of lint results."""
        if issues is None:
            issues = self.lint().issues
        if not issues:
            return "Wiki is healthy — no issues found."

        lines = [f"Wiki lint: {len(issues)} issue(s) found\n"]
        by_cat: dict[str, list[LintIssue]] = {}
        for issue in issues:
            by_cat.setdefault(issue.category, []).append(issue)

        for cat, cat_issues in by_cat.items():
            lines.append(f"  {cat} ({len(cat_issues)}):")
            for issue in cat_issues[:5]:
                lines.append(f"    [{issue.severity}] {issue.page}: {issue.detail}")
            if len(cat_issues) > 5:
                lines.append(f"    ... and {len(cat_issues) - 5} more")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_pages(self) -> dict[str, dict[str, Any]]:
        """Scan all wiki pages. Returns {relative_path: {frontmatter, content, links}}."""
        pages: dict[str, dict[str, Any]] = {}
        for subdir in _SUBDIRS:
            d = self.wiki_root / subdir
            if not d.exists():
                continue
            for md_file in d.glob("*.md"):
                rel = f"{subdir}/{md_file.name}"
                content = md_file.read_text(encoding="utf-8")
                frontmatter = self._parse_frontmatter(content)
                links = self._extract_wiki_links(content)
                entity_name = md_file.stem.replace("-", " ").replace("_", " ")
                pages[rel] = {
                    "path": md_file,
                    "frontmatter": frontmatter,
                    "content": content,
                    "links": links,
                    "entity_name": entity_name,
                }
        return pages

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        """Extract YAML frontmatter from markdown."""
        if not content.startswith("---"):
            return {}
        end = content.find("---", 3)
        if end < 0:
            return {}
        try:
            return yaml.safe_load(content[3:end]) or {}
        except yaml.YAMLError:
            return {}

    @staticmethod
    def _extract_wiki_links(content: str) -> list[str]:
        """Extract [[wiki-link]] targets from content."""
        return re.findall(r"\[\[([^\]]+)\]\]", content)

    # ------------------------------------------------------------------
    # Lint checks
    # ------------------------------------------------------------------

    def _find_orphans(self, pages: dict[str, dict]) -> list[LintIssue]:
        """Pages not linked to from any other page."""
        # Collect all entity names that are linked to
        linked_names: set[str] = set()
        for info in pages.values():
            for link in info["links"]:
                linked_names.add(link.lower())

        # Read index.md for listed entities
        index_path = self.wiki_root / "index.md"
        indexed_names: set[str] = set()
        if index_path.exists():
            index_content = index_path.read_text(encoding="utf-8")
            indexed_names = {n.lower() for n in self._extract_wiki_links(index_content)}

        issues = []
        for rel, info in pages.items():
            if rel.startswith("queries/"):
                continue  # Query results are allowed to be orphans
            name = info["entity_name"].lower()
            if name not in linked_names and name not in indexed_names:
                issues.append(LintIssue(
                    severity="warning",
                    category="orphan",
                    page=rel,
                    detail=f"No incoming links to '{info['entity_name']}'",
                ))
        return issues

    def _find_broken_links(self, pages: dict[str, dict]) -> list[LintIssue]:
        """[[links]] pointing to pages that don't exist."""
        known_names = {info["entity_name"].lower() for info in pages.values()}

        issues = []
        for rel, info in pages.items():
            for link in info["links"]:
                if link.lower() not in known_names:
                    issues.append(LintIssue(
                        severity="error",
                        category="broken_link",
                        page=rel,
                        detail=f"Broken link to [[{link}]]",
                        fixable=True,
                    ))
        return issues

    def _find_stale_pages(
        self, pages: dict[str, dict], *, days: int = 30
    ) -> list[LintIssue]:
        """Pages not updated in *days* or more."""
        cutoff = time.time() - (days * 86400)
        issues = []
        for rel, info in pages.items():
            fm = info["frontmatter"]
            last_updated = fm.get("last_updated")
            if last_updated is None:
                continue
            try:
                if isinstance(last_updated, str):
                    from datetime import datetime
                    ts = datetime.fromisoformat(last_updated).timestamp()
                elif isinstance(last_updated, (int, float)):
                    ts = float(last_updated)
                else:
                    continue
                if ts < cutoff:
                    issues.append(LintIssue(
                        severity="info",
                        category="stale",
                        page=rel,
                        detail=f"Last updated {int((time.time() - ts) / 86400)} days ago",
                    ))
            except (ValueError, TypeError):
                continue
        return issues

    def _find_potential_duplicates(self, pages: dict[str, dict]) -> list[LintIssue]:
        """Pages likely referring to the same entity."""
        names = [(rel, info["entity_name"]) for rel, info in pages.items()]
        issues = []
        seen: set[tuple[str, str]] = set()

        for i, (rel_a, name_a) in enumerate(names):
            norm_a = re.sub(r"\s+", " ", name_a.lower().strip())
            for rel_b, name_b in names[i + 1:]:
                norm_b = re.sub(r"\s+", " ", name_b.lower().strip())
                pair = (min(rel_a, rel_b), max(rel_a, rel_b))
                if pair in seen:
                    continue
                # Check if one is a substring of the other
                if norm_a in norm_b or norm_b in norm_a:
                    seen.add(pair)
                    issues.append(LintIssue(
                        severity="warning",
                        category="duplicate",
                        page=rel_a,
                        detail=f"Possible duplicate of '{name_b}' ({rel_b})",
                    ))
        return issues

    def _find_missing_crossrefs(self, pages: dict[str, dict]) -> list[LintIssue]:
        """Entities mentioned in text but not linked via [[]]."""
        issues = []
        all_names = {
            info["entity_name"]: rel for rel, info in pages.items()
        }

        for rel, info in pages.items():
            content_lower = info["content"].lower()
            linked_lower = {l.lower() for l in info["links"]}
            own_name = info["entity_name"].lower()

            for name, name_rel in all_names.items():
                if name_rel == rel:
                    continue  # Skip self
                if name.lower() in content_lower and name.lower() not in linked_lower and name.lower() != own_name:
                    issues.append(LintIssue(
                        severity="info",
                        category="missing_crossref",
                        page=rel,
                        detail=f"Mentions '{name}' but no [[{name}]] link",
                        fixable=True,
                    ))
        return issues

    def _check_category_balance(self, pages: dict[str, dict]) -> list[LintIssue]:
        """Warn if category distribution is heavily skewed."""
        counts: dict[str, int] = {}
        for rel in pages:
            cat = rel.split("/")[0]
            counts[cat] = counts.get(cat, 0) + 1

        total = sum(counts.values())
        if total < 5:
            return []

        issues = []
        for cat, count in counts.items():
            ratio = count / total
            if ratio > 0.6:
                issues.append(LintIssue(
                    severity="info",
                    category="imbalance",
                    page=f"{cat}/",
                    detail=f"Category '{cat}' has {ratio:.0%} of all pages ({count}/{total})",
                ))
        return issues

    # ------------------------------------------------------------------
    # Auto-fix helpers
    # ------------------------------------------------------------------

    def _fix_broken_link(self, issue: LintIssue) -> None:
        """Remove a broken [[link]] from a page."""
        match = re.search(r"Broken link to \[\[(.+)\]\]", issue.detail)
        if not match:
            return
        link_target = match.group(1)
        page_path = self.wiki_root / issue.page
        if not page_path.exists():
            return
        content = page_path.read_text(encoding="utf-8")
        content = content.replace(f"[[{link_target}]]", link_target)
        page_path.write_text(content, encoding="utf-8")

    def _fix_missing_crossref(self, issue: LintIssue) -> None:
        """Add a [[link]] for a mentioned entity."""
        match = re.search(r"Mentions '(.+)' but no", issue.detail)
        if not match:
            return
        entity = match.group(1)
        page_path = self.wiki_root / issue.page
        if not page_path.exists():
            return
        content = page_path.read_text(encoding="utf-8")

        # Add to Related section if it exists
        if "## Related" in content:
            content = content.replace(
                "## Related",
                f"## Related\n- [[{entity}]]",
            )
        else:
            content += f"\n\n## Related\n- [[{entity}]]\n"
        page_path.write_text(content, encoding="utf-8")

    def _append_log(self, message: str) -> None:
        """Append to wiki/log.md."""
        log_path = self.wiki_root / "log.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        entry = f"- [{timestamp}] SENTINEL WikiLint: {message}\n"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
