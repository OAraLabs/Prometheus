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

    # Category -> fixer(self, issue). EMPTY by design (SPRINT MEMORY-2):
    # links/crossrefs are owned solely by the WikiCompiler (fact-derived,
    # word-boundary, page-gated), so lint is detection-only and mutates
    # nothing. Any future fixer registered here must not touch compile-owned
    # state and is protected by the convergence invariant in auto_fix().
    _FIXERS: dict = {}

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

    def auto_fix(self, result: LintResult | None = None, *, max_passes: int = 5) -> int:
        """Apply safe fixes to a fixpoint, with a fail-loud monotonic guard.

        Lint no longer mutates links/crossrefs — the WikiCompiler is their sole
        writer — so ``_FIXERS`` is empty and this is a no-op today. The loop and
        invariant are retained as a regression guard: across passes the total
        issue count must strictly DECREASE while fixable issues remain. A pass
        that increases the count, stalls (fixable issues remain but none could
        be applied), or exhausts ``max_passes`` halts and alarms rather than
        spinning. (SPRINT MEMORY-2 2b)

        ``result`` is ignored (kept for back-compat) — the loop re-lints each
        pass so its convergence check sees live counts.
        """
        total_fixed = 0
        prev_count: int | None = None
        for _ in range(max_passes):
            issues = self.lint().issues
            fixable = [i for i in issues if i.fixable]
            if not fixable:
                break  # converged — nothing left to fix
            count = len(issues)
            if prev_count is not None and count >= prev_count:
                self._halt(
                    f"issue count {prev_count} -> {count} with "
                    f"{len(fixable)} fixable remaining (not strictly decreasing)"
                )
                return total_fixed
            prev_count = count
            applied = 0
            for issue in fixable:
                fixer = self._FIXERS.get(issue.category)
                if fixer is None:
                    continue
                try:
                    fixer(self, issue)
                    applied += 1
                except Exception:
                    log.exception("WikiLinter: failed to fix %s", issue)
            if applied == 0:
                self._halt(f"{len(fixable)} fixable issue(s) but none applied (stalled)")
                return total_fixed
            total_fixed += applied
        else:
            if any(i.fixable for i in self.lint().issues):
                self._halt(f"exceeded {max_passes} passes with fixable issues remaining")
        if total_fixed:
            self._append_log(f"Auto-fixed {total_fixed} issues")
        return total_fixed

    def _halt(self, reason: str) -> None:
        """Fail loud on non-convergence — never silently spin a mutating loop."""
        log.error("WikiLinter: auto_fix HALTED — %s", reason)
        try:
            self._append_log(f"HALT (non-convergent auto_fix): {reason}")
        except Exception:
            log.exception("WikiLinter: failed to write halt marker")

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
                # Match the link/file format exactly: links are written
                # ``[[<entity_name>]]`` and files are ``_safe_filename(name).md``,
                # so the stem is the correct key. The old -/_→space normalization
                # mis-flagged hyphenated/underscored names (e.g. Acme-Corp,
                # build-2.0) as broken links. (SPRINT MEMORY-2 2a-iii)
                entity_name = md_file.stem
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
    def _is_manual(info: dict) -> bool:
        """A page is manual-flagged (Phase 4b) if its frontmatter says so.

        Manual (/note) pages are exempt from the removal/degradation checks —
        orphan-sweep, stale-flag, and duplicate-prune — the same carve-out
        shape Sprint-2 gave queries/. You asserting a fact explicitly means it
        must never be swept, staled, or pruned out from under you.
        """
        return bool(info.get("frontmatter", {}).get("manual"))

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
            if rel.startswith("queries/") or self._is_manual(info):
                continue  # queries/ and manual (/note) pages may be orphans
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
        """[[links]] pointing to pages that don't exist.

        queries/ is exempt (mirrors the orphan exemption): those are
        synthesized insights whose links to non-page entities are inherent,
        not breakage. Detection-only — link state is owned by the WikiCompiler
        and is never auto-mutated here. (SPRINT MEMORY-2 2a / 2a-ii)
        """
        known_names = {info["entity_name"].lower() for info in pages.values()}

        issues = []
        for rel, info in pages.items():
            if rel.startswith("queries/"):
                continue  # synthesized insights — unresolved links are inherent
            for link in info["links"]:
                if link.lower() not in known_names:
                    issues.append(LintIssue(
                        severity="error",
                        category="broken_link",
                        page=rel,
                        detail=f"Broken link to [[{link}]]",
                        fixable=False,
                    ))
        return issues

    def _find_stale_pages(
        self, pages: dict[str, dict], *, days: int = 30
    ) -> list[LintIssue]:
        """Pages not updated in *days* or more."""
        cutoff = time.time() - (days * 86400)
        issues = []
        for rel, info in pages.items():
            if self._is_manual(info):
                continue  # manual (/note) pages are never stale-flagged
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
                    if self._is_manual(pages[rel_a]) or self._is_manual(pages[rel_b]):
                        continue  # never prune a manual (/note) page as a duplicate
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
                        fixable=False,
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

    # _fix_broken_link / _fix_missing_crossref REMOVED (SPRINT MEMORY-2 2a):
    # both mutated compile-owned link state. _fix_broken_link silently
    # corrupted queries/ (stripped synthesized-insight links) and false-
    # stripped hyphenated entity links; _fix_missing_crossref added substring-
    # noise crossrefs that compile then dropped (the oscillation). Link and
    # crossref repair is the WikiCompiler's sole responsibility.

    def _append_log(self, message: str) -> None:
        """Append to wiki/log.md."""
        log_path = self.wiki_root / "log.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        entry = f"- [{timestamp}] SENTINEL WikiLint: {message}\n"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
