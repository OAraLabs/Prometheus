"""Read-only access to the BRAIN VAULT (~/brain-vault).

Two tools, no writes, no embeddings, no vector store:

* ``vault_search`` — ranked search over the vault's ``wiki/`` tree, returning
  page paths plus the matching context.
* ``vault_read``   — return one page's contents.

WHAT THE BRAIN VAULT IS. A git repo of compiled knowledge from Will's full
claude.ai and Claude Code history — 95 wiki pages over ~340 raw source files.
It is NOT the Prometheus wiki (``get_wiki_root()``, default
``~/.prometheus/wiki``), which is a machine-owned projection of ``memory.db``.
Two roots, two idioms, deliberately not conflated; see ``config.paths``.

READ-ONLY, STRUCTURALLY. This module imports no write API. There is no
``open(..., "w")``, no ``write_text``, no ``mkdir``, no ``shutil``, no
``unlink`` anywhere in it — not "unused", *absent*. ``is_read_only()``
returning True is a claim; ``tests/test_vault_tools.py`` parses this file's
AST and fails the build if a write call appears, which is the part that
cannot rot. The vault's own CLAUDE.md §1 makes this binding rather than
tasteful: ``raw/`` is immutable, ``wiki/memory/`` belongs to the Prometheus
compiler, ``notes/`` is human-only, and the standing instruction for any
agent is *"when in doubt: read anywhere, write nowhere."*

CONFINEMENT. Every path is resolved and required to stay under the vault
root. The escape surface is real, not theoretical: the vault contains a
``.venv`` whose ``bin/python`` is a symlink to the uv interpreter well
outside the tree, so a naive resolve-and-read would happily follow it out.
``_confine`` refuses that, and the test suite exercises the actual symlink
where it exists rather than a synthetic stand-in.

FAILS LOUD. An absent or unreadable vault root returns an error naming the
resolved path and the config key that sets it. A search that matches nothing
says so AND states what it did not cover — the ``wiki/`` scope, and the
``raw/`` tree that holds the unsummarised sources. A silent empty result is
the exact shape of the LCM ``summary_store`` bug, where every recall returned
"no results" from the day the engine landed and nothing ever said why.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.config.paths import get_vault_root
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

# Directories that are not vault CONTENT. ``.venv`` is the one that matters:
# 7.7 MB of package files whose symlinks point outside the tree. The rest are
# git plumbing. Excluded from search traversal and refused by read.
_NON_CONTENT_DIRS: frozenset[str] = frozenset({
    ".git", ".venv", ".githooks", "__pycache__", ".pytest_cache",
})

# Search is scoped to the compiled knowledge tree, per the vault's own §5
# query procedure: read the index, drill into pages, and escalate to raw/ only
# when a fact needs verifying against its source.
_SEARCH_SUBDIR = "wiki"

# The raw tree, named in the miss message so an empty result states its scope.
_RAW_SUBDIR = "raw"

# Index entry: ``- [Name](sources/cat/Page.md) — summary``. Same format the
# Prometheus WikiQueryTool already parses.
_INDEX_ENTRY = re.compile(r"^- \[(.+?)\]\((.+?)\)\s*(?:—\s*(.*))?$")

_WORD = re.compile(r"[A-Za-z0-9_]{3,}")

# Chronological journals, not knowledge pages. ``wiki/log.md`` is ~80 KB of
# every ingest, compile, lint and finding ever recorded, so on a raw term count
# it outscores the entity page that actually answers the question — observed
# live: a query matching Standing-Principles at score 2 matched log.md at 61.
# Index-first ordering hides this whenever the index matches, which is exactly
# what would have let it ship unnoticed. Demoted below real pages rather than
# excluded: the log genuinely holds facts, it just should never outrank the
# page a fact was compiled INTO.
_JOURNAL_NAMES: frozenset[str] = frozenset({"log.md", "index.md"})


def _is_journal(rel: str) -> bool:
    name = Path(rel).name
    return name in _JOURNAL_NAMES or name.startswith("COMPILE-REPORT")

# Payload ceiling, same shape and reasoning as wiki_query: bound by SIZE, not
# page count, so a few large pages cannot blow past the working window.
_MAX_RESULT_TOKENS = 12_000
_CHARS_PER_TOKEN = 4
_MAX_RESULT_CHARS = _MAX_RESULT_TOKENS * _CHARS_PER_TOKEN

_MAX_PAGES = 8
_CONTEXT_LINES_PER_PAGE = 4
_CONTEXT_LINE_CHARS = 300


class VaultRootUnavailable(Exception):
    """The brain vault root is missing or unreadable.

    Carries the resolved path so the message can name it — "not found" without
    the path it looked in is the kind of error that costs an hour.
    """

    def __init__(self, root: Path, why: str) -> None:
        super().__init__(why)
        self.root = root
        self.why = why

    def as_result(self) -> ToolResult:
        return ToolResult(
            output=(
                f"Brain vault unavailable: {self.why}\n"
                f"  resolved root: {self.root}\n"
                f"  set it with the `vault.root` config key, the PROMETHEUS_VAULT "
                f"environment variable, or place the vault at ~/brain-vault.\n"
                f"(The brain vault is the brain-vault repo — NOT the Prometheus "
                f"wiki, which is a separate root.)"
            ),
            is_error=True,
        )


def _require_root() -> Path:
    """Return the vault root, or raise :class:`VaultRootUnavailable`."""
    root = get_vault_root()
    try:
        resolved = root.resolve()
    except OSError as exc:
        raise VaultRootUnavailable(root, f"path could not be resolved ({exc})") from exc
    if not resolved.exists():
        raise VaultRootUnavailable(root, "no such directory")
    if not resolved.is_dir():
        raise VaultRootUnavailable(root, "path exists but is not a directory")
    try:
        next(resolved.iterdir(), None)
    except OSError as exc:
        raise VaultRootUnavailable(root, f"directory is not readable ({exc})") from exc
    return resolved


def _is_excluded(rel: Path) -> bool:
    """True if any component names a non-content directory."""
    return any(part in _NON_CONTENT_DIRS for part in rel.parts)


def _confine(root: Path, candidate: Path) -> Path:
    """Resolve *candidate* and require it to stay inside *root*.

    ``Path.resolve()`` follows symlinks, so resolving FIRST and comparing
    after is what makes this a symlink guard and not merely a ``..`` guard —
    a link inside the tree pointing out of it resolves to its target, and the
    containment check then fails. Checking containment on the unresolved path
    would pass and then read the target anyway.

    Raises ``ValueError`` (the caller renders it), never returns an outside
    path.
    """
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"path could not be resolved ({exc})") from exc
    if not resolved.is_relative_to(root):
        raise ValueError(
            "path escapes the brain vault root "
            f"({resolved} is outside {root}) — refused"
        )
    rel = resolved.relative_to(root)
    if _is_excluded(rel):
        raise ValueError(
            f"path is inside a non-content directory ({rel}) — the brain vault "
            f"tools read compiled knowledge and sources, not git or venv internals"
        )
    return resolved


def _count_raw_sources(root: Path) -> int:
    """Number of files under ``raw/``. Counted, never hardcoded.

    The miss message quotes this figure, and a hardcoded one becomes a lie the
    first time an ingest runs — the same class as a config key asserting a
    control nobody maintains. Only walked on the miss path.
    """
    raw = root / _RAW_SUBDIR
    if not raw.is_dir():
        return 0
    try:
        return sum(1 for p in raw.rglob("*") if p.is_file())
    except OSError:
        return 0


def _keywords(query: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(query or "")}


def _inside(root: Path, candidate: Path) -> Path | None:
    """``_confine`` in filter form: the resolved path, or ``None`` to skip.

    SEARCH needs this as much as read does, and that was not obvious enough to
    get it right first time: ``rglob`` yields symlinks, and reading one
    followed it straight out of the vault into a file the tool then printed.
    Index entries need it too — ``index.md`` is just a file, and an entry of
    ``[x](../../../etc/passwd)`` is a traversal with a friendly label.
    """
    try:
        return _confine(root, candidate)
    except ValueError:
        return None


def _index_candidates(wiki: Path, words: set[str]) -> list[tuple[int, str, str]]:
    """Score ``index.md`` entries by keyword overlap → (score, name, relpath).

    Index-first, the way the vault's own §5 query procedure and the compile
    agents navigate: *"Read wiki/index.md first. Drill into linked pages."*
    """
    index = wiki / "index.md"
    if not index.is_file():
        return []
    try:
        text = index.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    scored: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        m = _INDEX_ENTRY.match(line.strip())
        if not m:
            continue
        name, rel, summary = m.group(1), m.group(2), m.group(3) or ""
        overlap = len(words & _keywords(f"{name} {summary}"))
        if overlap:
            scored.append((overlap, name, rel))
    scored.sort(key=lambda t: -t[0])
    return scored


def _content_matches(
    root: Path, wiki: Path, words: set[str]
) -> list[tuple[int, str, str]]:
    """Full sweep of ``wiki/`` scoring pages by keyword hits in their body.

    The index carries a one-line summary per page, so a term that appears only
    in a page BODY is invisible to :func:`_index_candidates`. Without this
    fallback the tool would answer "nothing found" for facts the vault plainly
    contains — a wrong answer that looks like an authoritative one.
    """
    scored: list[tuple[int, str, str]] = []
    for path in sorted(wiki.rglob("*.md")):
        rel = path.relative_to(wiki)
        if _is_excluded(rel):
            continue
        # rglob yields symlinks. Reading one followed it out of the vault and
        # printed the target's contents — caught by
        # test_a_symlink_escape_is_not_reachable_through_search_either.
        if _inside(root, path) is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        low = text.lower()
        hits = sum(low.count(w) for w in words)
        if hits:
            scored.append((hits, path.stem, str(rel)))
    # Entity pages first, then journals; by score within each group.
    scored.sort(key=lambda t: (_is_journal(t[2]), -t[0]))
    return scored


_STEM_MIN = 4


def _line_matches(line_low: str, words: set[str]) -> bool:
    """True if *line_low* matches any query word.

    Substring in EITHER direction on a shared prefix, because natural queries
    and page text disagree about inflection constantly: a query for "checks"
    must find "each check actually proves", and a query for "check" must find
    "checks". A plain ``w in line`` is directional and silently misses half of
    those — which surfaces as "the vault has nothing on that", the failure
    mode this tool exists to avoid.
    """
    if any(w in line_low for w in words):
        return True
    tokens = _WORD.findall(line_low)
    for w in words:
        if len(w) < _STEM_MIN:
            continue
        stem = w[:_STEM_MIN]
        if any(t.startswith(stem) for t in tokens):
            return True
    return False


def _context_for(page: Path, words: set[str]) -> list[str]:
    """Matching lines from *page*, padded with its opening prose.

    Padding matters: an index hit means the query matched the page's TITLE or
    SUMMARY, so the only "matching" line in the body is often the heading
    repeated back. Returning that alone is technically a match and practically
    useless — the caller learns the page exists and nothing about what it
    says. So thin results are topped up with the first substantive lines,
    skipping YAML frontmatter and headings.
    """
    try:
        text = page.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = text.splitlines()
    # Skip a leading YAML frontmatter block.
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break

    body = [ln.strip() for ln in lines[start:] if ln.strip()]

    out: list[str] = []
    for stripped in body:
        if _line_matches(stripped.lower(), words):
            out.append(stripped[:_CONTEXT_LINE_CHARS])
            if len(out) >= _CONTEXT_LINES_PER_PAGE:
                return out

    for stripped in body:
        if len(out) >= _CONTEXT_LINES_PER_PAGE:
            break
        if stripped.startswith("#") or stripped in out:
            continue
        out.append(stripped[:_CONTEXT_LINE_CHARS])
    return out


# ---------------------------------------------------------------------------
# vault_search
# ---------------------------------------------------------------------------

class VaultSearchInput(BaseModel):
    """Arguments for vault_search."""

    query: str = Field(
        description="What to look for in the brain vault's compiled knowledge.",
    )


class VaultSearchTool(BaseTool):
    """Search the brain vault's wiki tree."""

    name = "vault_search"
    description = (
        "Search the BRAIN VAULT — Will's second brain, compiled "
        "from his full claude.ai and Claude Code history: projects, decisions, "
        "people, companies and standing engineering principles. Returns "
        "matching page paths with context; read a full page with vault_read. "
        "Searches the vault's compiled wiki/ tree. This is NOT the Prometheus "
        "wiki (wiki_query) — different knowledge, different root."
    )
    input_model = VaultSearchInput
    example_call = {"query": "standing principles verification"}

    def is_read_only(self, arguments: VaultSearchInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: VaultSearchInput, context: ToolExecutionContext
    ) -> ToolResult:
        del context
        try:
            root = _require_root()
        except VaultRootUnavailable as exc:
            return exc.as_result()

        wiki = root / _SEARCH_SUBDIR
        if not wiki.is_dir():
            return ToolResult(
                output=(
                    f"Brain vault has no {_SEARCH_SUBDIR}/ tree at {wiki} — "
                    f"the root resolved ({root}) but its compiled knowledge "
                    f"directory is missing."
                ),
                is_error=True,
            )

        words = _keywords(arguments.query)
        if not words:
            return ToolResult(
                output=(
                    "Empty query — vault_search needs at least one word of "
                    "three or more characters."
                ),
                is_error=True,
            )

        seen: set[str] = set()
        ranked: list[tuple[int, str, str]] = []
        for score, name, rel in _index_candidates(wiki, words):
            # An index entry is an untrusted path: index.md is a file like any
            # other, and "[x](../../../etc/passwd)" is a traversal wearing a
            # label. Confine before the page is ever opened.
            if rel not in seen and _inside(root, wiki / rel) is not None:
                seen.add(rel)
                ranked.append((score, name, rel))
        for score, name, rel in _content_matches(root, wiki, words):
            if rel not in seen:
                seen.add(rel)
                ranked.append((score, name, rel))

        if not ranked:
            # THE VISIBLE MISS. An empty result that names its own scope is
            # honest; one that just says "nothing found" invites the reader to
            # conclude the vault has nothing on the subject, when in fact the
            # unsummarised sources were never looked at.
            raw_n = _count_raw_sources(root)
            return ToolResult(output=(
                f"No matches in the brain vault's {_SEARCH_SUBDIR}/ tree for "
                f"{arguments.query!r}.\n"
                f"\n"
                f"SCOPE OF THIS SEARCH: the compiled wiki/ pages only. It did "
                f"NOT search {_RAW_SUBDIR}/, which holds {raw_n} unsummarised "
                f"source files (conversation exports, specs, converted docs).\n"
                f"To go further: read a related page and follow its `sources:` "
                f"frontmatter, or vault_read a path under {_RAW_SUBDIR}/ "
                f"directly. That is the vault's own escalation procedure — the "
                f"wiki is the compiled view, raw/ is the record it came from."
            ))

        lines = [
            f"Brain vault — {len(ranked)} page(s) matching {arguments.query!r} "
            f"(showing up to {_MAX_PAGES}):",
            "",
        ]
        budget = _MAX_RESULT_CHARS
        shown = 0
        for score, name, rel in ranked[:_MAX_PAGES]:
            block = [f"## {name}  [{_SEARCH_SUBDIR}/{rel}]  (score {score})"]
            block += [f"    {c}" for c in _context_for(wiki / rel, words)]
            block.append("")
            text = "\n".join(block)
            if len(text) > budget:
                break
            budget -= len(text)
            lines.append(text)
            shown += 1

        if shown < len(ranked):
            lines.append(
                f"({len(ranked) - shown} further match(es) not shown — refine "
                f"the query, or vault_read a path above for the full page.)"
            )
        return ToolResult(output="\n".join(lines))


# ---------------------------------------------------------------------------
# vault_read
# ---------------------------------------------------------------------------

class VaultReadInput(BaseModel):
    """Arguments for vault_read."""

    path: str = Field(
        description=(
            "Path relative to the brain vault root, e.g. "
            "'wiki/sources/concepts/Standing-Principles.md' or "
            "'raw/claude-chats/2026-04-24-prometheus3.md'."
        ),
    )


class VaultReadTool(BaseTool):
    """Read one page from the brain vault."""

    name = "vault_read"
    description = (
        "Read a file from the BRAIN VAULT by its vault-relative "
        "path — compiled pages under wiki/, original sources under raw/, "
        "human notes under notes/. Read-only. Use vault_search first to find "
        "the path."
    )
    input_model = VaultReadInput
    example_call = {"path": "wiki/index.md"}

    def is_read_only(self, arguments: VaultReadInput) -> bool:
        del arguments
        return True

    async def execute(
        self, arguments: VaultReadInput, context: ToolExecutionContext
    ) -> ToolResult:
        del context
        try:
            root = _require_root()
        except VaultRootUnavailable as exc:
            return exc.as_result()

        raw_path = (arguments.path or "").strip()
        if not raw_path:
            return ToolResult(
                output="Empty path — vault_read needs a vault-relative path.",
                is_error=True,
            )

        candidate = Path(raw_path)
        if candidate.is_absolute():
            # Accept an absolute path only if it is already inside the vault;
            # _confine decides. Anything else is refused by the same check that
            # refuses '../'.
            target = candidate
        else:
            target = root / candidate

        try:
            resolved = _confine(root, target)
        except ValueError as exc:
            return ToolResult(output=f"Refused: {exc}", is_error=True)

        if not resolved.exists():
            return ToolResult(
                output=(
                    f"No such file in the brain vault: {raw_path}\n"
                    f"  (looked in {root})\n"
                    f"  Use vault_search to find the right path."
                ),
                is_error=True,
            )
        if resolved.is_dir():
            try:
                entries = sorted(
                    p.name + ("/" if p.is_dir() else "")
                    for p in resolved.iterdir()
                    if not _is_excluded(p.relative_to(root))
                )
            except OSError as exc:
                return ToolResult(
                    output=f"Directory is not readable: {raw_path} ({exc})",
                    is_error=True,
                )
            listing = "\n".join(f"  {e}" for e in entries) or "  (empty)"
            return ToolResult(
                output=f"{raw_path} is a directory:\n{listing}"
            )

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(
                output=f"Could not read {raw_path}: {exc}", is_error=True,
            )

        rel = resolved.relative_to(root)
        if len(text) > _MAX_RESULT_CHARS:
            text = (
                text[:_MAX_RESULT_CHARS]
                + f"\n\n[truncated at {_MAX_RESULT_CHARS} chars — "
                f"the full page is on disk at {rel}]"
            )
        return ToolResult(output=f"# brain vault: {rel}\n\n{text}")
