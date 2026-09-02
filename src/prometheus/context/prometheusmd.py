# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/prompts/claudemd.py
# License: MIT
# Modified: renamed from CLAUDE.md discovery to PROMETHEUS.md;
#           looks for PROMETHEUS.md and .prometheus/ directories
#           extended to recognize multi-agent convention files (HERMES.md,
#           CLAUDE.md, AGENTS.md, .cursorrules, .windsurfrules) and
#           stack multiple project files instead of first-match-wins;
#           GEMINI.md + .github/copilot-instructions.md; aggregate cap;
#           path-based exclusion for the identity-dir AGENTS.md

"""Project instruction file discovery and loading.

Walks from the working directory upward, collecting project instruction files
from any recognized agent convention (PROMETHEUS.md, HERMES.md, CLAUDE.md,
AGENTS.md, .cursorrules, etc.) and per-directory rules from
``.prometheus/rules/*.md``.

When ``stack`` is True (default), all found files are loaded with directory
labels. When ``stack`` is False, only the first match wins (legacy behavior).

Two caps: ``max_chars_per_file`` truncates one file, ``max_total_chars`` bounds
the whole section — the walk reaches ``/``, so without it N levels × 12 K was
unbounded. Files that do not fit are OMITTED and the omission is written into
the section, never dropped silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import List, Tuple

# All filenames that count as "project instruction" files, ordered by priority
# within each directory level. PROMETHEUS.md is primary.
CONVENTION_FILES: List[str] = [
    "PROMETHEUS.md",
    "HERMES.md",
    ".hermes.md",
    "CLAUDE.md",
    "AGENTS.md",
    "GEMINI.md",
    ".cursorrules",
    ".windsurfrules",
    ".github/copilot-instructions.md",
]

# Default aggregate cap for one "# Project Instructions" section (all levels,
# all rules files). Config: ``context.project_files_max_total_chars``.
DEFAULT_MAX_TOTAL_CHARS = 48000

# Below this many remaining chars a partial file is not worth including — it
# is omitted (and said so) rather than shipped as a useless stub.
_MIN_PARTIAL_CHARS = 1000

# Prometheus-specific per-directory rules directory.
_PROMETHEUS_RULES_DIR = ".prometheus/rules"


def discover_project_files(
    cwd: str | Path,
    stack: bool = True,
    exclude: Sequence[str] = (),
    exclude_paths: Sequence[str | Path] = (),
) -> list[Tuple[Path, str]]:
    """Walk from cwd upward to the filesystem root.

    At each level collect:
      - First recognized convention file (PROMETHEUS.md, HERMES.md, etc.)
      - .prometheus/rules/*.md

    When *stack* is True (default), collects files from ALL directory levels.
    When *stack* is False, returns only the first match (legacy behavior).

    *exclude* drops filenames from :data:`CONVENTION_FILES` for this call
    (every file of that name, at every level).

    *exclude_paths* drops specific FILES, compared by resolved path. This is
    the right tool for the identity-dir ``AGENTS.md`` (the subagent registry,
    ``~/.prometheus/AGENTS.md``): it shares a name with the project convention
    file but is a different document with its own loader and its own gate
    (``bootstrap.load_agents``). Excluding it by NAME — the previous shape —
    also threw away every repo's ``AGENTS.md`` whenever the registry was
    turned off, which conflated the two.

    Returns list of (path, label) tuples in specificity order:
    most specific (deepest) first. Label is the directory name for context.
    """
    current = Path(cwd).resolve()
    results: list[Tuple[Path, str]] = []
    seen: set[Path] = set()
    skip = {name.casefold() for name in exclude}
    skip_paths: set[Path] = set()
    for entry in exclude_paths:
        try:
            skip_paths.add(Path(entry).expanduser().resolve())
        except OSError:  # unresolvable entry — nothing to exclude
            continue

    for directory in [current, *current.parents]:
        found_at_level = False

        # --- Convention files (priority order, one per directory level) ---
        for fname in CONVENTION_FILES:
            if fname.casefold() in skip:
                continue
            candidate = directory / fname
            if candidate.is_file():
                if skip_paths and candidate.resolve() in skip_paths:
                    continue  # a different document that shares the name
                if candidate not in seen:
                    seen.add(candidate)
                    found_at_level = True
                    label = directory.name or str(directory)
                    results.append((candidate, label))
                break  # Only take the highest-priority file per level

        # --- .prometheus/rules/*.md ---
        rules_dir = directory / ".prometheus" / "rules"
        if rules_dir.is_dir():
            for rule in sorted(rules_dir.glob("*.md")):
                if rule not in seen:
                    seen.add(rule)
                    found_at_level = True
                    label = directory.name or str(directory)
                    results.append((rule, label))

        if not stack and results:
            return results  # Legacy: first match wins

        if directory.parent == directory:
            break

    return results


def discover_prometheus_md_files(cwd: str | Path) -> list[Path]:
    """Legacy interface — returns just paths, first-match-wins.

    Kept for backward compatibility with existing callers.
    """
    files = discover_project_files(cwd, stack=False)
    return [path for path, _ in files]


def load_project_files_prompt(
    cwd: str | Path,
    *,
    max_chars_per_file: int = 12000,
    max_total_chars: int | None = DEFAULT_MAX_TOTAL_CHARS,
    stack: bool = True,
    exclude: Sequence[str] = (),
    exclude_paths: Sequence[str | Path] = (),
) -> str | None:
    """Load discovered project instruction files into one prompt section.

    When *stack* is True, includes all files found walking upward, each
    labeled with the directory it came from. When False, only the first
    match is included. *exclude* / *exclude_paths* — see
    :func:`discover_project_files`.

    *max_chars_per_file* truncates any one file. *max_total_chars* bounds the
    file CONTENT across the whole section (``None`` = unbounded). Files are
    admitted in specificity order (deepest first), so under pressure the
    nearest instructions survive and the far ancestors go. A file that does
    not fit is truncated to the remaining budget when at least
    ``_MIN_PARTIAL_CHARS`` remain, otherwise omitted — and every omission is
    named in a trailing line so the model (and the operator reading the
    prompt) can see what was left out.

    Returns a formatted string ready for injection into the system prompt,
    or ``None`` if no files were found.
    """
    files = discover_project_files(cwd, stack=stack, exclude=exclude, exclude_paths=exclude_paths)
    if not files:
        return None

    lines = ["# Project Instructions"]
    remaining = max_total_chars if max_total_chars is not None else None
    omitted: list[Path] = []
    for path, label in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n...[truncated]..."
        if remaining is not None:
            if remaining <= 0 or (len(content) > remaining and remaining < _MIN_PARTIAL_CHARS):
                omitted.append(path)
                continue
            if len(content) > remaining:
                content = content[:remaining] + "\n...[truncated: aggregate project-file cap]..."
            remaining -= min(len(content), remaining)
        header = f"## {path}"
        if label and label != path.name:
            header = f"## {path} (from {label}/)"
        lines.extend(["", header, "```md", content.strip(), "```"])
    if omitted:
        names = ", ".join(str(p) for p in omitted)
        lines.extend([
            "",
            f"_{len(omitted)} project instruction file(s) omitted — aggregate cap of "
            f"{max_total_chars} chars reached: {names}_",
        ])
    return "\n".join(lines)


def load_prometheus_md_prompt(
    cwd: str | Path, *, max_chars_per_file: int = 12000
) -> str | None:
    """Legacy interface — loads first-match-wins, no stacking."""
    return load_project_files_prompt(cwd, max_chars_per_file=max_chars_per_file, stack=False)
