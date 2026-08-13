# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/prompts/claudemd.py
# License: MIT
# Modified: renamed from CLAUDE.md discovery to PROMETHEUS.md;
#           looks for PROMETHEUS.md and .prometheus/ directories
#           extended to recognize multi-agent convention files (HERMES.md,
#           CLAUDE.md, AGENTS.md, .cursorrules, .windsurfrules) and
#           stack multiple project files instead of first-match-wins

"""Project instruction file discovery and loading.

Walks from the working directory upward, collecting project instruction files
from any recognized agent convention (PROMETHEUS.md, HERMES.md, CLAUDE.md,
AGENTS.md, .cursorrules, etc.) and per-directory rules from
``.prometheus/rules/*.md``.

When ``stack`` is True (default), all found files are loaded with directory
labels. When ``stack`` is False, only the first match wins (legacy behavior).
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
    ".cursorrules",
    ".windsurfrules",
]

# Prometheus-specific per-directory rules directory.
_PROMETHEUS_RULES_DIR = ".prometheus/rules"


def discover_project_files(
    cwd: str | Path,
    stack: bool = True,
    exclude: Sequence[str] = (),
) -> list[Tuple[Path, str]]:
    """Walk from cwd upward to the filesystem root.

    At each level collect:
      - First recognized convention file (PROMETHEUS.md, HERMES.md, etc.)
      - .prometheus/rules/*.md

    When *stack* is True (default), collects files from ALL directory levels.
    When *stack* is False, returns only the first match (legacy behavior).

    *exclude* drops filenames from :data:`CONVENTION_FILES` for this call.
    It exists so a convention name that some OTHER config already gates —
    ``AGENTS.md`` under ``bootstrap.load_agents`` — cannot come back in
    through discovery after the user turned it off.

    Returns list of (path, label) tuples in specificity order:
    most specific (deepest) first. Label is the directory name for context.
    """
    current = Path(cwd).resolve()
    results: list[Tuple[Path, str]] = []
    seen: set[Path] = set()
    skip = {name.casefold() for name in exclude}

    for directory in [current, *current.parents]:
        found_at_level = False

        # --- Convention files (priority order, one per directory level) ---
        for fname in CONVENTION_FILES:
            if fname.casefold() in skip:
                continue
            candidate = directory / fname
            if candidate.exists():
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
    stack: bool = True,
    exclude: Sequence[str] = (),
) -> str | None:
    """Load discovered project instruction files into one prompt section.

    When *stack* is True, includes all files found walking upward, each
    labeled with the directory it came from. When False, only the first
    match is included. *exclude* drops convention filenames — see
    :func:`discover_project_files`.

    Returns a formatted string ready for injection into the system prompt,
    or ``None`` if no files were found.
    """
    files = discover_project_files(cwd, stack=stack, exclude=exclude)
    if not files:
        return None

    lines = ["# Project Instructions"]
    for path, label in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n...[truncated]..."
        header = f"## {path}"
        if label and label != path.name:
            header = f"## {path} (from {label}/)"
        lines.extend(["", header, "```md", content.strip(), "```"])
    return "\n".join(lines)


def load_prometheus_md_prompt(
    cwd: str | Path, *, max_chars_per_file: int = 12000
) -> str | None:
    """Legacy interface — loads first-match-wins, no stacking."""
    return load_project_files_prompt(cwd, max_chars_per_file=max_chars_per_file, stack=False)
