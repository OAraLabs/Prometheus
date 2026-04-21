# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/prompts/claudemd.py
# License: MIT
# Modified: renamed from CLAUDE.md discovery to PROMETHEUS.md;
#           looks for PROMETHEUS.md and .prometheus/ directories

"""PROMETHEUS.md discovery and loading.

Walks from the working directory upward, collecting PROMETHEUS.md instruction
files and per-directory rules from ``.prometheus/rules/*.md``.  This is the
Prometheus equivalent of the CLAUDE.md convention.
"""

from __future__ import annotations

from pathlib import Path


def discover_prometheus_md_files(cwd: str | Path) -> list[Path]:
    """Discover relevant PROMETHEUS.md instruction files from cwd upward.

    Checks each directory for:
      - ``PROMETHEUS.md``
      - ``.prometheus/PROMETHEUS.md``
      - ``.prometheus/rules/*.md``

    Returns paths in order from most-specific (cwd) to least-specific (root).
    """
    current = Path(cwd).resolve()
    results: list[Path] = []
    seen: set[Path] = set()

    for directory in [current, *current.parents]:
        for candidate in (
            directory / "PROMETHEUS.md",
            directory / ".prometheus" / "PROMETHEUS.md",
        ):
            if candidate.exists() and candidate not in seen:
                results.append(candidate)
                seen.add(candidate)

        rules_dir = directory / ".prometheus" / "rules"
        if rules_dir.is_dir():
            for rule in sorted(rules_dir.glob("*.md")):
                if rule not in seen:
                    results.append(rule)
                    seen.add(rule)

        if directory.parent == directory:
            break

    return results


def load_prometheus_md_prompt(
    cwd: str | Path, *, max_chars_per_file: int = 12000
) -> str | None:
    """Load discovered PROMETHEUS.md instruction files into one prompt section.

    Returns a formatted string ready for injection into the system prompt,
    or ``None`` if no files were found.
    """
    files = discover_prometheus_md_files(cwd)
    if not files:
        return None

    lines = ["# Project Instructions"]
    for path in files:
        content = path.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n...[truncated]..."
        lines.extend(["", f"## {path}", "```md", content.strip(), "```"])
    return "\n".join(lines)
