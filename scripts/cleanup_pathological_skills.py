#!/usr/bin/env python3
"""Find (and optionally delete) skill files whose filename slug doesn't
match the slug of the LLM-generated ``name:`` in their frontmatter.

Pre-PR-#20, :class:`prometheus.learning.skill_creator.SkillCreator` derived
filenames from the raw user message rather than the LLM's ``name:``
frontmatter. The result was a population of files in the shape
``<long-run-on-user-message-truncated-mid-word>-.md`` (the trailing
dash a separate strip-before-truncate bug in ``_slugify``) whose
*contents* were perfectly good skills (correct ``name:`` and body),
but whose *filenames* leaked raw user prompts onto disk.

This script identifies those files. The definition of "pathological" is
purely structural: ``_slugify(file.stem) != _slugify(name_field)``.
Files where the two match (the post-PR-#20 invariant) are left alone.

USAGE
-----

Dry-run (default — prints the divergent files, doesn't touch anything):

    python3 scripts/cleanup_pathological_skills.py
    python3 scripts/cleanup_pathological_skills.py ~/.prometheus/skills/auto

Delete the divergent files (interactive confirmation required):

    python3 scripts/cleanup_pathological_skills.py --delete

Skip the confirmation prompt (use carefully, e.g. in a one-shot script):

    python3 scripts/cleanup_pathological_skills.py --delete --yes

EXIT CODES
----------

  0  no divergent files found
  1  divergent files found (dry-run) OR delete was cancelled
  2  delete completed (some files removed)
  3  argument or filesystem error

Backup files written by SkillRefiner follow the pattern
``<slug>.bak-<unix-ts>.md`` and are NOT flagged by default — their
filename slug intentionally diverges from frontmatter. Use
``--include-bak`` to inspect them anyway.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Use the same slugifier as the production code. Importing keeps the
# definition single-sourced; if PR #20's slug rule ever drifts, this
# tool's "divergent" definition tracks automatically.
try:
    from prometheus.learning.skill_creator import _slugify
except ImportError:
    # Allow standalone use without a configured PYTHONPATH (e.g. from a
    # systemd timer or cron job that hasn't set up the package).
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
        return slug.strip("-")[:64].rstrip("-")


_DEFAULT_AUTO_DIR = Path.home() / ".prometheus" / "skills" / "auto"
_BAK_PATTERN = re.compile(r"\.bak-\d+$")


def _extract_name(content: str) -> str | None:
    """Mirror of SkillCreator._extract_name (kept inline so the tool
    works even if the import fails for whatever reason)."""
    in_fm = False
    for raw in content.splitlines():
        line = raw.strip()
        if line == "---":
            in_fm = not in_fm
            continue
        if in_fm and line.startswith("name:"):
            value = line.split(":", 1)[1].strip().strip("'\"")
            return value or None
    return None


def _is_bak_file(stem: str) -> bool:
    """``keep-going.bak-1779209848`` → True. ``keep-going`` → False."""
    return bool(_BAK_PATTERN.search(stem))


def scan(
    skills_dir: Path,
    *,
    include_bak: bool = False,
) -> list[tuple[Path, str, str | None]]:
    """Return list of (path, filename_slug, name_slug-or-None) for divergent files.

    A file is divergent iff ``_slugify(file.stem) != _slugify(name_field)``,
    where ``name_field`` is the LLM's frontmatter ``name:``. Files with no
    parseable ``name:`` are also returned (``name_slug == None``) so the
    operator can decide what to do with them.

    Backup files (``*.bak-<ts>``) are skipped unless ``include_bak`` is True.
    """
    divergent: list[tuple[Path, str, str | None]] = []
    for md in sorted(skills_dir.glob("*.md")):
        stem = md.stem
        if not include_bak and _is_bak_file(stem):
            continue

        try:
            content = md.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  [error] could not read {md}: {exc}", file=sys.stderr)
            continue

        name = _extract_name(content)
        file_slug = _slugify(stem)
        name_slug = _slugify(name) if name else None

        if name_slug is None or file_slug != name_slug:
            divergent.append((md, file_slug, name_slug))

    return divergent


def _format_divergence(
    rows: list[tuple[Path, str, str | None]],
) -> str:
    """Render the divergence list for human review."""
    if not rows:
        return "  (none)"
    lines: list[str] = []
    width = max(len(p.name) for p, _, _ in rows)
    for path, file_slug, name_slug in rows:
        target = name_slug if name_slug is not None else "<no frontmatter name:>"
        lines.append(f"  {path.name:<{width}}  ->  {target}")
    return "\n".join(lines)


def _confirm(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # newline after ^C / ^D
        return False
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cleanup_pathological_skills",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "skills_dir",
        nargs="?",
        default=str(_DEFAULT_AUTO_DIR),
        help=(
            f"Directory to scan (default: {_DEFAULT_AUTO_DIR}). "
            f"Pre-PR-#20 pathological files live in ``~/.prometheus/skills/auto/``."
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the divergent files. Requires confirmation unless --yes.",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt for --delete.",
    )
    parser.add_argument(
        "--include-bak",
        action="store_true",
        help="Include SkillRefiner backup files (*.bak-<ts>.md). Off by default.",
    )

    args = parser.parse_args(argv)
    skills_dir = Path(args.skills_dir).expanduser()

    if not skills_dir.is_dir():
        print(f"error: {skills_dir} is not a directory", file=sys.stderr)
        return 3

    print(f"Scanning {skills_dir}...")
    rows = scan(skills_dir, include_bak=args.include_bak)

    if not rows:
        print("No pathological skill files found.")
        return 0

    print(f"\nFound {len(rows)} divergent file(s):")
    print("  filename  ->  expected slug from frontmatter ``name:``")
    print(_format_divergence(rows))
    print()

    if not args.delete:
        # Dry-run mode — list and exit. Non-zero exit so a CI wrapper can
        # detect "pre-fix files still on disk."
        print("(dry-run; pass --delete to remove)")
        return 1

    # --delete path
    if not args.yes:
        print(
            f"About to DELETE {len(rows)} file(s) from {skills_dir}. "
            f"This cannot be undone."
        )
        if not _confirm("Type 'yes' to confirm: "):
            print("Cancelled. No files deleted.")
            return 1

    removed = 0
    for path, _, _ in rows:
        try:
            path.unlink()
            print(f"  removed {path}")
            removed += 1
        except OSError as exc:
            print(f"  [error] could not remove {path}: {exc}", file=sys.stderr)

    print(f"\nRemoved {removed} of {len(rows)} file(s).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
