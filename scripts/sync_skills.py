#!/usr/bin/env python3
"""Canonical Lane-1 skill sync + drift/leak guard.

Projects the repo's hand-authored skills (Lane 1) into the live runtime store
(``~/.prometheus/skills/``) and fails loud when the two diverge, or when a
tracked skill leaks a secret/PII pattern. Pure stdlib so the git pre-commit
hook can call it with any ``python3`` (no Prometheus import needed).

Lane model
  Lane 1  portable hand-authored skills  -> repo-canonical; this tool governs them
  Lane 2  Printing-Press installs        -> body carries ``printing-press-library``;
                                            an external registry owns it; never synced
  Lane 3  auto-generated (skills/auto/)   -> machine-owned; not top-level; never synced
  Lane 4  user-local / business-specific  -> LANE4_EXCLUDE; never tracked or synced

Usage
  sync_skills.py                 install: repo Lane-1 -> userdir (converge), guard-checked
  sync_skills.py --check         report repo<->userdir drift + lint; exit 1 if not clean
  sync_skills.py --lint          lint repo Lane-1 (working tree); exit 1 on findings
  sync_skills.py --lint-staged   lint staged skills/*.md (pre-commit); exit 1 on findings
  --repo-dir DIR / --user-dir DIR  override locations (for tests)
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Skills excluded from Lane 1 by name (user-local / business-specific).
LANE4_EXCLUDE = frozenset({"wordpress"})

# A skill is Lane 2 (Printing-Press) if its body carries this marker.
_PP_MARKER = re.compile(r"printing-press-library")

# /home/<user>/ absolute paths, except known generic placeholders.
_PLACEHOLDER_USERS = frozenset({"user", "youruser", "username", "me"})
_HOME_PATH = re.compile(r"/home/([a-z_][a-z0-9_-]*)/")

# Secret / private-infra shapes. This is the .md blind spot the main pre-commit
# hook leaves open (it skips markdown); the skill guard closes it.
_LEAK_PATTERNS = [
    ("Tailscale private IP", re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
    ("private domain", re.compile(r"capstonerecruiting|\.ts\.net\b")),
    ("OAra infra hostname", re.compile(r"[Oo][Aa]ra-(?!prometheus\b)")),
    ("Telegram bot token", re.compile(r"\b\d{8,}:[A-Za-z0-9_-]{30,}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{12,}\b")),
    ("OpenAI/Anthropic key", re.compile(r"\bsk-(?:live|test|proj|ant)[-_][A-Za-z0-9]{16,}")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9]{8,}")),
    ("SSH private key", re.compile(r"BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY")),
]


def is_printing_press(text: str) -> bool:
    """True if the skill body marks it as a Printing-Press install (Lane 2)."""
    return bool(_PP_MARKER.search(text))


def is_lane1(name: str, text: str) -> bool:
    """A top-level skill is Lane 1 unless it is Printing-Press or name-excluded."""
    return name not in LANE4_EXCLUDE and not is_printing_press(text)


def lint_text(text: str) -> list[str]:
    """Return human-readable leak findings for one skill body (empty = clean)."""
    findings: list[str] = []
    for match in _HOME_PATH.finditer(text):
        user = match.group(1)
        if user not in _PLACEHOLDER_USERS:
            findings.append(f"absolute home path /home/{user}/ — use ~/ or a placeholder")
    for label, pattern in _LEAK_PATTERNS:
        if pattern.search(text):
            findings.append(f"{label} pattern")
    out: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        if finding not in seen:
            seen.add(finding)
            out.append(finding)
    return out


def _lane1_map(skills_dir: Path) -> dict[str, str]:
    """name -> content for Lane-1 skills in a top-level skills dir (auto/ skipped)."""
    result: dict[str, str] = {}
    if not skills_dir.is_dir():
        return result
    for path in sorted(skills_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if is_lane1(path.stem, text):
            result[path.stem] = text
    return result


def default_repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "skills"


def default_user_dir() -> Path:
    return Path.home() / ".prometheus" / "skills"


def compute(repo_dir: Path, user_dir: Path):
    """Return (repo_map, to_install, to_remove, lint)."""
    repo = _lane1_map(repo_dir)
    user = _lane1_map(user_dir)
    to_install = sorted(name for name, content in repo.items() if user.get(name) != content)
    to_remove = sorted(name for name in user if name not in repo)
    lint = {name: f for name, content in repo.items() if (f := lint_text(content))}
    return repo, to_install, to_remove, lint


def _print_lint(lint: dict[str, list[str]]) -> None:
    print("Lane-1 skill lint FAILED:")
    for name in sorted(lint):
        print(f"  x {name}.md")
        for finding in lint[name]:
            print(f"      {finding}")


def cmd_install(repo_dir: Path, user_dir: Path) -> int:
    repo, to_install, to_remove, lint = compute(repo_dir, user_dir)
    if lint:
        _print_lint(lint)
        print("\nRefusing to sync a leak. Fix the skills above first.")
        return 1
    user_dir.mkdir(parents=True, exist_ok=True)
    for name in to_install:
        (user_dir / f"{name}.md").write_text(repo[name], encoding="utf-8")
    for name in to_remove:
        (user_dir / f"{name}.md").unlink(missing_ok=True)
    print(f"sync-skills: installed {len(to_install)}, removed {len(to_remove)} "
          f"(Lane-1 repo -> {user_dir})")
    return 0


def cmd_check(repo_dir: Path, user_dir: Path) -> int:
    _, to_install, to_remove, lint = compute(repo_dir, user_dir)
    ok = True
    if to_install:
        ok = False
        print(f"DRIFT: {len(to_install)} repo Lane-1 skill(s) missing/changed in userdir:")
        for name in to_install:
            print(f"  -> {name}")
    if to_remove:
        ok = False
        print(f"DRIFT: {len(to_remove)} userdir Lane-1 skill(s) not in repo:")
        for name in to_remove:
            print(f"  x  {name}")
    if lint:
        ok = False
        _print_lint(lint)
    if ok:
        print("sync-skills --check: OK (repo Lane-1 == userdir; lint clean)")
        return 0
    print("\nsync-skills --check: FAILED — run `python3 scripts/sync_skills.py` to converge.")
    return 1


def cmd_lint(repo_dir: Path) -> int:
    lint = {name: f for name, content in _lane1_map(repo_dir).items() if (f := lint_text(content))}
    if lint:
        _print_lint(lint)
        return 1
    print("sync-skills --lint: OK (no leaks in Lane-1 skills)")
    return 0


def _staged_skill_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=False,
    ).stdout
    return [f for f in out.splitlines() if f.startswith("skills/") and f.endswith(".md")]


def cmd_lint_staged() -> int:
    findings: dict[str, list[str]] = {}
    for staged in _staged_skill_files():
        name = Path(staged).stem
        blob = subprocess.run(
            ["git", "show", f":{staged}"], capture_output=True, text=True, check=False
        ).stdout
        if not is_lane1(name, blob):
            continue
        if (result := lint_text(blob)):
            findings[name] = result
    if findings:
        _print_lint(findings)
        print("\nCOMMIT BLOCKED: a tracked skill leaks a path/secret pattern.")
        print("(.md skills bypass the main hook; this is the skill-specific guard.)")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="report repo<->userdir drift + lint; no writes")
    mode.add_argument("--lint", action="store_true",
                      help="lint repo Lane-1 skills (working tree)")
    mode.add_argument("--lint-staged", action="store_true",
                      help="lint staged skills/*.md (pre-commit)")
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--user-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_dir = args.repo_dir or default_repo_dir()
    user_dir = args.user_dir or default_user_dir()
    if args.lint_staged:
        return cmd_lint_staged()
    if args.lint:
        return cmd_lint(repo_dir)
    if args.check:
        return cmd_check(repo_dir, user_dir)
    return cmd_install(repo_dir, user_dir)


if __name__ == "__main__":
    sys.exit(main())
