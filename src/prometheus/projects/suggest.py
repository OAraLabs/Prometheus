"""Which project a conversation's tool calls point at: a suggestion, never a filing (docs/design/chat-projects.md 3.6).

A port of the desktop's rules (``session-projects.ts``: ``projectOfPath``, ``projectsInToolInput``,
``tallyToolInputs``, ``suggestProject``), held to the desktop's own smoke cases, so the desktop, the phone and the
daemon answer alike:

* only paths under a home directory count (``/home/<u>/``, ``/Users/<u>/``, ``~/``);
* ``~/.prometheus/workspace/<x>`` is project ``<x>``; the rest of ``~/.prometheus`` and every other dot-directory is
  not a project;
* one container level is skipped (``~/projects/<x>``, ``~/src/<x>``, ...);
* a file directly in home or in a container is not a project; a bare directory is;
* the OS's own folders and toolchains are not projects;
* checkout suffixes fold (``prometheus-deploy`` is Prometheus, ``beacon-wt-3`` is beacon);
* a suggestion needs at least 3 tool calls that touched a project folder, one folder with at least 60% of them, and
  no tie at the top. The commonest spelling of the folder labels it.

**The daemon's one addition:** it knows its own chat projects. A path inside a project's stored folder belongs to that
project (key: its id, label: its name), and the projects root is a container like ``~/projects``. Without this every
managed folder would read as the project ``prometheus-projects``.

Pure: text in, answers out.

Source: novel code for Prometheus, 2026-11-13, ported from Beacon's ``session-projects.ts`` (same rules and thresholds).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

HINT_MIN_SHARE = 0.6
HINT_MIN_CALLS = 3

CONTAINERS = frozenset({"projects", "project", "src", "code", "repos", "repo", "dev", "work", "git", "github",
                        "developer", "workspace", "workspaces", "sites"})
NOT_PROJECTS = frozenset({"library", "downloads", "desktop", "documents", "applications", "pictures", "movies",
                          "music", "public", "models", "android-sdk", "go", "node_modules", "bin", "tmp"})

_HOME = re.compile(r"^(?:/home/[^/]+|/Users/[^/]+|~)(/.*)?$")
_FILE = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_JDK = re.compile(r"^jdk[-\d.]*$", re.IGNORECASE)
_CHECKOUT = re.compile(r"-(?:deploy|wt|worktree)(?:[-\w]*)?$", re.IGNORECASE | re.ASCII)
_PATH_IN_TEXT = re.compile(r"(?:/home/[\w.@+-]+|/Users/[\w.@+-]+|~)(?:/[\w.@+=-]+)+", re.ASCII)


@dataclass(frozen=True)
class KnownFolder:
    """A chat project's folder, for the daemon's addition."""

    id: str
    name: str
    folder: str


@dataclass
class Tally:
    counts: dict[str, int] = field(default_factory=dict)
    labels: dict[str, dict[str, int]] = field(default_factory=dict)
    calls: int = 0


@dataclass(frozen=True)
class Hint:
    key: str
    label: str
    share: float
    calls: int


def _home_relative(path: str) -> list[str] | None:
    m = _HOME.match(path)
    if not m or not m.group(1):
        return None
    return [p for p in m.group(1).split("/") if p]


def _expand(path: str) -> str:
    return os.path.expanduser(path) if path.startswith("~") else path


def project_of_path(
    path: str, *, known: Sequence[KnownFolder] = (), root: str | None = None,
) -> tuple[str, str] | None:
    """``(key, label)`` for the project folder *path* belongs to, or None."""
    if not isinstance(path, str):
        return None
    if known or root:
        full = _expand(path).rstrip("/")
        for k in known:
            folder = k.folder.rstrip("/")
            if full == folder or full.startswith(folder + "/"):
                return k.id, k.name
        if root:
            base = _expand(root).rstrip("/")
            if full.startswith(base + "/"):
                rest = [p for p in full[len(base) + 1:].split("/") if p]
                return _project_of_parts(rest, containers_skipped=True)
            if full == base:
                return None
    parts = _home_relative(path)
    if parts is None:
        return None
    return _project_of_parts(parts)


def _project_of_parts(parts: list[str], *, containers_skipped: bool = False) -> tuple[str, str] | None:
    if not parts:
        return None
    if not containers_skipped:
        if parts[0] == ".prometheus" and len(parts) >= 3 and parts[1] == "workspace":
            parts = parts[2:]
        if parts and parts[0].lower() in CONTAINERS:
            parts = parts[1:]
        if not parts:
            return None
    if len(parts) == 1 and _FILE.search(parts[0]):
        return None  # a file sitting in home or a container, not a directory
    head = parts[0]
    if head.startswith(".") or head.lower() in NOT_PROJECTS or _JDK.match(head):
        return None
    label = _CHECKOUT.sub("", head) or head
    return label.lower(), label


def projects_in_tool_input(
    text: str, *, known: Sequence[KnownFolder] = (), root: str | None = None,
) -> list[tuple[str, str]]:
    """The distinct projects one tool call's input touches, each once."""
    seen: dict[str, tuple[str, str]] = {}
    for found in _paths_in(text or "", known, root):
        hit = project_of_path(found, known=known, root=root)
        if hit and hit[0] not in seen:
            seen[hit[0]] = hit
    return list(seen.values())


def _paths_in(text: str, known: Sequence[KnownFolder], root: str | None) -> list[str]:
    """Paths in *text*: the desktop's home-directory pattern, plus any path inside a known project folder or the
    projects root, wherever on the disk they are (a projects root need not be under home)."""
    found = [m.group(0) for m in _PATH_IN_TEXT.finditer(text)]
    bases = [k.folder.rstrip("/") for k in known] + ([_expand(root).rstrip("/")] if root else [])
    bases = [b for b in bases if b]
    if bases:
        pattern = re.compile("(?:" + "|".join(re.escape(b) for b in sorted(set(bases), key=len, reverse=True))
                             + r")(?:/[\w.@+=-]+)*", re.ASCII)
        found += [m.group(0) for m in pattern.finditer(text)]
    return found


def tally(inputs: Iterable[str], *, known: Sequence[KnownFolder] = (), root: str | None = None) -> Tally:
    t = Tally()
    for text in inputs:
        hits = projects_in_tool_input(text, known=known, root=root)
        if not hits:
            continue
        t.calls += 1
        for key, label in hits:
            t.counts[key] = t.counts.get(key, 0) + 1
            spellings = t.labels.setdefault(key, {})
            spellings[label] = spellings.get(label, 0) + 1
    return t


def suggest(t: Tally) -> Hint | None:
    """The suggestion, or None: too few calls, no folder with 60%, or a tie at the top."""
    if t.calls < HINT_MIN_CALLS or not t.counts:
        return None
    ranked = sorted(t.counts.items(), key=lambda kv: -kv[1])
    key, n = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == n:
        return None
    share = n / t.calls
    if share < HINT_MIN_SHARE:
        return None
    spellings = sorted(t.labels.get(key, {}).items(), key=lambda kv: (-kv[1], kv[0]))
    return Hint(key=key, label=spellings[0][0] if spellings else key, share=share, calls=t.calls)
