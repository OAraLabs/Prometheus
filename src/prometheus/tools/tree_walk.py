"""A bounded, streaming glob walk shared by the ``grep`` and ``glob`` tools.

Why this exists (2026-11-27, a daemon frozen at 42 GB): both tools called
``Path.glob`` and then collected the whole result (``sorted(root.glob(...))``
in grep, ``list(root.glob(...))`` in glob) before applying their limit. With
grep's default ``**/*`` and a workspace of ``~/projects``, a single call built
about 5.8 million ``Path`` objects, roughly 6.5 GB traced in five minutes, only
to read files until 200 lines matched. The memory was freed afterwards but the
fragmented arenas were never returned to the OS, so every big call pushed the
process higher, and the walk held the GIL long enough to stall the event loop
for minutes.

This walk fixes the shape, not just the numbers:

* it **streams**: callers stop pulling as soon as their limit is reached;
* it **prunes**: dependency, cache and VCS folders (``.git``, ``node_modules``,
  ``.venv`` …) and ``.claude/worktrees`` are not entered unless the pattern
  names them literally;
* it **anchors**: a literal prefix (``src/pkg/*.py``) starts the walk at
  ``src/pkg``, and a pattern without ``**`` never goes deeper than it has
  segments;
* it is **capped**: after ``max_entries`` directory entries it stops and says
  so, rather than walking an unbounded tree.

Matching follows pathlib's rules for the patterns the tools see: ``*`` and
``?`` never cross ``/``, ``**`` as a whole segment matches zero or more
folders, hidden names are matched, and matching is case-sensitive. Symlinked
folders are listed but not entered below the anchor (no loops, no escapes).
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from glob import has_magic
from pathlib import Path

#: Folders never worth searching by wildcard: dependency trees, caches, VCS.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "__pycache__",
    ".tox", ".nox", ".eggs", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".gradle", ".next", ".cache", "DerivedData",
})

#: Skipped without a note: in nearly every repo, so saying so is only noise.
_UNSAID = frozenset({".git", ".hg", ".svn"})

#: Directory entries one call may look at before it stops.
DEFAULT_MAX_ENTRIES = 200_000


@dataclass
class WalkStats:
    """What one walk did, for the caller's note."""

    entries: int = 0
    truncated: bool = False
    pruned: set[str] = field(default_factory=set)
    #: True when ripgrep listed the files (so .gitignore'd files were left out)
    ripgrep: bool = False


class PatternError(ValueError):
    """A pattern this walk will not run (said to the model, not raised past the tool)."""


def _segment_regex(segment: str) -> str:
    """One path segment's glob as a regex: ``*``/``?`` stay inside the segment."""
    out: list[str] = []
    i, n = 0, len(segment)
    while i < n:
        c = segment[i]
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and segment[j] == "!":
                j += 1
            if j < n and segment[j] == "]":
                j += 1
            while j < n and segment[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                body = segment[i + 1 : j]
                negate = body[:1] == "!"          # as fnmatch: "!" negates, "^" is literal
                if negate:
                    body = body[1:]
                # Escape what re would read as set syntax (nested "[", "&&",
                # "~~", "||", a leading "^"), as fnmatch does.
                body = re.sub(r"([\\\[&~|])", r"\\\1", body)
                if body.startswith("^"):
                    body = "\\" + body
                out.append("[" + ("^" if negate else "") + body + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def _compile(segments: tuple[str, ...]) -> re.Pattern[str]:
    """The pattern's regex; a class re refuses (``[z-a]``) is said, not raised."""
    try:
        return _compile_cached(segments)
    except re.error as exc:
        raise PatternError(f"not a valid pattern ({exc})") from None


@functools.lru_cache(maxsize=256)
def _compile_cached(segments: tuple[str, ...]) -> re.Pattern[str]:
    parts: list[str] = []
    for k, seg in enumerate(segments):
        last = k == len(segments) - 1
        if seg == "**":
            # zero or more whole folders; as the final segment, anything below
            parts.append(".*" if last else "(?:[^/]+/)*")
        else:
            parts.append(_segment_regex(seg) + ("" if last else "/"))
    return re.compile("".join(parts) + r"\Z", re.DOTALL)


@dataclass(frozen=True)
class _Plan:
    root: Path
    anchor: Path
    prefix: str                 # the literal leading segments, "/"-joined
    rest: tuple[str, ...]       # the segments from the first wildcard on
    max_depth: int | None       # None when the pattern has a "**"
    allowed_skip: frozenset[str]
    dirs_only: bool = False     # the pattern ended in "/", as pathlib reads it

    def out(self, rel: str) -> Path:
        return self.root / self.prefix / rel if self.prefix else self.root / rel


def _plan(root: Path, pattern: str) -> _Plan | Path | None:
    """Parse *pattern*; a concrete pattern comes back as its one path (or None)."""
    raw = [s for s in pattern.replace("\\", "/").split("/") if s not in ("", ".")]
    if not raw:
        raise PatternError("empty pattern")
    # Literal leading segments become the anchor; the walk starts there.
    anchor_len = 0
    while anchor_len < len(raw) and not has_magic(raw[anchor_len]):
        anchor_len += 1
    anchor = root.joinpath(*raw[:anchor_len]) if anchor_len else root
    rest = tuple(raw[anchor_len:])
    if not rest:
        return anchor if os.path.lexists(anchor) else None
    if ".." in rest:
        raise PatternError("'..' after a wildcard is not supported; put it in root instead")
    return _Plan(
        root=root,
        anchor=anchor,
        prefix="/".join(raw[:anchor_len]),
        rest=rest,
        max_depth=None if "**" in rest else len(rest),
        # A skip folder the pattern names literally (e.g. "**/node_modules/x/*.js")
        # is the model asking for it: let it in.
        allowed_skip=frozenset(s for s in rest if s in SKIP_DIRS or s == "worktrees"),
        dirs_only=pattern.replace("\\", "/").endswith("/"),
    )


def walk(
    root: Path,
    pattern: str,
    *,
    stats: WalkStats | None = None,
    max_entries: int | None = None,
) -> Iterator[Path]:
    """Yield paths under *root* matching *pattern* (relative), lazily and bounded.

    Paths come back as ``root / <relative>`` in a stable order (each folder's
    entries sorted, depth first). Files and folders both match, as with
    ``Path.glob``. Raises :class:`PatternError` for an empty pattern or a
    ``..`` segment after a wildcard.
    """
    stats = stats if stats is not None else WalkStats()
    if max_entries is None:
        max_entries = DEFAULT_MAX_ENTRIES  # read at call time, so a test can lower it
    plan = _plan(root, pattern)
    if not isinstance(plan, _Plan):
        if plan is not None:
            yield plan
        return
    regex = _compile(plan.rest)

    # Each stack item: (directory path, its path relative to the anchor, depth)
    stack: list[tuple[str, str, int]] = [(str(plan.anchor), "", 0)]
    while stack:
        d, rel_d, depth = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        subdirs: list[tuple[str, str, int]] = []
        for entry in entries:
            stats.entries += 1
            if stats.entries > max_entries:
                stats.truncated = True
                return
            rel = f"{rel_d}{entry.name}"
            # Under "**" a symlinked folder is not entered (no loops, no
            # escapes); a bounded pattern follows it as pathlib does.
            try:
                is_dir = entry.is_dir(follow_symlinks=plan.max_depth is not None)
            except OSError:
                is_dir = False
            if regex.match(rel) and (is_dir or not plan.dirs_only):
                yield plan.out(rel)
            if plan.max_depth is not None and depth + 1 >= plan.max_depth:
                continue
            if not is_dir:
                continue
            skipped = _skipped_as(plan, entry.name, rel)
            if skipped:
                if skipped not in _UNSAID:
                    stats.pruned.add(skipped)
                continue
            subdirs.append((entry.path, rel + "/", depth + 1))
        # reversed so the stack pops them in sorted order
        stack.extend(reversed(subdirs))


def _skipped_as(plan: _Plan, name: str, rel: str) -> str | None:
    """The label a folder is pruned under, or None to enter it."""
    if name in plan.allowed_skip:
        return None
    if name in SKIP_DIRS:
        return name
    if name == "worktrees":
        full = f"{plan.prefix}/{rel}" if plan.prefix else rel
        if full == ".claude/worktrees" or full.endswith("/.claude/worktrees"):
            return ".claude/worktrees"
    return None


def ripgrep() -> str | None:
    """The ``rg`` binary on PATH, or None. A shell function or alias is not one."""
    return shutil.which("rg")


def walk_files_rg(
    rg: str,
    root: Path,
    pattern: str,
    *,
    stats: WalkStats | None = None,
    max_entries: int | None = None,
) -> Iterator[Path] | None:
    """Like :func:`walk`, files only, listed by ``rg --files`` (honours .gitignore).

    ripgrep only LISTS: the caller still opens each file itself, so every
    read goes through the same denied-path and no-link checks as without it.
    The listing streams from the child and the child is killed the moment the
    caller stops pulling. Returns None when ripgrep cannot run here (the
    caller falls back to :func:`walk`).
    """
    stats = stats if stats is not None else WalkStats()
    if max_entries is None:
        max_entries = DEFAULT_MAX_ENTRIES
    plan = _plan(root, pattern)
    if not isinstance(plan, _Plan):
        return iter([plan] if plan is not None and plan.is_file() else [])
    if not plan.anchor.is_dir() or plan.dirs_only:
        return iter([])
    argv = [rg, "--files", "--hidden", "--no-config", "--no-messages", "--sort", "path", "-0"]
    if plan.max_depth is not None:
        argv += ["--max-depth", str(plan.max_depth)]
    for name in sorted(SKIP_DIRS - plan.allowed_skip):
        argv += ["--glob", f"!{name}"]
    if "worktrees" not in plan.allowed_skip:
        argv += ["--glob", "!.claude/worktrees", "--glob", "!**/.claude/worktrees"]
    argv += ["--", "."]
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv, cwd=plan.anchor, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        )
    except OSError:
        return None
    return _drain_rg(proc, plan, pattern, stats, max_entries)


def _drain_rg(
    proc: subprocess.Popen[bytes], plan: _Plan, pattern: str, stats: WalkStats, max_entries: int,
) -> Iterator[Path]:
    regex = _compile(plan.rest)
    assert proc.stdout is not None
    stats.ripgrep = True
    listed = 0
    buf = b""
    failed = False
    try:
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            *names, buf = buf.split(b"\0")
            for raw_name in names:
                listed += 1
                stats.entries += 1
                if stats.entries > max_entries:
                    stats.truncated = True
                    return
                rel = os.fsdecode(raw_name)
                if rel.startswith("./"):
                    rel = rel[2:]
                if regex.match(rel):
                    yield plan.out(rel)
        # rg exits 1 for "no files", 2 for an error; an error before it listed
        # anything means it could not run here, so the walk does the job.
        failed = proc.wait() not in (0, 1) and listed == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.wait()
    if failed:
        stats.ripgrep = False
        yield from (p for p in walk(plan.root, pattern, stats=stats, max_entries=max_entries) if p.is_file())


def walk_note(stats: WalkStats, *, limit_hit: bool = False, limit: int | None = None) -> str:
    """The line a tool appends so the model knows the result is partial."""
    notes: list[str] = []
    if limit_hit and limit is not None:
        notes.append(f"stopped at the limit of {limit}; narrow root or the pattern for more")
    if stats.truncated:
        notes.append(
            f"stopped after looking at {stats.entries - 1:,} entries without finishing "
            "the tree; narrow root or the pattern"
        )
    if stats.pruned:
        names = ", ".join(sorted(stats.pruned))
        notes.append(f"skipped {names} (name one in the pattern to search it)")
    return "".join(f"\n({n})" for n in notes)


#: Said when a ripgrep-listed search finds nothing, so a miss is not taken as proof.
RIPGREP_MISS_NOTE = (
    "\n(files were listed by ripgrep, which leaves out .gitignore'd files; to search "
    "an ignored folder, start file_glob or root inside it)"
)
