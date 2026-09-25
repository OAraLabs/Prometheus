"""path_guard — shared write-boundary helper.

A single-function utility for asserting that a candidate path resolves
under one of an allow-listed set of roots. Used by autonomous components
that should not be able to write to arbitrary paths (MemoryExtractor's
ObsidianWriter, future self-improvement writers, etc.).

The path is resolved BEFORE the prefix check, so a ``../`` traversal that
lands outside the allow-list is rejected even when the literal input
string starts with an allowed prefix. This mirrors the GraftEngine pattern
documented in PROMETHEUS.md ("Path Traversal Defense").

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import fnmatch
import functools
import os
import unicodedata
from pathlib import Path
from typing import Iterable


def assert_path_under_roots(
    candidate: str | Path,
    allowed_roots: Iterable[str | Path],
) -> Path:
    """Resolve ``candidate`` and verify it lives under one of ``allowed_roots``.

    Returns the resolved path on success. Raises ``ValueError`` if the
    resolved path is not under any allowed root, or if the candidate
    cannot be resolved at all.

    Both the candidate and each root are expanduser+resolve'd before
    comparison, so the call is robust to ``~`` and ``../`` in the input.
    A path equal to a root counts as "under" the root.

    Example:
        >>> assert_path_under_roots(
        ...     "~/.prometheus/wiki/people/alice.md",
        ...     [Path.home() / ".prometheus"],
        ... )
        PosixPath('/home/.../.prometheus/wiki/people/alice.md')
    """
    try:
        target = Path(candidate).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve candidate path {candidate!r}: {exc}")

    resolved_roots: list[Path] = []
    for root in allowed_roots:
        try:
            resolved_roots.append(Path(root).expanduser().resolve())
        except (OSError, RuntimeError):
            continue
    if not resolved_roots:
        raise ValueError("allowed_roots is empty after resolution")

    for root in resolved_roots:
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target

    roots_str = ", ".join(str(r) for r in resolved_roots)
    raise ValueError(
        f"Path {target} is not under any allowed root ({roots_str})"
    )


def is_path_under_roots(
    candidate: str | Path,
    allowed_roots: Iterable[str | Path],
) -> bool:
    """Boolean variant of :func:`assert_path_under_roots`. Never raises."""
    try:
        assert_path_under_roots(candidate, allowed_roots)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Denied-path matching — ONE matcher, used by every layer that reads the list.
# ---------------------------------------------------------------------------

GLOB_CHARS = "*?["


def is_glob_pattern(entry: str) -> bool:
    """True when a ``denied_paths`` entry carries a wildcard."""
    return any(c in entry for c in GLOB_CHARS)


def denied_entry_matches(resolved_path: str, entry: str) -> bool:
    """Whether an absolute, already-resolved path is denied by ONE entry.

    THE ONE MATCHER. Two layers read ``security.denied_paths`` — the SecurityGate
    (which refuses a denied ROOT) and the grep/glob prune layer (which withholds
    denied paths from a legitimate root that contains them). They used to match
    with DIFFERENT semantics:

      * the gate used ``fnmatch``, where ``*`` spans ``/``;
      * the prune layer expanded entries with ``Path.glob``, where ``*`` matches
        ONE component.

    So the shipped credential floor ``/*/.ssh`` matched ``/home/<user>/.ssh`` at
    the gate and resolved to NOTHING in the prune layer — ``Path('/').glob
    ('*/.ssh')`` is ``[]`` because it only matches at depth one. Measured on a
    real box: ``resolve_denied(SHIPPED_DENIED_PATHS)`` returned
    ``(/etc, /sys, /boot)`` and ``is_denied(~/.ssh/id_rsa)`` was False, while the
    gate denied the same path. ``grep --root ~ 'PRIVATE KEY'`` then returned the
    key lines with no ``[N paths withheld]`` note, so the model read the result as
    complete. That is the #214 guarantee, silently absent.

    ``fnmatch``'s ``*`` spanning ``/`` is broader than a shell glob — deliberately:
    broader means MORE denied, and this is a deny list.

    A glob entry matches the path itself OR anything under it, so denying a
    directory denies its subtree without the entry having to say so. A literal
    entry matches on PATH COMPONENTS, not raw string prefix: a bare
    ``startswith`` denied ``/etcetera/notes`` for the entry ``/etc``, which is
    over-refusal that never announced itself.
    """
    if is_glob_pattern(entry):
        if fnmatch.fnmatch(resolved_path, entry):
            return True
        return fnmatch.fnmatch(resolved_path, entry.rstrip("/") + "/*")
    return (
        resolved_path == entry
        or resolved_path.startswith(entry.rstrip("/") + os.sep)
    )


def matches_any_denied(path: str | Path, entries: Iterable[str]) -> bool:
    """Whether *path* is denied by ANY entry.

    Resolves the path FIRST (so ``../`` cannot step around an entry) and fails
    CLOSED on a path that cannot be resolved — a broken symlink, a loop, or a
    path that is simply not there is not something to hand back from inside a
    search that may be rooted anywhere.

    ⚠ ``strict=True`` IS THE GUARANTEE, NOT A DETAIL. This read
    ``.resolve()`` — non-strict — and the fail-closed promise above was
    therefore delivered by accident and only for one of the three shapes it
    names. Non-strict resolve raises for a symlink CYCLE but returns the path
    unchanged for a DANGLING symlink and for a path that does not exist, so
    the ``except`` never fired for either and the guard answered on the
    path's SPELLING. Measured on 3.12.3, the version we ship: a dangling
    symlink and a nonexistent path both came back allowed. 3.13 then removed
    the last case that raised, which is how CI found it — but it was never a
    3.13 defect; 3.13 only took away the accident.

    Safe to be strict, and checked before relying on it: every caller reaches
    here through ``denied_prune.is_denied``, whose only two call sites prune
    the RESULTS of a filesystem walk (``glob.py`` filters ``root.glob(...)``;
    ``grep.py`` filters the same after ``is_file()`` has already passed). No
    caller asks about a path that has yet to be created, so nothing here can
    refuse a write target. A path arriving here that does not exist means
    reality disagrees with the caller — a file removed mid-scan — which is
    exactly when a boundary should refuse rather than guess.
    """
    try:
        resolved = str(Path(path).expanduser().resolve(strict=True))
    except (OSError, RuntimeError, ValueError):
        return True
    return denying_entry(resolved, entries) is not None


# ---------------------------------------------------------------------------
# Path identity: compare what a path IS, not how it is spelled (WP-X.23/27).
# ---------------------------------------------------------------------------
#
# A guard that resolves the path it is asked about but not the entry it
# compares against never fires where the two spellings differ: on macOS
# ``/etc``, ``/tmp`` and ``/var`` are symlinks into ``/private``, and on any
# host a ``~`` entry or a configured path can run through a symlink. And
# ``resolve()`` folds neither case nor firmlinks, so on a case-insensitive
# volume ``~/.SSH`` IS ``~/.ssh`` while no string says so. The download
# tool's guard was fixed this way first (#574); these are its helpers, shared.


def fold_component(name: str) -> str:
    """A path component as a case-insensitive, normalisation-insensitive
    volume compares it (APFS folds case, and ``ſ`` as ``s``)."""
    return unicodedata.normalize("NFC", name).casefold()


def file_identity(path: str | Path, *, follow_symlinks: bool = True) -> tuple[int, int] | None:
    """``(device, inode)`` of what ``path`` names, or None if it can't be stat'ed."""
    try:
        st = os.stat(path, follow_symlinks=follow_symlinks)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino)


def inside_by_identity(candidate: Path, prefixes: Iterable[Path]) -> Path | None:
    """The protected directory ``candidate`` is inside, by file identity.

    ``resolve()`` follows symlinks but folds neither case nor firmlinks. On
    macOS's case-insensitive volumes ``~/.SSH`` IS ``~/.ssh``, and
    ``/System/Volumes/Data/private/etc`` IS ``/private/etc``; no string
    comparison sees that. So each existing ancestor of the destination is
    compared with each protected directory by (device, inode).

    A protected directory that does not exist yet (a fresh account's
    ``~/.ssh``) has no identity to compare, and writing ``~/.SSH/...`` would
    CREATE it, as the directory sshd then reads as ``~/.ssh``. So for those,
    the ancestor that IS its parent (by identity) is found, and the next
    component of the destination is compared with the protected name folded
    for case. That refuses ``~/.SSH`` on a case-sensitive volume too, where
    it is merely a confusing name.
    """
    existing: dict[tuple[int, int], Path] = {}
    absent: dict[tuple[int, int], list[Path]] = {}
    for prefix in prefixes:
        key = file_identity(prefix)
        if key is not None:
            existing.setdefault(key, prefix)
            continue
        parent = file_identity(prefix.parent)
        if parent is None:
            continue  # neither it nor its parent exists (/proc on macOS)
        absent.setdefault(parent, []).append(prefix)
    chain = (candidate, *candidate.parents)
    for i, ancestor in enumerate(chain):
        key = file_identity(ancestor)
        if key is None:
            continue  # not created yet, or not reachable: its parents still are
        hit = existing.get(key)
        if hit is not None:
            return hit
        if i > 0:  # the component of the destination directly below `ancestor`
            below = fold_component(chain[i - 1].name)
            for prefix in absent.get(key, ()):
                if below == fold_component(prefix.name):
                    return prefix
    return None


def split_literal_prefix(entry: str) -> tuple[str, str]:
    """``entry``'s leading directories that hold no wildcard, and the rest.

    ``/tmp/*.secret`` -> ``("/tmp", "*.secret")``; ``/*/.ssh`` ->
    ``("/", "*/.ssh")``; a literal entry is all prefix: ``("/etc", "")``.
    """
    if not is_glob_pattern(entry):
        return entry, ""
    parts = entry.split("/")
    cut = next(i for i, part in enumerate(parts) if is_glob_pattern(part))
    prefix = "/".join(parts[:cut]) or "/"
    return prefix, "/".join(parts[cut:])


@functools.lru_cache(maxsize=1024)
def entry_spellings(entry: str) -> tuple[str, ...]:
    """A deny-list entry as written, and with its literal directories resolved.

    ``/etc`` -> ``("/etc", "/private/etc")`` on macOS; ``/tmp/*.secret`` ->
    ``("/tmp/*.secret", "/private/tmp/*.secret")``. Only the literal part is
    resolved: ``Path.resolve()`` on a wildcard would treat it as a name. A
    relative entry is returned as it is and never resolved, because resolving
    it against the daemon's working directory is the defect
    ``checker._normalise_denied_path`` refuses to start on.

    Cached: a deny list is fixed for the life of the process, and the prune
    layer asks once per search result. What a path IS is still checked live,
    per call, by :func:`denying_entry`'s identity pass.
    """
    prefix, rest = split_literal_prefix(entry)
    if not Path(prefix).is_absolute():
        return (entry,)
    try:
        resolved = str(Path(prefix).resolve())
    except (OSError, RuntimeError, ValueError):
        return (entry,)
    other = resolved if not rest else resolved.rstrip("/") + "/" + rest
    return (entry,) if other == entry else (entry, other)


def denying_entry(resolved_path: str | Path, entries: Iterable[str]) -> str | None:
    """The first entry that denies an already-resolved path, or None.

    THE decision for every reader of a deny list (the gate, workspace binding,
    the grep/glob prune layer, the coding sandboxes). Three passes, each
    broader than the last, so that whatever matched before still names the
    same entry:

    1. each entry exactly as given (``denied_entry_matches``: a glob matches
       the path or anything under it; a literal matches whole components);
    2. each entry with its literal directories resolved
       (:func:`entry_spellings`), for an entry that runs through a symlink;
    3. by identity: a literal entry through :func:`inside_by_identity`; a glob
       entry by finding the ancestor that IS its literal prefix and matching
       the rest of the path against the rest of the pattern. This catches
       what no spelling does: case and firmlinks on macOS, and a directory
       reached through a path the entry never named.
    """
    path = Path(resolved_path)
    text = str(path)
    entries = [e for e in entries if e]
    for entry in entries:
        if denied_entry_matches(text, entry):
            return entry
    for entry in entries:
        for spelling in entry_spellings(entry)[1:]:
            if denied_entry_matches(text, spelling):
                return entry
    literal: dict[Path, str] = {}
    globs: list[tuple[str, str, str]] = []
    for entry in entries:
        if not Path(entry).is_absolute():
            continue  # never resolved against the cwd; pass 1 already compared it
        if is_glob_pattern(entry):
            prefix, rest = split_literal_prefix(entry)
            globs.append((entry, prefix, rest))
        else:
            for spelling in entry_spellings(entry):
                literal.setdefault(Path(spelling), entry)
    if literal:
        hit = inside_by_identity(path, literal)
        if hit is not None:
            return literal[hit]
    if globs:
        ancestors = [(a, file_identity(a)) for a in path.parents]
        for entry, prefix, rest in globs:
            key = file_identity(prefix)
            if key is None:
                continue
            for ancestor, identity in ancestors:
                if identity != key:
                    continue
                rel = path.relative_to(ancestor).as_posix()
                if fnmatch.fnmatch(rel, rest) or fnmatch.fnmatch(rel, rest.rstrip("/") + "/*"):
                    return entry
    return None
