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
import os
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
    CLOSED on a path that cannot be resolved — a broken symlink or a loop is not
    something to hand back from inside a search that may be rooted anywhere.
    """
    try:
        resolved = str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return True
    return any(denied_entry_matches(resolved, e) for e in entries)
