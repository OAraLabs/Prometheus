"""path_guard — shared write-boundary helper.

A single-function utility for asserting that a candidate path resolves
under one of an allow-listed set of roots. Used by autonomous components
that should not be able to write to arbitrary paths (MemoryExtractor's
ObsidianWriter, future self-improvement writers, etc.).

The path is resolved BEFORE the prefix check, so a ``../`` traversal that
lands outside the allow-list is rejected even when the literal input
string starts with an allowed prefix. This mirrors the GraftEngine pattern
documented in PROMETHEUS.md ("Path Traversal Defense").

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

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
