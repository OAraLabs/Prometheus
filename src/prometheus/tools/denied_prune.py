"""Prune denied paths out of a read tool's results.

WHY PRUNE RATHER THAN REFUSE
-----------------------------
The security gate refuses a search whose ROOT is itself denied — there is
nothing legitimate to return from ``grep --root ~/.ssh``. But a root that
merely *contains* a denied path (``~`` contains ``~/.ssh``) is the common,
legitimate case, and refusing it outright has a cost the measurement makes
concrete: across 399 recorded grep/glob calls, refusing on "contains" would
have blocked exactly one — while making ``grep --root ~`` unusable for every
future call.

The behavioural argument is the stronger one. Refusing a broad search
teaches the model to route around the boundary, and we have watched it do
precisely that with ``bash`` — the file tools are confined and the shell is
not, so a blocked file tool becomes a shell command. A boundary that makes
the sanctioned path unusable does not prevent the read; it relocates it
somewhere with no boundary at all.

So: return the legitimate hits, minus anything under a denied path, and say
that something was withheld. The secret stays unread and the tool stays
worth using.

DEFENCE IN DEPTH, NOT THE ONLY CONTROL. The gate is the primary check and
runs first; this is the second layer, and it also covers construction sites
that build tools without a gate at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def resolve_denied(denied_paths: Iterable[str] | None) -> tuple[Path, ...]:
    """Expand configured denied paths to absolute, literal directories.

    Glob entries (``~/.config/*/*env``) are expanded against the filesystem;
    an entry matching nothing contributes nothing rather than raising, since
    a denied path that does not exist yet is not a present hazard.
    """
    out: list[Path] = []
    for raw in denied_paths or ():
        text = str(raw).strip()
        if not text:
            continue
        expanded = Path(text).expanduser()
        if any(ch in text for ch in "*?["):
            anchor = Path(expanded.anchor or ".")
            try:
                pattern = str(expanded.relative_to(anchor))
                out.extend(p.resolve() for p in anchor.glob(pattern))
            except (ValueError, OSError, IndexError):
                continue
        else:
            try:
                out.append(expanded.resolve())
            except OSError:
                continue
    return tuple(dict.fromkeys(out))


def is_denied(path: Path, denied: Sequence[Path]) -> bool:
    """True when *path* is a denied path or sits under one."""
    if not denied:
        return False
    try:
        real = path.resolve()
    except (OSError, RuntimeError):
        # Unresolvable (broken symlink, loop) — treat as denied. A path we
        # cannot reason about is not one to hand back from inside a search
        # that may be rooted anywhere.
        return True
    for d in denied:
        if real == d or real.is_relative_to(d):
            return True
    return False


def withheld_note(count: int) -> str:
    """The line appended when results were pruned.

    Stated rather than silent: a caller who cannot tell a complete result
    from a filtered one will read absence as proof, and that is how a
    boundary becomes a source of wrong conclusions.
    """
    if count <= 0:
        return ""
    noun = "path" if count == 1 else "paths"
    return (
        f"\n\n[{count} {noun} withheld: under a denied path "
        f"(security.denied_paths)]"
    )
