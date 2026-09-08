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

import logging
from pathlib import Path
from typing import Iterable, Sequence

from prometheus.security.path_guard import matches_any_denied

log = logging.getLogger(__name__)


def resolve_denied(denied_paths: Iterable[str] | None) -> tuple[str, ...]:
    """Normalise the configured denied paths to absolute matcher entries.

    Entries are returned as STRINGS and matched with :func:`matches_any_denied`,
    NOT expanded against the filesystem. Expansion was the defect: the gate
    matches ``/*/.ssh`` with ``fnmatch`` (where ``*`` spans ``/``) but this layer
    expanded it with ``Path.glob`` (where ``*`` is one component), so
    ``Path('/').glob('*/.ssh')`` returned ``[]`` and the shipped credential floor
    resolved to nothing here while the gate denied the same path. Keeping the
    pattern as a pattern makes the two layers symmetric by construction — there
    is no second interpretation of the entry left to drift.

    A relative entry is skipped with a warning rather than resolved: resolving it
    against the daemon's cwd is the exact defect ``_normalise_denied_path``
    raises on at boot, and this layer must not quietly reinstate it.
    """
    out: list[str] = []
    for raw in denied_paths or ():
        text = str(raw).strip()
        if not text:
            continue
        expanded = str(Path(text).expanduser())
        if not Path(expanded).is_absolute():
            log.warning(
                "denied_paths entry %r is not absolute — skipping it in the "
                "prune layer rather than resolving it against the daemon's cwd "
                "(the gate refuses to start on such an entry)",
                text,
            )
            continue
        if expanded not in out:
            out.append(expanded)
    return tuple(out)


def is_denied(path: Path | str, denied: Sequence[str]) -> bool:
    """True when *path* is denied by any entry — the SAME matcher the gate uses.

    Fails closed on a path that cannot be resolved (broken symlink, loop): a path
    we cannot reason about is not one to hand back from inside a search that may
    be rooted anywhere.
    """
    if not denied:
        return False
    return matches_any_denied(path, denied)


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
