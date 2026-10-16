"""Which ``context.model_overrides`` entry belongs to a model.

An override is keyed by the model id the server reports. For a llama.cpp box
that id is the GGUF's full path, so moving the file to another directory used
to drop its override with no message: in the 2026-10-16 audit gate the
request's ``max_tokens`` went from the override's 3,000 back to 8,192. A key
matches a model:

1. exactly, or
2. by GGUF file name, when the model id names a ``.gguf`` file and exactly one
   key names the same file (``/models/a/X.gguf``, ``X.gguf`` and ``X`` all
   name ``X.gguf``).

A renamed file is another model and matches nothing. The startup report
(context/window_report.py) names every key that no served model matches.

Every per-model lookup (the window, the output cap, the compaction threshold)
goes through :func:`override_key`, so the three cannot pick different entries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

_warned_ambiguous: set[tuple[str, tuple[str, ...]]] = set()


def gguf_name(model: str) -> str | None:
    """The GGUF file name of *model* without ``.gguf``, or None when it names no GGUF file."""
    base = model.replace("\\", "/").rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        return base[: -len(".gguf")]
    return None


def override_key(model: str | None, keys: Iterable[str]) -> str | None:
    """The key in *keys* that is *model*'s override, or None (see the module docstring)."""
    if not model:
        return None
    keys = [str(k) for k in keys]
    if model in keys:
        return model
    served = gguf_name(model)
    if served is None:
        return None
    hits = sorted(k for k in keys if (gguf_name(k) or k) == served)
    if len(hits) == 1:
        return hits[0]
    if hits:
        marker = (model, tuple(hits))
        if marker not in _warned_ambiguous:
            _warned_ambiguous.add(marker)
            log.warning(
                "context.model_overrides: %d entries name the GGUF file of %r (%s); none is applied. "
                "Key the override by the full id the server reports.", len(hits), model, ", ".join(hits),
            )
    return None


def lookup(model: str | None, table: Mapping[str, T]) -> T | None:
    """*model*'s entry in *table*, matched by :func:`override_key`."""
    key = override_key(model, table)
    return table[key] if key is not None else None
