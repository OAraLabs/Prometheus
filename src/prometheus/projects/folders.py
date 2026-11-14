"""Where managed project folders live, and how one is made (docs/design/chat-projects.md, section 1.1).

The projects root is config ``projects.root``, default ``~/prometheus-projects``: a visible folder in the owner's
home directory, not inside the agent's workspace, with no space in its name so the agent's shell commands need no
quoting. A managed folder is named from the project's name (:func:`slug`), with ``-2``, ``-3``, ... when that name is
already taken on disk, and made with mode 0700, as is the root. Nothing here deletes a folder.

Validation against ``security.denied_paths`` is the caller's (it has the security config); this module only makes
directories.

Source: novel code for Prometheus, 2026-11-13.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "~/prometheus-projects"
SLUG_MAX = 48
_MAX_SUFFIX = 1000


def projects_root(config: Mapping[str, Any] | None) -> Path:
    """The projects root from ``projects.root`` (``~`` expanded), or the default."""
    block = (config or {}).get("projects")
    raw = block.get("root") if isinstance(block, Mapping) else None
    return Path(os.path.expanduser(str(raw or DEFAULT_ROOT)))


def slug(name: str) -> str:
    """A folder name from a project name: lowercase ASCII letters, digits and ``-``; ``project`` when nothing is left."""
    ascii_only = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    out = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")[:SLUG_MAX].strip("-")
    return out or "project"


def make_managed_folder(root: Path, name: str) -> Path:
    """Make ``<root>/<slug>`` (or ``-2``, ``-3``, ...) with mode 0700, and the root too if it is missing.

    ``mkdir`` without ``exist_ok`` is the claim, so two creates at once never share a folder, and anything already at
    a name (a file, a folder) moves on to the next suffix.
    """
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    base = slug(name)
    for n in range(1, _MAX_SUFFIX):
        candidate = root / (base if n == 1 else f"{base}-{n}")
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        os.chmod(candidate, 0o700)  # mkdir's mode is masked by the umask
        return candidate
    raise FileExistsError(f"no free folder name for {base!r} under {root}")
