"""The version the daemon reports to its clients."""

from __future__ import annotations

import functools
from importlib import metadata

from prometheus import __version__

DISTRIBUTION = "oara-prometheus"


@functools.cache
def package_version() -> str:
    """The installed distribution's version, or ``__version__`` when there is none.

    A wheel or a ``uv sync`` (editable) install carries metadata. A source
    checkout run from ``PYTHONPATH=src`` does not, and neither does the deployed
    daemon: deploy.sh builds its venv with ``--no-install-project`` and the unit
    imports from the source tree. Those read ``prometheus.__version__``, which
    every release commit bumps alongside pyproject.toml.
    """
    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return __version__
