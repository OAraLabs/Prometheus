"""How the daemon starts ANOTHER Prometheus: its own interpreter, kept out of the working directory.

A local agent task (``tasks/manager.py``) and a managed coding run (``coding/managed.py``) each start
``python -m prometheus`` in a task's or a repository's directory. Under ``-m`` Python puts the working directory
first on ``sys.path``, so a ``json.py``, ``secrets.py`` or ``prometheus/`` there would be imported in place of the
real module, in a process holding the daemon's credentials. Prometheus.app's launcher closes the same hole for
the daemon itself with ``-I -B``; its children get:

* ``-P``: the working directory (and a script's directory) never goes on ``sys.path``;
* ``-s``: no user site (``~/.local/lib``, ``~/Library/Python``), as the deployed daemon's own
  ``PYTHONNOUSERSITE`` already has it.

Not ``-I``: it implies ``-E``, which ignores every ``PYTHON*`` variable, and these children need them.
``PYTHONPATH`` carries a source checkout's ``src`` and the sitecustomize that keeps a spawned agent off a
checkout's live config in tests (tests/test_spawn_key_not_persisted.py), and the app's launcher sets
``PYTHONDONTWRITEBYTECODE`` so a child of the signed bundle's interpreter never writes into the bundle.
tests/test_respawned_children_isolated.py proves both halves with the flags the two paths really build.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import sys

#: Between the interpreter and ``-m``.
CHILD_FLAGS = ("-P", "-s")


def prometheus_argv(*args: str) -> list[str]:
    """``<the daemon's interpreter> -P -s -m prometheus <args>``."""
    return [sys.executable, *CHILD_FLAGS, "-m", "prometheus", *args]
