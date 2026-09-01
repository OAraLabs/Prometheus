"""Default configuration values for Prometheus — and the ONE resolver that
finds ``prometheus.yaml`` in every install layout.

WHY THERE IS A RESOLVER HERE AND NOT A CONSTANT
------------------------------------------------
This module used to export::

    DEFAULTS_PATH = Path(__file__).parent.parent.parent.parent.parent \\
        / "config" / "prometheus.yaml"

``.parent`` #1 is this file's own ``config/`` directory, so five hops land one
directory ABOVE the repo root. On a checkout at ``~/Prometheus`` it named
``~/config/prometheus.yaml``; the ff-only deploy clone named the same
nonexistent path. Every caller that fell back to it therefore opened nothing,
swallowed the ``OSError``, and resolved against an empty config — silently, for
the life of the constant. Verified by outcome, not inferred:
``TokenBudget.from_config()`` answered 24000 on a box whose config says 72000.

The off-by-one is only half of it. **A constant cannot express this at all.**
``config/prometheus.yaml`` exists relative to the source tree in a checkout and
in the ff-only deploy clone, and *nowhere* under ``site-packages`` — the wheel
packages ``src/prometheus`` only (see ``config/template.py``, and the
``force-include`` stanza in ``pyproject.toml`` that had to be added to ship the
*template*). So a repo-relative path is right for two layouts out of three, and
a pip install has no repo to be relative to. That is very likely why nobody
noticed: the fallback was already dead for installed users, and the two live
layouts hid it behind an ``except OSError``.

``prometheus.__main__.load_config`` has always had the answer — a search order,
not a path. It is written down here once so the eight subsystems that reach for
a fallback config resolve the same file the CLI, the daemon and ``doctor`` do.
``__main__`` and ``cli/doctor`` delegate to it rather than keeping a third and
fourth copy of ``parents[N]``.
"""

from __future__ import annotations

from pathlib import Path

#: The checkout/deploy-clone candidate: ``<repo>/config/prometheus.yaml``.
#:
#: ⚠ FOUR parents, and the count is load-bearing —
#: ``src/prometheus/config/defaults.py`` -> ``<repo>`` is
#: ``parents[3]``. ``config/template.py`` resolves the same root the same way
#: and says so in as many words; ``tests/test_config_path_resolution.py`` pins
#: the two equal AND anchors both on ``pyproject.toml``, so the hop count is
#: checked against the filesystem rather than against someone's counting.
#:
#: Module-level and public so tests can neutralise it. The developer's own
#: gitignored ``config/prometheus.yaml`` is a live-state root exactly like
#: ``~/.prometheus`` — ``tests/conftest.py::_isolated_state_dirs`` points this
#: at tmp for the same reason it points ``PROMETHEUS_CONFIG_DIR`` there.
REPO_CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "config" / "prometheus.yaml"


def config_search_paths(explicit: str | Path | None = None) -> list[Path]:
    """The candidate config paths, most specific first.

    Mirrors ``prometheus.__main__.load_config`` — also documented in the README
    and in ``config/prometheus.yaml.default``:

    1. an explicit path (``--config``, or a caller's ``config_path=``)
    2. the repo-local ``config/prometheus.yaml`` (checkout + deploy-clone installs)
    3. ``$PROMETHEUS_CONFIG_DIR/prometheus.yaml`` — default
       ``~/.prometheus/prometheus.yaml`` (pip installs; written by
       ``prometheus setup``)

    An explicit path SHORT-CIRCUITS: a caller that named a file wants that file
    or an error, never a silent fall-through to somebody else's config.

    ⚠ CREATES NOTHING — ``config_dir_path()``, not ``get_config_dir()``. Asking
    where a file lives is not a reason to ``mkdir`` its directory, and one
    caller cannot afford it at all: ``web.setup_server.find_config_file`` runs
    BEFORE the daemon chooses setup mode, and setup mode must not create
    ``~/.prometheus`` state. That constraint is why it hand-rolled its own
    resolution instead of calling a helper — so the helper is now safe for it,
    and the hand-rolled copy is gone.
    """
    if explicit:
        return [Path(explicit).expanduser()]

    from prometheus.config.paths import config_dir_path

    return [REPO_CONFIG_PATH, config_dir_path() / "prometheus.yaml"]


def resolve_config_path(explicit: str | Path | None = None) -> Path:
    """The config file to read. **Always a Path, never None.**

    Returns the first candidate from :func:`config_search_paths` that exists,
    else the LAST one searched (``~/.prometheus/prometheus.yaml`` — where
    ``prometheus setup`` writes, so it is the useful name to print).

    ⚠ Never-None is a contract, not a convenience. The eight ``from_config``
    fallbacks hand this straight to ``open()``; four of them catch only
    ``(OSError, yaml.YAMLError)``, so a ``None`` here would become a
    ``TypeError`` from ``Path(None)`` and take the daemon's boot with it. A
    nonexistent Path raises ``FileNotFoundError`` — an ``OSError`` — which is
    exactly what every one of them already handles, so "no config anywhere"
    behaves precisely as it did when the constant was broken.
    """
    candidates = config_search_paths(explicit)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


DEFAULT_MODEL_PROVIDER = "llama_cpp"
DEFAULT_MODEL_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL_NAME = "qwen3.5-32b"

DEFAULT_CONTEXT_LIMIT = 24000
DEFAULT_COMPRESSION_TRIGGER = 0.75
DEFAULT_TOOL_RESULT_MAX = 4000
DEFAULT_RESERVED_OUTPUT = 2000
DEFAULT_FRESH_TAIL_COUNT = 32

DEFAULT_PERMISSION_MODE = "default"
