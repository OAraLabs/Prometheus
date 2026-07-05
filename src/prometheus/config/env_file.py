"""The daemon's environment file — read/write helpers.

Prometheus keeps runtime secrets (API token, gateway tokens, provider
keys) OUT of ``prometheus.yaml`` and in a plain ``KEY=value`` env file.
The canonical location is ``~/.config/prometheus/env`` — the same file
the shipped systemd unit loads via ``EnvironmentFile=`` (see
``packaging/prometheus.service``), so a value written here is visible
both to ``systemctl --user start prometheus`` and to a bare
``prometheus daemon`` (which calls :func:`load_env_file` at startup).

Resolution order for the path (tests MUST use the override so they
never touch the real file):

1. ``PROMETHEUS_ENV_FILE`` environment variable
2. ``$XDG_CONFIG_HOME/prometheus/env``
3. ``~/.config/prometheus/env``

Format: one ``KEY=value`` per line; blank lines and ``#`` comments are
ignored; an optional ``export `` prefix is tolerated. Values are taken
verbatim (surrounding single/double quotes stripped). This matches what
systemd's ``EnvironmentFile=`` accepts for the simple values we write.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def get_env_file_path() -> Path:
    """Return the env-file path (see module docstring for resolution order)."""
    override = os.environ.get("PROMETHEUS_ENV_FILE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "prometheus" / "env"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse the env file into a dict. Missing file → empty dict.

    Keys explicitly present with an empty value ARE included (as ``""``)
    so callers can distinguish "deliberately blank" from "absent" —
    the API-token bootstrap relies on that distinction.
    """
    path = path or get_env_file_path()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(raw)
        if m:
            values[m.group(1)] = _strip_quotes(m.group(2))
    return values


def load_env_file(path: Path | None = None) -> int:
    """Load the env file into ``os.environ`` (setdefault semantics).

    Variables already set in the environment win — under systemd the
    unit's ``EnvironmentFile=`` has already populated them and this is a
    no-op; run bare, this gives ``prometheus daemon`` the same view.
    Returns the number of variables newly set.
    """
    loaded = 0
    for key, value in parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def set_env_value(key: str, value: str, path: Path | None = None) -> Path:
    """Set ``key=value`` in the env file, creating it if needed.

    An existing (uncommented) ``key=`` line is rewritten in place so
    comments and ordering survive; otherwise the assignment is appended.
    New files are created ``0600`` — this file holds secrets.
    """
    path = path or get_env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    replaced = False
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            m = _LINE_RE.match(raw)
            if m and m.group(1) == key and not raw.lstrip().startswith("#"):
                if not replaced:
                    lines.append(f"{key}={value}")
                    replaced = True
                # Drop duplicate assignments of the same key.
                continue
            lines.append(raw)
    if not replaced:
        lines.append(f"{key}={value}")

    created = not path.exists()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if created:
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover — exotic filesystems
            logger.warning("could not chmod 600 %s", path)
    return path
