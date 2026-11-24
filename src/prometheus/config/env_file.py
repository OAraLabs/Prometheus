"""The daemon's environment file — read/write helpers.

Prometheus keeps runtime secrets (API token, gateway tokens, provider
keys) OUT of ``prometheus.yaml`` and in a plain ``KEY=value`` env file.
The canonical location is ``~/.config/prometheus/env`` — the same file
the shipped systemd unit loads via ``EnvironmentFile=`` (see
``packaging/prometheus.service``), so a value written here is visible
both to ``systemctl --user start prometheus`` and to a bare
``oara daemon`` (which calls :func:`load_env_file` at startup).

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
import unicodedata
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


#: Characters that end a line for SOME reader of this file. ``str.splitlines()`` (which this module once used) breaks on
#: all of these, and a value carrying one read back as a second assignment.
_LINE_BREAKS = frozenset("\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def env_value_problem(value: object) -> str | None:
    """Why *value* cannot be written to the env file, or ``None``.

    Refuses every control character (C0, DEL and C1, Unicode category ``Cc``) and every line or paragraph separator
    (``Zl``/``Zp``, and the information separators U+001C-U+001E, which are controls already). The file is line-based:
    any of these could start an assignment of its own. The message never quotes the value.
    """
    if not isinstance(value, str):
        return "must be a string"
    for ch in value:
        if ch in _LINE_BREAKS or unicodedata.category(ch) in ("Cc", "Zl", "Zp", "Cs"):
            # Cs: a lone surrogate cannot be written as UTF-8 at all.
            return "contains a line break or control character"
    if value.rstrip(" \t").endswith("\\") or value.lstrip(" \t")[:1] in ("'", '"'):
        # An older unit's EnvironmentFile= reads a trailing backslash as a line continuation and an opening quote
        # as running on to its closing one, either of which takes the NEXT line into this value.
        return "ends with a backslash or starts with a quote (paste the key without quotes)"
    return None


class EnvValueError(ValueError):
    """A name or value the env file cannot hold. The message never quotes the value."""


def _check(key: str, value: str) -> None:
    if not isinstance(key, str) or not _NAME_RE.fullmatch(key):
        raise EnvValueError("an env-file variable name is letters, digits and underscores")
    problem = env_value_problem(value)
    if problem:
        raise EnvValueError(f"the value for {key} {problem}")


def _lines(path: Path) -> list[str]:
    """The file's lines, split on ``\n`` ONLY (a trailing ``\r`` dropped): never on the separators that
    ``str.splitlines()`` also honours, so a value holding one stays inside its own line."""
    # newline="": no universal-newline translation, which would turn a lone \r into a line break.
    with open(path, encoding="utf-8", newline="") as handle:
        text = handle.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def _replace_atomically(path: Path, text: str) -> None:
    """Write *text* to a 0600 sibling and rename it over *path*: a crash, a full disk or a value that cannot be
    encoded leaves the old file whole, never a truncated one (this file holds every secret the daemon has). An
    existing file keeps its own mode; a symlink is written through, as an in-place write was."""
    import errno
    import tempfile

    if path.is_symlink():
        path = path.resolve()
    data = text.encode("utf-8")  # before anything is opened: an encoding failure must change nothing
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o600
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS) and path.exists():
            _write_in_place(path, data)  # a folder the daemon cannot create files in, the file itself writable
            return
        raise
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_name, mode)
        except OSError:  # pragma: no cover — exotic filesystems
            logger.warning("could not chmod %o %s", mode, path)
        try:
            os.replace(tmp_name, path)
        except OSError as exc:
            if exc.errno not in (errno.EBUSY, errno.EXDEV, errno.EACCES, errno.EPERM):
                raise
            # The file is mounted on its own (a container secret) and cannot be renamed over.
            os.unlink(tmp_name)
            _write_in_place(path, data)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write_in_place(path: Path, data: bytes) -> None:
    """The fallback when the folder will not take a sibling: rewrite the file through its own descriptor."""
    logger.warning("env file %s: writing in place (its folder cannot take a temporary file)", path)
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, data)
        os.ftruncate(fd, len(data))
        os.fsync(fd)
    finally:
        os.close(fd)


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
    for raw in _lines(path):
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
    no-op; run bare, this gives ``oara daemon`` the same view.
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
    return set_env_values({key: value}, path)


def set_env_values(values: dict[str, str], path: Path | None = None) -> Path:
    """Set several ``key=value`` assignments in ONE rewrite of the env file.

    For values that only work as a pair (a key and the endpoint it was issued for): one write, so a failure leaves
    neither changed rather than one. Same rules per key as :func:`set_env_value`: every name and value is checked
    before anything is written.
    """
    for key, value in values.items():
        _check(key, value)
    path = path or get_env_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    replaced: set[str] = set()
    if path.is_file():
        for raw in _lines(path):
            m = _LINE_RE.match(raw)
            if m and m.group(1) in values and not raw.lstrip().startswith("#"):
                key = m.group(1)
                if key not in replaced:
                    lines.append(f"{key}={values[key]}")
                    replaced.add(key)
                # Drop duplicate assignments of the same key.
                continue
            lines.append(raw)
    for key, value in values.items():
        if key not in replaced:
            lines.append(f"{key}={value}")

    _replace_atomically(path, "\n".join(lines) + "\n")
    return path


def remove_env_value(key: str, path: Path | None = None) -> bool:
    """Remove every uncommented ``key=`` line from the env file. True if one was there.

    Comments, ordering and every other line survive; the file's mode is left as it is.
    """
    path = path or get_env_file_path()
    if not path.is_file():
        return False
    kept: list[str] = []
    removed = False
    for raw in _lines(path):
        m = _LINE_RE.match(raw)
        if m and m.group(1) == key and not raw.lstrip().startswith("#"):
            removed = True
            continue
        kept.append(raw)
    if removed:
        _replace_atomically(path, "\n".join(kept) + ("\n" if kept else ""))
    return removed
