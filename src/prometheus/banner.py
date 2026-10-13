"""The startup banner: the wordmark and the few status lines a person sees when Prometheus starts.

``render_banner`` is pure: values in, text out, no terminal and no environment. ``show_banner``
decides whether anyone is there to read it and prints it. The banner is for a person at a
terminal, so it is never printed to a pipe, a service manager or CI, and a failure to print
it must not stop a start.

A status line states only what the caller knows. A value that is not known yet is left out;
it is never replaced by a guess. The listen address is the port alone, since a hostname, an
address, a token or a path in a banner ends up in screenshots and pasted bug reports.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from typing import TextIO

from prometheus.version import package_version

SANGUINE = (0xB8, 0x47, 0x3A)
BRASS = (0xB3, 0xA0, 0x75)

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"

_WORDMARK = (
    "┏━┓┏━┓┏━┓┏┳┓┏━╸╺┳╸╻ ╻┏━╸╻ ╻┏━┓",
    "┣━┛┣┳┛┃ ┃┃┃┃┣╸  ┃ ┣━┫┣╸ ┃ ┃┗━┓",
    "╹  ╹┗╸┗━┛╹ ╹┗━╸ ╹ ╹ ╹┗━╸┗━┛┗━┛",
)
_RULE = "─" * 30
_TAGLINE = "a fire you own doesn't go out"
_BEACON = "oara.ai/beacon"
_INDENT = "  "
_LABEL_WIDTH = 9  # "daemon" + padding: every value starts in one column

# Provider keys that read badly as a label. Anything else is shown as its key.
_BACKEND_NAMES = {"llama_cpp": "llama.cpp"}

_MAX_VALUE = 60
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_OFF = {"", "0", "false", "no", "off"}


def _paint(text: str, rgb: tuple[int, int, int], color: bool) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}{_RESET}" if color else text


def _dim(text: str, color: bool) -> str:
    return f"{_DIM}{text}{_RESET}" if color else text


def _clean(value: object) -> str:
    """One printable line, with no directory in it, or ``""`` when nothing usable is left.

    A model name comes from the server's answer, not from us. A llama.cpp server can name
    its model by the path of the file it loaded, and that path has a home directory in it.
    Only the last component is kept, and control characters (an escape sequence included)
    are replaced, so the name cannot move the cursor or recolour the terminal.
    """
    if not isinstance(value, str):
        return ""
    text = _CONTROL.sub(" ", value).replace("\\", "/").rsplit("/", 1)[-1]
    text = " ".join(text.split())
    return text if len(text) <= _MAX_VALUE else text[: _MAX_VALUE - 1] + "…"


def _is_port(port: object) -> bool:
    return isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536


def render_banner(
    *,
    version: str,
    port: int | None = None,
    model: str | None = None,
    backend: str | None = None,
    color: bool = True,
) -> str:
    """The banner text, ending in a newline. Pure: nothing here reads the terminal or the environment.

    *port* is the port the daemon listens on, *model* the model it serves and *backend*
    the provider behind it. Each is left out of the output when it is ``None`` or not usable;
    the model line needs a model, and carries the backend only when that is known too.
    """
    rows: list[tuple[str, str]] = []
    if _is_port(port):
        rows.append(("daemon", f"listening on :{port}"))
    name = _clean(model)
    if name:
        kind = _clean(backend)
        kind = _BACKEND_NAMES.get(kind, kind)
        rows.append(("model", f"{name} · {kind}" if kind else name))
    rows.append(("beacon", _BEACON))

    lines = [_paint(row, SANGUINE, color) for row in _WORDMARK]
    lines += [_paint(_RULE, BRASS, color), _paint(_TAGLINE, BRASS, color), ""]
    dot = _paint("◉", SANGUINE, color)
    for label, value in rows:
        pad = " " * (_LABEL_WIDTH - len(label))
        lines.append(f"{dot} {_dim(label, color)}{pad}{value}")
    lines += ["", _dim(f"OAra Labs · prometheus {_clean(version)}", color)]
    return "".join(f"{_INDENT}{line}\n" if line else "\n" for line in lines)


def version_line(version: str) -> str:
    """The one line ``--version`` prints. Plain text: scripts read it."""
    return f"◉ prometheus {_clean(version)} · OAra Labs"


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() not in _OFF


def color_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Colour unless ``NO_COLOR`` is set to something (no-color.org: an empty value does not count)."""
    return not (os.environ if env is None else env).get("NO_COLOR")


def should_show(
    stream: TextIO, *, quiet: bool = False, env: Mapping[str, str] | None = None,
) -> bool:
    """Whether a banner belongs on *stream*: a terminal, not ``--quiet``, not ``OARA_NO_BANNER``."""
    if quiet or _flag(os.environ if env is None else env, "OARA_NO_BANNER"):
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # no such method, or the stream is closed
        return False


def _writable(text: str, stream: TextIO) -> bool:
    """Whether *stream*'s encoding can carry *text* (a legacy code page cannot draw the wordmark)."""
    try:
        text.encode(getattr(stream, "encoding", None) or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def show_banner(
    *,
    port: int | None = None,
    model: str | None = None,
    backend: str | None = None,
    version: str | None = None,
    quiet: bool = False,
    stream: TextIO | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Print the banner to *stream* (default: stdout) if a person is there to read it.

    Returns whether it printed. It never raises: a decorative banner on a closed or
    unencodable stream is a skipped banner, not a failed start.
    """
    out = sys.stdout if stream is None else stream
    if not should_show(out, quiet=quiet, env=env):
        return False
    text = render_banner(
        version=package_version() if version is None else version,
        port=port, model=model, backend=backend, color=color_enabled(env),
    )
    if not _writable(text, out):
        return False
    try:
        out.write(text)
        out.flush()
    except (OSError, ValueError):
        return False
    return True


def write_version(stream: TextIO | None = None, version: str | None = None) -> None:
    """Write the ``--version`` line. Falls back to ASCII on a stream that cannot encode ``◉``."""
    out = sys.stdout if stream is None else stream
    shown = package_version() if version is None else version
    line = version_line(shown)
    if not _writable(line, out):
        line = f"prometheus {_clean(shown)} - OAra Labs"
    out.write(line + "\n")
    out.flush()
