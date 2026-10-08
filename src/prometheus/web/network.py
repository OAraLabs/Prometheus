"""What "this Mac", "home network" and "open" mean, and how an owner's choice is saved.

The mode is not a setting of its own. It is what the listen address (``web.bind``, #693) and two switches
mean together (docs/PAIRING-APPROVAL-API.md, 2.1):

* ``this_mac``: the bind is loopback. Nothing advertises.
* ``home_network``: the bind reaches the LAN **and** the owner turned it on (``network.home_network``) **and**
  it can be run safely: TLS (a later change) or the owner's explicit ``network.allow_plaintext_lan``.
* ``open``: everything else that reaches the LAN. This is what every install that predates the setting is, and
  it is reported honestly instead of being dressed up as "home network".

An owner who asked for ``home_network`` and cannot have it is told so and reported as ``open``; the daemon
does not quietly run plain HTTP on the LAN in a mode named for being safe.

Saving a choice edits the config file's TEXT (the comment-preserving editor ``PUT /api/tools/deferred``
already uses; a round-trip through a YAML dump once took the shipped template from 713 comment lines to 0),
verifies the result before writing, and writes atomically. It is saved, not applied: the listeners are bound
at start, so a change takes effect when the daemon restarts.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prometheus.web.bind import (
    DEFAULT_BIND,
    FRESH_INSTALL_BIND,
    BindError,
    all_interfaces_warning,
    is_all_interfaces,
    parse_bind,
)
from prometheus.web.loopback import is_loopback_address

logger = logging.getLogger("prometheus.network")

THIS_MAC = "this_mac"
HOME = "home_network"
OPEN = "open"

#: The two modes an owner can ask for. ``open`` is only ever a description of what is running.
CHOOSABLE = (THIS_MAC, HOME)


# ── settings ─────────────────────────────────────────────────────────────────

def _switch(section: Mapping[str, Any] | None, name: str, key: str, default: bool, quiet: bool) -> bool:
    """One boolean switch. Anything else is said out loud and ignored, never coerced (``"false"`` is truthy)."""
    value = section.get(key) if isinstance(section, Mapping) else None
    if value is None:                                  # a YAML key left empty is "unset", not an error
        return default
    if not isinstance(value, bool):
        if not quiet:
            logger.warning("%s.%s must be true or false, got %r — using %s", name, key, value, str(default).lower())
        return default
    return value


@dataclass(frozen=True)
class NetworkSettings:
    """The ``network:`` and ``discovery:`` sections, read here and nowhere else.

    The defaults change nothing for an install that predates them: no home network, no plaintext opt-out,
    and mDNS allowed (it still only ever runs in home-network mode, see ``web/discovery.py``).
    """

    allow_plaintext_lan: bool = False
    home_network: bool = False
    mdns: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None, *, quiet: bool = False) -> NetworkSettings:
        """Read the switches. *quiet* skips the warning for a bad value: for a route that reads on every request
        and would repeat what the daemon already said once at boot."""
        network = config.get("network") if isinstance(config, Mapping) else None
        discovery = config.get("discovery") if isinstance(config, Mapping) else None
        return cls(
            allow_plaintext_lan=_switch(network, "network", "allow_plaintext_lan", False, quiet),
            home_network=_switch(network, "network", "home_network", False, quiet),
            mdns=_switch(discovery, "discovery", "mdns", True, quiet),
        )


# ── what a bind and the switches mean ────────────────────────────────────────

_PLAINTEXT = (
    "Home network mode is running over plain HTTP (no TLS): a device's token and everything it sends cross "
    "your Wi-Fi unencrypted, and anyone on that network can read them. Use it only on a network you trust. "
    "TLS is planned; until then this is an explicit choice (network.allow_plaintext_lan)."
)


@dataclass(frozen=True)
class NetworkState:
    """What the daemon is listening for, as one value both the route and the advertiser read."""

    mode: str
    bind: str
    bind_source: str
    tls_enabled: bool
    warnings: list[str] = field(default_factory=list)


def describe(bind: str, source: str, settings: NetworkSettings, *, tls: bool = False) -> NetworkState:
    """The mode a listen address and the switches add up to, with every caveat said out loud."""
    warnings: list[str] = []
    if is_loopback_address(bind):
        if settings.home_network:
            warnings.append(
                f"network.home_network is on, but the daemon listens on {bind}, which is this machine only "
                "(loopback): nothing on the home network can reach it. Set web.bind to 0.0.0.0 or a LAN "
                "address, or choose home network from Beacon.")
        return NetworkState(THIS_MAC, bind, source, tls, warnings)
    if settings.home_network and (tls or settings.allow_plaintext_lan):
        if not tls:
            warnings.append(_PLAINTEXT)
        return NetworkState(HOME, bind, source, tls, warnings)
    if settings.home_network:
        warnings.append(
            "network.home_network is on, but this daemon has no TLS and network.allow_plaintext_lan is off, "
            "so it is reported as open, not as home network, and it does not advertise. To accept plain HTTP "
            "on a network you trust, set network.allow_plaintext_lan: true.")
    wide = all_interfaces_warning(bind)
    if wide:
        warnings.append(wide)
    return NetworkState(OPEN, bind, source, tls, warnings)


# ── saving a choice ──────────────────────────────────────────────────────────

class PersistError(Exception):
    """The choice could not be saved. The file is exactly as it was."""


Edit = tuple[list[str], str]


def _edit_text(text: str, edits: list[Edit]) -> str:
    """*text* with each ``(key path, YAML literal)`` set, every other byte as it was."""
    from prometheus.web.server import _set_yaml_scalar_preserving_comments

    for keys, literal in edits:
        text = _set_yaml_scalar_preserving_comments(text, keys, literal)
    return text


def _set_nested(document: dict[str, Any], keys: list[str], value: Any) -> None:
    node = document
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def _saved_bind(document: Mapping[str, Any]) -> str | None:
    """The address ``web.bind`` names in the file, normalised; ``None`` when it names none (or a bad one)."""
    web = document.get("web")
    raw = web.get("bind") if isinstance(web, Mapping) else None
    if raw is None:
        return None
    try:
        return parse_bind(raw, "web.bind")
    except BindError:
        return None


def _wanted(document: Mapping[str, Any], mode: str, current_bind: str) -> dict[tuple[str, ...], Any]:
    """The key paths this choice needs the file to say, leaving out what it already says."""
    saved = _saved_bind(document)
    wanted: dict[tuple[str, ...], Any] = {}
    if mode == HOME:
        # Never widen what the daemon is bound to. A file that names an address keeps it; a file that
        # names none while the daemon runs on one specific address gets that address pinned.
        basis = saved if saved is not None else current_bind
        if is_loopback_address(basis):
            wanted[("web", "bind")] = DEFAULT_BIND
        elif saved is None and not is_all_interfaces(basis):
            wanted[("web", "bind")] = basis
    else:
        # What the FILE gives when nothing overrides it: an unset bind is every interface.
        if not is_loopback_address(saved if saved is not None else DEFAULT_BIND):
            wanted[("web", "bind")] = FRESH_INSTALL_BIND
    switch = mode == HOME
    network = document.get("network")
    current = network.get("home_network") if isinstance(network, Mapping) else None
    if current is not switch:
        wanted[("network", "home_network")] = switch
    return wanted


def _literal(value: Any) -> str:
    return ("true" if value else "false") if isinstance(value, bool) else f'"{value}"'


def _comments(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


def persist_choice(path: str | os.PathLike[str], mode: str, current_bind: str) -> bool:
    """Save ``this_mac`` or ``home_network`` into the config file at *path*. ``True`` when it wrote.

    *current_bind* is the address the daemon is listening on now. ``home_network`` keeps whatever address the
    file or the daemon already has, and only widens a loopback bind to every interface.

    Raises :class:`PersistError` and leaves the file exactly as it was: no file, a file that is not YAML, an
    edit that does not read back as intended or that lost a comment, or a write that fails. The file is never
    created, and it is replaced atomically (a temp file in the same directory, ``os.replace``), keeping its mode.
    """
    if mode not in CHOOSABLE:
        raise PersistError(f"cannot save {mode!r}: only {' and '.join(CHOOSABLE)} can be chosen")
    target = Path(path)
    try:
        original = target.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise PersistError(f"no config file at {target}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise PersistError(f"cannot read {target}: {exc}") from exc
    try:
        document = yaml.safe_load(original)
    except yaml.YAMLError as exc:
        raise PersistError(f"{target} is not valid YAML ({type(exc).__name__}); not touching it") from exc
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise PersistError(f"{target} does not hold a mapping at the top level; not touching it")

    wanted = _wanted(document, mode, current_bind)
    if not wanted:
        return False

    expected = copy.deepcopy(document)
    for keys, value in wanted.items():
        _set_nested(expected, list(keys), value)
    updated = _edit_text(original, [(list(keys), _literal(value)) for keys, value in wanted.items()])

    # VERIFY BEFORE WRITING. A text edit can be wrong in ways a dict assignment cannot, so the whole document
    # must read back as the old one plus exactly these keys, with every comment line still there.
    try:
        reparsed = yaml.safe_load(updated)
    except yaml.YAMLError:
        reparsed = None
    if reparsed != expected or _comments(updated) != _comments(original):
        logger.warning("network: refusing to save %s to %s: the edit did not verify (file left unchanged)",
                       mode, target)
        raise PersistError("the edit did not verify against the file, so nothing was written")

    real = target.resolve()                           # a symlinked config is edited where it lives
    file_mode = stat.S_IMODE(real.stat().st_mode)
    fd, temp = tempfile.mkstemp(dir=real.parent, prefix=f".{real.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, file_mode)
        os.replace(temp, real)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise PersistError(f"could not write {real}: {exc}") from exc
    return True
