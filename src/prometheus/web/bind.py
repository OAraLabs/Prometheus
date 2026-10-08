"""Where the daemon listens: one setting, one resolver.

Every listener the daemon owns (the REST API, the WebSocket bridge, and the
setup-mode server that serves the unauthenticated pairing endpoint) binds the
address this module resolves. Before it existed each of them hard-coded
``0.0.0.0`` and there was no way to say "this machine only".

PRECEDENCE, highest first::

    --bind ADDRESS         oara daemon / python -m prometheus.daemon
    PROMETHEUS_WEB_BIND    the process environment (or the env file)
    web.bind               prometheus.yaml
    0.0.0.0                the default; unchanged, so a Mac mini reached over
                           Tailscale (or any existing deployment) keeps working

WHAT AN ADDRESS MAY BE: an IPv4 literal, an IPv6 literal (optionally in
brackets), or the name ``localhost``. Nothing else: no host names, no ports, no
CIDR, no zone ids. ``localhost`` is pinned to ``127.0.0.1`` here rather than
handed to the resolver, which would bind every address ``/etc/hosts`` lists for
it (including ``::1``, and anything an operator or an installer put there).
Spell ``::1`` to listen on IPv6 loopback.

VALIDATION FAILS CLOSED. A value that cannot be honoured raises
:class:`BindError`, and the daemon refuses to start. It is never replaced by the
next source in the precedence chain and never by the default: a typo in
``--bind 127.0.0.l`` must not become ``0.0.0.0``. Only the source that wins is
read, so a stale bad value in the file does not block an operator who overrides
it on the command line.

An empty value is "a value that cannot be honoured" too, except for a YAML key
left without a value (``bind:``), which is ``None`` and means unset.

This module is stdlib-only on purpose: the daemon's setup-mode gate imports it
before it knows whether the web stack is installed.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The address used when nothing asks for another one. Every interface.
DEFAULT_BIND = "0.0.0.0"

#: The environment variable (also honoured from the env file).
BIND_ENV_VAR = "PROMETHEUS_WEB_BIND"

_FLAG_LABEL = "--bind"
_ENV_LABEL = BIND_ENV_VAR
_CONFIG_LABEL = "web.bind"


class BindError(ValueError):
    """A listen address that cannot be honoured. The daemon refuses to start."""


@dataclass(frozen=True)
class ResolvedBind:
    """The address to listen on, and which source decided it."""

    address: str
    #: ``"flag"``, ``"env"``, ``"config"`` or ``"default"``.
    source: str

    def describe(self) -> str:
        label = {"flag": _FLAG_LABEL, "env": _ENV_LABEL, "config": _CONFIG_LABEL}.get(
            self.source, "the default")
        return f"{self.address} (from {label})"


def parse_bind(value: object, label: str = "bind") -> str:
    """Validate one listen address and return it in canonical form.

    *label* names where the value came from (``--bind``, ``web.bind``...) so the
    refusal says which setting to fix. Raises :class:`BindError`.
    """
    if not isinstance(value, str):
        raise BindError(
            f"{label} must be an address written as text, got "
            f"{type(value).__name__} {_shown(value)}. {_HELP}"
        )
    text = value.strip()
    if not text:
        raise BindError(f"{label} is empty. {_HELP}")
    if text.lower() == "localhost":
        return "127.0.0.1"
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        if "%" in text:
            raise ValueError("zone ids are not supported")
        return str(ipaddress.ip_address(text))
    except ValueError:
        raise BindError(
            f"{label}: {_shown(value)} is not an IPv4 address, an IPv6 address or "
            f'"localhost". {_HELP}'
        ) from None


_HELP = (
    "Examples: 127.0.0.1 (this machine only), ::1 (IPv6 loopback), a specific "
    "interface address, or 0.0.0.0 (every interface). "
    "Refusing to start rather than guess a wider address."
)


def _shown(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "...'"


def resolve_bind(
    config: Mapping[str, Any] | None = None,
    *,
    flag: str | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedBind:
    """The address to listen on: flag > environment > ``web.bind`` > default.

    *flag* is the ``--bind`` value (``None`` when not given). *env* defaults to
    the process environment; pass a mapping to read a different one. *config* is
    the parsed ``prometheus.yaml`` (``None`` where there is no config yet, as in
    setup mode).

    Raises :class:`BindError` when the winning source holds an invalid value.
    """
    if flag is not None:
        return ResolvedBind(parse_bind(flag, _FLAG_LABEL), "flag")
    environ = os.environ if env is None else env
    if BIND_ENV_VAR in environ:
        return ResolvedBind(parse_bind(environ[BIND_ENV_VAR], _ENV_LABEL), "env")
    web = config.get("web") if isinstance(config, Mapping) else None
    if isinstance(web, Mapping):
        configured = web.get("bind")
        if configured is not None:
            return ResolvedBind(parse_bind(configured, _CONFIG_LABEL), "config")
    return ResolvedBind(DEFAULT_BIND, "default")


def is_all_interfaces(address: str) -> bool:
    """True for the wildcard addresses (``0.0.0.0``, ``::``): every interface."""
    try:
        return ipaddress.ip_address(address).is_unspecified
    except ValueError:
        return False


def all_interfaces_warning(address: str) -> str | None:
    """The plain-language warning for listening on every interface, or ``None``.

    One wording for the startup log and ``oara doctor``. It states what is true
    today (plain HTTP, no TLS) and does not promise anything else.
    """
    if not is_all_interfaces(address):
        return None
    return (
        f"Listening on all interfaces ({address}) over plain HTTP (no TLS): "
        "anyone who can reach this machine's network address can reach the API, "
        "and the bearer token is the only access control. To listen on this "
        "machine only, set web.bind: 127.0.0.1 in prometheus.yaml or start the "
        "daemon with --bind 127.0.0.1."
    )


def format_host_port(host: str, port: int) -> str:
    """``host:port`` for a log line, bracketing an IPv6 literal."""
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
