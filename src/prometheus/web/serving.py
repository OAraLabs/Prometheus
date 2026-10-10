"""How the daemon's HTTP servers are built, and who they believe about where a request came from.

uvicorn, at its defaults, trusts ``X-Forwarded-For`` (and ``X-Forwarded-Proto``) from any client on
127.0.0.1 and REWRITES the request's client address from it before the application sees the request.
Everything in this daemon that keys on "who is this" reads that address: the per-source limits on the
unauthenticated pairing routes and on hello, and the same-Mac checks (``is_same_machine``). So a process on
the machine, or a reverse proxy on it that passes a caller's own header through, could pick its own source
with one header and rotate it to get a fresh rate-limit budget on every request. The Beacon session found it
testing against the real daemon; Starlette's test client has no such middleware, so nothing in the suite
could see it.

So the rule is made here, once, and both servers (the configured daemon's and setup mode's) build their
uvicorn configuration through :func:`serve_config`:

* **Forwarded headers are ignored** (``proxy_headers=False``), and the address every limit and check keys on
  is the TCP peer, unless
* ``web.trusted_proxies`` names the proxy: a list of addresses or networks (``"127.0.0.1"``,
  ``"10.0.0.0/8"``, ``"::1"``). Only a request whose TCP peer is one of those has its forwarded headers
  believed, which is what a reverse proxy in front of the daemon needs. Naming one is a decision about who
  to believe, so an entry that would believe EVERYONE (``*``, ``0.0.0.0/0``, ``::/0``) or is not an address
  is dropped, out loud, and never widens trust.

The cost of the default, stated: behind a reverse proxy on this machine (a tunnel, ``tailscale serve``,
nginx) that is NOT named, every request now looks like it comes from 127.0.0.1. Name it.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Mapping
from typing import Any

from prometheus.web.loopback import guard_if_loopback

logger = logging.getLogger(__name__)


def _normalise(entry: object) -> str | None:
    """One ``trusted_proxies`` entry as a canonical address or network, or ``None`` (and a warning)."""
    shown = repr(entry)[:60]
    if not isinstance(entry, str) or not entry.strip():
        logger.warning("web.trusted_proxies: ignoring %s: an entry is an IP address or a network, as text", shown)
        return None
    text = entry.strip()
    try:
        network = ipaddress.ip_network(text, strict=False)
    except ValueError:
        logger.warning("web.trusted_proxies: ignoring %s: not an IP address or network", shown)
        return None
    if network.prefixlen == 0:
        logger.warning("web.trusted_proxies: ignoring %s: it would believe forwarded headers from EVERY address, "
                       "which is the thing this setting exists to prevent", shown)
        return None
    return str(network.network_address) if network.num_addresses == 1 and "/" not in text else str(network)


def trusted_proxies(config: Mapping[str, Any] | None) -> list[str]:
    """The proxies named in ``web.trusted_proxies``, validated. Empty means: believe no forwarded header."""
    web = config.get("web") if isinstance(config, Mapping) else None
    value = web.get("trusted_proxies", []) if isinstance(web, Mapping) else []
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        logger.warning("web.trusted_proxies must be a list of addresses, got %s: ignoring it", repr(value)[:60])
        return []
    return [kept for kept in (_normalise(entry) for entry in value) if kept is not None]


def uvicorn_proxy_options(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``uvicorn.Config`` keywords that decide whose forwarded headers are believed."""
    named = trusted_proxies(config)
    if not named:
        return {"proxy_headers": False}
    return {"proxy_headers": True, "forwarded_allow_ips": ",".join(named)}


def serve_config(app: Any, host: str, port: int, config: Mapping[str, Any] | None = None) -> Any:
    """The ``uvicorn.Config`` the daemon (and setup mode) serves *app* with, on *host*.

    A loopback *host* also puts the Host-header check in front of the app (DNS rebinding; web/loopback.py).
    ``log_config=None``: uvicorn must NOT install its own handlers. Its default config gives
    ``uvicorn.access`` / ``uvicorn.error`` handlers with ``propagate=False`` — a path around the root
    handlers, i.e. around log redaction (security/log_redaction.py) and rotation.
    """
    import uvicorn

    return uvicorn.Config(
        guard_if_loopback(app, host), host=host, port=port, log_level="info", log_config=None,
        **uvicorn_proxy_options(config),
    )


def bound_port(server: Any) -> int | None:
    """The TCP port a RUNNING server is bound to, read from its own socket.

    *server* is a uvicorn ``Server`` (once started, its asyncio servers are in ``.servers``) or anything with
    ``.sockets`` (an asyncio server, a websockets server). ``None`` when it listens on nothing: not started yet,
    closed, or an app served without one (Starlette's test client). The port a config names is what was ASKED
    for; this is what was bound, which differs when the config was edited since or asked for ``0``.
    """
    if server is None:
        return None
    for listener in getattr(server, "servers", None) or [server]:
        for sock in getattr(listener, "sockets", None) or ():
            try:
                return int(sock.getsockname()[1])
            except (OSError, IndexError, TypeError, ValueError):
                continue
    return None
