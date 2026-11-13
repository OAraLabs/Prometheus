"""HTTPS over Tailscale with ``tailscale serve``: the daemon side.

``tailscale serve`` terminates TLS with the certificate Tailscale issues for this node's tailnet name and relays to
the daemon, which stays bound to loopback. Nothing here talks to Tailscale or opens a port; this is how the daemon
recognises a request Serve relayed, and nothing else.

Measured on a real node (Tailscale 1.104, plain-HTTP Serve leg): the daemon's TCP peer is always ``127.0.0.1``;
``Host`` is kept (``<name>:<port>``; Serve refuses a forged one itself); ``X-Forwarded-For`` is ONE value, the
caller's tailnet address, and a forged one is overwritten; ``X-Forwarded-Host`` is ``<name>:<port>``, a forged one
overwritten.

So a request counts as relayed by Serve (:func:`relayed_by_serve`) only when ALL hold: a loopback TCP peer; ``Host``
and ``X-Forwarded-Host`` each present once and naming the configured name, on the expected port; and exactly one
``X-Forwarded-For`` value that is a tailnet address (100.64.0.0/10 or fd7a:115c:a1e0::/48). Then, and only then, the
loopback Host guards accept the name (DNS rebinding needs a name the attacker controls; a page cannot use this one,
and cannot set ``X-Forwarded-*`` across origins), and the REST guard makes the tailnet address the request's client,
so per-source limits key on the real caller. A local process can forge all of that; it is on this machine already,
is never "this machine" (``web/loopback.is_same_machine``), so it reaches no same-machine route. What it can do is
name a new tailnet address per request, which would give it a fresh per-source budget each time: so hello's limiter
has a ceiling for relayed requests, pairing requests from relayed sources share a ceiling, and a relayed source is
shown to the owner as relayed ("<address> via tailscale serve"), never as a bare address. A request Funnel marks
(``Tailscale-Funnel-Request``: the public internet through the same Serve config) is never admitted.

Off unless ``web.tailscale`` is a valid, enabled block AND the daemon has an API token: an open daemon answering
relayed requests would answer the whole tailnet.

This module is stdlib-only, like loopback.py.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: The ranges Tailscale allocates node addresses from.
TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")

#: The ASGI scope key a request relayed by Serve carries (its :class:`ServeSettings`), set by the REST guard.
SCOPE_KEY = "prometheus.tailscale_serve"

_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_FORBIDDEN = re.compile(r"[\s@/\\?#,\[\]]")


@dataclass(frozen=True)
class ServeSettings:
    """``web.tailscale``, validated: the node's tailnet name and the two Serve ports."""

    name: str
    rest_port: int = 443
    ws_port: int = 8443

    def origin(self, port: int) -> str:
        """``<name>`` on 443, else ``<name>:<port>``."""
        return self.name if port == 443 else f"{self.name}:{port}"


def _valid_name(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 253 or not value.isascii() or value != value.lower():
        return False
    labels = value.split(".")
    return (len(labels) >= 4 and labels[-2:] == ["ts", "net"]
            and all(_LABEL.fullmatch(label) for label in labels))


def _valid_port(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535


def serve_settings(config: Mapping[str, Any] | None) -> ServeSettings | None:
    """The validated ``web.tailscale`` block, or None (off). A bad block is dropped with a warning, never guessed at."""
    web = (config or {}).get("web") if isinstance(config, Mapping) else None
    if not isinstance(web, Mapping):
        return None
    block = web.get("tailscale")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        logger.warning("web.tailscale must be a mapping: ignoring it (HTTPS over Tailscale is off)")
        return None
    if block.get("enabled") is not True:
        return None
    name = block.get("name")
    rest_port = block.get("rest_port", 443)
    ws_port = block.get("ws_port", 8443)
    if not _valid_name(name):
        logger.warning("web.tailscale.name %r is not a tailnet machine name (lowercase, <machine>.<tailnet>.ts.net): "
                       "ignoring the block (HTTPS over Tailscale is off)", str(name)[:80])
        return None
    if not _valid_port(rest_port) or not _valid_port(ws_port) or rest_port == ws_port:
        logger.warning("web.tailscale: rest_port and ws_port must be two different ports from 1 to 65535: ignoring "
                       "the block (HTTPS over Tailscale is off)")
        return None
    if not (web.get("api_token") or os.environ.get("PROMETHEUS_API_TOKEN")):
        logger.warning("web.tailscale: this daemon has no API token, and an open daemon must not answer requests "
                       "relayed from the tailnet: ignoring the block (HTTPS over Tailscale is off). A token made "
                       "after the daemon started takes effect at the next restart.")
        return None
    if _trusts_loopback_proxy(web.get("trusted_proxies")):
        logger.warning("web.tailscale: web.trusted_proxies covers this machine (127.0.0.1 or ::1), so the server "
                       "would rewrite every relayed request's client before the Host check sees it, and refuse them "
                       "all. Remove the loopback entry: the tailscale block replaces it. Ignoring the block (HTTPS "
                       "over Tailscale is off).")
        return None
    return ServeSettings(name=str(name), rest_port=int(rest_port), ws_port=int(ws_port))


def _trusts_loopback_proxy(entries: Any) -> bool:
    """Does a ``web.trusted_proxies`` list cover 127.0.0.1 or ::1? (Entries are parsed the lenient way; an entry that
    does not parse is ignored here, as web/serving.py drops it.)"""
    if not isinstance(entries, (list, tuple)):
        return False
    loopbacks = (ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1"))
    for entry in entries:
        try:
            network = ipaddress.ip_network(str(entry).strip(), strict=False)
        except ValueError:
            continue
        if any(ip.version == network.version and ip in network for ip in loopbacks):
            return True
    return False


def _host_and_port(value: str | None) -> tuple[str, int | None] | None:
    """``name[:port]`` from a header value; None for anything else (IP literals, odd characters, a bad port)."""
    if not isinstance(value, str) or not value or len(value) > 262 or not value.isascii() or _FORBIDDEN.search(value):
        return None
    name, sep, port_text = value.partition(":")
    if ":" in port_text:
        return None
    if not sep:
        return name.lower(), None
    if not (port_text.isdigit() and len(port_text) <= 5) or int(port_text) > 65535:
        return None
    return name.lower(), int(port_text)


def _names_us(value: str | None, settings: ServeSettings, port: int) -> bool:
    parsed = _host_and_port(value)
    if parsed is None or parsed[0] != settings.name:
        return False
    # No port is the scheme's default: right only when the Serve port IS 443.
    return parsed[1] == port if parsed[1] is not None else port == 443


def tailnet_address(value: str | None) -> str | None:
    """*value* as a tailnet IP literal (canonical text), or None."""
    if not isinstance(value, str):
        return None
    try:
        ip = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.scope_id is not None:
        return None  # a zone id carries arbitrary text; Serve never sends one
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    in_range = ip in TAILNET_V4 if isinstance(ip, ipaddress.IPv4Address) else ip in TAILNET_V6
    return str(ip) if in_range else None


def relayed_by_serve(hosts: list[str], forwarded_hosts: list[str], forwarded_fors: list[str], *,
                     peer_is_loopback: bool, settings: ServeSettings | None, port: int,
                     funnel: bool = False) -> str | None:
    """The tailnet caller's address when a request is relayed the way Serve relays it to *port*, else None.

    Each list is every value of that header (one entry per header line). *funnel*: the request carries
    ``Tailscale-Funnel-Request`` (Funnel relays the public internet through the same config), never admitted.
    See the module docstring for the rule."""
    if settings is None or not peer_is_loopback or funnel:
        return None
    if len(hosts) != 1 or len(forwarded_hosts) != 1 or len(forwarded_fors) != 1:
        return None
    if not _names_us(hosts[0], settings, port) or not _names_us(forwarded_hosts[0], settings, port):
        return None
    if "," in forwarded_fors[0]:
        return None
    return tailnet_address(forwarded_fors[0])


def came_through_serve(request_or_scope: Any) -> ServeSettings | None:
    """The :class:`ServeSettings` a request relayed by Serve carries (set by the REST guard), else None."""
    scope = getattr(request_or_scope, "scope", request_or_scope)
    value = scope.get(SCOPE_KEY) if isinstance(scope, Mapping) else None
    return value if isinstance(value, ServeSettings) else None
