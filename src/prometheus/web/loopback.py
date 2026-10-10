"""Loopback helpers, and the guard that makes a loopback bind mean it.

A server bound to ``127.0.0.1`` is still reachable by a web page in the user's
own browser. DNS rebinding turns ``evil.example`` into ``127.0.0.1`` after the
page loaded, and the browser then sends the page's requests to the daemon with
``Host: evil.example``. Refusing every request whose ``Host`` is not a loopback
name closes that: a rebinding attack needs a name the attacker controls, and
this accepts only the three spellings of "this machine".

THE PUBLIC SURFACE (stable names and signatures; another change imports them)

``is_loopback_address(address: str) -> bool``
    Is this string a loopback address: ``localhost``, an IPv4 address in
    ``127.0.0.0/8``, ``::1``, or the IPv4-mapped form of a loopback address
    (``::ffff:127.0.0.1``)? A bare address only: no brackets, no port, no zone
    id. Anything unparseable is ``False``.

``is_loopback_host_header(host: str | None, *, port: int | None = None) -> bool``
    Is this ``Host`` header value a loopback NAME? ``localhost``, a loopback
    IPv4 literal, or a bracketed loopback IPv6 literal, each with an optional
    ``:port``. ``None`` (no header), the empty string, anything with whitespace,
    ``@``, ``/``, ``\\``, ``?``, ``#`` or a comma, a bare IPv6 literal, a
    trailing-dot name, ``evil.localhost`` and every other name are ``False``.
    The port is ignored unless ``port`` is given; then a header that carries a
    different port is ``False`` (a header with no port passes: that is the
    scheme's default port, not a mismatch).

``is_loopback_peer(request_or_scope) -> bool``
    Did this request arrive from a loopback client? Takes an ASGI scope (a
    mapping) or any object with a ``scope`` attribute (a Starlette ``Request``
    or ``WebSocket``). True only when the ASGI ``client`` slot holds a loopback
    IP literal. A missing client (a unix socket), a host name (Starlette's
    TestClient reports ``testclient``) and anything unparseable are ``False``:
    "not known to be loopback" is "not loopback". Behind a local reverse proxy
    that sets forwarded headers, uvicorn's proxy handling reports the forwarded
    client here, so this answers for the real client, not the proxy.

Two further names are internal to the web layer: :class:`LoopbackHostGuard` (an
ASGI middleware) and :func:`guard_if_loopback`.

DECISIONS, so the next reader does not have to re-derive them

* The Host check is about the NAME, not the port. DNS rebinding needs an
  attacker-controlled name; the port adds nothing against it, and checking it
  would break ``ssh -L 9000:localhost:8005`` and a reverse proxy on another
  local port. Pass ``port=`` to enforce it anyway.
* A request with no Host header, or with two, is refused.
* Only the exact names above count. This is an allow-list of spellings, not a
  resolver: it never asks DNS or ``/etc/hosts`` anything.
* The guard is applied only to a server bound to a loopback address. A wide or
  specific-interface bind is left alone: tailnet clients reach it by MagicDNS
  name or address, and Starlette's TestClient sends ``Host: testserver``.
* A local reverse proxy in front of a loopback bind must send
  ``Host: localhost`` (or 127.0.0.1) to the daemon; forwarding the public name
  is refused.

This module is stdlib-only (see bind.py for why).
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# Characters that never appear in a bare host[:port] and that header-smuggling
# and parser-confusion tricks are built from.
_FORBIDDEN_IN_HOST = re.compile(r"[\s@/\\?#,]")
_MAX_HOST_HEADER = 262  # 255-byte name + ':65535'


def _ip_or_none(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def _is_loopback_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def is_loopback_address(address: str) -> bool:
    """True when *address* is ``localhost`` or a loopback IP literal.

    Accepts a bare address only: no brackets, no port, no zone id. Anything
    that does not parse is ``False``. See the module docstring.
    """
    if not isinstance(address, str):
        return False
    text = address.strip()
    if not text or "%" in text:
        return False
    if text.lower() == "localhost":
        return True
    ip = _ip_or_none(text)
    return ip is not None and _is_loopback_ip(ip)


def is_loopback_host_header(host: str | None, *, port: int | None = None) -> bool:
    """True when a ``Host`` header value names this machine: ``localhost``,
    ``127.0.0.1`` (any loopback IPv4 literal) or ``[::1]``, with an optional
    port. The port is ignored unless *port* is given. See the module docstring
    for everything that is refused.
    """
    if not isinstance(host, str) or not host or len(host) > _MAX_HOST_HEADER:
        return False
    # ASCII only: str.lower() maps a few non-ASCII characters onto ASCII ones, and
    # no real loopback Host header contains one.
    if not host.isascii() or _FORBIDDEN_IN_HOST.search(host):
        return False
    port_text: str | None
    if host.startswith("["):
        end = host.find("]")
        if end == -1:
            return False
        name, rest = host[1:end], host[end + 1:]
        if rest and not rest.startswith(":"):
            return False
        port_text = rest[1:] if rest else None
        ip = _ip_or_none(name)
        if not isinstance(ip, ipaddress.IPv6Address) or not _is_loopback_ip(ip):
            return False
    else:
        if host.count(":") > 1:
            return False
        name, sep, tail = host.partition(":")
        port_text = tail if sep else None
        if name.lower() != "localhost":
            ip = _ip_or_none(name)
            if not isinstance(ip, ipaddress.IPv4Address) or not _is_loopback_ip(ip):
                return False
    if port_text is not None:
        if not (port_text.isascii() and port_text.isdigit() and len(port_text) <= 5):
            return False
        number = int(port_text)
        if number > 65535:
            return False
        if port is not None and number != port:
            return False
    return True


def is_loopback_peer(request_or_scope: Any) -> bool:
    """True when the request's client is a loopback IP literal.

    Takes an ASGI scope or an object with a ``scope`` attribute. Anything that
    does not clearly say "loopback" is ``False``. See the module docstring.
    """
    scope = getattr(request_or_scope, "scope", request_or_scope)
    if not isinstance(scope, Mapping):
        return False
    client = scope.get("client")
    if not isinstance(client, (tuple, list)) or not client:
        return False
    host = client[0]
    if not isinstance(host, str):
        return False
    ip = _ip_or_none(host.partition("%")[0])
    return ip is not None and _is_loopback_ip(ip)


#: Headers a relay adds to say who it is acting for. A request that carries either was relayed by something, so it
#: did not come from "this machine" even when the relay is on it.
_RELAY_HEADERS = frozenset({b"x-forwarded-for", b"forwarded"})


def is_same_machine(request_or_scope: Any) -> bool:
    """True when the request came from THIS machine: a loopback TCP peer that relayed nothing.

    Stricter than :func:`is_loopback_peer` on purpose, and the one to use before an action that is only meant for
    the person at this machine (same-Mac pairing, minting an owner device). Two ways a loopback peer lies:

    * a reverse proxy on this machine (``tailscale serve``, ``cloudflared``, nginx) connects from 127.0.0.1 and
      relays a request from anywhere, so a remote request has a loopback peer;
    * uvicorn REWRITES ``scope["client"]`` from ``X-Forwarded-For`` when the peer is in ``web.trusted_proxies``, so
      a range that is too wide lets a forged header make a remote caller read as 127.0.0.1.

    Both relay with ``X-Forwarded-For`` or ``Forwarded``, so a request carrying either is not this machine,
    whatever its peer says. (A relay that strips both is still stopped by the Host-header test the secret routes
    also make.) Anything that does not clearly say "loopback, no relay" is False.
    """
    scope = getattr(request_or_scope, "scope", request_or_scope)
    if not is_loopback_peer(scope):
        return False
    for name, _value in scope.get("headers") or ():
        if isinstance(name, (bytes, bytearray)) and bytes(name).lower() in _RELAY_HEADERS:
            return False
    return True


def _single_host_header(scope: Scope) -> str | None:
    """The one Host header of an ASGI scope; ``None`` for none or for several."""
    values = [
        value for name, value in (scope.get("headers") or ())
        if name.lower() == b"host"
    ]
    if len(values) != 1:
        return None
    return values[0].decode("latin-1")


class LoopbackHostGuard:
    """ASGI middleware: answer only requests whose Host names this machine.

    Wrapped around a server that is bound to a loopback address, it refuses
    DNS-rebinding requests before anything else sees them (before auth, before
    routing, before the request body is read): HTTP gets a 403, and a WebSocket
    handshake is closed (uvicorn turns that into a 403 as well). Lifespan scopes
    pass through untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope["type"]
        if kind not in ("http", "websocket") or is_loopback_host_header(
            _single_host_header(scope)
        ):
            await self.app(scope, receive, send)
            return

        shown = _single_host_header(scope)
        logger.warning(
            "refused a %s request: its Host header %r is not a loopback name, and "
            "this server is bound to a loopback address (see web.bind)",
            kind, shown if shown is None or len(shown) <= 100 else shown[:100] + "...",
        )
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({
            "error": "host_not_allowed",
            "detail": "this server is bound to a loopback address and only "
                      "answers requests addressed to localhost, 127.0.0.1 or "
                      "[::1]",
        }).encode()
        await send({
            "type": "http.response.start", "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


def guard_if_loopback(app: ASGIApp, host: str) -> ASGIApp:
    """*app* wrapped in :class:`LoopbackHostGuard` when *host* is a loopback
    address; otherwise *app* itself, unrestricted."""
    return LoopbackHostGuard(app) if is_loopback_address(host) else app
