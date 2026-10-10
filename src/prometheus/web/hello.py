"""``GET /api/hello`` — the credential-free answer to "is there a Prometheus here, and what is it called".

A phone on the home network has no token and no address, so this is the first thing it can ask, and so
the most exposed route the daemon has. It therefore tells a stranger as little as it can and still be
useful: EXACTLY six fields (``HELLO_FIELDS``), the same six the mDNS TXT record carries
(``hello_txt``), so the two cannot drift and a seventh has to be argued for in review. Never in it:
addresses, the OS, user names, the model, uptime, session or device counts, whether anyone is paired.

Two applications serve it, the configured daemon (``web/server.py``) and setup mode
(``web/setup_server.py``), and a client must not need to know which one it reached, so both build the
answer here. What the route does for itself, because it is public and the bearer gate does not apply:

* refuse a request that carries an ``Origin`` header (a web page; this is for apps), with no CORS
  headers on the refusal;
* limit each TCP peer to ``HELLO_PER_MINUTE`` requests (never ``X-Forwarded-For``: the caller writes it);
* send ``Cache-Control: no-store``, and never write to disk.

``fp`` is a display hint and ``pair`` says what a client can really do next, so it is never optimistic:
``approve`` only when a request route is served (none is yet), ``code`` in setup mode, ``token`` when
only the API token pairs a device, ``none`` when the daemon has no token at all.

Contract: docs/PAIRING-APPROVAL-API.md, section 3.2.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from prometheus.config.display_name import DEFAULT_NAME, clean_label, display_name
from prometheus.config.instance_key import instance_fingerprint
from prometheus.version import package_version
from prometheus.web.source_limits import SourceLimiter

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import JSONResponse

#: The answer, and the TXT record, are exactly these keys.
HELLO_FIELDS = ("v", "name", "agent", "fp", "pair", "tls")

#: Requests per minute per TCP peer.
HELLO_PER_MINUTE = 60
_WINDOW_SECONDS = 60.0

_NO_STORE = {"Cache-Control": "no-store"}


def new_limiter() -> SourceLimiter:
    """A limiter at the CURRENT ``HELLO_PER_MINUTE`` (read here, so a test can change it)."""
    return SourceLimiter(HELLO_PER_MINUTE, _WINDOW_SECONDS)


def approval_offered(config: Mapping[str, Any] | None) -> bool:
    """Whether this daemon serves ``POST /api/pair/requests``. Not yet: no request route exists."""
    return False


def pair_mode(*, setup_mode: bool, auth_on: bool, approval: bool = False, code: bool = False) -> str:
    """What a client can do to join this daemon: ``approve``, ``code``, ``token`` or ``none``."""
    if setup_mode:
        return "code"
    if not auth_on:
        return "none"
    if approval:
        return "approve"
    if code:
        return "code"
    return "token"


def _agent_name(config: Mapping[str, Any] | None) -> str:
    system = config.get("system") if isinstance(config, Mapping) else None
    name = clean_label(system.get("name")) if isinstance(system, Mapping) else ""
    return name or DEFAULT_NAME


def build_hello(
    config: Mapping[str, Any] | None,
    *,
    setup_mode: bool = False,
    auth_on: bool = True,
    tls: bool = False,
) -> dict[str, Any]:
    """The six fields. *config* is ``None`` in setup mode, where there is none yet."""
    return {
        "v": package_version(),
        "name": display_name(dict(config) if isinstance(config, Mapping) else None),
        "agent": _agent_name(config),
        "fp": instance_fingerprint(),
        "pair": pair_mode(
            setup_mode=setup_mode, auth_on=auth_on, approval=approval_offered(config)),
        "tls": bool(tls),
    }


def hello_txt(hello: Mapping[str, Any]) -> dict[str, str]:
    """The mDNS TXT dictionary for a hello answer: the same six keys, every value a string.

    Raises :class:`ValueError` for a key outside ``HELLO_FIELDS`` or a missing one, so a field added to
    hello and not argued for here cannot reach the network by the other route.
    """
    extra = sorted(set(hello) - set(HELLO_FIELDS))
    missing = sorted(set(HELLO_FIELDS) - set(hello))
    if extra or missing:
        raise ValueError(f"unexpected hello fields {extra}, missing {missing}")
    return {
        key: (("1" if hello[key] else "0") if key == "tls" else str(hello[key]))
        for key in HELLO_FIELDS
    }


def hello_response(
    request: "Request",
    limiter: SourceLimiter,
    payload: Callable[[], dict[str, Any]],
) -> "JSONResponse":
    """The whole route: refuse a browser, limit the peer, answer with ``payload()``."""
    from fastapi.responses import JSONResponse

    if request.headers.get("origin"):
        return JSONResponse(
            status_code=400,
            content={"error": "browser_not_allowed",
                     "detail": "this route is for apps, not for a web page"},
            headers=_NO_STORE,
        )
    peer = request.client.host if request.client else "unknown"
    decision = limiter.check(peer)
    if not decision.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "reason": "hello",
                     "retry_after_seconds": decision.retry_after},
            headers={**_NO_STORE, "Retry-After": str(decision.retry_after)},
        )
    return JSONResponse(payload(), headers=_NO_STORE)
