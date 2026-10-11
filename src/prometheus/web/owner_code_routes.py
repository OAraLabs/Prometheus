"""Owner codes: the person at the machine makes a Beacon on another computer their own (OWNER) device.

Why: an owner device is the operator, and "a person" for a worker's approval (#744); same-Mac pairing is the only
other way to get one, so a Beacon reaching this daemon from another computer (a MacBook over Tailscale) could hold
the API token, which #744 refuses for that, or a scoped device token, which sees only its own conversations.

    POST   /api/pair/owner-codes         mint a code: 8 digits, 5 minutes, once (config/owner_codes.py)
    GET    /api/pair/owner-codes/{id}    what became of it: pending | redeemed (who, from where) | expired | ...
    DELETE /api/pair/owner-codes/{id}    cancel it
    POST   /api/pair/owner               PUBLIC: {code, device_name, platform} -> an owner device token

**Minting is the global token's, from THIS host.** ``oara pair --owner`` calls it with the token it reads from the
env file. An owner device is not root (it cannot enrol devices, ``POST /api/devices``), so it cannot mint the code
that would enrol one either; a scoped device never reaches the handler (the default-deny gate,
``web/route_access.py``). The global token from anywhere else is refused (``not_this_machine``), so the API token in
a remote Beacon cannot promote that Beacon: someone has to be at the machine. "This host" is
``loopback.is_this_host``: a loopback peer, or the daemon's own address for a daemon bound to one address, and never
a request carrying ``X-Forwarded-For`` or ``Forwarded``. A web page is refused (an ``Origin`` header), whatever it
carries. On a Mac whose shells run behind the read floor (#712), a model's shell cannot read the env file, so it
cannot mint a code this way either.

**Redeeming is public, and checks its own input**, like the other pairing doors (``web/public_routes.py``): no
``Origin``; ``application/json``, at most 4 KiB, a JSON object; a ``device_name`` of 1 to 64 characters with no
control or format characters (refused, not stripped, as for a pairing request). A bad body is a 400 and spends no
try. Then the code, and only the code, decides: wrong, expired, used, replaced, cancelled, locked or no code at all
are ONE answer (``401 invalid_code``), so a guesser learns nothing, and five wrong tries burn the code. The device
gets a token of its own (``DeviceStore.mint_owner``, ``marked_by = "owner-code"``), listed as an owner device and
revocable on its own; a later redemption, or this Mac's same-Mac re-pairing, never revokes it.

The answer carries the token in clear, as ``/api/pair/local`` and the setup pairing do, plus ``endpoints`` and
``tls`` built the way the approved pairing poll builds them (``https``/``wss`` and the SPKI pin when the request
came in on the TLS listener). It is not sealed: on a path where the reply could be read, every request the device
makes afterwards carries that token anyway, and sealing one message does nothing for those
(docs/PAIRING-APPROVAL-API.md, 7.1). TLS, a tailnet or loopback is what protects it.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.config.api_token import OWNER_SOURCE_OWNER_CODE
from prometheus.config.device_store import DeviceStore, registry_platform
from prometheus.config.owner_codes import OwnerCode, OwnerCodes
from prometheus.web.loopback import is_this_host

# The pairing doors read a body and answer a stranger one way; these are shared on purpose.
from prometheus.web.pairing_routes import (
    MAX_BODY_BYTES,
    _browser_refusal,
    _clean_name,
    _endpoints,
    _json,
    _peer,
    _read_limited,
    _tls_pin,
    _TooLarge,
)

logger = logging.getLogger("prometheus.pairing")

_INVALID_CODE = {"error": "invalid_code",
                 "detail": "wrong, expired or already-used owner code; run `oara pair --owner` on the machine "
                           "Prometheus runs on for a new one"}


def _status_payload(record: OwnerCode) -> dict[str, Any]:
    device = None
    if record.device_id is not None:
        device = {"id": record.device_id, "name": record.device_name, "platform": record.platform,
                  "source": record.source}
    return {"code_id": record.id, "status": record.status, "expires_at": int(record.expires_at), "device": device}


def register_owner_code_routes(
    app: Any,
    *,
    config: dict[str, Any] | None,
    devices: Callable[[], DeviceStore],
    auth_on: Callable[[], bool],
) -> OwnerCodes:
    """Mount the four routes on *app* (before the static catch-all) and return the store (also ``app.state.owner_codes``)."""
    codes = OwnerCodes()
    app.state.owner_codes = codes
    web_config = (config or {}).get("web") or {}

    def unavailable() -> JSONResponse:
        return _json(403, {"error": "pairing_unavailable",
                           "detail": "this daemon has no API token (the API is open), so it has no owner devices"})

    def minting_refusal(request: Request) -> JSONResponse | None:
        """Why this request may not mint, read or cancel an owner code, or None."""
        if not auth_on():
            return unavailable()
        if (refusal := _browser_refusal(request)) is not None:
            return refusal
        identity = getattr(request.state, "device_identity", None)
        if identity is None or not identity.is_global:
            return _json(403, {"error": "global_token_only",
                               "detail": "owner codes are minted with the daemon's API token (oara pair --owner); "
                                         "a device, even an owner device, cannot enrol another"})
        if not is_this_host(request):
            logger.warning("pairing: owner-code refused source=%s (not this machine, or relayed)", _peer(request))
            return _json(403, {"error": "not_this_machine",
                               "detail": "owner codes are minted only from the machine Prometheus runs on "
                                         "(oara pair --owner there), never through a proxy"})
        return None

    @app.post("/api/pair/owner-codes", status_code=201)
    async def issue_owner_code(request: Request):
        if (refusal := minting_refusal(request)) is not None:
            return refusal
        record, code = codes.issue()
        return _json(201, {"code_id": record.id, "code": code, "expires_at": int(record.expires_at),
                           "ttl_seconds": int(record.expires_at - record.created_at)})

    @app.get("/api/pair/owner-codes/{code_id}")
    async def owner_code_status(code_id: str, request: Request):
        if (refusal := minting_refusal(request)) is not None:
            return refusal
        record = codes.status(code_id)
        if record is None:
            return _json(404, {"error": "unknown_code"})
        return _json(200, _status_payload(record))

    @app.delete("/api/pair/owner-codes/{code_id}")
    async def cancel_owner_code(code_id: str, request: Request):
        if (refusal := minting_refusal(request)) is not None:
            return refusal
        if not codes.cancel(code_id):
            record = codes.status(code_id)
            if record is None:
                return _json(404, {"error": "unknown_code"})
            return _json(409, {"error": "not_pending", **_status_payload(record)})
        record = codes.status(code_id)
        return _json(200, _status_payload(record) if record is not None else {"code_id": code_id})

    @app.post("/api/pair/owner")
    async def redeem_owner_code(request: Request):
        if (refusal := _browser_refusal(request)) is not None:
            return refusal
        if not auth_on():
            return unavailable()
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            return _json(400, {"error": "invalid_request", "fields": [],
                               "detail": "Content-Type must be application/json"})
        try:
            raw = await _read_limited(request)
        except _TooLarge:
            return _json(413, {"error": "too_large", "detail": f"at most {MAX_BODY_BYTES} bytes"})
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return _json(400, {"error": "invalid_request", "fields": [], "detail": "body must be a JSON object"})
        name = _clean_name(body.get("device_name"))
        if name is None:
            return _json(400, {"error": "invalid_request", "fields": ["device_name"]})
        platform = registry_platform(body.get("platform"))
        minted = codes.redeem(
            body.get("code"),
            lambda: devices().mint_owner(name, platform, by=OWNER_SOURCE_OWNER_CODE),
            source=_peer(request))
        if minted is None:
            return _json(401, _INVALID_CODE)
        return _json(200, {"token": minted["token"], "device_id": minted["id"], "name": minted["name"],
                           "owner": True, "endpoints": _endpoints(request, web_config), "tls": _tls_pin(request)})

    return codes
