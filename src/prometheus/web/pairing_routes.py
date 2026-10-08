"""The pairing request routes: a new device asks to join, the owner approves, a scoped token arrives sealed.

Contract: docs/PAIRING-APPROVAL-API.md (sections 4, 5, 6 and 8). Six routes in two groups.

**Public** (the requester has no credential yet; ``web/public_routes.py`` lists the three, exactly):

    POST   /api/pair/requests            ask to join                       -> request id, poll secret, match code
    GET    /api/pair/requests/{id}       poll, with header X-Pairing-Secret -> pending | approved{sealed} | ...
    DELETE /api/pair/requests/{id}       cancel while pending, acknowledge once approved

**Operator** (bearer required, and ``identity.is_operator``: the global token or an owner device; a scoped
device is a 403 ``operator_only``, never a 401, because 401 tells a client its token is dead):

    GET    /api/pair/requests            who is waiting
    POST   /api/pair/requests/{id}/approve | /deny

A public route is outside the bearer gate, so each of these does its own checking: no browsers (an
``Origin`` header), the poll secret only in a header and only compared in constant time, the TCP peer (never
``X-Forwarded-For``) as the rate-limit key, and a body of at most 4 KiB. An unknown id, a wrong secret and
another request's secret are ONE identical 404, and wrong secrets from a source are limited, so a stranger
learns nothing and cannot guess.

What the operator's channels (Beacon, Telegram, the terminal) are told goes through ``PairingNotifier``, a
seam this module owns and later changes subscribe to. A listener that fails cannot break a request.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from prometheus.config import pair_seal
from prometheus.config.device_store import DeviceStore
from prometheus.config.instance_key import instance_public_key_der
from prometheus.config.pair_requests import (
    APPROVED,
    CANCELED,
    EXPIRED,
    PENDING,
    UNCOLLECTED,
    LimitExceeded,
    NotPending,
    PairRequest,
    PairRequestStore,
    PairingSettings,
    RequestExpired,
    UnknownRequest,
)
from prometheus.web.source_limits import SourceLimiter

logger = logging.getLogger("prometheus.pairing")

MAX_BODY_BYTES = 4096
POLL_INTERVAL_SECONDS = 2
MIN_POLL_GAP_SECONDS = 1.0
WRONG_SECRETS_PER_MINUTE = 5
SWEEP_INTERVAL_SECONDS = 30.0
_MAX_NAME_CHARS = 64
_PLATFORMS = frozenset({"ios", "macos", "windows", "linux", "android", "other"})
_PUBLIC_KEY_RE = re.compile(r"[A-Za-z0-9_-]{43}")        # 32 bytes, unpadded base64url
_MATCH_CODE_RE = re.compile(r"[0-9]{4}")
_HOST_RE = re.compile(r"(?P<host>[A-Za-z0-9.-]{1,253}|\[[0-9A-Fa-f:.]{2,45}\])(?::(?P<port>[0-9]{1,5}))?")
_NO_STORE = {"Cache-Control": "no-store"}
_UNKNOWN = {"error": "unknown_request"}


class PairingNotifier:
    """Who hears about a request. Later changes subscribe here: the Beacon sockets, Telegram.

    ``emit(kind, payload)`` returns True when at least one listener took it (that is what the 201's
    ``notified`` says). A listener that raises is logged and skipped.
    """

    def __init__(self) -> None:
        self._listeners: list[Callable[[str, dict[str, Any]], Any]] = []

    def subscribe(self, listener: Callable[[str, dict[str, Any]], Any]) -> None:
        self._listeners.append(listener)

    def clear(self) -> None:
        self._listeners.clear()

    def emit(self, kind: str, payload: dict[str, Any]) -> bool:
        taken = False
        for listener in tuple(self._listeners):
            try:
                taken = bool(listener(kind, dict(payload))) or taken
            except Exception:
                logger.warning("a pairing listener failed; the request is unaffected", exc_info=True)
        return taken


class PairingRuntime:
    """Everything the routes share: settings, the lazy store, the limiters, the notifier, the sweeper.

    ``clock`` is replaceable (tests); everything time-based reads it at call time.
    """

    def __init__(self, config: dict[str, Any] | None, devices: Callable[[], DeviceStore],
                 clock: Callable[[], float] = time.time) -> None:
        self.settings = PairingSettings.from_config(config)
        self.web_config = (config or {}).get("web") or {}
        self.notifier = PairingNotifier()
        self.clock = clock
        self.sweep_interval = SWEEP_INTERVAL_SECONDS
        self.bad_secret = SourceLimiter(WRONG_SECRETS_PER_MINUTE, 60.0, clock=lambda: self.clock())
        self._devices = devices
        self._store: PairRequestStore | None = None
        self._last_poll: OrderedDict[str, float] = OrderedDict()
        self._sweeper: asyncio.Task | None = None

    def store(self) -> PairRequestStore:
        """Built on first use, so registering the routes touches no database."""
        if self._store is None:
            self._store = PairRequestStore(self._devices(), settings=self.settings, clock=lambda: self.clock())
        return self._store

    # -- sweeping ---------------------------------------------------------

    def sweep(self) -> None:
        """Expire, revoke the uncollected, purge. Announces expiries. Safe to call on every request."""
        for req in self.store().sweep().expired:
            self.notifier.emit("resolved", _resolved(req.id, "expired", "system", self.clock()))

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval)
            try:
                self.sweep()
            except Exception:
                logger.warning("pairing: the background sweep failed; it will try again", exc_info=True)

    async def start_sweeper(self) -> None:
        if self._sweeper is None:
            self._sweeper = asyncio.get_running_loop().create_task(self._sweep_loop())

    async def stop_sweeper(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            self._sweeper = None

    # -- polling ----------------------------------------------------------

    def too_fast(self, request_id: str) -> int:
        """Seconds to wait if *request_id* was served less than a second ago, else 0 (and note this poll)."""
        now = self.clock()
        last = self._last_poll.get(request_id)
        if last is not None and now - last < MIN_POLL_GAP_SECONDS:
            return max(1, math.ceil(MIN_POLL_GAP_SECONDS - (now - last)))
        self._last_poll[request_id] = now
        self._last_poll.move_to_end(request_id)
        while len(self._last_poll) > 2048:
            self._last_poll.popitem(last=False)
        return 0


# ── helpers ──────────────────────────────────────────────────────────────────

def _json(status: int, content: dict[str, Any], **headers: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=content, headers={**_NO_STORE, **headers})


def _peer(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _browser_refusal(request: Request) -> JSONResponse | None:
    if request.headers.get("origin"):
        return _json(400, {"error": "browser_not_allowed",
                           "detail": "this route is for apps, not for a web page"})
    return None


def _rate_limited(reason: str, retry_after: int) -> JSONResponse:
    return _json(429, {"error": "rate_limited", "reason": reason, "retry_after_seconds": retry_after},
                 **{"Retry-After": str(retry_after)})


def _resolved(request_id: str, resolution: str, by: str, when: float) -> dict[str, Any]:
    return {"request_id": request_id, "resolution": resolution, "by": by, "resolved_at": int(when)}


def _pending_payload(req: PairRequest, ttl: int) -> dict[str, Any]:
    return {"request_id": req.id, "device_name": req.device_name, "platform": req.platform,
            "source_ip": req.source, "match_code": req.match_code, "created_at": int(req.created_at),
            "expires_at": int(req.expires_at), "ttl_seconds": ttl}


def _clean_name(value: object) -> str | None:
    """A device name: 1 to 64 characters after trimming, and NO control, format or line-separator
    characters. Rejected, never stripped, so what the operator sees is exactly what was sent."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not 1 <= len(text) <= _MAX_NAME_CHARS:
        return None
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] == "C" or category in ("Zl", "Zp"):
            return None
    return text


class _TooLarge(Exception):
    pass


async def _read_limited(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
        raise _TooLarge
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise _TooLarge
        chunks.append(chunk)
    return b"".join(chunks)


def _endpoints(request: Request, web_config: dict[str, Any]) -> dict[str, str | None]:
    """Where the requester should connect, from the Host it used (so the address is reachable from its side)."""
    host = request.headers.get("host", "")
    match = _HOST_RE.fullmatch(host)
    if match is None or (match["port"] is not None and int(match["port"]) > 65535):
        return {"rest": None, "ws": None}
    hostname = match["host"]
    try:
        ws_port = int(web_config.get("ws_port", 8010) or 8010)
    except (TypeError, ValueError):
        ws_port = 8010
    return {"rest": f"http://{host}", "ws": f"ws://{hostname}:{ws_port}"}


def _via(request: Request) -> str:
    """Which channel a REST decision came through, for the operator's other screens. Display only; it
    carries no authority. Only ``cli`` is believed, because the terminal route sets it."""
    return "cli" if request.headers.get("x-pairing-via", "").strip().lower() == "cli" else "beacon"


def _decider(request: Request, via: str) -> str:
    identity = getattr(request.state, "device_identity", None)
    who = getattr(identity, "name", None) or getattr(identity, "id", None) or "open"
    return f"{via}:{who}"


# ── registration ─────────────────────────────────────────────────────────────

def register_pairing_routes(
    app: Any,
    *,
    config: dict[str, Any] | None,
    devices: Callable[[], DeviceStore],
    auth_on: Callable[[], bool],
    clock: Callable[[], float] = time.time,
) -> PairingRuntime:
    """Mount the six routes on *app* (before the static catch-all) and return the shared runtime.

    The runtime is also ``app.state.pairing``. Registering touches no database: the store, and with it the
    ``pair_requests`` table, appear on the first request.
    """
    runtime = PairingRuntime(config, devices, clock)
    app.state.pairing = runtime
    app.router.add_event_handler("startup", runtime.start_sweeper)
    app.router.add_event_handler("shutdown", runtime.stop_sweeper)

    def operator_only(request: Request) -> JSONResponse | None:
        """403 for a live token that is not an operator. With auth off everyone is the operator."""
        if not auth_on():
            return None
        identity = getattr(request.state, "device_identity", None)
        if identity is None or not identity.is_operator:
            return _json(403, {"error": "operator_only",
                               "detail": "only the owner (the master token or an owner device) may do this"})
        return None

    def verified(request: Request, request_id: str) -> PairRequest | JSONResponse:
        """The request for a valid poll secret, or the one answer a stranger ever gets.

        The secret is checked FIRST and the failure budget second. The budget counts wrong guesses per
        SOURCE (the TCP peer), and it only decides what a wrong guess is told: 404 while the source has
        budget, 429 once it is spent. It never refuses the right secret. If it did, anyone who shared the
        requester's source (one NAT, one proxy, one machine) and knew the request id could kill a legitimate
        pairing with five guesses. A 256-bit secret cannot be found by guessing however many tries there
        are, so checking before limiting costs the limit nothing it was protecting.
        """
        source = _peer(request)
        req = runtime.store().verify(request_id, request.headers.get("x-pairing-secret", ""))
        if req is not None:
            return req
        blocked = runtime.bad_secret.peek(source)
        if not blocked.allowed:
            return _rate_limited("bad_secret", blocked.retry_after)
        runtime.bad_secret.check(source)
        return _json(404, _UNKNOWN)

    # ── requester: ask to join ───────────────────────────────────────────

    @app.post("/api/pair/requests")
    async def create_request(request: Request):
        if (refusal := _browser_refusal(request)) is not None:
            return refusal
        if not auth_on() or not runtime.settings.requests_enabled:
            return _json(403, {"error": "pairing_unavailable",
                               "detail": "this daemon is not offering to pair new devices"})
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

        fields: list[str] = []
        name = _clean_name(body.get("device_name"))
        if name is None:
            fields.append("device_name")
        platform = body.get("platform")
        platform = platform.strip().lower() if isinstance(platform, str) else "other"
        platform = platform if platform in _PLATFORMS else "other"
        public_key = body.get("public_key")
        if not (isinstance(public_key, str) and _PUBLIC_KEY_RE.fullmatch(public_key)):
            fields.append("public_key")
        else:
            try:
                pair_seal.check_requester_key(pair_seal.b64url_decode(public_key))
            except ValueError:
                fields.append("public_key")
        if fields or name is None or not isinstance(public_key, str):
            return _json(400, {"error": "invalid_request", "fields": fields})

        instance_key = instance_public_key_der()
        if instance_key is None:
            logger.warning("pairing: a request was refused because this daemon has no readable instance key")
            return _json(503, {"error": "identity_unavailable",
                               "detail": "this daemon has no instance key yet; restart it"})
        runtime.sweep()
        try:
            created = runtime.store().create(
                device_name=name, platform=platform, public_key=public_key,
                source=_peer(request), instance_public_key_der=instance_key)
        except LimitExceeded as exc:
            logger.info("pairing: refused source=%s reason=%s", _peer(request), exc.reason)
            return _rate_limited(exc.reason, exc.retry_after)
        req = created.request
        notified = runtime.notifier.emit("pending", _pending_payload(req, runtime.settings.request_ttl_seconds))
        return _json(201, {
            "request_id": req.id, "poll_secret": created.poll_secret, "match_code": req.match_code,
            "instance_public_key": pair_seal.b64url_encode(instance_key),
            "expires_at": int(req.expires_at), "ttl_seconds": runtime.settings.request_ttl_seconds,
            "poll_interval_seconds": POLL_INTERVAL_SECONDS, "notified": notified,
        })

    # ── requester: poll ──────────────────────────────────────────────────

    @app.get("/api/pair/requests/{request_id}")
    async def poll_request(request_id: str, request: Request):
        if (refusal := _browser_refusal(request)) is not None:
            return refusal
        runtime.sweep()
        req = verified(request, request_id)
        if isinstance(req, JSONResponse):
            return req
        wait = runtime.too_fast(req.id)
        if wait:
            return _rate_limited("poll_too_fast", wait)
        if req.state == PENDING:
            return _json(200, {"status": PENDING, "expires_at": int(req.expires_at)})
        if req.state == APPROVED:
            sealed = runtime.store().sealed_blob(req.id)
            if sealed is not None:
                return _json(200, {
                    "status": APPROVED, "device_id": req.device_id, "sealed": sealed,
                    "endpoints": _endpoints(request, runtime.web_config), "tls": None,
                    "approved_at": int(req.decided_at or 0)})
        status = EXPIRED if req.state in (UNCOLLECTED, APPROVED) else req.state
        return _json(200, {"status": status})

    # ── requester: cancel / acknowledge ──────────────────────────────────

    @app.delete("/api/pair/requests/{request_id}")
    async def finish_request(request_id: str, request: Request):
        if (refusal := _browser_refusal(request)) is not None:
            return refusal
        runtime.sweep()
        req = verified(request, request_id)
        if isinstance(req, JSONResponse):
            return req
        store = runtime.store()
        if store.cancel(req.id) is not None:
            runtime.notifier.emit("resolved", _resolved(req.id, CANCELED, "requester", runtime.clock()))
        else:
            store.acknowledge(req.id)
        return Response(status_code=204, headers=_NO_STORE)

    # ── operator ─────────────────────────────────────────────────────────

    @app.get("/api/pair/requests")
    async def list_requests(request: Request):
        if (refusal := operator_only(request)) is not None:
            return refusal
        runtime.sweep()
        ttl = runtime.settings.request_ttl_seconds
        return _json(200, {"requests": [_pending_payload(r, ttl) for r in runtime.store().pending()]})

    def parse_decision(raw: bytes, allowed: frozenset[str]) -> dict[str, Any] | JSONResponse:
        """The decision body (empty is fine), or the 400 for one that is not an object or has a stray key."""
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return _json(400, {"error": "invalid_request", "fields": [], "detail": "body must be a JSON object"})
        unknown = sorted(set(body) - allowed)
        if unknown:
            return _json(400, {"error": "invalid_request", "fields": unknown,
                               "detail": "approve accepts only name and match_code"})
        return body

    def refusal_for(exc: Exception) -> JSONResponse:
        if isinstance(exc, UnknownRequest):
            return _json(404, _UNKNOWN)
        if isinstance(exc, RequestExpired):
            return _json(410, {"error": "expired"})
        assert isinstance(exc, NotPending)
        return _json(409, {"error": "not_pending", "status": exc.status})

    @app.post("/api/pair/requests/{request_id}/approve")
    async def approve_request(request_id: str, request: Request):
        if (refusal := operator_only(request)) is not None:
            return refusal
        runtime.sweep()
        try:
            raw = await _read_limited(request)
        except _TooLarge:
            return _json(413, {"error": "too_large"})
        body = parse_decision(raw, frozenset({"name", "match_code"}))
        if isinstance(body, JSONResponse):
            return body
        name = None
        if "name" in body:
            name = _clean_name(body["name"])
            if name is None:
                return _json(400, {"error": "invalid_request", "fields": ["name"]})
        typed = body.get("match_code")
        if "match_code" in body and not (isinstance(typed, str) and _MATCH_CODE_RE.fullmatch(typed)):
            return _json(400, {"error": "invalid_request", "fields": ["match_code"]})
        store = runtime.store()
        req = store.get(request_id)
        if req is None:
            return _json(404, _UNKNOWN)
        if typed is not None:
            if not hmac.compare_digest(typed, req.match_code):
                return _json(422, {"error": "code_mismatch"})
        via = _via(request)
        try:
            approved = store.approve(request_id, name=name, decided_by=_decider(request, via))
        except (UnknownRequest, RequestExpired, NotPending) as exc:
            return refusal_for(exc)
        runtime.notifier.emit("resolved", _resolved(approved.request.id, APPROVED, via, runtime.clock()))
        return _json(200, {"request_id": approved.request.id, "status": APPROVED,
                           "device_id": approved.device_id, "name": approved.name,
                           "platform": approved.request.platform})

    @app.post("/api/pair/requests/{request_id}/deny")
    async def deny_request(request_id: str, request: Request):
        if (refusal := operator_only(request)) is not None:
            return refusal
        runtime.sweep()
        via = _via(request)
        try:
            denied = runtime.store().deny(request_id, decided_by=_decider(request, via))
        except (UnknownRequest, RequestExpired, NotPending) as exc:
            return refusal_for(exc)
        runtime.notifier.emit("resolved", _resolved(denied.id, "denied", via, runtime.clock()))
        return _json(200, {"request_id": denied.id, "status": "denied"})

    return runtime
