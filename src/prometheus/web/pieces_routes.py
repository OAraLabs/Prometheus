"""``/api/pieces`` — the fleet view of the pieces Prometheus connects out to, and joining one by a typed address.

Contract: docs/contracts/pieces.md, sections 3.2-3.4, 7, 8 and 10. Every route is the operator's
(``identity.is_operator``): a scoped device is a 403 ``operator_only``, from default-deny (``web/route_access.py``)
and again here. Beacon draws it; Prometheus owns it.

* ``GET /api/pieces``: pending pieces (in memory), stored ones (``devices.db``), and a read-only ``config_file`` row
  for each voice use prometheus.yaml itself names (that entry wins, section 10).
* ``POST /api/pieces/probe``: check a typed address, look for a voice box there (``infra/pieces.py``), and hold what
  is found pending with the one sentence the owner is asked.
* ``POST /api/pieces/{id}/approve|deny|ignore`` and ``DELETE /api/pieces/{id}``: the owner's answers.

What is not built yet answers as the contract's "nothing yet": health ``unknown`` and ``last_seen`` null (nothing
watches a piece yet). Never in an answer: a secret, or the name of the env variable that holds one. A piece found by
a typed address cannot prove who it is, so it is given no secret (``secret_needs_proof``).

Source: novel code for Prometheus, 2026-10-27; typed addresses 2026-11-02.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import ipaddress
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.config.piece_store import USES, Piece, PieceStore, node_fp
from prometheus.infra import pieces as joining
from prometheus.web.pairing_routes import operator_refusal

#: The answer's version (contract 8).
VERSION = 1
#: Ids of the read-only rows for what prometheus.yaml names. Never a stored piece's (those start ``pc_``).
_CONFIG_IDS = {"stt": "cfg_stt", "tts": "cfg_tts"}
#: The wire contracts an offer may name (contract 6.2). Anything else is left out, never guessed.
APIS = ("openai.audio.transcriptions", "openai.audio.speech", "openai.chat.completions", "openai.embeddings")

_NO_STORE = {"Cache-Control": "no-store"}


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.datetime.fromtimestamp(seconds, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _offers(card: dict[str, Any]) -> list[dict[str, str]]:
    """The card's offers the contract knows, as ``{kind, api, url}``."""
    offers = card.get("offers")
    if not isinstance(offers, list):
        return []
    known: list[dict[str, str]] = []
    for offer in offers:
        if not isinstance(offer, dict):
            continue
        kind, api, url = offer.get("kind"), offer.get("api"), offer.get("url")
        if kind in USES and api in APIS and isinstance(url, str) and url:
            known.append({"kind": kind, "api": api, "url": url})
    return known


def entry(piece: Piece) -> dict[str, Any]:
    """One ``pieces[]`` entry of the fleet view."""
    return {
        "id": piece.id,
        "name": piece.name,
        "kind": piece.kind,
        "state": piece.state,
        "source": piece.source,
        "verified": piece.node_pub is not None,
        "editable": True,
        "fp": node_fp(piece.node_pub),
        "tls_spki_sha256": piece.tls_spki_sha256,
        "address": piece.address,
        "address_class": piece.address_class,
        "offers": _offers(piece.card),
        "uses": list(piece.uses),
        "health": {"state": "unknown", "since": None},
        "created_at": _iso(piece.created_at),
        "updated_at": _iso(piece.updated_at),
        "last_seen": None,
    }


def _config_rows(config: Any) -> list[dict[str, Any]]:
    """One read-only row per voice use prometheus.yaml names itself (contract 10)."""
    from prometheus.infra.voice_remote import configured_remote

    if not isinstance(config, dict):
        return []
    rows: list[dict[str, Any]] = []
    for kind, api in (("stt", "openai.audio.transcriptions"), ("tts", "openai.audio.speech")):
        remote = configured_remote(kind, config)
        if remote is None:
            continue
        parts = urlsplit(remote.url)
        rows.append({
            "id": _CONFIG_IDS[kind], "name": f"{kind}.url in prometheus.yaml", "kind": "piece", "state": "approved",
            "source": "config_file", "verified": False, "editable": False, "fp": None, "tls_spki_sha256": None,
            "address": f"{parts.scheme}://{parts.netloc}", "address_class": _literal_class(parts.hostname),
            "offers": [{"kind": kind, "api": api, "url": remote.url}], "uses": [kind],
            "health": {"state": "unknown", "since": None}, "created_at": None, "updated_at": None, "last_seen": None,
        })
    return rows


def _literal_class(host: str | None) -> str | None:
    """The address class of a literal address (or ``localhost``); None for a name, which is not resolved here."""
    if not host:
        return None
    if host == "localhost":
        return "loopback"
    try:
        return joining.address_class(ipaddress.ip_address(host))
    except ValueError:
        return None


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail}, status_code=status, headers=_NO_STORE)


async def _body(request: Request) -> dict[str, Any] | None:
    """The JSON object sent, {} for none, or None when it is not an object."""
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def register_pieces_routes(app: Any, *, devices: Callable[[], Any], auth_on: Callable[[], bool]) -> None:
    """Mount ``/api/pieces`` on *app* (before the static catch-all)."""
    pending = joining.PendingPieces()
    app.state.pieces_pending = pending

    def store() -> PieceStore:
        return PieceStore(devices())

    def config_ids(request: Request) -> set[str]:
        return {row["id"] for row in _config_rows(getattr(request.app.state, "config", None))}

    @app.get("/api/pieces")
    async def list_pieces(request: Request) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        rows = [entry(piece) for piece in pending.list()] + [entry(piece) for piece in store().list()]
        rows += _config_rows(getattr(request.app.state, "config", None))
        return JSONResponse({"v": VERSION, "pieces": rows, "pending_limit": joining.PENDING_LIMIT},
                            headers=_NO_STORE)

    @app.post("/api/pieces/probe")
    async def probe(request: Request) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        body = await _body(request)
        if body is None or not isinstance(body.get("address"), str):
            return _error(400, "invalid_request", 'send {"address": "<host>[:<port>]"}')
        try:
            target = await asyncio.to_thread(joining.parse_address, body["address"])
        except joining.AddressError as exc:
            return _error(422, exc.code, exc.detail)
        # Already known (stored or waiting): the same piece, not a second one.
        known = next((p for p in store().list() if joining.same_box(p, target)), None) or pending.find(target)
        if known is not None:
            return JSONResponse({"found": [{**entry(known), "prompt": joining.prompt(known)}], "reason": None,
                                 "detail": None}, headers=_NO_STORE)
        result = await joining.probe_voice(target)
        if not result.offers:
            return JSONResponse({"found": [], "reason": result.reason, "detail": result.detail}, headers=_NO_STORE)
        piece = joining.new_piece(target, result.offers, now=pending.clock())
        try:
            pending.add(piece)
        except joining.PendingFull:
            return _error(429, "pending_full", "too many pieces are waiting for an answer; answer or wait for them")
        return JSONResponse({"found": [{**entry(piece), "prompt": joining.prompt(piece)}], "reason": None,
                             "detail": None}, headers=_NO_STORE)

    def known(request: Request, piece_id: str) -> tuple[Piece | None, JSONResponse | None]:
        if piece_id in config_ids(request):
            return None, _error(409, "config_file_entry", "this one is set in prometheus.yaml; change it there")
        piece = pending.get(piece_id) or store().get(piece_id)
        if piece is None:
            return None, _error(404, "unknown_piece", "no piece with that id (a pending one may have expired)")
        return piece, None

    @app.post("/api/pieces/{piece_id}/approve")
    async def approve(request: Request, piece_id: str) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        piece, problem = known(request, piece_id)
        if problem is not None:
            return problem
        assert piece is not None
        body = await _body(request)
        if body is None:
            return _error(400, "invalid_request", "send a JSON object, or nothing")
        if body.get("secret"):
            # Contract 5.2 rule 3. The value is not repeated.
            return _error(422, "secret_needs_proof",
                          "this piece can't prove who it is and is reached over plain HTTP, so it is given no key")
        offered = [offer["kind"] for offer in _offers(piece.card)]
        uses = body.get("uses", offered)
        if not isinstance(uses, list) or not uses or any(use not in offered for use in uses):
            return _error(400, "invalid_request", f"uses must be some of what it offers: {', '.join(offered)}")
        approved = dataclasses.replace(piece, state="approved", uses=tuple(dict.fromkeys(uses)),
                                       updated_at=pending.clock())
        store().save(approved)
        pending.pop(piece_id)
        return JSONResponse(entry(approved), headers=_NO_STORE)

    @app.post("/api/pieces/{piece_id}/deny")
    async def deny(request: Request, piece_id: str) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        if piece_id in config_ids(request):
            return _error(409, "config_file_entry", "this one is set in prometheus.yaml; change it there")
        if pending.pop(piece_id) is None:
            return _error(404, "unknown_piece", "no pending piece with that id (it may have expired)")
        return JSONResponse({"id": piece_id, "state": "denied"}, headers=_NO_STORE)

    @app.post("/api/pieces/{piece_id}/ignore")
    async def ignore(request: Request, piece_id: str) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        piece, problem = known(request, piece_id)
        if problem is not None:
            return problem
        assert piece is not None
        ignored = dataclasses.replace(piece, state="ignored", updated_at=pending.clock())
        store().save(ignored)
        pending.pop(piece_id)
        return JSONResponse(entry(ignored), headers=_NO_STORE)

    @app.delete("/api/pieces/{piece_id}")
    async def forget(request: Request, piece_id: str) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        if piece_id in config_ids(request):
            return _error(409, "config_file_entry", "this one is set in prometheus.yaml; change it there")
        if pending.pop(piece_id) is None and not store().forget(piece_id):
            return _error(404, "unknown_piece", "no piece with that id")
        return JSONResponse({"id": piece_id, "state": "forgotten"}, headers=_NO_STORE)
