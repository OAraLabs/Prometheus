"""``GET /api/pieces`` — the fleet view of the pieces Prometheus connects out to.

Contract: docs/contracts/pieces.md, section 8. The operator's (``identity.is_operator``): a scoped device is a 403
``operator_only``, from default-deny (``web/route_access.py``) and again here. Beacon draws it; Prometheus owns it.

What is not built yet answers as the contract's "nothing yet": no pending pieces (nothing notices one), health
``unknown`` and ``last_seen`` null (nothing watches one), and no ``config_file`` rows (no use resolves through pieces
yet). Never in an answer: the secret, or the name of the env variable that holds it.

Source: novel code for Prometheus, 2026-10-27.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.config.piece_store import USES, Piece, PieceStore, node_fp
from prometheus.web.pairing_routes import operator_refusal

#: The answer's version (contract 8).
VERSION = 1
#: How many pieces may wait for the owner at once (contract 3.4).
PENDING_LIMIT = 16
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


def register_pieces_routes(app: Any, *, devices: Callable[[], Any], auth_on: Callable[[], bool]) -> None:
    """Mount ``GET /api/pieces`` on *app* (before the static catch-all)."""

    @app.get("/api/pieces")
    async def list_pieces(request: Request) -> JSONResponse:
        refused = operator_refusal(request, auth_on())
        if refused is not None:
            return refused
        pieces = PieceStore(devices()).list()
        return JSONResponse(
            {"v": VERSION, "pieces": [entry(piece) for piece in pieces], "pending_limit": PENDING_LIMIT},
            headers=_NO_STORE)
