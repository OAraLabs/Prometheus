"""Pieces Prometheus connects out to: a typed address, the probe, pending pieces, and what voice resolves to.

Contract: docs/contracts/pieces.md. This module builds the typed-address half for voice boxes:

* **An address is checked before anything is asked** (contract 5). It is resolved once and every address it resolves
  to must be allowed: plain HTTP only to loopback or a mesh VPN (RFC 6598 shared space), never to a LAN or public
  address. HTTPS needs a pinned certificate, which this build does not do yet, so it is refused for now.
* **The probe is setup-probe's rules** (contract 3.3): fixed paths, no redirects, no body read, a deadline per
  request and for the whole look, and only on the one host the owner typed: its port, or a short fixed list.
  A voice box is recognised by ``/health`` answering and by its audio routes refusing an EMPTY request the way a
  real one does (400/415/422, or 401/403 when it wants a key). No audio, text or key is ever sent by a probe.
* **Pending pieces live in memory**, bounded and expiring (contract 3.4). Only approval writes ``devices.db``.
* **Voice reads approved pieces read-only** (:func:`approved_offer`). A missing registry is "none", never created.

Pieces found this way have no card and no key of their own: they are "not verified" and are given no secret
(contract 5.2 rule 3). Cards, the identity challenge and pinned TLS come with ``oara piece announce``.

Source: novel code for Prometheus, 2026-11-02.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import socket
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from prometheus.config.piece_store import Piece

logger = logging.getLogger(__name__)

#: Ports asked on a typed address that names none: the usual voice server ports (contract 3.2). Never a range.
VOICE_PORTS: tuple[int, ...] = (8000, 8001, 8002, 8003, 8880)
#: How many pieces may wait for the owner at once, and how many from one host (contract 3.4).
PENDING_LIMIT = 16
PENDING_PER_HOST = 4
#: How long a pending piece waits, in seconds.
PENDING_TTL_S = 600.0

_REQUEST_TIMEOUT_S = 3.0
_PROBE_DEADLINE_S = 10.0

#: RFC 6598 shared address space, which mesh VPNs hand out (built from integers: no address literal in source).
_MESH = ipaddress.IPv4Network(((100 << 24) | (64 << 16), 10))
_LAN = tuple(ipaddress.ip_network(net) for net in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "fc00::/7", "fe80::/10"))

_VOICE = (("stt", "/v1/audio/transcriptions", "openai.audio.transcriptions"),
          ("tts", "/v1/audio/speech", "openai.audio.speech"))
#: What a real voice server answers an empty request with: it read the route and wants a body (or a key).
_REFUSES_EMPTY = frozenset({400, 415, 422})
_WANTS_KEY = frozenset({401, 403})


class AddressError(ValueError):
    """A typed address the rules refuse, with the contract's error code (section 11)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def address_class(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """``loopback``, ``mesh``, ``lan`` or ``public`` (contract 5.1)."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return "loopback"
    if isinstance(ip, ipaddress.IPv4Address) and ip in _MESH:
        return "mesh"
    if any(ip in net for net in _LAN if net.version == ip.version):
        return "lan"
    return "public"


@dataclass(frozen=True)
class Target:
    """A typed address, checked: the scheme, the address it resolved to, its class, and the port if one was typed."""

    typed_host: str
    scheme: str
    ip: str
    address_class: str
    port: int | None

    def base(self, port: int | None = None) -> str:
        host = f"[{self.ip}]" if ":" in self.ip else self.ip
        chosen = port if port is not None else self.port
        return f"{self.scheme}://{host}" + (f":{chosen}" if chosen else "")


def parse_address(text: str, *, resolve: Callable[[str], list[str]] | None = None) -> Target:
    """Check a typed address: ``host``, ``host:port`` or ``http(s)://host[:port]``. Blocking (it resolves names)."""
    raw = (text or "").strip()
    if not raw:
        raise AddressError("address_not_allowed", "no address was given")
    if "://" not in raw:
        raw = "http://" + raw
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        raise AddressError("address_not_allowed", "that is not an address") from None
    if parts.scheme not in ("http", "https"):
        raise AddressError("address_not_allowed", "only http and https addresses can be a piece")
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise AddressError("address_not_allowed", "give the machine and port only: no path, login or query")
    host = parts.hostname
    if not host:
        raise AddressError("address_not_allowed", "that is not an address")
    ips = (resolve or _resolve)(host)
    if not ips:
        raise AddressError("address_not_allowed", f"{host} does not resolve to an address")
    classes = {address_class(ipaddress.ip_address(ip)) for ip in ips}
    if parts.scheme == "http" and classes & {"lan", "public"}:
        raise AddressError(
            "plaintext_not_allowed",
            "plain HTTP is allowed only to this machine or a mesh VPN address; a piece on a local network or the "
            "internet must use HTTPS")
    if parts.scheme == "https":
        raise AddressError("tls_not_supported", "this version of Prometheus cannot pin a piece's certificate yet")
    return Target(typed_host=host, scheme=parts.scheme, ip=ips[0], address_class=classes.pop() if len(classes) == 1
                  else _worst(classes), port=port)


def _worst(classes: set[str]) -> str:
    for name in ("public", "lan", "mesh", "loopback"):
        if name in classes:
            return name
    return "public"


def _resolve(host: str) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return []
    seen: list[str] = []
    for info in infos:
        address = str(ipaddress.ip_address(info[4][0].split("%", 1)[0]))
        if address not in seen:
            seen.append(address)
    return seen


# ── the probe ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeResult:
    """What a probe found: offers as ``{kind, api, url, auth}`` (url = the server root), or a reason."""

    offers: tuple[dict[str, str], ...]
    reason: str | None
    detail: str | None


async def _status(client: httpx.AsyncClient, method: str, url: str) -> int | None:
    """The status a fixed request gets, never reading the answer's body. None when nothing answered."""
    try:
        async with client.stream(method, url, content=b"" if method == "POST" else None) as response:
            return response.status_code
    except httpx.HTTPError:
        return None


async def _probe_port(client: httpx.AsyncClient, base: str) -> tuple[bool, list[dict[str, str]]]:
    health = await _status(client, "GET", f"{base}/health")
    answered = health is not None
    offers: list[dict[str, str]] = []
    for kind, route, api in _VOICE:
        status = await _status(client, "POST", f"{base}{route}")
        if status is None:
            continue
        answered = True
        if status in _REFUSES_EMPTY or status in _WANTS_KEY:
            offers.append({"kind": kind, "api": api, "url": base,
                           "auth": "bearer" if status in _WANTS_KEY else "none"})
    return answered, offers


async def probe_voice(target: Target) -> ProbeResult:
    """Look for a voice box on *target*: its port, or each of :data:`VOICE_PORTS` on that one host."""
    ports = (target.port,) if target.port else VOICE_PORTS
    bases = [target.base(port) for port in ports]
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S, follow_redirects=False) as client:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(_probe_port(client, base) for base in bases)), _PROBE_DEADLINE_S)
        except TimeoutError:
            results = []
    offers: dict[str, dict[str, str]] = {}
    for _, found in results:
        for offer in found:
            offers.setdefault(offer["kind"], offer)          # the first port that serves a kind
    if offers:
        return ProbeResult(tuple(offers[kind] for kind, _, _ in _VOICE if kind in offers), None, None)
    where = f"{target.typed_host} on port{'s' if len(ports) > 1 else ''} {', '.join(str(p) for p in ports)}"
    if any(answered for answered, _ in results):
        return ProbeResult((), "not_a_piece", f"something answered at {where}, but not as a voice server")
    return ProbeResult((), "no_answer", f"nothing answered at {where}")


# ── pending pieces ───────────────────────────────────────────────────────────

_SAY = {"stt": "speech-to-text", "tts": "text-to-speech"}


def new_piece(target: Target, offers: tuple[dict[str, str], ...], *, now: float) -> Piece:
    """A pending piece for what a probe found: no card of its own, so an unsigned one built from the probe."""
    kinds = tuple(offer["kind"] for offer in offers)
    single_port = len({offer["url"] for offer in offers}) == 1
    address = offers[0]["url"] if single_port else target.base(None)
    name = f"voice box at {target.typed_host}"
    return Piece(
        id="pc_" + secrets.token_hex(8), kind="piece", state="pending", name=name, address=address,
        address_class=target.address_class, node_pub=None, tls_spki_sha256=None, card_sha256=None,
        card={"pieceCardVersion": "1", "name": name, "offers": [dict(offer) for offer in offers]},
        uses=kinds, secret_env=None, source="manual", created_at=now, updated_at=now)


def prompt(piece: Piece) -> str:
    """The one sentence the owner is asked (contract 7.1)."""
    kinds = [offer.get("kind") for offer in piece.card.get("offers", []) if isinstance(offer, dict)]
    offered = " and ".join(_SAY[k] for k in ("stt", "tts") if k in kinds) or "nothing Prometheus can use"
    sent = {("stt",): "Your voice messages will be sent to it.",
            ("tts",): "Replies to be spoken will be sent to it.",
            ("stt", "tts"): "Your voice messages and the replies to be spoken will be sent to it."}.get(
        tuple(k for k in ("stt", "tts") if k in kinds), "")
    verified = "" if piece.node_pub else (
        " It can't prove who it is, so something else on your network could pretend to be it.")
    return f"Found a {piece.name}. It offers {offered}. {sent}{verified} Use it?".replace("  ", " ")


def host_of(address: str) -> str:
    return urlsplit(address).hostname or ""


def same_box(piece: Piece, target: Target) -> bool:
    """Whether *piece* is what *target* names: the same host, and the typed port (if any) among its own."""
    if host_of(piece.address) != target.ip:
        return False
    if target.port is None:
        return True
    urls = [piece.address] + [o.get("url", "") for o in piece.card.get("offers", []) if isinstance(o, dict)]
    return any(urlsplit(url).port == target.port for url in urls if url)


class PendingFull(Exception):
    """The pending limit (contract 3.4)."""


class PendingPieces:
    """Pieces found and not yet answered: in memory, at most :data:`PENDING_LIMIT`, each for :data:`PENDING_TTL_S`."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._items: dict[str, Piece] = {}

    def _sweep(self) -> None:
        now = self.clock()
        for piece_id in [i for i, p in self._items.items() if p.created_at + PENDING_TTL_S <= now]:
            del self._items[piece_id]

    def list(self) -> list[Piece]:
        self._sweep()
        return sorted(self._items.values(), key=lambda p: (p.created_at, p.id))

    def get(self, piece_id: str) -> Piece | None:
        self._sweep()
        return self._items.get(piece_id)

    def find(self, target: Target) -> Piece | None:
        return next((p for p in self.list() if same_box(p, target)), None)

    def add(self, piece: Piece) -> None:
        items = self.list()
        if len(items) >= PENDING_LIMIT or \
                sum(1 for p in items if host_of(p.address) == host_of(piece.address)) >= PENDING_PER_HOST:
            raise PendingFull
        self._items[piece.id] = piece

    def pop(self, piece_id: str) -> Piece | None:
        return self._items.pop(piece_id, None)


# ── what voice uses ──────────────────────────────────────────────────────────

def devices_db_path() -> Path:
    """The daemon's ``devices.db``, without creating its directory."""
    from prometheus.config.paths import data_dir_path

    return data_dir_path() / "devices.db"


def approved_offer(kind: str) -> str | None:
    """The server root of the approved piece serving *kind* (``stt``/``tts``), or None.

    Read-only: a missing registry, or one with no ``pieces`` table, is "none" and nothing is created. The oldest
    approved piece for the use wins, so approving another does not silently move voice.
    """
    path = devices_db_path()
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.debug("pieces: cannot open %s read-only (%s)", path, exc)
        return None
    try:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pieces'").fetchone() is None:
            return None
        rows = conn.execute(
            "SELECT uses, card_json FROM pieces WHERE state = 'approved' ORDER BY created_at, id").fetchall()
    except sqlite3.Error as exc:
        logger.debug("pieces: cannot read %s (%s)", path, exc)
        return None
    finally:
        conn.close()
    for uses_json, card_json in rows:
        try:
            uses, card = json.loads(uses_json), json.loads(card_json)
        except ValueError:
            continue
        if not isinstance(uses, list) or kind not in uses or not isinstance(card, dict):
            continue
        for offer in card.get("offers") or []:
            if isinstance(offer, dict) and offer.get("kind") == kind and isinstance(offer.get("url"), str):
                return str(offer["url"])
    return None
