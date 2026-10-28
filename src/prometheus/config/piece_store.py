"""Pieces: the services Prometheus connects out to that the owner approved (or chose to ignore).

A piece is a voice box, a model server, later a tool server or an agent. The owner approves it once; this table holds
what was approved: the pinned node key and/or TLS certificate, the card as last read, the uses, and the NAME of the
env variable that holds its secret (never the value). Contract: docs/contracts/pieces.md, section 12.

* **It lives in devices.db**, which the gate, Seatbelt and AppArmor already keep a model's shell out of, so a shell
  cannot plant an approved piece.
* **The table appears on the first write**, like ``pair_requests``: a daemon that never sees a piece keeps the exact
  schema the parity fixtures record. Reads before that find nothing and create nothing.
* **One node key is one piece** (``node_pub`` is UNIQUE); pieces without a key are told apart by their id.
* **Only the contract's values are stored.** A kind, state, source, address class or use outside it is a
  ``ValueError`` before anything is written.

Pending pieces and health are not here: they live in memory, in the daemon.

Source: novel code for Prometheus, 2026-10-27.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from prometheus.config.device_store import DeviceStore

KINDS = ("piece", "agent")
STATES = ("approved", "ignored")
SOURCES = ("discovered", "manual")
ADDRESS_CLASSES = ("loopback", "mesh", "lan", "public")
USES = ("stt", "tts", "chat", "embeddings")

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS pieces (
      id               TEXT PRIMARY KEY,
      kind             TEXT NOT NULL,
      state            TEXT NOT NULL,
      name             TEXT NOT NULL,
      address          TEXT NOT NULL,
      address_class    TEXT NOT NULL,
      node_pub         TEXT UNIQUE,
      tls_spki_sha256  TEXT,
      card_sha256      TEXT,
      card_json        TEXT NOT NULL,
      uses             TEXT NOT NULL,
      secret_env       TEXT,
      source           TEXT NOT NULL,
      created_at       REAL NOT NULL,
      updated_at       REAL NOT NULL
    );
"""


def node_fp(node_pub: str | None) -> str | None:
    """The first 16 hex of SHA-256 over the raw 32-byte node public key (contract 3.1), or None without one."""
    if not node_pub:
        return None
    try:
        raw = base64.b64decode(node_pub, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) != 32:
        return None
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class Piece:
    """One row of ``pieces``. ``card`` is the card as last read (or as built from a probe, contract 6.3)."""

    id: str
    kind: str
    state: str
    name: str
    address: str
    address_class: str
    node_pub: str | None
    tls_spki_sha256: str | None
    card_sha256: str | None
    card: dict[str, Any]
    uses: tuple[str, ...]
    secret_env: str | None
    source: str
    created_at: float
    updated_at: float

    def check(self) -> None:
        """``ValueError`` for anything the contract does not allow."""
        for field, value, allowed in (("kind", self.kind, KINDS), ("state", self.state, STATES),
                                      ("source", self.source, SOURCES),
                                      ("address_class", self.address_class, ADDRESS_CLASSES)):
            if value not in allowed:
                raise ValueError(f"piece {field} {value!r} is not one of {', '.join(allowed)}")
        unknown = [use for use in self.uses if use not in USES]
        if unknown:
            raise ValueError(f"piece uses {unknown!r} are not among {', '.join(USES)}")
        if not self.id or not self.name.strip() or not self.address:
            raise ValueError("a piece needs an id, a name and an address")


def _from_row(row: sqlite3.Row) -> Piece:
    try:
        card = json.loads(row["card_json"])
    except ValueError:
        card = {}
    try:
        uses = json.loads(row["uses"])
    except ValueError:
        uses = []
    return Piece(
        id=row["id"], kind=row["kind"], state=row["state"], name=row["name"], address=row["address"],
        address_class=row["address_class"], node_pub=row["node_pub"], tls_spki_sha256=row["tls_spki_sha256"],
        card_sha256=row["card_sha256"], card=card if isinstance(card, dict) else {},
        uses=tuple(use for use in uses if isinstance(use, str)) if isinstance(uses, list) else (),
        secret_env=row["secret_env"], source=row["source"], created_at=row["created_at"],
        updated_at=row["updated_at"])


class PieceStore:
    """The ``pieces`` table, on the registry's own connection."""

    def __init__(self, devices: DeviceStore) -> None:
        self._conn = devices.connection
        self._ready = False

    def _ensure(self) -> None:
        if not self._ready:
            self._conn.executescript(_SCHEMA)
            self._ready = True

    def _exists(self) -> bool:
        if self._ready:
            return True
        found = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pieces'").fetchone()
        self._ready = found is not None
        return self._ready

    def list(self) -> list[Piece]:
        """Every stored piece, oldest first. Empty, and nothing created, before the first write."""
        if not self._exists():
            return []
        rows = self._conn.execute("SELECT * FROM pieces ORDER BY created_at, id").fetchall()
        return [_from_row(row) for row in rows]

    def get(self, piece_id: str) -> Piece | None:
        if not self._exists():
            return None
        row = self._conn.execute("SELECT * FROM pieces WHERE id = ?", (str(piece_id)[:64],)).fetchone()
        return _from_row(row) if row is not None else None

    def save(self, piece: Piece) -> None:
        """Insert or replace one piece. Another piece's node key is an ``sqlite3.IntegrityError``."""
        piece.check()
        self._ensure()
        with self._conn:
            self._conn.execute(
                "INSERT INTO pieces (id, kind, state, name, address, address_class, node_pub, tls_spki_sha256,"
                " card_sha256, card_json, uses, secret_env, source, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind, state=excluded.state, name=excluded.name,"
                " address=excluded.address, address_class=excluded.address_class, node_pub=excluded.node_pub,"
                " tls_spki_sha256=excluded.tls_spki_sha256, card_sha256=excluded.card_sha256,"
                " card_json=excluded.card_json, uses=excluded.uses, secret_env=excluded.secret_env,"
                " source=excluded.source, updated_at=excluded.updated_at",
                (piece.id, piece.kind, piece.state, piece.name, piece.address, piece.address_class, piece.node_pub,
                 piece.tls_spki_sha256, piece.card_sha256, json.dumps(piece.card, sort_keys=True),
                 json.dumps(list(piece.uses)), piece.secret_env, piece.source, piece.created_at,
                 piece.updated_at))

    def forget(self, piece_id: str) -> bool:
        """Remove one piece. True if there was one."""
        if not self._exists():
            return False
        with self._conn:
            cursor = self._conn.execute("DELETE FROM pieces WHERE id = ?", (str(piece_id)[:64],))
        return cursor.rowcount > 0
