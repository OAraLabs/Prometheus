"""The ``pieces`` table in ``devices.db``: the owner's approved (and ignored) pieces.

Contract: docs/contracts/pieces.md, section 12. The table appears on the first write, like ``pair_requests``, so a
daemon that never sees a piece keeps the exact ``devices.db`` schema the parity fixtures record; reading before that
finds nothing and creates nothing.
"""

from __future__ import annotations

import base64
import hashlib
import sqlite3

import pytest

from prometheus.config.device_store import DeviceStore
from prometheus.config.piece_store import Piece, PieceStore, node_fp

NODE_PUB = base64.b64encode(bytes(range(32))).decode("ascii")


def tables(path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}


def piece(**over) -> Piece:
    fields = dict(
        id="pc_0123456789abcdef", kind="piece", state="approved", name="Kitchen voice",
        address="http://kitchen.example:8001", address_class="mesh", node_pub=NODE_PUB, tls_spki_sha256=None,
        card_sha256="ab" * 32,
        card={"pieceCardVersion": "1", "name": "Kitchen voice", "offers": [
            {"kind": "stt", "api": "openai.audio.transcriptions", "url": "http://kitchen.example:8001/v1/audio/transcriptions"}]},
        uses=("stt",), secret_env=None, source="manual", created_at=1_760_000_000.0, updated_at=1_760_000_000.0)
    fields.update(over)
    return Piece(**fields)


@pytest.fixture
def store(tmp_path):
    return PieceStore(DeviceStore(tmp_path / "devices.db")), tmp_path / "devices.db"


def test_reading_an_empty_registry_creates_no_table(store):
    pieces, path = store
    assert pieces.list() == []
    assert pieces.get("pc_0123456789abcdef") is None
    assert "pieces" not in tables(path)


def test_a_saved_piece_reads_back_whole(store):
    pieces, path = store
    pieces.save(piece())
    assert "pieces" in tables(path)
    assert pieces.list() == [piece()]
    assert pieces.get("pc_0123456789abcdef") == piece()


def test_saving_again_replaces_the_row(store):
    pieces, _ = store
    pieces.save(piece())
    pieces.save(piece(name="Voice in the kitchen", uses=("stt", "tts"), updated_at=1_760_000_100.0))
    assert [p.name for p in pieces.list()] == ["Voice in the kitchen"]
    assert pieces.list()[0].uses == ("stt", "tts")


def test_one_node_key_is_one_piece(store):
    pieces, _ = store
    pieces.save(piece())
    with pytest.raises(sqlite3.IntegrityError):
        pieces.save(piece(id="pc_fedcba9876543210"))


def test_pieces_without_a_key_do_not_collide(store):
    pieces, _ = store
    pieces.save(piece(node_pub=None))
    pieces.save(piece(id="pc_fedcba9876543210", node_pub=None, address="https://other.example:8443"))
    assert len(pieces.list()) == 2


def test_forget_removes_the_row(store):
    pieces, _ = store
    pieces.save(piece())
    assert pieces.forget("pc_0123456789abcdef") is True
    assert pieces.forget("pc_0123456789abcdef") is False
    assert pieces.list() == []


@pytest.mark.parametrize("field,value", [
    ("kind", "peer"), ("state", "pending"), ("source", "config_file"), ("address_class", "tailnet"),
    ("uses", ("gpu",)), ("name", ""),
])
def test_a_value_outside_the_contract_is_refused(store, field, value):
    pieces, path = store
    with pytest.raises(ValueError):
        pieces.save(piece(**{field: value}))
    assert "pieces" not in tables(path)


def test_the_fingerprint_is_the_contracts():
    """Section 3.1: the first 16 hex of SHA-256 over the raw 32-byte node public key."""
    assert node_fp(NODE_PUB) == hashlib.sha256(bytes(range(32))).hexdigest()[:16]
    assert node_fp(None) is None
    assert node_fp("not base64!") is None
    assert node_fp(base64.b64encode(b"short").decode()) is None
