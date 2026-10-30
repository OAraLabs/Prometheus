"""``GET /api/pieces``: the fleet view of the pieces Prometheus connects out to.

Contract: docs/contracts/pieces.md, section 8. Operator-only (``identity.is_operator``): a scoped device gets a 403
``operator_only`` from default-deny, never a 401. Until something approves a piece the list is empty, and reading it
creates no table.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

pytest.importorskip("fastapi")

from prometheus.config.piece_store import Piece, PieceStore  # noqa: E402
from tests.support.pairing_world import World  # noqa: E402

NODE_PUB = base64.b64encode(bytes(range(32))).decode("ascii")
ENTRY_KEYS = {"id", "name", "kind", "state", "source", "verified", "editable", "fp", "tls_spki_sha256", "address",
              "address_class", "offers", "uses", "health", "created_at", "updated_at", "last_seen"}


def save(world: World, **over) -> None:
    fields = dict(
        id="pc_0123456789abcdef", kind="piece", state="approved", name="Kitchen voice",
        address="http://kitchen.example:8001", address_class="mesh", node_pub=NODE_PUB, tls_spki_sha256=None,
        card_sha256="ab" * 32,
        card={"pieceCardVersion": "1", "name": "Kitchen voice", "offers": [
            {"kind": "stt", "api": "openai.audio.transcriptions",
             "url": "http://kitchen.example:8001/v1/audio/transcriptions", "auth": "bearer"},
            {"kind": "tts", "api": "openai.audio.speech", "url": "http://kitchen.example:8003/v1/audio/speech"}]},
        uses=("stt",), secret_env="PROMETHEUS_PIECE_PC_0123456789ABCDEF_KEY", source="manual",
        created_at=1_760_000_000.0, updated_at=1_760_000_060.0)
    fields.update(over)
    PieceStore(world.devices).save(Piece(**fields))


def test_an_empty_registry_answers_the_contracts_shape(tmp_path):
    world = World(tmp_path, recording=False)
    response = world.as_("global", "GET", "/api/pieces")
    assert response.status_code == 200
    assert response.json() == {"v": 1, "pieces": [], "pending_limit": 16}
    assert response.headers["cache-control"] == "no-store"
    assert "pieces" not in world.tables()


def test_an_owner_device_may_read_it(tmp_path):
    assert World(tmp_path, recording=False).as_("owner", "GET", "/api/pieces").status_code == 200


def test_a_scoped_device_is_refused(tmp_path):
    response = World(tmp_path, recording=False).as_("scoped", "GET", "/api/pieces")
    assert response.status_code == 403
    assert response.json()["error"] == "operator_only"


def test_no_token_is_a_401(tmp_path):
    assert World(tmp_path, recording=False).client.get("/api/pieces").status_code == 401


def test_an_approved_piece_is_listed_as_the_contract_says(tmp_path):
    world = World(tmp_path, recording=False)
    save(world)
    [entry] = world.as_("global", "GET", "/api/pieces").json()["pieces"]
    assert set(entry) == ENTRY_KEYS
    assert entry["id"] == "pc_0123456789abcdef"
    assert entry["kind"] == "piece" and entry["state"] == "approved" and entry["source"] == "manual"
    assert entry["verified"] is True
    assert entry["editable"] is True
    assert entry["fp"] == hashlib.sha256(bytes(range(32))).hexdigest()[:16]
    assert entry["address"] == "http://kitchen.example:8001" and entry["address_class"] == "mesh"
    assert entry["offers"] == [
        {"kind": "stt", "api": "openai.audio.transcriptions",
         "url": "http://kitchen.example:8001/v1/audio/transcriptions"},
        {"kind": "tts", "api": "openai.audio.speech", "url": "http://kitchen.example:8003/v1/audio/speech"}]
    assert entry["uses"] == ["stt"]
    assert entry["health"] == {"state": "unknown", "since": None}       # no health watch yet
    assert entry["created_at"] == "2025-10-09T08:53:20Z"
    assert entry["updated_at"] == "2025-10-09T08:54:20Z"
    assert entry["last_seen"] is None


def test_a_piece_without_a_key_is_not_verified(tmp_path):
    world = World(tmp_path, recording=False)
    save(world, node_pub=None, address="https://kitchen.example:8443", address_class="lan",
         tls_spki_sha256="cd" * 32)
    [entry] = world.as_("global", "GET", "/api/pieces").json()["pieces"]
    assert entry["verified"] is False
    assert entry["fp"] is None
    assert entry["tls_spki_sha256"] == "cd" * 32


def test_the_secret_is_never_named_or_shown(tmp_path):
    """The env variable's name is the daemon's business; the value never leaves the env file."""
    world = World(tmp_path, recording=False)
    save(world)
    text = world.as_("global", "GET", "/api/pieces").text
    assert "PROMETHEUS_PIECE" not in text
    assert "secret" not in text


def test_offers_the_contract_does_not_know_are_left_out(tmp_path):
    world = World(tmp_path, recording=False)
    save(world, card={"pieceCardVersion": "1", "name": "x", "offers": [
        {"kind": "gpu", "api": "openai.audio.speech", "url": "http://kitchen.example:8003/v1/audio/speech"},
        {"kind": "tts", "api": "something.else", "url": "http://kitchen.example:8003/x"},
        "not an offer",
        {"kind": "tts", "api": "openai.audio.speech", "url": "http://kitchen.example:8003/v1/audio/speech"}]})
    [entry] = world.as_("global", "GET", "/api/pieces").json()["pieces"]
    assert entry["offers"] == [
        {"kind": "tts", "api": "openai.audio.speech", "url": "http://kitchen.example:8003/v1/audio/speech"}]
