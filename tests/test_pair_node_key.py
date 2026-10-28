"""Pairing hands the new device the node key to pin, and the owner's 4-digit check covers it.

A client that will check ``POST /api/identity/challenge`` on every connect needs the node key it is checking
against, and it must not be trust-on-first-use. So every route a device joins through answers with
``instance_id`` and ``node_pub`` (equal to ``/api/status``'s), and on the approval path a requester that asks for
match code version 2 gets a code that also hashes ``node_pub``: the owner comparing four digits then confirms the
node key too. Version 1 stays for requesters that don't ask, unchanged, and a version-2 requester refuses a
version-1 answer (``docs/PAIRING-APPROVAL-API.md`` 4.4).

The version-2 derivation is computed HERE, independently of the server's, as Beacon computes it.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

pytest.importorskip("fastapi")

from prometheus.config import pair_seal  # noqa: E402
from prometheus.config.node_identity import ensure_node_identity, set_instance_id  # noqa: E402
from tests.support.pairing_world import Requester, World  # noqa: E402

INSTANCE = "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def world(tmp_path):
    ensure_node_identity()
    set_instance_id(INSTANCE)
    yield World(tmp_path)
    set_instance_id(None)


def _v2(device_public: bytes, instance_der: bytes, node_pub_b64: str, request_id: str) -> str:
    digest = hashlib.sha256(
        b"prometheus-pair-v2" + b"\x00" + device_public + instance_der + base64.b64decode(node_pub_b64)
        + request_id.encode("ascii")
    ).digest()
    return f"{int.from_bytes(digest[:4], 'big') % 10000:04d}"


def _status(world: World) -> dict:
    return world.as_("global", "GET", "/api/status").json()


# ── the approval path ────────────────────────────────────────────────────────

def test_a_version_2_request_gets_a_code_over_the_node_key_too(world):
    created, requester, _ = world.created(match_code_version=2)
    status = _status(world)
    assert created["match_code_version"] == 2
    assert created["node_pub"] == status["node_pub"] and created["instance_id"] == INSTANCE
    assert created["match_code"] == _v2(requester.public, world.der, created["node_pub"], created["request_id"])
    shown = world.as_("global", "GET", "/api/pair/requests").json()
    rows = shown if isinstance(shown, list) else shown.get("requests", [])
    assert [r["match_code"] for r in rows if r.get("request_id", r.get("id")) == created["request_id"]] == [
        created["match_code"]], "the owner is shown the same version-2 code"


def test_a_request_that_does_not_ask_keeps_version_1_unchanged(world):
    created, requester, _ = world.created()
    assert created["match_code_version"] == 1
    assert created["match_code"] == pair_seal.match_code(requester.public, world.der, created["request_id"])
    assert created["node_pub"] == _status(world)["node_pub"], "the key is still handed over, to pin"


def test_the_version_2_code_changes_when_the_node_key_changes(world):
    device = Requester().public
    a = base64.b64encode(b"\x01" * 32).decode()
    b = base64.b64encode(b"\x02" * 32).decode()
    rid = "0" * 32
    assert pair_seal.match_code(device, world.der, rid, node_pub=base64.b64decode(a), version=2) == _v2(
        device, world.der, a, rid)
    assert pair_seal.match_code(device, world.der, rid, node_pub=base64.b64decode(a), version=2) != \
        pair_seal.match_code(device, world.der, rid, node_pub=base64.b64decode(b), version=2)
    # Version 1 never reads it.
    assert pair_seal.match_code(device, world.der, rid) == pair_seal.match_code(device, world.der, rid, version=1)


def test_a_null_version_is_the_same_as_none_given(world):
    created, _, _ = world.created(match_code_version=None)
    assert created["match_code_version"] == 1


@pytest.mark.parametrize("version", [0, 3, "2", 2.0, True])
def test_an_unknown_match_code_version_is_a_400_that_names_the_field(world, version):
    response, _, _ = world.request(match_code_version=version)
    assert response.status_code == 400, response.text
    assert "match_code_version" in response.json().get("fields", [])


def test_version_2_without_a_node_key_is_refused_and_creates_nothing(tmp_path):
    from prometheus.config.paths import get_node_dir

    w = World(tmp_path)
    assert not (get_node_dir() / "node.key").exists()
    response, _, _ = w.request(match_code_version=2)
    assert response.status_code == 503 and response.json()["error"] == "identity_unavailable"
    assert not (get_node_dir() / "node.key").exists(), "a request never mints the node key"
    assert w.as_("global", "GET", "/api/pair/requests").json() in ([], {"requests": []})


def test_the_approved_poll_carries_the_instance_and_node_key(world):
    created, _, _ = world.created(match_code_version=2)
    world.approve(created)
    body = world.poll(created).json()
    status = _status(world)
    assert body["status"] == "approved"
    assert body["instance_id"] == status["instance_id"] == INSTANCE
    assert body["node_pub"] == status["node_pub"]


# ── the owner code and same-Mac pairing ─────────────────────────────────────

def test_an_owner_code_redemption_carries_the_instance_and_node_key(tmp_path):
    from tests.test_owner_code_routes import minted, redeem

    ensure_node_identity()
    set_instance_id(INSTANCE)
    try:
        w = World(tmp_path, bridge=True)
        w.app.state.owner_codes.clock = w.clock
        w.app.state.owner_codes.wall = w.clock
        response = redeem(w, minted(w)["code"])
        assert response.status_code == 200, response.text
        body = response.json()
        status = _status(w)
        assert body["instance_id"] == status["instance_id"] == INSTANCE
        assert body["node_pub"] == status["node_pub"]
    finally:
        set_instance_id(None)


def test_same_mac_pairing_carries_the_instance_and_node_key(tmp_path, monkeypatch):
    from prometheus.config import local_pairing as lp
    from tests.test_owner_credentials import pair_local

    ensure_node_identity()
    set_instance_id(INSTANCE)
    try:
        directory = tmp_path / "pairing"
        monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(directory))
        w = World(tmp_path)
        body = pair_local(w, lp.mint_secret(directory))
        status = _status(w)
        assert body["instance_id"] == status["instance_id"] == INSTANCE
        assert body["node_pub"] == status["node_pub"]
    finally:
        set_instance_id(None)
