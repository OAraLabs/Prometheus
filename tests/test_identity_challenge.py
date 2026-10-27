"""``POST /api/identity/challenge``: the instance signs a client's nonce with its node key.

A paired client (Beacon) sends a fresh nonce on every connect and checks the answer before it sends its token:
the signature must verify against the node key it pinned at pairing, over exactly the message below, and on a
TLS listener ``tls_spki_sha256`` must match the certificate it saw. FOUNDATION Part 3 (v4) allows the node key
to sign these challenges; it is still never proof of the user (the bearer token is).

The signed message is built HERE, independently of the server's builder, so a change to the wire format can't
pass by changing both sides at once.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time

import pytest

pytest.importorskip("fastapi")
from cryptography.exceptions import InvalidSignature  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.node_identity import ensure_node_identity, set_instance_id  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

ROUTE = "/api/identity/challenge"
INSTANCE = "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture(autouse=True)
def _clear_instance_pin():
    yield
    set_instance_id(None)


def _nonce(n: int = 32) -> str:
    return base64.urlsafe_b64encode(os.urandom(n)).rstrip(b"=").decode("ascii")


def _message(domain: str, nonce: str, body: dict) -> bytes:
    return (
        f"{domain}\n"
        f"nonce={nonce}\n"
        f"instance_id={body['instance_id'] or ''}\n"
        f"node_pub={body['node_pub']}\n"
        f"tls_spki_sha256={body['tls_spki_sha256'] or ''}\n"
        f"signed_at={body['signed_at']}\n"
    ).encode("utf-8")


def _verifies(body: dict, message: bytes) -> bool:
    key = Ed25519PublicKey.from_public_bytes(base64.b64decode(body["node_pub"]))
    try:
        key.verify(base64.b64decode(body["sig"]), message)
        return True
    except InvalidSignature:
        return False


def _client(https: bool = False) -> TestClient:
    return TestClient(create_app({}), base_url="https://testserver" if https else "http://testserver")


def _ask(client: TestClient, nonce: str | None = None, **extra):
    return client.post(ROUTE, json={"v": 1, "nonce": nonce if nonce is not None else _nonce(), **extra})


def test_a_good_nonce_gets_a_signature_over_exactly_the_documented_message():
    ensure_node_identity()
    set_instance_id(INSTANCE)
    nonce = _nonce()
    r = _ask(_client(), nonce)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["v"] == 1
    assert set(body) == {"v", "instance_id", "node_pub", "tls_spki_sha256", "signed_at", "sig"}
    assert isinstance(body["signed_at"], int) and abs(body["signed_at"] - time.time()) < 60
    assert _verifies(body, _message("prometheus-identity-challenge/v1", nonce, body))


def test_instance_id_and_node_pub_are_status_s():
    ensure_node_identity()
    set_instance_id(INSTANCE)
    client = _client()
    body = _ask(client).json()
    status = client.get("/api/status").json()
    assert body["instance_id"] == status["instance_id"] == INSTANCE
    assert body["node_pub"] == status["node_pub"]


def test_instance_id_is_null_without_an_adopted_vault_and_is_signed_as_empty():
    ensure_node_identity()
    nonce = _nonce()
    body = _ask(_client(), nonce).json()
    assert body["instance_id"] is None
    assert _verifies(body, _message("prometheus-identity-challenge/v1", nonce, body))


def test_tls_spki_is_the_served_certificate_s_on_tls_and_null_on_plain():
    from prometheus.config.instance_key import ensure_instance_key

    ensure_node_identity()
    spki_der = ensure_instance_key()       # the TLS certificate is made over this key
    nonce = _nonce()
    over_tls = _ask(_client(https=True), nonce).json()
    assert over_tls["tls_spki_sha256"] == hashlib.sha256(spki_der).hexdigest()
    assert _verifies(over_tls, _message("prometheus-identity-challenge/v1", nonce, over_tls))
    plain = _ask(_client(https=False)).json()
    assert plain["tls_spki_sha256"] is None


@pytest.mark.parametrize("nonce", [_nonce(31), _nonce(33), "", "not base64!", "a" * 400])
def test_a_nonce_that_is_not_32_bytes_is_400(nonce):
    ensure_node_identity()
    r = _ask(_client(), nonce)
    assert r.status_code == 400 and r.json()["error"] == "bad_nonce"


def test_no_body_is_400_and_the_route_needs_no_token_and_sets_no_cookie():
    ensure_node_identity()
    client = TestClient(create_app({"web": {"api_token": "a-token-the-client-does-not-have"}}))
    assert client.post(ROUTE).status_code == 400
    r = client.post(ROUTE, json={"v": 1, "nonce": _nonce()})
    assert r.status_code == 200, "answerable before a client sends its token"
    assert "set-cookie" not in {k.lower() for k in r.headers}


def test_a_browser_is_refused():
    ensure_node_identity()
    r = _client().post(ROUTE, json={"v": 1, "nonce": _nonce()}, headers={"Origin": "https://example.com"})
    assert r.status_code == 400 and r.json()["error"] == "browser_not_allowed"


def test_different_nonces_give_different_signatures_and_the_same_nonce_signs_again():
    ensure_node_identity()
    client = _client()
    a, b = _nonce(), _nonce()
    one, two = _ask(client, a).json(), _ask(client, b).json()
    assert one["sig"] != two["sig"]
    again = _ask(client, a).json()             # stateless: replay protection is the client's fresh nonce
    assert _verifies(again, _message("prometheus-identity-challenge/v1", a, again))


def test_before_a_node_key_exists_it_is_503_and_mints_nothing():
    from prometheus.config.paths import get_node_dir

    r = _ask(_client())
    assert r.status_code == 503 and r.json()["error"] == "no_node_key"
    assert not (get_node_dir() / "node.key").exists(), "a request must never mint the node identity"


def test_the_domain_line_separates_challenge_signatures_from_audit_ones():
    ensure_node_identity()
    nonce = _nonce()
    body = _ask(_client(), nonce).json()
    # The same fields under another domain line are a different message: a challenge signature proves nothing
    # there, so it can't be replayed as an audit record's (or the reverse).
    assert not _verifies(body, _message("prometheus-audit-record/v1", nonce, body))


def test_it_is_a_public_route_and_rate_limited_with_hello():
    from prometheus.web.public_routes import is_public_route

    assert is_public_route("POST", ROUTE)
    ensure_node_identity()
    client = _client()
    codes = [_ask(client).status_code for _ in range(70)]
    assert 429 in codes, "shares hello's per-peer limit (60 a minute)"
