"""'Is this request from this machine' is answered from the raw TCP peer, and a forwarded request is not.

THE HOLE (M1/M2 of the same review)
-----------------------------------
The same-Mac actions (``POST /api/pair/local``, ``POST /api/setup/pair`` with the file secret, and the global
token minting an OWNER device) trust ``is_loopback_peer``: the ASGI scope's ``client``. Two things make that a
weaker test than it looks:

* an unlisted reverse proxy on this machine (``tailscale serve``, ``cloudflared``, nginx) connects from
  127.0.0.1 and relays a request that came from anywhere, so a remote request has a loopback peer;
* uvicorn REWRITES ``scope["client"]`` from ``X-Forwarded-For`` when the peer is in ``web.trusted_proxies``,
  so a range that is too wide lets a forged header make a remote caller look like 127.0.0.1.

The fix does not try to be cleverer about which proxy is honest. A request that carries ``X-Forwarded-For`` or
``Forwarded`` was relayed by something, so it is not "this machine", whatever its peer says. A proxy that
strips both is still caught by the Host-header test the secret routes already make.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config import local_pairing as lp  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.setup_server import PairingState, create_setup_app  # noqa: E402

TOKEN = "same-machine-test-token-0123456789abcdef"
REMOTE = "203.0.113.9"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))


@pytest.fixture
def secret(tmp_path, monkeypatch):
    directory = tmp_path / "pairing"
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(directory))
    return lp.mint_secret(directory)


def scope(peer: str | None = "127.0.0.1", *headers: tuple[str, str]) -> dict:
    return {"type": "http", "client": (peer, 50123) if peer else None,
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}


# ── the test itself ──────────────────────────────────────────────────────────

def same_machine(*args, **kw):
    from prometheus.web.loopback import is_same_machine

    return is_same_machine(*args, **kw)


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "::ffff:127.0.0.1", "127.0.0.2"])
def test_a_loopback_peer_that_relayed_nothing_is_this_machine(peer):
    assert same_machine(scope(peer)) is True


@pytest.mark.parametrize("peer", [REMOTE, "192.168.1.20", "fe80::1", "testclient", "", None])
def test_any_other_peer_is_not(peer):
    assert same_machine(scope(peer)) is False


@pytest.mark.parametrize("header", [("X-Forwarded-For", REMOTE), ("x-forwarded-for", "127.0.0.1"),
                                    ("Forwarded", f"for={REMOTE}"), ("FORWARDED", "for=127.0.0.1;proto=https")])
def test_a_relayed_request_is_not_this_machine_whatever_its_peer_says(header):
    assert same_machine(scope("127.0.0.1", header)) is False


def test_even_a_relayed_request_that_names_loopback_is_not():
    """A forged header cannot make a remote caller local, and a true one still means something relayed it."""
    assert same_machine(scope("127.0.0.1", ("X-Forwarded-For", "127.0.0.1"))) is False


def test_it_takes_a_request_or_a_scope():
    class Request:
        def __init__(self, s):
            self.scope = s

    assert same_machine(Request(scope("127.0.0.1"))) is True
    assert same_machine(Request(scope("127.0.0.1", ("Forwarded", "for=x")))) is False
    assert same_machine("not a scope") is False


def test_unrelated_headers_do_not_matter():
    assert same_machine(scope("127.0.0.1", ("Host", "localhost:8005"), ("Accept", "*/*"))) is True


# ── POST /api/pair/local ─────────────────────────────────────────────────────

def _daemon():
    return create_app({"web": {"api_token": TOKEN, "api_port": 8123, "ws_port": 8124}})


def _client(app, peer="127.0.0.1") -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:8123", client=(peer, 50123))


@pytest.mark.parametrize("header", [{"X-Forwarded-For": REMOTE}, {"Forwarded": f"for={REMOTE}"}])
def test_the_same_mac_secret_is_not_accepted_from_a_relayed_request(secret, header):
    response = _client(_daemon()).post("/api/pair/local", json={"code": secret}, headers=header)
    assert response.status_code == 403 and response.json()["error"] == "not_loopback"


def test_and_a_refused_relay_does_not_burn_the_secret(secret):
    app = _daemon()
    refused = _client(app).post("/api/pair/local", json={"code": secret}, headers={"X-Forwarded-For": REMOTE})
    assert refused.status_code == 403
    accepted = _client(app).post("/api/pair/local", json={"code": secret})
    assert accepted.status_code == 200, "the real client on this Mac still gets its credential"


def test_the_same_mac_secret_still_works_from_this_mac(secret):
    assert _client(_daemon()).post("/api/pair/local", json={"code": secret}).status_code == 200


# ── POST /api/setup/pair (setup mode) ────────────────────────────────────────

def test_the_secret_in_setup_mode_is_not_accepted_from_a_relayed_request(secret):
    app = create_setup_app(PairingState(code="042999"), api_port=8123, ws_port=8124)
    refused = _client(app).post("/api/setup/pair", json={"code": secret}, headers={"X-Forwarded-For": REMOTE})
    assert refused.status_code == 403 and refused.json()["error"] == "not_loopback"
    assert _client(app).post("/api/setup/pair", json={"code": secret}).status_code == 200, "and it was not burned"


# ── the global token minting an OWNER device ─────────────────────────────────

def _mint_owner(client, **headers):
    return client.post("/api/devices", json={"name": "Beacon", "owner": True},
                       headers={"Authorization": f"Bearer {TOKEN}", **headers})


def test_an_owner_device_is_minted_from_this_machine():
    response = _mint_owner(_client(_daemon()))
    assert response.status_code == 201 and response.json().get("token")


@pytest.mark.parametrize("header", [{"X-Forwarded-For": REMOTE}, {"Forwarded": f"for={REMOTE}"}])
def test_but_not_through_a_relay_even_with_the_global_token(header):
    response = _mint_owner(_client(_daemon()), **header)
    assert response.status_code == 403 and "owner devices are issued only for this Mac" in response.json()["error"]


def test_an_ordinary_scoped_device_can_still_be_enrolled_through_a_relay():
    """Only the OWNER tier needs the machine; a scoped device is the safe default for any other computer."""
    response = _client(_daemon()).post(
        "/api/devices", json={"name": "Other computer"},
        headers={"Authorization": f"Bearer {TOKEN}", "X-Forwarded-For": REMOTE})
    assert response.status_code == 201
