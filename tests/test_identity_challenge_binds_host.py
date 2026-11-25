"""The identity challenge v2 binds the address the client dialled into what the daemon signs.

THE HOLE (GHSA-75cg-fxc2-f93j, private). ``POST /api/identity/challenge`` v1 signs the client's nonce, the instance id,
the node key, the TLS certificate digest and the time. It does not sign, and the daemon does not look at, the address the
client used to reach it. On a plain listener, or any pairing where the client has no pin yet, anything that can relay
the client's request to the real daemon gets a genuine signature back; the client sees the real node key, the real
signature, a fresh nonce, and sends its device token to the relay.

THE FIX. A v2 request carries the host the client dialled. The daemon

  * refuses a v2 request whose body host is not the ``Host`` header it actually received (a relay that rewrites the
    header, as most HTTP proxies do, is caught here);
  * refuses a host that is not one of its OWN names (a relay that keeps the header, as a TCP forwarder does, asks for a
    name the daemon does not have);
  * signs the host into a message under its own domain line ``prometheus-identity-challenge/v2``.

The client checks that the signed host is the one it dialled, and that the answer is v2 at all. v1 is unchanged for the
clients that exist today, and hello says which version a daemon speaks (``ic``) so a phone can require v2 for an
unpinned pairing before it sends anything.

WHY THE OWN-NAMES CHECK IS NOT OPTIONAL. Signing whatever host the client sends does nothing against a relay that
passes the request through unchanged: the relay's name is the host the client dialled, the daemon signs it, the client
sees a match. ``test_signing_any_host_would_not_stop_a_host_preserving_relay`` pins that, so nobody removes the check
and believes the binding still holds.

WHAT THIS DOES NOT DO (pinned below as limits, not hidden): a party on the path of the daemon's REAL address (ARP or
DNS spoofing of the name or IP the client dials, a router) relays under the daemon's own name and passes. Only a pin
(the TLS certificate digest) stops that, and on a TLS listener the v2 message carries it too.

The messages are built HERE, independently of the server's builder.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")
from cryptography.exceptions import InvalidSignature  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.node_identity import ensure_node_identity, set_instance_id  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

ROUTE = "/api/identity/challenge"
INSTANCE = "550e8400-e29b-41d4-a716-446655440000"
V1 = "prometheus-identity-challenge/v1"
V2 = "prometheus-identity-challenge/v2"


@pytest.fixture(autouse=True)
def _identity():
    ensure_node_identity()
    set_instance_id(INSTANCE)
    yield
    set_instance_id(None)


def _nonce() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode("ascii")


def _v2_message(nonce: str, body: dict, host: str) -> bytes:
    return (
        f"{V2}\n"
        f"nonce={nonce}\n"
        f"instance_id={body['instance_id'] or ''}\n"
        f"node_pub={body['node_pub']}\n"
        f"tls_spki_sha256={body['tls_spki_sha256'] or ''}\n"
        f"host={host}\n"
        f"signed_at={body['signed_at']}\n"
    ).encode("utf-8")


def _v1_message(nonce: str, body: dict) -> bytes:
    return (
        f"{V1}\n"
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


def _app(extra_hosts=()):
    config = {"web": {"identity_hosts": list(extra_hosts)}} if extra_hosts else {}
    return create_app(config)


def _client(app=None, base="http://127.0.0.1:8005") -> TestClient:
    return TestClient(app or _app(), base_url=base)


def _ask(client, *, nonce=None, host="127.0.0.1", v=2, headers=None, **extra):
    body = {"v": v, "nonce": nonce or _nonce(), **extra}
    if host is not None:
        body["host"] = host
    return client.post(ROUTE, json=body, headers=headers or {})


def accepts_v2(answer: dict, *, nonce: str, dialled: str, pinned_node_pub: str | None = None) -> bool:
    """What a client that REQUIRES v2 does with an answer. Written here, independent of the server."""
    if answer.get("v") != 2 or "host" in answer and answer["host"] != dialled:
        return False
    if pinned_node_pub is not None and answer.get("node_pub") != pinned_node_pub:
        return False
    return _verifies(answer, _v2_message(nonce, answer, dialled)) and abs(answer["signed_at"] - time.time()) < 300


# ── the wire format ──────────────────────────────────────────────────────────

def test_a_v2_request_gets_a_signature_over_exactly_the_documented_seven_lines():
    nonce = _nonce()
    r = _ask(_client(), nonce=nonce)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["v"] == 2
    assert set(body) == {"v", "instance_id", "node_pub", "tls_spki_sha256", "host", "signed_at", "sig"}
    assert body["host"] == "127.0.0.1"
    assert _verifies(body, _v2_message(nonce, body, "127.0.0.1"))


def test_the_v2_signature_cannot_be_replayed_as_v1_nor_the_reverse():
    nonce = _nonce()
    v2 = _ask(_client(), nonce=nonce).json()
    assert not _verifies(v2, _v1_message(nonce, v2)), "a v2 signature verified as a v1 message"
    v1 = _ask(_client(), nonce=nonce, v=1, host=None).json()
    assert not _verifies(v1, _v2_message(nonce, v1, "127.0.0.1")), "a v1 signature verified as a v2 message"


def test_the_signature_is_bound_to_the_host_it_was_made_for():
    nonce = _nonce()
    body = _ask(_client(), nonce=nonce).json()
    assert accepts_v2(body, nonce=nonce, dialled="127.0.0.1")
    assert not accepts_v2(body, nonce=nonce, dialled="localhost"), "an answer for one host was accepted for another"


def test_v1_is_unchanged_for_the_clients_that_exist_today():
    nonce = _nonce()
    r = _ask(_client(), nonce=nonce, v=1, host=None)
    body = r.json()
    assert r.status_code == 200 and body["v"] == 1
    assert set(body) == {"v", "instance_id", "node_pub", "tls_spki_sha256", "signed_at", "sig"}
    assert _verifies(body, _v1_message(nonce, body))


def test_a_request_with_no_version_at_all_still_gets_the_v1_answer():
    r = _client().post(ROUTE, json={"nonce": _nonce()})
    assert r.status_code == 200 and r.json()["v"] == 1


def test_a_host_field_on_a_v1_request_is_ignored_not_signed():
    nonce = _nonce()
    body = _ask(_client(), nonce=nonce, v=1, host="evil.example").json()
    assert body["v"] == 1 and "host" not in body
    assert _verifies(body, _v1_message(nonce, body))


def test_over_tls_the_v2_message_carries_the_certificate_digest_and_the_host():
    from prometheus.config.instance_key import ensure_instance_key

    der = ensure_instance_key()
    nonce = _nonce()
    body = _ask(_client(base="https://127.0.0.1:8006"), nonce=nonce).json()
    assert body["tls_spki_sha256"] == hashlib.sha256(der).hexdigest()
    assert _verifies(body, _v2_message(nonce, body, "127.0.0.1"))


# ── what is refused ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", [None, "", 12, ["127.0.0.1"], "127.0.0.1:8005", "LOCALHOST", "localhost.", " localhost",
                                  "local host", "a/b", "user@localhost", "[::1]", "x" * 300, "localhost\n", "127.0.0.1\x00"])
def test_a_v2_request_needs_a_canonical_host_in_the_body(host):
    r = _ask(_client(), host=host)
    assert r.status_code == 400 and r.json()["error"] == "bad_host", (host, r.text)


def test_the_ipv6_loopback_is_sent_without_brackets():
    nonce = _nonce()
    r = _ask(_client(base="http://[::1]:8005"), nonce=nonce, host="::1")
    assert r.status_code == 200 and r.json()["host"] == "::1"


def test_a_v2_request_is_never_answered_with_a_v1_signature():
    """Not a downgrade door: a v2 request that cannot be answered as v2 is an error, not a v1 answer."""
    for r in (_ask(_client(), host=None), _ask(_client(), host="not a host"), _ask(_client(), host="relay.example")):
        assert r.status_code >= 400 and "sig" not in r.json()


def test_the_body_host_must_be_the_host_header_the_daemon_received():
    """A relay that rewrites Host (most HTTP proxies do) is caught here."""
    r = _ask(_client(), host="localhost", headers={"Host": "127.0.0.1:8005"})
    assert r.status_code == 400 and r.json()["error"] == "host_mismatch"
    assert _ask(_client(), host="localhost", headers={"Host": "LocalHost:8005"}).status_code == 200


def test_a_forwarded_host_header_is_never_believed():
    r = _ask(_client(), host="relay.example", headers={"X-Forwarded-Host": "relay.example"})
    assert r.status_code != 200
    r = _ask(_client(), host="127.0.0.1", headers={"X-Forwarded-Host": "relay.example"})
    assert r.status_code == 200 and r.json()["host"] == "127.0.0.1"


# ── the relay ────────────────────────────────────────────────────────────────

def _relay_to(real_client, relay_name: str):
    """A relay at *relay_name* that passes a client's request to the real daemon UNCHANGED, Host included (a TCP
    forwarder; nginx with ``proxy_set_header Host $host``). Returns what the client would get back."""
    def forward(body: dict):
        return real_client.post(ROUTE, json=body, headers={"Host": relay_name})
    return forward


def test_a_relay_that_keeps_the_host_asks_the_daemon_to_sign_a_name_it_does_not_have():
    nonce = _nonce()
    forward = _relay_to(_client(base="http://relay.example"), "relay.example")
    r = forward({"v": 2, "nonce": nonce, "host": "relay.example"})
    assert r.status_code == 421 and r.json()["error"] == "host_not_mine"
    assert "sig" not in r.json()


def test_a_relay_that_rewrites_the_host_gets_a_signature_for_the_wrong_host():
    """It forwards to the daemon's real name; the client dialled the relay's. The signature is genuine and useless."""
    nonce = _nonce()
    real = _client()
    answer = real.post(ROUTE, json={"v": 2, "nonce": nonce, "host": "127.0.0.1"}, headers={"Host": "127.0.0.1:8005"}).json()
    assert not accepts_v2(answer, nonce=nonce, dialled="relay.example")


def test_a_relay_that_rewrites_only_the_body_host_is_caught_by_the_daemon():
    nonce = _nonce()
    r = _client(base="http://relay.example").post(ROUTE, json={"v": 2, "nonce": nonce, "host": "127.0.0.1"})
    assert r.status_code == 400 and r.json()["error"] == "host_mismatch"


def test_a_relay_that_downgrades_the_request_to_v1_gets_an_answer_a_v2_client_rejects():
    nonce = _nonce()
    answer = _ask(_client(), nonce=nonce, v=1, host=None).json()
    assert not accepts_v2(answer, nonce=nonce, dialled="127.0.0.1"), "a v1 answer passed a client that requires v2"


def test_an_old_daemon_answers_v2_with_v1_and_a_v2_client_rejects_it():
    """What an unpatched 0.9.18 to 0.9.20 daemon does with {"v": 2, "host": ...}: ignores both and answers v1."""
    nonce = _nonce()
    old = _ask(_client(), nonce=nonce, v=1, host=None).json()
    assert old["v"] == 1 and not accepts_v2(old, nonce=nonce, dialled="127.0.0.1")


def test_signing_any_host_would_not_stop_a_host_preserving_relay():
    """The reason the own-names check exists. Model a daemon that signs whatever host the client sent: the relay's
    name is what the client dialled, the daemon signs it, and the client's check passes."""
    nonce = _nonce()
    real = _client()
    honest = real.post(ROUTE, json={"v": 2, "nonce": nonce, "host": "127.0.0.1"}, headers={"Host": "127.0.0.1:8005"}).json()
    # a daemon without the check would have returned honest-with-host=relay.example and a signature over it; the
    # signature is the daemon's own key over a message naming the relay, which this test builds with the same key
    # material the real route used, by asking the real route for the message it would sign for that host:
    from prometheus.web import identity_challenge as mod

    forged_message = mod.challenge_message_v2(
        nonce=nonce, instance_id=honest["instance_id"], node_pub=honest["node_pub"], tls_spki_sha256=None,
        host="relay.example", signed_at=honest["signed_at"])
    assert forged_message == _v2_message(nonce, honest, "relay.example")
    # ...and it is only the daemon's REFUSAL that keeps that message from being signed:
    refused = _client(base="http://relay.example").post(ROUTE, json={"v": 2, "nonce": nonce, "host": "relay.example"})
    assert refused.status_code == 421


# ── what the daemon calls its own ────────────────────────────────────────────

def test_an_extra_name_in_the_config_is_this_daemons_and_only_that_name():
    app = _app(extra_hosts=["My-Server.tail1234.ts.net"])
    nonce = _nonce()
    r = _client(app, base="http://my-server.tail1234.ts.net").post(
        ROUTE, json={"v": 2, "nonce": nonce, "host": "my-server.tail1234.ts.net"})
    assert r.status_code == 200 and r.json()["host"] == "my-server.tail1234.ts.net"
    other = _client(app, base="http://relay.example").post(ROUTE, json={"v": 2, "nonce": _nonce(), "host": "relay.example"})
    assert other.status_code == 421


def test_a_bad_identity_hosts_value_adds_nothing_and_does_not_break_the_route():
    for value in ("relay.example", {"a": 1}, [1, None, "", "bad name", "a/b"], None):
        app = create_app({"web": {"identity_hosts": value}})
        r = _client(app, base="http://relay.example").post(ROUTE, json={"v": 2, "nonce": _nonce(), "host": "relay.example"})
        assert r.status_code == 421, (value, r.text)
        assert _client(app).post(ROUTE, json={"v": 2, "nonce": _nonce(), "host": "127.0.0.1"}).status_code == 200


def test_the_name_check_never_resolves_a_name():
    """A name that resolves to this machine is not thereby its name: the caller picks the name, and so the DNS
    answer (split-horizon, rebinding). Config names only."""
    import socket

    from prometheus.web.identity_challenge import is_own_name

    def boom(*a, **k):
        raise AssertionError("the name check resolved a host name")

    real = (socket.getaddrinfo, socket.gethostbyname, socket.getfqdn, socket.gethostbyaddr)
    socket.getaddrinfo = socket.gethostbyname = socket.getfqdn = socket.gethostbyaddr = boom      # type: ignore[assignment]
    try:
        assert is_own_name("rebind.attacker.example", local_ip="192.0.2.10", extra_hosts=()) is False
        assert is_own_name("192.0.2.10", local_ip="192.0.2.10", extra_hosts=()) is True
    finally:
        socket.getaddrinfo, socket.gethostbyname, socket.getfqdn, socket.gethostbyaddr = real


@pytest.mark.parametrize("host,local,mine", [
    ("localhost", None, True), ("127.0.0.1", None, True), ("127.9.9.9", None, True), ("::1", None, True),
    ("192.0.2.10", "192.0.2.10", True), ("192.0.2.11", "192.0.2.10", False), ("192.0.2.10", None, False),
    ("::ffff:192.0.2.10", "192.0.2.10", True), ("2001:db8::1", "2001:db8::1", True), ("2001:db8::2", "2001:db8::1", False),
    ("relay.example", "192.0.2.10", False), ("0.0.0.0", "0.0.0.0", False),
])
def test_what_counts_as_an_own_name(host, local, mine):
    from prometheus.web.identity_challenge import is_own_name

    assert is_own_name(host, local_ip=local, extra_hosts=()) is mine


def test_the_machines_own_host_name_is_its_name(monkeypatch):
    import socket

    from prometheus.web.identity_challenge import is_own_name

    monkeypatch.setattr(socket, "gethostname", lambda: "Will-Mini.Lan")
    for name in ("will-mini.lan", "will-mini", "will-mini.local"):
        assert is_own_name(name, local_ip=None, extra_hosts=()), name
    assert not is_own_name("mini", local_ip=None, extra_hosts=())


# ── the limits, pinned ───────────────────────────────────────────────────────

def test_the_limit_a_party_on_the_path_of_the_real_address_still_passes():
    """Not fixed by a host binding, by design of the check: a relay that is reached AT the daemon's real address (the
    client dialled 192.0.2.10, something on the path answers for it) is indistinguishable here. The pin is what stops
    it; this test exists so the limit stays written down next to the fix."""
    from prometheus.web.identity_challenge import is_own_name

    assert is_own_name("192.0.2.10", local_ip="192.0.2.10", extra_hosts=())


def test_the_port_is_not_part_of_what_is_signed():
    nonce = _nonce()
    body = _ask(_client(base="http://127.0.0.1:9999"), nonce=nonce, headers={"Host": "127.0.0.1:9999"}).json()
    assert body["host"] == "127.0.0.1" and _verifies(body, _v2_message(nonce, body, "127.0.0.1"))


# ── hello says which version ─────────────────────────────────────────────────

def test_hello_advertises_the_highest_challenge_version_and_the_txt_record_carries_it():
    from prometheus.web.hello import HELLO_FIELDS, hello_txt

    hello = _client().get("/api/hello").json()
    assert hello["ic"] == 2 and "ic" in HELLO_FIELDS
    assert hello_txt(hello)["ic"] == "2"


def test_the_limiter_and_the_browser_refusal_still_apply_to_v2():
    r = _ask(_client(), headers={"Origin": "https://example.com"})
    assert r.status_code == 400 and r.json()["error"] == "browser_not_allowed"


# ── found by the independent review ──────────────────────────────────────────

def test_the_name_the_daemon_advertises_over_mdns_is_its_own_name():
    """discovery registers the service at prometheus-<fp8>.local; a phone that dials the SRV target must not get 421."""
    from prometheus.config import instance_key
    from prometheus.web.discovery import _server_name

    instance_key.ensure_instance_key()
    fp = instance_key.instance_fingerprint()
    advertised = _server_name({"fp": fp.encode()}).rstrip(".")
    assert advertised.startswith("prometheus-") and advertised.endswith(".local")
    nonce = _nonce()
    r = _client(base=f"http://{advertised}:8005").post(ROUTE, json={"v": 2, "nonce": nonce, "host": advertised})
    assert r.status_code == 200 and r.json()["host"] == advertised, r.text


def test_the_advertised_name_with_a_different_fingerprint_is_not_its_name():
    from prometheus.config import instance_key

    instance_key.ensure_instance_key()
    r = _client(base="http://prometheus-00000000.local:8005").post(
        ROUTE, json={"v": 2, "nonce": _nonce(), "host": "prometheus-00000000.local"})
    assert r.status_code == 421


@pytest.mark.parametrize("host", ["::1%a", "::1%a\nsigned_at=1", "fe80::1%eth0", "::1%\x1c", "::ffff:127.0.0.1",
                                  "::ffff:7f00:1", "127.0.0.1%1"])
def test_a_zone_id_or_a_mapped_address_is_not_a_canonical_host(host):
    from prometheus.web.identity_challenge import canonical_host

    assert canonical_host(host) is None, host


@pytest.mark.parametrize("version", [2.0, "2", True, 3, 0, -1, "1", [2], {"v": 2}])
def test_a_version_that_is_not_exactly_1_or_2_is_refused_not_answered_as_v1(version):
    r = _client().post(ROUTE, json={"v": version, "nonce": _nonce(), "host": "127.0.0.1"})
    assert r.status_code == 400 and r.json()["error"] == "bad_version" and "sig" not in r.json(), (version, r.text)


def test_a_null_version_is_the_same_as_none():
    r = _client().post(ROUTE, json={"v": None, "nonce": _nonce()})
    assert r.status_code == 200 and r.json()["v"] == 1


def test_setup_mode_says_it_has_no_challenge_to_offer():
    """Setup mode has no /api/identity/challenge route; ic: 0 says so, so a client that trusts ic does not meet a 404."""
    from prometheus.web.hello import build_hello

    assert build_hello(None, setup_mode=True)["ic"] == 0
    assert build_hello({}, setup_mode=False)["ic"] == 2


def test_a_relay_on_the_daemons_own_address_at_another_port_is_a_known_limit():
    """Pinned as a limit, not a feature: the port is not part of the name check, so a forwarder on this address at
    another port gets a signature. It takes a process on this machine (or one that owns this address); the pin is
    what stops it."""
    from prometheus.web.identity_challenge import is_own_name

    assert is_own_name("192.0.2.10", local_ip="192.0.2.10", extra_hosts=())
