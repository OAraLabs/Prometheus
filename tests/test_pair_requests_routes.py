"""``/api/pair/requests`` — a new device asks to join; the owner approves; a scoped token arrives sealed.

Three routes are public (create, poll, cancel/acknowledge), so each does its own checking; three are the
operator's (list, approve, deny) and answer to ``identity.is_operator``. What is pinned, over real HTTP
against the real app, a real ``DeviceStore`` and a real SQLite file:

* **A stranger cannot probe.** Unknown id, wrong secret and another request's secret are one identical 404;
  wrong secrets are rate limited per peer; a poll faster than once a second is refused; ``X-Forwarded-For``
  never picks the bucket; a browser (an ``Origin`` header) is refused with no CORS headers.
* **The token never travels or rests in clear**: approve's response has none, the poll carries it only
  sealed to the requester's key, no log line holds it.
* **Approval grants an ordinary scoped device**: not an owner, not an approver, it owns no conversation and
  sees none of the operator's, and nothing in the request can ask for more (an unknown key on approve is a 400).
* **A scoped device gets 403 on operator routes, not 401**, because 401 tells a client its token is dead.
* The contract's limits, TTL and statuses, and that an uncollected device is revoked.
"""

from __future__ import annotations

import logging
import re
import secrets
import sqlite3
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config import instance_key, pair_seal  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "pair-test-global-" + secrets.token_hex(8)
T0 = 1_760_000_000.0
LOW_ORDER = pair_seal.b64url_encode(bytes(32))


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Requester:
    def __init__(self) -> None:
        self.private = X25519PrivateKey.generate()
        self.public = self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_b64 = pair_seal.b64url_encode(self.public)

    def unseal(self, request_id: str, sealed: dict) -> dict:
        return pair_seal.unseal_token(private_key=self.private, request_id=request_id, sealed=sealed)


class World:
    """A daemon with an owner device and a scoped device, a fake clock, and a recording notifier."""

    def __init__(self, tmp_path, **pairing) -> None:
        self.path = tmp_path / "devices.db"
        self.devices = DeviceStore(self.path)
        config = {"web": {"api_token": GLOBAL}}
        if pairing:
            config["pairing"] = pairing
        self.app = create_app(config, device_store=self.devices)
        self.clock = Clock()
        self.runtime = self.app.state.pairing
        self.runtime.clock = self.clock
        self.events: list[tuple[str, dict]] = []
        self.runtime.notifier.subscribe(lambda kind, payload: self.events.append((kind, payload)) or True)
        self.der = instance_key.ensure_instance_key()      # the daemon makes this at boot
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.scoped = self.devices.mint("a scoped phone", "ios")
        self.client = TestClient(self.app)

    def hdr(self, who: str) -> dict:
        token = {"global": GLOBAL, "owner": self.owner["token"], "scoped": self.scoped["token"]}.get(who, who)
        return {"Authorization": f"Bearer {token}"}

    def as_(self, who: str, method: str, url: str, **kw):
        headers = {**self.hdr(who), **kw.pop("headers", {})}
        return self.client.request(method, url, headers=headers, **kw)

    def request(self, requester=None, *, source="192.0.2.10", **body):
        requester = requester or Requester()
        payload = {"device_name": "Jennifer's MacBook", "platform": "macos", "public_key": requester.public_b64}
        payload.update(body)
        client = TestClient(self.app, client=(source, 50000))
        return client.post("/api/pair/requests", json=payload), requester, client

    def created(self, **kw):
        response, requester, client = self.request(**kw)
        assert response.status_code == 201, response.text
        return response.json(), requester, client

    def poll(self, created, client=None, secret=None, headers=None):
        merged = {"X-Pairing-Secret": secret if secret is not None else created["poll_secret"], **(headers or {})}
        return (client or self.client).get(f"/api/pair/requests/{created['request_id']}", headers=merged)

    def approve(self, created, who="global", **body):
        return self.as_(who, "POST", f"/api/pair/requests/{created['request_id']}/approve", json=body or None)

    def tables(self) -> set[str]:
        with sqlite3.connect(self.path) as conn:
            return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _no_cors(response) -> bool:
    return not [name for name in response.headers if name.lower().startswith("access-control-")]


# ── create ───────────────────────────────────────────────────────────────────

def test_a_stranger_can_ask_to_join_and_gets_what_the_contract_says(world):
    response, requester, _ = world.request()
    assert response.status_code == 201, response.text
    body = response.json()
    assert set(body) == {"request_id", "poll_secret", "match_code", "instance_public_key", "expires_at",
                         "ttl_seconds", "poll_interval_seconds", "notified"}
    assert re.fullmatch(r"[0-9a-f]{32}", body["request_id"])
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", body["poll_secret"])
    assert body["instance_public_key"] == pair_seal.b64url_encode(world.der)
    assert body["match_code"] == pair_seal.match_code(requester.public, world.der, body["request_id"])
    assert (body["expires_at"], body["ttl_seconds"], body["poll_interval_seconds"]) == (int(T0) + 300, 300, 2)
    assert response.headers["cache-control"] == "no-store" and _no_cors(response)


def test_notified_says_whether_anyone_took_the_prompt(tmp_path):
    (tmp_path / "quiet").mkdir()
    (tmp_path / "loud").mkdir()
    quiet = World(tmp_path / "quiet")
    quiet.runtime.notifier.clear()
    assert quiet.created()[0]["notified"] is False
    loud = World(tmp_path / "loud")
    assert loud.created()[0]["notified"] is True


def test_a_browser_is_refused_and_the_refusal_carries_no_cors_headers(world):
    response = world.client.post("/api/pair/requests", headers={"Origin": "https://evil.example"}, json={})
    assert response.status_code == 400 and response.json()["error"] == "browser_not_allowed"
    assert _no_cors(response)
    assert world.tables().isdisjoint({"pair_requests"})


@pytest.mark.parametrize("body, field", [
    ({"device_name": ""}, "device_name"),
    ({"device_name": "   "}, "device_name"),
    ({"device_name": "x" * 65}, "device_name"),
    ({"device_name": 7}, "device_name"),
    ({"device_name": "Jen\nnifer"}, "device_name"),
    ({"device_name": "Mac\x07"}, "device_name"),
    ({"device_name": "evil‮gnp.exe"}, "device_name"),
    ({"public_key": ""}, "public_key"),
    ({"public_key": "AAAA"}, "public_key"),
    ({"public_key": "a+b/" + "A" * 40}, "public_key"),
    ({"public_key": LOW_ORDER}, "public_key"),
    ({"public_key": 12}, "public_key"),
])
def test_a_bad_field_is_a_400_that_names_it_and_creates_nothing(world, body, field):
    response, _, _ = world.request(**body)
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request" and field in response.json()["fields"]
    assert "pair_requests" not in world.tables()


def test_a_name_is_trimmed_not_rewritten(world):
    created, _, _ = world.created(device_name="  Jennifer's MacBook  ")
    assert world.as_("global", "GET", "/api/pair/requests").json()["requests"][0]["device_name"] == "Jennifer's MacBook"
    assert created


def test_a_64_character_name_is_fine(world):
    assert world.request(device_name="é" * 64)[0].status_code == 201


@pytest.mark.parametrize("given, shown", [("macos", "macos"), ("MacOS", "macos"), ("windows", "windows"),
                                          ("beos", "other"), ("", "other"), (7, "other")])
def test_the_platform_is_one_of_six_and_anything_else_is_other(world, given, shown):
    created, _, _ = world.created(platform=given)
    assert world.as_("global", "GET", "/api/pair/requests").json()["requests"][0]["platform"] == shown
    assert created


def test_unknown_keys_are_ignored_for_forward_compatibility(world):
    assert world.request(owner=True, scope="all", future_field=[1])[0].status_code == 201


@pytest.mark.parametrize("kwargs", [
    {"content": "not json", "headers": {"Content-Type": "application/json"}},
    {"content": "[1, 2]", "headers": {"Content-Type": "application/json"}},
    {"content": "{}", "headers": {"Content-Type": "text/plain"}},
], ids=["not json", "not an object", "wrong content type"])
def test_a_body_that_is_not_a_json_object_is_a_400(world, kwargs):
    response = world.client.post("/api/pair/requests", **kwargs)
    assert response.status_code == 400 and response.json()["error"] == "invalid_request"


def test_a_body_over_4_kib_is_a_413(world):
    big = '{"device_name": "' + "x" * 5000 + '"}'
    response = world.client.post("/api/pair/requests", content=big, headers={"Content-Type": "application/json"})
    assert response.status_code == 413 and response.json()["error"] == "too_large"


def test_pairing_is_unavailable_when_the_owner_turned_requests_off(tmp_path):
    response, _, _ = World(tmp_path, requests_enabled=False).request()
    assert response.status_code == 403 and response.json()["error"] == "pairing_unavailable"


def test_pairing_is_unavailable_on_a_daemon_with_no_token(tmp_path, monkeypatch):
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    app = create_app({"web": {}}, device_store=DeviceStore(tmp_path / "devices.db"))
    response = TestClient(app).post("/api/pair/requests", json={
        "device_name": "x", "platform": "ios", "public_key": Requester().public_b64})
    assert response.status_code == 403 and response.json()["error"] == "pairing_unavailable"


def test_without_an_instance_key_it_says_so_and_creates_nothing(world):
    (Path(instance_key.node_dir_path()) / instance_key.INSTANCE_KEY_FILENAME).unlink()
    response, _, _ = world.request()
    assert response.status_code == 503 and response.json()["error"] == "identity_unavailable"
    assert "pair_requests" not in world.tables()


def test_a_request_touches_no_conversation(world):
    created, _, _ = world.created()
    world.approve(created)
    assert world.as_("global", "GET", "/api/sessions").json() == []


# ── limits over HTTP ─────────────────────────────────────────────────────────

def test_a_second_pending_request_from_one_source_is_a_429_with_a_hint(world):
    world.created(source="192.0.2.10")
    response, _, _ = world.request(source="192.0.2.10")
    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "rate_limited" and body["reason"] == "per_source_pending"
    assert int(response.headers["retry-after"]) == body["retry_after_seconds"] >= 1


def test_a_forged_forwarded_for_does_not_pick_the_bucket(world):
    world.created(source="192.0.2.10")
    response = TestClient(world.app, client=("192.0.2.10", 1)).post(
        "/api/pair/requests", headers={"X-Forwarded-For": "203.0.113.9"},
        json={"device_name": "x", "platform": "ios", "public_key": Requester().public_b64})
    assert response.status_code == 429 and response.json()["reason"] == "per_source_pending"


def test_the_fourth_pending_request_overall_is_refused(world):
    for n in range(3):
        world.created(source=f"192.0.2.{n + 1}")
    response, _, _ = world.request(source="192.0.2.99")
    assert response.status_code == 429 and response.json()["reason"] == "pending_full"


def test_the_hourly_limit_is_enforced(tmp_path):
    w = World(tmp_path, max_pending_per_source=50, max_pending=50, max_requests_per_source_per_hour=3)
    for _ in range(3):
        created, _, client = w.created(source="192.0.2.10")
        assert client.delete(f"/api/pair/requests/{created['request_id']}",
                             headers={"X-Pairing-Secret": created["poll_secret"]}).status_code == 204
    response, _, _ = w.request(source="192.0.2.10")
    assert response.status_code == 429 and response.json()["reason"] == "hourly"


# ── poll ─────────────────────────────────────────────────────────────────────

def test_a_pending_request_polls_as_pending(world):
    created, _, _ = world.created()
    response = world.poll(created)
    assert response.status_code == 200 and response.json() == {"status": "pending", "expires_at": int(T0) + 300}
    assert response.headers["cache-control"] == "no-store" and _no_cors(response)


def test_unknown_id_wrong_secret_and_another_requests_secret_are_one_identical_404(world):
    one, _, _ = world.created(source="192.0.2.1")
    two, _, _ = world.created(source="192.0.2.2")
    answers = [
        world.poll({"request_id": "f" * 32}, secret=one["poll_secret"]),
        world.poll(one, secret="x" * 43),
        world.poll(one, secret=two["poll_secret"]),
        world.client.get(f"/api/pair/requests/{one['request_id']}"),   # no header at all
    ]
    assert {(a.status_code, a.text) for a in answers} == {(404, '{"error":"unknown_request"}')}


def test_a_browser_may_not_poll(world):
    created, _, _ = world.created()
    response = world.client.get(f"/api/pair/requests/{created['request_id']}",
                                headers={"X-Pairing-Secret": created["poll_secret"], "Origin": "https://evil.example"})
    assert response.status_code == 400 and _no_cors(response)


def test_polling_faster_than_once_a_second_is_refused(world):
    created, _, _ = world.created()
    assert world.poll(created).status_code == 200
    fast = world.poll(created)
    assert fast.status_code == 429 and fast.json()["reason"] == "poll_too_fast"
    world.clock.now += 1.0
    assert world.poll(created).status_code == 200


def test_five_wrong_secrets_a_minute_lock_that_source_out_and_only_that_source(world):
    created, _, client = world.created(source="192.0.2.10")
    for _ in range(5):
        assert world.poll(created, client, secret="x" * 43).status_code == 404
        world.clock.now += 1.1
    blocked = world.poll(created, client)
    assert blocked.status_code == 429 and blocked.json()["reason"] == "bad_secret"
    other = TestClient(world.app, client=("192.0.2.77", 1))
    assert world.poll(created, other).status_code == 200
    world.clock.now += 61
    assert world.poll(created, client).status_code == 200


def test_a_request_expires_at_its_ttl(world):
    created, _, _ = world.created()
    world.clock.now += 301
    assert world.poll(created).json() == {"status": "expired"}
    assert world.as_("global", "GET", "/api/pair/requests").json() == {"requests": []}


# ── cancel and acknowledge ───────────────────────────────────────────────────

def test_the_requester_can_cancel_while_pending(world):
    created, _, client = world.created()
    response = client.delete(f"/api/pair/requests/{created['request_id']}",
                             headers={"X-Pairing-Secret": created["poll_secret"]})
    assert response.status_code == 204 and response.content == b""
    world.clock.now += 1.1
    assert world.poll(created).json() == {"status": "canceled"}
    assert ("resolved", {"request_id": created["request_id"], "resolution": "canceled", "by": "requester",
                         "resolved_at": int(T0)}) in world.events


def test_cancel_needs_the_secret_and_a_known_request(world):
    created, _, _ = world.created()
    assert world.client.delete(f"/api/pair/requests/{created['request_id']}",
                               headers={"X-Pairing-Secret": "x" * 43}).status_code == 404
    assert world.client.delete(f"/api/pair/requests/{'f' * 32}",
                               headers={"X-Pairing-Secret": created["poll_secret"]}).status_code == 404
    assert world.poll(created).json()["status"] == "pending"


# ── approve: what the operator sees and does ─────────────────────────────────

def test_the_operator_sees_the_pending_request_with_its_code_and_source_and_no_secret(world):
    created, requester, _ = world.created(source="192.0.2.42")
    listing = world.as_("global", "GET", "/api/pair/requests")
    assert listing.status_code == 200
    (item,) = listing.json()["requests"]
    assert item == {"request_id": created["request_id"], "device_name": "Jennifer's MacBook", "platform": "macos",
                    "source_ip": "192.0.2.42", "match_code": created["match_code"],
                    "created_at": int(T0), "expires_at": int(T0) + 300, "ttl_seconds": 300}
    assert created["poll_secret"] not in listing.text


def test_the_list_is_newest_first_and_pending_only(world):
    first, _, _ = world.created(source="192.0.2.1")
    world.clock.now += 5
    second, _, _ = world.created(source="192.0.2.2")
    world.approve(first)
    assert [r["request_id"] for r in world.as_("global", "GET", "/api/pair/requests").json()["requests"]] == [
        second["request_id"]]


def test_approving_answers_with_the_device_and_never_the_token(world):
    created, requester, _ = world.created()
    response = world.approve(created)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"request_id", "status", "device_id", "name", "platform"}
    assert body["status"] == "approved" and body["name"] == "Jennifer's MacBook"
    sealed = world.poll(created).json()["sealed"]
    token = requester.unseal(created["request_id"], sealed)["token"]
    assert token not in response.text


def test_the_requester_collects_the_token_sealed_and_it_authenticates(world):
    created, requester, _ = world.created()
    world.approve(created)
    polled = world.poll(created)
    body = polled.json()
    assert body["status"] == "approved" and set(body) == {
        "status", "device_id", "sealed", "endpoints", "tls", "approved_at"}
    assert set(body["sealed"]) == {"alg", "ephemeral_public_key", "nonce", "ciphertext"}
    opened = requester.unseal(created["request_id"], body["sealed"])
    assert opened["device_id"] == body["device_id"] and opened["name"] == "Jennifer's MacBook"
    assert opened["token"] not in polled.text, "the poll carries the token only sealed"
    assert world.client.get("/api/sessions", headers={"Authorization": f"Bearer {opened['token']}"}).status_code == 200
    assert polled.headers["cache-control"] == "no-store"


def test_a_lost_response_is_retried_with_the_same_blob_until_acknowledged(world):
    created, requester, client = world.created()
    world.approve(created)
    first = world.poll(created).json()
    world.clock.now += 1.1
    assert world.poll(created).json()["sealed"] == first["sealed"]
    assert client.delete(f"/api/pair/requests/{created['request_id']}",
                         headers={"X-Pairing-Secret": created["poll_secret"]}).status_code == 204
    world.clock.now += 1.1
    assert world.poll(created).json() == {"status": "delivered"}
    token = requester.unseal(created["request_id"], first["sealed"])["token"]
    assert world.client.get("/api/sessions", headers={"Authorization": f"Bearer {token}"}).status_code == 200, \
        "acknowledging wipes the blob, not the device"


def test_the_endpoints_are_built_from_the_host_the_requester_used(world):
    created, _, _ = world.created()
    world.approve(created)
    body = world.poll(created, headers={"Host": "192.0.2.20:8005"}).json()
    assert body["endpoints"] == {"rest": "http://192.0.2.20:8005", "ws": "ws://192.0.2.20:8010"}
    assert body["tls"] is None


@pytest.mark.parametrize("host", ["evil.example/../x", "a b", "host\r\nX: y", "[::1", "a:99999999"])
def test_a_host_that_is_not_a_host_yields_no_endpoints(world, host):
    created, _, _ = world.created()
    world.approve(created)
    try:
        response = world.poll(created, headers={"Host": host})
    except Exception:       # the client library itself may refuse to send it
        return
    if response.status_code == 200:
        assert response.json()["endpoints"] == {"rest": None, "ws": None}


def test_an_operator_may_rename_at_approval_and_the_prompt_code_may_be_retyped(world):
    created, requester, _ = world.created()
    response = world.approve(created, name="Jennifer's MacBook (kitchen)", match_code=created["match_code"])
    assert response.status_code == 200 and response.json()["name"] == "Jennifer's MacBook (kitchen)"
    assert requester.unseal(created["request_id"], world.poll(created).json()["sealed"])["name"] == \
        "Jennifer's MacBook (kitchen)"


def test_a_retyped_code_that_differs_is_a_422_and_decides_nothing(world):
    created, _, _ = world.created()
    wrong = "0000" if created["match_code"] != "0000" else "1111"
    response = world.approve(created, match_code=wrong)
    assert response.status_code == 422 and response.json()["error"] == "code_mismatch"
    assert world.poll(created).json()["status"] == "pending"
    assert len(world.devices.list_devices()) == 2, "only the owner and scoped devices the world started with"


@pytest.mark.parametrize("extra", [{"owner": True}, {"scope": "all"}, {"tier": "owner"}, {"role": "approver"},
                                   {"platform": "macos"}, {"unexpected": 1}])
def test_approve_accepts_exactly_name_and_match_code_so_nobody_thinks_they_granted_more(world, extra):
    created, _, _ = world.created()
    response = world.approve(created, **extra)
    assert response.status_code == 400 and response.json()["error"] == "invalid_request"
    assert world.poll(created).json()["status"] == "pending"


@pytest.mark.parametrize("body", [{"name": ""}, {"name": "x" * 65}, {"name": "a\nb"}, {"name": 5},
                                  {"match_code": "12"}, {"match_code": "abcd"}, {"match_code": 1234}])
def test_a_bad_name_or_code_on_approve_is_a_400(world, body):
    created, _, _ = world.created()
    assert world.approve(created, **body).status_code == 400


def test_a_second_decision_is_a_409_that_names_the_winner(world):
    created, _, _ = world.created()
    assert world.approve(created).status_code == 200
    again = world.approve(created)
    assert again.status_code == 409 and again.json() == {"error": "not_pending", "status": "approved"}
    deny = world.as_("global", "POST", f"/api/pair/requests/{created['request_id']}/deny")
    assert deny.status_code == 409 and deny.json()["status"] == "approved"


def test_deciding_after_the_ttl_is_a_410(world):
    created, _, _ = world.created()
    world.clock.now += 301
    assert world.approve(created).status_code == 410 and world.approve(created).json()["error"] == "expired"
    assert world.as_("global", "POST", f"/api/pair/requests/{created['request_id']}/deny").status_code == 410


def test_an_unknown_request_is_a_404_for_the_operator(world):
    assert world.as_("global", "POST", f"/api/pair/requests/{'f' * 32}/approve").status_code == 404
    assert world.as_("global", "POST", f"/api/pair/requests/{'f' * 32}/deny").status_code == 404


def test_denying_tells_the_requester_and_mints_nothing(world):
    created, _, _ = world.created()
    before = len(world.devices.list_devices())
    response = world.as_("global", "POST", f"/api/pair/requests/{created['request_id']}/deny")
    assert response.status_code == 200 and response.json() == {"request_id": created["request_id"], "status": "denied"}
    assert world.poll(created).json() == {"status": "denied"}
    assert len(world.devices.list_devices()) == before


# ── who may decide ───────────────────────────────────────────────────────────

def test_no_token_is_a_401_on_every_operator_route(world):
    created, _, _ = world.created()
    rid = created["request_id"]
    for method, url in (("GET", "/api/pair/requests"), ("POST", f"/api/pair/requests/{rid}/approve"),
                        ("POST", f"/api/pair/requests/{rid}/deny")):
        assert world.client.request(method, url).status_code == 401, url


def test_a_scoped_device_is_a_403_not_a_401_on_every_operator_route(world):
    created, _, _ = world.created()
    rid = created["request_id"]
    for method, url in (("GET", "/api/pair/requests"), ("POST", f"/api/pair/requests/{rid}/approve"),
                        ("POST", f"/api/pair/requests/{rid}/deny")):
        response = world.as_("scoped", method, url)
        assert (response.status_code, response.json()["error"]) == (403, "operator_only"), url
    assert world.poll(created).json()["status"] == "pending", "a refused decision decides nothing"


def test_an_owner_device_and_the_global_token_may_decide(world):
    one, _, _ = world.created(source="192.0.2.1")
    two, _, _ = world.created(source="192.0.2.2")
    assert world.approve(one, who="owner").status_code == 200
    assert world.approve(two, who="global").status_code == 200


def test_an_approved_device_is_scoped_and_cannot_do_what_a_stolen_phone_must_not(world):
    other = world.devices.mint("someone else", "macos")
    created, requester, _ = world.created()
    world.approve(created)
    token = requester.unseal(created["request_id"], world.poll(created).json()["sealed"])["token"]
    def mine(method, url, **kw):
        return world.client.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kw)

    # Not an operator: it cannot approve the next stranger, nor list who is waiting.
    nxt, _, _ = world.created(source="192.0.2.50")
    assert mine("GET", "/api/pair/requests").status_code == 403
    assert mine("POST", f"/api/pair/requests/{nxt['request_id']}/approve").status_code == 403
    # Not root: it cannot enrol another device.
    assert mine("POST", "/api/devices", json={"name": "attacker"}).status_code == 401
    # It owns nothing: it sees none of the operator's conversations, and cannot revoke another device.
    assert mine("GET", "/api/sessions").json() == []
    assert mine("DELETE", f"/api/devices/{other['id']}").status_code == 403
    # It is on the list as an ordinary device.
    row = next(d for d in world.as_("global", "GET", "/api/devices").json()
               if d["name"] == "Jennifer's MacBook")
    assert row["owner"] is False and row["owner_source"] is None
    assert world.devices.is_owner(row["id"]) is False


def test_the_operator_can_revoke_an_approved_device_and_its_token_dies(world):
    created, requester, _ = world.created()
    approved = world.approve(created).json()
    token = requester.unseal(created["request_id"], world.poll(created).json()["sealed"])["token"]
    assert world.as_("global", "DELETE", f"/api/devices/{approved['device_id']}").status_code == 200
    assert world.client.get("/api/sessions", headers={"Authorization": f"Bearer {token}"}).status_code == 401


# ── an approval nobody collects ──────────────────────────────────────────────

def test_an_uncollected_device_is_revoked_when_the_window_closes(world):
    created, requester, _ = world.created()
    approved = world.approve(created).json()
    token = requester.unseal(created["request_id"], world.poll(created).json()["sealed"])["token"]
    heard: list[list[str]] = []
    world.devices.add_revoke_listener(heard.append)
    world.clock.now += 301
    assert world.poll(created).json() == {"status": "expired"}, "reported as expired, never as approved"
    assert heard == [[approved["device_id"]]]
    assert world.client.get("/api/sessions", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_the_background_sweep_revokes_without_anyone_asking(world):
    created, requester, _ = world.created()
    approved = world.approve(created).json()
    world.runtime.sweep_interval = 0.05
    world.clock.now += 301
    reader = DeviceStore(world.path)    # a second connection: the sweeper owns the first while it runs

    def revoked_at():
        return next(d for d in reader.list_devices() if d.id == approved["device_id"]).revoked_at

    with TestClient(world.app):         # entering it runs the app's startup, which starts the sweeper
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not revoked_at():
            time.sleep(0.02)
    assert revoked_at() is not None


# ── what the operator's channels are told ────────────────────────────────────

def test_creation_is_announced_with_what_the_operator_needs_and_no_secret(world):
    created, _, _ = world.created(source="192.0.2.42")
    kind, payload = world.events[0]
    assert kind == "pending"
    assert payload == {"request_id": created["request_id"], "device_name": "Jennifer's MacBook",
                       "platform": "macos", "source_ip": "192.0.2.42", "match_code": created["match_code"],
                       "created_at": int(T0), "expires_at": int(T0) + 300, "ttl_seconds": 300}


def test_each_decision_is_announced_once_with_who_decided(world):
    one, _, _ = world.created(source="192.0.2.1")
    two, _, _ = world.created(source="192.0.2.2")
    world.approve(one)
    world.as_("global", "POST", f"/api/pair/requests/{two['request_id']}/deny", headers={"X-Pairing-Via": "cli"})
    resolved = [p for k, p in world.events if k == "resolved"]
    assert {(p["request_id"], p["resolution"], p["by"]) for p in resolved} == {
        (one["request_id"], "approved", "beacon"), (two["request_id"], "denied", "cli")}


def test_a_via_header_is_display_only_and_only_cli_is_believed(world):
    created, _, _ = world.created()
    world.as_("global", "POST", f"/api/pair/requests/{created['request_id']}/deny",
              headers={"X-Pairing-Via": "telegram"})
    assert [p["by"] for k, p in world.events if k == "resolved"] == ["beacon"]


def test_expiry_is_announced_by_the_sweep(world):
    created, _, _ = world.created()
    world.clock.now += 301
    world.poll(created)
    assert [(p["request_id"], p["resolution"], p["by"]) for k, p in world.events if k == "resolved"] == [
        (created["request_id"], "expired", "system")]


def test_a_listener_that_raises_cannot_break_a_request(world):
    world.runtime.notifier.subscribe(lambda kind, payload: 1 / 0)
    assert world.request()[0].status_code == 201


# ── the audit trail ──────────────────────────────────────────────────────────

def test_every_step_leaves_a_pairing_audit_line_and_no_line_holds_a_secret(world, caplog):
    with caplog.at_level(logging.INFO):
        created, requester, client = world.created(source="192.0.2.42")
        approved = world.approve(created).json()
        sealed = world.poll(created).json()["sealed"]
        token = requester.unseal(created["request_id"], sealed)["token"]
        client.delete(f"/api/pair/requests/{created['request_id']}", headers={"X-Pairing-Secret": created["poll_secret"]})
        other, _, _ = world.created(source="192.0.2.43")
        world.as_("global", "POST", f"/api/pair/requests/{other['request_id']}/deny")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("pairing:")]
    kinds = [re.match(r"pairing: (\w+)", ln).group(1) for ln in lines]
    assert {"created", "approved", "delivered", "denied"} <= set(kinds), lines
    assert any("192.0.2.42" in ln and "Jennifer's MacBook" in ln and created["request_id"] in ln for ln in lines)
    everything = caplog.text
    for secret in (created["poll_secret"], token, sealed["ciphertext"], other["poll_secret"]):
        assert secret not in everything
    assert approved["device_id"] in everything


# ── the gate ─────────────────────────────────────────────────────────────────

def test_only_the_three_requester_routes_are_public(world):
    from prometheus.web.public_routes import PUBLIC_ROUTES, is_public_route

    assert {("POST", "/api/pair/requests"), ("GET", "/api/pair/requests/{request_id}"),
            ("DELETE", "/api/pair/requests/{request_id}")} <= PUBLIC_ROUTES
    rid = "a" * 32
    for method, path in (("GET", "/api/pair/requests"), ("POST", f"/api/pair/requests/{rid}/approve"),
                         ("POST", f"/api/pair/requests/{rid}/deny"), ("GET", f"/api/pair/requests/{rid}/approve"),
                         ("PUT", f"/api/pair/requests/{rid}"), ("POST", f"/api/pair/requests/{rid}"),
                         ("GET", "/api/pair/requests/"), ("GET", "/api/pair/codes")):
        assert is_public_route(method, path) is False, (method, path)


def test_the_table_does_not_exist_until_someone_asks_to_join(world):
    world.client.get("/api/hello")
    world.poll({"request_id": "f" * 32, "poll_secret": "x"})
    world.as_("global", "GET", "/api/pair/requests")
    assert "pair_requests" not in world.tables()
    world.created()
    assert "pair_requests" in world.tables()
