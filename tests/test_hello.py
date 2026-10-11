"""``GET /api/hello`` — the credential-free answer to "is there a Prometheus here, and what is it called".

A phone on the home network has no token and no address, so this is the first thing it can ask. It is
therefore the most exposed route the daemon has, and what is pinned is what keeps it from telling a
stranger anything worth having:

* the answer is EXACTLY six fields, an allowlist test so a seventh fails here and has to be argued for;
* the mDNS TXT record is derived from the same dictionary, so the two cannot drift;
* it carries no CORS headers and a browser (an ``Origin`` header) is refused without any, because this is
  for apps, never for a web page;
* it is rate limited per TCP peer, never per ``X-Forwarded-For`` (the caller controls that);
* it never writes to disk (the instance key is made at boot), and a fingerprint is only ever reported for a
  key that exists;
* ``pair`` says what a client can actually do next, truthfully: ``token`` until a request route is served,
  ``none`` when the daemon has no token at all.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.version import package_version  # noqa: E402
from prometheus.web.loopback import guard_if_loopback  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

SIX = {"v", "name", "agent", "fp", "pair", "tls"}
TOKEN = "hello-test-token-0123456789abcdef"


@pytest.fixture(autouse=True)
def _token_env(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)


def _app(**overrides):
    config = {"web": {"api_token": TOKEN}}
    config.update(overrides)
    return create_app(config)


def _client(app=None, **kwargs):
    return TestClient(app or _app(), **kwargs)


def _node_dir() -> Path:
    return Path(os.environ["PROMETHEUS_CONFIG_DIR"]) / "node"


# ── the answer ───────────────────────────────────────────────────────────────

def test_it_answers_without_a_token_on_a_daemon_that_has_one():
    assert _client().get("/api/status").status_code == 401        # the gate is on
    reply = _client().get("/api/hello")
    assert reply.status_code == 200, reply.text


def test_it_is_exactly_six_fields():
    assert set(_client().get("/api/hello").json()) == SIX


def test_the_fields_say_what_they_should():
    body = _client(_app(system={"name": "Jarvis"}, pairing={"display_name": "Will's Mac mini"})).get(
        "/api/hello").json()
    assert body["v"] == package_version()
    assert body["name"] == "Will's Mac mini"
    assert body["agent"] == "Jarvis"
    assert body["tls"] is False
    assert body["fp"] == ""                       # no instance key has been made yet
    assert body["pair"] == "approve"              # the request routes are served


def test_the_agent_is_called_prometheus_unless_configured():
    assert _client().get("/api/hello").json()["agent"] == "Prometheus"


def test_requests_turned_off_means_only_the_token_pairs_a_device():
    assert _client(_app(pairing={"requests_enabled": False})).get("/api/hello").json()["pair"] == "token"


def test_a_daemon_with_no_token_says_there_is_nothing_to_pair_into():
    # create_app reads the token from the config, then the environment: neither is set here.
    body = _client(create_app({"web": {}})).get("/api/hello").json()
    assert body["pair"] == "none"


def test_the_fingerprint_is_the_instance_keys_and_only_when_it_exists():
    from cryptography.hazmat.primitives import serialization

    from prometheus.config import instance_key

    client = _client()
    assert client.get("/api/hello").json()["fp"] == ""
    der = instance_key.ensure_instance_key()
    fp = client.get("/api/hello").json()["fp"]
    assert fp == hashlib.sha256(der).hexdigest()[:16]
    assert len(fp) == 16
    assert serialization.load_der_public_key(der) is not None


def test_a_request_writes_nothing_to_disk():
    _client().get("/api/hello")
    assert not _node_dir().exists(), "hello is unauthenticated: a GET must not create the instance key"


def test_it_is_never_cached():
    assert _client().get("/api/hello").headers["cache-control"] == "no-store"


# ── the TXT record is the same dictionary ────────────────────────────────────

def test_the_txt_record_has_the_same_keys_and_the_same_values():
    from prometheus.web.hello import HELLO_FIELDS, hello_txt

    body = _client().get("/api/hello").json()
    txt = hello_txt(body)
    assert set(HELLO_FIELDS) == SIX == set(txt)
    assert all(isinstance(value, str) for value in txt.values())
    assert txt["tls"] == "0"
    assert {k: v for k, v in txt.items() if k != "tls"} == {k: v for k, v in body.items() if k != "tls"}
    assert hello_txt({**body, "tls": True})["tls"] == "1"


def test_a_field_added_to_hello_but_not_to_the_allowlist_fails_the_txt_conversion():
    from prometheus.web.hello import hello_txt

    body = _client().get("/api/hello").json()
    with pytest.raises(ValueError, match="unexpected"):
        hello_txt({**body, "uptime": 12})


# ── no credentials, no browsers ──────────────────────────────────────────────

def test_it_carries_no_cors_headers():
    headers = _client().get("/api/hello").headers
    assert not [name for name in headers if name.lower().startswith("access-control-")]


def test_a_browser_is_refused_and_the_refusal_carries_no_cors_headers():
    reply = _client().get("/api/hello", headers={"Origin": "https://evil.example"})
    assert reply.status_code == 400
    assert reply.json()["error"] == "browser_not_allowed"
    assert not [name for name in reply.headers if name.lower().startswith("access-control-")]


def test_a_preflight_for_it_is_not_answered_with_cors_headers():
    reply = _client().options("/api/hello", headers={
        "Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert not [name for name in reply.headers if name.lower().startswith("access-control-")]


def test_only_get_is_served():
    client = _client()
    for method in ("post", "put", "delete", "patch"):
        assert getattr(client, method)("/api/hello").status_code in (401, 404, 405), method


# ── the limit ────────────────────────────────────────────────────────────────

def test_the_limit_is_60_a_minute_per_peer():
    from prometheus.web import hello

    assert hello.HELLO_PER_MINUTE == 60


def test_a_peer_over_the_limit_gets_429_with_a_retry_hint(monkeypatch):
    from prometheus.web import hello

    monkeypatch.setattr(hello, "HELLO_PER_MINUTE", 3)
    client = _client()
    assert [client.get("/api/hello").status_code for _ in range(3)] == [200, 200, 200]
    refused = client.get("/api/hello")
    assert refused.status_code == 429
    body = refused.json()
    assert body["error"] == "rate_limited" and body["reason"] == "hello"
    assert int(refused.headers["retry-after"]) == body["retry_after_seconds"] >= 1


def test_the_limit_is_per_peer_not_per_forwarded_for(monkeypatch):
    from prometheus.web import hello

    monkeypatch.setattr(hello, "HELLO_PER_MINUTE", 2)
    app = _app()
    one = TestClient(app, client=("192.0.2.10", 40000))
    two = TestClient(app, client=("192.0.2.11", 40000))
    spoofed = [one.get("/api/hello", headers={"X-Forwarded-For": f"203.0.113.{n}"}).status_code
               for n in range(4)]
    assert spoofed == [200, 200, 429, 429], "a caller must not be able to pick its own bucket"
    assert two.get("/api/hello").status_code == 200


# ── behind the loopback Host guard (#693) ────────────────────────────────────

def test_it_answers_a_loopback_bind_that_is_addressed_as_localhost():
    client = TestClient(guard_if_loopback(_app(), "127.0.0.1"), base_url="http://127.0.0.1:8005")
    assert client.get("/api/hello").status_code == 200


def test_it_is_not_reachable_through_a_rebound_name_on_a_loopback_bind():
    client = TestClient(guard_if_loopback(_app(), "127.0.0.1"), base_url="http://evil.example:8005")
    assert client.get("/api/hello").status_code == 403
