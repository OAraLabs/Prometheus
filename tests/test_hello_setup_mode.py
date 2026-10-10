"""``GET /api/hello`` in setup mode.

A fresh install answers on the same port the configured daemon will, from a different application
(``web/setup_server.py``), so a client must not need to know which one it reached: the same function builds
the same six fields. What differs, and is pinned: ``pair`` is ``code`` (the six-digit code and the same-Mac
secret are what work), ``fp`` is empty (setup mode creates no ``~/.prometheus`` state, and a key is state),
and the ``Origin`` refusal lives in the route, because this app has no bearer gate to do it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.setup_server import PairingState, create_setup_app  # noqa: E402

SIX = {"v", "name", "agent", "fp", "pair", "tls"}


@pytest.fixture(autouse=True)
def _token_env(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)


def _client(**kwargs):
    app = create_setup_app(PairingState(code="042999"), api_port=8123, ws_port=8124)
    return TestClient(app, **kwargs)


def test_it_answers_in_setup_mode_with_the_same_six_fields():
    reply = _client().get("/api/hello")
    assert reply.status_code == 200, reply.text
    assert set(reply.json()) == SIX


def test_setup_mode_says_the_code_works_and_has_no_fingerprint():
    body = _client().get("/api/hello").json()
    assert body["pair"] == "code"
    assert body["fp"] == ""
    assert body["tls"] is False
    assert body["agent"] == "Prometheus"


def test_both_servers_build_the_answer_with_the_same_function():
    configured = TestClient(create_app({"web": {"api_token": "t"}})).get("/api/hello").json()
    setup = _client().get("/api/hello").json()
    assert set(configured) == set(setup)
    assert configured["v"] == setup["v"] and configured["name"] == setup["name"]


def test_setup_mode_creates_no_state_to_answer():
    _client().get("/api/hello")
    assert not (Path(os.environ["PROMETHEUS_CONFIG_DIR"]) / "node").exists()


def test_a_browser_is_refused_without_cors_headers():
    reply = _client().get("/api/hello", headers={"Origin": "https://evil.example"})
    assert reply.status_code == 400
    assert reply.json()["error"] == "browser_not_allowed"
    assert not [name for name in reply.headers if name.lower().startswith("access-control-")]


def test_it_is_never_cached_and_carries_no_cors_headers():
    reply = _client().get("/api/hello")
    assert reply.headers["cache-control"] == "no-store"
    assert not [name for name in reply.headers if name.lower().startswith("access-control-")]


def test_it_is_rate_limited_per_peer(monkeypatch):
    from prometheus.web import hello

    monkeypatch.setattr(hello, "HELLO_PER_MINUTE", 2)
    client = _client()
    assert [client.get("/api/hello").status_code for _ in range(3)] == [200, 200, 429]


def test_a_request_to_join_is_refused_in_setup_mode_with_the_contracts_code():
    """There is no operator yet who could approve the first device: that is setup-mode pairing."""
    reply = _client().post("/api/pair/requests", json={"device_name": "x", "platform": "ios", "public_key": "A" * 43})
    assert reply.status_code == 403 and reply.json()["error"] == "pairing_unavailable"
    assert not (Path(os.environ["PROMETHEUS_CONFIG_DIR"]) / "node").exists()


def test_the_rest_of_setup_mode_is_still_closed():
    """Adding hello must not open a neighbour: anything unlisted is still the honest 403."""
    client = _client()
    assert client.get("/api/hello/extra").status_code == 403
    assert client.post("/api/hello").status_code in (403, 405)
    assert client.get("/api/status").status_code == 403
