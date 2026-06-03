"""Audit #6: the web server must expose an unauthenticated /health -> 200.

Previously /health 404'd, flooding the journal with one failed probe/minute from
the monitoring host.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


def test_health_returns_200_ok():
    client = TestClient(create_app({}))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "prometheus"
    assert "uptime_seconds" in body


def test_health_stays_open_when_api_token_set():
    # With a token, /api/* is locked but /health must remain reachable.
    client = TestClient(create_app({"web": {"api_token": "secret"}}))
    assert client.get("/health").status_code == 200
    # sanity: an /api/ route IS blocked without the bearer token
    assert client.get("/api/status").status_code == 401
