"""Tests for Prometheus out-of-process Extension hooks (docs/design/extension-hooks.md)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prometheus.extensions.registry import (
    ExtensionConfigError,
    load_extension_registry,
)
from prometheus.web.extension_routes import register_extension_routes
from prometheus.web.route_access import is_scoped_allowed


def test_registry_enforces_loopback_and_credential_hygiene() -> None:
    with pytest.raises(ExtensionConfigError, match="allow_remote"):
        load_extension_registry({"extensions": {"gym": {"url": "http://100.64.0.5:18097"}}})

    with pytest.raises(ExtensionConfigError, match="credentials"):
        load_extension_registry({"extensions": {"gym": {"url": "http://user:pass@127.0.0.1:18097"}}})

    reg = load_extension_registry({
        "extensions": {
            "gym": {
                "url": "http://127.0.0.1:18097",
                "token_env": "TEST_GYM_READ_TOK",
                "control_url": "http://127.0.0.1:18098",
                "control_token_env": "TEST_GYM_CTRL_TOK",
            }
        }
    })
    ent = reg.get("gym")
    assert ent is not None
    summary = ent.public_summary(healthy=True)
    blob = json.dumps(summary)
    assert "18097" not in blob
    assert "18098" not in blob
    assert "TEST_GYM_READ_TOK" not in blob
    assert "TEST_GYM_CTRL_TOK" not in blob
    assert summary["has_control"] is True


def test_all_extension_routes_are_operator_only() -> None:
    for method, path in (
        ("GET", "/api/extensions"),
        ("GET", "/api/extensions/traces/export"),
        ("GET", "/api/extensions/gym/status"),
        ("POST", "/api/extensions/gym/control/intent"),
        ("POST", "/api/extensions/gym/control/confirm"),
    ):
        assert is_scoped_allowed(method, path) is False


def test_extension_routes_manifest_status_control_and_trace_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_READ_TOK", "read-secret-123")
    monkeypatch.setenv("TEST_CTRL_TOK", "ctrl-secret-456")

    seen_auth: list[tuple[str, str | None]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_auth.append((str(req.url), req.headers.get("authorization")))
        if str(req.url) == "http://127.0.0.1:18097/health":
            return httpx.Response(200, json={"status": "ok"})
        if str(req.url) == "http://127.0.0.1:18097/integration/manifest":
            return httpx.Response(200, json={
                "schema": "prometheus-extension-manifest/1",
                "name": "gym",
                "version": "0.1.0",
                "kind": "gym",
                "capabilities": ["status", "control"],
                "secret_field_should_be_dropped": "do-not-leak",
            })
        if str(req.url) == "http://127.0.0.1:18097/v1/status":
            return httpx.Response(200, json={"round": "r6", "phase": "training", "step": 150})
        if str(req.url) == "http://127.0.0.1:18098/v1/control/intent":
            return httpx.Response(200, json={"intent_id": "int-1", "plan_sha256": "abc123", "rollback_available": True})
        if str(req.url) == "http://127.0.0.1:18098/v1/control/confirm":
            return httpx.Response(200, json={"status": "started", "unit": "promote-1"})
        return httpx.Response(404, json={"error": "not_found"})

    mock_transport = httpx.MockTransport(handler)
    mock_client = httpx.AsyncClient(transport=mock_transport)

    # Create a tiny local telemetry SQLite DB
    db_path = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE responses (id INTEGER PRIMARY KEY, ts TEXT, session_id TEXT, model TEXT, "
        "provider TEXT, tool_ok INTEGER, turn_outcome TEXT, prompt_json TEXT, response_json TEXT)"
    )
    conn.execute(
        "INSERT INTO responses VALUES (1, '2026-11-27T21:00:00Z', 's1', 'qwen3.5-9b', 'llama_cpp', 1, 'ok', '{\"p\":1}', '{\"r\":1}')"
    )
    conn.commit()
    conn.close()

    cfg: dict[str, Any] = {
        "infrastructure": {"telemetry_enabled": True},
        "extensions": {
            "gym": {
                "enabled": True,
                "kind": "gym",
                "url": "http://127.0.0.1:18097",
                "token_env": "TEST_READ_TOK",
                "control_url": "http://127.0.0.1:18098",
                "control_token_env": "TEST_CTRL_TOK",
            },
            "readonly_ext": {
                "enabled": True,
                "kind": "voice",
                "url": "http://127.0.0.1:18097",
            },
        },
    }

    app = FastAPI()
    register_extension_routes(
        app,
        config=cfg,
        telemetry_db_path=lambda: db_path,
        client_factory=lambda: mock_client,
    )
    client = TestClient(app)

    # 1. Discovery never leaks URLs, tokens, or unexpected manifest keys
    r = client.get("/api/extensions")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    raw_text = r.text
    assert "18097" not in raw_text
    assert "read-secret-123" not in raw_text
    assert "secret_field_should_be_dropped" not in raw_text

    # 2. Read status uses read token only
    r = client.get("/api/extensions/gym/status")
    assert r.status_code == 200
    assert r.json()["step"] == 150

    # 3. Control endpoints use control_url + control token; read-only extension returns 403
    r = client.post("/api/extensions/readonly_ext/control/intent", json={"action": "promote"})
    assert r.status_code == 403
    assert r.json()["error"] == "control_not_configured"

    r = client.post("/api/extensions/gym/control/intent", json={"action": "promote", "candidate": "r6"})
    assert r.status_code == 200
    assert r.json()["intent_id"] == "int-1"

    r = client.post("/api/extensions/gym/control/confirm", json={"intent_id": "int-1"})
    assert r.status_code == 400

    r = client.post("/api/extensions/gym/control/confirm", json={"intent_id": "int-1", "plan_sha256": "abc123"})
    assert r.status_code == 200
    assert r.json()["status"] == "started"

    # Verify strict token separation between read URL and control URL
    for url, auth in seen_auth:
        if "18097" in url and auth is not None:
            assert auth == "Bearer read-secret-123"
        if "18098" in url:
            assert auth == "Bearer ctrl-secret-456"

    # 4. Trace export defaults to metadata-only (no prompt_json / response_json)
    r = client.get("/api/extensions/traces/export")
    assert r.status_code == 200
    assert r.json()["count"] == 1
    assert "prompt_json" not in r.json()["rows"][0]

    # With telemetry_enabled=False, trace export is refused
    cfg["infrastructure"]["telemetry_enabled"] = False
    r = client.get("/api/extensions/traces/export")
    assert r.status_code == 403
    assert r.json()["error"] == "telemetry_disabled"
