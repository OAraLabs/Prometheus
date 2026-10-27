"""`/api/status` carries node + instance identity — FOUNDATION Part 3.

The wire names (`node_pub`, `instance_id`) are a contract with the Beacon
iOS client, which already decodes both optionally and renders NODE /
INSTANCE rows once they are non-null. Renaming either breaks a shipped
client silently — the fields may gain siblings, never new names.

Bearer-side only: /health deliberately does not carry identity, matching
the standing /api/status-vs-/health disclosure split.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.node_identity import (  # noqa: E402
    ensure_node_identity,
    set_instance_id,
)
from prometheus.web.server import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_instance_pin():
    yield
    set_instance_id(None)


def _client() -> TestClient:
    return TestClient(create_app({}))


def test_status_keys_present_and_null_before_identity_exists() -> None:
    # Explicit nulls, not absent keys: the client distinguishes "daemon too
    # old to know about identity" (key missing) from "identity not minted
    # yet" (key null).
    body = _client().get("/api/status").json()
    assert "node_pub" in body and body["node_pub"] is None
    assert "instance_id" in body and body["instance_id"] is None


def test_status_serves_the_minted_node_and_pinned_instance() -> None:
    identity = ensure_node_identity()
    set_instance_id("550e8400-e29b-41d4-a716-446655440000")
    body = _client().get("/api/status").json()
    assert body["node_pub"] == identity.pubkey
    assert body["instance_id"] == "550e8400-e29b-41d4-a716-446655440000"


def test_health_carries_no_identity() -> None:
    body = _client().get("/health").json()
    assert "node_pub" not in body
    assert "instance_id" not in body


# ── PD-1: the same name and fp hello advertises, beside node_pub / instance_id ──
#
# Siblings, never renames. Read through hello's own functions (display_name, instance_fingerprint), so the
# two routes cannot drift; /health still carries neither.


def test_status_carries_hellos_name_and_fp() -> None:
    from prometheus.web.hello import build_hello

    config = {"pairing": {"display_name": "Work box"}}
    body = TestClient(create_app(config)).get("/api/status").json()
    hello = build_hello(config)
    assert body["name"] == hello["name"] == "Work box"
    assert body["fp"] == hello["fp"]


def test_status_fp_is_the_instance_fingerprint_once_the_key_exists() -> None:
    from prometheus.config.instance_key import ensure_instance_key, instance_fingerprint

    ensure_instance_key()
    fp = instance_fingerprint()
    assert len(fp) == 16
    assert _client().get("/api/status").json()["fp"] == fp


def test_status_fp_is_empty_before_the_key_exists_and_reading_mints_nothing() -> None:
    from prometheus.config.instance_key import instance_public_key_der

    assert instance_public_key_der() is None
    body = _client().get("/api/status").json()
    assert "fp" in body and body["fp"] == ""
    assert instance_public_key_der() is None, "a GET must not mint the instance key"


def test_health_carries_no_name_or_fp() -> None:
    body = _client().get("/health").json()
    assert "name" not in body and "fp" not in body
