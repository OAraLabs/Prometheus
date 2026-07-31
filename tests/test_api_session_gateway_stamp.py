"""POST /api/sessions stamps the origin gateway into the session id.

Session ids follow ``<gateway>:<id>`` — the convention Telegram, voice and the
bakeoff harness already use, and the one GET /api/sessions parses to report a
row's ``gateway``. This route used to mint a BARE uuid, so every Beacon-created
session reported ``gateway: "unknown"`` (30 of 127 on the live box) and rendered
as an unidentifiable "UN" chip, even though they were ordinary desktop chats.

The id is split on the FIRST colon, so a colon inside the gateway would silently
truncate the id — hence the charset guard, which is tested here rather than
trusted.
"""

from __future__ import annotations

import re
import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


def _client():
    mgr = SessionManager()
    return TestClient(create_app({}, session_mgr=mgr)), mgr


def _uuid_ok(text: str) -> bool:
    try:
        uuid.UUID(text)
        return True
    except ValueError:
        return False


def test_defaults_to_desktop_prefix():
    client, mgr = _client()
    body = client.post("/api/sessions").json()
    sid = body["session_id"]
    assert sid.startswith("desktop:"), sid
    assert body["gateway"] == "desktop"
    assert _uuid_ok(sid.split(":", 1)[1]), "the tail must still be a real uuid"
    # And the session actually exists under that exact id.
    assert mgr.get(sid) is not None


def test_explicit_gateway_is_honored():
    client, _ = _client()
    body = client.post("/api/sessions", json={"gateway": "voice"}).json()
    assert body["session_id"].startswith("voice:")
    assert body["gateway"] == "voice"


def test_gateway_is_lowercased_for_a_stable_parse():
    client, _ = _client()
    body = client.post("/api/sessions", json={"gateway": "Desktop"}).json()
    assert body["session_id"].startswith("desktop:")
    assert body["gateway"] == "desktop"


def test_no_body_still_works():
    """The route had no required input before; a bodyless POST must not 400."""
    client, _ = _client()
    resp = client.post("/api/sessions")
    assert resp.status_code == 200
    assert resp.json()["session_id"].startswith("desktop:")


def test_malformed_body_falls_back_to_default():
    client, _ = _client()
    resp = client.post(
        "/api/sessions", content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"].startswith("desktop:")


@pytest.mark.parametrize(
    "bad",
    [
        "has:colon",      # would truncate the id on the first-colon split
        "with space",
        "sym!bol",
        "",
        "x" * 33,         # over the length cap
        "-leading-dash",  # must start alnum
    ],
)
def test_invalid_gateways_are_rejected(bad):
    client, _ = _client()
    resp = client.post("/api/sessions", json={"gateway": bad})
    assert resp.status_code == 400, f"{bad!r} should be rejected"
    assert "invalid gateway" in resp.json()["error"]


def test_ids_are_unique_across_calls():
    client, _ = _client()
    ids = {client.post("/api/sessions").json()["session_id"] for _ in range(5)}
    assert len(ids) == 5


def test_stamped_session_reports_its_gateway_in_the_listing():
    """End-to-end: the whole point is that GET /api/sessions stops saying
    'unknown' for these."""
    client, _ = _client()
    sid = client.post("/api/sessions", json={"gateway": "desktop"}).json()["session_id"]
    rows = client.get("/api/sessions").json()
    row = next(r for r in rows if r["session_id"] == sid)
    assert row["gateway"] == "desktop"
    assert row["gateway"] != "unknown"


def test_legacy_bare_uuid_sessions_still_parse_as_unknown():
    """Existing bare-uuid sessions are immutable and must keep working."""
    client, mgr = _client()
    legacy = str(uuid.uuid4())
    mgr.get_or_create(legacy)
    rows = client.get("/api/sessions").json()
    row = next(r for r in rows if r["session_id"] == legacy)
    assert row["gateway"] == "unknown"


def test_generated_id_shape_is_parseable():
    client, _ = _client()
    sid = client.post("/api/sessions").json()["session_id"]
    assert re.fullmatch(r"[a-z0-9_-]{1,32}:[0-9a-f-]{36}", sid), sid
    # The daemon's own parse (split on first colon) recovers the gateway.
    assert sid[: sid.index(":")] == "desktop"
