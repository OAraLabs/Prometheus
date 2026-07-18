"""POST /api/paperclip/wake — webhook target for Paperclip's ``http`` adapter.

Route-level contract: 503 when the gateway isn't enabled, 400 on malformed
wakes, accepted + heartbeat spawn on a valid wake, and standard /api/ bearer
enforcement (the Paperclip adapter config carries the Authorization header).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.gateway.paperclip import PaperclipGateway  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.doubles import register_double  # noqa: E402


@register_double(
    "api_paperclip_wake._RecordingGateway",
    replaces="prometheus.gateway.paperclip.PaperclipGateway",
)
class _RecordingGateway:
    """Real wake parsing, recorded heartbeat spawn (no Paperclip server)."""

    parse_wake = PaperclipGateway.parse_wake

    def __init__(self) -> None:
        self.started = []

    def start_heartbeat(self, wake) -> None:
        self.started.append(wake)


def _client(gateway=None, cfg=None):
    app = create_app(cfg or {})
    if gateway is not None:
        app.state.paperclip_gateway = gateway
    return TestClient(app), app


_WAKE = {
    "runId": "run-1",
    "agentId": "agent-1",
    "companyId": "co-1",
    "context": {"taskId": "i-1", "wakeReason": "issue_assigned"},
}


def test_valid_wake_accepted_and_heartbeat_spawned():
    gw = _RecordingGateway()
    client, _ = _client(gw)

    resp = client.post("/api/paperclip/wake", json=_WAKE)

    assert resp.status_code == 200
    assert resp.json() == {"status": "accepted", "run_id": "run-1"}
    assert len(gw.started) == 1
    wake = gw.started[0]
    assert (wake.agent_id, wake.issue_id, wake.wake_reason) == (
        "agent-1", "i-1", "issue_assigned",
    )


def test_gateway_not_enabled_returns_503_not_silent_success():
    client, _ = _client(gateway=None)
    resp = client.post("/api/paperclip/wake", json=_WAKE)
    assert resp.status_code == 503
    assert "paperclip" in resp.json()["error"]


def test_invalid_json_returns_400():
    client, _ = _client(_RecordingGateway())
    resp = client.post(
        "/api/paperclip/wake",
        content=b"not-json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


def test_non_object_body_returns_400():
    client, _ = _client(_RecordingGateway())
    resp = client.post("/api/paperclip/wake", json=["not", "an", "object"])
    assert resp.status_code == 400


@pytest.mark.parametrize("bad", [{}, {"runId": "r"}, {"agentId": "a"}])
def test_missing_ids_return_400_and_no_spawn(bad):
    gw = _RecordingGateway()
    client, _ = _client(gw)
    resp = client.post("/api/paperclip/wake", json=bad)
    assert resp.status_code == 400
    assert gw.started == []


def test_bearer_token_enforced_like_every_api_route():
    gw = _RecordingGateway()
    client, _ = _client(gw, cfg={"web": {"api_token": "sekrit"}})

    resp = client.post("/api/paperclip/wake", json=_WAKE)
    assert resp.status_code == 401
    assert gw.started == []

    resp = client.post(
        "/api/paperclip/wake", json=_WAKE,
        headers={"authorization": "Bearer sekrit"},
    )
    assert resp.status_code == 200
    assert len(gw.started) == 1
