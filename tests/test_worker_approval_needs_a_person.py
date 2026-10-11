"""A worker's approval request is answered by a person: never the API token, never an open daemon.

#720 files a worker's request under its parent conversation for "a person" to
answer, but ``POST /api/approvals/{id}/approve`` took any credential, the
master token included. A worker's model shell can reach that token (its
parent's environment, a plain file the bash tool can read), so the worker
could approve its own request. Open mode is the same hole with no token at all.

The rule here, Will's call (2026-10-11): a device token is a person's (the
owner's device, or the scoped device whose conversation it is: the ownership
filter already decides which). The API token and open mode get 401
``not_a_person`` and the request stays pending. Unlike a desktop prompt, this
needs no computer-use mark and works with computer use off.
"""

from __future__ import annotations

import secrets

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
)
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "worker-person-global-" + secrets.token_hex(8)
ALICE_SESSION = "web:alice-1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))


class Daemon:
    def __init__(self, tmp_path, *, api_token: str | None = GLOBAL) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        web = {"api_token": api_token} if api_token else {}
        self.app = create_app({"web": web}, device_store=self.devices)
        assert getattr(self.app.state, "computer_runner", None) is None, "computer use stays off"
        self.queue = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT,
                                                              audit_logger=None))
        self.app.state.approval_queue = self.queue
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.alice = self.devices.mint("alice's phone", "ios")
        self.devices.claim_session(ALICE_SESSION, self.alice["id"])
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def approve(self, who: str | None, request_id: str, scope: str = "once"):
        headers = {}
        if who is not None:
            token = {"global": GLOBAL, "owner": self.owner["token"],
                     "alice": self.alice["token"]}[who]
            headers["Authorization"] = f"Bearer {token}"
        return self.client.post(f"/api/approvals/{request_id}/approve",
                                headers=headers, json={"scope": scope})

    def worker_request(self, request_id: str) -> PendingAction:
        action = PendingAction(request_id=request_id, tool_name="write_file",
                               description="write_file needs approval: ./notes.md",
                               session_id=ALICE_SESSION, once_only=True,
                               worker_task_id="a0000worker")
        self.queue.pending[request_id] = action
        return action


@pytest.fixture
def d(tmp_path) -> Daemon:
    return Daemon(tmp_path)


def _not_a_person(response) -> bool:
    return response.status_code == 401 and response.json().get("code") == "not_a_person"


def test_the_api_token_cannot_approve_a_workers_request(d):
    action = d.worker_request("f0f00001")
    r = d.approve("global", "f0f00001")
    assert _not_a_person(r), r.text
    assert "f0f00001" in d.queue.pending and action._result == ApprovalResult.TIMEOUT


def test_an_open_daemon_cannot_approve_a_workers_request(tmp_path):
    d = Daemon(tmp_path, api_token=None)
    action = d.worker_request("f0f00002")
    r = d.approve(None, "f0f00002")
    assert _not_a_person(r), r.text
    assert "f0f00002" in d.queue.pending and action._result == ApprovalResult.TIMEOUT


def test_the_owners_device_approves_a_workers_request(d):
    action = d.worker_request("f0f00003")
    r = d.approve("owner", "f0f00003")
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert action._result == ApprovalResult.APPROVED


def test_the_device_whose_conversation_it_is_approves_a_workers_request(d):
    """No computer-use mark needed: the mark is the desktop's grant, not a person test."""
    action = d.worker_request("f0f00004")
    assert not d.devices.computer_allowed(d.alice["id"])
    r = d.approve("alice", "f0f00004")
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert action._result == ApprovalResult.APPROVED


def test_a_lasting_scope_is_still_refused_by_name_before_the_credential(d):
    d.worker_request("f0f00005")
    r = d.approve("owner", "f0f00005", scope="always")
    assert r.status_code == 409 and r.json()["code"] == "once_only", r.text
    assert "f0f00005" in d.queue.pending


def test_approve_all_with_the_api_token_leaves_a_workers_request_pending(d):
    d.worker_request("f0f00006")
    r = d.approve("global", "all")
    assert r.status_code == 200, r.text
    assert "f0f00006" in d.queue.pending


def test_an_ordinary_request_is_unchanged_the_api_token_still_answers_it(d):
    """Only a worker's request needs a person here; W4 decides the rest."""
    action = PendingAction(request_id="bbbb0001", tool_name="bash", description="run echo hi",
                           grant_command="echo hi", session_id=ALICE_SESSION)
    d.queue.pending["bbbb0001"] = action
    r = d.approve("global", "bbbb0001")
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert action._result == ApprovalResult.APPROVED
