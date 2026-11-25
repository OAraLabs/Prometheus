"""The approve and deny routes take ONE request id, and nothing else (Will, 2026-11-13).

``POST /api/approvals/{request_id}/approve`` built its chat-command text from the path segment, so the "id" ``all``
answered every pending request at once. A client acts on exactly the request it shows: now any path segment that is
not an id the queues mint (8 lowercase hex, ``approval_queue.new_request_id``) is a 404, before anything reads it, on
approve and on deny. Approving several requests at once is a chat command (``/approve all``), not offered over REST.
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
    is_request_id,
    new_request_id,
)

GLOBAL = "one-id-" + secrets.token_hex(8)


class _Gate:
    def add_grant(self, *a, **kw):
        return None


@pytest.fixture
def d(tmp_path):
    from prometheus.web.server import create_app

    devices = DeviceStore(tmp_path / "devices.db")
    app = create_app({"web": {"api_token": GLOBAL}}, device_store=devices)
    queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    app.state.approval_queue = queue
    owner = devices.mint_owner("owner phone", "ios", by="test")
    client = TestClient(app, client=("127.0.0.1", 50123))
    actions = []
    for rid in ("a0a0a001", "a0a0a002", "a0a0a003"):
        a = PendingAction(request_id=rid, tool_name="bash", description="run ls", session_id="web:s1")
        queue._register(a)
        actions.append(a)
    return client, queue, actions, owner


@pytest.mark.parametrize("who", ["global", "owner"])
@pytest.mark.parametrize("verb", ["approve", "deny"])
@pytest.mark.parametrize("bad", ["all", "ALL", "always", "once", "until-restart", "A0A0A001", "a0a0a00",
                                 "a0a0a0011", "a0a0a00g", "a0a0a001 all", "%20all", "all%20a0a0a001", "-"])
def test_anything_but_one_id_is_a_404_and_answers_nothing(d, who, verb, bad):
    client, queue, actions, owner = d
    token = GLOBAL if who == "global" else owner["token"]
    r = client.post(f"/api/approvals/{bad}/{verb}", headers={"Authorization": f"Bearer {token}"},
                    json={"scope": "once"} if verb == "approve" else None)
    assert r.status_code == 404, (bad, r.status_code, r.text)
    assert all(a._result == ApprovalResult.TIMEOUT for a in actions), bad
    assert set(queue.pending) == {"a0a0a001", "a0a0a002", "a0a0a003"}


@pytest.mark.parametrize("verb", ["approve", "deny"])
def test_one_id_answers_that_request_and_no_other(d, verb):
    client, queue, actions, _owner = d
    r = client.post(f"/api/approvals/a0a0a002/{verb}", headers={"Authorization": f"Bearer {GLOBAL}"},
                    json={"scope": "once"} if verb == "approve" else None)
    assert r.status_code == 200, r.text
    want = ApprovalResult.APPROVED if verb == "approve" else ApprovalResult.DENIED
    assert actions[1]._result == want
    assert actions[0]._result == ApprovalResult.TIMEOUT and actions[2]._result == ApprovalResult.TIMEOUT


def test_the_id_shape_is_one_definition():
    for _ in range(50):
        assert is_request_id(new_request_id())
    assert not is_request_id("all") and not is_request_id("") and not is_request_id(None)


def test_every_queue_mints_through_the_one_helper():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "src" / "prometheus"
    for rel in ("permissions/approval_queue.py", "permissions/hold_lifts.py", "computer/approvals.py"):
        text = (root / rel).read_text()
        assert "request_id = uuid4().hex[:8]" not in text, rel
