"""Every APNs push names the node that sent it (Beacon iOS "Piece 14").

A phone paired with several Prometheus instances gets pushes from each. The body used to say nothing about
which one: ``request_id``, ``session_id`` and the alert text are the same shape whichever daemon sent them. The
node has a public identity already, the Ed25519 ``node_pub`` that ``GET /api/status`` and the identity challenge
carry and that a paired phone pinned when it paired. So every push carries it, under the top-level key
``node_pub``, and the ``subscribed`` ack lists ``push_node_pub`` in ``server_capabilities`` so a client knows
this daemon does (an older one does not, and a client must not wait for the key from it).

Held here:

  * every body the dispatcher builds carries it: both approval shapes, the quiet pushes, and both Live Activity
    bodies (update and end);
  * the value is the node's own public key, 32 raw bytes in base64: anything else is left out, never pushed, and
    a provider that raises, or a node with no key yet, costs the push nothing;
  * the capability is advertised only when it is true: push is on (a dispatcher exists) and the node has a key.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("cryptography")

from prometheus.config import node_identity  # noqa: E402
from prometheus.push.dispatcher import PushDispatcher  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402

PUB = base64.b64encode(bytes(range(32))).decode()


class _Store:
    def push_targets(self):
        return [SimpleNamespace(id="d1", apns_token="t", environment="sandbox", bundle_id="", push_failures=0)]

    def activity_targets(self, session_id):
        return [(self.push_targets()[0], "activity-token")]

    def clear_activity_token(self, device_id, session_id):
        pass


class _Sender:
    config = SimpleNamespace(topic="ai.test")

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, **kwargs):
        self.sent.append(kwargs["payload"])
        from prometheus.push.apns import SendResult

        return SendResult(outcome="ok", status=200)


def _dispatcher(**kwargs):
    sender = _Sender()
    d = PushDispatcher(_Store(), sender, bridge=None, **kwargs)
    return d, sender


def _all_pushes(d, sender) -> list[dict]:
    """One of every push the dispatcher makes."""
    run = asyncio.run
    sig = lambda kind, payload: SimpleNamespace(kind=kind, payload=payload)    # noqa: E731
    approval = {"request_id": "r1", "tool_name": "bash", "description": "ls", "expires_at": 1.0}
    run(d.on_signal(sig("approval_pending", approval)))
    run(d.on_signal(sig("approval_pending", {**approval, "task_id": "t1"})))
    for kind in ("turn_completed", "task_completed", "task_failed", "coding_complete"):
        run(d.on_signal(sig(kind, {"session_id": "s1", "task_id": "t9", "summary": "done"})))
    run(d.on_agent_progress("s1", {"phase": "tools", "round": 1}))
    run(d.on_chat_done("s1", {}))
    return sender.sent


def test_every_push_the_dispatcher_builds_carries_the_node_key():
    d, sender = _dispatcher(node_pub=lambda: PUB)
    bodies = _all_pushes(d, sender)
    assert len(bodies) == 8, f"expected one of every push, got {len(bodies)}"
    for body in bodies:
        assert body["node_pub"] == PUB, body


def test_the_value_is_the_nodes_own_public_key_by_default():
    node_identity.ensure_node_identity()
    pub = node_identity.get_node_pubkey()
    assert pub and len(base64.b64decode(pub)) == 32, "the fixture has no real node key; the test proves nothing"
    d, sender = _dispatcher()
    assert {b["node_pub"] for b in _all_pushes(d, sender)} == {pub}


@pytest.mark.parametrize("bad", ["", None, 5, b"x", "not base64!", base64.b64encode(b"short").decode(),
                                 base64.b64encode(bytes(33)).decode(), PUB + "\n", " " + PUB])
def test_a_value_that_is_not_a_node_key_is_not_pushed(bad):
    d, sender = _dispatcher(node_pub=lambda: bad)
    bodies = _all_pushes(d, sender)
    assert len(bodies) == 8, "a bad key stopped a push from being sent"
    assert all("node_pub" not in b for b in bodies)


def test_a_provider_that_raises_does_not_stop_a_push():
    def boom():
        raise OSError("the node directory is unreadable")

    d, sender = _dispatcher(node_pub=boom)
    bodies = _all_pushes(d, sender)
    assert len(bodies) == 8 and all("node_pub" not in b for b in bodies)


def test_with_no_node_key_yet_the_field_is_absent_and_the_push_goes():
    assert node_identity.get_node_pubkey() is None, "the fixture already has a node key"
    d, sender = _dispatcher()
    bodies = _all_pushes(d, sender)
    assert len(bodies) == 8 and all("node_pub" not in b for b in bodies)


def test_nothing_but_the_public_key_goes_out():
    node_identity.ensure_node_identity()
    d, sender = _dispatcher()
    blob = json.dumps(_all_pushes(d, sender))
    key_file = node_identity.get_node_dir() / node_identity.NODE_KEY_FILENAME
    assert key_file.exists()
    private = key_file.read_bytes()
    assert base64.b64encode(private).decode()[:40] not in blob and private[:40].hex() not in blob


# ── the capability ───────────────────────────────────────────────────────────

class _Sock:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))


def _capabilities(*, dispatcher) -> list[str]:
    bridge = WebSocketBridge(session_mgr=None, loop_context=None, config={})
    if dispatcher:
        bridge.push_dispatcher = PushDispatcher(_Store(), _Sender(), bridge=bridge)
    ws = _Sock()
    bridge._clients.add(ws)
    asyncio.run(bridge._handle_client_message(ws, json.dumps({"type": "subscribe", "payload": {}})))
    return [f for f in ws.frames if f["type"] == "subscribed"][-1]["payload"]["server_capabilities"]


def test_the_ack_lists_push_node_pub_when_push_is_on_and_the_node_has_a_key():
    node_identity.ensure_node_identity()
    assert "push_node_pub" in _capabilities(dispatcher=True)


def test_the_ack_does_not_list_it_without_push():
    node_identity.ensure_node_identity()
    assert "push_node_pub" not in _capabilities(dispatcher=False)


def test_the_ack_does_not_list_it_without_a_node_key():
    assert node_identity.get_node_pubkey() is None
    assert "push_node_pub" not in _capabilities(dispatcher=True)
