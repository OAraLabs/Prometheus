"""A daemon with an operator and two enrolled devices, for the device-scoping tests.

Real objects throughout: the FastAPI app, the WebSocketBridge, a SessionManager,
an LCM conversation store and a DeviceStore. The only stand-in is the engine
wrapper, which is just enough of LCMEngine to persist through the real store
(the same shape tests/test_api_sessions_durable.py uses).
"""

from __future__ import annotations

import secrets

from fastapi.testclient import TestClient

from prometheus.config.device_store import DeviceStore
from prometheus.engine.session import SessionManager
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_summary_store import LCMSummaryStore
from prometheus.memory.lcm_types import MessagePart, SummaryNode
from prometheus.web.server import create_app
from prometheus.web.ws_server import WebSocketBridge

GLOBAL = "glob-" + secrets.token_hex(8)

# What a device stored in its own session, and what the operator stored in a
# Telegram chat. The assertions look for these strings in response bodies.
A_SECRET = "a-private-thought-" + secrets.token_hex(4)
TG_SECRET = "operator-telegram-secret-" + secrets.token_hex(4)


def engine_over(store: LCMConversationStore, summaries: LCMSummaryStore | None = None):
    class _Engine:
        conversation_store = store
        summary_store = summaries

        def ingest_sync(self, session_id, role, content, turn_index=0, content_json=None,
                        provenance="user", is_trusted=True):
            m = MessagePart(role=role, content=content, session_id=session_id, turn_index=turn_index,
                            provenance=provenance, is_trusted=is_trusted)
            store.add_message(session_id, m)
            return m.message_id

        async def maybe_compact(self, session_id):
            return None  # nothing to compact; a turn's tail calls this

    return _Engine()


class World:
    """One daemon with an operator and two enrolled devices, A and B."""

    def __init__(self, tmp_path, *, with_ws_bridge: bool = True) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.lcm = LCMConversationStore(tmp_path / "lcm.db")
        self.summaries = LCMSummaryStore(tmp_path / "lcm.db")  # same file, as in the daemon
        self.engine = engine_over(self.lcm, self.summaries)
        self.mgr = SessionManager()
        self.mgr.lcm_engine = self.engine
        self.app = create_app(
            {"web": {"api_token": GLOBAL}},
            session_mgr=self.mgr, lcm_engine=self.engine, device_store=self.devices,
        )
        # loop_context=None: the user's message is persisted and broadcast, no agent runs.
        self.bridge = WebSocketBridge(
            session_mgr=self.mgr, api_token=GLOBAL, device_store=self.devices,
            loop_context=None,
        )
        if with_ws_bridge:
            self.app.state.ws_bridge = self.bridge
        self.client = TestClient(self.app)
        self.a = self.devices.mint("device-a", "ios")
        self.b = self.devices.mint("device-b", "macos")

    # -- who is calling -------------------------------------------------
    def hdr(self, who: str) -> dict:
        token = {"op": GLOBAL, "a": self.a["token"], "b": self.b["token"]}[who]
        return {"Authorization": f"Bearer {token}"}

    def call(self, who: str, method: str, url: str, **kw):
        return self.client.request(method, url, headers=self.hdr(who), **kw)

    # -- seeding --------------------------------------------------------
    def device_session(self, who: str = "a", text: str = A_SECRET) -> str:
        """A session a device legitimately created and spoke in (mint + first send)."""
        sid = self.call(who, "POST", "/api/sessions").json()["session_id"]
        r = self.call(who, "POST", "/api/chat/send", json={"session_id": sid, "message": text})
        assert r.status_code == 200, r.text
        return sid

    def operator_session(self, sid: str = "telegram:123", text: str = TG_SECRET) -> str:
        """A conversation nobody owns: written by a gateway, not by any device."""
        self.mgr.get_or_create(sid).add_user_message(text)
        return sid

    def seed_summary(self, sid: str, text: str) -> None:
        """A compaction summary for *sid*, anchored on its first message."""
        first = self.lcm.get_all_messages(sid)[0]
        self.summaries.insert_summary(
            SummaryNode(source_message_ids=[first.message_id], summary_text=text, depth=0),
            session_id=sid,
        )
