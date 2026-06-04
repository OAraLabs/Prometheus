"""Wire-contract fixes surfaced by Beacon Desktop Step 3 (branch feat/api-wire-contract).

Every test asserts the SIDE EFFECT, not just shape:
  #1 messages_since route actually FILTERS (timestamp > since) + returns the current
     watermark, and FAILS LOUD (400) on an unparseable ``since``.
  #2 client_msg_id is echoed on the WS user frame, correlated to the canonical msg-{turn_index}.
  #3 a new message's created_at is a real (non-zero) timestamp.
  #4 tool_call_start/end frames carry a consistent call_id.
  + the prerequisite bug fix: the WS/Beacon assistant turn is persisted to LCM.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.messages import ConversationMessage, TextBlock  # noqa: E402
from prometheus.engine.session import ChatSession, SessionManager  # noqa: E402
from prometheus.engine.stream_events import (  # noqa: E402
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402


# --------------------------------------------------------------------------- #
# #4 — tool_call_* frames carry a consistent call_id
# --------------------------------------------------------------------------- #


def test_tool_frames_carry_consistent_call_id(monkeypatch):
    from prometheus.engine import agent_loop as al

    async def fake_run_loop(ctx, messages):
        yield ToolExecutionStarted(tool_name="web_search", tool_input={"q": "x"}, tool_use_id="toolu_abc"), None
        yield ToolExecutionCompleted(tool_name="web_search", output="result", is_error=False, tool_use_id="toolu_abc"), None

    # _run_agent does `from prometheus.engine.agent_loop import run_loop` at call time.
    monkeypatch.setattr(al, "run_loop", fake_run_loop)

    captured: list[dict] = []
    bridge = WebSocketBridge(session_mgr=SessionManager(), loop_context=object())

    async def cap(ev):
        captured.append(ev)

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create("s")
    asyncio.run(bridge._run_agent("s", session))

    starts = [e for e in captured if e["type"] == "tool_call_start"]
    ends = [e for e in captured if e["type"] == "tool_call_end"]
    assert len(starts) == 1 and len(ends) == 1
    # SIDE EFFECT: the id is present AND the same on start + end (correlatable).
    assert starts[0]["payload"]["call_id"] == "toolu_abc"
    assert ends[0]["payload"]["call_id"] == "toolu_abc"
    assert starts[0]["payload"]["call_id"] == ends[0]["payload"]["call_id"]


# --------------------------------------------------------------------------- #
# #2 + #3 — user echo carries client_msg_id + canonical msg-N + real created_at
# --------------------------------------------------------------------------- #


def test_user_echo_correlates_client_msg_id_to_msg_n_with_real_timestamp():
    captured: list[dict] = []
    # loop_context=None → _handle_send_message broadcasts the echo but does NOT run an agent.
    bridge = WebSocketBridge(session_mgr=SessionManager())

    async def cap(ev):
        captured.append(ev)

    bridge.broadcast = cap
    asyncio.run(bridge.dispatch_user_message("desktop:test", "hello there", client_msg_id="cli-42"))

    msgs = [e for e in captured if e["type"] == "chat_message"]
    assert len(msgs) == 1
    p = msgs[0]["payload"]
    assert p["role"] == "user"
    # #2: echoed client id, correlated to the canonical durable ordinal (first msg → turn_index 0).
    assert p["client_msg_id"] == "cli-42"
    assert p["message_id"] == "msg-0"
    # #3: real timestamp, not 0.
    assert p["created_at"] > 0


def test_second_user_message_increments_msg_n():
    captured: list[dict] = []
    bridge = WebSocketBridge(session_mgr=SessionManager())

    async def cap(ev):
        captured.append(ev)

    bridge.broadcast = cap
    asyncio.run(bridge.dispatch_user_message("s", "first", client_msg_id="a"))
    asyncio.run(bridge.dispatch_user_message("s", "second", client_msg_id="b"))
    ids = [e["payload"]["message_id"] for e in captured if e["type"] == "chat_message"]
    assert ids == ["msg-0", "msg-1"]


# --------------------------------------------------------------------------- #
# prerequisite bug — the WS/Beacon assistant turn is persisted to LCM
# --------------------------------------------------------------------------- #


class _FakeLCM:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, str, int]] = []

    def ingest_sync(self, session_id, role, content, turn_index=0):
        self.ingested.append((role, content, turn_index))
        return f"id-{turn_index}"


def test_persist_loop_result_persists_tail_without_double_append():
    fake = _FakeLCM()
    s = ChatSession("s", lcm_engine=fake)
    s.add_user_message("hi")  # turn 0 (persisted by add_user_message)
    pre = len(s.get_messages())  # 1

    # Simulate run_loop appending the assistant turn IN PLACE (as it does).
    s.messages.append(ConversationMessage(role="assistant", content=[TextBlock(text="hello back")]))
    s.persist_loop_result(pre)

    # SIDE EFFECT: the assistant message reached LCM exactly once at turn_index 1...
    assert ("assistant", "hello back", 1) in fake.ingested
    assert sum(1 for r in fake.ingested if r[0] == "assistant") == 1
    # ...and was NOT re-appended to the in-memory list (still user + assistant).
    assert len(s.get_messages()) == 2


def test_run_agent_persists_assistant_turn(monkeypatch):
    from prometheus.engine import agent_loop as al

    async def fake_run_loop(ctx, messages):
        # run_loop mutates the list in place; mimic that for the assistant turn.
        messages.append(ConversationMessage(role="assistant", content=[TextBlock(text="answer")]))
        if False:
            yield  # make this an async generator

    monkeypatch.setattr(al, "run_loop", fake_run_loop)

    fake = _FakeLCM()
    # _run_agent operates on the session it's handed — drive it directly (dispatch
    # schedules it as a fire-and-forget task that wouldn't finish inside asyncio.run).
    session = ChatSession("s", lcm_engine=fake)
    session.add_user_message("question")  # turn 0

    bridge = WebSocketBridge(loop_context=object())

    async def cap(ev):
        pass

    bridge.broadcast = cap
    asyncio.run(bridge._run_agent("s", session))
    # The assistant turn run_loop appended in place must reach LCM at turn_index 1.
    assert ("assistant", "answer", 1) in fake.ingested


# --------------------------------------------------------------------------- #
# #1 — messages_since route filters + returns watermark + fails loud
# --------------------------------------------------------------------------- #


def _app_with_messages(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    for i, ts in enumerate([100.0, 200.0, 300.0]):
        store.insert_message(
            MessagePart(
                role="user" if i % 2 == 0 else "assistant",
                content=f"m{i}",
                session_id="s",
                turn_index=i,
                timestamp=ts,
            )
        )

    class _LCM:  # the route only needs `.conversation_store`
        conversation_store = store

    app = create_app({}, lcm_engine=_LCM())
    return TestClient(app)


def test_messages_full_read_has_real_timestamps_and_watermark(tmp_path):
    client = _app_with_messages(tmp_path)
    body = client.get("/api/sessions/s/messages").json()
    assert [m["message_id"] for m in body["messages"]] == ["msg-0", "msg-1", "msg-2"]
    # #3: real timestamps, not 0.
    assert [m["timestamp"] for m in body["messages"]] == [100.0, 200.0, 300.0]
    assert body["watermark"] == 300.0


def test_messages_since_filters_strictly_greater(tmp_path):
    client = _app_with_messages(tmp_path)
    body = client.get("/api/sessions/s/messages?since=150").json()
    # SIDE EFFECT: only messages with timestamp > 150 come back.
    assert [m["content"] for m in body["messages"]] == ["m1", "m2"]
    assert body["watermark"] == 300.0


def test_messages_since_caught_up_returns_empty_delta_but_current_watermark(tmp_path):
    client = _app_with_messages(tmp_path)
    body = client.get("/api/sessions/s/messages?since=300").json()
    assert body["messages"] == []
    # Empty delta, but the client still learns it's at the head.
    assert body["watermark"] == 300.0


def test_messages_since_invalid_is_400_not_silently_ignored(tmp_path):
    client = _app_with_messages(tmp_path)
    resp = client.get("/api/sessions/s/messages?since=notanumber")
    assert resp.status_code == 400
    assert "since" in resp.json()["error"]
