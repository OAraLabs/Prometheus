"""Wire-contract fixes surfaced by Beacon Desktop Step 3 (branch feat/api-wire-contract),
updated for durable message identity (branch fix/durable-message-id).

Every test asserts the SIDE EFFECT, not just shape:
  #1 messages route actually FILTERS by the durable rowid cursor + returns the watermark,
     and FAILS LOUD (400) on an unparseable ``since``.
  #2 client_msg_id is echoed on the WS user frame, correlated to the durable rowid message_id.
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
from tests.support.doubles import register_double  # noqa: E402


def _pin_run_loop_signature(fake) -> None:
    """Drift pin: fail LOUDLY if a fake_run_loop's signature no longer matches the
    real ``run_loop``. Without this, a new run_loop kwarg becomes a TypeError inside
    bridge._run_agent's broad try/except (swallowed) and surfaces as a misleading
    downstream assertion — the exact #74 double-drift this file shipped red with on
    bbfbb35 (fake lacked ``session_id`` after ws_server.py:510 started passing it).

    Compares positional arity + the exact set of keyword-only parameter names, so a
    positional rename doesn't false-alarm but any callable-interface change does.
    """
    import inspect

    from prometheus.engine.agent_loop import run_loop as _real

    real_sig, fake_sig = inspect.signature(_real), inspect.signature(fake)

    def _split(sig):
        pos = [p.name for p in sig.parameters.values()
               if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        kw = sorted(p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY)
        return pos, kw

    real_pos, real_kw = _split(real_sig)
    fake_pos, fake_kw = _split(fake_sig)
    assert len(fake_pos) == len(real_pos) and fake_kw == real_kw, (
        "fake_run_loop SIGNATURE DRIFTED from the real run_loop:\n"
        f"  real: {real_sig}\n  fake: {fake_sig}\n"
        "Update the double to match the real interface — do not rely on "
        "bridge._run_agent to surface the TypeError (it swallows it)."
    )


def _lcm_manager(tmp_path):
    """A SessionManager wired to a real LCM store on a temp db (so sessions persist and
    expose real, monotonic rowids)."""
    store = LCMConversationStore(tmp_path / "lcm.db")

    class _Engine:
        conversation_store = store

        def ingest_sync(self, session_id, role, content, turn_index=0, content_json=None,
                        provenance="user", is_trusted=True):
            m = MessagePart(role=role, content=content, session_id=session_id, turn_index=turn_index,
                            provenance=provenance, is_trusted=is_trusted)
            store.add_message(session_id, m)
            return m.message_id

    mgr = SessionManager()
    mgr.lcm_engine = _Engine()
    return mgr, store


# --------------------------------------------------------------------------- #
# #4 — tool_call_* frames carry a consistent call_id
# --------------------------------------------------------------------------- #


def test_tool_frames_carry_consistent_call_id(monkeypatch):
    from prometheus.engine import agent_loop as al

    async def fake_run_loop(ctx, messages, *, mode="agent", session_id=None, tool_choice=None):
        yield ToolExecutionStarted(tool_name="web_search", tool_input={"q": "x"}, tool_use_id="toolu_abc"), None
        yield ToolExecutionCompleted(tool_name="web_search", output="result", is_error=False, tool_use_id="toolu_abc"), None

    _pin_run_loop_signature(fake_run_loop)
    monkeypatch.setattr(
        al, "run_loop",
        register_double("wire_contract.fake_run_loop.tool_frames", replaces="prometheus.engine.agent_loop.run_loop")(fake_run_loop),
    )

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
    assert starts[0]["payload"]["call_id"] == "toolu_abc"
    assert ends[0]["payload"]["call_id"] == "toolu_abc"
    assert starts[0]["payload"]["call_id"] == ends[0]["payload"]["call_id"]


# --------------------------------------------------------------------------- #
# #2 + #3 — user echo carries client_msg_id + durable rowid message_id + real created_at
# --------------------------------------------------------------------------- #


def test_user_echo_uses_durable_rowid_and_correlates_client_msg_id(tmp_path):
    mgr, store = _lcm_manager(tmp_path)
    captured: list[dict] = []
    bridge = WebSocketBridge(session_mgr=mgr)  # loop_context=None → echo only, no agent

    async def cap(ev):
        captured.append(ev)

    bridge.broadcast = cap
    asyncio.run(bridge.dispatch_user_message("desktop:test", "hello there", client_msg_id="cli-42"))

    msgs = [e for e in captured if e["type"] == "chat_message"]
    assert len(msgs) == 1
    p = msgs[0]["payload"]
    assert p["role"] == "user"
    assert p["client_msg_id"] == "cli-42"  # #2 echoed
    # message_id is the durable LCM rowid (int, >0) and equals the just-persisted row.
    assert isinstance(p["message_id"], int) and p["message_id"] > 0
    assert p["message_id"] == store.max_rowid("desktop:test")
    assert p["ordinal"] == 0  # turn_index is the (non-unique) display position
    assert p["created_at"] > 0  # #3 real timestamp


def test_message_id_is_durable_rowid_not_resetting_ordinal(tmp_path):
    mgr, _ = _lcm_manager(tmp_path)
    captured: list[dict] = []
    bridge = WebSocketBridge(session_mgr=mgr)

    async def cap(ev):
        captured.append(ev)

    bridge.broadcast = cap
    asyncio.run(bridge.dispatch_user_message("s", "first", client_msg_id="a"))
    asyncio.run(bridge.dispatch_user_message("s", "second", client_msg_id="b"))
    payloads = [e["payload"] for e in captured if e["type"] == "chat_message"]
    ids = [p["message_id"] for p in payloads]
    ordinals = [p["ordinal"] for p in payloads]
    assert ordinals == [0, 1]
    # Durable ids are distinct + strictly increasing (monotonic), not "msg-0"/"msg-1".
    assert ids[0] < ids[1]
    assert len(set(ids)) == 2


# --------------------------------------------------------------------------- #
# prerequisite bug — the WS/Beacon assistant turn is persisted to LCM
# --------------------------------------------------------------------------- #


class _FakeLCM:
    def __init__(self) -> None:
        self.ingested: list[tuple[str, str, int]] = []

    def ingest_sync(self, session_id, role, content, turn_index=0, content_json=None,
                    provenance="user", is_trusted=True):
        self.ingested.append((role, content, turn_index))
        return f"id-{turn_index}"


def test_persist_loop_result_persists_tail_without_double_append():
    fake = _FakeLCM()
    s = ChatSession("s", lcm_engine=fake)
    s.add_user_message("hi")  # turn 0 (persisted by add_user_message)
    pre = len(s.get_messages())  # 1

    s.messages.append(ConversationMessage(role="assistant", content=[TextBlock(text="hello back")]))
    s.persist_loop_result(pre)

    assert ("assistant", "hello back", 1) in fake.ingested
    assert sum(1 for r in fake.ingested if r[0] == "assistant") == 1
    assert len(s.get_messages()) == 2


def test_run_agent_persists_assistant_turn(monkeypatch):
    from prometheus.engine import agent_loop as al

    async def fake_run_loop(ctx, messages, *, mode="agent", session_id=None, tool_choice=None):
        messages.append(ConversationMessage(role="assistant", content=[TextBlock(text="answer")]))
        if False:
            yield  # make this an async generator

    _pin_run_loop_signature(fake_run_loop)
    monkeypatch.setattr(
        al, "run_loop",
        register_double("wire_contract.fake_run_loop.persist", replaces="prometheus.engine.agent_loop.run_loop")(fake_run_loop),
    )

    fake = _FakeLCM()
    session = ChatSession("s", lcm_engine=fake)
    session.add_user_message("question")  # turn 0

    bridge = WebSocketBridge(loop_context=object())

    async def cap(ev):
        pass

    bridge.broadcast = cap
    asyncio.run(bridge._run_agent("s", session))
    assert ("assistant", "answer", 1) in fake.ingested


# --------------------------------------------------------------------------- #
# #1 — messages route filters by the rowid cursor + returns watermark + fails loud
# --------------------------------------------------------------------------- #


def _app_with_messages(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    rowids = []
    for i in range(3):
        m = MessagePart(
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i}",
            session_id="s",
            turn_index=i,
            timestamp=100.0 * (i + 1),
        )
        store.insert_message(m)
        rowids.append(m.row_id)

    class _LCM:  # the route only needs `.conversation_store`
        conversation_store = store

    return TestClient(create_app({}, lcm_engine=_LCM())), rowids


def test_messages_full_read_returns_rowid_ids_and_ordinals(tmp_path):
    client, rowids = _app_with_messages(tmp_path)
    body = client.get("/api/sessions/s/messages").json()
    assert [m["message_id"] for m in body["messages"]] == rowids          # durable rowids
    assert [m["ordinal"] for m in body["messages"]] == [0, 1, 2]          # turn_index display
    assert [m["timestamp"] for m in body["messages"]] == [100.0, 200.0, 300.0]
    assert body["watermark"] == rowids[-1]                                # max rowid


def test_messages_since_filters_by_rowid_cursor(tmp_path):
    client, rowids = _app_with_messages(tmp_path)
    body = client.get(f"/api/sessions/s/messages?since={rowids[0]}").json()
    assert [m["message_id"] for m in body["messages"]] == rowids[1:]
    assert body["watermark"] == rowids[-1]


def test_messages_since_caught_up_returns_empty_delta_but_current_watermark(tmp_path):
    client, rowids = _app_with_messages(tmp_path)
    body = client.get(f"/api/sessions/s/messages?since={rowids[-1]}").json()
    assert body["messages"] == []
    assert body["watermark"] == rowids[-1]


def test_messages_since_invalid_is_400_not_silently_ignored(tmp_path):
    client, _ = _app_with_messages(tmp_path)
    resp = client.get("/api/sessions/s/messages?since=notanumber")
    assert resp.status_code == 400
    assert "since" in resp.json()["error"]
