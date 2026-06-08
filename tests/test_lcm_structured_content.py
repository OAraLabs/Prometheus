"""LCM structured-content persistence (branch feat/lcm-structured-content).

The bug: ``_persist_to_lcm`` wrote ``content=msg.text``, and ``ConversationMessage.text`` joins
only ``TextBlock``s — so tool_use / tool_result turns persisted to LCM with ``content=''`` (the
empty history bubbles). This adds an ADDITIVE ``content_json`` field/column capturing the full
structured block list losslessly, while KEEPING the flat ``content`` for back-compat.

Every test asserts the SIDE EFFECT:
  * the structured serializer captures tool_use / tool_result (and round-trips);
  * a tool turn now persists NON-EMPTY content_json through the real session path, while the flat
    ``content`` is unchanged (still '');
  * the additive column is back-compat (legacy rows → None) and the migration is idempotent on a
    pre-existing DB;
  * REST + WS expose content_json alongside content.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import ChatSession, SessionManager
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart


def _store(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


class _StoreEngine:
    """Faithful LCM engine stub: threads content_json straight to a real store (mirrors the
    real LCMEngine.ingest_sync additive signature)."""

    def __init__(self, store: LCMConversationStore) -> None:
        self._store = store
        self.conversation_store = store

    def ingest_sync(
        self, session_id, role, content, *, turn_index=0, content_json=None,
        provenance="user", is_trusted=True,
    ):
        self._store.add_message(
            session_id,
            MessagePart(
                role=role,
                content=content,
                session_id=session_id,
                turn_index=turn_index,
                content_json=content_json,
                provenance=provenance,
                is_trusted=is_trusted,
            ),
        )
        return "id"


# --------------------------------------------------------------------------- #
# The serializer: ConversationMessage.content_json
# --------------------------------------------------------------------------- #


def test_content_json_captures_tool_use_block():
    m = ConversationMessage(
        role="assistant",
        content=[ToolUseBlock(id="toolu_x", name="sessions_list", input={"a": 1})],
    )
    assert m.text == ""  # the OLD lossy path drops the whole turn
    assert json.loads(m.content_json) == [
        {"type": "tool_use", "id": "toolu_x", "name": "sessions_list", "input": {"a": 1}}
    ]


def test_content_json_captures_tool_result_block():
    m = ConversationMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id="toolu_x", content="42 sessions", is_error=False)],
    )
    assert m.text == ""
    assert json.loads(m.content_json) == [
        {"type": "tool_result", "tool_use_id": "toolu_x", "content": "42 sessions", "is_error": False}
    ]


def test_content_json_round_trips_back_to_blocks():
    orig = ConversationMessage(
        role="assistant",
        content=[TextBlock(text="hi"), ToolUseBlock(id="toolu_y", name="grep", input={"q": "x"})],
    )
    rebuilt = ConversationMessage(role=orig.role, content=json.loads(orig.content_json))
    assert rebuilt == orig  # discriminated-union validation reconstructs the exact blocks


# --------------------------------------------------------------------------- #
# Store persistence: the content_json column
# --------------------------------------------------------------------------- #


def test_tool_turn_persists_nonempty_content_json(tmp_path):
    store = _store(tmp_path)
    msg = ConversationMessage(
        role="assistant", content=[ToolUseBlock(id="toolu_z", name="web_search", input={"q": "foo"})]
    )
    # exactly what _persist_to_lcm writes for a tool turn: flat content = msg.text (''), structured json present
    store.insert_message(
        MessagePart(role="assistant", content=msg.text, session_id="s", turn_index=1, content_json=msg.content_json)
    )

    got = store.get_all_messages("s")
    assert len(got) == 1
    assert got[0].content == ""  # flat text still lost (back-compat with the old wire `content`)
    assert got[0].content_json is not None
    assert json.loads(got[0].content_json)[0]["name"] == "web_search"  # tool activity RECOVERED


def test_flat_content_preserved_for_text_turns(tmp_path):
    store = _store(tmp_path)
    msg = ConversationMessage.from_user_text("hello there")
    store.insert_message(
        MessagePart(role="user", content=msg.text, session_id="s", turn_index=0, content_json=msg.content_json)
    )
    got = store.get_all_messages("s")[0]
    assert got.content == "hello there"  # unchanged
    assert json.loads(got.content_json) == [{"type": "text", "text": "hello there"}]


def test_legacy_none_content_json_is_back_compat(tmp_path):
    """A row written WITHOUT content_json (the default) reads back as None — no crash, content intact."""
    store = _store(tmp_path)
    store.insert_message(MessagePart(role="assistant", content="plain", session_id="s", turn_index=0))
    got = store.get_all_messages("s")[0]
    assert got.content == "plain"
    assert got.content_json is None


# --------------------------------------------------------------------------- #
# Migration: additive + idempotent on a pre-existing (legacy-schema) DB
# --------------------------------------------------------------------------- #


def test_migration_adds_column_to_legacy_db(tmp_path):
    db = tmp_path / "lcm.db"
    # Hand-build a DB with the PRE-content_json schema + a row.
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE lcm_messages ("
        " id TEXT PRIMARY KEY, session_id TEXT NOT NULL, turn_index INTEGER NOT NULL DEFAULT 0,"
        " role TEXT NOT NULL, content TEXT NOT NULL, token_count INTEGER NOT NULL DEFAULT 0,"
        " timestamp REAL NOT NULL, compacted INTEGER NOT NULL DEFAULT 0);"
    )
    conn.execute(
        "INSERT INTO lcm_messages (id, session_id, turn_index, role, content, timestamp)"
        " VALUES ('m0', 's', 0, 'user', 'old row', 1.0)"
    )
    conn.commit()
    conn.close()

    # Opening with the new store must ALTER content_json in (idempotent) and never touch the old row.
    store = LCMConversationStore(db)
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(lcm_messages)")}
    assert "content_json" in cols
    got = store.get_all_messages("s")[0]
    assert got.content == "old row"  # row preserved (append-only — never rewritten)
    assert got.content_json is None  # legacy row → NULL (unrecoverable, as documented)
    store.close()

    # Reopening an already-migrated DB is a no-op (no double-ALTER crash).
    LCMConversationStore(db).close()


# --------------------------------------------------------------------------- #
# The real fix: session persist path records the tool turn losslessly
# --------------------------------------------------------------------------- #


def test_session_persist_records_tool_turn_content_json(tmp_path):
    store = _store(tmp_path)
    session = ChatSession("s", lcm_engine=_StoreEngine(store))
    session.add_user_message("call the tool")  # turn 0 (text)
    pre = len(session.get_messages())
    # Simulate run_loop appending an assistant tool-call turn in place:
    session.messages.append(
        ConversationMessage(role="assistant", content=[ToolUseBlock(id="toolu_a", name="sessions_list", input={})])
    )
    session.persist_loop_result(pre)

    rows = store.get_all_messages("s")
    assert rows[0].content == "call the tool"  # user text intact
    assert json.loads(rows[0].content_json) == [{"type": "text", "text": "call the tool"}]
    tool_row = rows[-1]
    assert tool_row.content == ""  # flat text still '' (the old loss)
    assert json.loads(tool_row.content_json)[0]["name"] == "sessions_list"  # but the tool call survives


# --------------------------------------------------------------------------- #
# REST + WS additive exposure (must not break existing `content`)
# --------------------------------------------------------------------------- #


def test_rest_exposes_content_json_alongside_content(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    store = _store(tmp_path)
    tu = ConversationMessage(role="assistant", content=[ToolUseBlock(id="toolu_r", name="lcm_grep", input={})])
    store.insert_message(
        MessagePart(role="assistant", content=tu.text, session_id="s", turn_index=0, content_json=tu.content_json)
    )

    class _LCM:
        conversation_store = store

    client = TestClient(create_app({}, lcm_engine=_LCM()))
    m = client.get("/api/sessions/s/messages").json()["messages"][0]
    assert m["content"] == ""  # flat field unchanged (old clients keep working)
    assert json.loads(m["content_json"])[0]["name"] == "lcm_grep"  # new structured field


def test_ws_user_echo_carries_content_json(tmp_path):
    import asyncio

    from prometheus.web.ws_server import WebSocketBridge

    store = _store(tmp_path)
    mgr = SessionManager()
    mgr.lcm_engine = _StoreEngine(store)
    bridge = WebSocketBridge(session_mgr=mgr)
    captured: list[dict] = []

    async def cap(ev):
        captured.append(ev)

    bridge.broadcast = cap
    asyncio.run(bridge.dispatch_user_message("s", "hi there", client_msg_id="c1"))

    echo = next(e for e in captured if e["type"] == "chat_message")["payload"]
    assert echo["content"] == "hi there"  # unchanged
    assert json.loads(echo["content_json"]) == [{"type": "text", "text": "hi there"}]
