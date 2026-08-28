"""GRAFT-MOBILE-BRIDGE 3b — persist_loop_result surfaces the assistant turn's
durable rowid, end to end against a real LCMEngine.

The assistant turn's rowid was never returned to the streaming path; every
client reinvented a ``?since=`` reconciliation to learn it. These prove the
wiring the WS bridge now forwards on ``chat_done``:

  * persist_loop_result returns the durable rowid of the assistant row it wrote;
  * that rowid EQUALS the one a ``?since=`` (messages_after_id) read returns for
    the same row — the spec's acceptance criterion;
  * a tool-using turn returns the LAST assistant row's rowid, not a tool row's;
  * a turn that persists no assistant row returns None (so chat_done omits it).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.session import ChatSession
from prometheus.memory.lcm_engine import LCMEngine


def _engine(tmp_path) -> LCMEngine:
    # A real engine on a scratch DB. The provider is never called (no compaction
    # is triggered by these short turns), so a MagicMock stands in for it.
    return LCMEngine(MagicMock(), db_path=tmp_path / "lcm.db")


def _asst(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def test_persist_returns_the_assistant_row_id(tmp_path):
    eng = _engine(tmp_path)
    s = ChatSession("desktop:rk", lcm_engine=eng)
    s.messages.append(ConversationMessage(role="user", content=[TextBlock(text="hi")]))
    original_len = len(s.messages) - 1  # persist the user row + the reply
    s.messages.append(_asst("hello there"))

    row_id = s.persist_loop_result(original_len)

    assert isinstance(row_id, int) and row_id > 0
    # The returned id is exactly what a ?since= cursor returns for that row —
    # this is what lets the client skip the REST re-read entirely.
    rows = eng._conv_store.messages_after_id(row_id - 1, session_id="desktop:rk")
    assert any(r.row_id == row_id and r.role == "assistant" for r in rows)


def test_returns_the_last_assistant_row_of_a_tool_turn(tmp_path):
    eng = _engine(tmp_path)
    s = ChatSession("desktop:tools", lcm_engine=eng)
    start = len(s.messages)
    # A tool-using turn: assistant(tool_use) → tool_result → assistant(text).
    s.messages.append(
        ConversationMessage(role="assistant",
                            content=[ToolUseBlock(id="t1", name="bash", input={"c": "ls"})])
    )
    s.messages.append(_asst("the final answer"))

    row_id = s.persist_loop_result(start)

    # The rowid is the LAST assistant row (the settled reply), and it is the
    # maximum rowid in the session — the reply comes last. Read through the
    # cursor path (messages_after_id), which is the one that surfaces rowids.
    rows = eng._conv_store.messages_after_id(0, session_id="desktop:tools")
    max_row = max(r.row_id for r in rows)
    assert row_id == max_row
    settled = next(r for r in rows if r.row_id == row_id)
    assert settled.role == "assistant"


def test_returns_none_when_no_assistant_row_persisted(tmp_path):
    eng = _engine(tmp_path)
    s = ChatSession("desktop:useronly", lcm_engine=eng)
    start = len(s.messages)
    s.messages.append(ConversationMessage(role="user", content=[TextBlock(text="just me")]))

    assert s.persist_loop_result(start) is None


def test_returns_none_without_an_engine(tmp_path):
    s = ChatSession("desktop:noengine", lcm_engine=None)
    s.messages.append(_asst("hi"))
    assert s.persist_loop_result(0) is None


def test_rowid_capture_has_no_suspension_point(tmp_path):
    """The shared last_ingested_row_id slot is safe ONLY because writer and
    reader share one synchronous frame (see the invariant comment on the slot).
    Pin the property mechanically: the reader (_persist_to_lcm) is not a
    coroutine, and neither is ingest_sync — so under asyncio nothing can
    interleave between the slot write and the read two statements later. If
    someone makes either async, this fails and points at the fix: thread the
    rowid through the call chain instead of parking it on the engine."""
    import inspect

    assert not inspect.iscoroutinefunction(ChatSession._persist_to_lcm)
    assert not inspect.iscoroutinefunction(LCMEngine.ingest_sync)


def test_two_sessions_on_one_engine_each_get_their_own_rowid(tmp_path):
    """The slot lives on a SHARED engine; per-session correctness must not
    depend on which session persisted last."""
    eng = _engine(tmp_path)
    a = ChatSession("desktop:a", lcm_engine=eng)
    b = ChatSession("desktop:b", lcm_engine=eng)
    a.messages.append(_asst("reply in a"))
    b.messages.append(_asst("reply in b"))

    row_a = a.persist_loop_result(0)
    row_b = b.persist_loop_result(0)

    assert row_a != row_b
    rows_a = eng._conv_store.messages_after_id(0, session_id="desktop:a")
    rows_b = eng._conv_store.messages_after_id(0, session_id="desktop:b")
    assert [r.row_id for r in rows_a] == [row_a]
    assert [r.row_id for r in rows_b] == [row_b]
