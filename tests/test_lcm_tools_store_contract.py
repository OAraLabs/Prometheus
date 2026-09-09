"""P0.1 — the LCM tools call store methods that exist.

WHY THIS EXISTS
---------------
``lcm_expand``, ``lcm_describe`` and ``lcm_expand_query`` were written against
a store contract that the stores never defined:

- ``summary_store.get(...)`` — the real method is ``get_by_id``.
- ``conv_store.get_by_id(...)`` — the conversation store had ``has_message``
  (a bool) but no id-keyed row read at all.

Each call site wrapped the lookup in a ``hasattr`` guard or a broad
``except``, which converted the contract break into a plausible "not found":
every summary expansion reported its source messages as gone and the model
concluded its own history was missing. The bug was invisible from the inside —
a hasattr guard reads as defensive coding, not as a broken call.

These tests drive the REAL stores and the REAL tools (no doubles on the store
boundary) so the property — "the method the tool calls is a method the store
defines, and it returns the row" — is enforced rather than remembered. A
regression that renames the method again fails here, not silently in
production.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_types import MessagePart, SummaryNode
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import lcm_grep
from prometheus.tools.builtin.lcm_describe import LCMDescribeInput, LCMDescribeTool
from prometheus.tools.builtin.lcm_expand import LCMExpandInput, LCMExpandTool
from prometheus.tools.builtin.lcm_expand_query import (
    LCMExpandQueryInput,
    LCMExpandQueryTool,
)


class _StubProvider:
    """LCMEngine only needs a provider to construct; these tests never compact."""

    name = "stub"


def _engine(tmp_path: Path) -> LCMEngine:
    return LCMEngine(_StubProvider(), db_path=tmp_path / "lcm.db")


# --------------------------------------------------------------------------- #
# The store contract the tools depend on — asserted directly, not via hasattr.
# --------------------------------------------------------------------------- #


def test_conversation_store_defines_get_by_id(tmp_path):
    """The method the tools call must EXIST — no hasattr fallback.

    A hasattr guard around a missing method is exactly how this stayed broken:
    it reads as defensive and silently degrades to "not found". The contract
    is that the store defines it.
    """
    store = LCMConversationStore(tmp_path / "c.db")
    assert hasattr(store, "get_by_id")
    sig = store.get_by_id.__doc__ or ""
    assert sig, "get_by_id must carry its contract, not be a silent stub"


def test_get_by_id_round_trips_a_persisted_message(tmp_path):
    """Insert a row, read it back by its UUID — the expansion the tools do."""
    store = LCMConversationStore(tmp_path / "c.db")
    msg = MessagePart(
        role="user",
        content="the capital of France is Paris",
        session_id="telegram:1",
        turn_index=3,
        token_count=7,
    )
    mid = store.insert_message(msg)

    got = store.get_by_id(mid)
    assert got is not None, "a persisted message must be readable by its id"
    assert got.message_id == mid
    assert got.content == "the capital of France is Paris"
    assert got.turn_index == 3
    assert got.role == "user"


def test_get_by_id_returns_none_for_unknown_id(tmp_path):
    """A genuine absence is None — distinct from the broken-call "not found"."""
    store = LCMConversationStore(tmp_path / "c.db")
    assert store.get_by_id("does-not-exist") is None
    assert store.get_by_id("") is None


def test_summary_store_get_by_id_exists(tmp_path):
    """The summary lookup the tools call (was ``.get``, never defined)."""
    eng = _engine(tmp_path)
    assert hasattr(eng.summary_store, "get_by_id")
    assert not hasattr(eng.summary_store, "get"), (
        "the old broken name must not linger as a stub that masks a rename"
    )


# --------------------------------------------------------------------------- #
# The tools, end to end: a depth-0 summary expands to its REAL source messages.
# --------------------------------------------------------------------------- #


def test_lcm_expand_returns_the_source_messages(tmp_path):
    """The whole point: expansion shows the messages, not "not found"."""
    eng = _engine(tmp_path)
    lcm_grep.set_lcm_engine(eng)
    try:
        conv = eng.conversation_store
        m1 = conv.insert_message(
            MessagePart(role="user", content="what did we decide about auth",
                        session_id="s1", turn_index=0)
        )
        m2 = conv.insert_message(
            MessagePart(role="assistant", content="we chose bearer tokens",
                        session_id="s1", turn_index=1)
        )
        node = SummaryNode(
            parent_ids=[], source_message_ids=[m1, m2],
            summary_text="Decided on bearer-token auth.", depth=0, token_count=12,
        )
        eng.summary_store.insert_summary(node, session_id="s1")

        result = asyncio.run(
            LCMExpandTool().execute(
                LCMExpandInput(summary_id=node.id),
                ToolExecutionContext(cwd=tmp_path),
            )
        )
        assert not result.is_error, result.output
        # The actual source text must be in the expansion — this is the line
        # that was missing for the entire life of the tool.
        assert "what did we decide about auth" in result.output
        assert "we chose bearer tokens" in result.output
        assert "not found" not in result.output
    finally:
        lcm_grep.set_lcm_engine(None)
        eng.close()


def test_lcm_describe_names_a_real_node(tmp_path):
    """describe() resolves the node via get_by_id and reports its metadata."""
    eng = _engine(tmp_path)
    lcm_grep.set_lcm_engine(eng)
    try:
        node = SummaryNode(
            parent_ids=[], source_message_ids=["a", "b"],
            summary_text="Two messages summarised.", depth=1, token_count=20,
        )
        eng.summary_store.insert_summary(node, session_id="s1")

        result = asyncio.run(
            LCMDescribeTool().execute(
                LCMDescribeInput(summary_id=node.id),
                ToolExecutionContext(cwd=tmp_path),
            )
        )
        assert not result.is_error, result.output
        assert node.id in result.output
        assert "depth:              1" in result.output
        assert "2 message(s)" in result.output
    finally:
        lcm_grep.set_lcm_engine(None)
        eng.close()


def test_lcm_expand_query_expands_real_source_messages(tmp_path):
    """The expand_query path that the Phase-4 pilot hit — must show messages."""
    eng = _engine(tmp_path)
    lcm_grep.set_lcm_engine(eng)
    try:
        conv = eng.conversation_store
        mid = conv.insert_message(
            MessagePart(role="user", content="the database schema uses rowids",
                        session_id="s1", turn_index=0)
        )
        node = SummaryNode(
            parent_ids=[], source_message_ids=[mid],
            summary_text="Discussed the database schema and rowids.",
            depth=0, token_count=10,
        )
        eng.summary_store.insert_summary(node, session_id="s1")

        result = asyncio.run(
            LCMExpandQueryTool().execute(
                LCMExpandQueryInput(query="database schema"),
                ToolExecutionContext(cwd=tmp_path),
            )
        )
        # The source message text must surface — not "not found in store".
        assert "the database schema uses rowids" in result.output, result.output
        assert "not found in store" not in result.output
    finally:
        lcm_grep.set_lcm_engine(None)
        eng.close()
