"""Tests for the LCM (Lossless Context Management) subsystem.

Covers types, FTS5 sanitisation, conversation store, summary store,
assembler, compactor, and the top-level LCMEngine.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import engine.messages first to avoid circular import in providers
from prometheus.engine.messages import ConversationMessage  # noqa: F401
from prometheus.memory.lcm_types import (
    AssemblyResult,
    CompactionConfig,
    CompactionResult,
    MessagePart,
    SummaryNode,
)
from prometheus.memory.lcm_fts5 import sanitize_fts5_query
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_summary_store import LCMSummaryStore
from prometheus.providers.base import ApiTextDeltaEvent


# ---------------------------------------------------------------------------
# Mock ModelProvider (for engine / compactor tests)
# ---------------------------------------------------------------------------


class MockProvider:
    """Minimal provider that yields a single text delta with the canned response."""

    def __init__(self, response: str = "Summary of the conversation.") -> None:
        self._response = response

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text=self._response)


class MockSummarizer:
    """Mock summarizer that bypasses the real model call."""

    def __init__(self, summary_text: str = "This is a mock summary.") -> None:
        self._text = summary_text
        self._consecutive_failures = 0

    async def summarize_messages(self, messages):
        return self._text

    async def summarize_summaries(self, summaries):
        return self._text

    def reset(self) -> None:
        self._consecutive_failures = 0

    @property
    def circuit_open(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# LCM types
# ---------------------------------------------------------------------------


class TestLCMTypes:
    def test_message_part_defaults(self) -> None:
        msg = MessagePart(role="user", content="hello")
        assert msg.message_id  # non-empty UUID hex
        assert len(msg.message_id) == 32  # uuid4().hex length
        assert msg.timestamp > 0
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.session_id == ""
        assert msg.turn_index == 0
        assert msg.token_count == 0

    def test_summary_node_defaults(self) -> None:
        node = SummaryNode(summary_text="A summary")
        assert node.id
        assert len(node.id) == 32
        assert node.parent_ids == []
        assert node.source_message_ids == []
        assert node.summary_text == "A summary"
        assert node.depth == 0
        assert node.token_count == 0
        assert node.is_leaf is True
        assert node.created_at > 0

    def test_compaction_config_defaults(self) -> None:
        cfg = CompactionConfig()
        assert cfg.context_threshold == 18_000
        assert cfg.fresh_tail_count == 32
        assert cfg.summary_model == "default"
        assert cfg.max_summary_depth == 5
        assert cfg.compaction_batch_size == 10


# ---------------------------------------------------------------------------
# FTS5 sanitisation
# ---------------------------------------------------------------------------


class TestFTS5Sanitize:
    def test_fts5_sanitize_basic(self) -> None:
        # FTS5 operators like *, ", (, ), -, +, ^, :, {, }, ~, @, # should be stripped
        result = sanitize_fts5_query('hello "world" AND (test*)')
        assert '"' not in result
        assert "*" not in result
        assert "(" not in result
        assert ")" not in result
        assert "hello" in result
        assert "world" in result
        assert "test" in result

    def test_fts5_sanitize_empty(self) -> None:
        assert sanitize_fts5_query("") == ""
        # All-punctuation yields empty after stripping
        assert sanitize_fts5_query('***""()') == ""
        assert sanitize_fts5_query("   ") == ""


# ---------------------------------------------------------------------------
# LCMConversationStore
# ---------------------------------------------------------------------------


class TestConversationStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> LCMConversationStore:
        db = tmp_path / "test_lcm.db"
        s = LCMConversationStore(db_path=db)
        yield s
        s.close()

    def _make_msg(
        self, role: str, content: str, session: str = "s1", turn: int = 0
    ) -> MessagePart:
        return MessagePart(
            role=role,
            content=content,
            session_id=session,
            turn_index=turn,
            token_count=len(content) // 4,
        )

    def test_conversation_store_insert_and_get(self, store: LCMConversationStore) -> None:
        m1 = self._make_msg("user", "hello world", turn=1)
        m2 = self._make_msg("assistant", "hi there", turn=2)
        store.insert_message(m1)
        store.insert_message(m2)

        messages = store.get_messages("s1")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_conversation_store_fresh_tail(self, store: LCMConversationStore) -> None:
        for i in range(10):
            store.insert_message(self._make_msg("user", f"msg {i}", turn=i))

        tail = store.get_fresh_tail("s1", count=3)
        assert len(tail) == 3
        # Should be the last 3, in chronological order
        assert tail[0].content == "msg 7"
        assert tail[1].content == "msg 8"
        assert tail[2].content == "msg 9"

    def test_conversation_store_mark_compacted(self, store: LCMConversationStore) -> None:
        msgs = []
        for i in range(5):
            m = self._make_msg("user", f"msg {i}", turn=i)
            store.insert_message(m)
            msgs.append(m)

        # Mark first 3 as compacted
        ids_to_compact = [m.message_id for m in msgs[:3]]
        affected = store.mark_compacted(ids_to_compact)
        assert affected == 3

        # Fresh tail (uncompacted) should now be only the last 2
        uncompacted_count = store.count_uncompacted("s1")
        assert uncompacted_count == 2

    def test_conversation_store_search(self, store: LCMConversationStore) -> None:
        store.insert_message(self._make_msg("user", "the quick brown fox", turn=1))
        store.insert_message(self._make_msg("user", "lazy dog sleeps", turn=2))
        store.insert_message(self._make_msg("user", "hello world", turn=3))

        results = store.search("fox")
        assert len(results) >= 1
        assert any("fox" in r.content for r in results)

        # Search for something that doesn't exist
        results = store.search("nonexistenttermxyz")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# LCMSummaryStore
# ---------------------------------------------------------------------------


class TestSummaryStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> LCMSummaryStore:
        db = tmp_path / "test_lcm.db"
        s = LCMSummaryStore(db_path=db)
        yield s
        s.close()

    def test_summary_store_insert_and_get(self, store: LCMSummaryStore) -> None:
        node = SummaryNode(
            summary_text="User discussed deployment strategy",
            depth=0,
            token_count=10,
        )
        nid = store.insert_summary(node)
        assert nid == node.id

        retrieved = store.get_by_id(nid)
        assert retrieved is not None
        assert retrieved.summary_text == "User discussed deployment strategy"
        assert retrieved.is_leaf is True

    def test_summary_store_leaf_tracking(self, store: LCMSummaryStore) -> None:
        parent = SummaryNode(summary_text="Parent summary", depth=0)
        store.insert_summary(parent)

        # Parent should be a leaf initially
        assert store.get_by_id(parent.id).is_leaf is True

        # Insert child that references parent
        child = SummaryNode(
            summary_text="Child summary",
            parent_ids=[parent.id],
            depth=1,
        )
        store.insert_summary(child)

        # Parent should no longer be a leaf
        assert store.get_by_id(parent.id).is_leaf is False
        # Child should be a leaf
        assert store.get_by_id(child.id).is_leaf is True

    def test_summary_store_get_by_depth(self, store: LCMSummaryStore) -> None:
        for depth in (0, 0, 1, 1, 2):
            store.insert_summary(SummaryNode(summary_text=f"depth-{depth}", depth=depth))

        depth_0 = store.get_by_depth(0)
        assert len(depth_0) == 2
        depth_1 = store.get_by_depth(1)
        assert len(depth_1) == 2
        depth_2 = store.get_by_depth(2)
        assert len(depth_2) == 1

    def test_summary_store_search(self, store: LCMSummaryStore) -> None:
        store.insert_summary(SummaryNode(summary_text="deployment pipeline configuration"))
        store.insert_summary(SummaryNode(summary_text="user authentication flow"))
        store.insert_summary(SummaryNode(summary_text="database migration steps"))

        results = store.search("deployment")
        assert len(results) >= 1
        assert any("deployment" in r.summary_text for r in results)

        results = store.search("nonexistenttermxyz")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# LCMAssembler (fresh-only and with-summaries)
# ---------------------------------------------------------------------------


class TestAssembler:
    """Test the assembler using the real conversation and summary stores.

    The assembler calls methods on the stores that may differ from the raw
    LCMConversationStore/LCMSummaryStore API (get_uncompacted_messages,
    count_all, get_all_messages, get_leaf_summaries).  We patch those on
    the store instances so the assembler can run against the real SQLite
    stores.
    """

    @pytest.fixture()
    def stores(self, tmp_path: Path):
        db = tmp_path / "asm.db"
        conv = LCMConversationStore(db_path=db)
        sums = LCMSummaryStore(db_path=db)
        yield conv, sums
        conv.close()
        sums.close()

    def _insert_messages(self, conv: LCMConversationStore, count: int, session: str = "s1"):
        msgs = []
        for i in range(count):
            m = MessagePart(
                role="user" if i % 2 == 0 else "assistant",
                content=f"message number {i} with some text for tokens",
                session_id=session,
                turn_index=i,
                token_count=10,
            )
            conv.insert_message(m)
            msgs.append(m)
        return msgs

    def test_assembler_fresh_only(self, stores) -> None:
        conv, sums = stores
        msgs = self._insert_messages(conv, 5)

        # Patch the store methods that the assembler expects
        conv.get_uncompacted_messages = lambda sid: conv.get_messages(sid)
        conv.count_all = lambda sid: len(conv.get_messages(sid))
        conv.get_all_messages = lambda sid: conv.get_messages(sid)
        sums.get_leaf_summaries = lambda sid: []

        from prometheus.memory.lcm_assembler import LCMAssembler

        config = CompactionConfig(fresh_tail_count=32)
        assembler = LCMAssembler(conv, sums, config)
        result = assembler.assemble("s1", token_budget=50000)

        assert isinstance(result, AssemblyResult)
        assert len(result.fresh_messages) == 5
        assert len(result.summaries) == 0

    def test_assembler_with_summaries(self, stores) -> None:
        conv, sums = stores
        self._insert_messages(conv, 5)

        summary_node = SummaryNode(
            summary_text="Earlier the user discussed architecture",
            depth=0,
            token_count=10,
            is_leaf=True,
        )
        sums.insert_summary(summary_node)

        # Patch store methods
        conv.get_uncompacted_messages = lambda sid: conv.get_messages(sid)
        conv.count_all = lambda sid: len(conv.get_messages(sid))
        conv.get_all_messages = lambda sid: conv.get_messages(sid)
        sums.get_leaf_summaries = lambda sid: [summary_node]

        from prometheus.memory.lcm_assembler import LCMAssembler

        config = CompactionConfig(fresh_tail_count=32)
        assembler = LCMAssembler(conv, sums, config)
        result = assembler.assemble("s1", token_budget=50000)

        assert len(result.summaries) == 1
        assert result.summaries[0].summary_text == "Earlier the user discussed architecture"
        assert len(result.fresh_messages) == 5

        # Verify preamble formatting
        preamble = assembler.format_summary_preamble(result.summaries)
        assert "[Earlier conversation context (compressed)]" in preamble
        assert "architecture" in preamble


# ---------------------------------------------------------------------------
# LCMCompactor.should_compact
# ---------------------------------------------------------------------------


class TestCompactor:
    def test_compactor_should_compact(self, tmp_path: Path) -> None:
        db = tmp_path / "compact.db"
        conv = LCMConversationStore(db_path=db)
        sums = LCMSummaryStore(db_path=db)

        config = CompactionConfig(fresh_tail_count=5, compaction_batch_size=3)

        # Insert messages -- need more than fresh_tail_count + compaction_batch_size
        for i in range(10):
            conv.insert_message(
                MessagePart(
                    role="user",
                    content=f"message {i}",
                    session_id="s1",
                    turn_index=i,
                    token_count=5,
                )
            )

        from prometheus.memory.lcm_compaction import LCMCompactor
        from prometheus.memory.lcm_summarize import LCMSummarizer

        summarizer = LCMSummarizer(MockProvider())
        compactor = LCMCompactor(conv, sums, summarizer, config)

        # 10 uncompacted > 5 (tail) + 3 (batch) = 8, so should compact
        assert compactor.should_compact("s1") is True

        conv.close()
        sums.close()

    def test_compactor_should_not_compact_below_threshold(self, tmp_path: Path) -> None:
        db = tmp_path / "compact2.db"
        conv = LCMConversationStore(db_path=db)
        sums = LCMSummaryStore(db_path=db)

        config = CompactionConfig(fresh_tail_count=5, compaction_batch_size=3)

        # Only 5 messages -- not enough to exceed threshold
        for i in range(5):
            conv.insert_message(
                MessagePart(
                    role="user",
                    content=f"message {i}",
                    session_id="s1",
                    turn_index=i,
                    token_count=5,
                )
            )

        from prometheus.memory.lcm_compaction import LCMCompactor
        from prometheus.memory.lcm_summarize import LCMSummarizer

        summarizer = LCMSummarizer(MockProvider())
        compactor = LCMCompactor(conv, sums, summarizer, config)

        # 5 uncompacted <= 5 + 3 = 8, so should NOT compact
        assert compactor.should_compact("s1") is False

        conv.close()
        sums.close()


# ---------------------------------------------------------------------------
# LCMEngine (integration)
# ---------------------------------------------------------------------------


class TestLCMEngine:
    async def test_engine_ingest_and_assemble(self, tmp_path: Path) -> None:
        db = tmp_path / "engine.db"
        provider = MockProvider("Summary of conversation so far.")
        config = CompactionConfig(fresh_tail_count=5, compaction_batch_size=3)

        from prometheus.memory.lcm_engine import LCMEngine

        engine = LCMEngine(provider, config=config, db_path=db)

        # Patch the store methods the assembler/compactor expect
        engine._conv_store.get_uncompacted_messages = lambda sid: engine._conv_store.get_messages(sid)
        engine._conv_store.count_all = lambda sid: len(engine._conv_store.get_messages(sid))
        engine._conv_store.get_all_messages = lambda sid: engine._conv_store.get_messages(sid)
        engine._sum_store.get_leaf_summaries = lambda sid: []
        # Patch add_message to use insert_message
        original_insert = engine._conv_store.insert_message

        def add_message_shim(session_id, msg):
            msg.session_id = session_id
            return original_insert(msg)

        engine._conv_store.add_message = add_message_shim

        try:
            mid = await engine.ingest("s1", "user", "Hello!", turn_index=0)
            assert mid  # non-empty id
            mid2 = await engine.ingest("s1", "assistant", "Hi there!", turn_index=1)
            assert mid2

            result = engine.assemble("s1", token_budget=50000)
            assert isinstance(result, AssemblyResult)
            assert len(result.fresh_messages) == 2
        finally:
            engine.close()

    async def test_engine_compact(self, tmp_path: Path) -> None:
        db = tmp_path / "engine_compact.db"
        provider = MockProvider("This is a summary of the batch.")
        config = CompactionConfig(
            fresh_tail_count=3,
            compaction_batch_size=2,
        )

        from prometheus.memory.lcm_engine import LCMEngine

        engine = LCMEngine(provider, config=config, db_path=db)

        # Replace the summarizer with a mock that bypasses ConversationMessage validation
        engine._summarizer = MockSummarizer("Mock summary of compacted messages.")
        engine._compactor._summarizer = engine._summarizer

        # Patch store methods
        original_insert = engine._conv_store.insert_message

        def add_message_shim(session_id, msg):
            msg.session_id = session_id
            return original_insert(msg)

        engine._conv_store.add_message = add_message_shim

        def get_uncompacted(sid):
            return [
                m
                for m in engine._conv_store.get_messages(sid)
                if not _is_compacted(engine._conv_store, m.message_id)
            ]

        def _is_compacted(store, mid):
            row = store._conn.execute(
                "SELECT compacted FROM lcm_messages WHERE id = ?", (mid,)
            ).fetchone()
            return bool(row and row["compacted"])

        engine._conv_store.get_uncompacted_messages = get_uncompacted
        engine._conv_store.count_all = lambda sid: len(engine._conv_store.get_messages(sid))
        engine._conv_store.get_all_messages = lambda sid: engine._conv_store.get_messages(sid)

        # Track summaries per session
        _session_summaries: dict[str, list] = {}

        original_sum_insert = engine._sum_store.insert_summary

        def add_summary_shim(session_id, node):
            _session_summaries.setdefault(session_id, []).append(node)
            return original_sum_insert(node)

        engine._sum_store.add_summary = add_summary_shim
        engine._sum_store.get_leaf_summaries = lambda sid: [
            n for n in _session_summaries.get(sid, []) if n.is_leaf
        ]
        engine._sum_store.get_leaf_summaries_at_depth = (
            lambda sid, d: [n for n in _session_summaries.get(sid, []) if n.depth == d and n.is_leaf]
        )
        engine._sum_store.get_max_depth = lambda sid: max(
            (n.depth for n in _session_summaries.get(sid, [])), default=0
        )
        engine._sum_store.mark_non_leaf = lambda ids: None
        engine._sum_store.count_all = lambda sid: len(_session_summaries.get(sid, []))

        try:
            # Ingest enough messages to trigger compaction
            for i in range(10):
                await engine.ingest("s1", "user" if i % 2 == 0 else "assistant", f"msg {i}", turn_index=i)

            result = await engine.compact("s1")
            assert isinstance(result, CompactionResult)
            # With 10 msgs, fresh_tail=3, batch_size=2, we compact 7 msgs in ~3-4 batches
            assert result.summaries_created > 0
            assert result.messages_compacted > 0
        finally:
            engine.close()
