"""Tests for Feature 3: Tool Result MicroCompaction."""

import pytest
from unittest.mock import MagicMock

from prometheus.engine.agent_loop import _microcompact_old_results, LoopContext
from prometheus.engine.messages import ConversationMessage, ToolResultBlock, TextBlock


def _make_context(**overrides) -> LoopContext:
    ctx = MagicMock(spec=LoopContext)
    ctx.microcompact_after_turns = overrides.get("microcompact_after_turns", 3)
    ctx.microcompact_keep_chars = overrides.get("microcompact_keep_chars", 200)
    ctx.microcompact_keep_chars_no_lcm = overrides.get("microcompact_keep_chars_no_lcm", 500)
    ctx.lcm_engine = overrides.get("lcm_engine", None)
    return ctx


def _user_msg(text: str = "hi") -> ConversationMessage:
    return ConversationMessage(role="user", content=[TextBlock(text=text)])


def _tool_result_msg(content: str, tool_use_id: str = "t1") -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id=tool_use_id, content=content)],
    )


def _assistant_msg(text: str = "ok") -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


class TestMicroCompaction:
    def test_no_compaction_before_threshold(self):
        ctx = _make_context(microcompact_after_turns=3)
        long_content = "x" * 1000
        msgs = [_tool_result_msg(long_content), _user_msg(), _assistant_msg()]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        # Should not compact — turn < threshold
        assert msgs[0].content[0].content == long_content

    def test_compacts_old_results_after_threshold(self):
        ctx = _make_context(microcompact_after_turns=2, microcompact_keep_chars_no_lcm=100)
        long_content = "first line\n" + "x" * 2000
        msgs = [
            _tool_result_msg(long_content),
            _user_msg("turn 1"),
            _assistant_msg(),
            _user_msg("turn 2"),
            _assistant_msg(),
            _user_msg("turn 3"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=3)
        result = msgs[0].content[0].content
        assert "[microcompacted]" in result
        assert len(result) < len(long_content)

    def test_preserves_recent_results(self):
        ctx = _make_context(microcompact_after_turns=2, microcompact_keep_chars_no_lcm=100)
        long_content = "y" * 2000
        msgs = [
            _user_msg("old"),
            _assistant_msg(),
            _user_msg("recent"),
            _tool_result_msg(long_content),  # This is in the fresh window
        ]
        _microcompact_old_results(ctx, msgs, current_turn=3)
        # Recent result should NOT be compacted
        assert msgs[3].content[0].content == long_content

    def test_skips_error_results(self):
        ctx = _make_context(microcompact_after_turns=1, microcompact_keep_chars_no_lcm=50)
        msgs = [
            ConversationMessage(
                role="user",
                content=[ToolResultBlock(
                    tool_use_id="t1",
                    content="x" * 1000,
                    is_error=True,
                )],
            ),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        # Error results should be preserved
        assert msgs[0].content[0].content == "x" * 1000

    def test_skips_already_pruned(self):
        ctx = _make_context(microcompact_after_turns=1, microcompact_keep_chars_no_lcm=50)
        msgs = [
            _tool_result_msg("[content pruned — context compression]"),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        assert "[content pruned" in msgs[0].content[0].content

    def test_lcm_not_ingested_uses_longer_chars(self):
        lcm = MagicMock()
        lcm.is_ingested = MagicMock(return_value=False)
        ctx = _make_context(
            microcompact_after_turns=1,
            microcompact_keep_chars=50,
            microcompact_keep_chars_no_lcm=300,
            lcm_engine=lcm,
        )
        long_content = "a" * 1000
        msgs = [
            _tool_result_msg(long_content),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        result = msgs[0].content[0].content
        assert "[microcompacted]" in result
        # Should keep ~300 chars (no_lcm), not 50
        assert len(result) > 200

    def test_short_results_not_compacted(self):
        ctx = _make_context(microcompact_after_turns=1, microcompact_keep_chars_no_lcm=500)
        msgs = [
            _tool_result_msg("short"),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        assert msgs[0].content[0].content == "short"


class TestSharedEngineSafety:
    """The shared-state analysis for ``LoopContext.lcm_engine``, as evidence.

    The daemon's web ``LoopContext`` is ONE instance shared by every
    concurrent web/Beacon turn, and the file_mutation_verifier precedent says
    anything stateful on it needs a per-turn key. ``lcm_engine`` needs none,
    for a checkable reason: the loop's only consumer
    (``_microcompact_old_results``) calls ``is_ingested`` — an indexed
    ``SELECT 1`` on a WAL connection opened with ``check_same_thread=False`` —
    and never writes engine state. These tests run against a REAL engine on a
    throwaway db (not a mock) so they fail if ``is_ingested`` ever grows a
    write, cache, or per-call state that would make sharing unsafe.

    Premise worth naming: the ingested-branch tests pass a ``tool_use_id``
    that EQUALS an ingested message id. Live tool results don't have that yet
    (LCM ids are per-message, so ``is_ingested`` returns False for every real
    tool_use_id until the id-mapping follow-up); the tests exercise the branch
    the wiring makes reachable, not a claim about today's live hit-rate.
    """

    def _engine(self, tmp_path):
        from prometheus.memory.lcm_engine import LCMEngine
        from prometheus.memory.lcm_types import CompactionConfig

        return LCMEngine(
            MagicMock(), config=CompactionConfig(), db_path=tmp_path / "lcm.db"
        )

    @staticmethod
    def _rows(engine) -> list[tuple]:
        return [
            tuple(r)
            for r in engine._conv_store._conn.execute(
                "SELECT id, content, compacted FROM lcm_messages ORDER BY rowid"
            )
        ]

    def test_ingested_block_takes_the_short_branch_on_a_real_store(self, tmp_path):
        engine = self._engine(tmp_path)
        mid = engine.ingest_sync("s1", "assistant", "tool output " + "a" * 1000)
        ctx = _make_context(
            microcompact_after_turns=1,
            microcompact_keep_chars=50,
            microcompact_keep_chars_no_lcm=300,
            lcm_engine=engine,
        )
        msgs = [
            _tool_result_msg("b" * 1000, tool_use_id=mid),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        result = msgs[0].content[0].content
        assert "[microcompacted]" in result
        # 50-char branch (LCM has it → trim aggressively), not the 300 one
        assert len(result) < 200

    def test_unknown_id_is_conservative_and_the_read_path_writes_nothing(
        self, tmp_path
    ):
        engine = self._engine(tmp_path)
        engine.ingest_sync("s1", "user", "unrelated persisted turn")
        before = self._rows(engine)
        ctx = _make_context(
            microcompact_after_turns=1,
            microcompact_keep_chars=50,
            microcompact_keep_chars_no_lcm=300,
            lcm_engine=engine,
        )
        msgs = [
            _tool_result_msg("b" * 1000, tool_use_id="toolu_not_a_message_id"),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx, msgs, current_turn=2)
        result = msgs[0].content[0].content
        assert "[microcompacted]" in result
        # today's live shape: real tool_use_ids are never message ids → keep more
        assert len(result) > 200
        # the property that makes SHARING safe: microcompaction is a pure read
        assert self._rows(engine) == before

    def test_two_contexts_sharing_one_engine_do_not_cross_talk(self, tmp_path):
        engine = self._engine(tmp_path)
        mid_a = engine.ingest_sync("beacon-1", "assistant", "a" * 500)
        mid_b = engine.ingest_sync("beacon-2", "assistant", "b" * 500)
        before = self._rows(engine)

        ctx_a = _make_context(
            microcompact_after_turns=1,
            microcompact_keep_chars=50,
            microcompact_keep_chars_no_lcm=300,
            lcm_engine=engine,
        )
        ctx_b = _make_context(
            microcompact_after_turns=1,
            microcompact_keep_chars=250,
            microcompact_keep_chars_no_lcm=600,
            lcm_engine=engine,
        )
        msgs_a = [
            _tool_result_msg("x" * 1000, tool_use_id=mid_a),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        msgs_b = [
            _tool_result_msg("y" * 1000, tool_use_id=mid_b),
            _user_msg("turn 1"),
            _user_msg("turn 2"),
        ]
        _microcompact_old_results(ctx_a, msgs_a, current_turn=2)
        _microcompact_old_results(ctx_b, msgs_b, current_turn=2)

        result_a = msgs_a[0].content[0].content
        result_b = msgs_b[0].content[0].content
        # each context's OWN keep_chars applied — config lives on the context,
        # the engine holds none of it
        assert len(result_a) < 200
        assert 250 < len(result_b) < 450
        assert self._rows(engine) == before
