"""feat/session-rehydrate — the two-stores/no-rehydrate gap closes.

After a daemon restart a session's full history was servable to CLIENTS over
REST while the MODEL started blind. These tests drive the real pieces — a
real LCMEngine on a scratch DB (the test_chat_done_row_id pattern), the real
ChatSession persistence watermark, the real tombstone query — and pin the
four hazards the recon ranked: orphaned tool blocks, tombstone resurrection,
double-ingest, and turn_index collisions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import SessionManager
from prometheus.memory.lcm_engine import LCMEngine

SID = "desktop:rehydrate-test"


def _engine(tmp_path: Path) -> LCMEngine:
    return LCMEngine(MagicMock(), db_path=tmp_path / "lcm.db")


def _manager(engine: LCMEngine, *, enabled: bool = True) -> SessionManager:
    mgr = SessionManager()
    mgr.lcm_engine = engine
    mgr.rehydrate_enabled = enabled
    return mgr


def _user(text: str) -> ConversationMessage:
    return ConversationMessage.from_user_text(text)


def _asst(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _tool_pair() -> list[ConversationMessage]:
    return [
        ConversationMessage(role="assistant", content=[
            ToolUseBlock(id="t1", name="bash", input={"command": "ls"}),
        ]),
        ConversationMessage(role="user", content=[
            ToolResultBlock(tool_use_id="t1", content="a.py\nb.py"),
        ]),
    ]


def _seed(engine: LCMEngine, messages: list[ConversationMessage],
          sid: str = SID) -> None:
    """Persist history through the REAL session write path, then discard the
    manager — the discard IS the daemon restart."""
    mgr = _manager(engine, enabled=False)
    session = mgr.get_or_create(sid)
    session.messages = list(messages)
    session.persist_loop_result(0)
    assert session._lcm_persisted_len == len(messages)


# --------------------------------------------------------------------------- #
# The converter (ConversationMessage.from_stored)
# --------------------------------------------------------------------------- #


class TestFromStored:
    def test_tool_blocks_round_trip_losslessly(self) -> None:
        original = ConversationMessage(role="assistant", content=[
            TextBlock(text="running it"),
            ToolUseBlock(id="t9", name="grep", input={"pattern": "x"}),
        ])
        back = ConversationMessage.from_stored(
            role="assistant", content=original.text,
            content_json=original.content_json,
            provenance="user", is_trusted=True,
        )
        assert back.content[0] == original.content[0]
        assert back.content[1] == original.content[1]

    def test_legacy_null_json_degrades_to_flat_text(self) -> None:
        back = ConversationMessage.from_stored(
            role="user", content="plain old row", content_json=None,
        )
        assert [type(b).__name__ for b in back.content] == ["TextBlock"]
        assert back.content[0].text == "plain old row"

    def test_unknown_provenance_falls_back_untrusted(self) -> None:
        back = ConversationMessage.from_stored(
            role="user", content="x", content_json=None,
            provenance="from_the_future", is_trusted=True,
        )
        assert back.provenance == "user"
        assert back.is_trusted is False

    def test_evicted_image_becomes_a_placeholder_not_an_empty_block(
        self, tmp_path: Path
    ) -> None:
        img = ImageBlock(
            data="", media_type="image/png",
            source_path=str(tmp_path / "gone.png"),
            description="a chart of the flail",
        )
        msg = ConversationMessage(role="user", content=[img])
        back = ConversationMessage.from_stored(
            role="user", content="", content_json=msg.content_json,
        )
        assert [type(b).__name__ for b in back.content] == ["TextBlock"]
        assert "a chart of the flail" in back.content[0].text


# --------------------------------------------------------------------------- #
# restore() — the cold-start contract
# --------------------------------------------------------------------------- #


class TestRestore:
    def test_refuses_a_non_cold_session(self, tmp_path: Path) -> None:
        mgr = _manager(_engine(tmp_path))
        session = mgr.get_or_create(SID)
        session.add_user_message("hello")
        with pytest.raises(RuntimeError, match="refusing to clobber"):
            session.restore([_user("old")], next_turn_index=10)


# --------------------------------------------------------------------------- #
# rehydrate_if_cold — end to end against the real store
# --------------------------------------------------------------------------- #


class TestRehydrateIfCold:
    def test_cold_session_gets_its_tail_back(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        history = [_user("plan the sprint"), *_tool_pair(), _asst("done: 2 files")]
        _seed(engine, history)

        mgr = _manager(engine)  # the restart
        restored = mgr.rehydrate_if_cold(SID)
        assert restored == 4
        live = mgr.get_or_create(SID).get_messages()
        assert [m.role for m in live] == ["user", "assistant", "user", "assistant"]
        # Structured blocks came back as blocks, not flattened text.
        assert type(live[1].content[0]).__name__ == "ToolUseBlock"
        assert type(live[2].content[0]).__name__ == "ToolResultBlock"

    def test_tail_starts_at_a_clean_human_turn(self, tmp_path: Path) -> None:
        # History whose window BEGINS mid-tool-exchange: a naive cut would
        # hand the provider an orphaned tool_result — a hard 400 everywhere.
        engine = _engine(tmp_path)
        history = [*_tool_pair(), _user("clean turn"), _asst("answer")]
        _seed(engine, history)

        mgr = _manager(engine)
        assert mgr.rehydrate_if_cold(SID) == 2
        live = mgr.get_or_create(SID).get_messages()
        assert live[0].role == "user"
        assert type(live[0].content[0]).__name__ == "TextBlock"
        assert live[0].text == "clean turn"

    def test_no_clean_turn_restores_nothing(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        _seed(engine, [*_tool_pair(), _asst("only machinery here")])
        mgr = _manager(engine)
        assert mgr.rehydrate_if_cold(SID) == 0
        assert mgr.get_or_create(SID).get_messages() == []

    def test_tombstoned_session_stays_forgotten(self, tmp_path: Path) -> None:
        # DELETE /api/sessions writes a tombstone; every other store reader
        # is tombstone-blind, which is why eligibility goes through
        # list_sessions() — the one tombstone-aware query. A naive rehydrate
        # would turn "forget this chat" into "forget it for one round trip".
        engine = _engine(tmp_path)
        _seed(engine, [_user("secret plans"), _asst("noted")])
        engine.conversation_store.tombstone_session(SID)

        mgr = _manager(engine)
        assert mgr.rehydrate_if_cold(SID) == 0
        assert mgr.get_or_create(SID).get_messages() == []

    def test_disabled_flag_is_a_structural_noop(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        _seed(engine, [_user("hi"), _asst("hello")])
        mgr = _manager(engine, enabled=False)
        assert mgr.rehydrate_if_cold(SID) == 0

    def test_warm_session_is_authoritative(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        _seed(engine, [_user("old"), _asst("old reply")])
        mgr = _manager(engine)
        warm = mgr.get_or_create(SID)
        warm.add_user_message("live and unpersisted context")
        before = list(warm.get_messages())
        assert mgr.rehydrate_if_cold(SID) == 0
        assert warm.get_messages() == before

    def test_next_turn_neither_duplicates_nor_collides(self, tmp_path: Path) -> None:
        # The two quiet corruptions: re-ingesting restored rows (watermark
        # not seeded) and stamping new rows with turn_index values history
        # already used (offset not applied). Assert both on the real DB.
        engine = _engine(tmp_path)
        history = [_user("first"), _asst("second"), _user("third"), _asst("fourth")]
        _seed(engine, history)

        mgr = _manager(engine)
        assert mgr.rehydrate_if_cold(SID) == 4
        session = mgr.get_or_create(SID)
        session.add_user_message("fifth — after the restart")

        con = sqlite3.connect(str(tmp_path / "lcm.db"))
        rows = con.execute(
            "SELECT content, turn_index FROM lcm_messages "
            "WHERE session_id=? ORDER BY rowid", (SID,),
        ).fetchall()
        con.close()
        contents = [r[0] for r in rows]
        assert len(rows) == 5, contents  # 4 seeded + exactly 1 new — no re-ingest
        assert contents.count("first") == 1
        seeded_max = max(r[1] for r in rows[:4])
        assert rows[4][1] == seeded_max + 1, rows  # numbering continues
