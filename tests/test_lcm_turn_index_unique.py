"""``turn_index`` is unique per session: the numbering, the guard, the migration.

docs/audits/LCM-TURN-INDEX-DUPLICATES.md measured 7,468 rows in production that
re-used an index an older row of the same session already held, so
``ORDER BY turn_index`` zipped two conversations together in front of the LCM
summarizer. Each producer (P1–P6 in the report) and each reader it broke has a
test here. The producers and readers were reproduced first, as failing tests;
they now pass. The live-prompt control passed before the fix and still does.

* ``TestProducers``: every path that restarted the numbering now continues
  above the session's durable maximum.
* ``TestGuard``: the store refuses a repeated index without ever deleting a row.
* ``TestMigration``: the one-time repair of the rows already written.
* ``TestImpact``: after the repair, the compactor, the assembler and the
  golden-trace exporter read a session in the order the model saw it.
* ``TestControl``: the live prompt was never affected, and still is not.

All synthetic: a real LCMEngine on a tmp_path DB and the real ChatSession /
SessionManager write paths. A "daemon restart" is discarding the manager. Rows
that PRE-FIX code wrote are seeded straight into an un-migrated store, which is
exactly what those builds left behind.
"""

from __future__ import annotations

import ast
import asyncio
import os
import random
import sqlite3
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import ChatSession, SessionManager
from prometheus.memory import lcm_turn_index_migration as migration
from prometheus.memory.lcm_assembler import LCMAssembler
from prometheus.memory.lcm_compaction import LCMCompactor
from prometheus.memory.lcm_conversation_store import (
    LEGACY_TURN_INDEX_INDEX,
    TURN_INDEX_UNIQUE_INDEX,
    LCMConversationStore,
)
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.memory.lcm_summarize import LCMSummarizer
from prometheus.memory.lcm_summary_store import LCMSummaryStore
from prometheus.memory.lcm_turn_index_migration import (
    migrate_turn_index,
    plan_renumbering,
    run_turn_index_migration,
)
from prometheus.memory.lcm_types import CompactionConfig, MessagePart
from prometheus.sentinel.golden_trace_exporter import lcm_context_resolver

SID = "telegram:unique"
ROOT = Path(__file__).resolve().parent.parent


def _engine(tmp_path: Path) -> LCMEngine:
    return LCMEngine(MagicMock(), db_path=tmp_path / "lcm.db")


def _manager(engine: LCMEngine, *, rehydrate: bool = True) -> SessionManager:
    mgr = SessionManager()
    mgr.lcm_engine = engine
    mgr.rehydrate_enabled = rehydrate
    return mgr


def _user(text: str) -> ConversationMessage:
    return ConversationMessage.from_user_text(text)


def _asst(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _turns(n: int, tag: str) -> list[ConversationMessage]:
    out: list[ConversationMessage] = []
    for i in range(n):
        out += [_user(f"{tag} q{i}"), _asst(f"{tag} a{i}")]
    return out


def _seed(engine: LCMEngine, messages: list[ConversationMessage]) -> None:
    """One daemon lifetime writing ``messages`` through the real persist path."""
    session = _manager(engine, rehydrate=False).get_or_create(SID)
    session.messages = list(messages)
    session.persist_loop_result(0)


def _legacy(db: Path, rows: list[tuple[int, str]], sid: str = SID, t0: float = 1_000.0) -> None:
    """Rows as PRE-FIX code wrote them: explicit indices, repeats and all.

    Only possible on a store that has not been migrated (no UNIQUE index yet),
    which is every lcm.db written before this change.
    """
    store = LCMConversationStore(db)
    for k, (turn_index, content) in enumerate(rows):
        store.insert_message(MessagePart(
            session_id=sid, turn_index=turn_index,
            role="user" if turn_index % 2 == 0 else "assistant",
            content=content, timestamp=t0 + k,
        ))
    store.close()


def _dups(db: Path, sid: str = SID) -> list[tuple[int, int]]:
    con = sqlite3.connect(db)
    try:
        return con.execute(
            "SELECT turn_index, COUNT(*) FROM lcm_messages WHERE session_id = ?"
            " GROUP BY turn_index HAVING COUNT(*) > 1 ORDER BY turn_index",
            (sid,),
        ).fetchall()
    finally:
        con.close()


def _prompt_order(db: Path, sid: str = SID) -> list[tuple[int, str]]:
    """(turn_index, content) in the order every LCM reader now uses."""
    con = sqlite3.connect(db)
    try:
        return con.execute(
            "SELECT turn_index, content FROM lcm_messages WHERE session_id = ?"
            " ORDER BY turn_index, rowid",
            (sid,),
        ).fetchall()
    finally:
        con.close()


def _turn_index_of(db: Path, content: str) -> int:
    con = sqlite3.connect(db)
    try:
        return con.execute(
            "SELECT turn_index FROM lcm_messages WHERE content = ?", (content,)
        ).fetchone()[0]
    finally:
        con.close()


def _schema(db: Path) -> dict[str, list[str]]:
    """Every table in the FILE (never temp) and its columns."""
    con = sqlite3.connect(db)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        return {t: [c[1] for c in con.execute(f'PRAGMA table_info("{t}")')] for t in tables}
    finally:
        con.close()


def _indexes(db: Path) -> dict[str, bool]:
    """Index name -> unique, for the (session_id, turn_index) key."""
    con = sqlite3.connect(db)
    try:
        return {
            r[1]: bool(r[2]) for r in con.execute("PRAGMA index_list(lcm_messages)")
            if r[1] in (TURN_INDEX_UNIQUE_INDEX, LEGACY_TURN_INDEX_INDEX)
        }
    finally:
        con.close()


def _user_version(db: Path) -> int:
    con = sqlite3.connect(db)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def _all_rows(db: Path) -> list[tuple]:
    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT rowid, * FROM lcm_messages ORDER BY rowid").fetchall()
    finally:
        con.close()


class _Telemetry:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record_silent_failure(self, subsystem, operation, exc, context=None) -> None:
        self.records.append({"subsystem": subsystem, "operation": operation,
                             "exc": exc, "context": context})


@pytest.fixture
def telemetry(monkeypatch: pytest.MonkeyPatch) -> _Telemetry:
    fake = _Telemetry()
    monkeypatch.setattr("prometheus.telemetry.tracker.get_telemetry_handle", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# §3 of the report — every path that restarted the numbering. The invariant:
# no two rows of one session share a turn_index, and new rows land after old.
# --------------------------------------------------------------------------- #


class TestProducers:
    def test_reset_continues_above_the_cleared_rows(self, tmp_path: Path) -> None:
        """P4. /reset and /clear (Telegram, WS slash, Slack, Discord) call
        ChatSession.clear(): positions go back to 0, the durable numbering must not."""
        engine = _engine(tmp_path)
        session = _manager(engine).get_or_create(SID)
        session.add_user_message("before reset")
        session.add_result_messages([*session.messages, _asst("reply")], 1)

        session.clear()                       # /reset
        assert session.add_user_message("after reset") == 2

        assert _dups(tmp_path / "lcm.db") == []
        assert _prompt_order(tmp_path / "lcm.db") == [
            (0, "before reset"), (1, "reply"), (2, "after reset")]

    def test_restart_where_rehydrate_declines_continues_above_history(
        self, tmp_path: Path
    ) -> None:
        """P1. rehydrate_if_cold restores nothing when its window (40 rows,
        newest-first 8,000-token budget) holds no clean human turn: here the last
        turn's tool output alone exceeds the budget. The session starts cold."""
        engine = _engine(tmp_path)
        big = "x" * 40_000                    # ~10k tokens: fills the budget alone
        _seed(engine, [
            *_turns(3, "old"),
            _user("run the report"),
            ConversationMessage(role="assistant", content=[
                ToolUseBlock(id="t1", name="bash", input={"command": "report"})]),
            ConversationMessage(role="user", content=[
                ToolResultBlock(tool_use_id="t1", content=big)]),
            _asst("done"),
        ])                                              # 0..9

        mgr = _manager(engine, rehydrate=True)          # daemon restart
        assert mgr.rehydrate_if_cold(SID) == 0          # declined, as in production
        assert mgr.get_or_create(SID).add_user_message("next day") == 10

        assert _dups(tmp_path / "lcm.db") == []
        assert _turn_index_of(tmp_path / "lcm.db", "next day") == 10

    def test_restart_through_a_path_that_never_rehydrates(self, tmp_path: Path) -> None:
        """P2. inject_turn (task completions), POST /api/chat, Slack and Discord
        call get_or_create without rehydrate_if_cold; once that first write lands
        the session is warm, so a later human message never rehydrates either."""
        engine = _engine(tmp_path)
        _seed(engine, _turns(5, "old"))                 # 0..9

        mgr = _manager(engine, rehydrate=True)          # daemon restart
        session = mgr.get_or_create(SID)                # inject_turn's lookup
        ordinal = session.add_user_message(
            "task finished", provenance="task_supervisor", is_trusted=False)
        assert mgr.rehydrate_if_cold(SID) == 0          # warm now: never restores

        assert ordinal == 10
        assert _dups(tmp_path / "lcm.db") == []

    def test_rehydrate_continues_from_the_session_max_not_the_window(
        self, tmp_path: Path
    ) -> None:
        """P3, the largest live producer. The rehydrate window (<= 40 rows) of a
        session whose newest run restarted from 0 cannot see the older, higher
        run. next_turn_index now comes from the store, not the window."""
        engine = _engine(tmp_path)
        _seed(engine, _turns(50, "A"))                  # lifetime A: 0..99
        # Lifetime B, as a pre-fix daemon wrote it after a cold restart: 0..45 again.
        _legacy(tmp_path / "lcm.db", [(i, f"B{i}") for i in range(46)], t0=5_000.0)

        mgr = _manager(engine, rehydrate=True)          # restart; rehydrate works
        assert mgr.rehydrate_if_cold(SID) > 0
        mgr.get_or_create(SID).add_user_message("lifetime C")

        assert _turn_index_of(tmp_path / "lcm.db", "lifetime C") == 100

    def test_rollback_then_retry_takes_a_new_index(self, tmp_path: Path) -> None:
        """P5. Telegram/Slack/Discord: the user row is durable before the turn
        runs; a failed turn calls rollback_last() and the retry is persisted at
        the same POSITION. It must not get the same INDEX."""
        engine = _engine(tmp_path)
        session = _manager(engine).get_or_create(SID)
        session.add_user_message("q")
        session.rollback_last()                         # the turn raised
        assert session.add_user_message("q") == 1       # the user sends it again

        assert _dups(tmp_path / "lcm.db") == []
        assert _prompt_order(tmp_path / "lcm.db") == [(0, "q"), (1, "q")]

    def test_ws_failure_numbers_the_next_turn_above_a_dropped_durable_row(
        self, tmp_path: Path
    ) -> None:
        """P6. WS path: a message sent mid-turn is persisted at once, ahead of the
        turn's tail. If the turn fails, rollback_to() drops it from memory, but it
        stays durable (dropping it is a separate open issue); the next turn must be
        numbered above it."""
        engine = _engine(tmp_path)
        session = _manager(engine).get_or_create(SID)
        session.add_user_message("first")               # 0, durable
        original_len = len(session.messages)
        session.messages.append(_asst("partial"))       # loop appends in place (1)
        session.add_user_message("mid-turn steer")      # 2, durable ahead-set row
        session.rollback_to(original_len)               # turn failed
        assert session.add_user_message("next") == 3
        pre = len(session.messages)
        session.messages.append(_asst("answer"))
        session.persist_loop_result(pre)

        assert _dups(tmp_path / "lcm.db") == []
        assert _prompt_order(tmp_path / "lcm.db") == [
            (0, "first"), (2, "mid-turn steer"), (3, "next"), (4, "answer")]

    def test_a_mid_turn_message_keeps_its_prompt_position(self, tmp_path: Path) -> None:
        """The anchor runs only where the numbering (re)starts. A message sent
        mid-turn is persisted before the turn's tail; anchoring the tail to the
        store's maximum would number it AFTER that message and break the order
        the model saw (660 such rows in production)."""
        engine = _engine(tmp_path)
        session = _manager(engine).get_or_create(SID)
        session.add_user_message("q0")                  # 0
        pre = len(session.messages)
        session.messages.append(_asst("a1"))            # the loop, in place
        session.messages.append(_asst("a2"))
        session.add_user_message("mid-turn")            # 3, persisted now
        session.messages.append(_asst("a4"))
        session.persist_loop_result(pre)                # 1, 2, 4

        assert _prompt_order(tmp_path / "lcm.db") == [
            (0, "q0"), (1, "a1"), (2, "a2"), (3, "mid-turn"), (4, "a4")]

    def test_the_anchor_leaves_room_for_an_unwritten_tail(self, tmp_path: Path) -> None:
        """If the first write after a (re)start is a message persisted ahead of
        rows still waiting below it, those rows must clear the store's maximum
        too, in prompt order, instead of landing on indices already taken."""
        engine = _engine(tmp_path)
        _seed(engine, _turns(5, "old"))                 # 0..9
        session = _manager(engine, rehydrate=False).get_or_create(SID)
        session.messages = [_user("q"), _asst("a")]     # not persisted yet
        assert session.add_user_message("mid-turn") == 12
        session.persist_loop_result(0)

        assert _dups(tmp_path / "lcm.db") == []
        assert _prompt_order(tmp_path / "lcm.db")[-3:] == [
            (10, "q"), (11, "a"), (12, "mid-turn")]

    def test_add_user_message_returns_the_durable_ordinal(self, tmp_path: Path) -> None:
        """After a rehydrate the list position and the durable index differ; the
        wire ``ordinal`` must be the durable one."""
        engine = _engine(tmp_path)
        _seed(engine, _turns(30, "old"))                # 0..59
        mgr = _manager(engine, rehydrate=True)
        restored = mgr.rehydrate_if_cold(SID)
        assert 0 < restored < 60

        ordinal = mgr.get_or_create(SID).add_user_message("hello")

        assert ordinal == 60 == _turn_index_of(tmp_path / "lcm.db", "hello")

    def test_no_store_means_no_anchor(self) -> None:
        """An engine without a conversation store (test fakes, an ephemeral
        setup) numbers by position, exactly as before."""

        class _Engine:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def ingest_sync(self, **kw) -> str:
                self.calls.append(kw["turn_index"])
                return "id"

        eng = _Engine()
        session = ChatSession("s", lcm_engine=eng)
        session.add_user_message("a")
        session.add_user_message("b")
        assert eng.calls == [0, 1]

        mock = MagicMock()
        session = ChatSession("s", lcm_engine=mock)
        session.add_user_message("a")
        session.add_user_message("b")
        assert [c.kwargs["turn_index"] for c in mock.ingest_sync.call_args_list] == [0, 1]

    def test_cli_rows_are_appended_by_the_store(self, tmp_path: Path) -> None:
        """P8. The CLI REPL gave the user and the assistant row one shared index,
        restarted at 0 per run, and re-used it when a failed turn was retried. It
        now passes no index; the store appends each row atomically."""
        engine = _engine(tmp_path)
        assert migrate_turn_index(tmp_path / "lcm.db").status == "indexed"

        async def _repl_run() -> None:
            await engine.ingest("cli-1", "user", "question")      # failed turn
            await engine.ingest("cli-1", "user", "question")      # retried
            await engine.ingest("cli-1", "assistant", "answer")

        asyncio.run(_repl_run())
        asyncio.run(_repl_run())                                    # a second run
        assert _prompt_order(tmp_path / "lcm.db", "cli-1") == [
            (0, "question"), (1, "question"), (2, "answer"),
            (3, "question"), (4, "question"), (5, "answer")]

    def test_the_repl_passes_no_turn_index(self) -> None:
        tree = ast.parse((ROOT / "src" / "prometheus" / "__main__.py").read_text())
        ingests = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "ingest"
        ]
        assert ingests, "the REPL's ingest calls moved; update this test"
        assert all(kw.arg != "turn_index" for call in ingests for kw in call.keywords)


# --------------------------------------------------------------------------- #
# §5.2 — the guard on a migrated store
# --------------------------------------------------------------------------- #


class TestGuard:
    def test_a_taken_index_is_moved_never_overwritten(
        self, tmp_path: Path, telemetry: _Telemetry
    ) -> None:
        db = tmp_path / "lcm.db"
        store = LCMConversationStore(db)
        assert migrate_turn_index(db).status == "indexed"
        for i in range(2):
            store.insert_message(MessagePart(role="user", content=f"m{i}",
                                             session_id=SID, turn_index=i))
        late = MessagePart(role="user", content="late", session_id=SID, turn_index=0)
        store.insert_message(late)

        assert late.turn_index == 2
        assert _prompt_order(db) == [(0, "m0"), (1, "m1"), (2, "late")]
        [record] = telemetry.records
        assert (record["subsystem"], record["operation"]) == ("lcm", "turn_index_collision")
        assert record["context"]["requested_turn_index"] == 0
        assert record["context"]["stored_turn_index"] == 2

    def test_a_stored_id_is_left_as_it_is(self, tmp_path: Path) -> None:
        db = tmp_path / "lcm.db"
        store = LCMConversationStore(db)
        first = MessagePart(role="user", content="original words", session_id=SID,
                            turn_index=0, message_id="m-1")
        store.insert_message(first)
        again = MessagePart(role="user", content="replacement words", session_id=SID,
                            turn_index=5, message_id="m-1")
        assert store.insert_message(again) == "m-1"

        assert _prompt_order(db) == [(0, "original words")]
        assert again.row_id == first.row_id and again.turn_index == 0
        assert [m.content for m in store.search("original")] == ["original words"]
        assert store.search("replacement") == []

    def test_no_insert_ever_deletes_a_row(self, tmp_path: Path) -> None:
        db = tmp_path / "lcm.db"
        store = LCMConversationStore(db)
        assert migrate_turn_index(db).status == "indexed"
        for i in range(20):
            store.insert_message(MessagePart(role="user", content=f"c{i}",
                                             session_id=SID, turn_index=i % 3))
        assert sorted(c for _, c in _prompt_order(db)) == sorted(f"c{i}" for i in range(20))
        assert _dups(db) == []

    def test_the_session_moves_its_numbering_after_a_reassign(
        self, tmp_path: Path, telemetry: _Telemetry
    ) -> None:
        """If a row is ever reassigned, the session moves with it: one loud
        collision, not one per remaining row."""
        engine = _engine(tmp_path)
        _seed(engine, _turns(2, "old"))                 # 0..3
        assert migrate_turn_index(tmp_path / "lcm.db").status == "indexed"
        stale = _manager(engine).get_or_create(SID)
        stale._turn_index_anchored = True               # an anchor that went wrong
        stale.messages = _turns(2, "new")
        stale.persist_loop_result(0)

        assert len(telemetry.records) == 1
        assert [ti for ti, _ in _prompt_order(tmp_path / "lcm.db")] == list(range(8))

    def test_the_unique_index_replaces_the_legacy_one_for_good(self, tmp_path: Path) -> None:
        db = tmp_path / "lcm.db"
        LCMConversationStore(db).close()
        assert _indexes(db) == {LEGACY_TURN_INDEX_INDEX: False}
        migrate_turn_index(db)
        assert _indexes(db) == {TURN_INDEX_UNIQUE_INDEX: True}
        LCMConversationStore(db).close()                # every later open
        assert _indexes(db) == {TURN_INDEX_UNIQUE_INDEX: True}

    @pytest.mark.parametrize("migrated", [False, True])
    def test_readers_order_by_turn_index_then_rowid_without_sorting(
        self, tmp_path: Path, migrated: bool
    ) -> None:
        db = tmp_path / "lcm.db"
        store = LCMConversationStore(db)
        if migrated:
            migrate_turn_index(db)
        sql: list[str] = []
        store._conn.set_trace_callback(sql.append)
        store.get_messages(SID)
        store.get_fresh_tail(SID, 4)
        store.get_all_messages(SID)
        store.get_uncompacted_messages(SID)
        store._conn.set_trace_callback(None)

        readers = [s for s in sql if "ORDER BY turn_index" in s]
        assert len(readers) == 4
        for statement in readers:
            assert "rowid" in statement.split("ORDER BY", 1)[1]
            plan = " ".join(r[3] for r in store._conn.execute("EXPLAIN QUERY PLAN " + statement))
            assert "TEMP B-TREE" not in plan, plan


# --------------------------------------------------------------------------- #
# §5.3 — the migration
# --------------------------------------------------------------------------- #


def _legacy_history(db: Path) -> None:
    """One session in the shapes production held, plus an untouched neighbour.

    rowid order: lifetime A 0..5; a restart that numbered from 0 again, whose
    turn includes a mid-turn message persisted ahead of its tail (0, 3, 1, 2, 4);
    then /reset (0, 1). The neighbour's only out-of-rowid-order row is
    an ahead-set row, which is correct and must be left alone.
    """
    _legacy(db, [(i, f"A{i}") for i in range(6)], t0=1_000.0)
    _legacy(db, [(0, "B0"), (3, "B3"), (1, "B1"), (2, "B2"), (4, "B4")], t0=2_000.0)
    _legacy(db, [(0, "C0"), (1, "C1")], t0=3_000.0)
    _legacy(db, [(0, "n0"), (2, "n2"), (1, "n1")], sid="desktop:neighbour", t0=4_000.0)


class TestMigration:
    def test_no_duplicates_writes_only_the_index_and_the_version(self, tmp_path: Path) -> None:
        db = tmp_path / "lcm.db"
        _legacy(db, [(i, f"m{i}") for i in range(5)])
        schema, rows = _schema(db), _all_rows(db)

        result = migrate_turn_index(db, backup_path=tmp_path / "lcm.db.bak")

        assert result.status == "indexed"
        assert result.backup_path is None and not (tmp_path / "lcm.db.bak").exists()
        assert sorted(p.name for p in tmp_path.iterdir() if "bak" in p.name) == []
        assert _schema(db) == schema and _all_rows(db) == rows
        assert _user_version(db) == 1
        assert _indexes(db) == {TURN_INDEX_UNIQUE_INDEX: True}

    def test_duplicates_are_renumbered_in_prompt_order_after_a_backup(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        schema, rows_before = _schema(db), _all_rows(db)
        neighbour_before = _prompt_order(db, "desktop:neighbour")
        backup = tmp_path / "lcm.db.pre-turn-index-test.bak"

        result = migrate_turn_index(db, backup_path=backup)

        assert result.status == "migrated", result.error
        assert (result.sessions, result.runs, result.rows_renumbered) == (1, 3, 7)
        assert result.rows_reordered == 3            # B3 moves after B1, B2
        assert _prompt_order(db) == [
            (0, "A0"), (1, "A1"), (2, "A2"), (3, "A3"), (4, "A4"), (5, "A5"),
            (6, "B0"), (7, "B1"), (8, "B2"), (9, "B3"), (10, "B4"),
            (11, "C0"), (12, "C1")]
        assert _prompt_order(db, "desktop:neighbour") == neighbour_before
        # Nothing but turn_index moved, and no table or column was added.
        after = _all_rows(db)
        ti = 3  # rowid, id, session_id, turn_index, ...
        assert [r[:ti] + r[ti + 1:] for r in after] == [r[:ti] + r[ti + 1:] for r in rows_before]
        assert _schema(db) == schema
        assert _user_version(db) == 1
        assert _indexes(db) == {TURN_INDEX_UNIQUE_INDEX: True}
        # The backup is the untouched pre-migration state, private to the owner.
        assert result.backup_path == str(backup) and result.backup_bytes > 0
        assert _all_rows(backup) == rows_before
        assert _dups(backup) != []
        assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600

    def test_a_second_run_is_a_no_op(self, tmp_path: Path) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        assert migrate_turn_index(db, backup_path=tmp_path / "one.bak").status == "migrated"
        rows = _all_rows(db)
        result = migrate_turn_index(db, backup_path=tmp_path / "two.bak")
        assert result.status == "done_before"
        assert _all_rows(db) == rows and not (tmp_path / "two.bak").exists()

    def test_a_failed_check_rolls_everything_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        telemetry: _Telemetry, caplog: pytest.LogCaptureFixture,
    ) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        schema, rows = _schema(db), _all_rows(db)

        def _broken(*_a, **_k):
            raise migration.TurnIndexMigrationError("simulated verification failure")

        monkeypatch.setattr(migration, "_verify", _broken)
        result = migrate_turn_index(db, backup_path=tmp_path / "x.bak")

        assert result.status == "failed" and "simulated" in (result.error or "")
        assert _all_rows(db) == rows and _schema(db) == schema
        assert _user_version(db) == 0
        assert _indexes(db) == {LEGACY_TURN_INDEX_INDEX: False}
        assert (tmp_path / "x.bak").exists()           # kept for inspection
        assert "MIGRATION FAILED" in caplog.text
        assert [r["operation"] for r in telemetry.records] == ["turn_index_migration"]

    def test_a_write_between_backup_and_lock_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        real_backup = migration._backup

        def _backup_then_someone_writes(conn, path):
            out = real_backup(conn, path)
            _legacy(db, [(99, "late writer")], sid="web:other")
            return out

        monkeypatch.setattr(migration, "_backup", _backup_then_someone_writes)
        result = migrate_turn_index(db, backup_path=tmp_path / "x.bak")

        assert result.status == "failed"
        assert "between the backup and the write lock" in (result.error or "")
        assert _dups(db) != [] and _user_version(db) == 0
        assert _prompt_order(db, "web:other") == [(99, "late writer")]

    def test_not_enough_disk_refuses_before_writing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        rows = _all_rows(db)
        monkeypatch.setattr(migration.shutil, "disk_usage",
                            lambda _p: type("U", (), {"free": 0})())

        result = migrate_turn_index(db, backup_path=tmp_path / "x.bak")

        assert result.status == "failed" and "free space" in (result.error or "")
        assert not (tmp_path / "x.bak").exists()
        assert _all_rows(db) == rows and _user_version(db) == 0

    def test_a_backup_that_fails_leaves_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        rows = _all_rows(db)
        target = tmp_path / "x.bak"
        real_connect = sqlite3.connect

        def _connect(path, *a, **k):
            conn = real_connect(path, *a, **k)
            if str(path) == str(target):
                conn.close()                       # the copy cannot be written
            return conn

        monkeypatch.setattr(migration.sqlite3, "connect", _connect)
        result = migrate_turn_index(db, backup_path=target)

        assert result.status == "failed"
        assert not target.exists()                  # no partial copy posing as a backup
        assert _all_rows(db) == rows and _user_version(db) == 0

    def test_a_missing_db_is_not_created(self, tmp_path: Path) -> None:
        result = migrate_turn_index(tmp_path / "absent.db")
        assert result.status == "no_table"
        assert not (tmp_path / "absent.db").exists()

    def test_a_dry_run_in_memory_writes_nothing(self, tmp_path: Path) -> None:
        """The shape of the final pre-merge check: snapshot -> backup API ->
        :memory:, migrated there, and nothing written anywhere."""
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        rows = _all_rows(db)
        files = sorted(p.name for p in tmp_path.iterdir())

        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        mem = sqlite3.connect(":memory:", isolation_level=None)
        src.backup(mem)
        src.close()
        result = run_turn_index_migration(mem, backup_path=":memory:")

        assert result.status == "migrated" and result.rows_renumbered == 7
        assert result.backup_path == ":memory:" and result.backup_bytes == 0
        assert _all_rows(db) == rows
        assert sorted(p.name for p in tmp_path.iterdir()) == files

    def test_the_default_backup_is_not_picked_up_as_a_live_db(self) -> None:
        name = migration.default_backup_path("/x/data/lcm.db").name
        assert name.startswith("lcm.db.pre-turn-index-") and name.endswith(".bak")
        assert not name.endswith(".db")             # db_snapshot captures *.db only

    def test_the_migration_refuses_a_connection_inside_a_transaction(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "lcm.db"
        _legacy_history(db)
        conn = sqlite3.connect(db)                 # default mode: not autocommit
        result = run_turn_index_migration(conn, backup_path=":memory:")
        conn.close()
        assert result.status == "failed" and _user_version(db) == 0


class TestPlan:
    def test_a_restart_from_zero_is_lifted_above_the_older_run(self) -> None:
        plan = plan_renumbering([(1, "s", 0), (2, "s", 1), (3, "s", 0), (4, "s", 1)])
        assert plan.mapping == [(3, 2), (4, 3)] and plan.order == {"s": [1, 2, 3, 4]}

    def test_a_trim_plateau_becomes_consecutive(self) -> None:
        rows = [(r, "s", ti) for r, ti in enumerate([50, 51, 52, 50, 51, 50], start=1)]
        plan = plan_renumbering(rows)
        assert dict(plan.mapping) == {4: 53, 5: 54, 6: 55}

    def test_a_gap_fill_stays_in_its_run(self) -> None:
        rows = [(1, "s", 0), (2, "s", 3), (3, "s", 1), (4, "s", 2), (5, "s", 4), (6, "s", 0)]
        plan = plan_renumbering(rows)
        assert plan.runs == 2
        assert plan.order["s"] == [1, 3, 4, 2, 5, 6]
        assert plan.mapping == [(6, 5)]

    def test_sessions_without_a_repeat_are_untouched(self) -> None:
        plan = plan_renumbering([(1, "s", 0), (2, "s", 2), (3, "s", 1)])
        assert plan.mapping == [] and plan.order == {}

    def test_random_histories_come_out_unique_with_every_run_in_order(self) -> None:
        rng = random.Random(20260926)
        for _ in range(300):
            rows, rowid = [], 0
            runs: list[list[int]] = []
            for _life in range(rng.randint(1, 5)):
                start = rng.choice([0, 0, rng.randint(0, 30)])
                indices = list(range(start, start + rng.randint(1, 25)))
                if len(indices) > 3 and rng.random() < 0.5:    # an ahead-set row
                    k = rng.randrange(1, len(indices) - 1)
                    indices.insert(k, indices.pop(-1))
                runs.append(indices)
                for ti in indices:
                    rowid += 1
                    rows.append((rowid, "s", ti))
            plan = plan_renumbering(rows)
            final = {rid: ti for rid, _s, ti in rows} | dict(plan.mapping)
            values = list(final.values())
            assert len(set(values)) == len(values)
            if plan.order:
                ordered = sorted(final, key=lambda r: (final[r], r))
                assert ordered == plan.order["s"]


# --------------------------------------------------------------------------- #
# §4 — after the migration, the readers see the conversation the model saw
# --------------------------------------------------------------------------- #


def _two_lifetimes(db: Path, n: int = 12) -> LCMConversationStore:
    """Lifetime A then lifetime B, both numbered 0..n-1: the pre-fix state."""
    _legacy(db, [(i, f"A{i}") for i in range(n)], t0=1_000.0)
    _legacy(db, [(i, f"B{i}") for i in range(n)], t0=1_000.0 + n)
    assert migrate_turn_index(db, backup_path=db.with_name("lcm.db.bak")).status == "migrated"
    return LCMConversationStore(db)


class _CapturingSummarizer(LCMSummarizer):
    """The REAL summarize_messages prompt builder; only the model call is faked."""

    def __init__(self) -> None:
        super().__init__(MagicMock(), model="default")
        self.prompts: list[str] = []

    async def _call_model(self, prompt: str) -> str:  # type: ignore[override]
        self.prompts.append(prompt)
        return "summary"


class TestImpact:
    def test_the_summarizer_is_sent_one_conversation_in_order(self, tmp_path: Path) -> None:
        store = _two_lifetimes(tmp_path / "lcm.db")
        summ = _CapturingSummarizer()
        cfg = CompactionConfig(fresh_tail_count=4, compaction_batch_size=10)
        asyncio.run(LCMCompactor(store, LCMSummaryStore(tmp_path / "lcm.db"), summ, cfg)
                    .compact(SID))

        first = [ln.split("] ", 1)[1] for ln in summ.prompts[0].splitlines()
                 if ln.startswith(("[user] ", "[assistant] "))]
        assert first == ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"]

    def test_the_compactor_keeps_the_newest_rows_fresh(self, tmp_path: Path) -> None:
        store = _two_lifetimes(tmp_path / "lcm.db")
        cfg = CompactionConfig(fresh_tail_count=4, compaction_batch_size=10)
        asyncio.run(LCMCompactor(store, LCMSummaryStore(tmp_path / "lcm.db"),
                                 _CapturingSummarizer(), cfg).compact(SID))

        con = sqlite3.connect(tmp_path / "lcm.db")
        fresh = [r[0] for r in con.execute(
            "SELECT content FROM lcm_messages WHERE session_id = ? AND compacted = 0"
            " ORDER BY rowid", (SID,))]
        con.close()
        assert fresh == ["B8", "B9", "B10", "B11"]

    def test_the_assembler_fresh_tail_is_the_newest_rows(self, tmp_path: Path) -> None:
        store = _two_lifetimes(tmp_path / "lcm.db")
        asm = LCMAssembler(store, LCMSummaryStore(tmp_path / "lcm.db"),
                           CompactionConfig(fresh_tail_count=4))
        got = [m.content for m in asm.assemble(SID, token_budget=10_000).fresh_messages]
        assert got == ["B8", "B9", "B10", "B11"]

    def test_golden_trace_context_is_the_preceding_conversation(self, tmp_path: Path) -> None:
        store = _two_lifetimes(tmp_path / "lcm.db")
        resolve = lcm_context_resolver(store)
        call_ts = 1_000.0 + 12 + 6 + 0.5               # just after B6
        got = [m["content"] for m in resolve({"session_id": SID, "timestamp": call_ts})]
        assert got == ["A7", "A8", "A9", "A10", "A11", "B0", "B1", "B2", "B3", "B4", "B5", "B6"]


class TestControl:
    def test_live_prompt_after_restart_is_chronological(self, tmp_path: Path) -> None:
        """Passed before the fix and still does. The model's live context is
        ChatSession.messages, and rehydrate reads by rowid (messages_page), so
        even the duplicate indices pre-fix builds left behind never reordered
        it. The next row still lands above everything the session holds."""
        engine = _engine(tmp_path)
        _seed(engine, _turns(30, "A"))                  # 0..59
        _legacy(tmp_path / "lcm.db",                    # a pre-fix cold restart: 0..5
                [(0, "B q0"), (1, "B a0"), (2, "B q1"), (3, "B a1"),
                 (4, "B q2"), (5, "B a2")], t0=9_000.0)
        assert _dups(tmp_path / "lcm.db")               # the precondition holds

        mgr = _manager(engine, rehydrate=True)
        assert mgr.rehydrate_if_cold(SID) > 0
        session = mgr.get_or_create(SID)
        texts = [m.text for m in session.messages]
        assert texts[-6:] == ["B q0", "B a0", "B q1", "B a1", "B q2", "B a2"]
        assert len(texts) == len(set(texts))            # nothing doubled
        assert session.add_user_message("next") == 60


# --------------------------------------------------------------------------- #
# Ordering: the migration runs before any writer is wired to lcm.db
# --------------------------------------------------------------------------- #


class TestDaemonOrdering:
    def test_migration_runs_after_the_engine_and_before_every_writer(self) -> None:
        tree = ast.parse((ROOT / "src" / "prometheus" / "daemon.py").read_text())
        run = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_daemon")

        def calls(name: str) -> list[int]:
            return sorted(
                n.lineno for n in ast.walk(run) if isinstance(n, ast.Call) and (
                    (isinstance(n.func, ast.Name) and n.func.id == name)
                    or (isinstance(n.func, ast.Attribute) and n.func.attr == name))
            )

        [migrate] = calls("migrate_turn_index")
        [engine] = calls("LCMEngine")
        wired = [
            n.lineno for n in ast.walk(run) if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Name) and n.value.id == "lcm_engine"
            and any(isinstance(t, ast.Attribute) for t in n.targets)
        ]
        adapter_starts = [
            n.lineno for n in ast.walk(run)
            if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "start"
        ]
        writers = {
            "session_manager / agent_loop wiring": wired,
            "adapter / job .start()": adapter_starts,
            "cron scheduler": calls("run_scheduler_loop"),
            "memory extractor": calls("MemoryExtractor"),
            "golden-trace exporter": calls("GoldenTraceExporter"),
            "web server": calls("launch_web"),
        }
        assert engine < migrate
        for what, lines in writers.items():
            assert lines, f"no {what} found in run_daemon; update this test"
            assert migrate < min(lines), f"the migration runs after the {what}"
