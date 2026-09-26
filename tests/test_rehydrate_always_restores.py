"""A restart restores the recent conversation, and a failed turn keeps a saved mid-turn message.

(a) Rehydrate used to give up whenever the newest 40 rows, cut to an 8,000-token
    budget newest-first, held no clean human turn: one big tool result was
    enough, and so was any agentic turn longer than 40 rows. And four send paths
    never asked: ``inject_turn`` (task results), ``POST /api/chat``, Slack and
    Discord. Either way the model started blind after a restart. On the mini's
    2026-09-26 snapshot, 7 of the 27 resumes since rehydrate was switched on
    restored nothing.
(b) A failed WS turn's ``rollback_to`` dropped a message the user sent mid-turn,
    though it was already saved, so the model never saw it again.

The tests marked "was blind" / "was dropped" failed before the fix, each for the
reason in its docstring; the controls passed before and still pass.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import _REHYDRATE_TOKEN_BUDGET, SessionManager
from prometheus.memory.lcm_engine import LCMEngine

SID = "desktop:rehydrate-always"


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


def _round(i: int, result: str) -> list[ConversationMessage]:
    return [
        ConversationMessage(role="assistant", content=[
            ToolUseBlock(id=f"t{i}", name="bash", input={"command": f"step {i}"})]),
        ConversationMessage(role="user", content=[
            ToolResultBlock(tool_use_id=f"t{i}", content=result)]),
    ]


def _turns(n: int, tag: str) -> list[ConversationMessage]:
    out: list[ConversationMessage] = []
    for i in range(n):
        out += [_user(f"{tag} q{i}"), _asst(f"{tag} a{i}")]
    return out


def _seed(engine: LCMEngine, messages: list[ConversationMessage], sid: str = SID) -> None:
    """One daemon lifetime writing ``messages`` through the real persist path."""
    session = _manager(engine, rehydrate=False).get_or_create(sid)
    session.messages = list(messages)
    session.persist_loop_result(0)


def _texts(messages) -> list[str]:
    return [m.text for m in messages]


def _tokens(messages) -> int:
    return sum(max(1, len(m.content_json) // 4) for m in messages)


def _pairs_ok(messages) -> bool:
    """Every tool_result answers a tool_use that comes before it in the list."""
    seen: set[str] = set()
    for m in messages:
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                seen.add(b.id)
            if isinstance(b, ToolResultBlock) and b.tool_use_id not in seen:
                return False
    return True


# --------------------------------------------------------------------------- #
# (a) The window: restore the newest turn instead of nothing
# --------------------------------------------------------------------------- #


class TestTheWindow:
    def test_one_big_tool_result_no_longer_blanks_the_restart(self, tmp_path: Path) -> None:
        """Was blind: the newest-first budget stopped at the big result, so the
        budgeted window held only the final reply and no human turn."""
        engine = _engine(tmp_path)
        big = "line of report output\n" * 2_000               # ~44k chars, ~11k tokens
        _seed(engine, [*_turns(3, "old"), _user("run the report"), *_round(1, big),
                       _asst("done: the report is ready")])

        mgr = _manager(engine)
        restored = mgr.rehydrate_if_cold(SID)
        messages = mgr.get_or_create(SID).messages

        assert restored > 0, "rehydrate restored nothing: the model starts blind"
        assert _texts(messages)[0] == "run the report"
        assert _texts(messages)[-1] == "done: the report is ready"
        [result] = [b for m in messages for b in m.content if isinstance(b, ToolResultBlock)]
        assert len(result.content) < 2_000 and "shortened at restart" in result.content
        assert _pairs_ok(messages) and _tokens(messages) <= _REHYDRATE_TOKEN_BUDGET
        # The store still holds the whole result: only the restored copy is short.
        con = sqlite3.connect(tmp_path / "lcm.db")
        stored = con.execute("SELECT max(length(content_json)) FROM lcm_messages").fetchone()[0]
        con.close()
        assert stored > len(big)

    def test_a_turn_longer_than_the_window_restores_its_request_and_newest_rounds(
        self, tmp_path: Path
    ) -> None:
        """Was blind: 60 tool rounds put the request 120 rows back, outside the
        40-row window, so no clean human turn was in reach."""
        engine = _engine(tmp_path)
        rounds = [m for i in range(60) for m in _round(i, f"result {i}: " + "x" * 1_500)]
        _seed(engine, [*_turns(2, "old"), _user("refactor the module"), *rounds,
                       _asst("refactor done")])

        mgr = _manager(engine)
        restored = mgr.rehydrate_if_cold(SID)
        messages = mgr.get_or_create(SID).messages

        assert restored > 1, "rehydrate restored nothing: the model starts blind"
        first = messages[0]
        assert first.role == "user" and first.content[0].text == "refactor the module"
        assert any("left out" in getattr(b, "text", "") for b in first.content[1:])
        assert _texts(messages)[-1] == "refactor done"
        assert messages[1].role == "assistant"          # the kept rounds start cleanly
        assert _pairs_ok(messages) and _tokens(messages) <= _REHYDRATE_TOKEN_BUDGET

    def test_the_restored_copy_is_never_written_back(self, tmp_path: Path) -> None:
        engine = _engine(tmp_path)
        _seed(engine, [_user("run it"), *_round(1, "y" * 50_000), _asst("done")])
        con = sqlite3.connect(tmp_path / "lcm.db")
        before = con.execute("SELECT rowid, content_json FROM lcm_messages ORDER BY rowid").fetchall()

        mgr = _manager(engine)
        assert mgr.rehydrate_if_cold(SID) > 0
        mgr.get_or_create(SID).add_user_message("next")

        after = con.execute("SELECT rowid, content_json FROM lcm_messages ORDER BY rowid").fetchall()
        con.close()
        assert after[: len(before)] == before and len(after) == len(before) + 1

    def test_control_a_window_that_restored_still_restores_the_same(
        self, tmp_path: Path
    ) -> None:
        """Passed before: when the newest window already held a clean turn, the
        restore is exactly what it was."""
        engine = _engine(tmp_path)
        history = [*_turns(3, "a"), _user("plan"), *_round(1, "ok"), _asst("planned")]
        _seed(engine, history)

        mgr = _manager(engine)
        assert mgr.rehydrate_if_cold(SID) == len(history)
        assert _texts(mgr.get_or_create(SID).messages) == _texts(history)

    def test_control_no_human_turn_at_all_still_restores_nothing_blindly(
        self, tmp_path: Path
    ) -> None:
        """A session of tool rows with no user text anywhere has nowhere safe to
        start: restoring from a tool result would orphan it."""
        engine = _engine(tmp_path)
        _seed(engine, [*_round(1, "a"), *_round(2, "b")])
        assert _manager(engine).rehydrate_if_cold(SID) == 0


# --------------------------------------------------------------------------- #
# (a) The send paths that never asked
# --------------------------------------------------------------------------- #


class TestSendPaths:
    @pytest.mark.asyncio
    async def test_a_task_result_after_a_restart_sees_the_conversation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was blind: inject_turn created the session without rehydrate, and the
        session was warm from then on, so nothing restored it later either."""
        from prometheus.gateway.config import Platform, PlatformConfig
        from prometheus.gateway.telegram import TelegramAdapter
        from prometheus.tools.base import ToolRegistry

        monkeypatch.setattr("prometheus.engine.honesty.evaluate_and_record",
                            lambda *a, **k: None)
        engine = _engine(tmp_path)
        _seed(engine, [*_turns(2, "before"), _user("watch the build"), _asst("watching")])
        seen: list[list[str]] = []

        class _Loop:
            async def run_async(self, **kw):  # noqa: ANN003
                seen.append(_texts(kw["messages"]))
                return SimpleNamespace(text="noted", messages=list(kw["messages"]))

        adapter = TelegramAdapter(
            config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
            agent_loop=_Loop(), tool_registry=ToolRegistry(),
            session_manager=_manager(engine),               # a restarted daemon
        )
        await adapter.inject_turn(SID, "task done: build green")

        assert seen and seen[0][-1] == "task done: build green"
        assert "watch the build" in seen[0], f"the task result ran blind: {seen[0]}"

    def test_api_chat_after_a_restart_sees_the_conversation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was blind: POST /api/chat created web:<id> without rehydrate."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.skills.registry import SkillRegistry
        from prometheus.web.server import create_app

        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        engine = _engine(tmp_path)
        _seed(engine, [_user("what is the plan"), _asst("the plan is X")], sid="web:s1")
        seen: list[list[str]] = []

        class _Loop:
            async def run_async(self, **kw):  # noqa: ANN003
                seen.append(_texts(kw["messages"]))
                return SimpleNamespace(text="ok", turns=1, messages=[],
                                       usage=SimpleNamespace(input_tokens=1, output_tokens=1))

        app = create_app({"gateway": {"system_prompt": "sys"}}, session_mgr=_manager(engine),
                         skill_registry=SkillRegistry(), agent_loop=_Loop())
        TestClient(app).post("/api/chat", json={"session_id": "s1", "content": "and now?"})

        assert seen and seen[0][-1] == "and now?"
        assert "what is the plan" in seen[0], f"/api/chat ran blind: {seen[0]}"

    @pytest.mark.parametrize("module", ["slack", "discord"])
    def test_slack_and_discord_ask_before_the_turn(self, module: str) -> None:
        """Was blind: their message handlers created the session without rehydrate.
        Checked in the source, because neither SDK is installed everywhere the
        suite runs: in every method that runs a turn (it adds the user message),
        rehydrate_if_cold comes before get_or_create."""
        import ast

        import prometheus.gateway as gateway

        tree = ast.parse((Path(gateway.__file__).parent / f"{module}.py").read_text())
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            calls = [(c.lineno, c.func.attr) for c in ast.walk(node)
                     if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and c.func.attr in ("rehydrate_if_cold", "get_or_create", "add_user_message")]
            names = [n for _, n in sorted(calls)]
            if "add_user_message" in names and "get_or_create" in names:
                found = True
                assert "rehydrate_if_cold" in names, f"{module}.{node.name} never rehydrates"
                assert names.index("rehydrate_if_cold") < names.index("get_or_create")
        assert found, f"no turn-running method found in {module}"


# --------------------------------------------------------------------------- #
# (b) P6: a failed WS turn keeps the saved mid-turn message
# --------------------------------------------------------------------------- #


class TestFailedTurn:
    def test_a_saved_mid_turn_message_survives_the_rollback(self, tmp_path: Path) -> None:
        """Was dropped: rollback_to() discarded everything past the turn's user
        message, including a message sent mid-turn and already saved."""
        engine = _engine(tmp_path)
        session = _manager(engine).get_or_create(SID)
        session.add_user_message("first")
        original_len = len(session.messages)
        session.messages.append(_asst("partial"))          # the loop, in place
        session.add_user_message("mid-turn steer")          # saved at once
        session.messages.append(_asst("more partial"))

        session.rollback_to(original_len)                   # the turn failed

        assert _texts(session.messages) == ["first", "mid-turn steer"]
        session.add_user_message("next")
        assert _texts(session.messages) == ["first", "mid-turn steer", "next"]
        con = sqlite3.connect(tmp_path / "lcm.db")
        stored = [r[0] for r in con.execute(
            "SELECT content FROM lcm_messages ORDER BY turn_index, rowid")]
        con.close()
        # The store reads the conversation in the same order the model now sees it.
        assert stored == ["first", "mid-turn steer", "next"]

    def test_the_bridge_failure_path_keeps_it_for_the_next_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was dropped, end to end: the WS bridge's failure path is where the
        rollback runs."""
        import httpx

        import prometheus.engine.agent_loop as al
        from prometheus.web.ws_server import WebSocketBridge

        engine = _engine(tmp_path)
        mgr = _manager(engine)
        session = mgr.get_or_create(SID)
        session.add_user_message("start the deploy")

        class _Delta:
            def __init__(self, text: str) -> None:
                self.text = text

        _Delta.__name__ = "AssistantTextDelta"

        def failing(ctx, messages, **kw):  # noqa: ANN001, ANN003
            async def gen():
                yield _Delta("working"), None
                messages.append(_asst("half a thought"))
                session.add_user_message("also update the changelog")   # mid-turn send
                request = httpx.Request("POST", "http://backend/v1/chat/completions")
                raise httpx.HTTPStatusError(
                    "400", request=request, response=httpx.Response(400, request=request))
            return gen()

        seen: list[list[str]] = []

        def healthy(ctx, messages, **kw):  # noqa: ANN001, ANN003
            seen.append(_texts(messages))

            async def gen():
                yield _Delta("done"), None
                messages.append(_asst("done"))
            return gen()

        bridge = WebSocketBridge(session_mgr=mgr, loop_context=object(),
                                 agent_state_ref={"state": "thinking"})

        async def _drop(_frame):
            return None

        bridge.broadcast = _drop
        monkeypatch.setattr(al, "run_loop", failing)
        asyncio.run(bridge._run_agent(SID, session))
        assert "half a thought" not in _texts(session.messages)
        assert "also update the changelog" in _texts(session.messages), "the saved message was dropped"

        monkeypatch.setattr(al, "run_loop", healthy)
        session.add_user_message("go on")
        asyncio.run(bridge._run_agent(SID, session))
        assert seen[0] == ["start the deploy", "also update the changelog", "go on"]

    def test_control_the_failed_turns_own_rows_are_still_discarded(
        self, tmp_path: Path
    ) -> None:
        """Passed before: the loop's own rows (never saved) are what a failure
        throws away, so a poisoned tool result cannot brick the session."""
        engine = _engine(tmp_path)
        session = _manager(engine).get_or_create(SID)
        session.add_user_message("check the server")
        original_len = len(session.messages)
        session.messages.extend(_round(1, "poisoned output"))

        assert session.rollback_to(original_len) == 2
        assert _texts(session.messages) == ["check the server"]
