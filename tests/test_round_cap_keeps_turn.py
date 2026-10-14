"""#774 — a turn that hits its round cap keeps its work (in memory AND in LCM).

Incident (2026-10-13): a Beacon turn ran 200 model rounds / 223 tool calls,
then ``run_loop`` raised ``RuntimeError("Exceeded maximum turn limit (200)")``.
The web bridge's generic failure handler rolled the session back
(``rollback_to``, ``discarded=400``) and, because the web path persists a turn's
rows at turn END, none of the 400 messages ever reached LCM. The next turn
started at 81,686 tokens instead of 218,923 and the model said of its own two
hours of work: "I have not run any of those."

The rollback exists for REAL failures (a poisoned tool result that 400s every
later request — tests/test_web_turn_discard_on_failure.py). The round cap is a
budget stop: every round completed normally, so the history is valid and is
kept, like a user stop.

Tested by OUTCOME on the real path: the real ``run_loop``, ``ChatSession``,
``LCMEngine`` (scratch DB), ``WebSocketBridge`` and Telegram adapter with a real
``AgentLoop``. Only the model provider is scripted (a registered double).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import SessionManager
from prometheus.engine.usage import UsageSnapshot
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from tests.support.doubles import register_double


class _In(BaseModel):
    n: int


class _CountTool(BaseTool):
    name = "count_tool"
    description = "counts"
    input_model = _In

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"counted {arguments.n}")


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_CountTool())
    return reg


@register_double(
    "round_cap.scripted_provider",
    replaces="prometheus.providers.base.ModelProvider",
)
class _ToolCaller(ModelProvider):
    """Calls ``count_tool`` every round (never stops on its own) unless told to
    answer in prose. Records every request so a test can read what the NEXT
    turn's prompt contained."""

    def __init__(self, *, prose: str | None = None) -> None:
        self.prose = prose
        self.calls = 0
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        self.calls += 1
        if self.prose is not None:
            msg = ConversationMessage(role="assistant", content=[TextBlock(text=self.prose)])
        else:
            msg = ConversationMessage(role="assistant", content=[
                ToolUseBlock(id=f"call{self.calls}", name="count_tool", input={"n": self.calls}),
            ])
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="tool_calls" if self.prose is None else "stop",
        )


def _ctx(provider: ModelProvider, max_turns: int) -> LoopContext:
    return LoopContext(provider=provider, model="stub-model", system_prompt="",
                       max_tokens=128, tool_registry=_registry(), max_turns=max_turns)


def _engine(tmp_path) -> LCMEngine:
    # Real engine, scratch DB. These turns are far too short to trigger LCM
    # compaction, so the summarizer provider is never called.
    return LCMEngine(MagicMock(), db_path=tmp_path / "lcm.db")


def _tool_use_ids(messages) -> list[str]:
    return [b.id for m in messages for b in m.content if isinstance(b, ToolUseBlock)]


def _answered_ids(messages) -> set[str]:
    return {b.tool_use_id for m in messages for b in m.content if isinstance(b, ToolResultBlock)}


def _stored_rows(eng: LCMEngine, session_id: str):
    return eng._conv_store.messages_after_id(0, session_id=session_id)


# ---------------------------------------------------------------------------
# The loop itself
# ---------------------------------------------------------------------------


def test_cap_raises_a_dedicated_exception_carrying_the_turn():
    from prometheus.engine.agent_loop import MaxTurnsExhausted

    provider = _ToolCaller()
    messages = [ConversationMessage.from_user_text("go")]

    async def drain():
        async for _ in run_loop(_ctx(provider, 3), messages, surface="cli"):
            pass

    with pytest.raises(MaxTurnsExhausted) as info:
        asyncio.run(drain())

    exc = info.value
    # Still a RuntimeError with the old wording: gym/ladder and coding mode
    # match on it, and both must keep working unchanged.
    assert isinstance(exc, RuntimeError)
    assert "Exceeded maximum turn limit (3)" in str(exc)
    assert exc.max_turns == 3
    # The exception carries the very list the loop appended to, so callers
    # that handed run_async a copy can still save the turn.
    assert exc.messages is messages
    assert provider.calls == 3
    assert _tool_use_ids(messages) == ["call1", "call2", "call3"]
    # Every tool call is answered, so the next request on this history is valid.
    assert set(_tool_use_ids(messages)) <= _answered_ids(messages)
    # The turn ends on a plain assistant note naming the limit, which the user
    # sees and the model reads on the next turn.
    assert messages[-1].role == "assistant"
    assert not messages[-1].tool_uses
    assert "200" not in exc.notice and "3" in exc.notice
    assert messages[-1].text == exc.notice


def test_repair_answers_tool_calls_left_without_results():
    from prometheus.engine.agent_loop import _answer_unanswered_tool_uses

    messages = [
        ConversationMessage.from_user_text("go"),
        ConversationMessage(role="assistant", content=[
            ToolUseBlock(id="a", name="count_tool", input={"n": 1}),
            ToolUseBlock(id="b", name="count_tool", input={"n": 2}),
        ]),
    ]
    assert _answer_unanswered_tool_uses(messages) == 2
    assert messages[-1].role == "user"
    assert _answered_ids(messages) == {"a", "b"}
    assert all(
        b.is_error and "NOT EXECUTED" in b.content
        for b in messages[-1].content if isinstance(b, ToolResultBlock)
    )
    # Idempotent: a complete history is left alone.
    assert _answer_unanswered_tool_uses(messages) == 0

    # A partially answered round gets only the missing result.
    partial = [
        ConversationMessage(role="assistant", content=[
            ToolUseBlock(id="x", name="count_tool", input={"n": 1}),
            ToolUseBlock(id="y", name="count_tool", input={"n": 2}),
        ]),
        ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id="x", content="ok")]),
    ]
    assert _answer_unanswered_tool_uses(partial) == 1
    assert _answered_ids(partial) == {"x", "y"}


# ---------------------------------------------------------------------------
# Web / Beacon (WebSocketBridge._run_agent)
# ---------------------------------------------------------------------------


def _bridge(loop_context, eng):
    from prometheus.web.ws_server import WebSocketBridge

    mgr = SessionManager()
    mgr.lcm_engine = eng
    return WebSocketBridge(session_mgr=mgr, loop_context=loop_context,
                           agent_state_ref={"state": "idle"})


def test_web_turn_at_the_cap_is_kept_persisted_and_seen_by_the_next_turn(tmp_path):
    eng = _engine(tmp_path)
    sid = "beacon:roundcap"
    capped = _ToolCaller()
    bridge = _bridge(_ctx(capped, 2), eng)
    frames: list[dict] = []

    async def cap(ev):
        frames.append(ev)

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create(sid)
    session.add_user_message("build the thing")
    original_len = len(session.messages)

    text, _usage = asyncio.run(bridge._run_agent(sid, session))

    # In memory: the turn's work is still on the session.
    assert len(session.messages) > original_len
    assert _tool_use_ids(session.messages) == ["call1", "call2"]
    assert session.messages[-1].role == "assistant"
    # Durable: the same rows are in LCM, not just the user's message.
    stored = _stored_rows(eng, sid)
    stored_json = " ".join(r.content_json or "" for r in stored)
    assert "call1" in stored_json and "call2" in stored_json
    assert any(r.role == "assistant" and "round" in (r.content or "").lower() for r in stored)
    # The user is told plainly, and it is not reported as an error.
    assert not any(f.get("type") == "error" for f in frames)
    done = next(f for f in frames if f.get("type") == "chat_done")
    assert done["payload"]["stop_reason"] == "max_turns_exhausted"
    assert "round" in text.lower()
    assert json.dumps(done["payload"])

    # The NEXT turn's prompt contains the capped turn's work.
    follow = _ToolCaller(prose="carrying on")
    bridge.loop_context = _ctx(follow, 5)
    session.add_user_message("continue")
    asyncio.run(bridge._run_agent(sid, session))
    # The turn's own request (a session-title call may follow it; skip that).
    turn_requests = [r for r in follow.requests
                     if any(m.role == "user" and m.text == "continue" for m in r.messages)]
    assert turn_requests, "the follow-up turn never reached the model"
    seen = turn_requests[0].messages
    assert {"call1", "call2"} <= set(_tool_use_ids(seen))
    assert {"call1", "call2"} <= _answered_ids(seen)


def test_a_real_failure_still_discards_the_tail(tmp_path):
    """The asymmetry stays: only the round cap keeps; a provider failure rolls back."""
    eng = _engine(tmp_path)
    sid = "beacon:realfail"

    class _Boom(_ToolCaller):
        async def stream_message(self, request):  # noqa: ANN001
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("backend exploded")
            async for ev in super().stream_message(request):
                yield ev

    bridge = _bridge(_ctx(_Boom(), 5), eng)

    async def cap(ev):
        return None

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create(sid)
    session.add_user_message("go")
    original_len = len(session.messages)
    asyncio.run(bridge._run_agent(sid, session))
    assert len(session.messages) == original_len


# ---------------------------------------------------------------------------
# Telegram (AgentLoop.run_async hands the loop a COPY of the session list)
# ---------------------------------------------------------------------------


def test_telegram_turn_at_the_cap_keeps_the_user_message_and_the_turn(tmp_path, monkeypatch):
    from prometheus.gateway.config import Platform, PlatformConfig
    from prometheus.gateway.telegram import TelegramAdapter

    eng = _engine(tmp_path)
    sid = "telegram:roundcap"
    loop = AgentLoop(provider=_ToolCaller(), model="stub-model",
                     tool_registry=_registry(), max_turns=2)
    adapter = TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
        agent_loop=loop,
        tool_registry=ToolRegistry(),
    )
    adapter.session_manager.lcm_engine = eng
    monkeypatch.setattr("prometheus.engine.honesty.evaluate_and_record", lambda *a, **k: None)

    session = adapter.session_manager.get_or_create(sid)
    reply = asyncio.run(adapter._run_agent_turn(session, "build the thing", session_id=sid))

    assert "round" in reply.lower()
    assert not reply.startswith("Error")
    # The user's message was NOT rolled back, and the turn's work is kept.
    assert session.messages[0].role == "user"
    assert session.messages[0].text == "build the thing"
    assert _tool_use_ids(session.messages) == ["call1", "call2"]
    stored_json = " ".join(r.content_json or "" for r in _stored_rows(eng, sid))
    assert "call1" in stored_json and "call2" in stored_json


# ---------------------------------------------------------------------------
# Config: model.max_turns
# ---------------------------------------------------------------------------


def test_max_turns_is_a_config_key_defaulting_to_200():
    from prometheus.config.shipped_defaults import SHIPPED_MAX_TURNS, resolve_max_turns

    assert SHIPPED_MAX_TURNS == 200
    assert resolve_max_turns(None) == 200
    assert resolve_max_turns({}) == 200
    assert resolve_max_turns({"max_turns": 50}) == 50
    # A zero/negative/garbage cap would end every turn at once: not obeyed.
    assert resolve_max_turns({"max_turns": 0}) == 200
    assert resolve_max_turns({"max_turns": "lots"}) == 200
    # The dataclass and AgentLoop defaults are the one shipped constant.
    assert LoopContext.__dataclass_fields__["max_turns"].default == SHIPPED_MAX_TURNS


def test_every_entry_point_forwards_max_turns():
    """daemon.py builds AgentLoop and the web LoopContext separately; __main__.py
    builds a LoopContext for the CLI and workers. Each must forward the configured value or the
    key silently changes nothing on that surface (the max_tool_iterations_cloud
    lesson, tests/test_daemon_cloud_iteration_cap.py)."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "prometheus"

    def kwargs_for(path: Path, callee: str) -> list[set[str]]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if name == callee:
                    out.append({kw.arg for kw in node.keywords if kw.arg})
        return out

    for path, callee in ((src / "daemon.py", "AgentLoop"), (src / "daemon.py", "LoopContext"),
                         (src / "__main__.py", "LoopContext")):
        sites = [k for k in kwargs_for(path, callee) if "max_tool_iterations" in k]
        assert sites, f"sanity: {path.name} builds {callee} with the iteration caps"
        for kw in sites:
            assert "max_turns" in kw, f"{path.name} {callee}(...) does not forward max_turns"


def test_loop_context_parity_guard_covers_max_turns():
    from prometheus.daemon import loop_ceiling_divergence

    class _Loop:
        _max_tool_iterations = 1
        _max_tool_iterations_cloud = 1
        _max_turns = 200

    class _Ctx:
        max_tool_iterations = 1
        max_tool_iterations_cloud = 1
        max_turns = 150

    assert ("max_turns", 200, 150) in loop_ceiling_divergence(_Loop(), _Ctx())


# ---------------------------------------------------------------------------
# The other surfaces that run the loop: CLI REPL, workers, Slack, Discord, REST
# ---------------------------------------------------------------------------


@register_double(
    "round_cap.continue_provider",
    replaces="prometheus.providers.base.ModelProvider",
)
class _CallsUntilContinue(_ToolCaller):
    """Tool-calls every round until the latest user text is "continue", then
    answers in prose — one provider across two turns, like the real thing."""

    async def stream_message(self, request):  # noqa: ANN001
        users = [m for m in request.messages if m.role == "user" and m.text]
        self.prose = "carrying on" if users and users[-1].text == "continue" else None
        async for ev in super().stream_message(request):
            yield ev


def test_cli_repl_keeps_the_capped_turn_for_the_next_prompt(tmp_path, monkeypatch):
    from prometheus.__main__ import run_interactive

    eng = _engine(tmp_path)
    provider = _CallsUntilContinue()
    inputs = iter(["build the thing", "continue", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    asyncio.run(run_interactive(_ctx(provider, 2), eng, "cli:roundcap"))

    follow = [r for r in provider.requests
              if any(m.role == "user" and m.text == "continue" for m in r.messages)]
    assert follow, "the second prompt never reached the model"
    seen = follow[0].messages
    # Both rounds of the capped turn are in the next prompt, every call answered
    # (the old handler popped the last tool-result message, orphaning call2).
    assert {"call1", "call2"} <= set(_tool_use_ids(seen))
    assert {"call1", "call2"} <= _answered_ids(seen)
    assert any(r.role == "assistant" and "round" in (r.content or "").lower()
               for r in _stored_rows(eng, "cli:roundcap"))


def test_worker_reports_the_round_cap_honestly():
    from prometheus.cli.worker import run_worker_turn

    result = asyncio.run(run_worker_turn(_ctx(_ToolCaller(), 2), "do the brief"))
    assert result["status"] == "failed"
    assert result["reason"] == "max_turns_exhausted"


def test_slack_turn_at_the_cap_keeps_the_turn(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from prometheus.gateway.config import Platform, PlatformConfig
    from prometheus.gateway.slack import SlackAdapter

    eng = _engine(tmp_path)
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._signal_bus = None
    adapter._prometheus_config = {}
    adapter.config = PlatformConfig(platform=Platform.SLACK, token="t", app_token="a",
                                    allowed_channels=[])
    adapter._seen_messages = {}
    adapter._SEEN_TTL = 300
    adapter._SEEN_MAX = 2000
    adapter.session_manager = SessionManager()
    adapter.session_manager.lcm_engine = eng
    adapter.agent_loop = AgentLoop(provider=_ToolCaller(), model="stub-model",
                                   tool_registry=_registry(), max_turns=2)
    adapter.tool_registry = _registry()
    adapter.system_prompt = "sys"
    monkeypatch.setattr(adapter, "_add_reaction", AsyncMock())
    monkeypatch.setattr(adapter, "_remove_reaction", AsyncMock())
    monkeypatch.setattr(adapter, "_save_channel", lambda ch: None)
    say = AsyncMock()

    asyncio.run(adapter._dispatch_to_agent(channel="C1", user="U1", text="build the thing",
                                           ts="1.0", thread_ts=None, say=say))

    reply = say.await_args.kwargs.get("text") or say.await_args.args[0]
    assert "round" in reply.lower() and not reply.startswith("Error")
    [sid] = [s for s in adapter.session_manager._sessions]
    session = adapter.session_manager.get_or_create(sid)
    assert session.messages[0].text == "build the thing"
    assert _tool_use_ids(session.messages) == ["call1", "call2"]
    assert "call2" in " ".join(r.content_json or "" for r in _stored_rows(eng, sid))


def test_discord_turn_at_the_cap_keeps_the_turn(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from prometheus.gateway.config import Platform, PlatformConfig
    from prometheus.gateway.discord import DiscordAdapter
    from prometheus.gateway.platform_base import MessageEvent

    eng = _engine(tmp_path)
    adapter = DiscordAdapter(
        config=PlatformConfig(platform=Platform.DISCORD, token="t"),
        agent_loop=AgentLoop(provider=_ToolCaller(), model="stub-model",
                             tool_registry=_registry(), max_turns=2),
        tool_registry=_registry(), model_name="stub-model", model_provider="stub",
    )
    adapter.session_manager.lcm_engine = eng
    replies: list[str] = []

    async def deliver(_event, _message, text):
        replies.append(text)

    monkeypatch.setattr(adapter, "_add_reaction", AsyncMock())
    monkeypatch.setattr(adapter, "_remove_reaction", AsyncMock())
    monkeypatch.setattr(adapter, "_deliver_reply", deliver)
    event = MessageEvent(chat_id=7, user_id=8, text="build the thing", message_id=9,
                         platform=Platform.DISCORD, raw={"message": None})

    asyncio.run(adapter._dispatch_to_agent(event))

    assert replies and "round" in replies[0].lower() and not replies[0].startswith("Error")
    session = adapter.session_manager.get_or_create(event.session_key())
    assert session.messages[0].text == "build the thing"
    assert _tool_use_ids(session.messages) == ["call1", "call2"]
    stored = _stored_rows(eng, event.session_key())
    assert "call2" in " ".join(r.content_json or "" for r in stored)


def test_rest_chat_at_the_cap_keeps_the_turn(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    eng = _engine(tmp_path)
    mgr = SessionManager()
    mgr.lcm_engine = eng
    app = create_app({}, session_mgr=mgr, agent_loop=AgentLoop(
        provider=_ToolCaller(), model="stub-model", tool_registry=_registry(), max_turns=2))
    # As web/launcher wires it: the route reads the system prompt off the bridge.
    from types import SimpleNamespace
    bridge_ctx = _ctx(_ToolCaller(), 2)
    bridge_ctx.system_prompt = "sys"
    app.state.ws_bridge = SimpleNamespace(loop_context=bridge_ctx)

    response = TestClient(app).post("/api/chat", json={"session_id": "s1", "content": "build"})

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["stop_reason"] == "max_turns_exhausted"
    assert "round" in body["text"].lower()
    session = mgr.get_or_create("web:s1")
    assert _tool_use_ids(session.messages) == ["call1", "call2"]
    assert "call2" in " ".join(r.content_json or "" for r in _stored_rows(eng, "web:s1"))
