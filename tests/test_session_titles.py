"""Session display titles (GRAFT-MOBILE-BRIDGE 7).

A session id is <gateway>:<uuid> and carries no name anywhere in the daemon —
usable on a wide desktop sidebar, unusable as a phone list. These pin the
title surface end to end:

  * store: set/get/clear round-trip, and list_sessions carries the title;
  * REST: GET /api/sessions exposes ``title`` (null until one exists),
    PUT /api/sessions/{id}/title renames, clears, validates, and clips;
  * generation: clip_title obeys the ≤48-char word-boundary contract,
    first_exchange finds the first real user→assistant pair, and
    maybe_title_session fills ABSENCE ONLY — a manual rename is never
    overwritten, and a failing provider degrades to "no title".
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.session_titles import (
    MAX_TITLE_CHARS,
    clip_title,
    first_exchange,
    maybe_title_session,
)
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart


def _store(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


def _seed(store: LCMConversationStore, sid: str) -> None:
    store.insert_message(MessagePart(role="user", content="hello", session_id=sid))


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def test_title_round_trip_and_clear(tmp_path):
    store = _store(tmp_path)
    assert store.get_session_title("s1") is None
    store.set_session_title("s1", "Fix the build")
    assert store.get_session_title("s1") == "Fix the build"
    store.set_session_title("s1", "Fix the build, again")
    assert store.get_session_title("s1") == "Fix the build, again"
    # Blank clears the row — present/absent stays a clean signal.
    store.set_session_title("s1", "   ")
    assert store.get_session_title("s1") is None


def test_list_sessions_carries_the_title(tmp_path):
    store = _store(tmp_path)
    _seed(store, "telegram:1")
    _seed(store, "desktop:2")
    store.set_session_title("telegram:1", "Grocery planning")

    rows = {r["session_id"]: r for r in store.list_sessions()}
    assert rows["telegram:1"]["title"] == "Grocery planning"
    assert rows["desktop:2"]["title"] is None


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client_and_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = _store(tmp_path)

    class _Lcm:
        conversation_store = store

    app = create_app({})
    app.state.lcm_engine = _Lcm()
    return TestClient(app), store


def test_get_sessions_exposes_title(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:t")
    store.set_session_title("desktop:t", "Reactor maintenance")

    rows = client.get("/api/sessions").json()
    row = next(r for r in rows if r["session_id"] == "desktop:t")
    assert row["title"] == "Reactor maintenance"


def test_get_sessions_title_is_null_until_one_exists(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:untitled")
    rows = client.get("/api/sessions").json()
    row = next(r for r in rows if r["session_id"] == "desktop:untitled")
    assert row["title"] is None


def test_put_title_renames_and_clears(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:r")

    resp = client.put("/api/sessions/desktop:r/title", json={"title": "  Ship it  "})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "session_id": "desktop:r", "title": "Ship it"}
    assert store.get_session_title("desktop:r") == "Ship it"

    # Empty title = reset to automatic (clears the row).
    resp = client.put("/api/sessions/desktop:r/title", json={"title": ""})
    assert resp.status_code == 200
    assert resp.json()["title"] is None
    assert store.get_session_title("desktop:r") is None


def test_put_title_clips_to_the_contract(client_and_store):
    client, store = client_and_store
    long = "a word " * 30
    resp = client.put("/api/sessions/desktop:c/title", json={"title": long})
    stored = store.get_session_title("desktop:c")
    assert resp.status_code == 200
    assert stored is not None and len(stored) <= MAX_TITLE_CHARS + 1  # +1 for the ellipsis


def test_put_title_rejects_non_string(client_and_store):
    client, _ = client_and_store
    assert client.put("/api/sessions/x/title", json={"title": 7}).status_code == 400


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_clip_title_word_boundary_and_quotes():
    assert clip_title('  "Fix   the build"  ') == "Fix the build"
    long = "planning the greenhouse irrigation schedule for the spring beds"
    clipped = clip_title(long)
    assert len(clipped) <= MAX_TITLE_CHARS + 1
    assert clipped.endswith("…")
    assert not clipped[:-1].endswith(" ")


class _Msg:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


def test_first_exchange_skips_empty_and_tool_noise():
    msgs = [
        _Msg("user", ""),               # empty send — not the opener
        _Msg("user", "plan my week"),
        _Msg("assistant", ""),          # tool_use-only turn (text is '')
        _Msg("assistant", "Here is a plan."),
    ]
    assert first_exchange(msgs) == ("plan my week", "Here is a plan.")
    assert first_exchange([_Msg("user", "hi")]) is None  # no reply yet


class _FakeProvider:
    """Streams one complete-message event carrying the canned title."""

    def __init__(self, title: str) -> None:
        self._title = title

    def stream_message(self, request):
        from prometheus.providers.base import ApiTextDeltaEvent

        async def _gen():
            yield ApiTextDeltaEvent(text=self._title)

        return _gen()


class _BoomProvider:
    def stream_message(self, request):
        async def _gen():
            raise RuntimeError("provider down")
            yield  # pragma: no cover

        return _gen()


def _run(coro):
    return asyncio.run(coro)


def test_generation_fills_absence(tmp_path):
    store = _store(tmp_path)
    msgs = [_Msg("user", "plan my week"), _Msg("assistant", "Here is a plan.")]
    _run(maybe_title_session(store, _FakeProvider("Weekly planning"), "m", "s1", msgs))
    assert store.get_session_title("s1") == "Weekly planning"


def test_generation_never_overwrites(tmp_path):
    store = _store(tmp_path)
    store.set_session_title("s1", "My name, chosen by me")
    msgs = [_Msg("user", "x"), _Msg("assistant", "y")]
    _run(maybe_title_session(store, _FakeProvider("Machine name"), "m", "s1", msgs))
    assert store.get_session_title("s1") == "My name, chosen by me"


def test_generation_failure_degrades_to_no_title(tmp_path):
    store = _store(tmp_path)
    msgs = [_Msg("user", "x"), _Msg("assistant", "y")]
    _run(maybe_title_session(store, _BoomProvider(), "m", "s1", msgs))
    assert store.get_session_title("s1") is None


def test_generation_skips_sessions_with_no_exchange(tmp_path):
    store = _store(tmp_path)
    _run(maybe_title_session(store, _FakeProvider("nope"), "m", "s1",
                             [_Msg("user", "only me")]))
    assert store.get_session_title("s1") is None


# --------------------------------------------------------------------------- #
# The bridge hook: fire-and-forget that actually survives
# --------------------------------------------------------------------------- #


def test_bridge_schedules_title_task_with_a_strong_ref(tmp_path):
    """create_task results the loop holds only weakly can be GC'd mid-flight —
    the standard fire-and-forget trap. The bridge must retain the task until
    it completes (and then let it go), and the task must run detached from the
    turn: it is scheduled after chat_done, so the turn never waits on it."""
    from types import SimpleNamespace

    from prometheus.web.ws_server import WebSocketBridge

    store = _store(tmp_path)

    class _Lcm:
        conversation_store = store

    session = SimpleNamespace(
        lcm_engine=_Lcm(),
        messages=[_Msg("user", "plan my week"), _Msg("assistant", "Here is a plan.")],
    )
    bridge = WebSocketBridge(
        loop_context=SimpleNamespace(provider=_FakeProvider("Weekly planning"), model="m")
    )

    async def _drive():
        bridge._schedule_session_title("s1", session)
        assert bridge._bg_tasks, "task must be strongly referenced while in flight"
        await asyncio.gather(*bridge._bg_tasks)

    asyncio.run(_drive())
    assert store.get_session_title("s1") == "Weekly planning"
    assert not bridge._bg_tasks, "done-callback must release the reference"


def test_bridge_hook_is_a_noop_without_provider_or_store(tmp_path):
    from types import SimpleNamespace

    from prometheus.web.ws_server import WebSocketBridge

    session = SimpleNamespace(lcm_engine=None, messages=[])
    bridge = WebSocketBridge(loop_context=object())

    async def _drive():
        bridge._schedule_session_title("s1", session)
        assert not bridge._bg_tasks

    asyncio.run(_drive())


# --------------------------------------------------------------------------- #
# The shared scheduler (one implementation for bridge + telegram)
# --------------------------------------------------------------------------- #


def test_schedule_retains_then_releases(tmp_path):
    from prometheus.engine.session_titles import schedule

    store = _store(tmp_path)
    tasks: set = set()

    async def _drive():
        schedule(tasks, store=store, provider=_FakeProvider("Named"), model="m",
                 session_id="s1",
                 messages=[_Msg("user", "hi"), _Msg("assistant", "hello")])
        assert tasks, "task must be strongly referenced while in flight"
        await asyncio.gather(*tasks)

    asyncio.run(_drive())
    assert store.get_session_title("s1") == "Named"
    assert not tasks, "done-callback must release the reference"


def test_schedule_is_a_noop_without_store_or_provider(tmp_path):
    from prometheus.engine.session_titles import schedule

    tasks: set = set()

    async def _drive():
        schedule(tasks, store=None, provider=_FakeProvider("x"), model="m",
                 session_id="s1", messages=[])
        schedule(tasks, store=_store(tmp_path), provider=None, model="m",
                 session_id="s1", messages=[])
        assert not tasks

    asyncio.run(_drive())


def test_schedule_off_the_loop_never_raises(tmp_path):
    from prometheus.engine.session_titles import schedule

    tasks: set = set()
    # No running event loop here — a title is a nicety; this must not raise.
    schedule(tasks, store=_store(tmp_path), provider=_FakeProvider("x"),
             model="m", session_id="s1", messages=[])
    assert not tasks


# --------------------------------------------------------------------------- #
# Telegram turns auto-title (GRAFT 7 follow-up)
# --------------------------------------------------------------------------- #


def test_telegram_turn_titles_the_session(tmp_path):
    from types import SimpleNamespace

    from prometheus.engine.session import ChatSession, SessionManager
    from prometheus.gateway.config import Platform, PlatformConfig
    from prometheus.gateway.telegram import TelegramAdapter
    from prometheus.tools.base import ToolRegistry

    store = _store(tmp_path)

    class _Lcm:
        conversation_store = store

    class _Loop:
        _provider = _FakeProvider("Weekly planning")
        _model = "m"

        async def run_async(self, **kw):  # noqa: ANN003
            from prometheus.engine.messages import ConversationMessage, TextBlock

            return SimpleNamespace(
                text="Here is a plan.",
                messages=list(kw["messages"]) + [
                    ConversationMessage(role="assistant",
                                        content=[TextBlock(text="Here is a plan.")])
                ],
            )

    adapter = TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
        agent_loop=_Loop(),
        tool_registry=ToolRegistry(),
        session_manager=SessionManager(),
    )
    session = ChatSession("telegram:42", lcm_engine=_Lcm())

    async def _drive():
        await adapter._run_agent_turn(session, "plan my week",
                                      session_id="telegram:42")
        # The title task is fire-and-forget; drain it before asserting.
        await asyncio.gather(*adapter._bg_tasks)

    asyncio.run(_drive())
    assert store.get_session_title("telegram:42") == "Weekly planning"


def test_telegram_turn_never_overwrites_a_manual_title(tmp_path):
    from types import SimpleNamespace

    from prometheus.engine.session import ChatSession, SessionManager
    from prometheus.gateway.config import Platform, PlatformConfig
    from prometheus.gateway.telegram import TelegramAdapter
    from prometheus.tools.base import ToolRegistry

    store = _store(tmp_path)
    store.set_session_title("telegram:42", "My chosen name")

    class _Lcm:
        conversation_store = store

    class _Loop:
        _provider = _FakeProvider("Machine name")
        _model = "m"

        async def run_async(self, **kw):  # noqa: ANN003
            from prometheus.engine.messages import ConversationMessage, TextBlock

            return SimpleNamespace(
                text="ok",
                messages=list(kw["messages"]) + [
                    ConversationMessage(role="assistant",
                                        content=[TextBlock(text="ok")])
                ],
            )

    adapter = TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
        agent_loop=_Loop(),
        tool_registry=ToolRegistry(),
        session_manager=SessionManager(),
    )
    session = ChatSession("telegram:42", lcm_engine=_Lcm())

    async def _drive():
        await adapter._run_agent_turn(session, "x", session_id="telegram:42")
        await asyncio.gather(*adapter._bg_tasks)

    asyncio.run(_drive())
    assert store.get_session_title("telegram:42") == "My chosen name"
