"""Cross-surface turn serialization on the shared SessionManager lock.

The telegram gateway (audit M6) and the web/WS bridge (2026-08-11
duplicate-rows fix) each serialized turns per session WITHIN their surface —
two private lock maps keyed by the same session ids. A cross-surface pair
never contended: the managed-task completion handler re-engages through
``telegram.inject_turn`` (daemon wiring), and a managed task created from
Beacon targets a ``desktop:*`` session — so the injected turn ran under
telegram's lock while a live WS turn on the SAME session ran under the
bridge's lock, both appending to the shared ``ChatSession.messages``
concurrently (interleaved model rounds, scrambled order). The persistence
watermark caps the durable damage at zero duplicate LCM rows; it does not
prevent the in-memory interleave.

These tests hold both surfaces to ONE lock — ``SessionManager.turn_lock_for``
— in both directions, while keeping the two invariants the per-surface tests
already pin: different sessions never contend, and lock helpers stay
resilient to ``__new__``-built / stub-wired instances.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from prometheus.engine.session import SessionManager
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.tools.base import ToolRegistry
from prometheus.web.ws_server import WebSocketBridge


def _adapter(loop, session_manager) -> TelegramAdapter:
    return TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
        agent_loop=loop,
        tool_registry=ToolRegistry(),
        session_manager=session_manager,
    )


# ---------------------------------------------------------------------------
# The shared lock itself
# ---------------------------------------------------------------------------


def test_turn_lock_for_is_one_lock_per_session():
    mgr = SessionManager()
    lock = mgr.turn_lock_for("desktop:x")
    assert isinstance(lock, asyncio.Lock)
    assert mgr.turn_lock_for("desktop:x") is lock
    assert mgr.turn_lock_for("desktop:y") is not lock


def test_turn_lock_for_survives_new_built_manager():
    # Same __new__ resilience the surface-local helpers had.
    mgr = SessionManager.__new__(SessionManager)
    lock = mgr.turn_lock_for("s")
    assert isinstance(lock, asyncio.Lock)
    assert mgr.turn_lock_for("s") is lock


def test_both_surface_shims_resolve_to_the_shared_lock():
    mgr = SessionManager()
    adapter = _adapter(object(), mgr)
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=object())

    lock = mgr.turn_lock_for("desktop:x")
    assert adapter._turn_lock_for("desktop:x") is lock
    assert bridge._turn_lock_for("desktop:x") is lock


def test_shims_fall_back_to_local_maps_without_a_real_manager():
    # __new__-built adapter (bypasses __init__ entirely): local lazy map.
    gw = TelegramAdapter.__new__(TelegramAdapter)
    tg_lock = gw._turn_lock_for("telegram:1")
    assert isinstance(tg_lock, asyncio.Lock)
    assert gw._turn_lock_for("telegram:1") is tg_lock

    # Bridge with no manager, and bridge with a non-SessionManager stub
    # (the paperclip tests wire fakes): both keep the bridge-local map.
    for bridge in (
        WebSocketBridge(loop_context=object()),
        WebSocketBridge(session_mgr=SimpleNamespace(), loop_context=object()),
    ):
        ws_lock = bridge._turn_lock_for("desktop:s")
        assert isinstance(ws_lock, asyncio.Lock)
        assert bridge._turn_lock_for("desktop:s") is ws_lock


# ---------------------------------------------------------------------------
# Cross-surface serialization on one ChatSession
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_turn_waits_for_telegram_turn_on_same_session(monkeypatch):
    """Telegram holds the session; a bridge turn on the SAME session queues."""
    import prometheus.engine.agent_loop as agent_loop_mod

    order: list[str] = []
    tg_release = asyncio.Event()
    tg_inside = asyncio.Event()

    class _TgLoop:
        async def run_async(self, **kw):  # noqa: ANN003
            order.append("tg:enter")
            tg_inside.set()
            await tg_release.wait()
            order.append("tg:exit")
            return SimpleNamespace(text="reply", messages=list(kw["messages"]))

    async def fake_run_loop(context, messages, mode="agent", session_id=None,
                            tool_choice=None):
        order.append("ws:enter")
        order.append("ws:exit")
        if False:  # pragma: no cover — makes this an async generator
            yield

    monkeypatch.setattr(agent_loop_mod, "run_loop", fake_run_loop)
    monkeypatch.setattr(
        "prometheus.engine.honesty.evaluate_and_record", lambda *a, **k: None
    )

    mgr = SessionManager()
    adapter = _adapter(_TgLoop(), mgr)
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=object())

    sid = "desktop:x"
    session = mgr.get_or_create(sid)  # ONE ChatSession, used by both surfaces

    t1 = asyncio.create_task(
        adapter._run_agent_turn(session, "from telegram", session_id=sid)
    )
    await tg_inside.wait()  # telegram is inside run_async, holding the lock

    # A live bridge turn on the SAME session starts now...
    t2 = asyncio.create_task(bridge._run_agent(sid, session))
    await asyncio.sleep(0.05)  # ...and must NOT have entered run_loop yet.
    assert order == ["tg:enter"], (
        "bridge turn ran while the telegram turn still held the session"
    )

    tg_release.set()
    await asyncio.gather(t1, t2)
    assert order == ["tg:enter", "tg:exit", "ws:enter", "ws:exit"]


@pytest.mark.asyncio
async def test_injected_turn_waits_for_bridge_turn_on_same_session(monkeypatch):
    """The reverse direction, through the REAL re-engagement entrypoint:
    a bridge turn holds the session; ``inject_turn`` (the managed-task
    completion path) on the SAME session queues behind it."""
    import prometheus.engine.agent_loop as agent_loop_mod

    order: list[str] = []
    ws_release = asyncio.Event()
    ws_inside = asyncio.Event()

    class _TgLoop:
        async def run_async(self, **kw):  # noqa: ANN003
            order.append("tg:enter")
            order.append("tg:exit")
            return SimpleNamespace(text="reply", messages=list(kw["messages"]))

    async def fake_run_loop(context, messages, mode="agent", session_id=None,
                            tool_choice=None):
        order.append("ws:enter")
        ws_inside.set()
        await ws_release.wait()
        order.append("ws:exit")
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(agent_loop_mod, "run_loop", fake_run_loop)
    monkeypatch.setattr(
        "prometheus.engine.honesty.evaluate_and_record", lambda *a, **k: None
    )

    mgr = SessionManager()
    adapter = _adapter(_TgLoop(), mgr)
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=object())

    sid = "desktop:x"
    session = mgr.get_or_create(sid)

    t1 = asyncio.create_task(bridge._run_agent(sid, session))
    await ws_inside.wait()  # the bridge turn holds the session lock

    # Managed-task re-engagement lands on the SAME desktop:* session
    # (inject_turn resolves the session through the SHARED manager)...
    t2 = asyncio.create_task(adapter.inject_turn(sid, "task done"))
    await asyncio.sleep(0.05)  # ...and must be QUEUED, not running.
    assert order == ["ws:enter"], (
        "injected turn ran while the bridge turn still held the session"
    )

    ws_release.set()
    await asyncio.gather(t1, t2)
    assert order == ["ws:enter", "ws:exit", "tg:enter", "tg:exit"]


@pytest.mark.asyncio
async def test_different_sessions_stay_concurrent_across_surfaces(monkeypatch):
    """Sharing the lock map must not over-serialize: a telegram turn parked
    on desktop:a cannot delay a bridge turn on desktop:b."""
    import prometheus.engine.agent_loop as agent_loop_mod

    order: list[str] = []
    tg_release = asyncio.Event()
    tg_inside = asyncio.Event()

    class _TgLoop:
        async def run_async(self, **kw):  # noqa: ANN003
            order.append("tg:enter")
            tg_inside.set()
            await tg_release.wait()
            order.append("tg:exit")
            return SimpleNamespace(text="reply", messages=list(kw["messages"]))

    async def fake_run_loop(context, messages, mode="agent", session_id=None,
                            tool_choice=None):
        order.append(f"ws:exit:{session_id}")
        if False:  # pragma: no cover
            yield

    monkeypatch.setattr(agent_loop_mod, "run_loop", fake_run_loop)
    monkeypatch.setattr(
        "prometheus.engine.honesty.evaluate_and_record", lambda *a, **k: None
    )

    mgr = SessionManager()
    adapter = _adapter(_TgLoop(), mgr)
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=object())

    sa = mgr.get_or_create("desktop:a")
    sb = mgr.get_or_create("desktop:b")

    t1 = asyncio.create_task(
        adapter._run_agent_turn(sa, "blocks", session_id="desktop:a")
    )
    await tg_inside.wait()  # desktop:a is held by the telegram turn

    # desktop:b must run to completion while desktop:a is still parked.
    await bridge._run_agent("desktop:b", sb)
    assert "ws:exit:desktop:b" in order
    assert "tg:exit" not in order  # desktop:a still held up

    tg_release.set()
    await t1
    assert "tg:exit" in order
