"""Per-session turn serialization (audit M6).

A managed-task completion calling ``inject_turn`` can fire while a user turn is
mid-flight on the SAME session. Both paths funnel through ``_run_agent_turn`` and
append to ``session.messages``; run concurrently they interleave, scrambling
message order and duplicating ``turn_index``. The fix takes a per-session lock so
turns on one session serialize — while turns on different sessions still run
concurrently.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.tools.base import ToolRegistry


def _adapter(loop) -> TelegramAdapter:
    return TelegramAdapter(
        config=PlatformConfig(platform=Platform.TELEGRAM, token="test"),
        agent_loop=loop,
        tool_registry=ToolRegistry(),
    )


@pytest.mark.asyncio
async def test_same_session_turns_serialize(monkeypatch):
    order: list[str] = []
    release_first = asyncio.Event()
    first_inside = asyncio.Event()

    class _SerialLoop:
        async def run_async(self, **kw):  # noqa: ANN003
            order.append("enter")
            if len(order) == 1:  # the first turn blocks while holding the lock
                first_inside.set()
                await release_first.wait()
            order.append("exit")
            return SimpleNamespace(text="reply", messages=list(kw["messages"]))

    adapter = _adapter(_SerialLoop())
    # Keep the honesty post-processing out of the way of the timing assertions.
    monkeypatch.setattr(
        "prometheus.engine.honesty.evaluate_and_record", lambda *a, **k: None
    )

    sid = "telegram:1"
    session = adapter.session_manager.get_or_create(sid)

    t1 = asyncio.create_task(
        adapter._run_agent_turn(session, "first", session_id=sid)
    )
    await first_inside.wait()  # t1 is inside run_async, holding the session lock

    # A re-engagement turn on the SAME session starts now...
    t2 = asyncio.create_task(
        adapter._run_agent_turn(
            session, "second", session_id=sid,
            provenance="task_supervisor", is_trusted=False,
        )
    )
    await asyncio.sleep(0.02)  # ...and must NOT have entered run_async yet.
    assert order == ["enter"], "second turn ran before the first released the lock"

    release_first.set()
    await asyncio.gather(t1, t2)
    # Fully serialized: one turn finished entirely before the next began.
    assert order == ["enter", "exit", "enter", "exit"]


@pytest.mark.asyncio
async def test_different_sessions_do_not_block_each_other(monkeypatch):
    order: list[str] = []
    release = asyncio.Event()
    one_inside = asyncio.Event()

    class _Loop:
        async def run_async(self, **kw):  # noqa: ANN003
            sid = kw["session_id"]
            order.append(f"enter:{sid}")
            if sid == "telegram:1":
                one_inside.set()
                await release.wait()
            order.append(f"exit:{sid}")
            return SimpleNamespace(text="reply", messages=list(kw["messages"]))

    adapter = _adapter(_Loop())
    monkeypatch.setattr(
        "prometheus.engine.honesty.evaluate_and_record", lambda *a, **k: None
    )

    s1 = adapter.session_manager.get_or_create("telegram:1")
    s2 = adapter.session_manager.get_or_create("telegram:2")

    t1 = asyncio.create_task(
        adapter._run_agent_turn(s1, "blocks", session_id="telegram:1")
    )
    await one_inside.wait()  # session 1 is blocked inside run_async

    # Session 2 must run to completion while session 1 is still blocked.
    await adapter._run_agent_turn(s2, "free", session_id="telegram:2")
    assert "exit:telegram:2" in order
    assert "exit:telegram:1" not in order  # session 1 still held up

    release.set()
    await t1
    assert "exit:telegram:1" in order
