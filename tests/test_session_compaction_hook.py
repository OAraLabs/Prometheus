"""Sprint 2 (OAra): Session post-ingest LCM compaction hook.

Regression coverage for the config-dark-adjacent wiring gap: LCMEngine.maybe_compact
was CLI-only for the daemon's entire life — every gateway ingested messages into
lcm.db forever and nothing ever summarized. ChatSession._persist_to_lcm now
schedules maybe_compact as a background task after every ingest batch, so every
path that persists (telegram/slack via add_result_messages, web/Beacon/Bridge via
persist_loop_result) triggers the durable-DAG relief valve.

These tests use a stub engine (records calls, no LLM) against the REAL ChatSession
methods — no monkeypatching of the code under test.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.messages import ConversationMessage
from prometheus.engine.session import ChatSession


class StubLCMEngine:
    def __init__(self, compact_error: Exception | None = None):
        self.ingested: list[dict] = []
        self.maybe_compact_calls: list[str] = []
        self._compact_error = compact_error

    def ingest_sync(self, **kw):
        self.ingested.append(kw)

    async def maybe_compact(self, session_id: str):
        self.maybe_compact_calls.append(session_id)
        if self._compact_error is not None:
            raise self._compact_error
        return None


def _msgs():
    return [ConversationMessage.from_user_text("hello world")]


@pytest.mark.asyncio
async def test_persist_loop_result_schedules_maybe_compact():
    engine = StubLCMEngine()
    s = ChatSession("test:compact-hook", lcm_engine=engine)
    s.messages.extend(_msgs())
    s.persist_loop_result(0)
    # the hook is fire-and-forget — yield so the scheduled task runs
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert engine.ingested, "ingest must still happen"
    assert engine.maybe_compact_calls == ["test:compact-hook"]


@pytest.mark.asyncio
async def test_add_result_messages_schedules_maybe_compact():
    engine = StubLCMEngine()
    s = ChatSession("test:compact-hook-2", lcm_engine=engine)
    s.add_result_messages(_msgs(), 0)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert engine.maybe_compact_calls == ["test:compact-hook-2"]


def test_no_event_loop_is_a_quiet_noop():
    """Sync/CLI context: no running loop -> hook must not raise (the CLI path
    awaits maybe_compact itself)."""
    engine = StubLCMEngine()
    s = ChatSession("test:compact-hook-sync", lcm_engine=engine)
    s.messages.extend(_msgs())
    s.persist_loop_result(0)  # would raise RuntimeError if the hook grabbed a loop
    assert engine.ingested
    assert engine.maybe_compact_calls == []


@pytest.mark.asyncio
async def test_compaction_failure_is_loud_but_contained(caplog):
    engine = StubLCMEngine(compact_error=RuntimeError("summarizer exploded"))
    s = ChatSession("test:compact-hook-err", lcm_engine=engine)
    s.messages.extend(_msgs())
    with caplog.at_level("ERROR"):
        s.persist_loop_result(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    assert engine.maybe_compact_calls == ["test:compact-hook-err"]
    assert any("LCM compaction FAILED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_no_engine_no_hook():
    s = ChatSession("test:compact-hook-none", lcm_engine=None)
    s.messages.extend(_msgs())
    s.persist_loop_result(0)  # must not raise
    await asyncio.sleep(0)
