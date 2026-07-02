"""_run_agent fail-loud (#74-adjacent follow-up).

The bridge's broad except used to emit ONLY a WS error frame — no server-side
log, no telemetry — which masked the wire-contract interface drift (a TypeError
from a run_loop signature change surfaced as "the turn silently produced
nothing"). Now a _run_agent failure must: (1) log the full traceback,
(2) record a silent_failure telemetry row, (3) still broadcast the error frame,
(4) still reset agent state to idle.
"""

from __future__ import annotations

import asyncio
import logging

from prometheus.engine.session import SessionManager
from prometheus.web.ws_server import WebSocketBridge
from tests.support.doubles import register_double


@register_double("run_agent_loud.exploding_run_loop", replaces="prometheus.engine.agent_loop.run_loop")
class _ExplodingRunLoop:
    """Matches the real run_loop call shape, then raises — the drift-class error."""

    def __call__(self, ctx, messages, *, mode="agent", session_id=None, tool_choice=None):
        raise TypeError("simulated interface drift")


def test_run_agent_failure_is_loud(monkeypatch, caplog):
    import prometheus.engine.agent_loop as al

    monkeypatch.setattr(al, "run_loop", _ExplodingRunLoop())

    recorded: list[tuple] = []

    class _Handle:
        def record_silent_failure(self, subsystem, operation, exc, context=None):
            recorded.append((subsystem, operation, type(exc).__name__, context or {}))

    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "get_telemetry_handle", lambda: _Handle())

    frames: list[dict] = []
    state = {"state": "thinking"}
    bridge = WebSocketBridge(session_mgr=SessionManager(), loop_context=object(), agent_state_ref=state)

    async def cap(ev):
        frames.append(ev)

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create("s")

    with caplog.at_level(logging.ERROR, logger="prometheus.web.ws_server"):
        asyncio.run(bridge._run_agent("s", session))

    # (1) server-side log with the traceback — visible in the journal.
    assert any("_run_agent failed" in r.message and r.exc_info for r in caplog.records)
    # (2) telemetry silent_failure row.
    assert recorded == [("web_bridge", "_run_agent", "TypeError", {"session_id": "s", "mode": "agent"})]
    # (3) the WS error frame still goes out.
    assert any(f.get("type") == "error" and "drift" in f["payload"]["message"] for f in frames)
    # (4) state resets to idle.
    assert state["state"] == "idle"
