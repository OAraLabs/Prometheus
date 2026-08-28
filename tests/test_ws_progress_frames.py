"""Turn liveness on the WS bridge (feat/turn-progress-and-errors).

Three properties, all driven by the 2026-07-24 incident where a live turn was
indistinguishable from a dead daemon:

  * ``agent_progress`` pulses while a turn runs and reports what it is ACTUALLY
    doing (phase / tool / round / chars / elapsed) — the positive liveness
    signal that replaces a client-side "no reply after 30s" guess. It stops
    when the turn ends, so a finished turn never looks busy.
  * ``agent_state`` carries ``session_id``, so a multi-session client can scope
    "thinking" to the right conversation.
  * ``error`` carries ``session_id`` plus structured cause/hint, so a client can
    render the failure INLINE in the chat instead of filing it in a side panel.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from prometheus.web.ws_server import WebSocketBridge


@dataclass(frozen=True)
class AssistantTextDelta:
    text: str


@dataclass(frozen=True)
class ToolExecutionStarted:
    tool_use_id: str
    tool_name: str
    tool_input: dict


@dataclass(frozen=True)
class ToolExecutionCompleted:
    tool_use_id: str
    tool_name: str
    is_error: bool
    output: str


class _FakeSession:
    def __init__(self) -> None:
        self.messages: list = []

    def get_messages(self) -> list:
        return self.messages

    def persist_loop_result(self, original_len: int) -> None:
        pass


class _Recorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]


def _bridge() -> tuple[WebSocketBridge, _Recorder]:
    b = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    b._clients.add(rec)
    return b, rec


async def _wait_until(cond, timeout: float = 3.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached before timeout")
        await asyncio.sleep(0.01)


def _patch_loop(monkeypatch, run_loop) -> None:
    import prometheus.engine.agent_loop as agent_loop_mod

    monkeypatch.setattr(agent_loop_mod, "run_loop", run_loop)


def _fast_progress(monkeypatch, interval: float = 0.02) -> None:
    """Shrink the heartbeat cadence so tests stay fast and deterministic."""
    import prometheus.web.ws_server as ws_mod

    monkeypatch.setattr(ws_mod, "PROGRESS_INTERVAL_SECONDS", interval)


# ---------------------------------------------------------------------------
# agent_progress — the liveness pulse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_pulses_during_a_silent_turn(monkeypatch):
    # The killer case: a turn that emits NOTHING for a long time (slow model
    # thinking). Previously zero frames → clients assumed death.
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        await asyncio.Event().wait()
        yield  # pragma: no cover

    _patch_loop(monkeypatch, run_loop)
    task = asyncio.ensure_future(bridge._run_agent("desktop:p", _FakeSession()))
    await _wait_until(lambda: len(rec.of("agent_progress")) >= 3)

    p = rec.of("agent_progress")[0]["payload"]
    assert p["session_id"] == "desktop:p"
    assert p["phase"] == "thinking"
    assert p["round"] == 1
    assert p["elapsed_s"] >= 0
    # Elapsed advances across pulses — it's a real clock, not a constant.
    assert rec.of("agent_progress")[-1]["payload"]["elapsed_s"] >= p["elapsed_s"]

    bridge.interrupt_turn("desktop:p")
    await task


@pytest.mark.asyncio
async def test_progress_reports_phase_tool_and_chars(monkeypatch):
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield AssistantTextDelta(text="hello"), None
        yield ToolExecutionStarted("c1", "bash", {"cmd": "du -sh"}), None
        await asyncio.Event().wait()

    _patch_loop(monkeypatch, run_loop)
    task = asyncio.ensure_future(bridge._run_agent("desktop:p", _FakeSession()))
    # Wait for a pulse emitted AFTER the tool started.
    await _wait_until(lambda: any(
        f["payload"]["phase"] == "tool" for f in rec.of("agent_progress")
    ))

    p = [f["payload"] for f in rec.of("agent_progress") if f["payload"]["phase"] == "tool"][0]
    assert p["tool_name"] == "bash"      # names the ACTUAL running tool
    assert p["tool_calls"] == 1
    assert p["chars"] == len("hello")    # streamed text so far

    bridge.interrupt_turn("desktop:p")
    await task


@pytest.mark.asyncio
async def test_progress_stops_when_the_turn_finishes(monkeypatch):
    # A heartbeat outliving its turn would report a finished turn as busy.
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield AssistantTextDelta(text="done"), None

    _patch_loop(monkeypatch, run_loop)
    await bridge._run_agent("desktop:p", _FakeSession())
    assert rec.of("chat_done")

    after_done = len(rec.of("agent_progress"))
    await asyncio.sleep(0.15)  # several heartbeat intervals
    assert len(rec.of("agent_progress")) == after_done, "heartbeat outlived the turn"


@pytest.mark.asyncio
async def test_progress_stops_after_an_interrupt(monkeypatch):
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield AssistantTextDelta(text="partial"), None
        await asyncio.Event().wait()

    _patch_loop(monkeypatch, run_loop)
    task = asyncio.ensure_future(bridge._run_agent("desktop:p", _FakeSession()))
    await _wait_until(lambda: len(rec.of("agent_progress")) >= 2)
    bridge.interrupt_turn("desktop:p")
    await task

    after = len(rec.of("agent_progress"))
    await asyncio.sleep(0.15)
    assert len(rec.of("agent_progress")) == after


# ---------------------------------------------------------------------------
# agent_state / error — session scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_state_frames_carry_session_id(monkeypatch):
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield AssistantTextDelta(text="hi"), None

    _patch_loop(monkeypatch, run_loop)
    await bridge._run_agent("telegram:42", _FakeSession())

    states = [(f["payload"]["state"], f["payload"].get("session_id")) for f in rec.of("agent_state")]
    assert ("thinking", "telegram:42") in states
    assert ("idle", "telegram:42") in states


@pytest.mark.asyncio
async def test_error_frame_is_session_scoped_and_actionable(monkeypatch):
    """The live incident, end to end: an Anthropic billing 400 must reach the
    client as a renderable, attributable, human-readable failure."""
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    class _Resp:
        status_code = 400
        text = json.dumps({"error": {"message": "Your credit balance is too low to access the "
                                                "Anthropic API."}})

    class _Req:
        url = type("U", (), {"host": "api.anthropic.com"})()

    class Boom(Exception):
        response = _Resp()
        request = _Req()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        raise Boom("Client error '400 Bad Request' for url 'https://api.anthropic.com/v1/messages'")
        yield  # pragma: no cover

    _patch_loop(monkeypatch, run_loop)
    await bridge._run_agent("desktop:boom", _FakeSession())

    err = rec.of("error")[0]["payload"]
    assert err["session_id"] == "desktop:boom"   # attributable to a chat
    assert err["kind"] == "billing"              # machine-branchable cause
    assert err["provider"] == "anthropic"
    assert err["status"] == 400
    assert "credit balance is too low" in err["message"]   # the real reason
    assert "console.anthropic.com" in err["hint"]          # the next step
    # Still idles out cleanly so the UI isn't left spinning.
    assert ("idle", "desktop:boom") in [
        (f["payload"]["state"], f["payload"].get("session_id")) for f in rec.of("agent_state")
    ]


# ---------------------------------------------------------------------------
# GRAFT-MOBILE-BRIDGE 3a — session_id on tool frames
#
# broadcast() fans out to every client and _run_agent serializes per session,
# so a tool frame carrying no session_id cannot be attributed to a conversation
# by any client. These assert the id is now present on both tool frames.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_frames_carry_session_id(monkeypatch):
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield ToolExecutionStarted("c1", "bash", {"cmd": "ls"}), None
        yield ToolExecutionCompleted("c1", "bash", False, "ok"), None

    _patch_loop(monkeypatch, run_loop)
    await bridge._run_agent("desktop:tools", _FakeSession())

    start = rec.of("tool_call_start")[0]["payload"]
    end = rec.of("tool_call_end")[0]["payload"]
    assert start["session_id"] == "desktop:tools"
    assert end["session_id"] == "desktop:tools"
    # The existing fields are untouched (additive change).
    assert start["call_id"] == "c1" and start["tool_name"] == "bash"
    assert end["success"] is True


# ---------------------------------------------------------------------------
# GRAFT-MOBILE-BRIDGE 3b — row_id on chat_done
#
# The assistant turn's durable rowid was never broadcast; every client
# reinvented a ?since= reconciliation to learn it. persist_loop_result now
# returns it and _run_agent forwards it on chat_done — present when persistence
# yielded one, OMITTED (never null) otherwise, so older clients are unaffected.
# ---------------------------------------------------------------------------


class _RowIdSession(_FakeSession):
    """A session whose persist reports a durable assistant rowid (like the real
    ChatSession.persist_loop_result after this change)."""

    def __init__(self, row_id):
        super().__init__()
        self._row_id = row_id

    def persist_loop_result(self, original_len: int):
        return self._row_id


@pytest.mark.asyncio
async def test_chat_done_carries_row_id_when_persist_reports_one(monkeypatch):
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield AssistantTextDelta(text="done"), None

    _patch_loop(monkeypatch, run_loop)
    await bridge._run_agent("desktop:rk", _RowIdSession(4271))

    done = rec.of("chat_done")[0]["payload"]
    assert done["row_id"] == 4271
    assert done["session_id"] == "desktop:rk"
    assert done["message_id"].startswith("asst-")  # handle still present


@pytest.mark.asyncio
async def test_chat_done_omits_row_id_when_persist_reports_none(monkeypatch):
    # The default _FakeSession.persist_loop_result returns None (older daemon /
    # a turn that persisted no assistant row). The key must be ABSENT, not null:
    # the client's MessageID decoder treats an explicit null as a handle.
    _fast_progress(monkeypatch)
    bridge, rec = _bridge()

    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None):
        yield AssistantTextDelta(text="done"), None

    _patch_loop(monkeypatch, run_loop)
    await bridge._run_agent("desktop:nr", _FakeSession())

    done = rec.of("chat_done")[0]["payload"]
    assert "row_id" not in done
