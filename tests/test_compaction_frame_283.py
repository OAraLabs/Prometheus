"""Beacon #283 — the daemon says when a compaction pass is running, and how far along it is.

Once #773 chunks large spans, one pass can be many summarizer calls before the
model call: tens of seconds to minutes. The only frame Beacon got during that
time was the ordinary ``agent_progress`` pulse saying ``thinking``, and the
compactor's SignalBus events (``context_compaction`` …) fire only AFTER a pass,
go to every client as an unscoped ``sentinel_signal``, and Beacon drops those.

So:
  * the compactor reports ``start`` → ``progress`` (one per summarizer call,
    ``chunks_done``/``chunks_total``) → ``done`` | ``failed`` to the turn that
    is running it, plus ``noop`` once when there is nothing it can do;
  * the WS bridge turns those into a session-scoped ``context_compaction``
    frame, and the pulse says ``compacting`` while it runs.

Canned summarizer providers only — no live model.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from prometheus.context.compactor import COMPACTION_REPORTER, ContextCompactor
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.web.ws_server import WebSocketBridge
from tests.support.doubles import register_double

pytestmark = pytest.mark.asyncio


@register_double(
    "compaction283.summarizer",
    replaces="prometheus.providers.base.ModelProvider",
)
class _Summarizer:
    """A short canned summary per call. ``gate`` (when set) holds every call
    until released, so a test can look at the turn while the pass is running.
    ``fail`` returns an empty reply; ``boom`` raises."""

    def __init__(self, *, gate: asyncio.Event | None = None,
                 fail: bool = False, boom: bool = False) -> None:
        self.requests: list = []
        self.gate = gate
        self.fail = fail
        self.boom = boom

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if self.gate is not None:
            await self.gate.wait()
        if self.boom:
            raise RuntimeError("summarizer backend fell over")
        yield ApiTextDeltaEvent(text="" if self.fail else f"summary part {len(self.requests)}.")


def _history(n_turns: int, filler: int = 400, tag: str = "") -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"{tag}U{i:04d} " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"{tag}A{i:04d} " + "y" * filler)]))
    return msgs


def _compactor(provider=None, **kw) -> ContextCompactor:
    defaults = dict(
        provider=provider or _Summarizer(),
        model="local-27b",          # the summarizer's own model…
        effective_limit=3000,       # …and its small window
        cloud_default_limit=20_000,  # the cloud session's window (threshold 16k)
        protect_recent_turns=3,
    )
    defaults.update(kw)
    c = ContextCompactor(**defaults)
    c._envelope._telemetry = None
    return c


class _Reports:
    """A reporter: records ``(stage, payload)`` in arrival order."""

    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    async def __call__(self, stage: str, payload: dict) -> None:
        self.items.append((stage, dict(payload)))

    def stages(self) -> list[str]:
        return [s for s, _ in self.items]


async def _apply_reporting(c: ContextCompactor, msgs: list, reports, **kw) -> list:
    token = COMPACTION_REPORTER.set(reports)
    try:
        return await c.apply(msgs, **kw)
    finally:
        COMPACTION_REPORTER.reset(token)


async def _until(cond, what: str, timeout: float = 3.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        assert asyncio.get_event_loop().time() < deadline, what
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# The compactor's side: what it reports, and when
# ---------------------------------------------------------------------------


async def test_a_chunked_pass_reports_start_progress_per_call_then_done():
    provider = _Summarizer()
    c = _compactor(provider)
    reports = _Reports()
    msgs = _history(90)                                  # ~19k tokens, over 16k

    out = await _apply_reporting(c, msgs, reports, session_id="cloud-s", model="cloud-model")

    calls = len(provider.requests)
    assert calls > 2, "fixture: chunks plus a combine"
    assert reports.stages() == ["start"] + ["progress"] * calls + ["done"]

    start = reports.items[0][1]
    assert start["session_id"] == "cloud-s"
    assert start["chunks_done"] == 0 and start["progress"] == 0.0
    assert start["chunks_total"] == calls, "a span this size needs one combine round"
    assert start["tokens_before"] > start["threshold"] > 0

    progress = [p for s, p in reports.items if s == "progress"]
    assert [p["chunks_done"] for p in progress] == list(range(1, calls + 1))
    fractions = [p["progress"] for p in progress]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert all(0.0 < f <= 1.0 for f in fractions)

    done = reports.items[-1][1]
    assert done["tokens_before"] == start["tokens_before"]
    assert done["tokens_after"] == c.estimate_total("", out)
    assert done["tokens_after"] < done["tokens_before"]
    assert done["duration_ms"] >= 0
    assert done["converged"] is True


async def test_a_one_chunk_pass_reports_one_of_one():
    provider = _Summarizer()
    # A summarizer window twice the session's: the whole span is one request.
    c = _compactor(provider, effective_limit=40_000)
    reports = _Reports()

    await _apply_reporting(c, _history(90), reports, session_id="s", model="cloud-model")

    assert len(provider.requests) == 1
    assert reports.stages() == ["start", "progress", "done"]
    assert reports.items[0][1]["chunks_total"] == 1
    assert reports.items[1][1]["progress"] == 1.0


@register_double(
    "compaction283.long_parts",
    replaces="prometheus.providers.base.ModelProvider",
)
class _LongParts:
    """~300-token part summaries: too many to merge in one combine request."""

    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiTextDeltaEvent(text=f"part {len(self.requests)} " + "p " * 600)


async def test_a_second_combine_round_grows_the_total_and_progress_never_passes_it():
    provider = _LongParts()
    c = _compactor(provider)
    reports = _Reports()

    await _apply_reporting(c, _history(90), reports, session_id="s", model="cloud-model")

    calls = len(provider.requests)
    start = reports.items[0][1]
    assert calls > start["chunks_total"], "fixture: a further combine round was needed"
    assert reports.stages() == ["start"] + ["progress"] * calls + ["done"]
    progress = [p for st, p in reports.items if st == "progress"]
    assert [p["chunks_done"] for p in progress] == list(range(1, calls + 1))
    assert all(p["chunks_done"] <= p["chunks_total"] for p in progress)
    assert progress[-1]["chunks_total"] == calls and progress[-1]["progress"] == 1.0
    assert reports.items[-1][1]["chunks_total"] == calls


async def test_a_reused_summary_reports_nothing():
    # The anchored span re-substitutes its cached summary on every later round,
    # at zero model calls. That is not a pass anyone waits for: no frames, or
    # Beacon would flash "Freed ~68k tokens" on every round of a long session.
    provider = _Summarizer()
    c = _compactor(provider)
    msgs = _history(90)
    await _apply_reporting(c, msgs, _Reports(), session_id="s", model="cloud-model")
    calls = len(provider.requests)

    again = _Reports()
    await _apply_reporting(c, msgs, again, session_id="s", model="cloud-model")

    assert len(provider.requests) == calls, "fixture: the second round is a cache hit"
    assert again.items == []


async def test_a_summarizer_that_returns_nothing_reports_failed():
    c = _compactor(_Summarizer(fail=True))
    reports = _Reports()
    msgs = _history(90)

    out = await _apply_reporting(c, msgs, reports, session_id="s", model="cloud-model")

    assert out is msgs, "fails loud and sends the uncompacted prompt, as before"
    assert reports.stages()[0] == "start"
    assert reports.stages()[-1] == "failed"
    assert "done" not in reports.stages()
    failed = reports.items[-1][1]
    assert failed["reason"]
    assert failed["duration_ms"] >= 0


@register_double(
    "compaction283.wordy",
    replaces="prometheus.providers.base.ModelProvider",
)
class _Wordy:
    """A summary longer than what it replaces."""

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text="w " * 40_000)


async def test_a_summary_not_smaller_than_its_span_reports_failed():
    c = _compactor(_Wordy(), effective_limit=40_000)
    reports = _Reports()
    await _apply_reporting(c, _history(90), reports, session_id="s", model="cloud-model")

    assert reports.stages()[-1] == "failed"
    assert "not smaller" in reports.items[-1][1]["reason"]


async def test_a_summarizer_backend_that_raises_reports_failed():
    # The envelope turns a raising provider into "no summary" (fail loud, send
    # uncompacted); the bar still has to end.
    c = _compactor(_Summarizer(boom=True))
    reports = _Reports()
    msgs = _history(90)

    out = await _apply_reporting(c, msgs, reports, session_id="s", model="cloud-model")

    assert out is msgs
    assert reports.stages() == ["start", "failed"]


async def test_a_pass_that_raises_still_ends_the_bar_and_names_only_the_type(monkeypatch):
    # A start with no end leaves Beacon's bar up forever. The exception keeps
    # propagating (the loop catches it and sends uncompacted); the frame names
    # the exception's TYPE only, never its text (the error frame's rule).
    c = _compactor(_Summarizer())
    real = c._call_summarizer
    calls = 0

    async def second_call_breaks(*a, **k):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("secret-host.internal:8443/path?key=abc")
        return await real(*a, **k)

    monkeypatch.setattr(c, "_call_summarizer", second_call_breaks)
    reports = _Reports()

    with pytest.raises(RuntimeError):
        await _apply_reporting(c, _history(90), reports, session_id="s", model="cloud-model")

    assert reports.stages() == ["start", "progress", "failed"]
    reason = reports.items[-1][1]["reason"]
    assert "RuntimeError" in reason and "secret-host" not in reason


async def test_a_turn_stopped_between_calls_ends_the_bar_and_stays_stopped():
    # Cancelled while the pass is between summarizer calls (here: while the
    # first progress report is being sent). The bar ends with reason
    # "cancelled", and the cancellation keeps propagating: a Stop is a Stop.
    held = asyncio.Event()
    items: list[tuple[str, dict]] = []

    async def reporter(stage, payload):  # noqa: ANN001
        items.append((stage, payload))
        if stage == "progress" and not held.is_set():
            held.set()
            await asyncio.Event().wait()

    c = _compactor(_Summarizer())
    token = COMPACTION_REPORTER.set(reporter)
    try:
        task = asyncio.ensure_future(
            c.apply(_history(90), session_id="s", model="cloud-model"))
    finally:
        COMPACTION_REPORTER.reset(token)
    await asyncio.wait_for(held.wait(), 3.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [s for s, _ in items] == ["start", "progress", "failed"]
    assert items[-1][1]["reason"] == "cancelled"


async def test_a_turn_stopped_during_a_summarizer_call_ends_the_bar_and_stays_stopped():
    # Cancelled INSIDE a summarizer call. Since #787 the LLM envelope re-raises a
    # Stop whatever its on_failure says, so this is the same path as a Stop
    # between calls: the bar ends with reason "cancelled" and the cancellation
    # keeps propagating. Before #787 it surfaced as "summarizer returned
    # empty/failed" and the turn went on to the model call.
    gate = asyncio.Event()
    c = _compactor(_Summarizer(gate=gate))
    reports = _Reports()
    task = asyncio.ensure_future(
        _apply_reporting(c, _history(90), reports, session_id="s", model="cloud-model"))
    await _until(lambda: reports.stages() == ["start"], "the pass never started")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reports.stages() == ["start", "failed"]
    assert reports.items[-1][1]["reason"] == "cancelled"


async def test_nothing_compactable_reports_noop_once_per_stuck_floor():
    c = _compactor(_Summarizer(), effective_limit=4_000)
    huge = [ConversationMessage.from_user_text("HUGE " + "q" * 4 * 4000)]
    reports = _Reports()

    for _ in range(3):
        await _apply_reporting(c, huge, reports, session_id="s")

    assert reports.stages() == ["noop"]
    noop = reports.items[0][1]
    assert noop["session_id"] == "s"
    assert noop["tokens_before"] > noop["threshold"]


async def test_a_reporter_that_raises_never_breaks_the_pass():
    async def broken(stage, payload):  # noqa: ANN001
        raise RuntimeError("socket gone")

    c = _compactor(_Summarizer())
    msgs = _history(90)
    out = await _apply_reporting(c, msgs, broken, session_id="s", model="cloud-model")
    assert out is not msgs and len(out) < len(msgs)


async def test_no_reporter_no_reports_and_the_pass_still_runs():
    c = _compactor(_Summarizer())
    msgs = _history(90)
    assert COMPACTION_REPORTER.get() is None
    out = await c.apply(msgs, session_id="s", model="cloud-model")
    assert len(out) < len(msgs)


# ---------------------------------------------------------------------------
# The WS bridge's side: a session-scoped frame, and the pulse phase
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, messages: list) -> None:
        self.messages = messages

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


def _loop_through(monkeypatch, compactor: ContextCompactor,
                  model_call: asyncio.Event | None = None) -> None:
    """``run_loop`` reduced to the one thing under test: the compactor call the
    real loop makes before each provider call (agent_loop.py, ``apply``).
    *model_call*, when given, holds the "model call" after the pass."""
    import prometheus.engine.agent_loop as agent_loop_mod

    async def run_loop(context, messages, mode="agent", session_id=None,
                       tool_choice=None, surface=None):
        await compactor.apply(messages, session_id=session_id or "", model="cloud-model")
        if model_call is not None:
            await model_call.wait()
        yield AssistantTextDelta("ok"), None

    monkeypatch.setattr(agent_loop_mod, "run_loop", _RUN_LOOP_DOUBLE(run_loop))


_RUN_LOOP_DOUBLE = register_double("compaction283.run_loop",
                                   replaces="prometheus.engine.agent_loop.run_loop")


class ToolExecutionStarted:
    def __init__(self, tool_use_id: str, tool_name: str, tool_input: dict) -> None:
        self.tool_use_id, self.tool_name, self.tool_input = tool_use_id, tool_name, tool_input


class AssistantTextDelta:
    """Named as the loop's event: the bridge dispatches on the type's name."""

    def __init__(self, text: str) -> None:
        self.text = text


async def _subscribe(bridge: WebSocketBridge, ws: _Recorder, session: str) -> None:
    await bridge._handle_client_message(ws, json.dumps(
        {"type": "subscribe", "payload": {"sessions": [session]}}))
    ws.frames.clear()


async def test_ws_frames_run_start_progress_done_in_order_for_the_right_session_only(monkeypatch):
    provider = _Summarizer()
    c = _compactor(provider)          # ONE compactor, shared by every session (as in the daemon)
    _loop_through(monkeypatch, c)
    bridge = WebSocketBridge(loop_context=object())
    watching_a, watching_b, firehose = _Recorder(), _Recorder(), _Recorder()
    for ws in (watching_a, watching_b, firehose):
        bridge._clients.add(ws)
    await _subscribe(bridge, watching_a, "desktop:a")
    await _subscribe(bridge, watching_b, "desktop:b")

    # A compacts (a long history); B's concurrent turn does not (a short one).
    await asyncio.gather(
        bridge._run_agent("desktop:a", _FakeSession(_history(90, tag="A"))),
        bridge._run_agent("desktop:b", _FakeSession(_history(2, tag="B"))),
    )

    frames = watching_a.of("context_compaction")
    calls = len(provider.requests)
    assert calls > 2
    stages = [f["payload"]["stage"] for f in frames]
    assert stages == ["start"] + ["progress"] * calls + ["done"]
    assert all(f["payload"]["session_id"] == "desktop:a" for f in frames)
    # Tied to the turn's assistant bubble, like every other turn frame.
    [done_a] = [f for f in watching_a.of("chat_done")]
    assert {f["payload"]["message_id"] for f in frames} == {done_a["payload"]["message_id"]}
    progress = [f["payload"] for f in frames if f["payload"]["stage"] == "progress"]
    assert [p["chunks_done"] for p in progress] == list(range(1, calls + 1))
    assert all(p["chunks_total"] == calls for p in progress)
    assert frames[-1]["payload"]["tokens_after"] < frames[-1]["payload"]["tokens_before"]
    # Compaction finishes before the model's reply starts streaming.
    order = watching_a.frames
    assert order.index(frames[-1]) < order.index(watching_a.of("chat_delta")[0])

    # The client watching B hears nothing about A's compaction.
    assert watching_b.of("context_compaction") == []
    assert watching_b.of("chat_done"), "fixture: B's own turn did reach its watcher"
    # The firehose sees A's frames and only A's: B never compacted.
    assert {f["payload"]["session_id"] for f in firehose.of("context_compaction")} == {"desktop:a"}


async def test_context_compaction_is_a_session_scoped_frame():
    assert "context_compaction" in WebSocketBridge._SESSION_SCOPED_TYPES


async def test_the_pulse_says_compacting_while_the_pass_runs(monkeypatch):
    import prometheus.web.ws_server as ws_mod

    monkeypatch.setattr(ws_mod, "PROGRESS_INTERVAL_SECONDS", 0.02)
    gate, model_call = asyncio.Event(), asyncio.Event()
    provider = _Summarizer(gate=gate)
    c = _compactor(provider)
    _loop_through(monkeypatch, c, model_call=model_call)
    bridge = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    bridge._clients.add(rec)

    def phases() -> list[str]:
        return [f["payload"]["phase"] for f in rec.of("agent_progress")]

    task = asyncio.ensure_future(bridge._run_agent("desktop:a", _FakeSession(_history(90))))
    try:
        # The pass is held inside its first summarizer call.
        await _until(lambda: "compacting" in phases(), "no 'compacting' pulse during the pass")
        assert provider.requests and not rec.of("chat_done")

        # Release the pass; the turn now waits at the model call.
        gate.set()
        await _until(lambda: any(f["payload"]["stage"] == "done"
                                 for f in rec.of("context_compaction")), "the pass never finished")
        seen = len(rec.of("agent_progress"))
        await _until(lambda: len(rec.of("agent_progress")) >= seen + 3, "no pulses after the pass")
        # Back to "thinking" (waiting on the model) once the pass is over.
        assert phases()[seen:] and all(p == "thinking" for p in phases()[seen:])
    finally:
        gate.set()
        model_call.set()
        await asyncio.wait_for(task, 5.0)
    assert rec.of("chat_done")


async def test_a_nested_runs_pass_reports_into_the_waiting_turn_and_restores_its_phase(monkeypatch):
    # A subagent (or any run inside a tool call) compacts under its OWN session
    # id while the Beacon turn waits on the tool. Its frames belong to the chat
    # that is waiting, and when the pass ends the pulse says "tool" again.
    import prometheus.engine.agent_loop as agent_loop_mod

    c = _compactor(_Summarizer())

    async def run_loop(context, messages, mode="agent", session_id=None,
                       tool_choice=None, surface=None):
        yield ToolExecutionStarted("c1", "agent", {}), None
        await c.apply(messages, session_id="subagent:xyz", model="cloud-model")
        phases_after_pass.append(progress_phase())
        yield AssistantTextDelta("ok"), None

    monkeypatch.setattr(agent_loop_mod, "run_loop", _RUN_LOOP_DOUBLE(run_loop))
    bridge = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    bridge._clients.add(rec)
    phases_after_pass: list[str] = []
    from prometheus.engine import loop_watchdog

    def progress_phase() -> str:
        return loop_watchdog.current_progress()["phase"]

    await bridge._run_agent("desktop:a", _FakeSession(_history(90)))

    frames = rec.of("context_compaction")
    assert frames and {f["payload"]["session_id"] for f in frames} == {"desktop:a"}
    assert phases_after_pass == ["tool"]


async def test_the_reporter_is_gone_after_the_turn(monkeypatch):
    # A compaction run outside any WS turn (Telegram, a background wake) must
    # not report into a finished Beacon turn.
    c = _compactor(_Summarizer())
    _loop_through(monkeypatch, c)
    bridge = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    bridge._clients.add(rec)
    await bridge._run_agent("desktop:a", _FakeSession(_history(2)))
    assert COMPACTION_REPORTER.get() is None
