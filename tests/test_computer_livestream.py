"""The cockpit — a live action log and a Stop that works from any Beacon.

computer-use v1.1 PR 6 (design §5.2). Every step a person's desktop task
takes is visible on a phone as it happens; Stop works from Beacon desktop,
Beacon iOS and Telegram and halts before the next action; and nothing on the
wire or at rest carries an element token, a pid or window id, a snapshot id,
the label of a row the chooser did NOT pick, or the typed text — outside the
one live approval frame whose whole point is to show it (design §5.2.2).

* Layer 2 — the durable log: ``computer_*`` SignalBus kinds, declared ONCE by
  the emitter (``COMPUTER_FRAME_KINDS``) and promoted by ws_server from that
  tuple, exactly as ``CODING_FRAME_KINDS``. Both pinning tests are copied:
  an unpromoted kind ships as ``sentinel_signal`` and every client gate on
  ``type`` misses it — the iOS "dropped frame" bug.
* Layer 1 — the task renders in today's chat timeline: ``tool_call_start`` /
  ``tool_call_end`` with ``origin: "user_task"``, ``agent_progress`` while it
  runs, ``chat_done`` only when no chat turn is live, never
  ``turn_completed``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_computer_door import (  # noqa: F401 - _linux_site_rule is an autouse fixture
    APP,
    PERSON,
    SESSION,
    TYPED,
    _answer,
    _bind,
    _linux_site_rule,
    _next_pending,
    _rig,
    _start,
)

FORBIDDEN_KEYS = {"element_token", "snapshot_id", "pid", "window_id"}
#: Labels of rows the scripted chooser never picks in these runs, as app
#: text renders (``Element.describe`` quotes the label). The binding sentence
#: names "Send" in its OWN words, which is not app text.
UNCHOSEN = ("'Send'", "'File'")


def _bus(tmp_path):
    from prometheus.sentinel.signals import SignalBus
    from prometheus.telemetry.tracker import ToolCallTelemetry

    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    bus = SignalBus(telemetry=tel)
    seen: list[tuple[str, dict]] = []

    async def spy(signal):
        seen.append((signal.kind, signal.payload))

    bus.subscribe("*", spy)
    return bus, tel, seen


class _Recorder:
    """A fake WS client: every frame the bridge broadcasts."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]


def _bridge(runner):
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    bridge._clients.add(rec)
    bridge.computer_runner = runner
    return bridge, rec


def _live(rig, tmp_path, **kw):
    from prometheus.computer.livestream import ComputerLiveStream

    bus, tel, seen = _bus(tmp_path)
    bridge, rec = _bridge(rig.runner)
    bus.subscribe("*", bridge._on_signal)
    rig.channel.signal_bus = bus
    rig.runner.live = ComputerLiveStream(bus, bridge=bridge, telemetry=tel, **kw)
    return SimpleNamespace(bus=bus, tel=tel, seen=seen, bridge=bridge, rec=rec)


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}.{k}", k, v
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _violations(payload, *, allow_text: bool = False) -> list[str]:
    bad = []
    for where, key, value in _walk(payload):
        if key in FORBIDDEN_KEYS:
            bad.append(f"{where}: forbidden key")
        if isinstance(value, str):
            if not allow_text and TYPED in value:
                bad.append(f"{where}: the typed text")
            if "tok-" in value:
                bad.append(f"{where}: an element token")
            for label in UNCHOSEN:
                if label in value:
                    bad.append(f"{where}: an unchosen row's label {label!r}")
    return bad


# ── THE TWO PINNING TESTS, COPIED FROM CODING ───────────────────────────────

def test_computer_frame_kinds_matches_every_emit_site():
    from prometheus.computer import livestream
    from prometheus.computer.livestream import COMPUTER_FRAME_KINDS

    source = Path(livestream.__file__).read_text(encoding="utf-8")
    emitted: set[str] = set()
    non_literal = 0
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_emit"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            emitted.add(node.args[0].value)
        else:
            non_literal += 1
    assert emitted, "found no self._emit(...) call sites"
    assert non_literal == 0
    assert emitted == set(COMPUTER_FRAME_KINDS), (
        f"emitted but not declared: {sorted(emitted - set(COMPUTER_FRAME_KINDS))}; "
        f"declared but not emitted: {sorted(set(COMPUTER_FRAME_KINDS) - emitted)}")


async def test_every_computer_stream_kind_is_promoted():
    from prometheus.computer.livestream import COMPUTER_FRAME_KINDS
    from prometheus.web.ws_server import WebSocketBridge

    for kind in COMPUTER_FRAME_KINDS:
        captured: list[dict] = []
        bridge = WebSocketBridge()

        async def fake_broadcast(event, _c=captured):
            _c.append(event)

        bridge.broadcast = fake_broadcast
        signal = SimpleNamespace(kind=kind, payload={"session_id": SESSION, "x": 1},
                                 timestamp=100.0, source="computer")
        await bridge._on_signal(signal)
        assert [e["type"] for e in captured] == [kind]
        assert captured[0]["payload"] == {"session_id": SESSION, "x": 1}


# ── THE ACTION LOG ──────────────────────────────────────────────────────────

async def test_a_task_streams_its_action_log(tmp_path):
    rig = _rig(tmp_path, ["click-0", "set-2"])
    live = _live(rig, tmp_path)
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    await _answer(rig, ["approve"])
    await rig.runner.wait(task.task_id, timeout=10)
    kinds = [k for k, _ in live.seen if k.startswith("computer_")]
    assert kinds[0] == "computer_binding"
    assert "computer_task_started" in kinds and kinds[-1] in (
        "computer_task_ended", "computer_binding")
    steps = [p for k, p in live.seen if k == "computer_step"]
    statuses = [s["status"] for s in steps]
    assert statuses.count("executed") == 2
    assert "awaiting_approval" in statuses
    waiting = next(s for s in steps if s["status"] == "awaiting_approval")
    assert waiting["approval_request_id"]
    done = [s for s in steps if s["status"] == "executed"]
    assert [s["consent"] for s in done] == ["binding", "prompt"]
    assert [s["seq"] for s in steps] == sorted({s["seq"] for s in steps}), "seq de-dupes"
    for s in steps:
        assert s["session_id"] == SESSION and s["task_id"] == task.task_id
        assert s["action"] is None or s["action"]["app_text"] is True
    ended = [p for k, p in live.seen if k == "computer_task_ended"][0]
    assert ended["outcome"] == "done" and ended["steps"] == 2 and ended["approvals"] == 1
    started = [p for k, p in live.seen if k == "computer_task_started"][0]
    assert started["text_chars"] == len(TYPED)
    # And every one of them reached the socket as its own frame type.
    for kind in set(kinds):
        assert kind in live.rec.types()


async def test_the_content_policy_holds_emitted_persisted_and_backfilled(
        tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.telemetry import tracker
    from prometheus.web.server import create_app

    rig = _rig(tmp_path, ["click-0", "set-2"])
    live = _live(rig, tmp_path)
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    answered = await _answer(rig, ["approve"])
    await rig.runner.wait(task.task_id, timeout=10)

    # The ONE carve-out: the live approval frame shows what is approved.
    pending = [p for k, p in live.seen if k == "approval_pending"]
    assert pending and TYPED in json.dumps(pending[0]), "consent must see the text"
    assert answered

    # Everything else emitted on the bus, and every frame on the socket.
    for kind, payload in live.seen:
        assert not _violations(payload, allow_text=(kind == "approval_pending")), (
            kind, _violations(payload, allow_text=(kind == "approval_pending")))
    for frame in live.rec.frames:
        allow = frame["type"] == "approval_pending"
        assert not _violations(frame, allow_text=allow), (frame["type"], frame)

    # At rest: signal_events holds NO typed text, the approval included.
    rows = live.tel.signal_events_since(limit=500)
    assert rows
    for row in rows:
        assert not _violations(row["payload"]), (row["signal_type"], row["payload"])
    stored = [r for r in rows if r["signal_type"] == "approval_pending"]
    assert stored and stored[0]["payload"].get("text_chars") == len(TYPED)

    # Backfill: what a reconnecting client is handed.
    monkeypatch.setattr(tracker, "get_telemetry_handle", lambda: live.tel)
    client = TestClient(create_app({}))
    for path in ("/api/events/recent?limit=500",
                 f"/api/events/recent?types=computer_step,approval_pending"
                 f"&session_id={SESSION}"):
        body = client.get(path).json()
        assert body
        for row in body:
            assert not _violations(row), (path, row)


async def test_a_reconnecting_client_backfills_its_session(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.telemetry import tracker
    from prometheus.web.server import create_app

    rig = _rig(tmp_path, ["click-0"])
    live = _live(rig, tmp_path)
    await _bind(rig)
    task = await _start(rig)
    await rig.runner.wait(task.task_id, timeout=10)
    # Another session's noise must not come back.
    from prometheus.sentinel.signals import ActivitySignal

    await live.bus.emit(ActivitySignal(kind="computer_step", source="computer",
                                       payload={"session_id": "web:other", "seq": 1}))
    monkeypatch.setattr(tracker, "get_telemetry_handle", lambda: live.tel)
    client = TestClient(create_app({}))
    rows = client.get(
        "/api/events/recent", params={
            "types": "computer_task_started,computer_step,computer_task_ended",
            "session_id": SESSION, "since": "2000-01-01T00:00:00+00:00"}).json()
    kinds = {r["signal_type"] for r in rows}
    assert kinds == {"computer_task_started", "computer_step", "computer_task_ended"}
    assert all(r["payload"]["session_id"] == SESSION for r in rows)
    future = client.get("/api/events/recent", params={
        "types": "computer_step", "since": "2999-01-01T00:00:00+00:00"}).json()
    assert future == []


async def test_the_action_log_is_pruned_per_session(tmp_path):
    rig = _rig(tmp_path, ["click-0"] * 6)
    live = _live(rig, tmp_path, keep_per_session=4)
    await _bind(rig)
    task = await _start(rig)
    await rig.runner.wait(task.task_id, timeout=10)
    rows = live.tel.signal_events_since(signal_types=[
        "computer_binding", "computer_task_started", "computer_step",
        "computer_task_ended"], limit=500)
    mine = [r for r in rows if r["payload"].get("session_id") == SESSION]
    assert len(mine) == 4, [r["signal_type"] for r in mine]
    assert mine[0]["signal_type"] in ("computer_task_ended", "computer_binding"), (
        "the newest are kept")


# ── LAYER 1: THE TASK IN TODAY'S CHAT TIMELINE ──────────────────────────────

async def test_layer_one_renders_the_task_in_the_chat_timeline(tmp_path):
    rig = _rig(tmp_path, ["click-0"])
    live = _live(rig, tmp_path)
    await _bind(rig)
    task = await _start(rig)
    await rig.runner.wait(task.task_id, timeout=10)
    starts = [f["payload"] for f in live.rec.frames if f["type"] == "tool_call_start"]
    ends = [f["payload"] for f in live.rec.frames if f["type"] == "tool_call_end"]
    assert starts[0]["call_id"] == task.task_id
    assert starts[0]["tool_name"] == "computer_task"
    assert starts[0]["inputs"]["origin"] == "user_task"
    assert any(s["call_id"] == f"{task.task_id}:1" and s["tool_name"] == "computer_click"
               for s in starts)
    assert {e["call_id"] for e in ends} >= {task.task_id, f"{task.task_id}:1"}
    for e in ends:
        assert {"call_id", "tool_name", "success"} <= set(e), "iOS requires these"
    assert "chat_done" in live.rec.types(), "no chat turn was live"
    done = next(f["payload"] for f in live.rec.frames if f["type"] == "chat_done")
    assert done["message_id"] == f"computer:{task.task_id}", (
        "iOS's ChatDonePayload requires message_id; without it the frame is "
        "undecodable and dropped")
    assert isinstance(done["session_id"], str) and done["interrupted"] is False
    assert "turn_completed" not in live.rec.types(), "never a turn summary push"


async def test_layer_one_does_not_end_a_live_chat_turn(tmp_path):
    rig = _rig(tmp_path, ["click-0"])
    live = _live(rig, tmp_path)
    turn = asyncio.create_task(asyncio.sleep(30))
    live.bridge._turn_tasks[SESSION] = turn
    try:
        await _bind(rig)
        task = await _start(rig)
        await rig.runner.wait(task.task_id, timeout=10)
        assert "chat_done" not in live.rec.types(), (
            "chat_done would clear iOS's turnInFlight mid-turn")
    finally:
        turn.cancel()


# ── STOP FROM EVERY BEACON AND TELEGRAM, MID-TASK ───────────────────────────

async def _mid_task(tmp_path):
    """A task that has acted once and is waiting on its second step's prompt."""
    rig = _rig(tmp_path, ["click-0", "set-2"])
    live = _live(rig, tmp_path)
    await rig.runner.bind(SESSION, APP, scope="session", by=PERSON, surface="rest")
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())
    assert len(rig.driver.dispatched) == 1, "it acted once before the stop"
    return rig, live, task


async def _assert_stopped(rig, task):
    done = await rig.runner.wait(task.task_id, timeout=10)
    assert done.outcome == "stopped"
    assert len(rig.driver.dispatched) == 1, "nothing dispatched after the stop"
    assert not rig.channel.pending


async def test_stop_from_beacon_desktop_the_ws_interrupt(tmp_path):
    rig, live, task = await _mid_task(tmp_path)
    ws = _Recorder()
    await live.bridge._handle_client_message(ws, json.dumps(
        {"type": "interrupt", "payload": {"session_id": SESSION}}))
    assert ws.frames[0]["type"] == "interrupt_ack"
    assert ws.frames[0]["payload"]["stopped"] is True
    await _assert_stopped(rig, task)


async def test_stop_from_beacon_ios_the_http_interrupt(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    rig, live, task = await _mid_task(tmp_path)
    app = create_app({})
    app.state.ws_bridge = live.bridge
    res = await asyncio.to_thread(
        TestClient(app).post, "/api/chat/interrupt", json={"session_id": SESSION})
    assert res.json() == {"session_id": SESSION, "stopped": True}
    await _assert_stopped(rig, task)


async def test_stop_from_telegram(tmp_path):
    from prometheus.gateway.commands import cmd_computer

    rig, live, task = await _mid_task(tmp_path)
    reply = await cmd_computer(rig.runner, "stop", by=PERSON, surface="telegram",
                               session_id=SESSION, chat_id=456, notify=None)
    assert "Stopped" in reply
    await _assert_stopped(rig, task)


async def test_the_ws_stop_frame_acks_and_stops_one_task(tmp_path):
    rig, live, task = await _mid_task(tmp_path)
    ws = _Recorder()
    await live.bridge._handle_client_message(ws, json.dumps(
        {"type": "computer_task_stop", "payload": {"task_id": task.task_id}}))
    assert ws.frames[0]["type"] == "computer_task_stop_ack"
    assert ws.frames[0]["payload"] == {"task_id": task.task_id, "stopped": True}
    await _assert_stopped(rig, task)
    ended = [p for k, p in live.seen if k == "computer_task_ended"][0]
    assert ended["outcome"] == "stopped"


# ── PUSH CARRIES NO CONTENT ─────────────────────────────────────────────────

async def test_a_desktop_prompt_push_carries_no_content(tmp_path):
    from prometheus.push.dispatcher import PushDispatcher

    rig = _rig(tmp_path, ["set-2"])
    live = _live(rig, tmp_path)
    await _bind(rig)
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())
    payload = [p for k, p in live.seen if k == "approval_pending"][0]

    delivered: list[dict] = []

    class _Store:
        def push_targets(self):
            return ["d1"]

    d = PushDispatcher(_Store(), object(), bridge=None)

    async def _deliver(target, body):
        delivered.append(body)

    d._deliver = _deliver
    await d.on_signal(SimpleNamespace(kind="approval_pending", payload=payload))
    rig.runner.stop(task.task_id)
    await rig.runner.wait(task.task_id, timeout=10)
    assert delivered
    body = delivered[0]
    assert set(body) == {"aps", "request_id", "expires_at"}, body
    assert body["aps"]["alert"]["body"] == "Prometheus needs a decision"
    assert "category" not in body["aps"], (
        "no lock-screen Approve for a prompt the screen cannot show")
    blob = json.dumps(body)
    for content in (TYPED, "Name", "computer_", APP):
        assert content not in blob, content


def test_the_action_log_retention_key_ships():
    import yaml

    template = (Path(__file__).resolve().parents[1] / "config"
                / "prometheus.yaml.default")
    block = yaml.safe_load(template.read_text(encoding="utf-8"))["computer_use"]
    assert block["action_log"]["keep_per_session"] == 200
