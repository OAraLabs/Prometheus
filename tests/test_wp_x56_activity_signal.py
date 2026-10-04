"""WP-X.56: a person's message ends the heartbeat's idle, so AutoDream pauses.

Before this, nothing in the daemon moved the heartbeat's activity clock: its
only input was a ``message_received`` signal that nothing emitted, and
``record_activity()`` had no callers. Idle started once per boot, 15 minutes
in, and never ended (live: 293 ``idle_start`` rows, 0 ``idle_end``), so
AutoDream dreamed every 30 minutes around the clock, including its model
calls to the main backend.

The five ingress points that call T-4's ``outcomes.note_user_message`` now
also call ``activity.note_user_activity()``; the heartbeat reads that clock on
its next tick.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.gateway import heartbeat as heartbeat_mod
from prometheus.gateway.heartbeat import DEFAULT_IDLE_THRESHOLD, Heartbeat
from prometheus.sentinel import activity
from prometheus.sentinel.signals import ActivitySignal, SignalBus

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"
T0 = 1_800_000_000.0


@pytest.fixture(autouse=True)
def _fresh_activity():
    activity.reset()
    yield
    activity.reset()


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock(T0)
    monkeypatch.setattr(heartbeat_mod, "time", SimpleNamespace(time=c.time, monotonic=c.monotonic))
    return c


def _heartbeat_on(bus: SignalBus) -> Heartbeat:
    hb = Heartbeat()
    hb._last_activity = T0  # "booted" at T0
    hb.signal_bus = bus
    return hb


def _record(bus: SignalBus) -> list[str]:
    seen: list[str] = []

    async def _on(signal: ActivitySignal) -> None:
        seen.append(signal.kind)

    bus.subscribe("idle_start", _on)
    bus.subscribe("idle_end", _on)
    return seen


# --------------------------------------------------------------------------- #
# The heartbeat: a message ends idle; idle starts again after the threshold
# --------------------------------------------------------------------------- #


class TestHeartbeatIdle:
    @pytest.mark.asyncio
    async def test_a_message_ends_idle_and_idle_starts_again_after_the_threshold(self, clock):
        bus = SignalBus()
        seen = _record(bus)
        hb = _heartbeat_on(bus)

        clock.now = T0 + DEFAULT_IDLE_THRESHOLD + 1
        await hb._check_idle()
        assert seen == ["idle_start"], "nobody has written for 15 minutes"

        message_at = clock.now + 60
        activity.note_user_activity(at=message_at)
        clock.now = message_at + 30  # the next heartbeat tick
        await hb._check_idle()
        assert seen == ["idle_start", "idle_end"], "a message must end idle"

        clock.now = message_at + DEFAULT_IDLE_THRESHOLD - 1
        await hb._check_idle()
        assert seen == ["idle_start", "idle_end"], "still inside the threshold"

        clock.now = message_at + DEFAULT_IDLE_THRESHOLD + 1
        await hb._check_idle()
        assert seen == ["idle_start", "idle_end", "idle_start"], \
            "idle starts again once the threshold passes with no message"

    @pytest.mark.asyncio
    async def test_a_message_older_than_the_last_activity_changes_nothing(self, clock):
        bus = SignalBus()
        seen = _record(bus)
        hb = _heartbeat_on(bus)
        activity.note_user_activity(at=T0 - 5_000)  # before boot

        clock.now = T0 + DEFAULT_IDLE_THRESHOLD + 1
        await hb._check_idle()
        assert seen == ["idle_start"]

    @pytest.mark.asyncio
    async def test_no_message_keeps_the_old_behaviour(self, clock):
        bus = SignalBus()
        seen = _record(bus)
        hb = _heartbeat_on(bus)
        for step in range(1, 5):
            clock.now = T0 + DEFAULT_IDLE_THRESHOLD * step + 1
            await hb._check_idle()
        assert seen == ["idle_start"], "one idle_start, and nothing ends it"

    def test_note_user_activity_stamps_the_wall_clock(self, monkeypatch):
        monkeypatch.setattr(activity.time, "time", lambda: T0 + 7)
        assert activity.last_user_activity() is None
        activity.note_user_activity()
        assert activity.last_user_activity() == T0 + 7


# --------------------------------------------------------------------------- #
# AutoDream follows: it pauses while someone is using Prometheus
# --------------------------------------------------------------------------- #


class TestAutoDreamPauses:
    @pytest.mark.asyncio
    async def test_autodream_stops_on_a_message_and_resumes_after_the_threshold(self, clock):
        from prometheus.sentinel.autodream import AutoDreamEngine

        bus = SignalBus()
        hb = _heartbeat_on(bus)
        engine = AutoDreamEngine(bus, config={"synthesis_enabled": False})
        await engine.start()
        try:
            clock.now = T0 + DEFAULT_IDLE_THRESHOLD + 1
            await hb._check_idle()
            await asyncio.sleep(0)
            assert engine.dreaming

            activity.note_user_activity(at=clock.now + 10)
            clock.now += 30
            await hb._check_idle()
            assert not engine.dreaming, "a person is using Prometheus: no dreaming"

            clock.now += DEFAULT_IDLE_THRESHOLD
            await hb._check_idle()
            assert engine.dreaming, "quiet again past the threshold: dreaming resumes"
        finally:
            engine._dreaming = False
            for task in asyncio.all_tasks():
                if task is not asyncio.current_task():
                    task.cancel()


# --------------------------------------------------------------------------- #
# Every ingress point registers the activity
# --------------------------------------------------------------------------- #


class _Session:
    def __init__(self) -> None:
        self.messages: list = []

    def add_user_message(self, content, **kw):
        self.messages.append(content)
        return len(self.messages)

    def last_persisted_row_id(self):
        return 1


class _Mgr:
    def get(self, sid):
        return None

    def get_or_create(self, sid):
        return _Session()


class TestIngress:
    def test_beacon_ws_message_registers_activity(self, monkeypatch):
        from prometheus.web.ws_server import WebSocketBridge

        bridge = WebSocketBridge(loop_context=None, session_mgr=_Mgr())

        async def _noop(*a, **k):
            return None

        monkeypatch.setattr(bridge, "broadcast", _noop)
        asyncio.run(bridge._handle_send_message("beacon:wp-x56", "hello there"))
        assert activity.last_user_activity() is not None

    def test_rest_chat_registers_activity(self, tmp_path, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.engine.session import SessionManager
        from prometheus.skills.registry import SkillRegistry
        from prometheus.web.server import create_app
        from tests.test_api_chat_reaches_the_model import _RecordingLoop

        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        app = create_app({"gateway": {"system_prompt": "sys"}}, session_mgr=SessionManager(),
                         skill_registry=SkillRegistry(), agent_loop=_RecordingLoop())
        r = TestClient(app).post("/api/chat", json={"session_id": "abc", "content": "hello there"})
        assert r.status_code == 200, r.text
        assert activity.last_user_activity() is not None

    @staticmethod
    def _functions_calling(name: str) -> list[tuple[str, ast.AST]]:
        """The INNERMOST function around each ``<x>.<name>(...)`` call."""
        found: dict[str, ast.AST] = {}

        class _Visitor(ast.NodeVisitor):
            def __init__(self, rel: str) -> None:
                self.rel = rel
                self.stack: list[ast.AST] = []

            def _fn(self, node):
                self.stack.append(node)
                self.generic_visit(node)
                self.stack.pop()

            visit_FunctionDef = visit_AsyncFunctionDef = _fn

            def visit_Call(self, node):
                if (isinstance(node.func, ast.Attribute) and node.func.attr == name
                        and self.stack):
                    fn = self.stack[-1]
                    found[f"{self.rel}:{fn.name}"] = fn
                self.generic_visit(node)

        for path in sorted(SRC.rglob("*.py")):
            if path.name == "outcomes.py":
                continue
            _Visitor(str(path.relative_to(SRC))).visit(ast.parse(path.read_text()))
        return list(found.items())

    def test_every_ingress_that_calls_the_t4_hook_also_registers_activity(self):
        """The five ingress points of WP-X.54 T-4 are the activity sources too.

        Structural on purpose: a sixth surface that adds the T-4 hook and
        forgets this call would leave the heartbeat idle while it is in use.
        """
        sites = self._functions_calling("note_user_message")
        names = sorted(s for s, _ in sites)
        assert names == [
            "gateway/discord.py:_dispatch_to_agent",
            "gateway/slack.py:_dispatch_to_agent",
            "gateway/telegram.py:_dispatch_to_agent",
            "web/server.py:post_chat",
            "web/ws_server.py:_handle_send_message",
        ], names
        for site, fn in sites:
            body = ast.unparse(fn)
            assert "activity.note_user_activity()" in body, f"{site} does not register activity"
            assert body.index("activity.note_user_activity()") < body.index("note_user_message("), \
                f"{site}: register activity on arrival, beside the T-4 hook"
