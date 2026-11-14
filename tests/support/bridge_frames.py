"""Frames from a real ``WebSocketBridge`` turn, for tests of what listens to them (the speech route's reply holder).

The bridge is real; only the model is scripted: ``run_loop`` is replaced by one that yields real stream events
(``engine/stream_events``) in the order given, so every ``agent_state``, ``chat_delta``, ``tool_call_start``,
``chat_discard``, ``chat_done`` and ``error`` frame is the bridge's own, in the bridge's own order.

A script item is a stream event, :data:`HOLD` (park until :meth:`Turn.release`), or an exception instance (raised,
so the bridge takes its failure path).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from prometheus.engine.stream_events import AssistantTextDelta, ToolExecutionCompleted, ToolExecutionStarted
from prometheus.web.ws_server import WebSocketBridge

HOLD = object()


def text(t: str) -> AssistantTextDelta:
    return AssistantTextDelta(text=t)


def tool(name: str = "lookup", call_id: str = "t1") -> list[Any]:
    return [ToolExecutionStarted(tool_name=name, tool_input={}, tool_use_id=call_id),
            ToolExecutionCompleted(tool_name=name, output="ok", tool_use_id=call_id)]


class _Session:
    """Just enough ChatSession for ``_run_agent``: an in-place messages list."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    def get_messages(self) -> list[Any]:
        return self.messages

    def persist_loop_result(self, original_len: int) -> None:
        return None


class _Client:
    """A WebSocket client of the bridge: every frame is recorded and handed to *on_frame*, as it is broadcast."""

    def __init__(self, on_frame: Callable[[dict[str, Any]], None] | None) -> None:
        self.frames: list[dict[str, Any]] = []
        self._on_frame = on_frame

    async def send(self, raw: str) -> None:
        frame = json.loads(raw)
        self.frames.append(frame)
        if self._on_frame is not None:
            self._on_frame(frame)


class Turn:
    def __init__(self, task: asyncio.Future[Any], gate: asyncio.Event) -> None:
        self.task = task
        self._gate = gate

    def release(self) -> None:
        self._gate.set()


class Bridge:
    """A real bridge whose frames go to *on_frame*. ``start`` runs one scripted turn in a session."""

    def __init__(self, monkeypatch: Any, on_frame: Callable[[dict[str, Any]], None] | None = None) -> None:
        import prometheus.engine.agent_loop as agent_loop_mod

        self.bridge = WebSocketBridge(loop_context=object())
        self.client = _Client(on_frame)
        self.bridge._clients.add(self.client)
        self._sessions: dict[str, _Session] = {}
        self._scripts: dict[int, tuple[list[Any], asyncio.Event]] = {}
        self._next = 0
        bridge = self

        async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None, surface=None):
            script, gate = bridge._scripts.pop(min(bridge._scripts))
            for item in script:
                if item is HOLD:
                    await gate.wait()
                elif isinstance(item, BaseException):
                    raise item
                else:
                    yield item, None

        monkeypatch.setattr(agent_loop_mod, "run_loop", run_loop)

    @property
    def frames(self) -> list[dict[str, Any]]:
        return self.client.frames

    def start(self, session_id: str, script: list[Any]) -> Turn:
        """Start a turn (it queues behind a running one in the same session, as in the daemon)."""
        flat: list[Any] = []
        for item in script:
            flat.extend(item if isinstance(item, list) else [item])
        gate = asyncio.Event()
        # Turns take their scripts in the order they reach run_loop, which is the order they hold the session lock.
        self._scripts[self._next] = (flat, gate)
        self._next += 1
        session = self._sessions.setdefault(session_id, _Session())
        return Turn(asyncio.ensure_future(self.bridge._run_agent(session_id, session)), gate)

    def interrupt(self, session_id: str) -> bool:
        return self.bridge.interrupt_turn(session_id)


async def settle(cond: Callable[[], bool], timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not cond():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)
