"""The owner pauses and resumes from Telegram (Beacon iOS "Piece 17"; required before 0.9.21, Will 2026-11-13).

Until this, the only way to lift a pause was a device carrying the person mark, and a daemon that restarts comes
back paused, so an owner without a marked device was stuck. ``/pause [reason]`` and ``/resume`` are the OWNER's,
in their own Telegram: not another chat member, not the API token, not a device. ``/pause`` runs the same pause as
``PUT /api/pause`` (the state first, then the sweep), through the web app the daemon registered; ``/resume`` clears
it, tells every client, and starts nothing.
"""

from __future__ import annotations

import asyncio
import types

import pytest

pytest.importorskip("fastapi")

from prometheus.engine import pause  # noqa: E402
from prometheus.permissions import approver as _approver  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction  # noqa: E402


@pytest.fixture
def state(tmp_path):
    s = pause.PauseState(tmp_path / "pause.json")
    pause.install(s)
    yield s
    pause.install(None)


def _tg(user_id: str, owners=("42",)):
    return _approver.telegram(types.SimpleNamespace(id=int(user_id), username="u"), owners=list(owners))


class _Gate:
    pass


@pytest.fixture
def app(monkeypatch):
    from prometheus.web import pause_routes

    queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    queue._register(PendingAction(request_id="t1", tool_name="bash", description="run", session_id="web:x"))
    frames: list[dict] = []

    class _Bridge:
        _turn_tasks: dict = {}

        async def broadcast(self, frame):
            frames.append(frame)

    fake = types.SimpleNamespace(state=types.SimpleNamespace(approval_queue=queue, ws_bridge=_Bridge(),
                                                             task_manager=None))
    fake.frames = frames
    fake.queue = queue
    monkeypatch.setattr(pause_routes, "_app", fake)
    return fake


def test_the_owner_pauses_and_the_sweep_runs(state, app):
    from prometheus.gateway.commands import cmd_pause

    reply = asyncio.run(cmd_pause("going out", by=_tg("42")))
    info = pause.current()
    assert info is not None and info.reason == "going out" and "owner" in info.by.lower()
    assert app.queue.finished("t1")[0] == "withdrawn"
    assert any(f.get("type") == "paused_changed" and f["payload"]["paused"] for f in app.frames)
    assert "paused" in reply.lower()


def test_the_owner_resumes(state, app):
    from prometheus.gateway.commands import cmd_resume

    state.pause(by="API token")
    reply = asyncio.run(cmd_resume(by=_tg("42")))
    assert pause.current() is None and "resumed" in reply.lower()
    assert any(f.get("type") == "paused_changed" and f["payload"]["paused"] is False for f in app.frames)


def test_resume_when_nothing_is_paused_says_so(state, app):
    from prometheus.gateway.commands import cmd_resume

    reply = asyncio.run(cmd_resume(by=_tg("42")))
    assert "not paused" in reply.lower()


@pytest.mark.parametrize("who", ["not-the-owner", "api-token", "a-device", "open"])
def test_nobody_else_pauses_or_resumes(state, app, who):
    from prometheus.gateway.commands import cmd_pause, cmd_resume

    by = {"not-the-owner": _tg("7"),
          "api-token": _approver.Approver(_approver.GLOBAL_TOKEN, "global", "API token"),
          "a-device": _approver.Approver(_approver.DEVICE, "d1", "phone"),
          "open": _approver.Approver(_approver.OPEN, "none", "")}[who]
    asyncio.run(cmd_pause("x", by=by))
    assert pause.current() is None
    state.pause(by="API token")
    asyncio.run(cmd_resume(by=by))
    assert pause.current() is not None


def test_a_pause_without_the_web_app_still_holds(state, monkeypatch):
    """A daemon whose web app is not up (or a test) still pauses: the state is what every gate reads."""
    from prometheus.gateway.commands import cmd_pause
    from prometheus.web import pause_routes

    monkeypatch.setattr(pause_routes, "_app", None)
    asyncio.run(cmd_pause("", by=_tg("42")))
    assert pause.current() is not None


def test_telegram_registers_both_and_the_handlers_check_the_sender(state, app):
    from pathlib import Path

    import prometheus.gateway.telegram as tg

    src = Path(tg.__file__).read_text()
    assert 'CommandHandler("pause", self._cmd_pause)' in src
    assert 'CommandHandler("resume", self._cmd_resume)' in src

    sent: list[str] = []
    adapter = tg.TelegramAdapter.__new__(tg.TelegramAdapter)
    adapter.config = types.SimpleNamespace(owner_ids=lambda: frozenset({"42"}))

    async def _send(chat_id, text, **kw):
        sent.append(text)

    adapter.send = _send

    def update(user_id: int, text: str):
        return types.SimpleNamespace(message=types.SimpleNamespace(text=text),
                                     effective_chat=types.SimpleNamespace(id=1),
                                     effective_user=types.SimpleNamespace(id=user_id, username="u"))

    asyncio.run(adapter._cmd_pause(update(7, "/pause"), None))
    assert pause.current() is None
    asyncio.run(adapter._cmd_pause(update(42, "/pause stop now"), None))
    assert pause.current() is not None and pause.current().reason == "stop now"
    asyncio.run(adapter._cmd_resume(update(7, "/resume"), None))
    assert pause.current() is not None
    asyncio.run(adapter._cmd_resume(update(42, "/resume"), None))
    assert pause.current() is None
