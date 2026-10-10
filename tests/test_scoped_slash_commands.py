"""A scoped device's chat is a conversation, not a console: slash commands are default-deny too.

THE HOLE
--------
``docs/contracts/device-scoping.md`` listed it as a known gap: a command typed into a device's own session runs
with the OPERATOR's authority. Two of them mutate daemon-wide state:

* ``/gate off`` puts the permission gate into autonomous mode for the whole process: no approval prompts for
  anyone, until the daemon restarts. That makes every restriction on what a scoped device may approve moot;
* ``/revoke <grant>`` removes a remembered grant.

and a dozen read daemon-wide state (``/events``, ``/memory``, ``/wiki``, ``/doctor``, ``/grants``...) or write it
(``/note``, ``/workspace``).

THE RULE
--------
A message typed by a scoped device may be a command only if the command is in
``web/route_access.SCOPED_SLASH_COMMANDS``: ``/help`` and the commands that act on the device's own session
(``/reset``, ``/clear``, ``/steer``, ``/queue``, ``/unqueue``, ``/clearsteers``, ``/ephemeral``). Any other known
command is answered with a refusal IN THE CHAT (it is a chat message, so a refusal text and not an HTTP status)
and does nothing; it does not fall through to the agent either. An unknown ``/word`` is still just text for the
agent, as it always was. The operator (the master token or an owner device) and every internal caller keep every
command.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity  # noqa: E402
from prometheus.gateway import commands  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from tests.support.device_world import World  # noqa: E402

ALLOWED = {"help", "reset", "clear", "steer", "queue", "unqueue", "clearsteers", "ephemeral"}


class Sock:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def replies(self) -> list[str]:
        return [f["payload"]["content"] for f in self.frames
                if f["type"] == "chat_message" and f["payload"].get("role") == "assistant"]


@pytest.fixture
def w(tmp_path) -> World:
    world = World(tmp_path)
    world.gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
    world.queue = ApprovalQueue(security_gate=world.gate)
    world.bridge.approval_queue = world.queue
    world.app.state.approval_queue = world.queue
    world.sid = world.device_session("a")
    world.socket = Sock()                                      # the device's own socket: it owns the session
    world.bridge._clients.add(world.socket)
    world.bridge._ws_identity[world.socket] = DeviceIdentity(id=world.a["id"], name="device-a", platform="ios")
    return world


def say(w: World, who: str, text: str):
    return w.call(who, "POST", "/api/chat/send", json={"session_id": w.sid, "message": text})


def refused(reply: str) -> bool:
    return "owner" in reply.lower()


# ── what a scoped device cannot do from its chat ─────────────────────────────

def test_gate_off_from_a_scoped_devices_chat_does_nothing(w):
    assert say(w, "a", "/gate off").status_code == 200
    assert w.gate.current_mode() == PermissionMode.DEFAULT, "the daemon went autonomous on a phone's say-so"
    assert refused(w.socket.replies()[-1])


def test_the_operator_can_still_turn_the_gate_off(w):
    w.sid = w.device_session("a")
    assert w.call("op", "POST", "/api/chat/send",
                  json={"session_id": w.sid, "message": "/gate off"}).status_code == 200
    assert w.gate.current_mode() == PermissionMode.AUTONOMOUS


def _grant(w: World) -> str:
    """A real remembered grant, made the way an operator makes one."""
    from prometheus.permissions.approval_queue import PendingAction

    w.queue._register(PendingAction(request_id="aaaa0001", tool_name="bash", description="run echo hi",
                                    grant_command="echo hi"))
    made = w.call("op", "POST", "/api/approvals/aaaa0001/approve", json={"scope": "always"})
    assert made.status_code == 200 and made.json()["grant_id"], made.text
    return made.json()["grant_id"]


def test_revoke_from_a_scoped_devices_chat_removes_nothing(w):
    grant_id = _grant(w)
    say(w, "a", f"/revoke {grant_id}")
    assert [g.grant_id for g in w.gate.list_grants()] == [grant_id], "a phone revoked the owner's grant"
    assert refused(w.socket.replies()[-1])


def test_the_operator_can_still_revoke(w):
    grant_id = _grant(w)
    w.call("op", "POST", "/api/chat/send", json={"session_id": w.sid, "message": f"/revoke {grant_id}"})
    assert w.gate.list_grants() == []


@pytest.mark.parametrize("name", sorted((commands.formatter_command_names() | commands.session_command_names())
                                        - ALLOWED))
def test_every_other_command_is_refused_to_a_scoped_device_and_never_runs(w, monkeypatch, name):
    ran = []
    monkeypatch.setattr(commands, "run_formatter_command", _spy(ran))
    monkeypatch.setattr(commands, "run_session_command", _spy(ran))
    monkeypatch.setattr("prometheus.web.slash_router.run_formatter_command", _spy(ran))
    monkeypatch.setattr("prometheus.web.slash_router.run_session_command", _spy(ran))
    say(w, "a", f"/{name} x")
    assert ran == [], f"/{name} ran for a scoped device"
    assert refused(w.socket.replies()[-1]), (name, w.socket.replies()[-1:])


def _spy(ran):
    async def run(name, args, ctx):
        ran.append(name)
        return "ran"

    return run


# ── what it can still do ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_a_scoped_device_keeps_the_commands_that_act_on_its_own_session(w, name):
    say(w, "a", f"/{name}")
    replies = w.socket.replies()
    assert replies and not refused(replies[-1]), (name, replies[-1:])


def test_an_unknown_command_is_still_text_for_the_agent(w):
    say(w, "a", "/frobnicate the widgets")
    assert not any(refused(r) for r in w.socket.replies())


def test_plain_chat_is_untouched(w):
    assert say(w, "a", "hello there").status_code == 200
    assert not any(refused(r) for r in w.socket.replies())


# ── the same rule over the WebSocket ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_over_the_websocket_a_scoped_device_cannot_gate_off_either(w):
    ws = w.socket
    await w.bridge._handle_client_message(
        ws, json.dumps({"type": "send_message", "payload": {"session_id": w.sid, "content": "/gate off"}}))
    assert w.gate.current_mode() == PermissionMode.DEFAULT
    assert refused(ws.replies()[-1])


@pytest.mark.asyncio
async def test_over_the_websocket_the_operator_keeps_it(w):
    op = Sock()
    w.bridge._clients.add(op)
    w.bridge._ws_identity[op] = GLOBAL_IDENTITY
    await w.bridge._handle_client_message(
        op, json.dumps({"type": "send_message", "payload": {"session_id": w.sid, "content": "/gate off"}}))
    assert w.gate.current_mode() == PermissionMode.AUTONOMOUS


@pytest.mark.asyncio
async def test_an_internal_caller_that_names_no_one_keeps_every_command(w):
    """Paperclip, notes and the rest call the handler directly: they are the daemon, not a scoped device."""
    await w.bridge._handle_send_message(w.sid, "/gate off")
    assert w.gate.current_mode() == PermissionMode.AUTONOMOUS


# ── the router, on its own ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_router_refuses_without_running_and_is_the_operators_by_default():
    from types import SimpleNamespace

    from prometheus.web.slash_router import route_slash

    ctx = SimpleNamespace()
    scoped = await route_slash("/gate off", ctx, operator=False)
    assert scoped.handled is True and refused(scoped.reply or "")
    unknown = await route_slash("/frobnicate", ctx, operator=False)
    assert unknown.handled is False


def test_the_scoped_commands_are_exactly_these():
    from prometheus.web import route_access

    assert route_access.SCOPED_SLASH_COMMANDS == frozenset(ALLOWED)
    known = commands.formatter_command_names() | commands.session_command_names()
    assert ALLOWED <= known, "an allowed command that does not exist is a typo, not a policy"

