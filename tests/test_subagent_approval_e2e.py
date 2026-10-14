"""#791 — an approval raised inside a subagent, end to end.

Since #762 a subagent runs under the turn's own gate. A subagent has no session of its own, so the loop falls back to
the gate's queue for a call the gate holds. Nobody had run that: these tests do, on the pieces production uses.

  * the real ``SecurityGate`` with a real ``AuditLogger`` and the real ``ApprovalQueue`` (wired both ways, as the daemon
    does), so a held call genuinely waits and every resolution writes its row;
  * the real web app with a real ``DeviceStore``: an OWNER device (the Mac, made an owner by ``oara pair --owner``) and
    the shared token, answering over ``/api/approvals`` exactly as Beacon does;
  * a real subagent: the parent loop calls ``agent``, the spawner attached the daemon's way builds the subagent's loop,
    and the subagent calls ``bash`` with a command the gate's approve patterns hold (an ``echo`` whose text contains
    ``git push``, so it is safe even if it were to run).

The tool is a recording double: it never runs anything.

What they pin: the request reaches an owner device and the subagent blocks until it is answered; a denial refuses the
call; an unanswered request is denied at expiry; ending the parent turn withdraws it (every client is told, and a late
answer is 410); and every answer is audited against whoever gave it. Two tests RECORD current policy rather than assert
a wish: the shared token can answer a subagent's request as it can any non-worker request, and the request is filed
under the subagent's own id, not the parent conversation.
"""

from __future__ import annotations

import asyncio
import secrets
import sqlite3

import pytest
from pydantic import BaseModel

pytest.importorskip("fastapi")

import prometheus.coordinator.agent_definitions as agent_definitions  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.coordinator.agent_definitions import AgentDefinition  # noqa: E402
from prometheus.coordinator.subagent import attach_subagent_spawner  # noqa: E402
from prometheus.engine.agent_loop import AgentLoop  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue  # noqa: E402
from prometheus.permissions.audit import AuditLogger  # noqa: E402
from prometheus.permissions.checker import SecurityGate  # noqa: E402
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider  # noqa: E402
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult  # noqa: E402
from prometheus.tools.builtin.agent import AgentTool  # noqa: E402
from tests.support.doubles import register_double  # noqa: E402
from tests.test_worker_approvals import _client, _until  # noqa: E402

PARENT_MARK = "PARENT-TURN"
SHELL_AGENT = "shell-agent"
PARENT_SESSION = "beacon:parent-conversation"
#: Held by the gate's ``git\s+push`` approve pattern, and harmless: it only echoes.
HELD_COMMAND = 'echo "subagent approval test: git push"'
GLOBAL = "subagent-global-" + secrets.token_hex(8)


class _BashInput(BaseModel):
    command: str


class _Bash(BaseTool):
    """Stands in for the real bash tool: records what it is asked to run, runs nothing."""

    name = "bash"
    description = "test double for bash"
    input_model = _BashInput

    def __init__(self) -> None:
        self.ran: list[str] = []

    async def execute(self, arguments, context):  # noqa: ANN001
        self.ran.append(arguments.command)
        return ToolResult(output="ran", is_error=False)


def _reply(blocks):  # noqa: ANN001
    return ApiMessageCompleteEvent(
        message=ConversationMessage(role="assistant", content=blocks),
        usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        stop_reason="stop",
    )


@register_double("subagent_approval.TwoLoopProvider", replaces="prometheus.providers.base.ModelProvider")
class _TwoLoop(ModelProvider):
    """The parent turn calls ``agent``; the subagent calls ``bash`` with the held command and reports what came back."""

    def __init__(self) -> None:
        self.parent_saw: list[str] = []
        self.sub_saw: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        results = [b for b in request.messages[-1].content if isinstance(b, ToolResultBlock)]
        if PARENT_MARK in (request.system_prompt or ""):
            if results:
                self.parent_saw.extend(b.content for b in results)
                yield _reply([TextBlock(text="parent done")])
            else:
                yield _reply([ToolUseBlock(id="p1", name="agent", input={
                    "description": "run it", "prompt": "run the command", "subagent_type": SHELL_AGENT})])
        elif results:
            self.sub_saw.extend(b.content for b in results)
            yield _reply([TextBlock(text="subagent done")])
        else:
            yield _reply([ToolUseBlock(id="s1", name="bash", input={"command": HELD_COMMAND})])


class _Bus:
    def __init__(self) -> None:
        self.signals: list[tuple[str, dict]] = []

    async def emit(self, signal) -> None:  # noqa: ANN001
        self.signals.append((signal.kind, dict(signal.payload)))

    def of(self, kind: str) -> list[dict]:
        return [p for k, p in self.signals if k == kind]


class _Telegram:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, chat_id, text, **_kw) -> None:  # noqa: ANN001
        self.sent.append(text)


class World:
    """One daemon's worth of the real pieces, and a parent turn that makes a subagent ask for approval."""

    def __init__(self, tmp_path, timeout_seconds: int = 300) -> None:
        from prometheus.web.server import create_app

        self.devices = DeviceStore(tmp_path / "devices.db")
        self.audit = AuditLogger(tmp_path / "audit")
        self.gate = SecurityGate(audit_logger=self.audit)
        self.telegram = _Telegram()
        self.queue = ApprovalQueue(security_gate=self.gate, telegram_adapter=self.telegram,
                                   timeout_seconds=timeout_seconds, default_chat_id=4242)
        self.gate._approval_queue = self.queue            # the late-bound hop, as the daemon wires it
        self.bus = _Bus()
        self.queue.signal_bus = self.bus
        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        self.app.state.approval_queue = self.queue
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="owner-code")

        self.bash = _Bash()
        registry = ToolRegistry()
        registry.register(AgentTool())
        registry.register(self.bash)
        self.provider = _TwoLoop()
        self.loop = AgentLoop(provider=self.provider, model="m", tool_registry=registry,
                              permission_checker=self.gate)
        attach_subagent_spawner(self.loop)

    def start_turn(self) -> asyncio.Task:
        """The parent turn, as a Beacon conversation, running in the background so the test can answer."""
        return asyncio.create_task(self.loop.run_async(
            system_prompt=PARENT_MARK, user_message="please run it", session_id=PARENT_SESSION))

    async def answer(self, who: str, verb: str, request_id: str):
        token = GLOBAL if who == "global" else self.owner["token"]
        async with _client(self.app) as c:
            return await c.post(f"/api/approvals/{request_id}/{verb}", headers={"Authorization": f"Bearer {token}"},
                                json={"scope": "once"} if verb == "approve" else None)

    async def listed_for_owner(self) -> list[dict]:
        async with _client(self.app) as c:
            r = await c.get("/api/approvals", headers={"Authorization": f"Bearer {self.owner['token']}"})
        body = r.json()
        return body.get("approvals", body.get("pending", body)) if isinstance(body, dict) else body

    def audit_rows(self) -> list[tuple[str, str, str | None]]:
        with sqlite3.connect(self.audit.db_path) as conn:
            return conn.execute(
                "select tool_name, decision, user_id from permission_audit where decision like 'confirm%' order by id"
            ).fetchall()


@pytest.fixture(autouse=True)
def _shell_agent_type(monkeypatch):
    monkeypatch.setattr(agent_definitions, "_BUILTIN_AGENTS", dict(agent_definitions._BUILTIN_AGENTS))
    agent_definitions.register_agent_definition(
        AgentDefinition(name=SHELL_AGENT, description="test agent", tools=["bash"]))


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


async def _finish(turn: asyncio.Task, seconds: float = 10):
    return await asyncio.wait_for(turn, seconds)


@pytest.mark.asyncio
async def test_the_request_reaches_an_owner_device_and_the_subagent_blocks_until_it_is_answered(world):
    turn = world.start_turn()
    pending = await _until(world.queue.list_pending)
    assert pending, "the subagent's held call never reached the approval queue"
    action = pending[0]
    assert action.tool_name == "bash" and "git push" in action.description

    # it reaches an owner device: pushed to the owner's sockets, listed over REST, and sent to the chat
    assert any(p.get("request_id") == action.request_id for p in world.bus.of("approval_pending"))
    assert any(a.get("request_id") == action.request_id for a in await world.listed_for_owner())
    assert any("Permission requested" in m and "git push" in m for m in world.telegram.sent)

    # and it blocks: nothing runs, and the parent turn that is waiting on the subagent waits too
    await asyncio.sleep(0.3)
    assert not turn.done(), "the turn finished before anyone answered"
    assert world.bash.ran == []

    response = await world.answer("owner", "approve", action.request_id)
    assert response.status_code == 200, response.text
    await _finish(turn)
    assert world.bash.ran == [HELD_COMMAND]
    assert world.provider.parent_saw == ["subagent done"]

    # audited against the device that approved it
    rows = world.audit_rows()
    assert ("bash", "confirm_pending", None) in [(t, d, None) for t, d, _ in rows]
    approved = [u for t, d, u in rows if t == "bash" and d == "confirm_approved"]
    assert approved == [f"device-token:{world.owner['id']}"], rows


@pytest.mark.asyncio
async def test_the_shared_token_answers_it_as_it_does_for_any_request_that_is_not_a_workers(world):
    """CURRENT POLICY, recorded so it is a decision and not an accident (#791).

    Only a WORKER's request needs a person (#744: a worker is a separate process the token used to reach). A
    subagent runs inside the daemon and its request is an ordinary one, so the shared API token can approve it, exactly
    as it can approve any other conversation's request today. If the person-only rule (the W4 proposal) is extended to
    every approval, flip this to expect 401, as ``test_worker_approvals`` does for a worker.
    """
    turn = world.start_turn()
    action = (await _until(world.queue.list_pending))[0]
    response = await world.answer("global", "approve", action.request_id)
    assert response.status_code == 200, response.text
    await _finish(turn)
    assert world.bash.ran == [HELD_COMMAND]
    approved = [u for t, d, u in world.audit_rows() if t == "bash" and d == "confirm_approved"]
    assert approved == ["global-token"], "the audit row must say it was the shared token, not a person"


@pytest.mark.asyncio
async def test_how_the_request_is_filed_today(world):
    """CURRENT SHAPE, recorded so a change to it is deliberate (#791).

    The request is filed under the subagent's OWN record id (``subagent:<parent conversation>:<spawn id>``), not the
    parent conversation, and it is not marked once-only, so it offers lasting grants (the gate is shared, so a grant
    would apply). ``session_id`` is not on the wire. So an owner device (not scoped to a conversation) sees and answers
    it, but a scoped device that owns the parent conversation would not. If requests are to be filed under the parent's
    conversation, change the first assertion.
    """
    turn = world.start_turn()
    action = (await _until(world.queue.list_pending))[0]
    assert action.session_id == f"subagent:{PARENT_SESSION}:{action.session_id.rsplit(':', 1)[-1]}", action.session_id
    assert action.worker_task_id is None and action.once_only is False
    wire = world.queue.serialize_pending(action)
    assert "session_id" not in wire and "always" in wire["extents"]
    await world.answer("owner", "deny", action.request_id)
    await _finish(turn)


@pytest.mark.asyncio
async def test_a_denial_refuses_the_call_and_is_audited_against_the_device(world):
    turn = world.start_turn()
    action = (await _until(world.queue.list_pending))[0]
    response = await world.answer("owner", "deny", action.request_id)
    assert response.status_code == 200, response.text
    await _finish(turn)
    assert world.bash.ran == []
    assert any("Permission denied" in s for s in world.provider.sub_saw), world.provider.sub_saw
    denied = [u for t, d, u in world.audit_rows() if t == "bash" and d == "confirm_rejected"]
    assert denied == [f"device-token:{world.owner['id']}"]


@pytest.mark.asyncio
async def test_an_unanswered_request_is_denied_at_expiry(tmp_path):
    world = World(tmp_path, timeout_seconds=1)
    turn = world.start_turn()
    assert await _until(world.queue.list_pending), "the request never arrived"
    await _finish(turn, seconds=10)                              # nobody answers; the window closes
    assert world.bash.ran == [], "an expired request must not run the command"
    assert any("Permission denied" in s for s in world.provider.sub_saw), world.provider.sub_saw
    assert world.queue.list_pending() == []
    assert [d for t, d, _ in world.audit_rows() if t == "bash"][-1] == "confirm_timeout"
    assert any(p.get("resolution") == "expired" for p in world.bus.of("approval_resolved"))
    assert any("EXPIRED" in m for m in world.telegram.sent)


@pytest.mark.asyncio
async def test_ending_the_parent_turn_withdraws_the_request(world):
    turn = world.start_turn()
    action = (await _until(world.queue.list_pending))[0]
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    assert world.queue.list_pending() == [], "the request outlived the turn that asked for it"
    assert world.bash.ran == []

    # a late answer acts on nothing and says so (410, as for a worker's request)
    late = await world.answer("owner", "approve", action.request_id)
    assert late.status_code == 410 and late.json().get("error") == "no_longer_active", late.text

    # and every client is told, so the card drops: approval_resolved "withdrawn", with a resolution row
    assert any(p.get("request_id") == action.request_id and p.get("resolution") == "withdrawn"
               for p in world.bus.of("approval_resolved")), world.bus.signals
    assert [d for t, d, _ in world.audit_rows() if t == "bash"][-1] == "confirm_rejected"
