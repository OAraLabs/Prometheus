"""A scoped device cannot approve its own gated calls: only the owner approves.

A scoped device token (an approved phone, a server-to-server caller) is a chat client for its own
conversations. It used to be allowed to approve the gated tool calls of its own sessions, once or until the
daemon restarts. That let whatever sits behind the token (an upstream app, a relay, a prompt-injected client)
sanction a call the gate stopped for the owner to see, in the same breath as it asked for it. Now:

* a scoped device's approve is refused (403), whatever the scope, for a tool call and for a worker's request;
  the request stays pending for the owner;
* it may still DENY its own sessions' requests: a denial can only stop something;
* the owner (the global token or an owner device, as before) approves;
* a desktop prompt is unchanged: a device a person marked for computer use answers it (the door's rule).

Pinned through a real turn as well: a scoped token's ``POST /api/chat`` hits a gated tool, its own approve is
refused and the tool has not run; the owner's approve runs it.
"""

from __future__ import annotations

import asyncio
import secrets
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

import httpx  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.engine.agent_loop import AgentLoop, LoopContext  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue, ApprovalResult, PendingAction  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider  # noqa: E402
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "scoped-approve-global-" + secrets.token_hex(8)
SESSION = "web:relay-1"
RAN: list[str] = []


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))


class _Any(BaseModel):
    model_config = ConfigDict(extra="allow")


class _GatedTool(BaseTool):
    """An MCP-shaped tool: the gate asks for every ``mcp__*`` call (tests/test_mcp_gating.py)."""

    name = "mcp__upstream__write_record"
    description = "writes a record upstream"
    input_model = _Any

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return False

    async def execute(self, arguments, context):  # noqa: ANN001
        RAN.append(self.name)
        return ToolResult(output="written")


class _CallsTheTool(ModelProvider):
    def __init__(self) -> None:
        self.rounds = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.rounds += 1
        content = ([ToolUseBlock(id="t1", name=_GatedTool.name, input={"record": "x"})]
                   if self.rounds == 1 else [TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")


class _Daemon:
    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
        self.queue = ApprovalQueue(security_gate=self.gate)
        self.gate._approval_queue = self.queue
        registry = ToolRegistry()
        registry.register(_GatedTool())
        provider = _CallsTheTool()
        self.app = create_app(
            {"web": {"api_token": GLOBAL}}, device_store=self.devices, session_mgr=SessionManager(),
            approval_queue=self.queue,
            agent_loop=AgentLoop(provider=provider, model="stub", max_tokens=64, tool_registry=registry,
                                 permission_checker=self.gate),
        )
        self.app.state.ws_bridge = SimpleNamespace(loop_context=LoopContext(
            provider=provider, model="stub", system_prompt="sys", max_tokens=64,
            tool_registry=registry, permission_checker=self.gate))
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.relay = self.devices.mint("upstream relay", "other")
        self.devices.claim_session(SESSION, self.relay["id"])

    def client(self, token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {token}"}, timeout=30)

    def pend(self, request_id: str, **extra) -> PendingAction:
        action = PendingAction(request_id=request_id, tool_name="bash", description="run echo hi",
                               grant_command="echo hi", session_id=SESSION, **extra)
        self.queue.pending[request_id] = action
        return action


@pytest.fixture
def d(tmp_path) -> _Daemon:
    return _Daemon(tmp_path)


# ── the rule ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"scope": "once"}, {"scope": "until-restart"}, None])
async def test_a_scoped_device_cannot_approve_its_own_sessions_tool_call(d, body):
    action = d.pend("aaaa0001")
    async with d.client(d.relay["token"]) as c:
        r = await (c.post("/api/approvals/aaaa0001/approve", json=body) if body is not None
                   else c.post("/api/approvals/aaaa0001/approve"))
    assert r.status_code == 403, r.text
    assert "owner" in r.text
    assert "aaaa0001" in d.queue.pending and not action._event.is_set(), "the request must wait for the owner"
    assert d.gate.list_grants() == []


@pytest.mark.asyncio
async def test_nor_its_own_workers_request(d):
    d.pend("bbbb0001", worker_task_id="w1", once_only=True)
    async with d.client(d.relay["token"]) as c:
        r = await c.post("/api/approvals/bbbb0001/approve", json={"scope": "once"})
    assert r.status_code in (401, 403), r.text
    assert "bbbb0001" in d.queue.pending


@pytest.mark.asyncio
async def test_it_may_still_deny_its_own_sessions_tool_call(d):
    action = d.pend("cccc0001")
    async with d.client(d.relay["token"]) as c:
        r = await c.post("/api/approvals/cccc0001/deny")
    assert r.status_code == 200, r.text
    assert action._result == ApprovalResult.DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("who", ["global", "owner"])
async def test_the_owner_approves(d, who):
    action = d.pend("dddd0001")
    token = GLOBAL if who == "global" else d.owner["token"]
    async with d.client(token) as c:
        r = await c.post("/api/approvals/dddd0001/approve", json={"scope": "once"})
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert action._result == ApprovalResult.APPROVED


# ── through a real turn ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_scoped_devices_turn_cannot_approve_the_call_it_raised(d):
    RAN.clear()
    async with d.client(d.relay["token"]) as relay, d.client(d.owner["token"]) as owner:
        turn = asyncio.create_task(relay.post("/api/chat", json={"session_id": "relay-1", "content": "go"}))
        for _ in range(500):
            if d.queue.pending:
                break
            await asyncio.sleep(0.01)
        assert d.queue.pending, "the gated call never asked"
        rid = next(iter(d.queue.pending))

        mine = await relay.get("/api/approvals")
        assert rid in {a["request_id"] for a in mine.json()}, "the device still sees its own pending call"
        refused = await relay.post(f"/api/approvals/{rid}/approve", json={"scope": "once"})
        assert refused.status_code == 403, refused.text
        await asyncio.sleep(0.05)
        assert RAN == [], "the gated tool ran on the scoped device's own approval"
        assert not turn.done()

        approved = await owner.post(f"/api/approvals/{rid}/approve", json={"scope": "once"})
        assert approved.status_code == 200 and approved.json()["ok"] is True, approved.text
        reply = await asyncio.wait_for(turn, 15)
    assert reply.status_code == 200, reply.text
    assert RAN == [_GatedTool.name]
