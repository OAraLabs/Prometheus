"""A device the operator holds to a profile runs that profile's tools, whatever the model asks for.

One instance can serve its owner with every tool and, beside that, a server-to-server caller that must stay
read-only. The caller gets an ordinary (scoped) device token, and the operator holds that device to a profile::

    PUT /api/devices/<id>/profile  {"profile": "relay"}

Pinned here, through the real app, the real loop and a SecurityGate in AUTONOMOUS mode (so nothing but the
profile can be what stops a call):

* a held device's turn is offered only the profile's tools, and a tool outside it, called by name, is refused
  at the call and never runs: on ``POST /api/chat`` and on a run under one of the device's own sessions;
* the hold has no fallback that widens it: a held profile that matches no tool, or that no longer exists,
  runs nothing (an ordinary profile that matches nothing is ignored instead);
* the device cannot lift or change its hold, its sessions' profile or the instance's;
* nobody can rebind one of its sessions to another profile while the hold stands;
* an owner device cannot be held (it is the operator, and could lift the hold itself);
* a device that is not held keeps the instance default.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from prometheus.config.device_store import DeviceStore
from prometheus.config.profiles import ActiveProfileState, ProfileStore
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.session import SessionManager
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.modes import PermissionMode
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from prometheus.web.server import create_app

GLOBAL = "held-profile-global-token"
RAN: list[str] = []


class _Cmd(BaseModel):
    command: str


class _Write(BaseModel):
    path: str
    content: str


class _Path(BaseModel):
    path: str


def _tool(tool_name: str, model: type[BaseModel], *, read_only: bool) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool"
        input_model = model

        def is_read_only(self, arguments) -> bool:  # noqa: ANN001
            return read_only

        async def execute(self, arguments, context):  # noqa: ANN001
            RAN.append(tool_name)
            return ToolResult(output=f"{tool_name} ran", is_error=False)

    return _T()


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in (_tool("bash", _Cmd, read_only=False), _tool("write_file", _Write, read_only=False),
              _tool("read_file", _Path, read_only=True)):
        reg.register(t)
    return reg


class _AsksForEverything(ModelProvider):
    """Round 1 of every turn calls bash, write_file and read_file by name; round 2 answers."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.offered: list[list[str]] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.offered.append(sorted(t.get("name") for t in (request.tools or [])))
        answered = any(getattr(b, "type", "") == "tool_result"
                       for m in request.messages[-1:] for b in getattr(m, "content", []) or [])
        content = [TextBlock(text="done")] if answered else [
            ToolUseBlock(id="t-bash", name="bash", input={"command": "echo hi"}),
            ToolUseBlock(id="t-write", name="write_file",
                         input={"path": str(self.tmp_path / "x.txt"), "content": "hi"}),
            ToolUseBlock(id="t-read", name="read_file", input={"path": str(self.tmp_path / "a.txt")}),
        ]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _Rig:
    def __init__(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "relay.yaml").write_text("name: relay\ntools: [read_file]\n")
        (profiles_dir / "nothing.yaml").write_text("name: nothing\ntools: [no_such_tool]\n")
        self.profiles = ProfileStore(profiles_dir)
        self.state = ActiveProfileState(self.profiles, "full")
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.provider = _AsksForEverything(tmp_path)
        self.registry = _registry()
        self.gate = SecurityGate(mode=PermissionMode.AUTONOMOUS, workspace_root=tmp_path)
        self.app = create_app(
            {"web": {"api_token": GLOBAL}}, device_store=self.devices, session_mgr=SessionManager(),
            profile_store=self.profiles, profile_state=self.state,
            agent_loop=AgentLoop(provider=self.provider, model="stub", max_tokens=64,
                                 tool_registry=self.registry, permission_checker=self.gate,
                                 profile_resolver=self.state.get),
        )
        self.app.state.ws_bridge = SimpleNamespace(loop_context=LoopContext(
            provider=self.provider, model="stub", system_prompt="sys", max_tokens=64,
            tool_registry=self.registry, permission_checker=self.gate, profile_resolver=self.state.get,
        ))
        self.relay = self.devices.mint("upstream relay", "other")
        self.alice = self.devices.mint("alice's phone", "ios")
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.tmp_path = tmp_path

    def client(self, token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {token}"})

    async def hold(self, device: dict, profile: str | None) -> httpx.Response:
        async with self.client(GLOBAL) as c:
            return await c.put(f"/api/devices/{device['id']}/profile", json={"profile": profile})

    async def chat(self, device: dict, session: str = "s1") -> httpx.Response:
        RAN.clear()
        self.provider.offered.clear()
        async with self.client(device["token"]) as c:
            return await c.post("/api/chat", json={"session_id": session, "content": "go"})


@pytest.fixture
def rig(tmp_path) -> _Rig:
    return _Rig(tmp_path)


# ── the rule ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_held_device_runs_only_its_profiles_tools(rig):
    r = await rig.hold(rig.relay, "relay")
    assert r.status_code == 200, r.text
    r = await rig.chat(rig.relay)
    assert r.status_code == 200, r.text
    assert rig.provider.offered[0] == ["read_file"], "the held device was offered tools outside its profile"
    assert RAN == ["read_file"], "a tool outside the held profile ran"


@pytest.mark.asyncio
async def test_a_device_that_is_not_held_keeps_the_instance_default(rig):
    await rig.hold(rig.relay, "relay")
    r = await rig.chat(rig.alice)
    assert r.status_code == 200, r.text
    assert sorted(RAN) == ["bash", "read_file", "write_file"]


def test_a_run_under_a_held_devices_own_session_is_held_too(rig):
    # The WebSocket and /api/chat/send paths run the loop under the session's own id: the hold follows the
    # session's owner, not only the route that admitted it.
    assert asyncio.run(rig.hold(rig.relay, "relay")).status_code == 200
    assert rig.devices.claim_session("beacon:relay-1", rig.relay["id"])
    RAN.clear()
    ctx = LoopContext(provider=rig.provider, model="stub", system_prompt="", max_tokens=64,
                      tool_registry=rig.registry, permission_checker=rig.gate,
                      profile_resolver=rig.state.get, cwd=rig.tmp_path)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="beacon:relay-1"):
            pass

    asyncio.run(drain())
    assert RAN == ["read_file"]


# ── no fallback widens a hold ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_held_profile_that_matches_no_tool_runs_nothing(rig):
    # An ORDINARY profile that filters the catalog to nothing is ignored (the loop advertises everything and
    # logs an error). For a hold that would hand the device every tool.
    assert (await rig.hold(rig.relay, "nothing")).status_code == 200
    r = await rig.chat(rig.relay)
    assert r.status_code == 200, r.text
    assert RAN == [], "a hold that matched no tool fell back to every tool"
    assert rig.provider.offered[0] == []


@pytest.mark.asyncio
async def test_a_held_profile_that_no_longer_exists_runs_nothing(rig):
    # Deleted or renamed after the hold was set: an ordinary binding falls back to the global profile.
    assert rig.devices.set_profile(rig.relay["id"], "deleted-since", by="test")
    r = await rig.chat(rig.relay)
    assert r.status_code == 200, r.text
    assert RAN == [], "a hold on a missing profile fell back to the instance default"


# ── the device cannot get out of it ───────────────────────────────────────


@pytest.mark.asyncio
async def test_the_held_device_cannot_lift_or_change_its_profile(rig):
    await rig.hold(rig.relay, "relay")
    await rig.chat(rig.relay)  # brings web:s1 into existence as the relay's
    async with rig.client(rig.relay["token"]) as c:
        assert (await c.put(f"/api/devices/{rig.relay['id']}/profile", json={"profile": None})).status_code == 403
        assert (await c.put(f"/api/devices/{rig.relay['id']}/profile",
                            json={"profile": "full"})).status_code == 403
        assert (await c.put("/api/sessions/web:s1/profile", json={"profile": "full"})).status_code == 403
        assert (await c.put("/api/profiles/active", json={"name": "full"})).status_code == 403
    await rig.chat(rig.relay)
    assert RAN == ["read_file"]


@pytest.mark.asyncio
async def test_a_held_devices_session_cannot_be_rebound_to_another_profile(rig):
    await rig.hold(rig.relay, "relay")
    await rig.chat(rig.relay)
    async with rig.client(GLOBAL) as c:
        r = await c.put("/api/sessions/web:s1/profile", json={"profile": "full"})
        assert r.status_code == 409, r.text
        assert "relay" in r.text
        shown = (await c.get("/api/sessions/web:s1/profile")).json()
    assert shown["effective"] == "relay" and shown["source"] == "device"


# ── who can be held, and by whom ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_owner_device_cannot_be_held(rig):
    r = await rig.hold(rig.owner, "relay")
    assert r.status_code == 400, r.text
    assert "owner" in r.text


@pytest.mark.asyncio
async def test_the_hold_is_validated_and_listed(rig):
    r = await rig.hold(rig.relay, "no-such-profile")
    assert r.status_code == 400 and "no-such-profile" in r.text
    async with rig.client(GLOBAL) as c:
        assert (await c.put("/api/devices/not-a-device/profile", json={"profile": "relay"})).status_code == 404
        assert (await rig.hold(rig.relay, "relay")).status_code == 200
        listed = {d["id"]: d for d in (await c.get("/api/devices")).json()}
    assert listed[rig.relay["id"]]["profile"] == "relay"
    assert listed[rig.alice["id"]]["profile"] is None


@pytest.mark.asyncio
async def test_the_operator_can_lift_a_hold(rig):
    await rig.hold(rig.relay, "relay")
    assert (await rig.hold(rig.relay, None)).status_code == 200
    await rig.chat(rig.relay)
    assert sorted(RAN) == ["bash", "read_file", "write_file"]
