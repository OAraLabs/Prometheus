"""A device's hold is lifted only by a person, and a held device never reaches the desktop.

A hold (``PUT /api/devices/<id>/profile``) keeps a server-to-server caller to a profile beside the owner's
own assistant. Two ways round it are closed here:

* **Lifting it.** Setting a hold may use the API token: it only takes tools away. Lifting or changing one
  widens what a device may run, and the token is no proof of a person (a model with a shell can read it).
  So a lift with any credential only ASKS: it becomes a consent prompt on the owner's approval surfaces
  (Telegram ``/approve``, Beacon), and the hold stands until a person approves it. Never the API token,
  never the held device itself, never "approve all".
* **The desktop.** Computer use is its own runner, not the agent loop, so a profile does not reach it. A held
  device is refused there in code: a run under one of its sessions cannot start or bind a desktop task
  (whoever asks), the device is never a person for the door, it cannot be marked for computer use, and a
  hold that cannot be read refuses.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

import httpx  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.config.profiles import ActiveProfileState, ProfileStore  # noqa: E402
from prometheus.engine.agent_loop import AgentLoop, LoopContext  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.permissions.approver import Approver  # noqa: E402
from prometheus.permissions.checker import SecurityGate  # noqa: E402
from prometheus.permissions.modes import PermissionMode  # noqa: E402
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider  # noqa: E402
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "hold-person-global-token"
OWNER_ID = 456
#: The person: the Telegram owner (gateway.owner_user_ids), answering in their own chat.
PERSON = Approver("telegram", str(OWNER_ID), "owner", owner=True)


class _Rig:
    def __init__(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "relay.yaml").write_text("name: relay\ntools: [read_file]\n")
        self.profiles = ProfileStore(profiles_dir)
        self.state = ActiveProfileState(self.profiles, "full")
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.app = create_app(
            {"web": {"api_token": GLOBAL}, "gateway": {"owner_user_ids": [OWNER_ID]}},
            device_store=self.devices, session_mgr=SessionManager(),
            profile_store=self.profiles, profile_state=self.state,
        )
        self.relay = self.devices.mint("upstream relay", "other")
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.tmp_path = tmp_path

    def client(self, token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                 headers={"Authorization": f"Bearer {token}"})

    async def put_hold(self, profile: str | None, token: str = GLOBAL) -> httpx.Response:
        async with self.client(token) as c:
            return await c.put(f"/api/devices/{self.relay['id']}/profile", json={"profile": profile})

    @property
    def queue(self):
        return self.app.state.approval_queue

    def held(self) -> str | None:
        return self.devices.device_profile(self.relay["id"])


@pytest.fixture
def rig(tmp_path) -> _Rig:
    return _Rig(tmp_path)


async def _held_by_token(rig: _Rig) -> None:
    r = await rig.put_hold("relay")
    assert r.status_code == 200, r.text
    assert rig.held() == "relay"


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


# ── setting a hold still takes the token ──────────────────────────────────


@pytest.mark.asyncio
async def test_setting_a_hold_on_a_device_with_none_takes_effect_at_once(rig):
    await _held_by_token(rig)
    assert rig.queue is None or not rig.queue.pending


# ── lifting or changing one asks a person ─────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("to", [None, "full"])
async def test_a_lift_or_change_with_the_token_only_asks(rig, to):
    await _held_by_token(rig)
    r = await rig.put_hold(to)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["profile"] == "relay", "the reply must say the hold still stands"
    rid = body["pending"]["request_id"]
    assert rid in rig.queue.pending
    assert rig.held() == "relay", "the hold moved without a person"


@pytest.mark.asyncio
async def test_the_owner_approves_and_only_then_the_hold_lifts(rig):
    await _held_by_token(rig)
    rid = (await rig.put_hold(None)).json()["pending"]["request_id"]
    assert await rig.queue.approve(rid, by=PERSON)
    await _settle()
    assert rig.held() is None


@pytest.mark.asyncio
async def test_an_approved_change_moves_the_hold_to_the_new_profile(rig):
    await _held_by_token(rig)
    rid = (await rig.put_hold("full")).json()["pending"]["request_id"]
    assert await rig.queue.approve(rid, by=PERSON)
    await _settle()
    assert rig.held() == "full"


@pytest.mark.asyncio
async def test_a_denied_lift_leaves_the_hold(rig):
    await _held_by_token(rig)
    rid = (await rig.put_hold(None)).json()["pending"]["request_id"]
    assert await rig.queue.deny(rid, by=Approver("global-token", "global", "API token"))
    await _settle()
    assert rig.held() == "relay"
    assert rid not in rig.queue.pending


@pytest.mark.asyncio
@pytest.mark.parametrize("who", ["global-token", "held-device", "stranger", "owner-device"])
async def test_only_a_person_can_approve_a_lift(rig, who):
    await _held_by_token(rig)
    rid = (await rig.put_hold(None)).json()["pending"]["request_id"]
    # A hold lift is critical: each credential sends the digest it was served, so what refuses is who it is.
    digest = rig.queue.digest(rig.queue.pending[rid])
    if who == "global-token":
        async with rig.client(GLOBAL) as c:
            r = await c.post(f"/api/approvals/{rid}/approve", json={"scope": "once", "digest": digest})
        assert r.status_code in (401, 403) or r.json().get("ok") is False, r.text
    elif who == "held-device":
        async with rig.client(rig.relay["token"]) as c:
            r = await c.post(f"/api/approvals/{rid}/approve", json={"scope": "once"})
        assert r.status_code in (401, 403, 404) or r.json().get("ok") is False, r.text
    elif who == "stranger":
        assert not await rig.queue.approve(rid, by=Approver("telegram", "999", "stranger", owner=False))
    else:
        # An owner device is the operator's credential, and its token is minted by a local route a model can
        # reach: it asks like the token does, and is not the approval.
        async with rig.client(rig.owner["token"]) as c:
            r = await c.post(f"/api/approvals/{rid}/approve", json={"scope": "once", "digest": digest})
        assert r.status_code in (401, 403) or r.json().get("ok") is False, r.text
    await _settle()
    assert rig.held() == "relay", f"{who} lifted the hold"


@pytest.mark.asyncio
async def test_approve_all_never_lifts_a_hold(rig):
    from prometheus.gateway.commands import approve_detail

    await _held_by_token(rig)
    rid = (await rig.put_hold(None)).json()["pending"]["request_id"]
    await approve_detail(rig.queue, "all", by=PERSON)
    await _settle()
    assert rig.held() == "relay" and rid in rig.queue.pending


@pytest.mark.asyncio
async def test_the_pending_lift_is_on_the_approval_surface(rig):
    await _held_by_token(rig)
    rid = (await rig.put_hold(None)).json()["pending"]["request_id"]
    async with rig.client(GLOBAL) as c:
        listed = (await c.get("/api/approvals")).json()
    rows = listed if isinstance(listed, list) else listed.get("pending", listed.get("requests", []))
    row = next(r for r in rows if r.get("request_id") == rid)
    assert rig.relay["id"] in row["description"] and "relay" in row["description"]


# ── the model cannot do it on its own ─────────────────────────────────────


class _Http(BaseModel):
    method: str
    path: str
    body: dict | None = None


def test_a_full_tools_turn_cannot_lift_a_hold_through_the_api(rig):
    """The owner's assistant has every tool and can read the API token (here: a tool that makes any request
    with it, as `curl` from bash would). It asks to lift the hold and then approves its own request. The hold
    stands, and the request waits for a person."""
    replies: list[httpx.Response] = []

    class _Curl(BaseTool):
        name = "curl"
        description = "HTTP request to the local API with the API token"
        input_model = _Http

        def is_read_only(self, arguments) -> bool:  # noqa: ANN001
            return False

        async def execute(self, arguments, context):  # noqa: ANN001
            async with rig.client(GLOBAL) as c:
                r = await c.request(arguments.method, arguments.path, json=arguments.body)
            replies.append(r)
            return ToolResult(output=r.text)

    class _Model(ModelProvider):
        def __init__(self) -> None:
            self.round = 0

        async def stream_message(self, request):  # noqa: ANN001
            self.round += 1
            if self.round == 1:
                content = [ToolUseBlock(id="t1", name="curl", input={
                    "method": "PUT", "path": f"/api/devices/{rig.relay['id']}/profile", "body": {"profile": None}})]
            elif self.round == 2:
                rid = replies[-1].json().get("pending", {}).get("request_id", "none")
                # It fetched the request first (GET /api/approvals), so it sends the digest it was served: what
                # refuses is the person check, not a missing digest (#925 review).
                digest = rig.queue.digest(rig.queue.pending[rid]) if rid in rig.queue.pending else ""
                content = [ToolUseBlock(id="t2", name="curl", input={
                    "method": "POST", "path": f"/api/approvals/{rid}/approve",
                    "body": {"scope": "once", "digest": digest}})]
            else:
                content = [TextBlock(text="done")]
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=content),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")

    asyncio.run(_held_by_token(rig))
    registry = ToolRegistry()
    registry.register(_Curl())
    loop = AgentLoop(provider=_Model(), model="stub", max_tokens=64, tool_registry=registry,
                     permission_checker=SecurityGate(mode=PermissionMode.AUTONOMOUS, workspace_root=rig.tmp_path))

    async def turn() -> None:
        await loop.run_async("sys", "lift the relay's hold", session_id="telegram:456")
        await _settle()

    asyncio.run(turn())
    assert len(replies) == 2, "the model's two requests did not both run"
    assert replies[0].status_code == 202
    assert rig.held() == "relay", "a turn lifted a hold with nothing but the API token"


# ── the desktop ───────────────────────────────────────────────────────────

SESSION = "beacon:relay-desktop"


class _Integration:
    enabled = True

    def __init__(self) -> None:
        from prometheus.computer.discovery import AppRecord, WindowRecord
        from prometheus.computer.driver import FixtureDriver
        from prometheus.computer.types import Observation

        seen = Observation(target="box", app="gedit", pid=1, window_id=1, snapshot_id="s0", elements=(),
                           degraded=False, truncated=False, elements_complete=False, total_element_count=0,
                           returned_element_count=0, web_content_seen=False)
        self._driver = FixtureDriver([seen], apps=[AppRecord(pid=1, name="gedit")],
                                     windows=[WindowRecord(window_id=1, pid=1, app_name="gedit", z_index=1)])

    async def probe(self, *, force: bool = False):
        return {"state": "ready"}

    def driver(self):
        return self._driver

    def local_target(self):
        return "box"

    @property
    def state(self):
        return "ready"


def _door(devices) -> SimpleNamespace:
    from prometheus.computer.approvals import ComputerApprovalChannel
    from prometheus.computer.door import PersonCheck
    from prometheus.computer.task import ComputerTaskRunner
    from prometheus.permissions.audit import AuditLogger

    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=AuditLogger(devices._db_path.parent / "audit"))
    people = PersonCheck(device_store=devices, telegram_user_ids={OWNER_ID})
    channel = ComputerApprovalChannel(security_gate=gate, people=people)
    runner = ComputerTaskRunner(integration=_Integration(), gate=gate, channel=channel, people=people,
                                skip_preconditions=True, heartbeat_s=3600)
    return SimpleNamespace(runner=runner, people=people)


def _held_device_with_a_session(tmp_path) -> tuple[DeviceStore, dict]:
    devices = DeviceStore(tmp_path / "devices.db")
    relay = devices.mint("upstream relay", "other")
    assert devices.claim_session(SESSION, relay["id"])
    assert devices.set_profile(relay["id"], "relay", by="test")
    return devices, relay


@pytest.mark.asyncio
async def test_a_held_devices_session_cannot_start_a_desktop_task_even_for_a_person(tmp_path):
    from prometheus.computer.door import DoorRefused
    from prometheus.computer.task import ComputerTaskInput

    devices, _ = _held_device_with_a_session(tmp_path)
    door = _door(devices)
    with pytest.raises(DoorRefused) as refused:
        await door.runner.start(ComputerTaskInput(goal="save it", app="gedit"),
                                session_id=SESSION, surface="telegram", by=PERSON)
    assert refused.value.code == "held_device"


@pytest.mark.asyncio
async def test_a_held_devices_session_cannot_bind_an_app(tmp_path):
    from prometheus.computer.door import DoorRefused

    devices, _ = _held_device_with_a_session(tmp_path)
    door = _door(devices)
    with pytest.raises(DoorRefused) as refused:
        await door.runner.bind(SESSION, "gedit", scope="task", by=PERSON, surface="telegram")
    assert refused.value.code == "held_device"
    assert door.runner.binding_for(SESSION) is None


def test_a_held_device_is_never_a_person_for_the_door(tmp_path):
    devices = DeviceStore(tmp_path / "devices.db")
    relay = devices.mint("upstream relay", "other")
    assert devices.set_computer(relay["id"], True, by="telegram:456")  # marked BEFORE it was held
    assert devices.set_profile(relay["id"], "relay", by="test")
    ok, why = _door(devices).people.check(Approver("device-token", relay["id"], "upstream relay"))
    assert not ok and "held" in why


def test_a_held_device_cannot_be_marked_for_computer_use(tmp_path):
    devices, relay = _held_device_with_a_session(tmp_path)
    with pytest.raises(ValueError, match="held"):
        devices.set_computer(relay["id"], True, by="telegram:456")
    assert not devices.computer_allowed(relay["id"])


@pytest.mark.asyncio
async def test_a_hold_that_cannot_be_read_refuses_the_desktop(tmp_path, monkeypatch):
    from prometheus.computer.door import DoorRefused
    from prometheus.computer.task import ComputerTaskInput

    devices = DeviceStore(tmp_path / "devices.db")
    door = _door(devices)

    def unreadable(session_id):  # noqa: ANN001
        raise OSError("database is locked")

    monkeypatch.setattr(devices, "session_hold", unreadable)
    with pytest.raises(DoorRefused) as refused:
        await door.runner.start(ComputerTaskInput(goal="save it", app="gedit"),
                                session_id="telegram:456", surface="telegram", by=PERSON)
    assert refused.value.code == "held_device"


@pytest.mark.asyncio
async def test_an_unheld_session_still_reaches_the_door(tmp_path):
    # The control: the refusal is about the hold, not the door being shut. With no binding the door asks which
    # app (NeedsConsent), the ordinary first step.
    from prometheus.computer.door import NeedsConsent
    from prometheus.computer.task import ComputerTaskInput

    devices = DeviceStore(tmp_path / "devices.db")
    door = _door(devices)
    with pytest.raises(NeedsConsent):
        await door.runner.start(ComputerTaskInput(goal="save it", app="gedit"),
                                session_id="telegram:456", surface="telegram", by=PERSON)
