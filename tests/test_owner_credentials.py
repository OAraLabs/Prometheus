"""Per-device OWNER credentials (P2) — the person's own device holds a token of its own, not the master key.

Before this, ``POST /api/pair/local`` (same-Mac pairing, #694) handed back the daemon's one API token: a
device paired that way held the master key, which on macOS sits in ``~/.config/prometheus/env`` where the
bash tool can read it. Now ``issue_owner_credential`` mints an OWNER device in the shared DeviceStore and
returns ITS token: revocable on its own, listed in ``GET /api/devices``, and operator-equivalent for what the
person needs their own cockpit to do.

What "owner" is, and is not (docs/contracts/device-scoping.md, section 6):

* operator-equivalent for SESSION scoping (#692): it lists, reads and drives every session, the operator's
  Telegram and CLI chats included, and receives every frame;
* operator-equivalent for DEVICE management: it lists and revokes any device;
* it is the "approver" tier: ``DeviceIdentity.is_operator`` is the one test pairing-approval consults;
* it is NOT root. Minting a device (``POST /api/devices``) and defining an MCP server (which spawns a
  process) stay global-token-only, so a stolen owner device can neither enrol an attacker device nor run code.

A device APPROVED from Telegram or by another device is minted by ``DeviceStore.mint`` and is an ordinary,
scoped device. Nothing here lets a request ask for the owner tier.

Real FastAPI app, real WebSocketBridge, real DeviceStore, real LCM store. Tokens are random per-test values.
"""

from __future__ import annotations

import inspect
import json
import logging
import sqlite3

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config import api_token as api_token_module  # noqa: E402
from prometheus.config import local_pairing as lp  # noqa: E402
from prometheus.config.api_token import verify_token  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from tests.support.device_world import A_SECRET, GLOBAL, TG_SECRET, World  # noqa: E402


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


@pytest.fixture
def pairing_dir(tmp_path, monkeypatch):
    directory = tmp_path / "pairing"
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(directory))
    return directory


@pytest.fixture
def secret(pairing_dir):
    return lp.mint_secret(pairing_dir)


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def call(world: World, token: str, method: str, url: str, **kw):
    return world.client.request(method, url, headers=bearer(token), **kw)


def loopback(world: World) -> TestClient:
    """The same app, reached the way the app on this Mac reaches it: loopback peer and Host."""
    return TestClient(world.app, base_url="http://127.0.0.1:8123", client=("127.0.0.1", 50123))


def pair_local(world: World, secret: str, **body) -> dict:
    resp = loopback(world).post("/api/pair/local", json={"code": secret, **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def mint_owner(world: World, name: str = "Beacon on this Mac") -> dict:
    return world.devices.mint_owner(name, "macos", by="test")


def _ids(rows) -> set[str]:
    return {r["session_id"] for r in rows}


# --------------------------------------------------------------------------- #
# The defect: a device paired on this Mac held the master key
# --------------------------------------------------------------------------- #


def test_pair_local_does_not_hand_back_the_master_key(world, secret):
    body = pair_local(world, secret)
    assert body["token"] != GLOBAL, "the device paired on this Mac was handed the daemon's global token"


def test_pair_local_response_keeps_the_three_keys_beacon_parses_and_adds_a_count(world, secret):
    body = pair_local(world, secret)
    assert set(body) == {"token", "api_base_port", "ws_port", "revoked_previous"}
    assert body["revoked_previous"] == 0 and isinstance(body["revoked_previous"], int)


def test_the_global_token_is_not_in_anything_the_pairing_logs(world, secret, caplog):
    caplog.set_level(logging.DEBUG)
    body = pair_local(world, secret)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert GLOBAL not in joined and body["token"] not in joined


# --------------------------------------------------------------------------- #
# What the issued credential is
# --------------------------------------------------------------------------- #


def test_the_issued_token_is_a_registered_owner_device_of_its_own(world, secret):
    token = pair_local(world, secret)["token"]

    identity = verify_token(token, GLOBAL, world.devices)
    assert identity is not None and not identity.is_global
    assert identity.owner is True and identity.is_operator is True

    listed = call(world, token, "GET", "/api/devices").json()
    mine = [d for d in listed if d["is_self"]]
    assert len(mine) == 1 and mine[0]["owner"] is True
    assert mine[0]["platform"] == "macos" and mine[0]["name"] == "Beacon on this Mac"


def test_the_device_name_may_be_chosen_by_the_client_and_is_bounded(world, secret, pairing_dir):
    token = pair_local(world, secret, name="Will's MacBook Pro")["token"]
    assert [d["name"] for d in call(world, token, "GET", "/api/devices").json() if d["is_self"]] == ["Will's MacBook Pro"]

    fresh = lp.replace_secret(pairing_dir)
    token2 = pair_local(world, fresh, name="x" * 500)["token"]
    name = next(d["name"] for d in call(world, token2, "GET", "/api/devices").json() if d["is_self"])
    assert 0 < len(name) <= 64


def test_a_request_cannot_ask_for_a_different_tier(world, secret):
    """The body carries a code and an optional name. Nothing else changes what is issued."""
    token = pair_local(world, secret, owner=False, tier="scoped", platform="ios", is_operator=False)["token"]
    me = next(d for d in call(world, token, "GET", "/api/devices").json() if d["is_self"])
    assert me["owner"] is True and me["platform"] == "macos"


def test_a_device_paired_here_can_be_revoked_alone(world, secret):
    token = pair_local(world, secret)["token"]
    other = mint_owner(world, "another owner")["token"]
    me = next(d for d in call(world, token, "GET", "/api/devices").json() if d["is_self"])
    assert call(world, token, "DELETE", f"/api/devices/{me['id']}").status_code == 200

    assert call(world, token, "GET", "/api/sessions").status_code == 401, "a revoked owner token must die"
    assert call(world, other, "GET", "/api/sessions").status_code == 200
    assert call(world, GLOBAL, "GET", "/api/sessions").status_code == 200


def test_pairing_again_revokes_the_owner_devices_the_same_route_issued_before(world, secret, pairing_dir):
    """A Beacon reinstall must not leave a standing operator credential that nobody holds."""
    first = pair_local(world, secret)["token"]
    assert call(world, first, "GET", "/api/sessions").status_code == 200

    second = pair_local(world, lp.replace_secret(pairing_dir))["token"]
    assert second != first
    assert call(world, first, "GET", "/api/sessions").status_code == 401, "the old install's credential outlived the re-pair"
    assert call(world, second, "GET", "/api/sessions").status_code == 200
    live_owners = [d for d in call(world, second, "GET", "/api/devices").json() if d["owner"] and not d["revoked_at"]]
    assert len(live_owners) == 1


def test_pairing_again_touches_only_owner_devices_of_the_sources_it_replaces(world, secret):
    """Another source's owner device (only the store API can make one now), an ordinary device and the
    global token are all untouched."""
    elsewhere = world.devices.mint_owner("some other source", "macos", by="some-other-source")
    token = pair_local(world, secret)["token"]

    assert sessions_status(world, elsewhere["token"]) == 200
    assert sessions_status(world, world.a["token"]) == 200
    assert sessions_status(world, GLOBAL) == 200
    assert sessions_status(world, token) == 200


def test_the_device_list_says_where_each_owner_device_came_from(world, secret):
    pair_local(world, secret)
    mint_owner_via_api(world, name="exchanged")
    call(world, GLOBAL, "POST", "/api/devices", json={"name": "phone", "platform": "ios"})
    by_name = {d["name"]: d for d in call(world, GLOBAL, "GET", "/api/devices").json()}
    assert by_name["Beacon on this Mac"]["owner_source"] == "same-mac-pairing"
    assert by_name["exchanged"]["owner_source"] == "same-mac-mint"
    assert by_name["phone"]["owner_source"] is None and by_name["phone"]["owner"] is False


def test_a_revoked_owner_device_is_no_longer_an_owner(world):
    owner = mint_owner(world)
    assert world.devices.is_owner(owner["id"]) is True
    world.devices.revoke(owner["id"])
    assert world.devices.is_owner(owner["id"]) is False
    assert owner["id"] not in world.devices.owner_device_ids()


# --------------------------------------------------------------------------- #
# Operator-equivalent for sessions: the person's own cockpit sees everything
# --------------------------------------------------------------------------- #


def test_an_owner_device_lists_and_reads_every_session(world):
    sid_a = world.device_session("a")
    tg = world.operator_session()
    owner = mint_owner(world)["token"]

    assert _ids(call(world, owner, "GET", "/api/sessions").json()) == {sid_a, tg}
    assert TG_SECRET in call(world, owner, "GET", f"/api/sessions/{tg}/messages").text
    assert A_SECRET in call(world, owner, "GET", f"/api/sessions/{sid_a}/messages").text


def test_an_owner_device_can_drive_and_manage_any_session(world):
    sid_a = world.device_session("a")
    tg = world.operator_session()
    owner = mint_owner(world)["token"]
    before = world.lcm.count_all(tg)

    assert call(world, owner, "POST", "/api/chat/send", json={"session_id": tg, "message": "from my Mac"}).status_code == 200
    assert world.lcm.count_all(tg) == before + 1
    assert call(world, owner, "PUT", f"/api/sessions/{sid_a}/title", json={"title": "Mine"}).status_code == 200
    world.bridge.interrupt_turn = lambda sid: True
    assert call(world, owner, "POST", "/api/chat/interrupt", json={"session_id": tg}).json()["stopped"] is True


def test_an_owner_device_searches_every_session(world):
    world.device_session("a", text="zebra from a")
    world.operator_session(text="zebra from telegram")
    owner = mint_owner(world)["token"]
    body = call(world, owner, "POST", "/api/search", json={"q": "zebra", "scope": "messages"}).json()
    assert {h["session_id"] for h in body["messages"]} >= {"telegram:123"}
    assert len(body["messages"]) == 2


def test_an_owner_socket_receives_every_frame_and_an_ordinary_one_does_not(world):
    sid_a = world.device_session("a")
    tg = world.operator_session()
    owner_identity = verify_token(mint_owner(world)["token"], GLOBAL, world.devices)
    scoped_identity = verify_token(world.b["token"], GLOBAL, world.devices)
    assert owner_identity is not None and scoped_identity is not None

    class Sock:
        def __init__(self):
            self.frames = []

        async def send(self, raw):
            self.frames.append(json.loads(raw))

    import asyncio

    async def run():
        owner_ws, scoped_ws = Sock(), Sock()
        for ws, ident in ((owner_ws, owner_identity), (scoped_ws, scoped_identity)):
            world.bridge._clients.add(ws)
            world.bridge._ws_identity[ws] = ident
        for sid in (sid_a, tg):
            await world.bridge.broadcast({"type": "chat_delta", "timestamp": 1.0, "payload": {"session_id": sid}})
        return owner_ws, scoped_ws

    owner_ws, scoped_ws = asyncio.run(run())
    assert len(owner_ws.frames) == 2, "the person's own Beacon must see every session's frames"
    assert scoped_ws.frames == []


def test_an_ordinary_device_is_still_scoped(world):
    """The control: minting through DeviceStore.mint (what an approved device gets) changes nothing."""
    sid_a = world.device_session("a")
    world.operator_session()
    assert _ids(call(world, world.b["token"], "GET", "/api/sessions").json()) == set()
    assert call(world, world.b["token"], "GET", f"/api/sessions/{sid_a}/messages").status_code == 404


# --------------------------------------------------------------------------- #
# Operator-equivalent for devices; and the limits of "owner"
# --------------------------------------------------------------------------- #


def test_an_owner_device_can_revoke_any_device_and_an_ordinary_one_cannot(world):
    owner = mint_owner(world)
    assert call(world, world.b["token"], "DELETE", f"/api/devices/{world.a['id']}").status_code == 403
    assert call(world, owner["token"], "DELETE", f"/api/devices/{world.a['id']}").status_code == 200
    assert call(world, world.a["token"], "GET", "/api/sessions").status_code == 401


def test_an_owner_device_cannot_enrol_another_device(world):
    """A stolen owner device must not be able to mint a persistent attacker device."""
    owner = mint_owner(world)["token"]
    r = call(world, owner, "POST", "/api/devices", json={"name": "attacker", "platform": "other"})
    assert r.status_code == 401
    r = call(world, owner, "POST", "/api/devices", json={"name": "attacker", "platform": "other", "owner": True})
    assert r.status_code == 401


def test_an_owner_device_cannot_define_an_mcp_server(world):
    """That spawns a process as the daemon user: the global token's alone."""
    owner = mint_owner(world)["token"]
    r = call(world, owner, "POST", "/api/mcp/servers", json={"name": "x", "command": "true"})
    assert r.status_code == 401


def test_the_global_token_can_mint_an_owner_device_from_this_mac_and_the_default_is_ordinary(world):
    plain = call(world, GLOBAL, "POST", "/api/devices", json={"name": "phone", "platform": "ios"})
    owner = mint_owner_via_api(world, name="Beacon on this Mac")
    assert plain.status_code == owner.status_code == 201
    rows = {d["id"]: d for d in call(world, GLOBAL, "GET", "/api/devices").json()}
    assert rows[plain.json()["id"]]["owner"] is False
    assert rows[owner.json()["id"]]["owner"] is True
    # and the minted owner token really is operator-equivalent
    assert call(world, owner.json()["token"], "GET", "/api/sessions").status_code == 200


def test_owner_devices_come_only_from_this_mac(world):
    """Will, 2026-10-08: owner tokens come only from same-Mac pairing. Another computer is enrolled as an
    ordinary, scoped device — even by the global token."""
    before = len(world.devices.list_devices())
    refused = mint_owner_via_api(world, peer="192.168.1.20", name="Jennifer's Mac")
    assert refused.status_code == 403 and "this Mac" in refused.json()["error"]
    assert len(world.devices.list_devices()) == before, "a refused owner mint created a device"
    # the ordinary route from the same address is fine, and is scoped
    plain = call(world, GLOBAL, "POST", "/api/devices", json={"name": "Jennifer's Mac", "platform": "macos"})
    assert plain.status_code == 201
    assert verify_token(plain.json()["token"], GLOBAL, world.devices).is_operator is False


def test_owner_must_be_a_real_boolean(world):
    r = call(world, GLOBAL, "POST", "/api/devices", json={"name": "x", "platform": "macos", "owner": "yes"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# The device model
# --------------------------------------------------------------------------- #


def test_mint_never_makes_an_owner_and_has_no_parameter_to_ask_for_one(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    plain = store.mint("phone", "ios")
    assert store.is_owner(plain["id"]) is False and store.owner_device_ids() == set()
    assert list(inspect.signature(DeviceStore.mint).parameters) == ["self", "name", "platform"]


def test_mint_owner_marks_the_device_and_says_who_marked_it(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    minted = store.mint_owner("Mac", "macos", by="same-mac-pairing")
    assert set(minted) >= {"id", "name", "platform", "token", "created_at"}
    assert store.is_owner(minted["id"]) is True
    assert store.owner_device_ids() == {minted["id"]}
    con = sqlite3.connect(str(tmp_path / "devices.db"))
    try:
        (by,) = con.execute("SELECT marked_by FROM owner_devices WHERE device_id = ?", (minted["id"],)).fetchone()
    finally:
        con.close()
    assert by == "same-mac-pairing"
    # one transaction: the token authenticates AND is an owner, never one without the other
    ident = verify_token(minted["token"], GLOBAL, store)
    assert ident is not None and ident.is_operator


def test_the_owner_table_is_lazy_and_api_devices_keeps_its_columns(tmp_path):
    """The parity goldens record every table in devices.db and every api_devices column."""
    path = tmp_path / "devices.db"
    store = DeviceStore(path)
    store.mint("phone", "ios")
    store.is_owner("x"), store.owner_device_ids()          # reading creates nothing

    def tables() -> set[str]:
        con = sqlite3.connect(str(path))
        try:
            return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()

    assert "owner_devices" not in tables()
    cols_before = _columns(path, "api_devices")
    store.mint_owner("Mac", "macos", by="test")
    assert "owner_devices" in tables()
    assert _columns(path, "api_devices") == cols_before


def _columns(path, table: str) -> list[str]:
    con = sqlite3.connect(str(path))
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    finally:
        con.close()


def test_a_device_identity_is_not_an_owner_unless_the_store_says_so(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    plain = store.mint("phone", "ios")
    ident = verify_token(plain["token"], "g", store)
    assert ident is not None and ident.owner is False and ident.is_operator is False
    assert api_token_module.GLOBAL_IDENTITY.is_operator is True


# --------------------------------------------------------------------------- #
# The seam
# --------------------------------------------------------------------------- #


def test_issue_owner_credential_needs_a_device_store_and_never_falls_back_to_the_master_key(tmp_path):
    with pytest.raises(TypeError):
        api_token_module.issue_owner_credential({})                    # type: ignore[call-arg]
    store = DeviceStore(tmp_path / "devices.db")
    issued = api_token_module.issue_owner_credential({"web": {"api_token": GLOBAL}}, devices=store)
    assert isinstance(issued.token, str) and issued.token != GLOBAL and issued.revoked_previous == 0
    ident = verify_token(issued.token, GLOBAL, store)
    assert ident is not None and ident.is_operator and not ident.is_global


def test_setup_mode_has_its_own_named_credential_function():
    """Setup mutations authenticate with the global token and setup mode creates no devices.db, so
    its credential is the global token — said out loud, in a function named for it."""
    assert callable(api_token_module.issue_setup_credential)


# --------------------------------------------------------------------------- #
# Re-pairing replaces the earlier owner credential of THIS Mac (Will's rule)
# --------------------------------------------------------------------------- #

import asyncio  # noqa: E402


def mint_owner_via_api(world: World, *, peer="127.0.0.1", name="Beacon, exchanged"):
    """POST /api/devices {"owner": true} with the global token, from the given peer address."""
    client = TestClient(world.app, base_url=f"http://{peer}:8123", client=(peer, 50123))
    return client.post("/api/devices", headers=bearer(GLOBAL),
                       json={"name": name, "platform": "macos", "owner": True})


def sessions_status(world: World, token: str) -> int:
    return call(world, token, "GET", "/api/sessions").status_code


def test_pairing_says_how_many_owner_credentials_it_revoked(world, secret, pairing_dir):
    assert pair_local(world, secret)["revoked_previous"] == 0
    assert pair_local(world, lp.replace_secret(pairing_dir))["revoked_previous"] == 1


def test_an_explicit_mint_from_this_mac_replaces_the_earlier_same_mac_credential(world, secret):
    paired = pair_local(world, secret)["token"]
    minted = mint_owner_via_api(world)
    assert minted.status_code == 201
    assert minted.json()["revoked_previous"] == 1
    assert sessions_status(world, paired) == 401, "the earlier same-Mac credential outlived the exchange"
    assert sessions_status(world, minted.json()["token"]) == 200


def test_pairing_replaces_what_an_explicit_mint_from_this_mac_issued(world, secret):
    """Fresh install: Beacon trades the setup token for an owner device. Later it reinstalls and
    re-pairs through /api/pair/local. The first credential must not survive either way."""
    exchanged = mint_owner_via_api(world).json()["token"]
    paired = pair_local(world, secret)
    assert paired["revoked_previous"] == 1
    assert sessions_status(world, exchanged) == 401
    assert sessions_status(world, paired["token"]) == 200


def test_a_replacement_only_touches_the_sources_it_names(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    other = store.mint_owner("other", "macos", by="some-other-source")
    first = store.mint_owner("first", "macos", by="same-mac-pairing")
    second = store.mint_owner("second", "macos", by="same-mac-pairing", replaces=("same-mac-pairing",))
    assert second["revoked_previous"] == [first["id"]]
    assert store.is_owner(other["id"]) and store.is_owner(second["id"]) and not store.is_owner(first["id"])


def test_the_new_credential_and_non_owner_devices_are_never_among_those_revoked(world, secret, pairing_dir):
    phone = call(world, GLOBAL, "POST", "/api/devices", json={"name": "phone", "platform": "ios"}).json()
    first = pair_local(world, secret)
    second = pair_local(world, lp.replace_secret(pairing_dir))
    assert sessions_status(world, second["token"]) == 200
    assert sessions_status(world, phone["token"]) == 200
    assert sessions_status(world, world.a["token"]) == 200 and sessions_status(world, first["token"]) == 401


def test_the_list_marks_the_replaced_credential_revoked_and_keeps_its_source(world, secret, pairing_dir):
    pair_local(world, secret)
    pair_local(world, lp.replace_secret(pairing_dir))
    rows = [d for d in call(world, GLOBAL, "GET", "/api/devices").json() if d["owner"] or d["owner_source"]]
    assert sorted(bool(r["revoked_at"]) for r in rows) == [False, True]


def test_a_failed_mint_leaves_the_earlier_credential_valid_and_creates_nothing(world, secret, monkeypatch):
    """One transaction: a crash can leave neither zero owners nor two."""
    first = pair_local(world, secret)["token"]
    before = len(world.devices.list_devices())

    def boom(name, platform):
        raise RuntimeError("disk full")

    monkeypatch.setattr(world.devices, "_new_device", boom)
    with pytest.raises(RuntimeError):
        world.devices.mint_owner("again", "macos", by="same-mac-pairing", replaces=("same-mac-pairing",))
    monkeypatch.undo()

    assert sessions_status(world, first) == 200, "the earlier credential was revoked by a mint that never happened"
    assert len(world.devices.list_devices()) == before


def test_a_listener_that_fails_cannot_undo_or_break_the_new_credential(world, secret, pairing_dir, caplog):
    first = pair_local(world, secret)["token"]

    def explode(ids):
        raise RuntimeError("listener bug")

    world.devices.add_revoke_listener(explode)
    second = pair_local(world, lp.replace_secret(pairing_dir))["token"]
    assert sessions_status(world, second) == 200
    assert sessions_status(world, first) == 401


def test_the_registry_announces_a_revocation_once_and_only_when_something_was_revoked(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    seen: list[list[str]] = []
    store.add_revoke_listener(lambda ids: seen.append(list(ids)))
    a = store.mint("a", "ios")["id"]
    assert store.revoke(a) is True and seen == [[a]]
    assert store.revoke(a) is True and seen == [[a]], "revoking a revoked device announces nothing"
    assert store.revoke("nobody") is False and seen == [[a]]


# --------------------------------------------------------------------------- #
# A revoked token's OPEN WebSocket is closed (it used to stay open and keep receiving)
# --------------------------------------------------------------------------- #


class ClosableSock:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.closed: list[tuple[int, str]] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))


def attach_socket(world: World, token: str) -> ClosableSock:
    ws = ClosableSock()
    world.bridge._clients.add(ws)
    world.bridge._ws_identity[ws] = verify_token(token, GLOBAL, world.devices)
    return ws


def test_revoking_a_device_closes_its_open_socket_and_stops_its_frames(world):
    sid = world.device_session("a")
    ws = attach_socket(world, world.a["token"])
    other = attach_socket(world, world.b["token"])

    assert call(world, GLOBAL, "DELETE", f"/api/devices/{world.a['id']}").status_code == 200

    async def after():
        await asyncio.sleep(0)             # let the scheduled close run
        await world.bridge.broadcast({"type": "chat_delta", "timestamp": 1.0, "payload": {"session_id": sid}})

    asyncio.run(after())
    assert ws not in world.bridge._clients and ws.frames == []
    assert world.bridge._ws_identity.get(ws) is None
    assert other in world.bridge._clients and other.closed == []


def test_replacing_an_owner_credential_closes_the_old_ones_socket(world, secret, pairing_dir):
    first = pair_local(world, secret)["token"]
    old_ws = attach_socket(world, first)
    second = pair_local(world, lp.replace_secret(pairing_dir))["token"]
    new_ws = attach_socket(world, second)

    async def settle():
        await asyncio.sleep(0)

    asyncio.run(settle())
    assert old_ws not in world.bridge._clients
    assert new_ws in world.bridge._clients


def test_a_revoked_sockets_further_commands_do_nothing(world):
    """Between the revoke and the close finishing, the socket still reads frames; it must act as nobody."""
    owner = mint_owner(world)
    ws = attach_socket(world, owner["token"])
    world.devices.revoke(owner["id"])
    before = world.lcm.count_all("ios:late")

    async def late():
        await world.bridge._handle_client_message(ws, json.dumps(
            {"type": "send_message", "payload": {"session_id": "ios:late", "content": "after revoke"}}))

    asyncio.run(late())
    assert world.lcm.count_all("ios:late") == before


@pytest.mark.asyncio
async def test_over_a_real_socket_a_revoked_owner_is_closed_4401(world):
    websockets = pytest.importorskip("websockets")
    from websockets.exceptions import ConnectionClosed

    owner = mint_owner(world)
    await world.bridge.start(host="127.0.0.1", port=0)
    port = world.bridge._server.sockets[0].getsockname()[1]
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.send(json.dumps({"type": "auth", "token": owner["token"]}))
            assert json.loads(await asyncio.wait_for(ws.recv(), 3))["type"] == "connected"
            world.devices.revoke(owner["id"])
            with pytest.raises(ConnectionClosed) as closed:
                await asyncio.wait_for(ws.recv(), 3)
            assert closed.value.rcvd is not None and closed.value.rcvd.code == 4401
    finally:
        await world.bridge.stop()
