"""A remote Beacon becomes an OWNER device with a code minted on the machine Prometheus runs on.

THE GAP
-------
An owner device (operator-equivalent; "a person" for a worker's approval since #744) came only from same-Mac
pairing. Will's MacBook Beacon reaches the Mac mini over Tailscale with the API token, which #744 refuses for a
worker's approval (a model can read it), and an approved device is scoped to its own conversations. So nothing
on another computer could approve a worker.

THE ROUTES
----------
    POST   /api/pair/owner-codes         the global token, from THIS host: mint a code (8 digits, 5 min, once)
    GET    /api/pair/owner-codes/{id}    the same gate: pending | redeemed (who, from where) | expired | ...
    DELETE /api/pair/owner-codes/{id}    the same gate: cancel it
    POST   /api/pair/owner               PUBLIC: {code, device_name, platform} -> an owner device token

Nothing remote can mint: an owner device is not root (it cannot enrol devices, so it cannot mint owner codes
either), a scoped device is default-denied, and the global token is refused unless the request came from this
host (a loopback peer, or the daemon's own address) without a relay header. The code, not the caller, is what
the public route checks; every way it can be wrong answers the same ``401 invalid_code``. Five wrong tries shut a
source out and 25 in all burn the code. Each issue, redemption and lock is told to the operator's sockets (and
only theirs), so a code nobody at the machine meant to mint is seen.

Real FastAPI app and DeviceStore (tests/support/pairing_world.py). The peer and the address a request arrived
on are set through the test client (``client=``, ``base_url=``), which is what Starlette puts in the scope.
"""

from __future__ import annotations

import logging
import re

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config import api_token as api_token_module  # noqa: E402
from prometheus.config import owner_codes as oc  # noqa: E402
from prometheus.config.api_token import verify_token  # noqa: E402
from prometheus.web import route_access  # noqa: E402
from prometheus.web.public_routes import PUBLIC_ROUTES  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.pairing_world import GLOBAL, World  # noqa: E402

TAILNET_SELF = "192.0.2.50"
MACBOOK = "192.0.2.77"


@pytest.fixture
def world(tmp_path) -> World:
    w = World(tmp_path, bridge=True)
    w.app.state.owner_codes.clock = w.clock
    w.app.state.owner_codes.wall = w.clock
    return w


def here(world: World) -> TestClient:
    """A terminal on the daemon's machine, over loopback."""
    return TestClient(world.app, base_url="http://127.0.0.1:8005", client=("127.0.0.1", 50123))


def own_address(world: World) -> TestClient:
    """A terminal on the daemon's machine reaching a daemon bound to its tailnet address only."""
    return TestClient(world.app, base_url=f"http://{TAILNET_SELF}:8005", client=(TAILNET_SELF, 50124))


def remote(world: World, peer: str = MACBOOK) -> TestClient:
    return TestClient(world.app, base_url=f"http://{TAILNET_SELF}:8005", client=(peer, 50125))


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def mint(world: World, client: TestClient | None = None, token: str = GLOBAL, **kw):
    return (client or here(world)).post("/api/pair/owner-codes", headers={**bearer(token), **kw.pop("headers", {})},
                                         **kw)


def minted(world: World) -> dict:
    response = mint(world)
    assert response.status_code == 201, response.text
    return response.json()


def redeem(world: World, code: str, client: TestClient | None = None, **body):
    payload = {"code": code, "device_name": "Will's MacBook Pro", "platform": "macos", **body}
    return (client or remote(world)).post("/api/pair/owner", json=payload)


def status(world: World, code_id: str):
    return here(world).get(f"/api/pair/owner-codes/{code_id}", headers=bearer(GLOBAL))


def wrong_for(code: str) -> str:
    return f"{(int(code) + 1) % 10**8:08d}"


# ── minting: the global token, from this host ────────────────────────────────

def test_the_global_token_on_this_host_mints_a_code(world):
    body = minted(world)
    assert set(body) == {"code_id", "code", "expires_at", "ttl_seconds"}
    assert re.fullmatch(r"[0-9]{8}", body["code"])
    assert body["ttl_seconds"] == 300 and body["expires_at"] == int(world.clock.now) + 300


def test_a_daemon_bound_to_one_address_is_reached_from_its_own_machine_at_that_address(world):
    assert mint(world, own_address(world)).status_code == 201


def test_the_global_token_from_another_computer_mints_nothing(world):
    response = mint(world, remote(world))
    assert response.status_code == 403 and response.json()["error"] == "not_this_machine"


@pytest.mark.parametrize("relay", [{"X-Forwarded-For": MACBOOK}, {"Forwarded": f"for={MACBOOK}"},
                                   {"X-Real-IP": MACBOOK}, {"Via": "1.1 nginx"}, {"CF-Connecting-IP": MACBOOK}])
def test_a_relayed_request_mints_nothing_even_from_loopback(world, relay):
    response = mint(world, headers=relay)
    assert response.status_code == 403 and response.json()["error"] == "not_this_machine"


def test_an_owner_device_mints_nothing_it_is_not_root(world):
    response = mint(world, token=world.owner["token"])
    assert response.status_code == 403 and response.json()["error"] == "global_token_only"


def test_a_scoped_device_is_refused_by_the_gate(world):
    response = mint(world, token=world.scoped["token"])
    assert response.status_code == 403 and response.json()["error"] == "operator_only"


def test_no_token_is_a_401(world):
    assert here(world).post("/api/pair/owner-codes").status_code == 401


def test_a_web_page_mints_nothing(world):
    response = mint(world, headers={"Origin": "http://127.0.0.1:8005"})
    assert response.status_code == 400 and response.json()["error"] == "browser_not_allowed"


def test_the_status_and_cancel_routes_take_the_same_gate(world):
    code_id = minted(world)["code_id"]
    for method in ("GET", "DELETE"):
        url = f"/api/pair/owner-codes/{code_id}"
        assert remote(world).request(method, url, headers=bearer(GLOBAL)).status_code == 403
        assert here(world).request(method, url, headers=bearer(world.owner["token"])).status_code == 403
        assert here(world).request(method, url, headers=bearer(world.scoped["token"])).status_code == 403


def test_with_auth_off_there_are_no_owner_codes(tmp_path):
    from prometheus.config.device_store import DeviceStore

    app = create_app({"web": {"api_token": ""}}, device_store=DeviceStore(tmp_path / "devices.db"))
    client = TestClient(app, base_url="http://127.0.0.1:8005", client=("127.0.0.1", 50123))
    for method, url, body in (("POST", "/api/pair/owner-codes", None),
                              ("POST", "/api/pair/owner", {"code": "12345678", "device_name": "x"})):
        response = client.request(method, url, json=body)
        assert response.status_code == 403 and response.json()["error"] == "pairing_unavailable", (url, response.text)


# ── redeeming: the public route ──────────────────────────────────────────────

def test_the_code_makes_the_remote_beacon_an_owner_device_of_its_own(world):
    code = minted(world)["code"]
    response = redeem(world, code)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"token", "device_id", "name", "owner", "endpoints", "tls"}
    assert body["owner"] is True and body["name"] == "Will's MacBook Pro"
    assert body["token"] != GLOBAL
    identity = verify_token(body["token"], GLOBAL, world.devices)
    assert identity is not None and identity.id == body["device_id"]
    assert identity.owner is True and identity.is_operator is True and not identity.is_global
    assert response.headers.get("cache-control") == "no-store"


def test_it_is_listed_as_an_owner_device_from_an_owner_code(world):
    device_id = redeem(world, minted(world)["code"]).json()["device_id"]
    rows = {r["id"]: r for r in world.as_("global", "GET", "/api/devices").json()}
    assert rows[device_id]["owner"] is True
    assert rows[device_id]["owner_source"] == api_token_module.OWNER_SOURCE_OWNER_CODE == "owner-code"
    assert rows[device_id]["platform"] == "macos"


def test_the_answer_says_where_to_connect_from_the_address_it_used(world):
    body = redeem(world, minted(world)["code"]).json()
    assert body["endpoints"] == {"rest": f"http://{TAILNET_SELF}:8005", "ws": f"ws://{TAILNET_SELF}:8010"}
    assert body["tls"] is None


def test_the_new_owner_device_is_the_operator(world):
    token = redeem(world, minted(world)["code"]).json()["token"]
    assert world.as_(token, "GET", "/api/pair/requests").status_code == 200


def test_the_new_owner_device_is_not_root(world):
    token = redeem(world, minted(world)["code"]).json()["token"]
    response = world.as_(token, "POST", "/api/devices", json={"name": "attacker", "platform": "ios"})
    assert response.status_code == 401
    assert mint(world, token=token).status_code == 403


def test_same_mac_re_pairing_never_revokes_it(world):
    token = redeem(world, minted(world)["code"]).json()["token"]
    world.devices.mint_owner("Beacon on this Mac", "macos", by=api_token_module.OWNER_SOURCE_SAME_MAC,
                             replaces=api_token_module.SAME_MAC_OWNER_SOURCES)
    assert verify_token(token, GLOBAL, world.devices) is not None


def test_two_remote_computers_each_keep_their_own_owner_device(world):
    first = redeem(world, minted(world)["code"], device_name="MacBook").json()["token"]
    second = redeem(world, minted(world)["code"], device_name="iPad", platform="ios").json()["token"]
    assert verify_token(first, GLOBAL, world.devices) is not None
    assert verify_token(second, GLOBAL, world.devices) is not None


@pytest.mark.parametrize("typed", ["{a}-{b}", "{a} {b}"])
def test_the_code_may_be_typed_in_two_groups(world, typed):
    code = minted(world)["code"]
    assert redeem(world, typed.format(a=code[:4], b=code[4:])).status_code == 200


def test_it_works_once(world):
    code = minted(world)["code"]
    assert redeem(world, code).status_code == 200
    second = redeem(world, code, client=remote(world, "192.0.2.99"))
    assert second.status_code == 401 and second.json()["error"] == "invalid_code"


def test_every_failure_is_one_answer(world):
    """Wrong, expired, used, replaced, cancelled, locked or none at all: a guesser learns nothing."""
    answers = set()
    answers.add(redeem(world, "12345678").text)                                     # none live
    code = minted(world)["code"]
    answers.add(redeem(world, wrong_for(code)).text)                                # wrong
    redeem(world, code)
    answers.add(redeem(world, code).text)                                           # used
    old = minted(world)["code"]
    minted(world)
    answers.add(redeem(world, old).text)                                            # replaced
    cancelled = minted(world)
    here(world).delete(f"/api/pair/owner-codes/{cancelled['code_id']}", headers=bearer(GLOBAL))
    answers.add(redeem(world, cancelled["code"]).text)                              # cancelled
    shut = minted(world)["code"]
    for _ in range(oc.MAX_WRONG_PER_SOURCE):
        redeem(world, wrong_for(shut), client=remote(world, "198.51.100.4"))
    answers.add(redeem(world, shut, client=remote(world, "198.51.100.4")).text)   # this source shut out
    busy = minted(world)["code"]
    fill_the_window(world, busy)
    answers.add(redeem(world, busy).text)                                           # too many guesses this minute
    world.clock.now += 60
    stale = minted(world)["code"]
    world.clock.now += 301
    answers.add(redeem(world, stale).text)                                          # expired
    assert len(answers) == 1, answers
    assert redeem(world, "12345678").status_code == 401


def fill_the_window(world: World, code: str) -> None:
    """Ten wrong tries, five from each of two addresses: a minute's worth."""
    for n in range(10):
        redeem(world, wrong_for(code), client=remote(world, f"198.51.100.{10 + n // 5}"))


def test_five_wrong_tries_shut_that_source_out_and_leave_the_code_to_everyone_else(world):
    body = minted(world)
    guesser = remote(world, "198.51.100.4")
    for _ in range(oc.MAX_WRONG_PER_SOURCE):
        assert redeem(world, wrong_for(body["code"]), client=guesser).status_code == 401
    assert redeem(world, body["code"], client=guesser).status_code == 401
    assert status(world, body["code_id"]).json()["status"] == "pending"
    assert redeem(world, body["code"]).status_code == 200


def test_strangers_guessing_cannot_burn_the_code(world):
    """The redeem route is public: a wrong guess must never be an off switch for the owner's code. Thirty wrong
    tries from six addresses (25 used to lock it): while a minute's window is full even the right code is refused
    unread, and once it empties the code still works."""
    body = minted(world)
    for burst in range(3):
        for n in range(10):
            redeem(world, wrong_for(body["code"]), client=remote(world, f"198.51.100.{10 + burst * 2 + n // 5}"))
        assert redeem(world, body["code"]).status_code == 401
        world.clock.now += 61
    assert status(world, body["code_id"]).json()["status"] == "pending"
    assert status(world, body["code_id"]).json()["wrong_attempts"] == 30
    assert redeem(world, body["code"]).status_code == 200


def test_the_terminal_hears_how_many_wrong_tries_there_were(world):
    body = minted(world)
    redeem(world, wrong_for(body["code"]), client=remote(world, "198.51.100.4"))
    redeem(world, wrong_for(body["code"]), client=remote(world, "198.51.100.5"))
    assert status(world, body["code_id"]).json()["wrong_attempts"] == 2


def test_an_expired_code_mints_nothing(world):
    body = minted(world)
    world.clock.now += 300
    assert redeem(world, body["code"]).status_code == 401
    assert status(world, body["code_id"]).json()["status"] == "expired"


def test_a_cancelled_code_mints_nothing(world):
    body = minted(world)
    assert here(world).delete(f"/api/pair/owner-codes/{body['code_id']}", headers=bearer(GLOBAL)).status_code == 200
    assert redeem(world, body["code"]).status_code == 401
    assert status(world, body["code_id"]).json()["status"] == "canceled"


def test_the_terminal_learns_who_redeemed_it_and_from_where(world):
    body = minted(world)
    assert status(world, body["code_id"]).json() == {
        "code_id": body["code_id"], "status": "pending", "expires_at": body["expires_at"], "redeemed_by": None,
        "wrong_attempts": 0}
    device_id = redeem(world, body["code"]).json()["device_id"]
    assert status(world, body["code_id"]).json()["redeemed_by"] == {
        "id": device_id, "name": "Will's MacBook Pro", "platform": "macos", "source": MACBOOK, "this_host": False}


@pytest.mark.parametrize("who,client_for,this_host", [
    ("a loopback Beacon", lambda w: here(w), True),
    ("a Beacon reaching the daemon's own address", lambda w: own_address(w), True),
    ("a Beacon on another computer", lambda w: remote(w), False),
    ("a relay on this machine", lambda w: TestClient(w.app, base_url="http://127.0.0.1:8005", client=("127.0.0.1", 50126),
                                                      headers={"X-Forwarded-For": MACBOOK}), False),
])
def test_redeemed_by_says_whether_it_was_this_host(world, who, client_for, this_host):
    """is_this_host() at redeem time: oara pair --owner says "this Mac" only when it was, never for a relayed loopback."""
    body = minted(world)
    assert redeem(world, body["code"], client=client_for(world)).status_code == 200, who
    assert status(world, body["code_id"]).json()["redeemed_by"]["this_host"] is this_host, who


def test_an_unknown_code_id_is_a_404(world):
    response = status(world, "0" * 32)
    assert response.status_code == 404


# ── the public route checks its own input ────────────────────────────────────

def test_a_web_page_cannot_redeem(world):
    code = minted(world)["code"]
    response = remote(world).post("/api/pair/owner", json={"code": code, "device_name": "x"},
                                  headers={"Origin": "https://evil.example"})
    assert response.status_code == 400 and response.json()["error"] == "browser_not_allowed"


@pytest.mark.parametrize("body", [
    {"code": "12345678"},                                            # no device name
    {"code": "12345678", "device_name": ""},
    {"code": "12345678", "device_name": "x" * 65},
    {"code": "12345678", "device_name": "line\nbreak"},
    {"code": "12345678", "device_name": "bidi‮override"},
])
def test_a_bad_device_name_is_a_400_and_spends_no_try(world, body):
    issued = minted(world)
    response = remote(world).post("/api/pair/owner", json=body)
    assert response.status_code == 400 and response.json()["error"] == "invalid_request"
    assert world.app.state.owner_codes.status(issued["code_id"]).wrong_attempts == 0


@pytest.mark.parametrize("code", [12345678, None, ["1234", "5678"], {"code": "12345678"}])
def test_a_code_that_is_not_a_string_is_a_400_and_spends_no_try(world, code):
    issued = minted(world)
    response = remote(world).post("/api/pair/owner", json={"code": code, "device_name": "x"})
    assert response.status_code == 400 and response.json() == {"error": "invalid_request", "fields": ["code"]}
    assert world.app.state.owner_codes.status(issued["code_id"]).wrong_attempts == 0


def test_a_missing_code_is_a_400(world):
    response = remote(world).post("/api/pair/owner", json={"device_name": "x"})
    assert response.status_code == 400 and response.json()["fields"] == ["code"]


def test_a_body_that_is_not_a_json_object_is_a_400(world):
    client = remote(world)
    assert client.post("/api/pair/owner", content=b"[1]", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post("/api/pair/owner", content=b"nope", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.post("/api/pair/owner", content=b"{}", headers={"Content-Type": "text/plain"}).status_code == 400


def test_a_large_body_is_refused_unread(world):
    response = remote(world).post("/api/pair/owner", content=b"{" + b" " * 5000 + b"}",
                                  headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_the_platform_is_narrowed_like_every_enrolment(world):
    device_id = redeem(world, minted(world)["code"], platform="windows").json()["device_id"]
    rows = {r["id"]: r for r in world.as_("global", "GET", "/api/devices").json()}
    assert rows[device_id]["platform"] == "other"


# ── the operator is told ─────────────────────────────────────────────────────

def sockets(world: World) -> dict:
    """Fake operator, owner-device and scoped sockets on the real bridge, as tests/test_pairing_ws_frames.py does."""
    from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity
    from tests.test_pairing_ws_frames import Sock

    socks = {}
    for who, identity in (
        ("op", GLOBAL_IDENTITY),
        ("owner", DeviceIdentity(id=world.owner["id"], name="Beacon on this Mac", platform="macos", owner=True)),
        ("scoped", DeviceIdentity(id=world.scoped["id"], name="a scoped phone", platform="ios")),
    ):
        ws = Sock()
        world.bridge._clients.add(ws)
        world.bridge._ws_identity[ws] = identity
        socks[who] = ws
    return socks


def owner_frames(ws) -> list[dict]:
    return [f for f in ws.frames if f["type"] == "owner_code"]


def test_every_operator_socket_hears_a_code_issued_and_redeemed_and_a_scoped_one_hears_nothing(world):
    socks = sockets(world)
    body = minted(world)
    redeemed = redeem(world, body["code"]).json()
    for who in ("op", "owner"):
        frames = owner_frames(socks[who])
        assert [f["payload"]["event"] for f in frames] == ["issued", "redeemed"], who
        issued, done = (f["payload"] for f in frames)
        assert issued == {"event": "issued", "code_id": body["code_id"], "expires_at": body["expires_at"],
                          "redeemed_by": None}
        assert done["redeemed_by"] == {"id": redeemed["device_id"], "name": "Will's MacBook Pro",
                                       "platform": "macos", "source": MACBOOK, "this_host": False}
    assert socks["scoped"].frames == []


def test_guessing_sends_no_frame_of_its_own(world):
    """No event was added (Beacon 0.4.2 knows issued/redeemed/locked), and "locked" can no longer happen."""
    socks = sockets(world)
    fill_the_window(world, minted(world)["code"])
    assert [f["payload"]["event"] for f in owner_frames(socks["op"])] == ["issued"]


def test_no_frame_carries_the_code_or_the_token(world):
    socks = sockets(world)
    body = minted(world)
    token = redeem(world, body["code"]).json()["token"]
    text = socks["op"].text()
    assert body["code"] not in text and oc.format_code(body["code"]) not in text and token not in text


# ── nothing secret leaves ────────────────────────────────────────────────────

def test_no_log_line_carries_the_code_or_the_token(world, caplog):
    caplog.set_level(logging.DEBUG)
    code = minted(world)["code"]
    redeem(world, wrong_for(code))
    token = redeem(world, code).json()["token"]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert code not in text and oc.format_code(code) not in text and token not in text and GLOBAL not in text
    assert "owner-code redeemed" in text and MACBOOK in text


def test_the_status_never_carries_the_code(world):
    body = minted(world)
    assert body["code"] not in status(world, body["code_id"]).text


# ── the route tables ─────────────────────────────────────────────────────────

def test_only_the_redeem_route_is_public():
    assert ("POST", "/api/pair/owner") in PUBLIC_ROUTES
    for method, path in (("POST", "/api/pair/owner-codes"), ("GET", "/api/pair/owner-codes/{code_id}"),
                         ("DELETE", "/api/pair/owner-codes/{code_id}")):
        assert (method, path) not in PUBLIC_ROUTES
        assert (method, path) in route_access.OPERATOR_ONLY
        assert (method, path) not in route_access.SCOPED_ALLOWED


# ── on a real socket (what uvicorn puts in the scope, not what a test client does) ────

def _serve(app, host: str):
    """The app behind ``start_web`` on a real socket at *host*, port 0-chosen, for the length of the block."""
    import asyncio
    import contextlib
    import socket
    import threading
    import time

    from prometheus.web.server import start_web

    @contextlib.contextmanager
    def serving():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]
        stop = threading.Event()

        def run() -> None:
            async def main() -> None:
                task = asyncio.ensure_future(start_web(app, host=host, port=port))
                while not stop.is_set() and not task.done():
                    await asyncio.sleep(0.02)
                server = getattr(app.state, "http_server", None)
                if server is not None:
                    server.should_exit = True
                    await asyncio.wait_for(task, 10)

            asyncio.run(main())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                socket.create_connection((host, port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield f"http://{host}:{port}"
        finally:
            stop.set()
            thread.join(15)

    return serving()


def test_a_real_loopback_connection_mints_and_a_relayed_one_does_not(world):
    import httpx

    with _serve(world.app, "127.0.0.1") as url:
        assert httpx.post(f"{url}/api/pair/owner-codes", headers=bearer(GLOBAL)).status_code == 201
        relayed = httpx.post(f"{url}/api/pair/owner-codes", headers={**bearer(GLOBAL), "X-Forwarded-For": MACBOOK})
        assert relayed.status_code == 403 and relayed.json()["error"] == "not_this_machine"


def test_a_real_connection_from_this_machine_to_its_own_lan_address_mints(world):
    """Binds this machine's own non-loopback address (port 0) for the length of the test, as the bind tests do."""
    import httpx

    from tests.support.listeners import local_non_loopback_address

    address = local_non_loopback_address()
    if address is None:
        pytest.skip("this machine has no non-loopback IPv4 address")
    with _serve(world.app, address) as url:
        response = httpx.post(f"{url}/api/pair/owner-codes", headers=bearer(GLOBAL))
        assert response.status_code == 201, response.text
