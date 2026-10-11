"""A requester that paired over TLS is told to connect back over TLS (contract §7.3; found live by the Beacon session).

``GET /api/pair/requests/{id}`` answers an approved request with ``endpoints``: where the new device should connect
from now on. ``_endpoints`` built them as ``http://<host>`` and ``ws://<hostname>:<web.ws_port>`` whatever listener the
request came in on. In home-network mode that is wrong twice over: REST is plain ``http`` although the requester came in
on the TLS port, and the plain WebSocket port listens on ``127.0.0.1`` only, so ``ws://<lan address>:8010`` cannot be
reached from another computer at all. Beacon, correctly, refuses a plain address on a TLS join, so a second computer
could pair and then never connect. The contract promises (§7.2's example)::

    "endpoints": {"rest": "https://192.168.1.20:8006", "ws": "wss://192.168.1.20:8011"},
    "tls": {"spki_sha256": "<hex, 64 chars>"}          // null on a plain-HTTP instance

and the daemon also answered ``"tls": null`` unconditionally, so a TLS requester had no pin to check.

What is pinned:

* a request that came in on the TLS REST listener gets ``https`` for REST, ``wss`` on the CONFIGURED TLS WebSocket port
  (``web.tls_ws_port``, default 8011) for the socket, and the full SPKI pin;
* a plain request, loopback or otherwise, keeps exactly today's answer, home-network mode or not;
* which listener a request came in on is decided by the SERVER (the scheme it served and the local port), never by
  a header the requester sent: a ``Host`` that names the TLS port, or ``X-Forwarded-Proto: https``, on the plain
  listener changes nothing;
* the host in the answer is still the one the requester used, with IPv6 literals kept bracketed.

Expected values are literals, not computed from the code under test.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web import tls as T  # noqa: E402
from tests.support.pairing_world import GLOBAL, World  # noqa: E402

HOME = {"network": {"home_network": True}}
PEER = ("192.0.2.10", 50000)


def home_world(tmp_path, *, web: dict | None = None, plan: bool = True) -> World:
    """A daemon in home-network mode: the TLS listener set is up (``app.state.lan_tls``), as the launcher sets it."""
    world = World(tmp_path, config={**HOME, "web": {"api_token": GLOBAL, **(web or {})}})
    if plan:
        world.app.state.lan_tls = T.lan_tls_plan({**HOME, "web": web or {}}, "0.0.0.0")
    return world


def poll_as(world: World, *, base_url: str, headers: dict | None = None):
    """Approve a request from a stranger, then poll it from a client that came in on *base_url*."""
    created, _requester, _client = world.created()
    assert world.approve(created).status_code == 200
    client = TestClient(world.app, base_url=base_url, client=PEER)
    return world.poll(created, client=client, headers=headers)


class TestAnswerOverTls:
    def test_a_request_on_the_tls_listener_gets_https_wss_and_the_pin(self, tmp_path):
        world = home_world(tmp_path)
        body = poll_as(world, base_url="https://192.0.2.20:8006").json()
        assert body["status"] == "approved"
        assert body["endpoints"] == {"rest": "https://192.0.2.20:8006", "ws": "wss://192.0.2.20:8011"}
        assert body["tls"] == {"spki_sha256": hashlib.sha256(world.der).hexdigest()}
        assert len(body["tls"]["spki_sha256"]) == 64

    def test_the_configured_tls_ports_are_read_not_the_defaults(self, tmp_path):
        world = home_world(tmp_path, web={"tls_api_port": 9006, "tls_ws_port": 9011})
        body = poll_as(world, base_url="https://192.0.2.20:9006").json()
        assert body["endpoints"] == {"rest": "https://192.0.2.20:9006", "ws": "wss://192.0.2.20:9011"}

    def test_the_plain_ws_port_is_never_offered_over_tls(self, tmp_path):
        """8010 listens on this machine only in home-network mode: an address at it cannot work from another computer."""
        world = home_world(tmp_path, web={"ws_port": 8010})
        ws = poll_as(world, base_url="https://192.0.2.20:8006").json()["endpoints"]["ws"]
        assert ":8010" not in ws and ws.startswith("wss://")

    @pytest.mark.parametrize("host,rest,ws", [
        ("mac-mini.local:8006", "https://mac-mini.local:8006", "wss://mac-mini.local:8011"),
        ("192.0.2.20:8006", "https://192.0.2.20:8006", "wss://192.0.2.20:8011"),
        ("[fe80::1]:8006", "https://[fe80::1]:8006", "wss://[fe80::1]:8011"),
    ])
    def test_the_host_is_the_one_the_requester_used(self, tmp_path, host, rest, ws):
        """The connection arrives on the TLS port; the Host header is what the requester typed (an IPv6 literal
        stays bracketed, which the test client cannot express in a URL, so it goes in the header)."""
        world = home_world(tmp_path)
        body = poll_as(world, base_url="https://192.0.2.20:8006", headers={"Host": host}).json()
        assert body["endpoints"] == {"rest": rest, "ws": ws}

    def test_a_host_that_is_not_a_host_still_yields_no_endpoints(self, tmp_path):
        world = home_world(tmp_path)
        try:
            response = poll_as(world, base_url="https://192.0.2.20:8006", headers={"Host": "a b"})
        except Exception:       # the client library itself may refuse to send it
            return
        assert response.json()["endpoints"] == {"rest": None, "ws": None}


class TestPlainRequestsAreUnchanged:
    @pytest.mark.parametrize("plan", [True, False], ids=["home network up", "no tls listener"])
    def test_a_plain_loopback_request_gets_todays_answer(self, tmp_path, plan):
        world = home_world(tmp_path, plan=plan)
        body = poll_as(world, base_url="http://127.0.0.1:8005").json()
        assert body["endpoints"] == {"rest": "http://127.0.0.1:8005", "ws": "ws://127.0.0.1:8010"}
        assert body["tls"] is None

    def test_a_plain_request_on_another_host_gets_todays_answer(self, tmp_path):
        world = home_world(tmp_path, plan=False)
        body = poll_as(world, base_url="http://192.0.2.20:8005").json()
        assert body["endpoints"] == {"rest": "http://192.0.2.20:8005", "ws": "ws://192.0.2.20:8010"}
        assert body["tls"] is None

    def test_a_custom_plain_ws_port_is_still_read_for_a_plain_request(self, tmp_path):
        world = home_world(tmp_path, web={"ws_port": 9010}, plan=False)
        assert poll_as(world, base_url="http://192.0.2.20:8005").json()["endpoints"]["ws"] == "ws://192.0.2.20:9010"


class TestTheServerDecidesNotTheRequester:
    def test_a_plain_listener_does_not_turn_tls_because_the_host_names_the_tls_port(self, tmp_path):
        world = home_world(tmp_path)
        created, _r, _c = world.created()
        world.approve(created)
        plain = TestClient(world.app, base_url="http://127.0.0.1:8005", client=PEER)
        body = world.poll(created, client=plain, headers={"Host": "192.0.2.20:8006"}).json()
        assert body["endpoints"]["rest"].startswith("http://") and body["endpoints"]["ws"].startswith("ws://")
        assert body["tls"] is None

    def test_a_forwarded_proto_header_does_not_turn_a_plain_request_into_tls(self, tmp_path):
        world = home_world(tmp_path)
        body = poll_as(world, base_url="http://127.0.0.1:8005",
                       headers={"X-Forwarded-Proto": "https", "Forwarded": "proto=https"}).json()
        assert body["endpoints"] == {"rest": "http://127.0.0.1:8005", "ws": "ws://127.0.0.1:8010"}
        assert body["tls"] is None

    def test_the_tls_port_number_alone_is_not_tls(self, tmp_path):
        """Both conditions hold or it is plain: a request the daemon served as http is never answered as TLS."""
        world = home_world(tmp_path)
        body = poll_as(world, base_url="http://192.0.2.20:8006").json()
        assert body["endpoints"] == {"rest": "http://192.0.2.20:8006", "ws": "ws://192.0.2.20:8010"}
        assert body["tls"] is None

    def test_https_that_did_not_come_in_on_the_tls_listener_is_not_promised_tls_ports(self, tmp_path):
        """A TLS-terminating proxy in front of some other port: the TLS ws port is not reachable the way it implies."""
        world = home_world(tmp_path)
        body = poll_as(world, base_url="https://192.0.2.20:8443").json()
        assert body["endpoints"]["ws"].startswith("ws://") and body["tls"] is None

    def test_https_with_no_tls_listener_up_is_not_promised_tls_ports(self, tmp_path):
        world = home_world(tmp_path, plan=False)
        body = poll_as(world, base_url="https://192.0.2.20:8006").json()
        assert body["endpoints"]["ws"] == "ws://192.0.2.20:8010" and body["tls"] is None


class TestTheSharedHelper:
    """One rule for 'this request came in on the TLS listener', so no second place can disagree with it."""

    def test_it_returns_the_plan_only_for_a_request_on_the_tls_rest_port(self, tmp_path):
        world = home_world(tmp_path)

        class Probe:
            def __init__(self, scheme, port, plan):
                self.url = type("U", (), {"scheme": scheme})()
                self.scope = {"server": ("192.0.2.20", port)}
                self.app = type("A", (), {"state": type("S", (), {"lan_tls": plan})()})()

        plan = world.app.state.lan_tls
        assert T.request_tls_plan(Probe("https", 8006, plan)) is plan
        assert T.request_tls_plan(Probe("http", 8005, plan)) is None
        assert T.request_tls_plan(Probe("http", 8006, plan)) is None
        assert T.request_tls_plan(Probe("https", 8443, plan)) is None
        assert T.request_tls_plan(Probe("https", 8006, None)) is None

    def test_a_server_that_reports_no_local_port_is_judged_by_scheme_and_plan(self):
        class Probe:
            url = type("U", (), {"scheme": "https"})()
            scope: dict = {}

        probe = Probe()
        plan = T.lan_tls_plan(HOME, "0.0.0.0")
        probe.app = type("A", (), {"state": type("S", (), {"lan_tls": plan})()})()
        assert T.request_tls_plan(probe) is plan


# --------------------------------------------------------------------------- #
# Real listeners: what the daemon is told about each connection, not what a test client says
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_over_real_uvicorn_listeners_the_tls_one_answers_tls_and_the_plain_one_does_not(tmp_path, monkeypatch):
    """The rule reads ``scope["scheme"]`` and ``scope["server"]``; the test client makes both up from the URL. Here a
    real TLS listener and a real plain one serve the same pairing app, on ports the OS chose, and the same approved
    request is polled over each."""
    import httpx

    from prometheus.config.instance_cert import ensure_instance_certificate
    from prometheus.web.server import start_web
    from prometheus.web.serving import bound_port
    from tests.support.pairing_world import Requester
    from tests.test_lan_tls import _wait_started

    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(tmp_path / "node"))
    world = World(tmp_path, config={**HOME, "web": {"api_token": GLOBAL}})
    ensure_instance_certificate()                          # over the instance key the World made
    node = tmp_path / "node"
    boot_plan = T.LanTlsPlan(host="127.0.0.1", api_port=0, ws_port=0,
                             certfile=node / "instance.crt", keyfile=node / "instance.key")
    app = world.app
    plain = asyncio.create_task(start_web(app, host="127.0.0.1", port=0))
    secure = asyncio.create_task(start_web(app, host="127.0.0.1", port=0, tls=boot_plan))
    try:
        plain_port = bound_port(await _wait_started(app, "http_server"))
        tls_port = bound_port(await _wait_started(app, "https_server"))
        # What the launcher publishes once it knows the ports it bound (a real daemon names them in config).
        app.state.lan_tls = T.LanTlsPlan(host="127.0.0.1", api_port=tls_port, ws_port=49011,
                                         certfile=boot_plan.certfile, keyfile=boot_plan.keyfile)
        requester = Requester()
        async with httpx.AsyncClient(verify=False) as client:
            made = await client.post(f"http://127.0.0.1:{plain_port}/api/pair/requests", json={
                "device_name": "Jennifer's MacBook", "platform": "macos", "public_key": requester.public_b64})
            assert made.status_code == 201, made.text
            created = made.json()
            approved = await client.post(
                f"http://127.0.0.1:{plain_port}/api/pair/requests/{created['request_id']}/approve",
                headers={"Authorization": f"Bearer {GLOBAL}"}, json={})
            assert approved.status_code == 200, approved.text
            secret = {"X-Pairing-Secret": created["poll_secret"]}
            poll = f"/api/pair/requests/{created['request_id']}"
            over_tls = (await client.get(f"https://127.0.0.1:{tls_port}{poll}", headers=secret)).json()
            world.clock.now += 1.1                         # the contract's one-poll-a-second floor
            over_plain = (await client.get(f"http://127.0.0.1:{plain_port}{poll}", headers=secret)).json()
        assert over_tls["endpoints"] == {"rest": f"https://127.0.0.1:{tls_port}", "ws": "wss://127.0.0.1:49011"}
        assert over_tls["tls"] == {"spki_sha256": hashlib.sha256(world.der).hexdigest()}
        assert over_plain["endpoints"] == {"rest": f"http://127.0.0.1:{plain_port}", "ws": "ws://127.0.0.1:8010"}
        assert over_plain["tls"] is None
    finally:
        for name in ("http_server", "https_server"):
            server = getattr(app.state, name, None)
            if server is not None:
                server.should_exit = True
        await asyncio.wait_for(asyncio.gather(plain, secure), 10)
