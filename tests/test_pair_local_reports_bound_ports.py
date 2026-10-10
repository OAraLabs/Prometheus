"""``POST /api/pair/local`` tells the client the ports the daemon is LISTENING on, not the ones its config names.

Beacon pairs with the one-time file secret and then connects to ``api_base_port`` and ``ws_port`` from the
answer. Those came from ``config["web"]``: what was asked for when the daemon started. The config can have
been edited since (it is the live dict the config routes write to), and a configured ``0`` asks the OS for any
free port, which the old answer even reported as 8005 (``int(0 or 8005)``). A client told the wrong port pairs
successfully and then cannot connect. So the answer is read from the listeners' own sockets
(``web/serving.bound_port``), and the config is the answer only when no listener is there to ask, which is the
case under Starlette's test client and never in a running daemon.

These tests run the REAL listeners (``start_web`` and ``WebSocketBridge.start``) on port 0, on 127.0.0.1 only,
against a config that names other ports.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

import httpx  # noqa: E402

from prometheus.config import local_pairing as lp  # noqa: E402
from prometheus.web.serving import bound_port  # noqa: E402
from prometheus.web.server import create_app, start_web  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402

DAEMON_TOKEN = "daemon-test-token-0123456789abcdef"
CONFIGURED_API, CONFIGURED_WS = 8123, 8124        # what the config says; nothing listens there


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(tmp_path / "pairing"))


async def _serve(app):
    """The daemon's own listeners, on ports the OS picks: start_web publishes app.state.http_server, and the
    bridge is published as app.state.ws_bridge, exactly as web/launcher.py does."""
    bridge = WebSocketBridge()
    await bridge.start(host="127.0.0.1", port=0)
    app.state.ws_bridge = bridge
    task = asyncio.create_task(start_web(app, host="127.0.0.1", port=0))
    for _ in range(200):
        server = getattr(app.state, "http_server", None)
        if server is not None and server.started:
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("uvicorn did not start")
    return bridge, task


async def _stop(app, bridge, task):
    app.state.http_server.should_exit = True
    await asyncio.wait_for(task, 10)
    await bridge.stop()


@pytest.mark.asyncio
async def test_the_answer_names_the_ports_the_daemon_is_listening_on():
    app = create_app({"web": {"api_token": DAEMON_TOKEN, "api_port": CONFIGURED_API, "ws_port": CONFIGURED_WS}})
    bridge, task = await _serve(app)
    try:
        api_port = bound_port(app.state.http_server)
        ws_port = bridge.bound_port
        assert api_port and ws_port and {api_port, ws_port}.isdisjoint({CONFIGURED_API, CONFIGURED_WS})
        secret = lp.mint_secret()
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{api_port}") as client:
            resp = await client.post("/api/pair/local", json={"code": secret})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert (body["api_base_port"], body["ws_port"]) == (api_port, ws_port), (
            "the client was told the config's ports, where nothing listens")
    finally:
        await _stop(app, bridge, task)


@pytest.mark.asyncio
async def test_a_configured_port_0_is_reported_as_the_port_the_os_chose():
    app = create_app({"web": {"api_token": DAEMON_TOKEN, "api_port": 0, "ws_port": 0}})
    bridge, task = await _serve(app)
    try:
        api_port = bound_port(app.state.http_server)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{api_port}") as client:
            body = (await client.post("/api/pair/local", json={"code": lp.mint_secret()})).json()
        assert body["api_base_port"] == api_port != 8005
        assert body["ws_port"] == bridge.bound_port != 8010
    finally:
        await _stop(app, bridge, task)


# ── reading a listener's port ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bound_port_reads_a_running_servers_socket_and_nothing_else():
    server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    try:
        expected = server.sockets[0].getsockname()[1]
        assert bound_port(server) == expected

        class Uvicornish:                      # uvicorn.Server keeps its asyncio servers in .servers
            servers = [server]

        assert bound_port(Uvicornish()) == expected
    finally:
        server.close()
        await server.wait_closed()
    assert bound_port(server) is None, "a closed server listens on nothing"
    assert bound_port(None) is None and bound_port(object()) is None


def test_a_bridge_that_has_not_started_has_no_port():
    assert WebSocketBridge().bound_port is None


def test_the_test_client_path_still_answers_from_the_config():
    """No listener to ask (Starlette's test client runs the app without one): the config is all there is."""
    from fastapi.testclient import TestClient

    app = create_app({"web": {"api_token": DAEMON_TOKEN, "api_port": CONFIGURED_API, "ws_port": CONFIGURED_WS}})
    client = TestClient(app, base_url="http://127.0.0.1:8123", client=("127.0.0.1", 50123))
    body = client.post("/api/pair/local", json={"code": lp.mint_secret()}).json()
    assert (body["api_base_port"], body["ws_port"]) == (CONFIGURED_API, CONFIGURED_WS)


def test_nothing_listens_on_the_configured_ports_in_these_tests():
    """Guard for the first test: if something did listen on 8123/8124 here, 'reported the config' and
    'reported the socket' could not be told apart."""
    for port in (CONFIGURED_API, CONFIGURED_WS):
        with socket.socket() as probe:
            probe.settimeout(0.2)
            assert probe.connect_ex(("127.0.0.1", port)) != 0, f"something listens on {port}"
