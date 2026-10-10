"""web.bind on REAL sockets: what the kernel was handed, and what a client gets.

Nothing here mocks the host argument. Each test starts the real server on an
ephemeral port and then asks two independent witnesses:

* the live servers themselves -- ``uvicorn.Server.servers[*].sockets`` and the
  ``websockets`` server's ``.sockets``, read with ``getsockname()``;
* the kernel -- ``lsof`` / ``/proc/net/tcp*`` for this process (and for a child
  daemon), which is how a stray listener nobody wrote a test for shows up.

The Host-header tests send hand-built requests over raw sockets, so the header is
exactly what the test says (a client library would normalise or duplicate it).

WHICH TESTS BIND 0.0.0.0 (every other test binds loopback only, port 0):

* ``TestCompatibility.test_launch_without_web_bind_still_binds_every_interface``
  -- REST + WS, closed within the test.
* ``TestCompatibility.test_setup_mode_without_web_bind_still_binds_every_interface``
  -- the setup server, closed within the test.
* ``TestStartupWarning.test_wide_launch_warns_exactly_once`` -- REST + WS, closed
  within the test.

On the parent commit (before the feature exists) every test that configures a
loopback bind also binds wide, because the setting is ignored -- that is the
failure being demonstrated, and it is the only time those tests do.

Ports are always 0 (or a just-freed ephemeral port for a child), never the
daemon's real 8005/8010.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

import uvicorn  # noqa: E402

from tests.support.listeners import (  # noqa: E402
    Listener,
    ListenersUnavailable,
    free_port,
    http_response,
    http_status,
    ipv6_loopback_usable,
    live_instances,
    local_non_loopback_address,
    package_src_root,
    process_listeners,
    repo_config_of_loaded_package,
    sockname,
    tcp_connects,
    ws_handshake_status,
)

BIND_ENV = "PROMETHEUS_WEB_BIND"
LOOPBACK_V4 = "127.0.0.1"


def _ws_server_cls() -> type:
    try:
        from websockets.asyncio.server import Server

        return Server
    except ImportError:  # websockets < 13
        from websockets.legacy.server import WebSocketServer

        return WebSocketServer


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (BIND_ENV, "PROMETHEUS_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _baseline() -> list[Listener] | None:
    try:
        return process_listeners()
    except ListenersUnavailable:
        return None


def _new_listeners(baseline: list[Listener] | None) -> list[Listener]:
    if baseline is None:
        pytest.skip("this machine offers no way to list this process's listening sockets")
    known = set(baseline)
    return [x for x in process_listeners() if x not in known]


# ---------------------------------------------------------------------------
# Running the real servers inside the test's event loop
# ---------------------------------------------------------------------------


def _seen(obj: Any, earlier: list[Any]) -> bool:
    return any(obj is old for old in earlier)


@dataclass
class Running:
    uvicorn_server: Any
    ws_server: Any

    @property
    def rest(self) -> tuple[str, int]:
        return sockname(self.uvicorn_server.servers[0].sockets[0])

    @property
    def ws(self) -> tuple[str, int]:
        return sockname(self.ws_server.sockets[0])

    @property
    def all_sockets(self) -> list[tuple[str, int]]:
        socks = [s for sv in self.uvicorn_server.servers for s in sv.sockets]
        socks += list(self.ws_server.sockets)
        return [sockname(s) for s in socks]


@contextlib.asynccontextmanager
async def launched(config: dict | None = None, **kwargs: Any):
    """``launch_web`` with ephemeral ports. Raises whatever launch_web raised if
    it refused to start; always stops everything it started."""
    from prometheus.web.launcher import launch_web

    ws_cls = _ws_server_cls()
    # Strong references, not ids: a collected server's id can be handed to the
    # next one, and an id-based "already seen" set would then hide the new server.
    seen_u = live_instances(uvicorn.Server)
    seen_w = live_instances(ws_cls)
    task = asyncio.create_task(
        launch_web(config if config is not None else {}, api_port=0, ws_port=0, **kwargs))
    us: Any = None
    ws: Any = None
    try:
        for _ in range(150):
            if task.done():
                task.result()  # re-raises a refusal; a clean exit is a bug here
                raise AssertionError("launch_web returned before serving")
            us = next((o for o in live_instances(uvicorn.Server)
                       if not _seen(o, seen_u) and o.started and o.servers), None)
            ws = next((o for o in live_instances(ws_cls) if not _seen(o, seen_w)), None)
            if us is not None and ws is not None:
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("launch_web did not come up within 15s")
        yield Running(us, ws)
    finally:
        if us is not None:
            us.should_exit = True
        if ws is not None:
            ws.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(ws.wait_closed(), 10)
        if not task.done():
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, 15)
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


@contextlib.asynccontextmanager
async def setup_mode(bind: str | None = None):
    """The real setup-mode uvicorn server (``_serve_setup_mode``) on port 0."""
    from prometheus.web.setup_server import PairingState, SetupModeState, _serve_setup_mode

    pairing = PairingState(code="042999")
    seen = live_instances(uvicorn.Server)
    args: tuple = (pairing, SetupModeState(), 0)
    task = asyncio.create_task(
        _serve_setup_mode(*args) if bind is None else _serve_setup_mode(*args, bind))
    us: Any = None
    try:
        for _ in range(150):
            if task.done():
                task.result()
                raise AssertionError("setup mode returned before serving")
            us = next((o for o in live_instances(uvicorn.Server)
                       if not _seen(o, seen) and o.started and o.servers), None)
            if us is not None:
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError("setup mode did not come up within 15s")
        yield us, pairing
    finally:
        if us is not None:
            us.should_exit = True
        if not task.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, 15)
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task


def _is_loopback(host: str) -> bool:
    ip = ipaddress.ip_address(host)
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped is not None and mapped.is_loopback)


# ---------------------------------------------------------------------------
# The real launch path (REST + WebSocket bridge)
# ---------------------------------------------------------------------------


class TestTheInstrumentsReadTheirFormats:
    """The kernel-side witnesses are parsers. Pin them on captured samples, so a
    mistake in one shows up here rather than as a silent skip or a false pass in
    CI, which reads /proc where this Mac reads lsof."""

    PROC_TCP = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 0100007F:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 31337 1 ffff8f3b4c1b7800 100 0 0 10 0\n"
        "   1: 00000000:1F91 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 31338 1 ffff8f3b4c1b7801 100 0 0 10 0\n"
        "   2: 0100007F:1F90 0100007F:D3A2 01 00000000:00000000 00:00000000 00000000  1000        0 31339 1 ffff8f3b4c1b7802 20 4 30 10 -1\n"
        "   3: 0A00020F:1F92 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 99999 1 ffff8f3b4c1b7803 100 0 0 10 0\n"
    )
    PROC_TCP6 = (
        "  sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 00000000000000000000000001000000:1F93 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 31340 1 ffff8f3b4c1b7900 100 0 0 10 0\n"
        "   1: 00000000000000000000000000000000:1F94 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 31341 1 ffff8f3b4c1b7901 100 0 0 10 0\n"
        "   2: 0000000000000000FFFF00000100007F:1F95 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 31342 1 ffff8f3b4c1b7902 100 0 0 10 0\n"
    )

    def test_proc_net_tcp_listen_rows_of_this_process_only(self):
        from tests.support.listeners import Listener, parse_proc_net_tcp

        got = parse_proc_net_tcp(self.PROC_TCP, {"31337", "31338", "31339"})
        assert got == [Listener("127.0.0.1", 8080), Listener("0.0.0.0", 8081)], (
            "row 2 is ESTABLISHED, row 3 belongs to another process")

    def test_proc_net_tcp6_addresses(self):
        from tests.support.listeners import Listener, parse_proc_net_tcp

        got = parse_proc_net_tcp(self.PROC_TCP6, {"31340", "31341", "31342"})
        assert got == [
            Listener("::1", 8083), Listener("::", 8084), Listener("::ffff:127.0.0.1", 8085)]

    def test_lsof_output(self):
        from tests.support.listeners import Listener, parse_lsof_listeners

        out = "p4242\nf4\nn127.0.0.1:8005\nf5\nn*:8010\nf6\nn[::1]:8011\nf7\nn[::]:8012\n"
        assert parse_lsof_listeners(out) == [
            Listener("127.0.0.1", 8005), Listener("*", 8010),
            Listener("::1", 8011), Listener("::", 8012)]

    def test_classification(self):
        from tests.support.listeners import Listener

        assert Listener("*", 1).is_wildcard and not Listener("*", 1).is_loopback
        assert Listener("0.0.0.0", 1).is_wildcard and Listener("::", 1).is_wildcard
        for host in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            assert Listener(host, 1).is_loopback and not Listener(host, 1).is_wildcard
        assert not Listener("192.0.2.10", 1).is_loopback


class TestLaunchPath:
    async def test_web_bind_in_config_makes_both_servers_loopback(self):
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            assert run.rest[0] == LOOPBACK_V4, f"REST bound {run.rest}"
            assert run.ws[0] == LOOPBACK_V4, (
                f"the WebSocket bridge bound {run.ws}: websockets.serve takes "
                "its own host, separately from uvicorn's"
            )

    async def test_the_environment_variable_makes_both_servers_loopback(self, monkeypatch):
        monkeypatch.setenv(BIND_ENV, LOOPBACK_V4)
        async with launched({}) as run:
            assert (run.rest[0], run.ws[0]) == (LOOPBACK_V4, LOOPBACK_V4)

    async def test_the_environment_variable_outranks_the_config(self, monkeypatch):
        monkeypatch.setenv(BIND_ENV, LOOPBACK_V4)
        async with launched({"web": {"bind": "0.0.0.0"}}) as run:
            assert (run.rest[0], run.ws[0]) == (LOOPBACK_V4, LOOPBACK_V4)

    async def test_an_explicit_bind_argument_outranks_environment_and_config(
        self, monkeypatch,
    ):
        """The daemon resolves flag > env > config once and hands launch_web the
        answer as `bind`; that answer is used as given."""
        monkeypatch.setenv(BIND_ENV, "0.0.0.0")
        async with launched({"web": {"bind": "0.0.0.0"}}, bind=LOOPBACK_V4) as run:
            assert (run.rest[0], run.ws[0]) == (LOOPBACK_V4, LOOPBACK_V4)

    async def test_localhost_means_the_ipv4_loopback_address(self):
        async with launched({"web": {"bind": "localhost"}}) as run:
            assert (run.rest[0], run.ws[0]) == (LOOPBACK_V4, LOOPBACK_V4)
            assert len(run.all_sockets) == 2, (
                "localhost must not fan out to every address the resolver returns")

    async def test_ipv6_loopback(self):
        if not ipv6_loopback_usable():
            pytest.skip("this machine cannot bind ::1")
        async with launched({"web": {"bind": "::1"}}) as run:
            assert (run.rest[0], run.ws[0]) == ("::1", "::1")
            assert await asyncio.to_thread(
                http_status, run.rest[1], f"[::1]:{run.rest[1]}", ip="::1") == 200

    async def test_an_invalid_bind_refuses_to_start_and_listens_on_nothing(self):
        before = _baseline()
        with pytest.raises(ValueError) as err:  # BindError is a ValueError
            async with launched({"web": {"bind": "nonsense"}}):
                pytest.fail("launch_web started although web.bind is not an address")
        assert type(err.value).__name__ == "BindError", err.value
        if before is not None:
            assert _new_listeners(before) == [], "a refused start left a listener behind"

    async def test_an_invalid_environment_variable_does_not_fall_back_to_the_config(
        self, monkeypatch,
    ):
        monkeypatch.setenv(BIND_ENV, "nonsense")
        with pytest.raises(ValueError) as err:  # BindError is a ValueError
            async with launched({"web": {"bind": LOOPBACK_V4}}):
                pytest.fail("fell back to the config after an invalid env var")
        assert type(err.value).__name__ == "BindError", err.value

    async def test_no_listener_in_the_process_is_wider_than_the_setting(self):
        baseline = _baseline()
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            fresh = _new_listeners(baseline)
            ports = {run.rest[1], run.ws[1]}
            assert {x.port for x in fresh} >= ports, (fresh, ports)
            wider = [x for x in fresh if x.host != LOOPBACK_V4]
            assert not wider, f"listening wider than web.bind={LOOPBACK_V4}: {wider}"

    async def test_a_non_loopback_address_of_this_machine_cannot_connect(self):
        other = local_non_loopback_address()
        if other is None:
            pytest.skip("this machine has no non-loopback IPv4 address to connect to")
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            for port in (run.rest[1], run.ws[1]):
                # The control: the same port IS serving, on loopback.
                assert await asyncio.to_thread(tcp_connects, LOOPBACK_V4, port)
                assert not await asyncio.to_thread(tcp_connects, other, port), (
                    f"{other}:{port} accepted a connection although web.bind is {LOOPBACK_V4}")


class TestCompatibility:
    """No web.bind, no flag, no env: 0.0.0.0, exactly as before. A Mac mini
    reached over Tailscale must not break."""

    async def test_launch_without_web_bind_still_binds_every_interface(self):
        async with launched({"web": {"enabled": True, "api_port": 8005, "ws_port": 8010}}) as run:
            assert run.rest[0] == "0.0.0.0", f"REST bound {run.rest}"
            assert run.ws[0] == "0.0.0.0", f"WS bound {run.ws}"
            # ...and, being wide, it answers whatever name it is reached by.
            # Starlette's TestClient sends "testserver"; tailnet clients send a
            # MagicDNS name or an address. None of them may be refused.
            for host in ("testserver", "evil.example", "host.tailnet.example:8005",
                         f"192.0.2.10:{run.rest[1]}"):
                status = await asyncio.to_thread(http_status, run.rest[1], host)
                assert status == 200, f"Host {host!r} -> {status} on a wide bind"
                assert await asyncio.to_thread(
                    ws_handshake_status, run.ws[1], host) == 101, host

    async def test_setup_mode_without_web_bind_still_binds_every_interface(self):
        async with setup_mode() as (server, _pairing):
            host, port = sockname(server.servers[0].sockets[0])
            assert host == "0.0.0.0"
            assert await asyncio.to_thread(http_status, port, "evil.example",
                                           path="/api/setup/status") == 200


# ---------------------------------------------------------------------------
# DNS rebinding: a loopback bind answers only to loopback names
# ---------------------------------------------------------------------------

GOOD_HOSTS = ["localhost:{port}", "localhost", "127.0.0.1:{port}", "127.0.0.1", "[::1]:{port}", "[::1]"]
BAD_HOSTS = ["evil.example", "evil.example:{port}", "localhost.evil.example",
             "127.0.0.1.evil.example", "192.0.2.10:{port}", "0.0.0.0:{port}"]


class TestHostGuardOnRealServers:
    async def test_rest_refuses_a_foreign_host_and_serves_a_loopback_one(self):
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            port = run.rest[1]
            for host in BAD_HOSTS:
                got = await asyncio.to_thread(http_status, port, host.format(port=port))
                assert got == 403, f"Host {host!r}: expected 403, got {got}"
            for host in GOOD_HOSTS:
                got = await asyncio.to_thread(http_status, port, host.format(port=port))
                assert got == 200, f"Host {host!r}: expected 200, got {got}"

    async def test_the_guard_also_covers_the_bundled_dashboard(self):
        """The static UI is a mount inside the app; the guard sits outside the lot."""
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            port = run.rest[1]
            assert await asyncio.to_thread(
                http_status, port, f"localhost:{port}", path="/") == 200
            assert await asyncio.to_thread(
                http_status, port, "evil.example", path="/") == 403
            assert await asyncio.to_thread(
                http_status, port, "evil.example", path="/app.js") == 403

    async def test_a_refused_request_never_reaches_the_app(self):
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            raw = await asyncio.to_thread(http_response, run.rest[1], "evil.example")
            head, _, body = raw.partition(b"\r\n\r\n")
            assert head.startswith(b"HTTP/1.1 403")
            assert json.loads(body)["error"] == "host_not_allowed"
            assert b"state" not in body, "the status payload leaked into a refusal"

    async def test_rest_refuses_two_host_headers(self):
        """A regression pin, not a red test: uvicorn's h11 already answers 400 to
        two Host headers, so this passes before the feature exists. The guard's
        own duplicate handling is covered at the ASGI level in test_web_loopback."""
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            port = run.rest[1]
            for hosts in (["localhost", "evil.example"], ["evil.example", "localhost"]):
                got = await asyncio.to_thread(http_status, port, hosts)
                assert got in (400, 403), (hosts, got)

    async def test_websocket_handshake_refuses_a_foreign_host_before_upgrading(self):
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            port = run.ws[1]
            for host in BAD_HOSTS:
                got = await asyncio.to_thread(
                    ws_handshake_status, port, host.format(port=port))
                assert got == 403, f"WS Host {host!r}: expected 403, got {got}"

    async def test_websocket_handshake_accepts_loopback_hosts(self):
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            port = run.ws[1]
            for host in GOOD_HOSTS:
                got = await asyncio.to_thread(
                    ws_handshake_status, port, host.format(port=port))
                assert got == 101, f"WS Host {host!r}: expected 101, got {got}"

    async def test_websocket_handshake_refuses_two_host_headers(self):
        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            for hosts in (["localhost", "evil.example"], ["evil.example", "localhost"]):
                got = await asyncio.to_thread(ws_handshake_status, run.ws[1], hosts)
                assert got in (400, 403), (hosts, got)

    async def test_a_real_websocket_client_still_connects_to_a_loopback_bind(self):
        import websockets

        async with launched({"web": {"bind": LOOPBACK_V4}}) as run:
            async with websockets.connect(f"ws://localhost:{run.ws[1]}") as client:
                first = json.loads(await asyncio.wait_for(client.recv(), 5))
            assert first["type"] == "connected"


# ---------------------------------------------------------------------------
# Setup mode (the unauthenticated pairing endpoint)
# ---------------------------------------------------------------------------


class TestSetupModeSockets:
    async def test_the_environment_variable_makes_setup_mode_loopback(self, monkeypatch):
        monkeypatch.setenv(BIND_ENV, LOOPBACK_V4)
        async with setup_mode() as (server, _pairing):
            assert sockname(server.servers[0].sockets[0])[0] == LOOPBACK_V4

    async def test_an_explicit_bind_makes_setup_mode_loopback(self):
        baseline = _baseline()
        async with setup_mode(LOOPBACK_V4) as (server, _pairing):
            socks = [sockname(s) for sv in server.servers for s in sv.sockets]
            assert socks and all(h == LOOPBACK_V4 for h, _ in socks), socks
            if baseline is not None:
                wider = [x for x in _new_listeners(baseline) if x.host != LOOPBACK_V4]
                assert not wider, f"setup mode listens wider than asked: {wider}"

    async def test_setup_mode_refuses_a_foreign_host_and_keeps_its_pairing_code(self):
        async with setup_mode(LOOPBACK_V4) as (server, pairing):
            port = sockname(server.servers[0].sockets[0])[1]
            for host in BAD_HOSTS:
                got = await asyncio.to_thread(
                    http_status, port, host.format(port=port), path="/api/setup/status")
                assert got == 403, f"Host {host!r}: expected 403, got {got}"
            for host in GOOD_HOSTS:
                got = await asyncio.to_thread(
                    http_status, port, host.format(port=port), path="/api/setup/status")
                assert got == 200, f"Host {host!r}: expected 200, got {got}"
            assert pairing.attempts_remaining == pairing.max_attempts

    async def test_a_pairing_attempt_with_a_foreign_host_is_not_counted(self):
        async with setup_mode(LOOPBACK_V4) as (server, pairing):
            port = sockname(server.servers[0].sockets[0])[1]

            def post(host: str) -> int:
                import http.client

                conn = http.client.HTTPConnection(LOOPBACK_V4, port, timeout=5)
                conn.putrequest("POST", "/api/setup/pair", skip_host=True,
                                skip_accept_encoding=True)
                conn.putheader("Host", host)
                body = b'{"code": "000000"}'
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(len(body)))
                conn.endheaders(body)
                status = conn.getresponse().status
                conn.close()
                return status

            assert await asyncio.to_thread(post, "evil.example") == 403
            assert pairing.failed_attempts == 0
            assert await asyncio.to_thread(post, f"localhost:{port}") == 401
            assert pairing.failed_attempts == 1


# ---------------------------------------------------------------------------
# The whole daemon process: setup mode -> configure -> real daemon -> restart
# ---------------------------------------------------------------------------


def _api(port: int, path: str, *, method: str = "GET", payload: dict | None = None,
         token: str | None = None, timeout: float = 5) -> tuple[int, Any]:
    req = urllib.request.Request(f"http://{LOOPBACK_V4}:{port}{path}", method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode() or "{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def _wait_for(predicate, *, timeout: float, what: str, proc: subprocess.Popen,
              out_path) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        assert proc.poll() is None, (
            f"daemon exited while waiting for {what}:\n{out_path.read_text(errors='replace')[-3000:]}")
        try:
            if predicate():
                return
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError,
                json.JSONDecodeError):
            pass
        time.sleep(0.3)
    raise AssertionError(
        f"timed out waiting for {what}:\n{out_path.read_text(errors='replace')[-3000:]}")


def _stop(proc: subprocess.Popen) -> None:
    """Stop the child we started, by the Popen we hold -- nothing else."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


class TestWholeDaemonProcess:
    def test_the_choice_survives_setup_mode_the_same_process_and_a_restart(self, tmp_path):
        from tests.test_setup_api_phase2 import _FakeLlamaCppHandler, _serve

        if repo_config_of_loaded_package().exists():
            pytest.skip("a live repo config exists; a child daemon would boot from it")
        try:
            process_listeners()
        except ListenersUnavailable as exc:
            pytest.skip(f"cannot list the child's listening sockets: {exc}")

        confdir = tmp_path / "confdir"
        api_port, ws_port = free_port(), free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith("PROMETHEUS_")}
        env.update({
            "PYTHONPATH": str(package_src_root()),
            "PYTHONUNBUFFERED": "1",
            "PROMETHEUS_CONFIG_DIR": str(confdir),
            "PROMETHEUS_DATA_DIR": str(tmp_path / "datadir"),
            "PROMETHEUS_ENV_FILE": str(tmp_path / "envfile"),
            "PROMETHEUS_WEB_API_PORT": str(api_port),
            "PROMETHEUS_WEB_WS_PORT": str(ws_port),
        })

        def assert_only_loopback(proc: subprocess.Popen, stage: str, *, expect: set[int]) -> None:
            listeners = process_listeners(proc.pid)
            assert {x.port for x in listeners} >= expect, (stage, listeners)
            wider = [x for x in listeners if not _is_loopback(x.host if x.host != "*" else "0.0.0.0")]
            assert not wider, f"[{stage}] listening beyond loopback: {wider}"

        def foreign_host_refused(port: int, path: str) -> None:
            assert http_status(port, "evil.example", path=path) == 403
            assert http_status(port, f"localhost:{port}", path=path) in (200, 401)

        out1 = tmp_path / "daemon1.out"
        out2 = tmp_path / "daemon2.out"

        def text1() -> str:
            return out1.read_text(errors="replace")

        with _serve(_FakeLlamaCppHandler) as backend_url:
            # ── First run: setup mode, started with --bind ─────────────────
            with out1.open("wb") as out:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "prometheus.daemon", "--bind", LOOPBACK_V4],
                    stdout=out, stderr=subprocess.STDOUT, cwd=str(tmp_path), env=env)
                try:
                    _wait_for(lambda: _api(api_port, "/api/setup/status", timeout=2)[0] == 200,
                              timeout=30, what="setup mode", proc=proc, out_path=out1)
                    code = re.search(r"^\s{4}(\d{6})\s*$", text1(), re.M)
                    assert code, text1()
                    # 1. loopback only, and a foreign Host is refused.
                    assert_only_loopback(proc, "setup mode", expect={api_port})
                    foreign_host_refused(api_port, "/api/setup/status")
                    assert "all interfaces" not in text1(), "warned about a loopback bind"

                    status, body = _api(api_port, "/api/setup/pair", method="POST",
                                        payload={"code": code.group(1)})
                    assert status == 200, body
                    token = body["token"]
                    status, body = _api(api_port, "/api/setup/configure", method="POST",
                                        token=token, payload={
                                            "provider": "llama_cpp", "base_url": backend_url,
                                            "model": "gemma4-26b"})
                    assert status == 200, body
                    assert body["web"]["bind"] == LOOPBACK_V4

                    # 2. configure pinned it.
                    cfg = yaml.safe_load(
                        (confdir / "prometheus.yaml").read_text(encoding="utf-8"))
                    assert cfg["web"]["bind"] == LOOPBACK_V4
                    assert cfg["web"]["api_port"] == api_port

                    # 3. The same process becomes the real daemon: REST and WS, loopback.
                    status, body = _api(api_port, "/api/setup/complete", method="POST",
                                        token=token)
                    assert status == 200 and body["restarting"] is True
                    _wait_for(
                        lambda: _api(api_port, "/api/status", token=token, timeout=3)[0] == 200,
                        timeout=120, what="the real daemon", proc=proc, out_path=out1)
                    assert_only_loopback(
                        proc, "real daemon, same process", expect={api_port, ws_port})
                    foreign_host_refused(api_port, "/api/status")
                    assert ws_handshake_status(ws_port, "evil.example") == 403
                    assert ws_handshake_status(ws_port, f"localhost:{ws_port}") == 101
                finally:
                    _stop(proc)

            # ── Second run: no flag, no env -- the config alone decides ─────
            with out2.open("wb") as out:
                proc2 = subprocess.Popen(
                    [sys.executable, "-m", "prometheus.daemon"],
                    stdout=out, stderr=subprocess.STDOUT, cwd=str(tmp_path), env=env)
                try:
                    _wait_for(
                        lambda: http_status(api_port, f"localhost:{api_port}") in (200, 401),
                        timeout=120, what="the restarted daemon", proc=proc2, out_path=out2)
                    assert_only_loopback(
                        proc2, "restarted daemon", expect={api_port, ws_port})
                    foreign_host_refused(api_port, "/api/status")
                finally:
                    _stop(proc2)
        assert "all interfaces" not in out2.read_text(errors="replace")


class TestPrecedenceInTheRealDaemon:
    """flag > env (file) > config, in a REAL configured daemon process. The config
    here says 0.0.0.0, so if either higher source were ignored the child would
    listen on every interface (on free ports, closed by the test)."""

    @pytest.mark.parametrize(("how", "source"), [
        ("flag", "--bind"),
        ("env-file", "PROMETHEUS_WEB_BIND"),
        ("environment", "PROMETHEUS_WEB_BIND"),
    ])
    def test_a_higher_source_beats_a_wide_config(self, tmp_path, how, source):
        from tests.test_setup_api_phase2 import _FakeLlamaCppHandler, _serve

        if repo_config_of_loaded_package().exists():
            pytest.skip("a live repo config exists; a child daemon would boot from it")
        try:
            process_listeners()
        except ListenersUnavailable as exc:
            pytest.skip(f"cannot list the child's listening sockets: {exc}")

        api_port, ws_port = free_port(), free_port()
        env = {k: v for k, v in os.environ.items() if not k.startswith("PROMETHEUS_")}
        env.update({
            "PYTHONPATH": str(package_src_root()),
            "PYTHONUNBUFFERED": "1",
            "PROMETHEUS_CONFIG_DIR": str(tmp_path / "confdir"),
            "PROMETHEUS_DATA_DIR": str(tmp_path / "datadir"),
            "PROMETHEUS_ENV_FILE": str(tmp_path / "envfile"),
        })
        argv = [sys.executable, "-m", "prometheus.daemon", "--config", str(tmp_path / "p.yaml")]
        if how == "flag":
            argv += ["--bind", LOOPBACK_V4]
        elif how == "env-file":
            (tmp_path / "envfile").write_text(f"{BIND_ENV}={LOOPBACK_V4}\n", encoding="utf-8")
        else:
            env[BIND_ENV] = LOOPBACK_V4

        log = tmp_path / "daemon.out"
        with _serve(_FakeLlamaCppHandler) as backend_url:
            (tmp_path / "p.yaml").write_text(yaml.safe_dump({
                "model": {"provider": "llama_cpp", "base_url": backend_url,
                          "model": "gemma4-26b"},
                "gateway": {"telegram_enabled": False},
                "web": {"enabled": True, "api_port": api_port, "ws_port": ws_port,
                        "bind": "0.0.0.0"},
            }), encoding="utf-8")
            with log.open("wb") as out:
                proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT,
                                        cwd=str(tmp_path), env=env)
                try:
                    _wait_for(
                        lambda: http_status(api_port, f"localhost:{api_port}") in (200, 401),
                        timeout=120, what="the daemon", proc=proc, out_path=log)
                    listeners = process_listeners(proc.pid)
                    assert {x.port for x in listeners} >= {api_port, ws_port}, listeners
                    wider = [x for x in listeners if x.host != LOOPBACK_V4]
                    assert not wider, f"[{how}] config said 0.0.0.0 and {how} did not win: {wider}"
                    assert http_status(api_port, "evil.example") == 403
                finally:
                    _stop(proc)
        text = log.read_text(errors="replace")
        assert f"from {source}" in text, text[-2000:]
        assert "all interfaces" not in text


class TestStartupWarning:
    """One plain warning, once per start, only when listening on every interface."""

    async def test_wide_launch_warns_exactly_once(self, caplog):
        with caplog.at_level("INFO", logger="prometheus.web.launcher"):
            async with launched({}):
                pass
        hits = [r for r in caplog.records
                if r.levelname == "WARNING" and "all interfaces" in r.getMessage()]
        assert len(hits) == 1, [r.getMessage() for r in caplog.records]
        text = hits[0].getMessage()
        assert "no TLS" in text and "plain HTTP" in text
        assert "0.0.0.0" in text

    async def test_loopback_launch_does_not_warn(self, caplog):
        with caplog.at_level("INFO", logger="prometheus.web.launcher"):
            async with launched({"web": {"bind": LOOPBACK_V4}}):
                pass
        assert not [r for r in caplog.records if "all interfaces" in r.getMessage()]
        started = [r.getMessage() for r in caplog.records if "Mission Control" in r.getMessage()]
        assert started and all(LOOPBACK_V4 in m for m in started), started
