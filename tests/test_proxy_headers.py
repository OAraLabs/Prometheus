"""Forwarded headers are believed only from a proxy the owner NAMED.

The pairing routes and hello limit each source, and the "source" is the TCP peer. uvicorn, left at its
defaults, trusts ``X-Forwarded-For`` from any client on 127.0.0.1 and rewrites the request's client address
from it BEFORE the application sees the request. So a process on this machine (or a reverse proxy on it that
passes the caller's own header through) could pick its own source for every limit with one header, and rotate
it to get a fresh budget on each request. The Beacon session found it testing against the real daemon;
a test client never sees it, because Starlette's ``TestClient`` has no such middleware. So these tests serve
the app through ``start_web`` (the daemon's own entry point) on a real socket.

The fix is one decision in one place (``web/serving.py``): forwarded headers are ignored unless
``web.trusted_proxies`` names the proxy, and both servers (the daemon's and setup mode's) build their uvicorn
configuration there, which a test pins so a third place cannot quietly go back to the default.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from prometheus.web import hello as hello_module  # noqa: E402
from prometheus.web.server import start_web  # noqa: E402
from tests.support.pairing_world import GLOBAL, Requester, World  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def serving(app, config=None):
    """The app behind ``start_web`` on a real loopback socket, for the length of the block."""
    port = _free_port()
    stop = threading.Event()
    failure: list[BaseException] = []

    def run() -> None:
        async def main() -> None:
            kwargs = {"config": config} if config is not None else {}
            task = asyncio.ensure_future(start_web(app, host="127.0.0.1", port=port, **kwargs))
            while not stop.is_set() and not task.done():
                await asyncio.sleep(0.02)
            server = getattr(app.state, "http_server", None)
            if server is not None:                      # the daemon's server publishes a handle: stop it cleanly
                server.should_exit = True
                await asyncio.wait_for(task, 10)
            else:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        try:
            asyncio.run(main())
        except BaseException as exc:                    # pragma: no cover - reported below
            failure.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    else:
        raise AssertionError(f"the server never came up: {failure}")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        stop.set()
        thread.join(15)


def _ask(url: str, *, forwarded: str | None = None, source_name: str = "Jennifer's MacBook") -> httpx.Response:
    headers = {"X-Forwarded-For": forwarded} if forwarded else {}
    return httpx.post(f"{url}/api/pair/requests", headers=headers, json={
        "device_name": source_name, "platform": "macos", "public_key": Requester().public_b64})


def _waiting(url: str) -> list[dict]:
    return httpx.get(f"{url}/api/pair/requests", headers={"Authorization": f"Bearer {GLOBAL}"}).json()["requests"]


# ── the daemon's own server, over a real socket ──────────────────────────────

def test_a_forged_forwarded_for_does_not_change_who_a_request_is_from(tmp_path):
    world = World(tmp_path)
    with serving(world.app) as url:
        assert _ask(url, forwarded="203.0.113.7").status_code == 201
        assert [r["source_ip"] for r in _waiting(url)] == ["127.0.0.1"]


def test_rotating_the_forged_header_does_not_get_round_the_pending_limit(tmp_path):
    world = World(tmp_path)
    with serving(world.app) as url:
        assert _ask(url, forwarded="203.0.113.1").status_code == 201
        second = _ask(url, forwarded="203.0.113.2")
        assert second.status_code == 429 and second.json()["reason"] == "per_source_pending"


def test_rotating_the_forged_header_does_not_get_round_hellos_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(hello_module, "HELLO_PER_MINUTE", 3)
    world = World(tmp_path)
    with serving(world.app) as url:
        codes = [httpx.get(f"{url}/api/hello", headers={"X-Forwarded-For": f"203.0.113.{n}"}).status_code
                 for n in range(1, 6)]
    assert codes == [200, 200, 200, 429, 429]


def test_a_named_proxy_is_believed(tmp_path):
    world = World(tmp_path)
    with serving(world.app, {"web": {"trusted_proxies": ["127.0.0.1"]}}) as url:
        assert _ask(url, forwarded="203.0.113.1").status_code == 201
        assert _ask(url, forwarded="203.0.113.2").status_code == 201, "two real sources behind the proxy"
        assert sorted(r["source_ip"] for r in _waiting(url)) == ["203.0.113.1", "203.0.113.2"]


def test_naming_some_other_proxy_does_not_make_loopback_believed(tmp_path):
    world = World(tmp_path)
    with serving(world.app, {"web": {"trusted_proxies": ["10.9.9.9"]}}) as url:
        assert _ask(url, forwarded="203.0.113.7").status_code == 201
        assert [r["source_ip"] for r in _waiting(url)] == ["127.0.0.1"]


# ── the decision, in one place ───────────────────────────────────────────────

def test_nothing_is_trusted_unless_the_owner_names_it():
    from prometheus.web.serving import uvicorn_proxy_options

    for config in (None, {}, {"web": None}, {"web": {}}, {"web": {"trusted_proxies": []}},
                   {"web": {"trusted_proxies": None}}):
        assert uvicorn_proxy_options(config) == {"proxy_headers": False}, config


def test_named_proxies_are_handed_to_uvicorn_as_addresses_and_networks():
    from prometheus.web.serving import uvicorn_proxy_options

    options = uvicorn_proxy_options({"web": {"trusted_proxies": ["10.0.0.5", "192.0.2.0/24", "::1"]}})
    assert options == {"proxy_headers": True, "forwarded_allow_ips": "10.0.0.5,192.0.2.0/24,::1"}


def test_a_single_string_is_read_as_a_comma_separated_list():
    from prometheus.web.serving import trusted_proxies

    assert trusted_proxies({"web": {"trusted_proxies": "10.0.0.5, 10.0.0.6"}}) == ["10.0.0.5", "10.0.0.6"]


@pytest.mark.parametrize("entry", ["*", "0.0.0.0/0", "::/0", "not-an-address", "10.0.0.5/99", "", 7, None])
def test_an_entry_that_would_trust_everyone_or_is_not_an_address_is_dropped_out_loud(entry, caplog):
    from prometheus.web.serving import trusted_proxies

    with caplog.at_level("WARNING"):
        kept = trusted_proxies({"web": {"trusted_proxies": ["10.0.0.5", entry]}})
    assert kept == ["10.0.0.5"], "the good entry survives, the bad one never widens trust"
    assert any("trusted_proxies" in r.getMessage() for r in caplog.records)


def test_the_builder_produces_the_uvicorn_config_the_daemon_serves_with():
    from prometheus.web.serving import serve_config

    app = lambda scope, receive, send: None  # noqa: E731
    default = serve_config(app, "127.0.0.1", 8005)
    assert default.proxy_headers is False and (default.host, default.port) == ("127.0.0.1", 8005)
    assert default.log_config is None, "uvicorn must not install its own log handlers (they bypass redaction)"
    named = serve_config(app, "0.0.0.0", 8005, {"web": {"trusted_proxies": ["10.0.0.5"]}})
    assert named.proxy_headers is True and named.forwarded_allow_ips == "10.0.0.5"


def test_no_server_in_the_web_package_builds_its_own_uvicorn_config():
    """A second place that builds ``uvicorn.Config`` goes back to uvicorn's default, which believes loopback."""
    offenders = []
    for path in sorted((SRC / "web").glob("*.py")):
        if path.name == "serving.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "Config" and getattr(node.func.value, "id", "") == "uvicorn"):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"build it with web.serving.serve_config instead: {offenders}"
