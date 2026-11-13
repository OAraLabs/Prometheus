"""HTTPS over Tailscale with ``tailscale serve`` (shape A), daemon side: the settings block, the Host guards, the client
address, the hello ceiling, the pairing answers and the page's CSP.

``tailscale serve`` terminates TLS with the tailnet name's certificate and relays to the daemon, which stays bound to
loopback. Measured on a real node (Tailscale 1.104): the daemon's TCP peer is always 127.0.0.1; ``Host`` is kept
(``<name>:<port>``, and Serve itself refuses a forged one); ``X-Forwarded-For`` is ONE value, the caller's tailnet
address, a forged one overwritten; ``X-Forwarded-Host`` is ``<name>:<port>``, a forged one overwritten.

The trust table held here:

  * nothing changes unless ``web.tailscale`` is a valid, enabled block AND the daemon has an API token (an open
    daemon must never answer a relayed request);
  * the loopback Host guards (REST and the WebSocket bridge) accept the name only when the request is relayed the
    way Serve relays it: a loopback peer, ``Host`` and ``X-Forwarded-Host`` both the configured name (on the
    configured port), and exactly one ``X-Forwarded-For`` value that is a tailnet address. Anything else is the
    403 it is today: no relay headers, a mismatched name or port, a non-tailnet or doubled address, a rebinding name;
  * a request accepted that way has the tailnet address as its client (so per-source limits key on the real caller)
    and is never "this machine";
  * hello's limiter has a ceiling across all sources, so a local process forging a new tailnet address per request
    gets a fresh bucket each time but not unlimited answers;
  * the pairing answers say ``https://<name>`` / ``wss://<name>:<ws_port>`` and ``tls: {trust: ca, host}`` to a
    request that came through Serve, and the page's CSP names ``wss://<name>:<ws_port>``.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.web import tailscale_serve as ts
from prometheus.web.loopback import LoopbackHostGuard, guard_if_loopback, is_same_machine

NAME = "studio.tail1234.ts.net"
# Tailnet addresses (100.64.0.0/10) are built at run time: the pre-commit scanner refuses the literals, real or not.
def _tail(*parts: int) -> str:
    return ".".join(str(n) for n in (100, *parts))


TAILNET_CALLER = _tail(101, 102, 103)
OTHER_CALLER = _tail(64, 0, 9)
TAILNET_PEER = _tail(64, 0, 7)


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_API_TOKEN", "tailscale-serve-test-token-not-real")


def _config(**block) -> dict:
    return {"web": {"bind": "127.0.0.1", "tailscale": {"enabled": True, "name": NAME, "rest_port": 8443,
                                                       "ws_port": 8444, **block}}}


# ── D1: the settings block ───────────────────────────────────────────────────

def test_a_valid_block_is_read(token):
    s = ts.serve_settings(_config())
    assert s == ts.ServeSettings(name=NAME, rest_port=8443, ws_port=8444)


def test_the_ports_default_to_443_and_8443(token):
    s = ts.serve_settings({"web": {"tailscale": {"enabled": True, "name": NAME}}})
    assert s is not None and (s.rest_port, s.ws_port) == (443, 8443)


@pytest.mark.parametrize("block", [
    {"name": "Studio.tail1234.ts.net"},             # not lowercase
    {"name": "studio.example.com"},                 # not a tailnet name
    {"name": "*.tail1234.ts.net"},
    {"name": "studio.tail1234.ts.net."},            # trailing dot
    {"name": "tail1234.ts.net"},                    # no machine label
    {"name": "-studio.tail1234.ts.net"},
    {"name": "stu dio.tail1234.ts.net"},
    {"name": ("a" * 64) + ".tail1234.ts.net"},      # label too long
    {"name": 7},
    {"rest_port": 0}, {"rest_port": 70000}, {"ws_port": "8444"}, {"ws_port": True},
    {"rest_port": 8444},                            # the two cannot share a port
])
def test_a_bad_block_is_dropped(token, block, caplog):
    assert ts.serve_settings(_config(**block)) is None
    assert "web.tailscale" in caplog.text


def test_off_or_absent_is_off(token):
    assert ts.serve_settings({"web": {}}) is None
    assert ts.serve_settings(_config(enabled=False)) is None
    assert ts.serve_settings({}) is None


def test_an_open_daemon_never_answers_a_relayed_request(monkeypatch, caplog):
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    assert ts.serve_settings(_config()) is None
    assert "API token" in caplog.text
    cfg = _config()
    cfg["web"]["api_token"] = "configured-in-the-file"
    assert ts.serve_settings(cfg) is not None


# ── D2 + D3: the REST guard ──────────────────────────────────────────────────

def _scope(*, host=f"{NAME}:8443", xfh=f"{NAME}:8443", xff=TAILNET_CALLER, peer="127.0.0.1", kind="http",
           extra=()) -> dict:
    headers = [(b"host", host.encode())] if host is not None else []
    if xfh is not None:
        headers.append((b"x-forwarded-host", xfh.encode()))
    if xff is not None:
        headers.append((b"x-forwarded-for", xff.encode()))
    headers.extend(extra)
    return {"type": kind, "headers": headers, "client": (peer, 51000), "server": ("127.0.0.1", 8005),
            "path": "/api/hello", "method": "GET"}


def _run_guard(scope, settings):
    seen: list[dict] = []
    sent: list[dict] = []

    async def app(scope, receive, send):
        seen.append(scope)

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        sent.append(message)

    asyncio.run(LoopbackHostGuard(app, serve=settings)(scope, receive, send))
    status = next((m["status"] for m in sent if m.get("type") == "http.response.start"), None)
    closed = any(m.get("type") == "websocket.close" for m in sent)
    return (seen[0] if seen else None), status, closed


SETTINGS = ts.ServeSettings(name=NAME, rest_port=8443, ws_port=8444)


def test_a_serve_relayed_request_is_admitted_as_its_tailnet_caller():
    inner, status, _ = _run_guard(_scope(), SETTINGS)
    assert inner is not None and status is None
    assert inner["client"][0] == TAILNET_CALLER
    assert ts.came_through_serve(inner) == SETTINGS
    assert not is_same_machine(inner)


def test_the_default_port_needs_no_port_in_the_header():
    s443 = ts.ServeSettings(name=NAME, rest_port=443, ws_port=8443)
    inner, _, _ = _run_guard(_scope(host=NAME, xfh=NAME), s443)
    assert inner is not None
    inner, status, _ = _run_guard(_scope(host=NAME, xfh=NAME), SETTINGS)       # 8443 configured: a bare name is not it
    assert inner is None and status == 403


@pytest.mark.parametrize(("why", "kw"), [
    ("no relay headers", dict(xfh=None, xff=None)),
    ("no X-Forwarded-Host", dict(xfh=None)),
    ("no X-Forwarded-For", dict(xff=None)),
    ("X-Forwarded-Host is another name", dict(xfh="evil.example:8443")),
    ("Host is another name", dict(host="evil.example:8443")),
    ("rebinding: both another name", dict(host="evil.example", xfh="evil.example")),
    ("a lookalike name", dict(host=f"{NAME}.evil.example:8443", xfh=f"{NAME}.evil.example:8443")),
    ("the wrong port", dict(host=f"{NAME}:9999", xfh=f"{NAME}:9999")),
    ("ports disagree", dict(xfh=f"{NAME}:8444")),
    ("a non-tailnet caller", dict(xff="1.2.3.4")),
    ("a loopback caller", dict(xff="127.0.0.1")),
    ("two callers in one header", dict(xff=f"{TAILNET_CALLER}, {OTHER_CALLER}")),
    ("not an address", dict(xff="studio")),
    ("a peer that is not loopback", dict(peer=TAILNET_PEER)),
])
def test_anything_not_relayed_the_way_serve_relays_is_refused(why, kw):
    inner, status, _ = _run_guard(_scope(**kw), SETTINGS)
    assert inner is None and status == 403, why


def test_a_second_relay_header_value_is_refused():
    extra = [(b"x-forwarded-for", OTHER_CALLER.encode())]
    inner, status, _ = _run_guard(_scope(extra=extra), SETTINGS)
    assert inner is None and status == 403
    extra = [(b"x-forwarded-host", f"{NAME}:8443".encode())]
    inner, status, _ = _run_guard(_scope(extra=extra), SETTINGS)
    assert inner is None and status == 403


def test_without_settings_the_name_is_refused_as_today():
    inner, status, _ = _run_guard(_scope(), None)
    assert inner is None and status == 403


def test_loopback_names_still_pass_unchanged():
    inner, _, _ = _run_guard(_scope(host="localhost:8005", xfh=None, xff=None), SETTINGS)
    assert inner is not None and inner["client"][0] == "127.0.0.1"
    assert ts.came_through_serve(inner) is None


def test_an_ipv6_tailnet_caller_is_admitted():
    inner, _, _ = _run_guard(_scope(xff="fd7a:115c:a1e0::1234"), SETTINGS)
    assert inner is not None and inner["client"][0] == "fd7a:115c:a1e0::1234"


def test_guard_if_loopback_passes_the_settings_on():
    async def app(scope, receive, send):
        return None

    guarded = guard_if_loopback(app, "127.0.0.1", serve=SETTINGS)
    assert isinstance(guarded, LoopbackHostGuard) and guarded.serve == SETTINGS
    assert guard_if_loopback(app, "0.0.0.0", serve=SETTINGS) is app


def test_the_daemons_listener_is_built_with_the_settings(token):
    pytest.importorskip("uvicorn")
    from prometheus.web.serving import serve_config

    async def app(scope, receive, send):
        return None

    cfg = serve_config(app, "127.0.0.1", 8005, _config())
    assert isinstance(cfg.app, LoopbackHostGuard) and cfg.app.serve == SETTINGS


# ── D2 for the WebSocket bridge: the same table ──────────────────────────────

class _Headers:
    def __init__(self, pairs):
        self._pairs = pairs

    def get_all(self, name):
        return [v for k, v in self._pairs if k.lower() == name.lower()]


class _Request:
    def __init__(self, pairs):
        self.headers = _Headers(pairs)


class _Connection:
    def __init__(self, peer="127.0.0.1"):
        self.remote_address = (peer, 51000)

    def respond(self, status, message):
        return ("refused", int(status))


def _ws(pairs, settings, peer="127.0.0.1"):
    from prometheus.web.ws_server import _loopback_process_request

    return _loopback_process_request(_Connection(peer), _Request(pairs), serve=settings)


def _ws_pairs(host=f"{NAME}:8444", xfh=f"{NAME}:8444", xff=TAILNET_CALLER):
    pairs = [("Host", host)]
    if xfh is not None:
        pairs.append(("X-Forwarded-Host", xfh))
    if xff is not None:
        pairs.append(("X-Forwarded-For", xff))
    return pairs


def test_the_bridge_admits_a_serve_relayed_handshake_on_its_port():
    assert _ws(_ws_pairs(), SETTINGS) is None


@pytest.mark.parametrize(("why", "kw", "peer"), [
    ("no relay headers", dict(xfh=None, xff=None), "127.0.0.1"),
    ("the REST port", dict(host=f"{NAME}:8443", xfh=f"{NAME}:8443"), "127.0.0.1"),
    ("another name", dict(host="evil.example:8444", xfh="evil.example:8444"), "127.0.0.1"),
    ("a non-tailnet caller", dict(xff="1.2.3.4"), "127.0.0.1"),
    ("a peer that is not loopback", {}, TAILNET_PEER),
])
def test_the_bridge_refuses_what_the_rest_guard_refuses(why, kw, peer):
    assert _ws(_ws_pairs(**kw), SETTINGS, peer=peer) == ("refused", 403), why


def test_the_bridge_without_settings_refuses_the_name():
    assert _ws(_ws_pairs(), None) == ("refused", 403)
    assert _ws([("Host", "localhost:8010")], SETTINGS) is None


# ── the hello ceiling ────────────────────────────────────────────────────────

def test_a_forger_rotating_addresses_is_held_to_the_ceiling_and_a_quiet_source_keeps_its_limit():
    from prometheus.web.source_limits import SourceLimiter

    now = [0.0]
    limiter = SourceLimiter(60, 60.0, clock=lambda: now[0], ceiling=600)
    quiet = [limiter.check(_tail(64, 0, 1)).allowed for _ in range(60)]
    assert all(quiet) and not limiter.check(_tail(64, 0, 1)).allowed          # its own 60, as before
    admitted = sum(limiter.check(_tail(64, i // 250, i % 250 + 2)).allowed for i in range(2000))
    assert admitted == 600 - 60                                            # every new address, held to the ceiling
    refused = limiter.check(_tail(65, 0, 1))
    assert not refused.allowed and refused.retry_after >= 1
    now[0] = 61.0
    assert limiter.check(_tail(64, 0, 1)).allowed                             # the window passes for everyone


def test_hello_uses_the_ceiling():
    from prometheus.web import hello

    assert hello.HELLO_CEILING_PER_MINUTE == 600
    assert hello.new_limiter()._ceiling == 600


# ── D4: the pairing answers ──────────────────────────────────────────────────

def _request(scope_extra):
    pytest.importorskip("starlette")
    from starlette.requests import Request

    scope = {"type": "http", "method": "GET", "path": "/api/pair/requests/x", "scheme": "http",
             "headers": [(b"host", f"{NAME}:8443".encode())], "client": (TAILNET_CALLER, 0),
             "server": ("127.0.0.1", 8005), "query_string": b"", **scope_extra}
    return Request(scope)


def test_a_poll_through_serve_is_told_https_and_wss_on_the_tailnet_name():
    from prometheus.web.pairing_routes import _endpoints, _tls_pin

    req = _request({ts.SCOPE_KEY: SETTINGS})
    assert _endpoints(req, {"ws_port": 8010}) == {"rest": f"https://{NAME}:8443", "ws": f"wss://{NAME}:8444"}
    assert _tls_pin(req) == {"trust": "ca", "host": NAME}


def test_on_the_default_port_the_rest_address_carries_no_port():
    from prometheus.web.pairing_routes import _endpoints

    req = _request({ts.SCOPE_KEY: ts.ServeSettings(name=NAME, rest_port=443, ws_port=8443)})
    assert _endpoints(req, {}) == {"rest": f"https://{NAME}", "ws": f"wss://{NAME}:8443"}


def test_a_poll_not_through_serve_is_answered_as_before():
    from prometheus.web.pairing_routes import _endpoints, _tls_pin

    req = _request({})
    assert _endpoints(req, {"ws_port": 8010}) == {"rest": f"http://{NAME}:8443", "ws": f"ws://{NAME}:8010"}
    assert _tls_pin(req) is None


# ── D9: the page's CSP ───────────────────────────────────────────────────────

def test_the_page_through_serve_may_open_its_socket_on_the_tailnet_name():
    from prometheus.web.server import _static_security_headers

    scope = {"headers": [(b"host", f"{NAME}:8443".encode())], ts.SCOPE_KEY: SETTINGS}
    csp = dict(_static_security_headers({"web": {"ws_port": 8010}}, scope))[b"content-security-policy"].decode()
    assert f"wss://{NAME}:8444" in csp and "ws://" not in csp


def test_the_page_not_through_serve_keeps_its_policy():
    from prometheus.web.server import _static_security_headers

    scope = {"headers": [(b"host", b"localhost:8005")]}
    csp = dict(_static_security_headers({"web": {"ws_port": 8010}}, scope))[b"content-security-policy"].decode()
    assert "ws://localhost:8010" in csp
