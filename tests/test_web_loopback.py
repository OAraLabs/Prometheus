"""The loopback helpers (web/loopback.py): small, pure, and stable.

Three functions are the public surface another change imports --
``is_loopback_address``, ``is_loopback_host_header`` and ``is_loopback_peer`` --
so their names, signatures and edge cases are pinned here rather than left to
whatever the current implementation happens to do. The ASGI guard built on them
(``LoopbackHostGuard``) is tested at the protocol level: a scope in, the
messages the guard sends out.

Decisions pinned by these tests (each is also documented in loopback.py):

* The Host check is about the NAME, not the port. DNS rebinding needs an
  attacker-controlled name; a port adds nothing against it and would break an
  ``ssh -L`` forward or a local reverse proxy that uses a different port. A
  caller that wants the port enforced passes ``port=``.
* Only the literal names ``localhost``, a loopback IPv4 literal and a bracketed
  loopback IPv6 literal count. ``evil.localhost``, ``localhost.`` and friends do
  not: this is an allow-list of spellings, not a resolver.
* A missing Host header is refused, and so is a request carrying two of them.
* ``is_loopback_peer`` is true only for an IP literal in the ASGI ``client``
  slot. Starlette's TestClient reports the string ``testclient`` there; that is
  not a loopback peer and is not treated as one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from tests.support.listeners import package_src_root


def _lb():
    """Imported inside each test so a missing module fails THAT test, with the
    reason, instead of turning the whole file into one collection error."""
    from prometheus.web import loopback

    return loopback


# ---------------------------------------------------------------------------
# is_loopback_address
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [
    "localhost", "LOCALHOST", "127.0.0.1", "127.0.0.2", "127.255.255.254",
    "::1", "0:0:0:0:0:0:0:1", "::ffff:127.0.0.1",
])
def test_loopback_addresses(value):
    assert _lb().is_loopback_address(value) is True


@pytest.mark.parametrize("value", [
    "0.0.0.0", "::", "192.0.2.1", "10.1.2.3", "128.0.0.1", "2001:db8::1",
    "::ffff:192.0.2.1",
    "", "   ", "example.com", "localhost.", "evil.localhost",
    "127.0.0.1:8005", "[::1]", "127.0.0.1/8", "127.1", "0x7f.0.0.1",
    "localhost@evil.example", "fe80::1%en0",
])
def test_non_loopback_or_unparseable_addresses_are_false(value):
    assert _lb().is_loopback_address(value) is False


# ---------------------------------------------------------------------------
# is_loopback_host_header
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [
    "localhost", "localhost:8005", "LocalHost:8005",
    "127.0.0.1", "127.0.0.1:8005", "127.0.0.2:80",
    "[::1]", "[::1]:8005", "[0:0:0:0:0:0:0:1]:80",
])
def test_loopback_host_headers_pass(value):
    assert _lb().is_loopback_host_header(value) is True


@pytest.mark.parametrize("value", [
    None, "", "   ",
    "evil.example", "evil.example:8005", "localhost.evil.example",
    "evil.localhost", "localhost.", "127.0.0.1.evil.example",
    "localhost@evil.example", "evil.example@localhost",
    "localhost:8005@evil.example",
    "localhost:", "localhost:abc", "localhost:99999", "localhost:8005:9",
    "::1", "[::1", "[::1]x", "[::1]:", "[::1]:abc",
    "0.0.0.0", "0.0.0.0:8005", "[::]", "192.0.2.1", "192.0.2.1:8005",
    "localhost, evil.example", "localhost evil", "localhost\t",
    "http://localhost", "localhost/", "localhost?x", "localhost#x",
    "localhost\r\nHost: evil.example",
    "\uff4c\uff4f\uff43\uff41\uff4c\uff48\uff4f\uff53\uff54",   # fullwidth "localhost"
    "localhost\u212a",                                         # Kelvin sign: lower() is ASCII k
])
def test_foreign_or_malformed_host_headers_fail(value):
    assert _lb().is_loopback_host_header(value) is False


def test_port_is_ignored_by_default():
    assert _lb().is_loopback_host_header("localhost:9000") is True
    assert _lb().is_loopback_host_header("[::1]:1") is True


@pytest.mark.parametrize(("value", "expected"), [
    ("localhost:8005", True),
    ("127.0.0.1:8005", True),
    ("[::1]:8005", True),
    ("localhost", True),             # no port == the scheme's default; not a mismatch
    ("localhost:9000", False),
    ("127.0.0.1:9000", False),
    ("[::1]:9000", False),
    ("evil.example:8005", False),    # the right port does not rescue the wrong name
])
def test_port_is_enforced_when_the_caller_asks(value, expected):
    assert _lb().is_loopback_host_header(value, port=8005) is expected


def test_the_port_parameter_is_keyword_only():
    with pytest.raises(TypeError):
        _lb().is_loopback_host_header("localhost:8005", 8005)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# is_loopback_peer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("client", [
    ("127.0.0.1", 5000), ["127.0.0.1", 5000], ("::1", 5000),
    ("::ffff:127.0.0.1", 5000), ("127.0.0.2", 5000),
])
def test_loopback_peers(client):
    assert _lb().is_loopback_peer({"type": "http", "client": client}) is True


@pytest.mark.parametrize("scope", [
    {"type": "http", "client": ("192.0.2.5", 5000)},
    {"type": "http", "client": ("0.0.0.0", 5000)},
    {"type": "http", "client": ("testclient", 50000)},   # Starlette TestClient
    {"type": "http", "client": ("localhost", 5000)},     # a name is not a peer address
    {"type": "http", "client": None},                    # unix socket: no peer IP
    {"type": "http"},
    {"type": "http", "client": ()},
    {"type": "http", "client": (None, None)},
    {"type": "http", "client": (b"127.0.0.1", 1)},
])
def test_non_loopback_or_unknown_peers_are_false(scope):
    assert _lb().is_loopback_peer(scope) is False


def test_peer_accepts_a_starlette_request_and_rejects_junk():
    pytest.importorskip("starlette")
    from starlette.requests import Request

    lb = _lb()
    assert lb.is_loopback_peer(Request({"type": "http", "client": ("127.0.0.1", 1)})) is True
    assert lb.is_loopback_peer(Request({"type": "http", "client": ("192.0.2.9", 1)})) is False
    for junk in (None, 42, "127.0.0.1", object()):
        assert lb.is_loopback_peer(junk) is False


# ---------------------------------------------------------------------------
# The ASGI guard
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.called = False
        self.scopes: list[dict] = []

    async def __call__(self, scope, receive, send):
        self.called = True
        self.scopes.append(scope)
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})
        elif scope["type"] == "websocket":
            await send({"type": "websocket.accept"})


async def _drive(app, scope):
    sent: list[dict] = []

    async def receive():
        return {"type": "websocket.connect"} if scope["type"] == "websocket" else {
            "type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _http_scope(*hosts, path="/api/status"):
    return {
        "type": "http", "method": "GET", "path": path, "raw_path": path.encode(),
        "query_string": b"", "client": ("127.0.0.1", 50000),
        "headers": [(b"host", h.encode("latin-1")) for h in hosts],
    }


def _ws_scope(*hosts):
    return {
        "type": "websocket", "path": "/", "query_string": b"",
        "client": ("127.0.0.1", 50000),
        "headers": [(b"host", h.encode("latin-1")) for h in hosts],
    }


@pytest.mark.asyncio
async def test_guard_refuses_a_foreign_host_over_http_without_calling_the_app():
    inner = _Recorder()
    sent = await _drive(_lb().LoopbackHostGuard(inner), _http_scope("evil.example"))
    assert inner.called is False
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 403
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert json.loads(body)["error"] == "host_not_allowed"


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["localhost", "localhost:8005", "127.0.0.1:1", "[::1]:8005"])
async def test_guard_passes_a_loopback_host_over_http(host):
    inner = _Recorder()
    sent = await _drive(_lb().LoopbackHostGuard(inner), _http_scope(host))
    assert inner.called is True
    assert sent[0]["status"] == 204


@pytest.mark.asyncio
async def test_guard_refuses_a_request_with_no_host_header():
    inner = _Recorder()
    sent = await _drive(_lb().LoopbackHostGuard(inner), _http_scope())
    assert inner.called is False
    assert sent[0]["status"] == 403


@pytest.mark.asyncio
async def test_guard_refuses_two_host_headers_even_if_one_is_loopback():
    guard = _lb().LoopbackHostGuard
    for hosts in (("localhost", "evil.example"), ("evil.example", "localhost")):
        inner = _Recorder()
        sent = await _drive(guard(inner), _http_scope(*hosts))
        assert inner.called is False, hosts
        assert sent[0]["status"] == 403, hosts


@pytest.mark.asyncio
async def test_guard_matches_the_header_name_case_insensitively():
    inner = _Recorder()
    scope = _http_scope()
    scope["headers"] = [(b"Host", b"evil.example")]
    sent = await _drive(_lb().LoopbackHostGuard(inner), scope)
    assert inner.called is False
    assert sent[0]["status"] == 403


@pytest.mark.asyncio
async def test_guard_closes_a_websocket_handshake_with_a_foreign_host():
    inner = _Recorder()
    sent = await _drive(_lb().LoopbackHostGuard(inner), _ws_scope("evil.example"))
    assert inner.called is False
    assert sent == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.asyncio
async def test_guard_passes_a_websocket_handshake_with_a_loopback_host():
    inner = _Recorder()
    sent = await _drive(_lb().LoopbackHostGuard(inner), _ws_scope("localhost:8005"))
    assert inner.called is True
    assert sent == [{"type": "websocket.accept"}]


@pytest.mark.asyncio
async def test_guard_leaves_lifespan_alone():
    inner = _Recorder()
    sent = await _drive(_lb().LoopbackHostGuard(inner), {"type": "lifespan"})
    assert inner.called is True
    assert sent == []


@pytest.mark.parametrize(("host", "guarded"), [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.2", True),
    ("0.0.0.0", False), ("::", False), ("192.0.2.10", False), ("2001:db8::10", False),
])
def test_guard_if_loopback_only_wraps_loopback_binds(host, guarded):
    lb = _lb()
    inner = _Recorder()
    out = lb.guard_if_loopback(inner, host)
    if guarded:
        assert isinstance(out, lb.LoopbackHostGuard)
    else:
        assert out is inner, (
            "a non-loopback bind must be left unrestricted: tailnet clients "
            "reach it by name, and existing tests send Host: testserver"
        )


# ---------------------------------------------------------------------------
# The modules stay importable without the web stack
# ---------------------------------------------------------------------------

_BLOCKED_IMPORT = textwrap.dedent('''\
    import importlib.abc, sys

    class Absent(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in {"fastapi", "starlette", "uvicorn", "websockets"}:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, Absent())
    for probe in ("fastapi", "starlette", "uvicorn", "websockets"):
        try:
            __import__(probe)
        except ModuleNotFoundError:
            continue
        sys.exit(f"BLOCKER INERT: {probe} imported")

    # The daemon's setup-mode gate imports these before it knows whether the web
    # stack exists; they must be stdlib-only or that gate raises instead of
    # saying "install the web stack".
    import prometheus.web.bind as bind
    import prometheus.web.loopback as loopback
    assert bind.resolve_bind({}).address == "0.0.0.0"
    assert loopback.is_loopback_address("127.0.0.1")
    print("ok")
''')


def test_bind_and_loopback_import_without_the_web_stack(tmp_path):
    script = tmp_path / "blocked_import.py"
    script.write_text(_BLOCKED_IMPORT, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PROMETHEUS_WEB_BIND",)}
    env["PYTHONPATH"] = str(package_src_root())
    proc = subprocess.run(
        [sys.executable, str(script)], env=env, cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stdout + proc.stderr


def test_websocket_handshake_hook_speaks_both_library_generations():
    """websockets 12/13 call process_request(path, headers) and want a tuple;
    14+ call it (connection, request) and want connection.respond(...). The pin
    is >=12, so the hook has to cope with whichever is installed -- and only one
    of the two is installed here, so this drives both by hand."""
    pytest.importorskip("websockets")
    from websockets.datastructures import Headers

    from prometheus.web.ws_server import _loopback_process_request as hook

    good = Headers([("Host", "localhost:8010")])
    bad = Headers([("Host", "evil.example")])
    twice = Headers([("Host", "localhost"), ("Host", "evil.example")])
    none = Headers([])

    # legacy: (path, request_headers) -> None | (status, headers, body)
    assert hook("/", good) is None
    for headers in (bad, twice, none):
        status, _headers, body = hook("/", headers)
        assert int(status) == 403 and body, headers

    # asyncio implementation: (connection, request) -> None | connection.respond(...)
    class Conn:
        def respond(self, status, text):
            return ("response", int(status), text)

    class Req:
        def __init__(self, headers):
            self.headers = headers

    assert hook(Conn(), Req(good)) is None
    for headers in (bad, twice, none):
        kind, status, _text = hook(Conn(), Req(headers))
        assert (kind, status) == ("response", 403), headers


def test_loopback_module_is_documented_and_exports_its_public_names():
    lb = _lb()
    for name in ("is_loopback_address", "is_loopback_host_header", "is_loopback_peer"):
        fn = getattr(lb, name)
        assert fn.__doc__ and len(fn.__doc__.strip()) > 40, f"{name} needs a real docstring"
