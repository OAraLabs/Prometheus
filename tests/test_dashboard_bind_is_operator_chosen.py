"""The dashboard tool binds where the OPERATOR says, not where the model does.

THE DEFECT
----------
`tools/builtin/dashboard.py` bound `("0.0.0.0", port)` unconditionally and
served model-authored HTML on every interface of the host — no authentication,
no config gate, on a port the MODEL chose, registered unconditionally in
`__main__.create_tool_registry`. One tool call published arbitrary markup to
the LAN and the tailnet.

Two more defects lived in the same function:

  * `shutdown()` was called without `server_close()`, so the listening socket
    stayed open for the life of the daemon. The port was never released, and a
    later call reusing it hit "Address already in use" against a server this
    tool had itself abandoned.

  * that `shutdown()` ran inside `async def execute`, on the event loop. It
    blocks until the serving thread notices and exits, so a tool call stalled
    the whole daemon — every other session, gateway and timer with it.

THE ASYMMETRY THAT SHAPES THE FIX
----------------------------------
A blanket loopback-only bind is the simple answer and it removes a stated use:
the tool's own description advertises reaching the dashboard over Tailscale,
and an operator who wants that is making a legitimate choice about their own
machine. What is not legitimate is the MODEL making that choice, which is what
an unconditional "0.0.0.0" amounted to.

So: loopback by default, `security.dashboard_tool_bind_host` to widen. The
PORT stays model-chosen — on loopback a port is not an exposure decision, and
capping it would constrain what the agent may attempt rather than where its
data can flow.

A SEPARATE FINDING, MEASURED AND NOT FIXED HERE
------------------------------------------------
This is the SECOND HTML surface in the daemon, and unlike the bundled UI it
serves with NO security headers at all. Measured against the real tool over a
real request:

    dashboard tool (this file)        bundled UI (web/server.py, #425)
    ---------------------------       --------------------------------
    content-security-policy  ABSENT   default-src 'self'; script-src 'self';
                                      style-src 'self'; img-src 'self' data:;
                                      connect-src 'self' ws://…; object-src
                                      'none'; frame-ancestors 'none';
                                      form-action 'none'; base-uri 'none'
    x-content-type-options   ABSENT   nosniff
    x-frame-options          ABSENT   DENY
    referrer-policy          ABSENT   no-referrer

The full response header set is exactly: content-length, content-type, date,
last-modified, server. `SimpleHTTPRequestHandler` adds nothing else, and
nothing in this tool adds to it.

That is deliberately NOT fixed in this change — it is its own piece of work and
would double the diff. `test_the_missing_csp_is_recorded_not_fixed` pins the
measurement so the finding cannot quietly become untrue in either direction:
if someone adds a CSP, that test fails and this docstring gets corrected rather
than left describing a gap that has closed.

Loopback-by-default narrows WHO can reach the markup. It does not make the
markup safe, and the config key's comment says so where an operator widening
it will read it.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.tools.builtin.dashboard import (  # noqa: E402
    DEFAULT_BIND_HOST,
    DashboardInput,
    DashboardTool,
    _advertised_host,
    _stop_server,
)

REPO = Path(__file__).resolve().parents[1]
HTML = "<html><head></head><body><h1>hi</h1></body></html>"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def serve():
    """Start the real tool and always tear the server down afterwards."""
    started: list[tuple[DashboardTool, int]] = []

    def _go(tool: DashboardTool, port: int, html: str = HTML):
        result = asyncio.run(
            tool.execute(DashboardInput(html=html, port=port), None)
        )
        started.append((tool, port))
        return result

    yield _go

    for tool, port in started:
        server = type(tool)._servers.pop(port, None)
        if server is not None:
            _stop_server(server)


# ── The bind ────────────────────────────────────────────────────────────────

def test_the_default_bind_is_loopback(serve):
    """Constructed with no argument, the tool must not listen off-box."""
    assert DEFAULT_BIND_HOST == "127.0.0.1"
    tool = DashboardTool()
    port = _free_port()
    serve(tool, port)

    assert tool._bind_host == "127.0.0.1"
    # Assert on the SOCKET, not on the attribute: the attribute is what we
    # asked for, the socket is what happened.
    server = DashboardTool._servers[port]
    assert server.server_address[0] == "127.0.0.1", (
        f"the tool asked for 127.0.0.1 but bound {server.server_address[0]}"
    )


def test_the_default_bind_refuses_a_non_loopback_connection(serve):
    """The property that actually matters, tested over a real socket.

    A 0.0.0.0 bind accepts a connection addressed to the machine's routable
    address; a loopback bind refuses it. That is the difference the fix is
    about, so it is measured rather than inferred from the bind string.
    """
    routable = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        routable = s.getsockname()[0]
        s.close()
    except OSError:  # pragma: no cover - no route in this environment
        pass
    if not routable or routable.startswith("127."):
        pytest.skip("no non-loopback address available to test against")

    port = _free_port()
    serve(DashboardTool(), port)

    with pytest.raises((ConnectionRefusedError, OSError, TimeoutError)):
        with socket.create_connection((routable, port), timeout=3):
            pass


def test_an_operator_can_widen_the_bind(serve):
    """The stated use must survive the fix.

    Loopback-only with no way back would remove the Tailscale use the tool's
    own description advertises. The point is that the OPERATOR chooses it.
    """
    port = _free_port()
    tool = DashboardTool(bind_host="0.0.0.0")
    serve(tool, port)
    assert DashboardTool._servers[port].server_address[0] == "0.0.0.0"


def test_the_model_cannot_choose_the_bind():
    """Nothing in the tool's input schema selects an interface.

    The port IS model-chosen and that is intended. The interface must not be,
    and the cheapest way for that to regress is a new input field.
    """
    fields = set(DashboardInput.model_fields)
    assert fields == {"html", "port", "title"}, (
        f"the dashboard tool's input schema changed: {sorted(fields)}. If a "
        f"bind/host/interface field was added, the model is choosing the "
        f"exposure again — that is the defect, not a feature."
    )


@pytest.mark.parametrize("bind,expected", [
    ("127.0.0.1", "127.0.0.1"),
    ("localhost", "127.0.0.1"),
    ("::1", "127.0.0.1"),
    ("10.1.2.3", "10.1.2.3"),
])
def test_the_advertised_url_follows_the_bind(bind, expected):
    """A URL is a claim about reachability; it must not be a false one.

    Binding loopback while advertising a tailnet address would send the
    operator to debug the network instead of the config.
    """
    assert _advertised_host(bind) == expected


def test_the_url_handed_back_on_a_loopback_bind_is_loopback(serve):
    port = _free_port()
    result = serve(DashboardTool(), port)
    assert f"http://127.0.0.1:{port}/" in result.output
    assert result.metadata["bind_host"] == "127.0.0.1"


# ── The port is released ────────────────────────────────────────────────────

def test_replacing_a_dashboard_releases_the_previous_port(serve):
    """`shutdown()` without `server_close()` leaks the listening socket.

    Serving twice on one port is the exact sequence that used to fail: the
    first server was shut down but never closed, so the rebind hit "Address
    already in use" against a server this tool had abandoned.
    """
    port = _free_port()
    tool = DashboardTool()
    first = serve(tool, port)
    assert not first.is_error, first.output

    second = asyncio.run(
        tool.execute(DashboardInput(html=HTML, port=port), None)
    )
    assert not second.is_error, (
        f"rebinding the same port failed — the previous socket was not "
        f"released: {second.output}"
    )


def test_the_port_is_free_after_the_server_is_stopped():
    """Stopping must actually release, not merely stop serving."""
    port = _free_port()
    tool = DashboardTool()
    asyncio.run(tool.execute(DashboardInput(html=HTML, port=port), None))
    _stop_server(DashboardTool._servers.pop(port))

    # If server_close() had not run, this bind raises EADDRINUSE.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        probe.bind(("127.0.0.1", port))


# ── The event loop is not blocked ───────────────────────────────────────────

def test_stopping_a_previous_server_does_not_block_the_event_loop():
    """`shutdown()` blocks; it must not do so on the loop.

    A heartbeat coroutine runs alongside the replacement. If `execute` stalls
    the loop, the heartbeat stops ticking for the duration. Measured by count,
    not by timing a single call — a wall-clock threshold would be flaky on a
    loaded machine, whereas "did anything else get to run at all" is the
    property in question.
    """
    port = _free_port()
    tool = DashboardTool()

    async def scenario():
        await tool.execute(DashboardInput(html=HTML, port=port), None)

        ticks = 0
        stop = False

        async def heartbeat():
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.001)

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.01)
        before = ticks
        # The replacement stops the first server — the blocking part.
        await tool.execute(DashboardInput(html=HTML, port=port), None)
        after = ticks
        stop = True
        beat.cancel()
        try:
            await beat
        except asyncio.CancelledError:
            pass
        return before, after

    try:
        before, after = asyncio.run(scenario())
    finally:
        leftover = DashboardTool._servers.pop(port, None)
        if leftover is not None:
            _stop_server(leftover)

    assert after > before, (
        "the event loop made no progress while a dashboard was being "
        f"replaced (heartbeat ticks {before} -> {after}); the blocking "
        "shutdown() is running on the loop"
    )


# ── The finding this PR records but does not fix ────────────────────────────

def test_the_missing_csp_is_recorded_not_fixed(serve):
    """MEASURED, so the finding cannot quietly become untrue either way.

    The served markup is model-authored and carries no security headers. This
    is the second HTML surface in the daemon; the bundled UI got a CSP in #425
    and this one has none.

    If someone adds a CSP here, THIS TEST FAILS — deliberately. The module
    docstring describes a gap, and a closed gap should force that description
    to be corrected rather than left standing as a stale claim.
    """
    port = _free_port()
    serve(DashboardTool(), port)

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
        headers = {k.lower() for k in r.headers.keys()}

    security_headers = {
        "content-security-policy",
        "x-content-type-options",
        "x-frame-options",
        "referrer-policy",
    }
    present = sorted(security_headers & headers)
    assert not present, (
        f"the dashboard now sends security header(s) {present}. That is an "
        f"improvement — update this test and the module docstring, which "
        f"currently record their absence as an open finding."
    )
    assert headers == {
        "server", "date", "content-type", "content-length", "last-modified",
    }, (
        f"the dashboard's response header set changed: {sorted(headers)}. The "
        f"recorded finding was measured against the previous set."
    )
