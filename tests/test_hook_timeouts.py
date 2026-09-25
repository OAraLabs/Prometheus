"""Hook timeouts that really stop (WP-X.26).

COMMAND HOOKS
-------------
On timeout the executor killed bash and then waited for it. But waiting for
an asyncio subprocess means waiting for its output pipes to close, and
everything bash had started still held them: the ``sleep`` in
``sleep 4; echo ...``, both halves of a pipeline, a backgrounded job. So a
hook with a 1 s timeout returned after 4 s, and reported that it had timed
out after 1 s. The hook now runs in its own process group, and a timeout
kills the group.

HTTP HOOKS
----------
``httpx.AsyncClient(timeout=N)`` sets per-phase timeouts: connect, each read,
each write. A server that sends a byte every half second never trips a 1 s
read timeout. Measured on origin/main: such a hook with timeout_seconds=1
returned after 4.6 s and reported SUCCESS. timeout_seconds is now a total
deadline.

Each timeout also fails like every other hook failure: a result that honors
block_on_failure and names the hook, and one WARNING.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from prometheus.hooks.events import HookEvent
from prometheus.hooks.executor import HookExecutionContext, HookExecutor
from prometheus.hooks.registry import HookRegistry
from prometheus.hooks.schemas import CommandHookDefinition, HttpHookDefinition

EXECUTOR_LOGGER = "prometheus.hooks.executor"
TIMEOUT = 1
MARGIN = 1.0  # "within the timeout plus a small margin"


async def _run(hook, cwd: Path):
    registry = HookRegistry()
    registry.add(HookEvent.PRE_TOOL_USE, hook)
    executor = HookExecutor(
        registry, HookExecutionContext(cwd=cwd, provider=None, default_model="stub"))
    started = time.monotonic()
    # The test's own bound, so a hook that does not stop fails instead of hanging.
    result = await asyncio.wait_for(
        executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"}), timeout=30)
    return time.monotonic() - started, result


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == EXECUTOR_LOGGER and r.levelno == logging.WARNING]


def _gone(pid: int, within: float = 2.0) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


# ── command hooks ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    'sleep 4; echo "checking $ARGUMENTS"',   # the case measured at 4 s on main
    "sleep 4 | cat",                          # a pipeline
])
@pytest.mark.asyncio
async def test_a_command_hook_returns_within_its_timeout(command, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    hook = CommandHookDefinition(command=command, timeout_seconds=TIMEOUT,
                                 block_on_failure=True)

    elapsed, result = await _run(hook, tmp_path)

    assert elapsed < TIMEOUT + MARGIN, (
        f"a {TIMEOUT}s command hook returned after {elapsed:.2f}s")
    [one] = result.results
    assert one.success is False and one.blocked is True
    assert one.reason == f"pre_tool_use command hook #1 timed out after {TIMEOUT}s"
    [warning] = _warnings(caplog)
    assert "outcome=timeout" in warning


@pytest.mark.asyncio
async def test_a_command_hook_timeout_kills_what_the_hook_started(tmp_path):
    """Returning on time is not enough: the children must be DEAD, not
    orphaned and still running. Here bash exits at once and leaves a
    backgrounded sleep holding its output."""
    pid_file = tmp_path / "child.pid"
    hook = CommandHookDefinition(
        command=f"sleep 30 & echo $! > {pid_file}; echo started",
        timeout_seconds=TIMEOUT)

    elapsed, _ = await _run(hook, tmp_path)

    assert elapsed < TIMEOUT + MARGIN, f"returned after {elapsed:.2f}s"
    pid = int(pid_file.read_text())
    assert _gone(pid), f"the hook's background child {pid} is still running"


@pytest.mark.asyncio
async def test_a_command_hook_that_finishes_in_time_is_unaffected(tmp_path):
    hook = CommandHookDefinition(command='sleep 0.2; echo "done"', timeout_seconds=5)

    elapsed, result = await _run(hook, tmp_path)

    [one] = result.results
    assert one.success is True
    assert one.output == "done"
    assert elapsed < 5


# ── http hooks ──────────────────────────────────────────────────────────────

class _Trickle(BaseHTTPRequestHandler):
    """Sends its 10-byte body one byte every 0.5 s: never idle for a second."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Length", "10")
        self.end_headers()
        try:
            for _ in range(10):
                self.wfile.write(b"x")
                self.wfile.flush()
                time.sleep(0.5)
        except OSError:
            pass  # the client gave up, which is the point

    def log_message(self, *args):
        pass


@pytest.fixture
def trickle_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/hook"
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_an_http_hook_timeout_is_a_total_deadline(trickle_url, tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    hook = HttpHookDefinition(url=trickle_url, timeout_seconds=TIMEOUT,
                              block_on_failure=True)

    elapsed, result = await _run(hook, tmp_path)

    assert elapsed < TIMEOUT + MARGIN, (
        f"a {TIMEOUT}s http hook returned after {elapsed:.2f}s")
    [one] = result.results
    assert one.success is False, "a hook that never answered in time reported success"
    assert one.blocked is True
    assert one.reason == f"pre_tool_use http hook #1 timed out after {TIMEOUT}s"
    [warning] = _warnings(caplog)
    assert "outcome=timeout" in warning


def _escapee(pid_file: Path, seconds: int) -> str:
    """A child that leaves the hook's process group (setsid) and records its pid."""
    code = (f"import os,time; os.setsid(); open({str(pid_file)!r},'w').write(str(os.getpid())); "
            f"time.sleep({seconds})")
    return f"({sys.executable} -c {code!r})"


async def _reap_escapee(pid_file: Path) -> None:
    """Kill the escapee and let its pipes close before the test's loop does,
    so no subprocess transport is finalised on a closed loop."""
    try:
        os.kill(int(pid_file.read_text()), signal.SIGKILL)
    except (OSError, ValueError):
        pass
    await asyncio.sleep(0.3)


@pytest.mark.parametrize("bash_waits", [True, False], ids=["bash-waits", "bash-already-exited"])
@pytest.mark.asyncio
async def test_a_process_that_leaves_the_group_cannot_hold_the_hook_past_the_grace(
        bash_waits, tmp_path, caplog):
    """The one thing a process-group kill cannot reach: a child that called
    setsid() itself. The executor stops waiting for it after a short grace and
    names it in the one WARNING, instead of blocking the call. Whether bash is
    still waiting for it or has already exited, it is what holds the pipes."""
    from prometheus.hooks import executor as ex

    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    pid_file = tmp_path / "escapee.pid"
    tail = "& wait" if bash_waits else "& echo started"
    hook = CommandHookDefinition(command=f"{_escapee(pid_file, 6)} {tail}",
                                 timeout_seconds=TIMEOUT)
    try:
        elapsed, result = await _run(hook, tmp_path)

        assert elapsed < TIMEOUT + ex._REAP_GRACE_SECONDS + MARGIN, f"returned after {elapsed:.2f}s"
        [one] = result.results
        assert "left its group" in one.reason, one.reason
        assert len(_warnings(caplog)) == 1
    finally:
        await _reap_escapee(pid_file)


@pytest.mark.asyncio
async def test_cancelling_a_command_hook_stops_what_it_started(tmp_path):
    """A cancelled turn (or Ctrl-C, or daemon shutdown) cancels the hook. It
    runs in its own session, so no terminal signal reaches it: the executor
    must stop its group itself, or the hook's children outlive the turn."""
    pid_file = tmp_path / "child.pid"
    hook = CommandHookDefinition(command=f"sleep 30 & echo $! > {pid_file}; wait",
                                 timeout_seconds=20)
    registry = HookRegistry()
    registry.add(HookEvent.PRE_TOOL_USE, hook)
    executor = HookExecutor(
        registry, HookExecutionContext(cwd=tmp_path, provider=None, default_model="stub"))

    task = asyncio.create_task(executor.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"}))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid = int(pid_file.read_text())
    assert _gone(pid), f"the cancelled hook's child {pid} is still running"
    await asyncio.sleep(0.3)  # let the killed hook's pipes close on this loop


class _StalledTls:
    """Accepts TCP, reads the ClientHello, never answers. Counts client closes."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.closed_by_client = 0
        self._stop = threading.Event()
        self._conns: list[socket.socket] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            self._conns.append(conn)
            threading.Thread(target=self._hold, args=(conn,), daemon=True).start()

    def _hold(self, conn: socket.socket) -> None:
        conn.settimeout(0.2)
        while not self._stop.is_set():
            try:
                if conn.recv(4096) == b"":
                    self.closed_by_client += 1
                    return
            except TimeoutError:
                continue
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        for c in self._conns:
            c.close()
        self.sock.close()


@pytest.mark.asyncio
async def test_an_https_hook_that_times_out_in_the_handshake_closes_its_socket(tmp_path):
    """The total deadline must not cancel httpcore mid-handshake: start_tls
    closes the stream only on an Exception, so a cancel there leaked one socket
    per timed-out call for as long as the server held the handshake open.
    Every connection this hook opened must be closed by the time it returns."""
    server = _StalledTls()
    try:
        hook = HttpHookDefinition(url=f"https://127.0.0.1:{server.port}/hook",
                                  timeout_seconds=TIMEOUT, block_on_failure=True)
        for _ in range(3):
            elapsed, result = await _run(hook, tmp_path)
            assert elapsed < TIMEOUT + MARGIN, f"returned after {elapsed:.2f}s"
            [one] = result.results
            assert one.success is False and one.blocked is True
        for _ in range(40):
            if server.closed_by_client == 3:
                break
            await asyncio.sleep(0.05)
        assert server.closed_by_client == 3, (
            f"the hook left {3 - server.closed_by_client} of 3 connections open")
    finally:
        server.close()


class _Ok(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


@pytest.mark.asyncio
async def test_an_http_hook_whose_connection_setup_is_slow_but_in_time_succeeds(
        tmp_path, monkeypatch):
    """The deadline is the TOTAL, not a cap on each phase: name resolution
    that takes half the budget must not fail a request that completes inside
    it. (An earlier version of this change capped connection setup at 0.4 of
    the timeout and failed exactly this.)"""
    real_getaddrinfo = socket.getaddrinfo

    def slow_getaddrinfo(*args, **kwargs):
        time.sleep(0.5)
        return real_getaddrinfo(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        hook = HttpHookDefinition(url=f"http://localhost:{server.server_address[1]}/hook",
                                  timeout_seconds=TIMEOUT, block_on_failure=True)
        elapsed, result = await _run(hook, tmp_path)
        [one] = result.results
        assert one.success is True, f"{one.reason} after {elapsed:.2f}s"
        assert one.output == "ok"
    finally:
        server.shutdown()


class _RecordingProxy(BaseHTTPRequestHandler):
    """A forward proxy that answers every request itself and records it."""

    seen: list[str] = []

    def do_POST(self):
        _RecordingProxy.seen.append(self.path)  # absolute URL through a proxy
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Length", "7")
        self.end_headers()
        self.wfile.write(b"proxied")

    def log_message(self, *args):
        pass


@pytest.mark.asyncio
async def test_an_http_hook_still_goes_through_the_environment_proxy(tmp_path, monkeypatch):
    """Making the transport cancel-safe must not change routing: a host whose
    egress is a proxy (HTTP_PROXY, or the system proxy on macOS) still sends
    its hooks through it. (An earlier version of this change passed its own
    transport=, which makes httpx skip environment proxies altogether.)"""
    _RecordingProxy.seen = []
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), _RecordingProxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_address[1]}")
    try:
        hook = HttpHookDefinition(url="http://hooks.example.invalid/hook",
                                  timeout_seconds=3, block_on_failure=True)
        _, result = await _run(hook, tmp_path)
        [one] = result.results
        assert one.success is True, one.reason
        assert one.output == "proxied"
        assert _RecordingProxy.seen == ["http://hooks.example.invalid/hook"]
    finally:
        proxy.shutdown()


# ── what a hook that does NOT time out must still see ───────────────────────

@pytest.mark.asyncio
async def test_the_hooks_own_command_is_not_a_session_leader(tmp_path):
    """The hook's session gets a constant leader; the operator's command runs
    below it, as before. Were the command itself the leader, its setsid()
    would fail, and util-linux `setsid ./gate.sh` would fork and exit 0 at
    once: a failing gate would allow the call."""
    hook = CommandHookDefinition(
        command=f"{sys.executable} -c 'import os; os.setsid(); print(\"ok\")'",
        timeout_seconds=10)

    _, result = await _run(hook, tmp_path)

    [one] = result.results
    assert one.success is True, one.reason
    assert one.output == "ok"


@pytest.mark.asyncio
async def test_a_hooks_exit_status_passes_through(tmp_path):
    hook = CommandHookDefinition(command="exit 3", timeout_seconds=10)

    _, result = await _run(hook, tmp_path)

    [one] = result.results
    assert one.success is False
    assert one.metadata["returncode"] == 3


@pytest.fixture
def quiet_home(tmp_path, monkeypatch):
    """A HOME whose login files the test controls; hooks read HOME at call time."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bash_profile").write_text("export FROM_PROFILE=yes\n")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.mark.parametrize("command, printed", [
    (f"{sys.executable} -c 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)'", ""),
    ("echo before; kill -TERM $$", "before"),
], ids=["simple-command", "compound-command"])
@pytest.mark.asyncio
async def test_a_hook_killed_by_a_signal_reports_only_what_it_printed(
        command, printed, quiet_home, tmp_path):
    """The session's leader is a bash that waits for the command. A bash whose
    child dies from a signal writes a job-status line to its stderr
    ("... Killed: 9  /bin/bash -lc "$1""), which would become the hook's output
    and, under block_on_failure, the reason the model is shown."""
    hook = CommandHookDefinition(command=command, timeout_seconds=10, block_on_failure=True)

    _, result = await _run(hook, tmp_path)

    [one] = result.results
    assert one.blocked is True
    assert one.output == printed
    assert "/bin/bash" not in one.reason, one.reason


@pytest.mark.asyncio
async def test_a_hook_does_not_source_bashrc_when_stdin_is_a_socket(quiet_home, tmp_path):
    """bash sources ~/.bashrc for a non-login `-c` shell whose stdin is a socket
    (its rshd/sshd heuristic). A daemon started with socket stdio (a Node
    parent's pipes are socketpairs) hands its stdin to every hook, so the
    leader must not take that path; the login bash never did."""
    (quiet_home / ".bashrc").write_text("echo BASHRC-SOURCED\n")
    hook = CommandHookDefinition(command="echo hi", timeout_seconds=10)
    ours, theirs = socket.socketpair()
    saved = os.dup(0)
    try:
        os.dup2(ours.fileno(), 0)
        _, result = await _run(hook, tmp_path)
    finally:
        os.dup2(saved, 0)
        os.close(saved)
        ours.close()
        theirs.close()

    [one] = result.results
    assert one.output == "hi"


@pytest.mark.asyncio
async def test_an_allowlisted_bash_env_runs_once_after_the_profile(
        quiet_home, tmp_path, monkeypatch):
    """A non-interactive bash sources $BASH_ENV before its script. Read by the
    leader too, it would run twice, first without anything the operator's
    profile sets (and a `set -e` in it could stop the command running at all)."""
    bash_env = tmp_path / "bash_env.sh"
    bash_env.write_text('echo "sourced, FROM_PROFILE=${FROM_PROFILE-unset}"\n')
    monkeypatch.setenv("BASH_ENV", str(bash_env))
    hook = CommandHookDefinition(command='echo "BASH_ENV=$BASH_ENV"', timeout_seconds=10,
                                 env_allowlist=["BASH_ENV"])

    _, result = await _run(hook, tmp_path)

    [one] = result.results
    assert one.output.splitlines() == ["sourced, FROM_PROFILE=yes", f"BASH_ENV={bash_env}"]


@pytest.mark.asyncio
async def test_an_http_hook_still_runs_if_httpx_moves_its_pool(tmp_path, monkeypatch, caplog):
    """The cancel-safe wrapping reaches into httpx's private pool. If a future
    httpx moves it, the hook must degrade (run, and say the wrapping is off),
    not fail every call."""
    import httpx

    class _PoolMoved(httpx.AsyncHTTPTransport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._elsewhere = self._pool
            del self._pool

        async def handle_async_request(self, request):
            self._pool = self._elsewhere
            try:
                return await super().handle_async_request(request)
            finally:
                del self._pool

        async def __aenter__(self):
            await self._elsewhere.__aenter__()
            return self

        async def __aexit__(self, *exc):
            await self._elsewhere.__aexit__(*exc)

        async def aclose(self):
            await self._elsewhere.aclose()

    monkeypatch.setattr(httpx._client, "AsyncHTTPTransport", _PoolMoved)
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        hook = HttpHookDefinition(url=f"http://127.0.0.1:{server.server_address[1]}/hook",
                                  timeout_seconds=5, block_on_failure=True)
        _, result = await _run(hook, tmp_path)
        [one] = result.results
        assert one.success is True, one.reason
        assert any("no pool backend to wrap" in m for m in _warnings(caplog))
    finally:
        server.shutdown()
