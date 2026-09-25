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


@pytest.mark.asyncio
async def test_a_process_that_leaves_the_group_cannot_hold_the_hook_past_the_grace(
        tmp_path, caplog):
    """The one thing a process-group kill cannot reach: a child that called
    setsid() itself. The executor stops waiting for it after a short grace
    and says so in the one WARNING, instead of blocking the call."""
    import sys

    from prometheus.hooks import executor as ex

    caplog.set_level(logging.WARNING, logger=EXECUTOR_LOGGER)
    hook = CommandHookDefinition(
        command=f"({sys.executable} -c 'import os,time; os.setsid(); time.sleep(6)') & wait",
        timeout_seconds=TIMEOUT)

    elapsed, result = await _run(hook, tmp_path)

    assert elapsed < TIMEOUT + ex._REAP_GRACE_SECONDS + MARGIN, f"returned after {elapsed:.2f}s"
    [one] = result.results
    assert "left its group" in one.reason
    assert len(_warnings(caplog)) == 1
