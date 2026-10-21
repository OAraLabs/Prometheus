"""Stopping the web bridge the way the daemon does finishes uvicorn's lifespan: no CancelledError traceback, and the
app's shutdown handlers run at shutdown, not at loop teardown.

The daemon stops its listeners by cancelling their task (daemon.py: ``task.cancel()`` over every task, then gather).
Cancelled mid-serve, uvicorn never runs ``Server.shutdown()``, so it never sends ``lifespan.shutdown``: the lifespan
task is left waiting in ``receive()``, ``asyncio.run`` cancels it at teardown, Starlette reports
``lifespan.shutdown.failed`` and uvicorn logs the CancelledError traceback at ERROR on ``uvicorn.error``. The app's
shutdown handlers (the mDNS withdraw, the pairing sweeper) ran only then, after "Prometheus daemon stopped".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

pytest.importorskip("fastapi")

TOKEN = "shutdown-lifespan-test-token"


async def _wait_started(app, attr: str):
    for _ in range(300):
        server = getattr(app.state, attr, None)
        if server is not None and server.started:
            return server
        await asyncio.sleep(0.02)
    raise AssertionError(f"{attr} did not start")


def _orphaned_lifespans() -> list[asyncio.Task]:
    return [t for t in asyncio.all_tasks() if getattr(t.get_coro(), "__qualname__", "") == "LifespanOn.main"]


@pytest.mark.asyncio
async def test_cancelling_the_web_task_like_the_daemon_finishes_the_lifespan(caplog):
    from prometheus.web.server import create_app, start_web

    app = create_app({"web": {"api_token": TOKEN}})
    stopped: list[bool] = []
    app.router.add_event_handler("shutdown", lambda: stopped.append(True))
    task = asyncio.create_task(start_web(app, host="127.0.0.1", port=0))
    server = await _wait_started(app, "http_server")
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    task.cancel()  # what the daemon's shutdown does to every task
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 10)

    assert stopped == [True], "the app's shutdown handlers did not run at shutdown"
    assert server.lifespan.shutdown_event.is_set(), "uvicorn never sent lifespan.shutdown"
    orphans = _orphaned_lifespans()
    assert orphans == [], "a lifespan task outlived the server, for asyncio.run to cancel at teardown"

    # What asyncio.run does at teardown, in case one was left anyway: nothing may log a traceback for it.
    for t in orphans:
        t.cancel()
    await asyncio.gather(*orphans, return_exceptions=True)
    errors = [r for r in caplog.records if r.name == "uvicorn.error" and r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage()[:200] for r in errors]


@pytest.mark.asyncio
async def test_a_stuck_request_cannot_hold_the_shutdown_forever():
    """The graceful wait is bounded, or an open stream would keep the daemon from exiting (FL-1's 15 s)."""
    from prometheus.web.serving import serve_config
    from prometheus.web.server import create_app

    config = serve_config(create_app({"web": {"api_token": TOKEN}}), "127.0.0.1", 0)
    assert config.timeout_graceful_shutdown is not None and 0 < config.timeout_graceful_shutdown <= 10
