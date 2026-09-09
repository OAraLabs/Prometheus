"""The tools that walk trees and read files must not stall the event loop.

THE PROPERTY, not the implementation. Each test runs a tool whose underlying
blocking work has been made slow, while a ticker coroutine counts on the loop,
and asserts the ticker KEPT COUNTING. That is the thing that actually matters:
the daemon serves Telegram, Slack, Discord, a WebSocket, a heartbeat and the
REST API from one loop, so a synchronous tree walk freezes all of them at once.

Before this change `grep`, `glob` and `file_read` were `async def` for the tool
interface only — they contained no `await` at all — so their whole bodies ran
on the loop. The ticker would have advanced zero times.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.file_read import FileReadTool, FileReadToolInput
from prometheus.tools.builtin.glob import GlobTool, GlobToolInput
from prometheus.tools.builtin.grep import GrepTool, GrepToolInput

BLOCK_SECONDS = 0.30
TICK = 0.01
#: The loop should manage far more than this many ticks during the block; the
#: bar is deliberately low so a slow CI runner cannot make it flaky, while
#: still being unreachable if the loop is frozen (which yields exactly 0).
MIN_TICKS = 5


async def _run_while_ticking(coro):
    """Await *coro* while a ticker counts on the same loop."""
    ticks = 0
    stop = False

    async def ticker():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(TICK)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)          # let the ticker start
    try:
        result = await coro
    finally:
        stop = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return result, ticks


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("class BaseTool:\n    pass\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    return tmp_path


def _slow(monkeypatch, target, attr):
    """Make one blocking primitive genuinely slow, exactly as a big tree would."""
    import time

    original = getattr(target, attr)

    def slow(*args, **kwargs):
        time.sleep(BLOCK_SECONDS)
        return original(*args, **kwargs)

    monkeypatch.setattr(target, attr, slow)


@pytest.mark.asyncio
async def test_grep_does_not_freeze_the_loop(tree, monkeypatch):
    _slow(monkeypatch, Path, "glob")
    ctx = ToolExecutionContext(cwd=tree)
    result, ticks = await _run_while_ticking(
        GrepTool().execute(GrepToolInput(pattern="BaseTool", file_glob="*.py"), ctx)
    )
    assert not result.is_error, result.output
    assert ticks >= MIN_TICKS, (
        f"the loop advanced only {ticks} time(s) during a "
        f"{BLOCK_SECONDS}s tree walk — grep is blocking it"
    )


@pytest.mark.asyncio
async def test_glob_does_not_freeze_the_loop(tree, monkeypatch):
    _slow(monkeypatch, Path, "glob")
    ctx = ToolExecutionContext(cwd=tree)
    result, ticks = await _run_while_ticking(
        GlobTool().execute(GlobToolInput(pattern="*.py"), ctx)
    )
    assert not result.is_error, result.output
    assert ticks >= MIN_TICKS, f"the loop advanced only {ticks} time(s)"


@pytest.mark.asyncio
async def test_file_read_does_not_freeze_the_loop(tree, monkeypatch):
    _slow(monkeypatch, Path, "read_bytes")
    ctx = ToolExecutionContext(cwd=tree)
    result, ticks = await _run_while_ticking(
        FileReadTool().execute(FileReadToolInput(path=str(tree / "a.py")), ctx)
    )
    assert not result.is_error, result.output
    assert ticks >= MIN_TICKS, f"the loop advanced only {ticks} time(s)"


@pytest.mark.asyncio
async def test_the_ticker_would_have_caught_a_blocking_call(tree):
    """Guard the guard. If the harness cannot detect blocking, the three tests
    above pass for free — so prove it sees a deliberately blocking coroutine."""
    import time

    async def blocks():
        time.sleep(BLOCK_SECONDS)
        return "done"

    result, ticks = await _run_while_ticking(blocks())
    assert result == "done"
    assert ticks < MIN_TICKS, (
        f"the harness counted {ticks} ticks through a synchronous sleep — it "
        f"cannot tell a blocked loop from a free one, so the tests above prove "
        f"nothing"
    )
