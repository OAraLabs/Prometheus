"""``file_watch`` tasks work without watchdog, which a Mac on Python 3.14+ does not install.

watchdog 6.0.0 ships no macOS wheel for 3.14, so the base dependency is marked off on exactly that platform
(pyproject.toml) and a stock Mac installs with no compiler. ``tasks/watchers.py`` therefore imports watchdog only when
it watches, and polls the directory with the standard library when watchdog is absent. Pinned here with watchdog
hidden from the import system:

* the module imports;
* a file already there is found at once, one that appears later is found while waiting;
* the timeout still returns None, and a missing directory still answers at once;
* with watchdog installed, the event-driven observer is still the one used.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time

import pytest


@pytest.fixture
def without_watchdog(monkeypatch):
    """The watchers module as it loads on a machine with no watchdog."""
    for name in [m for m in sys.modules if m == "watchdog" or m.startswith("watchdog.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "watchdog", None)   # `import watchdog` now raises ImportError
    import prometheus.tasks.watchers as watchers

    reloaded = importlib.reload(watchers)
    yield reloaded
    monkeypatch.undo()
    importlib.reload(watchers)


def test_the_module_imports_without_watchdog(without_watchdog):
    assert callable(without_watchdog.watch_for_file)


def test_a_file_already_there_is_found_at_once(without_watchdog, tmp_path):
    (tmp_path / "done.txt").write_text("x")
    started = time.monotonic()
    got = asyncio.run(without_watchdog.watch_for_file(str(tmp_path), "*.txt", timeout_seconds=5))
    assert got == str(tmp_path / "done.txt")
    assert time.monotonic() - started < 1.0


def test_a_file_that_appears_later_is_found_while_waiting(without_watchdog, tmp_path):
    async def scenario():
        async def write_later():
            await asyncio.sleep(0.4)
            (tmp_path / "ignored.log").write_text("x")
            (tmp_path / "result.json").write_text("{}")

        writer = asyncio.create_task(write_later())
        got = await without_watchdog.watch_for_file(str(tmp_path), "*.json", timeout_seconds=10)
        await writer
        return got

    started = time.monotonic()
    assert asyncio.run(scenario()) == str(tmp_path / "result.json")
    assert time.monotonic() - started < 5, "polling finds it within a few intervals, not at the timeout"


def test_the_timeout_returns_none(without_watchdog, tmp_path):
    started = time.monotonic()
    assert asyncio.run(without_watchdog.watch_for_file(str(tmp_path), "*.json", timeout_seconds=0.6)) is None
    assert 0.5 <= time.monotonic() - started < 3


def test_a_missing_directory_answers_at_once(without_watchdog, tmp_path):
    started = time.monotonic()
    assert asyncio.run(without_watchdog.watch_for_file(str(tmp_path / "nope"), "*", timeout_seconds=10)) is None
    assert time.monotonic() - started < 1.0, "as with watchdog: an unwatchable directory is checked once"


def test_with_watchdog_installed_the_observer_is_used(tmp_path):
    pytest.importorskip("watchdog")
    import prometheus.tasks.watchers as watchers

    watchers = importlib.reload(watchers)
    assert watchers._observer_class() is not None, "watchdog present: the event-driven path, not polling"
