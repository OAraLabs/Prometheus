"""``file_watch`` tasks work without watchdog, and watchdog is a base dependency on every platform.

watchdog 6.0.0 shipped no macOS wheel for 3.14, so 0.9.18 marked the base dependency off on exactly that platform
(#872) and a stock Mac installed with no compiler. watchdog 7.0 has those wheels, so the marker is gone and the floor is
7.0. ``tasks/watchers.py`` still imports watchdog only when it watches, and polls the directory with the standard
library when watchdog is absent (an install that left it out). Pinned here with watchdog hidden from the import
system:

* the module imports;
* a file already there is found at once, one that appears later is found while waiting;
* the timeout still returns None, and a missing directory still answers at once;
* with watchdog installed, the event-driven observer is still the one used.
"""

from __future__ import annotations

import asyncio
import os
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


def test_watchdog_is_a_base_dependency_everywhere_from_the_first_release_with_mac_cp314_wheels():
    """watchdog 7.0.0 (2026-10-31) ships macOS wheels for CPython 3.14 (arm64, x86_64, universal2), so the
    platform marker #872 put on it goes, as its comment said it would. The floor is 7.0: an older watchdog would be
    a source build on a Mac with Python 3.14, which is what the marker existed to avoid. The polling fallback stays,
    for an install without watchdog."""
    import tomllib
    from pathlib import Path

    from packaging.requirements import Requirement
    from packaging.version import Version

    deps = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]["dependencies"]
    [req] = [Requirement(d) for d in deps if Requirement(d).name == "watchdog"]
    assert req.marker is None, f"no platform marker: {req}"
    assert not req.specifier.contains(Version("6.0.0")), f"floor below the first release with Mac cp314 wheels: {req}"
    assert req.specifier.contains(Version("7.0.0")), req


@pytest.mark.parametrize("pattern, name", [("*.json", "result.json"), ("result-*", "result-7.txt"), ("done", "done")])
def test_with_watchdog_a_file_that_appears_later_is_found_by_the_observer(tmp_path, pattern, name):
    """The event-driven path, end to end. watchdog 7 matches PatternMatchingEventHandler's patterns against the
    WHOLE path (PurePath.full_match), so `*.json` no longer matched `/dir/result.json` and a file_watch task never
    fired. The pattern is a file name in the watched folder, as the glob pre-check and the polling fallback read it."""
    pytest.importorskip("watchdog")
    import prometheus.tasks.watchers as watchers

    watchers = importlib.reload(watchers)
    assert watchers._observer_class() is not None

    async def scenario():
        async def write_later():
            await asyncio.sleep(0.4)
            (tmp_path / name).write_text("{}")

        writer = asyncio.create_task(write_later())
        got = await watchers.watch_for_file(str(tmp_path), pattern, timeout_seconds=8)
        await writer
        return got

    started = time.monotonic()
    got = asyncio.run(scenario())
    assert got is not None and os.path.basename(got) == name, got
    assert time.monotonic() - started < 6, "found by an event, not at the timeout"
