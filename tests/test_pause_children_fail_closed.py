"""A child that cannot read the pause state treats it as PAUSED, never as not paused (Will, 2026-11-12: a PR 2
must-fix; the fail-closed rule he approved).

Workers and coding runs are separate processes; they have no in-memory pause state and read the file. The daemon
now tells every child it builds the EXACT path (``PROMETHEUS_PAUSE_FILE``, in ``tasks/child_env.task_child_env``),
so a child never guesses a data folder. In a child that was told the path:

  * the folder cannot be reached (missing, unreadable, not mounted) → paused;
  * the folder is readable and the file absent → not paused (never paused here);
  * the file is there → what it says (an unreadable file is paused, as before).

The real-environment test spawns an actual child (``sys.executable``) with the environment the daemon builds for a
worker or a coding run, and asks it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.engine import pause

PROBE = ("import json, sys\n"
         "from prometheus.engine import pause\n"
         "info = pause.current()\n"
         "print(json.dumps({'paused': info is not None, 'by': info.by if info else None}))\n")


@pytest.fixture(autouse=True)
def _uninstalled():
    pause.install(None)
    yield
    pause.install(None)


def _child(env: dict) -> dict:
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=60,
                         cwd=str(Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def _daemon_env(tmp_path, monkeypatch, path: Path) -> dict:
    """The environment the daemon builds for a worker or a coding run, with this pause file installed."""
    from prometheus.tasks.child_env import task_child_env

    state = pause.PauseState(path)
    pause.install(state)
    env = task_child_env(None, config_path=None)
    pause.install(None)
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") or str(Path(__file__).resolve().parent.parent / "src")
    return env


def test_the_daemon_tells_its_children_the_exact_pause_file(tmp_path):
    from prometheus.tasks.child_env import task_child_env

    path = tmp_path / "data" / "pause.json"
    pause.install(pause.PauseState(path))
    try:
        assert task_child_env(None)["PROMETHEUS_PAUSE_FILE"] == str(path)
    finally:
        pause.install(None)


def test_a_real_child_sees_a_pause(tmp_path, monkeypatch):
    path = tmp_path / "data" / "pause.json"
    pause.PauseState(path).pause(by="owner device", reason="test")
    assert _child(_daemon_env(tmp_path, monkeypatch, path))["paused"] is True


def test_a_real_child_that_cannot_reach_the_folder_is_paused(tmp_path, monkeypatch):
    path = tmp_path / "gone" / "data" / "pause.json"                 # the folder does not exist for the child
    env = _daemon_env(tmp_path, monkeypatch, path)
    seen = _child(env)
    assert seen["paused"] is True and seen["by"] == "unreadable"


def test_a_real_child_with_an_unreadable_folder_is_paused(tmp_path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root reads any folder")
    folder = tmp_path / "data"
    folder.mkdir()
    path = folder / "pause.json"
    env = _daemon_env(tmp_path, monkeypatch, path)
    os.chmod(folder, 0)
    try:
        assert _child(env)["paused"] is True
    finally:
        os.chmod(folder, 0o700)


def test_a_real_child_with_a_readable_folder_and_no_file_is_not_paused(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    assert _child(_daemon_env(tmp_path, monkeypatch, folder / "pause.json"))["paused"] is False


def test_a_process_that_was_not_told_keeps_the_default_path(tmp_path, monkeypatch):
    """A CLI run (no daemon parent) reads the default data folder as before; a missing folder there is not a
    pause (nothing was ever paused on a fresh machine)."""
    monkeypatch.delenv("PROMETHEUS_PAUSE_FILE", raising=False)
    monkeypatch.setattr(pause, "pause_path", lambda: tmp_path / "nothing" / "pause.json")
    assert pause.current() is None


# ── chat-gateway turns run as tasks a pause can cancel ──────────────────────

def test_a_gateway_turn_is_cancelled_by_a_pause_and_its_poller_survives(tmp_path, monkeypatch):
    """Telegram, Slack and Discord run turns through AgentLoop.run_async, some inline in a long-lived poller task.
    run_async now runs the turn as a task of its own: a pause cancels the turn (the caller hears Paused) and never
    the poller."""
    import asyncio
    from types import SimpleNamespace

    from prometheus.engine.agent_loop import AgentLoop
    from prometheus.web import pause_routes

    loop = AgentLoop.__new__(AgentLoop)

    async def slow_body(*a, **kw):
        await asyncio.sleep(60)

    monkeypatch.setattr(loop, "_run_async_body", slow_body)
    pause.install(pause.PauseState(tmp_path / "pause.json"))

    async def go():
        heard: list[str] = []

        async def poller():
            try:
                await loop.run_async("system", "hi")
            except pause.Paused:
                heard.append("paused")
            await asyncio.sleep(0)
            return "poller alive"

        p = asyncio.create_task(poller())
        await asyncio.sleep(0.01)
        result = await pause_routes.pause_everything(SimpleNamespace(state=SimpleNamespace()), by="test", reason="")
        out = await asyncio.wait_for(p, 5)
        return heard, out, result

    heard, out, result = asyncio.run(go())
    assert heard == ["paused"] and out == "poller alive" and result["stopped"]["turns"] == 1
