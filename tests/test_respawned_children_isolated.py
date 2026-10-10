"""The daemon's own ``python -m prometheus`` children never import a module from their working directory.

Two paths start another Prometheus with the daemon's interpreter: a local agent task
(``tasks/manager.py``, ``create_agent_task``) and a managed coding run (``coding/managed.py``,
``build_coding_command``). Each runs in a task's or a repository's directory, and under ``-m`` Python puts the
WORKING DIRECTORY first on ``sys.path``: a ``json.py``, ``secrets.py`` or ``prometheus/`` there was imported in
place of the real module, inside a process holding the daemon's credentials. Prometheus.app's launcher closed
the same hole for the daemon itself (#695, ``-I -B``); these are its children.

The flags are ``-P -s``, not ``-I -B``:

* ``-P`` keeps the working directory (and a script's directory) off ``sys.path``: the hole;
* ``-s`` keeps the user site (``~/.local/lib``, ``~/Library/Python``) out, as the deployed daemon's
  ``PYTHONNOUSERSITE`` already does for itself;
* ``-I`` would also imply ``-E``, which ignores every ``PYTHON*`` variable, and both paths need them:
  ``PYTHONPATH`` carries a source checkout's ``src`` and the sitecustomize that keeps a spawned agent off a
  checkout's live config in tests (tests/test_spawn_key_not_persisted.py), and the app's launcher sets
  ``PYTHONDONTWRITEBYTECODE`` so a child of the signed bundle's interpreter never writes into the bundle. With
  ``-I`` that child would lose both. ``-B`` is not added: the environment already says it where it matters.

Each test reads the argv the code really builds (nothing is spawned for it) and then runs THIS interpreter with
exactly those flags against a shadowing ``./json.py``, beside a control that shows the shadowing without them.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.coding.managed import build_coding_command
from prometheus.tasks.manager import BackgroundTaskManager


class _Captured(Exception):
    pass


async def _agent_task_argv(tmp_path, monkeypatch) -> list[str]:
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    manager = BackgroundTaskManager()
    seen: dict[str, str] = {}

    async def _capture(*, command, **kwargs):
        seen["command"] = command
        raise _Captured

    monkeypatch.setattr(manager, "create_shell_task", _capture)
    with pytest.raises(_Captured):
        await manager.create_agent_task(prompt="p", description="d", cwd=str(tmp_path),
                                        api_key="not-a-key", model="m")
    return shlex.split(seen["command"])


def _coding_argv() -> list[str]:
    return shlex.split(build_coding_command(repo="/r", description="d", acceptance_command="true", task_id="t1"))


def _flags(argv: list[str]) -> list[str]:
    assert argv[0] == sys.executable, argv
    assert argv[argv.index("-m") + 1] == "prometheus", argv
    return argv[1: argv.index("-m")]


def _shadow_imported(directory: Path, flags: list[str]) -> bool:
    marker = directory / "shadow-ran"
    (directory / "json.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    subprocess.run([sys.executable, *flags, "-m", "json.tool"], cwd=directory, env=env, input="{}",
                   capture_output=True, text=True, timeout=60)
    return marker.exists()


def _assert_isolated(flags: list[str], tmp_path: Path) -> None:
    assert {"-P", "-s"} <= set(flags), f"the child is started without -P -s: {flags}"
    control, isolated = tmp_path / "control", tmp_path / "isolated"
    control.mkdir()
    isolated.mkdir()
    assert _shadow_imported(control, []), "the control must show the shadowing, or this proves nothing"
    assert not _shadow_imported(isolated, flags), f"{flags} still imports ./json.py from the working directory"


@pytest.mark.asyncio
async def test_a_local_agent_task_child_is_isolated_from_its_working_directory(tmp_path, monkeypatch):
    argv = await _agent_task_argv(tmp_path, monkeypatch)
    assert argv[argv.index("-m") + 2:] == ["--model", "m"], argv
    _assert_isolated(_flags(argv), tmp_path)


def test_a_managed_coding_child_is_isolated_from_its_working_directory(tmp_path):
    argv = _coding_argv()
    assert argv[argv.index("-m") + 2] == "code", argv
    _assert_isolated(_flags(argv), tmp_path)


@pytest.mark.asyncio
async def test_the_flags_still_let_pythonpath_reach_the_child(tmp_path, monkeypatch):
    """Why not -I: PYTHONPATH (a checkout's src, the tests' sitecustomize) must still reach these children."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "reached_by_pythonpath.py").write_text("", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env["PYTHONPATH"] = str(site)
    for flags in (_flags(await _agent_task_argv(tmp_path, monkeypatch)), _flags(_coding_argv())):
        done = subprocess.run([sys.executable, *flags, "-c", "import reached_by_pythonpath"], env=env,
                              cwd=tmp_path, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (flags, done.stderr[-300:])
