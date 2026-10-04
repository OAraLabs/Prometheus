"""A cron job's `python3` is the daemon's own interpreter; a task's is not.

THE DEFECT (found 2026-10-03). Since the venv deploys (``scripts/deploy.sh``),
the daemon's systemd drop-in sets ``PYTHONNOUSERSITE=1`` so the daemon reads
nothing from ``~/.local``. Every shell a daemon-run job gets inherits it.
A job that runs bare ``python3`` gets the SYSTEM interpreter, whose packages
were in the user site the flag switches off. ``daily_news_briefing_pm``
(``python3 -m prometheus.jobs.daily_briefing``) then died on
``No module named 'pydantic'`` every night from 2026-09-24 to 2026-10-03.

THE FIX. The cron scheduler runs every job command with ``python3`` and
``python`` resolving to the daemon's own interpreter (``sys.executable``: the
deployed venv), through wrapper scripts on PATH that the job's login shell
cannot shadow. An absolute interpreter path in a command is the job author's
choice and is left alone.

CRON ONLY (Will, 2026-10-04). Background tasks can be started by the model,
and model-run code must never get the daemon's own venv interpreter (a
``python3 -m pip install`` would target the production venv), so a task's
``python3`` is exactly what a plain shell finds.

These tests drive real ``/bin/bash -lc`` shells, the real ``execute_job`` and
the real ``BackgroundTaskManager``, and read what each actually ran.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.utils import job_python

PROBE = "import sys; print(sys.executable)"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    # A login shell sources $HOME/.profile; keep the host's out of it.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # The conditions the daemon runs under: the user site off, and a PATH
    # with no venv on it (the unit sets PATH itself), so bare python3 is the
    # system interpreter. Under `uv run` the test's own PATH starts with the
    # dev venv, which would hide the defect.
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")


def _fake_interpreter(tmp_path: Path, name: str, marker: str) -> Path:
    path = tmp_path / name
    path.write_text(f'#!/bin/sh\necho "{marker} $*"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _bash(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/bash", "-lc", command], capture_output=True, text=True,
                          timeout=30, env=dict(os.environ))


# --------------------------------------------------------------------------- #
# The shell command a job runs
# --------------------------------------------------------------------------- #


class TestJobShellCommand:

    @pytest.mark.parametrize("command", [
        "python3 -V",
        "python -V",
        "cd / && python3 -V",
        "timeout 5 python3 -V",
        "env python3 -V",
        "true; python3 -V | cat",
    ])
    def test_python3_anywhere_in_a_job_is_the_daemon_interpreter(self, tmp_path, monkeypatch,
                                                                 command):
        fake = _fake_interpreter(tmp_path, "daemon-python", "DAEMON-INTERPRETER")
        monkeypatch.setattr(sys, "executable", str(fake))
        out = _bash(job_python.job_shell_command(command))
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "DAEMON-INTERPRETER -V"

    def test_a_login_profile_cannot_shadow_it(self, tmp_path, monkeypatch):
        """bash -l sources ~/.profile, which prepends to PATH on the mini."""
        fake = _fake_interpreter(tmp_path, "daemon-python", "DAEMON-INTERPRETER")
        monkeypatch.setattr(sys, "executable", str(fake))
        evil = Path(os.environ["HOME"]) / "bin"
        evil.mkdir()
        _fake_interpreter(evil, "python3", "PROFILE-PYTHON")
        (Path(os.environ["HOME"]) / ".profile").write_text('PATH="$HOME/bin:$PATH"\n')
        out = _bash(job_python.job_shell_command("python3 -V"))
        assert out.stdout.strip() == "DAEMON-INTERPRETER -V"

    def test_an_absolute_interpreter_path_is_the_authors_choice(self, tmp_path, monkeypatch):
        fake = _fake_interpreter(tmp_path, "daemon-python", "DAEMON-INTERPRETER")
        monkeypatch.setattr(sys, "executable", str(fake))
        other = _fake_interpreter(tmp_path, "other-python", "OTHER-PYTHON")
        out = _bash(job_python.job_shell_command(f"{other} -V"))
        assert out.stdout.strip() == "OTHER-PYTHON -V"

    def test_the_exit_code_and_arguments_pass_through(self):
        out = _bash(job_python.job_shell_command(
            "python3 -c 'import sys; print(sys.argv[1:]); sys.exit(3)' 'a b' c"))
        assert out.returncode == 3
        assert out.stdout.strip() == "['a b', 'c']"

    def test_the_shim_follows_the_interpreter(self, tmp_path, monkeypatch):
        """After a deploy the daemon restarts on a new venv; the wrappers move with it."""
        first = _fake_interpreter(tmp_path, "venv-a", "VENV-A")
        second = _fake_interpreter(tmp_path, "venv-b", "VENV-B")
        monkeypatch.setattr(sys, "executable", str(first))
        assert _bash(job_python.job_shell_command("python3")).stdout.strip() == "VENV-A"
        monkeypatch.setattr(sys, "executable", str(second))
        assert _bash(job_python.job_shell_command("python3")).stdout.strip() == "VENV-B"

    def test_the_wrappers_live_in_the_data_dir(self):
        shims = job_python.ensure_python_shims()
        assert shims.parent == Path(os.environ["PROMETHEUS_DATA_DIR"])
        assert sorted(p.name for p in shims.iterdir()) == ["python", "python3"]

    def test_if_the_wrappers_cannot_be_written_the_job_still_runs(self, monkeypatch, caplog):
        """A job is never refused over the shim; the miss is a WARNING."""
        def boom():
            raise OSError("read-only data dir")

        monkeypatch.setattr(job_python, "ensure_python_shims", boom)
        assert job_python.job_shell_command("echo hi") == "echo hi"
        assert "python3" in caplog.text


# --------------------------------------------------------------------------- #
# The two daemon paths that run job commands
# --------------------------------------------------------------------------- #


class TestCronJob:

    async def test_a_python3_cron_job_imports_what_the_daemon_has(self, tmp_path, monkeypatch):
        """The briefing's failure, reproduced: PYTHONNOUSERSITE=1 inherited, bare
        python3, a third-party import. It must run in the daemon's interpreter."""
        from prometheus.gateway import cron_scheduler as cs

        monkeypatch.setattr(cs, "vet_cron_command", lambda command, cwd=None: (True, ""))
        monkeypatch.setattr(cs, "mark_job_run", lambda *a, **k: None)
        job = {"name": "py-job", "command": f'python3 -c "import pydantic; {PROBE}"',
               "cwd": str(tmp_path), "enabled": True}
        entry = await cs.execute_job(job)
        assert entry["returncode"] == 0, entry["stderr"]
        assert entry["stdout"].strip() == sys.executable
        assert entry["command"] == job["command"], "history keeps the command as written"


class TestBackgroundTaskIsUnchanged:

    async def test_a_python3_task_gets_what_a_plain_shell_finds(self, tmp_path):
        """Never the daemon's venv: the model can start tasks (Will, 2026-10-04)."""
        from prometheus.tasks.manager import BackgroundTaskManager
        from prometheus.tasks.store import TaskStore
        from tests.test_managed_tasks import AllowGate, _wait_terminal

        command = f'python3 -c "{PROBE}"; echo "PATH=$PATH"'
        plain = _bash(command)
        mgr = BackgroundTaskManager(store=TaskStore(), security_gate=AllowGate())
        rec = await mgr.create_shell_task(command=command, description="py task",
                                          cwd=str(tmp_path))
        done = await _wait_terminal(mgr, rec.id)
        output = Path(done.output_file).read_text()
        assert done.return_code == 0, output
        task_python, task_path = output.strip().splitlines()[-2:]
        assert task_python == plain.stdout.strip().splitlines()[0], \
            "a task's python3 is exactly what a plain login shell finds"
        assert task_python != sys.executable, "never the daemon's own interpreter"
        assert "job-python" not in task_path, "the cron wrappers are not on a task's PATH"
