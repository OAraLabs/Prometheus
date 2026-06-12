"""ProcessSandbox — cwd jail, symlink-resolved confinement, env scrub, timeouts.

SPRINT-coding-mode v2 scope item 3. The side effects asserted here are the
security properties: a planted secret in the daemon's environment must not
reach a sandboxed process; a symlink inside the jail pointing outside it is
denied, not followed; a timed-out command's process tree is actually dead.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from prometheus.coding.sandbox import (
    ProcessSandbox,
    SandboxViolation,
    clone_repo_for_sandbox,
)


@pytest.fixture
def box(tmp_path: Path) -> ProcessSandbox:
    root = tmp_path / "jail"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n")
    return ProcessSandbox(root=root)


# --------------------------------------------------------------------------- #
# Path confinement
# --------------------------------------------------------------------------- #


class TestResolve:

    def test_relative_path_resolves_inside(self, box: ProcessSandbox):
        assert box.resolve("src/app.py") == box.root / "src" / "app.py"

    def test_dotdot_escape_denied(self, box: ProcessSandbox):
        with pytest.raises(SandboxViolation, match="escapes the sandbox"):
            box.resolve("../../etc/passwd")

    def test_absolute_outside_denied(self, box: ProcessSandbox):
        with pytest.raises(SandboxViolation, match="escapes the sandbox"):
            box.resolve("/etc/passwd")

    def test_absolute_inside_allowed(self, box: ProcessSandbox):
        inside = box.root / "src" / "app.py"
        assert box.resolve(str(inside)) == inside

    def test_symlink_out_of_jail_denied(self, box: ProcessSandbox, tmp_path: Path):
        outside = tmp_path / "outside-secret.txt"
        outside.write_text("secret")
        link = box.root / "innocent.txt"
        link.symlink_to(outside)
        with pytest.raises(SandboxViolation, match="escapes the sandbox"):
            box.resolve("innocent.txt")

    def test_denied_paths_enforced_inside_root(self, tmp_path: Path):
        root = tmp_path / "jail2"
        (root / "secrets").mkdir(parents=True)
        (root / "secrets" / "k.pem").write_text("x")
        box = ProcessSandbox(root=root, denied_paths=(root / "secrets",))
        with pytest.raises(SandboxViolation, match="denied by policy"):
            box.resolve("secrets/k.pem")
        # …and the rest of the jail still resolves.
        (root / "ok.txt").write_text("y")
        assert box.resolve("ok.txt") == (root / "ok.txt").resolve()

    def test_root_must_exist(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not a directory"):
            ProcessSandbox(root=tmp_path / "missing")


# --------------------------------------------------------------------------- #
# Execution — scrub, capture, exit codes, timeout tree-kill
# --------------------------------------------------------------------------- #


class TestRun:

    def test_runs_in_root_and_captures(self, box: ProcessSandbox):
        r = asyncio.run(box.run("pwd && echo hi"))
        assert r.exit_code == 0
        assert str(box.root) in r.output
        assert "hi" in r.output
        assert not r.timed_out

    def test_nonzero_exit_reported_not_raised(self, box: ProcessSandbox):
        r = asyncio.run(box.run("exit 7"))
        assert r.exit_code == 7
        assert not r.timed_out

    def test_env_scrub_drops_secrets(self, box: ProcessSandbox, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", "sekrit-token-123")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-sekrit")
        r = asyncio.run(box.run("env"))
        assert "sekrit-token-123" not in r.output
        assert "sk-sekrit" not in r.output
        assert "PROMETHEUS_API_TOKEN" not in r.output
        # …while the allowlist survives (PATH is what test runners need).
        assert "PATH=" in r.output
        assert "PYTHONUNBUFFERED=1" in r.output

    def test_timeout_kills_process_tree(self, box: ProcessSandbox, tmp_path: Path):
        marker = tmp_path / "child-survived.txt"
        # Parent spawns a child that would write the marker after 3 s; the
        # 1 s timeout must kill the WHOLE group, so the marker never appears.
        cmd = f"(sleep 3 && touch {marker}) & sleep 30"
        r = asyncio.run(box.run(cmd, timeout_seconds=1.0))
        assert r.timed_out
        assert r.exit_code is None
        # Give a dead child a beat to prove it's dead, not slow.
        import time

        time.sleep(3.5)
        assert not marker.exists(), "child escaped the timeout kill"

    def test_long_output_head_tail_truncated(self, box: ProcessSandbox):
        r = asyncio.run(box.run("python3 -c \"print('x' * 50000)\""))
        assert r.exit_code == 0
        assert "truncated" in r.output
        assert len(r.output) < 50_000


# --------------------------------------------------------------------------- #
# clone_repo_for_sandbox — the dedicated FULL CLONE jail
# --------------------------------------------------------------------------- #


class TestClone:

    def test_full_clone_is_isolated(self, tmp_path: Path):
        import subprocess

        src = tmp_path / "origin"
        src.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=src, check=True)
        (src / "a.txt").write_text("hello\n")
        subprocess.run(["git", "add", "."], cwd=src, check=True)
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t",
             "commit", "-qm", "init"],
            cwd=src, check=True,
        )

        box = clone_repo_for_sandbox(src, tmp_path / "sandboxes", name="run1")
        assert (box.root / "a.txt").read_text() == "hello\n"
        assert box.root != src
        # Writes in the jail never touch the origin.
        (box.root / "a.txt").write_text("changed\n")
        assert (src / "a.txt").read_text() == "hello\n"

    def test_existing_target_refused(self, tmp_path: Path):
        import subprocess

        src = tmp_path / "origin2"
        src.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=src, check=True)
        dest_parent = tmp_path / "sb"
        (dest_parent / "run1").mkdir(parents=True)
        with pytest.raises(ValueError, match="already exists"):
            clone_repo_for_sandbox(src, dest_parent, name="run1")
