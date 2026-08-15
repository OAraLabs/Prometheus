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
        # Stale-pyc guard: same-size same-second edits must never let a
        # test run import old bytecode (CPython pyc mtime is whole-second).
        assert "PYTHONDONTWRITEBYTECODE=1" in r.output

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


# --------------------------------------------------------------------------- #
# DockerSandbox — container-based isolation (subprocess mocked)
# --------------------------------------------------------------------------- #

from unittest.mock import AsyncMock, MagicMock, patch

from prometheus.coding.sandbox import (
    CONTAINER_WORKDIR,
    DockerSandbox,
    docker_available,
)


class TestDockerSandbox:

    def _make_box(self, tmp_path: Path, task_id: str = "test-task", **kw) -> DockerSandbox:
        root = tmp_path / "jail"
        root.mkdir(exist_ok=True)
        with patch("prometheus.coding.sandbox.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
            box = DockerSandbox(root=root, task_id=task_id, **kw)
        return box

    def test_backend_identifier(self, tmp_path: Path):
        box = self._make_box(tmp_path)
        assert box.backend == "docker"

    def test_container_id_deterministic(self, tmp_path: Path):
        box1 = self._make_box(tmp_path, task_id="task-a")
        box2 = self._make_box(tmp_path, task_id="task-a")
        assert box1._container_id == box2._container_id

    def test_network_isolation_flag(self, tmp_path: Path):
        box = self._make_box(tmp_path, network_isolation=True)
        assert box._network_isolation is True

    def test_custom_image(self, tmp_path: Path):
        box = self._make_box(tmp_path, image="my-image:latest")
        assert box._image == "my-image:latest"

    def test_resolve_inside_jail(self, tmp_path: Path):
        box = self._make_box(tmp_path)
        (box.root / "src").mkdir(exist_ok=True)
        (box.root / "src" / "app.py").touch()
        assert box.resolve("src/app.py") == box.root / "src" / "app.py"

    def test_resolve_absolute_inside_jail(self, tmp_path: Path):
        box = self._make_box(tmp_path)
        x = tmp_path / "jail" / "x"
        x.touch()
        assert box.resolve(str(x)) == x

    def test_resolve_outside_jail_raises(self, tmp_path: Path):
        box = self._make_box(tmp_path)
        with pytest.raises(SandboxViolation):
            box.resolve("/etc/passwd")

    def test_run_executes_docker_exec(self, tmp_path: Path):
        """run() shells out to `docker exec` and reports its exit code.

        Patches ``asyncio.create_subprocess_exec``, not ``subprocess.run``:
        run() is a coroutine on the daemon's event loop, and a blocking
        subprocess call there stalls every other session for the duration of
        the command. The assertion below on the WORKDIR is the load-bearing
        one — see test_run_uses_container_workdir_not_host_path.
        """
        box = self._make_box(tmp_path)

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.communicate = AsyncMock(return_value=(b"hello\n", None))

        async def _fake_exec(*args, **kwargs):
            _fake_exec.argv = list(args)
            return fake_proc

        with patch(
            "prometheus.coding.sandbox.asyncio.create_subprocess_exec",
            side_effect=_fake_exec,
        ):
            result = asyncio.run(box.run("echo hello"))

        assert result.exit_code == 0
        assert "hello" in result.output
        argv = _fake_exec.argv
        assert argv[0] == "docker"
        assert argv[1] == "exec"
        assert "--workdir" in argv

    def test_run_uses_container_workdir_not_host_path(self, tmp_path: Path):
        """REGRESSION: --workdir must be the in-container mount point.

        The host path does not exist inside the container's mount namespace,
        so passing str(root) made every exec fail with "no such file or
        directory" — a failure that reads like the command's fault rather
        than the sandbox's. Verified against a live container: the host path
        is genuinely absent inside, only CONTAINER_WORKDIR resolves.
        """
        box = self._make_box(tmp_path)

        fake_proc = MagicMock()
        fake_proc.returncode = 0
        fake_proc.communicate = AsyncMock(return_value=(b"", None))

        async def _fake_exec(*args, **kwargs):
            _fake_exec.argv = list(args)
            return fake_proc

        with patch(
            "prometheus.coding.sandbox.asyncio.create_subprocess_exec",
            side_effect=_fake_exec,
        ):
            asyncio.run(box.run("true"))

        argv = _fake_exec.argv
        workdir = argv[argv.index("--workdir") + 1]
        assert workdir == CONTAINER_WORKDIR
        assert str(box.root) not in argv

    def test_run_on_closed_sandbox_raises(self, tmp_path: Path):
        box = self._make_box(tmp_path)
        box.close()
        with pytest.raises(RuntimeError, match="already closed"):
            asyncio.run(box.run("echo ok"))

    def test_close_removes_container(self, tmp_path: Path):
        box = self._make_box(tmp_path)
        fake_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
        with patch("prometheus.coding.sandbox.subprocess.run", return_value=fake_result) as mock_run:
            box.close()
            calls = [c[0][0] for c in mock_run.call_args_list]
            assert any(c[0] == "docker" and c[1] == "rm" for c in calls)

    def test_docker_available_false_when_no_docker(self):
        with patch("prometheus.coding.sandbox.subprocess.run", side_effect=FileNotFoundError()):
            assert docker_available() is False

    def test_docker_available_false_when_docker_fails(self):
        fake = MagicMock(returncode=1, stdout=b"", stderr=b"not found")
        with patch("prometheus.coding.sandbox.subprocess.run", return_value=fake):
            assert docker_available() is False

    def test_docker_available_true_when_docker_ok(self):
        fake = MagicMock(returncode=0, stdout=b"Server version:", stderr=b"")
        with patch("prometheus.coding.sandbox.subprocess.run", return_value=fake):
            assert docker_available() is True


# --------------------------------------------------------------------------- #
# DockerSandbox against a REAL daemon
#
# The mocked class above proves argv shape; it cannot prove containment. These
# do — and they are the only tests in the repo that demonstrate the coding
# sandbox actually refusing a shell escape, since bwrap is blocked on this
# host (see tests/test_bwrap_sandbox.py and the BwrapSandbox docstring).
# Gated on a reachable daemon so machines without Docker skip rather than fail.
# --------------------------------------------------------------------------- #

requires_docker = pytest.mark.skipif(
    not docker_available(), reason="no reachable Docker daemon"
)


@requires_docker
class TestDockerSandboxLive:

    def _live_box(self, tmp_path: Path) -> DockerSandbox:
        import uuid

        root = tmp_path / "jail"
        root.mkdir(exist_ok=True)
        (root / "app.py").write_text("x = 1\n")
        return DockerSandbox(root=root, task_id=str(uuid.uuid4()))

    def test_shell_redirect_outside_the_jail_is_contained(self, tmp_path: Path):
        """THE ACCEPTANCE TEST. ProcessSandbox fails this by design (rc=0,
        file lands outside); the whole point of a kernel/container backend is
        that it does not."""
        outside = tmp_path / "outside" / "escaped.txt"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.unlink(missing_ok=True)
        box = self._live_box(tmp_path)
        try:
            r = asyncio.run(box.run(f"echo pwned > {outside}"))
            assert not outside.exists(), "redirect escaped the container"
            assert r.exit_code != 0
        finally:
            box.close()

    def test_workspace_is_readable_and_writable(self, tmp_path: Path):
        """--cap-drop ALL removes CAP_DAC_OVERRIDE, so a root container
        cannot even READ a user-owned bind mount. Running as the host uid is
        what makes the jail usable at all — this fails loudly if that
        regresses."""
        box = self._live_box(tmp_path)
        try:
            assert "x = 1" in asyncio.run(box.run("cat app.py")).output
            r = asyncio.run(box.run("echo inside > new.txt"))
            assert r.exit_code == 0
            assert (box.root / "new.txt").exists()
        finally:
            box.close()

    def test_created_files_are_not_root_owned(self, tmp_path: Path):
        """A container running as root would litter the clone with
        root-owned artifacts the daemon user cannot clean up."""
        box = self._live_box(tmp_path)
        try:
            asyncio.run(box.run("echo hi > owned.txt"))
            assert (box.root / "owned.txt").stat().st_uid == os.getuid()
        finally:
            box.close()

    def test_root_filesystem_is_read_only(self, tmp_path: Path):
        box = self._live_box(tmp_path)
        try:
            r = asyncio.run(box.run("echo x > /etc/should-not-write"))
            assert r.exit_code != 0
            assert "read-only" in r.output.lower()
        finally:
            box.close()

    def test_nonzero_exit_is_reported_not_raised(self, tmp_path: Path):
        box = self._live_box(tmp_path)
        try:
            assert asyncio.run(box.run("exit 7")).exit_code == 7
        finally:
            box.close()

    def test_timeout_reports_and_leaves_sandbox_usable(self, tmp_path: Path):
        """After a timeout the container is killed and restarted, so the next
        iterate-to-green round still has a working sandbox rather than every
        later call failing against a dead container."""
        box = self._live_box(tmp_path)
        try:
            r = asyncio.run(box.run("sleep 30", timeout_seconds=1.0))
            assert r.timed_out
            assert r.exit_code is None
            follow_up = asyncio.run(box.run("echo still-alive"))
            assert follow_up.exit_code == 0
            assert "still-alive" in follow_up.output
        finally:
            box.close()
