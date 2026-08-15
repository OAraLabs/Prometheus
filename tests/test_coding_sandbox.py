"""ProcessSandbox — cwd jail, symlink-resolved confinement, env scrub, timeouts.

SPRINT-coding-mode v2 scope item 3. The side effects asserted here are the
security properties: a planted secret in the daemon's environment must not
reach a sandboxed process; a symlink inside the jail pointing outside it is
denied, not followed; a timed-out command's process tree is actually dead.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid
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

import json

from prometheus.coding.sandbox import (
    CONTAINER_WORKDIR,
    DockerSandbox,
    SandboxBackendUnavailable,
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
# Container reuse — adopting a leaked container (subprocess mocked)
#
# The container name is derived from the task id ALONE, so a run that dies
# without close() leaves a RUNNING container that the next run with that task
# id will find. Reusing it on "status == running" alone adopts a container
# whose /workspace bind points at a clone that no longer exists, and every
# exec into it fails with "current working directory is outside of container
# mount namespace" — so nothing runs, no escape attempt can succeed, and a
# containment check reports a FALSE PASS. These pin the reuse decision itself;
# TestDockerSandboxLive proves it against a real daemon.
# --------------------------------------------------------------------------- #


class _FakeDocker:
    """Scripted `docker` CLI standing in for subprocess.run."""

    def __init__(
        self,
        inspect_payload=None,
        *,
        exec_ok: bool = True,
        inspect_stdout: str | None = None,
    ):
        # None = `docker inspect` reports no such container.
        self.inspect_payload = inspect_payload
        # Raw stdout override, for output that is not valid JSON at all.
        self.inspect_stdout = inspect_stdout
        self.exec_ok = exec_ok
        self.calls: list[list[str]] = []

    def __call__(self, argv, *args, **kwargs):
        self.calls.append(list(argv))
        verb = argv[1] if len(argv) > 1 else ""
        if verb == "inspect":
            if self.inspect_stdout is not None:
                return MagicMock(returncode=0, stdout=self.inspect_stdout, stderr="")
            if self.inspect_payload is None:
                return MagicMock(returncode=1, stdout="", stderr="No such object")
            return MagicMock(
                returncode=0, stdout=json.dumps(self.inspect_payload), stderr=""
            )
        if verb == "exec":
            return MagicMock(
                returncode=0 if self.exec_ok else 126,
                stdout="",
                stderr="" if self.exec_ok else (
                    "OCI runtime exec failed: exec failed: unable to start "
                    "container process: current working directory is outside "
                    "of container mount namespace"
                ),
            )
        return MagicMock(returncode=0, stdout="", stderr="")

    @property
    def verbs(self) -> list[str]:
        return [c[1] for c in self.calls if len(c) > 1]


def _record(source, *, status: str = "running", destination: str = CONTAINER_WORKDIR):
    """A minimal `docker inspect` record with one bind mount."""
    return [{
        "State": {"Status": status},
        "Mounts": [{
            "Type": "bind",
            "Source": str(source),
            "Destination": destination,
        }],
    }]


class TestDockerSandboxContainerReuse:

    def _build(self, root: Path, fake: _FakeDocker, task_id: str = "reuse-task"):
        with patch("prometheus.coding.sandbox.subprocess.run", side_effect=fake):
            return DockerSandbox(root=root, task_id=task_id)

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        d = tmp_path / "jail"
        d.mkdir()
        return d

    def test_healthy_container_is_adopted(self, root: Path):
        """The reuse path still exists: a running container whose workspace
        bind is this run's root and which can exec is NOT rebuilt."""
        fake = _FakeDocker(_record(root))
        self._build(root, fake)
        assert "create" not in fake.verbs
        assert "rm" not in fake.verbs

    def test_deleted_workspace_source_is_rebuilt(self, root: Path, tmp_path: Path):
        """The leaked container is running, but its bind source was deleted
        with the crashed run's clone.

        These mount-shape cases pin the decision table going forward; they do
        NOT by themselves demonstrate the original defect, since the mock is
        written against the current single-inspect call shape. The live pair
        in TestDockerSandboxLive is what fails against the old code.
        """
        fake = _FakeDocker(_record(tmp_path / "gone"))
        self._build(root, fake)
        assert "rm" in fake.verbs
        assert "create" in fake.verbs

    def test_other_runs_clone_is_rebuilt(self, root: Path, tmp_path: Path):
        """Production shape: clone_repo_for_sandbox timestamps each clone
        directory, so a retry of the same task id has the same container name
        but a DIFFERENT root. The still-existing older clone must not be
        mistaken for this run's workspace."""
        stale = tmp_path / "jail-1700000000"
        stale.mkdir()
        fake = _FakeDocker(_record(stale))
        self._build(root, fake)
        assert "rm" in fake.verbs
        assert "create" in fake.verbs

    def test_container_without_workspace_mount_is_rebuilt(self, root: Path):
        fake = _FakeDocker(_record(root, destination="/elsewhere"))
        self._build(root, fake)
        assert "rm" in fake.verbs
        assert "create" in fake.verbs

    def test_exited_container_is_rebuilt(self, root: Path):
        fake = _FakeDocker(_record(root, status="exited"))
        self._build(root, fake)
        assert "rm" in fake.verbs
        assert "create" in fake.verbs

    def test_running_but_unexecable_container_is_rebuilt(self, root: Path):
        """Path checks cannot see a re-created directory — a bind mount holds
        the original dentry, so a deleted-then-recloned workspace leaves the
        source present and matching while the mount inside is a corpse. The
        probe exec is the only thing that catches it."""
        fake = _FakeDocker(_record(root), exec_ok=False)
        with pytest.raises(SandboxBackendUnavailable):
            self._build(root, fake)
        assert "rm" in fake.verbs
        assert "create" in fake.verbs

    def test_probe_uses_the_container_workdir(self, root: Path):
        """A probe that does not enter /workspace would pass against exactly
        the dead mount it is meant to detect."""
        fake = _FakeDocker(_record(root))
        self._build(root, fake)
        probe = next(c for c in fake.calls if len(c) > 1 and c[1] == "exec")
        assert probe[probe.index("--workdir") + 1] == CONTAINER_WORKDIR

    def test_unexecable_fresh_container_raises_and_is_removed(self, root: Path):
        """No leftover container, so this is a straight create — and it still
        cannot exec. Returning it would present a sandbox in which every
        command fails as one that contains everything."""
        fake = _FakeDocker(None, exec_ok=False)
        with pytest.raises(SandboxBackendUnavailable, match="cannot execute"):
            self._build(root, fake)
        assert "rm" in fake.verbs

    def test_unparseable_inspect_output_rebuilds(self, root: Path):
        """Unreadable inspect output must not be read as a healthy container."""
        fake = _FakeDocker(inspect_stdout="not json")
        self._build(root, fake)
        assert "create" in fake.verbs


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

    @staticmethod
    def _container_uuid(name: str) -> str:
        """Docker's long id for *name* — changes on rebuild, unlike the name."""
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{.Id}}", name],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()

    def test_leaked_container_with_recreated_workspace_is_rebuilt(
        self, tmp_path: Path
    ):
        """REGRESSION, against a real daemon: the leak this was found by.

        A run crashes without close(), leaving a RUNNING container. Its clone
        is cleaned up and the task retried at the same path. Reuse keyed on
        status alone adopts the corpse — and note the path checks alone would
        NOT save us here, because the directory exists again and matches: the
        bind mount holds the ORIGINAL dentry, so the mount inside the
        container is dead while every host-side check passes.

        The failure is silent in the worst direction. Every exec fails with
        "current working directory is outside of container mount namespace",
        so no command can write anywhere and a containment check reports a
        PASS with nothing having run at all.
        """
        task_id = str(uuid.uuid4())
        root = tmp_path / "jail"
        root.mkdir()
        (root / "app.py").write_text("x = 1\n")

        leaked = DockerSandbox(root=root, task_id=task_id)
        stale_uuid = self._container_uuid(leaked._container_id)
        assert stale_uuid, "leaked container should exist before the retry"

        try:
            # The crash: no close(), clone removed, task retried at the path.
            shutil.rmtree(root)
            root.mkdir()
            (root / "app.py").write_text("x = 2\n")

            fresh = DockerSandbox(root=root, task_id=task_id)
            try:
                assert self._container_uuid(fresh._container_id) != stale_uuid, (
                    "adopted the leaked container instead of rebuilding"
                )
                # The property that actually matters: commands RUN.
                r = asyncio.run(fresh.run("cat app.py"))
                assert r.exit_code == 0, f"exec failed against reused container: {r.output}"
                assert "x = 2" in r.output
            finally:
                fresh.close()
        finally:
            subprocess.run(
                ["docker", "rm", "-f", leaked._container_id],
                capture_output=True, timeout=15,
            )

    def test_leaked_container_from_another_root_is_rebuilt(self, tmp_path: Path):
        """The production shape: clone_repo_for_sandbox timestamps every clone
        directory, so a retry of the same task id has the same container name
        but a different root. The leaked container's workspace is the wrong
        tree — adopting it would run the retry against stale code."""
        task_id = str(uuid.uuid4())
        first = tmp_path / "jail-1"
        first.mkdir()
        (first / "marker.txt").write_text("first\n")
        leaked = DockerSandbox(root=first, task_id=task_id)

        try:
            second = tmp_path / "jail-2"
            second.mkdir()
            (second / "marker.txt").write_text("second\n")

            fresh = DockerSandbox(root=second, task_id=task_id)
            try:
                r = asyncio.run(fresh.run("cat marker.txt"))
                assert r.exit_code == 0
                assert "second" in r.output, (
                    "container is still bound to the previous run's clone"
                )
            finally:
                fresh.close()
        finally:
            subprocess.run(
                ["docker", "rm", "-f", leaked._container_id],
                capture_output=True, timeout=15,
            )
