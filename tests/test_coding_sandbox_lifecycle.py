"""Coding sandboxes must be released, uniquely named, and honest about git.

THREE DEFECTS, ONE RUN PATH
---------------------------
1. `Sandbox.close()` existed on the interface and was called from NOWHERE in
   `src/` — verified by grep. For `ProcessSandbox` it is a documented no-op;
   for `DockerSandbox` it is the only thing that stops the container. Every
   docker-backed run left a sleeping container with the jail clone still
   bind-mounted, until the daemon restarted.

2. The container name is derived from the task id
   (`_container_id_for_task`), and `--task-id` comes from the caller. Two runs
   with the same id addressed the SAME container:

       _container_id_for_task('my-task') = prometheus-coding-mytask
       _container_id_for_task('my-task') = prometheus-coding-mytask

3. `CodingSession._git` returned `result.output` and DISCARDED
   `result.exit_code`. A failed `git checkout -b` returned its error text as an
   ordinary string and the run continued on whatever branch the repo was on,
   while `self._branch` went on naming the branch that was never created.

   Reproduced: with `checkout -q -b` exiting 128, `_prepare_branch()` returned
   normally and the session still reported `branch='coding/t1'`.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.coding.sandbox import (  # noqa: E402
    SandboxResult,
    _container_id_for_task,
)
from prometheus.coding.session import CodingGitError, CodingSession  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# ── (3) git exit codes ──────────────────────────────────────────────────────

class _Sandbox:
    """Records commands; returns a scripted exit code."""

    root = Path(".")

    def __init__(self, exit_code: int = 0, output: str = ""):
        self.exit_code = exit_code
        self.output = output
        self.commands: list[str] = []
        self.closed = False

    async def run(self, cmd, timeout_seconds=None):
        self.commands.append(cmd)
        return SandboxResult(
            exit_code=self.exit_code, output=self.output,
            timed_out=False, duration_seconds=0.0,
        )

    def close(self):
        self.closed = True


def _session(sandbox) -> CodingSession:
    from types import SimpleNamespace

    session = CodingSession.__new__(CodingSession)
    session._sandbox = sandbox
    session._branch = "coding/t1"
    # _commit_artifact reads the task id for the commit message.
    session._task = SimpleNamespace(task_id="t1")
    return session


def test_a_failed_branch_creation_raises_instead_of_continuing():
    """THE defect: the run continued on the wrong branch."""
    sandbox = _Sandbox(exit_code=128, output="fatal: a branch named 'x' already exists")
    with pytest.raises(CodingGitError) as caught:
        asyncio.run(_session(sandbox)._prepare_branch())

    assert "checkout" in str(caught.value)
    assert "128" in str(caught.value)
    assert "already exists" in str(caught.value), (
        "the error text git produced is not in the message — the operator "
        "cannot tell WHY the branch was not created"
    )


def test_a_failed_commit_raises():
    """`_commit_artifact` runs three git commands; any of them can fail."""
    sandbox = _Sandbox(exit_code=1, output="nothing to commit")
    with pytest.raises(CodingGitError):
        asyncio.run(_session(sandbox)._commit_artifact("success"))


def test_a_successful_branch_creation_is_silent():
    """The fix must not make every git call an error.

    Without this, `raise` unconditionally would satisfy the tests above and
    break every coding run.
    """
    sandbox = _Sandbox(exit_code=0, output="")
    asyncio.run(_session(sandbox)._prepare_branch())
    assert sandbox.commands == ["git checkout -q -b coding/t1"]


def test_the_diff_summary_tolerates_a_non_zero_exit():
    """`diff --stat` over a range that does not exist yet is not a failure.

    The artifact summary is best-effort reporting, not a control — and making
    it fatal would turn a successful run into a failed one on an empty repo.
    """
    sandbox = _Sandbox(exit_code=0, output="")
    session = _session(sandbox)

    calls: list[tuple] = []
    original = CodingSession._git

    async def spy(self, *args, check=True):
        calls.append((args, check))
        return ""

    CodingSession._git = spy
    try:
        asyncio.run(session._commit_artifact("success"))
    finally:
        CodingSession._git = original

    diff_calls = [c for c in calls if c[0] and c[0][0] == "diff"]
    assert diff_calls, calls
    assert diff_calls[0][1] is False, (
        "the diff summary is checked, so an empty range would fail the run"
    )


# ── (1) the sandbox is released ─────────────────────────────────────────────

def test_the_run_path_closes_the_sandbox_on_success_and_on_failure():
    """`close()` must be called on BOTH exits of the coding run block.

    Asserted on the source of `_run_coding`, because driving the whole CLI
    entry point needs a provider, a repo and a model. What matters is that the
    call exists in a `finally` — the failure paths are exactly when a container
    is most likely to be left behind, and they are the two that returned early.
    """
    main_src = (REPO / "src" / "prometheus" / "__main__.py").read_text()
    assert "sandbox.close()" in main_src, (
        "nothing in __main__ closes the sandbox; DockerSandbox.close() is the "
        "only thing that stops the container"
    )
    # The call must be in a finally, not only on the success path.
    tail = main_src[main_src.index("report = asyncio.run(session.run())"):]
    finally_at = tail.index("finally:")
    close_at = tail.index("sandbox.close()")
    assert close_at > finally_at, (
        "sandbox.close() is not inside the finally — a failed run still leaks"
    )


def test_close_exists_on_every_sandbox_backend():
    """The interface promise the run path now relies on."""
    from prometheus.coding.sandbox import DockerSandbox, ProcessSandbox, Sandbox

    for cls in (Sandbox, ProcessSandbox, DockerSandbox):
        assert hasattr(cls, "close"), f"{cls.__name__} has no close()"


def test_a_cleanup_failure_does_not_replace_the_runs_verdict():
    """Teardown must not turn a successful run into a failure.

    The report is already printed by the time cleanup runs; a hiccup there is
    worth a warning, not a different exit code.
    """
    main_src = (REPO / "src" / "prometheus" / "__main__.py").read_text()
    tail = main_src[main_src.index("finally:"):]
    block = tail[: tail.index("# ---")] if "# ---" in tail else tail
    assert "try:" in block and "except Exception" in block, (
        "sandbox.close() is not guarded — a cleanup error would escape and "
        "replace the run's own outcome"
    )


# ── (2) a reused task id must not reuse a container ─────────────────────────

def test_the_container_name_is_still_derived_from_its_id():
    """Pins the premise: same id in, same container name out."""
    assert _container_id_for_task("my-task") == _container_id_for_task("my-task")


def test_two_runs_with_the_same_task_id_get_different_containers():
    """The sandbox INSTANCE id carries the uniqueness, not the task id.

    A caller may reuse a task id deliberately — a retry of the same logical
    task — and must still get a clean container.
    """
    main_src = (REPO / "src" / "prometheus" / "__main__.py").read_text()
    assert "sandbox_instance_id" in main_src, (
        "the sandbox is still named from the caller-supplied task id"
    )
    assert "task_id=sandbox_instance_id" in main_src, (
        "the unique id is computed but not passed to the sandbox"
    )
    # And the branch / report still use the REAL task id.
    assert "task_id=task_id," in main_src, (
        "the CodingTask no longer carries the caller's task id — the branch "
        "name and every report field would change"
    )


def test_the_instance_id_is_actually_unique():
    """Two derivations from one task id must differ."""
    import re
    from uuid import uuid4 as _u

    task_id = "my-task"
    a = f"{task_id}-{_u().hex[:8]}"
    b = f"{task_id}-{_u().hex[:8]}"
    assert a != b
    assert _container_id_for_task(a) != _container_id_for_task(b)
    assert re.match(rf"^{task_id}-[0-9a-f]{{8}}$", a), a
