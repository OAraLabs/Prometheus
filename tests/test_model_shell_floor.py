"""Every shell command a MODEL wrote runs behind the bash tool's floors.

The bash tool has three controls around the shell it starts: the READ floor
(AppArmor, ``security.bash_confinement``), the WRITE floor (bubblewrap,
``security.bash_write_confinement``) and the environment scrub
(``security/env_scrub.py``). They were wired into ``tools/builtin/bash.py``
and nowhere else, while three other doors take a command string from the
model and hand it to a shell:

    task_create type=local_bash   tasks/manager.py     _start_process
    task_create type=poll         tasks/watchers.py    _run_predicate
    cron_create                   gateway/cron_scheduler.py  execute_job

and a coding run hands the model's ``code_run`` commands to
``coding/sandbox.py`` ``ProcessSandbox.run``, which scrubbed the environment
but had no floor. So the floor that keeps ``bash`` out of ``~/.ssh`` was one
``task_create`` away from not existing.

The canary tests below read a file the profile denies through each door, the
way a model would — through the TOOL, with the path behind an indirection
(``cat "$(cat where.txt)"``) because the system-trust gate refuses a literal
``.ssh`` in a command string and passes this. That gate is a speed bump; the
kernel is the control, and that is what is measured.

The floors are configured the way an operator configures them: in
``prometheus.yaml``, at the path the conftest already isolates. Nothing here
calls a function that only exists after the fix to switch a floor on, so on a
tree without the fix these tests run and FAIL ON THEIR ASSERTIONS — the
canary is printed, the token is in the environment — rather than erroring at
an import.

Like ``test_bash_confinement.py``: the canary and write tests need the
``prometheus-bash`` profile loaded and bubblewrap installed. They skip where
those are absent, and **a skip is not a pass** — CI's ``security-floors`` job
runs this file with both present and fails on any skip.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest
import yaml

from prometheus.permissions import confinement as C
from prometheus.tasks.manager import get_task_manager
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput
from tests.test_bash_confinement import needs_profile, require_floor_subject

#: Set in the daemon's environment by every test; must never reach a child.
SECRET_NAME = "PROMETHEUS_API_TOKEN"

TERMINAL = {"completed", "failed", "killed", "blocked"}


@pytest.fixture(autouse=True)
def _fresh_preflight():
    C.reset_cache()
    C.reset_write_cache()
    yield
    C.reset_cache()
    C.reset_write_cache()


@pytest.fixture()
def secret(monkeypatch) -> str:
    value = f"tok-{uuid.uuid4().hex}"
    monkeypatch.setenv(SECRET_NAME, value)
    return value


def _configure(*, read: str = "off", write: str = "off",
               workspace: Path | None = None) -> None:
    """Write the floors into prometheus.yaml, as an operator would."""
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    sec: dict[str, object] = {
        "bash_confinement": read,
        "bash_write_confinement": write,
    }
    if workspace is not None:
        sec["workspace_root"] = str(workspace)
    (cfg_dir / "prometheus.yaml").write_text(yaml.safe_dump({"security": sec}))


async def _model_task(cwd: Path, **fields):
    """A task started the way a model starts one: the task_create TOOL."""
    manager = get_task_manager()
    manager.poll_initial_interval = 0.5
    manager.poll_max_interval = 0.5
    res = await TaskCreateTool().execute(
        TaskCreateToolInput(description="floor probe", **fields),
        ToolExecutionContext(cwd=cwd, metadata={}),
    )
    task_id = (res.metadata or {}).get("task_id")
    if task_id is None:  # refused at register: the id is in the message
        return res, None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        task = manager.get_task(task_id)
        if task is not None and task.status in TERMINAL:
            break
        await asyncio.sleep(0.05)
    task = manager.get_task(task_id)
    return res, task


def _output(task) -> str:
    return get_task_manager().read_task_output(task.id)


async def _model_cron_job(command: str, cwd: Path, *, name: str | None = None,
                          extra: dict | None = None) -> dict:
    """Create a cron job the way a MODEL does — the cron_create TOOL — and
    return it as stored. ``extra`` is merged into the raw tool arguments, so a
    test can try to smuggle a field past the tool's schema."""
    from prometheus.gateway.cron_service import get_cron_job
    from prometheus.tools.builtin.cron_create import (
        CronCreateTool,
        CronCreateToolInput,
    )

    name = name or f"floor-probe-{uuid.uuid4().hex[:6]}"
    args = {"name": name, "schedule": "0 0 1 1 *", "command": command,
            "cwd": str(cwd), **(extra or {})}
    res = await CronCreateTool().execute(
        CronCreateToolInput.model_validate(args),
        ToolExecutionContext(cwd=cwd, metadata={}),
    )
    assert not res.is_error, res.output
    job = get_cron_job(name)
    assert job is not None
    return job


def _operator_cron_job(command: str, cwd: Path, *, name: str | None = None) -> dict:
    """Create a cron job the way the OPERATOR does — ``POST /api/cron`` — and
    return it as stored."""
    from fastapi.testclient import TestClient

    from prometheus.gateway.cron_service import get_cron_job
    from prometheus.web.server import create_app

    name = name or f"operator-{uuid.uuid4().hex[:6]}"
    token = os.environ.get(SECRET_NAME, "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    res = TestClient(create_app({})).post(
        "/api/cron", headers=headers,
        json={"name": name, "schedule": "0 0 1 1 *", "command": command,
              "cwd": str(cwd)},
    )
    assert res.status_code == 201, res.text
    job = get_cron_job(name)
    assert job is not None
    return job


async def _run_job(job: dict) -> dict:
    from prometheus.gateway.cron_scheduler import execute_job

    return await execute_job(job)


async def _run_cron(command: str, cwd: Path) -> dict:
    """A cron job the MODEL created, run by the scheduler."""
    return await _run_job(await _model_cron_job(command, cwd))


def _coding_sandbox(root: Path):
    from prometheus.coding.sandbox import ProcessSandbox

    return ProcessSandbox(root=root)


def _bwrap_works() -> bool:
    ok, _ = C.write_preflight(force=True)
    C.reset_write_cache()
    return ok


needs_bwrap = pytest.mark.skipif(
    not _bwrap_works(),
    reason=(
        "bubblewrap cannot mount the write floor here (not installed, or "
        "unprivileged user namespaces are blocked). SKIPPED IS NOT PASSED."
    ),
)


# --------------------------------------------------------------------------- #
# The environment scrub — runs everywhere, needs no profile
# --------------------------------------------------------------------------- #


class TestTheDaemonsSecretsStayOutOfAModelsShell:
    """``printenv``, run where the model put it, must not print the token."""

    @pytest.mark.asyncio
    async def test_a_background_task(self, tmp_path, secret):
        _configure()
        _, task = await _model_task(
            tmp_path, command=f"echo RAN; printenv {SECRET_NAME} || echo SCRUBBED")
        out = _output(task)
        assert "RAN" in out, f"the task never ran: {task}"
        assert secret not in out, (
            "a model-started background task printed the daemon's API token")
        assert "SCRUBBED" in out

    @pytest.mark.asyncio
    async def test_a_poll_predicate(self, tmp_path, secret):
        """The predicate succeeds only when the token is NOT in its environment."""
        _configure()
        _, task = await _model_task(
            tmp_path, type="poll", timeout_seconds=3,
            poll_predicate=f'test -z "${SECRET_NAME}"')
        assert task.status == "completed", (
            f"the poll predicate saw the daemon's token ({task.status}, "
            f"{task.error})")

    @pytest.mark.asyncio
    async def test_a_cron_job(self, tmp_path, secret):
        _configure()
        entry = await _run_cron(
            f"echo RAN; printenv {SECRET_NAME} || echo SCRUBBED", tmp_path)
        assert "RAN" in entry["stdout"], entry
        assert secret not in entry["stdout"] + entry["stderr"], (
            "a cron job printed the daemon's API token")

    @pytest.mark.asyncio
    async def test_a_coding_run_command(self, tmp_path, secret):
        """Already true before this change (the sandbox allowlists six
        variables); pinned so flooring the sandbox cannot loosen it."""
        _configure()
        res = await _coding_sandbox(tmp_path).run(
            f"echo RAN; printenv {SECRET_NAME} || echo SCRUBBED")
        assert "RAN" in res.output
        assert secret not in res.output


# --------------------------------------------------------------------------- #
# Fail loud — runs everywhere, needs no profile
# --------------------------------------------------------------------------- #


@pytest.fixture()
def read_floor_unavailable(monkeypatch):
    """The read floor is required and this host cannot provide it."""
    monkeypatch.setattr(
        C, "preflight",
        lambda *a, **k: (False, "profile 'prometheus-bash' does not exist (test)"),
    )


class TestRequiredButUnavailableRefusesRatherThanRunningUnconfined:
    """``bash_confinement: required`` must refuse every model shell it can't
    confine — never run it without the floor. The bash tool already did."""

    @pytest.mark.asyncio
    async def test_a_background_task(self, tmp_path, read_floor_unavailable):
        _configure(read="required")
        res, task = await _model_task(tmp_path, command="echo I_RAN_UNCONFINED")
        if task is not None:
            assert "I_RAN_UNCONFINED" not in _output(task), (
                "the task ran with the read floor required and unavailable")
            assert task.status == "blocked", task
        assert res.is_error
        assert "REFUSED" in res.output

    @pytest.mark.asyncio
    async def test_a_poll_predicate(self, tmp_path, read_floor_unavailable):
        marker = tmp_path / "ran"
        _configure(read="required")
        res, task = await _model_task(
            tmp_path, type="poll", timeout_seconds=2,
            poll_predicate=f'touch "{marker}"')
        assert not marker.exists(), (
            "the poll predicate ran with the read floor required and unavailable")
        assert res.is_error
        assert "REFUSED" in res.output

    @pytest.mark.asyncio
    async def test_a_cron_job(self, tmp_path, read_floor_unavailable):
        _configure(read="required")
        entry = await _run_cron("echo I_RAN_UNCONFINED", tmp_path)
        assert "I_RAN_UNCONFINED" not in entry["stdout"], (
            "the cron job ran with the read floor required and unavailable")
        assert entry["status"] == "blocked", entry
        assert "REFUSED" in entry["stderr"]

    @pytest.mark.asyncio
    async def test_a_coding_run_command(self, tmp_path, read_floor_unavailable):
        _configure(read="required")
        marker = tmp_path / "ran"
        with pytest.raises(Exception, match="REFUSED"):
            await _coding_sandbox(tmp_path).run(f'touch "{marker}"')
        assert not marker.exists()


# --------------------------------------------------------------------------- #
# THE CANARY — the read floor bites at every door. Needs the profile.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def canary(tmp_path):
    """A secret in a path the profile denies, and an indirection to it.

    Written into ``~/.ssh`` because that is the subtree the profile guards and
    the one the D15 route goes after. Removed afterwards. The guard ERRORS
    (not skips) if ``~/.ssh`` is absent or HOME is outside ``@{HOME}``.
    """
    ssh = require_floor_subject(Path.home() / ".ssh")
    value = f"CANARY-{uuid.uuid4().hex}"
    path = ssh / f"prometheus-floor-canary-{uuid.uuid4().hex[:8]}"
    path.write_text(value + "\n")
    (tmp_path / "where.txt").write_text(str(path))
    (tmp_path / "want.txt").write_text(value)
    try:
        yield value
    finally:
        path.unlink(missing_ok=True)


#: What the profile's refusal looks like, as opposed to an absent file.
REFUSED = "Permission denied"


@needs_profile
class TestAModelCannotReadTheCanary:

    @pytest.mark.asyncio
    async def test_through_a_background_task(self, tmp_path, canary):
        _configure(read="required")
        _, task = await _model_task(
            tmp_path, command='echo RAN; cat "$(cat where.txt)"')
        out = _output(task)
        assert "RAN" in out, f"the task never ran: {task}"
        assert canary not in out, (
            "a model-started background task READ the denied-path canary")
        assert REFUSED in out, out

    @pytest.mark.asyncio
    async def test_through_a_poll_predicate(self, tmp_path, canary):
        """The predicate succeeds only if it can read the canary."""
        _configure(read="required")
        _, task = await _model_task(
            tmp_path, type="poll", timeout_seconds=2,
            poll_predicate='grep -q "$(cat want.txt)" "$(cat where.txt)"')
        assert task.status != "completed", (
            "a model-started poll predicate READ the denied-path canary")

    @pytest.mark.asyncio
    async def test_through_a_cron_job(self, tmp_path, canary):
        _configure(read="required")
        entry = await _run_cron('echo RAN; cat "$(cat where.txt)"', tmp_path)
        assert "RAN" in entry["stdout"], entry
        assert canary not in entry["stdout"], (
            "a model-created cron job READ the denied-path canary")
        assert REFUSED in entry["stderr"], entry

    @pytest.mark.asyncio
    async def test_through_a_coding_run(self, tmp_path, canary):
        _configure(read="required")
        res = await _coding_sandbox(tmp_path).run(
            'echo RAN; cat "$(cat where.txt)"')
        assert "RAN" in res.output, res
        assert canary not in res.output, (
            "a coding run's code_run READ the denied-path canary")
        assert REFUSED in res.output, res.output

    @pytest.mark.asyncio
    async def test_the_bash_tool_still_cannot(self, tmp_path, canary):
        """The control case: the one door that was already floored."""
        from prometheus.tools.builtin.bash import BashTool, BashToolInput

        res = await BashTool(confinement="required").execute(
            BashToolInput(command='echo RAN; cat "$(cat where.txt)"'),
            ToolExecutionContext(cwd=tmp_path),
        )
        assert "RAN" in res.output
        assert canary not in res.output
        assert REFUSED in res.output


# --------------------------------------------------------------------------- #
# The write floor bites at every door. Needs bubblewrap.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def outside(tmp_path):
    """A path outside the workspace the model is told to write."""
    target = Path.home() / f"prometheus-floor-outside-{uuid.uuid4().hex[:8]}"
    (tmp_path / "where.txt").write_text(str(target))
    try:
        yield target
    finally:
        target.unlink(missing_ok=True)


@needs_bwrap
class TestAModelCannotWriteOutsideTheWorkspace:

    @pytest.mark.asyncio
    async def test_through_a_background_task(self, tmp_path, outside):
        _configure(write="required", workspace=tmp_path)
        _, task = await _model_task(
            tmp_path, command='echo RAN; printf x > "$(cat where.txt)"')
        assert "RAN" in _output(task), task
        assert not outside.exists(), (
            "a model-started background task WROTE outside the workspace")

    @pytest.mark.asyncio
    async def test_through_a_poll_predicate(self, tmp_path, outside):
        _configure(write="required", workspace=tmp_path)
        await _model_task(
            tmp_path, type="poll", timeout_seconds=2,
            poll_predicate='printf x > "$(cat where.txt)"')
        assert not outside.exists(), (
            "a model-started poll predicate WROTE outside the workspace")

    @pytest.mark.asyncio
    async def test_through_a_cron_job(self, tmp_path, outside):
        _configure(write="required", workspace=tmp_path)
        entry = await _run_cron('echo RAN; printf x > "$(cat where.txt)"', tmp_path)
        assert "RAN" in entry["stdout"], entry
        assert not outside.exists(), (
            "a model-created cron job WROTE outside the workspace")

    @pytest.mark.asyncio
    async def test_through_a_coding_run(self, tmp_path, outside):
        _configure(write="required")
        res = await _coding_sandbox(tmp_path).run(
            'echo RAN; printf x > "$(cat where.txt)"')
        assert "RAN" in res.output, res
        assert not outside.exists(), (
            "a coding run's code_run WROTE outside its clone")

    @pytest.mark.asyncio
    async def test_the_workspace_itself_stays_writable(self, tmp_path):
        """The other half: a floor that blocks everything breaks the work."""
        _configure(write="required", workspace=tmp_path)
        _, task = await _model_task(tmp_path, command="printf ok > inside.txt")
        assert task.status == "completed", (task.status, _output(task))
        assert (tmp_path / "inside.txt").read_text() == "ok"


# --------------------------------------------------------------------------- #
# What must NOT change
# --------------------------------------------------------------------------- #


class TestDaemonAuthoredCommandsAreNotFloored:
    """Two launches go through the same spawn site with a command the DAEMON
    built, not the model: a sub-agent (``python -m prometheus``) and a coding
    run's launcher. They need the daemon's environment (provider keys) and
    write its state dirs, so flooring them would break them. The model's
    shells inside them are floored where they run (the child's own bash tool,
    the coding sandbox)."""

    @pytest.mark.asyncio
    async def test_a_sub_agent_launch_is_not_floored(self, tmp_path, monkeypatch):
        manager = get_task_manager()
        seen: dict = {}

        async def spy(**kwargs):
            seen.update(kwargs)
            raise RuntimeError("stop here")

        monkeypatch.setattr(manager, "create_shell_task", spy)
        with pytest.raises(RuntimeError, match="stop here"):
            await manager.create_agent_task(
                prompt="hi", description="d", cwd=tmp_path, api_key="k")
        assert seen.get("floored") is False, seen

    @pytest.mark.asyncio
    async def test_a_coding_run_launch_is_not_floored(self, tmp_path):
        from prometheus.coding import managed

        seen: dict = {}

        class _Manager:
            async def create_shell_task(self, **kwargs):
                seen.update(kwargs)
                return None

        await managed.create_coding_managed_task(
            _Manager(), repo=str(tmp_path), description="d",
            acceptance_command="true", task_id="c1", cwd=str(tmp_path))
        assert seen.get("floored") is False, seen

    @pytest.mark.asyncio
    async def test_an_unfloored_launch_keeps_the_environment(
        self, tmp_path, secret, read_floor_unavailable,
    ):
        """And the exemption really is one: required-but-unavailable does not
        refuse it, and it still sees the daemon's environment."""
        _configure(read="required")
        manager = get_task_manager()
        task = await manager.create_shell_task(
            command=f"echo RAN; printenv {SECRET_NAME}",
            description="daemon-authored", cwd=tmp_path, floored=False)
        deadline = time.monotonic() + 15
        while manager.get_task(task.id).status not in TERMINAL:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.05)
        out = manager.read_task_output(task.id)
        assert "RAN" in out and secret in out, out


@pytest.fixture()
def tokens(monkeypatch, secret):
    """Two ``*_TOKEN`` variables in the daemon's environment."""
    bot = f"bot-{uuid.uuid4().hex}"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", bot)
    return (secret, bot)


#: Prints every *_TOKEN variable the shell can see, or NO_TOKENS.
PRINT_TOKENS = 'echo RAN; env | grep "_TOKEN=" || echo NO_TOKENS'


def _api():
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    token = os.environ.get(SECRET_NAME, "")
    return (TestClient(create_app({})),
            {"Authorization": f"Bearer {token}"} if token else {})


class TestCronIsFlooredByProvenance:
    """A cron job is floored by WHO WROTE IT, not because it is a cron job.

    The operator's own jobs need what the floor removes — a briefing reads its
    Telegram token from the environment, watcher and vault jobs write outside
    the workspace — so only a job the MODEL created (the cron_create tool)
    runs floored and scrubbed. It is stored with ``origin: "model"``.

    * no ``origin`` (every job stored before this) and the operator's path
      (``POST /api/cron``): today's behaviour, unchanged;
    * the model's tool always writes ``origin: "model"`` and has no way to
      say anything else;
    * a model edit of an operator job (cron_create replaces by name) makes it
      the model's; an operator edit (``PUT``) never makes a model job the
      operator's.
    """

    @pytest.mark.asyncio
    async def test_a_job_the_model_creates_is_stored_as_the_models(self, tmp_path):
        _configure()
        job = await _model_cron_job("true", tmp_path)
        assert job.get("origin") == "model", job

    @pytest.mark.asyncio
    async def test_the_model_cannot_choose_its_jobs_origin(self, tmp_path):
        from prometheus.tools.builtin.cron_create import CronCreateToolInput

        assert "origin" not in CronCreateToolInput.model_fields
        _configure()
        job = await _model_cron_job(
            "true", tmp_path, extra={"origin": "operator"})
        assert job.get("origin") == "model", job

    @pytest.mark.asyncio
    async def test_a_model_job_has_no_token_in_its_environment(self, tmp_path, tokens):
        _configure()
        entry = await _run_cron(PRINT_TOKENS, tmp_path)
        out = entry["stdout"] + entry["stderr"]
        assert "RAN" in out, entry
        assert "NO_TOKENS" in entry["stdout"], entry
        for value in tokens:
            assert value not in out, "a model-created cron job saw a *_TOKEN"

    @pytest.mark.asyncio
    async def test_an_operator_job_keeps_its_environment(self, tmp_path, tokens):
        _configure()
        job = _operator_cron_job(PRINT_TOKENS, tmp_path)
        assert "origin" not in job, job
        entry = await _run_job(job)
        for value in tokens:
            assert value in entry["stdout"], (
                "the operator's cron job lost its environment", entry)

    @pytest.mark.asyncio
    async def test_an_operator_job_is_not_floored(
        self, tmp_path, read_floor_unavailable,
    ):
        """``required`` and unavailable refuses a model job; the operator's
        job runs exactly as it did."""
        _configure(read="required")
        entry = await _run_job(_operator_cron_job("echo OPERATOR_RAN", tmp_path))
        assert entry["status"] == "success", entry
        assert "OPERATOR_RAN" in entry["stdout"]

    @pytest.mark.asyncio
    async def test_a_job_stored_before_origin_existed_is_unchanged(
        self, tmp_path, tokens, read_floor_unavailable,
    ):
        from prometheus.gateway.cron_service import get_cron_job, upsert_cron_job

        _configure(read="required")
        upsert_cron_job({"name": "legacy", "schedule": "0 0 1 1 *",
                         "command": PRINT_TOKENS, "cwd": str(tmp_path)})
        entry = await _run_job(get_cron_job("legacy"))
        assert entry["status"] == "success", entry
        for value in tokens:
            assert value in entry["stdout"]

    @pytest.mark.asyncio
    async def test_a_model_edit_of_an_operator_job_makes_it_the_models(
        self, tmp_path, tokens,
    ):
        _configure()
        _operator_cron_job("echo OPERATOR", tmp_path, name="briefing")
        job = await _model_cron_job(PRINT_TOKENS, tmp_path, name="briefing")
        assert job.get("origin") == "model", job
        entry = await _run_job(job)
        assert "NO_TOKENS" in entry["stdout"], entry

    @pytest.mark.asyncio
    async def test_a_model_edit_of_an_operator_job_is_floored(
        self, tmp_path, read_floor_unavailable,
    ):
        _configure(read="required")
        _operator_cron_job("echo OPERATOR", tmp_path, name="vault")
        job = await _model_cron_job("echo I_RAN_UNCONFINED", tmp_path, name="vault")
        entry = await _run_job(job)
        assert entry["status"] == "blocked", entry
        assert "I_RAN_UNCONFINED" not in entry["stdout"]

    @pytest.mark.asyncio
    async def test_an_operator_edit_does_not_unfloor_a_models_job(
        self, tmp_path, tokens,
    ):
        from prometheus.gateway.cron_service import get_cron_job

        _configure()
        await _model_cron_job("echo A", tmp_path, name="m1")
        client, headers = _api()
        res = client.put("/api/cron/m1", headers=headers,
                         json={"command": PRINT_TOKENS, "origin": "operator"})
        assert res.status_code == 200, res.text
        job = get_cron_job("m1")
        assert job.get("origin") == "model", job
        entry = await _run_job(job)
        assert "NO_TOKENS" in entry["stdout"], entry

    def test_the_operator_path_writes_no_origin(self, tmp_path):
        client, headers = _api()
        res = client.post("/api/cron", headers=headers, json={
            "name": "op", "schedule": "0 0 1 1 *", "command": "true",
            "cwd": str(tmp_path), "origin": "model"})
        assert res.status_code == 201, res.text
        assert "origin" not in res.json()["job"]


class TestOneFloorForEveryDoor:
    def test_the_registry_wires_the_same_floor_the_bash_tool_uses(self):
        """``create_tool_registry`` is where both the daemon and the CLI build
        the bash tool. The other doors read the floor it wires there, so the
        two cannot be configured apart."""
        from prometheus.__main__ import create_tool_registry
        from prometheus.security import shell_floor

        try:
            create_tool_registry({
                "bash_confinement": "required",
                "bash_write_confinement": "off",
                "workspace_root": "/srv/ws",
            })
            floor = shell_floor.current_shell_floor()
            assert floor.read_mode == "required"
            assert floor.write_mode == "off"
            assert floor.workspaces == (Path("/srv/ws").resolve(),)
        finally:
            shell_floor.set_shell_floor(None)

    def test_unwired_reads_the_config(self):
        """Never left unwired: with nothing set, the floor comes from the
        config, the way cron's gate does."""
        from prometheus.security import shell_floor

        shell_floor.set_shell_floor(None)
        _configure(read="required", write="auto")
        floor = shell_floor.current_shell_floor()
        assert floor.read_mode == "required"
        assert floor.write_mode == "auto"

    def test_the_report_states_the_wider_scope(self):
        rep = C.floor_report(probe=False)
        scope = str(rep["scope"])
        for door in ("bash tool", "background", "poll", "cron", "coding"):
            assert door in scope, scope
        assert "hooks" in scope, (
            "the one shell site left out must be named, not implied")


def test_this_file_runs_in_the_security_floors_job():
    """A canary test that only ever skips proves nothing. The CI job that has
    the profile and bubblewrap must run this file and fail on a skip."""
    ci = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    assert "tests/test_model_shell_floor.py" in ci.read_text()
