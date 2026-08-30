"""bash writes outside the workspace, unprompted, at every shape of write.

THE HOLE
--------
``SecurityGate`` raises an approval for ``write_file``/``edit_file`` that
target a path outside the workspace by inspecting their ``file_path``
argument (checker.py, ``_APPROVE_TOOLS``). The bash tool has no
``file_path``: it is handed a command string. So the one tool that can
write anywhere is the one tool the outside-workspace check cannot see, and
``printf 'x' > ~/outside.txt`` lands on disk with no prompt.

``exfiltration.py`` reads ``<`` into network commands. Nothing reads ``>``.

WHY THESE SEVEN CASES
---------------------
They are not seven bugs; they are one bug seen seven ways. Each is a
different *syntax* producing the same *effect* — a write at a path outside
the workspace — and that is the point: a fix that reads the command string
has to enumerate all of them and every one it has not thought of yet.
``>``, ``>>``, ``>|``, ``tee``, ``dd of=``, ``sed -i``, ``cp``, ``mv``,
``install``, ``python -c``, heredocs, ``$(...)``, ``sh -c`` of any of the
above. So these cases exist to be closed BELOW the syntax, at the point
where the kernel resolves the path, and to stay here afterwards as the
proof that they were.

The target is a uuid-suffixed dotfile in ``$HOME`` — a real path outside any
workspace root, which is the exact shape observed on a live daemon — and it
is removed whether the test passes or fails.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import pytest

from prometheus.permissions import confinement as C
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.bash import BashTool, BashToolInput


def _floor_available() -> bool:
    ok, _ = C.write_preflight(force=True)
    C.reset_write_cache()
    return ok


needs_floor = pytest.mark.skipif(
    not _floor_available(),
    reason=(
        "bubblewrap cannot contain on this host (no bwrap, or the namespace "
        "is refused). SKIPPED IS NOT PASSED: on this machine bash can write "
        "anywhere and these eight cases are live holes, not covered ground."
    ),
)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "seed.txt").write_text("inside\n")
    return ws


@pytest.fixture()
def outside() -> Path:
    """A path outside every workspace root, cleaned up unconditionally."""
    target = Path.home() / f".prometheus-write-floor-probe-{uuid.uuid4().hex}"
    yield target
    try:
        target.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture()
def outside_seed() -> Path:
    """An EXISTING file outside the workspace, for in-place editors."""
    target = Path.home() / f".prometheus-write-floor-seed-{uuid.uuid4().hex}"
    target.write_text("ORIGINAL\n")
    yield target
    try:
        target.unlink()
    except FileNotFoundError:
        pass


def _run(tool: BashTool, command: str, cwd: Path, timeout: int = 30):
    ctx = ToolExecutionContext(cwd=cwd)
    return asyncio.run(tool.execute(
        BashToolInput(command=command, timeout_seconds=timeout), ctx))


def _bash(workspace: Path) -> BashTool:
    """The tool as production builds it: workspace locked, no other argument.

    Deliberately NOT a hand-built variant with the floor switched on. If the
    floor has to be asked for, the live daemon does not have it.
    """
    return BashTool(workspace=workspace)


@needs_floor
class TestWriteOutsideWorkspaceIsRefused:
    """Seven syntaxes, one effect. Every one of them must fail to land."""

    def test_redirect(self, workspace, outside):
        _run(_bash(workspace), f"printf 'x' > {outside}", workspace)
        assert not outside.exists(), f"bash wrote outside the workspace: {outside}"

    def test_append_redirect(self, workspace, outside):
        _run(_bash(workspace), f"printf 'x' >> {outside}", workspace)
        assert not outside.exists(), f"bash appended outside the workspace: {outside}"

    def test_tee(self, workspace, outside):
        _run(_bash(workspace), f"printf 'x' | tee {outside}", workspace)
        assert not outside.exists(), f"tee wrote outside the workspace: {outside}"

    def test_dd(self, workspace, outside):
        _run(_bash(workspace), f"printf 'x' | dd of={outside} 2>/dev/null", workspace)
        assert not outside.exists(), f"dd wrote outside the workspace: {outside}"

    def test_sed_in_place(self, workspace, outside_seed):
        _run(_bash(workspace), f"sed -i 's/ORIGINAL/REWRITTEN/' {outside_seed}", workspace)
        assert outside_seed.read_text() == "ORIGINAL\n", (
            f"sed -i rewrote a file outside the workspace: {outside_seed}")

    def test_cp(self, workspace, outside):
        _run(_bash(workspace), f"cp {workspace / 'seed.txt'} {outside}", workspace)
        assert not outside.exists(), f"cp wrote outside the workspace: {outside}"

    def test_python_c(self, workspace, outside):
        _run(
            _bash(workspace),
            f"python3 -c \"open('{outside}','w').write('x')\"",
            workspace,
        )
        assert not outside.exists(), f"python -c wrote outside the workspace: {outside}"

    def test_heredoc(self, workspace, outside):
        _run(
            _bash(workspace),
            f"cat > {outside} <<'EOF'\nx\nEOF\n",
            workspace,
        )
        assert not outside.exists(), f"a heredoc wrote outside the workspace: {outside}"


class TestWriteInsideWorkspaceStillWorks:
    """The floor must be invisible to ordinary work, not merely survivable."""

    def test_redirect_inside_lands_unprompted(self, workspace):
        res = _run(_bash(workspace), "printf 'hello' > made.txt", workspace)
        assert not res.is_error, res.output
        assert (workspace / "made.txt").read_text() == "hello"

    def test_edit_inside_lands_unprompted(self, workspace):
        res = _run(_bash(workspace), "sed -i 's/inside/edited/' seed.txt", workspace)
        assert not res.is_error, res.output
        assert (workspace / "seed.txt").read_text() == "edited\n"


@needs_floor
class TestTheBoundaryIsTheEffectNotTheSyntax:
    """Cases no command-string parser could have caught."""

    def test_symlink_out_of_the_workspace_does_not_escape(self, workspace, outside):
        """A path that LOOKS inside the workspace and resolves outside it.

        The command names ``link``, a relative path under the workspace root.
        Every syntactic check passes. The kernel resolves it to $HOME and the
        write lands on the read-only mount.
        """
        (workspace / "link").symlink_to(outside)
        _run(_bash(workspace), "printf 'x' > link", workspace)
        assert not outside.exists(), f"a symlink escaped the workspace: {outside}"

    def test_indirection_through_a_variable_does_not_escape(self, workspace, outside):
        """The target never appears as a literal path in the command."""
        _run(
            _bash(workspace),
            f"T=$(printf '%s' '{outside}'); printf 'x' > \"$T\"",
            workspace,
        )
        assert not outside.exists(), f"$VAR indirection escaped: {outside}"

    def test_nested_shell_does_not_escape(self, workspace, outside):
        """``sh -c`` of the write — a second parser the first never sees."""
        _run(_bash(workspace), f"sh -c \"echo x > {outside}\"", workspace)
        assert not outside.exists(), f"a nested shell escaped: {outside}"


@needs_floor
class TestWhatTheFloorDeliberatelyDoesNotChange:
    """A floor that broke ordinary work would be switched off within a day."""

    def test_reads_outside_the_workspace_still_work(self, workspace):
        """The write floor is not a read floor, and must not be read as one.

        ``--ro-bind / /`` mounts the host readable. Keeping bash out of
        ~/.ssh is ``bash_confinement``'s job; conflating them would let this
        suite's green be read as protection it does not provide.
        """
        res = _run(_bash(workspace), "head -1 /etc/hostname", workspace)
        assert not res.is_error, res.output

    def test_scratch_and_cache_are_writable(self, workspace):
        res = _run(
            _bash(workspace),
            'printf x > /tmp/prom-floor-scratch.$$ && rm -f /tmp/prom-floor-scratch.$$ '
            '&& mkdir -p "${XDG_CACHE_HOME:-$HOME/.cache}/prom-floor" && echo SCRATCH_OK',
            workspace,
        )
        assert "SCRATCH_OK" in res.output, res.output

    def test_network_is_not_isolated(self, workspace):
        """This is a write boundary, not a sandbox — no --unshare-net."""
        res = _run(_bash(workspace), "getent hosts localhost", workspace)
        assert not res.is_error, res.output

    def test_the_result_records_that_the_floor_was_on(self, workspace):
        res = _run(_bash(workspace), "true", workspace)
        assert res.metadata.get("write_floor") == "active", res.metadata

    def test_a_timed_out_command_still_dies_with_its_children(self, workspace):
        """--new-session would orphan the inner shell; it is not passed.

        bwrap's own process is what ``_kill_process_group`` addresses. If the
        inner shell were in a session of its own, killing the outer group
        would leave the real work running and reparented to init — the
        orphaned-``find`` bug, re-introduced invisibly.
        """
        marker = workspace / "still-running.txt"
        res = _run(
            _bash(workspace),
            f"(sleep 12; printf x > {marker}) & wait",
            workspace,
            timeout=2,
        )
        assert res.is_error and "timed out" in res.output.lower(), res.output
        # Long enough for the orphan to have written if the kill missed it,
        # and far short of the sleep's own end so a natural exit cannot be
        # mistaken for a successful kill.
        time.sleep(4)
        assert not marker.exists(), "the inner command outlived the timeout kill"


class TestWiringRunsEverywhere:
    """No bubblewrap needed — these hold on every host, including macOS."""

    def test_no_workspace_means_no_boundary_to_enforce(self):
        """Unchanged behaviour, and it must be legible as such.

        With no workspace root there is nothing to be outside OF. The floor
        does not silently invent a boundary; it records that there is none.
        """
        tool = BashTool(write_confinement="required")
        res = _run(tool, "true", Path("/tmp"))
        assert res.metadata.get("write_floor") == "no-workspace", res.metadata

    def test_off_is_off(self, workspace, outside):
        tool = BashTool(workspace=workspace, write_confinement="off")
        res = _run(tool, f"printf 'x' > {outside}", workspace)
        assert res.metadata.get("write_floor") == "off"
        assert outside.exists(), "write_confinement=off must not confine"

    def test_required_refuses_when_the_floor_is_unavailable(self, workspace, monkeypatch):
        """Never fall through to an unconfined shell — the #237 bargain."""
        # Value swap, not a callable one: this exercises the real
        # shutil.which lookup against a name that is genuinely not on PATH.
        monkeypatch.setattr(C, "BWRAP_BIN", "bwrap-not-installed-probe")
        C.reset_write_cache()
        tool = BashTool(workspace=workspace, write_confinement="required")
        res = _run(tool, "echo I_RAN_UNCONFINED", workspace)
        C.reset_write_cache()
        assert res.is_error
        assert "I_RAN_UNCONFINED" not in res.output, "the command RAN"
        assert "REFUSED" in res.output
        assert res.metadata.get("write_floor") == "refused"

    def test_auto_degrades_loudly_rather_than_refusing(self, workspace, outside, monkeypatch):
        """macOS has no bubblewrap. Refusing every bash call there would be
        worse than the hole, so auto runs — and says, per call, that it did.
        """
        monkeypatch.setattr(C, "BWRAP_BIN", "bwrap-not-installed-probe")
        C.reset_write_cache()
        tool = BashTool(workspace=workspace, write_confinement="auto")
        res = _run(tool, f"printf 'x' > {outside}", workspace)
        C.reset_write_cache()
        assert not res.is_error, res.output
        assert res.metadata.get("write_floor") == "unavailable", res.metadata

    def test_unknown_mode_fails_safe_and_loud(self, caplog):
        assert C.normalise_write_mode("enforce-ish") == C.WRITE_MODE_AUTO
        assert "not one of" in caplog.text

    def test_write_file_is_untouched(self, tmp_path):
        """The path-declaring tools keep their own gate, unchanged.

        write_file has a file_path, so SecurityGate already sees it and
        already raises the outside-workspace approval. This work adds nothing
        to that path and must subtract nothing either.
        """
        from prometheus.tools.builtin.file_write import FileWriteTool

        tool = FileWriteTool()
        target = tmp_path / "written.txt"
        res = asyncio.run(tool.execute(
            tool.input_model(path=str(target), content="hello"),
            ToolExecutionContext(cwd=tmp_path),
        ))
        assert not res.is_error, res.output
        assert target.read_text() == "hello"
