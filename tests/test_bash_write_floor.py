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
import uuid
from pathlib import Path

import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.bash import BashTool, BashToolInput


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


def _run(tool: BashTool, command: str, cwd: Path):
    ctx = ToolExecutionContext(cwd=cwd)
    return asyncio.run(tool.execute(BashToolInput(command=command), ctx))


def _bash(workspace: Path) -> BashTool:
    """The tool as production builds it: workspace locked, no other argument.

    Deliberately NOT a hand-built variant with the floor switched on. If the
    floor has to be asked for, the live daemon does not have it.
    """
    return BashTool(workspace=workspace)


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
