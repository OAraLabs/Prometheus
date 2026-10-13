"""A relative write in a workspace session lands in the workspace — and is seen there.

THE REPORT (parity re-record of ``checkpoint_undo``, 2026-10-13). A session
bound to the workspace ``scratch`` was asked to create ``greeting.txt``. The
model called ``write_file({"path": "greeting.txt"})`` and got
``Permission denied for write_file``; it fell back to
``cd <ws> && printf … > greeting.txt`` in bash, which worked, and then the file
mutation verifier reported ``greeting.txt — redirect_write: CLAIMED but FILE
ABSENT`` about a file that was sitting in the workspace.

Two defects, one shape — each layer resolved a relative path against a
different directory than the tool that used it:

* THE GATE (``permissions.tool_paths.gate_path_for``) treated every relative
  WRITE path as UNKNOWN, which forces an approval. That rule exists so that the
  directory the daemon process happens to run in never decides whether a write
  is allowed (#345). In a workspace session the base is not that: it is the
  session's workspace — the directory the system prompt names as the
  conversation's working directory, the directory ``write_file`` itself
  resolves against, and the write boundary the gate enforces. Refusing to
  resolve there made the gate ask about a path it could have ruled on exactly.

* THE VERIFIER (``hooks.file_mutation_verifier``) ``os.stat``-ed relative paths
  as given — that is, against the daemon process's cwd — and ignored the
  ``cd`` in front of the redirect. Its docstring said relative paths were left
  alone so as not to invent a path against the daemon's cwd; leaving them
  alone is exactly what did that.

Both are driven here through the real loop with the real ``write_file``,
``bash``, ``SecurityGate`` and verifier. Nothing touches a live daemon.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.hooks.file_mutation_verifier import FileMutationVerifier
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.tool_paths import gate_path_for
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.file_write import FileWriteTool


class _Scripted(ModelProvider):
    """Round 1: the given tool calls. Round 2: a final text reply."""

    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self.calls = calls
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if len(self.requests) == 1:
            content = [ToolUseBlock(id=f"t{i}", name=name, input=args)
                       for i, (name, args) in enumerate(self.calls)]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


class _RecordingVerifier(FileMutationVerifier):
    """The real verifier, keeping the summary the loop would inject."""

    def __init__(self) -> None:
        super().__init__()
        self.summaries: list[str] = []

    def post_turn(self, *, turn_key=None):  # noqa: ANN001
        summary = super().post_turn(turn_key=turn_key)
        if summary:
            self.summaries.append(summary)
        return summary


def _run(tmp_path: Path, calls, *, workspace: bool = True):
    """Daemon cwd ``boot``; session workspace ``ws`` (or none)."""
    boot = tmp_path / "boot"; boot.mkdir()
    ws = tmp_path / "ws"; ws.mkdir()
    registry = ToolRegistry()
    registry.register(FileWriteTool())
    registry.register(BashTool())
    asked: list[tuple[str, str]] = []

    async def deny(tool_name, reason, **_kw):  # noqa: ANN001
        # What an unattended parity run does with an approval: nobody says yes.
        asked.append((tool_name, reason))
        return False

    fmv = _RecordingVerifier()
    results: list = []
    ctx = LoopContext(
        provider=_Scripted(calls), model="stub", system_prompt="BASE", max_tokens=64,
        tool_registry=registry, permission_checker=SecurityGate(workspace_root=str(boot)),
        cwd=boot, permission_prompt=deny, file_mutation_verifier=fmv,
        workspace_resolver=(lambda sid: str(ws)) if workspace else None,
    )
    messages = [ConversationMessage.from_user_text("go")]

    async def go():
        async for _ in run_loop(ctx, messages, session_id="desktop:parity-undo"):
            pass
    asyncio.run(go())
    for msg in messages:
        for block in msg.content:
            if getattr(block, "tool_use_id", None):
                results.append(block)
    return boot, ws, asked, fmv.summaries, results


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #

def test_relative_write_file_lands_in_the_workspace(tmp_path: Path) -> None:
    """The reported call, verbatim in shape: allowed, written in the workspace,
    and never in the daemon's directory."""
    boot, ws, asked, summaries, results = _run(
        tmp_path, [("write_file", {"path": "greeting.txt", "content": "hello parity"})],
    )
    assert asked == [], f"a write inside the session workspace raised an approval: {asked}"
    assert [r.is_error for r in results] == [False], [r.content for r in results]
    assert (ws / "greeting.txt").read_text() == "hello parity"
    assert not (boot / "greeting.txt").exists()
    # And the verifier saw the same file the tool wrote.
    assert summaries and "✓" in summaries[0] and "FILE ABSENT" not in summaries[0], summaries


def test_relative_write_escaping_the_workspace_still_asks(tmp_path: Path) -> None:
    """Resolving is what lets the BOUNDARY rule: ``../`` out of the workspace is
    a write outside it, and the workspace rule (not the unknown-path rule) asks."""
    boot, ws, asked, _summaries, results = _run(
        tmp_path, [("write_file", {"path": "../escape.txt", "content": "x"})],
    )
    assert len(asked) == 1 and "outside workspace" in asked[0][1], asked
    assert results[0].is_error
    assert not (tmp_path / "escape.txt").exists()


def test_without_a_workspace_a_relative_write_still_asks(tmp_path: Path) -> None:
    """#345's rule, unchanged where its reason holds: with no session workspace
    the base IS the daemon's cwd, so a relative write stays unknown."""
    boot, _ws, asked, _summaries, results = _run(
        tmp_path, [("write_file", {"path": "greeting.txt", "content": "x"})], workspace=False,
    )
    assert len(asked) == 1 and "relative path" in asked[0][1], asked
    assert results[0].is_error and not (boot / "greeting.txt").exists()


@pytest.mark.parametrize("tool", ["download_file", "tts", "image_generate", "youtube_transcript"])
def test_tools_that_resolve_elsewhere_stay_unknown(tool: str, tmp_path: Path) -> None:
    """Only tools that resolve a relative path against the tool context's cwd
    get the workspace base. ``download_file`` resolves against ``Path.cwd()``
    (the process), so resolving it here would rule on a path it never writes."""
    from prometheus.permissions.tool_paths import TOOL_PATH_PARAM
    param = TOOL_PATH_PARAM[tool]
    path, unknown = gate_path_for(tool, {param: "out.bin"}, base=tmp_path, write_base=tmp_path)
    assert path is None and unknown and "relative path" in unknown


@pytest.mark.parametrize("tool", ["write_file", "edit_file", "notebook_edit"])
def test_workspace_relative_tools_really_resolve_against_the_context_cwd(tool: str, tmp_path: Path) -> None:
    """The membership claim, checked against each tool rather than asserted:
    a tool in WORKSPACE_RELATIVE_WRITE_TARGETS must write ``<cwd>/<relative>``,
    or the gate would rule on one path while the tool writes another."""
    from prometheus.permissions.tool_paths import WORKSPACE_RELATIVE_WRITE_TARGETS
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.file_edit import FileEditTool
    from prometheus.tools.builtin.notebook_edit import NotebookEditTool

    assert tool in WORKSPACE_RELATIVE_WRITE_TARGETS
    cwd = tmp_path / "ws"; (cwd / "sub").mkdir(parents=True)
    ctx = ToolExecutionContext(cwd=cwd)
    if tool == "write_file":
        rel = "sub/f.txt"
        t = FileWriteTool(); args = {"path": rel, "content": "x"}
    elif tool == "edit_file":
        rel = "sub/f.txt"; (cwd / rel).write_text("old")
        t = FileEditTool(); args = {"path": rel, "old_str": "old", "new_str": "x"}
    else:
        rel = "sub/f.ipynb"
        t = NotebookEditTool(); args = {"path": rel, "cell_index": 0, "new_source": "x", "mode": "append"}
    result = asyncio.run(t.execute(t.input_model(**args), ctx))
    assert not result.is_error, result.output
    assert (cwd / rel).is_file() and "x" in (cwd / rel).read_text()
    gate_path, unknown = gate_path_for(tool, args, base=cwd, write_base=cwd)
    assert unknown is None and gate_path == str((cwd / rel).resolve())


# --------------------------------------------------------------------------- #
# the verifier
# --------------------------------------------------------------------------- #

def test_verifier_follows_the_cd_in_the_bash_fallback(tmp_path: Path) -> None:
    """The model's fallback, verbatim in shape: ``cd <ws> && printf … > f``."""
    _boot, ws, _asked, summaries, results = _run(
        tmp_path, [("bash", {"command": f"cd {tmp_path / 'ws'} && printf 'hello parity' > greeting.txt"})],
    )
    assert not results[0].is_error, results[0].content
    assert (ws / "greeting.txt").read_text() == "hello parity"
    assert summaries, "the write was not reported at all"
    assert "FILE ABSENT" not in summaries[0], summaries[0]
    assert str(ws / "greeting.txt") in summaries[0], summaries[0]


def test_verifier_resolves_a_bare_bash_redirect_against_the_session_cwd(tmp_path: Path) -> None:
    """No ``cd`` at all: bash already runs in the workspace, so a bare
    ``> f`` lands there and must be stat'd there."""
    _boot, ws, _asked, summaries, _results = _run(
        tmp_path, [("bash", {"command": "printf hi > inventory.txt"})],
    )
    assert (ws / "inventory.txt").exists()
    assert summaries and "FILE ABSENT" not in summaries[0], summaries
    assert str(ws / "inventory.txt") in summaries[0], summaries[0]


class TestCdWalk:
    """The bash walk resolves relative targets where the shell actually was —
    and says it is blind wherever it cannot know that."""

    WS = "/srv/ws"

    def _audit(self, command: str, cwd: str | None = WS):
        from prometheus.hooks.file_mutation_verifier import audit_bash_command
        return audit_bash_command(command, cwd)

    def test_no_cwd_keeps_relative_targets_as_written(self) -> None:
        """Callers that name no cwd see exactly the old output."""
        audit = self._audit("cd /a && echo x > f.txt", cwd=None)
        assert audit.tracked == [("f.txt", "redirect_write")] and not audit.blind

    def test_bare_redirect_joins_the_cwd(self) -> None:
        assert self._audit("echo x > f.txt").tracked == [(f"{self.WS}/f.txt", "redirect_write")]

    def test_absolute_targets_are_untouched(self) -> None:
        assert self._audit("cd /a && echo x > /b/f.txt").tracked == [("/b/f.txt", "redirect_write")]

    @pytest.mark.parametrize("command,expected", [
        ("cd /a && echo x > f.txt", "/a/f.txt"),
        ("cd /a; echo x > f.txt", "/a/f.txt"),
        ("cd /a\necho x > f.txt", "/a/f.txt"),
        ("cd /a || exit 1\necho x > f.txt", "/a/f.txt"),
        ("cd sub && cd deeper && touch f.txt", "/srv/ws/sub/deeper/f.txt"),
        ("cd .. && touch f.txt", "/srv/f.txt"),
        ("cd /a 2>/dev/null && touch f.txt", "/a/f.txt"),
        ("cd -P /a && touch f.txt", "/a/f.txt"),
        ("if cd /a; then touch f.txt; fi", "/a/f.txt"),
    ])
    def test_a_followable_cd_moves_the_base(self, command: str, expected: str) -> None:
        audit = self._audit(command)
        assert [p for p, _ in audit.tracked] == [expected] and not audit.blind, audit

    def test_cd_alone_goes_home(self) -> None:
        import os
        audit = self._audit("cd && touch f.txt")
        assert audit.tracked == [(os.path.join(os.path.expanduser("~"), "f.txt"), "touch")]

    @pytest.mark.parametrize("command", [
        "cd - && touch f.txt",
        "cd $DIR && touch f.txt",
        "cd \"$(mktemp -d)\" && touch f.txt",
        "pushd && touch f.txt",
        "cd a b && touch f.txt",
        "ls | cd /a && touch f.txt",         # cd in a pipeline: a subshell
        "cd /a | cat && touch f.txt",
        "cd /a & touch f.txt",               # backgrounded: a subshell
        "(cd /a && make) ; touch f.txt",     # the walk cannot see the paren's scope
    ])
    def test_an_unfollowable_cd_makes_later_relative_targets_blind(self, command: str) -> None:
        audit = self._audit(command)
        assert audit.tracked == [] and audit.blind, (
            f"{command!r}: a relative target after a cd the walk cannot follow must be "
            f"reported blind, not stat'd somewhere the shell may never have been: {audit}"
        )

    def test_a_redirect_on_the_cd_itself_is_still_tracked(self) -> None:
        """`cd /a > log` opens `log` before the cd runs: the old directory."""
        audit = self._audit("cd /a > log.txt && touch f.txt")
        assert audit.tracked == [(f"{self.WS}/log.txt", "redirect_write"), ("/a/f.txt", "touch")], audit

    def test_an_unknown_dir_does_not_blind_absolute_targets(self) -> None:
        audit = self._audit("cd $DIR && touch /abs/f.txt")
        assert audit.tracked == [("/abs/f.txt", "touch")] and not audit.blind

    def test_sh_c_gets_its_own_walk_and_its_cd_ends_with_it(self) -> None:
        audit = self._audit("sh -c 'cd /a && touch f.txt' && touch g.txt")
        assert audit.tracked == [("/a/f.txt", "touch"), (f"{self.WS}/g.txt", "touch")], audit

    def test_the_bash_cwd_argument_is_where_the_command_runs(self, tmp_path: Path) -> None:
        """``bash(cwd=...)`` runs there, so the verifier stats there."""
        fmv = FileMutationVerifier()
        elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir()
        args = {"command": "printf hi > out.txt", "cwd": str(elsewhere)}
        fmv.pre_tool_use("bash", args, "b1", cwd=tmp_path)
        (elsewhere / "out.txt").write_text("hi")
        fmv.post_tool_use("bash", args, "b1", cwd=tmp_path)
        assert fmv.landed_paths() == [str(elsewhere / "out.txt")]

    def test_relative_write_file_path_joins_the_cwd(self, tmp_path: Path) -> None:
        fmv = FileMutationVerifier()
        args = {"path": "notes/a.md", "content": "x"}
        fmv.pre_tool_use("write_file", args, "w1", cwd=tmp_path)
        (tmp_path / "notes").mkdir(); (tmp_path / "notes" / "a.md").write_text("x")
        fmv.post_tool_use("write_file", args, "w1", cwd=tmp_path)
        assert fmv.landed_paths() == [str(tmp_path / "notes" / "a.md")]
