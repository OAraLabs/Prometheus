"""A file tool opens exactly the path the gate checked, and nothing a link could put in its place (GHSA-wgcv-xw9m-9q9q).

The loop asks the gate about a file tool's path, resolved; the tool then opened the path again by name, following
links. A folder in the workspace swapped for a link between the two sent the write or read somewhere the gate never
ruled on, and a hard link to a protected file inside the workspace passed the gate with no race at all.

The race is reproduced here the way it happens: the path is decided (``metadata["decided_path"]``, as the loop records
the gate's ruling), THEN a folder along it is swapped for a link to another folder, THEN the tool runs. Nothing may
land in, or be read from, the other folder. The second half swaps the link at the last moment, between the tool's own
resolution and its open (``safe_fs.before_open``).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from prometheus.security import safe_fs
from prometheus.tools.base import ToolExecutionContext


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A workspace with a folder ``sub`` the race swaps, and an ``elsewhere`` it points the swap at."""
    root = tmp_path.resolve()
    ws = root / "ws"
    (ws / "sub").mkdir(parents=True)
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    (ws / "sub" / "notes.txt").write_text("ws notes\n")
    (elsewhere / "notes.txt").write_text("PROTECTED\n")
    (ws / "sub" / "nb.ipynb").write_text(json.dumps(_notebook("ws")))
    (elsewhere / "nb.ipynb").write_text(json.dumps(_notebook("PROTECTED")))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(root / "cfg"))
    monkeypatch.setattr(safe_fs, "before_open", lambda path: None)
    monkeypatch.setattr(safe_fs, "_identity_cache", None, raising=False)
    return ws, elsewhere


def _notebook(text: str) -> dict:
    return {"cells": [{"cell_type": "code", "source": [text], "metadata": {}, "outputs": [], "execution_count": None}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _swap(ws: Path, elsewhere: Path) -> None:
    """Replace ws/sub with a link to elsewhere, as a shell running beside the tool could."""
    shutil.rmtree(ws / "sub")
    os.symlink(elsewhere, ws / "sub")


def _ctx(ws: Path, decided: Path | None) -> ToolExecutionContext:
    meta = {"workspace_roots": [str(ws)]}
    if decided is not None:
        meta["decided_path"] = str(decided)
    return ToolExecutionContext(cwd=ws, metadata=meta)


def _write_file():
    from prometheus.tools.builtin.file_write import FileWriteTool, FileWriteToolInput
    return FileWriteTool(), lambda p: FileWriteToolInput(path=str(p), content="EVIL\n")


def _edit_file():
    from prometheus.tools.builtin.file_edit import FileEditTool, FileEditToolInput
    return FileEditTool(), lambda p: FileEditToolInput(path=str(p), old_str="notes", new_str="EVIL")


def _read_file():
    from prometheus.tools.builtin.file_read import FileReadTool, FileReadToolInput
    return FileReadTool(), lambda p: FileReadToolInput(path=str(p))


def _notebook_edit():
    from prometheus.tools.builtin.notebook_edit import NotebookEditInput, NotebookEditTool
    return NotebookEditTool(), lambda p: NotebookEditInput(path=str(p), cell_index=0, new_source="EVIL")


TOOLS = {"write_file": (_write_file, "notes.txt"), "edit_file": (_edit_file, "notes.txt"),
         "read_file": (_read_file, "notes.txt"), "notebook_edit": (_notebook_edit, "nb.ipynb")}


def _run(tool, args, ctx):
    return asyncio.run(tool.execute(args, ctx))


def _untouched(elsewhere: Path) -> None:
    assert (elsewhere / "notes.txt").read_text() == "PROTECTED\n"
    assert "PROTECTED" in (elsewhere / "nb.ipynb").read_text() and "EVIL" not in (elsewhere / "nb.ipynb").read_text()
    assert sorted(p.name for p in elsewhere.iterdir()) == ["nb.ipynb", "notes.txt"], "nothing new landed there"


# ── a link swapped in after the gate's decision ──────────────────────────────

@pytest.mark.parametrize("name", list(TOOLS))
def test_a_folder_swapped_for_a_link_after_the_check_is_refused(world, name):
    ws, elsewhere = world
    make, filename = TOOLS[name]
    tool, args = make()
    target = ws / "sub" / filename
    decided = target.resolve()                      # what the gate ruled on
    _swap(ws, elsewhere)                            # then the race
    result = _run(tool, args(target), _ctx(ws, decided))
    assert result.is_error, f"{name} followed the swapped link: {result.output[:200]}"
    assert "PROTECTED" not in result.output, "nothing was read from the other folder"
    _untouched(elsewhere)


@pytest.mark.parametrize("name", list(TOOLS))
def test_a_link_swapped_in_at_the_last_moment_is_not_followed(world, name, monkeypatch):
    """Between the tool's own resolution and its open: the walk without following links refuses it."""
    ws, elsewhere = world
    make, filename = TOOLS[name]
    tool, args = make()
    target = ws / "sub" / filename
    swapped = []

    def swap_once(path):
        if not swapped:
            swapped.append(path)
            _swap(ws, elsewhere)

    monkeypatch.setattr(safe_fs, "before_open", swap_once)
    result = _run(tool, args(target), _ctx(ws, target.resolve()))
    assert swapped, f"{name} never went through safe_fs"
    assert result.is_error, f"{name} followed the swapped link: {result.output[:200]}"
    assert "PROTECTED" not in result.output
    _untouched(elsewhere)


@pytest.mark.parametrize("name", list(TOOLS))
def test_without_a_race_the_tool_works_as_before(world, name):
    ws, _ = world
    make, filename = TOOLS[name]
    tool, args = make()
    target = ws / "sub" / filename
    result = _run(tool, args(target), _ctx(ws, target.resolve()))
    assert not result.is_error, result.output


def test_a_link_that_was_already_there_when_checked_still_works(world):
    """A workspace that links to another folder is fine: the gate ruled on the target, and the target is opened."""
    ws, elsewhere = world
    (ws / "linked").symlink_to(elsewhere)
    tool, args = _read_file()
    target = ws / "linked" / "notes.txt"
    result = _run(tool, args(target), _ctx(ws, target.resolve()))
    assert not result.is_error and "PROTECTED" in result.output, result.output


def test_writing_a_symlinked_file_writes_its_target_as_before(world):
    ws, elsewhere = world
    (ws / "alias.txt").symlink_to(elsewhere / "notes.txt")
    tool, args = _write_file()
    result = _run(tool, args(ws / "alias.txt"), _ctx(ws, (ws / "alias.txt").resolve()))
    assert not result.is_error, result.output
    assert (elsewhere / "notes.txt").read_text() == "EVIL\n"
    assert (ws / "alias.txt").is_symlink(), "the link itself is kept"


def test_a_write_keeps_the_files_mode_and_creates_missing_folders(world):
    ws, _ = world
    target = ws / "sub" / "notes.txt"
    target.chmod(0o640)
    tool, args = _write_file()
    assert not _run(tool, args(target), _ctx(ws, target.resolve())).is_error
    assert (target.stat().st_mode & 0o777) == 0o640
    deep = ws / "a" / "b" / "c.txt"
    assert not _run(tool, args(deep), _ctx(ws, deep.resolve())).is_error
    assert deep.read_text() == "EVIL\n"


# ── a hard link to a protected file ──────────────────────────────────────────

@pytest.fixture
def protected_key(world):
    ws, _ = world
    cfg = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    (cfg / "node").mkdir(parents=True)
    key = cfg / "node" / "node.key"
    key.write_text("PRIVATE KEY\n")
    alias = ws / "innocent.txt"
    os.link(key, alias)
    return ws, key, alias


def test_writing_a_hard_link_to_a_protected_file_never_writes_the_protected_file(protected_key):
    """A write replaces the NAME in the workspace with a new file; the protected file's inode is never opened for
    writing, so its other name keeps its bytes."""
    ws, key, alias = protected_key
    tool, args = _write_file()
    result = _run(tool, args(alias), _ctx(ws, alias.resolve()))
    assert not result.is_error, result.output
    assert key.read_text() == "PRIVATE KEY\n"
    assert alias.read_text() == "EVIL\n" and os.stat(alias).st_ino != os.stat(key).st_ino


@pytest.mark.parametrize("name", ["read_file", "edit_file"])
def test_a_hard_link_to_a_protected_file_is_refused(protected_key, name):
    ws, key, alias = protected_key
    make, _ = TOOLS[name]
    tool, args = make()
    if name == "edit_file":
        from prometheus.tools.builtin.file_edit import FileEditToolInput
        a = FileEditToolInput(path=str(alias), old_str="PRIVATE", new_str="EVIL")
    else:
        a = args(alias)
    result = _run(tool, a, _ctx(ws, alias.resolve()))
    assert result.is_error, f"{name} used the hard link: {result.output[:200]}"
    assert "PRIVATE KEY" not in result.output
    assert key.read_text() == "PRIVATE KEY\n"


def test_an_ordinary_multi_link_file_is_still_readable(world):
    """uv installs a venv by hard-linking its cache: a second link alone is not a reason to refuse."""
    ws, _ = world
    os.link(ws / "sub" / "notes.txt", ws / "twin.txt")
    tool, args = _read_file()
    result = _run(tool, args(ws / "twin.txt"), _ctx(ws, (ws / "twin.txt").resolve()))
    assert not result.is_error and "ws notes" in result.output, result.output


# ── through the loop ─────────────────────────────────────────────────────────

async def test_the_loop_hands_the_tool_the_path_the_gate_ruled_on(world, monkeypatch):
    """The real path: a scripted model asks write_file for ws/sub/notes.txt, the loop asks the gate, records its
    ruling, and the folder is swapped for a link before the tool opens it. Nothing lands in the other folder."""
    from prometheus.__main__ import create_security_gate
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin.file_write import FileWriteTool

    ws, elsewhere = world
    target = ws / "sub" / "notes.txt"
    expected = str(target.resolve())
    seen: dict = {}

    class _Recording(FileWriteTool):
        async def execute(self, arguments, context):
            seen["decided_path"] = context.metadata.get("decided_path")
            _swap(ws, elsewhere)                    # the race, after the gate and before the tool opens it
            return await super().execute(arguments, context)

    class _Scripted(ModelProvider):
        def __init__(self) -> None:
            self.n = 0

        async def stream_message(self, request):  # noqa: ANN001
            self.n += 1
            content = ([ToolUseBlock(id="t1", name="write_file", input={"path": str(target), "content": "EVIL"})]
                       if self.n == 1 else [TextBlock(text="done")])
            yield ApiMessageCompleteEvent(message=ConversationMessage(role="assistant", content=content),
                                          usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")

    tools = ToolRegistry()
    tools.register(_Recording())
    gate = create_security_gate({"workspace_root": str(ws), "permission_mode": "autonomous"})
    ctx = LoopContext(provider=_Scripted(), model="stub", system_prompt="S", max_tokens=64,
                      tool_registry=tools, permission_checker=gate, cwd=ws)
    async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="cli"):
        pass
    assert seen["decided_path"] == expected, seen
    _untouched(elsewhere)


# ── the other file tools ─────────────────────────────────────────────────────

def test_grep_does_not_read_a_hard_link_to_a_protected_file(protected_key):
    from prometheus.tools.builtin.grep import GrepTool, GrepToolInput

    ws, _key, _alias = protected_key
    result = _run(GrepTool(), GrepToolInput(pattern="PRIVATE", root=str(ws), file_glob="**/*"), _ctx(ws, None))
    assert "PRIVATE KEY" not in result.output, result.output


def test_tts_saves_through_no_link(world, monkeypatch):
    from prometheus.tools.base import ToolResult
    from prometheus.tools.builtin import tts as T

    ws, elsewhere = world

    async def fake_synthesize(self, arguments, out_path):
        Path(out_path).write_bytes(b"RIFF-EVIL")
        return ToolResult(output=f"Audio saved to {out_path}")

    monkeypatch.setattr(T.TTSTool, "_synthesize", fake_synthesize)
    swapped = []

    def swap_once(path):
        if not swapped:
            swapped.append(path)
            _swap(ws, elsewhere)

    monkeypatch.setattr(safe_fs, "before_open", swap_once)
    target = ws / "sub" / "speech.wav"
    result = _run(T.TTSTool(), T.TTSInput(text="hi", output_path=str(target)), _ctx(ws, target.resolve()))
    assert swapped and result.is_error, result.output
    _untouched(elsewhere)


def test_tts_saves_where_asked_without_a_race(world, monkeypatch):
    from prometheus.tools.base import ToolResult
    from prometheus.tools.builtin import tts as T

    ws, _ = world

    async def fake_synthesize(self, arguments, out_path):
        Path(out_path).write_bytes(b"RIFF")
        return ToolResult(output=f"Audio saved to {out_path}")

    monkeypatch.setattr(T.TTSTool, "_synthesize", fake_synthesize)
    target = ws / "sub" / "speech.wav"
    result = _run(T.TTSTool(), T.TTSInput(text="hi", output_path=str(target)), _ctx(ws, target.resolve()))
    assert not result.is_error and target.read_bytes() == b"RIFF", result.output


def test_image_generate_saves_through_no_link(world, monkeypatch):
    from prometheus.tools.builtin import image_generate as IG

    ws, elsewhere = world
    monkeypatch.setattr(IG, "_allowed_image_roots", lambda: [ws])
    swapped = []

    def swap_once(path):
        if not swapped:
            swapped.append(path)
            _swap(ws, elsewhere)

    monkeypatch.setattr(safe_fs, "before_open", swap_once)
    with pytest.raises(safe_fs.PathChanged):
        IG._save_image_bytes(b"PNG-EVIL", ext="png", override_path=str(ws / "sub" / "pic.png"))
    _untouched(elsewhere)


def test_download_file_refuses_a_destination_that_moved_after_the_check(world):
    from prometheus.tools.builtin.download_file import DownloadFileInput, DownloadFileTool

    ws, elsewhere = world
    target = ws / "sub" / "file.bin"
    decided = target.resolve()
    _swap(ws, elsewhere)
    args = DownloadFileInput(url="https://example.com/file.bin", destination=str(target))
    result = _run(DownloadFileTool(), args, _ctx(ws, decided))
    assert result.is_error and "changed after the check" in result.output, result.output
    _untouched(elsewhere)


# ── the shell cannot make the hard link (macOS) ──────────────────────────────

@pytest.mark.skipif(not (os.uname().sysname == "Darwin" and Path("/usr/bin/sandbox-exec").exists()),
                    reason="needs macOS with /usr/bin/sandbox-exec")
@pytest.mark.parametrize("rel", [".prometheus/node/node.key", ".prometheus/data/devices.db", ".prometheus/data/lcm.db",
                                 ".ssh/id_x", ".prometheus/prometheus.yaml"])
def test_a_confined_shell_cannot_hard_link_a_protected_file(tmp_path, rel, monkeypatch):
    """Measured: the Seatbelt profile's write rules refuse link(2) on the source, so a model's shell on a Mac cannot
    make the second name the file tools now refuse anyway. The AppArmor profile's `l` does the same on Linux."""
    import subprocess

    from prometheus.permissions import confinement as C

    for name in ("PROMETHEUS_CONFIG_DIR", "PROMETHEUS_DATA_DIR", "PROMETHEUS_NODE_DIR"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path.resolve()
    src = home / rel
    src.parent.mkdir(parents=True)
    src.write_text("x\n")
    (home / "ws").mkdir()
    dst = home / "ws" / "alias"
    rc = subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/ln", str(src), str(dst)],
                        capture_output=True, text=True, timeout=30).returncode
    assert rc != 0 and not dst.exists(), rel


# ── found by the independent review ──────────────────────────────────────────

def test_image_generate_resolves_once_so_a_swap_during_generation_moves_nothing(world, monkeypatch):
    """Generation takes seconds. The images folder swapped for a link to somewhere else DURING it used to move the
    write, because the sink resolved the path and its root again. The path is resolved and checked once, at the
    start, and written exactly."""
    from prometheus.tools.builtin import image_generate as IG

    ws, elsewhere = world
    (ws / "images").mkdir()
    target = ws / "images" / "notes.txt"

    async def fake_backend(name, cfg):
        return "pollinations"

    async def generate_while_swapping(arguments):
        shutil.rmtree(ws / "images")
        os.symlink(elsewhere, ws / "images")          # the race, mid-generation
        try:
            path = IG._save_image_bytes(b"PNG-EVIL", ext="png", override_path=arguments.output_path)
        except ValueError as exc:
            return IG.ToolResult(output=str(exc), is_error=True)
        return IG.ToolResult(output=f"saved {path}")

    monkeypatch.setattr(IG, "_resolve_backend", fake_backend)
    monkeypatch.setattr(IG, "_generate_via_pollinations", generate_while_swapping)
    ctx = ToolExecutionContext(cwd=ws, metadata={"workspace_roots": [str(ws)], "decided_path": str(target.resolve())})
    result = _run(IG.ImageGenerateTool(), IG.ImageGenerateInput(prompt="x", output_path=str(target)), ctx)
    assert result.is_error, result.output
    _untouched(elsewhere)


def test_the_hard_link_check_covers_a_moved_data_directory_and_configured_denied_paths(tmp_path, monkeypatch):
    """The read floor is the gate's own list: a data directory PROMETHEUS_DATA_DIR moved, and a configured
    denied_paths entry, are covered as the gate covers them."""
    from prometheus.__main__ import create_security_gate
    from prometheus.tools.builtin.file_read import FileReadTool, FileReadToolInput

    root = tmp_path.resolve()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(root / "cfg"))
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(root / "moved-data"))
    monkeypatch.setattr(safe_fs, "_identity_cache", None)
    (root / "moved-data").mkdir()
    (root / "secrets").mkdir()
    (root / "ws").mkdir()
    registry = root / "moved-data" / "devices.db"
    registry.write_text("REGISTRY\n")
    configured = root / "secrets" / "token.txt"
    configured.write_text("CONFIGURED SECRET\n")
    create_security_gate({"denied_paths": [str(root / "secrets")]})
    for src, word in ((registry, "REGISTRY"), (configured, "CONFIGURED SECRET")):
        alias = root / "ws" / f"alias-{src.name}"
        os.link(src, alias)
        result = _run(FileReadTool(), FileReadToolInput(path=str(alias)), _ctx(root / "ws", alias.resolve()))
        assert result.is_error and word not in result.output, (src, result.output)


def test_a_project_file_sharing_an_inode_with_the_install_is_still_readable(world, monkeypatch):
    """uv hard-links a venv from its cache, and the install is floored for WRITES only: reading a project file that
    shares an inode with an installed file is ordinary and stays allowed."""
    import sys

    from prometheus.tools.builtin.file_read import FileReadTool, FileReadToolInput

    ws, _ = world
    fake_prefix = ws.parent / "fake-venv"
    (fake_prefix / "lib").mkdir(parents=True)
    installed = fake_prefix / "lib" / "module.py"
    installed.write_text("installed = True\n")
    monkeypatch.setattr(sys, "prefix", str(fake_prefix))
    project_copy = ws / "module.py"
    os.link(installed, project_copy)
    result = _run(FileReadTool(), FileReadToolInput(path=str(project_copy)), _ctx(ws, project_copy.resolve()))
    assert not result.is_error and "installed = True" in result.output, result.output


def test_a_file_the_user_may_not_write_is_not_replaced(world):
    ws, _ = world
    target = ws / "sub" / "notes.txt"
    target.chmod(0o444)
    tool, args = _write_file()
    result = _run(tool, args(target), _ctx(ws, target.resolve()))
    assert result.is_error and target.read_text() == "ws notes\n", result.output


def test_a_write_leaves_the_umask_alone_and_no_temp_file_behind(world, monkeypatch):
    ws, _ = world
    seen: list[int] = []
    real_umask = os.umask
    monkeypatch.setattr(os, "umask", lambda m: (seen.append(m), real_umask(m))[1])
    tool, args = _write_file()
    target = ws / "sub" / "fresh.txt"
    assert not _run(tool, args(target), _ctx(ws, target.resolve())).is_error
    assert seen == [], "the process umask was changed during a write"
    assert not [p for p in (ws / "sub").iterdir() if p.name.endswith(".tmp")]


def test_a_search_only_folder_along_the_path_still_works(world):
    ws, _ = world
    locked = ws / "sub" / "x-only"
    locked.mkdir()
    (locked / "f.txt").write_text("searchable\n")
    locked.chmod(0o311)
    try:
        tool, args = _read_file()
        result = _run(tool, args(locked / "f.txt"), _ctx(ws, (locked / "f.txt").resolve()))
        assert not result.is_error and "searchable" in result.output, result.output
    finally:
        locked.chmod(0o755)


def test_a_file_where_a_folder_should_be_is_not_called_a_link(world):
    ws, _ = world
    tool, args = _read_file()
    target = ws / "sub" / "notes.txt" / "inner.txt"
    result = _run(tool, args(target), _ctx(ws, target.resolve()))
    assert result.is_error and "link" not in result.output, result.output


def test_the_user_files_fallback_works_through_a_linked_config_dir(tmp_path, monkeypatch):
    from prometheus.tools.builtin.file_read import FileReadTool, FileReadToolInput

    real_cfg = tmp_path.resolve() / "real-cfg"
    (real_cfg / "files").mkdir(parents=True)
    (real_cfg / "files" / "brief.md").write_text("the brief\n")
    linked = tmp_path.resolve() / "linked-cfg"
    linked.symlink_to(real_cfg)
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(linked))
    ws = tmp_path.resolve() / "ws"
    ws.mkdir()
    result = _run(FileReadTool(), FileReadToolInput(path="brief.md"), ToolExecutionContext(cwd=ws, metadata={}))
    assert not result.is_error and "the brief" in result.output, result.output


@pytest.mark.parametrize("which", ["view", "str_replace", "create"])
def test_the_coding_run_tools_open_through_no_link(world, monkeypatch, which):
    """A coding run's tools resolve inside the repo jail, then opened by name: a command still running from code_run
    could swap a folder for a link between the two."""
    from prometheus.coding import tools as CT
    from prometheus.coding.sandbox import ProcessSandbox

    ws, elsewhere = world
    sandbox = ProcessSandbox(root=ws)
    swapped = []

    def swap_once(path):
        if not swapped:
            swapped.append(path)
            _swap(ws, elsewhere)

    monkeypatch.setattr(safe_fs, "before_open", swap_once)
    if which == "view":
        result = _run(CT.CodeViewTool(sandbox), CT.CodeViewArgs(path="sub/notes.txt"), _ctx(ws, None))
    elif which == "str_replace":
        result = _run(CT.CodeStrReplaceTool(sandbox),
                      CT.CodeStrReplaceArgs(path="sub/notes.txt", old_str="notes", new_str="EVIL"), _ctx(ws, None))
    else:
        result = _run(CT.CodeCreateTool(sandbox), CT.CodeCreateArgs(path="sub/new.txt", content="EVIL"), _ctx(ws, None))
    assert swapped and result.is_error, result.output
    assert "PROTECTED" not in result.output
    _untouched(elsewhere)
