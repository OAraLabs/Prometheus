"""The lsp tool never answers "nothing found" when it could not check.

Before this, every path where no server could answer (none configured, binary
not installed, a failed start, a failed request, diagnostics never published)
came back as an empty result, which the tool formatted as "none found" or
"No diagnostics": the same words a server uses for a clean answer. A failed
start also stayed failed until the daemon restarted, and nothing outside the
log recorded it.

Each test here pins one of those paths to an error that says why.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import pkgutil
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.lsp.client import HoverInfo, Location, LSPClient, LSPError
from prometheus.lsp.orchestrator import LSPOrchestrator, LSPUnavailable
from prometheus.tools.base import BaseTool, ToolExecutionContext
from prometheus.tools.builtin.lsp import LSPTool, LSPToolInput, set_lsp_orchestrator

# A binary no machine has, so the missing-server path is real on every runner.
_NO_SUCH_BINARY = "prometheus-test-no-such-language-server"

# The answers a server gives when it looked and found nothing. None of them
# may appear when nothing looked.
_EMPTY_ANSWERS = (
    "none found", "No diagnostics", "No hover", "No symbols", "no changes",
    "proposed no changes", "No information",
)


@pytest.fixture(autouse=True)
def _no_global_orchestrator():
    set_lsp_orchestrator(None)
    yield
    set_lsp_orchestrator(None)


@pytest.fixture
def py_project(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    src = tmp_path / "main.py"
    src.write_text("def foo():\n    return 42\n\nfoo()\n")
    return tmp_path, src


def _missing_server_orch() -> LSPOrchestrator:
    return LSPOrchestrator(custom_servers={
        "python": {"command": [_NO_SUCH_BINARY, "--stdio"], "install_command": ["pip", "install", "pyright"]},
    })


async def _run(orch, ctx, **args):
    ctx.metadata["lsp_orchestrator"] = orch
    return await LSPTool().execute(LSPToolInput(**args), ctx)


# ------------------------------------------------------------------
# No server installed: every action is an error that says how to fix it
# ------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("action,extra", [
    ("definition", {"symbol": "foo"}),
    ("references", {"symbol": "foo"}),
    ("hover", {"symbol": "foo"}),
    ("diagnostics", {}),
    ("symbols", {}),
    ("rename", {"symbol": "foo", "new_name": "bar"}),
    ("context", {"symbol": "foo"}),
])
async def test_missing_server_is_an_error_for_every_action(py_project, action, extra):
    root, src = py_project
    result = await _run(_missing_server_orch(), ToolExecutionContext(cwd=root), action=action, file=str(src), **extra)

    assert result.is_error, result.output
    assert "not installed" in result.output
    assert _NO_SUCH_BINARY in result.output
    assert "pip install pyright" in result.output
    for empty in _EMPTY_ANSWERS:
        assert empty not in result.output


@pytest.mark.asyncio
async def test_unsupported_file_type_is_an_error(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("foo\n")
    result = await _run(LSPOrchestrator(), ToolExecutionContext(cwd=tmp_path), action="definition", file=str(notes), symbol="foo")

    assert result.is_error, result.output
    assert "no language server is configured for `.txt` files" in result.output


# ------------------------------------------------------------------
# A failed request is an error, not an empty answer
# ------------------------------------------------------------------

def _live_client(orch, root, **methods):
    client = MagicMock(spec=LSPClient)
    client.server_def = MagicMock()
    client.server_def.language_id = "python"
    client.is_alive = True
    for name, value in methods.items():
        setattr(client, name, value)
    orch._clients[f"python:{root}"] = client
    return client


@pytest.mark.asyncio
async def test_failed_request_is_an_error(py_project):
    root, src = py_project
    orch = LSPOrchestrator()
    _live_client(orch, root, get_definition=AsyncMock(side_effect=LSPError({"message": "Request timed out: textDocument/definition"})))

    result = await _run(orch, ToolExecutionContext(cwd=root), action="definition", file=str(src), symbol="foo")

    assert result.is_error, result.output
    assert "failed the definition request: Request timed out" in result.output
    assert "none found" not in result.output


@pytest.mark.asyncio
async def test_context_says_which_part_could_not_be_checked(py_project):
    root, src = py_project
    orch = LSPOrchestrator()
    _live_client(
        orch, root,
        get_definition=AsyncMock(return_value=[Location(path=str(src), line=1, col=5)]),
        get_references=AsyncMock(side_effect=LSPError({"message": "refs boom"})),
        get_hover=AsyncMock(return_value=HoverInfo(contents="def foo() -> int")),
    )

    result = await _run(orch, ToolExecutionContext(cwd=root), action="context", file=str(src), symbol="foo")

    assert "References: couldn't check (refs boom)" in result.output
    assert "References: none found" not in result.output
    assert "Defined:" in result.output and "def foo() -> int" in result.output


@pytest.mark.asyncio
async def test_unreadable_file_is_not_reported_as_symbol_not_found(py_project, monkeypatch):
    root, src = py_project
    real_read_text = Path.read_text

    def deny(self, *a, **kw):
        if self.resolve() == src.resolve():
            raise PermissionError("denied")
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", deny)
    result = await _run(LSPOrchestrator(), ToolExecutionContext(cwd=root), action="definition", file=str(src), symbol="foo")

    assert result.is_error
    assert "could not read" in result.output
    assert "not found" not in result.output


# ------------------------------------------------------------------
# Diagnostics: "clean" only when the server said so
# ------------------------------------------------------------------

def _bare_client(root) -> LSPClient:
    sdef = MagicMock()
    sdef.language_id = "python"
    return LSPClient(sdef, root)


@pytest.mark.asyncio
async def test_client_distinguishes_never_published_from_clean(py_project):
    root, src = py_project
    client = _bare_client(root)

    assert await client.get_diagnostics(str(src)) is None  # nothing published

    client._handle_notification("textDocument/publishDiagnostics", {"uri": f"file://{src.resolve()}", "diagnostics": []})
    assert await client.get_diagnostics(str(src)) == []  # published clean


@pytest.mark.asyncio
async def test_percent_encoded_uri_reaches_its_file(tmp_path):
    src = tmp_path / "with space.py"
    src.write_text("x = (\n")
    client = _bare_client(tmp_path)
    uri = "file://" + str(src.resolve()).replace(" ", "%20")
    client._handle_notification("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": [
        {"range": {"start": {"line": 0, "character": 4}}, "severity": 1, "message": "'(' was never closed"},
    ]})

    diags = await client.get_diagnostics(str(src))
    assert diags is not None and len(diags) == 1


@pytest.mark.asyncio
async def test_diagnostics_wait_for_a_fresh_publish(py_project):
    root, src = py_project
    client = _bare_client(root)
    client._process = MagicMock(returncode=None)
    client._initialized = True
    path = str(src.resolve())
    # A stale publish from before the call must not count as the answer.
    client._handle_notification("textDocument/publishDiagnostics", {"uri": f"file://{path}", "diagnostics": [
        {"range": {"start": {"line": 0, "character": 0}}, "severity": 1, "message": "stale"},
    ]})

    async def reopen(_path):
        import asyncio
        asyncio.get_running_loop().call_later(
            0.1, client._handle_notification, "textDocument/publishDiagnostics",
            {"uri": f"file://{path}", "diagnostics": []},
        )

    client.did_open = reopen
    assert await client.get_diagnostics(path, wait_s=2) == []

    async def silent(_path):
        return None

    client.did_open = silent
    assert await client.get_diagnostics(path, wait_s=0.2) is None


@pytest.mark.asyncio
async def test_tool_reports_unpublished_diagnostics_as_unchecked(py_project):
    root, src = py_project
    orch = LSPOrchestrator()
    _live_client(orch, root, get_diagnostics=AsyncMock(return_value=None))

    result = await _run(orch, ToolExecutionContext(cwd=root), action="diagnostics", file=str(src))

    assert result.is_error, result.output
    assert "published no diagnostics" in result.output
    assert "No diagnostics" not in result.output


# ------------------------------------------------------------------
# A failed start is retried: when the binary appears, or after a backoff.
# A real subprocess on a real PATH, because that is where the daemon's bug
# lived: an install the daemon could not see until a restart.
# ------------------------------------------------------------------

# A language server that answers initialize/shutdown and nothing else. With a
# marker path, its first start exits at once (a failed start).
_FAKE_SERVER = r"""
import json, os, sys
marker, starts = __MARKER__, __STARTS__
with open(starts, "a") as f:
    f.write("start\n")
if marker and not os.path.exists(marker):
    open(marker, "w").close()
    sys.exit(1)

def read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            sys.exit(0)
        line = line.strip()
        if not line:
            break
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers["content-length"])))

def send(msg):
    body = json.dumps(msg).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
    sys.stdout.buffer.flush()

while True:
    msg = read()
    if msg.get("method") == "exit":
        sys.exit(0)
    if "id" in msg:
        result = {"capabilities": {}} if msg.get("method") == "initialize" else None
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
"""


def _install_fake_server(bin_dir: Path, name: str, *, fail_first_start: Path | None = None) -> Path:
    """Put an executable language server called *name* in *bin_dir*. Returns its start log."""
    starts = bin_dir / f"{name}.starts"
    script = bin_dir / name
    body = _FAKE_SERVER.replace("__MARKER__", repr(str(fail_first_start) if fail_first_start else ""))
    script.write_text(f"#!{sys.executable}\n" + body.replace("__STARTS__", repr(str(starts))))
    script.chmod(0o755)
    return starts


@pytest.fixture
def path_bin(tmp_path_factory, monkeypatch):
    bin_dir = tmp_path_factory.mktemp("bin")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return bin_dir


@pytest.mark.asyncio
async def test_missing_binary_is_retried_when_it_appears(py_project, path_bin):
    root, src = py_project
    name = "prometheus-test-appearing-lsp"
    orch = LSPOrchestrator(custom_servers={"python": {"command": [name]}})
    try:
        assert await orch.ensure_server(src) is None  # not installed yet

        _install_fake_server(path_bin, name)  # e.g. `npm install -g ...` while the daemon runs
        client = await orch.ensure_server(src)
        assert client is not None and client.is_alive, "an install was not picked up without a restart"
    finally:
        await orch.shutdown_all()


@pytest.mark.asyncio
async def test_failed_start_is_retried_after_backoff(py_project, path_bin, monkeypatch):
    root, src = py_project
    import prometheus.lsp.orchestrator as orch_mod

    # raising=False: on a tree without a backoff this sets nothing, and the
    # test then fails on behaviour rather than on the patch.
    monkeypatch.setattr(orch_mod, "_RETRY_FIRST_S", 0.5, raising=False)
    name = "prometheus-test-flaky-lsp"
    starts = _install_fake_server(path_bin, name, fail_first_start=path_bin / "failed-once")
    orch = LSPOrchestrator(custom_servers={"python": {"command": [name]}})
    try:
        assert await orch.ensure_server(src) is None  # first start fails

        # Inside the backoff: not retried, and the tool says why.
        result = await _run(orch, ToolExecutionContext(cwd=root), action="definition", file=str(src), symbol="foo")
        assert result.is_error, result.output
        assert "failed to start" in result.output and "retried in" in result.output
        assert starts.read_text().count("start") == 1

        await asyncio.sleep(0.6)
        client = await orch.ensure_server(src)
        assert client is not None and client.is_alive, "a failed start was never retried"
        assert starts.read_text().count("start") == 2
    finally:
        await orch.shutdown_all()


# ------------------------------------------------------------------
# rename computes edits and applies none; it must not say it renamed
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rename_says_nothing_was_written(py_project):
    root, src = py_project
    orch = MagicMock()
    orch.rename = AsyncMock(return_value={str(src): [{"start_line": 1, "end_line": 1, "newText": "bar"}]})

    result = await _run(orch, ToolExecutionContext(cwd=root), action="rename", file=str(src), symbol="foo", new_name="bar")

    assert "NOT APPLIED" in result.output and "nothing was written" in result.output
    assert "Renamed to" not in result.output


# ------------------------------------------------------------------
# oara doctor reports each language server, against the daemon's PATH
# ------------------------------------------------------------------

def test_doctor_reports_missing_and_present_servers(tmp_path, monkeypatch):
    from prometheus.cli import doctor

    binary = tmp_path / "typescript-language-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(doctor, "_daemon_path", lambda: (str(tmp_path), "the test unit's PATH"))

    rows = {r.name: r for r in doctor.check_language_servers({"lsp": {"enabled": True}})}

    assert rows["LSP typescript"].status == "ok"
    assert str(binary) in rows["LSP typescript"].message
    py = rows["LSP python"]
    assert py.status == "warning"
    assert "pyright-langserver is not on the test unit's PATH" in py.message
    assert "pip install pyright" in py.fix
    assert {"LSP go", "LSP rust", "LSP c"} <= rows.keys()


def test_doctor_says_when_lsp_is_off():
    from prometheus.cli import doctor

    rows = doctor.check_language_servers({})
    assert len(rows) == 1 and rows[0].status == "info" and "disabled" in rows[0].message


# ------------------------------------------------------------------
# Built-in tool examples use the tool's real argument names
# ------------------------------------------------------------------

def _builtin_tools_with_examples() -> dict[str, type[BaseTool]]:
    import prometheus.tools.builtin as pkg

    found: dict[str, type[BaseTool]] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        except Exception:
            continue  # optional dependency missing; the count guard below catches a vacuous run
        for obj in vars(module).values():
            if (isinstance(obj, type) and issubclass(obj, BaseTool) and obj is not BaseTool
                    and getattr(obj, "example_call", None) is not None and getattr(obj, "input_model", None)):
                found[obj.name] = obj
    return found


def test_every_builtin_example_uses_real_argument_names():
    tools = _builtin_tools_with_examples()
    assert {"edit_file", "read_file", "write_file"} <= tools.keys(), sorted(tools)

    wrong = {}
    for name, tool in tools.items():
        fields = set(tool.input_model.model_fields)
        unknown = set(tool.example_call) - fields
        if unknown:
            wrong[name] = sorted(unknown)
            continue
        tool.input_model.model_validate(tool.example_call)
    assert not wrong, f"examples use argument names the tool does not accept: {wrong}"
