"""Item W — a per-session working directory, and the gate follows it.

Contract under test, end to end where it matters:
- the store binds a path to a session and purge forgets it;
- one validator refuses relative / missing / root / denied paths on every surface;
- REST and the shared /workspace command write through that validator;
- run_loop resolves the workspace per run: tools execute in it, relative
  paths resolve there, the SecurityGate's write boundary IS it (a write inside
  the session workspace is allowed, the same write is APPROVE without one),
  and the boot "# Project Instructions" section is swapped for the workspace's;
- a session without a workspace is unchanged.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from prometheus.context.workspace import validate_workspace_path
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.path_schema import PATH_FIELD
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


# --------------------------------------------------------------------------- #
# validator
# --------------------------------------------------------------------------- #

class TestValidator:
    def test_accepts_an_existing_absolute_directory(self, tmp_path) -> None:
        resolved, why = validate_workspace_path(str(tmp_path), {})
        assert why is None and resolved == tmp_path.resolve()

    def test_expands_home(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "proj").mkdir()
        resolved, why = validate_workspace_path("~/proj", {})
        assert why is None and resolved == (tmp_path / "proj").resolve()

    @pytest.mark.parametrize("raw,fragment", [
        ("relative/dir", "absolute"),
        ("/definitely/not/here/xyz", "not an existing directory"),
        ("/", "filesystem root"),
        ("", "required"),
    ])
    def test_refusals_name_the_reason(self, raw, fragment) -> None:
        resolved, why = validate_workspace_path(raw, {})
        assert resolved is None and fragment in why

    def test_denied_paths_apply(self, tmp_path) -> None:
        secret = tmp_path / "secret"
        secret.mkdir()
        resolved, why = validate_workspace_path(str(secret), {"denied_paths": [str(tmp_path)]})
        assert resolved is None and "denied path" in why


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #

class TestStore:
    def _store(self, tmp_path):
        from prometheus.memory.lcm_conversation_store import LCMConversationStore
        return LCMConversationStore(tmp_path / "lcm.db")

    def test_bind_read_clear(self, tmp_path) -> None:
        store = self._store(tmp_path)
        assert store.get_session_workspace("s1") is None
        store.set_session_workspace("s1", "/tmp/w", set_by="rest")
        assert store.get_session_workspace("s1") == "/tmp/w"
        store.set_session_workspace("s1", "")
        assert store.get_session_workspace("s1") is None

    def test_purge_forgets_it(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.set_session_workspace("s1", "/tmp/w")
        counts = store.purge_session("s1")
        assert counts.get("session_workspaces") == 1
        assert store.get_session_workspace("s1") is None


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #

class TestRoutes:
    def _client(self, tmp_path, security=None):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from prometheus.memory.lcm_conversation_store import LCMConversationStore
        from prometheus.web.server import create_app

        app = create_app({"security": security or {}})
        app.state.lcm_engine = SimpleNamespace(conversation_store=LCMConversationStore(tmp_path / "lcm.db"))
        return TestClient(app)

    def test_default_is_the_daemon(self, tmp_path) -> None:
        body = self._client(tmp_path).get("/api/sessions/desktop:a/workspace").json()
        assert body["workspace"] is None and body["source"] == "daemon"
        assert body["daemon_workspace_roots"]

    def test_set_get_delete(self, tmp_path) -> None:
        client = self._client(tmp_path)
        ws = tmp_path / "repo"; ws.mkdir()
        resp = client.put("/api/sessions/desktop:a/workspace", json={"path": str(ws)})
        assert resp.status_code == 200, resp.text
        assert resp.json()["workspace"] == str(ws.resolve())
        assert client.get("/api/sessions/desktop:a/workspace").json()["source"] == "session"
        assert client.delete("/api/sessions/desktop:a/workspace").status_code == 200
        assert client.get("/api/sessions/desktop:a/workspace").json()["workspace"] is None

    @pytest.mark.parametrize("path", ["relative", "/nope/nope/nope", "/"])
    def test_refusals_are_400_with_a_reason(self, tmp_path, path) -> None:
        resp = self._client(tmp_path).put("/api/sessions/desktop:a/workspace", json={"path": path})
        assert resp.status_code == 400 and resp.json()["error"]

    def test_denied_path_is_refused(self, tmp_path) -> None:
        secret = tmp_path / "secret"; secret.mkdir()
        client = self._client(tmp_path, security={"denied_paths": [str(tmp_path)]})
        resp = client.put("/api/sessions/desktop:a/workspace", json={"path": str(secret)})
        assert resp.status_code == 400 and "denied" in resp.json()["error"]

    def test_no_store_is_503(self) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from prometheus.web.server import create_app
        client = TestClient(create_app({}))
        assert client.put("/api/sessions/x/workspace", json={"path": "/tmp"}).status_code == 503


# --------------------------------------------------------------------------- #
# the shared command
# --------------------------------------------------------------------------- #

class TestCommand:
    def _mgr(self, tmp_path):
        from prometheus.memory.lcm_conversation_store import LCMConversationStore
        return SimpleNamespace(lcm_engine=SimpleNamespace(conversation_store=LCMConversationStore(tmp_path / "lcm.db")))

    def test_show_set_refuse_clear(self, tmp_path) -> None:
        from prometheus.gateway.commands import cmd_workspace
        mgr = self._mgr(tmp_path)
        ws = tmp_path / "repo"; ws.mkdir()
        assert "No workspace bound" in cmd_workspace("telegram:1", "", session_manager=mgr, security_cfg={})
        assert "takes effect on the next message" in cmd_workspace("telegram:1", str(ws), session_manager=mgr, security_cfg={}, set_by="telegram")
        assert mgr.lcm_engine.conversation_store.get_session_workspace("telegram:1") == str(ws.resolve())
        assert str(ws.resolve()) in cmd_workspace("telegram:1", "", session_manager=mgr, security_cfg={})
        assert cmd_workspace("telegram:1", "relative/x", session_manager=mgr, security_cfg={}).startswith("Refused:")
        assert "cleared" in cmd_workspace("telegram:1", "clear", session_manager=mgr, security_cfg={})
        assert mgr.lcm_engine.conversation_store.get_session_workspace("telegram:1") is None

    def test_no_store_says_so(self) -> None:
        from prometheus.gateway.commands import cmd_workspace
        assert "unavailable" in cmd_workspace("s", "/tmp", session_manager=SimpleNamespace())

    def test_web_dispatch_reaches_it(self) -> None:
        from prometheus.gateway import commands as C
        assert C._SESSION_COMMANDS["workspace"] is C._sc_workspace


# --------------------------------------------------------------------------- #
# the loop: cwd, gate, prompt
# --------------------------------------------------------------------------- #

class _WriteInput(BaseModel):
    path: str = Field(json_schema_extra=PATH_FIELD)
    content: str = ""


class _ProbeWriteTool(BaseTool):
    """A write_file-shaped tool that records where it ran and what roots it saw."""
    input_model = _WriteInput

    def __init__(self) -> None:
        self.name = "write_file"
        self.description = "probe"
        self.seen: list[tuple[Path, object]] = []

    def is_read_only(self, arguments: BaseModel) -> bool:
        return False

    async def execute(self, arguments, context):  # noqa: ANN001
        self.seen.append((context.cwd, context.metadata.get("workspace_roots")))
        return ToolResult(output="written")


class _Scripted(ModelProvider):
    def __init__(self, tool_input: dict | None = None) -> None:
        self.tool_input = tool_input
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if self.tool_input is not None and len(self.requests) == 1:
            content = [ToolUseBlock(id="t1", name="write_file", input=self.tool_input)]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


def _drain(ctx, session_id="desktop:s1"):
    async def go():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=session_id):
            pass
    asyncio.run(go())


class TestLoop:
    def _ctx(self, provider, tool, gate, *, boot_cwd: Path, workspace: Path | None, boot_project: str | None = None, builder=None):
        registry = ToolRegistry(); registry.register(tool)
        return LoopContext(
            provider=provider, model="stub", system_prompt="BASE" + (f"\n\n{boot_project}" if boot_project else ""),
            max_tokens=64, tool_registry=registry, permission_checker=gate, cwd=boot_cwd,
            workspace_resolver=(lambda sid: str(workspace)) if workspace else None,
            boot_project_prompt=boot_project, project_prompt_builder=builder,
        )

    def test_tools_run_in_the_session_workspace_with_its_roots(self, tmp_path) -> None:
        boot = tmp_path / "boot"; boot.mkdir()
        ws = tmp_path / "ws"; ws.mkdir()
        tool = _ProbeWriteTool()
        gate = SecurityGate(workspace_root=str(boot))
        # Absolute on purpose: a RELATIVE write path prompts by design (#345),
        # so it would never reach the tool and prove nothing about cwd.
        _drain(self._ctx(_Scripted({"path": str(ws / "notes.md"), "content": "x"}), tool, gate, boot_cwd=boot, workspace=ws))
        assert tool.seen == [(ws.resolve(), (ws.resolve(),))]

    def test_without_a_workspace_nothing_changes(self, tmp_path) -> None:
        boot = tmp_path / "boot"; boot.mkdir()
        tool = _ProbeWriteTool()
        _drain(self._ctx(_Scripted({"path": str(boot / "notes.md"), "content": "x"}), tool, SecurityGate(workspace_root=str(boot)), boot_cwd=boot, workspace=None))
        assert tool.seen == [(boot, None)]

    def test_the_gate_follows_the_session(self, tmp_path) -> None:
        """The SAME write, under the gate's configured root, is APPROVE (outside
        the boundary) — and ALLOW once the session's workspace is that
        directory. That inversion is the whole decision."""
        boot = tmp_path / "boot"; boot.mkdir()
        ws = tmp_path / "ws"; ws.mkdir()
        target = str(ws / "notes.md")
        gate = SecurityGate(workspace_root=str(boot))
        assert gate.evaluate("write_file", file_path=target, origin="user").requires_confirmation
        assert not gate.evaluate("write_file", file_path=target, origin="user", workspace_roots=(ws,)).requires_confirmation
        # Through the loop: the tool executes (ALLOW) with the session workspace set.
        tool = _ProbeWriteTool()
        _drain(self._ctx(_Scripted({"path": target, "content": "x"}), tool, gate, boot_cwd=boot, workspace=ws))
        assert len(tool.seen) == 1
        # And the boundary moved WITH the session: a write back under the
        # daemon's root is now the one outside the workspace.
        assert gate.evaluate("write_file", file_path=str(boot / "x.md"), origin="user", workspace_roots=(ws,)).requires_confirmation

    def test_project_section_is_swapped_for_the_workspace(self, tmp_path) -> None:
        from prometheus.context.prompt_assembler import project_files_section
        boot = tmp_path / "boot"; boot.mkdir(); (boot / "PROMETHEUS.md").write_text("BOOT RULE: daemon repo\n")
        ws = tmp_path / "ws"; ws.mkdir(); (ws / "CLAUDE.md").write_text("HOUSE RULE: never use tabs\n")
        boot_section = project_files_section({}, boot)
        assert boot_section and "BOOT RULE" in boot_section
        provider = _Scripted()
        _drain(self._ctx(provider, _ProbeWriteTool(), SecurityGate(), boot_cwd=boot, workspace=ws,
                         boot_project=boot_section, builder=lambda cwd: project_files_section({}, cwd)))
        prompt = provider.requests[0].system_prompt
        assert "HOUSE RULE: never use tabs" in prompt
        assert "BOOT RULE" not in prompt                      # replaced, not stacked
        assert f"working directory is `{ws.resolve()}`" in prompt

    def test_the_swap_survives_the_adapter(self, tmp_path) -> None:
        """Production shape, 2026-09-01: the log said "replacing the boot
        section" and the model saw the boot section — the adapter's
        format_request was fed context.system_prompt, undoing the swap before
        the first request. A REAL ModelAdapter is in the loop here."""
        from prometheus.adapter import ModelAdapter
        from prometheus.context.prompt_assembler import project_files_section
        boot = tmp_path / "boot"; boot.mkdir(); (boot / "PROMETHEUS.md").write_text("BOOT RULE\n")
        ws = tmp_path / "ws"; ws.mkdir(); (ws / "CLAUDE.md").write_text("HOUSE RULE: codename PELICAN-7\n")
        provider = _Scripted()
        ctx = self._ctx(provider, _ProbeWriteTool(), SecurityGate(), boot_cwd=boot, workspace=ws,
                        boot_project=project_files_section({}, boot),
                        builder=lambda cwd: project_files_section({}, cwd))
        ctx.adapter = ModelAdapter()
        _drain(ctx)
        prompt = provider.requests[0].system_prompt
        assert "PELICAN-7" in prompt and "BOOT RULE" not in prompt

    def test_boot_prompt_untouched_without_a_workspace(self, tmp_path) -> None:
        boot = tmp_path / "boot"; boot.mkdir(); (boot / "PROMETHEUS.md").write_text("BOOT RULE\n")
        from prometheus.context.prompt_assembler import project_files_section
        section = project_files_section({}, boot)
        provider = _Scripted()
        _drain(self._ctx(provider, _ProbeWriteTool(), SecurityGate(), boot_cwd=boot, workspace=None, boot_project=section))
        assert "BOOT RULE" in provider.requests[0].system_prompt


class TestBashFollowsTheSession:
    def test_session_roots_replace_the_lock(self, tmp_path) -> None:
        from prometheus.tools.base import ToolExecutionContext
        from prometheus.tools.builtin.bash import BashTool, BashToolInput
        boot = tmp_path / "boot"; boot.mkdir()
        ws = tmp_path / "ws"; ws.mkdir()
        tool = BashTool(workspace=str(boot))
        # Configured lock: a cwd outside `boot` is refused …
        refused = asyncio.run(tool.execute(BashToolInput(command="pwd", cwd=str(ws)), ToolExecutionContext(cwd=boot)))
        assert refused.is_error and "Workspace lock violation" in refused.output
        # … and accepted when the session's roots say `ws` is the boundary.
        ok = asyncio.run(tool.execute(
            BashToolInput(command="pwd", cwd=str(ws)),
            ToolExecutionContext(cwd=ws, metadata={"workspace_roots": (ws,)}),
        ))
        assert not ok.is_error and str(ws.resolve()) in ok.output
