"""A profile limits what a run can CALL, not only what it is shown.

An agent profile (``tools`` / ``exclude_tools``, daemon-wide or bound to one
session) used to filter only the schemas a run advertised. ``_execute_tool_call``
looked the call up in the full registry, so a model that named a tool the
profile left out (by guessing, or because injected text told it to) ran it, and
the gate treated it exactly as under ``full``. ``research`` said "No file
mutations" and wrote files; ``exclude_tools: [bash]`` ran bash.

Pinned here, through real turns (``run_loop`` / ``AgentLoop``) and a real
SecurityGate in AUTONOMOUS mode, so nothing but the profile can be what stops a
call:

* a call outside the run's profile is refused, names the profile, and never runs;
* a deferred tool the profile allows still runs (a lucky guess is not a refusal);
* a tool dropped only by ``max_tool_schemas`` (a context budget) still runs;
* a profile that matches no tool is ignored for calls as it is for advertising;
* a subagent's tools are cut to the profile: a restricted session cannot spawn
  a worker with bash or file writes;
* the tools that start shells or worker processes (``task_create``,
  ``sessions_spawn``) are held to the same rule.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.config.profiles import AgentProfile
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.modes import PermissionMode
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

RAN: list[str] = []


class _Cmd(BaseModel):
    command: str


class _Write(BaseModel):
    path: str
    content: str


class _Path(BaseModel):
    path: str


class _Empty(BaseModel):
    pass


def _tool(tool_name: str, model: type[BaseModel], *, read_only: bool) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool"
        input_model = model

        def is_read_only(self, arguments) -> bool:  # noqa: ANN001
            return read_only

        async def execute(self, arguments, context):  # noqa: ANN001
            RAN.append(tool_name)
            return ToolResult(output=f"{tool_name} ran", is_error=False)

    return _T()


def _registry(*extra: BaseTool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in (
        _tool("bash", _Cmd, read_only=False),
        _tool("write_file", _Write, read_only=False),
        _tool("read_file", _Path, read_only=True),
        _tool("image_generate", _Empty, read_only=False),
        *extra,
    ):
        reg.register(t)
    return reg


class _Script(ModelProvider):
    """Answers each request with the next scripted reply; records the tool names each request offered."""

    def __init__(self, *replies: list) -> None:
        self.replies = list(replies)
        self.offered: list[list[str]] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.offered.append(sorted(t.get("name") for t in (request.tools or [])))
        content = self.replies.pop(0) if self.replies else [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _call(name: str, **args) -> list:
    return [ToolUseBlock(id=f"t-{name}", name=name, input=args)]


def _gate(tmp_path: Path) -> SecurityGate:
    return SecurityGate(mode=PermissionMode.AUTONOMOUS, workspace_root=tmp_path)


def _turn(tmp_path: Path, profile: AgentProfile | None, *replies: list, registry=None, loader=None):
    """One turn. Returns (provider, the text of every tool result the model was sent back)."""
    from prometheus.adapter import ModelAdapter

    RAN.clear()
    provider = _Script(*replies)
    ctx = LoopContext(
        provider=provider, model="stub", system_prompt="", max_tokens=64,
        tool_registry=registry or _registry(), tool_loader=loader,
        adapter=ModelAdapter(tier="light"),
        permission_checker=_gate(tmp_path),
        profile_resolver=(lambda session_id=None: profile) if profile is not None else None,
        cwd=tmp_path,
    )
    messages = [ConversationMessage.from_user_text("go")]

    async def drain():
        async for _ in run_loop(ctx, messages, session_id="probe:profile"):
            pass

    asyncio.run(drain())
    results = [
        getattr(block, "content", "")
        for m in messages for block in (m.content if isinstance(m.content, list) else [])
        if type(block).__name__ == "ToolResultBlock"
    ]
    return provider, results


RESEARCH = AgentProfile(name="research", tools=["read_file"])
NO_BASH = AgentProfile(name="no-bash", tools=None, exclude_tools=["bash"])


# ---------------------------------------------------------------------------
# The rule: a call outside the profile does not run
# ---------------------------------------------------------------------------


def test_a_tool_left_out_of_the_profile_is_refused_and_names_the_profile(tmp_path):
    provider, results = _turn(
        tmp_path, RESEARCH, _call("write_file", path=str(tmp_path / "x.txt"), content="hi"),
    )
    assert provider.offered[0] == ["read_file"], "the advertisement is already filtered"
    assert RAN == [], "write_file is not in the 'research' profile, so it must never run"
    assert len(results) == 1
    assert "research" in results[0] and "write_file" in results[0]


def test_an_excluded_tool_is_refused(tmp_path):
    _, results = _turn(tmp_path, NO_BASH, _call("bash", command="echo hi"))
    assert RAN == []
    assert "no-bash" in results[0] and "bash" in results[0]


def test_a_tool_the_profile_allows_still_runs(tmp_path):
    _turn(tmp_path, RESEARCH, _call("read_file", path=str(tmp_path / "a.txt")))
    assert RAN == ["read_file"]


def test_no_profile_refuses_nothing(tmp_path):
    _turn(tmp_path, None, _call("bash", command="echo hi"))
    assert RAN == ["bash"]


# ---------------------------------------------------------------------------
# What is NOT a refusal
# ---------------------------------------------------------------------------


def test_a_deferred_tool_the_profile_allows_still_runs(tmp_path):
    # Deferred loading advertises bash + read_file; image_generate is registered and allowed but not sent,
    # so calling it is a lucky guess, which must keep working.
    from prometheus.context.dynamic_tools import DynamicToolLoader

    registry = _registry()
    loader = DynamicToolLoader(registry, {"enabled": True, "always_loaded": ["bash", "read_file"],
                                          "local_catalog": "names"})
    profile = AgentProfile(name="art", tools=["bash", "read_file", "image_generate"])
    provider, _ = _turn(tmp_path, profile, _call("image_generate"), registry=registry, loader=loader)
    assert "image_generate" not in provider.offered[0]
    assert RAN == ["image_generate"]


def test_a_tool_cut_only_by_max_tool_schemas_still_runs(tmp_path):
    # max_tool_schemas is a context budget, not a permission.
    profile = AgentProfile(name="lean", tools=None, max_tool_schemas=1)
    provider, _ = _turn(tmp_path, profile, _call("write_file", path=str(tmp_path / "y.txt"), content="x"))
    assert "write_file" not in provider.offered[0]
    assert RAN == ["write_file"]


def test_a_profile_that_matches_no_tool_is_ignored_for_calls_too(tmp_path):
    # A profile naming no registered tool is a config error: the run advertises everything (and logs an
    # ERROR). Refusing every call while advertising every tool would be worse than either.
    profile = AgentProfile(name="typo", tools=["no_such_tool"])
    _turn(tmp_path, profile, _call("bash", command="echo hi"))
    assert RAN == ["bash"]


# ---------------------------------------------------------------------------
# Subagents: the profile follows the call into the subagent's registry
# ---------------------------------------------------------------------------


def _agent_loop(tmp_path: Path, profile: AgentProfile, provider: ModelProvider) -> AgentLoop:
    from prometheus.adapter import ModelAdapter
    from prometheus.coordinator.subagent import attach_subagent_spawner
    from prometheus.tools.builtin.agent import AgentTool

    loop = AgentLoop(
        provider=provider, model="stub", max_tokens=64,
        tool_registry=_registry(AgentTool()), adapter=ModelAdapter(tier="light"),
        permission_checker=_gate(tmp_path), cwd=tmp_path,
        profile_resolver=lambda session_id=None: profile,
    )
    attach_subagent_spawner(loop)
    return loop


def test_a_restricted_session_cannot_spawn_a_worker_with_bash_or_file_writes(tmp_path):
    RAN.clear()
    # Parent (round 1) spawns a worker; the worker (its round 1) tries bash, then write_file; then both stop.
    provider = _Script(
        _call("agent", description="do it", prompt="run something", subagent_type="worker"),
        _call("bash", command="echo hi"),
        _call("write_file", path=str(tmp_path / "z.txt"), content="x"),
    )
    profile = AgentProfile(name="research", tools=["read_file", "agent"])
    loop = _agent_loop(tmp_path, profile, provider)
    asyncio.run(loop.run_async(system_prompt="", user_message="go"))

    assert RAN == [], f"the worker ran {RAN} under a profile that allows neither"
    worker_offered = provider.offered[1]
    assert "bash" not in worker_offered and "write_file" not in worker_offered
    assert "read_file" in worker_offered, "the worker keeps the tools both its type and the profile allow"


def test_an_unrestricted_session_still_gives_a_worker_its_tools(tmp_path):
    RAN.clear()
    provider = _Script(
        _call("agent", description="do it", prompt="run something", subagent_type="worker"),
        _call("bash", command="echo hi"),
    )
    loop = _agent_loop(tmp_path, AgentProfile(name="full", tools=None), provider)
    asyncio.run(loop.run_async(system_prompt="", user_message="go"))
    assert RAN == ["bash"]


# ---------------------------------------------------------------------------
# Tools that start shells or worker processes
# ---------------------------------------------------------------------------


class _FakeManager:
    def __init__(self) -> None:
        self.started: list[str] = []

    async def create_shell_task(self, **kw):  # noqa: ANN003
        self.started.append("shell")
        raise AssertionError("a shell task started")

    async def create_poll_task(self, **kw):  # noqa: ANN003
        self.started.append("poll")
        raise AssertionError("a poll task started")

    async def create_agent_task(self, **kw):  # noqa: ANN003
        self.started.append("agent")
        raise AssertionError("a worker process started")


@pytest.fixture
def manager(monkeypatch):
    fake = _FakeManager()
    import prometheus.tools.builtin.sessions_spawn as ss
    import prometheus.tools.builtin.task_create as tc

    monkeypatch.setattr(tc, "get_task_manager", lambda: fake)
    monkeypatch.setattr(ss, "get_task_manager", lambda: fake)
    return fake


def _spawn_registry() -> ToolRegistry:
    from prometheus.tools.builtin.sessions_spawn import SessionsSpawnTool
    from prometheus.tools.builtin.task_create import TaskCreateTool

    return _registry(TaskCreateTool(), SessionsSpawnTool())


SPAWNER = AgentProfile(name="planner", tools=["read_file", "task_create", "sessions_spawn"])


def test_task_create_cannot_run_a_shell_command_when_the_profile_has_no_bash(tmp_path, manager):
    _, results = _turn(
        tmp_path, SPAWNER, _call("task_create", type="local_bash", description="d", command="echo hi"),
        registry=_spawn_registry(),
    )
    assert manager.started == []
    assert "planner" in results[0] and "bash" in results[0]


def test_task_create_cannot_poll_a_shell_command_when_the_profile_has_no_bash(tmp_path, manager):
    _, results = _turn(
        tmp_path, SPAWNER,
        _call("task_create", type="poll", description="d", poll_predicate="test -f /tmp/x"),
        registry=_spawn_registry(),
    )
    assert manager.started == []
    assert "planner" in results[0] and "bash" in results[0]


def test_task_create_cannot_start_a_worker_process_under_a_restricting_profile(tmp_path, manager):
    _, results = _turn(
        tmp_path, SPAWNER, _call("task_create", type="local_agent", description="d", prompt="work"),
        registry=_spawn_registry(),
    )
    assert manager.started == []
    assert "planner" in results[0]


def test_sessions_spawn_cannot_start_a_worker_process_under_a_restricting_profile(tmp_path, manager):
    _, results = _turn(tmp_path, SPAWNER, _call("sessions_spawn", prompt="work"), registry=_spawn_registry())
    assert manager.started == []
    assert "planner" in results[0]
