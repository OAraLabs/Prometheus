"""A run the operator HOLDS to a restricted profile gets that profile's instructions, not the owner's prompt, and
its tool search lists only what the profile allows.

A device held to a profile (``PUT /api/devices/<id>/profile``) is the caller that must see less than the owner: it
is offered the profile's tools, and a tool outside them is refused at the call. Two things still reached it:

* the system prompt. The daemon builds ONE prompt at boot for its owner (the persona, the owner's instructions, the
  machine and network summary, the "# Memory" section, the workspace's project files) and every run was sent it,
  whatever profile the run was held to; the profile's own ``bootstrap_files`` were never applied;
* ``tool_search``. It searched the whole registry, so a held run could list, by name and input schema, the tools
  its profile left out. A call to one was refused, but the listing was not.

Pinned here: a held run is sent the files its profile names in ``bootstrap_files``, read ONLY from the profiles folder
(``~/.prometheus/profiles/``, which every write floor already covers; a file named but kept anywhere else is refused with a
logged message) (a neutral line when it names none that exist), plus what the connecting client itself sent as instructions on the OpenAI-compatible route, and
nothing of the owner's prompt, recall, memory section or project files, unless the profile names the file; a run that is not held is unchanged; ``tool_search`` shows a restricted run
only the tools the profile permits.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.config.device_store import DeviceStore
from prometheus.config.profiles import RUN_PROFILE, ActiveProfileState, ProfileStore
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.modes import PermissionMode
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from prometheus.tools.tool_search import ToolSearchInput, ToolSearchTool

OWNER_PROMPT = (
    "SOUL-SENTINEL you are the owner's assistant.\n\nANATOMY-SENTINEL machine summary 10.0.0.9\n\n"
    "# Memory\nMEMORY-SENTINEL the owner's private fact\n\n# Workspace\nPROJECT-SENTINEL owner project file"
)
BOOT_MEMORY = "# Memory\nMEMORY-SENTINEL the owner's private fact"
CLIENT_HEADER = "# Instructions from the connecting client"
OWNER_SENTINELS = ("SOUL-SENTINEL", "ANATOMY-SENTINEL", "MEMORY-SENTINEL", "PROJECT-SENTINEL", "RECALL-SENTINEL")


class _Path(BaseModel):
    path: str


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool"
        input_model = _Path

        def is_read_only(self, arguments) -> bool:  # noqa: ANN001
            return True

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output=f"{tool_name} ran")

    return _T()


class _Recall:
    def __init__(self) -> None:
        self.calls = 0

    def recall(self, text: str) -> str:
        self.calls += 1
        return "RECALL-SENTINEL a fact from the owner's memory"


class _Capture(ModelProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.prompts.append(request.system_prompt or "")
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


class _Rig:
    def __init__(self, tmp_path: Path, monkeypatch, relay_yaml: str) -> None:
        # The config directory is where bootstrap files live (~/.prometheus): a scratch one with the OWNER's files
        # and the relay profile's own instruction file.
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(cfg))
        (cfg / "SOUL.md").write_text("SOUL-SENTINEL you are the owner's assistant.\n")
        (cfg / "AGENTS.md").write_text("OWNER-AGENTS-SENTINEL the owner's standing instructions.\n")
        (cfg / "ANATOMY.md").write_text("ANATOMY-SENTINEL machine summary: workstation at 10.0.0.9\n")
        # A held profile's own files live beside its YAML, in the profiles folder (floored by every layer).
        profiles_dir = cfg / "profiles"
        profiles_dir.mkdir()
        self.profiles_dir = profiles_dir
        (profiles_dir / "AGENTS-X.md").write_text("RELAY-FILE-SENTINEL answer only from the files you can read.\n")
        (profiles_dir / "relay.yaml").write_text(relay_yaml)
        self.state = ActiveProfileState(ProfileStore(profiles_dir), "full")
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.state.session_hold = self.devices.session_hold
        self.held = self.devices.mint("upstream relay", "other")
        self.free = self.devices.mint("alice's phone", "ios")
        assert self.devices.set_profile(self.held["id"], "relay", by="test")
        assert self.devices.claim_session("beacon:held", self.held["id"])
        assert self.devices.claim_session("beacon:free", self.free["id"])
        self.registry = ToolRegistry()
        for name in ("read_file", "write_file", "bash"):
            self.registry.register(_tool(name))
        self.provider = _Capture()
        self.recall = _Recall()
        self.built = {"memory": 0, "project": 0}
        self.tmp_path = tmp_path
        (tmp_path / "ws").mkdir()

    def context(self, system_prompt: str = OWNER_PROMPT, client_instructions: str = "") -> LoopContext:
        def memory_builder() -> str:
            self.built["memory"] += 1
            return "# Memory\nMEMORY-SENTINEL newer owner fact"

        def project_builder(path: str) -> str:
            self.built["project"] += 1
            return "PROJECT-SENTINEL the workspace's owner instructions"

        return LoopContext(
            provider=self.provider, model="stub", system_prompt=system_prompt, max_tokens=64,
            tool_registry=self.registry,
            permission_checker=SecurityGate(mode=PermissionMode.AUTONOMOUS, workspace_root=self.tmp_path),
            profile_resolver=self.state.get, cwd=self.tmp_path,
            memory_recall=self.recall, memory_prompt_builder=memory_builder, boot_memory_prompt=BOOT_MEMORY,
            project_prompt_builder=project_builder, boot_project_prompt="# Workspace\nPROJECT-SENTINEL owner project file",
            workspace_resolver=lambda sid: str(self.tmp_path / "ws"), client_instructions=client_instructions,
        )

    def run(self, session_id: str, system_prompt: str = OWNER_PROMPT, client_instructions: str = "") -> str:
        self.provider.prompts.clear()

        async def go() -> None:
            async for _ in run_loop(self.context(system_prompt, client_instructions),
                                    [ConversationMessage.from_user_text("hello")],
                                    session_id=session_id):
                pass

        asyncio.run(go())
        assert self.provider.prompts, "the model was never called"
        return self.provider.prompts[0]


RELAY = "name: relay\ntools: [read_file]\nbootstrap_files: [AGENTS-X.md]\n"
OWNER_FILES = ("SOUL-SENTINEL", "OWNER-AGENTS-SENTINEL", "ANATOMY-SENTINEL")


@pytest.fixture
def rig(tmp_path, monkeypatch) -> _Rig:
    return _Rig(tmp_path, monkeypatch, RELAY)


# -- the prompt ------------------------------------------------------------------------------------------------


def test_only_a_custom_profile_that_names_its_bootstrap_files_is_marked_so(rig):
    store = rig.state._store
    assert store.get("relay").bootstrap_declared and store.get("relay").bootstrap_files == ["AGENTS-X.md"]
    assert not store.get("coder").bootstrap_declared                  # a builtin names none for a hold


def test_a_held_run_is_sent_the_file_its_profile_names(rig):
    assert "RELAY-FILE-SENTINEL" in rig.run("beacon:held")


def test_a_held_run_is_sent_none_of_the_owners_prompt(rig):
    prompt = rig.run("beacon:held")
    leaked = [s for s in (*OWNER_SENTINELS, *OWNER_FILES) if s in prompt]
    assert not leaked, f"the held run was sent the owner's {leaked}"
    assert "10.0.0.9" not in prompt and "## Infrastructure" not in prompt


def test_a_held_run_never_asks_for_recall_memory_or_project_files(rig):
    rig.run("beacon:held")
    assert rig.recall.calls == 0, "recall from the owner's memory ran for a held run"
    assert rig.built == {"memory": 0, "project": 0}, "the owner's memory or project section was built for a held run"


def test_a_file_the_profile_names_is_loaded_from_the_profiles_folder_never_the_owners_of_the_same_name(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: [SOUL.md, AGENTS-X.md]\n")
    (rig.profiles_dir / "SOUL.md").write_text("PROFILE-SOUL-SENTINEL the relay's own persona.\n")
    prompt = rig.run("beacon:held")
    assert "PROFILE-SOUL-SENTINEL" in prompt and "RELAY-FILE-SENTINEL" in prompt
    assert "SOUL-SENTINEL you are the owner's" not in prompt, "the owner's SOUL.md was read for a held run"
    assert "OWNER-AGENTS-SENTINEL" not in prompt and "ANATOMY-SENTINEL" not in prompt


def test_a_name_with_no_file_in_the_profiles_folder_gets_nothing_not_the_owners_file(tmp_path, monkeypatch):
    # The profile names SOUL.md but keeps no copy in its folder: it must not fall back to the owner's.
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: [SOUL.md]\n")
    prompt = rig.run("beacon:held")
    assert not [s for s in (*OWNER_SENTINELS, *OWNER_FILES) if s in prompt]
    assert prompt.strip()


def test_a_file_kept_in_the_config_folder_is_refused_with_a_message_that_says_where_it_goes(tmp_path, monkeypatch, caplog):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: [AGENTS-Y.md]\n")
    (tmp_path / "cfg" / "AGENTS-Y.md").write_text("MISPLACED-SENTINEL kept beside the owner's files\n")
    import logging

    with caplog.at_level(logging.WARNING):
        prompt = rig.run("beacon:held")
    assert "MISPLACED-SENTINEL" not in prompt
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "AGENTS-Y.md" in messages and "profiles" in messages and "relay" in messages
    # It says the file IS in the config folder and to copy it, not just that nothing is there.
    assert "is in the config folder" in messages and "copy it into" in messages


def test_the_profiles_folder_is_under_every_write_floor():
    # What makes a file there safe from a model-written shell, by construction.
    from pathlib import Path

    from prometheus.permissions.confinement import TRUSTED_DIRS

    assert "profiles" in TRUSTED_DIRS
    profile = (Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash").read_text()
    assert "{checkpoints,profiles}/{,**}" in profile


def test_the_machine_summary_comes_only_when_the_profile_names_it(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: [AGENTS-X.md, ANATOMY.md]\n")
    (rig.profiles_dir / "ANATOMY.md").write_text("PROFILE-ANATOMY-SENTINEL a machine summary at 10.0.0.9\n")
    prompt = rig.run("beacon:held")
    # The summary section, in the form any prompt for this route carries it (the stub provider is a hosted route,
    # which swaps the summary for a pointer to the anatomy tool); never the raw text or its addresses.
    assert "## Infrastructure" in prompt and "RELAY-FILE-SENTINEL" in prompt
    assert "10.0.0.9" not in prompt, "the machine summary kept its address for a restricted run"
    assert "ANATOMY-SENTINEL machine summary" not in prompt, "the owner's ANATOMY.md was read for a held run"


def test_a_profile_that_did_not_name_its_files_does_not_get_the_default_ones(tmp_path, monkeypatch):
    # No `bootstrap_files:` key: the profile's default list is the OWNER's three files, which a held run must not get.
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\n")
    prompt = rig.run("beacon:held")
    assert prompt.strip(), "a held run was sent no system prompt at all"
    assert not [s for s in (*OWNER_SENTINELS, *OWNER_FILES) if s in prompt]


def test_a_missing_file_or_one_outside_the_config_directory_is_skipped(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch,
               "name: relay\ntools: [read_file]\nbootstrap_files: [no-such.md, ../outside.md, AGENTS-X.md]\n")
    (tmp_path / "outside.md").write_text("OUTSIDE-SENTINEL a file beside the config directory\n")
    prompt = rig.run("beacon:held")
    assert "RELAY-FILE-SENTINEL" in prompt and "OUTSIDE-SENTINEL" not in prompt


def test_the_connecting_clients_own_instructions_still_reach_a_held_run(rig):
    prompt = rig.run("beacon:held", f"{OWNER_PROMPT}\n\n{CLIENT_HEADER}\nCLIENT-SENTINEL talk like a pirate",
                     client_instructions="CLIENT-SENTINEL talk like a pirate")
    assert "CLIENT-SENTINEL" in prompt and "RELAY-FILE-SENTINEL" in prompt
    assert not [s for s in (*OWNER_SENTINELS, *OWNER_FILES) if s in prompt]


def test_owner_text_that_looks_like_the_clients_heading_is_not_taken_for_the_clients(rig):
    # The client's text travels in its own field, not found by a heading in the owner's prompt.
    owner = f"{OWNER_PROMPT}\n\n{CLIENT_HEADER}\nOWNER-LOOKALIKE-SENTINEL written in the owner's memory"
    prompt = rig.run("beacon:held", owner)
    assert "OWNER-LOOKALIKE-SENTINEL" not in prompt


def test_a_builtin_profile_names_no_files_for_a_hold(tmp_path, monkeypatch):
    # The builtins list the owner's persona and instructions (assistant its AGENTS.md): a device held to one gets
    # the neutral line, not the owner's files.
    rig = _Rig(tmp_path, monkeypatch, RELAY)
    assert rig.devices.set_profile(rig.held["id"], "assistant", by="test")
    prompt = rig.run("beacon:held")
    assert not [s for s in (*OWNER_SENTINELS, *OWNER_FILES) if s in prompt]
    assert prompt.strip()


def test_a_file_that_is_not_markdown_text_is_skipped_not_fatal(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: [devices.db, AGENTS-X.md]\n")
    (rig.profiles_dir / "devices.db").write_bytes(b"\xff\xfe\x00binary")
    prompt = rig.run("beacon:held")
    assert "RELAY-FILE-SENTINEL" in prompt


def test_a_link_out_of_the_config_directory_is_not_followed(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: [link.md, AGENTS-X.md]\n")
    (tmp_path / "elsewhere.md").write_text("LINKED-SENTINEL a file outside the config directory\n")
    (rig.profiles_dir / "link.md").symlink_to(tmp_path / "elsewhere.md")
    prompt = rig.run("beacon:held")
    assert "LINKED-SENTINEL" not in prompt and "RELAY-FILE-SENTINEL" in prompt


def test_one_file_named_as_a_string_is_one_file(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\ntools: [read_file]\nbootstrap_files: AGENTS-X.md\n")
    assert "RELAY-FILE-SENTINEL" in rig.run("beacon:held")


def test_a_run_that_is_not_held_is_unchanged(rig):
    prompt = rig.run("beacon:free")
    assert "SOUL-SENTINEL" in prompt and "RELAY-FILE-SENTINEL" not in prompt
    assert rig.recall.calls == 1


# -- tool_search ------------------------------------------------------------------------------------------------


def _search(registry: ToolRegistry, query: str, action: str = "search") -> str:
    tool = ToolSearchTool()
    tool.set_registry(registry)
    out = asyncio.run(tool.execute(ToolSearchInput(query=query, action=action),
                                   ToolExecutionContext(cwd=Path("."))))
    return out.output


@pytest.fixture
def restricted(rig):
    token = RUN_PROFILE.set(rig.state._store.get("relay"))
    try:
        yield rig
    finally:
        RUN_PROFILE.reset(token)


def test_tool_search_lists_every_tool_when_no_profile_limits_the_run(rig):
    listed = json.loads(_search(rig.registry, ""))["tools"]
    assert listed == ["bash", "read_file", "write_file"]


def test_tool_search_lists_only_the_profiles_tools_to_a_restricted_run(restricted):
    assert json.loads(_search(restricted.registry, ""))["tools"] == ["read_file"]


def test_a_restricted_search_by_description_finds_no_tool_the_profile_left_out(restricted):
    found = _search(restricted.registry, "bash test tool")
    assert "bash" not in [e["name"] for e in json.loads(found)]


def test_a_restricted_select_does_not_hand_over_a_tool_the_profile_left_out(restricted):
    out = _search(restricted.registry, "bash", action="select")
    assert '"input_schema"' not in out and '"properties"' not in out
    assert "write_file" not in out and "bash" not in json.loads(out).get("closest", [])


def test_a_restricted_run_can_still_select_a_tool_it_may_call(restricted):
    assert "read_file" in _search(restricted.registry, "read_file", action="select")


# -- a name the profile leaves out is refused the same way whether or not the tool exists ---------------------------


class _CallsThenAnswers(ModelProvider):
    """Round 1 calls each name in *names* (with no real arguments); round 2 answers. Keeps the tool results."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.results: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        from prometheus.engine.messages import ToolUseBlock

        last = request.messages[-1]
        blocks = getattr(last, "content", []) or []
        answered = any(getattr(b, "type", "") == "tool_result" for b in blocks)
        if answered:
            self.results = [str(getattr(b, "content", "")) for b in blocks if getattr(b, "type", "") == "tool_result"]
            content = [TextBlock(text="done")]
        else:
            content = [ToolUseBlock(id=f"t{i}", name=n, input={"path": "x"}) for i, n in enumerate(self.names)]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


def _run_calls(rig: _Rig, session: str, names: list[str]) -> list[str]:
    import dataclasses

    from prometheus.adapter import ModelAdapter

    rig.provider = _CallsThenAnswers(names)
    ctx = dataclasses.replace(rig.context(), provider=rig.provider, adapter=ModelAdapter(tier=ModelAdapter.TIER_FULL))

    async def go() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=session):
            pass

    asyncio.run(go())
    return rig.provider.results


def test_a_held_run_that_calls_a_made_up_tool_is_not_told_what_tools_exist(rig):
    results = _run_calls(rig, "beacon:held", ["zzz_not_a_tool", "bsh", "write_file"])
    text = "\n".join(results)
    assert results, "no tool result came back"
    assert "bash" not in text and "write_file" not in text.replace("write_file is not available", "")
    assert "read_file" in text                        # the tools it may use are named


def test_the_refusal_is_the_same_for_a_tool_that_exists_and_one_that_does_not(rig):
    results = _run_calls(rig, "beacon:held", ["write_file", "zzz_not_a_tool"])
    shapes = {r.replace("write_file", "X").replace("zzz_not_a_tool", "X") for r in results}
    assert len(shapes) == 1, results


def test_a_profile_that_only_excludes_tools_still_does_not_list_the_registry_for_a_made_up_name(tmp_path, monkeypatch):
    rig = _Rig(tmp_path, monkeypatch, "name: relay\nexclude_tools: [bash, write_file]\nbootstrap_files: [AGENTS-X.md]\n")
    results = _run_calls(rig, "beacon:held", ["zzz_not_a_tool", "bsh"])
    text = "\n".join(results)
    assert results and "bash" not in text and "write_file" not in text
    assert "read_file" in text
