"""A MEMORY.md / USER.md write reaches the NEXT turn's system prompt, no restart.

The daemon builds its system prompt once at startup, and every turn reused it:
Telegram/Slack/Discord hand ``self.system_prompt`` to ``AgentLoop.run_async``,
and the web bridge runs ``run_loop`` on one shared ``LoopContext``. The memory
tool and ``PUT /api/memory/current`` write the files at once, so the model kept
seeing the boot copy of its own memory until the daemon restarted. Measured on
3c0f5da before the fix: both paths below wrote the file, and the next turn's
prompt still said ``BOOT-FACT`` only.

The fix re-reads the files once per run and swaps the boot "# Memory" section
by exact substring (the item W project-section pattern). The contract here:

- the memory tool (called BY the model, through the loop) and the PUT both
  reach the next run; a hand edit does too;
- the section is replaced, not stacked; cleared files drop it;
- memory that was empty at boot appears without a restart;
- nothing changes within a run, and unchanged files give a byte-identical
  prompt, so a provider's cached prefix survives every run that wrote nothing;
- a caller's own prompt (POST /api/chat sends ``gateway.system_prompt``) is
  left alone, and a read error keeps the prompt it had;
- the daemon and the CLI actually wire it (AST), from ONE read of the files.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from prometheus.context.prompt_assembler import (
    _MEMORY_ETIQUETTE,
    _load_memory_and_user,
    build_runtime_system_prompt,
    memory_section,
)
from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.memory.hermes_memory_tool import MemoryTool
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import ToolRegistry

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


class _Scripted(ModelProvider):
    """Records every request; calls the memory tool on the first one if asked."""

    def __init__(self, memory_add: dict | None = None) -> None:
        self.memory_add = memory_add
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        if self.memory_add is not None and len(self.requests) == 1:
            content = [ToolUseBlock(id="m1", name="memory", input=self.memory_add)]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "prom-home"
    h.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(h))
    return h


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(MemoryTool())
    return r


def _daemon_boot(cwd: Path) -> tuple[str, str | None]:
    """The daemon's boot path: one read, the prompt built from it, its section."""
    content = _load_memory_and_user()
    prompt = build_runtime_system_prompt(cwd=str(cwd), config={}, skills=[], memory_content=content)
    return prompt, memory_section(content)


def _gateway_loop(provider, cwd: Path, boot_section: str | None) -> AgentLoop:
    """Shaped like daemon.py's AgentLoop (Telegram/Slack/Discord/REST)."""
    return AgentLoop(
        provider=provider, model="stub", tool_registry=_registry(), cwd=cwd,
        boot_memory_prompt=boot_section, memory_prompt_builder=memory_section,
    )


def _web_context(provider, cwd: Path, boot_prompt: str, boot_section: str | None) -> LoopContext:
    """Shaped like daemon.py's shared web LoopContext: built once, sealed."""
    return LoopContext(
        provider=provider, model="stub", system_prompt=boot_prompt, max_tokens=64,
        tool_registry=_registry(), cwd=cwd,
        boot_memory_prompt=boot_section, memory_prompt_builder=memory_section,
    ).seal()


def _web_turn(ctx: LoopContext) -> None:
    """The web bridge's shape: run_loop directly on the shared context."""
    async def go():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:s1"):
            pass
    asyncio.run(go())


def _put(body: dict) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    r = TestClient(create_app({})).put("/api/memory/current", json=body)
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# the two write paths
# --------------------------------------------------------------------------- #

def test_a_memory_tool_write_reaches_the_next_gateway_turn(home, tmp_path) -> None:
    (home / "MEMORY.md").write_text("BOOT-FACT\n", encoding="utf-8")
    boot_prompt, boot_section = _daemon_boot(tmp_path)
    provider = _Scripted(memory_add={"operation": "add", "entry": "TOOL-FACT"})
    loop = _gateway_loop(provider, tmp_path, boot_section)

    async def go():
        await loop.run_async(system_prompt=boot_prompt, user_message="remember TOOL-FACT")
        assert "TOOL-FACT" in (home / "MEMORY.md").read_text(encoding="utf-8"), "the tool did not write"
        await loop.run_async(system_prompt=boot_prompt, user_message="what do you know?")
    asyncio.run(go())

    run1, run2 = provider.requests[:2], provider.requests[2]
    assert len(provider.requests) == 3  # tool call + answer, then the second turn
    # Within the run the prompt is frozen: the write lands on the NEXT run, so
    # the rest of this one keeps its cached prefix.
    assert run1[0].system_prompt == run1[1].system_prompt
    assert "TOOL-FACT" not in run1[1].system_prompt
    assert "TOOL-FACT" in run2.system_prompt, "the next turn still saw the boot copy of memory"
    assert "BOOT-FACT" in run2.system_prompt
    assert run2.system_prompt.count(_MEMORY_ETIQUETTE) == 1  # replaced, not stacked


def test_a_beacon_put_reaches_the_next_web_turn(home, tmp_path) -> None:
    (home / "MEMORY.md").write_text("BOOT-FACT\n", encoding="utf-8")
    boot_prompt, boot_section = _daemon_boot(tmp_path)
    provider = _Scripted()
    ctx = _web_context(provider, tmp_path, boot_prompt, boot_section)

    _web_turn(ctx)
    _put({"memory": "PUT-FACT", "user": "Prefers short answers."})
    _web_turn(ctx)

    prompt = provider.requests[-1].system_prompt
    assert "PUT-FACT" in prompt and "Prefers short answers." in prompt
    assert "BOOT-FACT" not in prompt  # the PUT replaced it, and so did the prompt
    assert prompt.count(_MEMORY_ETIQUETTE) == 1
    assert ctx.system_prompt == boot_prompt  # the shared context was never written


def test_a_hand_edit_reaches_the_next_turn(home, tmp_path) -> None:
    boot_prompt, boot_section = _daemon_boot(tmp_path)
    provider = _Scripted()
    ctx = _web_context(provider, tmp_path, boot_prompt, boot_section)
    (home / "USER.md").write_text("Works from Lisbon.\n", encoding="utf-8")
    _web_turn(ctx)
    assert "Works from Lisbon." in provider.requests[-1].system_prompt


# --------------------------------------------------------------------------- #
# shape of the swap
# --------------------------------------------------------------------------- #

def test_unchanged_files_give_a_byte_identical_prompt(home, tmp_path) -> None:
    """The prompt-caching property: no write, no change, run after run."""
    (home / "MEMORY.md").write_text("BOOT-FACT\n", encoding="utf-8")
    (home / "USER.md").write_text("USER-FACT\n", encoding="utf-8")
    boot_prompt, boot_section = _daemon_boot(tmp_path)
    provider = _Scripted()
    ctx = _web_context(provider, tmp_path, boot_prompt, boot_section)
    _web_turn(ctx)
    _web_turn(ctx)
    assert provider.requests[0].system_prompt == boot_prompt
    assert provider.requests[1].system_prompt == boot_prompt


def test_memory_empty_at_boot_appears_without_a_restart(home, tmp_path) -> None:
    """A fresh install: no memory at boot, so the boot prompt has no section to swap."""
    boot_prompt, boot_section = _daemon_boot(tmp_path)
    assert boot_section is None and _MEMORY_ETIQUETTE not in boot_prompt
    provider = _Scripted(memory_add={"operation": "add", "entry": "FIRST-FACT"})
    loop = _gateway_loop(provider, tmp_path, boot_section)

    async def go():
        await loop.run_async(system_prompt=boot_prompt, user_message="remember FIRST-FACT")
        await loop.run_async(system_prompt=boot_prompt, user_message="and now?")
    asyncio.run(go())

    prompt = provider.requests[-1].system_prompt
    assert "FIRST-FACT" in prompt
    assert prompt.count(_MEMORY_ETIQUETTE) == 1
    # First in the dynamic part — never after text a caller appended to the boot prompt.
    dynamic = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY, 1)[1]
    assert dynamic.lstrip().startswith("# Memory")


def test_memory_empty_at_boot_lands_before_a_clients_own_system_text(home, tmp_path) -> None:
    """The OpenAI-compatible route appends the client's system messages to the
    boot prompt (web/openai_api.py); memory must not land among them."""
    import dataclasses

    boot_prompt, boot_section = _daemon_boot(tmp_path)
    provider = _Scripted()
    shared = _web_context(provider, tmp_path, boot_prompt, boot_section)
    ctx = dataclasses.replace(shared, system_prompt=f"{boot_prompt}\n\nCLIENT-SYSTEM-TEXT")
    (home / "MEMORY.md").write_text("LATE-FACT\n", encoding="utf-8")
    _web_turn(ctx)
    prompt = provider.requests[-1].system_prompt
    assert prompt.index("LATE-FACT") < prompt.index("CLIENT-SYSTEM-TEXT")


def test_cleared_files_drop_the_section(home, tmp_path) -> None:
    (home / "MEMORY.md").write_text("BOOT-FACT\n", encoding="utf-8")
    boot_prompt, boot_section = _daemon_boot(tmp_path)
    provider = _Scripted()
    ctx = _web_context(provider, tmp_path, boot_prompt, boot_section)
    _put({"memory": ""})
    _web_turn(ctx)
    prompt = provider.requests[-1].system_prompt
    assert "BOOT-FACT" not in prompt and _MEMORY_ETIQUETTE not in prompt
    assert "\n\n\n" not in prompt  # its separator went with it


def test_a_callers_own_prompt_is_left_alone(home, tmp_path) -> None:
    """POST /api/chat passes gateway.system_prompt, which never carried memory.
    Whether memory was empty at boot or not, that prompt is not edited."""
    for boot_memory in ("", "BOOT-FACT\n"):
        (home / "MEMORY.md").write_text(boot_memory, encoding="utf-8")
        _, boot_section = _daemon_boot(tmp_path)
        (home / "MEMORY.md").write_text("LATER-FACT\n", encoding="utf-8")
        provider = _Scripted()
        loop = _gateway_loop(provider, tmp_path, boot_section)
        asyncio.run(loop.run_async(system_prompt="You are Prometheus.", user_message="hi"))
        assert provider.requests[-1].system_prompt == "You are Prometheus."


def test_a_read_error_keeps_the_prompt(home, tmp_path) -> None:
    (home / "MEMORY.md").write_text("BOOT-FACT\n", encoding="utf-8")
    boot_prompt, boot_section = _daemon_boot(tmp_path)

    def boom() -> str:
        raise OSError("disk went away")

    provider = _Scripted()
    ctx = LoopContext(
        provider=provider, model="stub", system_prompt=boot_prompt, max_tokens=64,
        cwd=tmp_path, boot_memory_prompt=boot_section, memory_prompt_builder=boom,
    ).seal()
    _web_turn(ctx)
    assert provider.requests[-1].system_prompt == boot_prompt


def test_without_a_builder_nothing_is_read(home, tmp_path) -> None:
    """Gym, evals, subagents, coding mode: no builder, byte-identical prompts."""
    boot_prompt = build_runtime_system_prompt(cwd=str(tmp_path), config={}, skills=[])
    (home / "MEMORY.md").write_text("LATER-FACT\n", encoding="utf-8")
    provider = _Scripted()
    _web_turn(LoopContext(provider=provider, model="stub", system_prompt=boot_prompt,
                          max_tokens=64, cwd=tmp_path).seal())
    assert provider.requests[-1].system_prompt == boot_prompt


def test_explicit_empty_memory_content_means_no_section(home, tmp_path) -> None:
    """"" is "no memory", not "read the files": the daemon's one read decides."""
    (home / "MEMORY.md").write_text("ON-DISK-FACT\n", encoding="utf-8")
    prompt = build_runtime_system_prompt(cwd=str(tmp_path), config={}, memory_content="")
    assert "ON-DISK-FACT" not in prompt and _MEMORY_ETIQUETTE not in prompt
    assert "ON-DISK-FACT" in build_runtime_system_prompt(cwd=str(tmp_path), config={})


# --------------------------------------------------------------------------- #
# wiring: the daemon and the CLI actually pass it, from one read
# --------------------------------------------------------------------------- #

def _calls(path: Path, names: tuple[str, ...]) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) or getattr(n.func, "attr", None)) in names
    ]


def _kw(call: ast.Call) -> dict[str, str]:
    return {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}


def test_both_daemon_loops_refresh_memory() -> None:
    calls = _calls(SRC / "daemon.py", ("AgentLoop", "LoopContext"))
    assert {getattr(c.func, "id", None) for c in calls} == {"AgentLoop", "LoopContext"}
    for call in calls:
        kw = _kw(call)
        assert kw.get("memory_prompt_builder") == "_memory_section", ast.unparse(call.func)
        assert kw.get("boot_memory_prompt") == "_boot_memory_prompt", ast.unparse(call.func)


def test_every_daemon_boot_prompt_is_built_from_the_one_read() -> None:
    """If a build re-read the files, the section it carries could differ from
    _boot_memory_prompt, and the swap would find nothing to replace."""
    builds = _calls(SRC / "daemon.py", ("build_runtime_system_prompt",))
    assert len(builds) >= 4  # telegram, slack, discord, web
    for call in builds:
        assert _kw(call).get("memory_content") == "_boot_memory_content", ast.unparse(call)


def test_the_cli_repl_refreshes_memory() -> None:
    (ctx,) = _calls(SRC / "__main__.py", ("LoopContext",))
    kw = _kw(ctx)
    assert kw.get("memory_prompt_builder") == "memory_section"
    assert kw.get("boot_memory_prompt") == "memory_section(boot_memory_content)"
    (build,) = [c for c in _calls(SRC / "__main__.py", ("build_system_prompt",))
                if getattr(c.func, "id", None) == "build_system_prompt"]
    assert _kw(build).get("memory_content") == "boot_memory_content"
