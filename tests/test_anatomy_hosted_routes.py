"""This machine's layout reaches a hosted model only when asked for (Will, 2026-10-23).

On a computer-use test rig, the first reply from a hosted model listed the box's GPU and its
tailnet peers by name: the anatomy section (hardware, services, network, peers) went into every
system prompt, whichever model served the turn. Now:

- a LOCAL route (llama.cpp, Ollama, LM Studio, vLLM) carries the section, unchanged;
- a HOSTED route gets a one-line pointer to the ``anatomy`` tool in its place, so the layout
  leaves the machine only when a task calls for it;
- ``anatomy.include_in_hosted_prompt: true`` (a literal true) sends the full section to hosted
  routes too;
- a hosted FALLBACK for a local primary gets the same treatment as a hosted route.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from prometheus.config.profiles import AgentProfile
from prometheus.context.prompt_assembler import (
    INFRASTRUCTURE_END,
    INFRASTRUCTURE_ON_DEMAND,
    INFRASTRUCTURE_START,
    build_runtime_system_prompt,
    include_infrastructure_in_hosted_prompt,
    infrastructure_for_route,
    is_hosted_provider,
)
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"

#: What a scan writes, with the kinds of names that must not leave the box by default.
ANATOMY = """# Anatomy

## Active Configuration
- Host: box-one (10.9.8.7)
- GPU: Example GPU 24 GB
- Tailnet peers: peer-alpha, peer-bravo
- Workspace: /home/someone/projects

## Projects
- project-one at /home/someone/projects/one
"""
SECRET_BITS = ("box-one", "10.9.8.7", "peer-alpha", "peer-bravo", "/home/someone")


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "prom-home"
    h.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(h))
    (h / "ANATOMY.md").write_text(ANATOMY)
    return h


def _boot(cwd: Path, profile=None) -> str:
    return build_runtime_system_prompt(cwd=str(cwd), config={}, skills=[], profile=profile)


@pytest.mark.parametrize("name,hosted", [
    ("llama_cpp", False), ("ollama", False), ("lm_studio", False), ("vllm", False),
    ("LLAMA_CPP", False),
    ("anthropic", True), ("openai", True), ("qwen", True), ("openrouter", True),
    ("gemini", True), ("", True), (None, True), ("stub", True),
])
def test_which_routes_are_hosted(name, hosted):
    assert is_hosted_provider(name) is hosted


@pytest.mark.parametrize("config,on", [
    ({}, False), ({"anatomy": {}}, False),
    ({"anatomy": {"include_in_hosted_prompt": False}}, False),
    ({"anatomy": {"include_in_hosted_prompt": "true"}}, False),
    ({"anatomy": {"include_in_hosted_prompt": 1}}, False),
    ({"anatomy": {"include_in_hosted_prompt": True}}, True),
])
def test_only_a_literal_true_sends_the_layout_to_hosted_routes(config, on):
    assert include_infrastructure_in_hosted_prompt(config) is on


@pytest.mark.parametrize("profile", [None, AgentProfile(name="full")], ids=["toggles", "profile"])
def test_the_boot_prompt_fences_the_section_on_both_build_paths(home, tmp_path, profile):
    prompt = _boot(tmp_path, profile)
    assert prompt.count(INFRASTRUCTURE_START) == 1 and prompt.count(INFRASTRUCTURE_END) == 1
    assert "peer-alpha" in prompt, "a local route must still see the layout"


@pytest.mark.parametrize("profile", [None, AgentProfile(name="full")], ids=["toggles", "profile"])
def test_a_hosted_route_gets_a_pointer_not_the_layout(home, tmp_path, profile):
    prompt = _boot(tmp_path, profile)
    hosted = infrastructure_for_route(prompt, hosted=True)
    for bit in SECRET_BITS:
        assert bit not in hosted, bit
    assert INFRASTRUCTURE_ON_DEMAND in hosted and "`anatomy` tool" in hosted
    # Everything else is untouched: the prompt around the section is byte-identical.
    start, end = prompt.index(INFRASTRUCTURE_START), prompt.index(INFRASTRUCTURE_END) + len(INFRASTRUCTURE_END)
    assert hosted == prompt[:start] + INFRASTRUCTURE_ON_DEMAND + prompt[end:]
    assert infrastructure_for_route(hosted, hosted=True) == hosted, "idempotent"


def test_local_routes_and_the_setting_keep_the_layout(home, tmp_path):
    prompt = _boot(tmp_path)
    assert infrastructure_for_route(prompt, hosted=False) == prompt
    assert infrastructure_for_route(prompt, hosted=True, include_on_hosted=True) == prompt


def test_an_unterminated_section_is_left_alone():
    """Not a section this module wrote; cutting to the end would drop the rest of the prompt."""
    odd = f"before\n{INFRASTRUCTURE_START}\nsomething\nafter"
    assert infrastructure_for_route(odd, hosted=True) == odd


class _Recorder(ModelProvider):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


def _send(provider: _Recorder, prompt: str, cwd: Path, **kw) -> str:
    ctx = LoopContext(provider=provider, model="m", system_prompt=prompt, max_tokens=64, cwd=cwd, **kw).seal()

    async def go():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:s1"):
            pass
    asyncio.run(go())
    return provider.requests[-1].system_prompt


@pytest.mark.parametrize("provider_name", ["anthropic", "qwen", "openrouter"])
def test_what_a_hosted_model_is_sent_has_no_layout(home, tmp_path, provider_name):
    sent = _send(_Recorder(provider_name), _boot(tmp_path), tmp_path)
    for bit in SECRET_BITS:
        assert bit not in sent, f"{bit!r} reached {provider_name}"
    assert INFRASTRUCTURE_ON_DEMAND in sent


@pytest.mark.parametrize("provider_name", ["llama_cpp", "ollama"])
def test_what_a_local_model_is_sent_keeps_the_layout(home, tmp_path, provider_name):
    sent = _send(_Recorder(provider_name), _boot(tmp_path), tmp_path)
    assert "peer-alpha" in sent and INFRASTRUCTURE_ON_DEMAND not in sent


def test_the_setting_sends_the_layout_to_a_hosted_model(home, tmp_path):
    sent = _send(_Recorder("anthropic"), _boot(tmp_path), tmp_path, infrastructure_in_hosted_prompt=True)
    assert "peer-alpha" in sent


def test_a_session_routed_to_a_hosted_model_is_judged_by_its_route(home, tmp_path):
    """A /claude-style override names its provider on the context; that decides, not the class."""
    sent = _send(_Recorder("llama_cpp"), _boot(tmp_path), tmp_path, route_provider="anthropic")
    assert "peer-alpha" not in sent


def _calls_in(func_name: str, callee: str) -> int:
    tree = ast.parse((SRC / "engine" / "agent_loop.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == func_name)
    return sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == callee)


def test_a_hosted_fallback_is_treated_like_a_hosted_route():
    """_on_degrade rebinds the prompt the fallback request reads; both copies pass the rule."""
    assert _calls_in("_on_degrade", "prompt_for_route") == 2
    assert _calls_in("_on_degrade", "is_hosted_provider") == 1


def test_every_loop_the_daemon_and_cli_build_reads_the_setting():
    hits = 0
    for rel in ("daemon.py", "__main__.py"):
        tree = ast.parse((SRC / rel).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.keyword) and n.arg == "infrastructure_in_hosted_prompt":
                assert isinstance(n.value, ast.Call)
                assert getattr(n.value.func, "id", None) == "include_infrastructure_in_hosted_prompt"
                hits += 1
    assert hits == 3, "the web LoopContext, the gateway AgentLoop and the CLI loop"
