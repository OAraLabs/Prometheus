"""No person in a tool's words, and no account name in a hosted model's paths (Will, 2026-10-25).

Two things reached every hosted turn:

- The brain-vault tool described itself as "Will's second brain, compiled from his full claude.ai
  and Claude Code history". It named a person, on every install with a vault, in the tool list a
  cloud tier sends in full. Tool text is now neutral, and this file pins it for EVERY tool, by
  instantiating each one and reading what the model is sent.
- The working directory, the documents root, a session workspace and its project files were
  absolute paths (``/home/<user>/…``). On a hosted route, paths under the home directory are
  written ``~/…``; the file tools and the gate expand ``~``, so nothing changes about what the
  model may touch. Local routes are unchanged.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import pkgutil
import re
from pathlib import Path

import pytest

import prometheus.tools.builtin as builtin
from prometheus.context.prompt_assembler import build_runtime_system_prompt, home_relative, prompt_for_route
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool

ROOT = Path(__file__).resolve().parents[1]

#: Possessives a tool may use: vendors and products, never people. Add a product here if a tool
#: needs one; a person's name never belongs in this list.
PRODUCT_POSSESSIVES = {"Kling's"}
POSSESSIVE = re.compile(r"\b[A-Z][a-z]+'s\b")
PRONOUN = re.compile(r"\b(?:he|she|his|her|hers|him|himself|herself)\b", re.I)
#: The maintainer's first name as a name (not the verb at a sentence's start: "Will it rain?").
MAINTAINER = re.compile(r"\bWill(?:'s)?\b(?! (?:it|be|the|not|you|this|that)\b)")


def _every_tool_text() -> dict[str, str]:
    out: dict[str, str] = {}
    for info in pkgutil.walk_packages(builtin.__path__, builtin.__name__ + "."):
        mod = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(cls, BaseTool) and cls is not BaseTool
                    and cls.__module__ == mod.__name__ and not inspect.isabstract(cls)):
                # The schema is built from class attributes; no tool needs its runtime wiring here.
                out[cls.name] = json.dumps(cls.__new__(cls).to_api_schema())
    return out


def test_the_check_reads_every_builtin_tool():
    texts = _every_tool_text()
    assert len(texts) >= 50, f"only {len(texts)} tools read; the walk missed modules"
    assert {"vault_search", "vault_read", "bash", "read_file"} <= set(texts)


@pytest.mark.parametrize("name", sorted(_every_tool_text()))
def test_no_tool_names_a_person(name):
    text = _every_tool_text()[name]
    assert not MAINTAINER.search(text), f"{name} names a person: {MAINTAINER.search(text).group(0)!r}"
    assert not PRONOUN.search(text), f"{name} refers to someone as {PRONOUN.search(text).group(0)!r}"
    bad = {m.group(0) for m in POSSESSIVE.finditer(text)} - PRODUCT_POSSESSIVES
    assert not bad, f"{name} uses a possessive name {sorted(bad)}; tools describe the owner, never a person"


def test_the_vault_describes_the_owners_vault():
    text = _every_tool_text()["vault_search"]
    assert "the owner's second brain" in text


@pytest.mark.parametrize("template", sorted((ROOT / "templates").glob("*.template")), ids=lambda p: p.name)
def test_no_bootstrap_template_names_a_person(template):
    """SOUL.md and AGENTS.md are written from these and go into every prompt."""
    text = template.read_text()
    assert not MAINTAINER.search(text) and not PRONOUN.search(text)


# ── home-relative paths ─────────────────────────────────────────────────────

@pytest.mark.parametrize("before,after", [
    ("- Working directory: /home/alex/projX", "- Working directory: ~/projX"),
    ("root is: /home/alex/.prometheus/documents", "root is: ~/.prometheus/documents"),
    ("## /home/alex/projX/PROMETHEUS.md (from projX/)", "## ~/projX/PROMETHEUS.md (from projX/)"),
    ("directory is `/home/alex/work`.", "directory is `~/work`."),
    ("home: /home/alex", "home: ~"),
    ("('/home/alex', \"/home/alex/x\")", "('~', \"~/x\")"),
    # A different account whose name merely starts the same is not this home.
    ("/home/alexandra/projX", "/home/alexandra/projX"),
    ("/srv/docs and /tmp/x", "/srv/docs and /tmp/x"),
])
def test_home_relative(before, after):
    assert home_relative(before, "/home/alex") == after


@pytest.mark.parametrize("home", ["/", "", "~"])
def test_a_home_that_names_nothing_changes_nothing(home):
    assert home_relative("/home/alex/x and /etc", home) == "/home/alex/x and /etc"


def test_only_a_hosted_route_is_rewritten():
    prompt = "- Working directory: /home/alex/projX"
    assert prompt_for_route(prompt, hosted=False, home="/home/alex") == prompt
    assert prompt_for_route(prompt, hosted=True, home="/home/alex") == "- Working directory: ~/projX"


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


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "home" / "alex"
    (h / "projX").mkdir(parents=True)
    (h / "projX" / "PROMETHEUS.md").write_text("Project notes.")
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(h / ".prometheus"))
    return h


def _sent(provider: _Recorder, home: Path) -> str:
    prompt = build_runtime_system_prompt(cwd=str(home / "projX"), config={}, skills=[])
    assert str(home) in prompt, "the boot prompt carries absolute paths (the case under test)"
    ctx = LoopContext(provider=provider, model="m", system_prompt=prompt, max_tokens=64,
                      cwd=home / "projX").seal()

    async def go():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:s1"):
            pass
    asyncio.run(go())
    return provider.requests[-1].system_prompt


@pytest.mark.parametrize("provider_name", ["anthropic", "qwen", "openrouter"])
def test_a_hosted_model_is_sent_no_absolute_home_path(home, provider_name):
    sent = _sent(_Recorder(provider_name), home)
    assert str(home) not in sent
    assert "- Working directory: ~/projX" in sent


@pytest.mark.parametrize("provider_name", ["llama_cpp", "ollama"])
def test_a_local_model_is_sent_the_prompt_unchanged(home, provider_name):
    sent = _sent(_Recorder(provider_name), home)
    assert f"- Working directory: {home}/projX" in sent
