"""`POST /api/chat` gives the model the daemon's runtime system prompt.

THE DEFECT
----------
The handler sent ``config["gateway"]["system_prompt"]``, falling back to the
literal "You are Prometheus, a sovereign AI agent. Be concise and helpful."
Every other surface sends the prompt ``build_runtime_system_prompt`` assembles
at boot: SOUL.md, AGENTS.md, the environment, the skills, the project files and
the "# Memory" section (MEMORY.md + USER.md). Telegram, Slack and Discord get it
through ``AgentLoop.run_async``, and the web bridge through its shared
``LoopContext``. This route got none of it.

On the shipped template the key holds a placeholder, so the model's whole
system prompt was the line "# see docs; overridden in live config".

Both lines date to the initial commit (cfebf6c), and no commit since has
touched them. The literal is the same one ``__main__.build_system_prompt``
falls back to when the builder raises. The docstring said the route "mirrors
Telegram dispatch", and in this it did not.

THE FIX
-------
The route reads the prompt held by the web bridge's shared ``LoopContext``,
which ``/api/chat/send`` runs on and the OpenAI-compatible route already reads
(``web/openai_api.py``), so the chat routes start from one prompt. With no
bridge wired it answers 503 rather than guess a prompt, as those two routes do.

The tests marked "was wrong" failed before the fix, for the reason in their
docstrings.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.context.prompt_assembler import (  # noqa: E402
    _load_memory_and_user,
    build_runtime_system_prompt,
    memory_section,
)
from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY  # noqa: E402
from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.providers.base import (  # noqa: E402
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.web.server import create_app  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "config" / "prometheus.yaml.default"

SOUL_MARKER = "soul-marker-7f3a: you answer in haiku"
MEMORY_MARKER = "memory-marker-91c2: the deploy box is called lantern"
USER_MARKER = "user-marker-4be0: prefers metric units"


class _RecordingProvider(ModelProvider):
    """Answers every call with one text round; records each request."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        msg = ConversationMessage(role="assistant", content=[TextBlock(text="ok")])
        yield ApiTextDeltaEvent(text="ok")
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(), stop_reason="stop"
        )


def _template_gateway_prompt() -> str:
    """What a fresh install's config holds for ``gateway.system_prompt``."""
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))["gateway"]["system_prompt"]


@pytest.fixture
def runtime_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The prompt the daemon builds at boot, from a config dir with a soul and memory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    (config_dir / "SOUL.md").write_text(SOUL_MARKER + "\n", encoding="utf-8")
    (config_dir / "MEMORY.md").write_text(MEMORY_MARKER + "\n", encoding="utf-8")
    (config_dir / "USER.md").write_text(USER_MARKER + "\n", encoding="utf-8")
    workdir = tmp_path / "work"
    workdir.mkdir()
    return build_runtime_system_prompt(cwd=str(workdir), config={})


def _daemon_app(runtime_prompt: str, provider: _RecordingProvider):
    """The app as ``web/launcher.launch_web`` wires it: ``create_app`` with the
    daemon's AgentLoop, then the bridge, holding the shared web LoopContext, on
    ``app.state.ws_bridge``. The config carries the template's gateway prompt,
    as a config copied from the template does."""
    loop_context = LoopContext(
        provider=provider, model="test", system_prompt=runtime_prompt,
        max_tokens=256, session_id="web",
    )
    app = create_app(
        {"gateway": {"system_prompt": _template_gateway_prompt()}},
        session_mgr=SessionManager(),
        agent_loop=AgentLoop(provider=provider, model="test", max_tokens=256),
    )
    app.state.ws_bridge = SimpleNamespace(loop_context=loop_context)
    return app, loop_context


def test_the_model_receives_the_runtime_prompt(runtime_prompt: str) -> None:
    """Was wrong: the model received "# see docs; overridden in live config"."""
    provider = _RecordingProvider()
    app, _ = _daemon_app(runtime_prompt, provider)

    response = TestClient(app).post("/api/chat", json={"session_id": "s1", "content": "hi"})

    assert response.status_code == 200, response.json()
    assert len(provider.requests) == 1
    received = provider.requests[0].system_prompt
    assert received != _template_gateway_prompt()
    assert "sovereign AI agent. Be concise and helpful" not in received
    for marker in (SOUL_MARKER, MEMORY_MARKER, USER_MARKER):
        assert marker in received, f"{marker!r} is missing from the prompt /api/chat sent"
    assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in received
    assert "# Memory" in received


def test_the_two_chat_routes_send_the_same_prompt(runtime_prompt: str) -> None:
    """Was wrong: the routes disagreed. ``/api/chat/send`` runs ``run_loop`` on
    the shared context (``ws_server._run_agent_locked``); ``/api/chat`` runs
    ``AgentLoop.run_async``. With no per-run rewrites wired, the model must be
    handed byte-identical prompts by both."""
    provider = _RecordingProvider()
    app, loop_context = _daemon_app(runtime_prompt, provider)

    TestClient(app).post("/api/chat", json={"session_id": "s1", "content": "hi"})

    async def _bridge_turn() -> None:
        async for _event, _usage in run_loop(
            loop_context, [ConversationMessage.from_user_text("hi")], session_id="s2",
        ):
            pass

    asyncio.run(_bridge_turn())

    assert len(provider.requests) == 2
    via_api_chat, via_bridge = (r.system_prompt for r in provider.requests)
    assert via_api_chat == via_bridge


def test_a_memory_edit_reaches_the_next_api_chat_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-run memory refresh (#679) reaches this route now that it sends
    the runtime prompt. Wired as daemon.py wires it: the files are read ONCE,
    the boot prompt is built from that read, and the daemon's AgentLoop is
    told the section to swap. Before this PR the route's prompt held no
    "# Memory" section, so the refresh had nothing to replace."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    (config_dir / "MEMORY.md").write_text("boot-fact: the box is lantern\n", encoding="utf-8")
    boot_content = _load_memory_and_user()
    runtime = build_runtime_system_prompt(
        cwd=str(tmp_path), config={}, memory_content=boot_content,
    )
    provider = _RecordingProvider()
    app = create_app(
        {},
        session_mgr=SessionManager(),
        agent_loop=AgentLoop(
            provider=provider, model="test", max_tokens=256,
            boot_memory_prompt=memory_section(boot_content),
            memory_prompt_builder=memory_section,
        ),
    )
    app.state.ws_bridge = SimpleNamespace(loop_context=LoopContext(
        provider=provider, model="test", system_prompt=runtime, max_tokens=256,
        session_id="web",
    ))
    client = TestClient(app)

    assert client.post("/api/chat", json={"session_id": "s1", "content": "a"}).status_code == 200
    assert "boot-fact: the box is lantern" in provider.requests[-1].system_prompt

    (config_dir / "MEMORY.md").write_text("later-fact: the box is beacon\n", encoding="utf-8")
    assert client.post("/api/chat", json={"session_id": "s1", "content": "b"}).status_code == 200
    second = provider.requests[-1].system_prompt
    assert "later-fact: the box is beacon" in second
    assert "boot-fact" not in second


def test_no_bridge_is_a_503_not_a_made_up_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was wrong: with no runtime prompt in reach the route invented one. It now
    refuses before touching the session, as ``/api/chat/send`` does when no
    bridge is wired. The launcher always wires one, so only a bare
    ``create_app()`` gets here."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    provider = _RecordingProvider()
    sessions = SessionManager()
    app = create_app(
        {"gateway": {"system_prompt": "a configured prompt"}},
        session_mgr=sessions,
        agent_loop=AgentLoop(provider=provider, model="test", max_tokens=256),
    )

    response = TestClient(app).post("/api/chat", json={"session_id": "s1", "content": "hi"})

    assert response.status_code == 503
    assert "ws_bridge not wired" in response.json()["error"]
    assert provider.requests == []
    assert sessions.get_or_create("web:s1").get_messages() == [], (
        "the refused message was still added to the session"
    )
