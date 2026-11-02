"""A chat request can hand a named MCP server headers for that turn only.

An app in front of Prometheus signs its own users in. To let a tool server act
as that user, the app passes a short-lived, user-bound token with the request,
instead of a standing credential left on the box::

    POST /api/chat  {"content": ..., "mcp_headers": {"<server>": {"Authorization": "Bearer ..."}}}

(``/v1/chat/completions`` takes the same ``mcp_headers`` as a Prometheus
extension.) Pinned here, against a real MCP server on loopback and the real
agent loop:

* the header reaches the NAMED server, on that turn's calls, and nowhere else;
* the next turn without it goes back to the server's stored headers;
* the value is never in the model's requests, the conversation, the reply or
  the log (a canary is searched for afterwards);
* a header for a server that is not configured, or that is not reached over
  HTTP, is refused with a 400 before the turn runs, never silently dropped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

import httpx  # noqa: E402

from prometheus.engine.agent_loop import AgentLoop, LoopContext  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.mcp.adapter import register_mcp_tools  # noqa: E402
from prometheus.mcp.runtime import McpRuntime  # noqa: E402
from prometheus.permissions.checker import Grant, SecurityGate  # noqa: E402
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider  # noqa: E402
from prometheus.tools.base import ToolRegistry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.mcp_http_server import McpHttpServer  # noqa: E402

CANARY = "Bearer turn-canary-5d1c0b"
STATIC = "Bearer stored-for-the-server"
TOKEN = "turn-headers-test-token"


class _CallsTools(ModelProvider):
    """Each turn: round 1 calls every named tool, round 2 answers. Keeps every request."""

    def __init__(self, tools: list[str]) -> None:
        self._suppress_thinking = True
        self.tools = tools
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        already_called = any(
            getattr(block, "type", "") == "tool_result"
            for msg in request.messages[-1:] for block in getattr(msg, "content", []) or []
        )
        content = ([TextBlock(text="done")] if already_called else
                   [ToolUseBlock(id=f"t{i}", name=name, input={}) for i, name in enumerate(self.tools)])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


@pytest.fixture
def servers() -> Iterator[tuple[McpHttpServer, McpHttpServer]]:
    with McpHttpServer("streamable-http") as remote, McpHttpServer("sse") as other:
        yield remote, other


async def _rig(remote: McpHttpServer, other: McpHttpServer, tools: list[str], tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    runtime = McpRuntime({
        "remote": remote.config(Authorization=STATIC),
        "other": other.config(Authorization=STATIC),
        "local-only": {"command": "true"},  # stdio: configured, never reached over HTTP
    })
    await runtime.connect_server("remote", runtime._server_configs["remote"])
    await runtime.connect_server("other", runtime._server_configs["other"])
    registry = ToolRegistry()
    names = register_mcp_tools(registry, runtime)
    assert names >= 2
    provider = _CallsTools(tools)
    gate = SecurityGate(grants=[Grant(kind="tool", value="", tool_name=t) for t in tools])
    sessions = SessionManager()
    app = create_app(
        {"web": {"api_token": TOKEN}}, session_mgr=sessions,
        agent_loop=AgentLoop(provider=provider, model="stub", max_tokens=128,
                             tool_registry=registry, permission_checker=gate),
    )
    app.state.mcp_runtime = runtime
    app.state.ws_bridge = SimpleNamespace(loop_context=LoopContext(
        provider=provider, model="stub", system_prompt="sys", max_tokens=128,
        tool_registry=registry, permission_checker=gate,
    ))
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                               headers={"Authorization": f"Bearer {TOKEN}"})
    return client, runtime, provider, sessions


async def _close(client, runtime) -> None:
    await client.aclose()
    for name in ("remote", "other"):
        await runtime.disconnect_server(name, forget=True)


@pytest.mark.asyncio
async def test_the_header_reaches_the_named_server_for_that_turn_only(servers, tmp_path, monkeypatch):
    remote, other = servers
    client, runtime, _provider, _sessions = await _rig(
        remote, other, ["mcp__remote__whoami", "mcp__other__whoami"], tmp_path, monkeypatch)
    try:
        remote.seen_auth.clear(), other.seen_auth.clear()
        r = await client.post("/api/chat", json={
            "session_id": "s1", "content": "go", "mcp_headers": {"remote": {"Authorization": CANARY}},
        })
        assert r.status_code == 200, r.text
        assert CANARY in remote.seen_auth, "the turn's header never reached the named server"
        assert CANARY not in other.seen_auth, "the header leaked to a server it was not for"

        remote.seen_auth.clear()
        r = await client.post("/api/chat", json={"session_id": "s1", "content": "again"})
        assert r.status_code == 200, r.text
        assert remote.seen_auth and CANARY not in remote.seen_auth, "the header outlived its turn"
        assert STATIC in remote.seen_auth, "a turn without headers uses the server's stored ones"
    finally:
        await _close(client, runtime)


@pytest.mark.asyncio
async def test_the_header_is_kept_out_of_the_model_the_conversation_and_the_log(
        servers, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    remote, other = servers
    client, runtime, provider, sessions = await _rig(
        remote, other, ["mcp__remote__whoami"], tmp_path, monkeypatch)
    try:
        r = await client.post("/api/chat", json={
            "session_id": "s1", "content": "go", "mcp_headers": {"remote": {"Authorization": CANARY}},
        })
        assert r.status_code == 200, r.text
        assert CANARY in remote.seen_auth
    finally:
        await _close(client, runtime)

    assert CANARY not in r.text
    sent_to_model = json.dumps([
        {"system": req.system_prompt, "tools": req.tools,
         "messages": [m.model_dump(mode="json") for m in req.messages]}
        for req in provider.requests
    ], default=str)
    assert CANARY not in sent_to_model, "the model's context carried the header"
    kept = json.dumps([m.model_dump(mode="json") for m in sessions.get_or_create("web:s1").get_messages()])
    assert CANARY not in kept, "the conversation kept the header"
    assert CANARY not in caplog.text, "the header was logged"


@pytest.mark.asyncio
@pytest.mark.parametrize("mcp_headers, why", [
    ({"nowhere": {"Authorization": CANARY}}, "nowhere"),
    ({"local-only": {"Authorization": CANARY}}, "local-only"),
    ({"remote": {"Authorization": "Bearer x\r\nX-Injected: 1"}}, "control characters"),
    ({"remote": "Bearer x"}, "mcp_headers"),
    (["remote"], "mcp_headers"),
])
async def test_a_header_that_cannot_be_delivered_is_refused_before_the_turn(
        servers, tmp_path, monkeypatch, mcp_headers, why):
    remote, other = servers
    client, runtime, provider, _sessions = await _rig(
        remote, other, ["mcp__remote__whoami"], tmp_path, monkeypatch)
    try:
        r = await client.post("/api/chat", json={"session_id": "s1", "content": "go", "mcp_headers": mcp_headers})
    finally:
        await _close(client, runtime)
    assert r.status_code == 400, r.text
    assert why in r.text
    assert CANARY not in r.text, "the refusal echoed the header value"
    assert provider.requests == [], "the turn ran anyway"


@pytest.mark.asyncio
async def test_v1_takes_the_same_turn_headers(servers, tmp_path, monkeypatch):
    remote, other = servers
    client, runtime, _provider, _sessions = await _rig(
        remote, other, ["mcp__remote__whoami"], tmp_path, monkeypatch)
    try:
        remote.seen_auth.clear()
        r = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "go"}],
            "mcp_headers": {"remote": {"Authorization": CANARY}},
        })
        assert r.status_code == 200, r.text
        assert CANARY in remote.seen_auth
        assert CANARY not in r.text

        r = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "go"}],
            "mcp_headers": {"nowhere": {"Authorization": CANARY}},
        })
        assert r.status_code == 400 and "nowhere" in r.text
    finally:
        await _close(client, runtime)
