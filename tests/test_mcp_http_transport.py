"""A tool server reached over HTTP connects, lists its tools and answers calls.

``url:`` servers used to resolve and then fail with "HTTP/SSE transport not yet
implemented". Both MCP HTTP transports now connect: ``transport:
streamable-http`` and the default SSE. Driven against a real MCP server on
loopback (the SDK's own server, served by uvicorn), not a mock, so what is
pinned is the wire.

The stored ``headers`` (typically a bearer for that server) ride every request
to it, and never come back out: not in the status, the catalog's launch summary,
a failure's detail or the log. The store refuses a header value that could
split a header, and a ``url`` that is not http(s). ``allowed_tools`` holds at
the call, as it does for a stdio server.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator

import pytest

pytest.importorskip("mcp")
pytest.importorskip("uvicorn")

from prometheus.mcp.runtime import McpRuntime  # noqa: E402
from prometheus.mcp.store import McpServerStore, McpStoreError  # noqa: E402
from tests.support.mcp_http_server import McpHttpServer  # noqa: E402

CANARY = "Bearer canary-7f3a91e0"


@pytest.fixture(params=["streamable-http", "sse"])
def server(request) -> Iterator[McpHttpServer]:
    with McpHttpServer(request.param) as srv:
        yield srv


def _config(server: McpHttpServer) -> dict:
    return server.config(Authorization=CANARY)


@pytest.mark.asyncio
async def test_an_http_server_connects_lists_and_answers(server, caplog):
    caplog.set_level(logging.DEBUG, logger="prometheus.mcp")
    runtime = McpRuntime({"remote": _config(server)})
    try:
        await runtime.connect_all()
        status = {s.name: s for s in runtime.list_statuses()}["remote"]
        assert status.state == "connected", status.detail
        assert sorted(t.tool_name for t in runtime.list_tools()) == ["echo", "whoami"]
        assert await runtime.call_tool("remote", "echo", {"text": "hi"}) == "echo: hi"
        assert await runtime.probe("remote") is True
    finally:
        await runtime.disconnect_server("remote", forget=True)

    assert server.seen_auth and set(server.seen_auth) == {CANARY}, (
        "the stored header must ride every request to the server")
    summary = runtime.get_catalog().servers.get("remote")
    assert summary is None or CANARY not in summary.launch_summary
    assert CANARY not in caplog.text


@pytest.mark.asyncio
async def test_a_tool_outside_allowed_tools_is_refused_at_the_call(server):
    # Same rule as for a stdio server: the allowlist hides the tool from the
    # catalog AND refuses it by name, before anything is sent to the server.
    runtime = McpRuntime({"remote": {**_config(server), "allowed_tools": ["something_else"]}})
    try:
        await runtime.connect_all()
        assert runtime.list_tools() == []
        before = len(server.seen_auth)
        with pytest.raises(PermissionError, match="allowed_tools"):
            await runtime.call_tool("remote", "echo", {"text": "hi"})
        assert len(server.seen_auth) == before, "the refused call still reached the server"
    finally:
        await runtime.disconnect_server("remote", forget=True)


@pytest.mark.asyncio
async def test_an_unreachable_server_fails_without_echoing_its_header(caplog):
    caplog.set_level(logging.DEBUG, logger="prometheus.mcp")
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    runtime = McpRuntime({"gone": {
        "url": f"http://127.0.0.1:{port}/mcp", "transport": "streamable-http",
        "headers": {"Authorization": CANARY}, "connectionTimeoutMs": 3000,
    }})
    await runtime.connect_all()
    status = {s.name: s for s in runtime.list_statuses()}["gone"]
    assert status.state == "failed"
    assert "not yet implemented" not in (status.detail or "")
    assert CANARY not in (status.detail or "") and CANARY not in caplog.text


# ── the store ────────────────────────────────────────────────────────


@pytest.mark.parametrize("url", ["ftp://host/mcp", "file:///etc/passwd", "http://host/mcp\nX: y", "not a url"])
def test_the_store_refuses_a_url_that_is_not_http(url):
    with pytest.raises(McpStoreError):
        McpServerStore.validate("remote", {"url": url})


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer x\r\nX-Injected: 1"},
    {"Bad Name": "v"},
    {"Authorization": 5},
    ["Authorization", "v"],
])
def test_the_store_refuses_headers_that_could_split_a_request(headers):
    with pytest.raises(McpStoreError):
        McpServerStore.validate("remote", {"url": "https://host/mcp", "headers": headers})


def test_the_store_accepts_a_plain_http_server():
    definition = {"url": "https://host/mcp", "transport": "streamable-http",
                  "headers": {"Authorization": "Bearer x"}}
    assert McpServerStore.validate("remote", dict(definition)) == definition
