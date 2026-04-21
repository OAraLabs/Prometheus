"""Tests for Sprint 12: MCP transport resolution."""

from __future__ import annotations

import pytest

from prometheus.mcp.transport import (
    DEFAULT_CONNECTION_TIMEOUT_MS,
    ResolvedHttpTransport,
    ResolvedStdioTransport,
    resolve_http_config,
    resolve_stdio_config,
    resolve_transport,
)


class TestStdioTransport:
    def test_resolves_basic_command(self):
        raw = {"command": "npx", "args": ["-y", "@upstash/context7-mcp"]}
        result = resolve_stdio_config(raw)
        assert result is not None
        assert result.kind == "stdio"
        assert result.command == "npx"
        assert result.args == ["-y", "@upstash/context7-mcp"]

    def test_returns_none_without_command(self):
        raw = {"url": "http://localhost:3000"}
        assert resolve_stdio_config(raw) is None

    def test_returns_none_for_empty_command(self):
        raw = {"command": "  "}
        assert resolve_stdio_config(raw) is None

    def test_strips_command(self):
        raw = {"command": "  npx  "}
        result = resolve_stdio_config(raw)
        assert result.command == "npx"

    def test_env_dict(self):
        raw = {"command": "node", "env": {"KEY": "val", "NUM": 42}}
        result = resolve_stdio_config(raw)
        assert result.env == {"KEY": "val", "NUM": "42"}

    def test_cwd(self):
        raw = {"command": "node", "cwd": "/tmp/project"}
        result = resolve_stdio_config(raw)
        assert result.cwd == "/tmp/project"

    def test_custom_timeout(self):
        raw = {"command": "node", "connectionTimeoutMs": 60000}
        result = resolve_stdio_config(raw)
        assert result.timeout_ms == 60000

    def test_invalid_timeout_uses_default(self):
        raw = {"command": "node", "connectionTimeoutMs": -1}
        result = resolve_stdio_config(raw)
        assert result.timeout_ms == DEFAULT_CONNECTION_TIMEOUT_MS

    def test_description(self):
        raw = {"command": "npx", "args": ["-y", "server"]}
        result = resolve_stdio_config(raw)
        assert result.description == "npx -y server"


class TestHttpTransport:
    def test_resolves_url(self):
        raw = {"url": "http://localhost:3000/mcp"}
        result = resolve_http_config(raw)
        assert result is not None
        assert result.kind == "http"
        assert result.url == "http://localhost:3000/mcp"
        assert result.transport_type == "sse"

    def test_returns_none_without_url(self):
        raw = {"command": "npx"}
        assert resolve_http_config(raw) is None

    def test_returns_none_for_empty_url(self):
        raw = {"url": "  "}
        assert resolve_http_config(raw) is None

    def test_rejects_non_http_scheme(self):
        raw = {"url": "ftp://evil.com"}
        assert resolve_http_config(raw) is None

    def test_streamable_http(self):
        raw = {"url": "http://localhost:3000"}
        result = resolve_http_config(raw, "streamable-http")
        assert result.transport_type == "streamable-http"

    def test_headers(self):
        raw = {"url": "http://localhost:3000", "headers": {"Authorization": "Bearer tok"}}
        result = resolve_http_config(raw)
        assert result.headers == {"Authorization": "Bearer tok"}

    def test_description_redacts_password(self):
        raw = {"url": "http://user:secret@host.com/path"}
        result = resolve_http_config(raw)
        assert "secret" not in result.description
        assert "***" in result.description


class TestResolveTransport:
    def test_stdio_takes_priority(self):
        """When both command and url present, stdio wins."""
        raw = {"command": "npx", "args": ["-y", "server"], "url": "http://localhost"}
        result = resolve_transport("test", raw)
        assert isinstance(result, ResolvedStdioTransport)

    def test_falls_back_to_sse(self):
        raw = {"url": "http://localhost:3000"}
        result = resolve_transport("test", raw)
        assert isinstance(result, ResolvedHttpTransport)
        assert result.transport_type == "sse"

    def test_streamable_http_when_requested(self):
        raw = {"url": "http://localhost:3000", "transport": "streamable-http"}
        result = resolve_transport("test", raw)
        assert isinstance(result, ResolvedHttpTransport)
        assert result.transport_type == "streamable-http"

    def test_returns_none_for_invalid_config(self):
        raw = {"foo": "bar"}
        assert resolve_transport("test", raw) is None
