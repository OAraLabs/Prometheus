"""Tests for LSP client (Sprint 20: JSON-RPC over stdin/stdout)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.lsp.client import (
    Diagnostic,
    DocumentSymbol,
    HoverInfo,
    LSPClient,
    LSPError,
    Location,
    _parse_locations,
    _parse_symbols,
    _parse_workspace_edit,
    _path_to_uri,
    _uri_to_path,
)
from prometheus.lsp.languages import LSPServerDef


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def server_def():
    return LSPServerDef(
        language_id="python",
        extensions=[".py"],
        command=["pyright-langserver", "--stdio"],
        root_markers=["pyproject.toml", ".git"],
    )


@pytest.fixture
def mock_process():
    """Create a mock asyncio subprocess for testing."""
    proc = AsyncMock()
    proc.pid = 12345
    proc.returncode = None
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdout = AsyncMock()
    proc.stderr = AsyncMock()
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()
    return proc


# ------------------------------------------------------------------
# URI helpers
# ------------------------------------------------------------------

def test_path_to_uri():
    uri = _path_to_uri("/home/will/test.py")
    assert uri == "file:///home/will/test.py"


def test_uri_to_path():
    path = _uri_to_path("file:///home/will/test.py")
    assert path == "/home/will/test.py"


def test_uri_to_path_passthrough():
    path = _uri_to_path("/home/will/test.py")
    assert path == "/home/will/test.py"


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

def test_location_str():
    loc = Location(path="/src/main.py", line=42, col=5)
    assert str(loc) == "/src/main.py:42:5"


def test_diagnostic_severity_str():
    d = Diagnostic(path="f.py", line=1, col=1, severity=1, message="bad")
    assert d.severity_str == "ERROR"
    d2 = Diagnostic(path="f.py", line=1, col=1, severity=2, message="warn")
    assert d2.severity_str == "WARNING"
    d3 = Diagnostic(path="f.py", line=1, col=1, severity=3, message="info")
    assert d3.severity_str == "INFO"
    d4 = Diagnostic(path="f.py", line=1, col=1, severity=4, message="hint")
    assert d4.severity_str == "HINT"


def test_diagnostic_str():
    d = Diagnostic(path="f.py", line=10, col=1, severity=1, message="Type error")
    assert str(d) == "ERROR L10: Type error"


def test_hover_info_str():
    h = HoverInfo(contents="def foo() -> int")
    assert str(h) == "def foo() -> int"


def test_document_symbol_str():
    s = DocumentSymbol(
        name="MyClass", kind=5, range_start_line=10, range_end_line=50,
        detail="(BaseModel)",
    )
    assert "Class" in str(s)
    assert "MyClass" in str(s)
    assert "(BaseModel)" in str(s)


# ------------------------------------------------------------------
# Response parsers
# ------------------------------------------------------------------

def test_parse_locations_none():
    assert _parse_locations(None) == []


def test_parse_locations_single_location():
    result = {
        "uri": "file:///src/main.py",
        "range": {"start": {"line": 9, "character": 4}},
    }
    locs = _parse_locations(result)
    assert len(locs) == 1
    assert locs[0].path == "/src/main.py"
    assert locs[0].line == 10
    assert locs[0].col == 5


def test_parse_locations_list():
    result = [
        {"uri": "file:///a.py", "range": {"start": {"line": 0, "character": 0}}},
        {"uri": "file:///b.py", "range": {"start": {"line": 5, "character": 2}}},
    ]
    locs = _parse_locations(result)
    assert len(locs) == 2
    assert locs[0].line == 1
    assert locs[1].line == 6


def test_parse_locations_location_link():
    result = [
        {
            "targetUri": "file:///target.py",
            "targetRange": {"start": {"line": 14, "character": 0}},
            "targetSelectionRange": {"start": {"line": 14, "character": 4}},
        },
    ]
    locs = _parse_locations(result)
    assert len(locs) == 1
    assert locs[0].path == "/target.py"
    assert locs[0].line == 15
    assert locs[0].col == 5


def test_parse_symbols_hierarchical():
    result = [
        {
            "name": "MyClass",
            "kind": 5,
            "range": {"start": {"line": 0}, "end": {"line": 20}},
            "children": [
                {
                    "name": "method",
                    "kind": 6,
                    "range": {"start": {"line": 5}, "end": {"line": 10}},
                    "children": [],
                },
            ],
        },
    ]
    symbols = _parse_symbols(result)
    assert len(symbols) == 1
    assert symbols[0].name == "MyClass"
    assert symbols[0].kind_str == "Class"
    assert len(symbols[0].children) == 1
    assert symbols[0].children[0].name == "method"


def test_parse_symbols_flat():
    result = [
        {
            "name": "foo",
            "kind": 12,
            "location": {
                "uri": "file:///test.py",
                "range": {"start": {"line": 3}, "end": {"line": 8}},
            },
            "containerName": "module",
        },
    ]
    symbols = _parse_symbols(result)
    assert len(symbols) == 1
    assert symbols[0].name == "foo"
    assert symbols[0].kind_str == "Function"


def test_parse_workspace_edit_changes():
    result = {
        "changes": {
            "file:///a.py": [
                {"range": {"start": {"line": 4}, "end": {"line": 4}}, "newText": "new_name"},
                {"range": {"start": {"line": 10}, "end": {"line": 10}}, "newText": "new_name"},
            ],
        },
    }
    edits = _parse_workspace_edit(result)
    assert "/a.py" in edits
    assert len(edits["/a.py"]) == 2


def test_parse_workspace_edit_document_changes():
    result = {
        "documentChanges": [
            {
                "textDocument": {"uri": "file:///b.py", "version": 1},
                "edits": [
                    {"range": {"start": {"line": 0}, "end": {"line": 0}}, "newText": "x"},
                ],
            },
        ],
    }
    edits = _parse_workspace_edit(result)
    assert "/b.py" in edits
    assert len(edits["/b.py"]) == 1


# ------------------------------------------------------------------
# Client — initialize handshake (mocked)
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_client_sends_initialize(server_def, mock_process, tmp_path):
    """Client sends initialize and receives a response."""
    # Simulate server responding to initialize
    init_response = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"capabilities": {"textDocumentSync": 1}},
    }).encode("utf-8")
    header = f"Content-Length: {len(init_response)}\r\n\r\n".encode("ascii")

    reader = asyncio.StreamReader()
    reader.feed_data(header + init_response)

    mock_process.stdout = reader

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        client = LSPClient(server_def, tmp_path)
        # Start will hang waiting for initialize response — we feed it above
        # but _reader_loop will start reading. We need to be careful with the event loop.
        # Instead of full start(), test the request/response mechanics directly.
        client._process = mock_process
        client._process.stdout = reader
        client._reader_task = asyncio.create_task(client._reader_loop())

        # Send a request and verify the response comes back
        result = await client._send_request("initialize", {"processId": 123})
        assert result == {"capabilities": {"textDocumentSync": 1}}

        client._reader_task.cancel()
        try:
            await client._reader_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_client_handles_error_response(server_def, mock_process, tmp_path):
    """Client raises LSPError when server returns an error."""
    error_response = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32600, "message": "Invalid Request"},
    }).encode("utf-8")
    header = f"Content-Length: {len(error_response)}\r\n\r\n".encode("ascii")

    reader = asyncio.StreamReader()
    reader.feed_data(header + error_response)

    client = LSPClient(server_def, tmp_path)
    client._process = mock_process
    client._process.stdout = reader
    client._reader_task = asyncio.create_task(client._reader_loop())

    with pytest.raises(LSPError, match="Invalid Request"):
        await client._send_request("bad_method", {})

    client._reader_task.cancel()
    try:
        await client._reader_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_client_stores_diagnostics(server_def, mock_process, tmp_path):
    """Client stores diagnostics from publishDiagnostics notifications."""
    notification = json.dumps({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": "file:///test/main.py",
            "diagnostics": [
                {
                    "range": {"start": {"line": 9, "character": 4}, "end": {"line": 9, "character": 10}},
                    "severity": 1,
                    "message": "Type 'str' not assignable to 'int'",
                    "source": "pyright",
                },
                {
                    "range": {"start": {"line": 14, "character": 0}, "end": {"line": 14, "character": 5}},
                    "severity": 2,
                    "message": "Unused import",
                    "source": "pyright",
                },
            ],
        },
    }).encode("utf-8")
    header = f"Content-Length: {len(notification)}\r\n\r\n".encode("ascii")

    reader = asyncio.StreamReader()
    reader.feed_data(header + notification)
    reader.feed_eof()

    client = LSPClient(server_def, tmp_path)
    client._process = mock_process
    client._process.stdout = reader
    client._reader_task = asyncio.create_task(client._reader_loop())
    await client._reader_task  # runs until EOF

    diags = client._diagnostics.get("/test/main.py", [])
    assert len(diags) == 2
    assert diags[0].severity == 1
    assert diags[0].line == 10
    assert "str" in diags[0].message
    assert diags[1].severity == 2


@pytest.mark.asyncio
async def test_client_handles_server_crash(server_def, mock_process, tmp_path):
    """Client handles server process dying gracefully."""
    reader = asyncio.StreamReader()
    reader.feed_eof()  # simulate crash

    client = LSPClient(server_def, tmp_path)
    client._process = mock_process
    client._process.stdout = reader
    client._reader_task = asyncio.create_task(client._reader_loop())

    # Should complete without error
    await client._reader_task
    assert len(client._pending) == 0


@pytest.mark.asyncio
async def test_client_shutdown_sends_messages(server_def, mock_process, tmp_path):
    """Client shutdown sends shutdown + exit messages."""
    # Prepare response for shutdown request
    shutdown_resp = json.dumps({
        "jsonrpc": "2.0", "id": 1, "result": None,
    }).encode("utf-8")
    header = f"Content-Length: {len(shutdown_resp)}\r\n\r\n".encode("ascii")

    reader = asyncio.StreamReader()
    reader.feed_data(header + shutdown_resp)
    reader.feed_eof()

    client = LSPClient(server_def, tmp_path)
    client._process = mock_process
    client._process.stdout = reader
    client._initialized = True
    client._reader_task = asyncio.create_task(client._reader_loop())

    await client.stop()

    # Verify stdin.write was called (for shutdown request + exit notification)
    assert mock_process.stdin.write.call_count >= 1
    assert mock_process.kill.called


@pytest.mark.asyncio
async def test_client_server_request_gets_null_reply(server_def, mock_process, tmp_path):
    """Client replies with null to server-to-client requests."""
    server_request = json.dumps({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "window/workDoneProgress/create",
        "params": {"token": "abc"},
    }).encode("utf-8")
    header = f"Content-Length: {len(server_request)}\r\n\r\n".encode("ascii")

    reader = asyncio.StreamReader()
    reader.feed_data(header + server_request)
    reader.feed_eof()

    client = LSPClient(server_def, tmp_path)
    client._process = mock_process
    client._process.stdout = reader
    client._reader_task = asyncio.create_task(client._reader_loop())
    await client._reader_task

    # Should have written a response back
    calls = mock_process.stdin.write.call_args_list
    assert len(calls) >= 1
    # Find the response in the written data
    for call in calls:
        data = call[0][0]
        if b'"id": 99' in data or b'"id":99' in data:
            assert b'"result": null' in data or b'"result":null' in data
            break
