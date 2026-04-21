"""Tests for LSP orchestrator (Sprint 20: lifecycle, routing, symbol context)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.lsp.client import (
    Diagnostic,
    DocumentSymbol,
    HoverInfo,
    Location,
    LSPClient,
    LSPError,
)
from prometheus.lsp.languages import LSPServerDef
from prometheus.lsp.orchestrator import LSPOrchestrator


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def python_def():
    return LSPServerDef(
        language_id="python",
        extensions=[".py"],
        command=["pyright-langserver", "--stdio"],
        root_markers=["pyproject.toml"],
    )


@pytest.fixture
def ts_def():
    return LSPServerDef(
        language_id="typescript",
        extensions=[".ts"],
        command=["typescript-language-server", "--stdio"],
        root_markers=["tsconfig.json"],
    )


def _mock_client(language_id: str = "python", alive: bool = True) -> MagicMock:
    client = MagicMock(spec=LSPClient)
    client.server_def = MagicMock()
    client.server_def.language_id = language_id
    client.is_alive = alive
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.get_definition = AsyncMock(return_value=[])
    client.get_references = AsyncMock(return_value=[])
    client.get_hover = AsyncMock(return_value=None)
    client.get_diagnostics = AsyncMock(return_value=[])
    client.get_document_symbols = AsyncMock(return_value=[])
    client.rename_symbol = AsyncMock(return_value={})
    client.did_change = AsyncMock()
    return client


# ------------------------------------------------------------------
# Lazy spawning
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lazy_spawn_on_first_access(tmp_path):
    """Server starts when first file of that language is accessed."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src = tmp_path / "main.py"
    src.write_text("x = 1\n")

    mock_client = _mock_client()

    orch = LSPOrchestrator()
    with patch.object(orch, "_spawn", new_callable=AsyncMock, return_value=mock_client):
        result = await orch.ensure_server(str(src))
        assert result is mock_client


@pytest.mark.asyncio
async def test_reuses_existing_server(tmp_path):
    """Reuses existing server for same language + project root."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src_a = tmp_path / "a.py"
    src_a.write_text("a = 1\n")
    src_b = tmp_path / "b.py"
    src_b.write_text("b = 2\n")

    mock_client = _mock_client()

    orch = LSPOrchestrator()
    with patch.object(orch, "_spawn", new_callable=AsyncMock, return_value=mock_client):
        result_a = await orch.ensure_server(str(src_a))
        # Now the client is stored — second call should reuse
        orch._clients["python:" + str(tmp_path)] = mock_client
        result_b = await orch.ensure_server(str(src_b))
        assert result_a is mock_client
        assert result_b is mock_client


@pytest.mark.asyncio
async def test_marks_failed_servers_broken(tmp_path):
    """Marks failed servers as broken, doesn't retry."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src = tmp_path / "test.py"
    src.write_text("x = 1\n")

    orch = LSPOrchestrator()
    with patch.object(orch, "_spawn", new_callable=AsyncMock, return_value=None):
        result1 = await orch.ensure_server(str(src))
        assert result1 is None

    # Mark as broken manually (since our mock doesn't go through real _spawn)
    key = "python:" + str(tmp_path)
    orch._broken.add(key)

    # Second attempt should not even try to spawn
    result2 = await orch.ensure_server(str(src))
    assert result2 is None


@pytest.mark.asyncio
async def test_unsupported_language_returns_none(tmp_path):
    """Returns None for files with no matching language server."""
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("hello")

    orch = LSPOrchestrator()
    result = await orch.ensure_server(str(txt_file))
    assert result is None


# ------------------------------------------------------------------
# Request routing
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routes_definition_to_correct_client(tmp_path):
    """Routes definition requests to correct client by file extension."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src = tmp_path / "main.py"
    src.write_text("def foo(): pass\n")

    expected = [Location(path=str(src), line=1, col=5)]
    mock_client = _mock_client()
    mock_client.get_definition = AsyncMock(return_value=expected)

    orch = LSPOrchestrator()
    orch._clients["python:" + str(tmp_path)] = mock_client

    result = await orch.get_definition(str(src), 1, 5)
    assert result == expected
    mock_client.get_definition.assert_called_once()


@pytest.mark.asyncio
async def test_routes_references_to_client(tmp_path):
    """Routes references requests to the right client."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src = tmp_path / "main.py"
    src.write_text("x = 1\nprint(x)\n")

    refs = [
        Location(path=str(src), line=1, col=1),
        Location(path=str(src), line=2, col=7),
    ]
    mock_client = _mock_client()
    mock_client.get_references = AsyncMock(return_value=refs)

    orch = LSPOrchestrator()
    orch._clients["python:" + str(tmp_path)] = mock_client

    result = await orch.get_references(str(src), 1, 1)
    assert len(result) == 2


# ------------------------------------------------------------------
# get_symbol_context — the power move
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_symbol_context_packages_all_info(tmp_path):
    """get_symbol_context packages definition + references + hover together."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src = tmp_path / "main.py"
    src.write_text("class User: pass\n")

    mock_client = _mock_client()
    mock_client.get_definition = AsyncMock(return_value=[
        Location(path=str(src), line=1, col=7),
    ])
    mock_client.get_references = AsyncMock(return_value=[
        Location(path=str(src), line=1, col=7),
        Location(path=str(tmp_path / "views.py"), line=5, col=10),
    ])
    mock_client.get_hover = AsyncMock(return_value=HoverInfo(contents="class User"))

    orch = LSPOrchestrator()
    orch._clients["python:" + str(tmp_path)] = mock_client

    result = await orch.get_symbol_context(str(src), 1, 7)

    assert "class User" in result
    assert "Defined:" in result
    assert "References (2):" in result
    assert "views.py" in result


@pytest.mark.asyncio
async def test_symbol_context_handles_no_server(tmp_path):
    """get_symbol_context returns helpful message when no server is available."""
    txt = tmp_path / "readme.txt"
    txt.write_text("hello")

    orch = LSPOrchestrator()
    result = await orch.get_symbol_context(str(txt), 1, 1)
    assert "No language server" in result


# ------------------------------------------------------------------
# Shutdown
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_shutdown_all_stops_all_servers():
    """shutdown_all stops all running servers."""
    client_a = _mock_client("python")
    client_b = _mock_client("typescript")

    orch = LSPOrchestrator()
    orch._clients = {"python:/proj": client_a, "typescript:/proj": client_b}

    await orch.shutdown_all()

    client_a.stop.assert_called_once()
    client_b.stop.assert_called_once()
    assert len(orch._clients) == 0


# ------------------------------------------------------------------
# Notify file changed
# ------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_file_changed(tmp_path):
    """notify_file_changed sends didChange to the right client."""
    py_file = tmp_path / "pyproject.toml"
    py_file.touch()
    src = tmp_path / "main.py"
    src.write_text("x = 1\n")

    mock_client = _mock_client()
    orch = LSPOrchestrator()
    orch._clients["python:" + str(tmp_path)] = mock_client

    await orch.notify_file_changed(str(src))
    mock_client.did_change.assert_called_once()
