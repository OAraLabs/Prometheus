"""Sprint S4 A4 — MEMORY.md / USER.md root cause regression test.

Phase 1 audit baseline showed:
  - ``~/.prometheus/MEMORY.md``: 0 bytes, last touched 2026-04-06
  - ``~/.prometheus/USER.md``:   0 bytes, last touched 2026-04-06

Root cause: ``MemoryTool`` (``memory/hermes_memory_tool.py``) was defined
since the initial commit (``cfebf6c``) but never registered in
``create_tool_registry``. The README and ``prompt_assembler.py`` both
treat MEMORY.md / USER.md as the agent's note-to-self surface — but the
agent had no tool to write to them.

This is the same orphan pattern as Phase 1 audit B3 (AnatomyTool,
WhisperSTTTool).

The fix is one line in ``src/prometheus/__main__.py``:

    registry.register(MemoryTool())

These tests freeze that fix so a future refactor can't silently drop
the registration again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.__main__ import create_tool_registry

pytestmark = pytest.mark.integration


class TestMemoryToolRegistration:
    """The orphan-tool regression bar — keep MemoryTool in the registry."""

    def test_memory_tool_in_default_registry(self) -> None:
        """The default registry exposes a ``memory`` tool."""
        registry = create_tool_registry({})
        names = {t.name for t in registry.list_tools()}
        assert "memory" in names, (
            f"MemoryTool not registered. Registry has {len(names)} tools: "
            f"{sorted(names)[:25]}..."
        )

    def test_memory_tool_is_the_hermes_memory_tool(self) -> None:
        """Confirm the registered ``memory`` tool is the one that writes
        MEMORY.md / USER.md (not a placeholder or shim)."""
        from prometheus.memory.hermes_memory_tool import MemoryTool

        registry = create_tool_registry({})
        # Find the registered instance by name and assert its class.
        for tool in registry.list_tools():
            if tool.name == "memory":
                assert isinstance(tool, MemoryTool), (
                    f"'memory' tool is {type(tool).__name__}, expected MemoryTool"
                )
                return
        pytest.fail("'memory' tool not found in registry")


class TestMemoryToolEndToEnd:
    """Invoking MemoryTool writes to MEMORY.md / USER.md at the canonical path."""

    def test_add_appends_to_memory_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``operation='add'`` writes a line to MEMORY.md at the config dir."""
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        # Re-import to pick up the patched config dir.
        from importlib import reload
        from prometheus.config import paths
        from prometheus.memory import hermes_memory_tool

        reload(paths)
        reload(hermes_memory_tool)
        from prometheus.memory.hermes_memory_tool import (
            MemoryTool,
            MemoryToolInput,
        )
        from prometheus.tools.base import ToolExecutionContext

        tool = MemoryTool()
        ctx = ToolExecutionContext(cwd=str(tmp_path))

        result = asyncio.run(tool.execute(
            MemoryToolInput(
                operation="add",
                target="memory",
                entry="User prefers uv over pip for python execution",
            ),
            ctx,
        ))
        assert not result.is_error, result.output
        assert result.output == "added"

        memory_path = tmp_path / "MEMORY.md"
        assert memory_path.exists()
        content = memory_path.read_text(encoding="utf-8")
        assert "User prefers uv over pip" in content

    def test_add_to_user_md(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``operation='add'`` with ``target='user'`` writes to USER.md."""
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        from importlib import reload
        from prometheus.config import paths
        from prometheus.memory import hermes_memory_tool

        reload(paths)
        reload(hermes_memory_tool)
        from prometheus.memory.hermes_memory_tool import (
            MemoryTool,
            MemoryToolInput,
        )
        from prometheus.tools.base import ToolExecutionContext

        tool = MemoryTool()
        ctx = ToolExecutionContext(cwd=str(tmp_path))
        asyncio.run(tool.execute(
            MemoryToolInput(
                operation="add",
                target="user",
                entry="Solo operator on Linux, time zone EST",
            ),
            ctx,
        ))

        user_path = tmp_path / "USER.md"
        assert user_path.exists()
        assert "Solo operator on Linux" in user_path.read_text(encoding="utf-8")

    def test_list_returns_persisted_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After ``add``, ``list`` returns the entries."""
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        from importlib import reload
        from prometheus.config import paths
        from prometheus.memory import hermes_memory_tool

        reload(paths)
        reload(hermes_memory_tool)
        from prometheus.memory.hermes_memory_tool import (
            MemoryTool,
            MemoryToolInput,
        )
        from prometheus.tools.base import ToolExecutionContext

        tool = MemoryTool()
        ctx = ToolExecutionContext(cwd=str(tmp_path))
        for entry in ("fact one", "fact two"):
            asyncio.run(tool.execute(
                MemoryToolInput(operation="add", target="memory", entry=entry),
                ctx,
            ))

        result = asyncio.run(tool.execute(
            MemoryToolInput(operation="list", target="memory"),
            ctx,
        ))
        assert "fact one" in result.output
        assert "fact two" in result.output
