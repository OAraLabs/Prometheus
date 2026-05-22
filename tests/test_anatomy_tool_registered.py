"""Phase 2 (orphan-tools) — AnatomyTool registration regression test.

Phase 1 audit (``docs/audits/ORPHAN-TOOLS-AUDIT.md``) found that
``AnatomyTool`` was a fully-implemented BaseTool subclass with a working
``execute()`` method, never registered in any registration site. The
daemon wired ``set_anatomy_components(...)`` at
``scripts/daemon.py:550`` — substantial setup effort — but the tool
itself never reached the registry, so the agent could not invoke
``/anatomy`` despite all wiring being in place.

The fix is one ``try_register`` call in ``create_tool_registry``.
These tests freeze that fix and verify the tool's ``execute()`` returns
structured output (graceful-degrade path when the daemon hasn't wired
the scanner — the CLI / test-suite case).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.__main__ import create_tool_registry
from prometheus.tools.base import ToolExecutionContext, ToolResult


pytestmark = pytest.mark.integration


class TestAnatomyToolRegistration:
    """The Phase 2 registration bar — keep AnatomyTool in the registry."""

    def test_anatomy_tool_in_default_registry(self) -> None:
        registry = create_tool_registry({})
        names = {t.name for t in registry.list_tools()}
        assert "anatomy" in names, (
            f"AnatomyTool not registered. Registry has {len(names)} tools: "
            f"{sorted(names)[:30]}..."
        )

    def test_registered_instance_is_anatomy_tool(self) -> None:
        from prometheus.tools.builtin.anatomy import AnatomyTool

        registry = create_tool_registry({})
        for tool in registry.list_tools():
            if tool.name == "anatomy":
                assert isinstance(tool, AnatomyTool), (
                    f"'anatomy' tool is {type(tool).__name__}, expected AnatomyTool"
                )
                return
        pytest.fail("'anatomy' tool not found in registry")


class TestAnatomyToolExecute:
    """Functional wiring test — execute() returns structured output, doesn't raise.

    When the daemon hasn't wired ``set_anatomy_components`` (CLI mode,
    test runs), the tool's graceful-degrade path returns an is_error
    ToolResult with a "not initialized" message. That's the contract we
    want frozen — the tool reaches execute, the registry call resolves.
    """

    def test_execute_returns_tool_result_when_unwired(self, tmp_path: Path) -> None:
        from prometheus.tools.builtin.anatomy import AnatomyTool, AnatomyInput
        from prometheus.tools.builtin import anatomy as anatomy_module

        # Force the unwired state — the test process may have inherited
        # singletons from a previous test.
        anatomy_module._scanner = None
        anatomy_module._writer = None
        anatomy_module._project_store = None

        tool = AnatomyTool()
        ctx = ToolExecutionContext(cwd=tmp_path)
        result = asyncio.run(
            tool.execute(AnatomyInput(action="status"), ctx)
        )
        assert isinstance(result, ToolResult)
        assert result.is_error is True
        assert "not initialized" in (result.output or "").lower()

    def test_execute_returns_tool_result_when_wired(self, tmp_path: Path) -> None:
        """With the daemon-style wiring complete, a status call surfaces
        whatever ``writer.render_summary`` produces — verifies the
        execute() path doesn't crash when the scanner is real."""
        from prometheus.tools.builtin.anatomy import (
            AnatomyTool, AnatomyInput, set_anatomy_components,
        )

        scanner = MagicMock()
        scanner.quick_scan = MagicMock(
            return_value=_make_awaitable("mock_state")
        )
        writer = MagicMock()
        writer.render_summary = MagicMock(return_value="ok: mock state")

        set_anatomy_components(scanner, writer, None)
        try:
            tool = AnatomyTool()
            ctx = ToolExecutionContext(cwd=tmp_path)
            result = asyncio.run(
                tool.execute(AnatomyInput(action="status"), ctx)
            )
            assert isinstance(result, ToolResult)
            assert result.is_error is False
            assert "ok: mock state" in (result.output or "")
        finally:
            # Reset module singletons so other tests aren't affected.
            from prometheus.tools.builtin import anatomy as anatomy_module
            anatomy_module._scanner = None
            anatomy_module._writer = None
            anatomy_module._project_store = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_awaitable(result):
    async def _coro():
        return result
    return _coro()
