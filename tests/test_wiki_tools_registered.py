"""Wiring regression test for WikiCompileTool / WikiQueryTool.

Pre-cleanup: ``scripts/daemon.py::build_tool_registry`` re-registered both
tools in a ``try/except: pass`` block, even though
``create_tool_registry`` already registers them unconditionally
(``__main__.py:188-189``). The daemon block was dead code per PR #7's
drive-by finding in ``docs/audits/ORPHAN-TOOLS-AUDIT.md``.

This test pins the contract so a future refactor that removes the wiki
tools from ``create_tool_registry`` without restoring a daemon-side
fallback will fail loudly — instead of silently leaving the daemon
without wiki tools the way the dead try/except: pass would have.
"""

from __future__ import annotations

import pytest

from scripts.daemon import build_tool_registry


pytestmark = pytest.mark.integration


class TestWikiToolsRegisteredViaDaemonBuild:
    """The daemon's registry-builder must surface both wiki tools."""

    def test_wiki_compile_and_query_present(self) -> None:
        registry = build_tool_registry({})
        names = {t.name for t in registry.list_tools()}
        # Names come from the tool classes themselves; verify both surface.
        from prometheus.tools.builtin.wiki_compile import WikiCompileTool
        from prometheus.tools.builtin.wiki_query import WikiQueryTool

        registered_types = {type(t) for t in registry.list_tools()}
        assert WikiCompileTool in registered_types, (
            f"WikiCompileTool not in registry. Registered: "
            f"{sorted(t.__name__ for t in registered_types)[:30]}..."
        )
        assert WikiQueryTool in registered_types, (
            f"WikiQueryTool not in registry. Registered: "
            f"{sorted(t.__name__ for t in registered_types)[:30]}..."
        )
        # And the tools' canonical names are listed too — guards against
        # the (unlikely) case where the classes are imported but a
        # different one is registered under the same name.
        assert any("wiki" in n.lower() for n in names), (
            f"No wiki tool by name in {sorted(names)}"
        )
