"""Tests for Feature 1: Deferred Tool Loading."""

import pytest

from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.file_read import FileReadTool
from prometheus.tools.builtin.grep import GrepTool
from prometheus.tools.builtin.glob import GlobTool
from prometheus.tools.builtin.web_search import WebSearchTool
from prometheus.context.dynamic_tools import DynamicToolLoader


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    reg.register(GrepTool())
    reg.register(GlobTool())
    reg.register(WebSearchTool())
    return reg


class TestDeferredLoading:
    def test_disabled_returns_all(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={"enabled": False})
        schemas = loader.active_schemas()
        assert len(schemas) == 5

    def test_enabled_returns_only_always_loaded(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={
            "enabled": True,
            "always_loaded": ["bash", "read_file"],
        })
        schemas = loader.active_schemas()
        names = [s["name"] for s in schemas]
        assert "bash" in names
        assert "read_file" in names
        assert "grep" not in names
        assert "web_search" not in names
        assert len(schemas) == 2

    def test_deferred_count(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={
            "enabled": True,
            "always_loaded": ["bash", "read_file"],
        })
        assert loader.deferred_count == 3  # 5 total - 2 loaded

    def test_deferred_count_disabled(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={"enabled": False})
        assert loader.deferred_count == 0

    def test_keyword_matching_still_works_when_disabled(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={"enabled": False})
        schemas = loader.active_schemas("search for files")
        names = [s["name"] for s in schemas]
        # "search" keyword maps to grep and web_search
        assert "grep" in names or "web_search" in names

    def test_on_demand_returns_schema_regardless_of_deferred(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={
            "enabled": True,
            "always_loaded": ["bash"],
        })
        # on_demand should return schema even for non-loaded tools
        schema = loader.on_demand("web_search")
        assert schema is not None
        assert schema["name"] == "web_search"

    def test_all_schemas_returns_everything(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={
            "enabled": True,
            "always_loaded": ["bash"],
        })
        schemas = loader.all_schemas()
        assert len(schemas) == 5

    def test_empty_always_loaded_returns_empty(self, registry):
        loader = DynamicToolLoader(registry, deferred_config={
            "enabled": True,
            "always_loaded": [],
        })
        schemas = loader.active_schemas()
        assert len(schemas) == 0

    def test_default_config_no_deferred(self, registry):
        loader = DynamicToolLoader(registry)
        assert loader._deferred_enabled is False
        schemas = loader.active_schemas()
        assert len(schemas) == 5

    def test_token_savings(self, registry):
        """Deferred loading should produce fewer schemas = fewer tokens."""
        loader_full = DynamicToolLoader(registry, deferred_config={"enabled": False})
        loader_deferred = DynamicToolLoader(registry, deferred_config={
            "enabled": True,
            "always_loaded": ["bash"],
        })
        full = loader_full.active_schemas()
        deferred = loader_deferred.active_schemas()
        assert len(deferred) < len(full)
