"""Tests for Sprint 15b GRAFT: hook hot reload + loader."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from prometheus.hooks.events import HookEvent
from prometheus.hooks.loader import load_hook_registry
from prometheus.hooks.registry import HookRegistry


class TestHookLoader:

    def test_load_empty_config(self):
        registry = load_hook_registry({})
        assert registry.get(HookEvent.PRE_TOOL_USE) == []

    def test_load_command_hook(self):
        config = {
            "pre_tool_use": [
                {"type": "command", "command": "echo test", "block_on_failure": False}
            ]
        }
        registry = load_hook_registry(config)
        hooks = registry.get(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0].type == "command"
        assert hooks[0].command == "echo test"

    def test_load_http_hook(self):
        config = {
            "post_tool_use": [
                {"type": "http", "url": "http://localhost:9090/hook"}
            ]
        }
        registry = load_hook_registry(config)
        hooks = registry.get(HookEvent.POST_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0].url == "http://localhost:9090/hook"

    def test_skip_unknown_event(self):
        config = {"nonexistent_event": [{"type": "command", "command": "echo"}]}
        registry = load_hook_registry(config)
        # Should not crash, just skip
        for event in HookEvent:
            assert registry.get(event) == []

    def test_skip_unknown_hook_type(self):
        config = {"pre_tool_use": [{"type": "unknown_type"}]}
        registry = load_hook_registry(config)
        assert registry.get(HookEvent.PRE_TOOL_USE) == []

    def test_multiple_events_and_hooks(self):
        config = {
            "pre_tool_use": [
                {"type": "command", "command": "echo pre1"},
                {"type": "command", "command": "echo pre2"},
            ],
            "post_tool_use": [
                {"type": "http", "url": "http://localhost/post"},
            ],
        }
        registry = load_hook_registry(config)
        assert len(registry.get(HookEvent.PRE_TOOL_USE)) == 2
        assert len(registry.get(HookEvent.POST_TOOL_USE)) == 1


class TestHookReloader:

    def test_reload_on_file_change(self, tmp_path):
        from prometheus.hooks.hot_reload import HookReloader

        config_file = tmp_path / "prometheus.yaml"
        config_file.write_text(yaml.dump({"hooks": {
            "pre_tool_use": [{"type": "command", "command": "echo v1"}]
        }}))

        reloader = HookReloader(config_file)
        reg = reloader.current_registry()
        hooks = reg.get(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0].command == "echo v1"

        # Modify the file
        time.sleep(0.01)  # ensure mtime changes
        config_file.write_text(yaml.dump({"hooks": {
            "pre_tool_use": [
                {"type": "command", "command": "echo v2"},
                {"type": "command", "command": "echo v3"},
            ]
        }}))

        reg2 = reloader.current_registry()
        hooks2 = reg2.get(HookEvent.PRE_TOOL_USE)
        assert len(hooks2) == 2
        assert hooks2[0].command == "echo v2"

    def test_file_deleted_resets_registry(self, tmp_path):
        from prometheus.hooks.hot_reload import HookReloader

        config_file = tmp_path / "prometheus.yaml"
        config_file.write_text(yaml.dump({"hooks": {
            "pre_tool_use": [{"type": "command", "command": "echo exists"}]
        }}))

        reloader = HookReloader(config_file)
        reg = reloader.current_registry()
        assert len(reg.get(HookEvent.PRE_TOOL_USE)) == 1

        config_file.unlink()
        reg2 = reloader.current_registry()
        assert len(reg2.get(HookEvent.PRE_TOOL_USE)) == 0

    def test_no_change_no_reload(self, tmp_path):
        from prometheus.hooks.hot_reload import HookReloader

        config_file = tmp_path / "prometheus.yaml"
        config_file.write_text(yaml.dump({"hooks": {}}))

        reloader = HookReloader(config_file)
        reg1 = reloader.current_registry()
        reg2 = reloader.current_registry()
        # Same object — no reload happened
        assert reg1 is reg2

    def test_existing_hooks_work_without_reload(self):
        """Regression: hooks work fine when hot reload is never started."""
        registry = HookRegistry()
        from prometheus.hooks.schemas import CommandHookDefinition
        registry.add(
            HookEvent.PRE_TOOL_USE,
            CommandHookDefinition(type="command", command="echo static"),
        )
        assert len(registry.get(HookEvent.PRE_TOOL_USE)) == 1
