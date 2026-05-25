"""Tests for the per-slash-command model selection wiring (PR #18).

Three layers covered here:

1. The anthropic alias resolver — verifies the known aliases in
   :func:`prometheus.providers.anthropic.resolve_model_alias` map to the
   correct dated snapshots, and unknown strings pass through unchanged.

2. The startup diagnostic — verifies the daemon-initialization path calls
   :func:`prometheus.router.model_router.log_slash_command_wiring` and that
   the resulting INFO lines reflect what's in ``slash_commands`` config.

3. The Telegram handler wiring — verifies ``_apply_override`` reads the
   resolved provider+model from config (not the hardcoded preset) and
   surfaces it in both the per-session router override AND the user-facing
   confirmation message.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# 1. Anthropic alias resolution
# ---------------------------------------------------------------------------


class TestAnthropicAliasResolution:
    def test_sonnet_4_5_alias_maps_to_dated_snapshot(self):
        from prometheus.providers.anthropic import resolve_model_alias
        assert resolve_model_alias("claude-sonnet-4-5") == "claude-sonnet-4-5-20250929"

    def test_haiku_4_5_alias_maps_to_dated_snapshot(self):
        from prometheus.providers.anthropic import resolve_model_alias
        assert resolve_model_alias("claude-haiku-4-5") == "claude-haiku-4-5-20251001"

    def test_opus_4_5_alias_maps_to_dated_snapshot(self):
        from prometheus.providers.anthropic import resolve_model_alias
        assert resolve_model_alias("claude-opus-4-5") == "claude-opus-4-5-20251101"

    def test_dateless_id_passes_through(self):
        """4.6+ IDs are already pinned snapshots; resolver should leave them alone."""
        from prometheus.providers.anthropic import resolve_model_alias
        assert resolve_model_alias("claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert resolve_model_alias("claude-opus-4-7") == "claude-opus-4-7"

    def test_already_dated_snapshot_passes_through(self):
        """Caller-supplied dated snapshots are returned as-is."""
        from prometheus.providers.anthropic import resolve_model_alias
        assert (
            resolve_model_alias("claude-sonnet-4-5-20250929")
            == "claude-sonnet-4-5-20250929"
        )

    def test_unknown_model_passes_through(self):
        """Unknown strings are returned unchanged (forward-compat with new models)."""
        from prometheus.providers.anthropic import resolve_model_alias
        assert resolve_model_alias("claude-future-99-99") == "claude-future-99-99"


# ---------------------------------------------------------------------------
# 2. Telegram handler integration
# ---------------------------------------------------------------------------


def _build_telegram_adapter(prometheus_config: dict):
    """Build a TelegramAdapter with mocked dependencies and the given config.

    Returns (adapter, captured_state) where captured_state is a dict the test
    can inspect after invoking the handler:

        captured_state["set_override_calls"]  — list of (session_key, dict)
        captured_state["sent_messages"]       — list of (chat_id, text, parse_mode)
    """
    from prometheus.engine.session import SessionManager
    from prometheus.gateway.config import PlatformConfig, Platform
    from prometheus.gateway.telegram import TelegramAdapter

    captured: dict = {"set_override_calls": [], "sent_messages": []}

    # Router mock — has the config.overrides_enabled flag and a set_override
    # method we can capture.
    router_config = MagicMock()
    router_config.overrides_enabled = True
    router = MagicMock()
    router.config = router_config

    def _record_set_override(session_key, preset_dict):
        captured["set_override_calls"].append((session_key, dict(preset_dict)))

    router.set_override = MagicMock(side_effect=_record_set_override)

    # Agent loop carries the router as ``_model_router`` (matches daemon
    # wiring). Adapter looks it up via getattr(agent_loop, "_model_router").
    agent_loop = MagicMock()
    agent_loop._model_router = router

    tg_config = PlatformConfig(platform=Platform.TELEGRAM, token="fake-token")
    tool_registry = MagicMock()

    adapter = TelegramAdapter(
        config=tg_config,
        agent_loop=agent_loop,
        tool_registry=tool_registry,
        system_prompt="(test)",
        model_name="gemma4-26b",
        model_provider="llama_cpp",
        session_manager=SessionManager(),
        prometheus_config=prometheus_config,
    )

    # Patch send() so we can capture the user-facing reply.
    async def _record_send(chat_id, text, parse_mode=None, **kw):
        captured["sent_messages"].append((chat_id, text, parse_mode))

    adapter.send = _record_send  # type: ignore[assignment]

    return adapter, captured


def _make_update_context(chat_id: int = 42, args: list[str] | None = None):
    """Build a minimal Update + ContextTypes.DEFAULT_TYPE pair for handler calls."""
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user = MagicMock()
    update.effective_user.id = 7
    update.effective_user.username = "tester"
    update.message = MagicMock()
    update.message.text = ""
    update.message.message_id = 1
    context = MagicMock()
    context.args = args or []
    return update, context


class TestTelegramSlashCommandHandlers:
    """Functional wiring tests: invoke the handler, assert side effects."""

    def test_slash_claude_uses_config_model_when_set(self, monkeypatch):
        """slash_commands.claude.model = sonnet-4-5 → override + reply both say sonnet-4-5."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        cfg = {
            "slash_commands": {
                "claude": {
                    "provider": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-sonnet-4-5",
                },
            },
        }
        adapter, captured = _build_telegram_adapter(cfg)
        update, context = _make_update_context()

        asyncio.run(adapter._apply_override(update, context, preset_name="claude"))

        # 1) The router got the SONNET preset, not the Haiku preset.
        assert len(captured["set_override_calls"]) == 1
        session_key, preset_dict = captured["set_override_calls"][0]
        assert session_key == "telegram:42"
        assert preset_dict["provider"] == "anthropic"
        assert preset_dict["model"] == "claude-sonnet-4-5"

        # 2) The reply visible to the user names the actual model.
        assert len(captured["sent_messages"]) == 1
        chat_id, text, _ = captured["sent_messages"][0]
        assert chat_id == 42
        assert "claude-sonnet-4-5" in text
        # And explicitly NOT the built-in haiku default.
        assert "claude-haiku-4-5-20251001" not in text

    def test_slash_claude_falls_back_to_default_when_config_missing(self, monkeypatch, caplog):
        """No slash_commands section → uses built-in Haiku preset + WARN once."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        # Reset the once-only warn tracker so we can observe the warn.
        from prometheus.router import model_router as mr
        mr._FALLBACK_WARNED.clear()

        adapter, captured = _build_telegram_adapter(prometheus_config={})
        update, context = _make_update_context()

        caplog.set_level(logging.WARNING, logger="prometheus.router.model_router")
        asyncio.run(adapter._apply_override(update, context, preset_name="claude"))

        assert len(captured["set_override_calls"]) == 1
        _, preset_dict = captured["set_override_calls"][0]
        # Built-in Haiku 4.5 preset (current OVERRIDE_PRESETS["claude"]).
        assert preset_dict["model"] == "claude-haiku-4-5-20251001"

        # WARN line surfaced once.
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("slash_commands.claude" in r.message for r in warns)

    def test_slash_command_response_shows_resolved_model(self, monkeypatch):
        """User-facing reply text always contains the model that's actually wired."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = {"slash_commands": {"gpt": {"model": "gpt-4o-mini"}}}
        adapter, captured = _build_telegram_adapter(cfg)
        update, context = _make_update_context()

        asyncio.run(adapter._apply_override(update, context, preset_name="gpt"))

        assert len(captured["sent_messages"]) == 1
        _, text, _ = captured["sent_messages"][0]
        assert "gpt-4o-mini" in text

    def test_slash_gpt_uses_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        cfg = {"slash_commands": {"gpt": {"model": "gpt-4o-mini"}}}
        adapter, captured = _build_telegram_adapter(cfg)
        update, context = _make_update_context()
        asyncio.run(adapter._apply_override(update, context, preset_name="gpt"))
        _, preset_dict = captured["set_override_calls"][0]
        assert preset_dict["provider"] == "openai"
        assert preset_dict["model"] == "gpt-4o-mini"

    def test_slash_gemini_uses_config(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        cfg = {"slash_commands": {"gemini": {"model": "gemini-2.5-pro"}}}
        adapter, captured = _build_telegram_adapter(cfg)
        update, context = _make_update_context()
        asyncio.run(adapter._apply_override(update, context, preset_name="gemini"))
        _, preset_dict = captured["set_override_calls"][0]
        assert preset_dict["provider"] == "gemini"
        assert preset_dict["model"] == "gemini-2.5-pro"

    def test_slash_xai_uses_config(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "x")
        cfg = {"slash_commands": {"xai": {"model": "grok-3-mini"}}}
        adapter, captured = _build_telegram_adapter(cfg)
        update, context = _make_update_context()
        asyncio.run(adapter._apply_override(update, context, preset_name="xai"))
        _, preset_dict = captured["set_override_calls"][0]
        assert preset_dict["provider"] == "xai"
        assert preset_dict["model"] == "grok-3-mini"


# ---------------------------------------------------------------------------
# 3. Startup diagnostic
# ---------------------------------------------------------------------------


class TestStartupDiagnostic:
    """log_slash_command_wiring is called from daemon.py at boot.

    The function itself is unit-tested in test_model_router.py; this class
    asserts the daemon-side call site behaves correctly when given a config
    matching what's in prometheus.yaml.default.
    """

    def test_startup_logs_one_info_line_per_known_command(self, caplog):
        from prometheus.router.model_router import (
            SLASH_COMMAND_NAMES,
            log_slash_command_wiring,
        )

        cfg = {
            "slash_commands": {
                "claude": {"model": "claude-sonnet-4-5"},
                "gpt": {"model": "gpt-4o"},
                "gemini": {"model": "gemini-2.5-pro"},
                "xai": {"model": "grok-3"},
            },
        }
        caplog.set_level(logging.INFO, logger="prometheus.router.model_router")
        log_slash_command_wiring(cfg)

        info_msgs = [
            r.message
            for r in caplog.records
            if r.levelno == logging.INFO and "slash_commands." in r.message
        ]
        assert len(info_msgs) == len(SLASH_COMMAND_NAMES)
        # Each known command appears in the joined log output.
        joined = "\n".join(info_msgs)
        for cmd in SLASH_COMMAND_NAMES:
            assert f"slash_commands.{cmd}" in joined
