"""SPRINT G1 — reply-text pins for the Telegram command extraction.

These tests hard-code the EXACT reply text a handful of representative
Telegram handlers produced BEFORE the G1 refactor (logic moved from
telegram.py into the shared prometheus.gateway.commands layer). They were
written and run green against the pre-refactor tree, then kept green through
the refactor — proving the extraction preserved Telegram-visible behaviour
byte-for-byte.

No real adapters, tokens, or network: the adapter is constructed with a fake
token and ``send`` is mocked; nothing here starts polling or binds ports.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import SendResult
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.tools.base import ToolRegistry


def _make_update(chat_id: int = 123, text: str = "") -> MagicMock:
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user = MagicMock()
    update.effective_user.id = 456
    update.effective_user.username = "tester"
    update.message = MagicMock()
    update.message.text = text
    update.message.message_id = 1
    return update


def _make_context(args: list[str] | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


def _make_adapter(**kwargs) -> TelegramAdapter:
    config = PlatformConfig(platform=Platform.TELEGRAM, token="test")
    agent_loop = kwargs.pop("agent_loop", None)
    if agent_loop is None:
        agent_loop = AsyncMock()
        agent_loop._model_router = None
    adapter = TelegramAdapter(
        config=config,
        agent_loop=agent_loop,
        tool_registry=kwargs.pop("tool_registry", ToolRegistry()),
        model_name=kwargs.pop("model_name", "test-model-v1"),
        model_provider=kwargs.pop("model_provider", "llama_cpp"),
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id=1))
    return adapter


def _sent_text(adapter: TelegramAdapter) -> str:
    return adapter.send.call_args[0][1]


# ---------------------------------------------------------------------------
# Provider override family: /route, /local
# ---------------------------------------------------------------------------


class TestRouteLocalPins:
    # NOTE (CLOUD EXPANSION 2026-07): the /route override list deliberately
    # gained /deepseek /kimi /glm /mimo lines; the pins below were updated to
    # the new expected surface at the same time as the change.
    @pytest.mark.asyncio
    async def test_route_no_router_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_route(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Route\n"
            "Active: llama_cpp/test-model-v1  (no router)\n"
            "\n"
            "Override commands:\n"
            "  /claude  — Anthropic Claude\n"
            "  /gpt     — OpenAI GPT\n"
            "  /gemini  — Google Gemini\n"
            "  /xai     — xAI Grok\n"
            "  /grok    — alias for /xai\n"
            "  /deepseek — DeepSeek\n"
            "  /kimi    — Kimi (Moonshot)\n"
            "  /glm     — GLM (Z.ai)\n"
            "  /mimo    — MiMo (Xiaomi)\n"
            "  /local   — back to primary"
        )

    @pytest.mark.asyncio
    async def test_route_override_active_pin(self):
        agent_loop = AsyncMock()
        router = MagicMock()
        override = MagicMock()
        override.provider_config = {"provider": "anthropic", "model": "claude-x"}
        router.get_override_for_session.return_value = override
        agent_loop._model_router = router
        adapter = _make_adapter(agent_loop=agent_loop)
        await adapter._cmd_route(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Route\n"
            "Active: anthropic/claude-x  (override)\n"
            "Clear with: /local\n"
            "\n"
            "Override commands:\n"
            "  /claude  — Anthropic Claude\n"
            "  /gpt     — OpenAI GPT\n"
            "  /gemini  — Google Gemini\n"
            "  /xai     — xAI Grok\n"
            "  /grok    — alias for /xai\n"
            "  /deepseek — DeepSeek\n"
            "  /kimi    — Kimi (Moonshot)\n"
            "  /glm     — GLM (Z.ai)\n"
            "  /mimo    — MiMo (Xiaomi)\n"
            "  /local   — back to primary"
        )

    @pytest.mark.asyncio
    async def test_local_no_override_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_local(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Already on primary (llama_cpp/test-model-v1). No override was set."
        )

    @pytest.mark.asyncio
    async def test_local_clears_override_pin(self):
        agent_loop = AsyncMock()
        router = MagicMock()
        router.get_override_for_session.return_value = MagicMock()
        agent_loop._model_router = router
        adapter = _make_adapter(agent_loop=agent_loop)
        await adapter._cmd_local(_make_update(), _make_context())
        assert _sent_text(adapter) == "Back to primary (llama_cpp/test-model-v1)."
        router.clear_override.assert_called_once_with("telegram:123")

    @pytest.mark.asyncio
    async def test_override_no_router_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_claude(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Routing is not enabled. Provider overrides require a "
            "configured router in prometheus.yaml."
        )

    @pytest.mark.asyncio
    async def test_override_disabled_pin(self):
        agent_loop = AsyncMock()
        router = MagicMock()
        router.config.overrides_enabled = False
        agent_loop._model_router = router
        adapter = _make_adapter(agent_loop=agent_loop)
        await adapter._cmd_gpt(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Direct-mode provider overrides are disabled.\n"
            "Set router.overrides.enabled: true in config/prometheus.yaml "
            "and restart the daemon to enable."
        )


# ---------------------------------------------------------------------------
# Approval family: /approve, /deny, /pending
# ---------------------------------------------------------------------------


class TestApprovalPins:
    @pytest.mark.asyncio
    async def test_approve_usage_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_approve(_make_update(text="/approve"), _make_context())
        assert _sent_text(adapter) == "Usage: /approve {request_id}"

    @pytest.mark.asyncio
    async def test_approve_no_queue_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_approve(
            _make_update(text="/approve abc123"), _make_context()
        )
        assert _sent_text(adapter) == "Approval queue not active."

    @pytest.mark.asyncio
    async def test_approve_found_pin(self):
        adapter = _make_adapter()
        queue = MagicMock()
        queue.approve = AsyncMock(return_value=True)
        adapter._approval_queue = queue
        await adapter._cmd_approve(
            _make_update(text="/approve abc123"), _make_context()
        )
        assert _sent_text(adapter) == "Approved: abc123"

    @pytest.mark.asyncio
    async def test_deny_not_found_pin(self):
        adapter = _make_adapter()
        queue = MagicMock()
        queue.deny = AsyncMock(return_value=False)
        adapter._approval_queue = queue
        await adapter._cmd_deny(_make_update(text="/deny zzz"), _make_context())
        assert _sent_text(adapter) == "No pending request: zzz"

    @pytest.mark.asyncio
    async def test_pending_empty_pin(self):
        adapter = _make_adapter()
        queue = MagicMock()
        queue.list_pending.return_value = []
        adapter._approval_queue = queue
        await adapter._cmd_pending(_make_update(text="/pending"), _make_context())
        assert _sent_text(adapter) == "No pending requests."

    @pytest.mark.asyncio
    async def test_pending_list_pin(self):
        adapter = _make_adapter()
        action = MagicMock()
        action.request_id = "r1"
        action.tool_name = "bash"
        action.description = "rm -rf /tmp/x"
        queue = MagicMock()
        queue.list_pending.return_value = [action]
        adapter._approval_queue = queue
        await adapter._cmd_pending(_make_update(text="/pending"), _make_context())
        assert _sent_text(adapter) == (
            "Pending approval requests:\n  r1: bash — rm -rf /tmp/x"
        )


# ---------------------------------------------------------------------------
# /escalations
# ---------------------------------------------------------------------------


class TestEscalationsPins:
    @pytest.mark.asyncio
    async def test_escalations_absent_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_escalations(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Teacher escalation: not available in this build."
        )

    @pytest.mark.asyncio
    async def test_escalations_armed_pin(self):
        adapter = _make_adapter()
        engine = MagicMock()
        engine.stats.return_value = {
            "armed": True,
            "teacher": "anthropic/claude-x",
            "fired": 2,
            "skills_written": 1,
            "teacher_failed": 0,
            "refused_budget": 3,
            "max_per_session": 2,
            "sessions": {"telegram:123": 2},
        }
        adapter.escalation_engine = engine
        await adapter._cmd_escalations(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Teacher escalation\n"
            "Armed: yes — anthropic/claude-x\n"
            "Fired: 2   Skills written: 1\n"
            "Teacher failures: 0   Budget refusals: 3\n"
            "Budget: 2 per session (in-memory, resets on restart)\n"
            "Used: telegram:123=2"
        )


# ---------------------------------------------------------------------------
# /gepa
# ---------------------------------------------------------------------------


class TestGepaPins:
    @pytest.mark.asyncio
    async def test_gepa_status_no_engine_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_gepa(_make_update(text="/gepa"), _make_context())
        assert _sent_text(adapter) == (
            "GEPA: engine not active (set learning.gepa_enabled in config)."
        )

    @pytest.mark.asyncio
    async def test_gepa_status_no_report_pin(self):
        adapter = _make_adapter()
        engine = MagicMock()
        engine.last_report = None
        adapter._gepa_engine = engine
        await adapter._cmd_gepa(_make_update(text="/gepa status"), _make_context())
        assert _sent_text(adapter) == (
            "GEPA: no cycle has run yet. Use /gepa run to trigger one manually."
        )

    @pytest.mark.asyncio
    async def test_gepa_usage_pin(self):
        adapter = _make_adapter()
        adapter._gepa_engine = MagicMock()
        await adapter._cmd_gepa(_make_update(text="/gepa bogus"), _make_context())
        assert _sent_text(adapter) == "Usage: /gepa [status | run | history]"


# ---------------------------------------------------------------------------
# /symbiote, /audit, /press — inactive-subsystem + usage pins
# ---------------------------------------------------------------------------


class TestSymbioteAuditPressPins:
    @pytest.mark.asyncio
    async def test_symbiote_inactive_pin(self, monkeypatch):
        import prometheus.symbiote as sym
        monkeypatch.setattr(sym, "get_coordinator", lambda: None)
        adapter = _make_adapter()
        await adapter._cmd_symbiote(
            _make_update(text="/symbiote fix the parser"), _make_context()
        )
        assert _sent_text(adapter) == (
            "SYMBIOTE is not active. Set symbiote.enabled in config."
        )

    @pytest.mark.asyncio
    async def test_symbiote_no_session_status_pin(self, monkeypatch):
        import prometheus.symbiote as sym
        coordinator = MagicMock()
        coordinator.get_status.return_value = None
        monkeypatch.setattr(sym, "get_coordinator", lambda: coordinator)
        adapter = _make_adapter()
        await adapter._cmd_symbiote(_make_update(text="/symbiote"), _make_context())
        assert _sent_text(adapter) == "SYMBIOTE: no active session."

    @pytest.mark.asyncio
    async def test_audit_usage_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_audit(
            _make_update(text="/audit bogus-category"), _make_context()
        )
        assert _sent_text(adapter) == (
            "Usage:\n"
            "  /audit                — show last audit summary\n"
            "  /audit run            — full audit (~30–60 min)\n"
            "  /audit <category>     — single category\n"
            "Categories: search, fetch, youtube, download, research, graceful, railway"
        )

    @pytest.mark.asyncio
    async def test_press_inactive_pin(self):
        adapter = _make_adapter()
        await adapter._cmd_press(_make_update(text="/press list"), _make_context())
        assert _sent_text(adapter) == (
            "Printing Press is not active. The library clone is missing "
            "(searched ~/printing-press-library/ and /tmp/printing-press-library/) "
            "or the feature is disabled in config."
        )

    @pytest.mark.asyncio
    async def test_press_usage_pin(self):
        adapter = _make_adapter()
        press = MagicMock()
        press.is_available.return_value = True
        adapter._printing_press = press
        await adapter._cmd_press(_make_update(text="/press"), _make_context())
        assert _sent_text(adapter) == (
            "Printing Press — local CLI library\n\n"
            "  /press list [category]  — list available CLIs\n"
            "  /press search <query>   — fuzzy search\n"
            "  /press install <name>   — install (queues approval)\n"
            "  /press installed        — show what's installed\n"
            "  /press update           — git pull the library clone\n"
        )


# ---------------------------------------------------------------------------
# /voice
# ---------------------------------------------------------------------------


class TestVoicePins:
    @pytest.mark.asyncio
    async def test_voice_show_default_pin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        adapter = _make_adapter()
        await adapter._cmd_voice(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Voice mode: auto\n"
            "  auto — mirror input modality (default)\n"
            "  on   — always voice reply\n"
            "  off  — always text reply\n"
            "\nEngine: piper\nModel:  (unset)\n"
            "Set with: /voice [auto|on|off]"
        )

    @pytest.mark.asyncio
    async def test_voice_set_on_pin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        adapter = _make_adapter()
        await adapter._cmd_voice(_make_update(), _make_context(["on"]))
        assert _sent_text(adapter) == (
            "Voice mode set to: on\n  (always reply with voice)"
        )
        assert adapter._get_voice_mode(123) == "on"

    @pytest.mark.asyncio
    async def test_voice_unknown_pin(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        adapter = _make_adapter()
        await adapter._cmd_voice(_make_update(), _make_context(["loud"]))
        assert _sent_text(adapter) == (
            "Unknown voice mode: loud\nUse: /voice [auto|on|off]"
        )


# ---------------------------------------------------------------------------
# /tools and /pairs (formatters extracted to the shared layer in G1)
# ---------------------------------------------------------------------------


class TestToolsPairsPins:
    @pytest.mark.asyncio
    async def test_tools_pin(self, monkeypatch):
        import prometheus.telemetry.dashboard as dash

        fake = MagicMock()
        fake.get_stats.return_value = {
            "total_calls": 10,
            "overall_success_rate": 0.9,
            "total_denials": 0,
            "most_called": [{"tool_name": "bash", "calls": 6}],
            "success_rate_by_tool": {"bash": 1.0},
            "circuit_breaker_trips": 0,
            "adapter_repairs": 2,
            "lucky_guesses": 0,
        }
        monkeypatch.setattr(dash, "ToolDashboard", lambda: fake)
        adapter = _make_adapter()
        await adapter._cmd_tools(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Tool Call Stats (24h)\n\n"
            "Total calls: 10\n"
            "Success rate: 90%\n"
            "\nMost called:\n"
            "  bash: 6 calls (100% ok)\n"
            "Adapter repairs: 2"
        )

    @pytest.mark.asyncio
    async def test_pairs_pin(self, monkeypatch):
        import prometheus.learning.pair_capture as pc

        store = MagicMock()
        store.stats.return_value = {
            "total": 327,
            "last_7d": 14,
            "per_day_7d": 2.0,
            "by_source": {"repair": 300, "golden": 27},
            "by_tool": {"bash": 200},
        }
        monkeypatch.setattr(pc, "get_store", lambda: store)
        adapter = _make_adapter()
        await adapter._cmd_pairs(_make_update(), _make_context())
        assert _sent_text(adapter) == (
            "Training Pairs\n\n"
            "Total: 327\n"
            "Last 7d: 14 (~2.0/day)\n"
            "\nBy source:\n"
            "  repair: 300\n"
            "  golden: 27\n"
            "\nBy tool:\n"
            "  bash: 200"
        )
