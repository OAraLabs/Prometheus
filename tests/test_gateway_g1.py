"""SPRINT G1 — gateway-generic subsystem wiring + Slack parity handlers.

Covers:
  * GatewaySubsystemRegistry semantics (order-independent broadcast).
  * daemon.py grep-proof: no subsystem is injected into the Telegram
    adapter by name — everything goes through the registry.
  * The new Slack /prometheus-* handlers (thin wrappers over the shared
    commands layer), exercised with fakes — no slack-bolt connection, no
    tokens, no network.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import prometheus.daemon as daemon_mod
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import (
    BasePlatformAdapter,
    GatewaySubsystemRegistry,
    MessageEvent,
    SendResult,
)


class _FakeAdapter(BasePlatformAdapter):
    """Minimal concrete adapter for registry tests."""

    async def start(self) -> None:  # pragma: no cover - unused
        self._running = True

    async def stop(self) -> None:  # pragma: no cover - unused
        self._running = False

    async def send(self, chat_id, text, *, reply_to=None, parse_mode=None):
        return SendResult(success=True)

    async def on_message(self, event: MessageEvent) -> None:  # pragma: no cover
        pass


def _fake_adapter() -> _FakeAdapter:
    return _FakeAdapter(PlatformConfig(platform=Platform.CLI))


# ---------------------------------------------------------------------------
# GatewaySubsystemRegistry
# ---------------------------------------------------------------------------


class TestGatewaySubsystemRegistry:
    def test_base_adapter_has_none_slots(self):
        a = _fake_adapter()
        for slot in (
            "cost_tracker", "escalation_engine", "_approval_queue",
            "_gepa_engine", "_printing_press", "_backup_vault", "_morph_engine",
        ):
            assert getattr(a, slot) is None

    def test_attach_reaches_registered_adapter(self):
        reg = GatewaySubsystemRegistry()
        a = _fake_adapter()
        reg.register_adapter(a)
        queue = object()
        reg.attach("_approval_queue", queue)
        assert a._approval_queue is queue

    def test_late_registration_replays_earlier_attachments(self):
        """The ordering fix: approval queue + printing press are attached
        before the Slack adapter exists; registering later must replay."""
        reg = GatewaySubsystemRegistry()
        early = _fake_adapter()
        reg.register_adapter(early)
        queue, press = object(), object()
        reg.attach("_approval_queue", queue)
        reg.attach("_printing_press", press)

        late = _fake_adapter()
        reg.register_adapter(late)
        assert late._approval_queue is queue
        assert late._printing_press is press
        assert early._approval_queue is queue

    def test_register_none_and_duplicates_are_noops(self):
        reg = GatewaySubsystemRegistry()
        a = _fake_adapter()
        reg.register_adapter(None)
        reg.register_adapter(a)
        reg.register_adapter(a)
        assert reg.adapters == [a]

    def test_attach_invokes_property_setters(self):
        """setattr goes through properties — the signal_bus subscribe-on-set
        contract survives the generic path (Slack adapter, no bolt needed)."""
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._signal_bus = None

        reg = GatewaySubsystemRegistry()
        reg.register_adapter(adapter)
        bus = MagicMock()
        reg.attach("signal_bus", bus)

        kinds = {call.args[0] for call in bus.subscribe.call_args_list}
        assert kinds == {
            "skill_created", "skill_refined", "memory_updated", "curator_report",
        }

    def test_one_failing_setter_does_not_block_others(self):
        class _Exploding(_FakeAdapter):
            @property
            def boom(self):  # pragma: no cover - never read
                return None

            @boom.setter
            def boom(self, value):
                raise RuntimeError("setter exploded")

        reg = GatewaySubsystemRegistry()
        bad = _Exploding(PlatformConfig(platform=Platform.CLI))
        good = _fake_adapter()
        reg.register_adapter(bad)
        reg.register_adapter(good)
        marker = object()
        reg.attach("boom", marker)  # must not raise
        assert getattr(good, "boom") is marker


class TestDaemonUsesGenericWiring:
    """Grep-level proof (acceptance #4): daemon.py never injects a subsystem
    into the telegram adapter by name — the registry path covers all
    adapters. The only remaining `telegram` references hand the ADAPTER to a
    subsystem (delivery transport), which is the reverse direction."""

    def test_no_by_name_subsystem_injection(self):
        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        forbidden = re.findall(
            r"^\s*(?:telegram|slack_adapter)\.(?:_\w+|cost_tracker"
            r"|escalation_engine|signal_bus)\s*=",
            source,
            flags=re.MULTILINE,
        )
        assert not forbidden, (
            "daemon.py assigns subsystems to a specific gateway adapter by "
            f"name: {forbidden} — use gateway_registry.attach(...) instead"
        )

    def test_registry_wires_every_subsystem(self):
        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        for slot in (
            "cost_tracker", "_approval_queue", "_printing_press",
            "escalation_engine", "signal_bus", "_backup_vault",
            "_morph_engine", "_gepa_engine",
        ):
            assert f'gateway_registry.attach("{slot}"' in source, (
                f"daemon.py no longer attaches {slot!r} through the "
                "GatewaySubsystemRegistry"
            )
        assert source.count("gateway_registry.register_adapter(") >= 2


# ---------------------------------------------------------------------------
# New Slack handlers (SPRINT G1)
# ---------------------------------------------------------------------------


def _make_slack_adapter(**kwargs):
    from prometheus.gateway.slack import SlackAdapter

    config = PlatformConfig(
        platform=Platform.SLACK,
        token="xoxb-test-token",
        app_token="xapp-test-token",
    )
    agent_loop = kwargs.pop("agent_loop", None)
    if agent_loop is None:
        agent_loop = AsyncMock()
        agent_loop._model_router = None
    return SlackAdapter(
        config=config,
        agent_loop=agent_loop,
        tool_registry=MagicMock(),
        model_name=kwargs.pop("model_name", "test-model-v1"),
        model_provider=kwargs.pop("model_provider", "llama_cpp"),
    )


def _cmd(text: str = "", channel: str = "C123"):
    return {"text": text, "channel_id": channel}


class _Respond:
    def __init__(self):
        self.messages: list[str] = []

    async def __call__(self, text: str = "", **kw):
        self.messages.append(kw.get("text", text))


@pytest.fixture()
def respond():
    return _Respond()


@pytest.fixture()
def ack():
    return AsyncMock()


class TestSlackApprovalHandlers:
    @pytest.mark.asyncio
    async def test_approve_usage(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_approve(ack, _cmd(""), respond)
        assert respond.messages == ["Usage: /prometheus-approve {request_id}"]

    @pytest.mark.asyncio
    async def test_approve_no_queue(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_approve(ack, _cmd("abc123"), respond)
        assert respond.messages == ["Approval queue not active."]

    @pytest.mark.asyncio
    async def test_approve_with_queue(self, ack, respond):
        adapter = _make_slack_adapter()
        queue = MagicMock()
        queue.approve = AsyncMock(return_value=True)
        adapter._approval_queue = queue
        await adapter._slash_approve(ack, _cmd("abc123"), respond)
        assert respond.messages == ["Approved: abc123"]
        queue.approve.assert_awaited_once_with("abc123")

    @pytest.mark.asyncio
    async def test_deny_not_found(self, ack, respond):
        adapter = _make_slack_adapter()
        queue = MagicMock()
        queue.deny = AsyncMock(return_value=False)
        adapter._approval_queue = queue
        await adapter._slash_deny(ack, _cmd("zzz"), respond)
        assert respond.messages == ["No pending request: zzz"]

    @pytest.mark.asyncio
    async def test_pending_lists(self, ack, respond):
        adapter = _make_slack_adapter()
        action = MagicMock()
        action.request_id = "r1"
        action.tool_name = "bash"
        action.description = "do something"
        queue = MagicMock()
        queue.list_pending.return_value = [action]
        adapter._approval_queue = queue
        await adapter._slash_pending(ack, respond)
        assert respond.messages == [
            "Pending approval requests:\n  r1: bash — do something"
        ]


class TestSlackProviderOverrideHandlers:
    @pytest.mark.asyncio
    async def test_claude_no_router(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_claude(ack, _cmd(), respond)
        assert respond.messages == [
            "Routing is not enabled. Provider overrides require a "
            "configured router in prometheus.yaml."
        ]

    @pytest.mark.asyncio
    async def test_override_success_keys_on_slack_channel(
        self, ack, respond, monkeypatch,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        agent_loop = AsyncMock()
        router = MagicMock()
        router.config.overrides_enabled = True
        agent_loop._model_router = router
        adapter = _make_slack_adapter(agent_loop=agent_loop)
        adapter._prometheus_config = {
            "slash_commands": {
                "claude": {
                    "provider": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-x",
                },
            },
        }
        await adapter._slash_claude(ack, _cmd(channel="C42"), respond)
        router.set_override.assert_called_once()
        session_key, preset = router.set_override.call_args[0]
        assert session_key == "slack:C42"
        assert preset["model"] == "claude-x"
        assert "Switched to Claude (anthropic)." in respond.messages[0]
        # Slack-native command names in the reply, not Telegram's.
        assert "/prometheus-local" in respond.messages[0]
        assert "/prometheus-route" in respond.messages[0]

    @pytest.mark.asyncio
    async def test_override_inline_text_gets_honest_note(
        self, ack, respond, monkeypatch,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        agent_loop = AsyncMock()
        router = MagicMock()
        router.config.overrides_enabled = True
        agent_loop._model_router = router
        adapter = _make_slack_adapter(agent_loop=agent_loop)
        adapter._prometheus_config = {
            "slash_commands": {"claude": {
                "provider": "anthropic",
                "api_key_env": "ANTHROPIC_API_KEY",
                "model": "claude-x",
            }},
        }
        await adapter._slash_claude(ack, _cmd(text="what is 2+2?"), respond)
        assert "inline message dispatch isn't supported on Slack" in (
            respond.messages[0]
        )

    @pytest.mark.asyncio
    async def test_local_no_override(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_local(ack, _cmd(), respond)
        assert respond.messages == [
            "Already on primary (llama_cpp/test-model-v1). No override was set."
        ]

    @pytest.mark.asyncio
    async def test_route_no_router_uses_slack_prefix(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_route(ack, _cmd(), respond)
        text = respond.messages[0]
        assert text.startswith("Route\nActive: llama_cpp/test-model-v1  (no router)")
        assert "/prometheus-claude" in text
        assert "/prometheus-local" in text
        # No bare-Telegram command names leak into the Slack surface.
        assert "\n  /claude" not in text


class TestSlackSubsystemCommandHandlers:
    @pytest.mark.asyncio
    async def test_escalations_not_available(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_escalations(ack, respond)
        assert respond.messages == [
            "Teacher escalation: not available in this build."
        ]

    @pytest.mark.asyncio
    async def test_escalations_with_engine(self, ack, respond):
        adapter = _make_slack_adapter()
        engine = MagicMock()
        engine.stats.return_value = {
            "armed": False, "teacher": "", "fired": 0, "skills_written": 0,
            "teacher_failed": 0, "refused_budget": 0, "max_per_session": 2,
            "sessions": {},
        }
        adapter.escalation_engine = engine
        await adapter._slash_escalations(ack, respond)
        assert "Armed: no (escalation.teacher_model unset)" in respond.messages[0]

    @pytest.mark.asyncio
    async def test_gepa_status_no_engine(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_gepa(ack, _cmd(), respond)
        assert respond.messages == [
            "GEPA: engine not active (set learning.gepa_enabled in config)."
        ]

    @pytest.mark.asyncio
    async def test_gepa_usage_uses_slack_prefix(self, ack, respond):
        adapter = _make_slack_adapter()
        adapter._gepa_engine = MagicMock()
        await adapter._slash_gepa(ack, _cmd("bogus"), respond)
        assert respond.messages == [
            "Usage: /prometheus-gepa [status | run | history]"
        ]

    @pytest.mark.asyncio
    async def test_symbiote_inactive(self, ack, respond, monkeypatch):
        import prometheus.symbiote as sym
        monkeypatch.setattr(sym, "get_coordinator", lambda: None)
        adapter = _make_slack_adapter()
        adapter._app = None  # channel sender falls back to respond
        await adapter._slash_symbiote(ack, _cmd("fix the parser"), respond)
        assert respond.messages == [
            "SYMBIOTE is not active. Set symbiote.enabled in config."
        ]

    @pytest.mark.asyncio
    async def test_press_inactive(self, ack, respond):
        adapter = _make_slack_adapter()
        adapter._app = None
        await adapter._slash_press(ack, _cmd("list"), respond)
        assert respond.messages == [
            "Printing Press is not active. The library clone is missing "
            "(searched ~/printing-press-library/ and /tmp/printing-press-library/) "
            "or the feature is disabled in config."
        ]

    @pytest.mark.asyncio
    async def test_audit_usage_uses_slack_prefix(self, ack, respond):
        adapter = _make_slack_adapter()
        adapter._app = None
        await adapter._slash_audit(ack, _cmd("bogus-category"), respond)
        text = respond.messages[0]
        assert "/prometheus-audit run" in text
        assert "Categories: search, fetch, youtube" in text

    @pytest.mark.asyncio
    async def test_voice_is_platform_honest(self, ack, respond):
        adapter = _make_slack_adapter()
        await adapter._slash_voice(ack, respond)
        text = respond.messages[0]
        assert "not supported on Slack yet" in text
        assert "Telegram" in text  # says WHY, not just no

    @pytest.mark.asyncio
    async def test_note_without_store(self, ack, respond, monkeypatch):
        import prometheus.tools.builtin.wiki_compile as wc
        monkeypatch.setattr(wc, "_memory_store", None)
        adapter = _make_slack_adapter()
        await adapter._slash_note(ack, _cmd("remember this"), respond)
        assert respond.messages == ["Memory store unavailable — note not saved."]

    @pytest.mark.asyncio
    async def test_pairs(self, ack, respond, monkeypatch):
        import prometheus.learning.pair_capture as pc
        store = MagicMock()
        store.stats.return_value = {
            "total": 5, "last_7d": 1, "per_day_7d": 0.1,
            "by_source": {}, "by_tool": {},
        }
        monkeypatch.setattr(pc, "get_store", lambda: store)
        adapter = _make_slack_adapter()
        await adapter._slash_pairs(ack, respond)
        assert respond.messages == ["Training Pairs\n\nTotal: 5\nLast 7d: 1 (~0.1/day)"]


class TestSlackChannelSender:
    @pytest.mark.asyncio
    async def test_prefers_channel_post(self):
        adapter = _make_slack_adapter()
        app = MagicMock()
        app.client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})
        adapter._app = app
        r = _Respond()
        send = adapter._channel_sender(_cmd(channel="C9"), r)
        await send("hello")
        app.client.chat_postMessage.assert_awaited_once_with(
            channel="C9", text="hello",
        )
        assert r.messages == []

    @pytest.mark.asyncio
    async def test_falls_back_to_respond(self):
        adapter = _make_slack_adapter()
        adapter._app = None
        r = _Respond()
        send = adapter._channel_sender(_cmd(channel=""), r)
        await send("hello")
        assert r.messages == ["hello"]
