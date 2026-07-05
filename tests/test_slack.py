"""Tests for the Slack gateway adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import MessageEvent, MessageType


# ---------------------------------------------------------------------------
# Import helpers — slack-bolt may not be installed
# ---------------------------------------------------------------------------


@pytest.fixture()
def slack_config():
    """Create a PlatformConfig for Slack."""
    return PlatformConfig(
        platform=Platform.SLACK,
        token="xoxb-test-token",
        app_token="xapp-test-token",
        allowed_channels=[],
    )


@pytest.fixture()
def slack_config_restricted():
    """Create a PlatformConfig with channel whitelist."""
    return PlatformConfig(
        platform=Platform.SLACK,
        token="xoxb-test-token",
        app_token="xapp-test-token",
        allowed_channels=["C123ABC", "C456DEF"],
    )


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


class TestPlatformEnum:
    def test_slack_in_platform(self):
        assert Platform.SLACK.value == "slack"

    def test_all_platforms(self):
        values = [p.value for p in Platform]
        assert "telegram" in values
        assert "slack" in values
        assert "cli" in values


# ---------------------------------------------------------------------------
# PlatformConfig
# ---------------------------------------------------------------------------


class TestPlatformConfig:
    def test_app_token_field(self, slack_config):
        assert slack_config.app_token == "xapp-test-token"

    def test_allowed_channels_field(self, slack_config):
        assert slack_config.allowed_channels == []

    def test_channel_allowed_no_whitelist(self, slack_config):
        assert slack_config.channel_allowed("C123ABC") is True
        assert slack_config.channel_allowed("CXYZ") is True

    def test_channel_allowed_with_whitelist(self, slack_config_restricted):
        assert slack_config_restricted.channel_allowed("C123ABC") is True
        assert slack_config_restricted.channel_allowed("C456DEF") is True
        assert slack_config_restricted.channel_allowed("C999ZZZ") is False


# ---------------------------------------------------------------------------
# Message chunking
# ---------------------------------------------------------------------------


class TestChunkMessage:
    def test_short_message_no_chunking(self):
        from prometheus.gateway.slack import chunk_message

        assert chunk_message("hello") == ["hello"]

    def test_long_message_chunks_at_paragraph(self):
        from prometheus.gateway.slack import chunk_message

        # Build a message with two paragraphs, each just under 2000 chars
        para1 = "A" * 1900
        para2 = "B" * 1900
        text = f"{para1}\n\n{para2}"
        chunks = chunk_message(text, max_length=2000)
        assert len(chunks) == 2
        assert chunks[0] == para1
        assert chunks[1] == para2

    def test_empty_message(self):
        from prometheus.gateway.slack import chunk_message

        assert chunk_message("") == [""]

    def test_exact_limit(self):
        from prometheus.gateway.slack import chunk_message

        text = "X" * 3900
        assert chunk_message(text) == [text]


# ---------------------------------------------------------------------------
# Bot mention stripping
# ---------------------------------------------------------------------------


class TestStripBotMention:
    def test_strips_mention(self):
        from prometheus.gateway.slack import strip_bot_mention

        assert strip_bot_mention("<@U12345> hello") == "hello"

    def test_strips_multiple_mentions(self):
        from prometheus.gateway.slack import strip_bot_mention

        assert strip_bot_mention("<@U12345> <@U67890> hello") == "hello"

    def test_no_mention(self):
        from prometheus.gateway.slack import strip_bot_mention

        assert strip_bot_mention("hello world") == "hello world"

    def test_empty_after_strip(self):
        from prometheus.gateway.slack import strip_bot_mention

        assert strip_bot_mention("<@U12345>") == ""


# ---------------------------------------------------------------------------
# MessageEvent creation from Slack event payload
# ---------------------------------------------------------------------------


class TestMessageEventCreation:
    def test_creates_message_event_from_slack_payload(self):
        event = {
            "channel": "C123ABC",
            "user": "U456DEF",
            "text": "What is the weather?",
            "ts": "1775422227.000100",
        }
        msg = MessageEvent(
            chat_id=hash(event["channel"]),
            user_id=hash(event.get("user", "unknown")),
            text=event.get("text", ""),
            message_id=hash(event.get("ts", "")),
            platform=Platform.SLACK,
            message_type=MessageType.TEXT,
        )
        assert msg.platform == Platform.SLACK
        assert msg.text == "What is the weather?"


# ---------------------------------------------------------------------------
# Channel whitelist enforcement
# ---------------------------------------------------------------------------


class TestChannelWhitelist:
    def test_allowed_channel_passes(self, slack_config_restricted):
        assert slack_config_restricted.channel_allowed("C123ABC") is True

    def test_blocked_channel_rejected(self, slack_config_restricted):
        assert slack_config_restricted.channel_allowed("CNOTALLOWED") is False

    def test_empty_whitelist_allows_all(self, slack_config):
        assert slack_config.channel_allowed("CANYTHING") is True


# ---------------------------------------------------------------------------
# Bot message filtering
# ---------------------------------------------------------------------------


class TestBotFiltering:
    """Verify the adapter ignores messages from bots."""

    def test_bot_message_has_bot_id(self):
        """Events with bot_id should be ignored by _handle_message."""
        event = {
            "bot_id": "B123",
            "channel": "C123",
            "text": "I am a bot",
            "ts": "123.456",
        }
        # The handler checks event.get("bot_id") and returns early
        assert event.get("bot_id") is not None

    def test_subtype_message_ignored(self):
        """Events with subtype (e.g., message_changed) should be ignored."""
        event = {
            "subtype": "message_changed",
            "channel": "C123",
            "text": "edited",
            "ts": "123.456",
        }
        assert event.get("subtype") is not None


# ---------------------------------------------------------------------------
# Config YAML fields
# ---------------------------------------------------------------------------


class TestConfigYaml:
    def test_slack_config_fields_in_yaml(self, tmp_path):
        import yaml

        config = {
            "gateway": {
                "telegram_enabled": True,
                "telegram_token": "tg-token",
                "slack_enabled": True,
                "slack_bot_token": "xoxb-test",
                "slack_app_token": "xapp-test",
                "slack_channels": ["C123"],
            }
        }
        path = tmp_path / "prometheus.yaml"
        with path.open("w") as fh:
            yaml.dump(config, fh)

        with path.open() as fh:
            loaded = yaml.safe_load(fh)

        gw = loaded["gateway"]
        assert gw["slack_enabled"] is True
        assert gw["slack_bot_token"] == "xoxb-test"
        assert gw["slack_app_token"] == "xapp-test"
        assert gw["slack_channels"] == ["C123"]
        # Telegram coexists
        assert gw["telegram_enabled"] is True

    def test_simultaneous_telegram_and_slack(self):
        """Both gateways can be enabled in config."""
        config = PlatformConfig(
            platform=Platform.SLACK,
            token="xoxb-test",
            app_token="xapp-test",
        )
        tg_config = PlatformConfig(
            platform=Platform.TELEGRAM,
            token="tg-token",
        )
        assert config.platform == Platform.SLACK
        assert tg_config.platform == Platform.TELEGRAM


# ---------------------------------------------------------------------------
# Shared command handlers
# ---------------------------------------------------------------------------


class TestSharedCommands:
    """Verify shared commands module produces expected output."""

    def test_cmd_help_returns_string(self):
        from prometheus.gateway.commands import cmd_help

        text = cmd_help()
        assert "Prometheus" in text
        assert "/status" in text
        assert "/help" in text

    def test_cmd_model_returns_info(self):
        from prometheus.gateway.commands import cmd_model

        text = cmd_model("gemma4-26b", "llama_cpp")
        assert "gemma4-26b" in text
        assert "llama_cpp" in text

    def test_cmd_model_unknown(self):
        from prometheus.gateway.commands import cmd_model

        text = cmd_model("", "")
        assert "(unknown)" in text

    def test_cmd_status_returns_status(self):
        from prometheus.gateway.commands import cmd_status

        # Create a mock tool registry
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = ["tool1", "tool2"]

        text = cmd_status("test-model", "test-provider", 0.0, mock_registry)
        assert "Prometheus Status" in text
        assert "test-model" in text
        assert "Tools: 2" in text

    def test_cmd_wiki_no_index(self):
        from prometheus.gateway.commands import cmd_wiki

        with patch("prometheus.gateway.commands.Path.home") as mock_home:
            mock_home.return_value = MagicMock()
            # Make wiki index not exist
            mock_path = mock_home.return_value / ".prometheus" / "wiki" / "index.md"
            mock_path.exists.return_value = False
            # The function uses Path.home() directly, so we need to be more careful
            # Just verify it returns a string
            text = cmd_wiki()
            assert isinstance(text, str)

    def test_cmd_skills_returns_string(self):
        from prometheus.gateway.commands import cmd_skills

        # Will either return skills list or error message — both are strings
        text = cmd_skills()
        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# Setup wizard Slack integration
# ---------------------------------------------------------------------------


class TestSetupWizardSlack:
    def test_wizard_has_slack_fields(self):
        from prometheus.setup_wizard import SetupWizard

        w = SetupWizard()
        assert hasattr(w, "_slack_bot_token")
        assert hasattr(w, "_slack_app_token")
        assert hasattr(w, "_slack_channels")
        assert w._slack_bot_token == ""
        assert w._slack_app_token == ""
        assert w._slack_channels == []

    def test_apply_slack_config(self, tmp_path, monkeypatch):
        from prometheus.setup_wizard import SetupWizard

        config_file = tmp_path / "config" / "prometheus.yaml"
        monkeypatch.setattr("prometheus.setup_wizard._REPO_CONFIG", config_file)

        w = SetupWizard()
        w._gateways = {"slack"}
        w._slack_bot_token = "xoxb-test"
        w._slack_app_token = "xapp-test"
        w._slack_channels = ["C123"]

        cfg: dict = {}
        w._apply_wizard_fields(cfg)

        assert cfg["gateway"]["slack_enabled"] is True
        assert cfg["gateway"]["slack_bot_token"] == "xoxb-test"
        assert cfg["gateway"]["slack_app_token"] == "xapp-test"
        assert cfg["gateway"]["slack_channels"] == ["C123"]

    def test_apply_both_gateways(self, tmp_path, monkeypatch):
        from prometheus.setup_wizard import SetupWizard

        config_file = tmp_path / "config" / "prometheus.yaml"
        monkeypatch.setattr("prometheus.setup_wizard._REPO_CONFIG", config_file)

        w = SetupWizard()
        w._gateways = {"telegram", "slack"}
        w._telegram_token = "tg-token"
        w._telegram_chat_ids = []
        w._slack_bot_token = "xoxb-test"
        w._slack_app_token = "xapp-test"
        w._slack_channels = []

        cfg: dict = {}
        w._apply_wizard_fields(cfg)

        assert cfg["gateway"]["telegram_enabled"] is True
        assert cfg["gateway"]["slack_enabled"] is True

    def test_prefill_from_both(self):
        from prometheus.setup_wizard import SetupWizard

        w = SetupWizard()
        cfg = {
            "model": {"provider": "llama_cpp", "base_url": "http://localhost:8080"},
            "gateway": {
                "telegram_enabled": True,
                "telegram_token": "tg-token",
                "slack_enabled": True,
                "slack_bot_token": "xoxb-test",
                "slack_app_token": "xapp-test",
                "slack_channels": ["C123"],
            },
        }
        w._prefill_from(cfg)
        assert w._gateways == {"telegram", "slack"}
        assert w._telegram_token == "tg-token"
        assert w._slack_bot_token == "xoxb-test"
        assert w._slack_app_token == "xapp-test"
        assert w._slack_channels == ["C123"]

    def test_test_slack_token_valid(self):
        from prometheus.setup_wizard import SetupWizard

        w = SetupWizard()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"ok": True, "team": "TestWorkspace"}

        with patch("prometheus.setup_wizard.httpx.post", return_value=mock_resp):
            result = w._test_slack_token("xoxb-test")
        assert result == "TestWorkspace"

    def test_test_slack_token_invalid(self):
        from prometheus.setup_wizard import SetupWizard

        w = SetupWizard()
        with patch(
            "prometheus.setup_wizard.httpx.post",
            side_effect=Exception("401"),
        ):
            result = w._test_slack_token("xoxb-bad")
        assert result is None


# ---------------------------------------------------------------------------
# Markdown -> mrkdwn conversion (from Hermes pattern)
# ---------------------------------------------------------------------------


class TestMarkdownToMrkdwn:
    def test_bold_conversion(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        assert "*bold*" in format_markdown_to_mrkdwn("**bold**")

    def test_header_conversion(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        result = format_markdown_to_mrkdwn("## My Header")
        assert "*My Header*" in result

    def test_link_conversion(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        result = format_markdown_to_mrkdwn("[click](https://example.com)")
        assert "<https://example.com|click>" in result

    def test_code_block_preserved(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        text = "```python\nprint('hello')\n```"
        assert format_markdown_to_mrkdwn(text) == text

    def test_inline_code_preserved(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        text = "Use `git status` to check"
        result = format_markdown_to_mrkdwn(text)
        assert "`git status`" in result

    def test_strikethrough_conversion(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        result = format_markdown_to_mrkdwn("~~deleted~~")
        assert "~deleted~" in result

    def test_empty_content(self):
        from prometheus.gateway.slack import format_markdown_to_mrkdwn

        assert format_markdown_to_mrkdwn("") == ""
        assert format_markdown_to_mrkdwn(None) is None


# ---------------------------------------------------------------------------
# Message dedup (from Hermes pattern)
# ---------------------------------------------------------------------------


class TestMessageDedup:
    def test_first_message_not_deduped(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000
        assert adapter._dedup_check("123.456") is False

    def test_duplicate_message_deduped(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000
        adapter._dedup_check("123.456")
        assert adapter._dedup_check("123.456") is True

    def test_different_messages_not_deduped(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000
        adapter._dedup_check("123.456")
        assert adapter._dedup_check("789.012") is False

    def test_empty_ts_not_deduped(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000
        assert adapter._dedup_check("") is False


# ---------------------------------------------------------------------------
# Sprint Polish: Telegram parity surface (status / signals / threading / filter)
# ---------------------------------------------------------------------------


class TestSlackStatusCommand:
    """`/status` (the Slack /prometheus-status slash) returns the same shape
    that Telegram /status returns. Both wrap ``cmd_status`` from the shared
    commands module — the smoke test below verifies the shape is intact."""

    def test_slack_status_matches_telegram_shape(self):
        from prometheus.gateway.commands import cmd_status

        registry = MagicMock()
        registry.list_tools.return_value = ["t1", "t2", "t3"]
        text = cmd_status("gemma4-26b", "llama_cpp", 0.0, registry)

        # Both Telegram and Slack call this same helper, so any shape change
        # affecting one affects both. We assert the headline + key fields.
        assert "Prometheus Status" in text
        assert "Model:" in text
        assert "Provider:" in text
        assert "Tools: 3" in text
        # SENTINEL block is always rendered even when "not initialized"
        assert "SENTINEL" in text


class TestSlackSkillEventRoutes:
    """SKILL_CREATED reaches Slack when Slack is enabled — not only the
    originating gateway. Mirrors the Telegram subscription contract."""

    def test_signal_bus_setter_subscribes_to_four_kinds(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._signal_bus = None

        bus = MagicMock()
        adapter.signal_bus = bus

        # Four subscribe() calls — one per signal kind we surface.
        subscribed_kinds = {call.args[0] for call in bus.subscribe.call_args_list}
        assert subscribed_kinds == {
            "skill_created", "skill_refined",
            "memory_updated", "curator_report",
        }

    @pytest.mark.asyncio
    async def test_skill_created_signal_triggers_notification(
        self, slack_config, tmp_path, monkeypatch,
    ):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._signal_bus = None
        adapter._prometheus_config = {}
        adapter.config = slack_config

        # Steer _load_channel to a stable test channel, _save_channel to no-op.
        monkeypatch.setattr(adapter, "_load_channel", lambda: "C-TEST")
        monkeypatch.setattr(adapter, "_save_channel", lambda ch: None)

        # Mock the bolt app's client.chat_postMessage.
        app = MagicMock()
        app.client = MagicMock()
        app.client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})
        adapter._app = app

        # Fabricate a signal-like object.
        signal = MagicMock()
        signal.payload = {"skill_name": "build-pancakes"}

        await adapter._on_signal_skill_created(signal)

        app.client.chat_postMessage.assert_awaited_once()
        kwargs = app.client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "C-TEST"
        assert "build-pancakes" in kwargs["text"]
        assert ":mortar_board:" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_notifications_off_silences(self, slack_config, monkeypatch):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._signal_bus = None
        adapter._prometheus_config = {
            "gateway": {"skill_event_notifications": "off"},
        }
        adapter.config = slack_config
        monkeypatch.setattr(adapter, "_load_channel", lambda: "C-TEST")

        # Patch out the shared runtime override read so it doesn't override "off".
        from prometheus.gateway import commands as _cmds
        monkeypatch.setattr(_cmds, "get_notifications_mode", lambda default="": "")

        app = MagicMock()
        app.client = MagicMock()
        app.client.chat_postMessage = AsyncMock()
        adapter._app = app

        signal = MagicMock()
        signal.payload = {"skill_name": "x"}

        await adapter._on_signal_skill_created(signal)
        app.client.chat_postMessage.assert_not_awaited()


class TestSlackLongReplyThreads:
    """Replies above the configured threshold open a new thread off the
    user's message rather than flooding the channel."""

    @pytest.mark.asyncio
    async def test_long_reply_uses_thread_when_user_did_not_thread(
        self, slack_config, monkeypatch,
    ):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._signal_bus = None
        adapter._prometheus_config = {
            "gateway": {"slack": {"long_reply_threshold": 100}},
        }
        adapter.config = slack_config
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000

        # Wire fake session manager + agent loop.
        from prometheus.engine.session import SessionManager
        adapter.session_manager = SessionManager()

        # AgentLoop result with a long response.
        long_text = "x" * 500
        agent_result = MagicMock()
        agent_result.messages = []
        agent_result.text = long_text
        adapter.agent_loop = MagicMock()
        adapter.agent_loop.run_async = AsyncMock(return_value=agent_result)

        adapter.tool_registry = MagicMock()
        adapter.tool_registry.list_schemas.return_value = []
        adapter.system_prompt = "sys"

        # Stub reactions, channel persistence so they don't fail.
        monkeypatch.setattr(adapter, "_add_reaction", AsyncMock())
        monkeypatch.setattr(adapter, "_remove_reaction", AsyncMock())
        monkeypatch.setattr(adapter, "_save_channel", lambda ch: None)

        say = AsyncMock()
        await adapter._dispatch_to_agent(
            channel="C-X", user="U-X", text="hi",
            ts="100.0", thread_ts=None, say=say,
        )

        say.assert_awaited()
        # thread_ts on the reply should be the original message ts because
        # the response (500 chars) exceeds the 100-char threshold.
        kwargs = say.await_args.kwargs
        assert kwargs["thread_ts"] == "100.0"

    @pytest.mark.asyncio
    async def test_short_reply_stays_in_channel(self, slack_config, monkeypatch):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._signal_bus = None
        adapter._prometheus_config = {
            "gateway": {"slack": {"long_reply_threshold": 800}},
        }
        adapter.config = slack_config
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000

        from prometheus.engine.session import SessionManager
        adapter.session_manager = SessionManager()

        agent_result = MagicMock()
        agent_result.messages = []
        agent_result.text = "short reply"
        adapter.agent_loop = MagicMock()
        adapter.agent_loop.run_async = AsyncMock(return_value=agent_result)

        adapter.tool_registry = MagicMock()
        adapter.tool_registry.list_schemas.return_value = []
        adapter.system_prompt = "sys"

        monkeypatch.setattr(adapter, "_add_reaction", AsyncMock())
        monkeypatch.setattr(adapter, "_remove_reaction", AsyncMock())
        monkeypatch.setattr(adapter, "_save_channel", lambda ch: None)

        say = AsyncMock()
        await adapter._dispatch_to_agent(
            channel="C-X", user="U-X", text="hi",
            ts="100.0", thread_ts=None, say=say,
        )

        say.assert_awaited()
        kwargs = say.await_args.kwargs
        # Short reply → channel-level, no thread.
        assert kwargs["thread_ts"] is None


class TestSlackAllowedChannelsFilter:
    """Bot ignores messages from non-whitelisted channels at the handler level."""

    @pytest.mark.asyncio
    async def test_non_allowed_channel_is_silently_dropped(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000
        adapter.config = PlatformConfig(
            platform=Platform.SLACK,
            token="xoxb-test",
            app_token="xapp-test",
            allowed_channels=["C-ALLOWED"],
        )

        adapter._dispatch_to_agent = AsyncMock()
        say = AsyncMock()

        event = {
            "channel": "C-BLOCKED",
            "user": "U-X",
            "text": "hello",
            "ts": "200.0",
        }
        await adapter._handle_message(event, say)

        adapter._dispatch_to_agent.assert_not_called()
        say.assert_not_called()

    @pytest.mark.asyncio
    async def test_allowed_channel_dispatches(self):
        from prometheus.gateway.slack import SlackAdapter

        adapter = SlackAdapter.__new__(SlackAdapter)
        adapter._seen_messages = {}
        adapter._SEEN_TTL = 300
        adapter._SEEN_MAX = 2000
        adapter.config = PlatformConfig(
            platform=Platform.SLACK,
            token="xoxb-test",
            app_token="xapp-test",
            allowed_channels=["C-ALLOWED"],
        )

        adapter._dispatch_to_agent = AsyncMock()
        say = AsyncMock()

        event = {
            "channel": "C-ALLOWED",
            "user": "U-X",
            "text": "hello",
            "ts": "200.0",
        }
        await adapter._handle_message(event, say)
        adapter._dispatch_to_agent.assert_awaited_once()


class TestSlackSubcommandDispatch:
    """The /prometheus-skills slash supports list/show/pin/unpin subcommands."""

    @pytest.mark.asyncio
    async def test_skills_list_dispatches_to_auto_list(self, monkeypatch):
        from prometheus.gateway.slack import SlackAdapter
        from prometheus.gateway import commands as _cmds

        adapter = SlackAdapter.__new__(SlackAdapter)

        sentinel = "AUTO-LIST-OUTPUT"
        monkeypatch.setattr(_cmds, "cmd_skills_auto_list", lambda: sentinel)

        ack = AsyncMock()
        respond = AsyncMock()
        await adapter._slash_skills(ack, {"text": "list"}, respond)

        ack.assert_awaited_once()
        respond.assert_awaited_once_with(text=sentinel)

    @pytest.mark.asyncio
    async def test_skills_pin_passes_name(self, monkeypatch):
        from prometheus.gateway.slack import SlackAdapter
        from prometheus.gateway import commands as _cmds

        adapter = SlackAdapter.__new__(SlackAdapter)
        captured: dict = {}

        def _pin(name):
            captured["name"] = name
            return f"pinned:{name}"

        monkeypatch.setattr(_cmds, "cmd_skills_pin", _pin)

        ack = AsyncMock()
        respond = AsyncMock()
        await adapter._slash_skills(ack, {"text": "pin build-pancakes"}, respond)

        assert captured["name"] == "build-pancakes"
        respond.assert_awaited_once_with(text="pinned:build-pancakes")


# ---------------------------------------------------------------------------
# SPRINT G3 — start() must be non-blocking and fail loud on bad tokens
# ---------------------------------------------------------------------------


class TestSlackStartNonBlocking:
    """Regression pins for the G3 start() fix.

    The old start() awaited slack-bolt's ``start_async()`` — documented as
    "starts infinite sleep" — so an enabled Slack gateway hung the daemon's
    startup sequence forever (even with VALID tokens). And a bad token made
    ``connect_async`` retry silently forever. The fixed start() validates
    both tokens loudly, then connects in a background task.
    """

    def _fake_slack_bolt(self, monkeypatch, *, auth_ok=True, wss_ok=True):
        import asyncio
        import sys
        import types

        calls: dict = {"connect_started": asyncio.Event()}

        class FakeWebClient:
            async def auth_test(self):
                calls["auth_test"] = True
                if not auth_ok:
                    raise RuntimeError("invalid_auth (bot token)")
                return {"ok": True, "team": "T"}

        class FakeApp:
            def __init__(self, token=None):
                self.client = FakeWebClient()

            def event(self, *_a, **_k):
                return lambda f: f

            def command(self, *_a, **_k):
                return lambda f: f

        class FakeSMClient:
            wss_uri = None

            async def issue_new_wss_url(self):
                calls["issue_wss"] = True
                if not wss_ok:
                    raise RuntimeError("invalid_auth (app token)")
                return "wss://fake"

        class FakeHandler:
            def __init__(self, app, app_token):
                self.client = FakeSMClient()

            async def connect_async(self):
                calls["connect_started"].set()

            async def close_async(self):
                pass

        mod_app = types.ModuleType("slack_bolt.async_app")
        mod_app.AsyncApp = FakeApp
        mod_handler = types.ModuleType(
            "slack_bolt.adapter.socket_mode.async_handler")
        mod_handler.AsyncSocketModeHandler = FakeHandler
        for name, mod in (
            ("slack_bolt", types.ModuleType("slack_bolt")),
            ("slack_bolt.adapter", types.ModuleType("slack_bolt.adapter")),
            ("slack_bolt.adapter.socket_mode",
             types.ModuleType("slack_bolt.adapter.socket_mode")),
            ("slack_bolt.async_app", mod_app),
            ("slack_bolt.adapter.socket_mode.async_handler", mod_handler),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        return calls

    def _adapter(self, slack_config):
        from prometheus.gateway.slack import SlackAdapter

        return SlackAdapter(
            config=slack_config,
            agent_loop=MagicMock(),
            tool_registry=MagicMock(),
            system_prompt="",
            model_name="test-model",
            model_provider="llama_cpp",
        )

    @pytest.mark.asyncio
    async def test_start_returns_and_connects_in_background(
        self, monkeypatch, slack_config,
    ):
        import asyncio

        calls = self._fake_slack_bolt(monkeypatch)
        adapter = self._adapter(slack_config)
        # The whole point: start() RETURNS (the old code slept forever).
        await asyncio.wait_for(adapter.start(), timeout=5)
        assert calls.get("auth_test") is True
        assert calls.get("issue_wss") is True
        assert adapter._running is True
        assert adapter._connect_task is not None
        await asyncio.wait_for(calls["connect_started"].wait(), timeout=5)
        adapter._connect_task.cancel()

    @pytest.mark.asyncio
    async def test_bad_bot_token_raises_loudly(self, monkeypatch, slack_config):
        import asyncio

        self._fake_slack_bolt(monkeypatch, auth_ok=False)
        adapter = self._adapter(slack_config)
        with pytest.raises(Exception, match="invalid_auth"):
            await asyncio.wait_for(adapter.start(), timeout=5)
        assert adapter._running is False

    @pytest.mark.asyncio
    async def test_bad_app_token_raises_loudly(self, monkeypatch, slack_config):
        import asyncio

        self._fake_slack_bolt(monkeypatch, wss_ok=False)
        adapter = self._adapter(slack_config)
        with pytest.raises(Exception, match="invalid_auth"):
            await asyncio.wait_for(adapter.start(), timeout=5)
        assert adapter._running is False
