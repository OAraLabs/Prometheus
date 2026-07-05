"""SPRINT G2 — tests for the Discord gateway adapter.

Fakes only: no real Discord connection, no tokens, no network. Covers
construction + import-guard, inbound message flow → agent dispatch (session
key namespace), whitelist enforcement, 2000-char chunking, thin-wrapper
command handlers (approvals, provider overrides, subsystems), media routing
by content type through the shared media services, and subsystem-registry
inheritance.
"""

from __future__ import annotations

import builtins
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.discord import (
    DiscordAdapter,
    MAX_MESSAGE_LENGTH,
    REACTION_DONE,
    REACTION_PROCESSING,
    chunk_message,
    format_markdown_for_discord,
)
from prometheus.gateway.platform_base import (
    GatewaySubsystemRegistry,
    MessageEvent,
    MessageType,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _discord_config(**kwargs) -> PlatformConfig:
    return PlatformConfig(
        platform=Platform.DISCORD,
        token=kwargs.pop("token", "fake-discord-token"),
        allowed_guild_ids=kwargs.pop("allowed_guild_ids", []),
        allowed_channel_ids=kwargs.pop("allowed_channel_ids", []),
        **kwargs,
    )


def _make_adapter(**kwargs) -> DiscordAdapter:
    agent_loop = kwargs.pop("agent_loop", None)
    if agent_loop is None:
        agent_loop = AsyncMock()
        agent_loop._model_router = None
    return DiscordAdapter(
        config=kwargs.pop("config", _discord_config()),
        agent_loop=agent_loop,
        tool_registry=kwargs.pop("tool_registry", MagicMock()),
        model_name=kwargs.pop("model_name", "test-model-v1"),
        model_provider=kwargs.pop("model_provider", "llama_cpp"),
        prometheus_config=kwargs.pop("prometheus_config", None),
    )


class _FakeResponse:
    def __init__(self):
        self.messages: list[str] = []
        self.deferred = False
        self._done = False

    def is_done(self) -> bool:
        return self._done

    async def send_message(self, content=None, **kw):
        self.messages.append(content)
        self._done = True

    async def defer(self, **kw):
        self.deferred = True
        self._done = True


class _FakeFollowup:
    def __init__(self):
        self.messages: list[str] = []

    async def send(self, content=None, **kw):
        self.messages.append(content)


class _FakeInteraction:
    def __init__(self, channel_id: int | None = 123, channel=None):
        self.channel_id = channel_id
        self.channel = channel
        self.response = _FakeResponse()
        self.followup = _FakeFollowup()

    @property
    def all_messages(self) -> list[str]:
        return self.response.messages + self.followup.messages


def _fake_message(
    *,
    channel_id: int = 555,
    guild_id: int | None = None,
    content: str = "hello",
    attachments: list | None = None,
    author_bot: bool = False,
):
    msg = MagicMock()
    msg.id = 42
    msg.content = content
    msg.attachments = attachments or []
    msg.author = MagicMock()
    msg.author.bot = author_bot
    msg.author.id = 777
    msg.author.__str__ = lambda self: "tester#1"
    if guild_id is None:
        msg.guild = None
    else:
        msg.guild = MagicMock()
        msg.guild.id = guild_id
    msg.channel = MagicMock()
    msg.channel.id = channel_id
    msg.channel.send = AsyncMock()
    msg.add_reaction = AsyncMock()
    msg.remove_reaction = AsyncMock()
    msg.create_thread = AsyncMock()
    return msg


def _fake_attachment(
    *,
    filename: str,
    content_type: str,
    data: bytes = b"bytes",
    size: int = 10,
):
    att = MagicMock()
    att.filename = filename
    att.content_type = content_type
    att.size = size
    att.read = AsyncMock(return_value=data)
    return att


def _agent_result(text: str = "agent reply"):
    result = MagicMock()
    result.text = text
    result.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": text},
    ]
    return result


# ---------------------------------------------------------------------------
# Construction + import guard
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_platform_enum(self):
        assert Platform.DISCORD.value == "discord"

    def test_construction_needs_no_discord_lib(self):
        """The module + constructor never import discord.py (mirror of
        slack.py, whose module imports without slack-bolt)."""
        adapter = _make_adapter()
        assert adapter.platform == Platform.DISCORD
        assert adapter.running is False

    def test_base_subsystem_slots_default_none(self):
        adapter = _make_adapter()
        for slot in (
            "cost_tracker", "escalation_engine", "_approval_queue",
            "_gepa_engine", "_printing_press", "_backup_vault", "_morph_engine",
        ):
            assert getattr(adapter, slot) is None

    @pytest.mark.asyncio
    async def test_start_without_discordpy_raises_install_hint(self, monkeypatch):
        adapter = _make_adapter()
        real_import = builtins.__import__

        def _no_discord(name, *args, **kwargs):
            if name == "discord" or name.startswith("discord."):
                raise ImportError("No module named 'discord'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_discord)
        with pytest.raises(ImportError, match=r"oara-prometheus\[discord\]"):
            await adapter.start()

    @pytest.mark.asyncio
    async def test_start_without_token_raises(self):
        pytest.importorskip("discord")
        adapter = _make_adapter(config=_discord_config(token=""))
        with pytest.raises(ValueError, match="token"):
            await adapter.start()


# ---------------------------------------------------------------------------
# Whitelist semantics
# ---------------------------------------------------------------------------


class TestWhitelist:
    def test_dm_always_allowed(self):
        cfg = _discord_config()
        assert cfg.discord_inbound_allowed(is_dm=True, guild_id=None, channel_id=1)

    def test_empty_whitelist_means_dms_only(self):
        cfg = _discord_config()
        assert not cfg.discord_inbound_allowed(
            is_dm=False, guild_id=10, channel_id=20,
        )

    def test_channel_whitelist_hit(self):
        cfg = _discord_config(allowed_channel_ids=[20])
        assert cfg.discord_inbound_allowed(is_dm=False, guild_id=10, channel_id=20)
        assert not cfg.discord_inbound_allowed(is_dm=False, guild_id=10, channel_id=21)

    def test_guild_whitelist_hit(self):
        cfg = _discord_config(allowed_guild_ids=[10])
        assert cfg.discord_inbound_allowed(is_dm=False, guild_id=10, channel_id=99)
        assert not cfg.discord_inbound_allowed(is_dm=False, guild_id=11, channel_id=99)

    @pytest.mark.asyncio
    async def test_non_whitelisted_guild_message_never_reaches_agent(self):
        adapter = _make_adapter()
        adapter._dispatch_to_agent = AsyncMock()
        msg = _fake_message(guild_id=10, channel_id=20)
        await adapter._handle_discord_message(msg)
        adapter._dispatch_to_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitelisted_channel_message_dispatches(self):
        adapter = _make_adapter(config=_discord_config(allowed_channel_ids=[20]))
        adapter._dispatch_to_agent = AsyncMock()
        msg = _fake_message(guild_id=10, channel_id=20)
        await adapter._handle_discord_message(msg)
        adapter._dispatch_to_agent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bot_messages_ignored(self):
        adapter = _make_adapter()
        adapter._dispatch_to_agent = AsyncMock()
        await adapter._handle_discord_message(_fake_message(author_bot=True))
        adapter._dispatch_to_agent.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_message_rechecks_whitelist(self):
        adapter = _make_adapter()
        adapter._dispatch_to_agent = AsyncMock()
        event = MessageEvent(
            chat_id=20, user_id=1, text="hi", message_id=1,
            platform=Platform.DISCORD,
            raw={"is_dm": False, "guild_id": 10},
        )
        await adapter.on_message(event)
        adapter._dispatch_to_agent.assert_not_awaited()


# ---------------------------------------------------------------------------
# Inbound flow → agent dispatch
# ---------------------------------------------------------------------------


class TestMessageDispatch:
    @pytest.mark.asyncio
    async def test_dm_dispatches_with_discord_session_key(self):
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=_agent_result("hi there"))
        adapter = _make_adapter(agent_loop=agent_loop)
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        msg = _fake_message(channel_id=555, guild_id=None, content="hello")

        await adapter._handle_discord_message(msg)

        agent_loop.run_async.assert_awaited_once()
        assert agent_loop.run_async.call_args.kwargs["session_id"] == "discord:555"
        # Session was created under the discord:<channel_id> namespace.
        assert "discord:555" in adapter.session_manager._sessions
        # Reply went to the message's channel.
        sent = [c.args[0] for c in msg.channel.send.call_args_list]
        assert sent == ["hi there"]

    @pytest.mark.asyncio
    async def test_reaction_ack_cycle(self):
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=_agent_result())
        adapter = _make_adapter(agent_loop=agent_loop)
        adapter._client = MagicMock()  # remove_reaction needs client.user
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        msg = _fake_message()

        await adapter._handle_discord_message(msg)

        added = [c.args[0] for c in msg.add_reaction.call_args_list]
        assert added == [REACTION_PROCESSING, REACTION_DONE]
        removed = [c.args[0] for c in msg.remove_reaction.call_args_list]
        assert removed == [REACTION_PROCESSING]

    @pytest.mark.asyncio
    async def test_agent_error_rolls_back_and_reports(self):
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(side_effect=RuntimeError("model down"))
        adapter = _make_adapter(agent_loop=agent_loop)
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        msg = _fake_message(channel_id=9)

        await adapter._handle_discord_message(msg)

        session = adapter.session_manager._sessions["discord:9"]
        assert session.get_messages() == []  # rolled back
        sent = [c.args[0] for c in msg.channel.send.call_args_list]
        assert sent == ["Error: model down"]

    @pytest.mark.asyncio
    async def test_long_reply_opens_thread(self):
        long_text = "word " * 500  # > default 800-char threshold
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=_agent_result(long_text))
        adapter = _make_adapter(
            agent_loop=agent_loop,
            config=_discord_config(allowed_guild_ids=[10]),
        )
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        msg = _fake_message(guild_id=10, channel_id=20)
        thread = MagicMock()
        thread.send = AsyncMock()
        msg.create_thread = AsyncMock(return_value=thread)

        await adapter._handle_discord_message(msg)

        msg.create_thread.assert_awaited_once()
        assert thread.send.await_count >= 1
        msg.channel.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dm_long_reply_stays_in_channel(self):
        long_text = "word " * 500
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=_agent_result(long_text))
        adapter = _make_adapter(agent_loop=agent_loop)
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        msg = _fake_message(guild_id=None)  # DM — no threads

        await adapter._handle_discord_message(msg)

        msg.create_thread.assert_not_awaited()
        assert msg.channel.send.await_count >= 1

    @pytest.mark.asyncio
    async def test_queued_prompts_drain_after_turn(self):
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=_agent_result())
        adapter = _make_adapter(agent_loop=agent_loop)
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        session = adapter.session_manager.get_or_create("discord:555")
        session.enqueue_prompt("follow-up question")

        await adapter._handle_discord_message(_fake_message(channel_id=555))

        assert agent_loop.run_async.await_count == 2
        assert session.queued_prompts == []


# ---------------------------------------------------------------------------
# Chunking + markdown
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_message_single_chunk(self):
        assert chunk_message("hello") == ["hello"]

    def test_empty_message(self):
        assert chunk_message("") == [""]

    def test_chunks_respect_2000_limit(self):
        text = "a" * 4500
        chunks = chunk_message(text)
        assert all(len(c) <= MAX_MESSAGE_LENGTH for c in chunks)
        assert "".join(chunks) == text

    def test_prefers_paragraph_boundary(self):
        text = "para one\n\n" + ("b" * 1995) + "\n\npara three"
        chunks = chunk_message(text, max_length=2000)
        assert chunks[0] == "para one"
        assert all(len(c) <= 2000 for c in chunks)

    def test_default_limit_is_discord_2000(self):
        assert MAX_MESSAGE_LENGTH == 2000

    def test_markdown_passthrough(self):
        # Discord renders standard markdown natively — no conversion.
        text = "# H1\n**bold** *it* ~~strike~~ `code`\n```py\nx=1\n```\n[a](http://b)"
        assert format_markdown_for_discord(text) == text


# ---------------------------------------------------------------------------
# App-command handlers — thin wrappers over the shared layer
# ---------------------------------------------------------------------------


class TestApprovalHandlers:
    @pytest.mark.asyncio
    async def test_approve_usage_renders_discord_path(self):
        adapter = _make_adapter()
        queue = MagicMock()
        adapter._approval_queue = queue
        interaction = _FakeInteraction()
        await adapter._app_approve(interaction, "")
        assert interaction.all_messages == [
            "Usage: /prometheus ops approve {request_id}"
        ]

    @pytest.mark.asyncio
    async def test_approve_no_queue(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_approve(interaction, "abc123")
        assert interaction.all_messages == ["Approval queue not active."]

    @pytest.mark.asyncio
    async def test_approve_with_queue_delegates(self):
        adapter = _make_adapter()
        queue = MagicMock()
        queue.approve = AsyncMock(return_value=True)
        adapter._approval_queue = queue
        interaction = _FakeInteraction()
        await adapter._app_approve(interaction, "abc123")
        assert interaction.all_messages == ["Approved: abc123"]
        queue.approve.assert_awaited_once_with("abc123")

    @pytest.mark.asyncio
    async def test_deny_not_found(self):
        adapter = _make_adapter()
        queue = MagicMock()
        queue.deny = AsyncMock(return_value=False)
        adapter._approval_queue = queue
        interaction = _FakeInteraction()
        await adapter._app_deny(interaction, "zzz")
        assert interaction.all_messages == ["No pending request: zzz"]

    @pytest.mark.asyncio
    async def test_pending_lists(self):
        adapter = _make_adapter()
        action = MagicMock()
        action.request_id = "r1"
        action.tool_name = "bash"
        action.description = "do something"
        queue = MagicMock()
        queue.list_pending.return_value = [action]
        adapter._approval_queue = queue
        interaction = _FakeInteraction()
        await adapter._app_pending(interaction, "")
        assert interaction.all_messages == [
            "Pending approval requests:\n  r1: bash — do something"
        ]


class TestProviderOverrideHandlers:
    @pytest.mark.asyncio
    async def test_claude_no_router(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_claude(interaction, "")
        assert interaction.all_messages == [
            "Routing is not enabled. Provider overrides require a "
            "configured router in prometheus.yaml."
        ]

    @pytest.mark.asyncio
    async def test_override_success_keys_on_discord_channel(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        agent_loop = AsyncMock()
        router = MagicMock()
        router.config.overrides_enabled = True
        agent_loop._model_router = router
        adapter = _make_adapter(
            agent_loop=agent_loop,
            prometheus_config={
                "slash_commands": {"claude": {
                    "provider": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-x",
                }},
            },
        )
        interaction = _FakeInteraction(channel_id=42)
        await adapter._app_claude(interaction, "")
        router.set_override.assert_called_once()
        session_key, preset = router.set_override.call_args[0]
        assert session_key == "discord:42"
        assert preset["model"] == "claude-x"
        text = interaction.all_messages[0]
        assert "Switched to Claude (anthropic)." in text
        # Discord-native command paths in the reply, not Telegram's/Slack's.
        assert "/prometheus provider local" in text
        assert "/prometheus provider route" in text

    @pytest.mark.asyncio
    async def test_override_inline_text_gets_honest_note(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        agent_loop = AsyncMock()
        router = MagicMock()
        router.config.overrides_enabled = True
        agent_loop._model_router = router
        adapter = _make_adapter(
            agent_loop=agent_loop,
            prometheus_config={
                "slash_commands": {"claude": {
                    "provider": "anthropic",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-x",
                }},
            },
        )
        interaction = _FakeInteraction()
        await adapter._app_claude(interaction, "what is 2+2?")
        assert "inline message dispatch isn't supported on Discord" in (
            interaction.all_messages[0]
        )

    @pytest.mark.asyncio
    async def test_local_no_override(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_local(interaction, "")
        assert interaction.all_messages == [
            "Already on primary (llama_cpp/test-model-v1). No override was set."
        ]

    @pytest.mark.asyncio
    async def test_route_no_router_uses_discord_prefix(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_route(interaction, "")
        text = interaction.all_messages[0]
        assert text.startswith(
            "Route\nActive: llama_cpp/test-model-v1  (no router)"
        )
        assert "/prometheus provider claude" in text
        assert "/prometheus provider local" in text
        # No bare-Telegram command names leak into the Discord surface.
        assert "\n  /claude" not in text


class TestSubsystemCommandHandlers:
    @pytest.mark.asyncio
    async def test_escalations_not_available(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_escalations(interaction, "")
        assert interaction.all_messages == [
            "Teacher escalation: not available in this build."
        ]

    @pytest.mark.asyncio
    async def test_gepa_status_no_engine(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_gepa(interaction, "")
        assert interaction.all_messages == [
            "GEPA: engine not active (set learning.gepa_enabled in config)."
        ]
        assert interaction.response.deferred  # acked within Discord's 3s

    @pytest.mark.asyncio
    async def test_gepa_usage_uses_discord_prefix(self):
        adapter = _make_adapter()
        adapter._gepa_engine = MagicMock()
        interaction = _FakeInteraction()
        await adapter._app_gepa(interaction, "bogus")
        assert interaction.all_messages == [
            "Usage: /prometheus ops gepa [status | run | history]"
        ]

    @pytest.mark.asyncio
    async def test_symbiote_inactive(self, monkeypatch):
        import prometheus.symbiote as sym
        monkeypatch.setattr(sym, "get_coordinator", lambda: None)
        adapter = _make_adapter()
        interaction = _FakeInteraction(channel=None)
        await adapter._app_symbiote(interaction, "fix the parser")
        assert interaction.all_messages == [
            "SYMBIOTE is not active. Set symbiote.enabled in config."
        ]

    @pytest.mark.asyncio
    async def test_press_inactive(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction(channel=None)
        await adapter._app_press(interaction, "list")
        assert interaction.all_messages == [
            "Printing Press is not active. The library clone is missing "
            "(searched ~/printing-press-library/ and /tmp/printing-press-library/) "
            "or the feature is disabled in config."
        ]

    @pytest.mark.asyncio
    async def test_audit_usage_uses_discord_prefix(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction(channel=None)
        await adapter._app_audit(interaction, "bogus-category")
        text = interaction.all_messages[0]
        assert "/prometheus ops audit run" in text
        assert "Categories: search, fetch, youtube" in text

    @pytest.mark.asyncio
    async def test_voice_is_platform_honest(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_voice(interaction, "")
        text = interaction.all_messages[0]
        assert "not supported on Discord yet" in text
        assert "Telegram" in text  # says WHY, not just no
        assert "Whisper" in text   # and that voice INPUT works

    @pytest.mark.asyncio
    async def test_note_without_store(self, monkeypatch):
        import prometheus.tools.builtin.wiki_compile as wc
        monkeypatch.setattr(wc, "_memory_store", None)
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_note(interaction, "remember this")
        assert interaction.all_messages == [
            "Memory store unavailable — note not saved."
        ]

    @pytest.mark.asyncio
    async def test_memory_usage_shows_full_command_paths(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_memory(interaction, "")
        text = interaction.all_messages[0]
        assert "/prometheus core memory show" in text
        assert "/prometheus core memory limits" in text

    @pytest.mark.asyncio
    async def test_reset_clears_discord_session(self):
        adapter = _make_adapter()
        adapter.session_manager.get_or_create("discord:123").add_user_message("x")
        interaction = _FakeInteraction(channel_id=123)
        await adapter._app_reset(interaction, "")
        assert interaction.all_messages == ["Conversation context reset."]
        session = adapter.session_manager._sessions.get("discord:123")
        assert session is None or session.get_messages() == []

    @pytest.mark.asyncio
    async def test_steer_without_session(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction(channel_id=321)
        await adapter._app_steer(interaction, "focus")
        assert "No active session yet" in interaction.all_messages[0]

    @pytest.mark.asyncio
    async def test_queue_creates_session(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction(channel_id=321)
        await adapter._app_queue(interaction, "next task please")
        session = adapter.session_manager._sessions["discord:321"]
        assert session.queued_prompts == ["next task please"]


class TestHelpAndChunkedResponses:
    @pytest.mark.asyncio
    async def test_help_lists_all_sections(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._app_help(interaction, "")
        text = "\n".join(interaction.all_messages)
        for section in ("core", "session", "ops", "provider"):
            assert f"/prometheus {section}" in text

    @pytest.mark.asyncio
    async def test_long_interaction_reply_chunks_via_followup(self):
        adapter = _make_adapter()
        interaction = _FakeInteraction()
        await adapter._respond(interaction, "z" * 4500)
        assert len(interaction.response.messages) == 1
        assert len(interaction.followup.messages) >= 1
        for m in interaction.all_messages:
            assert len(m) <= MAX_MESSAGE_LENGTH


# ---------------------------------------------------------------------------
# Media routing — shared services, routed by content type
# ---------------------------------------------------------------------------


class TestMediaRouting:
    @pytest.mark.asyncio
    async def test_image_routes_to_vision(self, monkeypatch, tmp_path):
        import prometheus.gateway.media_cache as mc
        import prometheus.gateway.media_services as ms

        cached = tmp_path / "img.png"
        cached.write_bytes(b"png")
        monkeypatch.setattr(
            mc, "cache_image_from_bytes", lambda data, ext=".jpg": str(cached),
        )
        describe = AsyncMock(return_value="a red fox")
        monkeypatch.setattr(ms, "describe_image", describe)

        adapter = _make_adapter()
        att = _fake_attachment(filename="fox.png", content_type="image/png")
        text, urls, types, mtype = await adapter._ingest_attachments(
            [att], caption="look!", chat_id=1,
        )
        describe.assert_awaited_once()
        assert describe.call_args.args[0] == str(cached)
        assert text == "look!\n\n[Image: a red fox]"
        assert urls == [str(cached)]
        assert types == ["image/png"]
        assert mtype == MessageType.PHOTO

    @pytest.mark.asyncio
    async def test_audio_routes_to_whisper(self, monkeypatch, tmp_path):
        import prometheus.gateway.media_cache as mc
        import prometheus.gateway.media_services as ms

        cached = tmp_path / "voice.ogg"
        cached.write_bytes(b"ogg")
        monkeypatch.setattr(
            mc, "cache_audio_from_bytes", lambda data, ext=".ogg": str(cached),
        )
        transcribe = AsyncMock(return_value="hello prometheus")
        monkeypatch.setattr(ms, "transcribe_audio", transcribe)

        adapter = _make_adapter()
        att = _fake_attachment(filename="voice-message.ogg", content_type="audio/ogg")
        text, urls, types, mtype = await adapter._ingest_attachments(
            [att], caption="", chat_id=1,
        )
        transcribe.assert_awaited_once_with(str(cached))
        assert text == "hello prometheus"
        assert mtype == MessageType.VOICE

    @pytest.mark.asyncio
    async def test_audio_transcription_unavailable_fallback(self, monkeypatch, tmp_path):
        import prometheus.gateway.media_cache as mc
        import prometheus.gateway.media_services as ms

        monkeypatch.setattr(
            mc, "cache_audio_from_bytes",
            lambda data, ext=".ogg": str(tmp_path / "v.ogg"),
        )
        monkeypatch.setattr(ms, "transcribe_audio", AsyncMock(return_value=None))
        adapter = _make_adapter()
        att = _fake_attachment(filename="v.ogg", content_type="audio/ogg")
        text, *_ = await adapter._ingest_attachments([att], caption="", chat_id=1)
        assert text == "[Voice message received but transcription unavailable]"

    @pytest.mark.asyncio
    async def test_document_routes_to_extraction(self, monkeypatch, tmp_path):
        import prometheus.gateway.media_cache as mc
        import prometheus.utils.file_extract as fx

        cached = tmp_path / "doc_notes.txt"
        cached.write_text("file contents")
        monkeypatch.setattr(
            mc, "cache_document_from_bytes", lambda data, name: str(cached),
        )
        monkeypatch.setattr(fx, "extract_text", lambda path: "file contents")

        adapter = _make_adapter(
            prometheus_config={"context": {"effective_limit": 24000}},
        )
        adapter.agent_loop._provider = None
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        att = _fake_attachment(filename="notes.txt", content_type="text/plain")
        text, urls, types, mtype = await adapter._ingest_attachments(
            [att], caption="", chat_id=1,
        )
        assert text == "[Content of notes.txt]:\nfile contents"
        assert types == ["text/plain"]
        assert mtype == MessageType.DOCUMENT

    @pytest.mark.asyncio
    async def test_unsupported_document_reports_and_skips(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock()
        att = _fake_attachment(
            filename="virus.exe", content_type="application/x-msdownload",
        )
        text, urls, types, mtype = await adapter._ingest_attachments(
            [att], caption="", chat_id=1,
        )
        assert text == ""
        adapter.send.assert_awaited_once()
        att.read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_oversize_attachment_rejected(self):
        adapter = _make_adapter()
        adapter.send = AsyncMock()
        att = _fake_attachment(
            filename="big.png", content_type="image/png",
            size=25 * 1024 * 1024,
        )
        text, *_ = await adapter._ingest_attachments([att], caption="", chat_id=1)
        assert text == ""
        assert "too large" in adapter.send.call_args.args[1]
        att.read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_message_flows_to_agent_with_description(
        self, monkeypatch, tmp_path,
    ):
        """End-to-end: attachment → shared vision service → agent turn."""
        import prometheus.gateway.media_cache as mc
        import prometheus.gateway.media_services as ms

        monkeypatch.setattr(
            mc, "cache_image_from_bytes",
            lambda data, ext=".jpg": str(tmp_path / "i.png"),
        )
        monkeypatch.setattr(
            ms, "describe_image", AsyncMock(return_value="a whiteboard"),
        )
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=_agent_result())
        adapter = _make_adapter(agent_loop=agent_loop)
        adapter.tool_registry.list_schemas = MagicMock(return_value=[])
        att = _fake_attachment(filename="w.png", content_type="image/png")
        msg = _fake_message(channel_id=555, content="what's this?", attachments=[att])

        await adapter._handle_discord_message(msg)

        agent_loop.run_async.assert_awaited_once()
        messages = agent_loop.run_async.call_args.kwargs["messages"]
        user_texts = [
            m["content"] if isinstance(m, dict) else getattr(m, "content", "")
            for m in messages
        ]
        assert any(
            "what's this?" in str(t) and "[Image: a whiteboard]" in str(t)
            for t in user_texts
        )


# ---------------------------------------------------------------------------
# Subsystem registry inheritance (the G1 contract, exercised for Discord)
# ---------------------------------------------------------------------------


class TestRegistryInheritance:
    def test_register_adapter_inherits_all_subsystems(self):
        adapter = _make_adapter()
        reg = GatewaySubsystemRegistry()
        subsystems = {
            "cost_tracker": object(),
            "escalation_engine": object(),
            "_approval_queue": object(),
            "_gepa_engine": object(),
            "_printing_press": object(),
            "_backup_vault": object(),
            "_morph_engine": object(),
        }
        # Attach BEFORE registering — the replay path daemon.py relies on.
        for name, value in subsystems.items():
            reg.attach(name, value)
        reg.register_adapter(adapter)
        for name, value in subsystems.items():
            assert getattr(adapter, name) is value

    def test_signal_bus_setter_subscribes(self):
        adapter = _make_adapter()
        reg = GatewaySubsystemRegistry()
        reg.register_adapter(adapter)
        bus = MagicMock()
        reg.attach("signal_bus", bus)
        kinds = {call.args[0] for call in bus.subscribe.call_args_list}
        assert kinds == {
            "skill_created", "skill_refined", "memory_updated", "curator_report",
        }
        assert adapter.signal_bus is bus

    @pytest.mark.asyncio
    async def test_notifications_use_last_channel(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
        adapter = _make_adapter()
        adapter._client = MagicMock()  # "initialized"
        adapter.send = AsyncMock()
        adapter._save_channel(999)
        signal = MagicMock()
        signal.payload = {"skill_name": "pdf-wrangler"}
        await adapter._on_signal_skill_created(signal)
        adapter.send.assert_awaited_once()
        assert adapter.send.call_args.args[0] == 999
        assert "pdf-wrangler" in adapter.send.call_args.args[1]


# ---------------------------------------------------------------------------
# Outbound send()
# ---------------------------------------------------------------------------


class TestSend:
    @pytest.mark.asyncio
    async def test_send_not_initialized(self):
        adapter = _make_adapter()
        result = await adapter.send(1, "hi")
        assert not result.success
        assert result.error == "Bot not initialized"

    @pytest.mark.asyncio
    async def test_send_chunks_long_text(self):
        adapter = _make_adapter()
        channel = MagicMock()
        sent_msg = MagicMock()
        sent_msg.id = 77
        channel.send = AsyncMock(return_value=sent_msg)
        client = MagicMock()
        client.get_channel = MagicMock(return_value=channel)
        adapter._client = client

        result = await adapter.send(1, "x" * 4100)
        assert result.success
        assert result.message_id == 77
        assert channel.send.await_count == 3
        for call in channel.send.call_args_list:
            assert len(call.args[0]) <= MAX_MESSAGE_LENGTH
