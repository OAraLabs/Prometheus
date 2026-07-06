"""Discord platform adapter — gateway bot using discord.py (SPRINT G2).

Receives DMs + whitelisted guild-channel messages over the Discord gateway
WebSocket, dispatches to AgentLoop, sends responses back. Mirrors the
Telegram/Slack adapter patterns:

- Import-guarded: discord.py is an optional extra (``pip install
  'oara-prometheus[discord]'``); this module imports without it and only
  ``start()`` requires it (exactly like slack.py and slack-bolt).
- Whitelist: DMs always allowed; guild channels require an explicit
  ``guild_ids`` / ``channel_ids`` whitelist hit — with both empty the bot is
  DMs-only. See ``PlatformConfig.discord_inbound_allowed`` for why this is
  deliberately stricter than Telegram's empty-``allowed_chat_ids`` = open.
- Session keys: ``discord:<channel_id>`` (DM channels included), the same
  ``<platform>:<chat>`` namespace as telegram/slack.
- Replies chunk at Discord's 2000-char message limit; long replies open a
  thread off the user's message (mirror of Slack's long_reply_threshold).
- Emoji reactions for processing feedback (👀 while working → ✅ done),
  the Discord-native twin of Slack's eyes → white_check_mark.
- Media: attachments route by content type through the SAME shared services
  Telegram uses (``prometheus.gateway.media_services``): images → vision
  analysis, audio/voice → Whisper STT, documents → text extraction with
  context-budget truncation.

Slash commands are Discord *app commands* under ONE top-level ``/prometheus``
command (43 top-level commands would pollute the user's picker). Discord caps
a command at 25 options, so the 43 families cannot sit flat under
``/prometheus <family>``; they are grouped one level deeper —
``/prometheus <section> <family> [args]`` with sections ``core`` /
``session`` / ``ops`` / ``provider``. Sections are chosen so that every
cross-command reference rendered by the shared command layer (e.g. GEPA's
"watch for the …approve prompt") stays within one section and therefore one
``prefix`` string. Registration is guild-scoped when a guild whitelist is
present (commands appear instantly) and global otherwise (Discord takes up
to ~1 hour to propagate global commands).

Markdown: Discord renders standard markdown natively (bold, italic,
strikethrough, inline code, fenced code blocks, headers, block quotes,
lists, and masked links in bot messages), so agent output is passed through
unmodified. NOT translated: markdown tables and raw HTML render as plain
text — see ``format_markdown_for_discord``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

if TYPE_CHECKING:
    from prometheus.engine.agent_loop import AgentLoop
    from prometheus.engine.session import ChatSession, SessionManager
    from prometheus.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Discord message content hard limit (characters)
MAX_MESSAGE_LENGTH = 2000

# Threshold above which replies move into a thread (mirror of
# gateway.slack.long_reply_threshold; override via
# gateway.discord.long_reply_threshold in prometheus.yaml).
DEFAULT_LONG_REPLY_THRESHOLD = 800

# SPRINT G2: all families live under ONE ``/prometheus`` app command, one
# section group deep (Discord caps a command at 25 options — 43 families
# don't fit flat). Shared command text that references sibling commands is
# built with the section prefix so Discord users see commands they can
# actually type.
CORE_PREFIX = "/prometheus core "
SESSION_PREFIX = "/prometheus session "
OPS_PREFIX = "/prometheus ops "
PROVIDER_PREFIX = "/prometheus provider "

# Processing-feedback reactions (Slack's eyes → white_check_mark, natively)
REACTION_PROCESSING = "\N{EYES}"          # 👀
REACTION_DONE = "\N{WHITE HEAVY CHECK MARK}"  # ✅


def chunk_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks respecting Discord's 2000-char limit.

    Tries to split at paragraph boundaries, then newlines, then spaces.
    (Same strategy as the Slack adapter's chunker.)
    """
    if not text or len(text) <= max_length:
        return [text] if text else [""]

    chunks: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_length:
            chunks.append(remaining)
            break

        # Try paragraph boundary
        cut = remaining.rfind("\n\n", 0, max_length)
        if cut <= 0:
            # Try newline
            cut = remaining.rfind("\n", 0, max_length)
        if cut <= 0:
            # Try space
            cut = remaining.rfind(" ", 0, max_length)
        if cut <= 0:
            # Hard truncate
            cut = max_length

        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    return chunks


def format_markdown_for_discord(content: str) -> str:
    """Prepare agent markdown for Discord.

    Discord renders standard markdown natively — **bold**, *italic*,
    ~~strikethrough~~, `inline code`, ``` fenced code blocks, # headers,
    > block quotes, bullet/numbered lists, and masked ``[text](url)`` links
    (bot messages support them) — so unlike Slack's mrkdwn there is nothing
    to convert and the text is passed through unmodified.

    Deliberately NOT translated (Discord has no rendering for them; they
    appear as plain text): markdown tables, raw HTML tags, and LaTeX. If a
    model leans on tables heavily, a future pass can reflow them into code
    blocks — kept out of G2 to avoid mangling code samples.
    """
    return content


class DiscordAdapter(BasePlatformAdapter):
    """Discord bot adapter — receives messages, routes to AgentLoop."""

    def __init__(
        self,
        config: PlatformConfig,
        agent_loop: AgentLoop,
        tool_registry: ToolRegistry,
        system_prompt: str = "You are Prometheus, a helpful AI assistant.",
        model_name: str = "",
        model_provider: str = "",
        session_manager: SessionManager | None = None,
        prometheus_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.agent_loop = agent_loop
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.model_provider = model_provider
        self._client: Any = None
        self._tree: Any = None
        self._connect_task: asyncio.Task | None = None
        self._tree_synced = False
        self._start_time: float = 0.0
        self._prometheus_config: dict[str, Any] = prometheus_config or {}

        if session_manager is None:
            from prometheus.engine.session import SessionManager as _SM
            session_manager = _SM()
        self.session_manager: SessionManager = session_manager

        # SignalBus wired by daemon.py via the GatewaySubsystemRegistry
        # (property setter below). Until then it's None and the subscribe
        # path is a no-op — mirror of the Telegram/Slack pattern.
        self._signal_bus: object | None = None

    # ------------------------------------------------------------------
    # SignalBus subscription + last-channel tracking (mirror of both
    # existing adapters)
    # ------------------------------------------------------------------

    @property
    def signal_bus(self) -> object | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: object | None) -> None:
        """Subscribe to skill/memory/curator events on bus assignment.

        Set from daemon.py through the GatewaySubsystemRegistry after the
        SignalBus is constructed. Mirrors the Telegram/Slack subscriptions
        so SKILL_CREATED / SKILL_REFINED / MEMORY_UPDATED / CURATOR_REPORT
        notifications land in Discord too when Discord is enabled.
        """
        self._signal_bus = bus
        if bus is None:
            return
        try:
            bus.subscribe("skill_created", self._on_signal_skill_created)
            bus.subscribe("skill_refined", self._on_signal_skill_refined)
            bus.subscribe("memory_updated", self._on_signal_memory_updated)
            bus.subscribe("curator_report", self._on_signal_curator_report)
            logger.info("Discord: subscribed to skill/memory/curator signals")
        except Exception:
            logger.warning(
                "Discord: failed to subscribe to SignalBus", exc_info=True
            )

    def _notification_mode(self) -> str:
        """Return 'quiet' | 'verbose' | 'off'.

        Runtime override (set via /prometheus core notifications) wins.
        Falls back to gateway.discord.skill_event_notifications, then
        gateway.skill_event_notifications, then "quiet".
        """
        try:
            from prometheus.gateway.commands import get_notifications_mode
            mode = get_notifications_mode(default="")
            if mode:
                return mode
        except Exception:
            pass
        gw = self._prometheus_config.get("gateway", {}) or {}
        discord_cfg = gw.get("discord", {}) if isinstance(gw, dict) else {}
        if isinstance(discord_cfg, dict):
            mode = discord_cfg.get("skill_event_notifications")
            if mode:
                return str(mode).lower()
        if isinstance(gw, dict):
            return str(gw.get("skill_event_notifications", "quiet")).lower()
        return "quiet"

    def _long_reply_threshold(self) -> int:
        """Return char threshold above which replies move into a thread."""
        gw = self._prometheus_config.get("gateway", {}) or {}
        discord_cfg = gw.get("discord", {}) if isinstance(gw, dict) else {}
        if isinstance(discord_cfg, dict):
            try:
                return int(discord_cfg.get(
                    "long_reply_threshold", DEFAULT_LONG_REPLY_THRESHOLD,
                ))
            except (TypeError, ValueError):
                pass
        return DEFAULT_LONG_REPLY_THRESHOLD

    def _last_channel_path(self) -> str:
        from prometheus.config.paths import get_config_dir
        return str(get_config_dir() / "last_discord_channel")

    def _save_channel(self, channel_id: int | str) -> None:
        """Persist the last active Discord channel for signal notifications."""
        if not channel_id:
            return
        try:
            with open(self._last_channel_path(), "w") as f:
                f.write(str(channel_id))
        except Exception:
            pass

    def _load_channel(self) -> int | None:
        try:
            with open(self._last_channel_path()) as f:
                raw = f.read().strip()
                return int(raw) if raw else None
        except Exception:
            return None

    async def _send_notification(self, text: str) -> None:
        """Post *text* to the last active Discord channel (best-effort)."""
        channel_id = self._load_channel()
        if channel_id is None and self.config.allowed_channel_ids:
            channel_id = self.config.allowed_channel_ids[0]
        if channel_id is None or not self._client:
            return
        try:
            await self.send(channel_id, text)
        except Exception:
            logger.debug("Discord: notification send failed", exc_info=True)

    async def _on_signal_skill_created(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        name = payload.get("skill_name", "(unnamed)")
        text = f"🎓 New skill: **{name}**"
        if mode == "verbose":
            summary = payload.get("summary", "")
            trigger = payload.get("trigger_task", "")
            if summary:
                text += f"\n   {summary}"
            if trigger:
                text += f"\n   (built while: {trigger[:120]})"
        await self._send_notification(text)

    async def _on_signal_skill_refined(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        name = payload.get("skill_name", "(unnamed)")
        text = f"📚 Updated skill: **{name}**"
        if mode == "verbose":
            summary = payload.get("summary", "")
            if summary:
                text += f"\n   {summary}"
        await self._send_notification(text)

    async def _on_signal_memory_updated(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        target = payload.get("target", "memory")
        operation = payload.get("operation", "updated")
        text = f"🧠 {target} {operation}"
        if mode == "verbose":
            preview = payload.get("entry_preview", "")
            if preview:
                text += f"\n   {preview}"
        await self._send_notification(text)

    async def _on_signal_curator_report(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        reviewed = payload.get("skills_reviewed", 0)
        prunings = payload.get("prunings", 0)
        consolidations = payload.get("consolidations", 0)
        if mode == "verbose":
            text = (
                f"📋 Curator: {reviewed} skills reviewed, "
                f"{consolidations} consolidation suggestion(s), "
                f"{prunings} archived\n"
                f"   `{CORE_PREFIX}curator show` for the full report"
            )
        else:
            text = f"📋 Curator: {reviewed} reviewed, {prunings} archived"
        await self._send_notification(text)

    def _resolve_session_for_command(self, channel_id: Any) -> "ChatSession | None":
        """Return the live ChatSession for *channel_id*, or None.

        Mirrors the Telegram/Slack helper: used by steer/unqueue/clearsteers
        to find the session-state object the agent loop is consuming.
        Does NOT create a session.
        """
        if not channel_id:
            return None
        return self.session_manager._sessions.get(f"discord:{channel_id}")

    def _get_provider(self):
        """Return the model provider from the agent loop (for vision analysis)."""
        try:
            return self.agent_loop._provider
        except AttributeError:
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build the Discord client + app-command tree and connect."""
        try:
            import discord
            from discord import app_commands
        except ImportError:
            raise ImportError(
                "discord.py is required for Discord support. "
                "Install it with: pip install 'oara-prometheus[discord]'"
            )

        if not self.config.token:
            raise ValueError("Discord bot token is required")

        intents = discord.Intents.default()
        # Privileged intent: must ALSO be enabled on the bot's application
        # page in the Discord developer portal, or message.content arrives
        # empty. Required to read messages at all.
        intents.message_content = True

        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        self._client = client
        self._tree = tree
        self._register_app_commands(tree)

        adapter = self

        @client.event
        async def on_ready() -> None:  # pragma: no cover - needs live gateway
            await adapter._sync_command_tree()
            logger.info("Discord adapter ready as %s", client.user)

        @client.event
        async def on_message(message: Any) -> None:
            await adapter._handle_discord_message(message)

        # login() validates the token; connect() runs the gateway WebSocket
        # for the daemon's lifetime, so it goes to a background task (the
        # Slack Socket Mode handler makes the same non-blocking start).
        await client.login(self.config.token)
        self._connect_task = asyncio.create_task(
            client.connect(reconnect=True), name="discord_gateway",
        )
        self._running = True
        self._start_time = time.monotonic()
        logger.info("Discord adapter started (gateway connecting)")

    async def _sync_command_tree(self) -> None:
        """Sync app commands — guild-scoped when whitelisted, else global.

        Guild-scoped commands are available instantly in those guilds;
        global commands can take up to ~1 hour to propagate Discord-side.
        With a guild whitelist we deliberately skip the global sync, so the
        command picker only offers /prometheus inside whitelisted guilds.
        """
        if self._tree is None or self._tree_synced:
            return
        try:
            import discord
            if self.config.allowed_guild_ids:
                for gid in self.config.allowed_guild_ids:
                    guild = discord.Object(id=gid)
                    self._tree.copy_global_to(guild=guild)
                    await self._tree.sync(guild=guild)
                logger.info(
                    "Discord: app commands synced to %d guild(s) (instant)",
                    len(self.config.allowed_guild_ids),
                )
            else:
                await self._tree.sync()
                logger.info(
                    "Discord: app commands synced globally "
                    "(propagation can take up to ~1 hour)"
                )
            self._tree_synced = True
        except Exception:
            logger.warning("Discord: app-command sync failed", exc_info=True)

    async def stop(self) -> None:
        """Graceful shutdown of the Discord bot."""
        if not self._running:
            return
        self._running = False
        try:
            if self._client is not None:
                await self._client.close()
        except Exception:
            logger.debug("Discord: client close failed", exc_info=True)
        if self._connect_task is not None:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._connect_task = None
        logger.info("Discord adapter stopped")

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> SendResult:
        """Send a message to a Discord channel (DM channels included)."""
        if not self._client:
            return SendResult(success=False, error="Bot not initialized")

        channel = self._client.get_channel(int(chat_id))
        if channel is None:
            try:
                channel = await self._client.fetch_channel(int(chat_id))
            except Exception as exc:
                logger.error("Failed to resolve channel %s: %s", chat_id, exc)
                return SendResult(success=False, error=str(exc))

        last_id: int | None = None
        for chunk in chunk_message(format_markdown_for_discord(text)):
            try:
                msg = await channel.send(chunk)
                last_id = getattr(msg, "id", None)
            except Exception as exc:
                logger.error("Failed to send message to %s: %s", chat_id, exc)
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=last_id)

    # ------------------------------------------------------------------
    # Inbound
    # ------------------------------------------------------------------

    async def _handle_discord_message(self, message: Any) -> None:
        """Normalise a discord.Message into a MessageEvent and dispatch."""
        author = getattr(message, "author", None)
        if author is not None and getattr(author, "bot", False):
            return  # never react to bots (covers our own messages)

        guild = getattr(message, "guild", None)
        is_dm = guild is None
        channel = message.channel
        guild_id = getattr(guild, "id", None)

        if not self.config.discord_inbound_allowed(
            is_dm=is_dm, guild_id=guild_id, channel_id=channel.id,
        ):
            logger.debug(
                "Discord: ignoring message in non-whitelisted channel %s "
                "(guild %s)", channel.id, guild_id,
            )
            return

        caption = (getattr(message, "content", "") or "").strip()
        attachments = list(getattr(message, "attachments", []) or [])

        text = caption
        message_type = MessageType.TEXT
        media_urls: list[str] = []
        media_types: list[str] = []
        if attachments:
            ingested = await self._ingest_attachments(
                attachments, caption=caption, chat_id=channel.id,
            )
            text, media_urls, media_types, message_type = ingested

        if not text:
            return

        event = MessageEvent(
            chat_id=channel.id,
            user_id=getattr(author, "id", 0) or 0,
            text=text,
            message_id=getattr(message, "id", 0) or 0,
            platform=Platform.DISCORD,
            message_type=message_type,
            username=str(author) if author is not None else None,
            raw={
                "is_dm": is_dm,
                "guild_id": guild_id,
                "message": message,
            },
            media_urls=media_urls,
            media_types=media_types,
            caption=(caption or None) if attachments else None,
        )
        await self.on_message(event)

    async def on_message(self, event: MessageEvent) -> None:
        """Handle an incoming message — whitelist, then dispatch to agent."""
        raw = event.raw or {}
        if not self.config.discord_inbound_allowed(
            is_dm=bool(raw.get("is_dm")),
            guild_id=raw.get("guild_id"),
            channel_id=event.chat_id,
        ):
            logger.warning(
                "Ignoring message from non-whitelisted Discord channel %s",
                event.chat_id,
            )
            return
        # Remember the last active channel so SignalBus notifications land
        # somewhere meaningful (mirror of Telegram/Slack).
        self._save_channel(event.chat_id)
        await self._dispatch_to_agent(event)

    async def _dispatch_to_agent(self, event: MessageEvent) -> None:
        """Route a message through AgentLoop and send the response."""
        message = (event.raw or {}).get("message")

        # 👀 while processing (Discord's native twin of Slack's reaction ack)
        await self._add_reaction(message, REACTION_PROCESSING)

        session_id = event.session_key()  # "discord:<channel_id>"
        session = self.session_manager.get_or_create(session_id)
        session.add_user_message(event.text)
        pre_len = len(session.get_messages())

        try:
            result = await self.agent_loop.run_async(
                system_prompt=self.system_prompt,
                messages=session.get_messages(),
                tools=self.tool_registry.list_schemas(),
                # Per-channel override namespace (provider overrides etc.).
                session_id=session_id,
            )
            session.add_result_messages(result.messages, pre_len)
            session.trim(self.session_manager.MAX_SESSION_MESSAGES)
            response_text = result.text or "(no response)"
        except Exception as exc:
            logger.error("Agent error for channel %s: %s", event.chat_id, exc)
            session.rollback_last()
            response_text = f"Error: {exc}"

        response_text = format_markdown_for_discord(response_text)
        await self._deliver_reply(event, message, response_text)

        # Swap 👀 for ✅ when done
        await self._remove_reaction(message, REACTION_PROCESSING)
        await self._add_reaction(message, REACTION_DONE)

        # Drain queued prompts as fresh user turns (mirror of Telegram's
        # /queue semantics — loop until empty so back-to-back queues all run).
        while True:
            next_prompt = session.drain_prompt()
            if not next_prompt:
                break
            queued_event = MessageEvent(
                chat_id=event.chat_id,
                user_id=event.user_id,
                text=next_prompt,
                message_id=event.message_id,
                platform=event.platform,
                username=event.username,
                raw={k: v for k, v in (event.raw or {}).items()},
            )
            await self._dispatch_to_agent(queued_event)

    async def _deliver_reply(
        self, event: MessageEvent, message: Any, text: str,
    ) -> None:
        """Send *text* back — long replies open a thread off the message.

        Threading policy (mirror of Slack's long_reply_threshold pattern):
        replies longer than the threshold move into a thread created from
        the user's message so guild channels aren't flooded; DMs and
        thread-nested messages fall back to plain channel sends (Discord
        has no DM threads / nested threads).
        """
        target = None
        if message is not None and len(text) > self._long_reply_threshold():
            target = await self._maybe_create_thread(message, text)

        chunks = chunk_message(text)
        if target is not None:
            for chunk in chunks:
                try:
                    await target.send(chunk)
                except Exception as exc:
                    logger.error(
                        "Failed to send thread reply to %s: %s",
                        event.chat_id, exc,
                    )
            return

        if message is not None:
            for chunk in chunks:
                try:
                    await message.channel.send(chunk)
                except Exception as exc:
                    logger.error(
                        "Failed to send response to %s: %s", event.chat_id, exc,
                    )
        else:
            await self.send(event.chat_id, text)

    async def _maybe_create_thread(self, message: Any, text: str) -> Any:
        """Create a reply thread off *message*, or None when unsupported.

        create_thread raises for DM channels, messages already inside a
        thread, and missing permissions — all fall back to channel sends.
        """
        if getattr(message, "guild", None) is None:
            return None  # DMs have no threads
        create_thread = getattr(message, "create_thread", None)
        if create_thread is None:
            return None
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        name = (first_line[:80] or "Prometheus reply")
        try:
            return await create_thread(name=name)
        except Exception:
            logger.debug(
                "Discord: thread creation failed; replying in channel",
                exc_info=True,
            )
            return None

    async def _add_reaction(self, message: Any, emoji: str) -> None:
        """Add an emoji reaction to a message (best-effort)."""
        if message is None:
            return
        try:
            await message.add_reaction(emoji)
        except Exception:
            pass  # best-effort — may lack permission

    async def _remove_reaction(self, message: Any, emoji: str) -> None:
        """Remove our emoji reaction from a message (best-effort)."""
        if message is None or self._client is None:
            return
        try:
            await message.remove_reaction(emoji, self._client.user)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Media ingestion — routes through the SAME shared services Telegram
    # uses (prometheus.gateway.media_services / media_cache): images →
    # vision analysis, audio → Whisper STT, documents → text extraction
    # with context-budget truncation.
    # ------------------------------------------------------------------

    async def _ingest_attachments(
        self, attachments: list[Any], *, caption: str, chat_id: int,
    ) -> tuple[str, list[str], list[str], MessageType]:
        """Download + route attachments by content type.

        Returns (user_text, media_urls, media_types, message_type); empty
        user_text means nothing usable arrived (errors were already
        reported to the channel).
        """
        from prometheus.gateway import media_services
        from prometheus.gateway.media_cache import (
            SUPPORTED_DOCUMENT_TYPES,
            cache_audio_from_bytes,
            cache_document_from_bytes,
            cache_image_from_bytes,
        )
        from prometheus.utils.file_extract import (
            extract_text,
            is_supported as is_extractable,
            unsupported_message,
        )

        parts: list[str] = []
        media_urls: list[str] = []
        media_types: list[str] = []
        message_type = MessageType.TEXT
        max_bytes = self.config.max_file_size_mb * 1024 * 1024

        for att in attachments:
            filename = getattr(att, "filename", "") or "unknown"
            content_type = (
                (getattr(att, "content_type", "") or "")
                .split(";")[0].strip().lower()
            )
            size = getattr(att, "size", 0) or 0
            if size > max_bytes:
                await self.send(
                    chat_id,
                    f"Attachment too large "
                    f"(max {self.config.max_file_size_mb} MB): {filename}",
                )
                continue

            ext = Path(filename).suffix.lower()

            if content_type.startswith("image/"):
                data = await self._read_attachment(att, chat_id, filename)
                if data is None:
                    continue
                cached = cache_image_from_bytes(data, ext=ext or ".jpg")
                description = await media_services.describe_image(
                    cached, provider=self._get_provider(),
                )
                if description:
                    parts.append(f"[Image: {description}]")
                else:
                    parts.append(f"[The user sent an image: {filename}]")
                media_urls.append(cached)
                media_types.append(content_type or "image/unknown")
                message_type = MessageType.PHOTO

            elif content_type.startswith("audio/"):
                # Covers Discord voice messages (audio/ogg) and audio files.
                data = await self._read_attachment(att, chat_id, filename)
                if data is None:
                    continue
                cached = cache_audio_from_bytes(data, ext=ext or ".ogg")
                transcription = await media_services.transcribe_audio(cached)
                parts.append(
                    transcription
                    or "[Voice message received but transcription unavailable]"
                )
                media_urls.append(cached)
                media_types.append(content_type or "audio/unknown")
                message_type = MessageType.VOICE

            else:
                # Document path — same validation set as Telegram's handler.
                if ext not in SUPPORTED_DOCUMENT_TYPES and not is_extractable(filename):
                    await self.send(chat_id, unsupported_message(filename))
                    continue
                data = await self._read_attachment(att, chat_id, filename)
                if data is None:
                    continue
                cached = cache_document_from_bytes(data, filename)
                extracted = extract_text(cached)
                if extracted:
                    extracted = media_services.truncate_for_context(
                        extracted,
                        prometheus_config=self._prometheus_config,
                        system_prompt=self.system_prompt,
                        tool_registry=self.tool_registry,
                        provider=self._get_provider(),
                    )
                    parts.append(f"[Content of {filename}]:\n{extracted}")
                else:
                    parts.append(f"[The user sent a document: {filename}]")
                media_urls.append(cached)
                media_types.append(
                    SUPPORTED_DOCUMENT_TYPES.get(
                        ext, content_type or "application/octet-stream",
                    )
                )
                message_type = MessageType.DOCUMENT

        media_text = "\n".join(p for p in parts if p)
        if caption and media_text:
            text = f"{caption}\n\n{media_text}"
        else:
            text = caption or media_text
        return text, media_urls, media_types, message_type

    async def _read_attachment(
        self, att: Any, chat_id: int, filename: str,
    ) -> bytes | None:
        """Download attachment bytes (best-effort, error reported to chat)."""
        try:
            return bytes(await att.read())
        except Exception as exc:
            logger.error("Failed to download attachment %s: %s", filename, exc)
            await self.send(chat_id, f"Failed to download attachment: {filename}")
            return None

    # ------------------------------------------------------------------
    # App commands — ONE /prometheus command, four section groups.
    # Every handler below is a thin wrapper over the shared commands layer
    # (prometheus.gateway.commands); subsystems (approval queue, GEPA,
    # symbiote engines, printing press, escalation engine, cost tracker)
    # arrive via the daemon's GatewaySubsystemRegistry.
    # ------------------------------------------------------------------

    def _register(self, group: Any, name: str, handler: Any, description: str = "") -> None:
        """Attach one family as an app subcommand under *group*.

        The callback takes a single optional ``args`` string option —
        Discord's typed-options model is deliberately not used per-family so
        the shared command layer keeps parsing its own subcommand grammar
        (identical text semantics across telegram/slack/discord).
        """
        import discord
        from discord import app_commands

        async def _callback(interaction, args: str = "") -> None:
            await handler(interaction, (args or "").strip())

        _callback.__name__ = f"prometheus_{group.name}_{name}"
        # Real class objects (module imports discord lazily, so inline
        # string annotations would not resolve against module globals).
        _callback.__annotations__ = {
            "interaction": discord.Interaction, "args": str, "return": None,
        }
        command = app_commands.Command(
            name=name,
            description=(description or name)[:100],
            callback=_callback,
        )
        group.add_command(command)

    def _register_app_commands(self, tree: Any) -> None:
        """Build the /prometheus command tree (43 families, 4 sections)."""
        from discord import app_commands

        root = app_commands.Group(
            name="prometheus", description="Prometheus — sovereign AI agent",
        )
        core = app_commands.Group(
            name="core", description="Status, memory, skills, diagnostics",
            parent=root,
        )
        session = app_commands.Group(
            name="session", description="Steer and queue the running agent",
            parent=root,
        )
        ops = app_commands.Group(
            name="ops", description="Approvals and autonomy subsystems",
            parent=root,
        )
        provider = app_commands.Group(
            name="provider",
            description="Per-channel model provider overrides",
            parent=root,
        )

        # -- core ----------------------------------------------------------
        self._register(core, "help", self._app_help, "List all Prometheus commands")
        self._register(core, "start", self._app_start, "Check that Prometheus is online")
        self._register(core, "reset", self._app_reset, "Clear conversation context for this channel")
        self._register(core, "clear", self._app_clear, "Alias of reset — clear this channel's context")
        self._register(core, "status", self._app_status, "Model, uptime, tools, memory, SENTINEL")
        self._register(core, "model", self._app_model, "Current model and provider")
        self._register(core, "profile", self._app_profile, "Show / switch agent profile")
        self._register(core, "context", self._app_context, "Context window usage")
        self._register(core, "benchmark", self._app_benchmark, "Quick smoke test")
        self._register(core, "beacon", self._app_beacon, "Web bridge / dashboard URL")
        self._register(core, "wiki", self._app_wiki, "Wiki stats + recent entries")
        self._register(core, "anatomy", self._app_anatomy, "Host, GPU, VRAM, Tailscale")
        self._register(core, "doctor", self._app_doctor, "Diagnostic health check")
        self._register(core, "tools", self._app_tools, "Tool-call stats (24h)")
        self._register(core, "note", self._app_note, "Save a manual fact: [@entity] <text>")
        self._register(core, "sentinel", self._app_sentinel, "SENTINEL subsystem state")
        self._register(core, "health", self._app_health, "Silent-failure telemetry: [hours] [verbose]")
        self._register(core, "events", self._app_events, "Signal-bus events: recent | skills | memory | curator")
        self._register(core, "pairs", self._app_pairs, "Training-pair flywheel stats")
        self._register(core, "voice", self._app_voice, "Voice reply modes (Telegram-only pipeline)")
        self._register(core, "memory", self._app_memory, "MEMORY.md / USER.md: show [user] | limits")
        self._register(core, "skills", self._app_skills, "Skills: list | show | pin | unpin | history")
        self._register(core, "curator", self._app_curator, "Curator: status | show | run [dry]")
        self._register(core, "notifications", self._app_notifications, "Skill/memory notifications: off | quiet | verbose")

        # -- session -------------------------------------------------------
        self._register(session, "steer", self._app_steer, "Inject mid-turn guidance")
        self._register(session, "queue", self._app_queue, "Queue a follow-up turn")
        self._register(session, "unqueue", self._app_unqueue, "Drop the last queued prompt")
        self._register(session, "clearsteers", self._app_clearsteers, "Drop all pending steers")

        # -- ops -------------------------------------------------------------
        self._register(ops, "approve", self._app_approve, "Approve a pending tool request")
        self._register(ops, "deny", self._app_deny, "Deny a pending tool request")
        self._register(ops, "pending", self._app_pending, "List pending approval requests")
        self._register(ops, "gepa", self._app_gepa, "Skill evolution: status | run | history")
        self._register(ops, "symbiote", self._app_symbiote, "GitHub graft pipeline: status | history | …")
        self._register(ops, "audit", self._app_audit, "Web capability audit: run | <category>")
        self._register(ops, "press", self._app_press, "CLI library: list | search | install | …")
        self._register(ops, "escalations", self._app_escalations, "Teacher-escalation counters")

        # -- provider --------------------------------------------------------
        self._register(provider, "claude", self._app_claude, "Route this channel via Anthropic Claude")
        self._register(provider, "gpt", self._app_gpt, "Route this channel via OpenAI GPT")
        self._register(provider, "gemini", self._app_gemini, "Route this channel via Google Gemini")
        self._register(provider, "xai", self._app_xai, "Route this channel via xAI Grok")
        self._register(provider, "grok", self._app_grok, "Alias for xai")
        # CLOUD EXPANSION (2026-07): four more provider override commands
        self._register(provider, "deepseek", self._app_deepseek, "Route this channel via DeepSeek")
        self._register(provider, "kimi", self._app_kimi, "Route this channel via Kimi (Moonshot)")
        self._register(provider, "glm", self._app_glm, "Route this channel via GLM (Z.ai)")
        self._register(provider, "mimo", self._app_mimo, "Route this channel via MiMo (Xiaomi)")
        self._register(provider, "local", self._app_local, "Clear the override, back to primary")
        self._register(provider, "route", self._app_route, "Show current routing for this channel")

        tree.add_command(root)

    # ---- response plumbing --------------------------------------------

    @staticmethod
    def _cmd_channel(interaction: Any) -> str:
        cid = getattr(interaction, "channel_id", None)
        return str(cid) if cid else ""

    async def _respond(self, interaction: Any, text: str) -> None:
        """Reply to an interaction, chunked at the 2000-char limit.

        Interaction responses share the message-length cap; the first chunk
        goes through the (possibly deferred) response, the rest via followup.
        """
        chunks = chunk_message(format_markdown_for_discord(text))
        first, rest = chunks[0], chunks[1:]
        try:
            if interaction.response.is_done():
                await interaction.followup.send(first)
            else:
                await interaction.response.send_message(first)
            for chunk in rest:
                await interaction.followup.send(chunk)
        except Exception as exc:
            logger.error("Discord: interaction reply failed: %s", exc)

    async def _defer(self, interaction: Any) -> None:
        """Ack a slow command within Discord's 3s interaction deadline."""
        try:
            if not interaction.response.is_done():
                await interaction.response.defer()
        except Exception:
            logger.debug("Discord: defer failed", exc_info=True)

    def _channel_sender(self, interaction: Any):
        """Return ``async send(text)`` for multi-message shared flows.

        Prefers a durable channel post (an interaction's followup webhook
        expires after ~15 minutes — too tight for symbiote/audit/install
        flows that report back later); falls back to the followup when no
        channel is attached.
        """
        channel = getattr(interaction, "channel", None)

        async def _send(text: str) -> None:
            for chunk in chunk_message(format_markdown_for_discord(text)):
                if channel is not None:
                    try:
                        await channel.send(chunk)
                        continue
                    except Exception:
                        logger.debug(
                            "Discord: channel send failed; falling back to "
                            "followup", exc_info=True,
                        )
                try:
                    await interaction.followup.send(chunk)
                except Exception as exc:
                    logger.error("Discord: followup send failed: %s", exc)

        return _send

    # ---- core -----------------------------------------------------------

    async def _app_help(self, interaction: Any, args: str) -> None:
        # Hand-rolled (like Slack's) because every command lives under the
        # /prometheus <section> namespace — cmd_help() renders bare
        # Telegram-style names the Discord picker doesn't offer.
        lines = [
            "**Prometheus — Sovereign AI Agent**",
            "",
            "Core (`/prometheus core …`):",
            "  help · status · reset · clear · model · profile · context",
            "  benchmark · beacon · wiki · anatomy · doctor · tools · pairs",
            "  note · sentinel · health · events · voice · start",
            "  memory [show [user] | limits] · skills [list|show|pin|unpin|history]",
            "  curator [status|show|run [dry]] · notifications [off|quiet|verbose]",
            "",
            "Session (`/prometheus session …`):",
            "  steer <text> · queue <text> · unqueue · clearsteers",
            "",
            "Ops (`/prometheus ops …`):",
            "  approve <id> · deny <id> · pending",
            "  gepa [status|run|history] · symbiote <…> · audit [run|<category>]",
            "  press [list|search|install|installed|update] · escalations",
            "",
            "Provider (`/prometheus provider …`, this channel only):",
            "  claude · gpt · gemini · xai · grok",
            "  deepseek · kimi · glm · mimo · local · route",
            "",
            "Send a message (or DM the bot) to chat with the agent. "
            "Attachments are understood: images (vision), voice (Whisper), "
            "documents (text extraction).",
        ]
        await self._respond(interaction, "\n".join(lines))

    async def _app_start(self, interaction: Any, args: str) -> None:
        await self._respond(
            interaction,
            "Prometheus is online. Send me a message and I'll help you.",
        )

    async def _app_reset(self, interaction: Any, args: str) -> None:
        channel = self._cmd_channel(interaction)
        if channel:
            self.session_manager.clear(f"discord:{channel}")
        await self._respond(interaction, "Conversation context reset.")

    async def _app_clear(self, interaction: Any, args: str) -> None:
        channel = self._cmd_channel(interaction)
        if channel:
            self.session_manager.clear(f"discord:{channel}")
        await self._respond(interaction, "Conversation cleared.")

    async def _app_status(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_status

        await self._respond(interaction, cmd_status(
            self.model_name, self.model_provider,
            self._start_time, self.tool_registry,
            getattr(self, "cost_tracker", None),
        ))

    async def _app_model(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_model

        await self._respond(
            interaction, cmd_model(self.model_name, self.model_provider),
        )

    async def _app_profile(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_profile

        current = getattr(self, "_active_profile_name", "full")
        text = cmd_profile(arg=args, current=current)
        if args:
            try:
                from prometheus.config.profiles import ProfileStore
                profile = ProfileStore().get(args.strip())
                if profile is not None:
                    self._active_profile_name = profile.name
            except Exception:
                logger.debug("profile switch persistence skipped", exc_info=True)
        await self._respond(interaction, text)

    async def _app_context(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_context

        await self._respond(
            interaction, cmd_context(self.system_prompt, self.model_name),
        )

    async def _app_benchmark(self, interaction: Any, args: str) -> None:
        await self._defer(interaction)
        try:
            t0 = time.monotonic()
            result = await self.agent_loop.run_async(
                system_prompt="You are a helpful assistant. Be concise.",
                user_message="What is 2+2? Reply with just the number.",
                tools=[],
                # Diagnostic path — never inherit user overrides.
                session_id="system",
            )
            elapsed = time.monotonic() - t0
            response = (result.text or "").strip()
            passed = "4" in response
            lines = [
                f"Benchmark: {'PASS' if passed else 'FAIL'}",
                f"Latency: {elapsed:.2f}s",
                f"Response: {response[:100]}",
                f"Tokens: {result.usage.input_tokens} in / "
                f"{result.usage.output_tokens} out",
            ]
            await self._respond(interaction, "\n".join(lines))
        except Exception as exc:
            await self._respond(interaction, f"Benchmark: FAIL\nError: {exc}")

    async def _app_beacon(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_beacon

        await self._respond(interaction, cmd_beacon(self._prometheus_config))

    async def _app_wiki(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_wiki

        await self._respond(interaction, cmd_wiki())

    async def _app_anatomy(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_anatomy

        await self._defer(interaction)
        await self._respond(interaction, await cmd_anatomy())

    async def _app_doctor(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_doctor

        await self._defer(interaction)
        await self._respond(interaction, await cmd_doctor(self._prometheus_config))

    async def _app_tools(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_tools

        await self._respond(interaction, cmd_tools())

    async def _app_note(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_note
        from prometheus.tools.builtin.wiki_compile import _memory_store

        await self._respond(interaction, cmd_note(_memory_store, args))

    async def _app_sentinel(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_sentinel

        await self._respond(interaction, cmd_sentinel())

    async def _app_health(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_health

        verbose = False
        since_hours = 24.0
        for tok in args.split():
            t = tok.strip().lower()
            if t == "verbose":
                verbose = True
                continue
            try:
                since_hours = float(t)
            except ValueError:
                pass
        await self._respond(
            interaction, cmd_health(verbose=verbose, since_hours=since_hours),
        )

    async def _app_events(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_events

        await self._respond(interaction, cmd_events(arg=args))

    async def _app_pairs(self, interaction: Any, args: str) -> None:
        from prometheus.gateway.commands import cmd_pairs

        await self._respond(interaction, cmd_pairs())

    async def _app_voice(self, interaction: Any, args: str) -> None:
        """Platform-honest boundary: the voice-reply pipeline is Telegram-only.

        Discord voice-message INPUT works (audio attachments transcribe via
        Whisper); this is only about voice-note replies.
        """
        await self._respond(interaction, (
            "Voice replies are not supported on Discord yet — the TTS "
            "pipeline (piper → opus/ogg voice notes) is wired to Telegram's "
            "voice-message API only. Voice-message INPUT works: send an "
            "audio attachment and it is transcribed via Whisper. On Discord "
            "all replies are text."
        ))

    async def _app_memory(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        if not args:
            await self._respond(interaction, (
                "Memory commands:\n"
                f"  {CORE_PREFIX}memory show         — MEMORY.md content\n"
                f"  {CORE_PREFIX}memory show user    — USER.md content\n"
                f"  {CORE_PREFIX}memory limits       — char ceilings + usage"
            ))
            return
        parts = args.split(maxsplit=1)
        sub = parts[0].lower()
        tail = parts[1].strip().lower() if len(parts) > 1 else ""
        if sub == "show":
            text = _cmds.cmd_memory_show(
                target="user" if tail == "user" else "memory",
            )
        elif sub == "limits":
            text = _cmds.cmd_memory_limits()
        else:
            text = (
                f"Unknown subcommand: {sub}\n"
                f"Use: {CORE_PREFIX}memory [show [user] | limits]"
            )
        await self._respond(interaction, text)

    async def _app_skills(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        if not args:
            await self._respond(interaction, _cmds.cmd_skills())
            return
        parts = args.split(maxsplit=1)
        sub = parts[0].lower()
        name = parts[1].strip() if len(parts) > 1 else ""
        if sub == "list":
            text = _cmds.cmd_skills_auto_list()
        elif sub == "show":
            text = _cmds.cmd_skills_show(name)
        elif sub == "pin":
            text = _cmds.cmd_skills_pin(name)
        elif sub == "unpin":
            text = _cmds.cmd_skills_unpin(name)
        elif sub == "history":
            text = _cmds.cmd_skills_history(name)
        else:
            text = (
                f"Unknown subcommand: {sub}\n"
                f"Use: {CORE_PREFIX}skills [list | show <name> | pin <name> | "
                "unpin <name> | history <name>]"
            )
        await self._respond(interaction, text)

    async def _app_curator(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        if not args:
            await self._respond(interaction, (
                "Curator commands:\n"
                f"  {CORE_PREFIX}curator status      — last/next run, pinned skills\n"
                f"  {CORE_PREFIX}curator show        — most recent REPORT.md\n"
                f"  {CORE_PREFIX}curator run         — trigger an immediate pass\n"
                f"  {CORE_PREFIX}curator run dry     — dry-run (no file moves)"
            ))
            return
        parts = args.split()
        sub = parts[0].lower()
        if sub == "show":
            text = _cmds.cmd_curator_show()
        elif sub == "status":
            text = _cmds.cmd_curator_status()
        elif sub == "run":
            await self._defer(interaction)
            dry = len(parts) >= 2 and parts[1].lower().startswith("dry")
            text = await _cmds.cmd_curator_run(dry_run=dry)
        else:
            text = (
                f"Unknown subcommand: {sub}\n"
                f"Use: {CORE_PREFIX}curator [show | status | run [dry]]"
            )
        await self._respond(interaction, text)

    async def _app_notifications(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        await self._respond(interaction, _cmds.cmd_notifications(mode=args))

    # ---- session ---------------------------------------------------------

    async def _app_steer(self, interaction: Any, args: str) -> None:
        if not args:
            await self._respond(interaction, (
                f"{SESSION_PREFIX}steer <text> — inject mid-turn guidance.\n"
                "Arrives after the next tool call. Example:\n"
                f"{SESSION_PREFIX}steer focus on Ubuntu, skip the Mac instructions"
            ))
            return
        session = self._resolve_session_for_command(self._cmd_channel(interaction))
        if session is None:
            await self._respond(interaction, (
                "No active session yet. Send a message first, then "
                f"{SESSION_PREFIX}steer while the agent is running."
            ))
            return
        if not session.enqueue_steer(args):
            await self._respond(interaction, "Empty steer — nothing queued.")
            return
        preview = args if len(args) <= 80 else args[:77] + "..."
        await self._respond(interaction, (
            f"📍 Steered: {preview}\n"
            f"   Arrives after the next tool call. "
            f"Pending: {len(session.queued_steers)}."
        ))

    async def _app_queue(self, interaction: Any, args: str) -> None:
        if not args:
            await self._respond(interaction, (
                f"{SESSION_PREFIX}queue <text> — line up a follow-up turn.\n"
                "Runs after the current task ends."
            ))
            return
        channel = self._cmd_channel(interaction)
        if not channel:
            await self._respond(interaction, "Channel id missing from interaction.")
            return
        # queue creates the session if missing so a queued prompt on a quiet
        # channel fires when the user kicks off their first message.
        session = self.session_manager.get_or_create(f"discord:{channel}")
        if not session.enqueue_prompt(args):
            await self._respond(interaction, "Empty prompt — nothing queued.")
            return
        preview = args if len(args) <= 80 else args[:77] + "..."
        await self._respond(interaction, (
            f"📥 Queued: {preview}\n"
            f"   Position: {len(session.queued_prompts)}. "
            "Fires when current turn ends."
        ))

    async def _app_unqueue(self, interaction: Any, args: str) -> None:
        session = self._resolve_session_for_command(self._cmd_channel(interaction))
        if session is None or not session.queued_prompts:
            await self._respond(interaction, "No queued prompts to drop.")
            return
        dropped = session.queued_prompts.pop()
        preview = dropped if len(dropped) <= 80 else dropped[:77] + "..."
        await self._respond(interaction, (
            f"🗑️ Unqueued: {preview}\n"
            f"   Remaining: {len(session.queued_prompts)}."
        ))

    async def _app_clearsteers(self, interaction: Any, args: str) -> None:
        session = self._resolve_session_for_command(self._cmd_channel(interaction))
        if session is None:
            await self._respond(interaction, "No active session.")
            return
        n = session.clear_steers()
        await self._respond(
            interaction,
            f"🧹 Cleared {n} pending steer{'s' if n != 1 else ''}.",
        )

    # ---- ops ---------------------------------------------------------------

    async def _app_approve(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        request_id = args.split()[0] if args else ""
        text = await _cmds.cmd_approve(
            getattr(self, "_approval_queue", None), request_id,
            prefix=OPS_PREFIX,
        )
        await self._respond(interaction, text)

    async def _app_deny(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        request_id = args.split()[0] if args else ""
        text = await _cmds.cmd_deny(
            getattr(self, "_approval_queue", None), request_id,
            prefix=OPS_PREFIX,
        )
        await self._respond(interaction, text)

    async def _app_pending(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        await self._respond(
            interaction, _cmds.cmd_pending(getattr(self, "_approval_queue", None)),
        )

    async def _app_gepa(self, interaction: Any, args: str) -> None:
        """GEPA skill evolution: status | run | history.

        The approval prompt for ``run`` is delivered by the ApprovalQueue's
        transport (currently Telegram); progress/result messages land back
        in this Discord channel via the injected sender.
        """
        from prometheus.gateway import commands as _cmds

        await self._defer(interaction)
        sub = args.split()[0] if args else "status"
        text = await _cmds.cmd_gepa(
            getattr(self, "_gepa_engine", None),
            getattr(self, "_approval_queue", None),
            sub,
            chat_id=None,
            send=self._channel_sender(interaction),
            prefix=OPS_PREFIX,
        )
        if text:
            await self._respond(interaction, text)

    async def _app_symbiote(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        await self._defer(interaction)
        await _cmds.cmd_symbiote(
            self._channel_sender(interaction),
            args,
            approval_queue=getattr(self, "_approval_queue", None),
            morph_engine=getattr(self, "_morph_engine", None),
            backup_vault=getattr(self, "_backup_vault", None),
            chat_id=None,
            prefix=OPS_PREFIX,
        )

    async def _app_audit(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        await self._defer(interaction)
        await _cmds.cmd_audit(
            self._channel_sender(interaction), args, prefix=OPS_PREFIX,
        )

    async def _app_press(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        await self._defer(interaction)
        await _cmds.cmd_press(
            self._channel_sender(interaction),
            getattr(self, "_printing_press", None),
            args,
            approval_queue=getattr(self, "_approval_queue", None),
            chat_id=None,
            prefix=OPS_PREFIX,
        )

    async def _app_escalations(self, interaction: Any, args: str) -> None:
        from prometheus.gateway import commands as _cmds

        await self._respond(
            interaction,
            _cmds.cmd_escalations(getattr(self, "escalation_engine", None)),
        )

    # ---- provider overrides ------------------------------------------

    async def _apply_override(
        self, interaction: Any, args: str, preset_name: str,
    ) -> None:
        """Shared logic for the claude/gpt/gemini/xai/grok subcommands."""
        from prometheus.gateway import commands as _cmds

        channel = self._cmd_channel(interaction)
        if not channel:
            await self._respond(interaction, "Channel id missing from interaction.")
            return
        text, applied = _cmds.cmd_provider_override(
            self.agent_loop,
            self._prometheus_config,
            f"discord:{channel}",
            preset_name,
            prefix=PROVIDER_PREFIX,
        )
        # Platform-honest: Telegram dispatches an inline trailing message
        # ("/claude what is 2+2?") through the new provider; a Discord app
        # command's interaction has no message-dispatch context wired yet.
        if applied and args:
            text += (
                "\n\nNote: inline message dispatch isn't supported on "
                "Discord yet — send your question as a normal message."
            )
        await self._respond(interaction, text)

    async def _app_claude(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="claude")

    async def _app_gpt(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="gpt")

    async def _app_gemini(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="gemini")

    async def _app_xai(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="xai")

    async def _app_grok(self, interaction: Any, args: str) -> None:
        """Alias for /prometheus provider xai."""
        await self._apply_override(interaction, args, preset_name="xai")

    async def _app_deepseek(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="deepseek")

    async def _app_kimi(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="kimi")

    async def _app_glm(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="glm")

    async def _app_mimo(self, interaction: Any, args: str) -> None:
        await self._apply_override(interaction, args, preset_name="mimo")

    async def _app_local(self, interaction: Any, args: str) -> None:
        """Clear the per-channel provider override, back to primary."""
        from prometheus.gateway import commands as _cmds

        channel = self._cmd_channel(interaction)
        if not channel:
            await self._respond(interaction, "Channel id missing from interaction.")
            return
        text = _cmds.cmd_local_override(
            self.agent_loop, f"discord:{channel}",
            self.model_name, self.model_provider,
        )
        if args:
            text += (
                "\n\nNote: inline message dispatch isn't supported on "
                "Discord yet — send your question as a normal message."
            )
        await self._respond(interaction, text)

    async def _app_route(self, interaction: Any, args: str) -> None:
        """Show current routing (primary vs override) for this channel."""
        from prometheus.gateway import commands as _cmds

        channel = self._cmd_channel(interaction)
        text = _cmds.cmd_route(
            self.agent_loop, f"discord:{channel}",
            self.model_name, self.model_provider,
            prefix=PROVIDER_PREFIX,
        )
        await self._respond(interaction, text)
