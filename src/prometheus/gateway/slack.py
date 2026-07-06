"""Slack platform adapter — Socket Mode bot using slack-bolt.

Receives messages via WebSocket (no public URL needed), dispatches
to AgentLoop, sends responses back. Mirrors the TelegramAdapter pattern.

Key patterns adapted from NousResearch/hermes-agent Slack adapter:
- Message dedup (Socket Mode can redeliver events on reconnect)
- Markdown -> mrkdwn conversion (Slack uses its own format)
- Emoji reactions for processing feedback (eyes -> white_check_mark)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
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

# Slack message length limit (text field)
MAX_MESSAGE_LENGTH = 3900

# Sprint Polish: default threshold for moving long replies into a thread.
# Override via gateway.slack.long_reply_threshold in prometheus.yaml.
DEFAULT_LONG_REPLY_THRESHOLD = 800

# SPRINT G1: Slack workspace slash commands are global, so ours are
# namespaced. Shared command text that references sibling commands is built
# with this prefix so Slack users see commands they can actually type.
SLACK_COMMAND_PREFIX = "/prometheus-"


def chunk_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks respecting Slack's limit.

    Tries to split at paragraph boundaries, then newlines, then spaces.
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


def strip_bot_mention(text: str) -> str:
    """Remove <@BOTID> from mention text."""
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def format_markdown_to_mrkdwn(content: str) -> str:
    """Convert standard markdown to Slack mrkdwn format.

    Adapted from NousResearch/hermes-agent. Protected regions (code blocks,
    inline code) are extracted first so their contents are never modified.
    """
    if not content:
        return content

    placeholders: dict[str, str] = {}
    counter = [0]

    def _ph(value: str) -> str:
        key = f"\x00SL{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = value
        return key

    text = content

    # 1) Protect fenced code blocks (``` ... ```)
    text = re.sub(
        r"(```(?:[^\n]*\n)?[\s\S]*?```)",
        lambda m: _ph(m.group(0)),
        text,
    )

    # 2) Protect inline code (`...`)
    text = re.sub(r"(`[^`]+`)", lambda m: _ph(m.group(0)), text)

    # 3) Convert markdown links [text](url) -> <url|text>
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: _ph(f"<{m.group(2)}|{m.group(1)}>"),
        text,
    )

    # 4) Convert headers (## Title) -> *Title* (bold)
    def _convert_header(m: re.Match) -> str:
        inner = m.group(1).strip()
        inner = re.sub(r"\*\*(.+?)\*\*", r"\1", inner)
        return _ph(f"*{inner}*")

    text = re.sub(
        r"^#{1,6}\s+(.+)$", _convert_header, text, flags=re.MULTILINE
    )

    # 5) Convert bold: **text** -> *text* (Slack bold)
    text = re.sub(
        r"\*\*(.+?)\*\*",
        lambda m: _ph(f"*{m.group(1)}*"),
        text,
    )

    # 6) Convert strikethrough: ~~text~~ -> ~text~
    text = re.sub(
        r"~~(.+?)~~",
        lambda m: _ph(f"~{m.group(1)}~"),
        text,
    )

    # 7) Restore placeholders
    for key in reversed(list(placeholders.keys())):
        text = text.replace(key, placeholders[key])

    return text


class SlackAdapter(BasePlatformAdapter):
    """Slack bot adapter using Socket Mode — receives messages, routes to AgentLoop."""

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
        self._app: Any = None
        self._handler: Any = None
        self._connect_task: Any = None
        self._start_time: float = 0.0
        self._prometheus_config: dict[str, Any] = prometheus_config or {}

        if session_manager is None:
            from prometheus.engine.session import SessionManager as _SM
            session_manager = _SM()
        self.session_manager: SessionManager = session_manager
        # Dedup cache: event_ts -> timestamp. Prevents duplicate bot
        # responses when Socket Mode reconnects and redelivers events.
        # Pattern from NousResearch/hermes-agent.
        self._seen_messages: dict[str, float] = {}
        self._SEEN_TTL = 300  # 5 minutes
        self._SEEN_MAX = 2000  # prune threshold

        # Sprint Polish: SignalBus wired by daemon.py via the property setter.
        # Until then it's None and the subscribe path is a no-op (mirror of
        # Telegram's pattern).
        self._signal_bus: object | None = None

    # ------------------------------------------------------------------
    # Sprint Polish: SignalBus subscription + last-channel tracking
    # ------------------------------------------------------------------

    @property
    def signal_bus(self) -> object | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: object | None) -> None:
        """Subscribe to skill/memory/curator events on bus assignment.

        Called from daemon.py after the SignalBus is constructed. Mirrors
        the Telegram subscription so SKILL_CREATED / SKILL_REFINED /
        MEMORY_UPDATED / CURATOR_REPORT notifications land in Slack too
        when Slack is enabled — not just whichever gateway was the
        originating one.
        """
        self._signal_bus = bus
        if bus is None:
            return
        try:
            bus.subscribe("skill_created", self._on_signal_skill_created)
            bus.subscribe("skill_refined", self._on_signal_skill_refined)
            bus.subscribe("memory_updated", self._on_signal_memory_updated)
            bus.subscribe("curator_report", self._on_signal_curator_report)
            logger.info("Slack: subscribed to skill/memory/curator signals")
        except Exception:
            logger.warning(
                "Slack: failed to subscribe to SignalBus", exc_info=True
            )

    def _notification_mode(self) -> str:
        """Return 'quiet' | 'verbose' | 'off'.

        Runtime override (set via /prometheus-notifications) wins. Falls
        back to gateway.slack.skill_event_notifications, then
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
        slack_cfg = gw.get("slack", {}) if isinstance(gw, dict) else {}
        if isinstance(slack_cfg, dict):
            mode = slack_cfg.get("skill_event_notifications")
            if mode:
                return str(mode).lower()
        if isinstance(gw, dict):
            return str(gw.get("skill_event_notifications", "quiet")).lower()
        return "quiet"

    def _long_reply_threshold(self) -> int:
        """Return char threshold above which replies move into a thread."""
        gw = self._prometheus_config.get("gateway", {}) or {}
        slack_cfg = gw.get("slack", {}) if isinstance(gw, dict) else {}
        if isinstance(slack_cfg, dict):
            try:
                return int(slack_cfg.get("long_reply_threshold", DEFAULT_LONG_REPLY_THRESHOLD))
            except (TypeError, ValueError):
                pass
        return DEFAULT_LONG_REPLY_THRESHOLD

    def _last_channel_path(self) -> str:
        from prometheus.config.paths import get_config_dir
        return str(get_config_dir() / "last_slack_channel")

    def _save_channel(self, channel: str) -> None:
        """Persist the last active Slack channel for signal notifications."""
        if not channel:
            return
        try:
            with open(self._last_channel_path(), "w") as f:
                f.write(channel)
        except Exception:
            pass

    def _load_channel(self) -> str | None:
        try:
            with open(self._last_channel_path()) as f:
                ch = f.read().strip()
                return ch or None
        except Exception:
            return None

    async def _send_notification(self, text: str) -> None:
        """Post *text* to the last active Slack channel (best-effort)."""
        channel = self._load_channel()
        if not channel:
            # Fall back to first allowed_channel if configured.
            if self.config.allowed_channels:
                channel = self.config.allowed_channels[0]
        if not channel or not self._app:
            return
        try:
            await self._app.client.chat_postMessage(channel=channel, text=text)
        except Exception:
            logger.debug("Slack: notification send failed", exc_info=True)

    async def _on_signal_skill_created(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        name = payload.get("skill_name", "(unnamed)")
        if mode == "verbose":
            trigger = payload.get("trigger_task", "")
            summary = payload.get("summary", "")
            text = f":mortar_board: New skill: *{name}*"
            if summary:
                text += f"\n   {summary}"
            if trigger:
                text += f"\n   (built while: {trigger[:120]})"
        else:
            text = f":mortar_board: New skill: *{name}*"
        await self._send_notification(text)

    async def _on_signal_skill_refined(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        name = payload.get("skill_name", "(unnamed)")
        if mode == "verbose":
            summary = payload.get("summary", "")
            text = f":books: Updated skill: *{name}*"
            if summary:
                text += f"\n   {summary}"
        else:
            text = f":books: Updated skill: *{name}*"
        await self._send_notification(text)

    async def _on_signal_memory_updated(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        target = payload.get("target", "memory")
        operation = payload.get("operation", "updated")
        if mode == "verbose":
            preview = payload.get("entry_preview", "")
            text = f":brain: {target} {operation}"
            if preview:
                text += f"\n   {preview}"
        else:
            text = f":brain: {target} {operation}"
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
                f":clipboard: Curator: {reviewed} skills reviewed, "
                f"{consolidations} consolidation suggestion(s), "
                f"{prunings} archived\n"
                "   `/prometheus-curator show` for the full report"
            )
        else:
            text = f":clipboard: Curator: {reviewed} reviewed, {prunings} archived"
        await self._send_notification(text)

    def _resolve_session_for_command(self, channel_id: str) -> "ChatSession | None":
        """Return the live ChatSession for *channel_id*, or None.

        Mirrors Telegram's helper: used by /steer, /unqueue, /clearsteers
        to find the session-state object the agent loop is consuming.
        Does NOT create a session — if there's no active session yet,
        the command is a no-op.
        """
        if not channel_id:
            return None
        return self.session_manager._sessions.get(f"slack:{channel_id}")

    async def start(self) -> None:
        """Build the Slack app with Socket Mode and start listening."""
        try:
            from slack_bolt.async_app import AsyncApp
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
        except ImportError:
            raise ImportError(
                "slack-bolt is required for Slack support. "
                "Install it with: pip install slack-bolt slack-sdk"
            )

        if not self.config.token:
            raise ValueError("Slack bot token (xoxb-...) is required")
        if not self.config.app_token:
            raise ValueError("Slack app token (xapp-...) is required for Socket Mode")

        self._app = AsyncApp(token=self.config.token)

        # Register event handlers
        self._app.event("message")(self._handle_message)
        self._app.event("app_mention")(self._handle_mention)

        # Register slash command handlers
        self._app.command("/prometheus-status")(self._slash_status)
        self._app.command("/prometheus-help")(self._slash_help)
        self._app.command("/prometheus-reset")(self._slash_reset)
        self._app.command("/prometheus-model")(self._slash_model)
        self._app.command("/prometheus-wiki")(self._slash_wiki)
        self._app.command("/prometheus-sentinel")(self._slash_sentinel)
        self._app.command("/prometheus-benchmark")(self._slash_benchmark)
        self._app.command("/prometheus-context")(self._slash_context)
        self._app.command("/prometheus-skills")(self._slash_skills)
        # Sprint S4 A3: silent-failure telemetry surface.
        self._app.command("/prometheus-health")(self._slash_health)
        # SignalBus Persistence sprint: persisted signal-bus events.
        self._app.command("/prometheus-events")(self._slash_events)
        # Sprint Polish (Telegram parity): Sprint 1 / Sprint 2 surface
        self._app.command("/prometheus-memory")(self._slash_memory)
        self._app.command("/prometheus-curator")(self._slash_curator)
        self._app.command("/prometheus-notifications")(self._slash_notifications)
        self._app.command("/prometheus-steer")(self._slash_steer)
        self._app.command("/prometheus-queue")(self._slash_queue)
        self._app.command("/prometheus-unqueue")(self._slash_unqueue)
        self._app.command("/prometheus-clearsteers")(self._slash_clearsteers)
        self._app.command("/prometheus-profile")(self._slash_profile)
        self._app.command("/prometheus-anatomy")(self._slash_anatomy)
        self._app.command("/prometheus-doctor")(self._slash_doctor)
        self._app.command("/prometheus-beacon")(self._slash_beacon)
        self._app.command("/prometheus-tools")(self._slash_tools)
        # SPRINT G1 — gateway parity: fill the Telegram-only families.
        self._app.command("/prometheus-note")(self._slash_note)
        self._app.command("/prometheus-pairs")(self._slash_pairs)
        self._app.command("/prometheus-approve")(self._slash_approve)
        self._app.command("/prometheus-deny")(self._slash_deny)
        self._app.command("/prometheus-pending")(self._slash_pending)
        self._app.command("/prometheus-escalations")(self._slash_escalations)
        self._app.command("/prometheus-gepa")(self._slash_gepa)
        self._app.command("/prometheus-symbiote")(self._slash_symbiote)
        self._app.command("/prometheus-audit")(self._slash_audit)
        self._app.command("/prometheus-press")(self._slash_press)
        self._app.command("/prometheus-voice")(self._slash_voice)
        # Provider overrides (per-channel, sticky until /prometheus-local)
        self._app.command("/prometheus-claude")(self._slash_claude)
        self._app.command("/prometheus-gpt")(self._slash_gpt)
        self._app.command("/prometheus-gemini")(self._slash_gemini)
        self._app.command("/prometheus-xai")(self._slash_xai)
        self._app.command("/prometheus-grok")(self._slash_grok)
        # CLOUD EXPANSION (2026-07): four more provider override commands
        self._app.command("/prometheus-deepseek")(self._slash_deepseek)
        self._app.command("/prometheus-kimi")(self._slash_kimi)
        self._app.command("/prometheus-glm")(self._slash_glm)
        self._app.command("/prometheus-mimo")(self._slash_mimo)
        self._app.command("/prometheus-local")(self._slash_local)
        self._app.command("/prometheus-route")(self._slash_route)

        # Start Socket Mode connection.
        #
        # SPRINT G3 fix — this used to be `await self._handler.start_async()`,
        # which per slack-bolt "starts infinite sleep to prevent the
        # termination of this process": the daemon's startup sequence hung
        # HERE forever, even with valid tokens (nothing after the Slack
        # block — Discord, heartbeat, web server — ever ran). And plain
        # connect_async() retries forever on invalid_auth without raising,
        # which would hang startup on a bad token instead. So: validate both
        # tokens LOUDLY first (SlackApiError propagates to the daemon's
        # try/except → logged, non-fatal), then hand the connect to a
        # background task — the same non-blocking start the Discord adapter
        # uses.
        self._handler = AsyncSocketModeHandler(self._app, self.config.app_token)
        await self._app.client.auth_test()  # bot token (xoxb-...) — raises on bad auth
        # App-level token (xapp-...): apps.connections.open raises on bad
        # auth; cache the issued WSS URL so connect_async doesn't re-fetch.
        self._handler.client.wss_uri = await self._handler.client.issue_new_wss_url()
        self._connect_task = asyncio.create_task(
            self._handler.connect_async(), name="slack_socket_mode",
        )
        self._running = True
        self._start_time = time.monotonic()

        logger.info("Slack adapter started (Socket Mode)")

    async def stop(self) -> None:
        """Graceful shutdown of the Slack bot."""
        if self._handler and self._running:
            self._running = False
            if self._connect_task is not None and not self._connect_task.done():
                self._connect_task.cancel()
            await self._handler.close_async()
            logger.info("Slack adapter stopped")

    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> SendResult:
        """Send a message to a Slack channel.

        Note: chat_id/reply_to are typed as int for ABC compatibility,
        but Slack uses string channel IDs and timestamp message IDs.
        Use send_to_channel() for native Slack types.
        """
        return await self.send_to_channel(
            channel=str(chat_id),
            text=text,
            thread_ts=str(reply_to) if reply_to else None,
        )

    async def send_to_channel(
        self,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> SendResult:
        """Send a message to a Slack channel (native string types)."""
        if not self._app:
            return SendResult(success=False, error="Bot not initialized")

        chunks = chunk_message(text)
        last_ts: str | None = None

        for chunk in chunks:
            try:
                result = await self._app.client.chat_postMessage(
                    channel=channel,
                    text=chunk,
                    thread_ts=thread_ts,
                )
                last_ts = result.get("ts")
            except Exception as exc:
                logger.error("Failed to send message to %s: %s", channel, exc)
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=int(float(last_ts or "0")))

    async def on_message(self, event: MessageEvent) -> None:
        """Handle an incoming message — dispatch to agent and reply."""
        if not self.config.channel_allowed(str(event.chat_id)):
            logger.warning(
                "Ignoring message from non-whitelisted channel %s", event.chat_id
            )
            return
        # Dispatch handled internally via _handle_message / _handle_mention

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _dedup_check(self, event_ts: str) -> bool:
        """Return True if this event was already seen (should be skipped)."""
        if not event_ts:
            return False
        now = time.time()
        if event_ts in self._seen_messages:
            return True
        self._seen_messages[event_ts] = now
        # Prune old entries
        if len(self._seen_messages) > self._SEEN_MAX:
            cutoff = now - self._SEEN_TTL
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        return False

    async def _add_reaction(self, channel: str, ts: str, emoji: str) -> None:
        """Add an emoji reaction to a message (best-effort)."""
        if not self._app:
            return
        try:
            await self._app.client.reactions_add(
                channel=channel, timestamp=ts, name=emoji
            )
        except Exception:
            pass  # best-effort — may lack scope or already reacted

    async def _remove_reaction(self, channel: str, ts: str, emoji: str) -> None:
        """Remove an emoji reaction from a message (best-effort)."""
        if not self._app:
            return
        try:
            await self._app.client.reactions_remove(
                channel=channel, timestamp=ts, name=emoji
            )
        except Exception:
            pass

    async def _handle_message(self, event: dict[str, Any], say: Any) -> None:
        """Handle direct messages to the bot."""
        # Ignore bot's own messages and message_changed subtypes
        if event.get("bot_id") or event.get("subtype"):
            return

        # Dedup: Socket Mode can redeliver events after reconnects
        if self._dedup_check(event.get("ts", "")):
            return

        channel = event.get("channel", "")

        # Enforce channel whitelist
        if self.config.allowed_channels and channel not in self.config.allowed_channels:
            return

        text = event.get("text", "")
        if not text:
            return

        await self._dispatch_to_agent(
            channel=channel,
            user=event.get("user", "unknown"),
            text=text,
            ts=event.get("ts", ""),
            thread_ts=event.get("thread_ts"),
            say=say,
        )

    async def _handle_mention(self, event: dict[str, Any], say: Any) -> None:
        """Handle @mentions in channels."""
        if event.get("bot_id") or event.get("subtype"):
            return

        if self._dedup_check(event.get("ts", "")):
            return

        channel = event.get("channel", "")
        if self.config.allowed_channels and channel not in self.config.allowed_channels:
            return

        text = strip_bot_mention(event.get("text", ""))
        if not text:
            return

        await self._dispatch_to_agent(
            channel=channel,
            user=event.get("user", "unknown"),
            text=text,
            ts=event.get("ts", ""),
            thread_ts=event.get("thread_ts"),
            say=say,
        )

    async def _dispatch_to_agent(
        self,
        channel: str,
        user: str,
        text: str,
        ts: str,
        thread_ts: str | None,
        say: Any,
    ) -> None:
        """Route a message through AgentLoop and send the response."""
        # Add eyes reaction to acknowledge receipt
        await self._add_reaction(channel, ts, "eyes")

        # Sprint Polish: remember the last active channel so SignalBus
        # notifications land somewhere meaningful.
        self._save_channel(channel)

        session_id = f"slack:{channel}"
        session = self.session_manager.get_or_create(session_id)
        session.add_user_message(text)
        pre_len = len(session.get_messages())

        try:
            result = await self.agent_loop.run_async(
                system_prompt=self.system_prompt,
                messages=session.get_messages(),
                tools=self.tool_registry.list_schemas(),
                # Phase 3.5: per-channel override namespace.
                session_id=session_id,
            )
            session.add_result_messages(result.messages, pre_len)
            session.trim(self.session_manager.MAX_SESSION_MESSAGES)
            response_text = result.text or "(no response)"
        except Exception as exc:
            logger.error("Agent error for channel %s: %s", channel, exc)
            session.rollback_last()
            response_text = f"Error: {exc}"

        # Convert markdown to Slack mrkdwn format
        response_text = format_markdown_to_mrkdwn(response_text)

        # Threading policy:
        #   - If the user wrote in a thread, reply in that thread.
        #   - If the reply is long (> threshold), open a new thread off the
        #     user's message rather than flooding the channel.
        #   - Otherwise reply in the channel.
        reply_thread: str | None
        if thread_ts:
            reply_thread = thread_ts
        elif len(response_text) > self._long_reply_threshold():
            reply_thread = ts
        else:
            reply_thread = None

        chunks = chunk_message(response_text)
        for chunk in chunks:
            try:
                await say(text=chunk, thread_ts=reply_thread)
            except Exception as exc:
                logger.error("Failed to send response to %s: %s", channel, exc)

        # Replace eyes with checkmark when done
        await self._remove_reaction(channel, ts, "eyes")
        await self._add_reaction(channel, ts, "white_check_mark")

    # ------------------------------------------------------------------
    # Slash command handlers (reuse shared command logic)
    # ------------------------------------------------------------------

    async def _slash_status(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_status

        text = cmd_status(
            self.model_name, self.model_provider,
            self._start_time, self.tool_registry,
            getattr(self, "cost_tracker", None),
        )
        await respond(text=text)

    async def _slash_help(self, ack: Any, respond: Any) -> None:
        await ack()
        # Slack help is hand-rolled rather than ported from cmd_help() because
        # every command is prefixed with ``/prometheus-`` (Slack workspace
        # commands are workspace-global, so we namespace ours).
        lines = [
            "Prometheus — Sovereign AI Agent",
            "",
            "Core:",
            "  /prometheus-status         — model, uptime, tools, memory, SENTINEL",
            "  /prometheus-help           — this message",
            "  /prometheus-reset          — clear conversation context for this channel",
            "  /prometheus-model          — current model and provider",
            "  /prometheus-profile [name] — show / switch agent profile",
            "  /prometheus-context        — context window usage",
            "  /prometheus-benchmark      — quick smoke test",
            "  /prometheus-beacon         — web bridge / dashboard URL",
            "  /prometheus-wiki           — wiki stats + recent entries",
            "  /prometheus-anatomy        — host, GPU, VRAM, Tailscale",
            "  /prometheus-doctor         — diagnostic health check",
            "  /prometheus-tools          — tool-call stats (24h)",
            "  /prometheus-pairs          — training-pair flywheel stats",
            "  /prometheus-note           — save a manual fact: [@entity] <text>",
            "",
            "Provider overrides (this channel only, sticky until -local):",
            "  /prometheus-claude         — Anthropic Claude",
            "  /prometheus-gpt            — OpenAI GPT",
            "  /prometheus-gemini         — Google Gemini",
            "  /prometheus-xai            — xAI Grok  (alias: -grok)",
            "  /prometheus-deepseek       — DeepSeek",
            "  /prometheus-kimi           — Kimi (Moonshot)",
            "  /prometheus-glm            — GLM (Z.ai)",
            "  /prometheus-mimo           — MiMo (Xiaomi)",
            "  /prometheus-local          — back to primary",
            "  /prometheus-route          — current routing (primary vs override)",
            "",
            "Approvals & autonomy:",
            "  /prometheus-approve <id>   — approve a pending tool request",
            "  /prometheus-deny <id>      — deny a pending tool request",
            "  /prometheus-pending        — list pending approval requests",
            "  /prometheus-gepa           — skill evolution: status | run | history",
            "  /prometheus-symbiote       — GitHub graft pipeline (status | history | …)",
            "  /prometheus-audit          — web capability audit: run | <category>",
            "  /prometheus-press          — CLI library: list | search | install | …",
            "  /prometheus-escalations    — teacher-escalation counters",
            "",
            "Memory & skills (Sprint S1):",
            "  /prometheus-memory show [user]  — MEMORY.md / USER.md content",
            "  /prometheus-memory limits       — char ceilings + usage",
            "  /prometheus-skills              — registry list",
            "  /prometheus-skills list         — auto-skills with state",
            "  /prometheus-skills show <name>  — display SKILL.md",
            "  /prometheus-skills pin <name>   — protect from Curator prune",
            "  /prometheus-skills unpin <name>",
            "  /prometheus-skills history <name>",
            "  /prometheus-curator status      — last/next run + pinned",
            "  /prometheus-curator show        — most recent REPORT.md",
            "  /prometheus-curator run [dry]   — trigger an immediate pass",
            "  /prometheus-notifications [off|quiet|verbose]",
            "  /prometheus-events [recent|skills|memory|curator|show <id>]",
            "",
            "Durability & steering (Sprint 2):",
            "  /prometheus-steer <text>        — mid-turn guidance",
            "  /prometheus-queue <text>        — queue a follow-up turn",
            "  /prometheus-unqueue             — drop the last queued prompt",
            "  /prometheus-clearsteers         — drop all pending steers",
            "",
            "Observability:",
            "  /prometheus-health [hours] [verbose] — silent-failure telemetry",
            "  /prometheus-sentinel            — SENTINEL subsystem state",
            "",
            "Send a message or @mention the bot to chat with the agent.",
        ]
        await respond(text="\n".join(lines))

    async def _slash_reset(self, ack: Any, respond: Any, body: dict | None = None) -> None:
        await ack()
        if body:
            channel = body.get("channel_id", "")
            if channel:
                self.session_manager.clear(f"slack:{channel}")
        await respond(text="Conversation context reset.")

    async def _slash_model(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_model

        await respond(text=cmd_model(self.model_name, self.model_provider))

    async def _slash_wiki(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_wiki

        await respond(text=cmd_wiki())

    async def _slash_sentinel(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_sentinel

        await respond(text=cmd_sentinel())

    async def _slash_benchmark(self, ack: Any, respond: Any) -> None:
        await ack()
        await respond(text="Running benchmark...")

        try:
            t0 = time.monotonic()
            result = await self.agent_loop.run_async(
                system_prompt="You are a helpful assistant. Be concise.",
                user_message="What is 2+2? Reply with just the number.",
                tools=[],
                # Phase 3.5: diagnostic path — never inherit user overrides.
                session_id="system",
            )
            elapsed = time.monotonic() - t0

            response = (result.text or "").strip()
            passed = "4" in response

            lines = [
                f"Benchmark: {'PASS' if passed else 'FAIL'}",
                f"Latency: {elapsed:.2f}s",
                f"Response: {response[:100]}",
                f"Tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out",
            ]
            await respond(text="\n".join(lines))
        except Exception as exc:
            await respond(text=f"Benchmark: FAIL\nError: {exc}")

    async def _slash_context(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_context

        await respond(text=cmd_context(self.system_prompt, self.model_name))

    async def _slash_skills(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Mirror Telegram /skills with subcommand dispatch (Sprint S1)."""
        await ack()
        from prometheus.gateway import commands as _cmds

        try:
            text_arg = (command.get("text") or "").strip()
        except AttributeError:
            text_arg = ""

        if not text_arg:
            await respond(text=_cmds.cmd_skills())
            return

        parts = text_arg.split(maxsplit=1)
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
                "Use: /prometheus-skills [list | show <name> | pin <name> | "
                "unpin <name> | history <name>]"
            )
        await respond(text=text)

    async def _slash_health(self, ack: Any, command: Any, respond: Any) -> None:
        """Mirror of the Telegram /health command (Sprint 4 A3).

        Slack passes the command text via ``command["text"]``. Supported
        args mirror Telegram: ``verbose``, a numeric hours override, or both.
        """
        await ack()
        from prometheus.gateway.commands import cmd_health

        text_arg = ""
        try:
            text_arg = (command.get("text") or "").strip()
        except AttributeError:
            text_arg = ""
        verbose = False
        since_hours = 24.0
        for tok in text_arg.split():
            t = tok.strip().lower()
            if t == "verbose":
                verbose = True
                continue
            try:
                since_hours = float(t)
            except ValueError:
                pass
        await respond(text=cmd_health(verbose=verbose, since_hours=since_hours))

    async def _slash_events(self, ack: Any, command: Any, respond: Any) -> None:
        """Mirror of the Telegram /events command (SignalBus Persistence sprint).

        Slack passes the command text via ``command["text"]``. Passed verbatim
        to ``cmd_events`` which parses its own subcommands
        (``recent`` | ``skills`` | ``memory`` | ``curator`` | ``show <id>``).
        """
        await ack()
        from prometheus.gateway.commands import cmd_events

        try:
            text_arg = (command.get("text") or "").strip()
        except AttributeError:
            text_arg = ""
        await respond(text=cmd_events(arg=text_arg))

    # ------------------------------------------------------------------
    # Sprint Polish: Sprint 1 / Sprint 2 surface — Telegram parity
    # ------------------------------------------------------------------

    @staticmethod
    def _cmd_text(command: Any) -> str:
        """Pull the trailing ``text`` field from a Slack slash-command payload."""
        try:
            return (command.get("text") or "").strip()
        except AttributeError:
            return ""

    @staticmethod
    def _cmd_channel(command: Any) -> str:
        try:
            return command.get("channel_id") or ""
        except AttributeError:
            return ""

    async def _slash_memory(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Mirror /memory — show MEMORY.md / USER.md, or show limits."""
        await ack()
        from prometheus.gateway import commands as _cmds

        text_arg = self._cmd_text(command)
        if not text_arg:
            await respond(text=(
                "Memory commands:\n"
                "  /prometheus-memory show         — MEMORY.md content\n"
                "  /prometheus-memory show user    — USER.md content\n"
                "  /prometheus-memory limits       — char ceilings + usage"
            ))
            return

        parts = text_arg.split(maxsplit=1)
        sub = parts[0].lower()
        tail = parts[1].strip().lower() if len(parts) > 1 else ""

        if sub == "show":
            target = "user" if tail == "user" else "memory"
            text = _cmds.cmd_memory_show(target=target)
        elif sub == "limits":
            text = _cmds.cmd_memory_limits()
        else:
            text = (
                f"Unknown subcommand: {sub}\n"
                "Use: /prometheus-memory [show [user] | limits]"
            )
        await respond(text=text)

    async def _slash_curator(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Mirror /curator — show/status/run subcommand dispatcher."""
        await ack()
        from prometheus.gateway import commands as _cmds

        text_arg = self._cmd_text(command)
        if not text_arg:
            await respond(text=(
                "Curator commands:\n"
                "  /prometheus-curator status      — last/next run, pinned skills\n"
                "  /prometheus-curator show        — most recent REPORT.md\n"
                "  /prometheus-curator run         — trigger an immediate pass\n"
                "  /prometheus-curator run dry     — dry-run (no file moves)"
            ))
            return

        parts = text_arg.split()
        sub = parts[0].lower()
        if sub == "show":
            text = _cmds.cmd_curator_show()
        elif sub == "status":
            text = _cmds.cmd_curator_status()
        elif sub == "run":
            dry = len(parts) >= 2 and parts[1].lower().startswith("dry")
            text = await _cmds.cmd_curator_run(dry_run=dry)
        else:
            text = (
                f"Unknown subcommand: {sub}\n"
                "Use: /prometheus-curator [show | status | run [dry]]"
            )
        await respond(text=text)

    async def _slash_notifications(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Toggle skill/memory/curator notification verbosity (Sprint S1)."""
        await ack()
        from prometheus.gateway import commands as _cmds

        text_arg = self._cmd_text(command)
        await respond(text=_cmds.cmd_notifications(mode=text_arg))

    async def _slash_steer(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Mid-turn guidance — arrives after the current tool batch (Sprint 2)."""
        await ack()
        text_arg = self._cmd_text(command)
        if not text_arg:
            await respond(text=(
                "/prometheus-steer <text> — inject mid-turn guidance.\n"
                "Arrives after the next tool call. Example:\n"
                "/prometheus-steer focus on Ubuntu, skip the Mac instructions"
            ))
            return
        channel = self._cmd_channel(command)
        session = self._resolve_session_for_command(channel)
        if session is None:
            await respond(text=(
                "No active session yet. Send a message first, then "
                "/prometheus-steer while the agent is running."
            ))
            return
        if not session.enqueue_steer(text_arg):
            await respond(text="Empty steer — nothing queued.")
            return
        preview = text_arg if len(text_arg) <= 80 else text_arg[:77] + "..."
        await respond(text=(
            f":round_pushpin: Steered: {preview}\n"
            f"   Arrives after the next tool call. "
            f"Pending: {len(session.queued_steers)}."
        ))

    async def _slash_queue(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Queue a follow-up turn that fires after the current one ends."""
        await ack()
        text_arg = self._cmd_text(command)
        if not text_arg:
            await respond(text=(
                "/prometheus-queue <text> — line up a follow-up turn.\n"
                "Runs after the current task ends."
            ))
            return
        channel = self._cmd_channel(command)
        if not channel:
            await respond(text="Channel id missing from slash payload.")
            return
        # /queue creates the session if missing so a queued prompt on a
        # quiet channel fires when the user kicks off their first message.
        session = self.session_manager.get_or_create(f"slack:{channel}")
        if not session.enqueue_prompt(text_arg):
            await respond(text="Empty prompt — nothing queued.")
            return
        preview = text_arg if len(text_arg) <= 80 else text_arg[:77] + "..."
        position = len(session.queued_prompts)
        await respond(text=(
            f":inbox_tray: Queued: {preview}\n"
            f"   Position: {position}. Fires when current turn ends."
        ))

    async def _slash_unqueue(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Drop the most recently queued prompt."""
        await ack()
        channel = self._cmd_channel(command)
        session = self._resolve_session_for_command(channel)
        if session is None or not session.queued_prompts:
            await respond(text="No queued prompts to drop.")
            return
        dropped = session.queued_prompts.pop()
        preview = dropped if len(dropped) <= 80 else dropped[:77] + "..."
        remaining = len(session.queued_prompts)
        await respond(text=(
            f":wastebasket: Unqueued: {preview}\n"
            f"   Remaining: {remaining}."
        ))

    async def _slash_clearsteers(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Drop all pending steers without surfacing them to the agent."""
        await ack()
        channel = self._cmd_channel(command)
        session = self._resolve_session_for_command(channel)
        if session is None:
            await respond(text="No active session.")
            return
        n = session.clear_steers()
        await respond(text=(
            f":broom: Cleared {n} pending steer{'s' if n != 1 else ''}."
        ))

    async def _slash_profile(
        self, ack: Any, command: Any, respond: Any
    ) -> None:
        """Show or switch agent profile."""
        await ack()
        from prometheus.gateway.commands import cmd_profile

        text_arg = self._cmd_text(command)
        current = getattr(self, "_active_profile_name", "full")
        text = cmd_profile(arg=text_arg, current=current)

        if text_arg:
            try:
                from prometheus.config.profiles import ProfileStore
                store = ProfileStore()
                profile = store.get(text_arg.strip())
                if profile is not None:
                    self._active_profile_name = profile.name
            except Exception:
                logger.debug("profile switch persistence skipped", exc_info=True)

        await respond(text=text)

    async def _slash_anatomy(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_anatomy

        await respond(text=await cmd_anatomy())

    async def _slash_doctor(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_doctor

        await respond(text=await cmd_doctor(self._prometheus_config))

    async def _slash_beacon(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_beacon

        await respond(text=cmd_beacon(self._prometheus_config))

    async def _slash_tools(self, ack: Any, respond: Any) -> None:
        """Tool-call telemetry dashboard (24h)."""
        await ack()
        from prometheus.gateway.commands import cmd_tools

        await respond(text=cmd_tools())

    # ------------------------------------------------------------------
    # SPRINT G1 — gateway parity: Telegram-only families filled in.
    # Every handler below is a thin wrapper over the shared commands layer;
    # subsystems (approval queue, GEPA, symbiote engines, printing press,
    # escalation engine) arrive via the daemon's GatewaySubsystemRegistry.
    # ------------------------------------------------------------------

    def _channel_sender(self, command: Any, respond: Any):
        """Return ``async send(text)`` for multi-message shared flows.

        Prefers a durable ``chat_postMessage`` to the invoking channel (a
        slash command's ``respond`` URL expires after ~30 minutes and allows
        only 5 messages — too tight for symbiote/audit/install flows that
        report back later); falls back to ``respond`` when the channel post
        fails (e.g. bot not in channel).
        """
        channel = self._cmd_channel(command)

        async def _send(text: str) -> None:
            if channel and self._app:
                try:
                    await self._app.client.chat_postMessage(
                        channel=channel, text=text,
                    )
                    return
                except Exception:
                    logger.debug(
                        "Slack: channel send failed; falling back to respond",
                        exc_info=True,
                    )
            await respond(text=text)

        return _send

    async def _slash_note(self, ack: Any, command: Any, respond: Any) -> None:
        """Save a manual, max-trust fact: [@entity] <text>."""
        await ack()
        from prometheus.gateway.commands import cmd_note
        from prometheus.tools.builtin.wiki_compile import _memory_store

        await respond(text=cmd_note(_memory_store, self._cmd_text(command)))

    async def _slash_pairs(self, ack: Any, respond: Any) -> None:
        """Repair-pair flywheel stats."""
        await ack()
        from prometheus.gateway.commands import cmd_pairs

        await respond(text=cmd_pairs())

    async def _slash_approve(self, ack: Any, command: Any, respond: Any) -> None:
        """Approve a pending tool request."""
        await ack()
        from prometheus.gateway import commands as _cmds

        text_arg = self._cmd_text(command)
        request_id = text_arg.split()[0] if text_arg else ""
        text = await _cmds.cmd_approve(
            getattr(self, "_approval_queue", None), request_id,
            prefix=SLACK_COMMAND_PREFIX,
        )
        await respond(text=text)

    async def _slash_deny(self, ack: Any, command: Any, respond: Any) -> None:
        """Deny a pending tool request."""
        await ack()
        from prometheus.gateway import commands as _cmds

        text_arg = self._cmd_text(command)
        request_id = text_arg.split()[0] if text_arg else ""
        text = await _cmds.cmd_deny(
            getattr(self, "_approval_queue", None), request_id,
            prefix=SLACK_COMMAND_PREFIX,
        )
        await respond(text=text)

    async def _slash_pending(self, ack: Any, respond: Any) -> None:
        """List pending approval requests."""
        await ack()
        from prometheus.gateway import commands as _cmds

        await respond(text=_cmds.cmd_pending(getattr(self, "_approval_queue", None)))

    async def _slash_escalations(self, ack: Any, respond: Any) -> None:
        """Teacher-escalation counters + budget state."""
        await ack()
        from prometheus.gateway import commands as _cmds

        await respond(
            text=_cmds.cmd_escalations(getattr(self, "escalation_engine", None))
        )

    async def _slash_gepa(self, ack: Any, command: Any, respond: Any) -> None:
        """GEPA skill evolution: status | run | history.

        The approval prompt for ``run`` is delivered by the ApprovalQueue's
        transport (currently Telegram); progress/result messages land back
        in this Slack channel via the injected sender.
        """
        await ack()
        from prometheus.gateway import commands as _cmds

        text_arg = self._cmd_text(command)
        sub = text_arg.split()[0] if text_arg else "status"
        text = await _cmds.cmd_gepa(
            getattr(self, "_gepa_engine", None),
            getattr(self, "_approval_queue", None),
            sub,
            chat_id=None,
            send=self._channel_sender(command, respond),
            prefix=SLACK_COMMAND_PREFIX,
        )
        if text:
            await respond(text=text)

    async def _slash_symbiote(self, ack: Any, command: Any, respond: Any) -> None:
        """GitHub research → graft pipeline (shared dispatcher)."""
        await ack()
        from prometheus.gateway import commands as _cmds

        await _cmds.cmd_symbiote(
            self._channel_sender(command, respond),
            self._cmd_text(command),
            approval_queue=getattr(self, "_approval_queue", None),
            morph_engine=getattr(self, "_morph_engine", None),
            backup_vault=getattr(self, "_backup_vault", None),
            chat_id=None,
            prefix=SLACK_COMMAND_PREFIX,
        )

    async def _slash_audit(self, ack: Any, command: Any, respond: Any) -> None:
        """Web capability audit: show last | run | <category>."""
        await ack()
        from prometheus.gateway import commands as _cmds

        await _cmds.cmd_audit(
            self._channel_sender(command, respond),
            self._cmd_text(command),
            prefix=SLACK_COMMAND_PREFIX,
        )

    async def _slash_press(self, ack: Any, command: Any, respond: Any) -> None:
        """Printing Press CLI library: list | search | install | installed | update."""
        await ack()
        from prometheus.gateway import commands as _cmds

        await _cmds.cmd_press(
            self._channel_sender(command, respond),
            getattr(self, "_printing_press", None),
            self._cmd_text(command),
            approval_queue=getattr(self, "_approval_queue", None),
            chat_id=None,
            prefix=SLACK_COMMAND_PREFIX,
        )

    async def _slash_voice(self, ack: Any, respond: Any) -> None:
        """Platform-honest boundary: the voice-reply pipeline is Telegram-only."""
        await ack()
        await respond(text=(
            "Voice replies are not supported on Slack yet — the TTS pipeline "
            "(piper → opus/ogg voice notes) is wired to Telegram's voice-"
            "message API only. /voice modes remain a Telegram setting; on "
            "Slack all replies are text."
        ))

    # ---- provider overrides ------------------------------------------

    async def _apply_override(
        self, command: Any, respond: Any, preset_name: str,
    ) -> None:
        """Shared logic for /prometheus-claude, -gpt, -gemini, -xai, -grok."""
        from prometheus.gateway import commands as _cmds

        channel = self._cmd_channel(command)
        if not channel:
            await respond(text="Channel id missing from slash payload.")
            return
        session_key = f"slack:{channel}"
        text, applied = _cmds.cmd_provider_override(
            self.agent_loop,
            self._prometheus_config,
            session_key,
            preset_name,
            prefix=SLACK_COMMAND_PREFIX,
        )
        # Platform-honest: Telegram dispatches an inline trailing message
        # ("/claude what is 2+2?") through the new provider; Slack slash
        # payloads have no thread/reaction context wired for that yet.
        if applied and self._cmd_text(command):
            text += (
                "\n\nNote: inline message dispatch isn't supported on Slack "
                "yet — send your question as a normal message."
            )
        await respond(text=text)

    async def _slash_claude(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="claude")

    async def _slash_gpt(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="gpt")

    async def _slash_gemini(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="gemini")

    async def _slash_xai(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="xai")

    async def _slash_grok(self, ack: Any, command: Any, respond: Any) -> None:
        """Alias for /prometheus-xai."""
        await ack()
        await self._apply_override(command, respond, preset_name="xai")

    async def _slash_deepseek(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="deepseek")

    async def _slash_kimi(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="kimi")

    async def _slash_glm(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="glm")

    async def _slash_mimo(self, ack: Any, command: Any, respond: Any) -> None:
        await ack()
        await self._apply_override(command, respond, preset_name="mimo")

    async def _slash_local(self, ack: Any, command: Any, respond: Any) -> None:
        """Clear the per-channel provider override, back to primary."""
        await ack()
        from prometheus.gateway import commands as _cmds

        channel = self._cmd_channel(command)
        if not channel:
            await respond(text="Channel id missing from slash payload.")
            return
        text = _cmds.cmd_local_override(
            self.agent_loop, f"slack:{channel}",
            self.model_name, self.model_provider,
        )
        if self._cmd_text(command):
            text += (
                "\n\nNote: inline message dispatch isn't supported on Slack "
                "yet — send your question as a normal message."
            )
        await respond(text=text)

    async def _slash_route(self, ack: Any, command: Any, respond: Any) -> None:
        """Show current routing (primary vs override) for this channel."""
        await ack()
        from prometheus.gateway import commands as _cmds

        channel = self._cmd_channel(command)
        text = _cmds.cmd_route(
            self.agent_loop, f"slack:{channel}",
            self.model_name, self.model_provider,
            prefix=SLACK_COMMAND_PREFIX,
        )
        await respond(text=text)
