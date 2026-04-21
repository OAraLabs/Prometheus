"""Slack platform adapter — Socket Mode bot using slack-bolt.

Receives messages via WebSocket (no public URL needed), dispatches
to AgentLoop, sends responses back. Mirrors the TelegramAdapter pattern.

Key patterns adapted from NousResearch/hermes-agent Slack adapter:
- Message dedup (Socket Mode can redeliver events on reconnect)
- Markdown -> mrkdwn conversion (Slack uses its own format)
- Emoji reactions for processing feedback (eyes -> white_check_mark)
"""

from __future__ import annotations

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
    from prometheus.engine.session import SessionManager
    from prometheus.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Slack message length limit (text field)
MAX_MESSAGE_LENGTH = 3900


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
    ) -> None:
        super().__init__(config)
        self.agent_loop = agent_loop
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.model_provider = model_provider
        self._app: Any = None
        self._handler: Any = None
        self._start_time: float = 0.0

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

        # Start Socket Mode connection
        self._handler = AsyncSocketModeHandler(self._app, self.config.app_token)
        await self._handler.start_async()
        self._running = True
        self._start_time = time.monotonic()

        logger.info("Slack adapter started (Socket Mode)")

    async def stop(self) -> None:
        """Graceful shutdown of the Slack bot."""
        if self._handler and self._running:
            self._running = False
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

        session_id = f"slack:{channel}"
        session = self.session_manager.get_or_create(session_id)
        session.add_user_message(text)
        pre_len = len(session.get_messages())

        try:
            result = await self.agent_loop.run_async(
                system_prompt=self.system_prompt,
                messages=session.get_messages(),
                tools=self.tool_registry.list_schemas(),
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

        # Reply in thread if the original message was in a thread
        reply_thread = thread_ts or ts if thread_ts else None

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
        )
        await respond(text=text)

    async def _slash_help(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_help

        # Adapt command names for Slack (prefix with prometheus-)
        text = cmd_help().replace("/status", "/prometheus-status")
        text = text.replace("/model", "/prometheus-model")
        text = text.replace("/wiki", "/prometheus-wiki")
        text = text.replace("/sentinel", "/prometheus-sentinel")
        text = text.replace("/benchmark", "/prometheus-benchmark")
        text = text.replace("/context", "/prometheus-context")
        text = text.replace("/skills", "/prometheus-skills")
        text = text.replace("/reset", "/prometheus-reset")
        text = text.replace("/help", "/prometheus-help")
        await respond(text=text)

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

    async def _slash_skills(self, ack: Any, respond: Any) -> None:
        await ack()
        from prometheus.gateway.commands import cmd_skills

        await respond(text=cmd_skills())
