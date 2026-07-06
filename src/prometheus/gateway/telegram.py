"""Telegram platform adapter — full bot using python-telegram-bot.

Source: Novel code for Prometheus Sprint 6 (architecture inspired by Hermes
gateway.platforms.telegram).

Receives messages via long-polling, dispatches to AgentLoop.run_async(),
sends responses back with MarkdownV2 formatting and message chunking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram import BotCommand, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.commands import cmd_anatomy, cmd_beacon, cmd_doctor, cmd_profile
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

# MarkdownV2 special characters that must be escaped
_MARKDOWN_V2_ESCAPE = re.compile(r"([_\*\[\]\(\)~`>#+\-=|{}.!\\])")

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096

# Sprint 22 GRAFT-ROUTER-WIRE Phase 4: display labels for /claude, /gpt, etc.
# SPRINT G1: canonical table moved to the shared commands layer; kept as an
# alias for backward compatibility.
from prometheus.gateway.commands import (  # noqa: E402
    PROVIDER_PRESET_DISPLAY_NAMES as _PRESET_DISPLAY_NAMES,
)


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MARKDOWN_V2_ESCAPE.sub(r"\\\1", text)


def chunk_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks respecting Telegram's limit.

    Tries to split at newlines, then at spaces, then hard-truncates.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to split at a newline
        cut = text.rfind("\n", 0, max_length)
        if cut <= 0:
            # Try to split at a space
            cut = text.rfind(" ", 0, max_length)
        if cut <= 0:
            # Hard truncate
            cut = max_length

        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")

    return chunks


def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text, preserving code blocks and content."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)       # **bold** → bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)            # *italic* → italic
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # ## heading → heading
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)  # - item → • item
    text = re.sub(r'`{3}[\s\S]*?`{3}', lambda m: m.group(0), text)   # preserve code blocks
    text = re.sub(r'`(.+?)`', r'\1', text)              # `code` → code
    return text.strip()


class TelegramAdapter(BasePlatformAdapter):
    """Telegram bot adapter — receives messages, routes to AgentLoop."""

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
        # cost_tracker / escalation_engine / _approval_queue / _gepa_engine /
        # _printing_press / _backup_vault / _morph_engine: gateway-generic
        # subsystem slots, defaulted to None in BasePlatformAdapter.__init__
        # and attached by daemon.py via GatewaySubsystemRegistry (SPRINT G1).
        self._app: Application | None = None
        self._start_time: float = 0.0
        self._prometheus_config: dict[str, Any] = prometheus_config or {}

        if session_manager is None:
            from prometheus.engine.session import SessionManager as _SM
            session_manager = _SM()
        self.session_manager: SessionManager = session_manager

        # Sprint S1 Stream 2: SignalBus wired by daemon.py inside the
        # SENTINEL block via the ``signal_bus`` property setter. Until then
        # it's None and the subscribe path is a no-op.
        self._signal_bus: object | None = None

        # M6: one lock per session so a turn can't interleave with another on
        # the SAME session — e.g. a managed-task completion calling inject_turn
        # while a user turn is mid-flight, which would race both coroutines
        # appending to session.messages (scrambled order + duplicate
        # turn_index). Different sessions never contend; PTB already serializes
        # user×user, so this specifically guards user×re-engagement.
        self._turn_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Sprint S1 Stream 2: SignalBus subscription for user-visible events
    # ------------------------------------------------------------------

    @property
    def signal_bus(self) -> object | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: object | None) -> None:
        """Subscribe to skill/memory/curator events on bus assignment.

        Called from daemon.py after the SignalBus is constructed. The
        subscription wires four signal kinds to handler coroutines that
        post a short notification to the last active Telegram chat.

        Default behaviour is QUIET (one-line emoji + name). The
        ``/notifications verbose`` command (Sprint 1 Stream 3) toggles
        between quiet/verbose/off and persists in
        ``gateway.telegram.skill_event_notifications`` config.
        """
        self._signal_bus = bus
        if bus is None:
            return
        try:
            bus.subscribe("skill_created", self._on_signal_skill_created)
            bus.subscribe("skill_refined", self._on_signal_skill_refined)
            bus.subscribe("memory_updated", self._on_signal_memory_updated)
            bus.subscribe("curator_report", self._on_signal_curator_report)
            logger.info(
                "Telegram: subscribed to skill/memory/curator signals"
            )
        except Exception:
            logger.warning(
                "Telegram: failed to subscribe to SignalBus", exc_info=True
            )

    def _notification_mode(self) -> str:
        """Return 'quiet' | 'verbose' | 'off'.

        Checks the runtime override file first (set via /notifications),
        then falls back to ``gateway.skill_event_notifications`` in
        prometheus.yaml.
        """
        # Runtime override via /notifications has precedence.
        try:
            from prometheus.gateway.commands import get_notifications_mode
            mode = get_notifications_mode(default="")
            if mode:
                return mode
        except Exception:
            pass
        cfg = self._prometheus_config.get("gateway", {})
        if isinstance(cfg, dict):
            return str(cfg.get("skill_event_notifications", "quiet")).lower()
        return "quiet"

    async def _send_notification(self, text: str) -> None:
        """Send *text* to the last active chat if we have one."""
        chat_id = self._load_chat_id()
        if chat_id is None:
            return
        try:
            await self.send(chat_id, text, parse_mode=None)
        except Exception:
            logger.debug("Telegram: notification send failed", exc_info=True)

    async def _on_signal_skill_created(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        name = payload.get("skill_name", "(unnamed)")
        if mode == "verbose":
            trigger = payload.get("trigger_task", "")
            summary = payload.get("summary", "")
            text = f"🎓 New skill: {name}"
            if summary:
                text += f"\n   {summary}"
            if trigger:
                text += f"\n   (built while: {trigger[:120]})"
        else:
            text = f"🎓 New skill: {name}"
        await self._send_notification(text)

    async def _on_signal_skill_refined(self, signal: Any) -> None:
        mode = self._notification_mode()
        if mode == "off":
            return
        payload = getattr(signal, "payload", {}) or {}
        name = payload.get("skill_name", "(unnamed)")
        if mode == "verbose":
            summary = payload.get("summary", "")
            text = f"📚 Updated skill: {name}"
            if summary:
                text += f"\n   {summary}"
        else:
            text = f"📚 Updated skill: {name}"
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
            text = f"🧠 {target} {operation}"
            if preview:
                text += f"\n   {preview}"
        else:
            text = f"🧠 {target} {operation}"
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
                f"   /curator show for the full report"
            )
        else:
            text = f"📋 Curator: {reviewed} reviewed, {prunings} archived"
        await self._send_notification(text)

    async def start(self) -> None:
        """Build the telegram Application and start long-polling."""
        if not self.config.token:
            raise ValueError("Telegram bot token is required")

        builder = Application.builder().token(self.config.token)

        # Apply network config if proxy is set
        if self.config.proxy_url:
            builder.proxy(self.config.proxy_url)

        self._app = builder.build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("clear", self._cmd_clear))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))
        self._app.add_handler(CommandHandler("model", self._cmd_model))
        self._app.add_handler(CommandHandler("wiki", self._cmd_wiki))
        self._app.add_handler(CommandHandler("note", self._cmd_note))
        self._app.add_handler(CommandHandler("sentinel", self._cmd_sentinel))
        self._app.add_handler(CommandHandler("benchmark", self._cmd_benchmark))
        self._app.add_handler(CommandHandler("context", self._cmd_context))
        self._app.add_handler(CommandHandler("skills", self._cmd_skills))
        # Sprint S1 Stream 3: visible memory & skills inspection commands
        self._app.add_handler(CommandHandler("memory", self._cmd_memory))
        self._app.add_handler(CommandHandler("curator", self._cmd_curator))
        self._app.add_handler(CommandHandler("notifications", self._cmd_notifications))
        # Voice mode toggle (TTS-out): auto | on | off
        self._app.add_handler(CommandHandler("voice", self._cmd_voice))
        # Sprint S4 A3: /health — silent-failure telemetry surface
        self._app.add_handler(CommandHandler("health", self._cmd_health))
        # SignalBus Persistence sprint: /events — persisted signal-bus events
        self._app.add_handler(CommandHandler("events", self._cmd_events))
        # SPRINT-2 WS1: Durability & Steering — mid-turn steer + queued prompts
        self._app.add_handler(CommandHandler("steer", self._cmd_steer))
        self._app.add_handler(CommandHandler("queue", self._cmd_queue))
        self._app.add_handler(CommandHandler("unqueue", self._cmd_unqueue))
        # Telegram strips dashes from command names — use compact form.
        self._app.add_handler(CommandHandler("clearsteers", self._cmd_clear_steers))
        self._app.add_handler(CommandHandler("anatomy", self._cmd_anatomy))
        self._app.add_handler(CommandHandler("doctor", self._cmd_doctor))
        self._app.add_handler(CommandHandler("profile", self._cmd_profile))
        self._app.add_handler(CommandHandler("beacon", self._cmd_beacon))
        self._app.add_handler(CommandHandler("tools", self._cmd_tools))
        self._app.add_handler(CommandHandler("pairs", self._cmd_pairs))
        # Sprint 15b GRAFT: approval queue commands
        self._app.add_handler(CommandHandler("approve", self._cmd_approve))
        self._app.add_handler(CommandHandler("deny", self._cmd_deny))
        self._app.add_handler(CommandHandler("pending", self._cmd_pending))
        # SUNRISE Session B: GEPA skill evolution
        self._app.add_handler(CommandHandler("gepa", self._cmd_gepa))
        # GRAFT-SYMBIOTE Session A: GitHub research → graft pipeline
        self._app.add_handler(CommandHandler("symbiote", self._cmd_symbiote))
        # WEAVE Session B: web capability audit
        self._app.add_handler(CommandHandler("audit", self._cmd_audit))
        # WEAVE-PRESS: Printing Press CLI discovery and install
        self._app.add_handler(CommandHandler("press", self._cmd_press))
        # Sprint 22 GRAFT-ROUTER-WIRE Phase 4: direct-mode provider override commands
        self._app.add_handler(CommandHandler("claude", self._cmd_claude))
        self._app.add_handler(CommandHandler("gpt", self._cmd_gpt))
        self._app.add_handler(CommandHandler("gemini", self._cmd_gemini))
        self._app.add_handler(CommandHandler("xai", self._cmd_xai))
        self._app.add_handler(CommandHandler("grok", self._cmd_grok))
        # CLOUD EXPANSION (2026-07): four more provider override commands
        self._app.add_handler(CommandHandler("deepseek", self._cmd_deepseek))
        self._app.add_handler(CommandHandler("kimi", self._cmd_kimi))
        self._app.add_handler(CommandHandler("glm", self._cmd_glm))
        self._app.add_handler(CommandHandler("mimo", self._cmd_mimo))
        self._app.add_handler(CommandHandler("local", self._cmd_local))
        self._app.add_handler(CommandHandler("route", self._cmd_route))
        # SPRINT-TEACHER-ESCALATION Phase 3: escalation stats / budget state
        self._app.add_handler(
            CommandHandler("escalations", self._cmd_escalations)
        )
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        # Sprint 15 GRAFT: media handlers (additive — Hermes parity)
        self._app.add_handler(MessageHandler(filters.PHOTO, self._handle_photo))
        self._app.add_handler(MessageHandler(filters.VOICE, self._handle_voice))
        self._app.add_handler(MessageHandler(filters.Document.ALL, self._handle_document))
        self._app.add_handler(MessageHandler(filters.Sticker.ALL, self._handle_sticker))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        self._start_time = time.monotonic()

        # Register command menu with Telegram
        try:
            await self._app.bot.set_my_commands([
                BotCommand("start", "Check if Prometheus is online"),
                BotCommand("status", "Model, uptime, tools, SENTINEL state"),
                BotCommand("help", "List commands and capabilities"),
                BotCommand("reset", "Clear conversation context"),
                BotCommand("clear", "Clear conversation context"),
                BotCommand("model", "Show current model and provider"),
                BotCommand("wiki", "Wiki stats and recent entries"),
                BotCommand("sentinel", "SENTINEL subsystem status"),
                BotCommand("benchmark", "Run a quick smoke test"),
                BotCommand("context", "Context window usage"),
                BotCommand("skills", "Skills: list | show | pin | unpin | history"),
                # Sprint S1 Stream 3
                BotCommand("memory", "Memory: show [user] | limits"),
                BotCommand("curator", "Curator: status | show | run [dry]"),
                BotCommand("notifications", "Skill/memory/curator notifications: off | quiet | verbose"),
                BotCommand("voice", "Voice replies: auto | on | off (auto mirrors input modality)"),
                BotCommand("health", "Silent-failure telemetry: last 24h or `/health 168 verbose`"),
                BotCommand("anatomy", "Infrastructure snapshot"),
                BotCommand("doctor", "Diagnostic health check"),
                BotCommand("profile", "Show or switch agent profile"),
                BotCommand("beacon", "Web bridge / dashboard status"),
                BotCommand("approve", "Approve a pending tool request"),
                BotCommand("deny", "Deny a pending tool request"),
                BotCommand("pending", "List pending approval requests"),
                BotCommand("gepa", "GEPA skill evolution: status | run | history"),
                BotCommand("symbiote", "GitHub graft pipeline: <problem> | approve | graft | morph | swap | backup | backups | restore | status | abort | history"),
                BotCommand("audit", "Web capability audit: show last | run | web | <category>"),
                BotCommand("press", "Printing Press CLI library: list | search | install | installed | update"),
                # Phase 4: direct-mode provider overrides
                BotCommand("claude", "Route this chat through Anthropic Claude"),
                BotCommand("gpt", "Route this chat through OpenAI GPT"),
                BotCommand("gemini", "Route this chat through Google Gemini"),
                BotCommand("xai", "Route this chat through xAI Grok"),
                BotCommand("grok", "Alias for /xai"),
                BotCommand("deepseek", "Route this chat through DeepSeek"),
                BotCommand("kimi", "Route this chat through Kimi (Moonshot)"),
                BotCommand("glm", "Route this chat through GLM (Z.ai)"),
                BotCommand("mimo", "Route this chat through MiMo (Xiaomi)"),
                BotCommand("local", "Clear override, back to primary"),
                BotCommand("route", "Show current routing (primary vs override)"),
            ])
        except Exception as exc:
            logger.warning("Failed to register command menu: %s", exc)

        logger.info("Telegram adapter started (polling)")

        # Send startup greeting to create session + signal readiness
        await self._send_startup_greeting()

    def _last_chat_id_path(self) -> str:
        """Path to persist the last active Telegram chat ID across restarts."""
        from prometheus.config.paths import get_config_dir
        return str(get_config_dir() / "last_telegram_chat_id")

    def _save_chat_id(self, chat_id: int) -> None:
        """Persist chat ID so startup greeting works after restart."""
        try:
            import os
            path = self._last_chat_id_path()
            with open(path, "w") as f:
                f.write(str(chat_id))
        except Exception:
            pass

    def _load_chat_id(self) -> int | None:
        """Load persisted chat ID from last session."""
        try:
            with open(self._last_chat_id_path()) as f:
                return int(f.read().strip())
        except Exception:
            return None

    async def _send_startup_greeting(self) -> None:
        """Send a short greeting on daemon start to create the Telegram session
        and signal to the user (in Telegram + Beacon) that Prometheus is online."""
        import random
        greetings = [
            "Online and ready. What can I help with?",
            "Systems up. What are we working on?",
            "Back online — ready when you are.",
            "Prometheus is live. What's on the agenda?",
            "Daemon restarted. All systems nominal — fire away.",
        ]
        msg = random.choice(greetings)

        # Find chat ID: persisted file → session manager → config
        chat_id: int | None = self._load_chat_id()

        if not chat_id and self.session_manager:
            for key in list(self.session_manager._sessions.keys()):
                if key.startswith("telegram:"):
                    try:
                        chat_id = int(key.split(":", 1)[1])
                        break
                    except (ValueError, IndexError):
                        pass

        if not chat_id and self.config.allowed_chat_ids:
            chat_id = self.config.allowed_chat_ids[0]

        if chat_id:
            try:
                result = await self.send(chat_id, msg)
                if result.success:
                    logger.info("Sent startup greeting to chat %d", chat_id)
                    if self.session_manager:
                        session_key = f"telegram:{chat_id}"
                        self.session_manager.get_or_create(session_key)
            except Exception as exc:
                logger.warning("Failed to send startup greeting: %s", exc)
        else:
            logger.info("No known chat ID for startup greeting (will create session on first message)")

    async def stop(self) -> None:
        """Graceful shutdown of the Telegram bot."""
        if self._app and self._running:
            self._running = False
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram adapter stopped")

    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> SendResult:
        """Send a message to a Telegram chat, chunking if needed."""
        if not self._app:
            return SendResult(success=False, error="Bot not initialized")

        chunks = chunk_message(text)
        last_message_id: int | None = None

        for i, chunk in enumerate(chunks):
            try:
                # Try with MarkdownV2 first, fall back to plain text
                effective_parse_mode = parse_mode or self.config.parse_mode
                try:
                    msg = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=escape_markdown_v2(chunk) if effective_parse_mode == "MarkdownV2" else chunk,
                        parse_mode=effective_parse_mode,
                        reply_to_message_id=reply_to if i == 0 else None,
                    )
                except Exception:
                    # Fallback: send as plain text
                    msg = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        reply_to_message_id=reply_to if i == 0 else None,
                    )
                last_message_id = msg.message_id
            except Exception as exc:
                logger.error("Failed to send message to chat %d: %s", chat_id, exc)
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=last_message_id)

    async def on_message(self, event: MessageEvent) -> None:
        """Handle an incoming message — dispatch to agent and reply."""
        if not self.config.chat_allowed(event.chat_id):
            logger.warning(
                "Ignoring message from unauthorized chat %d (user %d)",
                event.chat_id,
                event.user_id,
            )
            return

        # Persist chat ID for startup greeting on next restart
        self._save_chat_id(event.chat_id)

        await self._dispatch_to_agent(event)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        if update.effective_chat is None:
            return
        await self.send(
            update.effective_chat.id,
            "Prometheus is online. Send me a message and I'll help you.",
            parse_mode=None,
        )

    async def _cmd_clear(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /clear command — reset conversation history."""
        if update.effective_chat is None:
            return
        session_key = f"{Platform.TELEGRAM.value}:{update.effective_chat.id}"
        self.session_manager.clear(session_key)
        await self.send(
            update.effective_chat.id,
            "Conversation cleared.",
            parse_mode=None,
        )

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command — list commands and capabilities."""
        if update.effective_chat is None:
            return
        text = (
            "Prometheus — Sovereign AI Agent\n"
            "\n"
            "Commands:\n"
            "/status    — Model, uptime, tools, memory, SENTINEL\n"
            "/route     — Current provider + available overrides\n"
            "/anatomy   — Hardware, GPU, VRAM, infrastructure\n"
            "/doctor    — Diagnostic health check against model registry\n"
            "/wiki      — Wiki stats and recent entries\n"
            "/sentinel  — SENTINEL subsystem status\n"
            "/benchmark — Run a quick smoke test\n"
            "/context   — Context window usage\n"
            "/skills    — list | show | pin | unpin | history\n"
            "/memory    — show [user] | limits\n"
            "/curator   — status | show | run [dry]\n"
            "/notifications — off | quiet | verbose\n"
            "/voice     — voice replies: auto | on | off (auto mirrors input)\n"
            "/health    — silent-failure telemetry (last 24h)\n"
            "/reset     — Clear conversation context\n"
            "/help      — This message\n"
            "\n"
            "Provider overrides (this chat only, sticky until /local):\n"
            "/claude    — Anthropic Claude\n"
            "/gpt       — OpenAI GPT\n"
            "/gemini    — Google Gemini\n"
            "/xai       — xAI Grok  (alias: /grok)\n"
            "/deepseek  — DeepSeek\n"
            "/kimi      — Kimi (Moonshot)\n"
            "/glm       — GLM (Z.ai)\n"
            "/mimo      — MiMo (Xiaomi)\n"
            "/local     — Back to primary\n"
            "\n"
            "Send any message to chat with the agent."
        )
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /reset command — clear conversation context."""
        if update.effective_chat is None:
            return
        session_key = f"{Platform.TELEGRAM.value}:{update.effective_chat.id}"
        self.session_manager.clear(session_key)
        await self.send(
            update.effective_chat.id,
            "Conversation context reset.",
            parse_mode=None,
        )

    async def _cmd_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /model command — delegates to /route (Phase 4 absorbed it).

        Kept as a backward-compatibility alias for /route so existing users
        / external docs that still reference /model keep working. The new
        /route command is override-aware; /model no longer returns a plain
        primary-only report.
        """
        await self._cmd_route(update, context)

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command — model, uptime, tools, memory, SENTINEL."""
        if update.effective_chat is None:
            return

        lines: list[str] = ["Prometheus Status\n"]

        # Model
        lines.append(f"Model: {self.model_name or '(unknown)'}")
        lines.append(f"Provider: {self.model_provider or '(unknown)'}")

        # Uptime
        if self._start_time:
            elapsed = int(time.monotonic() - self._start_time)
            h, remainder = divmod(elapsed, 3600)
            m, s = divmod(remainder, 60)
            lines.append(f"Uptime: {h}h {m}m {s}s")

        # Tools
        lines.append(f"Tools: {len(self.tool_registry.list_tools())}")

        # Cost tracking (cloud providers)
        if self.cost_tracker is not None:
            lines.append(self.cost_tracker.report())

        # Memory stats
        try:
            from prometheus.tools.builtin.wiki_compile import _memory_store
            if _memory_store is not None:
                facts = _memory_store.get_all_memories(limit=10000)
                lines.append(f"Memory facts: {len(facts)}")
            else:
                lines.append("Memory: not initialized")
        except Exception:
            lines.append("Memory: unavailable")

        # SENTINEL state
        try:
            from prometheus.tools.builtin.sentinel_status import (
                _autodream,
                _observer,
            )
            if _observer is not None and _autodream is not None:
                state = "dreaming" if _autodream.dreaming else (
                    "active" if _observer.started else "idle"
                )
                lines.append(f"\nSENTINEL: {state}")
                lines.append(f"Dream cycles: {_autodream.cycle_count}")
                if _autodream.last_results:
                    lines.append("Last dream results:")
                    for r in _autodream.last_results:
                        status = "OK" if not r.error else f"FAIL: {r.error}"
                        lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
            else:
                lines.append("\nSENTINEL: not initialized")
        except Exception:
            lines.append("\nSENTINEL: unavailable")

        # SPRINT-2 WS1: queued steers + queued prompts for THIS chat.
        # Per-session counters live on the ChatSession instance; surface
        # them here so /status doubles as the canonical "what's in flight"
        # view.
        session = self._resolve_session_for_command(update)
        if session is not None:
            if session.queued_steers:
                lines.append(
                    f"\n📍 Queued steers: {len(session.queued_steers)}"
                )
                for s in session.queued_steers[:3]:
                    preview = s if len(s) <= 60 else s[:57] + "..."
                    lines.append(f"   - {preview}")
                if len(session.queued_steers) > 3:
                    lines.append(
                        f"   ... and {len(session.queued_steers) - 3} more"
                    )
            if session.queued_prompts:
                lines.append(
                    f"📥 Queued prompts: {len(session.queued_prompts)}"
                )
                for p in session.queued_prompts[:3]:
                    preview = p if len(p) <= 60 else p[:57] + "..."
                    lines.append(f"   - {preview}")
                if len(session.queued_prompts) > 3:
                    lines.append(
                        f"   ... and {len(session.queued_prompts) - 3} more"
                    )

        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    async def _cmd_tools(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /tools command — tool call telemetry dashboard."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds
        await self.send(update.effective_chat.id, _cmds.cmd_tools(), parse_mode=None)

    async def _cmd_pairs(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /pairs command — repair-pair flywheel stats."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds
        await self.send(update.effective_chat.id, _cmds.cmd_pairs(), parse_mode=None)

    async def _cmd_wiki(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /wiki command — wiki stats and recent entries."""
        if update.effective_chat is None:
            return

        wiki_index = Path.home() / ".prometheus" / "wiki" / "index.md"
        if not wiki_index.exists():
            await self.send(
                update.effective_chat.id,
                "Wiki: no index found at ~/.prometheus/wiki/index.md",
                parse_mode=None,
            )
            return

        try:
            content = wiki_index.read_text(encoding="utf-8")
            entries: list[str] = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("- ["):
                    entries.append(line)

            lines = [f"Wiki: {len(entries)} pages"]

            # Last modified
            mtime = wiki_index.stat().st_mtime
            from datetime import datetime, timezone
            updated = datetime.fromtimestamp(mtime, tz=timezone.utc)
            lines.append(f"Last updated: {updated.strftime('%Y-%m-%d %H:%M UTC')}")

            # Show last 5 entries
            if entries:
                lines.append("\nRecent entries:")
                for entry in entries[-5:]:
                    lines.append(f"  {entry}")

            await self.send(
                update.effective_chat.id, "\n".join(lines), parse_mode=None
            )
        except Exception as exc:
            await self.send(
                update.effective_chat.id,
                f"Wiki: error reading index — {exc}",
                parse_mode=None,
            )

    async def _cmd_sentinel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /sentinel command — reuses SentinelStatusTool logic."""
        if update.effective_chat is None:
            return

        try:
            from prometheus.tools.builtin.sentinel_status import (
                _autodream,
                _observer,
                _signal_bus,
            )
        except ImportError:
            await self.send(
                update.effective_chat.id,
                "SENTINEL module not available.",
                parse_mode=None,
            )
            return

        if _signal_bus is None or _observer is None or _autodream is None:
            await self.send(
                update.effective_chat.id,
                "SENTINEL not initialized. Is the daemon running with sentinel enabled?",
                parse_mode=None,
            )
            return

        lines: list[str] = ["SENTINEL Status\n"]

        # Observer
        idle_secs = int(time.time() - _observer.last_activity)
        lines.append("Observer:")
        lines.append(f"  Active: {_observer.started}")
        lines.append(f"  Last activity: {idle_secs}s ago")
        lines.append(f"  Pending nudges: {len(_observer.pending_nudges)}")

        # AutoDream
        lines.append("\nAutoDream Engine:")
        lines.append(f"  Dreaming: {_autodream.dreaming}")
        lines.append(f"  Cycles completed: {_autodream.cycle_count}")
        if _autodream.last_cycle_time:
            ago = int(time.time() - _autodream.last_cycle_time)
            lines.append(f"  Last cycle: {ago}s ago")

        # Signal bus
        lines.append("\nSignal Bus:")
        lines.append(f"  Total signals: {_signal_bus.signal_count}")
        lines.append(f"  Subscribers: {_signal_bus.subscriber_count}")

        # Recent signals
        recent = _signal_bus.recent(limit=10)
        if recent:
            lines.append("\nRecent Signals:")
            for sig in recent:
                ago = int(time.time() - sig.timestamp)
                lines.append(f"  [{sig.kind}] from {sig.source} ({ago}s ago)")

        # Last dream results
        if _autodream.last_results:
            lines.append("\nLast Dream Cycle:")
            for r in _autodream.last_results:
                status = "OK" if not r.error else f"FAIL: {r.error}"
                lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
                for k, v in r.summary.items():
                    lines.append(f"    {k}: {v}")

        # Pending nudges
        if _observer.pending_nudges:
            lines.append("\nPending Nudges:")
            for nudge in _observer.pending_nudges[:5]:
                lines.append(f"  [{nudge.nudge_type}] {nudge.message[:80]}")

        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    async def _cmd_benchmark(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /benchmark command — run a quick smoke test."""
        if update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        await self.send(chat_id, "Running benchmark...", parse_mode=None)

        try:
            t0 = time.monotonic()
            result = await self.agent_loop.run_async(
                system_prompt="You are a helpful assistant. Be concise.",
                user_message="What is 2+2? Reply with just the number.",
                tools=[],
                # Phase 3.5: /benchmark is a diagnostic, not a user chat —
                # use reserved "system" so it never inherits user overrides.
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
            await self.send(chat_id, "\n".join(lines), parse_mode=None)
        except Exception as exc:
            await self.send(
                chat_id,
                f"Benchmark: FAIL\nError: {exc}",
                parse_mode=None,
            )

    async def _cmd_skills(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /skills — subcommand dispatcher.

        Subcommands (Sprint S1 Stream 3):
          (none)                    list builtin + user skills (existing)
          list                      list auto-skills with state + last-used
          show <name>               display SKILL.md content
          pin <name>                protect from Curator auto-prune
          unpin <name>
          history <name>            list SkillRefiner backup files
        """
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        args = list(context.args or [])

        # No-arg: keep existing registry-listing behaviour.
        if not args:
            try:
                from prometheus.skills.loader import load_skill_registry
                registry = load_skill_registry()
                skills = registry.list_skills()
            except Exception as exc:
                await self.send(chat_id, f"Skills: error loading registry — {exc}", parse_mode=None)
                return
            if not skills:
                await self.send(chat_id, "No skills available.", parse_mode=None)
                return
            lines = [f"Skills ({len(skills)})\n"]
            for skill in skills:
                source_tag = f" [{skill.source}]" if skill.source else ""
                lines.append(f"  {skill.name}{source_tag}")
                if skill.description:
                    lines.append(f"    {skill.description[:80]}")
            lines.append(
                "\nSubcommands: /skills list · show <name> · pin <name> "
                "· unpin <name> · history <name>"
            )
            await self.send(chat_id, "\n".join(lines), parse_mode=None)
            return

        from prometheus.gateway import commands as _cmds

        sub = args[0].lower()
        name = " ".join(args[1:]).strip()

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
                "Use: /skills [list | show <name> | pin <name> | "
                "unpin <name> | history <name>]"
            )
        await self.send(chat_id, text, parse_mode=None)

    async def _cmd_memory(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /memory — show/limits subcommand dispatcher (Sprint S1 Stream 3).

        Subcommands:
          show [user]               display MEMORY.md (default) or USER.md
          limits                    show char ceilings + current usage
        """
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        args = list(context.args or [])
        from prometheus.gateway import commands as _cmds

        if not args:
            text = (
                "Memory commands:\n"
                "  /memory show           — MEMORY.md content\n"
                "  /memory show user      — USER.md content\n"
                "  /memory limits         — char ceilings + usage"
            )
        else:
            sub = args[0].lower()
            tail = " ".join(args[1:]).strip().lower()
            if sub == "show":
                target = "user" if tail == "user" else "memory"
                text = _cmds.cmd_memory_show(target=target)
            elif sub == "limits":
                text = _cmds.cmd_memory_limits()
            else:
                text = (
                    f"Unknown subcommand: {sub}\n"
                    "Use: /memory [show [user] | limits]"
                )
        await self.send(chat_id, text, parse_mode=None)

    async def _cmd_note(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /note [@entity] <text> — file a manual, max-trust fact.

        Writes ONE fact to memory.db (source=manual, confidence 1.0, manual=1);
        compile projects it to a page. Never writes the wiki directly.
        """
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        raw = " ".join(context.args or [])
        from prometheus.gateway import commands as _cmds
        from prometheus.tools.builtin.wiki_compile import _memory_store
        text = _cmds.cmd_note(_memory_store, raw)
        await self.send(chat_id, text, parse_mode=None)

    async def _cmd_curator(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /curator — show/status/run subcommand dispatcher (Sprint S1 Stream 3)."""
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        args = list(context.args or [])
        from prometheus.gateway import commands as _cmds

        if not args:
            text = (
                "Curator commands:\n"
                "  /curator status        — last/next run, pinned skills\n"
                "  /curator show          — most recent REPORT.md\n"
                "  /curator run           — trigger an immediate pass\n"
                "  /curator run dry       — dry-run (no file moves)"
            )
        else:
            sub = args[0].lower()
            if sub == "show":
                text = _cmds.cmd_curator_show()
            elif sub == "status":
                text = _cmds.cmd_curator_status()
            elif sub == "run":
                dry = (len(args) >= 2 and args[1].lower().startswith("dry"))
                text = await _cmds.cmd_curator_run(dry_run=dry)
            else:
                text = (
                    f"Unknown subcommand: {sub}\n"
                    "Use: /curator [show | status | run [dry]]"
                )
        await self.send(chat_id, text, parse_mode=None)

    async def _cmd_health(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /health — surface silent-failure telemetry (Sprint 4 A3).

        Args:
          (no args)             24h summary, top-line per subsystem
          verbose               24h summary + tracebacks
          <hours>               numeric window override (e.g. /health 168 → 7d)
          <hours> verbose       both
        """
        if update.effective_chat is None:
            return
        args = list(context.args or [])
        verbose = False
        since_hours = 24.0
        for tok in args:
            t = tok.strip().lower()
            if t == "verbose":
                verbose = True
                continue
            try:
                since_hours = float(t)
            except ValueError:
                pass

        from prometheus.gateway import commands as _cmds
        text = _cmds.cmd_health(verbose=verbose, since_hours=since_hours)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    # ------------------------------------------------------------------
    # SPRINT-2 WS1 — Mid-turn steer + queued prompts
    # ------------------------------------------------------------------

    def _resolve_session_for_command(
        self, update: Update,
    ) -> "ChatSession | None":  # noqa: F821 — forward ref, ChatSession imported in module
        """Return the live ChatSession for the user's chat, or None.

        Used by /steer, /queue, /unqueue, /clear-steers to find the
        session-state object the agent loop is consuming. We don't
        ``get_or_create`` here — if there's no active session, the
        command is a no-op.
        """
        if update.effective_chat is None:
            return None
        # Build the session key the same way _dispatch_to_agent does.
        # MessageEvent.session_key() = f"{platform}:{chat_id}"
        # which here resolves to "telegram:<chat_id>".
        session_key = f"telegram:{update.effective_chat.id}"
        return self.session_manager._sessions.get(session_key)

    def _plain_sender(self, chat_id: int):
        """Return ``async send(text)`` bound to *chat_id* (plain text).

        SPRINT G1: multi-message shared command flows (gepa/symbiote/audit/
        press) take an injected send callable instead of an adapter handle.
        """
        async def _send(text: str) -> None:
            await self.send(chat_id, text, parse_mode=None)
        return _send

    async def _cmd_steer(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /steer — inject text into the current session.

        Arrives at the agent as a system-prompt addendum on the next
        model call (after the current tool batch finishes). Does NOT
        interrupt the agent or start a new user turn.
        """
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        text = " ".join(context.args or []).strip()
        session = self._resolve_session_for_command(update)
        await self.send(
            update.effective_chat.id, _cmds.cmd_steer(session, text), parse_mode=None
        )

    async def _cmd_queue(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /queue — fire <text> as a fresh turn after the current
        one ends."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        text = " ".join(context.args or []).strip()
        # /queue creates the session if missing so a queued prompt on a quiet
        # chat fires when the user kicks off their first message — but only when
        # there's something to queue (a bare /queue is usage, no session made).
        session = None
        if text:
            session = self.session_manager.get_or_create(
                f"telegram:{update.effective_chat.id}"
            )
        await self.send(
            update.effective_chat.id, _cmds.cmd_queue(session, text), parse_mode=None
        )

    async def _cmd_unqueue(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /unqueue — drop the most recently queued prompt."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        session = self._resolve_session_for_command(update)
        await self.send(
            update.effective_chat.id, _cmds.cmd_unqueue(session), parse_mode=None
        )

    async def _cmd_clear_steers(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /clearsteers — drop all pending steers without injection."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        session = self._resolve_session_for_command(update)
        await self.send(
            update.effective_chat.id, _cmds.cmd_clearsteers(session), parse_mode=None
        )

    async def _cmd_events(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /events — surface persisted SignalBus events.

        Subcommands:
          (no args) | recent  → last 20 events across all types
          skills              → recent skill_created / skill_refined
          memory              → recent memory_updated
          curator             → recent curator_report / curator_degraded
          show <id>           → full payload for a specific event
        """
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        arg = " ".join(context.args or []).strip()
        text = _cmds.cmd_events(arg=arg)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_notifications(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /notifications — toggle quiet/verbose/off (Sprint S1 Stream 3)."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        arg = " ".join(context.args or []).strip()
        text = _cmds.cmd_notifications(mode=arg)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /voice — per-chat voice reply mode.

        Modes:
          auto  (default) — mirror input modality (voice in → voice out)
          on              — always reply with voice (TTS for every response)
          off             — always reply with text (disable voice replies)

        With no argument, reports the current effective mode. Persists
        across daemon restarts in ~/.prometheus/voice_modes.json.
        """
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        arg = (" ".join(context.args or []).strip()).lower()

        from prometheus.gateway import commands as _cmds
        text = _cmds.cmd_voice(str(chat_id), arg, self._voice_config())
        await self.send(chat_id, text, parse_mode=None)

    async def _cmd_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /context command — show context window usage."""
        if update.effective_chat is None:
            return

        from prometheus.context.token_estimation import estimate_tokens

        # Read effective_limit from config (with model override)
        try:
            from prometheus.context.budget import TokenBudget
            budget = TokenBudget.from_config(model=self.model_name)
            effective_limit = budget.effective_limit
            reserved_output = budget.reserved_output
        except Exception:
            effective_limit = 24000
            reserved_output = 2000

        # Estimate system prompt cost
        prompt_tokens = estimate_tokens(self.system_prompt)

        # Available for conversation
        available = effective_limit - reserved_output
        headroom = max(0, available - prompt_tokens)
        usage_pct = (prompt_tokens / available * 100) if available > 0 else 0

        lines = [
            "Context Window\n",
            f"Window size:    {effective_limit:,} tokens",
            f"Reserved output: {reserved_output:,} tokens",
            f"Available:       {available:,} tokens",
            f"",
            f"System prompt:   {prompt_tokens:,} tokens ({usage_pct:.0f}%)",
            f"Headroom:        {headroom:,} tokens",
            f"",
            f"Model: {self.model_name or '(unknown)'}",
        ]

        # Show bar visualization
        bar_len = 20
        filled = round(usage_pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"[{bar}] {usage_pct:.0f}% used")

        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    # ------------------------------------------------------------------
    # Sprint 22 GRAFT-ROUTER-WIRE Phase 4: direct-mode provider overrides
    # ------------------------------------------------------------------
    # /claude, /gpt, /gemini, /xai, /grok  — set session override to cloud provider
    # /local                               — clear override, return to primary
    # /route                               — show current effective provider
    #
    # All of these key the override on the Telegram session_key
    # ("telegram:<chat_id>"), which matches what `_dispatch_to_agent` threads
    # into `AgentLoop.run_async(session_id=...)`. Overrides set here apply only
    # to this chat, not to other chats, Slack, web, or system paths (evals,
    # benchmarks, smoke tests) — those use reserved session_ids None/"system"
    # which the router always resolves to primary.

    async def _apply_override(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        preset_name: str,
    ) -> None:
        """Shared logic for /claude, /gpt, /gemini, /xai, /grok.

        Thin wrapper over commands.cmd_provider_override (SPRINT G1); if the
        command had an inline message (e.g., ``/claude what is 2+2?``) and the
        override applied, dispatches it immediately via the normal agent path
        so the user gets an answer in one shot.
        """
        if update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        session_key = f"{Platform.TELEGRAM.value}:{chat_id}"

        from prometheus.gateway import commands as _cmds
        text, applied = _cmds.cmd_provider_override(
            self.agent_loop,
            self._prometheus_config,
            session_key,
            preset_name,
        )
        await self.send(chat_id, text, parse_mode=None)
        if not applied:
            return

        # Inline message dispatch: /claude <message> answers <message> through
        # the override provider in one round-trip. No auth check here; the
        # command handler itself was already authorized by virtue of reaching
        # this point (same pattern as /benchmark).
        args = getattr(context, "args", None)
        if args:
            inline_message = " ".join(args)
            event = MessageEvent(
                chat_id=chat_id,
                user_id=update.effective_user.id if update.effective_user else 0,
                text=inline_message,
                message_id=update.message.message_id if update.message else 0,
                platform=Platform.TELEGRAM,
                message_type=MessageType.TEXT,
                username=(
                    update.effective_user.username if update.effective_user else None
                ),
            )
            await self._dispatch_to_agent(event)

    async def _cmd_claude(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /claude — set per-session override to Anthropic Claude."""
        await self._apply_override(update, context, preset_name="claude")

    async def _cmd_gpt(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /gpt — set per-session override to OpenAI GPT."""
        await self._apply_override(update, context, preset_name="gpt")

    async def _cmd_gemini(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /gemini — set per-session override to Google Gemini."""
        await self._apply_override(update, context, preset_name="gemini")

    async def _cmd_xai(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /xai — set per-session override to xAI Grok."""
        await self._apply_override(update, context, preset_name="xai")

    async def _cmd_grok(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /grok — alias for /xai."""
        await self._apply_override(update, context, preset_name="xai")

    async def _cmd_deepseek(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /deepseek — set per-session override to DeepSeek."""
        await self._apply_override(update, context, preset_name="deepseek")

    async def _cmd_kimi(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /kimi — set per-session override to Kimi (Moonshot)."""
        await self._apply_override(update, context, preset_name="kimi")

    async def _cmd_glm(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /glm — set per-session override to GLM (Z.ai)."""
        await self._apply_override(update, context, preset_name="glm")

    async def _cmd_mimo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /mimo — set per-session override to MiMo (Xiaomi)."""
        await self._apply_override(update, context, preset_name="mimo")

    async def _cmd_local(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /local — clear per-session override, return to primary.

        Silent no-op if no override was set. If the command has an inline
        message, dispatches it through primary.
        """
        if update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        session_key = f"{Platform.TELEGRAM.value}:{chat_id}"

        from prometheus.gateway import commands as _cmds
        text = _cmds.cmd_local_override(
            self.agent_loop, session_key, self.model_name, self.model_provider,
        )
        await self.send(chat_id, text, parse_mode=None)

        # Inline message dispatch
        args = getattr(context, "args", None)
        if args:
            inline_message = " ".join(args)
            event = MessageEvent(
                chat_id=chat_id,
                user_id=update.effective_user.id if update.effective_user else 0,
                text=inline_message,
                message_id=update.message.message_id if update.message else 0,
                platform=Platform.TELEGRAM,
                message_type=MessageType.TEXT,
                username=(
                    update.effective_user.username if update.effective_user else None
                ),
            )
            await self._dispatch_to_agent(event)

    async def _cmd_route(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /route — show the current effective provider for this chat.

        Reports one of three states:
          - override active: "{provider}/{model}  (override)"
          - no router:       "{primary_provider}/{primary_model}  (no router)"
          - primary:         "{primary_provider}/{primary_model}  (primary)"

        Followed by a list of available override commands.
        """
        if update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        session_key = f"{Platform.TELEGRAM.value}:{chat_id}"

        from prometheus.gateway import commands as _cmds
        text = _cmds.cmd_route(
            self.agent_loop, session_key, self.model_name, self.model_provider,
        )
        await self.send(chat_id, text, parse_mode=None)

    async def _handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text or not update.effective_chat:
            return

        event = MessageEvent(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id if update.effective_user else 0,
            text=update.message.text,
            message_id=update.message.message_id,
            platform=Platform.TELEGRAM,
            message_type=MessageType.TEXT,
            username=(
                update.effective_user.username if update.effective_user else None
            ),
        )
        await self.on_message(event)

    def _truncate_for_context(self, text: str) -> str:
        """Truncate extracted document text to fit the context window budget.

        SPRINT G2: logic extracted to prometheus.gateway.media_services so
        Discord shares it; behaviour pinned in test_gateway_media_pins.py.
        """
        from prometheus.gateway.media_services import truncate_for_context

        return truncate_for_context(
            text,
            prometheus_config=self._prometheus_config,
            system_prompt=self.system_prompt,
            tool_registry=self.tool_registry,
            provider=getattr(self.agent_loop, "_provider", None),
        )

    @staticmethod
    def _chat_id_from_session_id(session_id: str) -> int | None:
        """Parse the Telegram chat id from a ``telegram:<chat_id>`` session key."""
        if session_id and session_id.startswith("telegram:"):
            try:
                return int(session_id.split(":", 1)[1])
            except (ValueError, IndexError):
                return None
        return None

    def _turn_lock_for(self, session_id: str) -> asyncio.Lock:
        """Return the per-session turn lock (M6), creating it on first use.

        Resilient to adapters built via ``__new__`` (some tests bypass
        ``__init__``): the lock map lazily initializes if absent.
        """
        locks = getattr(self, "_turn_locks", None)
        if locks is None:
            locks = {}
            self._turn_locks = locks
        lock = locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[session_id] = lock
        return lock

    async def _run_agent_turn(
        self,
        session,
        content: str,
        *,
        session_id: str,
        provenance: str = "user",
        is_trusted: bool = True,
    ) -> str:
        """Shared agent-turn core: append the turn, run the loop, persist, return text.

        Used by both Telegram inbound (``provenance="user"``, trusted) and the
        managed-task completion handler via :meth:`inject_turn`
        (``provenance="task_supervisor"``, untrusted). Does NOT send anything —
        the caller delivers the reply (Telegram inbound adds voice/queue handling;
        :meth:`inject_turn` delivers plain text to the resolved chat).

        M6: the body runs under a per-session lock so a re-engagement turn
        (inject_turn) can't interleave with a live user turn on the same
        session — both would otherwise append to ``session.messages``
        concurrently, scrambling order and duplicating ``turn_index``.
        """
        async with self._turn_lock_for(session_id):
            return await self._run_agent_turn_locked(
                session, content, session_id=session_id,
                provenance=provenance, is_trusted=is_trusted,
            )

    async def _run_agent_turn_locked(
        self,
        session,
        content: str,
        *,
        session_id: str,
        provenance: str = "user",
        is_trusted: bool = True,
    ) -> str:
        """Serialized core of :meth:`_run_agent_turn` (holds the session lock)."""
        session.add_user_message(content, provenance=provenance, is_trusted=is_trusted)
        pre_len = len(session.get_messages())
        logger.debug(
            "THREAD session=%s messages=%d provenance=%s trusted=%s",
            session_id, pre_len, provenance, is_trusted,
        )
        try:
            result = await self.agent_loop.run_async(
                system_prompt=self.system_prompt,
                messages=session.get_messages(),
                tools=self.tool_registry.list_schemas(),
                # Phase 3.5: session_id = "telegram:<chat_id>" so any /claude,
                # /gpt etc. overrides set via Phase 4 commands apply only to
                # this chat and not other Telegram chats or Slack/CLI/web.
                session_id=session_id,
                # SPRINT-2 WS1: pass the live ChatSession so the loop can
                # drain ``queued_steers`` between tool batches.
                session_state=session,
            )
            # Append assistant response (and any tool call/result pairs) to session
            session.add_result_messages(result.messages, pre_len)
            session.trim(self.session_manager.MAX_SESSION_MESSAGES)
            logger.debug(
                "THREAD after: session=%s total_messages=%d result_messages=%d",
                session_id, len(session.get_messages()), len(result.messages),
            )
            response_text = result.text or "(no response)"
            # Honest async-promise guard (Layer b): if this USER turn promises a
            # later notification but registered no qualifying task_create this
            # turn, append a delimited system-note correction and record the
            # detection for lie-rate measurement. Skips inject_turn re-engagement
            # turns (provenance != "user").
            from prometheus.engine.honesty import evaluate_and_record
            from prometheus.telemetry.tracker import get_telemetry_handle

            correction = evaluate_and_record(
                response_text,
                result.messages[pre_len:],
                provenance=provenance,
                session_id=session_id,
                telemetry=get_telemetry_handle(),
            )
            if correction:
                response_text = f"{response_text}\n\n{correction}"
            # SPRINT-TEACHER-ESCALATION Phase 3: same post-turn anchor as the
            # honesty validator. The engine fails loud in telemetry; the hook
            # itself must never break the reply path.
            try:
                response_text = await self._maybe_escalate_turn(
                    session=session,
                    session_id=session_id,
                    response_text=response_text,
                    new_messages=result.messages[pre_len:],
                    user_request=content,
                )
            except Exception:
                logger.exception(
                    "teacher escalation hook failed (local reply unaffected)")
            return response_text
        except Exception as exc:
            logger.error("Agent error for session %s: %s", session_id, exc)
            session.rollback_last()
            return f"Error: {exc}"

    def _serving_provider_name(self, session_id: str) -> str:
        """Provider name that served this session's turn: the per-session
        override (/claude etc.) when set, else the daemon primary."""
        router = getattr(self.agent_loop, "_model_router", None)
        if router is not None:
            try:
                override = router.get_override_for_session(session_id)
            except Exception:
                override = None
            if override is not None:
                cfg = getattr(override, "provider_config", None)
                if isinstance(cfg, dict) and cfg.get("provider"):
                    return str(cfg["provider"])
        return self.model_provider or ""

    async def _maybe_escalate_turn(
        self,
        *,
        session,
        session_id: str,
        response_text: str,
        new_messages: list,
        user_request: str,
    ) -> str:
        """Run teacher escalation at turn finalization (SPRINT-TEACHER-ESCALATION).

        Returns the (possibly replaced or annotated) reply text. On
        escalation, the corrective reply is persisted into the session and
        LCM tagged ``provenance="teacher_escalation"``, ``is_trusted=False``
        — the conservative machine-injected convention — and the user sees
        a visible note. Never a silent substitution.
        """
        engine = getattr(self, "escalation_engine", None)
        if engine is None:
            return response_text

        from prometheus.escalation.teacher import build_trace_from_messages

        outcome = await engine.maybe_escalate(
            session_id=session_id,
            user_request=user_request,
            tool_results=build_trace_from_messages(new_messages),
            final_reply=response_text,
            agent_mode=bool(self.tool_registry.list_schemas()),
            primary_provider=self._serving_provider_name(session_id),
        )
        if outcome is None:
            return response_text

        if outcome.status == "escalated" and outcome.corrective_reply:
            from prometheus.engine.messages import ConversationMessage, TextBlock

            start = len(session.messages)
            session.messages.append(ConversationMessage(
                role="assistant",
                content=[TextBlock(text=outcome.corrective_reply)],
                provenance="teacher_escalation",
                is_trusted=False,
            ))
            # Reuse the in-place persistence path (WS bridge pattern) so the
            # teacher turn reaches LCM with its trust tag.
            session.persist_loop_result(start)
            return f"{outcome.corrective_reply}\n\n{outcome.note}"

        if outcome.note:
            # teacher_failed: local reply stands, with the visible note.
            return f"{response_text}\n\n{outcome.note}"
        # refused_budget: logged + traced inside the engine; reply unchanged.
        return response_text

    async def _cmd_escalations(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /escalations — teacher-escalation counters + budget state."""
        if update.effective_chat is None:
            return
        from prometheus.gateway import commands as _cmds

        engine = getattr(self, "escalation_engine", None)
        text = _cmds.cmd_escalations(engine)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def inject_turn(
        self,
        session_id: str,
        content: str,
        *,
        provenance: str = "task_supervisor",
        is_trusted: bool = False,
    ) -> str:
        """Inject a non-user turn into a session, run the agent, deliver the reply.

        The shared re-engagement primitive — the managed-task completion handler
        (and, in future, cron / the orchestrator) all converge here instead of
        building parallel mechanisms. ``content`` is recorded with structured
        ``provenance`` + ``is_trusted``; the untrusted banner is a derived
        rendering applied at context-assembly. The reply is delivered to the
        Telegram chat encoded in ``session_id`` (``telegram:<chat_id>``); for a
        non-Telegram session the turn still runs and persists, but this adapter
        cannot deliver the reply.
        """
        session = self.session_manager.get_or_create(session_id)
        response_text = await self._run_agent_turn(
            session,
            content,
            session_id=session_id,
            provenance=provenance,
            is_trusted=is_trusted,
        )
        chat_id = self._chat_id_from_session_id(session_id)
        if chat_id is not None:
            try:
                await self.send(chat_id, strip_markdown(response_text))
            except Exception:
                logger.exception("inject_turn: failed to deliver reply for %s", session_id)
        else:
            logger.info(
                "inject_turn: session %s is not a Telegram chat — reply not "
                "delivered (turn ran and persisted)", session_id,
            )
        return response_text

    async def _dispatch_to_agent(self, event: MessageEvent) -> None:
        """Route a message through AgentLoop and send the response."""
        if self._app:
            try:
                await self._app.bot.send_chat_action(
                    chat_id=event.chat_id, action=ChatAction.TYPING
                )
            except Exception:
                pass  # typing indicator is best-effort

        session = self.session_manager.get_or_create(event.session_key())
        # Telegram inbound is a real human: provenance="user", trusted. Routes
        # through the same shared core as inject_turn (the re-engagement path).
        response_text = await self._run_agent_turn(
            session,
            event.text,
            session_id=event.session_key(),
            provenance="user",
            is_trusted=True,
        )

        # Voice-mode routing — prefer voice reply when the chat is in
        # voice mode (and synthesis succeeds); fall back to plain text on
        # any failure so a missing piper binary or bad model path never
        # silences the bot. See _voice_should_reply / _synthesize_voice.
        sent_voice = False
        if self._voice_should_reply(event):
            try:
                ogg_path = await self._synthesize_voice(response_text)
            except Exception as exc:
                logger.warning(
                    "Voice synthesis raised for chat %d: %s",
                    event.chat_id, exc,
                )
                ogg_path = None
            if ogg_path:
                voice_result = await self._send_voice(
                    event.chat_id, ogg_path, reply_to=event.message_id,
                )
                sent_voice = voice_result.success

        if not sent_voice:
            await self.send(
                event.chat_id,
                strip_markdown(response_text),
                reply_to=event.message_id,
            )

        # SPRINT-2 WS1: after the turn ends, drain any queued prompts and
        # fire the next one as a fresh user turn. Loop until the queue is
        # empty so back-to-back ``/queue`` calls all run. Steers are NOT
        # drained here — they live on the session's queue and the next
        # loop iteration consumes them inside run_loop.
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
            )
            await self._dispatch_to_agent(queued_event)

    # ------------------------------------------------------------------
    # Sprint 18 ANATOMY: infrastructure self-awareness
    # ------------------------------------------------------------------

    async def _cmd_anatomy(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /anatomy command — show infrastructure summary."""
        if update.effective_chat is None:
            return
        text = await cmd_anatomy()
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_doctor(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /doctor command — run diagnostic health check."""
        if update.effective_chat is None:
            return
        text = await cmd_doctor(self._prometheus_config)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_beacon(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /beacon — show web bridge / dashboard status with clickable buttons."""
        if update.effective_chat is None:
            return
        import platform as _platform
        import yaml
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        config_path = Path(__file__).resolve().parents[3] / "config" / "prometheus.yaml"
        try:
            cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        except Exception:
            cfg = {}

        web = cfg.get("web", {})
        if not web.get("enabled", False):
            await self.send(update.effective_chat.id,
                            "Beacon: not running\n\nEnable in config:\n  web:\n    enabled: true",
                            parse_mode=None)
            return

        host = _platform.node()
        api_port = web.get("api_port", 8005)
        ws_port = web.get("ws_port", 8010)

        # Resolve a routable IP for inline button URLs (Telegram rejects bare hostnames)
        try:
            import subprocess
            ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3)
            link_host = ip.stdout.strip() if ip.returncode == 0 and ip.stdout.strip() else host
        except Exception:
            import socket
            link_host = socket.gethostbyname(host)

        text = (
            f"Beacon\n"
            f"  REST API:  {link_host}:{api_port}\n"
            f"  WebSocket: ws://{link_host}:{ws_port}\n"
            f"  Dashboard: {link_host}:{web.get('dashboard_port', 3002)}"
        )
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Open API", url=f"http://{link_host}:{api_port}/docs"),
                InlineKeyboardButton("Open Dashboard", url=f"http://{link_host}:{web.get('dashboard_port', 3002)}"),
            ],
        ])
        await self._app.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text,
            reply_markup=keyboard,
        )

    async def _cmd_profile(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /profile [name] — show or switch agent profiles."""
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        args = (update.message.text or "").split(maxsplit=1)
        arg = args[1].strip() if len(args) > 1 else ""
        current = getattr(self, "_active_profile_name", "full")

        text = cmd_profile(arg=arg, current=current)

        # If switching, store the new profile name on the adapter
        if arg:
            from prometheus.config.profiles import ProfileStore
            store = ProfileStore()
            profile = store.get(arg.strip())
            if profile is not None:
                self._active_profile_name = profile.name

        await self.send(chat_id, text, parse_mode=None)

    # ------------------------------------------------------------------
    # Sprint 15b GRAFT: approval queue commands
    # ------------------------------------------------------------------

    async def _cmd_approve(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /approve {request_id} command."""
        if not update.message or not update.effective_chat:
            return
        from prometheus.gateway import commands as _cmds

        args = (update.message.text or "").split()
        request_id = args[1] if len(args) >= 2 else ""
        queue = getattr(self, "_approval_queue", None)
        text = await _cmds.cmd_approve(queue, request_id)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_deny(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /deny {request_id} command."""
        if not update.message or not update.effective_chat:
            return
        from prometheus.gateway import commands as _cmds

        args = (update.message.text or "").split()
        request_id = args[1] if len(args) >= 2 else ""
        queue = getattr(self, "_approval_queue", None)
        text = await _cmds.cmd_deny(queue, request_id)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_pending(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /pending command — list pending approval requests."""
        if not update.message or not update.effective_chat:
            return
        from prometheus.gateway import commands as _cmds

        queue = getattr(self, "_approval_queue", None)
        text = _cmds.cmd_pending(queue)
        await self.send(update.effective_chat.id, text, parse_mode=None)

    # ------------------------------------------------------------------
    # SUNRISE Session B: GEPA skill evolution
    # ------------------------------------------------------------------

    async def _cmd_gepa(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /gepa {status|run|history} command."""
        if not update.message or not update.effective_chat:
            return
        from prometheus.gateway import commands as _cmds

        chat_id = update.effective_chat.id
        args = (update.message.text or "").split()
        sub = args[1].strip().lower() if len(args) > 1 else "status"

        text = await _cmds.cmd_gepa(
            getattr(self, "_gepa_engine", None),
            getattr(self, "_approval_queue", None),
            sub,
            chat_id=chat_id,
            send=self._plain_sender(chat_id),
        )
        if text:
            await self.send(chat_id, text, parse_mode=None)

    # ------------------------------------------------------------------
    # GRAFT-SYMBIOTE Session A: /symbiote command
    # ------------------------------------------------------------------

    async def _cmd_symbiote(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /symbiote subcommands.

        Forms:
          /symbiote <problem statement>      — start a scout
          /symbiote approve <full_name>      — request harvest approval, run on /approve
          /symbiote graft                    — request graft approval, run on /approve
          /symbiote status [session_id]      — show session state
          /symbiote abort                    — abort active session
          /symbiote history [N]              — last N sessions

        SPRINT G1: full dispatch + sub-handlers live in the shared
        commands layer (commands.cmd_symbiote); this wrapper only parses
        the Telegram Update and injects the wired subsystems.
        """
        if not update.message or not update.effective_chat:
            return
        chat_id = update.effective_chat.id

        text = (update.message.text or "").strip()
        # Strip the command itself (handles "/symbiote@bot ...")
        parts = text.split(maxsplit=1)
        body = parts[1].strip() if len(parts) > 1 else ""

        from prometheus.gateway import commands as _cmds
        await _cmds.cmd_symbiote(
            self._plain_sender(chat_id),
            body,
            approval_queue=getattr(self, "_approval_queue", None),
            morph_engine=getattr(self, "_morph_engine", None),
            backup_vault=getattr(self, "_backup_vault", None),
            chat_id=chat_id,
        )

    # ------------------------------------------------------------------
    # Sprint 15 GRAFT: media handlers (additive — Hermes parity)
    # ------------------------------------------------------------------

    async def _handle_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming photo messages."""
        if not update.message or not update.message.photo or not update.effective_chat:
            return

        from prometheus.gateway.media_cache import (
            cache_image_from_bytes,
            sniff_image_extension,
        )

        # Largest resolution is the last element
        photo = update.message.photo[-1]
        try:
            file_obj = await photo.get_file()
            image_bytes = await file_obj.download_as_bytearray()
            ext = sniff_image_extension(file_obj.file_path)
            cached_path = cache_image_from_bytes(bytes(image_bytes), ext=ext)
        except Exception as exc:
            logger.error("Failed to download photo: %s", exc)
            await self.send(update.effective_chat.id, "Failed to download the photo.")
            return

        caption = update.message.caption or ""

        # Try vision analysis to describe the image
        description = await self._describe_image(cached_path)
        if description:
            user_text = f"[Image: {description}]\n{caption}".strip()
        elif caption:
            user_text = f"[The user sent a photo with caption:] {caption}"
        else:
            user_text = "[The user sent a photo]"

        event = MessageEvent(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id if update.effective_user else 0,
            text=user_text,
            message_id=update.message.message_id,
            platform=Platform.TELEGRAM,
            message_type=MessageType.PHOTO,
            username=update.effective_user.username if update.effective_user else None,
            media_urls=[cached_path],
            media_types=[f"image/{ext.lstrip('.')}"],
            caption=caption or None,
        )
        await self.on_message(event)

    async def _handle_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming voice messages — transcribe via Whisper."""
        if not update.message or not update.message.voice or not update.effective_chat:
            return

        from prometheus.gateway.media_cache import cache_audio_from_bytes

        try:
            file_obj = await update.message.voice.get_file()
            audio_bytes = await file_obj.download_as_bytearray()
            cached_path = cache_audio_from_bytes(bytes(audio_bytes), ext=".ogg")
        except Exception as exc:
            logger.error("Failed to download voice memo: %s", exc)
            await self.send(update.effective_chat.id, "Failed to download voice memo.")
            return

        # Transcribe via Whisper
        transcription = await self._transcribe_audio(cached_path)
        if not transcription:
            transcription = "[Voice memo received but transcription unavailable]"

        event = MessageEvent(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id if update.effective_user else 0,
            text=transcription,
            message_id=update.message.message_id,
            platform=Platform.TELEGRAM,
            message_type=MessageType.VOICE,
            username=update.effective_user.username if update.effective_user else None,
            media_urls=[cached_path],
            media_types=["audio/ogg"],
        )
        await self.on_message(event)

    async def _handle_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming document messages."""
        if not update.message or not update.message.document or not update.effective_chat:
            return

        from prometheus.gateway.media_cache import (
            SUPPORTED_DOCUMENT_TYPES,
            cache_document_from_bytes,
        )
        from prometheus.utils.file_extract import (
            extract_text,
            is_supported as is_extractable,
            unsupported_message,
        )

        doc = update.message.document
        original_name = doc.file_name or "unknown"
        ext = Path(original_name).suffix.lower()

        # Validate document type — accept both media_cache types and file_extract types
        if ext not in SUPPORTED_DOCUMENT_TYPES and not is_extractable(original_name):
            await self.send(
                update.effective_chat.id,
                unsupported_message(original_name),
            )
            return

        # Size check
        if doc.file_size and doc.file_size > 20 * 1024 * 1024:
            await self.send(
                update.effective_chat.id,
                "Document too large (max 20 MB).",
            )
            return

        try:
            file_obj = await doc.get_file()
            doc_bytes = await file_obj.download_as_bytearray()
            cached_path = cache_document_from_bytes(bytes(doc_bytes), original_name)
        except Exception as exc:
            logger.error("Failed to download document: %s", exc)
            await self.send(update.effective_chat.id, "Failed to download the document.")
            return

        caption = update.message.caption or ""
        mime = SUPPORTED_DOCUMENT_TYPES.get(ext, "application/octet-stream")

        # Try to extract text for inline injection (shared extractor handles
        # plain text, PDF, DOCX, XLSX — same logic as Beacon web uploads)
        extracted = extract_text(cached_path)
        if extracted:
            # Truncate if extracted text would overflow the context window
            extracted = self._truncate_for_context(extracted)
            user_text = f"[Content of {original_name}]:\n{extracted}"
            if caption:
                user_text = f"{caption}\n\n{user_text}"
        else:
            user_text = caption or f"[The user sent a document: {original_name}]"

        event = MessageEvent(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id if update.effective_user else 0,
            text=user_text,
            message_id=update.message.message_id,
            platform=Platform.TELEGRAM,
            message_type=MessageType.DOCUMENT,
            username=update.effective_user.username if update.effective_user else None,
            media_urls=[cached_path],
            media_types=[mime],
            caption=caption or None,
        )
        await self.on_message(event)

    async def _handle_sticker(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming sticker messages."""
        if not update.message or not update.message.sticker or not update.effective_chat:
            return

        from prometheus.gateway.sticker_cache import (
            build_animated_sticker_injection,
            build_sticker_injection,
            cache_sticker_description,
            get_cached_description,
        )

        sticker = update.message.sticker
        emoji = sticker.emoji or ""
        set_name = sticker.set_name or ""
        file_unique_id = sticker.file_unique_id

        # Animated/video stickers can't be visually analyzed
        if sticker.is_animated or sticker.is_video:
            user_text = build_animated_sticker_injection(emoji)
        else:
            # Check cache first
            cached = get_cached_description(file_unique_id)
            if cached:
                user_text = build_sticker_injection(
                    cached["description"], cached.get("emoji", emoji), cached.get("set_name", set_name)
                )
            else:
                # Download and analyze via vision
                description = None
                try:
                    from prometheus.gateway.media_cache import cache_image_from_bytes
                    file_obj = await sticker.get_file()
                    sticker_bytes = await file_obj.download_as_bytearray()
                    cached_path = cache_image_from_bytes(bytes(sticker_bytes), ext=".webp")
                    description = await self._describe_image(cached_path)
                except Exception as exc:
                    logger.warning("Failed to analyze sticker: %s", exc)

                if description:
                    cache_sticker_description(file_unique_id, description, emoji, set_name)
                    user_text = build_sticker_injection(description, emoji, set_name)
                else:
                    user_text = build_animated_sticker_injection(emoji)

        event = MessageEvent(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id if update.effective_user else 0,
            text=user_text,
            message_id=update.message.message_id,
            platform=Platform.TELEGRAM,
            message_type=MessageType.STICKER,
            username=update.effective_user.username if update.effective_user else None,
        )
        await self.on_message(event)

    # ------------------------------------------------------------------
    # Media helper methods
    # ------------------------------------------------------------------

    async def _describe_image(self, image_path: str) -> str | None:
        """Run image through VisionTool for a text description. Returns None on failure.

        SPRINT G2: delegates to the shared media_services module (same
        service Discord uses); behaviour pinned in test_gateway_media_pins.py.
        """
        from prometheus.gateway.media_services import describe_image

        return await describe_image(image_path, provider=self._get_provider())

    async def _transcribe_audio(self, audio_path: str) -> str | None:
        """Run audio through WhisperSTT for transcription. Returns None on failure.

        SPRINT G2: delegates to the shared media_services module (same
        service Discord uses); behaviour pinned in test_gateway_media_pins.py.
        """
        from prometheus.gateway.media_services import transcribe_audio

        return await transcribe_audio(audio_path)

    def _get_provider(self):
        """Return the model provider from the agent loop (for vision analysis)."""
        try:
            return self.agent_loop._provider
        except AttributeError:
            return None

    # ------------------------------------------------------------------
    # Voice mode (TTS-out for Telegram) — added 2026-05
    # ------------------------------------------------------------------
    #
    # Pipeline: agent text reply → strip markdown for speech → piper TTS
    # → WAV → ffmpeg opus encode → ogg → bot.send_voice().
    #
    # Routing decision per chat:
    #   "auto" (default) — mirror input modality (voice in → voice out)
    #   "on"             — always reply with voice
    #   "off"            — always reply with text (disables voice replies)
    #
    # Per-chat mode is persisted to ~/.prometheus/voice_modes.json so
    # /voice toggles survive daemon restarts. Falls through to the
    # gateway.voice.default_mode config value when no override is set.
    #
    # Failure mode: if TTS synthesis or ffmpeg encoding fails (binary
    # missing, model path bad, etc.) the method returns None and
    # _dispatch_to_agent falls back to a normal text reply. A voice
    # mode is opt-in: an unconfigured deployment never tries to speak.

    def _voice_config(self) -> dict[str, Any]:
        """Return the gateway.voice config block (or empty dict).

        Defensive: tests construct ``TelegramAdapter.__new__(TelegramAdapter)``
        without running ``__init__``, so ``_prometheus_config`` may be
        missing entirely. Treat that as "no voice config" rather than
        raising — voice mode is opt-in and an unconfigured adapter
        should still dispatch text replies normally.
        """
        cfg = getattr(self, "_prometheus_config", None)
        if not isinstance(cfg, dict):
            return {}
        gw = cfg.get("gateway", {})
        if isinstance(gw, dict):
            voice = gw.get("voice", {})
            if isinstance(voice, dict):
                return voice
        return {}

    # SPRINT G1: voice-mode persistence moved to the shared commands layer
    # (platform-independent, keyed by str chat key). These thin delegates
    # remain because _voice_should_reply and existing tests use them.

    def _voice_modes_path(self) -> str:
        """Path to the per-chat voice mode override JSON file."""
        from prometheus.gateway import commands as _cmds
        return _cmds._voice_modes_path()

    def _load_voice_modes(self) -> dict[str, str]:
        """Read per-chat voice mode overrides. Returns {chat_id_str: mode}."""
        from prometheus.gateway import commands as _cmds
        return _cmds.load_voice_modes()

    def _save_voice_modes(self, modes: dict[str, str]) -> None:
        """Persist per-chat voice mode overrides."""
        from prometheus.gateway import commands as _cmds
        _cmds.save_voice_modes(modes)

    def _get_voice_mode(self, chat_id: int) -> str:
        """Return the effective voice mode for a chat ('auto' | 'on' | 'off')."""
        from prometheus.gateway import commands as _cmds
        default = str(self._voice_config().get("default_mode", "auto"))
        return _cmds.get_voice_mode(str(chat_id), default)

    def _set_voice_mode(self, chat_id: int, mode: str) -> None:
        """Persist a per-chat voice mode override."""
        from prometheus.gateway import commands as _cmds
        _cmds.set_voice_mode(str(chat_id), mode)

    def _voice_should_reply(self, event: MessageEvent) -> bool:
        """Decide whether the reply to *event* should be a voice message."""
        mode = self._get_voice_mode(event.chat_id)
        if mode == "off":
            return False
        if mode == "on":
            return True
        # auto: mirror input modality
        return event.message_type == MessageType.VOICE

    def _strip_text_for_tts(self, text: str) -> str:
        """Strip markdown/symbols that TTS would mispronounce.

        More aggressive than strip_markdown — also removes code blocks
        entirely (TTS reading code is unintelligible), URLs, and bullet
        glyphs. Caller truncates to gateway.voice.max_chars.
        """
        # Drop fenced code blocks wholesale
        text = re.sub(r"```[\s\S]*?```", " [code omitted] ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Headings, bold, italics
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # Bullets, list markers
        text = re.sub(r"^\s*[\*\-•]\s+", "", text, flags=re.MULTILINE)
        # URLs — speak as "link" rather than reading the whole URL aloud
        text = re.sub(r"https?://\S+", "link", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def _synthesize_voice(self, text: str) -> str | None:
        """Synthesize *text* to an opus-encoded .ogg file for Telegram.

        Returns the absolute path on success, None on any failure
        (caller should fall back to text). Uses piper if configured,
        else espeak. Output is an opus-encoded ogg suitable for
        bot.send_voice() — Telegram voice messages require this format
        for the in-app waveform/playback UI.
        """
        import tempfile

        cfg = self._voice_config()
        max_chars = int(cfg.get("max_chars", 800))
        clean = self._strip_text_for_tts(text)
        if not clean:
            return None
        if len(clean) > max_chars:
            clean = clean[: max_chars - 1].rsplit(" ", 1)[0] + "…"

        engine = (cfg.get("engine") or "piper").lower()
        wav_path = tempfile.mktemp(suffix=".wav")

        if engine == "piper":
            ok = await self._run_piper(clean, wav_path, cfg.get("model_path"))
        elif engine == "espeak":
            ok = await self._run_espeak(clean, wav_path)
        else:
            logger.warning("Unknown TTS engine: %s", engine)
            return None

        if not ok:
            Path(wav_path).unlink(missing_ok=True)
            return None

        bitrate = str(cfg.get("opus_bitrate", "32k"))
        ogg_path = await self._wav_to_opus_ogg(wav_path, bitrate)
        Path(wav_path).unlink(missing_ok=True)
        return ogg_path

    async def _run_piper(
        self, text: str, out_wav: str, model_path: str | None
    ) -> bool:
        """Invoke the piper TTS CLI. Returns True on success."""
        import shutil
        piper_bin = shutil.which("piper")
        if piper_bin is None:
            logger.debug("piper binary not found on PATH")
            return False
        if not model_path:
            logger.debug("No piper model_path configured in gateway.voice")
            return False
        model = Path(model_path).expanduser()
        if not model.is_file():
            logger.warning("Piper model not found: %s", model)
            return False

        proc = await asyncio.create_subprocess_exec(
            piper_bin, "-m", str(model), "-f", out_wav,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(text.encode("utf-8")), timeout=60
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning("piper timed out after 60s")
            return False
        if proc.returncode != 0:
            logger.warning(
                "piper failed (rc=%s): %s",
                proc.returncode, stderr.decode(errors="replace")[:300],
            )
            return False
        return Path(out_wav).is_file() and Path(out_wav).stat().st_size > 0

    async def _run_espeak(self, text: str, out_wav: str) -> bool:
        """Invoke espeak-ng (or espeak) as a piper fallback."""
        import shutil
        bin_path = shutil.which("espeak-ng") or shutil.which("espeak")
        if bin_path is None:
            return False
        proc = await asyncio.create_subprocess_exec(
            bin_path, "-w", out_wav,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(text.encode("utf-8")), timeout=30
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False
        if proc.returncode != 0:
            logger.warning(
                "espeak failed (rc=%s): %s",
                proc.returncode, stderr.decode(errors="replace")[:300],
            )
            return False
        return Path(out_wav).is_file() and Path(out_wav).stat().st_size > 0

    async def _wav_to_opus_ogg(self, wav_path: str, bitrate: str) -> str | None:
        """Re-encode a WAV to opus-in-ogg via ffmpeg. Returns the .ogg path or None."""
        import shutil
        import tempfile
        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not on PATH — cannot encode voice reply")
            return None
        ogg_path = tempfile.mktemp(suffix=".ogg")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", wav_path,
            "-c:a", "libopus", "-b:a", bitrate,
            "-application", "voip",
            ogg_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return None
        if proc.returncode != 0:
            logger.warning(
                "ffmpeg opus encode failed (rc=%s): %s",
                proc.returncode, stderr.decode(errors="replace")[:300],
            )
            Path(ogg_path).unlink(missing_ok=True)
            return None
        if not Path(ogg_path).is_file() or Path(ogg_path).stat().st_size == 0:
            return None
        return ogg_path

    async def _send_voice(
        self,
        chat_id: int,
        ogg_path: str,
        *,
        reply_to: int | None = None,
        caption: str | None = None,
    ) -> SendResult:
        """Upload an opus .ogg as a Telegram voice message."""
        if not self._app:
            return SendResult(success=False, error="Bot not initialized")
        try:
            with open(ogg_path, "rb") as f:
                msg = await self._app.bot.send_voice(
                    chat_id=chat_id,
                    voice=f,
                    caption=caption,
                    reply_to_message_id=reply_to,
                )
            return SendResult(success=True, message_id=msg.message_id)
        except Exception as exc:
            logger.error("Failed to send voice to chat %d: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))
        finally:
            # Best-effort cleanup; cache files are short-lived
            Path(ogg_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # WEAVE Session B: /audit command
    # ------------------------------------------------------------------

    async def _cmd_audit(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /audit subcommands.

        Forms:
          /audit                  — show summary of the most recent audit
          /audit run              — start a full web capability audit
          /audit <category>       — start an audit for a single category
                                    (search|fetch|youtube|download|research|graceful|railway)

        SPRINT G1: dispatch + show-last + kick-off live in the shared
        commands layer (commands.cmd_audit).
        """
        if not update.message or not update.effective_chat:
            return
        chat_id = update.effective_chat.id

        text = (update.message.text or "").strip()
        parts = text.split(maxsplit=1)
        body = parts[1].strip().lower() if len(parts) > 1 else ""

        from prometheus.gateway import commands as _cmds
        await _cmds.cmd_audit(self._plain_sender(chat_id), body)

    # ------------------------------------------------------------------
    # WEAVE-PRESS: /press command — Printing Press CLI library
    # ------------------------------------------------------------------

    async def _cmd_press(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /press subcommands.

        Forms:
          /press                   — show usage
          /press list [category]   — list available CLIs (optionally filtered)
          /press search <query>    — fuzzy search by name / description
          /press install <name>    — request approval, then go install + skill copy
          /press installed         — list CLIs whose binary is on PATH or in ~/go/bin
          /press update            — git pull the library clone

        SPRINT G1: dispatch + sub-handlers live in the shared commands
        layer (commands.cmd_press).
        """
        if not update.message or not update.effective_chat:
            return
        chat_id = update.effective_chat.id

        text = (update.message.text or "").strip()
        parts = text.split(maxsplit=1)
        body = parts[1].strip() if len(parts) > 1 else ""

        from prometheus.gateway import commands as _cmds
        await _cmds.cmd_press(
            self._plain_sender(chat_id),
            getattr(self, "_printing_press", None),
            body,
            approval_queue=getattr(self, "_approval_queue", None),
            chat_id=chat_id,
        )
