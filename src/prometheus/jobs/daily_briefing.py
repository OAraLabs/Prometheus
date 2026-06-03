#!/usr/bin/env python3
"""Deterministic daily news briefing job.

A fixed pipeline (NO agent loop):

  1. web_search  -> global headlines + key market news
  2. ONE provider completion -> a concise briefing
  3. message     -> Telegram delivery

Run::

    python3 -m prometheus.jobs.daily_briefing

Fail-loud contract: any stage failure raises :class:`BriefingError` (or lets the
underlying exception propagate), and ``main()`` returns a non-zero exit code.
There are no silent ``except`` blocks — the builtin tools soft-fail by returning
``ToolResult(is_error=True)``, so this job inspects ``is_error`` and raises.

Credential resolution (see PHASE-0 notes on branch fix/cron-briefing-tz):
  * Bot token: ``gateway.telegram_token`` or ``PROMETHEUS_TELEGRAM_TOKEN`` (the
    daemon's source). It is bridged into ``TELEGRAM_BOT_TOKEN`` because
    ``MessageTool`` reads the token only from that env var.
  * Chat id: ``gateway.briefing_chat_id`` -> ``gateway.allowed_chat_ids[0]`` (the
    daemon's source) -> ``TELEGRAM_CHAT_ID``. Fails loud if none is configured.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
)
from prometheus.providers.registry import ProviderRegistry
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.message import (
    MessageInput,
    MessagePlatform,
    MessageTool,
)
from prometheus.tools.builtin.web_search import WebSearchInput, WebSearchTool

logger = logging.getLogger("prometheus.jobs.daily_briefing")

SEARCH_QUERIES = (
    "top world news headlines today",
    "stock market today S&P 500 Nasdaq Dow Jones news",
)
MAX_BRIEFING_CHARS = 1500
SYSTEM_PROMPT = (
    "You are a concise news editor. From the raw web-search results provided, "
    "write a short daily briefing: 4-6 bullets of global headlines, then 2-3 "
    "bullets of key stock-market news. Plain text only, no preamble and no "
    "markdown headers. Be factual and brief."
)


class BriefingError(RuntimeError):
    """A briefing stage failed. Propagates to a non-zero process exit."""


# ---------------------------------------------------------------------------
# Config / provider / credential resolution (mirrors daemon.py)
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    """Load prometheus.yaml with env overrides (the CLI's canonical loader)."""
    from prometheus.__main__ import load_config

    return load_config()


def _build_provider(config: dict[str, Any]):
    """Build the REAL provider exactly as daemon.py does (NOT StubProvider)."""
    return ProviderRegistry.create(config.get("model", {}))


async def _resolve_model(provider: Any, config: dict[str, Any]) -> str:
    """Resolve the model name as daemon.py does: detect_loaded_model -> config."""
    config_model = config.get("model", {}).get("model", "qwen3.5-32b")
    if hasattr(provider, "detect_loaded_model"):
        detected = await provider.detect_loaded_model()
        if detected:
            return detected
    return config_model


def resolve_telegram_token(config: dict[str, Any]) -> str:
    """Bot token, the daemon's way. Fails loud if unset."""
    gateway = config.get("gateway", {}) or {}
    token = gateway.get("telegram_token") or os.environ.get(
        "PROMETHEUS_TELEGRAM_TOKEN", ""
    )
    if not token:
        raise BriefingError(
            "No Telegram bot token: set gateway.telegram_token or "
            "PROMETHEUS_TELEGRAM_TOKEN."
        )
    return token


def resolve_chat_id(config: dict[str, Any]) -> str:
    """Destination chat id with daemon-compatible precedence. Fails loud if none."""
    gateway = config.get("gateway", {}) or {}
    explicit = gateway.get("briefing_chat_id")
    if explicit:
        return str(explicit)
    allowed = gateway.get("allowed_chat_ids") or []
    if allowed:
        return str(allowed[0])
    env_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if env_chat:
        return env_chat
    raise BriefingError(
        "No Telegram chat target: set gateway.briefing_chat_id, add an id to "
        "gateway.allowed_chat_ids, or set TELEGRAM_CHAT_ID."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def _summarize(provider: Any, model: str, search_text: str) -> str:
    """Run ONE provider completion (no agent loop) and return the briefing text."""
    request = ApiMessageRequest(
        model=model,
        messages=[
            ConversationMessage.from_user_text(
                "Write today's news briefing from these search results:\n\n"
                + search_text
            )
        ],
        system_prompt=SYSTEM_PROMPT,
        max_tokens=600,
    )
    deltas: list[str] = []
    final_text = ""
    async for event in provider.stream_message(request):
        if isinstance(event, ApiTextDeltaEvent):
            deltas.append(event.text)
        elif isinstance(event, ApiMessageCompleteEvent):
            final_text = event.message.text
    text = ("".join(deltas) or final_text).strip()
    return text[:MAX_BRIEFING_CHARS]


async def run_briefing(
    *,
    provider: Any,
    model: str,
    chat_id: str,
    web_search: WebSearchTool | None = None,
    message: MessageTool | None = None,
) -> str:
    """Search -> summarize -> send. Returns the delivered text; fails loud."""
    web_search = web_search or WebSearchTool()
    message = message or MessageTool()
    ctx = ToolExecutionContext(cwd=Path.cwd())

    # 1. Search — web_search soft-fails (is_error), so raise on it ourselves.
    chunks: list[str] = []
    for query in SEARCH_QUERIES:
        result = await web_search.execute(
            WebSearchInput(query=query, max_results=5), ctx
        )
        if result.is_error:
            raise BriefingError(
                f"web_search failed for {query!r}: {result.output}"
            )
        chunks.append(result.output)

    # 2. One completion.
    briefing = await _summarize(provider, model, "\n\n".join(chunks))
    if not briefing:
        raise BriefingError("provider returned an empty briefing")

    # 3. Send.
    sent = await message.execute(
        MessageInput(
            platform=MessagePlatform.telegram,
            content=briefing,
            recipient=chat_id,
        ),
        ctx,
    )
    if sent.is_error:
        raise BriefingError(f"Telegram send failed: {sent.output}")

    logger.info("Briefing delivered to chat %s (%d chars)", chat_id, len(briefing))
    return briefing


async def _main_async() -> None:
    config = _load_config()
    provider = _build_provider(config)
    model = await _resolve_model(provider, config)
    token = resolve_telegram_token(config)
    chat_id = resolve_chat_id(config)
    # Bridge: MessageTool's Telegram path reads the token only from this env var.
    os.environ["TELEGRAM_BOT_TOKEN"] = token
    logger.info("Bridged Telegram token into TELEGRAM_BOT_TOKEN for MessageTool")
    await run_briefing(provider=provider, model=model, chat_id=chat_id)


def main() -> int:
    """CLI entry point. Returns 0 on success, non-zero on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(_main_async())
    except BriefingError as exc:
        logger.error("daily briefing failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
