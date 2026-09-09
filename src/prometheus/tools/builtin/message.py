# Provenance: NousResearch/hermes-agent (https://github.com/NousResearch/hermes-agent)
#             openclaw/openclaw (https://github.com/openclaw/openclaw)
# Original: tools/send_message_tool.py + src/agents/tools/message-tool.ts
# License: MIT
# Modified: Rewritten as Prometheus BaseTool; simplified to webhook/bot-token dispatch

"""Send messages to Discord, Slack, or generic webhook endpoints."""

from __future__ import annotations

from enum import Enum

import httpx
from pydantic import BaseModel, Field

from prometheus.permissions.checker import ORIGIN_SYSTEM, origin_from_session_id
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

import os


# ---------------------------------------------------------------------------
# Destination perimeter
# ---------------------------------------------------------------------------
#
# `message` is the only tool that POSTs model-authored CONTENT to a model-chosen
# URL. That makes it an exfiltration primitive, not merely an SSRF one:
# web_fetch READS from an address, this one WRITES to it.
#
# The rule constrains WHERE data may flow, never how much the agent may attempt
# — no caps, no approval prompts, no gate on the model's reasoning:
#
#   user origin   -> the model may choose the destination. A human asked for
#                    this turn and sees the result, so "post this to my ntfy
#                    endpoint" and "post to my own box" both keep working.
#   system origin -> the destination must be OPERATOR-supplied, via env var.
#                    Cron, Sentinel, GEPA and managed tasks have nobody in the
#                    loop, so a URL the model picked *after reading a fetched
#                    page* is indistinguishable from one that page chose for it.
#
# Deliberately NOT an address-class (SSRF) check. An attacker's collector sits
# at a perfectly public address, so `url_guard.is_blocked_address` cannot see
# the threat this tool poses; and applying it here would break posting to the
# operator's own local services, which is legitimate at user origin.
#
# Only the two arbitrary-URL platforms are governed. Slack and Telegram POST to
# hardcoded API hosts (slack.com, api.telegram.org) and treat `recipient` as a
# channel/chat id, so a steered value stays inside the operator's own workspace,
# reachable only with the operator's own bot token. Different blast radius,
# excluded on purpose rather than by oversight.

#: Operator-owned destination for the generic `webhook` platform. Discord and
#: Slack already had one; this platform had none, which would have left it dead
#: at system origin rather than merely constrained.
WEBHOOK_URL_ENV = "PROMETHEUS_WEBHOOK_URL"


def resolve_destination(
    model_supplied: str | None,
    env_var: str,
    origin: str,
) -> tuple[str, str | None]:
    """Resolve the URL to POST to, under the origin rule.

    Returns ``(url, refusal)``. A non-None *refusal* means the send must not
    proceed and carries the reason shown to the model. ``("", None)`` means
    nothing was supplied at all, leaving the caller's own "missing destination"
    message intact — a refusal and an absence are not the same answer.
    """
    configured = os.environ.get(env_var, "").strip()
    if origin != ORIGIN_SYSTEM:
        # User origin: model's choice wins, falling back to the operator's.
        # This is the pre-existing precedence, preserved exactly.
        return (model_supplied or configured), None
    if configured:
        return configured, None
    if model_supplied:
        return "", (
            f"Refused: a background session may only post to the destination "
            f"configured in {env_var}, not to a URL chosen at runtime. Set "
            f"{env_var} to enable this, or run the send from a user session."
        )
    return "", None


class MessagePlatform(str, Enum):
    discord = "discord"
    slack = "slack"
    telegram = "telegram"
    webhook = "webhook"


class MessageInput(BaseModel):
    """Arguments for sending a message."""

    platform: MessagePlatform = Field(description="Target platform")
    content: str = Field(description="Message text to send")
    recipient: str | None = Field(
        default=None,
        description="Channel ID, chat ID, or webhook URL depending on platform",
    )


class MessageTool(BaseTool):
    """Send a message to Discord, Slack, Telegram, or a generic webhook."""

    name = "message"
    description = (
        "Send a text message to Discord (webhook), Slack (webhook/bot), "
        "Telegram (bot API), or a generic webhook endpoint."
    )
    input_model = MessageInput

    async def execute(
        self, arguments: MessageInput, context: ToolExecutionContext
    ) -> ToolResult:
        platform = arguments.platform
        # The session id comes from the agent loop's tool metadata, which is
        # trusted context — the loop sets it from LoopContext, never from tool
        # arguments, so an injected argument cannot forge a user origin.
        # Contexts built without metadata (jobs, cron) classify as system,
        # which is the fail-closed direction.
        origin = origin_from_session_id((context.metadata or {}).get("session_id"))
        try:
            if platform == MessagePlatform.discord:
                return await _send_discord(arguments, origin)
            elif platform == MessagePlatform.slack:
                return await _send_slack(arguments)
            elif platform == MessagePlatform.telegram:
                return await _send_telegram(arguments)
            elif platform == MessagePlatform.webhook:
                return await _send_webhook(arguments, origin)
            else:
                return ToolResult(
                    output=f"Unsupported platform: {platform}", is_error=True
                )
        except httpx.HTTPError as exc:
            return ToolResult(output=f"message send failed: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"message error: {exc}", is_error=True)


async def _send_discord(args: MessageInput, origin: str) -> ToolResult:
    webhook_url, refusal = resolve_destination(
        args.recipient, "DISCORD_WEBHOOK_URL", origin
    )
    if refusal:
        return ToolResult(output=refusal, is_error=True)
    if not webhook_url:
        return ToolResult(
            output="Discord requires a webhook URL via recipient or DISCORD_WEBHOOK_URL env var.",
            is_error=True,
        )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(webhook_url, json={"content": args.content})
        resp.raise_for_status()
    return ToolResult(output="Message sent to Discord.")


async def _send_slack(args: MessageInput) -> ToolResult:
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")

    if bot_token and args.recipient:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {bot_token}"},
                json={"channel": args.recipient, "text": args.content},
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                return ToolResult(
                    output=f"Slack API error: {data.get('error', 'unknown')}",
                    is_error=True,
                )
        return ToolResult(output=f"Message sent to Slack channel {args.recipient}.")

    if webhook_url:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(webhook_url, json={"text": args.content})
            resp.raise_for_status()
        return ToolResult(output="Message sent to Slack webhook.")

    return ToolResult(
        output="Slack requires SLACK_BOT_TOKEN + recipient channel, or SLACK_WEBHOOK_URL.",
        is_error=True,
    )


async def _send_telegram(args: MessageInput) -> ToolResult:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = args.recipient or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return ToolResult(
            output="Telegram requires TELEGRAM_BOT_TOKEN and a chat_id (via recipient or TELEGRAM_CHAT_ID).",
            is_error=True,
        )
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={"chat_id": chat_id, "text": args.content})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            return ToolResult(
                output=f"Telegram API error: {data.get('description', 'unknown')}",
                is_error=True,
            )
    return ToolResult(output=f"Message sent to Telegram chat {chat_id}.")


async def _send_webhook(args: MessageInput, origin: str) -> ToolResult:
    url, refusal = resolve_destination(args.recipient, WEBHOOK_URL_ENV, origin)
    if refusal:
        return ToolResult(output=refusal, is_error=True)
    if not url:
        return ToolResult(
            output=(
                "Webhook platform requires a URL in the recipient field, or "
                f"{WEBHOOK_URL_ENV} set for background sessions."
            ),
            is_error=True,
        )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={"text": args.content})
        resp.raise_for_status()
    return ToolResult(output=f"Message sent to webhook {url}.")
