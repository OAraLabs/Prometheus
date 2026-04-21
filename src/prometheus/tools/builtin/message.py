# Provenance: NousResearch/hermes-agent (https://github.com/NousResearch/hermes-agent)
#             openclaw/openclaw (https://github.com/openclaw/openclaw)
# Original: tools/send_message_tool.py + src/agents/tools/message-tool.ts
# License: MIT
# Modified: Rewritten as Prometheus BaseTool; simplified to webhook/bot-token dispatch

"""Send messages to Discord, Slack, or generic webhook endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

import os


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
        try:
            if platform == MessagePlatform.discord:
                return await _send_discord(arguments)
            elif platform == MessagePlatform.slack:
                return await _send_slack(arguments)
            elif platform == MessagePlatform.telegram:
                return await _send_telegram(arguments)
            elif platform == MessagePlatform.webhook:
                return await _send_webhook(arguments)
            else:
                return ToolResult(
                    output=f"Unsupported platform: {platform}", is_error=True
                )
        except httpx.HTTPError as exc:
            return ToolResult(output=f"message send failed: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"message error: {exc}", is_error=True)


async def _send_discord(args: MessageInput) -> ToolResult:
    webhook_url = args.recipient or os.environ.get("DISCORD_WEBHOOK_URL", "")
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


async def _send_webhook(args: MessageInput) -> ToolResult:
    url = args.recipient
    if not url:
        return ToolResult(
            output="Webhook platform requires a URL in the recipient field.",
            is_error=True,
        )
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json={"text": args.content})
        resp.raise_for_status()
    return ToolResult(output=f"Message sent to webhook {url}.")
