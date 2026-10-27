"""Which enabled chat gateways have no owner, in words an operator can act on.

The owner is the only person on a chat gateway who may approve tool calls, change what the agent may do, or teach it;
everyone else in an allowed chat may chat. ``oara doctor`` and the daemon's startup warning both read this, so they
cannot disagree about which gateway needs a setting, or which setting it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OwnerGap:
    gateway: str   # "Telegram", "Slack", "Discord"
    key: str       # the config key that names the owner
    message: str   # what is true now, plainly
    fix: str       # what to do, naming the key


def _section(gw: dict, name: str) -> dict:
    nested = gw.get(name)
    return nested if isinstance(nested, dict) else {}


def ownerless_gateways(config: dict[str, Any] | None) -> list[OwnerGap]:
    """Every enabled gateway with no owner named, with the key to set. Mirrors the daemon's enabled rules."""
    from prometheus.config.shipped_defaults import (
        resolve_allowed_chat_ids,
        resolve_owner_user_ids,
        resolve_telegram_enabled,
        resolve_telegram_owner_ids,
    )

    gw = (config or {}).get("gateway") or {}
    gaps: list[OwnerGap] = []

    if resolve_telegram_enabled(gw) and resolve_allowed_chat_ids(gw) and not resolve_telegram_owner_ids(gw):
        gaps.append(OwnerGap(
            gateway="Telegram", key="gateway.owner_user_ids",
            message="allowed_chat_ids lists only group chats and no owner is named, so nobody on Telegram can "
                    "approve tool calls or teach Prometheus",
            fix="Set gateway.owner_user_ids to your Telegram user ID (message @userinfobot to find it).",
        ))

    slack = _section(gw, "slack")
    if bool(gw.get("slack_enabled", False) or slack.get("enabled", False)) and not resolve_owner_user_ids(slack):
        gaps.append(OwnerGap(
            gateway="Slack", key="gateway.slack.owner_user_ids",
            message="no owner is named, so nobody on Slack can approve tool calls or teach Prometheus, and Slack "
                    "messages don't add to its memory",
            fix="Set gateway.slack.owner_user_ids to your Slack member ID (it starts with U; find it in your Slack "
                "profile under ⋮ → Copy member ID).",
        ))

    discord = _section(gw, "discord")
    if bool(discord.get("enabled", False)) and not resolve_owner_user_ids(discord):
        gaps.append(OwnerGap(
            gateway="Discord", key="gateway.discord.owner_user_ids",
            message="no owner is named, so once connected the owner of the bot's Discord application is treated "
                    "as the owner",
            fix="To name yourself, set gateway.discord.owner_user_ids to your Discord user ID (Developer Mode on, "
                "then right-click your name → Copy User ID).",
        ))
    return gaps
