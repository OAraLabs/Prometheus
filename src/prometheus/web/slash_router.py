"""Slash-command routing for the web (Beacon) chat surface.

Web parity for Telegram's CommandHandler: in the chat surfaces a leading-slash
message is a *command*, not a conversation turn. This module decides, for a
given message, whether it's a command and what to do with it:

  * a side-effect-free command (``/help``, ``/status``, ``/wiki``, ``/note`` …),
    or a session-mutating command that enqueues onto the live run (``/steer``,
    ``/queue``, ``/unqueue``, ``/clearsteers``), runs through the shared tables
    in ``gateway.commands`` and its text is broadcast back — the message is NOT
    added to the session as a turn or run through the agent;
  * a command that mutates other live daemon state still bound to the Telegram
    gateway (``/route``, ``/approve``, ``/benchmark`` …) is not reachable here
    yet and gets an explicit "not on web" reply — never a silent drop into the
    agent;
  * a genuinely unknown ``/token`` falls through to the agent, matching Telegram
    (its CommandHandler only fires for registered names; everything else reaches
    the ordinary message handler).

The router has no transport or daemon dependencies — it's pure logic over a
CommandContext, so it tests in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from prometheus.gateway.commands import (
    CommandContext,
    is_formatter_command,
    is_session_command,
    run_formatter_command,
    run_session_command,
)

# Registered in the Telegram gateway but NOT reachable yet: they mutate live
# daemon state (model routing, the approval queue, benchmark/press/gepa/symbiote/
# audit runs) bound to the TelegramGateway instance. Mirror of telegram.py's
# CommandHandler registrations minus the shared formatter + session tables. Kept
# explicit (not derived) so the web surface gives a precise boundary message;
# the remaining "full parity" follow-up replaces these with real handlers once
# the bridge gains the needed daemon services.
#
# NOTE: /model is a formatter here (read-only model report, matching Slack), not
# native — even though Telegram's /model aliases the mutating /route. The
# steering group (/steer /queue /unqueue /clearsteers) IS wired — see the
# session-command table in gateway.commands.
WEB_NATIVE_ONLY: frozenset[str] = frozenset({
    "start", "clear", "reset",
    "route",
    "benchmark", "voice", "tools", "pairs",
    "approve", "deny", "pending",
    "gepa", "symbiote", "audit", "press",
    "claude", "gpt", "gemini", "xai", "grok", "local",
    # CLOUD EXPANSION (2026-07): mirror of the new Telegram override commands
    "deepseek", "kimi", "glm", "mimo",
    "escalations",
})

_NAME_RE = re.compile(r"[a-z0-9_-]+")


@dataclass
class SlashOutcome:
    """Result of routing a message.

    ``handled=True``  → ``reply`` holds text to broadcast; do NOT run the agent.
    ``handled=False`` → not a (web-handled) command; run the normal agent path.
    """

    handled: bool
    reply: str | None = None


def parse_slash(content: str) -> tuple[str, str] | None:
    """Split a leading-slash message into ``(command, args)``.

    Returns ``None`` when ``content`` is not a slash command — including bare
    ``/``, ``/ x`` (space right after the slash), and tokens that aren't a clean
    command name like ``/home/will/file`` or ``/3/4`` (so file paths and
    fractions fall through to the agent rather than being eaten as commands).
    """
    text = content.strip()
    if not text.startswith("/"):
        return None
    body = text[1:]
    if not body or body[0].isspace():
        return None
    head, _, rest = body.partition(" ")
    name = head.lower()
    if not _NAME_RE.fullmatch(name):
        return None
    return name, rest.strip()


async def route_slash(content: str, ctx: CommandContext) -> SlashOutcome:
    """Route a chat message; see the module docstring for the three outcomes."""
    parsed = parse_slash(content)
    if parsed is None:
        return SlashOutcome(handled=False)
    name, args = parsed

    if is_formatter_command(name):
        reply = await run_formatter_command(name, args, ctx)
        # is_formatter_command already excluded the only None case, so reply is
        # a string; the `or ""` guards a command that legitimately returns "".
        return SlashOutcome(handled=True, reply=reply or "")

    if is_session_command(name):
        reply = await run_session_command(name, args, ctx)
        return SlashOutcome(handled=True, reply=reply or "")

    if name in WEB_NATIVE_ONLY:
        return SlashOutcome(
            handled=True,
            reply=(
                f"/{name} isn't available on the web chat surface yet — it runs "
                f"on Telegram. (Web parity for mutating commands is tracked as a "
                f"follow-up.)"
            ),
        )

    # Unknown command — fall through to the agent (Telegram parity).
    return SlashOutcome(handled=False)


def build_command_context(
    loop_context: Any,
    config: dict | None = None,
    session: Any = None,
    ensure_session: Any = None,
) -> CommandContext:
    """Build a CommandContext from the web bridge's loop_context + config.

    Pulls model / provider / system_prompt / tool_registry off the agent-loop
    context (the only command inputs the web surface has). Uptime and a cost
    tracker are absent here, so /status simply omits those lines. With no
    loop_context, the model-info commands degrade but the rest (which read
    module globals — wiki, memory, skills, health, …) still work.

    ``session`` (non-creating active session, or None) and ``ensure_session``
    (a no-arg get_or_create callable) feed the session-mutating commands; the
    bridge resolves both from its session_mgr + the message's session_id.
    """
    if loop_context is None:
        return CommandContext(
            config=config, session=session, ensure_session=ensure_session
        )
    provider = getattr(loop_context, "provider", "")
    provider_str = getattr(provider, "value", None) or str(provider or "")
    return CommandContext(
        model_name=getattr(loop_context, "model", "") or "",
        model_provider=provider_str,
        system_prompt=getattr(loop_context, "system_prompt", "") or "",
        config=config,
        tool_registry=getattr(loop_context, "tool_registry", None),
        session=session,
        ensure_session=ensure_session,
    )
