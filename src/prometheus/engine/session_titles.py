"""Session display titles (GRAFT-MOBILE-BRIDGE 7).

A session id is ``<gateway>:<uuid>`` and carries no name anywhere in the
daemon. Clients survive on recency grouping and gateway badges on a wide
desktop sidebar; a phone list of raw ids is unusable. This module generates a
title from a session's first real exchange — once, after the first assistant
turn completes — with a budget-capped local model call, and stores it in the
conversation store's ``session_titles`` table. A manual rename
(``PUT /api/sessions/{id}/title``) writes the same table and, because
generation only ever fills ABSENCE, is never overwritten.

Fire-and-forget by design: a title is a nicety, the turn is the product. Every
failure path here degrades to "no title", which clients already handle with
their first-user-message snippet fallback.
"""

from __future__ import annotations

import logging

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
)

logger = logging.getLogger(__name__)

MAX_TITLE_CHARS = 48

# One short instruction, one exchange, one line back. max_tokens caps the
# spend; the char cap below is the contract with the UI.
_PROMPT = (
    "Name this conversation in at most six words. Respond with ONLY the "
    "title — no quotes, no punctuation at the end, no explanation.\n\n"
    "[user] {user}\n[assistant] {assistant}"
)


def clip_title(raw: str, limit: int = MAX_TITLE_CHARS) -> str:
    """Collapse whitespace and cap at ``limit`` on a word boundary.

    The same shape as Beacon's ``snippetTitle`` so a generated title and a
    client-side fallback read alike. Strips wrapping quotes a model may add
    despite the instruction — measured behaviour, not paranoia.
    """
    flat = " ".join((raw or "").split()).strip().strip("\"'“”‘’")
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    space = cut.rfind(" ")
    if space > int(limit * 0.6):
        cut = cut[:space]
    return cut.rstrip() + "…"


def first_exchange(messages: list) -> tuple[str, str] | None:
    """The first non-empty user text and the first non-empty assistant text
    after it, or None when the session has no such exchange yet (a slash
    command, a tool-only turn, an empty send)."""
    user_text = ""
    for msg in messages:
        text = (getattr(msg, "text", "") or "").strip()
        if not text:
            continue
        role = getattr(msg, "role", "")
        if not user_text:
            if role == "user":
                user_text = text
        elif role == "assistant":
            return user_text, text
    return None


async def generate_title(provider, model: str, user: str, assistant: str) -> str | None:
    """One budget-capped call → a clipped title, or None on any failure."""
    request = ApiMessageRequest(
        model=model,
        messages=[
            ConversationMessage(
                role="user",
                # The exchange is truncated hard: a title needs the opening,
                # not the transcript, and the tokens are the budget.
                content=[TextBlock(text=_PROMPT.format(
                    user=user[:500], assistant=assistant[:500]))],
            ),
        ],
        system_prompt="You name conversations. Respond with only the title.",
        max_tokens=24,
    )
    try:
        full_text = ""
        async for event in provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                full_text += event.text
            elif isinstance(event, ApiMessageCompleteEvent):
                content = getattr(event.message, "content", None)
                if isinstance(content, str) and content:
                    full_text = content
                elif isinstance(content, list):
                    parts = [b.text for b in content if hasattr(b, "text")]
                    if parts:
                        full_text = "".join(parts)
        title = clip_title(full_text)
        return title or None
    except Exception:
        logger.debug("session title generation failed", exc_info=True)
        return None


async def maybe_title_session(store, provider, model: str,
                              session_id: str, messages: list) -> None:
    """Generate-and-store, iff the session has no title yet.

    Absence-only: a manual rename (or an earlier generation) is never
    overwritten. Every failure is swallowed to a debug log — the surrounding
    turn must never notice this ran.
    """
    try:
        if store is None or provider is None:
            return
        if store.get_session_title(session_id):
            return
        exchange = first_exchange(messages)
        if exchange is None:
            return
        title = await generate_title(provider, model, *exchange)
        if title and not store.get_session_title(session_id):
            store.set_session_title(session_id, title)
            logger.info("titled session %s: %r", session_id, title)
    except Exception:
        logger.debug("maybe_title_session failed", exc_info=True)
