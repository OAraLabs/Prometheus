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


class _StoredPart:
    """Adapter for :func:`first_exchange`, which reads ``.role``/``.text`` (the
    live hook hands it ``ConversationMessage``s). Store rows are
    ``MessagePart``s carrying ``.content`` instead — same words, different
    attribute."""

    __slots__ = ("role", "text")

    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.text = text


def _adapt_store_messages(parts: list) -> list:
    return [_StoredPart(getattr(p, "role", ""), getattr(p, "content", "") or "")
            for p in parts]


async def backfill_titles(store, provider, model: str, *, dry_run: bool = False,
                          limit: int | None = None, delay_seconds: float = 0.0,
                          log=print) -> dict:
    """One-off backfill for sessions predating Piece 7 (GRAFT-MOBILE-BRIDGE 9).

    Generation is absence-only and fires after a completed turn, so sessions
    that predate it stay untitled unless they happen to be used again. This
    walks ``list_sessions()`` (which already excludes tombstoned sessions) and
    titles what qualifies — a script's engine, not a route and not a startup
    hook.

    - **Sequential, never concurrent** — the model is shared with the live
      daemon; ``delay_seconds`` paces calls further apart.
    - **Idempotent and resumable** — already-titled sessions are skipped, so an
      interrupted run costs nothing and a rerun is a no-op.
    - **Dry runs generate but never store** — "the title each would receive" is
      a real model call; a dry run followed by a real run pays twice, by design.
    - Real runs go through :func:`maybe_title_session`, so there is exactly one
      definition of what a title is and how it is written.
    - ``limit`` caps the number of GENERATION attempts (skips are free).

    Returns counters: ``eligible``, ``titled``, ``would_title``,
    ``already_titled``, ``too_short``, ``no_exchange``, ``failed``.
    """
    import asyncio

    counts = {"eligible": 0, "titled": 0, "would_title": 0,
              "already_titled": 0, "too_short": 0, "no_exchange": 0, "failed": 0}
    attempts = 0
    for row in store.list_sessions():
        sid = row["session_id"]
        if row.get("title"):
            counts["already_titled"] += 1
            continue
        if int(row.get("message_count") or 0) < 2:
            counts["too_short"] += 1
            continue
        messages = _adapt_store_messages(store.get_all_messages(sid))
        exchange = first_exchange(messages)
        if exchange is None:
            counts["no_exchange"] += 1
            continue
        counts["eligible"] += 1
        if limit is not None and attempts >= limit:
            continue
        attempts += 1
        if delay_seconds and attempts > 1:
            await asyncio.sleep(delay_seconds)
        if dry_run:
            title = await generate_title(provider, model, *exchange)
            if title:
                counts["would_title"] += 1
                log(f"would title {sid}: {title!r}")
            else:
                counts["failed"] += 1
                log(f"generation failed for {sid}")
        else:
            await maybe_title_session(store, provider, model, sid, messages)
            title = store.get_session_title(sid)
            if title:
                counts["titled"] += 1
                log(f"titled {sid}: {title!r}")
            else:
                counts["failed"] += 1
                log(f"generation failed for {sid}")
    return counts


def schedule(tasks: set, *, store, provider, model: str,
             session_id: str, messages: list) -> None:
    """Fire-and-forget `maybe_title_session`, done SAFELY: the event loop holds
    tasks only weakly, so a bare `create_task` nobody keeps can be
    garbage-collected mid-flight — the standard asyncio fire-and-forget trap.
    The caller passes its retention set; the done-callback releases, so the set
    stays O(in-flight). One implementation for every surface that titles
    (the WS bridge, the Telegram gateway) — two copies of this footgun is how
    one of them quietly loses it again.

    No-op (schedules nothing) without a store AND a provider, or off the event
    loop — a title is a nicety and must never be the thing that raises.
    """
    import asyncio

    try:
        if store is None or provider is None:
            return
        task = asyncio.get_running_loop().create_task(
            maybe_title_session(store, provider, model, session_id, list(messages))
        )
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    except Exception:
        logger.debug("session title scheduling failed", exc_info=True)


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
