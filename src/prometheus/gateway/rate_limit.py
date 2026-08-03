"""Per-chat rate limiting with a global ceiling, for the public gateway surface.

WHY THIS EXISTS
---------------
``gateway.rate_limits.messages_per_minute`` and ``media_downloads_per_minute``
were declared in the config with **no implementation anywhere in src/** — not a
limiter that was misconfigured, a limiter that did not exist. Telegram is the
one surface exposed to the public internet by design, so the config described a
control the system did not have.

DESIGN, and why each half is needed
-----------------------------------
* **Per-chat** is what protects you from one noisy peer. Global-only lets a
  single chat starve every other.
* **A global ceiling above it** is what protects the daemon from aggregate
  load. Per-chat-only leaves the box exposed as ``allowed_chat_ids`` grows.
* **Separate message and media budgets.** A text message costs a model call; a
  media download costs bandwidth, disk and possibly a vision call. Sharing one
  budget means the cheap thing starves the expensive one, or the reverse.
* **Warn once per window, not per message.** Warning on every drop makes the
  warning the flood. Silent dropping is worse still — indistinguishable from
  the bot being broken, and this codebase's standing rule is that silent
  failure is the enemy.

State is deliberately in-memory: a restart legitimately resets a per-minute
window. (Contrast the drift-nudge state in ``heartbeat.py``, which is persisted
precisely because a restart is the event that *resolves* the condition it
tracks.)

Source: novel code for Prometheus, 2026-08-03.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class Budget(str, Enum):
    """Independent budgets. Spending one must not spend the other."""

    MESSAGES = "messages"
    MEDIA = "media"


@dataclass
class Decision:
    """Outcome of one admission check."""

    allowed: bool
    budget: Budget
    scope: str          # "chat" | "global" | ""
    should_warn: bool   # True only on the FIRST refusal in this window


_WINDOW_SECONDS = 60.0


@dataclass
class _Window:
    """Sliding 60s window of event timestamps, plus one warn latch."""

    hits: deque[float] = field(default_factory=deque)
    warned_at: float | None = None

    def prune(self, now: float) -> None:
        cutoff = now - _WINDOW_SECONDS
        while self.hits and self.hits[0] <= cutoff:
            self.hits.popleft()
        # The latch expires with the window it belongs to, so the sender is
        # warned again in a NEW window rather than muted forever.
        if self.warned_at is not None and self.warned_at <= cutoff:
            self.warned_at = None


class RateLimiter:
    """Sliding-window limiter: per-chat budgets under a global ceiling.

    ``per_chat`` maps a budget to its per-minute allowance for a single chat.
    ``global_ceiling`` maps a budget to the aggregate allowance across all
    chats. A ceiling of 0 or None disables that ceiling.
    """

    def __init__(
        self,
        *,
        messages_per_minute: int = 30,
        media_per_minute: int = 10,
        global_messages_per_minute: int | None = None,
        global_media_per_minute: int | None = None,
    ) -> None:
        self._per_chat: dict[Budget, int] = {
            Budget.MESSAGES: int(messages_per_minute),
            Budget.MEDIA: int(media_per_minute),
        }
        # Default the ceiling well above the per-chat allowance rather than to
        # it — a ceiling equal to one chat's budget would make the second chat
        # unusable, which is a denial of service dressed as a control.
        self._global: dict[Budget, int | None] = {
            Budget.MESSAGES: global_messages_per_minute,
            Budget.MEDIA: global_media_per_minute,
        }
        self._chat_windows: dict[tuple[str, Budget], _Window] = {}
        self._global_windows: dict[Budget, _Window] = {}

    # -- internals ---------------------------------------------------------

    def _chat_window(self, chat_id: str, budget: Budget) -> _Window:
        return self._chat_windows.setdefault((chat_id, budget), _Window())

    def _global_window(self, budget: Budget) -> _Window:
        return self._global_windows.setdefault(budget, _Window())

    @staticmethod
    def _refuse(win: _Window, now: float, budget: Budget, scope: str) -> Decision:
        first = win.warned_at is None
        if first:
            win.warned_at = now
        return Decision(allowed=False, budget=budget, scope=scope, should_warn=first)

    # -- public ------------------------------------------------------------

    def check(self, chat_id: str | int, budget: Budget, *, now: float | None = None) -> Decision:
        """Admit or refuse one event. Records the spend only when admitted.

        A refusal must not consume budget, or a chat already over its limit
        would keep its window permanently full and never recover.
        """
        now = time.monotonic() if now is None else now
        chat_key = str(chat_id)

        chat_win = self._chat_window(chat_key, budget)
        chat_win.prune(now)
        if len(chat_win.hits) >= self._per_chat[budget]:
            return self._refuse(chat_win, now, budget, "chat")

        ceiling = self._global[budget]
        global_win = self._global_window(budget)
        global_win.prune(now)
        if ceiling and len(global_win.hits) >= ceiling:
            return self._refuse(global_win, now, budget, "global")

        chat_win.hits.append(now)
        global_win.hits.append(now)
        return Decision(allowed=True, budget=budget, scope="", should_warn=False)

    @staticmethod
    def warning_text(decision: Decision) -> str:
        """Sender-facing text. Names the budget so the limit is actionable."""
        what = "messages" if decision.budget is Budget.MESSAGES else "media downloads"
        if decision.scope == "global":
            return (
                f"Rate limit reached for {what} across all chats — please wait "
                f"a minute. Further {what} in this window are dropped silently."
            )
        return (
            f"Rate limit reached for {what} — please wait a minute. Further "
            f"{what} in this window are dropped silently."
        )
