"""When a person last sent Prometheus a message (WP-X.56).

The heartbeat's idle detector reads this clock. The five ingress points that
call ``outcomes.note_user_message`` (WP-X.54 T-4) call ``note_user_activity()``
beside it: Beacon over the WebSocket, ``/api/chat``, Telegram, Slack and
Discord. Slash commands and machine-built turns (re-engagement, cron) are not
a person writing, so they do not count, the same rule as the T-4 hook.

Why a clock and not a signal: the heartbeat used to listen for a
``message_received`` signal that nothing ever emitted, and its
``record_activity()`` had no callers, so idle started once per boot and never
ended. AutoDream (and GEPA, when enabled) dreamed every 30 minutes around the
clock, model calls included. A plain timestamp needs no bus, no telemetry and
no database write, and works on every surface the same way.

Never raises. The heartbeat takes the newer of this clock and its own.
"""

from __future__ import annotations

import time

_last_user_message_at: float | None = None


def note_user_activity(at: float | None = None) -> None:
    """A person sent a message just now (or at ``at``, wall-clock seconds)."""
    global _last_user_message_at
    _last_user_message_at = time.time() if at is None else at


def last_user_activity() -> float | None:
    """Wall-clock time of the last person's message, or None if none yet."""
    return _last_user_message_at


def reset() -> None:
    """Forget the last message (tests)."""
    global _last_user_message_at
    _last_user_message_at = None
