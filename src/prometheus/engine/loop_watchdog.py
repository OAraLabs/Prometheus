"""Event-loop lag watchdog.

⚠ RECONSTRUCTED SPEC. This was rebuilt from constraints stated in review, not
transcribed from an approved design document. The constraints are recorded
below so the next reader can tell which decisions were specified and which
were mine.

WHY IT EXISTS
-------------
Beacon closes the socket, not the daemon: PING_INTERVAL 20s, STALE_AFTER 45s,
then ``ws.terminate()``. On 2026-08-16 that fired 42 times, with reconnect
gaps of 1.3s and 4s (backoff attempts 1 and 2), and **zero** agent_progress
emit failures recorded. Two independent clients dropped in the same
millisecond, which is not network flakiness.

The hypothesis: the event loop is BLOCKED. ``_emit_progress`` never reported a
failure because it never RAN — it awaits ``asyncio.sleep(3)``
(ws_server.py:954), so a blocked loop stops the heartbeat without raising
anything. The client sees silence and terminates.

This task measures that directly, and it is falsifiable both ways:

  * lag spikes >= 45,000 ms clustering at the disconnect timestamps CONFIRM
    it — 45,000 ms is Beacon's STALE_AFTER_MS exactly.
  * sustained sub-250 ms lag while disconnects continue KILLS it, and the
    fault is network or proxy.

THE CONSTRAINTS, as given
-------------------------
1. An async task started ONCE with the daemon, not per-turn — so it also
   catches stalls while idle, which a per-turn timer cannot see.
2. ``sleep(1.0)`` in a loop; lateness is ``(loop.time() - before - 1.0) *
   1000``.
3. ``loop.time()`` specifically. It is MONOTONIC, so an NTP step cannot be
   misread as lag; a ``time.time()`` version would report a clock correction
   as a multi-second stall and manufacture exactly the evidence we are
   looking for.
4. WARNING, not debug. The progress emitter logs its own failures at debug
   and that is precisely why 42 disconnects produced no visible evidence.
   A measurement nobody sees is not a measurement.
5. Threshold 250 ms. Below that is ordinary scheduling jitter.
6. Include the live ``progress["phase"]`` and ``tool_name`` so a spike names
   its likely cause and not merely its size.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Lateness at or above this is reported. Below is scheduling jitter.
LAG_THRESHOLD_MS = 250.0

#: The sampling period. Also the unit lateness is measured against.
INTERVAL_SECONDS = 1.0

#: The live turn's progress dict, published by the WS handler while a turn is
#: in flight and cleared afterwards. A plain module slot rather than a
#: parameter because the watchdog outlives every turn: it is started once and
#: must keep sampling while nothing is running, which is when an idle stall
#: would otherwise be invisible.
_current_progress: dict[str, Any] | None = None


def publish_progress(progress: dict[str, Any] | None) -> None:
    """Point the watchdog at the live turn's progress dict, or clear it."""
    global _current_progress
    _current_progress = progress


def _annotation() -> str:
    """phase/tool for the warning line. Never raises — this runs inside the
    logging path of a diagnostic, and a diagnostic that can crash the thing
    it measures is worse than no diagnostic."""
    src = _current_progress
    if not isinstance(src, dict):
        return "phase=idle tool=None"
    return f"phase={src.get('phase', 'idle')} tool={src.get('tool_name')}"


async def watch(
    *,
    threshold_ms: float = LAG_THRESHOLD_MS,
    interval: float = INTERVAL_SECONDS,
) -> None:
    """Sample event-loop lateness forever. Never raises, never exits.

    ``loop.time()`` is the monotonic clock the scheduler itself uses, so the
    difference between the sleep we asked for and the sleep we got is exactly
    the time the loop spent unable to run us.
    """
    loop = asyncio.get_running_loop()
    logger.info(
        "event-loop watchdog started (threshold %.0f ms, interval %.1f s)",
        threshold_ms, interval,
    )
    while True:
        try:
            before = loop.time()
            await asyncio.sleep(interval)
            lag_ms = (loop.time() - before - interval) * 1000.0
            if lag_ms >= threshold_ms:
                logger.warning(
                    "event-loop lag %.0f ms (%s)", lag_ms, _annotation(),
                )
        except asyncio.CancelledError:
            logger.info("event-loop watchdog stopped")
            raise
        except Exception:  # pragma: no cover - defensive; must not die
            logger.warning("event-loop watchdog iteration failed", exc_info=True)
            await asyncio.sleep(interval)


def start(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task:
    """Create the watchdog task. Call once, at daemon startup."""
    loop = loop or asyncio.get_event_loop()
    return loop.create_task(watch(), name="event-loop-watchdog")


def current_progress() -> dict[str, Any] | None:
    """The dict the watchdog is currently annotating with.

    Exposed so a turn's teardown can clear the slot only when it still points
    at its OWN dict — a concurrent turn may have published since, and blindly
    clearing would blind the newer turn.
    """
    return _current_progress
