"""A sliding-window limit keyed by the TCP peer address.

The unauthenticated pairing routes cannot lean on a token, so they lean on how often one source may
ask (``GET /api/hello`` first; the request routes reuse this). Three properties are the point:

* **A refusal spends nothing.** Otherwise a source already over its limit keeps its window full and
  never recovers.
* **The retry hint is real**: the time until the oldest counted hit leaves the window, rounded up.
* **Memory is bounded.** The key is chosen by whoever can reach the port, so an unbounded dict keyed by
  it is a slow leak an outsider controls. Idle sources are dropped once their window has passed, and
  the least recently seen goes first if there are still too many.

The key is the TCP peer, never ``X-Forwarded-For``: an unauthenticated caller writes that header.
State is in memory on purpose; a restart legitimately resets a per-minute window.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    allowed: bool
    #: Whole seconds until a refused source may try again; 0 when it was allowed.
    retry_after: int = 0


class SourceLimiter:
    """At most *limit* admitted events per *window* seconds for each source."""

    def __init__(
        self,
        limit: int,
        window: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_sources: int = 4096,
    ) -> None:
        self._limit = int(limit)
        self._window = float(window)
        self._clock = clock
        self._max_sources = int(max_sources)
        # Ordered by when each source was last seen, oldest first.
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._hits)

    def __contains__(self, source: object) -> bool:
        return source in self._hits

    def check(self, source: str) -> Decision:
        """Admit or refuse one event from *source*. Records the spend only when admitted."""
        now = self._clock()
        cutoff = now - self._window
        self._forget_idle(cutoff)
        hits = self._hits.get(source)
        if hits is None:
            hits = deque()
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self._limit:
            self._hits[source] = hits
            self._hits.move_to_end(source)
            oldest = hits[0] if hits else now
            return Decision(False, max(1, math.ceil(oldest + self._window - now)))
        hits.append(now)
        self._hits[source] = hits
        self._hits.move_to_end(source)
        while len(self._hits) > self._max_sources:
            self._hits.popitem(last=False)
        return Decision(True)

    def _forget_idle(self, cutoff: float) -> None:
        """Drop the sources whose newest hit has left the window. They sit at the front."""
        while self._hits:
            source, hits = next(iter(self._hits.items()))
            if hits and hits[-1] > cutoff:
                break
            del self._hits[source]
