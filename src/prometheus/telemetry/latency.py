"""How a latency aggregate says where it came from.

WHY THIS EXISTS
---------------
`tool_calls.latency_ms` was declared ``REAL NOT NULL DEFAULT 0.0``, so
"nobody measured this" and "this took zero milliseconds" were the SAME STORED
VALUE. Of the telemetry record sites in the agent loop, only the real execution
path times anything; permission denials, validation failures and repeat-guard
blocks record calls that never ran and took the default.

Downstream, `AVG(latency_ms)` mixed never-ran rows in as zero-duration
executions and dragged every per-tool average toward zero.

The schema is fixed (v2 allows NULL, and new writes pass NULL when nothing was
timed). But a fix at the write side does not repair what is already stored, and
the rows written before v2 CANNOT BE REPAIRED: an existing 0.0 might be a
default or a genuine fast call, and nothing recorded which. Backfilling them to
NULL would encode the inference "an exact 0.0 always means unmeasured" — sound
today, and exactly the kind of thing that rots — into a permanent, irreversible
migration. So they are left alone, and readers say so instead.

THE SHAPE
---------
Every aggregate returns ``(value, source)``, the same shape as
``context.budget.resolve_effective_limit`` — the number plus how it was
obtained. A caller that wants one number has to decide what to do about the
provenance rather than having it silently averaged away.

    MEASURED    every contributing row was timed, on schema v2 or later
    UNMEASURED  v2+ rows exist and none of them was timed
    UNKNOWN     one or more contributing rows predate v2, so a stored 0.0
                cannot be told from a default

UNKNOWN is deliberately sticky: a single pre-v2 row makes the whole aggregate
unknown, because the average has already mixed provenances by the time anyone
looks at it. Reporting "12.3ms" for a window that is half guesswork is the
failure this module exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Every contributing row was timed, on schema v2 or later.
LATENCY_MEASURED = "measured"
#: v2+ rows exist for this key and none of them carried a measurement.
LATENCY_UNMEASURED = "unmeasured"
#: At least one contributing row predates schema v2 — a stored 0.0 there is
#: indistinguishable from the NOT NULL DEFAULT that used to be written.
LATENCY_UNKNOWN = "unknown"


@dataclass
class LatencyAggregate:
    """Accumulates latencies and the provenance of what it accumulated.

    Counting is separated from resolving so a caller can feed rows in any
    order and ask the question once, at the end.
    """

    total_ms: float = 0.0
    measured_rows: int = 0
    unmeasured_rows: int = 0
    pre_v2_rows: int = 0

    def add(self, latency_ms: float | None, *, pre_v2: bool) -> None:
        """Fold one row in.

        ``pre_v2`` is about the ROW, not the value: a row written before the
        schema allowed NULL cannot be trusted to distinguish the two, whatever
        number it holds.
        """
        if pre_v2:
            self.pre_v2_rows += 1
            if latency_ms is not None:
                self.total_ms += latency_ms
            return
        if latency_ms is None:
            self.unmeasured_rows += 1
            return
        self.measured_rows += 1
        self.total_ms += latency_ms

    def resolve(self) -> tuple[float | None, str]:
        """The mean and where it came from.

        A ``None`` value means there is no number to report — NOT zero. Zero is
        a measurement; absence is not, and collapsing them is the whole defect.
        """
        if self.pre_v2_rows:
            counted = self.pre_v2_rows + self.measured_rows
            return (self.total_ms / counted if counted else None), LATENCY_UNKNOWN
        if self.measured_rows:
            return self.total_ms / self.measured_rows, LATENCY_MEASURED
        return None, LATENCY_UNMEASURED


def render_latency(value: float | None, source: str) -> str:
    """The string a human reads. THE THREE SOURCES MUST NOT RENDER ALIKE.

    That is the property worth testing, and it is not "the code sets a tag" —
    it is that the OUTPUT differs depending on which rows fed it. A tag that
    exists in a dict and never reaches the page is a tag nobody can act on.
    """
    if source == LATENCY_MEASURED:
        return f"{value:.1f}ms" if value is not None else "—"
    if source == LATENCY_UNMEASURED:
        return "not measured"
    if source == LATENCY_UNKNOWN:
        shown = f"{value:.1f}ms" if value is not None else "—"
        return f"{shown} (unverified: includes rows written before schema v2)"
    return "—"
