"""Daemon overhead per round, measured from outside the daemon.

A ROUND is one model call inside a chat turn. For a turn with calls
c_0 .. c_{n-1}, every instant below is on the host's monotonic clock and
none of them is taken inside the daemon:

    t_post          the harness sends POST /api/chat/send
    c_k.recv        the model server has read request k in full
    c_k.sent        the model server has flushed the last byte of response k
    t_done          the harness receives the turn's ``chat_done`` frame

The gaps between them are the daemon's — everything it does that is not the
model answering:

    gap_0   = c_0.recv - t_post            (REST handler, persist, route, build prompt)
    gap_k   = c_k.recv - c_{k-1}.sent      (parse, adapter, gate, hooks, TOOLS, next prompt)
    tail    = t_done  - c_{n-1}.sent       (parse, persist, broadcast)

Tool execution is subtracted from the gap it ran in, using the daemon's own
``tool_calls.latency_ms`` (which brackets ``tool.execute`` and nothing else),
placed by the row's write time. Model time is excluded by construction: the
replay model answers from memory, and its own serve time lies between recv
and sent, outside every gap.

Round k's overhead is gap_k minus its tools; the tail is added to the last
round. Coding runs are not sampled: their end is only observable by polling,
and a polling interval would land in the tail.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from parity.runner import RunOutput


@dataclass
class RoundSample:
    scenario: str
    session: str
    index: int
    overhead_ms: float


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _union_ns(spans: list[tuple[int, int]]) -> int:
    total, cur_b, cur_e = 0, None, None
    for b, e in sorted(spans):
        if cur_e is None or b > cur_e:
            if cur_e is not None:
                total += cur_e - cur_b
            cur_b, cur_e = b, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_b
    return total


def samples(out: RunOutput) -> tuple[list[RoundSample], list[str]]:
    rounds: list[RoundSample] = []
    notes: list[str] = []
    for turn in out.turns:
        calls = sorted((s for s in out.served if turn.t_post <= s.t_recv_ns <= turn.t_done
                        and s.t_sent_ns), key=lambda s: s.t_recv_ns)
        if not calls:
            notes.append(f"{out.scenario}/{turn.session}: no model call inside the turn")
            continue
        bounds = [turn.t_post] + [c.t_sent_ns for c in calls]
        ends = [c.t_recv_ns for c in calls] + [turn.t_done]
        gaps_ns = [e - b for b, e in zip(bounds, ends)]           # n calls -> n+1 gaps
        spans: list[list[tuple[int, int]]] = [[] for _ in gaps_ns]
        for wall_ts, latency_ms in turn.tools_ms:
            end = int(wall_ts * 1e9) - out.wall_offset_ns     # the row is written as the tool ends
            for i, (b, e) in enumerate(zip(bounds, ends)):
                if b <= end <= e:
                    spans[i].append((max(b, end - int(latency_ms * 1e6)), end))
                    break
            else:
                notes.append(f"{out.scenario}/{turn.session}: a tool row fell outside every gap")
        # Parallel read-only tools overlap: subtract the UNION of their
        # execution spans, not the sum, or concurrency reads as negative overhead.
        tools_ns = [_union_ns(sp) for sp in spans]
        net = [g - t for g, t in zip(gaps_ns, tools_ns)]
        per_round = net[:-1]
        per_round[-1] += net[-1]                                    # the tail joins the last round
        for i, ns in enumerate(per_round):
            rounds.append(RoundSample(out.scenario, turn.session, i, ns / 1e6))
    return rounds, notes


@dataclass
class RunStats:
    rounds: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_hwm_kb: int
    max_rss_kb: int


def run_stats(samples_ms: list[float], rss: list[dict]) -> RunStats:
    return RunStats(
        rounds=len(samples_ms),
        p50_ms=_pct(samples_ms, 0.50),
        p95_ms=_pct(samples_ms, 0.95),
        mean_ms=statistics.fmean(samples_ms) if samples_ms else float("nan"),
        max_hwm_kb=max((r.get("VmHWM", 0) for r in rss), default=0),
        max_rss_kb=max((r.get("VmRSS", 0) for r in rss), default=0),
    )


def band(values: list[float]) -> str:
    if not values:
        return "n/a"
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    spread = (hi - lo) / 2
    rel = f" (±{100 * spread / med:.0f}%)" if med else ""
    return f"median {med:.1f}, range {lo:.1f}–{hi:.1f}{rel}"
