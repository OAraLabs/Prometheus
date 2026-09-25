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


# ---------------------------------------------------------------------------
# Regression judgement against a committed baseline
# ---------------------------------------------------------------------------
#
# The benchmark is only useful if its noise band is narrower than the smallest
# regression it must catch. The budget is PER ROUND (default 10 ms, the hook
# budget WP-1.2 is sized for). The rule:
#
#   * the statistic is the MEDIAN of a session's per-run p50s — one number per
#     session, robust to a single slow run;
#   * a session is FLAGGED when it exceeds the baseline's median by more than
#     half the budget, so a full-budget regression clears the threshold with
#     half a budget to spare;
#   * the judgement REFUSES (exit 2) when the baseline cannot resolve that:
#     if two baseline sessions of the same code already differ by half a
#     budget, a flag would be a coin toss, and a pass would prove nothing.
#
# Measured on stub-provider replay only: a live model's timing band is far
# wider than any hook budget, so a regression would hide in it.


def judge(new_runs: list[dict], sessions: list[list[dict]], budget_ms: float) -> tuple[int, list[str]]:
    base_all = [r["p50_ms"] for s in sessions for r in s]
    base_median = statistics.median(base_all)
    session_medians = [statistics.median([r["p50_ms"] for r in s]) for s in sessions]
    drift = (max(session_medians) - min(session_medians)) if len(sessions) > 1 else \
        (max(base_all) - min(base_all)) / 2
    threshold = budget_ms / 2
    new_median = statistics.median([r["p50_ms"] for r in new_runs])
    delta = new_median - base_median
    mean_delta = (statistics.median([r["mean_ms"] for r in new_runs])
                  - statistics.median([r["mean_ms"] for s in sessions for r in s]))
    rss_delta = (statistics.median([r["max_hwm_mb"] for r in new_runs])
                 - statistics.median([r["max_hwm_mb"] for s in sessions for r in s]))
    lines = [
        f"baseline: {len(sessions)} session(s), {len(base_all)} runs; per-run p50 "
        f"{min(base_all):.1f}–{max(base_all):.1f} ms, median {base_median:.1f} ms; "
        f"session medians {', '.join(f'{m:.1f}' for m in session_medians)} ms "
        f"(drift {drift:.1f} ms)",
        f"this session: median p50 {new_median:.1f} ms  (Δ {delta:+.1f} ms; "
        f"mean Δ {mean_delta:+.1f} ms; peak RSS Δ {rss_delta:+.1f} MB)",
        f"budget {budget_ms:.0f} ms/round -> flag above Δ +{threshold:.1f} ms",
    ]
    if drift >= threshold:
        lines.append(f"CANNOT JUDGE: the baseline's own drift ({drift:.1f} ms) is not below half "
                     f"the budget ({threshold:.1f} ms) — the band is too wide to resolve it")
        return 2, lines
    if delta > threshold:
        lines.append(f"REGRESSION: +{delta:.1f} ms per round (p50) is outside the noise band")
        return 1, lines
    lines.append("within the noise band")
    return 0, lines
