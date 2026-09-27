"""Shared helpers for the telemetry-gap scans (docs/audits/TELEMETRY-GAPS.md).

Every live-data scan runs ON THE MINI against a SQLite backup-API copy of the
live ``telemetry.db`` (``backup_copy.py``), never the live file, and prints
aggregates only: counts, shares and sums. No session ids, tool inputs, model
output or error text are printed. Row windows are anchored at the copy's own
``--anchor`` time (default: when the scan runs), so "last 14 days" means the
14 days before the copy.

Two scans (``wire_shapes.py``, ``golden_impact.py``) read only the committed
parity fixtures and run on any checkout.

Classes mirror ``docs/audits/skill-usage/_common.py`` (same surface rules, same
attribution idea) and extend the provider classes with Ollama, which the
skill-usage audit did not need.
"""

from __future__ import annotations

import bisect
import datetime as dt
import re
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

DAY = 86400.0
WINDOWS = (14, 30)


def open_ro(path: Path) -> sqlite3.Connection:
    """A read-only handle on a COPY. ``immutable=1``: the copy has no writer."""
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def utc(ts: float | None) -> str:
    if ts is None:
        return "-"
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def anchor_arg(value: str | None) -> float:
    return float(value) if value else time.time()


def in_window(ts: float, anchor: float, days: int) -> bool:
    return anchor - days * DAY <= ts <= anchor


def window_counts(rows: Iterable[float], anchor: float) -> dict[int, int]:
    """{14: n, 30: n} for a stream of row timestamps."""
    out = {d: 0 for d in WINDOWS}
    for ts in rows:
        for d in WINDOWS:
            if in_window(ts, anchor, d):
                out[d] += 1
    return out


# ── surfaces ────────────────────────────────────────────────────────────

# Session-id prefixes that are people talking to the daemon. "web" alone is the
# routing namespace the web path filed rows under before #458.
USER_SURFACES = frozenset({
    "beacon", "telegram", "desktop", "ios", "web", "voice", "slack", "discord", "cli",
})
TEST_SURFACES = frozenset({"smoke", "probe", "verify", "beacon-verify", "parity"})


def surface(session_id: str | None) -> str:
    """A coarse class for a session id. Never returns the id itself."""
    if session_id is None:
        return "no session id"
    if session_id == "system":
        return "evals/benchmarks ('system')"
    if session_id == "web":
        return "user: web (routing namespace)"
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*?)[:_\-]", session_id)
    prefix = m.group(1).lower() if m else session_id.lower()
    if prefix == "coding":
        return "coding runs"
    if prefix in TEST_SURFACES:
        return "test harness"
    if prefix in USER_SURFACES:
        return f"user: {prefix}"
    return "user: no prefix"


def surface_class(label: str) -> str:
    if label.startswith("user:"):
        return "user surfaces"
    if label.startswith("evals"):
        return "evals/benchmarks"
    if label in ("coding runs", "test harness"):
        return label
    return "unattributed"


# ── models and providers ────────────────────────────────────────────────

def model_label(model: str | None) -> str:
    """A model identifier without host paths (a GGUF path keeps its file name)."""
    m = (model or "").strip()
    if not m:
        return "<blank>"
    return m.rsplit("/", 1)[-1]


def provider_of(model: str | None) -> str:
    """Which backend a requested model name went to, from the name alone.

    The names on this box are distinct enough: a GGUF path or file, the two
    llama.cpp aliases and the blank coding-run name are the 4090's llama.cpp;
    an ``name:tag`` is Ollama (the mini's, or the 4090 host's for the router);
    the rest are the cloud providers' own names."""
    low = model_label(model).lower()
    if low == "<blank>" or low.endswith(".gguf") or low in {"gemma4-26b", "qwen3.8-27b"}:
        return "llama.cpp (local)"
    if ":" in low:
        return "ollama (local)"
    if low.startswith(("qwen3.8-max", "qwen3.8-flash", "qwen3.7", "qwen-")):
        return "qwen (cloud)"
    if low.startswith("grok"):
        return "xai (cloud)"
    if low.startswith("claude"):
        return "anthropic (cloud)"
    return "other"


def is_path_model(model: str | None) -> bool:
    m = (model or "").strip()
    return m.endswith(".gguf") or m.startswith("/")


# ── attribution of session-less rows ────────────────────────────────────

class Attributor:
    """Give a session-less row the surface of its nearest neighbour.

    Same rule as the skill-usage audit's ``session_gaps.py``: the closest row
    (either side, within 5 min) with the same model and a session id, else the
    latest ``loop_round`` row of the same model within 10 min. Only the
    neighbour's CLASS is returned, never its id.
    """

    def __init__(self, tel: sqlite3.Connection) -> None:
        self._calls: dict[str, list[tuple[float, str]]] = {}
        for r in tel.execute(
            "SELECT timestamp, model, session_id FROM tool_calls WHERE session_id IS NOT NULL"
        ):
            self._calls.setdefault(r["model"] or "", []).append((r["timestamp"], r["session_id"]))
        self._rounds: dict[str, list[tuple[float, str]]] = {}
        for r in tel.execute(
            "SELECT timestamp, model, session_id FROM subsystem_runs WHERE subsystem='agent_loop'"
            " AND operation='loop_round' AND session_id IS NOT NULL"
        ):
            self._rounds.setdefault(r["model"] or "", []).append((r["timestamp"], r["session_id"]))
        for d in (self._calls, self._rounds):
            for lst in d.values():
                lst.sort()

    def __call__(self, ts: float, model: str | None) -> str:
        lst = self._calls.get(model or "", [])
        i = bisect.bisect_left(lst, (ts, ""))
        near = [lst[j] for j in (i - 1, i) if 0 <= j < len(lst) and abs(lst[j][0] - ts) <= 300]
        if near:
            return surface(min(near, key=lambda x: abs(x[0] - ts))[1])
        rl = self._rounds.get(model or "", [])
        k = bisect.bisect_right(rl, (ts, "￿")) - 1
        if k >= 0 and ts - rl[k][0] <= 600:
            return surface(rl[k][1])
        return "unattributed"


def table(rows: Iterable[tuple], header: tuple[str, ...]) -> None:
    """Print a plain aligned table (aggregates only)."""
    rows = [tuple(str(c) for c in r) for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    print("   " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("   " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))
