#!/usr/bin/env python3
"""Round 2 generation automation on the 3090 Ti host (operator 2026-11-01 ~01:20 ET: "round 2 without Max").

    round2_chain.py config                      # read-only: the settings, the 4090 serve profile, the run names
    round2_chain.py sunday      [--execute]     # the waiting Sun -> Mon chain (transient unit r2-tower-chain-<date>)
    round2_chain.py tower-night [--execute]     # timer r2-tower-night (01:12 ET): the 4090 teacher + remote gen
    round2_chain.py mini-night start|end [--execute]   # timers window-mini-night-{start,end} (01:05 / 06:40 ET)
    round2_chain.py gen-local run [--execute] [--until T]   # GEN_RUNNER of the mini's own windows
    round2_chain.py heartbeat   [--send]        # timer r2-heartbeat (09:00 ET): <= 10 lines (stdout, or Telegram)
    round2_chain.py status                      # read-only

Runs ON the 3090 Ti host from ``~/distill/round2/bin/`` (stdlib only), settings from ``round2.conf`` (KEY=VALUE,
shell syntax; ``round2.conf.example`` beside this file; the live file names this site's hosts, so it is never
committed). Every command except ``config``/``status``/``heartbeat`` is a dry run unless ``--execute``.

The 4090 (``tower/round2_tower.sh`` there; this host may ssh to it, never the other way round):
* ``sunday``: waits for the post-train pipeline's ``PIPELINE-DONE.json`` (its outcome must start with
  ``completed``; a failed/stopped pipeline, its unit ending without the marker, or no marker by
  ``R2_SUN_LATEST_START`` (Mon 04:00 EST): nothing is done, only logged), then for the capacity block's
  ``CAPACITY-DONE.json`` (it owns the 4090 GPU until then; after ``R2_CAPACITY_TIMEOUT_H`` past the pipeline's
  end the chain goes on anyway, logged, and has the 4090 stop the block's still-running ``distill-cap-*`` units
  by name). Then ``round2_tower.sh start --mode cont``: business stopped the window machinery's way (or left
  down when the block left it down: no extra flap), the teacher in the configured serve profile, a one-shot
  end on the 4090 at ``R2_SUN_END`` (Mon 05:20 EST) that stops it and restores business (Telegram on down
  and back). Then this host's remote generation (``gen_runner.py remote``: the sandboxed gym here,
  concurrency = the server's slots) as its own unit ``r2-gen-tower-<stamp>`` into its own run directory,
  until ``R2_GEN_MARGIN_S`` before the end. Generation that finishes early (nothing left) ends the 4090
  window early; one that fails is restarted (at most 3 times). At the end the generation unit is stopped
  and, if the 4090's end timer did not run, the end is started from here.
* ``tower-night``: the same for a nightly 4090 window (``--mode night``), from ``R2_TOWER_NIGHTLY_FROM`` (after
  the continuous window), started at 01:12 ET, before the window machinery's own 01:20 ``window_start``
  (which then finds the window open and skips); generation until ``R2_TOWER_NIGHT_END`` - margin, the 4090's
  end at ``R2_TOWER_NIGHT_END`` (06:22 ET), so its 06:40 window end only verifies.
* The serve profile (``R2_TOWER_PROFILE``): ``standard`` (``R2_STANDARD_PROFILE``, 4x32768), ``longctx``
  (one slot at the capacity block's measured max stable context for this host and quant, from
  ``R2_TEACHER_CTX_JSON``, capped at ``R2_LONGCTX_CAP``; the standard profile when the file is missing or
  holds no such value) or an explicit ``SLOTSxCTX``. The 4090 tries the profiles in order and falls back
  to the standard one if the first does not verify.

The 3090 Ti's own windows (``mini_window.py``; the teacher takes Saluki's port, Saluki is restored at every
window end, the voice-probe guard stays in front of the local teacher, Ollama stays loaded):
* ``mini-night start``: from ``R2_MINI_NIGHTLY_FROM`` (after the continuous window), ``mini_window_start.sh
  --execute`` with a conf whose ``GEN_RUNNER`` is ``gen_local.sh`` beside this file (dead-man first, the
  window machinery's own gates); ``mini-night end``: ``mini_window_end.sh --execute`` (never before
  ``R2_MINI_NIGHTLY_FROM``: it would end the continuous window early).
* ``gen-local run``: ``gen_runner.py run`` from the round-2 code (``R2_CODE``) with the round-2 taskset and
  flags (``R2_LOCAL_*``) into its own run directory; the window's own runner if that code is not staged.

Settings that plug in later (the round-2 code agent's generation commands): ``R2_CODE`` (a ``gen_runner.py
stage-code --dest`` export), ``R2_TOWER_TASKS`` / ``R2_LOCAL_TASKS``, ``R2_TOWER_EXTRA_ARGS`` /
``R2_LOCAL_EXTRA_ARGS`` (extra ``teacher_gen.py run`` flags), ``R2_TOWER_PROFILE``, ``R2_GEN_STATS_CMD``, and
any ``GEN_*`` key (passed to gen_runner verbatim). The run directories' names carry a hash of the code
commit, taskset and flags (teacher_gen refuses to resume an output written with other settings); every
round-2 run skips the tasks the others already finished.

Units are started and stopped only by name; nothing is killed by pattern. Business names never appear here.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
HERE = Path(__file__).resolve().parent

R2_DEFAULTS: dict[str, str] = {
    "R2_HOME": "~/distill/round2",                  # state/, logs/, the generated confs; bin/ holds this script
    "R2_WIN_DIR": "~/distill/window",               # the mini window machinery (mini_window_*.sh, window.conf)
    "R2_WINDOW_CONF": "~/distill/window/window.conf",
    "R2_NOTIFY": "~/distill/notify.sh",
    "R2_PY": "/usr/bin/python3",
    "R2_CODE": "~/distill/round2/code",             # gen_runner.py stage-code --dest distill/round2/code
    "R2_RUNS": "~/distill/phase2b/runs",
    "R2_POOL": "~/distill/phase2b/pool",
    # the 4090 (host-specific: set in the live round2.conf)
    "R2_TOWER_SSH": "",
    "R2_TOWER_HOST": "",
    "R2_TOWER_URL": "",
    "R2_TOWER_QUANT": "UD-Q4_K_XL",
    "R2_TOWER_SCRIPT": "distill/window/round2/round2_tower.sh",   # relative to the 4090's home
    "R2_TOWER_TASKS": "~/distill/phase2b/tasksets/real-v3-excl.jsonl",
    "R2_TOWER_EXTRA_ARGS": "",
    "R2_TOWER_RUN_NAME": "",                        # empty: r2-tower-<host>-<quant>-<hash>
    "R2_TOWER_PROFILE": "standard",                 # standard | longctx | SLOTSxCTX
    "R2_STANDARD_PROFILE": "4x32768",
    "R2_LONGCTX_CTX": "",                           # explicit context; empty: from R2_TEACHER_CTX_JSON
    "R2_LONGCTX_CAP": "131072",
    "R2_LONGCTX_MIN": "32768",
    "R2_TEACHER_CTX_JSON": "~/distill/capacity/teacher-ctx.json",
    "R2_CTX_HOST_MATCH": "4090",
    "R2_CTX_QUANT_MATCH": "q4_k_xl",
    # the Sunday -> Monday chain
    "R2_PIPELINE_DONE": "~/distill/phase7/9b-r32-v4b-max-derived/PIPELINE-DONE.json",
    "R2_PIPELINE_UNIT": "",
    "R2_CAPACITY_DONE": "~/distill/capacity/CAPACITY-DONE.json",
    "R2_CAPACITY_TIMEOUT_H": "3",
    "R2_CAPACITY_PREFIX": "distill-cap-",
    "R2_SUN_LATEST_START": "2026-11-02T04:00:00-05:00",
    "R2_SUN_END": "2026-11-02T05:20:00-05:00",
    "R2_GEN_MARGIN_S": "120",
    "R2_GEN_RESTARTS": "3",
    "R2_POLL_S": "60",
    "R2_START_TIMEOUT_S": "1800",
    # nightly
    "R2_TOWER_NIGHTLY": "on",
    "R2_TOWER_NIGHTLY_FROM": "2026-11-02T06:10:00-05:00",
    "R2_TOWER_NIGHT_END": "06:22",
    "R2_MINI_NIGHTLY": "on",
    "R2_MINI_NIGHTLY_FROM": "2026-11-01T18:00:00-05:00",
    "R2_LOCAL_TASKS": "~/distill/phase2b/tasksets/real-v3-excl.jsonl",
    "R2_LOCAL_EXTRA_ARGS": "",
    "R2_LOCAL_NAME": "",                            # empty: local-r2-{alias}-<hash>
    # heartbeat
    "R2_START": "2026-11-01T01:30:00-04:00",
    "R2_TRAIN_RUN": "9b-r32-v4b-max-derived",
    "R2_TRAIN_DIR": "distill/phase5/runs",          # on the 4090, relative to its home
    "R2_POST_TRAIN_DIR": "~/distill/phase7/9b-r32-v4b-max-derived",
    "R2_DATASETS": "~/distill/phase3/build-v4b-max-derived",
    # auto: the round-2 code's gen_stats.py when it is staged in R2_CODE, else the ledgers here; or a command
    # ({py} {code} {runs} {since} {now} filled in) whose first 3 stdout lines replace ours; empty: ours
    "R2_GEN_STATS_CMD": "auto",
    "R2_HEARTBEAT_LINES": "10",
}
#: the mini's timers the heartbeat and status show (name, label)
MINI_TIMERS = (("window-mini-night-start", "mini night start"), ("window-mini-night-end", "mini night end"),
               ("r2-tower-night", "4090 r2 night"), ("r2-heartbeat", "heartbeat"),
               ("window-mini-window-end-20261101", "mini cont end"), ("distill-deadman-mini", "mini dead-man"))
TOWER_TIMERS = (("flywheel-r2-end", "4090 r2 end"), ("flywheel-window-start", "4090 window start"),
                ("flywheel-window-end", "4090 window end"), ("flywheel-window-cont-end", "4090 cont end"),
                ("distill-deadman-tower-window", "4090 dead-man"), ("flywheel-window-rearm", "4090 rearm"))
ACTIVE = ("active", "activating", "deactivating", "reloading")
FINAL_PHASES = ("done", "skipped", "failed")
CTX_KEY = re.compile(r"max.*(ctx|context)|(ctx|context).*max|stable", re.I)
CTX_KEY_BAD = re.compile(r"fail|tried|attempt|oom|unstable|request", re.I)
PROFILE_RE = re.compile(r"^(\d)x(\d+)$")


class Refused(Exception):
    """A precondition is not met: nothing was changed."""


# ── settings ─────────────────────────────────────────────────────────────────


def expand_home(value: str, home: str) -> str:
    value = value.replace("${HOME}", home).replace("$HOME", home)
    return " ".join(home + w[1:] if w == "~" or w.startswith("~/") else w for w in value.split(" ")) if value else value


def parse_kv(text: str, origin: str = "conf") -> dict[str, str]:
    """KEY=VALUE lines (shell-style: bash can source the same file); comments and blank lines skipped."""
    out: dict[str, str] = {}
    for n, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, eq, value = line.partition("=")
        if not eq or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise SystemExit(f"{origin}:{n}: not KEY=VALUE")
        try:
            out[key] = " ".join(shlex.split(value, comments=True))
        except ValueError as e:
            raise SystemExit(f"{origin}:{n}: {e}") from None
    return out


def load_conf(path: Path, home: str | None = None) -> dict[str, str]:
    home = home or os.path.expanduser("~")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}") from None
    conf = dict(R2_DEFAULTS)
    conf.update(parse_kv(text, str(path)))
    return {k: expand_home(v, home) for k, v in conf.items()}


def when(text: str) -> float:
    """An ISO time (a naive one is ET) as an epoch."""
    dt = datetime.fromisoformat(text.strip())
    return (dt if dt.tzinfo else dt.replace(tzinfo=ET)).timestamp()


def et(t: float, fmt: str = "%a %H:%M %Z") -> str:
    return datetime.fromtimestamp(t, ET).strftime(fmt)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, ET).isoformat(timespec="seconds")


def utc_cal(t: float) -> str:
    """systemd OnCalendar / date -d form."""
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def today_at(hhmm: str, now: float) -> float:
    """HH:MM ET on today's ET date (ZoneInfo: the offset that day has, DST included)."""
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime.fromtimestamp(now, ET).replace(hour=h, minute=m, second=0, microsecond=0).timestamp()


def on(v: str) -> bool:
    return v.strip().lower() in ("1", "on", "yes", "true")


# ── the 4090's serve profile ─────────────────────────────────────────────────


def parse_profile(text: str) -> tuple[int, int] | None:
    """``SLOTSxCTX`` (1..8 slots; ctx a multiple of 1024, at least 8192: serve_teacher.sh's own limits)."""
    m = PROFILE_RE.match(text.strip().lower())
    if not m:
        return None
    s, c = int(m.group(1)), int(m.group(2))
    if not 1 <= s <= 8 or c < 8192 or c % 1024:
        return None
    return s, c


def measured_ctx(data: Any, host_match: str, quant_match: str) -> int | None:
    """The capacity block's max stable context for this host and quant, from teacher-ctx.json of any shape:
    every number under a key like ``max_stable_ctx`` / ``max_context`` / ``stable_ctx`` whose path (keys, and
    the string values of its own and its parents' objects: ``host``, ``quant``) mentions both the host and the
    quant. Keys naming a failure or an attempt are ignored; ``stable`` keys win over the others; the largest."""
    host_match, quant_match = host_match.lower(), quant_match.lower()
    best: dict[bool, int] = {}

    def walk(node: Any, ctx: list[str]) -> None:
        if isinstance(node, dict):
            here = ctx + [str(v) for v in node.values() if isinstance(v, str)]
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, here + [str(k)])
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    key = str(k)
                    if not CTX_KEY.search(key) or CTX_KEY_BAD.search(key) or v < 8192:
                        continue
                    text = " ".join(here + [key]).lower()
                    if host_match in text and quant_match in text:
                        stable = "stable" in key.lower()
                        best[stable] = max(best.get(stable, 0), int(v))
        elif isinstance(node, list):
            for v in node:
                walk(v, ctx)

    walk(data, [])
    if True in best:
        return best[True]
    return best.get(False)


def resolve_profiles(c: dict[str, str], read: Callable[[Path], str] | None = None) -> tuple[list[str], str]:
    """The serve profiles the 4090 tries, in order (the standard one last as the fallback), and why."""
    read = read or (lambda p: p.read_text(encoding="utf-8"))
    std = c["R2_STANDARD_PROFILE"].strip()
    if not parse_profile(std):
        raise SystemExit(f"R2_STANDARD_PROFILE {std!r} is not SLOTSxCTX")
    want = c["R2_TOWER_PROFILE"].strip().lower() or "standard"
    if want == "standard":
        return [std], f"standard {std}"
    if want == "longctx":
        raw: int | None = None
        src = ""
        if c["R2_LONGCTX_CTX"].strip():
            raw, src = int(c["R2_LONGCTX_CTX"]), "R2_LONGCTX_CTX"
        else:
            path = Path(c["R2_TEACHER_CTX_JSON"])
            try:
                raw = measured_ctx(json.loads(read(path)), c["R2_CTX_HOST_MATCH"], c["R2_CTX_QUANT_MATCH"])
                src = path.name
            except (OSError, ValueError) as e:
                return [std], f"longctx asked, {path.name} unreadable ({type(e).__name__}): standard {std}"
            if raw is None:
                return [std], f"longctx asked, {path.name} has no stable context for this host/quant: standard {std}"
        ctx = min(raw, int(c["R2_LONGCTX_CAP"])) // 1024 * 1024
        if ctx < int(c["R2_LONGCTX_MIN"]):
            return [std], f"longctx asked, measured {raw} < R2_LONGCTX_MIN: standard {std}"
        prof = f"1x{ctx}"
        return ([prof, std] if prof != std else [std]), f"longctx {prof} (measured {raw} from {src}), fallback {std}"
    p = parse_profile(want)
    if not p:
        return [std], f"R2_TOWER_PROFILE {want!r} is not standard/longctx/SLOTSxCTX: standard {std}"
    prof = f"{p[0]}x{p[1]}"
    return ([prof, std] if prof != std else [std]), f"{prof}, fallback {std}"


# ── decisions (pure) ─────────────────────────────────────────────────────────


def pipeline_verdict(marker: dict[str, Any] | None, unit_state: str | None, now: float,
                     latest: float) -> tuple[str, str]:
    """go / stop / wait. ``unit_state`` is read BEFORE the marker (the pipeline writes the marker, then exits):
    a unit that has ended with no marker failed or was stopped by hand. None: no unit is watched."""
    if marker is not None:
        outcome = str(marker.get("outcome") or "")
        if outcome.startswith("completed"):
            return "go", f"post-train {outcome}"
        return "stop", f"post-train did not complete ({outcome or 'no outcome'})"
    if unit_state is not None and unit_state not in ACTIVE:
        return "stop", f"the post-train unit ended ({unit_state}) without PIPELINE-DONE.json"
    if now >= latest:
        return "stop", f"post-train not done by {et(latest)}"
    return "wait", "post-train still running"


def capacity_verdict(done: bool, pipeline_end: float, now: float, timeout_s: float,
                     latest: float) -> tuple[str, str]:
    """go / timeout (go anyway, the block's units preempted) / stop / wait."""
    if done:
        return "go", "CAPACITY-DONE.json"
    if now >= latest:
        return "stop", f"the capacity block is not done and it is past {et(latest)}"
    if now >= pipeline_end + timeout_s:
        return "timeout", (f"no CAPACITY-DONE.json {timeout_s / 3600:g} h after the pipeline ended "
                           f"({et(pipeline_end)}): going on anyway")
    return "wait", "capacity block still running"


def marker_end(marker: dict[str, Any], mtime: float) -> float:
    try:
        return when(str(marker.get("finished")))
    except (TypeError, ValueError):
        return mtime


def run_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:8]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._").lower() or "unknown"


def code_commit(c: dict[str, str]) -> str:
    try:
        return (Path(c["R2_CODE"]) / "COMMIT").read_text().strip()[:12] or "nocode"
    except OSError:
        return "nocode"


def tower_run_name(c: dict[str, str]) -> str:
    if c["R2_TOWER_RUN_NAME"].strip():
        return slug(c["R2_TOWER_RUN_NAME"])
    h = run_hash(code_commit(c), c["R2_TOWER_TASKS"], c["R2_TOWER_EXTRA_ARGS"])
    return slug(f"r2-tower-{c['R2_TOWER_HOST'] or 'tower'}-{c['R2_TOWER_QUANT']}-{h}")


def local_run_name(c: dict[str, str]) -> str:
    if c["R2_LOCAL_NAME"].strip():
        return c["R2_LOCAL_NAME"].strip()
    return f"local-r2-{{alias}}-{run_hash(code_commit(c), c['R2_LOCAL_TASKS'], c['R2_LOCAL_EXTRA_ARGS'])}"


def remote_overrides(c: dict[str, str], until: float) -> dict[str, str]:
    """window.conf keys for ``gen_runner.py remote`` against the 4090 (plus every GEN_* key of round2.conf)."""
    runs = c["R2_RUNS"]
    skips = f"--skip-done-in {runs}/r2-tower-*/*.tasks.jsonl --skip-done-in {runs}/remote-*/*.tasks.jsonl"
    o = {"GEN_RUNS": runs, "GEN_POOL": c["R2_POOL"], "GEN_REMOTE_CODE": c["R2_CODE"],
         "GEN_REMOTE_URL": c["R2_TOWER_URL"], "GEN_REMOTE_HOST": c["R2_TOWER_HOST"],
         "GEN_REMOTE_SSH": c["R2_TOWER_SSH"], "GEN_REMOTE_QUANT": c["R2_TOWER_QUANT"],
         "GEN_REMOTE_TASKS": c["R2_TOWER_TASKS"], "GEN_REMOTE_NAME": tower_run_name(c),
         "GEN_REMOTE_UNTIL": iso(until), "GEN_REMOTE_SKIP_DONE_IN": f"{runs}/local-*/*.tasks.jsonl",
         "GEN_REMOTE_EXTRA_ARGS": f"{skips} {c['R2_TOWER_EXTRA_ARGS']}".strip()}
    o.update({k: v for k, v in c.items() if k.startswith("GEN_")})
    return o


def local_overrides(c: dict[str, str]) -> dict[str, str]:
    """window.conf keys for ``gen_runner.py run`` in the mini's own window (plus every GEN_* key of round2.conf)."""
    runs = c["R2_RUNS"]
    skips = f"--skip-done-in {runs}/r2-tower-*/*.tasks.jsonl --skip-done-in {runs}/local-*/*.tasks.jsonl"
    o = {"GEN_RUNS": runs, "GEN_POOL": c["R2_POOL"], "GEN_CODE": c["R2_CODE"], "GEN_TASKS": c["R2_LOCAL_TASKS"],
         "GEN_LOCAL_NAME": local_run_name(c),
         "GEN_LOCAL_EXTRA_ARGS": f"{skips} {c['R2_LOCAL_EXTRA_ARGS']}".strip()}
    o.update({k: v for k, v in c.items() if k.startswith("GEN_")})
    return o


def derived_conf(base_text: str, overrides: dict[str, str], note: str) -> str:
    """window.conf as it is, then the overrides (load_conf: a later line wins), each value quoted for shlex."""
    lines = [base_text.rstrip("\n"), "", f"# --- round 2 ({note}); generated by round2_chain.py, do not edit ---"]
    lines += [f"{k}={shlex.quote(v)}" for k, v in overrides.items()]
    return "\n".join(lines) + "\n"


def parse_status(text: str) -> dict[str, str]:
    """``round2_tower.sh status`` key=value lines (``timer <name>=<epoch|->`` too)."""
    out: dict[str, str] = {}
    for line in text.split("\n"):
        m = re.match(r"^([a-z_]+(?: [A-Za-z0-9_.@-]+)?)=(.*)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


# ── generation stats ─────────────────────────────────────────────────────────


def ledger_events(paths: list[Path]) -> Iterator[tuple[str, dict[str, Any]]]:
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.split("\n"):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict):
                yield p.parent.name, ev


def run_group(run_dir: str) -> str:
    if run_dir.startswith(("r2-tower-", "remote-")):
        return "4090"
    if run_dir.startswith("local-"):
        return "3090Ti"
    if run_dir.startswith("max-"):
        return "Max"
    return "other"


def gen_stats(events: list[tuple[str, dict[str, Any]]], since: float, until: float) -> dict[str, Any]:
    """Verified rows (ledger status ``kept``, or ``released`` after a hold: the held event's rows) per kind,
    per domain and per host group in [since, until)."""
    by_kind: Counter[str] = Counter()
    by_domain: Counter[str] = Counter()
    by_group: Counter[str] = Counter()
    held: dict[tuple[str, str], int] = {}
    total = 0
    for run_dir, ev in events:
        try:
            ts = float(ev.get("ts") or 0)
            rows = int(ev.get("rows") or 0)
        except (TypeError, ValueError):
            continue
        key = (run_dir, str(ev.get("task_id")))
        if ev.get("status") == "held":
            held[key] = rows
        if ev.get("status") == "released":
            rows = held.get(key, 0)
        if ev.get("status") not in ("kept", "released") or rows <= 0 or not since <= ts < until:
            continue
        total += rows
        by_kind[str(ev.get("kind") or "?")] += rows
        by_domain[str(ev.get("domain") or "?")] += rows
        by_group[run_group(run_dir)] += rows
    hours = max((until - since) / 3600, 1e-9)
    return {"rows": total, "hours": hours, "per_h": total / hours, "kind": by_kind, "domain": by_domain,
            "group": by_group}


def rates(counter: Counter[str], hours: float, limit: int = 6) -> str:
    items = counter.most_common(limit)
    return ", ".join(f"{k} {v / hours:.1f}" for k, v in items) or "-"


def stats_line(label: str, s: dict[str, Any]) -> str:
    shown = ["3090Ti", "4090"] + [g for g in ("Max", "other") if s["group"].get(g)]
    groups = " ".join(f"{g} {s['group'].get(g, 0) / s['hours']:.1f}" for g in shown)
    return (f"{label}: {s['rows']} rows, {s['per_h']:.1f}/h ({groups}) | by kind/h: "
            f"{rates(s['kind'], s['hours'])}")


def domain_line(s: dict[str, Any]) -> str:
    return f"24h by area/h: {rates(s['domain'], s['hours'])}"


def clip(lines: list[str], n: int, width: int = 300) -> list[str]:
    return [ln if len(ln) <= width else ln[: width - 3] + "..." for ln in lines[:n]]


# ── the runtime ──────────────────────────────────────────────────────────────


def _run(argv: list[str], timeout: float = 120, input: str | None = None,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=input, env=env)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(argv, 124, "", f"timeout: {e}")
    except OSError as e:
        return subprocess.CompletedProcess(argv, 127, "", str(e))


class Ctx:
    def __init__(self, conf: dict[str, str], execute: bool, cmd: str, *,
                 clock: Callable[[], float] = time.time, run: Callable[..., Any] = _run,
                 sleep: Callable[[float], None] = time.sleep, out: Callable[[str], None] = print) -> None:
        self.c = conf
        self.execute = execute
        self.cmd = cmd
        self.now = clock
        self.run = run
        self.sleep = sleep
        self.out = out
        self.home = Path(conf["R2_HOME"])
        self.state_dir = self.home / "state"
        self.log_dir = self.home / "logs"
        self.log_file: Path | None = None
        if execute:
            for d in (self.home, self.state_dir, self.log_dir):
                d.mkdir(parents=True, exist_ok=True)
                os.chmod(d, 0o700)
            self.log_file = self.log_dir / f"{cmd}-{et(clock(), '%F')}.log"

    def log(self, msg: str) -> None:
        line = f"{et(self.now(), '%F %T %Z')} {self.cmd}: {msg}"
        self.out(line)
        if self.log_file is not None:
            with contextlib.suppress(OSError), open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def act(self, argv: list[str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
        if not self.execute:
            self.log("[dry run] would run: " + shlex.join(argv))
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.log("run: " + shlex.join(argv))
        return self.run(argv, timeout=timeout)

    # state

    def load_state(self, name: str) -> dict[str, Any]:
        try:
            return json.loads((self.state_dir / name).read_text())
        except (OSError, ValueError):
            return {}

    def save_state(self, name: str, st: dict[str, Any]) -> None:
        st["updated_at"] = iso(self.now())
        if not self.execute:
            return
        tmp = self.state_dir / f".{name}.{os.getpid()}"
        tmp.write_text(json.dumps(st, indent=1) + "\n")
        os.replace(tmp, self.state_dir / name)

    @contextlib.contextmanager
    def lock(self, name: str) -> Iterator[bool]:
        if not self.execute:
            yield True
            return
        fd = os.open(self.state_dir / name, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                yield False
                return
            yield True
        finally:
            os.close(fd)

    # units here

    def unit_state(self, unit: str) -> str:
        cp = self.run(["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState"], timeout=30)
        kv = dict(ln.split("=", 1) for ln in (cp.stdout or "").split("\n") if "=" in ln)
        if kv.get("LoadState") == "not-found":
            return "gone"
        return kv.get("ActiveState", "") or "unknown"

    def unit_result(self, unit: str) -> tuple[str, str]:
        cp = self.run(["systemctl", "--user", "show", unit, "-p", "Result", "-p", "ExecMainStatus"], timeout=30)
        kv = dict(ln.split("=", 1) for ln in (cp.stdout or "").split("\n") if "=" in ln)
        return kv.get("Result", ""), kv.get("ExecMainStatus", "")

    def timer_next(self, timer: str) -> float | None:
        cp = self.run(["systemctl", "--user", "show", f"{timer}.timer", "-p", "ActiveState",
                       "-p", "NextElapseUSecRealtime"], timeout=30)
        kv = dict(ln.split("=", 1) for ln in (cp.stdout or "").split("\n") if "=" in ln)
        nxt = kv.get("NextElapseUSecRealtime", "")
        if kv.get("ActiveState") != "active" or not nxt:
            return None
        cp = self.run(["date", "-d", nxt, "+%s"], timeout=10)
        try:
            return float(cp.stdout.strip())
        except ValueError:
            return None

    def notify(self, lines: list[str], n: int = 5) -> None:
        text = "\n".join(clip(lines, n))
        if not self.execute:
            self.log("[dry run] would notify:\n" + "\n".join("    | " + ln for ln in text.split("\n")))
            return
        cp = self.run([self.c["R2_NOTIFY"], str(n)], timeout=60, input=text + "\n")
        self.log(f"notify {'sent' if cp.returncode == 0 else 'FAILED (exit %d)' % cp.returncode}: "
                 + text.split("\n")[0][:120])

    # the 4090

    def ssh(self, remote: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        target = self.c["R2_TOWER_SSH"]
        if not target:
            raise Refused("R2_TOWER_SSH is not set (round2.conf)")
        return self.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote], timeout=timeout)

    def tower_script(self) -> str:
        return '"$HOME"/' + shlex.quote(self.c["R2_TOWER_SCRIPT"])

    def tower_status(self) -> dict[str, str]:
        cp = self.ssh(f"/bin/bash {self.tower_script()} status", timeout=60)
        st = parse_status(cp.stdout or "")
        st["_rc"] = str(cp.returncode)
        return st

    def tower_run(self, what: str, args: list[str], wait_s: float) -> tuple[int, dict[str, Any]]:
        """Run ``round2_tower.sh <what> <args>`` on the 4090 as a transient unit there (so a lost ssh link does not
        stop it), wait for it to end, and return (exit, state/r2-start.json for a start)."""
        stamp = datetime.fromtimestamp(self.now(), ET).strftime("%Y%m%d-%H%M%S")
        if not self.execute:
            cp = self.ssh(f"/bin/bash {self.tower_script()} {what} {shlex.join(args)} --dry-run", timeout=600)
            for line in (cp.stdout + cp.stderr).rstrip().split("\n"):
                self.log(f"    4090| {line}")
            return cp.returncode, {"status": "dry-run"}
        unit = f"flywheel-r2-{what}-{stamp}"
        remote = (f"systemd-run --user --unit={unit} --no-block -p KillMode=mixed -p TimeoutStopSec=30min "
                  f"--expand-environment=no --description={shlex.quote('flywheel round 2: ' + what)} "
                  f"/bin/bash {self.tower_script()} {what} {shlex.join(args)}")
        self.log(f"4090: {unit}: round2_tower.sh {what} {shlex.join(args)}")
        cp = self.ssh(remote, timeout=60)
        if cp.returncode != 0:
            self.log(f"4090: systemd-run exited {cp.returncode}: {(cp.stderr or cp.stdout).strip()[:300]}")
            return 1, {"status": "failed", "why": "systemd-run on the 4090 failed"}
        deadline = self.now() + wait_s
        state = ""
        while self.now() < deadline:
            self.sleep(15)
            q = self.ssh(f"systemctl --user show {unit} -p ActiveState -p ExecMainStatus -p Result", timeout=60)
            kv = dict(ln.split("=", 1) for ln in (q.stdout or "").split("\n") if "=" in ln)
            state = kv.get("ActiveState", "")
            if q.returncode == 0 and state and state not in ACTIVE:
                code = kv.get("ExecMainStatus", "")
                self.ssh(f"systemctl --user reset-failed {unit} 2>/dev/null; true", timeout=30)
                res: dict[str, Any] = {}
                if what == "start":
                    r = self.ssh("cat ~/distill/window/state/r2-start.json", timeout=30)
                    with contextlib.suppress(ValueError):
                        res = json.loads(r.stdout or "{}")
                self.log(f"4090: {unit} ended ({state}, exit {code or '?'}): {json.dumps(res)[:300]}")
                return (int(code) if code.isdigit() else 1), res
        self.log(f"4090: {unit} still '{state}' after {wait_s:.0f} s")
        return 1, {"status": "failed", "why": f"{unit} did not end in {wait_s:.0f} s"}

    # generation here

    def write_conf(self, name: str, overrides: dict[str, str], note: str) -> Path:
        """window.conf + the round-2 overrides, as ``R2_HOME/<name>`` (a dry run writes a private temp copy, so
        the window scripts' own dry runs read exactly what a real run would)."""
        base = Path(self.c["R2_WINDOW_CONF"]).read_text(encoding="utf-8")
        text = derived_conf(base, overrides, note)
        if self.execute:
            path = self.home / name
        else:
            path = Path(tempfile.mkdtemp(prefix="round2-dryrun-")) / name
            self.log(f"[dry run] {name} -> {path}: " + "; ".join(f"{k}={v}" for k, v in overrides.items())[:900])
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return path

    def gen_runner(self) -> Path:
        return Path(self.c["R2_CODE"]) / "scripts" / "distill" / "mini_window" / "gen_runner.sh"

    def start_remote_gen(self, until: float, tag: str) -> str | None:
        """The remote generation unit (dry run first; refused -> None)."""
        runner = self.gen_runner()
        if not os.access(runner, os.X_OK):
            self.log(f"no round-2 code at {runner} (gen_runner.py stage-code --dest distill/round2/code)")
            return None
        conf = self.write_conf("gen-remote.conf", remote_overrides(self.c, until), f"remote, until {iso(until)}")
        env = {**os.environ, "DISTILL_WINDOW_CONF": str(conf), "DISTILL_WINDOW_PY": self.c["R2_PY"]}
        # gen_runner's own dry run (its preflight: code, key file, bwrap, taskset exclusion check) first
        cp = self.run([str(runner), "remote", "--until", iso(until)], timeout=1200, env=env)
        for line in ((cp.stdout or "") + (cp.stderr or "")).rstrip().split("\n")[-8:]:
            self.log(f"    gen dry run| {line}")
        if cp.returncode != 0:
            self.log(f"the remote generation's own dry run refused (exit {cp.returncode}): not started")
            return None
        stamp = datetime.fromtimestamp(self.now(), ET).strftime("%Y%m%d-%H%M%S")
        unit = f"r2-gen-tower-{tag}-{stamp}"
        left = max(600, int(until - self.now()) + 900)
        cp = self.act(["systemd-run", "--user", f"--unit={unit}", "-p", "KillMode=mixed", "-p", "TimeoutStopSec=150",
                       "-p", f"RuntimeMaxSec={left}", "-p", f"WorkingDirectory={Path(self.c['R2_HOME']).parent}",
                       f"--setenv=DISTILL_WINDOW_CONF={conf}", f"--setenv=DISTILL_WINDOW_PY={self.c['R2_PY']}",
                       f"--description=distill round 2: remote generation against the 4090 ({tag})",
                       str(runner), "remote", "--execute", "--until", iso(until)], timeout=60)
        if self.execute and cp.returncode != 0:
            self.log(f"systemd-run {unit} exited {cp.returncode}: {(cp.stderr or cp.stdout).strip()[:300]}")
            return None
        self.log(f"remote generation {unit} until {et(until)} into {self.c['R2_RUNS']}/{tower_run_name(self.c)}")
        return unit

    def remote_progress(self) -> dict[str, Any]:
        try:
            return json.loads((Path(self.c["R2_RUNS"]) / "state" / "remote-progress.json").read_text())
        except (OSError, ValueError):
            return {}


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_config(cx: Ctx) -> int:
    c = cx.c
    profiles, why = resolve_profiles(c)
    cx.out(f"serve profile: {' then '.join(profiles)} ({why})")
    cx.out(f"code: {c['R2_CODE']} @ {code_commit(c)}")
    cx.out(f"4090 run: {c['R2_RUNS']}/{tower_run_name(c)} (tasks {c['R2_TOWER_TASKS']})")
    cx.out(f"local run: {c['R2_RUNS']}/{local_run_name(c)} (tasks {c['R2_LOCAL_TASKS']})")
    for k in sorted(c):
        if k.startswith(("R2_", "GEN_")):
            cx.out(f"  {k}={c[k]}")
    return 0


def tower_start(cx: Ctx, mode: str, end: float, preempt: bool) -> tuple[str, dict[str, Any]]:
    """ok / refused / failed, and the 4090's result."""
    profiles, why = resolve_profiles(cx.c)
    cx.log(f"serve profile: {' then '.join(profiles)} ({why})")
    req = f"{mode}-{datetime.fromtimestamp(cx.now(), ET).strftime('%Y%m%dT%H%M%S')}"
    args = ["--mode", mode, "--profiles", " ".join(profiles), "--end-utc", utc_cal(end), "--req", req]
    if preempt:
        args += ["--preempt-prefix", cx.c["R2_CAPACITY_PREFIX"]]
    rc, res = cx.tower_run("start", args, float(cx.c["R2_START_TIMEOUT_S"]))
    if not cx.execute:
        return ("ok" if rc == 0 else "refused"), res
    if res.get("req") != req:
        return "failed", {"why": f"no result for request {req} (exit {rc})"}
    status = str(res.get("status") or "failed")
    return (status if status in ("ok", "refused") else "failed"), res


def cmd_sunday(cx: Ctx) -> int:
    with cx.lock("tower.lock") as got:
        if not got:
            cx.log("another round-2 4090 command holds state/tower.lock")
            return 2
        return sunday(cx)


def sunday(cx: Ctx) -> int:
    c = cx.c
    name = "sunday.json"
    st = cx.load_state(name) or {"phase": "wait-pipeline", "created_at": iso(cx.now())}
    if st.get("phase") in FINAL_PHASES and cx.execute:
        cx.log(f"already {st['phase']}: {st.get('why', '')}")
        return 0
    latest, end = when(c["R2_SUN_LATEST_START"]), when(c["R2_SUN_END"])
    gen_until = end - float(c["R2_GEN_MARGIN_S"])
    poll = float(c["R2_POLL_S"])
    marker_path = Path(c["R2_PIPELINE_DONE"])
    cx.log(f"chain: post-train -> capacity -> 4090 teacher + remote generation until {et(end)} "
           f"(latest start {et(latest)}); phase {st['phase']}")

    def finish(phase: str, why: str) -> int:
        st.update(phase=phase, why=why, ended_at=iso(cx.now()))
        cx.save_state(name, st)
        cx.log(f"{phase}: {why}")
        return 0

    while st["phase"] == "wait-pipeline":
        unit = c["R2_PIPELINE_UNIT"].strip()
        ustate = cx.unit_state(unit) if unit else None
        marker = None
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text())
            except (OSError, ValueError):
                marker = None  # being written: next poll
        v, why = pipeline_verdict(marker, ustate, cx.now(), latest)
        if v == "go":
            st.update(phase="wait-capacity", pipeline=why,
                      pipeline_end=marker_end(marker or {}, marker_path.stat().st_mtime))
            cx.save_state(name, st)
            cx.log(f"{why}; ended {et(st['pipeline_end'])}")
        elif v == "stop":
            return finish("skipped", f"nothing done: {why}")
        else:
            if not cx.execute:
                cx.log(f"[dry run] {why} (unit {unit or '-'}: {ustate}); a real run polls every {poll:.0f} s")
                st.update(phase="wait-capacity", pipeline_end=cx.now())
                break
            cx.save_state(name, dict(st, waiting=why))
            cx.sleep(poll)

    while st["phase"] == "wait-capacity":
        done = Path(c["R2_CAPACITY_DONE"]).is_file()
        v, why = capacity_verdict(done, float(st["pipeline_end"]), cx.now(),
                                  3600 * float(c["R2_CAPACITY_TIMEOUT_H"]), latest)
        if v in ("go", "timeout"):
            st.update(phase="tower-start", capacity=why, preempt=(v == "timeout"))
            cx.save_state(name, st)
            cx.log(f"capacity: {why}")
        elif v == "stop":
            return finish("skipped", f"nothing done: {why}")
        else:
            if not cx.execute:
                cx.log(f"[dry run] {why}; a real run polls every {poll:.0f} s, up to "
                       f"{c['R2_CAPACITY_TIMEOUT_H']} h after the pipeline")
                st.update(phase="tower-start", preempt=False)
                break
            cx.save_state(name, dict(st, waiting=why))
            cx.sleep(poll)

    while st["phase"] == "tower-start":
        if cx.now() >= latest:
            return finish("skipped", f"nothing done: past {et(latest)} before the 4090 teacher started")
        status, res = tower_start(cx, "cont", end, bool(st.get("preempt")))
        if status == "ok":
            st.update(phase="generating", tower=res)
            cx.save_state(name, st)
        elif status == "refused" and cx.execute and cx.now() + 300 < latest:
            cx.log(f"4090 refused ({res.get('why')}): again in 5 min")
            cx.save_state(name, dict(st, waiting=f"4090 refused: {res.get('why')}"))
            cx.sleep(300)
        else:
            if not cx.execute:
                cx.log(f"[dry run] 4090 start dry run exit status '{status}'")
                st["phase"] = "generating"
                break
            return finish("failed", f"the 4090 teacher did not start: {res.get('why')}")

    if st["phase"] == "generating":
        return sunday_generate(cx, st, name, gen_until, end)
    return 0


def gen_watch(cx: Ctx, st: dict[str, Any], name: str, gen_until: float, end: float, tag: str) -> str:
    """Keeps the remote generation unit going until ``end`` (+3 min: the 4090's end timer runs first). Returns
    why it stopped watching: "end", or "early: ..." when generation is over before its time (nothing left, or
    it could not be (re)started ``R2_GEN_RESTARTS`` times): then the 4090 window should end now, not idle."""
    c = cx.c
    restarts = int(st.get("gen_restarts", 0))
    while cx.now() < end + 180:
        unit = st.get("gen_unit")
        ustate = cx.unit_state(unit) if unit else "none"
        if ustate not in ACTIVE and cx.now() < gen_until - 900:
            reason = str(cx.remote_progress().get("stop_reason") or "") if unit else ""
            result, code = cx.unit_result(unit) if unit else ("", "")
            if reason == "nothing left to do":
                return "early: remote generation finished (nothing left to do)"
            if restarts >= int(c["R2_GEN_RESTARTS"]):
                return f"early: remote generation is down ({unit or 'not started'}, {restarts} restarts)"
            restarts += 1
            cx.log(f"remote generation {unit or '(not started)'} is '{ustate}' (result {result or '-'}, exit "
                   f"{code or '-'}, {reason or '-'}): restart {restarts} in 5 min")
            if unit:
                cx.run(["systemctl", "--user", "reset-failed", unit], timeout=30)
            cx.sleep(300)
            st.update(gen_unit=cx.start_remote_gen(gen_until, tag), gen_restarts=restarts)
            cx.save_state(name, st)
            continue
        cx.sleep(min(float(c["R2_POLL_S"]) * 5, max(5.0, end + 180 - cx.now())))
    return "end"


def gen_close(cx: Ctx, st: dict[str, Any], why: str) -> None:
    """Stop the generation unit; end the 4090 window from here if its own end timer has not (or not yet)."""
    unit = st.get("gen_unit")
    if unit and cx.unit_state(unit) in ACTIVE:
        cx.act(["systemctl", "--user", "stop", unit], timeout=200)
    tw: dict[str, str] = {}
    with contextlib.suppress(Exception):
        tw = cx.tower_status()
    if tw.get("r2_units", "").strip() or tw.get("jobs_unit", "").startswith("distill-jobs-r2teacher@"):
        cx.log(f"the 4090's round-2 teacher is still up ({why}): running its end from here")
        rc, _ = cx.tower_run("end", [], 2400)
        cx.log(f"4090 end exit {rc}")
        with contextlib.suppress(Exception):
            tw = cx.tower_status()
    st["tower_end"] = tw.get("end_last", "")


def sunday_generate(cx: Ctx, st: dict[str, Any], name: str, gen_until: float, end: float) -> int:
    if not st.get("gen_unit") or cx.unit_state(st["gen_unit"]) not in ACTIVE:
        if cx.now() < gen_until - 600:
            st["gen_unit"] = cx.start_remote_gen(gen_until, "sun")
            cx.save_state(name, st)
    if not cx.execute:
        cx.log(f"[dry run] then: watch generation until {et(end)} (restart up to {cx.c['R2_GEN_RESTARTS']}x; "
               "finished early -> end the 4090 window early); stop it; the 4090's flywheel-r2-end restores "
               "business (from here if it did not run)")
        return 0
    why = gen_watch(cx, st, name, gen_until, end, "sun")
    cx.log(f"generation over: {why}")
    gen_close(cx, st, why)
    st.update(phase="done", why=why, ended_at=iso(cx.now()))
    cx.save_state(name, st)
    cx.log("done")
    return 0


def cmd_tower_night(cx: Ctx) -> int:
    c = cx.c
    now = cx.now()
    if not on(c["R2_TOWER_NIGHTLY"]):
        cx.log("R2_TOWER_NIGHTLY is off: the window machinery's own window_start runs the 4090 window")
        return 0
    if now < when(c["R2_TOWER_NIGHTLY_FROM"]):
        cx.log(f"before R2_TOWER_NIGHTLY_FROM ({et(when(c['R2_TOWER_NIGHTLY_FROM']))}): the continuous window "
               "and the Sunday chain own the 4090")
        return 0
    end = today_at(c["R2_TOWER_NIGHT_END"], now)
    if now > end - 3600:
        cx.log(f"less than an hour before {et(end)}: no nightly start")
        return 0
    sun = cx.load_state("sunday.json")
    if sun.get("phase") and sun["phase"] not in FINAL_PHASES:
        cx.log(f"the Sunday chain is still '{sun['phase']}': no nightly start")
        return 0
    with cx.lock("tower.lock") as got:
        if not got:
            cx.log("another round-2 4090 command holds state/tower.lock")
            return 2
        status, res = tower_start(cx, "night", end, False)
        if status != "ok":
            cx.log(f"4090 nightly teacher not started ({status}: {res.get('why')}); the window machinery's own "
                   "01:20 window_start runs its job queue instead")
            return 0 if status == "refused" or not cx.execute else 1
        gen_until = end - float(c["R2_GEN_MARGIN_S"])
        st: dict[str, Any] = {"date": et(now, "%F"), "phase": "generating", "tower": res,
                              "gen_unit": cx.start_remote_gen(gen_until, "night")}
        cx.save_state("night.json", st)
        if not cx.execute:
            cx.log(f"[dry run] then: watch generation until {et(end)}; stop it; the 4090's flywheel-r2-end "
                   "restores business (from here if it did not run)")
            return 0
        why = gen_watch(cx, st, "night.json", gen_until, end, "night")
        cx.log(f"generation over: {why}")
        gen_close(cx, st, why)
        st.update(phase="done", why=why, ended_at=iso(cx.now()))
        cx.save_state("night.json", st)
        return 0


def cmd_mini_night(cx: Ctx, which: str) -> int:
    c = cx.c
    now = cx.now()
    if now < when(c["R2_MINI_NIGHTLY_FROM"]):
        cx.log(f"before R2_MINI_NIGHTLY_FROM ({et(when(c['R2_MINI_NIGHTLY_FROM']))}): the continuous window owns "
               "the 3090 Ti")
        return 0
    win = Path(c["R2_WIN_DIR"])
    argv = [str(win / f"mini_window_{which}.sh")] + (["--execute"] if cx.execute else [])
    env = dict(os.environ)
    if which == "start":
        if not on(c["R2_MINI_NIGHTLY"]):
            cx.log("R2_MINI_NIGHTLY is off: no 3090 Ti nightly window")
            return 0
        runner = HERE / "gen_local.sh"
        overrides = {"GEN_RUNNER": str(runner)} if os.access(runner, os.X_OK) else {}
        if not overrides:
            cx.log(f"no {runner}: the window's own generation runner is used")
        conf = cx.write_conf("window-night.conf", overrides, "nightly window start")
        env["DISTILL_WINDOW_CONF"] = str(conf)
    cx.log("exec: " + shlex.join(argv))
    sys.stdout.flush()
    os.execve(argv[0], argv, env)
    return 0  # not reached


def cmd_gen_local(cx: Ctx, rest: list[str]) -> int:
    c = cx.c
    runner = cx.gen_runner()
    env = dict(os.environ)
    if os.access(runner, os.X_OK):
        conf = cx.write_conf("gen-local.conf", local_overrides(c), "local window generation")
        env["DISTILL_WINDOW_CONF"] = str(conf)
    else:
        runner = Path(c["R2_WIN_DIR"]) / "gen_runner.sh"
        env["DISTILL_WINDOW_CONF"] = c["R2_WINDOW_CONF"]
        cx.log(f"no round-2 code staged: the window's own runner {runner}")
    argv = [str(runner)] + rest
    cx.log("exec: " + shlex.join(argv))
    sys.stdout.flush()
    os.execve(argv[0], argv, env)
    return 0  # not reached


# ── heartbeat ────────────────────────────────────────────────────────────────

_TRAIN_PY = r"""
import json, sys, os
d = sys.argv[1]
out = {}
for f in ("manifest.json", "run_state.json"):
    try:
        out[f] = json.load(open(os.path.join(d, f)))
    except Exception as e:
        out[f] = {"error": type(e).__name__}
m, r = out["manifest.json"], out["run_state.json"]
p = r.get("progress") or {}
vh = r.get("val_history") or []
print(json.dumps({"status": m.get("status"), "epochs_done": len(m.get("epochs_done") or []),
                  "step": p.get("step"), "total": (r.get("schedule") or {}).get("total_steps"),
                  "epoch": p.get("epoch"), "epochs": p.get("epochs"), "eta_at": p.get("eta_at"),
                  "val": [v.get("val_loss") for v in vh][-2:], "val0": (vh[0].get("val_loss") if vh else None),
                  "probe": [v.get("probe_loss") for v in vh][-1:], "error": m.get("error")}))
"""


def train_line(cx: Ctx) -> str:
    c = cx.c
    run = c["R2_TRAIN_RUN"]
    cp = cx.ssh(f"python3 -I -c {shlex.quote(_TRAIN_PY)} \"$HOME\"/{shlex.quote(c['R2_TRAIN_DIR'] + '/' + run)}",
                timeout=30)
    try:
        t = json.loads(cp.stdout.strip().split("\n")[-1])
    except (ValueError, IndexError):
        return f"Train {run}: unreadable (ssh exit {cp.returncode})"
    val = ""
    if t.get("val0") is not None and t.get("val"):
        val = f", val {t['val0']:.3f}->{t['val'][-1]:.3f}"
    if t.get("status") == "done":
        return f"Train {run}: DONE, {t.get('epochs_done')} epochs{val}"
    eta = ""
    with contextlib.suppress(Exception):
        eta = f", ETA {et(datetime.fromisoformat(t['eta_at'].replace('Z', '+00:00')).timestamp())}"
    return (f"Train {run}: {t.get('status')} step {t.get('step')}/{t.get('total')} epoch "
            f"{t.get('epoch')}/{t.get('epochs')}{eta}{val}")


def post_train_line(cx: Ctx) -> str:
    c = cx.c
    d = Path(c["R2_POST_TRAIN_DIR"])
    parts = []
    try:
        mk = json.loads((d / "PIPELINE-DONE.json").read_text())
        parts.append(f"post-train {str(mk.get('outcome'))[:60]}")
    except (OSError, ValueError):
        try:
            s = json.loads((d / "state.json").read_text())
            steps = s.get("steps") or {}
            cur = [k for k, v in steps.items() if (v or {}).get("status") not in ("done", "skipped")]
            done = sum(1 for v in steps.values() if (v or {}).get("status") == "done")
            parts.append(f"post-train running ({done} steps done{', at ' + cur[0] if cur else ''})" if steps
                         else "post-train armed, waiting for training")
        except (OSError, ValueError):
            parts.append("post-train: no state yet")
    parts.append("capacity " + ("DONE" if Path(c["R2_CAPACITY_DONE"]).is_file() else "not done"))
    sun = cx.load_state("sunday.json")
    if sun:
        parts.append(f"4090 chain {sun.get('phase')}" + (f" ({str(sun.get('why'))[:60]})" if sun.get("why") else ""))
    return "; ".join(parts)


def data_line(cx: Ctx) -> str:
    c = cx.c
    parts = []
    for d in c["R2_DATASETS"].split():
        p = Path(d)
        counts = []
        for split in ("train", "val"):
            with contextlib.suppress(OSError):
                with open(p / f"{split}.jsonl", "rb") as fh:
                    counts.append(f"{sum(1 for _ in fh):,} {split}")
        parts.append(f"{p.name.replace('build-', '').replace('-max-derived', '')} {'/'.join(counts) or 'missing'}")
    try:
        h = json.loads((Path(c["R2_POOL"]) / "harvest.json").read_text())
        pool = ", ".join(f"{k} {v:,}" for k, v in sorted((h.get("rows_by_source") or {}).items()))
        parts.append(f"pool: {pool}")
    except (OSError, ValueError):
        parts.append("pool: unreadable")
    return "Data: " + "; ".join(parts)


def gen_stats_lines(day: dict[str, Any], r2: dict[str, Any], since_label: str) -> list[str]:
    """The 3 generation lines from two ``gen_stats.py --json`` outputs (--hours 24, --hours since round 2)."""
    out = []
    for label, js in (("Gen 24h", day), (since_label, r2)):
        tot = (js.get("all") or {}).get("last_hours") or {}
        groups: Counter[str] = Counter()
        for name, r in (js.get("runs") or {}).items():
            groups[run_group(name)] += int(((r or {}).get("last_hours") or {}).get("rows") or 0)
        hours = float(tot.get("hours") or js.get("hours") or 1) or 1.0
        shown = ["3090Ti", "4090"] + [g for g in ("Max", "other") if groups.get(g)]
        area = ", ".join(f"{k} {v:g}" for k, v in list((tot.get("by_area_per_hour") or {}).items())[:6]) or "-"
        out.append(f"{label}: {tot.get('rows', 0)} rows, {float(tot.get('rows_per_hour') or 0):.1f}/h ("
                   + " ".join(f"{g} {groups.get(g, 0) / hours:.1f}" for g in shown)
                   + f"), keep {tot.get('keep_rate')} | by area/h: {area}")
    cat = ((day.get("all") or {}).get("last_hours") or {}).get("by_category_per_hour") or {}
    out.append("24h by category/h: " + (", ".join(f"{k} {v:g}" for k, v in list(cat.items())[:6]) or "-"))
    return out


def gen_lines(cx: Ctx, now: float) -> list[str]:
    """Rows/hour for the last 24 h and since round 2 started: the staged round-2 ``gen_stats.py`` when it is
    there (R2_GEN_STATS_CMD=auto), a configured command, or computed here from the run ledgers."""
    c = cx.c
    cmd = c["R2_GEN_STATS_CMD"].strip()
    since = when(c["R2_START"])
    hours_r2 = max((now - since) / 3600, 0.01)
    since_label = f"Gen since r2 {et(since, '%a %H:%M')} ({hours_r2:.{0 if hours_r2 >= 10 else 1}f} h)"
    staged = Path(c["R2_CODE"]) / "scripts" / "distill" / "gen_stats.py"
    if cmd == "auto" and staged.is_file():
        try:
            js = []
            for h in (24.0, hours_r2):
                cp = cx.run([c["R2_PY"], "-I", str(staged), "--runs", f"{c['R2_RUNS']}/*", "--hours", f"{h:.4f}",
                             "--now", f"{now:.0f}", "--json"], timeout=120)
                js.append(json.loads(cp.stdout))
            return gen_stats_lines(js[0], js[1], since_label)
        except (ValueError, TypeError, AttributeError) as e:
            cx.log(f"gen_stats.py failed ({type(e).__name__}): computing from the ledgers instead")
    elif cmd and cmd != "auto":
        argv = [a.format(py=c["R2_PY"], code=c["R2_CODE"], runs=c["R2_RUNS"], since=iso(since), now=iso(now))
                for a in shlex.split(cmd)]
        cp = cx.run(argv, timeout=120)
        lines = [ln for ln in (cp.stdout or "").split("\n") if ln.strip()]
        if cp.returncode == 0 and lines:
            return lines[:3]
        cx.log(f"R2_GEN_STATS_CMD exited {cp.returncode}: computing from the ledgers instead")
    paths = sorted(Path(p) for p in glob.glob(os.path.join(c["R2_RUNS"], "*", "*.tasks.jsonl")))
    events = list(ledger_events(paths))
    day = gen_stats(events, now - 86400, now)
    r2 = gen_stats(events, since, now)
    return [stats_line("Gen 24h", day), stats_line(since_label, r2), domain_line(day)]


def windows_lines(cx: Ctx, now: float, tw: dict[str, str]) -> list[str]:
    nxt = []
    for t, label in MINI_TIMERS:
        n = cx.timer_next(t)
        if n:
            nxt.append((n, label))
    for t, label in TOWER_TIMERS:
        v = tw.get(f"timer {t}", "-")
        if v.isdigit():
            nxt.append((float(v), label))
    nxt.sort()
    gen = []
    cp = cx.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "--no-pager",
                 "--state=active", "distill-gen-*", "r2-gen-*", "distill-teacher*", "r2-tower-chain-*"], timeout=30)
    units = [ln.split()[0] for ln in (cp.stdout or "").split("\n") if ln.strip()]
    gen.append("mini: " + (", ".join(u.replace(".service", "") for u in units) or "no gen/teacher units"))
    biz = tw.get("business", "?")
    teacher = tw.get("r2_units", "").strip() or tw.get("distill_units", "").strip() or "none"
    gen.append(f"4090: business {biz}, distill units {teacher}")
    return ["Now: " + "; ".join(gen),
            "Next: " + ", ".join(f"{label} {et(n, '%a %H:%M')}" for n, label in nxt[:7])]


def noise(unit: str) -> bool:
    """Failed user units that are not ours to report (desktop snaps' notifiers)."""
    return unit.startswith("snap.")


def problems_line(cx: Ctx, tw: dict[str, str]) -> str:
    c = cx.c
    probs = []
    cp = cx.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "--no-pager", "--state=failed"],
                timeout=30)
    failed = [u for u in (ln.split()[0] for ln in (cp.stdout or "").split("\n") if ln.strip()) if not noise(u)]
    if failed:
        probs.append("mini failed units: " + ", ".join(failed[:4]))
    tower_failed = [u for u in tw.get("failed_units", "").split() if not noise(u)]
    if tower_failed:
        probs.append("4090 failed units: " + ", ".join(tower_failed[:4]))
    if tw.get("_rc") not in (None, "0"):
        probs.append(f"4090 status unreachable (ssh {tw.get('_rc')})")
    if tw.get("flag", "none") != "none":
        probs.append(f"4090 NO-SWAPS: {tw['flag'][:80]}")
    flag = Path(c["R2_WIN_DIR"]) / "NO-SWAPS-TONIGHT"
    if flag.exists():
        probs.append("mini NO-SWAPS flag set")
    for f, label in ((Path(c["R2_RUNS"]) / "state" / "progress.json", "local gen"),
                     (Path(c["R2_RUNS"]) / "state" / "remote-progress.json", "remote gen")):
        with contextlib.suppress(OSError, ValueError):
            p = json.loads(f.read_text())
            if p.get("state") == "failed":
                probs.append(f"{label} failed ({str(p.get('stop_reason'))[:60]})")
    sun = cx.load_state("sunday.json")
    if sun.get("phase") == "failed":
        probs.append(f"4090 chain failed: {str(sun.get('why'))[:80]}")
    return "Problems: " + ("; ".join(probs) if probs else "none")


def cmd_heartbeat(cx: Ctx, send: bool) -> int:
    now = cx.now()
    n = int(cx.c["R2_HEARTBEAT_LINES"])
    tw: dict[str, str] = {}
    try:
        tw = cx.tower_status()
    except Exception as e:  # noqa: BLE001 - the heartbeat reports, never fails
        tw = {"_rc": f"{type(e).__name__}"}
    lines = [f"Distill heartbeat {et(now, '%a %d %b %H:%M %Z')}"]
    for fn in (lambda: [train_line(cx)], lambda: [post_train_line(cx)], lambda: [data_line(cx)],
               lambda: gen_lines(cx, now), lambda: windows_lines(cx, now, tw), lambda: [problems_line(cx, tw)]):
        try:
            lines += fn()
        except Exception as e:  # noqa: BLE001
            lines.append(f"(section failed: {type(e).__name__}: {str(e)[:80]})")
    lines = clip(lines, n)
    if send:
        cx.execute = True
        cx.notify(lines, n)
    else:
        for ln in lines:
            cx.out(ln)
    return 0


def cmd_status(cx: Ctx) -> int:
    for name in ("sunday.json", "night.json"):
        st = cx.load_state(name)
        cx.out(f"{name}: {json.dumps(st)[:600] if st else 'none'}")
    for t, label in MINI_TIMERS:
        n = cx.timer_next(t)
        cx.out(f"  {label} ({t}.timer): {et(n, '%a %F %H:%M %Z') if n else 'not armed'}")
    return 0


def main(argv: list[str] | None = None, **kw: Any) -> int:
    ap = argparse.ArgumentParser(prog="round2_chain", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--conf", default=os.environ.get("R2_CONF") or str(Path("~/distill/round2/round2.conf").expanduser()))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config")
    sub.add_parser("status")
    for name in ("sunday", "tower-night"):
        sub.add_parser(name).add_argument("--execute", action="store_true")
    mn = sub.add_parser("mini-night")
    mn.add_argument("which", choices=("start", "end"))
    mn.add_argument("--execute", action="store_true")
    gl = sub.add_parser("gen-local")
    gl.add_argument("rest", nargs=argparse.REMAINDER)
    hb = sub.add_parser("heartbeat")
    hb.add_argument("--send", action="store_true")
    args = ap.parse_args(argv)
    conf = load_conf(Path(args.conf))
    execute = bool(getattr(args, "execute", False))
    if args.cmd == "gen-local":
        execute = "--execute" in args.rest
    cx = Ctx(conf, execute, args.cmd, **kw)
    try:
        if args.cmd == "config":
            return cmd_config(cx)
        if args.cmd == "status":
            return cmd_status(cx)
        if args.cmd == "sunday":
            return cmd_sunday(cx)
        if args.cmd == "tower-night":
            return cmd_tower_night(cx)
        if args.cmd == "mini-night":
            return cmd_mini_night(cx, args.which)
        if args.cmd == "gen-local":
            return cmd_gen_local(cx, args.rest)
        return cmd_heartbeat(cx, args.send)
    except Refused as e:
        cx.log(f"refused: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
