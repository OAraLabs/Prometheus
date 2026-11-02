#!/usr/bin/env python3
"""Round-2 9B QLoRA across nightly 4090 windows: the multi-night job (stdlib only).

    train_nights.py status   --job-dir DIR                          # the job state, short
    train_nights.py estimate --dataset TRAIN.jsonl [--epochs 2]      # nights from the dataset's tokens
    train_nights.py estimate --new-rows 3750 [--replay-share 0.4 ...] # nights from stated assumptions
    train_nights.py mini-night --tower USER@HOST --run RUN [--execute]  # 3090 Ti host, after 01:20 ET

Why: the 4090 is the flywheel's only in its nightly window (01:20 window_start -> 06:35 runner stop-at
-> 06:40 window_end, about 5 h of GPU), and the round-2 SFT needs more than one window. The operator
(2026-11-02): it starts once the round-2 dataset is built; its adapter is the starting point of GRPO.

Where it runs: inside the 4090 job runner (``job_runner.py``, step ``train-nights``, ``DJR_NIGHTS=1``,
usually alone: ``DJR_ONLY=train-nights``), which window_start starts every night as
``distill-jobs-<stamp>.service`` and window_end stops with SIGTERM (KillMode=mixed). This module holds
the job's state machine, its Telegram and its arithmetic; the runner does the processes. Installed
beside ``job_runner.py`` in ``~/distill/window/jobs/`` (a window bundle file: a new rehearsal).

State: ``<job dir>/nights.json`` (``distill-train-nights/1``), the job dir being the runner step's
directory (``$DJR_RESULTS/base-9b/train-nights-<run>``). Phases:

  train       each night the trainer (``train_qlora.py``) resumes the run in ``DJR_TRAIN_OUT``
              (``--resume`` once ``run_state.json`` exists: the same plan, position, optimizer, schedule
              and RNG, so the rows come in the order an uninterrupted run sees them) until the runner's
              stop-at (it checkpoints and exits two minutes before) or SIGTERM (window_end: it drops the
              accumulation group in progress, checkpoints and exits). Light checkpoints every
              ``DJR_TRAIN_SAVE_EVERY_MIN`` (20: an unclean end loses at most that), held-out val + probe
              every ``DJR_TRAIN_EVAL_EVERY_MIN`` and at every epoch end; retention is the newest
              ``DJR_TRAIN_KEEP`` (2) plus every epoch end (``--keep-epoch-checkpoints``), checked here
              every night. A stopped night stays ``train``; a trainer failure counts (two in a row:
              ``failed``); ``done`` goes through :func:`training_verdict` (status done, adapter on disk,
              every epoch recorded, finite held-out losses, val AND fixed-probe loss lower at the end
              than at step 0: the operator's smoke gate on the full run) to ``post-train``, or to
              ``failed`` with the reason.
  post-train  the existing post-training pipeline (``post_train.py`` on the 3090 Ti host: Phase 6
              merge, GGUF, imatrix, Q4_K_M/Q5_K_M, then scoring) runs in the following windows. Each
              night the runner HOLDS the open window for it (business stays down, the GPU free): it
              writes ``hold.json`` (night, until, the runner unit) and waits. ``mini-night`` (a timer on
              the 3090 Ti host shortly after window_start) sees the hold and arms ``post_train.py
              --window-hold`` with a stop-at inside it; the pipeline writes ``hold-taken-<night>.json``
              when it takes the GPU, ``hold-released-<night>.json`` when it stops for the night, and
              ``post-train.done.json`` once it completed. Business is always restored by the window
              machinery (watcher / window_end), never by the pipeline in this mode. No pipeline within
              ``DJR_NIGHTS_HOLD_MIN`` (45): the hold ends and the window closes early. Every stage
              resumes or restarts idempotently: the merge writes ``merged.partial`` then renames, the
              pushes and fetches are rsync + sha256 checks, ``quantize.py`` redoes only a ``.partial``
              output and exits 75 at its stop-at, ``score.py`` skips answered cases, and ``arm --rerun``
              keeps every finished step.
  done        nothing left; the runner's dry run then fails on purpose, so window_start skips the swap
              (no nightly down/up) until the operator disarms the job (``DJR_NIGHTS=0``).
  failed      parked until someone looks; same dry-run refusal.

Telegram: at most one message of at most 5 lines per night (:func:`night_message`): steps done/total,
tonight's rate and why it stopped, the val/probe curve, the ETA in nights, retention and disk.

Exit (CLI): 0 ok, 1 a check failed, 2 usage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo

    ET: Any = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - no tz database: UTC
    ET = timezone.utc

SCHEMA = "distill-train-nights/1"
HOLD_SCHEMA = "distill-train-nights-hold/1"
STATE_FILE = "nights.json"
HOLD_FILE = "hold.json"
DONE_FILE = "post-train.done.json"
PHASES = ("train", "post-train", "done", "failed")
TERMINAL = ("done", "failed")
MAX_FAILURES = 2          # trainer failures in a row before the job parks itself
NOTIFY_LINES = 5
CKPT_PREFIX = "checkpoint-"
# Measured on the 4090 (DISTILL-CAPACITY.md, seqlen_bench.py 2026-11-01): QLoRA r=32 9B, micro-batch 1,
# activation offload, tok/s at the packed length. 65,536 ran out of memory.
SPEED_TOK_S: Tuple[Tuple[int, float], ...] = ((16384, 2548.7), (32768, 2330.2), (49152, 1990.1))
# Round 1 (9b-r32-v4b-max-derived, 2026-11-01): 124.06M tokens at 2,400.8 tok/s wall against the bench's
# 2,548.7: held-out evaluations, checkpoints and the load cost ~5.8%.
OVERHEAD = 2400.8 / 2548.7
# A night: window_start ~01:22 ET (swap + runner start), the trainer stops at 06:33 (stop-at 06:35 - 2 min);
# minus ~3 min for the load and the longest-row preflight.
NIGHT_S = 5 * 3600 + 8 * 60 - 180
FULL_NIGHT_FRACTION = 0.6  # a night that trained at least this share of NIGHT_S measures the yield

Runner = Callable[..., Tuple[int, str]]  # (argv, timeout=...) -> (returncode, output)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def now_iso(t: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat(timespec="seconds")


def night_key(t: float) -> str:
    """The window's night: its ET date (a window opens after 01:00, so the morning's date)."""
    return datetime.fromtimestamp(t, ET).strftime("%Y-%m-%d")


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def f4(x: Any) -> str:
    return f"{x:.4f}" if finite(x) else "n/a"


def tree_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# the job state
# ---------------------------------------------------------------------------


def new_state(run: str, out: str, t: float) -> Dict[str, Any]:
    return {"schema": SCHEMA, "run": run, "out": out, "created_at": now_iso(t), "phase": "train",
            "failures": 0, "last_error": None, "nights": [], "verdict": None, "trained_at": None,
            "post_train": {"requested_at": None, "holds": [], "done": None}}


def load(job_dir: Path) -> Optional[Dict[str, Any]]:
    st = read_json(Path(job_dir) / STATE_FILE)
    return st if isinstance(st, dict) and st.get("schema") == SCHEMA else None


def save(job_dir: Path, st: Dict[str, Any]) -> None:
    write_json(Path(job_dir) / STATE_FILE, st)


def identity_problem(st: Dict[str, Any], run: str, out: str) -> Optional[str]:
    """A job dir belongs to one run: a changed DJR_TRAIN_OUT is a new job, never a silent switch."""
    if st.get("out") != out or st.get("run") != run:
        return (f"{STATE_FILE} is for run {st.get('run')!r} ({st.get('out')}), the settings name {run!r} "
                f"({out}): a new run needs a new job directory")
    if st.get("phase") not in PHASES:
        return f"{STATE_FILE} has an unknown phase {st.get('phase')!r}"
    return None


def progress_of(rs: Any) -> Dict[str, Any]:
    """The trainer's run_state.json ``progress`` (step, totals, tokens, rate), {} when none yet."""
    prog = (rs or {}).get("progress") if isinstance(rs, dict) else None
    return prog if isinstance(prog, dict) else {}


def begin_night(st: Dict[str, Any], t: float, rs: Any, *, window_s: float) -> Dict[str, Any]:
    prog = progress_of(rs)
    rec = {"night": night_key(t), "phase": st["phase"], "started_at": now_iso(t), "window_s": round(window_s),
           "step_from": int(prog.get("step") or 0), "tokens_from": int(prog.get("tokens_seen") or 0),
           "outcome": "running"}
    st["nights"].append(rec)
    return rec


def training_verdict(man: Any, rs: Any, *, adapter_present: bool) -> Tuple[bool, str, Dict[str, Any]]:
    """Did the run complete and pass the operator's held-out gate? (ok, one-line reason, facts).
    The same rule as ``post_train.py training_verdict`` (it re-checks before it quantizes)."""
    man = man if isinstance(man, dict) else {}
    rs = rs if isinstance(rs, dict) else {}
    status = man.get("status")
    info: Dict[str, Any] = {"status": status, "step": man.get("step"), "tokens_seen": man.get("tokens_seen")}
    if status != "done":
        why = man.get("reason")
        return False, f"training status {status or 'unknown'}" + (f" ({str(why)[:120]})" if why else ""), info
    if man.get("mode") not in (None, "train"):
        return False, f"mode {man.get('mode')}: not a full training run", info
    if not man.get("adapter"):
        return False, "the manifest names no adapter", info
    if not adapter_present:
        return False, "the adapter files (adapter_config.json, adapter_model.safetensors) are missing", info
    epochs = (man.get("config") or {}).get("epochs")
    done = man.get("epochs_done") or rs.get("epochs_done") or []
    got = sorted({int(e["epoch"]) for e in done if isinstance(e, dict) and isinstance(e.get("epoch"), int)})
    info["epochs_done"] = got
    if isinstance(epochs, int) and len(got) < epochs:
        return False, f"only {len(got)} of {epochs} epochs recorded", info
    hist = [h for h in (man.get("val_history") or rs.get("val_history") or []) if isinstance(h, dict)]
    base = next((h for h in hist if h.get("step") == 0), None)
    last = max(hist, key=lambda h: h.get("step") or 0) if hist else None
    if base is None or last is None or not (last.get("step") or 0) > 0:
        return False, "no held-out evaluation after step 0", info
    bad = sorted({int(h.get("step") or 0) for h in hist for k in ("val_loss", "probe_loss") if not finite(h.get(k))})
    if bad:
        return False, f"non-finite held-out loss at step(s) {bad}", info
    info.update(val=[base["val_loss"], last["val_loss"]], probe=[base["probe_loss"], last["probe_loss"]],
                last_eval_step=last.get("step"))
    if not last["val_loss"] < base["val_loss"]:
        return False, f"val loss did not fall ({base['val_loss']:.4f} -> {last['val_loss']:.4f})", info
    if not last["probe_loss"] < base["probe_loss"]:
        return False, f"probe loss did not fall ({base['probe_loss']:.4f} -> {last['probe_loss']:.4f})", info
    return True, (f"done at step {man.get('step')}: val {base['val_loss']:.4f} -> {last['val_loss']:.4f}, "
                  f"probe {base['probe_loss']:.4f} -> {last['probe_loss']:.4f}, epochs {got}"), info


def end_train_night(st: Dict[str, Any], rec: Dict[str, Any], *, rc: Optional[int], reason: Optional[str],
                    man: Any, rs: Any, adapter_present: bool, t: float, train_s: float,
                    chain: bool = True) -> Tuple[str, str]:
    """Record the night and move the job on. Returns the runner step's (status, detail):
    ``partial`` (more nights), ``failed`` (counted by the runner too) or ``done`` (nothing left)."""
    prog = progress_of(rs)
    man_d = man if isinstance(man, dict) else {}
    status = str(man_d.get("status") or "")
    fresh = rc == 0  # a crashed trainer may have left last night's manifest: never read it as tonight's
    sched = (rs or {}).get("schedule") if isinstance(rs, dict) else None
    rec.update(ended_at=now_iso(t), rc=rc, reason=reason or (man_d.get("reason") if fresh else None),
               train_s=round(train_s),
               step_to=int((man_d.get("step") if fresh else None) or prog.get("step") or rec["step_from"]),
               total_steps=prog.get("total_steps") or (sched or {}).get("total_steps"),
               tokens_to=int((man_d.get("tokens_seen") if fresh else None) or prog.get("tokens_seen")
                             or rec["tokens_from"]),
               tokens_left=prog.get("tokens_left"), epoch=prog.get("epoch"), epochs=prog.get("epochs"),
               tok_s=prog.get("tok_s"), trainer_status=status or None)
    hist = (rs or {}).get("val_history") if isinstance(rs, dict) else None
    if hist:
        last = max(hist, key=lambda h: h.get("step") or 0)
        rec["held_out_last"] = {"step": last.get("step"), "val": last.get("val_loss"), "probe": last.get("probe_loss")}
    if rc == 0 and status == "done":
        ok, why, info = training_verdict(man_d, rs, adapter_present=adapter_present)
        st["verdict"] = {"ok": ok, "why": why, **info}
        st["failures"] = 0
        if not ok:
            rec["outcome"] = "verdict-failed"
            st["phase"], st["last_error"] = "failed", f"verdict: {why}"
            return "failed", f"training ended but the gate failed: {why}"
        st["trained_at"] = now_iso(t)
        rec["outcome"] = "trained"
        if not chain:
            st["phase"] = "done"
            return "done", f"trained ({why}); post-train chaining is off"
        st["phase"] = "post-train"
        st["post_train"]["requested_at"] = now_iso(t)
        return "partial", f"trained ({why}); the post-training pipeline runs in the next window"
    if reason or (fresh and status == "stopped"):
        rec["outcome"] = "stopped"
        st["failures"] = 0
        return "partial", (f"stopped ({reason or man_d.get('reason') or 'stop'}) at step {rec['step_to']}/"
                           f"{rec.get('total_steps') or '?'}; resumes next window (--resume)")
    st["failures"] = int(st.get("failures") or 0) + 1
    rec["outcome"] = "failed"
    why = (f"train_qlora.py exit {rc} (manifest status {status or 'none'}"
           + (f": {str(man_d.get('reason'))[:120]}" if man_d.get("reason") else "") + ")") if fresh else (
        f"train_qlora.py exit {rc} (no manifest of its own tonight)")
    if st["failures"] >= MAX_FAILURES:
        st["phase"] = "failed"
        why += f"; {st['failures']} failures in a row: the job is parked"
    st["last_error"] = why
    return "failed", why


# ---------------------------------------------------------------------------
# the post-train hold (4090 side) and its markers
# ---------------------------------------------------------------------------


def taken_name(night: str) -> str:
    return f"hold-taken-{night}.json"


def released_name(night: str) -> str:
    return f"hold-released-{night}.json"


def hold_doc(st: Dict[str, Any], *, night: str, since: float, until: float, unit: Optional[str],
             pid: int) -> Dict[str, Any]:
    return {"schema": HOLD_SCHEMA, "run": st["run"], "out": st["out"], "night": night, "state": "holding",
            "since": now_iso(since), "since_epoch": since, "until": now_iso(until), "until_epoch": until,
            "unit": unit, "pid": pid}


def hold_events(job_dir: Path, night: str) -> Dict[str, Any]:
    jd = Path(job_dir)
    return {"taken": read_json(jd / taken_name(night)), "released": read_json(jd / released_name(night)),
            "done": read_json(jd / DONE_FILE)}


def hold_problem(hold: Any, night: str, t: float, *, min_left_s: float = 1800) -> Optional[str]:
    """Why a pipeline may NOT take the GPU from this hold now (None: it may)."""
    if not isinstance(hold, dict) or hold.get("schema") != HOLD_SCHEMA:
        return "no hold file from the 4090 window"
    if hold.get("state") != "holding":
        return f"the 4090 window's hold is {hold.get('state')!r}, not holding"
    if hold.get("night") != night:
        return f"the hold is for the night of {hold.get('night')}, not {night}"
    until = hold.get("until_epoch")
    if not isinstance(until, (int, float)) or not finite(until) or float(until) - t < min_left_s:
        return f"the hold ends at {hold.get('until')}: less than {min_left_s / 60:.0f} min left"
    return None


def end_hold(st: Dict[str, Any], rec: Dict[str, Any], ev: Dict[str, Any], *, why: str, t: float
             ) -> Tuple[str, str]:
    """Record the end of a post-train hold. (status, detail) for the runner step."""
    rec.update(ended_at=now_iso(t), outcome="hold", reason=why, taken=bool(ev.get("taken")),
               released=ev.get("released"))
    st["post_train"]["holds"].append({k: rec.get(k) for k in ("night", "started_at", "ended_at", "taken",
                                                              "reason")})
    done = ev.get("done")
    if isinstance(done, dict):
        st["phase"] = "done"
        st["post_train"]["done"] = done
        return "done", f"post-training pipeline completed ({str(done.get('outcome') or 'completed')[:120]})"
    if not ev.get("taken"):
        return "partial", f"post-train hold ended without the pipeline ({why})"
    return "partial", f"post-train night ended ({why}); the pipeline resumes next window"


# ---------------------------------------------------------------------------
# checkpoints, retention, disk
# ---------------------------------------------------------------------------


def checkpoints(out: Path) -> List[Dict[str, Any]]:
    """The run's checkpoint directories (complete or not), oldest first."""
    res: List[Dict[str, Any]] = []
    out = Path(out)
    if not out.is_dir():
        return res
    for p in sorted(out.iterdir()):
        if not p.is_dir() or not p.name.startswith(CKPT_PREFIX):
            continue
        partial = p.name.endswith(".partial")
        step_txt = p.name[len(CKPT_PREFIX):].split(".")[0]
        sj = read_json(p / "state.json") or {}
        res.append({"name": p.name, "step": int(step_txt) if step_txt.isdigit() else None,
                    "complete": (p / "COMPLETE").is_file() and not partial, "partial": partial,
                    "epoch_end": bool(sj.get("epoch_end")), "bytes": tree_bytes(p)})
    return res


def retention_problem(ckpts: List[Dict[str, Any]], *, keep: int, epochs_done: int) -> Optional[str]:
    """The retention rule (the newest *keep* + every epoch end) holds, and nothing half-written is left."""
    complete = [c for c in ckpts if c["complete"]]
    partial = [c["name"] for c in ckpts if not c["complete"]]
    extra = [c for c in complete[:-keep] if not c["epoch_end"]] if keep > 0 else complete
    bound = keep + max(0, epochs_done)
    if partial:
        return f"incomplete checkpoint dirs left: {', '.join(partial[:3])}"
    if extra or len(complete) > bound:
        return (f"{len(complete)} checkpoints kept (bound {bound}: newest {keep} + {epochs_done} epoch ends): "
                + ", ".join(c["name"] for c in extra[:3]))
    return None


def ckpt_gib(ckpts: List[Dict[str, Any]], default_gib: float) -> float:
    sizes = [c["bytes"] for c in ckpts if c["complete"] and c["bytes"]]
    return max(sizes) / 2**30 if sizes else default_gib


def disk_need_gib(ckpts: List[Dict[str, Any]], *, keep: int, epochs: int, epochs_done: int,
                  default_ckpt_gib: float, extra_gib: float = 0.7) -> Tuple[float, List[str]]:
    """What the rest of the run still writes, on top of what its checkpoints already hold: at the
    peak the newest *keep* + one being written + every epoch end (done and still to come), the
    adapter and logs (*extra_gib*)."""
    size = ckpt_gib(ckpts, default_ckpt_gib)
    have = sum(c["bytes"] for c in ckpts) / 2**30
    peak = (keep + 1 + max(epochs, epochs_done)) * size
    need = max(0.0, peak - have) + extra_gib
    return need, [f"checkpoints {keep}+1 newest + {epochs} epoch ends x {size:.2f} GiB = {peak:.1f} GiB "
                  f"(holding {have:.1f})", f"adapter + logs {extra_gib:g} GiB"]


# ---------------------------------------------------------------------------
# rates, ETA and the nightly Telegram
# ---------------------------------------------------------------------------


def tokens_per_night(st: Dict[str, Any], night_s: float = NIGHT_S) -> Optional[float]:
    """The median tokens one full night trained (nights that ran at least FULL_NIGHT_FRACTION of a
    night). None before the first such night."""
    vals = [int(n.get("tokens_to") or 0) - int(n.get("tokens_from") or 0) for n in st.get("nights") or []
            if n.get("outcome") in ("stopped", "trained") and (n.get("train_s") or 0) >= FULL_NIGHT_FRACTION * night_s]
    vals = [v for v in vals if v > 0]
    return float(statistics.median(vals)) if vals else None


def eta_nights(tokens_left: Any, per_night: Optional[float]) -> Optional[float]:
    if not finite(tokens_left) or not per_night or per_night <= 0:
        return None
    return float(tokens_left) / per_night


def curve(hist: Any, key: str, limit: int = 6) -> str:
    """'0:1.000 400:0.872 812:0.845' over the held-out points, thinned evenly (first and last kept)."""
    pts = [(h["step"], h[key]) for h in sorted(hist or [], key=lambda h: h.get("step") or 0)
           if isinstance(h, dict) and isinstance(h.get("step"), int) and finite(h.get(key))]
    if len(pts) > limit:
        idx = sorted({round(i * (len(pts) - 1) / (limit - 1)) for i in range(limit)})
        pts = [pts[i] for i in idx]
    return " ".join(f"{s}:{v:.3f}" for s, v in pts) or "none"


def night_message(st: Dict[str, Any], rec: Dict[str, Any], rs: Any, *, ckpts: List[Dict[str, Any]],
                  free_gib: Optional[float], keep: int, night_s: float = NIGHT_S) -> List[str]:
    """The night's Telegram (<= 5 lines)."""
    prog = progress_of(rs)
    n = sum(1 for x in st.get("nights") or [] if x.get("phase") == "train")
    total = rec.get("total_steps") or prog.get("total_steps")
    step = rec.get("step_to", rec.get("step_from"))
    pct = f" ({100.0 * float(step or 0) / float(total):.0f}%)" if finite(total) and total else ""
    head = {"trained": "DONE", "stopped": "stopped", "failed": "FAILED", "verdict-failed": "gate FAILED"}.get(
        str(rec.get("outcome")), str(rec.get("outcome")))
    lines = [f"9B round-2 training night {n}: steps {rec.get('step_from')}->{step} of {total or '?'}{pct}, "
             f"epoch {prog.get('epoch', '?')}/{prog.get('epochs', '?')}; {head}"
             + (f" ({str(rec.get('reason'))[:40]})" if rec.get("reason") and head != "DONE" else "") + "."]
    toks = int(rec.get("tokens_to") or 0) - int(rec.get("tokens_from") or 0)
    rate = toks / rec["train_s"] if rec.get("train_s") else None
    lines.append(f"Tonight {(rec.get('train_s') or 0) / 3600:.1f} h, {toks / 1e6:.1f}M tokens"
                 + (f" at {rate:,.0f} tok/s" if rate else "") + ".")
    hist = (rs or {}).get("val_history") if isinstance(rs, dict) else []
    lines.append(f"Val {curve(hist, 'val_loss', 5)}; probe {curve(hist, 'probe_loss', 3)}.")
    if rec.get("outcome") == "trained":
        v = st.get("verdict") or {}
        lines.append(f"Gate passed ({str(v.get('why', ''))[:90]}). Post-train (merge, GGUF, scoring) "
                     "starts next window.")
    elif rec.get("outcome") in ("failed", "verdict-failed"):
        lines.append(f"Error: {str(st.get('last_error') or '')[:150]}")
    else:
        per = tokens_per_night(st, night_s) or (float(prog["tok_s"]) * night_s if finite(prog.get("tok_s")) else None)
        left = eta_nights(prog.get("tokens_left"), per)
        if left is not None and per:
            lines.append(f"ETA ~{left:.1f} more nights ({(prog.get('tokens_left') or 0) / 1e6:.0f}M tokens left at "
                         f"~{per / 1e6:.0f}M/night), then post-train.")
        else:
            lines.append("ETA: not measured yet.")
    kept = [c for c in ckpts if c["complete"]]
    lines.append(f"Checkpoints {len(kept)} (newest {keep} + {sum(1 for c in kept if c['epoch_end'])} epoch ends), "
                 f"{sum(c['bytes'] for c in kept) / 2**30:.1f} GiB"
                 + (f"; disk {free_gib:.0f} GiB free." if free_gib is not None else "."))
    return [ln[:300] for ln in lines[:NOTIFY_LINES]]


def hold_message(st: Dict[str, Any], status: str, detail: str) -> List[str]:
    if status == "done":
        return [f"9B round-2: post-training pipeline completed for {st['run']}.",
                str(detail)[:200], "The 4090 multi-night job is done: disarm it (DJR_NIGHTS=0)."]
    return [f"9B round-2 post-train night: {str(detail)[:220]}.",
            "Business is restored by the window machinery as usual."]


# ---------------------------------------------------------------------------
# the estimate (nights for a dataset)
# ---------------------------------------------------------------------------


def speed_at(n_tokens: float, table: Sequence[Tuple[int, float]] = SPEED_TOK_S) -> float:
    """tok/s for a row of *n_tokens*: the measured points, linear between them, flat outside."""
    pts = sorted(table)
    if n_tokens <= pts[0][0]:
        return pts[0][1]
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if n_tokens <= b:
            return va + (vb - va) * (n_tokens - a) / (b - a)
    return pts[-1][1]


def estimate(lengths: Iterable[int], *, epochs: int = 2, night_s: float = NIGHT_S,
             overhead: float = OVERHEAD, max_seq: int = 49152) -> Dict[str, Any]:
    """Train time and nights for rows of these token lengths (each capped at *max_seq*)."""
    ls = [min(int(x), max_seq) for x in lengths if int(x) > 0]
    tokens = sum(ls) * epochs
    secs = sum(n / speed_at(n) for n in ls) * epochs / overhead
    return {"rows": len(ls), "epochs": epochs, "tokens": tokens, "tokens_per_epoch": sum(ls),
            "mean_row_tokens": round(sum(ls) / len(ls)) if ls else 0,
            "eff_tok_s": round(tokens / secs, 1) if secs else None, "train_h": round(secs / 3600, 2),
            "night_h": round(night_s / 3600, 2), "nights": round(secs / night_s, 2) if night_s else None}


def synthetic_lengths(new_rows: int, *, replay_share: float = 0.4, lc_share: float = 0.26,
                      lc_mean: int = 36000, other_mean: int = 18000, replay_mean: int = 13846
                      ) -> List[int]:
    """Row lengths from stated assumptions: *new_rows* new rows (a *lc_share* of them long-context at
    *lc_mean* tokens, the rest at *other_mean*) and replay at *replay_share* of the FINAL train set
    (build.py ``--replay-share``) at *replay_mean* (v4b's measured mean)."""
    replay = round(new_rows * replay_share / (1.0 - replay_share)) if replay_share < 1 else 0
    lc = round(new_rows * lc_share)
    return [lc_mean] * lc + [other_mean] * (new_rows - lc) + [replay_mean] * replay


def dataset_lengths(path: Path) -> List[int]:
    out: List[int] = []
    with open(path, "rb") as fh:
        for line in fh:
            m = re.search(rb'"n_tokens":\s*(\d+)', line)
            if m:
                out.append(int(m.group(1)))
    return out


# ---------------------------------------------------------------------------
# mini-night: the 3090 Ti host's nightly trigger for the post-train pipeline
# ---------------------------------------------------------------------------


def run_cmd(argv: Sequence[str], timeout: float = 120, stdin: Optional[str] = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, input=stdin)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return p.returncode, (p.stdout + p.stderr).strip()


def mini_night_plan(nights: Any, hold: Any, marker: Any, *, t: float) -> Tuple[str, str]:
    """What tonight's trigger does: ("arm"|"arm-rerun"|"mark-done"|"nothing", why). Pure."""
    night = night_key(t)
    if not isinstance(nights, dict) or nights.get("schema") != SCHEMA:
        return "nothing", "no multi-night job state on the 4090"
    phase = nights.get("phase")
    if phase != "post-train":
        return "nothing", f"the job is in phase {phase!r}, not post-train"
    if isinstance(marker, dict) and str(marker.get("outcome") or "").startswith("completed"):
        return "mark-done", f"the pipeline completed earlier ({str(marker.get('outcome'))[:80]})"
    why = hold_problem(hold, night, t)
    if why:
        return "nothing", why
    if isinstance(marker, dict):
        return "arm-rerun", f"the pipeline stopped earlier ({str(marker.get('outcome') or '?')[:80]}): resume it"
    return "arm", "first post-train night"


def mini_night(a: argparse.Namespace, run: Runner = run_cmd, clock: Callable[[], float] = time.time) -> int:
    t = clock()
    job = a.job_dir or f"distill/phase4/results/base-9b/train-nights-{a.run}"
    job = re.sub(r"^~/", "", job.rstrip("/"))
    if job.startswith("/"):
        print("--job-dir is relative to the 4090 home (or ~/...), not an absolute path", file=sys.stderr)
        return 2
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", a.tower]

    def remote_json(name: str) -> Any:
        rc, out = run(ssh + [f"cat {shlex.quote(job + '/' + name)} 2>/dev/null"])
        try:
            return json.loads(out) if rc == 0 and out.strip() else None
        except ValueError:
            return None

    p7 = Path(a.state_dir).expanduser() if a.state_dir else Path.home() / "distill" / "phase7" / a.run
    marker = read_json(p7 / "PIPELINE-DONE.json")
    nights, hold = remote_json(STATE_FILE), remote_json(HOLD_FILE)
    action, why = mini_night_plan(nights, hold, marker, t=t)
    print(f"mini-night {night_key(t)} run {a.run}: {action} ({why})")
    if action == "nothing":
        return 0
    if action == "mark-done":
        doc = json.dumps({"run": a.run, "outcome": marker.get("outcome"), "finished": marker.get("finished"),
                          "at": now_iso(t), "by": "mini-night"})
        cmd = (f"test -e {shlex.quote(job + '/' + DONE_FILE)} || {{ printf '%s\\n' {shlex.quote(doc)} > "
               f"{shlex.quote(job + '/' + DONE_FILE + '.tmp')} && mv {shlex.quote(job + '/' + DONE_FILE + '.tmp')} "
               f"{shlex.quote(job + '/' + DONE_FILE)}; }}")
        if not a.execute:
            print("  [plan] 4090: " + cmd)
            return 0
        rc, out = run(ssh + [cmd])
        print(f"  done marker on the 4090: rc {rc} {out[-200:]}")
        return 0 if rc == 0 else 1
    until = float(hold["until_epoch"])
    stop_at = datetime.fromtimestamp(until - a.margin_min * 60, ET).isoformat(timespec="seconds")
    hard_at = datetime.fromtimestamp(until - 300, ET).isoformat(timespec="seconds")
    argv = [sys.executable, "-I", str(Path(a.post_train).expanduser()), "arm"]
    if action == "arm-rerun":
        argv.append("--rerun")
    argv += ["--tower", a.tower, "--run", a.run, "--adapter-name", a.adapter_name, "--window-hold", job,
             "--stop-at", stop_at, "--hard-at", hard_at, "--trainer-unit", "none.service",
             "--watcher-unit", "none.service"]
    argv += shlex.split(a.post_train_args or "")
    if a.execute:
        argv.append("--execute")
    print("  " + shlex.join(argv))
    if not a.execute:
        print("  dry run: nothing armed (--execute arms it)")
        return 0
    rc, out = run(argv, timeout=300)
    print(out[-2000:])
    return 0 if rc == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_status(a: argparse.Namespace) -> int:
    st = load(Path(a.job_dir).expanduser())
    if not st:
        print(f"no job state in {a.job_dir}")
        return 1
    print(json.dumps({"run": st["run"], "phase": st["phase"], "failures": st.get("failures"),
                      "last_error": st.get("last_error"), "nights": [
                          {k: n.get(k) for k in ("night", "phase", "outcome", "step_from", "step_to", "total_steps",
                                                 "train_s", "reason")} for n in st["nights"]],
                      "tokens_per_night": tokens_per_night(st), "verdict": st.get("verdict"),
                      "post_train": st.get("post_train")}, indent=2, sort_keys=True))
    return 0


def cmd_estimate(a: argparse.Namespace) -> int:
    if a.dataset:
        lengths = dataset_lengths(Path(a.dataset).expanduser())
        src = f"{a.dataset} ({len(lengths)} rows with n_tokens)"
    else:
        lengths = synthetic_lengths(a.new_rows, replay_share=a.replay_share, lc_share=a.lc_share,
                                    lc_mean=a.lc_mean, other_mean=a.other_mean, replay_mean=a.replay_mean)
        src = (f"assumed: {a.new_rows} new rows ({a.lc_share:.0%} long-context at {a.lc_mean:,} tokens, the rest "
               f"at {a.other_mean:,}) + replay at {a.replay_share:.0%} of the final set at {a.replay_mean:,}")
    est = estimate(lengths, epochs=a.epochs, night_s=a.night_h * 3600, overhead=a.overhead, max_seq=a.max_seq)
    print(src)
    print(json.dumps(est, indent=2))
    print(f"-> {est['nights']} nights of {est['night_h']} h ({est['tokens'] / 1e6:.0f}M tokens over {a.epochs} "
          f"epochs at ~{est['eff_tok_s']:,} tok/s effective)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train_nights", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--job-dir", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--dataset", help="a build.py train.jsonl (its rows' n_tokens)")
    e.add_argument("--new-rows", type=int, default=3750)
    e.add_argument("--replay-share", type=float, default=0.4)
    e.add_argument("--lc-share", type=float, default=0.26)
    e.add_argument("--lc-mean", type=int, default=36000)
    e.add_argument("--other-mean", type=int, default=18000)
    e.add_argument("--replay-mean", type=int, default=13846)
    e.add_argument("--epochs", type=int, default=2)
    e.add_argument("--max-seq", type=int, default=49152)
    e.add_argument("--night-h", type=float, default=NIGHT_S / 3600)
    e.add_argument("--overhead", type=float, default=OVERHEAD)
    m = sub.add_parser("mini-night", help="on the 3090 Ti host: arm the post-train pipeline into tonight's hold")
    m.add_argument("--tower", required=True, help="ssh target of the 4090 host")
    m.add_argument("--run", required=True, help="the training run's directory name")
    m.add_argument("--adapter-name", default="adapter-max-derived")
    m.add_argument("--job-dir", default="", help="on the 4090, relative to its home "
                                                 "(default distill/phase4/results/base-9b/train-nights-<run>)")
    m.add_argument("--state-dir", default="", help="post_train.py's state dir here (default ~/distill/phase7/<run>)")
    m.add_argument("--post-train", default=str(Path(__file__).resolve().parent.parent / "post_train.py"))
    m.add_argument("--post-train-args", default="", help="extra post_train.py arguments (shell words)")
    m.add_argument("--margin-min", type=float, default=15.0, help="the pipeline stops this long before the hold ends")
    m.add_argument("--execute", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "status":
        return cmd_status(a)
    if a.cmd == "estimate":
        return cmd_estimate(a)
    return mini_night(a)


if __name__ == "__main__":
    sys.exit(main())
