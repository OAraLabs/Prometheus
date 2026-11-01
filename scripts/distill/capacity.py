#!/usr/bin/env python3
"""Teacher context and 9B QLoRA sequence-length capacity (operator, Sun 2026-11-01 ~01:30 ET).

    capacity.py mini [--execute]                                  # the 3090 Ti host, its window
    capacity.py block --tower USER@HOST --tower-url URL [--execute]   # the 4090 block (dry run without)
    capacity.py arm --tower USER@HOST --tower-url URL [--execute]     # start `block` as a waiting unit
    capacity.py report [--capdir DIR]                             # re-render DISTILL-CAPACITY.md
    capacity.py status

"Before long-context generation: measure the max stable context for the 27B teacher on the dedicated
4090 (Q4_K_XL and the IQ3 quant, q8 KV cache, flash attention), and the 3090 Ti's current limit. Use
the highest stable context for long-context teacher rows; cap long-context rows at what the teacher
can actually read. Also measure the longest sequence the 9B QLoRA can train at on the 4090 with
activation offload to its RAM (16K/32K/64K), and report tokens/sec at each."

Everything runs from the 3090 Ti host (the "here" host: the repo checkout, the wiki, the tokenizer,
the 4090's teacher key). Outputs go to ``~/distill/capacity/`` here.

Needle probes
-------------
A probe at context N is one request whose prompt fills ``N - (MAX_TOKENS + PROMPT_MARGIN)`` tokens
(measured with the student tokenizer over the teacher's base chat template: the two tokenizers count
identically, checked on the live teacher) of REAL repo docs/code and wiki text, built by
``build_eval_sets.LongctxBuilder`` (the eval-longctx builder; its own seed stream here, never an
eval case), with THREE invented notes at different depths (variant A: 10/50/90 %, read_file tool
results; variant B: 25/60/85 %, pasted files). The question asks for all three values; a reply is
correct only if it holds all three. Greedy decoding, thinking off. Each rung sends A and B (two full
prefills) and then a decode benchmark (B again: the prompt cache makes it decode-only, 128 tokens
with ignore_eos), so prefill tok/s comes from A and B and decode tok/s from the benchmark.

A rung is STABLE when the server loads (no OOM, the reviewed server's own checks and verify pass),
both needle requests answer all three needles, nothing crashed, and the VRAM headroom (nvidia-smi's
free VRAM, the minimum over the rung) stays >= HEADROOM_MIN_MIB. The ladder stops at the first
unstable rung; the max stable context is the last stable rung below it.

``mini`` (the 3090 Ti host, inside its window): the CURRENT teacher (distill-teacher.service on the
local model port, 2 slots sharing one unified KV of its configured context) is measured as it runs,
one request at a time through the generation runner's probe guard (its start window keeps the voice
probe clean; its forward lock serializes with generation), never during the nightly evals. Then the
headroom-limited maximum is COMPUTED from VRAM (free VRAM minus the headroom floor, at the measured
q8 KV bytes per token); the teacher is never restarted here.

``block`` (the 4090, as the waiting unit ``flywheel-capacity-<stamp>`` here, not distill-*): builds
the prompt set while it waits for the post-train pipeline's PIPELINE-DONE.json (skips with a
CAPACITY-DONE ``skipped`` if it fails or has not finished by ``--skip-at``), then: Telegram "<biz>
going down: teacher/trainer capacity tests", stops the business server through its unit under the
window lock (as post_train/train_full do), and measures, every GPU process a ``distill-cap-*`` user
unit on the 4090 (the continuous window's end and the dead-man stop them):

1. the UD-Q4_K_XL ladder: the reviewed teacher server (``serve_teacher.sh exec``: tailnet bind, key
   file, tools/agent/MCP/web UI off, env scrubbed, its own checks, then ``verify``) with only
   TEACHER_SLOTS=1 and TEACHER_CTX_PER_SLOT=N changed (it already runs -fa on, q8_0 K/V, no mmproj);
   its VRAM projection's overhead terms are zeroed, so it refuses only a context whose weights + KV
   alone exceed the card, and the real load decides the rest. Then one multi-slot load (2 x N/2) at
   the max stable N.
2. the IQ3_XXS ladder, the same, after copying the GGUF from here (sha256 verified on the 4090;
   started in the background at the block's start, only with the disk floor kept).
3. the 9B QLoRA sequence-length benchmark (``tower/seqlen_bench.py``, the trainer's own Unsloth
   backend with its offloaded gradient checkpointing, activations to CPU RAM) at 16384, 32768,
   65536, bracketed with 24576/49152 when one runs out of memory.

Then it writes ``teacher-ctx.json``, ``qlora-seqlen.json``, ``DISTILL-CAPACITY.md`` (scrubbed) and
``CAPACITY-DONE.json``, and Telegrams the headline. Business: LEFT DOWN for the cross-host
generation that follows (``business_left_down: true``) unless something failed or it is past
``--restore-after`` (Mon 04:00 EST): then ``train_end.sh`` restores it (its own "<biz> back" Telegram).
The block keeps to ``--budget-min`` (well under the generation chain's 3 h wait).

Exit: 0 finished (any outcome; CAPACITY-DONE says which), 1 crashed (the unit restarts and resumes),
2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import importlib.util
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo

    ET: Any = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - no tz database: UTC
    ET = timezone.utc

HERE = Path(__file__).resolve().parent

TEACHER_SCHEMA = "distill-capacity-teacher-ctx/1"
QLORA_SCHEMA = "distill-capacity-qlora-seqlen/1"
DONE_SCHEMA = "distill-capacity-done/1"
MINI_SCHEMA = "distill-capacity-mini/1"
STATE_SCHEMA = "distill-capacity-block/1"

LADDER_4090 = (32768, 49152, 65536, 98304, 131072, 163840, 196608, 262144)
MINI_LADDER = (16384, 32768, 49152)  # + the teacher's configured context
MAX_TOKENS = 512
PROMPT_MARGIN = 1024
FIT_SLACK = 256  # a built prompt must stay this far under N - MAX_TOKENS
HEADROOM_MIN_MIB = 300
KV_Q8_BYTES_PER_TOKEN = 34816  # 16 attention layers x 4 KV heads x 256 x K+V x 1.0625 B (serve_teacher.sh)
RS_MIB_PER_SLOT = 150  # recurrent state per slot (serve_teacher.sh)
CTX_GROWTH_MARGIN_MIB = 64  # compute-buffer growth with a larger context (the KQ mask), allowed for
DEPTHS = {"A": (10, 50, 90), "B": (25, 60, 85)}
FORMATS = {"A": "tool_results", "B": "pasted"}
DECODE_BENCH_TOKENS = 128
REPLY_RESERVE = 4096  # a long-context row's reply: reasoning budget 2048 + answer
REQUEST_TIMEOUT_S = 1800
LOAD_TIMEOUT_S = 600
IDLE_VRAM_MAX_MIB = 1500
DISK_FLOOR_GB = 15.0  # the 4090 window's MIN_FREE_DISK_GB: kept after the IQ3 copy
DISK_STOP_GB = 10.0  # post_train's floor: below it nothing starts (and a running block stops)
MAX_STARTS = 4
NOTIFY_LINES = 5
CAP_SEED = "capacity-2026-11-01"
QLORA_LENGTHS = (16384, 32768, 65536)
QLORA_BRACKET = {32768: 24576, 65536: 49152}
QLORA_STEPS = 5
QLORA_WARMUP = 1
QLORA_RUNTIME_S = 2400
QLORA_MEMORY_MAX = "54G"
EVAL_TIMES = ("03:00", "06:00")
EVAL_GUARD_MIN = 25
ACTIVE = ("active", "activating", "deactivating", "reloading")
LOOPBACK = "127.0.0.1"

# The two 4090 teacher files. Paths are relative to the 4090 user's home.
QUANTS: Dict[str, Dict[str, Any]] = {
    "q4": {"label": "UD-Q4_K_XL", "file": "models/Qwen3.8-27B-UD-Q4_K_XL.gguf", "bytes": 17923394624,
           "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
           "manifest": "distill/models/qwen38-27b-gguf/MANIFEST.json", "alias": "qwen3.8-27b-ud-q4_k_xl",
           "weights_mib": 16401},
    "iq3": {"label": "IQ3_XXS", "file": "distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf",
            "bytes": 10094357632, "sha256": "fdfcb6a29b11188956dfbfd904223588a6c1b77eb250c3e8a36e1bd269df91f7",
            "manifest": "distill/models/teacher/MANIFEST.json", "alias": "qwen3.8-27b-gsq-rco-iq3_xxs",
            "weights_mib": 9000},
}
# here (the 3090 Ti host): the IQ3 original and its manifest
IQ3_SRC = "distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"
IQ3_SRC_MANIFEST = "distill/models/teacher/MANIFEST.json"


class Stop(Exception):
    """Stop GPU work now (budget, flag, business back from elsewhere, disk, signal)."""

    def __init__(self, why: str, failure: bool = True) -> None:
        super().__init__(why)
        self.failure = failure


class NotRun(Exception):
    """End without touching the GPU (pipeline failed or late, a flag, the GPU busy)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def et(ts: float) -> str:
    return datetime.fromtimestamp(ts, ET).strftime("%a %H:%M %Z")


def parse_iso(value: str) -> float:
    d = datetime.fromisoformat(value)
    if d.tzinfo is None:
        d = d.replace(tzinfo=ET)
    return d.timestamp()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def kv_lines(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ln in text.splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            out.setdefault(k.strip(), []).append(v.strip())
    return out


def kv1(kv: Dict[str, List[str]], key: str, default: str = "") -> str:
    v = kv.get(key)
    return v[-1] if v else default


def to_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return default


def to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def k_label(n: Optional[int]) -> str:
    if not n:
        return "-"
    return f"{n // 1024}K" if n % 1024 == 0 else f"{n:,}"


def load_sibling(name: str, filename: str) -> Any:
    """A module beside this file, by path (``python3 -I`` puts no script directory on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"{filename} not found beside capacity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def bes() -> Any:
    return load_sibling("distill_build_eval_sets", "build_eval_sets.py")


q = shlex.quote


# ---------------------------------------------------------------------------
# Ladder and stability logic (pure)
# ---------------------------------------------------------------------------


def ladder_for(n_ctx_train: Optional[int], base: Sequence[int] = LADDER_4090) -> List[int]:
    """The rungs to try: ``base`` up to the model's trained context, plus that context when it is
    larger than the top rung (a rung above it would need rope scaling: dropped)."""
    rungs = sorted(set(int(x) for x in base))
    if n_ctx_train:
        rungs = [r for r in rungs if r <= n_ctx_train]
        if not rungs or n_ctx_train > rungs[-1]:
            rungs.append(int(n_ctx_train))
    return rungs


def prompt_target(ctx: int, max_tokens: int = MAX_TOKENS, margin: int = PROMPT_MARGIN) -> int:
    """Prompt tokens for a probe at context ``ctx``: N - (max_tokens + margin)."""
    t = int(ctx) - int(max_tokens) - int(margin)
    if t <= 0:
        raise ValueError(f"context {ctx} leaves no room for a prompt")
    return t


def prompt_limit(ctx: int, max_tokens: int = MAX_TOKENS, slack: int = FIT_SLACK) -> int:
    """The most prompt tokens a built probe may have at context ``ctx``."""
    return int(ctx) - int(max_tokens) - int(slack)


def judge_reply(text: Optional[str], values: Sequence[str]) -> List[bool]:
    """Per needle: its value appears in the reply (case-insensitive). Values are invented codes
    found nowhere in the haystack's sources."""
    t = (text or "").casefold()
    return [bool(v) and v.casefold() in t for v in values]


def rung_verdict(r: Dict[str, Any], headroom_min: int = HEADROOM_MIN_MIB,
                 requests: int = 2) -> Tuple[bool, str]:
    """(stable, why) for one measured rung (see the module doc)."""
    if r.get("load") != "ok":
        return False, f"did not load: {r.get('load_detail') or r.get('load') or 'unknown'}"
    reqs = [x for x in (r.get("requests") or []) if x.get("kind") == "needles"]
    if r.get("crashed"):
        return False, f"server crashed: {r.get('crash_detail') or 'exited during the rung'}"
    errs = [x for x in reqs if x.get("error")]
    if errs:
        return False, f"request failed: {errs[0].get('error')}"
    if len(reqs) < requests:
        return False, f"only {len(reqs)} of {requests} needle requests ran"
    missed = [x for x in reqs if not x.get("all_correct")]
    if missed:
        got = "; ".join(f"{x.get('variant')}: {sum(bool(c) for c in x.get('correct') or [])}/"
                        f"{len(x.get('correct') or [])}" for x in missed)
        return False, f"needles missed ({got})"
    hr = r.get("headroom_min_mib")
    if hr is None:
        return False, "VRAM headroom unknown"
    if hr < headroom_min:
        return False, f"VRAM headroom {hr} MiB < {headroom_min}"
    return True, "stable"


def next_rung(ladder: Sequence[int], rungs: Sequence[Dict[str, Any]]) -> Optional[int]:
    """The next rung to measure, or None: the ladder is done, or a rung was unstable."""
    done = {int(r["ctx"]) for r in rungs if r.get("ctx") is not None and r.get("final")}
    if any(r.get("final") and not r.get("stable") for r in rungs):
        return None
    for n in ladder:
        if n not in done:
            return n
    return None


def ladder_summary(ladder: Sequence[int], rungs: Sequence[Dict[str, Any]],
                   top_label: str = "top of the ladder (the model's trained context)") -> Dict[str, Any]:
    final = sorted((r for r in rungs if r.get("final")), key=lambda r: int(r["ctx"]))
    stable = [r for r in final if r.get("stable")]
    unstable = [r for r in final if not r.get("stable")]
    best = stable[-1] if stable else None
    first_bad = unstable[0] if unstable else None
    top_reached = bool(stable) and int(stable[-1]["ctx"]) == max(ladder) if ladder else False
    if first_bad is not None:
        bound = "unstable rung above"
    elif top_reached:
        bound = top_label
    else:
        bound = "not bounded (ladder not finished)"
    out: Dict[str, Any] = {
        "max_stable_ctx": int(best["ctx"]) if best else None,
        "first_unstable": ({"ctx": int(first_bad["ctx"]), "reason": first_bad.get("reason")}
                           if first_bad else None),
        "bounded_by": bound,
        "rungs_measured": [int(r["ctx"]) for r in final],
    }
    if best:
        out["at_max"] = {k: best.get(k) for k in ("prefill_tok_s", "decode_tok_s", "vram_used_mib",
                                                  "headroom_min_mib", "max_verified_prompt_tokens")}
    return out


def kv_fit(rungs: Sequence[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Least squares of load VRAM (MiB) on context over the rungs that loaded with one slot:
    {bytes_per_token, intercept_mib, points}."""
    pts = [(float(r["ctx"]), float(r["vram_used_mib"])) for r in rungs
           if r.get("load") == "ok" and r.get("vram_used_mib") is not None and int(r.get("slots") or 1) == 1]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    return {"bytes_per_token": round(slope * 1048576, 1), "intercept_mib": round(my - slope * mx, 1),
            "points": n}


def computed_max_ctx(free_mib: float, current_ctx: int, *, kv_bytes: float = KV_Q8_BYTES_PER_TOKEN,
                     headroom_mib: float = HEADROOM_MIN_MIB, growth_mib: float = CTX_GROWTH_MARGIN_MIB,
                     n_ctx_train: Optional[int] = None, step: int = 1024) -> int:
    """The largest context (a multiple of ``step``) a running server could be restarted with, from
    its free VRAM: the spare after the headroom floor and a compute-buffer growth allowance, at
    ``kv_bytes`` per token on top of what it holds now. Never below the current context's floor
    multiple; capped at the trained context."""
    spare = float(free_mib) - float(headroom_mib) - float(growth_mib)
    extra = int(max(0.0, spare) * 1048576 // float(kv_bytes)) if spare > 0 else 0
    total = (int(current_ctx) + extra) // step * step
    if spare < 0:
        # already under the floor: shrink by the deficit
        deficit = int(math.ceil(-spare * 1048576 / float(kv_bytes)))
        total = max(0, (int(current_ctx) - deficit) // step * step)
    if n_ctx_train:
        total = min(total, int(n_ctx_train))
    return total


def multi_slot_alternatives(total_ctx: Optional[int], fit: Optional[Dict[str, float]], usable_mib: Optional[float],
                            *, rs_mib: float = RS_MIB_PER_SLOT, headroom: float = HEADROOM_MIN_MIB,
                            slot_counts: Sequence[int] = (2, 4)) -> List[Dict[str, Any]]:
    """Projected multi-slot shapes holding the same total context (computed from the measured load
    VRAM line): each slot adds one recurrent state."""
    if not total_ctx or not fit or not usable_mib:
        return []
    out = []
    for s in slot_counts:
        per = int(total_ctx) // s // 1024 * 1024
        if per < 8192:
            continue
        proj = fit["intercept_mib"] + fit["bytes_per_token"] * per * s / 1048576 + rs_mib * (s - 1)
        out.append({"slots": s, "ctx_per_slot": per, "projected_mib": round(proj),
                    "fits": bool(proj + headroom <= usable_mib), "source": "computed"})
    return out


def qlora_next(results: Dict[str, Dict[str, Any]], lengths: Sequence[int] = QLORA_LENGTHS,
               bracket: Optional[Dict[int, int]] = None) -> Optional[int]:
    """The next sequence length to benchmark, or None. 16K -> 32K -> 64K; a length that runs out
    of memory is bracketed once (32K -> 24K, 64K -> 48K) and nothing longer is tried; a failure
    that is not out-of-memory ends the benchmark."""
    br = QLORA_BRACKET if bracket is None else bracket
    for n in lengths:
        r = results.get(str(n))
        if r is None:
            return n
        st = r.get("status")
        if st == "ok":
            continue
        if st == "oom":
            b = br.get(n)
            if b is not None and str(b) not in results:
                return b
            return None
        return None
    return None


def qlora_summary(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ok = sorted(int(k) for k, v in results.items() if v.get("status") == "ok")
    oom = sorted(int(k) for k, v in results.items() if v.get("status") == "oom")
    bad = sorted(int(k) for k, v in results.items() if v.get("status") not in ("ok", "oom"))
    return {"max_ok_seq": ok[-1] if ok else None, "first_oom_seq": oom[0] if oom else None,
            "failed": bad, "lengths_run": sorted(int(k) for k in results)}


def recommend(quants: Dict[str, Dict[str, Any]], mini: Optional[Dict[str, Any]] = None,
              reply_reserve: int = REPLY_RESERVE) -> Dict[str, Any]:
    """The long-context teacher: the measured quant with the highest max stable context on the
    4090 (a tie goes to the bigger quant, Q4_K_XL); rows are capped at what it read."""
    best_key, best = None, None
    for key in ("q4", "iq3"):
        s = (quants.get(key) or {}).get("summary") or {}
        m = s.get("max_stable_ctx")
        if m and (best is None or m > best):
            best_key, best = key, m
    rec: Dict[str, Any] = {"long_context_teacher": None, "long_context_row_cap_tokens": None,
                           "max_verified_prompt_tokens": None, "notes": []}
    if best_key:
        q = quants[best_key]
        s = q["summary"]
        rec["long_context_teacher"] = {"host": "4090", "quant": q.get("label"), "ctx": best,
                                       "profile": {"parallel": 1, "ctx": best, "flash_attn": "on",
                                                   "cache_type_k": "q8_0", "cache_type_v": "q8_0"}}
        rec["max_verified_prompt_tokens"] = (s.get("at_max") or {}).get("max_verified_prompt_tokens")
        rec["long_context_row_cap_tokens"] = best - reply_reserve
        rec["notes"].append(f"rows above {k_label(best - reply_reserve)} tokens (the {k_label(best)} "
                            f"context less a {reply_reserve:,}-token reply reserve) are not generated")
        other = "iq3" if best_key == "q4" else "q4"
        om = ((quants.get(other) or {}).get("summary") or {}).get("max_stable_ctx")
        if om and om < best:
            rec["notes"].append(f"{quants[other].get('label')} reads up to {k_label(om)}: rows up to "
                                f"{k_label(om - reply_reserve)} can use it too")
    if mini:
        mr = mini.get("max_readable_ctx")
        if mr:
            rec["notes"].append(f"the 3090 Ti teacher (as configured) reads up to {k_label(mr)}: rows up to "
                                f"{k_label(mr - reply_reserve)} can be generated there")
    return rec


def eval_clear(now: float, times: Sequence[str] = EVAL_TIMES, guard_min: float = EVAL_GUARD_MIN,
               need_s: float = 0.0) -> Tuple[bool, str]:
    """Whether a request taking ``need_s`` may start at ``now`` without overlapping a nightly eval
    (03:00 / 06:00 ET wall time, +- ``guard_min``)."""
    from datetime import timedelta

    today = datetime.fromtimestamp(now, ET).date()
    for day in (-1, 0, 1):
        dd = today + timedelta(days=day)
        for t in times:
            hh, mm = (int(x) for x in t.split(":"))
            when = datetime(dd.year, dd.month, dd.day, hh, mm, tzinfo=ET).timestamp()
            lo, hi = when - guard_min * 60, when + guard_min * 60
            if now < hi and now + need_s > lo:
                return False, f"the {t} ET nightly eval ({datetime.fromtimestamp(when, ET):%a %H:%M %Z}) is too close"
    return True, ""


# ---------------------------------------------------------------------------
# Output documents
# ---------------------------------------------------------------------------


def validate_teacher_ctx(doc: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(doc, dict):
        return ["not an object"]
    if doc.get("schema") != TEACHER_SCHEMA:
        errs.append(f"schema is {doc.get('schema')!r}")
    hosts = doc.get("hosts")
    if not isinstance(hosts, dict):
        return errs + ["no hosts"]
    for host, h in hosts.items():
        if host == "4090":
            for key, qd in (h.get("quants") or {}).items():
                if key not in QUANTS:
                    errs.append(f"4090: unknown quant {key}")
                if qd.get("status") not in ("measured", "partial", "skipped", "failed"):
                    errs.append(f"4090 {key}: status {qd.get('status')!r}")
                s = qd.get("summary") or {}
                if qd.get("status") in ("measured", "partial") and "max_stable_ctx" not in s:
                    errs.append(f"4090 {key}: no max_stable_ctx")
                for r in qd.get("rungs") or []:
                    for f in ("ctx", "load", "stable", "reason"):
                        if f not in r:
                            errs.append(f"4090 {key} rung {r.get('ctx')}: no {f}")
        elif host == "3090ti":
            if "measured" not in h or "computed" not in h:
                errs.append("3090ti: measured and computed are both required")
        else:
            errs.append(f"unknown host {host}")
    if "recommendation" not in doc:
        errs.append("no recommendation")
    return errs


def validate_qlora(doc: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(doc, dict):
        return ["not an object"]
    if doc.get("schema") != QLORA_SCHEMA:
        errs.append(f"schema is {doc.get('schema')!r}")
    lens = doc.get("lengths")
    if not isinstance(lens, dict):
        return errs + ["no lengths"]
    for k, v in lens.items():
        if not str(k).isdigit():
            errs.append(f"length key {k!r}")
        if v.get("status") not in ("ok", "oom", "failed", "killed", "skipped"):
            errs.append(f"{k}: status {v.get('status')!r}")
        if v.get("status") == "ok" and not v.get("tok_s"):
            errs.append(f"{k}: ok without tok_s")
    if "summary" not in doc:
        errs.append("no summary")
    return errs


def validate_done(doc: Any) -> List[str]:
    errs: List[str] = []
    if not isinstance(doc, dict):
        return ["not an object"]
    if doc.get("schema") != DONE_SCHEMA:
        errs.append(f"schema is {doc.get('schema')!r}")
    if doc.get("status") not in ("done", "partial", "failed", "skipped"):
        errs.append(f"status {doc.get('status')!r}")
    if not isinstance(doc.get("business_left_down"), bool):
        errs.append("business_left_down must be true or false")
    if not doc.get("finished"):
        errs.append("no finished time")
    return errs


_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PATH_RE = re.compile(r"(/home/|/Users/|~/|\.ts\.net)")


def scrub_problems(text: str, banned: Sequence[str] = ()) -> List[str]:
    """What a committed document must not hold: IPs, home paths, tailnet names, banned words."""
    out = []
    if _IP_RE.search(text):
        out.append("an IPv4 address")
    m = _PATH_RE.search(text)
    if m:
        out.append(f"a path or host name ({m.group(1)})")
    for b in banned:
        if b and b.casefold() in text.casefold():
            out.append("a banned name")
    return out


def _f(x: Any, nd: int = 0) -> str:
    v = to_float(x)
    if v is None:
        return "-"
    return f"{v:,.{nd}f}"


def render_markdown(teacher: Optional[Dict[str, Any]], qlora: Optional[Dict[str, Any]],
                    done: Optional[Dict[str, Any]], *, pending_4090: bool = False) -> str:
    """DISTILL-CAPACITY.md from the result documents: numbers only, no paths, prompts or names."""
    L: List[str] = ["# Distillation capacity: teacher context and 9B QLoRA sequence length", ""]
    L.append("Operator, Sun 2026-11-01 ~01:30 ET: measure the 27B teacher's max stable context on the 4090 "
             "(UD-Q4_K_XL and IQ3_XXS, q8_0 KV cache, flash attention) and the 3090 Ti's current limit; "
             "use the highest stable context for long-context teacher rows and cap them at what the teacher "
             "can read; measure the longest sequence the 9B QLoRA trains at on the 4090 with activation "
             "offload to RAM (16K/32K/64K), with tokens/sec.")
    L.append("")
    L.append("Code: `scripts/distill/capacity.py` (probes, ladder, block, outputs) and "
             "`scripts/distill/tower/seqlen_bench.py` (the QLoRA benchmark).")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- **Needle probe at context N:** one prompt of N - ({MAX_TOKENS} + {PROMPT_MARGIN}) tokens of real "
             "repo docs, code and wiki text (the eval-longctx builder with its own seed; no eval case), with "
             "three invented notes at different depths (A: 10/50/90 %, as read_file results; B: 25/60/85 %, "
             "pasted files). Correct = the reply holds all three values. Greedy, thinking off.")
    L.append("- **Each rung:** A and B (two full prefills; prefill tok/s from them), then a 128-token decode "
             "benchmark on B's cached prompt (decode tok/s).")
    L.append(f"- **Stable:** the server loads, both requests answer all three needles, nothing crashes, and "
             f"VRAM headroom (nvidia-smi free, minimum over the rung) stays >= {HEADROOM_MIN_MIB} MiB. "
             "The ladder stops at the first unstable rung.")
    L.append("- **4090 server:** the reviewed teacher server (tailnet bind, key file, tools/agent/MCP/web UI "
             "off) with only `--parallel 1 -c N` changed; it already runs `-fa on -ctk q8_0 -ctv q8_0` "
             "and no mmproj.")
    L.append("")
    if teacher:
        hosts = teacher.get("hosts") or {}
        t4 = hosts.get("4090")
        if t4:
            L.append("## 4090 teacher (dedicated GPU)")
            L.append("")
            for key, qd in (t4.get("quants") or {}).items():
                s = qd.get("summary") or {}
                L.append(f"### {qd.get('label', key)}: max stable context "
                         f"**{k_label(s.get('max_stable_ctx'))}** ({qd.get('status')})")
                L.append("")
                if qd.get("skip_reason"):
                    L.append(f"Not measured: {qd['skip_reason']}.")
                    L.append("")
                rungs = qd.get("rungs") or []
                if rungs:
                    L.append("| Context | Load VRAM MiB | Min headroom MiB | Prompt tokens | Prefill tok/s | "
                             "Decode tok/s | Needles A / B | Verdict |")
                    L.append("|---|---:|---:|---:|---:|---:|---|---|")
                    for r in rungs:
                        reqs = {x.get("variant"): x for x in r.get("requests") or [] if x.get("kind") == "needles"}

                        def nd(v: str) -> str:
                            x = reqs.get(v)
                            if not x:
                                return "-"
                            c = x.get("correct") or []
                            return f"{sum(bool(i) for i in c)}/{len(c)}"
                        L.append(f"| {k_label(r.get('ctx'))} | {_f(r.get('vram_used_mib'))} | "
                                 f"{_f(r.get('headroom_min_mib'))} | {_f(r.get('max_verified_prompt_tokens'))} | "
                                 f"{_f(r.get('prefill_tok_s'))} | {_f(r.get('decode_tok_s'), 1)} | "
                                 f"{nd('A')} / {nd('B')} | {'stable' if r.get('stable') else r.get('reason')} |")
                    L.append("")
                fit = qd.get("kv_fit")
                if fit:
                    L.append(f"Load VRAM vs context: {_f(fit.get('bytes_per_token'))} bytes/token over "
                             f"{_f(fit.get('intercept_mib'))} MiB ({fit.get('points')} points; the plan's q8_0 "
                             f"figure is {KV_Q8_BYTES_PER_TOKEN:,}).")
                    L.append("")
                ms = qd.get("multi_slot") or []
                if ms:
                    parts = []
                    for m in ms:
                        tag = "measured" if m.get("source") == "measured" else "projected"
                        ok = "fits" if m.get("fits") else "does not fit"
                        parts.append(f"{m['slots']} x {k_label(m['ctx_per_slot'])} ({tag}, "
                                     f"{_f(m.get('vram_used_mib') or m.get('projected_mib'))} MiB, {ok})")
                    L.append("Multi-slot at the same total: " + "; ".join(parts) + ".")
                    L.append("")
        elif pending_4090:
            L.append("## 4090 teacher (dedicated GPU)")
            L.append("")
            L.append("Pending: the measurement block is armed on the 3090 Ti host and starts when the "
                     "post-train pipeline finishes (Sun afternoon EST); this file is re-rendered with its "
                     "results.")
            L.append("")
        mini = hosts.get("3090ti")
        if mini:
            m, c = mini.get("measured") or {}, mini.get("computed") or {}
            L.append("## 3090 Ti teacher (shared GPU, as configured)")
            L.append("")
            cfg = mini.get("current_config") or {}
            L.append(f"Current: {cfg.get('quant', 'IQ3_XXS')}, {cfg.get('slots')} slots sharing one "
                     f"{'unified ' if cfg.get('kv_unified') else ''}KV of {k_label(cfg.get('ctx'))}, "
                     "beside the voice stack and Ollama. Probes went through the generation runner's probe "
                     "guard, one at a time, outside the nightly evals.")
            L.append("")
            rungs = m.get("rungs") or []
            if rungs:
                L.append("| Context | Prompt tokens | Prefill tok/s | Decode tok/s | Needles A / B | Min free VRAM MiB | Verdict |")
                L.append("|---|---:|---:|---:|---|---:|---|")
                for r in rungs:
                    reqs = {x.get("variant"): x for x in r.get("requests") or [] if x.get("kind") == "needles"}
                    cells = []
                    for v in ("A", "B"):
                        x = reqs.get(v)
                        cc = (x or {}).get("correct") or []
                        cells.append(f"{sum(bool(i) for i in cc)}/{len(cc)}" if x else "-")
                    L.append(f"| {k_label(r.get('ctx'))} | {_f(r.get('max_verified_prompt_tokens'))} | "
                             f"{_f(r.get('prefill_tok_s'))} | {_f(r.get('decode_tok_s'), 1)} | "
                             f"{cells[0]} / {cells[1]} | {_f(r.get('headroom_min_mib'))} | "
                             f"{'stable' if r.get('stable') else r.get('reason')} |")
                L.append("")
            L.append(f"- **Measured:** reads up to **{k_label(m.get('max_readable_ctx'))}** "
                     f"({m.get('bounded_by', '')}).")
            L.append(f"- **Computed (VRAM):** min free {_f(c.get('free_mib_min'))} MiB over the measurement; "
                     f"with the {HEADROOM_MIN_MIB} MiB floor and {CTX_GROWTH_MARGIN_MIB} MiB for buffer growth at "
                     f"{_f(c.get('kv_bytes_per_token'))} bytes/token, the most this config could hold is "
                     f"**{k_label(c.get('max_ctx_current_residents'))}**; with Ollama unloaded (nightly windows, "
                     f"Q9) {k_label(c.get('max_ctx_if_ollama_unloaded'))}.")
            L.append(f"- **Restart with more context:** {'advised' if c.get('restart_advised') else 'not advised'}"
                     f" ({c.get('restart_why', '')}).")
            L.append("")
        rec = teacher.get("recommendation") or {}
        if rec.get("long_context_teacher") or rec.get("notes"):
            L.append("## Recommendation")
            L.append("")
            lt = rec.get("long_context_teacher")
            if lt:
                L.append(f"- Long-context teacher rows: the 4090 with {lt.get('quant')}, one slot of "
                         f"{k_label(lt.get('ctx'))} (`--parallel 1 -c {lt.get('ctx')}`, q8_0 KV, flash attention).")
                L.append(f"- Row cap: **{k_label(rec.get('long_context_row_cap_tokens'))} tokens** of prompt "
                         f"(largest prompt read correctly: {_f(rec.get('max_verified_prompt_tokens'))} tokens).")
            for n in rec.get("notes") or []:
                L.append(f"- {n[0].upper() + n[1:]}.")
            L.append("")
    if qlora:
        L.append("## 9B QLoRA sequence length (4090, activation offload to RAM)")
        L.append("")
        cfg = qlora.get("config") or {}
        L.append(f"The trainer's own backend (`train_qlora.py` UnslothBackend: QLoRA r={cfg.get('r')}, "
                 f"8-bit AdamW, micro-batch 1) with Unsloth's offloaded gradient checkpointing (each "
                 f"checkpointed layer input goes to CPU RAM). {qlora.get('steps', QLORA_STEPS)} timed optimizer "
                 f"steps after {qlora.get('warmup_steps', QLORA_WARMUP)} warm-up, one sequence of the target "
                 "length per step, built by concatenating real v4b train rows; no adapter kept.")
        L.append("")
        L.append("| Max seq | Status | Tokens/s | Step s | Peak VRAM GiB (torch / nvidia-smi) | Peak RAM GiB (process / host used) |")
        L.append("|---:|---|---:|---:|---|---|")
        for k in sorted((qlora.get("lengths") or {}), key=int):
            v = qlora["lengths"][k]
            st = v.get("status")
            if st != "ok" and v.get("detail"):
                st = f"{st}: {v['detail']}"
            L.append(f"| {int(k):,} | {st} | {_f(v.get('tok_s'))} | {_f(v.get('step_s_median'), 1)} | "
                     f"{_f(v.get('peak_vram_reserved_gib'), 1)} / {_f(v.get('peak_vram_smi_gib'), 1)} | "
                     f"{_f(v.get('peak_rss_gib'), 1)} / {_f(v.get('host_used_peak_gib'), 1)} |")
        L.append("")
        s = qlora.get("summary") or {}
        L.append(f"Longest that trained: **{k_label(s.get('max_ok_seq'))}**"
                 + (f"; first out of memory: {k_label(s.get('first_oom_seq'))}." if s.get("first_oom_seq") else "."))
        L.append("")
    if done:
        L.append("## Block")
        L.append("")
        L.append(f"Status **{done.get('status')}**" + (f" ({done.get('reason')})" if done.get("reason") else "")
                 + f"; business server left down for generation: {str(done.get('business_left_down')).lower()}.")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Needle prompts (real repo/wiki text, three needles)
# ---------------------------------------------------------------------------


class ProbeBuilder:
    """Builds needle probes with build_eval_sets.LongctxBuilder over the host's real sources."""

    def __init__(self, docs: List[Any], meter: Any, system_prompt: str, tolerance: float = 0.002) -> None:
        B = bes()
        self.B = B
        self.lb = B.LongctxBuilder(docs, meter, seed=0, system_prompt=system_prompt, tools=[],
                                   tolerance=tolerance)
        self.count = meter.counter.count
        self.meter = meter

    @classmethod
    def from_host(cls, repo: Path, wiki: Optional[Path], tokenizer_dir: Path, template: Optional[Path]) -> "ProbeBuilder":
        B = bes()
        tracked = B.git_tracked(repo)
        docs = B.load_sources(repo, wiki, B.DEFAULT_WIKI_SECTIONS, tracked=tracked)
        counter = B.HFTokenCounter(tokenizer_dir / "tokenizer.json", name="qwen3.5-9b-tokenizer")
        tmpl = template.read_text(encoding="utf-8") if template and template.is_file() else None
        meter = B.PromptMeter(counter, tmpl)
        system_prompt, _ = B.load_system_prompt(repo)
        return cls(docs, meter, system_prompt)

    def build(self, ctx: int, variant: str, host_tag: str, *, max_tokens: int = MAX_TOKENS,
              margin: int = PROMPT_MARGIN) -> Dict[str, Any]:
        B, lb = self.B, self.lb
        rng = random.Random(f"{CAP_SEED}|{host_tag}|{ctx}|{variant}")
        used: set = set()
        needles, asks, values = [], [], []
        preds = list(B._RECALL_PREDICATES)
        rng.shuffle(preds)
        for i, d in enumerate(DEPTHS[variant]):
            subject = lb._subject(rng, used)
            pred, tmpl = preds[i % len(preds)]
            value = lb._fresh(rng, lambda r: lb._value("code", r), used)
            needles.append(B.Needle(tmpl.format(subject=subject, value=value), "answer", d))
            asks.append(f"{i + 1}) What is the {pred} for {subject}?")
            values.append(value)
        request = ("Based only on the files above, answer these three questions.\n" + "\n".join(asks)
                   + "\nReply with exactly three lines of the form '1: <value>', '2: <value>', '3: <value>' "
                     "and nothing else.")
        fmt = FORMATS[variant]
        intro = rng.choice(B._INTROS_TOOL if fmt == "tool_results" else B._INTROS_PASTED)
        target = prompt_target(ctx, max_tokens, margin)
        limit = prompt_limit(ctx, max_tokens)
        fit = None
        for _ in range(4):
            order = lb._mixed_order(random.Random(f"{CAP_SEED}|{host_tag}|{ctx}|{variant}|order"))
            fit = lb._fit(order, fmt, needles, intro, request, None, target)
            if fit.total <= limit:
                break
            target -= (fit.total - limit) + FIT_SLACK
        assert fit is not None
        if fit.total > limit:
            raise RuntimeError(f"could not fit a probe under {limit} tokens at context {ctx}")
        hay_tokens = max(self.count(fit.haystack), 1)
        depth_pct = []
        for nd in needles:
            at = fit.haystack.find(nd.text)
            if at < 0 or fit.haystack.count(nd.text) != 1:
                raise RuntimeError("a needle was not placed exactly once")
            depth_pct.append(round(100.0 * self.count(fit.haystack[:at]) / hay_tokens, 1))
        return {"ctx": ctx, "variant": variant, "format": fmt, "target_tokens": target,
                "prompt_tokens": int(fit.total), "count_method": self.meter.method,
                "messages": fit.messages, "values": values, "depths": list(DEPTHS[variant]),
                "depth_pct": depth_pct, "files": len(fit.spans)}


def probe_path(pdir: Path, host_tag: str, ctx: int, variant: str) -> Path:
    return pdir / host_tag / f"probe-{ctx}-{variant}.json"


def ensure_probe(builder_fn: Callable[[], ProbeBuilder], pdir: Path, host_tag: str, ctx: int, variant: str,
                 log: Callable[[str], None]) -> Dict[str, Any]:
    """A probe from disk (built earlier), else built now and kept (0600, private: real wiki text)."""
    p = probe_path(pdir, host_tag, ctx, variant)
    doc = read_json(p)
    if isinstance(doc, dict) and doc.get("ctx") == ctx and doc.get("variant") == variant and doc.get("messages"):
        return doc
    t0 = time.monotonic()
    doc = builder_fn().build(ctx, variant, host_tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(p.parent, 0o700)
    write_json(p, doc)
    log(f"  built probe {host_tag} {ctx} {variant}: {doc['prompt_tokens']:,} tokens, "
        f"{doc['files']} files, needles at {doc['depth_pct']} % ({time.monotonic() - t0:.0f} s)")
    return doc


# ---------------------------------------------------------------------------
# llama-server client
# ---------------------------------------------------------------------------

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read_key(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    with open(path, encoding="ascii") as fh:
        for ln in fh:
            s = ln.strip()
            if s and not s.startswith("#"):
                return s
    return None


def http_json(url: str, *, body: Optional[Dict[str, Any]] = None, key: Optional[str] = None,
              timeout: float = 30.0) -> Tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw or b"null")
            except ValueError:
                return resp.status, None
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
            doc = json.loads(raw or b"null")
        except Exception:  # noqa: BLE001
            doc = None
        return exc.code, doc
    except Exception as exc:  # noqa: BLE001 - refused, reset, timeout
        return 0, {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def _err_text(doc: Any) -> str:
    if isinstance(doc, dict):
        e = doc.get("error")
        if isinstance(e, dict):
            return str(e.get("message") or e.get("type") or e)[:200]
        if e:
            return str(e)[:200]
    return ""


def chat(url: str, messages: List[Dict[str, Any]], *, key: Optional[str] = None, max_tokens: int = MAX_TOKENS,
         ignore_eos: bool = False, timeout: float = REQUEST_TIMEOUT_S) -> Dict[str, Any]:
    """One completion; never returns the prompt. {http, content, finish, usage, timings, wall_s, error}."""
    body: Dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1,
                            "stream": False, "cache_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    t0 = time.monotonic()
    code, doc = http_json(url.rstrip("/") + "/v1/chat/completions", body=body, key=key, timeout=timeout)
    out: Dict[str, Any] = {"http": code, "wall_s": round(time.monotonic() - t0, 1)}
    if code != 200 or not isinstance(doc, dict):
        out["error"] = f"HTTP {code}: {_err_text(doc) or 'no answer'}"
        return out
    ch = (doc.get("choices") or [{}])[0] or {}
    msg = ch.get("message") or {}
    out["content"] = msg.get("content") or ""
    out["finish"] = ch.get("finish_reason")
    out["usage"] = doc.get("usage") or {}
    t = doc.get("timings") or {}
    out["timings"] = {k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n",
                                            "predicted_ms", "predicted_per_second", "cache_n")}
    return out


def needle_request(url: str, probe: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
    r = chat(url, probe["messages"], key=key)
    rec: Dict[str, Any] = {"kind": "needles", "variant": probe["variant"], "prompt_tokens_built": probe["prompt_tokens"],
                           "wall_s": r.get("wall_s")}
    if r.get("error"):
        rec["error"] = r["error"]
        return rec
    correct = judge_reply(r.get("content"), probe["values"])
    t = r.get("timings") or {}
    rec.update({"correct": correct, "all_correct": all(correct) and len(correct) == len(probe["values"]),
                "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
                "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                "prefill_tok_s": to_float(t.get("prompt_per_second")),
                "decode_tok_s": to_float(t.get("predicted_per_second")),
                "completion_tokens": (r.get("usage") or {}).get("completion_tokens"), "finish": r.get("finish")})
    return rec


def decode_request(url: str, probe: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
    r = chat(url, probe["messages"], key=key, max_tokens=DECODE_BENCH_TOKENS, ignore_eos=True)
    rec: Dict[str, Any] = {"kind": "decode", "variant": probe["variant"], "wall_s": r.get("wall_s")}
    if r.get("error"):
        rec["error"] = r["error"]
        return rec
    t = r.get("timings") or {}
    rec.update({"predicted_n": t.get("predicted_n"), "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
                "decode_tok_s": to_float(t.get("predicted_per_second"))})
    return rec


def rung_speeds(rung: Dict[str, Any]) -> None:
    """prefill/decode tok/s and the largest verified prompt, from the rung's requests."""
    reqs = rung.get("requests") or []
    pre = [x["prefill_tok_s"] for x in reqs if x.get("kind") == "needles" and x.get("prefill_tok_s")
           and (x.get("prompt_n") or 0) > 1024]
    if pre:
        rung["prefill_tok_s"] = round(sum(pre) / len(pre), 1)
    dec = [x["decode_tok_s"] for x in reqs if x.get("kind") == "decode" and x.get("decode_tok_s")]
    if not dec:
        dec = [x["decode_tok_s"] for x in reqs if x.get("kind") == "needles" and x.get("decode_tok_s")]
    if dec:
        rung["decode_tok_s"] = round(sum(dec) / len(dec), 1)
    ok = [int(x.get("prompt_tokens") or x.get("prompt_tokens_built") or 0) for x in reqs
          if x.get("kind") == "needles" and x.get("all_correct")]
    if ok:
        rung["max_verified_prompt_tokens"] = max(ok)


# ---------------------------------------------------------------------------
# The 3090 Ti host: the current teacher, measured through the probe guard
# ---------------------------------------------------------------------------


def nvidia_smi_here() -> Dict[str, Optional[int]]:
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,memory.reserved",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        parts = [to_int(x) for x in (p.stdout.splitlines() or [""])[0].split(",")]
    except (OSError, subprocess.TimeoutExpired):
        parts = []
    keys = ("used", "free", "total", "reserved")
    return {k: (parts[i] if i < len(parts) else None) for i, k in enumerate(keys)}


def gpu_apps_here() -> Dict[str, int]:
    """MiB per GPU process name (basename), for the computed limit (Ollama's share)."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    out: Dict[str, int] = {}
    for ln in p.stdout.splitlines():
        if "," not in ln:
            continue
        name, mib = ln.rsplit(",", 1)
        base = Path(name.strip()).name
        out[base] = out.get(base, 0) + (to_int(mib) or 0)
    return out


class VramWatch(threading.Thread):
    """Samples free VRAM here every few seconds; ``window(t0, t1)`` gives the minimum free in it."""

    def __init__(self, period: float = 3.0, sample: Callable[[], Dict[str, Optional[int]]] = nvidia_smi_here) -> None:
        super().__init__(daemon=True)
        self.period, self.sample = period, sample
        self.samples: List[Tuple[float, Optional[int], Optional[int]]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            s = self.sample()
            self.samples.append((time.time(), s.get("used"), s.get("free")))
            self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()

    def window(self, t0: float, t1: float) -> Dict[str, Optional[int]]:
        sel = [s for s in self.samples if t0 - self.period <= s[0] <= t1 + self.period]
        free = [s[2] for s in sel if s[2] is not None]
        used = [s[1] for s in sel if s[1] is not None]
        return {"free_min": min(free) if free else None, "used_max": max(used) if used else None,
                "samples": len(sel)}


def mini_window_conf(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#", 1)[0].strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def eval_running() -> bool:
    try:
        p = subprocess.run(["pgrep", "-f", "scripts/run_nightly_evals.py"], capture_output=True, text=True, timeout=10)
        return p.returncode == 0 and bool(p.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def computed_mini(free_min: Optional[int], current_ctx: int, n_ctx_train: Optional[int],
                  ollama_mib: Optional[int], kv_bytes: float = KV_Q8_BYTES_PER_TOKEN) -> Dict[str, Any]:
    """The 3090 Ti's headroom-limited maximum, from the minimum free VRAM seen while measuring."""
    if free_min is None:
        return {"free_mib_min": None, "kv_bytes_per_token": kv_bytes, "max_ctx_current_residents": None,
                "max_ctx_if_ollama_unloaded": None, "restart_advised": False,
                "restart_why": "free VRAM could not be read"}
    cur = computed_max_ctx(free_min, current_ctx, kv_bytes=kv_bytes, n_ctx_train=n_ctx_train)
    no_ollama = (computed_max_ctx(free_min + ollama_mib, current_ctx, kv_bytes=kv_bytes, n_ctx_train=n_ctx_train)
                 if ollama_mib else None)
    gain = cur - current_ctx
    # "clearly allows": at least 16K more beside every resident with the floor kept; the voice stack and
    # an image job grow on demand, so less than that is not worth a restart of the shared server
    advised = gain >= 16384
    why = (f"{k_label(gain) if gain > 0 else 'no'} more tokens fit beside the residents with the "
           f"{HEADROOM_MIN_MIB} MiB floor" + ("" if advised else "; under the 16K a restart would need to be "
                                                              "worth it, and the voice stack and image jobs "
                                                              "allocate on demand"))
    return {"free_mib_min": free_min, "headroom_floor_mib": HEADROOM_MIN_MIB, "growth_margin_mib": CTX_GROWTH_MARGIN_MIB,
            "kv_bytes_per_token": kv_bytes, "max_ctx_current_residents": cur,
            "max_ctx_if_ollama_unloaded": no_ollama, "ollama_mib": ollama_mib,
            "restart_advised": advised, "restart_why": why}


def cmd_mini(a: argparse.Namespace) -> int:
    capdir = Path(a.capdir).expanduser()
    capdir.mkdir(parents=True, exist_ok=True)
    os.chmod(capdir, 0o700)
    log_path = capdir / ("mini.log" if a.execute else "mini-dryrun.log")

    def log(msg: str) -> None:
        line = f"{datetime.now(ET):%F %T %Z} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    conf = mini_window_conf(Path(a.window_conf).expanduser())
    until = parse_iso(conf["CONTINUOUS_UNTIL"]) if conf.get("CONTINUOUS_UNTIL") else None
    log(f"mini capacity {'EXECUTE' if a.execute else 'DRY RUN'}: server {a.server_url}, guard {a.guard_url}")
    problems = []
    if until is not None and time.time() >= until - 1800:
        problems.append(f"the window ends {et(until)}: less than 30 min left")
    code, models = http_json(a.server_url.rstrip("/") + "/v1/models", timeout=20)
    data = ((models or {}).get("data") or [{}]) if isinstance(models, dict) else [{}]
    alias = (data[0] or {}).get("id")
    meta = (data[0] or {}).get("meta") or {}
    if code != 200 or not alias:
        problems.append(f"the teacher at {a.server_url} does not answer /v1/models (HTTP {code})")
    if conf.get("TEACHER_ALIAS") and alias and alias != conf["TEACHER_ALIAS"]:
        problems.append(f"the local model port serves {alias}, not the window's teacher {conf['TEACHER_ALIAS']}")
    code_g, gm = http_json(a.guard_url.rstrip("/") + "/v1/models", timeout=20)
    if code_g != 200:
        problems.append(f"the probe guard at {a.guard_url} does not answer (HTTP {code_g})")
    abort = Path(a.guard_abort).expanduser()
    if abort.exists():
        problems.append("the probe guard's abort switch has tripped")
    code_p, props = http_json(a.server_url.rstrip("/") + "/props", timeout=20)
    n_ctx = to_int(((props or {}).get("default_generation_settings") or {}).get("n_ctx")) if isinstance(props, dict) else None
    slots = to_int((props or {}).get("total_slots")) if isinstance(props, dict) else None
    n_ctx_train = to_int(meta.get("n_ctx_train"))
    if not n_ctx:
        problems.append("the teacher's /props gives no n_ctx")
    if problems:
        for p in problems:
            log(f"REFUSED: {p}")
        return 1
    assert n_ctx is not None
    ladder = [r for r in MINI_LADDER if r < n_ctx] + [n_ctx]
    log(f"teacher {alias}: n_ctx {n_ctx} ({slots} slots, unified KV), trained context {n_ctx_train}; "
        f"ladder {ladder}")
    pdir = capdir / "prompts"
    built: Dict[str, ProbeBuilder] = {}

    def builder() -> ProbeBuilder:
        if "b" not in built:
            built["b"] = ProbeBuilder.from_host(Path(a.repo).expanduser(), Path(a.wiki).expanduser(),
                                                Path(a.tokenizer_dir).expanduser(), Path(a.template).expanduser())
        return built["b"]

    probes = {(n, v): ensure_probe(builder, pdir, "3090ti", n, v, log) for n in ladder for v in ("A", "B")}
    if not a.execute:
        log("[dry run] probes built; a real run sends them through the guard, one at a time, outside the evals")
        return 0
    watch = VramWatch()
    watch.start()
    apps = gpu_apps_here()
    out_path = capdir / "mini-ctx.json"
    doc: Dict[str, Any] = {"schema": MINI_SCHEMA, "started": now_iso(), "teacher_alias": alias,
                           "current_config": {"ctx": n_ctx, "slots": slots, "kv_unified": True, "quant": "IQ3_XXS",
                                              "flash_attn": "on", "cache_type": "q8_0"},
                           "n_ctx_train": n_ctx_train, "route": "probe guard (one at a time)",
                           "rungs": [], "gpu_apps_mib": {k: v for k, v in apps.items()}}
    rungs: List[Dict[str, Any]] = doc["rungs"]
    try:
        for n in ladder:
            need = 2 * (n / 900.0) + 600  # two prefills at ~1k tok/s + guard waits
            while True:
                ok, why = eval_clear(time.time(), need_s=need)
                if ok and not eval_running():
                    break
                if until is not None and time.time() >= until - 1800:
                    raise Stop("the window is closing")
                log(f"waiting: {why or 'a nightly eval is running'}")
                time.sleep(120)
            rung: Dict[str, Any] = {"ctx": n, "slots": slots, "load": "ok", "requests": []}
            t0 = time.time()
            for v in ("A", "B"):
                pr = probes[(n, v)]
                log(f"  {k_label(n)} {v}: {pr['prompt_tokens']:,} tokens through the guard")
                r = needle_request(a.guard_url, pr, None)
                rung["requests"].append(r)
                log(f"    -> {r.get('error') or ('all 3' if r.get('all_correct') else str(r.get('correct')))}; "
                    f"prefill {r.get('prefill_tok_s')} tok/s, {r.get('wall_s')} s")
                if r.get("error") and "503" in str(r.get("error")):
                    raise Stop("the probe guard refused (abort switch)")
            if all(x.get("all_correct") for x in rung["requests"]):
                d = decode_request(a.guard_url, probes[(n, "B")], None)
                rung["requests"].append(d)
                log(f"    decode {d.get('decode_tok_s')} tok/s ({d.get('error') or 'ok'})")
            t1 = time.time()
            w = watch.window(t0, t1)
            rung["headroom_min_mib"] = w["free_min"]
            rung_speeds(rung)
            rung["stable"], rung["reason"] = rung_verdict(rung)
            rung["final"] = True
            rungs.append(rung)
            write_json(out_path, doc)
            log(f"  {k_label(n)}: {rung['reason']}")
            if not rung["stable"]:
                break
    except Stop as exc:
        doc["stopped"] = str(exc)
        log(f"stopped: {exc}")
    finally:
        watch.stop()
    summ = ladder_summary(ladder, rungs, top_label=f"the configured context ({n_ctx:,}) is the top")
    free_all = [s[2] for s in watch.samples if s[2] is not None]
    comp = computed_mini(min(free_all) if free_all else None, n_ctx, n_ctx_train, apps.get("ollama"))
    doc["measured"] = {"max_readable_ctx": summ["max_stable_ctx"], "bounded_by": summ["bounded_by"],
                       "first_unstable": summ["first_unstable"], "rungs": rungs,
                       "note": "the configured context is the ceiling: a single request can use the whole "
                               "unified KV when the other slot is idle (idle slots are purged as needed)"}
    doc["computed"] = comp
    doc["finished"] = now_iso()
    write_json(out_path, doc)
    log(f"done: reads up to {summ['max_stable_ctx']} ({summ['bounded_by']}); computed max "
        f"{comp.get('max_ctx_current_residents')} with the residents, {comp.get('max_ctx_if_ollama_unloaded')} "
        f"without Ollama; restart advised: {comp.get('restart_advised')}")
    return 0


def mini_host_doc(mini: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The 3090ti entry of teacher-ctx.json from mini-ctx.json."""
    if not isinstance(mini, dict) or "measured" not in mini:
        return None
    m = dict(mini["measured"])
    return {"current_config": mini.get("current_config"), "n_ctx_train": mini.get("n_ctx_train"),
            "route": mini.get("route"), "measured_at": mini.get("finished"), "measured": m,
            "computed": mini.get("computed"),
            "recommended_profile": {"ctx": (mini.get("current_config") or {}).get("ctx"), "restart": False,
                                    "long_context_row_cap_tokens": ((m.get("max_readable_ctx") or 0) - REPLY_RESERVE)
                                    if m.get("max_readable_ctx") else None}}


# ---------------------------------------------------------------------------
# The 4090 block
# ---------------------------------------------------------------------------


class Cfg:
    def __init__(self, a: argparse.Namespace) -> None:
        self.execute = bool(getattr(a, "execute", False))
        self.tower = a.tower or ""
        self.tower_url = (a.tower_url or "").rstrip("/")
        self.fake = Path(a.fake_root).expanduser().resolve() if getattr(a, "fake_root", None) else None
        here_home = (self.fake / "here") if self.fake else Path.home()
        self.here_home = here_home
        self.capdir = Path(a.capdir).expanduser() if a.capdir else here_home / "distill" / "capacity"
        self.key_file = Path(a.key_file).expanduser() if a.key_file else here_home / "distill/secrets/teacher-4090.key"
        self.pipeline_done = (Path(a.pipeline_done).expanduser() if a.pipeline_done else
                              here_home / "distill/phase7/9b-r32-v4b-max-derived/PIPELINE-DONE.json")
        self.pipeline_unit = a.pipeline_unit or ""
        self.skip_at = parse_iso(a.skip_at)
        self.restore_after = parse_iso(a.restore_after)
        self.window_end = parse_iso(a.window_end)
        self.budget_s = float(a.budget_min) * 60
        self.poll = float(a.poll)
        self.generation_follows = not a.no_generation_follows
        self.quants = [x for x in (a.quants or "q4,iq3").split(",") if x]
        self.ladder = [int(x) for x in a.ladder.split(",")] if a.ladder else list(LADDER_4090)
        self.qlora_lengths = [int(x) for x in a.qlora_lengths.split(",")] if a.qlora_lengths else list(QLORA_LENGTHS)
        self.skip_qlora = bool(a.skip_qlora)
        self.code_rel = a.code_rel  # the staged code on the 4090, relative to its home
        self.prefix = "capfake-" if self.fake else "distill-cap-"
        self.restore_prefix = "capfake-restore-" if self.fake else "flywheel-cap-restore-"
        self.unit_glob = "capfake-*" if self.fake else "distill-*"
        self.biz_unit = "capfake-biz.service" if self.fake else "llama-server.service"
        self.port = int(a.port)
        self.repo = Path(a.repo).expanduser()
        self.wiki = Path(a.wiki).expanduser()
        self.tokenizer_dir = Path(a.tokenizer_dir).expanduser()
        self.template = Path(a.template).expanduser()
        self.iq3_src = Path(a.iq3_src).expanduser() if a.iq3_src else here_home / IQ3_SRC
        self.iq3_manifest = Path(a.iq3_manifest).expanduser() if a.iq3_manifest else here_home / IQ3_SRC_MANIFEST
        self.quant_over: Dict[str, Dict[str, Any]] = json.loads(a.quant_override) if a.quant_override else {}
        self.qlora_runtime_s = int(a.qlora_runtime_s)

    def quant(self, key: str) -> Dict[str, Any]:
        d = dict(QUANTS[key])
        d.update(self.quant_over.get(key) or {})
        return d


class Tower:
    """Commands on the 4090 host by ssh (or, with --fake-root, a local bash whose home is the fake
    tree). Dry runs print what would change."""

    def __init__(self, cfg: Cfg, log: Callable[[str], None]) -> None:
        self.cfg, self.log = cfg, log
        self._home: Optional[str] = None

    @staticmethod
    def _run(argv: Sequence[str], timeout: float, env: Optional[Dict[str, str]] = None,
             stdin: Optional[str] = None, cwd: Optional[str] = None) -> Tuple[int, str]:
        try:
            p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, env=env, input=stdin,
                               cwd=cwd)
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout:.0f}s"
        except OSError as exc:
            return 127, str(exc)
        return p.returncode, (p.stdout + p.stderr).strip()

    def ssh_argv(self) -> List[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", self.cfg.tower]

    def run(self, script: str, *, mut: bool = False, timeout: float = 120, stdin: Optional[str] = None) -> Tuple[int, str]:
        if mut and not self.cfg.execute:
            self.log("  [plan] 4090: " + script[:600])
            return 0, ""
        if self.cfg.fake:
            # the fake home is also the working directory: a relative path must never reach this host's
            # real files (it did once: a rehearsal's cleanup removed a real model file here)
            home = str(self.cfg.fake / "tower")
            env = dict(os.environ, HOME=home, PATH=f"{self.cfg.fake / 'tower' / 'bin'}:{os.environ.get('PATH', '')}")
            return self._run(["bash", "-c", script], timeout, env=env, stdin=stdin, cwd=home)
        rc, out = 255, ""
        for attempt in range(3):
            rc, out = self._run(self.ssh_argv() + [script], timeout, stdin=stdin)
            if rc != 255:
                return rc, out
            self.log(f"  ssh to the 4090 failed ({out[-160:]}); retry {attempt + 1}/3 in 15 s")
            time.sleep(15)
        return rc, out

    @property
    def home(self) -> str:
        if self._home is None:
            if self.cfg.fake:
                self._home = str(self.cfg.fake / "tower")
            else:
                rc, out = self.run("echo $HOME")
                line = (out.splitlines() or [""])[-1].strip()
                if rc != 0 or not line.startswith("/"):
                    raise RuntimeError(f"cannot read the 4090 host's home: rc {rc}")
                self._home = line
        return self._home

    def p(self, rel: str) -> str:
        return f"{self.home}/{rel}"

    def rsync_to(self, src: Path, rel_dst: str, timeout: float = 3600) -> Tuple[int, str]:
        if not self.cfg.execute:
            self.log(f"  [plan] rsync {src} -> 4090:{rel_dst}")
            return 0, ""
        if self.cfg.fake:
            dst = self.cfg.fake / "tower" / rel_dst
            argv = ["rsync", "-a", "--partial", str(src), str(dst)]
        else:
            argv = ["rsync", "-a", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "-e", " ".join(self.ssh_argv()[:-1]),
                    str(src), f"{self.cfg.tower}:{rel_dst}"]
        return self._run(argv, timeout)


_HEALTH_TIMEOUT = 10.0


class Block:
    def __init__(self, cfg: Cfg, tower: Optional[Tower] = None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 builder_fn: Optional[Callable[[], ProbeBuilder]] = None,
                 http: Callable[..., Tuple[int, Any]] = http_json,
                 needle_fn: Callable[..., Dict[str, Any]] = needle_request,
                 decode_fn: Callable[..., Dict[str, Any]] = decode_request) -> None:
        self.cfg = cfg
        self.clock, self._sleep = clock, sleep
        cfg.capdir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(cfg.capdir, 0o700)
        except OSError:
            pass
        self.log_path = cfg.capdir / ("block.log" if cfg.execute else "block-dryrun.log")
        self.tower = tower or Tower(cfg, self.log)
        self.http, self.needle_fn, self.decode_fn = http, needle_fn, decode_fn
        self.state_path = cfg.capdir / ("block-state.json" if cfg.execute else "block-state-dryrun.json")
        st = read_json(self.state_path) if cfg.execute else None
        self.state: Dict[str, Any] = st if isinstance(st, dict) and st.get("schema") == STATE_SCHEMA else {
            "schema": STATE_SCHEMA, "created": now_iso(), "starts": 0, "steps": {}, "units": [],
            "telegrams": {}, "quants": {}, "qlora": {}, "notes": []}
        self.term = False
        self.biz_name = "business"
        self._builder: Optional[ProbeBuilder] = None
        self._builder_fn = builder_fn
        self._copy_thread: Optional[threading.Thread] = None
        self._key: Optional[str] = None
        self._probe_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._lock = threading.RLock()  # the IQ3 copy thread updates the state too

    # -- bookkeeping ----------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{datetime.now(ET):%F %T %Z} {msg}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def save(self) -> None:
        with self._lock:
            write_json(self.state_path, self.state)

    def mark(self, step: str, status: str, detail: str = "") -> None:
        self.state["steps"][step] = {"status": status, "detail": detail[:400], "at": now_iso()}
        self.save()
        self.log(f"[{step}] {status}" + (f": {detail}" if detail else ""))

    def note(self, text: str) -> None:
        self.state.setdefault("notes", []).append(text[:300])
        self.log(f"note: {text}")
        self.save()

    def sleep(self, s: float) -> None:
        end = self.clock() + s
        while self.clock() < end and not self.term:
            self._sleep(min(5.0, max(0.0, end - self.clock())))

    def notify(self, key: str, text: str) -> bool:
        """<= 5 lines through the 4090's notify.sh (stdin), once per key."""
        if key in self.state["telegrams"]:
            return True
        msg = "\n".join(text.splitlines()[:NOTIFY_LINES])
        if not self.cfg.execute:
            self.log("  [plan] Telegram:\n" + "\n".join("    | " + ln for ln in msg.splitlines()))
            return True
        rc, out = self.tower.run("~/distill/notify.sh 5", stdin=msg + "\n", timeout=90)
        ok = rc == 0
        self.log(f"Telegram {'sent' if ok else 'FAILED'}: {msg.splitlines()[0] if msg else ''}")
        if ok:
            self.state["telegrams"][key] = now_iso()
            self.save()
        return ok

    @property
    def key(self) -> Optional[str]:
        if self._key is None and self.cfg.key_file.is_file():
            self._key = read_key(self.cfg.key_file)
        return self._key

    # -- probes of the 4090 -----------------------------------------------------------------------
    def probe(self) -> Dict[str, List[str]]:
        flags = " ".join(q(p) for p in ("distill/window/NO-SWAPS-TONIGHT", "distill/NO-SWAPS",
                                         ".local/share/distill-setups/NO-SWAPS"))
        script = (f"cd ~ && echo biz=$(systemctl --user is-active {q(self.cfg.biz_unit)}); "
                  f"for f in {flags}; do [ -e \"$f\" ] && echo flag=$f; done; "
                  "echo free_gb=$(df -P -k ~ | awk 'NR==2{printf \"%.1f\", $4/1048576}'); "
                  "echo vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null"
                  " | head -1 | tr -dc 0-9); "
                  "echo vfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null"
                  " | head -1 | tr -dc 0-9); "
                  "systemctl --user list-units --plain --no-legend --no-pager "
                  f"--state=active,activating,deactivating,reloading {q(self.cfg.unit_glob)} 2>/dev/null"
                  " | awk '{print \"unit=\" $1}'")
        rc, out = self.tower.run(script, timeout=60)
        kv = kv_lines(out)
        if rc != 0 and "biz" not in kv:
            kv = {"error": [out[-200:]]}
        return kv

    def unit_state(self, unit: str) -> str:
        u = unit if unit.endswith(".service") else unit + ".service"
        rc, out = self.tower.run(f"systemctl --user is-active {q(u)}")
        st = (out.splitlines() or ["unknown"])[-1].strip() or "unknown"
        if rc == 255:
            return "unknown"
        return st

    def own_units(self) -> List[str]:
        return [u["name"] for u in self.state.get("units", []) if u.get("name")]

    def foreign_units(self, kv: Dict[str, List[str]]) -> List[str]:
        mine = set(self.own_units()) | {f"{u}.service" for u in self.own_units()}
        return [u for u in kv.get("unit", []) if not u.startswith("distill-deadman-") and u not in mine
                and not u.startswith(self.cfg.prefix) and u != self.cfg.biz_unit]

    def record_unit(self, name: str, kind: str) -> None:
        self.state.setdefault("units", []).append({"name": name, "kind": kind, "started": now_iso()})
        self.save()

    def stop_unit(self, name: str) -> None:
        st = self.unit_state(name)
        if st in ACTIVE:
            self.log(f"  stopping {name} ({st})")
            self.tower.run(f"systemctl --user stop {q(name if name.endswith('.service') else name + '.service')}",
                           mut=True, timeout=180)

    def stop_own_units(self) -> None:
        for name in reversed(self.own_units()):
            if self.unit_state(name) in ACTIVE:
                self.stop_unit(name)

    def vram(self) -> Tuple[Optional[int], Optional[int]]:
        kv = self.probe()
        return to_int(kv1(kv, "vram")), to_int(kv1(kv, "vfree"))

    def wait_vram_free(self, seconds: float) -> Tuple[bool, Optional[int]]:
        end = self.clock() + seconds
        used: Optional[int] = None
        while True:
            used, _ = self.vram()
            if used is not None and used <= IDLE_VRAM_MAX_MIB:
                return True, used
            if self.clock() >= end or self.term or not self.cfg.execute:
                return (not self.cfg.execute), used
            self.sleep(5)

    # -- guards ---------------------------------------------------------------------------------------
    def guard(self) -> None:
        if self.term:
            raise Stop("stopped (SIGTERM)")
        start = self.state.get("gpu_taken_at")
        if start and self.clock() >= start + self.cfg.budget_s:
            raise Stop(f"the block's time budget ({self.cfg.budget_s / 60:.0f} min) is used", failure=False)
        if self.clock() >= self.cfg.window_end - 1200:
            raise Stop(f"20 min before the 4090 window ends ({et(self.cfg.window_end)})")
        if not self.cfg.execute:
            return
        kv = self.probe()
        if "error" in kv:
            self.log(f"  probe failed ({kv1(kv, 'error')}); continuing")
            return
        if kv.get("flag"):
            raise Stop(f"NO-SWAPS flag present ({', '.join(kv['flag'])})")
        if self.state.get("gpu_taken") and kv1(kv, "biz") in ACTIVE:
            self.state["biz_back_elsewhere"] = True
            raise Stop(f"{self.cfg.biz_unit} is {kv1(kv, 'biz')} again (restored elsewhere)")
        free = to_float(kv1(kv, "free_gb"))
        if free is not None and free < DISK_STOP_GB:
            raise Stop(f"disk pressure on the 4090 ({free:.1f} GB free)")

    def time_left(self) -> float:
        start = self.state.get("gpu_taken_at") or self.clock()
        return min(start + self.cfg.budget_s, self.cfg.window_end - 1200) - self.clock()

    # -- steps --------------------------------------------------------------------------------------
    def builder(self) -> ProbeBuilder:
        if self._builder is None:
            if self._builder_fn is not None:
                self._builder = self._builder_fn()
            else:
                c = self.cfg
                self._builder = ProbeBuilder.from_host(c.repo, c.wiki, c.tokenizer_dir, c.template)
        return self._builder

    def get_probe(self, ctx: int, variant: str) -> Dict[str, Any]:
        k = (ctx, variant)
        if k not in self._probe_cache:
            self._probe_cache[k] = ensure_probe(self.builder, self.cfg.capdir / "prompts", "4090", ctx, variant, self.log)
        return self._probe_cache[k]

    def step_prompts(self) -> None:
        if self.state["steps"].get("prompts", {}).get("status") == "done" and self.cfg.execute:
            return
        t0 = self.clock()
        for n in self.cfg.ladder:
            for v in ("A", "B"):
                self.get_probe(n, v)
        self.mark("prompts", "done", f"{2 * len(self.cfg.ladder)} probes ready ({self.clock() - t0:.0f} s)")

    def step_wait(self) -> None:
        c = self.cfg
        if self.state["steps"].get("wait", {}).get("status") == "done":
            return
        dead_since: Optional[float] = None
        last_log = 0.0
        while True:
            if self.term:
                raise NotRun("stopped (SIGTERM) while waiting for the post-train pipeline")
            if c.pipeline_done.is_file():
                doc = read_json(c.pipeline_done) or {}
                self.state["pipeline"] = {"outcome": str(doc.get("outcome"))[:200], "finished": doc.get("finished"),
                                          "seen_at": now_iso()}
                self.mark("wait", "done", f"PIPELINE-DONE: {str(doc.get('outcome'))[:120]}")
                return
            now = self.clock()
            if now >= c.skip_at:
                raise NotRun(f"the post-train pipeline had not finished by {et(c.skip_at)}")
            if not c.execute:
                self.log(f"[wait] no PIPELINE-DONE yet ({c.pipeline_done.name}); a real run waits here until it "
                         f"appears or {et(c.skip_at)}; the dry run goes on as if it had")
                return
            if c.pipeline_unit:
                # a unit pattern (the pipeline may be re-armed under a new stamp): none of them running or
                # restarting for an hour, and still no marker, means it will not come
                alive = here_units_alive(c.pipeline_unit)
                if alive is None or alive:
                    dead_since = None
                elif dead_since is None:
                    dead_since = now
                elif now - dead_since > 3600:
                    raise NotRun(f"no {c.pipeline_unit} unit has run for an hour and there is no PIPELINE-DONE")
            if now - last_log >= 3600:
                self.log(f"waiting for {c.pipeline_done.name} (skip at {et(c.skip_at)})")
                last_log = now
            self.sleep(c.poll)

    def read_biz_name(self) -> None:
        rc, out = self.tower.run("sed -n 's/^BIZ_NAME=\\([^ ]*\\).*/\\1/p' ~/distill/window/window.conf 2>/dev/null")
        name = (out.strip().splitlines() or [""])[-1].strip()
        if rc == 0 and re.fullmatch(r"[A-Za-z0-9 ._-]{1,40}", name or ""):
            self.biz_name = name

    def step_take_gpu(self) -> None:
        if self.state.get("gpu_taken"):
            return
        c = self.cfg
        kv = self.probe()
        if "error" in kv and c.execute:
            raise NotRun(f"the 4090 does not answer ({kv1(kv, 'error')[:120]})")
        if kv.get("flag") and c.execute:
            raise NotRun(f"NO-SWAPS flag present on the 4090 ({', '.join(kv['flag'])[:150]})")
        free = to_float(kv1(kv, "free_gb"))
        if free is not None and free < DISK_STOP_GB and c.execute:
            raise NotRun(f"the 4090 has {free:.1f} GB free (< {DISK_STOP_GB:g})")
        others = self.foreign_units(kv)
        if others and c.execute:
            raise NotRun(f"other distill units run on the 4090: {', '.join(others)[:200]}")
        if self.clock() >= c.restore_after and c.execute:
            raise NotRun(f"it is past {et(c.restore_after)}: no new business downtime")
        biz = kv1(kv, "biz")
        eta = min(self.clock() + c.budget_s, c.window_end - 1200)
        down = "\n".join([
            f"{self.biz_name} going down: teacher/trainer capacity tests.",
            f"27B teacher context ladder (Q4_K_XL, IQ3), then 9B QLoRA 16K/32K/64K; ETA ~{et(eta)}.",
            f"After it: cross-host generation (business stays down), or {self.biz_name} is restored "
            f"if anything fails or past {et(c.restore_after)}."])
        lock = self.tower.p("distill/window/state/window.lock")
        stop_script = (
            f"exec 9>>{q(lock)}; flock -w 1800 9 || {{ echo lock=timeout; exit 9; }}; "
            f"st=$(systemctl --user is-active {q(c.biz_unit)}); echo before=$st; "
            f"case $st in active|activating|reloading) systemctl --user stop {q(c.biz_unit)}"
            " && echo stopped=yes ;; esac; "
            "for i in $(seq 1 18); do st=$(systemctl --user is-active "
            f"{q(c.biz_unit)}); used=$(nvidia-smi --query-gpu=memory.used "
            "--format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); "
            f"if [ \"$st\" != active ] && [ \"${{used:-99999}}\" -le {IDLE_VRAM_MAX_MIB} ]; then "
            "echo freed=$used; exit 0; fi; sleep 5; done; echo freed=no used=$used; exit 1")
        if not c.execute:
            self.log(f"[take-gpu] now: {c.biz_unit} {biz or '?'}, VRAM {kv1(kv, 'vram') or '?'} MiB, flags "
                     f"{kv.get('flag') or 'none'}, 4090 free {kv1(kv, 'free_gb')} GB, other units {others or 'none'}")
            self.notify("down", down)
            self.tower.run(stop_script, mut=True)
            self.mark("take-gpu", "planned")
            return
        if biz in ACTIVE:
            self.state["gpu_taken"] = True
            self.state["biz_down_by_us"] = True
            self.state["gpu_taken_at"] = self.clock()
            self.save()
            self.notify("down", down)
            rc, out = self.tower.run(stop_script, timeout=2000)
            if rc != 0:
                raise Stop(f"stopping {c.biz_unit} did not free the GPU (rc {rc}: {out[-160:]})")
            self.mark("take-gpu", "done", f"{c.biz_unit} stopped; VRAM {kv1(kv_lines(out), 'freed', '?')} MiB")
            return
        ok, used = self.wait_vram_free(600)
        if not ok:
            raise NotRun(f"the 4090 GPU is not free ({used} MiB) and {c.biz_unit} is {biz or '?'}")
        self.state["gpu_taken"] = True
        self.state["gpu_taken_at"] = self.clock()
        self.save()
        self.notify("down", down.replace("going down", "is down already"))
        self.mark("take-gpu", "done", f"{c.biz_unit} {biz or '?'} already; VRAM {used} MiB")

    # sampler: nvidia-smi + MemAvailable every second, read back per window
    def sampler_file(self) -> str:
        return self.tower.p("distill/capacity/sampler.csv")

    def step_sampler(self) -> None:
        name = f"{self.cfg.prefix}sampler"
        if self.unit_state(name) in ACTIVE:
            return
        script_path = self.tower.p(f"{self.cfg.code_rel}/scripts/distill/tower/cap_sampler.sh")
        f = self.sampler_file()
        script = (f"mkdir -p -m 700 {q(str(Path(f).parent))} && systemd-run --user --quiet --collect "
                  f"--unit={q(name)} --description={q('distill capacity: VRAM/RAM sampler')} "
                  "-p KillSignal=SIGTERM -p TimeoutStopSec=10 "
                  + (f"--setenv=PATH={q(str(self.cfg.fake / 'tower' / 'bin') + ':/usr/bin:/bin')} " if self.cfg.fake else "")
                  + f"-- /bin/bash {q(script_path)} {q(f)}")
        self.record_unit(name, "sampler")
        rc, out = self.tower.run(script, mut=True)
        if rc != 0:
            self.note(f"the sampler did not start (rc {rc}: {out[-120:]}): headroom falls back to point reads")

    def peaks(self, t0: float, t1: float) -> Dict[str, Optional[float]]:
        """From the sampler: max VRAM used, min VRAM free, min MemAvailable (kB) in [t0, t1]."""
        if not self.cfg.execute:
            return {"used_max": None, "free_min": None, "avail_min_kb": None, "samples": 0}
        f = self.sampler_file()
        rc, out = self.tower.run(
            f"awk -F, -v a={int(t0) + 1} -v b={int(t1)} '$1>=a && $1<=b && $2!=\"\" "
            "{n++; if ($2>u) u=$2; if (f==\"\" || $3<f) f=$3; if (m==\"\" || $4<m) m=$4} "
            f"END {{print \"n=\" n; print \"used=\" u; print \"free=\" f; print \"avail=\" m}}' {q(f)} 2>/dev/null")
        kv = kv_lines(out)
        n = to_int(kv1(kv, "n"), 0) or 0
        return {"used_max": to_float(kv1(kv, "used")) if n else None, "free_min": to_float(kv1(kv, "free")) if n else None,
                "avail_min_kb": to_float(kv1(kv, "avail")) if n else None, "samples": n}

    # IQ3: copied from here, sha256 checked on the 4090, in the background
    def step_copy_start(self) -> None:
        if "iq3" not in self.cfg.quants:
            return
        st = self.state.setdefault("iq3_copy", {})
        if st.get("status") in ("verified", "skipped"):
            return
        qd = self.cfg.quant("iq3")
        f = self.tower.p(qd["file"])
        rc, out = self.tower.run(f"test -s {q(f)} && stat -c %s {q(f)} || echo none", timeout=60) \
            if self.cfg.execute else (0, "none")
        size = to_int((out.splitlines() or [""])[-1])
        kv = self.probe()
        free = to_float(kv1(kv, "free_gb"))
        need_gb = qd["bytes"] / 1e9
        if size != qd["bytes"] and free is not None and free - need_gb < DISK_FLOOR_GB and self.cfg.execute:
            st.update(status="skipped", reason=f"the 4090 has {free:.1f} GB free; the copy needs {need_gb:.1f} + "
                                               f"the {DISK_FLOOR_GB:g} GB floor")
            self.save()
            self.note(f"IQ3 not copied: {st['reason']}")
            return
        self._copy_thread = threading.Thread(target=self._copy_iq3, name="iq3-copy", daemon=True)
        self._copy_thread.start()

    def _copy_iq3(self) -> None:
        with self._lock:
            st = self.state.setdefault("iq3_copy", {})
        qd = self.cfg.quant("iq3")
        f, man = self.tower.p(qd["file"]), self.tower.p(qd["manifest"])

        def put(**kw: Any) -> None:
            with self._lock:
                st.update(**kw)

        try:
            t0 = self.clock()
            if not self.cfg.iq3_src.is_file():
                put(status="failed", reason="the IQ3 source file is missing on this host")
                return
            rc, out = self.tower.run(f"mkdir -p -m 700 {q(str(Path(f).parent))} && "
                                     f"stat -c %s {q(f)} 2>/dev/null || echo 0", mut=True, timeout=60)
            have = to_int((out.splitlines() or ["0"])[-1], 0) if self.cfg.execute else 0
            transferred = have != qd["bytes"]
            if transferred:
                rc, out = self.tower.rsync_to(self.cfg.iq3_src, qd["file"], timeout=5400)
                if rc != 0:
                    put(status="failed", reason=f"rsync exit {rc}: {out[-160:]}")
                    return
            rc, out = self.tower.rsync_to(self.cfg.iq3_manifest, qd["manifest"], timeout=600)
            if rc != 0:
                put(status="failed", reason=f"manifest rsync exit {rc}")
                return
            if not self.cfg.execute:
                put(status="planned")
                return
            rc, out = self.tower.run(f"sha256sum {q(f)} | cut -d' ' -f1; "
                                     f"grep -o '\"sha256\": *\"[0-9a-f]\\{{64\\}}\"' {q(man)} | head -n 1 "
                                     "| grep -o '[0-9a-f]\\{64\\}'", timeout=900)
            lines = [x.strip() for x in out.splitlines() if re.fullmatch(r"[0-9a-f]{64}", x.strip())]
            if len(lines) >= 2 and lines[0] == qd["sha256"] and lines[1] == qd["sha256"]:
                put(status="verified", seconds=round(self.clock() - t0), sha256=qd["sha256"])
            else:
                if transferred and not self.cfg.fake and f.startswith(self.tower.home + "/distill/"):
                    put(status="failed", reason="sha256 mismatch after the copy (this run's copy is removed)")
                    self.tower.run(f"rm -f -- {q(f)}", mut=True)
                else:
                    put(status="failed", reason="sha256 mismatch (nothing removed)")
        except Exception as exc:  # noqa: BLE001 - reported, the IQ3 ladder is skipped
            put(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        finally:
            if st.get("status") is None:
                put(status="failed", reason="the copy ended without a result")

    def wait_copy(self) -> Dict[str, Any]:
        t = self._copy_thread
        while t is not None and t.is_alive():
            self.guard()
            t.join(timeout=10)
        self.save()
        return self.state.get("iq3_copy") or {}

    # -- teacher ladder -------------------------------------------------------------------------------
    def serve_env(self, key: str, ctx: int, slots: int) -> List[str]:
        qd = self.cfg.quant(key)
        env = {"TEACHER_SLOTS": slots, "TEACHER_CTX_PER_SLOT": ctx // slots, "TEACHER_PORT": self.cfg.port,
               "TEACHER_MODEL": self.tower.p(qd["file"]),
               "TEACHER_MODEL_BYTES": qd["bytes"], "TEACHER_SHA256": qd["sha256"],
               "TEACHER_MANIFEST": self.tower.p(qd["manifest"]),
               "TEACHER_ALIAS": qd["alias"], "TEACHER_WEIGHTS_MIB": qd["weights_mib"],
               # the projection refuses only what cannot load (weights + KV over the card); the load decides the rest
               "TEACHER_BUFFERS_MIB": 0, "TEACHER_RS_MIB_PER_SLOT": 0, "TEACHER_MARGIN_MIB": 0,
               "TEACHER_VERIFY_WAIT_S": 120}
        return [f"{k}={v}" for k, v in env.items()]

    def serve_script(self) -> str:
        return self.tower.p("distill/teacher-serve/serve_teacher.sh")

    def start_server(self, key: str, ctx: int, slots: int, rung: Dict[str, Any]) -> None:
        """Start + wait + verify; fills rung['load'] (ok | load-failed | refused | error)."""
        c = self.cfg
        n = len([u for u in self.own_units() if f"srv-{key}-c{ctx}" in u]) + 1
        name = f"{c.prefix}srv-{key}-c{ctx}" + (f"-s{slots}" if slots > 1 else "") + f"-{n}"
        logs = self.tower.p("distill/capacity/logs")
        logf = f"{logs}/{name}.log"
        envs = self.serve_env(key, ctx, slots)
        if c.execute:
            ok, used = self.wait_vram_free(300)
            if not ok:
                rung.update(load="error", load_detail=f"the GPU holds {used} MiB before the start")
                return
        setenv = " ".join(f"--setenv={q(e)}" for e in envs)
        script = (f"mkdir -p -m 700 {q(logs)} && systemd-run --user --quiet --collect --unit={q(name)} "
                  f"--description={q(f'distill capacity probe server {key} c{ctx} x{slots}')} "
                  "-p KillSignal=SIGTERM -p TimeoutStopSec=60 -p LimitCORE=0 "
                  f"-p StandardOutput=append:{q(logf)} -p StandardError=append:{q(logf)} {setenv} -- "
                  f"{q(self.serve_script())} exec")
        self.record_unit(name, "server")
        rung["unit"] = name
        rung["t_start"] = self.clock()
        rc, out = self.tower.run(script, mut=True)
        if not c.execute:
            rung.update(load="ok", load_detail="planned")
            return
        if rc != 0:
            rung.update(load="error", load_detail=f"systemd-run exited {rc}: {out[-160:]}")
            return
        t0 = self.clock()
        while True:
            self.guard()
            st = self.unit_state(name)
            if st not in ACTIVE and st != "unknown":
                rc, tail = self.tower.run(f"tail -n 40 {q(logf)}")
                rung.update(load=load_failure_kind(tail), load_detail=load_failure_reason(tail))
                return
            code, _ = self.http(f"{c.tower_url}/health", timeout=_HEALTH_TIMEOUT)
            if code == 200:
                break
            if self.clock() - t0 > LOAD_TIMEOUT_S:
                self.stop_unit(name)
                rung.update(load="error", load_detail=f"not healthy after {LOAD_TIMEOUT_S} s")
                return
            self.sleep(5)
        rung["load_s"] = round(self.clock() - t0)
        venv = " ".join(q(e) for e in envs)
        rc, out = self.tower.run(f"env {venv} {q(self.serve_script())} verify", timeout=300)
        if rc != 0:
            self.stop_unit(name)
            raise Stop(f"the probe server failed its verify ({out[-200:]})")
        code, props = self.http(f"{c.tower_url}/props", key=self.key, timeout=30)
        dgs = (props or {}).get("default_generation_settings") or {} if isinstance(props, dict) else {}
        n_ctx = to_int(dgs.get("n_ctx"))
        if n_ctx not in (ctx // slots, ctx):
            self.stop_unit(name)
            rung.update(load="error", load_detail=f"the server reports n_ctx {n_ctx}, asked {ctx // slots} per slot")
            return
        code, models = self.http(f"{c.tower_url}/v1/models", timeout=30)
        meta = (((models or {}).get("data") or [{}])[0] or {}).get("meta") or {} if isinstance(models, dict) else {}
        if meta.get("n_ctx_train"):
            self.state.setdefault("n_ctx_train", {})[key] = to_int(meta.get("n_ctx_train"))
        used, vfree = self.vram()
        rung.update(load="ok", load_detail="verified", vram_used_mib=used, vram_free_after_load_mib=vfree,
                    t_ready=self.clock())

    def server_alive(self, rung: Dict[str, Any]) -> bool:
        st = self.unit_state(rung["unit"])
        return st in ACTIVE or st == "unknown"

    def finish_rung(self, rung: Dict[str, Any]) -> None:
        # the headroom window: from the verified load to the last request, before the stop (a sample from
        # the previous server or from the release must not count)
        t1 = self.clock()
        pk = self.peaks(rung["t_ready"], t1) if rung.get("t_ready") else \
            {"used_max": None, "free_min": None, "avail_min_kb": None, "samples": 0}
        rung["sampler"] = pk
        if rung.get("unit"):
            self.stop_unit(rung["unit"])
            if self.cfg.execute:
                self.wait_vram_free(120)
        free_pts = [x for x in (pk.get("free_min"), rung.get("vram_free_after_load_mib")) if x is not None]
        if rung.get("load") == "ok":
            rung["headroom_min_mib"] = int(min(free_pts)) if free_pts else None
        if not self.cfg.execute:
            rung.setdefault("headroom_min_mib", None)

    def measure_rung(self, key: str, ctx: int) -> Dict[str, Any]:
        rung: Dict[str, Any] = {"ctx": ctx, "slots": 1, "requests": []}
        probes = [self.get_probe(ctx, v) for v in ("A", "B")]
        try:
            self.start_server(key, ctx, 1, rung)
            if rung.get("load") == "ok" and self.cfg.execute:
                for pr in probes:
                    self.guard()
                    r = self.needle_fn(self.cfg.tower_url, pr, self.key)
                    rung["requests"].append(r)
                    self.log(f"    {key} {k_label(ctx)} {pr['variant']}: "
                             f"{r.get('error') or ('3/3' if r.get('all_correct') else str(r.get('correct')))}, "
                             f"prefill {r.get('prefill_tok_s')} tok/s, {r.get('wall_s')} s")
                    if not self.server_alive(rung):
                        rung.update(crashed=True, crash_detail="the server exited during a request")
                        break
                if not rung.get("crashed") and all(x.get("all_correct") for x in rung["requests"]):
                    d = self.decode_fn(self.cfg.tower_url, probes[1], self.key)
                    rung["requests"].append(d)
                    if not self.server_alive(rung):
                        rung.update(crashed=True, crash_detail="the server exited during the decode benchmark")
        finally:
            self.finish_rung(rung)
        rung_speeds(rung)
        if self.cfg.execute:
            rung["stable"], rung["reason"] = rung_verdict(rung)
        else:
            rung["stable"], rung["reason"] = True, "planned"
        rung["final"] = True
        return rung

    def step_ladder(self, key: str) -> None:
        qd = self.cfg.quant(key)
        qs = self.state["quants"].setdefault(key, {"label": qd["label"], "status": "pending", "rungs": []})
        if qs.get("status") in ("measured", "skipped", "failed") and self.cfg.execute:
            return
        if key == "iq3":
            cp = self.wait_copy()
            if cp.get("status") not in ("verified", "planned") and self.cfg.execute:
                qs.update(status="skipped", skip_reason=f"the IQ3 copy is {cp.get('status')}: {cp.get('reason', '')}")
                self.save()
                self.note(f"IQ3 ladder skipped: {qs['skip_reason']}")
                return
        ladder = ladder_for((self.state.get("n_ctx_train") or {}).get(key) or 262144, self.cfg.ladder)
        qs["ladder"] = ladder
        self.log(f"[ladder {key}] {qd['label']}: {ladder}")
        while True:
            n = next_rung(ladder, qs["rungs"])
            if n is None:
                break
            est = self.estimate_rung_s(qs["rungs"], n)
            reserve = self.reserve_after(key)
            if self.cfg.execute and self.time_left() < est + reserve:
                qs["status"] = "partial"
                qs["stopped_at"] = {"ctx": n, "reason": f"time budget: ~{est / 60:.0f} min needed, "
                                                        f"{max(0, self.time_left() - reserve) / 60:.0f} left"}
                self.note(f"{key} ladder stopped before {k_label(n)}: {qs['stopped_at']['reason']}")
                break
            self.guard()
            rung = self.measure_rung(key, n)
            qs["rungs"].append(rung)
            self.save()
            self.log(f"  {key} {k_label(n)}: {rung['reason']} (VRAM {rung.get('vram_used_mib')} MiB, "
                     f"headroom {rung.get('headroom_min_mib')} MiB)")
            if not self.cfg.execute:
                break  # one planned rung shows the commands
            n_tr = (self.state.get("n_ctx_train") or {}).get(key)
            if n_tr and ladder != ladder_for(n_tr, self.cfg.ladder):
                ladder = ladder_for(n_tr, self.cfg.ladder)
                qs["ladder"] = ladder
        if qs["status"] == "pending":
            qs["status"] = "measured"
        qs["summary"] = ladder_summary(ladder, qs["rungs"])
        qs["kv_fit"] = kv_fit(qs["rungs"])
        self.save()

    def estimate_rung_s(self, rungs: Sequence[Dict[str, Any]], n: int) -> float:
        pre = [r.get("prefill_tok_s") for r in rungs if r.get("prefill_tok_s")]
        rate = (min(pre) * 0.8) if pre else 900.0
        load = max([r.get("load_s") or 0 for r in rungs] + [90])
        return 2 * n / rate + load + 90

    def reserve_after(self, key: str) -> float:
        """Time kept for what comes after this ladder."""
        r = 0.0
        if key == "q4" and "iq3" in self.cfg.quants:
            r += 15 * 60  # at least the IQ3 ladder's first rungs
        if not self.cfg.skip_qlora:
            r += 30 * 60
        return r

    def step_multi(self, key: str) -> None:
        qs = self.state["quants"].get(key) or {}
        if qs.get("multi_done") or qs.get("status") not in ("measured", "partial"):
            return
        best = (qs.get("summary") or {}).get("max_stable_ctx")
        out = []
        if best and self.cfg.execute and self.time_left() > 6 * 60 + self.reserve_after(key):
            rung: Dict[str, Any] = {"ctx": best, "slots": 2, "requests": []}
            try:
                self.start_server(key, best, 2, rung)
            finally:
                self.finish_rung(rung)
            m = {"slots": 2, "ctx_per_slot": best // 2, "source": "measured", "load": rung.get("load"),
                 "load_detail": rung.get("load_detail"), "vram_used_mib": rung.get("vram_used_mib"),
                 "headroom_min_mib": rung.get("headroom_min_mib")}
            m["fits"] = bool(m["load"] == "ok" and (m["headroom_min_mib"] or 0) >= HEADROOM_MIN_MIB)
            out.append(m)
        usable = self.state.get("usable_mib")
        for alt in multi_slot_alternatives(best, qs.get("kv_fit"), usable):
            if not any(o["slots"] == alt["slots"] for o in out):
                out.append(alt)
        qs["multi_slot"] = out
        qs["multi_done"] = True
        self.save()

    # -- QLoRA ---------------------------------------------------------------------------------------------
    def bench_argv(self, length: int, out: str) -> List[str]:
        c = self.cfg
        T = self.tower.p
        code = T(c.code_rel)
        return [T("distill/venv-train/bin/python"), f"{code}/scripts/distill/tower/seqlen_bench.py", "run",
                "--trainer", f"{code}/scripts/distill/train_qlora.py",
                "--data", T("distill/phase3/build-v4b-max-derived/train.jsonl"),
                "--tokenizer-dir", T("distill/models/qwen3.5-9b-tokenizer"),
                "--base-model", T("distill/models/qwen3.5-9b-bnb4"),
                "--base-revision", "3c78dfb3e32c496a2438fb1790859eff805284cf",
                "--seq-len", str(length), "--steps", str(QLORA_STEPS), "--warmup-steps", str(QLORA_WARMUP),
                "--out", out, "--i-have-operator-approval-for-max-derived-data"] + \
            (["--fake-backend"] if c.fake else [])

    def run_bench(self, length: int) -> Dict[str, Any]:
        c = self.cfg
        T = self.tower.p
        scratch = T("distill/capacity/qlora-bench-max-derived")
        out = f"{scratch}/qlora-{length}.json"
        logf = f"{scratch}/qlora-{length}.log"
        n = len([u for u in self.own_units() if f"qlora-{length}" in u]) + 1
        name = f"{c.prefix}qlora-{length}-{n}"
        argv = self.bench_argv(length, out)
        envs = ["PYTHONUNBUFFERED=1", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1",
                "DO_NOT_TRACK=1", "TZ=America/New_York"]
        props = ["-p", "KillMode=mixed", "-p", "TimeoutStopSec=60", "-p", f"RuntimeMaxSec={c.qlora_runtime_s}",
                 "-p", "LimitCORE=0", "-p", f"WorkingDirectory={T('distill')}",
                 "-p", f"StandardOutput=append:{logf}", "-p", f"StandardError=append:{logf}"]
        if not c.fake:
            props += ["-p", f"MemoryMax={QLORA_MEMORY_MAX}"]
        script = (f"mkdir -p -m 700 {q(scratch)} && rm -f {q(out)} && systemd-run --user --quiet --collect "
                  f"--unit={q(name)} --description={q(f'distill capacity: 9B QLoRA at {length}')} "
                  + " ".join(q(x) for x in props) + " " + " ".join(f"--setenv={q(e)}" for e in envs)
                  + " -- " + shlex.join(argv))
        if c.execute:
            ok, used = self.wait_vram_free(300)
            if not ok:
                return {"status": "failed", "detail": f"the GPU holds {used} MiB before the start"}
        self.record_unit(name, "qlora")
        t0 = self.clock()
        rc, o = self.tower.run(script, mut=True)
        if not c.execute:
            return {"status": "skipped", "detail": "planned"}
        if rc != 0:
            return {"status": "failed", "detail": f"systemd-run exited {rc}: {o[-160:]}"}
        while True:
            self.sleep(15)
            st = self.unit_state(name)
            if st not in ACTIVE and st != "unknown":
                break
            try:
                self.guard()
            except Stop:
                self.stop_unit(name)
                raise
        t1 = self.clock()
        rc, raw = self.tower.run(f"cat {q(out)} 2>/dev/null")
        doc = None
        try:
            doc = json.loads(raw) if rc == 0 and raw.strip().startswith("{") else None
        except ValueError:
            doc = None
        rc, res = self.tower.run(f"systemctl --user show {q(name + '.service')} -p Result -p ExecMainStatus 2>/dev/null")
        kv = kv_lines(res)
        result = kv1(kv, "Result")
        pk = self.peaks(t0, t1)
        if not isinstance(doc, dict):
            doc = {"status": "killed" if result in ("oom-kill", "signal", "timeout") else "failed",
                   "detail": f"no result file; unit result {result or '?'}, exit {kv1(kv, 'ExecMainStatus') or '?'}"}
        elif doc.get("status") == "running":
            doc["status"] = "killed" if result in ("oom-kill", "signal", "timeout") else "failed"
            doc["detail"] = f"ended mid-run; unit result {result or '?'}"
        if pk.get("used_max") is not None:
            doc["peak_vram_smi_gib"] = round(pk["used_max"] / 1024, 2)
        if pk.get("avail_min_kb") is not None and self.state.get("mem_total_kb"):
            doc["host_used_peak_gib"] = round((self.state["mem_total_kb"] - pk["avail_min_kb"]) / 1048576, 2)
        doc["unit_result"] = result
        doc["wall_s"] = round(t1 - t0)
        return doc

    def step_qlora(self) -> None:
        if self.cfg.skip_qlora:
            return
        res = self.state.setdefault("qlora", {})
        if not self.cfg.execute:
            self.run_bench(self.cfg.qlora_lengths[0])
            return
        if "mem_total_kb" not in self.state:
            rc, out = self.tower.run("awk '/MemTotal/{print $2}' /proc/meminfo")
            self.state["mem_total_kb"] = to_int((out.splitlines() or [""])[-1])
        while True:
            n = qlora_next(res, self.cfg.qlora_lengths)
            if n is None:
                break
            est = 300 + n / 1500.0 * (QLORA_STEPS + QLORA_WARMUP) + 120
            if self.time_left() < est:
                res[str(n)] = {"status": "skipped", "detail": f"time budget (~{est / 60:.0f} min needed)"}
                self.note(f"QLoRA {n} skipped: time budget")
                break
            self.guard()
            self.log(f"[qlora] {n}")
            r = self.run_bench(n)
            res[str(n)] = r
            self.save()
            self.log(f"  qlora {n}: {r.get('status')} {r.get('tok_s') or ''} tok/s "
                     f"{r.get('detail') or ''}".rstrip())

    # -- release, outputs, end ------------------------------------------------------------------------
    def release(self, why: str) -> Dict[str, Any]:
        """train_end.sh --msg-file as a flywheel-* unit: apply business + verify; its own '<biz> back' Telegram."""
        c = self.cfg
        if self.state.get("released"):
            return self.state.get("restore") or {}
        self.log(f"[release] {why}")
        self.stop_own_units()
        base = self.tower.p("distill/capacity")
        msg, rcf = f"{base}/restore.msg", f"{base}/restore.rc"
        line = f"Capacity tests ended: {why[:200]}."
        self.tower.run(f"mkdir -p -m 700 {q(base)} && rm -f {q(rcf)} && printf '%s\\n' {q(line)} > {q(msg)}", mut=True)
        unit = f"{c.restore_prefix}{datetime.now(ET).strftime('%Y%m%dT%H%M%S')}"
        te = self.tower.p("distill/window/train_end.sh")
        inner = f"/bin/bash {q(te)} --msg-file {q(msg)}; echo $? > {q(rcf)}"
        script = (f"systemd-run --user --quiet --collect --unit={q(unit)} -p KillMode=mixed -p TimeoutStopSec=1800 "
                  f"--description={q('flywheel: restore business after the capacity tests')} "
                  f"--expand-environment=no -- /bin/bash -c {q(inner)}")
        self.tower.run(script, mut=True)
        if not c.execute:
            return {"planned": True}
        t0 = self.clock()
        self._sleep(5)
        while self.unit_state(unit) in ACTIVE + ("unknown",) and self.clock() - t0 < 2700:
            self._sleep(10)
        rc, out = self.tower.run(f"cat {q(rcf)} 2>/dev/null")
        code = (out.strip().splitlines() or ["none"])[-1]
        rc, last = self.tower.run("cat ~/distill/window/state/verify.last 2>/dev/null")
        verify = (last.strip().splitlines() or [""])[-1]
        res = {"unit": unit, "train_end_exit": code, "verify": verify[:200], "ok": code == "0"}
        self.state["restore"] = res
        self.state["released"] = True
        self.save()
        return res

    def teacher_doc(self) -> Dict[str, Any]:
        quants: Dict[str, Any] = {}
        for key in self.cfg.quants:
            qs = self.state["quants"].get(key)
            qd = self.cfg.quant(key)
            if not qs:
                quants[key] = {"label": qd["label"], "status": "skipped", "skip_reason": "not reached",
                               "summary": {"max_stable_ctx": None}, "rungs": []}
                continue
            rungs = [{k: v for k, v in r.items() if k not in ("t_start", "t_ready")} for r in qs.get("rungs", [])]
            s = qs.get("summary") or ladder_summary(qs.get("ladder") or self.cfg.ladder, rungs)
            status = qs.get("status", "partial")
            if status not in ("measured", "partial", "skipped", "failed"):
                status = "partial" if any(r.get("final") for r in rungs) else "failed"
            entry = {"label": qd["label"], "sha256": qd["sha256"], "status": status,
                     "summary": s, "ladder": qs.get("ladder"), "rungs": rungs, "kv_fit": qs.get("kv_fit"),
                     "multi_slot": qs.get("multi_slot") or [], "stopped_at": qs.get("stopped_at"),
                     "skip_reason": qs.get("skip_reason"), "n_ctx_train": (self.state.get("n_ctx_train") or {}).get(key)}
            if s.get("max_stable_ctx"):
                m = s["max_stable_ctx"]
                entry["recommended_profile"] = {
                    "parallel": 1, "ctx": m, "flash_attn": "on", "cache_type_k": "q8_0", "cache_type_v": "q8_0",
                    "env": {"TEACHER_SLOTS": 1, "TEACHER_CTX_PER_SLOT": m},
                    "long_context_row_cap_tokens": m - REPLY_RESERVE}
            quants[key] = entry
        mini = mini_host_doc(read_json(self.cfg.capdir / "mini-ctx.json"))
        hosts: Dict[str, Any] = {"4090": {"gpu": "RTX 4090 24 GB, dedicated (business server stopped)",
                                          "usable_mib": self.state.get("usable_mib"), "quants": quants,
                                          "iq3_copy": self.state.get("iq3_copy")}}
        if mini:
            hosts["3090ti"] = mini
        doc = {"schema": TEACHER_SCHEMA, "written_at": now_iso(),
               "method": {"ladder": self.cfg.ladder, "needles_per_prompt": 3, "depths": DEPTHS,
                          "formats": FORMATS, "requests_per_rung": 2, "max_tokens": MAX_TOKENS,
                          "prompt": f"N - ({MAX_TOKENS} + {PROMPT_MARGIN}) tokens of real repo/wiki text",
                          "decode_bench_tokens": DECODE_BENCH_TOKENS, "headroom_min_mib": HEADROOM_MIN_MIB,
                          "stable": "loads, both requests answer all 3 needles, no crash, headroom >= floor",
                          "server": "serve_teacher.sh (tailnet, key, tools off) with --parallel 1 -c N; -fa on, "
                                    "q8_0 K/V, no mmproj"},
               "hosts": hosts}
        doc["recommendation"] = recommend(quants, (mini or {}).get("measured"))
        return doc

    def qlora_doc(self) -> Dict[str, Any]:
        res = self.state.get("qlora") or {}
        lengths = {}
        cfg_seen: Dict[str, Any] = {}
        for k, v in res.items():
            v = dict(v)
            if v.get("status") not in ("ok", "oom", "failed", "killed", "skipped"):
                v["status"] = "failed"
            cfg_seen = v.get("config") or cfg_seen
            lengths[k] = {kk: vv for kk, vv in v.items() if kk not in ("config", "losses_head")}
        return {"schema": QLORA_SCHEMA, "written_at": now_iso(), "model": "Qwen3.5-9B, bnb NF4 (provenance-verified copy)",
                "offload": "Unsloth offloaded gradient checkpointing (use_gradient_checkpointing='unsloth'): each "
                           "checkpointed layer input is kept in CPU RAM and reloaded for the backward pass",
                "steps": QLORA_STEPS, "warmup_steps": QLORA_WARMUP, "grad_accum": 1, "micro_batch": 1,
                "data": "v4b train rows (max-derived), concatenated to the target length; no adapter kept",
                "config": cfg_seen, "lengths": lengths, "summary": qlora_summary(res)}

    def write_outputs(self, status: str, reason: str, business_left_down: bool) -> Dict[str, Any]:
        cap = self.cfg.capdir
        teacher = self.teacher_doc()
        qlora = self.qlora_doc()
        errs = validate_teacher_ctx(teacher) + validate_qlora(qlora)
        if errs:
            self.note("output schema problems: " + "; ".join(errs)[:250])
        sfx = "" if self.cfg.execute else "-dryrun"
        write_json(cap / f"teacher-ctx{sfx}.json", teacher)
        if not self.cfg.skip_qlora:
            write_json(cap / f"qlora-seqlen{sfx}.json", qlora)
        rec = teacher.get("recommendation") or {}
        done = {"schema": DONE_SCHEMA, "status": status, "reason": reason or None, "finished": now_iso(),
                "business_left_down": business_left_down,
                "teacher_ctx": "teacher-ctx.json", "qlora_seqlen": None if self.cfg.skip_qlora else "qlora-seqlen.json",
                "max_stable_ctx": {k: ((v.get("summary") or {}).get("max_stable_ctx"))
                                   for k, v in teacher["hosts"]["4090"]["quants"].items()},
                "mini_max_readable_ctx": ((teacher["hosts"].get("3090ti") or {}).get("measured") or {}).get("max_readable_ctx"),
                "long_context_teacher": rec.get("long_context_teacher"),
                "long_context_row_cap_tokens": rec.get("long_context_row_cap_tokens"),
                "qlora_max_ok_seq": qlora["summary"].get("max_ok_seq"),
                "iq3_on_4090": (self.state.get("iq3_copy") or {}).get("status") == "verified",
                "restore": self.state.get("restore"), "notes": self.state.get("notes", [])[-10:]}
        md = render_markdown(teacher, None if self.cfg.skip_qlora else qlora, done)
        probs = scrub_problems(md, [self.biz_name] if self.biz_name != "business" else [])
        if probs:
            self.note("DISTILL-CAPACITY.md not written: it would hold " + ", ".join(probs))
        else:
            (cap / f"DISTILL-CAPACITY{sfx}.md").write_text(md, encoding="utf-8")
        return done

    def headline(self, done: Dict[str, Any]) -> str:
        teacher = read_json(self.cfg.capdir / ("teacher-ctx.json" if self.cfg.execute else "teacher-ctx-dryrun.json")) or {}
        q4 = ((teacher.get("hosts") or {}).get("4090") or {}).get("quants") or {}
        parts = []
        for k in ("q4", "iq3"):
            s = (q4.get(k) or {}).get("summary") or {}
            am = s.get("at_max") or {}
            if s.get("max_stable_ctx"):
                parts.append(f"{(q4[k]).get('label')} {k_label(s['max_stable_ctx'])} "
                             f"({_f(am.get('prefill_tok_s'))}/{_f(am.get('decode_tok_s'), 1)} tok/s)")
            elif k in q4:
                parts.append(f"{(q4[k]).get('label')} -")
        ql = (self.state.get("qlora") or {})
        qparts = []
        for k in sorted(ql, key=int):
            v = ql[k]
            qparts.append(f"{int(k) // 1024}K " + (f"{_f(v.get('tok_s'))} tok/s" if v.get("status") == "ok"
                                                   else str(v.get("status"))))
        mini = done.get("mini_max_readable_ctx")
        lines = [f"Capacity tests {done['status']}" + (f" ({done.get('reason')[:80]})" if done.get("reason") else "") + ".",
                 "4090 teacher max stable ctx: " + ("; ".join(parts) or "-") + ".",
                 f"3090 Ti reads {k_label(mini)}; 9B QLoRA+offload: " + (", ".join(qparts) or "-") + ".",
                 f"Long-context row cap: {k_label(done.get('long_context_row_cap_tokens'))} tokens.",
                 (f"{self.biz_name} left down for generation." if done["business_left_down"]
                  else f"{self.biz_name} restore: {'ok' if (done.get('restore') or {}).get('ok') else 'see log'}.")]
        return "\n".join(lines)

    # -- main -----------------------------------------------------------------------------------------
    def on_term(self, *_: Any) -> None:
        self.term = True

    def run(self) -> int:
        c = self.cfg
        done_path = c.capdir / "CAPACITY-DONE.json"
        if c.execute and done_path.is_file():
            self.log("CAPACITY-DONE.json exists: nothing to do (remove it to run again)")
            return 0
        if c.execute:
            self.state["starts"] = int(self.state.get("starts", 0)) + 1
            self.save()
        self.log(f"capacity block {'EXECUTE' if c.execute else 'DRY RUN'}; start #{self.state.get('starts')}; "
                 f"skip at {et(c.skip_at)}, restore after {et(c.restore_after)}, budget {c.budget_s / 60:.0f} min")
        if c.execute and self.state["starts"] > MAX_STARTS:
            self.note(f"{MAX_STARTS} restarts: giving up")
            if self.state.get("biz_down_by_us"):
                self.release("gave up after repeated crashes")
            done = self.write_outputs("failed", "gave up after repeated crashes", False)
            write_json(done_path, done)
            return 0
        if c.execute:
            for u in self.state.get("units", []):  # a crashed start may have left one running
                if self.unit_state(u["name"]) in ACTIVE and u.get("kind") != "sampler":
                    self.log(f"leftover unit from an earlier start: {u['name']}")
                    self.stop_unit(u["name"])
        status, reason, failure = "done", "", False
        try:
            self.step_prompts()
            self.step_wait()
            self.read_biz_name()
            self.step_take_gpu()
            rc, out = self.tower.run("nvidia-smi --query-gpu=memory.total,memory.reserved --format=csv,noheader,nounits")
            parts = [to_int(x) for x in ((out.splitlines() or [""])[-1]).split(",")]
            if len(parts) == 2 and parts[0] and parts[1] is not None:
                self.state["usable_mib"] = parts[0] - parts[1]
            self.step_sampler()
            self.step_copy_start()
            for key in c.quants:
                self.step_ladder(key)
                self.step_multi(key)
            self.step_qlora()
            partial = any((self.state["quants"].get(k) or {}).get("status") in ("partial", "skipped")
                          for k in c.quants) or any(v.get("status") == "skipped" for v in (self.state.get("qlora") or {}).values())
            status = "partial" if partial else "done"
        except NotRun as exc:
            self.mark("abort", "not-run", str(exc))
            self.notify("skip", f"Capacity tests skipped: {str(exc)[:200]}.\nGeneration is not blocked (CAPACITY-DONE: skipped).")
            done = self.write_outputs("skipped", str(exc), False)
            write_json(c.capdir / ("CAPACITY-DONE.json" if c.execute else "CAPACITY-DONE-dryrun.json"), done)
            return 0
        except Stop as exc:
            status, reason, failure = ("partial" if not exc.failure else "failed"), str(exc), exc.failure
            self.mark("abort", "stopped", str(exc))
        except Exception as exc:  # noqa: BLE001 - restore, report, mark done; never block generation silently
            status, reason, failure = "failed", f"{type(exc).__name__}: {str(exc)[:200]}", True
            self.log(traceback.format_exc())
            self.mark("abort", "crashed", reason)
        # end: stop ours, decide business, write the outputs and the marker
        self.stop_own_units()
        left_down = False
        biz_by_us = bool(self.state.get("biz_down_by_us"))
        late = self.clock() >= c.restore_after
        if self.state.get("biz_back_elsewhere"):
            left_down = False
        elif not failure and not late and c.generation_follows and self.state.get("gpu_taken"):
            left_down = True
        elif biz_by_us or (failure and self.state.get("gpu_taken")):
            why = reason or ("past " + et(c.restore_after) if late else "finished; no generation follows")
            self.release(why)
        if self.term and c.execute:
            reason = (reason + "; " if reason else "") + "stopped by hand"
        done = self.write_outputs(status, reason, left_down)
        write_json(c.capdir / ("CAPACITY-DONE.json" if c.execute else "CAPACITY-DONE-dryrun.json"), done)
        self.mark("end", status, reason)
        self.notify("results", self.headline(done))
        return 0


def load_failure_kind(tail: str) -> str:
    if "serve_teacher: REFUSED" in tail:
        return "refused"
    return "load-failed"


def load_failure_reason(tail: str) -> str:
    """A short, path-free reason from a probe server's log tail."""
    t = tail or ""
    m = re.search(r"serve_teacher: REFUSED: ([^\n]+)", t)
    if m:
        s = m.group(1)
        if "need" in s and "MiB" in s:
            return "refused by the VRAM projection: weights + KV exceed the card"
        return "refused by serve_teacher.sh's checks: " + re.sub(r"/\S+", "<path>", s)[:160]
    for pat, why in ((r"out of memory|cudaMalloc failed|failed to allocate", "out of memory at load"),
                     (r"failed to create context|failed to load model", "llama.cpp could not create the context"),
                     (r"KV cache", "KV cache allocation failed")):
        if re.search(pat, t, re.I):
            return why
    last = [ln for ln in t.splitlines() if ln.strip()][-1:] or ["exited during load"]
    return re.sub(r"/\S+", "<path>", last[0])[:160]


def here_units_alive(pattern: str) -> Optional[List[str]]:
    """User units here matching ``pattern`` that are running or (re)starting; None when systemd cannot
    be asked (then nothing is concluded)."""
    try:
        p = subprocess.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "--no-pager",
                            "--state=active,activating,deactivating,reloading", pattern],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return [ln.split()[0] for ln in p.stdout.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Fakes (GPU-free rehearsal of the block on the 3090 Ti host: --fake-root)
# ---------------------------------------------------------------------------

_NOTE_RE = re.compile(r"note: the (?:rollback code|maintenance passphrase|escalation code|current release tag) "
                      r"for (the [a-z]+ [a-z ]+?) is ([A-Z]+-\d{4})\.")
_ASK_RE = re.compile(r"(\d)\) What is the (?:rollback code|maintenance passphrase|escalation code|release tag) "
                     r"for (the [a-z]+ [a-z ]+?)\?")


def fake_answer(messages: List[Dict[str, Any]], readable_chars: Optional[int] = None) -> str:
    """What a perfect reader answers (tests and the fake server): every asked subject's value. With
    ``readable_chars``, notes past that many characters of context are 'not read'."""
    text = "\n".join(str(m.get("content") or "") for m in messages)
    seen = text if readable_chars is None else text[:readable_chars]
    notes = {s: v for s, v in _NOTE_RE.findall(seen)}
    last = str(messages[-1].get("content") or "") if messages else ""
    lines = []
    for num, subj in _ASK_RE.findall(last):
        lines.append(f"{num}: {notes.get(subj, 'UNKNOWN')}")
    return "\n".join(lines) or "UNKNOWN"


def fake_server(argv: Sequence[str]) -> int:
    """A stand-in llama-server for the block's rehearsal. Reads FAKE_* settings from its environment:
    FAKE_CTX, FAKE_SLOTS, FAKE_PORT, FAKE_ALIAS, FAKE_STATE (VRAM file for the fake nvidia-smi),
    FAKE_WEIGHTS_MIB, FAKE_MAX_READ (chars read per prompt)."""
    env = os.environ
    ctx, port = int(env.get("FAKE_CTX", "32768")), int(env.get("FAKE_PORT", "18194"))
    slots = int(env.get("FAKE_SLOTS", "1"))
    alias = env.get("FAKE_ALIAS", "fake")
    state = Path(env.get("FAKE_STATE", "/nonexistent"))
    weights = int(env.get("FAKE_WEIGHTS_MIB", "9000"))
    max_read = to_int(env.get("FAKE_MAX_READ"))
    used = weights + int(ctx * KV_Q8_BYTES_PER_TOKEN / 1048576) + 150 * slots + 1200
    try:
        state.write_text(str(used))
    except OSError:
        pass
    time.sleep(1.0)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send({"status": "ok"})
            elif self.path == "/props":
                self._send({"default_generation_settings": {"n_ctx": ctx // slots}, "total_slots": slots})
            elif self.path == "/v1/models":
                self._send({"data": [{"id": alias, "meta": {"n_ctx_train": 262144}}]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            msgs = req.get("messages") or []
            chars = sum(len(str(m.get("content") or "")) for m in msgs)
            content = fake_answer(msgs, max_read)
            pn = max(1, chars // 4)
            self._send({"choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": pn, "completion_tokens": 12, "total_tokens": pn + 12},
                        "timings": {"prompt_n": pn, "prompt_ms": pn / 2.0, "prompt_per_second": 2000.0,
                                    "predicted_n": int(req.get("max_tokens") or 12) if req.get("ignore_eos") else 12,
                                    "predicted_ms": 400.0, "predicted_per_second": 30.0, "cache_n": 0}})

    srv = http.server.ThreadingHTTPServer((LOOPBACK, port), H)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # a unit stop frees the fake VRAM (finally below)
    print(f"fake llama-server {alias} -c {ctx} x{slots} on {LOOPBACK}:{port} ({used} MiB)", flush=True)
    try:
        srv.serve_forever()
    finally:
        try:
            state.write_text("0")
        except OSError:
            pass
    return 0


def fake_setup(root: Path, code_rel: str, *, max_ctx: Dict[str, int], max_read: Optional[int] = None,
               qlora_max: int = 32768, data_dir: Optional[Path] = None, tokenizer_dir: Optional[Path] = None) -> None:
    """A fake 4090 tree under ROOT/tower (serve script, nvidia-smi, notify, train_end.sh, the bench
    python) and a fake here under ROOT/here. The business stand-in is the unit capfake-biz.service."""
    t, h = root / "tower", root / "here"
    for d in ("bin", "distill/teacher-serve", "distill/window/state", "distill/capacity", "models",
              "distill/models/teacher", "distill/models/qwen38-27b-gguf", "distill/venv-train/bin",
              "distill/phase3", "distill/models/qwen3.5-9b-bnb4"):
        (t / d).mkdir(parents=True, exist_ok=True)
    # the benchmark reads real rows with the real tokenizer here (read only), through links
    for rel, src in (("distill/phase3/build-v4b-max-derived", data_dir),
                     ("distill/models/qwen3.5-9b-tokenizer", tokenizer_dir)):
        dst = t / rel
        if src is not None and not dst.exists():
            dst.symlink_to(Path(src).expanduser().resolve())
        elif not dst.exists():
            dst.mkdir(parents=True)
    for d in ("distill/secrets", "distill/capacity", "distill/phase7/run", "distill/models/teacher"):
        (h / d).mkdir(parents=True, exist_ok=True)
    me = Path(__file__).resolve()
    sh = "#!/bin/bash\n"
    vram = root / "vram"
    vram.write_text("0")
    (t / "bin/nvidia-smi").write_text(
        sh + f"u=$(cat {q(str(vram))} 2>/dev/null || echo 0); case \"$*\" in\n"
        "  *memory.total,memory.reserved*) echo '24564, 491' ;;\n"
        "  *memory.used,memory.free*) echo \"$u, $((24073 - u))\" ;;\n"
        "  *memory.free*) echo $((24073 - u)) ;;\n"
        "  *) echo $u ;;\nesac\n")
    # the serve script: refuses above the control's max context, else runs the fake server
    mx = ",".join(f"{k}={v}" for k, v in max_ctx.items())
    (t / "distill/teacher-serve/serve_teacher.sh").write_text(
        sh + "cmd=$1\n"
        "case \"$TEACHER_ALIAS\" in *q4*) k=q4 ;; *) k=iq3 ;; esac\n"
        f"mx=$(echo {q(mx)} | tr ',' '\\n' | sed -n \"s/^$k=//p\")\n"
        "tot=$((TEACHER_SLOTS * TEACHER_CTX_PER_SLOT))\n"
        "if [ \"$cmd\" = verify ]; then echo 'serve_teacher: verified (fake)'; exit 0; fi\n"
        "if [ -n \"$mx\" ] && [ \"$tot\" -gt \"$mx\" ]; then echo 'ggml_backend_cuda_buffer_type_alloc_buffer: "
        "allocating 9000 MiB on device 0: cudaMalloc failed: out of memory'; exit 1; fi\n"
        f"exec env FAKE_CTX=$tot FAKE_SLOTS=$TEACHER_SLOTS FAKE_PORT=$TEACHER_PORT FAKE_ALIAS=$TEACHER_ALIAS "
        f"FAKE_STATE={q(str(vram))} FAKE_WEIGHTS_MIB=$TEACHER_WEIGHTS_MIB "
        + (f"FAKE_MAX_READ={int(max_read)} " if max_read else "")
        + f"/usr/bin/python3 -I {q(str(me))} fake-server\n")
    for p in (t / "bin/nvidia-smi", t / "distill/teacher-serve/serve_teacher.sh"):
        p.chmod(0o755)
    # the model files the block checks (small stand-ins)
    (t / "models/Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(b"q4")
    src = h / IQ3_SRC
    src.write_bytes(b"fake iq3 gguf\n")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    (h / IQ3_SRC_MANIFEST).write_text(json.dumps({"file": {"sha256": sha}}) + "\n")
    (root / "quant-override.json").write_text(json.dumps({"iq3": {"bytes": src.stat().st_size, "sha256": sha}}))
    # the bench: the system python runs seqlen_bench.py with its fake backend
    vpy = t / "distill/venv-train/bin/python"
    vpy.write_text(sh + f"export FAKE_QLORA_MAX={qlora_max}\nexec /usr/bin/python3 -E -P \"$@\"\n")
    vpy.chmod(0o755)
    code_dst = t / code_rel
    if not code_dst.exists():
        code_dst.parent.mkdir(parents=True, exist_ok=True)
        code_dst.symlink_to(me.parents[2])
    # notify + train_end + window.conf
    tg = root / "telegrams.txt"
    (t / "distill/notify.sh").write_text(sh + f"{{ echo \"--- $(date +%T)\"; head -\"${{1:-5}}\"; }} >> {q(str(tg))}\n")
    (t / "distill/notify.sh").chmod(0o755)
    (t / "distill/window/window.conf").write_text("BIZ_NAME=FakeBiz\n")
    (t / "distill/window/train_end.sh").write_text(
        sh + f"S={q(str(t / 'distill/window/state'))}; msg=''; [ \"$1\" = --msg-file ] && msg=$(head -3 \"$2\")\n"
        "systemd-run --user --quiet --collect --unit=capfake-biz.service /bin/sleep 86400 || true\n"
        "echo \"$(date '+%F %T') tower_verify: PASS fake\" > $S/verify.last\n"
        f"printf 'FakeBiz back: business RESTORED + verified.\\n%s\\n' \"$msg\" | {q(str(t / 'distill/notify.sh'))} 5\n")
    (t / "distill/window/train_end.sh").chmod(0o755)
    # the key, the pipeline marker's directory
    key = h / "distill/secrets/teacher-4090.key"
    key.write_text("0" * 64 + "\n")
    key.chmod(0o600)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_block_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--tower", help="USER@HOST of the 4090 (ssh)")
    ap.add_argument("--tower-url", help="the probe server's URL from here: http://<4090 tailnet name>:18094")
    ap.add_argument("--port", default="18094", help="the probe server's port (serve_teacher's)")
    ap.add_argument("--capdir", default=None)
    ap.add_argument("--key-file", default=None, help="here: the 4090 teacher key (default ~/distill/secrets/teacher-4090.key)")
    ap.add_argument("--pipeline-done", default=None)
    ap.add_argument("--pipeline-unit", default="flywheel-post-train-*",
                    help="the post-train unit pattern here (none alive for an hour and no marker: skip)")
    ap.add_argument("--skip-at", default="2026-11-02T03:00:00-05:00")
    ap.add_argument("--restore-after", default="2026-11-02T04:00:00-05:00")
    ap.add_argument("--window-end", default="2026-11-02T05:40:00-05:00")
    ap.add_argument("--budget-min", default="150")
    ap.add_argument("--poll", default="60")
    ap.add_argument("--no-generation-follows", action="store_true", help="restore business at the end even on success")
    ap.add_argument("--quants", default="q4,iq3")
    ap.add_argument("--ladder", default="")
    ap.add_argument("--qlora-lengths", default="")
    ap.add_argument("--qlora-runtime-s", default=str(QLORA_RUNTIME_S))
    ap.add_argument("--skip-qlora", action="store_true")
    ap.add_argument("--code-rel", default="distill/capacity/code/head", help="the staged code on the 4090 (from its home)")
    ap.add_argument("--repo", default="~/Prometheus")
    ap.add_argument("--wiki", default="~/.prometheus/wiki")
    ap.add_argument("--tokenizer-dir", default="~/distill/models/qwen3.5-9b-tokenizer")
    ap.add_argument("--template", default="~/distill/models/teacher/provenance/base-chat_template.jinja")
    ap.add_argument("--iq3-src", default=None)
    ap.add_argument("--iq3-manifest", default=None)
    ap.add_argument("--quant-override", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--fake-root", default=None, help="rehearse against a fake 4090 tree (fake-setup)")


def cmd_block(a: argparse.Namespace) -> int:
    cfg = Cfg(a)
    if not cfg.tower and not cfg.fake:
        print("block: --tower is required", file=sys.stderr)
        return 2
    if not cfg.tower_url:
        print("block: --tower-url is required", file=sys.stderr)
        return 2
    b = Block(cfg)
    signal.signal(signal.SIGTERM, b.on_term)
    signal.signal(signal.SIGINT, b.on_term)
    try:
        return b.run()
    except Exception:  # noqa: BLE001 - a crash: the unit restarts and resumes
        b.log(traceback.format_exc())
        return 1


def cmd_arm(a: argparse.Namespace) -> int:
    stamp = datetime.now(ET).strftime("%Y%m%dT%H%M%S")
    unit = f"flywheel-capacity-{stamp}"
    me = Path(__file__).resolve()
    capdir = Path(a.capdir).expanduser() if a.capdir else Path.home() / "distill" / "capacity"
    log = capdir / "unit.log"
    fwd: List[str] = []
    for k, v in sorted(vars(a).items()):
        if k in ("cmd", "execute", "func") or v in (None, "", False):
            continue
        flag = "--" + k.replace("_", "-")
        fwd += [flag] if v is True else [flag, str(v)]
    argv = ["systemd-run", "--user", f"--unit={unit}",
            "--description=flywheel: teacher/trainer capacity block (waits for PIPELINE-DONE)",
            "-p", "KillMode=mixed", "-p", "TimeoutStopSec=1800", "-p", "Restart=on-failure", "-p", "RestartSec=120",
            "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
            "--setenv=TZ=America/New_York", "--setenv=PYTHONUNBUFFERED=1", "--expand-environment=no",
            # -E -P, not -I: the tokenizer package lives in the user's site-packages here
            "/usr/bin/python3", "-E", "-P", str(me), "block", *fwd, "--execute"]
    print(shlex.join(argv))
    if not a.execute:
        print("dry run: nothing started (--execute starts it)")
        return 0
    capdir.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(argv).returncode
    if rc == 0:
        (capdir / "unit.name").write_text(unit + ".service\n", encoding="utf-8")
        print(f"armed: {unit}.service (stop: systemctl --user stop {unit}.service)")
    return rc


def cmd_report(a: argparse.Namespace) -> int:
    cap = Path(a.capdir).expanduser() if a.capdir else Path.home() / "distill" / "capacity"
    teacher = read_json(cap / "teacher-ctx.json")
    qlora = read_json(cap / "qlora-seqlen.json")
    done = read_json(cap / "CAPACITY-DONE.json")
    if teacher is None and (cap / "mini-ctx.json").is_file():
        mini = mini_host_doc(read_json(cap / "mini-ctx.json"))
        teacher = {"schema": TEACHER_SCHEMA, "hosts": {"3090ti": mini} if mini else {},
                   "recommendation": recommend({}, (mini or {}).get("measured"))}
    md = render_markdown(teacher, qlora, done, pending_4090=done is None)
    probs = scrub_problems(md, [a.banned] if a.banned else [])
    if probs:
        print("refused: the document would hold " + ", ".join(probs), file=sys.stderr)
        return 1
    out = Path(a.out).expanduser() if a.out else cap / "DISTILL-CAPACITY.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    cap = Path(a.capdir).expanduser() if a.capdir else Path.home() / "distill" / "capacity"
    st = read_json(cap / "block-state.json") or {}
    print(json.dumps({"starts": st.get("starts"), "steps": {k: v.get("status") for k, v in (st.get("steps") or {}).items()},
                      "gpu_taken": st.get("gpu_taken"), "iq3_copy": (st.get("iq3_copy") or {}).get("status"),
                      "quants": {k: {"status": v.get("status"), "rungs": [(r.get("ctx"), r.get("reason")) for r in v.get("rungs", [])]}
                                 for k, v in (st.get("quants") or {}).items()},
                      "qlora": {k: v.get("status") for k, v in (st.get("qlora") or {}).items()},
                      "done": (cap / "CAPACITY-DONE.json").is_file(), "mini": (cap / "mini-ctx.json").is_file()},
                     indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mini", help="the 3090 Ti teacher, as it runs")
    m.add_argument("--server-url", default="http://127.0.0.1:8092")
    m.add_argument("--guard-url", default="http://127.0.0.1:18593")
    m.add_argument("--guard-abort", default="~/distill/phase2b/runs/state/guard.abort")
    m.add_argument("--window-conf", default="~/distill/window/window.conf")
    m.add_argument("--capdir", default="~/distill/capacity")
    m.add_argument("--repo", default="~/Prometheus")
    m.add_argument("--wiki", default="~/.prometheus/wiki")
    m.add_argument("--tokenizer-dir", default="~/distill/models/qwen3.5-9b-tokenizer")
    m.add_argument("--template", default="~/distill/models/teacher/provenance/base-chat_template.jinja")
    m.add_argument("--execute", action="store_true")
    b = sub.add_parser("block", help="the 4090 measurement block")
    add_block_args(b)
    b.add_argument("--execute", action="store_true")
    ar = sub.add_parser("arm", help="start `block` as a waiting unit here")
    add_block_args(ar)
    ar.add_argument("--execute", action="store_true")
    r = sub.add_parser("report", help="re-render DISTILL-CAPACITY.md")
    r.add_argument("--capdir", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--banned", default=None, help="a word the document must not hold")
    s = sub.add_parser("status")
    s.add_argument("--capdir", default=None)
    fs = sub.add_parser("fake-setup", help="build a fake 4090 tree for a GPU-free rehearsal")
    fs.add_argument("--fake-root", required=True)
    fs.add_argument("--code-rel", default="distill/capacity/code/head")
    fs.add_argument("--max-ctx", default="q4=163840,iq3=262144")
    fs.add_argument("--max-read", default=None)
    fs.add_argument("--qlora-max", default="32768")
    fs.add_argument("--data-dir", default=None, help="a v4b build to link (read only)")
    fs.add_argument("--tokenizer-dir", default=None)
    sub.add_parser("fake-server", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.cmd == "mini":
        return cmd_mini(a)
    if a.cmd == "block":
        return cmd_block(a)
    if a.cmd == "arm":
        return cmd_arm(a)
    if a.cmd == "report":
        return cmd_report(a)
    if a.cmd == "status":
        return cmd_status(a)
    if a.cmd == "fake-setup":
        mc = {k: int(v) for k, v in (x.split("=") for x in a.max_ctx.split(",") if x)}
        fake_setup(Path(a.fake_root).expanduser().resolve(), a.code_rel, max_ctx=mc,
                   max_read=to_int(a.max_read), qlora_max=int(a.qlora_max),
                   data_dir=Path(a.data_dir) if a.data_dir else None,
                   tokenizer_dir=Path(a.tokenizer_dir) if a.tokenizer_dir else None)
        print(f"fake 4090 tree under {a.fake_root}")
        return 0
    if a.cmd == "fake-server":
        return fake_server([])
    return 2


if __name__ == "__main__":
    sys.exit(main())
