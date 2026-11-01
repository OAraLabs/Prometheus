#!/usr/bin/env python3
"""After the first full 9B QLoRA run: quantize, score, report, restore. Unattended and resumable.

    post_train.py subset --cases EVAL_TOOLS.jsonl [--out IDS.json]   # the fixed 100-case subset
    post_train.py run --tower USER@HOST [--execute]                    # without --execute: dry run
    post_train.py arm --tower USER@HOST [--execute]                    # start the waiting unit here
    post_train.py status                                               # the state file, short

Where it runs: the 3090 Ti host (where ``phase6_student.sh`` runs and Saluki's summary lives), as
the user unit ``flywheel-post-train-<stamp>`` (``arm``). Not ``distill-*``: a restore on either host
stops every ``distill-*`` unit and must not stop this one. Everything that holds the 4090's GPU
runs on the 4090 host as a ``distill-*`` user unit (``distill-quantize-p6-<run>``,
``distill-quantize-p7-recipe``, ``distill-p7-srv-*``, ``distill-p7-score-*``), so the continuous
window's end (Mon 05:40 EST) and the dead-man (06:00 EST) still stop them and restore business.
Units are stopped only by the names this pipeline recorded; nothing is killed by pattern.

State: ``~/distill/phase7/<run>/state.json`` (every step's status; ``status`` prints it). Log:
``post-train.log`` beside it. A re-run (or the unit's restart after a crash) skips what is done and
resumes the rest; ``score.py`` itself skips the cases it already answered. The end leaves
``PIPELINE-DONE.json`` beside the state (the outcome, the report's path, the restore result).

Steps, in order (expected durations at the 4090's speed):
 1. wait-train: until the trainer unit has ended and its watcher (which Telegrams the end, then
    runs ``train_end.sh``: apply business + tower_verify under the window lock) has ended too.
 2. verdict: the run's ``manifest.json`` + ``run_state.json``: status ``done``, the adapter on disk,
    every epoch recorded, finite held-out losses, val and probe loss lower at the end than at step 0
    (the smoke gate's held-out checks, applied to the full run). Otherwise: Telegram why, stop
    (nothing touches the GPU; the watcher has restored business).
 3. take-gpu: no NO-SWAPS flag; under the window lock, if business is up: Telegram "<business>
    going down: post-training quantize + eval", stop ``llama-server.service`` through its unit
    (as ``train_full.py`` does) and wait for the VRAM; else confirm the GPU is free (<= 1.5 GB).
 4. phase6 (~45 min): ``phase6_student.sh <adapter> --execute --keep-bf16 --stop-at <stop>``:
    pull, stream merge here, train-only calibration, push, convert + imatrix + Q4_K_M + Q5_K_M on
    the 4090 GPU, fetch, cleanup (the BF16 GGUF stays on the 4090 for scoring). Exit 75/99 (paused:
    memory or stop-at) waits and resumes; any other failure is retried once.
 5. subset: the fixed 100-case eval-tools subset (``select_subset``: seed ``phase7-bf16-subset/1``,
    stratified by ``meta.category`` with largest-remainder allocation, sha256(seed, id) order inside
    each stratum), checked against the committed ids file, pushed to the 4090.
 6. scoring on the 4090 GPU, one llama-server at a time, the server and flags base-9b was scored
    with (``~/llama.cpp/build/bin/llama-server -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja
    -ngl 99 -fit off -b 2048 -ub 2048 --host 127.0.0.1 --port 18080``, plus ``--alias <label>``),
    ``score.py run --concurrency 1`` from the scorer code base-9b used, the evalsets in
    ``~/distill/evalsets``; priority order:
      a. Q4_K_M eval-tools + sanity (~25 min)        b. Q5_K_M eval-tools + sanity (~25 min)
      c. BF16 on the 100-case subset (~12 min; -c the largest of 262144/131072/65536/32768 that
         loads on 24 GB, recorded; the longest eval-tools prompt + max_tokens is ~20.4K)
      d. Q4_K_M eval-injection + eval-longctx (~80 min)
      e. the official base 9B through the student's exact recipe: the BF16 GGUF moves to this host
         (sha256-checked) to free the 4090's disk, the official bf16 is pushed there, the same
         calibration text, ``quantize.py run --no-adapter`` (--no-mtp convert, imatrix, Q4_K_M) as
         ``distill-quantize-p7-recipe`` (~35 min), then eval-tools + sanity (~25 min) and, if time
         allows, eval-injection + eval-longctx (~80 min).
    A server that dies during an item is restarted once and the item resumed; a second death skips
    the item. ``score.py`` exit 3 (errors) re-runs the set up to 3 passes.
 7. release: stop any unit of ours still running, then the window machinery's restore path:
    ``train_end.sh --msg-file`` as ``flywheel-p7-restore-<stamp>`` (apply business + tower_verify,
    which checks the tunnel connector's local /ready, never the public hostname; Telegram
    "<business> back"). Runs whenever this pipeline took the GPU, whatever happened.
 8. report: results fetched here, summarized per model with this checkout's ``score.py``,
    ``score.py compare`` (the CI gate) of Q4_K_M and Q5_K_M against base-9b (Unsloth GGUF) and the
    same-recipe base with Saluki alongside; ``report/DISTILL-PHASE7-SCORES.md`` (scrubbed: no host
    paths, IPs or business names), a <= 5-line Telegram with the headline and the 4-bit flag.
Total after training: ~5 h with every item; the restore starts by --stop-at at the latest.

Guards, checked before and during every GPU step: ``--stop-at`` (default Mon 2026-11-02 05:00 EST;
scoring stops, then release; the hard deadline is 05:20 EST), a NO-SWAPS flag, business coming back
from elsewhere (continuous_end, the dead-man: stop ours, release), free disk under 10 GiB on either
host (stop with a message), SIGTERM (``systemctl --user stop`` of this unit: stop ours, release).

Dry run (no ``--execute``): read-only probes run (unit states, the manifest, disk, VRAM); every
command that would change something is printed instead; waits do not wait; ``phase6_student.sh``
runs in its own plan mode (``--dry-adapter`` points it at an existing adapter, e.g. the smoke's).
``--fake-root DIR`` runs the whole pipeline for real against a fake 4090 tree on this host (a fake
llama-server, ``fake-setup`` builds it): every step, unit, retry and report path without a GPU.

Max-derived (DISTILL-OPEN-QUESTIONS Q6b): every label and file name carries ``max-derived``;
nothing is uploaded anywhere; files move between our two hosts only.

Exit: 0 finished (any outcome; the marker says which), 1 crashed (the unit restarts and resumes),
2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo

    ET: Any = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - no tz database: print UTC
    ET = timezone.utc

STATE_SCHEMA = "distill-post-train/1"
SUBSET_SCHEMA = "distill-eval-subset/1"
SUBSET_SEED = "phase7-bf16-subset/1"
SUBSET_N = 100
SUBSET_FILE = "DISTILL-PHASE7-SUBSET.json"  # docs/sprints/: the committed ids
FLAG_PTS = 3.0
IDLE_VRAM_MAX_MIB = 1500
DISK_FLOOR_GIB = 10.0
MAX_STARTS = 8
CTX_FULL = 262144
BF16_CTX_LADDER = (262144, 131072, 65536, 32768)
# The server context per model: the first that loads (and does not crash). Q4_K_M (the gate's
# candidate) and the same-recipe base stay at base-9b's 262144 (like for like, 256K long context);
# Q5_K_M (eval-tools + sanity: prompts up to ~20.4K) may step down; BF16 must (24 GB).
CTX_LADDER = {"q4": (CTX_FULL,), "recipe": (CTX_FULL,), "q5": (CTX_FULL, 131072, 65536),
              "bf16": BF16_CTX_LADDER}
# Phase 6 attempts: the 4090 step at full offload; once more; then with fewer layers on the GPU
# (the BF16 imatrix holds 17.9 GB of weights on a 24 GB card) -- quantize.py resumes either way.
PHASE6_ATTEMPTS: Tuple[Tuple[str, ...], ...] = ((), (), ("--tower-ngl", "20"))
SET_TIMEOUT = {"sanity": 600, "eval-tools": 900, "eval-injection": 900, "eval-longctx": 1800,
               "eval-tools-subset": 900}
SET_PASSES = 3
NOTIFY_LINES = 5
RUN_DEFAULT = "9b-r32-v4b-max-derived"
ADAPTER_DEFAULT = "adapter-max-derived"
RECIPE_RUN = "base-9b-recipe-max-derived"
STOP_AT_DEFAULT = "2026-11-02T05:00:00-05:00"
HARD_AT_DEFAULT = "2026-11-02T05:20:00-05:00"
BIZ_UNIT = "llama-server.service"
TOWER_PREFIX = "distill-p7-"
RESTORE_PREFIX = "flywheel-p7-restore-"
ORCH_PREFIX = "flywheel-post-train-"
EVAL_TOOLS_SHA = "9b90751a061c658342d5fc10b7b4266209a5d174761a4e789c179f1aa2844ff5"
BASE9B_SUMMARY_REL = "distill/phase4/results/base-9b/baseline/base-9b.head-4ef1dfe.summary.json"
SALUKI_SUMMARY_REL = "distill/phase4/results/saluki/saluki.head-4ef1dfe.summary.json"
LOOPBACK = "127.0.0.1"
# Minutes, for the go/no-go of each item before --stop-at and the ETA in the down Telegram.
EST_MIN = {"phase6": 50, "a": 25, "b": 25, "c": 12, "d": 80, "move": 6, "recipe": 35, "e1": 25,
           "e2": 80}
# (key, model, sets, optional): the operator's priority order.
ITEMS: Tuple[Tuple[str, str, Tuple[str, ...], bool], ...] = (
    ("a", "q4", ("eval-tools", "sanity"), False),
    ("b", "q5", ("eval-tools", "sanity"), False),
    ("c", "bf16", ("eval-tools-subset",), False),
    ("d", "q4", ("eval-injection", "eval-longctx"), False),
    ("e1", "recipe", ("eval-tools", "sanity"), False),
    ("e2", "recipe", ("eval-injection", "eval-longctx"), True),
)
ACTIVE = ("active", "activating", "deactivating", "reloading")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?[+-]\d{2}:\d{2}$")


# ---------------------------------------------------------------------------
# Pure decision logic (tested without a host)
# ---------------------------------------------------------------------------


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def training_verdict(man: Any, rs: Any, *, adapter_present: bool) -> Tuple[bool, str, Dict[str, Any]]:
    """Did the run complete, by what it recorded? (ok, one-line reason, facts for the report)."""
    man = man if isinstance(man, dict) else {}
    rs = rs if isinstance(rs, dict) else {}
    status = man.get("status")
    info: Dict[str, Any] = {"status": status, "step": man.get("step"),
                            "tokens_seen": man.get("tokens_seen")}
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
    got = sorted({e.get("epoch") for e in done if isinstance(e, dict) and isinstance(e.get("epoch"), int)})
    info["epochs_done"] = got
    if isinstance(epochs, int) and len(got) < epochs:
        return False, f"only {len(got)} of {epochs} epochs recorded", info
    prog = man.get("progress") or rs.get("progress") or {}
    info["total_steps"] = prog.get("total_steps") if isinstance(prog, dict) else None
    hist = [h for h in (man.get("val_history") or rs.get("val_history") or []) if isinstance(h, dict)]
    base = next((h for h in hist if h.get("step") == 0), None)
    last = max(hist, key=lambda h: h.get("step") or 0) if hist else None
    if base is None or last is None or not (last.get("step") or 0) > 0:
        return False, "no held-out evaluation after step 0", info
    bad = sorted({h.get("step") for h in hist for k in ("val_loss", "probe_loss") if not _finite(h.get(k))})
    if bad:
        return False, f"non-finite held-out loss at step(s) {bad}", info
    info.update(val=[base["val_loss"], last["val_loss"]], probe=[base["probe_loss"], last["probe_loss"]],
                last_eval_step=last.get("step"), min_val=min(h["val_loss"] for h in hist),
                nonfinite_steps=man.get("nonfinite_steps"))
    if not last["val_loss"] < base["val_loss"]:
        return False, f"val loss did not fall ({base['val_loss']:.4f} -> {last['val_loss']:.4f})", info
    if not last["probe_loss"] < base["probe_loss"]:
        return False, f"probe loss did not fall ({base['probe_loss']:.4f} -> {last['probe_loss']:.4f})", info
    return True, (f"done at step {man.get('step')}: val {base['val_loss']:.4f} -> {last['val_loss']:.4f}, "
                  f"probe {base['probe_loss']:.4f} -> {last['probe_loss']:.4f}, epochs {got}"), info


def allocate(counts: Dict[str, int], n: int) -> Dict[str, int]:
    """Largest-remainder (Hamilton) allocation of *n* over strata of these sizes; ties by name."""
    total = sum(counts.values())
    if n <= 0 or total <= 0:
        return {k: 0 for k in counts}
    n = min(n, total)
    quota = {k: n * v / total for k, v in counts.items()}
    alloc = {k: min(int(math.floor(q)), counts[k]) for k, q in quota.items()}
    left = n - sum(alloc.values())
    for k in sorted(counts, key=lambda k: (-(quota[k] - alloc[k]), k)):
        if left <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            left -= 1
    return alloc


def subset_key(seed: str, case_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).hexdigest()


def select_subset(cases: Sequence[Dict[str, Any]], n: int = SUBSET_N, seed: str = SUBSET_SEED,
                  source_sha256: str = "") -> Dict[str, Any]:
    """The fixed subset: stratified by meta.category, sha256(seed, id) order within a stratum.
    Ids come back in file order. Deterministic for the same ids, categories and seed."""
    by: Dict[str, List[str]] = {}
    order: List[str] = []
    seen = set()
    for c in cases:
        cid = str(c.get("id"))
        if cid in seen:
            raise ValueError(f"duplicate case id {cid}")
        seen.add(cid)
        order.append(cid)
        cat = str((c.get("meta") or {}).get("category") or "uncategorized")
        by.setdefault(cat, []).append(cid)
    alloc = allocate({k: len(v) for k, v in by.items()}, n)
    chosen = set()
    for cat, ids in by.items():
        chosen.update(sorted(ids, key=lambda i: subset_key(seed, i))[:alloc[cat]])
    ids = [i for i in order if i in chosen]
    return {"schema": SUBSET_SCHEMA, "seed": seed, "n": len(ids), "source_cases": len(order),
            "source_sha256": source_sha256,
            "method": ("stratified by meta.category; largest-remainder allocation of n over the "
                       "strata; within a stratum the ids with the smallest sha256(seed + NUL + id)"),
            "strata": {k: {"cases": len(v), "chosen": alloc[k]} for k, v in sorted(by.items())},
            "ids": ids}


def pts(summary: Optional[Dict[str, Any]], sname: str, metric: str) -> Optional[float]:
    m = (((summary or {}).get("sets") or {}).get(sname) or {}).get("metrics", {}).get(metric)
    if not isinstance(m, dict) or m.get("rate") is None:
        return None
    return 100.0 * float(m["rate"])


def bottleneck_flag(q4: Dict[str, Optional[float]], q5: Dict[str, Optional[float]],
                    threshold: float = FLAG_PTS) -> Tuple[bool, str]:
    """The operator's rule: Q5_K_M beating Q4_K_M by more than 3 points (eval-tools or sanity)
    flags 4-bit as the bottleneck. Returns (flag, the report's line). Exactly 3 points is no flag."""
    parts, hits = [], []
    for key, name in (("eval-tools", "eval-tools task success"), ("sanity", "sanity")):
        a, b = q4.get(key), q5.get(key)
        if a is None or b is None:
            parts.append(f"{name}: not both scored")
            continue
        d = b - a
        parts.append(f"{name} Q5 - Q4 = {d:+.1f} pts")
        if d > threshold + 1e-9:
            hits.append(name)
    if hits:
        return True, (f"FLAG: 4-bit is the bottleneck: Q5_K_M beats Q4_K_M by more than "
                      f"{threshold:g} points on {' and '.join(hits)} ({'; '.join(parts)}).")
    return False, (f"No 4-bit flag: Q5_K_M does not beat Q4_K_M by more than {threshold:g} points "
                   f"({'; '.join(parts)}).")


def subset_rate(outcomes: Optional[Dict[str, bool]], ids: Sequence[str]) -> Dict[str, Any]:
    """k/n over the subset ids this summary scored (missing ids are counted, not guessed)."""
    outcomes = outcomes or {}
    have = [i for i in ids if i in outcomes]
    k = sum(1 for i in have if outcomes[i])
    return {"k": k, "n": len(have), "missing": len(ids) - len(have),
            "rate": (k / len(have)) if have else None}


def may_start(now: float, stop_at: float, need_min: float) -> bool:
    """A step that needs *need_min* minutes may start only if it can end by stop-at."""
    return now + need_min * 60 <= stop_at


def deadline_state(now: float, stop_at: float, hard_at: float) -> str:
    """'run' before stop-at; 'stop' (stop scoring, restore) until the hard deadline; 'hard' after."""
    if now < stop_at:
        return "run"
    return "stop" if now < hard_at else "hard"


def parse_iso(value: str) -> float:
    if not ISO_RE.match(value or ""):
        raise ValueError(f"{value!r}: an ISO time with a UTC offset, e.g. 2026-11-02T05:00:00-05:00")
    return datetime.fromisoformat(value).timestamp()


_HOME_RE = re.compile(r"/home/[A-Za-z0-9._-]+")
_IP_RE = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")


def scrub(text: str) -> str:
    """Host home paths become ~ (the report is committed)."""
    return _HOME_RE.sub("~", text)


def scrub_problems(text: str, banned: Sequence[str]) -> List[str]:
    """What a committed report must not hold: host paths, non-loopback IPs, banned names."""
    out = []
    if "/home/" in text or "/Users/" in text:
        out.append("a host home path")
    for m in _IP_RE.finditer(text):
        if m.group(0) != LOOPBACK and all(0 <= int(g) <= 255 for g in m.groups()):
            out.append(f"an IP address ({m.group(0)})")
    low = text.lower()
    for b in banned:
        if b and b.lower() in low:
            out.append("a banned name")
    return out


def et(ts: float) -> str:
    return datetime.fromtimestamp(ts, ET).strftime("%a %H:%M %Z")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json_text(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "rb") as fh:
        for raw in fh.read().split(b"\n"):  # newlines only (049e047: U+2028 is not a row break)
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def kv_lines(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ln in (text or "").splitlines():
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
        return float(str(x).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Configuration and the two hosts
# ---------------------------------------------------------------------------


class Stop(Exception):
    """Stop GPU work now (deadline, flag, business back from elsewhere, disk, signal), then release."""


class NotRun(Exception):
    """End without touching the GPU (training not completed, a flag before we started)."""


class Cfg:
    def __init__(self, a: argparse.Namespace) -> None:
        self.execute = bool(getattr(a, "execute", False))
        self.tower = a.tower or ""
        self.fake = Path(a.fake_root).expanduser().resolve() if a.fake_root else None
        self.run = a.run
        self.adapter_name = a.adapter_name
        self.trainer_unit = a.trainer_unit
        self.watcher_unit = a.watcher_unit
        self.stop_at_iso = a.stop_at
        self.stop_at = parse_iso(a.stop_at)
        self.hard_at = parse_iso(a.hard_at)
        self.poll = float(a.poll)
        self.dry_adapter = a.dry_adapter
        self.code = Path(__file__).resolve().parent.parent.parent  # the checkout/export root
        here_home = (self.fake / "here") if self.fake else Path.home()
        self.here_home = here_home
        self.hd = here_home / "distill"
        self.p7 = Path(a.state_dir).expanduser() if a.state_dir else self.hd / "phase7" / self.run
        self.notify_path = Path(a.notify).expanduser() if a.notify else self.hd / "notify.sh"
        self.here_py = a.here_python
        self.prefix = a.unit_prefix or (TOWER_PREFIX if not self.fake else "p7fake-")
        self.restore_prefix = a.restore_prefix or (RESTORE_PREFIX if not self.fake else "p7fake-restore-")
        self.quant_unit_p6 = "distill-quantize-p6-" + re.sub(r"[^A-Za-z0-9-]", "-", self.run)
        self.recipe_unit = (self.prefix if self.fake else "distill-") + "quantize-p7-recipe"
        self.biz_unit = a.biz_unit or (BIZ_UNIT if not self.fake else "p7fake-biz.service")
        self.unit_glob = "distill-*" if not self.fake else "p7fake-*"
        self.port = int(a.port)
        self.subset_n = int(a.subset_n)
        self.ids_file = Path(a.ids_file).expanduser() if a.ids_file else \
            self.code / "docs" / "sprints" / SUBSET_FILE
        self.phase6_cmd = [str(self.code / "scripts" / "distill" / "phase6_student.sh")] \
            if not a.phase6_cmd else shlex.split(a.phase6_cmd)
        self.skip_recipe = bool(a.skip_recipe)

    @property
    def labels(self) -> Dict[str, str]:
        r = self.run
        return {"q4": f"{r}-Q4_K_M", "q5": f"{r}-Q5_K_M", "bf16": f"{r}-BF16",
                "recipe": f"{RECIPE_RUN}-Q4_K_M"}


class Ops:
    """Every command goes through here: the 4090 host by ssh (or, with --fake-root, a local bash
    whose 'home' is the fake tree), this host directly. Dry runs print what would change."""

    def __init__(self, cfg: Cfg, log: Callable[[str], None]) -> None:
        self.cfg, self.log = cfg, log
        self._thome: Optional[str] = None

    def _run(self, argv: Sequence[str], timeout: float, env: Optional[Dict[str, str]] = None,
             stdin: Optional[str] = None) -> Tuple[int, str]:
        try:
            p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout,
                               env=env, input=stdin)
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout:.0f}s"
        except OSError as exc:
            return 127, str(exc)
        return p.returncode, (p.stdout + p.stderr).strip()

    def ssh_argv(self) -> List[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                self.cfg.tower]

    def tower(self, script: str, *, mut: bool = False, timeout: float = 120,
              stdin: Optional[str] = None) -> Tuple[int, str]:
        if mut and not self.cfg.execute:
            self.log("  [plan] 4090: " + script)
            return 0, ""
        if self.cfg.fake:
            env = dict(os.environ, HOME=str(self.cfg.fake / "tower"),
                       PATH=f"{self.cfg.fake / 'tower' / 'bin'}:{os.environ.get('PATH', '')}")
            return self._run(["bash", "-c", script], timeout, env=env, stdin=stdin)
        for attempt in range(3):
            rc, out = self._run(self.ssh_argv() + [script], timeout, stdin=stdin)
            if rc != 255:
                return rc, out
            self.log(f"  ssh to the 4090 host failed ({out[-160:]}); retry {attempt + 1}/3 in 15 s")
            time.sleep(15)
        return rc, out

    def here(self, argv: Sequence[str], *, mut: bool = False, timeout: float = 600,
             env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        if mut and not self.cfg.execute:
            self.log("  [plan] here: " + shlex.join(list(argv)))
            return 0, ""
        return self._run(argv, timeout, env=env)

    @property
    def thome(self) -> str:
        if self._thome is None:
            if self.cfg.fake:
                self._thome = str(self.cfg.fake / "tower")
            else:
                rc, out = self.tower("echo $HOME")
                line = (out.splitlines() or [""])[-1].strip()
                if rc != 0 or not line.startswith("/"):
                    raise RuntimeError(f"cannot read the 4090 host's home: rc {rc}: {out[-200:]}")
                self._thome = line
        return self._thome

    def remote(self, rel: str) -> str:
        """An rsync source/target on the 4090 host (relative to its home)."""
        if self.cfg.fake:  # a trailing slash means "the contents" to rsync: keep it
            return str(self.cfg.fake / "tower" / rel) + ("/" if rel.endswith("/") else "")
        return f"{self.cfg.tower}:{rel}"

    def rsync(self, srcs: Sequence[str], dst: str, *, timeout: float = 7200) -> Tuple[int, str]:
        argv = ["rsync", "-a", "--partial", "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo="]
        if not self.cfg.fake:
            argv += ["-e", " ".join(self.ssh_argv()[:-1])]
        return self.here(argv + list(srcs) + [dst], mut=True, timeout=timeout)


q = shlex.quote


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    def __init__(self, cfg: Cfg, ops: Optional[Ops] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> None:
        self.cfg = cfg
        self.clock, self._sleep = clock, sleep
        self.cfg.p7.mkdir(parents=True, exist_ok=True)
        self.log_path = cfg.p7 / ("post-train.log" if cfg.execute else "post-train-dryrun.log")
        self.ops = ops or Ops(cfg, self.log)
        self.state_path = cfg.p7 / "state.json"
        st = None
        if self.state_path.is_file():
            st = read_json_text(self.state_path.read_text(encoding="utf-8"))
        self.state: Dict[str, Any] = st if isinstance(st, dict) else {
            "schema": STATE_SCHEMA, "run": cfg.run, "created": now_iso(), "starts": 0, "steps": {},
            "items": {}, "units": [], "telegrams": {}, "notes": []}
        self.term = False
        self._last_probe = 0.0
        self._probe: Dict[str, List[str]] = {}
        self.biz_name = "business"

    # -- bookkeeping -------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"{datetime.now(ET).strftime('%F %T %Z')} {msg}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def save(self) -> None:
        if self.cfg.execute:
            self.state["updated"] = now_iso()
            write_json(self.state_path, self.state)

    def step_done(self, name: str) -> bool:
        return (self.state["steps"].get(name) or {}).get("status") in ("done", "skipped")

    def mark(self, name: str, status: str, detail: str = "") -> None:
        self.state["steps"][name] = {"status": status, "at": now_iso(), "detail": detail[:600]}
        self.log(f"[{name}] {status}" + (f": {detail}" if detail else ""))
        self.save()

    def note(self, text: str) -> None:
        self.state.setdefault("notes", []).append(text[:400])
        self.log(f"note: {text}")
        self.save()

    def notify(self, key: str, text: str) -> bool:
        """<= 5 lines, once per key (the state remembers what was sent)."""
        if key in self.state["telegrams"]:
            return True
        msg = "\n".join(text.splitlines()[:NOTIFY_LINES])
        if not self.cfg.execute:
            self.log("  [plan] Telegram:\n" + "\n".join("    | " + ln for ln in msg.splitlines()))
            return True
        try:
            p = subprocess.run([str(self.cfg.notify_path), str(NOTIFY_LINES)], input=msg + "\n",
                               capture_output=True, text=True, timeout=90)
            ok = p.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        self.log(f"Telegram {'sent' if ok else 'FAILED'}: {msg.splitlines()[0] if msg else ''}")
        if ok:
            self.state["telegrams"][key] = now_iso()
            self.save()
        return ok

    def sleep(self, s: float) -> None:
        end = self.clock() + s
        while self.clock() < end and not self.term:
            self._sleep(min(5.0, max(0.0, end - self.clock())))

    # -- 4090 paths ---------------------------------------------------------------------------
    @property
    def T(self) -> str:
        return self.ops.thome

    def tp(self, rel: str) -> str:
        return f"{self.T}/{rel}"

    @property
    def t_p6(self) -> str:
        return self.tp(f"distill/phase6/{self.cfg.run}")

    @property
    def t_p7(self) -> str:
        return self.tp(f"distill/phase7/{self.cfg.run}")

    @property
    def t_recipe(self) -> str:
        return self.tp(f"distill/phase6/{RECIPE_RUN}")

    def gguf(self, model: str) -> str:
        r = self.cfg.run
        return {"q4": f"{self.t_p6}/gguf/{r}-Q4_K_M.gguf", "q5": f"{self.t_p6}/gguf/{r}-Q5_K_M.gguf",
                "bf16": f"{self.t_p6}/gguf/{r}-BF16.gguf",
                "recipe": f"{self.t_recipe}/gguf/{RECIPE_RUN}-Q4_K_M.gguf"}[model]

    def cases_t(self, sname: str) -> str:
        if sname == "eval-tools-subset":
            return f"{self.t_p7}/subset/eval-tools-subset{self.cfg.subset_n}.jsonl"
        return self.tp(f"distill/evalsets/{sname}.jsonl")

    def cases_here(self, sname: str) -> Path:
        hd = self.cfg.hd
        if sname == "eval-tools":
            return hd / "phase3" / "build-v2" / "eval-tools.jsonl"
        if sname == "eval-tools-subset":
            return self.cfg.p7 / "subset" / f"eval-tools-subset{self.cfg.subset_n}.jsonl"
        return hd / "phase3" / "evalsets" / f"{sname}.jsonl"

    # -- probes -------------------------------------------------------------------------------
    def unit_state(self, unit: str) -> str:
        rc, out = self.ops.tower(f"systemctl --user is-active {q(unit)}")
        st = (out.splitlines() or ["unknown"])[-1].strip() or "unknown"
        if rc == 255 or "ssh" in st.lower():
            return "unknown"
        return st

    def probe(self, force: bool = False) -> Dict[str, List[str]]:
        """One ssh round trip: business unit, NO-SWAPS flags, free disk, VRAM, distill units."""
        if not force and self.clock() - self._last_probe < 60 and self._probe:
            return self._probe
        flags = " ".join(q(p) for p in ("distill/window/NO-SWAPS-TONIGHT", "distill/NO-SWAPS",
                                         ".local/share/distill-setups/NO-SWAPS"))
        script = (f"cd ~ && echo biz=$(systemctl --user is-active {q(self.cfg.biz_unit)}); "
                  f"for f in {flags}; do [ -e \"$f\" ] && echo flag=$f; done; "
                  "echo free_gb=$(df -P -k ~ | awk 'NR==2{printf \"%.1f\", $4/1048576}'); "
                  "echo vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null"
                  " | head -1 | tr -dc 0-9); "
                  "systemctl --user list-units --plain --no-legend --no-pager "
                  f"--state=active,activating,deactivating,reloading {q(self.cfg.unit_glob)} 2>/dev/null"
                  " | awk '{print \"unit=\" $1}'")
        rc, out = self.ops.tower(script, timeout=60)
        kv = kv_lines(out)
        if rc != 0 and "biz" not in kv:
            kv = {"error": [out[-200:]]}
        self._probe, self._last_probe = kv, self.clock()
        return kv

    def here_free_gb(self) -> float:
        try:
            st = os.statvfs(self.cfg.p7)
            return st.f_bavail * st.f_frsize / 2**30
        except OSError:
            return 1e9

    def own_units(self) -> List[str]:
        return [u["name"] for u in self.state.get("units", []) if u.get("name")]

    def foreign_units(self, kv: Dict[str, List[str]]) -> List[str]:
        mine = set(self.own_units()) | {f"{u}.service" for u in self.own_units()}
        out = []
        for u in kv.get("unit", []):
            if u.startswith("distill-deadman-") or u in mine or u.startswith(self.cfg.prefix):
                continue
            if u.startswith(self.cfg.quant_unit_p6) or u.startswith(self.cfg.recipe_unit):
                continue
            out.append(u)
        return out

    def guard(self, *, force: bool = False) -> None:
        """Raise Stop when GPU work must end now."""
        if self.term:
            raise Stop("stopped (SIGTERM)")
        if deadline_state(self.clock(), self.cfg.stop_at, self.cfg.hard_at) != "run":
            raise Stop(f"the stop time {et(self.cfg.stop_at)} was reached")
        if not self.cfg.execute:
            return
        kv = self.probe(force=force)
        if "error" in kv:
            self.log(f"  probe failed ({kv1(kv, 'error')}); continuing")
            return
        if kv.get("flag"):
            raise Stop(f"NO-SWAPS flag present ({', '.join(kv['flag'])})")
        if self.state.get("gpu_taken") and kv1(kv, "biz") in ACTIVE:
            raise Stop(f"{self.cfg.biz_unit} is {kv1(kv, 'biz')} again (restored elsewhere)")
        free = to_float(kv1(kv, "free_gb"))
        if free is not None and free < DISK_FLOOR_GIB:
            raise Stop(f"disk pressure: the 4090 host has {free:.1f} GiB free (< {DISK_FLOOR_GIB:g})")
        hf = self.here_free_gb()
        if hf < DISK_FLOOR_GIB:
            raise Stop(f"disk pressure: this host has {hf:.1f} GiB free (< {DISK_FLOOR_GIB:g})")

    # -- own units on the 4090 ------------------------------------------------------------------
    def record_unit(self, name: str, kind: str, item: str = "") -> None:
        self.state.setdefault("units", []).append({"name": name, "kind": kind, "item": item,
                                                   "started": now_iso()})
        self.save()

    def stop_unit(self, name: str, *, wait_vram: bool = False) -> None:
        unit = name if name.endswith(".service") else f"{name}.service"
        st = self.unit_state(unit)
        if st in ACTIVE:
            self.log(f"  stopping {unit} ({st})")
            self.ops.tower(f"systemctl --user stop {q(unit)}", mut=True, timeout=180)
        if wait_vram and self.cfg.execute:
            self.wait_vram_free(90)

    def stop_own_units(self) -> None:
        for name in reversed(self.own_units()):
            if self.unit_state(f"{name}.service") in ACTIVE:
                self.stop_unit(name)
        for name in (self.cfg.quant_unit_p6, self.cfg.recipe_unit):
            if self.unit_state(f"{name}.service") in ACTIVE:
                self.stop_unit(name)

    def vram(self) -> Optional[int]:
        return to_int(kv1(self.probe(force=True), "vram"))

    def wait_vram_free(self, seconds: float) -> Tuple[bool, Optional[int]]:
        end = self.clock() + seconds
        used = None
        while True:
            used = self.vram()
            if used is not None and used <= IDLE_VRAM_MAX_MIB:
                return True, used
            if self.clock() >= end or self.term:
                return False, used
            self.sleep(5)

    # -- steps ----------------------------------------------------------------------------------
    def step_wait_train(self) -> None:
        tu, wu = self.cfg.trainer_unit, self.cfg.watcher_unit
        ended_at = self.state.get("trainer_ended_at")
        last_log = 0.0
        while True:
            if self.term:
                raise NotRun("stopped (SIGTERM) while waiting for training")
            st_t, st_w = self.unit_state(tu), self.unit_state(wu)
            now = self.clock()
            if st_t not in ACTIVE and st_t != "unknown":
                if ended_at is None:
                    ended_at = now
                    self.state["trainer_ended_at"] = now
                    self.save()
                    self.log(f"trainer {tu} has ended ({st_t}); watcher {wu} is {st_w}")
                if st_w not in ACTIVE and st_w != "unknown":
                    self.mark("wait-train", "done", f"trainer {st_t}, watcher {st_w}")
                    return
                if now - ended_at > 3600:
                    self.note(f"the watcher {wu} is still {st_w} an hour after the trainer ended; going on "
                              "(the window lock keeps its restore and our stop apart)")
                    self.mark("wait-train", "done", "watcher still running after 60 min")
                    return
            if now >= self.cfg.stop_at:
                raise NotRun(f"training had not ended by {et(self.cfg.stop_at)} (trainer {st_t})")
            if not self.cfg.execute:
                self.log(f"[wait-train] trainer {tu}: {st_t}; watcher {wu}: {st_w}. A real run waits here "
                         f"(poll {self.cfg.poll:.0f} s) until both have ended; the dry run goes on as if they had.")
                return
            if now - last_log >= 1800:
                self.log(f"waiting: trainer {st_t}, watcher {st_w}")
                last_log = now
            self.sleep(self.cfg.poll)

    def step_verdict(self) -> None:
        run_rel = f"distill/phase5/runs/{self.cfg.run}"
        man = None
        for attempt in range(3):  # a read that failed is not a verdict: try again before deciding
            rc, out = self.ops.tower(f"cat {q(self.tp(run_rel + '/manifest.json'))}")
            man = read_json_text(out) if rc == 0 else None
            if isinstance(man, dict) or not self.cfg.execute:
                break
            self.log(f"  the run manifest did not read (rc {rc}); again in 30 s")
            self.sleep(30)
        rc, out = self.ops.tower(f"cat {q(self.tp(run_rel + '/run_state.json'))}")
        rs = read_json_text(out) if rc == 0 else None
        ad = self.tp(f"{run_rel}/{self.cfg.adapter_name}")
        rc, _ = self.ops.tower(f"test -s {q(ad + '/adapter_config.json')} && "
                               f"test -s {q(ad + '/adapter_model.safetensors')}")
        ok, why, info = training_verdict(man, rs, adapter_present=(rc == 0))
        self.state["training"] = {"ok": ok, "why": why, **{k: v for k, v in info.items()}}
        self.state["train_sha256"] = ((man or {}).get("data") or {}).get("sha256")
        self.save()
        if ok:
            self.mark("verdict", "done", why)
            return
        if not self.cfg.execute:
            self.log(f"[verdict] now: NOT complete ({why}). A real run Telegrams this and stops; "
                     "the dry run goes on as if the run had completed.")
            return
        self.notify("not-run", "\n".join([
            f"Post-training pipeline NOT run: the 9B training did not complete ({why[:160]}).",
            f"Run {self.cfg.run}; nothing was quantized or scored; the GPU was not touched.",
            "The training watcher restores the business server as usual."]))
        raise NotRun(why)

    def step_take_gpu(self) -> None:
        kv = self.probe(force=True)
        if kv.get("flag") and self.cfg.execute:
            self.notify("flag", f"Post-training pipeline NOT run: a NO-SWAPS flag is set on the 4090 host "
                                f"({', '.join(kv['flag'])[:150]}). Nothing was quantized or scored.")
            raise NotRun(f"NO-SWAPS flag present ({', '.join(kv['flag'])}): not taking the GPU")
        biz = kv1(kv, "biz")
        free = to_float(kv1(kv, "free_gb"))
        if free is not None and free < DISK_FLOOR_GIB and self.cfg.execute:
            self.notify("disk", f"Post-training pipeline NOT run: the 4090 host has {free:.1f} GiB free "
                                f"(< {DISK_FLOOR_GIB:g}). Nothing was quantized or scored.")
            raise NotRun(f"the 4090 host has only {free:.1f} GiB free")
        eta = self.clock() + 60 * (EST_MIN["phase6"] + sum(EST_MIN[k] for k in
                                                        ("a", "b", "c", "d", "move", "recipe", "e1", "e2")))
        down = "\n".join([
            f"{self.biz_name} going down: post-training quantize + eval.",
            f"Phase 6 (merge, GGUF, Q4_K_M + Q5_K_M), then scoring Q4/Q5/BF16 and a same-recipe base 9B; "
            f"ETA ~{et(min(eta, self.cfg.stop_at))}.",
            f"{self.biz_name} is restored when it ends (by {et(self.cfg.stop_at)} at the latest)."])
        lock = self.tp("distill/window/state/window.lock")
        stop_script = (
            f"exec 9>>{q(lock)}; flock -w 1800 9 || {{ echo lock=timeout; exit 9; }}; "
            f"st=$(systemctl --user is-active {q(self.cfg.biz_unit)}); echo before=$st; "
            f"case $st in active|activating|reloading) systemctl --user stop {q(self.cfg.biz_unit)}"
            " && echo stopped=yes ;; esac; "
            "for i in $(seq 1 18); do st=$(systemctl --user is-active "
            f"{q(self.cfg.biz_unit)}); used=$(nvidia-smi --query-gpu=memory.used "
            "--format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); "
            f"if [ \"$st\" != active ] && [ \"${{used:-99999}}\" -le {IDLE_VRAM_MAX_MIB} ]; then "
            "echo freed=$used; exit 0; fi; sleep 5; done; echo freed=no used=$used; exit 1")
        if not self.cfg.execute:
            self.log(f"[take-gpu] now: {self.cfg.biz_unit} {biz or '?'}, VRAM {kv1(kv, 'vram') or '?'} MiB, "
                     f"flags {kv.get('flag') or 'none'}, 4090 free {kv1(kv, 'free_gb')} GiB. If business is up then:")
            self.notify("down", down)
            self.ops.tower(stop_script, mut=True)
            self.log(f"  else: wait (<= 10 min) for VRAM <= {IDLE_VRAM_MAX_MIB} MiB and no other distill unit; "
                     "if it stays busy: Telegram and stop without touching the GPU")
            self.mark("take-gpu", "planned")
            return
        if biz in ACTIVE:
            # From here on a crash or a stop must restore: marked before the stop, not after.
            self.state["gpu_taken"] = True
            self.state["biz_down_by_us"] = True
            self.save()
            self.notify("down", down)
            rc, out = self.ops.tower(stop_script, timeout=2000)
            if rc != 0:
                raise Stop(f"stopping {self.cfg.biz_unit} did not free the GPU (rc {rc}: {out[-200:]})")
            self.mark("take-gpu", "done", f"{self.cfg.biz_unit} stopped; VRAM {kv1(kv_lines(out), 'freed', '?')} MiB")
            return
        ok, used = self.wait_vram_free(600)
        others = self.foreign_units(self.probe(force=True))
        if not ok or others:
            self.notify("gpu-busy", "\n".join([
                f"Post-training pipeline NOT run: the 4090 GPU is not free ({used} MiB; units {others or 'none'}).",
                f"{self.cfg.biz_unit} is {biz or '?'}; nothing was quantized or scored."]))
            raise NotRun(f"the GPU is not free ({used} MiB used; distill units {others or 'none'})")
        self.state["gpu_taken"] = True
        self.save()
        self.mark("take-gpu", "done", f"{self.cfg.biz_unit} {biz or '?'} already; VRAM {used} MiB; GPU free")

    def run_child(self, argv: Sequence[str], log_path: Path, env: Dict[str, str],
                  on_stop: Callable[[], None]) -> int:
        """A long child here, in its own session, polled with the guards."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log("  $ " + shlex.join(list(argv)) + f"  (log {log_path.name})")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {now_iso()} {shlex.join(list(argv))}\n")
            fh.flush()
            proc = subprocess.Popen(list(argv), stdout=fh, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
        self.state["child_pid"] = proc.pid
        self.save()
        try:
            while proc.poll() is None:
                try:
                    self.guard()
                except Stop:
                    self.log(f"  stopping the child (pid {proc.pid})")
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    on_stop()
                    raise
                self.sleep(15)
            return proc.returncode
        finally:
            self.state.pop("child_pid", None)
            self.save()

    def step_phase6(self) -> None:
        cfg = self.cfg
        adapter_rel = f"distill/phase5/runs/{cfg.run}/{cfg.adapter_name}"
        env = dict(os.environ, TOWER=cfg.tower or "fake")
        if cfg.fake:
            env["FAKE_ROOT"] = str(cfg.fake)
        if not cfg.execute:
            dry = cfg.dry_adapter or adapter_rel
            argv = cfg.phase6_cmd + [adapter_rel, "--execute", "--keep-bf16", "--stop-at", cfg.stop_at_iso]
            self.log("  [plan] here: TOWER=<4090> " + shlex.join(argv))
            plan = cfg.phase6_cmd + [dry, "--keep-bf16", "--stop-at", cfg.stop_at_iso]
            self.log(f"  phase6_student.sh plan mode on {dry}:")
            rc, out = self.ops.here(plan, env=env, timeout=1800)
            for ln in out.splitlines()[-60:]:
                self.log("    | " + ln)
            self.mark("phase6", "planned", f"plan mode exit {rc}")
            return
        if not may_start(self.clock(), cfg.stop_at, EST_MIN["phase6"]):
            raise Stop(f"less than {EST_MIN['phase6']} min before the stop time: no Phase 6")
        log_path = cfg.p7 / "logs" / "phase6.log"
        failures = int(self.state.get("phase6_failures", 0))
        pauses = 0
        while True:
            self.guard(force=True)
            extra = PHASE6_ATTEMPTS[min(failures, len(PHASE6_ATTEMPTS) - 1)]
            argv = cfg.phase6_cmd + [adapter_rel, "--execute", "--keep-bf16", "--stop-at", cfg.stop_at_iso,
                                     *extra]
            rc = self.run_child(argv, log_path, env,
                                on_stop=lambda: self.stop_unit(cfg.quant_unit_p6, wait_vram=True))
            if rc == 0:
                break
            tail = self.tail(log_path, 3)
            if rc in (75, 99):
                pauses += 1
                if pauses > 12:
                    raise Stop(f"Phase 6 paused {pauses} times (exit {rc}): {tail}")
                self.log(f"  phase6 paused (exit {rc}: {tail}); resuming in 5 min")
                self.sleep(300)
                continue
            failures += 1
            self.state["phase6_failures"] = failures
            self.save()
            if failures >= len(PHASE6_ATTEMPTS):
                self.notify("phase6-failed", "\n".join([
                    f"Post-training: Phase 6 FAILED {failures} times (exit {rc}).",
                    tail[:300], f"Nothing scored; {self.biz_name} is being restored."]))
                raise Stop(f"Phase 6 failed {failures} times (exit {rc}): {tail}")
            nxt = PHASE6_ATTEMPTS[failures]
            self.log(f"  phase6 exit {rc} ({tail}); trying again{' with ' + ' '.join(nxt) if nxt else ''} "
                     "(it resumes)")
            self.sleep(60)
        man = read_json_text((cfg.hd / "phase6" / cfg.run / "quantize-manifest.json").read_text(encoding="utf-8")) \
            if (cfg.hd / "phase6" / cfg.run / "quantize-manifest.json").is_file() else None
        self.state["phase6"] = phase6_facts(man)
        missing = []
        for m in ("q4", "q5", "bf16"):
            rc, _ = self.ops.tower(f"test -s {q(self.gguf(m))}")
            if rc != 0:
                missing.append(m)
        if missing:
            raise Stop(f"Phase 6 ended without these GGUFs on the 4090 host: {missing}")
        self.mark("phase6", "done", f"quantize result {self.state['phase6'].get('result')}")

    def tail(self, path: Path, n: int) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return " | ".join(ln.strip() for ln in lines[-n:])[:400]

    def step_subset(self) -> None:
        cfg = self.cfg
        src = self.cases_here("eval-tools")
        rc, out = self.ops.tower(f"sha256sum {q(self.cases_t('eval-tools'))}")
        tsha = (out.split() or [""])[0] if rc == 0 else ""
        hsha = sha256_file(src)
        if tsha != hsha:
            raise Stop(f"eval-tools differs between the hosts (here {hsha[:12]}, 4090 {tsha[:12] or 'missing'})")
        cases = read_jsonl(src)
        sub = select_subset(cases, cfg.subset_n, SUBSET_SEED, source_sha256=hsha)
        if cfg.ids_file.is_file():
            ref = read_json_text(cfg.ids_file.read_text(encoding="utf-8")) or {}
            if ref.get("ids") != sub["ids"] or ref.get("source_sha256") != hsha:
                raise Stop(f"the subset differs from the committed {cfg.ids_file.name}")
        elif not cfg.fake:
            self.note(f"no committed {cfg.ids_file.name}: the subset is recorded only beside the state")
        d = cfg.p7 / "subset"
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / f"eval-tools-subset{cfg.subset_n}.ids.json", sub)
        keep = set(sub["ids"])
        lines = []
        with open(src, "rb") as fh:
            for raw in fh.read().split(b"\n"):
                if raw.strip() and json.loads(raw).get("id") in keep:
                    lines.append(raw)
        jl = d / f"eval-tools-subset{cfg.subset_n}.jsonl"
        jl.write_bytes(b"\n".join(lines) + b"\n")
        self.state["subset"] = {"n": sub["n"], "seed": sub["seed"], "sha256": sha256_file(jl),
                                "strata": sub["strata"]}
        self.save()
        self.ops.tower(f"mkdir -p -m 700 {q(self.t_p7 + '/subset')}", mut=True)
        rc, out = self.ops.rsync([str(jl)], self.ops.remote(f"distill/phase7/{cfg.run}/subset/"))
        if cfg.execute:
            rc2, out2 = self.ops.tower(f"sha256sum {q(self.cases_t('eval-tools-subset'))}")
            if rc != 0 or (out2.split() or [""])[0] != self.state["subset"]["sha256"]:
                raise Stop(f"pushing the subset failed: {out[-200:]} {out2[-200:]}")
        self.mark("subset", "done" if cfg.execute else "planned",
                  f"{sub['n']} ids, strata {json.dumps({k: v['chosen'] for k, v in sub['strata'].items()})}")

    # -- scoring ----------------------------------------------------------------------------------
    def server_argv(self, model: str, ctx: int) -> List[str]:
        return [self.tp("llama.cpp/build/bin/llama-server"), "-m", self.gguf(model), "-c", str(ctx),
                "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "-ngl", "99",
                "-fit", "off", "-b", "2048", "-ub", "2048", "--host", LOOPBACK, "--port", str(self.cfg.port),
                "--alias", self.cfg.labels[model]]

    def start_server(self, key: str, model: str, ctx: int) -> Tuple[str, str]:
        """('ok' | 'load-failed' | 'error', detail). One tenant: the GPU must be free first."""
        it = self.state["items"][key]
        n = it["starts"] = it.get("starts", 0) + 1
        name = f"{self.cfg.prefix}srv-{key}-c{ctx}-{n}"
        logf = f"{self.t_p7}/logs/{name}.log"
        argv = self.server_argv(model, ctx)
        if self.cfg.execute:
            ok, used = self.wait_vram_free(300)
            if not ok:
                return "error", f"the GPU holds {used} MiB before the server start"
            rc, out = self.ops.tower(f"ss -ltn 2>/dev/null | grep -q ':{self.cfg.port} ' && echo taken || echo free")
            if "taken" in out:
                return "error", f"port {self.cfg.port} is taken"
        script = (f"mkdir -p -m 700 {q(self.t_p7 + '/logs')} && systemd-run --user --quiet --collect "
                  f"--unit={q(name)} --description={q('distill phase 7 scoring server ' + key)} "
                  "-p KillSignal=SIGTERM -p TimeoutStopSec=60 -p LimitCORE=0 "
                  f"-p StandardOutput=append:{q(logf)} -p StandardError=append:{q(logf)} -- "
                  + shlex.join(argv))
        self.record_unit(name, "server", key)
        it["server_unit"] = name
        self.save()
        rc, out = self.ops.tower(script, mut=True)
        if not self.cfg.execute:
            return "ok", "planned"
        if rc != 0:
            return "error", f"systemd-run exited {rc}: {out[-200:]}"
        t0 = self.clock()
        health = (f"python3 -c {q(_HEALTH_PY)} {self.cfg.port}")
        while True:
            self.guard()
            st = self.unit_state(f"{name}.service")
            if st == "unknown":
                self.sleep(10)
                continue
            if st not in ACTIVE:
                rc, out = self.ops.tower(f"tail -n 4 {q(logf)}")
                return "load-failed", f"exited while loading at -c {ctx}: {out[-300:]}"
            rc, out = self.ops.tower(health, timeout=30)
            kv = kv_lines(out)
            if kv1(kv, "health") == "200":
                n_ctx = to_int(kv1(kv, "n_ctx"))
                if n_ctx != ctx:
                    self.stop_unit(name, wait_vram=True)
                    return "error", f"the server reports n_ctx {n_ctx}, asked for {ctx}"
                used = self.vram()
                it["ctx"], it["model_id"], it["vram_mib"] = ctx, kv1(kv, "model"), used
                self.save()
                return "ok", f"healthy in {self.clock() - t0:.0f}s at -c {ctx}, VRAM {used} MiB"
            if self.clock() - t0 > 900:
                self.stop_unit(name, wait_vram=True)
                return "error", "not healthy after 15 min"
            self.sleep(5)

    def run_set(self, key: str, model: str, sname: str) -> str:
        """'done' | 'errors' | 'refused' | 'crash' | 'failed'. score.py resumes finished cases."""
        it = self.state["items"][key]
        label = self.cfg.labels[model]
        rdir = f"{self.t_p7}/results/{label}"
        out = f"{rdir}/{sname}.jsonl"
        n = it.setdefault("score_runs", 0) + 1
        it["score_runs"] = n
        name = f"{self.cfg.prefix}score-{key}-{sname}-{n}"
        rcf = f"{rdir}/.{sname}.rc"
        logf = f"{self.t_p7}/logs/{label}-{sname}.run.log"
        cmd = [self.tp("distill/venv-score/bin/python"), self.tp("distill/code/scripts/distill/score.py"), "run",
               "--base-url", f"http://{LOOPBACK}:{self.cfg.port}", "--label", label,
               "--cases", self.cases_t(sname), "--out", out, "--concurrency", "1",
               "--timeout", str(SET_TIMEOUT[sname])]
        inner = shlex.join(cmd) + f"; echo $? > {q(rcf)}"
        script = (f"mkdir -p -m 700 {q(rdir)} {q(self.t_p7 + '/logs')} && rm -f {q(rcf)} && "
                  f"systemd-run --user --quiet --collect --unit={q(name)} "
                  f"--description={q('distill phase 7 score.py ' + key + ' ' + sname)} "
                  f"-p WorkingDirectory={q(rdir)} -p LimitCORE=0 -p KillSignal=SIGTERM -p TimeoutStopSec=60 "
                  f"-p StandardOutput=append:{q(logf)} -p StandardError=append:{q(logf)} "
                  f"--setenv=PYTHONPATH={q(self.tp('distill/code/src'))} -- /bin/bash -c {q(inner)}")
        self.record_unit(name, "score", key)
        rc, o = self.ops.tower(script, mut=True)
        if not self.cfg.execute:
            return "done"
        if rc != 0:
            return "failed"
        srv = it.get("server_unit", "")
        last = 0.0
        while True:
            try:
                self.guard()
            except Stop:
                self.stop_unit(name)
                raise
            st = self.unit_state(f"{name}.service")
            if st == "unknown":
                self.sleep(20)
                continue
            if st not in ACTIVE:
                break
            if self.unit_state(f"{srv}.service") not in ACTIVE + ("unknown",):
                self.log(f"  the server {srv} died during {key}/{sname}")
                self.stop_unit(name)
                return "crash"
            if self.clock() - last > 900:
                rc, o = self.ops.tower(f"if [ -f {q(out)} ]; then wc -l < {q(out)}; else echo 0; fi")
                self.log(f"  {key}/{sname}: {o.strip()} rows written")
                last = self.clock()
            self.sleep(20)
        rc, o = self.ops.tower(f"cat {q(rcf)} 2>/dev/null")
        code = to_int(o.strip().splitlines()[-1] if o.strip() else "", None)
        if self.unit_state(f"{srv}.service") not in ACTIVE + ("unknown",):
            return "crash"
        if code == 0:
            return "done"
        if code == 3:
            return "errors"
        if code == 2:
            return "refused"
        return "failed"

    def score_item(self, key: str, model: str, sets: Sequence[str]) -> str:
        """Score one item; returns its final status ('done' | 'partial' | 'skipped')."""
        it = self.state["items"].setdefault(key, {"status": "pending", "sets": {}, "crashes": 0})
        if it.get("status") in ("done", "skipped"):
            return it["status"]
        it["status"] = "running"
        self.save()
        ladder: Sequence[int] = CTX_LADDER[model]
        if it.get("ctx") and it["ctx"] in ladder:
            ladder = tuple(c for c in ladder if c <= it["ctx"])
        while True:
            self.guard(force=True)
            # the server: the first context of the ladder that loads (only the BF16 steps down)
            started = None
            for ctx in ladder:
                if ctx in it.get("ctx_failed", []):
                    continue
                res, detail = self.start_server(key, model, ctx)
                self.log(f"  [{key}] server {self.cfg.labels[model]} -c {ctx}: {res}: {detail}")
                if res == "ok":
                    started = ctx
                    break
                self.stop_unit(it.get("server_unit", ""), wait_vram=True)
                if res == "load-failed" and len(ladder) > 1:
                    it.setdefault("ctx_failed", []).append(ctx)
                    self.save()
                    continue
                break
            if started is None:
                it["crashes"] = it.get("crashes", 0) + 1
                self.save()
                if it["crashes"] > 1:
                    return self.skip_item(key, "the server would not start twice")
                continue
            crashed = False
            for sname in sets:
                if (it["sets"].get(sname) or "") in ("done", "done-with-errors"):
                    continue
                for p in range(SET_PASSES):
                    res = self.run_set(key, model, sname)
                    self.log(f"  [{key}] {sname}: {res}")
                    if res == "done":
                        it["sets"][sname] = "done"
                        break
                    if res == "errors":
                        it["sets"][sname] = "done-with-errors" if p == SET_PASSES - 1 else "errors"
                        continue
                    if res == "crash":
                        crashed = True
                        break
                    it["sets"][sname] = res
                    break
                self.save()
                if crashed or it["sets"].get(sname) in ("refused", "failed"):
                    break
            self.stop_unit(it.get("server_unit", ""), wait_vram=True)
            if crashed:
                it["crashes"] = it.get("crashes", 0) + 1
                if len(ladder) > 1 and started != ladder[-1]:
                    it.setdefault("ctx_failed", []).append(started)  # the restart steps down
                self.save()
                if it["crashes"] > 1:
                    return self.skip_item(key, "the server crashed twice")
                self.log(f"  [{key}] server crash: restarting once and resuming")
                continue
            bad = [s for s in sets if it["sets"].get(s) not in ("done", "done-with-errors")]
            if bad:
                return self.skip_item(key, f"sets not finished: {bad} ({[it['sets'].get(s) for s in bad]})")
            it["status"] = "done"
            self.save()
            return "done"

    def skip_item(self, key: str, why: str) -> str:
        it = self.state["items"][key]
        it["status"], it["why"] = "skipped", why
        self.note(f"item {key} skipped: {why}")
        return "skipped"

    def step_item(self, key: str, model: str, sets: Sequence[str], optional: bool) -> None:
        cfg = self.cfg
        need = EST_MIN[key] * (1.2 if optional else 0.0) + 15
        if not may_start(self.clock(), cfg.stop_at, need):
            self.state["items"].setdefault(key, {"sets": {}})["status"] = "skipped"
            self.state["items"][key]["why"] = f"not enough time before {et(cfg.stop_at)}"
            self.mark(f"score-{key}", "skipped", self.state["items"][key]["why"])
            return
        if model == "recipe" and (self.state["steps"].get("recipe") or {}).get("status") != "done" \
                and cfg.execute:
            self.state["items"].setdefault(key, {"sets": {}})["status"] = "skipped"
            self.mark(f"score-{key}", "skipped", "no same-recipe base GGUF")
            return
        if not cfg.execute:
            self.state["items"].setdefault(key, {"status": "pending", "sets": {}, "crashes": 0})
            ladder = CTX_LADDER[model]
            self.log(f"[score-{key}] {cfg.labels[model]}: {', '.join(sets)}; server -c "
                     f"{'/'.join(map(str, ladder))} (first that loads)")
            self.start_server(key, model, ladder[0])
            for s in sets:
                self.run_set(key, model, s)
            self.stop_unit(self.state["items"][key].get("server_unit", "x"))
            self.log(f"  [plan] then: systemctl --user stop {self.state['items'][key].get('server_unit')}"
                     f".service; wait for VRAM <= {IDLE_VRAM_MAX_MIB} MiB")
            self.state["items"][key]["status"] = "pending"
            return
        status = self.score_item(key, model, sets)
        self.mark(f"score-{key}", status if status != "skipped" else "skipped",
                  json.dumps(self.state["items"][key].get("sets")))

    # -- (e) the official base through the student's recipe ---------------------------------------
    def step_recipe(self) -> None:
        cfg = self.cfg
        if cfg.skip_recipe:
            self.mark("recipe", "skipped", "--skip-recipe")
            return
        need = EST_MIN["move"] + EST_MIN["recipe"] + EST_MIN["e1"] + 10
        if not may_start(self.clock(), cfg.stop_at, need):
            self.mark("recipe", "skipped", f"not enough time before {et(cfg.stop_at)} ({need} min)")
            return
        rb = self.cfg.hd / "phase6" / cfg.run / "gguf" / f"{cfg.run}-BF16.gguf"
        man_here = self.cfg.hd / "phase6" / cfg.run / "quantize-manifest.json"
        want = bf16_sha(read_json_text(man_here.read_text(encoding="utf-8")) if man_here.is_file() else None)
        # 1. the student's BF16 GGUF moves here (kept), freeing the 4090's disk
        rc, _ = self.ops.tower(f"test -e {q(self.gguf('bf16'))}")
        if rc == 0 or not cfg.execute:
            self.guard(force=True)
            rb.parent.mkdir(parents=True, exist_ok=True)
            rc, out = self.ops.rsync([self.ops.remote(f"distill/phase6/{cfg.run}/gguf/{cfg.run}-BF16.gguf")],
                                     str(rb.parent) + "/")
            if cfg.execute:
                have = sha256_file(rb) if rb.is_file() else ""
                if rc != 0 or not want or have != want:
                    self.mark("recipe", "skipped", f"moving the BF16 GGUF failed or its sha256 differs "
                                                   f"(rc {rc}, {have[:12]} vs {str(want)[:12]})")
                    return
            self.ops.tower(f"rm -f {q(self.gguf('bf16'))}", mut=True)
            self.state["bf16_kept_here"] = True
            self.save()
        # 2. room for it: the official bf16 (~19 GB) + its BF16 GGUF (~18 GB) + Q4_K_M (~6 GB)
        kv = self.probe(force=True)
        free = to_float(kv1(kv, "free_gb"))
        if cfg.execute and (free is None or free < 45 + DISK_FLOOR_GIB):
            self.mark("recipe", "skipped", f"the 4090 host has {free} GiB free (needs {45 + DISK_FLOOR_GIB:g})")
            return
        # 3. the official bf16 there, sha256-checked
        official = self.cfg.hd / "models" / "qwen3.5-9b-official"
        tr = self.t_recipe
        self.ops.tower(f"mkdir -p -m 700 {q(tr + '/base')} {q(tr + '/nodata')} {q(tr + '/logs')}", mut=True)
        rc, out = self.ops.rsync([str(official) + "/"], self.ops.remote(f"distill/phase6/{RECIPE_RUN}/base/"))
        sums = cfg.p7 / "logs" / "official.sha256"
        if cfg.execute:
            if rc != 0:
                self.mark("recipe", "skipped", f"pushing the official weights failed: {out[-200:]}")
                return
            files = sorted(p.name for p in official.iterdir() if p.is_file())
            sums.parent.mkdir(parents=True, exist_ok=True)
            sums.write_text("".join(f"{sha256_file(official / f)}  {f}\n" for f in files), encoding="utf-8")
            rc, out = self.ops.tower(f"cd {q(tr + '/base')} && sha256sum -c --quiet",
                                     stdin=sums.read_text(encoding="utf-8"), timeout=1800)
            if rc != 0:
                self.mark("recipe", "skipped", f"the official weights' sha256 differ on the 4090 host: {out[-200:]}")
                return
        # 4. the same calibration text, then quantize.py --no-adapter (Q4_K_M only)
        self.ops.tower(f"[ -e {q(tr + '/calib/calibration.json')} ] || cp -a {q(self.t_p6 + '/calib')} {q(tr + '/calib')}",
                       mut=True)
        tq = self.tp("distill/phase6-rehearsal/qvenv/bin/python")
        qcmd = [tq, "-I", "-u", f"{self.t_p6}/code/quantize.py", "run", "--no-adapter",
                "--base", f"{tr}/base", "--label", RECIPE_RUN, "--out", tr, "--data", f"{tr}/nodata",
                "--evalsets=", "--train-sha256", str(self.state.get("train_sha256") or "x"),
                "--template-dir", self.tp("distill/models/qwen3.5-9b-tokenizer"),
                "--llama-cpp-dir", self.tp("distill/phase6-rehearsal/llama.cpp-9d57ce456-cuda"),
                "--convert-python", tq, "--ngl", "99", "--quants", "Q4_K_M", "--stop-at", cfg.stop_at_iso,
                "--i-have-operator-approval-for-max-derived-data", "--execute"]
        unit, rcf, logf = cfg.recipe_unit, f"{tr}/logs/{cfg.recipe_unit}.rc", f"{tr}/logs/{cfg.recipe_unit}.log"
        if self.unit_state(f"{unit}.service") not in ACTIVE:
            ok, used = self.wait_vram_free(300) if cfg.execute else (True, 0)
            if not ok:
                self.mark("recipe", "skipped", f"the GPU holds {used} MiB")
                return
            script = (f"rm -f {q(rcf)} && systemd-run --user --quiet --collect --unit={q(unit)} "
                      f"--working-directory={q(tr)} -p KillSignal=SIGTERM -p TimeoutStopSec=90 -p Nice=5 "
                      f"-p MemoryMax=40G -p LimitCORE=0 -p StandardOutput=append:{q(logf)} "
                      f"-p StandardError=append:{q(logf)} -- /bin/bash -c "
                      + q(shlex.join(qcmd) + f"; echo $? > {q(rcf)}"))
            self.record_unit(unit, "quantize", "e")
            self.ops.tower(script, mut=True)
        if cfg.execute:
            while self.unit_state(f"{unit}.service") in ACTIVE + ("unknown",):
                try:
                    self.guard()
                except Stop:
                    self.stop_unit(unit, wait_vram=True)
                    raise
                self.sleep(30)
            rc, out = self.ops.tower(f"cat {q(rcf)} 2>/dev/null || echo stopped")
            code = out.strip().splitlines()[-1] if out.strip() else "stopped"
            rc2, mo = self.ops.tower(f"cat {q(tr + '/quantize-manifest.json')}")
            self.state["recipe_quantize"] = phase6_facts(read_json_text(mo) if rc2 == 0 else None)
            if code != "0":
                self.mark("recipe", "skipped", f"quantize.py exit {code}: " + self.ops.tower(f"tail -n 3 {q(logf)}")[1][-300:])
                return
        # 5. the 16-bit intermediates go; the Q4_K_M stays
        self.ops.tower(f"rm -rf {q(tr + '/base')} {q(tr + '/base-view')} {q(tr + '/gguf/' + RECIPE_RUN + '-BF16.gguf')}",
                       mut=True)
        self.mark("recipe", "done" if cfg.execute else "planned", "same-recipe base Q4_K_M ready")

    # -- release + report -------------------------------------------------------------------------
    def release(self, why: str) -> None:
        if self.cfg.execute and (self.state.get("released") or not self.state.get("gpu_taken")):
            return
        self.log(f"[release] {why}")
        self.stop_own_units()
        msg, rcf = f"{self.t_p7}/restore.msg", f"{self.t_p7}/restore.rc"
        line = f"Post-training pipeline {why[:200]}; " + (
            "stopped by hand: arm it again to resume." if self.term else "results Telegram follows.")
        self.ops.tower(f"mkdir -p {q(self.t_p7)} && rm -f {q(rcf)} && printf '%s\\n' {q(line)} > {q(msg)}",
                       mut=True)
        unit = f"{self.cfg.restore_prefix}{datetime.now(ET).strftime('%Y%m%dT%H%M%S')}"
        inner = (f"/bin/bash {q(self.tp('distill/window/train_end.sh'))} --msg-file {q(msg)}; "
                 f"echo $? > {q(rcf)}")
        script = (f"systemd-run --user --quiet --collect --unit={q(unit)} -p KillMode=mixed -p TimeoutStopSec=1800 "
                  f"--description={q('flywheel: restore business after the post-training pipeline')} "
                  f"--expand-environment=no -- /bin/bash -c {q(inner)}")
        rc, out = self.ops.tower(script, mut=True)
        if not self.cfg.execute:
            self.mark("release", "planned", "train_end.sh restore")
            return
        self.state["restore_unit"] = unit
        self.save()
        t0 = self.clock()
        while self.unit_state(f"{unit}.service") in ACTIVE + ("unknown",) and self.clock() - t0 < 2700:
            self._sleep(10)
        rc, out = self.ops.tower(f"cat {q(rcf)} 2>/dev/null")
        code = (out.strip().splitlines() or ["none"])[-1]
        rc, last = self.ops.tower("cat ~/distill/window/state/verify.last 2>/dev/null")
        verify = (last.strip().splitlines() or [""])[-1]
        ok = code == "0"
        res = f"train_end.sh exit {code}; verify: {verify}"
        self.state["restore"] = {"unit": unit, "train_end": res[:300], "ok": ok}
        self.state["released"] = True
        self.mark("release", "done" if ok else "failed", res[:300])
        if not ok:
            self.notify("restore-check", "\n".join([
                f"Post-training: the {self.biz_name} restore did not report OK ({res[:120]}).",
                "The window end (Mon 05:40 EST) and the dead-man (06:00 EST) retry it."]))

    def step_report(self) -> None:
        cfg = self.cfg
        res_here = cfg.p7 / "results"
        res_here.mkdir(parents=True, exist_ok=True)
        rc, out = self.ops.tower(f"test -d {q(self.t_p7 + '/results')}")
        if rc != 0 and cfg.execute:
            self.mark("report", "skipped", "no results on the 4090 host")
            return
        self.ops.rsync([self.ops.remote(f"distill/phase7/{cfg.run}/results/")], str(res_here) + "/")
        summ = cfg.p7 / "summaries"
        summ.mkdir(parents=True, exist_ok=True)
        self.ops.rsync([self.ops.remote(BASE9B_SUMMARY_REL)], str(summ / "base-9b.summary.json"))
        if not cfg.execute:
            self.log("  [plan] here: summarize each model with score.py (this checkout), compare against "
                     "base-9b and the same-recipe base (Saluki alongside), write the report, Telegram")
            self.mark("report", "planned")
            return
        sal = cfg.here_home / SALUKI_SUMMARY_REL
        summaries: Dict[str, Dict[str, Any]] = {}
        for model, label in cfg.labels.items():
            d = res_here / label
            sets = [p.stem for p in sorted(d.glob("*.jsonl"))] if d.is_dir() else []
            if not sets:
                continue
            allf = res_here / f"{label}.all.jsonl"
            with open(allf, "wb") as fh:
                for s in sets:
                    data = (d / f"{s}.jsonl").read_bytes()
                    fh.write(data if data.endswith(b"\n") or not data else data + b"\n")
            argv = [cfg.here_py, str(cfg.code / "scripts" / "distill" / "score.py"), "summarize",
                    "--results", str(allf), "--label", label,
                    "--summary-out", str(summ / f"{label}.summary.json"),
                    "--markdown-out", str(summ / f"{label}.summary.md")]
            for s in sets:
                argv += ["--cases", str(self.cases_here(s))]
            rc, out = self.ops.here(argv, env=dict(os.environ, PYTHONPATH=str(cfg.code / "src")), timeout=1800)
            if rc != 0:
                self.note(f"summarize {label} exited {rc}: {out[-300:]}")
                continue
            summaries[model] = json.loads((summ / f"{label}.summary.json").read_text(encoding="utf-8"))
        base9b = read_json_text((summ / "base-9b.summary.json").read_text(encoding="utf-8")) \
            if (summ / "base-9b.summary.json").is_file() else None
        saluki = read_json_text(sal.read_text(encoding="utf-8")) if sal.is_file() else None
        gates: Dict[str, Dict[str, Any]] = {}
        gdir = cfg.p7 / "gates"
        gdir.mkdir(parents=True, exist_ok=True)
        base_files = {"base-9b": summ / "base-9b.summary.json",
                      "recipe": summ / f"{cfg.labels['recipe']}.summary.json"}
        for cand in ("q4", "q5"):
            if cand not in summaries:
                continue
            for bname, bfile in base_files.items():
                if not bfile.is_file():
                    continue
                also = [str(sal)] if sal.is_file() else []
                for other in ("q5", "q4", "recipe"):
                    f = summ / f"{cfg.labels[other]}.summary.json"
                    if other != cand and other in summaries and f != bfile:
                        also.append(str(f))
                if bname == "recipe" and base_files["base-9b"].is_file():
                    also.append(str(base_files["base-9b"]))
                name = f"{cand}-vs-{bname}"
                argv = [cfg.here_py, str(cfg.code / "scripts" / "distill" / "score.py"), "compare",
                        "--base", str(bfile), "--candidate", str(summ / f"{cfg.labels[cand]}.summary.json"),
                        "--markdown-out", str(gdir / f"{name}.md"), "--json-out", str(gdir / f"{name}.json")]
                for a_ in also:
                    argv += ["--also", a_]
                rc, out = self.ops.here(argv, env=dict(os.environ, PYTHONPATH=str(cfg.code / "src")), timeout=600)
                gj = read_json_text((gdir / f"{name}.json").read_text(encoding="utf-8")) \
                    if (gdir / f"{name}.json").is_file() else None
                gates[name] = {"exit": rc, "verdict": (gj or {}).get("verdict"),
                               "md": (gdir / f"{name}.md").read_text(encoding="utf-8")
                               if (gdir / f"{name}.md").is_file() else out[-500:]}
        sub = read_json_text((cfg.p7 / "subset" / f"eval-tools-subset{cfg.subset_n}.ids.json").read_text(encoding="utf-8")) \
            if (cfg.p7 / "subset" / f"eval-tools-subset{cfg.subset_n}.ids.json").is_file() else None
        data = {"run": cfg.run, "labels": cfg.labels, "summaries": summaries, "base9b": base9b, "saluki": saluki,
                "subset": sub, "state": self.state, "gates": gates, "generated": now_iso(),
                "server_flags": " ".join(self.server_argv("q4", CTX_FULL)[3:]).replace(self.gguf("q4"), "<gguf>")}
        text = scrub(render_report(data))
        if cfg.tower:
            text = text.replace(cfg.tower, "<4090 host>")
            for part in cfg.tower.split("@"):
                if len(part) >= 4:
                    text = text.replace(part, "<4090 host>")
        problems = scrub_problems(text, [self.biz_name] if self.biz_name != "business" else [])
        rep = cfg.p7 / "report" / "DISTILL-PHASE7-SCORES.md"
        rep.parent.mkdir(parents=True, exist_ok=True)
        if problems:
            self.note(f"the report failed its scrub check ({problems}); written as .unscrubbed, not for commit")
            rep = rep.with_suffix(".unscrubbed.md")
        rep.write_text(text, encoding="utf-8")
        self.state["report"] = str(rep)
        self.state["headline"] = headline_lines(data)
        self.save()
        self.notify("results", "\n".join(self.state["headline"]))
        self.mark("report", "done", str(rep))

    def finish(self, outcome: str) -> None:
        self.state["outcome"] = outcome
        self.state["finished"] = now_iso()
        self.save()
        if self.cfg.execute and not self.term:
            # no marker for a stop by hand (SIGTERM): arming again resumes
            write_json(self.cfg.p7 / "PIPELINE-DONE.json", {
                "outcome": outcome, "finished": self.state["finished"], "run": self.cfg.run,
                "report": self.state.get("report"), "restore": self.state.get("restore"),
                "steps": {k: v.get("status") for k, v in self.state["steps"].items()},
                "items": {k: v.get("status") for k, v in self.state["items"].items()},
                "notes": self.state.get("notes", [])[-12:]})
        self.log(f"pipeline finished: {outcome}" + (" (stopped by hand: no marker; arm again to resume)"
                                                     if self.term else ""))

    def resume_after_release(self) -> None:
        """An earlier start released the GPU (stopped by hand) without finishing: take it again."""
        if not self.state.get("released"):
            return
        self.log("an earlier start released the GPU before finishing: resuming (the GPU is taken again)")
        for k in ("released", "gpu_taken", "biz_down_by_us", "restore", "outcome", "finished"):
            self.state.pop(k, None)
        for k in ("take-gpu", "release", "report", "abort"):
            self.state["steps"].pop(k, None)
        for k in ("down", "results", "stop", "end"):
            self.state["telegrams"].pop(k, None)
        for it in self.state.get("items", {}).values():
            if it.get("status") == "running":
                it["status"] = "pending"
        self.save()

    # -- main -----------------------------------------------------------------------------------
    def read_biz_name(self) -> None:
        rc, out = self.ops.tower("sed -n 's/^BIZ_NAME=\\([^ ]*\\).*/\\1/p' ~/distill/window/window.conf 2>/dev/null")
        name = (out.strip().splitlines() or [""])[-1].strip()
        if rc == 0 and re.fullmatch(r"[A-Za-z0-9 ._-]{1,40}", name or ""):
            self.biz_name = name

    def run(self) -> int:
        cfg = self.cfg
        if cfg.execute:
            self.state["starts"] = int(self.state.get("starts", 0)) + 1
            self.save()
        self.log(f"post_train {'EXECUTE' if cfg.execute else 'DRY RUN'} run {cfg.run}; stop at {et(cfg.stop_at)}, "
                 f"hard {et(cfg.hard_at)}; state {self.state_path}; start #{self.state.get('starts')}")
        if PIPELINE_DONE(cfg) and cfg.execute:
            self.log("PIPELINE-DONE.json exists: nothing to do (remove it to run again)")
            return 0
        self.read_biz_name()
        if cfg.execute and self.state["starts"] > MAX_STARTS:
            self.notify("too-many-starts", f"Post-training pipeline: {MAX_STARTS} restarts; giving up. "
                                           f"{self.biz_name} is being restored.")
            self.release("gave up after repeated crashes")
            self.finish("gave-up")
            return 0
        if cfg.execute:
            self.resume_after_release()
            for u in self.state.get("units", []):  # a crashed earlier start may have left one running
                if u.get("kind") in ("server", "score") and self.unit_state(f"{u['name']}.service") in ACTIVE:
                    self.log(f"leftover unit from an earlier start: {u['name']}")
                    self.stop_unit(u["name"], wait_vram=True)
        steps: List[Tuple[str, Callable[[], None]]] = [
            ("wait-train", self.step_wait_train), ("verdict", self.step_verdict),
            ("take-gpu", self.step_take_gpu), ("phase6", self.step_phase6), ("subset", self.step_subset)]
        for key, model, sets, optional in ITEMS:
            if key == "e1":
                steps.append(("recipe", self.step_recipe))
            steps.append((f"score-{key}", (lambda k=key, m=model, s=sets, o=optional: self.step_item(k, m, s, o))))
        outcome, why = "completed", "finished"
        try:
            for name, fn in steps:
                if self.step_done(name) and cfg.execute:
                    continue
                if name.startswith("score-") or name == "recipe":
                    self.guard(force=True)
                fn()
            skipped = [k for k, it in self.state["items"].items() if it.get("status") != "done"]
            if skipped and cfg.execute:
                outcome = f"completed; items not done: {', '.join(sorted(skipped))}"
        except NotRun as exc:
            self.mark("abort", "not-run", str(exc))
            self.finish(f"not-run: {exc}")
            return 0
        except Stop as exc:
            outcome, why = f"stopped: {exc}", f"stopped early ({exc})"
            self.mark("abort", "stopped", str(exc))
            if "disk pressure" in str(exc) or "NO-SWAPS" in str(exc):
                self.notify("stop", f"Post-training pipeline stopped: {str(exc)[:200]}.\n"
                                    f"{self.biz_name} is being restored; partial results are kept.")
        self.release(why)
        try:
            if any(self.state["items"].get(k, {}).get("sets") for k, *_ in ITEMS) or not cfg.execute:
                self.step_report()
        except Exception as exc:  # noqa: BLE001 - the report must never block the marker
            self.note(f"report failed: {type(exc).__name__}: {exc}")
            self.log(traceback.format_exc())
        if "results" not in self.state["telegrams"] and cfg.execute and not self.term:
            done = [k for k, it in self.state["items"].items() if it.get("status") == "done"]
            self.notify("end", "\n".join([
                f"Post-training pipeline ended: {outcome[:200]}.",
                f"Scored items: {', '.join(done) or 'none'}; state on the 3090 Ti host (phase7/{cfg.run})."]))
        self.finish(outcome if not self.term else f"interrupted: {outcome}")
        return 0


_HEALTH_PY = r"""
import json, sys, urllib.request
port = sys.argv[1]
def get(p):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%s%s" % (port, p), timeout=5) as r:
            return r.status, json.loads(r.read() or b"null")
    except Exception:
        return 0, None
code, _ = get("/health")
print("health=%d" % code)
if code == 200:
    _, props = get("/props")
    dgs = (props or {}).get("default_generation_settings") or {}
    print("n_ctx=%s" % (dgs.get("n_ctx") or (props or {}).get("n_ctx")))
    _, models = get("/v1/models")
    print("model=%s" % ((((models or {}).get("data") or [{}])[0] or {}).get("id")))
"""


def PIPELINE_DONE(cfg: Cfg) -> bool:
    return (cfg.p7 / "PIPELINE-DONE.json").is_file()


def phase6_facts(man: Any) -> Dict[str, Any]:
    """The quantize manifest's result and per-GGUF perplexity (sanity.models.<quant>.ppl)."""
    man = man if isinstance(man, dict) else {}
    models = ((man.get("sanity") or {}).get("models") or {}) if isinstance(man.get("sanity"), dict) else {}
    return {"result": man.get("result"), "max_derived": man.get("max_derived"),
            "perplexity": {k: (v or {}).get("ppl") for k, v in models.items() if isinstance(v, dict)}}


def bf16_sha(man: Any) -> Optional[str]:
    for a in (man or {}).get("artifacts") or [] if isinstance(man, dict) else []:
        if isinstance(a, dict) and a.get("kind") == "gguf" and a.get("quant") in ("BF16", "F16"):
            return a.get("sha256")
    return None


# ---------------------------------------------------------------------------
# The report (pure: tested)
# ---------------------------------------------------------------------------


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


def _kn(summary: Optional[Dict[str, Any]], sname: str, metric: str) -> str:
    m = (((summary or {}).get("sets") or {}).get(sname) or {}).get("metrics", {}).get(metric)
    if not isinstance(m, dict) or m.get("rate") is None:
        return "n/a"
    return f"{100 * m['rate']:.1f}% ({m['k']}/{m['n']}) [{100 * m['ci_low']:.1f}, {100 * m['ci_high']:.1f}]"


def _sub(summary: Optional[Dict[str, Any]], sname: str, ids: Sequence[str]) -> Dict[str, Any]:
    oc = ((((summary or {}).get("sets") or {}).get(sname) or {}).get("outcomes")) or {}
    return subset_rate(oc, ids)


def _subtxt(r: Dict[str, Any]) -> str:
    if r.get("rate") is None:
        return "n/a"
    t = f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"
    return t + (f", {r['missing']} not scored" if r.get("missing") else "")


def headline_lines(data: Dict[str, Any]) -> List[str]:
    s = data.get("summaries") or {}
    b9 = data.get("base9b")
    q4, q5, rc = s.get("q4"), s.get("q5"), s.get("recipe")
    ids = ((data.get("subset") or {}).get("ids")) or []
    flag, fline = bottleneck_flag({"eval-tools": pts(q4, "eval-tools", "task_success"), "sanity": pts(q4, "sanity", "sanity_score")},
                                  {"eval-tools": pts(q5, "eval-tools", "task_success"), "sanity": pts(q5, "sanity", "sanity_score")})
    sub = {k: _sub(s.get(k), "eval-tools", ids) for k in ("q4", "q5")}
    sub["bf16"] = _sub(s.get("bf16"), "eval-tools", ids)
    gates = data.get("gates") or {}
    lines = [
        f"Phase 7 scores ({data.get('run')}): eval-tools task success Q4 {_fmt(pts(q4, 'eval-tools', 'task_success'))}, "
        f"Q5 {_fmt(pts(q5, 'eval-tools', 'task_success'))}; base-9b {_fmt(pts(b9, 'eval-tools', 'task_success'))}, "
        f"same-recipe base {_fmt(pts(rc, 'eval-tools', 'task_success'))}.",
        f"Sanity Q4 {_fmt(pts(q4, 'sanity', 'sanity_score'))}, Q5 {_fmt(pts(q5, 'sanity', 'sanity_score'))} "
        f"(base-9b {_fmt(pts(b9, 'sanity', 'sanity_score'))}). {len(ids)}-case subset: Q4 {_subtxt(sub['q4'])}, "
        f"Q5 {_subtxt(sub['q5'])}, BF16 {_subtxt(sub['bf16'])}.",
        fline[:240],
        f"Gate Q4 vs base-9b: {(gates.get('q4-vs-base-9b') or {}).get('verdict') or 'n/a'}; vs same-recipe base: "
        f"{(gates.get('q4-vs-recipe') or {}).get('verdict') or 'n/a'}. Long-ctx Q4 {_fmt(pts(q4, 'eval-longctx', 'recall'))} "
        f"(base-9b {_fmt(pts(b9, 'eval-longctx', 'recall'))}), injection Q4 {_fmt(pts(q4, 'eval-injection', 'injection_resistance'))}.",
        "Report: phase7/report/DISTILL-PHASE7-SCORES.md on the 3090 Ti host, ready to commit.",
    ]
    return lines[:NOTIFY_LINES]


def render_report(data: Dict[str, Any]) -> str:
    s = data.get("summaries") or {}
    labels = data.get("labels") or {}
    st = data.get("state") or {}
    b9, sal = data.get("base9b"), data.get("saluki")
    q4, q5, bf, rc = s.get("q4"), s.get("q5"), s.get("bf16"), s.get("recipe")
    sub = data.get("subset") or {}
    ids = sub.get("ids") or []
    flag, fline = bottleneck_flag(
        {"eval-tools": pts(q4, "eval-tools", "task_success"), "sanity": pts(q4, "sanity", "sanity_score")},
        {"eval-tools": pts(q5, "eval-tools", "task_success"), "sanity": pts(q5, "sanity", "sanity_score")})
    items = st.get("items") or {}
    tr = st.get("training") or {}
    L: List[str] = []
    L += [f"# Phase 7: scores of the first full 9B student ({data.get('run')})", "",
          f"**Generated:** {data.get('generated')} by `scripts/distill/post_train.py` (unattended). "
          f"Run `{data.get('run')}`, adapter private and max-derived (Q6b): never uploaded.", ""]
    if tr:
        L += [f"**Training:** {tr.get('why')}.", ""]
    L += [f"**{fline}**", ""]
    L += ["## Side by side: Q4_K_M, Q5_K_M, BF16", "",
          "Q4 and Q5 ran the full sets; BF16 ran only the fixed 100-case eval-tools subset (reference). "
          "The subset columns for Q4, Q5 and the bases come from their full runs, on exactly the same ids.", "",
          "| Measure | Q4_K_M | Q5_K_M | BF16 | base-9b (Unsloth Q4_K_M) | base-9b, same recipe (Q4_K_M) |",
          "|---|---|---|---|---|---|"]
    L.append("| eval-tools task success, full set | " + " | ".join([
        _kn(q4, "eval-tools", "task_success"), _kn(q5, "eval-tools", "task_success"), "not run (subset only)",
        _kn(b9, "eval-tools", "task_success"), _kn(rc, "eval-tools", "task_success")]) + " |")
    L.append(f"| eval-tools task success, {len(ids)}-case subset | " + " | ".join([
        _subtxt(_sub(q4, "eval-tools", ids)), _subtxt(_sub(q5, "eval-tools", ids)),
        _subtxt(_sub(bf, "eval-tools", ids)), _subtxt(_sub(b9, "eval-tools", ids)),
        _subtxt(_sub(rc, "eval-tools", ids))]) + " |")
    L.append("| eval-tools arg exactness, full set | " + " | ".join([
        _kn(q4, "eval-tools", "arg_exactness"), _kn(q5, "eval-tools", "arg_exactness"),
        _kn(bf, "eval-tools", "arg_exactness") + " (subset)", _kn(b9, "eval-tools", "arg_exactness"),
        _kn(rc, "eval-tools", "arg_exactness")]) + " |")
    L.append("| sanity | " + " | ".join([
        _kn(q4, "sanity", "sanity_score"), _kn(q5, "sanity", "sanity_score"), "not run",
        _kn(b9, "sanity", "sanity_score"), _kn(rc, "sanity", "sanity_score")]) + " |")
    d_tools = (pts(q5, "eval-tools", "task_success") or 0) - (pts(q4, "eval-tools", "task_success") or 0) \
        if q4 and q5 else None
    r4, rb = _sub(q4, "eval-tools", ids), _sub(bf, "eval-tools", ids)
    L += ["", f"- Q5_K_M − Q4_K_M, eval-tools: {'n/a' if d_tools is None else f'{d_tools:+.1f} pts'}; "
              f"BF16 − Q4_K_M on the subset: "
              + ("n/a" if r4.get("rate") is None or rb.get("rate") is None
                 else f"{100 * (rb['rate'] - r4['rate']):+.1f} pts") + ".",
          f"- BF16 server context: {items.get('c', {}).get('ctx') or 'n/a'} (the largest of "
          f"{'/'.join(map(str, BF16_CTX_LADDER))} that loaded on 24 GB; failed: {items.get('c', {}).get('ctx_failed') or 'none'}). "
          f"Q5_K_M at {items.get('b', {}).get('ctx') or 'n/a'}; Q4_K_M and the bases at 262144.", ""]
    gates = data.get("gates") or {}
    for name, title in (("q4-vs-base-9b", "Gate: Q4_K_M vs base-9b (Unsloth GGUF baseline)"),
                        ("q4-vs-recipe", "Gate, like for like: Q4_K_M vs base-9b through the student's recipe"),
                        ("q5-vs-base-9b", "Q5_K_M vs base-9b (eval-tools and sanity only)"),
                        ("q5-vs-recipe", "Q5_K_M vs the same-recipe base")):
        g = gates.get(name)
        L += [f"## {title}", ""]
        if not g:
            L += ["Not run (a summary was missing).", ""]
            continue
        md = re.sub(r"^## .*\n", "", g.get("md") or "", count=1)
        L += [f"`score.py compare` exit {g.get('exit')} (0 PASS, 1 FAIL, 2 INCONCLUSIVE); verdict "
              f"**{g.get('verdict')}**. Saluki (IQ2 mix, live server, long context at 32K only) is shown alongside.", "",
              md.strip(), ""]
    L += ["## Long context and injection (Q4_K_M)", "",
          "| Measure | Q4_K_M | base-9b | same-recipe base | Saluki |", "|---|---|---|---|---|"]
    for b in ("32k", "64k", "128k", "256k"):
        def by(sm: Any, b: str = b) -> str:
            m = ((((sm or {}).get("sets") or {}).get("eval-longctx") or {}).get("metrics") or {}).get("recall_by_depth") or {}
            r = m.get(b)
            return "n/a" if not r or r.get("rate") is None else f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"
        L.append(f"| long-context recall {b} | {by(q4)} | {by(b9)} | {by(rc)} | {by(sal)} |")
    L.append(f"| long-context recall, all | {_kn(q4, 'eval-longctx', 'recall')} | {_kn(b9, 'eval-longctx', 'recall')} | "
             f"{_kn(rc, 'eval-longctx', 'recall')} | {_kn(sal, 'eval-longctx', 'recall')} |")
    L.append(f"| injection resistance | {_kn(q4, 'eval-injection', 'injection_resistance')} | "
             f"{_kn(b9, 'eval-injection', 'injection_resistance')} | {_kn(rc, 'eval-injection', 'injection_resistance')} | "
             f"{_kn(sal, 'eval-injection', 'injection_resistance')} |")
    L += ["", "The paired long-context gate (DISTILL-PHASE4-BASELINES.md) is in the gate sections above.", ""]
    L += ["## The fixed eval-tools subset", "",
          f"- {sub.get('n')} of {sub.get('source_cases')} eval-tools cases; seed `{sub.get('seed')}`; {sub.get('method')}.",
          f"- Source: the Phase 4 eval-tools set, sha256 `{sub.get('source_sha256')}`. The ids are committed in "
          f"`docs/sprints/{SUBSET_FILE}`.", "", "| Category | Cases | Chosen |", "|---|---|---|"]
    for k, v in (sub.get("strata") or {}).items():
        L.append(f"| {k} | {v.get('cases')} | {v.get('chosen')} |")
    L += ["", "## Setup", "",
          f"- Scoring server (every model): `llama-server {data.get('server_flags')}` "
          "(the same binary and flags base-9b was scored with, plus `--alias`); BF16 at the context above.",
          "- `score.py run --concurrency 1` from the scorer code base-9b was scored with (scorer v4); summaries and "
          "gates by this checkout's `score.py` (scorer v4), as the head-4ef1dfe baselines were.",
          "- The same-recipe base: the official Qwen3.5-9B bf16 (revision as in Phase 5's provenance), converted with "
          "`--no-mtp`, an imatrix from the student's exact calibration text, Q4_K_M (`quantize.py run --no-adapter`). "
          "Its name carries max-derived because that calibration text is.",
          ""]
    p6 = st.get("phase6") or {}
    if p6:
        L.append(f"- Phase 6 quantize: result {p6.get('result')}; perplexity on the held-back train slice: "
                 + ", ".join(f"{k} {v:.3f}" if isinstance(v, (int, float)) else f"{k} n/a"
                             for k, v in (p6.get("perplexity") or {}).items()) + ".")
    rq = st.get("recipe_quantize") or {}
    if rq:
        L.append(f"- Same-recipe base quantize: result {rq.get('result')}; perplexity "
                 + ", ".join(f"{k} {v:.3f}" if isinstance(v, (int, float)) else f"{k} n/a"
                             for k, v in (rq.get("perplexity") or {}).items()) + ".")
    L += ["", "## What ran", "", "| Item | Model | Sets | Status | Notes |", "|---|---|---|---|---|"]
    for key, model, sets, _opt in ITEMS:
        it = items.get(key) or {}
        L.append(f"| {key} | {labels.get(model, model)} | {', '.join(sets)} | {it.get('status', 'not reached')} | "
                 f"{(it.get('why') or json.dumps(it.get('sets') or {}))[:160]} |")
    rest = st.get("restore") or {}
    L += ["", f"- Restore: {rest.get('train_end', 'n/a')[:200]}" if rest else "- Restore: not needed or not reached"]
    for n in (st.get("notes") or [])[-10:]:
        L.append(f"- Note: {n}")
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Fakes for the GPU-free rehearsal on this host (--fake-root)
# ---------------------------------------------------------------------------


def fake_server(argv: Sequence[str]) -> int:
    """A stand-in llama-server: /health, /props, /v1/models, /v1/chat/completions. Control files in
    $FAKE_ROOT/control: max-ctx-<alias> (refuse to load above it), crash-after-<alias> (exit after N
    replies, once)."""
    args = list(argv)
    opt = {k: args[i + 1] for i, k in enumerate(args[:-1]) if k in ("-m", "-c", "--port", "--alias", "--host")}
    a = argparse.Namespace(m=opt.get("-m"), c=int(opt.get("-c", 4096)), port=int(opt.get("--port", 8080)),
                           alias=opt.get("--alias", "fake"), host=opt.get("--host", LOOPBACK))
    ctl = Path(os.environ.get("FAKE_ROOT", "/nonexistent")) / "control"
    mx = ctl / f"max-ctx-{a.alias}"
    if mx.is_file() and a.c > int(mx.read_text().strip()):
        print(f"fake llama-server: failed to allocate the KV cache for -c {a.c}", flush=True)
        return 1
    if not a.m or not Path(a.m).is_file():
        print(f"fake llama-server: no model {a.m}", flush=True)
        return 1
    crash_file = ctl / f"crash-after-{a.alias}"
    crash_after = int(crash_file.read_text().strip()) if crash_file.is_file() else -1
    served = [0]
    time.sleep(1.0)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(self, obj: Any) -> None:
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send({"status": "ok"})
            elif self.path == "/props":
                self._send({"default_generation_settings": {"n_ctx": a.c}, "total_slots": 1})
            elif self.path == "/v1/models":
                self._send({"data": [{"id": a.alias}]})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            tools = req.get("tools") or []
            if tools:
                name = (tools[0].get("function") or {}).get("name", "x")
                msg = {"role": "assistant", "content": "",
                       "tool_calls": [{"id": "c1", "type": "function",
                                       "function": {"name": name, "arguments": "{}"}}]}
                fin = "tool_calls"
            else:
                msg = {"role": "assistant", "content": "ok"}
                fin = "stop"
            self._send({"choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
            served[0] += 1
            if 0 <= crash_after <= served[0]:
                crash_file.unlink(missing_ok=True)
                threading.Thread(target=lambda: (time.sleep(0.2), os._exit(134)), daemon=True).start()

    srv = http.server.ThreadingHTTPServer((a.host, a.port), H)
    print(f"fake llama-server {a.alias} -c {a.c} on {a.host}:{a.port}", flush=True)
    srv.serve_forever()
    return 0


def fake_setup(root: Path, code: Path, run: str, n_per_set: int = 6) -> None:
    """A fake 4090 tree under ROOT/tower and a fake 'here' under ROOT/here, from small slices of the
    real eval files (they stay on this host)."""
    t, h = root / "tower", root / "here"
    for d in ("bin", "llama.cpp/build/bin", "distill/evalsets", "distill/window/state",
              f"distill/phase5/runs/{run}/adapter-max-derived", "distill/phase4/results/base-9b/baseline",
              "distill/venv-score/bin", "distill/phase6-rehearsal/qvenv/bin", "distill/models/qwen3.5-9b-tokenizer"):
        (t / d).mkdir(parents=True, exist_ok=True)
    for d in ("distill/phase3/build-v2", "distill/phase3/evalsets", "distill/models/qwen3.5-9b-official",
              "distill/phase4/results/saluki"):
        (h / d).mkdir(parents=True, exist_ok=True)
    (root / "control").mkdir(exist_ok=True)
    real = Path.home() / "distill"
    srcs = {"eval-tools": real / "phase3/build-v2/eval-tools.jsonl",
            "sanity": real / "phase3/evalsets/sanity.jsonl",
            "eval-injection": real / "phase3/evalsets/eval-injection.jsonl",
            "eval-longctx": real / "phase3/evalsets/eval-longctx.jsonl"}
    for sname, src in srcs.items():
        rows = []
        with open(src, "rb") as fh:
            per_cat: Dict[str, int] = {}
            for raw in fh:
                if not raw.strip():
                    continue
                if sname == "eval-tools":
                    cat = str((json.loads(raw).get("meta") or {}).get("category"))
                    if per_cat.get(cat, 0) >= n_per_set:
                        continue
                    per_cat[cat] = per_cat.get(cat, 0) + 1
                elif len(rows) >= n_per_set:
                    break
                rows.append(raw.rstrip(b"\n"))
        blob = b"\n".join(rows) + b"\n"
        (t / "distill/evalsets" / f"{sname}.jsonl").write_bytes(blob)
        dst = h / ("distill/phase3/build-v2/eval-tools.jsonl" if sname == "eval-tools"
                   else f"distill/phase3/evalsets/{sname}.jsonl")
        dst.write_bytes(blob)
    me = Path(__file__).resolve()
    sh = "#!/bin/sh\n"
    (t / "llama.cpp/build/bin/llama-server").write_text(
        sh + f"FAKE_ROOT={q(str(root))} exec /usr/bin/python3 -I {q(str(me))} fake-server \"$@\"\n")
    (t / "bin/nvidia-smi").write_text(sh + "echo 0\n")
    for p in (t / "llama.cpp/build/bin/llama-server", t / "bin/nvidia-smi"):
        p.chmod(0o755)
    py = t / "distill/venv-score/bin/python"
    if not py.exists():
        py.symlink_to("/usr/bin/python3")
    code_link = t / "distill/code"
    if not code_link.exists():
        code_link.symlink_to(code)
    # the fake quantize.py interpreter: makes the recipe outputs quantize.py would
    (t / "distill/phase6-rehearsal/qvenv/bin/python").write_text(sh + f"exec /usr/bin/python3 -I {q(str(me))} fake-quantize \"$@\"\n")
    (t / "distill/phase6-rehearsal/qvenv/bin/python").chmod(0o755)
    # the trainer's records: a completed run
    rd = t / f"distill/phase5/runs/{run}"
    (rd / "adapter-max-derived/adapter_config.json").write_text("{}\n")
    (rd / "adapter-max-derived/adapter_model.safetensors").write_bytes(b"x" * 64)
    hist = [{"step": 0, "val_loss": 1.0, "probe_loss": 1.13}, {"step": 560, "val_loss": 0.80, "probe_loss": 0.90},
            {"step": 1120, "val_loss": 0.78, "probe_loss": 0.88}]
    write_json(rd / "manifest.json", {"status": "done", "mode": "train", "step": 1120, "adapter": str(rd / "adapter-max-derived"),
                                      "config": {"epochs": 2}, "val_history": hist, "nonfinite_steps": 0,
                                      "epochs_done": [{"epoch": 1, "step": 560}, {"epoch": 2, "step": 1120}],
                                      "progress": {"total_steps": 1120}, "data": {"sha256": "0" * 64}})
    write_json(rd / "run_state.json", {"val_history": hist})
    # the baselines: copy real summaries in by hand to exercise compare (fake cases differ: INCONCLUSIVE)
    (h / "distill/models/qwen3.5-9b-official/config.json").write_text('{"architectures": ["Fake"]}\n')
    (h / "distill/models/qwen3.5-9b-official/model.safetensors").write_bytes(b"w" * 1024)
    # the window: a fake train_end.sh and notify
    (t / "distill/window/window.conf").write_text("BIZ_NAME=FakeBiz\n")
    note = root / "telegrams.txt"
    notify = h / "distill/notify.sh"
    notify.write_text(sh + f"{{ echo \"--- $(date +%T)\"; cat; }} >> {q(str(note))}\n")
    notify.chmod(0o755)
    (t / "distill/window/train_end.sh").write_text(
        "#!/bin/bash\n"
        f"S={q(str(t / 'distill/window/state'))}; msg=''; [ \"$1\" = --msg-file ] && msg=$(head -3 \"$2\")\n"
        "echo \"OK $(date '+%F %T') restored: fake verify PASS (train_end)\" > $S/train-end.last\n"
        "echo \"$(date '+%F %T') tower_verify: PASS fake\" > $S/verify.last\n"
        f"printf 'FakeBiz back: business RESTORED + verified.\\n%s\\n' \"$msg\" | {q(str(notify))} 5\n")
    (t / "distill/window/train_end.sh").chmod(0o755)


def fake_quantize(argv: Sequence[str]) -> int:
    """Stands in for quantize.py run (recipe) and phase6_student.sh's outputs: writes the GGUFs and
    a manifest with their sha256, like the real ones."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-I", action="store_true")
    ap.add_argument("-u", action="store_true")
    a, rest = ap.parse_known_args(list(argv))
    out = rest[rest.index("--out") + 1] if "--out" in rest else None
    label = rest[rest.index("--label") + 1] if "--label" in rest else "x"
    quants = (rest[rest.index("--quants") + 1] if "--quants" in rest else "Q4_K_M").split(",")
    if not out:
        print("fake quantize: no --out")
        return 2
    o = Path(out)
    (o / "gguf").mkdir(parents=True, exist_ok=True)
    arts, models = [], {}
    for i, qn in enumerate(["BF16"] + quants):
        p = o / "gguf" / f"{label}-{qn}.gguf"
        p.write_bytes(f"fake {label} {qn}".encode() * 64)
        arts.append({"kind": "gguf", "quant": qn, "name": p.name, "bytes": p.stat().st_size,
                     "sha256": sha256_file(p)})
        models[qn] = {"ppl": 10.5 + 0.05 * i, "pass": True}
    write_json(o / "quantize-manifest.json", {"result": "PASS", "max_derived": True, "artifacts": arts,
                                              "sanity": {"models": models}})
    print(f"fake quantize: {label} {quants} -> {o}")
    return 0


def fake_phase6(argv: Sequence[str]) -> int:
    """Stands in for phase6_student.sh (--fake-root): GGUFs + calib + code on the fake 4090, the
    manifest here."""
    root = Path(os.environ["FAKE_ROOT"])
    adapter_rel = argv[0]
    run = Path(adapter_rel).parent.name
    if "--execute" not in argv:
        print(f"fake phase6 plan: {run}")
        return 0
    t6 = root / "tower" / "distill" / "phase6" / run
    (t6 / "calib").mkdir(parents=True, exist_ok=True)
    (t6 / "calib" / "calibration.json").write_text("{}\n")
    code = t6 / "code"
    code.mkdir(exist_ok=True)
    rc = fake_quantize(["--out", str(t6), "--label", run, "--quants", "Q4_K_M,Q5_K_M"])
    here = root / "here" / "distill" / "phase6" / run
    here.mkdir(parents=True, exist_ok=True)
    (here / "quantize-manifest.json").write_bytes((t6 / "quantize-manifest.json").read_bytes())
    (code / "quantize.py").write_text("# fake\n")
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tower", default="", help="ssh target of the 4090 host (user@host)")
    p.add_argument("--run", default=RUN_DEFAULT)
    p.add_argument("--adapter-name", default=ADAPTER_DEFAULT)
    p.add_argument("--trainer-unit", default="distill-train-full-20261031T195413.service")
    p.add_argument("--watcher-unit", default="flywheel-train-watch-20261031T195413.service")
    p.add_argument("--stop-at", default=STOP_AT_DEFAULT, help="scoring stops and the restore starts by then")
    p.add_argument("--hard-at", default=HARD_AT_DEFAULT, help="the hard deadline (05:20 EST)")
    p.add_argument("--poll", type=float, default=60.0, help="seconds between checks while waiting for training")
    p.add_argument("--state-dir", default="", help="default ~/distill/phase7/<run>")
    p.add_argument("--notify", default="", help="default ~/distill/notify.sh (this host)")
    p.add_argument("--here-python", default="/usr/bin/python3", help="python for score.py summarize/compare here")
    p.add_argument("--unit-prefix", default="", help=f"4090 unit prefix (default {TOWER_PREFIX})")
    p.add_argument("--restore-prefix", default="")
    p.add_argument("--biz-unit", default="")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--subset-n", type=int, default=SUBSET_N)
    p.add_argument("--ids-file", default="", help=f"default docs/sprints/{SUBSET_FILE} in this checkout")
    p.add_argument("--phase6-cmd", default="", help="default this checkout's phase6_student.sh")
    p.add_argument("--dry-adapter", default="", help="dry run: phase6 plan mode on this adapter instead "
                                                     "(e.g. distill/phase5/runs/9b-smoke-v4-max-derived/adapter-max-derived)")
    p.add_argument("--skip-recipe", action="store_true", help="no same-recipe base (item e)")
    p.add_argument("--fake-root", default="", help="run against a fake 4090 tree on this host (fake-setup)")
    p.add_argument("--execute", action="store_true")


def cmd_run(a: argparse.Namespace) -> int:
    if not a.tower and not a.fake_root:
        print("--tower USER@HOST is required (or --fake-root)", file=sys.stderr)
        return 2
    try:
        cfg = Cfg(a)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    pipe = Pipeline(cfg)

    def on_term(*_: Any) -> None:
        pipe.term = True
        pipe.log("SIGTERM: stopping our units, then the restore")

    signal.signal(signal.SIGTERM, on_term)
    try:
        return pipe.run()
    except Exception as exc:  # noqa: BLE001 - logged; the unit restarts and resumes
        pipe.log(f"CRASH: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        if not cfg.execute:
            return 1
        if pipe.state.get("starts", 0) >= MAX_STARTS:
            pipe.release("gave up after a crash")
            pipe.finish(f"crashed: {exc}")
            return 0
        return 1


def cmd_arm(a: argparse.Namespace) -> int:
    """Start the pipeline as a transient user unit on this host (restarts on a crash, resumes)."""
    stamp = datetime.now(ET).strftime("%Y%m%dT%H%M%S")
    unit = f"{ORCH_PREFIX}{stamp}"
    me = Path(__file__).resolve()
    p7 = Path(a.state_dir).expanduser() if a.state_dir else Path.home() / "distill" / "phase7" / a.run
    log = p7 / "unit.log"
    fwd: List[str] = []
    for k, v in sorted(vars(a).items()):
        if k in ("cmd", "execute", "func") or v in (None, "", False):
            continue
        flag = "--" + k.replace("_", "-")
        fwd += [flag] if v is True else [flag, str(v)]
    argv = ["systemd-run", "--user", f"--unit={unit}", "--description=flywheel: post-training quantize + score + "
            "report + restore (post_train.py)", "-p", "KillMode=mixed", "-p", "TimeoutStopSec=2700",
            "-p", "Restart=on-failure", "-p", "RestartSec=120",
            "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
            "--setenv=TZ=America/New_York", "--setenv=PYTHONUNBUFFERED=1", "--expand-environment=no",
            "/usr/bin/python3", "-I", str(me), "run", *fwd, "--execute"]
    print(shlex.join(argv))
    if not a.execute:
        print("dry run: nothing started (--execute starts it)")
        return 0
    p7.mkdir(parents=True, exist_ok=True)
    rc = subprocess.run(argv).returncode
    if rc == 0:
        (p7 / "unit.name").write_text(unit + ".service\n", encoding="utf-8")
        print(f"armed: {unit}.service (state {p7 / 'state.json'}; stop: systemctl --user stop {unit}.service)")
    return rc


def cmd_status(a: argparse.Namespace) -> int:
    p7 = Path(a.state_dir).expanduser() if a.state_dir else Path.home() / "distill" / "phase7" / a.run
    st = read_json_text((p7 / "state.json").read_text(encoding="utf-8")) if (p7 / "state.json").is_file() else None
    if not st:
        print(f"no state at {p7 / 'state.json'}")
        return 1
    print(json.dumps({"starts": st.get("starts"), "outcome": st.get("outcome"),
                      "gpu_taken": st.get("gpu_taken"), "released": st.get("released"),
                      "steps": {k: v.get("status") for k, v in (st.get("steps") or {}).items()},
                      "items": {k: {kk: v.get(kk) for kk in ("status", "sets", "ctx", "crashes", "why")}
                                for k, v in (st.get("items") or {}).items()},
                      "done": (p7 / "PIPELINE-DONE.json").is_file()}, indent=2))
    return 0


def cmd_subset(a: argparse.Namespace) -> int:
    src = Path(a.cases).expanduser()
    sub = select_subset(read_jsonl(src), a.n, a.seed, source_sha256=sha256_file(src))
    text = json.dumps(sub, indent=2) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text if not a.out else f"{sub['n']} ids -> {a.out}; strata {sub['strata']}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["fake-server"]:
        return fake_server(args[1:])
    if args[:1] == ["fake-quantize"]:
        return fake_quantize(args[1:])
    if args[:1] == ["fake-phase6"]:
        return fake_phase6(args[1:])
    ap = argparse.ArgumentParser(prog="post_train", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    add_run_args(r)
    ar = sub.add_parser("arm")
    add_run_args(ar)
    s = sub.add_parser("status")
    s.add_argument("--run", default=RUN_DEFAULT)
    s.add_argument("--state-dir", default="")
    ss = sub.add_parser("subset")
    ss.add_argument("--cases", required=True)
    ss.add_argument("--n", type=int, default=SUBSET_N)
    ss.add_argument("--seed", default=SUBSET_SEED)
    ss.add_argument("--out", default="")
    fs = sub.add_parser("fake-setup")
    fs.add_argument("--root", required=True)
    fs.add_argument("--run", default=RUN_DEFAULT)
    fs.add_argument("--n-per-set", type=int, default=6)
    a = ap.parse_args(args)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "arm":
        return cmd_arm(a)
    if a.cmd == "status":
        return cmd_status(a)
    if a.cmd == "subset":
        return cmd_subset(a)
    if a.cmd == "fake-setup":
        fake_setup(Path(a.root).expanduser().resolve(), Path(__file__).resolve().parent.parent.parent, a.run,
                   a.n_per_set)
        print(f"fake tree at {a.root}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
