#!/usr/bin/env python3
"""Teacher quant A/B night on the 3090 Ti host: the serving IQ3 teacher (arm A) against a partly offloaded
Q4_K_M (arm B, and the judge of all traces) on the local model port, then everything back as it was.

    mini_ab.py run      [--execute]        # the night (without --execute: FAKE, see below)
    mini_ab.py restore  [--execute]        # the IQ3 teacher back after an A/B (OnFailure= handler, backstop)
    mini_ab.py ngl                         # read-only: the -ngl the GPU allows the Q4 teacher right now
    mini_ab.py arm --date YYYY-MM-DD [--at HHMM] [--execute]   # the one-shot timers for that night

The operator's rule (DISTILL-OPEN-QUESTIONS, teacher A/B): the same 30 real-pattern tasks on IQ3 and on
Q4_K_M, one judge; IQ3 more than 5 points lower -> use the bigger quant (with less context), else keep IQ3.
``teacher_ab.py`` (staged as a HEAD export in ``AB_CODE``) does the A/B itself; this drives it on the live
host inside the flywheel window (OQ10: the daemon keeps the window's local teacher; no Max; Ollama stays
loaded, Q9). One ``run``:

1. pauses local generation: every running ``distill-gen-*`` unit is stopped BY NAME (gen_runner stops its
   children and its probe guard on SIGTERM; teacher_gen reruns the task in flight later). The units' ExecStart
   and the exact command that resumes it (``mini_window.start_gen``: a new ``distill-gen-<stamp>`` running the
   same ``gen_runner.sh run --execute``, which resumes from its ledgers) go to ``state/ab-night.json`` and the
   log;
2. starts this run's own probe guard on ``AB_GUARD_LISTEN`` (127.0.0.1:18593, the port gen_runner's guard
   just freed) in front of the model port. EVERY A/B request (both arms, the judge) goes through it: new
   requests start only just after a voice-probe result, a bad probe trips its abort switch;
3. ``teacher_ab.py select``: ``AB_N`` tasks from ``AB_TASKS`` (real-v3-excl), stratified and seeded, with
   explicit identical sampling and request settings for both arms, ``--allow-ctx-difference`` (the arms'
   servers differ in per-slot context: 57,344 vs 16,384);
4. arm A on the serving IQ3 teacher (through the guard);
5. the swap: the guard stopped, ``distill-teacher.service`` stopped, the free VRAM read (nvidia-smi,
   read-only), ``-ngl`` chosen from the Q4 GGUF's own tensor table (llama.cpp's placement: the output matrix,
   then trunk layers from the end; + KV + recurrent state + buffers + 600 MiB headroom must fit; fewer than
   ``AB_MIN_NGL`` refuses; a projected decode rate under the voice probe's floor refuses BEFORE any change,
   already at the preflight, so arm A then runs alone and 8092 is never swapped: ``AB_Q4_MIN_EST_TPS``), NGL and
   MIN_FREE_MIB written to ``AB_Q4_ENV``, ``distill-teacher-q4-partial.service`` started (its ExecStartPre
   refuses with less free VRAM), then healthy and serving its alias, enough VRAM left beside it, and the NEXT
   voice probe OK (>= PROBE_MIN_TPS and its own floor). Anything else: back to IQ3 at once and no Q4 tonight;
6. arm B, then the judge of all traces of both arms, on the Q4 server (through a new guard);
7. ``teacher_ab.py report``: ab_report.json + ab_summary.txt; five lines to ``NOTIFY``;
8. ALWAYS (normal end, an error, SIGTERM/SIGINT/SIGHUP, the time box): the step in flight stopped, the IQ3
   teacher back (Q4 unit stopped, distill-teacher started, healthy, the next voice probe OK, daemon /health;
   two attempts, then mini_window's business restore with the NO-SWAPS flag), the guard stopped, the report,
   and generation restarted (only if this run paused it, and only with the IQ3 teacher serving).

``AB_ARM_B=remote`` (OQ #12 option c): arm B and the judge run on the 4090's teacher over the tailnet instead
(``teacher_ab.py arm --arm B`` / ``judge`` with the remote flags, started apart from this script; the tools stay
on this host). The night then runs steps 1-4 and 7-8 only: no Q4 swap, no Q4 file read, arm B labelled
``AB_REMOTE_LABEL_B``.

Nightly evals (cron EVAL_TIMES, 03:00 and 06:00 ET) are left alone and score the IQ3 teacher, as every eval
in the window does: no Q4 phase runs within ``AB_Q4_EVAL_GAP_MIN`` of one (the swap back starts then) or while
one runs, arm A pauses ``AB_IQ3_EVAL_GAP_MIN`` before one (as gen_runner does), and after the eval the
remaining Q4 work swaps in again. The hard end is ``AB_HARD_STOP_HHMM`` (06:30 ET): no step runs past it and
the last Q4 phase ends ``AB_SWAPBACK_MARGIN_MIN`` before it. Times are epochs throughout; ET is only for
reading the settings (the night of 2026-11-01 falls back from EDT to EST at 02:00).

FAKE (``run`` without ``--execute``): the read-only checks run for real and report what a real run would refuse
(then go on); every change (unit stops/starts, generation, the guard, notify) is printed instead; the GPU
numbers are the prediction from the running teacher's own use; ``teacher_ab.py select``/``status``/``report``
run for real in a scratch directory (never ``AB_OUT``); arms and the judge are simulated on a virtual clock
from tonight's start, so the whole night (eval slots, the hard end, the trap) runs in seconds.
``--fake-probe-fail`` makes the Q4 voice probe fail; ``--fake-error-at STEP`` raises in a step.

Exit: 0 done (a decision, or the night ended cleanly), 1 failed (notified), 2 skipped by a check (nothing
changed), 64 usage. Standard library only; runs from ``~/distill/window/`` beside mini_window.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_runner as GR  # noqa: E402 - window.conf + GEN_* defaults, host_port, port_free
import mini_window as MW  # noqa: E402 - the window's Window (units, probes, swaps' waits, restore, notify)

AB_DEFAULTS: dict[str, str] = {
    "AB_OUT": "~/distill/teacher-ab/mini-v3",
    "AB_CODE": "~/distill/teacher-ab/code-head",     # gen_runner.py stage-code --dest distill/teacher-ab/code-head
    "AB_PY": "",                                       # empty: GEN_PY
    "AB_TASKS": "~/distill/phase2b/tasksets/real-v3-excl.jsonl",
    "AB_N": "30",
    # Where arm B (and the judge of both arms) runs: "local" (the Q4_K_M swap on this host, steps 5-6) or "remote"
    # (DISTILL-OPEN-QUESTIONS #12 option c: the 4090's teacher on the tailnet, run with teacher_ab.py arm --arm B /
    # judge and the remote flags; never by this script). With "remote" the night runs arm A only, never swaps the
    # local model port and never reads the Q4 file; the selection labels arm B AB_REMOTE_LABEL_B.
    "AB_ARM_B": "local",
    "AB_REMOTE_LABEL_B": "UD-Q4_K_XL",
    "AB_SEED": "teacher-ab-mini-v3",
    "AB_TEMPERATURE": "0.7",
    "AB_TOP_K": "20",
    "AB_TOP_P": "0.8",
    "AB_MIN_P": "0.0",
    "AB_MAX_TOKENS": "4096",
    "AB_CONCURRENCY": "1",                             # the daemon and the voice probe share the 2 slots
    "AB_REQUEST_TIMEOUT": "1800",                      # a guard hold (<= ~5 min) + a slow Q4 reply
    "AB_TASK_TIMEOUT": "5400",
    "AB_MIN_TASK_S": "300",
    "AB_GUARD_LISTEN": "127.0.0.1:18593",
    "AB_GUARD_WAIT_S": "20",
    "AB_Q4_UNIT": "distill-teacher-q4-partial.service",
    "AB_Q4_MODEL": "~/distill/models/teacher-q4/Qwen3.8-27B-Q4_K_M.gguf",
    "AB_Q4_ALIAS": "qwen3.8-27b-q4_k_m",
    "AB_Q4_ENV": "~/distill/window/state/q4-partial.env",
    "AB_Q4_CTX": "16384",
    "AB_KV_BYTES_PER_TOKEN": "34816",                  # q8_0, 16 attention layers x 4 KV heads x 256 x 2
    "AB_RECURRENT_MIB": "300",
    "AB_BUFFERS_MIB": "1192",
    "AB_PARTIAL_EXTRA_MIB": "512",
    "AB_HEADROOM_MIB": "600",
    "AB_MIN_NGL": "40",
    # The voice probe's floor holds on the Q4 server too: no swap when its decode rate, projected from the bytes
    # one token reads (GPU-resident weights at AB_GPU_GBPS, CPU-resident ones at AB_CPU_GBPS; calibrated to the
    # Q4 manifest's fit math: 31.8 tok/s all on the GPU, 8.5 at -ngl 46), is under this. Empty: PROBE_MIN_TPS;
    # 0: no projection, the first voice probe on Q4 decides (and a SLOW one alerts).
    "AB_Q4_MIN_EST_TPS": "",
    "AB_GPU_GBPS": "524",
    "AB_CPU_GBPS": "55",
    "AB_POST_MIN_FREE_MIB": "256",
    "AB_IQ3_MIN_FREE_MIB": "12800",                    # distill-teacher.service's own MIN_FREE_MIB
    "AB_START_HHMM": "0100",
    "AB_HARD_STOP_HHMM": "0630",
    "AB_BACKSTOP_HHMM": "0725",
    "AB_MIN_START_LEFT_MIN": "90",
    "AB_Q4_EVAL_GAP_MIN": "35",                        # EVAL_GAP_MIN + the swap back itself
    "AB_IQ3_EVAL_GAP_MIN": "5",                        # gen_runner's GEN_EVAL_PAUSE_BEFORE_MIN
    "AB_EVAL_AFTER_MIN": "3",                          # cron starts late: wait this long to see it run
    "AB_SWAPBACK_MARGIN_MIN": "20",
    "AB_MIN_PHASE_MIN": "20",
    "AB_SWAP_MIN": "12",                               # what a swap to Q4 takes (tick + load + probe), budgeted
    "AB_STEP_RETRIES": "2",
    "AB_RETRY_PAUSE_S": "30",
    "AB_POLL_S": "5",
    "AB_STEP_GRACE_S": "20",
    "AB_NICE": "nice -n 10",
    "AB_UNIT_PREFIX": "window-mini-ab",
    "AB_FAKE_TASK_S": "300 900 150",                   # simulated seconds per arm-A task, arm-B task, judge call
}
LABELS = {"A": "IQ3_XXS", "B": "Q4_K_M"}
STATE_NAME = "ab-night.json"
PID_NAME = "ab-run.pid"
MIB = 1 << 20


class _NoLocalArmB(Exception):
    """AB_ARM_B=remote: the preflight skips the Q4 VRAM plan (no local arm B)."""


class Q4Refused(Exception):
    """The Q4 teacher cannot serve tonight (refused, failed, or the probe was not OK): IQ3 is back."""


class Abort(Exception):
    """Stop the night (the cleanup still runs)."""


def arm_b_remote(c: dict[str, str]) -> bool:
    """AB_ARM_B: "remote" runs arm B on the remote teacher (not here); anything but local/remote is refused."""
    mode = c.get("AB_ARM_B", "local").strip().lower()
    if mode not in ("local", "remote"):
        raise ValueError(f"AB_ARM_B={c.get('AB_ARM_B')!r}: local or remote")
    return mode == "remote"


def load(conf_path: Path) -> dict[str, str]:
    c = GR.load(conf_path)
    home = os.path.expanduser("~")
    for k, v in AB_DEFAULTS.items():
        if k not in c:
            c[k] = home + v[1:] if v.startswith("~/") else v
    c["AB_PY"] = c["AB_PY"] or c["GEN_PY"]
    return c


# ── time ─────────────────────────────────────────────────────────────────────


def at_et(day: date, hhmm: str, tz: ZoneInfo) -> float:
    """Epoch of ``hhmm`` ET on ``day``; an hour that occurs twice (the fall back) is its FIRST occurrence."""
    h, m = MW.hhmm(hhmm)
    return datetime(day.year, day.month, day.day, h, m, tzinfo=tz, fold=0).timestamp()


def night_times(c: dict[str, str], start: float) -> dict[str, Any]:
    """The hard end (the next AB_HARD_STOP_HHMM after ``start``) and the nightly evals around the night."""
    tz = ZoneInfo(c["ET_TZ"])
    d = datetime.fromtimestamp(start, tz).date()
    hard = at_et(d, c["AB_HARD_STOP_HHMM"], tz)
    if hard <= start:
        hard = at_et(d + timedelta(days=1), c["AB_HARD_STOP_HHMM"], tz)
    evals = sorted({at_et(day, t.strip(), tz) for day in (d, d + timedelta(days=1))
                    for t in c["EVAL_TIMES"].split(",") if t.strip()})
    evals = [e for e in evals if start - 3600 < e < hard + 3600]
    return {"start": start, "hard": hard, "evals": evals}


def eval_free_at(now: float, evals: list[float], before_s: float, after_s: float, running: bool) -> float | None:
    """None when nothing eval-related holds this kind of work now, else when to look again."""
    if running:
        return now + 30
    for e in evals:
        if e - before_s <= now < e + after_s:
            return e + after_s
    return None


def next_eval_hold(now: float, evals: list[float], before_s: float) -> float:
    """When the next eval's hold for this kind of work starts (inf when none is ahead)."""
    return min((e - before_s for e in evals if e - before_s > now), default=float("inf"))


def calendar_utc(t: float) -> str:
    return datetime.fromtimestamp(t, ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S UTC")


def iso_et(t: float, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(t, tz).isoformat(timespec="seconds")


# ── the Q4 GGUF and the GPU ──────────────────────────────────────────────────

_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_layers(path: Path) -> dict[str, Any]:
    """The GGUF's per-layer weight sizes (bytes, from its tensor table: the gaps between tensor offsets),
    the trunk layer count (block_count less the MTP/nextn layers, never loaded) and the output matrix."""
    with Path(path).open("rb") as f:
        def read(n: int) -> bytes:
            b = f.read(n)
            if len(b) != n:
                raise ValueError(f"{path}: truncated GGUF header")
            return b

        def u32() -> int:
            return int(struct.unpack("<I", read(4))[0])

        def u64() -> int:
            return int(struct.unpack("<Q", read(8))[0])

        def string() -> str:
            n = u64()
            if n > 64 << 20:
                raise ValueError(f"{path}: implausible string length")
            return read(n).decode("utf-8", "replace")

        def value(kind: int) -> Any:
            if kind in _FIXED:
                return struct.unpack(_FIXED[kind], read(struct.calcsize(_FIXED[kind])))[0]
            if kind == 8:
                return string()
            if kind == 9:
                inner, n = u32(), u64()
                if inner in _FIXED:
                    f.seek(n * struct.calcsize(_FIXED[inner]), os.SEEK_CUR)
                else:
                    for _ in range(n):
                        value(inner)
                return None
            raise ValueError(f"{path}: unknown GGUF value type {kind}")

        if read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        if u32() not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version")
        n_tensors, n_kv = u64(), u64()
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            k = string()
            meta[k] = value(u32())
        infos: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = string()
            for _ in range(u32()):
                u64()
            u32()
            infos.append((name, u64()))
        align = int(meta.get("general.alignment") or 32)
        data_start = (f.tell() + align - 1) // align * align
    size = Path(path).stat().st_size
    infos.sort(key=lambda x: x[1])
    per_layer: dict[int, int] = {}
    other: dict[str, int] = {}
    for i, (name, off) in enumerate(infos):
        nbytes = (infos[i + 1][1] if i + 1 < len(infos) else size - data_start) - off
        m = re.match(r"blk\.(\d+)\.", name)
        if m:
            per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + nbytes
        else:
            other[name] = nbytes
    arch = str(meta.get("general.architecture") or "")
    blocks = int(meta.get(f"{arch}.block_count") or (max(per_layer) + 1 if per_layer else 0))
    nextn = int(meta.get(f"{arch}.nextn_predict_layers") or 0)
    trunk = blocks - nextn
    return {"trunk": [per_layer.get(i, 0) for i in range(trunk)], "output": other.get("output.weight", 0),
            "blocks": blocks, "nextn": nextn}


def gpu_weight_bytes(layers: dict[str, Any], ngl: int) -> int:
    """The weight bytes llama.cpp (9d57ce456, llama-model.cpp load_tensors) puts on the GPU for ``-ngl N``:
    the layer indices ``[blocks + 1 - N, blocks]``, where index ``blocks`` is the OUTPUT layer and the indices
    from the trunk's end up are the MTP block(s) (TENSOR_SKIP without MTP decoding: never loaded). So any
    N >= 1 puts the output matrix (995 MiB here) on the GPU first, and N - 1 - nextn trunk layers from the end
    (44 at -ngl 46); the token embeddings always stay on the CPU."""
    if ngl <= 0:
        return 0
    trunk = layers["trunk"]
    k = max(0, min(len(trunk), len(trunk) - (int(layers["blocks"]) + 1 - ngl)))
    return int(layers["output"]) + sum(trunk[len(trunk) - k:])


def plan_ngl(layers: dict[str, Any], free_mib: float, c: dict[str, str]) -> tuple[int, float]:
    """(-ngl, projected MiB on the GPU): the largest -ngl whose weights (gpu_weight_bytes: the output matrix,
    then trunk layers from the END) + KV + recurrent state + buffers + headroom fit in ``free_mib``. Counted
    conservatively: KV and recurrent state in full, the partial-offload extra. Everything fitting means -ngl
    99."""
    fixed = (int(c["AB_Q4_CTX"]) * int(c["AB_KV_BYTES_PER_TOKEN"]) / MIB + float(c["AB_RECURRENT_MIB"])
             + float(c["AB_BUFFERS_MIB"]))
    budget = free_mib - float(c["AB_HEADROOM_MIB"])
    full = (sum(layers["trunk"]) + layers["output"]) / MIB + fixed
    if full <= budget:
        return 99, full
    fixed += float(c["AB_PARTIAL_EXTRA_MIB"])
    best: tuple[int, float] = (0, fixed)
    for n in range(1, int(layers["blocks"]) + 1):
        proj = fixed + gpu_weight_bytes(layers, n) / MIB
        if proj > budget:
            break
        best = (n, proj)
    return best


def est_tps(layers: dict[str, Any], ngl: int, c: dict[str, str]) -> float:
    """Projected decode tok/s at ``-ngl``: one token reads every GPU-resident weight at AB_GPU_GBPS and every
    CPU-resident one (trunk + output; the embeddings are a row lookup) at AB_CPU_GBPS."""
    total = sum(layers["trunk"]) + int(layers["output"])
    gpu = total if ngl >= 99 else min(total, gpu_weight_bytes(layers, ngl))
    s = gpu / (float(c["AB_GPU_GBPS"]) * 1e9) + (total - gpu) / (float(c["AB_CPU_GBPS"]) * 1e9)
    return 1.0 / s if s > 0 else float("inf")


def q4_min_tps(c: dict[str, str]) -> float:
    v = c.get("AB_Q4_MIN_EST_TPS", "").strip()
    return float(v) if v else float(c["PROBE_MIN_TPS"])


# ── teacher_ab.py and the probe guard ────────────────────────────────────────


def last_json(text: str) -> dict[str, Any]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            with contextlib.suppress(ValueError):
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
    return {}


class Ab:
    """teacher_ab.py on the host (AB_PY, PYTHONPATH=AB_CODE/src), one child at a time."""

    def __init__(self, c: dict[str, str], out: Path, log: Callable[[str], None], clock: Callable[[], float],
                 sleep: Callable[[float], None]) -> None:
        self.c, self.out, self.log, self.clock, self.sleep = c, out, log, clock, sleep
        self.script = Path(c["AB_CODE"]) / "scripts" / "distill" / "teacher_ab.py"
        self.proc: subprocess.Popen[str] | None = None

    def argv(self, *args: str) -> list[str]:
        return shlex.split(self.c["AB_NICE"]) + [self.c["AB_PY"], str(self.script), *args]

    def env(self) -> dict[str, str]:
        return dict(os.environ, PYTHONPATH=str(Path(self.c["AB_CODE"]) / "src"))

    def call(self, *args: str, timeout: float = 900) -> tuple[int, str, str]:
        try:
            cp = subprocess.run(self.argv(*args), capture_output=True, text=True, timeout=timeout, env=self.env(),
                                cwd=str(Path(self.c["AB_CODE"])))
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
        except OSError as e:
            return 127, "", str(e)
        return cp.returncode, cp.stdout, cp.stderr

    def select_args(self) -> list[str]:
        c = self.c
        a = ["select", "--out", str(self.out), "--tasks", c["AB_TASKS"], "--n", c["AB_N"], "--seed", c["AB_SEED"],
             "--repo", c["GEN_REPO"], "--temperature", c["AB_TEMPERATURE"], "--top-k", c["AB_TOP_K"],
             "--top-p", c["AB_TOP_P"], "--min-p", c["AB_MIN_P"], "--max-tokens", c["AB_MAX_TOKENS"],
             "--concurrency", c["AB_CONCURRENCY"], "--request-timeout", c["AB_REQUEST_TIMEOUT"],
             "--task-timeout", c["AB_TASK_TIMEOUT"], "--min-task-seconds", c["AB_MIN_TASK_S"],
             "--label-a", LABELS["A"], "--label-b", c["AB_REMOTE_LABEL_B"] if arm_b_remote(c) else LABELS["B"],
             "--allow-ctx-difference"]
        return a + (["--pack", c["GEN_PACK"]] if c.get("GEN_PACK") else [])

    def select(self) -> dict[str, Any]:
        rc, out, err = self.call(*self.select_args())
        self.log(f"teacher_ab select exited {rc}: {(out + err).strip()[-400:]}")
        if rc != 0:
            raise Abort(f"teacher_ab select failed (exit {rc}): {err.strip()[-200:]}")
        return json.loads(out) if out.strip().startswith("{") else {}

    def status(self) -> dict[str, Any]:
        rc, out, err = self.call("status", "--out", str(self.out), timeout=120)
        st = last_json(out)
        if rc != 0 or not st:
            raise Abort(f"teacher_ab status failed (exit {rc}): {err.strip()[-200:]}")
        return st

    def report(self) -> tuple[int, list[str]]:
        rc, out, err = self.call("report", "--out", str(self.out), timeout=300)
        path = self.out / "ab_summary.txt"
        lines = path.read_text().splitlines() if rc in (0, 1) and path.exists() else []
        if not lines:
            self.log(f"teacher_ab report exited {rc}: {err.strip()[-200:]}")
        return rc, lines

    def step(self, args: list[str], watch: Callable[[], str | None]) -> tuple[int, dict[str, Any], str]:
        """One long step (arm / judge) as a child in its own process group, polled. ``watch`` returns a reason
        to stop it (SIGTERM to the group, SIGKILL after AB_STEP_GRACE_S). Returns (rc, its JSON, reason)."""
        argv = self.argv(*args)
        logf = Path(self.c["LOG_DIR"]) / f"ab-{args[0]}-{datetime.now().strftime('%F')}.log"
        self.log(f"start: {shlex.join(argv)}")
        with open(logf, "a") as err, tempfile.TemporaryFile("w+") as out:
            self.proc = subprocess.Popen(argv, stdout=out, stderr=err, text=True, env=self.env(),
                                         cwd=str(Path(self.c["AB_CODE"])), start_new_session=True)
            why = ""
            try:
                while self.proc.poll() is None:
                    why = watch() or ""
                    if why:
                        self.log(f"stopping {args[0]}: {why}")
                        self.stop()
                        break
                    self.sleep(float(self.c["AB_POLL_S"]))
            finally:
                if self.proc.poll() is None:
                    self.stop()
            rc = self.proc.returncode
            self.proc = None
            out.seek(0)
            res = last_json(out.read())
        self.log(f"{args[0]} exited {rc}: {json.dumps(res)}")
        return rc, res, why

    def stop(self) -> None:
        p = self.proc
        if p is None or p.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=float(self.c["AB_STEP_GRACE_S"]))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=30)


class FakeAb(Ab):
    """select/status/report for real in a scratch directory; arms and the judge simulated on the clock."""

    def __init__(self, *a: Any, error_at: str = "", **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.done = {"arm-A": 0, "arm-B": 0, "judge": 0}
        self.n = 0
        self.error_at = error_at
        self.task_s = dict(zip(("arm-A", "arm-B", "judge"), (float(x) for x in self.c["AB_FAKE_TASK_S"].split())))
        self.urls: list[str] = []

    def select(self) -> dict[str, Any]:
        if self.error_at == "select":
            raise RuntimeError("fake error in select")
        res = super().select()
        self.n = int(res.get("n") or 0)
        return res

    def status(self) -> dict[str, Any]:
        st = super().status()  # the real one runs (and must parse); the counts are the simulation's
        n = self.n or int(st.get("selected") or 0)
        return {"selected": n, "arm-A": {"done": self.done["arm-A"], "total": n},
                "arm-B": {"done": self.done["arm-B"], "total": n},
                "judge": {"done": self.done["judge"], "total": self.done["arm-A"] + self.done["arm-B"],
                          "arms_complete": self.done["arm-A"] == n and self.done["arm-B"] == n}}

    def step(self, args: list[str], watch: Callable[[], str | None]) -> tuple[int, dict[str, Any], str]:
        kind = f"arm-{args[args.index('--arm') + 1]}" if args[0] == "arm" else "judge"
        url = args[args.index("--url" if args[0] == "arm" else "--judge-url") + 1]
        until = datetime.fromisoformat(args[args.index("--until") + 1]).timestamp()
        self.urls.append(url)
        self.log(f"[fake] {shlex.join(self.argv(*args))}")
        if self.error_at == kind:
            raise RuntimeError(f"fake error in {kind}")
        total = self.n if kind != "judge" else self.done["arm-A"] + self.done["arm-B"]
        margin = float(self.c["AB_MIN_TASK_S"]) if kind != "judge" else 30.0
        while self.done[kind] < total:
            if self.clock() + margin > until:
                break
            why = watch()
            if why:
                return 1, {"stop": "watch"}, why
            if self.clock() + self.task_s[kind] > until:  # still running at --until: cancelled, not recorded
                self.sleep(until - self.clock())
                break
            self.sleep(self.task_s[kind])
            self.done[kind] += 1
        left = total - self.done[kind]
        self.log(f"[fake] {kind}: {self.done[kind]}/{total} done at {iso_et(self.clock(), ZoneInfo(self.c['ET_TZ']))}")
        return (0 if not left else 1), {"left": left, "stop": "done" if not left else "until"}, ""


class Guard:
    """This run's probe guard (gen_runner's guard CLI) on AB_GUARD_LISTEN in front of the model port."""

    def __init__(self, c: dict[str, str], w: MW.Window, deadline: float, fake: bool) -> None:
        self.c, self.w, self.deadline, self.fake = c, w, deadline, fake
        self.proc: subprocess.Popen[bytes] | None = None
        self.url = "http://" + c["AB_GUARD_LISTEN"]
        self.abort_file = w.state / "ab-guard.abort"
        self.pidfile = w.state / "ab-guard.pid"

    def argv(self, llama_unit: str) -> list[str]:
        c = self.c
        return [c["PY"], "-I", c["GEN_GUARD_PY"], "--listen", c["AB_GUARD_LISTEN"],
                "--upstream", GR.host_port(c["MODEL_URL"]), "--probe-unit", c["PROBE_UNIT"],
                "--probe-name", c["PROBE_NAME"], "--llama-unit", llama_unit,
                "--log", str(Path(c["LOG_DIR"]) / "ab-probe_guard.log"), "--pidfile", str(self.pidfile),
                "--deadline", str(int(self.deadline)), "--abort-file", str(self.abort_file),
                "--notify-cmd", c["NOTIFY"], "--min-tps", c["GEN_GUARD_MIN_TPS"] or c["PROBE_MIN_TPS"]]

    def start(self, llama_unit: str, alias: str) -> None:
        argv = self.argv(llama_unit)
        if self.fake:
            self.w.log(f"[dry run] would start the probe guard: {shlex.join(argv)}")
            return
        if self.running():
            return
        if not GR.port_free(self.c["AB_GUARD_LISTEN"]):
            raise Abort(f"{self.c['AB_GUARD_LISTEN']} is in use: not starting a second guard")
        if self.abort_file.exists():
            os.replace(self.abort_file, self.abort_file.with_suffix(".prev"))
        logf = open(Path(self.c["LOG_DIR"]) / "ab-probe_guard.out", "ab")
        self.proc = subprocess.Popen(argv, stdout=logf, stderr=logf, start_new_session=True)
        logf.close()
        self.w.log(f"probe guard pid {self.proc.pid}: {shlex.join(argv)}")
        deadline = time.monotonic() + float(self.c["AB_GUARD_WAIT_S"])
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise Abort(f"the probe guard exited at start (exit {self.proc.returncode})")
            st, body = self.w.http(self.url + "/v1/models")
            ids = [str(m.get("id")) for m in (body or {}).get("data", []) if isinstance(m, dict)] \
                if st == 200 and isinstance(body, dict) else []
            if alias in ids:
                self.w.log(f"probe guard up on {self.c['AB_GUARD_LISTEN']}: serves {alias}")
                return
            time.sleep(0.2)
        self.stop()
        raise Abort(f"the probe guard did not answer with {alias} within {self.c['AB_GUARD_WAIT_S']} s")

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def aborted(self) -> str | None:
        if self.fake or not self.running():
            return None if self.fake or self.proc is None else f"the probe guard is gone (exit {self.proc.returncode})"
        if self.abort_file.exists():
            return "the probe guard's abort switch tripped: " + self.abort_file.read_text().strip()[:160]
        return None

    def stop(self) -> None:
        if self.fake:
            self.w.log("[dry run] would stop the probe guard (its recorded PID)")
            return
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=15)
            self.w.log(f"probe guard pid {p.pid} stopped")
        self.proc = None

    def stop_recorded(self) -> None:
        """A guard a dead run left (restore): only the PID in its own pidfile, only if it is that guard."""
        try:
            pid = int(self.pidfile.read_text().split()[0])
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError, IndexError):
            return
        if "probe_guard.py" in cmd and self.c["AB_GUARD_LISTEN"] in cmd:
            if not self.w.execute:
                self.w.log(f"[dry run] would stop the A/B probe guard pid {pid}")
                return
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
            self.w.log(f"stopped the A/B probe guard pid {pid} (its pidfile)")


# ── the night ────────────────────────────────────────────────────────────────


class Night:
    def __init__(self, w: MW.Window, c: dict[str, str], ab: Ab, guard: Guard, times: dict[str, Any], *,
                 fake: bool, fake_probe_fail: bool = False) -> None:
        self.w, self.c, self.ab, self.guard, self.t = w, c, ab, guard, times
        self.fake = fake
        self.fake_probe_fail = fake_probe_fail
        self.tz = ZoneInfo(c["ET_TZ"])
        self.on_q4 = False
        self.business = False  # the business restore ran (the IQ3 teacher would not come back)
        self.q4_refused = ""
        self.gen_stopped: list[str] = []
        self.gen_resume = ""
        self.events: list[str] = []
        self.layers: dict[str, Any] | None = None
        self.state: dict[str, Any] = {}

    # small helpers

    def log(self, msg: str) -> None:
        self.w.log(msg)

    def event(self, msg: str) -> None:
        self.events.append(msg)
        self.log(msg)

    def now(self) -> float:
        return self.w.now()

    def et(self, t: float | None = None) -> str:
        return self.w.et(t).strftime("%H:%M %Z")

    def save(self, **kw: Any) -> None:
        self.state.update(kw)
        if self.w.execute:
            path = self.w.state / STATE_NAME
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1) + "\n")
            os.replace(tmp, path)

    def gpu(self) -> tuple[int | None, dict[int, int]]:
        """(free MiB of GPU 0, {pid: MiB} of its compute processes): nvidia-smi, read-only."""
        cp = self.w.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], timeout=30)
        free = None
        with contextlib.suppress(ValueError, IndexError):
            free = int(cp.stdout.split()[0])
        cp = self.w.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                        timeout=30)
        procs: dict[int, int] = {}
        for line in cp.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            with contextlib.suppress(ValueError, IndexError):
                procs[int(parts[0])] = int(parts[1])
        return free, procs

    def predicted_free(self) -> int | None:
        """Free VRAM once the IQ3 teacher stops: free now + what its process holds."""
        free, procs = self.gpu()
        pid = self.w.unit_prop(self.c["TEACHER_UNIT"], "MainPID")
        held = procs.get(int(pid), 0) if pid.isdigit() else 0
        return None if free is None else free + held

    def ngl_for(self, free: int) -> tuple[int, float]:
        if self.layers is None:
            self.layers = gguf_layers(Path(self.c["AB_Q4_MODEL"]))
        return plan_ngl(self.layers, free, self.c)

    def too_slow(self, ngl: int) -> str:
        """Why a Q4 server at ``-ngl`` would fail the voice probe's floor (projected), or ""."""
        floor = q4_min_tps(self.c)
        if floor <= 0 or self.layers is None:
            return ""
        tps = est_tps(self.layers, ngl, self.c)
        if tps >= floor:
            return ""
        return (f"Q4_K_M at -ngl {ngl} projects ~{tps:.1f} tok/s decode, under the voice probe's floor "
                f"{floor:g}: no swap (AB_Q4_MIN_EST_TPS=0 lets the probe decide)")

    def serving(self, alias: str) -> bool:
        s = self.w.served()
        return s["health"] == 200 and alias in s["ids"]

    def unit_state(self, unit: str) -> str:
        return self.w.unit_prop(unit, "ActiveState")

    def probe_now_ok(self, max_age_s: float = 660) -> tuple[bool, str]:
        ps = self.w.probes()
        if not ps:
            return False, "no voice-probe result in 15 min"
        p = ps[-1]
        age = self.now() - p.t
        ok = p.ok(float(self.c["PROBE_MIN_TPS"])) and (age <= max_age_s or self.fake)
        return ok, f"last voice probe {p.describe()} ({age:.0f} s ago)"

    # checks

    def preflight(self) -> None:
        """Read-only. Raises MW.Skip (a real run changes nothing then); FAKE logs it and goes on."""
        c, w = self.c, self.w
        why: list[str] = []
        left = (self.t["hard"] - self.now()) / 60
        if left < float(c["AB_MIN_START_LEFT_MIN"]):
            why.append(f"only {left:.0f} min before the hard end {self.et(self.t['hard'])}")
        if w.flag.exists():
            why.append(f"{w.flag} is present")
        if not w.active(c["TEACHER_UNIT"]) or not self.serving(c["TEACHER_ALIAS"]):
            why.append(f"{c['TEACHER_UNIT']} is not serving {c['TEACHER_ALIAS']} on {c['MODEL_URL']}")
        remote_b = arm_b_remote(c)
        if remote_b:
            # arm B and the judge run on the remote teacher (teacher_ab.py with the remote flags): this night runs
            # arm A only and never swaps the local model port
            if not self.q4_refused:
                self.q4_refused = "arm B runs on the remote teacher (AB_ARM_B=remote), not on this host"
                self.event(f"no Q4 tonight: {self.q4_refused}")
        elif w.unit_prop(c["AB_Q4_UNIT"], "LoadState") != "loaded":
            why.append(f"{c['AB_Q4_UNIT']} is not installed (install_teacher_units.sh --only ab)")
        elif w.active(c["AB_Q4_UNIT"]):
            why.append(f"{c['AB_Q4_UNIT']} is already running")
        for key in (("AB_TASKS", "GEN_GUARD_PY") if remote_b else ("AB_Q4_MODEL", "AB_TASKS", "GEN_GUARD_PY")):
            if not Path(c[key]).is_file():
                why.append(f"missing {c[key]} ({key})")
        if not self.ab.script.is_file() or not (Path(c["AB_CODE"]) / "src").is_dir():
            why.append(f"teacher_ab.py not staged in {c['AB_CODE']} (gen_runner.py stage-code --dest)")
        for key in ("AB_PY", "NOTIFY"):
            if not os.access(c[key], os.X_OK):
                why.append(f"{c[key]} ({key}) is not executable")
        if self.ab.script.is_file() and os.access(c["AB_PY"], os.X_OK):
            rc, out, _ = self.ab.call("select", "--help", timeout=180)
            if rc != 0 or "--allow-ctx-difference" not in out:
                why.append(f"the staged teacher_ab.py has no --allow-ctx-difference (exit {rc}): stage the code")
        ok, line = self.probe_now_ok()
        if not ok:
            why.append(line)
        ok, line = w.daemon_ok()
        if not ok:
            why.append(line)
        if w.eval_running():
            why.append("a nightly eval is running")
        try:
            if remote_b:
                raise _NoLocalArmB
            free = self.predicted_free()
            if free is None:
                why.append("nvidia-smi did not answer")
            else:
                ngl, proj = self.ngl_for(free)
                self.log(f"VRAM: ~{free} MiB free once {c['TEACHER_UNIT']} stops -> -ngl {ngl} "
                         f"(projected {proj:.0f} MiB + {c['AB_HEADROOM_MIB']} headroom, "
                         f"~{est_tps(self.layers or {}, ngl, c):.1f} tok/s projected)")
                if ngl < int(c["AB_MIN_NGL"]):
                    why.append(f"only -ngl {ngl} would fit (< AB_MIN_NGL {c['AB_MIN_NGL']})")
                slow = self.too_slow(ngl)
                if slow and not self.q4_refused:
                    # arm A still runs (its records keep for a night the Q4 can serve); 8092 is never swapped
                    self.q4_refused = slow
                    self.event(f"no Q4 tonight: {slow}")
        except _NoLocalArmB:
            pass
        except (OSError, ValueError) as e:
            why.append(f"cannot read {c['AB_Q4_MODEL']}: {e}")
        pidf = w.state / PID_NAME
        with contextlib.suppress(OSError, ValueError, IndexError):
            other = int(pidf.read_text().split()[0])
            if other != os.getpid() and Path(f"/proc/{other}").exists():
                why.append(f"another A/B run (pid {other}) is alive")
        self.log("evals tonight: " + (", ".join(self.et(e) for e in self.t["evals"]) or "none")
                 + f"; hard end {self.et(self.t['hard'])}")
        if why:
            if not self.fake:
                raise MW.Skip("; ".join(why))
            self.event("PRECHECK (a real run would SKIP here, nothing changed): " + "; ".join(why))

    # generation

    def stop_gen(self) -> None:
        c = self.c
        running = sorted(u for u, s in self.w.units_matching(c["GEN_UNIT_GLOB"], "service").items() if s in MW.ACTIVE)
        records = [{"unit": u, "exec_start": self.w.unit_prop(u, "ExecStart")} for u in running]
        self.gen_stopped = MW.stop_gen_units(self.w)
        self.gen_resume = self.gen_resume_command()
        for r in records:
            self.log(f"generation paused: {r['unit']} (ExecStart {r['exec_start'][:220]})")
        self.log("generation resumes with: " + (self.gen_resume or ("(not captured: mini_window.start_gen decides)"
                                                                if self.gen_stopped else "(none was running)")))
        self.save(gen_stopped=records, gen_resume=self.gen_resume, gen_resumed=False)

    def gen_resume_command(self) -> str:
        """The exact command mini_window.start_gen runs (captured from its own dry run; the unit name's stamp is
        the time of the restart)."""
        if not self.gen_stopped:
            return ""
        lines: list[str] = []
        dry = MW.Window(self.c, False, "ab-night", runner=self.w._runner, clock=self.w.clock, sleep=self.w._sleep,
                        out=lines.append)
        dry.units_matching = lambda glob, kind="service": {}  # type: ignore[method-assign]
        MW.start_gen(dry, "after the teacher A/B")
        return next((ln.split("would run: ", 1)[1] for ln in lines if "would run: " in ln), "")

    def resume_gen(self) -> str:
        if not self.gen_stopped:
            return "none was running before the A/B"
        if not self.fake and not self.serving(self.c["TEACHER_ALIAS"]):
            return f"NOT restarted: {self.c['MODEL_URL']} does not serve the IQ3 teacher"
        res = MW.start_gen(self.w, "after the teacher A/B")
        self.save(gen_resumed=res)
        return res

    # swaps

    def wait_unit_gone(self, unit: str, wait_s: float = 120) -> None:
        if not self.w.execute:
            return
        deadline = self.now() + wait_s
        while self.unit_state(unit) in ("active", "activating", "deactivating", "reloading"):
            if self.now() >= deadline:
                raise MW.Fail(f"{unit} still {self.unit_state(unit)} {wait_s:.0f} s after its stop")
            self.w.sleep(2)

    def wait_up(self, unit: str, alias: str) -> float:
        """Until the port serves ``alias``; a unit that fails or stops on the way fails at once."""
        if not self.w.execute:
            self.log(f"[dry run] would wait up to {self.c['TEACHER_WAIT_S']} s for {alias} on {self.c['MODEL_URL']}")
            return self.now()
        deadline = self.now() + float(self.c["TEACHER_WAIT_S"])
        while True:
            if self.serving(alias):
                self.log(f"{self.c['MODEL_URL']} healthy, serving {alias}")
                return self.now()
            st = self.unit_state(unit)
            if st in ("failed", "inactive"):
                raise MW.Fail(f"{unit} is {st} while loading (journalctl --user -u {unit})")
            if self.now() >= deadline:
                raise MW.Fail(f"{alias} not served {self.c['TEACHER_WAIT_S']} s after the start")
            self.w.sleep(5)

    def wait_free(self, want: float, wait_s: float = 60) -> int | None:
        if not self.w.execute:
            return None
        deadline = self.now() + wait_s
        while True:
            free, _ = self.gpu()
            if free is not None and (free >= want or self.now() >= deadline):
                return free
            if self.now() >= deadline:
                return free
            self.w.sleep(3)

    def to_q4(self, until: float) -> None:
        """IQ3 -> Q4 partial (step 5). Any failure: IQ3 back at once, then Q4Refused."""
        c, w = self.c, self.w
        pre = self.predicted_free()
        if pre is None:
            raise Q4Refused("nvidia-smi did not answer")
        ngl, proj = self.ngl_for(pre)
        if ngl < int(c["AB_MIN_NGL"]):
            raise Q4Refused(f"only -ngl {ngl} would fit in ~{pre} MiB (AB_MIN_NGL {c['AB_MIN_NGL']}); no swap")
        slow = self.too_slow(ngl)
        if slow:
            raise Q4Refused(slow)
        with w.lock(300) as got:
            if not got:
                raise Q4Refused("another window script holds the lock; no swap")
            if w.flag.exists():
                raise Q4Refused(f"{w.flag} is present; no swap")
            w.wait_tick()
            if self.now() + 60 * float(c["AB_MIN_PHASE_MIN"]) > until:
                raise Q4Refused("no time left for a Q4 phase after the probe tick; no swap")
            try:
                self.guard.stop()
                self.on_q4 = True  # from here on the cleanup swaps back, whatever happens
                self.save(on_q4=True)
                w.act(["systemctl", "--user", "stop", c["TEACHER_UNIT"]], timeout=150)
                self.wait_unit_gone(c["TEACHER_UNIT"])
                free = self.wait_free(pre - 300) if w.execute else pre
                if free is None:
                    raise MW.Fail("nvidia-smi did not answer after the IQ3 teacher stopped")
                ngl, proj = self.ngl_for(free)
                need = int(proj + float(c["AB_HEADROOM_MIB"]) + 0.5)
                self.log(f"VRAM free after the stop: {free} MiB -> -ngl {ngl}, projected {proj:.0f} MiB, "
                         f"MIN_FREE_MIB {need}")
                if ngl < int(c["AB_MIN_NGL"]):
                    raise MW.Fail(f"only -ngl {ngl} fits in {free} MiB free")
                slow = self.too_slow(ngl)
                if slow:
                    raise MW.Fail(slow)
                env = f"NGL={ngl}\nMIN_FREE_MIB={need}\n"
                if w.execute:
                    Path(c["AB_Q4_ENV"]).write_text(env)
                else:
                    w.log(f"[dry run] would write {c['AB_Q4_ENV']}: {env.strip().replace(chr(10), ' ')}")
                w.act(["systemctl", "--user", "reset-failed", c["AB_Q4_UNIT"]], timeout=30)
                cp = w.act(["systemctl", "--user", "start", c["AB_Q4_UNIT"]], timeout=180)
                if cp.returncode != 0:
                    raise MW.Fail(f"systemctl start {c['AB_Q4_UNIT']} exited {cp.returncode}: "
                                  f"{(cp.stderr or cp.stdout).strip()[:200]}")
                t_up = self.wait_up(c["AB_Q4_UNIT"], c["AB_Q4_ALIAS"])
                if w.execute:
                    free_after, _ = self.gpu()
                    self.log(f"VRAM free beside the Q4 teacher: {free_after} MiB")
                    if free_after is not None and free_after < int(c["AB_POST_MIN_FREE_MIB"]):
                        raise MW.Fail(f"only {free_after} MiB VRAM left beside the Q4 teacher "
                                      f"(< {c['AB_POST_MIN_FREE_MIB']})")
                if self.fake and self.fake_probe_fail:
                    raise MW.Fail("voice probe after the change: SLOW 8.6 tok/s (fake)")
                p = w.wait_probe_ok(t_up, float(c["PROBE_WAIT_S"]))
                ok, daemon = w.daemon_ok()
                self.event(f"Q4 teacher up (-ngl {ngl}): voice probe {p.describe()}, {daemon}")
                self.save(q4_ngl=ngl, q4_min_free_mib=need)
                self.guard.start(c["AB_Q4_UNIT"], c["AB_Q4_ALIAS"])
            except (MW.Fail, MW.Skip, Abort, OSError, ValueError) as e:
                self.log(f"Q4 swap failed ({e}): back to IQ3 now")
                ok, line = self.to_iq3(urgent=True, locked=True)
                if not ok:
                    raise Abort(f"Q4 swap failed ({e}) and the IQ3 teacher did not come back: {line}") from e
                raise Q4Refused(str(e)) from e

    def to_iq3(self, urgent: bool = False, locked: bool = False) -> tuple[bool, str]:
        """Q4 partial -> IQ3 teacher, verified (healthy, the next voice probe OK, daemon /health). Two attempts,
        then mini_window's business restore. Returns (IQ3 back, a line)."""
        c, w = self.c, self.w
        lock = contextlib.nullcontext(True) if locked else w.lock(120 if urgent else 300)
        with lock as got:
            if not got:
                self.log("the window lock is held: swapping back anyway")
            if not urgent:
                w.wait_tick()
            self.guard.stop()
            line = ""
            for attempt in (1, 2):
                try:
                    w.act(["systemctl", "--user", "stop", c["AB_Q4_UNIT"]], timeout=150)
                    self.wait_unit_gone(c["AB_Q4_UNIT"])
                    if w.execute:
                        self.wait_free(float(c["AB_IQ3_MIN_FREE_MIB"]), 60)
                    w.act(["systemctl", "--user", "reset-failed", c["TEACHER_UNIT"]], timeout=30)
                    cp = w.act(["systemctl", "--user", "start", c["TEACHER_UNIT"]], timeout=180)
                    if cp.returncode != 0:
                        raise MW.Fail(f"systemctl start {c['TEACHER_UNIT']} exited {cp.returncode}")
                    t_up = self.wait_up(c["TEACHER_UNIT"], c["TEACHER_ALIAS"])
                    self.on_q4 = False
                    self.save(on_q4=False)
                    p = w.wait_probe_ok(t_up, float(c["PROBE_WAIT_S"]))
                    ok, daemon = w.daemon_ok()
                    line = f"IQ3 teacher back: voice probe {p.describe()}, {daemon}"
                    self.event(line)
                    return True, line
                except MW.Fail as e:
                    line = str(e)
                    self.log(f"swap back, attempt {attempt}/2: {line}")
                    w.sleep(15)
            # the teacher will not come back: the business layout (the daemon's own model server) instead
            self.event(f"IQ3 teacher NOT back ({line}): restoring business")
            ok, rline = w.restore()
            self.on_q4 = False
            self.business = True
            if not ok:
                w.write_flag("teacher A/B: swap back and restore failed")
            else:
                w.write_flag("teacher A/B: the IQ3 teacher did not come back (business restored)")
            return False, rline

    def iq3_verified(self) -> tuple[bool, str]:
        """The final check when no swap is needed: IQ3 healthy, a recent OK voice probe, daemon /health."""
        c = self.c
        if not self.fake and (not self.w.active(c["TEACHER_UNIT"]) or not self.serving(c["TEACHER_ALIAS"])):
            return False, f"{c['TEACHER_UNIT']} not serving {c['TEACHER_ALIAS']}"
        ok_p, pline = self.probe_now_ok()
        ok_d, dline = self.w.daemon_ok()
        return ok_p and ok_d, f"IQ3 teacher serving; {pline}; {dline}"

    # steps

    def watch(self, until: float) -> Callable[[], str | None]:
        c = self.c
        misses = {"port": 0}

        def check() -> str | None:
            """A reason to stop the step, prefixed by its kind: time:, eval:, flag: or server:."""
            if self.now() >= until + 120:
                return "time: past the phase end"
            if self.now() >= self.t["hard"]:
                return "time: the hard end"
            if self.w.flag.exists():
                return "flag: the NO-SWAPS flag"
            g = self.guard.aborted()
            if g:
                return "server: " + g
            if self.fake:
                return None
            if self.on_q4 and self.unit_state(c["AB_Q4_UNIT"]) != "active":
                return f"server: {c['AB_Q4_UNIT']} is {self.unit_state(c['AB_Q4_UNIT'])}"
            if self.w.eval_running():
                return "eval: a nightly eval started"
            alias = c["AB_Q4_ALIAS"] if self.on_q4 else c["TEACHER_ALIAS"]
            misses["port"] = 0 if self.serving(alias) else misses["port"] + 1
            if misses["port"] >= 6:
                return f"server: {c['MODEL_URL']} has not served {alias} for 6 polls"
            return None

        return check

    def ensure_guard(self) -> None:
        """This run's guard in front of whichever teacher serves now (it is stopped around every swap)."""
        if self.fake or self.guard.running():
            return
        unit, alias = ((self.c["AB_Q4_UNIT"], self.c["AB_Q4_ALIAS"]) if self.on_q4
                       else (self.c["TEACHER_UNIT"], self.c["TEACHER_ALIAS"]))
        self.guard.start(unit, alias)

    def step(self, kind: str, until: float) -> None:
        """One teacher_ab step (``arm-A``, ``arm-B``, ``judge``) through the guard, with reruns."""
        iso = iso_et(until, self.tz)
        args = (["arm", "--out", str(self.ab.out), "--arm", kind[-1], "--url", self.guard.url, "--until", iso]
                if kind.startswith("arm") else
                ["judge", "--out", str(self.ab.out), "--judge-url", self.guard.url, "--until", iso])
        for attempt in range(int(self.c["AB_STEP_RETRIES"]) + 1):
            self.ensure_guard()
            rc, res, why = self.ab.step(args, self.watch(until))
            stop = str(res.get("stop") or "")
            if why.startswith("flag:"):
                raise Abort(f"{kind} stopped: {why[6:]}")
            if why.startswith("server:"):
                if self.on_q4:
                    raise Q4Refused(f"{kind} stopped: {why[8:]}")
                raise Abort(f"{kind} stopped: {why[8:]}")
            if why:  # time: or eval:
                return
            if rc == 0 or stop == "until":
                return
            if rc == 2:
                raise Abort(f"teacher_ab {kind} refused its input (exit 2): see the log")
            if self.now() + 60 * float(self.c["AB_MIN_PHASE_MIN"]) > until:
                return
            if rc == 3 and attempt >= 1:
                if self.on_q4:
                    raise Q4Refused(f"{kind}: the Q4 server failed (exit 3) twice")
                raise Abort(f"{kind}: the IQ3 server failed (exit 3) twice")
            if attempt < int(self.c["AB_STEP_RETRIES"]):
                self.log(f"{kind} left work (exit {rc}, {stop or 'no result'}); rerunning in "
                         f"{self.c['AB_RETRY_PAUSE_S']} s")
                self.w.sleep(float(self.c["AB_RETRY_PAUSE_S"]))

    def wait_until(self, t: float, why: str) -> None:
        self.log(f"waiting until {self.et(t)}: {why}")
        while self.now() < t:
            if self.w.flag.exists():
                raise Abort("the NO-SWAPS flag appeared")
            self.w.sleep(min(60.0, t - self.now()))

    def loop(self) -> str:
        """Arm A on IQ3, then arm B and the judge on Q4, around the nightly evals, until done or the hard end.
        Returns why it ended."""
        c = self.c
        hard = self.t["hard"]
        q4_gap = 60 * float(c["AB_Q4_EVAL_GAP_MIN"])
        iq3_gap = 60 * float(c["AB_IQ3_EVAL_GAP_MIN"])
        after = 60 * float(c["AB_EVAL_AFTER_MIN"])
        min_phase = 60 * float(c["AB_MIN_PHASE_MIN"])
        swap = 60 * float(c["AB_SWAP_MIN"])
        q4_end_cap = hard - 60 * float(c["AB_SWAPBACK_MARGIN_MIN"])
        while True:
            now = self.now()
            if now >= hard - min_phase:
                return "the time box"
            st = self.ab.status()
            n = int(st.get("selected") or 0)
            a_left = st["arm-A"]["done"] < n
            b_left = st["arm-B"]["done"] < n
            j = st["judge"]
            j_left = not (j.get("arms_complete") and j.get("done") == j.get("total"))
            if not (a_left or b_left or j_left):
                return "done"
            running = False if self.fake else self.w.eval_running()
            if a_left:
                hold = eval_free_at(now, self.t["evals"], iq3_gap, after, running)
                if hold is not None:
                    self.wait_until(min(hold, hard), "nightly eval (arm A pauses)")
                    continue
                until = min(hard, next_eval_hold(now, self.t["evals"], iq3_gap))
                if until - now < min_phase:
                    self.wait_until(min(until + 1, hard), "too little time before the next eval for arm A")
                    continue
                self.step("arm-A", until)
                continue
            if self.q4_refused:
                return f"no Q4 tonight: {self.q4_refused}"
            hold = eval_free_at(now, self.t["evals"], q4_gap, after, running)
            end = min(q4_end_cap, next_eval_hold(now, self.t["evals"], q4_gap))
            if hold is not None or end - now < min_phase + (0 if self.on_q4 else swap):
                if self.on_q4 and not running:
                    self.log("Q4 phase over (a nightly eval is close, or the hard end): back to IQ3")
                    ok, line = self.to_iq3()
                    if not ok:
                        raise Abort(line)
                    continue
                nxt = hold if hold is not None else end + q4_gap + after
                if nxt >= q4_end_cap - min_phase - swap:
                    return "no Q4 time left before the hard end"
                self.wait_until(nxt, "nightly eval (no Q4 phase now)")
                continue
            if not self.on_q4:
                try:
                    self.to_q4(end)
                except Q4Refused as e:
                    self.q4_refused = str(e)
                    self.event(f"no Q4 tonight: {e}")
                    continue
            try:
                self.step("arm-B" if b_left else "judge", end)
            except Q4Refused as e:
                self.q4_refused = str(e)
                self.event(f"Q4 stopped: {e}; back to IQ3")
                if self.on_q4:
                    ok, line = self.to_iq3(urgent=True)
                    if not ok:
                        raise Abort(line) from e

    # the whole night

    def run(self) -> int:
        c, w = self.c, self.w
        self.preflight()
        if w.execute:
            (w.state / PID_NAME).write_text(f"{os.getpid()}\n")
        self.save(started=iso_et(self.now(), self.tz), hard=iso_et(self.t["hard"], self.tz),
                  evals=[iso_et(e, self.tz) for e in self.t["evals"]], out=str(self.ab.out), finished=False)
        why = "?"
        failed = False
        try:
            self.stop_gen()
            if not self.fake and not GR.port_free(c["AB_GUARD_LISTEN"]):
                raise Abort(f"{c['AB_GUARD_LISTEN']} is still in use after the generation stopped")
            self.guard.start(c["TEACHER_UNIT"], c["TEACHER_ALIAS"])
            sel = self.ab.select()
            self.log(f"selection: {json.dumps(sel)[:400]}")
            why = self.loop()
        except KeyboardInterrupt as e:
            why, failed = f"stopped by {e or 'a signal'}", True
        except (Abort, MW.Fail, MW.Skip) as e:
            why, failed = f"aborted: {e}", True
        except Exception as e:  # noqa: BLE001 - whatever broke, the cleanup below still runs
            why, failed = f"failed: {type(e).__name__}: {e}", True
        finally:
            self.ab.stop()
            rc = self.finish(why, failed)
        return rc

    def finish(self, why: str, failed: bool) -> int:
        """Step 8: IQ3 back and verified, guard stopped, report, generation restarted, notify."""
        c, w = self.c, self.w
        self.event(f"night ends: {why}")
        try:
            q4_state = "inactive" if self.fake else self.unit_state(c["AB_Q4_UNIT"])
            needs_swap = self.on_q4 or q4_state not in ("inactive", "") or (
                not self.fake and not self.w.active(c["TEACHER_UNIT"]))
            if self.business or (not self.fake and self.w.active(c["LOCAL_MODEL_UNIT"])):
                if q4_state in MW.ACTIVE:
                    w.act(["systemctl", "--user", "stop", c["AB_Q4_UNIT"]], timeout=150)
                ok, teacher = False, f"business layout ({c['LOCAL_MODEL_UNIT']}): the IQ3 teacher is not back"
            elif needs_swap:
                ok, teacher = self.to_iq3(urgent=failed)
            else:
                ok, teacher = self.iq3_verified()
                self.log(f"IQ3 check: {teacher}")
        except BaseException as e:  # noqa: BLE001 - a second signal or a broken check: the business restore
            self.log(f"swap back raised {type(e).__name__}: {e}: restoring business")
            ok, teacher = w.restore()
            if not ok:
                w.write_flag("teacher A/B cleanup failed")
            ok = False
        self.guard.stop()
        rc_rep, lines = 2, list[str]()
        try:
            rc_rep, lines = self.ab.report()
        except Exception as e:  # noqa: BLE001
            self.log(f"report failed: {type(e).__name__}: {e}")
        gen = self.resume_gen() if ok else "NOT restarted (the IQ3 teacher is not back)"
        self.event(f"generation: {gen}")
        self.save(finished=True, ended=iso_et(self.now(), self.tz), why=why, teacher=teacher, generation=gen)
        status = f"night: {why}; {'IQ3 back OK' if ok else 'IQ3 NOT back'}; gen {gen.split(' (')[0][:60]}"
        if rc_rep == 0 and len(lines) >= 5:
            msg = lines[:5]
            if failed or self.q4_refused:
                msg[3] = status
        elif lines:
            msg = [lines[0], status, *lines[1:3], lines[4] if len(lines) > 4 else ""]
        else:
            msg = [f"Teacher A/B (mini): no report ({why[:120]})", status, f"teacher: {teacher[:200]}",
                   f"out: {self.ab.out}", ""]
        w.notify("\n".join(x for x in msg if x is not None))
        if w.execute:
            with contextlib.suppress(OSError):
                (w.state / PID_NAME).unlink()
        if not ok:
            return 1
        return 0 if rc_rep in (0, 1) and not failed and not self.q4_refused else 1


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_restore(night: Night) -> int:
    """OnFailure= of the Q4 unit and the backstop timer: the IQ3 teacher back when an A/B left it out."""
    c, w = night.c, night.w
    pidf = w.state / PID_NAME
    with contextlib.suppress(OSError, ValueError, IndexError):
        pid = int(pidf.read_text().split()[0])
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        if b"mini_ab.py" in cmd:
            w.log(f"the A/B run (pid {pid}) is alive: it swaps back on its own")
            return 0
    with contextlib.suppress(OSError, ValueError):
        night.state = json.loads((w.state / STATE_NAME).read_text())
    q4 = night.unit_state(c["AB_Q4_UNIT"])
    iq3 = w.active(c["TEACHER_UNIT"]) and night.serving(c["TEACHER_ALIAS"])
    night.guard.stop_recorded()
    if w.active(c["LOCAL_MODEL_UNIT"]):
        w.log(f"{c['LOCAL_MODEL_UNIT']} runs (business): no teacher to put back")
        if q4 in MW.ACTIVE:
            w.act(["systemctl", "--user", "stop", c["AB_Q4_UNIT"]], timeout=150)
        return 0
    did = ""
    if q4 in MW.ACTIVE or q4 == "failed" or not iq3:
        w.log(f"{c['AB_Q4_UNIT']} is {q4 or '?'}, IQ3 serving: {iq3}: putting the IQ3 teacher back")
        night.on_q4 = True
        ok, did = night.to_iq3(urgent=True)
        if not ok:
            w.notify(f"mini teacher A/B restore: the IQ3 teacher did not come back.\n{did}")
            return 1
    stopped = [s["unit"] for s in night.state.get("gen_stopped") or []]
    gen = ""
    if stopped and not night.state.get("finished"):
        running = [u for u, s in w.units_matching(c["GEN_UNIT_GLOB"], "service").items() if s in MW.ACTIVE]
        night.gen_stopped = stopped
        gen = f"already running ({', '.join(running)})" if running else night.resume_gen()
        night.save(finished=True, why="restore after an A/B run that did not finish", generation=gen)
    if did or gen:
        w.notify(f"mini teacher A/B restore: {did or 'IQ3 teacher was serving'}\ngeneration: {gen or 'not touched'}")
    else:
        w.log("the IQ3 teacher serves and no A/B left anything to undo: nothing to do")
    return 0


def cmd_ngl(night: Night) -> int:
    free = night.predicted_free()
    if free is None:
        print("nvidia-smi did not answer")
        return 1
    ngl, proj = night.ngl_for(free)
    layers = night.layers or {}
    print(json.dumps({"free_mib_after_iq3_stop": free, "ngl": ngl, "projected_mib": round(proj),
                      "min_free_mib": round(proj + float(night.c["AB_HEADROOM_MIB"])),
                      "trunk_layers": len(layers.get("trunk", [])), "min_ngl": int(night.c["AB_MIN_NGL"]),
                      "est_tok_s": round(est_tps(layers, ngl, night.c), 1), "min_tok_s": q4_min_tps(night.c),
                      "too_slow": night.too_slow(ngl) or None}))
    return 0


def cmd_arm(w: MW.Window, c: dict[str, str], day: date, at: str, replace: bool) -> int:
    """The one-shot timers for one night: the A/B at ``at`` ET (its first occurrence) and a backstop restore."""
    tz = ZoneInfo(c["ET_TZ"])
    start = at_et(day, at, tz)
    times = night_times(c, start)
    backstop = at_et(day, c["AB_BACKSTOP_HHMM"], tz)
    if backstop <= times["hard"]:
        backstop += 86400
    if start <= w.now():
        raise MW.Skip(f"{iso_et(start, tz)} has passed")
    unit = f"{c['AB_UNIT_PREFIX']}-{day.strftime('%Y%m%d')}"
    back = f"{c['AB_UNIT_PREFIX']}-backstop-{day.strftime('%Y%m%d')}"
    script = str(Path(c["WIN_DIR"]) / "mini_ab_tonight.sh")
    runtime = int(times["hard"] + 40 * 60 - start)
    for u in (unit, back):
        if w.unit_prop(u + ".timer", "LoadState") == "loaded" and w.unit_prop(u + ".timer", "ActiveState") == "active":
            if not replace:
                raise MW.Skip(f"{u}.timer is already armed (--replace to re-arm)")
            w.act(["systemctl", "--user", "stop", u + ".timer"], timeout=30)
        w.act(["systemctl", "--user", "reset-failed", u + ".timer", u + ".service"], timeout=30)
    common = ["--timer-property=AccuracySec=1s", "-p", "KillMode=mixed", "-p", "TimeoutStopSec=20min",
              "-p", f"WorkingDirectory={Path(c['WIN_DIR']).parent}"]
    runs = [
        ["systemd-run", "--user", f"--unit={unit}", f"--on-calendar={calendar_utc(start)}", *common,
         "-p", f"RuntimeMaxSec={runtime}",
         f"--description=distill teacher A/B night {day} (IQ3 vs Q4_K_M partial; mini_ab.py)", script, "--execute"],
        ["systemd-run", "--user", f"--unit={back}", f"--on-calendar={calendar_utc(backstop)}", *common,
         f"--description=distill teacher A/B backstop {day} (IQ3 teacher back if left out)", script, "restore",
         "--execute"],
    ]
    w.log(f"A/B at {iso_et(start, tz)} ({calendar_utc(start)}), hard end {iso_et(times['hard'], tz)}, "
          f"evals {', '.join(iso_et(e, tz) for e in times['evals'])}, backstop {iso_et(backstop, tz)}, "
          f"RuntimeMaxSec {runtime}")
    for argv in runs:
        cp = w.act(argv, timeout=60)
        if cp.returncode != 0:
            raise MW.Fail(f"systemd-run exited {cp.returncode}: {(cp.stderr or cp.stdout).strip()[:200]}")
    if w.execute:
        for u in (unit, back):
            w.log(f"{u}.timer: {w.unit_prop(u + '.timer', 'ActiveState')}/{w.unit_prop(u + '.timer', 'SubState')}, "
                  f"next {w.unit_prop(u + '.timer', 'NextElapseUSecRealtime')}")
    return 0


class VirtualClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(0.0, s)


def build(c: dict[str, str], execute: bool, name: str, *, runner: Callable[..., Any] | None = None,
          clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
          out: Callable[[str], None] | None = None, start: float | None = None, fake_out: Path | None = None,
          error_at: str = "", probe_fail: bool = False) -> Night:
    w = MW.Window(c, execute, name, runner=runner, clock=clock, sleep=sleep, out=out)
    fake = not execute and name == "ab-night"
    times = night_times(c, start if start is not None else clock())
    if fake:
        ab: Ab = FakeAb(c, fake_out or Path(tempfile.mkdtemp(prefix="mini-ab-fake-")), w.log, clock, sleep,
                        error_at=error_at)
    else:
        ab = Ab(c, Path(c["AB_OUT"]), w.log, clock, sleep)
    guard = Guard(c, w, times["hard"], fake=not execute)
    return Night(w, c, ab, guard, times, fake=fake, fake_probe_fail=probe_fail)


def main(argv: list[str] | None = None, *, runner: Callable[..., Any] | None = None,
         clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
         out: Callable[[str], None] | None = None, signals: bool = False) -> int:
    ap = argparse.ArgumentParser(prog="mini_ab", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--conf", default=os.environ.get("DISTILL_WINDOW_CONF") or str(HERE / "window.conf"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--execute", action="store_true", help="really do it (without: FAKE, nothing changes)")
    r.add_argument("--fake-start", help="FAKE: the virtual start (ISO; default tonight's AB_START_HHMM)")
    r.add_argument("--fake-out", help="FAKE: the scratch --out for teacher_ab (default a new temp dir)")
    r.add_argument("--fake-probe-fail", action="store_true", help="FAKE: the voice probe on Q4 is not OK")
    r.add_argument("--fake-error-at", default="", choices=("", "select", "arm-A", "arm-B", "judge"))
    sub.add_parser("restore").add_argument("--execute", action="store_true")
    sub.add_parser("ngl")
    a = sub.add_parser("arm")
    a.add_argument("--date", required=True, help="the night's date (YYYY-MM-DD, the date of AB_START_HHMM)")
    a.add_argument("--at", help="HHMM ET (default AB_START_HHMM; its first occurrence)")
    a.add_argument("--replace", action="store_true")
    a.add_argument("--execute", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    c = load(Path(args.conf))
    execute = bool(getattr(args, "execute", False))
    if args.cmd == "arm":
        w = MW.Window(c, execute, "ab-arm", runner=runner, clock=clock, sleep=sleep, out=out)
        w.start_log()
        try:
            return cmd_arm(w, c, date.fromisoformat(args.date), args.at or c["AB_START_HHMM"], args.replace)
        except (MW.Skip, MW.Fail) as e:
            w.log(f"refused: {e}")
            return 2
    if args.cmd == "run" and not execute:
        tz = ZoneInfo(c["ET_TZ"])
        if args.fake_start:
            t0 = datetime.fromisoformat(args.fake_start)
            start = (t0 if t0.tzinfo else t0.replace(tzinfo=tz)).timestamp()
        else:
            today = datetime.fromtimestamp(clock(), tz).date()
            start = at_et(today, c["AB_START_HHMM"], tz)
            if start <= clock():
                start = at_et(today + timedelta(days=1), c["AB_START_HHMM"], tz)
        vc = VirtualClock(start)
        night = build(c, False, "ab-night", runner=runner, clock=vc.now, sleep=vc.sleep, out=out, start=start,
                      fake_out=Path(args.fake_out) if args.fake_out else None, error_at=args.fake_error_at,
                      probe_fail=args.fake_probe_fail)
        night.log(f"FAKE night from {iso_et(start, tz)}: read-only checks are real, nothing is changed, arms are "
                  f"simulated; teacher_ab scratch out {night.ab.out}")
        return night.run()
    name = {"run": "ab-night", "restore": "ab-restore", "ngl": "ab-ngl"}[args.cmd]
    night = build(c, execute, name, runner=runner, clock=clock, sleep=sleep, out=out)
    if args.cmd == "ngl":
        return cmd_ngl(night)
    night.w.start_log()
    terminating: list[int] = []

    def on_term(signum: int, _frame: Any) -> None:
        if terminating:
            night.log(f"signal {signum} again: ignored while the swap back finishes")
            return
        terminating.append(signum)
        raise KeyboardInterrupt(f"signal {signum}")

    if signals:
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(s, on_term)
    try:
        if args.cmd == "restore":
            return cmd_restore(night)
        return night.run()
    except MW.Skip as e:
        night.log(f"SKIP (nothing changed): {e}")
        if execute:
            night.w.notify(f"mini teacher A/B skipped (nothing changed): {e}")
        return 2
    finally:
        night.log("end")


if __name__ == "__main__":
    sys.exit(main(signals=True))
