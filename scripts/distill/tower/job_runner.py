#!/usr/bin/env python3
"""Job runner for the GPU tower's nightly flywheel window.

    job_runner.sh                          # dry run (default): plan + checks, starts nothing
    job_runner.sh --execute [--stop-at 06:35]
    job_runner.sh --summary                # <= 5 lines from the last progress.json (window-end notify)
    job_runner.sh --status                 # progress.json as JSON

It runs after the window start has applied the ``flywheel`` setup, and stops at
``--stop-at`` (default 06:35 local time) or on SIGTERM, whichever comes first.
It never stops, starts or restarts a systemd unit: the window start and end own
those. It only starts its own processes, records their PIDs, and stops exactly
those (process group of a recorded PID, after checking its command line).

Queue, in order, one step at a time, each in its own resumable directory under
``$DJR_RESULTS/<model>/<kind>/``:

00. ``train-smoke`` (only with ``DJR_SMOKE=1``; first in the queue): the Phase 5
   QLoRA smoke of the 9B (``train_qlora.py --smoke``: 200 examples, 50 steps,
   the loss must drop, a checkpoint round trip) with the training python, from
   the HEAD-staged ``DJR_SMOKE_CODE``, on ``DJR_SMOKE_DATA`` (+ ``DJR_SMOKE_VAL``)
   with the base in ``DJR_SMOKE_BASE`` (a local, pinned download; its revision
   is ``DJR_SMOKE_BASE_REV`` or the ``revision`` in that directory's
   ``MANIFEST.json``), offline (``HF_HUB_OFFLINE=1``), into ``DJR_SMOKE_OUT``.
   ``DJR_SMOKE_MAX_DERIVED_OK=1`` passes the trainer's operator-approval flag
   for Max-derived data (DISTILL-OPEN-QUESTIONS Q6b; the dataset must record
   its own approval too, or the trainer refuses with exit 3); the trainer then
   requires ``max-derived`` in the ``--out`` name and tags every output. It is
   time-boxed like the A/B (``DJR_SMOKE_MAX_MIN`` from its start, never past
   the step deadline; not started with less than ``DJR_SMOKE_MIN_MIN``) and runs
   in a ``distill-train-smoke-*`` scope. The run manifest's ``status`` decides:
   ``smoke-train-pass`` is done; ``stopped`` (time box, SIGTERM) is partial and
   resumes with ``--resume``; ``smoke-train-fail``, a refusal or any other exit
   fails the step. Its inputs missing: skipped, never a reason to refuse the
   window. Only the training half of the smoke gate: the smoke adapter's eval
   (merge, quantize, score) is a later step.
01. ``train-full`` (only with ``DJR_TRAIN=1``; after the smoke): the Phase 5 full
   QLoRA run of the 9B (``train_qlora.py``, no ``--smoke``) from the HEAD-staged
   ``DJR_TRAIN_CODE`` on ``DJR_TRAIN_DATA`` (+ ``DJR_TRAIN_VAL``), base
   ``DJR_TRAIN_BASE`` (revision as for the smoke), into ``DJR_TRAIN_OUT`` (the
   name must carry ``max-derived`` for Max-derived data; Q6b, with
   ``DJR_TRAIN_MAX_DERIVED_OK=1`` for the operator flag). Settings:
   ``DJR_TRAIN_EPOCHS``/``_MAX_SEQ``/``_GRAD_ACCUM``/``_SAVE_EVERY`` (steps) /
   ``_SAVE_EVERY_MIN`` (minutes, whichever first) / ``_KEEP`` (checkpoints kept)
   and ``DJR_TRAIN_MIN_FREE_GIB`` (the trainer checkpoints and stops below it;
   the runner's own ``DJR_DISK_FLOOR_GIB`` still applies, so a window with
   little disk needs that lowered too). In the 2026-10-31 continuous window the
   run is started by ``train_full.py launch`` instead (business is down then,
   and window_start needs it up); this step resumes it in later windows. Its ``--stop-at`` is the earlier of ``DJR_TRAIN_STOP_AT``
   (ISO with offset) and the step deadline, so it checkpoints before the window
   closes; a run directory with ``run_state.json`` resumes (``--resume``). It runs
   in a ``distill-train-full-<stamp>`` scope with ``TimeoutStopSec`` =
   ``DJR_TRAIN_STOP_TIMEOUT_S``, so a restore that stops ``distill-*`` units
   gives the trainer time to checkpoint on SIGTERM. Manifest ``done`` is done;
   ``stopped`` (stop-at, SIGTERM, disk floor) is partial and resumes; ``failed``
   (diverged), a refusal or any other exit fails the step. Not started with
   less than ``DJR_TRAIN_MIN_MIN`` before its stop-at, or with free disk under
   ``DJR_TRAIN_MIN_FREE_GIB`` + 1 GiB.
0. ``teacher-ab`` (only with ``DJR_AB=1``; first in the queue): the teacher quant
   A/B (``scripts/distill/teacher_ab_run.sh --execute`` from the HEAD-staged
   ``DJR_AB_CODE``, never ``DJR_CODE``, so the scorer the baselines resume with
   does not change). The driver starts one llama-server at a time on
   ``DJR_HOST:DJR_PORT`` (the IQ3 copy, then the Q4_K_M, which also judges both
   arms), records and stops it by PID, and resumes where it stopped. The tasks
   are ``DJR_AB_TASKS`` (a taskset_gen.py real-pattern taskset, default the one
   ``taskset_gen.py exclude`` wrote without the excluded sessions' rows); the models are
   checked against ``DJR_AB_IQ3_SHA256`` / ``DJR_AB_Q4_SHA256`` when set. It is
   time-boxed: its ``--until`` is the earlier of ``DJR_AB_MAX_MIN`` after its
   start and the step deadline (stop-at minus what the later must-run steps
   reserve), so the rest of the queue keeps its time; it is not started with
   less than ``DJR_AB_MIN_MIN`` left. The driver runs in a ``distill-teacher-ab-*``
   scope, so a restore stops it and its server. Exit 0 (a decision) is done;
   1 (time box, work left) is partial and resumes next window; an ``invalid``
   report, 2 or 3 fail the step (two failures park it). Its inputs missing:
   skipped, never a reason to refuse the window. ``ab_report.json`` /
   ``ab_summary.txt`` stay in ``DJR_AB_OUT``.
1. ``base-9b-baseline``: its own llama-server on loopback (``DJR_HOST:DJR_PORT``)
   with the base 9B Q4_K_M at ``-c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0
   --jinja``, then ``score.py run`` (concurrency 1) over sanity, eval-tools,
   eval-injection and eval-longctx (all buckets, file order 32k..256k), one
   results file per set; then a combined summary. The server is stopped by PID.
2. ``qwen38-27b-feasibility``: ``feasibility_27b.py --execute`` once per
   sequence length (16k, 12k, 8k, 4k) with the training venv, one report per
   length, then a merged report (``--measured``) with the decision. A length
   whose result is a measurement (fits or OOM) is never probed again.
3. ``qwen38-27b-baseline`` and ``glimmer-30b-baseline`` (if time remains; the
   second only if its GGUF exists and llama.cpp knows its architecture): same
   sets, at the largest context that loads, trying the model's native context
   first, then 131072, 65536, 32768. The context that loaded is kept for later
   windows so a resumed run stays like for like. If a later window has to use
   another context (the server crashed at it, or it no longer loads), the
   results scored at the old one move to ``ctx-<n>-<run id>/`` and every set
   runs again, so one baseline is never a mix of contexts (``ctx_history`` in
   step.json says what moved).

Time: a step gets a deadline of ``stop-at`` minus the time reserved for a
later step that must run tonight (the feasibility probe, ``DJR_FEAS_RESERVE_MIN``),
and is not started with less than its minimum left. A step cut by its deadline
is ``partial`` and resumes in the next window (``score.py`` skips finished
cases; the probe skips measured lengths). A step that failed twice is skipped
until someone looks at it.

Stops: SIGTERM/SIGINT/SIGHUP, the NO-SWAPS flag (any path in ``DJR_NO_SWAPS``;
by default the window's ``~/distill/window/NO-SWAPS-TONIGHT`` and two older
paths), the production model unit (``DJR_PROD_UNIT``) becoming active, free disk
under ``DJR_DISK_FLOOR_GIB`` + ``DJR_DISK_MARGIN_GIB``, or a GPU child stopped from
outside (a restore stopping its ``distill-*`` scope: "external-stop"). Each stops
the current child (SIGTERM to its process group, SIGKILL after a grace period),
stops the runner's llama-server, writes progress and exits. The NO-SWAPS and
production-unit checks run every ``DJR_SLOW_CHECK_S``, and always right before a
GPU child is started. A llama-server or probe that ends by a stop signal is a
stop, never a crash: no context step-down, no failure counted. A crashed
earlier run's processes are found through the PID files in ``DJR_RUN`` and
stopped first. Every process it starts has core dumps off (RLIMIT_CORE 0).

Refuses to start (``--execute``) outside ``DJR_WINDOW_START``-``DJR_WINDOW_END``
(unless a continuous window ``DJR_WINDOW_UNTIL``, epoch seconds, is still open: then
stop-at must be by its end and the 7 h stop-at cap is off), with the NO-SWAPS flag present, while the production unit is active, with the
GPU holding more than ``DJR_GPU_MAX_USED_MIB``, with the port taken, or under the
disk floor; and (``DJR_GATES``, on unless ``DJR_ANYTIME``) unless the last
rehearsal PASSED (``DJR_REHEARSAL_LAST``) and the dead-man restore timer
(``DJR_DEADMAN_TIMER``) is armed to fire before ``DJR_DEADMAN_LATEST``. It is
launched only by the window start (window_start.sh), which passes the same
settings. GPU-holding children run in ``distill-*`` systemd scopes when
``systemd-run`` works, so a ``setups.py apply business`` restore can stop them
even if this runner is gone.

``DJR_ONLY`` (comma-separated step names) limits the queue to those steps.

Progress: ``$DJR_LOGS/progress.json`` (schema ``distill-tower-runner/1``),
rewritten atomically on every change. Logs: ``$DJR_LOGS/<run id>/``.
Settings: ``DJR_*`` environment variables, or ``KEY=VALUE`` lines in
``$DJR_ENV_FILE`` (default ``~/distill/job_runner.env``; read, never sourced).

Exit: 0 queue finished or ended at stop-at; 1 a step failed; 2 refused or usage;
3 stopped by NO-SWAPS, the production unit or the disk floor; 143 SIGTERM.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA = "distill-tower-runner/1"
STEP_SCHEMA = "distill-tower-step/1"

DEFAULTS: Dict[str, str] = {
    "DJR_HOME": "~/distill",
    "DJR_CODE": "{home}/code",
    "DJR_EVALS": "{home}/evalsets",
    "DJR_RESULTS": "{home}/phase4/results",
    "DJR_LOGS": "{home}/logs/window",
    "DJR_RUN": "{home}/run",
    "DJR_LLAMA_SERVER": "~/llama.cpp/build/bin/llama-server",
    "DJR_LLAMA_SRC": "~/llama.cpp",
    "DJR_HOST": "127.0.0.1",
    "DJR_PORT": "18080",
    "DJR_SCORE_PY": "{home}/venv-score/bin/python",
    "DJR_TRAIN_PY": "{home}/venv-train/bin/python",
    "DJR_MODEL_9B": "",          # empty: the one *9b*q4_k_m*.gguf under {home}/models
    "DJR_MODEL_27B_BNB": "{home}/models/qwen38-27b-bnb4",
    "DJR_MODEL_27B_GGUF": "~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "DJR_MODEL_GLIMMER": "",     # empty: the one *glimmer*.gguf under {home}/models
    "DJR_LENGTHS": "{home}/phase4/inputs/train-lengths.jsonl",
    "DJR_NO_SWAPS": "{home}/window/NO-SWAPS-TONIGHT:{home}/NO-SWAPS:~/.local/share/distill-setups/NO-SWAPS",
    "DJR_PROD_UNIT": "llama-server.service",
    # gates before taking the GPU (DJR_GATES: auto = on unless DJR_ANYTIME; 1 on; 0 off)
    "DJR_GATES": "auto",
    "DJR_REHEARSAL_LAST": "{home}/window/state/rehearsal.last",  # must start with PASS
    "DJR_DEADMAN_TIMER": "distill-deadman-tower-window.timer",  # armed, waiting ...
    "DJR_DEADMAN_LATEST": "07:30",  # ... and due by then (HH:MM local, the window's date)
    "DJR_SETTLE_S": "3",         # after a GPU child ends on its own: wait this long for our SIGTERM
    "DJR_EXPECT_COMMIT": "",     # dry run: warn when the staged code's COMMIT differs
    "DJR_NOTIFY": "{home}/notify.sh",
    "DJR_DISK_FLOOR_GIB": "15",
    "DJR_DISK_MARGIN_GIB": "1",
    "DJR_GPU_MAX_USED_MIB": "1500",
    "DJR_WINDOW_START": "01:00",
    "DJR_WINDOW_END": "07:00",
    "DJR_STOP_AT": "06:35",
    "DJR_ANYTIME": "0",          # tests only: skip the window-hours check
    # a continuous window (operator 2026-10-31: the 4090 until Mon 2026-11-02 11:00 UTC): epoch seconds.
    # Before it, the DJR_WINDOW_START/END hours and the 7 h stop-at cap do not apply; stop-at must be
    # no later than it. Empty or past: the nightly hours apply as before.
    "DJR_WINDOW_UNTIL": "",
    # comma-separated step names: run only these (empty: the whole queue)
    "DJR_ONLY": "",
    "DJR_FEAS_RESERVE_MIN": "75",
    "DJR_FEAS_SEQS": "16384,12288,8192,4096",
    "DJR_FEAS_CHILD_S": "1800",  # feasibility_27b.CHILD_TIMEOUT_S
    "DJR_UBATCH_9B": "2048",
    "DJR_UBATCH_BAKEOFF": "512",
    "DJR_SERVER_LOAD_S": "420",
    "DJR_POLL_S": "2",
    "DJR_SLOW_CHECK_S": "20",
    "DJR_SCORE_GRACE_S": "15",
    "DJR_SERVER_GRACE_S": "30",
    "DJR_FEAS_GRACE_S": "60",
    "DJR_GPU_RELEASE_S": "90",
    "DJR_USE_SCOPES": "auto",    # auto | 1 | 0
    "DJR_MIN_SET_S": "90",
    "DJR_MIN_9B_S": "600",       # least time worth starting the 9B baseline with
    "DJR_MIN_BAKE_S": "1200",    # same, a bake-off baseline
    "DJR_FEAS_MARGIN_S": "60",   # load/teardown margin around one feasibility length
    "DJR_MIN_RUN_S": "300",      # refuse a stop-at closer than this
    # the Phase 5 QLoRA smoke (step 00; off unless DJR_SMOKE=1)
    "DJR_SMOKE": "0",
    "DJR_SMOKE_CODE": "{home}/code-train/head",  # HEAD export: scripts/distill/{train_qlora,render,artifact_policy}.py
    "DJR_SMOKE_DATA": "{home}/phase3/build-v3-max-derived/train.jsonl",
    "DJR_SMOKE_VAL": "{home}/phase3/build-v3-max-derived/val.jsonl",  # empty: no val loss
    "DJR_SMOKE_TOKENIZER": "{home}/models/qwen3.5-9b-tokenizer",
    "DJR_SMOKE_BASE": "{home}/models/qwen3.5-9b-bnb4",
    "DJR_SMOKE_BASE_REV": "",          # empty: "revision" from DJR_SMOKE_BASE/MANIFEST.json
    "DJR_SMOKE_OUT": "{home}/phase5/runs/9b-smoke-max-derived",
    "DJR_SMOKE_MAX_DERIVED_OK": "0",   # 1: pass --i-have-operator-approval-for-max-derived-data (Q6b)
    "DJR_SMOKE_MAX_MIN": "90",         # time box from the step's start
    "DJR_SMOKE_MIN_MIN": "20",         # least time worth starting it with
    "DJR_SMOKE_GRACE_S": "300",        # SIGTERM -> the trainer checkpoints and exits -> SIGKILL
    # the Phase 5 full QLoRA run (step 01; off unless DJR_TRAIN=1)
    "DJR_TRAIN": "0",
    "DJR_TRAIN_CODE": "{home}/code-train/head",
    "DJR_TRAIN_DATA": "{home}/phase3/build-v4-max-derived/train.jsonl",
    "DJR_TRAIN_VAL": "{home}/phase3/build-v4-max-derived/val.jsonl",
    "DJR_TRAIN_TOKENIZER": "{home}/models/qwen3.5-9b-tokenizer",
    "DJR_TRAIN_BASE": "{home}/models/qwen3.5-9b-bnb4",
    "DJR_TRAIN_BASE_REV": "",          # empty: "revision" from DJR_TRAIN_BASE/MANIFEST.json
    "DJR_TRAIN_OUT": "{home}/phase5/runs/9b-r32-v4-max-derived",
    "DJR_TRAIN_MAX_DERIVED_OK": "0",   # 1: pass --i-have-operator-approval-for-max-derived-data (Q6b)
    "DJR_TRAIN_EPOCHS": "2",
    "DJR_TRAIN_MAX_SEQ": "16384",
    "DJR_TRAIN_GRAD_ACCUM": "8",
    "DJR_TRAIN_SAVE_EVERY": "200",     # optimizer steps (the runbook's ~200) ...
    "DJR_TRAIN_SAVE_EVERY_MIN": "60",  # ... or this many minutes, whichever first (operator: hourly)
    "DJR_TRAIN_KEEP": "2",             # checkpoints kept
    "DJR_TRAIN_MIN_FREE_GIB": "6",     # the trainer checkpoints and stops below it
    "DJR_TRAIN_STOP_AT": "",           # ISO with offset; empty: the step deadline only
    "DJR_TRAIN_MIN_MIN": "30",         # least time worth starting it with
    "DJR_TRAIN_GRACE_S": "600",        # SIGTERM -> the trainer checkpoints and exits -> SIGKILL
    "DJR_TRAIN_STOP_TIMEOUT_S": "900",  # its scope's TimeoutStopSec (a restore stopping distill-*)
    # the teacher quant A/B (step 0; off unless DJR_AB=1)
    "DJR_AB": "0",
    "DJR_AB_CODE": "{home}/code-ab",   # HEAD-staged code for the A/B only (teacher_ab*.py, teacher_gen.py, src/)
    "DJR_AB_PY": "",                   # empty: DJR_SCORE_PY (it holds Prometheus's base dependencies)
    "DJR_AB_TASKS": "{home}/phase2b/tasksets/real-v1-excl.jsonl",
    "DJR_AB_OUT": "{home}/teacher-ab",
    "DJR_AB_IQ3": "{home}/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf",
    "DJR_AB_Q4": "~/models/Qwen3.8-27B-regular-Q4_K_M-bartowski.gguf",
    "DJR_AB_IQ3_SHA256": "",
    "DJR_AB_Q4_SHA256": "",
    "DJR_AB_PACK": "",                 # optional build.py request pack (the teacher's system prompt)
    "DJR_AB_MAX_MIN": "120",           # time box from the step's start
    "DJR_AB_MIN_MIN": "40",            # least time worth starting it with
    "DJR_AB_GRACE_S": "120",           # SIGTERM -> the driver stops its step and server (60 s) -> SIGKILL
}

# Score sets in run order: (set name, case file, per-request timeout s).
SETS: List[Tuple[str, str, int]] = [
    ("sanity", "sanity.jsonl", 600),
    ("eval-tools", "eval-tools.jsonl", 900),
    ("eval-injection", "eval-injection.jsonl", 900),
    ("eval-longctx", "eval-longctx.jsonl", 1800),
]
BUCKET_CTX = (131072, 65536, 32768)  # step-down contexts: each still holds a whole bucket
# Stop reasons that end the runner's night (anything else is per step: "deadline", "server-died").
STOP_REASONS = ("sigterm", "no-swaps", "production-unit-active", "disk-floor", "stop-at", "external-stop")
# A child that ended by one of these was stopped (a restore stops distill-* units and scopes with
# SIGTERM), not crashed. llama-server exits 0 on SIGTERM once loaded (its handler), and dies by the
# signal while loading; a shell reports 128+n.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
SET_RETRIES = 3     # score.py passes per set that may end with errors before it is left as is
STEP_FAILURES = 2   # a step that failed this often is skipped until someone looks
KNOWN_STEPS = ("train-smoke", "train-full", "teacher-ab", "base-9b-baseline", "qwen38-27b-feasibility",
               "qwen38-27b-baseline", "glimmer-30b-baseline")  # DJR_ONLY must name these

# Rough speeds for the plan's time estimate only (RTX 4090; see DISTILL notes):
# prefill tok/s at short depth, decode tok/s, the depth at which attention costs as
# much as the weights (hybrid models: 1 layer in 4 is full attention).
SPEED = {
    "9b": {"pp": 7000.0, "tg": 120.0, "d_eq": 100_000.0},
    "27b": {"pp": 2400.0, "tg": 42.0, "d_eq": 92_000.0},
}
OUT_TOKENS = {"sanity": 60, "eval-tools": 150, "eval-injection": 200, "eval-longctx": 40}
CHARS_PER_TOKEN = 3.55  # measured on the eval files (JSON text / builder token counts)
# eval-longctx prompt sizes per bucket (the builder's targets, 9B tokenizer); text size overshoots
# by a few percent, which near 256K would wrongly count cases as over the context
BUCKET_TOKENS = {"32k": 31129, "64k": 62259, "128k": 124518, "256k": 249036}


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


class Cfg:
    def __init__(self, env: Dict[str, str]):
        raw = dict(DEFAULTS)
        env_file = Path(os.path.expanduser(env.get("DJR_ENV_FILE", "~/distill/job_runner.env")))
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k in DEFAULTS:
                    raw[k] = v.strip().strip("'\"")
        for k in DEFAULTS:
            if k in env:
                raw[k] = env[k]
        home = os.path.expanduser(raw["DJR_HOME"])
        self.raw = raw

        def path(key: str) -> str:
            v = raw[key]
            return os.path.expanduser(v.replace("{home}", home)) if v else ""

        self.home = home
        self.code, self.evals = path("DJR_CODE"), path("DJR_EVALS")
        self.results, self.logs, self.run = path("DJR_RESULTS"), path("DJR_LOGS"), path("DJR_RUN")
        self.llama_server, self.llama_src = path("DJR_LLAMA_SERVER"), path("DJR_LLAMA_SRC")
        self.host, self.port = raw["DJR_HOST"], int(raw["DJR_PORT"])
        self.score_py, self.train_py = path("DJR_SCORE_PY"), path("DJR_TRAIN_PY")
        self.model_9b, self.model_27b_bnb = path("DJR_MODEL_9B"), path("DJR_MODEL_27B_BNB")
        self.model_27b_gguf, self.model_glimmer = path("DJR_MODEL_27B_GGUF"), path("DJR_MODEL_GLIMMER")
        self.lengths, self.notify = path("DJR_LENGTHS"), path("DJR_NOTIFY")
        self.no_swaps = [os.path.expanduser(p.replace("{home}", home))
                         for p in raw["DJR_NO_SWAPS"].split(":") if p]
        self.prod_unit = raw["DJR_PROD_UNIT"]
        gates = raw["DJR_GATES"].strip().lower()
        self.gates = gates == "1" or (gates == "auto" and raw["DJR_ANYTIME"] != "1")
        self.rehearsal_last = path("DJR_REHEARSAL_LAST")
        self.deadman_timer = raw["DJR_DEADMAN_TIMER"].strip()
        self.deadman_latest = raw["DJR_DEADMAN_LATEST"].strip()
        self.settle_s = float(raw["DJR_SETTLE_S"])
        self.expect_commit = raw["DJR_EXPECT_COMMIT"].strip()
        self.disk_floor = float(raw["DJR_DISK_FLOOR_GIB"]) * 2**30
        self.disk_margin = float(raw["DJR_DISK_MARGIN_GIB"]) * 2**30
        self.gpu_max_used = int(raw["DJR_GPU_MAX_USED_MIB"])
        self.window = (raw["DJR_WINDOW_START"], raw["DJR_WINDOW_END"])
        self.stop_at_raw = raw["DJR_STOP_AT"]
        self.anytime = raw["DJR_ANYTIME"] == "1"
        until = raw["DJR_WINDOW_UNTIL"].strip()
        self.window_until = float(until) if until else None
        self.only = [x.strip() for x in raw["DJR_ONLY"].split(",") if x.strip()]
        self.feas_reserve = float(raw["DJR_FEAS_RESERVE_MIN"]) * 60
        self.feas_seqs = [int(s) for s in raw["DJR_FEAS_SEQS"].split(",") if s.strip()]
        self.feas_child_s = float(raw["DJR_FEAS_CHILD_S"])
        self.ub_9b, self.ub_bake = int(raw["DJR_UBATCH_9B"]), int(raw["DJR_UBATCH_BAKEOFF"])
        self.server_load_s = float(raw["DJR_SERVER_LOAD_S"])
        self.poll_s, self.slow_check_s = float(raw["DJR_POLL_S"]), float(raw["DJR_SLOW_CHECK_S"])
        self.score_grace = float(raw["DJR_SCORE_GRACE_S"])
        self.server_grace = float(raw["DJR_SERVER_GRACE_S"])
        self.feas_grace = float(raw["DJR_FEAS_GRACE_S"])
        self.gpu_release_s = float(raw["DJR_GPU_RELEASE_S"])
        self.use_scopes = raw["DJR_USE_SCOPES"]
        self.min_set_s = float(raw["DJR_MIN_SET_S"])
        self.min_9b_s, self.min_bake_s = float(raw["DJR_MIN_9B_S"]), float(raw["DJR_MIN_BAKE_S"])
        self.feas_margin = float(raw["DJR_FEAS_MARGIN_S"])
        self.min_run_s = float(raw["DJR_MIN_RUN_S"])
        self.ab = raw["DJR_AB"].strip() == "1"
        self.ab_code, self.ab_tasks, self.ab_out = path("DJR_AB_CODE"), path("DJR_AB_TASKS"), path("DJR_AB_OUT")
        self.ab_py = path("DJR_AB_PY") or self.score_py
        self.ab_iq3, self.ab_q4, self.ab_pack = path("DJR_AB_IQ3"), path("DJR_AB_Q4"), path("DJR_AB_PACK")
        self.ab_iq3_sha, self.ab_q4_sha = raw["DJR_AB_IQ3_SHA256"].strip(), raw["DJR_AB_Q4_SHA256"].strip()
        self.ab_max_s = float(raw["DJR_AB_MAX_MIN"]) * 60
        self.ab_min_s = float(raw["DJR_AB_MIN_MIN"]) * 60
        self.ab_grace = float(raw["DJR_AB_GRACE_S"])
        self.smoke = raw["DJR_SMOKE"].strip() == "1"
        self.smoke_code, self.smoke_data = path("DJR_SMOKE_CODE"), path("DJR_SMOKE_DATA")
        self.smoke_val, self.smoke_tok = path("DJR_SMOKE_VAL"), path("DJR_SMOKE_TOKENIZER")
        self.smoke_base, self.smoke_out = path("DJR_SMOKE_BASE"), path("DJR_SMOKE_OUT")
        self.smoke_base_rev = raw["DJR_SMOKE_BASE_REV"].strip()
        self.smoke_md_ok = raw["DJR_SMOKE_MAX_DERIVED_OK"].strip() == "1"
        self.smoke_max_s = float(raw["DJR_SMOKE_MAX_MIN"]) * 60
        self.smoke_min_s = float(raw["DJR_SMOKE_MIN_MIN"]) * 60
        self.smoke_grace = float(raw["DJR_SMOKE_GRACE_S"])
        self.train = raw["DJR_TRAIN"].strip() == "1"
        self.train_code, self.train_data = path("DJR_TRAIN_CODE"), path("DJR_TRAIN_DATA")
        self.train_val, self.train_tok = path("DJR_TRAIN_VAL"), path("DJR_TRAIN_TOKENIZER")
        self.train_base, self.train_out = path("DJR_TRAIN_BASE"), path("DJR_TRAIN_OUT")
        self.train_base_rev = raw["DJR_TRAIN_BASE_REV"].strip()
        self.train_md_ok = raw["DJR_TRAIN_MAX_DERIVED_OK"].strip() == "1"
        self.train_epochs = int(raw["DJR_TRAIN_EPOCHS"])
        self.train_max_seq = int(raw["DJR_TRAIN_MAX_SEQ"])
        self.train_grad_accum = int(raw["DJR_TRAIN_GRAD_ACCUM"])
        self.train_save_every = int(raw["DJR_TRAIN_SAVE_EVERY"])
        self.train_save_every_min = float(raw["DJR_TRAIN_SAVE_EVERY_MIN"])
        self.train_keep = int(raw["DJR_TRAIN_KEEP"])
        self.train_min_free = float(raw["DJR_TRAIN_MIN_FREE_GIB"])
        sa = raw["DJR_TRAIN_STOP_AT"].strip()
        self.train_stop_at = datetime.fromisoformat(sa).timestamp() if sa else None
        if sa and datetime.fromisoformat(sa).tzinfo is None:
            raise ValueError("DJR_TRAIN_STOP_AT needs a UTC offset")
        self.train_min_s = float(raw["DJR_TRAIN_MIN_MIN"]) * 60
        self.train_grace = float(raw["DJR_TRAIN_GRACE_S"])
        self.train_stop_timeout = int(float(raw["DJR_TRAIN_STOP_TIMEOUT_S"]))

    @property
    def train_script(self) -> str:
        return os.path.join(self.train_code, "scripts", "distill", "train_qlora.py")

    @property
    def smoke_script(self) -> str:
        return os.path.join(self.smoke_code, "scripts", "distill", "train_qlora.py")

    @property
    def ab_driver(self) -> str:
        return os.path.join(self.ab_code, "scripts", "distill", "teacher_ab_run.sh")

    @property
    def score_script(self) -> str:
        return os.path.join(self.code, "scripts", "distill", "score.py")

    @property
    def feas_script(self) -> str:
        return os.path.join(self.code, "scripts", "distill", "feasibility_27b.py")

    def case_file(self, name: str) -> str:
        return os.path.join(self.evals, name)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def now() -> float:
    return time.time()


def iso(t: Optional[float]) -> Optional[str]:
    return None if t is None else datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def hm(t: Optional[float]) -> str:
    return "-" if t is None else datetime.fromtimestamp(t).strftime("%H:%M")


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def pid_cmdline(pid: int) -> str:
    p = Path(f"/proc/{pid}/cmdline")
    try:
        if p.exists():
            return p.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def killpg(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def http_json(url: str, timeout: float = 5.0) -> Tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            body = r.read()
            try:
                return r.status, json.loads(body or b"null")
            except ValueError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def disk_free(path: str) -> float:
    p = Path(path)
    while not p.exists():
        p = p.parent
    return float(shutil.disk_usage(str(p)).free)


def parse_hhmm(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"not HH:MM: {s!r}")
    return int(m.group(1)), int(m.group(2))


def resolve_stop_at(value: str, ref: float) -> float:
    """HH:MM (today; tomorrow if already past), epoch seconds, or ISO with an offset."""
    v = value.strip()
    if re.fullmatch(r"\d{9,}(\.\d+)?", v):
        return float(v)
    if re.fullmatch(r"\d{1,2}:\d{2}", v):
        h, m = parse_hhmm(v)
        base = datetime.fromtimestamp(ref)
        t = base.replace(hour=h, minute=m, second=0, microsecond=0)
        if t.timestamp() <= ref:
            t += timedelta(days=1)
        return t.timestamp()
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        raise ValueError("--stop-at ISO time needs a UTC offset")
    return dt.timestamp()


# ---------------------------------------------------------------------------
# GGUF header (metadata only)
# ---------------------------------------------------------------------------

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}


def gguf_meta(path: str, max_kv: int = 4096) -> Dict[str, Any]:
    """Metadata of a GGUF file (arrays over 64 items are summarised as their length)."""
    out: Dict[str, Any] = {}
    with open(path, "rb") as f:
        def rd(n: int) -> bytes:
            b = f.read(n)
            if len(b) != n:
                raise ValueError("short read")
            return b

        def u(fmt: str) -> Any:
            return struct.unpack(fmt, rd(struct.calcsize(fmt)))[0]

        def s() -> str:
            return rd(u("<Q")).decode("utf-8", "replace")

        def val(t: int) -> Any:
            if t in _SCALAR:
                return u(_SCALAR[t])
            if t == 8:
                return s()
            if t == 9:
                et, n = u("<I"), u("<Q")
                if et in _SCALAR and n > 64:
                    f.seek(struct.calcsize(_SCALAR[et]) * n, 1)
                    return {"_len": n}
                items = [val(et) for _ in range(n)]
                return items if n <= 64 else {"_len": n}
            raise ValueError(f"bad GGUF value type {t}")

        if rd(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = u("<I")
        if version < 2:
            raise ValueError(f"GGUF v{version} not supported")
        u("<Q")
        n_kv = u("<Q")
        for _ in range(min(n_kv, max_kv)):
            key = s()
            out[key] = val(u("<I"))
    return out


def gguf_facts(path: str) -> Dict[str, Any]:
    meta = gguf_meta(path)
    arch = meta.get("general.architecture")
    a = f"{arch}."

    def g(k: str) -> Any:
        return meta.get(a + k)

    facts: Dict[str, Any] = {"arch": arch, "name": meta.get("general.name"),
                             "native_ctx": g("context_length")}
    blocks, interval = g("block_count"), g("full_attention_interval")
    kvh, kl, vl = g("attention.head_count_kv"), g("attention.key_length"), g("attention.value_length")
    if isinstance(kvh, list):
        attn_layers, kvh = sum(1 for x in kvh if x), max(kvh or [0])
    elif isinstance(blocks, int):
        attn_layers = blocks // interval if isinstance(interval, int) and interval > 0 else blocks
    else:
        attn_layers = None
    if attn_layers and isinstance(kvh, int) and isinstance(kl, int) and isinstance(vl, int):
        facts["kv_bytes_per_token_q8"] = int(attn_layers * kvh * (kl + vl) * 34 / 32)
    return facts


def arch_supported(llama_src: str, arch: Optional[str]) -> Optional[bool]:
    f = Path(llama_src) / "src" / "llama-arch.cpp"
    if not arch or not f.is_file():
        return None
    return f'"{arch}"' in f.read_text(errors="replace")


# ---------------------------------------------------------------------------
# host checks
# ---------------------------------------------------------------------------


def unit_state(unit: str) -> str:
    try:
        cp = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True,
                            text=True, timeout=20)
        return (cp.stdout.strip() or "unknown").splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def stopped_by_signal(rc: Optional[int], zero_is_stop: bool = False) -> bool:
    """*rc* is a stop by signal (-n or a shell's 128+n for SIGTERM/SIGINT/SIGHUP); with
    *zero_is_stop*, 0 too (llama-server's own SIGTERM handler exits 0: it never exits 0 by
    itself while serving)."""
    if rc is None:
        return False
    if zero_is_stop and rc == 0:
        return True
    return any(rc in (-int(s), 128 + int(s)) for s in STOP_SIGNALS)


def parse_systemd_time(value: str) -> Optional[float]:
    """Epoch of a ``systemctl show`` timestamp ("Sat 2026-10-31 07:15:00 EDT", or "@<epoch>")."""
    v = value.strip()
    if not v or v in ("n/a", "0"):
        return None
    if v.startswith("@"):
        try:
            return float(v[1:])
        except ValueError:
            return None
    try:
        cp = subprocess.run(["date", "-d", v, "+%s"], capture_output=True, text=True, timeout=10)
        if cp.returncode == 0 and cp.stdout.strip().isdigit():
            return float(cp.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    parts = v.split()
    if len(parts) >= 3:  # no GNU date: read it as local time
        try:
            return datetime.strptime(f"{parts[1]} {parts[2]}", "%Y-%m-%d %H:%M:%S").timestamp()
        except ValueError:
            return None
    return None


def timer_next(unit: str) -> Tuple[Optional[float], str]:
    """(next elapse epoch, state text) of a user timer that is loaded, active and waiting;
    (None, why) otherwise."""
    try:
        cp = subprocess.run(["systemctl", "--user", "show", unit, "-p", "LoadState", "-p", "ActiveState",
                             "-p", "SubState", "-p", "NextElapseUSecRealtime"],
                            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"systemctl show failed: {e}"
    props = {}
    for line in cp.stdout.splitlines():
        k, _, v = line.partition("=")
        props[k.strip()] = v.strip()
    state = (f"{props.get('LoadState', '?')}/{props.get('ActiveState', '?')}/{props.get('SubState', '?')}, "
             f"next '{props.get('NextElapseUSecRealtime', '')}'")
    if cp.returncode != 0 or (props.get("LoadState"), props.get("ActiveState"), props.get("SubState")) \
            != ("loaded", "active", "waiting"):
        return None, state
    nxt = parse_systemd_time(props.get("NextElapseUSecRealtime", ""))
    return (nxt, state) if nxt is not None else (None, state + " (no next elapse)")


def rehearsal_state(cfg: "Cfg") -> Tuple[bool, str]:
    try:
        first = Path(cfg.rehearsal_last).read_text(errors="replace").splitlines()[0].strip()
    except (OSError, IndexError):
        return False, f"no rehearsal result at {cfg.rehearsal_last}"
    return first.split(" ", 1)[0] == "PASS", first[:160]


def deadman_state(cfg: "Cfg", t: float) -> Tuple[bool, str]:
    """The dead-man restore timer is armed and fires after *t* and by DJR_DEADMAN_LATEST
    (on the window's date: today, or tomorrow when that time has passed)."""
    nxt, state = timer_next(cfg.deadman_timer)
    if nxt is None:
        return False, f"{cfg.deadman_timer} not armed ({state})"
    try:
        latest = resolve_stop_at(cfg.deadman_latest, t)
    except ValueError as e:
        return False, f"DJR_DEADMAN_LATEST: {e}"
    if not (t < nxt <= latest):
        return False, f"{cfg.deadman_timer} fires {iso(nxt)}, not before {iso(latest)}"
    return True, f"{cfg.deadman_timer} fires {iso(nxt)}"


def gpu_used_mib() -> Optional[int]:
    try:
        cp = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=30)
        vals = [int(x.strip()) for x in cp.stdout.split("\n") if x.strip().isdigit()]
        return max(vals) if vals and cp.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def gpu_apps() -> str:
    try:
        cp = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return " | ".join(x.strip() for x in cp.stdout.splitlines() if x.strip())[:300]
    except (OSError, subprocess.SubprocessError):
        return ""


def model_candidates(cfg: Cfg, want: Callable[[str], bool]) -> List[str]:
    root = os.path.join(cfg.home, "models")
    found = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.gguf"), recursive=True)):
        name = os.path.basename(p).lower()
        if "mmproj" in name or not want(name):
            continue
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", name)
        if m and m.group(1) != "00001":
            continue
        found.append(p)
    return found


def download_unfinished(path: str) -> Optional[str]:
    """A model directory's ``.download-status.json`` (the download tool's marker), when it
    says the download is not done; None when it is done or there is no marker."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    st = read_json(os.path.join(d, ".download-status.json"))
    if isinstance(st, dict) and st.get("state") != "done":
        return f"download {st.get('state')!r} in {d}"
    return None


def ready(found: Tuple[Optional[str], str]) -> Tuple[Optional[str], str]:
    path, why = found
    if path:
        busy = download_unfinished(path)
        if busy:
            return None, busy
    return path, why


def find_9b(cfg: Cfg) -> Tuple[Optional[str], str]:
    return ready(_find_9b(cfg))


def find_glimmer(cfg: Cfg) -> Tuple[Optional[str], str]:
    return ready(_find_glimmer(cfg))


def _find_9b(cfg: Cfg) -> Tuple[Optional[str], str]:
    if cfg.model_9b:
        return (cfg.model_9b, "") if os.path.isfile(cfg.model_9b) else (None, f"{cfg.model_9b} missing")
    c = model_candidates(cfg, lambda n: "9b" in n and "q4_k_m" in n)
    if len(c) == 1:
        return c[0], ""
    if not c:
        return None, f"no *9b*q4_k_m*.gguf under {cfg.home}/models"
    return None, "several 9B Q4_K_M files (set DJR_MODEL_9B): " + ", ".join(c)


def _find_glimmer(cfg: Cfg) -> Tuple[Optional[str], str]:
    if cfg.model_glimmer:
        return ((cfg.model_glimmer, "") if os.path.isfile(cfg.model_glimmer)
                else (None, f"{cfg.model_glimmer} missing"))
    c = model_candidates(cfg, lambda n: "glimmer" in n)
    q4 = [p for p in c if "q4_k_m" in os.path.basename(p).lower()]
    c = q4 or c
    if len(c) == 1:
        return c[0], ""
    if not c:
        return None, "not downloaded"
    return None, "several Glimmer GGUFs (set DJR_MODEL_GLIMMER): " + ", ".join(c)


def find_27b_gguf(cfg: Cfg) -> Tuple[Optional[str], str]:
    p = cfg.model_27b_gguf
    return ready((p, "") if p and os.path.isfile(p) else (None, f"{p} missing"))


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


class Stop(Exception):
    pass


class Runner:
    def __init__(self, cfg: Cfg, stop_at: float):
        self.cfg = cfg
        self.stop_at = stop_at
        self.run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.run_dir = os.path.join(cfg.logs, self.run_id)
        self.stop: Optional[str] = None
        self.last_slow = 0.0
        self.scopes = False
        self.server: Optional[Dict[str, Any]] = None  # {proc, label, ctx, log, cmd}
        self.child: Optional[subprocess.Popen] = None
        self.notified = 0
        self.progress: Dict[str, Any] = {
            "schema": SCHEMA, "run_id": self.run_id, "pid": os.getpid(), "unit": self_unit(),
            "state": "starting", "stop_reason": None, "started_at": iso(now()),
            "updated_at": None, "ended_at": None, "stop_at": iso(stop_at),
            "current": None, "server": None, "steps": [], "notes": [],
        }
        self.log_f = None

    # -- logging / progress ---------------------------------------------------

    def log(self, msg: str) -> None:
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        if self.log_f:
            self.log_f.write(line + "\n")
            self.log_f.flush()

    def save(self) -> None:
        self.progress["updated_at"] = iso(now())
        write_json(os.path.join(self.cfg.logs, "progress.json"), self.progress)
        write_json(os.path.join(self.run_dir, "progress.json"), self.progress)

    def note(self, msg: str) -> None:
        self.log("NOTE " + msg)
        self.progress["notes"].append(f"{hm(now())} {msg}")
        self.save()

    def notify(self, lines: List[str]) -> None:
        n = self.cfg.notify
        if self.notified >= 3 or not n or not os.access(n, os.X_OK):
            return
        self.notified += 1
        try:
            subprocess.run([n, "5"], input="\n".join(lines[:5]) + "\n", text=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            pass

    # -- stop conditions ------------------------------------------------------

    def on_signal(self, signum: int, _frame: Any) -> None:
        if not self.stop:
            self.stop = "sigterm" if signum == signal.SIGTERM else f"signal-{signum}"

    def no_swaps_flag(self) -> Optional[str]:
        for p in self.cfg.no_swaps:
            if os.path.exists(p):
                return p
        return None

    def check(self, deadline: Optional[float]) -> Optional[str]:
        """The reason to stop now, if any. Slow checks run every DJR_SLOW_CHECK_S."""
        if self.stop:
            return self.stop
        t = now()
        if t - self.last_slow >= self.cfg.slow_check_s:
            self.last_slow = t
            flag = self.no_swaps_flag()
            if flag:
                self.stop = "no-swaps"
                self.note(f"NO-SWAPS flag present ({flag}): stopping")
            elif unit_state(self.cfg.prod_unit) in ("active", "activating", "reloading"):
                self.stop = "production-unit-active"
                self.note(f"{self.cfg.prod_unit} is active: yielding the GPU")
            elif disk_free(self.cfg.home) < self.cfg.disk_floor + self.cfg.disk_margin:
                self.stop = "disk-floor"
                self.note(f"free disk {disk_free(self.cfg.home) / 2**30:.1f} GiB is at the floor: stopping")
            if self.stop:
                return self.stop
        if t >= self.stop_at:
            return "stop-at"
        if deadline is not None and t >= deadline:
            return "deadline"
        return None

    def sleep(self, s: float) -> None:
        end = now() + s
        while now() < end and not self.stop:
            time.sleep(min(0.25, max(0.0, end - now())))

    def fresh_check(self, deadline: Optional[float] = None) -> Optional[str]:
        """check() with the slow checks forced now: before a GPU child is started."""
        self.last_slow = 0.0
        return self.check(deadline)

    def settle(self) -> Optional[str]:
        """A GPU child ended on its own. A restore stops the distill-* units one by one, so
        this runner's SIGTERM may follow a moment later: wait DJR_SETTLE_S for it, then run
        the slow checks. The stop reason, if any."""
        self.sleep(self.cfg.settle_s)
        return self.fresh_check()

    def external_stop(self, what: str, rc: Optional[int]) -> str:
        """A GPU child was stopped from outside the runner (its scope stopped by a restore, or
        by hand): yield the GPU for the rest of the night, like a NO-SWAPS flag."""
        if not self.stop:
            self.stop = "external-stop"
            self.note(f"{what} was stopped from outside the runner (exit {rc}): yielding the GPU")
        return self.stop

    # -- pid files and children -----------------------------------------------

    def pidfile(self, name: str) -> str:
        return os.path.join(self.cfg.run, f"{name}.json")

    def record(self, name: str, proc: subprocess.Popen, signature: List[str], extra: Dict[str, Any]) -> None:
        write_json(self.pidfile(name), {"pid": proc.pid, "pgid": proc.pid, "signature": signature,
                                        "run_id": self.run_id, "started_at": iso(now()), **extra})

    def forget(self, name: str) -> None:
        try:
            os.unlink(self.pidfile(name))
        except FileNotFoundError:
            pass

    def cleanup_orphans(self) -> None:
        """Stop processes an earlier (crashed) run recorded, if they are still what was recorded."""
        for name in ("child", "server"):
            rec = read_json(self.pidfile(name))
            if not isinstance(rec, dict):
                self.forget(name)
                continue
            pid, pgid, sig = int(rec.get("pid", 0)), int(rec.get("pgid", 0)), rec.get("signature") or []
            cmd = pid_cmdline(pid) if pid > 1 and pid_alive(pid) else ""
            ours = bool(cmd) and all(s in cmd for s in sig)
            same_group = False
            if ours:
                try:
                    same_group = os.getpgid(pid) == pgid
                except OSError:
                    same_group = False
            if ours and same_group and pgid > 1:
                self.note(f"stopping a {name} left by run {rec.get('run_id')} (pid {pid})")
                self.stop_group(pgid, self.cfg.server_grace)
            elif cmd:
                self.log(f"{name} pid file names pid {pid}, now another process: left alone")
            self.forget(name)

    def stop_group(self, pgid: int, grace: float, proc: Optional[subprocess.Popen] = None) -> None:
        """SIGTERM the process group of a recorded PID, SIGKILL after *grace*. With *proc*
        (our own child, the group leader) the leader is reaped as it exits, so a zombie
        never keeps the group "alive", and a leader already reaped is never signalled
        (its PID may belong to someone else by now)."""
        if proc is not None and proc.poll() is not None:
            return
        if not killpg(pgid, signal.SIGTERM):
            return

        def alive() -> bool:
            if proc is not None:
                proc.poll()
            return group_alive(pgid)

        end = now() + grace
        while now() < end and alive():
            time.sleep(0.2)
        if alive():
            self.log(f"process group {pgid} still alive after {grace:.0f}s: SIGKILL")
            killpg(pgid, signal.SIGKILL)
            end = now() + 10
            while now() < end and alive():
                time.sleep(0.2)

    def spawn(self, name: str, cmd: List[str], log_path: str, *, cwd: str, env: Dict[str, str],
              signature: List[str], extra: Optional[Dict[str, Any]] = None) -> subprocess.Popen:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as lf:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=cwd, env=env, start_new_session=True)
        self.record(name, proc, signature, extra or {})
        if name == "child":
            self.child = proc
        return proc

    def supervise(self, proc: subprocess.Popen, deadline: Optional[float], grace: float,
                  extra: Optional[Callable[[], Optional[str]]] = None) -> Tuple[Optional[int], Optional[str]]:
        """Wait for *proc*; stop its group on a stop reason. (returncode, reason)."""
        try:
            while True:
                rc = proc.poll()
                if rc is not None:
                    return rc, None
                reason = self.check(deadline) or (extra() if extra else None)
                if reason:
                    self.log(f"stopping pid {proc.pid} ({reason})")
                    self.stop_group(proc.pid, grace, proc)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    return proc.returncode, reason
                try:
                    proc.wait(timeout=self.cfg.poll_s)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if self.child is proc:
                self.child = None

    def scope_prefix(self, unit: str) -> List[str]:
        return (["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", "--"]
                if self.scopes else [])

    def probe_scopes(self) -> None:
        mode = self.cfg.use_scopes
        if mode == "0" or not shutil.which("systemd-run"):
            self.scopes = False
        else:
            try:
                cp = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect",
                                     f"--unit=distill-runner-check-{self.run_id}", "--", "true"],
                                    capture_output=True, text=True, timeout=30)
                self.scopes = cp.returncode == 0
            except (OSError, subprocess.SubprocessError):
                self.scopes = False
        if not self.scopes:
            self.note("no systemd scopes: a restore cannot stop this runner's GPU children by unit")

    # -- GPU / server -----------------------------------------------------------

    def wait_gpu_free(self, timeout: float) -> Tuple[bool, str]:
        end = now() + timeout
        used = gpu_used_mib()
        while (used is None or used > self.cfg.gpu_max_used) and now() < end and not self.stop:
            self.sleep(2)
            used = gpu_used_mib()
        if used is None:
            return False, "nvidia-smi gave no reading"
        if used > self.cfg.gpu_max_used:
            return False, f"GPU holds {used} MiB (> {self.cfg.gpu_max_used}): {gpu_apps()}"
        return True, f"{used} MiB used"

    def server_cmd(self, model: str, ctx: int, ub: int) -> List[str]:
        return [self.cfg.llama_server, "-m", model, "-c", str(ctx), "-np", "1", "-fa", "on",
                "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "-ngl", "99", "-fit", "off",
                "-b", str(max(2048, ub)), "-ub", str(ub),
                "--host", self.cfg.host, "--port", str(self.cfg.port)]

    def start_server(self, label: str, model: str, ctx: int, ub: int) -> Tuple[Optional[str], str]:
        """Start and wait healthy. (None, info) on success; (reason, detail) otherwise.
        reason: "load-failed" (try a smaller context), "stopped", or "error"."""
        if port_in_use(self.cfg.host, self.cfg.port):
            return "error", f"{self.cfg.host}:{self.cfg.port} is already taken"
        ok, info = self.wait_gpu_free(self.cfg.gpu_release_s)
        if not ok:
            return ("stopped" if self.stop else "error"), info
        reason = self.fresh_check()
        if reason:
            return "stopped", reason
        cmd = self.server_cmd(model, ctx, ub)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        full = self.scope_prefix(f"distill-llama-{label}-c{ctx}-{stamp}") + cmd
        log_path = os.path.join(self.run_dir, f"server-{label}-c{ctx}.log")
        self.log(f"server: {' '.join(cmd)}")
        proc = self.spawn("server", full, log_path, cwd=self.cfg.home, env=dict(os.environ),
                          signature=[self.cfg.llama_server, f"--port {self.cfg.port}"],
                          extra={"label": label, "ctx": ctx})
        self.server = {"proc": proc, "label": label, "ctx": ctx, "log": log_path, "cmd": cmd}
        self.progress["server"] = {"pid": proc.pid, "label": label, "ctx": ctx, "state": "loading",
                                   "started_at": iso(now())}
        self.save()
        base = f"http://{self.cfg.host}:{self.cfg.port}"
        t0 = now()
        while True:
            rc = proc.poll()
            if rc is not None:
                tail = log_tail(log_path, 4)
                self.stop_server()
                if "transient scope" in tail or "transient unit" in tail:
                    # systemd-run could not make the scope: not the model's fault, no step-down
                    return "error", f"systemd-run failed (exit {rc}): {tail}"
                reason = self.settle()
                if reason or stopped_by_signal(rc, zero_is_stop=True):
                    # stopped while loading (a restore): not a context that does not fit
                    return "stopped", reason or self.external_stop("the loading llama-server", rc)
                return "load-failed", f"exit {rc} after {now() - t0:.0f}s: {tail}"
            reason = self.check(None)
            if reason in STOP_REASONS or (reason and reason.startswith("signal")):
                self.stop_server()
                return "stopped", reason or ""
            code, body = http_json(base + "/health", timeout=3)
            if code == 200:
                break
            if now() - t0 > self.cfg.server_load_s:
                self.stop_server()
                return "error", f"not healthy after {self.cfg.server_load_s:.0f}s"
            time.sleep(1)
        _, props = http_json(base + "/props", timeout=10)
        _, models = http_json(base + "/v1/models", timeout=10)
        dgs = (props or {}).get("default_generation_settings") or {}
        n_ctx = dgs.get("n_ctx") or (props or {}).get("n_ctx")
        mid = (((models or {}).get("data") or [{}])[0] or {}).get("id")
        if n_ctx != ctx:
            self.stop_server()
            return "error", f"server reports n_ctx {n_ctx}, asked for {ctx}"
        used = gpu_used_mib()
        self.progress["server"].update(state="healthy", ready_s=round(now() - t0, 1), model_id=mid,
                                       vram_used_mib=used)
        self.save()
        return None, f"healthy in {now() - t0:.0f}s, n_ctx {n_ctx}, model {mid}, VRAM {used} MiB"

    def server_died(self) -> Optional[str]:
        s = self.server
        if s and s["proc"].poll() is not None:
            return "server-died"
        return None

    def stop_server(self) -> None:
        s = self.server
        if not s:
            return
        proc = s["proc"]
        self.log(f"stopping llama-server pid {proc.pid}")
        self.stop_group(proc.pid, self.cfg.server_grace, proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.forget("server")
        self.server = None
        if self.progress.get("server"):
            self.progress["server"].update(state="stopped", stopped_at=iso(now()),
                                           exit=proc.returncode)
        self.save()

    # -- step state -------------------------------------------------------------

    def step_dir(self, step: Dict[str, Any]) -> str:
        return os.path.join(self.cfg.results, step["out"])

    def load_state(self, step: Dict[str, Any]) -> Dict[str, Any]:
        st = read_json(os.path.join(self.step_dir(step), "step.json"))
        if not isinstance(st, dict) or st.get("schema") != STEP_SCHEMA:
            st = {"schema": STEP_SCHEMA, "name": step["name"], "status": "not-started",
                  "failures": 0, "sets": {}, "lengths": {}, "history": []}
        return st

    def save_state(self, step: Dict[str, Any], st: Dict[str, Any]) -> None:
        write_json(os.path.join(self.step_dir(step), "step.json"), st)

    def pstep(self, step: Dict[str, Any]) -> Dict[str, Any]:
        for p in self.progress["steps"]:
            if p["name"] == step["name"]:
                return p
        p = {"name": step["name"], "status": "pending", "out_dir": self.step_dir(step)}
        self.progress["steps"].append(p)
        return p

    # -- the queue ----------------------------------------------------------------

    def steps(self) -> List[Dict[str, Any]]:
        ab: List[Dict[str, Any]] = [
            {"name": "teacher-ab", "kind": "ab", "out": "teacher-ab", "min_s": self.cfg.ab_min_s,
             "must": False},
        ] if self.cfg.ab else []
        smoke: List[Dict[str, Any]] = [
            {"name": "train-smoke", "kind": "train", "out": "base-9b/train-smoke-max-derived",
             "min_s": self.cfg.smoke_min_s, "must": False},
        ] if self.cfg.smoke else []
        full: List[Dict[str, Any]] = [
            {"name": "train-full", "kind": "train-full", "out": "base-9b/train-full-max-derived",
             "min_s": self.cfg.train_min_s, "must": False},
        ] if self.cfg.train else []
        queue = smoke + full + ab + [
            {"name": "base-9b-baseline", "kind": "baseline", "label": "base-9b", "out": "base-9b/baseline",
             "find": find_9b, "fixed_ctx": 262144, "ub": self.cfg.ub_9b, "min_s": self.cfg.min_9b_s,
             "must": True},
            {"name": "qwen38-27b-feasibility", "kind": "feasibility", "out": "qwen38-27b/feasibility",
             "min_s": self.cfg.feas_child_s + 2 * self.cfg.feas_margin, "must": True},
            {"name": "qwen38-27b-baseline", "kind": "baseline", "label": "qwen38-27b",
             "out": "qwen38-27b/baseline", "find": find_27b_gguf, "fixed_ctx": None,
             "ub": self.cfg.ub_bake, "min_s": self.cfg.min_bake_s, "must": False},
            {"name": "glimmer-30b-baseline", "kind": "baseline", "label": "glimmer-30b",
             "out": "glimmer-30b/baseline", "find": find_glimmer, "fixed_ctx": None,
             "ub": self.cfg.ub_bake, "min_s": self.cfg.min_bake_s, "must": False,
             "optional_model": True},
        ]
        if self.cfg.only:
            queue = [s for s in queue if s["name"] in self.cfg.only]
        return queue

    def reserve_after(self, steps: List[Dict[str, Any]], i: int) -> float:
        """Time to keep for later must-run steps that are still to do."""
        r = 0.0
        for later in steps[i + 1:]:
            if not later["must"]:
                continue
            st = self.load_state(later)
            if st["status"] == "done" or st["failures"] >= STEP_FAILURES:
                continue
            if later["kind"] == "feasibility" and not self.feas_inputs_ok()[0]:
                continue
            r += self.cfg.feas_reserve if later["kind"] == "feasibility" else later["min_s"]
        return r

    def feas_inputs_ok(self) -> Tuple[bool, str]:
        c = self.cfg
        missing = [p for p in (c.train_py, c.feas_script, c.lengths) if not os.path.exists(p)]
        model = Path(c.model_27b_bnb)
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            missing.append(f"{c.model_27b_bnb} (config.json + *.safetensors)")
        elif download_unfinished(str(model)):
            missing.append(str(download_unfinished(str(model))))
        return (not missing), ("missing: " + ", ".join(missing) if missing else "")

    def execute(self) -> int:
        steps = self.steps()
        self.progress["state"] = "running"
        self.save()
        failed = False
        for i, step in enumerate(steps):
            p = self.pstep(step)
            st = self.load_state(step)
            if self.stop:
                p.update(status="not-started", detail=f"runner stopping ({self.stop})")
                continue
            if st["status"] == "done":
                p.update(status="done", detail="done in an earlier window", summary=st.get("summary"))
                self.save()
                continue
            if st["failures"] >= STEP_FAILURES:
                p.update(status="failed", detail=f"failed {st['failures']} times; skipped until checked: "
                                                 f"{st.get('last_error', '')}"[:300])
                self.save()
                continue
            deadline = self.stop_at - self.reserve_after(steps, i)
            left = deadline - now()
            if left < step["min_s"]:
                p.update(status="not-started",
                         detail=f"{max(0, left) / 60:.0f} min left before its deadline "
                                f"{hm(deadline)} (needs {step['min_s'] / 60:.0f})")
                self.save()
                continue
            t0 = now()
            p.update(status="running", started_at=iso(t0), deadline=iso(deadline), detail="")
            self.progress["current"] = {"step": step["name"], "set": None, "since": iso(t0)}
            self.save()
            self.log(f"=== {step['name']} (deadline {iso(deadline)})")
            try:
                if step["kind"] == "baseline":
                    status, detail = self.run_baseline(step, st, deadline, p)
                elif step["kind"] == "ab":
                    status, detail = self.run_ab(step, st, deadline, p)
                elif step["kind"] == "train":
                    status, detail = self.run_smoke(step, st, deadline, p)
                elif step["kind"] == "train-full":
                    status, detail = self.run_train_full(step, st, deadline, p)
                else:
                    status, detail = self.run_feasibility(step, st, deadline, p)
            except Exception as exc:  # noqa: BLE001 - one step's bug must not end the window
                status, detail = "failed", f"runner error: {type(exc).__name__}: {exc}"
                self.stop_server()
            if status == "failed":
                st["failures"] = st.get("failures", 0) + 1
                st["last_error"] = detail[:500]
                failed = True
            if status in ("done", "partial", "failed", "skipped"):
                st["status"] = "done" if status == "done" else (
                    st["status"] if status == "skipped" else status)
            st["history"].append({"run_id": self.run_id, "started_at": iso(t0), "ended_at": iso(now()),
                                  "outcome": status, "detail": detail[:500]})
            self.save_state(step, st)
            p.update(status=status, ended_at=iso(now()), detail=detail[:500])
            self.progress["current"] = None
            self.save()
            self.log(f"=== {step['name']}: {status}: {detail}")
            if status == "failed":
                self.notify([f"4090 runner: {step['name']} FAILED", detail[:300]])
        return 3 if self.stop in ("no-swaps", "production-unit-active", "disk-floor", "external-stop") else (
            143 if self.stop == "sigterm" else (1 if failed else 0))

    # -- baseline step -----------------------------------------------------------

    def results_progress(self, out: str, total: int) -> int:
        done: Dict[Tuple[Any, Any], str] = {}
        try:
            with open(out, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r, dict):
                        key = (r.get("set"), r.get("id"))
                        if done.get(key) == "ok" and r.get("status") != "ok":
                            continue
                        done[key] = r.get("status") or ""
        except OSError:
            return 0
        return min(total, sum(1 for v in done.values() if v == "ok"))

    def score_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(self.cfg.code, "src")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def summarize(self, label: str, results: str, cases: List[str], summary_out: str,
                  log_path: str) -> Tuple[int, Optional[str]]:
        """(exit code, stop reason): a reason other than "deadline" (its own 15-min cap) means
        the runner is stopping, not that the summary failed."""
        md = summary_out[:-5] + ".md" if summary_out.endswith(".json") else summary_out + ".md"
        cmd = [self.cfg.score_py, self.cfg.score_script, "summarize", "--results", results,
               "--label", label, "--summary-out", summary_out, "--markdown-out", md]
        for c in cases:
            cmd += ["--cases", c]
        proc = self.spawn("child", cmd, log_path, cwd=os.path.dirname(results), env=self.score_env(),
                          signature=["score.py", "summarize"])
        rc, reason = self.supervise(proc, now() + 900, self.cfg.score_grace)
        self.forget("child")
        return (124 if reason else (rc if rc is not None else 1)), reason

    def ctx_candidates(self, step: Dict[str, Any], st: Dict[str, Any], model: str) -> Tuple[List[int], str]:
        if step["fixed_ctx"]:
            return [step["fixed_ctx"]], ""
        try:
            facts = gguf_facts(model)
        except (OSError, ValueError) as e:
            return [], f"cannot read the GGUF header: {e}"
        native = facts.get("native_ctx")
        if not isinstance(native, int) or native <= 0:
            return [], f"no context_length in the GGUF header ({facts.get('arch')})"
        sup = arch_supported(self.cfg.llama_src, facts.get("arch"))
        if sup is False:
            return [], f"llama.cpp has no architecture {facts.get('arch')!r}"
        cands = [native] + [c for c in BUCKET_CTX if c < native]
        bad = st.get("ctx_bad") or []
        if bad:
            cands = [c for c in cands if c < min(bad)]
        chosen = st.get("ctx_chosen")
        if chosen in cands:
            cands = [chosen] + [c for c in cands if c < chosen]
        return cands, ""

    def run_baseline(self, step: Dict[str, Any], st: Dict[str, Any], deadline: float,
                     p: Dict[str, Any]) -> Tuple[str, str]:
        cfg = self.cfg
        model, why = step["find"](cfg)
        if not model:
            return "skipped", f"model: {why}"
        sdir = self.step_dir(step)
        Path(sdir).mkdir(parents=True, exist_ok=True)
        label = step["label"]
        sets = []
        for name, fname, timeout in SETS:
            path = cfg.case_file(fname)
            if not os.path.isfile(path):
                return "failed", f"case file missing: {path}"
            sets.append((name, path, timeout))
        totals = {name: count_lines(path) for name, path, _ in sets}
        for name, path, _ in sets:
            s = st["sets"].setdefault(name, {"status": "pending", "attempts": 0})
            s["total"] = totals[name]
            s["done"] = self.results_progress(os.path.join(sdir, f"{name}.jsonl"), totals[name])
        p["sets"] = {n: dict(v) for n, v in st["sets"].items()}
        p["model"] = os.path.basename(model)
        self.save_state(step, st)
        self.save()
        pending = [s for s in sets if st["sets"][s[0]]["status"] not in ("done", "done-with-errors")]
        if pending:
            cands, why = self.ctx_candidates(step, st, model)
            if not cands:
                return ("skipped" if step.get("optional_model") else "failed"), why
            started = False
            tried = []
            for ctx in cands:
                reason, detail = self.start_server(label, model, ctx, step["ub"])
                if reason is None:
                    self.log(f"server up: {detail}")
                    st["ctx_chosen"] = ctx
                    st["server_cmd"] = " ".join(self.server["cmd"]) if self.server else ""
                    p["ctx"] = ctx
                    p["ctx_tried"] = tried + [f"{ctx} ok"]
                    started = True
                    break
                tried.append(f"{ctx} {reason}")
                self.log(f"server at -c {ctx}: {reason}: {detail}")
                p["ctx_tried"] = tried
                self.save()
                if reason != "load-failed" or step["fixed_ctx"]:
                    if reason == "stopped":
                        return "partial", f"stopped while loading ({detail})"
                    return "failed", f"server at -c {ctx}: {reason}: {detail}"
            self.save_state(step, st)
            if not started:
                return "failed", "no context loaded: " + "; ".join(tried)
            ctx = st["ctx_chosen"]
            prev = st.get("ctx_scored")
            if prev and prev != ctx:
                # like for like: never finish a baseline scored partly at another context
                moved = self.archive_results(sdir, prev)
                st.setdefault("ctx_history", []).append(
                    {"run_id": self.run_id, "from": prev, "to": ctx, "archived": moved,
                     "why": "the context changed (a crash at it, or it no longer loads): every set re-runs"})
                st["sets"] = {}
                for name, path, _ in sets:
                    st["sets"][name] = {"status": "pending", "attempts": 0, "total": totals[name], "done": 0}
                st.pop("summary", None)
                pending = list(sets)
                p["sets"] = {n: dict(v) for n, v in st["sets"].items()}
                self.note(f"{step['name']}: context {prev} -> {ctx}; results at {prev} moved to {moved}, "
                          "every set runs again")
            st["ctx_scored"] = ctx
            self.save_state(step, st)
            outcome = self.run_sets(step, st, pending, label, sdir, deadline, p)
            self.stop_server()
            if outcome:
                return outcome
        if all(st["sets"][n]["status"] in ("done", "done-with-errors") for n, _, _ in sets):
            return self.finish_baseline(step, st, sets, label, sdir, p)
        left = [f"{n} {st['sets'][n]['done']}/{st['sets'][n]['total']}" for n, _, _ in sets
                if st["sets"][n]["status"] not in ("done", "done-with-errors")]
        return "partial", "left for the next window: " + ", ".join(left)

    def archive_results(self, sdir: str, ctx: int) -> str:
        """Move a step's result files (not step.json) into ctx-<ctx>-<run id>/; nothing is deleted."""
        dest = os.path.join(sdir, f"ctx-{ctx}-{self.run_id}")
        moved = 0
        for f in sorted(os.listdir(sdir)):
            src = os.path.join(sdir, f)
            if f == "step.json" or f.startswith("step.json.tmp") or not os.path.isfile(src):
                continue
            Path(dest).mkdir(parents=True, exist_ok=True)
            os.replace(src, os.path.join(dest, f))
            moved += 1
        return dest if moved else "(no result files)"

    def run_sets(self, step: Dict[str, Any], st: Dict[str, Any], pending: List[Tuple[str, str, int]],
                 label: str, sdir: str, deadline: float, p: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        cfg = self.cfg
        base = f"http://{cfg.host}:{cfg.port}"
        for name, path, timeout in pending:
            s = st["sets"][name]
            out = os.path.join(sdir, f"{name}.jsonl")
            if deadline - now() < cfg.min_set_s:
                s["status"] = "partial"
                break
            self.progress["current"] = {"step": step["name"], "set": name, "since": iso(now())}
            p["sets"][name] = dict(s, status="running")
            self.save()
            cmd = [cfg.score_py, cfg.score_script, "run", "--base-url", base, "--label", label,
                   "--cases", path, "--out", out, "--concurrency", "1", "--timeout", str(timeout)]
            log_path = os.path.join(self.run_dir, f"{label}-{name}.run.log")
            self.log(f"{name}: score.py run ({s['done']}/{s['total']} done before)")
            proc = self.spawn("child", cmd, log_path, cwd=sdir, env=self.score_env(),
                              signature=["score.py", f"--label {label}"])
            rc, reason = self.supervise(proc, deadline, cfg.score_grace, self.server_died)
            self.forget("child")
            s["done"] = self.results_progress(out, s["total"])
            s["last_rc"] = rc
            if reason:
                s["status"] = "partial"
                p["sets"][name] = dict(s)
                self.save_state(step, st)
                if reason == "server-died":
                    srv_rc = self.server["proc"].returncode if self.server else None
                    tail = log_tail(self.server["log"], 3) if self.server else ""
                    why = self.settle()
                    if why or stopped_by_signal(srv_rc, zero_is_stop=True):
                        # a restore (or a hand) stopped it: not a crash, no step-down, no failure
                        why = why or self.external_stop("the runner's llama-server", srv_rc)
                        return "partial", (f"llama-server was stopped during {name} ({why}, exit {srv_rc}); "
                                           f"{s['done']}/{s['total']} done")
                    ctx = st.get("ctx_chosen")
                    if not step["fixed_ctx"] and ctx:
                        st.setdefault("ctx_bad", []).append(ctx)
                        st.pop("ctx_chosen", None)
                    self.save_state(step, st)
                    return "failed", (f"llama-server died during {name} ({s['done']}/{s['total']}, "
                                      f"exit {srv_rc}): {tail}")
                if reason not in ("deadline", "stop-at"):
                    return "partial", f"stopped during {name} ({reason}); {s['done']}/{s['total']} done"
                break
            if rc == 0:
                s["status"] = "done"
            elif rc == 3:
                s["attempts"] = s.get("attempts", 0) + 1
                s["status"] = "done-with-errors" if s["attempts"] >= SET_RETRIES else "errors"
            elif rc == 2:
                s["status"] = "failed"
                p["sets"][name] = dict(s)
                self.save_state(step, st)
                return "failed", f"score.py refused {name} (exit 2): {log_tail(log_path, 2)}"
            else:
                s["status"] = "failed"
                p["sets"][name] = dict(s)
                self.save_state(step, st)
                return "failed", f"score.py exit {rc} on {name}: {log_tail(log_path, 2)}"
            s["summary"] = headline(read_json(os.path.join(sdir, f"{name}.summary.json")), name)
            s["ctx"] = st.get("ctx_scored")
            s["scorer"] = scorer_tag(cfg)
            p["sets"][name] = dict(s)
            self.save_state(step, st)
            self.save()
        return None

    def finish_baseline(self, step: Dict[str, Any], st: Dict[str, Any], sets: List[Tuple[str, str, int]],
                        label: str, sdir: str, p: Dict[str, Any]) -> Tuple[str, str]:
        combined = os.path.join(sdir, f"{label}.all.jsonl")
        tmp = combined + ".tmp"
        with open(tmp, "wb") as fo:
            for name, _, _ in sets:
                part = os.path.join(sdir, f"{name}.jsonl")
                if os.path.isfile(part):
                    with open(part, "rb") as fi:
                        data = fi.read()
                    fo.write(data if data.endswith(b"\n") or not data else data + b"\n")
        os.replace(tmp, combined)
        summary = os.path.join(sdir, f"{label}.summary.json")
        rc, reason = self.summarize(label, combined, [path for _, path, _ in sets], summary,
                                    os.path.join(self.run_dir, f"{label}-summary.log"))
        if reason and reason != "deadline":
            # every set is scored; only the CPU-only summary is left for the next window
            return "partial", f"combined summarize stopped ({reason}); it runs again next window"
        if rc != 0:
            return "failed", f"combined summarize exit {rc}" + (" (15-min cap)" if reason else "")
        sm = read_json(summary) or {}
        st["summary"] = {n: headline(sm, n) for n, _, _ in sets}
        tag = scorer_tag(self.cfg)
        ctxs = sorted({st["sets"][n].get("ctx") for n, _, _ in sets if st["sets"][n].get("ctx")})
        scorers = sorted({st["sets"][n].get("scorer") for n, _, _ in sets if st["sets"][n].get("scorer")})
        st["summary_info"] = {"n_ctx": st.get("ctx_scored"), "set_ctx": ctxs, "summarized_by": tag,
                              "scored_by": scorers}
        p["summary"] = st["summary"]
        errs = [n for n, _, _ in sets if st["sets"][n]["status"] == "done-with-errors"]
        mixed = []
        if len(ctxs) > 1:
            mixed.append(f"sets scored at several contexts {ctxs}")
        if len(scorers) > 1 or (scorers and scorers != [tag]):
            mixed.append(f"rows scored by {', '.join(scorers)}, summarized by {tag}")
        return "done", (f"summary {summary} (n_ctx {st.get('ctx_scored')}, {tag})"
                        + (f"; errors left in {', '.join(errs)}" if errs else "")
                        + (f"; NOTE {'; '.join(mixed)}" if mixed else ""))

    # -- teacher A/B step -------------------------------------------------------------

    def ab_inputs_ok(self) -> Tuple[bool, str]:
        c = self.cfg
        missing = [p for p in (c.ab_driver, os.path.join(c.ab_code, "scripts", "distill", "teacher_ab.py"),
                               c.ab_tasks, c.ab_iq3, c.ab_q4, c.ab_py, c.llama_server) if not os.path.exists(p)]
        if c.ab_pack and not os.path.exists(c.ab_pack):
            missing.append(c.ab_pack)
        return (not missing), ("missing: " + ", ".join(missing) if missing else "")

    def ab_cmd(self, until: Optional[float]) -> List[str]:
        """The driver's command line (``until`` None: its dry run)."""
        c = self.cfg
        cmd = ["/bin/bash", c.ab_driver] + (["--execute", "--until", str(iso(until))] if until is not None else [])
        cmd += ["--tasks", c.ab_tasks, "--out", c.ab_out, "--iq3", c.ab_iq3, "--q4", c.ab_q4,
                "--server", c.llama_server, "--port", str(c.port), "--py", c.ab_py, "--code", c.ab_code]
        if c.ab_iq3_sha:
            cmd += ["--iq3-sha256", c.ab_iq3_sha]
        if c.ab_q4_sha:
            cmd += ["--q4-sha256", c.ab_q4_sha]
        if c.ab_pack:
            cmd += ["--pack", c.ab_pack]
        if until is not None and c.notify and os.access(c.notify, os.X_OK):
            cmd += ["--notify-cmd", c.notify]
        return cmd

    def ab_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TEACHER_AB_CODE", "TEACHER_AB_PY")}
        env.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
        return env

    def run_ab(self, step: Dict[str, Any], st: Dict[str, Any], deadline: float,
               p: Dict[str, Any]) -> Tuple[str, str]:
        """The teacher quant A/B through its own driver, time-boxed (see the module docstring)."""
        cfg = self.cfg
        ok, why = self.ab_inputs_ok()
        if not ok:
            return "skipped", why
        until = min(deadline, now() + cfg.ab_max_s)
        if until - now() < step["min_s"]:
            return "skipped", (f"{max(0.0, until - now()) / 60:.0f} min left in its time box "
                               f"(needs {step['min_s'] / 60:.0f})")
        sdir = self.step_dir(step)
        Path(sdir).mkdir(parents=True, exist_ok=True)
        if port_in_use(cfg.host, cfg.port):
            return "failed", f"{cfg.host}:{cfg.port} is already taken"
        ok, info = self.wait_gpu_free(cfg.gpu_release_s)
        if not ok:
            return ("partial" if self.stop else "failed"), f"GPU not free: {info}"
        reason = self.fresh_check(until)
        if reason:
            return "partial", f"not started ({reason})"
        cmd = self.ab_cmd(until)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        full = self.scope_prefix(f"distill-teacher-ab-{stamp}") + cmd
        log_path = os.path.join(self.run_dir, "teacher-ab.log")
        self.progress["current"] = {"step": step["name"], "set": "teacher_ab_run.sh", "since": iso(now())}
        p["until"] = iso(until)
        self.save()
        self.log(f"teacher A/B until {iso(until)}: {' '.join(cmd)}")
        proc = self.spawn("child", full, log_path, cwd=sdir, env=self.ab_env(),
                          signature=["teacher_ab_run.sh", "--execute"])
        # the driver ends itself at --until; this deadline is only the backstop
        rc, reason = self.supervise(proc, until + cfg.ab_grace + 120, cfg.ab_grace)
        self.forget("child")
        rep = read_json(os.path.join(cfg.ab_out, "ab_report.json"))
        dec = (rep.get("decision") or {}) if isinstance(rep, dict) else {}
        verdict = str(dec.get("verdict") or "")
        summary = ""
        try:
            summary = Path(cfg.ab_out, "ab_summary.txt").read_text(errors="replace").strip().splitlines()[0]
        except (OSError, IndexError):
            pass
        if verdict:
            p["decision"] = {"verdict": verdict, "a_minus_b_pts": dec.get("a_minus_b_pts"),
                             "pairs": dec.get("pairs"), "report": os.path.join(cfg.ab_out, "ab_report.json")}
            st["decision"] = p["decision"]
        if not reason and stopped_by_signal(rc):
            # a restore stopped its distill-teacher-ab-* scope: our own SIGTERM may be a moment behind
            reason = self.settle() or self.external_stop("the teacher A/B driver", rc)
        if reason:
            return "partial", f"stopped ({reason}); resumes next window" + (f"; {summary}" if summary else "")
        if rc == 0:
            return "done", summary or f"decision: {verdict}"
        if rc == 1 and verdict == "invalid":
            return "failed", summary or "the A/B report is invalid"
        if rc == 1:
            return "partial", (summary or "not finished") + " (time box or work left; resumes next window)"
        return "failed", f"teacher_ab_run.sh exit {rc}: {log_tail(log_path, 3)}"

    # -- QLoRA smoke step -------------------------------------------------------------

    def smoke_base_rev(self) -> str:
        c = self.cfg
        if c.smoke_base_rev:
            return c.smoke_base_rev
        man = read_json(os.path.join(c.smoke_base, "MANIFEST.json"))
        rev = man.get("revision") if isinstance(man, dict) else None
        return rev if isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev) else ""

    def smoke_inputs_ok(self) -> Tuple[bool, str]:
        c = self.cfg
        missing = [p for p in (c.train_py, c.smoke_script, c.smoke_data, c.smoke_tok) if not os.path.exists(p)]
        for name in ("render.py", "artifact_policy.py"):
            p = os.path.join(os.path.dirname(c.smoke_script), name)
            if not os.path.exists(p):
                missing.append(p)
        if c.smoke_val and not os.path.exists(c.smoke_val):
            missing.append(c.smoke_val)
        base = Path(c.smoke_base)
        if not (base / "config.json").is_file() or not list(base.glob("*.safetensors")):
            missing.append(f"{c.smoke_base} (config.json + *.safetensors)")
        elif download_unfinished(str(base)):
            missing.append(str(download_unfinished(str(base))))
        if not self.smoke_base_rev():
            missing.append(f"base revision (DJR_SMOKE_BASE_REV or {c.smoke_base}/MANIFEST.json)")
        return (not missing), ("missing: " + ", ".join(missing) if missing else "")

    def smoke_cmd(self, until: Optional[float]) -> List[str]:
        """The trainer's command line (``until`` None: the plan's placeholder)."""
        c = self.cfg
        cmd = [c.train_py, c.smoke_script, "--data", c.smoke_data]
        if c.smoke_val:
            cmd += ["--val", c.smoke_val]
        cmd += ["--tokenizer-dir", c.smoke_tok, "--base-model", c.smoke_base,
                "--base-revision", self.smoke_base_rev() or "<missing>", "--out", c.smoke_out,
                "--log-dir", os.path.join(c.smoke_out, "logs"), "--smoke",
                "--stop-at", str(iso(until)) if until is not None else "<time box end>"]
        if os.path.isfile(os.path.join(c.smoke_out, "run_state.json")):
            cmd.append("--resume")
        if c.smoke_md_ok:
            cmd.append("--i-have-operator-approval-for-max-derived-data")
        return cmd

    def smoke_env(self) -> Dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
        env.update(PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                   TRANSFORMERS_OFFLINE="1")
        return env

    def run_smoke(self, step: Dict[str, Any], st: Dict[str, Any], deadline: float,
                  p: Dict[str, Any]) -> Tuple[str, str]:
        """The 9B QLoRA smoke (train_qlora.py --smoke), time-boxed (see the module docstring)."""
        cfg = self.cfg
        ok, why = self.smoke_inputs_ok()
        if not ok:
            return "skipped", why
        until = min(deadline, now() + cfg.smoke_max_s)
        if until - now() < step["min_s"]:
            return "skipped", (f"{max(0.0, until - now()) / 60:.0f} min left in its time box "
                               f"(needs {step['min_s'] / 60:.0f})")
        sdir = self.step_dir(step)
        Path(sdir).mkdir(parents=True, exist_ok=True)
        ok, info = self.wait_gpu_free(cfg.gpu_release_s)
        if not ok:
            return ("partial" if self.stop else "failed"), f"GPU not free: {info}"
        reason = self.fresh_check(until)
        if reason:
            return "partial", f"not started ({reason})"
        cmd = self.smoke_cmd(until)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        full = self.scope_prefix(f"distill-train-smoke-{stamp}") + cmd
        log_path = os.path.join(self.run_dir, "train-smoke.log")
        self.progress["current"] = {"step": step["name"], "set": "train_qlora.py --smoke", "since": iso(now())}
        p["until"] = iso(until)
        self.save()
        self.log(f"train smoke until {iso(until)}: {' '.join(cmd)}")
        proc = self.spawn("child", full, log_path, cwd=sdir, env=self.smoke_env(),
                          signature=["train_qlora.py", "--smoke"])
        # the trainer checkpoints and exits before --stop-at; this deadline is only the backstop
        rc, reason = self.supervise(proc, until + cfg.smoke_grace + 120, cfg.smoke_grace)
        self.forget("child")
        man = read_json(os.path.join(cfg.smoke_out, "manifest.json"))
        status = str(man.get("status") or "") if isinstance(man, dict) else ""
        sm = (man.get("smoke") or {}) if isinstance(man, dict) else {}
        summary = ""
        if sm.get("val_first") is not None and sm.get("val_last") is not None:
            # the gate since 2026-10-31 (operator): val loss down from step 0 + the fixed probe
            # batch's loss down from before training + finite + round trip
            pf, pl = sm.get("probe_first"), sm.get("probe_last")
            summary = (f"{sm.get('steps')} steps, val {sm['val_first']:.4f} -> {sm['val_last']:.4f}"
                       + (f", probe {pf:.4f} -> {pl:.4f}" if isinstance(pf, (int, float))
                          and isinstance(pl, (int, float)) else ", no probe")
                       + f" ({'ok' if sm.get('loss_decreased') else 'GATE FAILED'}), round trip "
                       f"{'ok' if sm.get('checkpoint_round_trip') else 'FAILED'}")
        elif sm.get("loss_first10") is not None:
            summary = (f"{sm.get('steps')} steps, loss {sm.get('loss_first10'):.4f} -> {sm.get('loss_last10'):.4f}"
                       f" ({'decreased' if sm.get('loss_decreased') else 'DID NOT DECREASE'}), round trip "
                       f"{'ok' if sm.get('checkpoint_round_trip') else 'FAILED'}")
        if summary:
            p["smoke"] = {"status": status, "summary": summary, "adapter": man.get("adapter"),
                          "max_derived": man.get("max_derived")}
            st["smoke"] = p["smoke"]
        if not reason and stopped_by_signal(rc):
            reason = self.settle() or self.external_stop("the QLoRA smoke", rc)
        if reason:
            return "partial", f"stopped ({reason}); resumes next window"
        if rc == 0 and status == "smoke-train-pass":
            return "done", f"smoke-train-pass: {summary}; eval (merge, quantize, score) still to run"
        if rc == 0 and status == "stopped":
            return "partial", "stopped at its time box; resumes next window (--resume)"
        if rc == 1 and status == "smoke-train-fail":
            return "failed", f"smoke-train-fail: {summary}"
        if rc == 3:
            return "failed", f"refused: Max-derived data without both approvals: {log_tail(log_path, 2)}"
        return "failed", f"train_qlora.py exit {rc} (manifest status {status or 'none'}): {log_tail(log_path, 3)}"

    # -- full QLoRA run ---------------------------------------------------------------

    def train_base_rev(self) -> str:
        c = self.cfg
        if c.train_base_rev:
            return c.train_base_rev
        man = read_json(os.path.join(c.train_base, "MANIFEST.json"))
        rev = man.get("revision") if isinstance(man, dict) else None
        return rev if isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev) else ""

    def train_inputs_ok(self) -> Tuple[bool, str]:
        c = self.cfg
        missing = [p for p in (c.train_py, c.train_script, c.train_data, c.train_tok) if not os.path.exists(p)]
        for name in ("render.py", "artifact_policy.py"):
            p = os.path.join(os.path.dirname(c.train_script), name)
            if not os.path.exists(p):
                missing.append(p)
        if c.train_val and not os.path.exists(c.train_val):
            missing.append(c.train_val)
        base = Path(c.train_base)
        if not (base / "config.json").is_file() or not list(base.glob("*.safetensors")):
            missing.append(f"{c.train_base} (config.json + *.safetensors)")
        elif download_unfinished(str(base)):
            missing.append(str(download_unfinished(str(base))))
        if not self.train_base_rev():
            missing.append(f"base revision (DJR_TRAIN_BASE_REV or {c.train_base}/MANIFEST.json)")
        return (not missing), ("missing: " + ", ".join(missing) if missing else "")

    def train_cmd(self, until: Optional[float]) -> List[str]:
        """The full run's command line (``until`` None: the plan's placeholder)."""
        c = self.cfg
        cmd = [c.train_py, c.train_script, "--data", c.train_data]
        if c.train_val:
            cmd += ["--val", c.train_val]
        cmd += ["--tokenizer-dir", c.train_tok, "--base-model", c.train_base,
                "--base-revision", self.train_base_rev() or "<missing>", "--out", c.train_out,
                "--log-dir", os.path.join(c.train_out, "logs"),
                "--epochs", str(c.train_epochs), "--max-seq", str(c.train_max_seq),
                "--grad-accum", str(c.train_grad_accum), "--save-every", str(c.train_save_every),
                "--save-every-minutes", f"{c.train_save_every_min:g}",
                "--keep-checkpoints", str(c.train_keep), "--min-free-gib", f"{c.train_min_free:g}",
                "--stop-at", str(iso(until)) if until is not None else "<stop-at>"]
        if os.path.isfile(os.path.join(c.train_out, "run_state.json")):
            cmd.append("--resume")
        if c.train_md_ok:
            cmd.append("--i-have-operator-approval-for-max-derived-data")
        return cmd

    def train_until(self, deadline: float) -> float:
        c = self.cfg
        return min(deadline, c.train_stop_at) if c.train_stop_at is not None else deadline

    def run_train_full(self, step: Dict[str, Any], st: Dict[str, Any], deadline: float,
                       p: Dict[str, Any]) -> Tuple[str, str]:
        """The full 9B QLoRA run (see the module docstring), resumable across windows."""
        cfg = self.cfg
        ok, why = self.train_inputs_ok()
        if not ok:
            return "skipped", why
        until = self.train_until(deadline)
        if until - now() < step["min_s"]:
            return "skipped", (f"{max(0.0, until - now()) / 60:.0f} min left before its stop-at "
                               f"(needs {step['min_s'] / 60:.0f})")
        free = disk_free(cfg.train_out if os.path.isdir(cfg.train_out) else cfg.home)
        if free < (cfg.train_min_free + 1) * 2**30:
            return "skipped", (f"free disk {free / 2**30:.1f} GiB is under the run's floor "
                               f"{cfg.train_min_free:g} + 1 GiB")
        sdir = self.step_dir(step)
        Path(sdir).mkdir(parents=True, exist_ok=True)
        ok, info = self.wait_gpu_free(cfg.gpu_release_s)
        if not ok:
            return ("partial" if self.stop else "failed"), f"GPU not free: {info}"
        reason = self.fresh_check(until)
        if reason:
            return "partial", f"not started ({reason})"
        cmd = self.train_cmd(until)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        unit = f"distill-train-full-{stamp}"
        scope = self.scope_prefix(unit)
        if scope:  # a restore stopping distill-* gives the trainer time to checkpoint
            scope = scope[:-1] + [f"--property=TimeoutStopSec={cfg.train_stop_timeout}", "--"]
        log_path = os.path.join(self.run_dir, "train-full.log")
        self.progress["current"] = {"step": step["name"], "set": "train_qlora.py", "since": iso(now()),
                                    "unit": f"{unit}.scope" if scope else None}
        p["until"] = iso(until)
        p["unit"] = f"{unit}.scope" if scope else None
        self.save()
        self.log(f"train full until {iso(until)} in {unit}: {' '.join(cmd)}")
        proc = self.spawn("child", scope + cmd, log_path, cwd=sdir, env=self.smoke_env(),
                          signature=["train_qlora.py", cfg.train_out])
        # the trainer checkpoints and exits before --stop-at; this deadline is only the backstop
        rc, reason = self.supervise(proc, until + cfg.train_grace + 120, cfg.train_grace)
        self.forget("child")
        man = read_json(os.path.join(cfg.train_out, "manifest.json"))
        status = str(man.get("status") or "") if isinstance(man, dict) else ""
        summary = ""
        if isinstance(man, dict):
            vh = man.get("val_history") or []
            vs = ""
            if len(vh) >= 2:
                vs = f", val {vh[0].get('val_loss', 0):.4f} -> {vh[-1].get('val_loss', 0):.4f}"
            summary = (f"step {man.get('step')}, {man.get('tokens_seen', 0) / 1e6:.1f}M tokens{vs}"
                       + (f", stop {man.get('reason')}" if man.get("reason") else ""))
            p["train"] = {"status": status, "summary": summary, "adapter": man.get("adapter"),
                          "max_derived": man.get("max_derived")}
            st["train"] = p["train"]
        if not reason and stopped_by_signal(rc):
            reason = self.settle() or self.external_stop("the 9B training run", rc)
        if rc == 0 and status == "done":
            return "done", f"done: {summary}; adapter {man.get('adapter')}"
        if reason:
            return "partial", f"stopped ({reason}): {summary}; resumes next window (--resume)"
        if rc == 0 and status == "stopped":
            return "partial", f"stopped: {summary}; resumes next window (--resume)"
        if rc == 3:
            return "failed", f"refused: Max-derived data without both approvals: {log_tail(log_path, 2)}"
        return "failed", (f"train_qlora.py exit {rc} (manifest status {status or 'none'}): "
                          f"{summary or log_tail(log_path, 3)}")

    # -- feasibility step ---------------------------------------------------------

    def run_feasibility(self, step: Dict[str, Any], st: Dict[str, Any], deadline: float,
                        p: Dict[str, Any]) -> Tuple[str, str]:
        cfg = self.cfg
        ok, why = self.feas_inputs_ok()
        if not ok:
            return "skipped", why
        sdir = self.step_dir(step)
        Path(sdir).mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        env.pop("PYTHONPATH", None)
        lengths = st.setdefault("lengths", {})
        for seq in cfg.feas_seqs:
            L = lengths.setdefault(str(seq), {"attempts": 0})
            rep_path = os.path.join(sdir, f"len-{seq}.json")
            res = feas_result(read_json(rep_path))
            if res is not None and res.get("fits") is not None:
                L.update(measured=True, **slim(res))
                continue
            if L["attempts"] >= STEP_FAILURES:
                continue
            if self.stop:
                break
            left = deadline - now()
            if left < cfg.feas_child_s + cfg.feas_margin:
                self.log(f"feasibility {seq}: {left / 60:.0f} min left; the probe needs "
                         f"{(cfg.feas_child_s + cfg.feas_margin) / 60:.0f}")
                break
            ok, info = self.wait_gpu_free(cfg.gpu_release_s)
            if not ok:
                return ("partial" if self.stop else "failed"), f"GPU not free: {info}"
            if self.fresh_check(deadline):
                break
            stop_iso = datetime.fromtimestamp(deadline).astimezone().isoformat(timespec="seconds")
            cmd = [cfg.train_py, cfg.feas_script, "--data", cfg.lengths, "--base-model", cfg.model_27b_bnb,
                   "--seqs", str(seq), "--execute", "--stop-at", stop_iso, "--out", rep_path]
            if not self.scopes:
                cmd.append("--no-scope")
            log_path = os.path.join(self.run_dir, f"feasibility-{seq}.log")
            self.progress["current"] = {"step": step["name"], "set": f"len {seq}", "since": iso(now())}
            self.save()
            self.log(f"feasibility {seq}: probing")
            if os.path.exists(rep_path):
                os.replace(rep_path, rep_path + ".prev")
            proc = self.spawn("child", cmd, log_path, cwd=sdir, env=env,
                              signature=["feasibility_27b.py", f"--seqs {seq}"])
            rc, reason = self.supervise(proc, deadline, cfg.feas_grace)
            self.forget("child")
            res = feas_result(read_json(rep_path))
            if not reason:
                # the probe ended on its own; if a restore stopped its distill-feas27b-* scope
                # (it reports "stopped (signal n)"), our SIGTERM may be a moment behind
                err = str((res or {}).get("error") or "")
                if err.startswith("stopped") or stopped_by_signal(rc):
                    reason = self.settle() or self.external_stop(f"the feasibility probe at {seq}", rc)
                else:
                    reason = self.fresh_check(deadline)
                    if reason == "deadline":
                        reason = None  # a finished length is kept; the loop checks the time left
            if reason:
                L["last"] = f"stopped ({reason})"
                if res is not None and res.get("fits") is not None:
                    L.update(measured=True, **slim(res))
                p["lengths"] = {k: dict(v) for k, v in lengths.items()}
                self.save_state(step, st)
                break
            if res is None:
                L["attempts"] += 1
                L["last"] = f"exit {rc} without a report: {log_tail(log_path, 2)}"
            elif res.get("fits") is None:
                if "stopped" not in str(res.get("error")) and "not probed" not in str(res.get("error")):
                    L["attempts"] += 1
                L["last"] = str(res.get("error"))[:200]
            else:
                L.update(measured=True, **slim(res))
            p["lengths"] = {k: dict(v) for k, v in lengths.items()}
            self.save_state(step, st)
            self.save()
        p["lengths"] = {k: dict(v) for k, v in lengths.items()}
        merged = self.merge_feasibility(sdir)
        if merged:
            st["decision"] = merged
            p["decision"] = merged
        unmeasured = [s for s in cfg.feas_seqs if not lengths.get(str(s), {}).get("measured")
                      and lengths.get(str(s), {}).get("attempts", 0) < STEP_FAILURES]
        if not unmeasured:
            return "done", (merged or {}).get("say", "merged report written")[:300]
        return "partial", "not measured yet: " + ", ".join(map(str, unmeasured))

    def merge_feasibility(self, sdir: str) -> Optional[Dict[str, Any]]:
        results = []
        for seq in self.cfg.feas_seqs:
            r = feas_result(read_json(os.path.join(sdir, f"len-{seq}.json")))
            if r is not None:
                results.append(r)
        if not results:
            return None
        measured = os.path.join(sdir, "measured.json")
        write_json(measured, {"schema": "distill-feasibility-27b/1", "results": results})
        out = os.path.join(sdir, "feasibility-27b.json")
        cmd = [self.cfg.score_py, self.cfg.feas_script, "--data", self.cfg.lengths,
               "--seqs", ",".join(map(str, self.cfg.feas_seqs)), "--measured", measured, "--out", out]
        with open(os.path.join(self.run_dir, "feasibility-merge.log"), "ab") as lf:
            try:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=300,
                                    env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        rep = read_json(out) if rc == 0 else None
        if not isinstance(rep, dict):
            return {"verdict": "merge-failed", "say": f"merge exit {rc}"}
        d = dict(rep.get("decision") or {})
        d["report"] = out
        return d

    # -- shutdown -------------------------------------------------------------------

    def shutdown(self, rc: int) -> int:
        if self.child is not None:
            self.stop_group(self.child.pid, self.cfg.feas_grace, self.child)
            self.child = None
            self.forget("child")
        self.stop_server()
        reason = self.stop
        if self.progress["state"] != "crashed":
            self.progress["state"] = ("stopped" if reason else ("failed" if rc == 1 else "finished"))
        self.progress["stop_reason"] = reason or ("stop-at" if now() >= self.stop_at else None)
        self.progress["ended_at"] = iso(now())
        self.progress["exit"] = rc
        self.progress["current"] = None
        self.save()
        return rc


def self_unit() -> Optional[str]:
    try:
        text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None
    for part in text.strip().split("/"):
        if part.startswith("distill-") and (part.endswith(".service") or part.endswith(".scope")):
            return part
    return None


def scorer_tag(cfg: Cfg) -> str:
    """"scorer v<SCORER_VERSION> code <COMMIT[:9]>" of the staged code (what scores and summarizes)."""
    ver, commit = "?", "?"
    try:
        m = re.search(r'^SCORER_VERSION\s*=\s*["\']?([\w.]+)', Path(cfg.score_script).read_text(errors="replace"),
                      re.M)
        ver = m.group(1) if m else "?"
    except OSError:
        pass
    try:
        commit = (Path(cfg.code) / "COMMIT").read_text().strip()[:9] or "?"
    except OSError:
        pass
    return f"scorer v{ver} code {commit}"


def log_tail(path: str, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4000))
            lines = f.read().decode("utf-8", "replace").strip().splitlines()
        return " | ".join(x.strip() for x in lines[-n:])[:400]
    except OSError:
        return ""


def headline(summary: Any, set_name: str) -> Optional[str]:
    try:
        e = summary["sets"][set_name]
    except (KeyError, TypeError):
        return None
    m = e.get("metrics") or {}
    key = {"sanity": "sanity_score", "eval-tools": "task_success",
           "eval-injection": "injection_resistance", "eval-longctx": "recall"}.get(set_name)
    r = m.get(key) if key else None
    rate = r.get("rate") if isinstance(r, dict) else None
    head = f"{key} {rate * 100:.1f}%" if isinstance(rate, (int, float)) else "no score"
    extra = []
    if e.get("errors"):
        extra.append(f"{e['errors']} errors")
    if e.get("missing"):
        extra.append(f"{e['missing']} missing")
    if e.get("context_exceeded"):
        extra.append(f"{e['context_exceeded']} over ctx")
    if set_name == "eval-longctx" and isinstance(m.get("recall_by_depth"), dict):
        by = []
        for b, v in m["recall_by_depth"].items():
            if isinstance(v, dict) and isinstance(v.get("rate"), (int, float)):
                by.append(f"{b} {v['rate'] * 100:.0f}")
        if by:
            extra.append("by depth " + " ".join(by))
    return head + (f" ({'; '.join(extra)})" if extra else "")


def feas_result(rep: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(rep, dict):
        return None
    rs = rep.get("results")
    if not isinstance(rs, list) or not rs or not isinstance(rs[0], dict):
        return None
    return rs[0]


def slim(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in ("fits", "peak_gb", "tokens_per_s", "tps_source", "error",
                                  "share_untruncated", "wall_clock_h")}


# ---------------------------------------------------------------------------
# dry run (plan + checks; starts nothing, writes nothing)
# ---------------------------------------------------------------------------


def estimate_set_s(path: str, set_name: str, speed: Dict[str, float], ctx: int) -> Tuple[float, int, int]:
    """(seconds, cases, prompt tokens) from the case file's text size."""
    total_s, n, toks = 0.0, 0, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                c = json.loads(line)
                chars = len(json.dumps(c.get("messages"))) + len(json.dumps(c.get("tools") or []))
            except ValueError:
                c, chars = {}, len(line)
            t = BUCKET_TOKENS.get(c.get("depth_bucket"), int(chars / CHARS_PER_TOKEN))
            toks += t
            if t >= ctx:  # rejected by the server at once
                total_s += 2.0
                continue
            pp = speed["pp"] / (1.0 + (t / 2.0) / speed["d_eq"])
            total_s += t / pp + OUT_TOKENS.get(set_name, 100) / speed["tg"] + 0.3
    return total_s, n, toks


def dry_run(cfg: Cfg, stop_at: float) -> int:
    t = now()
    ok_all = True
    lines: List[str] = []

    def chk(ok: Optional[bool], what: str, detail: str = "", required: bool = True) -> None:
        nonlocal ok_all
        tag = "ok" if ok else ("--" if ok is None else ("MISSING" if required else "absent"))
        if required and ok is False:
            ok_all = False
        lines.append(f"  [{tag:>7}] {what}" + (f": {detail}" if detail else ""))

    # A step's own inputs are required only while it still has work to do: execute skips a
    # step whose inputs are missing and never needs them again once it is done (or parked).
    runner = Runner(cfg, stop_at)
    states = {s["name"]: runner.load_state(s) for s in runner.steps()}

    def needed(step_name: str) -> bool:
        st = states.get(step_name)  # absent: not in the queue (DJR_ONLY, or a step that is off)
        return st is not None and st["status"] != "done" and st.get("failures", 0) < STEP_FAILURES

    def why_not(step_name: str) -> str:
        st = states.get(step_name)
        if st is None:
            return f" (not needed: {step_name} is not in the queue)"
        return "" if needed(step_name) else (
            f" (not needed: {step_name} is {st['status']}"
            + (f", failed {st['failures']} times" if st.get("failures", 0) >= STEP_FAILURES else "") + ")")

    print("tower job runner: DRY RUN (nothing is started or written)")
    ws, we = cfg.window
    h = datetime.fromtimestamp(t)
    in_window = cfg.anytime or continuous(cfg, t) or (parse_hhmm(ws) <= (h.hour, h.minute) < parse_hhmm(we))
    if continuous(cfg, t):
        ws, we = "continuous", f"until {iso(cfg.window_until)}"
    if cfg.only:
        print(f"queue limited to DJR_ONLY: {', '.join(cfg.only)}")
    print(f"now {iso(t)}; stop-at {iso(stop_at)}; window {ws}-{we}"
          + ("" if in_window else " (--execute would refuse now: outside the window)"))
    print("checks:")
    ls = cfg.llama_server
    chk(os.path.isfile(ls) and os.access(ls, os.X_OK), "llama-server binary", ls)
    rev = ""
    head = Path(cfg.llama_src) / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            ref = (Path(cfg.llama_src) / ".git" / ref.split(" ", 1)[1]).read_text().strip()
        rev = ref[:9]
    except OSError:
        pass
    libs = [lib for lib in ("libllama.so", "libggml-cuda.so", "libmtmd.so")
            if os.path.exists(os.path.join(os.path.dirname(ls), lib))]
    chk(bool(rev) or None, "llama.cpp source revision", f"{rev or 'unknown'}; libs beside the binary: "
        f"{', '.join(libs) or 'none found'}", required=False)
    argcpp = Path(cfg.llama_src) / "common" / "arg.cpp"
    flags = ["-c", "-np", "-fa", "-ctk", "-ctv", "--jinja", "-ngl", "-fit", "-b", "-ub", "--host", "--port"]
    if argcpp.is_file():
        src = argcpp.read_text(errors="replace")
        miss = [fl for fl in flags if f'"{fl}"' not in src]
        fa_on = "is_truthy(value)" in src and '"-fa"' in src
        chk(not miss and fa_on, "server flags known to this llama.cpp", "all present" if not miss
            else "unknown: " + " ".join(miss))
    else:
        chk(None, "server flags", f"{argcpp} not found; not checked", required=False)
    chk(os.path.isfile(cfg.score_py), "scorer python", cfg.score_py)
    chk(os.path.isfile(cfg.score_script), "score.py", cfg.score_script)
    commit = Path(cfg.code) / "COMMIT"
    staged = commit.read_text().strip() if commit.is_file() else ""
    if cfg.expect_commit:
        same = bool(staged) and (staged.startswith(cfg.expect_commit) or cfg.expect_commit.startswith(staged))
        chk(same, "staged code commit", f"{staged[:12] or 'no COMMIT file'} (expected "
            f"{cfg.expect_commit[:12]}{'' if same else ': re-stage the code'}); {scorer_tag(cfg)}",
            required=False)
    else:
        chk(bool(staged) or None, "staged code commit", f"{staged[:12] or 'no COMMIT file'}; {scorer_tag(cfg)}",
            required=False)
    if os.path.isfile(cfg.score_py) and os.path.isfile(cfg.score_script):
        try:
            env = dict(os.environ, PYTHONPATH=os.path.join(cfg.code, "src"))
            cp = subprocess.run([cfg.score_py, cfg.score_script, "run", "--help"], capture_output=True,
                                text=True, timeout=120, env=env)
            chk(cp.returncode == 0, "score.py imports (run --help)",
                "" if cp.returncode == 0 else (cp.stderr.strip().splitlines() or ["?"])[-1][:200])
        except (OSError, subprocess.SubprocessError) as e:
            chk(False, "score.py imports", str(e))
    sums = {}
    sums_file = Path(cfg.evals) / "SHA256SUMS"
    if sums_file.is_file():
        for line in sums_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                sums[parts[1].lstrip("*")] = parts[0]
    case_counts = {}
    for name, fname, _ in SETS:
        path = cfg.case_file(fname)
        if not os.path.isfile(path):
            chk(False, f"eval set {name}", f"{path} missing")
            continue
        hsh = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                hsh.update(block)
        want = sums.get(fname)
        case_counts[name] = count_lines(path)
        chk(want == hsh.hexdigest() if want else None, f"eval set {name}",
            f"{case_counts[name]} cases, sha256 {hsh.hexdigest()[:12]}"
            + ("" if want else " (no SHA256SUMS entry)"), required=bool(want))
    m9, why9 = find_9b(cfg)
    facts9 = {}
    if m9:
        try:
            facts9 = gguf_facts(m9)
        except (OSError, ValueError) as e:
            why9 = f"unreadable GGUF: {e}"
    chk(bool(m9) and bool(facts9), "base-9B GGUF", ((f"{m9} ({facts9.get('arch')}, native ctx "
                                                   f"{facts9.get('native_ctx')})") if m9 and facts9 else why9)
        + why_not("base-9b-baseline"), required=needed("base-9b-baseline"))
    m27, why27 = find_27b_gguf(cfg)
    facts27 = {}
    if m27:
        try:
            facts27 = gguf_facts(m27)
        except (OSError, ValueError) as e:
            why27 = f"unreadable GGUF: {e}"
    chk(bool(facts27), "Qwen3.8-27B GGUF (read-only)", (f"{m27} ({facts27.get('arch')}, native ctx "
                                                         f"{facts27.get('native_ctx')}, supported "
                                                         f"{arch_supported(cfg.llama_src, facts27.get('arch'))})")
        if facts27 else why27, required=False)
    mg, whyg = find_glimmer(cfg)
    factsg = {}
    if mg:
        try:
            factsg = gguf_facts(mg)
        except (OSError, ValueError) as e:
            whyg = f"unreadable GGUF: {e}"
    supg = arch_supported(cfg.llama_src, factsg.get("arch")) if factsg else None
    chk(bool(factsg) and supg is not False, "Glimmer GGUF (optional)",
        (f"{mg} ({factsg.get('arch')}, native ctx {factsg.get('native_ctx')}, supported {supg})")
        if factsg else whyg, required=False)
    feas = "qwen38-27b-feasibility"
    feas_req, feas_note = needed(feas), why_not(feas)
    # the smoke needs the training python too (its own inputs are checked below, never required)
    py_req = feas_req or (cfg.smoke and needed("train-smoke")) or (cfg.train and needed("train-full"))
    py_note = "" if py_req else feas_note
    chk(os.path.isfile(cfg.train_py), "training python", cfg.train_py + py_note, required=py_req)
    if os.path.isfile(cfg.train_py):
        try:
            cp = subprocess.run([cfg.train_py, "-I", "-c", "import importlib.util as u; "
                                 "print(' '.join(m for m in ('torch','unsloth','bitsandbytes','peft') "
                                 "if u.find_spec(m) is None))"],
                                capture_output=True, text=True, timeout=60)
            missing = cp.stdout.strip()
            chk(cp.returncode == 0 and not missing, "training packages present (not imported: "
                "importing unsloth touches the GPU)", (missing and "missing: " + missing) + py_note,
                required=py_req)
        except (OSError, subprocess.SubprocessError) as e:
            chk(False, "training packages", str(e) + py_note, required=py_req)
    chk(os.path.isfile(cfg.feas_script), "feasibility_27b.py", cfg.feas_script + feas_note, required=feas_req)
    mb = Path(cfg.model_27b_bnb)
    st_files = list(mb.glob("*.safetensors")) if mb.is_dir() else []
    chk((mb / "config.json").is_file() and bool(st_files), "27B bnb-4bit model dir",
        (f"{mb} ({len(st_files)} safetensors, "
         f"{sum(f.stat().st_size for f in st_files) / 1e9:.1f} GB)" if st_files else f"{mb} missing") + feas_note,
        required=feas_req)
    chk(os.path.isfile(cfg.lengths), "train lengths file (feasibility --data)",
        (f"{cfg.lengths} ({count_lines(cfg.lengths)} rows)" if os.path.isfile(cfg.lengths) else cfg.lengths)
        + feas_note, required=feas_req)
    if cfg.smoke:
        sm_ok, sm_why = runner.smoke_inputs_ok()
        chk(sm_ok or None, "QLoRA smoke inputs (DJR_SMOKE=1; never required: execute skips the step)",
            (f"{cfg.smoke_script}; data {cfg.smoke_data} ({count_lines(cfg.smoke_data)} rows); base "
             f"{cfg.smoke_base} @{runner.smoke_base_rev()[:12]}; tokenizer {cfg.smoke_tok}" if sm_ok else sm_why)
            + why_not("train-smoke"), required=False)
        commit_tr = Path(cfg.smoke_code) / "COMMIT"
        chk(commit_tr.is_file() or None, "smoke code commit",
            commit_tr.read_text().strip()[:12] if commit_tr.is_file() else f"no {commit_tr}", required=False)
        chk(cfg.smoke_md_ok or None, "Max-derived approval flag passed (Q6b)",
            "yes: the trainer also needs the dataset's own recorded approval" if cfg.smoke_md_ok
            else "no (DJR_SMOKE_MAX_DERIVED_OK=0): Max-derived data would be refused", required=False)
    if cfg.train:
        tr_ok, tr_why = runner.train_inputs_ok()
        chk(tr_ok or None, "QLoRA full-run inputs (DJR_TRAIN=1; never required: execute skips the step)",
            (f"{cfg.train_script}; data {cfg.train_data} ({count_lines(cfg.train_data)} rows); base "
             f"{cfg.train_base} @{runner.train_base_rev()[:12]}; tokenizer {cfg.train_tok}" if tr_ok else tr_why)
            + why_not("train-full"), required=False)
        commit_tf = Path(cfg.train_code) / "COMMIT"
        chk(commit_tf.is_file() or None, "full-run code commit",
            commit_tf.read_text().strip()[:12] if commit_tf.is_file() else f"no {commit_tf}", required=False)
        chk(cfg.train_md_ok or None, "full run: Max-derived approval flag passed (Q6b)",
            "yes: the trainer also needs the dataset's own recorded approval" if cfg.train_md_ok
            else "no (DJR_TRAIN_MAX_DERIVED_OK=0): Max-derived data would be refused", required=False)
    if cfg.ab:
        ab_ok, ab_why = runner.ab_inputs_ok()
        chk(ab_ok or None, "teacher A/B inputs (DJR_AB=1; never required: execute skips the step)",
            (f"{cfg.ab_driver}; tasks {cfg.ab_tasks}; IQ3 {cfg.ab_iq3}; Q4 {cfg.ab_q4}; python {cfg.ab_py}"
             if ab_ok else ab_why) + why_not("teacher-ab"), required=False)
        commit_ab = Path(cfg.ab_code) / "COMMIT"
        chk(commit_ab.is_file() or None, "A/B code commit",
            commit_ab.read_text().strip()[:12] if commit_ab.is_file() else f"no {commit_ab}", required=False)
    free = disk_free(cfg.home)
    chk(free >= cfg.disk_floor + cfg.disk_margin, "free disk", f"{free / 2**30:.1f} GiB (floor "
        f"{cfg.disk_floor / 2**30:.0f} + margin {cfg.disk_margin / 2**30:.0f})")
    flag = next((f for f in cfg.no_swaps if os.path.exists(f)), None)
    chk(flag is None, "NO-SWAPS flag absent", flag or ", ".join(cfg.no_swaps))
    prod = unit_state(cfg.prod_unit)
    chk(None, f"production unit {cfg.prod_unit}", f"{prod} now ({'execute refuses while active' if prod == 'active' else 'execute may start'})",
        required=False)
    busy = port_in_use(cfg.host, cfg.port)
    chk(not busy, f"port {cfg.host}:{cfg.port} free", "in use" if busy else "")
    sr = shutil.which("systemd-run")
    chk(bool(sr) or None, "systemd-run (distill-* scopes)", sr or "not found: children would run unscoped",
        required=False)
    chk(os.access(cfg.notify, os.X_OK) or None, "notify command", cfg.notify, required=False)
    if cfg.gates:
        # --execute refuses without these; the window start creates them just before the launch
        # (rehearsal at 01:00, dead-man armed after this dry run), so they are not required here
        r_ok, r_why = rehearsal_state(cfg)
        chk(r_ok or None, "rehearsal PASS (execute gate)", r_why, required=False)
        d_ok, d_why = deadman_state(cfg, t)
        chk(d_ok or None, "dead-man timer armed (execute gate)", d_why, required=False)
    else:
        chk(None, "rehearsal + dead-man gates", "off (DJR_GATES / DJR_ANYTIME)", required=False)
    print("\n".join(lines))

    # the plan
    print("\nplan (in order; each step resumes from its own directory):")
    est_total = 0.0
    steps = runner.steps()
    for i, step in enumerate(steps):
        st = runner.load_state(step)
        sdir = runner.step_dir(step)
        status = st["status"]
        print(f" {i + 1}. {step['name']}  -> {sdir}  [state: {status}"
              + (f", failures {st['failures']}" if st["failures"] else "") + "]")
        if step["kind"] == "train":
            ok, why = runner.smoke_inputs_ok()
            print("    " + ("inputs ok" if ok else f"skip: {why}"))
            print(f"    time box: the earlier of {cfg.smoke_max_s / 60:.0f} min from its start and the step "
                  f"deadline; not started with less than {cfg.smoke_min_s / 60:.0f} min")
            print("    trainer (HF_HUB_OFFLINE=1): " + " ".join(runner.smoke_cmd(None)))
            if st.get("smoke"):
                print(f"    result so far: {st['smoke']}")
            print("    then: the smoke adapter's eval (merge, quantize, score) is a later step, not this one")
            est_total += 60 * 60
            continue
        if step["kind"] == "train-full":
            ok, why = runner.train_inputs_ok()
            print("    " + ("inputs ok" if ok else f"skip: {why}"))
            print(f"    stop-at: the earlier of DJR_TRAIN_STOP_AT ({iso(cfg.train_stop_at) or 'unset'}) and the "
                  f"step deadline; not started with less than {cfg.train_min_s / 60:.0f} min or under "
                  f"{cfg.train_min_free:g} + 1 GiB free; scope TimeoutStopSec {cfg.train_stop_timeout}s")
            print("    trainer (HF_HUB_OFFLINE=1): " + " ".join(runner.train_cmd(None)))
            if st.get("train"):
                print(f"    result so far: {st['train']}")
            est_total += max(0.0, runner.train_until(stop_at) - t)
            continue
        if step["kind"] == "ab":
            ok, why = runner.ab_inputs_ok()
            print("    " + ("inputs ok" if ok else f"skip: {why}"))
            print(f"    time box: the earlier of {cfg.ab_max_s / 60:.0f} min from its start and stop-at minus "
                  f"the later must-run steps' reserve; not started with less than {cfg.ab_min_s / 60:.0f} min")
            print("    driver: " + " ".join(runner.ab_cmd(stop_at)).replace(f"--until {iso(stop_at)}",
                                                                          "--until <time box end>"))
            if st.get("decision"):
                print(f"    decision so far: {st['decision']}")
            if ok and needed(step["name"]):
                try:
                    env = dict(runner.ab_env(), PYTHONDONTWRITEBYTECODE="1")
                    cp = subprocess.run(runner.ab_cmd(None), capture_output=True, text=True, timeout=180, env=env)
                    print(f"    driver dry run (exit {cp.returncode}):")
                    for ln in (cp.stdout + cp.stderr).strip().splitlines()[:40]:
                        print("      " + ln)
                except (OSError, subprocess.SubprocessError) as e:
                    print(f"    driver dry run failed: {e}")
            est_total += 90 * 60
            continue
        if step["kind"] == "baseline":
            model, why = step["find"](cfg)
            if not model:
                print(f"    skip: model {why}")
                continue
            facts = facts9 if model == m9 else (facts27 if model == m27 else factsg)
            if step["fixed_ctx"]:
                ctxs = [step["fixed_ctx"]]
            else:
                native = facts.get("native_ctx") or 0
                ctxs = [native] + [c for c in BUCKET_CTX if c < native]
                if st.get("ctx_chosen"):
                    ctxs = [st["ctx_chosen"]] + [c for c in ctxs if c < st["ctx_chosen"]]
            kvb = facts.get("kv_bytes_per_token_q8")
            size_gib = os.path.getsize(model) / 2**30
            print(f"    model {model} ({size_gib:.1f} GiB, {facts.get('arch')})")
            print(f"    contexts to try: {', '.join(map(str, ctxs))}"
                  + (f"; q8_0 KV ~{kvb / 1024:.1f} KiB/token -> "
                     + ", ".join(f"{c}: {kvb * c / 2**30:.1f} GiB" for c in ctxs) if kvb else ""))
            print("    server: " + " ".join(runner.server_cmd(model, ctxs[0], step["ub"])))
            speed = SPEED["9b"] if step["label"] == "base-9b" else SPEED["27b"]
            # the context it will most likely get: a 27B-class Q4 does not hold 256K of q8_0 KV in 24 GB
            likely = ctxs[0] if step["fixed_ctx"] or 131072 not in ctxs else 131072
            print(f"    likely context for the estimate: {likely}")
            step_s = 60.0
            for name, fname, timeout in SETS:
                path = cfg.case_file(fname)
                if not os.path.isfile(path):
                    continue
                secs, n, toks = estimate_set_s(path, name, speed, likely)
                done = (st.get("sets") or {}).get(name, {}).get("status") in ("done", "done-with-errors")
                step_s += 0 if done else secs
                print(f"    score.py run {name:<15} {n:>4} cases, ~{toks / 1e6:5.2f}M prompt tokens, "
                      f"timeout {timeout}s/req, est {secs / 60:5.0f} min" + (" (done)" if done else ""))
            print(f"    estimate {step_s / 60:.0f} min at ~{speed['pp']:.0f} tok/s prefill (short depth), "
                  f"~{speed['tg']:.0f} tok/s decode (rough)")
            est_total += step_s
        else:
            ok, why = runner.feas_inputs_ok()
            print("    " + ("inputs ok" if ok else f"skip: {why}"))
            for seq in cfg.feas_seqs:
                res = feas_result(read_json(os.path.join(sdir, f"len-{seq}.json")))
                tag = "measured" if res and res.get("fits") is not None else "to probe"
                print(f"    {cfg.train_py} {cfg.feas_script} --data {cfg.lengths} --base-model "
                      f"{cfg.model_27b_bnb} --seqs {seq} --execute --stop-at <step deadline> "
                      f"--out {sdir}/len-{seq}.json  [{tag}]")
            print(f"    then --measured merge -> {sdir}/feasibility-27b.json; reserved "
                  f"{cfg.feas_reserve / 60:.0f} min before stop-at; estimate 25-45 min (4 loads of the "
                  f"27B + 3 steps each), 30 min worst case per length")
            est_total += 40 * 60
    span = stop_at - (t if in_window else max(t, stop_at - 5.25 * 3600))
    print(f"\nrough total if nothing is done yet: {est_total / 3600:.1f} h of GPU work; the window gives "
          f"~{span / 3600:.1f} h before stop-at (from a ~01:20 start).")
    print(f"progress file: {os.path.join(cfg.logs, 'progress.json')}; logs: {cfg.logs}/<run id>/; "
          f"pid files: {cfg.run}/")
    print("result: " + ("all required inputs present" if ok_all else "MISSING inputs: see above"))
    return 0 if ok_all else 1


# ---------------------------------------------------------------------------
# summary for the window-end notify
# ---------------------------------------------------------------------------


def summary_lines(progress: Any) -> List[str]:
    if not isinstance(progress, dict):
        return ["4090 runner: no progress file"]
    st = progress.get("state")
    why = progress.get("stop_reason")
    first = (f"4090 runner {st}" + (f" ({why})" if why else "")
             + f", {hm_iso(progress.get('started_at'))}-{hm_iso(progress.get('ended_at') or progress.get('updated_at'))}")
    out = [first]
    for s in progress.get("steps", []):
        name = s.get("name", "?")
        short = {"base-9b-baseline": "9B base", "qwen38-27b-feasibility": "27B feas",
                 "qwen38-27b-baseline": "27B bake", "glimmer-30b-baseline": "Glimmer",
                 "teacher-ab": "A/B", "train-smoke": "smoke", "train-full": "9B train"}.get(name, name)
        status = s.get("status")
        if name == "train-smoke":
            sm = (s.get("smoke") or {}).get("summary")
            out.append(f"{short} {status}: " + (sm or str(s.get("detail", "")))[:110])
        elif name == "train-full":
            tr = (s.get("train") or {}).get("summary")
            out.append(f"{short} {status}: " + (tr or str(s.get("detail", "")))[:110])
        elif name == "teacher-ab":
            dec = (s.get("decision") or {}).get("verdict")
            out.append(f"{short} {status}" + (f": {dec}" if dec else "")
                       + ("" if dec else f": {str(s.get('detail', ''))[:90]}"))
        elif s.get("sets"):
            parts = []
            for n, v in s["sets"].items():
                ab = {"sanity": "san", "eval-tools": "tools", "eval-injection": "inj",
                      "eval-longctx": "long"}.get(n, n)
                score = ""
                summ = v.get("summary") or ""
                m = re.match(r"\S+ ([\d.]+%)", summ)
                if m:
                    score = " " + m.group(1)
                parts.append(f"{ab} {v.get('done', 0)}/{v.get('total', '?')}{score}")
            ctx = f" c{s['ctx'] // 1024}k" if s.get("ctx") else ""
            out.append(f"{short} {status}{ctx}: " + ", ".join(parts))
        elif s.get("lengths"):
            parts = []
            for seq, v in s["lengths"].items():
                k = f"{int(seq) // 1024}k"
                if v.get("fits") is True:
                    parts.append(f"{k} fits {v.get('peak_gb')}G {v.get('tokens_per_s')}t/s")
                elif v.get("fits") is False:
                    parts.append(f"{k} OOM")
                else:
                    parts.append(f"{k} ?")
            dec = (s.get("decision") or {}).get("verdict")
            out.append(f"{short} {status}: " + ", ".join(parts) + (f" -> {dec}" if dec else ""))
        else:
            out.append(f"{short} {status}: {str(s.get('detail', ''))[:90]}")
    srv = progress.get("server") or {}
    if srv and srv.get("state") not in ("stopped", None):
        out.append(f"WARNING runner server pid {srv.get('pid')} state {srv.get('state')}")
    if len(out) > 5:
        out = out[:4] + ["; ".join(out[4:])[:200]]
    return out


def hm_iso(s: Optional[str]) -> str:
    if not s:
        return "-"
    try:
        return datetime.fromisoformat(s).strftime("%H:%M")
    except ValueError:
        return "?"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="job_runner", description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="run the queue (default: dry run)")
    mode.add_argument("--summary", action="store_true", help="print <= 5 lines from progress.json")
    mode.add_argument("--status", action="store_true", help="print progress.json")
    ap.add_argument("--stop-at", help="HH:MM local, epoch seconds, or ISO with offset "
                                      "(default DJR_STOP_AT, 06:35)")
    a = ap.parse_args(argv)
    cfg = Cfg(dict(os.environ))
    if a.summary or a.status:
        prog = read_json(os.path.join(cfg.logs, "progress.json"))
        if a.status:
            print(json.dumps(prog, indent=1))
        else:
            print("\n".join(summary_lines(prog)))
        return 0
    t = now()
    try:
        stop_at = resolve_stop_at(a.stop_at or cfg.stop_at_raw, t)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not a.execute:
        return dry_run(cfg, stop_at)

    # --- execute: lock, refusals, then the queue -----------------------------------
    no_core_dumps()
    Path(cfg.run).mkdir(parents=True, exist_ok=True)
    lock_f = open(os.path.join(cfg.run, "job_runner.lock"), "a+")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another job runner holds the lock; exiting", file=sys.stderr)
        return 2
    r = Runner(cfg, stop_at)
    Path(r.run_dir).mkdir(parents=True, exist_ok=True)
    r.log_f = open(os.path.join(r.run_dir, "runner.log"), "a")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, r.on_signal)
    pid_path = os.path.join(cfg.run, "job_runner.pid")
    with open(pid_path, "w") as f:
        f.write(f"{os.getpid()}\n")
    r.log(f"runner {r.run_id} pid {os.getpid()} stop-at {iso(stop_at)} unit {r.progress['unit']}")
    rc = 2
    try:
        refusal = refuse_reason(cfg, t, stop_at)
        if refusal:
            r.progress["state"] = "refused"
            r.progress["stop_reason"] = refusal
            r.progress["ended_at"] = iso(now())
            r.save()
            r.log(f"REFUSED: {refusal}")
            r.notify([f"4090 runner refused to start: {refusal}"[:300]])
            return 2
        r.cleanup_orphans()
        if port_in_use(cfg.host, cfg.port):
            r.progress["state"] = "refused"
            r.progress["stop_reason"] = f"{cfg.host}:{cfg.port} already in use"
            r.save()
            r.log("REFUSED: port in use")
            return 2
        ok, info = r.wait_gpu_free(cfg.gpu_release_s)
        if not ok:
            r.progress["state"] = "refused"
            r.progress["stop_reason"] = info
            r.save()
            r.log(f"REFUSED: {info}")
            r.notify([f"4090 runner refused to start: {info}"[:300]])
            return 2
        r.probe_scopes()
        rc = r.execute()
    except Exception as exc:  # noqa: BLE001 - always stop children and record
        r.note(f"runner crashed: {type(exc).__name__}: {exc}")
        r.progress["state"] = "crashed"
        rc = 1
    finally:
        rc = r.shutdown(rc) if r.progress["state"] not in ("refused",) else rc
        r.log(f"runner exit {rc} ({r.progress['state']}, {r.progress.get('stop_reason')})")
        try:
            if open(pid_path).read().strip() == str(os.getpid()):
                os.unlink(pid_path)
        except OSError:
            pass
    return rc


def no_core_dumps() -> None:
    """RLIMIT_CORE 0 (soft and hard) for the runner and everything it starts: a crashing
    llama-server or probe must never write a multi-GB core into its cwd on the home disk
    (apport writes one there for a non-package binary when the soft limit allows it)."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass


def continuous(cfg: Cfg, t: float) -> bool:
    """A continuous window (DJR_WINDOW_UNTIL) is open at *t*."""
    return cfg.window_until is not None and t < cfg.window_until


def refuse_reason(cfg: Cfg, t: float, stop_at: float) -> Optional[str]:
    h = datetime.fromtimestamp(t)
    ws, we = cfg.window
    if cfg.only:
        unknown = [n for n in cfg.only if n not in KNOWN_STEPS]
        if unknown:
            return f"DJR_ONLY names unknown steps: {', '.join(unknown)}"
    if cfg.window_until is not None and continuous(cfg, t):
        if stop_at > cfg.window_until:
            return f"stop-at {iso(stop_at)} is after the continuous window's end {iso(cfg.window_until)}"
    elif not cfg.anytime:
        if not (parse_hhmm(ws) <= (h.hour, h.minute) < parse_hhmm(we)):
            return f"outside the window {ws}-{we} (now {h:%H:%M})"
        end = h.replace(hour=parse_hhmm(we)[0], minute=parse_hhmm(we)[1], second=0, microsecond=0)
        if stop_at > end.timestamp():
            return f"stop-at {iso(stop_at)} is after the window end {we}"
    if stop_at - t < cfg.min_run_s:
        return f"stop-at {iso(stop_at)} is less than {cfg.min_run_s / 60:.0f} min away"
    if stop_at - t > 7 * 3600 and not continuous(cfg, t):
        return f"stop-at {iso(stop_at)} is more than 7 h away"
    for p in cfg.no_swaps:
        if os.path.exists(p):
            return f"NO-SWAPS flag present: {p}"
    if cfg.gates:
        ok, why = rehearsal_state(cfg)
        if not ok:
            return f"no rehearsal PASS ({why}): the window start launches this runner only after one"
        ok, why = deadman_state(cfg, t)
        if not ok:
            return f"dead-man restore not armed for tonight: {why}"
    state = unit_state(cfg.prod_unit)
    if state in ("active", "activating", "reloading", "unknown"):
        return f"{cfg.prod_unit} is {state}: the flywheel setup is not applied"
    if disk_free(cfg.home) < cfg.disk_floor + cfg.disk_margin:
        return f"free disk {disk_free(cfg.home) / 2**30:.1f} GiB is under the floor"
    for path in (cfg.llama_server, cfg.score_py, cfg.score_script):
        if not os.path.exists(path):
            return f"missing {path}"
    return None


if __name__ == "__main__":
    sys.exit(main())
