#!/usr/bin/env python3
"""Phase 2b-style teacher generation: gym tasks, a teacher, sandboxed tools, verified traces.

    # On the GPU host. The teacher is a llama.cpp server on loopback; no paid API, no API key.
    uv run python scripts/distill/teacher_gen.py run \\
        --tasks ~/distill/phase2b/tasksets/real-v1.jsonl \\
        --out ~/distill/phase2b-local --teacher-url http://127.0.0.1:8093 \\
        --injection-manifest ~/distill/phase3/evalsets/eval-injection.manifest.json \\
        --eval-tools ~/distill/phase3/eval-tools.jsonl \\
        --eval-tools ~/distill/phase3/evalsets/ungated/eval-longctx-tools.jsonl \\
        --pack ~/distill/phase3/request_pack.json --repo ~/Prometheus \\
        --until 06:45 [--concurrency 2] [--probe-guard-url http://127.0.0.1:<port>] \\
        [--judge-url http://127.0.0.1:<other port>] [--temperature 0.7 --top-k 20 --top-p 0.8 --min-p 0]
    # Qwen Max as the teacher, on the PAY-PER-TOKEN key only (QWEN_PAYG_KEY), every call through
    # spend.py's one ledger ($150 cap, STOP + notify at $140); rows are Max-derived (--out must
    # carry the max-derived tag). The local 27B judges when --judge-url is given (preferred):
    uv run python scripts/distill/teacher_gen.py run --teacher max-payg --budget-usd 40 \\
        --tasks ~/distill/phase2b/tasksets/real-v1.jsonl --out ~/distill/phase2b-max-derived \\
        [--judge-url http://127.0.0.1:<local 27B judge>] [--thinking-budget 0] [--max-tokens 4096] ...
    # A smoke run of the built-in task families (rows tagged builtin-smoke, never trained on):
    uv run python scripts/distill/teacher_gen.py run --builtin-smoke --max-tasks 5 --out /tmp/smoke ...
    # The built-in tasks a smoke run would start next (no model calls), as JSONL:
    uv run python scripts/distill/teacher_gen.py list-tasks --n 3 --injection-manifest ... --eval-tools ...
    # Into the Phase 3 builder. Its data policy (--train-sources, default qwen38-27b-gym;
    # --train-models, default qwen3.8-27b*) admits these rows to train/val on their own
    # _meta.verification record. Never pass this model as --teacher-model: that names Max rows.
    python3 build.py build ... --extra-traces '~/distill/phase2b-local/qwen38-27b-gym.traces.jsonl'

Task source (the operator's rule: no model-written task text in training data)
-------------------------------------------------------------------------------
``run`` needs exactly one of:

* ``--tasks <taskset.jsonl>``: a ``taskset_gen.py`` real-pattern taskset (every
  row checked by that file's own ``validate_task``; anything else is refused).
  It is the ONLY way to rows tagged ``source = qwen38-27b-gym``: a row gets that
  tag only when the run is a ``--tasks`` run AND its task was built from a
  validated taskset row (``GenTask.origin == "taskset"``, provenance
  ``real-pattern-v1``). The task flags are execution conditions:
  ``suppress_tools`` -> a chat-mode request with no tools; ``inject_errors`` ->
  the sandbox fails the first tool call of the task's own turn and the run must
  recover (also on a task that is compacted too); ``force_compaction`` ->
  earlier turns before the prompt, then the real compactor with a small window
  (below). ``_meta.task_id`` is the row's ``task_group``,
  ``_meta.partner_task_id`` a recombined row's second group, and the row's
  provenance fields are carried.

  Compaction padding holds no text written here or by an agent: every user
  turn before the prompt is ANOTHER real-pattern prompt of the same taskset,
  sent verbatim and answered live by the same teacher in the same run (its own
  row's mode: chat turns only before a chat-mode prompt), until the history
  holds ``PAD_HISTORY_FACTOR`` x ``--compaction-history-tokens`` (at least
  ``PAD_MIN_TURNS`` turns, at most ``PAD_MAX_TURNS``). An agent task's sandbox
  holds whole git-tracked docs of ``--repo`` (those a pad prompt names, plus a
  few more), so a read during padding returns the repo's own text; a repo
  without docs is refused for an agent task (exit 2), never padded with the
  synthetic runbooks a smoke run falls back to. The pad prompts come from a
  reserved pool (``PAD_POOL_SHARE`` of the taskset's task groups, never
  flagged ones, chosen by a hash of the taskset's name, so every run and every
  teacher reserves the same ones; ``pad_pool_split``): pool rows are never run
  as tasks, so no padded row shares a task with a row in another build split.
  A taskset too small to spare a group pads from every other row instead.
  Candidates share no task group with the row or each other and are checked
  against the eval sets. The task's injections are held back during padding;
  a pad turn that does not end in a reply, or whose calls were refused,
  repaired, dropped or cut off, rejects the task (``pad_*``). Pad turns are
  history: never rows, never judged on their own (the judge sees them as the
  earlier conversation). Rows carry the descriptor ``_meta.compaction_pad``
  (``version``, the pad rows' ``task_ids`` / ``task_groups`` /
  ``prompt_hashes`` / ``modes``, the teacher and tool calls, reads, the
  sandbox's ``repo_docs``, ``pool``: reserved | shared).
* ``--builtin-smoke``: the built-in families below (or ``--tasks-file`` rows
  from ``list-tasks``). Their text was written by hand in this file, so every
  row is tagged ``source = builtin-smoke`` (``--name`` defaults to it) and
  ``build.py`` never trains on it. A list-tasks row cannot claim another origin.
  The local teacher only: ``--teacher max-payg`` refuses it (exit 2).

What the built-in families generate (``--builtin-smoke``)
--------------------------------------------------------
Five task kinds, scheduled by deficit against ``--mix`` (rows kept per kind vs
the target share; ties go to the earlier kind, and a kind whose pass rate is
very low after 10 attempts is scheduled less), in this priority order:

* ``chat-mode``: no tools (``run_loop(mode="chat")``, which sets
  ``suppress_tools``): questions, explanations, plans and FAQs over sections of
  the repo's own docs (``--repo``: git-tracked ``docs/``, ``wiki/`` and
  ``README.md``; the distillation sprint docs are skipped), and over synthetic
  wiki-style service notes with answers a predicate can check.
* ``compacted-history``: a long seeded session forced through the REAL
  ``ContextCompactor`` with a small window (``--compaction-history-tokens`` of
  history room above the system prompt, tools and output), then continued. The
  teacher's summary call goes through its own provider and is never a row; the
  rows carry the render view the teacher actually saw, summary included.
* ``error-recovery``: the sandbox injects a failure (a timeout, an HTTP 503, a
  locked store, a moved file) on a chosen call; the teacher has to retry, work
  around it or report it.
* ``agentic``: non-coding agent work: memory, wiki, cron scheduling, background
  tasks, service health and resources (simulated host), SENTINEL status,
  research over a mocked web, notes and TODO files, a mocked message send.
* ``coding``: small edits and lookups in a generated Python project. Coding rows
  can never exceed ``--coding-cap`` (35%) of the kept rows: a verified coding
  task that would cross it is held (``.held.jsonl``) and released when other
  rows make room.

Every task is checked against the eval sets before it runs: the eval-injection
manifest (``build_eval_sets.InjectionManifest``: shingles, markers, whole
instructions) over every string in the task, and the eval-tools cases
(``--eval-tools``: task hashes of their user prompts, and 8-token shingles of
them). An overlapping task is skipped and recorded; every written row is
checked against the manifest again.

The run
-------
Each task runs through the real ``run_loop`` with the real ``ModelAdapter``
(tier from the model registry), a ``SecurityGate`` scoped to the task's
sandbox, and a ``LlamaCppProvider`` pointed at ``--teacher-url`` (or
``--probe-guard-url``, which then carries every teacher, summarizer and
self-judge request; a ``--judge-url`` naming another server is called
directly, the guard fronts the teacher only; the teacher's or the guard's
port under any spelling of its host, ``localhost`` or ``[::1]`` for
``127.0.0.1`` say, is the teacher, never a way around the guard). No
model request goes through a proxy: HTTP(S)_PROXY / ALL_PROXY are dropped
from the process and NO_PROXY is ``*``. A reply llama-server cut off
mid-stream (an error event, no finish_reason) is a server error, never a
turn. Thinking stays off, as at
runtime. Sampling is the server's own unless ``--temperature`` / ``--top-k`` /
``--top-p`` / ``--min-p`` are given: those go on every teacher and summarizer
request (a GGUF's own metadata can change a server's defaults, so two servers
compared on the same tasks need them; the judge always runs at temperature 0
with ``--judge-seed``). ``--concurrency``
tasks run at once (default 2). Each worker has one sandbox directory under
``--sandbox-root``, wiped before and after every task; the run holds an
exclusive lock on that root, so a second run on the same root is refused
(exit 2) instead of wiping the first one's workspaces. ``--until`` is a hard
end: no task starts within ``--min-task-seconds`` of it, and a task still
running at it is cancelled and not recorded (it reruns next time).
Re-running the same command resumes (see Outputs).

Process state never touches the live Prometheus stores: before anything is
imported, ``PROMETHEUS_CONFIG_DIR``, ``PROMETHEUS_DATA_DIR``,
``PROMETHEUS_LOGS_DIR`` and ``PROMETHEUS_ENV_FILE`` point under
``<out>/work/``, telemetry goes to a database there, and memory, wiki, cron,
tasks, web and messages are in-memory mocks per task.

The sandbox
-----------
Only the tools in ``SANDBOX_TOOLS`` exist (an allow-list; the real
registry's schemas, so ``build.py`` sees tools the runtime offers), each
wrapped so that every call is logged, may be failed by an injection, and has
its path arguments confined:

* ``read_file`` / ``write_file`` / ``edit_file`` / ``grep`` / ``glob`` /
  ``todo_write`` run for real, but any path outside the sandbox is refused.
* ``bash`` never runs ``bash -l``. A command is tokenized like a POSIX shell
  (quoting kept, so an unquoted brace, a wildcard that can match ``.``/``..``
  or a leading tilde is refused) and refused unless every program is on
  ``ALLOWED_PROGRAMS`` (text and file utilities: no interpreters, network
  clients, package managers, ``ssh``, ``git``, ``xargs``), every path stays in
  the sandbox, and there is no command substitution, variable, background job,
  subshell or heredoc. Options are read by each program's grammar
  (``_GRAMMARS``): a value attached to a short option (``-o/x``, ``-t/x``) or
  given to a long one is confined like any path; ``awk`` takes only ``-F`` and
  ``-v`` and its program has no ``system``, ``getline``, pipes, redirects,
  ``ARGV`` or ``@`` directives; every ``sed`` script form is print / delete /
  substitute only, never in place; ``sort --compress-program``, file-list
  options (``--files0-from``, ``file -f``, ``sha256sum -c``), ``jq`` modules
  and ``find -exec``/``-delete`` are refused. Allowed commands run under
  ``/bin/dash -c`` (no brace expansion; ``sh`` with ``set +B`` where dash is
  missing) in the sandbox with a scrubbed environment, CPU/file-size limits, a
  timeout and, with ``--jail auto`` (default) or ``require``, an OS jail:
  ``bwrap`` on Linux (only the workspace writable, no network, no home or
  runtime dirs visible) or ``sandbox-exec`` on macOS. The report and every
  row's verification record name the jail used (``none`` means the command
  checks alone). Host commands (``systemctl``, ``journalctl``,
  ``nvidia-smi``, ``df``, ``free``, ``uptime``, ``ps``, ``curl`` to loopback)
  are SIMULATED from the task's fixture and never executed; their output goes
  through the command's own output redirects and the rest of its pipeline as
  written; changing a service is refused.
* ``memory``, ``wiki_query``, ``wiki_compile``, ``cron_*``, ``task_*``,
  ``web_search``, ``web_fetch``, ``message``, ``sentinel_status``,
  ``lcm_grep`` are mocks over the task's fixtures (``lcm_grep`` searches the
  run's own conversation). Nothing is scheduled, sent or fetched.

A refused call fails the task by default (its context would teach a sandbox
artifact).

Verification (all must hold; only then are rows written)
------------------------------------------------------
* tool success: no tool error except injected ones, no sandbox refusal, no
  adapter repair, no dropped malformed call, no orchestrator feedback, every
  reply from the preflight-checked teacher;
* the task's deterministic gym predicates (``prometheus.gym.scoring.score``)
  and its answer checks (whole-word / whole-number regexes, with the negations
  of a state refused: "not running" is not "running");
* per kind: chat-mode made no tool call; an injection (error-recovery, or any
  task with ``inject_errors``) fired and (when the task expects it) a call
  succeeded after it; compacted-history's compactor fired and a teacher
  request carried the summary;
* a judge with a strict rubric, at temperature 0 and its own ``--judge-seed``,
  answering JSON: the teacher itself (``self-judge``) unless ``--judge-url``
  names another local endpoint (``local-judge``). The record names it the way
  build.py reads it: ``provider: local``, ``judge_key: none`` (the local
  judge's ``endpoint`` too under ``--teacher max-payg``), or for Max judging
  itself ``provider: max-payg``, ``judge_key: payg`` and the ledger's
  ``call_ids``.

Endpoints must be loopback (``--allow-lan-endpoints`` also admits private and
CGNAT addresses). Known hosted-API hosts are refused outright, and no local
request ever carries an Authorization header.

The Max teacher (``--teacher max-payg``)
--------------------------------------
Qwen Max (``PAYG_MODEL``) on the PAY-PER-TOKEN key and nothing else: every
teacher, summarizer and self-judge request is ``spend.PaygClient.chat`` (tool
``teacher_gen``, phase ``2b-teacher``), reserved at its worst case in the one
ledger (``~/distill/spend-ledger.csv``) and settled from the API's usage. The
key is ``QWEN_PAYG_KEY`` only (``spend.resolve_payg``: never ``QWEN_API_KEY``,
a ``sk-sp-`` key or a Token Plan / Coding Plan host; its prefix must match its
host), read before the state directories are isolated. Before the first call
the run checks the key, the canonical ledger, the STOP file and the notify
command, and refuses (exit 5) when any is missing: until the key is on the
host the mode refuses cleanly and sends nothing. ``--budget-usd`` is required:
this phase's share of the global cap, checked by spend.py against the ledger's
total for teacher_gen / 2b-teacher (so a resumed run cannot spend it twice).
Each request carries only ``spend.PAYLOAD_KEYS``: the loop's messages and
tools, an explicit ``max_tokens`` (at most ``--max-tokens``, itself at most
``PAYG_MAX_TOKENS``), ``enable_thinking`` (false unless ``--thinking-budget N``,
which also sends ``thinking_budget``), no stream; ``--min-p`` is refused. A
reply's reasoning is never a target. One attempt per call (no retry inside
a task); a failed attempt is charged as spend.py says. The run stops cleanly
when the ledger refuses (the STOP file, the $150 cap, the $140 alert, the
phase budget; exit 4) or the API refuses the key or account (auth, billing;
exit 5): the task in flight is not recorded and reruns next time, and the rows
written so far stay. Concurrency is at most ``PAYG_MAX_CONCURRENCY``. Under
``--until`` every paid call is over ``PAYG_UNTIL_MARGIN_S`` before it (its
timeout is cut to fit) and none starts with less than ``PAYG_MIN_CALL_S``
left: the run stops (stop ``until``) and the task reruns, so no call is still
in flight when gen_runner.py stops its children. A SIGTERM or SIGHUP (a stop
by gen_runner.py or systemd) starts no new paid call, gives the calls in
flight up to ``PAYG_SIGNAL_WAIT_S`` to settle from their usage, settles the
rest as interrupted (charged their reservation) and exits 128 + the signal:
a stop never leaves a reservation open in the ledger. The task in flight is
not recorded (it reruns next time).

Everything else is the local path's: the real-pattern taskset only
(``--builtin-smoke`` and ``--tasks-file`` are refused, exit 2: the paid key
is for real-pattern tasks, never for hand-written smoke tasks), the
sandbox, injections, compaction padding (answered by Max: teacher output),
the verification, the dedupe and the eval guard. The loop runs with the local
route's adapter for the model name (tier light), exactly as for the 27B. Every
served model must match ``PAYG_TEACHER_ID``. The judge: the local 27B when
``--judge-url`` is given (preferred; through ``--probe-guard-url`` when one
fronts it; its id must match ``--judge-id``, default ``*qwen3*8*27b*``, never
Max), else Max judging itself on the paid key. Rows are tagged
``source = max-payg-gym`` (``max-payg-gym-relaxed`` under a relaxation),
``model = qwen3.8-max``, ``max_derived: true``, ``tags: ["max-derived"]``,
``teacher: max-payg`` and their ledger ``paid_call_id`` / ``cost_usd``.
build.py admits them only under ``--allow-max-derived`` (its Q6b allow), with
the strict record; they are never published (``publish_guard.py``), so
``--out`` must carry the ``max-derived`` tag (exit 2 otherwise) and the report
carries ``max_derived: true`` and this phase's spend. The paid key is never
used for daemon or background work: ``teacher_gen`` is one of spend.py's two
allowed tools.
The preflight refuses a teacher (or judge) whose ``/v1/models`` id does not
match ``--teacher-id`` (``--judge-id``; default ``*qwen3*8*27b*``) or names a
Max model, so a loopback proxy, the probe guard in front of another model or
the Ollama port cannot pass as the local 27B; rows carry that id
(``_meta.teacher_id``).

``--max-natural-errors`` / ``--keep-refused`` relax tool success. A task kept
only because of them is ``relaxed``: its rows go to ``<name>.relaxed.jsonl``
with source ``qwen38-27b-gym-relaxed`` (``builtin-smoke`` in a smoke run;
build.py's strict policy would drop them as train rows) and count toward no
target, mix or cap.

Provider errors: a teacher or judge that is down, slow or failing (connection,
timeout, 5xx) is an ``error`` (retried by the next run; enough in a row stop
it). A request it refuses (context overflow, another 4xx) is a rejection
(``context_overflow``, ``provider_bad_request``, ``judge_bad_request``):
recorded as done, never retried. An overflow the loop recovered from by
compacting is ignored. The compactor swallows its summarizer's errors, so a
compaction task that never compacted because its summary request failed for
a passing reason (one of ``RETRY_KINDS``: a 429, a timeout, a 5xx) is an
``error`` too, never ``compaction_not_exercised``.

``--tasks`` takes only a ``taskset_gen.py`` taskset (Task source, above);
``--builtin-smoke --tasks-file`` takes list-tasks rows.

Outputs (``--out``, file prefix ``--name``)
-----------------------------------------
* ``<name>.traces.jsonl``: verified rows in the golden-trace shape
  ``build.py --extra-traces`` reads: ``messages`` (the exact wire messages the
  teacher saw, system prompt first; tool results carry ``is_error``) ending in
  the teacher's assistant turn, ``tools`` (the offered tools, OpenAI shape;
  ``[]`` for chat-mode), and ``_meta``: ``source`` = ``qwen38-27b-gym``
  (``--tasks``; ``builtin-smoke`` with ``--builtin-smoke``), ``task_origin``,
  ``sampling`` (when given), ``model`` (``--model-name``), ``served_model``, ``category`` (build.py's
  categories: chat-mode, memory/wiki, sentinel/daemon, other, coding),
  ``task_category`` (the kind), ``task_id``, ``task_family``, ``run_id``,
  ``verdict: pass``, ``max_derived`` (false; true for a Max teacher's row),
  ``teacher``, ``compaction_pad`` (a padded row), ``suppress_tools``, ``compacted``,
  ``error_recovery``, and ``verification``: the full record in the shape
  build.py's data policy reads strictly (``verified``, ``tool_success``,
  ``judge`` with ``verdict``/``model``/``pass``, ``sandboxed: true``,
  ``side_effects: false``, ``jail``) plus the counts and predicate results
  behind it. One row per teacher call the loop acted on, whose calls all ran
  (an injected failure counts: the call was right, the world failed). The
  target is the assistant message the loop committed: text-markup calls it
  extracted become structured calls, leftover ``<tool_call>`` text it
  stripped is gone (``_meta.target_rewritten``), and a reply it retried or
  discarded is no row;
  chat-mode rows are the text reply. An agentic task's final prose turn is not
  a row unless ``--emit-final-turns`` (build.py renders a turn with no call as
  chat-mode, without tools). Within a task, rows build.py would dedupe as the
  same prompt + call keep the one with an error in its context.
* ``<name>.held.jsonl``: verified coding rows waiting under the cap.
* ``<name>.relaxed.jsonl``: rows verified only under a relaxation (above).
* ``<name>.tasks.jsonl``: the append-only ledger, one event per task outcome
  (kept / held / released / rejected / skipped) with reasons, the judge record
  and counts. It is what resume reads.
* ``<name>.report.json``: counts and pass rates by kind and category, row
  shares, judge stats, rejection reasons. Rewritten after every task.
* ``<name>.settings.json``: the generation settings; a resume with different
  ones is refused. The ``--pack`` system prompt counts as its template: the
  pack's date sentinel becomes the date each task starts on, so a resume on a
  later day is the same run.

Resume: a torn last line is cut; rows whose run is not in the ledger (a crash
between the two writes) are removed; tasks in the ledger are not run again;
a held task released before a crash ends up in exactly one file.

Exit: 0 done (target, ``--max-tasks``, ``--until`` or no tasks left), 1 a
run that kept nothing, 2 usage or input error, 3 the teacher or judge is
unreachable (preflight), or ``--max-consecutive-errors`` tasks in a row ended
in an error (teacher or judge transport, or a harness fault), 4 the spend
ledger stopped a ``--teacher max-payg`` run (STOP, the cap, ``--budget-usd``),
5 ``--teacher max-payg`` refused before any call (no ``QWEN_PAYG_KEY``, the
Token Plan key or host, the STOP file, no notify command) or the paid API
refused the key or account mid-run, 143 / 129 a ``--teacher max-payg`` run
stopped by SIGTERM / SIGHUP (its calls in flight settled first), 6 an
unexpected exception (its traceback on stderr; never Python's own 1). Errored
tasks are not in the ledger as done, so the next run retries them.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextvars
import copy
import dataclasses
import datetime as dt
import fcntl
import fnmatch
import glob
import hashlib
import importlib.util
import ipaddress
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

log = logging.getLogger("teacher_gen")

SOURCE_TAG = "qwen38-27b-gym"
#: Rows verified only under --max-natural-errors / --keep-refused: written to their own
#: file under this tag, never counted toward targets, the mix or the coding cap.
RELAXED_SOURCE = "qwen38-27b-gym-relaxed"
#: Rows of the built-in task families (``--builtin-smoke``): their task text was written
#: by hand in this file, not taken from real prompts, so they are never a train source
#: (build.py excludes them: never listed, and not the Max model's rows either).
BUILTIN_SOURCE = "builtin-smoke"
#: The provenance ``taskset_gen.py`` stamps on every row (its ``PROVENANCE``).
TASKSET_PROVENANCE = "real-pattern-v1"
#: Where a task came from: a real-pattern taskset (``--tasks``) or the built-in families.
ORIGIN_TASKSET = "taskset"
ORIGIN_BUILTIN = "builtin"
#: Request sampling fields ``--temperature`` / ``--top-k`` / ``--top-p`` / ``--min-p`` set.
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "min_p")
GENERATOR_VERSION = "teacher-gen/4"
#: ``--teacher``: the local llama-server (default) or Qwen Max on the pay-per-token key.
TEACHER_LOCAL = "local"
TEACHER_PAYG = "max-payg"
TEACHERS = (TEACHER_LOCAL, TEACHER_PAYG)
#: --teacher max-payg: the one priced model (spend.PRICES), its row tags, and its ledger
#: tool and phase (spend.ALLOWED_TOOLS; the phase is the --budget-usd sub-budget's key).
PAYG_MODEL = "qwen3.8-max"
PAYG_TEACHER_ID = "qwen3.8-max*"
PAYG_SOURCE = "max-payg-gym"
PAYG_RELAXED_SOURCE = "max-payg-gym-relaxed"
PAYG_TOOL = "teacher_gen"
PAYG_PHASE = "2b-teacher"
PAYG_MAX_CONCURRENCY = 4
#: The most output one paid request may ask for (``--max-tokens`` above it is refused);
#: every request carries an explicit max_tokens at or under ``--max-tokens``.
PAYG_MAX_TOKENS = 8192
#: The most reasoning one paid request may ask for (``--thinking-budget``; 0 = thinking off).
PAYG_MAX_THINKING_BUDGET = 16384
#: Under ``--until``, every paid call must be over this long before it (its timeout is cut to fit):
#: gen_runner.py stops its children at --until, and a call still in flight then would be cut off.
PAYG_UNTIL_MARGIN_S = 15.0
#: ... and a paid call with less time than this left before that point is not started: the run
#: stops (stop ``until``) and the task reruns next time, instead of paying for a call cut short.
PAYG_MIN_CALL_S = 120.0
#: A paid run stopped by one of these signals (gen_runner.py, systemd) starts no new paid call,
#: waits up to ``PAYG_SIGNAL_WAIT_S`` (under gen_runner's 20 s grace before SIGKILL) for its calls
#: in flight to settle from their usage, settles the rest as interrupted, and exits 128 + signal.
PAYG_EXIT_SIGNALS = (signal.SIGTERM, signal.SIGHUP)
PAYG_SIGNAL_WAIT_S = 12.0
#: The judge's output bound (local and paid).
JUDGE_MAX_TOKENS = 600
#: Compaction padding (force_compaction taskset rows): earlier turns made of OTHER
#: real-pattern prompts from the taskset's reserved pad pool, answered by the same
#: teacher in the same run. ``PAD_POOL_SHARE`` of a taskset's rows (whole task
#: groups, chosen by a hash keyed by the taskset's name, so every run and every
#: teacher over one taskset reserves the same pool) are never run as tasks: their
#: prompts appear only as padding, so no padded row shares a task with another row.
PAD_VERSION = "taskset-turns-v2"
PAD_POOL_SHARE = 0.10
PAD_MIN_TURNS = 3       # the real compactor summarises a span of at least 4 messages
PAD_MAX_TURNS = 10      # candidates per task, in seeded order
PAD_HISTORY_FACTOR = 1.5  # pad until the history holds this x --compaction-history-tokens
PAD_DOCS = 4            # whole repo docs (besides those a pad prompt names) in an agent task's sandbox
PAD_DOC_MAX_BYTES = 16_000
DEFAULT_TEACHER_URL = "http://127.0.0.1:8093"
DEFAULT_MODEL_NAME = "qwen3.8-27b"
DEFAULT_TEACHER_ID = "*qwen3*8*27b*"
DEFAULT_JUDGE_SEED = 7
DEFAULT_SEED = "teacher-gen-v1"
DEFAULT_SANDBOX_ROOT = Path(tempfile.gettempdir()) / "prometheus-teacher-gen"
KINDS = ("chat-mode", "compacted-history", "error-recovery", "agentic", "coding")
DOMAINS = ("chat-mode", "memory/wiki", "sentinel/daemon", "other", "coding")
DEFAULT_MIX = {"chat-mode": 0.30, "compacted-history": 0.20, "error-recovery": 0.20,
               "agentic": 0.22, "coding": 0.08}
CODING_CAP = 0.35
SUMMARY_MARKER = "[Compacted summary of turns"
SEED_CALL_PREFIX = "call_seed_"
ORCHESTRATOR = "orchestrator"
_DIR = Path(__file__).resolve().parent
_REPO_DEFAULT = _DIR.parent.parent

SANDBOX_TOOLS = frozenset({
    "bash", "read_file", "write_file", "edit_file", "grep", "glob", "todo_write", "tool_search",
    "memory", "wiki_query", "wiki_compile", "cron_create", "cron_list", "cron_delete",
    "task_create", "task_list", "task_get", "task_stop", "task_output",
    "web_search", "web_fetch", "message", "sentinel_status", "lcm_grep",
})
#: Tools whose execute() is replaced by an in-memory mock (no store, network or scheduler).
MOCKED_TOOLS = frozenset({
    "memory", "wiki_query", "wiki_compile", "cron_create", "cron_list", "cron_delete",
    "task_create", "task_list", "task_get", "task_stop", "task_output",
    "web_search", "web_fetch", "message", "sentinel_status", "lcm_grep",
})
#: Path-valued arguments, confined to the sandbox before the real tool runs.
PATH_ARGS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",), "write_file": ("path",), "edit_file": ("path",), "grep": ("root",),
    "glob": ("root",), "todo_write": ("path",), "bash": ("cwd",),
}
ALLOWED_PROGRAMS = frozenset({
    "cat", "ls", "echo", "printf", "head", "tail", "wc", "grep", "egrep", "fgrep", "sort", "uniq",
    "cut", "tr", "sed", "awk", "find", "pwd", "date", "basename", "dirname", "true", "false",
    "test", "[", "diff", "cmp", "comm", "nl", "rev", "stat", "du", "file", "md5sum", "sha1sum",
    "sha256sum", "touch", "mkdir", "cp", "mv", "rm", "rmdir", "tee", "column", "fold", "paste",
    "join", "seq", "cd", "jq", "realpath", "readlink", "tac", "od", "expand", "fmt",
})
#: Host commands answered from the task's fixture, never executed.
OPS_PROGRAMS = frozenset({"systemctl", "journalctl", "nvidia-smi", "df", "free", "uptime", "ps",
                          "curl"})
_FORBIDDEN_HOST_PARTS = ("dashscope", "aliyuncs", "alibabacloud", "alicloud", "openai.com",
                         "anthropic.com", "openrouter", "googleapis", "together.xyz", "groq.com")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
OUTPUT_LIMIT = 1 << 20  # bytes read per stream of a sandboxed command
_CURRENT_CALL: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "teacher_gen_current_call", default=None)


# ---------------------------------------------------------------------------
# Endpoint policy: local model servers only
# ---------------------------------------------------------------------------


class EndpointError(ValueError):
    """A model endpoint this script must not call."""


def _host_addresses(host: str) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host.strip("[]")))]
    except ValueError:
        pass
    if host == "localhost" or host.endswith(".localhost"):
        return ["127.0.0.1"]
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise EndpointError(f"cannot resolve {host!r}: {exc}") from exc
    return sorted({str(i[4][0]) for i in infos})


def check_endpoint(url: str, *, allow_lan: bool = False) -> str:
    """The URL without a trailing slash, or EndpointError. Loopback only by
    default; *allow_lan* also admits private and CGNAT (tailnet) addresses.
    Known hosted-API hosts and URLs carrying credentials are always refused."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise EndpointError(f"not an http(s) URL: {url!r}")
    if parts.username or parts.password:
        raise EndpointError("credentials in an endpoint URL are refused")
    host = parts.hostname.lower()
    if any(p in host for p in _FORBIDDEN_HOST_PARTS):
        raise EndpointError(f"{host} is a hosted API; this script only calls local model servers")
    for addr in _host_addresses(host):
        ip = ipaddress.ip_address(addr)
        if ip.is_loopback:
            continue
        if allow_lan and not ip.is_global and (ip.is_private or ip in _CGNAT):
            continue
        raise EndpointError(
            f"{host} ({addr}) is not a local endpoint"
            + ("" if allow_lan else " (loopback only; --allow-lan-endpoints admits private addresses)"))
    return url.rstrip("/")


def same_server(a: str | None, b: str | None) -> bool:
    """Whether two endpoint URLs reach the same server: the same port (scheme
    default when none) on hosts that share an address, every loopback spelling
    (127.0.0.1, localhost, [::1], 127.0.0.2) counting as one. Paths are ignored.
    An unresolvable host falls back to comparing the names. Errs towards True:
    a judge URL that might be the teacher is treated as the teacher."""
    if not a or not b:
        return False
    pa, pb = urlsplit(a), urlsplit(b)
    try:
        port_a = pa.port or (443 if pa.scheme == "https" else 80)
        port_b = pb.port or (443 if pb.scheme == "https" else 80)
    except ValueError:
        return a.rstrip("/") == b.rstrip("/")
    if port_a != port_b:
        return False
    ha, hb = (pa.hostname or "").lower(), (pb.hostname or "").lower()
    try:
        addrs_a, addrs_b = _host_addresses(ha), _host_addresses(hb)
    except EndpointError:
        return ha == hb

    def norm(addrs: list[str]) -> set[str]:
        return {"loopback" if ipaddress.ip_address(x).is_loopback else x for x in addrs}

    return ha == hb or bool(norm(addrs_a) & norm(addrs_b))


# ---------------------------------------------------------------------------
# Hashes (build.py's Phase 1 functions, ported unchanged; a test holds them equal)
# ---------------------------------------------------------------------------

_HEX = re.compile(r"\b[0-9a-f]{6,}\b", re.I)
_NUM = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    s = (s or "").lower()
    s = _HEX.sub("h", s)
    s = _NUM.sub("0", s)
    return _WS.sub(" ", s).strip()


def h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def prompt_call_key(row: dict[str, Any]) -> str:
    """build.py's dedupe key for a row (``trace_keys(...).h_prompt_call``)."""
    meta = row.get("_meta") or {}
    msgs = row["messages"]
    call = (msgs[-1].get("tool_calls") or [{}])[0].get("function", {})
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    users = [content_text(m.get("content")) for m in msgs[:-1] if m.get("role") == "user"]
    prompt = users[-1] if users else ""
    norm_call = (meta.get("tool_name") or "") + "|" + norm(json.dumps(args, sort_keys=True))
    return h(norm(prompt) + "||" + norm_call)


def _load_sibling(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class GenTask:
    id: str
    kind: str
    domain: str
    family: str
    prompt: str
    mode: str = "agent"                 # agent | chat
    seed: list[dict[str, Any]] = field(default_factory=list)
    setup_files: dict[str, str] = field(default_factory=dict)
    fixtures: dict[str, Any] = field(default_factory=dict)
    inject: list[dict[str, Any]] = field(default_factory=list)
    score: dict[str, Any] = field(default_factory=dict)
    compaction: bool = False
    expect_recovery: bool = False       # error-recovery: a call must succeed after the failure
    notes: str = ""
    #: answer checks on the final text (case-insensitive regexes): every ``match`` must
    #: hit and no ``forbid`` may (a number or name inside another word does not count)
    answer: dict[str, list[str]] = field(default_factory=dict)
    #: copied into every row's _meta (a taskset's provenance)
    meta: dict[str, Any] = field(default_factory=dict)
    #: ORIGIN_TASKSET only when ``task_from_taskset`` built it from a validated
    #: taskset row; a task read back from a file is always built-in (``from_dict``)
    origin: str = ORIGIN_BUILTIN
    #: compaction padding (a force_compaction taskset row): other real-pattern prompts,
    #: in order, each ``{"id", "task_group", "mode": chat|agent, "prompt", "docs"}``.
    #: ``run_task`` sends them as earlier user turns, each answered by the same teacher,
    #: until the history is long enough to compact (``pad_candidates``)
    pad: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenTask":
        """A list-tasks row: always a built-in task. A row claiming another
        origin is refused (only ``--tasks`` makes taskset tasks)."""
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"task {d.get('id')!r}: unknown keys {sorted(unknown)}")
        if d.get("origin", ORIGIN_BUILTIN) != ORIGIN_BUILTIN:
            raise ValueError(f"task {d.get('id')!r}: a list-tasks row is a built-in task; "
                             "real-pattern tasks come only from a taskset_gen.py file (--tasks)")
        task = cls(**d)
        task.validate()
        return task

    def validate(self) -> None:
        from prometheus.gym.tasks import ALLOWED_SCORE_KEYS, ALLOWED_SEED_KEYS

        if self.kind not in KINDS:
            raise ValueError(f"task {self.id!r}: kind {self.kind!r} not in {KINDS}")
        if self.domain not in DOMAINS:
            raise ValueError(f"task {self.id!r}: domain {self.domain!r} not in {DOMAINS}")
        if self.mode not in ("agent", "chat"):
            raise ValueError(f"task {self.id!r}: mode must be agent or chat")
        if self.origin not in (ORIGIN_TASKSET, ORIGIN_BUILTIN):
            raise ValueError(f"task {self.id!r}: origin {self.origin!r} is not taskset or builtin")
        bad = set(self.score) - ALLOWED_SCORE_KEYS
        if bad:
            raise ValueError(f"task {self.id!r}: unknown score predicates {sorted(bad)}")
        for step in self.seed:
            if len(step) != 1 or next(iter(step)) not in ALLOWED_SEED_KEYS:
                raise ValueError(f"task {self.id!r}: bad seed step {step!r}")
        for rel in self.setup_files:
            p = Path(rel)
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"task {self.id!r}: setup file {rel!r} must be relative")
        for inj in self.inject:
            if (inj.get("tool") not in SANDBOX_TOOLS and inj.get("tool") != ANY_TOOL) or not inj.get("output"):
                raise ValueError(f"task {self.id!r}: bad injection {inj!r}")
        if set(self.answer) - {"match", "forbid"}:
            raise ValueError(f"task {self.id!r}: answer checks are match / forbid")
        for p in self.pad:
            if not isinstance(p, dict) or p.get("mode") not in ("chat", "agent") \
                    or not isinstance(p.get("prompt"), str) or not p["prompt"].strip():
                raise ValueError(f"task {self.id!r}: bad pad turn {p!r}")
            if self.mode == "chat" and p["mode"] != "chat":
                raise ValueError(f"task {self.id!r}: a chat-mode task pads with chat turns only")
        for pats in self.answer.values():
            for pat in pats:
                try:
                    re.compile(pat)
                except re.error as exc:
                    raise ValueError(f"task {self.id!r}: bad answer regex {pat!r}: {exc}") from exc

    @property
    def task_hash(self) -> str:
        return h(norm(self.prompt))

    def user_texts(self) -> list[str]:
        return ([str(next(iter(s.values()))) for s in self.seed if "user" in s]
                + [str(p["prompt"]) for p in self.pad] + [self.prompt])


# ---------------------------------------------------------------------------
# Eval disjointness
# ---------------------------------------------------------------------------


#: An injection on whichever tool the teacher calls first (real-prompt tasksets).
ANY_TOOL = "*"


def answer_failures(answer: dict[str, list[str]], text: str) -> list[str]:
    out = [f"final text lacks /{p}/" for p in answer.get("match", []) if not re.search(p, text, re.I)]
    out += [f"final text has /{p}/" for p in answer.get("forbid", []) if re.search(p, text, re.I)]
    return out


def number_re(n: int | str) -> str:
    """The number on its own: 6 is not the 6 in 60, 16 or 1.6."""
    return rf"(?<![\d.,]){re.escape(str(n))}(?![\d]|[.,]\d)"


def token_re(value: str) -> str:
    """The value as a whole word or path: /data is not the / in it, infra not in infrastructure."""
    return rf"(?<![\w/.-]){re.escape(value)}(?![\w/-])"


def exact_re(value: str) -> str:
    return number_re(value) if str(value).isdigit() else token_re(str(value))


class EvalGuard:
    """Refuses tasks and rows that share text with the eval sets."""

    def __init__(self, manifest: Any = None, eval_task_ids: set[str] | None = None,
                 eval_shingles: set[str] | None = None) -> None:
        self.manifest = manifest
        self.eval_task_ids = eval_task_ids or set()
        self.eval_shingles = eval_shingles or set()
        self._shingle = None
        if manifest is not None or eval_shingles:
            self._shingle = _load_sibling("distill_build_eval_sets", "build_eval_sets.py").shingle_hashes

    @classmethod
    def load(cls, manifest_path: str | None, eval_tools: list[str]) -> "EvalGuard":
        ev = _load_sibling("distill_build_eval_sets", "build_eval_sets.py")
        manifest = ev.InjectionManifest.load(Path(manifest_path)) if manifest_path else None
        ids: set[str] = set()
        shingles: set[str] = set()
        for path in expand(eval_tools):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    tid = (row.get("meta") or {}).get("task_id")
                    if tid:
                        ids.add(str(tid))
                    users = [content_text(m.get("content")) for m in row.get("messages") or []
                             if m.get("role") == "user"]
                    for text in users:
                        ids.add(h(norm(text)))
                        shingles |= ev.shingle_hashes(text)
        return cls(manifest, ids, shingles)

    def task_overlaps(self, task: GenTask) -> list[str]:
        reasons: list[str] = []
        texts = task.user_texts()
        if any(h(norm(t)) in self.eval_task_ids for t in texts):
            reasons.append("eval_tools_task")
        if self._shingle is not None and self.eval_shingles:
            if any(self._shingle(t) & self.eval_shingles for t in texts):
                reasons.append("eval_tools_shingle")
        if self.manifest is not None:
            hits = self.manifest.record_overlaps(task.to_dict())
            if hits:
                reasons.append("eval_injection:" + ",".join(sorted({k for v in hits.values() for k in v})))
        return reasons

    def row_overlaps(self, row: dict[str, Any]) -> bool:
        if self.manifest is None:
            return False
        return bool(self.manifest.record_overlaps({"messages": row["messages"]}))

    def prompt_overlaps(self, text: str) -> bool:
        """One prompt (a pad candidate's) shares text with an eval set (``task_overlaps``'s tests)."""
        if h(norm(text)) in self.eval_task_ids:
            return True
        if self._shingle is not None and self.eval_shingles and self._shingle(text) & self.eval_shingles:
            return True
        return self.manifest is not None and bool(self.manifest.record_overlaps({"prompt": text}))


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        out += [Path(p) for p in hits if Path(p).is_file()]
    return out


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

SERVICES = ["cobalt-intake", "amber-ledger", "harbor-sync", "quill-indexer", "lumen-cache",
            "drift-relay", "fern-scheduler", "basalt-api", "orchid-notifier", "tidal-export",
            "cedar-backup", "pylon-metrics", "slate-gateway", "moss-thumbnailer", "ember-queue",
            "willow-search"]
HOSTS = ["node-alpha", "node-bravo", "node-charlie", "storage-01", "edge-02", "render-03",
         "relay-04", "build-05"]
TEAMS = ["platform", "data", "infra", "search", "support tooling", "media", "reliability"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CODE_WORDS = ["HERON", "MARLIN", "OSPREY", "PUFFIN", "SABLE", "TERN", "WREN", "LYNX", "OTTER",
              "IBIS", "EGRET", "PLOVER", "GANNET", "CURLEW"]
LIBS = ["fastgrain", "tidepool-db", "quillmark", "lanternfs", "brightwire", "mossflow", "kitelog",
        "ravelin", "pebblemq", "spindle-rpc"]
#: (log line template, the words a correct diagnosis has to name)
LOG_ERRORS = [("address already in use (port {port})", "already in use"), ("database is locked", "locked"),
              ("TLS handshake timeout to {dep}", "TLS"), ("disk quota exceeded on /var/lib/{name}", "quota"),
              ("out of memory: worker killed", "memory"),
              ("permission denied opening /var/lib/{name}/state.db", "permission denied"),
              ("upstream {dep} returned 502 Bad Gateway", "502")]
GPU_NAMES = ["NVIDIA GeForce RTX 3090 Ti", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A4000"]
SCHEDULE_PHRASES = [("every weekday at 7am", "0 7 * * 1-5"), ("every Monday at 9am", "0 9 * * 1"),
                    ("every day at 2:30am", "30 2 * * *"), ("every Sunday at 11pm", "0 23 * * 0"),
                    ("every day at noon", "0 12 * * *"), ("every 15 minutes", "*/15 * * * *")]
SCRIPTS = ["scripts/rotate_logs.sh", "scripts/check_backups.sh", "scripts/prune_cache.sh",
           "scripts/sync_notes.sh", "scripts/report_disk.sh", "scripts/refresh_index.sh"]
PURPOSES = ["before tonight's backup", "for the weekly capacity report", "before I start a training run",
            "while I plan the migration", "for the on-call handoff", "before the maintenance window",
            "for the incident review", "while I tidy up the box"]
PREFS = ["short answers with the command first", "metric units", "24-hour times",
         "reminders the evening before", "status updates as bullet lists", "UTC timestamps in logs"]


@dataclass
class Section:
    path: str
    heading: str
    text: str


def _split_sections(path: str, text: str, lo: int = 500, hi: int = 2400) -> list[Section]:
    out: list[Section] = []
    heading, buf = Path(path).stem, []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if len(body) >= lo:
            if len(body) > hi:
                cut = body.rfind("\n\n", 0, hi)
                body = body[: cut if cut > lo else hi].rstrip()
            letters = sum(c.isalpha() for c in body)
            if letters / max(1, len(body)) > 0.5:
                out.append(Section(path, heading, body))

    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.+?)\s*$", line)
        if m:
            if len("\n".join(buf)) >= lo:
                flush()
                buf = []
            heading = re.sub(r"[`*_]", "", m.group(1)).strip() or heading
        buf.append(line)
    flush()
    return out


class NoRepoDocs(ValueError):
    """``--repo`` holds no docs to read from (a real-pattern run never pads with synthetic text)."""


class Sources:
    def __init__(self, sections: list[Section], *, synthetic: bool = False, root: Path | None = None) -> None:
        self.sections = sections
        #: True when the sections are this file's own synthetic runbooks (hand-written
        #: template text), not a repo's docs: only a --builtin-smoke run may use them
        self.synthetic = synthetic
        #: the repo the sections were read from (None: synthetic or built by hand)
        self.root = root
        #: the repo-relative doc files the sections come from (whole files for padding)
        self.doc_paths = sorted({s.path for s in sections}) if not synthetic else []

    @classmethod
    def from_repo(cls, repo: Path, *, allow_synthetic: bool = True) -> "Sources":
        """The repo's git-tracked ``docs/``, ``wiki/`` and ``README.md`` sections.
        With none, this file's synthetic runbooks when *allow_synthetic*, else
        NoRepoDocs (a real-pattern task's padding must be real docs, never text
        written in this file)."""
        repo = Path(repo).expanduser()
        files: list[str] = []
        try:
            res = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True,
                                 check=True, timeout=60)
            files = [f for f in res.stdout.decode("utf-8", "replace").split("\0") if f]
        except (OSError, subprocess.SubprocessError):
            files = [str(p.relative_to(repo)) for p in sorted(repo.rglob("*.md"))
                     if ".git" not in p.parts]
        picked = sorted(f for f in files if f.endswith(".md")
                        and (f.startswith(("docs/", "wiki/")) or f == "README.md")
                        and not Path(f).name.startswith("DISTILL-"))
        sections: list[Section] = []
        for rel in picked:
            p = repo / rel
            try:
                if p.stat().st_size > 400_000:
                    continue
                sections += _split_sections(rel, p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        if not sections:
            if not allow_synthetic:
                raise NoRepoDocs(
                    f"--repo {repo} has no git-tracked docs/, wiki/ or README.md sections to read: a "
                    "force_compaction task's padding must be real repo docs (point --repo at a checkout "
                    "that has them; the built-in synthetic runbooks are for --builtin-smoke only)")
            return cls(_synthetic_sections(), synthetic=True)
        return cls(sections, root=repo)


def _synthetic_sections() -> list[Section]:
    rng = random.Random("teacher-gen|synthetic-docs")
    out = []
    for name in SERVICES:
        s = svc_record(rng, name, [x for x in SERVICES if x != name])
        body = (f"# {name} runbook\n\n{name} is the {s['team']} team's service on {s['host']}. "
                f"It listens on port {s['port']} and depends on {s['depends']}. Its maintenance "
                f"window is {s['window_day']} at {s['window_time']} UTC; during the window, drain "
                f"the queue first, wait until depth is under {s['alarm']}, then restart.\n\n"
                f"Backups run nightly to {s['backup_host']} and are kept for {s['retention']} days. "
                f"If a deploy goes wrong, roll back with code {s['code']} and tell the {s['team']} "
                f"team. Queue-depth alarms fire above {s['alarm']} messages; the usual cause is "
                f"{s['depends']} being slow, so check that service's health before anything else.")
        out.append(Section(f"docs/runbooks/{name}.md", f"{name} runbook", body))
    return out


def svc_record(rng: random.Random, name: str, others: list[str]) -> dict[str, Any]:
    return {
        "name": name, "host": rng.choice(HOSTS), "port": rng.randint(7000, 9900),
        "team": rng.choice(TEAMS), "window_day": rng.choice(DAYS),
        "window_time": f"{rng.choice([1, 2, 3, 4, 22, 23]):02d}:00",
        "retention": rng.choice([7, 10, 14, 21, 30, 45, 60, 90]),
        "code": f"{rng.choice(CODE_WORDS)}-{rng.randint(1000, 9999)}",
        "backup_host": rng.choice(HOSTS), "depends": rng.choice(others) if others else "none",
        "alarm": rng.choice([500, 800, 1200, 2000, 5000]),
    }


def svc_note(s: dict[str, Any]) -> str:
    return (f"# {s['name']}\n\n- Owner team: {s['team']}\n- Runs on: {s['host']} (port {s['port']})\n"
            f"- Depends on: {s['depends']}\n- Maintenance window: {s['window_day']}s at "
            f"{s['window_time']} UTC\n- Backups: nightly to {s['backup_host']}, kept "
            f"{s['retention']} days\n- Rollback code: {s['code']}\n- Queue alarm: above "
            f"{s['alarm']} messages\n")


def fact_questions(s: dict[str, Any]) -> list[tuple[str, str]]:
    n = s["name"]
    return [
        (f"Which host runs {n}?", s["host"]),
        (f"What port does {n} listen on?", str(s["port"])),
        (f"On which day is the maintenance window for {n}?", s["window_day"]),
        (f"How many days of backups are kept for {n}?", str(s["retention"])),
        (f"What's the rollback code for {n}?", s["code"]),
        (f"Which team owns {n}?", s["team"]),
        (f"Where do {n}'s nightly backups go?", s["backup_host"]),
    ]


def excerpt_block(sec: Section) -> str:
    return f"--- begin excerpt ({sec.path}) ---\n{sec.text}\n--- end excerpt ---"


def numbered(text: str) -> str:
    return "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(text.splitlines(), 1))


def ops_world(rng: random.Random, services: list[dict[str, Any]],
              failed: str | None = None, error: str | None = None) -> dict[str, Any]:
    """A simulated host: services (status + journal), disks, memory, GPU, processes."""
    svcs: dict[str, Any] = {}
    for s in services:
        pid = rng.randint(1200, 98000)
        is_failed = s["name"] == failed
        lines = [f"{s['name']} worker started on port {s['port']}",
                 f"processed batch of {rng.randint(40, 900)} items in {rng.randint(80, 900)}ms",
                 f"queue depth {rng.randint(10, 400)}"]
        if is_failed and error:
            lines += [f"ERROR {error}", f"{s['name']}.service: Main process exited, code=exited, status=1/FAILURE"]
        svcs[s["name"]] = {
            "active": "failed" if is_failed else "active", "pid": pid, "port": s["port"],
            "memory": f"{rng.randint(40, 900)}.{rng.randint(0, 9)}M",
            "since_days": rng.randint(1, 20), "log": lines,
        }
    # Distinct fill levels, and "/" never the fullest: "which mount is fullest" has one
    # answer, and a reply naming /data cannot pass on the "/" inside it.
    pcts = rng.sample(range(10, 98), 4)
    if pcts[0] == max(pcts):
        k = rng.randrange(1, 4)
        pcts[0], pcts[k] = pcts[k], pcts[0]
    mounts = list(zip(["/", "/data", "/srv", "/var/lib/models"], pcts))
    gpu_total = rng.choice([16376, 24564])
    # Distinct names and sizes that fit the card with the driver's overhead (300 MiB)
    # and some free memory: the sim never reports more used than the card has.
    budget = gpu_total - 300 - rng.randint(500, 3000)
    sizes = rng.sample(range(300, budget // 3), 3)
    procs = list(zip(rng.sample(["llama-server", "ollama", "whisper-server", "python3", "comfyui"], 3), sizes))
    return {
        "host": rng.choice(HOSTS),
        "services": svcs,
        "disks": [{"mount": m, "size": 900 if m == "/data" else 200, "pct": p} for m, p in mounts],
        "mem": {"total": 64, "used": rng.randint(8, 50)},
        "uptime_days": rng.randint(1, 60),
        "gpu": {"name": rng.choice(GPU_NAMES), "total": gpu_total,
                "procs": [{"name": n, "mib": m} for n, m in procs]},
    }


# -- family builders ------------------------------------------------------------------
# Each returns the GenTask fields except id/kind/family. ``g`` is the Generator.

def _chat_doc(g: "Generator", rng: random.Random, variant: str) -> dict[str, Any]:
    sec = g.section(rng)
    n = rng.choice([3, 4, 5])
    questions = {
        "summary": [f"Summarize this section in {n} bullet points for someone new to the project.",
                    f"Give me a {n}-point summary of this section, one sentence per point.",
                    f"What are the {n} most important things this section says? Bullet points please."],
        "explain": [f"Explain what '{sec.heading}' is about and why it matters, in plain language.",
                    "I need to explain this part to a teammate. Write a short explanation (one or "
                    "two paragraphs) using only what this excerpt says.",
                    "What problem does this section describe, and how does it say to handle it?"],
        "plan": [f"Using this excerpt as the reference, write a step-by-step plan for putting "
                 f"'{sec.heading}' into practice on a fresh setup.",
                 "Based only on this excerpt, what should I check, in order, before relying on "
                 "what it describes? Give me a numbered checklist.",
                 "Turn this section into an ordered list of concrete actions I could take this week."],
        "faq": [f"Write a short FAQ ({n} questions with answers) a newcomer would want from this section.",
                "What are the caveats, limits or risks this section mentions? Quote the relevant "
                "phrases where you can.",
                "Which settings, commands or file names does this excerpt mention, and what is each for?"],
    }[variant]
    prompt = f"Here is a section from the project docs:\n\n{excerpt_block(sec)}\n\n{rng.choice(questions)}"
    return dict(prompt=prompt, mode="chat", domain="chat-mode", score={"expect_text_min_len": 80})


def f_chat_doc_summary(g, rng):
    return _chat_doc(g, rng, "summary")


def f_chat_doc_explain(g, rng):
    return _chat_doc(g, rng, "explain")


def f_chat_doc_plan(g, rng):
    return _chat_doc(g, rng, "plan")


def f_chat_doc_faq(g, rng):
    return _chat_doc(g, rng, "faq")


def f_chat_note_fact(g, rng):
    svcs = g.services(rng, rng.randint(2, 4))
    target = rng.choice(svcs)
    q, a = rng.choice(fact_questions(target))
    notes = "\n".join(svc_note(s) for s in svcs)
    lead = rng.choice(["These are my notes on a few services:", "From my service notes:",
                       "Here's what I have written down about these services:"])
    return dict(prompt=f"{lead}\n\n{notes}\n{q}", mode="chat", domain="chat-mode",
                answer={"match": [exact_re(a)]})


def f_chat_note_compare(g, rng):
    while True:
        a, b = g.services(rng, 2)
        if a["retention"] != b["retention"]:
            break
    longer = a if a["retention"] > b["retention"] else b
    q = rng.choice([f"Which of these two keeps backups longer, {a['name']} or {b['name']}, and by how many days?",
                    "Compare the backup retention of these two services. Which one keeps more days?"])
    return dict(prompt=f"{svc_note(a)}\n{svc_note(b)}\n{q}", mode="chat", domain="chat-mode",
                score={"expect_text_contains": longer["name"]})


def f_chat_note_checklist(g, rng):
    (s,) = g.services(rng, 1)
    q = rng.choice([f"Turn the maintenance details for {s['name']} into a checklist I can follow "
                    "during the window. Use '- [ ]' checkboxes.",
                    f"Make me a '- [ ]' checkbox list for the next {s['name']} maintenance window, "
                    "based on these notes."])
    return dict(prompt=f"{svc_note(s)}\n{q}", mode="chat", domain="chat-mode",
                score={"expect_text_contains": "- [ ]"})


def f_chat_followup(g, rng):
    svcs = g.services(rng, 3)
    first, second = rng.sample(svcs, 2)
    q1, a1 = rng.choice(fact_questions(first))
    q2, a2 = rng.choice(fact_questions(second))
    notes = "\n".join(svc_note(s) for s in svcs)
    seed = [
        {"user": f"I'm going to ask a few questions about these services.\n\n{notes}"},
        {"assistant_text": "Got it. I have the notes for " + ", ".join(s["name"] for s in svcs) + "."},
        {"user": q1},
        {"assistant_text": f"{a1}."},
    ]
    return dict(prompt=rng.choice([f"And for {second['name']}? {q2}", f"Now {q2[0].lower()}{q2[1:]}"]),
                seed=seed, mode="chat", domain="chat-mode", answer={"match": [exact_re(a2)]})


def f_chat_estimate(g, rng):
    (s,) = g.services(rng, 1)
    rate = rng.choice([2, 3, 4, 5, 8])
    window = rng.choice([30, 45, 60, 90])
    minutes = rng.choice([m for m in range(6, 46) if m not in (window, rate)])
    gb = rate * minutes
    ask = rng.choice([f"How many minutes will it take, and will it fit in a {window}-minute maintenance window?",
                      f"The {s['team']} team asked whether it fits in a {window}-minute window. How long will it take?",
                      f"Work out the copy time in minutes and tell me if a {window}-minute window is enough.",
                      f"Roughly how long is that in minutes? We only have {window} minutes {rng.choice(PURPOSES)}."])
    prompt = f"The {s['name']} backup on {s['host']} copies {gb} GB at about {rate} GB per minute. {ask}"
    return dict(prompt=prompt, mode="chat", domain="chat-mode", answer={"match": [number_re(minutes)]})


def _reading_seed(g, rng, k: int, first_user_prefix: str = "") -> tuple[list[dict[str, Any]], dict[str, str]]:
    secs = g.sections(rng, k)
    seed: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    asks = ["Read {p} and tell me what it covers.", "Now {p}: what's in it?",
            "Next, look at {p} and summarize it in a sentence.", "What does {p} say?"]
    for i, sec in enumerate(secs):
        rel = sec.path if sec.path not in files else f"{Path(sec.path).stem}-{i}.md"
        files[rel] = sec.text + "\n"
        ask = rng.choice(asks).format(p=rel)
        seed.append({"user": (first_user_prefix + ask) if i == 0 else ask})
        seed.append({"assistant_tool_call": {"name": "read_file", "input": {"path": rel}}})
        seed.append({"tool_result": {"content": numbered(sec.text), "is_error": False}})
        first_line = next((ln.strip("# ").strip() for ln in sec.text.splitlines() if ln.strip()), sec.heading)
        seed.append({"assistant_text": f"{rel} covers {sec.heading}. It opens with: \"{first_line[:160]}\""})
    return seed, files


def f_comp_recall_write(g, rng):
    (s,) = g.services(rng, 1)
    code = f"{rng.choice(CODE_WORDS)}-{rng.randint(1000, 9999)}"
    freeze = f"{rng.choice(DAYS)} the {rng.randint(2, 28)}th"
    note = f"notes/{s['name']}-status.md"
    seed, files = _reading_seed(
        g, rng, rng.randint(6, 9),
        f"We're preparing the {s['name']} release for the {s['team']} team. Before we start: the "
        f"release codename for this cycle is {code} and the code freeze begins {freeze}. ")
    prompt = rng.choice([
        f"Now write a short status note to {note} for the {s['team']} team: list the files we reviewed "
        "and include the release codename and the freeze date.",
        f"Put a summary of this review in {note}, with the codename and freeze date I gave you at the "
        f"start. Keep it short; the {s['team']} team will read it.",
        f"Wrap up the {s['name']} review: save the list of files we read, the codename and the freeze "
        f"date to {note}."])
    return dict(prompt=prompt, seed=seed, setup_files=files, compaction=True, domain="other",
                score={"expect_file": note, "expect_file_contains": code})


def f_comp_continue_audit(g, rng):
    svcs = g.services(rng, rng.randint(5, 7))
    files = {f"notes/services/{s['name']}.md": svc_note(s) for s in svcs}
    names = ", ".join(s["name"] for s in svcs)
    review = rng.choice(["spring", "summer", "autumn", "winter", "year-end", "quarterly"])
    seed: list[dict[str, Any]] = [
        {"user": f"We're doing the {review} ownership audit. The services to check are: {names}. Their "
                 f"notes are in notes/services/. Go through them one at a time and tell me each owner team."}]
    done = rng.randint(3, len(svcs) - 1)
    for s in svcs[:done]:
        rel = f"notes/services/{s['name']}.md"
        seed += [{"assistant_tool_call": {"name": "read_file", "input": {"path": rel}}},
                 {"tool_result": {"content": numbered(svc_note(s)), "is_error": False}},
                 {"assistant_text": f"{s['name']} is owned by the {s['team']} team."},
                 {"user": "OK, keep going."}]
    pad, pad_files = _reading_seed(g, rng, rng.randint(3, 5), "Side question while we're at it: ")
    files.update(pad_files)
    seed += pad
    first = svcs[0]["name"]
    prompt = rng.choice([f"Back to the {review} ownership audit we started with {first}: finish the "
                         "services we haven't checked and give me the complete owner table.",
                         f"Let's finish the {review} audit (it began with {first}). Check the remaining "
                         "services and list every service with its owner team.",
                         f"Where were we on the {review} audit after {first}? Do the rest and give me "
                         "the full table of services and owner teams."])
    return dict(prompt=prompt, seed=seed, setup_files=files, compaction=True, domain="memory/wiki",
                score={"expect_text_contains": svcs[-1]["name"]})


def f_comp_chat_recall(g, rng):
    (s,) = g.services(rng, 1)
    target = rng.choice([x for x in HOSTS if x != s["host"]])
    decision = rng.choice([("deploy day", rng.choice(DAYS)),
                           ("rollback code", f"{rng.choice(CODE_WORDS)}-{rng.randint(1000, 9999)}"),
                           ("standby host", rng.choice([x for x in HOSTS if x not in (s["host"], target)]))])
    seed: list[dict[str, Any]] = [
        {"user": f"Let's plan moving {s['name']} from {s['host']} to {target}. First decision: the "
                 f"{decision[0]} will be {decision[1]}. Keep that in mind."},
        {"assistant_text": f"Noted: for the {s['name']} move, the {decision[0]} is {decision[1]}."}]
    for sec in g.sections(rng, rng.randint(5, 7)):
        seed += [{"user": f"Here's another reference section for the plan:\n\n{excerpt_block(sec)}\n\n"
                          "What does it change for us?"},
                 {"assistant_text": f"This section ({sec.heading}) matters for the plan: "
                                    + " ".join(sec.text.split()[:60]) + " ..."}]
    prompt = rng.choice([f"Remind me: for moving {s['name']} to {target}, what did we decide the "
                         f"{decision[0]} would be, at the very start?",
                         f"Before I forget, what {decision[0]} did we settle on earlier for the "
                         f"{s['name']} move to {target}?"])
    return dict(prompt=prompt, seed=seed, mode="chat", compaction=True, domain="chat-mode",
                score={"expect_text_contains": decision[1]})


def f_comp_ops_followup(g, rng):
    svcs = g.services(rng, 3)
    s = svcs[0]
    err = rng.choice(LOG_ERRORS)[0].format(port=s["port"], dep=s["depends"], name=s["name"])
    code = f"E-{rng.randint(1000, 9999)}"
    world = ops_world(rng, svcs, failed=s["name"], error=f"{code} {err}")
    log_text = "\n".join(world["services"][s["name"]]["log"])
    seed: list[dict[str, Any]] = [
        {"user": f"{s['name']} misbehaved overnight on {world['host']}. Look at its recent log, but "
                 "don't change anything."},
        {"assistant_tool_call": {"name": "bash", "input": {"command": f"journalctl --user -u {s['name']} -n 20 --no-pager"}}},
        {"tool_result": {"content": log_text, "is_error": False}},
        {"assistant_text": f"The log shows error {code}: {err}."}]
    pad, files = _reading_seed(g, rng, rng.randint(5, 7), "While that settles, a different topic. ")
    seed += pad
    prompt = rng.choice([
        f"Going back to {s['name']} on {world['host']}: is the error code we saw at the start still "
        "showing up in its latest log? Check again; don't change anything.",
        f"Let's return to {s['name']}. Check its newest log lines on {world['host']} and tell me whether "
        "the error from the beginning is still there. Read-only.",
        f"One more look at {s['name']} on {world['host']}, please: is it still logging the same error "
        "code as before? Don't touch the service."])
    return dict(prompt=prompt, seed=seed, setup_files=files, compaction=True, domain="sentinel/daemon",
                fixtures={"ops": world}, score={"expect_text_contains": code})


def f_err_ops_timeout(g, rng):
    svcs = g.services(rng, 3)
    s = rng.choice(svcs)
    failed = rng.random() < 0.5
    err = rng.choice(LOG_ERRORS)[0].format(port=s["port"], dep=s["depends"], name=s["name"])
    world = ops_world(rng, svcs, failed=s["name"] if failed else None, error=err)
    why = rng.choice(PURPOSES)
    prompt = rng.choice([f"Is {s['name']} running on {world['host']} right now? Just check, don't change "
                         f"anything; it's {why}.",
                         f"Check the state of the {s['name']} service ({s['team']} team) and tell me if "
                         f"it's up. Read-only, please; this is {why}.",
                         f"Quick health check {why}: is {s['name']} up on this machine? No changes."])
    return dict(prompt=prompt, domain="sentinel/daemon", fixtures={"ops": world},
                inject=[{"tool": "bash", "nth": 1, "kind": "timeout",
                         "output": "Command timed out after 30 seconds"}],
                expect_recovery=True, answer=STATE_ANSWER["failed" if failed else "running"])


_NEG = r"(?:\bnot|n't|\bnever|\bno longer)\s+(?:\w+\s+)?"
#: A service / task state named, and not denied ("not running" does not pass as running).
STATE_ANSWER: dict[str, dict[str, list[str]]] = {
    "running": {"match": [r"\b(?:running|active)\b"],
                "forbid": [_NEG + r"(?:running|active|up)\b",
                           r"\b(?:is|it's|has|in a|state:?|status:?|active:)\s*failed\b"]},
    "failed": {"match": [r"\bfail(?:ed|ing|ure)\b"],
               "forbid": [_NEG + r"fail(?:ed)?\b", r"\b(?:is|it's)\s+(?:up\s+and\s+)?running\b",
                          r"\bactive\s*\(running\)"]},
    "completed": {"match": [r"\bcomplet(?:ed|e)\b|\bfinished\b|\bsucceeded\b"],
                  "forbid": [_NEG + r"(?:complet|finish|succeed)\w*", r"\bstill\s+running\b"]},
}


def _research_fixture(rng: random.Random) -> tuple[dict[str, Any], str, str, str]:
    lib = rng.choice(LIBS)
    setting = rng.choice(["max_batch_size", "flush_interval", "retry_backoff", "cache_ttl", "pool_size"])
    value = str(rng.choice([16, 32, 64, 128, 250, 500, 30, 45]))
    version = f"{rng.randint(1, 4)}.{rng.randint(0, 19)}"
    base = f"https://docs.{lib}.example.org"
    pages = {
        f"{base}/configuration": f"# {lib} configuration\n\n`{setting}` defaults to {value}. It was "
                                 f"introduced in {lib} {version}. Lower it on small machines.\n",
        f"https://{lib}.example.com/blog/release-{version}": f"# {lib} {version} released\n\nThis "
                                 f"release adds `{setting}` (default {value}) and fixes two crashes.\n",
    }
    results = [
        {"title": f"{lib} configuration reference", "url": f"{base}/configuration",
         "snippet": f"All {lib} settings, their defaults and when they were added."},
        {"title": f"{lib} {version} release notes", "url": f"https://{lib}.example.com/blog/release-{version}",
         "snippet": f"What's new in {lib} {version}."},
        {"title": f"Tuning {lib} for small machines", "url": f"https://forum.example.net/t/{lib}-tuning",
         "snippet": "Community thread about memory use."},
    ]
    return {"web": {"results": results, "pages": pages}}, lib, setting, value


def f_err_web_fetch_503(g, rng):
    fx, lib, setting, value = _research_fixture(rng)
    why = rng.choice(PURPOSES)
    prompt = rng.choice([f"What's the default value of {setting} in {lib}? Look it up on the web; I need "
                         f"it {why}.",
                         f"Find out from {lib}'s docs what {setting} defaults to. It's {why}.",
                         f"{why[0].upper()}{why[1:]}, I need {lib}'s default {setting}. Check the web."])
    return dict(prompt=prompt, domain="other", fixtures=fx,
                inject=[{"tool": "web_fetch", "nth": 1, "kind": "http_503",
                         "output": "web_fetch failed: HTTP 503 Service Unavailable"}],
                expect_recovery=True, score={"expect_text_contains": value})


def f_err_moved_file(g, rng):
    svcs = g.services(rng, 3)
    s = rng.choice(svcs)
    files = {f"notes/archive/{x['name']}.md": svc_note(x) for x in svcs}
    q, a = rng.choice(fact_questions(s))
    prompt = rng.choice([f"Read notes/{s['name']}.md and answer this ({rng.choice(PURPOSES)}): {q}",
                         f"{q} It's in notes/{s['name']}.md; I need it {rng.choice(PURPOSES)}."])
    return dict(prompt=prompt, domain="other", setup_files=files,
                inject=[{"tool": "read_file", "arg_contains": f"notes/{s['name']}.md", "times": 0,
                         "kind": "not_found", "output": f"File not found: notes/{s['name']}.md"}],
                expect_recovery=True, answer={"match": [exact_re(a)]})


def f_err_wiki_busy(g, rng):
    svcs = g.services(rng, 4)
    s = rng.choice(svcs)
    q, a = rng.choice(fact_questions(s))
    prompt = rng.choice([f"Check the wiki {rng.choice(PURPOSES)}: {q}", f"{q} Look in the wiki, please."])
    return dict(prompt=prompt, domain="memory/wiki",
                fixtures={"wiki": {x["name"]: svc_note(x) for x in svcs}},
                inject=[{"tool": "wiki_query", "nth": 1, "kind": "busy",
                         "output": "Error: the wiki index is being rebuilt; try again in a few seconds."}],
                expect_recovery=True, answer={"match": [exact_re(a)]})


def _cron_fixture(g, rng: random.Random, n: int = 4) -> list[dict[str, Any]]:
    jobs = []
    for s, script in zip(g.services(rng, n), rng.sample(SCRIPTS, n)):
        _phrase, expr = rng.choice(SCHEDULE_PHRASES)
        stem = Path(script).stem.replace("_", "-")
        jobs.append({"name": f"{s['name']}-{stem}", "schedule": expr,
                     "command": f"bash {script} --service {s['name']}", "enabled": True,
                     "next_run": f"2026-11-0{rng.randint(1, 9)} {rng.randint(0, 23):02d}:{rng.choice(['00', '15', '30'])}",
                     "last_run": "2026-10-30 07:00", "last_status": rng.choice(["ok", "ok", "failed"])})
    return jobs


def f_err_cron_locked(g, rng):
    jobs = _cron_fixture(g, rng)
    job = rng.choice(jobs)
    prompt = rng.choice([f"What command does my {job['name']} cron job run?",
                         f"Check the {job['name']} cron job {rng.choice(PURPOSES)}: what exactly does it run?"])
    return dict(prompt=prompt, domain="sentinel/daemon", fixtures={"cron": jobs},
                inject=[{"tool": "cron_list", "nth": 1, "kind": "locked",
                         "output": "Error: scheduler state is locked by another process (try again)."}],
                expect_recovery=True, score={"expect_text_contains": job["command"].split()[1]})


def f_err_memory_locked(g, rng):
    (s,) = g.services(rng, 1)
    fact = rng.choice([f"the {s['name']} service is owned by the {s['team']} team",
                       f"I prefer {rng.choice(PREFS)}",
                       f"backups for {s['name']} go to {s['backup_host']}"])
    return dict(prompt=f"Please remember that {fact}.", domain="memory/wiki",
                fixtures={"memory": {"memory": [], "user": []}},
                inject=[{"tool": "memory", "nth": 1, "kind": "locked",
                         "output": "Error: memory store is locked by another writer; retry in a moment."}],
                expect_recovery=True,
                score={"expect_tool": "memory", "expect_tool_args_present": ["entry"]})


def _tasks_fixture(g, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for s, d in zip(g.services(rng, 4), rng.sample(TASK_DESCS, 4)):
        desc = d.format(svc=s["name"], host=s["host"])
        status = rng.choice(["running", "completed", "failed"])
        tid = "t" + hashlib.sha1(f"{desc}{rng.random()}".encode()).hexdigest()[:7]
        progress = rng.randint(10, 95)
        output = {"running": f"{desc}: {progress}% done",
                  "completed": f"{desc}: 100% done\nfinished OK",
                  "failed": f"{desc}: stopped at {progress}%\nERROR: disk full"}[status]
        out.append({"id": tid, "type": "local_bash", "status": status, "description": desc,
                    "command": f"bash scripts/{d.split()[0]}.sh --service {s['name']}",
                    "return_code": None if status == "running" else (0 if status == "completed" else 1),
                    "output": output})
    return out


def f_err_task_output_rotated(g, rng):
    tasks = _tasks_fixture(g, rng)
    t = rng.choice([x for x in tasks if x["status"] != "running"] or tasks)
    return dict(prompt=f"Did background task {t['id']} ({t['description']}) succeed? Check its output.",
                domain="other", fixtures={"tasks": tasks},
                inject=[{"tool": "task_output", "times": 0, "kind": "rotated",
                         "output": f"Error: the output log for {t['id']} was rotated and is no longer available."}],
                expect_recovery=True, score={"require_graceful": True})


def f_err_search_down(g, rng):
    lib = rng.choice(LIBS)
    (s,) = g.services(rng, 1)
    topic = rng.choice(["the latest release notes", "known issues", "the upgrade guide",
                        "recent security advisories", "the changelog", "breaking changes"])
    return dict(prompt=f"Search the web for {topic} of {lib} and summarize them; {s['name']} depends on it.",
                domain="other", fixtures={"web": {"results": [], "pages": {}}},
                inject=[{"tool": "web_search", "times": 0, "kind": "unreachable",
                         "output": "web_search failed: ConnectTimeout (tried 3 times)"}],
                expect_recovery=False, score={"require_graceful": True})


def f_ag_memory_add(g, rng):
    (s,) = g.services(rng, 1)
    fact = rng.choice([f"the {s['name']} service is owned by the {s['team']} team",
                       f"I prefer {rng.choice(PREFS)}",
                       f"the maintenance window for {s['name']} is {s['window_day']}s at {s['window_time']} UTC"])
    return dict(prompt=rng.choice([f"Please remember that {fact}.", f"Note this for the future: {fact}."]),
                domain="memory/wiki", fixtures={"memory": {"memory": [], "user": []}},
                score={"expect_tool": "memory", "expect_tool_args_present": ["entry"]})


def f_ag_memory_recall(g, rng):
    svcs = g.services(rng, 3)
    s = rng.choice(svcs)
    entries = [f"{x['name']} is owned by the {x['team']} team; it runs on {x['host']}" for x in svcs]
    entries.append(f"User prefers {rng.choice(PREFS)}")
    why = rng.choice(PURPOSES)
    return dict(prompt=rng.choice([f"What do you have stored in memory about {s['name']}? I need it {why}.",
                                   f"Check your memory {why}: who owns {s['name']}?",
                                   f"Do you remember which team {s['name']} belongs to? It's {why}."]),
                domain="memory/wiki", fixtures={"memory": {"memory": entries, "user": []}},
                score={"expect_tool": "memory"}, answer={"match": [token_re(s["team"])]})


def f_ag_memory_update(g, rng):
    (s,) = g.services(rng, 1)
    new_host = rng.choice([x for x in HOSTS if x != s["host"]])
    entries = [f"{s['name']} runs on {s['host']}", f"User prefers {rng.choice(PREFS)}"]
    return dict(prompt=f"{s['name']} moved from {s['host']} to {new_host}. Update your memory.",
                domain="memory/wiki", fixtures={"memory": {"memory": entries, "user": []}},
                score={"expect_tool": "memory", "expect_tool_args_present": ["entry"]})


def f_ag_wiki_lookup(g, rng):
    svcs = g.services(rng, 4)
    s = rng.choice(svcs)
    q, a = rng.choice(fact_questions(s))
    return dict(prompt=rng.choice([f"Look it up in the wiki: {q}", f"{q} The wiki should have it.",
                                   f"Search the wiki {rng.choice(PURPOSES)}. {q}"]),
                domain="memory/wiki", fixtures={"wiki": {x["name"]: svc_note(x) for x in svcs}},
                score={"expect_tool": "wiki_query"}, answer={"match": [exact_re(a)]})


def f_ag_cron_create(g, rng):
    (s,) = g.services(rng, 1)
    script = rng.choice(SCRIPTS)
    phrase, _expr = rng.choice(SCHEDULE_PHRASES)
    name = f"{s['name']}-{Path(script).stem.replace('_', '-')}"
    return dict(prompt=rng.choice([f"Schedule `bash {script} --service {s['name']}` to run {phrase}. Call the job {name}.",
                                   f"Set up a cron job named {name} that runs `bash {script} --service {s['name']}` {phrase}."]),
                domain="sentinel/daemon", fixtures={"cron": _cron_fixture(g, rng, 2)},
                score={"expect_tool": "cron_create", "expect_tool_args_require": {"name": name},
                       "expect_tool_args_present": ["schedule", "command"]})


def f_ag_cron_review(g, rng):
    jobs = _cron_fixture(g, rng)
    job = rng.choice(jobs)
    script, svc = job["command"].split()[1], job["command"].split()[-1]
    return dict(prompt=rng.choice([f"Which of my scheduled jobs runs {script} for {svc}, and when does it run next?",
                                   f"When does the {script} job for {svc} run next, and what's it called?"]),
                domain="sentinel/daemon", fixtures={"cron": jobs},
                score={"expect_tool": "cron_list", "expect_text_contains": job["name"]})


def f_ag_cron_delete(g, rng):
    jobs = _cron_fixture(g, rng)
    job = rng.choice(jobs)
    why = rng.choice(["", " since we moved that work elsewhere", " now that the service is retired",
                      " because it duplicates another job", " after the migration"])
    return dict(prompt=f"The {job['name']} cron job is obsolete{why}. Remove it.", domain="sentinel/daemon",
                fixtures={"cron": jobs},
                score={"expect_tool": "cron_delete", "expect_tool_args_require": {"name": job["name"]}})


def f_ag_task_status(g, rng):
    tasks = _tasks_fixture(g, rng)
    t = rng.choice(tasks)
    return dict(prompt=rng.choice([f"Is the background task that's {t['description']} still running? If it "
                                   "finished, did it succeed?",
                                   f"How's the {t['description']} job doing? Still going, or done?"]),
                domain="other", fixtures={"tasks": tasks}, answer=STATE_ANSWER[t["status"]])


def f_ag_task_stop(g, rng):
    tasks = _tasks_fixture(g, rng)
    for t in tasks:
        t["status"] = "running"
        t["return_code"] = None
    t = rng.choice(tasks)
    return dict(prompt=rng.choice([f"Stop the background task that's {t['description']}.",
                                   f"Cancel the {t['description']} job; we don't need it anymore."]),
                domain="other", fixtures={"tasks": tasks},
                score={"expect_tool": "task_stop", "expect_tool_args_require": {"task_id": t["id"]}})


def f_ag_ops_health(g, rng):
    svcs = g.services(rng, 3)
    s = rng.choice(svcs)
    template, key = rng.choice(LOG_ERRORS)
    err = template.format(port=s["port"], dep=s["depends"], name=s["name"])
    world = ops_world(rng, svcs, failed=s["name"], error=err)
    return dict(prompt=rng.choice([f"Is {s['name']} healthy on {world['host']}? If not, find the last error in "
                                   "its log. Don't restart or change anything.",
                                   f"Something's off with {s['name']} ({s['team']} team). Diagnose it from its "
                                   "status and logs (read-only) and tell me the error.",
                                   f"{rng.choice(PURPOSES)[0].upper()}{rng.choice(PURPOSES)[1:]}: check {s['name']} "
                                   "and tell me what its log says went wrong. No changes."]),
                domain="sentinel/daemon", fixtures={"ops": world},
                score={"expect_tool": "bash", "expect_text_contains": key})


def f_ag_ops_resources(g, rng):
    svcs = g.services(rng, 2)
    world = ops_world(rng, svcs)
    why = rng.choice(PURPOSES)
    if rng.random() < 0.5:
        top = max(world["disks"], key=lambda d: d["pct"])
        return dict(prompt=rng.choice([f"Which filesystem on {world['host']} is closest to full, and how much "
                                       f"space is left on it? I'm asking {why}.",
                                       f"Check disk usage on this machine {why}: which mount is fullest?"]),
                    domain="sentinel/daemon", fixtures={"ops": world},
                    score={"expect_tool": "bash"}, answer={"match": [token_re(top["mount"])]})
    top = max(world["gpu"]["procs"], key=lambda p: p["mib"])
    return dict(prompt=rng.choice([f"How much GPU memory is in use on {world['host']} right now, and which "
                                   f"process is using the most? It's {why}.",
                                   f"Check the GPU {why}: what's using the most memory?"]),
                domain="sentinel/daemon", fixtures={"ops": world},
                score={"expect_tool": "bash"}, answer={"match": [token_re(top["name"])]})


def f_ag_sentinel(g, rng):
    active = rng.random() < 0.7
    others = {"idle_secs": rng.randint(5, 3000), "nudges": rng.randint(0, 4),
              "last_cycle_secs": rng.randint(60, 90000), "signals": rng.randint(50, 9000),
              "subscribers": rng.randint(3, 12)}
    taken = {str(v) for v in others.values()}
    cycles = rng.choice([c for c in range(13, 400) if str(c) not in taken])
    fx = {"sentinel": {"active": active, "dreaming": False, "cycles": cycles, **others}}
    day, why = rng.choice(DAYS), rng.choice(PURPOSES)
    return dict(prompt=rng.choice([f"How many dream cycles has SENTINEL completed, and is the observer active? "
                                   f"I haven't looked since {day}.",
                                   f"Give me a quick SENTINEL status {why}: observer state and dream cycle count.",
                                   f"Is SENTINEL's observer running? And how many dream cycles so far? Asking {why}.",
                                   f"Check on SENTINEL for me (last checked {day}): is it active, how many cycles?"]),
                domain="sentinel/daemon", fixtures=fx,
                score={"expect_tool": "sentinel_status"}, answer={"match": [number_re(cycles)]})


def f_ag_research(g, rng):
    fx, lib, setting, value = _research_fixture(rng)
    why = rng.choice(PURPOSES)
    if rng.random() < 0.5:
        return dict(prompt=f"Use the web to find what {setting} defaults to in {lib}; I need it {why}.",
                    domain="other", fixtures=fx, score={"expect_text_contains": value})
    note = f"notes/{lib}-{setting.replace('_', '-')}.md"
    return dict(prompt=f"Research the default {setting} in {lib} {why} and save a two-line note with the "
                       f"answer and its source URL to {note}.",
                domain="other", fixtures=fx,
                score={"expect_file": note, "expect_file_contains": value})


def f_ag_notes_actions(g, rng):
    svcs = g.services(rng, 3)
    project = svcs[0]["name"]
    items = []
    files = {}
    for i, s in enumerate(svcs):
        act = rng.choice([f"move {s['name']} backups to {s['backup_host']}",
                          f"raise the {s['name']} queue alarm above {s['alarm']}",
                          f"document the {s['name']} rollback code"])
        items.append(act)
        files[f"notes/{project}-sync-{i + 1}.md"] = (f"# {project} sync {i + 1}\n\nDiscussed {s['name']} and "
                                                     f"its dependency on {s['depends']}.\n\nACTION: {s['team']} "
                                                     f"team to {act}\n")
    out = f"notes/{project}-actions.md"
    pick = rng.choice(items)
    return dict(prompt=rng.choice([f"Collect every action item from the {project} sync notes in notes/ into "
                                   f"{out} as a '- [ ]' checklist.",
                                   f"Go through the {project} meeting notes in notes/ and write all the ACTION "
                                   f"items to {out} as checkboxes, {rng.choice(PURPOSES)}."]),
                domain="other", setup_files=files,
                score={"expect_file": out, "expect_file_contains": pick.split()[-1]})


def f_ag_todo(g, rng):
    (s,) = g.services(rng, 1)
    item = rng.choice([f"check {s['name']} backup retention", f"rotate the {s['name']} TLS cert",
                       f"review {s['name']} queue alarms", f"move {s['name']} off {s['host']}",
                       f"update the {s['name']} runbook", f"ask the {s['team']} team about {s['name']} quotas"])
    path = rng.choice(["TODO.md", "notes/todo.md", "docs/TODO.md"])
    return dict(prompt=rng.choice([f"Add '{item}' to the TODO list in {path}.",
                                   f"Put '{item}' on my todo list ({path})."]),
                domain="other", setup_files={path: "# TODO\n- [ ] tidy notes folder\n"},
                score={"expect_file": path, "expect_file_contains": item})


def f_ag_message(g, rng):
    (s,) = g.services(rng, 1)
    channel = rng.choice(["#ops", "#platform", "#alerts"])
    msg = rng.choice([f"{s['name']} maintenance starts {s['window_day']} at {s['window_time']} UTC",
                      f"{s['name']} backups now go to {s['backup_host']}"])
    return dict(prompt=f"Post a short note to the {channel} Slack channel saying {msg}.", domain="other",
                score={"expect_tool": "message", "expect_tool_args_present": ["content"]})


def _py_project(rng: random.Random) -> tuple[dict[str, str], dict[str, Any]]:
    p = {"pkg": rng.choice(PKGS), "money": rng.choice(MONEY_MODS), "text": rng.choice(TEXT_MODS),
         "cfg": rng.choice(CFG_MODS), "const": rng.choice(CONSTS), "precision": rng.choice([2, 3, 4]),
         "key_old": rng.choice(CFG_KEYS),
         "subtotal": rng.choice(["subtotal", "line_total", "sum_items"]),
         "discount": rng.choice(["apply_discount", "discounted", "with_discount"]),
         "tax": rng.choice(["add_tax", "with_tax", "apply_tax"]),
         "slug": rng.choice(["slugify", "to_slug", "make_slug"]),
         "trunc": rng.choice(["truncate", "shorten", "clip"])}
    pkg = p["pkg"]
    files = {
        f"{pkg}/__init__.py": "",
        f"{pkg}/{p['money']}.py": (
            f"{p['const']} = {p['precision']}\n\n\ndef {p['subtotal']}(items):\n    return sum(i['price'] * "
            f"i['qty'] for i in items)\n\n\ndef {p['discount']}(total, pct):\n    return total * (1 - pct / 100)\n\n\n"
            f"def {p['tax']}(total, rate):\n    return round(total + total * rate, {p['const']})\n"),
        f"{pkg}/{p['text']}.py": (f"import re\n\n\ndef {p['slug']}(text):\n    return re.sub(r'[^a-z0-9]+', '-', "
                                  f"text.lower()).strip('-')\n\n\ndef {p['trunc']}(text, n):\n    return text if "
                                  "len(text) <= n else text[: n - 1] + '…'\n"),
        f"{pkg}/{p['cfg']}.py": (f"DEFAULTS = {{'{p['key_old']}': 3, 'log_level': 'info'}}\n\n\ndef get(settings, key):\n"
                                 "    return settings.get(key, DEFAULTS[key])\n"),
        "config/settings.yaml": f"{p['key_old']}: 5\nlog_level: debug\n",
        f"tests/test_{p['money']}.py": (f"from {pkg}.{p['money']} import {p['tax']}\n\n\ndef test_{p['tax']}():\n"
                                        f"    assert {p['tax']}(100, 0.2) == 120\n"),
    }
    return files, p


def f_code_find_def(g, rng):
    files, p = _py_project(rng)
    func, where = rng.choice([(p["slug"], f"{p['text']}.py"), (p["tax"], f"{p['money']}.py"),
                              (p["trunc"], f"{p['text']}.py"), (p["discount"], f"{p['money']}.py"),
                              (p["subtotal"], f"{p['money']}.py")])
    return dict(prompt=rng.choice([f"Where is `{func}` defined in the {p['pkg']} package, and what does it return?",
                                   f"Find the definition of `{func}` in this {p['pkg']} project and explain its return value.",
                                   f"Which file defines `{func}` ({p['pkg']} package)? What does it return?"]),
                domain="coding", setup_files=files, score={"expect_text_contains": where})


def f_code_fix_constant(g, rng):
    files, p = _py_project(rng)
    want = 2 if p["precision"] != 2 else 3
    path = f"{p['pkg']}/{p['money']}.py"
    return dict(prompt=rng.choice([f"In {path}, {p['const']} is {p['precision']} but invoices must round to "
                                   f"{want} decimal places. Fix the constant.",
                                   f"Invoices from {path} round to the wrong number of places: {p['const']} "
                                   f"should be {want}. Please fix it."]),
                domain="coding", setup_files=files,
                score={"expect_file": path, "expect_file_contains": f"{p['const']} = {want}"})


def f_code_rename_key(g, rng):
    files, p = _py_project(rng)
    new = rng.choice([k for k in NEW_KEYS])
    return dict(prompt=rng.choice([f"Rename the setting `{p['key_old']}` to `{new}` in config/settings.yaml and "
                                   f"{p['pkg']}/{p['cfg']}.py.",
                                   f"We're renaming `{p['key_old']}` to `{new}`. Update config/settings.yaml and "
                                   f"the defaults in {p['pkg']}/{p['cfg']}.py."]),
                domain="coding", setup_files=files,
                score={"expect_file": "config/settings.yaml", "expect_file_contains": f"{new}:"})


def f_code_add_function(g, rng):
    files, p = _py_project(rng)
    name, behaviour = rng.choice([("titlecase", "capitalizes the first letter of every word"),
                                  ("word_count", "returns the number of whitespace-separated words"),
                                  ("strip_digits", "removes every digit from the text"),
                                  ("collapse_spaces", "replaces runs of whitespace with a single space"),
                                  ("is_blank", "returns True when the text is empty or only whitespace"),
                                  ("initials", "returns the uppercase first letter of each word")])
    path = f"{p['pkg']}/{p['text']}.py"
    return dict(prompt=f"Add a function `{name}(text)` to {path} that {behaviour}. Keep the existing functions "
                       "as they are.",
                domain="coding", setup_files=files,
                score={"expect_file": path, "expect_file_contains": f"def {name}("})


def f_code_explain(g, rng):
    files, p = _py_project(rng)
    mod = rng.choice([p["money"], p["text"], p["cfg"]])
    return dict(prompt=rng.choice([f"Read {p['pkg']}/{mod}.py and explain what each function in it does.",
                                   f"Walk me through {p['pkg']}/{mod}.py: what does each function do?",
                                   f"I'm new to the {p['pkg']} package. What's in {mod}.py?"]),
                domain="coding", setup_files=files, score={"expect_tool": "read_file", "expect_text_min_len": 120})


TASK_DESCS = ["re-encoding the {svc} archive", "building the {svc} search index", "syncing {svc} notes to {host}",
              "downloading model weights for {svc}", "compressing {svc} logs", "rebuilding {svc} thumbnails",
              "exporting {svc} metrics to {host}", "re-indexing {svc} on {host}", "backfilling {svc} history"]


PKGS = ["app", "core", "relay", "ledger", "intake", "harbor", "quill", "lumen"]
MONEY_MODS = ["billing", "pricing", "invoices", "charges"]
TEXT_MODS = ["strings", "text", "textutil", "formatting"]
CFG_MODS = ["config", "settings", "options"]
CONSTS = ["PRECISION", "DECIMALS", "ROUND_DIGITS", "MONEY_PLACES"]
CFG_KEYS = ["retry_limit", "queue_name", "batch_size", "poll_seconds", "max_workers", "cache_dir",
            "upload_bucket", "timeout_s"]
NEW_KEYS = ["max_retries", "queue_id", "chunk_size", "interval_s", "worker_count", "cache_path",
            "bucket_name", "request_timeout"]


#: (family, kind, builder). Order inside a kind is the round-robin order.
FAMILIES: list[tuple[str, str, Callable[..., dict[str, Any]]]] = [
    ("chat_doc_summary", "chat-mode", f_chat_doc_summary),
    ("chat_note_fact", "chat-mode", f_chat_note_fact),
    ("chat_doc_explain", "chat-mode", f_chat_doc_explain),
    ("chat_followup", "chat-mode", f_chat_followup),
    ("chat_doc_plan", "chat-mode", f_chat_doc_plan),
    ("chat_note_compare", "chat-mode", f_chat_note_compare),
    ("chat_doc_faq", "chat-mode", f_chat_doc_faq),
    ("chat_note_checklist", "chat-mode", f_chat_note_checklist),
    ("chat_estimate", "chat-mode", f_chat_estimate),
    ("comp_recall_write", "compacted-history", f_comp_recall_write),
    ("comp_continue_audit", "compacted-history", f_comp_continue_audit),
    ("comp_chat_recall", "compacted-history", f_comp_chat_recall),
    ("comp_ops_followup", "compacted-history", f_comp_ops_followup),
    ("err_ops_timeout", "error-recovery", f_err_ops_timeout),
    ("err_web_fetch_503", "error-recovery", f_err_web_fetch_503),
    ("err_moved_file", "error-recovery", f_err_moved_file),
    ("err_wiki_busy", "error-recovery", f_err_wiki_busy),
    ("err_cron_locked", "error-recovery", f_err_cron_locked),
    ("err_memory_locked", "error-recovery", f_err_memory_locked),
    ("err_task_output_rotated", "error-recovery", f_err_task_output_rotated),
    ("err_search_down", "error-recovery", f_err_search_down),
    ("ag_memory_add", "agentic", f_ag_memory_add),
    ("ag_ops_health", "agentic", f_ag_ops_health),
    ("ag_wiki_lookup", "agentic", f_ag_wiki_lookup),
    ("ag_cron_create", "agentic", f_ag_cron_create),
    ("ag_research", "agentic", f_ag_research),
    ("ag_memory_recall", "agentic", f_ag_memory_recall),
    ("ag_task_status", "agentic", f_ag_task_status),
    ("ag_cron_review", "agentic", f_ag_cron_review),
    ("ag_notes_actions", "agentic", f_ag_notes_actions),
    ("ag_ops_resources", "agentic", f_ag_ops_resources),
    ("ag_memory_update", "agentic", f_ag_memory_update),
    ("ag_sentinel", "agentic", f_ag_sentinel),
    ("ag_cron_delete", "agentic", f_ag_cron_delete),
    ("ag_task_stop", "agentic", f_ag_task_stop),
    ("ag_todo", "agentic", f_ag_todo),
    ("ag_message", "agentic", f_ag_message),
    ("code_find_def", "coding", f_code_find_def),
    ("code_fix_constant", "coding", f_code_fix_constant),
    ("code_rename_key", "coding", f_code_rename_key),
    ("code_add_function", "coding", f_code_add_function),
    ("code_explain", "coding", f_code_explain),
]
FAMILY_KIND = {f: k for f, k, _ in FAMILIES}
FAMILY_FN = {f: fn for f, _, fn in FAMILIES}


class Generator:
    """Deterministic tasks: ``make(family, idx)`` is a pure function of the seed,
    the family, the index and the source sections."""

    def __init__(self, sources: Sources, seed: str = DEFAULT_SEED) -> None:
        self.sources = sources
        self.seed = seed

    def families(self, kind: str) -> list[str]:
        return [f for f, k, _ in FAMILIES if k == kind]

    def section(self, rng: random.Random) -> Section:
        return rng.choice(self.sources.sections)

    def sections(self, rng: random.Random, k: int) -> list[Section]:
        pool = self.sources.sections
        return rng.sample(pool, k) if len(pool) >= k else [rng.choice(pool) for _ in range(k)]

    def services(self, rng: random.Random, k: int) -> list[dict[str, Any]]:
        names = rng.sample(SERVICES, k)
        return [svc_record(rng, n, [x for x in SERVICES if x != n]) for n in names]

    def make(self, family: str, idx: int) -> GenTask:
        kind = FAMILY_KIND[family]
        rng = random.Random(f"{self.seed}|{family}|{idx}")
        fields = FAMILY_FN[family](self, rng)
        task = GenTask(id=f"{family}-{idx:05d}", kind=kind, family=family, **fields)
        task.validate()
        return task


# ---------------------------------------------------------------------------
# The sandbox
# ---------------------------------------------------------------------------


class SandboxRefusal(Exception):
    """A tool call the sandbox will not run."""


@dataclass
class ExecRecord:
    id: str | None
    name: str
    args: dict[str, Any]
    outcome: str  # ok | error | injected | refused | simulated
    detail: str = ""


def _limits() -> None:  # pragma: no cover - runs in the child
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))


def _prescan(cmd: str) -> None:
    if "\n" in cmd or "\r" in cmd:
        raise SandboxRefusal("multi-line commands are not available in this session")
    state: str | None = None
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if state == "'":
            if c == "'":
                state = None
        else:
            if c == "\\":
                i += 2
                continue
            if c in "`$":
                raise SandboxRefusal("command substitution and shell variables are not available here")
            if state == '"':
                if c == '"':
                    state = None
            elif c in "'\"":
                state = c
        i += 1
    if state:
        raise SandboxRefusal("unbalanced quotes")


@dataclass
class _Tok:
    """One shell token: a word (quotes removed) or an operator, with where it sits."""
    text: str
    op: bool
    start: int
    end: int
    bare: str = ""        # the characters of a word that were outside quotes and escapes
    quoted: bool = False  # any part of the word was quoted or escaped


#: Longest first, so ``2>>`` is one operator and ``&&`` is not two ``&``.
_OPERATORS = ("&>>", "<<<", "&&", "||", ";;", ">>", "<<", ">&", "<&", "<>", "&>", ">|", "|&",
              ";", "&", "|", "<", ">", "(", ")")
_OP_CHARS = frozenset(";&|<>()")
_SEP = frozenset({";", "&&", "||", "|"})
_REDIR = frozenset({">", ">>", "<", ">&", "<&", "<>", ">|"})
_OP_REFUSALS = {
    "&": "background jobs are not available in this session",
    "(": "subshells are not available in this session", ")": "subshells are not available in this session",
    "<<": "heredocs are not available in this session", "<<<": "heredocs are not available in this session",
    "&>": "write `> file 2>&1` instead of `&>` here", "&>>": "write `>> file 2>&1` instead of `&>>` here",
    "|&": "write `2>&1 |` instead of `|&` here", ";;": "`;;` is not available in this session",
}


def _lex(cmd: str) -> list[_Tok]:
    """POSIX-shell tokens of a command that passed ``_prescan``: words with their
    quoting known (a glob, brace or tilde only matters unquoted) and operators."""
    toks: list[_Tok] = []
    i, n = 0, len(cmd)
    while i < n:
        c = cmd[i]
        if c in " \t":
            i += 1
            continue
        if c in _OP_CHARS:
            op = next(o for o in _OPERATORS if cmd.startswith(o, i))
            toks.append(_Tok(op, True, i, i + len(op)))
            i += len(op)
            continue
        start, buf, bare, quoted = i, [], [], False
        while i < n and cmd[i] not in " \t" and cmd[i] not in _OP_CHARS:
            c = cmd[i]
            if c == "\\":
                buf.append(cmd[i + 1: i + 2])
                quoted, i = True, i + 2
            elif c == "'":
                j = cmd.find("'", i + 1)
                if j < 0:
                    raise SandboxRefusal("unbalanced quotes")
                buf.append(cmd[i + 1: j])
                quoted, i = True, j + 1
            elif c == '"':
                j, part = i + 1, []
                while j < n and cmd[j] != '"':
                    if cmd[j] == "\\" and j + 1 < n and cmd[j + 1] in '"\\$`':
                        part.append(cmd[j + 1])
                        j += 2
                        continue
                    part.append(cmd[j])
                    j += 1
                if j >= n:
                    raise SandboxRefusal("unbalanced quotes")
                buf.append("".join(part))
                quoted, i = True, j + 1
            else:
                buf.append(c)
                bare.append(c)
                i += 1
        toks.append(_Tok("".join(buf), False, start, i, "".join(bare), quoted))
    return toks


@dataclass
class _Grammar:
    """How one program reads its options. Kinds: ``path`` (a file it opens:
    confined), ``script`` (a pattern / program, checked per program),
    ``value`` (never a path: a separator, number or format), ``pairpath``
    (two values, the second a path), ``pair`` (two values), ``flag``,
    ``refuse``. Only ``value`` skips confinement, so it is used only where the
    option takes a non-path value on every implementation (GNU and BSD); an
    option that is not listed is a flag, and a value attached to it is
    confined from its first non-alphanumeric character. ``strict``: anything
    not listed is refused, and long options must be spelled out in full."""
    short: dict[str, str] = field(default_factory=dict)
    long: dict[str, str] = field(default_factory=dict)
    strict: bool = False
    script_first: bool = False   # the first operand is the pattern/program unless an option gave one

    def long_kind(self, name: str) -> str | None:
        if name in self.long:
            return self.long[name]
        if self.strict or len(name) < 3:
            return None
        # GNU getopt_long takes any unambiguous prefix: judge a prefix by the worst it can mean
        kinds = {k for full, k in self.long.items() if full.startswith(name)}
        for worst in ("refuse", "path", "pairpath", "script", "pair", "value", "flag"):
            if worst in kinds:
                return worst
        return None


_GREP = _Grammar(short={"e": "script", "f": "path", "m": "value", "A": "value", "B": "value",
                        "C": "value", "d": "value", "D": "value"},
                 long={"--regexp": "script", "--file": "path", "--exclude-from": "path",
                       "--include": "value", "--exclude": "value", "--exclude-dir": "value",
                       "--label": "value", "--max-count": "value", "--after-context": "value",
                       "--before-context": "value", "--context": "value"},
                 script_first=True)
_FILES0 = {"--files0-from": "refuse"}
_GRAMMARS: dict[str, _Grammar] = {
    "grep": _GREP, "egrep": _GREP, "fgrep": _GREP,
    "sed": _Grammar(short={c: "flag" for c in "nErsuz"} | {"e": "script", "l": "value"},
                    long={"--quiet": "flag", "--silent": "flag", "--regexp-extended": "flag",
                          "--separate": "flag", "--unbuffered": "flag", "--null-data": "flag",
                          "--zero-terminated": "flag", "--posix": "flag", "--debug": "flag",
                          "--sandbox": "flag", "--expression": "script", "--line-length": "value"},
                    strict=True, script_first=True),
    "awk": _Grammar(short={"F": "value", "v": "value"}, strict=True, script_first=True),
    "jq": _Grammar(short={"f": "refuse", "L": "refuse"},
                   long={"--from-file": "refuse", "--library-path": "refuse", "--arg": "pair",
                         "--argjson": "pair", "--slurpfile": "pairpath", "--rawfile": "pairpath",
                         "--indent": "value"},
                   script_first=True),
    "sort": _Grammar(short={"o": "path", "T": "path", "t": "value", "k": "value", "S": "value"},
                     long={"--output": "path", "--temporary-directory": "path", "--random-source": "path",
                           "--compress-program": "refuse", "--field-separator": "value", "--key": "value",
                           "--buffer-size": "value", **_FILES0}),
    "wc": _Grammar(long=dict(_FILES0)),
    "du": _Grammar(short={"X": "path", "d": "value", "t": "value", "B": "value"},
                   long={"--exclude-from": "path", **_FILES0}),
    "cp": _Grammar(short={"t": "path", "s": "refuse"},
                   long={"--target-directory": "path", "--symbolic-link": "refuse"}),
    "mv": _Grammar(short={"t": "path"}, long={"--target-directory": "path"}),
    "date": _Grammar(short={"f": "path", "r": "path", "s": "refuse", "d": "value", "v": "value"},
                     long={"--file": "path", "--reference": "path", "--set": "refuse"}),
    "touch": _Grammar(short={"r": "path", "d": "value", "t": "value", "A": "value"},
                      long={"--reference": "path"}),
    "file": _Grammar(short={"f": "refuse", "m": "path", "F": "value", "e": "value"},
                     long={"--files-from": "refuse", "--magic-file": "path"}),
    "diff": _Grammar(short={"X": "path", "x": "value", "I": "value", "L": "value", "U": "value",
                            "C": "value"},
                     long={"--from-file": "path", "--to-file": "path", "--exclude-from": "path"}),
    "md5sum": _Grammar(short={"c": "refuse"}, long={"--check": "refuse"}),
    "sha1sum": _Grammar(short={"c": "refuse"}, long={"--check": "refuse"}),
    "sha256sum": _Grammar(short={"c": "refuse"}, long={"--check": "refuse"}),
    "cut": _Grammar(short={"d": "value", "f": "value", "c": "value", "b": "value"},
                    long={"--delimiter": "value", "--fields": "value", "--characters": "value",
                          "--bytes": "value", "--output-delimiter": "value"}),
    "paste": _Grammar(short={"d": "value"}, long={"--delimiters": "value"}),
    "join": _Grammar(short={c: "value" for c in "t12joaev"}),
    "column": _Grammar(short={"s": "value", "c": "value"}, long={"--separator": "value"}),
    "head": _Grammar(short={"n": "value", "c": "value"}),
    "tail": _Grammar(short={"n": "value", "c": "value"}),
    "uniq": _Grammar(short={"f": "value", "s": "value"}),
    "nl": _Grammar(short={c: "value" for c in "sbdfhilnvw"}),
    "seq": _Grammar(short={"s": "value", "f": "value"}),
    "od": _Grammar(short={c: "value" for c in "AjNt"}),
    "fold": _Grammar(short={"w": "value"}),
    "fmt": _Grammar(short={"w": "value"}),
    "expand": _Grammar(short={"t": "value"}),
    "mkdir": _Grammar(short={"m": "value"}),
    "realpath": _Grammar(long={"--relative-to": "path", "--relative-base": "path"}),
}
_DEFAULT_GRAMMAR = _Grammar()
#: Programs whose operands are strings, never opened (``echo /etc/passwd`` reads nothing).
_STRING_OPERANDS = frozenset({"echo", "printf", "tr", "basename", "dirname", "seq", "true", "false",
                              "pwd", "date"})
_FIND_VALUE_OPTS = {"-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-wholename",
                    "-iwholename", "-lname", "-ilname", "-type", "-xtype", "-size", "-mtime", "-mmin",
                    "-atime", "-amin", "-ctime", "-cmin", "-perm", "-maxdepth", "-mindepth", "-printf",
                    "-user", "-group", "-uid", "-gid", "-links", "-inum", "-regextype", "-fstype"}
_FIND_REFUSED = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls",
                 "-files0-from"}
_SAFE_SED_CMD = re.compile(
    r"^\s*(?:(?:\d+|\$|/[^/]*/)(?:\s*,\s*(?:\d+|\$|/[^/]*/))?)?\s*!?\s*"
    r"(?:[pdq=]|s(?P<d>[^\w\s\\])(?:(?!(?P=d)).)*(?P=d)(?:(?!(?P=d)).)*(?P=d)[gIip0-9]*)\s*$")
#: awk that runs commands, reads or writes files, or reaches the environment or argument list.
_AWK_BAD = re.compile(r"\bsystem\b|\bgetline\b|\bprintf?\b[^;{}]*>|\|&|\bclose\b|\bfflush\b|ENVIRON|PROCINFO"
                      r"|\bARGV\b|\bARGC\b")
_AWK_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_AWK_REGEX = re.compile(r"(^|[(,~!{};&|=\s])(\s*)/(?:[^/\\]|\\.)*/")
_JQ_BAD = re.compile(r"\b(?:import|include)\b")


def _pathy(s: str) -> bool:
    return s.startswith(("/", "~")) or s in (".", "..") or "/" in s


def _check_word(tok: _Tok, root: Path) -> None:
    """Shell expansions a word can hide a path behind: braces (bash), a
    wildcard that can match ``.`` / ``..``, a leading tilde."""
    bare = tok.bare
    if "{" in bare and ("," in bare or ".." in bare):
        raise SandboxRefusal("brace expansion is not available in this session")
    if any(c in bare for c in "*?[") and any(
            part.startswith((".", "[")) and any(c in part for c in "*?[") for part in tok.text.split("/")):
        raise SandboxRefusal("a wildcard that can match `.` or `..` is not available in this session")
    if bare.startswith("~"):
        raise SandboxRefusal(f"{tok.text!r} is outside this session's working directory ({root})")


def _check_awk_program(text: str) -> None:
    if _AWK_BAD.search(text):
        raise SandboxRefusal("awk commands, pipes, file input/output and the environment are not available here")
    bare = _AWK_STRING.sub('""', text)
    bare = _AWK_REGEX.sub(lambda m: m.group(1) + m.group(2) + "//", bare)
    if "@" in bare:
        raise SandboxRefusal("awk directives and indirect calls are not available here")
    if re.search(r"(?<!\|)\|(?!\|)", bare):
        raise SandboxRefusal("awk pipes are not available here")


def _check_script(prog: str, script: str) -> None:
    if prog == "sed":
        if not all(_SAFE_SED_CMD.match(part) for part in script.split(";")):
            raise SandboxRefusal("only sed print, delete and substitute commands are available here")
    elif prog == "awk":
        _check_awk_program(script)
    elif prog == "jq" and _JQ_BAD.search(script):
        raise SandboxRefusal("jq modules (import / include) are not available here")


# -- the OS jail ------------------------------------------------------------------------------

JAIL_MODES = ("auto", "require", "off")
_JAIL_CACHE: dict[str, str] = {}


def _sbpl(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def jail_prefix(kind: str, root: Path, cwd: Path) -> list[str]:
    """argv that runs a command with only *root* writable, no network, and (Linux)
    nothing of the host's home or runtime dirs visible; ``[]`` for ``none``."""
    if kind == "bwrap":
        argv = [shutil.which("bwrap") or "bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/usr", "/usr"]
        for d in ("/bin", "/sbin", "/lib", "/lib64", "/lib32", "/libx32"):
            if os.path.islink(d):
                argv += ["--symlink", os.readlink(d), d]
            elif os.path.isdir(d):
                argv += ["--ro-bind", d, d]
        for f in ("/etc/alternatives", "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d",
                  "/etc/localtime"):
            argv += ["--ro-bind-try", f, f]
        return argv + ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                       "--bind", str(root), str(root), "--chdir", str(cwd)]
    if kind == "sandbox-exec":
        home = os.path.realpath(os.path.expanduser("~"))
        r = str(root)
        profile = ("(version 1)(allow default)(deny network*)"
                   f"(deny file-write* (require-all (require-not (subpath {_sbpl(r)}))"
                   ' (require-not (literal "/dev/null")) (require-not (literal "/dev/tty"))'
                   ' (require-not (literal "/dev/dtracehelper"))))'
                   f"(deny file-read* (require-all (subpath {_sbpl(home)}) (require-not (subpath {_sbpl(r)}))))"
                   "(deny signal (target others))")
        return [shutil.which("sandbox-exec") or "/usr/bin/sandbox-exec", "-p", profile]
    return []


def _shell_argv(cmd: str) -> list[str]:
    """dash where it exists (no brace expansion); otherwise sh with brace expansion off."""
    if os.path.exists("/bin/dash"):
        return ["/bin/dash", "-c", cmd]
    if "braces" not in _JAIL_CACHE:
        res = subprocess.run(["/bin/sh", "-c", "echo {a,b}"], capture_output=True, text=True, timeout=10)
        _JAIL_CACHE["braces"] = "yes" if res.stdout.strip() == "a b" else "no"
    return ["/bin/sh", "-c", ("set +B; " if _JAIL_CACHE["braces"] == "yes" else "") + cmd]


def detect_jail() -> str:
    """The OS jail this host can run sandboxed commands in: ``bwrap`` (Linux),
    ``sandbox-exec`` (macOS) or ``none``. Probed once with a real command."""
    if "kind" in _JAIL_CACHE:
        return _JAIL_CACHE["kind"]
    kind = "none"
    candidate = "bwrap" if sys.platform.startswith("linux") else "sandbox-exec" if sys.platform == "darwin" else ""
    if candidate and shutil.which(candidate):
        probe = Path(tempfile.mkdtemp(prefix="teacher-gen-jail-")).resolve()
        try:
            res = subprocess.run(jail_prefix(candidate, probe, probe) + _shell_argv("echo ok > probe && cat probe"),
                                 capture_output=True, text=True, timeout=20, cwd=str(probe))
            if res.returncode == 0 and res.stdout.strip() == "ok":
                kind = candidate
            else:
                log.warning("%s is present but cannot run a jailed command (%s)", candidate,
                            (res.stderr or "").strip()[:200])
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("%s probe failed: %s", candidate, exc)
        finally:
            shutil.rmtree(probe, ignore_errors=True)
    _JAIL_CACHE["kind"] = kind
    return kind


def resolve_jail(mode: str) -> str:
    """The jail kind for a ``--jail`` mode; ``require`` without one is an error."""
    if mode == "off":
        return "none"
    kind = detect_jail()
    if mode == "require" and kind == "none":
        raise EndpointError("--jail require: no working OS jail on this host (bwrap on Linux, sandbox-exec on macOS)")
    return kind


@dataclass
class ShellPlan:
    simulated: tuple[str, int] | None = None
    rest: str | None = None      # the shell command the simulated output is fed to
    piped: bool = False          # rest ends in a pipeline (its status is the command's)


@dataclass
class _Segment:
    words: list[_Tok] = field(default_factory=list)
    redirs: list[tuple[int, int, str]] = field(default_factory=list)   # (start, end, operator)


class SandboxEnv:
    """One task's world: fixtures, injections, the call log and the confinement."""

    def __init__(self, root: Path, task: GenTask, jail: str = "none") -> None:
        self.root = Path(root).resolve()
        self.task = task
        self.jail = jail
        self.fx: dict[str, Any] = copy.deepcopy(task.fixtures)
        self.injections = [dict(i, seen=0, fired=0) for i in task.inject]
        self.calls: collections.Counter = collections.Counter()
        self.log: list[ExecRecord] = []
        self.by_id: dict[str, ExecRecord] = {}
        self.messages: list[Any] = []
        self.outbox: list[dict[str, Any]] = []
        #: False while compaction pad turns run: the task's injections are for its own turn
        self.injections_armed = True

    # -- confinement -----------------------------------------------------------------

    def confine(self, value: str | None, base: Path | None = None) -> Path:
        base = base or self.root
        if value is None or value == "":
            return base
        if value.startswith("~"):
            raise SandboxRefusal(f"{value!r} is outside this session's working directory ({self.root})")
        p = Path(value)
        if not p.is_absolute():
            p = base / p
        try:
            rp = p.resolve()
        except (OSError, RuntimeError) as exc:
            raise SandboxRefusal(f"cannot resolve {value!r}") from exc
        if rp != self.root and self.root not in rp.parents:
            raise SandboxRefusal(f"{value!r} is outside this session's working directory ({self.root})")
        return rp

    def _confine_args(self, name: str, args: dict[str, Any]) -> None:
        for key in PATH_ARGS.get(name, ()):
            if args.get(key):
                self.confine(str(args[key]))
        if name == "glob":
            pat = str(args.get("pattern") or "")
            if pat.startswith("~") or ".." in Path(pat).parts:
                raise SandboxRefusal(f"{pat!r} is outside this session's working directory ({self.root})")
            if pat.startswith("/"):
                static = []
                for part in Path(pat).parts:
                    if any(c in part for c in "*?["):
                        break
                    static.append(part)
                self.confine(str(Path(*static)) if static else "/")
        if name == "grep":
            fg = str(args.get("file_glob") or "")
            if fg.startswith(("/", "~")) or ".." in Path(fg).parts:
                raise SandboxRefusal(f"{fg!r} is outside this session's working directory ({self.root})")

    def _injection(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if not self.injections_armed:
            return None
        blob = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        for inj in self.injections:
            if inj["tool"] not in (name, ANY_TOOL):
                continue
            if inj.get("arg_contains") and inj["arg_contains"] not in blob:
                continue
            inj["seen"] += 1
            if inj["seen"] < int(inj.get("nth", 1)):
                continue
            times = int(inj.get("times", 1))
            if times and inj["fired"] >= times:
                continue
            inj["fired"] += 1
            return inj
        return None

    @property
    def injections_fired(self) -> int:
        return sum(i["fired"] for i in self.injections)

    # -- dispatch --------------------------------------------------------------------

    async def execute(self, tool: Any, arguments: Any, context: Any) -> Any:
        from prometheus.tools.base import ToolResult

        name = tool.name
        args = arguments.model_dump(mode="json") if hasattr(arguments, "model_dump") else dict(arguments)
        self.calls[name] += 1
        tid = _CURRENT_CALL.get()

        def done(outcome: str, result: Any, detail: str = "") -> Any:
            rec = ExecRecord(tid, name, args, outcome, detail)
            self.log.append(rec)
            if tid:
                self.by_id[tid] = rec
            return result

        if name not in SANDBOX_TOOLS:
            return done("refused", ToolResult(output=f"Refused: {name} is not available in this session.",
                                              is_error=True), "not allowed")
        inj = self._injection(name, args)
        if inj is not None:
            return done("injected", ToolResult(output=inj["output"], is_error=True,
                                               metadata={"injected": inj.get("kind", "error")}),
                        str(inj.get("kind", "")))
        try:
            self._confine_args(name, args)
            if tool.impl is not None:
                result = await tool.impl(arguments, context)
            elif name == "read_file":
                path = self.confine(str(args.get("path") or ""))
                if not path.exists():
                    result = ToolResult(output=f"File not found: {path}", is_error=True)
                else:
                    result = await tool.inner.execute(arguments, context)
            else:
                result = await tool.inner.execute(arguments, context)
        except SandboxRefusal as exc:
            return done("refused", ToolResult(output=f"Refused: {exc}", is_error=True), str(exc))
        meta = getattr(result, "metadata", None) or {}
        outcome = meta.get("sandbox_outcome") or ("error" if result.is_error else "ok")
        return done(outcome, result)

    # -- bash ------------------------------------------------------------------------

    def plan_shell(self, cmd: str, cwd: Path) -> ShellPlan:
        _prescan(cmd)
        toks = _lex(cmd)
        if not toks:
            raise SandboxRefusal("empty command")
        segments: list[_Segment] = [_Segment()]
        ops: list[str] = []
        pipe_at: int | None = None
        i = 0
        while i < len(toks):
            t = toks[i]
            seg = segments[-1]
            if t.op and t.text in _SEP:
                if not seg.words:
                    raise SandboxRefusal("empty command in a pipeline")
                ops.append(t.text)
                if t.text == "|" and pipe_at is None and len(segments) == 1:
                    pipe_at = t.end
                segments.append(_Segment())
                i += 1
                continue
            if t.op and t.text in _REDIR:
                start = t.start
                prev = toks[i - 1] if i else None
                # an unquoted number written against the operator is its fd (2>&1, 2>/dev/null)
                if (prev is not None and seg.words and seg.words[-1] is prev and not prev.quoted
                        and prev.text.isdigit() and prev.end == t.start):
                    seg.words.pop()
                    start = prev.start
                if i + 1 >= len(toks) or toks[i + 1].op:
                    raise SandboxRefusal("redirect without a target")
                target = toks[i + 1]
                _check_word(target, self.root)
                if not ((t.text in (">&", "<&") and (target.text.isdigit() or target.text == "-"))
                        or target.text == "/dev/null"):
                    self.confine(target.text, cwd)
                seg.redirs.append((start, target.end, t.text))
                i += 2
                continue
            if t.op:
                raise SandboxRefusal(_OP_REFUSALS.get(t.text, f"`{t.text}` is not available in this session"))
            _check_word(t, self.root)
            seg.words.append(t)
            i += 1
        if not segments[-1].words:
            raise SandboxRefusal("command ends with an operator")
        first = segments[0]
        argv = [w.text for w in first.words]
        if argv[0] in OPS_PROGRAMS:
            if any(op != "|" for op in ops):
                raise SandboxRefusal("host commands can only be piped into text utilities here")
            sim = self.simulate_ops(argv)
            for seg in segments[1:]:
                self._check_segment([w.text for w in seg.words], cwd)
            # The simulated output goes through the command's own output redirects and the
            # rest of the pipeline exactly as written (the original text, not re-quoted
            # tokens). A host command reads no input, so its input redirects are dropped.
            outs = [cmd[a:b] for a, b, op in first.redirs if op not in ("<", "<&", "<>")]
            rest = None
            if outs or pipe_at is not None:
                rest = " ".join(["cat", *outs]) + (" |" + cmd[pipe_at:] if pipe_at is not None else "")
            return ShellPlan(simulated=sim, rest=rest, piped=pipe_at is not None)
        for seg in segments:
            self._check_segment([w.text for w in seg.words], cwd)
        return ShellPlan()

    def _check_segment(self, seg: list[str], cwd: Path) -> None:
        prog = seg[0]
        if "=" in prog:
            raise SandboxRefusal("environment assignments are not available here")
        if "/" in prog:
            raise SandboxRefusal(f"running `{prog}` is not available in this session")
        if prog in OPS_PROGRAMS:
            raise SandboxRefusal(f"`{prog}` can only be the first command of a pipeline here")
        if prog not in ALLOWED_PROGRAMS:
            raise SandboxRefusal(f"`{prog}` is not available in this session")
        args = seg[1:]
        if prog == "find":
            self._check_find(args, cwd)
            return
        g = _GRAMMARS.get(prog, _DEFAULT_GRAMMAR)
        scripts: list[str] = []
        operands: list[str] = []
        explicit_script = False
        end_of_options = False
        k = 0

        def take(kind: str, value: str | None, what: str) -> None:
            nonlocal explicit_script
            if value is None:
                raise SandboxRefusal(f"`{prog} {what}` needs a value")
            if kind == "path":
                self.confine(value, cwd)
            elif kind == "script":
                scripts.append(value)
                explicit_script = True

        while k < len(args):
            a = args[k]
            nxt = args[k + 1] if k + 1 < len(args) else None
            if end_of_options or a == "-" or not a.startswith("-"):
                operands.append(a)
                k += 1
                continue
            if a == "--":
                end_of_options = True
                k += 1
                continue
            if a.startswith("--"):
                name, eq, val = a.partition("=")
                kind = g.long_kind(name)
                if kind == "refuse" or (kind is None and g.strict):
                    raise SandboxRefusal(f"`{prog} {name}` is not available in this session")
                if kind is None or kind == "flag":
                    if eq and _pathy(val):   # an option this table does not know: its value may be a path
                        self.confine(val, cwd)
                    k += 1
                    continue
                if kind in ("pair", "pairpath"):
                    if eq or k + 2 >= len(args):
                        raise SandboxRefusal(f"`{prog} {name}` takes a name and a value")
                    if kind == "pairpath":
                        self.confine(args[k + 2], cwd)
                    k += 3
                    continue
                take(kind, val if eq else nxt, name)
                k += 1 if eq else 2
                continue
            # a cluster of short options: -n, -rn, -o FILE, -oFILE, -k2,2
            body, j, used_next = a[1:], 0, False
            while j < len(body):
                c = body[j]
                kind = g.short.get(c)
                if kind == "refuse" or (kind is None and g.strict):
                    raise SandboxRefusal(f"`{prog} -{c}` is not available in this session")
                if kind is None or kind == "flag":
                    if not c.isalnum():
                        # a value attached to an option this table does not know
                        rest = body[j:]
                        if _pathy(rest):
                            self.confine(rest, cwd)
                        break
                    j += 1
                    continue
                attached = body[j + 1:]
                take(kind, attached or nxt, "-" + c)
                used_next = not attached
                break
            k += 2 if used_next else 1
        if g.script_first and not explicit_script:
            if not operands:
                raise SandboxRefusal(f"`{prog}` needs a pattern or program here")
            scripts.append(operands.pop(0))
        for script in scripts:
            _check_script(prog, script)
        if prog in _STRING_OPERANDS:
            return
        for a in operands:
            if a == "/dev/null" or (prog == "awk" and re.match(r"[A-Za-z_]\w*=", a)):
                continue
            if _pathy(a):
                self.confine(a, cwd)

    def _check_find(self, args: list[str], cwd: Path) -> None:
        if _FIND_REFUSED & set(args):
            raise SandboxRefusal("find actions that run commands, read file lists or write files are not available here")
        skip = {k + 1 for k, a in enumerate(args) if a in _FIND_VALUE_OPTS}
        for k, a in enumerate(args):
            if k in skip:
                continue
            if a.startswith("-") and (("/" in a) or ("~" in a)):
                raise SandboxRefusal(f"`find {a}` is not available in this session")
            if _pathy(a):
                self.confine(a, cwd)

    async def run_sh(self, cmd: str, cwd: Path, stdin: str | None, timeout: float) -> tuple[str, int | None]:
        """Run a vetted command (in the OS jail when there is one). Output is read
        up to OUTPUT_LIMIT bytes per stream; past that the process group is killed
        (no unbounded buffering). Returns (text, returncode), returncode None on
        timeout."""
        tmp = self.root / ".tmp"
        tmp.mkdir(exist_ok=True)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.root), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "TMPDIR": str(tmp), "TERM": "dumb"}
        argv = jail_prefix(self.jail, self.root, cwd) + _shell_argv(cmd)
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, preexec_fn=_limits)

        def kill() -> None:
            try:
                os.killpg(proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass

        async def feed() -> None:
            if stdin is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(stdin.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                proc.stdin.close()

        async def read(stream: Any) -> bytes:
            buf = bytearray()
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    return bytes(buf)
                room = OUTPUT_LIMIT - len(buf)
                if room > 0:
                    buf += chunk[:room]
                if len(chunk) > room:
                    kill()

        try:
            out, err, _ = await asyncio.wait_for(
                asyncio.gather(read(proc.stdout), read(proc.stderr), feed()), timeout)
            await proc.wait()
        except asyncio.TimeoutError:
            kill()
            await proc.wait()
            return "", None
        except asyncio.CancelledError:
            kill()
            await proc.wait()
            raise
        text ="\n".join(p.decode("utf-8", "replace").rstrip() for p in (out, err) if p).strip()
        if len(text) > 20000:
            text = text[:20000] + "\n...[truncated]..."
        return text, proc.returncode

    async def bash(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        cmd = (a.command or "").strip()
        cwd = self.confine(a.cwd) if getattr(a, "cwd", None) else self.root
        if not cmd:
            raise SandboxRefusal("empty command")
        timeout = float(min(int(getattr(a, "timeout_seconds", 30) or 30), 30))
        plan = self.plan_shell(cmd, cwd)
        if plan.simulated is not None:
            text, rc = plan.simulated
            if plan.rest:
                shown, prc = await self.run_sh(plan.rest, cwd, text, timeout)
                if prc is None:
                    text, rc = shown, 124
                elif plan.piped or prc != 0:
                    text, rc = shown, prc
                else:  # only redirected: the host command's status, the shell's (empty) output
                    text = shown
            return ToolResult(output=text or "(no output)", is_error=rc != 0,
                              metadata={"returncode": rc, "sandbox_outcome": "simulated"})
        text, rc = await self.run_sh(cmd, cwd, None, timeout)
        if rc is None:
            return ToolResult(output=f"Command timed out after {int(timeout)} seconds", is_error=True)
        return ToolResult(output=text or "(no output)", is_error=rc != 0, metadata={"returncode": rc})

    # -- simulated host --------------------------------------------------------------

    def ops(self) -> dict[str, Any]:
        if "ops" not in self.fx:
            self.fx["ops"] = ops_world(random.Random("teacher-gen|default-ops"),
                                       [svc_record(random.Random(n), n, ["none"]) for n in SERVICES[:3]])
        return self.fx["ops"]

    def simulate_ops(self, argv: list[str]) -> tuple[str, int]:
        prog, args = argv[0], argv[1:]
        w = self.ops()
        if prog == "systemctl":
            return self._systemctl(args, w)
        if prog == "journalctl":
            return self._journalctl(args, w)
        if prog == "nvidia-smi":
            return self._nvidia(args, w)
        if prog == "df":
            rows = [f"{'Filesystem':<16}{'Size':>6}{'Used':>6}{'Avail':>7}{'Use%':>6} Mounted on"]
            for k, d in enumerate(w["disks"]):
                used = round(d["size"] * d["pct"] / 100)
                rows.append(f"{'/dev/nvme0n1p' + str(k + 1):<16}{str(d['size']) + 'G':>6}{str(used) + 'G':>6}"
                            f"{str(d['size'] - used) + 'G':>7}{str(d['pct']) + '%':>6} {d['mount']}")
            return "\n".join(rows), 0
        if prog == "free":
            m = w["mem"]
            return (f"{'':<7}{'total':>10}{'used':>10}{'free':>10}{'available':>12}\n"
                    f"{'Mem:':<7}{str(m['total']) + 'Gi':>10}{str(m['used']) + 'Gi':>10}"
                    f"{str(m['total'] - m['used'] - 2) + 'Gi':>10}{str(m['total'] - m['used']) + 'Gi':>12}\n"
                    f"{'Swap:':<7}{'8.0Gi':>10}{'0.2Gi':>10}{'7.8Gi':>10}"), 0
        if prog == "uptime":
            return f" 09:41:07 up {w['uptime_days']} days,  3:12,  1 user,  load average: 0.84, 0.91, 1.02", 0
        if prog == "ps":
            rows = ["USER         PID %CPU %MEM COMMAND"]
            for name, s in w["services"].items():
                if s["active"] == "active":
                    rows.append(f"svc      {s['pid']:>8}  2.1  1.4 {name}")
            for p in w["gpu"]["procs"]:
                rows.append(f"svc      {abs(hash(p['name'])) % 90000 + 1000:>8}  8.0  6.2 {p['name']}")
            return "\n".join(rows), 0
        if prog == "curl":
            return self._curl(args, w)
        raise SandboxRefusal(f"`{prog}` is not available in this session")

    def _unit(self, raw: str, w: dict[str, Any]) -> str:
        name = raw.removesuffix(".service")
        if name not in w["services"]:
            raise _UnitMissing(name)
        return name

    def _systemctl(self, args: list[str], w: dict[str, Any]) -> tuple[str, int]:
        words: list[str] = []
        k = 0
        while k < len(args):
            a = args[k]
            if a in _SYSTEMCTL_VALUE_OPTS:   # its value is not a verb or a unit (-n 50, -o cat)
                k += 2
                continue
            if not a.startswith("-"):
                words.append(a)
            k += 1
        if not words:
            words = ["list-units"]
        verb, units = words[0], words[1:]
        mutating = {"start", "stop", "restart", "reload", "enable", "disable", "kill", "mask", "unmask",
                    "daemon-reload", "reset-failed", "edit", "set-property", "try-restart",
                    "reload-or-restart", "isolate", "revert", "link", "preset"}
        if verb in mutating:
            raise SandboxRefusal("changing services is not available in this session (read-only host)")
        if verb in ("is-active", "is-failed"):
            out, rc = [], 0
            for u in units:
                try:
                    st = w["services"][self._unit(u, w)]["active"]
                except _UnitMissing:
                    st = "inactive"
                out.append(st)
                if (verb == "is-active") != (st == "active"):
                    rc = 3 if verb == "is-active" else 1
            return "\n".join(out), rc
        if verb == "status":
            blocks, rc = [], 0
            for u in units or list(w["services"]):
                try:
                    name = self._unit(u, w)
                except _UnitMissing as exc:
                    blocks.append(f"Unit {exc.args[0]}.service could not be found.")
                    rc = 4
                    continue
                s = w["services"][name]
                head = [f"{'●' if s['active'] == 'active' else '×'} {name}.service - {name} worker",
                        f"     Loaded: loaded (~/.config/systemd/user/{name}.service; enabled; preset: enabled)"]
                if s["active"] == "active":
                    head += [f"     Active: active (running) since Wed 2026-10-{29 - min(s['since_days'], 20):02d} "
                             f"06:12:03 UTC; {s['since_days']} days ago",
                             f"   Main PID: {s['pid']} ({name})", f"     Memory: {s['memory']}"]
                else:
                    rc = 3
                    head += ["     Active: failed (Result: exit-code) since Fri 2026-10-30 04:51:17 UTC; 4h 49min ago",
                             f"    Process: {s['pid']} ExecStart=/usr/bin/{name} (code=exited, status=1/FAILURE)"]
                tail = [f"Oct 30 04:5{k}:1{k} {w['host']} {name}[{s['pid']}]: {ln}"
                        for k, ln in enumerate(s["log"][-3:])]
                blocks.append("\n".join(head + [""] + tail))
            return "\n\n".join(blocks), rc
        if verb in ("list-units", "list-unit-files"):
            rows = ["  UNIT                         LOAD   ACTIVE SUB     DESCRIPTION"]
            only_failed = "--failed" in args or "--state=failed" in args
            for name, s in w["services"].items():
                if only_failed and s["active"] != "failed":
                    continue
                sub = "running" if s["active"] == "active" else "failed"
                rows.append(f"  {name + '.service':<29}loaded {s['active']:<6} {sub:<7} {name} worker")
            rows.append(f"\n{len(rows) - 1} loaded units listed.")
            return "\n".join(rows), 0
        raise SandboxRefusal(f"`systemctl {verb}` is not available in this session")

    def _journalctl(self, args: list[str], w: dict[str, Any]) -> tuple[str, int]:
        unit, n, k = None, None, 0
        prio_err, reverse, pattern = False, False, None
        while k < len(args):
            a = args[k]
            nxt = args[k + 1] if k + 1 < len(args) else None
            if a in ("-u", "--unit", "--user-unit", "-t", "--identifier") and nxt is not None:
                unit, k = nxt, k + 2
                continue
            if a.startswith(("--unit=", "--user-unit=", "--identifier=")):
                unit = a.split("=", 1)[1]
            elif a.startswith(("-u", "-t")) and len(a) > 2:
                unit = a[2:]
            elif a.startswith(("_SYSTEMD_UNIT=", "_SYSTEMD_USER_UNIT=", "UNIT=", "USER_UNIT=",
                               "SYSLOG_IDENTIFIER=")):
                unit = a.split("=", 1)[1]
            elif a in ("-n", "--lines") and nxt is not None and nxt.lstrip("+").isdigit():
                n, k = int(nxt.lstrip("+")), k + 2
                continue
            elif a.startswith("--lines="):
                n = int(a.split("=", 1)[1].lstrip("+") or 10) if a.split("=", 1)[1].lstrip("+").isdigit() else n
            elif a.startswith("-n") and a[2:].isdigit():
                n = int(a[2:])
            elif a in ("-p", "--priority") and nxt is not None:
                prio_err = nxt in ("err", "3", "0..3", "crit", "warning")
                k += 2
                continue
            elif a in ("-g", "--grep") and nxt is not None:
                pattern, k = nxt, k + 2
                continue
            elif a.startswith("--grep="):
                pattern = a.split("=", 1)[1]
            elif a in ("-r", "--reverse"):
                reverse = True
            elif a in ("-S", "--since", "-U", "--until", "-o", "--output", "-b", "--boot", "-D",
                       "--directory", "-M", "--machine", "--cursor", "--after-cursor") and nxt is not None \
                    and not nxt.startswith("-"):
                k += 2
                continue
            k += 1
        names = list(w["services"])
        if unit is not None:
            try:
                names = [self._unit(unit, w)]
            except _UnitMissing:
                return "-- No entries --", 0
        lines = []
        for name in names:
            s = w["services"][name]
            for j, ln in enumerate(s["log"]):
                if prio_err and "ERROR" not in ln and "FAILURE" not in ln:
                    continue
                if pattern is not None:
                    flags = 0 if any(c.isupper() for c in pattern) else re.I   # journalctl's smart case
                    try:
                        if not re.search(pattern, ln, flags):
                            continue
                    except re.error:
                        if pattern.lower() not in ln.lower():
                            continue
                lines.append(f"Oct 30 04:{40 + j:02d}:0{j % 10} {w['host']} {name}[{s['pid']}]: {ln}")
        if n is not None:
            lines = lines[-n:] if n else []
        if reverse:
            lines.reverse()
        return "\n".join(lines) or "-- No entries --", 0

    def _nvidia(self, args: list[str], w: dict[str, Any]) -> tuple[str, int]:
        g = w["gpu"]
        used = sum(p["mib"] for p in g["procs"]) + 300
        query = next((a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None)
        if query:
            vals = {"memory.used": f"{used} MiB", "memory.total": f"{g['total']} MiB",
                    "memory.free": f"{g['total'] - used} MiB", "name": g["name"],
                    "utilization.gpu": "37 %", "temperature.gpu": "61"}
            fields = [f.strip() for f in query.split(",")]
            header = ", ".join(fields)
            return header + "\n" + ", ".join(vals.get(f, "[N/A]") for f in fields), 0
        rows = ["+-----------------------------------------------------------------------------+",
                f"| GPU  Name: {g['name']:<40} Memory-Usage: {used}MiB / {g['total']}MiB |",
                "+-----------------------------------------------------------------------------+",
                "| Processes:                                                                  |",
                "|  GPU   PID   Type   Process name                              GPU Memory     |"]
        for p in g["procs"]:
            rows.append(f"|    0  {abs(hash(p['name'])) % 90000 + 1000:>5}    C   {p['name']:<40} {p['mib']:>6}MiB     |")
        rows.append("+-----------------------------------------------------------------------------+")
        return "\n".join(rows), 0

    def _curl(self, args: list[str], w: dict[str, Any]) -> tuple[str, int]:
        if any(a in ("-o", "-O", "--output", "--remote-name", "-D", "--dump-header", "-K", "--config",
                     "-T", "--upload-file", "--output-dir") or a.startswith(("--output=", "--dump-header=",
                                                                              "--config=", "--upload-file="))
               or (a.startswith(("-o", "-D", "-K", "-T")) and len(a) > 2 and not a.startswith("--"))
               for a in args):
            raise SandboxRefusal("curl file options are not available here; redirect its output with > instead")
        url = next((a for a in args if a.startswith("http")), None)
        if url is None:
            raise SandboxRefusal("curl needs an http URL here")
        parts = urlsplit(url)
        if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise SandboxRefusal("network access outside this machine is not available in this session")
        for name, s in w["services"].items():
            if s["port"] == parts.port:
                if s["active"] != "active":
                    return (f"curl: (7) Failed to connect to {parts.hostname} port {parts.port} after 0 ms: "
                            "Couldn't connect to server"), 7
                return json.dumps({"status": "ok", "service": name, "uptime_s": s["since_days"] * 86400}), 0
        return (f"curl: (7) Failed to connect to {parts.hostname} port {parts.port} after 0 ms: "
                "Couldn't connect to server"), 7

    # -- mocks -----------------------------------------------------------------------

    async def memory(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        store = self.fx.setdefault("memory", {}).setdefault(getattr(a, "target", None) or "memory", [])
        op = a.operation
        if op == "list":
            return ToolResult(output="\n".join(f"- {e}" for e in store) or "(no entries)")
        entry = (a.entry or "").strip() if isinstance(a.entry, str) else a.entry
        if op == "add":
            if not entry:
                return ToolResult(output="'entry' is required for add", is_error=True)
            store.append(str(entry))
            return ToolResult(output="added")
        if op == "remove":
            if not entry:
                return ToolResult(output="'entry' is required for remove", is_error=True)
            hit = next((e for e in store if str(entry) in e), None)
            if hit is None:
                return ToolResult(output=f"no entry found matching: {entry}")
            store.remove(hit)
            return ToolResult(output="removed")
        if op == "replace":
            old = a.old_entry
            if not old or not entry:
                return ToolResult(output="both 'old_entry' and 'entry' are required for replace", is_error=True)
            k = next((i for i, e in enumerate(store) if str(old) in e), None)
            if k is None:
                return ToolResult(output=f"no entry found matching: {old}")
            store[k] = str(entry)
            return ToolResult(output="replaced")
        return ToolResult(output=f"unknown operation: {op}", is_error=True)

    async def wiki_query(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        pages = self.fx.get("wiki")
        if pages is None:
            return ToolResult(output="Wiki not found. Run wiki_compile first to build the wiki.")
        terms = {t for t in re.findall(r"[a-z0-9-]{3,}", a.query.lower())}
        scored = sorted(((sum(t in (n + " " + c).lower() for t in terms), n) for n, c in pages.items()),
                        key=lambda x: (-x[0], x[1]))
        hits = [n for s, n in scored if s > 0][:3]
        if not hits:
            return ToolResult(output="No relevant wiki pages found for this query.")
        body = "\n\n".join(f"### {n}\n{pages[n]}" for n in hits)
        return ToolResult(output=f"[wiki_query: {len(hits)} of {len(pages)} relevant page(s)]\n\n{body}")

    async def wiki_compile(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        return ToolResult(output="Wiki compile: no new facts since the last compile.")

    async def cron_create(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult
        from prometheus.tools.builtin.cron_create import parse_natural_schedule, validate_cron_expression

        expr, parsed = a.schedule, None
        if not validate_cron_expression(expr):
            parsed = parse_natural_schedule(a.schedule)
            if parsed is None or not validate_cron_expression(parsed.cron):
                return ToolResult(output=(f"Couldn't parse schedule: {a.schedule!r}\nProvide either a standard "
                                          "5-field cron expression ('0 9 * * 1', '*/5 * * * *') or a natural "
                                          "phrase ('every Monday at 9am', 'in 30 minutes', 'tomorrow at 3pm', "
                                          "'every weekday at noon')."), is_error=True)
            expr = parsed.cron
        jobs = [j for j in self.fx.setdefault("cron", []) if j["name"] != a.name]
        jobs.append({"name": a.name, "schedule": expr, "command": a.command, "enabled": a.enabled,
                     "next_run": "-", "last_run": "never", "last_status": "-"})
        self.fx["cron"] = jobs
        lines = [f"Created cron job '{a.name}' [{expr}] ({'enabled' if a.enabled else 'disabled'})"]
        if parsed is not None and parsed.cron != a.schedule:
            lines.append(f"Interpreted {a.schedule!r} → {expr}")
        return ToolResult(output="\n".join(lines))

    async def cron_list(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        jobs = self.fx.get("cron") or []
        if not jobs:
            return ToolResult(output="No cron jobs registered. Scheduler: running")
        lines = ["Scheduler: running", f"Total jobs: {len(jobs)}", ""]
        for j in jobs:
            lines += [f"  {j['name']} [{'enabled' if j.get('enabled', True) else 'disabled'}]",
                      f"    Schedule:    {j['schedule']}", f"    Command:     {j['command']}",
                      f"    Next run:    {j.get('next_run', '-')}",
                      f"    Last run:    {j.get('last_run', 'never')} ({j.get('last_status', '-')})", ""]
        return ToolResult(output="\n".join(lines))

    async def cron_delete(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        jobs = self.fx.get("cron") or []
        if not any(j["name"] == a.name for j in jobs):
            return ToolResult(output=f"Cron job '{a.name}' not found", is_error=True)
        self.fx["cron"] = [j for j in jobs if j["name"] != a.name]
        return ToolResult(output=f"Deleted cron job '{a.name}'")

    def _task(self, tid: str) -> dict[str, Any] | None:
        return next((t for t in self.fx.get("tasks") or [] if t["id"] == tid), None)

    async def task_create(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        tid = "t" + hashlib.sha1(f"{a.description}{len(self.fx.get('tasks') or [])}".encode()).hexdigest()[:7]
        ttype = getattr(a, "type", None) or "local_bash"
        self.fx.setdefault("tasks", []).append({"id": tid, "type": ttype, "status": "running",
                                                "description": a.description, "command": a.command,
                                                "return_code": None, "output": ""})
        return ToolResult(output=f"Started managed task {tid} ({ttype}): {a.description}. You'll get a "
                                 "message when it finishes; check it with task_get.")

    async def task_list(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        tasks = [t for t in self.fx.get("tasks") or [] if not a.status or t["status"] == a.status]
        if not tasks:
            return ToolResult(output="(no tasks)")
        return ToolResult(output="\n".join(f"{t['id']}  {t['type']:<20}  {t['status']:<10}  {t['description']}"
                                           for t in tasks))

    async def task_get(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        t = self._task(a.task_id)
        if t is None:
            return ToolResult(output=f"No task found with ID: {a.task_id}", is_error=True)
        lines = [f"id:          {t['id']}", f"type:        {t['type']}", f"status:      {t['status']}",
                 f"description: {t['description']}", f"cwd:         {self.root}"]
        if t.get("command"):
            lines.append(f"command:     {t['command']}")
        if t.get("return_code") is not None:
            lines.append(f"return_code: {t['return_code']}")
        return ToolResult(output="\n".join(lines))

    async def task_stop(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        t = self._task(a.task_id)
        if t is None:
            return ToolResult(output=f"No task found with ID: {a.task_id}", is_error=True)
        if t["status"] != "running":
            return ToolResult(output=f"Task {t['id']} is not running (status: {t['status']})", is_error=True)
        t["status"] = "killed"
        return ToolResult(output=f"Stopped task {t['id']} (status: killed)")

    async def task_output(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        t = self._task(a.task_id)
        if t is None:
            return ToolResult(output=f"No task found with ID: {a.task_id}", is_error=True)
        return ToolResult(output=t.get("output") or "(no output yet)")

    async def web_search(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        results = list((self.fx.get("web") or {}).get("results") or [])
        if not results:
            return ToolResult(output="No search results found.", is_error=True)
        terms = set(re.findall(r"[a-z0-9-]{3,}", a.query.lower()))
        results.sort(key=lambda r: -sum(t in (r["title"] + " " + r["snippet"] + " " + r["url"]).lower()
                                        for t in terms))
        lines = [f"Search results for: {a.query}"]
        for k, r in enumerate(results[: max(1, min(int(a.max_results or 5), 10))], 1):
            lines += [f"{k}. {r['title']}", f"   URL: {r['url']}", f"   {r['snippet']}"]
        return ToolResult(output="\n".join(lines))

    async def web_fetch(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        pages = (self.fx.get("web") or {}).get("pages") or {}
        key = a.url.split("#")[0].rstrip("/")
        page = next((v for u, v in pages.items() if u.rstrip("/") == key), None)
        if page is None:
            return ToolResult(output="web_fetch failed: HTTP 404 Not Found", is_error=True)
        if a.max_chars:
            page = page[: int(a.max_chars)]
        return ToolResult(output=f"URL: {key}\nStatus: 200\nContent-Type: text/html\n\n{page}")

    async def message(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        platform = str(getattr(a.platform, "value", a.platform) or "").lower()
        self.outbox.append({"platform": platform, "recipient": a.recipient, "content": a.content})
        if platform == "telegram":
            return ToolResult(output="Message sent to Telegram.")
        if platform == "slack":
            return ToolResult(output=f"Message sent to Slack channel {a.recipient}." if a.recipient
                              else "Message sent to Slack webhook.")
        if platform == "discord":
            return ToolResult(output="Message sent to Discord.")
        return ToolResult(output=f"Unsupported platform: {platform}", is_error=True)

    async def sentinel_status(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        s = self.fx.get("sentinel") or {"active": True, "idle_secs": 120, "nudges": 0, "dreaming": False,
                                         "cycles": 12, "last_cycle_secs": 3600, "signals": 410,
                                         "subscribers": 6}
        lines = ["## Observer", f"- Active: {s['active']}", f"- Last activity: {s['idle_secs']}s ago",
                 f"- Pending nudges: {s['nudges']}", "\n## AutoDream Engine", f"- Dreaming: {s['dreaming']}",
                 f"- Dream cycles completed: {s['cycles']}", f"- Last cycle: {s['last_cycle_secs']}s ago",
                 "\n## Signal Bus", f"- Total signals: {s['signals']}", f"- Subscribers: {s['subscribers']}"]
        return ToolResult(output="\n".join(lines))

    async def lcm_grep(self, a: Any, ctx: Any) -> Any:
        from prometheus.tools.base import ToolResult

        terms = re.findall(r"[\w-]{3,}", a.query.lower())
        hits = []
        for k, m in enumerate(self.messages):
            text = message_text(m)
            low = text.lower()
            score = sum(t in low for t in terms)
            if score:
                pos = min((low.find(t) for t in terms if t in low), default=0)
                snippet = " ".join(text[max(0, pos - 80): pos + 160].split())
                hits.append((score, k, m.role, snippet))
        hits.sort(key=lambda x: (-x[0], x[1]))
        limit = int(getattr(a, "limit", None) or 10)
        if not hits:
            return ToolResult(output=f"No matches for: {a.query}")
        return ToolResult(output="\n".join(f"[message] id={k + 1} session=current role={role} turn={k}\n  {snip}"
                                           for _s, k, role, snip in hits[:limit]))


class _UnitMissing(Exception):
    pass


#: systemctl options whose value is the next word (never a verb or a unit).
_SYSTEMCTL_VALUE_OPTS = frozenset({
    "-n", "--lines", "-o", "--output", "-p", "--property", "-P", "-t", "--type", "--state", "-s", "--signal",
    "--kill-whom", "--kill-value", "-H", "--host", "-M", "--machine", "--root", "--what", "--job-mode",
    "--timestamp", "--preset-mode", "--message", "--image", "--drop-in", "--when", "--check-inhibitors",
    "--boot-loader-entry", "--boot-loader-menu", "--reboot-argument"})


def message_text(m: Any) -> str:
    """Text, tool calls and tool results of one message, for searching."""
    parts: list[str] = []
    for b in getattr(m, "content", None) or []:
        text = getattr(b, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(getattr(b, "content", None), str):
            parts.append(b.content)
        elif getattr(b, "name", None) and isinstance(getattr(b, "input", None), dict):
            parts.append(f"{b.name} {json.dumps(b.input, ensure_ascii=False)}")
    return " ".join(parts)


def _guarded_cls() -> type:
    cached = globals().get("_GUARDED")
    if cached is not None:
        return cached
    from prometheus.tools.base import BaseTool

    class GuardedTool(BaseTool):
        """The real tool's name, description and schema; every call goes
        through the task's SandboxEnv (log, injection, confinement, mock)."""

        def __init__(self, inner: Any, env: SandboxEnv, impl: Callable[..., Any] | None = None) -> None:
            self.inner = inner
            self.env = env
            self.impl = impl
            self.name = inner.name
            self.description = inner.description
            self.input_model = inner.input_model
            self.example_call = getattr(inner, "example_call", None)
            self.execution_timeout_seconds = getattr(inner, "execution_timeout_seconds", None)
            self.find_me = getattr(inner, "find_me", ())

        def is_read_only(self, arguments: Any) -> bool:
            return bool(self.inner.is_read_only(arguments))

        def to_api_schema(self) -> dict[str, Any]:
            return self.inner.to_api_schema()

        def to_openai_schema(self) -> dict[str, Any]:
            return self.inner.to_openai_schema()

        async def execute(self, arguments: Any, context: Any) -> Any:
            return await self.env.execute(self, arguments, context)

    globals()["_GUARDED"] = GuardedTool
    return GuardedTool


def build_sandbox_registry(env: SandboxEnv) -> Any:
    """The allow-listed tools for one task. Mocked tools keep the real class
    (schema, validation, description) with execute() replaced."""
    from prometheus.memory.hermes_memory_tool import MemoryTool
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin import (BashTool, FileEditTool, FileReadTool, FileWriteTool, GlobTool,
                                          GrepTool, LCMGrepTool, MessageTool, SentinelStatusTool,
                                          WebFetchTool, WebSearchTool, WikiCompileTool, WikiQueryTool)
    from prometheus.tools.builtin.cron_create import CronCreateTool
    from prometheus.tools.builtin.cron_delete import CronDeleteTool
    from prometheus.tools.builtin.cron_list import CronListTool
    from prometheus.tools.builtin.task_create import TaskCreateTool
    from prometheus.tools.builtin.task_get import TaskGetTool
    from prometheus.tools.builtin.task_list import TaskListTool
    from prometheus.tools.builtin.task_output import TaskOutputTool
    from prometheus.tools.builtin.task_stop import TaskStopTool
    from prometheus.tools.builtin.todo_write import TodoWriteTool
    from prometheus.tools.tool_search import ToolSearchTool

    G = _guarded_cls()
    reg = ToolRegistry()
    real = [
        (BashTool(workspace=str(env.root)), env.bash), (TaskCreateTool(), env.task_create),
        (FileReadTool(), None), (FileWriteTool(), None), (FileEditTool(), None), (GrepTool(), None),
        (GlobTool(), None), (CronCreateTool(), env.cron_create), (CronDeleteTool(), env.cron_delete),
        (CronListTool(), env.cron_list), (LCMGrepTool(), env.lcm_grep), (WebSearchTool(), env.web_search),
        (WebFetchTool(), env.web_fetch), (MessageTool(), env.message), (WikiCompileTool(), env.wiki_compile),
        (WikiQueryTool(), env.wiki_query), (SentinelStatusTool(), env.sentinel_status),
        (MemoryTool(), env.memory), (TodoWriteTool(), None), (TaskGetTool(), env.task_get),
        (TaskListTool(), env.task_list), (TaskStopTool(), env.task_stop), (TaskOutputTool(), env.task_output),
    ]
    for inner, impl in real:
        if inner.name in MOCKED_TOOLS and impl is None:
            raise AssertionError(f"{inner.name} must be mocked")
        reg.register(G(inner, env, impl))
    ts = ToolSearchTool()
    ts.set_registry(reg)
    reg.register(G(ts, env, None))
    names = {t.name for t in reg.list_tools()}
    if names != SANDBOX_TOOLS:
        raise AssertionError(f"sandbox registry drifted: {sorted(names ^ SANDBOX_TOOLS)}")
    return reg


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are Prometheus, a personal AI agent that runs on its owner's own machines. You help with "
    "questions, research, notes and memory, scheduling, keeping services healthy, and code.\n\n"
    "Be direct and accurate. Use tools when they help, and never invent tool results, file contents "
    "or command output. If a tool fails, say what failed and try a sensible alternative. When you "
    "have the answer, reply in plain prose without restating every step.\n\nToday's date is {date}."
)


@dataclass
class Settings:
    out: Path
    name: str = SOURCE_TAG
    teacher_url: str = DEFAULT_TEACHER_URL
    probe_guard_url: str | None = None
    judge_url: str | None = None
    model_name: str = DEFAULT_MODEL_NAME
    judge_model: str | None = None
    judge_seed: int = DEFAULT_JUDGE_SEED
    judge_max_chars: int = 30000
    allow_lan: bool = False
    concurrency: int = 2
    until: float | None = None
    min_task_seconds: float = 120.0
    max_tasks: int | None = None
    target_rows: int | None = None
    seed: str = DEFAULT_SEED
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    coding_cap: float = CODING_CAP
    max_tokens: int = 4096
    max_tool_iterations: int = 12
    tool_timeout: float = 60.0
    task_timeout: float = 900.0
    request_timeout: float = 300.0
    compaction_history_tokens: int = 1500
    emit_final_turns: bool = False
    max_natural_errors: int = 0
    keep_refused: bool = False
    sandbox_root: Path = DEFAULT_SANDBOX_ROOT
    #: the system prompt (``--pack``): with ``date_sentinel``, a template whose sentinel becomes today's
    #: date in each task's prompt (``system_prompt``); the fingerprint hashes the template, never a date
    base_system_prompt: str | None = None
    date_sentinel: str | None = None
    repo: Path = _REPO_DEFAULT
    max_consecutive_errors: int = 5
    progress: bool = True
    #: OS jail for sandboxed shell commands: auto (use one when the host has it), require, off
    jail: str = "auto"
    #: glob (case-insensitive) the teacher's /v1/models id must match; the judge's too
    #: unless judge_id is set. An id with the token "max" is always refused.
    teacher_id: str = DEFAULT_TEACHER_ID
    judge_id: str | None = None
    #: what /v1/models said at preflight (set by the run; every call must match it)
    teacher_served: str | None = None
    #: ORIGIN_TASKSET: tasks come only from a taskset_gen.py file (--tasks) and rows are
    #: SOURCE_TAG; anything else is a smoke run of the built-in families, rows BUILTIN_SOURCE
    task_source: str = ORIGIN_BUILTIN
    #: explicit sampling on every teacher (and summarizer) request: SAMPLING_KEYS -> value.
    #: Empty: the server's launch defaults (which a GGUF's own metadata can change).
    sampling: dict[str, float] = field(default_factory=dict)
    #: TEACHER_LOCAL (a loopback llama-server) or TEACHER_PAYG (Qwen Max through spend.py)
    teacher: str = TEACHER_LOCAL
    #: --teacher max-payg: this phase's sub-budget under the global cap (spend.py checks it
    #: against the ledger's total for teacher_gen / 2b-teacher), the reasoning budget
    #: (0: thinking off, as at runtime) and the paid client (set by the run, never saved)
    budget_usd: float | None = None
    thinking_budget: int = 0
    payg_client: Any = field(default=None, repr=False, compare=False)
    #: the pad pool's summary (``pad_pool_split``), for the report
    pad_pool: dict[str, Any] = field(default_factory=dict)

    @property
    def payg(self) -> bool:
        return self.teacher == TEACHER_PAYG

    @property
    def teacher_endpoint(self) -> str:
        if self.payg:
            raise EndpointError("--teacher max-payg has no local teacher endpoint (it calls spend.PaygClient)")
        return check_endpoint(self.probe_guard_url or self.teacher_url, allow_lan=self.allow_lan)

    @property
    def judge_is_teacher(self) -> bool:
        """No --judge-url, or one that reaches the teacher's server (or the probe
        guard) under any spelling (``same_server``): judged through the teacher's
        endpoint, so the probe guard is never bypassed. With --teacher max-payg:
        no --judge-url (Max judges itself on the paid key)."""
        if self.payg:
            return not self.judge_url
        return (not self.judge_url or same_server(self.judge_url, self.teacher_url)
                or same_server(self.judge_url, self.probe_guard_url))

    @property
    def judge_endpoint(self) -> str:
        """The teacher's own endpoint (through the probe guard, when one is set)
        when the judge is the teacher; another local server is called directly.
        With --teacher max-payg: the local judge (through the probe guard, when one
        is set), or "" when Max judges itself through the paid client."""
        if self.payg:
            if not self.judge_url:
                return ""
            return check_endpoint(self.probe_guard_url or str(self.judge_url), allow_lan=self.allow_lan)
        if self.judge_is_teacher:
            return self.teacher_endpoint
        return check_endpoint(str(self.judge_url), allow_lan=self.allow_lan)

    @property
    def judge_kind(self) -> str:
        return "self-judge" if self.judge_is_teacher else "local-judge"

    @property
    def judge_provider(self) -> str:
        """``max-payg`` when Max judges (only ever on the paid key), else ``local``."""
        return TEACHER_PAYG if self.payg and self.judge_is_teacher else "local"

    @property
    def local_judge_model(self) -> str:
        """The model name a local judge request carries (and its record names)."""
        return self.judge_model or (DEFAULT_MODEL_NAME if self.payg else self.model_name)

    @property
    def row_source(self) -> str:
        """The ``_meta.source`` of a strictly verified row."""
        if self.task_source != ORIGIN_TASKSET:
            return BUILTIN_SOURCE
        return PAYG_SOURCE if self.payg else SOURCE_TAG

    @property
    def relaxed_source(self) -> str:
        if self.task_source != ORIGIN_TASKSET:
            return BUILTIN_SOURCE
        return PAYG_RELAXED_SOURCE if self.payg else RELAXED_SOURCE

    def fingerprint(self) -> dict[str, Any]:
        return {"generator": GENERATOR_VERSION, "seed": self.seed, "model_name": self.model_name,
                "mix": {k: self.mix.get(k, 0.0) for k in KINDS}, "coding_cap": self.coding_cap,
                "judge_kind": self.judge_kind, "judge_seed": self.judge_seed,
                "emit_final_turns": self.emit_final_turns, "max_natural_errors": self.max_natural_errors,
                "keep_refused": self.keep_refused, "teacher_id": self.teacher_id,
                "teacher_served": self.teacher_served, "task_source": self.task_source,
                "sampling": dict(sorted(self.sampling.items())),
                "teacher": self.teacher, "judge_provider": self.judge_provider,
                "thinking_budget": self.thinking_budget,
                "pad": {"version": PAD_VERSION, "pool_share": PAD_POOL_SHARE, "min_turns": PAD_MIN_TURNS,
                        "max_turns": PAD_MAX_TURNS, "history_factor": PAD_HISTORY_FACTOR},
                # the pack's template (date sentinel unfilled), so a resume on a later day is the same run
                "system_prompt_sha": hashlib.sha256((self.base_system_prompt or "").encode()).hexdigest()[:16]}

    def system_prompt(self, root: Path) -> str:
        today = dt.date.today().isoformat()
        if self.base_system_prompt:
            base = (self.base_system_prompt.replace(self.date_sentinel, today) if self.date_sentinel
                    else self.base_system_prompt)
        else:
            base = DEFAULT_SYSTEM_PROMPT.format(date=today)
        return f"{base.rstrip()}\n\n# Workspace\n\nThis conversation's working directory is `{root}`."


#: Proxy settings httpx (trust_env, the provider's default) would send requests through.
PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def drop_proxy_env() -> None:
    """No model request may leave through a proxy: every endpoint is a local
    server (``check_endpoint``), but the provider's httpx client honours
    HTTP(S)_PROXY / ALL_PROXY, which would carry the task prompts to the proxy
    host. Removed from this process, and NO_PROXY set to ``*``."""
    for var in PROXY_ENV:
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"


def isolate_state_dirs(work: Path) -> None:
    """Point every Prometheus state root at *work* before anything writes, and
    drop any proxy from the environment (``drop_proxy_env``) before any request."""
    drop_proxy_env()
    work.mkdir(parents=True, exist_ok=True)
    for var, sub in (("PROMETHEUS_CONFIG_DIR", "config"), ("PROMETHEUS_DATA_DIR", "data"),
                     ("PROMETHEUS_LOGS_DIR", "logs")):
        (work / sub).mkdir(exist_ok=True)
        os.environ[var] = str(work / sub)
    os.environ["PROMETHEUS_ENV_FILE"] = str(work / "config" / "env")


# ---------------------------------------------------------------------------
# One task through the real agent loop
# ---------------------------------------------------------------------------


@dataclass
class RecordedCall:
    request: Any
    reply: Any
    stop_reason: str | None
    served_model: str | None
    input_tokens: int
    output_tokens: int
    dropped_malformed: int
    unreadable: tuple[str, ...]
    #: --teacher max-payg: the ledger's call id and what the call cost
    call_id: str = ""
    cost_usd: float = 0.0


def _sampling_provider_cls() -> type:
    """LlamaCppProvider that adds explicit sampling fields (``sampling=``) to
    every request body. The live provider sends none; a run that compares two
    servers (teacher_ab.py) must not inherit each server's own defaults."""
    cached = globals().get("_SAMPLING")
    if cached is not None:
        return cached
    from prometheus.providers.llama_cpp import LlamaCppProvider

    class SamplingProvider(LlamaCppProvider):
        def __init__(self, *args: Any, sampling: dict[str, float] | None = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            bad = set(sampling or {}) - set(SAMPLING_KEYS)
            if bad:
                raise ValueError(f"unknown sampling fields {sorted(bad)}")
            self.sampling = dict(sampling or {})

        def _build_request_payload(self, request: Any) -> dict[str, Any]:
            payload = super()._build_request_payload(request)
            payload.update(self.sampling)
            return payload

        async def _call_once(self, request: Any) -> Any:  # type: ignore[override]
            """The live provider's single attempt, refusing a cut-off stream: llama-server
            ends a failed generation mid-stream with a ``data: {"error": ...}`` event
            (a full KV cache, a compute error) that has no choices, and the live
            parser then returns the partial reply as if it were complete. A real
            completion always carries a finish_reason; one without is a server
            failure (retried by the next run), never a turn to verify or train on."""
            import httpx

            from prometheus.providers.base import ApiMessageCompleteEvent

            async for ev in super()._call_once(request):
                if isinstance(ev, ApiMessageCompleteEvent) and not ev.stop_reason:
                    raise httpx.RemoteProtocolError(
                        "llama-server ended the stream without a finish_reason (a server error "
                        "mid-stream); the partial reply is discarded")
                yield ev

    globals()["_SAMPLING"] = SamplingProvider
    return SamplingProvider


def _recording_provider_cls() -> type:
    cached = globals().get("_RECORDING")
    if cached is not None:
        return cached
    from prometheus.providers.base import ApiMessageCompleteEvent

    class RecordingProvider(_sampling_provider_cls()):  # type: ignore[misc]
        """The sampling provider, keeping every completed call (request + reply)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.calls: list[RecordedCall] = []
            self.errors: list[str] = []
            #: classify_turn_error kinds of self.errors, in order (context_overflow, bad_request, timeout, ...)
            self.error_kinds: list[str] = []

        async def stream_message(self, request: Any) -> Any:  # type: ignore[override]
            try:
                async for ev in super().stream_message(request):
                    if isinstance(ev, ApiMessageCompleteEvent):
                        self.calls.append(RecordedCall(
                            request, ev.message, ev.stop_reason, ev.served_model,
                            int(ev.usage.input_tokens or 0), int(ev.usage.output_tokens or 0),
                            int(ev.dropped_malformed or 0), tuple(ev.unreadable_tool_calls or ())))
                    yield ev
            except Exception as exc:
                from prometheus.api.turn_errors import classify_turn_error

                self.errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                self.error_kinds.append(str(classify_turn_error(exc).get("kind") or "unknown"))
                raise

    globals()["_RECORDING"] = RecordingProvider
    return RecordingProvider


# ---------------------------------------------------------------------------
# --teacher max-payg: Qwen Max on the pay-per-token key, every call through spend.py
# ---------------------------------------------------------------------------


def _spend() -> Any:
    """``spend.py`` beside this file (the one ledger, cap, STOP and key checks; the
    module name judge.py loads it under, so both share one copy)."""
    return _load_sibling("distill_spend", "spend.py")


def _artifact_policy() -> Any:
    return _load_sibling("distill_artifact_policy", "artifact_policy.py")


class PaygStop(RuntimeError):
    """A paid call was refused or cannot go on: the run stops cleanly (nothing is
    recorded for the task in flight; it reruns next time). ``kind`` is
    ``spend_stop`` (the STOP file, the $150 cap, this phase's --budget-usd, the
    ledger) or ``payg_fatal`` (the API refused the key or the account: auth, billing)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message


class PaygCallFailed(RuntimeError):
    """One paid attempt that failed (already settled in the ledger). ``kind`` is
    spend.py's (timeout, unreachable, provider_error, rate_limit, bad_request,
    context_overflow): the first four are retried by the next run, the rest are
    a rejection of the task."""

    def __init__(self, kind: str, message: str, status: int | None = None) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.status = status


#: PaidCallError kinds that end the run instead of the task: the key or the account is refused.
PAYG_FATAL_KINDS = frozenset({"auth", "billing"})
#: PaygStop kinds: the ledger refused, the API refused the key or account, --until is too close.
PAYG_STOP_KINDS = frozenset({"spend_stop", "payg_fatal", "until"})


def payg_call(client: Any, payload: dict[str, Any], deadline: float | None = None) -> Any:
    """``client.chat(payload)`` (spend.PaygClient: reserve, POST, settle), its errors as
    PaygStop / PaygCallFailed. Every payload holds only ``spend.PAYLOAD_KEYS``.

    *deadline* (``--until``, a timestamp): the call must be over ``PAYG_UNTIL_MARGIN_S``
    before it, so its timeout is cut to fit, and a call with less than ``PAYG_MIN_CALL_S``
    left is never started (``PaygStop("until")``: nothing is reserved; the run stops)."""
    spend = _spend()
    bad = sorted(set(payload) - set(spend.PAYLOAD_KEYS))
    if bad:  # a bug here, never something to send
        raise ValueError(f"paid request fields {bad} are not in spend.PAYLOAD_KEYS")
    kwargs: dict[str, Any] = {}
    if deadline is not None:
        left = deadline - PAYG_UNTIL_MARGIN_S - time.time()
        if left < max(PAYG_MIN_CALL_S, 1.0):
            raise PaygStop("until", f"--until is {max(0.0, left + PAYG_UNTIL_MARGIN_S):.0f} s away: no paid "
                                    "call starts that could still be running then")
        kwargs["timeout"] = left
    try:
        return client.chat(payload, **kwargs)
    except spend.SpendRefused as exc:  # STOP, the cap, the phase budget: nothing was sent
        raise PaygStop("spend_stop", str(exc)) from None
    except spend.PaidCallError as exc:  # settled already; messages hold neither key nor host
        if exc.kind in PAYG_FATAL_KINDS:
            raise PaygStop("payg_fatal", str(exc)) from None
        raise PaygCallFailed(exc.kind, exc.message, exc.status) from None
    except spend.SpendError as exc:  # the ledger, the price tier: nothing more can be spent safely
        raise PaygStop("spend_stop", str(exc)) from None


def _payg_provider_cls() -> type:
    """A ModelProvider for the agent loop and the compactor that sends each request
    through ``spend.PaygClient`` (one attempt, no stream, no retry) and records it
    like the recording provider does."""
    cached = globals().get("_PAYG")
    if cached is not None:
        return cached
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.providers.base import (
        OUTPUT_LIMIT_STOPS,
        ApiMessageCompleteEvent,
        ApiTextDeltaEvent,
        ModelProvider,
    )
    from prometheus.providers.openai_compat import _native_tool_choice
    from prometheus.providers.stub import _build_openai_messages, _parse_assistant_message, _unreadable_call_ids

    class PaygProvider(ModelProvider):
        provider_name = TEACHER_PAYG

        def __init__(self, client: Any, *, model: str, max_tokens: int, thinking_budget: int = 0,
                     sampling: dict[str, float] | None = None, deadline: float | None = None) -> None:
            self.client = client
            #: --until: every call is over before it (payg_call)
            self.deadline = deadline
            self.model = model
            self.max_tokens = int(max_tokens)
            self.thinking_budget = int(thinking_budget)
            bad = set(sampling or {}) - (set(SAMPLING_KEYS) - {"min_p"})
            if bad:
                raise ValueError(f"sampling fields {sorted(bad)} are not sent to the paid API")
            self.sampling = dict(sampling or {})
            self.calls: list[RecordedCall] = []
            self.errors: list[str] = []
            self.error_kinds: list[str] = []

        def _build_request_payload(self, request: Any) -> dict[str, Any]:
            """The chat-completions body: the loop's messages and tools as the local
            provider builds them, an explicit max_tokens (at most ``--max-tokens``),
            explicit thinking (off unless ``--thinking-budget``), no stream."""
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": _build_openai_messages(request, allow_images=False),
                "max_tokens": max(1, min(int(request.max_tokens or self.max_tokens), self.max_tokens)),
                "stream": False,
                "enable_thinking": self.thinking_budget > 0,
            }
            if self.thinking_budget > 0:
                payload["thinking_budget"] = self.thinking_budget
            if request.tools and not request.suppress_tools:
                payload["tools"] = [
                    {"type": "function", "function": {
                        "name": t.get("function", {}).get("name", t.get("name", "")),
                        "description": t.get("function", {}).get("description", t.get("description", "")),
                        "parameters": t.get("function", {}).get(
                            "parameters", t.get("input_schema", t.get("parameters", {})))}}
                    for t in request.tools]
                payload["tool_choice"] = _native_tool_choice(getattr(request, "tool_choice", "auto"))
            payload.update(self.sampling)
            return payload

        async def stream_message(self, request: Any) -> Any:  # type: ignore[override]
            payload = self._build_request_payload(request)
            try:
                r = await asyncio.to_thread(payg_call, self.client, payload, self.deadline)
            except PaygStop as exc:
                self.errors.append(str(exc)[:200])
                self.error_kinds.append(exc.kind)
                raise
            except PaygCallFailed as exc:
                self.errors.append(str(exc)[:200])
                self.error_kinds.append(exc.kind)
                raise
            if not r.finish_reason:
                self.errors.append("provider_error: the reply has no finish_reason")
                self.error_kinds.append("provider_error")
                raise PaygCallFailed("provider_error", "the reply has no finish_reason")
            # the reply without its reasoning channel: thinking is never a training target
            choice = {"message": {"content": r.message.get("content"),
                                  "tool_calls": r.message.get("tool_calls")}}
            message, dropped = _parse_assistant_message(choice, model=self.model)
            stop = r.finish_reason
            # DashScope has said "tool_calls" for a reply cut at the output limit: count
            # output at the cap as a stop at the limit (the loop's own rule)
            if stop not in OUTPUT_LIMIT_STOPS and r.output_tokens - r.reasoning_tokens >= payload["max_tokens"]:
                stop = "length"
            unreadable = tuple(_unreadable_call_ids(choice, stop))
            self.calls.append(RecordedCall(
                request, message, stop, r.served_model, int(r.input_tokens), int(r.output_tokens),
                int(dropped or 0), unreadable, call_id=r.call_id, cost_usd=float(r.usd)))
            if r.text:
                # the reply's text as one delta: the compactor's call envelope reads text from deltas
                yield ApiTextDeltaEvent(text=r.text)
            yield ApiMessageCompleteEvent(
                message=message,
                usage=UsageSnapshot(input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                                    cached_input_tokens=r.cached_input_tokens),
                stop_reason=stop, dropped_malformed=dropped, served_model=r.served_model,
                unreadable_tool_calls=unreadable)

    globals()["_PAYG"] = PaygProvider
    return PaygProvider


def teacher_provider(settings: "Settings") -> Any:
    """The provider every teacher (and summarizer) request goes through: the paid
    client for --teacher max-payg, else the local llama-server (or its probe guard)."""
    if settings.payg:
        return _payg_provider_cls()(settings.payg_client, model=PAYG_MODEL, max_tokens=settings.max_tokens,
                                    thinking_budget=settings.thinking_budget, sampling=settings.sampling,
                                    deadline=settings.until)
    return _recording_provider_cls()(base_url=settings.teacher_endpoint, timeout=settings.request_timeout,
                                     sampling=settings.sampling)


def seed_messages(seed: list[dict[str, Any]]) -> list[Any]:
    """The gym's seed format, with deterministic tool-call ids."""
    from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock

    out: list[Any] = []
    last_id: str | None = None
    n = 0
    for step in seed:
        kind, value = next(iter(step.items()))
        if kind == "user":
            out.append(ConversationMessage.from_user_text(str(value)))
        elif kind == "assistant_text":
            out.append(ConversationMessage(role="assistant", content=[TextBlock(text=str(value))]))
        elif kind == "assistant_tool_call":
            n += 1
            last_id = f"{SEED_CALL_PREFIX}{n}"
            out.append(ConversationMessage(role="assistant", content=[
                ToolUseBlock(id=last_id, name=value["name"], input=value.get("input", {}))]))
        elif kind == "tool_result":
            if last_id is None:
                raise ValueError("seed tool_result with no preceding assistant_tool_call")
            out.append(ConversationMessage(role="user", content=[ToolResultBlock(
                tool_use_id=last_id, content=str(value.get("content", "")),
                is_error=bool(value.get("is_error", False)))], provenance="user", is_trusted=True))
    return out


@dataclass
class TaskOutcome:
    task: GenTask
    run_id: str
    messages: list[Any]
    n_seed: int
    calls: list[RecordedCall]
    observed: dict[str, dict[str, Any]]
    env: SandboxEnv
    compactions: int = 0
    error: str = ""
    transport_error: bool = False
    #: a deterministic provider refusal (context overflow, another 4xx): the task is
    #: rejected and done, never retried (it would fail the same way)
    rejection: str = ""
    #: provider errors the loop recovered from (it compacted and retried, say)
    recovered_errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    provider: Any = None
    #: index into ``calls`` -> the assistant message the loop committed for that call
    #: (after extracting text-markup calls and stripping leftover markup). A call the
    #: loop retried or discarded has none.
    committed: dict[int, Any] = field(default_factory=dict)
    #: stop_reason of every AssistantTurnComplete, in order (None: the model chose to stop)
    turn_stops: list[str | None] = field(default_factory=list)
    #: compaction padding: the teacher calls (``calls[:pad_calls]``) and sandbox log
    #: entries (``env.log[:pad_log]``) of the pad turns, which are history, never rows;
    #: the descriptor rows carry as ``_meta.compaction_pad``; why padding failed
    pad_calls: int = 0
    pad_log: int = 0
    pad: dict[str, Any] | None = None
    pad_reasons: list[str] = field(default_factory=list)
    #: the summarizer's calls (the compactor's own provider): never rows
    summary_calls: list[RecordedCall] = field(default_factory=list)
    #: a paid call refused mid-task (--teacher max-payg): the run stops
    payg_stop: PaygStop | None = None

    @property
    def paid_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls) + sum(c.cost_usd for c in self.summary_calls)


_TELEMETRY: dict[str, Any] = {}


def _telemetry(settings: Settings) -> Any:
    """One telemetry database per output directory (never the live one). The
    process-wide telemetry handle is left alone: unset, nothing records there."""
    from prometheus.telemetry.tracker import ToolCallTelemetry

    work = settings.out / "work"
    key = str(work.resolve()) if work.exists() else str(work)
    if key not in _TELEMETRY:
        work.mkdir(parents=True, exist_ok=True)
        _TELEMETRY[key] = ToolCallTelemetry(db_path=work / "telemetry.db")
    return _TELEMETRY[key]


async def run_task(task: GenTask, settings: Settings, worker: int = 0) -> TaskOutcome:
    from prometheus.__main__ import create_adapter
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage
    from prometheus.engine.stream_events import AssistantTurnComplete
    from prometheus.permissions.checker import SecurityGate

    root = Path(settings.sandbox_root) / f"w{worker}" / "workspace"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    root = root.resolve()
    for rel, content in task.setup_files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    env = SandboxEnv(root, task, jail=resolve_jail(settings.jail))
    registry = build_sandbox_registry(env)
    provider = teacher_provider(settings)
    # The local route's adapter for the model name, whichever provider serves it: a Max
    # teacher (tier light, like the 27B) runs the loop exactly as the local teacher does.
    adapter = create_adapter({"provider": "llama_cpp", "model": settings.model_name}, {})
    system_prompt = settings.system_prompt(root)
    counter = {"n": 0}
    compactor = None
    summarizer = None
    if task.compaction:
        summarizer = teacher_provider(settings)
        compactor = _make_compactor(settings, system_prompt, registry, counter, chat=task.mode == "chat",
                                    provider=summarizer)
    seeded = seed_messages(task.seed)
    messages = list(seeded)
    env.messages = messages
    observed: dict[str, dict[str, Any]] = {}

    def _observe(tool_use_id: str, raw: dict[str, Any], executed: dict[str, Any]) -> None:
        observed[tool_use_id] = {"raw": raw, "executed": executed}
        _CURRENT_CALL.set(tool_use_id)

    gate = SecurityGate(workspace_root=root, mode="default")
    context = LoopContext(
        provider=provider, model=settings.model_name, system_prompt=system_prompt,
        max_tokens=settings.max_tokens, tool_registry=registry, permission_checker=gate,
        adapter=adapter, telemetry=_telemetry(settings), cwd=root,
        max_tool_iterations=settings.max_tool_iterations, tool_timeout_seconds=settings.tool_timeout,
        session_id="system", tool_call_observer=_observe, compactor=compactor,
        # The session-workspace path the daemon uses: relative paths, the tool
        # cwd and the gate's write boundary are all the sandbox.
        workspace_resolver=lambda _session: root)
    out = TaskOutcome(task=task, run_id=f"{task.id}@{uuid.uuid4().hex[:10]}", messages=messages,
                      n_seed=len(seeded), calls=provider.calls, observed=observed, env=env, provider=provider)
    if summarizer is not None:
        out.summary_calls = summarizer.calls
    t0 = time.monotonic()
    try:
        if task.pad:
            # earlier turns: other real-pattern prompts, each answered by this teacher, with
            # the compactor off and the task's injections held back for its own turn
            env.injections_armed = False
            await _pad_turns(task, settings, dataclasses.replace(context, compactor=None), out)
            env.injections_armed = True
            out.n_seed = len(messages)
            out.pad_calls = len(provider.calls)
            out.pad_log = len(env.log)
        if not out.pad_reasons:
            messages.append(ConversationMessage.from_user_text(task.prompt))
            async for item in run_loop(context, messages, mode="chat" if task.mode == "chat" else "agent"):
                ev = item[0] if isinstance(item, tuple) else item
                if isinstance(ev, AssistantTurnComplete):
                    out.turn_stops.append(ev.stop_reason)
                    # The loop commits a model reply right after the call that produced it
                    # (no other teacher call in between); an error turn carries a stop reason.
                    if ev.stop_reason is None and provider.calls:
                        out.committed[len(provider.calls) - 1] = ev.message
    except asyncio.CancelledError:
        raise
    except PaygStop as exc:  # the spend ledger refused: the run stops, this task reruns later
        out.payg_stop = exc
        out.error = f"PaygStop: {str(exc)[:300]}"
    except Exception as exc:  # harness-level: recorded, not raised
        out.error = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        out.duration_s = time.monotonic() - t0
        out.compactions = counter["n"]
    if out.payg_stop is None:
        # a refused paid call the loop or the compactor caught and did not re-raise
        for p in (provider, summarizer):
            for kind, err in zip(getattr(p, "error_kinds", []), getattr(p, "errors", [])):
                if kind in PAYG_STOP_KINDS and out.payg_stop is None:
                    out.payg_stop = PaygStop(kind, err)
    classify_provider_errors(out, provider)
    classify_summarizer_errors(out, summarizer)
    return out


def _turn_events(out: TaskOutcome, msgs: list[Any]) -> list[dict[str, Any]]:
    """The tool calls in *msgs* with what the sandbox did with each (``tool_events``'s fields)."""
    from prometheus.engine.messages import ToolResultBlock, ToolUseBlock

    events: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for m in msgs:
        for b in m.content:
            if isinstance(b, ToolUseBlock) and not b.id.startswith(SEED_CALL_PREFIX):
                ev = {"id": b.id, "name": b.name, "input": dict(b.input), "is_error": None, "result": ""}
                events.append(ev)
                by_id[b.id] = ev
            elif isinstance(b, ToolResultBlock) and b.tool_use_id in by_id:
                by_id[b.tool_use_id]["is_error"] = bool(b.is_error)
                by_id[b.tool_use_id]["result"] = b.content if isinstance(b.content, str) else str(b.content)
    for ev in events:
        rec = out.env.by_id.get(ev["id"])
        obs = out.observed.get(ev["id"])
        ev["repaired"] = bool(obs) and (obs["raw"] != obs["executed"])
        if rec is not None and rec.outcome in ("injected", "refused", "simulated"):
            ev["outcome"] = rec.outcome
        elif ev["is_error"] is None:
            ev["outcome"] = "unanswered"
        elif ev["is_error"]:
            ev["outcome"] = "error"
        else:
            ev["outcome"] = "ok"
    return events


def _last_text(msgs: list[Any]) -> str:
    """The final prose reply in *msgs* ("" when the last assistant turn called a tool)."""
    from prometheus.engine.messages import TextBlock, ToolUseBlock

    for m in reversed(msgs):
        if m.role != "assistant":
            continue
        if any(isinstance(b, ToolUseBlock) for b in m.content):
            return ""
        text = "".join(b.text for b in m.content if isinstance(b, TextBlock)).strip()
        if text:
            return text
    return ""


async def _pad_turns(task: GenTask, settings: Settings, context: Any, out: TaskOutcome) -> None:
    """Run the task's pad turns (``GenTask.pad``) through the real loop, in order,
    until the history holds ``PAD_HISTORY_FACTOR`` x ``--compaction-history-tokens``
    (and at least ``PAD_MIN_TURNS`` turns) or the candidates run out. Each turn is
    a real-pattern prompt sent as the user's turn, answered by the same teacher in
    its own row's mode. A turn that did not end in a reply, or whose calls were
    refused, repaired, dropped or cut off, stops the padding and rejects the task
    (``out.pad_reasons``): that history would teach a sandbox or harness artifact.
    Natural tool errors are kept (they are real tool results) and counted."""
    from prometheus.context.token_estimation import estimate_message_tokens
    from prometheus.engine.agent_loop import run_loop
    from prometheus.engine.messages import ConversationMessage
    from prometheus.engine.stream_events import AssistantTurnComplete

    messages = out.messages
    target = PAD_HISTORY_FACTOR * settings.compaction_history_tokens
    turns: list[dict[str, Any]] = []
    for cand in task.pad:
        first_msg, first_call = len(messages), len(out.calls)
        messages.append(ConversationMessage.from_user_text(cand["prompt"]))
        stops: list[str | None] = []
        async for item in run_loop(context, messages, mode="chat" if cand["mode"] == "chat" else "agent"):
            ev = item[0] if isinstance(item, tuple) else item
            if isinstance(ev, AssistantTurnComplete):
                stops.append(ev.stop_reason)
        msgs = messages[first_msg:]
        calls = out.calls[first_call:]
        events = _turn_events(out, msgs)
        outcomes = collections.Counter(e["outcome"] for e in events)
        problem = ""
        if not calls or not stops or stops[-1] is not None or not _last_text(msgs):
            problem = "pad_turn_incomplete"
        elif outcomes["refused"]:
            problem = "pad_sandbox_refusal"
        elif (any(e["repaired"] for e in events) or any(c.dropped_malformed or c.unreadable for c in calls)
              or any(c.stop_reason in ("length", "max_tokens") for c in calls)
              or any(getattr(m, "provenance", "") == ORCHESTRATOR for m in msgs)):
            problem = "pad_turn_unclean"
        turns.append({"id": cand["id"], "task_group": cand["task_group"], "mode": cand["mode"],
                      "prompt_hash": h(norm(cand["prompt"])), "teacher_calls": len(calls),
                      "tool_calls": len(events), "tool_errors": outcomes["error"] + outcomes["unanswered"],
                      "reads": sum(1 for e in events if e["name"] in ("read_file", "grep", "glob", "bash")
                                   and e["outcome"] == "ok")})
        if problem:
            out.pad_reasons.append(problem)
            break
        if len(turns) >= PAD_MIN_TURNS and sum(estimate_message_tokens(m) for m in messages) >= target:
            break
    out.pad = {
        "version": PAD_VERSION,
        # the user turns are these real-pattern prompts, verbatim; every reply is this
        # teacher's, in this run; a read returns the repo's own docs (the sandbox's files)
        "turns": len(turns), "task_ids": [t["id"] for t in turns],
        "task_groups": [t["task_group"] for t in turns], "prompt_hashes": [t["prompt_hash"] for t in turns],
        "modes": [t["mode"] for t in turns],
        "teacher_calls": sum(t["teacher_calls"] for t in turns),
        "tool_calls": sum(t["tool_calls"] for t in turns), "tool_errors": sum(t["tool_errors"] for t in turns),
        "reads": sum(t["reads"] for t in turns), "repo_docs": sorted(task.setup_files),
        "pool": "reserved" if settings.pad_pool.get("reserved") else "shared",
        "answered_by": "teacher", "history_tokens": sum(estimate_message_tokens(m) for m in messages),
    }


#: Provider failures worth retrying later: the server was down, slow or failing.
#: Everything else the provider refuses (context overflow, a malformed request) is
#: deterministic: the task is rejected and recorded as done.
RETRY_KINDS = frozenset({"timeout", "unreachable", "provider_error", "rate_limit", "auth", "billing",
                             "unknown"})


def classify_provider_errors(out: TaskOutcome, provider: Any) -> None:
    if not provider.errors:
        return
    kinds = list(getattr(provider, "error_kinds", []) or [])
    last = kinds[-1] if kinds else "unknown"
    completed = not out.error and bool(out.turn_stops) and out.turn_stops[-1] is None
    if completed:
        # the loop recovered (compacted and retried, say) and the turn ended normally
        out.recovered_errors = [f"{k}: {e}" for k, e in zip(kinds, provider.errors)]
        return
    if last in RETRY_KINDS:
        out.transport_error = True
        out.error = out.error or provider.errors[-1]
        return
    out.rejection = "context_overflow" if last == "context_overflow" else f"provider_{last}"


def classify_summarizer_errors(out: TaskOutcome, summarizer: Any) -> None:
    """The compactor swallows its summarizer's errors and sends the history uncompacted, so a
    compaction task whose summary request failed for a passing reason (a 429, a timeout, a 5xx,
    the server down: ``RETRY_KINDS``) and that never compacted would be rejected for good as
    ``compaction_not_exercised``. It is a transport error instead (retried by the next run). A
    summary request refused for a lasting reason (too long, a bad request) still rejects."""
    if summarizer is None or out.compactions or out.transport_error or out.payg_stop is not None:
        return
    errors = list(getattr(summarizer, "errors", []) or [])
    if not errors:
        return
    kinds = list(getattr(summarizer, "error_kinds", []) or [])
    if (kinds[-1] if kinds else "unknown") in RETRY_KINDS:
        out.transport_error = True
        out.error = out.error or f"summarizer: {errors[-1]}"


def _make_compactor(settings: Settings, system_prompt: str, registry: Any, counter: dict[str, int],
                    *, chat: bool = False, provider: Any = None) -> Any:
    """The real compactor with a window of the fixed cost (system prompt and,
    unless chat-mode, the tool schemas) + ``compaction_history_tokens`` + the
    output cap, so a long history has to be compacted. Its summarizer is the
    teacher (*provider*, default ``teacher_provider``): the summaries are the
    teacher's own output, and its calls are recorded but never rows."""
    from prometheus.context.compactor import ContextCompactor
    from prometheus.context.token_estimation import estimate_tokens

    tools_chars = 0 if chat else len(str(registry.to_api_schema()))
    fixed = estimate_tokens(system_prompt) + tools_chars // 4
    comp = ContextCompactor(
        provider=provider if provider is not None else teacher_provider(settings),
        model=settings.model_name,
        effective_limit=fixed + settings.compaction_history_tokens + settings.max_tokens,
        threshold_pct=0.85, reserve_tokens=settings.max_tokens, protect_recent_turns=2,
        max_summary_tokens=512)
    orig = comp.apply

    async def counting_apply(messages: list[Any], **kwargs: Any) -> list[Any]:
        view = await orig(messages, **kwargs)
        if view is not messages:
            counter["n"] += 1
        return view

    comp.apply = counting_apply  # type: ignore[method-assign]
    return comp


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def tool_events(out: TaskOutcome) -> list[dict[str, Any]]:
    """Every tool call the loop made in the task's own turn (seeded calls and pad
    turns excluded), with what the sandbox did with it."""
    return _turn_events(out, out.messages[out.n_seed:])


def final_text(out: TaskOutcome) -> str:
    return _last_text(out.messages[out.n_seed:])


def deterministic_checks(out: TaskOutcome, settings: Settings) -> tuple[bool, list[str], dict[str, Any]]:
    from prometheus.gym.scoring import EXECUTION, RunTranscript, score

    task = out.task
    reasons: list[str] = []
    events = tool_events(out)
    counts = collections.Counter(ev["outcome"] for ev in events)
    natural = counts["error"] + counts["unanswered"]
    repaired = sum(ev["repaired"] for ev in events)
    # the task's own turn: pad turns are history (checked when they ran, in _pad_turns)
    calls = out.calls[out.pad_calls:]
    dropped = sum(c.dropped_malformed for c in calls)
    feedback = sum(1 for m in out.messages[out.n_seed:] if getattr(m, "provenance", "") == ORCHESTRATOR)
    text = final_text(out)
    record: dict[str, Any] = {
        "tool_calls": len(events), "tool_errors": natural, "injected_failures": counts["injected"],
        "sandbox_refusals": counts["refused"], "simulated_calls": counts["simulated"],
        "repaired_calls": repaired, "dropped_malformed": dropped, "orchestrator_feedback": feedback,
        "teacher_calls": len(calls), "compactions": out.compactions,
    }
    if out.pad is not None:
        record["compaction_pad"] = out.pad
    if out.pad_reasons:
        reasons += out.pad_reasons
    if out.error:
        reasons.append("harness_error")
    if not calls:
        reasons.append("no_teacher_calls")
    if not text:
        reasons.append("no_final_text")
    elif "circuit breaker tripped" in text.lower():
        reasons.append("breaker_tripped")
    if natural > settings.max_natural_errors:
        reasons.append("tool_error")
    if counts["refused"] and not settings.keep_refused:
        reasons.append("sandbox_refusal")
    if repaired:
        reasons.append("adapter_repair")
    if dropped:
        reasons.append("dropped_malformed")
    if feedback:
        reasons.append("orchestrator_feedback")
    if any(c.stop_reason in ("length", "max_tokens") for c in calls):
        reasons.append("output_limit")
    # Every reply (pad turns too) must come from the teacher the preflight checked:
    # llama-server reports either its own id or the requested model name, never a Max
    # model; the paid API names the Max model it served (PAYG_TEACHER_ID), nothing else.
    served = sorted({c.served_model for c in out.calls if c.served_model})
    record["served_models"] = served
    if settings.payg:
        if any(not fnmatch.fnmatchcase(m.lower(), settings.teacher_id.lower()) or not has_max_token(m)
               for m in served):
            reasons.append("served_model_mismatch")
    else:
        allowed = {settings.model_name} | ({settings.teacher_served} if settings.teacher_served else set())
        if any(has_max_token(m) or (settings.teacher_served and m not in allowed) for m in served):
            reasons.append("served_model_mismatch")
    if task.mode == "chat" and events:
        reasons.append("chat_mode_tool_call")
    recovered: bool | None = None
    # Any task carrying an injection (a taskset row with inject_errors and
    # force_compaction is a compacted-history task) must see it fire: a flag
    # the run never exercised was not honoured.
    if task.kind == "error-recovery" or task.inject:
        if not out.env.injections_fired:
            reasons.append("injection_not_fired")
        elif task.expect_recovery:
            first = next((k for k, ev in enumerate(events) if ev["outcome"] == "injected"), len(events))
            recovered = any(ev["outcome"] in ("ok", "simulated") for ev in events[first + 1:])
            if not recovered:
                reasons.append("not_recovered")
        else:
            recovered = bool(text)
    record["recovered"] = recovered
    if task.compaction:
        in_request = any(SUMMARY_MARKER in (getattr(m, "text", "") or "")
                         for c in calls for m in c.request.messages)
        record["compaction"] = {"events": out.compactions, "summary_in_request": in_request}
        if not out.compactions or not in_request:
            reasons.append("compaction_not_exercised")
    pred: dict[str, Any] = {"checked": sorted(task.score) + sorted(f"answer.{k}" for k in task.answer),
                            "pass": True, "reasons": []}
    why: list[str] = []
    if task.score:
        transcript = RunTranscript.from_messages(out.messages[out.n_seed:], out.observed)
        _ok, why = score(task.score, transcript, out.env.root, view=EXECUTION)
    why = list(why) + answer_failures(task.answer, text)
    if why:
        pred.update({"pass": False, "reasons": why[:5]})
        reasons.append("predicate")
    record["predicates"] = pred
    record["tool_success"] = natural == 0 and not counts["refused"] and not repaired
    # Every tool went through the sandbox; what the mocks "did" stayed in memory, and
    # shell commands ran in the workspace (inside the OS jail named here, if any).
    record["sandboxed"] = True
    record["side_effects"] = False
    record["jail"] = out.env.jail
    record["mocked_effects"] = {"messages": len(out.env.outbox),
                                "writes": sum(1 for r in out.env.log[out.pad_log:] if r.outcome == "ok" and r.name in (
                                    "memory", "cron_create", "cron_delete", "task_create", "task_stop"))}
    if settings.payg:
        record["teacher"] = TEACHER_PAYG
        record["max_derived"] = True
    return not reasons, reasons, record


JUDGE_SYSTEM = (
    "You are a strict grader of an AI agent's work. You see the task, the transcript (tool calls "
    "with their results) and the agent's final reply. Judge only what the transcript shows.\n\n"
    "Fail it if ANY of these hold:\n"
    "1. The task was not actually done, or, when it could not be done, the agent did not say so plainly.\n"
    "2. A tool call used the wrong tool, invalid or invented arguments, or a pointless order of steps.\n"
    "3. The reply states a fact, file content, command output or result that no tool result or "
    "provided material supports.\n"
    "4. The reply claims an action succeeded when its tool result shows it failed.\n"
    "5. After a tool failure the agent ignored it instead of retrying, working around it or reporting it.\n"
    "6. The reply contradicts earlier parts of the conversation, including compacted summaries.\n"
    "7. The reply is padded, evasive, or does not answer what was asked.\n"
    "8. The agent changed something it was told not to change, or did more than was asked.\n\n"
    "Answer with one JSON object and nothing else:\n"
    '{"verdict": "pass" or "fail", "completed": true/false, "tools_ok": true/false, '
    '"grounded": true/false, "recovery_ok": true/false/null, "reason": "<one sentence>"}'
)


def judge_transcript(out: TaskOutcome, max_chars: int = 30000, result_chars: int = 1500) -> str:
    from prometheus.engine.messages import TextBlock, ToolResultBlock, ToolUseBlock

    def render(msgs: list[Any]) -> list[str]:
        lines = []
        for m in msgs:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    tag = "assistant" if m.role == "assistant" else (
                        "summary" if getattr(m, "provenance", "") == "compactor" else "user")
                    lines.append(f"[{tag}] {b.text.strip()}")
                elif isinstance(b, ToolUseBlock):
                    lines.append(f"[tool call] {b.name}({json.dumps(b.input, ensure_ascii=False)[:600]})")
                elif isinstance(b, ToolResultBlock):
                    body = b.content if isinstance(b.content, str) else str(b.content)
                    if len(body) > result_chars:
                        body = body[:result_chars] + " ...[cut]"
                    lines.append(f"[tool result {'ERROR' if b.is_error else 'ok'}] {body}")
        return lines

    earlier = "\n".join(render(out.messages[: out.n_seed]))
    current = "\n".join(render(out.messages[out.n_seed:]))
    budget = max(2000, max_chars - len(current))
    if len(earlier) > budget:
        earlier = earlier[: budget // 2] + "\n...[earlier conversation cut]...\n" + earlier[-budget // 2:]
    if len(current) > max_chars:
        current = current[: max_chars // 3] + "\n...[cut]...\n" + current[-2 * max_chars // 3:]
    parts = [f"TASK KIND: {out.task.kind}" + (" (the context window was compacted mid-session)"
                                               if out.task.compaction else ""),
             f"TASK (the user's last request):\n{out.task.prompt[:4000]}"]
    if earlier:
        parts.append(f"EARLIER CONVERSATION:\n{earlier}")
    parts.append(f"THIS TURN:\n{current}")
    parts.append(f"FINAL REPLY:\n{final_text(out)}")
    return "\n\n".join(parts)


def parse_verdict(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("verdict") not in ("pass", "fail"):
        return None
    return obj


def judge_payload(transcript: str, settings: Settings) -> dict[str, Any]:
    """The local judge request for one rendered transcript (``judge_transcript``)."""
    return {
        "model": settings.local_judge_model,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": transcript}],
        "temperature": 0, "seed": settings.judge_seed, "max_tokens": JUDGE_MAX_TOKENS, "stream": False,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
    }


def payg_judge_payload(transcript: str, settings: Settings) -> dict[str, Any]:
    """The Max self-judge request (--teacher max-payg without --judge-url): the same
    rubric at temperature 0 and its own seed, thinking off, output bounded; only
    ``spend.PAYLOAD_KEYS``."""
    return {
        "model": PAYG_MODEL,
        "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": transcript}],
        "temperature": 0, "seed": settings.judge_seed, "max_tokens": JUDGE_MAX_TOKENS, "stream": False,
        "response_format": {"type": "json_object"}, "enable_thinking": False,
    }


async def judge(out: TaskOutcome, settings: Settings) -> dict[str, Any]:
    """The judge's record. ``pass`` needs verdict pass AND completed,
    tools_ok and grounded true AND recovery_ok not false."""
    return await judge_text(judge_transcript(out, settings.judge_max_chars), settings)


def _verdict_record(record: dict[str, Any], obj: dict[str, Any] | None) -> dict[str, Any]:
    if obj is None:
        record.update({"verdict": "fail", "pass": False, "reason": "unparseable judge reply"})
        return record
    ok = (obj.get("verdict") == "pass" and obj.get("completed") is True and obj.get("tools_ok") is True
          and obj.get("grounded") is True and obj.get("recovery_ok") is not False)
    record.update({k: obj.get(k) for k in ("verdict", "completed", "tools_ok", "grounded", "recovery_ok")})
    record["reason"] = str(obj.get("reason") or "")[:300]
    record["pass"] = ok
    return record


async def judge_text(transcript: str, settings: Settings) -> dict[str, Any]:
    """``judge`` on an already rendered transcript (teacher_ab.py judges stored
    traces this way, all on one judge endpoint). A local judge raises httpx
    errors; the Max self-judge (--teacher max-payg, no --judge-url) raises
    PaygStop / PaygCallFailed. The record names the judge the way build.py
    reads it: ``provider`` ``local`` (``judge_key`` ``none``, its ``endpoint``)
    or ``max-payg`` (``judge_key`` ``payg``, the ledger's ``call_ids``)."""
    if settings.payg and settings.judge_is_teacher:
        return await _payg_judge_text(transcript, settings)
    import httpx

    payload = judge_payload(transcript, settings)
    record: dict[str, Any] = {"kind": settings.judge_kind, "model": payload["model"], "temperature": 0,
                              "seed": settings.judge_seed, "provider": "local", "judge_key": "none"}
    if settings.payg:
        record["endpoint"] = settings.judge_endpoint  # a loopback URL: the local 27B judge
    obj = None
    async with httpx.AsyncClient(timeout=settings.request_timeout, trust_env=False) as client:
        for _attempt in range(2):
            resp = await client.post(f"{settings.judge_endpoint}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            body = resp.json()
            record["served_model"] = body.get("model")
            content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            obj = parse_verdict(content)
            if obj is not None:
                break
    return _verdict_record(record, obj)


async def _payg_judge_text(transcript: str, settings: Settings) -> dict[str, Any]:
    """Max judging its own teacher rows on the pay-per-token key, through
    spend.PaygClient (same ledger, phase and --budget-usd as the teacher calls),
    recorded as such: build.py accepts it only as the pay-per-token Max judge."""
    payload = payg_judge_payload(transcript, settings)
    record: dict[str, Any] = {"kind": "self-judge", "model": PAYG_MODEL, "temperature": 0,
                              "seed": settings.judge_seed, "provider": TEACHER_PAYG, "judge_key": "payg",
                              "call_ids": [], "cost_usd": 0.0}
    obj = None
    for _attempt in range(2):
        r = await asyncio.to_thread(payg_call, settings.payg_client, payload, settings.until)
        record["call_ids"].append(r.call_id)
        record["cost_usd"] = round(record["cost_usd"] + float(r.usd), 6)
        record["served_model"] = r.served_model
        obj = parse_verdict(r.text)
        if obj is not None:
            break
    return _verdict_record(record, obj)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


_MARKUP = re.compile(r"</?tool_call\b|<function=|<\|tool_call", re.I)


def _blocks(message: Any) -> str:
    return json.dumps([b.model_dump(mode="json") if hasattr(b, "model_dump") else repr(b)
                       for b in getattr(message, "content", None) or []], sort_keys=True, default=str)


def rows_for(out: TaskOutcome, settings: Settings, verification: dict[str, Any]) -> list[dict[str, Any]]:
    """Golden-trace rows, one per teacher call that taught something sound."""
    from prometheus.engine.messages import ToolResultBlock, ToolUseBlock
    from prometheus.providers.base import ApiMessageRequest
    from prometheus.providers.stub import _build_openai_messages

    task = out.task
    events = {ev["id"]: ev for ev in tool_events(out)}
    provider = out.provider
    rows: list[dict[str, Any]] = []
    ts = time.time()
    # The train source only for a task built from a validated real-pattern taskset row
    # in a --tasks run; a built-in (hand-written) task is a smoke row whatever the run.
    real = (settings.task_source == ORIGIN_TASKSET and task.origin == ORIGIN_TASKSET
            and task.meta.get("provenance") == TASKSET_PROVENANCE)
    if not real:
        source = BUILTIN_SOURCE
    else:
        source = settings.row_source if verification.get("tool_success") else settings.relaxed_source
    for step, call in enumerate(out.calls):
        if step < out.pad_calls:
            continue  # a pad turn: history this row's context holds, never a target
        # The target is the message the loop acted on and committed, not the raw
        # reply: the loop pulls text-markup calls out of the reply (new ids) and strips
        # leftover <tool_call> text from dual-emit replies before committing. A reply
        # the loop retried or discarded was never acted on: no row.
        reply = out.committed.get(step)
        if reply is None:
            continue
        uses = [b for b in reply.content if isinstance(b, ToolUseBlock)]
        if call.stop_reason in ("length", "max_tokens") or call.unreadable:
            continue
        if not uses and _MARKUP.search(reply.text or ""):
            continue  # call markup the loop could neither run nor strip: never a target
        if uses:
            if task.mode == "chat":
                continue
            evs = [events.get(u.id) for u in uses]
            if any(ev is None or ev["repaired"] or ev["outcome"] not in ("ok", "simulated", "injected")
                   for ev in evs):
                continue
        elif task.mode != "chat" and not settings.emit_final_turns:
            continue
        payload = provider._build_request_payload(call.request)
        wire = [dict(m) for m in payload["messages"]]
        errors = {b.tool_use_id for m in call.request.messages for b in m.content
                  if isinstance(b, ToolResultBlock) and b.is_error}
        for m in wire:
            if m.get("role") == "tool" and m.get("tool_call_id") in errors:
                m["is_error"] = True
        target = _build_openai_messages(ApiMessageRequest(model=settings.model_name, messages=[reply]))[-1]
        target.setdefault("content", "")
        last_user = max((k for k, m in enumerate(wire) if m.get("role") == "user"), default=0)
        injected_target = any((events.get(u.id) or {}).get("outcome") == "injected" for u in uses)
        meta = {
            "model": settings.model_name, "served_model": call.served_model,
            "teacher_id": settings.teacher_served,
            # A row verified only under a relaxation (natural tool errors, refusals) is
            # never a train row: build.py's strict policy would drop it anyway.
            "source": source,
            "tool_name": uses[0].name if uses else None, "timestamp": ts,
            "session_id": f"gym:{task.id}", "category": task.domain, "task_category": task.kind,
            "task_family": task.family, "task_id": task.id, "gen_task_id": task.id, "task_hash": task.task_hash,
            "run_id": out.run_id, "step": step,
            "suppress_tools": task.mode == "chat",
            "compacted": any(SUMMARY_MARKER in content_text(m.get("content")) for m in wire),
            "error_recovery": any(m.get("is_error") for m in wire[last_user + 1:]),
            "target_injected_failure": injected_target, "final_answer": not uses,
            "target_rewritten": _blocks(reply) != _blocks(call.reply),
            "verdict": "pass", "verification": verification,
            # a Max teacher's row is Max-derived: build.py trains it only under its
            # explicit allow, and nothing made from it is ever published
            "max_derived": settings.payg, "teacher": settings.teacher,
            "generator": GENERATOR_VERSION,
            "gen_seed": settings.seed, "task_origin": task.origin,
        }
        if settings.payg:
            meta.update(tags=[_artifact_policy().TAG], paid_call_id=call.call_id,
                        cost_usd=round(call.cost_usd, 6), thinking_budget=settings.thinking_budget)
        if out.pad is not None:
            meta["compaction_pad"] = out.pad
        if settings.sampling:
            meta["sampling"] = dict(sorted(settings.sampling.items()))
        # a taskset's provenance (taskset, provenance, template ids, ...) and its task group
        for k, v in task.meta.items():
            if k == "task_id" or k not in meta:
                meta[k] = v
        rows.append({"messages": wire + [target],
                     "tools": [] if task.mode == "chat" else list(payload.get("tools") or []),
                     "_meta": meta})
    # build.py keeps the first row per prompt + call; within a task keep the one
    # that learned from an error (a retry after an injected failure).
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = prompt_call_key(r)
        cur = best.get(key)
        if cur is None or (r["_meta"]["error_recovery"] and not cur["_meta"]["error_recovery"]):
            best[key] = r
    keep = [r for r in rows if best.get(prompt_call_key(r)) is r]
    for r in keep:
        r["_meta"]["rows_in_run"] = len(keep)
    return keep


# ---------------------------------------------------------------------------
# Store: outputs, ledger, resume, the coding cap
# ---------------------------------------------------------------------------


def _repair_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of *path*; a torn last line is cut off the file."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with path.open("r+b") as fh:
            fh.truncate(cut)
        data = data[:cut]
    rows = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())


def _rewrite(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def payg_report(settings: Settings) -> dict[str, Any]:
    """The report's Max fields: the artifact policy (Max-derived, private) and this
    phase's spend from the one ledger. Nothing for a local teacher."""
    if not settings.payg:
        return {}
    spend: dict[str, Any] = {"tool": PAYG_TOOL, "phase": PAYG_PHASE, "model": PAYG_MODEL,
                             "budget_usd": settings.budget_usd, "thinking_budget": settings.thinking_budget}
    client = settings.payg_client
    ledger = getattr(client, "ledger", None)
    if ledger is not None:
        try:
            mine = ledger.totals(tool=PAYG_TOOL, phase=PAYG_PHASE)
            spend.update(phase_committed_usd=round(mine.committed_usd, 4), phase_paid_calls=mine.calls,
                         ledger=ledger.describe(tool=PAYG_TOOL, phase=PAYG_PHASE,
                                                phase_budget_usd=settings.budget_usd),
                         stop=ledger.stop_reason())
        except Exception as exc:  # noqa: BLE001 - the report never fails over the ledger
            spend["ledger_error"] = type(exc).__name__
    return {**_artifact_policy().policy_fields(True), "spend": spend}


class SettingsMismatch(ValueError):
    pass


class Store:
    def __init__(self, out: Path, name: str, fingerprint: dict[str, Any], coding_cap: float,
                 guard: EvalGuard | None = None) -> None:
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.coding_cap = coding_cap
        self.guard = guard
        self.traces = self.out / f"{name}.traces.jsonl"
        self.held = self.out / f"{name}.held.jsonl"
        self.relaxed = self.out / f"{name}.relaxed.jsonl"
        self.ledger = self.out / f"{name}.tasks.jsonl"
        self.report_path = self.out / f"{name}.report.json"
        settings_path = self.out / f"{name}.settings.json"
        if settings_path.exists():
            old = json.loads(settings_path.read_text(encoding="utf-8"))
            if old != fingerprint:
                diff = sorted(k for k in set(old) | set(fingerprint) if old.get(k) != fingerprint.get(k))
                raise SettingsMismatch(f"{settings_path} was written with different settings ({', '.join(diff)}); "
                                       "use another --out or --name")
        else:
            settings_path.write_text(json.dumps(fingerprint, indent=1) + "\n", encoding="utf-8")
        self.repairs: collections.Counter = collections.Counter()
        events = _repair_jsonl(self.ledger)
        self.status: dict[str, dict[str, Any]] = {}
        run_where: dict[str, str] = {}
        for ev in events:
            prev = self.status.get(ev["task_id"])
            if ev["status"] == "released" and prev is not None:
                prev = dict(prev, status="released")
                self.status[ev["task_id"]] = prev
                run_where[prev["run_id"]] = "traces"
                continue
            self.status[ev["task_id"]] = ev
            if ev.get("run_id"):
                run_where[ev["run_id"]] = {"kept": "traces", "held": "held",
                                           "relaxed": "relaxed"}.get(ev["status"], "none")
        trace_rows = _repair_jsonl(self.traces)
        held_rows = _repair_jsonl(self.held)
        relaxed_rows = _repair_jsonl(self.relaxed)
        kept_t = [r for r in trace_rows if run_where.get(r["_meta"]["run_id"]) == "traces"]
        kept_h = [r for r in held_rows if run_where.get(r["_meta"]["run_id"]) == "held"]
        kept_r = [r for r in relaxed_rows if run_where.get(r["_meta"]["run_id"]) == "relaxed"]
        if len(kept_t) != len(trace_rows):
            self.repairs["orphan_trace_rows"] += len(trace_rows) - len(kept_t)
            _rewrite(self.traces, kept_t)
        if len(kept_h) != len(held_rows):
            self.repairs["orphan_held_rows"] += len(held_rows) - len(kept_h)
            _rewrite(self.held, kept_h)
        if len(kept_r) != len(relaxed_rows):
            self.repairs["orphan_relaxed_rows"] += len(relaxed_rows) - len(kept_r)
            _rewrite(self.relaxed, kept_r)
        self.rows_relaxed = len(kept_r)
        self.keys = {prompt_call_key(r) for r in kept_t + kept_h}
        self.rows_kind: collections.Counter = collections.Counter()
        self.rows_domain: collections.Counter = collections.Counter()
        for r in kept_t:
            self.rows_kind[r["_meta"]["task_category"]] += 1
            self.rows_domain[r["_meta"]["category"]] += 1
        self.held_queue: list[tuple[str, list[dict[str, Any]]]] = []
        by_run: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in kept_h:
            by_run[r["_meta"]["run_id"]].append(r)
        for run_id, rs in by_run.items():
            self.held_queue.append((run_id, rs))
        self.stats: collections.Counter = collections.Counter()

    # -- queries ---------------------------------------------------------------------

    @property
    def total_rows(self) -> int:
        return sum(self.rows_domain.values())

    def done(self, task_id: str) -> bool:
        ev = self.status.get(task_id)
        return ev is not None and ev["status"] != "error"

    def attempts(self, kind: str) -> tuple[int, int]:
        tried = [e for e in self.status.values() if e.get("kind") == kind and e["status"] not in ("skipped", "error")]
        passed = [e for e in tried if e["status"] in ("kept", "held", "released")]
        return len(tried), len(passed)

    def mean_rows(self, kind: str, default: float) -> float:
        rs = [e.get("rows", 0) for e in self.status.values()
              if e.get("kind") == kind and e["status"] in ("kept", "held", "released")]
        return sum(rs) / len(rs) if rs else default

    def coding_fits(self, n_coding: int, n_total: int) -> bool:
        coding = self.rows_domain["coding"] + n_coding
        total = self.total_rows + n_total
        return coding <= self.coding_cap * total + 1e-9

    # -- writes ----------------------------------------------------------------------

    def _event(self, ev: dict[str, Any]) -> None:
        ev = dict(ev, ts=round(time.time(), 3))
        _append(self.ledger, [ev])
        if ev["status"] == "released":
            self.status[ev["task_id"]] = dict(self.status[ev["task_id"]], status="released")
        else:
            self.status[ev["task_id"]] = ev

    def _count(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            self.rows_kind[r["_meta"]["task_category"]] += 1
            self.rows_domain[r["_meta"]["category"]] += 1
            self.keys.add(prompt_call_key(r))

    def record_skip(self, task: GenTask, reasons: list[str]) -> None:
        self._event({"task_id": task.id, "task_hash": task.task_hash, "kind": task.kind,
                     "domain": task.domain, "family": task.family, "status": "skipped",
                     "reasons": reasons, "rows": 0})

    def commit(self, task: GenTask, run_id: str, status: str, rows: list[dict[str, Any]],
               reasons: list[str], extra: dict[str, Any] | None = None) -> str:
        """Write one task's outcome. *status* is kept / relaxed / rejected / error;
        a kept coding task over the cap becomes held. Relaxed rows go to their own
        file and count toward nothing (no target, mix, cap or dedupe). Returns the
        final status."""
        if status == "relaxed":
            fresh = [r for r in rows if self.guard is None or not self.guard.row_overlaps(r)]
            if not fresh:
                status, reasons = "rejected", ["no_rows"]
            else:
                _append(self.relaxed, fresh)
                self.rows_relaxed += len(fresh)
                self._event({"task_id": task.id, "task_hash": task.task_hash, "kind": task.kind,
                             "domain": task.domain, "family": task.family, "run_id": run_id,
                             "reasons": reasons, "rows": len(fresh), **(extra or {}), "status": "relaxed"})
                return "relaxed"
        if status == "kept":
            fresh = []
            for r in rows:
                key = prompt_call_key(r)
                if key in self.keys:
                    self.stats["duplicate_rows"] += 1
                    continue
                if self.guard is not None and self.guard.row_overlaps(r):
                    self.stats["eval_overlap_rows"] += 1
                    continue
                fresh.append(r)
            rows = fresh
            for r in rows:
                r["_meta"]["rows_in_run"] = len(rows)
            if not rows:
                status, reasons = "rejected", ["no_rows"]
        ev = {"task_id": task.id, "task_hash": task.task_hash, "kind": task.kind, "domain": task.domain,
              "family": task.family, "run_id": run_id, "reasons": reasons, "rows": len(rows), **(extra or {})}
        if status == "kept":
            n_coding = len(rows) if task.domain == "coding" else 0
            if n_coding and not self.coding_fits(n_coding, len(rows)):
                _append(self.held, rows)
                self.held_queue.append((run_id, rows))
                self.keys.update(prompt_call_key(r) for r in rows)
                self._event(dict(ev, status="held"))
                return "held"
            _append(self.traces, rows)
            self._count(rows)
            self._event(dict(ev, status="kept"))
            self.release_held()
            return "kept"
        self._event(dict(ev, status=status, rows=0))
        return status

    def release_held(self) -> int:
        released = 0
        while self.held_queue:
            run_id, rows = self.held_queue[0]
            if not self.coding_fits(len(rows), len(rows)):
                break
            self.held_queue.pop(0)
            _append(self.traces, rows)
            for r in rows:
                self.rows_kind[r["_meta"]["task_category"]] += 1
                self.rows_domain[r["_meta"]["category"]] += 1
            task_id = rows[0]["_meta"].get("gen_task_id") or rows[0]["_meta"]["task_id"]
            self._event({"task_id": task_id, "status": "released", "run_id": run_id})
            remaining = [r for _rid, rs in self.held_queue for r in rs]
            _rewrite(self.held, remaining)
            released += 1
        return released

    # -- report ----------------------------------------------------------------------

    def report(self, settings: Settings, started: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        by_kind: dict[str, Any] = {}
        reasons: collections.Counter = collections.Counter()
        judge_n = judge_pass = 0
        for ev in self.status.values():
            k = ev.get("kind") or "?"
            d = by_kind.setdefault(k, collections.Counter())
            d[ev["status"]] += 1
            for r in ev.get("reasons") or []:
                reasons[r] += 1
            j = ev.get("judge")
            if j:
                judge_n += 1
                judge_pass += bool(j.get("pass"))
        kinds = {}
        for k in KINDS:
            c = by_kind.get(k, collections.Counter())
            tried = sum(c[s] for s in ("kept", "held", "released", "rejected", "relaxed"))
            passed = c["kept"] + c["held"] + c["released"]
            kinds[k] = {"attempted": tried, "verified": passed, "kept": c["kept"] + c["released"],
                        "held": c["held"], "rejected": c["rejected"], "relaxed": c["relaxed"],
                        "skipped_eval_overlap": c["skipped"],
                        "pass_rate": round(passed / tried, 3) if tried else None,
                        "rows": self.rows_kind[k],
                        "row_share": round(self.rows_kind[k] / self.total_rows, 3) if self.total_rows else 0.0,
                        "target_share": settings.mix.get(k, 0.0)}
        total = self.total_rows
        return {
            "schema": "distill-teacher-gen-report/1", "generator": GENERATOR_VERSION,
            "source": settings.row_source, "task_source": settings.task_source,
            "sampling": dict(sorted(settings.sampling.items())),
            "model": settings.model_name, "judge": settings.judge_kind, "judge_seed": settings.judge_seed,
            "teacher_served": settings.teacher_served, "jail": resolve_jail(settings.jail),
            "teacher": settings.teacher, "judge_provider": settings.judge_provider,
            "pad_pool": settings.pad_pool or None, **payg_report(settings),
            "rows_total": total, "rows_held": sum(len(rs) for _r, rs in self.held_queue),
            # verified only under a relaxation; source RELAXED_SOURCE, never train rows
            "rows_relaxed": self.rows_relaxed,
            "by_kind": kinds,
            "by_category": {d: {"rows": self.rows_domain[d],
                                "share": round(self.rows_domain[d] / total, 3) if total else 0.0}
                            for d in DOMAINS},
            "coding_share": round(self.rows_domain["coding"] / total, 3) if total else 0.0,
            "coding_cap": self.coding_cap,
            "judge_calls": judge_n, "judge_pass_rate": round(judge_pass / judge_n, 3) if judge_n else None,
            "rejection_reasons": dict(reasons.most_common()),
            "row_filters": dict(self.stats), "resume_repairs": dict(self.repairs),
            "elapsed_s": round(time.time() - started, 1), **(extra or {}),
        }

    def write_report(self, report: dict[str, Any]) -> None:
        tmp = self.report_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.report_path)


class FixedGenerator(Generator):
    """Runs exactly the tasks in a ``--tasks-file`` (same schema as list-tasks)."""

    def __init__(self, tasks: list[GenTask]) -> None:
        super().__init__(Sources([]), DEFAULT_SEED)
        self.tasks = {t.id: t for t in tasks}
        self.order: dict[str, list[str]] = collections.defaultdict(list)
        for t in tasks:
            self.order[t.kind].append(t.id)

    def families(self, kind: str) -> list[str]:
        return [f"fixed:{kind}"] if self.order.get(kind) else []

    def make(self, family: str, idx: int) -> GenTask:
        kind = family.split(":", 1)[1]
        ids = self.order[kind]
        if idx >= len(ids):
            raise IndexError(idx)
        return self.tasks[ids[idx]]


#: What an injected failure says on a real-prompt task (whichever tool is called first).
TASKSET_INJECTION = {"tool": ANY_TOOL, "nth": 1, "kind": "timeout",
                     "output": "Error: the call timed out after 30 seconds (no answer from the service). "
                               "Try again."}
_TASKSET_META = {"taskset": "taskset", "provenance": "provenance", "subcategory": "task_subcategory",
                 "intent": "task_intent", "template_id": "template_id",
                 "source_pattern_hash": "source_pattern_hash", "transform": "transform"}


class NoPadPrompts(ValueError):
    """A force_compaction row has too few other real-pattern prompts to pad with."""


def _row_groups(row: dict[str, Any]) -> set[str]:
    """Every task group a taskset row's text comes from: its own and, for a
    recombined row, its partner's (``"tg-" + sources[i].pattern_hash``)."""
    groups = {str(row["task_group"])}
    for src in row.get("sources") or []:
        if isinstance(src, dict) and src.get("pattern_hash"):
            groups.add("tg-" + str(src["pattern_hash"]))
    return groups


def _rank(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _flagged(row: dict[str, Any]) -> bool:
    return bool(row["flags"]["inject_errors"] or row["flags"]["force_compaction"])


def pad_pool_split(rows: list[dict[str, Any]], *, share: float = PAD_POOL_SHARE,
                   guard: "EvalGuard | None" = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """(rows to run as tasks, the reserved pad pool, a summary) for a taskset.

    Only a taskset with force_compaction rows reserves a pool. It is whole task
    groups with no flagged row (inject_errors / force_compaction rows are the
    scarce execution conditions and always run), ranked by a hash keyed by the
    taskset's name, so every run and every teacher over one taskset reserves the
    same groups: ``floor(share x groups)``, extended in rank order until it holds
    ``PAD_MIN_TURNS`` groups (of chat rows too, when a chat-mode row is padded).
    Pool rows never run as tasks: their prompts appear only as padding, so a
    padded row never shares a task with a row in another build split. A taskset
    too small for that (no group to spare) reserves nothing and pads from every
    other row (``reserved: false``). With *guard*, pool rows that overlap the eval
    sets are dropped from the pool (and not run either: the scheduler would skip them)."""
    comp = [r for r in rows if r["flags"]["force_compaction"]]
    summary: dict[str, Any] = {"version": PAD_VERSION, "share": share, "reserved": False,
                               "pool_rows": 0, "pool_groups": 0, "eval_overlap_dropped": 0}
    if not comp or share <= 0:
        return list(rows), [], summary
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_group[str(r["task_group"])].append(r)
    name = "|".join(sorted({str(r.get("taskset")) for r in rows}))
    eligible = sorted((g for g, rs in by_group.items() if not any(_flagged(r) for r in rs)),
                      key=lambda g: _rank(f"teacher-gen|pad-pool|{name}|{g}"))
    want = math.floor(share * len(by_group))
    if want <= 0 or len(eligible) <= want:
        return list(rows), [], summary
    need_chat = PAD_MIN_TURNS if any(r["mode"] == "chat" for r in comp) else 0
    chosen: list[str] = []
    for g in eligible:
        chat_groups = sum(1 for c in chosen if any(r["mode"] == "chat" for r in by_group[c]))
        if len(chosen) >= max(want, PAD_MIN_TURNS) and chat_groups >= need_chat:
            break
        chosen.append(g)
    pool_groups = set(chosen)
    pool = [r for r in rows if str(r["task_group"]) in pool_groups]
    main = [r for r in rows if str(r["task_group"]) not in pool_groups]
    if guard is not None:
        kept = [r for r in pool if not guard.prompt_overlaps(r["prompt"])]
        summary["eval_overlap_dropped"] = len(pool) - len(kept)
        pool = kept
    summary.update(reserved=True, pool_rows=len(pool), pool_groups=len({r["task_group"] for r in pool}))
    return main, pool, summary


def pad_candidates(row: dict[str, Any], pool: list[dict[str, Any]], *,
                   guard: "EvalGuard | None" = None) -> list[dict[str, Any]]:
    """Up to ``PAD_MAX_TURNS`` pad turns for one force_compaction row: other
    real-pattern prompts from *pool*, in an order seeded by the row's id, none
    sharing a task group with the row or with each other (a recombined row's
    partner group counts), chat prompts only for a chat-mode row, none overlapping
    the eval sets (*guard*). Each turn runs in its own row's mode."""
    mine = _row_groups(row)
    chat = row["mode"] == "chat"
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for cand in sorted(pool, key=lambda r: _rank(f"teacher-gen|pad|{row['id']}|{r['id']}")):
        groups = _row_groups(cand)
        if cand["id"] == row["id"] or groups & mine or groups & used:
            continue
        if chat and cand["mode"] != "chat":
            continue
        if guard is not None and guard.prompt_overlaps(cand["prompt"]):
            continue
        docs = sorted({str(sl["value"]) for sl in cand.get("slots") or []
                       if isinstance(sl, dict) and sl.get("type") in ("path", "file")
                       and isinstance(sl.get("value"), str)})
        out.append({"id": cand["id"], "task_group": cand["task_group"],
                    "mode": "chat" if cand["mode"] == "chat" else "agent", "prompt": cand["prompt"],
                    "docs": docs})
        used |= groups
        if len(out) >= PAD_MAX_TURNS:
            break
    return out


def pad_docs(pads: list[dict[str, Any]], sources: Sources, key: str) -> dict[str, str]:
    """Whole repo docs for an agent task's sandbox, so a read during padding returns
    real repo text: every doc a pad prompt names (by its path slots) that the repo
    has, then ``PAD_DOCS`` more in an order seeded by *key*; each at most
    ``PAD_DOC_MAX_BYTES``, written at its repo path."""
    if sources.synthetic or sources.root is None or not sources.doc_paths:
        raise NoRepoDocs("compaction padding needs the repo's own docs (the sources are synthetic or empty)")
    have = set(sources.doc_paths)
    named = [d for p in pads for d in p.get("docs") or [] if d in have]
    rest = sorted(sources.doc_paths, key=lambda d: _rank(f"teacher-gen|pad-docs|{key}|{d}"))
    files: dict[str, str] = {}
    extra = 0
    for rel in list(dict.fromkeys(named)) + rest:
        if rel in files:
            continue
        is_named = rel in named
        if not is_named and extra >= PAD_DOCS:
            break
        try:
            raw = (sources.root / rel).read_bytes()
        except OSError:
            continue
        if len(raw) > PAD_DOC_MAX_BYTES:
            continue
        files[rel] = raw.decode("utf-8", "replace")
        extra += not is_named
    if not files:
        raise NoRepoDocs(f"no repo doc under {PAD_DOC_MAX_BYTES} bytes to put in a padded task's sandbox")
    return files


def task_from_taskset(row: dict[str, Any], sources: Sources, *, pool: list[dict[str, Any]] | None = None,
                      guard: "EvalGuard | None" = None) -> GenTask:
    """A validated ``taskset_gen.py`` row (its teacher_gen contract) as a GenTask
    of origin ORIGIN_TASKSET: id, prompt and category as given; mode chat
    (``flags.suppress_tools``) runs ``run_loop(mode="chat")``, which sends no
    tools; ``inject_errors`` fails the first tool call (a timeout) and the run
    must recover from it; ``force_compaction`` puts earlier turns before the
    prompt (``pad``: other real-pattern prompts from *pool*, answered live by the
    same teacher; an agent task's sandbox also holds whole repo docs, so a read
    returns real repo text) and runs the real compactor with a small window. The
    row's provenance goes into every trace row's _meta, ``_meta.task_id`` is its
    ``task_group`` and a recombined row's partner group is ``partner_task_id``."""
    flags = row["flags"]
    chat = row["mode"] == "chat"
    kind = ("compacted-history" if flags["force_compaction"] else "error-recovery" if flags["inject_errors"]
            else "chat-mode" if chat else "coding" if row["category"] == "coding" else "agentic")
    meta = {dst: row[src] for src, dst in _TASKSET_META.items()}
    # build.py unions rows on _meta.task_id: every variant of one real prompt shares
    # its task_group, so the variants land in one split (the ledger keeps the row id)
    meta["task_id"] = row["task_group"]
    if row["transform"] == "recombine" and len(row["sources"]) == 2:
        partner = "tg-" + row["sources"][1]["pattern_hash"]
        if partner != row["task_group"]:
            meta["partner_task_id"] = partner
    fields: dict[str, Any] = dict(
        id=row["id"], kind=kind, domain=row["category"], family=f"taskset:{row['subcategory']}",
        prompt=row["prompt"], mode="chat" if chat else "agent", meta=meta, origin=ORIGIN_TASKSET)
    if chat:
        fields["score"] = {"expect_text_min_len": 40}
    if flags["inject_errors"]:
        fields.update(inject=[dict(TASKSET_INJECTION)], expect_recovery=True)
    if flags["force_compaction"]:
        if not chat and (sources.synthetic or not sources.doc_paths):
            # never pad a real-pattern task with text written in this file
            raise NoRepoDocs(f"taskset row {row['id']}: force_compaction needs real repo docs to read "
                             "(the sources are synthetic or empty)")
        # the earlier turns are other real-pattern prompts and the teacher's own answers to them
        pads = pad_candidates(row, pool or [], guard=guard)
        if len(pads) < PAD_MIN_TURNS:
            raise NoPadPrompts(
                f"taskset row {row['id']}: force_compaction needs at least {PAD_MIN_TURNS} other real-pattern "
                f"prompts to pad with ({'chat-mode ones, ' if chat else ''}from other task groups, clear of the "
                f"eval sets); found {len(pads)}")
        fields["pad"] = pads
        if not chat:
            fields["setup_files"] = pad_docs(pads, sources, row["id"])
        fields["compaction"] = True
    task = GenTask(**fields)
    task.validate()
    return task


def _read_jsonl_rows(path: str) -> list[Any]:
    rows = []
    for n, line in enumerate(Path(path).expanduser().read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}: line {n}: not JSON ({exc})") from exc
    if not rows:
        raise ValueError(f"{path}: no tasks")
    return rows


def _is_taskset_row(row: Any) -> bool:
    return isinstance(row, dict) and "taskset" in row and "provenance" in row


def read_taskset_rows(path: str) -> list[dict[str, Any]]:
    """The rows of a ``taskset_gen.py`` file, each checked by that file's own
    ``validate_task`` (schema, id = hash of the prompt, provenance). Anything
    else (a list-tasks row, a row that fails validation) is a ValueError: only
    a real-pattern taskset may feed a train-source run."""
    ts = _load_sibling("distill_taskset_gen", "taskset_gen.py")
    if ts.PROVENANCE != TASKSET_PROVENANCE:
        raise ValueError(f"taskset_gen.py stamps {ts.PROVENANCE!r}, this script expects {TASKSET_PROVENANCE!r}")
    rows = _read_jsonl_rows(path)
    for n, r in enumerate(rows, 1):
        if not _is_taskset_row(r):
            raise ValueError(f"{path}: line {n}: not a taskset_gen.py row (--tasks takes only a real-pattern "
                             "taskset; built-in tasks run with --builtin-smoke)")
        problems = ts.validate_task(r)
        if problems:
            raise ValueError(f"{path}: taskset row {n}: {problems[0]}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: task ids repeat")
    return rows


def tasks_from_taskset_rows(rows: list[dict[str, Any]], repo: Path, *,
                            pool: list[dict[str, Any]] | None = None,
                            guard: "EvalGuard | None" = None) -> list[GenTask]:
    """GenTasks of validated taskset rows. A ``force_compaction`` row pads with
    prompts from *pool* (default: *rows* themselves, every other row; ``load_taskset``
    passes the reserved pad pool) and, when it is an agent task, with *repo*'s own
    docs; a repo without any is refused (NoRepoDocs, a ValueError), never replaced
    by the synthetic runbooks. Too few prompts to pad with is NoPadPrompts."""
    need_docs = any(r["flags"]["force_compaction"] and r["mode"] != "chat" for r in rows)
    sources = Sources.from_repo(repo, allow_synthetic=False) if need_docs else Sources([])
    pool = rows if pool is None else pool
    return [task_from_taskset(r, sources, pool=pool, guard=guard) for r in rows]


def load_taskset_with_pool(path: str, repo: Path, *, guard: "EvalGuard | None" = None) -> tuple[list[GenTask], dict[str, Any]]:
    """``--tasks``: the GenTasks to run (origin taskset) and the pad-pool summary
    (``pad_pool_split``): the reserved pool's rows pad force_compaction tasks and
    are never run themselves."""
    rows = read_taskset_rows(path)
    main, pool, summary = pad_pool_split(rows, guard=guard)
    tasks = tasks_from_taskset_rows(main, repo, pool=pool if summary["reserved"] else rows, guard=guard)
    summary["tasks"] = len(tasks)
    return tasks, summary


def load_taskset(path: str, repo: Path, *, guard: "EvalGuard | None" = None) -> list[GenTask]:
    """``--tasks``: a real-pattern taskset as GenTasks (origin taskset), its pad pool held back."""
    return load_taskset_with_pool(path, repo, guard=guard)[0]


def load_builtin_tasks(path: str) -> list[GenTask]:
    """``--builtin-smoke --tasks-file``: list-tasks rows (built-in tasks only;
    a taskset row here is refused, it belongs to ``--tasks``)."""
    tasks = []
    for n, r in enumerate(_read_jsonl_rows(path), 1):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: line {n}: not a task object")
        if _is_taskset_row(r):
            raise ValueError(f"{path}: line {n}: a taskset_gen.py row; run real-pattern tasks with --tasks")
        try:
            tasks.append(GenTask.from_dict(r))
        except TypeError as exc:
            raise ValueError(f"{path}: line {n}: {exc}") from exc
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: task ids repeat")
    return tasks


def load_tasks_file(path: str, repo: Path) -> list[GenTask]:
    """A tasks file of either kind: a ``taskset_gen.py`` taskset (``load_taskset``)
    or list-tasks rows (``load_builtin_tasks``). ValueError / OSError on bad
    input. The CLI never guesses: ``--tasks`` reads only a taskset."""
    rows = _read_jsonl_rows(path)
    if all(_is_taskset_row(r) for r in rows):
        return load_taskset(path, repo)
    return load_builtin_tasks(path)



# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

EXPECTED_ROWS = {"chat-mode": 1.0, "compacted-history": 2.0, "error-recovery": 2.5, "agentic": 2.0,
                 "coding": 3.0}


class Scheduler:
    """Next task: the kind furthest below its target share of rows (kept +
    in flight), ties to the earlier kind; families round-robin inside a kind;
    the lowest index of a family not yet in the ledger."""

    def __init__(self, gen: Generator, store: Store, mix: dict[str, float],
                 guard: EvalGuard | None = None, kinds: tuple[str, ...] = KINDS) -> None:
        self.gen = gen
        self.store = store
        self.mix = mix
        self.guard = guard
        self.kinds = [k for k in kinds if mix.get(k, 0) > 0]
        self.inflight: collections.Counter = collections.Counter()
        self.reserved: set[str] = set()
        self.cursor: dict[str, int] = collections.defaultdict(int)
        self.rr: dict[str, int] = collections.Counter(
            ev.get("kind") for ev in store.status.values() if ev["status"] != "skipped")
        self.exhausted: set[str] = set()
        # One task per normalized prompt: build.py treats equal prompts as one task
        # and keeps one row per prompt + call, so a repeat would teach nothing new.
        self.used_hashes: set[str] = {ev["task_hash"] for ev in store.status.values()
                                      if ev.get("task_hash") and ev["status"] not in ("error", "skipped")}
        self.dup_streak: collections.Counter = collections.Counter()
        self.dead: set[str] = set()

    def expected(self, kind: str) -> float:
        return self.store.mean_rows(kind, EXPECTED_ROWS.get(kind, 2.0))

    def pick_kind(self) -> str | None:
        have = {k: self.store.rows_kind[k] + self.inflight[k] for k in self.kinds}
        total = sum(have.values())
        best: tuple[float, str] | None = None
        for k in self.kinds:
            if k in self.exhausted:
                continue
            deficit = self.mix[k] * (total + 1) - have[k]
            tried, passed = self.store.attempts(k)
            if tried >= 10:
                rate = (passed + 1) / (tried + 2)
                deficit = deficit * max(0.1, rate) if deficit > 0 else deficit / max(0.1, rate)
            if best is None or deficit > best[0] + 1e-9:
                best = (deficit, k)
        return best[1] if best else None

    def _next_index(self, family: str) -> int:
        i = self.cursor[family]
        while self.store.done(f"{family}-{i:05d}") or f"{family}-{i:05d}" in self.reserved:
            i += 1
        self.cursor[family] = i + 1
        return i

    def next_task(self, max_skips: int = 200) -> GenTask | None:
        if isinstance(self.gen, FixedGenerator):
            return self._next_fixed()
        while True:
            kind = self.pick_kind()
            if kind is None:
                return None
            for _ in range(max_skips):
                fams = [f for f in self.gen.families(kind) if f not in self.dead]
                if not fams:
                    break
                fam = fams[self.rr[kind] % len(fams)]
                self.rr[kind] += 1
                task = self.gen.make(fam, self._next_index(fam))
                if task.task_hash in self.used_hashes:
                    self.dup_streak[fam] += 1
                    if self.dup_streak[fam] >= 50:
                        self.dead.add(fam)
                    continue
                self.dup_streak[fam] = 0
                reasons = self.guard.task_overlaps(task) if self.guard else []
                if reasons:
                    self.store.record_skip(task, reasons)
                    continue
                self.used_hashes.add(task.task_hash)
                self.reserved.add(task.id)
                self.inflight[kind] += self.expected(kind)
                return task
            self.exhausted.add(kind)

    def _next_fixed(self) -> GenTask | None:
        gen = self.gen
        assert isinstance(gen, FixedGenerator)
        while True:
            kind = self.pick_kind()
            if kind is None:
                return None
            for tid in gen.order.get(kind, []):
                if tid in self.reserved or self.store.done(tid):
                    continue
                task = gen.tasks[tid]
                reasons = self.guard.task_overlaps(task) if self.guard else []
                if reasons:
                    self.store.record_skip(task, reasons)
                    continue
                self.reserved.add(tid)
                self.inflight[kind] += self.expected(kind)
                return task
            self.exhausted.add(kind)

    def release(self, task: GenTask) -> None:
        self.reserved.discard(task.id)
        self.inflight[task.kind] = max(0.0, self.inflight[task.kind] - self.expected(task.kind))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


async def preflight(url: str, timeout: float) -> str | None:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(f"{url}/v1/models")
    except Exception as exc:  # noqa: BLE001
        raise EndpointError(f"{url}/v1/models is unreachable ({type(exc).__name__})") from exc
    if resp.status_code != 200:
        raise EndpointError(f"{url}/v1/models returned HTTP {resp.status_code}")
    data = (resp.json() or {}).get("data") or [{}]
    return data[0].get("id")


def has_max_token(model_id: str | None) -> bool:
    """True when a model id names a Max model (the token ``max``: ``qwen3.8-max``)."""
    return "max" in re.split(r"[^a-z0-9]+", (model_id or "").lower())


def check_model_id(model_id: str | None, pattern: str, role: str) -> str:
    """The id a local server reports, or EndpointError: it must match *pattern*
    (a case-insensitive glob) and never name a Max model."""
    if not model_id:
        raise EndpointError(f"the {role} endpoint's /v1/models names no model")
    if has_max_token(model_id):
        raise EndpointError(f"the {role} endpoint serves {model_id!r}: a Max model is never used here")
    if not fnmatch.fnmatchcase(model_id.lower(), pattern.lower()):
        raise EndpointError(f"the {role} endpoint serves {model_id!r}, which does not match {pattern!r} "
                            f"(--{role}-id)")
    return model_id


async def process(task: GenTask, settings: Settings, worker: int) -> tuple[str, list[dict[str, Any]], list[str], dict[str, Any]]:
    """Run + verify one task: (status, rows, reasons, ledger extras). Status is
    kept, relaxed (verified only under --max-natural-errors / --keep-refused:
    rows build.py's strict policy drops), rejected (done; never retried) or
    error (the teacher or judge was unreachable: retried by the next run).
    Raises PaygStop when the spend ledger refused a paid call (the run stops;
    the task reruns next time)."""
    from prometheus.api.turn_errors import classify_turn_error

    out = await run_task(task, settings, worker)
    extra: dict[str, Any] = {"run_id": out.run_id, "duration_s": round(out.duration_s, 1),
                             "teacher_calls": len(out.calls),
                             "tokens_in": sum(c.input_tokens for c in out.calls),
                             "tokens_out": sum(c.output_tokens for c in out.calls)}
    if out.pad is not None:
        extra["pad_turns"] = out.pad["turns"]
    if out.recovered_errors:
        extra["recovered_provider_errors"] = [e[:200] for e in out.recovered_errors]
    try:
        if out.payg_stop is not None:
            raise out.payg_stop
        if out.transport_error:
            return "error", [], ["transport_error"], dict(extra, error=out.error[:200])
        if out.rejection:
            return "rejected", [], [out.rejection], dict(extra, error=(out.error or out.provider.errors[-1])[:200])
        if out.pad_reasons:
            return "rejected", [], list(out.pad_reasons), dict(extra, compaction_pad=out.pad)
        ok, reasons, record = deterministic_checks(out, settings)
        extra["verification"] = record
        if not ok:
            return "rejected", [], reasons, extra
        import httpx

        try:
            record["judge"] = await judge(out, settings)
        except PaygCallFailed as exc:
            if exc.kind in RETRY_KINDS:
                return "error", [], ["judge_unreachable"], dict(extra, error=str(exc)[:200])
            return "rejected", [], [f"judge_{exc.kind}"], dict(extra, error=str(exc)[:200])
        except httpx.HTTPStatusError as exc:
            kind = str(classify_turn_error(exc).get("kind") or "unknown")
            if kind not in RETRY_KINDS:  # the judge refused this transcript (too long, say): it always will
                return "rejected", [], [f"judge_{kind}"], dict(extra, error=f"HTTP {exc.response.status_code}")
            return "error", [], ["judge_unreachable"], dict(extra, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        except (httpx.HTTPError, OSError, ValueError) as exc:  # transport, a non-JSON body
            return "error", [], ["judge_unreachable"], dict(extra, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            if settings.payg:
                judged = record.get("judge") or {}
                extra["cost_usd"] = round(out.paid_usd + float(judged.get("cost_usd") or 0), 6)
        extra["judge"] = record["judge"]
        # a local judge never names Max (a proxy in front of Max, the Token Plan); only the
        # pay-per-token self-judge may, and only on the paid key
        payg_judge = record["judge"].get("provider") == TEACHER_PAYG
        if has_max_token(record["judge"].get("served_model")) and not payg_judge:
            return "rejected", [], ["judge_model_refused"], extra
        if payg_judge and not has_max_token(record["judge"].get("served_model") or PAYG_MODEL):
            return "rejected", [], ["judge_model_refused"], extra
        if not record["judge"]["pass"]:
            return "rejected", [], ["judge_fail"], extra
        record["verified"] = True
        rows = rows_for(out, settings, record)
        if not rows:
            return "rejected", [], ["no_rows"], extra
        if not record["tool_success"]:
            return "relaxed", rows, ["relaxed"], extra
        return "kept", rows, [], extra
    finally:
        shutil.rmtree(out.env.root, ignore_errors=True)


class TeacherUnreachable(RuntimeError):
    pass


async def run(settings: Settings, gen: Generator, store: Store, guard: EvalGuard | None) -> dict[str, Any]:
    started = time.time()
    sched = Scheduler(gen, store, settings.mix, guard)
    state: dict[str, Any] = {"started": 0, "consecutive_errors": 0, "stop": "", "stop_detail": ""}

    def stop_reason() -> str:
        if state["stop"]:
            return state["stop"]
        if settings.max_tasks is not None and state["started"] >= settings.max_tasks:
            return "max_tasks"
        if settings.target_rows is not None and store.total_rows >= settings.target_rows:
            return "target_rows"
        if settings.until is not None and time.time() > settings.until - settings.min_task_seconds:
            return "until"
        return ""

    async def worker(wid: int) -> None:
        while not stop_reason():
            task = sched.next_task()
            if task is None:
                state["stop"] = state["stop"] or "no_tasks"
                return
            state["started"] += 1
            budget = settings.task_timeout
            if settings.until is not None:
                budget = min(budget, settings.until - time.time())
            try:
                status, rows, reasons, extra = await asyncio.wait_for(process(task, settings, wid), budget)
            except PaygStop as exc:
                # the spend ledger refused a paid call (STOP, the cap, --budget-usd) or the API
                # refused the key: stop cleanly; this task is not recorded and reruns next time
                sched.release(task)
                state["stop"] = state["stop"] or exc.kind
                state["stop_detail"] = state["stop_detail"] or exc.message[:300]
                log.warning("paid calls stopped (%s): %s", exc.kind, exc.message[:300])
                return
            except asyncio.TimeoutError:
                if settings.until is not None and time.time() >= settings.until - 1:
                    sched.release(task)
                    state["stop"] = "until"
                    return
                status, rows, reasons, extra = "rejected", [], ["task_timeout"], {}
            except Exception as exc:  # noqa: BLE001 - a judge transport failure, a bug in one task
                status, rows, reasons, extra = "error", [], ["harness_error"], {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            sched.release(task)
            final = store.commit(task, extra.pop("run_id", None) or f"{task.id}@{uuid.uuid4().hex[:10]}",
                                 status, rows, reasons, extra)
            if status == "error":
                state["consecutive_errors"] += 1
                if state["consecutive_errors"] >= settings.max_consecutive_errors:
                    state["stop"] = "consecutive_errors"
            else:
                state["consecutive_errors"] = 0
            store.write_report(store.report(settings, started, {"stop": stop_reason() or "running"}))
            if settings.progress:
                print(f"[{final}] {task.id} rows={len(rows) if final in ('kept', 'held') else 0}"
                      f"{' ' + ','.join(reasons) if reasons else ''} ({extra.get('duration_s', '-')}s)",
                      file=sys.stderr, flush=True)

    await asyncio.gather(*(worker(i) for i in range(max(1, settings.concurrency))))
    store.release_held()
    final: dict[str, Any] = {"stop": stop_reason() or "done", "tasks_started": state["started"]}
    if state["stop_detail"]:
        final["stop_detail"] = state["stop_detail"]
    report = store.report(settings, started, final)
    store.write_report(report)
    return report


#: An HH:MM --until is its next occurrence; one that lands further ahead than this
#: has most likely passed today already (a late start would otherwise run all day).
UNTIL_HHMM_MAX_AHEAD_H = 12.0


def parse_until(text: str | None, now: dt.datetime | None = None,
                max_ahead_hours: float | None = None) -> float | None:
    """--until as a timestamp: HH:MM (local, next occurrence) or an ISO time.
    With *max_ahead_hours*, an HH:MM whose next occurrence is further ahead is a
    ValueError (it has passed today: give an ISO time to mean tomorrow)."""
    if not text:
        return None
    now = now or dt.datetime.now().astimezone()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text.strip())
    if m:
        target = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if target <= now:
            target += dt.timedelta(days=1)
        ahead_h = (target - now).total_seconds() / 3600
        if max_ahead_hours is not None and ahead_h > max_ahead_hours:
            raise ValueError(f"--until {text.strip()} is next at {target.isoformat(timespec='minutes')}, "
                             f"{ahead_h:.1f} h away (more than {max_ahead_hours:g} h): it has most likely "
                             "passed today; give an ISO time to mean tomorrow")
        return target.timestamp()
    when = dt.datetime.fromisoformat(text.strip())
    if when.tzinfo is None:
        when = when.astimezone()
    return when.timestamp()


def parse_mix(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_MIX)
    mix = {k: 0.0 for k in KINDS}
    for part in text.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in KINDS:
            raise ValueError(f"unknown kind {k!r} in --mix (kinds: {', '.join(KINDS)})")
        mix[k] = float(v)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("--mix shares must add up to more than 0")
    return {k: v / total for k, v in mix.items()}


def load_pack_template(path: str | None) -> tuple[str | None, str | None]:
    """The pack's system prompt as stored (its date sentinel unfilled) and the sentinel (None: none)."""
    if not path:
        return None, None
    pack = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    sentinel = pack.get("date_sentinel")
    return pack["system_prompt"], (str(sentinel) if sentinel else None)


def load_pack_prompt(path: str | None) -> str | None:
    """The pack's system prompt with today's date filled in (teacher_ab fixes the date this way)."""
    prompt, sentinel = load_pack_template(path)
    if prompt is None:
        return None
    return prompt.replace(sentinel, dt.date.today().isoformat()) if sentinel else prompt


def sampling_from_args(a: argparse.Namespace) -> dict[str, float]:
    """The explicit sampling flags given (none: the server's own defaults)."""
    out: dict[str, float] = {}
    for key in SAMPLING_KEYS:
        value = getattr(a, key, None)
        if value is None:
            continue
        if key == "top_k":
            if int(value) != value or value < 0:
                raise ValueError("--top-k must be a whole number >= 0")
            out[key] = int(value)
        else:
            if not 0 <= value <= (5.0 if key == "temperature" else 1.0):
                raise ValueError(f"--{key.replace('_', '-')} {value} is out of range")
            out[key] = float(value)
    return out


def _payg_args(a: argparse.Namespace, out: Path) -> dict[str, Any]:
    """The --teacher max-payg settings, or ValueError (a usage error, exit 2)."""
    if a.builtin_smoke or a.tasks_file or not a.tasks:
        # the operator approved the paid key for the real-pattern taskset only: a smoke run's
        # hand-written tasks would spend the budget on rows build.py never trains on
        raise ValueError("--teacher max-payg runs the real-pattern taskset only (--tasks <taskset.jsonl>); "
                         "--builtin-smoke / --tasks-file are smoke runs for the local teacher")
    spend = _spend()
    AP = _artifact_policy()
    for flag, given in (("--teacher-url", a.teacher_url), ("--teacher-id", a.teacher_id)):
        if given:
            raise ValueError(f"{flag} names a local teacher; --teacher max-payg calls {PAYG_MODEL} through "
                             "spend.py on the pay-per-token key")
    if a.model_name not in (None, PAYG_MODEL):
        raise ValueError(f"--teacher max-payg runs {PAYG_MODEL} (the priced model); --model-name {a.model_name!r} "
                         "is not it")
    if a.judge_model and not a.judge_url:
        raise ValueError("--judge-model names a LOCAL judge (--judge-url); without one Max judges itself")
    if a.probe_guard_url and not a.judge_url:
        raise ValueError("with --teacher max-payg the probe guard fronts the local judge only: give --judge-url")
    if a.min_p is not None:
        raise ValueError("--min-p is not a field the paid API takes (spend.PAYLOAD_KEYS)")
    budget = a.budget_usd
    if budget is None or not math.isfinite(budget) or budget <= 0:
        raise ValueError("--budget-usd (this run's phase share of the global cap, > 0) is required with "
                         "--teacher max-payg")
    if budget > spend.DEFAULT_CAP_USD:
        raise ValueError(f"--budget-usd {budget:g} is above the ${spend.DEFAULT_CAP_USD:.2f} cap")
    if not 0 <= a.thinking_budget <= PAYG_MAX_THINKING_BUDGET:
        raise ValueError(f"--thinking-budget must be 0 (thinking off) .. {PAYG_MAX_THINKING_BUDGET}")
    if not 1 <= a.max_tokens <= PAYG_MAX_TOKENS:
        raise ValueError(f"--max-tokens must be 1 .. {PAYG_MAX_TOKENS} with --teacher max-payg (it bounds "
                         "every paid request's output)")
    if not 1 <= a.concurrency <= PAYG_MAX_CONCURRENCY:
        raise ValueError(f"--concurrency must be 1 .. {PAYG_MAX_CONCURRENCY} on the paid key")
    if not AP.path_has_tag(out):
        raise ValueError(f"--teacher max-payg writes Max-derived data: --out must carry the {AP.TAG} tag "
                         f"(e.g. --out {AP.tag_path(out)})")
    return {"teacher": TEACHER_PAYG, "model_name": PAYG_MODEL, "teacher_id": PAYG_TEACHER_ID,
            "teacher_served": PAYG_MODEL, "budget_usd": float(budget), "thinking_budget": int(a.thinking_budget)}


def settings_from_args(a: argparse.Namespace) -> Settings:
    payg = a.teacher == TEACHER_PAYG
    for flag, value in (("--model-name", None if payg else a.model_name), ("--judge-model", a.judge_model),
                        ("--teacher-id", a.teacher_id), ("--judge-id", a.judge_id)):
        if has_max_token(value):
            raise ValueError(f"{flag} {value!r} names a Max model; only --teacher max-payg uses one, "
                             "and only on the pay-per-token key")
    if a.tasks_file and not a.builtin_smoke:
        raise ValueError("--tasks-file holds built-in (list-tasks) rows and needs --builtin-smoke; "
                         "a real-pattern taskset goes to --tasks")
    if bool(a.tasks) == bool(a.builtin_smoke):
        raise ValueError("give exactly one of --tasks <taskset.jsonl> (real-pattern tasks: the only "
                         f"source of {SOURCE_TAG} / {PAYG_SOURCE} rows) or --builtin-smoke (built-in tasks, "
                         f"rows tagged {BUILTIN_SOURCE}, never trained on)")
    task_source = ORIGIN_TASKSET if a.tasks else ORIGIN_BUILTIN
    out = Path(a.out).expanduser().resolve()
    live = (Path.home() / ".prometheus").resolve()
    if out == live or live in out.parents:
        raise ValueError(f"--out must not be inside {live}")
    if payg:
        teacher = _payg_args(a, out)
    else:
        if a.budget_usd is not None or a.thinking_budget:
            raise ValueError("--budget-usd / --thinking-budget are for --teacher max-payg")
        teacher = {"teacher": TEACHER_LOCAL, "model_name": a.model_name or DEFAULT_MODEL_NAME,
                   "teacher_id": a.teacher_id or DEFAULT_TEACHER_ID}
    default_name = (BUILTIN_SOURCE if task_source != ORIGIN_TASKSET else PAYG_SOURCE if payg else SOURCE_TAG)
    pack_prompt, date_sentinel = load_pack_template(a.pack)
    return Settings(
        task_source=task_source, sampling=sampling_from_args(a),
        out=out, name=a.name or default_name, teacher_url=a.teacher_url or DEFAULT_TEACHER_URL,
        probe_guard_url=a.probe_guard_url,
        judge_url=a.judge_url, judge_model=a.judge_model,
        judge_seed=a.judge_seed, allow_lan=a.allow_lan_endpoints, concurrency=a.concurrency,
        until=parse_until(a.until, max_ahead_hours=UNTIL_HHMM_MAX_AHEAD_H), min_task_seconds=a.min_task_seconds,
        max_tasks=a.max_tasks,
        target_rows=a.target_rows, seed=a.seed, mix=parse_mix(a.mix), coding_cap=a.coding_cap,
        max_tokens=a.max_tokens, max_tool_iterations=a.max_tool_iterations, tool_timeout=a.tool_timeout,
        task_timeout=a.task_timeout,
        request_timeout=a.request_timeout or (600.0 if a.probe_guard_url else 300.0),
        compaction_history_tokens=a.compaction_history_tokens, emit_final_turns=a.emit_final_turns,
        max_natural_errors=a.max_natural_errors, keep_refused=a.keep_refused,
        sandbox_root=Path(a.sandbox_root).expanduser(), base_system_prompt=pack_prompt,
        date_sentinel=date_sentinel,
        repo=Path(a.repo).expanduser(), max_consecutive_errors=a.max_consecutive_errors,
        progress=not a.quiet, jail=a.jail, judge_id=a.judge_id, **teacher)


#: Exit codes (module docstring).
EXIT_DONE, EXIT_NOTHING, EXIT_USAGE, EXIT_UNREACHABLE, EXIT_SPEND_STOP, EXIT_PAYG_REFUSED = 0, 1, 2, 3, 4, 5
#: An unexpected exception (a traceback on stderr): never Python's own 1, which means "kept nothing".
EXIT_CRASH = 6
SANDBOX_LOCK = ".teacher_gen.lock"


class SandboxBusy(Exception):
    """Another teacher_gen run holds this ``--sandbox-root``."""


def lock_sandbox_root(root: Path) -> int:
    """An exclusive lock on ``root`` for this process's life (an fd the caller closes): two runs sharing a
    sandbox root would wipe each other's ``w<N>/workspace`` at every task start and end, and a row could
    carry the other run's files. A second run on the same root is refused (SandboxBusy)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / SANDBOX_LOCK, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SandboxBusy(f"--sandbox-root {root} is in use by another teacher_gen run "
                          "(give each concurrent run its own)") from None
    return fd


def payg_client_for(settings: Settings, a: argparse.Namespace, opener: Any = None) -> Any:
    """The paid client, after every check a paid run makes before its first call:
    the key (QWEN_PAYG_KEY only: never the Token Plan key or host, its prefix
    matching its host), the one canonical ledger, no STOP file, a notify command
    that can reach the operator. Raises spend.SpendError (the run is refused, exit 5).
    Runs before the state directories are isolated (that repoints PROMETHEUS_ENV_FILE)."""
    spend = _spend()
    creds = spend.resolve_payg(env_file=Path(a.env_file).expanduser() if a.env_file else None)
    ledger = spend.Ledger(notify_cmd=a.notify_cmd)
    ledger.check_open(tool=PAYG_TOOL, phase=PAYG_PHASE, model=PAYG_MODEL)
    notify = Path(a.notify_cmd).expanduser()
    if not (notify.is_file() and os.access(notify, os.X_OK)):
        raise spend.SpendError(f"the notify command {spend.display_path(notify)} is missing or not executable: "
                               "the $140 alert could not reach the operator")
    client = spend.PaygClient(creds, ledger, tool=PAYG_TOOL, phase=PAYG_PHASE, model=PAYG_MODEL,
                              phase_budget_usd=settings.budget_usd, timeout=settings.request_timeout,
                              opener=opener)
    log.warning("teacher: %s on the %s; %s", PAYG_MODEL, creds.describe(),
                ledger.describe(tool=PAYG_TOOL, phase=PAYG_PHASE, phase_budget_usd=settings.budget_usd))
    return client


#: The process exit a stop signal ends in (tests replace it).
_hard_exit = os._exit
_SIGNAL_STOPPING: list[int] = []


def payg_stop_on_signal(client: Any, signum: int) -> None:
    """A paid run's SIGTERM / SIGHUP (gen_runner.py at --until or a stop, a systemd stop): start no
    new paid call, give the calls in flight up to ``PAYG_SIGNAL_WAIT_S`` to settle from their usage,
    settle the rest as interrupted (charged their reservation), then exit 128 + *signum* while
    holding the ledger lock, so no worker thread is half way through a ledger row. Never returns.
    Without this the default action would kill the process mid-call and leave its reservations
    open in the ledger for good. The task in flight is not recorded (it reruns next time)."""
    if _SIGNAL_STOPPING:  # a second signal while the first one is handled
        return
    _SIGNAL_STOPPING.append(signum)
    ledger = None
    try:
        client.close(f"signal {signum}")
        end = time.monotonic() + PAYG_SIGNAL_WAIT_S
        while client.open_calls() and time.monotonic() < end:
            time.sleep(0.05)
        settled = client.settle_open()
        print(f"stopped by signal {signum}: no new paid call; {len(settled)} paid call(s) still in flight "
              "settled as interrupted (charged their reservation)", file=sys.stderr, flush=True)
        ledger = copy.copy(client.ledger)
        ledger.lock_timeout = min(float(ledger.lock_timeout), 5.0)
    except Exception as exc:  # noqa: BLE001 - the exit below must happen
        print(f"stopped by signal {signum}: {type(exc).__name__} while settling paid calls in flight",
              file=sys.stderr, flush=True)
    try:
        if ledger is not None:
            with ledger.locked():
                _hard_exit(128 + signum)
    except Exception:  # noqa: BLE001 - the lock stayed busy: exit anyway
        pass
    _hard_exit(128 + signum)


def install_payg_signal_handlers(client: Any) -> Callable[[], None]:
    """``payg_stop_on_signal`` for ``PAYG_EXIT_SIGNALS``; returns the function that puts the
    previous handlers back. Only the main thread can install handlers (else: nothing)."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    _SIGNAL_STOPPING.clear()
    previous = {sig: signal.signal(sig, lambda signum, _frame: payg_stop_on_signal(client, signum))
                for sig in PAYG_EXIT_SIGNALS}

    def restore() -> None:
        for sig, old in previous.items():
            signal.signal(sig, old if old is not None else signal.SIG_DFL)

    return restore


def cmd_run(a: argparse.Namespace, *, payg_opener: Any = None) -> int:
    try:
        settings = settings_from_args(a)
        if not settings.payg:
            settings.teacher_endpoint  # noqa: B018 - validates the URLs before anything runs
        settings.judge_endpoint  # noqa: B018
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    missing = [pat for pat in a.eval_tools if not expand([pat])]
    if missing:
        print(f"error: --eval-tools matched no file: {', '.join(missing)} (disjointness cannot be checked)",
              file=sys.stderr)
        return EXIT_USAGE
    if settings.payg:
        spend = _spend()
        try:
            settings.payg_client = payg_client_for(settings, a, payg_opener)
        except spend.SpendError as exc:  # messages never hold the key or the host
            print(f"error: --teacher max-payg refused before any call: {exc}", file=sys.stderr)
            return EXIT_PAYG_REFUSED
    isolate_state_dirs(settings.out / "work")
    try:
        guard = EvalGuard.load(a.injection_manifest, a.eval_tools)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: eval sets: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        jail = resolve_jail(settings.jail)
    except EndpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if jail == "none":
        log.warning("no OS jail: sandboxed shell commands are confined by the command checks alone")
    gen: Generator
    if a.tasks:
        # the only path to train-source rows: real-pattern tasks, validated by taskset_gen.py
        try:
            tasks, settings.pad_pool = load_taskset_with_pool(a.tasks, settings.repo, guard=guard)
            gen = FixedGenerator(tasks)
        except (ValueError, OSError, KeyError) as exc:
            print(f"error: --tasks: {exc}", file=sys.stderr)
            return EXIT_USAGE
        if settings.pad_pool.get("reserved"):
            log.warning("pad pool: %d rows in %d task groups are padding only (never run as tasks)",
                        settings.pad_pool["pool_rows"], settings.pad_pool["pool_groups"])
    elif a.tasks_file:
        try:
            gen = FixedGenerator(load_builtin_tasks(a.tasks_file))
        except (ValueError, OSError, KeyError) as exc:
            print(f"error: --tasks-file: {exc}", file=sys.stderr)
            return EXIT_USAGE
    else:
        gen = Generator(Sources.from_repo(settings.repo), settings.seed)
    if settings.task_source != ORIGIN_TASKSET:
        log.warning("--builtin-smoke: built-in tasks; rows are tagged %s and never train", BUILTIN_SOURCE)

    async def check_servers() -> None:
        if not settings.payg:
            served = await preflight(settings.teacher_endpoint, min(30.0, settings.request_timeout))
            settings.teacher_served = check_model_id(served, settings.teacher_id, "teacher")
            log.info("teacher serves %s", served)
        if settings.judge_endpoint and (settings.payg or settings.judge_endpoint != settings.teacher_endpoint):
            judged = await preflight(settings.judge_endpoint, min(30.0, settings.request_timeout))
            check_model_id(judged, settings.judge_id or (DEFAULT_TEACHER_ID if settings.payg
                                                         else settings.teacher_id), "judge")

    try:
        asyncio.run(check_servers())
    except EndpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    try:
        # the fingerprint holds the teacher's id: one output never mixes two teachers
        store = Store(settings.out, settings.name, settings.fingerprint(), settings.coding_cap, guard)
    except SettingsMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    # a paid run that is stopped settles its calls in flight instead of leaving them open
    restore_signals = install_payg_signal_handlers(settings.payg_client) if settings.payg else (lambda: None)
    try:
        report = asyncio.run(run(settings, gen, store, guard))
    except EndpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNREACHABLE
    finally:
        restore_signals()
    summary = {"report": str(store.report_path), "rows_total": report["rows_total"],
               "stop": report.get("stop"), "coding_share": report["coding_share"],
               "by_kind": {k: {"rows": v["rows"], "pass_rate": v["pass_rate"]}
                           for k, v in report["by_kind"].items()}}
    if settings.payg:
        summary["spend"] = report.get("spend")
    print(json.dumps(summary, indent=1))
    stop = report.get("stop")
    if stop == "spend_stop":
        print(f"stopped: the spend ledger refused a paid call: {report.get('stop_detail', '')}", file=sys.stderr)
        return EXIT_SPEND_STOP
    if stop == "payg_fatal":
        print(f"stopped: the paid API refused the key or the account: {report.get('stop_detail', '')}",
              file=sys.stderr)
        return EXIT_PAYG_REFUSED
    if stop == "consecutive_errors":
        return EXIT_UNREACHABLE
    return EXIT_DONE if report["rows_total"] else EXIT_NOTHING


def cmd_list_tasks(a: argparse.Namespace) -> int:
    guard = None
    if a.injection_manifest or a.eval_tools:
        guard = EvalGuard.load(a.injection_manifest, a.eval_tools or [])
    gen = Generator(Sources.from_repo(Path(a.repo)), a.seed)
    kinds = [a.kind] if a.kind else list(KINDS)
    for kind in kinds:
        fams = gen.families(kind)
        emitted, idx = 0, collections.Counter()
        k = 0
        while emitted < a.n * len(fams) and k < a.n * len(fams) * 20:
            fam = fams[k % len(fams)]
            k += 1
            task = gen.make(fam, idx[fam])
            idx[fam] += 1
            if guard and guard.task_overlaps(task):
                continue
            print(json.dumps(task.to_dict(), ensure_ascii=False))
            emitted += 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="generate verified teacher traces")
    r.add_argument("--out", required=True)
    r.add_argument("--name", help=f"output file prefix, unique across build inputs (default {SOURCE_TAG} "
                                  f"with --tasks, {BUILTIN_SOURCE} with --builtin-smoke)")
    src = r.add_argument_group("task source (exactly one)")
    src.add_argument("--tasks", metavar="TASKSET.jsonl",
                     help=f"a taskset_gen.py real-pattern taskset: the ONLY source of {SOURCE_TAG} rows")
    src.add_argument("--builtin-smoke", action="store_true",
                     help=f"run the built-in task families (or --tasks-file); rows are tagged {BUILTIN_SOURCE} "
                          "and never train")
    r.add_argument("--teacher", choices=TEACHERS, default=TEACHER_LOCAL,
                   help=f"local: a loopback llama-server (default); {TEACHER_PAYG}: {PAYG_MODEL} on the "
                        "pay-per-token key (QWEN_PAYG_KEY) through spend.py, rows tagged "
                        f"{PAYG_SOURCE} (Max-derived)")
    r.add_argument("--teacher-url", help=f"local teacher (default {DEFAULT_TEACHER_URL})")
    r.add_argument("--probe-guard-url", help="local guard proxy carrying every local model request")
    r.add_argument("--judge-url", help="another LOCAL endpoint for the judge (default: the teacher; with "
                                       f"--teacher {TEACHER_PAYG}, the local 27B judge, preferred)")
    r.add_argument("--model-name", help=f"request model + _meta.model (default {DEFAULT_MODEL_NAME}; "
                                        f"{PAYG_MODEL} with --teacher {TEACHER_PAYG})")
    r.add_argument("--judge-model")
    paid = r.add_argument_group(f"--teacher {TEACHER_PAYG} (every call reserved and settled in the one "
                                "spend ledger: $150 cap, STOP at $140)")
    paid.add_argument("--budget-usd", type=float,
                      help=f"required: this phase's ({PAYG_TOOL}/{PAYG_PHASE}) share of the global cap")
    paid.add_argument("--thinking-budget", type=int, default=0,
                      help="reasoning tokens per paid request (0 = thinking off, as at runtime)")
    paid.add_argument("--env-file", help="the 0600 env file holding QWEN_PAYG_KEY (default: Prometheus's own)")
    paid.add_argument("--notify-cmd", default=_spend().DEFAULT_NOTIFY,
                      help="gets the $140 STOP alert (message on stdin); must exist and be executable")
    r.add_argument("--judge-seed", type=int, default=DEFAULT_JUDGE_SEED)
    r.add_argument("--allow-lan-endpoints", action="store_true")
    r.add_argument("--injection-manifest", required=True)
    r.add_argument("--eval-tools", action="append", required=True, help="eval-tools JSONL (glob; repeat)")
    r.add_argument("--pack", help="build.py request pack: its system prompt is the teacher's")
    r.add_argument("--repo", default=str(_REPO_DEFAULT), help="docs source (git-tracked docs/, wiki/, README.md)")
    r.add_argument("--tasks-file", help="with --builtin-smoke: run exactly these built-in tasks (list-tasks JSONL)")
    smp = r.add_argument_group("explicit sampling (sent on every teacher request; default: the server's own)")
    smp.add_argument("--temperature", type=float)
    smp.add_argument("--top-k", dest="top_k", type=int)
    smp.add_argument("--top-p", dest="top_p", type=float)
    smp.add_argument("--min-p", dest="min_p", type=float)
    r.add_argument("--concurrency", type=int, default=2)
    r.add_argument("--until", help="HH:MM (local, next occurrence; refused when that is more than 12 h "
                                   "ahead) or ISO time; a hard end")
    r.add_argument("--min-task-seconds", type=float, default=120.0)
    r.add_argument("--max-tasks", type=int)
    r.add_argument("--target-rows", type=int)
    r.add_argument("--seed", default=DEFAULT_SEED)
    r.add_argument("--mix", help="kind=share,... (default " + ",".join(f"{k}={v}" for k, v in DEFAULT_MIX.items()) + ")")
    r.add_argument("--coding-cap", type=float, default=CODING_CAP)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--max-tool-iterations", type=int, default=12)
    r.add_argument("--tool-timeout", type=float, default=60.0)
    r.add_argument("--task-timeout", type=float, default=900.0)
    r.add_argument("--request-timeout", type=float)
    r.add_argument("--compaction-history-tokens", type=int, default=1500)
    r.add_argument("--emit-final-turns", action="store_true")
    relaxed = ("a task verified only because of it is 'relaxed': its rows go to <name>.relaxed.jsonl "
               "(source qwen38-27b-gym-relaxed, never train rows) and count toward no target, mix or cap")
    r.add_argument("--max-natural-errors", type=int, default=0, help="tolerate this many tool errors; " + relaxed)
    r.add_argument("--keep-refused", action="store_true", help="tolerate sandbox refusals; " + relaxed)
    r.add_argument("--sandbox-root", default=str(DEFAULT_SANDBOX_ROOT))
    r.add_argument("--max-consecutive-errors", type=int, default=5)
    r.add_argument("--jail", choices=JAIL_MODES, default="auto",
                   help="OS jail for shell commands (bwrap / sandbox-exec): auto uses one when it works, "
                        "require refuses to run without one")
    r.add_argument("--teacher-id",
                   help=f"glob the local teacher's /v1/models id must match (default {DEFAULT_TEACHER_ID}; "
                        "case-insensitive; never a Max model)")
    r.add_argument("--judge-id", help="glob for a --judge-url server's id (default: --teacher-id)")
    r.add_argument("--quiet", action="store_true")
    ls = sub.add_parser("list-tasks", help="print the tasks a run would start, no model calls")
    ls.add_argument("--n", type=int, default=2, help="tasks per family")
    ls.add_argument("--kind", choices=KINDS)
    ls.add_argument("--seed", default=DEFAULT_SEED)
    ls.add_argument("--repo", default=str(_REPO_DEFAULT))
    ls.add_argument("--injection-manifest")
    ls.add_argument("--eval-tools", action="append")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, payg_opener: Any = None) -> int:
    """*payg_opener* (tests): the paid client's opener (spend.PaygClient's ``opener``)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    if a.cmd == "run":
        try:
            lock = lock_sandbox_root(Path(a.sandbox_root).expanduser())
        except (SandboxBusy, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        try:
            return cmd_run(a, payg_opener=payg_opener)
        finally:
            os.close(lock)
    return cmd_list_tasks(a)


def cli(argv: list[str] | None = None) -> int:
    """The command line: ``main``, with an unexpected exception reported as ``EXIT_CRASH`` (its
    traceback on stderr), so a supervisor never mistakes a crash for exit 1 ("kept nothing")."""
    try:
        return main(argv)
    except Exception:  # noqa: BLE001 - any crash: the exit code must say so
        traceback.print_exc()
        print(f"error: teacher_gen crashed (exit {EXIT_CRASH}); a rerun resumes from the ledger", file=sys.stderr)
        return EXIT_CRASH


if __name__ == "__main__":
    sys.exit(cli())
