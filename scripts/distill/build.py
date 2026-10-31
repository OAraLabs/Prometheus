#!/usr/bin/env python3
"""Phase 3 dataset builder: train / val / eval-tools from the teacher's golden traces.

Runs on the host that holds the traces. Nothing it writes leaves that host
except ``report.json``, which holds aggregates only (counts, shares, token
percentiles, gate results: no text, no ids, no hashes).

    # 1. Once per deployed build: the request pack (system prompt + tool list),
    #    made by Prometheus's own prompt builder and tool registry.
    PYTHONPATH=~/prometheus-deploy/src python3 build.py pack \\
        --config ~/prometheus-deploy/config/prometheus.yaml --cwd ~/prometheus-deploy \\
        --out ~/distill/phase3/request_pack.json
    # 2. The build. --dry-run computes everything and writes report.json only.
    PYTHONPATH=~/prometheus-deploy/src nice -n 19 ionice -c3 python3 build.py build \\
        --index ~/distill/phase1/trace_index.jsonl --pack ~/distill/phase3/request_pack.json \\
        --tokenizer-dir ~/distill/models/qwen3.5-9b-tokenizer --out ~/distill/phase3 \\
        [--verdicts judge-sample.jsonl ...] [--fail-traces ~/distill/phase2/fail_traces.jsonl] \\
        [--extra-traces '~/distill/gym/*.jsonl'] [--train-sources qwen38-27b-gym] \\
        [--train-models 'qwen3.8-27b*'] [--judge-models 'qwen3.8-27b*'] \\
        [--lcm-snapshot ~/distill/snapshot/lcm.db] \\
        [--spot-check-server http://127.0.0.1:8080] [--dry-run]
    # 3. Spot check (needs llama-server serving the student GGUF with --jinja):
    python3 build.py spot-check --server http://127.0.0.1:8080 \\
        --rows ~/distill/phase3/train.jsonl --tokenizer-dir ~/distill/models/qwen3.5-9b-tokenizer

Inputs
------
* Golden traces (``--traces``, default ``~/.prometheus/trajectories/golden_traces_*.jsonl``),
  read-only, in file-name order. Each line: ``messages`` (context + the
  teacher's assistant turn with ``tool_calls``), ``tools`` (the called tool's
  schema), ``_meta`` (model, tool_name, timestamp, session_id).
* Phase 1 ``trace_index.jsonl`` (same order): telemetry facts the traces lack
  (previous call failed, turn outcome). Hashes and categories are recomputed
  here with the Phase 1 functions (ported below) so later additions are treated
  the same way; disagreements with the index are counted.
* ``--verdicts``: judge JSONL. ``judge.py``'s ``judge-sample.jsonl`` rows are
  read as it writes them: only ``status: ok`` rows count, the verdict is
  ``verdict`` (never the sampled ``label``), the trace is ``source.file`` +
  ``source.line`` (judge.py's 0-based line), and ``source.telemetry_rowid`` is
  checked against the Phase 1 index; fail-side rows (no file) are skipped.
  Other rows are keyed by ``trace_id`` (``<file name>:<line>``, 1-based) or
  ``h_exact``, verdict in ``verdict`` / ``label`` (``pass``|``fail``) or a
  boolean ``pass``. Later files override earlier ones (a full re-judge after
  the sample). A verdict row that is unusable, matches no trace or disagrees
  with the index fails the ``verdicts`` gate. So does a verdict whose judge
  model names Max (judge.py's ``model``; ``judge_model`` / ``model`` on other
  rows): the Max key never judges. Such a pass is not applied; such a fail
  still is (dropping a row is always safe). Unjudged golden traces fall back
  to the tool-success label (the runbook's rule): every golden trace's call ran
  without error, by construction.
* Input file names must be unique across ``--traces``, ``--extra-traces`` and
  ``--fail-traces`` (a trace is ``<file name>:<line>``); a repeat is an input
  error.
* ``--extra-traces``: teacher traces in the same shape, e.g. ``teacher_gen.py``
  output (tagged ``_meta.source = "qwen38-27b-gym"``) or older Max gym rows
  (``max-gym-<date>``). Each must be verified: a pass verdict (in
  ``--verdicts`` or ``_meta.verdict``) or a passing ``_meta.verification``
  record (below); unverified rows are excluded. A train-source row (data
  policy, below) is verified only by its own record, read strictly: a pass
  verdict alone never verifies it. They are deduped against
  everything before them and may carry ``_meta.category``, ``_meta.task_id``
  (the generator's task: every row of one task lands in one split),
  ``_meta.suppress_tools`` (chat-mode), ``_meta.compacted``,
  ``_meta.error_recovery``. A missing ``_meta.tool_name`` is read from the
  target's first call.
* ``_meta.verification`` (``teacher_gen.py``'s record), honoured on any row::

      {"verified": true,                 # the generator's decision (or
                                         #  "status"/"verdict": "pass"|"fail")
       "tool_success": true,
       "judge": {"verdict": "pass", "model": "<local judge>"},
       "sandboxed": true,
       "side_effects": false}            # (or "real_side_effects")

  Every part is read fail-closed: ``tool_success`` and ``sandboxed`` pass
  only as exactly ``true``, the side-effect fields only as ``false`` or empty
  (``[]``, ``{}``, ``""``); any other value (``"false"``, ``0``, a list of
  effects) fails the row. A judge must give a pass verdict (a judge record
  with no verdict, or with ``pass`` not ``true``, fails), and a judge whose
  model names Max fails the row (the Max key never judges). On a Max row
  (eval-only) a part may be absent. On a train-source row every part is
  required: tool success, a passing judge whose model matches
  ``--judge-models`` (default: the ``--train-models``), sandboxed, and no side
  effects; a missing part fails it with its own reason. A failing record
  excludes the row even beside a pass verdict, and a fail verdict excludes it
  even beside a passing record. A train-source row claims its dedupe key only
  once verified (one task's retries share their first call), a Max row as
  soon as it is read (its verdicts are per call).
* ``--fail-traces``: rebuilt failed teacher calls (same shape), used only as
  hard eval-tools cases, never for training. A row may carry its own
  ``expect``; otherwise the case expects the tool the teacher picked, called
  with schema-valid arguments.
* ``--lcm-snapshot``: a READ-ONLY snapshot of the conversation store (never the
  live file). When given, each trace's context is rebuilt from the structured
  ``content_json`` blocks (earlier tool calls and tool results included) up to
  the teacher's own assistant message; otherwise the stored ≤12-message
  window is used. ``context_source`` on each row says which.

Data policy (``--train-sources``)
---------------------------------
Only rows whose own ``_meta.source`` tag matches a ``--train-sources`` pattern
(fnmatch; default ``qwen38-27b-gym``) may enter train or val (a ``builtin-smoke``
row, from teacher_gen.py's hand-written task families, never does, whatever the
patterns: it is excluded as ``non_teacher_model``, or ``never_train_source`` when
the ``--max-model`` wrote it, so it is neither train nor eval data). Such a row needs
a ``_meta.model`` that matches a ``--train-models`` pattern (the local teacher;
case-insensitive fnmatch, default ``qwen3.8-27b*``; otherwise
``train_source_wrong_model``) and does not name Max (nor may its
``served_model``: ``train_source_names_max``), no ``_meta.max_derived`` (or
record ``max_derived``) other than ``false`` (``train_source_max_derived``),
and a passing strict verification record (above; a train-source row is never
trained on a verdict alone or on the tool-success fallback). Every other row the build
keeps is Max-derived: the ``--max-model``'s rows (default ``qwen3.8-max``, also
``--teacher-model``), a tag naming Max, or no tag at all. Max-derived rows,
the failed calls included, are eval-tools material only: a group holding one
goes to eval-tools whole (so a train-source row sharing its task or context is
held out with it, and the Max part of eval-tools stays put as train data
grows), and the eval-tools quota (``--eval-task-frac``) is taken from the
remaining train-source groups. Train/val rows carry ``source``, ``model`` and
``max_derived``; the report's ``data_policy`` states the rule with per-source
counts (input, exclusions by reason, verification outcomes, models, rows per
split), and the ``data_policy`` gate fails on any Max-derived, unlisted or
wrong-model row in train/val. A source, train model or judge model naming Max
cannot be listed (exit 2). The size, val-share, category and presence gates
count the listed sources' rows only; leakage is checked across every source in
every split. The report's ``settings.teacher_model`` is what the trainer's and
the flywheel's Max guards read: the one model whose rows train/val hold when
that is a single model that does not name Max, else the ``--max-model`` (the
guards then treat every row as Max-derived, which fails closed).

Max-derived training (``--allow-max-derived``, DISTILL-OPEN-QUESTIONS Q6b)
--------------------------------------------------------------------------
The operator may let Max-derived rows train a PRIVATE model. It is an explicit,
recorded allow, never the default: ``--allow-max-derived`` and
``--max-derived-approval "<ref>"`` go together (one without the other is a usage
error, exit 2), and the reference (e.g. ``"operator Q6b 2026-10-30"``) is
recorded verbatim in ``report.json`` (``settings``, ``data_policy.max_derived_allow``)
and ``dataset-manifest.json``. Without them nothing changes: Max rows are
eval-only. With them a Max row (the ``--max-model``'s: harvested golden traces,
untagged or Max-tagged, and Max gym rows) may enter train/val under the same
gates as any row: dedupe, leakage, category caps, and verification -- a golden
trace has its tool-success label (it ran without error, by construction), an
extra (gym) Max row needs the strict record a train-source gym row needs (tool
success, a passing judge of the ``--judge-models`` or the pay-per-token Max
judge, sandboxed, no side effects; otherwise it stays eval-only, counted under
``max_derived_eval_only:<why>``), a ``relaxed`` tag never trains, and a verdict
on the row must pass (a fail verdict or a failing record excludes it). Failed
calls stay hard cases only. Each such row carries ``max_derived: true`` and the
approval (``max_derived_approval``), and the ``data_policy`` gate passes only if
every Max row in train/val is covered by the recorded approval.

Max teacher gym rows (``_meta.source = "max-payg-gym"``: ``teacher_gen.py
--teacher max-payg``, Qwen Max on the pay-per-token key over the real-pattern
taskset) are Max-derived and are admitted ONLY under the allow. Without it every
such row is excluded (``max_payg_gym_needs_allow``): never train, val or eval
material, so they pin no group and leave the local 27B rows' splits as they
were. Under it a row is kept only as an ``--extra-traces`` row
(``max_payg_gym_not_extra``), not relaxed (``max_payg_gym_relaxed``), saying
``max_derived: true`` (``max_payg_gym_not_max_derived``), with a record that
passes strictly: tool success, a passing judge that is a ``--judge-models``
judge (the local 27B, preferred) or the pay-per-token Max judge (``provider:
max-payg``, ``judge_key: payg``), sandboxed, no side effects
(``max_payg_gym_unverified:<why>`` otherwise). It claims its dedupe key once
verified (as a gym row), trains under the approval like any Max row, and counts
as ``max`` in ``data_policy.share`` (its own entry in ``data_policy.sources``).
It is left out of the Max-only hold-out plan (which stays the plan over the
harvested Max rows) and its groups may give eval quota, as gym groups.

The allow also needs ``--exclude-sessions FILE`` (session ids, one per line):
the 2026-10-13 Token-Plan incident session (DISTILL-OPEN-QUESTIONS #6) is
excluded from any training data. Every row of a listed session (``_meta.session_id``)
is dropped before anything else (``excluded_session``), and so is a gym row built
from any user message such a session holds (in the input traces or the
``--lcm-snapshot``): its ``_meta.task_id``, ``partner_task_id`` or a
``compaction_pad.task_groups`` entry is ``tg-`` + that message's task proxy
(``excluded_session_task``; gym rows carry no session id). The report records how
many sessions were listed and rows dropped (``data_policy.max_derived_allow``),
never the ids or hashes. The list may be given without the allow too.

Judges. The Token Plan key never judges: a verdict or record whose judge names
Max is refused (``verdict_rows_max_judge`` fails the verdicts gate;
``judge_is_max``). The one exception is the pay-per-token Max judge (the
operator's PAYG ruling: Phase 2 judging): a ``judge.py`` row with ``provider:
max-payg`` and ``judge_key: payg`` is applied like a local judge's verdict and
counted in ``verdict_rows_payg_max_judge`` (``gates.verdicts.payg_max_judge``,
never a failure), and a Max row's record whose ``judge`` has ``provider:
max-payg`` verifies it. A train-source (local 27B) row still needs a
``--judge-models`` judge.

The eval-tools hold-out stays Max-derived eval data: the Max groups a Max-only
build holds out (``plan_splits`` over the Max rows alone, same seed and
fractions: the Phase 3 eval-tools tasks, given the same inputs) and the tasks
named in ``--eval-holdout`` case files (e.g. the Phase 3 ``eval-tools.jsonl``)
are pinned to eval-tools whole; the eval quota for the rest is taken from the
train-source groups only. Failed calls do not pin their groups under the allow
(as in a Max-only build, a hard case whose task trains is dropped).

Everything such a build writes is tagged (``artifact_policy.py``): the output
directory's name must carry ``max-derived`` (``--out .../phase3-max-derived``,
else exit 2; a ``--report`` path too), and ``report.json`` and
``dataset-manifest.json`` carry ``max_derived: true`` and ``tags: ["max-derived"]``
whenever train/val hold a Max row. ``data_policy.share`` gives rows and percent
per class (``max`` / ``qwen38-27b-gym`` / ``other``) per split with the approval
reference, and each ``data_policy.sources`` entry its own share per split.
Max-derived data and models are private: ``publish_guard.py`` refuses them and
``modal_train.py`` never rents them.

Rebuilding the request
----------------------
Each example is the request ``LlamaCppProvider._build_request_payload`` would
send for a local run of the student, followed by the teacher's turn:

* System prompt: ``build_runtime_system_prompt`` (the daemon's call:
  ``project_files=False``, ``skills_for_prompt()``), captured by ``pack``.
* Tools: every tool ``create_tool_registry`` registers with the host config's
  ``security`` section, in the catalog a local run sends: ``stubs`` (the
  default ``tools.deferred_loading``: ten tools in full, the rest as
  skeletons, sorted) or ``full`` (every schema in full). ``--catalog auto``
  follows the config, as the runtime does. The provider maps them to the
  OpenAI function format.
* Messages: ``render_messages_for_model`` (untrusted banners, adjacent
  assistants merged), then the provider's own message builder; the result is
  redacted with ``redact_capture`` like the trace exporter does.

What cannot be reconstructed (every row is marked ``approximate: true`` with
the codes below in ``approx``):

* ``bootstrap_now``: SOUL.md / AGENTS.md / ANATOMY.md and the skill list as
  they are on the host today, not on the trace's date.
* ``no_memory``: the "# Memory" section (MEMORY.md + USER.md) is left out. It
  is per-day content that cannot be recovered for past dates, and baking the
  owner's personal facts into weights is not wanted.
* ``no_saved_files``: the saved-files listing is left out (per-day content).
* ``env_now``: the environment section is today's (OS, shell, cwd, git branch,
  host) except the date, which is set to the trace's UTC date.
* ``no_per_run``: per-run additions are absent: a session workspace swap
  (``# Workspace`` + its project files), passive memory recall, mid-turn
  steers and retry/periodic nudges.
* ``tools_now``: the current registry's schemas (they may have drifted since
  the trace); daemon-time additions (LSP, MCP servers, packs) and profile
  filters are not applied.
* ``stored_context`` (no snapshot): the stored window is prose only, capped at
  12 messages, with no earlier tool calls or results and no provenance (so no
  untrusted banners). ``lcm_full_history`` (snapshot): the whole history up to
  ``--lcm-max-messages`` rows, without the compactor's render view (the long
  start is cut from the left instead). The snapshot row holding the teacher's
  call is the turn learned, every parallel call in it included, so a row where
  any of its calls returned an error is excluded (``failed_sibling_call``).
  A request whose last message is an assistant turn, or whose history holds a
  tool call not answered by the tool messages right after it, is not a request
  the runtime sends (llama-server would treat a trailing assistant message as a
  prefill) and is excluded (``target_after_assistant``,
  ``unanswered_tool_call``).
* ``teacher_route``: the teacher ran on a hosted route, which sees a
  different system prompt (no infrastructure section) and full schemas; the
  rebuilt request is the local route's.

Rendering, length, labels
-------------------------
Rendered with the student's chat template (``render.py``): exact text,
assistant-only loss span (the teacher's turn through ``<|im_end|>``), token
offsets from the student tokenizer, thinking off (the live default). An
example over ``--max-tokens`` (16,384) keeps its system prompt, tools, last
user query and the turn being learned, and loses its oldest context: earlier
conversation first, then (unless ``--no-split-turns``) the oldest steps of the
current agentic turn, so one long turn becomes a window over its latest steps
(``truncated.dropped_turn_steps`` on the row). The untruncated original
also goes to ``eval-longctx-candidates.jsonl`` (with its task's split, so the
long-context builder can avoid train tasks). An example whose request cannot
fit even then is only a long-context candidate. With a
snapshot, history more than ``--prewindow-chars-per-token`` x the budget in
characters back is not rendered (no text runs above ~6 characters per token,
so it cannot survive the cut); its length is estimated at Phase 1's 3.93
characters per token, and the candidate row names the snapshot row so the
long-context builder can rebuild the whole history.

Splits and gates
----------------
* Task = the Phase 1 proxy: hash of the normalized last user prompt. Tasks
  are joined into one group (union-find) when they share a context-prefix hash
  (first ``--prefix-n`` non-system messages of the final request) or when one
  example's final request holds a user message of the same session that
  another example's request also holds (its own query included), so a task
  whose query, calls and results sit in another example's window, and the
  steps of one turn whose rebuilt context ends on a different query than the
  stored prompt, land in one group. Groups are
  assigned whole: seeded order, eval-tools gets ``--eval-task-frac`` of the
  tasks, val ~``--val-frac`` of the remaining rows, the rest is train. Groups
  larger than ``--holdout-max-group-share`` of the pool stay in train.
* eval-tools: held-out tasks (at most ``--eval-max-per-task`` rows each,
  seeded) + fail traces whose group is not in train/val, in score.py's
  eval-tools schema. Cases carry the same <= ``--max-tokens`` request that was
  grouped and leak-checked (the long originals are long-context material); a
  fail trace whose request cannot fit is excluded (``fail_unfittable``). Fail
  traces skip the training-only argument checks (a missing required argument
  is often why the call failed) and expect their own call's tool.
  ``val-cases.jsonl`` is the val split in score.py's val schema.
* Flags describe the context that survived the cut: ``error_recovery`` = a
  tool result after the last user query, still in the request, that is an
  error. ``error_signal`` (telemetry or ``_meta`` says a call failed nearby)
  is reported but not gated.
* Chat mode is a request that offered no tools: ``_meta.suppress_tools`` /
  ``chat_mode`` / ``tool_choice: "none"`` / ``mode: "chat"``, or a trace with
  no ``tools`` and no call. It is rebuilt the way the runtime sends it (no
  tools, no ``tool_choice``), so the student template renders no tools block,
  and the row counts toward the chat-mode presence gate. A turn that answers
  in prose while tools were offered is not chat mode: it keeps the catalog
  (``final_answer`` flag). A tool call in a chat-mode turn is excluded
  (``call_in_chat_mode``): no tool was offered to call.
* Category cap: no category above ``--category-cap`` (35%) of train; the
  largest fixed point is kept, round-robin over tasks so task diversity
  survives; val keeps the same per-category share train kept, so it mirrors
  train's mix. What was dropped is reported.
* Gates (in code; FAIL is reported, never worked around): no task / group /
  prefix / exact / prompt+call / generator-task hash, and no session user
  message of a final request, in two splits (whatever the rows' sources);
  every verdict row applied; nothing but the listed train sources in
  train/val (``data_policy``); train size (listed sources) in
  ``--train-range``; val share in ``--val-share-range``; eval-tools at least
  ``--eval-min``; max train category share ≤ cap; chat-mode, error/recovery and
  compacted-history examples present in train. The byte-for-byte spot check
  needs a server: ``--spot-check-server`` runs it on the written train.jsonl
  as the last gate; without it the gate reads NOT RUN and a build whose other
  gates pass is ``INCOMPLETE`` (exit 1), never ``PASS``. The standalone
  ``spot-check`` command prints the same comparison row by row.
* ``spot-check`` compares, for each sampled row, our render of the request
  with llama-server's (``/apply-template``); for a training row also the
  stored ``text`` up to its loss span with that render, and the stored
  ``text`` (the loss-bearing turn included) with the server's render of the
  row followed by a stub result per call (a trailing assistant turn would be
  a prefill, so a turn without calls is reported as not checkable).

Exit: 0 all gates pass (the spot check included), 1 a gate failed or the
build is INCOMPLETE, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import glob
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

PACK_SCHEMA = "distill-request-pack/1"
REPORT_SCHEMA = "distill-phase3-report/1"
DATE_SENTINEL = "@@DISTILL_TRACE_DATE@@"
TEACHER_MODEL = "qwen3.8-max"  # the hosted Max teacher: its rows are eval-only
#: The sources that may train (OPEN-QUESTIONS #6: the local 27B gym teacher).
DEFAULT_TRAIN_SOURCES = ("qwen38-27b-gym",)
#: Tags that never train, whatever ``--train-sources`` matches: teacher_gen.py's
#: built-in task families (hand-written task text, not real-pattern tasks). The
#: build excludes such a row: non_teacher_model, or never_train_source when the
#: --max-model wrote it (a Max-mode smoke run), whatever the Q6b allow.
NEVER_TRAIN_SOURCES = frozenset({"builtin-smoke"})
#: The pay-per-token Max judge (judge.py ``--provider max-payg``; DISTILL-OPEN-QUESTIONS,
#: the operator's PAYG ruling): its verdicts and records count. Any other judge naming
#: Max may be the Token Plan key, which never judges.
PAYG_PROVIDER = "max-payg"
PAYG_JUDGE_KEY = "payg"
#: teacher_gen.py --teacher max-payg rows: Qwen Max on the pay-per-token key, run
#: through the gym on the real-pattern taskset. Max-derived: admitted ONLY under the
#: Q6b allow (--allow-max-derived), and then only with the strict record a gym row
#: needs; refused outright without it (never eval-only, so they pin no group).
PAYG_GYM_SOURCE = "max-payg-gym"
PAYG_GYM_SOURCES = frozenset({PAYG_GYM_SOURCE, PAYG_GYM_SOURCE + "-relaxed"})
#: The models a train-source row may come from: the local 27B teacher only.
DEFAULT_TRAIN_MODELS = ("qwen3.8-27b*",)
#: The local 27B teacher's tag: a share class of its own in the report.
GYM_SOURCE = "qwen38-27b-gym"
DATASET_MANIFEST = "dataset-manifest.json"  # artifact_policy.DATASET_MANIFEST
DEFAULT_TRACES = "~/.prometheus/trajectories/golden_traces_*.jsonl"
LIMIT_TOKENS = 16384
SPLITS = ("train", "val", "eval-tools")
BAD_TURN_OUTCOMES = frozenset({"abandoned", "forced_stop", "forced stop"})
#: Phase 1's measured density, used only to estimate history left unrendered.
CHARS_PER_TOKEN = 3.93


def _load_sibling(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_render() -> Any:
    return _load_sibling("distill_render", "render.py")


R = _load_render()
#: The Max-derived artifact rule (names, manifest fields, approvals): Q6b.
AP = _load_sibling("distill_artifact_policy", "artifact_policy.py")


# ---------------------------------------------------------------------------
# Phase 1 functions (ported unchanged: classifier, normalization, hashes)
# ---------------------------------------------------------------------------

HUMAN_SURFACES = ("beacon:", "telegram:", "slack:", "discord:")
MEMORY_TOOLS = {"memory", "wiki_query", "wiki_write", "wiki_ingest", "vault_search",
                "lcm_grep", "lcm_expand_query", "lcm_describe", "lcm_expand", "note"}
CODING_TOOLS = {"edit_file", "write_file", "read_file", "grep", "glob", "lsp",
                "todo_write", "apply_patch", "notebook_edit"}
DAEMON_TOOLS_PREFIX = ("cron_", "sentinel", "symbiote_", "daemon_")
DAEMON_BASH = re.compile(
    r"\b(systemctl|journalctl|loginctl|crontab)\b|\.prometheus/|\bprometheus\.ya?ml\b"
    r"|\b(oara|prometheus)\s+(daemon|status|doctor|setup|pair|logs?)\b|\bsentinel\b"
    r"|\bdaemon\.log\b|\bnvidia-smi\b|\bllama-server\b|\bollama\b|\bdocker\b", re.I)
MEMORY_BASH = re.compile(r"\bwiki/|MEMORY\.md|USER\.md|\bmemory\.db\b|\blcm\.db\b", re.I)
CODING_BASH = re.compile(
    r"\b(git|python3?|pytest|pip|uv|ruff|mypy|npm|npx|node|pnpm|yarn|cargo|rustc|go|make|"
    r"cmake|gcc|clang|tsc|eslint|jest|vitest|swift|xcodebuild|gradle|mvn|bun|deno)\b"
    r"|\.(py|ts|tsx|js|jsx|rs|go|swift|kt|java|c|cc|cpp|h|toml|json|ya?ml|md|sh|css|html)\b"
    r"|\b(sed|awk|diff|patch|wc|head|tail|cat|ls|find|rg|grep)\b", re.I)
DAEMON_PROMPT = re.compile(
    r"^\s*(A worker you started|\[Restored after a restart|\[SENTINEL|SENTINEL\b|\[cron|"
    r"\[Scheduled|Scheduled (job|task|prompt)|Heartbeat|\[heartbeat|\[Task result|"
    r"Task: .*\(t[a-z0-9]+\))", re.I)
COMPACTION = re.compile(r"\[End of compressed context\]|\[Compacted summary of turns|"
                        r"\[microcompacted\]|\[content pruned|tool result trimmed by the compactor")

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


def classify(trace: dict[str, Any], prompt: str, args: dict[str, Any], *,
             human_surface: bool | None = None) -> tuple[str, str]:
    """(category, reason), first matching rule wins (Phase 1 ``classify``).

    *human_surface* overrides the session-prefix test, for gym traces whose
    synthetic session ids are not a human surface but whose prompts are.
    """
    meta = trace.get("_meta") or {}
    tool = meta.get("tool_name") or ""
    sid = meta.get("session_id") or ""
    if not trace.get("tools") or not trace["messages"][-1].get("tool_calls"):
        return "chat-mode", "no tools / no call"
    human = sid.startswith(HUMAN_SURFACES) if human_surface is None else human_surface
    if not human:
        return "sentinel/daemon", "non-human session"
    if DAEMON_PROMPT.search(prompt or ""):
        return "sentinel/daemon", "injected prompt"
    if tool.startswith(DAEMON_TOOLS_PREFIX):
        return "sentinel/daemon", "daemon tool"
    cmd = json.dumps(args, ensure_ascii=False) if tool == "bash" else ""
    if tool == "bash" and DAEMON_BASH.search(cmd):
        return "sentinel/daemon", "bash on daemon/services"
    if tool in MEMORY_TOOLS or tool.startswith(("wiki_", "lcm_", "memory")):
        return "memory/wiki", "memory tool"
    if tool in ("bash", "read_file", "grep", "edit_file", "write_file") and MEMORY_BASH.search(
            json.dumps(args, ensure_ascii=False)):
        return "memory/wiki", "file op on memory/wiki"
    if tool in CODING_TOOLS or tool.startswith(("coding_", "git_")):
        return "coding", "file/code tool"
    if sid.startswith("coding:"):
        return "coding", "coding session"
    if tool == "bash":
        if CODING_BASH.search(cmd):
            return "coding", "bash dev command"
        return "other", "bash other"
    if tool.startswith("task_") or tool.startswith("sessions_"):
        return "other", "worker/task orchestration"
    if tool in ("web_search", "web_fetch", "github_search"):
        return "other", "web"
    return "other", "misc tool"


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


@dataclass
class TraceKeys:
    prompt: str
    call: dict[str, Any]
    args: dict[str, Any]
    h_exact: str
    h_prompt: str
    h_prompt_call: str
    h_call: str


def trace_keys(trace: dict[str, Any]) -> TraceKeys:
    """The Phase 1 hashes of one stored trace (``audit_traces.py`` ``main``)."""
    meta = trace.get("_meta") or {}
    msgs = trace["messages"]
    target = msgs[-1]
    call = (target.get("tool_calls") or [{}])[0].get("function", {})
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    ctx = msgs[:-1]
    users = [content_text(m.get("content")) for m in ctx if m.get("role") == "user"]
    prompt = users[-1] if users else ""
    norm_call = (meta.get("tool_name") or "") + "|" + norm(json.dumps(args, sort_keys=True))
    exact = h(json.dumps([[m["role"], m.get("content")] for m in msgs[:-1]])
              + json.dumps(call, sort_keys=True) + (target.get("content") or ""))
    return TraceKeys(prompt=prompt, call=call, args=args, h_exact=exact, h_prompt=h(norm(prompt)),
                     h_prompt_call=h(norm(prompt) + "||" + norm_call), h_call=h(norm_call))


def prefix_hash(messages: list[dict[str, Any]], n: int) -> str:
    """Hash of the first *n* non-system messages of a request (role, normalized
    text, normalized tool calls). Two examples that share it share context.
    Pass the request only: the turn being learned is not context."""
    parts: list[str] = []
    for m in [m for m in messages if m.get("role") != "system"][:n]:
        calls = [
            (tc.get("function") or {}).get("name", "") + "|"
            + norm(json.dumps((tc.get("function") or {}).get("arguments"), sort_keys=True))
            for tc in m.get("tool_calls") or []
        ]
        parts.append(m.get("role", "") + "\x1f" + norm(content_text(m.get("content")))
                     + "\x1f" + "\x1e".join(calls))
    return h("\x1d".join(parts))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    trace_id: str
    kind: str  # golden | extra | fail
    file: str
    line: int
    raw: dict[str, Any]

    @property
    def meta(self) -> dict[str, Any]:
        return self.raw.get("_meta") or {}


def expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        out += [Path(p) for p in hits if Path(p).is_file()]
    return out


def duplicate_names(paths: list[Path]) -> list[str]:
    """File names given more than once. A trace id is ``<file name>:<line>``,
    so two inputs with one name would share ids (and verdicts)."""
    seen = collections.Counter(p.name for p in paths)
    return sorted(name for name, n in seen.items() if n > 1)


def iter_traces(paths: list[Path], kind: str, stats: collections.Counter) -> Iterator[Trace]:
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or not raw.get("messages"):
                        raise ValueError("no messages")
                except ValueError:
                    stats[f"unreadable_lines_{kind}"] += 1
                    continue
                yield Trace(f"{path.name}:{lineno}", kind, path.name, lineno, raw)


# ---------------------------------------------------------------------------
# Data policy: which rows may train, and what verifies a row
# ---------------------------------------------------------------------------

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
PASS_WORDS = frozenset({"pass", "passed", "verified", "ok"})
FAIL_WORDS = frozenset({"fail", "failed", "unverified", "rejected", "error"})
POLICY_RULE = (
    "train/val hold only rows whose _meta.source matches --train-sources, whose model matches "
    "--train-models, whose model and source do not name Max, and whose own verification record "
    "shows tool success, a passing judge matching --judge-models, sandboxed tools and no side "
    "effects; Max-derived rows (the --max-model's, a source naming Max, or no source tag, "
    "failed calls included) are eval-tools only, and a group holding one goes to eval-tools "
    "whole")
MAX_ALLOW_RULE = (
    "Max-derived rows may also train under the operator's recorded approval (--allow-max-derived "
    "with --max-derived-approval; DISTILL-OPEN-QUESTIONS Q6b): the --max-model's rows, with their "
    "tool-success label (golden) or the strict record a gym row needs (extra: tool success, a "
    "local or pay-per-token Max judge, sandboxed, no side effects), never against a failing "
    "verdict or record, relaxed tags and failed calls never, listed sessions (--exclude-sessions: "
    "the Token-Plan incident session) never; every Max row in train/val must be covered by the "
    "approval, and the Max-only eval-tools hold-out and --eval-holdout tasks stay eval-tools")
SHARE_CLASSES = ("max", GYM_SOURCE, "other")


def share_class(max_derived: bool, source: str) -> str:
    """The report's share class of a row: Max-derived, the local 27B teacher's, or other."""
    return "max" if max_derived else GYM_SOURCE if source == GYM_SOURCE else "other"


def names_max(name: Any) -> bool:
    """True when *name* holds the token ``max``: ``qwen3.8-max``, ``max-golden``,
    ``max-gym-2026-11-01`` do, ``maximal`` does not. The rule the trainer's and
    the flywheel's guards use; over-matching only keeps a row out of training."""
    return isinstance(name, str) and "max" in _TOKEN_SPLIT.split(name.lower())


def source_tag(meta: dict[str, Any]) -> str | None:
    """The row's own ``_meta.source`` tag, or None (the builder's fallback
    labels ``max-golden`` / ``extra`` / ``fail`` are not tags)."""
    tag = meta.get("source")
    return tag.strip() if isinstance(tag, str) and tag.strip() else None


def source_label(trace: Trace) -> str:
    """The source a row is counted and written under."""
    return source_tag(trace.meta) or ("max-golden" if trace.kind == "golden" else trace.kind)


def matches_source(tag: str | None, patterns: tuple[str, ...] | None) -> bool:
    return tag is not None and any(fnmatch.fnmatchcase(tag, p) for p in patterns or ())


def matches_model(model: Any, patterns: tuple[str, ...] | None) -> bool:
    """*model* (a non-empty string) matches one of *patterns*, ignoring case
    (served model ids differ in case only: ``Qwen3.8-27B`` / ``qwen3.8-27b``)."""
    return (isinstance(model, str) and bool(model.strip())
            and any(fnmatch.fnmatchcase(model.strip().lower(), p.lower()) for p in patterns or ()))


def says_max_derived(holder: Any) -> bool:
    """*holder* carries a ``max_derived`` other than an explicit ``false``
    (absent is silent; null, true, "false" or 0 all count: fail closed)."""
    return isinstance(holder, dict) and "max_derived" in holder \
        and holder["max_derived"] is not False


def no_effects(value: Any) -> bool:
    """A side-effect field that says "none": exactly ``false`` or empty."""
    return value is False or (isinstance(value, (str, list, tuple, dict)) and not value)


def chat_mode_of(raw: dict[str, Any]) -> bool:
    """Whether the trace's request offered no tools (module docstring)."""
    found = raw.get("_meta")
    meta: dict[str, Any] = found if isinstance(found, dict) else {}
    if meta.get("suppress_tools") or meta.get("chat_mode") \
            or str(meta.get("tool_choice") or "").strip().lower() == "none" \
            or str(meta.get("mode") or "").strip().lower() == "chat":
        return True
    target = (raw.get("messages") or [{}])[-1]
    return not target.get("tool_calls") and not raw.get("tools")


@dataclass
class Verification:
    decision: bool | None  # None: the row carries no record
    basis: str  # on a pass, the label it earns; otherwise why it failed
    payg_judge: bool = False  # the judge was the pay-per-token Max judge


def is_relaxed_source(tag: Any) -> bool:
    """A ``relaxed`` tag (teacher_gen's ``qwen38-27b-gym-relaxed``: verified only
    under a relaxation of tool success): never a train row."""
    return isinstance(tag, str) and "relaxed" in _TOKEN_SPLIT.split(tag.lower())


def is_payg_judge(judge: Any, model: Any) -> bool:
    """*judge* (a record's ``judge`` object) is the pay-per-token Max judge:
    ``provider`` is ``max-payg``, a ``judge_key`` (when given) is ``payg``, and
    its model names Max. Anything else naming Max is not."""
    return (isinstance(judge, dict) and judge.get("provider") == PAYG_PROVIDER
            and judge.get("judge_key", PAYG_JUDGE_KEY) == PAYG_JUDGE_KEY and names_max(model))


EFFECT_KEYS = ("side_effects", "real_side_effects")


def verification_of(meta: dict[str, Any], *, strict: bool = False,
                    judge_models: tuple[str, ...] | None = None,
                    payg_judge: bool = False) -> Verification:
    """The row's ``_meta.verification`` record, read fail-closed (module docstring).

    Each part is either absent or exactly its safe value; any other value
    fails. *strict* (a train-source row) also fails a record that leaves a
    part out: tool success, a passing judge whose model matches
    *judge_models*, sandboxed, no side effects. A judge naming Max fails
    (``judge_is_max``) unless *payg_judge* allows the pay-per-token Max judge
    (:func:`is_payg_judge`; a Max row's record, never a train-source row's),
    which then also stands in for a *judge_models* match."""
    rec = meta.get("verification")
    if rec is None:
        return Verification(None, "absent")
    if not isinstance(rec, dict):
        return Verification(False, "unreadable")
    decision = rec.get("verified")
    if not isinstance(decision, bool):
        word = str(rec.get("status") or rec.get("verdict") or "").strip().lower()
        decision = True if word in PASS_WORDS else False if word in FAIL_WORDS else None
    if decision is None:
        return Verification(False, "no_decision")
    if not decision:
        return Verification(False, "not_verified")
    if "tool_success" in rec and rec["tool_success"] is not True:
        return Verification(False, "tool_failed")
    judge = rec.get("judge")
    j_verdict: Any = None
    j_model: Any = None
    if isinstance(judge, dict):
        j_verdict, j_model = judge.get("verdict"), judge.get("model")
    elif isinstance(judge, str):
        j_verdict = judge
    j_verdict = j_verdict if j_verdict is not None else rec.get("judge_verdict")
    j_model = j_model if j_model is not None else rec.get("judge_model")
    judged = judge is not None or j_verdict is not None or j_model is not None
    payg = payg_judge and is_payg_judge(judge, j_model)
    if not payg and (names_max(j_model)
                     or (isinstance(judge, dict) and names_max(judge.get("served_model")))):
        return Verification(False, "judge_is_max")
    if judged and (j_verdict is None or (isinstance(judge, dict) and judge.get("error"))):
        return Verification(False, "judge_no_verdict")  # a judge that never answered
    if judged and (str(j_verdict).strip().lower() not in PASS_WORDS
                   or (isinstance(judge, dict) and "pass" in judge and judge["pass"] is not True)):
        return Verification(False, "judge_not_pass")
    if "sandboxed" in rec and rec["sandboxed"] is not True:
        return Verification(False, "not_sandboxed")
    effects = [k for k in EFFECT_KEYS if k in rec]
    if any(not no_effects(rec[k]) for k in effects):
        return Verification(False, "side_effects")
    if strict:
        if "tool_success" not in rec:
            return Verification(False, "no_tool_success")
        if not judged:
            return Verification(False, "no_judge")
        if not isinstance(j_model, str) or not j_model.strip():
            return Verification(False, "judge_no_model")
        if judge_models is not None and not matches_model(j_model, judge_models) and not payg:
            return Verification(False, "judge_not_allowed")
        if "sandboxed" not in rec:
            return Verification(False, "no_sandbox_record")
        if not effects:
            return Verification(False, "no_side_effects_record")
        return Verification(True, "judge-pass", payg)
    if judged:
        return Verification(True, "judge-pass", payg)
    return Verification(True, "tool-success" if "tool_success" in rec else "verified")


def load_index(path: str | None) -> dict[tuple[str, int], dict[str, Any]]:
    """Phase 1 index rows keyed by (file, ordinal within the file, 1-based).
    The index has one row per non-empty trace line, in file order."""
    if not path:
        return {}
    out: dict[tuple[str, int], dict[str, Any]] = {}
    seen: collections.Counter = collections.Counter()
    with Path(path).expanduser().open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                seen[row["file"]] += 1
                out[(row["file"], seen[row["file"]])] = row
    return out


def load_verdicts(paths: list[str], stats: collections.Counter,
                   checks: dict[str, Any] | None = None) -> dict[str, str]:
    """trace key -> "pass" | "fail". Later files (and later rows) win.

    A ``judge.py`` row (it has a ``source`` object) counts only with
    ``status: ok``, and its verdict is ``verdict``: an ``error`` or
    ``unparseable`` row holds no verdict, and its ``label`` is the label the
    item was sampled with, not a judgement. Its trace is ``source.file`` +
    ``source.line``, judge.py's 0-based line, so the trace id is line + 1; its
    ``source.telemetry_rowid`` goes to *checks* (key -> rowid) for the build
    to compare with the Phase 1 index. A fail-side row (no file: a failed call
    judged from telemetry) is skipped, since fail traces are never trained.

    A verdict whose judge model names Max (judge.py's ``model``; on other rows
    ``judge_model``, else ``model``) is counted in ``verdict_rows_max_judge``,
    which fails the verdicts gate: the Token Plan key never judges. Its pass is
    not loaded; its fail is (dropping a row is always safe). The one exception
    is a judge.py row from the pay-per-token Max judge (``provider: max-payg``,
    ``judge_key: payg``; the operator's PAYG ruling for Phase 2 judging): it is
    applied like a local judge's and counted in ``verdict_rows_payg_max_judge``,
    which does not fail the gate.
    """
    out: dict[str, str] = {}

    def max_judged(model: Any, v: str) -> bool:
        if not names_max(model):
            return False
        stats["verdict_rows_max_judge"] += 1
        return v == "pass"
    for path in paths:
        p = Path(path).expanduser()
        if not p.is_file():
            stats["verdict_files_missing"] += 1
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("source")
            if isinstance(src, dict):
                if row.get("status") != "ok":
                    stats["verdict_rows_not_ok"] += 1
                    continue
                file, lineno = src.get("file"), src.get("line")
                if not file and str(row.get("label")) == "fail":
                    stats["verdict_rows_fail_side"] += 1
                    continue
                v = str(row.get("verdict") or "").strip().lower()
                if not file or type(lineno) is not int or lineno < 0 or v not in ("pass", "fail"):
                    stats["verdict_rows_unusable"] += 1
                    continue
                key = f"{file}:{lineno + 1}"
                if row.get("provider") == PAYG_PROVIDER and row.get("judge_key") == PAYG_JUDGE_KEY \
                        and names_max(row.get("model")):
                    stats["verdict_rows_payg_max_judge"] += 1
                elif max_judged(row.get("model"), v):
                    continue
                out[key] = v
                if checks is not None:
                    checks[key] = src.get("telemetry_rowid")
                continue
            key = row.get("trace_id") or row.get("h_exact")
            if not key and row.get("file") and row.get("line") is not None:
                key = f"{row['file']}:{row['line']}"
            v = row.get("verdict", row.get("label"))
            if isinstance(row.get("pass"), bool):
                v = "pass" if row["pass"] else "fail"
            v = str(v).strip().lower() if v is not None else ""
            if not key or v not in ("pass", "fail"):
                stats["verdict_rows_unusable"] += 1
                continue
            judge = row.get("judge_model") if row.get("judge_model") is not None \
                else row.get("model")
            if max_judged(judge, v):
                continue
            out[str(key)] = v
            if checks is not None:
                checks.pop(str(key), None)
    return out


def load_pack(path: str) -> dict[str, Any]:
    pack = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if pack.get("schema") != PACK_SCHEMA:
        raise SystemExit(f"{path}: not a request pack ({pack.get('schema')!r})")
    return pack


# ---------------------------------------------------------------------------
# The request pack (system prompt + tools), from Prometheus's own builders
# ---------------------------------------------------------------------------

PACK_APPROX = ("bootstrap_now", "no_memory", "no_saved_files", "env_now", "no_per_run",
               "tools_now", "teacher_route")


def runtime_catalog(config: dict[str, Any]) -> str:
    """The tool catalog a LOCAL run sends under this config: ``stubs`` when
    deferred loading is on or auto (auto resolves to on for local backends)
    with the stubs catalog, else ``full``. The 0.9.12 ``names`` catalog offers
    only a few tools and is not a training target; it maps to ``full``."""
    from prometheus.context.tool_stubs import catalog_mode

    cfg = ((config.get("tools") or {}).get("deferred_loading")) or {}
    enabled = str(cfg.get("enabled", "auto")).strip().lower() if isinstance(cfg, dict) else "auto"
    if enabled in ("auto", "true") and catalog_mode(cfg) == "stubs":
        return "stubs"
    return "full"


def build_pack(config_path: str, cwd: str) -> dict[str, Any]:
    """System prompt (with a date sentinel) and both tool catalogs."""
    import yaml

    config = yaml.safe_load(Path(config_path).expanduser().read_text(encoding="utf-8")) or {}
    from prometheus.context import prompt_assembler as pa
    from prometheus.skills.loader import skills_for_prompt

    real_env = pa.get_environment_info

    def env_with_sentinel(cwd: str | None = None) -> Any:
        env = real_env(cwd=cwd)
        env.date = DATE_SENTINEL
        return env

    saved_files = None
    try:
        import prometheus.utils.user_files as user_files

        saved_files = user_files.files_context_block
        user_files.files_context_block = lambda: ""  # type: ignore[assignment]
    except ImportError:
        user_files = None  # type: ignore[assignment]
    pa.get_environment_info = env_with_sentinel  # type: ignore[assignment]
    try:
        system = pa.build_runtime_system_prompt(
            cwd=str(Path(cwd).expanduser()), config=config, project_files=False,
            skills=skills_for_prompt(), memory_content="",
        )
    finally:
        pa.get_environment_info = real_env  # type: ignore[assignment]
        if user_files is not None and saved_files is not None:
            user_files.files_context_block = saved_files  # type: ignore[assignment]
    if DATE_SENTINEL not in system:
        raise SystemExit("the system prompt has no date line to set per trace")

    from prometheus.__main__ import create_tool_registry
    from prometheus.context.tool_stubs import catalog

    registry = create_tool_registry(config.get("security", {}) or {})
    full = registry.to_api_schema()
    stubs, undescribed = catalog(full)
    try:
        commit = subprocess.run(["git", "-C", str(Path(cwd).expanduser()), "rev-parse", "--short",
                                 "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {
        "schema": PACK_SCHEMA,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prometheus_commit": commit,
        "date_sentinel": DATE_SENTINEL,
        "system_prompt": system,
        "tools": {"full": full, "stubs": stubs},
        "stub_names": sorted(undescribed),
        "runtime_catalog": runtime_catalog(config),
        "approx": list(PACK_APPROX),
    }


# ---------------------------------------------------------------------------
# Rebuilding one request
# ---------------------------------------------------------------------------


def openai_to_conversation(messages: list[dict[str, Any]]) -> list[Any]:
    """OpenAI-shape messages -> ConversationMessages, so the provider's own
    builder produces the wire messages. Consecutive tool results become one
    user message of ToolResultBlocks, as the agent loop stores them."""
    from prometheus.engine.messages import (
        ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock)

    out: list[Any] = []
    results: list[Any] = []

    def flush() -> None:
        if results:
            out.append(ConversationMessage(role="user", content=list(results), is_trusted=True))
            results.clear()

    for m in messages:
        role = m.get("role")
        if role == "tool":
            results.append(ToolResultBlock(tool_use_id=str(m.get("tool_call_id") or "call"),
                                           content=content_text(m.get("content")),
                                           is_error=bool(m.get("is_error", False))))
            continue
        flush()
        text = content_text(m.get("content"))
        if role == "user":
            out.append(ConversationMessage.from_user_text(text))
        elif role == "assistant":
            blocks: list[Any] = [TextBlock(text=text)] if text else []
            for i, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                blocks.append(ToolUseBlock(id=str(tc.get("id") or f"call_{i}"),
                                           name=fn.get("name", ""),
                                           input=args if isinstance(args, dict) else {}))
            out.append(ConversationMessage(role="assistant", content=blocks, is_trusted=True))
        # a stored "system" message is dropped: the rebuilt system prompt replaces it
    flush()
    return out


@dataclass
class LcmSession:
    rows: list[sqlite3.Row]
    calls: dict[tuple[str, str], list[int]]  # (tool name, canonical input) -> row indexes
    errors: frozenset[str] = frozenset()  # tool_use ids whose result is an error


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class LcmSnapshot:
    """Read-only access to a conversation-store snapshot, one session at a time."""

    def __init__(self, path: str, cache_sessions: int = 32) -> None:
        p = Path(path).expanduser().resolve()
        self.con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        self.con.row_factory = sqlite3.Row
        self._cache: dict[str, LcmSession] = {}
        self.cache_sessions = cache_sessions

    def session(self, session_id: str) -> LcmSession:
        if session_id not in self._cache:
            if len(self._cache) >= self.cache_sessions:
                self._cache.pop(next(iter(self._cache)))
            rows = self.con.execute(
                "SELECT rowid AS row_id, * FROM lcm_messages WHERE session_id = ?"
                " ORDER BY turn_index, rowid", (session_id,)).fetchall()
            calls: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
            errors: set[str] = set()
            for i, row in enumerate(rows):
                if not row["content_json"]:
                    continue
                try:
                    blocks = json.loads(row["content_json"])
                except ValueError:
                    continue
                for b in blocks if isinstance(blocks, list) else []:
                    if not isinstance(b, dict):
                        continue
                    if row["role"] == "assistant" and b.get("type") == "tool_use":
                        calls[(str(b.get("name")), canonical(b.get("input")))].append(i)
                    elif row["role"] != "assistant" and b.get("type") == "tool_result" \
                            and b.get("is_error"):
                        errors.add(str(b.get("tool_use_id")))
            self._cache[session_id] = LcmSession(rows, dict(calls), frozenset(errors))
        return self._cache[session_id]

    def rows(self, session_id: str) -> list[sqlite3.Row]:
        return self.session(session_id).rows


def row_message(row: sqlite3.Row) -> Any:
    """An LCM row as a ConversationMessage (``from_stored`` without reading
    the media cache: a picture becomes its placeholder text)."""
    from prometheus.engine.messages import ConversationMessage, ImageBlock, TextBlock

    blocks: list[Any] = []
    if row["content_json"]:
        try:
            msg = ConversationMessage(role="assistant" if row["role"] == "assistant" else "user",
                                      content=json.loads(row["content_json"]))
            blocks = list(msg.content)
        except Exception:
            blocks = []
    if not blocks:
        blocks = [TextBlock(text=row["content"] or "")]
    blocks = [TextBlock(text=b.placeholder_text()) if isinstance(b, ImageBlock) else b
              for b in blocks]
    keys = row.keys()
    provenance = row["provenance"] if "provenance" in keys else "user"
    trusted = bool(row["is_trusted"]) if "is_trusted" in keys else True
    try:
        return ConversationMessage(role="assistant" if row["role"] == "assistant" else "user",
                                   content=blocks, provenance=provenance, is_trusted=trusted)
    except Exception:
        return ConversationMessage(role="assistant" if row["role"] == "assistant" else "user",
                                   content=blocks, provenance="user", is_trusted=False)


def locate_target(sess: LcmSession, ts: float, name: str, args: dict[str, Any],
                  slack: float = 300.0) -> tuple[list[int], str]:
    """Assistant rows that may hold the teacher's call, best first, and how
    they were found.

    A turn's rows are persisted when the turn ends, so the call's own row is
    at or after the call's timestamp: rows in ``[ts - slack, ...)`` with the
    same tool and input, nearest first, then in prompt order. The same call
    made twice in one turn gives two rows with one timestamp; the caller takes
    the first one no earlier trace has claimed.
    """
    hits = [i for i in sess.calls.get((name, canonical(args)), [])
            if sess.rows[i]["timestamp"] >= ts - slack]
    if not hits:
        return [], "unmatched"
    hits.sort(key=lambda i: (abs(sess.rows[i]["timestamp"] - ts), i))
    return hits, "ambiguous" if len(hits) > 1 else "exact"


def row_is_query(row: sqlite3.Row) -> bool:
    """A user row a person (or the runtime) asked something in: text, and no
    tool results."""
    if row["role"] != "user" or not (row["content"] or "").strip():
        return False
    try:
        blocks = json.loads(row["content_json"]) if row["content_json"] else []
    except ValueError:
        blocks = []
    return not any(isinstance(b, dict) and b.get("type") == "tool_result"
                   for b in (blocks if isinstance(blocks, list) else []))


def row_chars(row: sqlite3.Row) -> int:
    return len(row["content_json"] or row["content"] or "")


def row_tool_use_ids(row: sqlite3.Row) -> list[str]:
    try:
        blocks = json.loads(row["content_json"]) if row["content_json"] else []
    except ValueError:
        return []
    return [str(b.get("id")) for b in (blocks if isinstance(blocks, list) else [])
            if isinstance(b, dict) and b.get("type") == "tool_use"]


def request_shape_problem(messages: list[dict[str, Any]]) -> str | None:
    """Why ``messages[:-1]`` is not a request the runtime would send, or None.

    The loop never sends a request that ends on an assistant turn (llama-server
    would take it as a prefill and render something else), nor a tool call
    without its results in the tool messages right after it.
    """
    req = messages[:-1]
    if req and req[-1].get("role") == "assistant":
        return "target_after_assistant"
    for i, m in enumerate(req):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        answered: set[str] = set()
        k = i + 1
        while k < len(req) and req[k].get("role") == "tool":
            answered.add(str(req[k].get("tool_call_id")))
            k += 1
        if any(str(tc.get("id")) not in answered for tc in m["tool_calls"]):
            return "unanswered_tool_call"
    return None


def error_after_last_query(messages: list[dict[str, Any]], error_ids: frozenset[str]) -> bool:
    """A tool result after the last user message of the request (``messages[:-1]``)
    whose call failed: the example as trained shows an error to recover from."""
    req = messages[:-1]
    last_user = max((i for i, m in enumerate(req) if m.get("role") == "user"), default=-1)
    return any(m.get("role") == "tool" and str(m.get("tool_call_id")) in error_ids
               for m in req[last_user + 1:])


def pick_call(calls: list[dict[str, Any]], name: str, args: dict[str, Any]) -> dict[str, Any]:
    """The call among *calls* with this name and these arguments, else the first."""
    for tc in calls:
        fn = tc.get("function") or {}
        if fn.get("name") != name:
            continue
        try:
            got = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            continue
        if got == args:
            return tc
    return calls[0]


@dataclass
class Request:
    messages: list[dict[str, Any]]  # system + context + target, wire format
    tools: list[dict[str, Any]]
    tool_choice: Any
    template_kwargs: dict[str, Any]
    context_source: str
    chat_mode: bool
    error_in_turn: bool
    lcm_target: str | None = None
    lcm_match: str | None = None
    omitted_messages: int = 0  # history left out before rendering (too long to matter)
    omitted_chars: int = 0
    error_ids: frozenset[str] = frozenset()  # tool_call ids whose result is an error
    query_text: str = ""  # the user query the rebuilt context ends on


class Rebuilder:
    """The request LlamaCppProvider would send, rebuilt for one trace."""

    def __init__(self, pack: dict[str, Any], catalog: str, lcm: LcmSnapshot | None = None,
                 lcm_max_messages: int = 400, prewindow_chars: int | None = None) -> None:
        from prometheus.providers.llama_cpp import LlamaCppProvider

        self.pack = pack
        self.catalog = catalog
        self.tools = pack["tools"][catalog]
        self.full_schemas = {t["name"]: t for t in pack["tools"]["full"]}
        self.provider = LlamaCppProvider(suppress_thinking=True)
        self.lcm = lcm
        self.lcm_max_messages = lcm_max_messages
        # History further back than this many characters cannot survive the
        # left truncation (no text runs above ~6 characters per token), so it
        # is counted, not rendered. None renders everything.
        self.prewindow_chars = prewindow_chars

    def system_prompt(self, ts: float | None) -> str:
        date = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).strftime("%Y-%m-%d")
        return self.pack["system_prompt"].replace(self.pack["date_sentinel"], date)

    def payload(self, conv: list[Any], system: str, chat_mode: bool) -> dict[str, Any]:
        from prometheus.engine.messages import render_messages_for_model
        from prometheus.providers.base import ApiMessageRequest
        from prometheus.security.log_redaction import redact_capture

        request = ApiMessageRequest(
            model="local", messages=render_messages_for_model(conv), system_prompt=system,
            max_tokens=4096, tools=[] if chat_mode else list(self.tools),
            suppress_tools=chat_mode, tool_choice="none" if chat_mode else "auto",
        )
        payload = self.provider._build_request_payload(request)
        payload["messages"] = redact_capture(payload["messages"])
        return payload

    def rebuild(self, trace: Trace, keys: TraceKeys, used_lcm: set[str] | frozenset[str] = frozenset()
                ) -> Request | str:
        """A Request, or the reason the trace cannot be rebuilt. *used_lcm*: snapshot
        rows earlier traces already took as their target."""
        meta = trace.meta
        msgs = trace.raw["messages"]
        target = msgs[-1]
        if target.get("role") != "assistant":
            return "no_assistant_target"
        chat_mode = chat_mode_of(trace.raw)
        if chat_mode and target.get("tool_calls"):
            return "call_in_chat_mode"  # no tool was offered to call
        conv: list[Any] | None = None
        source, lcm_target, lcm_match = "stored", None, None
        error_in_turn = False
        query_text = ""
        omitted_messages = omitted_chars = 0
        if self.lcm is not None and meta.get("session_id") and not chat_mode:
            sess = self.lcm.session(meta["session_id"])
            rows = sess.rows
            hits, lcm_match = locate_target(sess, float(meta.get("timestamp") or 0),
                                            keys.call.get("name", ""), keys.args)
            free = [i for i in hits if str(rows[i]["id"]) not in used_lcm]
            if hits and not free:
                return "duplicate_lcm_target"  # a parallel call: one turn, one example
            if free:
                j = free[0]
                # The whole row is the turn learned: every parallel call in it
                # must have succeeded (a failed teacher call is never a target).
                if trace.kind != "fail" and any(i in sess.errors for i in row_tool_use_ids(rows[j])):
                    return "failed_sibling_call"
                start = max(0, j - self.lcm_max_messages)
                if any((r["provenance"] if "provenance" in r.keys() else "user") == "member"
                       for r in rows[start:j + 1]):
                    return "member_conversation"
                q = next((i for i in range(j - 1, start - 1, -1) if row_is_query(rows[i])), None)
                if q is None:
                    return "no_user_query_in_lcm"
                cut = start
                if self.prewindow_chars is not None:
                    total, cut = 0, j
                    while cut > start and total + row_chars(rows[cut - 1]) <= self.prewindow_chars:
                        cut -= 1
                        total += row_chars(rows[cut])
                if q >= cut:
                    # start on a user query, never an orphaned tool result
                    first = next(i for i in range(cut, q + 1) if row_is_query(rows[i]))
                    keep = list(range(first, j + 1))
                else:
                    # the turn alone is longer than the window: its query and
                    # its latest steps, each step an assistant call + results
                    step = next((i for i in range(cut, j + 1) if rows[i]["role"] == "assistant"), j)
                    keep = [q, *range(step, j + 1)]
                conv = [row_message(rows[i]) for i in keep]
                kept = set(keep)
                omitted_messages = sum(1 for i in range(start, j) if i not in kept)
                omitted_chars = sum(row_chars(rows[i]) for i in range(start, j) if i not in kept)
                source, lcm_target = "lcm", str(rows[j]["id"])
                query_text = rows[q]["content"] or ""
                last_user = max((i for i, m in enumerate(conv[:-1])
                                 if m.role == "user" and m.text), default=0)
                error_in_turn = any(getattr(b, "is_error", False)
                                    for m in conv[last_user:-1] for b in m.content)
        if conv is None:
            conv = openai_to_conversation(msgs)
            query_text = keys.prompt
        error_ids = {str(b.tool_use_id) for m in conv for b in m.content
                     if getattr(b, "is_error", False) and hasattr(b, "tool_use_id")}
        payload = self.payload(conv, self.system_prompt(meta.get("timestamp")), chat_mode)
        wire = payload["messages"]
        if not wire or wire[-1].get("role") != "assistant":
            return "target_lost_in_build"
        if source == "stored" and any(m.get("role") == "tool" for m in wire[:-1]):
            last_user = max((i for i, m in enumerate(wire[:-1]) if m.get("role") == "user"),
                            default=0)
            failed = [m for m in wire[last_user:-1] if m.get("role") == "tool"
                      and str(m.get("content", "")).lstrip().lower().startswith("error")]
            error_in_turn = bool(failed)
            error_ids |= {str(m.get("tool_call_id")) for m in failed}
        # tool_choice as sent: a chat-mode payload (no tools) carries none at all
        return Request(messages=wire, tools=payload.get("tools") or [],
                       tool_choice=payload.get("tool_choice"),
                       template_kwargs=dict(payload.get("chat_template_kwargs") or {}),
                       context_source=source, chat_mode=chat_mode, error_in_turn=error_in_turn,
                       lcm_target=lcm_target, lcm_match=lcm_match,
                       omitted_messages=omitted_messages, omitted_chars=omitted_chars,
                       error_ids=frozenset(error_ids), query_text=query_text)

    def call_problem(self, req: Request, *, hard_case_tool: str | None = None) -> str | None:
        """Why the target's calls are not valid against today's tools, or None.

        For a hard case (a failed call, used only as an eval case that expects
        *hard_case_tool*) only that tool has to be offered: the arguments are
        often why the call failed, and the case asks for schema-valid ones."""
        if req.chat_mode:
            return None
        offered = {(t.get("function") or {}).get("name") for t in req.tools}
        if hard_case_tool is not None:
            return None if hard_case_tool in offered else "tool_not_offered_now"
        for tc in req.messages[-1].get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            if name not in offered:
                return "tool_not_offered_now"
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                return "arguments_not_json"
            if not isinstance(args, dict):
                return "arguments_not_object"
            missing = [k for k in self.required_args(name) if k not in args]
            if missing:
                return "required_argument_missing_now"
        return None

    def required_args(self, name: str) -> list[str]:
        schema = (self.full_schemas.get(name) or {}).get("input_schema") or {}
        return list(schema.get("required") or [])


class HeadCachedEncoder:
    """Tokenize the system block once per distinct head.

    Exact because the head ends with ``<|im_end|>\\n`` and the rest starts
    with ``<|im_start|>``: the tokenizer splits on special tokens before it
    pre-tokenizes, so the two sides never merge. Used only when both hold.
    """

    HEAD_END = "<|im_end|>\n"
    NEXT = "<|im_start|>"

    def __init__(self, inner: Any, max_entries: int = 256) -> None:
        self.inner = inner
        self.cache: dict[str, list[int]] = {}
        self.max_entries = max_entries

    def encode_ids(self, text: str) -> list[int]:
        if text.startswith(self.NEXT + "system"):
            cut = text.find(self.HEAD_END)
            if cut > 0:
                cut += len(self.HEAD_END)
                rest = text[cut:]
                if not rest or rest.startswith(self.NEXT):
                    head = text[:cut]
                    key = hashlib.sha1(head.encode("utf-8", "surrogatepass")).hexdigest()
                    ids = self.cache.get(key)
                    if ids is None:
                        if len(self.cache) >= self.max_entries:
                            self.cache.clear()
                        ids = self.cache[key] = self.inner.encode_ids(head)
                    return ids + (self.inner.encode_ids(rest) if rest else [])
        return self.inner.encode_ids(text)


# ---------------------------------------------------------------------------
# One example
# ---------------------------------------------------------------------------


@dataclass
class Example:
    id: str
    trace_id: str
    kind: str
    source: str
    category: str
    task_id: str
    h_exact: str
    h_prompt_call: str
    label: str
    flags: dict[str, bool]
    context_source: str
    n_tokens_original: int
    n_tokens: int
    dropped_messages: int
    fits: bool
    longctx: bool
    boundary_stable: bool
    prefix: str
    expect_tool: str | None
    turn_outcome: str | None = None
    dropped_steps: int = 0
    omitted_messages: int = 0
    lcm_target: str | None = None
    group: str = ""
    split: str = ""
    spool: int | None = None
    # session-scoped hashes of the user messages in the final request (its own
    # query included): examples that share one share context
    ctx_keys: list[str] = field(default_factory=list)
    model: str = ""
    # data policy: a Max-derived row is eval-only; train_ok = may enter train/val
    max_derived: bool = True
    train_ok: bool = True
    gen_task: str = ""  # "<source>\x1f<generator task id>", "" when none
    # the recorded approval a Max-derived row trains under (Q6b), "" when none
    approval: str = ""


def approx_codes(context_source: str) -> list[str]:
    return list(PACK_APPROX) + ["lcm_full_history" if context_source == "lcm" else "stored_context"]


def make_expect(req: Request, rebuilder: Rebuilder, success_mode: str,
                own: tuple[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """score.py's eval-tools/val ``expect`` for the teacher's turn: its first
    call, or with *own* (name, arguments) that call (a hard case expects the
    call that failed, which need not be the first of a parallel batch)."""
    target = req.messages[-1]
    calls = target.get("tool_calls") or []
    if req.chat_mode or not calls:
        return {"tool": None}
    fn = (pick_call(calls, *own) if own else calls[0]).get("function") or {}
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        args = {}
    expect: dict[str, Any] = {"tool": name, "args": args if isinstance(args, dict) else {},
                              "args_match": "exact"}
    if success_mode == "presence":
        present = [k for k in rebuilder.required_args(name)
                   if isinstance(args, dict) and args.get(k) not in (None, "")]
        succ: dict[str, Any] = {"expect_tool": name}
        if present:
            succ["expect_tool_args_present"] = present
        expect["success"] = succ
    return expect


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def seeded(seed: str, key: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{key}".encode()).hexdigest()


def assign_groups(examples: list[Example]) -> None:
    """Join tasks that share a context prefix, a session user message of their
    final requests or a generator task id; group = smallest task id in it.
    Rows of every source take part, so a train-source task sharing context with
    a Max-derived one lands in its group."""
    uf = UnionFind()
    for ex in examples:
        uf.union("t:" + ex.task_id, "p:" + ex.prefix)
        for key in ex.ctx_keys:
            uf.union("t:" + ex.task_id, "c:" + key)
        if ex.gen_task:  # every row of one generator task, whatever its prompt
            uf.union("t:" + ex.task_id, "g:" + ex.gen_task)
    members: dict[str, list[str]] = collections.defaultdict(list)
    for node in list(uf.parent):
        if node.startswith("t:"):
            members[uf.find(node)].append(node[2:])
    name = {root: min(tasks) for root, tasks in members.items()}
    for ex in examples:
        ex.group = name[uf.find("t:" + ex.task_id)]


@dataclass
class SplitPlan:
    groups: dict[str, str]  # group -> split
    ineligible_groups: int
    eval_tasks: int
    total_tasks: int
    # data policy: groups holding a row that may not train go to eval-tools whole
    pinned_groups: int = 0
    pinned_tasks: int = 0
    pinned_train_rows: int = 0  # train-source rows held out with them
    # Q6b allow: the Max-only plan's eval-tools groups, kept as eval data
    max_holdout_groups: int = 0
    max_holdout_tasks: int = 0
    # groups holding a task named by an --eval-holdout case file
    listed_holdout_groups: int = 0


def plan_splits(pool: list[Example], *, seed: str, eval_task_frac: float, val_frac: float,
                max_group_share: float, pinned: set[str] | frozenset[str] = frozenset(),
                no_eval_quota: set[str] | frozenset[str] = frozenset()) -> SplitPlan:
    """Assign groups to splits, seeded. *pinned* groups go to eval-tools whole
    and take no part in the quotas or the hold-out size limit, which are
    computed over the other groups. *no_eval_quota* groups are never picked
    for the eval-tools quota (which is then a share of the other free
    groups' tasks); they go to val or train."""
    free = [ex for ex in pool if ex.group not in pinned]
    rows = collections.Counter(ex.group for ex in free)
    tasks: dict[str, set[str]] = collections.defaultdict(set)
    for ex in free:
        tasks[ex.group].add(ex.task_id)
    total_rows = len(free)
    total_tasks = len({ex.task_id for ex in free})
    quota_tasks = len({ex.task_id for ex in free if ex.group not in no_eval_quota})
    order = sorted(rows, key=lambda g: seeded(seed, g))
    limit = max_group_share * total_rows
    eligible = {g for g in order if rows[g] <= limit}
    plan: dict[str, str] = {}
    eval_tasks = 0
    want_eval = eval_task_frac * quota_tasks
    for g in order:
        if g in eligible and g not in no_eval_quota and eval_tasks < want_eval:
            plan[g] = "eval-tools"
            eval_tasks += len(tasks[g])
    rest_rows = sum(rows[g] for g in order if g not in plan)
    want_val = val_frac * rest_rows
    val_rows = 0
    for g in order:
        if g in plan:
            continue
        if g in eligible and val_rows < want_val and val_rows + rows[g] <= 2 * want_val:
            plan[g] = "val"
            val_rows += rows[g]
    for g in order:
        plan.setdefault(g, "train")
    for g in pinned:
        plan[g] = "eval-tools"
    held = [ex for ex in pool if ex.group in pinned]
    return SplitPlan(plan, len(order) - len(eligible), eval_tasks, total_tasks,
                     pinned_groups=len({ex.group for ex in held}),
                     pinned_tasks=len({ex.task_id for ex in held}),
                     pinned_train_rows=sum(1 for ex in held
                                           if ex.train_ok and not ex.max_derived))


def round_robin(examples: list[Example], seed: str) -> list[Example]:
    """Examples ordered one task at a time (seeded), so a prefix of the list
    covers as many tasks as possible."""
    by_task: dict[str, list[Example]] = collections.defaultdict(list)
    for ex in examples:
        by_task[ex.task_id].append(ex)
    queues = [sorted(v, key=lambda e: seeded(seed, e.id))
              for _, v in sorted(by_task.items(), key=lambda kv: seeded(seed, kv[0]))]
    out: list[Example] = []
    i = 0
    while queues:
        q = queues[i % len(queues)]
        out.append(q.pop(0))
        if not q:
            queues.remove(q)
            continue
        i += 1
    return out


def cap_categories(examples: list[Example], cap: float, seed: str
                   ) -> tuple[list[Example], dict[str, dict[str, int]]]:
    """Keep the largest set in which no category exceeds *cap* of the total.

    The kept total T is the largest fixed point of T = sum_c min(n_c, floor(cap*T)),
    reached by iterating down from the full size. Within a capped category the
    rows kept are a round-robin over its tasks.
    """
    by_cat: dict[str, list[Example]] = collections.defaultdict(list)
    for ex in examples:
        by_cat[ex.category].append(ex)
    n = {c: len(v) for c, v in by_cat.items()}
    total = sum(n.values())
    while True:
        keep = {c: min(k, math.floor(cap * total)) for c, k in n.items()}
        nxt = sum(keep.values())
        if nxt == total:
            break
        total = nxt
    kept: list[Example] = []
    report: dict[str, dict[str, int]] = {}
    for c, rows in sorted(by_cat.items()):
        chosen = round_robin(rows, seed)[:keep[c]] if keep[c] < len(rows) else rows
        kept += chosen
        report[c] = {"before": len(rows), "kept": len(chosen), "dropped": len(rows) - len(chosen)}
    return kept, report


def apply_ratios(examples: list[Example], ratios: dict[str, float], seed: str
                 ) -> tuple[list[Example], dict[str, dict[str, int]]]:
    """Keep the same share of each category that the train cap kept, so val
    mirrors train's mix. (Capping a few hundred val rows on their own can
    empty a split that happens to hold one category.)"""
    by_cat: dict[str, list[Example]] = collections.defaultdict(list)
    for ex in examples:
        by_cat[ex.category].append(ex)
    kept: list[Example] = []
    report: dict[str, dict[str, int]] = {}
    for c, rows in sorted(by_cat.items()):
        ratio = ratios.get(c, 1.0)
        k = len(rows) if ratio >= 1.0 else max(1 if ratio > 0 else 0, round(len(rows) * ratio))
        chosen = round_robin(rows, seed)[:k] if k < len(rows) else rows
        kept += chosen
        report[c] = {"before": len(rows), "kept": len(chosen), "dropped": len(rows) - len(chosen)}
    return kept, report


def sample_per_task(examples: list[Example], k: int, seed: str) -> list[Example]:
    by_task: dict[str, list[Example]] = collections.defaultdict(list)
    for ex in examples:
        by_task[ex.task_id].append(ex)
    out: list[Example] = []
    for _, rows in sorted(by_task.items()):
        out += sorted(rows, key=lambda e: seeded(seed, e.id))[:k]
    return out


def leakage(splits: dict[str, list[Example]]) -> dict[str, Any]:
    """Overlaps between every two splits, over the rows of every source (a
    Max-derived eval case and a train-source train row are compared like any
    two rows). ``sources`` names what each split held when checked."""
    keys = {"task": lambda e: [e.task_id], "group": lambda e: [e.group],
            "prefix": lambda e: [e.prefix], "exact": lambda e: [e.h_exact],
            "prompt_call": lambda e: [e.h_prompt_call], "context": lambda e: e.ctx_keys,
            "source_task": lambda e: [e.gen_task] if e.gen_task else []}
    out: dict[str, Any] = {}
    names = list(splits)
    ok = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row = {}
            for kname, fn in keys.items():
                n = len({k for e in splits[a] for k in fn(e)}
                        & {k for e in splits[b] for k in fn(e)})
                row[kname] = n
                ok = ok and n == 0
            out[f"{a}|{b}"] = row
    return {"pass": ok, "overlaps": out,
            "sources": {k: sorted({e.source for e in v}) for k, v in splits.items()}}


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def pct(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def tstats(xs: list[int], limit: int) -> dict[str, int]:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "p99": pct(xs, 99),
            "max": max(xs), "over_limit": sum(1 for x in xs if x > limit)}


def split_summary(rows: list[Example], limit: int) -> dict[str, Any]:
    C = collections.Counter
    n = len(rows)
    cats = C(e.category for e in rows)
    return {
        "rows": n,
        "tasks": len({e.task_id for e in rows}),
        "groups": len({e.group for e in rows}),
        "category": dict(cats.most_common()),
        "category_share": {c: round(k / n, 4) for c, k in cats.most_common()} if n else {},
        "label": dict(C(e.label for e in rows).most_common()),
        "kind": dict(C(e.kind for e in rows).most_common()),
        "context_source": dict(C(e.context_source for e in rows).most_common()),
        "source": dict(C(e.source for e in rows).most_common()),
        "max_derived": sum(1 for e in rows if e.max_derived),
        "chat_mode": sum(e.flags["chat_mode"] for e in rows),
        "final_answer": sum(e.flags.get("final_answer", False) for e in rows),
        "error_recovery": sum(e.flags["error_recovery"] for e in rows),
        "error_signal": sum(e.flags.get("error_signal", False) for e in rows),
        "compacted": sum(e.flags["compacted"] for e in rows),
        "truncated": sum(1 for e in rows if e.dropped_messages),
        "split_in_turn": sum(1 for e in rows if e.dropped_steps),
        "turn_outcome_bad": sum(1 for e in rows if (e.turn_outcome or "") in BAD_TURN_OUTCOMES),
        "tokens": tstats([e.n_tokens for e in rows], limit),
    }


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    max_tokens: int = LIMIT_TOKENS
    category_cap: float = 0.35
    val_frac: float = 0.03
    eval_task_frac: float = 0.10
    eval_max_per_task: int = 20
    holdout_max_group_share: float = 0.02
    prefix_n: int = 4
    seed: str = "phase3-v1"
    eval_success: str = "presence"
    # The hosted Max teacher whose harvested rows are kept (as eval-only rows
    # under a train-source policy). The report's settings.teacher_model is not
    # this setting: it is derived from what train/val hold (Builder.
    # teacher_model), because the trainer's and the flywheel's Max guards read
    # it as "every train/val row is this model's". The trainer falls back to
    # the rows' own provenance without it; the flywheel counts every row as
    # Max-derived without it.
    max_model: str = TEACHER_MODEL
    # Sources that may train (fnmatch patterns on _meta.source). None = no
    # policy, every accepted row may train: library default for the split and
    # gate machinery; the CLI always sets one (default DEFAULT_TRAIN_SOURCES).
    train_sources: tuple[str, ...] | None = None
    # Under a policy, the models a train-source row may come from (the local
    # teacher; case-insensitive fnmatch) and the judges its record may name
    # (None = the train models: the teacher judges itself).
    train_models: tuple[str, ...] = DEFAULT_TRAIN_MODELS
    judge_models: tuple[str, ...] | None = None
    exclude_bad_turns: bool = False
    # Q6b: the operator's explicit, recorded allow for Max-derived rows to
    # train (both or neither; the CLI refuses one without the other). The
    # approval reference is recorded verbatim.
    allow_max_derived: bool = False
    max_derived_approval: str | None = None
    # Sessions whose rows never enter the dataset (--exclude-sessions): the
    # 2026-10-13 Token-Plan incident session (OPEN-QUESTIONS #6). The allow
    # needs a list. The report records how many, never the ids.
    exclude_sessions: tuple[str, ...] = ()
    train_range: tuple[int, int] = (5000, 10000)
    val_share_range: tuple[float, float] = (0.02, 0.05)
    eval_min: int = 200
    min_presence: int = 1
    lcm_max_messages: int = 400
    prewindow_chars_per_token: int = 6
    split_turns: bool = True
    limit: int | None = None


class Builder:
    def __init__(self, *, student: Any, pack: dict[str, Any], catalog: str, settings: Settings,
                 index: dict[tuple[str, int], dict[str, Any]], verdicts: dict[str, str],
                 lcm: LcmSnapshot | None = None, spool: Path | None = None,
                 verdict_checks: dict[str, Any] | None = None,
                 eval_holdout_tasks: set[str] | frozenset[str] = frozenset(),
                 excluded_prompts: set[str] | frozenset[str] = frozenset()) -> None:
        self.student = student
        # tasks (Phase 1 prompt hashes) an earlier eval set holds: they stay eval-tools
        self.eval_holdout_tasks = frozenset(eval_holdout_tasks)
        # task proxies of every user message a --exclude-sessions session holds
        # (excluded_session_prompts): a gym row whose task group, recombined partner
        # or compaction padding came from one of them is dropped too (never the hashes
        # in any output)
        self.excluded_prompts = frozenset(excluded_prompts)
        self.student.encoder = HeadCachedEncoder(student.encoder)
        self.s = settings
        self.rebuilder = Rebuilder(pack, catalog, lcm, settings.lcm_max_messages,
                                   prewindow_chars=settings.prewindow_chars_per_token
                                   * settings.max_tokens)
        self.index = index
        self.verdicts = verdicts
        self.verdict_checks = verdict_checks or {}
        self.verdicts_used: set[str] = set()
        self.stats: collections.Counter = collections.Counter()
        self.excluded: list[tuple[str, str]] = []  # (trace_id, reason)
        self.seen_prompt_call: set[str] = set()
        self.seen_fail: set[str] = set()
        self.seen_lcm: set[str] = set()
        self.examples: list[Example] = []
        self.fails: list[Example] = []
        self.unfittable: list[Example] = []
        self.index_disagree: collections.Counter = collections.Counter()
        self.lcm_match: collections.Counter = collections.Counter()
        self.final: dict[str, list[Example]] = {k: [] for k in SPLITS}
        self.spool_path = spool
        self._spool = spool.open("wb") if spool else None
        self._reader: Any = None
        self._file_ordinal: collections.Counter = collections.Counter()
        # per-source aggregates for the report's data_policy section
        self.by_source: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)
        self.models_by_source: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter)

    def exclude(self, trace: Trace, reason: str) -> None:
        self.stats[f"excluded:{reason}"] += 1
        self.by_source[source_label(trace)][f"excluded:{reason}"] += 1
        self.excluded.append((trace.trace_id, reason))

    @property
    def judge_models(self) -> tuple[str, ...]:
        """The judge models a train-source row's record may name."""
        return self.s.judge_models if self.s.judge_models is not None else self.s.train_models

    @property
    def max_allowed(self) -> bool:
        """Q6b: under a train-source policy, the operator's recorded allow lets
        Max-derived rows train. Both parts are needed: the flag and a
        non-empty approval reference."""
        s = self.s
        return (s.train_sources is not None and s.allow_max_derived
                and isinstance(s.max_derived_approval, str) and bool(s.max_derived_approval.strip()))

    def max_row_trains(self, trace: Trace) -> tuple[bool, str]:
        """Under the allow, whether a kept Max row may enter train/val, and why
        not: a golden trace has its tool-success label by construction; an
        extra (gym) row needs the record a train-source gym row needs (tool
        success, a passing judge of the --judge-models or the pay-per-token
        Max judge, sandboxed, no side effects: :func:`verification_of` strict),
        and a verdict alone shows none of it. A relaxed tag never trains.
        Failing verdicts and records were excluded before this is asked."""
        if is_relaxed_source(source_tag(trace.meta)):
            return False, "relaxed_source"
        if trace.kind == "golden":
            return True, ""
        if trace.kind != "extra":
            return False, "not_a_teacher_row"
        strict = verification_of(trace.meta, strict=True, judge_models=self.judge_models,
                                 payg_judge=True)
        if strict.decision is True:
            return True, ""
        return False, "no_record" if strict.decision is None else strict.basis

    def teacher_model(self) -> str:
        """``settings.teacher_model`` for the report: the one model whose rows
        train/val hold, read by the trainer's and the flywheel's Max guards as
        "every train/val row is this model's". The ``--max-model`` (every row
        Max-derived, which fails closed) unless train/val hold only
        train-source rows of a single model that does not name Max."""
        s = self.s
        if s.train_sources is None:
            return s.max_model  # no policy: only the Max model's rows are kept
        rows = self.final["train"] + self.final["val"]
        models = {e.model for e in rows}
        if len(models) != 1 or not all(e.train_ok and not e.max_derived for e in rows):
            return s.max_model
        (model,) = models
        if not model or names_max(model) or model == s.max_model \
                or not matches_model(model, s.train_models):
            return s.max_model
        return model

    def payg_gym_refusal(self, trace: Trace) -> str:
        """Why a ``max-payg-gym`` row (teacher_gen.py --teacher max-payg) may not be
        kept, or "". It is Max-derived, so only the Q6b allow admits it, and only
        as a verified gym row: an extra trace, not a relaxed tag, saying it is
        Max-derived, whose own record passes strictly (tool success, a passing
        judge that is a --judge-models judge or the pay-per-token Max judge,
        sandboxed, no side effects). Refused rows go to no split at all: an
        unverified or unallowed PAYG gym row is not eval material either."""
        meta = trace.meta
        if not self.max_allowed:
            return "max_payg_gym_needs_allow"
        if trace.kind != "extra":
            return "max_payg_gym_not_extra"
        if is_relaxed_source(source_tag(meta)):
            return "max_payg_gym_relaxed"
        if meta.get("max_derived") is not True:
            return "max_payg_gym_not_max_derived"
        strict = verification_of(meta, strict=True, judge_models=self.judge_models, payg_judge=True)
        if strict.decision is not True:
            return "max_payg_gym_unverified:" + ("no_record" if strict.decision is None else strict.basis)
        return ""

    def is_train_source(self, trace: Trace) -> bool:
        """A row the data policy lets train: a listed tag that does not name Max
        and is not a never-train tag (``NEVER_TRAIN_SOURCES``), whatever the
        patterns: ``--train-sources '*'`` still never trains on them."""
        tag = source_tag(trace.meta)
        return (self.s.train_sources is not None and trace.kind != "fail"
                and matches_source(tag, self.s.train_sources) and not names_max(tag)
                and tag not in NEVER_TRAIN_SOURCES)

    def claim(self, trace: Trace, keys: TraceKeys) -> bool:
        """Take the row's dedupe key; False (and excluded) when an earlier row
        holds it. Phase 1's rule: first row per normalized prompt + call."""
        seen = self.seen_fail if trace.kind == "fail" else self.seen_prompt_call
        key = keys.h_exact if trace.kind == "fail" else keys.h_prompt_call
        if key in seen:
            self.exclude(trace, "duplicate")
            return False
        seen.add(key)
        return True

    def add(self, trace: Trace) -> None:
        self.stats[f"input_{trace.kind}"] += 1
        if trace.kind == "golden":
            self._file_ordinal[trace.file] += 1
        if not isinstance(trace.raw.get("_meta"), dict):
            trace.raw["_meta"] = {}
        meta = trace.raw["_meta"]  # (trace.meta would hand back a copy of an empty one)
        src = source_label(trace)
        self.by_source[src]["input"] += 1
        session_id = meta.get("session_id")
        if self.s.exclude_sessions and session_id is not None \
                and str(session_id).strip() in self.s.exclude_sessions:
            # a listed session (the Token-Plan incident session): no split at all.
            # A verdict on its row is accounted for (not "unmatched": the row is
            # known, and deliberately left out).
            try:
                vkeys: tuple[str, ...] = (trace.trace_id, trace_keys(trace.raw).h_exact)
            except Exception:  # a malformed row: its id is the only key it has
                vkeys = (trace.trace_id,)
            for k in vkeys:
                if k in self.verdicts:
                    self.verdicts_used.add(k)
                    self.stats["verdicts_on_excluded_sessions"] += 1
            self.exclude(trace, "excluded_session")
            return
        if self.excluded_prompts and row_task_prompts(meta) & self.excluded_prompts:
            # a gym row built from a listed session's prompt (its task group, a
            # recombined partner, or a padding turn): it has no session id of its
            # own, so the session check above cannot see it
            self.exclude(trace, "excluded_session_task")
            return
        model = meta.get("model")
        model_s = model.strip() if isinstance(model, str) else ""
        train_source = self.is_train_source(trace)
        if train_source:
            if not model_s:
                self.exclude(trace, "train_source_no_model")
                return
            if names_max(model_s) or model_s == self.s.max_model \
                    or names_max(meta.get("served_model")):
                # a Max row under a train-source tag: its provenance is wrong
                self.exclude(trace, "train_source_names_max")
                return
            if not matches_model(model_s, self.s.train_models):
                # not the local teacher (a hosted route the same key serves, say)
                self.exclude(trace, "train_source_wrong_model")
                return
            if says_max_derived(meta) or says_max_derived(meta.get("verification")):
                # its producer says it came from Max (a task seeded from a Max chat)
                self.exclude(trace, "train_source_max_derived")
                return
        elif (trace.kind != "fail" or meta.get("model")) and meta.get("model") != self.s.max_model:
            self.exclude(trace, "non_teacher_model")
            return
        elif source_tag(meta) in NEVER_TRAIN_SOURCES:
            # a hand-written smoke task the Max model answered (teacher_gen.py
            # --builtin-smoke): not real-pattern data, so neither train nor eval
            self.exclude(trace, "never_train_source")
            return
        payg_gym = source_tag(meta) in PAYG_GYM_SOURCES and trace.kind != "fail"
        if payg_gym:
            why = self.payg_gym_refusal(trace)
            if why:
                self.exclude(trace, why)
                return
        # Data policy: under one, everything kept that is not a train source is
        # the Max teacher's; with none (library default) every row may train.
        max_derived = not train_source
        train_ok = trace.kind != "fail" and (train_source or self.s.train_sources is None)
        if trace.kind == "extra" and not meta.get("tool_name"):
            calls = trace.raw["messages"][-1].get("tool_calls") or []
            name = ((calls[0] if calls and isinstance(calls[0], dict) else {})
                    .get("function") or {}).get("name")
            if name:
                meta["tool_name"] = name
        keys = trace_keys(trace.raw)
        idx = self.index.get((trace.file, self._file_ordinal[trace.file])) \
            if trace.kind == "golden" else None
        if idx is not None and (idx.get("tool") != meta.get("tool_name")
                                or abs(float(idx.get("ts") or 0) - float(meta.get("timestamp") or 0))
                                > 1e-3):
            self.index_disagree["row_mismatch"] += 1
            idx = None
        if idx is not None:
            for k in ("h_exact", "h_prompt", "h_prompt_call"):
                if idx.get(k) != getattr(keys, k):
                    self.index_disagree[k] += 1
        # Dedupe exactly as Phase 1: first teacher row per normalized prompt +
        # call. A Max row claims its key now (a verdict on it is a verdict on
        # the call); a train-source row once verified (its record verifies the
        # whole trajectory, and a failed attempt's first call may be a retry's).
        # A PAYG gym row claims its key once verified, like a train-source gym row.
        if not train_source and not payg_gym and not self.claim(trace, keys):
            return
        # label
        vkey = next((k for k in (trace.trace_id, keys.h_exact) if k in self.verdicts), None)
        verdict = self.verdicts[vkey] if vkey is not None else None
        if vkey is not None:
            self.verdicts_used.add(vkey)
            want = self.verdict_checks.get(vkey)
            if want is not None and self.index and (idx is None or idx.get("rowid") != want):
                # the corpus moved since judging: the verdict may be another trace's
                self.stats["verdict_index_mismatch"] += 1
                verdict = None
        if verdict is None and meta.get("verdict"):
            verdict = str(meta["verdict"]).lower()
        # A Max row's record may name the pay-per-token Max judge; a train-source
        # (local 27B) row's needs a --judge-models judge (OPEN-QUESTIONS #6)
        record = (verification_of(meta, strict=train_source or payg_gym, judge_models=self.judge_models,
                                  payg_judge=not train_source)
                  if trace.kind != "fail" else Verification(None, "absent"))
        if record.decision is not None:
            outcome_key = "verification:" + ("pass" if record.decision else record.basis)
            self.stats[outcome_key] += 1
            self.by_source[src][outcome_key] += 1
            if record.payg_judge:
                self.stats["verification:payg_max_judge"] += 1
        if trace.kind == "fail":
            label = "hard-case"
        elif verdict == "fail":
            self.exclude(trace, "judge_fail")
            return
        elif record.decision is False:
            self.exclude(trace, "verification_failed")
            return
        elif train_source:
            # Only the row's own strict record verifies it: a verdict (from
            # --verdicts or _meta.verdict) shows no tool success, sandbox or
            # local judge, and may come from any judge.
            if not record.decision:
                self.exclude(trace, "unverified")
                return
            label = record.basis
        elif verdict == "pass":
            label = "judge-pass"
        elif record.decision:
            label = record.basis
        elif trace.kind == "extra":
            # the tool-success fallback is for harvested teacher rows only
            self.exclude(trace, "unverified")
            return
        else:
            label = "tool-success"
        if (train_source or payg_gym) and not self.claim(trace, keys):
            return
        outcome = (idx or {}).get("turn_outcome")
        if self.s.exclude_bad_turns and label == "tool-success" and outcome in BAD_TURN_OUTCOMES:
            self.exclude(trace, "bad_turn_outcome")
            return
        approval = ""
        if max_derived and trace.kind != "fail" and self.max_allowed:
            # Q6b: the operator's recorded allow. The row has passed every gate
            # a row passes here (dedupe, verdict, record); it trains only with
            # tool success shown, and under the approval it was admitted with.
            train_ok, why_not = self.max_row_trains(trace)
            if train_ok:
                approval = str(self.s.max_derived_approval)
            else:
                self.stats[f"max_derived_eval_only:{why_not}"] += 1
                self.by_source[src][f"max_derived_eval_only:{why_not}"] += 1
        chat_mode = chat_mode_of(trace.raw)
        if trace.kind == "extra" and meta.get("category"):
            category = str(meta["category"])
        else:
            category, _ = classify(trace.raw, keys.prompt, keys.args,
                                   human_surface=True if trace.kind == "extra" else None)
            if category == "chat-mode" and not chat_mode and trace.kind != "golden":
                category = "other"  # a prose answer to a request that offered tools
        if idx is not None and idx.get("category") != category:
            self.index_disagree["category"] += 1
        try:
            req = self.rebuilder.rebuild(trace, keys, self.seen_lcm)
        except Exception as exc:  # never print trace content
            self.exclude(trace, f"rebuild_error:{type(exc).__name__}")
            return
        if isinstance(req, str):
            self.exclude(trace, req)
            return
        if req.lcm_match:
            self.lcm_match[req.lcm_match] += 1
        if req.context_source == "lcm" and h(norm(req.query_text)) != keys.h_prompt:
            self.stats["lcm_query_differs_from_prompt"] += 1
        own_call = (str(keys.call.get("name") or ""), keys.args)
        problem = self.rebuilder.call_problem(
            req, hard_case_tool=own_call[0] if trace.kind == "fail" else None)
        if problem:
            # A turn that calls a tool the request does not offer, or leaves out
            # an argument the current schema requires, would teach a call that
            # fails today.
            self.exclude(trace, problem)
            return
        if req.lcm_target is not None:
            self.seen_lcm.add(req.lcm_target)
        try:
            first = R.render_example(self.student, req.messages, req.tools, req.template_kwargs)
            fit = R.fit_to_budget(self.student, req.messages, req.tools, self.s.max_tokens,
                                  req.template_kwargs, first=first,
                                  split_turns=self.s.split_turns) \
                if first.n_tokens > self.s.max_tokens else R.Fit(req.messages, 0, first)
        except R.RenderError as exc:
            self.exclude(trace, "render_error:" + str(exc).split(":")[0][:60])
            return
        except Exception as exc:
            self.exclude(trace, f"render_error:{type(exc).__name__}")
            return
        if trace.kind == "fail" and fit is None:
            # an eval case carries the <= --max-tokens request, never a longer one
            self.exclude(trace, "fail_unfittable")
            return
        final_msgs = fit.messages if fit else req.messages
        shape = request_shape_problem(final_msgs)
        if shape:
            self.exclude(trace, shape)
            return
        # flags describe the example as trained: the context that survived the cut
        context_text = "\n".join(content_text(m.get("content")) for m in final_msgs[1:-1])
        flags = {
            "chat_mode": req.chat_mode,
            # a prose answer while tools were offered (not chat mode: tools rendered)
            "final_answer": not req.chat_mode and not final_msgs[-1].get("tool_calls"),
            "error_recovery": error_after_last_query(final_msgs, req.error_ids),
            "compacted": bool(COMPACTION.search(context_text)) or bool(meta.get("compacted")),
            # diagnostic only: telemetry / _meta say a call failed nearby, which
            # the kept context may not show
            "error_signal": bool((idx or {}).get("prev_call_failed")) or req.error_in_turn
            or bool(meta.get("error_recovery")),
        }
        # (an example that cannot fit is in no split: its long context joins nothing)
        session = str(meta.get("session_id") or "") if fit is not None else ""
        ctx_keys = sorted({session + "\x1f" + h(norm(content_text(m.get("content"))))
                           for m in final_msgs[:-1] if m.get("role") == "user"}) if session else []
        gen_task = meta.get("task_id")
        gen_task = (src + "\x1f" + str(gen_task).strip()
                    if isinstance(gen_task, (str, int)) and str(gen_task).strip() else "")
        ex = Example(
            id=("p3f-" if trace.kind == "fail" else "p3-") + h(trace.trace_id),
            trace_id=trace.trace_id, kind=trace.kind, source=src,
            category=category, task_id=keys.h_prompt, h_exact=keys.h_exact,
            h_prompt_call=keys.h_prompt_call, label=label, flags=flags,
            context_source=req.context_source,
            n_tokens_original=first.n_tokens + round(req.omitted_chars / CHARS_PER_TOKEN),
            n_tokens=fit.rendered.n_tokens if fit else 0,
            dropped_messages=fit.dropped if fit else 0, fits=fit is not None,
            dropped_steps=fit.dropped_steps if fit else 0,
            longctx=first.n_tokens > self.s.max_tokens or req.omitted_messages > 0,
            boundary_stable=(fit.rendered.boundary_stable if fit else first.boundary_stable),
            prefix=prefix_hash(final_msgs[:-1], self.s.prefix_n),
            expect_tool=None if req.chat_mode else own_call[0] if trace.kind == "fail" else
            ((req.messages[-1].get("tool_calls") or [{}])[0].get("function") or {}).get("name"),
            turn_outcome=outcome, omitted_messages=req.omitted_messages,
            lcm_target=req.lcm_target, ctx_keys=ctx_keys, model=model_s,
            max_derived=max_derived, train_ok=train_ok, gen_task=gen_task, approval=approval,
        )
        if self._spool is not None:
            heavy = {
                "id": ex.id, "messages": final_msgs, "tools": req.tools,
                "tool_choice": req.tool_choice, "chat_template_kwargs": req.template_kwargs,
                "render": fit.rendered.as_row() if fit else None,
                "original_messages": req.messages if ex.longctx else None,
                "expect": make_expect(req, self.rebuilder, self.s.eval_success,
                                      own_call if trace.kind == "fail" else None),
                "eval_expect": trace.raw.get("expect") if trace.kind == "fail" else None,
            }
            ex.spool = self._spool.tell()
            self._spool.write((json.dumps(heavy, ensure_ascii=False) + "\n").encode("utf-8"))
        if trace.kind == "fail":
            self.fails.append(ex)
        elif not ex.fits:
            self.unfittable.append(ex)
        else:
            self.examples.append(ex)
        counts = self.by_source[src]
        counts["accepted"] += 1
        counts["train_eligible"] += ex.train_ok
        counts["max_derived"] += ex.max_derived
        counts["max_derived_allowed"] += bool(ex.approval)
        counts["chat_mode"] += flags["chat_mode"]
        self.models_by_source[src][model_s or "unknown"] += 1

    def close_spool(self) -> None:
        if self._spool is not None:
            self._spool.close()
            self._spool = None

    def read_heavy(self, ex: Example) -> dict[str, Any]:
        assert self.spool_path is not None and ex.spool is not None
        if self._spool is not None:
            self._spool.flush()
        if self._reader is None:
            self._reader = self.spool_path.open("rb")
        self._reader.seek(ex.spool)
        return json.loads(self._reader.readline().decode("utf-8"))

    def close_reader(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    # ----------------------------------------------------------------- split

    def split(self) -> dict[str, Any]:
        s = self.s
        pool = self.examples
        assign_groups(pool + self.fails + self.unfittable)
        # Tasks an earlier eval set holds (--eval-holdout) stay eval-tools, whole groups.
        listed = {e.group for e in pool if e.task_id in self.eval_holdout_tasks}
        hold: set[str] = set()
        no_quota: set[str] = set()
        if s.train_sources is None:
            pinned = set(listed)
        elif self.max_allowed:
            # Q6b allow: the eval-tools hold-out stays Max-derived eval data.
            # The Max groups a Max-only build holds out (the same plan over the
            # Max rows alone) and the groups holding a row that still may not
            # train are pinned; the eval quota for the rest comes from the
            # train-source groups. A failed call pins nothing: as in a Max-only
            # build, a hard case whose task trains is dropped.
            # PAYG gym rows (gym data on the real-pattern taskset, like the train
            # source's) are left out of that plan, so it holds out the same Max
            # groups with or without them, and their groups may give eval quota.
            max_only = plan_splits([e for e in pool if e.max_derived and e.source not in PAYG_GYM_SOURCES],
                                   seed=s.seed, eval_task_frac=s.eval_task_frac, val_frac=s.val_frac,
                                   max_group_share=s.holdout_max_group_share)
            hold = {g for g, sp in max_only.groups.items() if sp == "eval-tools"}
            pinned = hold | listed | {e.group for e in pool if not e.train_ok}
            no_quota = {e.group for e in pool if e.max_derived and e.source not in PAYG_GYM_SOURCES}
        else:
            # Under a data policy a group holding a row that may not train (a
            # Max-derived row or a failed call) is eval-tools material whole.
            pinned = listed | {e.group for e in pool + self.fails if not e.train_ok}
        plan = plan_splits(pool, seed=s.seed, eval_task_frac=s.eval_task_frac,
                           val_frac=s.val_frac, max_group_share=s.holdout_max_group_share,
                           pinned=pinned, no_eval_quota=no_quota)
        plan.max_holdout_groups = len(hold)
        plan.max_holdout_tasks = len({e.task_id for e in pool if e.group in hold})
        plan.listed_holdout_groups = len(listed)
        for ex in pool:
            ex.split = plan.groups[ex.group]
        # (train_ok is already true of every row here; kept as the last word)
        train_all = [e for e in pool if e.split == "train" and e.train_ok]
        val_all = [e for e in pool if e.split == "val" and e.train_ok]
        held = [e for e in pool if e.split == "eval-tools"]
        train, train_caps = cap_categories(train_all, s.category_cap, s.seed)
        ratios = {c: r["kept"] / r["before"] for c, r in train_caps.items() if r["before"]}
        val, val_caps = apply_ratios(val_all, ratios, s.seed)
        eval_rows = sample_per_task(held, s.eval_max_per_task, s.seed)
        fail_used, fail_dropped = [], 0
        for ex in self.fails:
            if plan.groups.get(ex.group, "eval-tools") == "eval-tools":
                ex.split = "eval-tools"
                fail_used.append(ex)
            else:
                fail_dropped += 1
        for ex in self.unfittable:
            ex.split = plan.groups.get(ex.group, "unassigned")
        self.final = {"train": train, "val": val, "eval-tools": eval_rows + fail_used}
        return {
            "plan": plan, "train_caps": train_caps, "val_caps": val_caps,
            "held_rows": len(held), "eval_sampled": len(eval_rows),
            "fail_used": len(fail_used), "fail_dropped_task_in_train_or_val": fail_dropped,
            "not_train_ok_in_train_val_groups": sum(
                1 for e in pool if e.split in ("train", "val") and not e.train_ok),
        }

    def policy_gate(self) -> dict[str, Any]:
        """Nothing but the listed train sources' rows in train/val."""
        s = self.s
        if s.train_sources is None:
            return {"status": "NO POLICY",
                    "note": "library default: every accepted row may train; the CLI always "
                            "sets --train-sources"}
        rows = self.final["train"] + self.final["val"]

        def is_max(e: Example) -> bool:
            return e.max_derived or not e.train_ok

        def outside(e: Example) -> bool:
            return (not matches_source(e.source, s.train_sources) or names_max(e.source)
                    or names_max(e.model) or e.model == s.max_model)

        def unlisted(e: Example) -> bool:
            return not matches_model(e.model, s.train_models)

        def covered(e: Example) -> bool:
            # Q6b: a Max row admitted under the recorded approval, as the Max model's
            return (self.max_allowed and e.max_derived and e.train_ok and e.kind != "fail"
                    and e.approval == s.max_derived_approval and e.model == s.max_model)

        cov = [e for e in rows if covered(e)]
        rest = [e for e in rows if not covered(e)]
        uncovered = sum(1 for e in rest if is_max(e))
        other = sum(1 for e in rest if outside(e))
        wrong_model = sum(1 for e in rest if unlisted(e))
        return {"train_sources": list(s.train_sources), "train_models": list(s.train_models),
                "max_model": s.max_model,
                # every Max-derived (or not-train-ok) row; the next two count the
                # rows the approval does not cover (all rows without the allow)
                "max_derived_in_train_val": sum(1 for e in rows if is_max(e)),
                "outside_train_sources_in_train_val": other,
                "unlisted_model_in_train_val": wrong_model,
                # Q6b: Max rows the recorded approval covers, and the rest
                "max_derived_allowed": self.max_allowed,
                "max_derived_approval": s.max_derived_approval if self.max_allowed else None,
                "max_derived_covered_in_train_val": len(cov),
                "max_derived_uncovered_in_train_val": uncovered,
                "pass": uncovered == 0 and other == 0 and wrong_model == 0}

    def gates(self) -> dict[str, Any]:
        s = self.s
        # size, share, category and presence count the rows that may train only
        tr, va = ([e for e in self.final[k] if e.train_ok] for k in ("train", "val"))
        ev = self.final["eval-tools"]
        g: dict[str, Any] = {"leakage": leakage(self.final)}
        vg = {"files_missing": self.stats["verdict_files_missing"],
              "rows_unusable": self.stats["verdict_rows_unusable"],
              "max_judge": self.stats["verdict_rows_max_judge"],
              "unmatched": len(set(self.verdicts) - self.verdicts_used),
              "index_mismatch": self.stats["verdict_index_mismatch"]}
        g["verdicts"] = {"loaded": len(self.verdicts), **vg,
                         # the pay-per-token Max judge's verdicts (applied; never a failure)
                         "payg_max_judge": self.stats["verdict_rows_payg_max_judge"],
                         "pass": not any(vg.values())}
        g["data_policy"] = self.policy_gate()
        g["train_size"] = {"value": len(tr), "range": list(s.train_range),
                           "by_source": dict(collections.Counter(e.source for e in tr)
                                             .most_common()),
                           "pass": s.train_range[0] <= len(tr) <= s.train_range[1]}
        share = len(va) / (len(tr) + len(va)) if (tr or va) else 0.0
        g["val_share"] = {"value": round(share, 4), "range": list(s.val_share_range),
                          "pass": s.val_share_range[0] <= share <= s.val_share_range[1]}
        g["eval_tools_size"] = {"value": len(ev), "min": s.eval_min, "pass": len(ev) >= s.eval_min}
        cats = collections.Counter(e.category for e in tr)
        top = max(cats.values()) / len(tr) if tr else 0.0
        g["category_cap"] = {"max_share": round(top, 4), "cap": s.category_cap,
                             "pass": bool(tr) and top <= s.category_cap + 1e-9}
        pres = {k: sum(e.flags[k] for e in tr) for k in ("chat_mode", "error_recovery", "compacted")}
        g["presence"] = {**pres, "min": s.min_presence,
                         "pass": all(v >= s.min_presence for v in pres.values())}
        g["spot_check"] = {"status": "NOT RUN",
                           "how": "build --spot-check-server, or build.py spot-check, against "
                                  "llama-server serving the student GGUF"}
        g["overall"] = overall_verdict(g)
        return g

    # ----------------------------------------------------------------- write

    def training_row(self, ex: Example, heavy: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": ex.id, "split": ex.split, "task_id": ex.task_id, "group_id": ex.group,
            "category": ex.category, "source": ex.source, "label": ex.label,
            # provenance the trainer's Max guard reads (it fails closed without it)
            "model": ex.model or None, "max_derived": ex.max_derived,
            # Q6b: the recorded approval a Max-derived row trains under
            "max_derived_approval": ex.approval or None,
            "context_source": ex.context_source, "approximate": True,
            "approx": approx_codes(ex.context_source), "flags": ex.flags,
            "truncated": ({"dropped_messages": ex.dropped_messages,
                           "dropped_turn_steps": ex.dropped_steps,
                           "original_tokens": ex.n_tokens_original}
                          if ex.dropped_messages else None),
            "messages": heavy["messages"], "tools": heavy["tools"],
            "tool_choice": heavy["tool_choice"],
            "chat_template_kwargs": heavy["chat_template_kwargs"],
            **heavy["render"],
        }

    def case_row(self, ex: Example, heavy: dict[str, Any], set_name: str) -> dict[str, Any]:
        msgs = heavy["messages"]
        expect = heavy["expect"]
        if ex.kind == "fail":
            expect = heavy.get("eval_expect") or (
                {"tool": None} if expect.get("tool") is None else {
                    "tool": expect["tool"], "args": {}, "args_match": "subset",
                    "success": dict(expect.get("success") or {"expect_tool": expect["tool"]}),
                })
        row: dict[str, Any] = {
            "id": ex.id, "set": set_name, "messages": msgs[:-1],
            "tools": heavy["tools"], "tool_choice": "none" if ex.flags["chat_mode"] else "auto",
            "expect": expect,
            "meta": {"source": ex.source, "model": ex.model or None,
                     "max_derived": ex.max_derived, "task_id": ex.task_id, "group_id": ex.group,
                     "category": ex.category, "label": ex.label,
                     "context_source": ex.context_source, "approximate": True,
                     "approx": approx_codes(ex.context_source), "trace_id": ex.trace_id,
                     "hard_case": ex.kind == "fail",
                     **({"expect_basis": "teacher tool, failed call"}
                        if ex.kind == "fail" and not heavy.get("eval_expect") else {})},
        }
        if ex.flags["chat_mode"]:
            row["suppress_tools"] = True
            row["tools"] = []
        return row

    def write(self, out: Path) -> dict[str, int]:
        counts: dict[str, int] = {}

        def dump(name: str, rows: Iterator[dict[str, Any]]) -> None:
            n = 0
            tmp = out / (name + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
            tmp.replace(out / name)
            counts[name] = n

        dump("train.jsonl", (self.training_row(e, self.read_heavy(e)) for e in self.final["train"]))
        dump("val.jsonl", (self.training_row(e, self.read_heavy(e)) for e in self.final["val"]))
        dump("val-cases.jsonl", (self.case_row(e, self.read_heavy(e), "val")
                                 for e in self.final["val"]))
        dump("eval-tools.jsonl", (self.case_row(e, self.read_heavy(e), "eval-tools")
                                  for e in self.final["eval-tools"]))

        def longctx() -> Iterator[dict[str, Any]]:
            for e in self.examples + self.unfittable:
                if not e.longctx:
                    continue
                heavy = self.read_heavy(e)
                yield {"id": e.id, "trace_id": e.trace_id, "task_id": e.task_id,
                       "group_id": e.group, "task_split": e.split or "dropped",
                       "fits_truncated": e.fits, "n_tokens": e.n_tokens_original,
                       "n_tokens_estimated": e.omitted_messages > 0,
                       "omitted_messages": e.omitted_messages, "lcm_target": e.lcm_target,
                       "category": e.category,
                       "messages": heavy["original_messages"] or heavy["messages"],
                       "tools": heavy["tools"], "approx": approx_codes(e.context_source)}

        dump("eval-longctx-candidates.jsonl", longctx())
        dump("excluded.jsonl", ({"trace_id": t, "reason": r} for t, r in self.excluded))
        self.close_reader()
        return counts

    # ---------------------------------------------------------------- report

    def report(self, split_info: dict[str, Any], gates: dict[str, Any], *, catalog: str,
               head_tokens: dict[str, int], inputs: dict[str, Any], elapsed: float
               ) -> dict[str, Any]:
        s = self.s
        all_built = self.examples + self.unfittable
        plan: SplitPlan = split_info["plan"]
        groups = collections.Counter(e.group for e in self.examples)
        return {
            "schema": REPORT_SCHEMA,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Q6b: max_derived / tags / private say whether train/val hold a
            # Max-derived row (what a model trained on this dataset inherits);
            # holds_max_rows whether any written set does (eval-tools included)
            **AP.policy_fields(self.train_val_max_derived()),
            "holds_max_rows": self.holds_max_rows(),
            "elapsed_s": round(elapsed, 1),
            "settings": {**{k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in vars(s).items()},
                         # how many sessions --exclude-sessions listed (never the ids)
                         "exclude_sessions": len(s.exclude_sessions),
                         # what the trainer's and the flywheel's Max guards read
                         "teacher_model": self.teacher_model()},
            "catalog": catalog,
            "inputs": inputs,
            "data_policy": self.policy_report(split_info),
            "counts": dict(sorted(self.stats.items())),
            "index_disagreements": dict(self.index_disagree),
            "lcm_match": dict(self.lcm_match),
            "pool": {
                "rows": len(self.examples),
                "tasks": len({e.task_id for e in self.examples}),
                "train_eligible_rows": sum(1 for e in self.examples if e.train_ok),
                "groups": len(groups), "largest_group_rows": max(groups.values(), default=0),
                "groups_too_big_to_hold_out": plan.ineligible_groups,
                "source": dict(collections.Counter(e.source for e in self.examples)
                               .most_common()),
                "category": dict(collections.Counter(e.category for e in self.examples)
                                 .most_common()),
                "label": dict(collections.Counter(e.label for e in self.examples).most_common()),
                "turn_outcome_bad": sum(1 for e in self.examples
                                        if (e.turn_outcome or "") in BAD_TURN_OUTCOMES),
            },
            "tokens": {
                "head_system_plus_tools": head_tokens,
                "rebuilt_original": tstats([e.n_tokens_original for e in all_built],
                                           s.max_tokens),
                "final_in_splits": tstats([e.n_tokens for k in ("train", "val")
                                           for e in self.final[k]], s.max_tokens),
                "over_limit_original": sum(1 for e in all_built if e.longctx),
                "original_estimated": sum(1 for e in all_built if e.omitted_messages),
                "truncated_kept": sum(1 for e in self.examples if e.dropped_messages),
                "split_in_turn": sum(1 for e in self.examples if e.dropped_steps),
                "unfittable": len(self.unfittable),
                "boundary_unstable": sum(1 for e in all_built if not e.boundary_stable),
            },
            "longctx_candidates": {
                "rows": sum(1 for e in all_built if e.longctx),
                "by_task_split": dict(collections.Counter(
                    e.split or "dropped" for e in all_built if e.longctx)),
            },
            "splits": {k: split_summary(v, s.max_tokens) for k, v in self.final.items()},
            "caps": {"train": split_info["train_caps"], "val": split_info["val_caps"]},
            "eval_tools": {"held_out_rows": split_info["held_rows"],
                           "sampled": split_info["eval_sampled"],
                           "fail_cases": split_info["fail_used"],
                           "fail_dropped_task_in_train_or_val":
                               split_info["fail_dropped_task_in_train_or_val"],
                           "held_out_tasks": plan.eval_tasks,
                           # tasks the quotas were taken from (not pinned)
                           "splittable_tasks": plan.total_tasks,
                           "pinned_groups": plan.pinned_groups,
                           "pinned_tasks": plan.pinned_tasks,
                           # Q6b allow: the Max-only hold-out kept as eval data
                           "max_holdout_groups": plan.max_holdout_groups,
                           "max_holdout_tasks": plan.max_holdout_tasks,
                           "eval_holdout_tasks_listed": len(self.eval_holdout_tasks),
                           "eval_holdout_groups": plan.listed_holdout_groups},
            "gates": gates,
        }

    def train_val_max_derived(self) -> bool:
        """Train or val holds a Max-derived row: the dataset, and anything
        trained on it, is Max-derived (Q6b)."""
        return any(e.max_derived for k in ("train", "val") for e in self.final[k])

    def holds_max_rows(self) -> bool:
        """Any set this build writes holds a Max-derived row (eval-tools and the
        long-context candidates are built from Max traces too)."""
        return any(e.max_derived for e in self.examples + self.fails + self.unfittable)

    def share(self) -> dict[str, Any]:
        """Rows and percent per share class (Max-derived / the local 27B
        teacher's / other) per split, with the approval Max rows train under."""
        by_split: dict[str, Any] = {}
        for k, rows in self.final.items():
            c = collections.Counter(share_class(e.max_derived, e.source) for e in rows)
            n = len(rows)
            by_split[k] = {"rows": n, **{cls: {"rows": c[cls],
                                               "pct": round(100.0 * c[cls] / n, 2) if n else 0.0}
                                         for cls in SHARE_CLASSES}}
        return {"classes": list(SHARE_CLASSES), "by_split": by_split,
                "max_derived_allowed": self.max_allowed,
                "approval": self.s.max_derived_approval if self.max_allowed else None}

    def dataset_manifest(self, out: Path, written: dict[str, int], report: dict[str, Any],
                         report_path: Path) -> dict[str, Any]:
        """``dataset-manifest.json``: what this dataset is, for the trainer, the
        publish guard and the rental tool (Q6b): the Max-derived flag and tag,
        the recorded approval, the share table and each written file's rows
        and sha256. Aggregates and hashes only."""
        files = {}
        for name, n in sorted(written.items()):
            p = out / name
            if p.is_file():
                digest = hashlib.sha256()
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        digest.update(chunk)
                files[name] = {"rows": n, "sha256": digest.hexdigest()}
        dp = report["data_policy"]
        return {
            "schema": AP.DATASET_MANIFEST_SCHEMA,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **AP.policy_fields(self.train_val_max_derived()),
            "holds_max_rows": self.holds_max_rows(),
            "max_derived_allow": dp["max_derived_allow"],
            "teacher_model": report["settings"]["teacher_model"],
            "share": dp["share"],
            "report": report_path.name if report_path.parent == out else str(report_path),
            "files": files,
            "gates_overall": report["gates"].get("overall"),
        }

    def policy_report(self, split_info: dict[str, Any]) -> dict[str, Any]:
        """The data policy and what it did, per source (aggregates only)."""
        s = self.s
        plan: SplitPlan = split_info["plan"]
        per_split: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        classes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for k, rows in self.final.items():
            for e in rows:
                per_split[e.source][k] += 1
                classes[e.source][share_class(e.max_derived, e.source)] += 1
        split_rows = {k: len(rows) for k, rows in self.final.items()}
        sources: dict[str, Any] = {}
        for src in sorted(self.by_source):
            c = self.by_source[src]

            def part(prefix: str, c: collections.Counter = c) -> dict[str, int]:
                return {k[len(prefix):]: v for k, v in sorted(c.items()) if k.startswith(prefix)}

            excluded = part("excluded:")
            sources[src] = {
                "listed": matches_source(src, s.train_sources) and not names_max(src),
                "input": c["input"], "accepted": c["accepted"],
                "excluded": sum(excluded.values()), "excluded_by_reason": excluded,
                "verification": part("verification:"),
                "train_eligible": c["train_eligible"], "max_derived": c["max_derived"],
                "max_derived_allowed": c["max_derived_allowed"],
                "chat_mode": c["chat_mode"],
                "models": dict(self.models_by_source[src].most_common()),
                **{k: per_split[src][k] for k in SPLITS},
                # this source's rows as a percent of each split, and its share class
                "share": {k: round(100.0 * per_split[src][k] / split_rows[k], 2)
                          if split_rows[k] else 0.0 for k in SPLITS},
                "class": (classes[src].most_common(1)[0][0] if classes[src]
                          else share_class(c["max_derived"] > 0, src)),
            }
        allow = AP.approval_record(s.max_derived_approval if self.max_allowed else None)
        train_val = self.final["train"] + self.final["val"]
        allow.update({
            "rows_in_train_val": sum(1 for e in train_val if e.max_derived),
            "covered_in_train_val": sum(1 for e in train_val if e.max_derived and e.approval),
            # --exclude-sessions (the Token-Plan incident session): listed, rows dropped
            "excluded_sessions_listed": len(s.exclude_sessions),
            "excluded_session_rows": self.stats["excluded:excluded_session"],
            # gym rows built from one of those sessions' prompts (task group, partner, padding)
            "excluded_session_prompts": len(self.excluded_prompts),
            "excluded_session_task_rows": self.stats["excluded:excluded_session_task"],
            "max_holdout_groups": plan.max_holdout_groups,
            "max_holdout_tasks": plan.max_holdout_tasks,
        })
        rule = POLICY_RULE + ("; " + MAX_ALLOW_RULE if self.max_allowed else "")
        return {
            "train_sources": list(s.train_sources) if s.train_sources is not None else None,
            "train_models": list(s.train_models), "judge_models": list(self.judge_models),
            "max_model": s.max_model, "teacher_model": self.teacher_model(),
            "rule": rule if s.train_sources is not None else
            "none (library default): every accepted row may train",
            # Q6b: the explicit, recorded allow (or its absence) and the shares
            "max_derived_allow": allow,
            "share": self.share(),
            "sources": sources,
            "train_source_rows_held_out_with_max_rows": plan.pinned_train_rows,
            "not_train_ok_in_train_val_groups": split_info.get(
                "not_train_ok_in_train_val_groups", 0),
        }


def overall_verdict(gates: dict[str, Any]) -> str:
    """FAIL if any gate failed; INCOMPLETE while the spot check (a Phase 3 gate
    too) has not passed; PASS only then."""
    if not all(v.get("pass") for k, v in gates.items()
               if k != "overall" and isinstance(v, dict) and "pass" in v):
        return "FAIL"
    if (gates.get("spot_check") or {}).get("status") != "PASS":
        return "INCOMPLETE"
    return "PASS"


def head_token_counts(student: Any, pack: dict[str, Any]) -> dict[str, int]:
    """Tokens of the system block (system prompt + tools) for each catalog."""
    from prometheus.providers.base import ApiMessageRequest
    from prometheus.providers.llama_cpp import LlamaCppProvider

    out = {}
    system = pack["system_prompt"].replace(pack["date_sentinel"], "2026-01-01")
    for name in ("stubs", "full"):
        payload = LlamaCppProvider(suppress_thinking=True)._build_request_payload(ApiMessageRequest(
            model="local", messages=[{"role": "user", "content": "x"}], system_prompt=system,
            max_tokens=16, tools=list(pack["tools"][name])))
        text = R.render_prompt(student, payload["messages"], payload.get("tools"),
                               payload.get("chat_template_kwargs"))
        head = text[: text.find("<|im_end|>\n") + len("<|im_end|>\n")]
        out[name] = student.count(head)
    out["system_prompt_only"] = student.count(system)
    return out


def file_facts(paths: list[Path]) -> dict[str, Any]:
    return {"files": len(paths), "bytes": sum(p.stat().st_size for p in paths)}


class _CountingHandler(logging.Handler):
    """Library log records during a build: counted by logger, never printed
    (a record could quote conversation text)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: collections.Counter = collections.Counter()

    def emit(self, record: logging.LogRecord) -> None:
        self.counts[f"{record.name}:{record.levelname}"] += 1


def cmd_build(a: argparse.Namespace) -> int:
    t0 = time.time()
    golden = expand(a.traces or [DEFAULT_TRACES])
    extra = expand(a.extra_traces or [])
    fail = expand(a.fail_traces or [])
    dupes = duplicate_names(golden + extra + fail)
    if dupes:
        print(f"error: input file names must be unique (trace ids are <file name>:<line>);"
              f" given more than once: {', '.join(dupes)}", file=sys.stderr)
        return 2
    problem = max_allow_problem(a)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    excluded_sessions: set[str] = set()
    if a.exclude_sessions:
        try:
            excluded_sessions = read_session_list(Path(a.exclude_sessions).expanduser())
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            print(f"error: --exclude-sessions {a.exclude_sessions}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 2
    holdout_tasks: set[str] = set()
    for path in a.eval_holdout or []:
        try:
            holdout_tasks |= read_holdout_tasks(Path(path).expanduser())
        except (OSError, ValueError) as exc:
            print(f"error: --eval-holdout {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
    if a.allow_max_derived:
        # the explicit, logged allow (the reference is recorded in report.json
        # and dataset-manifest.json too); counts only, never row text
        print(f"ALLOW (Q6b): Max-derived rows may train under the operator's approval "
              f"{a.max_derived_approval!r}; every output is tagged {AP.TAG} and stays private; "
              f"{len(excluded_sessions)} listed session(s) excluded", file=sys.stderr)
    policy: dict[str, tuple[str, ...]] = {}
    for flag, given, default in (("--train-sources", a.train_sources, DEFAULT_TRAIN_SOURCES),
                                 ("--train-models", a.train_models, DEFAULT_TRAIN_MODELS),
                                 ("--judge-models", a.judge_models, None)):
        if given is None and default is None:
            continue
        patterns = tuple(p.strip() for p in (given or default or ()) if p.strip())
        if not patterns:
            print(f"error: {flag} needs at least one pattern", file=sys.stderr)
            return 2
        refused = [p for p in patterns if names_max(p) or p.lower() == a.max_model.lower()]
        if refused:
            print(f"error: {flag} may not name Max: Max-derived rows never train and the Max "
                  f"key never judges (DISTILL-OPEN-QUESTIONS #6); refused: {', '.join(refused)}",
                  file=sys.stderr)
            return 2
        policy[flag] = patterns
    handler = _CountingHandler()
    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    s = Settings(
        max_tokens=a.max_tokens, category_cap=a.category_cap, val_frac=a.val_frac,
        eval_task_frac=a.eval_task_frac, eval_max_per_task=a.eval_max_per_task,
        holdout_max_group_share=a.holdout_max_group_share, prefix_n=a.prefix_n, seed=a.seed,
        eval_success=a.eval_success, max_model=a.max_model,
        train_sources=policy["--train-sources"], train_models=policy["--train-models"],
        judge_models=policy.get("--judge-models"),
        exclude_bad_turns=a.exclude_bad_turns,
        allow_max_derived=a.allow_max_derived, max_derived_approval=a.max_derived_approval,
        exclude_sessions=tuple(sorted(excluded_sessions)),
        train_range=(a.train_range[0], a.train_range[1]),
        val_share_range=(a.val_share_range[0], a.val_share_range[1]), eval_min=a.eval_min,
        min_presence=a.min_presence, lcm_max_messages=a.lcm_max_messages,
        prewindow_chars_per_token=a.prewindow_chars_per_token,
        split_turns=not a.no_split_turns, limit=a.limit,
    )
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pack = load_pack(a.pack)
    catalog = pack["runtime_catalog"] if a.catalog == "auto" else a.catalog
    student = R.Student.load(a.tokenizer_dir)
    stats: collections.Counter = collections.Counter()
    checks: dict[str, Any] = {}
    verdicts = load_verdicts(a.verdicts or [], stats, checks)
    lcm = LcmSnapshot(a.lcm_snapshot) if a.lcm_snapshot else None
    spool = None if a.dry_run else out / "spool.jsonl"
    # a pre-pass: the listed sessions' prompts, so a gym row built from one (no
    # session id of its own) is dropped wherever it comes in the input order
    excluded_prompts = (excluded_session_prompts(excluded_sessions, golden + extra + fail, lcm)
                        if excluded_sessions else set())
    b = Builder(student=student, pack=pack, catalog=catalog, settings=s,
                index=load_index(a.index), verdicts=verdicts, lcm=lcm, spool=spool,
                verdict_checks=checks, eval_holdout_tasks=holdout_tasks,
                excluded_prompts=excluded_prompts)
    b.stats.update(stats)
    n = 0
    for paths, kind in ((golden, "golden"), (extra, "extra"), (fail, "fail")):
        for trace in iter_traces(paths, kind, b.stats):
            if s.limit is not None and n >= s.limit:
                break
            b.add(trace)
            n += 1
            if a.progress and n % 500 == 0:
                print(f"... {n} traces, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    b.close_spool()
    split_info = b.split()
    gates = b.gates()
    inputs = {"golden": file_facts(golden), "extra": file_facts(extra), "fail": file_facts(fail),
              "fail_traces_present": bool(fail), "verdicts": len(verdicts),
              "lcm_snapshot": bool(lcm), "pack_commit": pack.get("prometheus_commit"),
              "dry_run": a.dry_run}
    if not a.dry_run:
        inputs["written"] = b.write(out)
        if spool is not None and not a.keep_spool:
            spool.unlink(missing_ok=True)
        if a.spot_check_server:
            gates["spot_check"] = spot_check_gate(a.spot_check_server, out / "train.jsonl",
                                                  student)
            gates["overall"] = overall_verdict(gates)
    report = b.report(split_info, gates, catalog=catalog,
                      head_tokens=head_token_counts(student, pack), inputs=inputs,
                      elapsed=time.time() - t0)
    report["library_log_records"] = dict(handler.counts)
    report_path = Path(a.report).expanduser() if a.report else out / "report.json"
    report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"report": str(report_path), "max_derived": report["max_derived"],
                               "tags": report["tags"]}
    if not a.dry_run:
        manifest = b.dataset_manifest(out, inputs["written"], report, report_path)
        (out / DATASET_MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n",
                                            encoding="utf-8")
        summary["dataset_manifest"] = str(out / DATASET_MANIFEST)
    summary["gates"] = {k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
                        for k, v in gates.items()}
    print(json.dumps(summary, indent=1))
    return 0 if gates["overall"] == "PASS" else 1


def max_allow_problem(a: argparse.Namespace) -> str | None:
    """Why the command line's Max-derived allow (Q6b) is unusable, or None.
    The flag and the approval reference go together, the reference is not
    blank, and the output directory (and a ``--report`` path) carries the
    ``max-derived`` tag so the dataset's name says what it is."""
    allow, ref = bool(a.allow_max_derived), a.max_derived_approval
    if allow != (ref is not None):
        return (f"{AP.BUILD_ALLOW_FLAG} and {AP.BUILD_APPROVAL_FLAG} go together: the allow for "
                "Max-derived rows to train is explicit and records the operator's approval "
                f"(DISTILL-OPEN-QUESTIONS Q6b), e.g. {AP.BUILD_ALLOW_FLAG} "
                f"{AP.BUILD_APPROVAL_FLAG} \"operator Q6b 2026-10-30\"")
    if not allow:
        return None
    if not ref.strip():
        return f"{AP.BUILD_APPROVAL_FLAG} needs the approval reference (not blank)"
    out = Path(a.out).expanduser()
    if not AP.has_tag(out.name):
        return (f"with {AP.BUILD_ALLOW_FLAG} the output is Max-derived: its directory name must "
                f"carry the {AP.TAG} tag (e.g. --out {AP.tag_path(out)})")
    if a.report and not AP.path_has_tag(Path(a.report).expanduser()):
        return (f"with {AP.BUILD_ALLOW_FLAG} the report's path must carry the {AP.TAG} tag "
                f"(e.g. --report {AP.tag_path(Path(a.report))})")
    if not a.exclude_sessions:
        return (f"with {AP.BUILD_ALLOW_FLAG}, --exclude-sessions FILE is required: it lists the "
                "2026-10-13 Token-Plan incident session (DISTILL-OPEN-QUESTIONS #6), whose rows "
                "never enter training data")
    return None


def read_session_list(path: Path) -> set[str]:
    """Session ids, one per line (blank lines and ``#`` comments skipped);
    a file that lists none is an error, so an empty list cannot stand in for
    the exclusion the allow requires."""
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            ids.add(text)
    if not ids:
        raise ValueError("lists no session id")
    return ids


def excluded_session_prompts(sessions: set[str], paths: list[Path], lcm: LcmSnapshot | None) -> set[str]:
    """Task proxies (``h(norm(text))``, taskset_gen's ``pattern_hash``) of every user
    message the listed sessions hold: in the input traces (golden / fail files) and,
    with a snapshot, in their user rows there. A gym row carries no session id, so
    this is how a row built from one of these prompts is found (``row_task_prompts``)."""
    out: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if str((raw.get("_meta") or {}).get("session_id") or "").strip() not in sessions:
                        continue
                    msgs = raw.get("messages") or []
                except (ValueError, AttributeError):
                    continue
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "user":
                        text = content_text(m.get("content"))
                        if text.strip():
                            out.add(h(norm(text)))
    if lcm is not None:
        for sid in sorted(sessions):
            for row in lcm.con.execute("SELECT content, content_json FROM lcm_messages "
                                       "WHERE session_id = ? AND role = 'user'", (sid,)):
                texts = [row["content"] or ""]
                try:
                    blocks = json.loads(row["content_json"]) if row["content_json"] else None
                except ValueError:
                    blocks = None
                if isinstance(blocks, list):
                    texts.append(content_text(blocks))
                out.update(h(norm(t)) for t in texts if t.strip())
    return out


def row_task_prompts(meta: dict[str, Any]) -> set[str]:
    """The real prompts a gym row was built from: its task group (``tg-<hash>``),
    a recombined partner's group and the groups of its compaction padding."""
    ids = [meta.get("task_id"), meta.get("partner_task_id")]
    pad = meta.get("compaction_pad")
    if isinstance(pad, dict) and isinstance(pad.get("task_groups"), list):
        ids += pad["task_groups"]
    return {str(x)[3:] for x in ids if isinstance(x, str) and x.startswith("tg-")}


def read_holdout_tasks(path: Path) -> set[str]:
    """Task ids (Phase 1 prompt hashes) of an earlier eval set: score.py case
    rows (``meta.task_id``) or build.py training rows (``task_id``)."""
    tasks: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta") if isinstance(row, dict) else None
            task = (meta or {}).get("task_id") if isinstance(meta, dict) else None
            task = task or (row.get("task_id") if isinstance(row, dict) else None)
            if isinstance(task, str) and task:
                tasks.add(task)
    if not tasks:
        raise ValueError("no task ids (meta.task_id or task_id) in it")
    return tasks


def cmd_pack(a: argparse.Namespace) -> int:
    pack = build_pack(a.config, a.cwd)
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pack, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "commit": pack["prometheus_commit"],
                      "system_prompt_chars": len(pack["system_prompt"]),
                      "tools": len(pack["tools"]["full"]), "stubs": len(pack["stub_names"]),
                      "runtime_catalog": pack["runtime_catalog"]}, indent=1))
    return 0


def is_chat_row(row: dict[str, Any]) -> bool:
    """A written row of a chat-mode request (no tools offered)."""
    return bool((row.get("flags") or {}).get("chat_mode") or row.get("suppress_tools"))


def spot_sample(rows: list[dict[str, Any]], n: int, seed: str) -> list[dict[str, Any]]:
    """*n* rows in seeded order, with at least one chat-mode row and one row
    that offers tools when the file holds both (the two render differently:
    with and without the tools block)."""
    def key(r: dict[str, Any]) -> str:
        return seeded(seed, str(r.get("id")))

    order = sorted(rows, key=key)
    must = [r for r in (next((r for r in order if is_chat_row(r)), None),
                        next((r for r in order if not is_chat_row(r)), None)) if r is not None]
    must = must[:max(0, n)]
    rest = [r for r in order if not any(r is m for m in must)]
    return sorted(must + rest[:max(0, n - len(must))], key=key)


def spot_check_rows(server: str, rows_path: str | Path, student: Any, *, n: int = 5,
                    seed: str = "spot-v1", show_diff: bool = False, post: Any = None
                    ) -> list[dict[str, Any]]:
    """Render N rows ourselves and with llama-server; compare byte for byte."""
    rows = [json.loads(line) for line in Path(rows_path).expanduser().read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = spot_sample(rows, n, seed)
    results = []

    def diff(res: dict[str, Any], name: str, ours: str, theirs: str) -> bool:
        at = R.first_mismatch(ours, theirs)
        if at is not None:
            res[name + "_first_mismatch"] = at
            if show_diff:
                res[name + "_ours"] = ours[max(0, at - 40): at + 40]
                res[name + "_other"] = theirs[max(0, at - 40): at + 40]
        return at is None

    for r in rows:
        # a training row ends with the teacher's turn; a case row is the request
        training = bool(r.get("split"))
        msgs = r["messages"][:-1] if training else r["messages"]
        kwargs = r.get("chat_template_kwargs") or dict(R.THINKING_OFF)
        payload = {"messages": msgs, "tools": r.get("tools") or None,
                   "tool_choice": r.get("tool_choice") or ("auto" if r.get("tools") else None),
                   "chat_template_kwargs": kwargs}
        ours = R.render_prompt(student, msgs, r.get("tools"), kwargs)
        theirs = R.apply_template(server, payload, post=post)
        res: dict[str, Any] = {"id": r.get("id"), "ours_chars": len(ours),
                               "server_chars": len(theirs)}
        ok = res["prompt_match"] = diff(res, "prompt", ours, theirs)
        if training:
            text = r["text"]
            # The stored text is what trains: its request part must be this
            # render, or the row was built with another template.
            ok &= diff(res, "stored", text[: r["loss_spans_char"][0][0]], ours)
            res["stored_match"] = res.get("stored_first_mismatch") is None
            calls = r["messages"][-1].get("tool_calls") or []
            if calls:
                # The loss-bearing turn as the server renders it in history: the
                # row plus a stub result per call (a trailing assistant message
                # would be taken as a prefill and rendered differently).
                stubs = [{"role": "tool", "tool_call_id": tc.get("id"), "content": "ok"}
                         for tc in calls]
                full = R.apply_template(server, {**payload, "messages": r["messages"] + stubs},
                                        post=post, add_generation_prompt=False)
                ok &= diff(res, "completion", text, full[: len(text)])
                res["completion_match"] = res.get("completion_first_mismatch") is None
            else:
                res["completion_match"] = "not checkable (no call: the server would prefill)"
        res["match"] = ok
        results.append(res)
    return results


def spot_check_gate(server: str, rows_path: Path, student: Any, *, post: Any = None
                    ) -> dict[str, Any]:
    """The spot check as a build gate (counts only: the report holds no ids)."""
    try:
        results = spot_check_rows(server, rows_path, student, post=post)
    except Exception as exc:  # a server that cannot answer is not a pass
        return {"status": "ERROR", "error": type(exc).__name__, "pass": False}
    ok = bool(results) and all(x["match"] for x in results)
    return {"status": "PASS" if ok else "FAIL", "rows": len(results),
            "mismatched": sum(1 for x in results if not x["match"]), "pass": ok}


def cmd_spot_check(a: argparse.Namespace, *, post: Any = None) -> int:
    """Render N rows ourselves and with llama-server; compare byte for byte."""
    student = R.Student.load(a.tokenizer_dir)
    results = spot_check_rows(a.server, a.rows, student, n=a.n, seed=a.seed,
                              show_diff=a.show_diff, post=post)
    ok = bool(results) and all(x["match"] for x in results)
    print(json.dumps({"spot_check": "PASS" if ok else "FAIL", "rows": results}, indent=1))
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("pack", help="build the request pack (system prompt + tools)")
    pk.add_argument("--config", required=True)
    pk.add_argument("--cwd", required=True, help="the daemon's working directory")
    pk.add_argument("--out", required=True)

    bd = sub.add_parser("build", help="build train / val / eval-tools")
    bd.add_argument("--traces", action="append", help=f"glob (default {DEFAULT_TRACES})")
    bd.add_argument("--index")
    bd.add_argument("--pack", required=True)
    bd.add_argument("--tokenizer-dir", required=True)
    bd.add_argument("--out", required=True)
    bd.add_argument("--report")
    bd.add_argument("--verdicts", action="append")
    bd.add_argument("--extra-traces", action="append")
    bd.add_argument("--fail-traces", action="append")
    bd.add_argument("--lcm-snapshot")
    bd.add_argument("--lcm-max-messages", type=int, default=400)
    bd.add_argument("--prewindow-chars-per-token", type=int, default=6)
    bd.add_argument("--no-split-turns", action="store_true",
                    help="never drop earlier steps of the current turn")
    bd.add_argument("--catalog", choices=("auto", "stubs", "full"), default="auto")
    bd.add_argument("--max-tokens", type=int, default=LIMIT_TOKENS)
    bd.add_argument("--category-cap", type=float, default=0.35)
    bd.add_argument("--val-frac", type=float, default=0.03)
    bd.add_argument("--eval-task-frac", type=float, default=0.10)
    bd.add_argument("--eval-max-per-task", type=int, default=20)
    bd.add_argument("--holdout-max-group-share", type=float, default=0.02)
    bd.add_argument("--prefix-n", type=int, default=4)
    bd.add_argument("--seed", default="phase3-v1")
    bd.add_argument("--eval-success", choices=("presence", "exact"), default="presence")
    bd.add_argument("--max-model", "--teacher-model", dest="max_model", default=TEACHER_MODEL,
                    help="the hosted Max teacher of the harvested traces: its rows are kept "
                         f"as eval-only rows (default {TEACHER_MODEL})")
    bd.add_argument("--train-sources", nargs="+", metavar="SOURCE",
                    help="_meta.source tags (fnmatch patterns) whose verified rows may train; "
                         "every other row is eval-only (default: "
                         f"{' '.join(DEFAULT_TRAIN_SOURCES)}). A source naming Max is refused.")
    bd.add_argument("--train-models", nargs="+", metavar="MODEL",
                    help="_meta.model patterns (fnmatch, any case) a train-source row may come "
                         "from: the local teacher (default: "
                         f"{' '.join(DEFAULT_TRAIN_MODELS)}). A model naming Max is refused.")
    bd.add_argument("--judge-models", nargs="+", metavar="MODEL",
                    help="judge model patterns a train-source row's verification record may "
                         "name (default: the --train-models). A model naming Max is refused.")
    bd.add_argument("--allow-max-derived", action="store_true",
                    help="Q6b: let Max-derived rows train a PRIVATE model under the same gates; "
                         "needs --max-derived-approval, an --out named *max-derived*, and tags "
                         "every output max-derived")
    bd.add_argument("--max-derived-approval", metavar="REF",
                    help="the operator's approval for --allow-max-derived, recorded verbatim "
                         "(e.g. \"operator Q6b 2026-10-30\"); only together with it")
    bd.add_argument("--exclude-sessions", metavar="FILE",
                    help="session ids (one per line, # comments) whose rows never enter the "
                         "dataset; required with --allow-max-derived: it lists the 2026-10-13 "
                         "Token-Plan incident session (OPEN-QUESTIONS #6)")
    bd.add_argument("--eval-holdout", action="append", metavar="CASES",
                    help="an earlier eval set (eval-tools.jsonl / val-cases.jsonl): its tasks "
                         "stay in eval-tools, whole groups")
    bd.add_argument("--exclude-bad-turns", action="store_true")
    bd.add_argument("--train-range", type=int, nargs=2, default=[5000, 10000])
    bd.add_argument("--val-share-range", type=float, nargs=2, default=[0.02, 0.05])
    bd.add_argument("--eval-min", type=int, default=200)
    bd.add_argument("--min-presence", type=int, default=1)
    bd.add_argument("--limit", type=int)
    bd.add_argument("--dry-run", action="store_true", help="report only, no dataset files")
    bd.add_argument("--spot-check-server",
                    help="llama-server (student GGUF, --jinja) to spot-check train.jsonl against "
                         "after writing it; without it the build is INCOMPLETE at best")
    bd.add_argument("--keep-spool", action="store_true")
    bd.add_argument("--progress", action="store_true")

    sc = sub.add_parser("spot-check", help="compare renders with llama-server /apply-template")
    sc.add_argument("--server", required=True)
    sc.add_argument("--rows", required=True)
    sc.add_argument("--tokenizer-dir", required=True)
    sc.add_argument("--n", type=int, default=5)
    sc.add_argument("--seed", default="spot-v1")
    sc.add_argument("--show-diff", action="store_true",
                    help="print 80 characters around a mismatch (trace text: host only)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    if a.cmd == "pack":
        return cmd_pack(a)
    if a.cmd == "build":
        return cmd_build(a)
    return cmd_spot_check(a)


if __name__ == "__main__":
    sys.exit(main())
