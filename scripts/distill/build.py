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
        [--extra-traces '~/distill/phase2b/*.jsonl'] [--lcm-snapshot ~/distill/snapshot/lcm.db] \\
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
  with the index fails the ``verdicts`` gate. Unjudged golden traces fall back
  to the tool-success label (the runbook's rule): every golden trace's call ran
  without error, by construction.
* Input file names must be unique across ``--traces``, ``--extra-traces`` and
  ``--fail-traces`` (a trace is ``<file name>:<line>``); a repeat is an input
  error.
* ``--extra-traces``: Phase 2b teacher traces in the same shape, tagged
  ``_meta.source = "max-gym-<date>"``. They must carry a pass verdict (in
  ``--verdicts`` or ``_meta.verdict``); unverified rows are excluded. They are
  deduped against everything before them and may carry ``_meta.category``,
  ``_meta.suppress_tools`` (chat-mode), ``_meta.compacted``.
* ``--fail-traces``: rebuilt failed teacher calls (same shape), used only as
  hard eval-tools cases, never for training. A row may carry its own
  ``expect``; otherwise the case expects the tool the teacher picked, called
  with schema-valid arguments.
* ``--lcm-snapshot``: a READ-ONLY snapshot of the conversation store (never the
  live file). When given, each trace's context is rebuilt from the structured
  ``content_json`` blocks (earlier tool calls and tool results included) up to
  the teacher's own assistant message; otherwise the stored ≤12-message
  window is used. ``context_source`` on each row says which.

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
* Category cap: no category above ``--category-cap`` (35%) of train; the
  largest fixed point is kept, round-robin over tasks so task diversity
  survives; val keeps the same per-category share train kept, so it mirrors
  train's mix. What was dropped is reported.
* Gates (in code; FAIL is reported, never worked around): no task / group /
  prefix / exact / prompt+call hash, and no session user message of a final
  request, in two splits; every verdict row applied; train size in
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
TEACHER_MODEL = "qwen3.8-max"
DEFAULT_TRACES = "~/.prometheus/trajectories/golden_traces_*.jsonl"
LIMIT_TOKENS = 16384
SPLITS = ("train", "val", "eval-tools")
BAD_TURN_OUTCOMES = frozenset({"abandoned", "forced_stop", "forced stop"})
#: Phase 1's measured density, used only to estimate history left unrendered.
CHARS_PER_TOKEN = 3.93


def _load_render() -> Any:
    name = "distill_render"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("render.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_render()


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
    """
    out: dict[str, str] = {}
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
        chat_mode = bool(meta.get("suppress_tools")) or not target.get("tool_calls")
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
        return Request(messages=wire, tools=payload.get("tools") or [],
                       tool_choice=payload.get("tool_choice", "none" if chat_mode else "auto"),
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
    """Join tasks that share a context prefix or a session user message of
    their final requests; group = smallest task id in it."""
    uf = UnionFind()
    for ex in examples:
        uf.union("t:" + ex.task_id, "p:" + ex.prefix)
        for key in ex.ctx_keys:
            uf.union("t:" + ex.task_id, "c:" + key)
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


def plan_splits(pool: list[Example], *, seed: str, eval_task_frac: float, val_frac: float,
                max_group_share: float) -> SplitPlan:
    rows = collections.Counter(ex.group for ex in pool)
    tasks: dict[str, set[str]] = collections.defaultdict(set)
    for ex in pool:
        tasks[ex.group].add(ex.task_id)
    total_rows = len(pool)
    total_tasks = len({ex.task_id for ex in pool})
    order = sorted(rows, key=lambda g: seeded(seed, g))
    limit = max_group_share * total_rows
    eligible = {g for g in order if rows[g] <= limit}
    plan: dict[str, str] = {}
    eval_tasks = 0
    want_eval = eval_task_frac * total_tasks
    for g in order:
        if g in eligible and eval_tasks < want_eval:
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
    return SplitPlan(plan, len(order) - len(eligible), eval_tasks, total_tasks)


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
    keys = {"task": lambda e: [e.task_id], "group": lambda e: [e.group],
            "prefix": lambda e: [e.prefix], "exact": lambda e: [e.h_exact],
            "prompt_call": lambda e: [e.h_prompt_call], "context": lambda e: e.ctx_keys}
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
    return {"pass": ok, "overlaps": out}


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
        "chat_mode": sum(e.flags["chat_mode"] for e in rows),
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
    teacher_model: str = TEACHER_MODEL
    exclude_bad_turns: bool = False
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
                 verdict_checks: dict[str, Any] | None = None) -> None:
        self.student = student
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

    def exclude(self, trace: Trace, reason: str) -> None:
        self.stats[f"excluded:{reason}"] += 1
        self.excluded.append((trace.trace_id, reason))

    def add(self, trace: Trace) -> None:
        self.stats[f"input_{trace.kind}"] += 1
        if trace.kind == "golden":
            self._file_ordinal[trace.file] += 1
        meta = trace.meta
        if (trace.kind != "fail" or meta.get("model")) and meta.get("model") != self.s.teacher_model:
            self.exclude(trace, "non_teacher_model")
            return
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
        # dedupe exactly as Phase 1: first teacher row per normalized prompt + call
        if trace.kind == "fail":
            if keys.h_exact in self.seen_fail:
                self.exclude(trace, "duplicate")
                return
            self.seen_fail.add(keys.h_exact)
        else:
            if keys.h_prompt_call in self.seen_prompt_call:
                self.exclude(trace, "duplicate")
                return
            self.seen_prompt_call.add(keys.h_prompt_call)
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
        if trace.kind == "fail":
            label = "hard-case"
        elif verdict == "fail":
            self.exclude(trace, "judge_fail")
            return
        elif verdict == "pass":
            label = "judge-pass"
        elif trace.kind == "extra":
            self.exclude(trace, "unverified")
            return
        else:
            label = "tool-success"
        outcome = (idx or {}).get("turn_outcome")
        if self.s.exclude_bad_turns and label == "tool-success" and outcome in BAD_TURN_OUTCOMES:
            self.exclude(trace, "bad_turn_outcome")
            return
        if trace.kind == "extra" and meta.get("category"):
            category = str(meta["category"])
        else:
            category, _ = classify(trace.raw, keys.prompt, keys.args,
                                   human_surface=True if trace.kind == "extra" else None)
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
        ex = Example(
            id=("p3f-" if trace.kind == "fail" else "p3-") + h(trace.trace_id),
            trace_id=trace.trace_id, kind=trace.kind,
            source=str(meta.get("source") or ("max-golden" if trace.kind == "golden" else trace.kind)),
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
            lcm_target=req.lcm_target, ctx_keys=ctx_keys,
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
        plan = plan_splits(pool, seed=s.seed, eval_task_frac=s.eval_task_frac,
                           val_frac=s.val_frac, max_group_share=s.holdout_max_group_share)
        for ex in pool:
            ex.split = plan.groups[ex.group]
        train_all = [e for e in pool if e.split == "train"]
        val_all = [e for e in pool if e.split == "val"]
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
        }

    def gates(self) -> dict[str, Any]:
        s = self.s
        tr, va, ev = (self.final[k] for k in SPLITS)
        g: dict[str, Any] = {"leakage": leakage(self.final)}
        vg = {"files_missing": self.stats["verdict_files_missing"],
              "rows_unusable": self.stats["verdict_rows_unusable"],
              "unmatched": len(set(self.verdicts) - self.verdicts_used),
              "index_mismatch": self.stats["verdict_index_mismatch"]}
        g["verdicts"] = {"loaded": len(self.verdicts), **vg,
                         "pass": not any(vg.values())}
        g["train_size"] = {"value": len(tr), "range": list(s.train_range),
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
            "meta": {"source": ex.source, "task_id": ex.task_id, "group_id": ex.group,
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
            "elapsed_s": round(elapsed, 1),
            "settings": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in vars(s).items()},
            "catalog": catalog,
            "inputs": inputs,
            "counts": dict(sorted(self.stats.items())),
            "index_disagreements": dict(self.index_disagree),
            "lcm_match": dict(self.lcm_match),
            "pool": {
                "rows": len(self.examples), "tasks": plan.total_tasks,
                "groups": len(groups), "largest_group_rows": max(groups.values(), default=0),
                "groups_too_big_to_hold_out": plan.ineligible_groups,
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
                           "held_out_tasks": plan.eval_tasks},
            "gates": gates,
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
        eval_success=a.eval_success, teacher_model=a.teacher_model,
        exclude_bad_turns=a.exclude_bad_turns, train_range=(a.train_range[0], a.train_range[1]),
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
    b = Builder(student=student, pack=pack, catalog=catalog, settings=s,
                index=load_index(a.index), verdicts=verdicts, lcm=lcm, spool=spool,
                verdict_checks=checks)
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
    print(json.dumps({"report": str(report_path), "gates": {
        k: (v.get("pass") if isinstance(v, dict) and "pass" in v else v)
        for k, v in gates.items()}}, indent=1))
    return 0 if gates["overall"] == "PASS" else 1


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


def spot_check_rows(server: str, rows_path: str | Path, student: Any, *, n: int = 5,
                    seed: str = "spot-v1", show_diff: bool = False, post: Any = None
                    ) -> list[dict[str, Any]]:
    """Render N rows ourselves and with llama-server; compare byte for byte."""
    rows = [json.loads(line) for line in Path(rows_path).expanduser().read_text(
        encoding="utf-8").splitlines() if line.strip()]
    rows = sorted(rows, key=lambda r: seeded(seed, str(r.get("id"))))[:n]
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
    bd.add_argument("--teacher-model", default=TEACHER_MODEL)
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
