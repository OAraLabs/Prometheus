#!/usr/bin/env python3
"""The weekly fine-tuning flywheel: harvest, verify, build, train, quantize, score, report.

Run::

    python3 -m prometheus.jobs.flywheel_weekly                # the timer, inside the window
    python3 -m prometheus.jobs.flywheel_weekly --dry-run      # any time: the plan, nothing changed
    python3 -m prometheus.jobs.flywheel_weekly --recover      # the OnFailure= unit
    python3 -m prometheus.jobs.flywheel_weekly --report-only  # any time: refresh the reviewer's
                                                              # input from the recorded state

A fixed pipeline (no agent loop) driven by the ``flywheel:`` block of
prometheus.yaml, which the job validates when it loads it: an unknown key, an
unquoted time or an unknown command placeholder refuses the run. It runs in the
weekly window (Saturday 02:00-06:00 America/New_York by default) and is out of
it before it closes: no stage starts in the last ``stop_margin_minutes``, and a
running stage is sent SIGTERM at that point (time to save a checkpoint) and
SIGKILL one minute before the end.

STAGES
------
1. **harvest**  New traces from the configured sources, read only, deduped by
   content against every earlier harvest. Each source file has a watermark in
   the state (the byte offset read to, at a whole line, and a hash of the bytes
   before it): a file that only grew is read from there, so a new golden export
   (a new interactive /qwen turn, say) or a teacher run that appended rows is
   picked up without re-reading the rest; a file rewritten under its name (a
   teacher_gen.py resume trims its tail) is read whole again, and the content
   dedupe keeps what was harvested before. A last line without its newline is
   still being written and waits for the next harvest. Each trace is tagged
   with its source model, adapter version, origin (teacher / offspring /
   unknown) and whether it is Max-derived; the counts are kept per source tag
   (``_meta.source``: the local 27B's ``qwen38-27b-gym``, a Max teacher run's
   own tag, ``golden-export`` for the daemon's exports) and per share class
   (``max`` / ``qwen38-27b-gym`` / ``other``, build.py's classes).
2. **verify**   Decides which harvested traces may be trained on (rules below)
   and puts the trainable extra-source ones in the verified pool.
3. **build**    The dataset builder (scripts/distill/build_datasets.py, or
   build.py) over a copy of the golden sources holding only the traces verify
   passed, plus the verified pool. Sets whose byte-for-byte spot check has not
   passed are never trained on. ``max_derived.exclude_sessions_file``, when
   set, goes to the builder as ``--exclude-sessions``; the Q6b allow (below)
   adds ``--allow-max-derived --max-derived-approval <approval_ref>`` and puts
   the sets in ``sets-max-derived/`` (build.py wants the tag in the name). The
   job owns these flags: ``build.args`` may not set them. Any edit to the
   ids the list names rebuilds the sets, new traces or not.
4. **train**    Only with >= ``train.min_new_traces`` verified traces since the
   last train's data was frozen, and as many train rows the last train did not
   have, on sets built under the session exclusion list as it is now. A train
   the window cuts short resumes from its newest checkpoint next window, on
   the same frozen data and under the approval that data records, unless the
   list changed since it was frozen: it is then superseded (kept under
   ``superseded_trains`` in the state, its data and checkpoints on disk) and a
   new train starts on sets rebuilt under the current list. A train on
   Max-derived rows is named for it (``fw-<run>-max-derived``: its run
   directory, frozen data and models), the trainer gets
   ``--i-have-operator-approval-for-max-derived-data``, and its adapter is
   found under the tagged name the trainer gives it (``adapter-max-derived``,
   scripts/distill/artifact_policy.py); an untagged adapter from Max-derived
   data fails the train.
5. **quantize** The finished adapter, into ``models/<version>`` (tagged, with
   the quantizer's approval flag, when the adapter is Max-derived: its record
   says so, says nothing, or its provenance chain shows it). quantize.py
   decides the same from the adapter's own chain, failing closed, so the
   trainer must leave its run manifest beside the adapter (train_qlora.py
   does); without one it refuses the untagged ``models/<version>``.
6. **score**    The quantized candidate (scripts/distill/score.py), only while
   the scoring server serves it: the Phase 6 gate against the base model's
   summary, and wins and losses against the previous model.

Then the report, always (below).

THE MONDAY REVIEWER'S INPUT
---------------------------
The reviewer (a scheduled task outside this repo, run Monday mornings) reads
one file and nothing else. This job writes it as
``<data_dir>/reports/LATEST.md`` (data_dir defaults to
``~/.prometheus/data/flywheel``) on the host it runs on. The cadence sprint's
reviewer fetches ``flywheel/reports/LATEST.md`` from the public repo, and
nothing here commits, copies or publishes the file there: whether and how it
is published is the operator's decision (docs/sprints/DISTILL-OPEN-QUESTIONS.md
#11). It is Markdown, with ``LATEST.json`` beside it as its machine-readable
twin (schema :data:`REPORT_SCHEMA`, validated against :class:`WeeklyReport`
before it is written). The Markdown is a title, a status
line whose first words are ``FLYWHEEL weekly · OK|WARN|FAIL|OFF``, then exactly
the sections of :data:`REVIEWER_SECTIONS`, in that order, each always present
("None." when empty): at a glance; what ran (every stage, its result, its
time); why stages did not run; traces (per source, and the newest dataset's
Max vs local-27B share per split); the model, with wins and losses vs the
previous model; spend; guards (the Q6b allow, and every Max-derived artifact
the state holds, each private: never published); issues; operator notes;
next steps (who acts: the job, the operator, or the runbook's plan; the last
is always the next window). The JSON holds the same facts.

It is written every window: a skipped, refused or paused stage says why, and a
disabled flywheel, a bad config, an unreadable state, a crashed stage, a report
that could not be built (a minimal one replaces it) or a run the recovery unit
had to finish still writes one. ``--report-only`` rewrites it from the recorded
state at any time (no stage runs, nothing else changes), so the input is
current even in a week no window ran. Each file is replaced atomically (tmp +
fsync + rename): the dated copy ``<run>-weekly.{json,md}`` first, then
``LATEST.json``, then ``LATEST.md``; a reader sees a whole old file or a whole
new one, and the pair carries the same ``run_id`` and ``generated``. Paths under
the home directory are written as ``~``; nothing here publishes the report.

Every stage is resumable. A stage that finished earlier in the same window is
not repeated; harvest and verify commit by atomic rename, so a run cut short
leaves nothing half-recorded; build reruns; train resumes from checkpoints;
score.py resumes per case. ``--dry-run`` reads the sources and the state, writes
``reports/dry-run/LATEST.md`` and changes nothing else.

WHAT IS NEVER TRAINED ON
------------------------
* **Unverified offspring output.** A trace from the student (a served model
  matching ``harvest.student_models``, or any trace that carries an adapter
  version) is trainable only with a ``pass`` verdict from a named verifier
  that is not the student itself, under any of its names (its model, served
  id or adapter, any adapter this job trained, any ``student_models``
  match), and not Max (never a judge of a student). Its tool calls having
  succeeded is not a verification: that is how a model learns its own
  mistakes. A teacher's golden trace counts on its tool-success label (the
  runbook's rule); an extra-source teacher trace needs a pass from a judge that
  is not Max, or, on a Max-derived teacher trace only, from the pay-per-token
  Max judge (the operator's PAYG ruling: ``provider: max-payg``, ``judge_key``
  ``payg`` when given; build.py's rule, which also never lets it verify a local
  27B row). Any other judge naming Max may be the Token Plan key and verifies
  nothing. Anything else is tagged and dropped, golden traces included: the
  builder reads a copy of them in which every trace verify did not pass is
  withheld. The train stage checks the frozen data again before every start
  and every resume.
* **The score cases.** A training row that shares an id, task or group with
  a configured ``score.cases`` file refuses the train (and the scoring of an
  adapter trained on it): the builder re-splits every week, so a task held out
  once can land in a later train split.
* **Max-derived data, without the Q6b allow.** The operator's Q6b ruling
  (docs/sprints/DISTILL-OPEN-QUESTIONS.md #6) lets Max-derived rows train a
  PRIVATE model under an explicit, recorded allow: ``max_derived`` records
  ``approved: true``, an ``approval_ref`` and an ``exclude_sessions_file`` (the
  Token-Plan incident session, never trained on). The train stage refuses a
  dataset holding any such row unless all three are set, the file lists a
  session, and the dataset itself recorded the same approval when build.py
  made it. The offspring of an adapter trained on Max-derived data is
  Max-derived too, and so is the offspring of an adapter this job did not
  train (nothing shows otherwise).
* Everything Max-derived stays private: it is tagged ``max-derived`` in its
  name and manifest, ``publish_guard.py`` refuses it, the rental tool never
  uploads it, and nothing here publishes anything. The report lists what the
  state holds of it.

WIRING (the operator half: nothing here installs it)
----------------------------------------------------
deploy/systemd/ ships the user timer, its service and the OnFailure= recovery
unit. The daemon is never stopped by this job (the distillation runbook's rule;
the cadence sprint's daemon-down window is not used). When ``gpu_setup`` names
a setup, scripts/distill/setups.py arms its dead-man restore, applies the
setup (only with time left in the window for it and the stage) and restores
``business`` when the run ends. ``gpu_setup: none`` runs train and quantize on
the GPUs as they are, which the operator opts into; left empty, they do not
run. Scoring only queries ``score.base_url``, which must already serve the
candidate (the setup, or the operator, loads it there).

Exit codes: 0 the run finished (stages may have been skipped or refused, with
reasons); 1 a stage failed, the GPU restore failed or the report could not be
written; 2 bad config, or a scheduled run outside the window; 3 another run
holds the lock (a report-only refresh leaves that run's report alone).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import errno
import fcntl
import glob
import hashlib
import importlib.util
import json
import logging
import os
import re
import shlex
import shutil
import signal
import string
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Callable, Iterator, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("prometheus.jobs.flywheel_weekly")

REPORT_SCHEMA = "flywheel-weekly-report/3"
STATE_SCHEMA = "flywheel-weekly-state/1"

STAGES = ("harvest", "verify", "build", "train", "quantize", "score")

OK = "OK"
SKIPPED = "SKIPPED"     # nothing to do, or a precondition is not met (the reason says which)
REFUSED = "REFUSED"     # a guard or a gate said no
PAUSED = "PAUSED"       # stopped for the window; picks up next window
FAILED = "FAILED"
DRY_RUN = "DRY-RUN"     # would run (the summary says what)
NOT_RUN = "NOT RUN"     # no run reached it (the reason says what holds it)

# The Monday reviewer's input: LATEST.md holds exactly these "## " sections, in
# this order, every time (module docstring). LATEST.json is the twin.
REVIEWER_SECTIONS = (
    "At a glance",
    "What ran",
    "Why stages did not run",
    "Traces",
    "Model: wins and losses vs the previous model",
    "Spend",
    "Guards",
    "Issues",
    "Operator notes",
    "Next steps",
)
REPORT_MODES = ("scheduled", "dry-run", "recovery", "config-error", "state-error",
                "report-only")
# Modes in which the job never read its state (and, for config-error, holds
# only fallback defaults in place of the config): the report states neither
# as fact.
UNREAD_MODES = ("config-error", "state-error")
# Who a next step waits on: the job itself (next window), the operator, or the
# plan the runbook keeps outside the job (--next-step).
STEP_OWNERS = ("job", "operator", "runbook")
# Operator notes and next steps given on the command line are capped, one line each.
NOTE_LIMIT = 600

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_CONFIG = 2
EXIT_BUSY = 3

# Window time a stage needs before it may start. A stage that cannot start is
# SKIPPED with the minutes that were left, not started and killed.
MIN_STAGE_SECONDS = {
    "harvest": 120, "verify": 120, "build": 900,
    "train": 1800, "quantize": 900, "score": 900,
}
# SIGTERM goes at the stop time (window end minus the margin); SIGKILL this long
# before the window end, so nothing the job started outlives the window.
KILL_BEFORE_END = timedelta(seconds=60)
# A child stopped because the job itself got SIGTERM gets this long before SIGKILL
# (the units' TimeoutStopSec is 3 min).
TERM_GRACE = timedelta(seconds=150)
# Harvest and verify consult the clock every this many lines, not only at the top.
CLOCK_EVERY_LINES = 200
# How often a running child's stop and kill times are checked (seconds).
POLL_SECONDS = 5.0
# Every setups.py call is bounded. The restore is not window-bound (it runs
# after the window if it must); the swap into gpu_setup is (Run.swap_command).
SETUP_COMMAND_LIMIT = timedelta(minutes=30)
# A swap into gpu_setup starts only with this long left for it on top of the
# stage's own minimum (one setups.py apply attempt: stop, start, health wait),
# and it is held to the stop time.
SETUP_APPLY_SECONDS = 600
DEADMAN_AFTER_END = timedelta(minutes=15)
RESTORE_SETUP = "business"
# gpu_setup's opt-in for running the GPU stages on the GPUs as they are.
NO_SWAP_SETUP = "none"

BUILD_SCRIPTS = ("scripts/distill/build_datasets.py", "scripts/distill/build.py")
SCORE_SCRIPT = "scripts/distill/score.py"
SETUPS_SCRIPT = "scripts/distill/setups.py"
# The Max-derived rule (DISTILL-OPEN-QUESTIONS Q6b): names, manifests,
# provenance. One module for every distill script, loaded under the name they
# all use (:func:`load_policy`).
POLICY_SCRIPT = "scripts/distill/artifact_policy.py"
POLICY_MODULE = "distill_artifact_policy"
# build.py's Q6b allow and its session exclusion list: the job sets them from
# flywheel.max_derived (stage_build).
BUILD_ALLOW_FLAG = "--allow-max-derived"
BUILD_APPROVAL_FLAG = "--max-derived-approval"
BUILD_EXCLUDE_FLAG = "--exclude-sessions"
# The trainer's and the quantizer's operator flag for Max-derived data
# (train_qlora.APPROVAL_FLAG, quantize.APPROVAL_FLAG): the job appends it to a
# train on Max-derived rows, and to the quantize of its adapter, only under a
# complete Q6b allow.
MAX_DERIVED_APPROVAL_FLAG = "--i-have-operator-approval-for-max-derived-data"
# The job owns these builder flags; build.args may not set them.
BUILD_OWNED_FLAGS = ("--traces", "--extra-traces", "--out", "--report", "--dry-run",
                     BUILD_ALLOW_FLAG, BUILD_APPROVAL_FLAG, BUILD_EXCLUDE_FLAG)
# The builder's output directory: sets/, or sets-max-derived/ under the allow.
SETS_DIR = "sets"
# The pay-per-token Max judge (build.py's PAYG_PROVIDER / PAYG_JUDGE_KEY; the
# operator's PAYG ruling for Phase 2 judging).
PAYG_PROVIDER = "max-payg"
PAYG_JUDGE_KEY = "payg"
# The teachers whose traces count as teacher traces by default: the hosted Max
# (its golden exports and pay-per-token teacher runs) and the local 27B teacher
# (teacher_gen.py's gym runs).
DEFAULT_TEACHER_MODELS = ("qwen3.8-max", "qwen3.8-27b*")
# build.py's share classes: Max-derived rows, the local 27B teacher's, the rest.
GYM_SOURCE = "qwen38-27b-gym"
SHARE_CLASSES = ("max", GYM_SOURCE, "other")
# A golden export with no source tag of its own is counted under this label.
GOLDEN_EXPORT_SOURCE = "golden-export"
# The daemon's incremental exporter names its files golden_traces_<time>_<rowid>.jsonl.
_EXPORT_RE = re.compile(r"^golden_traces_\d+_(\d+)\.jsonl$")

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
TRAIN_FIELDS = frozenset({
    "python", "repo", "train", "val", "data_dir", "run_dir",
    "adapter_version", "stop_at", "stop_at_epoch", "max_seconds",
})
RESUME_FIELDS = TRAIN_FIELDS | {"checkpoint"}
QUANTIZE_FIELDS = frozenset({
    "python", "repo", "adapter_dir", "data_dir", "out_dir", "adapter_version",
    "stop_at", "stop_at_epoch", "max_seconds",
})
# The trainer's run directory ({run_dir}) holds checkpoint-<step>/ directories,
# the finished adapter in adapter/, and optionally manifest.json whose "status"
# ("done" / "stopped") says how an exit 0 ended (scripts/distill/train_qlora.py
# exits 0 for both).
ADAPTER_SUBDIR = "adapter"
TRAINER_MANIFEST = "manifest.json"
# The split files a train freezes (train and val are required; the rest let the
# quantizer calibrate on exactly the data the adapter was trained on).
FROZEN_FILES = ("train.jsonl", "val.jsonl", "val-cases.jsonl", "eval-tools.jsonl")

# The only label an offspring row may train under. "tool-success" is a
# teacher-only basis: the teacher is the reference, the offspring is not.
OFFSPRING_TRAINABLE_LABELS = frozenset({"judge-pass"})
# Verify's reasons for dropping an offspring trace (the report counts them).
OFFSPRING_DROPS = ("unverified_offspring", "self_verified_offspring", "max_judged_offspring")
# A golden trace verify did not pass keeps its line in the builder's copy under
# this model id, which no --teacher-model is: the builder drops it first thing
# (non_teacher_model), and every other trace keeps its id and index position.
WITHHELD_MODEL = "flywheel-withheld"
# The keys a training row and a score case share when they are the same task:
# build.py's row id (from the trace id), task id, and group id (the group's
# smallest task id).
HOLDOUT_KEYS = ("id", "task_id", "group_id")
# A score delta under this many points is "unchanged" in the report.
WIN_LOSS_PTS = 1.0

_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _never() -> bool:
    return False


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def is_max_name(name: str) -> bool:
    """True when *name* names the hosted teacher (Qwen3.8 Max) as a token.

    ``qwen3.8-max``, ``max-golden``, ``max-gym-2026-11-01`` and
    ``flywheel/teacher/qwen3.8-max@none`` all hold the token ``max``;
    ``maxwell`` does not. Over-matching only ever refuses a train, which is the
    safe direction for this guard.
    """
    return "max" in _TOKEN_SPLIT.split(name.lower())


def _redact(text: str) -> str:
    from prometheus.security.log_redaction import redact_secrets

    return redact_secrets(text)


def load_policy(repo: Path | None = None) -> Any:
    """scripts/distill/artifact_policy.py, the Max-derived rule (Q6b): from the
    run's checkout, else this package's own. Loaded once, under the module name
    every distill script loads it by.

    Raises FileNotFoundError when neither checkout holds it: nothing is then
    named, trained or quantized as if it were clean (the callers refuse).
    """
    loaded = sys.modules.get(POLICY_MODULE)
    if loaded is not None:
        return loaded
    roots = [r for r in (repo, _repo_root()) if r is not None]
    for root in roots:
        path = root / POLICY_SCRIPT
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(POLICY_MODULE, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        sys.modules[POLICY_MODULE] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(POLICY_MODULE, None)
            raise
        return mod
    raise FileNotFoundError(
        f"{POLICY_SCRIPT} is in neither {' nor '.join(str(r) for r in roots)}: the Max-derived "
        "rule (DISTILL-OPEN-QUESTIONS Q6b) cannot be read")


def _policy_or_none(repo: Path | None) -> Any:
    try:
        return load_policy(repo)
    except (OSError, ImportError, SyntaxError) as exc:
        logger.warning("artifact_policy.py unavailable: %s", exc)
        return None


def _session_ids(path: Path) -> set[str]:
    """The session ids *path* lists, as build.py's ``read_session_list`` reads
    them (one per line, ``#`` comments and blank lines skipped). Raises
    OSError / ValueError / UnicodeDecodeError when the file cannot be read or
    lists none."""
    ids = {line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()}
    ids.discard("")
    if not ids:
        raise ValueError("lists no session id")
    return ids


def read_session_list(path: Path) -> int:
    """How many session ids *path* lists. The ids themselves are never read
    into the report. Raises as :func:`_session_ids` does."""
    return len(_session_ids(path))


def session_list_digest(path: Path | None) -> str | None:
    """The session exclusion list as the builder applies it: a sha256 of the
    sorted ids *path* lists; "" when no list is set, None when it is unusable
    (:func:`session_list_problem` says why). The sets and a frozen train
    record it, so an edit to the list is seen: the build runs again, and data
    built under the old list neither starts a train nor resumes one."""
    if path is None:
        return ""
    try:
        ids = _session_ids(path)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def session_list_problem(path: Path | None) -> str:
    """Why the session exclusion list is unusable ("" when it is usable)."""
    if path is None:
        return "flywheel.max_derived.exclude_sessions_file is not set"
    try:
        read_session_list(path)
    except FileNotFoundError:
        return f"the session exclusion list {path} does not exist"
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return f"the session exclusion list {path} is unusable ({type(exc).__name__}: {exc})"
    return ""


# ---------------------------------------------------------------------------
# Config: the flywheel: block, validated at load
# ---------------------------------------------------------------------------


class FlywheelConfigError(ValueError):
    """The flywheel: block is invalid. Every problem is listed; the job refuses to run."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


class FlywheelStateError(RuntimeError):
    """state.json cannot be read. Never reset silently: it holds the train record."""


@dataclass(frozen=True)
class Window:
    day: int                  # 0 = Monday
    start: dtime
    end: dtime
    tz: str
    stop_margin: timedelta

    @property
    def length(self) -> timedelta:
        return (datetime.combine(date.min, self.end)
                - datetime.combine(date.min, self.start))

    def describe(self) -> str:
        return (f"{DAYS[self.day].capitalize()} {self.start:%H:%M}-{self.end:%H:%M} {self.tz}")


@dataclass(frozen=True)
class MaxDerivedAllow:
    """``flywheel.max_derived``: the operator's Q6b allow, passed to build.py.

    ``complete`` (approved, with a reference and a session exclusion list) is
    the only state in which a Max-derived row trains. The list may be set
    without the allow: the builder then drops those sessions from every split.
    """

    approved: bool = False
    approval_ref: str = ""
    exclude_sessions_file: Path | None = None

    @property
    def complete(self) -> bool:
        return (self.approved and bool(self.approval_ref.strip())
                and self.exclude_sessions_file is not None)


@dataclass(frozen=True)
class FlywheelConfig:
    enabled: bool
    data_dir: Path
    window: Window
    golden: tuple[str, ...]
    extra: tuple[str, ...]
    teacher_models: tuple[str, ...]
    student_models: tuple[str, ...]
    build_script: str
    build_args: tuple[str, ...]
    min_new_traces: int
    train_command: tuple[str, ...]
    resume_args: tuple[str, ...]
    incomplete_exit_code: int
    max_failed_attempts: int
    max_derived: MaxDerivedAllow
    quantize_command: tuple[str, ...]
    score_base_url: str
    score_cases: tuple[str, ...]
    baseline_summary: str
    gpu_setup: str
    spend_ledger: Path


class _Section:
    """One mapping of the block. Tracks the keys it read; anything else is an error."""

    def __init__(self, raw: Any, path: str, problems: list[str]) -> None:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            problems.append(f"{path} must be a mapping, not {type(raw).__name__}")
            raw = {}
        self.raw: dict[Any, Any] = raw
        self.path = path
        self.problems = problems
        self.read: set[str] = set()

    def _get(self, key: str, default: Any) -> Any:
        self.read.add(key)
        value = self.raw.get(key, default)
        return default if value is None else value

    def section(self, key: str) -> _Section:
        return _Section(self._get(key, {}), f"{self.path}.{key}", self.problems)

    def flag(self, key: str, default: bool) -> bool:
        value = self._get(key, default)
        if not isinstance(value, bool):
            self.problems.append(f"{self.path}.{key} must be true or false, not {value!r}")
            return default
        return value

    def integer(self, key: str, default: int, lo: int, hi: int) -> int:
        value = self._get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            self.problems.append(f"{self.path}.{key} must be an integer from {lo} to {hi}, "
                                 f"not {value!r}")
            return default
        return value

    def text(self, key: str, default: str) -> str:
        value = self._get(key, default)
        if not isinstance(value, str):
            self.problems.append(f"{self.path}.{key} must be a string, not {value!r}")
            return default
        return value.strip()

    def texts(self, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        value = self._get(key, list(default))
        if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
            self.problems.append(f"{self.path}.{key} must be a list of non-empty strings, "
                                 f"not {value!r}")
            return default
        return tuple(v.strip() for v in value)

    def clock_time(self, key: str, default: str) -> dtime:
        value = self._get(key, default)
        if isinstance(value, int) and not isinstance(value, bool):
            # YAML 1.1 reads an unquoted 02:00 as the base-60 integer 120.
            self.problems.append(f"{self.path}.{key} must be a quoted \"HH:MM\" string "
                                 f"(YAML read the unquoted time as the number {value})")
            return dtime.fromisoformat(default)
        try:
            if not isinstance(value, str) or not re.fullmatch(r"\d\d:\d\d", value):
                raise ValueError(value)
            return dtime.fromisoformat(value)
        except ValueError:
            self.problems.append(f"{self.path}.{key} must be \"HH:MM\", not {value!r}")
            return dtime.fromisoformat(default)

    def done(self) -> None:
        for key in sorted(str(k) for k in self.raw if str(k) not in self.read):
            self.problems.append(f"{self.path}.{key} is not a flywheel setting")


def _resolve_under(raw: str, base: Path) -> Path:
    path = Path(os.path.expanduser(raw))
    return path if path.is_absolute() else base / path


def _data_root() -> Path:
    from prometheus.config.paths import data_dir_path

    return data_dir_path()


def _check_command(argv: tuple[str, ...], fields: frozenset[str], where: str,
                   problems: list[str]) -> None:
    for arg in argv:
        try:
            names = {name for _, name, _, _ in string.Formatter().parse(arg) if name is not None}
        except ValueError as exc:
            problems.append(f"{where}: {arg!r} is not a valid template ({exc})")
            continue
        unknown = sorted(n for n in names if n not in fields)
        if unknown:
            problems.append(f"{where}: unknown placeholder(s) {unknown} in {arg!r}; "
                            f"known: {sorted(fields)}")


def parse_config(raw: Any) -> FlywheelConfig:
    """The ``flywheel:`` block as a :class:`FlywheelConfig`, or :class:`FlywheelConfigError`.

    ``None`` (no block at all) is the documented defaults, which is a disabled
    flywheel. Every problem is collected before raising, so one run names them
    all.
    """
    problems: list[str] = []
    top = _Section(raw, "flywheel", problems)

    enabled = top.flag("enabled", False)
    data_dir = _resolve_under(top.text("data_dir", "flywheel") or "flywheel", _data_root())

    w = top.section("window")
    day = w.text("day", "sat").lower()
    if day not in DAYS:
        problems.append(f"flywheel.window.day must be one of {list(DAYS)}, not {day!r}")
        day = "sat"
    start = w.clock_time("start", "02:00")
    end = w.clock_time("end", "06:00")
    tz = w.text("timezone", "America/New_York")
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        problems.append(f"flywheel.window.timezone {tz!r} is not a known time zone")
        tz = "America/New_York"
    margin = w.integer("stop_margin_minutes", 15, 5, 120)
    w.done()
    window = Window(DAYS.index(day), start, end, tz, timedelta(minutes=margin))
    if end <= start:
        problems.append("flywheel.window.end must be after window.start on the same day "
                        "(a window may not cross midnight)")
    elif window.length <= window.stop_margin + timedelta(minutes=30):
        problems.append("flywheel.window is too short: it must be at least 30 minutes longer "
                        "than stop_margin_minutes")

    h = top.section("harvest")
    golden = h.texts("golden", ("~/.prometheus/trajectories/golden_traces_*.jsonl",))
    extra = h.texts("extra", ())
    teacher_models = h.texts("teacher_models", DEFAULT_TEACHER_MODELS)
    student_models = h.texts("student_models", ())
    h.done()

    b = top.section("build")
    build_script = b.text("script", "")
    build_args = b.texts("args", ())
    b.done()
    for arg in build_args:
        flag = arg.split("=", 1)[0]
        if flag in BUILD_OWNED_FLAGS:
            problems.append(f"flywheel.build.args may not set {flag}: the job sets it "
                            "(golden sources come from harvest.golden)")

    t = top.section("train")
    min_new = t.integer("min_new_traces", 500, 1, 10_000_000)
    train_command = t.texts("command", ())
    resume_args = t.texts("resume_args", ())
    incomplete_code = t.integer("incomplete_exit_code", 75, 1, 255)
    max_failed = t.integer("max_failed_attempts", 2, 1, 20)
    t.done()
    _check_command(train_command, TRAIN_FIELDS, "flywheel.train.command", problems)
    _check_command(resume_args, RESUME_FIELDS, "flywheel.train.resume_args", problems)
    if resume_args and not train_command:
        problems.append("flywheel.train.resume_args is set but train.command is empty")

    if isinstance(top.raw, dict) and "max_data" in top.raw:
        top.read.add("max_data")
        problems.append("flywheel.max_data was replaced by flywheel.max_derived (approved, "
                        "approval_ref, exclude_sessions_file: the operator's Q6b allow, "
                        "DISTILL-OPEN-QUESTIONS #6)")
    m = top.section("max_derived")
    approved = m.flag("approved", False)
    approval_ref = m.text("approval_ref", "")
    exclude_raw = m.text("exclude_sessions_file", "")
    m.done()
    approval = MaxDerivedAllow(
        approved=approved, approval_ref=approval_ref,
        exclude_sessions_file=_resolve_under(exclude_raw, _data_root()) if exclude_raw else None)
    if approved and not approval_ref:
        problems.append("flywheel.max_derived.approved is true but approval_ref is empty: the "
                        "allow records the operator's approval verbatim (e.g. \"operator Q6b "
                        "2026-10-30\"), and build.py records it in the dataset")
    if approved and not exclude_raw:
        problems.append("flywheel.max_derived.approved is true but exclude_sessions_file is "
                        "empty: build.py's allow needs the session exclusion list (the "
                        "2026-10-13 Token-Plan incident session is never trained on)")

    q = top.section("quantize")
    quantize_command = q.texts("command", ())
    q.done()
    _check_command(quantize_command, QUANTIZE_FIELDS, "flywheel.quantize.command", problems)

    s = top.section("score")
    base_url = s.text("base_url", "")
    cases = s.texts("cases", ())
    baseline = s.text("baseline_summary", "")
    s.done()
    if base_url and not re.match(r"^https?://", base_url):
        problems.append(f"flywheel.score.base_url must start with http:// or https:// "
                        f"({base_url!r})")

    gpu_setup = top.text("gpu_setup", "")
    if gpu_setup and (not _NAME_RE.match(gpu_setup) or gpu_setup == RESTORE_SETUP):
        problems.append(f"flywheel.gpu_setup must be a setup name other than "
                        f"{RESTORE_SETUP!r} ({gpu_setup!r})")
    ledger = _resolve_under(top.text("spend_ledger", "~/distill/spend-ledger.csv")
                            or "~/distill/spend-ledger.csv", _data_root())
    top.done()

    if problems:
        raise FlywheelConfigError(problems)
    return FlywheelConfig(
        enabled=enabled, data_dir=data_dir, window=window, golden=golden, extra=extra,
        teacher_models=teacher_models, student_models=student_models,
        build_script=build_script, build_args=build_args, min_new_traces=min_new,
        train_command=train_command, resume_args=resume_args,
        incomplete_exit_code=incomplete_code, max_failed_attempts=max_failed,
        max_derived=approval, quantize_command=quantize_command,
        score_base_url=base_url, score_cases=cases, baseline_summary=baseline,
        gpu_setup=gpu_setup, spend_ledger=ledger,
    )


def window_bounds(now: datetime, window: Window) -> tuple[datetime, datetime] | None:
    """(start, end) of the window *now* falls in, as aware local times, else None."""
    local = now.astimezone(ZoneInfo(window.tz))
    if local.weekday() != window.day:
        return None
    start = local.replace(hour=window.start.hour, minute=window.start.minute,
                          second=0, microsecond=0)
    end = local.replace(hour=window.end.hour, minute=window.end.minute,
                        second=0, microsecond=0)
    return (start, end) if start <= local < end else None


def window_on(day: date, window: Window) -> tuple[datetime, datetime]:
    """The window's (start, end) on *day*, as aware local times.

    The window takes its own date's UTC offset, so a window on the first
    Saturday after a daylight-saving change starts at 02:00 local, not at the
    previous week's 02:00 shifted by an hour.
    """
    tz = ZoneInfo(window.tz)
    return (datetime.combine(day, window.start, tzinfo=tz),
            datetime.combine(day, window.end, tzinfo=tz))


def next_window(now: datetime, window: Window) -> tuple[datetime, datetime]:
    """The window *now* is in, else the next one to start."""
    local = now.astimezone(ZoneInfo(window.tz))
    for ahead in range(8):
        day = local.date() + timedelta(days=ahead)
        if day.weekday() == window.day:
            start, end = window_on(day, window)
            if end > local:
                return start, end
    raise AssertionError("a weekly window always falls within eight days")


def last_window_start(now: datetime, window: Window) -> datetime:
    """The start of the newest window that has begun at or before *now*."""
    local = now.astimezone(ZoneInfo(window.tz))
    for back in range(8):
        day = local.date() - timedelta(days=back)
        if day.weekday() == window.day:
            start, _ = window_on(day, window)
            if start <= local:
                return start
    raise AssertionError("a weekly window always falls within eight days")


def describe_window(start: datetime, end: datetime) -> str:
    """``Sat 2026-11-07 02:00-06:00 EST (07:00-11:00 UTC)``."""
    utc_s, utc_e = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    return (f"{start:%a %Y-%m-%d %H:%M}-{end:%H:%M %Z} "
            f"({utc_s:%H:%M}-{utc_e:%H:%M} UTC)")


# ---------------------------------------------------------------------------
# Files: atomic writes, JSON, the state
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: str | bytes) -> None:
    """tmp + fsync + rename: a reader sees the old file or the new one, never half.

    The directory is synced after the rename (best effort), so a power cut
    right after the job exits does not take the new name back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data.encode("utf-8") if isinstance(data, str) else data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _read_json(path: Path) -> Any:
    """Parsed JSON, or None when the file does not exist. Anything else raises."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return json.loads(text)


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def load_state(path: Path) -> dict[str, Any]:
    try:
        state = _read_json(path)
    except (OSError, ValueError) as exc:
        raise FlywheelStateError(f"{path} is unreadable ({type(exc).__name__}: {exc}); "
                                 "fix or move it by hand: it holds the train record") from exc
    if state is None:
        return {"schema": STATE_SCHEMA}
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise FlywheelStateError(f"{path} is not a {STATE_SCHEMA} file")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(state, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Commands: a child process held to the window
# ---------------------------------------------------------------------------


@dataclass
class CommandSpec:
    argv: list[str]
    cwd: Path
    log_path: Path
    stop_at: datetime | None = None     # SIGTERM here: the child checkpoints and exits
    kill_at: datetime | None = None     # SIGKILL here, whatever it is doing
    should_stop: Callable[[], bool] | None = None
    poll_s: float = 5.0
    # The clock the stop and kill times are read against (the run's own clock;
    # None = the wall clock).
    clock: Callable[[], datetime] | None = None


@dataclass
class CommandResult:
    returncode: int
    stopped: bool = False    # this job sent the stop signal (window or SIGTERM)
    seconds: float = 0.0
    tail: str = ""


Runner = Callable[[CommandSpec], CommandResult]
Clock = Callable[[], datetime]


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.send_signal(sig)


def _tail(path: Path, lines: int = 3, limit: int = 300) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4096))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"(log unreadable: {exc})"
    kept = [ln for ln in text.splitlines() if ln.strip()][-lines:]
    return _redact(" | ".join(kept))[-limit:]


def run_command(spec: CommandSpec) -> CommandResult:
    """Run *spec*, its output appended to its log, held to its stop and kill times.

    The child gets its own session, so SIGTERM and SIGKILL reach its whole
    process group (a trainer's dataloader workers included). A child sent
    SIGTERM by this job is reported ``stopped`` whatever its exit code: the
    caller treats it as paused, not failed.
    """
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    clock = spec.clock or _utcnow
    t0 = time.monotonic()
    stopped = False
    term_at: datetime | None = None
    kill_at = spec.kill_at
    with open(spec.log_path, "ab") as log:
        log.write(f"\n=== {clock().isoformat(timespec='seconds')} "
                  f"{shlex.join(spec.argv)}\n".encode())
        log.flush()
        proc = subprocess.Popen(spec.argv, cwd=spec.cwd, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        while True:
            try:
                rc = proc.wait(timeout=spec.poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            now = clock()
            if term_at is None:
                window = spec.stop_at is not None and now >= spec.stop_at
                asked = spec.should_stop is not None and spec.should_stop()
                if window or asked:
                    logger.warning("stopping %s (%s): SIGTERM", spec.argv[0],
                                   "window" if window else "SIGTERM to the job")
                    _signal_group(proc, signal.SIGTERM)
                    term_at, stopped = now, True
                    if asked and not window:
                        # systemd kills the whole unit TimeoutStopSec after its own
                        # SIGTERM; give the child most of that, not the window.
                        kill_at = min(t for t in (spec.kill_at, now + TERM_GRACE) if t)
            if term_at is not None and kill_at is None:
                kill_at = term_at + TERM_GRACE
            if kill_at is not None and now >= kill_at:
                logger.error("%s still running at its kill time: SIGKILL", spec.argv[0])
                _signal_group(proc, signal.SIGKILL)
                stopped = True
                rc = proc.wait()
                break
    return CommandResult(rc, stopped, round(time.monotonic() - t0, 1), _tail(spec.log_path))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    status: str
    seconds: float = 0.0
    summary: str = ""
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Run:
    cfg: FlywheelConfig
    run_id: str
    window_start: datetime
    window_end: datetime
    dry_run: bool
    clock: Clock
    runner: Runner
    repo: Path
    state: dict[str, Any]
    interrupted: Callable[[], bool] = _never
    mode: str = "scheduled"
    started: datetime = field(default_factory=_utcnow)
    results: dict[str, StageResult] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    gpu: str = ""            # "" not needed yet | "ready" | "unavailable"
    gpu_note: str = ""
    restore_failed: bool = False
    crashed: bool = False
    # For the report only: --note lines, --next-step lines, the run a
    # report-only refresh re-renders ("" = none recorded), and the start of
    # the spend week (default: a week before the window start).
    notes: list[str] = field(default_factory=list)
    extra_steps: list[str] = field(default_factory=list)
    reported_run: str = ""
    spend_since: datetime | None = None

    @property
    def data(self) -> Path:
        return self.cfg.data_dir

    @property
    def run_dir(self) -> Path:
        return self.data / "runs" / self.run_id

    @property
    def state_path(self) -> Path:
        return self.data / "state.json"

    @property
    def stop_at(self) -> datetime:
        return self.window_end - self.cfg.window.stop_margin

    @property
    def kill_at(self) -> datetime:
        return self.window_end - KILL_BEFORE_END

    def time_left(self) -> float:
        return (self.stop_at - self.clock()).total_seconds()

    def out_of_time(self) -> bool:
        return self.interrupted() or (not self.dry_run and self.time_left() <= 0)

    def save(self) -> None:
        if not self.dry_run:
            save_state(self.state_path, self.state)

    def issue(self, text: str) -> None:
        if text not in self.issues:
            self.issues.append(text)

    def command(self, argv: list[str], log_name: str, *, windowed: bool = True) -> CommandSpec:
        log = self.data / "logs" / f"{self.run_id}-{log_name}.log"
        if windowed:
            return CommandSpec(argv, self.repo, log, stop_at=self.stop_at, kill_at=self.kill_at,
                               should_stop=self.interrupted, poll_s=POLL_SECONDS,
                               clock=self.clock)
        return CommandSpec(argv, self.repo, log, kill_at=self.clock() + SETUP_COMMAND_LIMIT,
                           poll_s=POLL_SECONDS, clock=self.clock)

    def swap_command(self, argv: list[str]) -> CommandSpec:
        """A step of the swap into gpu_setup: bounded, and held to the window too.

        The restore is not (it runs after the window if it must); a swap that
        has not finished by the stop time is stopped, and the restore follows.
        """
        log = self.data / "logs" / f"{self.run_id}-gpu-setup.log"
        return CommandSpec(argv, self.repo, log, stop_at=self.stop_at,
                           kill_at=min(self.clock() + SETUP_COMMAND_LIMIT, self.kill_at),
                           should_stop=self.interrupted, poll_s=POLL_SECONDS, clock=self.clock)

    def fill(self, template: tuple[str, ...], values: dict[str, str]) -> list[str]:
        stop = self.stop_at.astimezone(timezone.utc)
        base = {
            "python": sys.executable, "repo": str(self.repo),
            "stop_at": stop.isoformat(timespec="seconds"),
            "stop_at_epoch": str(int(stop.timestamp())),
            "max_seconds": str(max(0, int(self.time_left()))),
        }
        return [arg.format(**base, **values) for arg in template]


def _quote(argv: list[str]) -> str:
    return _redact(shlex.join(argv))[:400]


# ---------------------------------------------------------------------------
# Harvest: new traces, tagged
# ---------------------------------------------------------------------------


def trace_hash(trace: dict[str, Any]) -> str:
    """Content identity: messages + tools, not ``_meta`` (a re-export is the same trace)."""
    body = {"messages": trace.get("messages"), "tools": trace.get("tools")}
    blob = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def shape_problem(trace: Any) -> str:
    """Why a line is not a usable trace (empty = it is)."""
    if not isinstance(trace, dict):
        return "not an object"
    msgs = trace.get("messages")
    if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
        return "no messages"
    last = msgs[-1]
    if last.get("role") != "assistant" or not (last.get("tool_calls") or last.get("content")):
        return "does not end with an assistant turn"
    if not isinstance(trace.get("tools", []), list):
        return "tools is not a list"
    if not isinstance(trace.get("_meta", {}), dict):
        return "_meta is not an object"
    return ""


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(fnmatchcase(low, p.lower()) for p in patterns)


def adapter_named(name: str, versions: Any) -> str:
    """The adapter version *name* names, of *versions*; "" for none.

    The longest version that stands in *name* as a token (any case) wins, and
    a version followed by ``-<n>`` is not read: :func:`_freeze` names a second
    train of the same run ``fw-<run>-2``, so a served id of ``fw-2026-10-31-2``
    is not ``fw-2026-10-31``.
    """
    for v in sorted({str(v) for v in versions if v}, key=lambda v: (-len(v), v)):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])(?!-\d+(?![A-Za-z0-9]))",
                     name, re.IGNORECASE):
            return v
    return ""


def tag_trace(meta: dict[str, Any], cfg: FlywheelConfig,
              adapters: dict[str, Any]) -> dict[str, Any]:
    """Source model, adapter version, origin, Max-derivation and verdict of one trace.

    Offspring is decided first, so a student whose served id happens to match
    a teacher pattern is still offspring: an adapter version (other than
    ``none``) on the trace, a served id that names an adapter this job trained,
    or a ``student_models`` match all make it one.
    """
    model = str(meta.get("source_model") or meta.get("served_model")
                or meta.get("model") or "unknown")
    declared = str(meta.get("adapter_version") or "").strip()
    named = adapter_named(model, adapters)
    if not named and any(v and v.lower() in model.lower() for v in adapters):
        named = "unknown"     # one of ours, but not which: offspring, Max-derived
    adapter = declared or named
    if (adapter and adapter != "none") or bool(meta.get("offspring")) \
            or _matches(model, cfg.student_models):
        origin = "offspring"
        adapter = adapter if adapter and adapter != "none" else "unknown"
    elif _matches(model, cfg.teacher_models):
        origin, adapter = "teacher", adapter or "none"
    else:
        origin, adapter = "unknown", adapter or "unknown"
    record = adapters.get(adapter)
    max_derived = (is_max_name(model) or bool(meta.get("max_derived"))
                   or (origin == "offspring"
                       and (not isinstance(record, dict) or record.get("max_derived", True) is not False)))
    verdict = str(meta.get("verdict") or "").strip().lower()
    rec_verdict, rec_judge = record_verdict(meta.get("verification"))
    if rec_verdict == "fail" or str(meta.get("label") or "").strip().lower() == "fail":
        verdict = "fail"
    elif not verdict and rec_verdict == "pass":
        verdict = "pass"
    judge = meta.get("judge")
    # Every id the trace's own model goes by: a verifier naming any of them is
    # the model judging itself (verification).
    ids = sorted({str(meta.get(k)).strip() for k in ("source_model", "served_model", "model")
                  if meta.get(k) and str(meta.get(k)).strip()})
    by = str(meta.get("verified_by") or (judge if isinstance(judge, str) else "")
             or rec_judge).strip()
    rec = meta.get("verification")
    # The pay-per-token Max judge, when it is the very judge named above.
    payg = bool(by) and any(
        is_payg_judge(j, _dict(j).get("model")) and str(_dict(j).get("model")).strip() == by
        for j in (_dict(rec).get("judge"), judge))
    return {
        "source_model": model, "adapter_version": adapter, "origin": origin,
        "max_derived": bool(max_derived),
        "verdict": verdict if verdict in ("pass", "fail") else "",
        "verified_by": by,
        "payg_judge": payg,
        "model_ids": ids,
    }


def is_payg_judge(judge: Any, model: Any) -> bool:
    """*judge* (a record's ``judge`` object) is the pay-per-token Max judge:
    ``provider`` is ``max-payg``, a ``judge_key`` (when given) is ``payg``, and
    its model names Max. build.py's ``is_payg_judge``; anything else naming
    Max is not (it may be the Token Plan key, which never judges)."""
    return (isinstance(judge, dict) and judge.get("provider") == PAYG_PROVIDER
            and judge.get("judge_key", PAYG_JUDGE_KEY) == PAYG_JUDGE_KEY
            and isinstance(model, str) and is_max_name(model))


def record_verdict(rec: Any) -> tuple[str, str]:
    """("pass" | "fail" | "", the judge's model) of a ``_meta.verification`` record.

    The record teacher_gen.py writes (and build.py reads strictly): any
    failing part makes it "fail" (``verified`` false, a fail status or
    verdict, a judge verdict other than pass, tool calls that failed); a
    pass needs ``verified`` true (or a pass status) and, when a judge is
    named, its pass. The full strict check is the builder's; this only lets
    a failed record veto, and a passed, judged one name its judge.
    """
    if not isinstance(rec, dict):
        return "", ""
    judge = rec.get("judge")
    j_verdict = judge.get("verdict") if isinstance(judge, dict) else judge
    j_model = judge.get("model") if isinstance(judge, dict) else rec.get("judge_model")
    word = str(rec.get("status") or rec.get("verdict") or "").strip().lower()
    decision = rec.get("verified")
    failed = (decision is False or word == "fail" or rec.get("tool_success") is False
              or (j_verdict is not None and str(j_verdict).strip().lower() != "pass")
              or (isinstance(judge, dict) and "pass" in judge and judge["pass"] is not True))
    if failed:
        return "fail", ""
    passed = decision is True or word == "pass"
    return ("pass" if passed else ""), (str(j_model).strip() if passed and j_model else "")


def _expand(patterns: tuple[str, ...]) -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in patterns:
        for hit in sorted(glob.glob(os.path.expanduser(pat))):
            if os.path.isfile(hit):
                seen.setdefault(os.path.realpath(hit), Path(hit))
    return list(seen.values())


def harvested_hashes(data: Path) -> set[str]:
    """Every trace any earlier run harvested. The run files are the record (no
    separate seen-list that could disagree with them)."""
    out: set[str] = set()
    for path in sorted((data / "runs").glob("*/harvest.jsonl")):
        for row in _read_jsonl(path):
            if isinstance(row.get("h"), str):
                out.add(row["h"])
    return out


def _prefix_hash(path: Path, size: int) -> Any:
    """A sha256 object over the first *size* bytes of *path* (more can be fed
    to it), or None when the file is shorter or unreadable."""
    digest = hashlib.sha256()
    left = size
    try:
        with open(path, "rb") as fh:
            while left > 0:
                block = fh.read(min(left, 1 << 20))
                if not block:
                    return None
                digest.update(block)
                left -= len(block)
    except OSError:
        return None
    return digest


def harvest_marks(state: dict[str, Any]) -> dict[str, Any]:
    """The harvest's own watermarks: per source file (by real path), the byte
    offset read to (a whole line), the lines before it and the sha256 of the
    bytes before it."""
    return _dict(_dict(state.get("harvest")).get("files"))


def share_class(max_derived: bool, source: str) -> str:
    """build.py's share class of a trace: Max-derived, the local 27B teacher's, or other."""
    return "max" if max_derived else GYM_SOURCE if source == GYM_SOURCE else "other"


def source_label(kind: str, meta: dict[str, Any]) -> str:
    """The source a harvested trace is counted under: its own ``_meta.source``
    tag, else what it is (a golden export, an untagged extra-source trace)."""
    tag = meta.get("source")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    return GOLDEN_EXPORT_SOURCE if kind == "golden" else "untagged"


def stage_harvest(run: Run) -> StageResult:
    cfg = run.cfg
    seen = harvested_hashes(run.data)
    raw_adapters = run.state.get("adapters")
    adapters: dict[str, Any] = raw_adapters if isinstance(raw_adapters, dict) else {}
    marks = harvest_marks(run.state)
    new_marks: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    files = looked = 0
    export_rowid = 0
    paused = StageResult("harvest", PAUSED, reason=(
        "the window closed mid-harvest; nothing was recorded, and the next window harvests "
        "these traces again"))
    for kind, patterns in (("golden", cfg.golden), ("extra", cfg.extra)):
        for path in _expand(patterns):
            files += 1
            key = os.path.realpath(path)
            m = _EXPORT_RE.match(path.name) if kind == "golden" else None
            if m:
                export_rowid = max(export_rowid, int(m.group(1)))
            mark = _dict(marks.get(key))
            offset, lineno = 0, 0
            digest: Any = None
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            at = mark.get("offset")
            if isinstance(at, int) and not isinstance(at, bool) and at > 0 \
                    and mark.get("kind") == kind:
                digest = _prefix_hash(path, at) if size >= at else None
                if digest is not None and digest.hexdigest() == mark.get("sha256"):
                    offset, lineno = at, int(_num(mark.get("lines")))
                    file_counts["unchanged" if size == at else "grown"] += 1
                else:
                    digest = None
                    file_counts["rewritten"] += 1     # read whole again; the dedupe holds
            else:
                file_counts["new"] += 1
            if digest is None:
                digest = hashlib.sha256()
            with open(path, "rb") as fh:
                fh.seek(offset)
                for raw in fh:
                    looked += 1
                    if looked % CLOCK_EVERY_LINES == 0 and run.out_of_time():
                        return paused
                    whole = raw.endswith(b"\n")
                    if whole:
                        # Only whole lines move the watermark: a last line still
                        # being written is read again next time.
                        lineno += 1
                        offset += len(raw)
                        digest.update(raw)
                    line = raw.decode("utf-8", errors="replace")
                    if not line.strip():
                        continue
                    try:
                        trace = json.loads(line)
                    except ValueError:
                        counts["unreadable" if whole else "partial_last_line"] += 1
                        continue
                    if shape_problem(trace):
                        counts["unreadable" if whole else "partial_last_line"] += 1
                        continue
                    h = trace_hash(trace)
                    if h in seen:
                        counts["already_harvested"] += 1
                        continue
                    seen.add(h)
                    meta = trace.get("_meta") if isinstance(trace.get("_meta"), dict) else {}
                    tag = tag_trace(meta, cfg, adapters)
                    source = source_label(kind, meta)
                    rows.append({"h": h, "kind": kind, "file": path.name,
                                 "line": lineno + (0 if whole else 1), **tag, "source": source,
                                 "class": share_class(tag["max_derived"], source)})
                    if kind == "extra":
                        contents.append({"h": h, "trace": trace})
            new_marks[key] = {"kind": kind, "offset": offset, "lines": lineno,
                              "sha256": digest.hexdigest()}
    by_origin = Counter(r["origin"] for r in rows)
    table = Counter((r["origin"], r["source_model"], r["adapter_version"]) for r in rows)
    sources = Counter((r["kind"], r["source"], r["class"]) for r in rows)
    by_class = Counter(r["class"] for r in rows)
    before = int(_num(_dict(run.state.get("harvest")).get("golden_export_rowid")))
    details = {
        "new": len(rows), "files": files, "by_origin": dict(by_origin),
        "already_harvested": counts["already_harvested"], "unreadable": counts["unreadable"],
        "by_source": [{"origin": o, "source_model": m, "adapter_version": a, "new": n}
                      for (o, m, a), n in sorted(table.items())],
        # Per source tag and share class: the Max teacher's and the local 27B
        # teacher's runs, the golden exports (new interactive /qwen turns).
        "by_source_tag": [{"kind": k, "source": s, "class": c, "new": n}
                          for (k, s, c), n in sorted(sources.items())],
        "by_class": {c: by_class.get(c, 0) for c in SHARE_CLASSES},
        "watermarks": {k: file_counts.get(k, 0) for k in ("new", "grown", "unchanged",
                                                          "rewritten")},
        "golden_export_rowid": max(export_rowid, before) or None,
        "golden_export_rowid_before": before or None,
        "offspring_adapter_unknown": sum(1 for r in rows if r["origin"] == "offspring"
                                         and r["adapter_version"] == "unknown"),
    }
    if counts["partial_last_line"]:
        details["partial_last_lines"] = counts["partial_last_line"]
    origin_text = ", ".join(f"{k} {v}" for k, v in sorted(by_origin.items())) or "none"
    class_text = ", ".join(f"{c} {by_class[c]}" for c in SHARE_CLASSES if by_class.get(c))
    summary = (f"{len(rows)} new trace(s) from {files} file(s) ({origin_text}"
               + (f"; {class_text}" if class_text else "") + "); "
               f"{counts['already_harvested']} already harvested; "
               f"{counts['unreadable']} unreadable")
    if details["offspring_adapter_unknown"]:
        run.issue(f"{details['offspring_adapter_unknown']} offspring trace(s) carry no adapter "
                  "version; they are tagged 'unknown' and treated as Max-derived")
    if counts["unreadable"]:
        run.issue(f"{counts['unreadable']} harvest line(s) were not usable traces")
    if not files:
        run.issue("no harvest source matched a file (flywheel.harvest.golden / extra)")
    if run.dry_run:
        run.facts["dry_harvest"] = rows
        return StageResult("harvest", DRY_RUN, summary="would record " + summary, details=details)
    # Content first, then the row file: harvest.jsonl is the commit point, and
    # every later run's dedupe reads it. The watermarks move only after it: a
    # crash in between re-reads those lines next time, and the dedupe drops them.
    run.run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(run.run_dir / "harvest-traces.jsonl", contents)
    _atomic_write_jsonl(run.run_dir / "harvest.jsonl", rows)
    run.state["harvest"] = {"files": new_marks, "updated": run.run_id,
                            "golden_export_rowid": max(export_rowid, before) or None}
    run.save()
    return StageResult("harvest", OK, summary=summary, details=details)


# ---------------------------------------------------------------------------
# Verify: what may be trained on
# ---------------------------------------------------------------------------


def _tokens(name: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT.split(name.lower()) if t]


def _names_within(name: str, inside: str) -> bool:
    """True when *name*'s tokens stand, in order and adjacent, in *inside*'s.

    ``distill-9b-fw-x-Q4_K_M`` is within ``distill-9b-fw-x-Q4_K_M.gguf``, and
    ``student-9b`` within ``models/student-9b@fw-x``; ``qwen3.5-9b`` is not
    within ``qwen3.8-27b``.
    """
    want, have = _tokens(name), _tokens(inside)
    return bool(want) and any(have[i:i + len(want)] == want
                              for i in range(len(have) - len(want) + 1))


# Placeholder names that never identify a model (tag_trace's fallbacks).
_NO_NAME = frozenset({"", "unknown", "none"})


def self_verifier(row: dict[str, Any], by: str, student_models: tuple[str, ...] = (),
                  adapters: Any = ()) -> bool:
    """True when the verifier *by* of an offspring trace is the student itself.

    The student goes by many names: any id of the trace's own model (its
    ``_meta.model``, served id or source model, with or without a path or a
    ``.gguf`` suffix), its adapter version, any adapter this job trained, and
    any ``harvest.student_models`` match. Over-matching only drops a trace,
    the safe direction for this guard.
    """
    if _matches(by, student_models):
        return True
    # Any adapter of ours anywhere in the name (tag_trace's "unknown" rule).
    if any(str(v).lower() in by.lower() for v in adapters or ()
           if str(v).lower() not in _NO_NAME):
        return True
    own = {str(row.get("source_model") or ""), str(row.get("adapter_version") or ""),
           *(str(x) for x in row.get("model_ids") or [])}
    return any(_names_within(name, by) for name in own if name.lower() not in _NO_NAME)


def verification(row: dict[str, Any], student_models: tuple[str, ...] = (),
                 adapters: Any = ()) -> tuple[str, str]:
    """(basis, "") when the trace may be trained on, else ("", why not).

    * a ``fail`` verdict drops anything;
    * offspring: only a ``pass`` from a named verifier that is neither the
      student itself (:func:`self_verifier`; *student_models* and *adapters*
      are the config's student patterns and the adapters this job trained)
      nor the hosted teacher (build.py's ``judge_is_max``: no Max judge, the
      pay-per-token one included, verifies a student). Tool success is never
      enough;
    * teacher: a ``pass`` verdict from a judge that is not Max, or for a
      golden trace its tool-success label (the exporter writes only calls
      that ran without error). An extra-source teacher trace needs such a
      verdict, as the builder requires. The pay-per-token Max judge
      (:func:`is_payg_judge`, ``payg_judge`` on the row) counts on a
      Max-derived teacher trace, as build.py applies it to a Max row; on a
      local 27B row it does not (build.py wants a ``--judge-models`` judge
      there), and any other Max judge never counts (``max_judged``);
    * unknown origin: never.
    """
    if row.get("verdict") == "fail":
        return "", "verdict_fail"
    origin = row.get("origin")
    by = str(row.get("verified_by") or "").strip()
    passed = row.get("verdict") == "pass"
    if origin == "offspring":
        if not passed or by.lower() in _NO_NAME:
            return "", "unverified_offspring"
        if self_verifier(row, by, student_models, adapters):
            return "", "self_verified_offspring"
        if is_max_name(by):
            return "", "max_judged_offspring"
        return "judge-pass", ""
    if origin == "teacher":
        max_judged = passed and is_max_name(by)
        payg = max_judged and row.get("payg_judge") is True and row.get("max_derived") is True
        if passed and (not max_judged or payg):
            return "judge-pass", ""
        if row.get("kind") == "golden":
            return "tool-success", ""
        return "", "max_judged" if max_judged else "unverified_extra"
    return "", "unknown_origin"


def pool_source(row: dict[str, Any]) -> str:
    """The flywheel's provenance tag for a pooled trace.

    The builder copies ``_meta.source`` onto every training row as
    ``source``, which is what the train-time guards read: origin, model and
    adapter survive into the dataset, and a Max-derived row says so in a
    token the Max guard matches. :func:`pooled_meta` decides whether it
    replaces the trace's own tag (only an offspring or untagged trace's).
    """
    tail = "/max-derived" if row.get("max_derived") else ""
    return (f"flywheel/{row.get('origin')}/{row.get('source_model')}"
            f"@{row.get('adapter_version')}{tail}")


def pooled_meta(meta: dict[str, Any], row: dict[str, Any], basis: str,
                run_id: str) -> dict[str, Any]:
    """The ``_meta`` a verified trace enters the pool with.

    The builder's data policy keys on the trace's own ``source`` tag, so a
    tagged teacher trace keeps it, Max-derived or not: ``--train-sources``
    (e.g. ``qwen38-27b-gym``) lets it train, ``builtin-smoke`` (hand-written
    task text) never trains, and ``max-payg-gym`` gets build.py's PAYG rules
    (refused without the Q6b allow, no early dedupe claim, out of the
    Max-only hold-out plan). Renamed, any of those rows would pass as a plain
    Max row. A Max-derived one still says so to the builder and the train
    guards: ``max_derived`` is set here, which build.py refuses on a
    train-source row, and its model or tag names Max. The generator's
    ``verification`` record is read strictly and stays untouched.

    Offspring and untagged traces are tagged :func:`pool_source` instead
    (their own tag kept as ``source_original``): never a listed train
    source, and named offspring or Max for the train-time guards. The job's
    own decision and tag are recorded under ``flywheel``, never in
    ``verification``: a record there without a decision fails the builder's
    check.
    """
    out = dict(meta)
    tag = out.get("source")
    own = tag.strip() if isinstance(tag, str) and tag.strip() else ""
    if row.get("origin") != "teacher" or not own:
        if own and "source_original" not in out:
            out["source_original"] = tag
        out["source"] = pool_source(row)
    out.update({
        "source_model": row["source_model"], "adapter_version": row["adapter_version"],
        "origin": row["origin"], "max_derived": row["max_derived"], "flywheel_run": run_id,
        "flywheel": {"tag": pool_source(row), "run": run_id,
                     "verification": {"basis": basis, "by": row.get("verified_by") or None}},
    })
    if basis == "judge-pass":
        out["verdict"] = "pass"   # the builder's own verified-extra check reads it
    return out


def _verify_rows(run: Run, rows: list[dict[str, Any]]
                 ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decided: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    verified_by_origin: Counter[str] = Counter()
    verified_by_class: Counter[str] = Counter()
    payg = 0
    adapters = _dict(run.state.get("adapters"))
    for row in rows:
        basis, why = verification(row, run.cfg.student_models, adapters)
        decided.append({"h": row["h"], "basis": basis, "reason": why})
        if basis:
            verified_by_origin[str(row.get("origin"))] += 1
            cls = row.get("class")
            verified_by_class[cls if cls in SHARE_CLASSES
                              else "max" if row.get("max_derived") else "other"] += 1
            payg += basis == "judge-pass" and row.get("payg_judge") is True
        else:
            dropped[why] += 1
    verified = sum(verified_by_origin.values())
    counts: dict[str, Any] = {"verified": verified, "dropped": dict(dropped),
                              "verified_by_origin": dict(verified_by_origin),
                              "verified_by_class": {c: verified_by_class.get(c, 0)
                                                    for c in SHARE_CLASSES},
                              "rows": len(rows)}
    if payg:
        counts["payg_max_judge"] = payg
    return decided, counts


def stage_verify(run: Run) -> StageResult:
    if run.dry_run:
        rows = run.facts.get("dry_harvest")
        if rows is None:
            return StageResult("verify", SKIPPED, reason="harvest did not run")
        _, counts = _verify_rows(run, rows)
        run.facts["dry_verified"] = counts["verified"]
        return StageResult("verify", DRY_RUN, summary=(
            f"would verify {counts['verified']} of {counts['rows']} new trace(s); "
            f"dropped {counts['dropped'] or 'none'}"), details=counts)

    runs_dir = run.data / "runs"
    pending = sorted(p.parent for p in runs_dir.glob("*/harvest.jsonl")
                     if not (p.parent / "verify.json").exists())
    if not pending:
        return StageResult("verify", SKIPPED, reason="no harvest is waiting to be verified")
    from prometheus.security.log_redaction import redact_capture

    total: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    this_run: dict[str, Any] = {}
    for rdir in pending:
        if run.out_of_time():
            return StageResult("verify", PAUSED, reason=(
                f"the window closed with {len(pending)} harvest(s) unverified; the next window "
                "verifies them"), details={"verified": total["verified"]})
        rows = list(_read_jsonl(rdir / "harvest.jsonl"))
        decided, counts = _verify_rows(run, rows)
        trainable = {d["h"]: d["basis"] for d in decided if d["basis"]}
        tags = {r["h"]: r for r in rows}
        pooled: list[dict[str, Any]] = []
        contents = rdir / "harvest-traces.jsonl"
        items: Iterator[dict[str, Any]] = _read_jsonl(contents) if contents.exists() else iter([])
        for i, item in enumerate(items):
            if i % CLOCK_EVERY_LINES == 0 and run.out_of_time():
                return StageResult("verify", PAUSED, reason=(
                    f"the window closed while verifying {rdir.name}; nothing of it was "
                    "recorded and the next window verifies it again"))
            h = item.get("h")
            if h not in trainable:
                continue
            row = tags[h]
            trace = redact_capture(item["trace"])
            trace["_meta"] = pooled_meta(dict(trace.get("_meta") or {}), row, trainable[h],
                                         rdir.name)
            pooled.append(trace)
        # The pool file first, verify.json last: verify.json is the commit point.
        _atomic_write_jsonl(run.data / "pool" / f"{rdir.name}.jsonl", pooled)
        record = {**counts, "pooled": len(pooled), "decisions": decided,
                  "verified_at": run.clock().isoformat(timespec="seconds")}
        _atomic_write(rdir / "verify.json", json.dumps(record, indent=1) + "\n")
        total["verified"] += counts["verified"]
        total["pooled"] += len(pooled)
        total["payg_max_judge"] += int(counts.get("payg_max_judge") or 0)
        for cls, n in _dict(counts.get("verified_by_class")).items():
            total[f"class:{cls}"] += int(n)
        dropped.update(counts["dropped"])
        if rdir.name == run.run_id:
            this_run = counts
    details = {"runs": [p.name for p in pending], "verified": total["verified"],
               "pooled": total["pooled"], "dropped": dict(dropped), "this_run": this_run,
               "verified_by_class": {c: total[f"class:{c}"] for c in SHARE_CLASSES}}
    if total["payg_max_judge"]:
        details["payg_max_judge"] = total["payg_max_judge"]
    offspring_dropped = sum(dropped[k] for k in OFFSPRING_DROPS)
    if offspring_dropped:
        details["offspring_dropped"] = offspring_dropped
    drop_text = ", ".join(f"{k} {v}" for k, v in sorted(dropped.items())) or "none"
    return StageResult("verify", OK, summary=(
        f"{total['verified']} verified ({total['pooled']} into the pool) from "
        f"{len(pending)} harvest(s); dropped: {drop_text}"), details=details)


def verified_by_run(data: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in sorted((data / "runs").glob("*/verify.json")):
        rec = _read_json(path)
        if isinstance(rec, dict):
            out[path.parent.name] = int(rec.get("verified") or 0)
    return out


def new_verified_since_last_train(run: Run) -> int:
    cutoff = str((run.state.get("last_train") or {}).get("cutoff") or "")
    n = sum(v for rid, v in verified_by_run(run.data).items() if rid > cutoff)
    return n + int(run.facts.get("dry_verified") or 0)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def verify_decisions(data: Path) -> dict[str, str]:
    """Trace hash -> "" (may be trained on) or why not, over every verified harvest."""
    out: dict[str, str] = {}
    for path in sorted((data / "runs").glob("*/verify.json")):
        rec = _read_json(path)
        for d in (rec.get("decisions") or []) if isinstance(rec, dict) else []:
            if isinstance(d, dict) and isinstance(d.get("h"), str):
                out[d["h"]] = "" if d.get("basis") else str(d.get("reason") or "not_verified")
    return out


def write_golden_input(run: Run, files: list[Path], dest: Path) -> dict[str, Any] | None:
    """The builder's copy of the golden sources: what verify passed, nothing else.

    The builder reads golden traces itself (with their Phase 1 index and judge
    verdicts, both keyed by file name and position), so verify's decisions
    reach it here: a trace it passed is copied byte for byte; any other (an
    offspring, unknown origin or fail-labelled trace, or one exported after
    this window's harvest) becomes a stub under :data:`WITHHELD_MODEL` on the
    same line. File names, line numbers and the order of readable traces stay
    the original's. None when the window closed first.
    """
    decisions = verify_decisions(run.data)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    kept = 0
    withheld: Counter[str] = Counter()
    for path in files:
        with open(path, "rb") as src, open(dest / path.name, "wb") as out:
            for lineno, raw in enumerate(src, 1):
                if lineno % CLOCK_EVERY_LINES == 0 and run.out_of_time():
                    return None
                text = raw.decode("utf-8", errors="replace")
                try:
                    trace = json.loads(text) if text.strip() else None
                except ValueError:
                    trace = None
                if not isinstance(trace, dict) or not trace.get("messages"):
                    out.write(raw)            # the builder cannot read it either
                    continue
                why = ("unusable" if shape_problem(trace)
                       else decisions.get(trace_hash(trace), "not_harvested"))
                if not why:
                    out.write(raw)
                    kept += 1
                    continue
                stub = {"messages": [{"role": "user", "content": ""}],
                        "_meta": {"model": WITHHELD_MODEL, "flywheel_withheld": why}}
                out.write((json.dumps(stub) + "\n").encode("utf-8"))
                withheld[why] += 1
    return {"files": len(files), "kept": kept, "withheld": dict(withheld)}


def resolve_build_script(cfg: FlywheelConfig, repo: Path) -> Path | None:
    if cfg.build_script:
        path = _resolve_under(cfg.build_script, repo)
        return path if path.is_file() else None
    for rel in BUILD_SCRIPTS:
        if (repo / rel).is_file():
            return repo / rel
    return None


def dataset_facts(pol: Any, out: Path, report: dict[str, Any]) -> dict[str, Any]:
    """What the state keeps about a dataset's Max-derived side (aggregates only):

    * ``max_derived``: train/val hold Max-derived rows (artifact_policy's
      provenance of the directory: build.py's per-split counts or its
      ``max_derived`` flag; ``None`` when nothing shows either way);
    * ``holds_max_rows``: any set holds them (eval-tools is built from Max
      traces), so the directory is private either way;
    * ``approval``: the Q6b approval the dataset recorded (build.py's
      ``data_policy.max_derived_allow``), or None;
    * ``share``: build.py's ``data_policy.share`` (rows and percent per class
      per split: Max-derived, the local 27B teacher's, other).
    """
    md: bool | None = None
    holds = False
    if pol is not None:
        try:
            prov = pol.provenance(out)
            md, holds = prov.max_derived, bool(prov.holds_max_rows)
        except (OSError, ValueError) as exc:
            logger.warning("provenance of %s unreadable: %s", out, exc)
    if md is None and isinstance(report.get("max_derived"), bool):
        md = report["max_derived"]
    approval = None
    if pol is not None:
        approval = pol.dataset_approval(report) \
            or pol.dataset_approval(_read_json_quiet(out / "dataset-manifest.json"))
    share = _dict(_dict(report.get("data_policy")).get("share"))
    by_split = _dict(share.get("by_split"))
    return {"max_derived": md, "holds_max_rows": holds or md is True, "approval": approval,
            "share": ({"by_split": by_split, "approval": share.get("approval")}
                      if by_split else None)}


def _read_json_quiet(path: Path) -> Any:
    try:
        return _read_json(path)
    except (OSError, ValueError):
        return None


def stage_build(run: Run) -> StageResult:
    cfg = run.cfg
    script = resolve_build_script(cfg, run.repo)
    if script is None:
        wanted = cfg.build_script or " or ".join(BUILD_SCRIPTS)
        return StageResult("build", FAILED, reason=f"no dataset builder at {wanted} "
                                                   f"(under {run.repo})")
    if not cfg.build_args:
        return StageResult("build", SKIPPED, reason=(
            "flywheel.build.args is empty: the builder needs this host's --pack and "
            "--tokenizer-dir (and --index for the golden traces)"))
    sets = run.state.get("sets") if isinstance(run.state.get("sets"), dict) else None
    verified = verified_by_run(run.data)
    newest = max((rid for rid, n in verified.items() if n > 0), default="")
    if run.dry_run and run.facts.get("dry_verified"):
        newest = run.run_id
    allow = cfg.max_derived
    listed = session_list_digest(allow.exclude_sessions_file)
    # Same inputs, same builder, same arguments: the same sets. A changed
    # argument (the operator fixing --pack, say), a changed Q6b allow or an
    # edit to the session exclusion list (its ids, not only its path)
    # rebuilds without new traces.
    inputs = hashlib.sha256(json.dumps(
        [str(script), list(cfg.build_args), list(cfg.golden), allow.complete,
         allow.approval_ref if allow.complete else "",
         str(allow.exclude_sessions_file or ""), listed]).encode()).hexdigest()[:16]
    if sets and newest <= str(sets.get("run") or "") and sets.get("inputs") == inputs:
        return StageResult("build", SKIPPED, reason=(
            f"no verified traces since the {sets.get('run')} build; its sets stand"),
            details={"sets": sets})
    if allow.exclude_sessions_file is not None:
        problem = session_list_problem(allow.exclude_sessions_file)
        if problem:
            return StageResult("build", REFUSED, reason=(
                f"{problem} (flywheel.max_derived.exclude_sessions_file): the builder drops "
                "those sessions from every split, so it was not started without them"))
    pol = _policy_or_none(run.repo)
    if allow.complete and pol is None:
        return StageResult("build", FAILED, reason=(
            f"the Q6b allow is set but {POLICY_SCRIPT} cannot be loaded, so the Max-derived "
            "sets cannot be named for it; the builder was not started"))
    # build.py's allow wants the tag in the output's name (artifact_policy).
    out = run.run_dir / (pol.tag_name(SETS_DIR) if allow.complete else SETS_DIR)
    golden_dir = run.data / "build-input" / "golden"
    golden_files = _expand(cfg.golden)
    pool = sorted((run.data / "pool").glob("*.jsonl"))
    argv = [sys.executable, str(script), "build", *cfg.build_args]
    # Always at least one --traces: without it the builder reads its default
    # golden glob, around the copy.
    for name in [p.name for p in golden_files] or ["*.jsonl"]:
        argv += ["--traces", str(golden_dir / name)]
    for path in pool:
        argv += ["--extra-traces", str(path)]
    argv += ["--out", str(out)]
    if allow.exclude_sessions_file is not None:
        argv += [BUILD_EXCLUDE_FLAG, str(allow.exclude_sessions_file)]
    if allow.complete:
        # The explicit, recorded allow: build.py writes the reference into
        # report.json and dataset-manifest.json, which the train stage reads back.
        argv += [BUILD_ALLOW_FLAG, BUILD_APPROVAL_FLAG, allow.approval_ref]
    if run.dry_run:
        said = []
        if allow.complete:
            said.append(f"under the Q6b allow {allow.approval_ref!r}, into {out.name}/")
        if allow.exclude_sessions_file is not None:
            said.append("with the session exclusion list")
        return StageResult("build", DRY_RUN, summary=(
            "would run" + (f" ({', '.join(said)})" if said else "") + f": {_quote(argv)}"))
    names = Counter(p.name for p in golden_files)
    repeated = sorted(n for n, k in names.items() if k > 1)
    if repeated:
        return StageResult("build", FAILED, reason=(
            f"harvest.golden matches more than one file named {', '.join(repeated)}; the "
            "builder names traces <file name>:<line>, so names must be unique"))
    # A rerun in the same window starts clean: a builder that crashes must not
    # leave the previous attempt's report.json to be read as its own.
    if out.exists():
        shutil.rmtree(out)
    if sets and sets.get("dir") == str(out):
        run.state.pop("sets", None)
        run.save()
    golden = write_golden_input(run, golden_files, golden_dir)
    if golden is None:
        return StageResult("build", PAUSED, reason=(
            "the window closed while copying the verified golden traces; the next window "
            "builds again"))
    res = run.runner(run.command(argv, "build"))
    if res.stopped:
        return StageResult("build", PAUSED, reason=(
            "the window closed before the builder finished; the next window builds again"))
    report = _read_json(out / "report.json") if res.returncode in (0, 1) else None
    gates = (report or {}).get("gates") if isinstance(report, dict) else None
    if res.returncode not in (0, 1) or not isinstance(gates, dict):
        return StageResult("build", FAILED, reason=(
            f"the builder exited {res.returncode} without a report ({res.tail})"),
            details={"golden": golden})
    assert isinstance(report, dict)
    failed = sorted(k for k, v in gates.items() if isinstance(v, dict) and v.get("pass") is False)
    settings = _dict(report.get("settings"))
    written = _dict(_dict(report.get("inputs")).get("written"))
    overall = str(gates.get("overall") or "")
    # The runbook's Phase 3 gate includes the byte-for-byte spot check; sets
    # without a passed one are not trainable, whatever the overall verdict says.
    spot = str(_dict(gates.get("spot_check")).get("status") or "missing")
    verdict = "INCOMPLETE" if overall == "PASS" and spot != "PASS" else overall
    if verdict == "PASS" and res.returncode != 0:
        verdict = f"exit {res.returncode}"
    run.state["sets"] = {
        "run": run.run_id, "dir": str(out), "gates": verdict, "spot_check": spot,
        "failed_gates": failed, "teacher_model": settings.get("teacher_model"),
        "rows": written, "inputs": inputs, "sessions_sha256": listed or "",
        **dataset_facts(pol, out, report),
    }
    run.save()
    details = {"failed_gates": failed, "spot_check": spot, "rows": written,
               "pool_files": len(pool), "golden": golden,
               "max_derived": run.state["sets"]["max_derived"],
               "approval": run.state["sets"]["approval"]}
    if run.state["sets"].get("share"):
        details["share"] = run.state["sets"]["share"]
    if verdict.startswith("exit "):
        return StageResult("build", REFUSED, reason=(
            f"the builder exited {res.returncode} with a passing report; nothing trains on "
            "these sets"), details=details)
    if failed or verdict not in ("PASS", "INCOMPLETE"):
        return StageResult("build", REFUSED, reason=(
            f"the sets failed the builder's gates ({', '.join(failed) or overall or 'overall'}); "
            "nothing trains on them"), details=details)
    if verdict != "PASS":
        return StageResult("build", REFUSED, reason=(
            f"the byte-for-byte spot check has not passed (spot check {spot}; the builder's "
            f"verdict {overall or 'none'}), so nothing trains on these sets. Give the builder "
            "--spot-check-server <a llama-server serving the student> in flywheel.build.args"),
            details=details)
    withheld = sum(golden["withheld"].values())
    return StageResult("build", OK, summary=(
        f"sets built: train {written.get('train.jsonl', '?')}, val {written.get('val.jsonl', '?')}, "
        f"eval-tools {written.get('eval-tools.jsonl', '?')}; gates PASS; "
        f"{withheld} golden trace(s) withheld as unverified"), details=details)


# ---------------------------------------------------------------------------
# Train: the gates, the guards, the frozen data, resume
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    rows: int = 0
    max_rows: int = 0
    offspring_rows: int = 0
    offspring_unverified: int = 0
    unknown_rows: int = 0
    unreadable: int = 0
    leaked_rows: int = 0
    teacher_model: str = ""
    refusals: list[str] = field(default_factory=list)
    # Rows per share class (Max-derived / the local 27B teacher's / other).
    share: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SHARE_CLASSES, 0))


def row_keys(row: dict[str, Any]) -> set[str]:
    """A row's :data:`HOLDOUT_KEYS` values, from the row or its ``meta``.

    One set for all three: a group id is its smallest task id, so a task id
    on one side matching a group id on the other is the same task.
    """
    meta = _dict(row.get("meta"))
    return {v for k in HOLDOUT_KEYS for v in (row.get(k), meta.get(k))
            if isinstance(v, str) and v}


def score_case_keys(cfg: FlywheelConfig, repo: Path) -> set[str]:
    """Every :func:`row_keys` value of the configured score cases."""
    keys: set[str] = set()
    for raw in cfg.score_cases:
        path = _resolve_under(raw, repo)
        if not path.is_file():
            continue             # the score stage reports a missing case file
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line) if line.strip() else None
                except ValueError:
                    continue
                if isinstance(row, dict):
                    keys |= row_keys(row)
    return keys


def row_ids(paths: list[Path]) -> set[str]:
    """The rows of *paths* by id (build.py's ``id``, else the line's hash)."""
    out: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                rid = row.get("id") if isinstance(row, dict) else None
                out.add(rid if isinstance(rid, str) and rid
                        else hashlib.sha256(line.strip().encode()).hexdigest())
    return out


def check_training_data(paths: list[Path], cfg: FlywheelConfig,
                        teacher_model: str | None,
                        held_out: set[str] | frozenset[str] = frozenset()) -> GuardResult:
    """The offspring, Max-data and score-case guards over the exact rows a train would read.

    A row's origin comes from its ``source`` (see :func:`pool_source`). The
    builder keeps only rows of its ``--teacher-model``, so when that model is
    the hosted teacher every row is Max-derived whatever its own source says;
    an unknown teacher model counts as Max-derived (nothing shows otherwise),
    and so does a row whose own ``model`` names Max or whose ``max_derived``
    is true.
    A row sharing a :func:`row_keys` value with *held_out* (the score cases)
    is a task the candidate would be scored on.
    """
    g = GuardResult(teacher_model=str(teacher_model or ""))
    all_max = not teacher_model or is_max_name(str(teacher_model))
    for path in paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    g.unreadable += 1
                    continue
                if not isinstance(row, dict):
                    g.unreadable += 1
                    continue
                meta = _dict(row.get("meta"))
                src = str(row.get("source") or meta.get("source") or "")
                label = str(row.get("label") or meta.get("label") or "")
                g.rows += 1
                parts = src.split("/")
                origin = parts[1] if len(parts) > 2 and parts[0] == "flywheel" else ""
                if origin == "offspring":
                    g.offspring_rows += 1
                    if label not in OFFSPRING_TRAINABLE_LABELS:
                        g.offspring_unverified += 1
                elif origin == "unknown":
                    g.unknown_rows += 1
                # build.py also writes the row's model and max_derived: either
                # naming Max makes the row Max-derived whatever its source says.
                model = str(row.get("model") or meta.get("model") or "")
                flag = row["max_derived"] if "max_derived" in row else meta.get("max_derived")
                own_max = (is_max_name(src) or is_max_name(model) or flag is True
                           or meta.get("max_derived") is True)
                is_max = all_max or own_max
                g.max_rows += is_max
                # The share is the row's own: build.py names the Max model as
                # its teacher for any set that mixes in a Max row, which says
                # nothing of a local 27B row beside it (max_derived false). A
                # row that does not say is counted as the guard counts it.
                cls = share_class(own_max or (all_max and flag is not False), src)
                g.share[cls] = g.share.get(cls, 0) + 1
                if held_out and row_keys(row) & held_out:
                    g.leaked_rows += 1
    if g.unreadable:
        g.refusals.append(f"{g.unreadable} training row(s) are unreadable, so their origin "
                          "cannot be checked")
    if g.offspring_unverified:
        g.refusals.append(f"{g.offspring_unverified} offspring row(s) carry no judge verdict "
                          "(unverified offspring output is never trained on)")
    if g.unknown_rows:
        g.refusals.append(f"{g.unknown_rows} row(s) are of unknown origin")
    if g.leaked_rows:
        g.refusals.append(f"{g.leaked_rows} training row(s) share an id, task or group with the "
                          "score cases (flywheel.score.cases): the candidate would be scored on "
                          "what it trained on")
    if g.max_rows and not cfg.max_derived.complete:
        g.refusals.append(
            f"{g.max_rows} of {g.rows} row(s) are Max-derived and no operator approval is "
            "recorded (flywheel.max_derived: approved, approval_ref, exclude_sessions_file; "
            "DISTILL-OPEN-QUESTIONS #6, Q6b ruling). Max-derived rows train only under that "
            "explicit allow")
    return g


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
            raise
        shutil.copy2(src, dst)


def latest_checkpoint(directory: Path) -> Path | None:
    """The highest-step ``checkpoint-<step>`` directory (the trainer convention)."""
    best: tuple[int, Path] | None = None
    if directory.is_dir():
        for child in directory.iterdir():
            m = _CHECKPOINT_RE.match(child.name)
            if m and child.is_dir() and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), child)
    return best[1] if best else None


def _frozen_problem(t: dict[str, Any]) -> str:
    data_dir = Path(str(t.get("data_dir") or ""))
    for name, sha in (t.get("sha256") or {}).items():
        path = data_dir / name
        if not path.is_file():
            return f"the frozen {name} of {t.get('adapter_version')} is gone ({path})"
        if _sha256(path) != sha:
            return (f"the frozen {name} of {t.get('adapter_version')} changed since the train "
                    "started; a resume would train on different data")
    return ""


def _freeze(run: Run, sets: dict[str, Any], guard: GuardResult, pol: Any) -> dict[str, Any]:
    """Pin the train's data: hard links (or copies) of the split files, with hashes.

    A resume weeks later trains on these exact bytes, whatever later builds
    write. The run directory is named after the adapter version, so a trainer
    that names its logs after its output directory names them usefully. A
    train on Max-derived rows is named for it (``fw-<run>-max-derived``, by
    artifact_policy's ``tag_name``): its run directory (the trainer refuses an
    untagged ``--out`` for such data), its frozen data and, later, its models.
    """
    max_derived = guard.max_rows > 0
    base, n = f"fw-{run.run_id}", 1
    while True:
        version = pol.name_for(base if n == 1 else f"{base}-{n}", max_derived)
        if not ((run.data / "train" / version).exists()
                or (run.data / "train" / f"{version}.data").exists()):
            break
        n += 1
    run_dir = run.data / "train" / version
    data_dir = run.data / "train" / f"{version}.data"
    data_dir.mkdir(parents=True)
    shas: dict[str, str] = {}
    for name in (*FROZEN_FILES, "report.json", "dataset-manifest.json"):
        src = Path(str(sets["dir"])) / name
        if src.is_file():
            _link_or_copy(src, data_dir / name)
            shas[name] = _sha256(data_dir / name)
    return {
        "adapter_version": version, "status": "running", "data_dir": str(data_dir),
        "run_dir": str(run_dir),
        "adapter_dir": str(run_dir / pol.name_for(ADAPTER_SUBDIR, max_derived)),
        "cutoff": str(sets.get("run") or run.run_id), "sets": str(sets["dir"]),
        "teacher_model": guard.teacher_model, "max_derived": max_derived,
        "share": dict(guard.share),
        "approval": run.cfg.max_derived.approval_ref if max_derived else None,
        # the session exclusion list the data was built under (a resume needs it unchanged)
        "sessions_sha256": str(sets.get("sessions_sha256") or ""),
        "sha256": shas, "rows": guard.rows, "attempts": 0, "failures": 0,
        "frozen_in": run.run_id,
    }


def find_adapter(pol: Any, run_dir: Path, max_derived: bool) -> tuple[Path | None, bool, str]:
    """The adapter a finished train left in *run_dir*: (path, Max-derived, problem).

    The trainer names it by artifact_policy's rule: ``adapter-max-derived``
    for Max-derived data (train_qlora.py), ``adapter`` otherwise. A tagged
    adapter is Max-derived whatever the data check said (the trainer's own
    check fails closed). An untagged one is refused when the data is
    Max-derived, or when its own provenance chain (the trainer's run
    manifest) shows it is: its name would not say what it is.
    """
    tagged = run_dir / pol.tag_name(ADAPTER_SUBDIR)
    plain = run_dir / ADAPTER_SUBDIR
    if tagged.is_dir():
        return tagged, True, ""
    if plain.is_dir():
        if max_derived:
            return None, True, (f"the data is Max-derived but the trainer left an untagged "
                                f"{plain} (expected {tagged.name}/, DISTILL-OPEN-QUESTIONS Q6b)")
        prov = pol.provenance(plain)
        if prov.max_derived is True:
            return None, True, (f"{plain} is Max-derived ({'; '.join(prov.evidence[:2])}) but "
                                f"its name is not tagged (expected {tagged.name}/)")
        return plain, False, ""
    want = tagged if max_derived else plain
    return None, max_derived, f"left no adapter at {want}"


def max_derived_train_problem(run: Run, data_dir: Path, *, resuming: bool = False
                              ) -> tuple[str, str | None]:
    """(why a train on Max-derived rows may not start or resume, "" when it
    may; the approval its data records).

    The allow is complete (:func:`check_training_data` refuses Max rows
    without it), the session exclusion list is usable, and the dataset itself
    recorded an approval when build.py made it (``--allow-max-derived
    --max-derived-approval``), as the trainer will check too. A new train
    needs the very approval the config names: a build under the current
    allow comes first. A resume trains on its frozen data, which keeps the
    approval it was built and frozen under; a re-worded approval_ref applies
    from the next build on and never strands it.
    """
    allow = run.cfg.max_derived
    if not allow.complete:
        return ("no complete Q6b allow is recorded for these Max-derived rows "
                "(flywheel.max_derived: approved, approval_ref, exclude_sessions_file; "
                "DISTILL-OPEN-QUESTIONS Q6b)"), None
    problem = session_list_problem(allow.exclude_sessions_file)
    if problem:
        return f"{problem}: a Max-derived train needs it (DISTILL-OPEN-QUESTIONS Q6b)", None
    pol = load_policy(run.repo)
    recorded = pol.dataset_approval(_read_json_quiet(data_dir / "report.json")) \
        or pol.dataset_approval(_read_json_quiet(data_dir / "dataset-manifest.json"))
    if resuming:
        if recorded:
            return "", recorded
        return (f"the frozen data in {data_dir} records no Max-derived approval, so no allow "
                f"can resume it: remove 'train' from {run.state_path} to start a new train on "
                "sets built under the allow"), None
    if recorded != allow.approval_ref:
        said = f"the approval {recorded!r}" if recorded else "no Max-derived approval"
        return (f"the dataset records {said}, not flywheel.max_derived.approval_ref "
                f"{allow.approval_ref!r}: it was not built under this allow (build.py "
                f"{BUILD_ALLOW_FLAG} {BUILD_APPROVAL_FLAG}). A build under the allow comes "
                "first"), recorded
    return "", recorded


def _supersede(run: Run, t: dict[str, Any], why: str) -> dict[str, Any]:
    """Retire a running or paused train that may not resume (*why*), so a new
    one can start on sets built under the current settings. Its record moves
    to ``state["superseded_trains"]``; its frozen data and checkpoints stay on
    disk (private when Max-derived). A dry run only says so."""
    rec = {k: t.get(k) for k in ("adapter_version", "data_dir", "run_dir", "max_derived",
                                 "frozen_in", "attempts")}
    rec.update(why=why, superseded_in=run.run_id)
    run.issue(f"train {t.get('adapter_version')} is not resumed: {why}. Its frozen data and "
              "checkpoints stay on disk (private when Max-derived); a new train starts on sets "
              "built under the current list")
    if not run.dry_run:
        run.state.setdefault("superseded_trains", []).append(rec)
        run.state.pop("train", None)
        run.save()
    return rec


def _placeholders(args: tuple[str, ...]) -> set[str]:
    return {name for arg in args for _, name, _, _ in string.Formatter().parse(arg)
            if name is not None}


def _trainer_status(run_dir: Path) -> str:
    """The trainer's own word on how an exit 0 ended ("" when it keeps none)."""
    try:
        manifest = _read_json(run_dir / TRAINER_MANIFEST)
    except (OSError, ValueError) as exc:
        logger.warning("trainer manifest in %s unreadable: %s", run_dir, exc)
        return "unreadable"
    return str(manifest.get("status") or "") if isinstance(manifest, dict) else ""


def stage_train(run: Run) -> StageResult:
    cfg = run.cfg
    t = run.state.get("train") if isinstance(run.state.get("train"), dict) else None
    pending = new_verified_since_last_train(run)
    details: dict[str, Any] = {"new_verified_since_last_train": pending,
                               "min_new_traces": cfg.min_new_traces}
    if t and t.get("status") == "abandoned":
        return StageResult("train", REFUSED, reason=(
            f"train {t.get('adapter_version')} failed {t.get('failures')} time(s) and was "
            f"abandoned; the operator decides (fix the cause, then remove 'train' from "
            f"{run.state_path})"), details=details)
    # The builder drops the listed sessions from every split, Max-derived or not:
    # no train starts or resumes on data built under another list.
    listed = session_list_digest(cfg.max_derived.exclude_sessions_file)
    if listed is None:
        return StageResult("train", REFUSED, reason=(
            f"{session_list_problem(cfg.max_derived.exclude_sessions_file)} "
            "(flywheel.max_derived.exclude_sessions_file): the builder drops those sessions from "
            "every split, so no train starts or resumes until it is readable"), details=details)
    if t and t.get("status") in ("running", "incomplete") \
            and str(t.get("sessions_sha256") or "") != listed:
        details["superseded"] = _supersede(run, t, (
            "the session exclusion list (flywheel.max_derived.exclude_sessions_file) changed "
            "since its data was frozen, so that data may hold rows of a session listed now"))
        t = None
    resuming = bool(t and t.get("status") in ("running", "incomplete"))
    if resuming and not cfg.train_command:
        return StageResult("train", SKIPPED, reason=(
            f"train {t.get('adapter_version') if t else ''} waits to resume, but "
            "flywheel.train.command is empty"), details=details)
    sets: dict[str, Any] = {}
    if resuming:
        assert t is not None
        details["adapter_version"] = t.get("adapter_version")
        problem = _frozen_problem(t)
        if problem:
            return StageResult("train", REFUSED, reason=problem, details=details)
        data_dir = Path(str(t["data_dir"]))
        train_p, val_p = data_dir / "train.jsonl", data_dir / "val.jsonl"
        teacher = t.get("teacher_model")
    else:
        if pending < cfg.min_new_traces:
            return StageResult("train", SKIPPED, reason=(
                f"{pending} new verified trace(s) since the last train; a train needs "
                f"{cfg.min_new_traces} (flywheel.train.min_new_traces)"), details=details)
        if not cfg.train_command:
            return StageResult("train", SKIPPED, reason=(
                "flywheel.train.command is empty: no trainer is configured"), details=details)
        built = run.results.get("build")
        if run.dry_run and built is not None and built.status == DRY_RUN:
            max_note = (f"approved under the Q6b allow ({cfg.max_derived.approval_ref!r}); "
                        "the train is tagged max-derived and stays private"
                        if cfg.max_derived.complete
                        else "refused: no operator approval is recorded (flywheel.max_derived)")
            return StageResult("train", DRY_RUN, summary=(
                f"would train on this window's sets if they pass the builder's gates and the "
                f"guards (Max-derived rows: {max_note})"), details=details)
        raw_sets = run.state.get("sets")
        if not isinstance(raw_sets, dict) or not raw_sets.get("dir"):
            return StageResult("train", SKIPPED, reason="no dataset has been built yet",
                               details=details)
        sets = raw_sets
        if sets.get("gates") != "PASS":
            why = (f"failed: {', '.join(sets.get('failed_gates') or [])}"
                   if sets.get("failed_gates") else
                   f"spot check {sets.get('spot_check')}" if sets.get("gates") == "INCOMPLETE"
                   else "overall")
            return StageResult("train", REFUSED, reason=(
                f"the newest sets ({sets.get('run')}) are not trainable: the builder's verdict "
                f"is {sets.get('gates')} ({why})"), details=details)
        if str(sets.get("sessions_sha256") or "") != listed:
            return StageResult("train", REFUSED, reason=(
                f"the newest sets ({sets.get('run')}) were built under another session exclusion "
                "list than flywheel.max_derived.exclude_sessions_file holds now; the build runs "
                "again first (any edit to the list rebuilds)"), details=details)
        newest = max((rid for rid, n in verified_by_run(run.data).items() if n > 0), default="")
        if newest > str(sets.get("run") or ""):
            return StageResult("train", SKIPPED, reason=(
                f"the newest sets ({sets.get('run')}) predate the traces verified in {newest}; "
                "the build has to run first"), details=details)
        train_p = Path(str(sets["dir"])) / "train.jsonl"
        val_p = Path(str(sets["dir"])) / "val.jsonl"
        if not train_p.is_file() or not val_p.is_file():
            return StageResult("train", FAILED, reason=f"the sets in {sets['dir']} have no "
                                                       "train.jsonl / val.jsonl", details=details)
        teacher = sets.get("teacher_model")
        # The verified count above is what came in; this is what the builder
        # kept (its exact --teacher-model, deduped). A week whose new traces all
        # fell out there would retrain on the last train's rows.
        last = t if t and t.get("status") == "complete" else None
        if last:
            before = row_ids([Path(str(last.get("data_dir") or "")) / n
                              for n in ("train.jsonl", "val.jsonl")])
            fresh = len(row_ids([train_p]) - before)
            details["new_train_rows"] = fresh
            if fresh < cfg.min_new_traces:
                return StageResult("train", SKIPPED, reason=(
                    f"the newest sets ({sets.get('run')}) hold {fresh} train row(s) that "
                    f"{last.get('adapter_version')} was not trained on; a train needs "
                    f"{cfg.min_new_traces} (flywheel.train.min_new_traces). The builder keeps "
                    "only its --teacher-model's traces, deduped"), details=details)

    guard = check_training_data([train_p, val_p], cfg, teacher,
                                score_case_keys(cfg, run.repo))
    details["guard"] = asdict(guard)
    if guard.refusals:
        return StageResult("train", REFUSED, reason="; ".join(guard.refusals), details=details)
    pol = _policy_or_none(run.repo)
    if pol is None:
        return StageResult("train", FAILED, reason=(
            f"{POLICY_SCRIPT} cannot be loaded: no train starts without the Max-derived rule "
            "that names its outputs (DISTILL-OPEN-QUESTIONS Q6b)"), details=details)
    max_derived = guard.max_rows > 0 or bool(resuming and t and t.get("max_derived", True))
    if max_derived:
        problem, recorded = max_derived_train_problem(
            run, Path(str(t["data_dir"])) if resuming and t else Path(str(sets["dir"])),
            resuming=resuming)
        if problem:
            return StageResult("train", REFUSED, reason=problem, details=details)
        if resuming and t and recorded != cfg.max_derived.approval_ref:
            details["approval"] = recorded
            run.issue(f"train {t.get('adapter_version')} resumes under the approval its frozen "
                      f"data records ({recorded!r}); flywheel.max_derived.approval_ref now reads "
                      f"{cfg.max_derived.approval_ref!r}, which the next build records")
    verb = f"resume {t.get('adapter_version')}" if resuming and t else "start a train"
    if run.dry_run:
        return StageResult("train", DRY_RUN, summary=(
            f"would {verb} on {guard.rows} row(s) with: {' '.join(cfg.train_command)}"
            + (" (Max-derived: tagged max-derived, private)" if max_derived else "")),
            details=details)
    note = ensure_gpu(run, "train")
    if note:
        return StageResult("train", SKIPPED, reason=note, details=details)

    if not resuming:
        t = _freeze(run, sets, guard, pol)
    assert t is not None
    version = str(t["adapter_version"])
    run_dir = Path(str(t["run_dir"]))
    values = {
        "train": str(Path(t["data_dir"]) / "train.jsonl"),
        "val": str(Path(t["data_dir"]) / "val.jsonl"),
        "data_dir": str(t["data_dir"]), "run_dir": str(run_dir), "adapter_version": version,
    }
    ckpt = latest_checkpoint(run_dir)
    argv = run.fill(cfg.train_command, values)
    # A train that has run before resumes: its run directory already holds its
    # state (train_qlora.py refuses a second start in it). Resume arguments that
    # name {checkpoint} need one to exist.
    if int(t.get("attempts") or 0) > 0 and cfg.resume_args \
            and (ckpt is not None or "checkpoint" not in _placeholders(cfg.resume_args)):
        argv += run.fill(cfg.resume_args, {**values, "checkpoint": str(ckpt or "")})
    if max_derived and MAX_DERIVED_APPROVAL_FLAG not in argv:
        # The operator's say-so on the trainer's own command line (the Q6b
        # allow is complete: checked above); the dataset's recorded approval is
        # the trainer's second key.
        argv.append(MAX_DERIVED_APPROVAL_FLAG)
    t.update(status="running", attempts=int(t.get("attempts") or 0) + 1, last_run=run.run_id)
    run.state["train"] = t
    run.save()
    details.update(adapter_version=version, resumed_from=ckpt.name if ckpt else None)

    res = run.runner(run.command(argv, "train"))
    now = run.clock().isoformat(timespec="seconds")
    said = _trainer_status(run_dir) if res.returncode == 0 and not res.stopped else ""
    finished = res.returncode == 0 and not res.stopped and said in ("", "done")
    found, found_md, missing = (find_adapter(pol, run_dir, bool(t.get("max_derived", True)))
                                if finished else (None, False, ""))
    if finished and found is not None:
        md = found_md or bool(t.get("max_derived", True))
        t.update(status="complete", finished=now, adapter_dir=str(found), max_derived=md)
        run.state["last_train"] = {"adapter_version": version, "cutoff": t["cutoff"],
                                   "finished": now}
        adapters = run.state.setdefault("adapters", {})
        adapters[version] = {"max_derived": md, "adapter_dir": str(found),
                             "data_dir": t["data_dir"], "trained": now}
        run.save()
        return StageResult("train", OK, summary=(
            f"{version} finished ({t['attempts']} window(s), {guard.rows} rows"
            + ("; Max-derived, private" if md else "") + ")"), details=details)
    if res.returncode == 0 and not res.stopped and said != "stopped":
        res = CommandResult(1, False, res.seconds, (
            f"exited 0 but {missing}" if said in ("", "done")
            else f"exited 0 with its {TRAINER_MANIFEST} saying {said!r}"))
    after = latest_checkpoint(run_dir)
    details["checkpoint"] = after.name if after else None
    if res.stopped or res.returncode == cfg.incomplete_exit_code \
            or (res.returncode == 0 and said == "stopped"):
        t["status"] = "incomplete"
        run.save()
        if after is None:
            run.issue(f"train {version} stopped for the window with no checkpoint saved; the "
                      "next window starts it over on the same frozen data")
        return StageResult("train", PAUSED, summary=f"{version} paused", reason=(
            f"stopped for the window at {after.name if after else 'no checkpoint'}; "
            "resumes next window"), details=details)
    t["failures"] = int(t.get("failures") or 0) + 1
    if t["failures"] >= cfg.max_failed_attempts:
        t["status"] = "abandoned"
        tail = f"abandoned after {t['failures']} failure(s); the operator decides"
    else:
        t["status"] = "incomplete"
        tail = (f"retried from the newest checkpoint next window "
                f"({t['failures']}/{cfg.max_failed_attempts} failures)")
    run.save()
    return StageResult("train", FAILED, reason=(
        f"the trainer exited {res.returncode} ({res.tail}); {tail}"), details=details)


# ---------------------------------------------------------------------------
# Quantize and score
# ---------------------------------------------------------------------------


def stage_quantize(run: Run) -> StageResult:
    cfg = run.cfg
    adapters = _dict(run.state.get("adapters"))
    quantized = run.state.setdefault("quantized", {}) if not run.dry_run \
        else dict(run.state.get("quantized") or {})
    # Every finished adapter, not only the newest train: one that finished while
    # no quantizer was configured is still quantized once one is.
    waiting = sorted(v for v in adapters if (quantized.get(v) or {}).get("status") != "done")
    if not waiting:
        t = run.state.get("train") if isinstance(run.state.get("train"), dict) else None
        why = (f" (train {t.get('adapter_version')} is {t.get('status')})"
               if t and t.get("status") != "complete" else "")
        return StageResult("quantize", SKIPPED,
                           reason=f"no finished adapter waits to be quantized{why}")
    version = waiting[-1]
    if not cfg.quantize_command:
        return StageResult("quantize", SKIPPED, reason=(
            f"flywheel.quantize.command is empty: {version} waits for a quantizer"))
    record = _dict(adapters.get(version))
    adapter_dir = str(record.get("adapter_dir") or "")
    pol = _policy_or_none(run.repo)
    if pol is None:
        return StageResult("quantize", FAILED, reason=(
            f"{POLICY_SCRIPT} cannot be loaded: {version} is not quantized without the "
            "Max-derived rule that names its models (DISTILL-OPEN-QUESTIONS Q6b)"))
    # The adapter's own record, else its provenance chain (artifact_policy):
    # a model merged from a Max-derived adapter, and calibrated on its rows,
    # is Max-derived; a record that does not say counts as Max-derived.
    max_derived = record.get("max_derived", True) is not False
    if not max_derived and adapter_dir and pol.provenance(adapter_dir).max_derived is True:
        max_derived = True
    if max_derived and not cfg.max_derived.complete:
        return StageResult("quantize", REFUSED, reason=(
            f"{version}'s adapter is Max-derived and flywheel.max_derived records no complete "
            "Q6b allow (approved, approval_ref, exclude_sessions_file): its quantize develops "
            "a model from Max-derived data, so it waits for the allow"))
    out_dir = run.data / "models" / pol.name_for(version, max_derived)
    argv = run.fill(cfg.quantize_command, {
        "adapter_dir": adapter_dir,
        "data_dir": str(record.get("data_dir") or ""),
        "out_dir": str(out_dir), "adapter_version": version})
    if max_derived and MAX_DERIVED_APPROVAL_FLAG not in argv:
        argv.append(MAX_DERIVED_APPROVAL_FLAG)
    if run.dry_run:
        return StageResult("quantize", DRY_RUN, summary=f"would run: {_quote(argv)}")
    note = ensure_gpu(run, "quantize")
    if note:
        return StageResult("quantize", SKIPPED, reason=note)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run.runner(run.command(argv, "quantize"))
    if res.returncode == 0 and not res.stopped:
        quantized[version] = {"status": "done", "out_dir": str(out_dir),
                              "max_derived": max_derived,
                              "finished": run.clock().isoformat(timespec="seconds")}
        run.save()
        return StageResult("quantize", OK, summary=(
            f"{version} quantized into {out_dir}"
            + (" (Max-derived: private, never published)" if max_derived else "")))
    if res.stopped or res.returncode == cfg.incomplete_exit_code:
        return StageResult("quantize", PAUSED, reason=(
            f"stopped for the window; {version} is quantized again next window"))
    return StageResult("quantize", FAILED, reason=(
        f"the quantizer exited {res.returncode} ({res.tail}); retried next window"))


def metric_points(summary: dict[str, Any]) -> dict[str, float]:
    """``set / metric[ @sub]`` -> points, from a score.py summary."""
    out: dict[str, float] = {}
    for sname, entry in _dict(summary.get("sets")).items():
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            continue
        for mname, val in metrics.items():
            if not isinstance(val, dict):
                continue
            if "rate" in val:
                if isinstance(val.get("rate"), (int, float)):
                    out[f"{sname} / {mname}"] = 100.0 * float(val["rate"])
                continue
            for sub, r in val.items():
                if isinstance(r, dict) and isinstance(r.get("rate"), (int, float)):
                    out[f"{sname} / {mname} @{sub}"] = 100.0 * float(r["rate"])
    return out


def wins_and_losses(previous: dict[str, Any], candidate: dict[str, Any]
                    ) -> dict[str, Any]:
    prev, cand = metric_points(previous), metric_points(candidate)
    wins, losses, same = [], [], []
    for key in sorted(prev.keys() & cand.keys()):
        delta = round(cand[key] - prev[key], 1)
        entry = {"metric": key, "previous": round(prev[key], 1), "candidate": round(cand[key], 1),
                 "delta": delta}
        if delta >= WIN_LOSS_PTS:
            wins.append(entry)
        elif delta <= -WIN_LOSS_PTS:
            losses.append(entry)
        else:
            same.append(entry)
    return {"wins": wins, "losses": losses, "unchanged": len(same),
            "only_previous": sorted(prev.keys() - cand.keys()),
            "only_candidate": sorted(cand.keys() - prev.keys())}


def _baseline(run: Run) -> tuple[str, Path] | None:
    """The base model's score.py summary: what the Phase 6 gate measures against."""
    if not run.cfg.baseline_summary:
        return None
    path = _resolve_under(run.cfg.baseline_summary, run.repo)
    return f"baseline ({path.name})", path


def _previous_summary(run: Run) -> tuple[str, Path] | None:
    """The last scored model's summary (else the baseline's): wins and losses are against it."""
    for model in reversed(run.state.get("models") or []):
        if isinstance(model, dict) and model.get("summary"):
            return str(model.get("adapter_version")), Path(str(model["summary"]))
    return _baseline(run)


def served_model_id(base_url: str, timeout: float = 10.0) -> str | None:
    """The model id the server reports first on ``GET /v1/models`` (the id
    score.py stamps on its rows), or None when it cannot be read."""
    import http.client
    import urllib.request

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models",
                                    timeout=timeout) as resp:
            body = json.loads(resp.read(1 << 20).decode("utf-8", errors="replace"))
    except (OSError, ValueError, http.client.HTTPException):
        return None
    items = body.get("data") if isinstance(body, dict) else None
    first = items[0] if isinstance(items, list) and items else None
    ident = first.get("id") if isinstance(first, dict) else None
    return str(ident) if ident else None


def _scored_cases(path: Path) -> int:
    """Cases a score.py summary scored (0 when it is missing or unreadable)."""
    try:
        summary = _read_json(path)
    except (OSError, ValueError):
        return 0
    sets = _dict(summary.get("sets")) if isinstance(summary, dict) else {}
    return sum(int(_num(_dict(e).get("scored"))) for e in sets.values())


def stage_score(run: Run) -> StageResult:
    cfg = run.cfg
    scored = {m.get("adapter_version") for m in run.state.get("models") or []
              if isinstance(m, dict)}
    quantized = run.state.get("quantized") or {}
    ready = sorted(v for v, q in quantized.items()
                   if isinstance(q, dict) and q.get("status") == "done" and v not in scored)
    if not ready:
        return StageResult("score", SKIPPED, reason="no new quantized model to score")
    candidate = ready[-1]
    if not cfg.score_base_url or not cfg.score_cases:
        return StageResult("score", SKIPPED, reason=(
            "flywheel.score.base_url and flywheel.score.cases must both be set to score "
            f"{candidate}"))
    cases = [_resolve_under(c, run.repo) for c in cfg.score_cases]
    missing = [str(c) for c in cases if not c.is_file()]
    if missing:
        return StageResult("score", FAILED, reason=f"eval case file(s) missing: {missing}")
    script = run.repo / SCORE_SCRIPT
    if not script.is_file():
        return StageResult("score", FAILED, reason=f"no scorer at {script}")
    scores = run.data / "scores"
    out = scores / f"{candidate}.jsonl"
    summary_path = scores / f"{candidate}.summary.json"
    summary_md = scores / f"{candidate}.summary.md"
    argv = [sys.executable, str(script), "run", "--base-url", cfg.score_base_url,
            "--label", candidate, "--out", str(out)]
    for c in cases:
        argv += ["--cases", str(c)]
    argv += ["--summary-out", str(summary_path), "--markdown-out", str(summary_md)]
    if run.dry_run:
        return StageResult("score", DRY_RUN, summary=f"would run: {_quote(argv)}")
    details: dict[str, Any] = {"candidate": candidate}
    # The cases may have changed since the adapter's data was frozen.
    adapters = _dict(run.state.get("adapters"))
    frozen = str(_dict(adapters.get(candidate)).get("data_dir") or "")
    if frozen:
        leaked = check_training_data([Path(frozen) / "train.jsonl", Path(frozen) / "val.jsonl"],
                                     cfg, None, score_case_keys(cfg, run.repo)).leaked_rows
        if leaked:
            return StageResult("score", REFUSED, reason=(
                f"{leaked} row(s) {candidate} was trained on share an id, task or group with the "
                "score cases, so its scores would not be held-out scores; nothing was scored"),
                details=details)
    note = ensure_gpu(run, "score")
    if note:
        return StageResult("score", SKIPPED, reason=note, details=details)
    # Nothing in the job loads the candidate into the server: score it only
    # while the server says it serves it, by exact version.
    known = set(quantized) | set(adapters)
    served = served_model_id(cfg.score_base_url)
    details["served_model"] = served
    if adapter_named(served or "", known) != candidate:
        what = f"model {served!r}" if served else "no model it names (GET /v1/models)"
        where = _dict(quantized.get(candidate)).get("out_dir") or "its models directory"
        return StageResult("score", REFUSED, reason=(
            f"the scoring server (flywheel.score.base_url) reports {what}, not {candidate}; "
            f"nothing was scored. Load {candidate} ({where}) there and the next window "
            "scores it"), details=details)
    scores.mkdir(parents=True, exist_ok=True)
    res = run.runner(run.command(argv, "score"))
    if res.stopped:
        return StageResult("score", PAUSED, reason=(
            f"stopped for the window; score.py resumes {out.name} next window"), details=details)
    if res.returncode == 3:
        done = _scored_cases(summary_path)
        if not done:
            return StageResult("score", FAILED, reason=(
                "score.py scored none of the cases (exit 3: every case errored or is missing); "
                f"is the scoring server up? ({res.tail})"), details=details)
        run.issue(f"score.py could not score every case of {candidate} ({done} scored, exit 3); "
                  "the rest are retried next window")
        return StageResult("score", PAUSED, reason=(
            f"some cases errored ({done} scored); score.py resumes {out.name} next window"),
            details=details)
    if res.returncode != 0:
        return StageResult("score", FAILED, reason=f"score.py exited {res.returncode} "
                                                   f"({res.tail})", details=details)
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        return StageResult("score", FAILED, reason=f"score.py wrote no summary at {summary_path}")
    details.update(summary=str(summary_path), served_model=summary.get("served_model"))
    answered = str(summary.get("served_model") or "")
    if adapter_named(answered, known) != candidate:
        # The server changed under the run. Set the rows aside, or score.py
        # refuses to resume a file answered by another model.
        tag = f"rejected-{run.clock():%Y%m%dT%H%M%S}"
        for path in (out, summary_path, summary_md):
            if path.exists():
                os.replace(path, path.with_name(f"{path.name}.{tag}"))
        return StageResult("score", REFUSED, reason=(
            f"score.py's rows were answered by {answered or 'more than one model'!r}, not "
            f"{candidate}; they were set aside (.{tag}) and the next window scores again"),
            details=details)
    verdict = "no gate"
    base = _baseline(run)
    if base and base[1].is_file():
        label, base_path = base
        cmp_md = scores / f"{candidate}.compare.md"
        res2 = run.runner(run.command(
            [sys.executable, str(script), "compare", "--base", str(base_path),
             "--candidate", str(summary_path), "--markdown-out", str(cmp_md)], "score-compare"))
        verdict = {0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE"}.get(
            res2.returncode, f"compare error (exit {res2.returncode})")
        details.update(gate_against=label, gate_against_summary=str(base_path),
                       verdict=verdict, compare_markdown=str(cmp_md))
    elif base:
        run.issue(f"the base model's summary {base[1]} is missing; the Phase 6 gate was not "
                  "evaluated")
    else:
        run.issue("no base model summary (flywheel.score.baseline_summary): the Phase 6 gate "
                  "was not evaluated")
    previous = _previous_summary(run)
    if previous and previous[1].is_file():
        details.update(previous=previous[0], previous_summary=str(previous[1]))
    elif previous and previous != base:
        run.issue(f"the previous model's summary {previous[1]} is missing; no wins and losses")
    run.state.setdefault("models", []).append({
        "adapter_version": candidate, "summary": str(summary_path), "verdict": verdict,
        "gate_against": details.get("gate_against"),
        "gate_against_summary": details.get("gate_against_summary"),
        "against": details.get("previous"), "against_summary": details.get("previous_summary"),
        "scored_in": run.run_id,
    })
    run.save()
    return StageResult("score", OK, summary=(
        f"{candidate} scored; Phase 6 gate vs {details.get('gate_against', 'nothing')}: "
        f"{verdict}; compared with {details.get('previous', 'nothing')} (promotion is the "
        "operator's call)"), details=details)


# ---------------------------------------------------------------------------
# GPU setup hooks (optional)
# ---------------------------------------------------------------------------


def ensure_gpu(run: Run, stage: str) -> str:
    """Apply ``gpu_setup`` once, before the first GPU stage that has work.

    Returns "" when *stage* may run, else the reason it may not. An empty
    ``gpu_setup`` keeps train and quantize from starting beside whatever the
    hosts serve (no swap, no dead-man restore) unless the operator opted into
    that with ``none``; score only queries ``score.base_url`` and needs no
    swap. The dead-man restore is armed BEFORE anything is swapped, and the
    state records the swap before it starts, so ``--recover`` knows to restore
    even if this process dies mid-apply. A swap starts only with time left for
    it and the stage, and is held to the stop time.
    """
    cfg = run.cfg
    if not cfg.gpu_setup:
        if stage == "score":
            return ""
        return (f"flywheel.gpu_setup is empty: {stage} would start GPU work beside whatever "
                "the hosts serve, with no dead-man restore. Set it to the setup to apply "
                f"(scripts/distill/setups.py), or to {NO_SWAP_SETUP!r} to run on the GPUs as "
                "they are")
    if cfg.gpu_setup == NO_SWAP_SETUP or run.gpu == "ready":
        return ""
    if run.gpu == "unavailable":
        return run.gpu_note

    def unavailable(note: str) -> str:
        run.gpu, run.gpu_note = "unavailable", note
        run.issue(note)
        return note

    need = MIN_STAGE_SECONDS[stage] + SETUP_APPLY_SECONDS
    if run.time_left() < need:
        return (f"{max(0, int(run.time_left() // 60))} min left before the stop time: too "
                f"little to apply {cfg.gpu_setup!r} and run {stage} (needs {need // 60} min); "
                "nothing was swapped")
    script = run.repo / SETUPS_SCRIPT
    if not script.is_file():
        return unavailable(f"gpu_setup {cfg.gpu_setup!r} is set but {script} does not exist; "
                           "GPU stages skipped")
    at = run.window_end + DEADMAN_AFTER_END
    res = run.runner(run.swap_command(
        [sys.executable, str(script), "deadman", RESTORE_SETUP, "--at",
         at.isoformat(timespec="minutes"), "--execute"]))
    if res.returncode != 0 or res.stopped:
        return unavailable(f"the dead-man restore could not be armed (setups.py deadman exited "
                           f"{res.returncode}); nothing was swapped, GPU stages skipped")
    run.state["gpu"] = {"applied": True, "setup": cfg.gpu_setup,
                        "since": run.clock().isoformat(timespec="seconds"),
                        "deadman_at": at.isoformat(timespec="minutes")}
    run.save()
    res = run.runner(run.swap_command([sys.executable, str(script), "apply", cfg.gpu_setup,
                                       "--execute"]))
    if res.stopped:
        return unavailable(f"setups.py apply {cfg.gpu_setup} had not finished at the stop time "
                           f"and was stopped; GPU stages skipped ({RESTORE_SETUP} is restored "
                           "after the run)")
    if res.returncode != 0:
        return unavailable(f"setups.py apply {cfg.gpu_setup} exited {res.returncode}; GPU "
                           f"stages skipped ({RESTORE_SETUP} is restored after the run)")
    run.gpu = "ready"
    return ""


def release_gpu(run: Run) -> None:
    """Restore ``business`` if a swap was recorded. Runs in the job's finally."""
    g = run.state.get("gpu")
    if not (isinstance(g, dict) and g.get("applied")):
        return
    script = run.repo / SETUPS_SCRIPT
    res = run.runner(run.command([sys.executable, str(script), "apply", RESTORE_SETUP,
                                  "--execute"], "gpu-restore", windowed=False))
    if res.returncode == 0:
        run.state["gpu"] = {"applied": False,
                            "restored": run.clock().isoformat(timespec="seconds")}
        run.save()
        return
    run.restore_failed = True
    run.issue(f"GPU RESTORE FAILED: setups.py apply {RESTORE_SETUP} exited {res.returncode} "
              f"({res.tail}); the dead-man timer restores it at {g.get('deadman_at')}")


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


STAGE_FUNCS: dict[str, Callable[[Run], StageResult]] = {
    "harvest": stage_harvest, "verify": stage_verify, "build": stage_build,
    "train": stage_train, "quantize": stage_quantize, "score": stage_score,
}


def _stage_from(raw: dict[str, Any], name: str) -> StageResult:
    return StageResult(name, str(raw.get("status")), _num(raw.get("seconds")),
                       str(raw.get("summary") or ""), "done earlier in this window",
                       _dict(raw.get("details")))


def _stage_record(raw: dict[str, Any], name: str) -> StageResult:
    """A recorded stage as it was recorded (its own reason), for a report rebuilt later."""
    status = str(raw.get("status") or "")
    known = (OK, SKIPPED, REFUSED, PAUSED, FAILED, DRY_RUN, NOT_RUN)
    return StageResult(name, status if status in known else FAILED, _num(raw.get("seconds")),
                       str(raw.get("summary") or ""),
                       str(raw.get("reason") or "") if status in known
                       else f"the record holds an unknown status {status!r}",
                       _dict(raw.get("details")))


def run_pipeline(run: Run) -> None:
    """Each stage in order, each held to the window; every outcome recorded."""
    earlier: dict[str, Any] = {}
    record = run.run_dir / "stages.json"
    if not run.dry_run:
        loaded = _read_json(record)
        earlier = loaded if isinstance(loaded, dict) else {}
    for name in STAGES:
        prior = earlier.get(name)
        if run.interrupted():
            result = StageResult(name, PAUSED, reason="the job was told to stop (SIGTERM) "
                                                      "before this stage")
        elif isinstance(prior, dict) and prior.get("status") == OK:
            result = _stage_from(prior, name)
        elif not run.dry_run and run.time_left() < MIN_STAGE_SECONDS[name]:
            left = max(0, int(run.time_left() // 60))
            result = StageResult(name, SKIPPED, reason=(
                f"{left} min left before the stop time "
                f"({run.stop_at.astimezone(ZoneInfo(run.cfg.window.tz)):%H:%M}); {name} needs "
                f"at least {MIN_STAGE_SECONDS[name] // 60} min"))
        else:
            # Timed on the run's clock, the one the window is held to.
            t0 = run.clock()
            try:
                result = STAGE_FUNCS[name](run)
            except Exception as exc:  # noqa: BLE001 — one stage's crash must not cost the report
                logger.exception("flywheel stage %s crashed", name)
                result = StageResult(name, FAILED,
                                     reason=_redact(f"{type(exc).__name__}: {exc}")[:500])
            result.seconds = round(max(0.0, (run.clock() - t0).total_seconds()), 1)
        run.results[name] = result
        logger.info("flywheel %s: %s %s", name, result.status, result.reason or result.summary)
        if not run.dry_run:
            _atomic_write(record, json.dumps({k: asdict(v) for k, v in run.results.items()},
                                             indent=1) + "\n")


def overall_status(run: Run) -> tuple[str, int]:
    statuses = {r.status for r in run.results.values()}
    if FAILED in statuses or run.restore_failed or run.crashed:
        return "FAIL", EXIT_FAILED
    if not run.cfg.enabled:
        return "OFF", EXIT_OK
    if REFUSED in statuses or run.issues:
        return "WARN", EXIT_OK
    return "OK", EXIT_OK


# ---------------------------------------------------------------------------
# Spend and the report
# ---------------------------------------------------------------------------


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_ts(value: Any) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def spend_summary(path: Path, since: datetime) -> dict[str, Any]:
    """Totals from the spend ledger (read only): the last week, all time, by phase."""
    out: dict[str, Any] = {"ledger": str(path), "present": path.is_file(),
                           "since": since.isoformat(timespec="seconds")}
    if not out["present"]:
        return out

    def blank() -> dict[str, Any]:
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "est_usd": 0.0, "quota": {}}

    def add(acc: dict[str, Any], row: dict[str, Any]) -> None:
        acc["calls"] += int(_num(row.get("calls")))
        acc["input_tokens"] += int(_num(row.get("input_tokens")))
        acc["output_tokens"] += int(_num(row.get("output_tokens")))
        acc["est_usd"] = round(acc["est_usd"] + _num(row.get("est_usd")), 4)
        unit = str(row.get("quota_unit") or "")
        if unit:
            acc["quota"][unit] = acc["quota"].get(unit, 0.0) + _num(row.get("quota_units"))

    week, total = blank(), blank()
    phases: dict[str, dict[str, Any]] = {}
    undated = 0
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                add(total, row)
                phase = str(row.get("phase") or "?")
                add(phases.setdefault(phase, blank()), row)
                ts = _parse_ts(row.get("ts"))
                if ts is None:
                    undated += 1
                elif ts >= since:
                    add(week, row)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        logger.warning("spend ledger %s unreadable: %s", path, exc)
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out.update(week=week, total=total, by_phase=phases, undated_rows=undated)
    return out


def _fmt_spend(acc: dict[str, Any]) -> str:
    quota = ", ".join(f"{v:g} {k}" for k, v in sorted(acc["quota"].items()))
    return (f"{acc['calls']} call(s), {acc['input_tokens']:,} in / {acc['output_tokens']:,} out "
            f"tokens, ${acc['est_usd']:.2f}" + (f", quota {quota}" if quota else ""))


def _this_weeks_model(run: Run) -> dict[str, Any] | None:
    for model in reversed(run.state.get("models") or []):
        if isinstance(model, dict) and model.get("scored_in") == run.run_id:
            return model
    return None


def _all_stages(run: Run) -> list[StageResult]:
    """Every stage, in order: one the run never reached is NOT RUN, saying so."""
    return [run.results.get(name) or StageResult(
        name, NOT_RUN, reason="the run stopped before this stage (see Issues)")
        for name in STAGES]


def _teachers_all_max(cfg: FlywheelConfig) -> bool:
    return bool(cfg.teacher_models) and all(is_max_name(p) for p in cfg.teacher_models)


def _one_line(text: Any, limit: int = NOTE_LIMIT) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3].rstrip() + "..."


def config_gaps(cfg: FlywheelConfig) -> list[str]:
    """The settings a train-to-score week still needs (empty: none missing)."""
    gaps = []
    if not cfg.build_args:
        gaps.append("build.args (this host's --pack, --tokenizer-dir, --index)")
    if not cfg.train_command:
        gaps.append("train.command")
    if not cfg.gpu_setup:
        gaps.append(f"gpu_setup (a setup name, or {NO_SWAP_SETUP!r})")
    if not cfg.quantize_command:
        gaps.append("quantize.command")
    if not cfg.score_base_url or not cfg.score_cases:
        gaps.append("score.base_url and score.cases")
    if not cfg.baseline_summary:
        gaps.append("score.baseline_summary")
    return gaps


def _report_after(run: Run) -> datetime:
    """The time the report's next window follows: the end of the run's own
    window once that window has begun (a run inside its window, or a refresh
    of it, is not "next"); now when the report's window is still ahead, or a
    refresh has no run to report."""
    now = run.clock()
    if run.window_start > now or (run.mode == "report-only" and not run.reported_run):
        return now
    return max(now, run.window_end)


def plan_next_steps(run: Run, model: dict[str, Any]) -> list[dict[str, str]]:
    """Who acts next, and on what: the operator's decisions first, then what
    the job itself does next window, then the runbook's plan, then the next
    window (always the last line)."""
    cfg = run.cfg
    steps: list[dict[str, str]] = []

    def add(by: str, text: str) -> None:
        line = _one_line(text)
        if line and all(s["step"] != line for s in steps):
            steps.append({"by": by, "step": line})

    t = _dict(run.state.get("train"))
    results = run.results
    if run.mode == "config-error":
        # cfg holds the defaults in place of the block (_fallback_config): none
        # of them is the operator's setting, so no step is drawn from them.
        add("operator", "Fix the flywheel: block in prometheus.yaml (every problem is listed under "
                        "Issues). The job refuses to run until it parses, and changes nothing "
                        "meanwhile.")
        for text in run.extra_steps:
            add("runbook", text)
        start, end = next_window(_report_after(run), cfg.window)
        add("job", (f"Next window: {describe_window(start, end)}; the job refuses to run then "
                    "too unless the block parses." if not run.facts.get("window_unknown") else
                    "Next window: unknown, flywheel.window is invalid (the default window would "
                    f"be {describe_window(start, end)})."))
        return steps
    if run.mode == "state-error":
        add("operator", f"Repair or move {run.state_path} by hand: it holds the train record, and "
                        "the job will not reset it.")
    if run.restore_failed:
        deadman = _dict(run.state.get("gpu")).get("deadman_at")
        add("operator", "Check the GPU hosts now: restoring the business setup failed (see Issues)"
                        + (f"; the dead-man timer restores it at {deadman}." if deadman else "."))
    for name in STAGES:
        r = results.get(name)
        if r is None or r.status != FAILED:
            continue
        if name == "train" and t.get("status") == "abandoned":
            add("operator", f"Train {t.get('adapter_version')} was abandoned after "
                            f"{t.get('failures')} failed window(s): fix the cause, then remove "
                            "'train' from state.json to start over.")
        elif run.mode == "recovery":
            add("operator", f"Find out why the run died during {name} (the job's journal); the "
                            "next window starts again.")
        else:
            add("operator", f"Look into the {name} failure (the reason is under What ran; a "
                            "child's output is in the data dir's logs/); the next window retries "
                            "it.")
    train = results.get("train")
    guard = _dict(train.details.get("guard")) if train else {}
    approved = cfg.max_derived.complete
    max_refusal = bool(guard.get("max_rows")) and not approved
    if not approved and (max_refusal or _teachers_all_max(cfg)):
        add("operator", "Record the Q6b allow to train on Max-derived rows (DISTILL-OPEN-QUESTIONS "
                        "#6: a private model only): flywheel.max_derived approved: true, "
                        "approval_ref and exclude_sessions_file. Until then no train uses a "
                        "Max-derived row.")
    if not approved and _teachers_all_max(cfg):
        add("operator", "flywheel.harvest.teacher_models names only Max models, so every verified "
                        "trace is Max-derived: add the local teacher's model id (its gym traces) "
                        "to train without Max data.")
    for name in STAGES:
        r = results.get(name)
        if r is None or r.status != REFUSED:
            continue
        refusals = [str(x) for x in guard.get("refusals") or []] if name == "train" else []
        if refusals and all("Max-derived" in x for x in refusals):
            continue            # the OPEN-QUESTIONS #6 step above says it
        add("operator", f"{name} was refused: {r.reason}")
    gaps = config_gaps(cfg)
    if gaps:
        add("operator", "Before the flywheel can train, quantize and score, set flywheel."
                        + ", flywheel.".join(gaps) + ".")
    if not cfg.enabled:
        add("operator", "The flywheel is off (flywheel.enabled: false), so a fired timer only "
                        "writes this report. To turn it on, install the units in deploy/systemd/ "
                        "(systemd --user) and set enabled: true.")

    if t.get("status") in ("running", "incomplete"):
        ckpt = latest_checkpoint(Path(str(t["run_dir"]))) if t.get("run_dir") else None
        add("job", f"Resume train {t.get('adapter_version')} from "
                   f"{ckpt.name if ckpt else 'its start (no checkpoint was saved)'} in the next "
                   "window, on the same frozen data.")
    elif t.get("status") != "abandoned":
        pending = _safe_pending(run)
        if pending is not None and pending < cfg.min_new_traces:
            add("job", f"Train waits for {cfg.min_new_traces - pending} more verified trace(s) "
                       f"({pending} of {cfg.min_new_traces} since the last train).")
    adapters = _dict(run.state.get("adapters"))
    quantized = _dict(run.state.get("quantized"))
    waiting = sorted(v for v in adapters if _dict(quantized.get(v)).get("status") != "done")
    if waiting and cfg.quantize_command:
        add("job", f"Quantize {waiting[-1]} in the next window.")
    scored = {m.get("adapter_version") for m in run.state.get("models") or []
              if isinstance(m, dict)}
    unscored = sorted(v for v, q in quantized.items()
                      if _dict(q).get("status") == "done" and v not in scored)
    if unscored:
        add("operator", f"Load {unscored[-1]} on the scoring server (flywheel.score.base_url); "
                        "the job scores it only while the server serves it.")
    cand = model.get("candidate")
    if cand:
        verdict = str(model.get("verdict") or "")
        if verdict == "PASS":
            add("operator", f"Decide whether to promote {cand}: it passed the Phase 6 gate vs "
                            f"{model.get('gate_against')}. Nothing was swapped; promotion is the "
                            "operator's call.")
        elif verdict in ("FAIL", "INCONCLUSIVE"):
            add("operator", f"{cand} did not pass the Phase 6 gate ({verdict}): the runbook allows "
                            "one adjusted run (e.g. 1 epoch or a lower learning rate), then stop "
                            "and report.")
        else:
            add("operator", f"{cand} was scored without a gate verdict ({verdict or 'none'}): set "
                            "flywheel.score.baseline_summary to the base model's summary.")
    for text in run.extra_steps:
        add("runbook", text)
    start, end = next_window(_report_after(run), cfg.window)
    add("job", f"Next window: {describe_window(start, end)}"
               + ("; the flywheel is off, so it only writes this report." if not cfg.enabled
                  else "."))
    return steps


def build_report(run: Run, status: str) -> dict[str, Any]:
    cfg = run.cfg
    tz = ZoneInfo(cfg.window.tz)
    model_section: dict[str, Any]
    model = _this_weeks_model(run)
    if model:
        model_section = {"candidate": model.get("adapter_version"), "verdict": model.get("verdict"),
                         "gate_against": model.get("gate_against"),
                         "previous": model.get("against")}
        try:
            cand = _read_json(Path(str(model["summary"])))
            prev = _read_json(Path(str(model["against_summary"]))) \
                if model.get("against_summary") else None
            if isinstance(cand, dict) and isinstance(prev, dict):
                model_section.update(wins_and_losses(prev, cand))
        except (OSError, ValueError) as exc:
            model_section["error"] = f"summaries unreadable: {exc}"
    else:
        why = [f"{name}: {r.status}" + (f" ({r.reason})" if r.reason else "")
               for name in ("train", "quantize", "score")
               if (r := run.results.get(name)) and r.status != OK]
        last = next((m for m in reversed(run.state.get("models") or []) if isinstance(m, dict)),
                    None)
        model_section = {"candidate": None, "why": why,
                         "last_scored": ({"adapter_version": last.get("adapter_version"),
                                          "verdict": last.get("verdict"),
                                          "scored_in": last.get("scored_in")} if last else None)}
    t = run.state.get("train") if isinstance(run.state.get("train"), dict) else {}
    train_result = run.results.get("train")
    guard = train_result.details.get("guard") if train_result else None
    harvest = run.results.get("harvest")
    verify = run.results.get("verify")
    now = run.clock()
    nxt_start, nxt_end = next_window(_report_after(run), cfg.window)
    since = run.spend_since or (run.window_start - timedelta(days=7))
    return {
        "schema": REPORT_SCHEMA, "kind": "weekly", "run_id": run.run_id, "mode": run.mode,
        "dry_run": run.dry_run, "status": status,
        "generated": now.astimezone(tz).isoformat(timespec="seconds"),
        "reported_run": run.reported_run or None,
        "window": {"describe": cfg.window.describe(), "timezone": cfg.window.tz,
                   "start": run.window_start.astimezone(tz).isoformat(timespec="minutes"),
                   "end": run.window_end.astimezone(tz).isoformat(timespec="minutes"),
                   "stop_at": run.stop_at.astimezone(tz).isoformat(timespec="minutes")},
        "next_window": {"describe": describe_window(nxt_start, nxt_end),
                        "start": nxt_start.isoformat(timespec="minutes"),
                        "end": nxt_end.isoformat(timespec="minutes")},
        "started": run.started.astimezone(tz).isoformat(timespec="seconds"),
        "finished": now.astimezone(tz).isoformat(timespec="seconds"),
        "stages": [asdict(s) for s in _all_stages(run)],
        "traces": {
            "harvest": harvest.details if harvest else {},
            "verify": verify.details if verify else {},
            "new_verified_since_last_train": _safe_pending(run),
            "min_new_traces": None if run.mode == "config-error" else cfg.min_new_traces,
            "dataset": dataset_share(run),
        },
        "train": {k: t.get(k) for k in ("adapter_version", "status", "attempts", "failures",
                                        "cutoff", "max_derived", "rows")} if t else None,
        "model": model_section,
        "spend": spend_summary(cfg.spend_ledger, since),
        "guards": {
            "max_derived": _max_derived_guard(run),
            "private_artifacts": private_artifacts(run),
            "train_data": guard if isinstance(guard, dict) else None,
            "offspring_dropped_at_verify": int(_num(verify.details.get("offspring_dropped", 0)))
            if verify else 0,
        },
        "issues": list(run.issues),
        "notes": [n for n in (_one_line(_redact(x)) for x in run.notes) if n],
        "next_steps": plan_next_steps(run, model_section),
    }


def _max_derived_guard(run: Run) -> dict[str, Any]:
    """The recorded Q6b allow; ``approved`` and ``complete`` are None
    (unknown) when the config is invalid and only the fallback defaults were
    read. The exclusion list is counted, never quoted."""
    a = run.cfg.max_derived
    if run.mode == "config-error":
        return {"approved": None, "approval_ref": "", "exclude_sessions_file": "",
                "sessions_listed": None, "complete": None}
    listed: int | None = None
    if a.exclude_sessions_file is not None:
        try:
            listed = read_session_list(a.exclude_sessions_file)
        except (OSError, ValueError, UnicodeDecodeError):
            listed = None
    return {"approved": a.approved, "approval_ref": a.approval_ref,
            "exclude_sessions_file": str(a.exclude_sessions_file or ""),
            "sessions_listed": listed, "complete": a.complete}


PRIVATE_RULE = ("Max-derived artifacts are private: never pushed to Hugging Face or anywhere "
                "public. publish_guard.py refuses them and the rental tool never uploads them "
                "(DISTILL-OPEN-QUESTIONS Q6b); nothing in this job publishes anything.")


def private_artifacts(run: Run) -> dict[str, Any]:
    """Every Max-derived artifact the state records, each private (never
    published): the newest sets (when they hold Max-derived rows, or nothing
    shows they do not), the current and any superseded train's frozen data
    and run directory, every Max-derived adapter and the models quantized
    from them. ``tagged``
    says whether the name carries ``max-derived`` (artifact_policy)."""
    out: dict[str, Any] = {"rule": PRIVATE_RULE, "known": run.mode not in UNREAD_MODES,
                           "items": []}
    if run.mode in UNREAD_MODES:
        return out
    pol = _policy_or_none(run.repo)
    items: list[dict[str, Any]] = []

    def add(kind: str, name: str, path: Any, why: str, *, named: bool = True) -> None:
        """*named*: the name must carry the tag (it is trained on, or holds in
        train/val, Max-derived rows); a set holding Max rows only in eval does
        not, and is private all the same."""
        p = str(path or "")
        if not p or any(i["path"] == p for i in items):
            return
        items.append({"kind": kind, "name": name, "path": p, "why": why,
                      "tagged": (bool(pol.path_has_tag(p)) if pol is not None and named
                                 else None)})

    sets = _dict(run.state.get("sets"))
    if sets.get("dir"):
        md = sets.get("max_derived")
        if md is True:
            add("dataset", f"sets of {sets.get('run')}", sets["dir"],
                "train/val hold Max-derived rows")
        elif sets.get("holds_max_rows") or md is None:
            add("dataset", f"sets of {sets.get('run')}", sets["dir"],
                "holds Max-derived rows (eval)" if sets.get("holds_max_rows")
                else "provenance unknown: private until shown clean", named=False)
    t = _dict(run.state.get("train"))
    if t and t.get("max_derived", True) is not False:
        add("train data", str(t.get("adapter_version")), t.get("data_dir"),
            "frozen Max-derived training rows")
        add("train run", str(t.get("adapter_version")), t.get("run_dir"),
            "checkpoints of a train on Max-derived rows")
    # A superseded train's frozen data and checkpoints stay on disk.
    for old in run.state.get("superseded_trains") or []:
        old = _dict(old)
        if old and old.get("max_derived", True) is not False:
            name = f"{old.get('adapter_version')} (superseded)"
            add("train data", name, old.get("data_dir"),
                "frozen Max-derived training rows of a superseded train")
            add("train run", name, old.get("run_dir"),
                "checkpoints of a superseded train on Max-derived rows")
    adapters = _dict(run.state.get("adapters"))
    for version, rec in sorted(adapters.items()):
        if _dict(rec).get("max_derived", True) is not False:
            add("adapter", version, _dict(rec).get("adapter_dir"), "trained on Max-derived rows")
    for version, q in sorted(_dict(run.state.get("quantized")).items()):
        rec = _dict(adapters.get(version))
        if _dict(q).get("max_derived", rec.get("max_derived", True)) is not False:
            add("model", version, _dict(q).get("out_dir"), "merged from a Max-derived adapter")
    out["items"] = items
    return out


def dataset_share(run: Run) -> dict[str, Any] | None:
    """The Max vs local-27B share: the newest sets' (build.py's
    ``data_policy.share``, per split) and the current train's rows per class
    (as the train guard counted them). None when the state was not read."""
    if run.mode in UNREAD_MODES:
        return None
    sets = _dict(run.state.get("sets"))
    t = _dict(run.state.get("train"))
    out: dict[str, Any] = {}
    if sets.get("dir"):
        out["sets"] = {"run": sets.get("run"), "max_derived": sets.get("max_derived"),
                       "approval": sets.get("approval"), "share": sets.get("share")}
    if t and isinstance(t.get("share"), dict):
        out["train"] = {"adapter_version": t.get("adapter_version"),
                        "max_derived": t.get("max_derived"), "rows": t.get("rows"),
                        "share": t.get("share")}
    return out or None


def minimal_report(run: Run, exc: BaseException) -> dict[str, Any]:
    """The report when :func:`build_report` (or rendering it) failed: built
    only from what cannot fail, so the reviewer still learns the window broke."""
    cfg = run.cfg
    tz = ZoneInfo(cfg.window.tz)
    now = run.clock()
    known = (OK, SKIPPED, REFUSED, PAUSED, FAILED, DRY_RUN, NOT_RUN)
    stages = []
    for name in STAGES:
        r = run.results.get(name)
        stages.append({"name": name,
                       "status": r.status if r is not None and r.status in known else NOT_RUN,
                       "seconds": max(0.0, _num(r.seconds)) if r is not None else 0.0,
                       "summary": _one_line(r.summary) if r is not None else "",
                       "reason": _one_line(r.reason) if r is not None
                       else "unknown: the report could not be built", "details": {}})
    nxt_start, nxt_end = next_window(_report_after(run), cfg.window)
    return {
        "schema": REPORT_SCHEMA, "kind": "weekly", "run_id": run.run_id,
        "mode": run.mode if run.mode in REPORT_MODES else "scheduled",
        "dry_run": run.dry_run, "status": "FAIL",
        "generated": now.astimezone(tz).isoformat(timespec="seconds"),
        "reported_run": run.reported_run or None,
        "window": {"describe": cfg.window.describe(), "timezone": cfg.window.tz,
                   "start": run.window_start.astimezone(tz).isoformat(timespec="minutes"),
                   "end": run.window_end.astimezone(tz).isoformat(timespec="minutes"),
                   "stop_at": run.stop_at.astimezone(tz).isoformat(timespec="minutes")},
        "next_window": {"describe": describe_window(nxt_start, nxt_end),
                        "start": nxt_start.isoformat(timespec="minutes"),
                        "end": nxt_end.isoformat(timespec="minutes")},
        "started": run.started.astimezone(tz).isoformat(timespec="seconds"),
        "finished": now.astimezone(tz).isoformat(timespec="seconds"),
        "stages": stages, "traces": {}, "train": None,
        "model": {"candidate": None, "why": [], "last_scored": None},
        "spend": {"ledger": str(cfg.spend_ledger), "present": False, "since": "",
                  "error": "not read: the report could not be built"},
        "guards": {"max_derived": _max_derived_guard(run),
                   "private_artifacts": {"rule": PRIVATE_RULE, "known": False, "items": []},
                   "train_data": None, "offspring_dropped_at_verify": 0},
        "issues": [*(str(i) for i in run.issues),
                   _one_line(_redact(f"the full report could not be built "
                                     f"({type(exc).__name__}: {exc}); this is a minimal one"))],
        "notes": [n for n in (_one_line(_redact(x)) for x in run.notes) if n],
        "next_steps": [
            {"by": "operator", "step": "Find out why the report could not be built (the job's "
                                       "journal holds the traceback)."},
            {"by": "job", "step": f"Next window: {describe_window(nxt_start, nxt_end)}."},
        ],
    }


def _safe_pending(run: Run) -> int | None:
    if run.mode in UNREAD_MODES:
        return None              # the train record (its cutoff) was not read
    try:
        return new_verified_since_last_train(run)
    except (OSError, ValueError) as exc:
        logger.warning("could not count verified traces: %s", exc)
        return None


# ---------------------------------------------------------------------------
# The reviewer's input: schema, Markdown, files
# ---------------------------------------------------------------------------


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportStage(_Closed):
    name: Literal["harvest", "verify", "build", "train", "quantize", "score"]
    status: Literal["OK", "SKIPPED", "REFUSED", "PAUSED", "FAILED", "DRY-RUN", "NOT RUN"]
    seconds: float = Field(ge=0)
    summary: str
    reason: str
    details: dict[str, Any]


class ReportWindow(_Closed):
    describe: str
    timezone: str
    start: str
    end: str
    stop_at: str


class ReportNextWindow(_Closed):
    describe: str
    start: str
    end: str


class ReportStep(_Closed):
    by: Literal["job", "operator", "runbook"]
    step: str = Field(min_length=1)


class ReportWinLoss(_Closed):
    metric: str
    previous: float
    candidate: float
    delta: float


class ReportModelSection(BaseModel):
    """``candidate`` is null for a week without a new model (then ``why``
    says what held it); with one, the wins and losses against the previous."""

    model_config = ConfigDict(extra="allow")
    candidate: str | None
    wins: list[ReportWinLoss] = Field(default_factory=list)
    losses: list[ReportWinLoss] = Field(default_factory=list)


class ReportSpend(BaseModel):
    model_config = ConfigDict(extra="allow")
    ledger: str
    present: bool
    since: str


class ReportPrivateItem(_Closed):
    kind: Literal["dataset", "train data", "train run", "adapter", "model"]
    name: str
    path: str
    why: str
    tagged: bool | None


class ReportPrivate(_Closed):
    """The Max-derived artifacts the state records: private, never published.
    ``known`` is false when the state was not read (the list is then empty)."""

    rule: str = Field(min_length=1)
    known: bool
    items: list[ReportPrivateItem]


class ReportGuards(_Closed):
    max_derived: dict[str, Any]
    private_artifacts: ReportPrivate
    train_data: dict[str, Any] | None
    offspring_dropped_at_verify: int = Field(ge=0)


class WeeklyReport(_Closed):
    """``LATEST.json``: the machine-readable twin of the reviewer's ``LATEST.md``."""

    report_schema: Literal["flywheel-weekly-report/3"] = Field(alias="schema")
    kind: Literal["weekly"]
    run_id: str = Field(min_length=1)
    mode: Literal["scheduled", "dry-run", "recovery", "config-error", "state-error",
                  "report-only"]
    dry_run: bool
    status: Literal["OK", "WARN", "FAIL", "OFF"]
    generated: str
    reported_run: str | None
    window: ReportWindow
    next_window: ReportNextWindow
    started: str
    finished: str
    stages: list[ReportStage]
    traces: dict[str, Any]
    train: dict[str, Any] | None
    model: ReportModelSection
    spend: ReportSpend
    guards: ReportGuards
    issues: list[str]
    notes: list[str]
    next_steps: list[ReportStep] = Field(min_length=1)

    @field_validator("stages")
    @classmethod
    def _every_stage_in_order(cls, stages: list[ReportStage]) -> list[ReportStage]:
        names = [s.name for s in stages]
        if names != list(STAGES):
            raise ValueError(f"stages must be exactly {list(STAGES)} in order, not {names}")
        return stages


def report_problems(report: dict[str, Any]) -> list[str]:
    """Where *report* departs from :class:`WeeklyReport` (empty: it does not)."""
    try:
        WeeklyReport.model_validate(report)
    except ValidationError as exc:
        return [f"{'.'.join(str(p) for p in e['loc']) or '(top)'}: {e['msg']}"
                for e in exc.errors()][:20]
    return []


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_seconds(seconds: float) -> str:
    s = int(round(seconds))
    if s < 120:
        return f"{s} s"
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def _local(iso: str, tz: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso).astimezone(ZoneInfo(tz))
    except (ValueError, ZoneInfoNotFoundError):
        return None


MODE_TEXT = {
    "scheduled": "scheduled run",
    "dry-run": "DRY RUN: nothing was changed",
    "recovery": "written by the recovery unit after the run died",
    "config-error": "config error: nothing ran",
    "state-error": "state unreadable: nothing ran",
    "report-only": "report only: no stage ran",
}
STATUS_TEXT = {
    "OK": "nothing failed and nothing was refused",
    "WARN": "something was refused or needs a look (see Issues)",
    "FAIL": "a stage failed, the run died, or a restore or the report failed",
    "OFF": "the flywheel is switched off",
}


def _unread(report: dict[str, Any]) -> str:
    """``unknown (why)`` when the run never read its state, else empty."""
    if report.get("mode") == "config-error":
        return "unknown (the config is invalid, so the state was not read)"
    if report.get("mode") == "state-error":
        return "unknown (the state is unreadable)"
    return ""


def _threshold(traces: dict[str, Any]) -> str:
    need = traces.get("min_new_traces")
    return (f"a train needs {need}" if need is not None
            else "the train threshold is unknown: the config is invalid")


def _glance(report: dict[str, Any]) -> list[str]:
    m = report["model"]
    if m.get("candidate"):
        missing = len(m.get("only_previous") or []) + len(m.get("only_candidate") or [])
        model = (f"{m['candidate']} scored; Phase 6 gate vs {m.get('gate_against') or 'nothing'}: "
                 f"{m.get('verdict')}; vs {m.get('previous') or 'nothing'}: "
                 f"{len(m.get('wins') or [])} win(s), {len(m.get('losses') or [])} loss(es)"
                 + (f", {missing} metric(s) not compared" if missing else ""))
    else:
        held = [f"{s['name']} {s['status']}" for s in report["stages"]
                if s["name"] in ("train", "quantize", "score") and s["status"] != OK]
        model = "no new model" + (f" ({', '.join(held)})" if held else "")
    t = report.get("train")
    train = (f"{t.get('adapter_version')} is {t.get('status')} (window(s) used "
             f"{t.get('attempts')}, failures {t.get('failures')})" if t
             else _unread(report) or "no train has started yet")
    tr = report["traces"]
    hv, vf = tr.get("harvest") or {}, tr.get("verify") or {}
    pending = tr.get("new_verified_since_last_train")
    traces = (f"{hv.get('new', 0)} new this window, {vf.get('verified', 0)} verified; "
              if hv or vf else "none harvested in this run; ")
    traces += (f"{'unknown' if pending is None else pending} verified since the last train "
               f"({_threshold(tr)})")
    sp = report["spend"]
    if sp.get("error"):
        spend = f"not read ({sp['error']})"
    elif not sp.get("present"):
        spend = "no ledger: no API spend recorded"
    else:
        spend = _fmt_spend(sp["week"])
    lines = [
        f"- Status: {report['status']} ({STATUS_TEXT.get(report['status'], '?')})",
        f"- Model: {model}",
        f"- Train: {train}",
        f"- Traces: {traces}",
    ]
    share = _glance_share(tr.get("dataset"))
    if share:
        lines.append(f"- Max vs local 27B: {share}")
    private = _dict(report["guards"].get("private_artifacts"))
    if private.get("known"):
        n = len(private.get("items") or [])
        lines.append(f"- Max-derived artifacts: {n}, all private (never published)" if n
                     else "- Max-derived artifacts: none recorded")
    lines += [
        f"- Spend, last 7 days: {spend}",
        f"- Issues: {len(report['issues'])}; operator steps: "
        f"{sum(1 for s in report['next_steps'] if s['by'] == 'operator')}",
        f"- Next window: {report['next_window']['describe']}",
    ]
    return lines


SHARE_NAMES = {"max": "Max", GYM_SOURCE: "local 27B", "other": "other"}


def _share_text(entry: Any) -> str:
    """One split of build.py's share table (``{"rows": n, "<class>": {"rows",
    "pct"}}``) or a plain ``{"<class>": rows}`` count, as one line."""
    e = _dict(entry)
    counts = {c: (int(_num(_dict(e[c]).get("rows"))) if isinstance(e.get(c), dict)
                  else int(_num(e.get(c)))) for c in SHARE_CLASSES}
    total = int(_num(e.get("rows"))) if "rows" in e else sum(counts.values())
    if not total:
        return ""
    return ", ".join(f"{SHARE_NAMES[c]} {counts[c]} ({100.0 * counts[c] / total:.1f}%)"
                     for c in SHARE_CLASSES) + f" of {total}"


def _glance_share(dataset: Any) -> str:
    d = _dict(dataset)
    sets = _dict(d.get("sets"))
    train_split = _dict(_dict(_dict(sets.get("share")).get("by_split")).get("train"))
    text = _share_text(train_split)
    if text:
        return f"train split of the {sets.get('run')} sets: {text}"
    t = _dict(d.get("train"))
    text = _share_text(t.get("share"))
    return f"{t.get('adapter_version')} trains on {text}" if text else ""


def render_markdown(report: dict[str, Any]) -> str:
    """LATEST.md: the title, the status line, then :data:`REVIEWER_SECTIONS` in order."""
    w = report["window"]
    tz = w.get("timezone") or "UTC"
    mode = MODE_TEXT.get(report["mode"], report["mode"])
    if report["mode"] == "report-only":
        mode += (f"; the {report['reported_run']} run's record" if report.get("reported_run")
                 else "; no run recorded yet")
    lines = [
        f"# Flywheel weekly report: {report['run_id']}",
        "",
        f"**FLYWHEEL weekly · {report['status']}** · {mode} · window {w['describe']}",
        "",
    ]
    gen = _local(report["generated"], tz)
    if report["mode"] == "report-only":
        lines.append(f"Refreshed {gen:%a %Y-%m-%d %H:%M %Z}." if gen else "Refreshed.")
    else:
        s, f = _local(report["started"], tz), _local(report["finished"], tz)
        if s and f:
            lines.append(f"Ran {s:%a %Y-%m-%d %H:%M}-{f:%H:%M %Z}.")
    lines += [f"Machine-readable twin: `LATEST.json` (`{report['schema']}`), generated "
              f"{report['generated']}.", ""]

    lines += ["## At a glance", "", *_glance(report)]

    lines += ["", "## What ran", "", "| Stage | Result | Time | Summary |", "|---|---|---|---|"]
    for s in report["stages"]:
        if s["status"] == NOT_RUN:
            text, took = "not run (see below)", "-"
        else:
            text, took = s["summary"] or s["reason"], _fmt_seconds(s["seconds"])
        lines.append(f"| {s['name']} | {s['status']} | {took} | {_md_escape(text)} |")

    not_run = [s for s in report["stages"] if s["status"] not in (OK, DRY_RUN)]
    lines += ["", "## Why stages did not run", ""]
    if not_run:
        lines += [f"- **{s['name']}** {s['status']}: {_md_escape(s['reason'] or s['summary'])}"
                  for s in not_run]
    else:
        lines.append("Every stage ran.")

    tr = report["traces"]
    hv, vf = tr.get("harvest") or {}, tr.get("verify") or {}
    lines += ["", "## Traces", ""]
    if hv:
        origins = ", ".join(f"{k} {v}" for k, v in sorted((hv.get("by_origin") or {}).items()))
        lines.append(f"- New this window: {hv.get('new', 0)} ({origins or 'none'}); "
                     f"already harvested {hv.get('already_harvested', 0)}; "
                     f"unreadable {hv.get('unreadable', 0)}")
        by_class = _dict(hv.get("by_class"))
        if by_class:
            lines.append("- New by class: " + ", ".join(
                f"{SHARE_NAMES[c]} {int(_num(by_class.get(c)))}" for c in SHARE_CLASSES))
        marks = _dict(hv.get("watermarks"))
        if marks:
            rowid = hv.get("golden_export_rowid")
            was = hv.get("golden_export_rowid_before")
            lines.append(
                f"- Source files: {marks.get('new', 0)} new, {marks.get('grown', 0)} grown "
                f"(read from the watermark), {marks.get('unchanged', 0)} unchanged, "
                f"{marks.get('rewritten', 0)} rewritten (read whole again)"
                + (f"; golden exports through rowid {rowid}"
                   + (f" (was {was})" if was and was != rowid else "") if rowid else ""))
    if vf:
        dropped = ", ".join(f"{k} {v}" for k, v in sorted((vf.get("dropped") or {}).items()))
        lines.append(f"- Verified: {vf.get('verified', 0)}; dropped: {dropped or 'none'}")
        vclass = _dict(vf.get("verified_by_class"))
        if any(int(_num(v)) for v in vclass.values()):
            lines.append("- Verified by class: " + ", ".join(
                f"{SHARE_NAMES[c]} {int(_num(vclass.get(c)))}" for c in SHARE_CLASSES)
                + (f"; {vf['payg_max_judge']} passed by the pay-per-token Max judge"
                   if vf.get("payg_max_judge") else ""))
    if not hv and not vf:
        lines.append("- Nothing was harvested or verified in this run.")
    pending = tr.get("new_verified_since_last_train")
    lines.append(f"- New verified since the last train: "
                 f"{'unknown' if pending is None else pending} ({_threshold(tr)})")
    ds = _dict(tr.get("dataset"))
    sets = _dict(ds.get("sets"))
    by_split = _dict(_dict(sets.get("share")).get("by_split"))
    if by_split:
        lines.append(f"- Max vs local 27B in the {sets.get('run')} sets"
                     + (f" (Q6b approval {sets['approval']!r})" if sets.get("approval") else "")
                     + ":")
        lines += [f"  - {split}: {_share_text(entry) or 'no rows'}"
                  for split, entry in by_split.items()]
    elif sets:
        lines.append(f"- Max vs local 27B in the {sets.get('run')} sets: the builder's report "
                     "has no share table")
    tshare = _share_text(_dict(ds.get("train")).get("share"))
    if tshare:
        lines.append(f"- Train {_dict(ds.get('train')).get('adapter_version')} rows: {tshare}")
    by_source = hv.get("by_source") or []
    if by_source:
        lines += ["", "| Origin | Source model | Adapter | New |", "|---|---|---|---|"]
        lines += [f"| {r['origin']} | {_md_escape(r['source_model'])} | "
                  f"{_md_escape(r['adapter_version'])} | {r['new']} |" for r in by_source]
    by_tag = hv.get("by_source_tag") or []
    if by_tag:
        lines += ["", "| Kind | Source | Class | New |", "|---|---|---|---|"]
        lines += [f"| {r['kind']} | {_md_escape(str(r['source']))} | "
                  f"{SHARE_NAMES.get(r['class'], r['class'])} | {r['new']} |" for r in by_tag]

    lines += ["", "## Model: wins and losses vs the previous model", ""]
    m = report["model"]
    if m.get("candidate"):
        lines.append(f"Candidate **{m['candidate']}**: Phase 6 gate vs "
                     f"{m.get('gate_against') or 'nothing'}: **{m.get('verdict')}**. Promotion "
                     "is the operator's call; nothing was swapped.")
        lines.append(f"\nCompared with {m.get('previous') or 'nothing'} (a comparison, not "
                     "the gate):")
        for title, key in (("Wins", "wins"), ("Losses", "losses")):
            rows = m.get(key) or []
            lines += ["", f"{title} ({len(rows)}):"]
            lines += [f"- {r['metric']}: {r['previous']} -> {r['candidate']} "
                      f"({r['delta']:+.1f} pts)" for r in rows] or ["- none"]
        lines.append(f"\nUnchanged within {WIN_LOSS_PTS:g} pt: {m.get('unchanged', 0)} metric(s).")
        # A metric only one side scored is neither a win nor a loss: say which.
        for key, side in (("only_previous", f"{m.get('previous') or 'the previous model'} only, "
                                            "the candidate has no score"),
                          ("only_candidate", "the candidate only, "
                                             f"{m.get('previous') or 'the previous model'} has "
                                             "no score")):
            if m.get(key):
                lines.append(f"\nNot compared ({side}): "
                             + ", ".join(_md_escape(str(x)) for x in m[key]) + ".")
        if m.get("error"):
            lines.append(f"\n{m['error']}")
    else:
        held = [f"{s['name']} {s['status']}" for s in report["stages"]
                if s["name"] in ("train", "quantize", "score") and s["status"] != OK]
        lines.append("No new model this week"
                     + (f": {', '.join(held)} (reasons under Why stages did not run)."
                        if held else "."))
        last = m.get("last_scored")
        if last:
            lines.append(f"- Last scored: {last['adapter_version']} ({last['verdict']}, "
                         f"{last['scored_in']})")
        elif _unread(report):
            lines.append(f"- Last scored: {_unread(report)}.")
        else:
            lines.append("- No model has been scored yet.")
    t = report.get("train")
    if t:
        lines.append(f"\nTrain record: {t.get('adapter_version')} is {t.get('status')} "
                     f"(attempts {t.get('attempts')}, failures {t.get('failures')}).")

    sp = report["spend"]
    lines += ["", "## Spend", ""]
    if sp.get("error"):
        lines.append(f"Spend ledger not read ({sp['error']}).")
    elif not sp.get("present"):
        lines.append(f"No spend ledger at {sp['ledger']}: no API spend recorded.")
    else:
        lines.append(f"- Last 7 days (since {sp['since']}): {_fmt_spend(sp['week'])}")
        lines.append(f"- All time: {_fmt_spend(sp['total'])}")
        for phase, acc in sorted(sp["by_phase"].items()):
            lines.append(f"  - {phase}: {_fmt_spend(acc)}")
        if sp.get("undated_rows"):
            lines.append(f"- {sp['undated_rows']} ledger row(s) have no readable timestamp "
                         "(counted in all time only)")

    g = report["guards"]
    lines += ["", "## Guards", ""]
    md = g["max_derived"]
    if md.get("complete") is None:
        lines.append("- Max-derived data: unknown (the config is invalid, so its max_derived "
                     "allow was not read)")
    elif md.get("complete"):
        listed = md.get("sessions_listed")
        lines.append(
            f"- Max-derived data: training ALLOWED under the Q6b allow (approval "
            f"{md.get('approval_ref')!r}; "
            + (f"{listed} session(s) excluded via {md.get('exclude_sessions_file')}"
               if listed else f"the session exclusion list {md.get('exclude_sessions_file')} is "
                              "unusable, so every Max-derived train is refused")
            + "). Everything trained on it is tagged max-derived and private")
    else:
        lines.append("- Max-derived data: training refused (no operator approval recorded; "
                     "DISTILL-OPEN-QUESTIONS #6): the Q6b ruling allows it for a private model "
                     "under flywheel.max_derived (approved, approval_ref, exclude_sessions_file)")
    td = g.get("train_data")
    if td:
        lines.append(f"- Train data checked: {td['rows']} row(s), {td['max_rows']} Max-derived, "
                     f"{td['offspring_rows']} offspring ({td['offspring_unverified']} unverified), "
                     f"{td.get('leaked_rows', 0)} sharing a task with the score cases")
    else:
        lines.append("- Train data: not checked in this run (no train was started or refused)")
    lines.append(f"- Offspring traces dropped as unverified this window: "
                 f"{g.get('offspring_dropped_at_verify', 0)}")
    priv = _dict(g.get("private_artifacts"))
    lines.append(f"- Private: {priv.get('rule') or PRIVATE_RULE}")
    if not priv.get("known"):
        lines.append("  - What the state holds of them: unknown (the state was not read)")
    elif not priv.get("items"):
        lines.append("  - None recorded.")
    for item in priv.get("items") or []:
        untagged = " (its name does NOT carry the max-derived tag)" \
            if item.get("tagged") is False else ""
        lines.append(f"  - {item['kind']} {_md_escape(str(item['name']))}: "
                     f"{_md_escape(str(item['path']))} ({item['why']}){untagged}")

    lines += ["", "## Issues", ""]
    lines += [f"- {_md_escape(i)}" for i in report["issues"]] or ["None."]

    lines += ["", "## Operator notes", ""]
    lines += [f"- {_md_escape(n)}" for n in report.get("notes") or []] or ["None."]

    lines += ["", "## Next steps", ""]
    lines += [f"- [{s['by']}] {_md_escape(s['step'])}" for s in report["next_steps"]]
    return "\n".join(lines) + "\n"


def markdown_sections(text: str) -> list[str]:
    """The ``## `` headings of a rendered report, in order."""
    return [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]


def _home_relative(value: Any, home: re.Pattern[str]) -> Any:
    if isinstance(value, str):
        return home.sub("~", value)
    if isinstance(value, dict):
        return {k: _home_relative(v, home) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_home_relative(v, home) for v in value]
    return value


def public_paths(report: dict[str, Any]) -> dict[str, Any]:
    """*report* with the home directory written as ``~`` in every string.

    The report is the reviewer's input and may be copied into the repo; an
    absolute home path names the account. Only a whole path component
    matches (a home of ``/srv/op`` is not taken out of ``/srv/opx``).
    """
    home = os.path.expanduser("~").rstrip("/")
    if not home or home == "~":
        return report
    out = _home_relative(report, re.compile(re.escape(home) + r"(?![A-Za-z0-9_-])"))
    return out if isinstance(out, dict) else report


def write_report(data: Path, report: dict[str, Any], *, stem: str | None = None) -> Path:
    """The dated copy first, then LATEST.json, then LATEST.md, each by atomic rename.

    A report that departs from :class:`WeeklyReport` is still written, marked
    FAIL with the departures under Issues (a schema slip must not cost the
    reviewer the week's report).
    """
    report = public_paths(report)
    problems = report_problems(report)
    if problems:
        logger.error("the flywheel report departs from its schema: %s", problems)
        report = {**report, "status": "FAIL",
                  "issues": [*(report.get("issues") or []),
                             *(f"report schema: {p}" for p in problems)]}
    rdir = data / "reports" / ("dry-run" if report.get("dry_run") else "")
    text = render_markdown(report)
    blob = json.dumps(report, indent=1, default=str) + "\n"
    stem = stem or f"{report['run_id']}-weekly"
    _atomic_write(rdir / f"{stem}.json", blob)
    _atomic_write(rdir / f"{stem}.md", text)
    _atomic_write(rdir / "LATEST.json", blob)
    _atomic_write(rdir / "LATEST.md", text)
    return rdir / "LATEST.md"


def emit_report(run: Run, status: str, *, stem: str | None = None) -> bool:
    """Write the run's report, or a minimal one when it cannot be built.

    True when the full report was written; False when only the minimal one
    (or nothing) was, which the exit code carries.
    """
    try:
        path = write_report(run.data, build_report(run, status), stem=stem)
        logger.info("flywheel report (%s): %s", status, path)
        return True
    except Exception as exc:  # noqa: BLE001 — the reviewer still gets a report
        logger.exception("could not build or write the flywheel report")
        failure: BaseException = exc
    try:
        write_report(run.data, minimal_report(run, failure), stem=stem)
        logger.info("a minimal flywheel report was written in its place")
    except Exception:  # noqa: BLE001 — logged; the exit code carries it
        logger.exception("could not write even the minimal flywheel report")
    return False


# ---------------------------------------------------------------------------
# Report only: the reviewer's input from the recorded state
# ---------------------------------------------------------------------------


_DATED_REPORT = re.compile(r"^(\d{4}-\d{2}-\d{2})-weekly\.json$")


def last_recorded_run(data: Path, state: dict[str, Any]) -> str:
    """The newest run with a stage record, a dated report of its own, or the
    state's current run ("" = none).

    A run refused for a bad config or an unreadable state writes only its
    dated report (``<run>-weekly.json``; a refresh's is ``-report-only``), so
    the report counts as the record of that window.
    """
    ids = {p.parent.name for p in (data / "runs").glob("*/stages.json")}
    ids |= {m.group(1) for p in (data / "reports").glob("*-weekly.json")
            if (m := _DATED_REPORT.match(p.name))}
    cur = _dict(state.get("current_run")).get("id")
    if isinstance(cur, str) and cur:
        ids.add(cur)
    return max(ids, default="")


def stage_blockers(run: Run) -> dict[str, str]:
    """What holds each stage, read from the config and the state alone (no
    stage runs): the report-only reasons when no run has been recorded."""
    cfg = run.cfg
    held: dict[str, list[str]] = {
        name: ([] if cfg.enabled else ["flywheel.enabled is false"]) for name in STAGES}
    if not _expand(cfg.golden) and not _expand(cfg.extra):
        held["harvest"].append("no harvest source matches a file (flywheel.harvest.golden / extra)")
        held["verify"].append("nothing is harvested to verify")
    if not cfg.build_args:
        held["build"].append("flywheel.build.args is empty (the builder needs this host's --pack "
                             "and --tokenizer-dir)")
    if resolve_build_script(cfg, run.repo) is None:
        held["build"].append(f"no dataset builder at {cfg.build_script or ' or '.join(BUILD_SCRIPTS)}")
    t = _dict(run.state.get("train"))
    if t.get("status") in ("running", "incomplete"):
        held["train"].append(f"train {t.get('adapter_version')} waits to resume")
    elif t.get("status") == "abandoned":
        held["train"].append(f"train {t.get('adapter_version')} was abandoned; the operator decides")
    else:
        pending = _safe_pending(run)
        if pending is not None and pending < cfg.min_new_traces:
            held["train"].append(f"{pending} new verified trace(s) since the last train; a train "
                                 f"needs {cfg.min_new_traces} (flywheel.train.min_new_traces)")
    if not cfg.train_command:
        held["train"].append("flywheel.train.command is empty")
    if not cfg.max_derived.complete and _teachers_all_max(cfg):
        held["train"].append("every teacher model in flywheel.harvest.teacher_models names Max and "
                             "no Q6b allow is recorded in flywheel.max_derived "
                             "(DISTILL-OPEN-QUESTIONS #6)")
    adapters = _dict(run.state.get("adapters"))
    quantized = _dict(run.state.get("quantized"))
    if all(_dict(quantized.get(v)).get("status") == "done" for v in adapters):
        held["quantize"].append("no finished adapter waits to be quantized")
    if not cfg.quantize_command:
        held["quantize"].append("flywheel.quantize.command is empty")
    if not cfg.gpu_setup:
        for name in ("train", "quantize"):
            held[name].append("flywheel.gpu_setup is empty")
    scored = {m.get("adapter_version") for m in run.state.get("models") or []
              if isinstance(m, dict)}
    if not any(_dict(q).get("status") == "done" and v not in scored
               for v, q in quantized.items()):
        held["score"].append("no quantized model waits to be scored")
    if not cfg.score_base_url or not cfg.score_cases:
        held["score"].append("flywheel.score.base_url and flywheel.score.cases are not both set")
    return {name: "; ".join(why) or "nothing holds it: the next window runs it"
            for name, why in held.items()}


_SEVERITY = {"OK": 0, "OFF": 0, "WARN": 1, "FAIL": 2}


def report_only(cfg: FlywheelConfig, *, clock: Clock, runner: Runner, repo: Path,
                notes: list[str] | None = None, steps: list[str] | None = None) -> int:
    """Rewrite the reviewer's input from the recorded state. Runs no stage.

    It re-renders the newest recorded run (its stages, reasons and model) with
    the current state, spend and next steps, or, when no run is recorded,
    says what holds each stage. It never changes the state, never starts a
    child process, and takes the run lock first: while a run holds it, that
    run's own report is the current one and this writes nothing. LATEST.* are
    replaced; the run's own dated copy is not (the refresh is dated
    ``<today>-weekly-report-only``).
    """
    lock = _Lock(cfg.data_dir / "flywheel.lock")
    if not lock.acquire():
        logger.error("a flywheel run holds %s and writes its own report; nothing done", lock.path)
        return EXIT_BUSY
    try:
        return _report_only(cfg, clock=clock, runner=runner, repo=repo,
                            notes=list(notes or []), steps=list(steps or []))
    finally:
        lock.release()


def _report_only(cfg: FlywheelConfig, *, clock: Clock, runner: Runner, repo: Path,
                 notes: list[str], steps: list[str]) -> int:
    now = clock()
    tz = ZoneInfo(cfg.window.tz)
    today = now.astimezone(tz).date().isoformat()
    stem = f"{today}-weekly-report-only"
    since = (now - timedelta(days=7)).astimezone(tz)
    try:
        state = load_state(cfg.data_dir / "state.json")
    except FlywheelStateError as exc:
        start, end = next_window(now, cfg.window)
        run = Run(cfg=cfg, run_id=today, window_start=start, window_end=end, dry_run=False,
                  clock=clock, runner=runner, repo=repo, state={}, mode="state-error",
                  started=now, notes=notes, extra_steps=steps, spend_since=since)
        run.issue(str(exc))
        for name in STAGES:
            run.results[name] = StageResult(name, NOT_RUN, reason="the state is unreadable "
                                                                  "(see Issues)")
        emit_report(run, "FAIL", stem=stem)
        return EXIT_FAILED
    reported = last_recorded_run(cfg.data_dir, state)
    bounds: tuple[datetime, datetime] | None = None
    if reported:
        try:
            bounds = window_on(date.fromisoformat(reported), cfg.window)
        except ValueError:
            bounds = None
    start, end = bounds or next_window(now, cfg.window)
    cur = _dict(state.get("current_run"))
    # The run's own issues, status and (for a run that recorded no stages:
    # flywheel off, config or state refused) stage reasons live in its own
    # dated report.
    own: Any = None
    if reported:
        try:
            own = _read_json(cfg.data_dir / "reports" / f"{reported}-weekly.json")
        except (OSError, ValueError):
            own = None
    own = own if isinstance(own, dict) else {}
    began = (_parse_ts(cur.get("started")) if reported and cur.get("id") == reported
             else _parse_ts(own.get("started")) if own.get("started") else None)
    run = Run(cfg=cfg, run_id=reported or today, window_start=start, window_end=end,
              dry_run=False, clock=clock, runner=runner, repo=repo, state=state,
              mode="report-only", started=began or now, notes=notes, extra_steps=steps,
              reported_run=reported, spend_since=since)
    carried = "OK"
    if reported:
        try:
            loaded = _read_json(run.run_dir / "stages.json")
        except (OSError, ValueError) as exc:
            run.issue(f"the {reported} run's stage record is unreadable ({type(exc).__name__})")
            loaded = None
        recorded = loaded if isinstance(loaded, dict) else {}
        died = cur.get("id") == reported and cur.get("status") == "running"
        own_stages = {s.get("name"): s for s in own.get("stages") or [] if isinstance(s, dict)}
        blockers: dict[str, str] = {}
        for name in STAGES:
            raw, mine = recorded.get(name), own_stages.get(name)
            if isinstance(raw, dict):
                result = _stage_record(raw, name)
            elif isinstance(mine, dict) and not died:
                result = _stage_record(mine, name)
            elif recorded or died:
                result = StageResult(name, NOT_RUN, reason=f"the {reported} run did not reach it")
            else:
                blockers = blockers or stage_blockers(run)
                result = StageResult(name, NOT_RUN, reason=(
                    f"the {reported} run left no stage record; {blockers[name]}"))
            run.results[name] = result
        for item in own.get("issues") or []:
            run.issue(f"({reported} run) {item}")
        if own.get("status") in _SEVERITY:
            carried = str(own["status"])
        if died:
            run.issue(f"The {reported} run is recorded as running but no process holds the run "
                      "lock: it died. The recovery unit (--recover) restores any GPU swap and "
                      "finishes its report.")
            carried = "FAIL"
    else:
        for name, why in stage_blockers(run).items():
            run.results[name] = StageResult(name, NOT_RUN,
                                            reason=f"no run has been recorded yet; {why}")
    last = last_window_start(now, cfg.window)
    _, last_end = window_on(last.date(), cfg.window)
    if cfg.enabled and now >= last_end and reported < last.date().isoformat():
        run.issue(f"No run is recorded for the {last.date()} window"
                  + (f" (the newest is {reported})" if reported else "")
                  + ": the timer did not fire (units not installed, or the host was off), or "
                    "the run never started.")
    status, _ = overall_status(run)
    if _SEVERITY[carried] > _SEVERITY.get(status, 0):
        status = carried
    return EXIT_OK if emit_report(run, status, stem=stem) else EXIT_FAILED


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_config(path: str | None) -> tuple[dict[str, Any], list[str]]:
    from prometheus.config.load import ConfigReadError, load_config_resolved

    try:
        raw = load_config_resolved(path, subsystem="flywheel",
                                   substituting="built-in defaults", strict=True,
                                   strict_action="run the flywheel")
    except ConfigReadError as exc:
        return {}, [str(exc)]
    return (raw if isinstance(raw, dict) else {}), []


class _Lock:
    """One run at a time: a POSIX record lock on ``flywheel.lock``, held until exit.

    The kernel drops it when the process dies, so a killed run never leaves a
    stale lock for the next window to trip on.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.fh: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.lockf(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            self.fh.close()
            self.fh = None
            return False
        return True

    def release(self) -> None:
        if self.fh is not None:
            fcntl.lockf(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


def _finish(run: Run) -> int:
    """Restore the GPU, write the report, close the run record. The finally of every run."""
    if run.interrupted():
        run.issue("the job was sent SIGTERM (systemd's outer wall, or a manual stop) and "
                  "stopped early; stages it did not reach resume next window")
    if not run.dry_run:
        try:
            release_gpu(run)
        except Exception as exc:  # noqa: BLE001 — the report must still be written
            logger.exception("GPU restore crashed")
            run.restore_failed = True
            run.issue(f"GPU RESTORE CRASHED: {type(exc).__name__}: {exc}")
    status, code = overall_status(run)
    if not emit_report(run, status):
        code = EXIT_FAILED
    if not run.dry_run:
        cur = run.state.get("current_run")
        if isinstance(cur, dict) and cur.get("id") == run.run_id:
            cur.update(status="done", result=status)
        try:
            run.save()
        except OSError:
            logger.exception("could not save the flywheel state")
            code = EXIT_FAILED
    return code


def recover(cfg: FlywheelConfig, *, clock: Clock, runner: Runner, repo: Path) -> int:
    """The OnFailure= unit: restore a recorded GPU swap; finish a run that died.

    Deliberately small. It does not retry stages; it puts the GPUs back and
    makes sure the reviewer's input says the window failed. It takes the run
    lock first: a weekly start that found the lock held exits non-zero, which
    starts this unit while the run holding it is alive, mid-train, and restores
    its own swap when it ends.
    """
    lock = _Lock(cfg.data_dir / "flywheel.lock")
    if not lock.acquire():
        logger.warning("a flywheel run holds %s and is alive; it restores its own GPU swap and "
                       "writes its own report, so recovery does nothing", lock.path)
        return EXIT_OK
    try:
        return _recover(cfg, clock=clock, runner=runner, repo=repo)
    finally:
        lock.release()


def _recover(cfg: FlywheelConfig, *, clock: Clock, runner: Runner, repo: Path) -> int:
    try:
        state = load_state(cfg.data_dir / "state.json")
    except FlywheelStateError as exc:
        logger.error("recovery cannot read the state: %s", exc)
        return EXIT_FAILED
    cur = _dict(state.get("current_run"))
    now = clock()
    run_id = str(cur.get("id") or now.astimezone(ZoneInfo(cfg.window.tz)).date())
    try:
        start, end = window_on(date.fromisoformat(run_id), cfg.window)
    except ValueError:
        start, end = now, now
    run = Run(cfg=cfg, run_id=run_id, window_start=start, window_end=end, dry_run=False,
              clock=clock, runner=runner, repo=repo, state=state, mode="recovery",
              started=_parse_ts(cur.get("started")) or now)
    release_gpu(run)
    written = True
    if cur.get("status") == "running":
        try:
            loaded = _read_json(run.run_dir / "stages.json")
        except (OSError, ValueError) as exc:
            logger.warning("recovery: the stage record is unreadable: %s", exc)
            loaded = None
        recorded = loaded if isinstance(loaded, dict) else {}
        died_in = ""
        for name in STAGES:
            raw = recorded.get(name)
            if isinstance(raw, dict):
                run.results[name] = _stage_record(raw, name)
            elif not died_in:
                died_in = name
                run.results[name] = StageResult(name, FAILED, reason=(
                    "the job died during this stage, or before it started"))
            else:
                run.results[name] = StageResult(name, NOT_RUN, reason="not reached: the job died "
                                                                     "first")
        run.issue("The weekly job ended without finishing (killed, crashed or timed out); the "
                  "recovery unit wrote this report from the stages it had recorded")
        cur.update(status="died")
        written = emit_report(run, "FAIL")
    save_state(run.state_path, run.state)
    return EXIT_FAILED if run.restore_failed or not written else EXIT_OK


def _fallback_config(block: Any) -> FlywheelConfig:
    """The defaults, but at the block's own data_dir, window and spend ledger
    where those are usable.

    A typo elsewhere in the block must not move the report away from where the
    Monday reviewer reads it: the refusal lands in the configured place, under
    the configured window's run id. Every other value is a default, not the
    operator's setting, and the config-error report states none of them.
    """
    defaults = parse_config(None)
    if not isinstance(block, dict):
        return defaults
    raw = block.get("data_dir")
    if isinstance(raw, str) and raw.strip():
        defaults = replace(defaults, data_dir=_resolve_under(raw.strip(), _data_root()))
    window = _block_window(block)
    if window is not None:
        defaults = replace(defaults, window=window)
    ledger = block.get("spend_ledger")
    if isinstance(ledger, str) and ledger.strip():
        defaults = replace(defaults, spend_ledger=_resolve_under(ledger.strip(), _data_root()))
    return defaults


def _block_window(block: Any) -> Window | None:
    """The block's own window, or None when it does not parse."""
    try:
        return parse_config({"window": _dict(block).get("window")}).window
    except FlywheelConfigError:
        return None


def main(argv: list[str] | None = None, *, clock: Clock | None = None,
         runner: Runner | None = None, repo: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m prometheus.jobs.flywheel_weekly",
        description="Weekly fine-tuning flywheel: harvest, verify, build, train, quantize, "
                    "score, report. Runs inside its window; --dry-run and --report-only any "
                    "time.")
    parser.add_argument("--config", default=None, help="prometheus.yaml (default: the search order)")
    how = parser.add_mutually_exclusive_group()
    how.add_argument("--dry-run", action="store_true",
                     help="read sources and state, write reports/dry-run/, change nothing")
    how.add_argument("--recover", action="store_true",
                     help="the OnFailure= unit: restore a GPU swap, finish a dead run's report")
    how.add_argument("--report-only", action="store_true",
                     help="rewrite the Monday reviewer's input (reports/LATEST.md and .json) from "
                          "the recorded state: no stage runs, the state is not changed")
    parser.add_argument("--note", action="append", default=[], metavar="TEXT",
                        help="context the job cannot read from its state (an operator decision, "
                             "a runbook gate); listed under Operator notes. Repeatable")
    parser.add_argument("--next-step", action="append", default=[], metavar="TEXT",
                        help="a step planned outside the job (the runbook's schedule); listed "
                             "under Next steps. Repeatable")
    args = parser.parse_args(argv)
    notes = [n for n in (_one_line(x) for x in args.note) if n]
    steps = [n for n in (_one_line(x) for x in args.next_step) if n]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from prometheus.security import install_log_redaction

    install_log_redaction()
    clock = clock or _utcnow
    runner = runner or run_command
    repo = repo or _repo_root()

    raw, problems = _read_config(args.config)
    unreadable = bool(problems)          # the file itself, so no setting of it is known
    cfg: FlywheelConfig | None = None
    if not problems:
        try:
            cfg = parse_config(raw.get("flywheel"))
        except FlywheelConfigError as exc:
            problems = exc.problems
    effective = cfg or _fallback_config(raw.get("flywheel"))
    if args.recover:
        return recover(effective, clock=clock, runner=runner, repo=repo)

    now = clock()
    if args.report_only and cfg is not None:
        return report_only(cfg, clock=clock, runner=runner, repo=repo, notes=notes, steps=steps)
    bounds = window_bounds(now, effective.window)
    # A bad config is reported wherever the run falls: the window it would be
    # checked against may be the broken part, and a refusal changes nothing.
    if bounds is None and not args.dry_run and not problems:
        logger.error("outside the flywheel window (%s); nothing done. --dry-run shows the plan, "
                     "--report-only refreshes the report.", effective.window.describe())
        return EXIT_CONFIG
    start, end = bounds or (now, now + effective.window.length)
    run_id = start.astimezone(ZoneInfo(effective.window.tz)).date().isoformat()

    if problems:
        for p in problems:
            logger.error("flywheel config: %s", p)
        run = Run(cfg=effective, run_id=run_id, window_start=start, window_end=end,
                  dry_run=args.dry_run, clock=clock, runner=runner, repo=repo, started=now,
                  state={}, mode="config-error", notes=notes, extra_steps=steps)
        if unreadable or _block_window(raw.get("flywheel")) is None:
            run.facts["window_unknown"] = True
        for name in STAGES:
            run.results[name] = StageResult(name, NOT_RUN, reason="the flywheel config is "
                                                                  "invalid (see Issues)")
        for p in problems:
            run.issue(f"config: {p}")
        stem = f"{run_id}-weekly-report-only" if args.report_only else None
        emit_report(run, "FAIL", stem=stem)
        return EXIT_CONFIG
    assert cfg is not None

    lock = _Lock(cfg.data_dir / "flywheel.lock")
    if not args.dry_run and not lock.acquire():
        logger.error("another flywheel run holds %s; this one does nothing", lock.path)
        return EXIT_BUSY
    stop_requested = {"flag": False}

    def on_sigterm(signum: int, frame: Any) -> None:
        stop_requested["flag"] = True
        logger.warning("SIGTERM: stopping after the current stage checkpoints")

    previous_handler = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        try:
            state = load_state(cfg.data_dir / "state.json")
        except FlywheelStateError as exc:
            run = Run(cfg=cfg, run_id=run_id, window_start=start, window_end=end,
                      dry_run=args.dry_run, clock=clock, runner=runner, repo=repo, started=now,
                      state={}, mode="state-error", notes=notes, extra_steps=steps)
            run.issue(str(exc))
            run.results["harvest"] = StageResult("harvest", FAILED, reason=str(exc))
            for name in STAGES[1:]:
                run.results[name] = StageResult(name, NOT_RUN, reason="the state is unreadable "
                                                                      "(see Issues)")
            emit_report(run, "FAIL")
            return EXIT_FAILED
        run = Run(cfg=cfg, run_id=run_id, window_start=start, window_end=end,
                  dry_run=args.dry_run, clock=clock, runner=runner, repo=repo, started=now,
                  state=state, interrupted=lambda: stop_requested["flag"],
                  mode="dry-run" if args.dry_run else "scheduled", notes=notes,
                  extra_steps=steps)
        try:
            if not run.dry_run:
                state["current_run"] = {"id": run_id, "status": "running", "pid": os.getpid(),
                                        "started": run.started.isoformat(timespec="seconds")}
                run.save()
            if not cfg.enabled:
                for name in STAGES:
                    run.results[name] = StageResult(name, SKIPPED, reason=(
                        "flywheel.enabled is false: the timer fired, the flywheel is off"))
            else:
                run_pipeline(run)
        except Exception as exc:  # noqa: BLE001 — recorded; the report still goes out
            logger.exception("flywheel run crashed")
            run.crashed = True
            run.issue(_redact(f"the run crashed outside a stage: {type(exc).__name__}: {exc}"))
        return _finish(run)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
