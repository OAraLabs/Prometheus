#!/usr/bin/env python3
"""The weekly fine-tuning flywheel: harvest, verify, build, train, quantize, score, report.

Run::

    python3 -m prometheus.jobs.flywheel_weekly             # the timer, inside the window
    python3 -m prometheus.jobs.flywheel_weekly --dry-run   # any time: the plan, nothing changed
    python3 -m prometheus.jobs.flywheel_weekly --recover   # the OnFailure= unit

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
   content against every earlier harvest. Each is tagged with its source
   model, adapter version, origin (teacher / offspring / unknown) and whether
   it is Max-derived.
2. **verify**   Decides which harvested traces may be trained on (rules below)
   and puts the trainable extra-source ones in the verified pool.
3. **build**    The dataset builder (scripts/distill/build_datasets.py, or
   build.py) over a copy of the golden sources holding only the traces verify
   passed, plus the verified pool. Sets whose byte-for-byte spot check has not
   passed are never trained on.
4. **train**    Only with >= ``train.min_new_traces`` verified traces since the
   last train's data was frozen, and as many train rows the last train did not
   have. A train the window cuts short resumes from its newest checkpoint next
   window, on the same frozen data.
5. **quantize** The finished adapter.
6. **score**    The quantized candidate (scripts/distill/score.py), only while
   the scoring server serves it: the Phase 6 gate against the base model's
   summary, and wins and losses against the previous model.

Then the report, always: ``<data_dir>/reports/LATEST.md`` plus a dated copy and
JSON. It is the Monday reviewer's input. A skipped, refused or paused stage
says why in it, and a disabled flywheel, a bad config, a crashed stage or a run
the recovery unit had to finish still writes one. Paths under the home
directory are written as ``~``; nothing here publishes the report.

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
  that is not the student itself. Its tool calls having succeeded is not a
  verification: that is how a model learns its own mistakes. A teacher's
  golden trace counts on its tool-success label (the runbook's rule); anything
  else is tagged and dropped, golden traces included: the builder reads a
  copy of them in which every trace verify did not pass is withheld. The train
  stage checks the frozen data again before every start and every resume.
* **The score cases.** A training row that shares an id, task or group with
  a configured ``score.cases`` file refuses the train (and the scoring of an
  adapter trained on it): the builder re-splits every week, so a task held out
  once can land in a later train split.
* **Max-derived data, without the operator's recorded approval.** Open
  question #6 (docs/sprints/DISTILL-OPEN-QUESTIONS.md) holds training on data
  derived from the hosted teacher. The train stage refuses a dataset holding
  any such row unless ``max_data`` records ``approved``, ``approved_by`` and
  ``approved_on``. The offspring of an adapter trained on Max-derived data is
  Max-derived too, and so is the offspring of an adapter this job did not
  train (nothing shows otherwise).

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
holds the lock.
"""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import glob
import hashlib
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
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger("prometheus.jobs.flywheel_weekly")

REPORT_SCHEMA = "flywheel-weekly-report/1"
STATE_SCHEMA = "flywheel-weekly-state/1"

STAGES = ("harvest", "verify", "build", "train", "quantize", "score")

OK = "OK"
SKIPPED = "SKIPPED"     # nothing to do, or a precondition is not met (the reason says which)
REFUSED = "REFUSED"     # a guard or a gate said no
PAUSED = "PAUSED"       # stopped for the window; picks up next window
FAILED = "FAILED"
DRY_RUN = "DRY-RUN"     # would run (the summary says what)

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
# The job owns these builder flags; build.args may not set them.
BUILD_OWNED_FLAGS = ("--traces", "--extra-traces", "--out", "--report", "--dry-run")

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
class MaxApproval:
    approved: bool = False
    approved_by: str = ""
    approved_on: str = ""
    reference: str = ""


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
    max_approval: MaxApproval
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
    teacher_models = h.texts("teacher_models", ("qwen3.8-max",))
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

    m = top.section("max_data")
    approval = MaxApproval(
        approved=m.flag("approved", False),
        approved_by=m.text("approved_by", ""),
        approved_on=str(m._get("approved_on", "")).strip(),
        reference=m.text("reference", ""),
    )
    m.done()
    if approval.approved:
        if not approval.approved_by:
            problems.append("flywheel.max_data.approved is true but approved_by is empty: "
                            "an approval records who gave it")
        try:
            on = date.fromisoformat(approval.approved_on)
            if on > _utcnow().date():
                problems.append("flywheel.max_data.approved_on is in the future")
        except ValueError:
            problems.append("flywheel.max_data.approved is true but approved_on is not a "
                            f"YYYY-MM-DD date ({approval.approved_on!r})")

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
        max_approval=approval, quantize_command=quantize_command,
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


# ---------------------------------------------------------------------------
# Files: atomic writes, JSON, the state
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: str | bytes) -> None:
    """tmp + fsync + rename: a reader sees the old file or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data.encode("utf-8") if isinstance(data, str) else data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


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
    t0 = time.monotonic()
    stopped = False
    term_at: datetime | None = None
    kill_at = spec.kill_at
    with open(spec.log_path, "ab") as log:
        log.write(f"\n=== {_utcnow().isoformat(timespec='seconds')} "
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
            now = _utcnow()
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
                               should_stop=self.interrupted)
        return CommandSpec(argv, self.repo, log, kill_at=self.clock() + SETUP_COMMAND_LIMIT)

    def swap_command(self, argv: list[str]) -> CommandSpec:
        """A step of the swap into gpu_setup: bounded, and held to the window too.

        The restore is not (it runs after the window if it must); a swap that
        has not finished by the stop time is stopped, and the restore follows.
        """
        log = self.data / "logs" / f"{self.run_id}-gpu-setup.log"
        return CommandSpec(argv, self.repo, log, stop_at=self.stop_at,
                           kill_at=min(self.clock() + SETUP_COMMAND_LIMIT, self.kill_at),
                           should_stop=self.interrupted)

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
    if str(meta.get("label") or "").strip().lower() == "fail":
        verdict = "fail"
    return {
        "source_model": model, "adapter_version": adapter, "origin": origin,
        "max_derived": bool(max_derived),
        "verdict": verdict if verdict in ("pass", "fail") else "",
        "verified_by": str(meta.get("verified_by") or meta.get("judge") or "").strip(),
    }


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


def stage_harvest(run: Run) -> StageResult:
    cfg = run.cfg
    seen = harvested_hashes(run.data)
    raw_adapters = run.state.get("adapters")
    adapters: dict[str, Any] = raw_adapters if isinstance(raw_adapters, dict) else {}
    rows: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    files = 0
    for kind, patterns in (("golden", cfg.golden), ("extra", cfg.extra)):
        for path in _expand(patterns):
            files += 1
            with open(path, encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if lineno % CLOCK_EVERY_LINES == 0 and run.out_of_time():
                        return StageResult("harvest", PAUSED, reason=(
                            "the window closed mid-harvest; nothing was recorded, and the next "
                            "window harvests these traces again"))
                    if not line.strip():
                        continue
                    try:
                        trace = json.loads(line)
                    except ValueError:
                        counts["unreadable"] += 1
                        continue
                    if shape_problem(trace):
                        counts["unreadable"] += 1
                        continue
                    h = trace_hash(trace)
                    if h in seen:
                        counts["already_harvested"] += 1
                        continue
                    seen.add(h)
                    meta = trace.get("_meta") or {}
                    tag = tag_trace(meta, cfg, adapters)
                    rows.append({"h": h, "kind": kind, "file": path.name, "line": lineno, **tag})
                    if kind == "extra":
                        contents.append({"h": h, "trace": trace})
    by_origin = Counter(r["origin"] for r in rows)
    table = Counter((r["origin"], r["source_model"], r["adapter_version"]) for r in rows)
    details = {
        "new": len(rows), "files": files, "by_origin": dict(by_origin),
        "already_harvested": counts["already_harvested"], "unreadable": counts["unreadable"],
        "by_source": [{"origin": o, "source_model": m, "adapter_version": a, "new": n}
                      for (o, m, a), n in sorted(table.items())],
        "offspring_adapter_unknown": sum(1 for r in rows if r["origin"] == "offspring"
                                         and r["adapter_version"] == "unknown"),
    }
    origin_text = ", ".join(f"{k} {v}" for k, v in sorted(by_origin.items())) or "none"
    summary = (f"{len(rows)} new trace(s) from {files} file(s) ({origin_text}); "
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
    # every later run's dedupe reads it.
    run.run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(run.run_dir / "harvest-traces.jsonl", contents)
    _atomic_write_jsonl(run.run_dir / "harvest.jsonl", rows)
    return StageResult("harvest", OK, summary=summary, details=details)


# ---------------------------------------------------------------------------
# Verify: what may be trained on
# ---------------------------------------------------------------------------


def verification(row: dict[str, Any]) -> tuple[str, str]:
    """(basis, "") when the trace may be trained on, else ("", why not).

    * a ``fail`` verdict drops anything;
    * offspring: only a ``pass`` from a named verifier that is not the
      offspring itself. Tool success is never enough;
    * teacher: a ``pass`` verdict, or for a golden trace its tool-success
      label (the exporter writes only calls that ran without error). An
      extra-source teacher trace needs a verdict, as the builder requires;
    * unknown origin: never.
    """
    if row.get("verdict") == "fail":
        return "", "verdict_fail"
    origin = row.get("origin")
    if origin == "offspring":
        by = str(row.get("verified_by") or "")
        if row.get("verdict") != "pass" or not by:
            return "", "unverified_offspring"
        if by.lower() in (str(row.get("source_model", "")).lower(),
                          str(row.get("adapter_version", "")).lower()):
            return "", "self_verified_offspring"
        return "judge-pass", ""
    if origin == "teacher":
        if row.get("verdict") == "pass":
            return "judge-pass", ""
        if row.get("kind") == "golden":
            return "tool-success", ""
        return "", "unverified_extra"
    return "", "unknown_origin"


def pool_source(row: dict[str, Any]) -> str:
    """The provenance a pooled trace hands the builder as ``_meta.source``.

    The builder copies it onto every training row as ``source``, which is what
    the train-time guards read: origin, model and adapter survive into the
    dataset, and a Max-derived row says so in a token the Max guard matches.
    """
    tail = "/max-derived" if row.get("max_derived") else ""
    return (f"flywheel/{row.get('origin')}/{row.get('source_model')}"
            f"@{row.get('adapter_version')}{tail}")


def _verify_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decided: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    verified_by_origin: Counter[str] = Counter()
    for row in rows:
        basis, why = verification(row)
        decided.append({"h": row["h"], "basis": basis, "reason": why})
        if basis:
            verified_by_origin[str(row.get("origin"))] += 1
        else:
            dropped[why] += 1
    verified = sum(verified_by_origin.values())
    return decided, {"verified": verified, "dropped": dict(dropped),
                     "verified_by_origin": dict(verified_by_origin), "rows": len(rows)}


def stage_verify(run: Run) -> StageResult:
    if run.dry_run:
        rows = run.facts.get("dry_harvest")
        if rows is None:
            return StageResult("verify", SKIPPED, reason="harvest did not run")
        _, counts = _verify_rows(rows)
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
        decided, counts = _verify_rows(rows)
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
            meta = dict(trace.get("_meta") or {})
            if meta.get("source") and "source_original" not in meta:
                meta["source_original"] = meta["source"]
            meta.update({
                "source": pool_source(row), "source_model": row["source_model"],
                "adapter_version": row["adapter_version"], "origin": row["origin"],
                "max_derived": row["max_derived"], "flywheel_run": rdir.name,
                "verification": {"basis": trainable[h], "by": row.get("verified_by") or None},
            })
            if trainable[h] == "judge-pass":
                meta["verdict"] = "pass"   # the builder's own verified-extra check reads it
            trace["_meta"] = meta
            pooled.append(trace)
        # The pool file first, verify.json last: verify.json is the commit point.
        _atomic_write_jsonl(run.data / "pool" / f"{rdir.name}.jsonl", pooled)
        record = {**counts, "pooled": len(pooled), "decisions": decided,
                  "verified_at": run.clock().isoformat(timespec="seconds")}
        _atomic_write(rdir / "verify.json", json.dumps(record, indent=1) + "\n")
        total["verified"] += counts["verified"]
        total["pooled"] += len(pooled)
        dropped.update(counts["dropped"])
        if rdir.name == run.run_id:
            this_run = counts
    details = {"runs": [p.name for p in pending], "verified": total["verified"],
               "pooled": total["pooled"], "dropped": dict(dropped), "this_run": this_run}
    if dropped.get("unverified_offspring") or dropped.get("self_verified_offspring"):
        details["offspring_dropped"] = (dropped["unverified_offspring"]
                                        + dropped["self_verified_offspring"])
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
    # Same inputs, same builder, same arguments: the same sets. A changed
    # argument (the operator fixing --pack, say) rebuilds without new traces.
    inputs = hashlib.sha256(json.dumps(
        [str(script), list(cfg.build_args), list(cfg.golden)]).encode()).hexdigest()[:16]
    if sets and newest <= str(sets.get("run") or "") and sets.get("inputs") == inputs:
        return StageResult("build", SKIPPED, reason=(
            f"no verified traces since the {sets.get('run')} build; its sets stand"),
            details={"sets": sets})
    out = run.run_dir / "sets"
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
    if run.dry_run:
        return StageResult("build", DRY_RUN, summary=f"would run: {_quote(argv)}")
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
        "rows": written, "inputs": inputs,
    }
    run.save()
    details = {"failed_gates": failed, "spot_check": spot, "rows": written,
               "pool_files": len(pool), "golden": golden}
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
    an unknown teacher model counts as Max-derived (nothing shows otherwise).
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
                if all_max or is_max_name(src):
                    g.max_rows += 1
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
    if g.max_rows and not cfg.max_approval.approved:
        g.refusals.append(
            f"{g.max_rows} of {g.rows} row(s) are Max-derived and no operator approval is "
            "recorded (flywheel.max_data: approved, approved_by, approved_on). "
            "DISTILL-OPEN-QUESTIONS #6 holds training on Max-derived data")
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


def _freeze(run: Run, sets: dict[str, Any], guard: GuardResult) -> dict[str, Any]:
    """Pin the train's data: hard links (or copies) of the split files, with hashes.

    A resume weeks later trains on these exact bytes, whatever later builds
    write. The run directory is named after the adapter version, so a trainer
    that names its logs after its output directory names them usefully.
    """
    base = f"fw-{run.run_id}"
    version, n = base, 1
    while (run.data / "train" / version).exists() or (run.data / "train" / f"{version}.data").exists():
        n += 1
        version = f"{base}-{n}"
    run_dir = run.data / "train" / version
    data_dir = run.data / "train" / f"{version}.data"
    data_dir.mkdir(parents=True)
    shas: dict[str, str] = {}
    for name in (*FROZEN_FILES, "report.json"):
        src = Path(str(sets["dir"])) / name
        if src.is_file():
            _link_or_copy(src, data_dir / name)
            shas[name] = _sha256(data_dir / name)
    return {
        "adapter_version": version, "status": "running", "data_dir": str(data_dir),
        "run_dir": str(run_dir), "adapter_dir": str(run_dir / ADAPTER_SUBDIR),
        "cutoff": str(sets.get("run") or run.run_id), "sets": str(sets["dir"]),
        "teacher_model": guard.teacher_model, "max_derived": guard.max_rows > 0,
        "sha256": shas, "rows": guard.rows, "attempts": 0, "failures": 0,
        "frozen_in": run.run_id,
    }


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
            max_note = ("approved" if cfg.max_approval.approved
                        else "refused: no max_data approval is recorded")
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
    verb = f"resume {t.get('adapter_version')}" if resuming and t else "start a train"
    if run.dry_run:
        return StageResult("train", DRY_RUN, summary=(
            f"would {verb} on {guard.rows} row(s) with: {' '.join(cfg.train_command)}"),
            details=details)
    note = ensure_gpu(run, "train")
    if note:
        return StageResult("train", SKIPPED, reason=note, details=details)

    if not resuming:
        t = _freeze(run, sets, guard)
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
    t.update(status="running", attempts=int(t.get("attempts") or 0) + 1, last_run=run.run_id)
    run.state["train"] = t
    run.save()
    details.update(adapter_version=version, resumed_from=ckpt.name if ckpt else None)

    res = run.runner(run.command(argv, "train"))
    now = run.clock().isoformat(timespec="seconds")
    said = _trainer_status(run_dir) if res.returncode == 0 and not res.stopped else ""
    adapter_dir = Path(str(t["adapter_dir"]))
    if res.returncode == 0 and not res.stopped and said in ("", "done") and adapter_dir.is_dir():
        t.update(status="complete", finished=now)
        run.state["last_train"] = {"adapter_version": version, "cutoff": t["cutoff"],
                                   "finished": now}
        adapters = run.state.setdefault("adapters", {})
        adapters[version] = {"max_derived": bool(t.get("max_derived", True)),
                             "adapter_dir": str(adapter_dir), "data_dir": t["data_dir"],
                             "trained": now}
        run.save()
        return StageResult("train", OK, summary=(
            f"{version} finished ({t['attempts']} window(s), {guard.rows} rows)"),
            details=details)
    if res.returncode == 0 and not res.stopped and said != "stopped":
        res = CommandResult(1, False, res.seconds, (
            f"exited 0 but left no adapter at {adapter_dir}" if said in ("", "done")
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
    out_dir = run.data / "models" / version
    record = _dict(adapters.get(version))
    argv = run.fill(cfg.quantize_command, {
        "adapter_dir": str(record.get("adapter_dir") or ""),
        "data_dir": str(record.get("data_dir") or ""),
        "out_dir": str(out_dir), "adapter_version": version})
    if run.dry_run:
        return StageResult("quantize", DRY_RUN, summary=f"would run: {_quote(argv)}")
    note = ensure_gpu(run, "quantize")
    if note:
        return StageResult("quantize", SKIPPED, reason=note)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run.runner(run.command(argv, "quantize"))
    if res.returncode == 0 and not res.stopped:
        quantized[version] = {"status": "done", "out_dir": str(out_dir),
                              "finished": run.clock().isoformat(timespec="seconds")}
        run.save()
        return StageResult("quantize", OK, summary=f"{version} quantized into {out_dir}")
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
    return StageResult(name, str(raw.get("status")), float(raw.get("seconds") or 0.0),
                       str(raw.get("summary") or ""), "done earlier in this window",
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
            t0 = time.monotonic()
            try:
                result = STAGE_FUNCS[name](run)
            except Exception as exc:  # noqa: BLE001 — one stage's crash must not cost the report
                logger.exception("flywheel stage %s crashed", name)
                result = StageResult(name, FAILED,
                                     reason=_redact(f"{type(exc).__name__}: {exc}")[:500])
            result.seconds = round(time.monotonic() - t0, 1)
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
    approval = cfg.max_approval
    harvest = run.results.get("harvest")
    verify = run.results.get("verify")
    return {
        "schema": REPORT_SCHEMA, "kind": "weekly", "run_id": run.run_id, "mode": run.mode,
        "dry_run": run.dry_run, "status": status,
        "window": {"describe": cfg.window.describe(),
                   "start": run.window_start.isoformat(timespec="minutes"),
                   "end": run.window_end.isoformat(timespec="minutes"),
                   "stop_at": run.stop_at.astimezone(tz).isoformat(timespec="minutes")},
        "started": run.started.astimezone(tz).isoformat(timespec="seconds"),
        "finished": run.clock().astimezone(tz).isoformat(timespec="seconds"),
        "stages": [asdict(run.results[name]) for name in STAGES if name in run.results],
        "traces": {
            "harvest": harvest.details if harvest else {},
            "verify": verify.details if verify else {},
            "new_verified_since_last_train": _safe_pending(run),
            "min_new_traces": cfg.min_new_traces,
        },
        "train": {k: t.get(k) for k in ("adapter_version", "status", "attempts", "failures",
                                        "cutoff", "max_derived", "rows")} if t else None,
        "model": model_section,
        "spend": spend_summary(cfg.spend_ledger, run.window_start - timedelta(days=7)),
        "guards": {
            "max_data": {"approved": approval.approved, "approved_by": approval.approved_by,
                         "approved_on": approval.approved_on, "reference": approval.reference},
            "train_data": guard if isinstance(guard, dict) else None,
            "offspring_dropped_at_verify": (verify.details.get("offspring_dropped", 0)
                                            if verify else 0),
        },
        "issues": list(run.issues),
    }


def _safe_pending(run: Run) -> int | None:
    try:
        return new_verified_since_last_train(run)
    except (OSError, ValueError) as exc:
        logger.warning("could not count verified traces: %s", exc)
        return None


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    w = report["window"]
    lines = [
        f"# Flywheel weekly report: {report['run_id']}",
        "",
        f"**Status: {report['status']}**"
        + (" (DRY RUN: nothing was changed)" if report["dry_run"] else "")
        + f" · window {w['describe']} · ran {report['started'][11:16]}-{report['finished'][11:16]}"
        + f" · mode: {report['mode']}",
        "",
        "| Stage | Result | Time | Summary |",
        "|---|---|---|---|",
    ]
    for s in report["stages"]:
        text = s["summary"] or s["reason"]
        lines.append(f"| {s['name']} | {s['status']} | {s['seconds']:.0f} s | "
                     f"{_md_escape(text)} |")
    not_run = [s for s in report["stages"] if s["status"] not in (OK, DRY_RUN)]
    lines += ["", "## Why stages did not run", ""]
    if not_run:
        lines += [f"- **{s['name']}** {s['status']}: {s['reason'] or s['summary']}"
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
    if vf:
        dropped = ", ".join(f"{k} {v}" for k, v in sorted((vf.get("dropped") or {}).items()))
        lines.append(f"- Verified: {vf.get('verified', 0)}; dropped: {dropped or 'none'}")
    pending = tr.get("new_verified_since_last_train")
    lines.append(f"- New verified since the last train: "
                 f"{'unknown' if pending is None else pending} "
                 f"(a train needs {tr.get('min_new_traces')})")
    by_source = hv.get("by_source") or []
    if by_source:
        lines += ["", "| Origin | Source model | Adapter | New |", "|---|---|---|---|"]
        lines += [f"| {r['origin']} | {_md_escape(r['source_model'])} | "
                  f"{_md_escape(r['adapter_version'])} | {r['new']} |" for r in by_source]

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
        if m.get("error"):
            lines.append(f"\n{m['error']}")
    else:
        lines.append("No new model this week.")
        lines += [f"- {w}" for w in m.get("why") or []]
        last = m.get("last_scored")
        if last:
            lines.append(f"- Last scored: {last['adapter_version']} ({last['verdict']}, "
                         f"{last['scored_in']})")
    t = report.get("train")
    if t:
        lines.append(f"\nTrain record: {t.get('adapter_version')} is {t.get('status')} "
                     f"(attempts {t.get('attempts')}, failures {t.get('failures')}).")

    sp = report["spend"]
    lines += ["", "## Spend", ""]
    if not sp.get("present"):
        lines.append(f"No spend ledger at {sp['ledger']}: no API spend recorded.")
    elif sp.get("error"):
        lines.append(f"Spend ledger unreadable ({sp['error']}).")
    else:
        lines.append(f"- Last 7 days: {_fmt_spend(sp['week'])}")
        lines.append(f"- All time: {_fmt_spend(sp['total'])}")
        for phase, acc in sorted(sp["by_phase"].items()):
            lines.append(f"  - {phase}: {_fmt_spend(acc)}")
        if sp.get("undated_rows"):
            lines.append(f"- {sp['undated_rows']} ledger row(s) have no readable timestamp "
                         "(counted in all time only)")

    g = report["guards"]
    lines += ["", "## Guards", ""]
    md = g["max_data"]
    if md["approved"]:
        lines.append(f"- Max-derived data: training ALLOWED, approved by {md['approved_by']} on "
                     f"{md['approved_on']}" + (f" ({md['reference']})" if md["reference"] else ""))
    else:
        lines.append("- Max-derived data: training refused (no operator approval recorded; "
                     "DISTILL-OPEN-QUESTIONS #6)")
    td = g.get("train_data")
    if td:
        lines.append(f"- Train data checked: {td['rows']} row(s), {td['max_rows']} Max-derived, "
                     f"{td['offspring_rows']} offspring ({td['offspring_unverified']} unverified), "
                     f"{td.get('leaked_rows', 0)} sharing a task with the score cases")
    lines.append(f"- Offspring traces dropped as unverified this window: "
                 f"{g.get('offspring_dropped_at_verify', 0)}")

    lines += ["", "## Issues", ""]
    lines += [f"- {i}" for i in report["issues"]] or ["None."]
    return "\n".join(lines) + "\n"


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


def write_report(data: Path, report: dict[str, Any]) -> Path:
    """The dated copy first, then LATEST, each by atomic rename."""
    report = public_paths(report)
    rdir = data / "reports" / ("dry-run" if report["dry_run"] else "")
    text = render_markdown(report)
    blob = json.dumps(report, indent=1, default=str) + "\n"
    stem = f"{report['run_id']}-weekly"
    _atomic_write(rdir / f"{stem}.json", blob)
    _atomic_write(rdir / f"{stem}.md", text)
    _atomic_write(rdir / "LATEST.json", blob)
    _atomic_write(rdir / "LATEST.md", text)
    return rdir / "LATEST.md"


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
    try:
        path = write_report(run.data, build_report(run, status))
        logger.info("flywheel report (%s): %s", status, path)
    except Exception:  # noqa: BLE001 — logged; the exit code carries it
        logger.exception("could not write the flywheel report")
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
    run_id = str(cur.get("id") or clock().astimezone(ZoneInfo(cfg.window.tz)).date())
    now = clock()
    run = Run(cfg=cfg, run_id=run_id, window_start=now, window_end=now, dry_run=False,
              clock=clock, runner=runner, repo=repo, state=state, mode="recovery", started=now)
    release_gpu(run)
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
                run.results[name] = _stage_from(raw, name)
                run.results[name].reason = str(raw.get("reason") or "")
            elif not died_in:
                died_in = name
                run.results[name] = StageResult(name, FAILED, reason=(
                    "the job died during this stage, or before it started"))
            else:
                run.results[name] = StageResult(name, SKIPPED, reason="not reached")
        run.issue("The weekly job ended without finishing (killed, crashed or timed out); the "
                  "recovery unit wrote this report from the stages it had recorded")
        cur.update(status="died")
        write_report(run.data, build_report(run, "FAIL"))
    save_state(run.state_path, run.state)
    return EXIT_FAILED if run.restore_failed else EXIT_OK


def _fallback_config(block: Any) -> FlywheelConfig:
    """The defaults, but at the block's own data_dir and window where those are usable.

    A typo elsewhere in the block must not move the report away from where the
    Monday reviewer reads it: the refusal lands in the configured place, under
    the configured window's run id.
    """
    defaults = parse_config(None)
    if not isinstance(block, dict):
        return defaults
    raw = block.get("data_dir")
    if isinstance(raw, str) and raw.strip():
        defaults = replace(defaults, data_dir=_resolve_under(raw.strip(), _data_root()))
    try:
        defaults = replace(defaults, window=parse_config({"window": block.get("window")}).window)
    except FlywheelConfigError:
        pass
    return defaults


def main(argv: list[str] | None = None, *, clock: Clock | None = None,
         runner: Runner | None = None, repo: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m prometheus.jobs.flywheel_weekly",
        description="Weekly fine-tuning flywheel: harvest, verify, build, train, quantize, "
                    "score, report. Runs inside its window; --dry-run any time.")
    parser.add_argument("--config", default=None, help="prometheus.yaml (default: the search order)")
    parser.add_argument("--dry-run", action="store_true",
                        help="read sources and state, write reports/dry-run/, change nothing")
    parser.add_argument("--recover", action="store_true",
                        help="the OnFailure= unit: restore a GPU swap, finish a dead run's report")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from prometheus.security import install_log_redaction

    install_log_redaction()
    clock = clock or _utcnow
    runner = runner or run_command
    repo = repo or _repo_root()

    raw, problems = _read_config(args.config)
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
    bounds = window_bounds(now, effective.window)
    # A bad config is reported wherever the run falls: the window it would be
    # checked against may be the broken part, and a refusal changes nothing.
    if bounds is None and not args.dry_run and not problems:
        logger.error("outside the flywheel window (%s); nothing done. --dry-run shows the plan.",
                     effective.window.describe())
        return EXIT_CONFIG
    start, end = bounds or (now, now + effective.window.length)
    run_id = start.astimezone(ZoneInfo(effective.window.tz)).date().isoformat()

    if problems:
        for p in problems:
            logger.error("flywheel config: %s", p)
        run = Run(cfg=effective, run_id=run_id, window_start=start, window_end=end,
                  dry_run=args.dry_run, clock=clock, runner=runner, repo=repo, started=now,
                  state={}, mode="config-error")
        for name in STAGES:
            run.results[name] = StageResult(name, SKIPPED, reason="the flywheel config is "
                                                                  "invalid (see Issues)")
        for p in problems:
            run.issue(f"config: {p}")
        write_report(run.data, build_report(run, "FAIL"))
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
                      dry_run=args.dry_run, clock=clock, runner=runner, repo=repo, started=now, state={},
                      mode="state-error")
            run.issue(str(exc))
            run.results["harvest"] = StageResult("harvest", FAILED, reason=str(exc))
            write_report(run.data, build_report(run, "FAIL"))
            return EXIT_FAILED
        run = Run(cfg=cfg, run_id=run_id, window_start=start, window_end=end,
                  dry_run=args.dry_run, clock=clock, runner=runner, repo=repo, started=now, state=state,
                  interrupted=lambda: stop_requested["flag"],
                  mode="dry-run" if args.dry_run else "scheduled")
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
