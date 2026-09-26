"""The model-ladder suite: loading and validating the frozen task set.

A ladder suite is a DIRECTORY (``gym/ladder/v1/``): ``suite.yaml`` names the
task classes, their budgets and the shared system prompt, and each class
lives in its own file. The suite is FROZEN the way a gym task set is —
``sha256`` covers every YAML file in the directory, and every recorded run
carries it, so a row can always be traced to the exact bytes it ran against.

What the loader enforces is what makes a row a VERDICT rather than an
impression:

* every task has a verdict path — deterministic predicates, an acceptance
  command, or a judge rubric;
* an acceptance command and a judge never share a task: where a command can
  decide, a model does not grade;
* every task that can be checked mechanically carries a ``reference`` — what
  a correct agent leaves behind — so the test suite can prove each check
  FAILS on the untouched setup and PASSES on a correct solution;
* every web task says whether it is offline (fixture pages) or live, and
  whether its answer changes over time.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from prometheus.gym.tasks import ALLOWED_SCORE_KEYS as GYM_SCORE_KEYS

# The eight classes, in report order. Changing this list is a new suite
# version, not an edit.
CLASS_IDS: tuple[str, ...] = (
    "qa",
    "single_tool",
    "multi_step",
    "file_edit",
    "web_research",
    "memory_recall",
    "scheduling",
    "long_haul",
)

# Predicates the ladder adds on top of the gym's (all deterministic).
LADDER_SCORE_KEYS = frozenset({
    "expect_answer",       # str regex: re.fullmatch against the value of the reply's
                           # LAST "ANSWER: <value>" line (see verdict.read_answer)
    "answer_shape",        # str regex: what ANY value of this answer looks like (a
                           # number, a weekday, a path). A reply with no ANSWER line
                           # whose bare last line has this shape but is wrong is a
                           # checkable FAIL; without a shape it is a format miss
    "expect_text_any",     # list[str]: final text contains ≥1 (case-insensitive)
    "expect_text_all",     # list[str]: final text contains every one
    "expect_text_regex",   # str: re.search over the final text (IGNORECASE)
    "forbid_text",         # list[str]: final text contains none (case-insensitive)
    "forbid_text_regex",   # list[str]: no re.search match in the final text
    "expect_tool_any",     # list[str]: ≥1 successful call of any of these tools
    "expect_tools_all",    # list[str]: ≥1 successful call of EACH of these tools
    "expect_file_regex",   # {path, pattern}: file exists and re.search matches
    "expect_file_absent",  # str path: file must NOT exist after the run
    "expect_cron_job",     # {name, schedule_any?, schedule_regex?,
                           #  command_contains?, enabled?}
    "forbid_cron_job",     # str name: no job of that name after the run
})
ALL_SCORE_KEYS = GYM_SCORE_KEYS | LADDER_SCORE_KEYS

TASK_KEYS = frozenset({
    "id", "prompt", "score", "acceptance", "acceptance_files", "judge",
    "setup_files", "fixtures", "seed", "reference", "web",
    "answer_changes_over_time", "smoke", "max_rounds", "max_tool_calls",
    "max_tokens", "timeout_s", "notes", "difficulty",
})
DIFFICULTIES = ("easy", "medium", "hard")
FIXTURE_KEYS = frozenset({
    "web_pages", "wiki_pages", "lcm_history", "cron_jobs", "memory_md", "user_md",
})
REFERENCE_KEYS = frozenset({
    "answer", "files", "cron_jobs", "tools", "delete_files", "wrong_answers",
    "wrong_files", "wrong_cron_jobs", "right_answers", "format_misses",
})
JUDGE_KEYS = frozenset({"rubric", "reference", "threshold"})
WEB_MODES = frozenset({"offline", "live"})
BUDGET_KEYS = ("max_rounds", "max_tool_calls", "max_tokens", "timeout_s")


CLASS_STATUSES = ("active", "deferred")


@dataclass
class ClassSpec:
    id: str
    # "active": the class has a task file and runs. "deferred": defined — its
    # budgets and success criterion are fixed here — but it has no tasks yet.
    status: str
    file: str | None
    max_rounds: int
    max_tool_calls: int
    max_tokens: int
    timeout_s: float
    success_criterion: str
    description: str = ""


@dataclass
class LadderTask:
    id: str
    task_class: str
    prompt: str
    score: dict[str, Any] = field(default_factory=dict)
    # ``{unittest} <modules>``: tests the HARNESS runs in the workspace after
    # the agent finishes, through its own isolated runner (verdict.py,
    # accept.py) — at least one test must run and none fail.
    # ``acceptance_files`` are (re)written just before, so an agent that
    # edits the tests cannot make them pass.
    acceptance: str | None = None
    acceptance_files: dict[str, str] = field(default_factory=dict)
    # {rubric, reference, threshold} — graded by the pinned local judge.
    judge: dict[str, Any] | None = None
    setup_files: dict[str, str] = field(default_factory=dict)
    fixtures: dict[str, Any] = field(default_factory=dict)
    seed: list[dict[str, Any]] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)
    web: str | None = None
    answer_changes_over_time: bool | None = None
    smoke: bool = False
    # The author's grading WITHIN the class, so a report can say "passes the
    # easy file edits, fails the hard ones" rather than one blended rate.
    difficulty: str | None = None
    max_rounds: int = 8
    max_tool_calls: int = 8
    max_tokens: int = 1024
    timeout_s: float = 180.0
    notes: str = ""

    @property
    def has_deterministic_check(self) -> bool:
        return bool(self.score) or bool(self.acceptance)

    @property
    def verdict_source(self) -> str:
        parts = []
        if self.score:
            parts.append("predicates")
        if self.acceptance:
            parts.append("acceptance")
        if self.judge:
            parts.append("judge")
        return "+".join(parts)


@dataclass
class LadderSuite:
    name: str
    version: int
    path: str
    sha256: str
    system_prompt: str
    workspace: str
    classes: dict[str, ClassSpec]
    tasks: list[LadderTask]

    @property
    def active_classes(self) -> list[str]:
        return [c for c, spec in self.classes.items() if spec.status == "active"]

    @property
    def deferred_classes(self) -> list[str]:
        return [c for c, spec in self.classes.items() if spec.status == "deferred"]

    def by_class(self) -> dict[str, list[LadderTask]]:
        out: dict[str, list[LadderTask]] = {c: [] for c in self.classes}
        for t in self.tasks:
            out[t.task_class].append(t)
        return out


def _fail(where: str, msg: str) -> None:
    raise ValueError(f"{where}: {msg}")


def _validate_task(raw: dict[str, Any], where: str, shared_fixtures: dict) -> None:
    unknown = set(raw) - TASK_KEYS
    if unknown:
        _fail(where, f"unknown task key(s) {sorted(unknown)}")
    for key in ("id", "prompt"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            _fail(where, f"{key!r} must be a non-empty string (quote it — YAML reads "
                         f"bare on/off/yes/no as booleans)")

    score = raw.get("score") or {}
    bad = set(score) - ALL_SCORE_KEYS
    if bad:
        _fail(where, f"unknown score predicate(s) {sorted(bad)}")
    if "answer_shape" in score and "expect_answer" not in score:
        _fail(where, "answer_shape only means something next to expect_answer")
    for key in ("expect_answer", "answer_shape"):
        if key in score:
            try:
                re.compile(score[key])
            except re.error as exc:
                _fail(where, f"{key} does not compile: {exc}")

    judge = raw.get("judge")
    if judge is not None:
        if not isinstance(judge, dict) or set(judge) - JUDGE_KEYS:
            _fail(where, f"judge must be a mapping with keys from {sorted(JUDGE_KEYS)}")
        if not judge.get("rubric") or not judge.get("reference"):
            _fail(where, "judge needs both a rubric and a reference answer")
        thr = judge.get("threshold", 0.7)
        if not 0.0 < float(thr) <= 1.0:
            _fail(where, f"judge threshold {thr!r} outside (0, 1]")

    acceptance = raw.get("acceptance")
    if acceptance is not None:
        from prometheus.gym.ladder.verdict import ACCEPTANCE_RE

        if not isinstance(acceptance, str) or not ACCEPTANCE_RE.match(acceptance):
            _fail(where, "acceptance must be '{unittest} <test modules>' — the harness "
                         "runs it through its own runner; a shell command's exit "
                         "status proves nothing")
    if acceptance and judge:
        _fail(where, "acceptance and judge on one task — where a command can "
                     "decide, a model does not grade")
    if raw.get("acceptance_files") and not acceptance:
        _fail(where, "acceptance_files without an acceptance command")
    if not score and not acceptance and not judge:
        _fail(where, "no verdict path (needs score, acceptance or judge)")

    fixtures = raw.get("fixtures") or {}
    bad = set(fixtures) - FIXTURE_KEYS
    if bad:
        _fail(where, f"unknown fixture key(s) {sorted(bad)}")

    reference = raw.get("reference") or {}
    bad = set(reference) - REFERENCE_KEYS
    if bad:
        _fail(where, f"unknown reference key(s) {sorted(bad)}")
    if (score or acceptance) and not reference:
        _fail(where, "a mechanically checked task needs a reference solution, so "
                     "the test suite can prove the check discriminates")

    web = raw.get("web")
    uses_web = bool(fixtures.get("web_pages") or shared_fixtures.get("web_pages")) or web
    if web is not None and web not in WEB_MODES:
        _fail(where, f"web must be one of {sorted(WEB_MODES)}, got {web!r}")
    if uses_web:
        if web is None:
            _fail(where, "a web task must declare web: offline|live")
        if not isinstance(raw.get("answer_changes_over_time"), bool):
            _fail(where, "a web task must declare answer_changes_over_time: true|false")
    if web == "offline" and raw.get("answer_changes_over_time"):
        _fail(where, "an offline (fixture) answer cannot change over time")

    if "difficulty" in raw and raw["difficulty"] not in DIFFICULTIES:
        _fail(where, f"difficulty must be one of {DIFFICULTIES}, got {raw['difficulty']!r}")

    for key in BUDGET_KEYS:
        if key in raw and (not isinstance(raw[key], (int, float)) or raw[key] <= 0):
            _fail(where, f"{key} must be a positive number")


def _hash_dir(root: Path) -> str:
    """sha256 over every YAML file under *root*, path-sorted.

    Path AND bytes go into the digest, so renaming a class file is a new
    suite exactly as editing one is.
    """
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.yaml")):
        rel = p.relative_to(root).as_posix().encode()
        h.update(rel + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def load_suite(path: str | Path) -> LadderSuite:
    root = Path(path)
    meta_path = root / "suite.yaml"
    meta = yaml.safe_load(meta_path.read_text())
    if not isinstance(meta, dict) or "classes" not in meta:
        raise ValueError(f"{meta_path}: not a ladder suite (missing 'classes')")

    classes: dict[str, ClassSpec] = {}
    for cid, spec in meta["classes"].items():
        where = f"{meta_path} class {cid!r}"
        if cid not in CLASS_IDS:
            _fail(str(meta_path), f"unknown class {cid!r}; known: {list(CLASS_IDS)}")
        status = spec.get("status", "active")
        if status not in CLASS_STATUSES:
            _fail(where, f"status must be one of {CLASS_STATUSES}, got {status!r}")
        for key in BUDGET_KEYS:
            if key not in spec:
                _fail(where, f"missing {key!r}")
        criterion = str(spec.get("success_criterion") or "").strip()
        if not criterion:
            _fail(where, "missing success_criterion — a class without one is not defined")
        if status == "active" and not spec.get("file"):
            _fail(where, "an active class needs a task file")
        if status == "deferred" and spec.get("file"):
            _fail(where, "a deferred class has no task file yet — drop 'file' or make it active")
        classes[cid] = ClassSpec(
            id=cid,
            status=status,
            file=spec.get("file"),
            max_rounds=int(spec["max_rounds"]),
            max_tool_calls=int(spec["max_tool_calls"]),
            max_tokens=int(spec["max_tokens"]),
            timeout_s=float(spec["timeout_s"]),
            success_criterion=criterion,
            description=str(spec.get("description", "")).strip(),
        )

    tasks: list[LadderTask] = []
    seen: set[str] = set()
    for cid, cspec in classes.items():
        if cspec.file is None:
            continue
        cpath = root / cspec.file
        data = yaml.safe_load(cpath.read_text())
        if not isinstance(data, dict) or data.get("class") != cid:
            _fail(str(cpath), f"file must declare class: {cid}")
        shared = data.get("fixtures") or {}
        bad = set(shared) - FIXTURE_KEYS
        if bad:
            _fail(str(cpath), f"unknown shared fixture key(s) {sorted(bad)}")
        for i, raw in enumerate(data.get("tasks") or []):
            where = f"{cpath.name} task #{i} ({raw.get('id', '?')})"
            _validate_task(raw, where, shared)
            if raw["id"] in seen:
                _fail(where, f"duplicate task id {raw['id']!r}")
            seen.add(raw["id"])
            # File-level fixtures are shared by every task in the file; a
            # task's own entries extend (lists) or override (maps) them.
            fixtures = _merge_fixtures(shared, raw.get("fixtures") or {})
            tasks.append(LadderTask(
                id=raw["id"],
                task_class=cid,
                prompt=str(raw["prompt"]).strip(),
                score=raw.get("score") or {},
                acceptance=raw.get("acceptance"),
                acceptance_files=raw.get("acceptance_files") or {},
                judge=raw.get("judge"),
                setup_files=raw.get("setup_files") or {},
                fixtures=fixtures,
                seed=raw.get("seed") or [],
                reference=raw.get("reference") or {},
                web=raw.get("web"),
                answer_changes_over_time=raw.get("answer_changes_over_time"),
                smoke=bool(raw.get("smoke", False)),
                difficulty=raw.get("difficulty"),
                max_rounds=int(raw.get("max_rounds", cspec.max_rounds)),
                max_tool_calls=int(raw.get("max_tool_calls", cspec.max_tool_calls)),
                max_tokens=int(raw.get("max_tokens", cspec.max_tokens)),
                timeout_s=float(raw.get("timeout_s", cspec.timeout_s)),
                notes=str(raw.get("notes", "")).strip(),
            ))

    return LadderSuite(
        name=str(meta.get("name", root.name)),
        version=int(meta.get("version", 1)),
        path=str(root),
        sha256=_hash_dir(root),
        system_prompt=str(meta.get("system_prompt", "")).strip(),
        workspace=str(meta.get("workspace", "/tmp/prometheus-ladder")),
        classes=classes,
        tasks=tasks,
    )


def _merge_fixtures(shared: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in FIXTURE_KEYS:
        a, b = shared.get(key), own.get(key)
        if a is None and b is None:
            continue
        if isinstance(a, list) or isinstance(b, list):
            out[key] = list(a or []) + list(b or [])
        elif isinstance(a, dict) or isinstance(b, dict):
            out[key] = {**(a or {}), **(b or {})}
        else:
            out[key] = b if b is not None else a
    return out


def select_tasks(
    suite: LadderSuite,
    *,
    classes: list[str] | None = None,
    task_ids: list[str] | None = None,
    smoke: bool = False,
    include_live_web: bool = False,
) -> list[LadderTask]:
    """The tasks a run will execute, in suite order.

    Live-web tasks are OFF unless asked for: they reach public hosts and
    their verdicts are not reproducible across machines or dates.
    """
    out = []
    for t in suite.tasks:
        if classes and t.task_class not in classes:
            continue
        if task_ids and t.id not in task_ids:
            continue
        if smoke and not t.smoke:
            continue
        if t.web == "live" and not include_live_web:
            continue
        out.append(t)
    return out
