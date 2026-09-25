"""Verdicts for ladder runs: predicates, acceptance commands, the pinned judge.

Order of authority, per task:

1. **Deterministic predicates** over the finished run (the gym's, plus the
   ladder's text / tool / file / cron checks below).
2. **Acceptance tests** — ``{unittest} <modules>``, run by the HARNESS in the
   workspace after the agent stops, through its own runner (``accept.py``):
   isolated interpreter, standard library ahead of the workspace, at least one
   test must RUN, and the verdict comes from a result file, not the exit code.
   Tests the agent could see are restored from ``acceptance_files`` first.
3. **Judge** — only where no command can decide, and only once the
   deterministic checks have passed. The judge is a local model PINNED to a
   different model from the contestant (enforced in the runner).

A check that did not run is never a pass. A judge that errors, times out or
returns anything but a parsed score on the 0-1 scale makes the run
``unscored`` (success NULL), and an acceptance run the harness could not even
start does the same. Both log a WARNING. What the MODEL did to the workspace
— a directory where a file should be, a symlinked test, code that exits the
interpreter at import — is never a harness failure: it fails the run.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.gym.scoring import EMISSION, EXECUTION, RunTranscript
from prometheus.gym.scoring import score as gym_score
from prometheus.gym.tasks import ALLOWED_SCORE_KEYS as GYM_SCORE_KEYS

log = logging.getLogger(__name__)

ACCEPTANCE_TIMEOUT_S = 120.0
ACCEPT_RUNNER = Path(__file__).with_name("accept.py")
ACCEPTANCE_RE = re.compile(r"^\{unittest\}(\s+[A-Za-z_][\w.]*)+\s*$")

# Predicates whose result can differ between the raw emitted calls and the
# executed (repaired) ones. Only a task using one of these has an emission
# view distinct from its verdict; for every other task emission_pass is NULL
# rather than a copy of success.
VIEW_DEPENDENT_KEYS = frozenset({
    "expect_tool", "expect_tool_args_string", "expect_tool_args_present",
    "expect_tool_args_require", "prompt_not_json_blob", "forbid_bash_containing",
    "expect_tool_any", "expect_tools_all",
})

PASS, FAIL, UNSCORED, ERROR = "pass", "fail", "unscored", "error"
# The reply gave no ANSWER line AND its answer cannot be isolated. Not a wrong
# answer — a format miss, reported on its own (success NULL), so a model is
# never rated as failing a task it may have got right.
FORMAT_MISS = "format_miss"
FORMAT_MISS_REASON = "format miss: the committed answer cannot be read"


@dataclass
class Verdict:
    verdict: str                     # pass | fail | unscored | error
    success: bool | None             # None = not decided (unscored / error)
    emission_pass: bool | None       # the same checks over the RAW emitted calls
    source: str                      # which checks decided: predicates+acceptance+judge
    fail_reasons: list[str] = field(default_factory=list)
    acceptance: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None
    # For tasks checked on an ANSWER line: did the reply HAVE one? False keeps
    # "did not follow the answer format" apart from "answered wrong". None for
    # tasks without an answer line, and for runs that never finished.
    answer_format_ok: bool | None = None


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def _ok_tools(t: RunTranscript, view: str) -> set[str]:
    return {e.view_name(view) for e in t.tool_events if e.view_ok(view)}


def _load_cron_jobs() -> list[dict[str, Any]]:
    from prometheus.gateway.cron_service import load_cron_jobs

    return load_cron_jobs()


def _norm_cron(expr: str) -> str:
    return " ".join(str(expr).split()).lower()


def _cron_job_reasons(spec: dict[str, Any], jobs: list[dict[str, Any]]) -> list[str]:
    candidates = jobs
    if spec.get("name"):
        candidates = [j for j in jobs if j.get("name") == spec["name"]]
        if not candidates:
            return [
                f"no cron job named {spec['name']!r} "
                f"(jobs: {[j.get('name') for j in jobs] or 'none'})"
            ]

    def job_problems(job: dict[str, Any]) -> list[str]:
        problems = []
        sched = _norm_cron(job.get("schedule", ""))
        if "schedule_any" in spec and sched not in {
            _norm_cron(s) for s in spec["schedule_any"]
        }:
            problems.append(f"schedule {job.get('schedule')!r} not in {spec['schedule_any']!r}")
        if "schedule_regex" in spec and not re.fullmatch(
            spec["schedule_regex"], " ".join(str(job.get("schedule", "")).split())
        ):
            problems.append(
                f"schedule {job.get('schedule')!r} does not match {spec['schedule_regex']!r}"
            )
        for needle in _as_list(spec.get("command_contains")):
            if needle not in str(job.get("command", "")):
                problems.append(f"command {job.get('command')!r} lacks {needle!r}")
        if "enabled" in spec and bool(job.get("enabled", True)) != bool(spec["enabled"]):
            problems.append(f"enabled={job.get('enabled')!r}, wanted {spec['enabled']!r}")
        return problems

    best: list[str] | None = None
    for job in candidates:
        problems = job_problems(job)
        if not problems:
            return []
        if best is None or len(problems) < len(best):
            best = [f"job {job.get('name')!r}: {p}" for p in problems]
    return best or ["no cron job matched"]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _resolve(workspace: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else workspace / p


# THE ANSWER LINE. Tasks that check a value end their prompt with "End your
# reply with a final line `ANSWER: <...>`", so the check reads the model's
# COMMITTED answer and nothing else: whatever the reply discusses above it —
# distractors, a derivation, rows quoted from a file — cannot pass or fail
# the run. Searching free text for the right value cannot be both tight and
# fair (the WP-2.1 audit showed it failing both ways, round after round).
#
# FORMAT MISSES STAY SEPARATE FROM WRONG ANSWERS (Will, 2026-09-25). The reader
# credits a PASS only when the committed value is unambiguously right, and a
# FAIL only when it is unambiguously a wrong value of the answer's kind
# (``answer_shape``). Everything else — no line and no readable last line, a
# hedge with two candidate values, a value qualified by a parenthetical — is a
# FORMAT MISS: never credited, never counted as wrong.
_ANSWER_LINE = re.compile(
    r"(?:^|(?<=[.!?])[ \t]+)[ \t>*_`#-]*(?:final[ \t]+)?answer[ \t*_`]*[:=][ \t]*(?P<value>.*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_DECORATION = re.compile(r"^[\s*_`\"'“”‘’]+|[\s*_`\"'“”‘’]+$")
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_MATH_OPEN = re.compile(r"^\s*(?:\$\$|\\\[)\s*$")
_MATH_CLOSE = re.compile(r"^\s*(?:\$\$|\\\])\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
                          "\u2212": "-"})
# A short lead-in before a value: "So 1081.", "The answer is: 1081".
_LEAD_IN = re.compile(
    r"^(?:(?:so|therefore|thus|hence)[,:]?\s+)?"
    r"(?:(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is\b\s*[:=]?|:|=)\s*)?",
    re.IGNORECASE,
)

PASS_READ, FAIL_READ, MISS_READ = "pass", "fail", "miss"


def _normalize_value(value: str) -> str:
    """How a model spells a value it means literally. Every rule here removes
    PRESENTATION, never content — a rule that could turn a wrong value into a
    right one does not belong here (so a parenthetical is left in place: it
    may be commentary, or a qualifier like "(minutes)", or a hedge)."""
    value = value.translate(_HYPHENS)
    value = value.replace("`", "").replace("**", "")  # inline code / bold anywhere
    for _ in range(3):  # decoration nests either way round: **$16$**, $**16**$
        value = _DECORATION.sub("", value)
        if value.endswith("."):
            value = value[:-1].rstrip()
        # The prompt's own placeholder brackets, copied: ANSWER: <48217>
        if value.startswith("<") and value.endswith(">"):
            value = value[1:-1].strip()
        # LaTeX, as thinking models write answers: $...$, \(...\), \[...\],
        # \boxed{...}, \text{...}, and the escapes math mode needs.
        wrapped = re.fullmatch(r"(\$+)(.*?)\1", value)  # $...$ / $$...$$, both ends
        if wrapped:
            value = wrapped.group(2).strip()
        value = re.sub(r"^\\[(\[]|\\[)\]]$", "", value).strip()
        boxed = re.fullmatch(r"\\boxed\{(.*)\}", value)
        if boxed:
            value = boxed.group(1).strip()
        value = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}", r"\1", value)
        value = value.replace(r"\%", "%").replace(r"\$", "$").replace("{,}", ",")
        value = re.sub(r"\\[,; ]", " ", value).strip()
    return value


def _block_below(lines: list[str]) -> tuple[str, bool]:
    """The value written UNDER a bare label, and whether it was one plain line:
    a fenced block, display math or a list is read whole (joined); otherwise
    the next non-blank line."""
    lines = [ln for ln in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return "", True
    first = lines[0]
    if _FENCE.match(first) or _MATH_OPEN.match(first):
        closer = _FENCE if _FENCE.match(first) else _MATH_CLOSE
        body = []
        for ln in lines[1:]:
            if closer.match(ln):
                break
            body.append(ln.strip())
        return " ".join(b for b in body if b), False
    if _LIST_ITEM.match(first):
        items = []
        for ln in lines:
            if not _LIST_ITEM.match(ln):
                break
            items.append(_LIST_ITEM.sub("", ln).strip())
        return ", ".join(items), False
    return first, True


def answer_region(text: str) -> tuple[str, bool] | None:
    """The committed answer as written: the last ``ANSWER:`` line's value, or —
    when that label has nothing after it — the block below it. Returns
    (region, provisional); provisional means it is the one plain line below a
    bare label, which may open a derivation rather than state the answer. A
    list, fence or display math below the label is the answer as a whole.
    None when the reply has no answer label."""
    matches = list(_ANSWER_LINE.finditer(text or ""))
    if not matches:
        return None
    m = matches[-1]
    inline = _normalize_value(m.group("value"))
    if inline:
        return inline, False
    block, one_line = _block_below(text[m.end():].splitlines())
    return _normalize_value(block), one_line


def bare_region(text: str) -> str | None:
    """For a reply with no answer line: its last line, read as a candidate
    answer. A reply that ENDS in a multi-line fenced block is quoting something
    (a file, a log), and one that ends in a list of two or more items is
    listing — neither last line is an answer, so None."""
    lines = (text or "").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    before = next((ln for ln in reversed(lines[:-1]) if ln.strip()), "")
    if _LIST_ITEM.match(lines[-1]) and _LIST_ITEM.match(before):
        # A reply that ENDS in a list of two or more items is listing
        # candidates (or summing up): its last item alone would decide the
        # task by item order. Unreadable, like a quoted block.
        return None
    if _FENCE.match(lines[-1]):
        body: list[str] = []
        for ln in reversed(lines[:-1]):
            if _FENCE.match(ln):
                break
            if ln.strip():
                body.append(ln.strip())
        return _normalize_value(body[0]) if len(body) == 1 else None
    if _MATH_CLOSE.match(lines[-1]):
        body = []
        for ln in reversed(lines[:-1]):
            if _MATH_OPEN.match(ln):
                break
            if ln.strip():
                body.append(ln.strip())
        return _normalize_value(" ".join(reversed(body))) or None
    last = _LIST_ITEM.sub("", lines[-1])
    value = _normalize_value(_LEAD_IN.sub("", _normalize_value(last), count=1))
    return value or None


_HEDGE = re.compile(r"\b(?:or|either|and/or|maybe|possibly|perhaps|probably)\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:not|no|never|isn't|isnt|aren't|wasn't|don't|doesn't|cannot|can't)\b",
                       re.IGNORECASE)


def read_value(region: str | None, expect: str, shape: str | None, *, committed: bool) -> str:
    """PASS / FAIL / MISS for one region.

    ``committed``: the region is what the reply put on its answer line. There a
    clean value that is wrong is a wrong answer. Without a line the region is
    just the reply's last line — prose that may be a procedure, a question or a
    sign-off — so it can be CREDITED when it states the right value, and
    FAILED only when it is itself exactly a value of the answer's kind; any
    other prose is a format miss, never a wrong answer."""
    if not region:
        return MISS_READ
    region = _normalize_value(_LEAD_IN.sub("", region, count=1)) or region
    if re.fullmatch(expect, region, re.IGNORECASE):
        return PASS_READ
    if shape and re.fullmatch(shape, region, re.IGNORECASE):
        return FAIL_READ
    inner = re.fullmatch(r"\(([^()]+)\)", region)
    if inner:  # the whole value in one pair — a tuple's own brackets, not a qualifier
        return read_value(inner.group(1), expect, shape, committed=committed)
    # A parenthetical may be commentary, a qualifier ("(minutes)") or a hedge
    # ("(or Jupiter)"); a hedge commits to nothing. Unreadable.
    if "(" in region or ")" in region or _HEDGE.search(region):
        return MISS_READ
    candidates = set()
    if shape:
        candidates = {
            _normalize_value(m.group(0)).lower()
            for m in re.finditer(rf"(?<!\w)(?:{shape})(?!\w)", region, re.IGNORECASE)
            if m.group(0).strip()
        }
    if _NEGATION.search(region):
        # "It is not 48217" commits to nothing. But a committed line that names
        # no value of the answer's kind and denies there is one ("No request ID
        # found", "... does not contain a definition") is a wrong answer where
        # the task has one — it was not missed, it was answered wrongly.
        return FAIL_READ if committed and shape and not candidates else MISS_READ
    if len(candidates) == 1:
        (only,) = candidates
        if re.fullmatch(expect, only, re.IGNORECASE):
            return PASS_READ
        return FAIL_READ if committed else MISS_READ
    if committed and not candidates and not re.search(r"\s", region):
        # One clean token committed on the answer line, of the wrong kind
        # altogether (a commit hash where a build tag was asked): wrong.
        return FAIL_READ
    return MISS_READ


def read_answer(text: str, expect: str, shape: str | None) -> tuple[str, bool, str]:
    """(PASS/FAIL/MISS, had_answer_line, what was read).

    The answer line decides when it holds a value. One plain line under a bare
    label that yields no clean value falls back to the reply's last line (a
    derivation under the label), as does a reply with no label at all. A list,
    fence or display math under the label never falls back: its last line is
    part of it, and reading it would decide a list of candidates by order."""
    found = answer_region(text)
    if found is not None:
        region, provisional = found
        result = read_value(region, expect, shape, committed=True)
        if result != MISS_READ or not provisional:
            return result, True, region
    bare = bare_region(text)
    return read_value(bare, expect, shape, committed=False), False, bare or ""


def ladder_predicates(
    spec: dict[str, Any],
    t: RunTranscript,
    workspace: Path,
    view: str = EXECUTION,
) -> list[str]:
    """The ladder's own predicates. Returns failure reasons (empty = pass)."""
    reasons: list[str] = []
    text = t.final_text or ""
    low = text.lower()
    shown = text[:160]

    if "expect_answer" in spec:
        result, line, read = read_answer(text, spec["expect_answer"], spec.get("answer_shape"))
        where = "ANSWER" if line else "no 'ANSWER:' line; the last line"
        if result == FAIL_READ:
            reasons.append(f"{where} {read[:120]!r} is a wrong value "
                           f"(not /{spec['expect_answer']}/)")
        elif result == MISS_READ:
            reasons.append(f"{FORMAT_MISS_REASON} ({where.lower()} reads {read[:120]!r})")

    if "expect_text_any" in spec:
        wants = _as_list(spec["expect_text_any"])
        if not any(w.lower() in low for w in wants):
            reasons.append(f"final text has none of {wants!r} (got: {shown!r})")
    for w in _as_list(spec.get("expect_text_all")):
        if w.lower() not in low:
            reasons.append(f"final text lacks {w!r} (got: {shown!r})")
    if "expect_text_regex" in spec:
        if not re.search(spec["expect_text_regex"], text, re.IGNORECASE | re.DOTALL):
            reasons.append(
                f"final text does not match /{spec['expect_text_regex']}/ (got: {shown!r})"
            )
    for w in _as_list(spec.get("forbid_text")):
        if w.lower() in low:
            reasons.append(f"final text contains forbidden {w!r}")
    for pattern in _as_list(spec.get("forbid_text_regex")):
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            reasons.append(f"final text matches forbidden /{pattern}/")

    ok = _ok_tools(t, view)
    if "expect_tool_any" in spec:
        wants = _as_list(spec["expect_tool_any"])
        if not ok & set(wants):
            reasons.append(
                f"no successful call of any of {wants!r} "
                f"(attempted: {[e.view_name(view) for e in t.tool_events] or 'none'})"
            )
    for tool in _as_list(spec.get("expect_tools_all")):
        if tool not in ok:
            reasons.append(f"no successful {tool!r} call")

    if "expect_file_regex" in spec:
        fr = spec["expect_file_regex"]
        p = _resolve(workspace, fr["path"])
        if not p.is_file():
            reasons.append(f"expected {fr['path']} to be a regular file")
        else:
            try:
                content = p.read_text(errors="replace")
            except OSError as exc:
                reasons.append(f"could not read {fr['path']}: {type(exc).__name__}")
            else:
                if not re.search(fr["pattern"], content, re.MULTILINE):
                    reasons.append(f"file {fr['path']} does not match /{fr['pattern']}/")
    for rel in _as_list(spec.get("expect_file_absent")):
        if _resolve(workspace, rel).exists():
            reasons.append(f"file {rel} should not exist")

    if "expect_cron_job" in spec or "forbid_cron_job" in spec:
        jobs = _load_cron_jobs()
        if "expect_cron_job" in spec:
            reasons.extend(_cron_job_reasons(spec["expect_cron_job"], jobs))
        for name in _as_list(spec.get("forbid_cron_job")):
            if any(j.get("name") == name for j in jobs):
                reasons.append(f"cron job {name!r} should not exist")
    return reasons


def check_predicates(
    spec: dict[str, Any],
    t: RunTranscript,
    workspace: Path,
    view: str = EXECUTION,
) -> tuple[bool, list[str]]:
    """Gym predicates AND ladder predicates, under one view."""
    gym_part = {k: v for k, v in spec.items() if k in GYM_SCORE_KEYS}
    reasons: list[str] = []
    if gym_part:
        try:
            _ok, gym_reasons = gym_score(gym_part, t, workspace, view=view)
        except OSError as exc:
            # e.g. the model left a directory where expect_file_contains reads
            gym_reasons = [f"expected file could not be read: {type(exc).__name__}"]
        reasons.extend(gym_reasons)
    reasons.extend(ladder_predicates(spec, t, workspace, view=view))
    return (not reasons, reasons)


# ---------------------------------------------------------------------------
# Acceptance command
# ---------------------------------------------------------------------------


def acceptance_argv(command: str, workspace: Path, result_path: Path) -> list[str]:
    """``{unittest} a b`` → the isolated runner's argv. Raises ValueError on
    any other form: an arbitrary shell command's exit status proves nothing."""
    if not ACCEPTANCE_RE.match(command):
        raise ValueError(f"acceptance must be '{{unittest}} <modules>', got {command!r}")
    modules = shlex.split(command)[1:]
    return [sys.executable, "-I", "-B", str(ACCEPT_RUNNER), str(workspace),
            str(result_path), *modules]


def _clear_the_way(workspace: Path, rel: str) -> None:
    """Make ``workspace/rel`` writable as a plain file, whatever the agent left.

    A symlink, a directory, a read-only file, or a package directory that would
    shadow the module (``test_x/`` next to ``test_x.py``) are all removed. Only
    paths inside the workspace are touched.
    """
    import shutil
    import stat

    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError(f"{rel!r} escapes the workspace")
    ws = workspace.resolve()
    target = workspace / rel
    # Every parent must be a real directory inside the workspace.
    cur = workspace
    for part in Path(rel).parts[:-1]:
        cur = cur / part
        if cur.is_symlink() or (cur.exists() and not cur.is_dir()):
            cur.unlink()
        cur.mkdir(exist_ok=True)
        cur.chmod(cur.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
    if ws not in target.parent.resolve().parents and target.parent.resolve() != ws:
        raise ValueError(f"{rel!r} escapes the workspace")  # belt and braces
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.chmod(target.stat().st_mode | stat.S_IWUSR)
    if target.suffix == ".py":
        shadow = target.with_suffix("")
        if shadow.is_symlink():
            shadow.unlink()
        elif shadow.is_dir():
            shutil.rmtree(shadow)


def run_acceptance(
    command: str,
    acceptance_files: dict[str, str],
    workspace: Path,
    *,
    home: Path | None = None,
    timeout_s: float = ACCEPTANCE_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the acceptance tests. ``status`` is ``ran`` or ``not_run``.

    ``not_run`` means the HARNESS could not execute the check (no interpreter,
    no runner) — recorded UNSCORED, never as a pass or a model failure.
    Everything the model's workspace does to the check — a test file it
    replaced with a symlink, code that exits at import, a hang — ran, and
    failed.
    """
    import uuid

    from prometheus.gym.ladder.fixtures import SANDBOX_ENV_KEYS

    base = {"command": command, "status": "ran", "passed": False}
    for rel, content in acceptance_files.items():
        try:
            _clear_the_way(workspace, rel)
            (workspace / rel).write_text(content)
        except (OSError, ValueError) as exc:
            return {**base, "error": f"could not restore acceptance file {rel!r} "
                                     f"over what the run left: {type(exc).__name__}: {exc}"}

    result_path = (home or workspace.parent) / f".acceptance-{uuid.uuid4().hex}.json"
    try:
        argv = acceptance_argv(command, workspace, result_path)
    except ValueError as exc:
        log.warning("ladder acceptance: %s", exc)
        return {**base, "status": "not_run", "passed": None, "error": str(exc)}
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": str(home or workspace),
        "PYTHONHASHSEED": "0",
        **{k: os.environ[k] for k in SANDBOX_ENV_KEYS if k in os.environ},
    }
    try:
        proc = subprocess.run(argv, cwd=workspace, env=env, capture_output=True,
                              text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result_path.unlink(missing_ok=True)
        return {**base, "exit_code": None, "error": f"timed out after {timeout_s:.0f}s"}
    except OSError as exc:
        log.warning("ladder acceptance: could not start the runner: %s", exc)
        return {**base, "status": "not_run", "passed": None,
                "error": f"{type(exc).__name__}: {exc}"}
    tail = ((proc.stdout or "") + (proc.stderr or ""))[-800:]
    try:
        import json as _json

        result = _json.loads(result_path.read_text())
    except (OSError, ValueError):
        return {**base, "exit_code": proc.returncode, "output_tail": tail,
                "error": "no result: the test process ended before reporting "
                         "(code under test exited the interpreter?)"}
    finally:
        result_path.unlink(missing_ok=True)
    ran = int(result.get("tests_run", 0)) - int(result.get("skipped", 0))
    passed = bool(result.get("ok")) and ran > 0
    error = result.get("error") or (None if ran > 0 else "no tests ran")
    return {**base, "passed": passed, "exit_code": proc.returncode,
            "tests_run": result.get("tests_run"), "failures": result.get("failures"),
            "errors": result.get("errors"), "skipped": result.get("skipped"),
            "error": error, "output_tail": tail}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


async def run_judge(
    judge: Any, task: Any, t: RunTranscript, *, prompt: str | None = None
) -> dict[str, Any]:
    """Grade with the pinned judge. ``status`` is ``ok`` or ``unavailable``.

    ``PrometheusJudge`` turns an empty or unparseable reply into a score
    (0.0 for empty, a stray number for garbage). Taken at face value that is
    a judge outage recorded as a model failure — or, for a stray "1", as a
    pass. Both are refused here: only a parsed verdict counts.
    """
    spec = task.judge
    threshold = float(spec.get("threshold", 0.7))
    if judge is None:
        return {"status": "unavailable", "error": "judge disabled for this run",
                "threshold": threshold, "provenance": None}
    expected = f"{spec['rubric'].strip()}\n\nReference answer:\n{spec['reference'].strip()}"
    try:
        v = await judge.evaluate(
            task_input=prompt or task.prompt,
            agent_output=t.final_text or "",
            expected_behavior=expected,
            tool_trace=[{"tool_name": e.exec_name} for e in t.tool_events],
        )
    except Exception as exc:  # noqa: BLE001 — any judge failure is "unavailable"
        log.warning("ladder judge unavailable for %s: %s: %s", task.id, type(exc).__name__, exc)
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}",
                "threshold": threshold, "provenance": _provenance(judge)}
    raw = (v.raw_response or "").strip()
    score = strict_judge_score(raw)
    if score is None:
        log.warning("ladder judge returned no 0-1 score for %s: %r", task.id, raw[:200])
        return {"status": "unavailable", "error": "judge reply is not a score on the 0-1 scale",
                "raw": raw[:300], "threshold": threshold, "provenance": _provenance(judge)}
    return {"status": "ok", "score": score, "passed": score >= threshold,
            "threshold": threshold, "reasoning": v.reasoning[:500],
            "provenance": _provenance(judge)}


def _provenance(judge: Any) -> dict[str, Any]:
    """Who graded — model and whether it was pinned. The endpoint is left out:
    telemetry.db is backed up and copied between machines."""
    prov = judge.provenance() if judge is not None else {}
    return {"model": prov.get("model"), "pinned": prov.get("pinned")}


def strict_judge_score(raw: str) -> float | None:
    """The judge's own number, or None. ``PrometheusJudge`` defaults a missing
    score to 0.0 and clamps an out-of-range one — a malformed reply would read
    as a model FAIL, a score of 7 as a PASS. Only a finite number in [0, 1]
    under a ``score`` key counts."""
    import json as _json
    import math

    candidates = [raw, re.sub(r"```(?:json)?\s*\n?", "", raw).strip()]
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.index("{"): raw.rindex("}") + 1])
    for text in candidates:
        try:
            obj = _json.loads(text)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        s = obj.get("score")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            return None
        s = float(s)
        return s if math.isfinite(s) and 0.0 <= s <= 1.0 else None
    return None


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


async def decide(
    task: Any,
    t: RunTranscript,
    workspace: Path,
    *,
    judge: Any = None,
    harness_error: str = "",
    halted: str = "",
    home: Path | None = None,
    prompt: str | None = None,
) -> Verdict:
    """The run's verdict.

    ``harness_error``: the run could not be carried out (provider down, harness
    bug) — ERROR, success NULL. ``halted``: the run ended without the model
    finishing — out of time, rounds or tool calls, or stopped by the loop
    (repeat halt, circuit breaker, empty replies) — FAIL with that reason,
    whatever the workspace looks like.
    """
    source = task.verdict_source
    if harness_error:
        return Verdict(ERROR, None, None, source, [harness_error])
    if halted:
        return Verdict(FAIL, False, False if _has_view(task) else None, source, [halted])

    reasons: list[str] = []
    exec_ok = True
    emit_ok: bool | None = None
    fmt = (read_answer(t.final_text or "", task.score["expect_answer"],
                       task.score.get("answer_shape"))[1]
           if "expect_answer" in (task.score or {}) else None)
    if task.score:
        exec_ok, reasons = check_predicates(task.score, t, workspace, view=EXECUTION)
        if _has_view(task):
            emit_ok, _ = check_predicates(task.score, t, workspace, view=EMISSION)

    acceptance = None
    if task.acceptance:
        acceptance = run_acceptance(task.acceptance, task.acceptance_files, workspace, home=home)
        if acceptance["status"] == "not_run":
            return Verdict(UNSCORED, None, None, source,
                           reasons + [f"acceptance not run: {acceptance['error']}"],
                           acceptance=acceptance, answer_format_ok=fmt)
        if not acceptance["passed"]:
            exec_ok = False
            emit_ok = False if emit_ok is not None else None
            reasons.append(
                f"acceptance failed ({acceptance.get('tests_run') or 0} run, "
                f"{acceptance.get('failures') or 0} failed, {acceptance.get('errors') or 0} errors"
                f"{'; ' + acceptance['error'] if acceptance.get('error') else ''})"
            )

    if not exec_ok and reasons and all(r.startswith(FORMAT_MISS_REASON) for r in reasons):
        # The ONLY thing wrong is that the answer could not be read.
        return Verdict(FORMAT_MISS, None, None, source, reasons,
                       acceptance=acceptance, answer_format_ok=fmt)
    if not exec_ok:
        return Verdict(FAIL, False, emit_ok, source, reasons, acceptance=acceptance, answer_format_ok=fmt)

    judged = None
    if task.judge:
        judged = await run_judge(judge, task, t, prompt=prompt)
        if judged["status"] != "ok":
            return Verdict(UNSCORED, None, None, source,
                           [f"judge unavailable: {judged.get('error')}"],
                           acceptance=acceptance, judge=judged, answer_format_ok=fmt)
        if not judged["passed"]:
            return Verdict(FAIL, False, False if emit_ok is not None else None, source,
                           [f"judge score {judged['score']:.2f} < {judged['threshold']:.2f}: "
                            f"{judged.get('reasoning', '')[:160]}"],
                           acceptance=acceptance, judge=judged, answer_format_ok=fmt)

    return Verdict(PASS, True, emit_ok, source, [], acceptance=acceptance, judge=judged, answer_format_ok=fmt)


def _has_view(task: Any) -> bool:
    return bool(set(task.score or {}) & VIEW_DEPENDENT_KEYS)
