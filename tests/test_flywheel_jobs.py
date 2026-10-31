"""The weekly fine-tuning flywheel (``prometheus.jobs.flywheel_weekly``).

Side-effect assertions: a recording fake runner stands in for every child
process (builder, trainer, quantizer, scorer, setups.py) and a fake clock puts
the run inside or outside the Saturday window. No model, no GPU, no network.
The two ``run_command`` tests start a plain ``sys.executable -c`` child, which
ends on SIGTERM or SIGKILL (never a crash signal).
"""

from __future__ import annotations

import ast
import contextlib
import csv
import importlib.util
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

import pytest
import yaml

import prometheus.jobs.flywheel_weekly as fw

ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]
SAT = datetime(2026, 10, 31, 2, 0, 30, tzinfo=ET)        # inside the window (EDT)
NEXT_SAT = datetime(2026, 11, 7, 2, 0, 30, tzinfo=ET)    # a week on, after the DST change
FRIDAY = datetime(2026, 10, 30, 14, 0, tzinfo=ET)
# The Q6b allow; the block fixture sets the session exclusion list it needs.
REF = "operator Q6b test"
APPROVED = {"max_derived": {"approved": True, "approval_ref": REF}}
TAG = "max-derived"


# --------------------------------------------------------------------------- #
# Fakes and builders
# --------------------------------------------------------------------------- #


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now.astimezone(timezone.utc)


class Runner:
    """Records every command. A handler per log name (build, train, quantize,
    score, score-compare, gpu-setup, gpu-restore) decides each result."""

    def __init__(self, **handlers: Callable[[fw.CommandSpec], Any]) -> None:
        self.handlers = handlers
        self.calls: list[tuple[str, fw.CommandSpec]] = []

    def __call__(self, spec: fw.CommandSpec) -> fw.CommandResult:
        name = spec.log_path.stem.split("-", 3)[-1]   # "<YYYY-MM-DD>-<name>"
        self.calls.append((name, spec))
        handler = self.handlers.get(name.replace("-", "_"))
        out = handler(spec) if handler else 0
        return out if isinstance(out, fw.CommandResult) else fw.CommandResult(int(out))

    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def argv(self, name: str) -> list[str]:
        return next(s.argv for n, s in self.calls if n == name)


def arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def trace(i: int, model: str = "qwen3.8-max", **meta: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"task {i}"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": f"echo {i}"})},
            }]},
        ],
        "tools": [{"type": "function", "function": {"name": "bash",
                                                    "parameters": {"type": "object"}}}],
        "_meta": {"model": model, "tool_name": "bash", "timestamp": 1.0 + i,
                  "session_id": "s", **meta},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def summary(label: str, task: float, recall: float) -> dict[str, Any]:
    return {"schema": "distill-score-summary/2", "label": label, "served_model": f"{label}.gguf",
            "sets": {"eval-tools": {"metrics": {"task_success": {"k": 1, "n": 2, "rate": task}}},
                     "eval-longctx": {"metrics": {"recall_by_depth": {
                         "32k": {"k": 1, "n": 2, "rate": recall}}}}}}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in ("scripts/distill/build.py", fw.SCORE_SCRIPT, fw.SETUPS_SCRIPT):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("# stand-in: the fake runner never executes it\n")
    return root


@pytest.fixture
def block(tmp_path: Path) -> dict[str, Any]:
    cases = write_jsonl(tmp_path / "cases" / "eval-tools.jsonl", [{"id": "c1"}])
    baseline = tmp_path / "baseline.summary.json"
    baseline.write_text(json.dumps(summary("base-9b", 0.40, 0.90)))
    sessions = tmp_path / "exclude-sessions.txt"
    sessions.write_text("# the incident session\nsess-incident\n")
    return {
        "enabled": True, "data_dir": str(tmp_path / "fw"),
        "max_derived": {"exclude_sessions_file": str(sessions)},
        "harvest": {"golden": [str(tmp_path / "golden" / "*.jsonl")],
                    "extra": [str(tmp_path / "extra" / "*.jsonl")],
                    "student_models": ["student-*"]},
        "build": {"args": ["--pack", "pack.json", "--tokenizer-dir", "tok"]},
        "train": {"min_new_traces": 2,
                  "command": ["{python}", "train.py", "--data", "{train}", "--val", "{val}",
                              "--out", "{run_dir}", "--stop-at", "{stop_at}"],
                  "resume_args": ["--resume"]},
        "quantize": {"command": ["{python}", "quantize.py", "--adapter", "{adapter_dir}",
                                 "--data", "{data_dir}", "--out", "{out_dir}"]},
        "score": {"base_url": "http://127.0.0.1:8080", "cases": [str(cases)],
                  "baseline_summary": str(baseline)},
        "gpu_setup": "none",
        "spend_ledger": str(tmp_path / "ledger.csv"),
    }


@pytest.fixture(autouse=True)
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The scoring server's ``GET /v1/models`` id. By default it serves the
    newest quantized model, named by its path as llama-server does; a test sets
    ``id`` to serve something else ("" = nothing readable). No network."""
    probe: dict[str, Any] = {"id": None}

    def served_model_id(base_url: str, timeout: float = 10.0) -> str | None:
        if probe["id"] is not None:
            return probe["id"] or None
        models = sorted((tmp_path / "fw" / "models").glob("*"))
        return f"{models[-1]}/{models[-1].name}-Q4_K_M.gguf" if models else None

    monkeypatch.setattr(fw, "served_model_id", served_model_id)
    return probe


def write_config(tmp_path: Path, block: dict[str, Any]) -> Path:
    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump({"flywheel": block}), encoding="utf-8")
    return path


def golden(tmp_path: Path, n: int = 3, **meta: Any) -> None:
    write_jsonl(tmp_path / "golden" / "golden_traces_1.jsonl",
                [trace(i, **meta) for i in range(n)])


def build_ok(*, source: str = "max-golden", teacher: str = "qwen3.8-max",
             label: str = "tool-success", rows: int = 3, gates: str = "PASS",
             tag: str | None = None) -> Callable[[fw.CommandSpec], fw.CommandResult]:
    """A builder that writes the split files and report.json build.py writes
    (a PASS includes the spot check). Row ids carry the run id unless *tag*
    fixes them, so each week's build holds rows no earlier train had. Given
    build.py's Q6b allow it records the approval as build.py does."""
    def handler(spec: fw.CommandSpec) -> fw.CommandResult:
        out = Path(arg(spec.argv, "--out"))
        prefix = tag or out.parent.name
        train = [{"id": f"p3-{prefix}-{i}", "split": "train", "source": source,
                  "label": label} for i in range(rows)]
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", train[:1])
        write_jsonl(out / "val-cases.jsonl", [{"id": "v"}])
        write_jsonl(out / "eval-tools.jsonl", [{"id": "e"}])
        report: dict[str, Any] = {
            "settings": {"teacher_model": teacher},
            "gates": {"train_size": {"pass": gates == "PASS"},
                      "spot_check": {"status": "PASS" if gates == "PASS" else "NOT RUN"},
                      "overall": gates},
            "inputs": {"written": {"train.jsonl": rows, "val.jsonl": 1,
                                   "eval-tools.jsonl": 1}}}
        if fw.BUILD_ALLOW_FLAG in spec.argv:
            assert TAG in out.name                      # build.py's rule
            report["data_policy"] = {"max_derived_allow": {
                "allowed": True, "approval": arg(spec.argv, fw.BUILD_APPROVAL_FLAG)}}
        (out / "report.json").write_text(json.dumps(report))
        return fw.CommandResult(0 if gates == "PASS" else 1)
    return handler


def train_finishes(spec: fw.CommandSpec) -> int:
    """train_qlora.py's layout: the adapter is tagged when the run is."""
    run_dir = Path(arg(spec.argv, "--out"))
    tagged = TAG in run_dir.name
    assert tagged == (fw.MAX_DERIVED_APPROVAL_FLAG in spec.argv)
    (run_dir / ("adapter-max-derived" if tagged else "adapter")).mkdir(parents=True,
                                                                        exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "done"}))
    return 0


def train_paused(spec: fw.CommandSpec) -> fw.CommandResult:
    run_dir = Path(arg(spec.argv, "--out"))
    (run_dir / "checkpoint-200").mkdir(parents=True, exist_ok=True)
    return fw.CommandResult(0, stopped=True)


def score_ok(spec: fw.CommandSpec) -> int:
    out = Path(arg(spec.argv, "--summary-out"))
    label = arg(spec.argv, "--label")
    out.write_text(json.dumps(summary(label, 0.55, 0.85)))
    return 0


def run_main(tmp_path: Path, block: dict[str, Any], runner: Runner, repo: Path,
             now: datetime = SAT, *extra: str) -> int:
    return fw.main(["--config", str(write_config(tmp_path, block)), *extra],
                   clock=Clock(now), runner=runner, repo=repo)


def report(tmp_path: Path, kind: str = "") -> dict[str, Any]:
    path = tmp_path / "fw" / "reports" / kind / "LATEST.json"
    return json.loads(path.read_text())


def stage(rep: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in rep["stages"] if s["name"] == name)


def make_run(block: dict[str, Any], repo: Path, *, now: datetime = SAT,
             runner: Runner | None = None, state: dict[str, Any] | None = None,
             clock: Callable[[], datetime] | None = None) -> fw.Run:
    cfg = fw.parse_config(block)
    start, end = fw.window_bounds(now, cfg.window) or (now, now + cfg.window.length)
    return fw.Run(cfg=cfg, run_id=start.date().isoformat(), window_start=start,
                  window_end=end, dry_run=False, clock=clock or Clock(now),
                  runner=runner or Runner(), repo=repo,
                  state=state if state is not None else {"schema": fw.STATE_SCHEMA})


# --------------------------------------------------------------------------- #
# Config: validated at load
# --------------------------------------------------------------------------- #


def test_the_template_block_is_valid_off_and_states_the_code_defaults():
    data = yaml.safe_load((REPO / "config" / "prometheus.yaml.default").read_text())
    cfg = fw.parse_config(data["flywheel"])
    assert cfg.enabled is False
    assert cfg.max_derived == fw.MaxDerivedAllow() and not cfg.max_derived.complete
    assert cfg.teacher_models == fw.DEFAULT_TEACHER_MODELS == ("qwen3.8-max", "qwen3.8-27b*")
    assert cfg.window.describe() == "Sat 02:00-06:00 America/New_York"
    # The template documents the defaults: writing it changes nothing.
    assert cfg == fw.parse_config(None)


@pytest.mark.parametrize("over, needle", [
    ({"enabeld": True}, "flywheel.enabeld is not a flywheel setting"),
    ({"window": {"start": 120}}, "quoted"),
    ({"window": {"timezone": "Mars/Olympus"}}, "time zone"),
    ({"window": {"start": "06:00", "end": "02:00"}}, "after window.start"),
    ({"window": {"day": "someday"}}, "window.day"),
    ({"train": {"command": ["train.py", "--out", "{outdir}"]}}, "unknown placeholder"),
    ({"train": {"resume_args": ["--resume"]}}, "train.command is empty"),
    ({"train": {"min_new_traces": 0}}, "min_new_traces"),
    ({"quantize": {"command": ["q.py", "{train}"]}}, "unknown placeholder"),
    ({"max_derived": {"approved": True, "exclude_sessions_file": "/x"}},
     "approval_ref is empty"),
    ({"max_derived": {"approved": True, "approval_ref": "op"}}, "exclude_sessions_file is empty"),
    ({"max_derived": {"approved": "yes"}}, "true or false"),
    ({"max_derived": {"approved_by": "op"}}, "max_derived.approved_by is not a flywheel setting"),
    # the old block is named, not silently ignored
    ({"max_data": {"approved": True}}, "flywheel.max_data was replaced by flywheel.max_derived"),
    ({"build": {"args": ["--out", "x"]}}, "may not set --out"),
    ({"build": {"args": ["--allow-max-derived"]}}, "may not set --allow-max-derived"),
    ({"build": {"args": ["--max-derived-approval=x"]}}, "may not set --max-derived-approval"),
    ({"build": {"args": ["--exclude-sessions", "f"]}}, "may not set --exclude-sessions"),
    ({"gpu_setup": "business"}, "gpu_setup"),
    ({"score": {"base_url": "127.0.0.1:8080"}}, "base_url"),
    ({"harvest": {"golden": "not-a-list"}}, "list of non-empty strings"),
])
def test_a_bad_block_is_refused_naming_the_key(over, needle):
    with pytest.raises(fw.FlywheelConfigError) as exc:
        fw.parse_config(over)
    assert any(needle in p for p in exc.value.problems), exc.value.problems


def test_every_problem_is_reported_at_once():
    with pytest.raises(fw.FlywheelConfigError) as exc:
        fw.parse_config({"enabeld": True, "window": {"start": 120}})
    assert len(exc.value.problems) == 2


def test_a_recorded_approval_parses(block, tmp_path):
    cfg = fw.parse_config(merge(block, APPROVED))
    assert cfg.max_derived.complete and cfg.max_derived.approval_ref == REF
    assert cfg.max_derived.exclude_sessions_file == tmp_path / "exclude-sessions.txt"
    # The list alone is no allow; the allow alone is no config.
    assert not fw.parse_config(block).max_derived.complete


# --------------------------------------------------------------------------- #
# Stage order, the happy path, the report
# --------------------------------------------------------------------------- #


def test_stages_run_in_order_and_the_report_is_written(tmp_path, block, repo):
    golden(tmp_path, 3)
    runner = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    code = run_main(tmp_path, merge(block, APPROVED), runner, repo)

    assert code == fw.EXIT_OK
    assert runner.names() == ["build", "train", "quantize", "score", "score-compare"]
    rep = report(tmp_path)
    assert [s["name"] for s in rep["stages"]] == list(fw.STAGES)
    assert {s["name"]: s["status"] for s in rep["stages"]} == dict.fromkeys(fw.STAGES, fw.OK)
    # Max-derived rows trained under the Q6b allow: every name carries the tag
    # (artifact_policy), the trainer and quantizer get the operator's flag.
    version = "fw-2026-10-31-max-derived"
    build = runner.argv("build")
    assert Path(arg(build, "--out")).name == "sets-max-derived"
    assert arg(build, fw.BUILD_APPROVAL_FLAG) == REF and fw.BUILD_ALLOW_FLAG in build
    assert arg(build, fw.BUILD_EXCLUDE_FLAG) == str(tmp_path / "exclude-sessions.txt")
    assert Path(arg(runner.argv("train"), "--out")).name == version
    assert fw.MAX_DERIVED_APPROVAL_FLAG in runner.argv("train")
    q = runner.argv("quantize")
    assert Path(arg(q, "--adapter")).name == "adapter-max-derived"
    assert Path(arg(q, "--out")).name == version and fw.MAX_DERIVED_APPROVAL_FLAG in q
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert state["adapters"][version]["max_derived"] is True
    assert state["quantized"][version]["max_derived"] is True
    private = {i["kind"]: i for i in rep["guards"]["private_artifacts"]["items"]}
    assert {"train data", "train run", "adapter", "model"} <= set(private)
    assert all(private[k]["tagged"] for k in ("train data", "train run", "adapter", "model"))
    # wins and losses vs the previous model (the baseline summary here)
    assert rep["model"]["candidate"] == version
    assert [w["metric"] for w in rep["model"]["wins"]] == ["eval-tools / task_success"]
    assert [x["metric"] for x in rep["model"]["losses"]] == ["eval-longctx / recall_by_depth @32k"]
    reports = tmp_path / "fw" / "reports"
    assert (reports / "LATEST.md").read_text() == (reports / "2026-10-31-weekly.md").read_text()
    md = (reports / "LATEST.md").read_text()
    assert "Wins (1)" in md and "Losses (1)" in md and "## Spend" in md


def test_the_build_gets_the_golden_sources_and_the_verified_pool(tmp_path, block, repo):
    golden(tmp_path, 1)
    write_jsonl(tmp_path / "extra" / "gym.jsonl",
                [trace(9, verdict="pass", verified_by="judge-27b")])
    runner = Runner(build=build_ok())
    run_main(tmp_path, block, runner, repo)
    argv = runner.argv("build")
    assert argv[2] == "build"
    # the golden source reaches the builder as its verified copy
    assert arg(argv, "--traces") == str(tmp_path / "fw" / "build-input" / "golden"
                                        / "golden_traces_1.jsonl")
    assert arg(argv, "--extra-traces") == str(tmp_path / "fw" / "pool" / "2026-10-31.jsonl")
    assert arg(argv, "--out") == str(tmp_path / "fw" / "runs" / "2026-10-31" / "sets")


def test_the_builder_reads_only_the_golden_traces_verify_passed(tmp_path, block, repo):
    original = write_jsonl(tmp_path / "golden" / "golden_traces_1.jsonl", [
        trace(0),                                         # the teacher's: verified
        trace(1, adapter_version="fw-2026-10-24"),        # offspring under the teacher's id
        trace(2, label="fail"),
    ])
    before = original.read_bytes()
    runner = Runner(build=build_ok())
    run = make_run(block, repo, runner=runner)
    assert fw.stage_harvest(run).status == fw.OK and fw.stage_verify(run).status == fw.OK
    # exported between this window's harvest and its build: never verified
    write_jsonl(tmp_path / "golden" / "golden_traces_2.jsonl", [trace(3)])

    result = fw.stage_build(run)
    assert result.status == fw.OK
    argv = runner.argv("build")
    copy = tmp_path / "fw" / "build-input" / "golden"
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--traces"] == [
        str(copy / "golden_traces_1.jsonl"), str(copy / "golden_traces_2.jsonl")]
    lines = (copy / "golden_traces_1.jsonl").read_bytes().splitlines(keepends=True)
    assert lines[0] == before.splitlines(keepends=True)[0]          # byte for byte
    withheld = [json.loads(x)["_meta"] for x in lines[1:]]           # same lines, stubs
    assert withheld == [{"model": fw.WITHHELD_MODEL, "flywheel_withheld": "unverified_offspring"},
                        {"model": fw.WITHHELD_MODEL, "flywheel_withheld": "verdict_fail"}]
    late = json.loads((copy / "golden_traces_2.jsonl").read_text())
    assert late["_meta"] == {"model": fw.WITHHELD_MODEL, "flywheel_withheld": "not_harvested"}
    assert result.details["golden"] == {"files": 2, "kept": 1, "withheld": {
        "unverified_offspring": 1, "verdict_fail": 1, "not_harvested": 1}}
    assert original.read_bytes() == before                          # the source is untouched


def test_a_builder_crash_on_a_rerun_does_not_read_the_last_attempts_report(
        tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, block, Runner(build=build_ok(gates="FAIL")), repo)
    assert stage(report(tmp_path), "build")["status"] == fw.REFUSED
    # The operator changes the builder's arguments; the rerun's builder crashes
    # (exit 1, as an uncaught exception exits) before writing a report.
    changed = merge(block, {"build": {"args": ["--pack", "pack.json", "--tokenizer-dir", "t2"]}})
    runner = Runner(build=lambda spec: 1)
    run_main(tmp_path, changed, runner, repo, SAT + timedelta(minutes=20))
    assert runner.names() == ["build"]
    build = stage(report(tmp_path), "build")
    assert build["status"] == fw.FAILED and "exited 1 without a report" in build["reason"]
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert "sets" not in state            # next week builds again instead of trusting it


@pytest.mark.parametrize("gates, code", [
    # build.py without --spot-check-server
    ({"train_size": {"pass": True}, "spot_check": {"status": "NOT RUN"},
      "overall": "INCOMPLETE"}, 1),
    # a builder whose PASS records no spot check
    ({"train_size": {"pass": True}, "overall": "PASS"}, 0),
])
def test_sets_without_a_passed_spot_check_are_never_trained_on(tmp_path, block, repo,
                                                                gates, code):
    golden(tmp_path, 3)

    def builder(spec: fw.CommandSpec) -> int:
        build_ok()(spec)
        out = Path(arg(spec.argv, "--out"))
        written = json.loads((out / "report.json").read_text())
        (out / "report.json").write_text(json.dumps({**written, "gates": gates}))
        return code

    runner = Runner(build=builder, train=train_finishes)
    run_main(tmp_path, merge(block, APPROVED), runner, repo)
    rep = report(tmp_path)
    assert stage(rep, "build")["status"] == fw.REFUSED
    assert "spot check has not passed" in stage(rep, "build")["reason"]
    assert "--spot-check-server" in stage(rep, "build")["reason"]
    train = stage(rep, "train")
    assert train["status"] == fw.REFUSED and "INCOMPLETE" in train["reason"]
    assert "train" not in runner.names()


# --------------------------------------------------------------------------- #
# Train gate: N new verified traces
# --------------------------------------------------------------------------- #


def test_train_is_skipped_below_min_new_traces_and_the_report_says_why(tmp_path, block, repo):
    golden(tmp_path, 3)
    runner = Runner(build=build_ok())
    run_main(tmp_path, merge(merge(block, APPROVED), {"train": {"min_new_traces": 5}}),
             runner, repo)

    assert "train" not in runner.names()
    rep = report(tmp_path)
    train = stage(rep, "train")
    assert train["status"] == fw.SKIPPED
    assert "3 new verified trace(s)" in train["reason"] and "needs 5" in train["reason"]
    assert stage(rep, "quantize")["status"] == fw.SKIPPED
    md = (tmp_path / "fw" / "reports" / "LATEST.md").read_text()
    assert "## Why stages did not run" in md and "needs 5" in md


def test_the_count_starts_again_after_a_finished_train(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED),
             Runner(build=build_ok(), train=train_finishes, score=score_ok), repo)
    # A week later, one new trace: below the bar of 2 since the last train's cutoff.
    write_jsonl(tmp_path / "golden" / "golden_traces_2.jsonl", [trace(50)])
    runner = Runner(build=build_ok(), train=train_finishes)
    run_main(tmp_path, merge(block, APPROVED), runner, repo, NEXT_SAT)
    train = stage(report(tmp_path), "train")
    assert train["status"] == fw.SKIPPED and "1 new verified" in train["reason"]
    assert "train" not in runner.names()


def test_a_train_needs_new_rows_not_only_new_verified_traces(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED),
             Runner(build=build_ok(tag="same"), train=train_finishes, score=score_ok), repo)
    # A week on, two new verified traces that the builder drops (another
    # --teacher-model, or duplicates): its sets are the rows already trained on.
    write_jsonl(tmp_path / "golden" / "golden_traces_2.jsonl", [trace(50), trace(51)])
    runner = Runner(build=build_ok(tag="same"), train=train_finishes)
    run_main(tmp_path, merge(block, APPROVED), runner, repo, NEXT_SAT)
    assert "build" in runner.names() and "train" not in runner.names()
    train = stage(report(tmp_path), "train")
    assert train["status"] == fw.SKIPPED
    assert "0 train row(s) that fw-2026-10-31-max-derived was not trained on" in train["reason"]
    assert train["details"]["new_verified_since_last_train"] == 2


# --------------------------------------------------------------------------- #
# The window: stop before it closes, resume across weekends
# --------------------------------------------------------------------------- #


def test_a_paused_train_resumes_next_window_on_the_same_frozen_data(tmp_path, block, repo):
    golden(tmp_path, 3)
    first = Runner(build=build_ok(), train=train_paused)
    run_main(tmp_path, merge(block, APPROVED), first, repo)

    spec = next(s for n, s in first.calls if n == "train")
    end = datetime(2026, 10, 31, 6, 0, tzinfo=ET)
    assert spec.stop_at == end - timedelta(minutes=15)
    assert spec.kill_at == end - timedelta(seconds=60)
    assert arg(spec.argv, "--stop-at") == spec.stop_at.astimezone(timezone.utc).isoformat()
    assert "--resume" not in spec.argv
    rep = report(tmp_path)
    assert stage(rep, "train")["status"] == fw.PAUSED
    assert "checkpoint-200" in stage(rep, "train")["reason"]
    frozen = arg(spec.argv, "--data")
    assert frozen.endswith("train/fw-2026-10-31-max-derived.data/train.jsonl")

    # Next Saturday: no new traces (below the bar), yet the train resumes.
    second = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    run_main(tmp_path, merge(block, APPROVED), second, repo, NEXT_SAT)
    resumed = second.argv("train")
    assert "--resume" in resumed and fw.MAX_DERIVED_APPROVAL_FLAG in resumed
    assert arg(resumed, "--data") == frozen
    assert "build" not in second.names()          # nothing new to build
    rep = report(tmp_path)
    assert stage(rep, "train")["status"] == fw.OK
    assert stage(rep, "quantize")["status"] == fw.OK
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert state["train"]["attempts"] == 2 and state["train"]["status"] == "complete"


def test_exit_0_with_a_stopped_manifest_is_a_pause_not_a_finish(tmp_path, block, repo):
    """train_qlora.py exits 0 when it stops cleanly at --stop-at."""
    golden(tmp_path, 3)

    def stops_itself(spec: fw.CommandSpec) -> int:
        run_dir = Path(arg(spec.argv, "--out"))
        (run_dir / "checkpoint-400").mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"status": "stopped"}))
        return 0

    run_main(tmp_path, merge(block, APPROVED), Runner(build=build_ok(), train=stops_itself), repo)
    assert stage(report(tmp_path), "train")["status"] == fw.PAUSED


def test_a_finished_exit_without_an_adapter_is_a_failure(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED), Runner(build=build_ok(), train=lambda s: 0), repo)
    train = stage(report(tmp_path), "train")
    assert train["status"] == fw.FAILED and "no adapter" in train["reason"]
    assert "adapter-max-derived" in train["reason"]          # the name it looked for


def test_an_untagged_adapter_from_max_derived_data_fails_the_train(tmp_path, block, repo):
    """A trainer that ignores the tag leaves an adapter whose name does not
    say what it is: never recorded, never quantized (Q6b)."""
    golden(tmp_path, 3)

    def untagged(spec: fw.CommandSpec) -> int:
        run_dir = Path(arg(spec.argv, "--out"))
        (run_dir / "adapter").mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"status": "done"}))
        return 0

    runner = Runner(build=build_ok(), train=untagged)
    run_main(tmp_path, merge(block, APPROVED), runner, repo)
    rep = report(tmp_path)
    train = stage(rep, "train")
    assert train["status"] == fw.FAILED
    assert "the data is Max-derived but the trainer left an untagged" in train["reason"]
    assert "quantize" not in runner.names()
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    assert "adapters" not in state


def test_find_adapter_reads_the_tag_and_the_trainers_manifest(tmp_path):
    pol = fw.load_policy()
    run_dir = tmp_path / "fw-x"
    (run_dir / "adapter-max-derived").mkdir(parents=True)
    # A tagged adapter is Max-derived even when the data check said it was not.
    assert fw.find_adapter(pol, run_dir, False) == (run_dir / "adapter-max-derived", True, "")
    clean = tmp_path / "fw-y"
    (clean / "adapter").mkdir(parents=True)
    assert fw.find_adapter(pol, clean, False) == (clean / "adapter", False, "")
    # The trainer's run manifest says Max-derived: the untagged name is refused.
    (clean / "manifest.json").write_text(json.dumps(
        {"schema": "distill-train-manifest/1", "max_derived": True, "tags": [TAG]}))
    found, md, problem = fw.find_adapter(pol, clean, False)
    assert found is None and md is True and "its name is not tagged" in problem


def test_no_stage_starts_too_close_to_the_window_end(tmp_path, block, repo):
    golden(tmp_path, 3)
    runner = Runner(build=build_ok(), train=train_finishes)
    late = datetime(2026, 10, 31, 5, 40, tzinfo=ET)       # 5 min before the 05:45 stop
    run_main(tmp_path, merge(block, APPROVED), runner, repo, late)
    rep = report(tmp_path)
    assert stage(rep, "harvest")["status"] == fw.OK
    for name in ("build", "train", "quantize", "score"):
        assert stage(rep, name)["status"] == fw.SKIPPED
        assert "min left before the stop time (05:45)" in stage(rep, name)["reason"]
    assert runner.calls == []


def test_harvest_checks_the_clock_between_lines_and_records_nothing_partial(
        tmp_path, block, repo, monkeypatch):
    monkeypatch.setattr(fw, "CLOCK_EVERY_LINES", 1)
    golden(tmp_path, 20)
    ticks = iter(range(10_000))

    def ticking() -> datetime:                # a minute per look at the clock, from 05:40
        return datetime(2026, 10, 31, 5, 40, tzinfo=ET) + timedelta(minutes=next(ticks))

    run = make_run(block, repo, clock=ticking)
    result = fw.stage_harvest(run)
    assert result.status == fw.PAUSED
    assert not (run.run_dir / "harvest.jsonl").exists()
    # The next look, with time to spare, finds all 20 still new.
    again = fw.stage_harvest(make_run(block, repo))
    assert again.status == fw.OK and again.details["new"] == 20


def test_a_rerun_in_the_same_window_does_not_repeat_finished_stages(tmp_path, block, repo):
    golden(tmp_path, 3)
    first = Runner(build=build_ok())
    run_main(tmp_path, block, first, repo)                 # train refused: no approval
    second = Runner(build=build_ok())
    run_main(tmp_path, block, second, repo, SAT + timedelta(minutes=20))
    assert first.names().count("build") == 1 and "build" not in second.names()
    rep = report(tmp_path)
    assert stage(rep, "harvest")["reason"] == "done earlier in this window"
    assert stage(rep, "build")["status"] == fw.OK
    assert stage(rep, "train")["status"] == fw.REFUSED


# --------------------------------------------------------------------------- #
# Tags and the offspring guard
# --------------------------------------------------------------------------- #


def test_every_new_trace_is_tagged_and_unverified_offspring_never_reaches_the_pool(
        tmp_path, block, repo):
    golden(tmp_path, 1)
    write_jsonl(tmp_path / "golden" / "golden_traces_2.jsonl", [trace(5, model="qwen3.8-flash")])
    write_jsonl(tmp_path / "extra" / "offspring.jsonl", [
        trace(10, model="student-9b", adapter_version="fw-2026-10-24"),           # no verdict
        trace(11, model="student-9b", adapter_version="fw-x", verdict="pass",
              verified_by="student-9b"),                                           # self-judged
        trace(12, model="student-9b", adapter_version="fw-x", verdict="pass",
              verified_by="judge-27b"),                                            # judged
        trace(13),                                                                 # extra, no verdict
        trace(14, verdict="pass", verified_by="judge-27b"),                        # extra, judged
    ])
    run = make_run(block, repo)
    assert fw.stage_harvest(run).status == fw.OK
    assert fw.stage_verify(run).status == fw.OK

    rows = [json.loads(x) for x in (run.run_dir / "harvest.jsonl").read_text().splitlines()]
    assert len(rows) == 7
    assert all(r["source_model"] and r["adapter_version"] for r in rows)
    by_model = {(r["source_model"], r["adapter_version"], r["origin"]) for r in rows}
    assert ("qwen3.8-max", "none", "teacher") in by_model
    assert ("qwen3.8-flash", "unknown", "unknown") in by_model
    assert ("student-9b", "fw-2026-10-24", "offspring") in by_model

    verify = json.loads((run.run_dir / "verify.json").read_text())
    assert verify["verified"] == 3          # the golden teacher trace + two judged extras
    assert verify["dropped"] == {"unverified_offspring": 1, "self_verified_offspring": 1,
                                 "unverified_extra": 1, "unknown_origin": 1}
    pool = [json.loads(x) for x in (tmp_path / "fw" / "pool" / "2026-10-31.jsonl")
            .read_text().splitlines()]
    assert sorted(p["messages"][0]["content"] for p in pool) == ["task 12", "task 14"]
    off = next(p["_meta"] for p in pool if p["messages"][0]["content"] == "task 12")
    assert off["source"] == "flywheel/offspring/student-9b@fw-x/max-derived"
    assert off["source_model"] == "student-9b" and off["adapter_version"] == "fw-x"
    # The job's decision has its own key; a generator's verification record is never touched.
    assert off["flywheel"]["verification"] == {"basis": "judge-pass", "by": "judge-27b"}
    assert "verification" not in off


def test_a_teacher_gen_trace_keeps_its_source_tag_and_verification_record(tmp_path, block, repo):
    """build.py's data policy keys training on the trace's own source tag and
    reads its verification record strictly; the pool must hand both over."""
    record = {"verified": True, "tool_success": True, "sandboxed": True, "side_effects": False,
              "judge": {"verdict": "pass", "model": "qwen3.8-27b", "pass": True}}
    write_jsonl(tmp_path / "extra" / "gym.jsonl", [
        trace(20, model="qwen3.8-27b", source="qwen38-27b-gym", verdict="pass",
              verification=record, max_derived=False),
        # a failed record vetoes a pass verdict beside it
        trace(21, model="qwen3.8-27b", source="qwen38-27b-gym", verdict="pass",
              verification={**record, "judge": {"verdict": "fail", "model": "qwen3.8-27b"}}),
        # judged offspring named by its record's judge, not by a verified_by key
        trace(22, model="student-9b", adapter_version="fw-x",
              verification={**record, "judge": {"verdict": "pass", "model": "judge-27b"}}),
    ])
    cfg_block = merge(block, {"harvest": {"teacher_models": ["qwen3.8-max", "qwen3.8-27b*"]}})
    run = make_run(cfg_block, repo)
    assert fw.stage_harvest(run).status == fw.OK and fw.stage_verify(run).status == fw.OK
    verify = json.loads((run.run_dir / "verify.json").read_text())
    assert verify["dropped"] == {"verdict_fail": 1}
    pool = {p["messages"][0]["content"]: p["_meta"] for p in
            (json.loads(x) for x in (tmp_path / "fw" / "pool" / "2026-10-31.jsonl")
             .read_text().splitlines())}
    gym = pool["task 20"]
    assert gym["source"] == "qwen38-27b-gym" and gym["verification"] == record
    assert gym["flywheel"]["tag"] == "flywheel/teacher/qwen3.8-27b@none"
    off = pool["task 22"]
    assert off["source"] == "flywheel/offspring/student-9b@fw-x/max-derived"
    assert off["flywheel"]["verification"] == {"basis": "judge-pass", "by": "judge-27b"}


PAYG = {"verdict": "pass", "model": "qwen3.8-max", "pass": True, "provider": "max-payg",
        "judge_key": "payg"}
RECORD = {"verified": True, "tool_success": True, "sandboxed": True, "side_effects": False}


def test_verify_accepts_the_payg_max_judge_as_build_py_does_and_no_other_max_judge(
        tmp_path, block, repo):
    """The operator's PAYG ruling: the pay-per-token Max judge verifies a
    Max-derived teacher trace (build.py applies it to a Max row). The Token
    Plan key (any other Max judge) never does, nor does the PAYG judge verify
    a local 27B row (build.py wants a --judge-models judge there) or a student."""
    def judged(i: int, judge: dict[str, Any], **meta: Any) -> dict[str, Any]:
        return trace(i, verification={**RECORD, "judge": judge}, **meta)

    write_jsonl(tmp_path / "extra" / "teacher-runs.jsonl", [
        judged(60, PAYG, source="max-payg-gym"),                                # verified
        judged(61, {**PAYG, "judge_key": "payg"}, model="qwen3.8-max"),         # verified
        judged(62, {k: v for k, v in PAYG.items() if k != "provider"}),        # Token Plan?
        judged(63, {**PAYG, "judge_key": "token-plan"}),                        # not PAYG
        judged(64, {**PAYG, "provider": "max-payg", "model": "judge-27b"}),     # not Max: ok
        judged(65, PAYG, model="qwen3.8-27b", source="qwen38-27b-gym", max_derived=False),
        judged(66, PAYG, model="student-9b", adapter_version="fw-x"),           # a student
    ])
    run = make_run(block, repo)
    assert fw.stage_harvest(run).status == fw.OK
    result = fw.stage_verify(run)
    verify = json.loads((run.run_dir / "verify.json").read_text())
    assert verify["dropped"] == {"max_judged": 3, "max_judged_offspring": 1}
    assert verify["verified"] == 3 and verify["payg_max_judge"] == 2
    assert result.details["payg_max_judge"] == 2
    pool = {p["messages"][0]["content"] for p in
            (json.loads(x) for x in (tmp_path / "fw" / "pool" / "2026-10-31.jsonl")
             .read_text().splitlines())}
    assert pool == {"task 60", "task 61", "task 64"}
    # The rule on its own: a PAYG pass on a row that is not Max-derived is no pass.
    row = {"origin": "teacher", "kind": "extra", "verdict": "pass",
           "verified_by": "qwen3.8-max", "payg_judge": True, "max_derived": False}
    assert fw.verification(row) == ("", "max_judged")
    assert fw.verification({**row, "max_derived": True}) == ("judge-pass", "")
    assert fw.verification({**row, "max_derived": True, "payg_judge": False}) == (
        "", "max_judged")
    assert fw.is_payg_judge(PAYG, "qwen3.8-max")
    assert not fw.is_payg_judge({**PAYG, "judge_key": "plan"}, "qwen3.8-max")
    assert not fw.is_payg_judge(PAYG, "qwen3.8-27b")


def test_the_flywheels_payg_rule_is_build_pys(tmp_path):
    """One rule: build.py's is_payg_judge and the flywheel's agree."""
    spec = importlib.util.spec_from_file_location(
        "distill_build_for_fw", REPO / "scripts" / "distill" / "build.py")
    assert spec and spec.loader
    build = importlib.util.module_from_spec(spec)
    sys.modules["distill_build_for_fw"] = build
    try:
        spec.loader.exec_module(build)
    finally:
        sys.modules.pop("distill_build_for_fw", None)
    assert (fw.PAYG_PROVIDER, fw.PAYG_JUDGE_KEY) == (build.PAYG_PROVIDER, build.PAYG_JUDGE_KEY)
    assert fw.GYM_SOURCE == build.GYM_SOURCE and fw.SHARE_CLASSES == build.SHARE_CLASSES
    for judge, model in ((PAYG, "qwen3.8-max"), ({**PAYG, "judge_key": "x"}, "qwen3.8-max"),
                         ({"model": "qwen3.8-max"}, "qwen3.8-max"), (PAYG, "judge-27b"),
                         ({k: v for k, v in PAYG.items() if k != "judge_key"}, "qwen3.8-max")):
        assert fw.is_payg_judge(judge, model) == build.is_payg_judge(judge, model)
    for flag in ("BUILD_ALLOW_FLAG", "BUILD_APPROVAL_FLAG"):
        assert getattr(fw, flag) == getattr(fw.load_policy(), flag)
    qspec = importlib.util.spec_from_file_location(
        "distill_quantize_for_fw", REPO / "scripts" / "distill" / "quantize.py")
    assert qspec and qspec.loader
    q = importlib.util.module_from_spec(qspec)
    sys.modules["distill_quantize_for_fw"] = q
    try:
        qspec.loader.exec_module(q)
    finally:
        sys.modules.pop("distill_quantize_for_fw", None)
    # the trainer's flag too (quantize.py's test keeps the two equal)
    assert fw.MAX_DERIVED_APPROVAL_FLAG == q.APPROVAL_FLAG


# --------------------------------------------------------------------------- #
# Harvest: watermarks and counts per source
# --------------------------------------------------------------------------- #


def gym_trace(i: int, model: str = "qwen3.8-27b", source: str = "qwen38-27b-gym",
              judge: dict[str, Any] | None = None, **meta: Any) -> dict[str, Any]:
    """A teacher_gen.py row: the local 27B's, or a Max teacher run's."""
    rec = {**RECORD, "judge": judge or {"verdict": "pass", "model": "qwen3.8-27b",
                                        "pass": True}}
    return trace(i, model=model, source=source, verdict="pass", verification=rec,
                 max_derived=not model.endswith("27b"), **meta)


def test_harvest_reads_past_its_watermark_and_counts_each_source(tmp_path, block, repo):
    exports = tmp_path / "golden"
    write_jsonl(exports / "golden_traces_1790000000_100.jsonl", [trace(0), trace(1)])
    local = write_jsonl(tmp_path / "extra" / "qwen38-27b-gym.traces.jsonl",
                        [gym_trace(10), gym_trace(11)])
    maxrun = write_jsonl(tmp_path / "extra" / "max-gym.traces.jsonl",
                         [gym_trace(20, model="qwen3.8-max", source="max-payg-gym", judge=PAYG)])
    state: dict[str, Any] = {"schema": fw.STATE_SCHEMA}
    first = fw.stage_harvest(make_run(block, repo, state=state))
    assert first.status == fw.OK and first.details["new"] == 5
    d = first.details
    assert d["by_class"] == {"max": 3, "qwen38-27b-gym": 2, "other": 0}
    assert {(r["kind"], r["source"], r["class"], r["new"]) for r in d["by_source_tag"]} == {
        ("golden", "golden-export", "max", 2), ("extra", "qwen38-27b-gym", "qwen38-27b-gym", 2),
        ("extra", "max-payg-gym", "max", 1)}
    assert d["watermarks"] == {"new": 3, "grown": 0, "unchanged": 0, "rewritten": 0}
    assert d["golden_export_rowid"] == 100 and d["golden_export_rowid_before"] is None
    marks = fw.harvest_marks(state)
    assert marks[os.path.realpath(local)]["offset"] == local.stat().st_size
    assert json.loads((tmp_path / "fw" / "state.json").read_text())["harvest"]["files"] == marks

    # A week on: a new export (a new interactive /qwen turn), more rows from the
    # local teacher, a line the Max run is still writing; nothing else changed.
    write_jsonl(exports / "golden_traces_1790600000_140.jsonl", [trace(2)])
    with open(local, "a") as fh:
        fh.write(json.dumps(gym_trace(12)) + "\n")
    with open(maxrun, "a") as fh:
        fh.write(json.dumps(gym_trace(21, model="qwen3.8-max", source="max-payg-gym"))[:40])
    second = fw.stage_harvest(make_run(block, repo, now=NEXT_SAT, state=state))
    d = second.details
    assert second.status == fw.OK and d["new"] == 2 and d["already_harvested"] == 0
    assert d["unreadable"] == 0 and d["partial_last_lines"] == 1
    assert d["watermarks"] == {"new": 1, "grown": 2, "unchanged": 1, "rewritten": 0}
    assert d["golden_export_rowid"] == 140 and d["golden_export_rowid_before"] == 100
    assert d["by_class"] == {"max": 1, "qwen38-27b-gym": 1, "other": 0}

    # The Max run finishes its line; teacher_gen.py's resume rewrites the local
    # file (drops a row): read whole again, nothing harvested twice.
    with open(maxrun, "a") as fh:
        fh.write(json.dumps(gym_trace(21, model="qwen3.8-max",
                                      source="max-payg-gym"))[40:] + "\n")
    write_jsonl(local, [gym_trace(11), gym_trace(12), gym_trace(13)])
    third = fw.stage_harvest(make_run(block, repo, now=NEXT_SAT + timedelta(days=7),
                                      state=state))
    d = third.details
    assert d["new"] == 2 and d["already_harvested"] == 2 and d["unreadable"] == 0
    assert d["watermarks"] == {"new": 0, "grown": 1, "unchanged": 2, "rewritten": 1}
    rows = [json.loads(x) for x in (tmp_path / "fw" / "runs" / "2026-11-14" / "harvest.jsonl")
            .read_text().splitlines()]
    assert sorted((r["file"], r["line"]) for r in rows) == [
        ("max-gym.traces.jsonl", 2), ("qwen38-27b-gym.traces.jsonl", 3)]


def test_a_harvest_the_window_cuts_short_moves_no_watermark(tmp_path, block, repo, monkeypatch):
    monkeypatch.setattr(fw, "CLOCK_EVERY_LINES", 1)
    golden(tmp_path, 5)
    ticks = iter(range(10_000))

    def ticking() -> datetime:
        return datetime(2026, 10, 31, 5, 43, tzinfo=ET) + timedelta(minutes=next(ticks))

    state: dict[str, Any] = {"schema": fw.STATE_SCHEMA}
    assert fw.stage_harvest(make_run(block, repo, clock=ticking, state=state)).status == fw.PAUSED
    assert "harvest" not in state


def test_the_report_shows_the_share_and_that_max_derived_artifacts_are_private(
        tmp_path, block, repo):
    golden(tmp_path, 3)

    def with_share(spec: fw.CommandSpec) -> fw.CommandResult:
        res = build_ok()(spec)
        out = Path(arg(spec.argv, "--out"))
        report_ = json.loads((out / "report.json").read_text())
        report_["data_policy"]["share"] = {
            "classes": list(fw.SHARE_CLASSES), "approval": REF, "max_derived_allowed": True,
            "by_split": {"train": {"rows": 5, "max": {"rows": 2, "pct": 40.0},
                                   "qwen38-27b-gym": {"rows": 3, "pct": 60.0},
                                   "other": {"rows": 0, "pct": 0.0}},
                         "val": {"rows": 1, "max": {"rows": 0, "pct": 0.0},
                                 "qwen38-27b-gym": {"rows": 1, "pct": 100.0},
                                 "other": {"rows": 0, "pct": 0.0}}}}
        (out / "report.json").write_text(json.dumps(report_))
        return res

    run_main(tmp_path, merge(block, APPROVED),
             Runner(build=with_share, train=train_finishes, score=score_ok), repo)
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    sets = rep["traces"]["dataset"]["sets"]
    assert sets["approval"] == REF and sets["share"]["by_split"]["train"]["rows"] == 5
    assert rep["traces"]["dataset"]["train"]["share"] == {"max": 4, "qwen38-27b-gym": 0,
                                                          "other": 0}
    assert ("- Max vs local 27B: train split of the 2026-10-31 sets: Max 2 (40.0%), local 27B "
            "3 (60.0%), other 0 (0.0%) of 5") in md
    assert f"- Max vs local 27B in the 2026-10-31 sets (Q6b approval {REF!r}):" in md
    assert "  - val: Max 0 (0.0%), local 27B 1 (100.0%), other 0 (0.0%) of 1" in md
    assert "- New by class: Max 3, local 27B 0, other 0" in md
    g = rep["guards"]
    assert g["max_derived"] == {"approved": True, "approval_ref": REF,
                                "exclude_sessions_file": str(tmp_path / "exclude-sessions.txt"),
                                "sessions_listed": 1, "complete": True}
    kinds = {i["kind"]: i for i in g["private_artifacts"]["items"]}
    assert set(kinds) == {"dataset", "train data", "train run", "adapter", "model"}
    assert kinds["adapter"]["name"] == "fw-2026-10-31-max-derived"
    assert "- Max-derived artifacts: 5, all private (never published)" in md
    assert "training ALLOWED under the Q6b allow (approval 'operator Q6b test'; 1 session(s) " \
           "excluded via" in md
    assert "- Private: Max-derived artifacts are private: never pushed to Hugging Face or " \
           "anywhere public." in md
    assert "  - adapter fw-2026-10-31-max-derived: " in md and "NOT carry" not in md
    assert "sess-incident" not in md and "sess-incident" not in json.dumps(rep)


def test_offspring_judged_by_the_student_under_any_of_its_names_or_by_max_is_dropped(
        tmp_path, block, repo):
    """The student judging itself is no verification, whichever name the judge
    is given; nor is a Max judge (never used to judge)."""
    served = "distill-9b-fw-2026-10-31-Q4_K_M"
    student = {"model": "student-9b", "served_model": served, "adapter_version": "fw-2026-10-31"}
    judged = {"verified": True, "tool_success": True}

    def by(name: str) -> dict[str, Any]:
        return {"verdict": "pass", "verified_by": name}

    write_jsonl(tmp_path / "extra" / "shadow.jsonl", [
        trace(30, **student, **by("student-9b")),               # its _meta.model
        trace(31, **student, verification={**judged, "judge": {"verdict": "pass",
                                                                "model": "student-9b"}}),
        trace(32, **student, **by(f"{served}.gguf")),           # its served id as a file
        trace(33, **student, **by(f"models/{served}")),
        trace(34, **student, **by("qwen3.5-9b")),               # the base student
        trace(35, **student, **by("distill-fw-2026-10-24-Q5_K_M")),   # an earlier adapter
        trace(36, **student, **by("qwen3.8-max")),              # Max never judges
        trace(37, **student, **by("unknown")),                  # no name at all
        trace(38, **student, **by("judge-27b")),                # a real, other judge
        # a teacher's own pass from a Max judge is no verdict either
        trace(39, model="qwen3.8-27b", **by("qwen3.8-max")),
    ])
    cfg_block = merge(block, {"harvest": {"student_models": ["student-*", "qwen3.5-9b*"],
                                          "teacher_models": ["qwen3.8-27b*"]}})
    state = {"schema": fw.STATE_SCHEMA, "adapters": {"fw-2026-10-24": {"max_derived": False}}}
    run = make_run(cfg_block, repo, state=state)
    assert fw.stage_harvest(run).status == fw.OK
    result = fw.stage_verify(run)
    assert result.status == fw.OK
    verify = json.loads((run.run_dir / "verify.json").read_text())
    assert verify["verified"] == 1
    assert verify["dropped"] == {"self_verified_offspring": 6, "max_judged_offspring": 1,
                                 "unverified_offspring": 1, "max_judged": 1}
    assert result.details["offspring_dropped"] == 8
    pool = [json.loads(x) for x in (tmp_path / "fw" / "pool" / "2026-10-31.jsonl")
            .read_text().splitlines()]
    assert [p["messages"][0]["content"] for p in pool] == ["task 38"]
    # Each name check on its own, without the student patterns.
    row = fw.tag_trace({**student, "verified_by": "x", "verdict": "pass"},
                       fw.parse_config(block), {})
    assert fw.self_verifier(row, "Student-9B")
    assert fw.self_verifier(row, "fw-2026-10-31")
    assert not fw.self_verifier(row, "judge-27b")
    assert not fw.self_verifier(row, "student")            # a looser name is not the id


def test_the_max_guard_reads_a_rows_own_model_and_max_derived(tmp_path, block):
    rows = write_jsonl(tmp_path / "rows.jsonl", [
        {"source": "qwen38-27b-gym", "model": "qwen3.8-27b", "max_derived": False},
        {"source": "gym", "model": "qwen3.8-max", "max_derived": False},
        {"source": "gym", "model": "qwen3.8-27b", "max_derived": True},
    ])
    g = fw.check_training_data([rows], fw.parse_config(block), "qwen3.8-27b")
    assert g.rows == 3 and g.max_rows == 2 and any("Max-derived" in r for r in g.refusals)


@pytest.mark.parametrize("meta, expect", [
    ({"model": "qwen3.8-max"}, ("teacher", "none", True)),
    ({"model": "teacher-x"}, ("unknown", "unknown", False)),
    ({"model": "student-9b"}, ("offspring", "unknown", True)),
    # an adapter version marks offspring even under a teacher's model id
    ({"model": "qwen3.8-max", "adapter_version": "fw-1"}, ("offspring", "fw-1", True)),
    # a served id naming an adapter this job trained, recorded as not Max-derived
    ({"served_model": "distill-fw-2026-10-31-Q4_K_M"}, ("offspring", "fw-2026-10-31", False)),
])
def test_tagging(block, meta, expect):
    cfg = fw.parse_config(block)
    tag = fw.tag_trace(meta, cfg, {"fw-2026-10-31": {"max_derived": False}})
    assert (tag["origin"], tag["adapter_version"], tag["max_derived"]) == expect


def test_tagging_reads_the_exact_adapter_version_a_served_id_names(block):
    cfg = fw.parse_config(block)
    adapters = {"fw-2026-10-31": {"max_derived": False}, "fw-2026-10-31-2": {"max_derived": True}}
    second = fw.tag_trace({"served_model": "distill-fw-2026-10-31-2-Q4_K_M"}, cfg, adapters)
    assert (second["adapter_version"], second["max_derived"]) == ("fw-2026-10-31-2", True)
    first = fw.tag_trace({"served_model": "distill-fw-2026-10-31-Q4_K_M"}, cfg, adapters)
    assert (first["adapter_version"], first["max_derived"]) == ("fw-2026-10-31", False)
    # A sibling this state has no record of is one of ours, but not which.
    other = fw.tag_trace({"served_model": "distill-fw-2026-10-31-3-Q4_K_M"}, cfg,
                         {"fw-2026-10-31": {"max_derived": False}})
    assert (other["origin"], other["adapter_version"], other["max_derived"]) == (
        "offspring", "unknown", True)


def _seed_sets(run: fw.Run, rows: list[dict[str, Any]], teacher: str,
               approval: str | None = REF) -> None:
    """Sets as build.py leaves them; with an *approval*, built under the Q6b allow."""
    sets = run.run_dir / "sets"
    write_jsonl(sets / "train.jsonl", rows)
    write_jsonl(sets / "val.jsonl", rows[-1:])
    report: dict[str, Any] = {"settings": {"teacher_model": teacher}}
    if approval is not None:
        report["data_policy"] = {"max_derived_allow": {"allowed": True, "approval": approval}}
    (sets / "report.json").write_text(json.dumps(report))
    (run.run_dir / "verify.json").write_text(json.dumps({"verified": 10}))
    run.state["sets"] = {"run": run.run_id, "dir": str(sets), "gates": "PASS",
                         "teacher_model": teacher,
                         "sessions_sha256": fw.session_list_digest(
                             run.cfg.max_derived.exclude_sessions_file) or ""}


def test_the_train_refuses_offspring_rows_without_a_judge_verdict(tmp_path, block, repo):
    rows = [{"source": "flywheel/offspring/student-9b@fw-x", "label": "tool-success"},
            {"source": "flywheel/teacher/teacher-x@none", "label": "tool-success"}]
    runner = Runner(train=train_finishes)
    run = make_run(block, repo, runner=runner)
    _seed_sets(run, rows, teacher="teacher-x")
    result = fw.stage_train(run)
    assert result.status == fw.REFUSED
    assert "1 offspring row(s) carry no judge verdict" in result.reason
    assert runner.calls == [] and "train" not in run.state

    rows[0]["label"] = "judge-pass"
    run = make_run(block, repo, runner=runner)
    _seed_sets(run, rows, teacher="teacher-x")
    assert fw.stage_train(run).status == fw.OK
    assert runner.names() == ["train"]


def test_the_train_refuses_rows_that_share_a_task_with_the_score_cases(tmp_path, block, repo):
    # The fixed score cases: an earlier build's eval-tools, re-split since.
    write_jsonl(tmp_path / "cases" / "eval-tools.jsonl",
                [{"id": "p3-old", "meta": {"task_id": "t-held", "group_id": "t-held"}}])
    rows = [{"id": "p3-b", "task_id": "t-joined", "group_id": "t-held",       # its group
             "source": "max-golden", "label": "tool-success"},
            {"id": "p3-a", "task_id": "t-new", "group_id": "t-new",
             "source": "max-golden", "label": "tool-success"}]
    runner = Runner(train=train_finishes)
    run = make_run(merge(block, APPROVED), repo, runner=runner)
    _seed_sets(run, rows, teacher="qwen3.8-max")
    result = fw.stage_train(run)
    assert result.status == fw.REFUSED
    assert "1 training row(s) share an id, task or group with the score cases" in result.reason
    assert runner.calls == [] and "train" not in run.state


# --------------------------------------------------------------------------- #
# The Max-data guard (DISTILL-OPEN-QUESTIONS #6)
# --------------------------------------------------------------------------- #


def test_the_train_refuses_max_derived_data_without_recorded_approval(tmp_path, block, repo):
    rows = [{"source": "max-golden", "label": "tool-success"}] * 3
    runner = Runner(train=train_finishes)
    run = make_run(block, repo, runner=runner)
    _seed_sets(run, rows, teacher="qwen3.8-max")
    result = fw.stage_train(run)
    assert result.status == fw.REFUSED
    assert "Max-derived" in result.reason and "OPEN-QUESTIONS #6" in result.reason
    assert runner.calls == []

    run = make_run(merge(block, APPROVED), repo, runner=runner)
    _seed_sets(run, rows, teacher="qwen3.8-max")
    assert fw.stage_train(run).status == fw.OK
    argv = runner.argv("train")
    assert Path(arg(argv, "--out")).name == "fw-2026-10-31-max-derived"
    assert fw.MAX_DERIVED_APPROVAL_FLAG in argv
    t = run.state["train"]
    assert t["max_derived"] is True and t["approval"] == REF
    assert t["share"] == {"max": 4, "qwen38-27b-gym": 0, "other": 0}     # train + val rows
    assert Path(t["data_dir"]).name == "fw-2026-10-31-max-derived.data"
    assert run.state["adapters"]["fw-2026-10-31-max-derived"]["adapter_dir"].endswith(
        "/adapter-max-derived")


@pytest.mark.parametrize("approval, over, why", [
    # sets built without the allow (or under another approval) record none of it
    (None, {}, "the dataset records no Max-derived approval"),
    ("another ref", {}, "the dataset records the approval 'another ref'"),
    # the allow needs a usable exclusion list, at train time too
    (REF, {"max_derived": {"exclude_sessions_file": "/nonexistent/sessions.txt"}},
     "does not exist"),
])
def test_a_max_derived_train_needs_the_datasets_own_approval_and_the_session_list(
        tmp_path, block, repo, approval, over, why):
    rows = [{"source": "max-golden", "label": "tool-success"}] * 3
    runner = Runner(train=train_finishes)
    run = make_run(merge(merge(block, APPROVED), over), repo, runner=runner)
    _seed_sets(run, rows, teacher="qwen3.8-max", approval=approval)
    result = fw.stage_train(run)
    assert result.status == fw.REFUSED and why in result.reason, result.reason
    assert runner.calls == [] and "train" not in run.state


def test_an_empty_session_list_refuses_the_build(tmp_path, block, repo):
    golden(tmp_path, 3)
    (tmp_path / "exclude-sessions.txt").write_text("# nothing listed\n\n")
    runner = Runner(build=build_ok())
    run_main(tmp_path, merge(block, APPROVED), runner, repo)
    build = stage(report(tmp_path), "build")
    assert build["status"] == fw.REFUSED and "lists no session id" in build["reason"]
    assert "build" not in runner.names()


def test_without_the_allow_the_build_gets_only_the_session_list(tmp_path, block, repo):
    golden(tmp_path, 1)
    runner = Runner(build=build_ok())
    run_main(tmp_path, block, runner, repo)
    argv = runner.argv("build")
    assert Path(arg(argv, "--out")).name == "sets"
    assert arg(argv, fw.BUILD_EXCLUDE_FLAG) == str(tmp_path / "exclude-sessions.txt")
    assert fw.BUILD_ALLOW_FLAG not in argv and fw.BUILD_APPROVAL_FLAG not in argv


def test_a_changed_allow_rebuilds_the_sets(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, block, Runner(build=build_ok()), repo)
    # A week on, no new traces, but the operator recorded the Q6b allow.
    runner = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    run_main(tmp_path, merge(block, APPROVED), runner, repo, NEXT_SAT)
    assert runner.names()[:2] == ["build", "train"]
    assert Path(arg(runner.argv("build"), "--out")).name == "sets-max-derived"


def test_an_edit_to_the_session_list_rebuilds_and_older_sets_never_train(tmp_path, block, repo):
    """The build's inputs hold the ids the list names, not only its path; and
    sets built under another list never start a train, even when the rebuild
    fails."""
    golden(tmp_path, 3)
    held = merge(merge(block, APPROVED), {"train": {"min_new_traces": 100}})   # no train yet
    first = Runner(build=build_ok())
    run_main(tmp_path, held, first, repo)
    assert first.names() == ["build"]
    state_path = tmp_path / "fw" / "state.json"
    before = json.loads(state_path.read_text())["sets"]
    assert before["sessions_sha256"] == fw.session_list_digest(
        tmp_path / "exclude-sessions.txt") != ""
    # A comment or reordering is no edit to the list: its sets stand.
    sessions = tmp_path / "exclude-sessions.txt"
    sessions.write_text("sess-incident   # 2026-10-13\n\n# the incident session\n")
    second = Runner(build=build_ok())
    run_main(tmp_path, held, second, repo, NEXT_SAT)
    assert "build" not in second.names()
    assert stage(report(tmp_path), "build")["status"] == fw.SKIPPED
    # A second incident session listed, and no new traces: the build runs again.
    # It fails here, and the sets of 2026-10-31 (built without that session
    # excluded) do not train.
    sessions.write_text("sess-incident\nsess-second\n")
    third = Runner(build=lambda spec: fw.CommandResult(2, tail="boom"), train=train_finishes)
    run_main(tmp_path, merge(held, {"train": {"min_new_traces": 2}}), third, repo,
             NEXT_SAT + timedelta(days=7))
    assert third.names() == ["build"]
    rep = report(tmp_path)
    assert stage(rep, "build")["status"] == fw.FAILED
    train = stage(rep, "train")
    assert train["status"] == fw.REFUSED
    assert "were built under another session exclusion list" in train["reason"]
    assert "train" not in json.loads(state_path.read_text())
    assert "sess-second" not in json.dumps(rep)


def test_a_paused_train_is_superseded_when_the_session_list_changes(tmp_path, block, repo):
    """Its frozen data may hold a session listed since: it is not resumed, and
    a new train starts on sets rebuilt under the current list."""
    golden(tmp_path, 3)
    first = Runner(build=build_ok(), train=train_paused)
    run_main(tmp_path, merge(block, APPROVED), first, repo)
    frozen = arg(first.argv("train"), "--data")
    assert frozen.endswith("train/fw-2026-10-31-max-derived.data/train.jsonl")
    (tmp_path / "exclude-sessions.txt").write_text("sess-incident\nsess-second\n")

    second = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    run_main(tmp_path, merge(block, APPROVED), second, repo, NEXT_SAT)
    assert second.names()[:2] == ["build", "train"]
    argv = second.argv("train")
    assert "--resume" not in argv and arg(argv, "--data") != frozen
    assert arg(argv, "--data").endswith("train/fw-2026-11-07-max-derived.data/train.jsonl")
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    train = stage(rep, "train")
    assert train["status"] == fw.OK
    assert train["details"]["superseded"]["adapter_version"] == "fw-2026-10-31-max-derived"
    assert any(i.startswith("train fw-2026-10-31-max-derived is not resumed: the session "
                            "exclusion list") for i in rep["issues"])
    state = json.loads((tmp_path / "fw" / "state.json").read_text())
    old = state["superseded_trains"]
    assert [o["adapter_version"] for o in old] == ["fw-2026-10-31-max-derived"]
    assert old[0]["superseded_in"] == "2026-11-07" and old[0]["max_derived"] is True
    assert state["train"]["adapter_version"] == "fw-2026-11-07-max-derived"
    assert state["train"]["sessions_sha256"] == state["sets"]["sessions_sha256"] \
        == fw.session_list_digest(tmp_path / "exclude-sessions.txt")
    # its frozen data and checkpoints stay on disk, and private
    items = {(i["kind"], i["name"]) for i in rep["guards"]["private_artifacts"]["items"]}
    assert ("train data", "fw-2026-10-31-max-derived (superseded)") in items
    assert ("train run", "fw-2026-10-31-max-derived (superseded)") in items
    assert Path(frozen).is_file()
    assert "sess-second" not in md and "sess-second" not in json.dumps(rep)


def test_a_dry_run_says_a_train_would_be_superseded_and_changes_nothing(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED), Runner(build=build_ok(), train=train_paused), repo)
    state_path = tmp_path / "fw" / "state.json"
    (tmp_path / "exclude-sessions.txt").write_text("sess-incident\nsess-second\n")
    before = state_path.read_text()
    run = make_run(merge(block, APPROVED), repo, now=NEXT_SAT,
                   state=json.loads(before))
    run.dry_run = True
    result = fw.stage_train(run)
    assert result.details["superseded"]["adapter_version"] == "fw-2026-10-31-max-derived"
    assert "superseded_trains" not in run.state and run.state["train"]["status"] == "incomplete"
    assert state_path.read_text() == before


def test_a_reworded_approval_ref_does_not_strand_a_paused_train(tmp_path, block, repo):
    """The frozen data keeps the approval it was built under; the trainer reads
    it there. A new approval_ref applies from the next build on."""
    golden(tmp_path, 3)
    first = Runner(build=build_ok(), train=train_paused)
    run_main(tmp_path, merge(block, APPROVED), first, repo)
    frozen = arg(first.argv("train"), "--data")
    reworded = merge(block, {"max_derived": {"approved": True,
                                             "approval_ref": REF + " (reworded)"}})
    second = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    run_main(tmp_path, reworded, second, repo, NEXT_SAT)
    resumed = second.argv("train")
    assert "--resume" in resumed and arg(resumed, "--data") == frozen
    assert fw.MAX_DERIVED_APPROVAL_FLAG in resumed
    rep = report(tmp_path)
    train = stage(rep, "train")
    assert train["status"] == fw.OK and train["details"]["approval"] == REF
    assert any(f"resumes under the approval its frozen data records ({REF!r})" in i
               for i in rep["issues"])
    assert not any("A build under the allow comes first" in s for s in steps_by(rep, "operator"))
    # The allow itself still counts: without it the resume is refused, not superseded.
    third_state = json.loads((tmp_path / "fw" / "state.json").read_text())
    third_state["train"]["status"] = "incomplete"
    run = make_run(block, repo, now=NEXT_SAT + timedelta(days=7), state=third_state)
    result = fw.stage_train(run)
    assert result.status == fw.REFUSED and "Max-derived" in result.reason
    assert run.state["train"]["status"] == "incomplete"


def test_quantize_refuses_a_max_derived_adapter_without_the_allow(tmp_path, block, repo):
    state = {"schema": fw.STATE_SCHEMA, "adapters": {
        "fw-x-max-derived": {"adapter_dir": "a", "data_dir": "d", "max_derived": True}}}
    runner = Runner()
    result = fw.stage_quantize(make_run(block, repo, runner=runner, state=state))
    assert result.status == fw.REFUSED and "no complete Q6b allow" in result.reason
    assert runner.calls == []
    # A record that does not say counts as Max-derived; its models dir is tagged.
    state = {"schema": fw.STATE_SCHEMA, "adapters": {"fw-y": {"adapter_dir": "a",
                                                              "data_dir": "d"}}}
    runner = Runner()
    run = make_run(merge(block, APPROVED), repo, runner=runner, state=state)
    assert fw.stage_quantize(run).status == fw.OK
    argv = runner.argv("quantize")
    assert Path(arg(argv, "--out")).name == "fw-y-max-derived"
    assert fw.MAX_DERIVED_APPROVAL_FLAG in argv
    assert run.state["quantized"]["fw-y"]["max_derived"] is True


def test_the_train_share_counts_each_row_by_its_own_provenance(tmp_path, block):
    """build.py names the Max model as the teacher of any set that mixes in a
    Max row; the guard still counts every row as Max-derived, but the share
    reports the local 27B's rows as theirs."""
    rows = write_jsonl(tmp_path / "mixed.jsonl", [
        *[{"source": "max-golden", "model": "qwen3.8-max", "max_derived": True}] * 6,
        *[{"source": "qwen38-27b-gym", "model": "qwen3.8-27b", "max_derived": False}] * 4,
        {"source": "qwen38-27b-gym"},                    # says nothing: counted as Max
    ])
    g = fw.check_training_data([rows], fw.parse_config(merge(block, APPROVED)), "qwen3.8-max")
    assert g.rows == 11 and g.max_rows == 11 and g.refusals == []
    assert g.share == {"max": 7, "qwen38-27b-gym": 4, "other": 0}
    local = fw.check_training_data([rows], fw.parse_config(block), "qwen3.8-27b")
    assert local.share == {"max": 6, "qwen38-27b-gym": 5, "other": 0}


def test_an_unknown_builder_teacher_counts_as_max_derived(tmp_path, block):
    # A row whose own source names no Max model: only the unknown teacher makes it count.
    rows = write_jsonl(tmp_path / "rows.jsonl",
                       [{"source": "flywheel/teacher/teacher-x@none", "label": "tool-success"}])
    g = fw.check_training_data([rows], fw.parse_config(block), None)
    assert g.rows == 1 and g.max_rows == 1
    assert any("Max-derived" in r for r in g.refusals)
    named = fw.check_training_data([rows], fw.parse_config(block), "teacher-x")
    assert named.max_rows == 0 and named.refusals == []
    approved = fw.check_training_data([rows], fw.parse_config(merge(block, APPROVED)), None)
    assert approved.max_rows == 1 and approved.refusals == []
    assert fw.is_max_name("flywheel/teacher/qwen3.8-max@none")
    assert not fw.is_max_name("maxwell-7b")


def test_a_refused_week_says_so_in_the_report(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, block, Runner(build=build_ok()), repo)
    rep = report(tmp_path)
    assert rep["status"] == "WARN"
    assert stage(rep, "train")["status"] == fw.REFUSED
    md = (tmp_path / "fw" / "reports" / "LATEST.md").read_text()
    assert "Max-derived data: training refused" in md


# --------------------------------------------------------------------------- #
# The report is always written
# --------------------------------------------------------------------------- #


def test_a_disabled_flywheel_still_writes_the_report(tmp_path, block, repo):
    runner = Runner()
    code = run_main(tmp_path, merge(block, {"enabled": False}), runner, repo)
    assert code == fw.EXIT_OK and runner.calls == []
    rep = report(tmp_path)
    assert rep["status"] == "OFF"
    assert all("flywheel.enabled is false" in s["reason"] for s in rep["stages"])


def test_a_crashing_stage_fails_the_run_but_not_the_report(tmp_path, block, repo):
    golden(tmp_path, 3)

    def boom(spec: fw.CommandSpec) -> int:
        raise RuntimeError("builder exploded")

    code = run_main(tmp_path, block, Runner(build=boom), repo)
    assert code == fw.EXIT_FAILED
    rep = report(tmp_path)
    assert rep["status"] == "FAIL"
    assert stage(rep, "build")["status"] == fw.FAILED
    assert "builder exploded" in stage(rep, "build")["reason"]
    assert [s["name"] for s in rep["stages"]] == list(fw.STAGES)   # later stages still recorded


def test_an_invalid_config_still_writes_the_report(tmp_path, block, repo):
    code = run_main(tmp_path, merge(block, {"window": {"start": 120}}), Runner(), repo)
    assert code == fw.EXIT_CONFIG
    rep = report(tmp_path)
    assert rep["status"] == "FAIL"
    assert any("quoted" in i for i in rep["issues"])


def test_an_invalid_config_with_its_own_window_still_writes_the_report(tmp_path, block, repo):
    # The timer moved to Sunday with the window, and a typo elsewhere in the block.
    over = {"window": {"day": "sun", "start": "01:00", "end": "07:00"},
            "train": {"min_new_trace": 100}}
    sunday = datetime(2026, 10, 25, 1, 0, 30, tzinfo=ET)
    assert run_main(tmp_path, merge(block, over), Runner(), repo, sunday) == fw.EXIT_CONFIG
    rep = report(tmp_path)
    assert rep["status"] == "FAIL" and rep["run_id"] == "2026-10-25"
    assert rep["window"]["describe"] == "Sun 01:00-07:00 America/New_York"
    assert any("min_new_trace is not a flywheel setting" in i for i in rep["issues"])


def test_a_config_error_report_states_nothing_from_the_fallback_defaults(tmp_path, block, repo):
    """A fully set, enabled, approved block with one typo: the report may not
    describe the defaults the job fell back to as the operator's config."""
    over = {"harvest": {"teacher_models": ["qwen3.8-27b*"]}, "train": {"min_new_trace": 100}}
    golden(tmp_path, 3)
    assert run_main(tmp_path, merge(merge(block, APPROVED), over), Runner(), repo) == \
        fw.EXIT_CONFIG
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["next_steps"] == [
        {"by": "operator", "step": "Fix the flywheel: block in prometheus.yaml (every problem is "
                                   "listed under Issues). The job refuses to run until it parses, "
                                   "and changes nothing meanwhile."},
        {"by": "job", "step": "Next window: Sat 2026-11-07 02:00-06:00 EST (07:00-11:00 UTC); the "
                              "job refuses to run then too unless the block parses."}]
    assert rep["traces"]["min_new_traces"] is None
    assert rep["traces"]["new_verified_since_last_train"] is None
    assert rep["guards"]["max_derived"]["approved"] is None
    assert rep["guards"]["max_derived"]["complete"] is None
    assert rep["guards"]["private_artifacts"]["known"] is False
    assert rep["spend"]["ledger"] == block["spend_ledger"]       # the block's own ledger
    for wrong in ("needs 500", "500 more", "OPEN-QUESTIONS #6", "names only Max", "flywheel is off",
                  "training refused", "no train has started", "No model has been scored",
                  "flywheel.build.args"):
        assert wrong not in md, wrong
    assert "- Max-derived data: unknown (the config is invalid" in md
    assert "- Train: unknown (the config is invalid, so the state was not read)" in md
    assert "the train threshold is unknown: the config is invalid" in md


def test_a_config_error_in_the_window_block_does_not_state_the_default_window(
        tmp_path, block, repo):
    run_main(tmp_path, merge(block, {"window": {"start": 120}}), Runner(), repo)
    rep, _ = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["next_steps"][-1]["step"].startswith(
        "Next window: unknown, flywheel.window is invalid (the default window would be ")


def test_the_report_writes_the_home_directory_as_tilde(tmp_path, block, repo, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED),
             Runner(build=build_ok(), train=train_finishes, score=score_ok), repo)
    reports = tmp_path / "fw" / "reports"
    for name in ("LATEST.md", "LATEST.json", "2026-10-31-weekly.md", "2026-10-31-weekly.json"):
        assert str(tmp_path) not in (reports / name).read_text(), name
    rep = report(tmp_path)
    assert rep["spend"]["ledger"] == "~/ledger.csv"
    assert stage(rep, "score")["details"]["summary"] == (
        "~/fw/scores/fw-2026-10-31-max-derived.summary.json")


def test_outside_the_window_a_scheduled_run_does_nothing(tmp_path, block, repo):
    runner = Runner()
    assert run_main(tmp_path, block, runner, repo, FRIDAY) == fw.EXIT_CONFIG
    assert runner.calls == [] and not (tmp_path / "fw").exists()


def test_a_dry_run_changes_nothing_but_its_own_report(tmp_path, block, repo):
    golden(tmp_path, 3)
    runner = Runner()
    code = run_main(tmp_path, merge(block, APPROVED), runner, repo, FRIDAY, "--dry-run")
    assert code == fw.EXIT_OK and runner.calls == []
    data = tmp_path / "fw"
    assert sorted(p.name for p in data.iterdir()) == ["reports"]
    assert not (data / "reports" / "LATEST.md").exists()
    rep = report(tmp_path, "dry-run")
    assert rep["dry_run"] is True
    assert stage(rep, "harvest")["status"] == fw.DRY_RUN
    assert "would record 3 new trace(s)" in stage(rep, "harvest")["summary"]
    assert stage(rep, "build")["status"] == fw.DRY_RUN
    assert stage(rep, "train")["status"] == fw.DRY_RUN
    assert "Max-derived rows: approved" in stage(rep, "train")["summary"]


def test_sigterm_to_the_job_stops_it_between_stages_and_the_report_says_so(
        tmp_path, block, repo):
    golden(tmp_path, 3)

    def build_then_sigterm(spec: fw.CommandSpec) -> fw.CommandResult:
        os.kill(os.getpid(), signal.SIGTERM)       # what systemd sends at its outer wall
        assert spec.should_stop is not None and spec.should_stop()
        return fw.CommandResult(0, stopped=True)

    handler_before = signal.getsignal(signal.SIGTERM)
    code = run_main(tmp_path, merge(block, APPROVED), Runner(build=build_then_sigterm), repo)
    assert signal.getsignal(signal.SIGTERM) is handler_before    # restored
    assert code == fw.EXIT_OK
    rep = report(tmp_path)
    assert stage(rep, "build")["status"] == fw.PAUSED
    for name in ("train", "quantize", "score"):
        assert stage(rep, name)["status"] == fw.PAUSED
        assert "SIGTERM" in stage(rep, name)["reason"]
    assert rep["status"] == "WARN" and any("SIGTERM" in i for i in rep["issues"])


@contextlib.contextmanager
def lock_held(lock: Path) -> Iterator[None]:
    """flywheel.lock held by another live process (a plain sys.executable child)."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, sys, time\n"
         f"f = open({str(lock)!r}, 'a+')\n"
         "fcntl.lockf(f.fileno(), fcntl.LOCK_EX)\n"
         "print('held', flush=True)\n"
         "time.sleep(30)\n"],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        yield
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_a_second_run_while_one_holds_the_lock_does_nothing(tmp_path, block, repo):
    with lock_held(tmp_path / "fw" / "flywheel.lock"):
        runner = Runner()
        assert run_main(tmp_path, block, runner, repo) == fw.EXIT_BUSY
        assert runner.calls == [] and not (tmp_path / "fw" / "reports").exists()


def test_recovery_restores_the_gpus_and_finishes_a_dead_run(tmp_path, block, repo):
    cfg = fw.parse_config(merge(block, {"gpu_setup": "flywheel"}))
    run_dir = cfg.data_dir / "runs" / "2026-10-31"
    run_dir.mkdir(parents=True)
    (run_dir / "stages.json").write_text(json.dumps({
        "harvest": {"name": "harvest", "status": "OK", "summary": "3 new", "seconds": 1.0}}))
    fw.save_state(cfg.data_dir / "state.json", {
        "schema": fw.STATE_SCHEMA,
        "current_run": {"id": "2026-10-31", "status": "running"},
        "gpu": {"applied": True, "setup": "flywheel", "deadman_at": "2026-10-31T06:15-04:00"}})
    runner = Runner()
    code = fw.recover(cfg, clock=Clock(SAT + timedelta(hours=4)), runner=runner, repo=repo)

    assert code == fw.EXIT_OK
    assert runner.names() == ["gpu-restore"]
    assert runner.argv("gpu-restore")[-3:] == ["apply", "business", "--execute"]
    rep = report(tmp_path)
    assert rep["status"] == "FAIL" and rep["mode"] == "recovery"
    assert stage(rep, "harvest")["status"] == fw.OK
    assert stage(rep, "verify")["status"] == fw.FAILED
    state = json.loads((cfg.data_dir / "state.json").read_text())
    assert state["gpu"]["applied"] is False and state["current_run"]["status"] == "died"


def test_recovery_leaves_a_live_run_alone(tmp_path, block, repo):
    """A timer start that found the lock held exits 3, which starts recovery
    while the run holding the lock is mid-train."""
    cfg = fw.parse_config(merge(block, {"gpu_setup": "flywheel"}))
    live = {"schema": fw.STATE_SCHEMA, "current_run": {"id": "2026-10-31", "status": "running"},
            "gpu": {"applied": True, "setup": "flywheel", "deadman_at": "2026-10-31T06:15-04:00"}}
    fw.save_state(cfg.data_dir / "state.json", live)
    runner = Runner()
    with lock_held(cfg.data_dir / "flywheel.lock"):
        code = fw.recover(cfg, clock=Clock(SAT + timedelta(minutes=1)), runner=runner, repo=repo)
    assert code == fw.EXIT_OK and runner.calls == []
    assert json.loads((cfg.data_dir / "state.json").read_text()) == live
    assert not (cfg.data_dir / "reports").exists()


def test_a_gpu_swap_is_armed_first_and_restored_even_when_training_fails(tmp_path, block, repo):
    golden(tmp_path, 3)
    runner = Runner(build=build_ok(), train=lambda s: 1)
    code = run_main(tmp_path, merge(merge(block, APPROVED), {"gpu_setup": "flywheel"}),
                    runner, repo)
    assert code == fw.EXIT_FAILED
    assert runner.names() == ["build", "gpu-setup", "gpu-setup", "train", "gpu-restore"]
    setup_calls = [s.argv for n, s in runner.calls if n == "gpu-setup"]
    assert setup_calls[0][2:4] == ["deadman", "business"]      # the dead-man first
    assert setup_calls[1][2:] == ["apply", "flywheel", "--execute"]
    assert stage(report(tmp_path), "train")["status"] == fw.FAILED


def test_train_and_quantize_need_a_gpu_setup_or_the_no_swap_opt_in(tmp_path, block, repo):
    golden(tmp_path, 3)
    runner = Runner(build=build_ok(), train=train_finishes)
    run_main(tmp_path, merge(merge(block, APPROVED), {"gpu_setup": ""}), runner, repo)
    train = stage(report(tmp_path), "train")
    assert train["status"] == fw.SKIPPED and "flywheel.gpu_setup is empty" in train["reason"]
    assert runner.names() == ["build"]
    waiting = {"schema": fw.STATE_SCHEMA, "adapters": {"fw-x": {"adapter_dir": "a",
                                                                "data_dir": "d",
                                                                "max_derived": False}}}
    run = make_run(merge(block, {"gpu_setup": ""}), repo, runner=runner, state=waiting)
    quantize = fw.stage_quantize(run)
    assert quantize.status == fw.SKIPPED and "'none'" in quantize.reason
    assert runner.names() == ["build"]


def test_a_gpu_swap_starts_only_with_time_for_it_and_is_held_to_the_stop_time(
        tmp_path, block, repo):
    swapping = merge(block, {"gpu_setup": "flywheel"})

    def waiting() -> dict[str, Any]:
        return {"schema": fw.STATE_SCHEMA, "adapters": {"fw-x": {"adapter_dir": "a",
                                                                 "data_dir": "d",
                                                                 "max_derived": False}}}

    # 20 min before the 05:45 stop: enough for quantize alone, not for a swap too.
    runner = Runner()
    run = make_run(swapping, repo, now=datetime(2026, 10, 31, 5, 25, tzinfo=ET),
                   runner=runner, state=waiting())
    result = fw.stage_quantize(run)
    assert result.status == fw.SKIPPED and "too little to apply 'flywheel'" in result.reason
    assert runner.calls == [] and "gpu" not in run.state

    runner = Runner()
    run = make_run(swapping, repo, now=datetime(2026, 10, 31, 5, 5, tzinfo=ET),
                   runner=runner, state=waiting())
    assert fw.stage_quantize(run).status == fw.OK
    swaps = [spec for name, spec in runner.calls if name == "gpu-setup"]
    assert len(swaps) == 2
    for spec in swaps:                     # SIGTERM at the stop time, SIGKILL inside the window
        assert spec.stop_at == run.stop_at and spec.kill_at is not None
        assert spec.kill_at <= run.kill_at


def scored_state(tmp_path: Path, version: str = "fw-2026-10-31") -> dict[str, Any]:
    """A quantized candidate waiting to be scored."""
    out_dir = tmp_path / "fw" / "models" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"schema": fw.STATE_SCHEMA,
            "quantized": {version: {"status": "done", "out_dir": str(out_dir)}},
            "adapters": {version: {}, f"{version}-2": {}}}


@pytest.mark.parametrize("served_id", [
    "/models/base-9b-Q4_K_M.gguf",               # last week's model, or the base
    "/models/fw-2026-10-31-2/fw-2026-10-31-2-Q4_K_M.gguf",   # a sibling version
    "",                                          # unreachable, or no id
])
def test_score_refuses_a_server_that_does_not_serve_the_candidate(tmp_path, block, repo,
                                                                   served, served_id):
    served["id"] = served_id
    runner = Runner(score=score_ok)
    run = make_run(block, repo, runner=runner, state=scored_state(tmp_path))
    result = fw.stage_score(run)
    assert result.status == fw.REFUSED
    assert "not fw-2026-10-31; nothing was scored" in result.reason
    assert runner.calls == [] and "models" not in run.state


def test_scores_answered_by_another_model_are_set_aside_not_recorded(tmp_path, block, repo):
    def swapped_mid_run(spec: fw.CommandSpec) -> int:
        write_jsonl(Path(arg(spec.argv, "--out")), [{"id": "c1", "status": "ok"}])
        out = Path(arg(spec.argv, "--summary-out"))
        out.write_text(json.dumps({**summary("x", 0.9, 0.9), "served_model": "base-9b.gguf"}))
        return 0

    run = make_run(block, repo, runner=Runner(score=swapped_mid_run),
                   state=scored_state(tmp_path))
    result = fw.stage_score(run)
    assert result.status == fw.REFUSED and "base-9b.gguf" in result.reason
    assert "models" not in run.state
    scores = tmp_path / "fw" / "scores"
    assert not (scores / "fw-2026-10-31.jsonl").exists()      # score.py starts clean next time
    assert len(list(scores.glob("fw-2026-10-31.jsonl.rejected-*"))) == 1


def score_exits_3(scored: int) -> Callable[[fw.CommandSpec], int]:
    """score.py run with errored cases: it writes the summary, then exits 3."""
    def handler(spec: fw.CommandSpec) -> int:
        out = summary(arg(spec.argv, "--label"), 0.5, 0.5)
        out["sets"]["eval-tools"].update(cases=2, scored=scored, errors=2 - scored)
        Path(arg(spec.argv, "--summary-out")).write_text(json.dumps(out))
        return 3
    return handler


def test_a_score_run_that_scored_nothing_fails_and_a_partial_one_warns(tmp_path, block, repo):
    run = make_run(block, repo, runner=Runner(score=score_exits_3(0)),
                   state=scored_state(tmp_path))
    result = fw.stage_score(run)
    assert result.status == fw.FAILED and "scored none of the cases" in result.reason
    assert "models" not in run.state

    run = make_run(block, repo, runner=Runner(score=score_exits_3(1)),
                   state=scored_state(tmp_path))
    run.results["score"] = result = fw.stage_score(run)
    assert result.status == fw.PAUSED and "1 scored" in result.reason
    assert any("could not score every case" in i for i in run.issues)
    assert fw.overall_status(run) == ("WARN", fw.EXIT_OK)
    assert "models" not in run.state


def test_score_refuses_a_candidate_trained_on_a_score_case(tmp_path, block, repo):
    frozen = tmp_path / "frozen"
    write_jsonl(frozen / "train.jsonl", [{"id": "c1", "task_id": "t1"}])   # the case's id
    write_jsonl(frozen / "val.jsonl", [])
    state = scored_state(tmp_path)
    state["adapters"]["fw-2026-10-31"] = {"data_dir": str(frozen)}
    runner = Runner(score=score_ok)
    result = fw.stage_score(make_run(block, repo, runner=runner, state=state))
    assert result.status == fw.REFUSED and "share an id, task or group" in result.reason
    assert runner.calls == []


def test_the_gate_is_against_the_base_model_and_wins_against_the_previous(
        tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED),
             Runner(build=build_ok(), train=train_finishes, score=score_ok), repo)
    write_jsonl(tmp_path / "golden" / "golden_traces_2.jsonl", [trace(50), trace(51)])
    second = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    run_main(tmp_path, merge(block, APPROVED), second, repo, NEXT_SAT)

    assert arg(second.argv("score-compare"), "--base") == str(tmp_path / "baseline.summary.json")
    m = report(tmp_path)["model"]
    assert m["candidate"] == "fw-2026-11-07-max-derived"
    assert m["gate_against"] == "baseline (baseline.summary.json)"
    assert m["previous"] == "fw-2026-10-31-max-derived"
    assert m["wins"] == [] and m["losses"] == []      # score_ok scores every week alike
    md = (tmp_path / "fw" / "reports" / "LATEST.md").read_text()
    assert "Phase 6 gate vs baseline (baseline.summary.json)" in md
    assert "Compared with fw-2026-10-31-max-derived (a comparison, not the gate)" in md


# --------------------------------------------------------------------------- #
# Wins/losses, spend
# --------------------------------------------------------------------------- #


def test_wins_and_losses_between_summaries():
    out = fw.wins_and_losses(summary("a", 0.40, 0.90), summary("b", 0.405, 0.70))
    assert out["wins"] == []
    assert [x["metric"] for x in out["losses"]] == ["eval-longctx / recall_by_depth @32k"]
    assert out["unchanged"] == 1


def test_spend_comes_from_the_ledger(tmp_path):
    ledger = tmp_path / "ledger.csv"
    fields = ("ts", "phase", "calls", "input_tokens", "output_tokens", "est_usd",
              "quota_units", "quota_unit", "model", "note")
    with open(ledger, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow({"ts": "2026-10-29T03:00:00+00:00", "phase": "phase2", "calls": 3,
                    "input_tokens": 100, "output_tokens": 10, "est_usd": 1.5, "model": "m"})
        w.writerow({"ts": "2026-09-01T03:00:00+00:00", "phase": "phase2", "calls": 1,
                    "est_usd": 2.0, "quota_units": 4, "quota_unit": "requests"})
        w.writerow({"ts": "garbage", "phase": "phase2b", "calls": 1, "est_usd": 0.25})
    out = fw.spend_summary(ledger, SAT - timedelta(days=7))
    assert out["week"]["est_usd"] == 1.5 and out["week"]["calls"] == 3
    assert out["total"]["est_usd"] == 3.75 and out["total"]["quota"] == {"requests": 4.0}
    assert set(out["by_phase"]) == {"phase2", "phase2b"} and out["undated_rows"] == 1
    missing = fw.spend_summary(tmp_path / "none.csv", SAT)
    assert missing["present"] is False


# --------------------------------------------------------------------------- #
# run_command: a real child held to its stop and kill times
# --------------------------------------------------------------------------- #


def test_a_child_gets_sigterm_at_its_stop_time_and_may_checkpoint(tmp_path):
    code = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: (print('checkpointed', flush=True), "
            "sys.exit(75)))\n"
            "time.sleep(30)\n")
    now = datetime.now(timezone.utc)
    res = fw.run_command(fw.CommandSpec(
        [sys.executable, "-c", code], tmp_path, tmp_path / "log.txt",
        stop_at=now + timedelta(seconds=0.5), kill_at=now + timedelta(seconds=20), poll_s=0.05))
    assert res.stopped is True and res.returncode == 75
    assert "checkpointed" in (tmp_path / "log.txt").read_text()


def test_a_child_that_ignores_sigterm_is_killed_at_its_kill_time(tmp_path):
    code = ("import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n")
    now = datetime.now(timezone.utc)
    res = fw.run_command(fw.CommandSpec(
        [sys.executable, "-c", code], tmp_path, tmp_path / "log.txt",
        stop_at=now, kill_at=now + timedelta(seconds=0.5), poll_s=0.05))
    assert res.stopped is True and res.returncode == -9


# --------------------------------------------------------------------------- #
# Shipped, not installed; light to import
# --------------------------------------------------------------------------- #


def test_the_systemd_units_are_shipped_as_templates():
    units = REPO / "deploy" / "systemd"
    timer = (units / "prometheus-flywheel-weekly.timer").read_text()
    service = (units / "prometheus-flywheel-weekly.service").read_text()
    recover = (units / "prometheus-flywheel-recover.service").read_text()
    assert "OnCalendar=Sat *-*-* 02:00:00 America/New_York" in timer
    assert "Persistent=false" in timer
    assert "-m prometheus.jobs.flywheel_weekly\n" in service
    assert "OnFailure=prometheus-flywheel-recover.service" in service
    assert "-m prometheus.jobs.flywheel_weekly --recover" in recover
    # Nothing in the job installs or enables a unit.
    src = Path(fw.__file__).read_text()
    assert "systemctl" not in src.replace("``systemctl", "")


def _span(text: str) -> timedelta:
    """A systemd time span such as ``4h30min`` (the units these files use)."""
    import re

    units = {"h": 3600, "min": 60, "m": 60, "s": 1}
    parts = re.findall(r"(\d+)\s*(h|min|m|s)", text)
    assert parts and "".join(n + u for n, u in parts) == text.replace(" ", ""), text
    return timedelta(seconds=sum(int(n) * units[u] for n, u in parts))


def test_the_weekly_units_outer_wall_falls_after_the_restore_and_the_report():
    """SIGTERM at the wall lands mid-restore if the wall is the window end, and
    TimeoutStopSec's SIGKILL then takes the restore and the report with it."""
    service = (REPO / "deploy" / "systemd" / "prometheus-flywheel-weekly.service").read_text()
    wall = _span(service.split("TimeoutStartSec=", 1)[1].splitlines()[0])
    window = fw.parse_config(None).window
    # The last stage is killed a minute before the end; the restore then has
    # its own limit; the report takes seconds (a minute is ample).
    assert wall >= window.length - fw.KILL_BEFORE_END + fw.SETUP_COMMAND_LIMIT + \
        timedelta(minutes=1)


def test_the_job_imports_no_training_stack():
    tree = ast.parse(Path(fw.__file__).read_text())
    imported = {(n.module or "").split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom)}
    imported |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert not imported & {"torch", "unsloth", "transformers", "peft", "modal", "trl"}


# --------------------------------------------------------------------------- #
# The Monday reviewer's input: one contract, every mode
# --------------------------------------------------------------------------- #


def reviewer_file(reports: Path) -> tuple[dict[str, Any], str]:
    """LATEST.json and LATEST.md, checked against the contract the reviewer reads."""
    rep = json.loads((reports / "LATEST.json").read_text())
    md = (reports / "LATEST.md").read_text()
    assert fw.report_problems(rep) == [], fw.report_problems(rep)
    assert rep["schema"] == fw.REPORT_SCHEMA
    assert [s["name"] for s in rep["stages"]] == list(fw.STAGES)
    assert fw.markdown_sections(md) == list(fw.REVIEWER_SECTIONS)
    title, _, status = md.splitlines()[:3]
    assert title == f"# Flywheel weekly report: {rep['run_id']}"
    assert status.startswith(f"**FLYWHEEL weekly · {rep['status']}** · ")
    assert rep["generated"] in md                         # the pair is one report
    last = rep["next_steps"][-1]
    assert last["by"] == "job" and last["step"].startswith("Next window: ")
    assert not list(reports.glob(".*.tmp-*"))             # no half-written file left
    return rep, md


def _scheduled_ok(tmp_path, block, repo):
    golden(tmp_path, 3)
    return run_main(tmp_path, merge(block, APPROVED),
                    Runner(build=build_ok(), train=train_finishes, score=score_ok), repo)


def _disabled(tmp_path, block, repo):
    return run_main(tmp_path, merge(block, {"enabled": False}), Runner(), repo)


def _config_error(tmp_path, block, repo):
    return run_main(tmp_path, merge(block, {"window": {"start": 120}}), Runner(), repo)


def _config_error_report_only(tmp_path, block, repo):
    return run_main(tmp_path, merge(block, {"train": {"min_new_trace": 1}}), Runner(), repo,
                    FRIDAY, "--report-only")


def _state_error(tmp_path, block, repo):
    (tmp_path / "fw").mkdir()
    (tmp_path / "fw" / "state.json").write_text("{not json")
    return run_main(tmp_path, block, Runner(), repo)


def _crash_outside_a_stage(tmp_path, block, repo):
    def half_then_crash(run: fw.Run) -> None:
        run.results["harvest"] = fw.StageResult("harvest", fw.OK, summary="0 new")
        raise RuntimeError("the pipeline itself broke")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fw, "run_pipeline", half_then_crash)
        return run_main(tmp_path, block, Runner(), repo)


def _recovery(tmp_path, block, repo):
    cfg = fw.parse_config(block)
    fw.save_state(cfg.data_dir / "state.json", {
        "schema": fw.STATE_SCHEMA, "current_run": {"id": "2026-10-31", "status": "running"}})
    return fw.recover(cfg, clock=Clock(SAT + timedelta(hours=4)), runner=Runner(), repo=repo)


def _report_only_fresh(tmp_path, block, repo):
    return run_main(tmp_path, block, Runner(), repo, FRIDAY, "--report-only")


def _dry_run(tmp_path, block, repo):
    golden(tmp_path, 3)
    return run_main(tmp_path, block, Runner(), repo, FRIDAY, "--dry-run")


@pytest.mark.parametrize("make, code, status, mode, kind", [
    (_scheduled_ok, fw.EXIT_OK, "OK", "scheduled", ""),
    (_disabled, fw.EXIT_OK, "OFF", "scheduled", ""),
    (_config_error, fw.EXIT_CONFIG, "FAIL", "config-error", ""),
    (_config_error_report_only, fw.EXIT_CONFIG, "FAIL", "config-error", ""),
    (_state_error, fw.EXIT_FAILED, "FAIL", "state-error", ""),
    (_crash_outside_a_stage, fw.EXIT_FAILED, "FAIL", "scheduled", ""),
    (_recovery, fw.EXIT_OK, "FAIL", "recovery", ""),
    (_report_only_fresh, fw.EXIT_OK, "WARN", "report-only", ""),
    (_dry_run, fw.EXIT_OK, "OK", "dry-run", "dry-run"),
])
def test_the_reviewer_file_has_every_section_in_every_mode(tmp_path, block, repo, make, code,
                                                           status, mode, kind):
    assert make(tmp_path, block, repo) == code
    rep, md = reviewer_file(tmp_path / "fw" / "reports" / kind)
    assert (rep["status"], rep["mode"]) == (status, mode)
    if mode == "config-error":
        assert all(s["status"] == fw.NOT_RUN for s in rep["stages"])
        # Only the fix: every other step would come from the fallback defaults.
        assert [s["by"] for s in rep["next_steps"]] == ["operator", "job"]
        assert steps_by(rep, "operator")[0].startswith("Fix the flywheel: block")
    if make is _crash_outside_a_stage:
        assert [s["status"] for s in rep["stages"]] == [fw.OK] + [fw.NOT_RUN] * 5
        assert any("the pipeline itself broke" in i for i in rep["issues"])


def steps_by(rep: dict[str, Any], by: str) -> list[str]:
    return [s["step"] for s in rep["next_steps"] if s["by"] == by]


def test_a_report_only_refresh_after_a_config_error_keeps_the_runs_dated_copy(
        tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, merge(block, APPROVED),
             Runner(build=build_ok(), train=train_finishes, score=score_ok), repo)
    dated = (tmp_path / "fw" / "reports" / "2026-10-31-weekly.md").read_bytes()
    _config_error_report_only(tmp_path, block, repo)
    reports = tmp_path / "fw" / "reports"
    assert (reports / "2026-10-31-weekly.md").read_bytes() == dated
    assert (reports / "2026-10-30-weekly-report-only.md").read_text() == \
        (reports / "LATEST.md").read_text()


def test_report_only_with_no_run_says_what_holds_each_stage(tmp_path, repo):
    """The template config (flywheel off, nothing configured) with Max as the
    only teacher, no run ever, operator context given."""
    data = yaml.safe_load((REPO / "config" / "prometheus.yaml.default").read_text())["flywheel"]
    data["data_dir"] = str(tmp_path / "fw")
    data["spend_ledger"] = str(tmp_path / "no-ledger.csv")
    data["harvest"]["teacher_models"] = ["qwen3.8-max"]
    runner = Runner()
    code = run_main(tmp_path, data, runner, repo, FRIDAY, "--report-only",
                    "--note", "Training on the hosted teacher's traces is on hold (#6).",
                    "--next-step", "Sat ~07:00: rehearse the GPU swap, then generate gym traces.")
    assert code == fw.EXIT_OK and runner.calls == []
    assert not (tmp_path / "fw" / "state.json").exists()      # nothing but the reports
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    assert (rep["status"], rep["mode"], rep["reported_run"]) == ("OFF", "report-only", None)
    assert rep["run_id"] == "2026-10-30"
    for s in rep["stages"]:
        assert s["status"] == fw.NOT_RUN
        assert s["reason"].startswith("no run has been recorded yet; flywheel.enabled is false")
    train = stage(rep, "train")["reason"]
    assert "0 new verified trace(s) since the last train; a train needs 500" in train
    assert "flywheel.train.command is empty" in train and "flywheel.gpu_setup is empty" in train
    assert ("names Max and no Q6b allow is recorded in flywheel.max_derived "
            "(DISTILL-OPEN-QUESTIONS #6)") in train
    operator = steps_by(rep, "operator")
    assert operator[0].startswith("Record the Q6b allow to train on Max-derived rows")
    assert operator[1].startswith("flywheel.harvest.teacher_models names only Max models")
    assert operator[2].startswith("Before the flywheel can train, quantize and score, set "
                                  "flywheel.build.args")
    assert operator[3].startswith("The flywheel is off (flywheel.enabled: false)")
    assert steps_by(rep, "runbook") == [
        "Sat ~07:00: rehearse the GPU swap, then generate gym traces."]
    assert rep["next_steps"][-1]["step"] == (
        "Next window: Sat 2026-10-31 02:00-06:00 EDT (06:00-10:00 UTC); the flywheel is off, so "
        "it only writes this report.")
    assert rep["notes"] == ["Training on the hosted teacher's traces is on hold (#6)."]
    assert "report only: no stage ran; no run recorded yet" in md
    assert "- Model: no new model (train NOT RUN, quantize NOT RUN, score NOT RUN)" in md
    assert "| harvest | NOT RUN | - | not run (see below) |" in md      # said once, under Why
    assert "No new model this week: train NOT RUN, quantize NOT RUN, score NOT RUN" in md
    assert "No spend ledger at" in md
    assert (tmp_path / "fw" / "reports" / "2026-10-30-weekly-report-only.md").read_text() == md


def test_report_only_leaves_a_live_run_alone(tmp_path, block, repo):
    with lock_held(tmp_path / "fw" / "flywheel.lock"):
        assert run_main(tmp_path, block, Runner(), repo, FRIDAY, "--report-only") == fw.EXIT_BUSY
    assert not (tmp_path / "fw" / "reports").exists()


def test_report_only_flags_a_missed_window_and_a_run_that_died(tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, block, Runner(build=build_ok()), repo)      # Sat 10-31: train refused
    state_path = tmp_path / "fw" / "state.json"
    state = json.loads(state_path.read_text())
    two_weeks_on = datetime(2026, 11, 9, 9, 0, tzinfo=ET)
    run_main(tmp_path, block, Runner(), repo, two_weeks_on, "--report-only")
    rep, _ = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["reported_run"] == "2026-10-31" and rep["status"] == "WARN"
    assert stage(rep, "train")["status"] == fw.REFUSED          # the run's own record
    assert any(i.startswith("No run is recorded for the 2026-11-07 window (the newest is "
                            "2026-10-31)") for i in rep["issues"])
    # The record says running, but nothing holds the lock: it died.
    state["current_run"]["status"] = "running"
    fw.save_state(state_path, state)
    before = state_path.read_bytes()
    run_main(tmp_path, block, Runner(), repo, two_weeks_on, "--report-only")
    rep, _ = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["status"] == "FAIL"
    assert any("recorded as running but no process holds the run lock" in i
               for i in rep["issues"])
    assert state_path.read_bytes() == before                    # report-only never writes state


def test_report_only_after_a_config_error_window_carries_that_windows_fail(
        tmp_path, block, repo):
    """Sat 10-31 runs; Sat 11-07 refuses a typo (it writes only its report);
    the operator fixes the block and refreshes on Monday."""
    golden(tmp_path, 3)
    run_main(tmp_path, block, Runner(build=build_ok()), repo)
    typo = merge(block, {"train": {"min_new_trace": 100}})
    assert run_main(tmp_path, typo, Runner(), repo, NEXT_SAT) == fw.EXIT_CONFIG
    monday = datetime(2026, 11, 9, 9, 0, tzinfo=ET)
    assert run_main(tmp_path, block, Runner(), repo, monday, "--report-only") == fw.EXIT_OK
    rep, _ = reviewer_file(tmp_path / "fw" / "reports")
    assert (rep["reported_run"], rep["status"]) == ("2026-11-07", "FAIL")
    assert "(2026-11-07 run) config: flywheel.train.min_new_trace is not a flywheel setting" \
        in rep["issues"]
    assert not any(i.startswith("No run is recorded") for i in rep["issues"])
    for s in rep["stages"]:
        assert (s["status"], s["reason"]) == (fw.NOT_RUN,
                                              "the flywheel config is invalid (see Issues)")


def test_report_only_after_a_disabled_run_keeps_its_reasons(tmp_path, block, repo):
    off = merge(block, {"enabled": False})
    assert run_main(tmp_path, off, Runner(), repo) == fw.EXIT_OK
    monday = datetime(2026, 11, 2, 9, 0, tzinfo=ET)
    assert run_main(tmp_path, off, Runner(), repo, monday, "--report-only") == fw.EXIT_OK
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    assert (rep["reported_run"], rep["status"]) == ("2026-10-31", "OFF")
    for s in rep["stages"]:
        assert s["status"] == fw.SKIPPED
        assert s["reason"] == "flywheel.enabled is false: the timer fired, the flywheel is off"
    assert "did not reach it" not in md


def test_report_only_with_no_stage_record_or_report_says_what_holds_each_stage(
        tmp_path, block, repo):
    cfg = fw.parse_config(block)
    fw.save_state(cfg.data_dir / "state.json", {
        "schema": fw.STATE_SCHEMA, "current_run": {"id": "2026-10-31", "status": "done"}})
    run_main(tmp_path, block, Runner(), repo, datetime(2026, 11, 2, 9, 0, tzinfo=ET),
             "--report-only")
    rep, _ = reviewer_file(tmp_path / "fw" / "reports")
    harvest = stage(rep, "harvest")
    assert harvest["status"] == fw.NOT_RUN
    assert harvest["reason"].startswith("the 2026-10-31 run left no stage record; no harvest "
                                        "source matches a file")


def test_report_only_inside_the_window_after_its_run_names_the_next_window(
        tmp_path, block, repo):
    golden(tmp_path, 3)
    run_main(tmp_path, block, Runner(build=build_ok()), repo)
    later = datetime(2026, 10, 31, 5, 0, tzinfo=ET)        # the same window, after its run
    run_main(tmp_path, block, Runner(), repo, later, "--report-only")
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["reported_run"] == "2026-10-31"
    assert rep["next_window"]["start"] == "2026-11-07T02:00-05:00"
    assert rep["next_steps"][-1]["step"] == \
        "Next window: Sat 2026-11-07 02:00-06:00 EST (07:00-11:00 UTC)."


def test_report_only_with_an_unreadable_state_names_the_coming_window(tmp_path, block, repo):
    (tmp_path / "fw").mkdir()
    (tmp_path / "fw" / "state.json").write_text("{not json")
    assert run_main(tmp_path, block, Runner(), repo, FRIDAY, "--report-only") == fw.EXIT_FAILED
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["mode"] == "state-error"
    assert rep["next_window"]["start"] == "2026-10-31T02:00-04:00"
    assert "- Train: unknown (the state is unreadable)" in md
    assert rep["traces"]["new_verified_since_last_train"] is None


def test_metrics_only_one_side_scored_are_listed_as_not_compared(tmp_path, block, repo):
    previous = summary("base-9b", 0.40, 0.90)
    previous["sets"]["sanity"] = {"metrics": {"sanity": {"k": 4, "n": 5, "rate": 0.8}}}
    candidate = summary("fw-2026-10-31", 0.55, 0.90)
    candidate["sets"]["eval-injection"] = {"metrics": {"resistance": {"k": 3, "n": 4,
                                                                      "rate": 0.75}}}
    paths = {}
    for name, body in (("prev", previous), ("cand", candidate)):
        paths[name] = tmp_path / f"{name}.summary.json"
        paths[name].write_text(json.dumps(body))
    state = {"schema": fw.STATE_SCHEMA, "models": [{
        "adapter_version": "fw-2026-10-31", "scored_in": "2026-10-31", "verdict": "PASS",
        "summary": str(paths["cand"]), "against": "baseline (prev.summary.json)",
        "against_summary": str(paths["prev"]), "gate_against": "baseline (prev.summary.json)"}]}
    run = make_run(block, repo, state=state)
    for name in fw.STAGES:
        run.results[name] = fw.StageResult(name, fw.OK)
    rep = fw.build_report(run, "OK")
    assert rep["model"]["only_previous"] == ["sanity / sanity"]
    assert rep["model"]["only_candidate"] == ["eval-injection / resistance"]
    md = fw.render_markdown(rep)
    section = md.split("## Model: wins and losses vs the previous model")[1].split("\n## ")[0]
    assert ("Not compared (baseline (prev.summary.json) only, the candidate has no score): "
            "sanity / sanity.") in section
    assert ("Not compared (the candidate only, baseline (prev.summary.json) has no score): "
            "eval-injection / resistance.") in section
    assert "1 win(s), 0 loss(es), 2 metric(s) not compared" in md


def test_a_report_that_cannot_be_built_is_replaced_by_a_minimal_one(tmp_path, block, repo,
                                                                    monkeypatch):
    def broken(run: fw.Run, status: str) -> dict[str, Any]:
        raise KeyError("summary")

    monkeypatch.setattr(fw, "build_report", broken)
    golden(tmp_path, 3)
    code = run_main(tmp_path, block, Runner(build=build_ok()), repo)
    assert code == fw.EXIT_FAILED
    rep, md = reviewer_file(tmp_path / "fw" / "reports")
    assert rep["status"] == "FAIL"
    assert any("the full report could not be built (KeyError" in i for i in rep["issues"])
    assert stage(rep, "build")["status"] == fw.OK            # what the run did still shows
    assert "Spend ledger not read" in md


def test_a_report_off_its_schema_is_still_written_and_marked_fail(tmp_path, block, repo):
    run = make_run(block, repo)
    for name in fw.STAGES:
        run.results[name] = fw.StageResult(name, fw.SKIPPED, reason="r")
    rep = fw.build_report(run, "OK")
    rep["stages"][0]["status"] = "MAYBE"
    fw.write_report(run.data, rep)
    written = json.loads((run.data / "reports" / "LATEST.json").read_text())
    assert written["status"] == "FAIL"
    assert any(i.startswith("report schema: stages.0.status") for i in written["issues"])


def test_the_schema_model_names_the_schema_constant():
    import typing

    annotation = fw.WeeklyReport.model_fields["report_schema"].annotation
    assert typing.get_args(annotation) == (fw.REPORT_SCHEMA,)
    assert typing.get_args(fw.WeeklyReport.model_fields["mode"].annotation) == fw.REPORT_MODES
    assert typing.get_args(fw.ReportStep.model_fields["by"].annotation) == fw.STEP_OWNERS


def test_report_only_is_its_own_mode_on_the_command_line(tmp_path, block, repo):
    with pytest.raises(SystemExit) as exc:
        run_main(tmp_path, block, Runner(), repo, FRIDAY, "--report-only", "--dry-run")
    assert exc.value.code == 2


# --------------------------------------------------------------------------- #
# Windows across daylight-saving changes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("now, next_start, last_start", [
    (datetime(2026, 10, 30, 14, 0, tzinfo=ET), "2026-10-31T02:00:00-04:00",
     "2026-10-24T02:00:00-04:00"),
    (datetime(2026, 10, 31, 4, 0, tzinfo=ET), "2026-10-31T02:00:00-04:00",     # inside it
     "2026-10-31T02:00:00-04:00"),
    (datetime(2026, 10, 31, 6, 0, tzinfo=ET), "2026-11-07T02:00:00-05:00",
     "2026-10-31T02:00:00-04:00"),
    # 01:30 on the change night, the second time (EST)
    (datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=ET), "2026-11-07T02:00:00-05:00",
     "2026-10-31T02:00:00-04:00"),
    # the first Saturday after the March change (Sun 2027-03-14)
    (datetime(2027, 3, 13, 7, 0, tzinfo=ET), "2027-03-20T02:00:00-04:00",
     "2027-03-13T02:00:00-05:00"),
])
def test_windows_across_daylight_saving_changes(now, next_start, last_start):
    window = fw.parse_config(None).window
    start, end = fw.next_window(now.astimezone(timezone.utc), window)
    assert start.isoformat() == next_start and end - start == timedelta(hours=4)
    assert fw.last_window_start(now.astimezone(timezone.utc), window).isoformat() == last_start


def test_the_timer_hour_after_the_november_change_is_utc_seven():
    window = fw.parse_config(None).window
    assert fw.window_bounds(datetime(2026, 11, 7, 6, 0, 30, tzinfo=timezone.utc), window) is None
    start, end = fw.window_bounds(datetime(2026, 11, 7, 7, 0, 30, tzinfo=timezone.utc), window)
    assert fw.describe_window(start, end) == "Sat 2026-11-07 02:00-06:00 EST (07:00-11:00 UTC)"
    assert fw.describe_window(*fw.window_on(datetime(2026, 10, 31).date(), window)) == \
        "Sat 2026-10-31 02:00-06:00 EDT (06:00-10:00 UTC)"
