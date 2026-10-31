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
APPROVED = {"max_data": {"approved": True, "approved_by": "operator",
                         "approved_on": "2020-01-01", "reference": "test decision"}}


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
    return {
        "enabled": True, "data_dir": str(tmp_path / "fw"),
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
    fixes them, so each week's build holds rows no earlier train had."""
    def handler(spec: fw.CommandSpec) -> fw.CommandResult:
        out = Path(arg(spec.argv, "--out"))
        prefix = tag or out.parent.name
        train = [{"id": f"p3-{prefix}-{i}", "split": "train", "source": source,
                  "label": label} for i in range(rows)]
        write_jsonl(out / "train.jsonl", train)
        write_jsonl(out / "val.jsonl", train[:1])
        write_jsonl(out / "val-cases.jsonl", [{"id": "v"}])
        write_jsonl(out / "eval-tools.jsonl", [{"id": "e"}])
        (out / "report.json").write_text(json.dumps({
            "settings": {"teacher_model": teacher},
            "gates": {"train_size": {"pass": gates == "PASS"},
                      "spot_check": {"status": "PASS" if gates == "PASS" else "NOT RUN"},
                      "overall": gates},
            "inputs": {"written": {"train.jsonl": rows, "val.jsonl": 1,
                                   "eval-tools.jsonl": 1}}}))
        return fw.CommandResult(0 if gates == "PASS" else 1)
    return handler


def train_finishes(spec: fw.CommandSpec) -> int:
    run_dir = Path(arg(spec.argv, "--out"))
    (run_dir / "adapter").mkdir(parents=True, exist_ok=True)
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
    assert cfg.max_approval.approved is False
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
    ({"max_data": {"approved": True, "approved_on": "2020-01-01"}}, "approved_by is empty"),
    ({"max_data": {"approved": True, "approved_by": "op"}}, "YYYY-MM-DD"),
    ({"max_data": {"approved": True, "approved_by": "op", "approved_on": "2999-01-01"}},
     "in the future"),
    ({"max_data": {"approved": "yes"}}, "true or false"),
    ({"build": {"args": ["--out", "x"]}}, "may not set --out"),
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


def test_a_recorded_approval_parses(block):
    cfg = fw.parse_config(merge(block, APPROVED))
    assert cfg.max_approval.approved and cfg.max_approval.approved_by == "operator"


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
    # wins and losses vs the previous model (the baseline summary here)
    assert rep["model"]["candidate"] == "fw-2026-10-31"
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
    assert "0 train row(s) that fw-2026-10-31 was not trained on" in train["reason"]
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
    assert frozen.endswith("train/fw-2026-10-31.data/train.jsonl")

    # Next Saturday: no new traces (below the bar), yet the train resumes.
    second = Runner(build=build_ok(), train=train_finishes, score=score_ok)
    run_main(tmp_path, merge(block, APPROVED), second, repo, NEXT_SAT)
    resumed = second.argv("train")
    assert "--resume" in resumed
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
    assert off["verification"] == {"basis": "judge-pass", "by": "judge-27b"}


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


def _seed_sets(run: fw.Run, rows: list[dict[str, Any]], teacher: str) -> None:
    sets = run.run_dir / "sets"
    write_jsonl(sets / "train.jsonl", rows)
    write_jsonl(sets / "val.jsonl", rows[-1:])
    (sets / "report.json").write_text(json.dumps({"settings": {"teacher_model": teacher}}))
    (run.run_dir / "verify.json").write_text(json.dumps({"verified": 10}))
    run.state["sets"] = {"run": run.run_id, "dir": str(sets), "gates": "PASS",
                         "teacher_model": teacher}


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
    assert stage(rep, "score")["details"]["summary"] == "~/fw/scores/fw-2026-10-31.summary.json"


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
                                                                "data_dir": "d"}}}
    run = make_run(merge(block, {"gpu_setup": ""}), repo, runner=runner, state=waiting)
    quantize = fw.stage_quantize(run)
    assert quantize.status == fw.SKIPPED and "'none'" in quantize.reason
    assert runner.names() == ["build"]


def test_a_gpu_swap_starts_only_with_time_for_it_and_is_held_to_the_stop_time(
        tmp_path, block, repo):
    swapping = merge(block, {"gpu_setup": "flywheel"})

    def waiting() -> dict[str, Any]:
        return {"schema": fw.STATE_SCHEMA, "adapters": {"fw-x": {"adapter_dir": "a",
                                                                 "data_dir": "d"}}}

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
    assert m["candidate"] == "fw-2026-11-07"
    assert m["gate_against"] == "baseline (baseline.summary.json)"
    assert m["previous"] == "fw-2026-10-31"
    assert m["wins"] == [] and m["losses"] == []      # score_ok scores every week alike
    md = (tmp_path / "fw" / "reports" / "LATEST.md").read_text()
    assert "Phase 6 gate vs baseline (baseline.summary.json)" in md
    assert "Compared with fw-2026-10-31 (a comparison, not the gate)" in md


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


def test_the_job_imports_no_training_stack():
    tree = ast.parse(Path(fw.__file__).read_text())
    imported = {(n.module or "").split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom)}
    imported |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                 for a in n.names}
    assert not imported & {"torch", "unsloth", "transformers", "peft", "modal", "trl"}
