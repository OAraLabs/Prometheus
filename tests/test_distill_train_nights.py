"""The round-2 multi-night job: scripts/distill/tower/train_nights.py, its job_runner.py step and the
post_train.py --window-hold mode.

Three layers, no GPU, no systemd, no host:
* the job's state machine, Telegram, retention and ETA arithmetic, and the nights estimate (pure);
* the 3090 Ti host's nightly trigger (``mini-night``) and post_train.py's hold rule, with a fake ssh;
* the window contract end to end: the real job runner (tests/fixtures/distill_job_runner fakes for
  nvidia-smi / systemctl / systemd-run) runs the REAL train_qlora.py loop on a fake backend
  (tests/fixtures/distill_train_nights/trainer_driver.py) across simulated nights the way
  window_start and window_end drive it: the runner's dry run first, then ``--execute --stop-at``;
  a night ends by stop-at or by SIGTERM to the runner's main process only (KillMode=mixed) with
  window.conf's JOB_STOP_TIMEOUT_S, before the (simulated) business restore. Then post-train nights
  hold the window for a simulated pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests.test_distill_job_runner import BASE_REV, RUNNER_SH, Box

REPO = Path(__file__).resolve().parents[1]
DIST = REPO / "scripts" / "distill"
DRIVER = REPO / "tests" / "fixtures" / "distill_train_nights" / "trainer_driver.py"

pytestmark = pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="posix only")


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TN = _load("distill_train_nights", DIST / "tower" / "train_nights.py")
PT = _load("distill_post_train", DIST / "post_train.py")


# ---------------------------------------------------------------------------
# the state machine (pure)
# ---------------------------------------------------------------------------


def _rs(step: int, total: int = 100, tokens: int = 0, left: int = 0, hist: list | None = None,
        epochs_done: int = 0, tok_s: float | None = 2000.0) -> dict:
    return {"progress": {"step": step, "total_steps": total, "tokens_seen": tokens, "tokens_left": left,
                         "epoch": min(epochs_done + 1, 2), "epochs": 2, "tok_s": tok_s},
            "val_history": hist if hist is not None else [{"step": 0, "val_loss": 1.0, "probe_loss": 1.13}],
            "epochs_done": [{"epoch": e + 1} for e in range(epochs_done)], "schedule": {"total_steps": total}}


GOOD_HIST = [{"step": 0, "val_loss": 1.0, "probe_loss": 1.13}, {"step": 50, "val_loss": 0.9, "probe_loss": 1.0},
             {"step": 100, "val_loss": 0.8, "probe_loss": 0.88}]


def _done_man(hist: list = GOOD_HIST) -> dict:
    return {"status": "done", "mode": "train", "step": 100, "tokens_seen": 9e7, "adapter": "/x/adapter-max-derived",
            "config": {"epochs": 2}, "epochs_done": [{"epoch": 1}, {"epoch": 2}], "val_history": hist}


def test_a_job_trains_across_nights_then_requests_post_train_then_is_done(tmp_path: Path) -> None:
    st = TN.new_state("9b-r32-r2-max-derived", "/r/9b-r32-r2-max-derived", 1_000.0)
    rec = TN.begin_night(st, 1_000.0, None, window_s=18_000)
    status, detail = TN.end_train_night(st, rec, rc=0, reason=None, man={"status": "stopped", "reason": "stop-at"},
                                        rs=_rs(40, tokens=4e7, left=6e7), adapter_present=False, t=19_000.0,
                                        train_s=18_000)
    assert (status, st["phase"], rec["outcome"]) == ("partial", "train", "stopped") and "resumes" in detail
    assert rec["step_from"] == 0 and rec["step_to"] == 40 and rec["tokens_to"] == 4e7
    rec = TN.begin_night(st, 90_000.0, _rs(40, tokens=4e7), window_s=18_000)
    assert rec["step_from"] == 40
    status, _ = TN.end_train_night(st, rec, rc=0, reason=None, man=_done_man(),
                                   rs=_rs(100, tokens=9e7, hist=GOOD_HIST, epochs_done=2),
                                   adapter_present=True, t=100_000.0, train_s=12_000)
    assert status == "partial" and st["phase"] == "post-train" and st["verdict"]["ok"]
    assert st["post_train"]["requested_at"] and rec["outcome"] == "trained"
    TN.save(tmp_path, st)
    # post-train: a hold the pipeline took and completed
    night = TN.night_key(180_000.0)
    TN.write_json(tmp_path / TN.taken_name(night), {"night": night})
    TN.write_json(tmp_path / TN.DONE_FILE, {"outcome": "completed"})
    rec = TN.begin_night(st, 180_000.0, None, window_s=18_000)
    status, detail = TN.end_hold(st, rec, TN.hold_events(tmp_path, night), why="the pipeline completed",
                                 t=190_000.0)
    assert status == "done" and st["phase"] == "done" and st["post_train"]["done"]["outcome"] == "completed"
    assert TN.load(tmp_path)["phase"] == "post-train"  # saved before: save is the caller's
    TN.save(tmp_path, st)
    assert TN.load(tmp_path)["phase"] == "done"


def test_a_crash_with_last_nights_manifest_is_a_failure_and_two_park_the_job() -> None:
    st = TN.new_state("r", "/r", 0.0)
    stale = {"status": "stopped", "reason": "stop-at", "step": 40}  # last night's
    for i in (1, 2):
        rec = TN.begin_night(st, float(i * 86_400), _rs(40), window_s=18_000)
        status, detail = TN.end_train_night(st, rec, rc=1, reason=None, man=stale, rs=_rs(40),
                                            adapter_present=False, t=float(i * 86_400 + 60), train_s=60)
        assert status == "failed" and rec["outcome"] == "failed" and "exit 1" in detail
    assert st["phase"] == "failed" and st["failures"] == 2 and "parked" in st["last_error"]
    # a stop by the window resets the count
    st2 = TN.new_state("r", "/r", 0.0)
    rec = TN.begin_night(st2, 0.0, None, window_s=1)
    TN.end_train_night(st2, rec, rc=1, reason=None, man=None, rs=None, adapter_present=False, t=1.0, train_s=1)
    rec = TN.begin_night(st2, 2.0, None, window_s=1)
    status, _ = TN.end_train_night(st2, rec, rc=None, reason="sigterm", man=None, rs=_rs(5), adapter_present=False,
                                   t=3.0, train_s=1)
    assert status == "partial" and st2["failures"] == 0 and st2["phase"] == "train"


@pytest.mark.parametrize("hist,why", [
    ([{"step": 0, "val_loss": 1.0, "probe_loss": 1.1}, {"step": 100, "val_loss": 1.01, "probe_loss": 0.9}],
     "val loss did not fall"),
    ([{"step": 0, "val_loss": 1.0, "probe_loss": 1.1}, {"step": 100, "val_loss": 0.9, "probe_loss": 1.2}],
     "probe loss did not fall"),
    ([{"step": 0, "val_loss": 1.0, "probe_loss": 1.1}, {"step": 100, "val_loss": float("nan"), "probe_loss": 1.0}],
     "non-finite"),
    ([{"step": 100, "val_loss": 0.9, "probe_loss": 1.0}], "no held-out evaluation after step 0"),
])
def test_the_operators_gate_on_the_finished_run(hist: list, why: str) -> None:
    st = TN.new_state("r", "/r", 0.0)
    rec = TN.begin_night(st, 0.0, None, window_s=1)
    status, detail = TN.end_train_night(st, rec, rc=0, reason=None, man=_done_man(hist), rs={"val_history": hist},
                                        adapter_present=True, t=1.0, train_s=1)
    assert status == "failed" and st["phase"] == "failed" and why in detail
    # the same rule as the post-train pipeline's own verdict
    assert PT.training_verdict(_done_man(hist), {}, adapter_present=True)[0] is False
    ok, _, _ = TN.training_verdict(_done_man(), {}, adapter_present=False)
    assert ok is False
    assert TN.training_verdict(_done_man(), {}, adapter_present=True)[0] is True
    assert PT.training_verdict(_done_man(), {}, adapter_present=True)[0] is True


def test_a_job_dir_belongs_to_one_run() -> None:
    st = TN.new_state("a-max-derived", "/r/a-max-derived", 0.0)
    assert TN.identity_problem(st, "a-max-derived", "/r/a-max-derived") is None
    assert "new job directory" in TN.identity_problem(st, "b-max-derived", "/r/b-max-derived")


def test_eta_uses_the_measured_nightly_yield() -> None:
    st = TN.new_state("r", "/r", 0.0)
    st["nights"] = [
        {"phase": "train", "outcome": "stopped", "train_s": TN.NIGHT_S, "tokens_from": 0, "tokens_to": 40_000_000},
        {"phase": "train", "outcome": "stopped", "train_s": 600, "tokens_from": 4e7, "tokens_to": 41_000_000},  # short
        {"phase": "train", "outcome": "stopped", "train_s": TN.NIGHT_S, "tokens_from": 4.1e7, "tokens_to": 85_000_000},
    ]
    per = TN.tokens_per_night(st)
    assert per == pytest.approx((40e6 + 44e6) / 2)  # the short night does not count
    assert TN.eta_nights(100e6, per) == pytest.approx(100e6 / per)
    assert TN.eta_nights(None, per) is None and TN.eta_nights(1e6, None) is None


def test_the_nightly_telegram_has_steps_curve_eta_and_at_most_five_lines(tmp_path: Path) -> None:
    st = TN.new_state("r", "/r", 0.0)
    rec = TN.begin_night(st, 0.0, _rs(400, total=2960, tokens=4e7), window_s=TN.NIGHT_S)
    hist = [{"step": s, "val_loss": 1.0 - s / 10_000, "probe_loss": 1.13 - s / 9_000} for s in range(0, 1300, 100)]
    rs = _rs(1240, total=2960, tokens=86e6, left=120e6, hist=hist)
    TN.end_train_night(st, rec, rc=0, reason=None, man={"status": "stopped", "reason": "stop-at", "step": 1240,
                                                         "tokens_seen": 86e6},
                       rs=rs, adapter_present=False, t=TN.NIGHT_S, train_s=TN.NIGHT_S)
    ck = tmp_path / "checkpoint-001240"
    ck.mkdir()
    (ck / "COMPLETE").write_text("x")
    (ck / "state.json").write_text(json.dumps({"step": 1240}))
    lines = TN.night_message(st, rec, rs, ckpts=TN.checkpoints(tmp_path), free_gib=61.2, keep=2)
    assert len(lines) <= 5
    text = "\n".join(lines)
    assert "steps 400->1240 of 2960 (42%)" in text and "stopped (stop-at)" in text
    assert "Val 0:1.000" in text and "1200:0.880" in text and "probe 0:1.130" in text
    assert "ETA ~2.6 more nights" in text and "disk 61 GiB free" in text
    # the night that finishes: the gate instead of an ETA
    st2 = TN.new_state("r", "/r", 0.0)
    rec2 = TN.begin_night(st2, 0.0, None, window_s=1)
    TN.end_train_night(st2, rec2, rc=0, reason=None, man=_done_man(), rs=_rs(100, hist=GOOD_HIST, epochs_done=2),
                       adapter_present=True, t=1.0, train_s=3600)
    text2 = "\n".join(TN.night_message(st2, rec2, _rs(100, hist=GOOD_HIST), ckpts=[], free_gib=None, keep=2))
    assert "DONE" in text2 and "Gate passed" in text2 and "Post-train" in text2


def _mk_ckpt(out: Path, step: int, *, epoch_end: bool = False, complete: bool = True, size: int = 1000) -> None:
    d = out / (f"checkpoint-{step:06d}" + ("" if complete else ".partial"))
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"step": step, "epoch_end": epoch_end}))
    (d / "blob").write_bytes(b"\0" * size)
    if complete:
        (d / "COMPLETE").write_text("x")


def test_retention_is_the_newest_two_plus_each_epoch_end(tmp_path: Path) -> None:
    _mk_ckpt(tmp_path, 560, epoch_end=True)
    _mk_ckpt(tmp_path, 900)
    _mk_ckpt(tmp_path, 940)
    assert TN.retention_problem(TN.checkpoints(tmp_path), keep=2, epochs_done=1) is None
    _mk_ckpt(tmp_path, 980)
    bad = TN.retention_problem(TN.checkpoints(tmp_path), keep=2, epochs_done=1)
    assert bad and "checkpoint-000900" in bad
    shutil.rmtree(tmp_path / "checkpoint-000900")
    _mk_ckpt(tmp_path, 1000, complete=False)
    assert "incomplete" in TN.retention_problem(TN.checkpoints(tmp_path), keep=2, epochs_done=1)


def test_the_disk_need_counts_only_what_the_run_still_writes(tmp_path: Path) -> None:
    need0, lines = TN.disk_need_gib([], keep=2, epochs=2, epochs_done=0, default_ckpt_gib=0.5)
    assert need0 == pytest.approx((2 + 1 + 2) * 0.5 + 0.7) and len(lines) == 2
    _mk_ckpt(tmp_path, 100, size=2**29)  # 0.5 GiB already on disk
    need1, _ = TN.disk_need_gib(TN.checkpoints(tmp_path), keep=2, epochs=2, epochs_done=0, default_ckpt_gib=0.1)
    assert need1 == pytest.approx(5 * 0.5 - 0.5 + 0.7, rel=1e-3)


def test_the_estimate_from_measured_speeds() -> None:
    assert TN.speed_at(8000) == 2548.7 and TN.speed_at(49152) == 1990.1 and TN.speed_at(60000) == 1990.1
    assert TN.speed_at(24576) == pytest.approx((2548.7 + 2330.2) / 2)
    # round 1 for calibration: 4,480 rows averaging 13,846 tokens, 2 epochs -> ~14.4 h (measured 14.4 h wall)
    r1 = TN.estimate([13846] * 4480, epochs=2)
    assert 14.0 < r1["train_h"] < 14.8 and r1["eff_tok_s"] == pytest.approx(2400.8, rel=0.01)
    lengths = TN.synthetic_lengths(3750)
    assert len(lengths) == 3750 + 2500 and lengths.count(36000) == 975
    est = TN.estimate(lengths, epochs=2)
    assert 5.0 < est["nights"] < 7.0
    assert TN.estimate([60000], epochs=1)["tokens"] == 49152  # rows over the length are cut to it


def test_estimate_cli_reads_a_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "train.jsonl").write_text("".join(json.dumps({"id": i, "n_tokens": 20000}) + "\n" for i in range(900)))
    assert TN.main(["estimate", "--dataset", str(tmp_path / "train.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "900 rows" in out and "nights of" in out


# ---------------------------------------------------------------------------
# the 3090 Ti host's nightly trigger and post_train.py's hold rule
# ---------------------------------------------------------------------------


def _hold(night: str, until: float, state: str = "holding") -> dict:
    return {"schema": TN.HOLD_SCHEMA, "night": night, "state": state, "until_epoch": until,
            "until": "x", "unit": "distill-jobs-1.service"}


def test_the_hold_rule_is_the_same_on_both_hosts() -> None:
    t = 1_700_000_000.0
    night = TN.night_key(t)
    cases = [None, {"schema": "x"}, _hold(night, t + 7200, "ended"), _hold("2000-01-01", t + 7200),
             _hold(night, t + 600), _hold(night, t + 7200)]
    for h in cases:
        assert (TN.hold_problem(h, night, t) is None) == (PT.hold_problem(h, night, t) is None)
    assert TN.hold_problem(_hold(night, t + 7200), night, t) is None
    assert PT.HOLD_SCHEMA == TN.HOLD_SCHEMA


def test_mini_night_plans_tonight_from_the_4090_state() -> None:
    t = 1_700_000_000.0
    night = TN.night_key(t)
    st = TN.new_state("r", "/r", 0.0)
    assert TN.mini_night_plan(None, None, None, t=t)[0] == "nothing"
    assert TN.mini_night_plan(st, None, None, t=t)[0] == "nothing"  # still training
    st["phase"] = "post-train"
    assert TN.mini_night_plan(st, None, None, t=t) == ("nothing", "no hold file from the 4090 window")
    assert TN.mini_night_plan(st, _hold(night, t + 7200), None, t=t)[0] == "arm"
    assert TN.mini_night_plan(st, _hold(night, t + 7200), {"outcome": "stopped: x"}, t=t)[0] == "arm-rerun"
    assert TN.mini_night_plan(st, None, {"outcome": "completed"}, t=t)[0] == "mark-done"


def test_mini_night_arms_post_train_into_the_hold(tmp_path: Path) -> None:
    t = time.time()
    night = TN.night_key(t)
    st = TN.new_state("9b-r2-max-derived", "/r", 0.0)
    st["phase"] = "post-train"
    remote = {"nights.json": json.dumps(st), "hold.json": json.dumps(_hold(night, t + 4 * 3600))}
    calls: list[list[str]] = []

    def run(argv: Any, timeout: float = 0, stdin: Any = None) -> tuple[int, str]:
        calls.append(list(argv))
        if argv[0] == "ssh":
            name = next((k for k in remote if f"/{k} " in argv[-1]), None)
            if not argv[-1].startswith("cat "):
                return 0, ""  # a write on the 4090
            return (0, remote[name]) if name else (1, "")
        return 0, "armed"

    p7 = tmp_path / "p7"
    p7.mkdir()
    (p7 / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "stopped: the stop time was reached"}))
    a = TN.build_parser().parse_args(["mini-night", "--tower", "u@h", "--run", "9b-r2-max-derived",
                                      "--state-dir", str(p7), "--post-train", "/c/post_train.py", "--execute"])
    assert TN.mini_night(a, run=run, clock=lambda: t) == 0
    arm = calls[-1]
    assert arm[2:4] == ["/c/post_train.py", "arm"] and "--rerun" in arm and "--execute" in arm
    assert arm[arm.index("--window-hold") + 1] == "distill/phase4/results/base-9b/train-nights-9b-r2-max-derived"
    stop = PT.parse_iso(arm[arm.index("--stop-at") + 1])
    hard = PT.parse_iso(arm[arm.index("--hard-at") + 1])
    assert stop == pytest.approx(t + 4 * 3600 - 900, abs=2) and hard == pytest.approx(t + 4 * 3600 - 300, abs=2)
    # completed earlier: only the 4090's done marker, idempotently
    calls.clear()
    (p7 / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed", "finished": "x"}))
    assert TN.mini_night(a, run=run, clock=lambda: t) == 0
    assert len(calls) == 3 and calls[-1][0] == "ssh" and "test -e" in calls[-1][-1] and "post-train.done.json" in calls[-1][-1]


# ---------------------------------------------------------------------------
# the window contract end to end: real runner, real trainer loop, fake backend
# ---------------------------------------------------------------------------

JOB_STOP_TIMEOUT_S = 30  # window.conf: 420 on the 4090; the trainer must be out well before


class Nights:
    """A temp HOME with the round-2 job's inputs, and window_start / window_end as they drive the runner."""

    RUN = "9b-r32-r2test-max-derived"

    def __init__(self, box: Box) -> None:
        self.box = box
        d = box.d
        code = d / "code-train" / "head" / "scripts" / "distill"
        code.mkdir(parents=True)
        for f in ("train_qlora.py", "render.py", "artifact_policy.py"):
            shutil.copy2(DIST / f, code / f)
        (d / "code-train" / "head" / "COMMIT").write_text("c" * 40 + "\n")
        self.data = d / "phase3" / "build-r2-max-derived"
        subprocess.run([sys.executable, str(DRIVER), "make-data", str(self.data), "16", "24"], check=True)
        tok = d / "models" / "qwen3.5-9b-tokenizer"
        tok.mkdir(parents=True)
        (tok / "tokenizer.json").write_text("{}")
        base = d / "models" / "qwen3.5-9b-bnb4"
        base.mkdir(parents=True)
        (base / "config.json").write_text("{}")
        (base / "model.safetensors").write_bytes(b"\0" * 16)
        (base / "MANIFEST.json").write_text(json.dumps({"repo": "x/y", "revision": BASE_REV}))
        py = box.root / "train-python"
        py.write_text('#!/bin/bash\nif [ "$1" = "-I" ] && [ "$2" = "-c" ]; then echo; exit 0; fi\n'
                      f'exec "{sys.executable}" "{DRIVER}" "$@"\n')
        py.chmod(0o755)
        notify = d / "notify.sh"
        notify.write_text('#!/bin/bash\nn=$(ls "$FAKE_STATE"/notify-*.txt 2>/dev/null | wc -l | tr -d " ")\n'
                          'cat > "$FAKE_STATE/notify-$n.txt"\n')
        notify.chmod(0o755)
        self.out = d / "phase5" / "runs" / self.RUN
        self.job = d / "phase4" / "results" / "base-9b" / f"train-nights-{self.RUN}"
        box.env.update({
            "DJR_TRAIN_PY": str(py), "DJR_NIGHTS": "1", "DJR_ONLY": "train-nights",
            "DJR_TRAIN_DATA": str(self.data / "train.jsonl"), "DJR_TRAIN_VAL": str(self.data / "val.jsonl"),
            "DJR_TRAIN_OUT": str(self.out), "DJR_TRAIN_MAX_DERIVED_OK": "1", "DJR_TRAIN_EPOCHS": "2",
            "DJR_TRAIN_MAX_SEQ": "49152", "DJR_TRAIN_LADDER": "49152,32768,16384", "DJR_TRAIN_GRAD_ACCUM": "2",
            "DJR_TRAIN_SAVE_EVERY": "1000", "DJR_TRAIN_SAVE_EVERY_MIN": "0.02", "DJR_TRAIN_EVAL_EVERY_MIN": "0.05",
            "DJR_TRAIN_KEEP": "2", "DJR_TRAIN_KEEP_EPOCHS": "1", "DJR_TRAIN_MIN_FREE_GIB": "0",
            "DJR_TRAIN_MIN_MIN": "0.01", "DJR_NIGHTS_GRACE_S": "20", "DJR_NIGHTS_HOLD_MIN": "0.3",
            "DJR_NIGHTS_HOLD_LEAST_MIN": "0.01", "DJR_NIGHTS_HOLD_POLL_S": "0.2", "DJR_NIGHTS_CKPT_GIB": "0.001",
            "FAKE_SEC_PER_TOKEN": "0.012", "FAKE_STOP_MARGIN_S": "0.5"})

    # -- the window scripts' calls -------------------------------------------------------

    def window_start(self, stop_in: float) -> subprocess.Popen:
        """window_start.sh: the runner's dry run must pass, then the runner as the unit's main process."""
        dry = self.box.run(execute=False, stop_in=stop_in)
        assert dry.returncode == 0, dry.stdout[-3000:]
        return self.box.start(stop_in=stop_in)

    def window_end(self, runner: subprocess.Popen, *, sigterm: bool) -> list[str]:
        """window_end.sh: `systemctl --user stop` of the runner unit (KillMode=mixed: SIGTERM to the main
        process only, SIGKILL to everything after JOB_STOP_TIMEOUT_S), then the restore, then the
        runner's --summary. The trainer must have checkpointed and exited BEFORE the restore."""
        if sigterm and runner.poll() is None:
            runner.send_signal(signal.SIGTERM)
        try:
            runner.wait(timeout=JOB_STOP_TIMEOUT_S if sigterm else 120)
        except subprocess.TimeoutExpired:
            runner.kill()
            raise AssertionError("the runner outlived the unit's TimeoutStopSec: SIGKILL")
        pid = int((self.box.state / "trainer.pid").read_text()) if (self.box.state / "trainer.pid").exists() else 0
        if pid:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)  # no trainer left for the restore to find
        cp = subprocess.run([str(RUNNER_SH), "--summary"], env=self.box.env, capture_output=True, text=True)
        lines = cp.stdout.splitlines()
        assert 1 <= len(lines) <= 5
        return lines

    # -- reading ---------------------------------------------------------------------------

    def steps(self, out_name: str | None = None) -> list[dict]:
        p = self.box.state / f"steps-{out_name or self.RUN}.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def state(self) -> dict:
        return json.loads((self.job / "nights.json").read_text())

    def telegrams(self) -> list[list[str]]:
        files = sorted(self.box.state.glob("notify-*.txt"), key=lambda p: int(p.stem.split("-")[1]))
        return [f.read_text().splitlines() for f in files]

    def ckpts(self) -> list[str]:
        return sorted(p.name for p in self.out.iterdir() if p.name.startswith("checkpoint-"))


@pytest.fixture
def nights(tmp_path: Path):
    box = Box(tmp_path).build()
    yield Nights(box)
    box.kill_all()


def _wait(cond: Any, timeout: float, what: str) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_round2_trains_across_nights_resumes_exactly_and_chains_into_post_train(nights: Nights) -> None:
    n = nights
    # night 1: ends by the runner's stop-at (the trainer checkpoints and exits before it)
    r = n.window_start(stop_in=7)
    summary = n.window_end(r, sigterm=False)
    assert r.returncode == 0 and "9B nights partial" in "\n".join(summary)
    s1 = len(n.steps())
    assert 0 < s1 < 16 and n.state()["phase"] == "train"
    # night 2: window_end's SIGTERM in the middle of training
    r = n.window_start(stop_in=120)
    _wait(lambda: len(n.steps()) >= s1 + 2, 60, "two more steps")
    n.window_end(r, sigterm=True)
    assert r.returncode == 143
    job = n.state()
    assert job["phase"] == "train" and job["nights"][-1]["outcome"] == "stopped"
    assert job["nights"][-1]["reason"] == "sigterm" and job["nights"][-1]["step_from"] == s1
    calls = [json.loads(x) for x in (n.box.state / "trainer-calls.log").read_text().splitlines()]
    assert "--resume" not in calls[0]["argv"] and "--resume" in calls[1]["argv"]
    for flag, want in (("--ladder", "49152,32768,16384"), ("--max-seq", "49152"), ("--eval-every-minutes", "0.05"),
                       ("--save-every-minutes", "0.02"), ("--keep-checkpoints", "2")):
        argv = calls[1]["argv"]
        assert argv[argv.index(flag) + 1] == want, flag
    assert "--keep-epoch-checkpoints" in calls[1]["argv"]
    # the SIGTERM lost nothing: the newest checkpoint is the last step trained (a stop's own, or the
    # interval one at that very step)
    newest = json.loads((n.out / n.ckpts()[-1] / "state.json").read_text())
    assert newest["step"] == len(n.steps()) == job["nights"][-1]["step_to"]
    assert newest["why"] in ("stop:SIGTERM", "interval", "epoch")
    # night 3: trains to the end; the gate passes; the job moves to post-train
    r = n.window_start(stop_in=120)
    n.window_end(r, sigterm=False)
    assert r.returncode == 0
    job = n.state()
    assert job["phase"] == "post-train" and job["verdict"]["ok"] and job["nights"][-1]["outcome"] == "trained"
    assert all(x.get("retention") == "ok" for x in job["nights"])
    man = json.loads((n.out / "manifest.json").read_text())
    assert man["status"] == "done" and man["gate"]["passed"]
    # the same rows in the same order, the same learning rates and weights as one uninterrupted run
    ref_out = n.box.d / "phase5" / "runs" / "ref-max-derived"
    argv = list(calls[0]["argv"])
    i = argv.index("--stop-at")
    del argv[i:i + 2]
    argv[argv.index("--out") + 1] = str(ref_out)
    argv[argv.index("--log-dir") + 1] = str(ref_out / "logs")
    script = n.box.d / "code-train" / "head" / "scripts" / "distill" / "train_qlora.py"
    cp = subprocess.run([sys.executable, str(DRIVER), str(script), *argv], env=n.box.env, capture_output=True,
                        text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr[-2000:]
    ref, got = n.steps("ref-max-derived"), n.steps()
    assert len(got) == len(ref) == man["step"] and len({x["pid"] for x in got}) == 3
    assert [x["ids"] for x in got] == [x["ids"] for x in ref]
    assert [x["lr"] for x in got] == [x["lr"] for x in ref] and got[-1]["w"] == ref[-1]["w"]
    # retention at the end: the newest two plus both epoch ends
    ends = [c for c in n.ckpts() if json.loads((n.out / c / "state.json").read_text()).get("epoch_end")]
    assert len(ends) == 2 and len(n.ckpts()) <= 4
    # one Telegram per training night, <= 5 lines, with the steps, the curve and the ETA
    tg = n.telegrams()
    assert len(tg) == 3 and all(len(m) <= 5 for m in tg)
    assert tg[0][0].startswith("9B round-2 training night 1: steps 0->") and "of 16" in tg[0][0]
    assert any(ln.startswith("Val 0:") for ln in tg[1]) and any("ETA" in ln for ln in tg[1])
    assert "DONE" in tg[2][0] and any("Gate passed" in ln for ln in tg[2])

    # night 4: the window is held for the post-train pipeline (simulated: it takes, releases, completes)
    def pipeline() -> None:
        _wait(lambda: (json.loads((n.job / "hold.json").read_text()).get("state") == "holding"
                       if (n.job / "hold.json").exists() else False), 60, "the hold")
        hold = json.loads((n.job / "hold.json").read_text())
        assert TN.hold_problem(hold, hold["night"], time.time(), min_left_s=30) is None
        assert PT.hold_problem(hold, hold["night"], time.time(), min_left_s=30) is None
        night = hold["night"]
        TN.write_json(n.job / TN.taken_name(night), {"night": night})
        time.sleep(0.5)
        TN.write_json(n.job / TN.DONE_FILE, {"outcome": "completed"})
        TN.write_json(n.job / TN.released_name(night), {"night": night, "completed": True})

    th = threading.Thread(target=pipeline)
    th.start()
    r = n.window_start(stop_in=120)
    th.join(timeout=90)
    n.window_end(r, sigterm=False)
    assert r.returncode == 0
    job = n.state()
    assert job["phase"] == "done" and job["post_train"]["holds"][-1]["taken"] is True
    assert json.loads((n.job / "hold.json").read_text())["state"] == "ended"
    assert n.box.step("train-nights")["status"] == "done"
    assert "disarm" in "\n".join(n.telegrams()[-1])
    # night 5: nothing left, so the runner's dry run fails and window_start would skip the swap
    dry = n.box.run(execute=False, stop_in=120)
    assert dry.returncode == 1 and "train-nights job has work left" in dry.stdout and "disarm" in dry.stdout


def test_a_hold_nobody_takes_ends_early_and_says_so(nights: Nights) -> None:
    n = nights
    st = TN.new_state(n.RUN, str(n.out), time.time())
    st["phase"] = "post-train"
    n.job.mkdir(parents=True)
    TN.save(n.job, st)
    t0 = time.time()
    r = n.window_start(stop_in=120)
    n.window_end(r, sigterm=False)
    assert r.returncode == 0 and time.time() - t0 < 60  # 0.3 min hold grace, not the whole window
    job = n.state()
    assert job["phase"] == "post-train" and job["post_train"]["holds"][-1]["taken"] is False
    assert "no pipeline took the 4090" in n.box.step("train-nights")["detail"]
    assert "without the pipeline" in "\n".join(n.telegrams()[-1])


def test_a_night_without_disk_room_does_not_start_and_telegrams(nights: Nights) -> None:
    n = nights
    n.box.env["DJR_NIGHTS_CKPT_GIB"] = "1000000"
    r = n.box.start(stop_in=60)
    r.wait(timeout=60)
    step = n.box.step("train-nights")
    assert step["status"] == "not-started" and "GiB" in step["detail"]
    assert not (n.box.state / "trainer-calls.log").exists()
    assert any("4090" in "\n".join(m) for m in n.telegrams())
