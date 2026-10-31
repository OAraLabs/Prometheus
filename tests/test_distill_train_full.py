"""Tests for scripts/distill/tower/train_full.py: the full 9B run's launcher and watcher.

No systemd, no GPU, no Telegram: a fake host answers the probes and records
what would be sent; the launch is only ever a dry run here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_F = Path(__file__).resolve().parent.parent / "scripts" / "distill" / "tower" / "train_full.py"


def _load() -> Any:
    name = "distill_train_full"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _F)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TF = _load()
NOW = 1_793_467_800.0  # Sat 2026-10-31 13:30 EDT
WINDOW_END = 1_793_616_000.0  # Mon 2026-11-02 05:40 EST: the continuous window's end


class FakeHost(TF.Host):
    def __init__(self, tmp: Path, **kw: Any) -> None:
        super().__init__(run=self._run, win_dir=tmp / "window", notify_path=tmp / "notify.sh",
                         clock=lambda: self.now)
        self.now = NOW
        self.units: dict[str, str] = kw.get("units", {})
        self.others: list[str] = kw.get("others", [])
        self.timers = kw.get("timers", {TF.CONT_END_TIMER: WINDOW_END, TF.DEADMAN_TIMER: WINDOW_END + 1200})
        self.vram = kw.get("vram", 9)
        self.free = kw.get("free", 17.0)
        self.sent: list[str] = []
        self.ran: list[list[str]] = []

    def _run(self, cmd: Any) -> tuple[int, str]:
        self.ran.append(list(cmd))
        return 0, ""

    def unit_state(self, unit: str) -> str:
        return self.units.get(unit, "inactive")

    def active_distill_units(self) -> list[str]:
        return list(self.others)

    def timer_next(self, timer: str) -> float | None:
        return self.timers.get(timer)

    def vram_used_mib(self) -> int | None:
        return self.vram

    def free_gib(self, path: Path) -> float:
        return self.free

    def notify(self, text: str, *, execute: bool) -> bool:
        assert len(text.splitlines()) <= TF.NOTIFY_LINES
        self.sent.append(text)
        return True


def dataset(tmp: Path, *, max_derived: bool = True, corrupt: bool = False) -> Path:
    ds = tmp / "build-v4-max-derived"
    ds.mkdir()
    train = "".join(json.dumps({"id": f"r{i}", "n_tokens": 10_000}) + "\n" for i in range(100))
    val = "".join(json.dumps({"id": f"v{i}", "n_tokens": 9_000}) + "\n" for i in range(10))
    (ds / "train.jsonl").write_text(train)
    (ds / "val.jsonl").write_text(val)
    (ds / "report.json").write_text("{}")
    files = {n: {"rows": t.count("\n"), "sha256": hashlib.sha256(t.encode()).hexdigest()}
             for n, t in (("train.jsonl", train), ("val.jsonl", val))}
    if corrupt:
        files["train.jsonl"]["sha256"] = "0" * 64
    (ds / "dataset-manifest.json").write_text(json.dumps({"max_derived": max_derived, "files": files}))
    return ds


def staged(tmp: Path) -> dict[str, Path]:
    code = tmp / "code"
    (code / "scripts" / "distill").mkdir(parents=True)
    for f in ("train_qlora.py", "render.py", "artifact_policy.py"):
        (code / "scripts" / "distill" / f).write_text("")
    (code / "COMMIT").write_text("a" * 40 + "\n")
    base = tmp / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_text("")
    (base / "MANIFEST.json").write_text(json.dumps({"revision": "3c78dfb3e32c496a2438fb1790859eff805284cf"}))
    tok = tmp / "tok"
    tok.mkdir()
    py = tmp / "python"
    py.write_text("")
    (tmp / "window").mkdir()
    return {"code": code, "base": base, "tok": tok, "py": py}


def launch_args(tmp: Path, ds: Path, *extra: str, stop_at: str = "2026-11-02T05:20:00-05:00") -> list[str]:
    s = staged(tmp) if not (tmp / "code").exists() else {"code": tmp / "code", "base": tmp / "base",
                                                            "tok": tmp / "tok", "py": tmp / "python"}
    return ["launch", "--dataset", str(ds), "--out", str(tmp / "runs" / "9b-r32-v4-max-derived"),
            "--code", str(s["code"]), "--base", str(s["base"]), "--tokenizer", str(s["tok"]),
            "--python", str(s["py"]), "--win-dir", str(tmp / "window"), "--stop-at", stop_at,
            "--stamp", "20261031T140000", "--biz-name", "biz", *extra]


def test_launch_dry_run_checks_everything_and_prints_the_units(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    host = FakeHost(tmp_path, units={TF.BUSINESS_UNIT: "active"}, vram=20262)
    ds = dataset(tmp_path)
    assert TF.main(launch_args(tmp_path, ds, "--max-derived-ok"), host=host) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "dry run: every check passed" in out
    trainer = next(ln for ln in out.splitlines() if "[plan] trainer:" in ln)
    for want in ("--unit=distill-train-full-20261031T140000", "KillMode=mixed", "KillSignal=SIGTERM",
                 "TimeoutStopSec=900", "--epochs 2", "--max-seq 16384", "--grad-accum 8",
                 "--save-every-minutes 60", "--keep-checkpoints 2", "--min-free-gib 6",
                 "--stop-at 2026-11-02T05:20:00-05:00", "--i-have-operator-approval-for-max-derived-data",
                 "--base-revision 3c78dfb3e32c496a2438fb1790859eff805284cf", "HF_HUB_OFFLINE=1",
                 f"--val {ds / 'val.jsonl'}"):
        assert want in trainer, want
    assert "--resume" not in trainer
    watcher = next(ln for ln in out.splitlines() if "[plan] watcher:" in ln)
    assert "--unit=flywheel-train-watch-20261031T140000" in watcher
    assert "watch --unit distill-train-full-20261031T140000.service" in watcher
    assert "1,000,000" not in out and "2.0M tokens over 2 epochs" in out
    assert "Telegram down, stop it through its unit" in out
    assert host.sent == [] and host.ran == []  # a dry run changes nothing and sends nothing


@pytest.mark.parametrize("case, kw, extra, why", [
    ("sha", {}, (), "train.jsonl sha256"),
    ("untagged", {}, (), "carries max-derived"),
    ("no-flag", {}, (), "operator flag"),
    ("busy", {"others": ["distill-teacher-serve.service"]}, (), "no distill unit running"),
    ("no-timer", {"timers": {}}, (), "armed"),
    ("late", {}, (), "--stop-at in the window"),
    ("disk", {"free": 8.0}, (), "free disk"),
    ("gpu", {"vram": 9000}, (), "GPU free"),
])
def test_launch_refuses(tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str,
                        kw: dict[str, Any], extra: tuple[str, ...], why: str) -> None:
    host = FakeHost(tmp_path, **kw)
    ds = dataset(tmp_path, corrupt=case == "sha")
    args = launch_args(tmp_path, ds, *(() if case == "no-flag" else ("--max-derived-ok",)),
                       stop_at="2026-11-04T05:20:00-05:00" if case == "late" else "2026-11-02T05:20:00-05:00")
    if case == "untagged":
        args[args.index("--out") + 1] = str(tmp_path / "runs" / "9b-r32-v4")
    assert TF.main(args, host=host) == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out and why in out.split("REFUSED")[1], out
    assert host.ran == [] and host.sent == []


def test_a_run_directory_with_state_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    host = FakeHost(tmp_path)
    ds = dataset(tmp_path)
    run = tmp_path / "runs" / "9b-r32-v4-max-derived"
    run.mkdir(parents=True)
    (run / "run_state.json").write_text("{}")
    assert TF.main(launch_args(tmp_path, ds, "--max-derived-ok"), host=host) == 0
    assert "--resume" in next(ln for ln in capsys.readouterr().out.splitlines() if "[plan] trainer:" in ln)


def run_dir(tmp: Path) -> Path:
    out = tmp / "run-max-derived"
    out.mkdir()
    return out


def put_state(out: Path, **rs: Any) -> None:
    (out / "run_state.json").write_text(json.dumps(rs))


HIST = [{"step": 0, "val_loss": 1.644, "probe_loss": 1.680}, {"step": 120, "val_loss": 1.40, "probe_loss": 1.45},
        {"step": 450, "val_loss": 1.25, "probe_loss": 1.30}]


def test_the_watcher_reports_start_each_epoch_and_the_end_once(tmp_path: Path) -> None:
    out = run_dir(tmp_path)
    unit = "distill-train-full-20261031T140000.service"
    host = FakeHost(tmp_path, units={unit: "active"})
    state: dict[str, Any] = {}
    put_state(out, val_history=HIST[:1], progress={"step": 3, "total_steps": 900})
    assert TF.watch_once(out, unit, host, state, execute=False) is None and host.sent == []
    prog = {"step": 10, "total_steps": 900, "epoch": 1, "epochs": 2, "tok_s": 2540.0, "eta_s": 36000,
            "tokens_left": 91_000_000, "max_seq": 16384}
    put_state(out, val_history=HIST[:1], progress=prog)
    TF.watch_once(out, unit, host, state, execute=False)
    assert len(host.sent) == 1 and "step 10/900" in host.sent[0] and "2,540 tok/s" in host.sent[0]
    assert "ETA" in host.sent[0] and "val 1.6440" in host.sent[0]
    ep1 = {"epoch": 1, "of": 2, "step": 450, "val_loss": 1.25, "probe_loss": 1.30,
           "at": "2026-10-31T23:00:00+00:00"}
    put_state(out, val_history=HIST, progress={**prog, "step": 460}, epochs_done=[ep1])
    TF.watch_once(out, unit, host, state, execute=False)
    TF.watch_once(out, unit, host, state, execute=False)  # nothing new: nothing sent
    assert len(host.sent) == 2 and "epoch 1/2 done at step 450" in host.sent[1]
    assert "1.6440 -> 1.2500" in host.sent[1] and "0:1.644 120:1.400 450:1.250" in host.sent[1]
    host.units[unit] = "inactive"
    (out / "manifest.json").write_text(json.dumps({
        "status": "done", "step": 900, "tokens_seen": 96_800_000,
        "adapter": "/x/runs/9b-r32-v4-max-derived/adapter-max-derived"}))
    hist = HIST + [{"step": 900, "val_loss": 1.20, "probe_loss": 1.24}]
    put_state(out, val_history=hist, progress={**prog, "step": 900}, epochs_done=[ep1, {**ep1, "epoch": 2,
                                                                                          "step": 900}])
    assert TF.watch_once(out, unit, host, state, execute=False) == "ended"
    assert TF.watch_once(out, unit, host, state, execute=False) == "ended"
    assert len(host.sent) == 4  # epoch 2, then the end, each once
    end = host.sent[-1]
    assert end.startswith("9B full training DONE: step 900/900, 96.8M tokens")
    assert "0:1.644 120:1.400 450:1.250 900:1.200" in end and "Probe 1.6800 -> 1.2400 (step 900)" in end
    assert "9b-r32-v4-max-derived/adapter-max-derived" in end


def test_the_end_message_for_a_stop_or_a_crash(tmp_path: Path) -> None:
    m = TF.end_message({"status": "stopped", "step": 300, "reason": "disk-floor", "tokens_seen": 3e7},
                       {"val_history": HIST}, "inactive")
    assert m.startswith("9B full training STOPPED (resumable): step 300") and "disk-floor" in m
    m = TF.end_message(None, {}, "failed")
    assert "ended (none)" in m and "unit failed" in m and "none" in m.splitlines()[1]


def test_the_val_curve_is_thinned_but_keeps_both_ends() -> None:
    hist = [{"step": s, "val_loss": 2.0 - s / 1000} for s in range(0, 1000, 25)]
    c = TF.val_curve(hist, limit=6)
    pts = c.split()
    assert len(pts) == 6 and pts[0] == "0:2.000" and pts[-1] == "975:1.025"
    assert TF.val_curve([]) == "none"


def test_watch_refuses_a_unit_it_did_not_start(tmp_path: Path) -> None:
    host = FakeHost(tmp_path)
    assert TF.main(["watch", "--unit", "llama-server.service", "--out", str(tmp_path)], host=host) == 2
    assert TF.main(["watch", "--unit", "distill-train-full-1.service", "--out", str(run_dir(tmp_path))],
                   host=host) == 0  # a dry run: one pass
