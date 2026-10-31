#!/usr/bin/env python3
"""The first full 9B QLoRA run on the 4090 in its continuous window: launcher and watcher.

    train_full.py launch --dataset DIR [--stop-at ISO] [settings]            # dry run: checks + plan
    train_full.py launch --dataset DIR --stop-at 2026-11-02T05:20:00-05:00 --execute
    train_full.py watch --unit distill-train-full-<stamp>.service --out RUN_DIR [--execute]
    train_full.py status --out RUN_DIR                                      # progress, epochs, val curve

``launch`` (stdlib only; /usr/bin/python3 on the tower) checks, without changing anything:
the dataset directory (``train.jsonl`` and ``val.jsonl`` against the sha256 in its
``dataset-manifest.json``; a Max-derived dataset needs ``max-derived`` in the run name, Q6b),
the staged trainer code (``<code>/scripts/distill/train_qlora.py`` + ``COMMIT``), the base model and
tokenizer, the training python, no NO-SWAPS flag, no ``distill-*`` unit running but the dead-man (a
teacher or judge server must have ended), the continuous window's end and dead-man timers armed,
``--stop-at`` at least 30 min away and at least 10 min before the window end, free disk at least
``--min-free-gib`` + 3 GiB, and the GPU free (or only the business server on it). It prints the
trainer command, the projected end (train tokens x epochs at ``--assume-tok-s``) and the units.

With ``--execute`` it then: Telegrams "<business> going down" with the projected ETA (or "9B training
starting" when business is down already), stops ``llama-server.service`` through its unit and waits for
the VRAM; starts the trainer as ``distill-train-full-<stamp>.service`` (KillMode=mixed, SIGTERM,
``TimeoutStopSec`` long enough to finish a step, a held-out evaluation and a checkpoint; ``RuntimeMaxSec``
past ``--stop-at`` as a backstop), records it in ``state/train-full.unit``, and starts the watcher as
``flywheel-train-watch-<stamp>.service`` (not ``distill-*``: a restore must not stop it).

The trainer (``train_qlora.py``, no ``--smoke``): ``--epochs 2 --max-seq 16384 --grad-accum 8`` (micro-batch
1; the runbook's Phase 5), checkpoints every ``--save-every-minutes 60`` (and every ``--save-every`` steps,
at each epoch end, on SIGTERM and at ``--stop-at``), keeps ``--keep 2``, stops and checkpoints under
``--min-free-gib 6``, held-out loss (val subset + fixed probe batch) at step 0 and every checkpoint,
``--resume`` when the run directory already holds a run. A stop (SIGTERM, stop-at, disk floor) is
resumable: launch again with the same settings.

``watch`` (every ``--poll`` s) reads the run's ``run_state.json`` and ``manifest.json`` and Telegrams
(<= 5 lines each, once each; ``watch-state.json`` in the run directory remembers what was sent):
the start, once 10 steps give a measured token rate (steps, tok/s, ETA); each epoch end (val and probe
loss so far, ETA); the end (status, the val loss curve, the probe before/after, the adapter). Then, with
``--execute`` and unless ``--no-restore``, it execs the window's ``train_end.sh``: one ``apply business``
+ ``tower_verify`` under the window lock, "<business> back" Telegram (or a loud failure: NO-SWAPS flag,
the dead-man retries). It exits quietly a minute before the continuous window's end (``continuous_end.sh``
stops the trainer, which checkpoints, and restores business itself).

Exit: 0 ok (dry run: every required check passed), 1 a check or action failed, 2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo

    ET: Any = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - no tz database: print UTC
    ET = timezone.utc

HOME = Path.home()
D = HOME / "distill"
DEFAULTS = {
    "code": D / "code-train" / "head",
    "base": D / "models" / "qwen3.5-9b-bnb4",
    "tokenizer": D / "models" / "qwen3.5-9b-tokenizer",
    "python": D / "venv-train" / "bin" / "python",
    "win_dir": D / "window",
    "notify": D / "notify.sh",
}
BUSINESS_UNIT = "llama-server.service"
CONT_END_TIMER = "flywheel-window-cont-end.timer"
DEADMAN_TIMER = "distill-deadman-tower-window.timer"
UNIT_PREFIX = "distill-train-full-"
WATCH_PREFIX = "flywheel-train-watch-"
IDLE_VRAM_MAX_MIB = 1500
STOP_TIMEOUT_S = 900   # a micro-step (<= ~15 s at 16k) + a held-out evaluation (~3 min) + a checkpoint
WATCH_STOP_TIMEOUT_S = 1800  # the restore it may be running (apply business + verify)
ASSUMED_TOK_S = 2540.0  # the 9B smoke at 16,384 on the 4090 (2026-10-31)
START_AFTER_STEPS = 10
STATE_FILE = "watch-state.json"
UNIT_RECORD = "train-full.unit"
NOTIFY_LINES = 5

Runner = Callable[[Sequence[str]], Tuple[int, str]]


def run_cmd(cmd: Sequence[str], timeout: float = 60) -> Tuple[int, str]:
    try:
        p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return p.returncode, (p.stdout + p.stderr).strip()


def et(ts: float) -> str:
    return datetime.fromtimestamp(ts, ET).strftime("%a %H:%M %Z")


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_iso(value: str) -> float:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"{value!r} needs a UTC offset (e.g. 2026-11-02T05:20:00-05:00)")
    return dt.timestamp()


def f4(x: Any) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "n/a"


# ---------------------------------------------------------------------------
# Messages (pure: tested without a host)
# ---------------------------------------------------------------------------


def val_curve(history: List[Dict[str, Any]], key: str = "val_loss", limit: int = 14) -> str:
    """'0:1.644 75:1.52 ...' over the evaluations, thinned evenly to *limit*
    points (the first and the last always kept)."""
    pts = [(h["step"], h[key]) for h in sorted(history, key=lambda h: h.get("step", 0))
           if isinstance(h.get(key), (int, float))]
    if len(pts) > limit:
        idx = sorted({round(i * (len(pts) - 1) / (limit - 1)) for i in range(limit)})
        pts = [pts[i] for i in idx]
    return " ".join(f"{s}:{v:.3f}" for s, v in pts) or "none"


def start_message(prog: Dict[str, Any], held: Dict[str, Any] | None, now: float) -> str:
    eta = now + float(prog.get("eta_s") or 0)
    h0 = held or {}
    return "\n".join([
        f"9B full training running: step {prog.get('step')}/{prog.get('total_steps')}, "
        f"epoch {prog.get('epoch')}/{prog.get('epochs')}, {prog.get('tok_s', 0):,.0f} tok/s.",
        f"ETA {et(eta)} ({(prog.get('tokens_left') or 0) / 1e6:.1f}M tokens left, max_seq "
        f"{prog.get('max_seq')}).",
        f"Held out at step 0: val {f4(h0.get('val_loss'))}, probe {f4(h0.get('probe_loss'))}.",
    ])


def epoch_message(ep: Dict[str, Any], history: List[Dict[str, Any]], prog: Dict[str, Any] | None,
                  now: float) -> str:
    base = next((h for h in history if h.get("step") == 0), {})
    lines = [f"9B training: epoch {ep.get('epoch')}/{ep.get('of')} done at step {ep.get('step')} "
             f"({et(parse_iso(ep['at'])) if ep.get('at') else '?'}).",
             f"Val loss {f4(base.get('val_loss'))} -> {f4(ep.get('val_loss'))}; probe "
             f"{f4(base.get('probe_loss'))} -> {f4(ep.get('probe_loss'))}.",
             f"Val curve (step:loss): {val_curve(history)}"]
    if prog and prog.get("eta_s") is not None and (prog.get("tokens_left") or 0) > 0:
        lines.append(f"ETA {et(now + float(prog['eta_s']))}.")
    return "\n".join(lines)


def end_message(man: Dict[str, Any] | None, rs: Dict[str, Any] | None, unit_state: str) -> str:
    man = man or {}
    rs = rs or {}
    hist = rs.get("val_history") or man.get("val_history") or []
    status = str(man.get("status") or "none")
    base = next((h for h in hist if h.get("step") == 0), {})
    last = max(hist, key=lambda h: h.get("step", 0)) if hist else {}
    prog = rs.get("progress") or {}
    head = {"done": "9B full training DONE", "stopped": "9B full training STOPPED (resumable)",
            "failed": "9B full training FAILED"}.get(status, f"9B full training ended ({status})")
    why = man.get("reason")
    lines = [f"{head}: step {man.get('step', prog.get('step'))}/{prog.get('total_steps', '?')}, "
             f"{(man.get('tokens_seen') or 0) / 1e6:.1f}M tokens"
             + (f"; {str(why)[:80]}" if why else "") + (f"; unit {unit_state}" if status == "none" else "")
             + ".",
             f"Val loss curve (step:loss): {val_curve(hist)}",
             f"Probe {f4(base.get('probe_loss'))} -> {f4(last.get('probe_loss'))} (step "
             f"{last.get('step', '?')}); val {f4(base.get('val_loss'))} -> {f4(last.get('val_loss'))}."]
    if man.get("adapter"):
        lines.append(f"Adapter: {Path(str(man['adapter'])).parent.name}/{Path(str(man['adapter'])).name} "
                     "(private, max-derived).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Host probes
# ---------------------------------------------------------------------------


class Host:
    """The tower as the launcher and watcher see it; every call goes through *run*."""

    def __init__(self, run: Runner = run_cmd, win_dir: Path = DEFAULTS["win_dir"],
                 notify_path: Path = DEFAULTS["notify"], clock: Callable[[], float] = time.time) -> None:
        self.run, self.win_dir, self.notify_path, self.clock = run, Path(win_dir), Path(notify_path), clock

    def unit_state(self, unit: str) -> str:
        rc, out = self.run(["systemctl", "--user", "is-active", unit])
        return (out.splitlines() or ["unknown"])[-1].strip() or "unknown"

    def active_distill_units(self) -> List[str]:
        rc, out = self.run(["systemctl", "--user", "list-units", "--plain", "--no-legend", "--no-pager",
                            "--state=active,activating,deactivating,reloading", "distill-*"])
        units = [ln.split()[0] for ln in out.splitlines() if ln.strip()]
        return [u for u in units if not u.startswith("distill-deadman-")]

    def timer_next(self, timer: str) -> Optional[float]:
        rc, out = self.run(["systemctl", "--user", "show", timer, "-p", "ActiveState", "-p",
                            "NextElapseUSecRealtime"])
        kv = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
        if kv.get("ActiveState") != "active" or not kv.get("NextElapseUSecRealtime"):
            return None
        rc, out = self.run(["date", "-d", kv["NextElapseUSecRealtime"], "+%s"])
        try:
            return float(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None

    def vram_used_mib(self) -> Optional[int]:
        rc, out = self.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        try:
            return int(out.strip().splitlines()[0].strip())
        except (ValueError, IndexError):
            return None

    def free_gib(self, path: Path) -> float:
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free / 2**30

    def notify(self, text: str, *, execute: bool) -> bool:
        msg = "\n".join(text.splitlines()[:NOTIFY_LINES])
        if not execute:
            print("  [dry-run] would notify:\n" + "\n".join("    | " + ln for ln in msg.splitlines()))
            return True
        try:
            p = subprocess.run([str(self.notify_path), str(NOTIFY_LINES)], input=msg + "\n",
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return p.returncode == 0


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


class Plan:
    def __init__(self) -> None:
        self.checks: List[Tuple[bool, str, str]] = []

    def chk(self, ok: bool, what: str, detail: str = "") -> bool:
        self.checks.append((ok, what, detail))
        print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""))
        return ok

    @property
    def ok(self) -> bool:
        return all(c[0] for c in self.checks)


def dataset_check(plan: Plan, ds: Path) -> Dict[str, Any]:
    man = read_json(ds / "dataset-manifest.json")
    info: Dict[str, Any] = {"max_derived": None, "rows": None, "tokens": None}
    if not plan.chk(isinstance(man, dict), "dataset manifest", str(ds / "dataset-manifest.json")):
        return info
    info["manifest_sha256"] = sha256_file(ds / "dataset-manifest.json")
    info["max_derived"] = bool(man.get("max_derived"))
    files = man.get("files") or {}
    for name in ("train.jsonl", "val.jsonl"):
        want = (files.get(name) or {}).get("sha256")
        p = ds / name
        have = sha256_file(p) if p.is_file() else None
        plan.chk(bool(want) and have == want, f"{name} sha256 matches the manifest",
                 f"{(have or 'missing')[:16]}... rows {(files.get(name) or {}).get('rows')}")
    plan.chk((ds / "report.json").is_file(), "build report beside train.jsonl (the dataset's own approval)")
    tokens = rows = 0
    try:
        with open(ds / "train.jsonl", "rb") as fh:
            for line in fh:
                m = re.search(rb'"n_tokens":\s*(\d+)', line)
                rows += 1
                tokens += int(m.group(1)) if m else 0
    except OSError:
        pass
    info.update(rows=rows, tokens=tokens)
    return info


def trainer_cmd(a: argparse.Namespace, out: Path, resume: bool) -> List[str]:
    ds = Path(a.dataset)
    cmd = [str(a.python), str(Path(a.code) / "scripts" / "distill" / "train_qlora.py"),
           "--data", str(ds / "train.jsonl"), "--val", str(ds / "val.jsonl"),
           "--tokenizer-dir", str(a.tokenizer), "--base-model", str(a.base),
           "--base-revision", a.base_rev, "--out", str(out), "--log-dir", str(out / "logs"),
           "--epochs", str(a.epochs), "--max-seq", str(a.max_seq), "--grad-accum", str(a.grad_accum),
           "--save-every", str(a.save_every), "--save-every-minutes", f"{a.save_every_minutes:g}",
           "--keep-checkpoints", str(a.keep), "--min-free-gib", f"{a.min_free_gib:g}",
           "--stop-at", a.stop_at]
    if resume:
        cmd.append("--resume")
    if a.max_derived_ok:
        cmd.append("--i-have-operator-approval-for-max-derived-data")
    return cmd


def base_revision(a: argparse.Namespace) -> str:
    if a.base_rev:
        return str(a.base_rev)
    man = read_json(Path(a.base) / "MANIFEST.json")
    rev = man.get("revision") if isinstance(man, dict) else None
    return rev if isinstance(rev, str) and re.fullmatch(r"[0-9a-f]{40}", rev) else ""


def launch(a: argparse.Namespace, host: Host) -> int:
    execute = bool(a.execute)
    now = host.clock()
    stamp = a.stamp or datetime.fromtimestamp(now, ET).strftime("%Y%m%dT%H%M%S")
    unit, watch = f"{UNIT_PREFIX}{stamp}", f"{WATCH_PREFIX}{stamp}"
    ds = Path(a.dataset).expanduser()
    a.dataset = ds
    name = ds.name[len("build-"):] if ds.name.startswith("build-") else ds.name
    out = Path(a.out).expanduser() if a.out else D / "phase5" / "runs" / f"9b-r32-{name}"
    plan = Plan()
    print(f"train_full launch ({'EXECUTE' if execute else 'dry run'}) {et(now)}")
    info = dataset_check(plan, ds)
    if info["max_derived"]:
        plan.chk("max-derived" in out.name, "run name carries max-derived (Q6b: private, tagged)", out.name)
        plan.chk(bool(a.max_derived_ok), "operator flag for Max-derived data (Q6b) passed",
                 "--max-derived-ok" if a.max_derived_ok else "missing --max-derived-ok")
    resume = (out / "run_state.json").is_file()
    plan.chk(not out.exists() or resume, "run directory", f"{out} ({'resume' if resume else 'new run'})")
    code = Path(a.code).expanduser()
    commit = (code / "COMMIT").read_text().strip() if (code / "COMMIT").is_file() else ""
    missing = [f for f in ("train_qlora.py", "render.py", "artifact_policy.py")
               if not (code / "scripts" / "distill" / f).is_file()]
    plan.chk(not missing and bool(commit), "trainer code", f"{code} @{commit[:12]}"
             + (f"; missing {missing}" if missing else ""))
    plan.chk(Path(a.python).is_file(), "training python", str(a.python))
    a.base_rev = base_revision(a)
    base = Path(a.base)
    plan.chk((base / "config.json").is_file() and any(base.glob("*.safetensors")) and bool(a.base_rev),
             "base model", f"{base} @{a.base_rev[:12] or 'no revision'}")
    plan.chk(Path(a.tokenizer).is_dir(), "tokenizer", str(a.tokenizer))
    flags = [p for p in (host.win_dir / "NO-SWAPS-TONIGHT", D / "NO-SWAPS") if p.exists()]
    plan.chk(not flags, "no NO-SWAPS flag", ", ".join(map(str, flags)) or "none")
    others = host.active_distill_units()
    plan.chk(not others, "no distill unit running (the GPU is the trainer's)", ", ".join(others) or "none")
    cont_end, deadman = host.timer_next(CONT_END_TIMER), host.timer_next(DEADMAN_TIMER)
    plan.chk(cont_end is not None, f"{CONT_END_TIMER} armed", et(cont_end) if cont_end else "not armed")
    plan.chk(deadman is not None, f"{DEADMAN_TIMER} armed", et(deadman) if deadman else "not armed")
    try:
        stop_at = parse_iso(a.stop_at)
    except ValueError as exc:
        plan.chk(False, "--stop-at", str(exc))
        stop_at = now
    latest = (cont_end - 600) if cont_end else now
    plan.chk(stop_at >= now + 1800 and stop_at <= latest, "--stop-at in the window",
             f"{et(stop_at)} (>= 30 min from now; by {et(latest)}, 10 min before the window end)")
    free = host.free_gib(out)
    plan.chk(free >= a.min_free_gib + 3, "free disk", f"{free:.1f} GiB (needs {a.min_free_gib:g} + 3 GiB)")
    biz = host.unit_state(BUSINESS_UNIT)
    used = host.vram_used_mib()
    if biz in ("active", "activating"):
        print(f"  [plan] {BUSINESS_UNIT} is {biz}: Telegram down, stop it through its unit, wait for the VRAM")
    else:
        plan.chk(used is not None and used <= IDLE_VRAM_MAX_MIB, "GPU free", f"{used} MiB used, {BUSINESS_UNIT} {biz}")
    total = (info.get("tokens") or 0) * a.epochs
    secs = total / a.assume_tok_s if a.assume_tok_s > 0 else 0
    proj = now + 240 + secs * 1.05  # load + held-out evaluations and checkpoints (~5%)
    print(f"  [plan] {info.get('rows')} rows, {total / 1e6:.1f}M tokens over {a.epochs} epochs: ~"
          f"{secs / 3600:.1f} h at {a.assume_tok_s:,.0f} tok/s -> ~{et(proj)}"
          + ("" if proj <= stop_at else f"; AFTER --stop-at {et(stop_at)}: it stops there and resumes later"))
    cmd = trainer_cmd(a, out, resume)
    runtime_max = int(max(stop_at - now, 0) + 1800)
    trainer_unit = ["systemd-run", "--user", f"--unit={unit}",
                    "--description=distill full 9B QLoRA (Phase 5)",
                    "-p", "KillMode=mixed", "-p", "KillSignal=SIGTERM", "-p", f"TimeoutStopSec={a.stop_timeout}",
                    "-p", f"RuntimeMaxSec={runtime_max}", "-p", f"WorkingDirectory={D}",
                    "-p", f"StandardOutput=append:{out / 'logs' / 'unit.log'}",
                    "-p", f"StandardError=append:{out / 'logs' / 'unit.log'}",
                    "--setenv=PYTHONUNBUFFERED=1", "--setenv=HF_HUB_OFFLINE=1", "--setenv=TRANSFORMERS_OFFLINE=1",
                    "--setenv=HF_HUB_DISABLE_TELEMETRY=1", "--setenv=DO_NOT_TRACK=1",
                    "--setenv=TZ=America/New_York", "--expand-environment=no", *cmd]
    watcher = [sys.executable if a.watch_python is None else a.watch_python, str(Path(__file__).resolve()),
               "watch", "--unit", f"{unit}.service", "--out", str(out), "--win-dir", str(host.win_dir),
               "--execute"]
    watcher_unit = ["systemd-run", "--user", f"--unit={watch}", "--collect",
                    "--description=flywheel: report the full 9B run and restore business when it ends",
                    "-p", "KillMode=mixed", "-p", f"TimeoutStopSec={WATCH_STOP_TIMEOUT_S}",
                    "--expand-environment=no", *watcher]
    print("  [plan] trainer: " + " ".join(trainer_unit))
    print("  [plan] watcher: " + " ".join(watcher_unit))
    if not plan.ok:
        print("REFUSED: " + "; ".join(w for ok, w, _ in plan.checks if not ok))
        return 1
    if not execute:
        print("dry run: every check passed; nothing was started (--execute launches)")
        return 0
    down = biz in ("active", "activating")
    eta_line = f"Projected end ~{et(proj)} (stop-at {et(stop_at)}), {total / 1e6:.0f}M tokens, {a.epochs} epochs."
    if down:
        host.notify(f"{a.biz_name} going down: full 9B QLoRA training on {ds.name} ({info.get('rows')} rows).\n"
                    f"{eta_line}\nTelegrams at each epoch and at the end; {a.biz_name} is restored when it ends.",
                    execute=True)
        rc, msg = host.run(["systemctl", "--user", "stop", BUSINESS_UNIT])
        if rc != 0:
            print(f"stop {BUSINESS_UNIT} exited {rc}: {msg}")
    else:
        host.notify(f"9B full training starting on {ds.name} ({info.get('rows')} rows); {a.biz_name} is already down.\n"
                    f"{eta_line}", execute=True)
    for _ in range(36):
        used = host.vram_used_mib()
        if host.unit_state(BUSINESS_UNIT) not in ("active", "activating") and used is not None \
                and used <= IDLE_VRAM_MAX_MIB:
            break
        time.sleep(5)
    else:
        print(f"the GPU did not free ({used} MiB): restoring business")
        os.execv("/bin/bash", ["/bin/bash", str(host.win_dir / "train_end.sh")])
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rc, msg = host.run(trainer_unit)
    print(msg)
    if rc != 0:
        print("the trainer unit did not start: restoring business")
        os.execv("/bin/bash", ["/bin/bash", str(host.win_dir / "train_end.sh")])
    rec = host.win_dir / "state" / UNIT_RECORD
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(f"{unit}.service {out} {watch}.service\n", encoding="utf-8")
    rc, msg = host.run(watcher_unit)
    print(msg)
    if rc != 0:
        host.notify(f"9B training {unit} started, but its watcher did not (no epoch Telegrams; "
                    f"continuous_end restores {a.biz_name} at the window end).", execute=True)
        return 1
    print(f"started: {unit}.service (out {out}); watcher {watch}.service")
    return 0


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


def watch_once(out: Path, unit: str, host: Host, state: Dict[str, Any], *, execute: bool
               ) -> Optional[str]:
    """One pass: sends what is due, records it in *state*. Returns "ended"
    when the training unit has ended (end message sent), else None."""
    rs = read_json(out / "run_state.json") or {}
    hist = rs.get("val_history") or []
    prog = rs.get("progress") or {}
    now = host.clock()
    if not state.get("start") and (prog.get("step") or 0) >= START_AFTER_STEPS and prog.get("tok_s"):
        base = next((h for h in hist if h.get("step") == 0), None)
        if host.notify(start_message(prog, base, now), execute=execute):
            state["start"] = now
    for ep in rs.get("epochs_done") or []:
        k = str(ep.get("epoch"))
        if k not in (state.get("epochs") or {}):
            if host.notify(epoch_message(ep, hist, prog, now), execute=execute):
                state.setdefault("epochs", {})[k] = now
    st = host.unit_state(unit)
    if st in ("active", "activating", "deactivating", "reloading"):
        return None
    man = read_json(out / "manifest.json")
    if not state.get("end"):
        if host.notify(end_message(man, rs, st), execute=execute):
            state["end"] = now
    return "ended"


def watch(a: argparse.Namespace, host: Host) -> int:
    out = Path(a.out).expanduser()
    if not re.fullmatch(rf"{UNIT_PREFIX}[0-9T]+\.service", a.unit):
        print(f"refusing unit {a.unit!r}", file=sys.stderr)
        return 2
    sp = out / STATE_FILE
    state = read_json(sp) or {}
    cutoff = host.timer_next(CONT_END_TIMER)
    while True:
        before = json.dumps(state, sort_keys=True)
        res = watch_once(out, a.unit, host, state, execute=a.execute)
        if a.execute and json.dumps(state, sort_keys=True) != before:
            write_json(sp, state)
        if res == "ended":
            break
        if not a.execute:
            print("dry run: one pass done")
            return 0
        if cutoff is not None and host.clock() >= cutoff - 60:
            print("the continuous window ends within a minute: continuous_end takes over")
            return 0
        time.sleep(a.poll)
    if not a.execute or a.no_restore:
        print("ended; no restore (" + ("dry run" if not a.execute else "--no-restore") + ")")
        return 0
    man = read_json(out / "manifest.json") or {}
    msg = out / "logs" / "train-end.msg"
    msg.parent.mkdir(parents=True, exist_ok=True)
    msg.write_text(f"9B training {man.get('status', 'ended')} at step {man.get('step', '?')}; "
                   f"val {val_curve((read_json(out / 'run_state.json') or {}).get('val_history') or [], limit=4)}\n",
                   encoding="utf-8")
    sys.stdout.flush()
    os.execv("/bin/bash", ["/bin/bash", str(host.win_dir / "train_end.sh"), "--msg-file", str(msg)])
    return 1  # not reached


def status(a: argparse.Namespace) -> int:
    out = Path(a.out).expanduser()
    rs = read_json(out / "run_state.json") or {}
    man = read_json(out / "manifest.json") or {}
    print(json.dumps({"status": man.get("status"), "progress": rs.get("progress"),
                      "epochs_done": rs.get("epochs_done"),
                      "val_curve": val_curve(rs.get("val_history") or []),
                      "probe_curve": val_curve(rs.get("val_history") or [], "probe_loss"),
                      "held_out": {k: (v if k != "probe" else {kk: vv for kk, vv in v.items() if kk != "ids"})
                                   for k, v in (rs.get("held_out") or {}).items()}},
                     indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train_full", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("launch")
    la.add_argument("--dataset", required=True, help="a build.py output dir (train/val.jsonl + manifest)")
    la.add_argument("--out", help="run directory (default ~/distill/phase5/runs/9b-r32-<dataset dir name "
                                  "without build->, e.g. 9b-r32-v4-max-derived)")
    la.add_argument("--code", default=str(DEFAULTS["code"]), help="a HEAD export of the trainer code")
    la.add_argument("--base", default=str(DEFAULTS["base"]))
    la.add_argument("--base-rev", default="", help="default: 'revision' in <base>/MANIFEST.json")
    la.add_argument("--tokenizer", default=str(DEFAULTS["tokenizer"]))
    la.add_argument("--python", default=str(DEFAULTS["python"]), help="the training venv's python")
    la.add_argument("--watch-python", default=None, help="python for the watcher (default: this one)")
    la.add_argument("--stop-at", required=True, help="ISO with offset: the trainer checkpoints and exits before it")
    la.add_argument("--epochs", type=int, default=2)
    la.add_argument("--max-seq", type=int, default=16384)
    la.add_argument("--grad-accum", type=int, default=8)
    la.add_argument("--save-every", type=int, default=200)
    la.add_argument("--save-every-minutes", type=float, default=60.0)
    la.add_argument("--keep", type=int, default=2)
    la.add_argument("--min-free-gib", type=float, default=6.0)
    la.add_argument("--stop-timeout", type=int, default=STOP_TIMEOUT_S)
    la.add_argument("--assume-tok-s", type=float, default=ASSUMED_TOK_S)
    la.add_argument("--max-derived-ok", action="store_true",
                    help="pass the trainer's operator flag for Max-derived data (Q6b)")
    la.add_argument("--biz-name", default="business", help="how Telegram names the business service")
    la.add_argument("--win-dir", default=str(DEFAULTS["win_dir"]))
    la.add_argument("--stamp", default="", help="unit name stamp (default: now, ET)")
    la.add_argument("--execute", action="store_true")
    wa = sub.add_parser("watch")
    wa.add_argument("--unit", required=True)
    wa.add_argument("--out", required=True)
    wa.add_argument("--poll", type=float, default=60.0)
    wa.add_argument("--win-dir", default=str(DEFAULTS["win_dir"]))
    wa.add_argument("--no-restore", action="store_true")
    wa.add_argument("--execute", action="store_true")
    sa = sub.add_parser("status")
    sa.add_argument("--out", required=True)
    return ap


def main(argv: Optional[List[str]] = None, *, host: Optional[Host] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "status":
        return status(a)
    h = host or Host(win_dir=Path(a.win_dir).expanduser())
    if a.cmd == "launch":
        if a.biz_name == "business":
            conf = Path(a.win_dir).expanduser() / "window.conf"
            m = re.search(r"^BIZ_NAME=(\S+)", conf.read_text(), re.M) if conf.is_file() else None
            a.biz_name = m.group(1) if m else a.biz_name
        return launch(a, h)
    return watch(a, h)


if __name__ == "__main__":
    sys.exit(main())
