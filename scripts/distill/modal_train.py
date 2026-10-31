#!/usr/bin/env python3
"""Phase 5 rental packet: the large student's QLoRA run on Modal. Prepared, never launched here.

    # Cost table (no Modal, no network); --pretokenized uses the real token count.
    python3 scripts/distill/modal_train.py estimate [--pretokenized DIR] [--markdown]
    # Pre-tokenize on the host that holds the data and the winner's tokenizer dir.
    python3 scripts/distill/modal_train.py pretokenize --data ~/distill/phase3-27b/train.jsonl \\
        --tokenizer-dir ~/distill/models/qwen3.8-27b-tokenizer --out data/distill/pretok/train-27b
    # The plan for one GPU and cap, offline (no Modal client needed, spends nothing).
    python3 scripts/distill/modal_train.py plan --data data/distill/pretok/train-27b \\
        --gpu H100 --max-usd 150
    # Launch: the operator only (DISTILL-RENTAL-PACKET.md has the exact line).
    modal run --detach scripts/distill/modal_train.py::rent --data data/distill/pretok/train-27b \\
        --gpu H100 --max-usd 150 --max-hours 40 --launch [...]
    # Fetch the adapter of a run whose client went away.
    modal run scripts/distill/modal_train.py::fetch --run-id <id>

Flow (``rent``): refuse Max-derived data without the operator's approval flag
(DISTILL-OPEN-QUESTIONS #6) -> print the cost plan and refuse when even the
optimistic estimate is over ``--max-usd`` -> without ``--launch`` stop there ->
upload the pre-tokenized data to a Volume (skipped when the same content is
there) -> download the base weights into a Volume on a CPU-only function (no
GPU billed for downloads) -> one GPU call: load, run ``--calib-steps`` (50)
real training steps, measure tokens/sec, compute the exact remaining cost and
continue only if the whole run fits ``--max-usd`` and ``--max-hours``,
otherwise checkpoint and return -> train to the end, checkpointing to the
runs Volume every ``--save-every`` steps -> download the adapter. Every later
call (the next of a chain, or a relaunch with ``--run-id``) re-projects from
the measured speed against its own cap and deadline before training on.

Hard limits in code, inside the GPU container: a spend ledger on the Volume
(every segment's billed time, preempted ones included, at the list rate for
the GPU plus the container's CPU and memory) is checked before the load (with
the load and one step reserved), after it, and after every step; the run
checkpoints and stops before the next step could pass ``--max-usd``. It also
stops before ``--max-hours`` from launch. Modal's own per-call ``timeout`` (at
most 24 h) is the backstop kill. A run longer than one call continues in a
fresh call (a fresh container) from its checkpoint. One call trains a run at a
time: a call that finds another call's live segment in the ledger returns
"busy" before loading. The local side reads the ledger before every call and
does not start one that could not train. The GPU is never idle: no download,
upload or tokenization runs on it, a stop returns at once, ``scaledown_window``
is the minimum and nothing is kept warm. The image is pulled by digest.

Training is ``train_qlora.py`` itself (same loop, checkpoints, step-down and
resume as the local 9B run), on the pinned ``unsloth/unsloth`` image.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

try:  # the Modal client exists only where the operator launches from
    import modal
except ImportError:  # pragma: no cover - exercised by the tests' fake module
    modal = None

HERE = Path(__file__).resolve().parent


def _load_trainer() -> Any:
    name = "distill_train_qlora"
    if name in sys.modules:
        return sys.modules[name]
    for cand in (HERE / "train_qlora.py", Path("/root/distill/train_qlora.py")):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location(name, cand)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("train_qlora.py not found next to modal_train.py")


T = _load_trainer()

APP_NAME = "distill-27b-qlora"
# Official Unsloth image, training stack only (no Studio): PyTorch 2.11 + CUDA
# 12.8. Pulled by the digest the release tag pointed at when this packet was
# written (the multi-arch index; Modal takes its linux/amd64 image), so a
# re-pushed tag cannot change what the paid run trains in. The tag is for people.
UNSLOTH_IMAGE = "unsloth/unsloth:core-v0.1.910-beta"
UNSLOTH_IMAGE_DIGEST = "sha256:8703818b64f3541f0b93b947c31702b84d39e23746a64d1c8702e3fda56ed73a"
UNSLOTH_IMAGE_REF = f"unsloth/unsloth@{UNSLOTH_IMAGE_DIGEST}"
DEFAULT_BASE = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
PARAMS = 27e9
DATA_VOLUME, RUNS_VOLUME, MODELS_VOLUME = "distill-data", "distill-runs", "distill-models"
DATA_MOUNT, RUNS_MOUNT, MODELS_MOUNT = "/data", "/runs", "/models"
MODAL_MAX_TIMEOUT_S = 24 * 3600  # Modal's per-call limit
DEADLINE_MARGIN_S = 600  # stop this long before a deadline: save + commit
MIN_SEGMENT_S = 3600  # a call shorter than this is not worth its load time
MAX_SEGMENTS = 8
MAX_FAILED_CALLS = 1
APPROVAL_FLAG = T.APPROVAL_FLAG

# ---------------------------------------------------------------------------
# Prices (Modal list price, https://modal.com/pricing, read 2026-10-30) and GPUs
# ---------------------------------------------------------------------------

CPU_CORES = 4.0
MEMORY_MIB = 48 * 1024
USD_PER_CORE_S = 0.0000131
USD_PER_GIB_S = 0.00000222
EGRESS_USD_PER_GIB = 0.04


@dataclass(frozen=True)
class Gpu:
    key: str  # Modal's gpu= string
    name: str
    vram_gb: int
    usd_per_s: float
    bf16_dense_tflops: float  # vendor tensor-core peak without sparsity
    util: float  # assumed effective utilization of that peak (likely case)
    fit_16k: str  # 4-bit QLoRA at 16k, the default path
    note: str = ""

    @property
    def usd_per_h(self) -> float:
        return self.usd_per_s * 3600


GPUS: dict[str, Gpu] = {g.key: g for g in (
    Gpu("L40S", "L40S 48 GB", 48, 0.000542, 362.05, 0.30, "likely",
        "48 GB class; GDDR6 864 GB/s"),
    Gpu("A100-80GB", "A100 80 GB", 80, 0.000694, 312.0, 0.30, "yes"),
    Gpu("RTX-PRO-6000", "RTX PRO 6000 96 GB", 96, 0.000842, 500.0, 0.25, "yes",
        "Blackwell; the NVFP4 packed-LoRA benchmark card"),
    Gpu("H100", "H100 80 GB", 80, 0.001097, 989.5, 0.25, "yes",
        "Modal may upgrade H100 to H200 at the same price"),
    Gpu("H200", "H200 141 GB", 141, 0.001261, 989.5, 0.27, "yes", "same compute as H100, more HBM"),
    Gpu("B200", "B200 180 GB", 180, 0.001736, 2250.0, 0.18, "yes",
        "least-proven stack for this model; utilization assumed lowest"),
)}

# How far the measured tokens/sec may plausibly land from the likely case.
TPS_LOW, TPS_HIGH = 0.6, 1.4

# Run size, from the Phase 1 audit: a run happens only after Phase 3 passes its
# 5-10K train-size gate; mean tokens per rebuilt example = Phase 1's stored
# context (p50 2,738 / p90 6,152) + the rebuilt head (system prompt + tool
# list: the 26-tool catalog alone is ~5.6K tokens), many cut at 16,384.
SCENARIOS = {
    "low": {"examples": 5000, "mean_tokens": 8500},
    "likely": {"examples": 7500, "mean_tokens": 11000},
    "high": {"examples": 10000, "mean_tokens": 13500},
}
TODAY = {"examples": 1900, "mean_tokens": 11000}  # what the corpus supports without Phase 2b
# GPU seconds per call that are not training: container start, image, weights
# from the Volume, compile (low / likely / high), and per checkpoint.
LOAD_S = {"low": 600, "likely": 900, "high": 1800}
SAVE_S = 60
STARTUP_ALLOWANCE_S = 300  # billed before our code runs; counted per call
HEARTBEAT_S = 300
# A call that has not closed its ledger segment holds the run while it was
# heard from within LEASE_S (a heartbeat is at most HEARTBEAT_S + one step
# old); before its first heartbeat after the load, the load's worst case on top.
LEASE_S = 3 * HEARTBEAT_S


def rate_per_s(gpu: Gpu) -> float:
    """Dollars per second for one container: GPU + CPU cores + memory."""
    return gpu.usd_per_s + CPU_CORES * USD_PER_CORE_S + (MEMORY_MIB / 1024) * USD_PER_GIB_S


def likely_tps(gpu: Gpu, params: float = PARAMS) -> float:
    """Tokens/sec: utilization x dense BF16 peak / (6 x params) FLOPs per token
    (forward + checkpoint recompute + input gradients; the frozen base gets no
    weight gradients and the LoRA's are negligible)."""
    return gpu.util * gpu.bf16_dense_tflops * 1e12 / (6 * params)


def tps_levels(gpu: Gpu) -> dict[str, float]:
    t = likely_tps(gpu)
    return {"low": t * TPS_LOW, "likely": t, "high": t * TPS_HIGH}


def run_tokens(examples: int, mean_tokens: float, epochs: int = 2) -> int:
    return int(examples * mean_tokens * epochs)


@dataclass
class Estimate:
    gpu: str
    tokens: int
    tps: float
    train_s: float
    overhead_s: float
    calls: int
    seconds: float
    usd: float

    @property
    def hours(self) -> float:
        return self.seconds / 3600


def estimate(gpu: Gpu, tokens: int, tps: float, *, load_s: float, grad_accum: int = 8,
             save_every: int = 200, mean_tokens: float = 11000.0) -> Estimate:
    """Billed seconds and dollars for a run of *tokens* at *tps*."""
    train_s = tokens / tps
    steps = tokens / (mean_tokens * grad_accum)
    saves = math.ceil(steps / save_every) + 1
    per_call = MODAL_MAX_TIMEOUT_S - DEADLINE_MARGIN_S - load_s - STARTUP_ALLOWANCE_S
    work_s = train_s + saves * SAVE_S
    calls = max(1, math.ceil(work_s / per_call))
    overhead_s = calls * (load_s + STARTUP_ALLOWANCE_S) + saves * SAVE_S
    seconds = train_s + overhead_s
    usd = seconds * rate_per_s(gpu) + EGRESS_USD_PER_GIB  # ~1 GB adapter download
    return Estimate(gpu.key, tokens, tps, train_s, overhead_s, calls, seconds, usd)


def run_estimates(gpu: Gpu, tokens_by_level: dict[str, int], mean_tokens: float = 11000.0
                  ) -> dict[str, Estimate]:
    """low = fewest tokens at the fastest plausible speed; high = most tokens
    at the slowest; likely in between."""
    t = tps_levels(gpu)
    speed = {"low": t["high"], "likely": t["likely"], "high": t["low"]}
    return {lvl: estimate(gpu, tokens_by_level[lvl], speed[lvl], load_s=LOAD_S[lvl],
                          mean_tokens=mean_tokens) for lvl in ("low", "likely", "high")}


def scenario_tokens(epochs: int = 2) -> dict[str, int]:
    return {k: run_tokens(v["examples"], v["mean_tokens"], epochs) for k, v in SCENARIOS.items()}


def cost_table(tokens_by_level: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for g in GPUS.values():
        est = run_estimates(g, tokens_by_level)
        rows.append({
            "gpu": g.key, "name": g.name, "usd_per_h": round(rate_per_s(g) * 3600, 4),
            "gpu_usd_per_h": round(g.usd_per_h, 4),
            "tps": {k: round(v) for k, v in tps_levels(g).items()},
            "hours": {k: round(e.hours, 1) for k, e in est.items()},
            "usd": {k: round(e.usd, 2) for k, e in est.items()},
            "calls_likely": est["likely"].calls, "fit_16k": g.fit_16k, "note": g.note,
            "usd_per_mtok_likely": round(rate_per_s(g) / likely_tps(g) * 1e6, 3),
        })
    rows.sort(key=lambda r: r["usd"]["likely"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    out = ["| Rank | GPU (Modal `gpu=`) | $/h all-in | tok/s low / likely / high | "
           "Hours likely | Cost per run low / likely / high | Fits 16k (4-bit) |",
           "|---:|---|---:|---|---:|---|---|"]
    for r in rows:
        t, u = r["tps"], r["usd"]
        out.append(f"| {r['rank']} | {r['name']} (`{r['gpu']}`) | ${r['usd_per_h']:.2f} | "
                   f"{t['low']:,} / {t['likely']:,} / {t['high']:,} | {r['hours']['likely']:,} | "
                   f"${u['low']:,.0f} / **${u['likely']:,.0f}** / ${u['high']:,.0f} | "
                   f"{r['fit_16k']} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The spend ledger and the in-container budget (pure; tested without Modal)
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    id: int
    gpu: str
    rate_per_s: float
    started: float
    last_seen: float
    closed: bool = False
    status: str = "running"  # "lost": a preempted attempt its own restart replaced
    call_id: str = ""  # Modal's function call id (a preempted input restarts under it)
    loaded: bool = False  # the weights are loaded and heartbeats have started

    def holds_run(self, now: float, call_id: str = "") -> bool:
        """Another call may still be training this run: it never closed its
        segment, it is not this call's own earlier attempt, and it was heard
        from recently enough."""
        if self.closed or self.status != "running" or (call_id and self.call_id == call_id):
            return False
        lease = LEASE_S if self.loaded else LOAD_S["high"] + LEASE_S
        return now < self.last_seen + lease


@dataclass
class Ledger:
    """Every GPU call of a run, on the runs Volume next to its checkpoints."""

    path: Path
    segments: list[Segment] = field(default_factory=list)
    calibrations: list[dict[str, Any]] = field(default_factory=list)
    run: dict[str, Any] = field(default_factory=dict)  # what the run started on

    @classmethod
    def from_dict(cls, path: Path, d: dict[str, Any]) -> "Ledger":
        return cls(Path(path), [Segment(**s) for s in d.get("segments", [])],
                   d.get("calibrations", []), d.get("run") or {})

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        p = Path(path)
        if not p.is_file():
            return cls(p)
        return cls.from_dict(p, json.loads(p.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"segments": [asdict(s) for s in self.segments],
                                   "calibrations": self.calibrations, "run": self.run},
                                  indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def open(self, gpu: str, rate: float, now: float, call_id: str = "") -> Segment:
        seg = Segment(len(self.segments) + 1, gpu, rate, now, now, call_id=call_id)
        self.segments.append(seg)
        return seg

    def busy(self, now: float, call_id: str = "") -> Segment | None:
        """A segment of another call that may still be training this run."""
        return next((s for s in self.segments if s.holds_run(now, call_id)), None)

    def close(self, seg: Segment, status: str, now: float) -> None:
        seg.last_seen, seg.closed, seg.status = now, True, status

    def spent_usd(self, now: float | None = None, current: Segment | None = None) -> float:
        """Billed dollars so far. Every call adds the startup we cannot see; a
        call that never closed (killed, preempted) adds a heartbeat interval
        for the time after its last write."""
        total = 0.0
        for s in self.segments:
            end = now if (s is current and now is not None) else s.last_seen
            tail = 0.0 if (s.closed or s is current) else HEARTBEAT_S
            total += (max(0.0, end - s.started) + STARTUP_ALLOWANCE_S + tail) * s.rate_per_s
        return total

    def calibration_for(self, gpu: str, max_seq: int | None = None) -> dict[str, Any] | None:
        """The newest calibration (or re-check) on *gpu*, at *max_seq* if given."""
        for c in reversed(self.calibrations):
            if c["gpu"] == gpu and (max_seq is None or c["max_seq"] == max_seq):
                return c
        return None


def call_ahead_s(ledger: Ledger, job: dict[str, Any]) -> float:
    """GPU seconds a call spends before its first step has finished: the load
    plus one step, as calibrated on this GPU, else the plan's slow-case
    estimates (``expect_load_s`` / ``expect_step_s``)."""
    cal = ledger.calibration_for(job["gpu"])
    load_s = (cal or {}).get("load_s")
    step_s = (cal or {}).get("step_s")
    return (float(load_s if load_s is not None else job.get("expect_load_s", LOAD_S["high"]))
            + float(step_s if step_s is not None else job.get("expect_step_s", 0.0)))


@dataclass
class Budget:
    ledger: Ledger
    seg: Segment
    max_usd: float
    deadline: float  # epoch seconds: launch + max_hours
    seg_deadline: float  # epoch seconds: this call's timeout
    commit: Callable[[], None]
    clock: Callable[[], float] = time.time
    last_step_s: float = 0.0
    _beat: float = 0.0

    def check(self, ahead_s: float = 0.0) -> str | None:
        """Why the run must stop before another step, or None. The reserve is
        the stop's own checkpoint plus the next step (1.5x the last one), so a
        step never starts unless it can finish under the cap. *ahead_s* is
        work that must happen before that, unchecked: the weights load and
        the first step before a call has timed one."""
        now = self.clock()
        reserve = (SAVE_S * 2 + ahead_s + 1.5 * self.last_step_s) * self.seg.rate_per_s
        if self.ledger.spent_usd(now, self.seg) + reserve >= self.max_usd:
            return "max_usd"
        if now + ahead_s >= self.deadline - DEADLINE_MARGIN_S:
            return "max_hours"
        if now + ahead_s >= self.seg_deadline - DEADLINE_MARGIN_S:
            return "call_timeout"
        return None

    def hook(self, trainer: Any, rec: Any) -> str | None:
        now = self.clock()
        self.seg.last_seen = now
        self.last_step_s = float(getattr(rec, "seconds", 0.0))
        if now - self._beat >= HEARTBEAT_S:
            self._beat = now
            self.ledger.save()
            self.commit()
        return self.check()


def measure_tps(records: list[Any], skip: int = 10) -> float:
    """Tokens/sec over the calibration steps after the first *skip* (compile
    and allocator warm-up)."""
    use = records[skip:] if len(records) > skip else records
    secs = sum(r.seconds for r in use)
    return sum(r.tokens for r in use) / secs if secs > 0 else 0.0


def measure_step_s(records: list[Any], skip: int = 10) -> float:
    """Mean seconds per step over the same calibration steps."""
    use = records[skip:] if len(records) > skip else records
    return sum(r.seconds for r in use) / len(use) if use else 0.0


@dataclass
class Projection:
    usd: float
    seconds: float
    calls: int
    tps: float
    remaining_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def project(*, spent_usd: float, remaining_tokens: int, tps: float, rate: float, seg_left_s: float,
            load_s: float, remaining_saves: int) -> Projection:
    """The exact cost of finishing from here at the measured speed: spent so
    far + the remaining training + checkpoints + the load of every further call
    + the adapter download."""
    if tps <= 0:
        return Projection(float("inf"), float("inf"), 0, tps, remaining_tokens)
    work_s = remaining_tokens / tps + (remaining_saves + 1) * SAVE_S
    per_call = MODAL_MAX_TIMEOUT_S - DEADLINE_MARGIN_S - load_s - STARTUP_ALLOWANCE_S
    extra = max(0, math.ceil((work_s - max(0.0, seg_left_s - DEADLINE_MARGIN_S)) / per_call))
    seconds = work_s + extra * (load_s + STARTUP_ALLOWANCE_S)
    return Projection(spent_usd + seconds * rate + EGRESS_USD_PER_GIB, seconds, 1 + extra, tps,
                      remaining_tokens)


# ---------------------------------------------------------------------------
# One GPU call (runs in the container; Modal-free so it is testable)
# ---------------------------------------------------------------------------


def run_segment(job: dict[str, Any], *, make_trainer: Callable[[dict[str, Any]], Any],
                commit: Callable[[], None], run_dir: Path, max_derived_rows: int,
                clock: Callable[[], float] = time.time, call_id: str = "") -> dict[str, Any]:
    t0 = clock()
    ledger = Ledger.load(run_dir / "ledger.json")
    holder = ledger.busy(t0, call_id)
    if holder is not None:
        # Two calls on one run would each budget against a stale view of the
        # other's spend and overwrite its ledger and checkpoints. Nothing is
        # written here: the ledger is the running call's.
        return {"status": "busy", "segment": holder.id, "spent_usd": round(ledger.spent_usd(), 2),
                "reason": f"segment {holder.id} (call {holder.call_id or '?'}) is still running "
                          "this run"}
    for s in ledger.segments:  # this call's own attempt, before a preemption restarted it
        if call_id and s.call_id == call_id and not s.closed and s.status == "running":
            s.status = "lost"
    seg = ledger.open(job["gpu"], job["rate_per_s"], t0, call_id)
    ledger.save()
    commit()

    def finish(status: str, **extra: Any) -> dict[str, Any]:
        ledger.close(seg, status, clock())
        ledger.save()
        commit()
        return {"status": status, "segment": seg.id, "spent_usd": round(ledger.spent_usd(), 2),
                **extra}

    if max_derived_rows and not job.get("approved"):
        return finish("refused", reason="Max-derived data without the operator's approval")
    # A resume needs the weights the run started on: a moved branch would
    # change the config hash, and the trainer would refuse after the load.
    started_on = ledger.run.get("base_revision")
    if ledger.run and started_on != job.get("base_revision"):
        return finish("refused", reason=f"the run started on base revision {started_on}; this "
                                        f"call has {job.get('base_revision')}")
    if not ledger.run:
        ledger.run = {"base_model": job.get("base_model"),
                      "base_revision": job.get("base_revision"),
                      "data_sha256": job.get("data_sha256"), "started_at": T.now_iso()}
    budget = Budget(ledger, seg, job["max_usd"], job["deadline"],
                    t0 + job["call_timeout_s"], commit, clock)
    # The budget is next consulted after the first step, so the load and that
    # step are reserved before anything is loaded.
    early = budget.check(ahead_s=call_ahead_s(ledger, job))
    if early:
        return finish(f"stop:{early}")
    trainer = make_trainer(job)

    def decide(tps: float, step_s: float, load_s: float, *, recheck: bool) -> dict[str, Any]:
        """The exact cost of the rest at *tps*, against this job's cap and
        deadline, recorded in the ledger."""
        now = clock()
        steps_left = max(0, trainer.total_steps - trainer.step)
        proj = project(spent_usd=ledger.spent_usd(now, seg),
                       remaining_tokens=trainer.remaining_tokens(), tps=tps,
                       rate=job["rate_per_s"], seg_left_s=t0 + job["call_timeout_s"] - now,
                       load_s=load_s,
                       remaining_saves=math.ceil(steps_left / trainer.cfg.save_every))
        over_time = now + proj.seconds > job["deadline"] - DEADLINE_MARGIN_S
        decision = ("over_cap" if proj.usd > job["max_usd"] else
                    "over_time" if over_time else "continue")
        rec = {"gpu": job["gpu"], "max_seq": trainer.max_seq, "at_step": trainer.step,
               "tokens_per_s": round(tps, 1), "step_s": round(step_s, 1), "load_s": round(load_s),
               "projection": proj.as_dict(), "max_usd": job["max_usd"],
               "decision": decision, "at": T.now_iso(), **({"recheck": True} if recheck else {})}
        ledger.calibrations.append(rec)
        ledger.save()
        commit()
        return rec

    try:
        trainer.setup(resume=True)
        load_s = clock() - t0
        seg.loaded, seg.last_seen = True, clock()
        ledger.save()
        commit()
        calib = ledger.calibration_for(job["gpu"], trainer.max_seq)
        late = budget.check(ahead_s=float((calib or {}).get("step_s")
                                          or job.get("expect_step_s", 0.0)))
        if late:
            return finish(f"stop:{late}", step=trainer.step)
        if calib is None:
            # Calibration: real steps of the run (kept if it continues), timed
            # after the first few, then the exact cost of the rest.
            res = trainer.run(until_step=trainer.step + job["calib_steps"], hook=budget.hook)
            if res.status == "until_step":
                skip = job.get("calib_skip", 10)
                trainer.save("calibration")
                calib = decide(measure_tps(res.records, skip), measure_step_s(res.records, skip),
                               load_s, recheck=False)
                if calib["decision"] != "continue":
                    return finish(calib["decision"], calibration=calib, step=trainer.step)
                res = trainer.run(hook=budget.hook)
        else:
            # A relaunch (or the next call of a chain) re-projects from the
            # measured speed against this job's cap and deadline: a run that
            # stopped over_cap does not train on to the hard cap.
            calib = decide(float(calib["tokens_per_s"]), float(calib.get("step_s") or 0.0),
                           load_s, recheck=True)
            if calib["decision"] != "continue":
                return finish(calib["decision"], calibration=calib, step=trainer.step)
            res = trainer.run(hook=budget.hook)
    except T.StepDown as sd:
        return finish("stepdown", max_seq=sd.new, step=sd.step)
    except (T.UsageError, T.DataError) as exc:  # a retry would fail the same way
        return finish("refused", reason=str(exc))
    except T.RestoreError as exc:
        return finish("failed", reason=str(exc))
    except BaseException as exc:
        finish("error", reason=f"{type(exc).__name__}: {exc}")  # not left holding the run
        raise
    if res.status != "done":
        return finish(f"stop:{res.reason}", step=trainer.step)
    adapter = trainer.save_adapter(run_dir / "adapter")
    T.write_json(run_dir / "manifest.json", {
        "schema": T.MANIFEST_SCHEMA, "mode": "modal", "run_id": job["run_id"],
        "written_at": T.now_iso(), "gpu": job["gpu"], "rate_per_s": job["rate_per_s"],
        "max_usd": job["max_usd"], "data_sha256": job.get("data_sha256"),
        "base_model": {"id": job.get("base_model"), "revision_resolved": job.get("base_revision"),
                       "load": trainer.cfg.load},
        "config": trainer.cfg.as_dict(), "seed": trainer.cfg.seed, "step": trainer.step,
        "tokens_seen": trainer.tokens_seen, "seq_history": trainer.run_state.get("seq_history"),
        "calibrations": ledger.calibrations, "versions": trainer.backend.versions(),
        "image": {"ref": UNSLOTH_IMAGE_REF, "tag": UNSLOTH_IMAGE,
                  "digest": UNSLOTH_IMAGE_DIGEST}})
    commit()
    return finish("done", step=trainer.step, adapter=str(adapter),
                  calibration=ledger.calibration_for(job["gpu"], trainer.max_seq))


# ---------------------------------------------------------------------------
# The local side: plan, guard, upload, calls, download (Modal behind Ops)
# ---------------------------------------------------------------------------


class Ops(Protocol):
    def upload_data(self, local: Path, name: str, meta: dict[str, Any]) -> str: ...
    def prefetch_model(self, repo: str, revision: str | None) -> dict[str, Any]: ...
    def read_ledger(self, run_id: str) -> dict[str, Any] | None: ...
    def train_call(self, job: dict[str, Any], gpu: str, timeout_s: int) -> dict[str, Any]: ...
    def download(self, run_id: str, dest: Path) -> list[str]: ...


def precheck(ledger_doc: dict[str, Any] | None, job: dict[str, Any], now: float
             ) -> dict[str, Any] | None:
    """Why the next call must not start, decided locally from the run's
    ledger before a GPU container is paid for; None to go ahead. The call
    itself checks again (this read can be stale)."""
    if not ledger_doc:
        return None
    led = Ledger.from_dict(Path("ledger.json"), ledger_doc)
    holder = led.busy(now)
    if holder is not None:
        return {"status": "busy", "reason": f"segment {holder.id} (call {holder.call_id or '?'}) "
                                            "has not ended"}
    spent = led.spent_usd()
    need = (STARTUP_ALLOWANCE_S + 2 * SAVE_S + call_ahead_s(led, job)) * job["rate_per_s"]
    if spent + need >= job["max_usd"]:
        return {"status": "stop:max_usd", "spent_usd": round(spent, 2),
                "reason": f"${spent:,.2f} spent; a call needs ~${need:,.2f} before its first step"}
    cal = led.calibration_for(job["gpu"])
    if cal and cal.get("decision") != "continue":
        proj = cal.get("projection") or {}
        if float(proj.get("usd", 0.0)) > job["max_usd"]:
            return {"status": "over_cap", "calibration": cal}
        if now + float(proj.get("seconds", 0.0)) > job["deadline"] - DEADLINE_MARGIN_S:
            return {"status": "over_time", "calibration": cal}
    return None


# What to do after a launch that did not finish (by its last status).
NEXT_STEP = {
    "over_cap": "the projected cost is over --max-usd: to go on, relaunch with a higher "
                "--max-usd and --run-id {run_id}",
    "over_time": "the projected run ends after --max-hours: to go on, relaunch with a higher "
                 "--max-hours and --run-id {run_id}",
    "stop:max_usd": "the dollar cap is spent: to go on, relaunch with a higher --max-usd and "
                    "--run-id {run_id}",
    "busy": "another call is still running this run: wait until it has ended (`modal app "
            "list`), then relaunch with --run-id {run_id}",
    "refused": "the call refused the run (see its reason): fix that before relaunching",
}


@dataclass
class RentArgs:
    data: Path
    gpu: str = "H100"
    max_usd: float = 0.0
    max_hours: float = 40.0
    run_id: str = ""
    base_model: str = DEFAULT_BASE
    base_revision: str = ""
    max_seq: int = 16384
    epochs: int = 2
    r: int = 32
    alpha: int = 32
    lr: float = 1e-4
    grad_accum: int = 8
    save_every: int = 200
    calib_steps: int = 50
    out: Path = Path("data/distill/rental")
    launch: bool = False
    approved: bool = False


def train_config(a: RentArgs, model_dir: str) -> dict[str, Any]:
    cfg = T.TrainConfig(base_model=model_dir, load="4bit", r=a.r, alpha=a.alpha, lr=a.lr,
                        epochs=a.epochs, max_seq=a.max_seq, grad_accum=a.grad_accum,
                        save_every=a.save_every)
    errs = cfg.validate()
    if errs:
        raise ValueError("; ".join(errs))
    return cfg.as_dict()


def orchestrate(a: RentArgs, ops: Ops | None, *, clock: Callable[[], float] = time.time,
                echo: Callable[[str], None] = print) -> int:
    """0 done (or print-only plan), 1 failed/stopped, 2 usage, 3 refused
    (Max-derived data), 4 over the cap before spending, 5 stopped by the
    calibration decision, 6 another call is still running this run."""
    meta_path = Path(a.data) / "meta.json"
    if not meta_path.is_file():
        echo(f"error: {a.data} is not a pre-tokenized directory (run `pretokenize` first)")
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    gpu = GPUS.get(a.gpu)
    if gpu is None:
        echo(f"error: --gpu must be one of {', '.join(GPUS)}")
        return 2
    if not a.max_usd or a.max_usd <= 0:
        echo("error: --max-usd is required (the dollar cap; the run stops before it)")
        return 2
    if a.max_hours <= 0:
        echo("error: --max-hours must be positive")
        return 2
    if str(a.max_seq) not in {str(x) for x in meta.get("ladder", [])}:
        echo(f"error: the data has no variants for max_seq {a.max_seq}")
        return 2
    md = int(meta.get("max_derived_rows", meta.get("rows", 1)))
    if md and not a.approved:
        echo(f"REFUSED: {md}/{meta.get('rows')} rows are Max-derived. Training on Max-derived data "
             f"is on hold (DISTILL-OPEN-QUESTIONS #6); {APPROVAL_FLAG} only with the operator's "
             "approval.")
        return 3
    tokens = int(meta["tokens_by_seq"][str(a.max_seq)]) * a.epochs
    kept = int(meta["kept_by_seq"][str(a.max_seq)]) or 1
    est = run_estimates(gpu, {k: tokens for k in ("low", "likely", "high")},
                        mean_tokens=tokens / a.epochs / kept)
    rate = rate_per_s(gpu)
    echo(f"plan: {gpu.name} at ${rate * 3600:.2f}/h all-in, {tokens:,} tokens "
         f"({kept} examples x {a.epochs} epochs at max_seq {a.max_seq})")
    for lvl, e in est.items():
        echo(f"  {lvl:>6}: {e.tps:,.0f} tok/s (assumed) -> {e.hours:,.1f} h, {e.calls} call(s), "
             f"${e.usd:,.2f}")
    echo(f"  cap ${a.max_usd:,.2f}, max {a.max_hours:g} h; the 50-step calibration measures the "
         "real speed and decides")
    if est["low"].usd > a.max_usd:
        echo(f"REFUSED: even the optimistic estimate (${est['low'].usd:,.2f}) is over the cap")
        return 4
    if not a.launch:
        echo("print-only: nothing launched (pass --launch to rent)")
        return 0
    if ops is None:
        echo("error: launching needs the Modal client (`modal run ...::rent`)")
        return 2
    run_id = a.run_id or f"27b-{gpu.key.lower()}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    # A relaunch continues on the weights the run started on (the branch may
    # have moved since; the config hash holds the resolved revision).
    prior: dict[str, Any] = {}
    if a.run_id:
        prior = (ops.read_ledger(run_id) or {}).get("run") or {}
    if prior.get("base_model") not in (None, a.base_model):
        echo(f"REFUSED: run {run_id} started on {prior['base_model']}, not {a.base_model}")
        return 2
    name = Path(a.data).name
    echo(f"upload: {ops.upload_data(Path(a.data), name, meta)}")
    model = ops.prefetch_model(a.base_model, a.base_revision or prior.get("base_revision"))
    if prior.get("base_revision") and model["sha"] != prior["base_revision"]:
        echo(f"REFUSED: run {run_id} started on {a.base_model}@{prior['base_revision']}; this "
             f"launch resolves to {model['sha']} (drop --base-revision to reuse the run's)")
        return 2
    echo(f"weights: {a.base_model}@{model['sha']} in the models Volume")
    deadline = clock() + a.max_hours * 3600
    job = {"run_id": run_id, "data": name, "gpu": gpu.key, "rate_per_s": rate,
           "max_usd": a.max_usd, "deadline": deadline, "calib_steps": a.calib_steps,
           "calib_skip": 10, "approved": a.approved, "base_model": a.base_model,
           "base_revision": model["sha"],
           "config": train_config(a, model["dir"]), "data_sha256": meta.get("content_sha256"),
           # until a calibration measures them: the worst-case load and one
           # step at the slow-case speed (the budget reserves both up front)
           "expect_load_s": LOAD_S["high"],
           "expect_step_s": round(a.grad_accum * tokens / a.epochs / kept
                                  / (likely_tps(gpu) * TPS_LOW), 1)}
    calls = failures = 0
    res: dict[str, Any] = {"status": "not-started"}
    while calls < MAX_SEGMENTS:
        left = deadline - clock()
        if left < MIN_SEGMENT_S:
            echo("stop: under an hour left of --max-hours")
            break
        stop = precheck(ops.read_ledger(run_id), job, clock())
        if stop is not None:
            res = stop
            echo(f"not starting call {calls + 1}: {json.dumps(stop, default=str)}")
            break
        timeout = int(min(MODAL_MAX_TIMEOUT_S, left))
        job["call_timeout_s"] = timeout
        calls += 1
        try:
            res = ops.train_call(job, gpu.key, timeout)
        except Exception as exc:  # preempted or crashed: the ledger counted it
            failures += 1
            echo(f"call {calls} failed: {type(exc).__name__}: {exc}")
            if failures > MAX_FAILED_CALLS:
                res = {"status": "failed"}
                break
            continue
        echo(f"call {calls}: {json.dumps(res, default=str)}")
        if res["status"] in ("stepdown", "stop:call_timeout"):
            continue
        break
    status = res["status"]
    if status == "done":
        files = ops.download(run_id, Path(a.out) / run_id)
        echo(f"adapter: {len(files)} files in {Path(a.out) / run_id}")
        return 0
    nxt = NEXT_STEP.get(status, "once no call of this run is running (`modal app list`), "
                                "relaunch with --run-id {run_id}")
    echo(f"run {run_id} ended: {status}; checkpoints stay in the '{RUNS_VOLUME}' Volume. "
         + nxt.format(run_id=run_id))
    return 5 if status in ("over_cap", "over_time") else 6 if status == "busy" else 1


# ---------------------------------------------------------------------------
# Modal wiring (only where the client is installed)
# ---------------------------------------------------------------------------


def build_image(m: Any) -> Any:
    return (m.Image.from_registry(UNSLOTH_IMAGE_REF)
            .entrypoint([])  # the image's own entrypoint starts services; not here
            .env({"PYTHONUNBUFFERED": "1", "UNSLOTH_SKIP_NOTEBOOK_SYNC": "1",
                  "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            .add_local_file(str(HERE / "train_qlora.py"), "/root/distill/train_qlora.py")
            .add_local_file(str(HERE / "render.py"), "/root/distill/render.py"))


def remote_trainer(job: dict[str, Any], run_dir: Path, commit: Callable[[], None]) -> Any:
    d = dict(job["config"])
    d["target_modules"], d["ladder"] = tuple(d["target_modules"]), tuple(d["ladder"])
    cfg = T.TrainConfig(**d)
    source = T.PretokSource(Path(DATA_MOUNT) / job["data"])
    backend = T.UnslothBackend(cfg)
    return T.Trainer(cfg, source, run_dir, backend, metrics_path=run_dir / "metrics.jsonl",
                     on_save=lambda _p: commit(), revision=None)


def download_run(vol: Any, run_id: str, dest: Path) -> list[str]:
    """The adapter and the run's records (not the checkpoints) from the Volume."""
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for entry in vol.listdir(run_id, recursive=True):
        path = str(entry.path)
        rel = path[len(run_id):].lstrip("/") if path.startswith(run_id) else path
        kind = str(getattr(entry.type, "name", entry.type)).upper()
        if not kind.endswith("FILE"):
            continue
        if not (rel.startswith("adapter/") or rel in ("ledger.json", "run_state.json",
                                                      "metrics.jsonl", "manifest.json")):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            for chunk in vol.read_file(path):
                fh.write(chunk)
        got.append(rel)
    return got


if modal is not None:
    app = modal.App(APP_NAME)
    image = build_image(modal)
    data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
    runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
    models_vol = modal.Volume.from_name(MODELS_VOLUME, create_if_missing=True)

    @app.function(image=image, cpu=2.0, memory=8192, timeout=3 * 3600,
                  volumes={MODELS_MOUNT: models_vol})
    def prefetch(repo: str, revision: str | None) -> dict[str, Any]:
        """Base weights into the models Volume, on CPU (no GPU is billed for it)."""
        from huggingface_hub import HfApi, snapshot_download

        sha = HfApi().model_info(repo, revision=revision or "main").sha
        dest = Path(MODELS_MOUNT) / f"{repo.replace('/', '--')}@{sha}"
        if not (dest / ".complete").is_file():
            snapshot_download(repo, revision=sha, local_dir=str(dest))
            (dest / ".complete").write_text(sha + "\n")
            models_vol.commit()
        return {"dir": str(dest), "sha": sha}

    # single_use_containers: a step-down or a call-timeout stop continues in a
    # new container, so the next call gets the fresh CUDA context (no OOM'd
    # allocator, no second copy of the weights) that a step-down assumes.
    @app.function(image=image, gpu="H100", cpu=CPU_CORES, memory=MEMORY_MIB,
                  timeout=MODAL_MAX_TIMEOUT_S, scaledown_window=2, single_use_containers=True,
                  volumes={DATA_MOUNT: data_vol, RUNS_MOUNT: runs_vol, MODELS_MOUNT: models_vol})
    def train(job: dict[str, Any]) -> dict[str, Any]:
        """One GPU call: calibrate (first call), then train until done or a limit."""
        run_dir = Path(RUNS_MOUNT) / job["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        src_meta = json.loads((Path(DATA_MOUNT) / job["data"] / "meta.json").read_text())
        if src_meta.get("content_sha256") != job["data_sha256"]:
            return {"status": "refused", "reason": "data on the Volume is not the planned data"}
        return run_segment(job, make_trainer=lambda j: remote_trainer(j, run_dir, runs_vol.commit),
                           commit=runs_vol.commit, run_dir=run_dir,
                           max_derived_rows=int(src_meta.get("max_derived_rows", 1)),
                           call_id=str(modal.current_function_call_id() or ""))

    class ModalOps:
        def upload_data(self, local: Path, name: str, meta: dict[str, Any]) -> str:
            try:
                there = json.loads(b"".join(data_vol.read_file(f"{name}/meta.json")))
            except Exception:
                there = None
            if there and there.get("content_sha256") == meta.get("content_sha256"):
                return "already on the Volume"
            with data_vol.batch_upload(force=True) as batch:
                batch.put_directory(str(local), f"/{name}")
            return f"{name} uploaded"

        def prefetch_model(self, repo: str, revision: str | None) -> dict[str, Any]:
            return prefetch.remote(repo, revision)

        def read_ledger(self, run_id: str) -> dict[str, Any] | None:
            try:
                return json.loads(b"".join(runs_vol.read_file(f"{run_id}/ledger.json")))
            except Exception:  # a new run has none; each call checks its ledger again
                return None

        def train_call(self, job: dict[str, Any], gpu: str, timeout_s: int) -> dict[str, Any]:
            return train.with_options(gpu=gpu, timeout=timeout_s).remote(job)

        def download(self, run_id: str, dest: Path) -> list[str]:
            return download_run(runs_vol, run_id, dest)

    @app.local_entrypoint()
    def rent(data: str, gpu: str = "H100", max_usd: float = 0.0, max_hours: float = 40.0,
             run_id: str = "", base_model: str = DEFAULT_BASE, base_revision: str = "",
             max_seq: int = 16384, epochs: int = 2, r: int = 32, alpha: int = 32,
             lr: float = 1e-4, grad_accum: int = 8, save_every: int = 200, calib_steps: int = 50,
             out: str = "data/distill/rental", launch: bool = False,
             i_have_operator_approval_for_max_derived_data: bool = False) -> None:
        a = RentArgs(Path(data), gpu, max_usd, max_hours, run_id, base_model, base_revision,
                     max_seq, epochs, r, alpha, lr, grad_accum, save_every, calib_steps,
                     Path(out), launch, i_have_operator_approval_for_max_derived_data)
        code = orchestrate(a, ModalOps())
        if code:
            sys.exit(code)

    @app.local_entrypoint()
    def fetch(run_id: str, out: str = "data/distill/rental") -> None:
        files = download_run(runs_vol, run_id, Path(out) / run_id)
        print(f"{len(files)} files in {Path(out) / run_id}")


# ---------------------------------------------------------------------------
# Offline CLI: estimate, pretokenize, plan
# ---------------------------------------------------------------------------


def cmd_estimate(a: argparse.Namespace) -> int:
    if a.pretokenized:
        meta = json.loads((a.pretokenized / "meta.json").read_text(encoding="utf-8"))
        n = int(meta["tokens_by_seq"][str(a.max_seq)]) * a.epochs
        tokens = {"low": n, "likely": n, "high": n}
        basis = f"{n:,} tokens from {a.pretokenized} (max_seq {a.max_seq}, {a.epochs} epochs)"
    else:
        tokens = scenario_tokens(a.epochs)
        basis = ", ".join(f"{k} {v['examples']:,} x {v['mean_tokens']:,} tok x {a.epochs} = "
                          f"{tokens[k] / 1e6:.0f}M" for k, v in SCENARIOS.items())
    rows = cost_table(tokens)
    if a.json:
        print(json.dumps({"basis": basis, "rows": rows}, indent=2))
    elif a.markdown:
        print(markdown_table(rows))
    else:
        print(f"tokens: {basis}")
        print(markdown_table(rows))
    return 0


def cmd_pretokenize(a: argparse.Namespace) -> int:
    rows = T.read_jsonl(a.data)
    encoder = T.hf_encoder(a.tokenizer_dir)
    refit = T.template_refitter(a.tokenizer_dir)
    tok_json = Path(a.tokenizer_dir).expanduser() / "tokenizer.json"
    meta = T.write_pretokenized(rows, encoder, a.out, refit=refit,
                                data_sha256=T.sha256_files([a.data]),
                                tokenizer_sha256=T.sha256_files([tok_json]),
                                teacher_max=T.builder_teacher_is_max(a.data))
    print(json.dumps({k: meta[k] for k in ("rows", "kept_by_seq", "tokens_by_seq",
                                           "max_derived_rows", "content_sha256")}, indent=2))
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    ra = RentArgs(data=a.data, gpu=a.gpu, max_usd=a.max_usd, max_hours=a.max_hours,
                  max_seq=a.max_seq, epochs=a.epochs, approved=a.approved)
    return orchestrate(ra, None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modal_train", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate", help="cost per run by GPU (offline)")
    e.add_argument("--pretokenized", type=Path)
    e.add_argument("--max-seq", type=int, default=16384)
    e.add_argument("--epochs", type=int, default=2)
    e.add_argument("--markdown", action="store_true")
    e.add_argument("--json", action="store_true")
    p = sub.add_parser("pretokenize", help="tokenize locally, with per-length variants")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    pl = sub.add_parser("plan", help="the launch plan for one GPU and cap (offline)")
    pl.add_argument("--data", type=Path, required=True)
    pl.add_argument("--gpu", default="H100", choices=list(GPUS))
    pl.add_argument("--max-usd", type=float, required=True)
    pl.add_argument("--max-hours", type=float, default=40.0)
    pl.add_argument("--max-seq", type=int, default=16384)
    pl.add_argument("--epochs", type=int, default=2)
    pl.add_argument(APPROVAL_FLAG, dest="approved", action="store_true")
    a = ap.parse_args(argv)
    return {"estimate": cmd_estimate, "pretokenize": cmd_pretokenize, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
