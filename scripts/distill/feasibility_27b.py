#!/usr/bin/env python3
"""Phase 5: can the large student's QLoRA run on the local 24 GB card?

    # Dry run (default): data shares and wall-clock estimates, no torch, no GPU.
    python3 scripts/distill/feasibility_27b.py --data ~/distill/phase3/train.jsonl
    # Re-estimate from a saved probe without the GPU (writes a new report):
    python3 scripts/distill/feasibility_27b.py --data ... --measured data/distill/feasibility-27b.json

The probe itself (``--execute``) runs only as the GPU tower job runner's
``qwen38-27b-feasibility`` step (scripts/distill/tower/job_runner.py), inside
the window's detached ``distill-jobs-*`` unit that the window start launches
after its rehearsal gate and ``setup apply flywheel``. The runner checks the
window hours, the NO-SWAPS flag, the production model unit and a free GPU,
runs one length per call with ``--out``, passes its step deadline (06:35 at
the latest, never the window's 07:00 end: the restore needs that time) as
``--stop-at``, and sets ``HF_HUB_OFFLINE=1``. A hand run needs all of that
first: a passed rehearsal, the production unit inactive, the GPU at <= 1500
MiB, no NO-SWAPS flag, then a detached launch (``systemd-run --user``, not a
foreground ssh), ``--out`` under the run's results directory and
``--stop-at <date>T06:35`` with that morning's UTC offset (06:35 ET on
2026-11-01 is -05:00: the DST change falls inside that night's window). This
script checks only what it can see itself: ``--stop-at`` and ``--out`` are
required, no child starts while the GPU holds more than ``--gpu-max-used-mib``
(default 1500) or its use cannot be read, and the children run offline
(``HF_HUB_OFFLINE=1`` unless the caller set it).

For each sequence length (16k, 12k, 8k, 4k) ``--execute`` loads the bake-off
winner (default the Unsloth 4-bit Qwen3.8-27B) with Unsloth, attaches the
run's LoRA (r=32 on the language model's linear layers, ``TARGET_MODULES``:
the attention, gated-deltanet and MLP projections, as train_qlora.py trains),
and times ``--steps`` optimizer steps on one sequence of that length. CPU
offload: "unsloth" gradient checkpointing keeps activations in system RAM and
``--offload-embedding`` keeps the input embedding there. ``--offload-layers``
(``auto`` or a count; off by default) uses Unsloth's block swap, which keeps
decoder layers the card cannot hold in system RAM and streams them in (slower).
A load that does not fit on the card without it ("Some modules are dispatched
on the CPU": bitsandbytes cannot quantize there) is recorded as not fitting
(``cpu_spill``), like a CUDA OOM (``cuda_oom``). Each length runs in its own
child process (``sys.executable``), so an OOM at 16k leaves a clean CUDA
context for 12k.

The probe holds the GPU, so it keeps to the window: a length starts only when
its worst case (``CHILD_TIMEOUT_S``, load included) ends before ``--stop-at``,
and each child runs in its own ``distill-feas27b-*`` systemd scope (where
``systemd-run`` exists), which a ``setups.py apply business`` restore stops
like any distill job. A child stopped that way ends the probe; the lengths
not probed are reported as such.

The probe trains on synthetic token ids, never on the dataset: memory and
speed depend on the length, not the content, and the dataset is Max-derived
(DISTILL-OPEN-QUESTIONS #6). The dataset is read for lengths only: the share
of training examples that fit each length untruncated, and the tokens a full
run would process there.

Per length the report has: fits, peak GiB, tokens/sec, the untruncated share,
the full run's tokens and the estimated wall-clock. Without a measurement the
tokens/sec is an *assumption* (stated) and "fits" is unknown. The runbook's
rule: a local run is viable when a length >= 8k fits and the full run is
<= 5 days of compute; then the operator picks CONTINUOUS / NIGHTLY / NO.
It is a rented run (DISTILL-RENTAL-PACKET.md) only when every length >= 8k
was measured not to fit or to be too slow; a length >= 8k that crashed, timed
out or was not probed makes the verdict "incomplete" (probe it again).

Lengths come from the rows as rendered for the 9B student; the 27B's own
tokenizer gives close but not identical counts (re-render with its tokenizer
dir before a real run).

Exit: 0 report written, 2 usage or data error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "distill-feasibility-27b/1"
DEFAULT_MODEL = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
SEQS = (16384, 12288, 8192, 4096)
MIN_LOCAL_SEQ = 8192  # runbook: ">= 8k fits"
MAX_LOCAL_DAYS = 5.0  # runbook: "full run <= 5 days of compute"
PARAMS_27B = 27e9
# Assumed only when nothing was measured. RTX 4090 dense BF16 tensor peak with
# FP32 accumulate (Ada whitepaper) 165.2 TFLOPS; 15% effective utilization with
# CPU-offloaded activations; 6*N FLOPs per trained token (forward, checkpoint
# recompute, input gradients; frozen base, LoRA weight gradients negligible).
ASSUMED_PEAK_TFLOPS = 165.2
ASSUMED_UTIL = 0.15
CHILD_TIMEOUT_S = 1800
DEFAULT_OUT = REPO / "data" / "distill" / "feasibility-27b.json"
# A child ended by one of these was stopped (a restore stops distill-* scopes
# with SIGTERM), not crashed: the probe stops with it.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)
# The LoRA targets: train_qlora.TARGET_MODULES (a test keeps the two equal),
# so the probe measures the adapter the run trains. Attention (q/k/v/o), the
# gated-deltanet token mixer of the linear-attention layers (in_proj_qkv,
# in_proj_z, out_proj) and the MLP. The vision tower's names (qkv, proj,
# linear_fc*) match none of them.
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                  "gate_proj", "up_proj", "down_proj")
# No child starts while the GPU holds more than this: an idle card is a few
# hundred MiB; the business model server alone holds ~20 GiB.
GPU_MAX_USED_MIB = 1500
# transformers' bitsandbytes refusal when a device map puts weights on the CPU,
# and Unsloth's rewrite of it: the model does not fit on the card.
CPU_SPILL_MARKERS = ("Some modules are dispatched on the CPU", "does not fit in GPU memory")


def assumed_tps(params: float = PARAMS_27B) -> float:
    return ASSUMED_PEAK_TFLOPS * 1e12 * ASSUMED_UTIL / (6 * params)


@dataclass
class LengthResult:
    max_seq: int
    fits: bool | None = None  # None: not measured / probe error
    peak_gb: float | None = None
    tokens_per_s: float | None = None
    tps_source: str = "assumed"
    share_untruncated: float = 0.0
    run_tokens: int = 0
    wall_clock_h: float | None = None
    error: str | None = None


def read_lengths(path: Path) -> list[tuple[int, int]]:
    """(tokens as trained, original tokens) per training row; text is not kept."""
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            n = int(row.get("n_tokens") or 0)
            orig = int((row.get("truncated") or {}).get("original_tokens") or n)
            out.append((n, orig))
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def data_at(lengths: list[tuple[int, int]], seq: int, epochs: int) -> tuple[float, int]:
    """Share of rows whose original fits *seq* untruncated, and the tokens a
    full run at *seq* processes (rows over it cut to about *seq*)."""
    share = sum(1 for _, orig in lengths if orig <= seq) / len(lengths)
    tokens = sum(min(n, seq) for n, _ in lengths) * epochs
    return share, tokens


def assess(lengths: list[tuple[int, int]], seqs: tuple[int, ...], epochs: int,
           measured: dict[int, dict[str, Any]] | None = None) -> list[LengthResult]:
    out = []
    for seq in seqs:
        share, tokens = data_at(lengths, seq, epochs)
        r = LengthResult(max_seq=seq, share_untruncated=round(share, 4), run_tokens=tokens)
        m = (measured or {}).get(seq)
        if m:
            r.fits, r.peak_gb, r.error = m.get("fits"), m.get("peak_gb"), m.get("error")
            if m.get("tokens_per_s"):
                r.tokens_per_s, r.tps_source = float(m["tokens_per_s"]), "measured"
        if r.tokens_per_s is None and r.fits is not False:
            r.tokens_per_s = round(assumed_tps(), 1)
        if r.tokens_per_s and r.fits is not False:
            r.wall_clock_h = round(tokens / r.tokens_per_s / 3600, 1)
        out.append(r)
    return out


def decide(results: list[LengthResult]) -> dict[str, Any]:
    """The runbook's local-27B rule, on measured results only. "rent" needs
    every length >= 8k measured (not fitting, or fitting too slowly); one
    that crashed, timed out or was skipped makes the verdict "incomplete"."""
    measured = [r for r in results if r.fits is not None]
    unknown = [r for r in results if r.fits is None and r.max_seq >= MIN_LOCAL_SEQ]
    if not measured and not any(r.error for r in results):
        return {"verdict": "not-measured",
                "say": "nothing measured yet: run with --execute inside a GPU window"}
    ok = [r for r in results if r.fits and r.max_seq >= MIN_LOCAL_SEQ and r.tps_source == "measured"
          and r.wall_clock_h is not None and r.wall_clock_h <= MAX_LOCAL_DAYS * 24]
    if ok:
        best = max(ok, key=lambda r: r.max_seq)
        longer = [r.max_seq for r in unknown if r.max_seq > best.max_seq]
        return {"verdict": "local-viable", "max_seq": best.max_seq,
                "wall_clock_days": round(best.wall_clock_h / 24, 2),  # type: ignore[operator]
                "unmeasured_longer": longer,
                "say": (f"{best.max_seq} fits locally, full run ~{best.wall_clock_h / 24:.1f} days"  # type: ignore[operator]
                        + (f" ({', '.join(map(str, longer))} not measured)" if longer else "")
                        + ": ask the operator CONTINUOUS / NIGHTLY / NO; do nothing without "
                          "a reply")}
    if unknown:
        return {"verdict": "incomplete",
                "unmeasured": [{"max_seq": r.max_seq, "error": r.error} for r in unknown],
                "say": ("not measured at " + ", ".join(f"{r.max_seq} ({r.error or 'no result'})"
                                                       for r in unknown)
                        + ": probe those lengths again in a GPU window before deciding local vs "
                          "rented")}
    longest = max((r.max_seq for r in results if r.fits), default=None)
    return {"verdict": "rent", "longest_fit": longest,
            "say": ("no length >= 8k fits within 5 days of local compute: a rented-GPU run "
                    "(docs/sprints/DISTILL-RENTAL-PACKET.md)")}


def render_table(results: list[LengthResult]) -> str:
    lines = ["| max_seq | fits | peak GiB | tok/s | untruncated | run tokens | wall clock |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for r in results:
        fits = {True: "yes", False: "no", None: "?"}[r.fits]
        tps = f"{r.tokens_per_s:,.0f} ({r.tps_source})" if r.tokens_per_s else "-"
        wall = f"{r.wall_clock_h:,.1f} h ({r.wall_clock_h / 24:.1f} d)" if r.wall_clock_h else "-"
        lines.append(f"| {r.max_seq:,} | {fits}{' ' + r.error if r.error else ''} | "
                     f"{r.peak_gb if r.peak_gb is not None else '-'} | {tps} | "
                     f"{r.share_untruncated:.1%} | {r.run_tokens:,} | {wall} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The GPU probe (child process, one length)
# ---------------------------------------------------------------------------


def parse_offload_layers(value: str) -> str | int:
    """``--offload-layers``: "auto" or a positive layer count (Unsloth's block swap)."""
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--offload-layers takes 'auto' or a layer count") from None
    if n < 1:
        raise argparse.ArgumentTypeError("--offload-layers takes 'auto' or a count >= 1")
    return n


def probe_one(seq: int, *, model: str, revision: str | None, steps: int, r: int,
              offload_embedding: bool, offload_layers: str | int | None = None,
              seed: int = 3407) -> dict[str, Any]:
    """Load, attach LoRA, run *steps* QLoRA steps on one synthetic sequence of
    *seq* tokens. Imports torch/unsloth here only."""
    import unsloth  # noqa: F401  (first: it patches transformers and peft)
    import torch
    from unsloth import FastLanguageModel

    out: dict[str, Any] = {"max_seq": seq, "offload_layers": offload_layers,
                           "target_modules": list(TARGET_MODULES)}
    try:
        kwargs: dict[str, Any] = {}
        if offload_embedding:
            kwargs["offload_embedding"] = True
        if offload_layers:
            kwargs["offload_layers"] = offload_layers
        m, _tok = FastLanguageModel.from_pretrained(
            model_name=model, revision=revision, max_seq_length=seq, load_in_4bit=True,
            full_finetuning=False, dtype=None, **kwargs)
        m = FastLanguageModel.get_peft_model(
            m, r=r, lora_alpha=r, lora_dropout=0.0, target_modules=list(TARGET_MODULES),
            bias="none", use_gradient_checkpointing="unsloth", random_state=seed,
            max_seq_length=seq)
        m.train()
        import bitsandbytes as bnb

        params = [p for p in m.parameters() if p.requires_grad]
        opt = bnb.optim.AdamW8bit(params, lr=1e-4)
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(1000, 100000, (1, seq), generator=g).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = m(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        timed = times[1:] or times  # the first step pays for compilation
        out.update(fits=True, peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   tokens_per_s=round(seq / (sum(timed) / len(timed)), 1),
                   step_seconds=[round(t, 2) for t in times],
                   gpu=torch.cuda.get_device_name(0))
    except Exception as exc:
        kind = oom_kind(exc)
        out.update(fits=False if kind else None,
                   error=kind or f"{type(exc).__name__}: {str(exc)[:200]}")
    return out


def oom_kind(exc: BaseException) -> str | None:
    """How *exc* says the model does not fit, or None when it says something
    else. "cuda_oom": out of GPU memory (train_qlora.is_oom's rule, a cuBLAS
    allocation failure included, plus any "out of memory" message).
    "cpu_spill": the load needed the CPU for weights, which bitsandbytes
    refuses (transformers' ValueError, or Unsloth's rewrite of it)."""
    msg = str(exc)
    if (type(exc).__name__ == "OutOfMemoryError" or "out of memory" in msg.lower()
            or "CUBLAS_STATUS_ALLOC_FAILED" in msg):
        return "cuda_oom"
    if isinstance(exc, ValueError) and any(m in msg for m in CPU_SPILL_MARKERS):
        return "cpu_spill"
    return None


def is_oom(exc: BaseException) -> bool:
    """The model does not fit on the card (see ``oom_kind``)."""
    return oom_kind(exc) is not None


def gpu_used_mib() -> int | None:
    """MiB in use on GPU 0 (nvidia-smi, read-only), or None when unreadable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                              "-i", "0"], capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def say(msg: str, *, err: bool = False) -> None:
    """Print, and keep going when stdout is gone (a closed ssh session): the
    report file is what counts."""
    try:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)
    except (BrokenPipeError, OSError):
        pass


Runner = Callable[[list[str], float], "subprocess.CompletedProcess[str]"]


def child_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The probe children's environment: offline unless the caller said
    otherwise, so a hub id that is not on disk fails instead of downloading
    ~20 GB past the disk floor."""
    env = dict(os.environ if base is None else base)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=child_env())


def stopped_by_signal(returncode: int) -> bool:
    """The child was stopped (killed by SIGTERM/SIGINT, or a shell's 128+n)."""
    return any(returncode in (-int(s), 128 + int(s)) for s in STOP_SIGNALS)


def execute(seqs: tuple[int, ...], a: argparse.Namespace, runner: Runner = _run, *,
            stop_at: float, scope: bool, clock: Callable[[], float] = time.time,
            stamp: str | None = None,
            gpu_used: Callable[[], int | None] = gpu_used_mib) -> dict[int, dict[str, Any]]:
    """Each length in a fresh child (``sys.executable``), in a ``distill-feas27b-*``
    systemd scope when *scope*; a length starts only if its worst case ends
    before *stop_at* and the GPU is free (*gpu_used* at most
    ``a.gpu_max_used_mib``). Results by length."""
    results: dict[int, dict[str, Any]] = {}
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    limit = getattr(a, "gpu_max_used_mib", GPU_MAX_USED_MIB)
    for i, seq in enumerate(seqs):
        left = stop_at - clock()
        if left < CHILD_TIMEOUT_S:
            for rest in seqs[i:]:
                results[rest] = {"fits": None, "error": f"not probed: {CHILD_TIMEOUT_S}s worst "
                                 "case would pass --stop-at"}
            say(f"stop: under {CHILD_TIMEOUT_S}s left before --stop-at; not probed: "
                f"{', '.join(map(str, seqs[i:]))}")
            break
        used = gpu_used()
        if used is None or used > limit:
            why = ("GPU memory use unreadable (nvidia-smi)" if used is None else
                   f"the GPU holds {used} MiB (> {limit}): another program is on it")
            for rest in seqs[i:]:
                results[rest] = {"fits": None, "error": f"not probed: {why}"}
            say(f"stop: {why}; not probed: {', '.join(map(str, seqs[i:]))}", err=True)
            break
        with tempfile.TemporaryDirectory(prefix="feas27b-") as tmp:
            out = Path(tmp) / "one.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(seq),
                   "--json-out", str(out), "--base-model", a.base_model, "--steps", str(a.steps),
                   "--r", str(a.r)]
            if a.base_revision:
                cmd += ["--base-revision", a.base_revision]
            if not a.offload_embedding:
                cmd.append("--no-offload-embedding")
            if a.offload_layers:
                cmd += ["--offload-layers", str(a.offload_layers)]
            if scope:  # a restore stops distill-*.scope units; the scope execs the child
                cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                       f"--unit=distill-feas27b-{seq}-{stamp}", *cmd]
            say(f"probing max_seq {seq} ...")
            stopped = False
            try:
                proc = runner(cmd, CHILD_TIMEOUT_S)  # ends before stop_at (checked above)
                stopped = not out.is_file() and stopped_by_signal(proc.returncode)
                res = json.loads(out.read_text()) if out.is_file() else {
                    "fits": None, "error": f"child exited {proc.returncode} without a result: "
                    + (proc.stderr or "")[-300:]}
                if stopped:
                    res["error"] = f"stopped (signal {abs(proc.returncode) % 128}): a restore?"
            except subprocess.TimeoutExpired:
                res = {"fits": None, "error": f"timeout after {CHILD_TIMEOUT_S}s"}
        results[seq] = res
        say(f"  {seq}: {json.dumps({k: res.get(k) for k in ('fits', 'peak_gb', 'tokens_per_s', 'error')})}")
        if stopped:  # the GPU is being taken back: start nothing more
            for rest in seqs[i + 1:]:
                results[rest] = {"fits": None, "error": "not probed: the probe was stopped"}
            break
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="feasibility_27b", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, help="Phase 3 train.jsonl (lengths only)")
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    ap.add_argument("--base-revision")
    ap.add_argument("--seqs", default=",".join(str(s) for s in SEQS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--no-offload-embedding", dest="offload_embedding", action="store_false")
    ap.add_argument("--offload-layers", type=parse_offload_layers, default=None,
                    help="Unsloth block swap: 'auto' or a count of decoder layers kept in system "
                         "RAM and streamed in (slower; default off)")
    ap.add_argument("--gpu-max-used-mib", type=int, default=GPU_MAX_USED_MIB,
                    help=f"with --execute: no length starts while the GPU holds more than this "
                         f"(default {GPU_MAX_USED_MIB})")
    ap.add_argument("--execute", action="store_true", help="run the GPU probe (default: dry run)")
    ap.add_argument("--stop-at", help="with --execute (required): ISO time with a UTC offset no "
                                      "length may pass: the job runner's stop (06:35 ET), never "
                                      "the window's 07:00 end")
    ap.add_argument("--scope", action=argparse.BooleanOptionalAction, default=None,
                    help="run each probe child in a distill-feas27b-* systemd scope a restore "
                         "can stop (default: when systemd-run exists)")
    ap.add_argument("--measured", type=Path, help="a previous report: reuse its measurements")
    ap.add_argument("--out", type=Path, help=f"report (required with --execute; else default "
                                             f"{DEFAULT_OUT.relative_to(REPO)}, or with --measured "
                                             "<that file>-reestimate.json)")
    ap.add_argument("--one", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--json-out", type=Path, help=argparse.SUPPRESS)
    return ap


def parse_stop_at(value: str) -> float:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("--stop-at needs a UTC offset (e.g. 2026-11-01T06:30-05:00)")
    return dt.timestamp()


def kept_measurement(r: dict[str, Any]) -> bool:
    """A saved result worth re-reading: an OOM (it has no tokens/sec, so its
    tok/s stays "assumed"), a probe error, or a fit with a measured speed."""
    return (r.get("fits") is False or bool(r.get("error"))
            or (bool(r.get("fits")) and r.get("tps_source", "measured") == "measured"))


def main(argv: list[str] | None = None, *, runner: Runner = _run,
         clock: Callable[[], float] = time.time,
         gpu_used: Callable[[], int | None] = gpu_used_mib) -> int:
    a = build_parser().parse_args(argv)
    if a.one:
        res = probe_one(a.one, model=a.base_model, revision=a.base_revision, steps=a.steps, r=a.r,
                        offload_embedding=a.offload_embedding, offload_layers=a.offload_layers)
        a.json_out.write_text(json.dumps(res) + "\n")
        return 0
    if not a.data:
        say("--data is required", err=True)
        return 2
    if a.execute and not a.out:
        say("error: --execute needs --out (the report; the job runner writes one per length "
            "under its results directory)", err=True)
        return 2
    out_path = a.out or (a.measured.with_name(f"{a.measured.stem}-reestimate.json")
                         if a.measured and not a.execute else DEFAULT_OUT)
    if a.measured and out_path.resolve() == a.measured.resolve():
        say("error: --out would overwrite the --measured report; pass another --out", err=True)
        return 2
    try:
        seqs = tuple(sorted({int(s) for s in a.seqs.split(",")}, reverse=True))
        lengths = read_lengths(a.data)
        stop_at = parse_stop_at(a.stop_at) if a.stop_at else None
    except (OSError, ValueError) as exc:
        say(f"error: {exc}", err=True)
        return 2
    measured: dict[int, dict[str, Any]] | None = None
    if a.execute:
        if stop_at is None:
            say("error: --execute needs --stop-at (the job runner's 06:35 stop, with its UTC "
                "offset): the probe holds the GPU and must end before the window's restore",
                err=True)
            return 2
        scope = a.scope if a.scope is not None else shutil.which("systemd-run") is not None
        if not scope:
            say("warning: no systemd scope: a restore cannot stop the probe's children", err=True)
        measured = execute(seqs, a, runner, stop_at=stop_at, scope=scope, clock=clock,
                           gpu_used=gpu_used)
    elif a.measured:
        prev = json.loads(a.measured.read_text())
        measured = {int(r["max_seq"]): r for r in prev.get("results", []) if kept_measurement(r)}
    results = assess(lengths, seqs, a.epochs, measured)
    verdict = decide(results)
    report = {
        "schema": SCHEMA, "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "execute" if a.execute else ("measured" if a.measured else "dry-run"),
        "base_model": a.base_model, "base_revision": a.base_revision, "rows": len(lengths),
        "epochs": a.epochs, "lora": {"r": a.r, "target_modules": list(TARGET_MODULES)},
        "offload_layers": a.offload_layers,
        "assumption": (None if measured else
                       f"{assumed_tps():.0f} tok/s = {ASSUMED_PEAK_TFLOPS} TFLOPS x "
                       f"{ASSUMED_UTIL:.0%} / (6 x {PARAMS_27B / 1e9:.0f}B)"),
        "results": [asdict(r) for r in results], "decision": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    say(render_table(results))
    if report["assumption"]:
        say(f"tok/s is ASSUMED ({report['assumption']}); fits is unknown until --execute")
    say(f"decision: {verdict['verdict']}: {verdict['say']}")
    say(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
