#!/usr/bin/env python3
"""9B QLoRA sequence-length benchmark on the 4090 (operator, Sun 2026-11-01 ~01:30 ET).

    seqlen_bench.py run --trainer .../train_qlora.py --data .../build-v4b-max-derived/train.jsonl \\
        --tokenizer-dir ... --base-model ... --base-revision SHA --seq-len 65536 \\
        --out .../qlora-bench-max-derived/qlora-65536.json --i-have-operator-approval-for-max-derived-data
    seqlen_bench.py run ... --dry-run        # no torch: the guard, the packing, the plan (CPU only)

How long a sequence the 9B QLoRA trains at, and how fast, with activation offload to CPU RAM. One
process per length (``capacity.py``'s block runs each as the unit ``distill-cap-qlora-<len>``), so
the CUDA context and the peak counters belong to that length alone.

* **Model and recipe:** ``train_qlora.py``'s own ``UnslothBackend`` and ``TrainConfig`` (QLoRA r=32 /
  alpha=32 on the same target modules, 8-bit AdamW, micro-batch 1, the base the full run used) with
  ``max_seq`` = the length and grad_accum 1. ``gradient_checkpointing="unsloth"`` (the trainer's
  default) is Unsloth's offloaded gradient checkpointing: each checkpointed layer's input is copied
  to CPU RAM (pinned up to a cap, then pageable) and brought back for the backward pass; the result
  records the evidence (the patched checkpoint function, the host buffers it holds).
* **Data:** real v4b train rows, tokenized exactly as the trainer does (``tokenize_row``, checked
  against each row's recorded tokens), concatenated in a seeded order and cut to exactly the length;
  loss only on the teacher spans. ``--warmup-steps`` (untimed: kernels compile and tune) then
  ``--steps`` timed optimizer steps, one sequence each.
* **Measured:** tokens/s per step (median and mean of the timed steps), peak VRAM (torch allocated
  and reserved; the block adds nvidia-smi's peak), peak RAM (this process's VmHWM; host MemAvailable
  low point), and out-of-memory (CUDA OOM: status ``oom`` with the step and phase).
* **Private (Q6b):** the data is Max-derived, so the same two approvals as the trainer are required
  (this command line's flag and the dataset's recorded approval), the output path must carry the
  ``max-derived`` tag, and nothing is saved but the result JSON: no adapter, no checkpoint.

The result JSON is rewritten after every step (``status: running``), so a process killed by the host
(out of RAM) still leaves what it measured. Exit: 0 ok, 3 CUDA out of memory, 1 failed, 2 usage/refused.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCHEMA = "distill-capacity-qlora-bench/1"
APPROVAL_FLAG = "--i-have-operator-approval-for-max-derived-data"
PACK_SEED = "capacity-qlora-2026-11-01"
SAMPLE_ROWS = 160


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_trainer(path: Path) -> Any:
    name = "distill_train_qlora"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Rows and packing (pure)
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Rows split on newlines only (Unicode line separators inside a row stay in it)."""
    with open(path, "rb") as fh:
        for raw in fh:
            if raw.strip():
                yield json.loads(raw)


def sample_rows(rows: Iterable[Dict[str, Any]], k: int, seed: str,
                on_row: Optional[Callable[[Dict[str, Any]], None]] = None) -> List[Dict[str, Any]]:
    """A seeded reservoir sample of ``k`` rows (every row is passed to ``on_row`` first)."""
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if on_row is not None:
            on_row(row)
        if len(out) < k:
            out.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = row
    rng.shuffle(out)
    return out


def pack(tokenized: Iterable[Tuple[Sequence[int], Sequence[Tuple[int, int]]]], length: int,
         count: int) -> List[Tuple[List[int], List[Tuple[int, int]]]]:
    """``count`` sequences of exactly ``length`` tokens: rows concatenated in order, the last one cut;
    loss spans shifted with their row and clipped at the cut. Raises when the rows run out."""
    out: List[Tuple[List[int], List[Tuple[int, int]]]] = []
    ids: List[int] = []
    spans: List[Tuple[int, int]] = []
    for row_ids, row_spans in tokenized:
        pos = 0
        n = len(row_ids)
        while pos < n:
            room = length - len(ids)
            take = min(room, n - pos)
            base = len(ids)
            ids.extend(row_ids[pos:pos + take])
            for a, b in row_spans:
                lo, hi = max(a, pos), min(b, pos + take)
                if lo < hi:
                    spans.append((base + lo - pos, base + hi - pos))
            pos += take
            if len(ids) == length:
                out.append((ids, spans))
                if len(out) == count:
                    return out
                ids, spans = [], []
            # the rest of a cut row is not reused: every sequence starts at a row start
            if pos < n and len(ids) == 0:
                break
    raise ValueError(f"the sampled rows give only {len(out)} of {count} sequences of {length} tokens")


def loss_positions(spans: Sequence[Tuple[int, int]]) -> int:
    return sum(max(0, b - max(a, 1)) for a, b in spans)


# ---------------------------------------------------------------------------
# Memory watch
# ---------------------------------------------------------------------------


def _meminfo_kb(key: str) -> Optional[int]:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for ln in fh:
                if ln.startswith(key + ":"):
                    return int(ln.split()[1])
    except OSError:
        return None
    return None


def _status_kb(key: str) -> Optional[int]:
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for ln in fh:
                if ln.startswith(key + ":"):
                    return int(ln.split()[1])
    except OSError:
        return None
    return None


class MemWatch(threading.Thread):
    """Host MemAvailable low point and this process's RSS high point, every half second."""

    def __init__(self, period: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.period = period
        self.avail_min: Optional[int] = None
        self.rss_max: Optional[int] = None
        self.avail_start = _meminfo_kb("MemAvailable")
        self.total = _meminfo_kb("MemTotal")
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            a, r = _meminfo_kb("MemAvailable"), _status_kb("VmRSS")
            if a is not None:
                self.avail_min = a if self.avail_min is None else min(self.avail_min, a)
            if r is not None:
                self.rss_max = r if self.rss_max is None else max(self.rss_max, r)
            self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()

    def report(self) -> Dict[str, Any]:
        gib = 1048576.0
        hwm = _status_kb("VmHWM")
        rss = max(x for x in (hwm, self.rss_max, 0) if x is not None)
        out: Dict[str, Any] = {"peak_rss_gib": round(rss / gib, 2) if rss else None}
        if self.total and self.avail_min is not None:
            out["host_used_peak_gib"] = round((self.total - self.avail_min) / gib, 2)
            out["host_avail_min_gib"] = round(self.avail_min / gib, 2)
        if self.avail_start is not None and self.avail_min is not None:
            out["host_used_delta_gib"] = round((self.avail_start - self.avail_min) / gib, 2)
        return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class FakeOOM(RuntimeError):
    """What the fake backend raises past FAKE_QLORA_MAX (named like torch's, so is_oom sees it)."""


FakeOOM.__name__ = "OutOfMemoryError"


class FakeBackend:
    """GPU-free stand-in with the UnslothBackend interface: sleeps a little, runs out of memory above
    the environment's FAKE_QLORA_MAX."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.limit = int(os.environ.get("FAKE_QLORA_MAX", "32768"))

    def load(self, max_seq: int, revision: Optional[str]) -> None:
        time.sleep(0.2)

    def zero_grad(self) -> None:
        pass

    def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
        if len(ids) > self.limit:
            raise FakeOOM(f"CUDA out of memory (fake: {len(ids)} > {self.limit})")
        time.sleep(len(ids) / 400000.0)
        return 1.0

    def step(self, lr: float) -> float:
        return 0.5

    def peak_memory_gb(self) -> float:
        return round(4.0 + self.cfg.max_seq / 16384.0, 2)


def torch_peaks() -> Dict[str, Any]:
    try:
        import torch

        return {"peak_vram_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    except Exception:  # noqa: BLE001 - no torch (fake) or no CUDA
        return {}


def sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def offload_evidence() -> Dict[str, Any]:
    """Whether Unsloth's offloaded gradient checkpointing is in place and holding host buffers."""
    out: Dict[str, Any] = {}
    try:
        import torch.utils.checkpoint as tuc

        out["checkpoint_fn"] = getattr(tuc.checkpoint, "__name__", None)
    except Exception:  # noqa: BLE001
        return out
    try:
        import unsloth_zoo.gradient_checkpointing as gc

        bufs = getattr(gc, "CPU_BUFFERS", None)
        out["host_buffers"] = len(bufs) if isinstance(bufs, (list, tuple)) else None
        pinned = getattr(gc, "GC_PINNED_BYTES", None)
        out["pinned_gib"] = round(pinned / 2**30, 2) if isinstance(pinned, (int, float)) else None
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(a: argparse.Namespace) -> int:
    T = load_trainer(Path(a.trainer).expanduser())
    out = Path(a.out).expanduser()
    L = int(a.seq_len)
    doc: Dict[str, Any] = {"schema": SCHEMA, "seq_len": L, "status": "running", "started": now_iso(),
                           "steps": a.steps, "warmup_steps": a.warmup_steps, "fake": bool(a.fake_backend)}
    data = Path(a.data).expanduser()
    # Q6b: the same two approvals as the trainer, and a tagged output
    teacher_max = T.builder_teacher_is_max(data)
    flags: List[bool] = []
    rows = sample_rows(iter_jsonl(data), SAMPLE_ROWS, f"{PACK_SEED}|{L}",
                       on_row=lambda r: flags.append(T.is_max_derived(r, teacher_max)))
    rec = T.dataset_record(data)
    g = T.guard(flags, bool(a.approved), rec.get("approval"))
    doc["provenance"] = {"rows": g.total, "max_derived_rows": g.max_derived,
                         "dataset_approval": bool(g.dataset_approval), "operator_flag": bool(a.approved)}
    if not g.allowed:
        print(g.message(), file=sys.stderr)
        return 2
    if g.data_max_derived:
        pol = T._load_policy()
        if not pol.has_tag(out.parent.name) and not pol.has_tag(out.name):
            print(f"refused: the data is Max-derived; {out} must sit under a name tagged {pol.TAG}", file=sys.stderr)
            return 2
        doc["tags"] = ["max-derived"]
        doc["private"] = True
    need = int(a.warmup_steps) + int(a.steps)
    enc = T.hf_encoder(Path(a.tokenizer_dir).expanduser())

    def tokenized() -> Iterator[Tuple[Any, Any]]:
        for r in rows:
            T.check_row(r, f"row {r.get('id')}")
            ids, spans = T.tokenize_row(r, enc)
            yield list(ids), spans

    t0 = time.monotonic()
    seqs = pack(tokenized(), L, need)
    doc["packing"] = {"sequences": len(seqs), "tokens_each": L, "rows_sampled": len(rows),
                      "loss_positions": [loss_positions(s) for _, s in seqs], "seconds": round(time.monotonic() - t0, 1)}
    cfg = T.TrainConfig(base_model=str(Path(a.base_model).expanduser()), base_revision=a.base_revision,
                        max_seq=L, ladder=(L,), grad_accum=1, epochs=1)
    errs = cfg.validate()
    if errs:
        print("config: " + "; ".join(errs), file=sys.stderr)
        return 2
    doc["config"] = {"r": cfg.r, "alpha": cfg.alpha, "target_modules": list(cfg.target_modules), "optim": cfg.optim,
                     "gradient_checkpointing": cfg.gradient_checkpointing, "lr": cfg.lr, "load": cfg.load,
                     "max_seq": cfg.max_seq, "micro_batch": cfg.micro_batch, "grad_accum": cfg.grad_accum}
    if a.dry_run:
        doc["status"] = "planned"
        write_json(out, doc)
        print(json.dumps({k: doc[k] for k in ("seq_len", "provenance", "packing", "config")}, indent=1))
        return 0
    write_json(out, doc)
    mem = MemWatch()
    mem.start()
    backend = FakeBackend(cfg) if a.fake_backend else T.UnslothBackend(cfg)
    phase = "load"
    steps: List[Dict[str, Any]] = []
    rc = 0
    try:
        t_load = time.monotonic()
        backend.load(L, a.base_revision)
        doc["load_s"] = round(time.monotonic() - t_load, 1)
        for i, (ids, spans) in enumerate(seqs):
            warm = i < int(a.warmup_steps)
            phase = f"{'warm-up' if warm else 'timed'} step {i + 1}"
            sync()
            t = time.monotonic()
            backend.zero_grad()
            loss = backend.forward_backward(ids, spans, 1.0 / max(1, loss_positions(spans)))
            norm = backend.step(cfg.lr)
            sync()
            dt = time.monotonic() - t
            steps.append({"step": i + 1, "warmup": warm, "seconds": round(dt, 2), "tok_s": round(L / dt, 1),
                          "loss": round(float(loss), 4), "grad_norm": None if norm is None else round(float(norm), 4)})
            if i == 0:
                doc["offload"] = offload_evidence()
            doc["steps_done"] = steps
            doc.update(torch_peaks())
            doc.update(mem.report())
            write_json(out, doc)
            print(f"step {i + 1}/{len(seqs)} {'(warm-up) ' if warm else ''}{dt:.1f}s {L / dt:.0f} tok/s loss {loss:.3f}",
                  flush=True)
        timed = [s for s in steps if not s["warmup"]]
        doc["status"] = "ok"
        doc["tok_s"] = round(statistics.median(s["tok_s"] for s in timed), 1)
        doc["tok_s_mean"] = round(L * len(timed) / sum(s["seconds"] for s in timed), 1)
        doc["step_s_median"] = round(statistics.median(s["seconds"] for s in timed), 2)
    except BaseException as exc:  # noqa: BLE001 - classified and recorded
        if T.is_oom(exc):
            doc["status"] = "oom"
            doc["detail"] = f"CUDA out of memory at {phase}"
            rc = 3
        elif isinstance(exc, KeyboardInterrupt):
            doc["status"] = "failed"
            doc["detail"] = f"interrupted at {phase}"
            rc = 1
        else:
            doc["status"] = "failed"
            doc["detail"] = f"{type(exc).__name__} at {phase}: {str(exc)[:200]}"
            rc = 1
    finally:
        mem.stop()
        doc.update(torch_peaks())
        doc.update(mem.report())
        if a.fake_backend and hasattr(backend, "peak_memory_gb"):
            doc.setdefault("peak_vram_reserved_gib", backend.peak_memory_gb())
        doc["steps_done"] = steps
        doc["finished"] = now_iso()
        write_json(out, doc)
    print(f"{L}: {doc['status']} {doc.get('tok_s', '')} {doc.get('detail', '')}".rstrip(), flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--trainer", required=True, help="the staged train_qlora.py")
    r.add_argument("--data", required=True)
    r.add_argument("--tokenizer-dir", required=True)
    r.add_argument("--base-model", required=True)
    r.add_argument("--base-revision", default=None)
    r.add_argument("--seq-len", type=int, required=True)
    r.add_argument("--steps", type=int, default=5)
    r.add_argument("--warmup-steps", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument(APPROVAL_FLAG, dest="approved", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="no torch: guard, packing and plan only")
    r.add_argument("--fake-backend", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.seq_len < 1024 or a.steps < 1 or a.warmup_steps < 0:
        print("usage: --seq-len >= 1024, --steps >= 1, --warmup-steps >= 0", file=sys.stderr)
        return 2
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
