#!/usr/bin/env python3
"""Merge a PEFT LoRA into a safetensors checkpoint one tensor at a time.

quantize.py's default merge (PEFT ``merge_and_unload``) holds the whole base in
RAM: about 19 GB for Qwen3.5-9B in bf16, plus what saving copies. The 3090 Ti
host keeps at least 8 GB free for its services and has about 13-20 GB
available, so ``quantize.py run --merge-mode stream`` merges with this instead:
the same arithmetic, one weight matrix at a time, reading the base's
safetensors files and writing the merged checkpoint shard by shard. Resident
memory is the adapter (fp32) plus one weight and its delta: under 1 GB for the
9B, whatever the model's size.

What it reproduces (PEFT 0.21.2, the trainer's version, as quantize.py drives it)
---------------------------------------------------------------------------------
* The base as ``from_pretrained(dtype=bf16)`` loads it: every floating tensor
  in the target dtype (the official Qwen3.5-9B stores 48 tensors,
  ``linear_attn.A_log`` and ``linear_attn.norm.weight``, in F32; transformers
  has no keep-in-fp32 rule for them and casts them, and the trainer's loader
  did too), and ``mtp.*`` left out (Qwen3.5's
  ``_keys_to_ignore_on_load_unexpected``; the converter runs with ``--no-mtp``
  either way, and a tuned trunk would leave the base's MTP head stale).
* The adapter as ``PeftModel.from_pretrained`` holds it: lora_A and lora_B in
  fp32 (``autocast_adapter_dtype``), whatever the file stores.
* ``Linear.merge(safe_merge=True)``: ``delta = (B @ A) * scaling`` in fp32 with
  ``scaling = lora_alpha / r`` (``/ sqrt(r)`` with rslora), then
  ``W += delta.to(W.dtype)``; a non-finite result is refused.
* ``config.json`` is the base's own file (save_pretrained would rewrite it).

Checked on the 3090 Ti host on 2026-10-31 against PEFT 0.21.2's own path
(``PeftModel.from_pretrained`` + ``merge_and_unload(safe_merge=True)``) on 30
modules of the 9B smoke adapter, three of each of the ten targeted kinds:
bit-identical. The whole 9B merge took 1.5-7 min (the scope's memory cap
throttles writeback) at 1.3 GB peak RSS.

Refused, like quantize.py's PEFT path, and before anything is written: not
LoRA; DoRA, lora_bias, fan_in_fan_out, modules_to_save, trainable tokens, layer
replication, rank or alpha patterns; an embedding LoRA; adapter keys outside
``base_model.model.``; a LoRA module with no base weight of the shape its A
and B imply (PEFT would not attach it, and the merge would silently skip it);
a zero lora_B (never trained, or keys that did not load); a quantized base.

The arithmetic needs torch; the rest (headers, plan, copies) is stdlib, so the
tests drive it with a pure-Python stand-in (``math=``).
"""

from __future__ import annotations

import json
import math as _math
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ADAPTER_PREFIX = "base_model.model."
INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
DROP_PREFIXES = ("mtp.",)  # transformers ignores these on load (Qwen3.5)
COPY_CHUNK = 64 * 1024 * 1024

# safetensors dtype -> bytes per element
DTYPE_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
              "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
FLOATING = ("F64", "F32", "F16", "BF16")
TARGET = {"bf16": "BF16", "fp16": "F16"}

# adapter_config.json fields whose non-default value this merge does not reproduce
UNSUPPORTED = {"use_dora": False, "lora_bias": False, "fan_in_fan_out": False,
               "modules_to_save": None, "trainable_token_indices": None,
               "layer_replication": None, "rank_pattern": {}, "alpha_pattern": {}}


class MergeError(RuntimeError):
    """An adapter or base this merge must not (or cannot) merge."""


@dataclass
class Entry:
    """One tensor of a safetensors file: where its bytes are."""
    name: str
    dtype: str
    shape: list[int]
    file: Path
    start: int  # absolute file offsets
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class Lora:
    module: str  # the base module, e.g. model.language_model.layers.0.mlp.up_proj
    a: Entry
    b: Entry


@dataclass
class Plan:
    scaling: float
    rank: int
    alpha: float
    loras: dict[str, Lora]  # base weight name -> its LoRA
    shards: dict[str, list[Entry]]  # output shard name -> base tensors kept, in file order
    dropped: list[str] = field(default_factory=list)
    out_dtype: str = "BF16"


def read_header(path: Path) -> tuple[dict[str, Entry], dict[str, Any]]:
    """The tensors of a .safetensors file (8-byte length + JSON header)."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) != 8:
            raise MergeError(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", head)
        meta = json.loads(fh.read(n))
    base = 8 + n
    extra = meta.pop("__metadata__", None) or {}
    out = {}
    for name, v in meta.items():
        s, e = v["data_offsets"]
        out[name] = Entry(name, v["dtype"], list(v["shape"]), path, base + s, base + e)
    return out, extra


def base_files(base: Path) -> list[Path]:
    idx = base / INDEX
    if idx.is_file():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        return [base / f for f in sorted(set(wm.values()))]
    if (base / SINGLE).is_file():
        return [base / SINGLE]
    raise MergeError(f"no {INDEX} or {SINGLE} in {base}")


def scaling_of(cfg: dict[str, Any], rank: int) -> float:
    alpha = float(cfg.get("lora_alpha", 8))
    return alpha / _math.sqrt(rank) if cfg.get("use_rslora") else alpha / rank


def check_config(cfg: dict[str, Any]) -> None:
    if str(cfg.get("peft_type", "LORA")).upper() != "LORA":
        raise MergeError(f"adapter is {cfg.get('peft_type')}, not LoRA")
    bad = [k for k, ok in UNSUPPORTED.items() if cfg.get(k, ok) not in (ok, None, {}, [], False)]
    if bad:
        raise MergeError(f"adapter_config sets {bad}: the stream merge reproduces vanilla LoRA "
                         "only; use --merge-mode peft")


def make_plan(base: Path, adapter: Path, *, dtype: str = "bf16",
              drop_prefixes: tuple[str, ...] = DROP_PREFIXES) -> Plan:
    if dtype not in TARGET:
        raise MergeError(f"dtype {dtype}: one of {sorted(TARGET)}")
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    check_config(cfg)
    bcfg_path = base / "config.json"
    bcfg = json.loads(bcfg_path.read_text(encoding="utf-8")) if bcfg_path.is_file() else {}
    if bcfg.get("quantization_config"):
        raise MergeError(f"{base} is a quantized checkpoint; merge into the 16-bit base")
    atensors, _ = read_header(adapter / "adapter_model.safetensors")
    pairs: dict[str, dict[str, Entry]] = {}
    for k, e in atensors.items():
        if ".lora_embedding_" in k or ".lora_magnitude_vector" in k:
            raise MergeError(f"{k}: embedding LoRA / DoRA weights are not merged here")
        m = re.fullmatch(r"(.+)\.lora_(A|B)(?:\.default)?\.weight", k)
        if not m:
            raise MergeError(f"{k}: not a LoRA A/B weight (modules_to_save? use --merge-mode peft)")
        if not m.group(1).startswith(ADAPTER_PREFIX):
            raise MergeError(f"{k}: adapter keys are expected under {ADAPTER_PREFIX!r}")
        pairs.setdefault(m.group(1)[len(ADAPTER_PREFIX):], {})[m.group(2)] = e
    if not pairs:
        raise MergeError(f"{adapter} holds no LoRA weights")
    files = base_files(base)
    btensors: dict[str, Entry] = {}
    order: dict[str, list[str]] = {}
    for f in files:
        t, _ = read_header(f)
        dup = set(t) & set(btensors)
        if dup:
            raise MergeError(f"{sorted(dup)[:3]} in more than one base shard")
        btensors.update(t)
        order[f.name] = sorted(t, key=lambda n: t[n].start)
    loras: dict[str, Lora] = {}
    ranks = set()
    problems = []
    for mod, ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            problems.append(f"{mod}: has lora_{''.join(sorted(ab))} only")
            continue
        a, b = ab["A"], ab["B"]
        w = btensors.get(mod + ".weight")
        if (w is None or len(w.shape) != 2 or len(a.shape) != 2 or len(b.shape) != 2
                or a.shape[0] != b.shape[1] or w.shape != [b.shape[0], a.shape[1]]):
            problems.append(f"{mod}: no base weight of shape [{b.shape[0]}, {a.shape[1]}]"
                            + (f" (base has {w.shape})" if w is not None else ""))
            continue
        if w.dtype not in FLOATING or a.dtype not in FLOATING or b.dtype not in FLOATING:
            problems.append(f"{mod}: dtypes {w.dtype}/{a.dtype}/{b.dtype} are not floating")
            continue
        ranks.add(a.shape[0])
        loras[w.name] = Lora(mod, a, b)
    if problems:
        raise MergeError(f"{len(problems)} of {len(pairs)} LoRA modules would not attach to "
                         f"{base} (merging would silently drop them): {problems[:3]}")
    if len(ranks) != 1:
        raise MergeError(f"LoRA ranks differ across modules {sorted(ranks)} (a rank pattern?)")
    rank = ranks.pop()
    if int(cfg.get("r", rank)) != rank:
        raise MergeError(f"adapter_config r={cfg.get('r')} but the weights have rank {rank}")
    shards: dict[str, list[Entry]] = {}
    dropped = []
    for fname, names in order.items():
        keep = []
        for n in names:
            if n.startswith(drop_prefixes):
                dropped.append(n)
            else:
                keep.append(btensors[n])
        shards[fname] = keep
    return Plan(scaling=scaling_of(cfg, rank), rank=rank, alpha=float(cfg.get("lora_alpha", 8)),
                loras=loras, shards=shards, dropped=sorted(dropped), out_dtype=TARGET[dtype])


def read_bytes(e: Entry) -> bytearray:
    buf = bytearray(e.nbytes)
    with open(e.file, "rb") as fh:
        fh.seek(e.start)
        view = memoryview(buf)
        got = 0
        while got < e.nbytes:
            n = fh.readinto(view[got:])
            if not n:
                raise MergeError(f"{e.file}: short read for {e.name}")
            got += n
    return buf


def copy_bytes(e: Entry, out: Any) -> None:
    with open(e.file, "rb") as fh:
        fh.seek(e.start)
        left = e.nbytes
        while left:
            chunk = fh.read(min(COPY_CHUNK, left))
            if not chunk:
                raise MergeError(f"{e.file}: short read for {e.name}")
            out.write(chunk)
            left -= len(chunk)


# ---------------------------------------------------------------------------
# The arithmetic (torch). ``Math`` is the seam the tests replace.
# ---------------------------------------------------------------------------


class TorchMath:
    """PEFT's arithmetic on CPU tensors (see the module docstring)."""

    def __init__(self) -> None:
        import torch

        self.torch = torch
        self.dt = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
                   "F64": torch.float64}

    def _t(self, buf: bytearray, dtype: str, shape: list[int]) -> Any:
        return self.torch.frombuffer(buf, dtype=self.dt[dtype]).reshape(shape)

    def _bytes(self, t: Any) -> memoryview:
        return memoryview(t.contiguous().reshape(-1).view(dtype=self.torch.uint8).numpy())

    def cast(self, buf: bytearray, dtype: str, shape: list[int], out_dtype: str) -> memoryview:
        return self._bytes(self._t(buf, dtype, shape).to(self.dt[out_dtype]))

    def b_nonzero(self, buf: bytearray, dtype: str, shape: list[int]) -> bool:
        return bool(self._t(buf, dtype, shape).abs().max().item() > 0)

    def merge(self, w: bytearray, w_dtype: str, w_shape: list[int], a: bytearray, a_dtype: str,
              a_shape: list[int], b: bytearray, b_dtype: str, b_shape: list[int],
              scaling: float, out_dtype: str) -> memoryview:
        torch = self.torch
        weight = self._t(w, w_dtype, w_shape).to(self.dt[out_dtype])  # from_pretrained(dtype=)
        la = self._t(a, a_dtype, a_shape).to(torch.float32)  # autocast_adapter_dtype
        lb = self._t(b, b_dtype, b_shape).to(torch.float32)
        delta = (lb @ la) * scaling  # get_delta_weight (fan_in_fan_out False)
        merged = weight.clone()  # merge(safe_merge=True)
        merged += delta.to(merged.dtype)
        if not bool(torch.isfinite(merged).all()):
            raise MergeError("non-finite values in a merged weight: the adapter is broken")
        return self._bytes(merged)


def _write_shard(path: Path, plan: Plan, entries: list[Entry], m: Any,
                 progress: Callable[[str], None]) -> dict[str, Any]:
    """One output shard: header first (sizes are known), then each tensor."""
    size = DTYPE_SIZE
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    off = 0
    layout = []
    for e in entries:
        dt = plan.out_dtype if e.dtype in FLOATING else e.dtype
        n = _math.prod(e.shape) * size[dt] if e.shape else size[dt]
        if e.dtype == dt and n != e.nbytes:
            raise MergeError(f"{e.name}: {e.nbytes} bytes for shape {e.shape} {e.dtype}")
        header[e.name] = {"dtype": dt, "shape": e.shape, "data_offsets": [off, off + n]}
        layout.append((e, dt, n))
        off += n
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    stats = {"tensors": len(entries), "merged": 0, "cast": 0, "copied": 0, "bytes": off}
    tmp = path.with_name(path.name + ".partial")
    try:
        _write_tensors(tmp, raw, layout, plan, m, stats)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.rename(path)
    progress(f"  {path.name}: {stats['merged']} merged, {stats['cast']} cast, "
             f"{stats['copied']} copied, {off:,} B")
    return stats


def _write_tensors(tmp: Path, raw: bytes, layout: list[tuple[Entry, str, int]], plan: Plan,
                   m: Any, stats: dict[str, int]) -> None:
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for e, dt, n in layout:
            lo = plan.loras.get(e.name)
            if lo is not None:
                data = m.merge(read_bytes(e), e.dtype, e.shape, read_bytes(lo.a), lo.a.dtype,
                               lo.a.shape, read_bytes(lo.b), lo.b.dtype, lo.b.shape,
                               plan.scaling, dt)
                stats["merged"] += 1
            elif e.dtype != dt:
                data = m.cast(read_bytes(e), e.dtype, e.shape, dt)
                stats["cast"] += 1
            else:
                copy_bytes(e, fh)
                stats["copied"] += 1
                continue
            if len(data) != n:
                raise MergeError(f"{e.name}: wrote {len(data)} bytes, expected {n}")
            fh.write(data)
            del data
        fh.flush()
        os.fsync(fh.fileno())


def stream_merge(base: Path, adapter: Path, out: Path, *, dtype: str = "bf16",
                 math: Any = None, progress: Callable[[str], None] = print) -> dict[str, Any]:
    """Write the merged weights (shards + index) into ``out`` (an existing
    empty directory). Returns what quantize.py records as merge-info."""
    plan = make_plan(base, adapter, dtype=dtype)
    m = math if math is not None else TorchMath()
    zero = [lo.module for lo in plan.loras.values()
            if not m.b_nonzero(read_bytes(lo.b), lo.b.dtype, lo.b.shape)]
    if zero:
        raise MergeError(f"{len(zero)} of {len(plan.loras)} LoRA B matrices are zero (e.g. "
                         f"{zero[0]}): their weights never trained")
    weight_map: dict[str, str] = {}
    totals = {"merged": 0, "cast": 0, "copied": 0, "bytes": 0}
    for fname, entries in plan.shards.items():
        if not entries:
            continue
        st = _write_shard(out / fname, plan, entries, m, progress)
        for k in totals:
            totals[k] += st[k]
        weight_map.update({e.name: fname for e in entries})
    merged_names = [n for n in plan.loras if n in weight_map]
    if len(merged_names) != len(plan.loras):
        raise MergeError("a LoRA target was dropped from the output")
    index = {"metadata": {"total_size": totals["bytes"]},
             "weight_map": dict(sorted(weight_map.items()))}
    (out / INDEX).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    cast_from = sorted({e.dtype for es in plan.shards.values() for e in es
                        if e.dtype in FLOATING and e.dtype != plan.out_dtype})
    return {
        "merge_mode": "stream", "dtype": dtype, "scaling": plan.scaling, "rank": plan.rank,
        "lora_alpha": plan.alpha, "lora_modules": len(plan.loras), "attached": len(plan.loras),
        "b_nonzero": len(plan.loras), "tensors_out": len(weight_map),
        "tensors_merged": totals["merged"], "tensors_cast": totals["cast"],
        "tensors_cast_from": cast_from, "tensors_copied": totals["copied"],
        "dropped": {"count": len(plan.dropped), "prefixes": list(DROP_PREFIXES),
                    "names": plan.dropped},
        "bytes_out": totals["bytes"],
    }
