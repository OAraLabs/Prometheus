#!/usr/bin/env python3
"""Phase 5: Unsloth QLoRA SFT of the student on the Phase 3 dataset.

    # 1. Validate data, masks and config. No torch, no GPU; prints two examples
    #    with the masked and trained spans marked, writes the manifest.
    python3 scripts/distill/train_qlora.py --data ~/distill/phase3/train.jsonl \\
        --tokenizer-dir ~/distill/models/qwen3.5-9b-tokenizer \\
        --out data/distill/runs/9b-r32 --dry-run
    # 2. Smoke: 200 examples, 50 steps, loss must drop, checkpoint round trip
    #    (the training half of the smoke gate; the eval pipeline runs separately).
    #    The base comes from a staged, pinned download (see "Base model" below).
    HF_HUB_OFFLINE=1 python3 scripts/distill/train_qlora.py ... \\
        --base-model ~/distill/models/qwen3.5-9b-base \\
        --base-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \\
        --out data/distill/runs/smoke --smoke
    # 3. Full run (detached: scripts/distill/launch_detached.sh), resumable.
    python3 scripts/distill/train_qlora.py ... --out data/distill/runs/9b-r32 [--resume] \\
        [--stop-at 2026-11-01T06:45-05:00]

Base model
----------
A real run (anything but ``--dry-run``) loads ``--base-model`` from a local
directory: the full-precision checkpoint staged by a pinned, sha256-checked
download that keeps the host's disk floor (Unsloth quantizes it to 4-bit on
load; no 4-bit Qwen3.5-9B checkpoint is published). Pass its commit as
``--base-revision`` so the manifest records it. A hub id is refused unless
``HF_HUB_OFFLINE=1`` (load from the HF cache only, fail when absent) or
``--allow-hub-download`` is given: from_pretrained would otherwise pull ~19 GB
into the HF cache with no free-space check.

Nothing here trains on teacher (Max-derived) data without the operator: a real
run (anything but ``--dry-run``, smoke included) on data with a Max-derived row
refuses (exit 3) unless BOTH approvals are there (DISTILL-OPEN-QUESTIONS #6,
Q6b ruling): ``--i-have-operator-approval-for-max-derived-data`` on this
command line, and the dataset's own recorded approval -- build.py's
``--allow-max-derived --max-derived-approval "<ref>"``, read from the
``report.json`` beside ``train.jsonl`` (``data_policy.max_derived_allow``), else
from the ``dataset-manifest.json`` there (a build whose ``--report`` went
elsewhere; only while ``train.jsonl`` still has the sha256 it lists), or from a
pre-tokenized directory's ``meta.json`` (copied from that report). The
flag alone, or a dataset built without the allow, is refused. The guard fails
closed: a row counts as Max-derived unless something shows otherwise. A
provenance field (``source``, ``model``, also under ``_meta`` / ``meta``)
naming Max always counts. Otherwise the builder's teacher decides: build.py
names the one model whose rows train/val hold in its report
(``settings.teacher_model``); with no such report, only a builder-written
``max_derived: false`` on the row clears it.

A run on Max-derived data is private (``artifact_policy.py``): its ``--out``
must carry the tag in its name (``runs/9b-r32-max-derived``; otherwise exit 2),
its adapter directory is ``<out>/adapter-max-derived`` (not ``<out>/adapter``),
and its run manifest carries ``max_derived: true``, ``tags: ["max-derived"]``,
``private: true`` and both approvals (``approvals``). Every checkpoint carries
its own ``manifest.json`` with the same fields (and the run's data), so a
checkpoint copied out of the run directory (a window close leaves the weights
only there) still says what it is. ``publish_guard.py`` refuses them all;
``modal_train.py`` never rents the data.

Data
----
``train.jsonl`` from ``build.py`` (rows carry ``text``, ``loss_spans_char``,
``loss_spans_tok``, ``n_tokens``, ``tokenize: "segments"``, ``source``,
``messages``/``tools``/``chat_template_kwargs``), or a pre-tokenized directory
from ``modal_train.py pretokenize``. Rows are tokenized the way ``render.py``
counted them: each span boundary splits the text into segments, each segment
is encoded alone, and the ids are concatenated. The result must reproduce the
row's ``n_tokens`` and ``loss_spans_tok`` exactly, or the run refuses (the
tokenizer is not the one the data was rendered with). Labels are the ids
inside the loss spans and -100 everywhere else: only the teacher's turn
carries loss.

Training
--------
QLoRA r=32 / alpha=32 on the language model's attention, gated-deltanet and
MLP projections (``TARGET_MODULES``), lr 1e-4 with linear warmup then
cosine to zero, 2 epochs, micro-batch 1 (no padding) with gradient
accumulation, "unsloth" gradient checkpointing, 8-bit AdamW, loss normalised
over the accumulation group's loss tokens. The sample order is a seeded
permutation of the row ids per epoch (the *plan*); the position in the plan is
part of every checkpoint.

* **Sequence step-down.** Before training, one forward+backward on the
  longest example probes memory. A CUDA OOM there or during training records
  a step-down (16,384 -> 12,288 -> 8,192) in ``run_state.json`` and the
  process re-executes itself with ``--resume``: a fresh CUDA context, the
  latest checkpoint, and the data re-fitted to the new length. Rows over it are
  re-fitted with ``render.fit_to_budget`` (oldest context dropped, system
  prompt, tools, last query and the learned turn kept) when ``--tokenizer-dir``
  holds the chat template, else dropped; every step-down is recorded.
* **Checkpoints** every ``--save-every`` steps (and on SIGTERM, at
  ``--stop-at``, and at the end): adapter, optimizer, RNG states and
  ``state.json`` (step, plan position, tokens seen, a fingerprint of the
  weights), written to a ``.partial`` directory and renamed once complete.
  ``--resume`` takes the newest complete one; the plan, data hash and config
  hash must match the run's, and the restored weights must match the
  fingerprint (exit 1 otherwise: a restore that drops weights would train on
  from fresh LoRA weights).
* **Stop.** SIGTERM (a ``setup apply business`` stops ``distill-*`` units) or
  ``--stop-at`` abandons the accumulation group in progress and checkpoints
  at the last completed step, so a window close loses at most one group.

Exit: 0 done (or stopped cleanly), 1 training failed or the smoke check
failed, 2 usage or data error, 3 refused (Max-derived data without both approvals).
"""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
import json
import logging
import math
import mmap
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

REPO = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "distill-train-manifest/1"
RUN_STATE_SCHEMA = "distill-train-run-state/1"
PRETOK_SCHEMA = "distill-pretok/1"
STUDENT = "unsloth/Qwen3.5-9B"
SEQ_LADDER = (16384, 12288, 8192)
APPROVAL_FLAG = "--i-have-operator-approval-for-max-derived-data"
RUN_STATE = "run_state.json"
CKPT_PREFIX = "checkpoint-"
CKPT_MANIFEST = "manifest.json"  # a checkpoint's own provenance (Q6b), the run manifest's schema
PARTIAL = ".partial"
COMPLETE = "COMPLETE"
IGNORE = -100
# The language model's linear layers: attention (q/k/v/o, the full-attention
# layers), the gated-deltanet token mixer (in_proj_qkv, in_proj_z, out_proj: the
# linear-attention layers, 24 of the 9B's 32 and 48 of the 27B's 64) and the MLP.
# Not targeted: the deltanet's tiny in_proj_a/in_proj_b gates, the embeddings and
# head, and the vision tower (its layers are qkv, proj, linear_fc1/2: no match).
# feasibility_27b.TARGET_MODULES is the same list (a test keeps them equal).
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                  "gate_proj", "up_proj", "down_proj")
# Teacher provenance: "max-golden", "max-gym-<date>", "qwen3.8-max" ... A word
# "max", not a substring of another word.
MAX_RE = re.compile(r"(?<![a-z0-9])max(?![a-z0-9])", re.I)
# build.py's report, written beside train.jsonl; settings.teacher_model names
# the only model whose rows the build kept.
BUILD_REPORT = "report.json"
# build.py's dataset manifest (artifact_policy.DATASET_MANIFEST), always written
# into its --out: the record the trainer falls back to when --report went elsewhere.
DATASET_MANIFEST = "dataset-manifest.json"
ADAPTER_DIR = "adapter"  # <out>/adapter; <out>/adapter-max-derived for Max-derived data
VERSIONED = ("torch", "transformers", "unsloth", "unsloth_zoo", "peft", "trl", "bitsandbytes",
             "accelerate", "tokenizers", "huggingface_hub", "safetensors")

log = logging.getLogger("distill.train")


class DataError(ValueError):
    """The data cannot be trained on as given."""


class UsageError(ValueError):
    """The command line or the run directory does not allow this."""


class RestoreError(RuntimeError):
    """A checkpoint restore did not bring back the weights it saved."""


class StepDown(Exception):
    """A CUDA OOM moved the run to a shorter sequence length (already persisted)."""

    def __init__(self, old: int, new: int, step: int) -> None:
        super().__init__(f"CUDA OOM at max_seq {old}: stepping down to {new} (step {step})")
        self.old, self.new, self.step = old, new, step


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    base_model: str = STUDENT
    base_revision: str | None = None
    load: str = "4bit"  # "4bit" = QLoRA (runbook); "16bit" = LoRA on bf16 weights
    r: int = 32
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = TARGET_MODULES
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    epochs: int = 2
    max_seq: int = SEQ_LADDER[0]
    ladder: tuple[int, ...] = SEQ_LADDER
    micro_batch: int = 1
    grad_accum: int = 8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_8bit"
    gradient_checkpointing: str = "unsloth"
    save_every: int = 200
    keep_checkpoints: int = 3
    seed: int = 3407
    log_every: int = 10
    max_steps: int | None = None  # smoke only

    # Fields that change how a run trains. Operational knobs (logging,
    # checkpoint retention) and max_seq (moved by step-down, recorded in
    # run_state) are left out so a resume can still change them.
    _HASHED = ("base_model", "base_revision", "load", "r", "alpha", "dropout", "target_modules",
               "lr", "warmup_ratio", "epochs", "ladder", "micro_batch", "grad_accum",
               "weight_decay", "max_grad_norm", "optim", "gradient_checkpointing", "seed",
               "max_steps")

    def validate(self) -> list[str]:
        errs = []
        if self.load not in ("4bit", "16bit"):
            errs.append(f"load must be 4bit or 16bit, not {self.load!r}")
        if self.r <= 0 or self.alpha <= 0:
            errs.append("LoRA r and alpha must be positive")
        if not 0 <= self.dropout < 1:
            errs.append("dropout must be in [0, 1)")
        if not 0 < self.lr <= 1e-2:
            errs.append(f"lr {self.lr} is outside (0, 1e-2]")
        if not 0 <= self.warmup_ratio < 0.5:
            errs.append("warmup_ratio must be in [0, 0.5)")
        if self.epochs < 1:
            errs.append("epochs must be >= 1")
        if list(self.ladder) != sorted(set(self.ladder), reverse=True) or not self.ladder:
            errs.append("ladder must be strictly decreasing")
        if self.max_seq not in self.ladder:
            errs.append(f"max_seq {self.max_seq} is not on the ladder {list(self.ladder)}")
        if self.micro_batch != 1:
            errs.append("micro_batch must be 1 (variable-length rows, no padding)")
        if self.grad_accum < 1:
            errs.append("grad_accum must be >= 1")
        if self.save_every < 1 or self.keep_checkpoints < 1:
            errs.append("save_every and keep_checkpoints must be >= 1")
        if self.optim not in ("adamw_8bit", "adamw_torch"):
            errs.append(f"optim must be adamw_8bit or adamw_torch, not {self.optim!r}")
        if self.max_steps is not None and self.max_steps < 1:
            errs.append("max_steps must be >= 1")
        return errs

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        d["ladder"] = list(self.ladder)
        return d

    def sha(self) -> str:
        d = self.as_dict()
        return sha256_text(json.dumps({k: d[k] for k in self._HASHED}, sort_keys=True))


def smoke_config(cfg: TrainConfig) -> TrainConfig:
    """200 examples x 1 epoch / 4 per step = 50 steps; a checkpoint halfway."""
    d = {k: v for k, v in asdict(cfg).items()}
    d.update(epochs=1, grad_accum=4, save_every=25, max_steps=50, log_every=5)
    d["target_modules"] = tuple(d["target_modules"])
    d["ladder"] = tuple(d["ladder"])
    return TrainConfig(**d)


SMOKE_EXAMPLES = 200
# Steps the smoke trains past its round-trip checkpoint, so the restore has
# weights to undo.
SMOKE_MOVE_STEPS = 3


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Learning rate for the update with 0-based index *step*: linear warmup
    over *warmup* updates (the first one already non-zero), then cosine to 0."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (step - warmup) / span))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def next_seq(current: int, ladder: Iterable[int] = SEQ_LADDER) -> int | None:
    """The next shorter length on the ladder, or None at the bottom."""
    shorter = [x for x in ladder if x < current]
    return max(shorter) if shorter else None


def is_oom(exc: BaseException) -> bool:
    """A CUDA out-of-memory error, recognised without importing torch."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    msg = str(exc)
    return "CUDA out of memory" in msg or "CUBLAS_STATUS_ALLOC_FAILED" in msg


# ---------------------------------------------------------------------------
# Hashing, rows, provenance guard
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with Path(p).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise DataError(f"{path}:{n}: not JSON ({exc})") from None
    return rows


def provenance(row: dict[str, Any]) -> list[str]:
    out = []
    for holder in (row, row.get("_meta"), row.get("meta")):
        if isinstance(holder, dict):
            for key in ("source", "model"):
                v = holder.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
    return out


def row_source(row: dict[str, Any]) -> str:
    p = provenance(row)
    return p[0] if p else "unknown"


def says_not_max(row: dict[str, Any]) -> bool:
    """The row's builder marked it ``max_derived: false``."""
    return any(isinstance(h, dict) and h.get("max_derived") is False
               for h in (row, row.get("_meta"), row.get("meta")))


def is_max_derived(row: dict[str, Any], teacher_max: bool | None = None) -> bool:
    """True unless something shows the row is not the teacher's (fail closed).

    A provenance field naming Max always counts. Otherwise *teacher_max* (the
    builder's teacher is a Max model) decides when it is known: build.py
    labels a teacher row ``extra`` or ``gym-<date>`` as readily as
    ``max-golden``, so its ``source`` alone proves nothing. When the teacher is
    unknown, only a builder-written ``max_derived: false`` clears the row."""
    if any(MAX_RE.search(v) for v in provenance(row)):
        return True
    if teacher_max is not None:
        return teacher_max
    return not says_not_max(row)


def build_record(data: Path) -> tuple[Path, dict[str, Any]] | None:
    """build.py's record beside *data*: its ``report.json``, else (a build run
    with ``--report`` elsewhere) its ``dataset-manifest.json``, which always
    goes into the output directory. None when neither is a readable object."""
    for name in (BUILD_REPORT, DATASET_MANIFEST):
        where = Path(data).parent / name
        try:
            doc = json.loads(where.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            return where, doc
    return None


def builder_teacher_is_max(data: Path) -> bool | None:
    """Whether the build that wrote *data* kept a Max teacher's rows, from
    build.py's report beside it (``settings.teacher_model``) or its dataset
    manifest (``teacher_model``); None when there is no readable record or it
    names no teacher."""
    found = build_record(data)
    if found is None:
        return None
    where, rep = found
    if where.name == DATASET_MANIFEST:
        teacher = rep.get("teacher_model")
    else:
        settings = rep.get("settings")
        teacher = settings.get("teacher_model") if isinstance(settings, dict) else None
    if not isinstance(teacher, str) or not teacher.strip():
        return None
    return bool(MAX_RE.search(teacher))


def _load_policy() -> Any:
    """``artifact_policy.py`` beside this file (Q6b: names, manifest fields,
    approvals), loaded when first needed: the Modal image carries only this
    file and render.py, and its container never trains Max-derived data."""
    name = "distill_artifact_policy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().with_name("artifact_policy.py"))
    if spec is None or spec.loader is None:
        raise ImportError("artifact_policy.py not found next to train_qlora.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def dataset_record(data: Path | None = None, pretokenized: Path | None = None
                   ) -> dict[str, Any]:
    """The dataset's own record of the Max-derived allow (Q6b): build.py's
    report beside ``train.jsonl`` (``data_policy.max_derived_allow``), or what a
    pre-tokenized directory's ``meta.json`` copied from it. ``approval`` is the
    recorded reference, None when the dataset records no allow."""
    if pretokenized is not None:
        where = Path(pretokenized) / "meta.json"
        try:
            meta = json.loads(where.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        rec = meta.get("dataset_policy") if isinstance(meta, dict) else None
        rec = rec if isinstance(rec, dict) else {}
        return {"source": str(where), "report": rec.get("report"),
                "approval": _load_policy().dataset_approval(rec)}
    if data is None:
        return {"source": None, "report": None, "approval": None}
    found = build_record(data)
    if found is None:
        return {"source": None, "report": None, "approval": None}
    where, rep = found
    if where.name != DATASET_MANIFEST:
        return {"source": str(where), "report": str(where),
                "approval": _load_policy().dataset_approval(rep)}
    # The dataset manifest's approval covers the files it lists: a train.jsonl
    # whose sha256 is not the one recorded there is not the approved data.
    files = rep.get("files") if isinstance(rep.get("files"), dict) else {}
    entry = files.get(Path(data).name)
    want = entry.get("sha256") if isinstance(entry, dict) else None
    try:
        same = isinstance(want, str) and want == sha256_files([Path(data)])
    except OSError:
        same = False
    report = rep.get("report")
    return {"source": str(where), "report": report if isinstance(report, str) else None,
            "approval": _load_policy().dataset_approval(rep) if same else None}


@dataclass
class GuardResult:
    total: int
    max_derived: int
    approved: bool  # the operator's flag on this command line
    # the approval the dataset recorded (build.py --max-derived-approval), None: none
    dataset_approval: str | None = None

    @property
    def allowed(self) -> bool:
        return self.max_derived == 0 or (self.approved and bool(self.dataset_approval))

    @property
    def data_max_derived(self) -> bool:
        """The run trains on Max-derived data: its outputs are tagged and private."""
        return self.max_derived > 0

    def message(self) -> str:
        if self.max_derived == 0:
            return f"provenance: none of {self.total} rows is Max-derived"
        if self.allowed:
            return (f"provenance: {self.max_derived}/{self.total} rows are Max-derived; "
                    f"training them on the operator's approval ({APPROVAL_FLAG}) and the "
                    f"dataset's recorded approval ({self.dataset_approval!r}) under the Q6b "
                    "ruling: the adapter is private and tagged max-derived")
        if not self.approved:
            return (f"REFUSED: {self.max_derived}/{self.total} rows are Max-derived (or nothing "
                    "shows they are not). Training on Max-derived data needs the operator "
                    f"(DISTILL-OPEN-QUESTIONS #6, Q6b ruling): pass {APPROVAL_FLAG} only with "
                    "that approval, on a dataset built with --allow-max-derived "
                    "--max-derived-approval.")
        return (f"REFUSED: {self.max_derived}/{self.total} rows are Max-derived and the dataset "
                "records no Max-derived approval: build it with --allow-max-derived "
                "--max-derived-approval \"<ref>\" (DISTILL-OPEN-QUESTIONS #6, Q6b ruling). "
                f"{APPROVAL_FLAG} alone is not enough.")


def guard(flags: Iterable[bool], approved: bool, dataset_approval: str | None = None
          ) -> GuardResult:
    fl = list(flags)
    ref = dataset_approval if isinstance(dataset_approval, str) and dataset_approval.strip() \
        else None
    return GuardResult(total=len(fl), max_derived=sum(fl), approved=approved,
                       dataset_approval=ref)


def adapter_dir(out: Path, data_max_derived: bool) -> Path:
    """Where the run's adapter goes: ``<out>/adapter``, tagged
    (``<out>/adapter-max-derived``) when the data is Max-derived."""
    name = _load_policy().tag_name(ADAPTER_DIR) if data_max_derived else ADAPTER_DIR
    return Path(out) / name


def out_tag_problem(out: Path, data_max_derived: bool) -> str | None:
    """Why *out* may not hold a run on this data, or None: a run on
    Max-derived data is named for it (Q6b), as build.py's ``--out`` is, so its
    checkpoints, logs and adapter all sit under a tagged name."""
    if not data_max_derived:
        return None
    pol = _load_policy()
    if pol.has_tag(Path(out).name):
        return None
    return (f"the data is Max-derived: the run directory's name must carry the {pol.TAG} tag "
            f"(DISTILL-OPEN-QUESTIONS Q6b), e.g. --out {pol.tag_path(out)}")


def checkpoint_manifest(run_id: str, data_path: Path, source: Source,
                        guard_result: GuardResult) -> dict[str, Any]:
    """What every checkpoint of the run records about itself (Q6b): the
    policy fields, the run, and the data it trains on, in the run manifest's
    schema, so ``artifact_policy.provenance`` reads a copied checkpoint the
    way it reads the run."""
    return {"schema": MANIFEST_SCHEMA, "kind": "checkpoint", "run_id": run_id,
            **_load_policy().policy_fields(guard_result.data_max_derived),
            "approvals": {"operator_flag": {"flag": APPROVAL_FLAG,
                                            "given": guard_result.approved},
                          "dataset": {"approval": guard_result.dataset_approval}},
            "data": {"path": str(Path(data_path).expanduser().resolve()),
                     "sha256": source.sha256(), "rows": guard_result.total,
                     "max_derived_rows": guard_result.max_derived}}


def check_row(row: dict[str, Any], where: str) -> None:
    """Shape checks a row must pass before anything trains on it."""
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise DataError(f"{where}: no 'text'")
    if row.get("tokenize", "segments") != "segments":
        raise DataError(f"{where}: tokenize={row.get('tokenize')!r}, expected 'segments'")
    spans = row.get("loss_spans_char")
    tok = row.get("loss_spans_tok")
    if not spans or not isinstance(spans, list):
        raise DataError(f"{where}: no loss span (nothing to learn)")
    if not isinstance(tok, list) or len(tok) != len(spans):
        raise DataError(f"{where}: loss_spans_tok does not pair with loss_spans_char")
    prev = 0
    for a, b in spans:
        if not (prev <= a < b <= len(text)):
            raise DataError(f"{where}: loss span {a}..{b} is empty, overlapping or outside the text")
        prev = b
    n = row.get("n_tokens")
    prev = 0
    for a, b in tok:
        if not (isinstance(n, int) and prev <= a < b <= n):
            raise DataError(f"{where}: token span {a}..{b} is empty or outside 0..{n}")
        prev = b
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise DataError(f"{where}: no 'id'")


def original_tokens(row: dict[str, Any]) -> int:
    trunc = row.get("truncated") or {}
    return int(trunc.get("original_tokens") or row.get("n_tokens") or 0)


# ---------------------------------------------------------------------------
# Tokenization (the render.py segment recipe)
# ---------------------------------------------------------------------------


class Encoder(Protocol):
    def encode_ids(self, text: str) -> list[int]: ...


def segments(text: str, spans: list[list[int]]) -> list[tuple[str, bool]]:
    """The text cut at every span edge: (segment, carries_loss)."""
    out: list[tuple[str, bool]] = []
    pos = 0
    for a, b in spans:
        if a > pos:
            out.append((text[pos:a], False))
        out.append((text[a:b], True))
        pos = b
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def tokenize_row(row: dict[str, Any], encoder: Encoder) -> tuple[array.array, list[tuple[int, int]]]:
    """Ids and token loss spans, checked against what the row recorded."""
    ids = array.array("I")
    spans: list[tuple[int, int]] = []
    for seg, loss in segments(row["text"], row["loss_spans_char"]):
        part = encoder.encode_ids(seg)
        if loss:
            spans.append((len(ids), len(ids) + len(part)))
        ids.extend(part)
    want = [tuple(s) for s in row["loss_spans_tok"]]
    if len(ids) != row["n_tokens"] or spans != want:
        raise DataError(
            f"row {row.get('id')}: tokenizer gives {len(ids)} tokens / spans {spans}, the row "
            f"says {row['n_tokens']} / {want}: not the tokenizer the data was rendered with")
    return ids, spans


def loss_tokens(spans: Iterable[tuple[int, int]]) -> int:
    """Positions that score a loss after the causal shift (position 0 never does)."""
    return sum(max(0, b - max(a, 1)) for a, b in spans)


def load_render() -> Any:
    name = "distill_render"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("render.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hf_encoder(tokenizer_dir: Path) -> Encoder:
    return load_render().HFTokenizer(Path(tokenizer_dir).expanduser() / "tokenizer.json")


Refit = Callable[[dict[str, Any], int], "dict[str, Any] | None"]


def template_refitter(tokenizer_dir: Path) -> Refit:
    """Re-fit a row to a shorter budget with the student's own template:
    oldest context dropped, system prompt, tools, last query and the learned
    turn kept (``render.fit_to_budget``)."""
    R = load_render()
    student = R.Student.load(tokenizer_dir)

    def refit(row: dict[str, Any], budget: int) -> dict[str, Any] | None:
        fit = R.fit_to_budget(student, row["messages"], row.get("tools") or None, budget,
                              row.get("chat_template_kwargs"))
        if fit is None:
            return None
        out = dict(row)
        out.update(fit.rendered.as_row())
        out["messages"] = fit.messages
        prev = row.get("truncated") or {}
        out["truncated"] = {
            "dropped_messages": int(prev.get("dropped_messages") or 0) + fit.dropped,
            "dropped_turn_steps": int(prev.get("dropped_turn_steps") or 0) + fit.dropped_steps,
            "original_tokens": original_tokens(row), "refit_to": budget}
        return out

    return refit


# ---------------------------------------------------------------------------
# Data sources: rows fitted to one max_seq
# ---------------------------------------------------------------------------


@dataclass
class FitReport:
    max_seq: int
    total: int
    kept: int
    refit: int
    dropped: int
    tokens: int  # tokens of the kept examples, one epoch

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Source(Protocol):
    def ids(self) -> list[str]: ...
    def sha256(self) -> str: ...
    def max_derived(self) -> list[bool]: ...
    def sources(self) -> dict[str, int]: ...
    def fit(self, max_seq: int) -> FitReport: ...
    def kept(self) -> set[str]: ...
    def example(self, ex_id: str) -> tuple[Any, list[tuple[int, int]]]: ...
    def n_tokens(self, ex_id: str) -> int: ...


class JsonlSource:
    """``train.jsonl`` rows, tokenized at fit time with *encoder*."""

    def __init__(self, path: Path, encoder: Encoder | None, refit: Refit | None = None,
                 rows: list[dict[str, Any]] | None = None) -> None:
        self.path = Path(path)
        self.rows = rows if rows is not None else read_jsonl(self.path)
        self.encoder = encoder
        self.refit = refit
        seen: set[str] = set()
        for i, row in enumerate(self.rows):
            check_row(row, f"{self.path.name}:{i + 1}")
            if row["id"] in seen:
                raise DataError(f"{self.path.name}: duplicate id {row['id']}")
            seen.add(row["id"])
        self.by_id = {r["id"]: r for r in self.rows}
        self._fitted: dict[str, tuple[Any, list[tuple[int, int]]]] = {}
        self._sha: str | None = None

    def ids(self) -> list[str]:
        return sorted(self.by_id)

    def sha256(self) -> str:
        if self._sha is None:
            self._sha = sha256_files([self.path]) if self.path.is_file() else sha256_text(
                json.dumps(self.rows, sort_keys=True))
        return self._sha

    def max_derived(self) -> list[bool]:
        teacher_max = builder_teacher_is_max(self.path)
        return [is_max_derived(r, teacher_max) for r in self.rows]

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[row_source(r)] = out.get(row_source(r), 0) + 1
        return dict(sorted(out.items()))

    def fit(self, max_seq: int) -> FitReport:
        if self.encoder is None:
            raise UsageError("tokenizing needs --tokenizer-dir")
        self._fitted = {}
        refit = dropped = tokens = 0
        for ex_id in self.ids():
            row = self.by_id[ex_id]
            if row["n_tokens"] > max_seq:
                new = self.refit(row, max_seq) if self.refit else None
                if new is None or new["n_tokens"] > max_seq:
                    dropped += 1
                    continue
                check_row(new, f"refit {ex_id}")
                row, refit = new, refit + 1
            ids, spans = tokenize_row(row, self.encoder)
            self._fitted[ex_id] = (ids, spans)
            tokens += len(ids)
        return FitReport(max_seq, len(self.rows), len(self._fitted), refit, dropped, tokens)

    def kept(self) -> set[str]:
        return set(self._fitted)

    def example(self, ex_id: str) -> tuple[Any, list[tuple[int, int]]]:
        return self._fitted[ex_id]

    def n_tokens(self, ex_id: str) -> int:
        return len(self._fitted[ex_id][0])


def write_pretokenized(rows: list[dict[str, Any]], encoder: Encoder, out_dir: Path, *,
                       ladder: tuple[int, ...] = SEQ_LADDER, refit: Refit | None = None,
                       data_sha256: str = "", tokenizer_sha256: str = "",
                       teacher_max: bool | None = None,
                       dataset_policy: dict[str, Any] | None = None,
                       source: str = "") -> dict[str, Any]:
    """Pre-tokenize rows for a run that must not tokenize on paid time.

    ``tokens.bin`` holds uint32 ids back to back; ``index.jsonl`` has one line
    per row with a variant per ladder length (offset, length, loss spans), so a
    step-down needs no template or tokenizer. A row already within a length
    shares its base variant; a row over it is re-fitted (*refit*) or has no
    variant there (dropped at that length). Each row's ``max_derived`` flag is
    :func:`is_max_derived` with the builder's *teacher_max*
    (:func:`builder_teacher_is_max` of the source ``train.jsonl``).
    ``meta.json`` carries the Q6b fields (``max_derived``, ``tags``,
    ``private``), the *source* path, and *dataset_policy*: the source
    dataset's recorded Max-derived allow (``{"report": ..., "max_derived_allow":
    ...}``), which a run on this directory needs for Max-derived rows.
    """
    if sys.byteorder != "little":
        raise UsageError("pre-tokenized files are little-endian; this host is not")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    kept = {str(L): 0 for L in ladder}
    tok_by_seq = {str(L): 0 for L in ladder}
    flags = []
    with (out_dir / "tokens.bin.tmp").open("wb") as tb, \
            (out_dir / "index.jsonl.tmp").open("w", encoding="utf-8") as ix:
        for i, row in enumerate(rows):
            check_row(row, f"row {i + 1}")
            ids, spans = tokenize_row(row, encoder)
            tb.write(ids.tobytes())
            base = {"off": offset, "n": len(ids), "spans": [list(s) for s in spans]}
            offset += len(ids)
            variants: dict[str, Any] = {}
            for L in ladder:
                if len(ids) <= L:
                    variants[str(L)] = base
                    continue
                new = refit(row, L) if refit else None
                if new is None or new["n_tokens"] > L:
                    variants[str(L)] = None
                    continue
                v_ids, v_spans = tokenize_row(new, encoder)
                tb.write(v_ids.tobytes())
                variants[str(L)] = {"off": offset, "n": len(v_ids),
                                    "spans": [list(s) for s in v_spans]}
                offset += len(v_ids)
            for L in ladder:
                v = variants[str(L)]
                if v:
                    kept[str(L)] += 1
                    tok_by_seq[str(L)] += v["n"]
            md = is_max_derived(row, teacher_max)
            flags.append(md)
            ix.write(json.dumps({"id": row["id"], "source": row_source(row), "max_derived": md,
                                 "original_tokens": original_tokens(row),
                                 "variants": variants}) + "\n")
    (out_dir / "tokens.bin.tmp").replace(out_dir / "tokens.bin")
    (out_dir / "index.jsonl.tmp").replace(out_dir / "index.jsonl")
    meta = {
        "schema": PRETOK_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(rows), "ladder": list(ladder), "kept_by_seq": kept,
        "tokens_by_seq": tok_by_seq, "total_ids": offset,
        "max_derived_rows": sum(flags), "data_sha256": data_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "content_sha256": sha256_files([out_dir / "index.jsonl", out_dir / "tokens.bin"]),
        **_load_policy().policy_fields(sum(flags) > 0),
        "source": source, "dataset_policy": dict(dataset_policy or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


class PretokSource:
    """A ``write_pretokenized`` directory; ids are read from a memory map."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        if self.meta.get("schema") != PRETOK_SCHEMA:
            raise DataError(f"{self.path}: not a pre-tokenized directory ({PRETOK_SCHEMA})")
        self.index = {r["id"]: r for r in read_jsonl(self.path / "index.jsonl")}
        self._mm: mmap.mmap | None = None
        self._fitted: dict[str, dict[str, Any]] = {}

    def _ids_view(self) -> memoryview:
        if self._mm is None:
            fh = (self.path / "tokens.bin").open("rb")
            self._mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            fh.close()
        return memoryview(self._mm).cast("I")

    def ids(self) -> list[str]:
        return sorted(self.index)

    def sha256(self) -> str:
        return str(self.meta["content_sha256"])

    def max_derived(self) -> list[bool]:
        # the flag stored at pre-tokenize time, re-derived when absent
        return [bool(r.get("max_derived", True)) for r in self.index.values()]

    def sources(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.index.values():
            out[r.get("source") or "unknown"] = out.get(r.get("source") or "unknown", 0) + 1
        return dict(sorted(out.items()))

    def fit(self, max_seq: int) -> FitReport:
        if str(max_seq) not in {str(x) for x in self.meta["ladder"]}:
            raise DataError(f"{self.path}: no variants for max_seq {max_seq}")
        self._fitted = {}
        refit = dropped = tokens = 0
        for ex_id, r in self.index.items():
            v = r["variants"].get(str(max_seq))
            if not v:
                dropped += 1
                continue
            base = r["variants"][str(self.meta["ladder"][0])]
            refit += int(base is not None and v["off"] != base["off"])
            self._fitted[ex_id] = v
            tokens += v["n"]
        return FitReport(max_seq, len(self.index), len(self._fitted), refit, dropped, tokens)

    def kept(self) -> set[str]:
        return set(self._fitted)

    def example(self, ex_id: str) -> tuple[Any, list[tuple[int, int]]]:
        v = self._fitted[ex_id]
        view = self._ids_view()
        return view[v["off"]:v["off"] + v["n"]], [tuple(s) for s in v["spans"]]

    def n_tokens(self, ex_id: str) -> int:
        return int(self._fitted[ex_id]["n"])


# ---------------------------------------------------------------------------
# The plan (sample order) and checkpoints
# ---------------------------------------------------------------------------


def epoch_plan(ids: Iterable[str], epochs: int, seed: int) -> list[str]:
    """Every id once per epoch, in a seeded order per epoch (string seeds are
    stable across processes, unlike hash())."""
    base = sorted(ids)
    plan: list[str] = []
    for e in range(epochs):
        order = base[:]
        random.Random(f"distill:{seed}:{e}").shuffle(order)
        plan.extend(order)
    return plan


def next_group(plan: list[str], pos: int, kept: set[str], size: int) -> tuple[list[str], int]:
    """The next *size* kept ids from plan position *pos*, and the position after
    them. Ids not kept at this length are skipped, so a position stays valid
    across a step-down."""
    group: list[str] = []
    i = pos
    while i < len(plan) and len(group) < size:
        if plan[i] in kept:
            group.append(plan[i])
        i += 1
    while i < len(plan) and plan[i] not in kept:
        i += 1  # the position after the group lands on the next kept id (or the end)
    return group, i


def remaining_steps(plan: list[str], pos: int, kept: set[str], size: int) -> int:
    n = sum(1 for x in plan[pos:] if x in kept)
    return math.ceil(n / size)


def select_smoke(ids: Iterable[str], seed: int, n: int = SMOKE_EXAMPLES) -> list[str]:
    order = sorted(ids)
    random.Random(f"distill-smoke:{seed}").shuffle(order)
    return sorted(order[:n])


def loss_decreased(losses: list[float], window: int = 10, min_drop: float = 0.02
                   ) -> tuple[bool, float, float]:
    """Mean of the last *window* step losses is below the first window's by at
    least *min_drop* (a fraction). Needs two non-overlapping windows."""
    if len(losses) < 2 * window:
        return False, float("nan"), float("nan")
    first = sum(losses[:window]) / window
    last = sum(losses[-window:]) / window
    return last < first * (1.0 - min_drop), first, last


def ckpt_step(path: Path) -> int | None:
    name = path.name
    if not name.startswith(CKPT_PREFIX) or name.endswith(PARTIAL):
        return None
    try:
        return int(name[len(CKPT_PREFIX):])
    except ValueError:
        return None


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    """The newest complete checkpoint: a renamed directory with a COMPLETE
    marker and a state.json whose step matches its name. Anything else (a
    ``.partial`` left by a kill, a corrupt state) is skipped."""
    best: tuple[int, Path] | None = None
    if not Path(out_dir).is_dir():
        return None
    for p in Path(out_dir).iterdir():
        step = ckpt_step(p)
        if step is None or not p.is_dir() or not (p / COMPLETE).is_file():
            continue
        try:
            st = json.loads((p / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("skipping %s: unreadable state.json", p.name)
            continue
        if st.get("step") != step:
            log.warning("skipping %s: state.json says step %s", p.name, st.get("step"))
            continue
        if best is None or step > best[0]:
            best = (step, p)
    return best[1] if best else None


def prune_checkpoints(out_dir: Path, keep: int) -> list[Path]:
    """Remove all but the newest *keep* complete checkpoints, and stale
    ``.partial`` directories. Only this run's own output directory."""
    complete = sorted((s, p) for p in Path(out_dir).iterdir()
                      if (s := ckpt_step(p)) is not None and (p / COMPLETE).is_file())
    removed = [p for _, p in complete[:-keep]] if keep > 0 else []
    removed += [p for p in Path(out_dir).iterdir() if p.name.endswith(PARTIAL)]
    for p in removed:
        shutil.rmtree(p, ignore_errors=True)
    return removed


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Backend: the only part that touches torch / unsloth
# ---------------------------------------------------------------------------


class Backend(Protocol):
    def load(self, max_seq: int, revision: str | None) -> None: ...
    def forward_backward(self, ids: Any, spans: list[tuple[int, int]], scale: float) -> float: ...
    def step(self, lr: float) -> None: ...
    def zero_grad(self) -> None: ...
    def eval_loss(self, ids: Any, spans: list[tuple[int, int]]) -> float: ...
    def save(self, d: Path) -> None: ...
    def restore(self, d: Path) -> None: ...
    def rng_state(self) -> Any: ...
    def set_rng_state(self, state: Any) -> None: ...
    def save_rng(self, path: Path, state: Any) -> None: ...
    def load_rng(self, path: Path) -> Any: ...
    def save_adapter(self, d: Path) -> None: ...
    def fingerprint(self) -> float: ...
    def peak_memory_gb(self) -> float | None: ...
    def free(self) -> None: ...
    def versions(self) -> dict[str, Any]: ...


def package_versions(names: Iterable[str] = VERSIONED) -> dict[str, str | None]:
    from importlib import metadata

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n.replace("_", "-"))
        except metadata.PackageNotFoundError:
            try:
                out[n] = metadata.version(n)
            except metadata.PackageNotFoundError:
                out[n] = None
    return out


class UnslothBackend:
    """Unsloth model + PEFT LoRA + 8-bit AdamW. Imports happen in ``load``."""

    def __init__(self, cfg: TrainConfig, *, load_kwargs: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.load_kwargs = dict(load_kwargs or {})
        self.model: Any = None
        self.opt: Any = None
        self.params: list[Any] = []
        self.torch: Any = None

    def load(self, max_seq: int, revision: str | None) -> None:
        import unsloth  # noqa: F401  (first: it patches transformers and peft)
        import torch
        from unsloth import FastLanguageModel

        self.torch = torch
        cfg = self.cfg
        model, _tok = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model, revision=revision, max_seq_length=max_seq,
            load_in_4bit=cfg.load == "4bit", load_in_16bit=cfg.load == "16bit",
            full_finetuning=False, dtype=None, **self.load_kwargs)
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
            target_modules=list(cfg.target_modules), bias="none",
            use_gradient_checkpointing=cfg.gradient_checkpointing,
            random_state=cfg.seed, max_seq_length=max_seq)
        model.train()
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        if cfg.optim == "adamw_8bit":
            import bitsandbytes as bnb

            self.opt = bnb.optim.AdamW8bit(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        else:
            self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        torch.cuda.reset_peak_memory_stats()

    def _inputs(self, ids: Any, spans: list[tuple[int, int]]) -> tuple[Any, Any]:
        torch = self.torch
        x = torch.tensor(list(ids), dtype=torch.long, device="cuda").unsqueeze(0)
        labels = torch.full_like(x, IGNORE)
        for a, b in spans:
            labels[0, a:b] = x[0, a:b]
        return x, labels

    def forward_backward(self, ids: Any, spans: list[tuple[int, int]], scale: float) -> float:
        x, labels = self._inputs(ids, spans)
        loss = self.model(input_ids=x, labels=labels).loss
        (loss * scale).backward()
        return float(loss.detach())

    def step(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        if self.cfg.max_grad_norm:
            self.torch.nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    def eval_loss(self, ids: Any, spans: list[tuple[int, int]]) -> float:
        x, labels = self._inputs(ids, spans)
        self.model.eval()
        try:
            with self.torch.no_grad():
                return float(self.model(input_ids=x, labels=labels).loss)
        finally:
            self.model.train()

    def save(self, d: Path) -> None:
        self.model.save_pretrained(str(d / "adapter"))
        self.torch.save(self.opt.state_dict(), d / "optimizer.pt")

    def restore(self, d: Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(self.model, load_file(str(d / "adapter" / "adapter_model.safetensors")))
        self.opt.load_state_dict(self.torch.load(d / "optimizer.pt", map_location="cpu",
                                                 weights_only=False))

    def rng_state(self) -> Any:
        t = self.torch
        return {"python": random.getstate(), "torch": t.get_rng_state(),
                "cuda": t.cuda.get_rng_state_all()}

    def set_rng_state(self, state: Any) -> None:
        random.setstate(state["python"])
        self.torch.set_rng_state(state["torch"])
        self.torch.cuda.set_rng_state_all(state["cuda"])

    def save_rng(self, path: Path, state: Any) -> None:
        self.torch.save(state, path)

    def load_rng(self, path: Path) -> Any:
        return self.torch.load(path, weights_only=False)

    def save_adapter(self, d: Path) -> None:
        self.model.save_pretrained(str(d))

    def fingerprint(self) -> float:
        # sum + sum of squares in float64: never near zero (so a relative
        # tolerance means something) and the same on any GPU that resumes
        with self.torch.no_grad():
            return float(sum(p.detach().double().sum().item()
                             + p.detach().double().square().sum().item() for p in self.params))

    def peak_memory_gb(self) -> float | None:
        return round(self.torch.cuda.max_memory_allocated() / 2**30, 2)

    def free(self) -> None:
        self.torch.cuda.empty_cache()

    def versions(self) -> dict[str, Any]:
        t = self.torch
        v: dict[str, Any] = dict(package_versions())
        v["cuda"] = getattr(t.version, "cuda", None) if t else None
        v["gpu"] = t.cuda.get_device_name(0) if t and t.cuda.is_available() else None
        return v


# ---------------------------------------------------------------------------
# Trainer: the loop, independent of torch
# ---------------------------------------------------------------------------


@dataclass
class StopFlag:
    """Set by SIGTERM / a deadline; checked between micro-steps."""

    reason: str | None = None
    stop_at: float | None = None  # epoch seconds

    def check(self, now: float | None = None) -> str | None:
        if self.reason:
            return self.reason
        if self.stop_at is not None and (now if now is not None else time.time()) >= self.stop_at:
            self.reason = "stop-at"
        return self.reason


@dataclass
class StepRecord:
    step: int
    loss: float
    lr: float
    tokens: int
    seconds: float


@dataclass
class RunResult:
    status: str  # done | stopped | until_step
    step: int
    reason: str | None = None
    checkpoint: str | None = None
    records: list[StepRecord] = field(default_factory=list)


Hook = Callable[["Trainer", StepRecord], "str | None"]


class Trainer:
    """The training loop over a :class:`Source` and a :class:`Backend`."""

    def __init__(self, cfg: TrainConfig, source: Source, out_dir: Path, backend: Backend, *,
                 stop: StopFlag | None = None, only_ids: list[str] | None = None,
                 val: Source | None = None, val_max: int = 64,
                 metrics_path: Path | None = None,
                 on_save: Callable[[Path], None] | None = None,
                 revision: str | None = None,
                 checkpoint_manifest: dict[str, Any] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        # Q6b: written into every checkpoint (CKPT_MANIFEST), so a copied
        # checkpoint carries its own provenance; None writes none
        self.checkpoint_manifest = checkpoint_manifest
        self.revision = revision  # the resolved base revision (a sha when known)
        self.source = source
        self.out = Path(out_dir)
        self.backend = backend
        self.stop = stop or StopFlag()
        self.only_ids = only_ids
        self.val = val
        self.val_max = val_max
        self.metrics_path = metrics_path
        self.on_save = on_save
        self.clock = clock
        self.step = 0
        self.pos = 0
        self.tokens_seen = 0
        self.losses: list[float] = []
        self.records: list[StepRecord] = []
        self.run_state: dict[str, Any] = {}
        self.fit_report: FitReport | None = None
        self.resumed_from: str | None = None
        self.restore_check: dict[str, Any] | None = None
        self.plan: list[str] = []
        self.kept: set[str] = set()

    # ------------------------------------------------------------ setup

    @property
    def max_seq(self) -> int:
        return int(self.run_state["max_seq"])

    @property
    def total_steps(self) -> int:
        return int(self.run_state["schedule"]["total_steps"])

    def setup(self, *, resume: bool) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        ids = self.only_ids if self.only_ids is not None else self.source.ids()
        self.plan = epoch_plan(ids, self.cfg.epochs, self.cfg.seed)
        plan_sha = sha256_text("\n".join(self.plan))
        rs_path = self.out / RUN_STATE
        latest = find_latest_checkpoint(self.out)
        if resume and rs_path.is_file():
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            for key, have in (("data_sha256", self.source.sha256()),
                              ("config_sha", self.cfg.sha()), ("plan_sha", plan_sha)):
                if rs.get(key) != have:
                    raise UsageError(f"cannot resume: {key} differs from the run's "
                                     f"({rs.get(key)} vs {have}); exact resume needs the same "
                                     "data and config")
        else:
            if latest is not None or rs_path.is_file():
                raise UsageError(f"{self.out} already holds a run; pass --resume or use a new --out")
            rs = {"schema": RUN_STATE_SCHEMA, "created_at": now_iso(),
                  "data_sha256": self.source.sha256(), "config_sha": self.cfg.sha(),
                  "plan_sha": plan_sha, "plan_len": len(self.plan),
                  "max_seq": self.cfg.max_seq, "seq_history": [], "schedule": None,
                  "base_revision_resolved": self.revision or self.cfg.base_revision}
        self.run_state = rs
        # a resume loads the weights the run started on, even if the branch moved
        self.revision = rs.get("base_revision_resolved") or self.revision
        self.fit_report = self.source.fit(self.max_seq)
        self.kept = self.source.kept() & set(self.plan)
        if not self.kept:
            raise DataError(f"no example fits max_seq {self.max_seq}")
        self.backend.load(self.max_seq, self.revision)
        if resume and latest is not None:
            self.restore_check = self.restore_from(latest)
            if self.restore_check["ok"] is False:
                raise RestoreError(
                    f"cannot resume: restoring {latest.name} gave weights with fingerprint "
                    f"{self.restore_check['after']!r}, it saved {self.restore_check['saved']!r}")
        sched = rs.get("schedule")
        if not sched or sched.get("max_seq") != self.max_seq:
            total = self.step + remaining_steps(self.plan, self.pos, self.kept, self.cfg.grad_accum)
            if self.cfg.max_steps is not None:
                total = min(total, self.cfg.max_steps)
            warmup = (sched or {}).get("warmup_steps")
            if warmup is None:
                warmup = max(1, round(self.cfg.warmup_ratio * total)) if self.cfg.warmup_ratio else 0
            rs["schedule"] = {"max_seq": self.max_seq, "total_steps": total, "warmup_steps": warmup,
                              "from_step": self.step}
        rs["updated_at"] = now_iso()
        write_json(rs_path, rs)
        self.preflight()

    def preflight(self) -> None:
        """One forward+backward on the longest example at this length, so an
        OOM shows up before any step instead of hours in."""
        longest = max(self.kept, key=self.source.n_tokens)
        ids, spans = self.source.example(longest)
        rng = self.backend.rng_state()
        try:
            self.backend.forward_backward(ids, spans, 1.0)
        except Exception as exc:
            if is_oom(exc):
                self.step_down("preflight")
            raise
        finally:
            self.backend.zero_grad()
            self.backend.set_rng_state(rng)
        log.info("preflight ok: %d tokens at max_seq %d, peak %s GiB", len(ids), self.max_seq,
                 self.backend.peak_memory_gb())

    def step_down(self, where: str) -> None:
        """Persist the next ladder length and raise :class:`StepDown`."""
        old = self.max_seq
        new = next_seq(old, self.cfg.ladder)
        self.backend.zero_grad()
        self.backend.free()
        if new is None:
            raise RuntimeError(f"CUDA OOM at the smallest max_seq {old} ({where})")
        rs = self.run_state
        rs["seq_history"].append({"from": old, "to": new, "where": where, "step": self.step,
                                  "at": now_iso(), "reason": "cuda_oom"})
        rs["max_seq"] = new
        rs["updated_at"] = now_iso()
        write_json(self.out / RUN_STATE, rs)
        raise StepDown(old, new, self.step)

    # ------------------------------------------------------------- loop

    def remaining_tokens(self) -> int:
        return sum(self.source.n_tokens(x) for x in self.plan[self.pos:] if x in self.kept)

    def finished(self) -> bool:
        if self.cfg.max_steps is not None and self.step >= self.cfg.max_steps:
            return True
        return next_group(self.plan, self.pos, self.kept, 1)[0] == []

    def run(self, *, until_step: int | None = None, hook: Hook | None = None) -> RunResult:
        """Train until the plan is done, *until_step*, a stop request or a hook's
        stop. A CUDA OOM becomes :class:`StepDown` (after persisting it)."""
        sched = self.run_state["schedule"]
        start = len(self.records)
        while not self.finished():
            if until_step is not None and self.step >= until_step:
                return RunResult("until_step", self.step, records=self.records[start:])
            reason = self.stop.check()
            if reason:
                return self._stopped(reason, start)
            group, new_pos = next_group(self.plan, self.pos, self.kept, self.cfg.grad_accum)
            exs = [self.source.example(x) for x in group]
            weight = sum(loss_tokens(s) for _, s in exs) or 1
            rng = self.backend.rng_state()
            t0 = self.clock()
            loss = 0.0
            tokens = 0
            try:
                for ids, spans in exs:
                    if self.stop.check():
                        break
                    n = loss_tokens(spans)
                    loss += self.backend.forward_backward(ids, spans, n / weight) * n / weight
                    tokens += len(ids)
            except Exception as exc:
                if is_oom(exc):
                    self.step_down(f"step {self.step + 1}")
                raise
            if self.stop.reason:  # mid-group: drop the partial group, keep the last step
                self.backend.zero_grad()
                self.backend.set_rng_state(rng)
                return self._stopped(self.stop.reason, start)
            lr = lr_at(self.step, sched["total_steps"], sched["warmup_steps"], self.cfg.lr)
            self.backend.step(lr)
            self.step += 1
            self.pos = new_pos
            self.tokens_seen += tokens
            self.losses.append(loss)
            rec = StepRecord(self.step, loss, lr, tokens, self.clock() - t0)
            self.records.append(rec)
            self._metrics(rec)
            if self.step % self.cfg.save_every == 0:
                self.save("interval")
            if hook is not None:
                hook_reason = hook(self, rec)
                if hook_reason:
                    self.stop.reason = hook_reason
                    return self._stopped(hook_reason, start)
        path = self.save("end")
        return RunResult("done", self.step, checkpoint=path.name, records=self.records[start:])

    def _stopped(self, reason: str, start: int) -> RunResult:
        path = self.save(f"stop:{reason}")
        log.info("stopped (%s) at step %d; resume with --resume", reason, self.step)
        return RunResult("stopped", self.step, reason=reason, checkpoint=path.name,
                         records=self.records[start:])

    def _metrics(self, rec: StepRecord) -> None:
        if self.step % self.cfg.log_every == 0 or self.step == 1:
            tps = rec.tokens / rec.seconds if rec.seconds > 0 else 0.0
            log.info("step %d/%d loss %.4f lr %.2e %d tok %.0f tok/s", rec.step, self.total_steps,
                     rec.loss, rec.lr, rec.tokens, tps)
        if self.metrics_path is not None:
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**asdict(rec), "max_seq": self.max_seq, "at": now_iso(),
                                     "peak_gb": self.backend.peak_memory_gb()}) + "\n")

    # ------------------------------------------------------- checkpoints

    def val_loss(self) -> float | None:
        if self.val is None:
            return None
        ids = sorted(self.val.kept())[: self.val_max]
        num = den = 0.0
        for x in ids:
            t, s = self.val.example(x)
            n = loss_tokens(s)
            num += self.backend.eval_loss(t, s) * n
            den += n
        return num / den if den else None

    def save(self, why: str) -> Path:
        """Checkpoint at the current step (no-op when the newest one is it)."""
        final = self.out / f"{CKPT_PREFIX}{self.step:06d}"
        if final.is_dir() and (final / COMPLETE).is_file():
            return final
        tmp = self.out / (final.name + PARTIAL)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        self.backend.save(tmp)
        self.backend.save_rng(tmp / "rng.pt", self.backend.rng_state())
        vl = self.val_loss() if why == "interval" or why == "end" else None
        write_json(tmp / "state.json", {
            "step": self.step, "pos": self.pos, "tokens_seen": self.tokens_seen,
            "max_seq": self.max_seq, "total_steps": self.total_steps, "why": why,
            "losses": self.losses[-200:], "val_loss": vl, "saved_at": now_iso(),
            "plan_sha": self.run_state["plan_sha"],
            # what a restore of this checkpoint must bring back (checked on resume)
            "fingerprint": self.backend.fingerprint()})
        if self.checkpoint_manifest is not None:
            write_json(tmp / CKPT_MANIFEST, {**self.checkpoint_manifest, "checkpoint": final.name,
                                             "step": self.step, "written_at": now_iso()})
        (tmp / COMPLETE).write_text(now_iso() + "\n", encoding="utf-8")
        tmp.replace(final)
        prune_checkpoints(self.out, self.cfg.keep_checkpoints)
        log.info("checkpoint %s (%s)%s", final.name, why,
                 f" val_loss {vl:.4f}" if vl is not None else "")
        if self.on_save is not None:
            self.on_save(final)
        return final

    def restore_from(self, ckpt: Path) -> dict[str, Any]:
        """Load *ckpt* into the backend and the loop (weights, optimizer, RNG,
        step, plan position, losses) and check the weights against the
        fingerprint recorded when it was saved. ``ok`` is None for a
        checkpoint saved without one; ``moved`` says the weights in memory
        before the restore were not the checkpoint's."""
        st = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        before = self.backend.fingerprint()
        self.backend.restore(ckpt)
        self.backend.set_rng_state(self.backend.load_rng(ckpt / "rng.pt"))
        after = self.backend.fingerprint()
        self.step, self.pos = int(st["step"]), int(st["pos"])
        self.tokens_seen = int(st.get("tokens_seen", 0))
        self.losses = [float(x) for x in st.get("losses", [])]
        self.resumed_from = ckpt.name
        saved = st.get("fingerprint")

        def same(x: float, y: float) -> bool:
            return math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6)

        return {"checkpoint": ckpt.name, "saved": saved, "before": before, "after": after,
                "ok": None if saved is None else same(after, float(saved)),
                "moved": None if saved is None else not same(before, float(saved))}

    def verify_resume(self, ckpt: Path) -> dict[str, Any]:
        """The smoke's checkpoint round trip, once the weights have trained
        past *ckpt*: restoring it must bring back exactly the weights it saved,
        and those must differ from the ones it replaced, so a restore that
        silently does nothing fails. The loop continues from *ckpt*."""
        check = self.restore_from(ckpt)
        check["passed"] = bool(check["ok"] and check["moved"])
        return check

    def save_adapter(self, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        self.backend.save_adapter(dest)
        return dest


# ---------------------------------------------------------------------------
# Dry run: validation and token preview without torch
# ---------------------------------------------------------------------------


def _clip(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n  [... {len(text) - head - tail:,} chars elided ...]\n{text[-tail:]}"


def preview(row: dict[str, Any], *, ids_spans: tuple[Any, list[tuple[int, int]]] | None = None,
            head: int = 400, tail: int = 300) -> str:
    """One example with its masked (no loss) and trained spans marked."""
    tok = ""
    if ids_spans is not None:
        ids, spans = ids_spans
        tok = f", {len(ids):,} tokens, {loss_tokens(spans):,} carry loss (tokenizer-checked)"
    else:
        tok = f", {row.get('n_tokens', '?'):,} tokens (from the row, not re-tokenized)"
    out = [f"=== {row['id']}  source={row_source(row)}  category={row.get('category', '?')}"
           f"{tok} ==="]
    for seg, loss in segments(row["text"], row["loss_spans_char"]):
        if loss:
            out.append(f"--- [LOSS: trained, {len(seg):,} chars] " + "-" * 20)
            out.append(_clip(seg, 1600, 400))
            out.append("--- [/LOSS] " + "-" * 20)
        else:
            out.append(f"--- [MASKED: no loss, {len(seg):,} chars] " + "-" * 20)
            out.append(_clip(seg, head, tail))
    return "\n".join(out)


def length_table(rows: list[dict[str, Any]], ladder: Iterable[int]) -> list[dict[str, Any]]:
    out = []
    for L in ladder:
        fits = [r for r in rows if r["n_tokens"] <= L]
        out.append({"max_seq": L, "fit_as_is": len(fits), "over": len(rows) - len(fits),
                    "untruncated": sum(1 for r in rows if original_tokens(r) <= L)})
    return out


# ---------------------------------------------------------------------------
# Manifest, revision, CLI
# ---------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def base_source_problem(base_model: str, *, env: Mapping[str, str],
                        allow_download: bool) -> str | None:
    """Why a real run must not load *base_model* this way, or None. A local
    directory is fine (a staged, pinned download). A hub id loads from the HF
    cache only under ``HF_HUB_OFFLINE=1`` (missing: the load fails), and
    otherwise only with ``--allow-hub-download``: from_pretrained would
    download the full-precision checkpoint (~19 GB for the 9B) into the HF
    cache with no free-space check."""
    if Path(base_model).expanduser().is_dir():
        return None
    if Path(base_model).expanduser().exists() or base_model.startswith(("/", "~", ".")):
        return f"--base-model {base_model} is not a model directory"
    if env.get("HF_HUB_OFFLINE", "") not in ("", "0", "false", "False") or allow_download:
        return None
    return (f"--base-model {base_model} is a hub id: loading it would download the base into the HF "
            "cache with no disk-floor check. Stage it with a pinned download and pass its directory "
            "(plus --base-revision <commit>), or set HF_HUB_OFFLINE=1 to load it from the cache "
            "only, or pass --allow-hub-download")


def resolve_revision(repo: str, revision: str | None) -> str | None:
    """The commit sha a Hugging Face repo revision points at (network), or
    None when it cannot be resolved here (local path, offline, no hub client)."""
    if Path(repo).expanduser().exists():
        return None
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo, revision=revision or "main").sha)
    except Exception as exc:  # offline or missing package: recorded, not fatal
        log.warning("could not resolve %s@%s: %s", repo, revision or "main", type(exc).__name__)
        return None


def manifest(cfg: TrainConfig, source: Source, *, mode: str, run_id: str, data_path: Path,
             approved: bool, guard_result: GuardResult, revision_resolved: str | None,
             fit: list[dict[str, Any]], versions: dict[str, Any], extra: dict[str, Any],
             dataset: dict[str, Any] | None = None) -> dict[str, Any]:
    script = Path(__file__).resolve()
    ds = dataset or {}
    return {
        "schema": MANIFEST_SCHEMA, "run_id": run_id, "mode": mode, "written_at": now_iso(),
        # Q6b: a run on Max-derived data is private and tagged; both approvals recorded
        **_load_policy().policy_fields(guard_result.data_max_derived),
        "approvals": {
            "operator_flag": {"flag": APPROVAL_FLAG, "given": approved},
            "dataset": {"approval": guard_result.dataset_approval,
                        "recorded_in": ds.get("source")},
            "ruling": "DISTILL-OPEN-QUESTIONS #6, Q6b",
        },
        "script": {"path": str(script.relative_to(REPO)) if script.is_relative_to(REPO)
                   else script.name, "sha256": sha256_files([script]), "git_commit": git_commit()},
        "data": {"path": str(data_path), "sha256": source.sha256(), "rows": len(source.ids()),
                 "sources": source.sources(), "max_derived_rows": guard_result.max_derived,
                 "fit": fit},
        "base_model": {"id": cfg.base_model, "revision_requested": cfg.base_revision,
                       "revision_resolved": revision_resolved, "load": cfg.load},
        "config": cfg.as_dict(), "config_sha": cfg.sha(), "seed": cfg.seed,
        "operator_approval_max_derived": approved, "versions": versions,
        "host": platform.node(), "python": platform.python_version(), **extra,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="train_qlora", description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="Phase 3 train.jsonl")
    src.add_argument("--pretokenized", type=Path, help="a modal_train.py pretokenize directory")
    ap.add_argument("--val", type=Path, help="val.jsonl: val loss at each checkpoint")
    ap.add_argument("--tokenizer-dir", type=Path,
                    help="the student's tokenizer.json + chat template (the dir build.py used)")
    ap.add_argument("--out", type=Path, required=True, help="run directory (checkpoints, adapter)")
    ap.add_argument("--log-dir", type=Path, help="default: data/distill/logs/<out name>")
    ap.add_argument("--base-model", default=STUDENT,
                    help="a staged model directory (a real run refuses a hub id without "
                         "HF_HUB_OFFLINE=1 or --allow-hub-download)")
    ap.add_argument("--base-revision", default=None, help="HF revision; resolved to a sha")
    ap.add_argument("--allow-hub-download", action="store_true",
                    help="let a hub-id --base-model download into the HF cache (no disk check)")
    ap.add_argument("--load", choices=("4bit", "16bit"), default="4bit")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=SEQ_LADDER[0], choices=SEQ_LADDER)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--keep-checkpoints", type=int, default=3)
    ap.add_argument("--optim", choices=("adamw_8bit", "adamw_torch"), default="adamw_8bit")
    ap.add_argument("--seed", type=int, default=3407)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="validate data, masks and config; no torch")
    mode.add_argument("--smoke", action="store_true", help="200 examples, 50 steps, loss must drop")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    ap.add_argument("--stop-at", help="ISO time: checkpoint and exit cleanly before it")
    ap.add_argument("--preview", type=int, default=2, help="examples to print in --dry-run")
    ap.add_argument("--smoke-min-drop", type=float, default=0.02)
    ap.add_argument(APPROVAL_FLAG, dest="approved", action="store_true",
                    help="the operator approved training on Max-derived data (OPEN-QUESTIONS #6, "
                         "Q6b); the dataset must also record its build-time approval")
    return ap


def config_from_args(a: argparse.Namespace) -> TrainConfig:
    return TrainConfig(base_model=a.base_model, base_revision=a.base_revision, load=a.load,
                       r=a.r, alpha=a.alpha, lr=a.lr, epochs=a.epochs, max_seq=a.max_seq,
                       grad_accum=a.grad_accum, warmup_ratio=a.warmup_ratio,
                       save_every=a.save_every, keep_checkpoints=a.keep_checkpoints,
                       optim=a.optim, seed=a.seed)


def parse_stop_at(value: str | None) -> float | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise UsageError("--stop-at needs a UTC offset (e.g. 2026-11-01T06:45-05:00)")
    return dt.timestamp() - 120  # leave two minutes for the checkpoint


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("distill")
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def reexec_resume(argv: list[str]) -> None:
    """Replace this process with the same command plus --resume (a fresh CUDA
    context after an OOM)."""
    args = list(argv)
    if "--resume" not in args:
        args.append("--resume")
    for h in logging.getLogger("distill").handlers:
        h.flush()
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *args])


def make_source(a: argparse.Namespace, *, need_encoder: bool) -> Source:
    if a.pretokenized:
        return PretokSource(a.pretokenized)
    encoder = refit = None
    if a.tokenizer_dir:
        encoder = hf_encoder(a.tokenizer_dir)
        try:
            refit = template_refitter(a.tokenizer_dir)
        except (ImportError, OSError, ValueError) as exc:
            log.warning("no template re-fit (%s): rows over a stepped-down length are dropped",
                        type(exc).__name__)
    elif need_encoder:
        raise UsageError("--data needs --tokenizer-dir (or use --pretokenized)")
    return JsonlSource(a.data, encoder, refit)


def dry_run(a: argparse.Namespace, cfg: TrainConfig, run_id: str, log_dir: Path) -> int:
    """Everything that can be checked without torch: rows, masks, tokenization
    (when a tokenizer is given), the length ladder, provenance and config."""
    if a.pretokenized:
        source: Source = PretokSource(a.pretokenized)
        rows = None
    else:
        encoder = hf_encoder(a.tokenizer_dir) if a.tokenizer_dir else None
        js = JsonlSource(a.data, encoder)
        source, rows = js, js.rows
    ds = dataset_record(a.data, a.pretokenized)
    g = guard(source.max_derived(), a.approved, ds["approval"])
    fits = []
    tokenized = rows is None or getattr(source, "encoder", None) is not None
    for i, L in enumerate(cfg.ladder):
        if rows is None or (tokenized and i == 0):
            # every row is tokenized and checked against its recorded spans here
            fits.append({**source.fit(L).as_dict(), "tokenized": True})
        else:
            n = sum(1 for r in rows if r["n_tokens"] <= L)
            fits.append({"max_seq": L, "kept_without_refit": n, "over": len(rows) - n,
                         "tokenized": False})
    print(f"data: {len(source.ids())} rows, sha256 {source.sha256()[:16]}...")
    print(f"sources: {source.sources()}")
    print(g.message().replace("REFUSED", "a real run would be REFUSED"))
    for f in fits:
        print(f"max_seq {f['max_seq']}: {json.dumps({k: v for k, v in f.items() if k != 'max_seq'})}")
    if rows is not None:
        for t in length_table(rows, cfg.ladder):
            print(f"  untruncated originals within {t['max_seq']}: {t['untruncated']}/{len(rows)}")
        js_enc = getattr(source, "encoder", None)
        for row in rows[: max(0, a.preview)]:
            print()
            print(preview(row, ids_spans=tokenize_row(row, js_enc) if js_enc else None))
    steps = math.ceil(len(source.ids()) * cfg.epochs / cfg.grad_accum)
    print(f"\nconfig {cfg.sha()[:12]}: {json.dumps(cfg.as_dict())}")
    print(f"~{steps} optimizer steps at {cfg.grad_accum} examples/step; checkpoint every "
          f"{cfg.save_every}")
    problem = base_source_problem(cfg.base_model, env=os.environ,
                                  allow_download=a.allow_hub_download)
    if problem:
        print(f"note: a real run would refuse this: {problem}")
    if g.data_max_derived:
        print(f"note: the data is Max-derived: a real run's adapter would be "
              f"{adapter_dir(a.out, True)} (private, tagged; Q6b)")
        problem = out_tag_problem(a.out, True)
        if problem:
            print(f"note: a real run would refuse this: {problem}")
    m = manifest(cfg, source, mode="dry-run", run_id=run_id, data_path=a.data or a.pretokenized,
                 approved=a.approved, guard_result=g, revision_resolved=None, fit=fits,
                 versions=package_versions(), extra={"status": "validated"}, dataset=ds)
    write_json(log_dir / "manifest.json", m)
    print(f"manifest: {log_dir / 'manifest.json'}")
    return 0


def main(argv: list[str] | None = None, *, backend_factory: Callable[[TrainConfig], Backend]
         | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)
    if a.smoke:
        cfg = smoke_config(cfg)
    errs = cfg.validate()
    if errs:
        print("config: " + "; ".join(errs), file=sys.stderr)
        return 2
    run_id = a.out.name
    log_dir = a.log_dir or (REPO / "data" / "distill" / "logs" / run_id)
    setup_logging(log_dir)
    try:
        if a.dry_run:
            return dry_run(a, cfg, run_id, log_dir)
        source = make_source(a, need_encoder=True)
        ds = dataset_record(a.data, a.pretokenized)
        g = guard(source.max_derived(), a.approved, ds["approval"])
        log.info("%s", g.message())
        if not g.allowed:
            print(g.message(), file=sys.stderr)
            return 3
        problem = out_tag_problem(a.out, g.data_max_derived)
        if problem:
            raise UsageError(problem)
        if backend_factory is None:  # the real backend loads (and may fetch) the base
            problem = base_source_problem(cfg.base_model, env=os.environ,
                                          allow_download=a.allow_hub_download)
            if problem:
                raise UsageError(problem)
        stop = StopFlag(stop_at=parse_stop_at(a.stop_at))
        prev_handler = signal.signal(signal.SIGTERM, lambda *_: setattr(stop, "reason", "SIGTERM"))
        val = None
        if a.val:
            val = make_source(argparse.Namespace(**{**vars(a), "data": a.val, "pretokenized": None}),
                              need_encoder=True)
            val.fit(SEQ_LADDER[-1])
        only = select_smoke(source.ids(), cfg.seed) if a.smoke else None
        backend = (backend_factory or (lambda c: UnslothBackend(c)))(cfg)
        resolved = cfg.base_revision if cfg.base_revision and re.fullmatch(
            r"[0-9a-f]{40}", cfg.base_revision) else None
        if resolved is None and not (a.resume and (a.out / RUN_STATE).is_file()):
            resolved = resolve_revision(cfg.base_model, cfg.base_revision)
        trainer = Trainer(cfg, source, a.out, backend, stop=stop, only_ids=only, val=val,
                          metrics_path=log_dir / "metrics.jsonl", revision=resolved,
                          checkpoint_manifest=checkpoint_manifest(
                              run_id, a.data or a.pretokenized, source, g))
        mode = "smoke" if a.smoke else "train"
        dest = adapter_dir(a.out, g.data_max_derived)  # tagged for Max-derived data (Q6b)

        def write_manifest(status: str, extra: dict[str, Any]) -> None:
            m = manifest(cfg, source, mode=mode, run_id=run_id, data_path=a.data or a.pretokenized,
                         approved=a.approved, guard_result=g,
                         revision_resolved=trainer.revision,
                         fit=[trainer.fit_report.as_dict()] if trainer.fit_report else [],
                         versions=backend.versions(),
                         extra={"status": status, "seq_history": trainer.run_state.get(
                             "seq_history", []), "resumed_from": trainer.resumed_from,
                             "step": trainer.step, "tokens_seen": trainer.tokens_seen, **extra},
                         dataset=ds)
            write_json(log_dir / "manifest.json", m)
            write_json(a.out / "manifest.json", m)

        try:
            trainer.setup(resume=a.resume)
            write_manifest("running", {})
            if a.smoke:
                return smoke(trainer, a, write_manifest, adapter=dest)
            res = trainer.run()
        except StepDown as sd:
            log.warning("%s; re-executing with --resume", sd)
            reexec_resume(argv)
            return 1  # only reached when execv is mocked
        finally:
            signal.signal(signal.SIGTERM, prev_handler)
        if res.status == "done":
            adapter = trainer.save_adapter(dest)
            write_manifest("done", {"adapter": str(adapter), "peak_gb": backend.peak_memory_gb()})
            log.info("done: %d steps, adapter %s", trainer.step, adapter)
        else:
            write_manifest("stopped", {"reason": res.reason})
        return 0
    except (UsageError, DataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RestoreError as exc:
        log.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def smoke(trainer: Trainer, a: argparse.Namespace,
          write_manifest: Callable[[str, dict[str, Any]], None], *,
          adapter: Path | None = None) -> int:
    """50 steps. At the halfway checkpoint the weights train a few steps on,
    then the checkpoint is restored and must bring back exactly the weights it
    saved; the loss must drop. A stop (SIGTERM, ``--stop-at``) is no verdict:
    the manifest says "stopped" and ``--smoke --resume`` finishes the smoke.

    This is the training half of the runbook's smoke gate only, so the status
    is "smoke-train-pass"; the eval pipeline (merge, quantize, score) still has
    to run on the smoke adapter before a full run."""
    half = min(trainer.cfg.save_every, max(1, trainer.total_steps // 2))
    check: dict[str, Any] | None = None
    res = trainer.run(until_step=half)
    if res.status == "until_step":
        ckpt = trainer.save("smoke-round-trip")  # the halfway checkpoint (or the resumed one)
        res = trainer.run(until_step=trainer.step + SMOKE_MOVE_STEPS)
        if res.status == "until_step":
            check = trainer.verify_resume(ckpt)
            res = trainer.run()
    if res.status == "stopped":
        write_manifest("stopped", {"reason": res.reason, "smoke": {"steps": trainer.step}})
        print(f"smoke stopped ({res.reason}) at step {trainer.step}: no verdict; "
              "finish it with --smoke --resume")
        return 0
    if check is None and trainer.restore_check is not None:
        # resumed too close to the end to train past a checkpoint: the resume
        # itself restored one into freshly loaded weights
        check = {**trainer.restore_check,
                 "passed": bool(trainer.restore_check["ok"] and trainer.restore_check["moved"])}
    round_trip = bool(check and check["passed"])
    ok, first, last = loss_decreased(trainer.losses, min_drop=a.smoke_min_drop)
    adapter = trainer.save_adapter(adapter if adapter is not None else a.out / ADAPTER_DIR)
    passed = ok and round_trip
    write_manifest("smoke-train-pass" if passed else "smoke-train-fail", {
        "smoke": {"steps": trainer.step, "loss_first10": first, "loss_last10": last,
                  "loss_decreased": ok, "checkpoint_round_trip": round_trip,
                  "round_trip": check, "eval_pipeline": "not run"},
        "adapter": str(adapter)})
    print(f"smoke (training): {trainer.step} steps, loss {first:.4f} -> {last:.4f} "
          f"({'decreased' if ok else 'DID NOT DECREASE'}); checkpoint round trip "
          f"{'ok' if round_trip else 'FAILED'}")
    if passed:
        print(f"the smoke gate also needs the eval pipeline (merge, quantize, score) on {adapter}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
