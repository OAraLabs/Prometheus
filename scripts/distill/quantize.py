#!/usr/bin/env python3
"""Phase 6 quantize: merge the LoRA, GGUF at a pinned llama.cpp, train-only imatrix, Q4_K_M + Q5_K_M.

Runs on the GPU host after training, inside a window. Prints the plan and runs
nothing unless ``--execute``.

    # The plan: every step and command, plus the calibration's train-only check
    # over whatever inputs are present. Nothing runs, nothing is written.
    python3 scripts/distill/quantize.py run --adapter ~/distill/phase5/<run>/adapter \\
        --base ~/distill/models/Qwen3.5-9B --data ~/distill/phase3 \\
        --evalsets ~/distill/phase3/evalsets --out ~/distill/phase6/<run>
    # Do it, on the GPU host inside a window: the same arguments plus --execute,
    # --detach (a distill-quantize-<UTC stamp> systemd user unit, which a
    # `setups.py apply business` restore stops) and --stop-at (the window's end).
    # A re-run resumes (finished steps are kept); an --out holding outputs made
    # from different inputs is refused.
    python3 scripts/distill/quantize.py run ... --execute --detach \\
        --stop-at 2026-11-01T06:45-05:00 --i-have-operator-approval-for-max-derived-data
    # The base model through the same recipe (the Phase 4 base 9B GGUF):
    python3 scripts/distill/quantize.py run --no-adapter --base ... --out ~/distill/phase6/base-9b
    # The calibration text alone (stdlib only, no GPU), on the host with the splits:
    python3 scripts/distill/quantize.py calibrate --data ... --evalsets ... --out ... [--execute]
    # The llama-server line for Phase 6 scoring at 256k (prints, never runs):
    python3 scripts/distill/quantize.py serve --gguf ~/distill/phase6/<run>/gguf/<run>-Q4_K_M.gguf

Nothing here develops a model from teacher (Max-derived) text without the
operator: the imatrix is computed from the calibration text, so ``run
--execute`` refuses (exit 3) when any calibration row is Max-derived unless
``--i-have-operator-approval-for-max-derived-data`` is passed
(DISTILL-OPEN-QUESTIONS #6). Provenance is train_qlora.py's rule, which fails
closed: a row naming Max counts, and so does any row unless build.py's
``report.json`` beside ``train.jsonl`` names a non-Max teacher or the row says
``max_derived: false``. Writing the calibration text (``calibrate``) is data
preparation and is not held.

The window: ``--stop-at`` (and SIGTERM, which a ``setups.py`` restore or the
weekly flywheel sends) stops the run, killing a running step, whose
``.partial`` output is redone next time; it exits 75 and a re-run resumes.

Steps (``--execute``)
---------------------
1. preflight: the llama.cpp checkout's HEAD is the pinned commit and its
   converter and gguf-py are unmodified; the binaries report that build
   (``--version``); the converter registers the base's architecture
   (``--print-supported-models``); free disk covers the outputs.
2. calibrate: the imatrix text and a held-back perplexity slice, both from the
   TRAIN split only (below). First, so a leak stops the run before any merge.
   Then the provenance gate above. ``run.json`` (the inputs' fingerprint) is
   written only after this step: until an output exists, a re-run with the
   fix a refusal suggests is not refused for changing an input.
3. merge (skipped with ``--no-adapter``): PEFT ``merge_and_unload`` into the
   base at bf16 (or fp16), on the CPU by default (``--merge-device auto`` lets
   accelerate split GPU / CPU / disk). Refused when the base loads with
   missing weights (they would be random), when not every LoRA module in the
   adapter attaches to the loaded base, or when any attached LoRA B matrix is
   zero: PEFT injects fresh layers (B = 0) for every targeted module and then
   loads what matches, so a key mismatch leaves zero B's and a merge that
   silently does nothing for those modules. Tokenizer files come from
   the base; the chat template from ``--template-dir``, the one Phase 3 rendered
   the training text with, so the GGUF serves the template the model learned.
   With ``--no-adapter`` the base is read through a symlink view carrying the
   same template, so base and tuned GGUFs embed the same one.
4. convert: ``convert_hf_to_gguf.py --outtype bf16 --no-mtp``. transformers
   drops Qwen3.5's MTP head on load (and a tuned trunk would leave the base's
   head stale) while ``config.json`` still declares ``mtp_num_hidden_layers``;
   without ``--no-mtp`` the converter declares a nextn layer it has no
   tensors for. The base goes through the same flag, so both share a layout.
5. imatrix: ``llama-imatrix --parse-special`` on the bf16 GGUF, ``-c`` = ``-b`` =
   ``--imatrix-ctx`` so each chunk is one sequence (no multi-sequence
   recurrent state for the hybrid linear-attention layers).
6. quantize: ``llama-quantize --imatrix`` for each of ``--quants``.
7. sanity: each GGUF (bf16 reference included) loads in ``llama-perplexity``
   and scores the perplexity slice; a quant passes when its perplexity is
   finite, under ``--max-ppl`` (a broken-model ceiling) and within
   ``--max-ppl-ratio`` of the bf16 reference, and its metadata reads back
   right: the architecture, the file type, the trained chat template, no
   nextn layers, and a context length that holds the 256k serve line.
   llama-perplexity tokenizes special-token text as plain text (it has no
   ``--parse-special``); every model scores the same tokens, so the ratio holds.
   The metadata is read with gguf-py under ``--convert-python`` (the
   interpreter the converter ran with), in a child process.
8. manifest: ``quantize-manifest.json``: inputs, settings, each command, every
   GGUF / imatrix / calibration file with size + sha256, the sanity results and
   the serve line for each quant.

The llama.cpp pin
-----------------
``--llama-cpp-commit`` defaults to ``9d57ce456``, the build the hosts serve
with. Checked 2026-10-30 against that commit's source: ``conversion/qwen.py``
registers ``Qwen3_5ForConditionalGeneration`` and ``Qwen3_5ForCausalLM`` (arch
``qwen35``; the student's config names the first), the text converter drops
``visual.`` tensors and strips ``language_model.``, ``--no-mtp`` exists,
``llama-imatrix`` has ``--parse-special`` and writes GGUF imatrix files, and
``-fa`` takes a value (``on|off|auto``): a bare ``-fa`` in front of ``-ctk``
would swallow ``-ctk``, so the serve line says ``-fa on``. Another commit is
allowed; preflight then decides support from that converter's own list.

Calibration (train only)
------------------------
Reads ``<data>/train.jsonl`` (``build.py``'s training rows: ``split``, ``id``,
``task_id``, ``group_id``, rendered ``text``). Refused outright when a row is
not ``split: train``, or shares an ``id`` / ``task_id`` / ``group_id`` /
``trace_id`` with any row of the eval and val files (``val.jsonl``,
``val-cases.jsonl``, ``eval-tools.jsonl`` in ``--data``, required; the
``--evalsets`` files, ``ungated/eval-longctx-tools.jsonl`` included, and any
``--eval`` file when present): Phase 3's leak gate should have made that
impossible, so it is a broken build, not a row to skip. Rows that overlap
eval-injection (its hash manifest, when present) are left out and counted.
Rows are taken round-robin over tasks in a seeded order until
``--calib-tokens``; the next rows form the perplexity slice. The final check
repeats over the ids actually used. Each row's text is the exact rendered
chat-template text, special tokens included, as the model sees it.

A run whose ``--data`` has no ``train.jsonl`` uses the calibration
``calibrate --execute`` left in ``<out>/calib``, and only when it was made
with the run's own ``--seed``, ``--calib-tokens``, ``--imatrix-ctx``,
``--ppl-ctx`` and ``--ppl-chunks``, and from the train split this model goes
with: the one the adapter's trainer manifest (``<adapter>/../manifest.json``)
names, or ``--train-sha256``. With the splits present, a trainer manifest that
names another train split is refused too.

Exit: 0 done (or plan printed, or the detached unit started), 1 a sanity check
failed, 2 usage, input or step error, 3 refused (Max-derived calibration text
without the operator's approval), 75 stopped by ``--stop-at`` or SIGTERM
(finished steps are kept; the same command resumes).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

MANIFEST_SCHEMA = "distill-quantize-manifest/1"
CALIB_SCHEMA = "distill-calibration/1"
RUN_SCHEMA = "distill-quantize-run/1"
TRAIN_MANIFEST_SCHEMA = "distill-train-manifest/1"  # train_qlora.py's run manifest

EXIT_SANITY = 1
EXIT_ERROR = 2
EXIT_REFUSED = 3
EXIT_PAUSED = 75  # the weekly flywheel's incomplete_exit_code: paused, not failed
# train_qlora.py's APPROVAL_FLAG: the one operator say-so for Max-derived data.
APPROVAL_FLAG = "--i-have-operator-approval-for-max-derived-data"
# Detached runs are distill-* units, which a setups.py restore stops.
UNIT_PREFIX = "distill-quantize-"
STOP_MARGIN_S = 60  # stop this long before --stop-at: the kill and the exit

DEFAULT_COMMIT = "9d57ce456"
DEFAULT_LLAMA_CPP = "~/llama.cpp"
DEFAULT_BASE = "~/distill/models/Qwen3.5-9B"
DEFAULT_TEMPLATE_DIR = "~/distill/models/qwen3.5-9b-tokenizer"
DEFAULT_DATA = "~/distill/phase3"
DEFAULT_EVALSETS = "~/distill/phase3/evalsets"
DEFAULT_OUT_ROOT = "~/distill/phase6"
DEFAULT_QUANTS = ("Q4_K_M", "Q5_K_M")

# The Phase 4/6 serve line: 256k, one slot, flash attention, q8_0 KV cache.
SERVE_CTX = 262144
SERVE_PARALLEL = 1
SERVE_KV_TYPE = "q8_0"

# llama_ftype values (llama.h) the metadata check expects back.
FILE_TYPES = {"F16": 1, "Q4_K_S": 14, "Q4_K_M": 15, "Q5_K_S": 16, "Q5_K_M": 17,
              "Q6_K": 18, "Q8_0": 7, "BF16": 32}
# Size of a quant relative to the 16-bit file, for the disk preflight only.
SIZE_RATIO = {"Q4_K_S": 0.29, "Q4_K_M": 0.31, "Q5_K_S": 0.35, "Q5_K_M": 0.36,
              "Q6_K": 0.41, "Q8_0": 0.54}
DISK_MARGIN = 5 * 1024**3

# HF architectures -> the GGUF arch the converter writes. The Qwen3.5 rows
# were read from conversion/qwen.py at 9d57ce456 on 2026-10-30.
GGUF_ARCH = {
    "Qwen3_5ForConditionalGeneration": "qwen35",
    "Qwen3_5ForCausalLM": "qwen35",
    "Qwen3_5MoeForConditionalGeneration": "qwen35moe",
    "Qwen3_5MoeForCausalLM": "qwen35moe",
}
VERIFIED_SUPPORT = {DEFAULT_COMMIT: frozenset(GGUF_ARCH)}

# Eval / val files whose ids must never reach the calibration text.
DATA_EVAL_FILES = ("val.jsonl", "val-cases.jsonl", "eval-tools.jsonl")
REQUIRED_EVAL_FILES = ("val.jsonl", "eval-tools.jsonl")
# build_eval_sets.py's OUT_FILES, relative to --evalsets: its long-context
# tool-call cases sit in ungated/, out of the gated sets' top level.
EVALSET_FILES = ("eval-longctx.jsonl", "ungated/eval-longctx-tools.jsonl",
                 "eval-injection.jsonl", "sanity.jsonl")
INJECTION_MANIFEST = "eval-injection.manifest.json"
KEY_FIELDS = ("id", "task_id", "group_id", "trace_id")
#: Phase 1's measured density; only for rows that carry no token count.
CHARS_PER_TOKEN = 3.93

# Files the merged checkpoint takes from the base (config.json and the
# weights come from save_pretrained).
BASE_SIDE_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "generation_config.json",
                   "preprocessor_config.json", "video_preprocessor_config.json")
TEMPLATE_FILE = "chat_template.jinja"

_PPL = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+(?:[eE][+-]?\d+)?|nan|inf)", re.I)
_BUILD = re.compile(r"\((?:b\d+-)?([0-9a-f]{7,40})\)")


class QuantizeError(RuntimeError):
    """A step that cannot (or must not) go on."""


class CalibrationError(QuantizeError):
    """The calibration text cannot be shown to be train-only."""


class UsageError(QuantizeError):
    """The command line does not allow this."""


class Refused(QuantizeError):
    """Max-derived calibration text without the operator's approval."""


class Paused(BaseException):
    """--stop-at or SIGTERM. A BaseException, so no ``except Exception`` on the
    way (a metadata read, transformers) can take it for a failed step."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p)))


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seeded(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(path: Path, data: Any) -> None:
    _write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QuantizeError(f"{path.name}:{n}: not JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise QuantizeError(f"{path.name}:{n}: not a JSON object")
            yield row


def _partial(path: Path) -> Path:
    """Where a step writes before its output is complete (renamed on success)."""
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.") or "model"


# ---------------------------------------------------------------------------
# Commands (pure: the same lists are printed by the plan and run by --execute)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaCpp:
    dir: Path
    commit: str
    bin_dir: Path

    @property
    def converter(self) -> Path:
        return self.dir / "convert_hf_to_gguf.py"

    def tool(self, name: str) -> Path:
        return self.bin_dir / name


def git_head_cmd(llama: LlamaCpp) -> list[str]:
    return ["git", "-C", str(llama.dir), "rev-parse", "HEAD"]


def git_dirty_cmd(llama: LlamaCpp) -> list[str]:
    # The converter and gguf-py are what write the GGUF; a local edit to either
    # makes the output something the commit does not describe.
    return ["git", "-C", str(llama.dir), "status", "--porcelain", "--",
            "convert_hf_to_gguf.py", "conversion", "gguf-py"]


def version_cmd(llama: LlamaCpp, tool: str) -> list[str]:
    return [str(llama.tool(tool)), "--version"]


def supported_models_cmd(python: str, llama: LlamaCpp) -> list[str]:
    return [python, str(llama.converter), "--print-supported-models"]


def convert_cmd(python: str, llama: LlamaCpp, model_dir: Path, outfile: Path, *,
                outtype: str = "bf16", no_mtp: bool = True,
                model_name: str | None = None) -> list[str]:
    cmd = [python, str(llama.converter), str(model_dir), "--outtype", outtype,
           "--outfile", str(outfile)]
    if no_mtp:
        cmd.append("--no-mtp")
    if model_name:
        cmd += ["--model-name", model_name]
    return cmd


def imatrix_cmd(llama: LlamaCpp, model: Path, calib: Path, out: Path, *, ctx: int,
                chunks: int, ngl: str = "99") -> list[str]:
    return [str(llama.tool("llama-imatrix")), "-m", str(model), "-f", str(calib),
            "-o", str(out), "--output-format", "gguf", "-c", str(ctx), "-b", str(ctx),
            "--chunks", str(chunks), "--parse-special", "-ngl", str(ngl)]


def quantize_cmd(llama: LlamaCpp, src: Path, imatrix: Path, dst: Path, qtype: str) -> list[str]:
    return [str(llama.tool("llama-quantize")), "--imatrix", str(imatrix), str(src), str(dst),
            qtype]


def perplexity_cmd(llama: LlamaCpp, model: Path, text: Path, *, ctx: int, chunks: int,
                   ngl: str = "99") -> list[str]:
    return [str(llama.tool("llama-perplexity")), "-m", str(model), "-f", str(text),
            "-c", str(ctx), "-b", str(ctx), "--chunks", str(chunks), "-ngl", str(ngl)]


def serve_command(gguf: str | Path, *, server: str | Path = "llama-server",
                  alias: str | None = None, host: str = "127.0.0.1", port: int = 8080,
                  ngl: str = "99", ctx: int = SERVE_CTX, parallel: int = SERVE_PARALLEL,
                  extra: tuple[str, ...] | list[str] = ()) -> list[str]:
    """The llama-server line Phase 6 scores a GGUF with: the runbook's
    ``-c 262144 -np 1 -fa -ctk q8_0 -ctv q8_0``, with ``-fa on`` (the flag
    takes a value at the pinned build) and ``--jinja`` (tool calls need the
    GGUF's own template). The base 9B is served with the same line."""
    cmd = [str(server), "-m", str(gguf), "-c", str(ctx), "-np", str(parallel),
           "-fa", "on", "-ctk", SERVE_KV_TYPE, "-ctv", SERVE_KV_TYPE, "--jinja",
           "-ngl", str(ngl), "--host", host, "--port", str(port)]
    if alias:
        cmd += ["--alias", alias]
    return cmd + list(extra)


# ---------------------------------------------------------------------------
# Calibration: train split only
# ---------------------------------------------------------------------------


def row_keys(row: dict[str, Any]) -> dict[str, str]:
    """A row's identity keys, top level first, then ``meta`` (eval cases)."""
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    out: dict[str, str] = {}
    for k in KEY_FIELDS:
        v = row.get(k) if k in row else meta.get(k)
        if isinstance(v, str) and v:
            out[k] = v
    return out


@dataclass
class EvalDenylist:
    """Every id / task / group / trace key of the eval and val files."""

    keys: dict[str, dict[str, str]] = field(default_factory=dict)  # kind -> value -> file
    files: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[Path]) -> "EvalDenylist":
        d = cls()
        for p in paths:
            n = 0
            for row in read_jsonl(p):
                n += 1
                for kind, value in row_keys(row).items():
                    d.keys.setdefault(kind, {}).setdefault(value, p.name)
            d.files.append({"name": p.name, "rows": n, "sha256": sha256_file(p)})
        return d

    def hits(self, row: dict[str, Any]) -> list[tuple[str, str, str]]:
        return [(kind, value, self.keys[kind][value])
                for kind, value in row_keys(row).items()
                if value in self.keys.get(kind, {})]

    def summary(self) -> dict[str, Any]:
        return {"files": self.files,
                "keys": {k: len(v) for k, v in sorted(self.keys.items())}}


def eval_files(data: Path, evalsets: Path | None, extra: list[Path]) -> list[Path]:
    """The files whose rows must never be calibration text. The two Phase 3
    always writes are required: without them nothing shows train-only."""
    missing = [n for n in REQUIRED_EVAL_FILES if not (data / n).is_file()]
    if missing:
        raise CalibrationError(f"{', '.join(missing)} not found in {data}: the calibration "
                               "cannot be checked against the eval / val splits")
    out = [data / n for n in DATA_EVAL_FILES if (data / n).is_file()]
    if evalsets is not None:
        out += [evalsets / n for n in EVALSET_FILES if (evalsets / n).is_file()]
    for p in extra:
        if not p.is_file():
            raise CalibrationError(f"--eval {p} not found")
        out.append(p)
    return out


def _sibling(name: str, file: str) -> ModuleType:
    """A script next to this one, loaded once under the name its other users
    (modal_train.py, the tests) load it by. Both are stdlib-only at import."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def load_injection_manifest(path: Path) -> Any:
    """eval-injection's hash-only manifest, through build_eval_sets.py's own
    class (read only; nothing of the eval text is held)."""
    return _sibling("distill_build_eval_sets", "build_eval_sets.py").InjectionManifest.load(path)


def _trainer() -> ModuleType:
    """train_qlora.py: its provenance rule (``is_max_derived``) is the one rule."""
    return _sibling("distill_train_qlora", "train_qlora.py")


@dataclass
class _Cand:
    id: str
    task: str
    text: str
    tokens: int
    source: str = "unknown"
    max_derived: bool = True


@dataclass
class Calibration:
    text: str
    ppl_text: str
    calib_ids: list[str]
    ppl_ids: list[str]
    info: dict[str, Any]


def _round_robin(rows: list[_Cand], seed: int) -> list[_Cand]:
    """One row per task per round, tasks and rows in a seeded order, so the
    text covers as many tasks as the budget allows before any task repeats."""
    by_task: dict[str, list[_Cand]] = collections.defaultdict(list)
    for r in rows:
        by_task[r.task].append(r)
    for t in by_task.values():
        t.sort(key=lambda r: _seeded(seed, "row:" + r.id))
    tasks = sorted(by_task, key=lambda t: _seeded(seed, "task:" + t))
    out: list[_Cand] = []
    depth = 0
    while True:
        added = False
        for t in tasks:
            if depth < len(by_task[t]):
                out.append(by_task[t][depth])
                added = True
        if not added:
            return out
        depth += 1


def _join(texts: list[str]) -> str:
    return "".join(t if t.endswith("\n") else t + "\n" for t in texts)


def build_calibration(train: Path, deny: EvalDenylist, *, seed: int, calib_tokens: int,
                      ppl_tokens: int, min_calib_tokens: int, min_ppl_tokens: int,
                      injection: Any = None) -> Calibration:
    trainer = _trainer()
    teacher_max = trainer.builder_teacher_is_max(train)  # build.py's report, beside the data
    not_train: list[str] = []
    leaks: list[str] = []
    rows: list[_Cand] = []
    no_text = read = 0
    for n, row in enumerate(read_jsonl(train), 1):
        read += 1
        rid = row.get("id")
        where = rid if isinstance(rid, str) and rid else f"line {n}"
        if row.get("split") != "train":
            not_train.append(f"{where} (split={row.get('split')!r})")
            continue
        hits = deny.hits(row)
        if hits:
            kind, value, fname = hits[0]
            leaks.append(f"{where}: {kind} {value} is in {fname}")
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            no_text += 1
            continue
        if not isinstance(rid, str) or not rid:
            raise CalibrationError(f"{train.name}:{n}: a train row without an id")
        nt = row.get("n_tokens")
        tokens = nt if isinstance(nt, int) and nt > 0 else math.ceil(len(text) / CHARS_PER_TOKEN)
        task = row.get("task_id")
        rows.append(_Cand(rid, task if isinstance(task, str) and task else rid, text, tokens,
                          trainer.row_source(row), trainer.is_max_derived(row, teacher_max)))
    if not_train:
        raise CalibrationError(f"{len(not_train)} rows of {train.name} are not the train split "
                               f"(e.g. {'; '.join(not_train[:3])}); refusing")
    if leaks:
        raise CalibrationError(
            f"{len(leaks)} train rows share a key with the eval / val sets "
            f"(e.g. {'; '.join(leaks[:3])}). Phase 3's leak gate should have refused this "
            "build; refusing to calibrate on it")

    calib: list[_Cand] = []
    ppl: list[_Cand] = []
    ct = pt = dropped_injection = 0
    for c in _round_robin(rows, seed):
        if ct >= calib_tokens and pt >= ppl_tokens:
            break
        if injection is not None and injection.record_overlaps({"text": c.text}):
            dropped_injection += 1
            continue
        if ct < calib_tokens:
            calib.append(c)
            ct += c.tokens
        else:
            ppl.append(c)
            pt += c.tokens
    if ct < min_calib_tokens:
        raise CalibrationError(f"the train split gives {ct} calibration tokens; llama-imatrix "
                               f"needs at least {min_calib_tokens} at this context")
    if pt < min_ppl_tokens:
        raise CalibrationError(f"{pt} tokens left for the perplexity slice after calibration; "
                               f"need {min_ppl_tokens} (lower --calib-tokens or --ppl-chunks)")

    # The guarantee, once more over exactly what is used.
    calib_ids = [c.id for c in calib]
    ppl_ids = [c.id for c in ppl]
    used = set(calib_ids) | set(ppl_ids)
    bad = sorted(used & set(deny.keys.get("id", {})))
    if bad:
        raise CalibrationError(f"internal: eval ids in the calibration ({bad[:3]})")
    if set(calib_ids) & set(ppl_ids):
        raise CalibrationError("internal: a row is in both the calibration and the ppl slice")

    text, ppl_text = _join([c.text for c in calib]), _join([c.text for c in ppl])
    info = {
        "schema": CALIB_SCHEMA, "source": "train split only", "seed": seed,
        "train": {"file": train.name, "sha256": sha256_file(train), "rows_read": read,
                  "rows_usable": len(rows), "rows_without_text": no_text,
                  "tasks": len({r.task for r in rows})},
        "eval_check": {**deny.summary(), "train_rows_sharing_a_key": 0,
                       "injection_manifest": injection is not None,
                       "dropped_injection_overlap": dropped_injection},
        "calibration": {"rows": len(calib), "tasks": len({c.task for c in calib}),
                        "tokens_est": ct, "sha256": sha256_text(text)},
        "ppl_slice": {"rows": len(ppl), "tasks": len({c.task for c in ppl}),
                      "tokens_est": pt, "sha256": sha256_text(ppl_text)},
        # The imatrix develops the model from the calibration rows: the
        # provenance gate (OPEN-QUESTIONS #6) reads this.
        "provenance": {
            "rule": "train_qlora.is_max_derived (Max-derived unless shown otherwise)",
            "calibration_max_derived": sum(c.max_derived for c in calib),
            "ppl_max_derived": sum(c.max_derived for c in ppl),
            "sources": dict(sorted(collections.Counter(c.source for c in calib + ppl).items())),
        },
    }
    return Calibration(text, ppl_text, calib_ids, ppl_ids, info)


def calib_settings(s: "Settings") -> dict[str, int]:
    """What a calibration is made for: the text (seed, sizes) and the context
    floors llama-imatrix and llama-perplexity hold it to."""
    return {"seed": s.seed, "calib_tokens": s.calib_tokens, "imatrix_ctx": s.imatrix_ctx,
            "ppl_ctx": s.ppl_ctx, "ppl_chunks": s.ppl_chunks}


def calibration_from_settings(s: "Settings") -> Calibration:
    files = eval_files(s.data, s.evalsets, s.extra_eval)
    deny = EvalDenylist.load(files)
    injection = None
    manifest = s.injection_manifest
    if manifest is None and s.evalsets is not None and (s.evalsets / INJECTION_MANIFEST).is_file():
        manifest = s.evalsets / INJECTION_MANIFEST
    if manifest is not None:
        injection = load_injection_manifest(manifest)
    cal = build_calibration(
        s.data / "train.jsonl", deny, seed=s.seed, calib_tokens=s.calib_tokens,
        ppl_tokens=(s.ppl_chunks + 1) * s.ppl_ctx, min_calib_tokens=2 * s.imatrix_ctx,
        min_ppl_tokens=2 * s.ppl_ctx, injection=injection)
    cal.info["settings"] = calib_settings(s)
    return cal


def write_calibration(cal: Calibration, calib_dir: Path) -> dict[str, Any]:
    _write_atomic(calib_dir / "calibration.txt", cal.text)
    _write_atomic(calib_dir / "ppl.txt", cal.ppl_text)
    record = {**cal.info, "written_at": _now(), "calibration_ids": cal.calib_ids,
              "ppl_ids": cal.ppl_ids}
    _write_json(calib_dir / "calibration.json", record)
    return record


def recorded_calibration(calib_dir: Path) -> dict[str, Any] | None:
    """A calibration ``calibrate --execute`` wrote, if its files still match."""
    rec_path = calib_dir / "calibration.json"
    if not rec_path.is_file():
        return None
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    for name, key in (("calibration.txt", "calibration"), ("ppl.txt", "ppl_slice")):
        f = calib_dir / name
        if not f.is_file() or sha256_file(f) != rec[key]["sha256"]:
            raise CalibrationError(f"{f} does not match {rec_path.name}; rebuild it")
    return rec


# ---------------------------------------------------------------------------
# Merge (heavy dependencies imported lazily: the module runs without them)
# ---------------------------------------------------------------------------


def _import_merge_deps() -> tuple[Any, Any, Any]:
    import peft
    import torch
    import transformers

    return torch, transformers, peft


def safetensors_keys(path: Path) -> list[str]:
    """Tensor names from a .safetensors header (8-byte length + JSON), stdlib only."""
    with open(path, "rb") as fh:
        head = fh.read(8)
        if len(head) != 8:
            raise QuantizeError(f"{path}: not a safetensors file")
        (n,) = struct.unpack("<Q", head)
        meta = json.loads(fh.read(n))
    return [k for k in meta if k != "__metadata__"]


def adapter_lora_modules(adapter: Path) -> list[str]:
    """The modules an adapter carries LoRA weights for (the ``lora_A`` side)."""
    f = adapter / "adapter_model.safetensors"
    if not f.is_file():
        raise QuantizeError(f"{f} not found (a safetensors PEFT adapter is expected)")
    mods = set()
    for k in safetensors_keys(f):
        if ".lora_A." in k:
            mods.add(k.split(".lora_A.")[0])
        elif k.endswith(".lora_embedding_A"):
            mods.add(k[: -len(".lora_embedding_A")])
    return sorted(mods)


def resolve_model_class(choice: str, base_cfg: dict[str, Any], modules: list[str]) -> str:
    """The transformers auto class to load the base with: the one the adapter
    was trained on. A LoRA over ``language_model.`` modules was trained on the
    multimodal wrapper; anything else on the text-only causal LM."""
    if choice == "causal-lm":
        return "AutoModelForCausalLM"
    if choice == "image-text-to-text":
        return "AutoModelForImageTextToText"
    arch = (base_cfg.get("architectures") or [""])[0]
    if arch.endswith("ForConditionalGeneration") and any(".language_model." in m for m in modules):
        return "AutoModelForImageTextToText"
    return "AutoModelForCausalLM"


def _dtype_kw(transformers: Any) -> str:
    # transformers 4.56 renamed torch_dtype to dtype (the old name still warns).
    nums = [int(x) for x in re.findall(r"\d+", str(getattr(transformers, "__version__", "0")))[:2]]
    return "dtype" if tuple(nums + [0, 0])[:2] >= (4, 56) else "torch_dtype"


def _has_adapter(module: Any, name: str = "default") -> bool:
    for attr in ("lora_A", "lora_embedding_A"):
        d = getattr(module, attr, None)
        if d is not None and name in d:
            return True
    return False


def _b_nonzero(module: Any, name: str = "default") -> bool:
    b = getattr(module, "lora_B", None)
    w = None
    if b is not None and name in b:
        w = b[name].weight
    else:
        eb = getattr(module, "lora_embedding_B", None)
        if eb is not None and name in eb:
            w = eb[name]
    return w is not None and float(w.detach().abs().max().item()) > 0.0


def read_template(template_dir: Path) -> str:
    """``chat_template.jinja`` when present, else the string in
    ``tokenizer_config.json`` (render.py's order)."""
    jinja = template_dir / TEMPLATE_FILE
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    cfg_path = template_dir / "tokenizer_config.json"
    tpl = None
    if cfg_path.is_file():
        tpl = json.loads(cfg_path.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(tpl, list):
            tpl = next((t.get("template") for t in tpl if t.get("name") == "default"), None)
    if not isinstance(tpl, str) or not tpl:
        raise QuantizeError(f"no chat template in {template_dir}")
    return tpl


def install_template(dst: Path, base: Path, template_dir: Path) -> dict[str, Any]:
    """Put the trained-on chat template into a checkpoint dir, in both places
    a converter may read it, after checking the tokenizer is the same one."""
    tb, tt = base / "tokenizer.json", template_dir / "tokenizer.json"
    if tb.is_file() and tt.is_file() and sha256_file(tb) != sha256_file(tt):
        raise QuantizeError(f"{tt} differs from the base's tokenizer.json: the training text "
                            "was tokenized for another vocabulary")
    tpl = read_template(template_dir)
    jinja = dst / TEMPLATE_FILE
    if jinja.is_symlink():
        jinja.unlink()
    jinja.write_text(tpl, encoding="utf-8")
    cfg_path = dst / "tokenizer_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "chat_template" in cfg:
            cfg["chat_template"] = tpl
            cfg_path.unlink()  # a symlink into the base must not be written through
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    return {"source": str(template_dir), "sha256": sha256_text(tpl)}


def prepare_base_view(base: Path, view: Path, template_dir: Path) -> dict[str, Any]:
    """``--no-adapter``: the base read through symlinks, with the template
    installed the way a merge installs it, so both GGUFs embed the same one."""
    tmp = view.with_name(view.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f in sorted(base.iterdir()):
        if f.is_file():
            (tmp / f.name).symlink_to(f.resolve())
    info = {"template": install_template(tmp, base, template_dir)}
    tmp.rename(view)
    return info


def merge_adapter(base: Path, adapter: Path, out: Path, *, template_dir: Path,
                  dtype: str = "bf16", device: str = "cpu", model_class: str = "auto",
                  max_gpu_gib: int = 20, max_cpu_gib: int = 48) -> dict[str, Any]:
    torch, transformers, peft = _import_merge_deps()
    acfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if str(acfg.get("peft_type", "LORA")).upper() != "LORA":
        raise QuantizeError(f"adapter is {acfg.get('peft_type')}, not LoRA")
    modules = adapter_lora_modules(adapter)
    if not modules:
        raise QuantizeError(f"{adapter} holds no LoRA weights")
    base_cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    cls_name = resolve_model_class(model_class, base_cfg, modules)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    kwargs: dict[str, Any] = {_dtype_kw(transformers): torch_dtype, "low_cpu_mem_usage": True,
                              "output_loading_info": True}
    peft_kwargs: dict[str, Any] = {}
    offload = out.with_name(out.name + ".offload")
    if device == "auto":
        kwargs.update(device_map="auto", offload_folder=str(offload),
                      max_memory={0: f"{max_gpu_gib}GiB", "cpu": f"{max_cpu_gib}GiB"})
        peft_kwargs["offload_folder"] = str(offload)
    else:
        kwargs["device_map"] = "cpu"
    model, loading = getattr(transformers, cls_name).from_pretrained(str(base), **kwargs)
    missing = list(loading.get("missing_keys") or [])
    if missing:
        raise QuantizeError(
            f"{cls_name} loaded the base with {len(missing)} missing weights "
            f"(e.g. {missing[:3]}): they would be random. Try --model-class")
    unexpected = list(loading.get("unexpected_keys") or [])
    pm = peft.PeftModel.from_pretrained(model, str(adapter), is_trainable=False, **peft_kwargs)
    attached = [name for name, m in pm.named_modules() if _has_adapter(m)]
    if len(attached) != len(modules):
        raise QuantizeError(
            f"the adapter has LoRA weights for {len(modules)} modules but {len(attached)} "
            f"attached to the base loaded as {cls_name}; merging would silently drop the "
            "rest. Load the class the adapter was trained on (--model-class)")
    zero = [name for name, m in pm.named_modules() if _has_adapter(m) and not _b_nonzero(m)]
    if zero:
        raise QuantizeError(
            f"{len(zero)} of {len(attached)} LoRA B matrices are zero (e.g. {zero[0]}): their "
            "weights did not load (adapter keys that do not match the base) or never trained")
    nonzero = len(attached)
    merged = pm.merge_and_unload(safe_merge=True)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    merged.save_pretrained(str(tmp), safe_serialization=True, max_shard_size="5GB")
    for name in BASE_SIDE_FILES:
        if (base / name).is_file():
            shutil.copy2(base / name, tmp / name)
    template = install_template(tmp, base, template_dir)
    info = {
        # merge_device, not plain device: tests/test_config_drift.py counts any quoted key name
        # as a reader, so that one would pass for a reader of the template's whisper.device.
        "model_class": cls_name, "dtype": dtype, "merge_device": device,
        "lora_modules": len(modules), "attached": len(attached), "b_nonzero": nonzero,
        "unexpected_base_keys": len(unexpected),
        "unexpected_base_key_prefixes": sorted({k.split(".")[0] for k in unexpected})[:10],
        "adapter_config": {k: acfg.get(k) for k in ("r", "lora_alpha", "lora_dropout",
                                                     "use_rslora", "use_dora", "target_modules",
                                                     "base_model_name_or_path")},
        "template": template,
    }
    _write_json(tmp / "merge-info.json", info)
    del merged, pm, model
    gc.collect()
    tmp.rename(out)
    if offload.exists():
        shutil.rmtree(offload)
    return info


# ---------------------------------------------------------------------------
# GGUF metadata (read back through the pinned checkout's gguf-py)
# ---------------------------------------------------------------------------


# Run as ``<convert-python> -c META_SCRIPT <gguf-py dir> <file>``: gguf-py needs
# numpy, which the converter's interpreter has and this one may not.
META_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import gguf

reader = gguf.GGUFReader(sys.argv[2])


def val(key):
    f = reader.get_field(key)
    if f is None:
        return None
    if hasattr(f, "contents"):
        return f.contents()
    part = f.parts[f.data[0]]
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(part).decode("utf-8")
    return part.tolist()[0]


arch = val("general.architecture")
print(json.dumps({
    "architecture": arch,
    "file_type": val("general.file_type"),
    "name": val("general.name"),
    "context_length": val(arch + ".context_length") if arch else None,
    "nextn_predict_layers": val(arch + ".nextn_predict_layers") if arch else None,
    "chat_template": val("tokenizer.chat_template"),
}, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
"""


def read_gguf_metadata(llama_dir: Path, path: Path, python: str = sys.executable) -> dict[str, Any]:
    cp = subprocess.run([python, "-c", META_SCRIPT, str(llama_dir / "gguf-py"), str(path)],
                        capture_output=True, text=True, check=False)
    lines = (cp.stdout or "").strip().splitlines()
    if cp.returncode != 0 or not lines:
        tail = " | ".join((cp.stderr or "").strip().splitlines()[-3:])
        raise QuantizeError(f"gguf-py under {python} could not read {path.name} "
                            f"(exit {cp.returncode}): {tail[:300]}")
    try:
        meta = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise QuantizeError(f"gguf-py metadata of {path.name} is not JSON ({exc.msg})") from exc
    tpl = meta.pop("chat_template", None)
    meta["chat_template_sha256"] = sha256_text(tpl) if isinstance(tpl, str) else None
    return meta


def parse_ppl(text: str) -> float | None:
    """The last ``Final estimate: PPL = x`` line (llama-perplexity, llama-imatrix)."""
    found = _PPL.findall(text)
    return float(found[-1]) if found else None


def metadata_checks(meta: dict[str, Any], *, qtype: str, arch: str | None,
                    template_sha: str | None, serve_ctx: int = SERVE_CTX) -> dict[str, bool]:
    checks = {
        "file_type": meta.get("file_type") == FILE_TYPES.get(qtype),
        "no_nextn_layers": not meta.get("nextn_predict_layers"),
        "context_holds_serve_line": isinstance(meta.get("context_length"), int)
        and meta["context_length"] >= serve_ctx,
    }
    if arch is not None:
        checks["architecture"] = meta.get("architecture") == arch
    if template_sha is not None:
        checks["chat_template_is_trained_one"] = meta.get("chat_template_sha256") == template_sha
    return checks


# ---------------------------------------------------------------------------
# Settings and the run
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    out: Path
    data: Path
    evalsets: Path | None
    extra_eval: list[Path]
    injection_manifest: Path | None
    seed: int
    calib_tokens: int
    imatrix_ctx: int
    ppl_ctx: int
    ppl_chunks: int
    execute: bool = False
    # run only
    base: Path | None = None
    adapter: Path | None = None
    template_dir: Path | None = None
    label: str = "model"
    llama: LlamaCpp | None = None
    python: str = sys.executable
    dtype: str = "bf16"
    merge_device: str = "cpu"
    model_class: str = "auto"
    quants: tuple[str, ...] = DEFAULT_QUANTS
    ngl: str = "99"
    max_ppl_ratio: dict[str, float] = field(default_factory=dict)
    max_ppl: float = 100.0
    port: int = 8080
    approved: bool = False  # APPROVAL_FLAG: Max-derived calibration text allowed
    stop_at: float | None = None  # epoch seconds (STOP_MARGIN_S already taken off)
    train_sha256: str | None = None  # the train.jsonl this model goes with

    @property
    def outtype(self) -> str:
        return {"bf16": "bf16", "fp16": "f16"}[self.dtype]

    @property
    def ref_name(self) -> str:
        return {"bf16": "BF16", "fp16": "F16"}[self.dtype]

    def gguf(self, qtype: str) -> Path:
        return self.out / "gguf" / f"{self.label}-{qtype}.gguf"

    @property
    def imatrix(self) -> Path:
        return self.out / "gguf" / f"{self.label}.imatrix.gguf"

    @property
    def imatrix_chunks(self) -> int:
        return max(1, self.calib_tokens // self.imatrix_ctx)

    def fingerprint(self) -> dict[str, Any]:
        """What makes two runs the same run: an --out holding another is refused."""
        assert self.llama is not None and self.base is not None
        adapter = None
        if self.adapter is not None:
            adapter = {n: sha256_file(self.adapter / n)
                       for n in ("adapter_config.json", "adapter_model.safetensors")}
        base_files = {f.name: f.stat().st_size for f in sorted(self.base.iterdir())
                      if f.is_file()}
        return {
            "schema": RUN_SCHEMA, "label": self.label, "llama_cpp_commit": self.llama.commit,
            "base": {"dir": str(self.base), "files": base_files},
            "adapter": adapter, "template_dir": str(self.template_dir),
            "template_sha256": sha256_text(read_template(self.template_dir)),
            "dtype": self.dtype, "model_class": self.model_class,
            "seed": self.seed, "calib_tokens": self.calib_tokens,
            "imatrix_ctx": self.imatrix_ctx, "ppl_ctx": self.ppl_ctx,
            "ppl_chunks": self.ppl_chunks,
        }


def made_outputs(s: Settings) -> list[str]:
    """What an earlier run in ``--out`` made from its inputs. Partials do not
    count (every step redoes them); neither does calib/, which is checked
    against the train split on its own."""
    found = [d for d in ("merged", "base-view") if (s.out / d).exists()]
    g = s.out / "gguf"
    if g.is_dir():
        found += [f"gguf/{f.name}" for f in sorted(g.iterdir()) if ".partial" not in f.name]
    return found


def trainer_train_sha(adapter: Path | None) -> str | None:
    """The train.jsonl sha256 in the trainer's run manifest next to the
    adapter (train_qlora.py writes <run>/adapter and <run>/manifest.json).
    None for a pre-tokenized run (that hash is of the token files) or when
    there is no such manifest."""
    if adapter is None:
        return None
    p = adapter.parent / "manifest.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    data = m.get("data") if isinstance(m, dict) and m.get("schema") == TRAIN_MANIFEST_SCHEMA \
        else None
    if not isinstance(data, dict) or not str(data.get("path", "")).endswith(".jsonl"):
        return None
    sha = data.get("sha256")
    return sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None


def check_train_split(s: Settings, rec: dict[str, Any], *, built: bool) -> dict[str, Any]:
    """The calibration's train split is the one this model goes with. A
    calibration taken from ``<out>/calib`` (``built`` False) must be shown to
    be: its eval check was against the eval sets of the build it came from,
    and a rebuild may have moved its tasks into eval."""
    have = rec["train"]["sha256"]
    want, source = s.train_sha256, "--train-sha256"
    if want is None:
        want, source = trainer_train_sha(s.adapter), "the adapter's trainer manifest"
    if want is not None and want != have:
        raise CalibrationError(
            f"the calibration's train split (sha256 {have[:16]}...) is not the one {source} "
            f"names ({want[:16]}...): calibrate on the split the model was trained with")
    if want is None and not built:
        raise CalibrationError(
            f"{s.data / 'train.jsonl'} not found, and nothing shows that the calibration in "
            f"{s.out / 'calib'} (train.jsonl sha256 {have}) is from the train split this model "
            "goes with: pass --data with the splits, or --train-sha256 with that split's hash")
    return {"train_sha256": have, "matched": source if want is not None else None}


def provenance_gate(s: Settings, rec: dict[str, Any]) -> dict[str, Any]:
    """OPEN-QUESTIONS #6: llama-imatrix develops the model from the calibration
    rows, so Max-derived ones need the operator's approval. A record without
    the counts counts every row."""
    prov = rec.get("provenance")
    prov = prov if isinstance(prov, dict) else {}
    rows = int(rec["calibration"]["rows"])
    n = prov.get("calibration_max_derived")
    n = rows if not isinstance(n, int) else n
    out = {"calibration_rows": rows, "calibration_max_derived": n,
           "ppl_max_derived": prov.get("ppl_max_derived"), "approved": s.approved}
    if n and not s.approved:
        raise Refused(
            f"{n} of {rows} calibration rows are Max-derived (or have no provenance), and "
            "llama-imatrix would develop the model from them. That is on hold until the "
            f"operator answers DISTILL-OPEN-QUESTIONS #6; pass {APPROVAL_FLAG} only with that "
            "approval")
    return out


def provenance_note(info: dict[str, Any], approved: bool) -> str:
    prov = info.get("provenance") or {}
    n, rows = prov.get("calibration_max_derived"), info["calibration"]["rows"]
    if not n:
        return f"provenance: none of {rows} calibration rows is Max-derived"
    held = ("approved (" + APPROVAL_FLAG + ")" if approved else
            f"run --execute REFUSED without {APPROVAL_FLAG} (DISTILL-OPEN-QUESTIONS #6)")
    return f"provenance: {n}/{rows} calibration rows are Max-derived or unknown: {held}"


def default_label(adapter: Path | None, base: Path) -> str:
    if adapter is None:
        return _label("base-" + base.name)
    name = adapter.name
    if name in ("adapter", "final", "lora", "checkpoint-final") and adapter.parent.name:
        name = adapter.parent.name
    return _label(name)


def _check_output(cmd: list[str]) -> tuple[int, str]:
    cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def _run_logged(cmd: list[str], log: Path) -> tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        fh.write("$ " + shlex.join(cmd) + "\n")
        fh.flush()
        cp = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
    return cp.returncode, log.read_text(encoding="utf-8", errors="replace")


def _step(cmd: list[str], log: Path, what: str) -> str:
    print(f"  $ {shlex.join(cmd)}")
    rc, text = _run_logged(cmd, log)
    if rc != 0:
        tail = "\n    ".join(text.strip().splitlines()[-8:])
        raise QuantizeError(f"{what} failed (exit {rc}); log {log}\n    {tail}")
    return text


def arch_note(arch: str | None, commit: str) -> str:
    if arch is None:
        return "architecture unknown here (config.json not readable); preflight checks it"
    if arch in VERIFIED_SUPPORT.get(commit, ()):
        return (f"{arch}: converter support at {commit} verified (gguf arch "
                f"{GGUF_ARCH.get(arch)}); preflight re-checks it")
    return (f"{arch}: not verified for {commit}; preflight asks that converter "
            "(--print-supported-models)")


def preflight(s: Settings, arch: str | None, *, need_bytes: int) -> dict[str, Any]:
    llama = s.llama
    assert llama is not None
    if not llama.converter.is_file():
        raise QuantizeError(f"{llama.converter} not found (--llama-cpp-dir)")
    rc, out = _check_output(git_head_cmd(llama))
    head = out.strip().splitlines()[0] if rc == 0 and out.strip() else ""
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise QuantizeError(f"{llama.dir} is not a git checkout (rev-parse: {out.strip()[:200]})")
    if not head.startswith(llama.commit.lower()):
        raise QuantizeError(f"{llama.dir} is at {head[:12]}, not the pinned {llama.commit}: "
                            "check out and build the pin, or pass --llama-cpp-commit")
    rc, out = _check_output(git_dirty_cmd(llama))
    if rc != 0 or out.strip():
        raise QuantizeError(f"the converter or gguf-py in {llama.dir} has local changes:\n"
                            f"{out.strip()[:500]}")
    builds = {}
    for tool in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-server"):
        p = llama.tool(tool)
        if not (p.is_file() and os.access(p, os.X_OK)):
            raise QuantizeError(f"{p} not found or not executable (--llama-bin-dir)")
    for tool in ("llama-imatrix", "llama-perplexity", "llama-server"):
        rc, out = _check_output(version_cmd(llama, tool))
        m = _BUILD.search(out)
        if rc != 0 or m is None or not head.startswith(m.group(1)):
            raise QuantizeError(f"{tool} is not built from {head[:12]} "
                                f"(--version: {out.strip()[:200]!r}); rebuild at the pin")
        builds[tool] = m.group(1)
    if arch is None:
        raise QuantizeError(f"no architectures in {s.base}/config.json")
    rc, out = _check_output(supported_models_cmd(s.python, llama))
    if rc != 0 or not re.search(rf"\b{re.escape(arch)}\b", out):
        raise QuantizeError(
            f"convert_hf_to_gguf.py at {head[:12]} does not register {arch}. Use a newer "
            "llama.cpp whose conversion/ registers it (9d57ce456 registers the Qwen3.5 "
            "family), rebuild, and pass --llama-cpp-commit")
    free = shutil.disk_usage(s.out).free
    if free < need_bytes:
        raise QuantizeError(f"{free / 1e9:.1f} GB free under {s.out}; this run needs about "
                            f"{need_bytes / 1e9:.1f} GB")
    return {"head": head, "builds": builds, "converter_registers": arch,
            "disk_free_bytes": free, "disk_need_bytes": need_bytes}


def need_bytes(s: Settings) -> int:
    assert s.base is not None
    base = sum(f.stat().st_size for f in s.base.glob("*.safetensors"))
    need = DISK_MARGIN
    if s.adapter is not None and not (s.out / "merged").exists():
        need += base
    if not s.gguf(s.ref_name).exists():
        need += base
    for q in s.quants:
        if not s.gguf(q).exists():
            need += int(base * SIZE_RATIO.get(q, 0.6))
    return need


def _file_entry(path: Path, kind: str, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "path": str(path), "name": path.name, "bytes": path.stat().st_size,
            "sha256": sha256_file(path), **extra}


def print_plan(s: Settings, arch: str | None, cal_note: str) -> None:
    llama = s.llama
    assert llama is not None and s.base is not None

    def show(cmd: list[str], expect: str = "") -> None:
        print(f"  $ {shlex.join(cmd)}" + (f"    -> {expect}" if expect else ""))

    ref = s.gguf(s.ref_name)
    model_dir = s.out / ("merged" if s.adapter is not None else "base-view")
    calib, ppl_txt = s.out / "calib" / "calibration.txt", s.out / "calib" / "ppl.txt"
    adapter = s.adapter if s.adapter is not None else "(none: the base through the same recipe)"
    print("DRY RUN: nothing runs and nothing is written. Add --execute to run it.\n")
    print(f"llama.cpp    {llama.dir} @ {llama.commit} (binaries {llama.bin_dir})")
    print(f"base         {s.base}  [{arch_note(arch, llama.commit)}]")
    print(f"adapter      {adapter}")
    print(f"template     {s.template_dir}")
    print(f"out          {s.out}  (label {s.label})")
    stop = (datetime.fromtimestamp(s.stop_at, timezone.utc).isoformat(timespec="seconds")
            if s.stop_at is not None else "no --stop-at")
    print(f"window       stops at {stop}; SIGTERM stops it too (exit {EXIT_PAUSED}, a re-run "
          f"resumes); --detach runs it as {UNIT_PREFIX}<UTC stamp>")
    print(f"calibration  {cal_note}\n")
    print("[1/8] preflight")
    show(git_head_cmd(llama), f"starts with {llama.commit}")
    show(git_dirty_cmd(llama), "empty")
    for tool in ("llama-imatrix", "llama-perplexity", "llama-server"):
        show(version_cmd(llama, tool), f"built from {llama.commit}")
    show(supported_models_cmd(s.python, llama), f"lists {arch or '<arch>'}")
    print("  free disk under out >= the outputs not made yet + 5 GiB")
    print(f"[2/8] calibrate -> {calib}, {ppl_txt}, calibration.json (train split only)")
    if s.adapter is not None:
        print(f"[3/8] merge -> {model_dir}  (in process: transformers + peft, {s.dtype}, "
              f"device {s.merge_device}, model class {s.model_class})")
    else:
        print(f"[3/8] base view -> {model_dir}  (symlinks + the trained template)")
    print("[4/8] convert")
    show(convert_cmd(s.python, llama, model_dir, _partial(ref), outtype=s.outtype,
                     model_name=s.label))
    print("[5/8] imatrix")
    show(imatrix_cmd(llama, ref, calib, _partial(s.imatrix), ctx=s.imatrix_ctx,
                     chunks=s.imatrix_chunks, ngl=s.ngl))
    print("[6/8] quantize")
    for q in s.quants:
        show(quantize_cmd(llama, ref, s.imatrix, _partial(s.gguf(q)), q))
    print(f"[7/8] sanity (max ppl {s.max_ppl}; ratio to {s.ref_name}: "
          + ", ".join(f"{q} <= {s.max_ppl_ratio[q]}" for q in s.quants) + ")")
    for name in (s.ref_name, *s.quants):
        show(perplexity_cmd(llama, s.gguf(name), ppl_txt, ctx=s.ppl_ctx, chunks=s.ppl_chunks,
                            ngl=s.ngl))
    print(f"  metadata: arch {GGUF_ARCH.get(arch or '', '<arch>')}, file type, trained chat "
          f"template, no nextn layers, context >= {SERVE_CTX}")
    print(f"[8/8] manifest -> {s.out / 'quantize-manifest.json'}\n")
    print(f"serve (Phase 6 scoring, {SERVE_CTX // 1024}k):")
    for q in s.quants:
        show(serve_command(s.gguf(q), server=llama.tool("llama-server"), alias=f"{s.label}-{q}",
                           ngl=s.ngl, port=s.port))


def run_pipeline(s: Settings, arch: str | None) -> int:
    llama = s.llama
    assert llama is not None and s.base is not None and s.template_dir is not None
    s.out.mkdir(parents=True, exist_ok=True)
    logs = s.out / "logs"
    fp = s.fingerprint()
    run_json = s.out / "run.json"
    old = json.loads(run_json.read_text(encoding="utf-8")) if run_json.is_file() else None
    if old is not None and old != fp:
        diff = sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))
        made = made_outputs(s)
        if made:
            raise QuantizeError(f"{s.out} holds a run made from other inputs ({', '.join(diff)}): "
                                f"{', '.join(made[:4])}; use a new --out")
        # Nothing was made from the old inputs (a refusal stopped it, perhaps
        # one whose fix is the very input that changed): written anew below.
        print(f"  run.json: earlier inputs ({', '.join(diff)}) made nothing yet; replaced")

    manifest: dict[str, Any] = {"schema": MANIFEST_SCHEMA, "started_at": _now(), "label": s.label,
                                "inputs": {}, "steps": {}, "artifacts": [], "serve": {}}

    def save() -> None:
        manifest["updated_at"] = _now()
        _write_json(s.out / "quantize-manifest.json", manifest)

    try:
        return _run_steps(s, arch, fp, old, manifest, save, logs)
    except Paused:
        manifest["result"] = "PAUSED"
        save()
        raise


def _run_steps(s: Settings, arch: str | None, fp: dict[str, Any], old: dict[str, Any] | None,
               manifest: dict[str, Any], save: Callable[[], None], logs: Path) -> int:
    llama = s.llama
    assert llama is not None and s.base is not None and s.template_dir is not None
    print("[1/8] preflight")
    pre = preflight(s, arch, need_bytes=need_bytes(s))
    manifest["llama_cpp"] = {"dir": str(llama.dir), "commit": llama.commit, **pre}
    manifest["inputs"]["base"] = {
        "dir": str(s.base), "architecture": arch,
        "files": {f.name: {"bytes": f.stat().st_size,
                           **({"sha256": sha256_file(f)} if f.suffix == ".json" else {})}
                  for f in sorted(s.base.iterdir()) if f.is_file()}}
    if s.adapter is not None:
        manifest["inputs"]["adapter"] = {"dir": str(s.adapter), **fp["adapter"]}
    save()

    print("[2/8] calibrate")
    calib_dir = s.out / "calib"
    built = (s.data / "train.jsonl").is_file()
    checked: dict[str, Any] = {}

    def check(rec: dict[str, Any]) -> None:
        checked["train_split"] = check_train_split(s, rec, built=built)
        checked["max_derived"] = provenance_gate(s, rec)

    rec = ensure_calibration(s, calib_dir, write=True, check=check)
    manifest["steps"]["calibrate"] = {k: v for k, v in rec.items()
                                      if k not in ("calibration_ids", "ppl_ids")}
    manifest["steps"]["calibrate"]["train_split"] = checked["train_split"]
    manifest["max_derived"] = checked["max_derived"]
    print(f"  {provenance_note(rec, s.approved)}")
    save()
    # The first step that makes an output from the inputs comes next.
    if old != fp:
        _write_json(s.out / "run.json", fp)

    model_dir = s.out / ("merged" if s.adapter is not None else "base-view")
    if model_dir.exists():
        print(f"[3/8] {model_dir.name}: kept (already made)")
        info_path = model_dir / "merge-info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    elif s.adapter is not None:
        print(f"[3/8] merge -> {model_dir}")
        info = merge_adapter(s.base, s.adapter, model_dir, template_dir=s.template_dir,
                             dtype=s.dtype, device=s.merge_device, model_class=s.model_class)
    else:
        print(f"[3/8] base view -> {model_dir}")
        info = prepare_base_view(s.base, model_dir, s.template_dir)
        _write_json(model_dir / "merge-info.json", info)
    template_sha = (info.get("template") or {}).get("sha256") or sha256_text(
        read_template(s.template_dir))
    manifest["steps"]["merge" if s.adapter is not None else "base_view"] = info
    manifest["inputs"]["template_sha256"] = template_sha
    save()

    ref = s.gguf(s.ref_name)
    ref.parent.mkdir(parents=True, exist_ok=True)
    cmd = convert_cmd(s.python, llama, model_dir, _partial(ref), outtype=s.outtype,
                      model_name=s.label)
    print("[4/8] convert")
    if ref.exists():
        print(f"  kept {ref.name}")
    else:
        _partial(ref).unlink(missing_ok=True)
        _step(cmd, logs / "convert.log", "convert_hf_to_gguf.py")
        _partial(ref).rename(ref)
    manifest["steps"]["convert"] = {"cmd": cmd}

    calib, ppl_txt = calib_dir / "calibration.txt", calib_dir / "ppl.txt"
    cmd = imatrix_cmd(llama, ref, calib, _partial(s.imatrix), ctx=s.imatrix_ctx,
                      chunks=s.imatrix_chunks, ngl=s.ngl)
    print("[5/8] imatrix")
    if s.imatrix.exists():
        print(f"  kept {s.imatrix.name}")
    else:
        _partial(s.imatrix).unlink(missing_ok=True)
        _step(cmd, logs / "imatrix.log", "llama-imatrix")
        _partial(s.imatrix).rename(s.imatrix)
    manifest["steps"]["imatrix"] = {"cmd": cmd, "ctx": s.imatrix_ctx,
                                    "chunks": s.imatrix_chunks, "parse_special": True}

    print("[6/8] quantize")
    manifest["steps"]["quantize"] = {}
    for q in s.quants:
        dst = s.gguf(q)
        cmd = quantize_cmd(llama, ref, s.imatrix, _partial(dst), q)
        if dst.exists():
            print(f"  kept {dst.name}")
        else:
            _partial(dst).unlink(missing_ok=True)
            _step(cmd, logs / f"quantize-{q}.log", f"llama-quantize {q}")
            _partial(dst).rename(dst)
        manifest["steps"]["quantize"][q] = {"cmd": cmd}
    save()

    print("[7/8] sanity")
    sanity = run_sanity(s, arch, template_sha, ppl_txt, logs)
    manifest["sanity"] = sanity

    print("[8/8] manifest")
    manifest["artifacts"] = (
        [_file_entry(ref, "gguf", quant=s.ref_name),
         _file_entry(s.imatrix, "imatrix")]
        + [_file_entry(s.gguf(q), "gguf", quant=q) for q in s.quants]
        + [_file_entry(calib, "calibration"), _file_entry(ppl_txt, "ppl_slice")])
    manifest["serve"] = {q: shlex.join(serve_command(s.gguf(q), server=llama.tool("llama-server"),
                                                     alias=f"{s.label}-{q}", ngl=s.ngl,
                                                     port=s.port))
                         for q in s.quants}
    manifest["finished_at"] = _now()
    manifest["result"] = "PASS" if sanity["pass"] else "FAIL"
    save()
    for a in manifest["artifacts"]:
        print(f"  {a['name']}  {a['bytes']:,} B  sha256 {a['sha256'][:16]}")
    print(f"sanity {manifest['result']}; manifest {s.out / 'quantize-manifest.json'}")
    return 0 if sanity["pass"] else EXIT_SANITY


def run_sanity(s: Settings, arch: str | None, template_sha: str, ppl_txt: Path,
               logs: Path) -> dict[str, Any]:
    llama = s.llama
    assert llama is not None
    want_arch = GGUF_ARCH.get(arch or "")
    results: dict[str, dict[str, Any]] = {}
    for name in (s.ref_name, *s.quants):
        path = s.gguf(name)
        cmd = perplexity_cmd(llama, path, ppl_txt, ctx=s.ppl_ctx, chunks=s.ppl_chunks, ngl=s.ngl)
        print(f"  $ {shlex.join(cmd)}")
        rc, text = _run_logged(cmd, logs / f"ppl-{name}.log")
        ppl = parse_ppl(text) if rc == 0 else None
        try:
            meta: dict[str, Any] = read_gguf_metadata(llama.dir, path, s.python)
        except Exception as exc:  # a file the reader cannot parse fails the check
            meta = {"error": f"{type(exc).__name__}: {exc}"}
        checks = ({"metadata_readable": False} if "error" in meta else
                  metadata_checks(meta, qtype=name, arch=want_arch, template_sha=template_sha))
        checks["loads"] = rc == 0
        checks["ppl_finite"] = ppl is not None and math.isfinite(ppl)
        checks["ppl_under_ceiling"] = checks["ppl_finite"] and ppl is not None and ppl <= s.max_ppl
        results[name] = {"ppl": ppl, "exit": rc, "metadata": meta, "checks": checks}
    ref_ppl = results[s.ref_name]["ppl"]
    for q in s.quants:
        r = results[q]
        ratio = (r["ppl"] / ref_ppl) if r["ppl"] is not None and ref_ppl else None
        r["ratio_to_ref"] = ratio
        r["max_ratio"] = s.max_ppl_ratio[q]
        r["checks"]["ppl_ratio"] = ratio is not None and ratio <= s.max_ppl_ratio[q]
    for name, r in results.items():
        r["pass"] = all(r["checks"].values())
        ratio = r.get("ratio_to_ref")
        print(f"  {name}: ppl {r['ppl']}" + (f" (x{ratio:.4f})" if ratio else "")
              + f" -> {'PASS' if r['pass'] else 'FAIL'}"
              + ("" if r["pass"] else f" {[k for k, v in r['checks'].items() if not v]}"))
    return {"ppl_ctx": s.ppl_ctx, "ppl_chunks": s.ppl_chunks, "max_ppl": s.max_ppl,
            "special_tokens": "tokenized as text (llama-perplexity has no --parse-special)",
            "models": results, "pass": all(r["pass"] for r in results.values())}


def ensure_calibration(s: Settings, calib_dir: Path, *, write: bool,
                       check: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """The calibration this run uses: built from ``--data`` when its train
    split is here (and then equal to any already on disk), else the one
    ``calibrate --execute`` left in ``<out>/calib``. ``check`` sees it before
    anything is written, so a refusal leaves no calibration behind that a
    corrected re-run would then have to match."""
    recorded = recorded_calibration(calib_dir)
    check = check or (lambda rec: None)
    if (s.data / "train.jsonl").is_file():
        cal = calibration_from_settings(s)
        if recorded is not None:
            if (recorded["calibration"]["sha256"] != cal.info["calibration"]["sha256"]
                    or recorded["ppl_slice"]["sha256"] != cal.info["ppl_slice"]["sha256"]):
                raise CalibrationError(f"{calib_dir} holds a different calibration than the "
                                       "train split builds now; use a new --out")
            check(recorded)
            print(f"  kept {calib_dir} (matches the train split)")
            return recorded
        check(cal.info)
        if not write:
            return cal.info
        rec = write_calibration(cal, calib_dir)
        c = rec["calibration"]
        print(f"  {c['rows']} rows / ~{c['tokens_est']:,} tokens from {c['tasks']} tasks; "
              f"ppl slice {rec['ppl_slice']['rows']} rows")
        return rec
    if recorded is not None:
        # Made elsewhere: it must be made for this run's settings, or the
        # imatrix chunks and the perplexity slice do not fit them.
        want, have = calib_settings(s), recorded.get("settings")
        have = have if isinstance(have, dict) else {}
        diff = [f"{k} {have.get(k)} (this run: {v})" for k, v in want.items() if have.get(k) != v]
        if diff:
            raise CalibrationError(
                f"{calib_dir} was calibrated with other settings ({'; '.join(diff)}): run "
                "calibrate --execute again with this run's --seed, --calib-tokens, "
                "--imatrix-ctx, --ppl-ctx and --ppl-chunks, or give the run the calibration's")
        check(recorded)
        print(f"  using {calib_dir} (written by calibrate; files and settings match its record)")
        return recorded
    raise CalibrationError(f"{s.data / 'train.jsonl'} not found and no calibration in {calib_dir}")


def _cal_note(s: Settings) -> str:
    """The dry run's calibration line: the real train-only check, in memory."""
    if not (s.data / "train.jsonl").is_file():
        rec = s.out / "calib" / "calibration.json"
        return (f"not checked here: {s.data / 'train.jsonl'} not found"
                + (f" (a calibration exists in {rec.parent})" if rec.is_file() else ""))
    cal = calibration_from_settings(s)
    i = cal.info
    files = len(i["eval_check"]["files"])
    ids = sum(i["eval_check"]["keys"].values())
    return (f"train-only check PASS: {i['train']['rows_read']} train rows, none shares a key "
            f"with {files} eval/val files ({ids} keys); calibration {i['calibration']['rows']} "
            f"rows / ~{i['calibration']['tokens_est']:,} tokens from "
            f"{i['calibration']['tasks']} tasks; ppl slice {i['ppl_slice']['rows']} rows"
            + (f"; {i['eval_check']['dropped_injection_overlap']} rows left out for "
               "eval-injection overlap" if i["eval_check"]["dropped_injection_overlap"] else "")
            + f"\n             {provenance_note(i, s.approved)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def window(stop_at: float | None) -> Iterator[None]:
    """Hold the run to the window: SIGTERM (a ``setups.py`` restore, the weekly
    flywheel, ``systemctl --user stop``) and ``stop_at`` (through SIGALRM)
    raise :class:`Paused` once, wherever the run is. ``subprocess.run`` kills
    the step it was waiting on; a step's ``.partial`` output is redone next
    time. The handlers are restored on the way out."""
    fired = False

    def stop(reason: str) -> None:
        nonlocal fired
        if not fired:
            fired = True
            raise Paused(reason)

    if stop_at is not None and time.time() >= stop_at:
        raise Paused("--stop-at has passed")
    old_term = signal.signal(signal.SIGTERM, lambda *_: stop("SIGTERM"))
    old_alarm = None
    if stop_at is not None:
        old_alarm = signal.signal(signal.SIGALRM, lambda *_: stop("--stop-at"))
        signal.setitimer(signal.ITIMER_REAL, max(0.001, stop_at - time.time()))
    try:
        yield
    finally:
        fired = True  # a signal from here on is too late to stop anything
        if stop_at is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm or signal.SIG_DFL)
        signal.signal(signal.SIGTERM, old_term or signal.SIG_DFL)


def parse_stop_at(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise UsageError(f"--stop-at {value!r}: not an ISO time ({exc})") from exc
    if dt.tzinfo is None:
        raise UsageError("--stop-at needs a UTC offset (e.g. 2026-11-01T06:45-05:00)")
    return dt.timestamp() - STOP_MARGIN_S


def detach_cmd(argv: list[str], out: Path, *, ts: str, cwd: Path,
               python: str = sys.executable) -> list[str]:
    """``run --execute`` as a transient systemd user unit named ``distill-*``,
    so a ``setups.py apply business`` restore stops it (SIGTERM: exit 75, the
    same command resumes) and its GPU processes are never unmanaged ones."""
    unit = f"{UNIT_PREFIX}{ts}"
    log = out / "logs" / f"{unit}.log"
    inner = [python, str(Path(__file__).resolve()), *(x for x in argv if x != "--detach")]
    if "--execute" not in inner:
        inner.append("--execute")
    return ["systemd-run", "--user", f"--unit={unit}", "--collect",
            f"--working-directory={cwd}", "-p", "KillSignal=SIGTERM", "-p", "TimeoutStopSec=60",
            "-p", "Nice=5", "-p", f"StandardOutput=append:{log}",
            "-p", f"StandardError=append:{log}", "--setenv=PYTHONUNBUFFERED=1", *inner]


def _ratio_map(text: str, quants: tuple[str, ...]) -> dict[str, float]:
    defaults = {"Q4_K_S": 1.10, "Q4_K_M": 1.10, "Q5_K_S": 1.05, "Q5_K_M": 1.05,
                "Q6_K": 1.03, "Q8_0": 1.02}
    out = {q: defaults.get(q, 1.10) for q in quants}
    for part in filter(None, (p.strip() for p in text.split(","))):
        q, _, v = part.partition("=")
        q = q.strip().upper()
        if q not in out:
            raise UsageError(f"--max-ppl-ratio names {q}, not one of --quants")
        try:
            out[q] = float(v)
        except ValueError:
            raise UsageError(f"--max-ppl-ratio {part}: not a number") from None
        if not out[q] > 0:
            raise UsageError(f"--max-ppl-ratio {part}: must be positive")
    return out


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="Phase 3 output dir: train.jsonl, val.jsonl, eval-tools.jsonl")
    p.add_argument("--evalsets", default=DEFAULT_EVALSETS,
                   help="build_eval_sets.py output dir ('' to skip)")
    p.add_argument("--eval", action="append", default=[],
                   help="another eval file whose ids must never be calibration text")
    p.add_argument("--injection-manifest", default=None,
                   help=f"default: <evalsets>/{INJECTION_MANIFEST} when present")
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--calib-tokens", type=int, default=524_288,
                   help="calibration size (tokens, the train rows' own counts)")
    p.add_argument("--imatrix-ctx", type=int, default=4096)
    p.add_argument("--ppl-ctx", type=int, default=4096)
    p.add_argument("--ppl-chunks", type=int, default=16)
    p.add_argument("--execute", action="store_true", help="really run (default: print the plan)")


def _settings(a: argparse.Namespace, out: Path) -> Settings:
    for name in ("calib_tokens", "imatrix_ctx", "ppl_ctx", "ppl_chunks"):
        if getattr(a, name) <= 0:
            raise UsageError(f"--{name.replace('_', '-')} must be positive")
    return Settings(
        out=out, data=_path(a.data), evalsets=_path(a.evalsets) if a.evalsets else None,
        extra_eval=[_path(p) for p in a.eval],
        injection_manifest=_path(a.injection_manifest) if a.injection_manifest else None,
        seed=a.seed, calib_tokens=a.calib_tokens, imatrix_ctx=a.imatrix_ctx,
        ppl_ctx=a.ppl_ctx, ppl_chunks=a.ppl_chunks, execute=a.execute)


def _guarded(fn: Callable[[], int]) -> int:
    """Exit codes: 1 comes only from the sanity check (run_pipeline's own
    return), so a caller can tell a failed quant from anything else."""
    try:
        return fn()
    except Paused as exc:
        print(f"PAUSED ({exc}): finished steps are kept; run the same command again to resume",
              file=sys.stderr)
        return EXIT_PAUSED
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except QuantizeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # a merge OOM, a missing interpreter, a bad manifest ...
        traceback.print_exc()
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def cmd_run(a: argparse.Namespace) -> int:
    return _guarded(lambda: _cmd_run(a))


def _cmd_run(a: argparse.Namespace) -> int:
    if (a.adapter is None) == (not a.no_adapter):
        raise UsageError("give --adapter DIR, or --no-adapter for the base model")
    base = _path(a.base)
    adapter = _path(a.adapter) if a.adapter else None
    label = _label(a.label) if a.label else default_label(adapter, base)
    out = _path(a.out) if a.out else _path(DEFAULT_OUT_ROOT) / label
    quants = tuple(q.strip().upper() for q in a.quants.split(",") if q.strip())
    unknown = [q for q in quants if q not in SIZE_RATIO]
    if not quants or unknown:
        raise UsageError(f"--quants: unknown {unknown}; one of {sorted(SIZE_RATIO)}")
    if a.train_sha256 and not re.fullmatch(r"[0-9a-f]{64}", a.train_sha256.lower()):
        raise UsageError("--train-sha256 takes a sha256 (64 hex digits)")
    llama_dir = _path(a.llama_cpp_dir)
    s = _settings(a, out)
    s.base, s.adapter, s.label = base, adapter, label
    s.template_dir = _path(a.template_dir)
    s.llama = LlamaCpp(llama_dir, a.llama_cpp_commit.lower(),
                       _path(a.llama_bin_dir) if a.llama_bin_dir else llama_dir / "build" / "bin")
    s.python, s.dtype, s.merge_device = a.convert_python, a.dtype, a.merge_device
    s.model_class, s.quants, s.ngl, s.port = a.model_class, quants, a.ngl, a.port
    s.max_ppl_ratio = _ratio_map(a.max_ppl_ratio, quants)
    s.max_ppl = a.max_ppl
    s.approved, s.stop_at = a.approved, parse_stop_at(a.stop_at)
    s.train_sha256 = a.train_sha256.lower() if a.train_sha256 else None
    cfg = base / "config.json"
    arch = None
    if cfg.is_file():
        arch = (json.loads(cfg.read_text(encoding="utf-8")).get("architectures") or [None])[0]
    detached = None
    if a.detach:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        detached = detach_cmd(list(a.argv), out, ts=ts, cwd=Path.cwd())
    if not s.execute:
        print_plan(s, arch, _cal_note(s))
        if detached is not None:
            print(f"\ndetached (with --execute): {shlex.join(detached)}")
        return 0
    if not base.is_dir():
        raise QuantizeError(f"--base {base} not found")
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise QuantizeError(f"--adapter {adapter} has no adapter_config.json")
    if detached is not None:
        if shutil.which("systemd-run") is None:
            raise QuantizeError("systemd-run is not available on this host (--detach)")
        (out / "logs").mkdir(parents=True, exist_ok=True)
        print(f"  $ {shlex.join(detached)}")
        rc = subprocess.run(detached, check=False).returncode
        if rc != 0:
            raise QuantizeError(f"systemd-run exited {rc}: nothing was started")
        unit = next(x for x in detached if x.startswith("--unit=")).split("=", 1)[1]
        print(f"started {unit}; log {out / 'logs' / (unit + '.log')}; stop it (finished steps "
              f"are kept) with: systemctl --user stop {unit}")
        return 0
    with window(s.stop_at):
        return run_pipeline(s, arch)


def cmd_calibrate(a: argparse.Namespace) -> int:
    return _guarded(lambda: _cmd_calibrate(a))


def _cmd_calibrate(a: argparse.Namespace) -> int:
    s = _settings(a, _path(a.out))
    if not s.execute:
        print("DRY RUN: nothing is written. Add --execute to write the calibration.\n")
        print(f"calibration  {_cal_note(s)}")
        print(f"would write  {s.out / 'calib'}/calibration.txt, ppl.txt, calibration.json")
        return 0
    rec = ensure_calibration(s, s.out / "calib", write=True)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("calibration_ids", "ppl_ids")},
                     indent=2))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    gguf = _path(a.gguf)
    llama_dir = _path(a.llama_cpp_dir)
    server = (_path(a.llama_bin_dir) if a.llama_bin_dir else llama_dir / "build" / "bin") \
        / "llama-server"
    alias = a.alias or gguf.stem
    print(shlex.join(serve_command(gguf, server=server, alias=alias, host=a.host, port=a.port,
                                   ngl=a.ngl)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantize", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def llama_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--llama-cpp-dir", default=DEFAULT_LLAMA_CPP,
                       help="llama.cpp checkout (converter, gguf-py)")
        p.add_argument("--llama-bin-dir", default=None, help="default: <llama-cpp-dir>/build/bin")
        p.add_argument("--ngl", default="99", help="layers on the GPU")

    r = sub.add_parser("run", help="merge, convert, imatrix, quantize, check (plan unless --execute)")
    r.add_argument("--adapter", default=None, help="PEFT LoRA dir (adapter_config.json + weights)")
    r.add_argument("--no-adapter", action="store_true", help="quantize the base itself")
    r.add_argument("--base", default=DEFAULT_BASE, help="base model dir (HF safetensors)")
    r.add_argument("--template-dir", default=DEFAULT_TEMPLATE_DIR,
                   help="the tokenizer dir Phase 3 rendered the training text with")
    r.add_argument("--out", default=None, help=f"default: {DEFAULT_OUT_ROOT}/<label>")
    r.add_argument("--label", default=None, help="name stem of the GGUFs")
    r.add_argument("--llama-cpp-commit", default=DEFAULT_COMMIT,
                   help="the pinned llama.cpp commit (default: the hosts' build)")
    r.add_argument("--convert-python", default=sys.executable,
                   help="python with the converter's requirements")
    r.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    r.add_argument("--merge-device", choices=("cpu", "auto"), default="cpu",
                   help="cpu: all in system RAM; auto: GPU + CPU + disk offload")
    r.add_argument("--model-class", choices=("auto", "causal-lm", "image-text-to-text"),
                   default="auto", help="transformers class to load the base with")
    r.add_argument("--quants", default=",".join(DEFAULT_QUANTS))
    r.add_argument("--max-ppl-ratio", default="",
                   help="per quant, e.g. Q4_K_M=1.10,Q5_K_M=1.05 (the defaults)")
    r.add_argument("--max-ppl", type=float, default=100.0,
                   help="perplexity ceiling that marks a broken model")
    r.add_argument("--port", type=int, default=8080, help="port in the printed serve line")
    r.add_argument(APPROVAL_FLAG, dest="approved", action="store_true",
                   help="the operator approved Max-derived calibration text (OPEN-QUESTIONS #6)")
    r.add_argument("--stop-at", default=None,
                   help="ISO time with a UTC offset (the window's end): stop before it, exit 75")
    r.add_argument("--detach", action="store_true",
                   help=f"with --execute: run as systemd user unit {UNIT_PREFIX}<UTC stamp>")
    r.add_argument("--train-sha256", default=None,
                   help="sha256 of the train.jsonl this model goes with (default: from the "
                        "adapter's trainer manifest)")
    llama_args(r)
    _common(r)

    c = sub.add_parser("calibrate", help="the train-only calibration text alone (no GPU)")
    c.add_argument("--out", required=True, help="run dir; writes <out>/calib/")
    _common(c)

    v = sub.add_parser("serve", help="print the llama-server line for Phase 6 scoring (256k)")
    v.add_argument("--gguf", required=True)
    v.add_argument("--alias", default=None, help="served model id (default: the file stem)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8080)
    llama_args(v)

    args = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(args)
    a.argv = args  # --detach re-runs this command line inside the unit
    return {"run": cmd_run, "calibrate": cmd_calibrate, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
