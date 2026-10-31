#!/usr/bin/env python3
"""quantize.py for real, on a tiny model of the student's architecture, CPU only.

The unit tests fake every tool quantize.py drives. This runs the real chain on
a laptop: merge -> GGUF bf16 -> imatrix (train split only) -> Q4_K_M + Q5_K_M
-> load/perplexity sanity -> manifest, with the pinned llama.cpp built for the
CPU alone (no Metal, CUDA or BLAS) and the merge on the CPU.

    # 1. Everything the run needs, into one scratch dir. Idempotent (a step
    #    that is done is skipped). Network: GitHub, PyPI, Hugging Face; ~4 GB.
    python3 scripts/distill/quantize_smoke.py prepare --dir /scratch/qsmoke
    # 2. The eval-leak refusal, then quantize.py run --execute end to end for
    #    each variant (below); ~11 GB of outputs in <dir>/phase6, ~13 minutes
    #    on an M2 Pro. Writes <dir>/smoke-report.json (timings, sizes,
    #    perplexities, the manifests' key facts).
    python3 scripts/distill/quantize_smoke.py run --dir /scratch/qsmoke
    # 3. The same as tests; skipped unless the variable names a prepared dir.
    DISTILL_QUANTIZE_SMOKE_DIR=/scratch/qsmoke uv run --frozen pytest \\
        tests/test_distill_quantize_smoke.py -q -p no:cacheprovider

What ``prepare`` makes in ``--dir``
-----------------------------------
``.quantize-smoke``  the mark that the smoke owns the dir. Its subdirectories
    have the names the production scripts use under a distill root (phase3/,
    models/, phase6/), and the smoke deletes and remakes its own, so ``prepare``
    takes only a new or empty ``--dir`` and every subcommand refuses a dir
    without the mark; ``make-data`` also replaces only a ``phase3*/`` whose
    ``report.json`` it wrote (``"schema": "smoke"``).
``llama.cpp-<commit>/``  the pin, fetched at its full sha and built CPU-only
    (llama-imatrix, llama-quantize, llama-perplexity, llama-server, llama-cli),
    with a native cmake + ninja from ``buildtools/`` (a Homebrew cmake may be
    an x86_64 binary under Rosetta on Apple silicon, which builds for x86_64).
``qvenv/``  the converter's interpreter: llama.cpp's own
    ``requirements/requirements-convert_hf_to_gguf.txt`` (CPU torch).
``mvenv/``  the interpreter quantize.py runs under (the merge happens in
    process): CPU torch, a transformers that knows Qwen3.5, peft.
``models/Qwen3.5-0.8B/``  the smallest Qwen3.5, the 9B student's architecture
    (``Qwen3_5ForConditionalGeneration``: hybrid linear + full attention, MTP
    head, vision tower), at a pinned revision; the weights are checked against
    the hub's sha256 and only data files are fetched (no code).
``models/qwen3.5-0.8b-tokenizer/``  the template dir Phase 3 renders with: the
    base's tokenizer and chat template, the template marked with a Jinja comment
    (same renders, other bytes) so the GGUFs must carry this one.
``phase3/``  a synthetic Phase 3 build in build.py's row format, rendered by
    render.py with the real template and tokenizer: train / val / val-cases /
    eval-tools, ``report.json`` naming a non-Max teacher, and ``evalsets/``
    from the repo's eval-set fixtures. One train row (a task of its own) holds
    an eval-injection instruction: the calibration must leave it out.
``phase3-leak/``  the same, with one train row moved into an eval-tools task:
    quantize.py must refuse it before any merge.
``runs/smoke-lora/``, ``runs/smoke-lora-text/``  a random LoRA (r=4 on q_proj
    and v_proj, B non-zero) saved the way train_qlora.py saves one:
    ``adapter/`` plus the run's ``manifest.json`` naming the train split, so the
    calibration is tied to it. The first is made on the multimodal wrapper
    (module names under ``model.language_model``), the second on the text-only
    causal LM (``model.layers``): which one a trained adapter is depends on the
    trainer's loader, and quantize.py must merge either.

What ``run`` runs
-----------------
The leak refusal (``phase3-leak/``: exit 2, nothing merged), then
``quantize.py run --execute`` for each of ``--variants``: ``wrapper`` (the
proof run, ``--dtype`` bf16 by default), ``text`` and ``base`` (``--no-adapter``,
the Phase 4 base recipe), the last two in fp16 (``--other-dtype``). All on the
CPU (``--ngl 0``, a build without GPU backends), with the converter under
``qvenv`` and quantize.py under ``mvenv``.

The downloads are data: Python that reads them runs with ``-I``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "distill_evalsets"

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
# quantize.py's DEFAULT_COMMIT, in full (a fetch by sha needs all 40 digits).
LLAMA_CPP_SHA = "9d57ce456c94d241dde672b2db9cf18879766568"
LLAMA_TARGETS = ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-server",
                 "llama-cli")
CMAKE_FLAGS = ("-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF",
               "-DGGML_BLAS=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_BUILD_UI=OFF",
               "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF")

HF = "https://huggingface.co"
MODEL_REPO = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_DIR = "Qwen3.5-0.8B"
TOKENIZER_DIR = "qwen3.5-0.8b-tokenizer"
# Everything but the weights (which the hub's file list names), all data.
MODEL_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json",
               "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja")
# A Jinja comment whose "-" also eats the newline before it: renders nothing.
TEMPLATE_MARK = "\n{#- quantize smoke: the template Phase 3 rendered with #}\n"

PYTHON = "3.12"
BUILD_TOOLS = ("cmake==3.31.6", "ninja==1.11.1.4")
TORCH = "torch==2.11.0"  # the converter's pin, in both venvs
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
MERGE_PACKAGES = ("transformers==5.20.2", "peft==0.21.2", "accelerate==1.15.0",
                  "safetensors==0.8.0", "jinja2==3.1.6")

LORA = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "b_std": 0.01,
        "seed": 20261030}
RUN_LABEL = "smoke-lora"
# Which transformers class the LoRA was made on decides its module names: the
# multimodal wrapper (model.language_model.layers.N...: the HF architecture)
# or the text-only causal LM (model.layers.N...). The trainer's loader decides
# which one a real adapter is, so the smoke merges both; "base" is the base
# model through the same recipe (--no-adapter, the Phase 4 base GGUF).
VARIANTS = {"wrapper": ("AutoModelForImageTextToText", RUN_LABEL),
            "text": ("AutoModelForCausalLM", RUN_LABEL + "-text"),
            "base": (None, "smoke-base")}
# CPU-sized: 8 imatrix chunks of 512 tokens, 4 perplexity chunks of 512. At the
# pin a BF16 matmul on an ARM CPU is a scalar loop (ggml_vec_dot_bf16 has x86
# SIMD only): ~37 s per 512-token pass of this model on an M2 Pro, against ~2 s
# for F16. bf16 (the production recipe) is the default; --dtype fp16 is the
# fast CPU loop.
RUN_SETTINGS = {"calib_tokens": 4096, "imatrix_ctx": 512, "ppl_ctx": 512, "ppl_chunks": 4}
DTYPES = ("bf16", "fp16")
# quantize.py's bars (1.10 / 1.05) are for the 9B over 16 x 4096 tokens; over
# 4 x 512 tokens of a 0.8B the ratio moves by ~0.1 from noise alone (seen:
# 0.88-1.04). These still fail a broken quant, which lands at x2 and up.
PPL_RATIO = "Q4_K_M=1.25,Q5_K_M=1.15"
SEED = 20261030  # quantize.py's default --seed, passed explicitly
TEACHER = "smoke-teacher"  # report.json's teacher: not a Max model
N_TASKS, ROWS_PER_TASK, N_VAL_TASKS, N_EVAL_TASKS = 60, 5, 6, 8
LEAK_ROW = "smoke-train-0007"  # moved into eval task smoke-etask-000 in phase3-leak/
MARKER = ".quantize-smoke"  # in --dir: the smoke made (and owns) it


class SmokeError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
       log_to: Path | None = None) -> float:
    """Run a command (output to *log_to* when given); the seconds it took."""
    log("$ " + shlex.join(cmd))
    t = time.monotonic()
    if log_to is None:
        rc = subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode
    else:
        log_to.parent.mkdir(parents=True, exist_ok=True)
        with log_to.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                check=False).returncode
    if rc != 0:
        tail = log_to.read_text(errors="replace").splitlines()[-15:] if log_to else []
        raise SmokeError(f"exit {rc}: {shlex.join(cmd)}" + "".join("\n  " + x for x in tail))
    return time.monotonic() - t


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def sibling(name: str, file: str) -> ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, HERE / file)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


class Layout:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.llama = self.root / f"llama.cpp-{LLAMA_CPP_SHA[:9]}"
        self.bin = self.llama / "build" / "bin"
        self.buildtools = self.root / "buildtools"
        self.qvenv = self.root / "qvenv"
        self.mvenv = self.root / "mvenv"
        self.model = self.root / "models" / MODEL_DIR
        self.tokenizer = self.root / "models" / TOKENIZER_DIR
        self.data = self.root / "phase3"
        self.leak = self.root / "phase3-leak"
        self.run = self.run_dir("wrapper")
        self.adapter = self.run / "adapter"
        self.logs = self.root / "logs"

    def run_dir(self, variant: str) -> Path:
        return self.root / "runs" / VARIANTS[variant][1]

    def adapter_for(self, variant: str) -> Path | None:
        return None if VARIANTS[variant][0] is None else self.run_dir(variant) / "adapter"

    @staticmethod
    def py(venv: Path) -> Path:
        return venv / "bin" / "python"

    def missing(self) -> list[str]:
        """What ``run`` needs that ``prepare`` has not made."""
        need = [self.root / MARKER,
                self.llama / "convert_hf_to_gguf.py", self.py(self.qvenv), self.py(self.mvenv),
                self.model / "config.json", self.tokenizer / "tokenizer.json",
                self.data / "train.jsonl", self.leak / "train.jsonl"]
        for v in VARIANTS:
            if self.adapter_for(v) is not None:
                need += [self.run_dir(v) / "adapter" / "adapter_model.safetensors",
                         self.run_dir(v) / "manifest.json"]
        need += [self.bin / t for t in LLAMA_TARGETS]
        return [str(p) for p in need if not p.exists()]


def claim(root: Path, cmd: str) -> None:
    """Go on only in a dir the smoke owns: one with its mark, or (``prepare``) a
    new or empty one, which it marks. ``run`` may also start on an empty dir, to
    list what ``prepare`` has not made."""
    root = root.expanduser().resolve()
    if (root / MARKER).is_file():
        return
    empty = not root.exists() or (root.is_dir() and not any(root.iterdir()))
    if empty and cmd == "prepare":
        root.mkdir(parents=True, exist_ok=True)
        (root / MARKER).write_text(
            "quantize_smoke.py owns this dir: it deletes and remakes phase3/, phase3-leak/, "
            "runs/, phase6/ and pytest-run/ here.\n", encoding="utf-8")
        return
    if empty and cmd == "run":
        return
    raise SmokeError(f"{root} is not a quantize smoke dir (no {MARKER}) and is not empty: the "
                     "smoke deletes and remakes phase3/, runs/ and phase6/ in its --dir. "
                     "prepare a new or empty dir")


def smoke_build(d: Path) -> bool:
    """*d* holds a Phase 3 build that make-data wrote (its report.json says so)."""
    try:
        report = json.loads((d / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(report, dict) and report.get("schema") == "smoke"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def uv() -> str:
    exe = shutil.which("uv")
    if exe is None:
        raise SmokeError("uv is not on PATH (https://docs.astral.sh/uv/)")
    return exe


def make_venv(venv: Path, installs: list[list[str]]) -> None:
    if Layout.py(venv).exists() and (venv / ".smoke-done").exists():
        log(f"kept {venv.name}")
        return
    sh([uv(), "venv", "-q", "--clear", "--python", PYTHON, str(venv)])
    for args in installs:
        sh([uv(), "pip", "install", "-q", "--python", str(Layout.py(venv)), *args])
    (venv / ".smoke-done").write_text("ok\n")


def prepare_tools(L: Layout) -> None:
    make_venv(L.buildtools, [list(BUILD_TOOLS)])
    make_venv(L.qvenv, [["-r", str(L.llama / "requirements" /
                                  "requirements-convert_hf_to_gguf.txt")]])
    # CPU torch from its own index first, then the rest from PyPI.
    make_venv(L.mvenv, [["--index-url", TORCH_CPU_INDEX, TORCH], [TORCH, *MERGE_PACKAGES]])


def prepare_llama_cpp(L: Layout) -> None:
    if not (L.llama / ".git").is_dir():
        L.llama.mkdir(parents=True, exist_ok=True)
        sh(["git", "-C", str(L.llama), "init", "-q"])
        sh(["git", "-C", str(L.llama), "remote", "add", "origin", LLAMA_CPP_REPO])
    head = subprocess.run(["git", "-C", str(L.llama), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    if head != LLAMA_CPP_SHA:
        sh(["git", "-C", str(L.llama), "fetch", "-q", "--depth", "1", "origin", LLAMA_CPP_SHA])
        sh(["git", "-C", str(L.llama), "-c", "advice.detachedHead=false", "checkout", "-q",
            "FETCH_HEAD"])


def build_llama_cpp(L: Layout, jobs: int) -> None:
    if all((L.bin / t).exists() for t in LLAMA_TARGETS):
        log(f"kept the llama.cpp build in {L.bin}")
        return
    tools = L.buildtools / "bin"
    flags = list(CMAKE_FLAGS) + [f"-DCMAKE_MAKE_PROGRAM={tools / 'ninja'}"]
    if platform.system() == "Darwin":
        # The interpreter's own arch: an x86_64 cmake under Rosetta would
        # otherwise build x86_64 binaries on Apple silicon.
        flags.append(f"-DCMAKE_OSX_ARCHITECTURES={platform.machine()}")
    env = {**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ.get('PATH', '')}"}
    sh([str(tools / "cmake"), "-S", str(L.llama), "-B", str(L.llama / "build"), "-G", "Ninja",
        *flags], env=env, log_to=L.logs / "cmake-configure.log")
    sh([str(tools / "cmake"), "--build", str(L.llama / "build"), "-j", str(jobs), "--target",
        *LLAMA_TARGETS], env=env, log_to=L.logs / "cmake-build.log")


def _download(url: str, dst: Path, sha256: str | None = None) -> None:
    tmp = dst.with_name(dst.name + ".partial")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(r, fh, 8 * 1024 * 1024)
    if sha256 is not None and sha256_file(tmp) != sha256:
        tmp.unlink()
        raise SmokeError(f"{dst.name}: sha256 does not match the hub's ({sha256[:16]}...)")
    tmp.replace(dst)


def fetch_model(L: Layout) -> None:
    """The pinned revision's data files; weights checked against the hub's sha256."""
    api = f"{HF}/api/models/{MODEL_REPO}/tree/{MODEL_REVISION}"
    with urllib.request.urlopen(api, timeout=60) as r:  # noqa: S310
        tree = json.loads(r.read())
    weights = {e["path"]: (e.get("lfs") or {}).get("oid") for e in tree
               if e.get("type") == "file" and e["path"].endswith(".safetensors")}
    if not weights:
        raise SmokeError(f"{MODEL_REPO}@{MODEL_REVISION} lists no .safetensors")
    for name in (*MODEL_FILES, *sorted(weights)):
        dst = L.model / name
        if dst.exists():
            continue
        log(f"download {MODEL_REPO}@{MODEL_REVISION[:12]} {name}")
        _download(f"{HF}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}", dst, weights.get(name))


def prepare_model(L: Layout) -> None:
    L.model.mkdir(parents=True, exist_ok=True)
    stamp = L.model / "REVISION"
    pinned = f"{MODEL_REPO}@{MODEL_REVISION}"
    if stamp.is_file() and stamp.read_text().strip() == pinned:
        log(f"kept {L.model}")
    else:
        fetch_model(L)
        stamp.write_text(pinned + "\n")
    L.tokenizer.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        shutil.copy2(L.model / name, L.tokenizer / name)
    # The template Phase 3 rendered with, marked so it differs from the base's
    # by its bytes and not by what it renders: the GGUF metadata check then
    # shows the converter embedded the installed template, not the base's.
    tpl = (L.model / "chat_template.jinja").read_text(encoding="utf-8")
    (L.tokenizer / "chat_template.jinja").write_text(tpl.rstrip("\n") + TEMPLATE_MARK,
                                                     encoding="utf-8")


def prepare(a: argparse.Namespace) -> int:
    L = Layout(Path(a.dir))
    L.root.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    prepare_llama_cpp(L)
    prepare_tools(L)
    build_llama_cpp(L, a.jobs)
    prepare_model(L)
    me = str(Path(__file__).resolve())
    mpy = str(Layout.py(L.mvenv))
    # Under the merge interpreter (tokenizers, jinja2, torch, peft), isolated.
    if not (L.data / "train.jsonl").exists() or not (L.leak / "train.jsonl").exists():
        sh([mpy, "-I", me, "make-data", "--dir", str(L.root)])
    for v in VARIANTS:
        adapter = L.adapter_for(v)
        if adapter is not None and not (adapter / "adapter_model.safetensors").exists():
            sh([mpy, "-I", me, "make-adapter", "--dir", str(L.root), "--variant", v])
    missing = L.missing()
    if missing:
        raise SmokeError("prepare left these missing: " + ", ".join(missing))
    log(f"prepared {L.root} in {time.monotonic() - t0:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# make-data: a synthetic Phase 3 build (runs under the merge venv)
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_record", "description": "Look up stored records about a topic.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "what to look for"},
            "limit": {"type": "integer", "description": "most records to return"}},
            "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "write_note", "description": "Save a short note under a title.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "body": {"type": "string"}},
            "required": ["title", "body"]}}},
]
SYSTEM = ("You are a careful assistant for a small community workshop. Use the tools when a "
          "request needs stored records, and answer plainly otherwise.")
ADJ = ("blue", "quiet", "early", "spare", "round", "narrow", "late", "warm", "plain", "tall")
NOUN = ("bench", "lantern", "kettle", "ladder", "basket", "window", "garden", "bicycle",
        "teapot", "workshop", "fence", "drawer")
MONTH = ("March", "April", "May", "June", "July", "August", "September", "October")
VERB = ("Find", "Look up", "Pull up", "Show me", "Fetch")


def _conversation(rng: random.Random, task: int, n: int) -> list[dict[str, Any]]:
    adj, noun, month = rng.choice(ADJ), rng.choice(NOUN), rng.choice(MONTH)
    topic = f"{adj} {noun} {month}"
    kind = (task + n) % 3
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    if kind == 0:
        limit = rng.randint(2, 6)
        msgs += [{"role": "user", "content": f"{rng.choice(VERB)} the notes on the {adj} "
                  f"{noun} from {month}, at most {limit} of them."},
                 {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                  "function": {"name": "lookup_record",
                               "arguments": json.dumps({"topic": topic, "limit": limit})}}]}]
    elif kind == 1:
        count = rng.randint(1, 4)
        msgs += [{"role": "user", "content": f"How many records mention the {adj} {noun}?"},
                 {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                  "function": {"name": "lookup_record",
                               "arguments": json.dumps({"topic": f"{adj} {noun}"})}}]},
                 {"role": "tool", "content": json.dumps(
                     {"records": [f"{noun} note {i + 1} ({month})" for i in range(count)]})},
                 {"role": "assistant", "content": f"There {'is' if count == 1 else 'are'} "
                  f"{count} record{'' if count == 1 else 's'} about the {adj} {noun}, all "
                  f"from {month}."}]
    else:
        msgs += [{"role": "user", "content": f"Save a note titled '{noun} check' saying the "
                  f"{adj} {noun} was looked at in {month}."},
                 {"role": "assistant", "content": "Saving it now.", "tool_calls": [
                     {"type": "function", "function": {
                         "name": "write_note", "arguments": json.dumps(
                             {"title": f"{noun} check",
                              "body": f"The {adj} {noun} was looked at in {month}."})}}]}]
    return msgs


def first_task(rows: list[dict[str, Any]], seed: int = SEED) -> str:
    """A task id quantize.py's round-robin orders before every task in *rows*."""
    seeded = sibling("distill_quantize", "quantize.py")._seeded

    def key(task: str) -> str:
        return str(seeded(seed, "task:" + task))

    lowest = min(key(r["task_id"]) for r in rows)
    return next(t for t in (f"smoke-task-injection-{n}" for n in range(100_000))
                if key(t) < lowest)


def make_data(a: argparse.Namespace) -> int:
    L = Layout(Path(a.dir))
    for d in (L.data, L.leak):
        if d.exists() and not smoke_build(d):
            raise SmokeError(f"{d} is not a smoke build (its report.json does not say "
                             "\"schema\": \"smoke\"); not replacing it")
    render = sibling("distill_render", "render.py")
    student = render.Student.load(L.tokenizer)
    rng = random.Random(LORA["seed"])

    def rendered(msgs: list[dict[str, Any]]) -> dict[str, Any]:
        r = render.render_example(student, msgs, TOOLS, dict(render.THINKING_OFF))
        return r.as_row()

    def train_row(rid: str, task: str, msgs: list[dict[str, Any]], split: str) -> dict[str, Any]:
        return {"id": rid, "split": split, "task_id": task, "group_id": task.replace("task",
                                                                                     "group"),
                "category": "smoke", "source": "smoke-gym", "label": "pass",
                "model": TEACHER, "max_derived": False, "messages": msgs, "tools": TOOLS,
                "tool_choice": "auto", "chat_template_kwargs": dict(render.THINKING_OFF),
                **rendered(msgs)}

    def case_row(rid: str, set_name: str, task: str, msgs: list[dict[str, Any]],
                 n: int) -> dict[str, Any]:
        return {"id": rid, "set": set_name, "messages": msgs[:-1], "tools": TOOLS,
                "tool_choice": "auto", "expect": {"tool": None},
                "meta": {"source": "smoke-gym", "model": TEACHER, "max_derived": False,
                         "task_id": task, "group_id": task.replace("task", "group"),
                         "trace_id": f"smoke_traces.jsonl:{n}"}}

    train: list[dict[str, Any]] = []
    for t in range(N_TASKS):
        for k in range(ROWS_PER_TASK):
            i = t * ROWS_PER_TASK + k
            train.append(train_row(f"smoke-train-{i:04d}", f"smoke-task-{t:03d}",
                                   _conversation(rng, t, k), "train"))
    # A task of its own that quantize.py's seeded round-robin takes first, so
    # even the smallest budget reaches it: its tool result carries
    # eval-injection's first instruction, which the calibration must drop.
    inj = json.loads((FIXTURES / "eval-injection.sample.jsonl").read_text(
        encoding="utf-8").splitlines()[0])["injection"]["instruction"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What does the stored record about the spare ladder say?"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "lookup_record", "arguments": json.dumps({"topic": "spare ladder"})}}]},
            {"role": "tool", "content": "lookup result: " + inj},
            {"role": "assistant", "content": "The record holds an instruction, not data about "
             "the ladder; I did not act on it."}]
    train.append(train_row(f"smoke-train-{len(train):04d}", first_task(train), msgs, "train"))
    val, val_cases, evals = [], [], []
    for t in range(N_VAL_TASKS):
        for k in range(2):
            msgs = _conversation(rng, t, k)
            rid = f"smoke-val-{t * 2 + k:03d}"
            val.append(train_row(rid, f"smoke-vtask-{t:03d}", msgs, "val"))
            val_cases.append(case_row(rid, "val", f"smoke-vtask-{t:03d}", msgs, 900 + t * 2 + k))
    for t in range(N_EVAL_TASKS):
        for k in range(2):
            msgs = _conversation(rng, t, k)
            evals.append(case_row(f"smoke-eval-{t * 2 + k:03d}", "eval-tools",
                                  f"smoke-etask-{t:03d}", msgs, t * 2 + k))
    report = {"schema": "smoke", "settings": {"teacher_model": TEACHER},
              "note": "synthetic quantize smoke build (quantize_smoke.py); invented text"}

    for d in (L.data, L.leak):
        if d.exists():
            shutil.rmtree(d)  # a smoke build: checked above
        (d / "evalsets" / "ungated").mkdir(parents=True)
        # First, so a build cut short still names itself as the smoke's.
        (d / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        rows = train
        if d == L.leak:
            rows = [dict(r, task_id="smoke-etask-000") if r["id"] == LEAK_ROW else r
                    for r in train]
        for name, part in (("train.jsonl", rows), ("val.jsonl", val),
                           ("val-cases.jsonl", val_cases), ("eval-tools.jsonl", evals)):
            (d / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                          for r in part), encoding="utf-8")
        ev = d / "evalsets"
        for src, dst in (("sanity.sample.jsonl", "sanity.jsonl"),
                         ("eval-longctx.sample.jsonl", "eval-longctx.jsonl"),
                         ("eval-injection.sample.jsonl", "eval-injection.jsonl"),
                         ("eval-injection.manifest.sample.json", "eval-injection.manifest.json"),
                         ("eval-longctx-tools.sample.jsonl", "ungated/eval-longctx-tools.jsonl")):
            shutil.copy2(FIXTURES / src, ev / dst)
    tokens = sum(r["n_tokens"] for r in train)
    log(f"phase3: {len(train)} train rows ({tokens:,} tokens, {N_TASKS + 1} tasks), "
        f"{len(val)} val, {len(evals)} eval-tools; phase3-leak: {LEAK_ROW} in smoke-etask-000")
    return 0


# ---------------------------------------------------------------------------
# make-adapter: a random LoRA saved like train_qlora.py's (merge venv)
# ---------------------------------------------------------------------------


def make_adapter(a: argparse.Namespace) -> int:
    import peft
    import torch
    import transformers

    L = Layout(Path(a.dir))
    cls_name, label = VARIANTS[a.variant]
    if cls_name is None:
        raise SmokeError(f"variant {a.variant} has no adapter")
    run_dir = L.run_dir(a.variant)
    torch.manual_seed(LORA["seed"])
    model = getattr(transformers, cls_name).from_pretrained(
        str(L.model), dtype=torch.bfloat16, device_map="cpu")
    cfg = peft.LoraConfig(r=LORA["r"], lora_alpha=LORA["lora_alpha"], lora_dropout=0.0,
                          target_modules=list(LORA["target_modules"]), bias="none",
                          task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, cfg)
    # PEFT starts every B at zero (a no-op adapter, which quantize.py refuses
    # to merge); a trained adapter has moved them.
    g = torch.Generator().manual_seed(LORA["seed"])
    n = 0
    with torch.no_grad():
        for name, p in pm.named_parameters():
            if ".lora_B." in name:
                p.copy_(torch.randn(p.shape, generator=g) * LORA["b_std"])
                n += 1
    if run_dir.exists():
        shutil.rmtree(run_dir)
    (run_dir / "adapter").mkdir(parents=True)
    pm.save_pretrained(str(run_dir / "adapter"))
    train = L.data / "train.jsonl"
    # The fields of train_qlora.py's manifest that quantize.py reads.
    manifest = {
        "schema": "distill-train-manifest/1", "run_id": label, "mode": "smoke-random-lora",
        "data": {"path": str(train), "sha256": sha256_file(train),
                 "rows": sum(1 for _ in train.open(encoding="utf-8"))},
        "base_model": {"id": MODEL_REPO, "revision_resolved": MODEL_REVISION, "load": "16bit"},
        "config": {k: v for k, v in LORA.items() if k != "b_std"}, "model_class": cls_name,
        "status": "done", "note": "random LoRA B ~ N(0, %s): never trained" % LORA["b_std"],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                           encoding="utf-8")
    log(f"adapter: {n} LoRA modules on {type(model).__name__} -> {run_dir / 'adapter'}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def quantize_argv(L: Layout, out: Path, *, data: Path | None = None, dtype: str = "bf16",
                  variant: str = "wrapper") -> list[str]:
    """The quantize.py command line of the smoke (the test uses it too)."""
    data = data or L.data
    s = RUN_SETTINGS
    adapter = L.adapter_for(variant)
    which = ["--no-adapter"] if adapter is None else ["--adapter", str(adapter)]
    # -I ignores PYTHONUNBUFFERED: -u, so the step lines arrive (and are timed) as they print.
    return [str(Layout.py(L.mvenv)), "-I", "-u", str(HERE / "quantize.py"), "run",
            *which, "--base", str(L.model),
            "--template-dir", str(L.tokenizer), "--data", str(data),
            "--evalsets", str(data / "evalsets"), "--out", str(out),
            "--label", VARIANTS[variant][1],
            "--llama-cpp-dir", str(L.llama), "--convert-python", str(Layout.py(L.qvenv)),
            "--ngl", "0", "--calib-tokens", str(s["calib_tokens"]),
            "--imatrix-ctx", str(s["imatrix_ctx"]), "--ppl-ctx", str(s["ppl_ctx"]),
            "--ppl-chunks", str(s["ppl_chunks"]), "--seed", str(SEED), "--dtype", dtype,
            "--max-ppl-ratio", PPL_RATIO, "--execute"]


def run_timed(cmd: list[str], log_path: Path) -> dict[str, Any]:
    """Run quantize.py, tee its output, time each ``[n/8]`` step."""
    log("$ " + shlex.join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    marks: list[tuple[str, float]] = []
    t0 = time.monotonic()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    with log_path.open("w", encoding="utf-8") as fh, subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as p:
        assert p.stdout is not None
        for line in p.stdout:
            fh.write(line)
            print(line, end="", flush=True)
            if line.startswith("[") and "/8]" in line[:6]:
                marks.append((line.split("]", 1)[1].split()[0], time.monotonic() - t0))
        rc = p.wait()
    total = time.monotonic() - t0
    steps = {name: round((marks[i + 1][1] if i + 1 < len(marks) else total) - at, 1)
             for i, (name, at) in enumerate(marks)}
    return {"cmd": cmd, "exit": rc, "seconds": round(total, 1), "steps": steps,
            "log": str(log_path), "output": log_path.read_text(encoding="utf-8")}


def summarize(out: Path) -> dict[str, Any]:
    m = json.loads((out / "quantize-manifest.json").read_text(encoding="utf-8"))
    if "sanity" not in m:  # stopped or failed before the end
        return {"result": m.get("result"), "steps_done": sorted(m.get("steps", {}))}
    cal = m["steps"]["calibrate"]
    return {
        "result": m.get("result"),
        "llama_cpp": {k: m["llama_cpp"].get(k) for k in ("commit", "head", "builds",
                                                         "converter_registers")},
        "calibration": {"rows": cal["calibration"]["rows"],
                        "tokens_est": cal["calibration"]["tokens_est"],
                        "tasks": cal["calibration"]["tasks"],
                        "ppl_rows": cal["ppl_slice"]["rows"],
                        "dropped_injection_overlap": cal["eval_check"]["dropped_injection_overlap"],
                        "eval_keys": cal["eval_check"]["keys"],
                        "train_split": cal.get("train_split"),
                        "max_derived": m.get("max_derived")},
        "merge": {k: m["steps"].get("merge", m["steps"].get("base_view", {})).get(k)
                  for k in ("model_class", "lora_modules",
                                                          "attached", "b_nonzero",
                                                          "unexpected_base_keys",
                                                          "unexpected_base_key_prefixes")},
        "artifacts": [{k: a[k] for k in ("name", "kind", "bytes", "sha256")}
                      for a in m["artifacts"]],
        "sanity": {name: {"ppl": r["ppl"], "ratio_to_ref": r.get("ratio_to_ref"),
                          "pass": r["pass"], "metadata": r["metadata"]}
                   for name, r in m["sanity"]["models"].items()},
        "serve": m["serve"],
    }


def run(a: argparse.Namespace) -> int:
    L = Layout(Path(a.dir))
    missing = L.missing()
    if missing:
        raise SmokeError("not prepared (run prepare first): " + ", ".join(missing))
    report: dict[str, Any] = {"dir": str(L.root), "llama_cpp": LLAMA_CPP_SHA,
                              "model": f"{MODEL_REPO}@{MODEL_REVISION}",
                              "settings": {**RUN_SETTINGS, "dtype": a.dtype},
                              "lora": LORA, "host": {"system": platform.system(),
                                                     "machine": platform.machine(),
                                                     "cpus": os.cpu_count()}}
    leak_out = L.root / "phase6" / "leak"
    if leak_out.exists():
        shutil.rmtree(leak_out)
    r = run_timed(quantize_argv(L, leak_out, data=L.leak), L.logs / "run-leak.log")
    refused = r["exit"] == 2 and "share a key with the eval / val sets" in r["output"]
    report["leak"] = {"exit": r["exit"], "refused": refused,
                      "merged": (leak_out / "merged").exists(),
                      "message": next((x for x in r["output"].splitlines()
                                       if x.startswith("ERROR:")), None)}
    if not refused or report["leak"]["merged"]:
        raise SmokeError(f"the leaking build was not refused before the merge: {report['leak']}")
    rc = 0
    for variant in a.variants.split(","):
        dtype = a.dtype if variant == "wrapper" else a.other_dtype
        out = L.root / "phase6" / VARIANTS[variant][1]
        if out.exists() and not a.resume:
            shutil.rmtree(out)
        r = run_timed(quantize_argv(L, out, dtype=dtype, variant=variant),
                      L.logs / f"run-{variant}.log")
        res = {"dtype": dtype, **{k: r[k] for k in ("cmd", "exit", "seconds", "steps", "log")}}
        if (out / "quantize-manifest.json").is_file():
            res["summary"] = summarize(out)
        report[variant] = res
        rc = rc or r["exit"]
        (L.root / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
    log(f"report {L.root / 'smoke-report.json'}")
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="quantize_smoke", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build llama.cpp, the venvs, the model, data, adapter")
    p.add_argument("--dir", required=True, help="scratch dir (never inside the repo)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    r = sub.add_parser("run", help="the leak refusal, then quantize.py --execute end to end")
    r.add_argument("--dir", required=True)
    r.add_argument("--resume", action="store_true", help="keep an earlier run's outputs")
    r.add_argument("--dtype", choices=DTYPES, default="bf16",
                   help="the wrapper run's merge and reference GGUF (fp16: ~17x faster on an "
                        "ARM CPU)")
    r.add_argument("--variants", default="wrapper,text,base",
                   help=f"comma list of {', '.join(VARIANTS)} (wrapper is the proof run)")
    r.add_argument("--other-dtype", choices=DTYPES, default="fp16",
                   help="dtype of the text and base runs (they also cover the fp16 path)")
    for name in ("make-data", "make-adapter"):
        x = sub.add_parser(name, help="(prepare runs this under the merge venv)")
        x.add_argument("--dir", required=True)
        if name == "make-adapter":
            x.add_argument("--variant", choices=[v for v, (c, _) in VARIANTS.items() if c],
                           default="wrapper")
    a = ap.parse_args(argv)
    bad = [v for v in getattr(a, "variants", "").split(",") if v and v not in VARIANTS]
    if bad:
        ap.error(f"--variants: unknown {bad}")
    if Path(a.dir).expanduser().resolve().is_relative_to(REPO):
        print(f"--dir {a.dir} is inside the repo; use a scratch dir", file=sys.stderr)
        return 2
    fn = {"prepare": prepare, "run": run, "make-data": make_data, "make-adapter": make_adapter}
    try:
        claim(Path(a.dir), a.cmd)
        return fn[a.cmd](a)
    except SmokeError as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
