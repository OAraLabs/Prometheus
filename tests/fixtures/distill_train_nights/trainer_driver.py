"""The REAL train_qlora.py training loop on a fake backend, as a process: what the 4090 job runner starts
for the train-nights tests (tests/test_distill_train_nights.py).

    trainer_driver.py <path to train_qlora.py> <train_qlora.py arguments...>   # the runner's command line
    trainer_driver.py make-data <dir> <n_train> <n_val>                          # rows in build.py's shape

The job runner runs ``<train python> <code>/scripts/distill/train_qlora.py ...``; the test's train python
is a wrapper that runs this file first. It loads that train_qlora.py, swaps only what needs a GPU or a
tokenizer (the backend, the tokenizer, the template re-fit, the hub revision lookup) and calls its own
``main()``: argument parsing, the Q6b guard, setup/resume, the loop, checkpoints, retention, SIGTERM and
``--stop-at`` are the shipped code. The fake backend's "weights" are one float, its RNG a
``random.Random``; every optimizer step is appended to ``$FAKE_STATE/steps-<out name>.jsonl`` (the row
token ids, lr, weight after), so a test can compare the data order across nights with an uninterrupted
run. ``FAKE_SEC_PER_TOKEN`` makes it take wall time; ``FAKE_STOP_MARGIN_S`` replaces the trainer's two
minutes before ``--stop-at`` (default 0.5 s).
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"<\|[a-z_]+\|>|\w+|\s+|[^\w\s]")


class Encoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode_ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(t, len(self.vocab) + 1) for t in TOKEN_RE.findall(text)]


def make_row(enc: Encoder, i: int, words: int, split: str) -> dict[str, Any]:
    prompt = ("<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\n"
              + " ".join(f"w{i}x{j}" for j in range(words)) + "<|im_end|>\n<|im_start|>assistant\n")
    answer = f"<tool_call>\nrun {i}\n</tool_call><|im_end|>"
    p, c, t = (enc.encode_ids(s) for s in (prompt, answer, "\n"))
    return {"id": f"p3-{i:04d}", "split": split, "category": "coding", "source": "max-golden",
            "task_id": f"task-{i}", "text": prompt + answer + "\n",
            "loss_spans_char": [[len(prompt), len(prompt) + len(answer)]],
            "loss_spans_tok": [[len(p), len(p) + len(c)]], "n_tokens": len(p) + len(c) + len(t),
            "prompt_tokens": len(p), "boundary_stable": True, "tokenize": "segments",
            "messages": [], "tools": [], "truncated": None}


def make_data(d: Path, n_train: int, n_val: int) -> None:
    enc = Encoder()
    d.mkdir(parents=True, exist_ok=True)
    rows = [make_row(enc, i, 3 + (i % 4), "train") for i in range(n_train)]
    val = [make_row(enc, 5000 + i, 3 + (i % 3), "val") for i in range(n_val)]
    (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    # build.py's report: the dataset's own recorded Q6b approval
    (d / "report.json").write_text(json.dumps({
        "schema": "distill-phase3-report/1", "max_derived": True, "settings": {"teacher_model": "qwen3.8-max"},
        "data_policy": {"max_derived_allow": {"allowed": True, "approval": "operator Q6b test"}}}))


class Backend:
    def __init__(self, log: Path) -> None:
        self.w = 0.0
        self.grad = 0.0
        self.rng = random.Random(7)
        self.pending: list[list[int]] = []
        self.log = log
        self.step_n = 0
        self.spt = float(os.environ.get("FAKE_SEC_PER_TOKEN", "0"))

    def load(self, max_seq: int, revision: Any) -> None:
        pass

    def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
        if self.spt:
            time.sleep(len(ids) * self.spt)
        self.pending.append(list(ids))
        self.grad += scale
        return 2.0 / (1.0 + self.w) + self.rng.random() * 1e-3

    def step(self, lr: float) -> float:
        self.w += lr * 1e4 * self.grad
        self.step_n += 1
        with self.log.open("a") as fh:
            fh.write(json.dumps({"ids": self.pending, "lr": lr, "w": self.w, "pid": os.getpid()}) + "\n")
        self.pending, self.grad = [], 0.0
        return 0.5

    def zero_grad(self) -> None:
        self.pending, self.grad = [], 0.0

    def eval_loss(self, ids: Any, spans: Any) -> float:
        self.rng.random()  # an eval that touches the RNG: the trainer must keep that from training
        if self.spt:
            time.sleep(len(ids) * self.spt * 0.2)
        return 2.0 / (1.0 + self.w)

    def save(self, d: Path) -> None:
        (d / "adapter").mkdir(parents=True)
        (d / "adapter" / "w.json").write_text(json.dumps({"w": self.w}))
        (d / "optimizer.pt").write_bytes(b"\0" * 2048)

    def restore(self, d: Path) -> None:
        self.w = json.loads((d / "adapter" / "w.json").read_text())["w"]

    def rng_state(self) -> Any:
        return self.rng.getstate()

    def set_rng_state(self, state: Any) -> None:
        self.rng.setstate(state)

    def save_rng(self, path: Path, state: Any) -> None:
        path.write_bytes(pickle.dumps(state))

    def load_rng(self, path: Path) -> Any:
        return pickle.loads(path.read_bytes())

    def save_adapter(self, d: Path) -> None:
        (d / "adapter_config.json").write_text(json.dumps({"r": 32}))
        (d / "adapter_model.safetensors").write_text(json.dumps({"w": self.w}))

    def fingerprint(self) -> float:
        return self.w

    def peak_memory_gb(self) -> float:
        return 1.0

    def free(self) -> None:
        pass

    def versions(self) -> dict[str, Any]:
        return {"fake": "1"}


def main() -> int:
    if sys.argv[1] == "make-data":
        make_data(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        return 0
    script, argv = Path(sys.argv[1]), sys.argv[2:]
    spec = importlib.util.spec_from_file_location("distill_train_qlora", script)
    assert spec and spec.loader
    T = importlib.util.module_from_spec(spec)
    sys.modules["distill_train_qlora"] = T
    spec.loader.exec_module(T)
    enc = Encoder()
    margin = float(os.environ.get("FAKE_STOP_MARGIN_S", "0.5"))
    T.hf_encoder = lambda d: enc

    def no_template(d: Any) -> Any:
        raise ImportError("no chat template in the fake")

    T.template_refitter = no_template
    T.resolve_revision = lambda repo, rev: "b" * 40
    T.parse_stop_at = lambda v: (datetime.fromisoformat(v).timestamp() - margin) if v else None
    out = Path(argv[argv.index("--out") + 1])
    log = Path(os.environ["FAKE_STATE"]) / f"steps-{out.name}.jsonl"
    with (Path(os.environ["FAKE_STATE"]) / "trainer-calls.log").open("a") as fh:
        fh.write(json.dumps({"argv": argv, "pid": os.getpid()}) + "\n")
    (Path(os.environ["FAKE_STATE"]) / "trainer.pid").write_text(f"{os.getpid()}\n")
    return T.main(argv, backend_factory=lambda cfg: Backend(log))


if __name__ == "__main__":
    sys.exit(main())
