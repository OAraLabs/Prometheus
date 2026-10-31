"""Tests for the Phase 5 training code: train_qlora.py, launch_detached.sh,
feasibility_27b.py and modal_train.py.

No torch, no GPU, no Modal, no network. The training loop runs against a fake
backend whose "weights" are one float and whose RNG is a ``random.Random``,
so exact resume, step-down and stop behaviour can be checked step for step.
Modal is a fake module that records what the script registers.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import random
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _load("distill_train_qlora", "train_qlora.py")
F = _load("distill_feasibility_27b", "feasibility_27b.py")


def _load_modal_train(name: str, fake_modal: Any) -> Any:
    saved = sys.modules.get("modal", "absent")
    sys.modules["modal"] = fake_modal  # None makes `import modal` fail
    try:
        sys.modules.pop(name, None)
        return _load(name, "modal_train.py")
    finally:
        if saved == "absent":
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved


M = _load_modal_train("distill_modal_train", None)

# ── fixtures ───────────────────────────────────────────────────────────────

TOKEN_RE = re.compile(r"<\|[a-z_]+\|>|\w+|\s+|[^\w\s]")


class FakeEncoder:
    """Chat special tokens are single ids; other text splits into words,
    whitespace runs and symbols. Ids are assigned on first sight."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode_ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(t, len(self.vocab) + 1) for t in TOKEN_RE.findall(text)]


ENC = FakeEncoder()


def make_row(i: int, words: int = 5, source: str | None = "max-golden") -> dict[str, Any]:
    """A rendered row in build.py's shape: the assistant turn through
    <|im_end|> is the only loss span; spans counted with the segment recipe."""
    prompt = ("<|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\n"
              + " ".join(f"w{i}x{j}" for j in range(words))
              + "<|im_end|>\n<|im_start|>assistant\n")
    answer = f"<tool_call>\nrun {i}\n</tool_call><|im_end|>"
    p, c, t = (ENC.encode_ids(s) for s in (prompt, answer, "\n"))
    row = {"id": f"p3-{i:04d}", "split": "train", "category": "coding",
           "text": prompt + answer + "\n",
           "loss_spans_char": [[len(prompt), len(prompt) + len(answer)]],
           "loss_spans_tok": [[len(p), len(p) + len(c)]], "n_tokens": len(p) + len(c) + len(t),
           "prompt_tokens": len(p), "boundary_stable": True, "tokenize": "segments",
           "messages": [], "tools": [], "truncated": None, "_words": words}
    if source is not None:
        row["source"] = source
    return row


def fake_refit(row: dict[str, Any], budget: int) -> dict[str, Any] | None:
    """Drop words from the query until the row fits (None when it cannot)."""
    i = int(row["id"].split("-")[1])
    for w in range(row["_words"] - 1, 0, -1):
        new = make_row(i, w, row.get("source"))
        if new["n_tokens"] <= budget:
            new["truncated"] = {"original_tokens": row["n_tokens"], "refit_to": budget}
            return new
    return None


def write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class FakeOOM(RuntimeError):
    pass


FakeOOM.__name__ = "OutOfMemoryError"


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t


class FakeBackend:
    """One float of 'weights', an optimizer step counter and a Random as the
    RNG. Losses fall as the weight grows and carry RNG noise, so a resume that
    restores the wrong RNG or weights shows up as different numbers."""

    def __init__(self, *, oom_over: int | None = None, oom_on_call: int | None = None,
                 clock: Clock | None = None, sec_per_token: float = 0.0,
                 on_call: Any = None, load_s: float = 0.0) -> None:
        self.w = 0.0
        self.opt_steps = 0
        self.grad = 0.0
        self.rng = random.Random(7)
        self.pending: list[tuple[int, ...]] = []
        self.steps: list[tuple[tuple[int, ...], ...]] = []
        self.lrs: list[float] = []
        self.calls = 0
        self.oom_over, self.oom_on_call = oom_over, oom_on_call
        self.clock, self.sec_per_token = clock, sec_per_token
        self.on_call = on_call
        self.load_s = load_s
        self.loaded: tuple[int, Any] | None = None

    def load(self, max_seq: int, revision: Any) -> None:
        self.loaded = (max_seq, revision)
        if self.clock:
            self.clock.t += self.load_s

    def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
        self.calls += 1
        if self.on_call:
            self.on_call(self.calls)
        if (self.oom_over is not None and len(ids) > self.oom_over) or \
                self.oom_on_call == self.calls:
            raise FakeOOM("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.clock:
            self.clock.t += len(ids) * self.sec_per_token
        self.pending.append(tuple(ids))
        self.grad += scale
        return 2.0 / (1.0 + self.w) + self.rng.random() * 1e-3

    def step(self, lr: float) -> None:
        self.w += lr * 1e4 * self.grad
        self.opt_steps += 1
        self.lrs.append(lr)
        self.steps.append(tuple(self.pending))
        self.pending, self.grad = [], 0.0

    def zero_grad(self) -> None:
        self.pending, self.grad = [], 0.0

    def eval_loss(self, ids: Any, spans: Any) -> float:
        return 2.0 / (1.0 + self.w)

    def save(self, d: Path) -> None:
        (d / "adapter").mkdir(parents=True)
        (d / "adapter" / "w.json").write_text(json.dumps({"w": self.w, "opt": self.opt_steps}))

    def restore(self, d: Path) -> None:
        st = json.loads((d / "adapter" / "w.json").read_text())
        self.w, self.opt_steps = st["w"], st["opt"]

    def rng_state(self) -> Any:
        return self.rng.getstate()

    def set_rng_state(self, state: Any) -> None:
        self.rng.setstate(state)

    def save_rng(self, path: Path, state: Any) -> None:
        path.write_bytes(pickle.dumps(state))

    def load_rng(self, path: Path) -> Any:
        return pickle.loads(path.read_bytes())

    def save_adapter(self, d: Path) -> None:
        (d / "adapter.json").write_text(json.dumps({"w": self.w}))

    def fingerprint(self) -> float:
        return self.w

    def peak_memory_gb(self) -> float:
        return 1.0

    def free(self) -> None:
        pass

    def versions(self) -> dict[str, Any]:
        return {"fake": "1"}


def small_cfg(**kw: Any) -> Any:
    d = dict(max_seq=80, ladder=(80, 60, 40), grad_accum=2, epochs=2, save_every=3,
             keep_checkpoints=2, warmup_ratio=0.1, log_every=1)
    d.update(kw)
    return T.TrainConfig(**d)


def source(tmp_path: Path, rows: list[dict[str, Any]], refit: Any = None) -> Any:
    return T.JsonlSource(tmp_path / "train.jsonl", ENC, refit, rows=rows)


def trainer(tmp_path: Path, rows: list[dict[str, Any]], backend: FakeBackend, *,
            out: str = "run", cfg: Any = None, refit: Any = None, **kw: Any) -> Any:
    return T.Trainer(cfg or small_cfg(), source(tmp_path, rows, refit), tmp_path / out, backend,
                     **kw)


# ── config, schedule, step-down rules ─────────────────────────────────────


def test_default_config_is_the_runbooks() -> None:
    c = T.TrainConfig()
    assert (c.r, c.alpha, c.lr, c.epochs, c.max_seq) == (32, 32, 1e-4, 2, 16384)
    assert c.ladder == (16384, 12288, 8192)
    assert c.gradient_checkpointing == "unsloth" and c.save_every == 200 and c.load == "4bit"
    assert set(c.target_modules) >= {"q_proj", "v_proj", "down_proj"}
    assert c.validate() == []


def test_config_validation_and_hash() -> None:
    bad = T.TrainConfig(r=0, lr=1.0, max_seq=4096, micro_batch=2, ladder=(8192, 16384))
    errs = " ".join(bad.validate())
    for word in ("r and alpha", "lr", "ladder", "micro_batch"):
        assert word in errs
    a, b = T.TrainConfig(), T.TrainConfig(save_every=50, keep_checkpoints=9, max_seq=12288)
    assert a.sha() == b.sha()  # operational knobs and the stepped-down length are not hashed
    assert a.sha() != T.TrainConfig(lr=2e-4).sha()


def test_smoke_config_runs_200_examples_in_50_steps() -> None:
    s = T.smoke_config(T.TrainConfig())
    assert s.max_steps == 50 and s.epochs == 1 and s.grad_accum * 50 == T.SMOKE_EXAMPLES
    assert s.save_every == 25 and s.validate() == []


def test_lr_warms_up_then_decays_by_cosine_to_zero() -> None:
    lrs = [T.lr_at(s, 100, 10, 1e-4) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-5) and lrs[9] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[-1] < 1e-7 and T.lr_at(55, 100, 10, 1e-4) == pytest.approx(5e-5)


def test_step_down_ladder_and_oom_detection() -> None:
    assert T.next_seq(16384) == 12288 and T.next_seq(12288) == 8192
    assert T.next_seq(8192) is None
    assert T.is_oom(FakeOOM("x"))
    assert T.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not T.is_oom(RuntimeError("shape mismatch"))


# ── provenance guard ───────────────────────────────────────────────────────


@pytest.mark.parametrize("row,expect", [
    ({"source": "max-golden"}, True),
    ({"source": "max-gym-2026-10-31"}, True),
    ({"_meta": {"source": "local-gym", "model": "qwen3.8-max"}}, True),
    ({"meta": {"source": "Qwen3.8-MAX"}}, True),
    ({}, True),  # no provenance: counts as Max-derived
    # build.py's fallbacks for teacher rows: a source without the word proves nothing
    ({"source": "extra"}, True),
    ({"source": "golden"}, True),
    ({"source": "gym-2026-11-01"}, True),
    ({"source": "local-gym-2026-11"}, True),
    # only a builder-written max_derived: false clears a row (never one naming Max)
    ({"source": "local-gym-2026-11", "max_derived": False}, False),
    ({"source": "maximal-synthetic", "max_derived": False}, False),
    ({"_meta": {"source": "x", "max_derived": False}}, False),
    ({"source": "max-golden", "max_derived": False}, True),
    ({"source": "local-gym", "max_derived": "false"}, True),  # not a bool: no evidence
])
def test_max_derived_rows(row: dict[str, Any], expect: bool) -> None:
    assert T.is_max_derived(row) is expect


def test_the_builders_teacher_decides_rows_that_do_not_name_max(tmp_path: Path) -> None:
    assert T.is_max_derived({"source": "extra"}, teacher_max=True)
    assert T.is_max_derived({"source": "local-gym", "max_derived": False}, teacher_max=True)
    assert not T.is_max_derived({"source": "local-gym"}, teacher_max=False)
    assert T.is_max_derived({"source": "max-golden"}, teacher_max=False)
    data = tmp_path / "train.jsonl"
    assert T.builder_teacher_is_max(data) is None  # no report
    report = tmp_path / T.BUILD_REPORT
    report.write_text(json.dumps({"settings": {"teacher_model": "qwen3.8-max"}}))
    assert T.builder_teacher_is_max(data) is True
    report.write_text(json.dumps({"settings": {"teacher_model": "local-teacher-30b"}}))
    assert T.builder_teacher_is_max(data) is False
    report.write_text(json.dumps({"settings": {}}))
    assert T.builder_teacher_is_max(data) is None
    report.write_text("not json")
    assert T.builder_teacher_is_max(data) is None


def test_build_py_teacher_rows_without_max_in_their_source_are_refused(
        tmp_path: Path, cli: dict[str, Any]) -> None:
    """build.py keeps only its teacher's rows and labels an untagged Phase 2b
    trace "extra": with a Max teacher (or no report) the guard must count it."""
    rows = [make_row(i, 4, "extra" if i % 2 else "gym-2026-11-01") for i in range(8)]
    data = write_rows(tmp_path / "train.jsonl", rows)
    built: list[Any] = []
    assert T.main(base_args(data, cli),
                  backend_factory=lambda c: built.append(c) or FakeBackend()) == 3
    (tmp_path / T.BUILD_REPORT).write_text(
        json.dumps({"settings": {"teacher_model": "qwen3.8-max"}}))
    assert T.main(base_args(data, cli, "--smoke"), backend_factory=lambda c: FakeBackend()) == 3
    assert built == [] and not cli["out"].exists()
    meta = T.write_pretokenized(rows, ENC, tmp_path / "pt", ladder=(16384,),
                                teacher_max=T.builder_teacher_is_max(data))
    assert meta["max_derived_rows"] == 8
    no_report = T.write_pretokenized(rows, ENC, tmp_path / "pt2", ladder=(16384,))
    assert no_report["max_derived_rows"] == 8


def test_guard_refuses_max_rows_without_the_flag() -> None:
    assert not T.guard([True, False], approved=False).allowed
    assert "OPEN-QUESTIONS #6" in T.guard([True], approved=False).message()
    # Q6b: the flag and the dataset's recorded approval, both
    assert not T.guard([True, False], approved=True).allowed
    assert T.guard([True, False], approved=True, dataset_approval=APPROVAL_REF).allowed
    assert T.guard([False, False], approved=False).allowed


# ── rows, tokenization, data sources ──────────────────────────────────────


def test_tokenization_reproduces_the_rows_spans() -> None:
    row = make_row(1, 6)
    ids, spans = T.tokenize_row(row, ENC)
    assert len(ids) == row["n_tokens"] and spans == [tuple(row["loss_spans_tok"][0])]
    labels = [i if any(a <= k < b for a, b in spans) else T.IGNORE for k, i in enumerate(ids)]
    assert labels[: row["prompt_tokens"]] == [T.IGNORE] * row["prompt_tokens"]
    assert labels[-1] == T.IGNORE  # the trailing newline after <|im_end|> carries no loss
    bad = dict(row, n_tokens=row["n_tokens"] + 1)
    with pytest.raises(T.DataError, match="not the tokenizer"):
        T.tokenize_row(bad, ENC)


def test_row_shape_checks() -> None:
    row = make_row(2)
    for patch, msg in (({"loss_spans_char": []}, "no loss span"),
                       ({"loss_spans_char": [[5, 3]]}, "empty, overlapping"),
                       ({"tokenize": "joined"}, "segments"),
                       ({"loss_spans_tok": [[0, 999]]}, "outside")):
        with pytest.raises(T.DataError, match=msg):
            T.check_row({**row, **patch}, "x")


def test_fit_refits_or_drops_rows_over_a_shorter_length(tmp_path: Path) -> None:
    rows = [make_row(i, w) for i, w in enumerate((3, 10, 25))]
    big = rows[2]["n_tokens"]
    src = source(tmp_path, rows, fake_refit)
    full = src.fit(big)
    assert (full.kept, full.refit, full.dropped) == (3, 0, 0)
    short = src.fit(rows[1]["n_tokens"])
    assert (short.kept, short.refit, short.dropped) == (3, 1, 0)
    assert src.n_tokens(rows[2]["id"]) <= rows[1]["n_tokens"]
    no_refit = source(tmp_path, rows).fit(rows[1]["n_tokens"])
    assert (no_refit.kept, no_refit.dropped) == (2, 1)


def test_pretokenized_round_trip_with_per_length_variants(tmp_path: Path) -> None:
    rows = [make_row(i, w) for i, w in enumerate((3, 10, 25))] + [
        {**make_row(9, 4, "local-gym"), "max_derived": False}]
    ladder = (rows[2]["n_tokens"], rows[1]["n_tokens"], 20)
    meta = T.write_pretokenized(rows, ENC, tmp_path / "pt", ladder=ladder, refit=fake_refit,
                                data_sha256="d" * 64)
    assert meta["max_derived_rows"] == 3 and meta["rows"] == 4
    src = T.PretokSource(tmp_path / "pt")
    rep = src.fit(ladder[0])
    assert rep.kept == 4 and rep.refit == 0
    ids, spans = src.example(rows[0]["id"])
    assert list(ids) == list(T.tokenize_row(rows[0], ENC)[0]) and spans == [
        tuple(rows[0]["loss_spans_tok"][0])]
    mid = src.fit(ladder[1])
    assert mid.kept == 4 and mid.refit == 1 and src.n_tokens(rows[2]["id"]) <= ladder[1]
    low = src.fit(20)
    assert low.dropped == 4  # nothing fits 20 tokens, not even refit
    assert src.max_derived().count(True) == 3


# ── plan, groups, checkpoints ─────────────────────────────────────────────


def test_plan_is_seeded_per_epoch_and_stable() -> None:
    ids = [f"r{i}" for i in range(20)]
    p = T.epoch_plan(ids, 2, 3407)
    assert p == T.epoch_plan(reversed(ids), 2, 3407) and len(p) == 40
    assert sorted(p[:20]) == sorted(ids) and p[:20] != p[20:]
    assert p != T.epoch_plan(ids, 2, 1)


def test_groups_skip_ids_that_no_longer_fit() -> None:
    plan = ["a", "b", "c", "d", "e"]
    assert T.next_group(plan, 0, {"a", "b", "c", "d", "e"}, 2) == (["a", "b"], 2)
    assert T.next_group(plan, 0, {"a", "c", "e"}, 2) == (["a", "c"], 4)
    assert T.next_group(plan, 4, {"a", "c"}, 2) == ([], 5)
    assert T.remaining_steps(plan, 1, {"a", "c", "e"}, 2) == 1


def _ckpt(out: Path, step: int, *, complete: bool = True, state_step: int | None = None,
          partial: bool = False) -> Path:
    d = out / (f"checkpoint-{step:06d}" + (".partial" if partial else ""))
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"step": step if state_step is None else state_step}))
    if complete:
        (d / "COMPLETE").write_text("x")
    return d


def test_resume_picks_the_newest_complete_checkpoint(tmp_path: Path) -> None:
    out = tmp_path / "run"
    assert T.find_latest_checkpoint(out) is None
    _ckpt(out, 200)
    _ckpt(out, 400)
    _ckpt(out, 600, complete=False)  # killed before COMPLETE
    _ckpt(out, 800, partial=True)  # never renamed
    _ckpt(out, 1000, state_step=7)  # corrupt
    (out / "checkpoint-001200").mkdir()  # no state at all
    assert T.find_latest_checkpoint(out).name == "checkpoint-000400"


def test_prune_keeps_the_newest_and_clears_partials(tmp_path: Path) -> None:
    out = tmp_path / "run"
    for s in (100, 200, 300, 400):
        _ckpt(out, s)
    _ckpt(out, 500, partial=True)
    T.prune_checkpoints(out, 2)
    assert sorted(p.name for p in out.iterdir()) == ["checkpoint-000300", "checkpoint-000400"]


# ── the training loop (fake backend) ──────────────────────────────────────


def rows_n(n: int, words: int = 4) -> list[dict[str, Any]]:
    return [make_row(i, words + (i % 5)) for i in range(n)]


def test_full_run_trains_every_planned_example_and_checkpoints(tmp_path: Path) -> None:
    rows = rows_n(12)
    be = FakeBackend()
    tr = trainer(tmp_path, rows, be)
    tr.setup(resume=False)
    res = tr.run()
    assert res.status == "done" and tr.step == 12  # 12 rows x 2 epochs / 2 per step
    assert be.opt_steps == 12 and len(be.lrs) == 12 and tr.total_steps == 12
    trained = sorted(ids for step in be.steps for ids in step)
    assert len(trained) == 24 and len(set(trained)) == 12
    ckpts = sorted(p.name for p in (tmp_path / "run").iterdir() if p.name.startswith("checkpoint"))
    assert ckpts == ["checkpoint-000009", "checkpoint-000012"]  # keep 2
    st = json.loads((tmp_path / "run" / "checkpoint-000012" / "state.json").read_text())
    assert st["pos"] == 24 and st["val_loss"] is None and st["step"] == 12


def test_stop_and_resume_is_exact(tmp_path: Path) -> None:
    rows = rows_n(14)
    ref_be = FakeBackend()
    ref = trainer(tmp_path, rows, ref_be, out="ref")
    ref.setup(resume=False)
    ref.run()

    be1 = FakeBackend()
    t1 = trainer(tmp_path, rows, be1, out="cut")
    t1.setup(resume=False)
    res = t1.run(hook=lambda tr, rec: "window-close" if tr.step == 5 else None)
    assert res.status == "stopped" and res.checkpoint == "checkpoint-000005"
    be2 = FakeBackend()
    t2 = trainer(tmp_path, rows, be2, out="cut")
    t2.setup(resume=True)
    assert t2.resumed_from == "checkpoint-000005" and t2.step == 5
    t2.run()
    assert be1.steps + be2.steps == ref_be.steps
    assert be1.lrs + be2.lrs == ref_be.lrs
    assert t1.losses + t2.losses[5:] == ref.losses and t2.losses[:5] == t1.losses
    assert be2.w == pytest.approx(ref_be.w, rel=1e-12)


def test_sigterm_mid_group_checkpoints_the_last_full_step(tmp_path: Path) -> None:
    rows = rows_n(10)
    ref_be = FakeBackend()
    ref = trainer(tmp_path, rows, ref_be, out="ref")
    ref.setup(resume=False)
    ref.run()

    stop = T.StopFlag()
    # preflight is call 1; step k's examples are calls 2k and 2k+1: stop inside step 4
    be1 = FakeBackend(on_call=lambda n: setattr(stop, "reason", "SIGTERM") if n == 9 else None)
    t1 = trainer(tmp_path, rows, be1, out="cut", stop=stop)
    t1.setup(resume=False)
    res = t1.run()
    assert res.status == "stopped" and res.reason == "SIGTERM" and t1.step == 3
    assert be1.pending == []  # the partial group was dropped, not applied
    be2 = FakeBackend()
    t2 = trainer(tmp_path, rows, be2, out="cut")
    t2.setup(resume=True)
    t2.run()
    assert be1.steps + be2.steps == ref_be.steps and t2.losses == ref.losses


def test_preflight_oom_steps_down_and_the_resume_trains_shorter(tmp_path: Path) -> None:
    rows = [make_row(i, w) for i, w in enumerate((3, 4, 5, 6, 20, 25))]
    cfg = small_cfg(max_seq=80, ladder=(80, 60, 40))
    assert max(r["n_tokens"] for r in rows) <= 80
    limit = sorted(r["n_tokens"] for r in rows)[3]  # the two longest do not fit
    t1 = trainer(tmp_path, rows, FakeBackend(oom_over=limit), cfg=cfg)
    with pytest.raises(T.StepDown) as e:
        t1.setup(resume=False)
    assert (e.value.old, e.value.new) == (80, 60)
    rs = json.loads((tmp_path / "run" / T.RUN_STATE).read_text())
    assert rs["max_seq"] == 60 and rs["seq_history"][0]["where"] == "preflight"

    be = FakeBackend(oom_over=limit)
    t2 = trainer(tmp_path, rows, be, cfg=cfg)
    if max(r["n_tokens"] for r in rows) > 60 >= limit:
        t2.setup(resume=True)
    else:  # the shorter length still holds a too-long row: one more step-down
        with pytest.raises(T.StepDown):
            t2.setup(resume=True)
        t2 = trainer(tmp_path, rows, be, cfg=cfg)
        t2.setup(resume=True)
    assert t2.max_seq < 80 and be.loaded[0] == t2.max_seq
    assert t2.fit_report.dropped >= 1
    assert t2.run().status == "done"
    assert all(len(ids) <= t2.max_seq for step in be.steps for ids in step)


def test_mid_run_oom_resumes_from_the_last_checkpoint(tmp_path: Path) -> None:
    rows = rows_n(12)
    cfg = small_cfg(max_seq=80, ladder=(80, 60, 40))
    t1 = trainer(tmp_path, rows, FakeBackend(oom_on_call=12), cfg=cfg)
    t1.setup(resume=False)
    with pytest.raises(T.StepDown) as e:
        t1.run()
    assert e.value.step == 5  # step 6's first example hit the OOM
    t2 = trainer(tmp_path, rows, FakeBackend(), cfg=cfg)
    t2.setup(resume=True)
    assert t2.max_seq == 60 and t2.step == 3 and t2.resumed_from == "checkpoint-000003"
    sched = t2.run_state["schedule"]
    assert sched["max_seq"] == 60 and sched["from_step"] == 3 and sched["warmup_steps"] == 1
    assert t2.run().status == "done"


def test_resume_refuses_other_data_or_config(tmp_path: Path) -> None:
    rows = rows_n(6)
    t1 = trainer(tmp_path, rows, FakeBackend())
    t1.setup(resume=False)
    t1.run(hook=lambda tr, rec: "stop" if tr.step == 2 else None)
    with pytest.raises(T.UsageError, match="config_sha"):
        trainer(tmp_path, rows, FakeBackend(), cfg=small_cfg(lr=3e-4)).setup(resume=True)
    with pytest.raises(T.UsageError, match="data_sha256"):
        trainer(tmp_path, rows_n(7), FakeBackend()).setup(resume=True)
    with pytest.raises(T.UsageError, match="already holds a run"):
        trainer(tmp_path, rows, FakeBackend()).setup(resume=False)


def test_resume_loads_the_revision_the_run_started_on(tmp_path: Path) -> None:
    rows = rows_n(6)
    t1 = trainer(tmp_path, rows, FakeBackend(), revision="a" * 40)
    t1.setup(resume=False)
    t1.run(hook=lambda tr, rec: "stop")
    be = FakeBackend()
    trainer(tmp_path, rows, be, revision=None).setup(resume=True)
    assert be.loaded == (80, "a" * 40)


# ── the CLI: dry run, smoke, guard, step-down re-exec ─────────────────────


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.setattr(T, "hf_encoder", lambda d: ENC)

    def no_template(d: Any) -> Any:
        raise ImportError("jinja2")

    monkeypatch.setattr(T, "template_refitter", no_template)
    monkeypatch.setattr(T, "resolve_revision", lambda repo, rev: "b" * 40)
    tok = tmp_path / "tok"
    tok.mkdir()
    # tagged: most runs here train on approved Max-derived rows, and such a run's
    # --out must carry the tag (Q6b; test_q6b_a_max_run_needs_a_tagged_out)
    return {"tok": tok, "logs": tmp_path / "logs", "out": tmp_path / "out-max-derived"}


def base_args(data: Path, cli: dict[str, Any], *extra: str) -> list[str]:
    return ["--data", str(data), "--tokenizer-dir", str(cli["tok"]), "--out", str(cli["out"]),
            "--log-dir", str(cli["logs"]), *extra]


APPROVAL_REF = "operator Q6b 2026-10-30"


def approve_dataset(data: Path, ref: str = APPROVAL_REF) -> Path:
    """build.py's report beside *data*, recording the operator's Q6b allow
    (--allow-max-derived --max-derived-approval): what a run on Max-derived
    rows needs besides the trainer's own flag."""
    (data.parent / T.BUILD_REPORT).write_text(json.dumps({
        "schema": "distill-phase3-report/1", "max_derived": True,
        "settings": {"teacher_model": "qwen3.8-max"},
        "data_policy": {"max_derived_allow": {"allowed": True, "approval": ref}}}))
    return data


def test_dry_run_validates_previews_and_writes_a_manifest(tmp_path: Path, cli: dict[str, Any],
                                                          capsys: pytest.CaptureFixture[str]
                                                          ) -> None:
    data = write_rows(tmp_path / "train.jsonl", [make_row(i, 3 + i) for i in range(5)])
    args = ["--data", str(data), "--out", str(cli["out"]), "--log-dir", str(cli["logs"]),
            "--dry-run"]
    assert T.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("--- [LOSS: trained") == 2 and "[MASKED: no loss" in out
    assert "<tool_call>\nrun 0" in out and "a real run would be REFUSED" in out
    assert "torch" not in sys.modules and "unsloth" not in sys.modules
    m = json.loads((cli["logs"] / "manifest.json").read_text())
    assert m["mode"] == "dry-run" and m["data"]["max_derived_rows"] == 5
    assert m["data"]["sha256"] == T.sha256_files([data]) and m["seed"] == 3407
    assert m["base_model"]["id"] == T.STUDENT and "versions" in m
    # with the tokenizer: every row is re-tokenized and checked against its spans
    assert T.main(base_args(data, cli, "--dry-run")) == 0
    assert "tokenizer-checked" in capsys.readouterr().out
    bad = write_rows(tmp_path / "bad.jsonl", [dict(make_row(1), n_tokens=3)])
    assert T.main(base_args(bad, cli, "--dry-run")) == 2


def test_real_run_refuses_max_derived_data_without_the_flag(tmp_path: Path, cli: dict[str, Any]
                                                             ) -> None:
    data = write_rows(tmp_path / "train.jsonl", rows_n(8))
    built: list[Any] = []
    rc = T.main(base_args(data, cli), backend_factory=lambda c: built.append(c) or FakeBackend())
    assert rc == 3 and built == [] and not cli["out"].exists()
    assert T.main(base_args(data, cli, "--smoke"), backend_factory=lambda c: FakeBackend()) == 3


def test_base_must_be_staged_or_offline(tmp_path: Path) -> None:
    """A hub-id base would download ~19 GB into the HF cache with no disk-floor
    check: a real run takes a staged directory, the cache only (HF_HUB_OFFLINE=1),
    or an explicit --allow-hub-download."""
    staged = tmp_path / "qwen3.5-9b-base"
    staged.mkdir()
    assert T.base_source_problem(str(staged), env={}, allow_download=False) is None
    hub = T.base_source_problem(T.STUDENT, env={}, allow_download=False)
    assert hub and "hub id" in hub and "HF_HUB_OFFLINE=1" in hub
    assert T.base_source_problem(T.STUDENT, env={"HF_HUB_OFFLINE": "1"}, allow_download=False) is None
    assert T.base_source_problem(T.STUDENT, env={"HF_HUB_OFFLINE": "0"}, allow_download=False)
    assert T.base_source_problem(T.STUDENT, env={}, allow_download=True) is None
    missing = T.base_source_problem(str(tmp_path / "not-staged-yet"), env={"HF_HUB_OFFLINE": "1"},
                                    allow_download=True)
    assert missing and "not a model directory" in missing


def test_real_backend_refuses_a_hub_base_before_anything_loads(tmp_path: Path, cli: dict[str, Any],
                                                               monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]
                                                               ) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(8)))

    def no_backend(c: Any) -> Any:
        raise AssertionError("the real backend is never built for a refused base")

    monkeypatch.setattr(T, "UnslothBackend", no_backend)
    assert T.main(base_args(data, cli, T.APPROVAL_FLAG)) == 2
    assert "hub id" in capsys.readouterr().err and not cli["out"].exists()
    # the dry run says so without refusing
    assert T.main(base_args(data, cli, "--dry-run")) == 0
    assert "a real run would refuse this" in capsys.readouterr().out


def test_non_max_data_trains_without_the_flag(tmp_path: Path, cli: dict[str, Any]) -> None:
    rows = [make_row(i, 4, "local-gym-2026-11") for i in range(8)]
    data = write_rows(tmp_path / "train.jsonl", rows)
    # the builder's report names a non-Max teacher: that is the evidence
    (tmp_path / T.BUILD_REPORT).write_text(
        json.dumps({"settings": {"teacher_model": "local-teacher-30b"}}))
    rc = T.main(base_args(data, cli, "--grad-accum", "2", "--save-every", "3"),
                backend_factory=lambda c: FakeBackend())
    assert rc == 0
    m = json.loads((cli["out"] / "manifest.json").read_text())
    assert m["status"] == "done" and m["step"] == 8 and m["base_model"]["revision_resolved"] == "b" * 40
    assert (cli["out"] / "adapter" / "adapter.json").is_file()
    assert (cli["logs"] / "train.log").is_file()
    assert len((cli["logs"] / "metrics.jsonl").read_text().splitlines()) == 8


def with_val(tmp_path: Path, n: int = 6) -> list[str]:
    """A val split (ids disjoint from rows_n's) and the flag that passes it."""
    val = write_rows(tmp_path / "val.jsonl", [make_row(5000 + i) for i in range(n)])
    return ["--val", str(val)]


def test_smoke_runs_50_steps_round_trips_a_checkpoint_and_checks_the_loss(
        tmp_path: Path, cli: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(230)))
    flag = T.APPROVAL_FLAG
    val = with_val(tmp_path, 40)
    assert T.main(base_args(data, cli, "--smoke", flag, *val),
                  backend_factory=lambda c: FakeBackend()) == 0
    m = json.loads((cli["logs"] / "manifest.json").read_text())
    # the training half of the smoke gate only; the eval pipeline is still owed
    assert m["status"] == "smoke-train-pass" and m["smoke"]["steps"] == 50
    assert m["smoke"]["eval_pipeline"] == "not run"
    assert m["smoke"]["checkpoint_round_trip"] and m["smoke"]["loss_decreased"]
    gate = m["smoke"]["gate"]
    assert gate["passed"] and all(gate["checks"].values())
    assert set(gate["checks"]) == {"val_loss_decreased", "probe_loss_decreased", "losses_finite",
                                   "grad_norms_finite", "grad_norm_stable", "checkpoint_round_trip"}
    # step 0 (before any update) and both checkpoints; the round trip's replay kept no stale point
    for key in ("val", "probe"):
        assert [p[0] for p in gate[key]["points"]] == [0, 25, 50], key
        assert gate[key]["first_step"] == 0 and gate[key]["last"] < gate[key]["first"], key
    assert gate["probe"]["rows_improved"] == [20, 20]
    assert m["smoke"]["probe_last"] < m["smoke"]["probe_first"]
    assert m["smoke"]["tok_s"] is None or m["smoke"]["tok_s"] > 0
    rs = json.loads((cli["out"] / "run_state.json").read_text())
    assert [h["step"] for h in rs["val_history"]] == [0, 25, 50]
    assert all(h["probe_loss"] is not None for h in rs["val_history"])
    held = rs["held_out"]  # 40 val rows: the probe takes at most half, the val subset the rest
    assert held["probe"]["n"] == 20 and held["probe"]["seed"] == 3407 and held["val_subset"]["n"] == 20
    assert held["train_overlap"] == {"ids": 0, "tasks": 0}
    assert m["held_out"] == held and m["sample_order"]["plan"][0]["n"] == 200
    rt = m["smoke"]["round_trip"]
    assert rt["checkpoint"] == "checkpoint-000025" and rt["ok"] and rt["moved"]
    assert rt["after"] == rt["saved"] != rt["before"]  # trained past it, then brought back
    assert m["operator_approval_max_derived"] is True
    assert "PASS" in capsys.readouterr().out

    class Flat(FakeBackend):
        """No learning: the weights move, nothing held out gets better."""

        def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
            super().forward_backward(ids, spans, scale)
            return 1.0

        def eval_loss(self, ids: Any, spans: Any) -> float:
            return 1.0

    shutil.rmtree(cli["out"])
    assert T.main(base_args(data, cli, "--smoke", flag, *val), backend_factory=lambda c: Flat()) == 1
    m = json.loads((cli["logs"] / "manifest.json").read_text())
    checks = m["smoke"]["gate"]["checks"]
    assert m["status"] == "smoke-train-fail" and not m["smoke"]["loss_decreased"]
    assert not checks["val_loss_decreased"] and not checks["probe_loss_decreased"]
    assert checks["losses_finite"] and checks["checkpoint_round_trip"]

    # no --val: nothing held out shows the loss fell, so the gate cannot pass
    shutil.rmtree(cli["out"])
    assert T.main(base_args(data, cli, "--smoke", flag), backend_factory=lambda c: FakeBackend()) == 1
    m = json.loads((cli["logs"] / "manifest.json").read_text())
    assert m["status"] == "smoke-train-fail"
    assert "step 0" in m["smoke"]["gate"]["val"]["why"] and "step 0" in m["smoke"]["gate"]["probe"]["why"]


class Diverging(FakeBackend):
    """Each update makes things worse: the held-out loss climbs and the
    gradient grows, though every number stays finite."""

    def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
        super().forward_backward(ids, spans, scale)
        return 1.0 + 0.05 * self.opt_steps + self.rng.random() * 0.05

    def step(self, lr: float) -> float | None:
        super().step(lr)
        return 0.5 * 1.2 ** self.opt_steps

    def eval_loss(self, ids: Any, spans: Any) -> float:
        return 1.0 + 0.01 * self.opt_steps


class NaNLoss(FakeBackend):
    """One step's loss is NaN (its update is skipped; one alone is not divergence)."""

    def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
        loss = super().forward_backward(ids, spans, scale)
        if self.opt_steps == 30 and not getattr(self, "fired", False):
            self.fired = True
            return float("nan")
        return loss


class InfGrad(FakeBackend):
    """One step's gradient overflows: the backend reports the norm and skips the update."""

    def step(self, lr: float) -> float | None:
        if self.opt_steps == 12 and not getattr(self, "fired", False):
            self.fired = True
            self.zero_grad()
            return float("inf")
        super().step(lr)
        return 0.5


def test_smoke_fails_a_diverging_run_a_nan_loss_or_a_non_finite_grad_norm(
        tmp_path: Path, cli: dict[str, Any]) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(230)))
    args = base_args(data, cli, "--smoke", T.APPROVAL_FLAG, *with_val(tmp_path, 40))
    for backend, failing in ((Diverging, {"val_loss_decreased", "probe_loss_decreased",
                                          "grad_norm_stable"}),
                             (NaNLoss, {"losses_finite"}), (InfGrad, {"grad_norms_finite"})):
        shutil.rmtree(cli["out"], ignore_errors=True)
        assert T.main(args, backend_factory=lambda c, b=backend: b()) == 1, backend.__name__
        m = json.loads((cli["logs"] / "manifest.json").read_text())
        gate = m["smoke"]["gate"]
        assert m["status"] == "smoke-train-fail", backend.__name__
        assert {k for k, ok in gate["checks"].items() if not ok} == failing, (backend.__name__, gate)
    # the diverging run's numbers: held-out up, the median grad norm far past tenfold
    shutil.rmtree(cli["out"])
    T.main(args, backend_factory=lambda c: Diverging())
    gate = json.loads((cli["logs"] / "manifest.json").read_text())["smoke"]["gate"]
    assert gate["val"]["last"] > gate["val"]["first"] and gate["probe"]["rows_improved"][0] == 0
    assert "grew" in gate["grad_norm"]["why"]


# The first real 9B smoke (2026-10-31, 4090, build v3, 50 steps at 16,384): per-step
# train loss as its metrics.jsonl kept it, and the val loss at its two checkpoints.
SMOKE_9B_LOSSES = [
    1.1381, 1.0914, 1.2229, 1.1248, 1.0271, 1.0065, 0.7388, 0.6389, 1.2738, 0.8693,
    1.2397, 1.5734, 1.5185, 0.7081, 1.0084, 1.066, 1.6551, 1.0265, 0.8531, 0.788,
    0.7102, 1.2772, 0.7712, 0.5564, 1.2153, 1.1394, 0.8769, 0.8436, 0.9461, 0.608,
    0.8495, 1.0431, 1.2751, 0.8043, 0.7957, 1.0258, 0.7709, 0.5248, 1.2927, 1.1655,
    1.4288, 1.2251, 2.2623, 0.9897, 1.3095, 0.8493, 0.783, 1.24, 0.5884, 0.9264]
SMOKE_9B_VAL = [{"step": 25, "val_loss": 1.4576861033851052},
                {"step": 50, "val_loss": 1.443403955733422}]


def test_per_step_train_loss_is_not_gated_and_the_start_must_be_step_0() -> None:
    # the old rule failed this curve: first-10 mean 1.013 vs last-10 mean 1.160
    first, last = sum(SMOKE_9B_LOSSES[:10]) / 10, sum(SMOKE_9B_LOSSES[-10:]) / 10
    assert round(first, 3) == 1.013 and round(last, 3) == 1.160
    history = [{"step": 0, "val_loss": 1.47, "probe_loss": 1.52, "probe_per_row": [1.5, 1.6]},
               *({**h, "probe_loss": 1.50} for h in SMOKE_9B_VAL[:1]),
               {**SMOKE_9B_VAL[1], "probe_loss": 1.49, "probe_per_row": [1.4, 1.7]}]
    v = T.smoke_verdict(SMOKE_9B_LOSSES, history, [0.4] * 50, True)
    assert v["passed"], v
    assert not v["train_loss"]["gated"] and v["probe"]["rows_improved"] == [1, 2]
    assert "lower" in T.smoke_line(v) and T.smoke_line(v).endswith("PASS")
    # that smoke itself measured nothing before training: no start to compare with
    v = T.smoke_verdict(SMOKE_9B_LOSSES, SMOKE_9B_VAL, [None] * 50, True)
    assert not v["passed"] and v["val"]["why"] == "no val_loss at step 0 (before training)"
    # a held-out loss that rises, or a NaN train loss, fails the same curve
    worse = [dict(h) for h in history]
    worse[-1]["probe_loss"] = 1.53
    assert not T.smoke_verdict(SMOKE_9B_LOSSES, worse, [0.4] * 50, True)["checks"][
        "probe_loss_decreased"]
    nan = SMOKE_9B_LOSSES[:-1] + [float("nan")]
    assert not T.smoke_verdict(nan, history, [0.4] * 50, True)["checks"]["losses_finite"]
    assert not T.smoke_verdict(SMOKE_9B_LOSSES, history, [0.4] * 50, False)["passed"]


def test_held_out_and_grad_norm_check_edges() -> None:
    hc = T.held_out_check
    assert not hc([{"step": 0, "val_loss": 1.0}], "val_loss")["ok"]  # nothing after training
    assert not hc([{"step": 0, "val_loss": 1.0}, {"step": 9, "val_loss": 1.0}], "val_loss")["ok"]
    assert hc([{"step": 9, "val_loss": 0.9}, {"step": 0, "val_loss": 1.0}], "val_loss")["ok"]
    assert not hc([{"step": 3, "val_loss": 1.0}, {"step": 9, "val_loss": 0.5}], "val_loss")["ok"]
    assert not hc([{"step": 0, "val_loss": 1.0}, {"step": 9, "val_loss": float("nan")}],
                  "val_loss")["ok"]
    assert not hc([{"step": 0, "val_loss": 1.0}, {"step": 9, "val_loss": 0.99}], "val_loss",
                  min_drop=0.02)["ok"]
    # the last point decides, not the best one
    assert not hc([{"step": 0, "probe_loss": 1.0}, {"step": 5, "probe_loss": 0.5},
                   {"step": 9, "probe_loss": 1.01}], "probe_loss")["ok"]
    g = T.grad_norm_check
    assert g([0.5] * 40)["stable"] and g([None] * 40)["all_finite"]  # no norms reported: nothing to fail
    assert g([3.0] * 10 + [0.3] * 30)["stable"]  # falling is fine
    assert not g([0.3] * 10 + [0.3 * 20] * 30)["stable"]
    assert not g([0.5] * 39 + [float("inf")])["all_finite"]
    assert T.finite(1) and not T.finite(True) and not T.finite(None) and not T.finite(float("inf"))


def test_the_probe_batch_is_seeded_fixed_disjoint_and_never_trained(tmp_path: Path) -> None:
    ids = [f"v{i:03d}" for i in range(114)]
    probe = T.select_probe(ids, 3407)
    assert len(probe) == 32 and probe == T.select_probe(list(reversed(ids)), 3407)
    assert probe != T.select_probe(ids, 1)
    sub, pr = T.held_out_sets(ids, 3407)
    assert pr == probe and len(sub) == 64 and not set(sub) & set(pr)
    sub, pr = T.held_out_sets(ids[:20], 3407)  # a small val split: the probe takes half
    assert len(pr) == 10 and sorted(sub + pr) == ids[:20]
    rows = rows_n(12)
    val = T.JsonlSource(tmp_path / "val.jsonl", ENC, rows=[make_row(5000 + i) for i in range(40)])
    val.fit(80)
    t = trainer(tmp_path, rows, FakeBackend(), val=val)
    t.setup(resume=False)
    assert t.probe_ids == T.select_probe(val.kept(), 3407, 20) and not set(t.probe_ids) & set(t.plan)
    assert t.val_history[0]["step"] == 0 and len(t.val_history[0]["probe_per_row"]) == 20
    t.run(hook=lambda tr, rec: "stop" if tr.step == 3 else None)
    # a resume with another probe seed would compare different rows: refused
    with pytest.raises(T.UsageError, match="probe batch"):
        trainer(tmp_path, rows, FakeBackend(), val=val, probe_seed=1).setup(resume=True)
    t2 = trainer(tmp_path, rows, FakeBackend(), val=val)
    t2.setup(resume=True)
    assert [h["step"] for h in t2.val_history] == [0, 3]  # the step-0 point survives a resume
    # a held-out row from a task that also trains is refused, by id or by task
    same_task = [dict(make_row(5000 + i), task_id=f"t{i}") for i in range(10)]
    train_rows = [dict(r, task_id="t3" if i == 0 else f"x{i}") for i, r in enumerate(rows_n(12))]
    vs = T.JsonlSource(tmp_path / "v2.jsonl", ENC, rows=same_task)
    vs.fit(80)
    with pytest.raises(T.DataError, match="also train"):
        trainer(tmp_path, train_rows, FakeBackend(), val=vs, out="leak").setup(resume=False)
    dup = T.JsonlSource(tmp_path / "v3.jsonl", ENC, rows=rows_n(3))
    dup.fit(80)
    with pytest.raises(T.DataError, match="also train"):
        trainer(tmp_path, rows, FakeBackend(), val=dup, out="leak2").setup(resume=False)


def test_the_plan_is_shuffled_whatever_the_file_order(tmp_path: Path) -> None:
    """build.py writes train.jsonl grouped by category; the trainer never trains
    in file order: each epoch is a seeded permutation, recorded with the stats."""
    cats = ["coding"] * 60 + ["chat-mode"] * 20 + ["other"] * 20
    rows = [dict(make_row(i, 3 + i // 10), category=c) for i, c in enumerate(cats)]  # grouped + long
    t = trainer(tmp_path, rows, FakeBackend(), cfg=small_cfg(save_every=1000))
    t.setup(resume=False)
    order = json.loads((tmp_path / "run" / T.RUN_STATE).read_text())["order"]
    f, p0, p1 = order["file"], *order["plan"]
    assert order["seed"] == 3407 and order["epochs_differ"] and "seeded permutation" in order["sampler"]
    assert f["category_runs"] == 3 and f["category_longest_run"] == 60
    assert f["len_ac1"] > 0.8 and f["len_rank_corr_with_position"] > 0.8
    for p in (p0, p1):
        assert p["n"] == 100 and p["categories"] == 3
        assert abs(p["category_runs"] - p["category_runs_if_shuffled"]) < 15
        assert p["category_longest_run"] < 15
        assert abs(p["len_ac1"]) < 0.3 and abs(p["len_rank_corr_with_position"]) < 0.3
    # a sorted order is caught by the same stats
    by_len = sorted(t.kept, key=t.source.n_tokens)
    assert T.order_stats(by_len, t.source.n_tokens)["len_rank_corr_with_position"] > 0.95


def test_time_based_checkpoints_epoch_ends_and_progress(tmp_path: Path) -> None:
    clock = Clock()
    rows = rows_n(12)
    val = T.JsonlSource(tmp_path / "val.jsonl", ENC, rows=[make_row(5000 + i) for i in range(6)])
    val.fit(80)
    be = FakeBackend(clock=clock, sec_per_token=1.0)  # ~30 s per step
    cfg = small_cfg(save_every=1000, save_every_minutes=2.0, keep_checkpoints=10)
    t = trainer(tmp_path, rows, be, cfg=cfg, val=val, clock=clock.now)
    t.setup(resume=False)
    assert t.run().status == "done" and t.step == 12
    steps = sorted(int(p.name.split("-")[1]) for p in (tmp_path / "run").iterdir()
                   if p.name.startswith("checkpoint-"))
    gaps = [b - a for a, b in zip([0] + steps, steps)]
    assert 6 in steps and 12 in steps and len(steps) >= 4 and max(gaps) <= 6
    rs = json.loads((tmp_path / "run" / T.RUN_STATE).read_text())
    ep = rs["epochs_done"]
    assert [(e["epoch"], e["step"], e["of"]) for e in ep] == [(1, 6, 2), (2, 12, 2)]
    assert all(e["val_loss"] is not None and e["probe_loss"] is not None for e in ep)
    assert ep[1]["val_loss"] < ep[0]["val_loss"] < rs["val_history"][0]["val_loss"]
    assert {h["step"] for h in rs["val_history"]} >= {0, 6, 12}
    pr = rs["progress"]
    assert pr["step"] == 12 and pr["tokens_left"] == 0 and pr["tok_s"] > 0 and pr["eta_s"] == 0
    assert pr["epochs"] == 2 and pr["total_steps"] == 12
    # mid-run: the ETA is the tokens left at this process's own rate
    t2 = trainer(tmp_path, rows, FakeBackend(clock=clock, sec_per_token=1.0), cfg=cfg, out="r2",
                 clock=clock.now)
    t2.setup(resume=False)
    t2.run(hook=lambda tr, rec: "stop" if tr.step == 4 else None)
    pr = json.loads((tmp_path / "r2" / T.RUN_STATE).read_text())["progress"]
    assert pr["step"] == 4 and pr["eta_s"] == pytest.approx(pr["tokens_left"] / pr["tok_s"], rel=0.01)


def test_a_run_checkpoints_and_stops_under_the_disk_floor(tmp_path: Path) -> None:
    free = {"bytes": 50.0 * 2**30}
    t = trainer(tmp_path, rows_n(12), FakeBackend(), min_free_bytes=6 * 2**30,
                disk_free=lambda p: free["bytes"])
    t.setup(resume=False)

    def shrink(tr: Any, rec: Any) -> None:
        if tr.step == 2:
            free["bytes"] = 5.0 * 2**30

    res = t.run(hook=shrink)
    assert res.status == "stopped" and res.reason == "disk-floor" and res.step == 2
    assert T.find_latest_checkpoint(tmp_path / "run").name == "checkpoint-000002"


def test_three_non_finite_steps_in_a_row_stop_the_run_as_diverged(tmp_path: Path,
                                                                 cli: dict[str, Any]) -> None:
    class AllNaN(FakeBackend):
        def forward_backward(self, ids: Any, spans: Any, scale: float) -> float:
            loss = super().forward_backward(ids, spans, scale)
            return float("nan") if self.calls > 8 else loss

    t = trainer(tmp_path, rows_n(20), AllNaN())
    t.setup(resume=False)
    with pytest.raises(T.Diverged):
        t.run()
    assert t.nonfinite == 3 and t.backend.opt_steps < t.step  # skipped steps never updated
    # through the CLI: exit 1 and a "failed" manifest naming it
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(20)))
    assert T.main(base_args(data, cli, T.APPROVAL_FLAG, "--grad-accum", "2"),
                  backend_factory=lambda c: AllNaN()) == 1
    m = json.loads((cli["out"] / "manifest.json").read_text())
    assert m["status"] == "failed" and "diverged" in m["reason"]


class NoOpRestore(FakeBackend):
    """A restore that silently keeps the weights in memory (e.g. a state dict
    whose keys did not match)."""

    def restore(self, d: Path) -> None:
        st = json.loads((d / "adapter" / "w.json").read_text())
        self.opt_steps = st["opt"]


def test_smoke_round_trip_fails_when_the_restore_does_nothing(tmp_path: Path,
                                                              cli: dict[str, Any]) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(230)))
    rc = T.main(base_args(data, cli, "--smoke", T.APPROVAL_FLAG, *with_val(tmp_path)),
                backend_factory=lambda c: NoOpRestore())
    m = json.loads((cli["logs"] / "manifest.json").read_text())
    assert rc == 1 and m["status"] == "smoke-train-fail"
    assert m["smoke"]["loss_decreased"] and not m["smoke"]["checkpoint_round_trip"]
    assert m["smoke"]["round_trip"]["ok"] is False


def test_a_resume_refuses_a_restore_that_does_not_bring_the_weights_back(tmp_path: Path) -> None:
    rows = rows_n(12)
    t1 = trainer(tmp_path, rows, FakeBackend())
    t1.setup(resume=False)
    t1.run(hook=lambda tr, rec: "stop" if tr.step == 4 else None)
    st = json.loads((tmp_path / "run" / "checkpoint-000004" / "state.json").read_text())
    assert st["fingerprint"] == t1.backend.w
    with pytest.raises(T.RestoreError, match="checkpoint-000004"):
        trainer(tmp_path, rows, NoOpRestore()).setup(resume=True)
    t2 = trainer(tmp_path, rows, FakeBackend())
    t2.setup(resume=True)
    assert t2.restore_check["ok"] and t2.restore_check["moved"]


def test_a_stopped_smoke_is_no_verdict_and_resumes(tmp_path: Path, cli: dict[str, Any],
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(230)))
    args = base_args(data, cli, "--smoke", T.APPROVAL_FLAG, *with_val(tmp_path))
    for stop_step in (10, 30):  # in the first half, then in the second
        shutil.rmtree(cli["out"], ignore_errors=True)

        def stop_after(tr: Any, rec: Any, at: int = stop_step) -> str | None:
            return "SIGTERM" if tr.step == at else None  # a window close at step *at*

        orig = T.Trainer.run

        def run(self: Any, *, until_step: Any = None, hook: Any = None) -> Any:
            return orig(self, until_step=until_step, hook=hook or stop_after)

        monkeypatch.setattr(T.Trainer, "run", run)
        assert T.main(args, backend_factory=lambda c: FakeBackend()) == 0
        m = json.loads((cli["out"] / "manifest.json").read_text())
        assert m["status"] == "stopped" and m["reason"] == "SIGTERM" and m["step"] == stop_step
        monkeypatch.setattr(T.Trainer, "run", orig)
        assert T.main([*args, "--resume"], backend_factory=lambda c: FakeBackend()) == 0
        m = json.loads((cli["out"] / "manifest.json").read_text())
        assert m["status"] == "smoke-train-pass" and m["smoke"]["steps"] == 50
        assert m["smoke"]["checkpoint_round_trip"]


def test_oom_re_executes_with_resume(tmp_path: Path, cli: dict[str, Any],
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(6)))
    calls: list[list[str]] = []
    monkeypatch.setattr(T.os, "execv", lambda exe, args: calls.append(args))
    args = base_args(data, cli, T.APPROVAL_FLAG)
    rc = T.main(args, backend_factory=lambda c: FakeBackend(oom_over=0))
    assert rc == 1 and calls and calls[0][-1] == "--resume"
    assert calls[0][0] == sys.executable and calls[0][1].endswith("train_qlora.py")
    rs = json.loads((cli["out"] / T.RUN_STATE).read_text())
    assert rs["max_seq"] == 12288 and rs["seq_history"][0]["reason"] == "cuda_oom"


# ── launch_detached.sh ─────────────────────────────────────────────────────


def run_launcher(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(_DIR / "launch_detached.sh"), *args],
                          capture_output=True, text=True, timeout=30)


def test_launcher_prints_a_systemd_run_command_and_starts_nothing() -> None:
    p = run_launcher("--ts", "20261031T060000Z", "--resume", "--",
                     "--data", "d/train.jsonl", "--out", "data/distill/runs/x")
    assert p.returncode == 0, p.stderr
    line = p.stdout.splitlines()[0]
    assert line.startswith("systemd-run --user --unit=distill-train-20261031T060000Z ")
    assert "data/distill/logs/distill-train-20261031T060000Z/unit.log" in line
    assert "--log-dir" in line and line.rstrip().endswith("--resume")
    assert "TimeoutStopSec=110" in line and "KillSignal=SIGTERM" in line
    assert "print-only" in p.stdout


def test_launcher_rejects_missing_flags() -> None:
    assert run_launcher().returncode == 2
    assert run_launcher("--", "--data", "x").returncode == 2  # no --out
    assert run_launcher("--", "--out", "o", "--dry-run").returncode == 2
    assert run_launcher("--bogus").returncode == 2


# ── feasibility_27b.py ─────────────────────────────────────────────────────


def lengths_file(tmp_path: Path) -> Path:
    rows = [{"n_tokens": n, "truncated": ({"original_tokens": o} if o else None)}
            for n, o in ((3000, None), (7000, None), (10000, None), (14000, None),
                         (16384, 30000))]
    return write_rows(tmp_path / "train.jsonl", rows)


def test_feasibility_shares_and_tokens(tmp_path: Path) -> None:
    lengths = F.read_lengths(lengths_file(tmp_path))
    share, tokens = F.data_at(lengths, 8192, 2)
    assert share == pytest.approx(0.4) and tokens == (3000 + 7000 + 8192 * 3) * 2
    assert F.data_at(lengths, 16384, 1)[0] == pytest.approx(0.8)


def test_feasibility_dry_run_assumes_speed_and_measures_nothing(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]
                                                                ) -> None:
    def no_gpu(cmd: Any, timeout: float) -> Any:
        raise AssertionError("dry run must not start a probe")

    out = tmp_path / "rep.json"
    assert F.main(["--data", str(lengths_file(tmp_path)), "--out", str(out)], runner=no_gpu) == 0
    rep = json.loads(out.read_text())
    assert rep["mode"] == "dry-run" and rep["decision"]["verdict"] == "not-measured"
    assert all(r["fits"] is None and r["tps_source"] == "assumed" for r in rep["results"])
    assert "ASSUMED" in capsys.readouterr().out


# a window end far ahead, children run directly (no systemd scope)
WINDOW = ("--stop-at", "2099-01-01T06:30-05:00", "--no-scope")


def idle_gpu() -> int:
    return 300  # MiB: an idle card


def fake_probe(results: dict[int, dict[str, Any]]) -> Any:
    def runner(cmd: list[str], timeout: float) -> Any:
        assert cmd[0] == sys.executable and "--one" in cmd
        seq = int(cmd[cmd.index("--one") + 1])
        Path(cmd[cmd.index("--json-out") + 1]).write_text(json.dumps(results[seq]))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return runner


def test_feasibility_execute_local_viable(tmp_path: Path) -> None:
    res = {16384: {"fits": False, "error": "cuda_oom"}, 12288: {"fits": False, "error": "cuda_oom"},
           8192: {"fits": True, "peak_gb": 22.9, "tokens_per_s": 900.0},
           4096: {"fits": True, "peak_gb": 19.0, "tokens_per_s": 1000.0}}
    out = tmp_path / "rep.json"
    assert F.main(["--data", str(lengths_file(tmp_path)), "--execute", *WINDOW, "--out", str(out)],
                  runner=fake_probe(res), gpu_used=idle_gpu) == 0
    rep = json.loads(out.read_text())
    assert rep["decision"]["verdict"] == "local-viable" and rep["decision"]["max_seq"] == 8192
    r8 = next(r for r in rep["results"] if r["max_seq"] == 8192)
    assert r8["tps_source"] == "measured" and r8["wall_clock_h"] == pytest.approx(
        (3000 + 7000 + 8192 * 3) * 2 / 900 / 3600, abs=0.05)
    assert next(r for r in rep["results"] if r["max_seq"] == 16384)["wall_clock_h"] is None


def test_feasibility_rent_when_too_slow_or_only_short_fits(tmp_path: Path) -> None:
    lengths = [(16000, 16000)] * 10000
    slow = F.assess(lengths, F.SEQS, 2, {8192: {"fits": True, "tokens_per_s": 50.0},
                                         16384: {"fits": False}, 12288: {"fits": False},
                                         4096: {"fits": True, "tokens_per_s": 60.0}})
    assert F.decide(slow)["verdict"] == "rent"
    short = F.assess(lengths, F.SEQS, 2, {s: {"fits": s == 4096, "tokens_per_s": 5e5}
                                          for s in F.SEQS})
    assert F.decide(short) == {**F.decide(short), "verdict": "rent", "longest_fit": 4096}


def test_feasibility_records_a_crashed_child(tmp_path: Path) -> None:
    def crash(cmd: list[str], timeout: float) -> Any:
        return subprocess.CompletedProcess(cmd, -11, "", "Segmentation fault")

    a = F.build_parser().parse_args(["--data", "x"])
    got = F.execute((8192,), a, runner=crash, stop_at=4e9, scope=False, gpu_used=idle_gpu)
    assert got[8192]["fits"] is None and "without a result" in got[8192]["error"]


def test_feasibility_measured_keeps_the_ooms(tmp_path: Path) -> None:
    """An OOM has no tokens/sec, so its saved tok/s source stays "assumed";
    re-reading the report must keep it (and must not overwrite the probe)."""
    oom = {s: {"fits": False, "error": "cuda_oom"} for s in F.SEQS}
    probe = tmp_path / "feasibility-27b.json"
    data = str(lengths_file(tmp_path))
    assert F.main(["--data", data, "--execute", *WINDOW, "--out", str(probe)],
                  runner=fake_probe(oom), gpu_used=idle_gpu) == 0
    first = json.loads(probe.read_text())
    assert first["decision"]["verdict"] == "rent"
    assert F.main(["--data", data, "--measured", str(probe)]) == 0
    again = json.loads((tmp_path / "feasibility-27b-reestimate.json").read_text())
    assert again["decision"]["verdict"] == "rent"
    assert [r["fits"] for r in again["results"]] == [False] * 4
    assert json.loads(probe.read_text()) == first  # the probe itself is untouched
    assert F.main(["--data", data, "--measured", str(probe), "--out", str(probe)]) == 2
    mixed = {16384: {"fits": False, "error": "cuda_oom"},
             12288: {"fits": False, "error": "cuda_oom"},
             8192: {"fits": True, "peak_gb": 22.9, "tokens_per_s": 900.0},
             4096: {"fits": True, "peak_gb": 19.0, "tokens_per_s": 1000.0}}
    assert F.main(["--data", data, "--execute", *WINDOW, "--out", str(probe)],
                  runner=fake_probe(mixed), gpu_used=idle_gpu) == 0
    assert F.main(["--data", data, "--measured", str(probe)]) == 0
    again = json.loads((tmp_path / "feasibility-27b-reestimate.json").read_text())
    assert [r["fits"] for r in again["results"]] == [False, False, True, True]
    assert again["results"][0]["wall_clock_h"] is None
    assert again["decision"]["verdict"] == "local-viable"


def test_feasibility_is_incomplete_not_rent_when_8k_or_12k_was_not_measured() -> None:
    lengths = [(10000, 10000)] * 100
    timed_out = F.assess(lengths, F.SEQS, 2, {
        16384: {"fits": False, "error": "cuda_oom"},
        12288: {"fits": None, "error": "timeout after 1800s"},
        8192: {"fits": None, "error": "child exited -9 without a result: "},
        4096: {"fits": True, "tokens_per_s": 5e5}})
    d = F.decide(timed_out)
    assert d["verdict"] == "incomplete"
    assert [u["max_seq"] for u in d["unmeasured"]] == [12288, 8192] and "timeout" in d["say"]
    every_error = F.assess(lengths, F.SEQS, 2, {s: {"fits": None, "error": "boom"} for s in F.SEQS})
    assert F.decide(every_error)["verdict"] == "incomplete"
    # a viable length still decides, and says what longer length is unknown
    viable = F.assess(lengths, F.SEQS, 2, {16384: {"fits": None, "error": "timeout"},
                                           12288: {"fits": True, "tokens_per_s": 5e5}})
    d = F.decide(viable)
    assert d["verdict"] == "local-viable" and d["max_seq"] == 12288
    assert d["unmeasured_longer"] == [16384]
    assert F.is_oom(RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
                                 "cublasCreate(handle)"))
    assert not F.is_oom(RuntimeError("CUDA error: an illegal memory access"))


def test_feasibility_execute_keeps_to_the_window(tmp_path: Path) -> None:
    a = F.build_parser().parse_args(["--data", "x"])
    clock = Clock(1_000_000.0)
    started: list[list[str]] = []

    def probe(cmd: list[str], timeout: float) -> Any:
        started.append(cmd)
        clock.t += 1500
        Path(cmd[cmd.index("--json-out") + 1]).write_text(json.dumps({"fits": False}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    # room for two worst cases: 16k and 12k run, 8k and 4k are not started
    got = F.execute(F.SEQS, a, runner=probe, stop_at=clock.t + 2 * F.CHILD_TIMEOUT_S + 100,
                    scope=True, clock=clock.now, stamp="T1", gpu_used=idle_gpu)
    assert len(started) == 2
    assert started[0][:6] == ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                              "--unit=distill-feas27b-16384-T1"]
    assert started[0][6] == sys.executable
    assert got[8192]["fits"] is None and "--stop-at" in got[8192]["error"]
    assert F.decide(F.assess([(9000, 9000)], F.SEQS, 2, got))["verdict"] == "incomplete"
    # --execute without a window end, or without a report path, is refused before anything starts
    assert F.main(["--data", str(lengths_file(tmp_path)), "--execute", "--no-scope", "--out",
                   str(tmp_path / "r.json")], runner=probe, gpu_used=idle_gpu) == 2
    assert F.main(["--data", str(lengths_file(tmp_path)), "--execute", *WINDOW],
                  runner=probe, gpu_used=idle_gpu) == 2
    assert len(started) == 2 and not (tmp_path / "r.json").exists()


def test_feasibility_stops_probing_when_a_restore_stops_a_child() -> None:
    a = F.build_parser().parse_args(["--data", "x"])
    started: list[int] = []

    def probe(cmd: list[str], timeout: float) -> Any:
        started.append(int(cmd[cmd.index("--one") + 1]))
        return subprocess.CompletedProcess(cmd, -15, "", "")  # systemctl stop on the scope

    got = F.execute(F.SEQS, a, runner=probe, stop_at=4e9, scope=True, gpu_used=idle_gpu)
    assert started == [16384]
    assert "stopped" in got[16384]["error"]
    assert all(got[s]["fits"] is None and "not probed" in got[s]["error"] for s in F.SEQS[1:])


def test_feasibility_starts_nothing_while_the_gpu_is_taken(tmp_path: Path) -> None:
    """A hand run before the swap (the business model server still on the card,
    ~20 GiB) or with nvidia-smi unreadable starts no child."""
    a = F.build_parser().parse_args(["--data", "x"])
    started: list[list[str]] = []

    def probe(cmd: list[str], timeout: float) -> Any:
        started.append(cmd)
        raise AssertionError("no child may start")

    busy = F.execute(F.SEQS, a, runner=probe, stop_at=4e9, scope=False, gpu_used=lambda: 20306)
    assert not started
    assert all(busy[s]["fits"] is None and "20306 MiB" in busy[s]["error"]
               and busy[s]["error"].startswith("not probed") for s in F.SEQS)
    unknown = F.execute(F.SEQS, a, runner=probe, stop_at=4e9, scope=False, gpu_used=lambda: None)
    assert not started and all("unreadable" in unknown[s]["error"] for s in F.SEQS)
    assert F.decide(F.assess([(9000, 9000)], F.SEQS, 2, busy))["verdict"] == "incomplete"
    # the limit is a flag; the job runner's own check uses the same 1500 MiB
    a2 = F.build_parser().parse_args(["--data", "x", "--gpu-max-used-mib", "25000"])
    ok = F.execute((4096,), a2, runner=fake_probe({4096: {"fits": True, "tokens_per_s": 1.0}}),
                   stop_at=4e9, scope=False, gpu_used=lambda: 20306)
    assert ok[4096]["fits"] is True


def test_feasibility_cpu_spill_is_does_not_fit() -> None:
    """transformers' bitsandbytes refusal of a CPU device map, and Unsloth's
    rewrite of it, mean the model does not fit: a measurement, not an error."""
    hf = ValueError("Some modules are dispatched on the CPU or the disk. Make sure you have enough "
                    "GPU RAM to fit the quantized model.")
    us = ValueError("Unsloth: /m does not fit in GPU memory (cuda:0 has 1.20 GB free), so "
                    "transformers placed part of it on the CPU, which bitsandbytes cannot quantize.")
    assert F.oom_kind(hf) == F.oom_kind(us) == "cpu_spill" and F.is_oom(hf) and F.is_oom(us)
    assert F.oom_kind(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) == "cuda_oom"
    assert F.oom_kind(ValueError("rope_scaling must be a dict")) is None
    lengths = [(9000, 9000)] * 10
    spilled = F.assess(lengths, F.SEQS, 2, {s: {"fits": False, "error": "cpu_spill"} for s in F.SEQS})
    assert F.decide(spilled)["verdict"] == "rent"
    assert all(r.wall_clock_h is None for r in spilled)


def test_feasibility_offload_layers_replaces_gpu_gib_and_reaches_the_child() -> None:
    a = F.build_parser().parse_args(["--data", "x", "--offload-layers", "auto"])
    assert a.offload_layers == "auto"
    assert F.build_parser().parse_args(["--data", "x", "--offload-layers", "6"]).offload_layers == 6
    for bad in (["--offload-layers", "0"], ["--offload-layers", "many"], ["--gpu-gib", "18"]):
        with pytest.raises(SystemExit):
            F.build_parser().parse_args(["--data", "x", *bad])
    seen: list[list[str]] = []

    def probe(cmd: list[str], timeout: float) -> Any:
        seen.append(cmd)
        Path(cmd[cmd.index("--json-out") + 1]).write_text(json.dumps({"fits": False}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    F.execute((4096,), a, runner=probe, stop_at=4e9, scope=False, gpu_used=idle_gpu)
    assert seen[0][seen[0].index("--offload-layers") + 1] == "auto"
    F.execute((4096,), F.build_parser().parse_args(["--data", "x"]), runner=probe, stop_at=4e9,
              scope=False, gpu_used=idle_gpu)
    assert "--offload-layers" not in seen[1]


def test_feasibility_children_run_offline_unless_told_otherwise() -> None:
    env = F.child_env({"PATH": "/usr/bin"})
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["PATH"] == "/usr/bin"
    assert F.child_env({"HF_HUB_OFFLINE": "0"})["HF_HUB_OFFLINE"] == "0"


def test_feasibility_probes_the_adapter_the_trainer_trains() -> None:
    """Same LoRA targets as train_qlora.py, including the gated-deltanet
    projections of the linear-attention layers (48 of the 27B's 64 layers,
    24 of the 9B's 32)."""
    assert tuple(F.TARGET_MODULES) == tuple(T.TARGET_MODULES)
    assert {"in_proj_qkv", "in_proj_z", "out_proj"} <= set(T.TARGET_MODULES)
    assert {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"} <= set(
        T.TARGET_MODULES)
    # the vision tower's linear layers are not targeted
    assert not {"qkv", "proj", "linear_fc1", "linear_fc2"} & set(T.TARGET_MODULES)
    assert T.TrainConfig().target_modules == T.TARGET_MODULES


# ── modal_train.py: cost math, ledger, budget ─────────────────────────────


def test_rates_are_modal_list_prices_plus_cpu_and_memory() -> None:
    h100 = M.GPUS["H100"]
    assert h100.usd_per_h == pytest.approx(3.9492)
    assert M.rate_per_s(h100) * 3600 == pytest.approx(3.9492 + 4 * 0.04716 + 48 * 0.007992, rel=1e-4)
    assert M.GPUS["L40S"].vram_gb == 48 and M.GPUS["L40S"].usd_per_s == 0.000542
    assert M.likely_tps(h100) == pytest.approx(0.25 * 989.5e12 / (6 * 27e9))


def test_estimate_splits_long_runs_into_calls_and_adds_their_load() -> None:
    g = M.GPUS["H100"]
    short = M.estimate(g, 10_000_000, 1000.0, load_s=900)
    assert short.calls == 1 and short.train_s == 10_000
    long = M.estimate(g, 200_000_000, 1000.0, load_s=900)
    assert long.calls == 3  # 200,000 s of training > 2 x one 24 h call
    assert long.overhead_s >= 3 * (900 + M.STARTUP_ALLOWANCE_S)
    assert long.usd == pytest.approx(long.seconds * M.rate_per_s(g) + M.EGRESS_USD_PER_GIB)


def test_cost_table_ranks_by_cost_per_run_and_brackets_likely() -> None:
    rows = M.cost_table(M.scenario_tokens())
    assert [r["rank"] for r in rows] == list(range(1, len(M.GPUS) + 1))
    likely = [r["usd"]["likely"] for r in rows]
    assert likely == sorted(likely)
    for r in rows:
        assert r["usd"]["low"] < r["usd"]["likely"] < r["usd"]["high"]
    table = M.markdown_table(rows)
    for key in M.GPUS:
        assert f"(`{key}`)" in table


def test_ledger_counts_every_call_including_lost_ones(tmp_path: Path) -> None:
    led = M.Ledger(tmp_path / "ledger.json")
    s1 = led.open("H100", 0.001, 1000.0)
    led.close(s1, "stop:call_timeout", 2000.0)
    s2 = led.open("H100", 0.001, 3000.0)
    s2.last_seen = 3500.0  # preempted: never closed
    led.save()
    again = M.Ledger.load(tmp_path / "ledger.json")
    expect = ((1000 + M.STARTUP_ALLOWANCE_S) + (500 + M.STARTUP_ALLOWANCE_S + M.HEARTBEAT_S)) * 0.001
    assert again.spent_usd() == pytest.approx(expect)
    s3 = again.open("H100", 0.001, 5000.0)
    assert again.spent_usd(5100.0, s3) == pytest.approx(expect + (100 + M.STARTUP_ALLOWANCE_S) * 0.001)


def test_budget_stops_before_the_cap_and_the_deadlines(tmp_path: Path) -> None:
    clock = Clock(0.0)
    led = M.Ledger(tmp_path / "l.json")
    seg = led.open("H100", 0.01, 0.0)
    b = M.Budget(led, seg, max_usd=100.0, deadline=50_000.0, seg_deadline=40_000.0,
                 commit=lambda: None, clock=clock.now)
    assert b.check() is None
    clock.t = 100.0 / 0.01 - M.STARTUP_ALLOWANCE_S - 2 * M.SAVE_S  # spent + reserve == cap
    assert b.check() == "max_usd"
    b.max_usd = 1e9
    clock.t = 40_000.0 - M.DEADLINE_MARGIN_S
    assert b.check() == "call_timeout"
    clock.t = 50_000.0 - M.DEADLINE_MARGIN_S
    assert b.check() == "max_hours"


def test_projection_counts_further_calls() -> None:
    one = M.project(spent_usd=5.0, remaining_tokens=1_000_000, tps=100.0, rate=0.001,
                    seg_left_s=80_000, load_s=600, remaining_saves=2)
    assert one.calls == 1 and one.usd == pytest.approx(5.0 + (10_000 + 3 * M.SAVE_S) * 0.001
                                                       + M.EGRESS_USD_PER_GIB)
    two = M.project(spent_usd=0.0, remaining_tokens=10_000_000, tps=100.0, rate=0.001,
                    seg_left_s=50_000, load_s=600, remaining_saves=2)
    assert two.calls == 2 and two.seconds > 100_000 + 600
    assert M.project(spent_usd=0, remaining_tokens=1, tps=0, rate=1, seg_left_s=1, load_s=1,
                     remaining_saves=0).usd == float("inf")


# ── modal_train.py: one GPU call, with the real Trainer on the fake backend


def segment_env(tmp_path: Path, *, n_rows: int = 40, sec_per_token: float = 1.0,
                slow_after: int | None = None, load_s: float = 0.0) -> dict[str, Any]:
    clock = Clock(1_000_000.0)
    rows = rows_n(n_rows)
    commits: list[int] = []
    run_dir = tmp_path / "runs" / "r1"
    backends: list[FakeBackend] = []

    def make_trainer(job: dict[str, Any]) -> Any:
        be = FakeBackend(clock=clock, sec_per_token=sec_per_token, load_s=load_s)
        if slow_after is not None:
            orig = be.forward_backward

            def fb(ids: Any, spans: Any, scale: float) -> float:
                if be.opt_steps >= slow_after:
                    clock.t += len(ids) * 2.0  # 3x slower after calibration
                return orig(ids, spans, scale)
            be.forward_backward = fb  # type: ignore[method-assign]
        backends.append(be)
        cfg = small_cfg(max_seq=80, ladder=(80, 60, 40), epochs=1, save_every=4)
        return T.Trainer(cfg, source(tmp_path, rows), run_dir, be, clock=clock.now,
                         on_save=lambda p: commits.append(1))

    job = {"run_id": "r1", "gpu": "H100", "rate_per_s": 0.001, "max_usd": 100.0,
           "deadline": clock.t + 10 * 86400, "call_timeout_s": 86400, "calib_steps": 5,
           "calib_skip": 1, "base_revision": "c" * 40,
           "expect_load_s": load_s, "expect_step_s": 0.0}
    return {"clock": clock, "job": job, "make": make_trainer, "run_dir": run_dir,
            "commits": commits, "backends": backends}


def call(env: dict[str, Any], call_id: str = "", max_derived_rows: int = 0,
         **job: Any) -> dict[str, Any]:
    return M.run_segment({**env["job"], **job}, make_trainer=env["make"],
                         commit=lambda: env["commits"].append(0), run_dir=env["run_dir"],
                         max_derived_rows=max_derived_rows, clock=env["clock"].now,
                         call_id=call_id)


def test_segment_calibrates_then_trains_to_the_end(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    res = call(env)
    assert res["status"] == "done" and res["step"] == 20
    cal = res["calibration"]
    assert cal["decision"] == "continue" and cal["at_step"] == 5 and cal["tokens_per_s"] == 1.0
    assert (env["run_dir"] / "adapter" / "adapter.json").is_file()
    man = json.loads((env["run_dir"] / "manifest.json").read_text())
    assert man["image"]["ref"] == M.UNSLOTH_IMAGE_REF and man["versions"] == {"fake": "1"}
    led = M.Ledger.load(env["run_dir"] / "ledger.json")
    assert led.segments[0].closed and led.segments[0].status == "done"
    assert res["spent_usd"] <= env["job"]["max_usd"]


def test_segment_stops_after_calibration_when_the_run_would_pass_the_cap(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    res = call(env, max_usd=1.2)  # calibration costs < $1; the whole run ~$1.6
    assert res["status"] == "over_cap" and res["step"] == 5
    assert res["calibration"]["projection"]["usd"] > 1.2
    assert T.find_latest_checkpoint(env["run_dir"]).name == "checkpoint-000005"
    assert res["spent_usd"] <= 1.2


def test_segment_hard_cap_stops_a_run_that_slows_down(tmp_path: Path) -> None:
    env = segment_env(tmp_path, slow_after=5)
    res = call(env, max_usd=3.0)  # calibration projects ~$2.3; at 3x the run needs ~$5
    assert res["status"] == "stop:max_usd" and 5 < res["step"] < 20
    assert res["spent_usd"] <= 3.0
    assert T.find_latest_checkpoint(env["run_dir"]).name == f"checkpoint-{res['step']:06d}"


def test_segment_refuses_max_data_before_loading_whatever_the_job_says(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    res = call(env, max_derived_rows=40)
    assert res["status"] == "refused" and env["backends"] == [] and "Q6b" in res["reason"]
    # Q6b: no approval in the job lets Max-derived data train on a rented GPU
    res = call(env, max_derived_rows=1, approved=True)
    assert res["status"] == "refused" and env["backends"] == []


def test_segments_chain_across_call_timeouts_without_recalibrating(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    first = call(env, call_timeout_s=M.DEADLINE_MARGIN_S + 700)  # ~8 steps of ~82 s
    assert first["status"] == "stop:call_timeout" and 5 < first["step"] < 20
    second = call(env)
    assert second["status"] == "done" and second["step"] == 20
    led = M.Ledger.load(env["run_dir"] / "ledger.json")
    assert len(led.segments) == 2
    assert [c.get("recheck", False) for c in led.calibrations] == [False, True]  # no new steps


def test_a_relaunch_after_over_cap_rechecks_instead_of_training_to_the_cap(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    first = call(env, max_usd=1.2)
    assert first["status"] == "over_cap" and first["step"] == 5
    again = call(env, max_usd=1.5)  # still short of the ~$2.7 the projection needs
    assert again["status"] == "over_cap" and again["step"] == 5  # no step was trained
    assert again["calibration"]["recheck"] and again["spent_usd"] <= 1.5
    led = M.Ledger.load(env["run_dir"] / "ledger.json")
    assert sum(not c.get("recheck") for c in led.calibrations) == 1
    done = call(env, max_usd=10.0)  # a raised cap goes on from the checkpoint
    assert done["status"] == "done" and done["step"] == 20


def test_the_cap_reserves_the_load_and_the_first_step(tmp_path: Path) -> None:
    env = segment_env(tmp_path, load_s=1800)
    # expecting an 1800 s load: the call stops before loading anything
    res = call(env, max_usd=0.6)
    assert res["status"] == "stop:max_usd" and env["backends"] == []
    assert res["spent_usd"] <= 0.6
    # a load slower than expected is caught after it, before any step
    env2 = segment_env(tmp_path / "b", load_s=1800)
    res = call(env2, max_usd=2.2, expect_load_s=0.0)
    assert res["status"] == "stop:max_usd" and res["step"] == 0
    assert env2["backends"][0].opt_steps == 0


def live_segment(env: dict[str, Any], *, call_id: str = "other", age_s: float = 60.0,
                 loaded: bool = True) -> None:
    led = M.Ledger(env["run_dir"] / "ledger.json")
    seg = led.open("H100", 0.001, env["clock"].t - age_s - 1, call_id)
    seg.last_seen, seg.loaded = env["clock"].t - age_s, loaded
    led.save()


def test_a_second_call_on_a_live_run_is_busy_and_writes_nothing(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    live_segment(env)
    before = (env["run_dir"] / "ledger.json").read_text()
    res = call(env, call_id="mine")
    assert res["status"] == "busy" and env["backends"] == []
    assert (env["run_dir"] / "ledger.json").read_text() == before
    # still loading: the load's worst case counts too
    env2 = segment_env(tmp_path / "b")
    live_segment(env2, age_s=M.LEASE_S + 60, loaded=False)
    assert call(env2, call_id="mine")["status"] == "busy"
    # silent past the lease: dead, the run is free
    env3 = segment_env(tmp_path / "c")
    live_segment(env3, age_s=M.LEASE_S + 1)
    assert call(env3, call_id="mine")["status"] == "done"


def test_a_preempted_call_restarting_under_its_own_id_is_not_busy(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    live_segment(env, call_id="fc-1")
    res = call(env, call_id="fc-1")
    assert res["status"] == "done"
    led = M.Ledger.load(env["run_dir"] / "ledger.json")
    assert led.segments[0].status == "lost" and not led.segments[0].closed
    assert led.busy(env["clock"].t) is None


def test_a_call_on_other_weights_or_config_is_refused_not_raised(tmp_path: Path) -> None:
    env = segment_env(tmp_path)
    first = call(env, call_timeout_s=M.DEADLINE_MARGIN_S + 700)
    assert first["status"] == "stop:call_timeout"
    moved = call(env, base_revision="d" * 40)  # main moved since the launch
    assert moved["status"] == "refused" and "c" * 40 in moved["reason"]
    assert len(env["backends"]) == 1  # refused before loading

    def other_config(job: dict[str, Any]) -> Any:
        cfg = small_cfg(max_seq=80, ladder=(80, 60, 40), epochs=1, save_every=4, lr=3e-4)
        return T.Trainer(cfg, source(tmp_path, rows_n(40)), env["run_dir"], FakeBackend(),
                         clock=env["clock"].now)

    env["make"] = other_config
    res = call(env)
    assert res["status"] == "refused" and "config_sha" in res["reason"]
    led = M.Ledger.load(env["run_dir"] / "ledger.json")
    assert all(s.closed for s in led.segments) and led.busy(env["clock"].t) is None


# ── modal_train.py: the local side ────────────────────────────────────────


class FakeOps:
    def __init__(self, results: list[Any], ledger: dict[str, Any] | None = None,
                 sha: str = "c" * 40) -> None:
        self.results = list(results)
        self.log: list[tuple[str, Any]] = []
        self.ledger = ledger
        self.sha = sha

    def upload_data(self, local: Path, name: str, meta: dict[str, Any]) -> str:
        self.log.append(("upload", name))
        return "uploaded"

    def prefetch_model(self, repo: str, revision: str | None) -> dict[str, Any]:
        self.log.append(("prefetch", (repo, revision)))
        sha = revision or self.sha
        return {"dir": "/models/x@" + sha, "sha": sha}

    def read_ledger(self, run_id: str) -> dict[str, Any] | None:
        return self.ledger

    def train_call(self, job: dict[str, Any], gpu: str, timeout_s: int) -> dict[str, Any]:
        self.log.append(("train", (gpu, timeout_s, dict(job))))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def download(self, run_id: str, dest: Path) -> list[str]:
        self.log.append(("download", run_id))
        return ["adapter/adapter_model.safetensors"]


def pretok_dir(tmp_path: Path, *, sources: tuple[str, ...] = ("local-gym",),
               teacher_max: bool | None = False) -> Path:
    """Pre-tokenized rows; by default clean (non-Max rows, a non-Max teacher):
    Max-derived data never rents (Q6b)."""
    rows = [make_row(i, 4, sources[i % len(sources)]) for i in range(10)]
    T.write_pretokenized(rows, ENC, tmp_path / "pt", ladder=(16384, 12288, 8192),
                         teacher_max=teacher_max)
    return tmp_path / "pt"


def rent_args(data: Path, **kw: Any) -> Any:
    return M.RentArgs(**{"data": data, "max_usd": 150.0, "launch": True,
                         "run_id": "r1", **kw})


def test_plan_is_print_only_without_launch(tmp_path: Path) -> None:
    lines: list[str] = []
    a = rent_args(pretok_dir(tmp_path), launch=False)
    assert M.orchestrate(a, None, echo=lines.append) == 0
    assert any("print-only" in s for s in lines) and any("tok/s (assumed)" in s for s in lines)


def test_rent_refuses_max_data_without_approval(tmp_path: Path) -> None:
    ops = FakeOps([])
    lines: list[str] = []
    maxd = pretok_dir(tmp_path, sources=("max-golden",), teacher_max=None)
    assert M.orchestrate(rent_args(maxd), ops, echo=lines.append) == 3
    assert ops.log == [] and "OPEN-QUESTIONS #6" in lines[-1] and "Q6b" in lines[-1]
    # no evidence: Max-derived
    unknown = pretok_dir(tmp_path / "b", sources=("local-gym",), teacher_max=None)
    assert M.orchestrate(rent_args(unknown, launch=False), None, echo=print) == 3
    ok = pretok_dir(tmp_path / "c", sources=("local-gym",), teacher_max=False)
    assert M.orchestrate(rent_args(ok, launch=False), None, echo=print) == 0


def test_rent_refuses_a_cap_below_the_optimistic_estimate(tmp_path: Path) -> None:
    ops = FakeOps([])
    assert M.orchestrate(rent_args(pretok_dir(tmp_path), max_usd=0.01), ops, echo=print) == 4
    assert ops.log == []
    assert M.orchestrate(rent_args(pretok_dir(tmp_path / "b"), max_usd=0), ops, echo=print) == 2
    assert M.orchestrate(rent_args(pretok_dir(tmp_path / "c"), gpu="A10"), ops, echo=print) == 2


def test_rent_chains_calls_and_downloads_the_adapter(tmp_path: Path) -> None:
    ops = FakeOps([{"status": "stepdown", "max_seq": 12288}, {"status": "stop:call_timeout"},
                   {"status": "done", "spent_usd": 90.0}])
    clock = Clock(0.0)
    assert M.orchestrate(rent_args(pretok_dir(tmp_path)), ops, clock=clock.now, echo=print) == 0
    kinds = [k for k, _ in ops.log]
    assert kinds == ["upload", "prefetch", "train", "train", "train", "download"]
    gpu, timeout, job = ops.log[2][1]
    assert gpu == "H100" and timeout == M.MODAL_MAX_TIMEOUT_S
    assert job["max_usd"] == 150.0 and "approved" not in job and job["calib_steps"] == 50
    assert job["config"]["r"] == 32 and job["config"]["base_model"].startswith("/models/")
    assert job["base_revision"] == "c" * 40
    assert job["rate_per_s"] == pytest.approx(M.rate_per_s(M.GPUS["H100"]))


def test_rent_stops_on_the_calibration_decision_and_on_failures(tmp_path: Path) -> None:
    over = FakeOps([{"status": "over_cap", "calibration": {}}])
    assert M.orchestrate(rent_args(pretok_dir(tmp_path)), over, echo=print) == 5
    assert "download" not in [k for k, _ in over.log]
    broken = FakeOps([RuntimeError("preempted"), RuntimeError("again")])
    assert M.orchestrate(rent_args(pretok_dir(tmp_path / "b")), broken, echo=print) == 1


def test_rent_never_starts_a_call_past_max_hours(tmp_path: Path) -> None:
    clock = Clock(0.0)

    class Slow(FakeOps):
        def train_call(self, job: dict[str, Any], gpu: str, timeout_s: int) -> dict[str, Any]:
            clock.t += timeout_s
            return super().train_call(job, gpu, timeout_s)

    ops = Slow([{"status": "stop:call_timeout"}] * 5)
    assert M.orchestrate(rent_args(pretok_dir(tmp_path), max_hours=30), ops, clock=clock.now,
                         echo=print) == 1
    timeouts = [v[1] for k, v in ops.log if k == "train"]
    assert timeouts == [86400, 30 * 3600 - 86400]


def ledger_doc(*segments: dict[str, Any], calibrations: list[dict[str, Any]] | None = None,
               run: dict[str, Any] | None = None) -> dict[str, Any]:
    segs = [{"id": i, "gpu": "H100", "rate_per_s": M.rate_per_s(M.GPUS["H100"]), "started": 0.0,
             "last_seen": 0.0, "closed": True, "status": "done", **s}
            for i, s in enumerate(segments, 1)]
    return {"segments": segs, "calibrations": calibrations or [],
            "run": run if run is not None else {"base_model": M.DEFAULT_BASE,
                                                "base_revision": "c" * 40}}


def test_a_relaunch_while_a_call_still_runs_starts_no_container(tmp_path: Path) -> None:
    clock = Clock(10_000.0)
    live = ledger_doc({"closed": False, "status": "running", "started": 9_000.0,
                       "last_seen": 9_900.0, "loaded": True, "call_id": "fc-1"})
    ops = FakeOps([{"status": "done"}], ledger=live)
    lines: list[str] = []
    assert M.orchestrate(rent_args(pretok_dir(tmp_path)), ops, clock=clock.now,
                         echo=lines.append) == 6
    assert "train" not in [k for k, _ in ops.log]
    assert "modal app list" in lines[-1] and "same command" not in lines[-1]


def test_a_relaunch_reuses_the_revision_the_run_started_on(tmp_path: Path) -> None:
    done = ledger_doc({"started": 0.0, "last_seen": 3600.0, "status": "stop:call_timeout"})
    ops = FakeOps([{"status": "done"}], ledger=done, sha="d" * 40)  # main has moved since
    assert M.orchestrate(rent_args(pretok_dir(tmp_path)), ops, clock=Clock(1e6).now,
                         echo=print) == 0
    assert ("prefetch", (M.DEFAULT_BASE, "c" * 40)) in ops.log
    job = next(v[2] for k, v in ops.log if k == "train")
    assert job["base_revision"] == "c" * 40 and job["config"]["base_model"].endswith("c" * 40)
    pinned = FakeOps([], ledger=done)
    assert M.orchestrate(rent_args(pretok_dir(tmp_path / "b"), base_revision="e" * 40), pinned,
                         clock=Clock(1e6).now, echo=print) == 2
    assert "train" not in [k for k, _ in pinned.log]


def test_no_container_starts_once_the_cap_cannot_reach_a_step(tmp_path: Path) -> None:
    rate = M.rate_per_s(M.GPUS["H100"])
    spent_s = 149.0 / rate - M.STARTUP_ALLOWANCE_S  # $149 of a $150 cap
    doc = ledger_doc({"started": 0.0, "last_seen": spent_s, "status": "stop:max_usd"})
    ops = FakeOps([{"status": "done"}], ledger=doc)
    lines: list[str] = []
    assert M.orchestrate(rent_args(pretok_dir(tmp_path)), ops, clock=Clock(1e6).now,
                         echo=lines.append) == 1
    assert "train" not in [k for k, _ in ops.log]
    assert "higher --max-usd" in lines[-1]


def test_a_relaunch_after_over_cap_with_the_same_cap_stops_locally(tmp_path: Path) -> None:
    cal = {"gpu": "H100", "max_seq": 16384, "tokens_per_s": 1500.0, "step_s": 60.0,
           "load_s": 900, "decision": "over_cap", "projection": {"usd": 260.0, "seconds": 2e5}}
    doc = ledger_doc({"started": 0.0, "last_seen": 1500.0, "status": "over_cap"},
                     calibrations=[cal])
    lines: list[str] = []
    ops = FakeOps([{"status": "done"}], ledger=doc)
    assert M.orchestrate(rent_args(pretok_dir(tmp_path)), ops, clock=Clock(1e6).now,
                         echo=lines.append) == 5
    assert "train" not in [k for k, _ in ops.log] and "higher --max-usd" in lines[-1]
    raised = FakeOps([{"status": "done"}], ledger=doc)
    assert M.orchestrate(rent_args(pretok_dir(tmp_path / "b"), max_usd=300.0, max_hours=80),
                         raised, clock=Clock(1e6).now, echo=print) == 0


# ── modal_train.py: the Modal wiring, against a fake modal module ─────────


class _Fn:
    def __init__(self, f: Any, kw: dict[str, Any]) -> None:
        self.f, self.kw, self.options, self.calls = f, kw, [], []

    def with_options(self, **kw: Any) -> "_Fn":
        self.options.append(kw)
        return self

    def remote(self, *a: Any, **kw: Any) -> Any:
        self.calls.append((a, kw))
        return {"status": "done"}


class _Image:
    def __init__(self, log: list[Any]) -> None:
        self.log = log

    def __getattr__(self, name: str) -> Any:
        def chain(*a: Any, **kw: Any) -> "_Image":
            self.log.append((name, a, kw))
            return self
        return chain


def fake_modal() -> Any:
    m = types.ModuleType("modal")
    m.image_log = []  # type: ignore[attr-defined]

    class App:
        def __init__(self, name: str) -> None:
            self.name, self.functions, self.entrypoints = name, {}, {}

        def function(self, **kw: Any) -> Any:
            def deco(f: Any) -> _Fn:
                fn = _Fn(f, kw)
                self.functions[f.__name__] = fn
                return fn
            return deco

        def local_entrypoint(self) -> Any:
            def deco(f: Any) -> Any:
                self.entrypoints[f.__name__] = f
                return f
            return deco

    class Image:
        @staticmethod
        def from_registry(ref: str, **kw: Any) -> _Image:
            m.image_log.append(("from_registry", (ref,), kw))  # type: ignore[attr-defined]
            return _Image(m.image_log)  # type: ignore[attr-defined]

    class Volume:
        @staticmethod
        def from_name(name: str, **kw: Any) -> Any:
            return types.SimpleNamespace(name=name, kw=kw, commit=lambda: None)

    m.App, m.Image, m.Volume = App, Image, Volume  # type: ignore[attr-defined]
    return m


def test_modal_wiring_pins_the_image_and_limits_the_gpu_function() -> None:
    fm = fake_modal()
    W = _load_modal_train("distill_modal_train_wired", fm)
    app = W.app
    train = app.functions["train"].kw
    assert train["gpu"] == "H100" and train["timeout"] == M.MODAL_MAX_TIMEOUT_S
    assert train["scaledown_window"] == 2 and "min_containers" not in train
    assert train["single_use_containers"] is True  # a step-down gets a fresh CUDA context
    assert set(train["volumes"]) == {M.DATA_MOUNT, M.RUNS_MOUNT, M.MODELS_MOUNT}
    assert train["cpu"] == M.CPU_CORES and train["memory"] == M.MEMORY_MIB
    assert "gpu" not in app.functions["prefetch"].kw
    assert set(app.entrypoints) == {"rent", "fetch"}
    log = fm.image_log
    assert log[0] == ("from_registry", (M.UNSLOTH_IMAGE_REF,), {})  # by digest, not the tag
    assert M.UNSLOTH_IMAGE_REF == "unsloth/unsloth@" + M.UNSLOTH_IMAGE_DIGEST
    assert M.UNSLOTH_IMAGE.startswith("unsloth/unsloth:") and "latest" not in M.UNSLOTH_IMAGE
    assert ("entrypoint", ([],), {}) in log
    files = [a[1] for name, a, _ in log if name == "add_local_file"]
    assert files == ["/root/distill/train_qlora.py", "/root/distill/render.py"]
    ops = W.ModalOps()
    assert ops.train_call({"run_id": "x"}, "L40S", 3600) == {"status": "done"}
    assert app.functions["train"].options == [{"gpu": "L40S", "timeout": 3600}]


def test_modal_entrypoint_is_print_only_by_default(tmp_path: Path) -> None:
    W = _load_modal_train("distill_modal_train_wired2", fake_modal())
    data = pretok_dir(tmp_path)
    W.rent(data=str(data), max_usd=150.0)
    assert W.app.functions["train"].calls == []
    maxd = pretok_dir(tmp_path / "m", sources=("max-golden",), teacher_max=None)
    with pytest.raises(SystemExit) as e:
        W.rent(data=str(maxd), max_usd=150.0, launch=True)  # Max-derived: never rents
    assert e.value.code == 3 and W.app.functions["train"].calls == []


# ── the rental packet document ─────────────────────────────────────────────


def test_rental_packet_matches_the_code() -> None:
    doc = (_DIR.parents[1] / "docs" / "sprints" / "DISTILL-RENTAL-PACKET.md").read_text()
    assert M.markdown_table(M.cost_table(M.scenario_tokens())) in doc
    for must in ("not launched", "operator", "OPEN-QUESTIONS #6", M.UNSLOTH_IMAGE,
                 M.UNSLOTH_IMAGE_DIGEST, "modal run --detach scripts/distill/modal_train.py::rent",
                 "https://modal.com/pricing"):
        assert must in doc


def test_the_packets_launch_line_carries_no_approval_and_the_operators_cap() -> None:
    """A "RENT YES $<cap>" answers the spend only: the line an agent would copy
    must not also answer OPEN-QUESTIONS #6, nor fix a cap of its own."""
    doc = (_DIR.parents[1] / "docs" / "sprints" / "DISTILL-RENTAL-PACKET.md").read_text()
    lines = [x for x in doc.splitlines() if x.startswith("modal run --detach")]
    assert len(lines) == 1
    assert T.APPROVAL_FLAG not in lines[0] and "--max-usd <cap>" in lines[0]
    for line in doc.splitlines():
        if T.APPROVAL_FLAG in line:  # only ever described, never part of a command
            assert not line.lstrip().startswith(("modal ", "python"))


# ── Q6b: Max-derived training is explicit, recorded, tagged and never rented ──


def test_q6b_flag_alone_or_dataset_approval_alone_is_refused(
        tmp_path: Path, cli: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    """Max-derived rows train only with BOTH the operator's flag on the command
    line and the approval the dataset recorded when build.py ran with
    --allow-max-derived --max-derived-approval."""
    data = write_rows(tmp_path / "train.jsonl", rows_n(8))
    built: list[Any] = []

    def factory(c: Any) -> Any:
        built.append(c)
        return FakeBackend()

    # the flag, but the dataset records no allow (no report, then a report without one)
    assert T.main(base_args(data, cli, T.APPROVAL_FLAG), backend_factory=factory) == 3
    assert "records no Max-derived approval" in capsys.readouterr().err
    (tmp_path / T.BUILD_REPORT).write_text(json.dumps({
        "settings": {"teacher_model": "qwen3.8-max"},
        "data_policy": {"max_derived_allow": {"allowed": False, "approval": None}}}))
    assert T.main(base_args(data, cli, T.APPROVAL_FLAG), backend_factory=factory) == 3
    # a blank reference is no approval
    approve_dataset(data, ref="  ")
    assert T.main(base_args(data, cli, T.APPROVAL_FLAG), backend_factory=factory) == 3
    # the dataset's approval, but not the flag
    approve_dataset(data)
    assert T.main(base_args(data, cli), backend_factory=factory) == 3
    assert "OPEN-QUESTIONS #6" in capsys.readouterr().err
    assert built == [] and not cli["out"].exists()


def test_q6b_run_on_approved_data_is_tagged_and_records_both_approvals(
        tmp_path: Path, cli: dict[str, Any]) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(8)))
    rc = T.main(base_args(data, cli, T.APPROVAL_FLAG, "--grad-accum", "2", "--save-every", "3"),
                backend_factory=lambda c: FakeBackend())
    assert rc == 0
    # the adapter's directory name carries the tag; no untagged adapter is written
    assert (cli["out"] / "adapter-max-derived" / "adapter.json").is_file()
    assert not (cli["out"] / "adapter").exists()
    for m in (json.loads((cli["out"] / "manifest.json").read_text()),
              json.loads((cli["logs"] / "manifest.json").read_text())):
        assert m["max_derived"] is True and m["tags"] == ["max-derived"] and m["private"] is True
        assert m["approvals"]["operator_flag"] == {"flag": T.APPROVAL_FLAG, "given": True}
        assert m["approvals"]["dataset"]["approval"] == APPROVAL_REF
        assert m["approvals"]["dataset"]["recorded_in"] == str(tmp_path / T.BUILD_REPORT)
        assert m["operator_approval_max_derived"] is True
        assert m["adapter"].endswith("adapter-max-derived")
        assert m["data"]["max_derived_rows"] == 8


def test_q6b_clean_run_keeps_the_plain_adapter_name(tmp_path: Path, cli: dict[str, Any]) -> None:
    rows = [make_row(i, 4, "qwen38-27b-gym") for i in range(8)]
    for r in rows:
        r["max_derived"] = False
    data = write_rows(tmp_path / "train.jsonl", rows)
    (tmp_path / T.BUILD_REPORT).write_text(
        json.dumps({"settings": {"teacher_model": "qwen3.8-27b"}}))
    rc = T.main(base_args(data, cli, "--grad-accum", "2"), backend_factory=lambda c: FakeBackend())
    assert rc == 0 and (cli["out"] / "adapter" / "adapter.json").is_file()
    m = json.loads((cli["out"] / "manifest.json").read_text())
    assert m["max_derived"] is False and m["tags"] == [] and m["private"] is False
    assert m["approvals"]["dataset"]["approval"] is None


def test_q6b_dry_run_names_the_tagged_adapter_and_records_the_tag(
        tmp_path: Path, cli: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(4)))
    assert T.main(base_args(data, cli, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "adapter-max-derived" in out and "a real run would be REFUSED" in out
    m = json.loads((cli["logs"] / "manifest.json").read_text())
    assert m["max_derived"] is True and m["tags"] == ["max-derived"]
    assert m["approvals"]["dataset"]["approval"] == APPROVAL_REF
    assert m["approvals"]["operator_flag"]["given"] is False


def test_q6b_pretokenized_data_carries_the_dataset_approval(tmp_path: Path, cli: dict[str, Any]
                                                            ) -> None:
    rows = rows_n(6)
    approve_dataset(write_rows(tmp_path / "train.jsonl", rows))
    rec = T.dataset_record(tmp_path / "train.jsonl")
    assert rec["approval"] == APPROVAL_REF
    policy = {"report": rec["report"], "max_derived_allow": {"allowed": True,
                                                             "approval": rec["approval"]}}
    meta = T.write_pretokenized(rows, ENC, tmp_path / "pt", ladder=(16384,), teacher_max=True,
                                dataset_policy=policy, source=str(tmp_path / "train.jsonl"))
    assert meta["max_derived"] is True and meta["tags"] == ["max-derived"] and meta["private"]
    assert meta["dataset_policy"] == policy and meta["source"].endswith("train.jsonl")
    assert T.dataset_record(pretokenized=tmp_path / "pt")["approval"] == APPROVAL_REF
    args = ["--pretokenized", str(tmp_path / "pt"), "--out", str(cli["out"]),
            "--log-dir", str(cli["logs"]), "--grad-accum", "2"]
    assert T.main(args, backend_factory=lambda c: FakeBackend()) == 3  # no flag
    assert T.main([*args, T.APPROVAL_FLAG], backend_factory=lambda c: FakeBackend()) == 0
    assert (cli["out"] / "adapter-max-derived").is_dir()
    # without the recorded approval the flag is not enough
    meta2 = T.write_pretokenized(rows, ENC, tmp_path / "pt2", ladder=(16384,), teacher_max=True)
    assert meta2["dataset_policy"] == {}
    assert T.dataset_record(pretokenized=tmp_path / "pt2")["approval"] is None
    shutil.rmtree(cli["out"])
    args2 = ["--pretokenized", str(tmp_path / "pt2"), "--out", str(cli["out"]),
             "--log-dir", str(cli["logs"]), T.APPROVAL_FLAG]
    assert T.main(args2, backend_factory=lambda c: FakeBackend()) == 3


def test_q6b_rental_refuses_max_data_at_every_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    # plan: the old approval flag is gone, not inert
    clean = pretok_dir(tmp_path / "clean")
    with pytest.raises(SystemExit) as e:
        M.main(["plan", "--data", str(clean), "--max-usd", "150", T.APPROVAL_FLAG])
    assert e.value.code == 2
    assert M.main(["plan", "--data", str(clean), "--max-usd", "150"]) == 0
    # a pre-tokenized dir whose meta says clean but whose name carries the tag
    tagged = tmp_path / "pt-max-derived"
    shutil.copytree(clean, tagged)
    lines: list[str] = []
    assert M.orchestrate(rent_args(tagged, launch=False), None, echo=lines.append) == 3
    assert "Q6b" in lines[-1]
    # meta without a Max-derived count: refused (fail closed)
    nocount = tmp_path / "nocount"
    shutil.copytree(clean, nocount)
    meta = json.loads((nocount / "meta.json").read_text())
    meta.pop("max_derived_rows")
    (nocount / "meta.json").write_text(json.dumps(meta))
    assert M.orchestrate(rent_args(nocount, launch=False), None, echo=print) == 3
    # pretokenize, the rental's first step, refuses Max-derived rows (approved dataset or not)
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(4)))
    monkeypatch.setattr(T, "hf_encoder", lambda d: ENC)
    monkeypatch.setattr(T, "template_refitter", lambda d: None)
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    rc = M.main(["pretokenize", "--data", str(data), "--tokenizer-dir", str(tok),
                 "--out", str(tmp_path / "pt-out")])
    assert rc == 3 and not (tmp_path / "pt-out").exists()
    assert "never rents" in capsys.readouterr().err
    # clean rows pre-tokenize, and the meta records the source and its (absent) allow
    clean_rows = [dict(make_row(i, 4, "qwen38-27b-gym"), max_derived=False) for i in range(4)]
    (tmp_path / "c").mkdir()
    cdata = write_rows(tmp_path / "c" / "train.jsonl", clean_rows)
    (tmp_path / "c" / T.BUILD_REPORT).write_text(
        json.dumps({"settings": {"teacher_model": "qwen3.8-27b"}}))
    assert M.main(["pretokenize", "--data", str(cdata), "--tokenizer-dir", str(tok),
                   "--out", str(tmp_path / "pt-clean")]) == 0
    meta = json.loads((tmp_path / "pt-clean" / "meta.json").read_text())
    assert meta["max_derived"] is False and meta["max_derived_rows"] == 0
    assert meta["dataset_policy"]["max_derived_allow"]["allowed"] is False
    assert meta["source"] == str(cdata)


def test_q6b_modal_upload_refuses_max_data_before_touching_the_volume(tmp_path: Path) -> None:
    W = _load_modal_train("distill_modal_train_wired3", fake_modal())
    maxd = pretok_dir(tmp_path / "m", sources=("max-golden",), teacher_max=None)
    meta = json.loads((maxd / "meta.json").read_text())
    with pytest.raises(RuntimeError, match="never rents"):
        W.ModalOps().upload_data(maxd, "m", meta)
    # a meta edited to say clean does not get past the rows' own count either
    with pytest.raises(RuntimeError, match="Q6b"):
        W.ModalOps().upload_data(maxd, "m", {**meta, "max_derived_rows": 0, "max_derived": False})


# ── Q6b reviewer findings: rental upload, run names, checkpoints, records ──

AP = T._load_policy()


class FakeDataVolume:
    """The data Volume's upload side: records what would leave our machines."""

    def __init__(self) -> None:
        self.put: list[str] = []

    def read_file(self, path: str) -> Any:
        raise FileNotFoundError(path)  # nothing there yet

    def batch_upload(self, force: bool = False) -> Any:
        vol = self

        class Batch:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

            def put_file(self, local: str, remote: str) -> None:
                assert Path(local).is_file()
                vol.put.append(remote)

            def put_directory(self, local: str, remote: str) -> None:
                raise AssertionError("a whole directory is never uploaded")

        return Batch()


def test_q6b_rental_refuses_max_rows_beside_a_clean_pretok_set_and_uploads_only_its_files(
        tmp_path: Path) -> None:
    # pretokenize --out <the dataset directory>: the pre-tokenized meta and the
    # dataset's reports all say max_derived false, but the directory holds the
    # dataset's Max eval rows (its holds_max_rows says so)
    ds = tmp_path / "phase3"
    ds.mkdir()
    clean_rows = [dict(make_row(i, 4, "qwen38-27b-gym"), max_derived=False) for i in range(6)]
    write_rows(ds / "train.jsonl", clean_rows)
    write_rows(ds / "eval-tools.jsonl", [make_row(i, 4, "max-golden") for i in range(3)])
    (ds / "report.json").write_text(json.dumps({
        "schema": AP.REPORT_SCHEMA, **AP.policy_fields(False), "holds_max_rows": True,
        "settings": {"teacher_model": "qwen3.8-27b"},
        "splits": {"train": {"max_derived": 0}, "val": {"max_derived": 0},
                   "eval-tools": {"max_derived": 3}}}))
    (ds / "dataset-manifest.json").write_text(json.dumps({
        "schema": AP.DATASET_MANIFEST_SCHEMA, **AP.policy_fields(False), "holds_max_rows": True}))
    meta = T.write_pretokenized(clean_rows, ENC, ds, ladder=(16384, 12288, 8192),
                                teacher_max=False, source=str(ds / "train.jsonl"))
    assert meta["max_derived_rows"] == 0 and not AP.is_max_derived(ds)  # what passed before
    refusal = M.rental_refusal(ds, meta)
    assert refusal and "holds Max-derived rows" in refusal and "eval-tools.jsonl" in refusal
    lines: list[str] = []
    assert M.orchestrate(rent_args(ds, launch=False), None, echo=lines.append) == 3
    assert "may not leave our machines" in lines[-1]
    # the same pre-tokenized files in a directory of their own rent
    pt = tmp_path / "pt-clean"
    pt.mkdir()
    for f in M.PRETOK_FILES:
        shutil.copy(ds / f, pt / f)
    assert M.rental_refusal(pt, meta) is None
    # a stray JSONL copied in, or a missing pre-tokenized file, is refused
    shutil.copy(ds / "eval-tools.jsonl", pt / "notes.jsonl")
    assert "notes.jsonl" in (M.rental_refusal(pt, meta) or "")
    (pt / "notes.jsonl").unlink()
    (pt / "tokens.bin").rename(pt / "tokens.bak")
    assert "no tokens.bin" in (M.rental_refusal(pt, meta) or "")
    (pt / "tokens.bak").rename(pt / "tokens.bin")
    # the upload sends the three pre-tokenized files, nothing else in the directory
    (pt / "scratch.txt").write_text("not data")
    W = _load_modal_train("distill_modal_train_wired4", fake_modal())
    vol = FakeDataVolume()
    W.data_vol = vol
    assert "uploaded" in W.ModalOps().upload_data(pt, "pt-clean", meta)
    assert sorted(vol.put) == sorted(f"/pt-clean/{f}" for f in M.PRETOK_FILES)
    with pytest.raises(RuntimeError, match="may not leave"):
        W.ModalOps().upload_data(ds, "phase3", meta)
    assert len(vol.put) == 3  # the dataset directory never reached the Volume


def test_rented_adapter_records_clean_provenance_and_publishes(tmp_path: Path) -> None:
    assert M.CLEAN_POLICY_FIELDS == AP.policy_fields(False)  # the inlined copy
    env = segment_env(tmp_path)
    res = call(env)
    assert res["status"] == "done"
    man = json.loads((env["run_dir"] / "manifest.json").read_text())
    assert {k: man[k] for k in M.CLEAN_POLICY_FIELDS} == M.CLEAN_POLICY_FIELDS
    assert man["data"]["max_derived_rows"] == 0 and man["data"]["sha256"] is None
    # what `fetch` brings home: the adapter beside the run's manifest
    dest = tmp_path / "rental" / "r1"
    shutil.copytree(env["run_dir"] / "adapter", dest / "adapter")
    shutil.copy(env["run_dir"] / "manifest.json", dest / "manifest.json")
    prov, reasons = AP.check_publishable(dest / "adapter")
    assert prov.max_derived is False and reasons == []


def test_q6b_a_max_run_needs_a_tagged_out_and_every_checkpoint_says_so(
        tmp_path: Path, cli: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    data = approve_dataset(write_rows(tmp_path / "train.jsonl", rows_n(8)))
    args = ["--data", str(data), "--tokenizer-dir", str(cli["tok"]), "--log-dir",
            str(cli["logs"]), "--grad-accum", "2", "--save-every", "2", T.APPROVAL_FLAG]
    plain = tmp_path / "runs" / "9b-r32"
    assert T.main([*args, "--out", str(plain)], backend_factory=lambda c: FakeBackend()) == 2
    assert "9b-r32-max-derived" in capsys.readouterr().err and not plain.exists()
    assert T.main([*args, "--out", str(plain), "--dry-run"]) == 0  # the dry run says so
    assert "would refuse this" in capsys.readouterr().out
    run = tmp_path / "runs" / "9b-r32-max-derived"
    assert T.main([*args, "--out", str(run)], backend_factory=lambda c: FakeBackend()) == 0
    ckpts = sorted(run.glob(T.CKPT_PREFIX + "*"))
    assert ckpts
    for ck in ckpts:
        m = json.loads((ck / T.CKPT_MANIFEST).read_text())
        assert m["schema"] == T.MANIFEST_SCHEMA and m["checkpoint"] == ck.name
        assert m["max_derived"] is True and m["tags"] == ["max-derived"] and m["private"]
        assert m["data"]["max_derived_rows"] == 8 and m["run_id"] == run.name
        assert m["approvals"]["dataset"]["approval"] == APPROVAL_REF
    # a checkpoint copied into a clean run's directory still carries its tag
    clean = tmp_path / "runs" / "9b-gym"
    clean.mkdir()
    (clean / "manifest.json").write_text(json.dumps({
        "schema": T.MANIFEST_SCHEMA, **AP.policy_fields(False), "data": {"max_derived_rows": 0}}))
    copied = clean / ckpts[-1].name
    shutil.copytree(ckpts[-1], copied)
    for level in (copied, copied / "adapter"):
        prov, reasons = AP.check_publishable(level)
        assert prov.max_derived is True and reasons, level
    # a clean run's checkpoints say so too, and a copy of one publishes on its own
    rows = [dict(make_row(i, 4, "qwen38-27b-gym"), max_derived=False) for i in range(8)]
    (tmp_path / "c").mkdir()
    cdata = write_rows(tmp_path / "c" / "train.jsonl", rows)
    (tmp_path / "c" / T.BUILD_REPORT).write_text(json.dumps({
        "schema": AP.REPORT_SCHEMA, **AP.policy_fields(False),
        "settings": {"teacher_model": "qwen3.8-27b"},
        "splits": {"train": {"max_derived": 0}, "val": {"max_derived": 0}}}))
    crun = tmp_path / "runs" / "9b-clean"
    assert T.main(["--data", str(cdata), "--tokenizer-dir", str(cli["tok"]), "--out", str(crun),
                   "--log-dir", str(tmp_path / "clogs"), "--grad-accum", "2", "--save-every",
                   "2"], backend_factory=lambda c: FakeBackend()) == 0
    cck = sorted(crun.glob(T.CKPT_PREFIX + "*"))[-1]
    assert json.loads((cck / T.CKPT_MANIFEST).read_text())["max_derived"] is False
    alone = tmp_path / "elsewhere" / cck.name
    shutil.copytree(cck, alone)
    assert AP.check_publishable(alone / "adapter")[1] == []


def test_q6b_the_dataset_manifest_carries_the_approval_when_the_report_went_elsewhere(
        tmp_path: Path, cli: dict[str, Any]) -> None:
    ds = tmp_path / "phase3-max-derived"
    ds.mkdir()
    data = write_rows(ds / "train.jsonl", rows_n(8))
    (ds / T.DATASET_MANIFEST).write_text(json.dumps({
        "schema": AP.DATASET_MANIFEST_SCHEMA, **AP.policy_fields(True), "holds_max_rows": True,
        "teacher_model": "qwen3.8-max", "report": str(tmp_path / "rep-max-derived.json"),
        "max_derived_allow": {"allowed": True, "approval": APPROVAL_REF},
        "files": {"train.jsonl": {"rows": 8, "sha256": T.sha256_files([data])}}}))
    rec = T.dataset_record(data)
    assert rec["approval"] == APPROVAL_REF and rec["source"] == str(ds / T.DATASET_MANIFEST)
    assert rec["report"] == str(tmp_path / "rep-max-derived.json")
    assert T.builder_teacher_is_max(data) is True
    args = base_args(data, cli, T.APPROVAL_FLAG, "--grad-accum", "2")
    assert T.main(args, backend_factory=lambda c: FakeBackend()) == 0
    m = json.loads((cli["out"] / "manifest.json").read_text())
    assert m["approvals"]["dataset"] == {"approval": APPROVAL_REF,
                                         "recorded_in": str(ds / T.DATASET_MANIFEST)}
    # a train.jsonl the manifest does not list (other content) carries no approval
    shutil.rmtree(cli["out"])
    write_rows(data, rows_n(7))
    assert T.dataset_record(data)["approval"] is None
    assert T.main(args, backend_factory=lambda c: FakeBackend()) == 3
    # report.json beside the data is still read first
    approve_dataset(data, ref="from the report")
    assert T.dataset_record(data)["approval"] == "from the report"
