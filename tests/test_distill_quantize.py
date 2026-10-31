"""scripts/distill/quantize.py: commands, train-only calibration, manifest, dry-run default.

Nothing here runs llama.cpp, torch or PEFT: subprocess is a fake that writes
the files each tool would, the merge's heavy imports are fakes, and GGUF
metadata is read back from a stub.
"""

import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "distill_quantize", REPO / "scripts" / "distill" / "quantize.py")
Q = importlib.util.module_from_spec(_spec)
sys.modules["distill_quantize"] = Q
_spec.loader.exec_module(Q)

HEAD = "9d57ce456" + "0" * 31
ARCH = "Qwen3_5ForConditionalGeneration"
TEMPLATE = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
ROW_TOKENS = 3000
N_TRAIN = 42
N_TASKS = 7


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_safetensors(path, keys):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]}
              for i, k in enumerate(keys)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 4 * len(keys))


def train_row(i, **over):
    row = {"id": f"p3-train{i:03d}", "split": "train", "task_id": f"task{i % N_TASKS}",
           "group_id": f"group{i % N_TASKS}", "category": "coding",
           "text": f"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nTRAINTEXT{i:03d}"
                   f"<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n",
           "n_tokens": ROW_TOKENS}
    row.update(over)
    return row


@pytest.fixture
def phase3(tmp_path):
    """Phase 3 + eval-set outputs: train, val, eval-tools, and two synthetic sets."""
    data, evalsets = tmp_path / "phase3", tmp_path / "phase3" / "evalsets"
    jsonl(data / "train.jsonl", [train_row(i) for i in range(N_TRAIN)])
    jsonl(data / "val.jsonl", [
        {"id": f"p3-val{i}", "split": "val", "task_id": f"vtask{i}", "group_id": f"vgroup{i}",
         "text": f"VALTEXT{i}", "n_tokens": ROW_TOKENS} for i in range(3)])
    jsonl(data / "val-cases.jsonl", [
        {"id": f"p3-val{i}", "set": "val", "messages": [{"role": "user", "content": "VALTEXT"}],
         "meta": {"task_id": f"vtask{i}", "group_id": f"vgroup{i}"}} for i in range(3)])
    jsonl(data / "eval-tools.jsonl", [
        {"id": f"p3f-eval{i}", "set": "eval-tools",
         "messages": [{"role": "user", "content": f"EVALTEXT{i}"}],
         "meta": {"task_id": f"etask{i}", "group_id": f"egroup{i}",
                  "trace_id": f"golden_traces_1_2.jsonl:{i}"}} for i in range(4)])
    jsonl(evalsets / "sanity.jsonl", [
        {"id": "sanity-knowledge-000", "set": "sanity",
         "messages": [{"role": "user", "content": "SANITYTEXT"}]}])
    jsonl(evalsets / "eval-longctx.jsonl", [
        {"id": "longctx-32k-recall-d10-0", "set": "eval-longctx",
         "messages": [{"role": "user", "content": "LONGCTXTEXT"}]}])
    return SimpleNamespace(data=data, evalsets=evalsets)


EVAL_IDS = {f"p3-val{i}" for i in range(3)} | {f"p3f-eval{i}" for i in range(4)} | {
    "sanity-knowledge-000", "longctx-32k-recall-d10-0"}
EVAL_MARKERS = ("VALTEXT", "EVALTEXT", "SANITYTEXT", "LONGCTXTEXT")


def calibrate(phase3, *, injection=None, calib_tokens=20_000, ppl_tokens=6_000, seed=1):
    files = Q.eval_files(phase3.data, phase3.evalsets, [])
    deny = Q.EvalDenylist.load(files)
    return Q.build_calibration(phase3.data / "train.jsonl", deny, seed=seed,
                               calib_tokens=calib_tokens, ppl_tokens=ppl_tokens,
                               min_calib_tokens=4096, min_ppl_tokens=4096, injection=injection)


# ── commands ────────────────────────────────────────────────────────────────


def llama(tmp_path):
    return Q.LlamaCpp(tmp_path / "llama.cpp", "9d57ce456", tmp_path / "llama.cpp" / "build" / "bin")


def test_serve_command_is_the_runbook_line_with_a_valued_fa():
    cmd = Q.serve_command("/m/tuned-Q4_K_M.gguf", server="/b/llama-server",
                          alias="tuned-Q4_K_M", port=8081)
    assert cmd == ["/b/llama-server", "-m", "/m/tuned-Q4_K_M.gguf", "-c", "262144", "-np", "1",
                   "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "-ngl", "99",
                   "--host", "127.0.0.1", "--port", "8081", "--alias", "tuned-Q4_K_M"]
    # At the pinned build -fa takes a value: a bare one would swallow -ctk.
    assert cmd[cmd.index("-fa") + 1] == "on"


def test_serve_subcommand_prints_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Q.subprocess, "run", lambda *a, **k: pytest.fail("serve ran something"))
    gguf = tmp_path / "run-Q5_K_M.gguf"
    assert Q.main(["serve", "--gguf", str(gguf), "--llama-cpp-dir", str(tmp_path / "lc")]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(str(tmp_path / "lc" / "build" / "bin" / "llama-server"))
    assert "-c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja" in line
    assert line.endswith("--alias run-Q5_K_M")


def test_convert_imatrix_quantize_perplexity_commands(tmp_path):
    lc = llama(tmp_path)
    conv = Q.convert_cmd("py", lc, Path("/m/merged"), Path("/o/x-BF16.partial.gguf"),
                         model_name="x")
    assert conv == ["py", str(lc.converter), "/m/merged", "--outtype", "bf16", "--outfile",
                    "/o/x-BF16.partial.gguf", "--no-mtp", "--model-name", "x"]
    im = Q.imatrix_cmd(lc, Path("/o/x-BF16.gguf"), Path("/o/calib/calibration.txt"),
                       Path("/o/x.imatrix.partial.gguf"), ctx=4096, chunks=128)
    assert im[0] == str(lc.tool("llama-imatrix"))
    assert im[im.index("-f") + 1] == "/o/calib/calibration.txt"
    assert im[im.index("-c") + 1] == im[im.index("-b") + 1] == "4096"
    assert im[im.index("--chunks") + 1] == "128"
    assert "--parse-special" in im and im[im.index("--output-format") + 1] == "gguf"
    qz = Q.quantize_cmd(lc, Path("/o/x-BF16.gguf"), Path("/o/x.imatrix.gguf"),
                        Path("/o/x-Q4_K_M.partial.gguf"), "Q4_K_M")
    assert qz == [str(lc.tool("llama-quantize")), "--imatrix", "/o/x.imatrix.gguf",
                  "/o/x-BF16.gguf", "/o/x-Q4_K_M.partial.gguf", "Q4_K_M"]
    pp = Q.perplexity_cmd(lc, Path("/o/x-Q4_K_M.gguf"), Path("/o/calib/ppl.txt"), ctx=4096,
                          chunks=16)
    assert pp[pp.index("-f") + 1] == "/o/calib/ppl.txt"
    assert pp[pp.index("--chunks") + 1] == "16"


def test_fp16_converts_to_f16():
    s = Q.Settings(out=Path("/o"), data=Path("/d"), evalsets=None, extra_eval=[],
                   injection_manifest=None, seed=1, calib_tokens=8192, imatrix_ctx=4096,
                   ppl_ctx=4096, ppl_chunks=4, dtype="fp16", label="x")
    assert s.outtype == "f16" and s.gguf(s.ref_name).name == "x-F16.gguf"
    assert s.imatrix_chunks == 2


def test_verified_arch_support_covers_the_student():
    assert ARCH in Q.VERIFIED_SUPPORT["9d57ce456"]
    assert Q.GGUF_ARCH[ARCH] == "qwen35"
    assert "verified" in Q.arch_note(ARCH, "9d57ce456")
    assert "not verified" in Q.arch_note(ARCH, "abcdef123")


# ── calibration: train only ────────────────────────────────────────────────


def test_calibration_is_train_rows_only(phase3):
    cal = calibrate(phase3)
    used = set(cal.calib_ids) | set(cal.ppl_ids)
    assert used and all(i.startswith("p3-train") for i in used)
    assert not used & EVAL_IDS
    assert not set(cal.calib_ids) & set(cal.ppl_ids)
    for marker in EVAL_MARKERS:
        assert marker not in cal.text and marker not in cal.ppl_text
    assert "TRAINTEXT" in cal.text and "<|im_start|>" in cal.text
    info = cal.info
    assert info["source"] == "train split only"
    assert info["eval_check"]["train_rows_sharing_a_key"] == 0
    assert {f["name"] for f in info["eval_check"]["files"]} == {
        "val.jsonl", "val-cases.jsonl", "eval-tools.jsonl", "sanity.jsonl", "eval-longctx.jsonl"}
    assert info["eval_check"]["keys"]["id"] == len(EVAL_IDS)
    assert info["calibration"]["sha256"] == sha(cal.text)
    assert info["calibration"]["tokens_est"] >= 20_000


@pytest.mark.parametrize("over, why", [
    ({"task_id": "etask1"}, "task_id etask1 is in eval-tools.jsonl"),
    ({"group_id": "vgroup0"}, "group_id vgroup0 is in val.jsonl"),
    ({"id": "p3f-eval2"}, "id p3f-eval2 is in eval-tools.jsonl"),
    ({"trace_id": "golden_traces_1_2.jsonl:3"}, "trace_id"),
])
def test_calibration_refuses_a_train_row_that_shares_an_eval_key(phase3, over, why):
    rows = [train_row(i) for i in range(N_TRAIN)] + [train_row(99, **over)]
    jsonl(phase3.data / "train.jsonl", rows)
    with pytest.raises(Q.CalibrationError, match="share a key") as e:
        calibrate(phase3)
    assert why in str(e.value)


def test_calibration_refuses_rows_that_are_not_split_train(phase3):
    rows = [train_row(i) for i in range(N_TRAIN)] + [train_row(77, split="val")]
    jsonl(phase3.data / "train.jsonl", rows)
    with pytest.raises(Q.CalibrationError, match="not the train split"):
        calibrate(phase3)


def test_calibration_needs_the_eval_files_to_prove_train_only(phase3):
    (phase3.data / "eval-tools.jsonl").unlink()
    with pytest.raises(Q.CalibrationError, match="eval-tools.jsonl not found"):
        Q.eval_files(phase3.data, phase3.evalsets, [])


def test_calibration_takes_one_row_per_task_before_repeating(phase3):
    cal = calibrate(phase3, calib_tokens=N_TASKS * ROW_TOKENS, ppl_tokens=2 * ROW_TOKENS)
    assert len(cal.calib_ids) == N_TASKS
    tasks = {f"task{int(i[-3:]) % N_TASKS}" for i in cal.calib_ids}
    assert len(tasks) == N_TASKS


def test_calibration_is_seeded(phase3):
    a, b, c = calibrate(phase3, seed=1), calibrate(phase3, seed=1), calibrate(phase3, seed=2)
    assert a.text == b.text and a.calib_ids == b.calib_ids
    assert a.calib_ids != c.calib_ids


def test_calibration_leaves_out_rows_that_overlap_eval_injection(phase3):
    rows = [train_row(i) for i in range(N_TRAIN)]
    rows[0]["text"] += "attacker-site.example payload"
    rows[1]["text"] += "attacker-site.example payload"
    jsonl(phase3.data / "train.jsonl", rows)

    class Manifest:
        def record_overlaps(self, record):
            return {"inj-tool-direct-00-x": ["marker"]} if "attacker-site" in record["text"] else {}

    # Budgets that need every usable row, so both overlapping rows are met.
    cal = calibrate(phase3, injection=Manifest(), calib_tokens=(N_TRAIN - 4) * ROW_TOKENS,
                    ppl_tokens=2 * ROW_TOKENS + 1)
    assert "attacker-site" not in cal.text + cal.ppl_text
    assert cal.info["eval_check"]["dropped_injection_overlap"] == 2


def test_calibration_too_small_is_refused(phase3):
    jsonl(phase3.data / "train.jsonl", [train_row(0, n_tokens=100)])
    with pytest.raises(Q.CalibrationError, match="calibration tokens"):
        calibrate(phase3)


def test_rows_without_text_are_skipped_not_used(phase3):
    rows = [train_row(i) for i in range(N_TRAIN)] + [train_row(90, text=None)]
    jsonl(phase3.data / "train.jsonl", rows)
    cal = calibrate(phase3)
    assert cal.info["train"]["rows_without_text"] == 1
    assert "p3-train090" not in cal.calib_ids + cal.ppl_ids


# ── merge (heavy deps faked) ───────────────────────────────────────────────


class FakeW:
    def __init__(self, v):
        self.v = v

    def detach(self):
        return self

    def abs(self):
        return self

    def max(self):
        return self

    def item(self):
        return abs(self.v)


class FakeLora:
    def __init__(self, b=0.5):
        self.lora_A = {"default": object()}
        self.lora_B = {"default": SimpleNamespace(weight=FakeW(b))}


MODULES = [f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj" for i in range(3)]


def fake_deps(calls, *, attached=3, b=0.5, zero_b=0, missing=(),
              unexpected=("mtp.fc.weight",), version="4.57.1"):
    class Auto:
        @classmethod
        def from_pretrained(cls, path, **kw):
            calls["load"] = (cls.__name__, path, kw)
            return object(), {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    class AutoModelForCausalLM(Auto):
        pass

    class AutoModelForImageTextToText(Auto):
        pass

    class Merged:
        def save_pretrained(self, path, **kw):
            calls["save"] = kw
            p = Path(path)
            p.mkdir(parents=True)
            (p / "config.json").write_text(json.dumps({"architectures": [ARCH]}))
            (p / "model.safetensors").write_bytes(b"merged-weights")
            (p / "generation_config.json").write_text('{"from": "save_pretrained"}')

    class PeftModel:
        def __init__(self):
            self.mods = [("", object())] + [(m, FakeLora(0.0 if i < zero_b else b))
                                            for i, m in enumerate(MODULES[:attached])]

        @classmethod
        def from_pretrained(cls, model, path, **kw):
            calls["peft"] = (path, kw)
            return cls()

        def named_modules(self):
            return iter(self.mods)

        def merge_and_unload(self, safe_merge=False):
            calls["safe_merge"] = safe_merge
            return Merged()

    torch = SimpleNamespace(bfloat16="torch.bfloat16", float16="torch.float16")
    transformers = SimpleNamespace(__version__=version, AutoModelForCausalLM=AutoModelForCausalLM,
                                   AutoModelForImageTextToText=AutoModelForImageTextToText)
    return lambda: (torch, transformers, SimpleNamespace(PeftModel=PeftModel))


@pytest.fixture
def model_dirs(tmp_path):
    base, tdir, adapter = tmp_path / "base", tmp_path / "tok", tmp_path / "run7" / "adapter"
    for d in (base, tdir, adapter):
        d.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps({"architectures": [ARCH]}))
    (base / "model.safetensors").write_bytes(b"base-weights")
    (base / "tokenizer.json").write_text('{"vocab": 1}')
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD", "x": 1}))
    (base / "generation_config.json").write_text('{"temperature": 1.0}')
    (tdir / "tokenizer.json").write_text('{"vocab": 1}')
    (tdir / "chat_template.jinja").write_text(TEMPLATE)
    (adapter / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 32, "lora_alpha": 32, "target_modules": ["q_proj"],
         "base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    write_safetensors(adapter / "adapter_model.safetensors",
                      [f"{m}.lora_{s}.weight" for m in MODULES for s in "AB"])
    return SimpleNamespace(base=base, tdir=tdir, adapter=adapter)


def test_safetensors_keys_and_adapter_modules(model_dirs):
    assert Q.adapter_lora_modules(model_dirs.adapter) == sorted(MODULES)


def test_merge_writes_a_checkpoint_with_the_trained_template(model_dirs, tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps(calls))
    out = tmp_path / "out" / "merged"
    out.parent.mkdir()
    info = Q.merge_adapter(model_dirs.base, model_dirs.adapter, out,
                           template_dir=model_dirs.tdir)
    cls, _, kw = calls["load"]
    # language_model.* LoRA modules were trained on the multimodal wrapper.
    assert cls == "AutoModelForImageTextToText"
    assert kw["dtype"] == "torch.bfloat16" and kw["device_map"] == "cpu"
    assert kw["output_loading_info"] is True
    assert calls["safe_merge"] is True and calls["save"]["safe_serialization"] is True
    assert info["lora_modules"] == info["attached"] == info["b_nonzero"] == 3
    assert (out / "chat_template.jinja").read_text() == TEMPLATE
    assert json.loads((out / "tokenizer_config.json").read_text())["chat_template"] == TEMPLATE
    assert (out / "tokenizer.json").read_text() == '{"vocab": 1}'
    assert json.loads((out / "generation_config.json").read_text()) == {"temperature": 1.0}
    assert info["template"]["sha256"] == sha(TEMPLATE)
    assert json.loads((out / "merge-info.json").read_text())["attached"] == 3
    assert info["merge_device"] == "cpu"
    # The base is never written through.
    assert json.loads((model_dirs.base / "tokenizer_config.json").read_text())["chat_template"] == "OLD"


def test_merge_refuses_an_adapter_that_does_not_fully_attach(model_dirs, tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps({}, attached=1))
    with pytest.raises(Q.QuantizeError, match="3 modules but 1 attached"):
        Q.merge_adapter(model_dirs.base, model_dirs.adapter, tmp_path / "merged",
                        template_dir=model_dirs.tdir)
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("deps, why", [
    ({"b": 0.0}, "3 of 3 LoRA B matrices are zero"),
    # PEFT injects B = 0 layers for every target and loads what matches, so a
    # partial key mismatch shows up as some zero B's.
    ({"zero_b": 1}, "1 of 3 LoRA B matrices are zero"),
])
def test_merge_refuses_zero_lora_b(model_dirs, tmp_path, monkeypatch, deps, why):
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps({}, **deps))
    with pytest.raises(Q.QuantizeError, match=why):
        Q.merge_adapter(model_dirs.base, model_dirs.adapter, tmp_path / "merged",
                        template_dir=model_dirs.tdir)
    assert not (tmp_path / "merged").exists()


def test_merge_refuses_a_base_loaded_with_missing_weights(model_dirs, tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "_import_merge_deps",
                        fake_deps({}, missing=("model.layers.0.self_attn.q_proj.weight",)))
    with pytest.raises(Q.QuantizeError, match="missing weights"):
        Q.merge_adapter(model_dirs.base, model_dirs.adapter, tmp_path / "merged",
                        template_dir=model_dirs.tdir)


def test_merge_uses_torch_dtype_on_old_transformers_and_offloads_on_auto(model_dirs, tmp_path,
                                                                         monkeypatch):
    calls = {}
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps(calls, version="4.51.3"))
    Q.merge_adapter(model_dirs.base, model_dirs.adapter, tmp_path / "merged",
                    template_dir=model_dirs.tdir, device="auto", dtype="fp16")
    _, _, kw = calls["load"]
    assert kw["torch_dtype"] == "torch.float16" and "dtype" not in kw
    assert kw["device_map"] == "auto" and "offload_folder" in kw and "cpu" in kw["max_memory"]
    assert "offload_folder" in calls["peft"][1]


def test_model_class_follows_the_adapter():
    cfg = {"architectures": [ARCH]}
    text_only = ["base_model.model.model.layers.0.self_attn.q_proj"]
    assert Q.resolve_model_class("auto", cfg, MODULES) == "AutoModelForImageTextToText"
    assert Q.resolve_model_class("auto", cfg, text_only) == "AutoModelForCausalLM"
    assert Q.resolve_model_class("causal-lm", cfg, MODULES) == "AutoModelForCausalLM"


def test_template_install_refuses_another_vocabulary(model_dirs, tmp_path):
    (model_dirs.tdir / "tokenizer.json").write_text('{"vocab": 2}')
    view = tmp_path / "view"
    with pytest.raises(Q.QuantizeError, match="another vocabulary"):
        Q.prepare_base_view(model_dirs.base, view, model_dirs.tdir)


def test_base_view_symlinks_the_base_and_carries_the_template(model_dirs, tmp_path):
    view = tmp_path / "base-view"
    info = Q.prepare_base_view(model_dirs.base, view, model_dirs.tdir)
    assert (view / "model.safetensors").is_symlink()
    assert not (view / "tokenizer_config.json").is_symlink()
    assert json.loads((view / "tokenizer_config.json").read_text())["chat_template"] == TEMPLATE
    assert json.loads((model_dirs.base / "tokenizer_config.json").read_text())["chat_template"] == "OLD"
    assert info["template"]["sha256"] == sha(TEMPLATE)


# ── metadata + ppl parsing ─────────────────────────────────────────────────


def test_parse_ppl_takes_the_final_estimate():
    log = "[1]5.1,[2]5.0\nFinal estimate: PPL = 4.9876 +/- 0.03456\n"
    assert Q.parse_ppl(log) == pytest.approx(4.9876)
    assert math.isnan(Q.parse_ppl("Final estimate: PPL = nan +/- nan"))
    assert Q.parse_ppl("no estimate") is None


def test_metadata_checks():
    good = {"architecture": "qwen35", "file_type": 15, "context_length": 262144,
            "nextn_predict_layers": None, "chat_template_sha256": "t"}
    assert all(Q.metadata_checks(good, qtype="Q4_K_M", arch="qwen35", template_sha="t").values())
    bad = {**good, "nextn_predict_layers": 1, "context_length": 32768, "file_type": 17,
           "chat_template_sha256": "other"}
    failed = {k for k, v in Q.metadata_checks(bad, qtype="Q4_K_M", arch="qwen35",
                                              template_sha="t").items() if not v}
    assert failed == {"file_type", "no_nextn_layers", "context_holds_serve_line",
                      "chat_template_is_trained_one"}


# ── the run: dry-run default, execute with fakes ───────────────────────────


class FakeProc:
    """subprocess.run as the host would answer it; writes each tool's output."""

    def __init__(self, *, head=HEAD, build=None, supported=(ARCH,), ppl=None):
        self.calls = []
        self.head, self.build = head, build or HEAD[:9]
        self.quantize_build = None  # llama-quantize's own, when it differs
        # Builds before llama_print_build_info: "--version" prints "version: N (sha)"
        # and a llama-quantize run starts with print_build_info's "main: build = N (sha)".
        self.old_format = False
        self.supported = supported
        self.ppl = ppl or {"BF16": 4.0, "Q4_K_M": 4.2, "Q5_K_M": 4.08}
        self.hooks = {}  # tool name -> callable(argv), run before the tool "works"

    def tools(self):
        """One name per call: the tool, or what a check asked."""
        names = []
        for c in self.calls:
            if c[0] == "git":
                names.append("git")
            elif "--print-supported-models" in c:
                names.append("supported-models")
            elif c[-1] == "--version":
                names.append("version")
            elif len(c) > 1 and c[1].endswith("convert_hf_to_gguf.py"):
                names.append("convert_hf_to_gguf.py")
            else:
                names.append(Path(c[0]).name)
        return names

    def __call__(self, cmd, **kw):
        argv = [str(c) for c in cmd]
        self.calls.append(argv)
        out = ""
        name = Path(argv[0]).name
        if name in self.hooks:
            self.hooks[name](argv)
        if argv[0] == "git" and "rev-parse" in argv:
            out = self.head + "\n"
        elif argv[0] == "git" and "status" in argv:
            out = ""
        elif argv[-1] == "--version":
            # The pin's format (common/build-info.cpp.in at 9d57ce456).
            out = (f"version: 6789 ({self.build})\n" if self.old_format else
                   f"version: 0.1.0-dev (build 1, commit {self.build})\n")
            out += "built with cc for x86_64\n"
        elif "--print-supported-models" in argv:
            out = "TEXT models:\n" + "".join(f"  - {a}\n" for a in self.supported)
        elif len(argv) > 1 and argv[1].endswith("convert_hf_to_gguf.py"):
            Path(argv[argv.index("--outfile") + 1]).write_bytes(b"GGUF-bf16")
        elif name == "llama-imatrix":
            Path(argv[argv.index("-o") + 1]).write_bytes(b"GGUF-imatrix")
        elif name == "llama-quantize":
            Path(argv[-2]).write_bytes(b"GGUF-" + argv[-1].encode())
            # It has no --version; every run starts with the build (as at the pin).
            qb = self.quantize_build or self.build
            out = (f"main: build = 6789 ({qb})\nmain: built with cc for x86_64\n"
                   if self.old_format else f"version: 0.1.0-dev (build 1, commit {qb})\n")
            out += "llama_quantize: quantizing ...\n"
        elif name == "llama-perplexity":
            q = Path(argv[argv.index("-m") + 1]).stem.rsplit("-", 1)[-1]
            out = f"[1]4.1,[2]4.0\nFinal estimate: PPL = {self.ppl[q]:.4f} +/- 0.02\n"
        stdout = kw.get("stdout")
        if hasattr(stdout, "write"):
            stdout.write(out)
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def fake_meta(llama_dir, path, python=None):
    q = path.stem.rsplit("-", 1)[-1]
    return {"architecture": "qwen35", "file_type": Q.FILE_TYPES[q], "name": "x",
            "context_length": 262144, "nextn_predict_layers": None,
            "chat_template_sha256": sha(TEMPLATE)}


@pytest.fixture
def host(tmp_path, phase3, model_dirs, monkeypatch):
    lc = tmp_path / "llama.cpp"
    (lc / "build" / "bin").mkdir(parents=True)
    (lc / "convert_hf_to_gguf.py").write_text("# converter\n")
    for t in ("llama-imatrix", "llama-quantize", "llama-perplexity", "llama-server"):
        p = lc / "build" / "bin" / t
        p.write_text("#!/bin/sh\n")
        p.chmod(0o755)
    monkeypatch.setattr(Q, "DISK_MARGIN", 0)
    monkeypatch.setattr(Q, "read_gguf_metadata", fake_meta)
    calls = {}
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps(calls))
    proc = FakeProc()
    monkeypatch.setattr(Q.subprocess, "run", proc)
    out = tmp_path / "phase6" / "run7"
    argv = ["run", "--adapter", str(model_dirs.adapter), "--base", str(model_dirs.base),
            "--template-dir", str(model_dirs.tdir), "--data", str(phase3.data),
            "--evalsets", str(phase3.evalsets), "--out", str(out), "--llama-cpp-dir", str(lc),
            "--calib-tokens", "20000", "--imatrix-ctx", "2048", "--ppl-ctx", "2048",
            "--ppl-chunks", "2",
            # The fixture's train rows carry no provenance, so they count as
            # Max-derived (tests of the gate itself drop the flag).
            Q.APPROVAL_FLAG]
    return SimpleNamespace(argv=argv, out=out, proc=proc, lc=lc, merge_calls=calls)


def test_dry_run_is_the_default_and_runs_nothing(host, monkeypatch, capsys):
    monkeypatch.setattr(Q.subprocess, "run", lambda *a, **k: pytest.fail("dry run ran a command"))
    monkeypatch.setattr(Q, "_import_merge_deps", lambda: pytest.fail("dry run merged"))
    assert Q.main(host.argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN")
    assert not host.out.exists()
    assert "train-only check PASS" in out and "5 eval/val files" in out
    assert "--no-mtp" in out and "--parse-special" in out
    assert "-fa on -ctk q8_0 -ctv q8_0" in out
    assert f"{ARCH}: converter support at 9d57ce456 verified" in out


def test_dry_run_reports_a_leak_and_fails(host, phase3, capsys):
    jsonl(phase3.data / "train.jsonl",
          [train_row(i) for i in range(N_TRAIN)] + [train_row(5, id="p3-x", task_id="etask0")])
    assert Q.main(host.argv) == 2
    assert "share a key with the eval / val sets" in capsys.readouterr().err
    assert not host.out.exists()


def test_execute_stops_on_a_leak_before_any_merge(host, phase3, capsys):
    jsonl(phase3.data / "train.jsonl",
          [train_row(i) for i in range(N_TRAIN)] + [train_row(5, id="p3-x", group_id="egroup1")])
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "share a key with the eval / val sets" in capsys.readouterr().err
    assert "load" not in host.merge_calls
    assert not (host.out / "calib").exists() and not (host.out / "merged").exists()
    assert "convert_hf_to_gguf.py" not in host.proc.tools()


def test_execute_runs_every_step_and_writes_the_manifest(host, capsys):
    assert Q.main(host.argv + ["--execute"]) == 0
    tools = host.proc.tools()
    order = [t for t in tools if t in ("convert_hf_to_gguf.py", "llama-imatrix",
                                       "llama-quantize", "llama-perplexity")]
    assert order == ["convert_hf_to_gguf.py", "llama-imatrix", "llama-quantize",
                     "llama-quantize", "llama-perplexity", "llama-perplexity",
                     "llama-perplexity"]
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["schema"] == "distill-quantize-manifest/1" and m["result"] == "PASS"
    assert m["llama_cpp"]["head"] == HEAD and m["llama_cpp"]["converter_registers"] == ARCH
    by_name = {a["name"]: a for a in m["artifacts"]}
    assert set(by_name) == {"run7-BF16.gguf", "run7.imatrix.gguf", "run7-Q4_K_M.gguf",
                            "run7-Q5_K_M.gguf", "calibration.txt", "ppl.txt"}
    for a in m["artifacts"]:
        p = Path(a["path"])
        assert a["bytes"] == p.stat().st_size
        assert a["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert m["sanity"]["pass"] is True
    assert m["sanity"]["models"]["Q4_K_M"]["ratio_to_ref"] == pytest.approx(4.2 / 4.0)
    assert "-c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0" in m["serve"]["Q4_K_M"]
    assert m["steps"]["merge"]["attached"] == 3
    assert m["inputs"]["template_sha256"] == sha(TEMPLATE)
    # No partial file survives, and the ids stay in calib/, not the manifest.
    assert not list(host.out.rglob("*.partial*"))
    assert "calibration_ids" not in m["steps"]["calibrate"]
    # The imatrix read the calibration file, and that file is train text only.
    im = next(c for c in host.proc.calls if c[0].endswith("llama-imatrix") and "-f" in c)
    calib = Path(im[im.index("-f") + 1])
    text = calib.read_text()
    assert "TRAINTEXT" in text and not any(mk in text for mk in EVAL_MARKERS)
    rec = json.loads((host.out / "calib" / "calibration.json").read_text())
    assert not set(rec["calibration_ids"] + rec["ppl_ids"]) & EVAL_IDS


def test_execute_fails_sanity_on_a_degraded_quant(host):
    host.proc.ppl["Q4_K_M"] = 4.0 * 1.2
    assert Q.main(host.argv + ["--execute"]) == 1
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["result"] == "FAIL"
    assert m["sanity"]["models"]["Q4_K_M"]["checks"]["ppl_ratio"] is False
    assert m["sanity"]["models"]["Q5_K_M"]["pass"] is True


def test_execute_refuses_another_llama_cpp_commit(host, capsys):
    host.proc.head = "abcdef0123" + "0" * 30
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "not the pinned 9d57ce456" in capsys.readouterr().err
    assert "convert_hf_to_gguf.py" not in host.proc.tools()


def test_execute_refuses_binaries_from_another_build(host, capsys):
    host.proc.build = "1234567ab"
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "rebuild at the pin" in capsys.readouterr().err


@pytest.mark.parametrize("line", [
    "version: 0.1.0-dev (build 1, commit 9d57ce4)",  # the pin's own format (a shallow clone)
    "version: 6789 (9d57ce45)",  # older builds
])
def test_the_version_line_of_a_build_at_the_pin_is_recognised(line):
    # The real run at 9d57ce456 printed the first line; the old pattern only
    # knew the second, so preflight refused a correct build.
    m = Q._BUILD.search(line + "\nbuilt with AppleClang 17.0.0 for Darwin arm64\n")
    assert m is not None and HEAD.startswith(m.group(1))


def test_a_llama_quantize_from_another_build_is_refused_by_its_own_log(host, capsys):
    # Preflight cannot ask llama-quantize (--version prints its usage, exit 1
    # at the pin), so a partial rebuild could leave an old one beside the rest.
    host.proc.quantize_build = "1234567ab"
    assert Q.main(host.argv + ["--execute"]) == 2
    err = capsys.readouterr().err
    assert "llama-quantize is not built from 9d57ce456" in err and "1234567ab" in err
    assert not list((host.out / "gguf").glob("*Q4_K_M*"))  # the partial is gone too


def test_the_manifest_records_llama_quantizes_build(host):
    assert Q.main(host.argv + ["--execute"]) == 0
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["llama_cpp"]["builds"]["llama-quantize"] == HEAD[:9]
    # A rerun (or a resume after a pause in sanity) keeps both quants; the
    # manifest still names the build that made them, read from their logs.
    host.proc.calls.clear()
    assert Q.main(host.argv + ["--execute"]) == 0
    assert "llama-quantize" not in host.proc.tools()
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["result"] == "PASS" and m["llama_cpp"]["builds"]["llama-quantize"] == HEAD[:9]


@pytest.mark.parametrize("log", ["missing", "another build", "no build line"])
def test_a_kept_quant_is_checked_by_the_log_of_the_run_that_made_it(host, capsys, log):
    assert Q.main(host.argv + ["--execute"]) == 0
    qlog = host.out / "logs" / "quantize-Q4_K_M.log"
    if log == "missing":
        qlog.unlink()
    elif log == "another build":
        qlog.write_text(qlog.read_text().replace(f"commit {HEAD[:9]}", "commit 1234567ab"))
    else:
        qlog.write_text("$ llama-quantize ...\nllama_quantize: quantizing ...\n")
    capsys.readouterr()
    assert Q.main(host.argv + ["--execute"]) == 2
    err = capsys.readouterr().err
    assert "run7-Q4_K_M.gguf" in err and "delete it" in err, err
    if log == "another build":
        assert "1234567ab" in err
    # Refused, not removed: the operator decides.
    assert (host.out / "gguf" / "run7-Q4_K_M.gguf").exists()
    assert "llama-perplexity" not in host.proc.tools()[-3:]


def test_a_build_at_the_pin_that_logs_the_older_way_passes(host):
    # An allowed --llama-cpp-commit from before llama_print_build_info: preflight
    # took its "version: N (sha)", and step 6 then refused its own log's
    # "main: build = N (sha)" after the whole quantize.
    host.proc.old_format = True
    assert Q.main(host.argv + ["--execute"]) == 0
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["llama_cpp"]["builds"]["llama-quantize"] == HEAD[:9]
    assert m["llama_cpp"]["builds"]["llama-imatrix"] == HEAD[:9]


def test_an_older_format_log_from_another_build_is_still_refused(host, capsys):
    host.proc.old_format = True
    host.proc.quantize_build = "1234567ab"
    assert Q.main(host.argv + ["--execute"]) == 2
    err = capsys.readouterr().err
    assert "llama-quantize is not built from 9d57ce456" in err and "1234567ab" in err


@pytest.mark.parametrize("text, want", [
    ("version: 0.1.0-dev (build 1, commit 9d57ce4)\nbuilt with cc for x86_64\n", "9d57ce4"),
    ("version: 6789 (9d57ce45)\n", "9d57ce45"),
    ("main: build = 6789 (9d57ce45)\nmain: built with cc for x86_64\n", "9d57ce45"),
    ("llama_model_loader: - kv 0: general.name str = x (9d57ce45)\n", None),
    ("main: build = 6789 (unknown)\n", None),
])
def test_built_from_reads_both_build_line_formats(text, want):
    assert Q.built_from("$ llama-quantize ...\n" + text) == want


def test_a_build_with_an_unknown_commit_is_refused(host, capsys):
    # build-info.cmake writes "unknown" outside a git checkout.
    host.proc.build = "unknown"
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "rebuild at the pin" in capsys.readouterr().err


def test_execute_refuses_a_converter_without_the_arch(host, capsys):
    host.proc.supported = ("Qwen3ForCausalLM",)
    assert Q.main(host.argv + ["--execute"]) == 2
    err = capsys.readouterr().err
    assert f"does not register {ARCH}" in err and "newer" in err
    assert not (host.out / "merged").exists()


def test_execute_resumes_and_keeps_finished_steps(host):
    assert Q.main(host.argv + ["--execute"]) == 0
    host.proc.calls.clear()
    host.merge_calls.clear()
    assert Q.main(host.argv + ["--execute"]) == 0
    tools = host.proc.tools()
    assert not {"convert_hf_to_gguf.py", "llama-imatrix", "llama-quantize"} & set(tools)
    assert tools.count("llama-perplexity") == 3
    assert "load" not in host.merge_calls


def test_a_later_run_may_add_a_quant_type(host):
    assert Q.main(host.argv + ["--execute"]) == 0
    host.proc.calls.clear()
    host.proc.ppl["Q6_K"] = 4.02
    assert Q.main(host.argv + ["--execute", "--quants", "Q4_K_M,Q5_K_M,Q6_K"]) == 0
    qz = [c for c in host.proc.calls if c[0].endswith("llama-quantize")]
    assert [c[-1] for c in qz] == ["Q6_K"]
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert set(m["serve"]) == {"Q4_K_M", "Q5_K_M", "Q6_K"} and m["result"] == "PASS"


def test_execute_refuses_an_out_made_from_other_inputs(host, model_dirs, capsys):
    assert Q.main(host.argv + ["--execute"]) == 0
    write_safetensors(model_dirs.adapter / "adapter_model.safetensors",
                      [f"{m}.lora_{s}.weight" for m in MODULES for s in "AB"] + ["extra.lora_A.weight"])
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "made from other inputs (adapter)" in capsys.readouterr().err


def test_no_adapter_runs_the_base_through_the_same_recipe(host, model_dirs):
    argv = [a for a in host.argv]
    i = argv.index("--adapter")
    del argv[i:i + 2]
    argv[argv.index("--out") + 1] = str(host.out.parent / "base-9b")
    assert Q.main(argv + ["--no-adapter", "--execute"]) == 0
    out = host.out.parent / "base-9b"
    assert (out / "base-view" / "model.safetensors").is_symlink()
    assert "load" not in host.merge_calls
    conv = next(c for c in host.proc.calls if "--outfile" in c)
    assert conv[2] == str(out / "base-view") and "--no-mtp" in conv
    m = json.loads((out / "quantize-manifest.json").read_text())
    assert m["result"] == "PASS" and m["inputs"]["template_sha256"] == sha(TEMPLATE)
    assert {a["name"] for a in m["artifacts"]} >= {"base-base-Q4_K_M.gguf"}


def test_calibrate_subcommand_writes_only_with_execute(phase3, tmp_path, monkeypatch):
    monkeypatch.setattr(Q.subprocess, "run", lambda *a, **k: pytest.fail("calibrate ran a command"))
    out = tmp_path / "calrun"
    args = ["calibrate", "--data", str(phase3.data), "--evalsets", str(phase3.evalsets),
            "--out", str(out), "--calib-tokens", "20000", "--imatrix-ctx", "2048",
            "--ppl-ctx", "2048", "--ppl-chunks", "2"]
    assert Q.main(args) == 0 and not out.exists()
    assert Q.main(args + ["--execute"]) == 0
    rec = json.loads((out / "calib" / "calibration.json").read_text())
    assert rec["calibration"]["sha256"] == sha((out / "calib" / "calibration.txt").read_text())
    assert not set(rec["calibration_ids"]) & EVAL_IDS


def calibrate_args(phase3, out, *extra):
    return ["calibrate", "--data", str(phase3.data), "--evalsets", str(phase3.evalsets),
            "--out", str(out), "--calib-tokens", "20000", "--imatrix-ctx", "2048",
            "--ppl-ctx", "2048", "--ppl-chunks", "2", "--execute", *extra]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_run_uses_a_calibration_made_elsewhere_when_the_splits_are_absent(host, phase3, tmp_path,
                                                                          capsys):
    assert Q.main(calibrate_args(phase3, host.out)) == 0
    argv = list(host.argv)
    argv[argv.index("--data") + 1] = str(tmp_path / "not-here")
    # Nothing ties that calibration to the split this model goes with: a
    # rebuild may have moved its tasks into eval since.
    capsys.readouterr()
    assert Q.main(argv + ["--execute"]) == 2
    train_sha = file_sha(phase3.data / "train.jsonl")
    assert train_sha in capsys.readouterr().err
    assert "convert_hf_to_gguf.py" not in host.proc.tools()
    argv += ["--train-sha256", train_sha]
    assert Q.main(argv + ["--execute"]) == 0
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["steps"]["calibrate"]["source"] == "train split only"
    assert m["steps"]["calibrate"]["train_split"] == {"train_sha256": train_sha,
                                                      "matched": "--train-sha256"}
    # A calibration file edited after the fact is refused.
    (host.out / "calib" / "calibration.txt").write_text("EVALTEXT0")
    assert Q.main(argv + ["--execute"]) == 2


@pytest.mark.parametrize("change", [["--seed", "9"], ["--ppl-ctx", "8192"],
                                    ["--imatrix-ctx", "4096"], ["--calib-tokens", "30000"]])
def test_a_calibration_made_with_other_settings_is_not_used(host, phase3, tmp_path, capsys,
                                                            change):
    assert Q.main(calibrate_args(phase3, host.out, "--seed", "5")) == 0
    rec = json.loads((host.out / "calib" / "calibration.json").read_text())
    assert rec["settings"] == {"seed": 5, "calib_tokens": 20000, "imatrix_ctx": 2048,
                               "ppl_ctx": 2048, "ppl_chunks": 2}
    argv = list(host.argv) + ["--seed", "5", "--train-sha256",
                              file_sha(phase3.data / "train.jsonl")]
    argv[argv.index("--data") + 1] = str(tmp_path / "not-here")
    capsys.readouterr()
    # A ppl slice sized for -c 2048 would leave llama-perplexity short of the
    # 2 x 8192 tokens it needs, and every model would "fail" sanity.
    assert Q.main(argv + change + ["--execute"]) == 2
    err = capsys.readouterr().err
    assert "other settings" in err and change[0][2:].replace("-", "_") in err
    assert "convert_hf_to_gguf.py" not in host.proc.tools()
    assert Q.main(argv + ["--execute"]) == 0


def test_the_trainer_manifest_ties_the_calibration_to_the_adapters_split(host, phase3, model_dirs,
                                                                        tmp_path, capsys):
    train = phase3.data / "train.jsonl"
    tm = {"schema": "distill-train-manifest/1", "data": {"path": str(train), "sha256": "0" * 64}}
    (model_dirs.adapter.parent / "manifest.json").write_text(json.dumps(tm))
    # The splits here are not the ones the adapter trained on.
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "trainer manifest" in capsys.readouterr().err
    # Refused before anything was written, so the right --data is not refused next.
    assert not (host.out / "calib").exists() and not (host.out / "run.json").exists()
    tm["data"]["sha256"] = file_sha(train)
    (model_dirs.adapter.parent / "manifest.json").write_text(json.dumps(tm))
    assert Q.main(host.argv + ["--execute"]) == 0
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["steps"]["calibrate"]["train_split"]["matched"] == "the adapter's trainer manifest"
    # Without the splits, the manifest is what ties a calibrate-made calibration.
    out2 = tmp_path / "phase6" / "run7b"
    assert Q.main(calibrate_args(phase3, out2)) == 0
    argv = list(host.argv)
    argv[argv.index("--data") + 1] = str(tmp_path / "not-here")
    argv[argv.index("--out") + 1] = str(out2)
    assert Q.main(argv + ["--execute"]) == 0


# ── run.json: written once an output depends on it ─────────────────────────


@pytest.mark.parametrize("make_fail, fix, said", [
    ("commit", ["--llama-cpp-commit", "abcdef012"], "or pass --llama-cpp-commit"),
    ("calib", ["--calib-tokens", "20000"], "lower --calib-tokens"),
])
def test_the_fix_a_refusal_suggests_works_on_the_same_out(host, capsys, make_fail, fix, said):
    argv = list(host.argv)
    if make_fail == "commit":
        host.proc.head, host.proc.build = "abcdef0123" + "0" * 30, "abcdef012"
    else:
        argv[argv.index("--calib-tokens") + 1] = "125000"  # every row: no ppl slice left
    assert Q.main(argv + ["--execute"]) == 2
    assert said in capsys.readouterr().err
    assert not (host.out / "run.json").exists()
    assert Q.main(argv + fix + ["--execute"]) == 0
    assert json.loads((host.out / "run.json").read_text())["llama_cpp_commit"] == (
        "abcdef012" if make_fail == "commit" else "9d57ce456")


def test_a_run_json_whose_inputs_made_nothing_is_replaced(host, monkeypatch, capsys):
    # The merge refuses after calibration (run.json is written by then) and
    # says to try --model-class; nothing was merged, so that fix runs.
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps(host.merge_calls, attached=1))
    assert Q.main(host.argv + ["--execute", "--model-class", "causal-lm"]) == 2
    assert "--model-class" in capsys.readouterr().err
    assert json.loads((host.out / "run.json").read_text())["model_class"] == "causal-lm"
    monkeypatch.setattr(Q, "_import_merge_deps", fake_deps(host.merge_calls))
    assert Q.main(host.argv + ["--execute", "--model-class", "image-text-to-text"]) == 0
    assert "made nothing yet; replaced" in capsys.readouterr().out
    assert json.loads((host.out / "run.json").read_text())["model_class"] == "image-text-to-text"


def test_the_real_injection_manifest_drops_a_planted_case_and_nothing_else(phase3, tmp_path):
    fixtures = REPO / "tests" / "fixtures" / "distill_evalsets"
    case = json.loads((fixtures / "eval-injection.sample.jsonl").read_text().splitlines()[0])
    shutil.copy(fixtures / "eval-injection.manifest.sample.json",
                phase3.evalsets / "eval-injection.manifest.json")
    rows = [train_row(i) for i in range(N_TRAIN)]
    rows[3]["text"] += "tool result: " + case["injection"]["instruction"]
    jsonl(phase3.data / "train.jsonl", rows)
    out = tmp_path / "calrun"
    # The manifest in --evalsets is picked up without being named. The budgets
    # ask for more than the rows hold, so every row is checked.
    assert Q.main(["calibrate", "--data", str(phase3.data), "--evalsets", str(phase3.evalsets),
                   "--out", str(out), "--calib-tokens", str((N_TRAIN - 3) * ROW_TOKENS),
                   "--imatrix-ctx", "2048", "--ppl-ctx", "2048", "--ppl-chunks", "2",
                   "--execute"]) == 0
    rec = json.loads((out / "calib" / "calibration.json").read_text())
    assert rec["eval_check"]["injection_manifest"] is True
    assert rec["eval_check"]["dropped_injection_overlap"] == 1
    assert "p3-train003" not in rec["calibration_ids"] + rec["ppl_ids"]
    assert len(rec["calibration_ids"]) + len(rec["ppl_ids"]) == N_TRAIN - 1


# ── the leak check reads every eval set build_eval_sets.py writes ──────────


def test_the_evalset_paths_are_the_ones_the_builder_writes():
    B = Q._sibling("distill_build_eval_sets", "build_eval_sets.py")
    assert set(Q.EVALSET_FILES) == {B.OUT_FILES[k] for k in ("longctx", "longctx_tools",
                                                             "injection", "sanity")}


def test_the_ungated_longctx_tool_cases_are_checked(phase3):
    jsonl(phase3.evalsets / "ungated" / "eval-longctx-tools.jsonl", [
        {"id": "longctx-32k-toolcall-d50-0", "set": "eval-tools", "depth_bucket": "32k",
         "messages": [{"role": "user", "content": "LONGTOOLTEXT"}],
         "meta": {"task_id": "ltask0"}}])
    cal = calibrate(phase3)
    assert "eval-longctx-tools.jsonl" in {f["name"] for f in cal.info["eval_check"]["files"]}
    jsonl(phase3.data / "train.jsonl",
          [train_row(i) for i in range(N_TRAIN)] + [train_row(99, task_id="ltask0")])
    with pytest.raises(Q.CalibrationError, match="ltask0 is in eval-longctx-tools.jsonl"):
        calibrate(phase3)


# ── provenance: OPEN-QUESTIONS #6 ──────────────────────────────────────────


def _base_argv(host):
    argv = list(host.argv)
    i = argv.index("--adapter")
    del argv[i:i + 2]
    argv[argv.index("--out") + 1] = str(host.out.parent / "base-9b")
    return argv + ["--no-adapter"]


@pytest.mark.parametrize("tuned", [True, False])
def test_max_derived_calibration_is_refused_without_the_operators_approval(host, capsys, tuned):
    argv = [a for a in (host.argv if tuned else _base_argv(host)) if a != Q.APPROVAL_FLAG]
    out = Path(argv[argv.index("--out") + 1])
    assert Q.main(argv + ["--execute"]) == 3
    err = capsys.readouterr().err
    assert "DISTILL-OPEN-QUESTIONS #6" in err and Q.APPROVAL_FLAG in err
    tools = set(host.proc.tools())
    assert not {"convert_hf_to_gguf.py", "llama-imatrix", "llama-quantize"} & tools
    assert "load" not in host.merge_calls
    assert not (out / "calib").exists() and not (out / "run.json").exists()
    assert not (out / "base-view").exists() and not (out / "merged").exists()
    # The plan says so before anything runs.
    assert Q.main(argv) == 0
    assert f"run --execute REFUSED without {Q.APPROVAL_FLAG}" in capsys.readouterr().out


def test_rows_with_another_source_need_no_approval_and_the_counts_are_recorded(host, phase3):
    # One task's rows are Max-derived: the first round-robin round takes a row
    # of every task, so one is in the calibration.
    rows = [train_row(i, source="max-golden" if i % N_TASKS == 0 else "local-gym")
            for i in range(N_TRAIN)]
    jsonl(phase3.data / "train.jsonl", rows)
    argv = [a for a in host.argv if a != Q.APPROVAL_FLAG]
    assert Q.main(argv + ["--execute"]) == 3
    jsonl(phase3.data / "train.jsonl", [train_row(i, source="local-gym") for i in range(N_TRAIN)])
    # a source without the word "max" is no evidence on its own (build.py labels
    # teacher rows "extra" too): the builder's report must name a non-Max teacher
    assert Q.main(argv + ["--execute"]) == 3
    (phase3.data / "report.json").write_text(
        json.dumps({"settings": {"teacher_model": "local-teacher-30b"}}))
    assert Q.main(argv + ["--execute"]) == 0
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["max_derived"]["calibration_max_derived"] == 0
    assert m["max_derived"]["approved"] is False
    cal = m["steps"]["calibrate"]
    assert cal["provenance"]["sources"] == {
        "local-gym": cal["calibration"]["rows"] + cal["ppl_slice"]["rows"]}


def test_an_approved_run_records_the_approval(host):
    assert Q.main(host.argv + ["--execute"]) == 0
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    rows = m["steps"]["calibrate"]["calibration"]["rows"]
    assert m["max_derived"] == {"calibration_rows": rows, "calibration_max_derived": rows,
                                "ppl_max_derived": m["steps"]["calibrate"]["ppl_slice"]["rows"],
                                "approved": True}


def test_the_approval_flag_is_the_trainers():
    T = Q._trainer()
    assert Q.APPROVAL_FLAG == T.APPROVAL_FLAG


# ── GGUF metadata: read under --convert-python ─────────────────────────────


GGUF_STUB = '''
import json


class GGUFValueType:
    STRING = 8


class _Field:
    def __init__(self, v):
        self.v = v

    def contents(self):
        return self.v


class GGUFReader:
    def __init__(self, path):
        with open(path) as fh:
            self.fields = json.load(fh)

    def get_field(self, key):
        return _Field(self.fields[key]) if key in self.fields else None
'''


def test_gguf_metadata_is_read_under_the_convert_python_not_this_one(tmp_path, monkeypatch):
    # A json.py in the working directory would shadow the stdlib under a bare -c.
    work = tmp_path / "cwd"
    work.mkdir()
    (work / "json.py").write_text("raise SystemExit('planted json.py ran')\n")
    monkeypatch.chdir(work)
    lc = tmp_path / "llama.cpp"
    (lc / "gguf-py" / "gguf").mkdir(parents=True)
    (lc / "gguf-py" / "gguf" / "__init__.py").write_text(GGUF_STUB)
    good = tmp_path / "m-Q4_K_M.gguf"
    good.write_text(json.dumps({
        "general.architecture": "qwen35", "general.file_type": 15, "general.name": "m",
        "qwen35.context_length": 262144, "tokenizer.chat_template": TEMPLATE}))
    convert_python = "/opt/lc-venv/bin/python"  # the one with numpy; stood in for below
    seen = []
    real = subprocess.run

    def spy(cmd, **kw):
        seen.append(cmd[:3])
        return real([sys.executable, *cmd[1:]], **kw)

    monkeypatch.setattr(Q.subprocess, "run", spy)
    meta = Q.read_gguf_metadata(lc, good, convert_python)
    assert seen == [[convert_python, "-I", "-c"]]
    assert meta == {"architecture": "qwen35", "file_type": 15, "name": "m",
                    "context_length": 262144, "nextn_predict_layers": None,
                    "chat_template_sha256": sha(TEMPLATE)}
    # gguf-py (and the numpy it needs) never came into this interpreter.
    assert "gguf" not in sys.modules and str(lc / "gguf-py") not in sys.path
    bad = tmp_path / "broken.gguf"
    bad.write_text("not a gguf")
    with pytest.raises(Q.QuantizeError, match="could not read broken.gguf"):
        Q.read_gguf_metadata(lc, bad, convert_python)


def test_sanity_reads_metadata_with_the_convert_python(host, monkeypatch):
    used = []

    def meta(llama_dir, path, python=None):
        used.append(python)
        return fake_meta(llama_dir, path)

    monkeypatch.setattr(Q, "read_gguf_metadata", meta)
    assert Q.main(host.argv + ["--execute", "--convert-python", "/opt/lc-venv/bin/python"]) == 0
    assert used and set(used) == {"/opt/lc-venv/bin/python"}


# ── exit codes: 1 is only a failed sanity check ────────────────────────────


@pytest.mark.parametrize("extra", [
    ["--quants", "Q3_K"],
    ["--max-ppl-ratio", "Q4_K_M=x"],
    ["--max-ppl-ratio", "Q8_0=1.2"],
    ["--no-adapter"],  # beside --adapter
    ["--calib-tokens", "0"],
    ["--stop-at", "2026-11-01T06:45"],  # no UTC offset
    ["--train-sha256", "abc"],
])
def test_usage_errors_exit_2_not_the_sanity_code(host, capsys, extra):
    assert Q.main(host.argv + extra + ["--execute"]) == 2
    assert "ERROR:" in capsys.readouterr().err
    assert host.proc.calls == []


def test_a_lowercase_quant_in_max_ppl_ratio_is_accepted(host, capsys):
    assert Q.main(host.argv + ["--max-ppl-ratio", "q4_k_m=1.2"]) == 0
    assert "Q4_K_M <= 1.2" in capsys.readouterr().out


def test_an_unexpected_exception_in_a_step_exits_2(host, monkeypatch, capsys):
    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(Q, "merge_adapter", oom)
    assert Q.main(host.argv + ["--execute"]) == 2
    assert "ERROR: RuntimeError: CUDA out of memory" in capsys.readouterr().err


# ── the window: --stop-at, SIGTERM, a distill-* unit ──────────────────────


def _iso(delta):
    return (datetime.now(timezone.utc) + delta).isoformat(timespec="seconds")


def test_a_stop_at_that_has_passed_stops_before_anything_runs(host, capsys):
    assert Q.main(host.argv + ["--execute", "--stop-at", _iso(-timedelta(hours=1))]) == 75
    assert "PAUSED" in capsys.readouterr().err
    assert host.proc.calls == [] and not host.out.exists()


def test_sigterm_mid_step_pauses_and_the_same_command_resumes(host):
    before = signal.getsignal(signal.SIGTERM)

    def term(argv):
        if argv[-1] == "--version":  # preflight's build check, not the step
            return
        # Only with this run's handler in place: the default would end pytest.
        assert signal.getsignal(signal.SIGTERM) is not before
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)  # the handler raises before this ends

    host.proc.hooks["llama-imatrix"] = term
    assert Q.main(host.argv + ["--execute"]) == 75
    assert signal.getsignal(signal.SIGTERM) is before
    m = json.loads((host.out / "quantize-manifest.json").read_text())
    assert m["result"] == "PAUSED"
    gguf = host.out / "gguf"
    assert (gguf / "run7-BF16.gguf").exists() and not (gguf / "run7.imatrix.gguf").exists()
    host.proc.hooks.clear()
    host.proc.calls.clear()
    assert Q.main(host.argv + ["--execute"]) == 0
    tools = host.proc.tools()
    assert "convert_hf_to_gguf.py" not in tools and tools.count("llama-imatrix") == 1


def test_stop_at_arms_a_timer_that_stops_the_running_step(host):
    armed = []

    def alarm(argv):
        armed.append(signal.getitimer(signal.ITIMER_REAL)[0])
        assert callable(signal.getsignal(signal.SIGALRM))  # the default would end pytest
        os.kill(os.getpid(), signal.SIGALRM)
        time.sleep(1)

    host.proc.hooks["llama-quantize"] = alarm
    assert Q.main(host.argv + ["--execute", "--stop-at", _iso(timedelta(hours=1))]) == 75
    # An hour ahead, less the margin for the kill and the exit.
    assert 3600 - Q.STOP_MARGIN_S - 30 < armed[0] <= 3600 - Q.STOP_MARGIN_S
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    assert not (host.out / "gguf" / "run7-Q4_K_M.gguf").exists()


def test_detach_runs_the_same_command_as_a_distill_unit(host, monkeypatch, capsys):
    import fnmatch

    stop = _iso(timedelta(hours=5))
    assert Q.main(host.argv + ["--detach", "--stop-at", stop]) == 0
    assert "detached (with --execute): systemd-run --user --unit=distill-quantize-" in (
        capsys.readouterr().out)
    launched = []
    monkeypatch.setattr(Q.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Q.subprocess, "run", lambda cmd, **kw: launched.append(cmd)
                        or subprocess.CompletedProcess(cmd, 0))
    assert Q.main(host.argv + ["--execute", "--detach", "--stop-at", stop]) == 0
    (cmd,) = launched
    unit = next(x for x in cmd if x.startswith("--unit="))[len("--unit="):]
    # setups.py's DISTILL_JOBS: a restore stops it, and its GPU processes are
    # in a unit it manages.
    assert unit.startswith("distill-quantize-") and fnmatch.fnmatch(f"{unit}.service", "distill-*")
    assert cmd[:2] == ["systemd-run", "--user"] and "KillSignal=SIGTERM" in cmd
    inner = cmd[cmd.index(sys.executable):]
    assert inner[1] == str(REPO / "scripts" / "distill" / "quantize.py")
    assert "--detach" not in inner and "--execute" in inner
    assert inner[inner.index("--stop-at") + 1] == stop and Q.APPROVAL_FLAG in inner
    assert not (host.out / "calib").exists()  # nothing ran here
