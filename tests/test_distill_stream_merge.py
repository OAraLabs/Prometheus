"""scripts/distill/stream_merge.py and quantize.py --merge-mode stream.

The plan, the refusals and the shard writing run on tiny safetensors files with
a pure-Python stand-in for the arithmetic (torch is not in the test env). The
arithmetic itself is checked against PEFT only where torch and peft import
(skipped otherwise); on the 3090 Ti host it was checked against PEFT's own
merge on 30 modules (every targeted kind) of the 9B smoke adapter, 2026-10-31: bit-identical.
"""

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name, file):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "distill" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# quantize.py loads its sibling under this name, so a patch here is the one it uses
SM = _load("distill_stream_merge", "stream_merge.py")
Q = _load("distill_quantize", "quantize.py")

PFX = "base_model.model."
Q_W = "model.language_model.layers.0.self_attn.q_proj.weight"
V_W = "model.language_model.layers.0.self_attn.v_proj.weight"


# ── a pure-Python stand-in for TorchMath (same contract, f32/bf16 only) ─────────


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def to_bf16(x):
    try:
        u = f32_bits(x)
    except OverflowError:  # beyond f32: inf, as a float32 product would be
        u = f32_bits(math.copysign(math.inf, x))
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


def from_bf16(h):
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def decode(buf, dtype):
    if dtype == "F32":
        return list(struct.unpack(f"<{len(buf) // 4}f", bytes(buf)))
    if dtype == "BF16":
        return [from_bf16(h) for h in struct.unpack(f"<{len(buf) // 2}H", bytes(buf))]
    raise AssertionError(dtype)


def encode(vals, dtype):
    if dtype == "F32":
        return struct.pack(f"<{len(vals)}f", *vals)
    if dtype == "BF16":
        return struct.pack(f"<{len(vals)}H", *(to_bf16(v) for v in vals))
    raise AssertionError(dtype)


class PyMath:
    calls = []

    def cast(self, buf, dtype, shape, out_dtype):
        return memoryview(encode(decode(buf, dtype), out_dtype))

    def b_nonzero(self, buf, dtype, shape):
        return any(v != 0 for v in decode(buf, dtype))

    def merge(self, w, w_dtype, w_shape, a, a_dtype, a_shape, b, b_dtype, b_shape, scaling,
              out_dtype):
        PyMath.calls.append((tuple(w_shape), scaling))
        W = [from_bf16(to_bf16(v)) if out_dtype == "BF16" else v for v in decode(w, w_dtype)]
        A, B = decode(a, a_dtype), decode(b, b_dtype)
        rows, cols, r = w_shape[0], w_shape[1], a_shape[0]
        out = []
        for i in range(rows):
            for j in range(cols):
                d = sum(B[i * r + k] * A[k * cols + j] for k in range(r)) * scaling
                out.append(W[i * cols + j] + from_bf16(to_bf16(d)))
        if any(not math.isfinite(v) for v in out):
            raise SM.MergeError("non-finite")
        return memoryview(encode(out, out_dtype))


def write_st(path, tensors):
    """tensors: {name: (dtype, shape, raw bytes)} -> a .safetensors file."""
    header, off, data = {}, 0, b""
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data += raw
    raw_h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw_h)) + raw_h + data)


def read_st(path, aligned=True):
    blob = path.read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    assert n % 8 == 0 or not aligned  # what we write is 8-byte aligned, as safetensors writes
    header = json.loads(blob[8:8 + n])
    meta = header.pop("__metadata__", None)
    out, end = {}, 0
    for name, v in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        s, e = v["data_offsets"]
        assert s == end  # contiguous, as the safetensors loader requires
        end = e
        out[name] = (v["dtype"], v["shape"], blob[8 + n + s:8 + n + e])
    assert 8 + n + end == len(blob)
    return meta, out


EMBED = ("BF16", [3, 2], encode([0.5, -1.0, 2.0, 0.25, 1.5, -0.75], "BF16"))
QW = ("BF16", [3, 2], encode([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "BF16"))
NORM = ("F32", [2], encode([1.0000001, 0.3], "F32"))
MTP = ("BF16", [2], encode([9.0, 9.0], "BF16"))
VW = ("BF16", [2, 2], encode([0.1, 0.2, 0.3, 0.4], "BF16"))
POS = ("I64", [2], struct.pack("<2q", 7, 11))


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(SM, "TorchMath", PyMath)
    PyMath.calls = []
    base, adapter, tdir = tmp_path / "base", tmp_path / "run" / "adapter-max-derived", tmp_path / "tok"
    for d in (base, adapter, tdir):
        d.mkdir(parents=True)
    s1, s2 = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    write_st(base / s1, {"model.language_model.embed_tokens.weight": EMBED, Q_W: QW,
                         "model.language_model.layers.0.linear_attn.norm.weight": NORM,
                         "mtp.fc.weight": MTP})
    write_st(base / s2, {V_W: VW, "model.visual.pos_ids": POS})
    wm = {"model.language_model.embed_tokens.weight": s1, Q_W: s1, "mtp.fc.weight": s1,
          "model.language_model.layers.0.linear_attn.norm.weight": s1, V_W: s2,
          "model.visual.pos_ids": s2}
    (base / SM.INDEX).write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (base / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen3_5ForConditionalGeneration"], "keep": "the base's own"}))
    (base / "tokenizer.json").write_text('{"vocab": 1}')
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": "OLD"}))
    (tdir / "tokenizer.json").write_text('{"vocab": 1}')
    (tdir / "chat_template.jinja").write_text("TRAINED {{ messages }}")
    write_adapter(adapter)
    return base, adapter, tdir


def write_adapter(adapter, cfg=None, tensors=None):
    c = {"peft_type": "LORA", "r": 1, "lora_alpha": 2, "use_dora": False, "fan_in_fan_out": False,
         "lora_bias": False, "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {},
         "target_modules": ["q_proj", "v_proj"]}
    c.update(cfg or {})
    (adapter / "adapter_config.json").write_text(json.dumps(c))
    t = {
        f"{PFX}{Q_W[:-7]}.lora_A.weight": ("F32", [1, 2], encode([0.5, -0.5], "F32")),
        f"{PFX}{Q_W[:-7]}.lora_B.weight": ("F32", [3, 1], encode([0.25, 0.0, 1.0], "F32")),
        f"{PFX}{V_W[:-7]}.lora_A.weight": ("F32", [1, 2], encode([1.0, 1.0], "F32")),
        f"{PFX}{V_W[:-7]}.lora_B.weight": ("F32", [2, 1], encode([0.125, -0.125], "F32")),
    }
    if tensors is not None:
        t = tensors(t)
    write_st(adapter / "adapter_model.safetensors", t)


def expected(w, a, b, scaling):
    return encode(decode(PyMath().merge(bytearray(w[2]), w[0], w[1], bytearray(a[2]), a[0], a[1],
                                        bytearray(b[2]), b[0], b[1], scaling, "BF16"), "BF16"),
                  "BF16")


def test_stream_merge_writes_the_merged_checkpoint(dirs, tmp_path):
    base, adapter, _ = dirs
    out = tmp_path / "out"
    out.mkdir()
    info = SM.stream_merge(base, adapter, out, progress=lambda m: None)
    _, s1 = read_st(out / "model-00001-of-00002.safetensors")
    _, s2 = read_st(out / "model-00002-of-00002.safetensors")
    # mtp.* is left out, as transformers leaves it out on load
    assert sorted(s1) == sorted(["model.language_model.embed_tokens.weight", Q_W,
                                 "model.language_model.layers.0.linear_attn.norm.weight"])
    assert info["dropped"]["names"] == ["mtp.fc.weight"]
    # untouched tensors are the base's bytes; integers are never cast
    assert s1["model.language_model.embed_tokens.weight"] == EMBED
    assert s2["model.visual.pos_ids"] == ("I64", [2], POS[2])
    # an F32 tensor is cast to bf16 the way from_pretrained(dtype=bf16) casts it
    assert s1["model.language_model.layers.0.linear_attn.norm.weight"] == (
        "BF16", [2], encode(decode(NORM[2], "F32"), "BF16"))
    # the LoRA targets: W + bf16((B @ A) * alpha / r)
    aq = (adapter / "adapter_model.safetensors")
    _, at = read_st(aq, aligned=False)
    assert s1[Q_W] == ("BF16", [3, 2], expected(QW, at[f"{PFX}{Q_W[:-7]}.lora_A.weight"],
                                                at[f"{PFX}{Q_W[:-7]}.lora_B.weight"], 2.0))
    assert s1[Q_W][2] != QW[2]
    assert s2[V_W][0] == "BF16" and s2[V_W][2] != VW[2]
    assert {c[1] for c in PyMath.calls} == {2.0}
    index = json.loads((out / SM.INDEX).read_text())
    assert index["weight_map"][V_W] == "model-00002-of-00002.safetensors"
    assert "mtp.fc.weight" not in index["weight_map"]
    assert index["metadata"]["total_size"] == sum(len(v[2]) for v in {**s1, **s2}.values())
    assert info["tensors_merged"] == 2 and info["tensors_cast"] == 1
    assert info["tensors_copied"] == 2 and info["tensors_cast_from"] == ["F32"]
    assert info["lora_modules"] == info["attached"] == info["b_nonzero"] == 2
    assert not list(out.glob("*.partial"))


def test_rslora_scales_by_the_root_of_the_rank(dirs, tmp_path):
    base, adapter, _ = dirs
    write_adapter(adapter, {"use_rslora": True, "lora_alpha": 4})
    assert SM.make_plan(base, adapter).scaling == 4.0  # 4 / sqrt(1)
    assert SM.scaling_of({"lora_alpha": 32, "use_rslora": True}, 16) == 8.0
    assert SM.scaling_of({"lora_alpha": 32}, 16) == 2.0


@pytest.mark.parametrize("cfg, why", [
    ({"use_dora": True}, "use_dora"),
    ({"fan_in_fan_out": True}, "fan_in_fan_out"),
    ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
    ({"rank_pattern": {"q_proj": 4}}, "rank_pattern"),
    ({"peft_type": "IA3"}, "not LoRA"),
    ({"r": 8}, "rank 1"),
])
def test_refuses_what_it_does_not_reproduce(dirs, tmp_path, cfg, why):
    base, adapter, _ = dirs
    write_adapter(adapter, cfg)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SM.MergeError, match=why):
        SM.stream_merge(base, adapter, out, progress=lambda m: None)
    assert not list(out.iterdir())


def _drop_v(t):
    return {k: v for k, v in t.items() if "v_proj" not in k or "lora_A" in k}


def _rename(t):
    return {k.replace("layers.0", "layers.9"): v for k, v in t.items()}


def _bad_prefix(t):
    return {k.replace(PFX, ""): v for k, v in t.items()}


def _wrong_shape(t):
    t = dict(t)
    t[f"{PFX}{V_W[:-7]}.lora_A.weight"] = ("F32", [1, 3], encode([1.0, 1.0, 1.0], "F32"))
    return t


def _zero_b(t):
    t = dict(t)
    t[f"{PFX}{V_W[:-7]}.lora_B.weight"] = ("F32", [2, 1], encode([0.0, 0.0], "F32"))
    return t


def _embedding(t):
    return {**t, f"{PFX}model.language_model.embed_tokens.lora_embedding_A":
            ("F32", [1, 3], encode([1.0, 1.0, 1.0], "F32"))}


@pytest.mark.parametrize("edit, why", [
    (_drop_v, "lora_A only"),
    (_rename, "would not attach"),
    (_bad_prefix, "expected under"),
    (_wrong_shape, "would not attach"),
    (_zero_b, "B matrices are zero"),
    (_embedding, "embedding LoRA"),
])
def test_refuses_an_adapter_that_would_merge_silently_wrong(dirs, tmp_path, edit, why):
    base, adapter, _ = dirs
    write_adapter(adapter, tensors=edit)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SM.MergeError, match=why):
        SM.stream_merge(base, adapter, out, progress=lambda m: None)
    assert not list(out.iterdir())


def test_refuses_a_quantized_base(dirs, tmp_path):
    base, adapter, _ = dirs
    (base / "config.json").write_text(json.dumps({"quantization_config": {"bits": 4}}))
    with pytest.raises(SM.MergeError, match="quantized"):
        SM.make_plan(base, adapter)


def test_a_non_finite_merge_is_refused_and_leaves_no_shard(dirs, tmp_path):
    base, adapter, _ = dirs

    def huge(t):
        t = dict(t)
        t[f"{PFX}{Q_W[:-7]}.lora_B.weight"] = ("F32", [3, 1], encode([3e38, 3e38, 3e38], "F32"))
        t[f"{PFX}{Q_W[:-7]}.lora_A.weight"] = ("F32", [1, 2], encode([3e38, 3e38], "F32"))
        return t

    write_adapter(adapter, tensors=huge)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SM.MergeError, match="non-finite"):
        SM.stream_merge(base, adapter, out, progress=lambda m: None)
    assert not list(out.iterdir())


# ── quantize.py --merge-mode stream ─────────────────────────────────────────────


def test_quantize_stream_merge_makes_the_checkpoint_like_the_peft_path(dirs, tmp_path):
    base, adapter, tdir = dirs
    out = tmp_path / "phase6" / "merged"
    out.parent.mkdir()
    info = Q.merge_adapter_stream(base, adapter, out, template_dir=tdir)
    assert info["merge_mode"] == "stream" and info["lora_modules"] == 2
    assert info["adapter_config"]["r"] == 1
    assert info["template"]["sha256"] == Q.sha256_text("TRAINED {{ messages }}")
    assert (out / "chat_template.jinja").read_text() == "TRAINED {{ messages }}"
    assert json.loads((out / "tokenizer_config.json").read_text())["chat_template"] == \
        "TRAINED {{ messages }}"
    assert json.loads((out / "config.json").read_text())["keep"] == "the base's own"
    assert json.loads((out / "merge-info.json").read_text())["merge_mode"] == "stream"
    assert not out.with_name("merged.partial").exists()


def test_quantize_stream_merge_refusal_is_a_quantize_error(dirs, tmp_path):
    base, adapter, tdir = dirs
    write_adapter(adapter, {"use_dora": True})
    with pytest.raises(Q.QuantizeError, match="stream merge"):
        Q.merge_adapter_stream(base, adapter, tmp_path / "merged", template_dir=tdir)
    assert not (tmp_path / "merged").exists()


def test_merge_mode_is_in_the_fingerprint_only_when_stream(dirs, tmp_path):
    base, adapter, tdir = dirs
    s = Q.Settings(out=tmp_path, data=tmp_path, evalsets=None, extra_eval=[],
                   injection_manifest=None, seed=1, calib_tokens=1, imatrix_ctx=1, ppl_ctx=1,
                   ppl_chunks=1, base=base, adapter=adapter, template_dir=tdir,
                   llama=Q.LlamaCpp(tmp_path, "9d57ce456", tmp_path))
    assert "merge_mode" not in s.fingerprint()
    s.merge_mode = "stream"
    assert s.fingerprint()["merge_mode"] == "stream"


# ── the arithmetic against PEFT's own (only where torch and peft import) ────────


def test_torch_math_is_bit_identical_to_peft_merge(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    import peft
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(7)
    w = torch.randn(48, 32, generator=g).to(torch.bfloat16)
    a = torch.randn(4, 32, generator=g) * 0.01
    b = torch.randn(48, 4, generator=g) * 0.01

    class One(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(32, 48, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.lin.weight.copy_(w)

    (tmp_path / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 4, "lora_alpha": 8, "target_modules": ["lin"]}))
    save_file({f"{PFX}lin.lora_A.weight": a, f"{PFX}lin.lora_B.weight": b},
              str(tmp_path / "adapter_model.safetensors"))
    ref = peft.PeftModel.from_pretrained(One(), str(tmp_path)).merge_and_unload(
        safe_merge=True).lin.weight.detach()

    def raw(t):
        return bytearray(t.contiguous().view(torch.uint8).numpy().tobytes())

    m = SM.TorchMath()
    got = torch.frombuffer(bytearray(m.merge(raw(w), "BF16", [48, 32], raw(a), "F32", [4, 32],
                                             raw(b), "F32", [48, 4], 2.0, "BF16")),
                           dtype=torch.bfloat16).reshape(48, 32)
    assert torch.equal(got, ref)


# ── quantize.py merge (the merge alone, for a run on another host) ─────────────


def _merge_argv(base, adapter, tdir, out, *extra):
    return ["merge", "--adapter", str(adapter), "--base", str(base), "--template-dir", str(tdir),
            "--out", str(out), *extra]


def test_merge_subcommand_plans_then_merges_then_keeps(dirs, tmp_path, capsys):
    base, adapter, tdir = dirs  # the adapter's dir carries the tag: Max-derived
    out = tmp_path / "phase6" / "run-max-derived" / "merged"
    assert Q.main(_merge_argv(base, adapter, tdir, out)) == 0
    assert "DRY RUN" in capsys.readouterr().out and not out.exists()
    assert Q.main(_merge_argv(base, adapter, tdir, out, "--execute")) == 0
    info = json.loads((out / "merge-info.json").read_text())
    assert info["merge_mode"] == "stream" and info["attached"] == 2
    assert info["max_derived"] is True and info["private"] is True and info["tags"] == ["max-derived"]
    files = info["adapter"]["files"]
    assert files["adapter_model.safetensors"]["sha256"] == Q.sha256_file(
        adapter / "adapter_model.safetensors")
    assert info["base"]["files"]["config.json"]["sha256"] == Q.sha256_file(base / "config.json")
    capsys.readouterr()
    assert Q.main(_merge_argv(base, adapter, tdir, out, "--execute")) == 0
    assert "kept" in capsys.readouterr().out


def test_merge_subcommand_refuses_an_untagged_out_for_a_max_derived_adapter(dirs, tmp_path,
                                                                           capsys):
    base, adapter, tdir = dirs
    out = tmp_path / "phase6" / "run" / "merged"
    assert Q.main(_merge_argv(base, adapter, tdir, out, "--execute")) == 2
    assert "max-derived" in capsys.readouterr().err and not out.exists()


def test_merge_subcommand_refuses_an_out_holding_another_merge(dirs, tmp_path, capsys):
    base, adapter, tdir = dirs
    out = tmp_path / "phase6" / "run-max-derived" / "merged"
    out.mkdir(parents=True)
    (out / "merge-info.json").write_text(json.dumps({"adapter": {"files": {"x": 1}}}))
    assert Q.main(_merge_argv(base, adapter, tdir, out, "--execute")) == 2
    assert "not a merge of this adapter" in capsys.readouterr().err


def test_a_merged_base_makes_a_no_adapter_run_max_derived(tmp_path):
    clean = {"calibration": {"rows": 3}, "ppl_slice": {"rows": 1},
             "provenance": {"calibration_max_derived": 0, "ppl_max_derived": 0}}
    plain = tmp_path / "models" / "qwen3.5-9b"
    plain.mkdir(parents=True)
    assert Q.output_provenance(None, clean, plain)["max_derived"] is False
    merged = tmp_path / "phase6" / "run" / "merged"
    merged.mkdir(parents=True)
    (merged / "merge-info.json").write_text(json.dumps({"merge_mode": "stream"}))
    prov = Q.output_provenance(None, clean, merged)
    assert prov["max_derived"] is True and "merged checkpoint" in prov["evidence"][0]
    (merged / "merge-info.json").write_text(json.dumps({"max_derived": False}))
    assert Q.output_provenance(None, clean, merged)["max_derived"] is False
    tagged = tmp_path / "phase6" / "run-max-derived" / "merged"
    tagged.mkdir(parents=True)
    assert Q.output_provenance(None, clean, tagged)["max_derived"] is True
