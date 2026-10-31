"""quantize.py for real: a tiny Qwen3.5 through the whole Phase 6 chain, CPU only.

The real-run tests are skipped unless ``DISTILL_QUANTIZE_SMOKE_DIR`` names a
directory that ``scripts/distill/quantize_smoke.py prepare`` filled (the
pinned llama.cpp built CPU-only, the converter and merge venvs, Qwen3.5-0.8B,
a synthetic Phase 3 build and random LoRAs). Nothing is downloaded here and
nothing runs on a GPU. On an M2 Pro they take ~15 minutes, most of it the BF16
reference on the CPU; ``DISTILL_QUANTIZE_SMOKE_DTYPE=fp16`` makes that run
fast. Outputs go to ``<smoke dir>/pytest-run``; the big files are removed when
the module's tests finish (manifests and logs stay for a look).

    DISTILL_QUANTIZE_SMOKE_DIR=/scratch/qsmoke uv run --frozen pytest \\
        tests/test_distill_quantize_smoke.py -q -p no:cacheprovider

The harness checks at the bottom always run (stdlib only, no smoke dir).
"""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SMOKE_ENV = "DISTILL_QUANTIZE_SMOKE_DIR"
SMOKE = os.environ.get(SMOKE_ENV)
DTYPE = os.environ.get("DISTILL_QUANTIZE_SMOKE_DTYPE", "bf16")

needs_smoke = pytest.mark.skipif(
    not SMOKE, reason=f"{SMOKE_ENV} not set (a dir made by quantize_smoke.py prepare)")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "distill" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("distill_quantize_smoke", "quantize_smoke.py")
Q = sys.modules.get("distill_quantize") or _load("distill_quantize", "quantize.py")


def _env():
    return {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}


def quantize(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=_env(), check=False)


@pytest.fixture(scope="module")
def layout():
    L = S.Layout(Path(SMOKE))
    missing = L.missing()
    if missing:
        pytest.fail(f"{SMOKE} is not a prepared smoke dir; missing: {missing[:4]}")
    root = L.root / "pytest-run"
    if root.exists():
        shutil.rmtree(root)
    yield L
    # Several GB per run: keep what explains a run, drop what it made.
    for out in root.glob("*"):
        drop_outputs(out)


def drop_outputs(out):
    for d in ("merged", "base-view", "gguf"):
        shutil.rmtree(out / d, ignore_errors=True)


def run_variant(layout, variant, dtype):
    """One more real run; its big outputs go as soon as it is read."""
    out = layout.root / "pytest-run" / S.VARIANTS[variant][1]
    cp = quantize(S.quantize_argv(layout, out, dtype=dtype, variant=variant))
    try:
        assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
        return json.loads((out / "quantize-manifest.json").read_text())
    finally:
        drop_outputs(out)


@pytest.fixture(scope="module")
def e2e(layout):
    """The proof run (the wrapper-shaped adapter), stopped by SIGTERM as soon
    as llama-imatrix starts (as a window's end stops it), then resumed with the
    same command: the pause, the resume and the whole chain, all for real."""
    out = layout.root / "pytest-run" / S.RUN_LABEL
    cmd = S.quantize_argv(layout, out, dtype=DTYPE)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=_env())
    assert p.stdout is not None
    seen = []
    for line in p.stdout:
        seen.append(line)
        if line.startswith("  $ ") and "llama-imatrix" in line:
            p.send_signal(signal.SIGTERM)
            break
    rest, _ = p.communicate(timeout=300)
    paused_out = "".join(seen) + (rest or "")
    paused = json.loads((out / "quantize-manifest.json").read_text())
    cp = quantize(cmd)
    assert cp.returncode == 0, cp.stdout[-3000:] + cp.stderr[-3000:]
    manifest = json.loads((out / "quantize-manifest.json").read_text())
    return {"out": out, "manifest": manifest, "stdout": cp.stdout, "paused_rc": p.returncode,
            "paused_out": paused_out, "paused": paused}


@needs_smoke
def test_a_build_whose_train_row_is_in_an_eval_task_is_refused_before_the_merge(layout):
    out = layout.root / "pytest-run" / "leak"
    cp = quantize(S.quantize_argv(layout, out, data=layout.leak))
    assert cp.returncode == 2, cp.stdout[-2000:] + cp.stderr[-2000:]
    assert (f"{S.LEAK_ROW}: task_id smoke-etask-000 is in eval-tools.jsonl"
            in cp.stderr), cp.stderr
    assert not (out / "merged").exists() and not (out / "gguf").exists()


@needs_smoke
def test_sigterm_pauses_the_run_and_the_same_command_resumes_it(e2e):
    assert e2e["paused_rc"] == Q.EXIT_PAUSED, e2e["paused_out"][-3000:]
    assert "PAUSED (SIGTERM)" in e2e["paused_out"]
    assert e2e["paused"]["result"] == "PAUSED"
    assert set(e2e["paused"]["steps"]) == {"calibrate", "merge", "convert"}
    # The finished steps were kept; the killed imatrix was redone.
    assert "[3/8] merged: kept (already made)" in e2e["stdout"]
    assert f"kept {S.RUN_LABEL}-{'BF16' if DTYPE == 'bf16' else 'F16'}.gguf" in e2e["stdout"]
    assert "$ " in e2e["stdout"].split("[5/8] imatrix", 1)[1].split("[6/8]", 1)[0]


def check_manifest(m, label, ref):
    assert m["result"] == "PASS"
    cal = m["steps"]["calibrate"]
    assert cal["source"] == "train split only"
    assert cal["eval_check"]["train_rows_sharing_a_key"] == 0
    # The planted eval-injection row is left out, and nothing else.
    assert cal["eval_check"]["dropped_injection_overlap"] == 1
    assert m["calibration_provenance"]["calibration_max_derived"] == 0
    assert m["max_derived"] is False and m["tags"] == []      # clean data: untagged names
    names = {a["name"]: a for a in m["artifacts"]}
    sizes = {q: names[f"{label}-{q}.gguf"]["bytes"] for q in (ref, "Q4_K_M", "Q5_K_M")}
    assert sizes["Q4_K_M"] < sizes["Q5_K_M"] < sizes[ref]
    assert f"{label}.imatrix.gguf" in names
    for name, r in m["sanity"]["models"].items():
        assert r["pass"] and all(r["checks"].values()), (name, r["checks"])
        assert r["metadata"]["architecture"] == "qwen35"
        assert not r["metadata"]["nextn_predict_layers"]
        assert r["metadata"]["context_length"] >= Q.SERVE_CTX
    for q in ("Q4_K_M", "Q5_K_M"):
        assert m["sanity"]["models"][q]["ratio_to_ref"] <= m["sanity"]["models"][q]["max_ratio"]
        assert "-fa on" in m["serve"][q] and "--jinja" in m["serve"][q]


@needs_smoke
def test_the_pipeline_makes_checked_quants_from_the_train_split_only(e2e):
    m = e2e["manifest"]
    check_manifest(m, S.RUN_LABEL, "BF16" if DTYPE == "bf16" else "F16")
    assert m["steps"]["calibrate"]["train_split"]["matched"] == "the adapter's trainer manifest"
    merge = m["steps"]["merge"]
    assert merge["model_class"] == "AutoModelForImageTextToText"
    assert merge["attached"] == merge["lora_modules"] == merge["b_nonzero"] > 0


MERGE_CHECK = r"""
import json, sys
from safetensors import safe_open
def tensors(d):
    out = {}
    import pathlib
    for f in sorted(pathlib.Path(d).glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                if k.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight")):
                    out[k.replace("model.language_model.", "model.")] = fh.get_tensor(k)
    return out
b, m = tensors(sys.argv[1]), tensors(sys.argv[2])
print(json.dumps({k: bool((b[k].float() - m[k].float()).abs().max() > 0) for k in m if k in b}))
"""


@needs_smoke
def test_the_quants_carry_the_template_phase_3_rendered_with_not_the_bases(e2e, layout):
    import hashlib

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    trained, base = sha(layout.tokenizer / "chat_template.jinja"), sha(
        layout.model / "chat_template.jinja")
    assert trained != base
    for r in e2e["manifest"]["sanity"]["models"].values():
        assert r["metadata"]["chat_template_sha256"] == trained


@needs_smoke
def test_the_merge_changed_exactly_the_weights_the_adapter_targets(e2e, layout):
    cp = subprocess.run([str(S.Layout.py(layout.mvenv)), "-I", "-c", MERGE_CHECK,
                         str(layout.model), str(e2e["out"] / "merged")],
                        capture_output=True, text=True, check=False)
    assert cp.returncode == 0, cp.stderr[-2000:]
    diff = json.loads(cp.stdout.strip().splitlines()[-1])
    q = [v for k, v in diff.items() if "q_proj" in k]
    k = [v for k, v in diff.items() if "k_proj" in k]
    assert q and all(q), "a q_proj the adapter targets was not changed by the merge"
    assert k and not any(k), "a k_proj the adapter does not target changed"


@needs_smoke
def test_a_rerun_keeps_every_finished_step(e2e, layout):
    out, before = e2e["out"], {a["name"]: a["sha256"] for a in e2e["manifest"]["artifacts"]}
    cp = quantize(S.quantize_argv(layout, out, dtype=DTYPE))
    assert cp.returncode == 0, cp.stdout[-2000:] + cp.stderr[-2000:]
    assert "merged: kept" in cp.stdout
    for name in ("Q4_K_M", "Q5_K_M"):
        assert f"kept {S.RUN_LABEL}-{name}.gguf" in cp.stdout
    again = json.loads((out / "quantize-manifest.json").read_text())
    assert {a["name"]: a["sha256"] for a in again["artifacts"]} == before


@needs_smoke
def test_an_adapter_made_on_the_text_only_model_merges_through_the_causal_lm(layout):
    m = run_variant(layout, "text", "fp16")
    check_manifest(m, S.VARIANTS["text"][1], "F16")
    merge = m["steps"]["merge"]
    assert merge["model_class"] == "AutoModelForCausalLM"
    assert merge["attached"] == merge["lora_modules"] == merge["b_nonzero"] > 0


@needs_smoke
def test_the_base_goes_through_the_same_recipe_with_the_same_template(layout, e2e):
    m = run_variant(layout, "base", "fp16")
    check_manifest(m, S.VARIANTS["base"][1], "F16")
    tuned = e2e["manifest"]["sanity"]["models"]["Q4_K_M"]["metadata"]["chat_template_sha256"]
    assert m["sanity"]["models"]["Q4_K_M"]["metadata"]["chat_template_sha256"] == tuned


# ── the harness itself (always run: no smoke dir, no network) ─────────────


def test_the_smoke_runs_the_pinned_build_on_the_cpu_only(tmp_path):
    L = S.Layout(tmp_path / "smoke")
    argv = S.quantize_argv(L, tmp_path / "out")
    assert argv[:3] == [str(L.mvenv / "bin" / "python"), "-I", "-u"]
    assert argv[argv.index("--adapter") + 1] == str(L.adapter)
    assert argv[argv.index("--ngl") + 1] == "0"
    assert argv[argv.index("--convert-python") + 1] == str(L.qvenv / "bin" / "python")
    assert argv[argv.index("--dtype") + 1] == "bf16" and argv[-1] == "--execute"
    assert "--no-adapter" in S.quantize_argv(L, tmp_path / "out", variant="base")
    assert {"-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_BLAS=OFF"} <= set(S.CMAKE_FLAGS)
    assert S.LLAMA_CPP_SHA.startswith(Q.DEFAULT_COMMIT) and len(S.LLAMA_CPP_SHA) == 40


def test_the_smoke_never_writes_into_the_repo(capsys):
    assert S.main(["prepare", "--dir", str(REPO / "data" / "qsmoke")]) == 2
    assert "inside the repo" in capsys.readouterr().err


def a_distill_root(tmp_path):
    """A host's distill root: the names the smoke would make, holding real work."""
    root = tmp_path / "distill"
    for d in ("phase3", "phase3-leak", "runs/smoke-lora", "phase6/leak"):
        (root / d).mkdir(parents=True)
        (root / d / "keep.txt").write_text("real\n")
    (root / "phase3" / "report.json").write_text('{"schema": "distill-build-report/2"}\n')
    return root


@pytest.mark.parametrize("cmd", [["prepare"], ["run"], ["make-data"],
                                 ["make-adapter", "--variant", "wrapper"]])
def test_every_subcommand_refuses_a_dir_the_smoke_did_not_make(tmp_path, capsys, cmd):
    # prepare on a distill root once ran make-data there, which deleted the
    # real phase3/ and wrote invented rows in its place.
    root = a_distill_root(tmp_path)
    assert S.main([cmd[0], "--dir", str(root), *cmd[1:]]) == 2
    err = capsys.readouterr().err
    assert "not a quantize smoke dir" in err and S.MARKER in err, err
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("keep.txt")) == [
        "phase3-leak/keep.txt", "phase3/keep.txt", "phase6/leak/keep.txt",
        "runs/smoke-lora/keep.txt"]
    assert not (root / S.MARKER).exists()


@pytest.mark.parametrize("cmd", ["make-data", "make-adapter", "run", "prepare"])
def test_no_subcommand_writes_into_the_repo(capsys, cmd):
    assert S.main([cmd, "--dir", str(REPO / "data" / "qsmoke")]) == 2
    assert "inside the repo" in capsys.readouterr().err


def test_prepare_claims_only_a_new_or_empty_dir(tmp_path):
    new, empty = tmp_path / "new" / "qsmoke", tmp_path / "empty"
    empty.mkdir()
    for d in (new, empty):
        S.claim(d, "prepare")
        assert (d / S.MARKER).is_file()
        S.claim(d, "make-data")  # a dir it owns: every subcommand goes on
    (tmp_path / "file").write_text("x")
    with pytest.raises(S.SmokeError):
        S.claim(tmp_path / "file", "prepare")
    # A marked dir lacks nothing prepare makes but the rest.
    assert str(new / S.MARKER) not in S.Layout(new).missing()
    assert str(tmp_path / S.MARKER) in S.Layout(tmp_path).missing()


def test_make_data_replaces_only_a_smoke_build(tmp_path, capsys):
    # Even inside a dir the smoke owns, a phase3/ it did not write is kept.
    root = a_distill_root(tmp_path)
    (root / S.MARKER).write_text("x\n")
    assert S.main(["make-data", "--dir", str(root)]) == 2
    assert "not a smoke build" in capsys.readouterr().err
    assert (root / "phase3" / "keep.txt").exists() and (root / "phase3-leak" / "keep.txt").exists()
    assert S.smoke_build(root / "phase3") is False
    (root / "phase3" / "report.json").write_text('{"schema": "smoke"}\n')
    assert S.smoke_build(root / "phase3") is True


def test_run_on_an_unprepared_dir_says_what_is_missing(tmp_path, capsys):
    assert S.main(["run", "--dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "not prepared" in err and "convert_hf_to_gguf.py" in err


def test_the_planted_injection_row_is_the_first_task_the_calibration_takes():
    # So the smallest calibration budget still reaches it (and must drop it).
    rows = [{"task_id": f"smoke-task-{t:03d}"} for t in range(60)]
    first = S.first_task(rows)
    cands = [Q._Cand(f"r{i}", r["task_id"], "x", 1) for i, r in enumerate(rows)]
    cands.append(Q._Cand("planted", first, "x", 1))
    assert Q._round_robin(cands, S.SEED)[0].id == "planted"
