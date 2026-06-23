"""Tests for scripts/run_model_ab_eval.py — the Gemma vs Qwen A/B orchestrator.

These exercise the PURE functions (recipe derivation, props sanity, quant
parse, report building) and the ProductionGuard restore-on-exit EFFECT. No GPU,
no network, no production impact — safe to run anywhere. Assertions check real
output / observable effects, not mock call counts.

Fixtures deliberately use neutral paths (/opt, /models, localhost) so the
committed file carries no infra hostnames/IPs (pre-commit hook stays green).
"""

from __future__ import annotations

import json
import shlex
import sys
import types
from pathlib import Path

import pytest

# Allow direct import from scripts/ (same pattern as test_vibe_check.py)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_model_ab_eval as ab  # noqa: E402


# A realistic multi-flag launcher (single line is fine — derive_recipes takes a
# string; read_live_execstart is what handles the multi-line unit file).
GEMMA_EXEC = (
    "/opt/llama.cpp/build/bin/llama-server "
    "-m /models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf "
    "--mmproj /models/mmproj-BF16.gguf "
    "--ubatch-size 2048 --batch-size 2048 "
    "--port 8080 --host 0.0.0.0 "
    "-ngl 99 -c 81920 --parallel 1 --flash-attn on --jinja "
    "--reasoning-budget 2048"
)


# ── 1. derive_recipes: Gemma verbatim; Qwen swaps model, drops mmproj, KEEPS jinja/reasoning ──
def test_derive_recipes_core_correctness_and_fairness():
    recipes = ab.derive_recipes(GEMMA_EXEC)
    gemma, qwen = recipes["gemma"], recipes["qwen"]

    # Gemma == the canonical launcher, verbatim
    assert gemma == shlex.split(GEMMA_EXEC)

    # Qwen has the Qwen GGUF at the -m position (not the Gemma gguf)
    m_idx = qwen.index("-m")
    qwen_model = qwen[m_idx + 1]
    assert Path(qwen_model).name == Path(ab.QWEN_GGUF).name
    assert "Qwen" in Path(qwen_model).name
    assert "gemma" not in qwen_model.lower()

    # Qwen has NO vision projector — neither flag form
    assert "--mmproj" not in qwen
    assert not any(t.startswith("--mmproj") for t in qwen)

    # Fairness: shared knobs are INHERITED, not dropped
    assert "--jinja" in qwen
    assert "--reasoning-budget" in qwen
    assert "2048" in qwen  # the reasoning-budget value rode along
    assert "-c" in qwen and "81920" in qwen


# ── 2. derive_recipes: inline --mmproj=PATH form is also dropped ──
def test_derive_recipes_drops_inline_mmproj():
    exec_inline = (
        "/opt/llama.cpp/build/bin/llama-server "
        "-m /models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf "
        "--mmproj=/models/mmproj-BF16.gguf "
        "--jinja --reasoning-budget 2048"
    )
    qwen = ab.derive_recipes(exec_inline)["qwen"]
    assert not any(t.startswith("--mmproj") for t in qwen)
    # the projector path itself must not survive either
    assert not any("mmproj" in t for t in qwen)
    # but the shared knobs still survive
    assert "--jinja" in qwen and "--reasoning-budget" in qwen


# ── 3. props_sanity: three-state (problem names the fix; missing -> UNKNOWN, never clean) ──
def test_props_sanity_textonly_with_vision_says_drop_mmproj():
    props = {
        "chat_template_caps": {"supports_tool_calls": True},
        "modalities": {"vision": True, "audio": False},
    }
    problems = ab.props_sanity("qwen", props, "text-only")
    assert any("drop --mmproj" in p for p in problems)


def test_props_sanity_missing_tool_calls_names_jinja():
    props = {
        "chat_template_caps": {"supports_tool_calls": False},
        "modalities": {"vision": False},
    }
    problems = ab.props_sanity("gemma", props, "text-only")
    assert any("--jinja" in p for p in problems)


def test_props_sanity_empty_props_is_unknown_not_clean():
    problems = ab.props_sanity("gemma", {}, "vision")
    # exactly the UNKNOWN result — NOT an empty (clean) list
    assert len(problems) == 1
    assert "UNKNOWN" in problems[0]
    assert problems != []


# ── 4. gguf_quant: parsed from the served gguf filename ──
def test_gguf_quant_gemma_and_qwen():
    gemma_recipe = ["llama-server", "-m", "/models/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"]
    qwen_recipe = ["llama-server", "-m", "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"]
    assert ab.gguf_quant(gemma_recipe) == "Q4_K_M"
    assert ab.gguf_quant(qwen_recipe) == "UD-Q4_K_XL"


# ── 5. build_report: hard drift -> validity WARNING; quant-only -> soft note ──
def _write_results(tmp_path: Path, name: str) -> str:
    """A minimal but schema-real results file (top-level 'results', metrics[])."""
    data = {
        "results": [
            {
                "task_id": "t1",
                "error": None,
                "metrics": [
                    {"metric_name": "Tool Usage", "score": 1.0, "passed": True},
                    {"metric_name": "Task Completion", "score": 0.9, "passed": True},
                    {"metric_name": "No Hallucination", "score": 1.0, "passed": True},
                ],
            }
        ]
    }
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def _manifest(sha: str, quant: str, results_file: str,
              judge_base: str = "http://localhost:11434",
              judge_model: str = "llama3.1:8b") -> dict:
    return {
        "harness_git_sha": sha,
        "judge_base_url": judge_base,
        "judge_model": judge_model,
        "quant": quant,
        "results_file": results_file,
        "smoke_gate_passed": True,
        "served_command": "/opt/llama.cpp/build/bin/llama-server -m /models/x.gguf",
        "gguf_sha256_12": "abc123def456",
    }


def test_build_report_differing_harness_sha_emits_validity_warning(tmp_path):
    manifests = {
        "gemma": _manifest("sha_AAA", "Q4_K_M", _write_results(tmp_path, "g.json")),
        "qwen": _manifest("sha_BBB", "Q4_K_M", _write_results(tmp_path, "q.json")),
    }
    report = ab.build_report(tmp_path, manifests)
    text = report.read_text()
    assert "A/B VALIDITY WARNING" in text
    assert "harness git SHA differs" in text


def test_build_report_quant_diff_is_soft_note_not_hard_warning(tmp_path):
    # same SHA, same judge, only quant differs (the real-deploy case)
    manifests = {
        "gemma": _manifest("sha_SAME", "Q4_K_M", _write_results(tmp_path, "g.json")),
        "qwen": _manifest("sha_SAME", "UD-Q4_K_XL", _write_results(tmp_path, "q.json")),
    }
    report = ab.build_report(tmp_path, manifests)
    text = report.read_text()
    assert "quant differs" in text
    assert "EXPECTED for a real-deploy" in text
    # crucially NOT escalated to a hard validity warning
    assert "A/B VALIDITY WARNING" not in text


# ── 6. ProductionGuard: an exception inside the `with` still restores BOTH units ──
def test_production_guard_restores_both_units_on_exception(monkeypatch):
    recorded: list[tuple[str, str]] = []

    def fake_ssh(cmd, check=True, capture=True):
        recorded.append(("ssh", cmd))
        return types.SimpleNamespace(stdout="", returncode=0)

    def fake_sh(cmd, check=True, capture=True):
        recorded.append(("sh", cmd))
        return types.SimpleNamespace(stdout="", returncode=0)

    # Recording fakes for the side-effecting calls; no real ssh/systemctl/sleep.
    monkeypatch.setattr(ab, "ssh", fake_ssh)
    monkeypatch.setattr(ab, "sh", fake_sh)
    monkeypatch.setattr(ab, "LLAMA_BASE", None)          # _wait_port_free returns fast
    monkeypatch.setattr(ab.time, "sleep", lambda *a, **k: None)

    guard = ab.ProductionGuard(manual=False)

    def boom(*a, **k):
        raise RuntimeError("serve failed")

    monkeypatch.setattr(guard, "serve", boom)

    with pytest.raises(RuntimeError):
        with guard:
            guard.serve(["llama-server"], "gemma")

    cmds = [c for _, c in recorded]
    # EFFECT: both units were stopped on entry...
    assert any(f"stop {ab.PROMETHEUS_UNIT}" in c for c in cmds)
    assert any(f"stop {ab.LLAMA_UNIT}" in c for c in cmds)
    # ...and crucially RESTORED on exit despite the exception
    assert any(f"start {ab.LLAMA_UNIT}" in c for c in cmds)
    assert any(f"start {ab.PROMETHEUS_UNIT}" in c for c in cmds)


# ── 7. ctx override: Qwen carries the overridden -c AND the deviation is recorded ──
def test_apply_ctx_override_qwen_carries_value_and_records_deviation():
    recipe = ["/opt/llama.cpp/build/bin/llama-server",
              "-m", "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
              "-c", "81920", "--jinja", "--reasoning-budget", "2048"]
    new, dev = ab.apply_ctx_override(recipe, 32768)

    # the overridden -c value rides in the recipe; the canonical value is gone
    assert new[new.index("-c") + 1] == "32768"
    assert "81920" not in new
    # untouched flags survive
    assert "--jinja" in new and "--reasoning-budget" in new
    # the deviation is a documented "VRAM fit" record for the manifest
    assert dev == {"flag": "-c", "canonical": 81920, "applied": 32768,
                   "reason": "VRAM fit"}


def test_apply_ctx_override_none_inherits_canonical():
    recipe = ["llama-server", "-c", "81920"]
    new, dev = ab.apply_ctx_override(recipe, None)
    assert new == recipe
    assert dev is None


def test_apply_ctx_override_equal_value_is_not_a_deviation():
    recipe = ["llama-server", "-c", "32768"]
    new, dev = ab.apply_ctx_override(recipe, 32768)
    assert new[new.index("-c") + 1] == "32768"
    assert dev is None


def test_apply_ctx_override_inline_form():
    recipe = ["llama-server", "-c=81920", "--jinja"]
    new, dev = ab.apply_ctx_override(recipe, 32768)
    assert "-c=32768" in new
    assert "-c=81920" not in new
    assert dev["canonical"] == 81920 and dev["applied"] == 32768


def test_manifest_records_ctx_deviation(tmp_path, monkeypatch):
    # hermetic: no ssh / git
    monkeypatch.setattr(ab, "remote_gguf_sha", lambda recipe: "deadbeefcafe")
    monkeypatch.setattr(ab, "git_sha", lambda: "testsha123")

    recipe = ["llama-server", "-m", "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "-c", "32768"]
    dev = {"flag": "-c", "canonical": 81920, "applied": 32768, "reason": "VRAM fit"}

    manifest = ab.write_manifest(
        "qwen", recipe, props={}, smoke_ok=True, scored_ok=True,
        results_path=None, eval_config=tmp_path / "eval_config.yaml",
        run_dir=tmp_path, ctx_deviation=dev,
    )
    # returned dict carries it...
    assert manifest["ctx_deviation"] == dev
    # ...and so does the persisted manifest file (the audit artifact)
    written = json.loads((tmp_path / "qwen_manifest.json").read_text())
    assert written["ctx_deviation"] == dev


def test_build_report_surfaces_ctx_deviation_in_audit(tmp_path):
    m_g = _manifest("sha_SAME", "Q4_K_M", _write_results(tmp_path, "g.json"))
    m_q = _manifest("sha_SAME", "UD-Q4_K_XL", _write_results(tmp_path, "q.json"))
    m_q["ctx_deviation"] = {"flag": "-c", "canonical": 81920,
                            "applied": 32768, "reason": "VRAM fit"}
    report = ab.build_report(tmp_path, {"gemma": m_g, "qwen": m_q})
    text = report.read_text()
    assert "ctx deviation" in text
    assert "81920" in text and "32768" in text
    assert "VRAM fit" in text


# ── 8. (Amendment 2) judge identity + weak-discriminator caveat on the artifact ──
def test_build_report_header_names_judge_with_weakness_caveat(tmp_path):
    manifests = {
        "gemma": _manifest("sha_SAME", "Q4_K_M", _write_results(tmp_path, "g.json"),
                           judge_base="http://localhost:11434", judge_model="llama3.1:8b"),
        "qwen": _manifest("sha_SAME", "Q4_K_M", _write_results(tmp_path, "q.json"),
                          judge_base="http://localhost:11434", judge_model="llama3.1:8b"),
    }
    report = ab.build_report(tmp_path, manifests)
    text = report.read_text()
    # judge identity is ON THE ARTIFACT, not just in chat
    assert "llama3.1:8b" in text
    assert "localhost:11434" in text
    # ...with the discriminator-strength caveat
    assert "weaker discriminator" in text
    assert "independent of BOTH contestants" in text
