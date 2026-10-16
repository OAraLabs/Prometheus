"""Context-window resolution through the real daemon: which window, from where.

What the code reads today (e3c17877):

* llama.cpp: ``GET /props`` → ``default_generation_settings.n_ctx``, the SERVED
  per-slot window (providers/llama_cpp.py detect_context_size; the backend
  registry reads the same field). Never ``n_ctx_train``.
* Ollama: the backend probe reads ``POST /api/show`` →
  ``model_info.*.context_length``, the TRAINED length (providers/backends.py),
  and the provider talks to ``/v1/chat/completions`` with no ``num_ctx``, so
  Ollama serves its own default. Measured on this box: a 7b loaded at 32768
  (it happens to equal the trained 32768); the 4090's Ollama reports 262,144
  for qwen3.5:9b, which it would not serve unasked. An over-window prompt is
  cut from the FRONT with no error (tests/support/fake_ollama.py).
* Overrides (``context.model_overrides``) match the model id EXACTLY. The 27B's
  id is its full GGUF path; a renamed or moved GGUF silently stops matching.
* Reply room: the request's ``max_tokens`` is the per-model output cap
  (providers/output_caps.py), and that cap is what the compactor's projected
  rule counts. ``context.reserved_output`` (2,000 live) feeds only the
  ``/context`` budget and media sizing; ``compaction.reserve_tokens`` (4,096)
  only a call with no cap. Three numbers for one quantity.
* A cloud model with no override gets ``cloud_default_limit`` (1,000,000), as a
  configured floor; 23 of the operator's reachable cloud models do.

Checks that 0.9.12 fails are ``xfail(strict=True)``.
"""

from __future__ import annotations

import pytest

from tests.support.fake_ollama import FakeOllama
from tests.support.gate_daemon import CLOUD_KEY, build_and_boot
from tests.support.scripted_llama import MODEL_ID, ScriptedLlamaServer
from tests.support.served_template import recorded_props

pytestmark = pytest.mark.integration

STALE_OVERRIDE = "/models/Qwen3.8-27B-OLD-NAME.gguf"
OLLAMA_BACKEND = "slow_ollama"


@pytest.fixture(scope="module")
def main(tmp_path_factory):
    """The served 27B, an override for it, a stale override, an Ollama backend, a cloud route."""
    tmp = tmp_path_factory.mktemp("cw-main")
    ws = str(tmp / "workspace")
    with ScriptedLlamaServer(props=recorded_props(), workspace=ws, extra_props={"n_ctx_train": 262_144}) as model, \
            ScriptedLlamaServer(n_ctx=1_000_000, native_tool_grammar=False, workspace=ws) as cloud, \
            FakeOllama() as ollama:
        daemon = build_and_boot(tmp, model.url, cloud_url=cloud.url, extra={
            "context.model_overrides": {
                MODEL_ID: {"max_output_tokens": 3000},
                STALE_OVERRIDE: {"effective_limit": 20000},
            },
            f"backends.{OLLAMA_BACKEND}": {"provider": "ollama", "base_url": ollama.url, "model": ollama.model},
        })
        try:
            yield daemon, model, cloud, ollama
        finally:
            daemon.stop()


def _boot_with(tmp_path_factory, name: str, **server_kw):
    tmp = tmp_path_factory.mktemp(name)
    server = ScriptedLlamaServer(props=recorded_props(), workspace=str(tmp / "workspace"), **server_kw)
    server.__enter__()
    return tmp, server


# ---------------------------------------------------------------- llama.cpp


def test_cw_llamacpp_budgets_the_served_n_ctx_not_n_ctx_train(main):
    daemon, *_ = main
    ctx = daemon.get("/api/status")["context"]
    assert ctx["detected_n_ctx"] == 32768 and ctx["limit"] == 32768, ctx
    assert ctx["source"] == "detected", ctx


def test_cw_a_per_slot_window_is_budgeted_per_slot(tmp_path_factory):
    # -c 32768 --parallel 2 serves 16384 per slot: /props reports the slot's n_ctx.
    tmp, server = _boot_with(tmp_path_factory, "cw-slot", n_ctx=16384)
    try:
        daemon = build_and_boot(tmp, server.url)
        try:
            assert daemon.get("/api/status")["context"]["limit"] == 16384
        finally:
            daemon.stop()
    finally:
        server.__exit__()


def test_cw_a_fallback_window_is_loud(tmp_path_factory):
    tmp, server = _boot_with(tmp_path_factory, "cw-fallback", report_n_ctx=False)
    try:
        daemon = build_and_boot(tmp, server.url)
        try:
            ctx = daemon.get("/api/status")["context"]
            assert ctx["source"] != "detected", f"precondition: detection should have failed: {ctx}"
            log = daemon.log.read_text(encoding="utf-8", errors="replace")
            assert any("WARNING" in line and "context" in line.lower() and str(ctx["limit"]) in line
                       for line in log.splitlines()), f"no WARNING names the fallback window {ctx}"
        finally:
            daemon.stop()
    finally:
        server.__exit__()


# ---------------------------------------------------------------- overrides


def test_cw_an_override_keyed_by_the_served_id_applies(main):
    daemon, model, *_ = main
    daemon.turn("[gate:hello] Hi.")
    sent = [e.request.get("max_tokens") for e in model.of("hello")]
    assert sent and all(m == 3000 for m in sent), sent


def test_cw_an_override_matching_no_served_model_is_reported(main):
    daemon, *_ = main
    log = daemon.log.read_text(encoding="utf-8", errors="replace")
    assert any("WARNING" in line and STALE_OVERRIDE in line for line in log.splitlines()), (
        f"no startup WARNING names the override {STALE_OVERRIDE!r}, which no backend serves")


def test_cw_renaming_the_gguf_keeps_its_override_or_says_why(tmp_path_factory):
    tmp, server = _boot_with(tmp_path_factory, "cw-rename", model_id="/models/Qwen3.8-27B-UD-Q4_K_XL-v2.gguf")
    try:
        daemon = build_and_boot(tmp, server.url, extra={"context.model_overrides": {MODEL_ID: {"max_output_tokens": 3000}}})
        try:
            daemon.turn("[gate:hello] Hi.")
            sent = [e.request.get("max_tokens") for e in server.of("hello")]
            log = daemon.log.read_text(encoding="utf-8", errors="replace")
            warned = any("WARNING" in line and MODEL_ID in line for line in log.splitlines())
            assert sent and (all(m == 3000 for m in sent) or warned), (
                f"after the rename the request asked for max_tokens {sent} (the override said 3000) "
                "and no WARNING named the override")
        finally:
            daemon.stop()
    finally:
        server.__exit__()


# ---------------------------------------------------------------- reply room


def test_cw_one_reply_room_everywhere(main):
    daemon, model, *_ = main
    sid = "beacon:cw-context"
    daemon.turn("[gate:hello] Hi.", sid=sid)
    asked = next(e.request.get("max_tokens") for e in model.of("hello"))
    report = daemon.turn("/context", sid=sid)
    text = report.streamed + " ".join(p.get("content", "") for p in report.of("chat_message"))
    assert f"{asked:,}" in text or str(asked) in text, (
        f"the request asks for {asked} reply tokens; /context reports: {text[:400]!r}")


# ---------------------------------------------------------------- Ollama


@pytest.mark.xfail(strict=True, reason="0.9.12: an Ollama backend is budgeted at its TRAINED length (/api/show)")
def test_cw_ollama_is_budgeted_at_the_window_it_serves(main):
    daemon, *_, ollama = main
    rows = daemon.get("/api/backends")
    row = next((b for b in (rows.get("backends") if isinstance(rows, dict) else rows) or []
                if b.get("name") == OLLAMA_BACKEND), None) if not isinstance(rows, dict) or "backends" in rows else rows.get(OLLAMA_BACKEND)
    assert row is not None, f"{OLLAMA_BACKEND} missing from /api/backends: {str(rows)[:300]}"
    assert row.get("n_ctx") in (ollama.served_ctx, None) and row.get("n_ctx") != ollama.trained_ctx, (
        f"budgeted {row.get('n_ctx')} (trained {ollama.trained_ctx}); Ollama serves {ollama.served_ctx}")


def _ollama_turn(main):
    daemon, *_, ollama = main
    sid = "beacon:cw-ollama"
    daemon.use_model(sid, OLLAMA_BACKEND)
    daemon.turn("Hi.", sid=sid)
    return ollama.requests


@pytest.mark.xfail(strict=True, reason="0.9.12: Prometheus sends Ollama no num_ctx (the OpenAI endpoint has none)")
def test_cw_prometheus_sends_ollama_the_num_ctx_it_budgets_for(main):
    reqs = _ollama_turn(main)
    assert reqs, "the session never reached the Ollama backend"
    assert all(r["num_ctx"] for r in reqs), f"num_ctx sent: {[r['num_ctx'] for r in reqs]}"


@pytest.mark.xfail(strict=True, reason="0.9.12: an Ollama prompt over the served window is cut from the front, silently")
def test_cw_an_ollama_prompt_is_never_cut_silently(main):
    reqs = main[3].requests or _ollama_turn(main)
    cut = [r for r in reqs if r["kept_tokens"] < r["sent_tokens"]]
    assert not cut, f"Ollama kept {cut[0]['kept_tokens']} of {cut[0]['sent_tokens']} prompt tokens"


# ---------------------------------------------------------------- cloud


def test_cw_a_cloud_model_on_the_default_window_is_flagged(main):
    daemon, *_ = main
    sid = "beacon:cw-cloud"
    daemon.use_model(sid, CLOUD_KEY)
    daemon.turn("[gate:hello] Hi.", sid=sid)
    log = daemon.log.read_text(encoding="utf-8", errors="replace")
    assert any("WARNING" in line and "qwen3.8-max" in line and "cloud_default" in line for line in log.splitlines()), (
        "no WARNING says qwen3.8-max is budgeted at cloud_default_limit for want of an override")
