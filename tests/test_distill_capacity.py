"""Tests for scripts/distill/capacity.py and scripts/distill/tower/seqlen_bench.py.

No ssh, no systemd, no GPU, no Telegram: the ladder and stability rules, the computed limits, the
output schemas and the rendered document are pure; the needle probes are built from a synthetic
corpus and answered by the fake server's perfect reader; the 4090 block runs against a fake 4090
host that answers the commands it sends, with a fake clock.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("distill_capacity", _DIR / "capacity.py")
SB = _load("distill_seqlen_bench", _DIR / "tower" / "seqlen_bench.py")
BES = C.bes()


# ---------------------------------------------------------------------------
# ladder and stability rules
# ---------------------------------------------------------------------------


def test_ladder_for_adds_a_larger_trained_context_and_drops_rungs_above_it() -> None:
    assert C.ladder_for(262144) == list(C.LADDER_4090)
    assert C.ladder_for(524288)[-1] == 524288 and C.ladder_for(524288)[:-1] == list(C.LADDER_4090)
    assert C.ladder_for(131072) == [32768, 49152, 65536, 98304, 131072]
    assert C.ladder_for(100000) == [32768, 49152, 65536, 98304, 100000]
    assert C.ladder_for(None) == list(C.LADDER_4090)


def test_prompt_target_and_limit() -> None:
    assert C.prompt_target(32768) == 32768 - 512 - 1024
    assert C.prompt_limit(32768) == 32768 - 512 - 256
    assert C.prompt_target(32768) < C.prompt_limit(32768)
    with pytest.raises(ValueError):
        C.prompt_target(1000)


def test_judge_reply_needs_each_value() -> None:
    vals = ["KESTREL-4471", "OSPREY-1203", "WREN-9001"]
    assert C.judge_reply("1: KESTREL-4471\n2: osprey-1203\n3: WREN-9001", vals) == [True, True, True]
    assert C.judge_reply("1: KESTREL-4471\n2: OSPREY-1204\n3: WREN-9001", vals) == [True, False, True]
    assert C.judge_reply(None, vals) == [False, False, False]
    assert C.judge_reply("anything", [""]) == [False]


def _req(variant: str, ok: bool = True, **kw: Any) -> Dict[str, Any]:
    r = {"kind": "needles", "variant": variant, "correct": [True, True, ok], "all_correct": ok,
         "prefill_tok_s": 1500.0, "prompt_n": 30000, "prompt_tokens": 31000, "decode_tok_s": 30.0}
    r.update(kw)
    return r


def _rung(ctx: int, **kw: Any) -> Dict[str, Any]:
    r = {"ctx": ctx, "load": "ok", "requests": [_req("A"), _req("B")], "headroom_min_mib": 900}
    r.update(kw)
    return r


@pytest.mark.parametrize("over,stable,why", [
    ({}, True, "stable"),
    ({"load": "load-failed", "load_detail": "out of memory at load"}, False, "did not load: out of memory"),
    ({"crashed": True}, False, "server crashed"),
    ({"requests": [_req("A"), _req("B", ok=False)]}, False, "needles missed (B: 2/3)"),
    ({"requests": [_req("A")]}, False, "only 1 of 2"),
    ({"requests": [_req("A"), {"kind": "needles", "variant": "B", "error": "HTTP 500: boom"}]}, False, "request failed"),
    ({"headroom_min_mib": 299}, False, "VRAM headroom 299 MiB < 300"),
    ({"headroom_min_mib": None}, False, "headroom unknown"),
])
def test_rung_verdict(over: Dict[str, Any], stable: bool, why: str) -> None:
    ok, reason = C.rung_verdict(_rung(65536, **over))
    assert ok is stable
    assert why in reason


def test_next_rung_stops_at_the_first_unstable_rung() -> None:
    ladder = [32768, 49152, 65536]
    assert C.next_rung(ladder, []) == 32768
    assert C.next_rung(ladder, [{"ctx": 32768, "final": True, "stable": True}]) == 49152
    rungs = [{"ctx": 32768, "final": True, "stable": True}, {"ctx": 49152, "final": True, "stable": False}]
    assert C.next_rung(ladder, rungs) is None
    rungs = [{"ctx": n, "final": True, "stable": True} for n in ladder]
    assert C.next_rung(ladder, rungs) is None


def test_ladder_summary_bounds() -> None:
    ladder = [32768, 49152, 65536]
    s = C.ladder_summary(ladder, [{"ctx": 32768, "final": True, "stable": True, "prefill_tok_s": 2000},
                                  {"ctx": 49152, "final": True, "stable": False, "reason": "did not load: oom"}])
    assert s["max_stable_ctx"] == 32768
    assert s["first_unstable"] == {"ctx": 49152, "reason": "did not load: oom"}
    assert s["bounded_by"] == "unstable rung above"
    assert s["at_max"]["prefill_tok_s"] == 2000
    s = C.ladder_summary(ladder, [{"ctx": n, "final": True, "stable": True} for n in ladder])
    assert s["max_stable_ctx"] == 65536 and s["bounded_by"].startswith("top of the ladder")
    s = C.ladder_summary(ladder, [{"ctx": 32768, "final": True, "stable": True}])
    assert s["bounded_by"].startswith("not bounded")
    s = C.ladder_summary(ladder, [{"ctx": 32768, "final": True, "stable": False, "reason": "x"}])
    assert s["max_stable_ctx"] is None


def test_kv_fit_recovers_bytes_per_token() -> None:
    rungs = [{"ctx": n, "load": "ok", "vram_used_mib": 17601 + n * 34816 / 1048576} for n in (32768, 65536, 131072)]
    fit = C.kv_fit(rungs)
    assert fit is not None
    assert abs(fit["bytes_per_token"] - 34816) < 1
    assert abs(fit["intercept_mib"] - 17601) < 1
    assert C.kv_fit(rungs[:1]) is None
    two_slot = [dict(r, slots=2) for r in rungs]
    assert C.kv_fit(two_slot) is None


def test_computed_max_ctx() -> None:
    # the 3090 Ti tonight: 728 MiB free with 57,344 configured
    got = C.computed_max_ctx(728, 57344, n_ctx_train=262144)
    extra = int((728 - 300 - 64) * 1048576 // 34816)
    assert got == (57344 + extra) // 1024 * 1024
    assert got % 1024 == 0 and got > 57344
    # under the floor: shrinks
    assert C.computed_max_ctx(200, 57344) < 57344
    # capped at the trained context
    assert C.computed_max_ctx(20000, 57344, n_ctx_train=262144) == 262144


def test_computed_mini_restart_advice() -> None:
    small = C.computed_mini(728, 57344, 262144, 5386)
    assert small["restart_advised"] is False
    assert small["max_ctx_current_residents"] < small["max_ctx_if_ollama_unloaded"]
    big = C.computed_mini(4000, 57344, 262144, 5386)
    assert big["restart_advised"] is True
    none = C.computed_mini(None, 57344, 262144, None)
    assert none["max_ctx_current_residents"] is None and none["restart_advised"] is False


def test_multi_slot_alternatives() -> None:
    fit = {"bytes_per_token": 34816.0, "intercept_mib": 17601.0}
    alts = C.multi_slot_alternatives(163840, fit, 24073)
    assert [a["slots"] for a in alts] == [2, 4]
    assert alts[0]["ctx_per_slot"] == 81920
    assert alts[1]["projected_mib"] > alts[0]["projected_mib"]
    assert C.multi_slot_alternatives(None, fit, 24073) == []


@pytest.mark.parametrize("results,want", [
    ({}, 16384),
    ({"16384": {"status": "ok"}}, 32768),
    ({"16384": {"status": "ok"}, "32768": {"status": "oom"}}, 24576),
    ({"16384": {"status": "ok"}, "32768": {"status": "oom"}, "24576": {"status": "ok"}}, None),
    ({"16384": {"status": "ok"}, "32768": {"status": "ok"}}, 65536),
    ({"16384": {"status": "ok"}, "32768": {"status": "ok"}, "65536": {"status": "oom"}}, 49152),
    ({"16384": {"status": "ok"}, "32768": {"status": "ok"}, "65536": {"status": "oom"}, "49152": {"status": "oom"}}, None),
    ({"16384": {"status": "ok"}, "32768": {"status": "ok"}, "65536": {"status": "ok"}}, None),
    ({"16384": {"status": "failed"}}, None),
    ({"16384": {"status": "oom"}}, None),
    ({"16384": {"status": "ok"}, "32768": {"status": "killed"}}, None),
])
def test_qlora_next(results: Dict[str, Any], want: Optional[int]) -> None:
    assert C.qlora_next(results) == want


def test_qlora_summary() -> None:
    s = C.qlora_summary({"16384": {"status": "ok"}, "32768": {"status": "ok"}, "65536": {"status": "oom"},
                         "49152": {"status": "ok"}})
    assert s == {"max_ok_seq": 49152, "first_oom_seq": 65536, "failed": [], "lengths_run": [16384, 32768, 49152, 65536]}


def test_recommend_picks_the_highest_context_and_caps_rows() -> None:
    quants = {"q4": {"label": "UD-Q4_K_XL", "summary": {"max_stable_ctx": 163840, "at_max": {"max_verified_prompt_tokens": 162000}}},
              "iq3": {"label": "IQ3_XXS", "summary": {"max_stable_ctx": 262144, "at_max": {"max_verified_prompt_tokens": 260400}}}}
    rec = C.recommend(quants, {"max_readable_ctx": 57344})
    assert rec["long_context_teacher"]["quant"] == "IQ3_XXS"
    assert rec["long_context_row_cap_tokens"] == 262144 - C.REPLY_RESERVE
    assert rec["max_verified_prompt_tokens"] == 260400
    assert any("UD-Q4_K_XL reads up to 160K" in n for n in rec["notes"])
    assert any("3090 Ti" in n for n in rec["notes"])
    tie = C.recommend({"q4": {"label": "UD-Q4_K_XL", "summary": {"max_stable_ctx": 131072}},
                       "iq3": {"label": "IQ3_XXS", "summary": {"max_stable_ctx": 131072}}})
    assert tie["long_context_teacher"]["quant"] == "UD-Q4_K_XL"
    assert C.recommend({})["long_context_teacher"] is None


def test_eval_clear_keeps_off_the_nightly_evals() -> None:
    at = lambda s: C.parse_iso(s)  # noqa: E731
    assert C.eval_clear(at("2026-11-01T02:20:00-05:00"))[0] is True
    assert C.eval_clear(at("2026-11-01T02:40:00-05:00"))[0] is False
    assert C.eval_clear(at("2026-11-01T03:20:00-05:00"))[0] is False
    assert C.eval_clear(at("2026-11-01T03:30:00-05:00"))[0] is True
    # a 15-minute request starting 05:25 would run into the 06:00 window (opens 05:35)
    assert C.eval_clear(at("2026-11-01T05:25:00-05:00"), need_s=900)[0] is False
    assert C.eval_clear(at("2026-11-01T05:00:00-05:00"), need_s=900)[0] is True
    ok, why = C.eval_clear(at("2026-11-01T05:50:00-05:00"))
    assert not ok and "06:00" in why


# ---------------------------------------------------------------------------
# output documents
# ---------------------------------------------------------------------------


def _teacher_doc() -> Dict[str, Any]:
    rung = dict(_rung(32768), stable=True, reason="stable", final=True, vram_used_mib=17800,
                prefill_tok_s=2100.0, decode_tok_s=33.5, max_verified_prompt_tokens=31200)
    q4 = {"label": "UD-Q4_K_XL", "status": "measured", "rungs": [rung],
          "summary": C.ladder_summary([32768], [rung]), "multi_slot": [{"slots": 2, "ctx_per_slot": 16384,
                                                                         "source": "measured", "fits": True,
                                                                         "vram_used_mib": 17950}],
          "kv_fit": {"bytes_per_token": 34816.0, "intercept_mib": 16700.0, "points": 3}}
    mini = {"current_config": {"ctx": 57344, "slots": 2, "kv_unified": True, "quant": "IQ3_XXS"},
            "measured": {"max_readable_ctx": 57344, "bounded_by": "top", "rungs": [rung]},
            "computed": C.computed_mini(728, 57344, 262144, 5386)}
    return {"schema": C.TEACHER_SCHEMA, "hosts": {"4090": {"quants": {"q4": q4}}, "3090ti": mini},
            "recommendation": C.recommend({"q4": q4}, mini["measured"])}


def test_validate_teacher_ctx() -> None:
    doc = _teacher_doc()
    assert C.validate_teacher_ctx(doc) == []
    bad = json.loads(json.dumps(doc))
    bad["hosts"]["4090"]["quants"]["q4"]["status"] = "great"
    del bad["hosts"]["3090ti"]["computed"]
    errs = C.validate_teacher_ctx(bad)
    assert any("status" in e for e in errs) and any("3090ti" in e for e in errs)
    assert C.validate_teacher_ctx([]) == ["not an object"]


def test_validate_qlora_and_done() -> None:
    q = {"schema": C.QLORA_SCHEMA, "lengths": {"16384": {"status": "ok", "tok_s": 2500.0},
                                               "65536": {"status": "oom"}}, "summary": {}}
    assert C.validate_qlora(q) == []
    assert C.validate_qlora({"schema": C.QLORA_SCHEMA, "lengths": {"x": {"status": "ok"}}, "summary": {}})
    d = {"schema": C.DONE_SCHEMA, "status": "done", "business_left_down": True, "finished": "now"}
    assert C.validate_done(d) == []
    assert C.validate_done(dict(d, business_left_down="yes"))
    assert C.validate_done(dict(d, status="ok"))


def test_render_markdown_is_scrubbed_and_holds_the_numbers() -> None:
    q = {"schema": C.QLORA_SCHEMA, "config": {"r": 32}, "lengths": {
        "16384": {"status": "ok", "tok_s": 2540.0, "step_s_median": 6.4, "peak_vram_reserved_gib": 13.0,
                  "peak_rss_gib": 9.5, "host_used_peak_gib": 20.0},
        "65536": {"status": "oom", "detail": "CUDA out of memory at timed step 2"}},
        "summary": {"max_ok_seq": 16384, "first_oom_seq": 65536}}
    md = C.render_markdown(_teacher_doc(), q, {"status": "done", "business_left_down": True})
    assert C.scrub_problems(md, ["FakeBiz"]) == []
    for s in ("UD-Q4_K_XL: max stable context **32K**", "2,100", "33.5", "3/3 / 3/3", "| 16,384 | ok | 2,540",
              "oom: CUDA out of memory", "Longest that trained: **16K**", "left down for generation: true",
              "3090 Ti teacher", "not advised"):
        assert s in md, s
    pending = C.render_markdown({"schema": C.TEACHER_SCHEMA, "hosts": {"3090ti": _teacher_doc()["hosts"]["3090ti"]},
                                 "recommendation": {}}, None, None, pending_4090=True)
    assert "Pending: the measurement block is armed" in pending


def test_scrub_problems() -> None:
    assert C.scrub_problems("served on 192.0.2.7:18094") == ["an IPv4 address"]
    assert C.scrub_problems("see /home/will/distill") and C.scrub_problems("~/distill/x")
    assert C.scrub_problems("box.tail1234.ts.net")
    assert C.scrub_problems("Acme back up", ["acme"]) == ["a banned name"]
    assert C.scrub_problems("Qwen3.8-27B at 131072 tokens, 0.95 val") == []


def test_load_failure_reason_has_no_paths() -> None:
    tail = "llama_model_load: /home/x/models/m.gguf\nggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n"
    assert C.load_failure_reason(tail) == "out of memory at load"
    assert C.load_failure_kind(tail) == "load-failed"
    ref = "serve_teacher: REFUSED: 1 slots x 262144 need 25105 MiB (projection + 0), more than this card's usable 24073"
    assert C.load_failure_kind(ref) == "refused"
    assert "projection" in C.load_failure_reason(ref)
    assert "/home" not in C.load_failure_reason("something odd at /home/x/y")


# ---------------------------------------------------------------------------
# needle probes (synthetic corpus, word-count tokens)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def builder(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("corpus")
    repo, wiki = BES.synthetic_corpus(root, n_docs=80, n_code=60, n_wiki=30)
    docs = BES.load_sources(repo, wiki)
    meter = BES.PromptMeter(BES.RegexTokenCounter(), None)
    return C.ProbeBuilder(docs, meter, "You are a test assistant.")


@pytest.mark.parametrize("variant", ["A", "B"])
def test_probe_fills_the_context_with_three_needles(builder: Any, variant: str) -> None:
    ctx = 16384
    pr = builder.build(ctx, variant, "test")
    assert C.prompt_target(ctx) * 0.99 <= pr["prompt_tokens"] <= C.prompt_limit(ctx)
    assert pr["format"] == C.FORMATS[variant] and pr["depths"] == list(C.DEPTHS[variant])
    assert len(pr["values"]) == 3 and len(set(pr["values"])) == 3
    text = "\n".join(str(m.get("content") or "") for m in pr["messages"])
    for v in pr["values"]:
        assert text.count(v) == 1
    # depths land near their targets and in order
    assert pr["depth_pct"] == sorted(pr["depth_pct"])
    for got, want in zip(pr["depth_pct"], C.DEPTHS[variant]):
        assert abs(got - want) < 6
    # a perfect reader answers all three; one that stops early does not
    assert C.judge_reply(C.fake_answer(pr["messages"]), pr["values"]) == [True, True, True]
    partial = C.judge_reply(C.fake_answer(pr["messages"], readable_chars=len(text) // 2), pr["values"])
    assert partial[0] is True and partial[2] is False


def test_probes_are_deterministic_and_differ_by_variant(builder: Any) -> None:
    a1, a2 = builder.build(16384, "A", "h"), builder.build(16384, "A", "h")
    b = builder.build(16384, "B", "h")
    assert a1["values"] == a2["values"] and a1["messages"] == a2["messages"]
    assert set(a1["values"]).isdisjoint(b["values"])
    assert builder.build(16384, "A", "other")["values"] != a1["values"]


def test_ensure_probe_caches_privately(builder: Any, tmp_path: Path) -> None:
    logs: List[str] = []
    pr = C.ensure_probe(lambda: builder, tmp_path, "h", 16384, "A", logs.append)
    p = C.probe_path(tmp_path, "h", 16384, "A")
    assert p.is_file() and oct(p.stat().st_mode & 0o777) == "0o600"
    assert oct(p.parent.stat().st_mode & 0o777) == "0o700"
    again = C.ensure_probe(lambda: (_ for _ in ()).throw(AssertionError("rebuilt")), tmp_path, "h", 16384, "A", logs.append)
    assert again["values"] == pr["values"]


def test_needle_and_decode_requests_against_the_fake_server(builder: Any, tmp_path: Path) -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, FAKE_CTX="16384", FAKE_PORT=str(port), FAKE_ALIAS="fake-q4",
               FAKE_STATE=str(tmp_path / "vram"), FAKE_WEIGHTS_MIB="9000")
    proc = subprocess.Popen([sys.executable, "-I", str(_DIR / "capacity.py"), "fake-server"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            code, _ = C.http_json(url + "/health", timeout=2)
            if code == 200:
                break
            time.sleep(0.1)
        assert int((tmp_path / "vram").read_text()) > 9000
        pr = builder.build(16384, "A", "srv")
        r = C.needle_request(url, pr, "k" * 64)
        assert r["all_correct"] is True and r["prefill_tok_s"] == 2000.0 and r["prompt_tokens"] > 1000
        d = C.decode_request(url, pr, None)
        assert d["kind"] == "decode" and d["decode_tok_s"] == 30.0 and d["predicted_n"] == C.DECODE_BENCH_TOKENS
        rung = {"requests": [r, dict(r, variant="B"), d]}
        C.rung_speeds(rung)
        assert rung["prefill_tok_s"] == 2000.0 and rung["decode_tok_s"] == 30.0
        assert rung["max_verified_prompt_tokens"] == r["prompt_tokens"]
        bad = C.chat("http://127.0.0.1:9", [{"role": "user", "content": "x"}], timeout=2)
        assert bad["error"].startswith("HTTP 0")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# the 4090 block against a fake 4090 host
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(s, 0.5)


class FakeTower:
    """Answers the commands the block sends to the 4090, from a small model of the host."""

    def __init__(self, clock: FakeClock, *, max_ctx: Dict[str, int], qlora_max: int = 49152,
                 flags: Tuple[str, ...] = (), verify_rc: int = 0, free_gb: float = 70.0) -> None:
        self.clock = clock
        self.max_ctx, self.qlora_max = max_ctx, qlora_max
        self.units: Dict[str, str] = {"llama-server.service": "active"}
        self.unit_env: Dict[str, Dict[str, str]] = {}
        self.logs: Dict[str, str] = {}
        self.qlora_out: Dict[str, str] = {}
        self.vram = 20288
        self.flags = list(flags)
        self.verify_rc = verify_rc
        self.free_gb = free_gb
        self.telegrams: List[str] = []
        self.calls: List[str] = []
        self.restored = False
        self.home = "/fake/home"

    def p(self, rel: str) -> str:
        return f"{self.home}/{rel}"

    def rsync_to(self, src: Path, rel: str, timeout: float = 0) -> Tuple[int, str]:
        self.calls.append(f"rsync {rel}")
        return 0, ""

    def run(self, script: str, *, mut: bool = False, timeout: float = 120, stdin: Optional[str] = None) -> Tuple[int, str]:
        self.calls.append(script)
        s = script
        if s.startswith("cd ~ && echo biz="):
            out = [f"biz={self.units.get('llama-server.service', 'inactive')}"]
            out += [f"flag={f}" for f in self.flags]
            out += [f"free_gb={self.free_gb}", f"vram={self.vram}", f"vfree={24073 - self.vram}"]
            out += [f"unit={u}" for u, st in self.units.items() if u.startswith("distill-") and st == "active"]
            return 0, "\n".join(out)
        if "notify.sh" in s:
            self.telegrams.append(stdin or "")
            return 0, '"ok":true'
        if s.startswith("sed -n 's/^BIZ_NAME"):
            return 0, "FakeBiz"
        if s.startswith("exec 9>>"):
            self.units["llama-server.service"] = "inactive"
            self.vram = 300
            return 0, "before=active\nstopped=yes\nfreed=300"
        if s.startswith("nvidia-smi --query-gpu=memory.total"):
            return 0, "24564, 491"
        m = re.match(r"systemctl --user is-active '?([\w.@-]+)'?", s)
        if m:
            return 0, self.units.get(m.group(1), "inactive")
        m = re.match(r"systemctl --user stop '?([\w.@-]+)'?", s)
        if m:
            self.units[m.group(1)] = "inactive"
            if "srv-" in m.group(1) or "qlora-" in m.group(1):
                self.vram = 300
            return 0, ""
        if "systemd-run" in s:
            name = re.search(r"--unit='?([\w.@-]+)'?", s).group(1)
            unit = name if name.endswith(".service") else name + ".service"
            env = dict(re.findall(r"--setenv='?([A-Z_]+)=([^' ]*)'?", s))
            self.unit_env[unit] = env
            if "srv-" in name:
                key = "q4" if "q4" in env.get("TEACHER_ALIAS", "") else "iq3"
                tot = int(env["TEACHER_SLOTS"]) * int(env["TEACHER_CTX_PER_SLOT"])
                if tot > self.max_ctx[key]:
                    self.units[unit] = "failed"
                    self.logs[unit] = "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory"
                else:
                    self.units[unit] = "active"
                    w = 16401 if key == "q4" else 9100
                    self.vram = int(w + 1200 + tot * 34816 / 1048576)
                return 0, ""
            if "qlora-" in name:
                n = int(re.search(r"--seq-len (\d+)", s).group(1))
                st = "ok" if n <= self.qlora_max else "oom"
                doc = {"schema": "x", "seq_len": n, "status": st, "config": {"r": 32}}
                if st == "ok":
                    doc.update(tok_s=round(2600.0 - n / 100.0, 1), step_s_median=round(n / 2500.0, 2),
                               peak_vram_reserved_gib=6 + n / 8192)
                else:
                    doc["detail"] = "CUDA out of memory at timed step 1"
                self.qlora_out[str(n)] = json.dumps(doc)
                self.units[unit] = "inactive"
                return 0, ""
            if "restore-" in name:
                self.units[unit] = "inactive"
                self.units["llama-server.service"] = "active"
                self.restored = True
                self.vram = 20288
                return 0, ""
            self.units[unit] = "active"  # sampler
            return 0, ""
        if s.startswith("tail -n 40"):
            unit = re.search(r"/([\w.-]+)\.log", s).group(1) + ".service"
            return 0, self.logs.get(unit, "")
        if "serve_teacher.sh" in s and s.rstrip().endswith("verify"):
            return self.verify_rc, "serve_teacher: verified" if self.verify_rc == 0 else "VERIFY FAILED: tools on"
        m = re.match(r"cat '?[^ ]*/qlora-(\d+)\.json", s)
        if m:
            return 0, self.qlora_out.get(m.group(1), "")
        if s.startswith("systemctl --user show"):
            return 0, "Result=success\nExecMainStatus=0"
        if "MemTotal" in s:
            return 0, "65670664"
        if s.startswith("test -s"):
            return 0, "none"
        if "stat -c %s" in s:
            return 0, "0"
        if s.startswith("sha256sum"):
            sha = C.QUANTS["iq3"]["sha256"]
            return 0, f"{sha}\n{sha}"
        if "restore.rc" in s and s.startswith("cat"):
            return 0, "0"
        if "verify.last" in s:
            return 0, "2026-11-01 17:00:00 EST tower_verify: PASS"
        if s.startswith("awk -F,"):
            return 0, "n=0"
        return 0, ""


def _cfg(tmp_path: Path, **over: Any) -> Any:
    ns = argparse.Namespace(
        execute=True, tower="u@fake", tower_url="http://fake:18094", fake_root=None, capdir=str(tmp_path / "cap"),
        key_file=str(tmp_path / "key"), pipeline_done=str(tmp_path / "PIPELINE-DONE.json"), pipeline_unit="",
        skip_at="2026-11-02T03:00:00-05:00", restore_after="2026-11-02T04:00:00-05:00",
        window_end="2026-11-02T05:40:00-05:00", budget_min="150", poll="60", no_generation_follows=False,
        quants="q4,iq3", ladder="", qlora_lengths="", qlora_runtime_s="2400", skip_qlora=False,
        code_rel="distill/capacity/code/head", port="18094", repo=".", wiki=".", tokenizer_dir=".", template=".",
        iq3_src=str(tmp_path / "iq3.gguf"), iq3_manifest=str(tmp_path / "MANIFEST.json"), quant_override=None)
    for k, v in over.items():
        setattr(ns, k, v)
    (tmp_path / "key").write_text("k" * 64 + "\n")
    return C.Cfg(ns)


def _block(tmp_path: Path, tower: FakeTower, clock: FakeClock, builder: Any, *, max_read: Optional[Dict[str, int]] = None,
           **over: Any) -> Any:
    cfg = _cfg(tmp_path, **over)

    def http(url: str, *, body: Any = None, key: Optional[str] = None, timeout: float = 30) -> Tuple[int, Any]:
        live = [u for u, st in tower.units.items() if "srv-" in u and st == "active"]
        if not live:
            return 0, {"error": "refused"}
        env = tower.unit_env[live[-1]]
        if url.endswith("/health"):
            return 200, {"status": "ok"}
        if url.endswith("/props"):
            return 200, {"default_generation_settings": {"n_ctx": int(env["TEACHER_CTX_PER_SLOT"])},
                         "total_slots": int(env["TEACHER_SLOTS"])}
        if url.endswith("/v1/models"):
            return 200, {"data": [{"id": env["TEACHER_ALIAS"], "meta": {"n_ctx_train": 262144}}]}
        return 404, None

    def needle(url: str, pr: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
        live = [u for u, st in tower.units.items() if "srv-" in u and st == "active"][-1]
        k = "q4" if "q4" in tower.unit_env[live]["TEACHER_ALIAS"] else "iq3"
        ok = pr["ctx"] <= (max_read or {}).get(k, 10**9)
        clock.t += pr["ctx"] / 1500.0
        return {"kind": "needles", "variant": pr["variant"], "correct": [True, True, ok], "all_correct": ok,
                "prompt_tokens": pr["prompt_tokens"], "prompt_n": pr["prompt_tokens"],
                "prefill_tok_s": 1500.0, "decode_tok_s": 30.0}

    def decode(url: str, pr: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
        return {"kind": "decode", "variant": "B", "decode_tok_s": 31.0, "predicted_n": 128}

    # the 4090 probes are built at small sizes: the ladder is scaled down by 1/16 for the builder
    def small_builder() -> Any:
        class Scaled:
            def build(self, ctx: int, variant: str, host: str) -> Dict[str, Any]:
                pr = builder.build(max(8192, ctx // 16), variant, host)
                return dict(pr, ctx=ctx, prompt_tokens=C.prompt_target(ctx))
        return Scaled()

    b = C.Block(cfg, tower=tower, clock=clock, sleep=clock.sleep, builder_fn=small_builder, http=http,
                needle_fn=needle, decode_fn=decode)
    b.tower = tower
    return b


SUN_1530 = C.parse_iso("2026-11-01T15:30:00-05:00")


def test_block_full_run_leaves_business_down_for_generation(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 163840, "iq3": 262144}, qlora_max=49152)
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder)
    assert b.run() == 0
    cap = tmp_path / "cap"
    done = json.loads((cap / "CAPACITY-DONE.json").read_text())
    assert C.validate_done(done) == []
    assert done["status"] == "done" and done["business_left_down"] is True
    assert done["max_stable_ctx"] == {"q4": 163840, "iq3": 262144}
    assert done["long_context_teacher"]["quant"] == "IQ3_XXS"
    assert done["qlora_max_ok_seq"] == 49152
    teacher = json.loads((cap / "teacher-ctx.json").read_text())
    assert C.validate_teacher_ctx(teacher) == []
    q4 = teacher["hosts"]["4090"]["quants"]["q4"]
    assert [r["ctx"] for r in q4["rungs"]] == [32768, 49152, 65536, 98304, 131072, 163840, 196608]
    assert q4["rungs"][-1]["load"] == "load-failed" and "out of memory" in q4["rungs"][-1]["reason"]
    assert q4["summary"]["first_unstable"]["ctx"] == 196608
    assert abs(q4["kv_fit"]["bytes_per_token"] - 34816) < 50
    assert q4["multi_slot"][0]["source"] == "measured" and q4["multi_slot"][0]["slots"] == 2
    assert q4["recommended_profile"]["env"] == {"TEACHER_SLOTS": 1, "TEACHER_CTX_PER_SLOT": 163840}
    iq3 = teacher["hosts"]["4090"]["quants"]["iq3"]
    assert iq3["summary"]["max_stable_ctx"] == 262144 and iq3["summary"]["bounded_by"].startswith("top")
    qlora = json.loads((cap / "qlora-seqlen.json").read_text())
    assert C.validate_qlora(qlora) == []
    assert sorted(qlora["lengths"], key=int) == ["16384", "32768", "49152", "65536"]
    assert qlora["lengths"]["65536"]["status"] == "oom"
    md = (cap / "DISTILL-CAPACITY.md").read_text()
    assert C.scrub_problems(md, ["FakeBiz"]) == []
    # the business server was stopped once, under the window lock, and not restored
    assert sum(1 for c in tower.calls if c.startswith("exec 9>>")) == 1
    assert not tower.restored
    assert tower.units["llama-server.service"] == "inactive"
    # every server and bench unit is distill-cap-* and stopped at the end
    started = [re.search(r"--unit='?([\w.@-]+)", c).group(1) for c in tower.calls if "systemd-run" in c]
    assert started and all(u.startswith("distill-cap-") for u in started)
    assert all(st != "active" for u, st in tower.units.items() if u.startswith("distill-cap-"))
    # the server env changes only the slots and the context (plus the zeroed projection overheads)
    env = tower.unit_env["distill-cap-srv-q4-c32768-1.service"]
    assert env["TEACHER_SLOTS"] == "1" and env["TEACHER_CTX_PER_SLOT"] == "32768"
    assert env["TEACHER_BUFFERS_MIB"] == "0" and env["TEACHER_MARGIN_MIB"] == "0"
    # Telegrams: the down notice and the headline, <= 5 lines each, the business name only there
    assert len(tower.telegrams) == 2
    assert tower.telegrams[0].startswith("FakeBiz going down: teacher/trainer capacity tests")
    assert all(len(t.strip().splitlines()) <= 5 for t in tower.telegrams)
    assert "left down for generation" in tower.telegrams[1]


def test_block_restores_business_when_something_fails(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 163840, "iq3": 262144}, verify_rc=1)
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder)
    assert b.run() == 0
    done = json.loads((tmp_path / "cap" / "CAPACITY-DONE.json").read_text())
    assert done["status"] == "failed" and "verify" in done["reason"]
    assert done["business_left_down"] is False
    assert tower.restored and done["restore"]["ok"] is True


def test_block_late_at_the_end_restores(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(C.parse_iso("2026-11-02T03:58:00-05:00"))
    tower = FakeTower(clock, max_ctx={"q4": 98304, "iq3": 98304})
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder, quants="q4", skip_qlora=True, skip_at="2026-11-02T05:00:00-05:00",
               restore_after="2026-11-02T03:59:00-05:00")
    assert b.run() == 0
    assert clock.t > C.parse_iso("2026-11-02T03:59:00-05:00")
    done = json.loads((tmp_path / "cap" / "CAPACITY-DONE.json").read_text())
    assert done["status"] == "done" and done["business_left_down"] is False
    assert tower.restored and done["restore"]["ok"] is True


def test_block_skips_when_the_pipeline_never_finishes(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(C.parse_iso("2026-11-02T01:00:00-05:00"))
    tower = FakeTower(clock, max_ctx={"q4": 163840, "iq3": 262144})
    b = _block(tmp_path, tower, clock, builder)
    assert b.run() == 0
    done = json.loads((tmp_path / "cap" / "CAPACITY-DONE.json").read_text())
    assert done["status"] == "skipped" and "had not finished" in done["reason"]
    assert done["business_left_down"] is False
    assert not any(c.startswith("exec 9>>") or "systemd-run" in c for c in tower.calls)
    assert tower.units["llama-server.service"] == "active"
    assert len(tower.telegrams) == 1 and "skipped" in tower.telegrams[0]


def test_block_skips_on_a_no_swaps_flag(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 163840, "iq3": 262144}, flags=("distill/window/NO-SWAPS-TONIGHT",))
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder)
    assert b.run() == 0
    done = json.loads((tmp_path / "cap" / "CAPACITY-DONE.json").read_text())
    assert done["status"] == "skipped" and "NO-SWAPS" in done["reason"]
    assert tower.units["llama-server.service"] == "active"


def test_block_stops_a_ladder_on_missed_needles(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 262144, "iq3": 262144})
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder, max_read={"q4": 98304}, quants="q4", skip_qlora=True)
    assert b.run() == 0
    teacher = json.loads((tmp_path / "cap" / "teacher-ctx.json").read_text())
    q4 = teacher["hosts"]["4090"]["quants"]["q4"]
    assert q4["summary"]["max_stable_ctx"] == 98304
    assert "needles missed" in q4["summary"]["first_unstable"]["reason"]


def test_block_time_budget_makes_a_partial_result(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 262144, "iq3": 262144})
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder, budget_min="60")
    assert b.run() == 0
    done = json.loads((tmp_path / "cap" / "CAPACITY-DONE.json").read_text())
    assert done["status"] == "partial"
    assert done["business_left_down"] is True  # a budget stop is not a failure
    state = json.loads((tmp_path / "cap" / "block-state.json").read_text())
    assert any(q.get("status") == "partial" or q.get("stopped_at") for q in state["quants"].values()) \
        or any(v.get("status") == "skipped" for v in state["qlora"].values())


def test_block_skips_iq3_without_disk(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 65536, "iq3": 262144}, free_gb=20.0)
    (tmp_path / "PIPELINE-DONE.json").write_text(json.dumps({"outcome": "completed"}))
    b = _block(tmp_path, tower, clock, builder, skip_qlora=True)
    assert b.run() == 0
    teacher = json.loads((tmp_path / "cap" / "teacher-ctx.json").read_text())
    iq3 = teacher["hosts"]["4090"]["quants"]["iq3"]
    assert iq3["status"] == "skipped" and "GB free" in iq3["skip_reason"]
    assert not any(c.startswith("rsync") for c in tower.calls)
    done = json.loads((tmp_path / "cap" / "CAPACITY-DONE.json").read_text())
    assert done["status"] == "partial" and done["iq3_on_4090"] is False


def test_block_done_marker_means_nothing_to_do(tmp_path: Path, builder: Any) -> None:
    clock = FakeClock(SUN_1530)
    tower = FakeTower(clock, max_ctx={"q4": 65536, "iq3": 65536})
    (tmp_path / "cap").mkdir()
    (tmp_path / "cap" / "CAPACITY-DONE.json").write_text("{}")
    b = _block(tmp_path, tower, clock, builder)
    assert b.run() == 0
    assert tower.calls == []


# ---------------------------------------------------------------------------
# seqlen_bench
# ---------------------------------------------------------------------------


def test_pack_makes_exact_lengths_with_shifted_spans() -> None:
    rows = [(list(range(100 * i, 100 * i + 60)), [(10, 20), (50, 60)]) for i in range(10)]
    seqs = SB.pack(iter(rows), 100, 3)
    assert [len(ids) for ids, _ in seqs] == [100, 100, 100]
    ids0, sp0 = seqs[0]
    assert ids0[:60] == list(range(0, 60)) and ids0[60:] == list(range(100, 140))
    assert sp0 == [(10, 20), (50, 60), (70, 80)]  # the second row's (50, 60) is past the cut
    ids1, sp1 = seqs[1]
    assert ids1[0] == 200  # a sequence starts at a row start: the rest of the cut row is dropped
    with pytest.raises(ValueError):
        SB.pack(iter(rows[:2]), 100, 3)


def test_sample_rows_is_seeded_and_sees_every_row() -> None:
    seen: List[int] = []
    a = SB.sample_rows(({"i": i} for i in range(1000)), 10, "s", on_row=lambda r: seen.append(r["i"]))
    b = SB.sample_rows(({"i": i} for i in range(1000)), 10, "s")
    assert a == b and len(a) == 10 and len(seen) == 1000
    assert SB.loss_positions([(0, 5), (10, 12)]) == 4 + 2


def _bench_rows(path: Path, n: int, max_derived: bool) -> Path:
    rows = []
    for i in range(n):
        prompt = "<|im_start|>user\n" + " ".join(f"w{i}x{j}" for j in range(400)) + "<|im_end|>\n<|im_start|>assistant\n"
        answer = " ".join(f"a{i}y{j}" for j in range(200)) + "<|im_end|>"
        rows.append({"id": f"r{i}", "text": prompt + answer + "\n",
                     "loss_spans_char": [[len(prompt), len(prompt) + len(answer)]], "tokenize": "segments",
                     "source": "local-27b" if not max_derived else "max-golden", "max_derived": max_derived})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class _WordEnc:
    TOK = re.compile(r"<\|[a-z_]+\|>|\w+|\s+|[^\w\s]")

    def __init__(self) -> None:
        self.v: Dict[str, int] = {}

    def encode_ids(self, text: str) -> List[int]:
        return [self.v.setdefault(t, len(self.v) + 1) for t in self.TOK.findall(text)]


def _finish_rows(path: Path, T: Any, enc: Any) -> None:
    out = []
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        ids, spans = [], []
        for seg, loss in T.segments(r["text"], r["loss_spans_char"]):
            part = enc.encode_ids(seg)
            if loss:
                spans.append([len(ids), len(ids) + len(part)])
            ids.extend(part)
        r["n_tokens"], r["loss_spans_tok"] = len(ids), spans
        out.append(json.dumps(r))
    path.write_text("\n".join(out) + "\n")


@pytest.mark.parametrize("length,limit,status,rc", [(2048, 4096, "ok", 0), (4096, 2048, "oom", 3)])
def test_bench_run_with_the_fake_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length: int, limit: int,
                                         status: str, rc: int) -> None:
    T = SB.load_trainer(_DIR / "train_qlora.py")
    enc = _WordEnc()
    monkeypatch.setattr(T, "hf_encoder", lambda d: enc)
    data = _bench_rows(tmp_path / "train.jsonl", 40, max_derived=False)
    _finish_rows(data, T, enc)
    monkeypatch.setenv("FAKE_QLORA_MAX", str(limit))
    out = tmp_path / "bench" / f"qlora-{length}.json"
    got = SB.main(["run", "--trainer", str(_DIR / "train_qlora.py"), "--data", str(data), "--tokenizer-dir", str(tmp_path),
                   "--base-model", str(tmp_path), "--seq-len", str(length), "--steps", "3", "--warmup-steps", "1",
                   "--out", str(out), "--fake-backend"])
    assert got == rc
    doc = json.loads(out.read_text())
    assert doc["status"] == status
    assert doc["packing"]["sequences"] == 4 and doc["packing"]["tokens_each"] == length
    assert doc["config"]["max_seq"] == length and doc["config"]["gradient_checkpointing"] == "unsloth"
    if status == "ok":
        assert doc["tok_s"] > 0 and len(doc["steps_done"]) == 4 and doc["steps_done"][0]["warmup"] is True
    else:
        assert doc["detail"] == "CUDA out of memory at warm-up step 1"
    assert oct(out.stat().st_mode & 0o777) == "0o600"


def test_bench_refuses_max_derived_rows_without_both_approvals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    T = SB.load_trainer(_DIR / "train_qlora.py")
    enc = _WordEnc()
    monkeypatch.setattr(T, "hf_encoder", lambda d: enc)
    data = _bench_rows(tmp_path / "train.jsonl", 20, max_derived=True)
    _finish_rows(data, T, enc)
    out = tmp_path / "bench-max-derived" / "qlora-2048.json"
    args = ["run", "--trainer", str(_DIR / "train_qlora.py"), "--data", str(data), "--tokenizer-dir", str(tmp_path),
            "--base-model", str(tmp_path), "--seq-len", "2048", "--out", str(out), "--fake-backend"]
    assert SB.main(args) == 2  # no flag, no dataset approval
    assert SB.main(args + [SB.APPROVAL_FLAG]) == 2  # the flag alone is not enough
    assert not out.exists()


def test_bench_dry_run_plans_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    T = SB.load_trainer(_DIR / "train_qlora.py")
    enc = _WordEnc()
    monkeypatch.setattr(T, "hf_encoder", lambda d: enc)
    data = _bench_rows(tmp_path / "train.jsonl", 30, max_derived=False)
    _finish_rows(data, T, enc)
    out = tmp_path / "b" / "q.json"
    assert SB.main(["run", "--trainer", str(_DIR / "train_qlora.py"), "--data", str(data), "--tokenizer-dir",
                    str(tmp_path), "--base-model", str(tmp_path), "--seq-len", "4096", "--out", str(out), "--dry-run"]) == 0
    assert json.loads(out.read_text())["status"] == "planned"


def test_sampler_script_parses() -> None:
    assert subprocess.run(["bash", "-n", str(_DIR / "tower" / "cap_sampler.sh")]).returncode == 0
