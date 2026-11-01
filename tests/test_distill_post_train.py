"""Tests for scripts/distill/post_train.py: the post-training pipeline's decision logic and flow.

No ssh, no systemd, no GPU, no Telegram: a fake 4090 host answers the commands the pipeline
sends, a fake clock advances on every sleep, and Telegrams go to a list.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

_F = Path(__file__).resolve().parent.parent / "scripts" / "distill" / "post_train.py"


def _load() -> Any:
    name = "distill_post_train"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _F)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PT = _load()
STOP_AT = "2026-11-02T05:00:00-05:00"
HARD_AT = "2026-11-02T05:20:00-05:00"
T_STOP = PT.parse_iso(STOP_AT)
SUN_10 = PT.parse_iso("2026-11-01T10:00:00-05:00")


def hist(*pairs: Tuple[int, float, float]) -> List[Dict[str, Any]]:
    return [{"step": s, "val_loss": v, "probe_loss": p} for s, v, p in pairs]


def done_manifest(**over: Any) -> Dict[str, Any]:
    m: Dict[str, Any] = {
        "status": "done", "mode": "train", "step": 1120, "adapter": "/x/adapter-max-derived",
        "config": {"epochs": 2}, "epochs_done": [{"epoch": 1, "step": 560}, {"epoch": 2, "step": 1120}],
        "val_history": hist((0, 1.0, 1.13), (560, 0.82, 0.93), (1120, 0.80, 0.90)),
        "progress": {"total_steps": 1120}, "nonfinite_steps": 0, "data": {"sha256": "a" * 64}}
    m.update(over)
    return m


# ---------------------------------------------------------------------------
# training_verdict
# ---------------------------------------------------------------------------


def test_verdict_done_run_passes() -> None:
    ok, why, info = PT.training_verdict(done_manifest(), {}, adapter_present=True)
    assert ok, why
    assert info["val"] == [1.0, 0.80] and info["probe"] == [1.13, 0.90]
    assert "epochs [1, 2]" in why


@pytest.mark.parametrize("status,reason", [("failed", "diverged: 8 steps in a row"), ("stopped", "SIGTERM"),
                                           ("running", None), (None, None)])
def test_verdict_refuses_anything_but_done(status: Optional[str], reason: Optional[str]) -> None:
    m = done_manifest(status=status, reason=reason)
    ok, why, _ = PT.training_verdict(m, {}, adapter_present=True)
    assert not ok
    assert f"training status {status or 'unknown'}" in why
    if reason:
        assert reason in why


def test_verdict_needs_the_adapter_on_disk_and_in_the_manifest() -> None:
    assert not PT.training_verdict(done_manifest(), {}, adapter_present=False)[0]
    assert not PT.training_verdict(done_manifest(adapter=None), {}, adapter_present=True)[0]


def test_verdict_needs_every_epoch() -> None:
    ok, why, _ = PT.training_verdict(done_manifest(epochs_done=[{"epoch": 1}]), {}, adapter_present=True)
    assert not ok and "1 of 2 epochs" in why


def test_verdict_held_out_losses_must_fall_and_be_finite() -> None:
    up = done_manifest(val_history=hist((0, 1.0, 1.1), (1120, 1.01, 0.9)))
    assert "val loss did not fall" in PT.training_verdict(up, {}, adapter_present=True)[1]
    probe = done_manifest(val_history=hist((0, 1.0, 1.1), (1120, 0.9, 1.1)))
    assert "probe loss did not fall" in PT.training_verdict(probe, {}, adapter_present=True)[1]
    nan = done_manifest(val_history=hist((0, 1.0, 1.1), (1120, float("nan"), 0.9)))
    assert "non-finite" in PT.training_verdict(nan, {}, adapter_present=True)[1]
    only0 = done_manifest(val_history=hist((0, 1.0, 1.1)))
    assert "after step 0" in PT.training_verdict(only0, {}, adapter_present=True)[1]


def test_verdict_reads_run_state_when_the_manifest_lacks_history() -> None:
    m = done_manifest(val_history=None, epochs_done=None)
    rs = {"val_history": hist((0, 1.0, 1.1), (1120, 0.8, 0.9)), "epochs_done": [{"epoch": 1}, {"epoch": 2}]}
    assert PT.training_verdict(m, rs, adapter_present=True)[0]


def test_verdict_survives_garbage() -> None:
    assert not PT.training_verdict(None, "x", adapter_present=True)[0]


# ---------------------------------------------------------------------------
# the fixed subset
# ---------------------------------------------------------------------------


def cases(counts: Dict[str, int]) -> List[Dict[str, Any]]:
    out = []
    for cat, n in counts.items():
        for i in range(n):
            out.append({"id": f"p3-{cat[:3]}{i:04d}", "meta": {"category": cat}})
    return out


REAL = {"coding": 259, "sentinel/daemon": 34, "other": 32, "memory/wiki": 4}


def test_allocation_is_largest_remainder_on_the_real_mix() -> None:
    a = PT.allocate(REAL, 100)
    assert a == {"coding": 79, "sentinel/daemon": 10, "other": 10, "memory/wiki": 1}
    assert sum(a.values()) == 100


@pytest.mark.parametrize("counts,n", [({"a": 1, "b": 1, "c": 1}, 2), ({"a": 5}, 9), ({"a": 3, "b": 0}, 2),
                                      ({"a": 7, "b": 2, "c": 1}, 10)])
def test_allocation_sums_and_never_exceeds_a_stratum(counts: Dict[str, int], n: int) -> None:
    a = PT.allocate(counts, n)
    assert sum(a.values()) == min(n, sum(counts.values()))
    assert all(a[k] <= counts[k] for k in counts)


def test_subset_is_deterministic_stratified_and_seeded() -> None:
    cs = cases(REAL)
    s1 = PT.select_subset(cs, 100, "seed-a")
    shuffled = cs[:]
    random.Random(7).shuffle(shuffled)
    s2 = PT.select_subset(shuffled, 100, "seed-a")
    assert sorted(s1["ids"]) == sorted(s2["ids"])  # the same ids whatever the file order
    assert s1["ids"] == PT.select_subset(cs, 100, "seed-a")["ids"]  # and the same order for the same file
    assert len(s1["ids"]) == 100 == s1["n"]
    assert {k: v["chosen"] for k, v in s1["strata"].items()} == PT.allocate(REAL, 100)
    by_cat = {c["id"]: c["meta"]["category"] for c in cs}
    got = {k: sum(1 for i in s1["ids"] if by_cat[i] == k) for k in REAL}
    assert got == PT.allocate(REAL, 100)
    assert PT.select_subset(cs, 100, "seed-b")["ids"] != s1["ids"]
    # within a stratum: the smallest sha256(seed, id)
    coding = sorted((i for i in by_cat if by_cat[i] == "coding"), key=lambda i: PT.subset_key("seed-a", i))
    assert set(coding[:79]) == {i for i in s1["ids"] if by_cat[i] == "coding"}


def test_subset_refuses_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        PT.select_subset([{"id": "x"}, {"id": "x"}], 1)


def test_committed_subset_file_matches_the_selection() -> None:
    ref = _F.parent.parent.parent / "docs" / "sprints" / PT.SUBSET_FILE
    if not ref.is_file():
        pytest.skip("no committed subset file")
    sub = json.loads(ref.read_text(encoding="utf-8"))
    assert sub["seed"] == PT.SUBSET_SEED and sub["n"] == len(sub["ids"]) == PT.SUBSET_N
    assert sub["source_sha256"] == PT.EVAL_TOOLS_SHA
    assert {k: v["chosen"] for k, v in sub["strata"].items()} == PT.allocate(
        {k: v["cases"] for k, v in sub["strata"].items()}, PT.SUBSET_N)


def test_subset_rate_counts_only_scored_ids() -> None:
    r = PT.subset_rate({"a": True, "b": False, "c": True}, ["a", "b", "z"])
    assert r == {"k": 1, "n": 2, "missing": 1, "rate": 0.5}
    assert PT.subset_rate(None, ["a"])["rate"] is None


# ---------------------------------------------------------------------------
# the 4-bit flag
# ---------------------------------------------------------------------------


def test_flag_when_q5_beats_q4_by_more_than_3_points_on_tools() -> None:
    flag, line = PT.bottleneck_flag({"eval-tools": 80.0, "sanity": 96.0}, {"eval-tools": 83.5, "sanity": 96.0})
    assert flag and line.startswith("FLAG: 4-bit is the bottleneck") and "eval-tools task success" in line


def test_flag_on_sanity_alone() -> None:
    flag, line = PT.bottleneck_flag({"eval-tools": 80.0, "sanity": 90.0}, {"eval-tools": 80.0, "sanity": 93.4})
    assert flag and "sanity" in line


def test_exactly_3_points_or_q4_ahead_is_no_flag() -> None:
    assert not PT.bottleneck_flag({"eval-tools": 80.0, "sanity": 90.0}, {"eval-tools": 83.0, "sanity": 93.0})[0]
    flag, line = PT.bottleneck_flag({"eval-tools": 85.0, "sanity": 97.0}, {"eval-tools": 80.0, "sanity": 90.0})
    assert not flag and line.startswith("No 4-bit flag") and "-5.0 pts" in line


def test_flag_with_a_missing_score_is_not_raised() -> None:
    flag, line = PT.bottleneck_flag({"eval-tools": None, "sanity": 90.0}, {"eval-tools": 99.0, "sanity": 90.0})
    assert not flag and "not both scored" in line


# ---------------------------------------------------------------------------
# deadlines
# ---------------------------------------------------------------------------


def test_deadline_states() -> None:
    hard = PT.parse_iso(HARD_AT)
    assert PT.deadline_state(T_STOP - 1, T_STOP, hard) == "run"
    assert PT.deadline_state(T_STOP, T_STOP, hard) == "stop"
    assert PT.deadline_state(hard + 1, T_STOP, hard) == "hard"
    assert PT.may_start(T_STOP - 3600, T_STOP, 60)
    assert not PT.may_start(T_STOP - 3599, T_STOP, 60)


def test_parse_iso_needs_an_offset() -> None:
    with pytest.raises(ValueError):
        PT.parse_iso("2026-11-02T05:00:00")
    assert PT.parse_iso("2026-11-02T05:00:00-05:00") == PT.parse_iso("2026-11-02T10:00:00+00:00")


# ---------------------------------------------------------------------------
# scrubbing and the report
# ---------------------------------------------------------------------------


def test_scrub_and_its_check() -> None:
    t = PT.scrub("served `/home/someone/distill/models/x.gguf` at 127.0.0.1:18080")
    assert "/home/" not in t and "~/distill" in t
    assert PT.scrub_problems(t, ["AcmeBiz"]) == []
    assert PT.scrub_problems("ip 192.0.2.7", []) == ["an IP address (192.0.2.7)"]
    assert PT.scrub_problems("the acmebiz server", ["AcmeBiz"]) == ["a banned name"]
    assert PT.scrub_problems("/Users/x/y", []) == ["a host home path"]
    assert PT.scrub_problems("version 1.2.3.4567", []) == []


def summary(label: str, sets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"label": label, "sets": {}}
    for sname, spec in sets.items():
        oc = spec.get("outcomes", {})
        k, n = sum(oc.values()), len(oc)
        metric = {"eval-tools": "task_success", "sanity": "sanity_score",
                  "eval-injection": "injection_resistance", "eval-longctx": "recall"}[sname]
        out["sets"][sname] = {"metrics": {metric: {"k": k, "n": n, "rate": k / n if n else None,
                                                   "ci_low": 0.0, "ci_high": 1.0}},
                              "outcomes": oc}
    return out


def report_data(q4_tools: int, q5_tools: int) -> Dict[str, Any]:
    ids = [f"c{i}" for i in range(10)]
    def tools(k: int) -> Dict[str, bool]:
        return {i: n < k for n, i in enumerate(ids)}
    return {
        "run": "9b-run-max-derived", "labels": {"q4": "L4", "q5": "L5", "bf16": "LB", "recipe": "LR"},
        "summaries": {"q4": summary("L4", {"eval-tools": {"outcomes": tools(q4_tools)},
                                           "sanity": {"outcomes": {"s1": True}}}),
                      "q5": summary("L5", {"eval-tools": {"outcomes": tools(q5_tools)},
                                           "sanity": {"outcomes": {"s1": True}}}),
                      "bf16": summary("LB", {"eval-tools": {"outcomes": tools(9)}})},
        "base9b": summary("base-9b", {"eval-tools": {"outcomes": tools(7)}}), "saluki": None,
        "subset": {"ids": ids[:5], "n": 5, "seed": "s", "strata": {"coding": {"cases": 10, "chosen": 5}},
                   "source_cases": 10, "method": "m", "source_sha256": "x"},
        "state": {"items": {"c": {"ctx": 65536, "ctx_failed": [262144, 131072]}}}, "gates": {},
        "generated": "now", "server_flags": "-c 262144"}


def test_report_carries_the_flag_line_and_the_subset_columns() -> None:
    d = report_data(q4_tools=6, q5_tools=10)  # 60% vs 100%: +40 pts
    text = PT.render_report(d)
    assert "**FLAG: 4-bit is the bottleneck" in text
    assert "| eval-tools task success, 5-case subset | 100.0% (5/5) | 100.0% (5/5) | 100.0% (5/5)" in text
    assert "BF16 server context: 65536" in text
    head = PT.headline_lines(d)
    assert len(head) <= 5 and head[2].startswith("FLAG: 4-bit is the bottleneck")
    d2 = report_data(q4_tools=8, q5_tools=9)  # +10 pts on 10 cases: still a flag; same counts: none
    assert "FLAG" in PT.render_report(d2)
    d3 = report_data(q4_tools=9, q5_tools=9)
    assert "**No 4-bit flag" in PT.render_report(d3)


# ---------------------------------------------------------------------------
# The flow, against a fake 4090 host
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += max(s, 1.0)


class FakeTower:
    """Answers the pipeline's shell snippets by what they contain; records every command."""

    def __init__(self, tmp: Path, manifest: Dict[str, Any]) -> None:
        self.tmp = tmp
        self.units: Dict[str, str] = {"llama-server.service": "active"}
        self.manifest = manifest
        self.cmds: List[str] = []
        self.server_deaths = 0  # how many scoring servers die right after their first set starts
        self.rc_files: Dict[str, str] = {}
        self.ctx: Dict[str, int] = {}

    def __call__(self, script: str) -> Tuple[int, str]:
        self.cmds.append(script)
        if script == "echo $HOME":
            return 0, "/home/t"
        m = re.match(r"systemctl --user is-active (\S+)$", script)
        if m:
            u = m.group(1).strip("'")
            st = self.units.get(u, "inactive")
            if u.startswith("distill-p7-score-") and st == "active":
                # a scoring unit finishes on its first poll, unless its server is to die
                srv = next((k for k, v in self.units.items() if k.startswith("distill-p7-srv-") and v == "active"), None)
                if self.server_deaths and srv:
                    self.server_deaths -= 1
                    self.units[srv] = "failed"
                    return 0, "active"
                self.units[u] = "inactive"
                self.rc_files[u] = "0"
            return 0, st
        if "echo biz=" in script:
            used = 0 if self.units.get("llama-server.service") != "active" else 20000
            if any(v == "active" for k, v in self.units.items() if k.startswith("distill-p7-srv-")):
                used = 9000
            return 0, f"biz={self.units.get('llama-server.service', 'inactive')}\nfree_gb=60.0\nvram={used}\n"
        if script.startswith("cat ") and "manifest.json" in script and "phase5" in script:
            return 0, json.dumps(self.manifest)
        if script.startswith("cat ") and "run_state.json" in script:
            return 0, "{}"
        if script.startswith("test -s") or script.startswith("test -e") or script.startswith("test -d"):
            return 0, ""
        if "flock -w 1800" in script:
            self.units["llama-server.service"] = "inactive"
            return 0, "before=active\nstopped=yes\nfreed=0\n"
        if "systemd-run" in script:
            u = re.search(r"--unit=(\S+)", script).group(1).strip("'")
            self.units[f"{u}.service"] = "active"
            if "restore" in u:
                self.units[f"{u}.service"] = "inactive"
            m = re.search(r" -c (\d+) ", script)
            if m and "-srv-" in u:
                self.ctx[u] = int(m.group(1))
            return 0, ""
        m = re.match(r"systemctl --user stop (\S+)", script)
        if m:
            self.units[m.group(1).strip("'")] = "inactive"
            return 0, ""
        if script.startswith("python3 -c"):
            srv = next((k for k, v in self.units.items() if k.startswith("distill-p7-srv-") and v == "active"), "")
            return 0, f"health=200\nn_ctx={self.ctx.get(srv[:-8], 0)}\nmodel=x\n"
        if script.startswith("ss -ltn"):
            return 0, "free"
        if script.startswith("cat ") and script.rstrip().endswith("2>/dev/null"):
            if "restore.rc" in script:
                return 0, "0"
            return 0, "0"
        if script.startswith("cat ") and "verify.last" in script:
            return 0, "PASS"
        if "sed -n" in script and "BIZ_NAME" in script:
            return 0, "TestBiz"
        return 0, ""


class FakeOps:
    def __init__(self, cfg: Any, tower: FakeTower) -> None:
        self.cfg, self.t = cfg, tower
        self.here_cmds: List[List[str]] = []

    def tower(self, script: str, *, mut: bool = False, timeout: float = 120, stdin: Optional[str] = None
              ) -> Tuple[int, str]:
        if mut and not self.cfg.execute:
            return 0, ""
        return self.t(script)

    def here(self, argv: Any, *, mut: bool = False, timeout: float = 600, env: Any = None) -> Tuple[int, str]:
        self.here_cmds.append(list(argv))
        return 0, ""

    def rsync(self, srcs: Any, dst: str, *, timeout: float = 7200) -> Tuple[int, str]:
        self.here_cmds.append(["rsync", *srcs, dst])
        return 0, ""

    def remote(self, rel: str) -> str:
        return f"tower:{rel}"

    @property
    def thome(self) -> str:
        return "/home/t"


def make(tmp: Path, manifest: Dict[str, Any], now: float, **over: Any) -> Tuple[Any, FakeTower, Clock, List[str]]:
    ns = argparse.Namespace(
        execute=True, tower="u@t", fake_root="", run="9b-run-max-derived", adapter_name="adapter-max-derived",
        trainer_unit="distill-train-full-x.service", watcher_unit="flywheel-train-watch-x.service",
        stop_at=STOP_AT, hard_at=HARD_AT, poll=60.0, dry_adapter="", state_dir=str(tmp / "p7"),
        notify=str(tmp / "notify.sh"), here_python="/usr/bin/python3", unit_prefix="", restore_prefix="",
        biz_unit="", port=18080, subset_n=100, ids_file=str(tmp / "none.json"), phase6_cmd="", skip_recipe=False)
    for k, v in over.items():
        setattr(ns, k, v)
    cfg = PT.Cfg(ns)
    tower = FakeTower(tmp, manifest)
    clock = Clock(now)
    pipe = PT.Pipeline(cfg, ops=None, clock=clock, sleep=clock.sleep)
    pipe.ops = FakeOps(cfg, tower)
    sent: List[str] = []

    def notify(key: str, text: str) -> bool:
        if key in pipe.state["telegrams"]:
            return True
        assert len(text.splitlines()) <= 5
        sent.append(text)
        pipe.state["telegrams"][key] = "now"
        return True

    pipe.notify = notify  # type: ignore[method-assign]
    return pipe, tower, clock, sent


def gpu_commands(tower: FakeTower) -> List[str]:
    return [c for c in tower.cmds if "systemd-run" in c or "systemctl --user stop" in c or "flock" in c]


def test_failed_training_telegrams_and_never_touches_the_gpu(tmp_path: Path) -> None:
    m = done_manifest(status="failed", reason="diverged: 8 steps in a row with a non-finite loss")
    pipe, tower, _, sent = make(tmp_path, m, SUN_10)
    assert pipe.run() == 0
    assert gpu_commands(tower) == []
    assert len(sent) == 1 and "NOT run" in sent[0] and "diverged" in sent[0]
    done = json.loads((tmp_path / "p7" / "PIPELINE-DONE.json").read_text())
    assert done["outcome"].startswith("not-run")
    assert "take-gpu" not in pipe.state["steps"]
    assert not pipe.state.get("gpu_taken")


def test_training_still_running_at_the_stop_time_is_not_run(tmp_path: Path) -> None:
    pipe, tower, clock, sent = make(tmp_path, done_manifest(), T_STOP - 120)
    tower.units["distill-train-full-x.service"] = "active"
    assert pipe.run() == 0
    assert gpu_commands(tower) == []
    assert "not-run" in pipe.state["outcome"] and "had not ended" in pipe.state["outcome"]


def test_waits_for_the_watchers_restore_before_stopping_business(tmp_path: Path) -> None:
    pipe, tower, clock, _ = make(tmp_path, done_manifest(), SUN_10)
    tower.units["flywheel-train-watch-x.service"] = "active"
    calls = {"n": 0}
    orig = tower.__call__

    def watcher_ends_later(script: str) -> Tuple[int, str]:
        if "flywheel-train-watch-x.service" in script:
            calls["n"] += 1
            if calls["n"] >= 3:
                tower.units["flywheel-train-watch-x.service"] = "inactive"
        return orig(script)

    pipe.ops.t = watcher_ends_later  # type: ignore[assignment]
    pipe.step_wait_train()
    assert calls["n"] >= 3
    assert not any("flock" in c for c in tower.cmds)
    pipe.step_verdict()
    pipe.step_take_gpu()
    assert pipe.state["gpu_taken"] and pipe.state["biz_down_by_us"]
    assert any("flock -w 1800" in c and "systemctl --user stop llama-server.service" in c for c in tower.cmds)


def test_server_crash_is_retried_once_then_the_item_is_skipped(tmp_path: Path) -> None:
    pipe, tower, _, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "inactive"
    tower.server_deaths = 1
    assert pipe.score_item("a", "q4", ("eval-tools", "sanity")) == "done"
    assert pipe.state["items"]["a"]["crashes"] == 1
    starts = [c for c in tower.cmds if "systemd-run" in c and "-srv-a-" in c]
    assert len(starts) == 2  # restarted once
    tower.server_deaths = 2
    assert pipe.score_item("b", "q5", ("eval-tools", "sanity")) == "skipped"
    assert "crashed twice" in pipe.state["items"]["b"]["why"]
    # the next item still runs
    assert pipe.score_item("d", "q4", ("eval-injection", "eval-longctx")) == "done"
    # never two scoring servers at once: every start found the previous one stopped
    assert all(v != "active" for k, v in tower.units.items() if k.startswith("distill-p7-srv-"))


def test_scoring_uses_base9bs_server_flags(tmp_path: Path) -> None:
    pipe, tower, _, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "inactive"
    pipe.score_item("a", "q4", ("eval-tools",))
    start = next(c for c in tower.cmds if "systemd-run" in c and "-srv-a-" in c)
    want = ("-c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja -ngl 99 -fit off -b 2048 -ub 2048 "
            "--host 127.0.0.1 --port 18080 --alias 9b-run-max-derived-Q4_K_M")
    assert want in start and "/home/t/llama.cpp/build/bin/llama-server" in start
    score = next(c for c in tower.cmds if "systemd-run" in c and "-score-a-" in c)
    assert "--concurrency 1" in score and "/home/t/distill/evalsets/eval-tools.jsonl" in score
    assert "/home/t/distill/code/scripts/distill/score.py" in score


def test_bf16_steps_down_the_context_ladder_without_counting_a_crash(tmp_path: Path) -> None:
    pipe, tower, _, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "inactive"
    orig = tower.__call__
    fail_ctx = {262144, 131072}

    def oom_at_big_ctx(script: str) -> Tuple[int, str]:
        rc, out = orig(script)
        if "systemd-run" in script and "-srv-c-" in script:
            ctx = int(re.search(r" -c (\d+) ", script).group(1))
            if ctx in fail_ctx:
                u = re.search(r"--unit=(\S+)", script).group(1).strip("'")
                tower.units[f"{u}.service"] = "failed"
        return rc, out

    pipe.ops.t = oom_at_big_ctx  # type: ignore[assignment]
    assert pipe.score_item("c", "bf16", ("eval-tools-subset",)) == "done"
    it = pipe.state["items"]["c"]
    assert it["ctx"] == 65536 and it["ctx_failed"] == [262144, 131072] and it["crashes"] == 0
    score = next(c for c in tower.cmds if "systemd-run" in c and "-score-c-" in c)
    assert "eval-tools-subset100.jsonl" in score


def test_deadline_during_scoring_stops_ours_and_restores(tmp_path: Path) -> None:
    pipe, tower, clock, _ = make(tmp_path, done_manifest(), T_STOP - 20 * 60)
    for s in ("wait-train", "verdict", "take-gpu", "phase6", "subset"):
        pipe.state["steps"][s] = {"status": "done"}
    pipe.state["gpu_taken"] = True
    pipe.state["biz_down_by_us"] = True
    tower.units["llama-server.service"] = "inactive"
    orig = tower.__call__

    def slow_scoring(script: str) -> Tuple[int, str]:
        m = re.match(r"systemctl --user is-active (\S+)$", script)
        if m and m.group(1).strip("'").startswith("distill-p7-score-"):
            clock.t += 600  # each poll of a scoring unit: ten minutes pass
            tower.cmds.append(script)
            return 0, "active"
        return orig(script)

    pipe.ops.t = slow_scoring  # type: ignore[assignment]
    pipe.step_report = lambda: None  # type: ignore[method-assign]
    assert pipe.run() == 0
    assert pipe.state["outcome"].startswith("stopped: the stop time")
    assert any("systemctl --user stop" in c and "-score-a-" in c for c in tower.cmds)
    assert any("systemctl --user stop" in c and "-srv-a-" in c for c in tower.cmds)
    restore = [c for c in tower.cmds if "systemd-run" in c and "flywheel-p7-restore-" in c]
    assert len(restore) == 1 and "train_end.sh" in restore[0]
    assert pipe.state["released"] and pipe.state["restore"]["ok"]
    # the restore unit is not distill-*: apply business must not stop it
    assert "--unit=distill" not in restore[0]


def test_business_back_from_elsewhere_stops_gpu_work(tmp_path: Path) -> None:
    pipe, tower, clock, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "active"
    with pytest.raises(PT.Stop, match="restored elsewhere"):
        pipe.guard(force=True)


def test_disk_pressure_stops(tmp_path: Path) -> None:
    pipe, tower, clock, _ = make(tmp_path, done_manifest(), SUN_10)
    orig = tower.__call__

    def full(script: str) -> Tuple[int, str]:
        rc, out = orig(script)
        return rc, out.replace("free_gb=60.0", "free_gb=9.5")

    pipe.ops.t = full  # type: ignore[assignment]
    with pytest.raises(PT.Stop, match="disk pressure"):
        pipe.guard(force=True)


def test_no_flag_present_means_not_run_before_taking_the_gpu(tmp_path: Path) -> None:
    pipe, tower, clock, sent = make(tmp_path, done_manifest(), SUN_10)
    orig = tower.__call__

    def flagged(script: str) -> Tuple[int, str]:
        rc, out = orig(script)
        return (rc, out + "flag=distill/window/NO-SWAPS-TONIGHT\n") if "echo biz=" in script else (rc, out)

    pipe.ops.t = flagged  # type: ignore[assignment]
    with pytest.raises(PT.NotRun, match="NO-SWAPS"):
        pipe.step_take_gpu()
    assert not pipe.state.get("gpu_taken") and not any("flock" in c for c in tower.cmds)


def test_release_runs_once_and_only_after_taking_the_gpu(tmp_path: Path) -> None:
    pipe, tower, _, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.release("x")
    assert not [c for c in tower.cmds if "restore" in c]
    pipe.state["gpu_taken"] = True
    pipe.release("x")
    pipe.release("x")
    assert len([c for c in tower.cmds if "systemd-run" in c and "restore" in c]) == 1


def test_resume_after_a_stop_by_hand_takes_the_gpu_again(tmp_path: Path) -> None:
    pipe, tower, _, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state.update(released=True, gpu_taken=True, biz_down_by_us=True)
    pipe.state["steps"] = {"take-gpu": {"status": "done"}, "phase6": {"status": "done"}}
    pipe.state["telegrams"] = {"down": "t"}
    pipe.state["items"] = {"a": {"status": "running", "sets": {}}}
    pipe.resume_after_release()
    assert "take-gpu" not in pipe.state["steps"] and pipe.state["steps"]["phase6"]["status"] == "done"
    assert not pipe.state.get("released") and "down" not in pipe.state["telegrams"]
    assert pipe.state["items"]["a"]["status"] == "pending"


def test_arm_names_a_unit_no_restore_stops(tmp_path: Path, capsys: Any) -> None:
    rc = PT.main(["arm", "--tower", "u@t", "--state-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    unit = re.search(r"--unit=(\S+)", out).group(1)
    assert unit.startswith("flywheel-post-train-") and not unit.startswith("distill-")
    cmd = out.splitlines()[0]
    assert "Restart=on-failure" in cmd and " run " in cmd and cmd.count("--execute") == 1
    assert "RuntimeMaxSec" not in cmd  # a timeout would count as a failure and restart it


def test_q5_crash_restarts_one_step_down_the_ladder(tmp_path: Path) -> None:
    pipe, tower, _, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "inactive"
    tower.server_deaths = 1
    assert pipe.score_item("b", "q5", ("eval-tools", "sanity")) == "done"
    it = pipe.state["items"]["b"]
    assert it["crashes"] == 1 and it["ctx_failed"] == [262144] and it["ctx"] == 131072
    # Q4_K_M (the gate's candidate) never steps down: like for like with base-9b at 262144
    assert PT.CTX_LADDER["q4"] == (262144,) and PT.CTX_LADDER["recipe"] == (262144,)


def test_phase6_retries_then_lowers_the_gpu_layers_then_stops(tmp_path: Path) -> None:
    pipe, tower, _, sent = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "inactive"
    seen: List[List[str]] = []

    def failing_child(argv: Any, log_path: Path, env: Any, on_stop: Any) -> int:
        seen.append(list(argv))
        return 2

    pipe.run_child = failing_child  # type: ignore[method-assign]
    with pytest.raises(PT.Stop, match="Phase 6 failed 3 times"):
        pipe.step_phase6()
    assert len(seen) == 3
    assert "--tower-ngl" not in seen[0] and "--tower-ngl" not in seen[1]
    assert seen[2][-2:] == ["--tower-ngl", "20"]
    assert all("--keep-bf16" in a and "--stop-at" in a and a[a.index("--stop-at") + 1] == STOP_AT for a in seen)
    assert any("Phase 6 FAILED" in m for m in sent)


def test_phase6_pause_exit_waits_and_resumes(tmp_path: Path) -> None:
    pipe, tower, clock, _ = make(tmp_path, done_manifest(), SUN_10)
    pipe.state["gpu_taken"] = True
    tower.units["llama-server.service"] = "inactive"
    rcs = [75, 99, 0]

    def child(argv: Any, log_path: Path, env: Any, on_stop: Any) -> int:
        return rcs.pop(0)

    pipe.run_child = child  # type: ignore[method-assign]
    t0 = clock.t
    pipe.step_phase6()
    assert rcs == [] and clock.t - t0 >= 600  # two 5-min waits
    assert pipe.state["steps"]["phase6"]["status"] == "done"
    assert any(c.startswith("test -s") and "Q5_K_M.gguf" in c for c in tower.cmds)
