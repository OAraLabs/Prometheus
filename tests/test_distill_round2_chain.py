"""scripts/distill/mini_window/round2_chain.py (+ gen_runner's round-2 keys): the decisions, against fakes.

Nothing here touches systemd, ssh, a GPU or Telegram: the 4090, the units and the clock are stand-ins.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
MW_DIR = REPO / "scripts" / "distill" / "mini_window"
sys.path.insert(0, str(MW_DIR))

import gen_runner as G  # noqa: E402
import mini_window as MW  # noqa: E402
import round2_chain as R  # noqa: E402

ET = ZoneInfo("America/New_York")


def t(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def conf(tmp_path: Path, extra: str = "") -> dict[str, str]:
    p = tmp_path / "round2.conf"
    p.write_text("R2_TOWER_SSH=u@tower\nR2_TOWER_HOST=tower\nR2_TOWER_URL=http://tower:18094\n"
                 f"R2_HOME={tmp_path}/r2\nR2_RUNS={tmp_path}/runs\nR2_CODE={tmp_path}/code\n"
                 f"R2_TEACHER_CTX_JSON={tmp_path}/teacher-ctx.json\n" + extra)
    return R.load_conf(p, home=str(tmp_path / "home"))


# ── settings ─────────────────────────────────────────────────────────────────


def test_conf_expands_home_and_passes_gen_keys(tmp_path: Path) -> None:
    c = conf(tmp_path, "R2_LOCAL_EXTRA_ARGS=--skip-done-in ~/x/*.jsonl --coding-cap 0.2\nGEN_HARVEST_EVERY_MIN=15\n")
    home = str(tmp_path / "home")
    assert c["R2_POOL"] == f"{home}/distill/phase2b/pool"
    assert c["R2_LOCAL_EXTRA_ARGS"] == f"--skip-done-in {home}/x/*.jsonl --coding-cap 0.2"
    assert R.remote_overrides(c, t("2026-11-02T05:18:00-05:00"))["GEN_HARVEST_EVERY_MIN"] == "15"
    assert R.local_overrides(c)["GEN_HARVEST_EVERY_MIN"] == "15"
    bad = tmp_path / "bad.conf"
    bad.write_text("lower=1\n")
    with pytest.raises(SystemExit):
        R.load_conf(bad)


def test_derived_conf_round_trips_through_the_window_loader(tmp_path: Path) -> None:
    c = conf(tmp_path, "R2_TOWER_EXTRA_ARGS=--mix error-recovery=0.5,chat-mode=0.5\n")
    base = ("LOCAL_MODEL_UNIT=m.service\nPROBE_UNIT=p.service\nOLLAMA_MODEL=j:7b\n"
            "GEN_TASKS=$HOME/old.jsonl\nGEN_EVAL_TOOLS=/a/one.jsonl /b/two.jsonl\n")
    until = t("2026-11-02T05:18:00-05:00")
    text = R.derived_conf(base, R.remote_overrides(c, until), "test")
    p = tmp_path / "derived.conf"
    p.write_text(text)
    w = MW.load_conf(p, home="/h")
    assert w["GEN_EVAL_TOOLS"] == "/a/one.jsonl /b/two.jsonl", "untouched keys keep their value"
    assert w["GEN_REMOTE_TASKS"].endswith("real-v3-excl.jsonl"), "default: the real-v3-excl taskset"
    assert w["GEN_REMOTE_URL"] == "http://tower:18094" and w["GEN_REMOTE_HOST"] == "tower"
    assert w["GEN_REMOTE_UNTIL"] == "2026-11-02T05:18:00-05:00"
    assert w["GEN_REMOTE_NAME"].startswith("r2-tower-tower-ud-q4_k_xl-")
    extra = w["GEN_REMOTE_EXTRA_ARGS"].split()
    assert extra[-2:] == ["--mix", "error-recovery=0.5,chat-mode=0.5"]
    assert f"{tmp_path}/runs/r2-tower-*/*.tasks.jsonl" in extra
    loc = MW.load_conf(_write(tmp_path / "l.conf", R.derived_conf(base, R.local_overrides(c), "t")), home="/h")
    assert loc["GEN_TASKS"].endswith("real-v3-excl.jsonl"), "a later line wins over window.conf's"
    assert loc["GEN_CODE"] == f"{tmp_path}/code" and loc["GEN_LOCAL_NAME"].startswith("local-r2-{alias}-")


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def test_run_names_change_with_code_taskset_and_flags(tmp_path: Path) -> None:
    c = conf(tmp_path)
    a = R.tower_run_name(c)
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "COMMIT").write_text("abc123\n")
    b = R.tower_run_name(c)
    assert a != b and a.endswith(R.run_hash("nocode", c["R2_TOWER_TASKS"], ""))
    assert R.tower_run_name(dict(c, R2_TOWER_EXTRA_ARGS="--coding-cap 0.1")) != b
    assert R.tower_run_name(dict(c, R2_TOWER_TASKS="/x.jsonl")) != b
    assert R.tower_run_name(dict(c, R2_TOWER_RUN_NAME="My Run")) == "my_run"
    assert R.local_run_name(dict(c, R2_LOCAL_NAME="local-x-{alias}")) == "local-x-{alias}"


# ── the serve profile ────────────────────────────────────────────────────────


def test_parse_profile_matches_serve_teachers_limits() -> None:
    assert R.parse_profile("4x32768") == (4, 32768)
    assert R.parse_profile(" 1X131072 ") == (1, 131072)
    for bad in ("9x32768", "0x32768", "4x4096", "4x32769", "4*32768", "longctx", ""):
        assert R.parse_profile(bad) is None, bad


@pytest.mark.parametrize("data,want", [
    ({"gpu-tower-4090": {"UD-Q4_K_XL": {"max_stable_ctx": 147456, "max_ctx_tried": 196608},
                    "IQ3_XXS": {"max_stable_ctx": 200000}}, "mini": {"IQ3_XXS": {"max_stable_ctx": 57344}}}, 147456),
    ([{"host": "gpu-tower-4090", "quant": "UD-Q4_K_XL", "max_context": 131072},
      {"host": "gpu-tower-4090", "quant": "IQ3_XXS", "max_context": 262144}], 131072),
    ({"results": [{"host": "tower-4090", "quant": "ud-q4_k_xl", "ctx": 65536, "stable": True,
                   "max_stable_ctx": 98304, "max_failed_ctx": 163840}]}, 98304),
    ({"gpu-tower-4090": {"UD-Q4_K_XL": {"max_ctx": 120000, "stable_ctx": 110000}}}, 110000),
    ({"mini-host": {"UD-Q4_K_XL": {"max_stable_ctx": 90000}}}, None),
    ({"gpu-tower-4090": {"UD-Q4_K_XL": {"max_stable_ctx": "big"}}}, None),
])
def test_measured_ctx_reads_any_reasonable_shape(data: Any, want: int | None) -> None:
    assert R.measured_ctx(data, "4090", "q4_k_xl") == want


def test_resolve_profiles(tmp_path: Path) -> None:
    c = conf(tmp_path)
    assert R.resolve_profiles(c)[0] == ["4x32768"]
    lc = dict(c, R2_TOWER_PROFILE="longctx")
    profs, why = R.resolve_profiles(lc)
    assert profs == ["4x32768"] and "unreadable" in why, "no file: today's 4x32K"
    (tmp_path / "teacher-ctx.json").write_text(json.dumps({"gpu-tower-4090": {"UD-Q4_K_XL": {"max_stable_ctx": 150000}}}))
    assert R.resolve_profiles(dict(lc, R2_LONGCTX_CAP="131072"))[0] == ["1x131072", "4x32768"], \
        "capped at R2_LONGCTX_CAP, standard as the fallback"
    assert R.resolve_profiles(lc)[0] == ["1x149504", "4x32768"], "floored to 1024 (default cap 262144)"
    (tmp_path / "teacher-ctx.json").write_text(json.dumps({"gpu-tower-4090": {"UD-Q4_K_XL": {"max_stable_ctx": 20000}}}))
    assert R.resolve_profiles(lc)[0] == ["4x32768"]
    (tmp_path / "teacher-ctx.json").write_text("{}")
    assert R.resolve_profiles(lc)[0] == ["4x32768"]
    assert R.resolve_profiles(dict(lc, R2_LONGCTX_CTX="65536"))[0] == ["1x65536", "4x32768"]
    assert R.resolve_profiles(dict(c, R2_TOWER_PROFILE="2x65536"))[0] == ["2x65536", "4x32768"]
    assert R.resolve_profiles(dict(c, R2_TOWER_PROFILE="4x32768"))[0] == ["4x32768"]
    assert R.resolve_profiles(dict(c, R2_TOWER_PROFILE="huge"))[0] == ["4x32768"]


# ── the Sunday decisions ─────────────────────────────────────────────────────


def test_pipeline_verdict() -> None:
    now, latest = t("2026-11-01T15:00:00-05:00"), t("2026-11-02T04:00:00-05:00")
    assert R.pipeline_verdict({"outcome": "completed"}, "inactive", now, latest)[0] == "go"
    assert R.pipeline_verdict({"outcome": "completed; items not done: e2"}, None, now, latest)[0] == "go"
    for bad in ("not-run: verdict: val loss rose", "stopped: NO-SWAPS", "gave-up", "interrupted: completed", ""):
        assert R.pipeline_verdict({"outcome": bad}, "inactive", now, latest)[0] == "stop", bad
    assert R.pipeline_verdict(None, "active", now, latest)[0] == "wait"
    assert R.pipeline_verdict(None, "activating", now, latest)[0] == "wait", "a restart after a crash is not an end"
    v, why = R.pipeline_verdict(None, "failed", now, latest)
    assert v == "stop" and "without PIPELINE-DONE" in why
    assert R.pipeline_verdict(None, "gone", now, latest)[0] == "stop"
    assert R.pipeline_verdict(None, None, now, latest)[0] == "wait"
    assert R.pipeline_verdict(None, "active", latest, latest)[0] == "stop"
    # a marker wins even when it came with the latest-start time passed
    assert R.pipeline_verdict({"outcome": "completed"}, "inactive", latest + 1, latest)[0] == "go"


def test_capacity_verdict() -> None:
    end = t("2026-11-01T15:30:00-05:00")
    latest = t("2026-11-02T04:00:00-05:00")
    assert R.capacity_verdict(True, end, end + 60, 3 * 3600, latest)[0] == "go"
    assert R.capacity_verdict(False, end, end + 3600, 3 * 3600, latest)[0] == "wait"
    assert R.capacity_verdict(False, end, end + 3 * 3600, 3 * 3600, latest)[0] == "timeout"
    assert R.capacity_verdict(False, end, latest, 30 * 3600, latest)[0] == "stop"
    assert R.marker_end({"finished": "2026-11-01T15:30:00-05:00"}, 0.0) == end
    assert R.marker_end({"finished": None}, 123.0) == 123.0


def test_today_at_follows_the_zone_across_the_fall_back() -> None:
    mon = t("2026-11-02T02:00:00-05:00")
    assert R.utc_cal(R.today_at("06:22", mon)) == "2026-11-02 11:22:00 UTC"
    sat = t("2026-10-31T02:00:00-04:00")
    assert R.utc_cal(R.today_at("06:22", sat)) == "2026-10-31 10:22:00 UTC"
    assert R.utc_cal(R.when("2026-11-02T05:20:00-05:00")) == "2026-11-02 10:20:00 UTC"


def test_parse_status() -> None:
    st = R.parse_status("x\njobs_unit=distill-jobs-r2teacher@20261101-1630-1.service\n"
                        "timer flywheel-r2-end=1793600000\ntimer flywheel-window-start=-\n"
                        'start_result={"status": "ok", "why": "a=b"}\n2026-11-01 round2_status: end (rc 0)\n')
    assert st["jobs_unit"].startswith("distill-jobs-r2teacher@")
    assert st["timer flywheel-r2-end"] == "1793600000" and st["timer flywheel-window-start"] == "-"
    assert json.loads(st["start_result"])["why"] == "a=b"


# ── stats ────────────────────────────────────────────────────────────────────


def test_gen_stats_counts_kept_rows_in_the_period(tmp_path: Path) -> None:
    now = t("2026-11-01T09:00:00-05:00")
    runs = tmp_path / "runs"
    rows = {
        "local-r2-iq3-1/qwen38-27b-gym.tasks.jsonl": [
            {"status": "kept", "rows": 2, "kind": "chat-mode", "domain": "chat-mode", "ts": now - 3600},
            {"status": "rejected", "rows": 0, "kind": "agentic", "domain": "other", "ts": now - 3600},
            {"status": "held", "rows": 1, "kind": "agentic", "domain": "other", "ts": now - 3600},
            {"status": "kept", "rows": 1, "kind": "coding", "domain": "coding", "ts": now - 90000}],
        "r2-tower-x/qwen38-27b-gym.tasks.jsonl": [
            {"status": "kept", "rows": 3, "kind": "error-recovery", "domain": "other", "ts": now - 600}],
    }
    for rel, evs in rows.items():
        p = runs / rel
        p.parent.mkdir(parents=True)
        p.write_text("\n".join(json.dumps(e) for e in evs) + "\n{torn")
    events = list(R.ledger_events(sorted(runs.glob("*/*.tasks.jsonl"))))
    s = R.gen_stats(events, now - 86400, now)
    assert s["rows"] == 5 and s["group"] == {"3090Ti": 2, "4090": 3}
    assert s["kind"] == {"error-recovery": 3, "chat-mode": 2}
    line = R.stats_line("Gen 24h", s)
    assert line.startswith("Gen 24h: 5 rows, 0.2/h (3090Ti 0.1 4090 0.1)") and "error-recovery 0.1" in line
    s2 = R.gen_stats(events, now - 2 * 3600, now)
    assert s2["rows"] == 5 and round(s2["per_h"], 1) == 2.5


# ── gen_runner's round-2 keys ────────────────────────────────────────────────


def _gen(tmp_path: Path, extra: str, remote: bool = False) -> Any:
    p = tmp_path / "w.conf"
    p.write_text("LOCAL_MODEL_UNIT=m.service\nPROBE_UNIT=p.service\nOLLAMA_MODEL=j:7b\n"
                 "GEN_REMOTE_URL=http://gpu-tower:18094\nGEN_REMOTE_HOST=gpu-tower\n" + extra)
    cls = G.RemoteGen if remote else G.Gen
    return cls(G.load(p), False, out=lambda s: None, clock=lambda: t("2026-11-02T01:30:00-05:00"), env={})


def test_gen_runner_takes_extra_flags_and_a_local_run_name(tmp_path: Path) -> None:
    until = t("2026-11-02T06:35:00-05:00")
    g = _gen(tmp_path, "")
    g.alias = "qwen3.8-27b-gsq-rco-iq3_xxs"
    assert g.local_out().name == "local-qwen3.8-27b-gsq-rco-iq3_xxs", "unchanged without the round-2 keys"
    assert g.local_argv(until)[-1] == "--quiet"
    g = _gen(tmp_path, "GEN_LOCAL_NAME='local-r2-{alias}-ab12'\n"
                       "GEN_LOCAL_EXTRA_ARGS='--skip-done-in /r/r2-tower-*/*.tasks.jsonl --coding-cap 0.2'\n")
    g.alias = "qwen3.8-27b-gsq-rco-iq3_xxs"
    assert g.local_out().name == "local-r2-qwen3.8-27b-gsq-rco-iq3_xxs-ab12"
    a = g.local_argv(until)
    assert a[-4:] == ["--skip-done-in", "/r/r2-tower-*/*.tasks.jsonl", "--coding-cap", "0.2"]
    assert a[a.index("--out") + 1].endswith("local-r2-qwen3.8-27b-gsq-rco-iq3_xxs-ab12")
    r = _gen(tmp_path, "GEN_REMOTE_EXTRA_ARGS='--mix error-recovery=1.0'\nGEN_REMOTE_NAME=r2-tower-x\n", remote=True)
    a = r.remote_argv(until)
    assert a[-2:] == ["--mix", "error-recovery=1.0"] and a[a.index("--out") + 1].endswith("/r2-tower-x")
    assert _gen(tmp_path, "", remote=True).remote_argv(until)[-1] == "--quiet"


# ── the Sunday chain, end to end against fakes ───────────────────────────────


class Fake:
    """The clock, units, the 4090 and the generation unit."""

    def __init__(self, start: str) -> None:
        self.now = t(start)
        self.units: dict[str, str] = {}
        self.tower_calls: list[tuple[str, list[str]]] = []
        self.gen_starts: list[float] = []
        self.tower: dict[str, str] = {"r2_units": "", "jobs_unit": "none", "end_last": "OK"}
        self.on_sleep: list[Any] = []
        self.gen_ends_at: float | None = None
        self.progress: dict[str, Any] = {}
        self.iq3 = True
        self.ab_jobs: list[dict[str, list[str]]] = []
        self.notes: list[list[str]] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s
        for f in list(self.on_sleep):
            f(self)

    def run(self, argv: list[str], timeout: float = 0, input: str | None = None, env: Any = None) -> Any:
        return subprocess.CompletedProcess(argv, 0, "", "")


def ctx(tmp_path: Path, fake: Fake, extra: str = "") -> R.Ctx:
    c = conf(tmp_path, f"R2_PIPELINE_UNIT=pt.service\nR2_POLL_S=60\n{extra}")
    cx = R.Ctx(c, True, "sunday", clock=fake.clock, run=fake.run, sleep=fake.sleep, out=lambda s: None)
    cx.unit_state = lambda u: fake.units.get(u, "gone")  # type: ignore[method-assign]
    cx.unit_result = lambda u: ("success", "0")  # type: ignore[method-assign]
    cx.tower_status = lambda: dict(fake.tower)  # type: ignore[method-assign]
    cx.remote_progress = lambda: dict(fake.progress)  # type: ignore[method-assign]

    def tower_run(what: str, args: list[str], wait_s: float) -> tuple[int, dict[str, Any]]:
        fake.tower_calls.append((what, args))
        if what in ("start", "swap"):
            req = args[args.index("--req") + 1]
            first = args[args.index("--profiles") + 1].split()[0]
            model, _, shape = first.rpartition(":")
            model = model or "q4"
            if model == "iq3" and not fake.iq3:
                model, shape = "q4", args[args.index("--profiles") + 1].split()[-1]
            slots, ctx_ = shape.split("x")
            fake.tower.update(jobs_unit="distill-jobs-r2teacher@x-1.service", r2_units="distill-jobs-r2teacher@x-1.service")
            return 0, {"status": "ok", "req": req, "unit": "distill-jobs-r2teacher@x-1.service", "model": model,
                       "slots": slots, "ctx": ctx_}
        fake.tower.update(jobs_unit="none", r2_units="")
        return 0, {}

    def start_gen(until: float, tag: str) -> str:
        fake.gen_starts.append(until)
        u = f"r2-gen-tower-{tag}-{len(fake.gen_starts)}"
        fake.units[u] = "active"
        return u

    def ab_run(jobs: dict[str, list[str]], deadline: float) -> dict[str, int]:
        fake.ab_jobs.append(jobs)
        fake.now += 600
        for argv in jobs.values():
            out = Path(argv[argv.index("--out") + 1])
            if argv[5] == "arm":
                arm = argv[argv.index("--arm") + 1]
                (out / f"arm-{arm}.jsonl").write_text("\n".join(json.dumps(
                    {"label": arm, "server": {"model_file": f"{arm}.gguf", "n_ctx": 32768}}) for _ in range(3)) + "\n")
            if argv[5] == "report":
                (out / "ab_summary.txt").write_text(f"Teacher A/B: keep IQ3 ({out.name})\n")
        return {k: 0 for k in jobs}

    cx.tower_run = tower_run  # type: ignore[method-assign]
    cx.start_remote_gen = start_gen  # type: ignore[method-assign]
    cx.ab_run = ab_run  # type: ignore[method-assign]
    cx.notify = lambda lines, n=5: fake.notes.append(lines)  # type: ignore[method-assign]
    return cx


def marker(tmp_path: Path, outcome: str, finished: str) -> None:
    p = Path(conf(tmp_path)["R2_PIPELINE_DONE"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"outcome": outcome, "finished": finished}))


def test_sunday_does_nothing_when_post_train_fails(tmp_path: Path) -> None:
    fake = Fake("2026-11-01T12:00:00-05:00")
    fake.units["pt.service"] = "active"
    cx = ctx(tmp_path, fake)

    def pipeline_fails(f: Fake) -> None:
        if f.now >= t("2026-11-01T14:00:00-05:00") and "pt.service" in f.units:
            marker(tmp_path, "not-run: verdict: val loss rose", "2026-11-01T14:00:00-05:00")
            f.units["pt.service"] = "inactive"

    fake.on_sleep.append(pipeline_fails)
    assert R.cmd_sunday(cx) == 0
    st = cx.load_state("sunday.json")
    assert st["phase"] == "skipped" and "did not complete" in st["why"]
    assert fake.tower_calls == [] and fake.gen_starts == []
    assert R.cmd_sunday(cx) == 0 and fake.tower_calls == [], "a finished chain does nothing again"


def test_sunday_skips_when_the_unit_ends_without_a_marker(tmp_path: Path) -> None:
    fake = Fake("2026-11-01T12:00:00-05:00")
    fake.units["pt.service"] = "failed"
    cx = ctx(tmp_path, fake)
    assert R.cmd_sunday(cx) == 0
    assert "without PIPELINE-DONE" in cx.load_state("sunday.json")["why"] and fake.tower_calls == []


def test_sunday_skips_when_post_train_is_not_done_by_the_latest_start(tmp_path: Path) -> None:
    fake = Fake("2026-11-02T03:58:00-05:00")
    fake.units["pt.service"] = "active"
    cx = ctx(tmp_path, fake)
    assert R.cmd_sunday(cx) == 0
    assert "not done by" in cx.load_state("sunday.json")["why"] and fake.tower_calls == []


def test_sunday_waits_for_capacity_then_runs_to_the_end(tmp_path: Path) -> None:
    fake = Fake("2026-11-01T15:00:00-05:00")
    marker(tmp_path, "completed", "2026-11-01T15:00:00-05:00")
    cap = Path(conf(tmp_path)["R2_CAPACITY_DONE"])

    def capacity_done(f: Fake) -> None:
        if f.now >= t("2026-11-01T16:00:00-05:00"):
            cap.parent.mkdir(parents=True, exist_ok=True)
            cap.write_text("{}")

    def teacher_ends_at_its_timer(f: Fake) -> None:  # the 4090's flywheel-r2-end
        if f.now >= t("2026-11-02T05:20:00-05:00"):
            f.tower.update(jobs_unit="none", r2_units="")
            for u in f.units:
                if u.startswith("r2-gen"):
                    f.units[u] = "inactive"

    fake.on_sleep += [capacity_done, teacher_ends_at_its_timer]
    cx = ctx(tmp_path, fake)
    assert R.cmd_sunday(cx) == 0
    st = cx.load_state("sunday.json")
    assert st["phase"] == "done" and st["why"] == "end"
    (what, args), = fake.tower_calls
    assert what == "start" and "--preempt-prefix" not in args
    assert args[args.index("--mode") + 1] == "cont" and args[args.index("--profiles") + 1] == "4x32768"
    assert args[args.index("--end-utc") + 1] == "2026-11-02 10:20:00 UTC"
    assert fake.gen_starts == [t("2026-11-02T05:18:00-05:00")]
    assert t("2026-11-01T16:00:00-05:00") <= fake.now <= t("2026-11-02T05:30:00-05:00")


def test_sunday_capacity_timeout_preempts_and_early_finish_ends_the_window(tmp_path: Path) -> None:
    fake = Fake("2026-11-01T15:00:00-05:00")
    marker(tmp_path, "completed", "2026-11-01T15:00:00-05:00")

    def gen_runs_dry(f: Fake) -> None:
        if f.now >= t("2026-11-01T22:00:00-05:00") and f.gen_starts:
            f.units["r2-gen-tower-sun-1"] = "inactive"
            f.progress = {"stop_reason": "nothing left to do"}

    fake.on_sleep.append(gen_runs_dry)
    cx = ctx(tmp_path, fake)
    assert R.cmd_sunday(cx) == 0
    st = cx.load_state("sunday.json")
    assert st["phase"] == "done" and "nothing left" in st["why"]
    assert [w for w, _ in fake.tower_calls] == ["start", "end"], "an early finish ends the 4090 window early"
    start_args = fake.tower_calls[0][1]
    assert start_args[start_args.index("--preempt-prefix") + 1] == "distill-cap-"
    assert st["capacity"].startswith("no CAPACITY-DONE.json 3 h")
    assert fake.now < t("2026-11-01T23:00:00-05:00")
    # the operator hears it, in <= 5 lines: nothing left, which tasksets, how early the window ends
    early = [n for n in fake.notes if "nothing left to do" in n[0]]
    assert len(early) == 1 and len(early[0]) <= 5
    assert "fallbacks: none" in early[0][1] and "h early" in early[0][2]


def test_sunday_restarts_failed_generation_then_gives_the_gpu_back(tmp_path: Path) -> None:
    fake = Fake("2026-11-01T15:00:00-05:00")
    marker(tmp_path, "completed", "2026-11-01T15:00:00-05:00")
    cap = Path(conf(tmp_path)["R2_CAPACITY_DONE"])
    cap.parent.mkdir(parents=True, exist_ok=True)
    cap.write_text("{}")

    def gen_dies(f: Fake) -> None:
        for u in f.units:
            if u.startswith("r2-gen"):
                f.units[u] = "failed"

    fake.on_sleep.append(gen_dies)
    cx = ctx(tmp_path, fake, "R2_GEN_RESTARTS=2\n")
    assert R.cmd_sunday(cx) == 0
    st = cx.load_state("sunday.json")
    assert len(fake.gen_starts) == 3 and st["gen_restarts"] == 2
    assert "remote generation is down" in st["why"]
    assert [w for w, _ in fake.tower_calls] == ["start", "end"]


def test_tower_night_waits_for_the_continuous_window_and_the_sunday_chain(tmp_path: Path) -> None:
    fake = Fake("2026-11-02T01:12:00-05:00")
    cx = ctx(tmp_path, fake)
    cx.cmd = "tower-night"
    assert R.cmd_tower_night(cx) == 0 and fake.tower_calls == [], "Mon 01:12 is the continuous window's"
    fake.now = t("2026-11-03T01:12:00-05:00")
    cx.save_state("sunday.json", {"phase": "generating"})
    assert R.cmd_tower_night(cx) == 0 and fake.tower_calls == []
    cx.save_state("sunday.json", {"phase": "done"})

    def night_end(f: Fake) -> None:
        if f.now >= t("2026-11-03T06:22:00-05:00"):
            f.tower.update(jobs_unit="none", r2_units="")

    fake.on_sleep.append(night_end)
    assert R.cmd_tower_night(cx) == 0
    (what, args), = fake.tower_calls
    assert args[args.index("--mode") + 1] == "night"
    assert args[args.index("--end-utc") + 1] == "2026-11-03 11:22:00 UTC"
    assert fake.gen_starts == [t("2026-11-03T06:20:00-05:00")]
    assert cx.load_state("night.json")["phase"] == "done"
    assert R.cmd_tower_night(dict_ctx(cx, R2_TOWER_NIGHTLY="off")) == 0 and len(fake.tower_calls) == 1


def dict_ctx(cx: R.Ctx, **kw: str) -> R.Ctx:
    cx.c = dict(cx.c, **kw)
    return cx


def test_the_tower_script_parses() -> None:
    sh = REPO / "scripts" / "distill" / "tower" / "round2_tower.sh"
    for f in (sh, MW_DIR / "round2_chain.sh", MW_DIR / "gen_local.sh"):
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f
    text = sh.read_text()
    assert "distill-jobs-r2teacher@" in text and "cloudflared" not in text


def test_released_rows_come_from_the_hold(tmp_path: Path) -> None:
    now = t("2026-11-01T09:00:00-05:00")
    evs = [("local-r2-a", {"task_id": "x", "status": "held", "rows": 4, "kind": "agentic", "ts": now - 50}),
           ("local-r2-a", {"task_id": "x", "status": "released", "rows": 0, "kind": "agentic", "ts": now - 40})]
    assert R.gen_stats(evs, now - 3600, now)["rows"] == 4


def test_gen_stats_json_lines() -> None:
    def js(rows: int, hours: float, runs: dict[str, int]) -> dict[str, Any]:
        return {"hours": hours, "runs": {k: {"last_hours": {"rows": v}} for k, v in runs.items()},
                "all": {"last_hours": {"rows": rows, "hours": hours, "rows_per_hour": rows / hours, "keep_rate": 0.4,
                                       "by_area_per_hour": {"error-recovery": 1.5, "round-1": 0.5},
                                       "by_category_per_hour": {"other": 1.0, "coding": 0.5}}}}
    lines = R.gen_stats_lines(js(48, 24, {"local-r2-a": 24, "r2-tower-b": 24}), js(10, 5, {"r2-tower-b": 10}),
                              "Gen since r2 Sun 01:30 (5.0 h)")
    assert lines[0] == ("Gen 24h: 48 rows, 2.0/h (3090Ti 1.0 4090 1.0), keep 0.4 | by area/h: error-recovery 1.5, "
                        "round-1 0.5")
    assert lines[1].startswith("Gen since r2 Sun 01:30 (5.0 h): 10 rows, 2.0/h (3090Ti 0.0 4090 2.0)")
    assert lines[2] == "24h by category/h: other 1, coding 0.5"


def test_gen_stats_lines_read_the_real_gen_stats_output(tmp_path: Path) -> None:
    gs = REPO / "scripts" / "distill" / "gen_stats.py"
    if not gs.is_file():
        pytest.skip("gen_stats.py not in this checkout")
    now = t("2026-11-01T09:00:00-05:00")
    d = tmp_path / "runs" / "r2-tower-x"
    d.mkdir(parents=True)
    (d / "g.tasks.jsonl").write_text(json.dumps({"task_id": "a", "status": "kept", "rows": 3, "area": "error-recovery",
                                                 "domain": "other", "ts": now - 600}) + "\n")
    out = [json.loads(subprocess.run([sys.executable, "-I", str(gs), "--runs", f"{tmp_path}/runs/*", "--hours", h,
                                      "--now", str(int(now)), "--json"], capture_output=True, text=True,
                                     check=True).stdout) for h in ("24", "2")]
    lines = R.gen_stats_lines(out[0], out[1], "since")
    assert lines[0].startswith("Gen 24h: 3 rows") and "4090 0.1" in lines[0] and "error-recovery" in lines[0]
    assert lines[1].startswith("since: 3 rows, 1.5/h") and lines[2].startswith("24h by category/h: other")


def test_measured_ctx_reads_the_capacity_blocks_teacher_ctx() -> None:
    doc = {"schema": "distill-capacity-teacher-ctx/1", "hosts": {
        "4090": {"gpu": "RTX 4090 24 GB", "usable_mib": 24073, "quants": {
            "q4": {"label": "UD-Q4_K_XL", "status": "measured",
                   "summary": {"max_stable_ctx": 163840, "first_unstable": {"ctx": 196608, "reason": "oom"},
                               "max_verified_prompt_tokens": 160000},
                   "rungs": [{"ctx": 131072, "stable": True}, {"ctx": 196608, "stable": False}],
                   "multi_slot": [{"slots": 2, "ctx": 163840, "stable": True}],
                   "recommended_profile": {"parallel": 1, "ctx": 163840,
                                           "env": {"TEACHER_SLOTS": 1, "TEACHER_CTX_PER_SLOT": 163840},
                                           "long_context_row_cap_tokens": 155000}},
            "iq3": {"label": "IQ3_XXS", "status": "measured", "summary": {"max_stable_ctx": 262144}}}},
        "3090ti": {"measured": {"max_readable_ctx": 57344}}},
        "recommendation": {"quant": "iq3", "max_stable_ctx": 262144}}
    assert R.measured_ctx(doc, "4090", "q4_k_xl") == 163840
    assert R.measured_ctx(doc, "4090", "iq3_xxs") == 262144


def test_released_rows_keep_the_held_kind() -> None:
    now = t("2026-11-01T09:00:00-05:00")
    evs = [("local-r2-a", {"task_id": "x", "status": "held", "rows": 2, "kind": "chat-mode", "domain": "chat-mode",
                           "ts": now - 50}),
           ("local-r2-a", {"task_id": "x", "status": "released", "ts": now - 40})]
    s = R.gen_stats(evs, now - 3600, now)
    assert s["kind"] == {"chat-mode": 2} and s["domain"] == {"chat-mode": 2}


def test_sunday_gives_business_back_when_the_teacher_never_starts(tmp_path: Path) -> None:
    fake = Fake("2026-11-02T03:00:00-05:00")
    marker(tmp_path, "completed", "2026-11-01T15:00:00-05:00")
    cap = Path(conf(tmp_path)["R2_CAPACITY_DONE"])
    cap.parent.mkdir(parents=True, exist_ok=True)
    cap.write_text("{}")
    cx = ctx(tmp_path, fake)

    def refused(what: str, args: list[str], wait_s: float) -> tuple[int, dict[str, Any]]:
        fake.tower_calls.append((what, args))
        if what == "start":
            return 2, {"status": "refused", "why": "the GPU is still owned by: distill-x", "req": args[args.index("--req") + 1]}
        return 0, {}

    cx.tower_run = refused  # type: ignore[method-assign]
    assert R.cmd_sunday(cx) == 0
    st = cx.load_state("sunday.json")
    assert st["phase"] == "skipped" and "still refused at Mon 04:00 EST" in st["why"]
    assert [w for w, _ in fake.tower_calls][-1] == "end" and fake.gen_starts == []
    assert sum(1 for w, _ in fake.tower_calls if w == "start") == 12, "every 5 min until the latest start"


def ab_dirs(tmp_path: Path, arm_a: int = 3) -> tuple[Path, Path]:
    mini, tower = tmp_path / "ab" / "mini-v3", tmp_path / "ab" / "tower-v3"
    mini.mkdir(parents=True)
    for f in ("ab_selection.json", "ab_settings.json"):
        (mini / f).write_text("{}")
    (mini / "ab_tasks.jsonl").write_text("{}\n{}\n{}\n")
    (mini / "arm-A.jsonl").write_text("{}\n" * arm_a)
    (mini / "ab_summary.txt").write_text("Teacher A/B: INCOMPLETE (0 of 3 pairs done; rerun to resume)\n")
    return mini, tower


def ab_conf(tmp_path: Path) -> str:
    code = tmp_path / "code-ab" / "scripts" / "distill"
    code.mkdir(parents=True, exist_ok=True)
    (code / "teacher_ab.py").write_text("")
    return f"R2_AB_DIR={tmp_path}/ab/mini-v3\nR2_AB_TOWER_DIR={tmp_path}/ab/tower-v3\nR2_AB_CODE={tmp_path}/code-ab\n"


def test_ab_wanted(tmp_path: Path) -> None:
    fake = Fake("2026-11-01T16:00:00-05:00")
    cx = ctx(tmp_path, fake, ab_conf(tmp_path))
    assert R.ab_wanted(cx) == "", "no arm A recorded"
    mini, tower = ab_dirs(tmp_path, arm_a=2)
    assert R.ab_wanted(cx) == "", "arm A not complete"
    (mini / "arm-A.jsonl").write_text("{}\n" * 3)
    assert R.ab_wanted(cx) == "both"
    tower.mkdir()
    (tower / "arm-A.jsonl").write_text("{}\n" * 3)
    assert R.ab_wanted(cx) == "b"
    (mini / "arm-B.jsonl").write_text("{}\n" * 3)
    (mini / "ab_summary.txt").write_text("Teacher A/B: keep IQ3\n")
    assert R.ab_wanted(cx) == ""
    assert R.ab_wanted(dict_ctx(cx, R2_AB="off")) == ""


def run_sunday_with_ab(tmp_path: Path, fake: Fake, extra: str = "") -> R.Ctx:
    marker(tmp_path, "completed", "2026-11-01T15:00:00-05:00")
    cap = Path(conf(tmp_path)["R2_CAPACITY_DONE"])
    cap.parent.mkdir(parents=True, exist_ok=True)
    cap.write_text("{}")

    def teacher_ends(f: Fake) -> None:
        if f.now >= t("2026-11-02T05:20:00-05:00"):
            f.tower.update(jobs_unit="none", r2_units="")
            for u in f.units:
                if u.startswith("r2-gen"):
                    f.units[u] = "inactive"

    fake.on_sleep.append(teacher_ends)
    cx = ctx(tmp_path, fake, ab_conf(tmp_path) + extra)
    assert R.cmd_sunday(cx) == 0
    return cx


def test_sunday_runs_the_ab_on_the_4090_first_then_generates(tmp_path: Path) -> None:
    mini, tower = ab_dirs(tmp_path)
    fake = Fake("2026-11-01T15:00:00-05:00")
    cx = run_sunday_with_ab(tmp_path, fake)
    calls = [(w, a[a.index("--profiles") + 1] if "--profiles" in a else "") for w, a in fake.tower_calls]
    assert calls == [("start", "iq3:4x32768 4x32768"), ("swap", "4x32768")], "IQ3 first, Q4 after; gen keeps it"
    steps = [sorted(j) for j in fake.ab_jobs]
    assert steps == [["A'"], ["mini", "tower"], ["mini", "tower"], ["mini", "tower"]]
    arm_a = fake.ab_jobs[0]["A'"]
    assert arm_a[arm_a.index("--teacher-quant") + 1] == "IQ3_XXS" and "--record-server-diff" not in arm_a
    arm_b = fake.ab_jobs[1]["mini"]
    assert arm_b[arm_b.index("--arm") + 1] == "B" and "--record-server-diff" in arm_b
    assert arm_b[arm_b.index("--teacher-quant") + 1] == "UD-Q4_K_XL"
    assert arm_b[arm_b.index("--tool-host") + 1] == R.this_host()
    assert (tower / "ab_tasks.jsonl").read_bytes() == (mini / "ab_tasks.jsonl").read_bytes()
    sv = json.loads((mini / "ab_servers.json").read_text())
    assert sv["reasoning_budget"] == 2048 and "model_file" in sv["differences"]
    assert fake.notes and fake.notes[0][0].startswith("Teacher A/B, arm B on the 4090")
    assert fake.gen_starts, "generation after the A/B"
    assert cx.load_state("sunday.json")["phase"] == "done"


def test_sunday_ab_without_the_iq3_copy_and_a_longctx_generation_profile(tmp_path: Path) -> None:
    mini, tower = ab_dirs(tmp_path)
    (tmp_path / "teacher-ctx.json").write_text(json.dumps(
        {"hosts": {"4090": {"quants": {"q4": {"label": "UD-Q4_K_XL", "summary": {"max_stable_ctx": 98304}}}}}}))
    fake = Fake("2026-11-01T15:00:00-05:00")
    fake.iq3 = False
    run_sunday_with_ab(tmp_path, fake, "R2_TOWER_PROFILE=longctx\n")
    calls = [(w, a[a.index("--profiles") + 1] if "--profiles" in a else "") for w, a in fake.tower_calls]
    assert calls == [("start", "iq3:4x32768 4x32768"), ("swap", "1x98304 4x32768")]
    assert [sorted(j) for j in fake.ab_jobs] == [["mini"], ["mini"], ["mini"]], "no same-hardware pair"
    assert fake.gen_starts


def test_mini_night_start_skips_without_the_teacher_gguf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Fake("2026-11-02T01:05:00-05:00")
    cx = ctx(tmp_path, fake, f"R2_MINI_TEACHER_GGUF={tmp_path}/none.gguf\n")
    cx.cmd = "mini-night"
    monkeypatch.setattr(R.os, "execve", lambda *a: pytest.fail("must not start the window"))
    assert R.cmd_mini_night(cx, "start") == 0
    assert fake.notes and "missing" in fake.notes[0][0]
    fake.now = t("2026-11-01T12:00:00-05:00")
    assert R.cmd_mini_night(cx, "end") == 0, "before R2_MINI_NIGHTLY_FROM: the continuous window's"


def test_night_plan_rotation(tmp_path: Path) -> None:
    c = conf(tmp_path, "R2_TOWER_TASKS=/t/real-v4-r2.jsonl\nR2_TOWER_TASKS_LONGCTX=/t/real-v4-r2-longctx.jsonl\n")
    def kinds(day: str, cc: dict[str, str] = c) -> list[tuple[str, str, str]]:
        now = t(f"{day}T01:12:00-05:00")
        return [(k, p["R2_TOWER_PROFILE"], p["R2_TOWER_TASKS"]) for k, p, _ in R.night_plan(cc, now, now + 5 * 3600)]
    assert kinds("2026-11-03") == [("longctx", "longctx", "/t/real-v4-r2-longctx.jsonl")], "Tue: long context"
    assert kinds("2026-11-04") == [("standard", "standard", "/t/real-v4-r2.jsonl")], "Wed: standard"
    assert kinds("2026-11-05")[0][0] == "longctx" and kinds("2026-11-06")[0][0] == "standard"
    assert kinds("2026-11-02")[0][0] == "longctx", "before the start date: the first entry"
    split = dict(c, R2_TOWER_NIGHT_ROTATION="split")
    now = t("2026-11-03T01:12:00-05:00")
    parts = R.night_plan(split, now, now + 4 * 3600)
    assert [(k, e - now) for k, _, e in parts] == [("longctx", 7200), ("standard", 4 * 3600)]
    assert kinds("2026-11-03", dict(c, R2_TOWER_NIGHT_ROTATION="bogus")) == [("standard", "standard", "/t/real-v4-r2.jsonl")]


def test_fallback_tasksets_reach_both_runners_and_the_longctx_part(tmp_path: Path) -> None:
    c = conf(tmp_path, "R2_TOWER_TASKS=/t/main.jsonl\nR2_TOWER_TASKS_FALLBACK=/t/fb1.jsonl ~/fb2.jsonl\n"
                       "R2_TOWER_TASKS_LONGCTX=/t/lc.jsonl\nR2_TOWER_TASKS_LONGCTX_FALLBACK=/t/main.jsonl\n"
                       "R2_LOCAL_TASKS=/t/main.jsonl\nR2_LOCAL_TASKS_FALLBACK=/t/fb1.jsonl\n")
    home = str(tmp_path / "home")
    until = t("2026-11-04T06:20:00-05:00")
    assert R.remote_overrides(c, until)["GEN_REMOTE_TASKS_FALLBACK"] == f"/t/fb1.jsonl {home}/fb2.jsonl"
    assert R.local_overrides(c)["GEN_TASKS_FALLBACK"] == "/t/fb1.jsonl"
    now = t("2026-11-03T01:12:00-05:00")
    (_k, lc, _e), = R.night_plan(c, now, now + 5 * 3600)
    assert lc["R2_TOWER_TASKS"] == "/t/lc.jsonl" and lc["R2_TOWER_TASKS_FALLBACK"] == "/t/main.jsonl"
    # the derived conf gen_runner reads carries them, and gen_runner maps the remote one onto its branch
    base = "LOCAL_MODEL_UNIT=m.service\nPROBE_UNIT=p.service\nOLLAMA_MODEL=j:7b\nNOTIFY=/bin/true\n"
    w = G.load(_write(tmp_path / "w.conf", R.derived_conf(base, R.remote_overrides(lc, until), "t")))
    rg = G.RemoteGen(w, False, out=lambda s: None)
    assert rg.c["GEN_TASKS"] == "/t/lc.jsonl" and rg.fallbacks == ["/t/main.jsonl"]
    # unset: no fallback, as before
    d = conf(tmp_path)
    assert R.remote_overrides(d, until)["GEN_REMOTE_TASKS_FALLBACK"] == "" and R.local_overrides(d)["GEN_TASKS_FALLBACK"] == ""

def test_a_split_night_swaps_half_way(tmp_path: Path) -> None:
    (tmp_path / "teacher-ctx.json").write_text(json.dumps(
        {"hosts": {"4090": {"quants": {"q4": {"label": "UD-Q4_K_XL", "summary": {"max_stable_ctx": 163840}}}}}}))
    fake = Fake("2026-11-03T01:12:00-05:00")
    cx = ctx(tmp_path, fake, "R2_TOWER_NIGHT_ROTATION=split\nR2_TOWER_TASKS_LONGCTX=/t/lc.jsonl\n")
    cx.cmd = "tower-night"
    seen: list[str] = []
    real_start = cx.start_remote_gen

    def start_gen(until: float, tag: str) -> str:
        seen.append(cx.c["R2_TOWER_TASKS"])
        return real_start(until, tag)

    cx.start_remote_gen = start_gen  # type: ignore[method-assign]

    def night_end(f: Fake) -> None:
        if f.now >= t("2026-11-03T06:22:00-05:00"):
            f.tower.update(jobs_unit="none", r2_units="")

    fake.on_sleep.append(night_end)
    assert R.cmd_tower_night(cx) == 0
    calls = [(w, a[a.index("--profiles") + 1]) for w, a in fake.tower_calls]
    assert calls == [("start", "1x163840 4x32768"), ("swap", "4x32768")]
    mid = t("2026-11-03T01:12:00-05:00") + (t("2026-11-03T06:22:00-05:00") - t("2026-11-03T01:12:00-05:00")) / 2
    assert fake.gen_starts == [mid - 60, t("2026-11-03T06:20:00-05:00")]
    assert seen[0] == "/t/lc.jsonl" and seen[1].endswith("real-v3-excl.jsonl")
    assert cx.c["R2_TOWER_PROFILE"] == "standard", "the settings are restored after the night"
    assert cx.load_state("night.json")["plan"] == ["longctx", "standard"]


def test_a_longctx_night_without_a_measurement_uses_the_standard_taskset(tmp_path: Path) -> None:
    fake = Fake("2026-11-03T01:12:00-05:00")
    cx = ctx(tmp_path, fake, "R2_TOWER_TASKS=/t/full.jsonl\nR2_TOWER_TASKS_LONGCTX=/t/lc.jsonl\n")
    cx.cmd = "tower-night"
    seen: list[str] = []
    real_start = cx.start_remote_gen
    cx.start_remote_gen = lambda until, tag: (seen.append(cx.c["R2_TOWER_TASKS"]), real_start(until, tag))[1]  # type: ignore[method-assign]
    fake.on_sleep.append(lambda f: f.tower.update(jobs_unit="none", r2_units="")
                         if f.now >= t("2026-11-03T06:22:00-05:00") else None)
    assert R.cmd_tower_night(cx) == 0
    assert [a[a.index("--profiles") + 1] for _, a in fake.tower_calls] == ["4x32768"]
    assert seen == ["/t/full.jsonl"]



# ── the long-context profile on the 4090's IQ3 copy (capacity block, 2026-11-01) ──


CAPACITY_CTX = {"hosts": {"4090": {"quants": {
    "q4": {"label": "UD-Q4_K_XL", "status": "measured", "summary": {"max_stable_ctx": 163840}},
    "iq3": {"label": "IQ3_XXS", "status": "measured", "summary": {"max_stable_ctx": 262144}}}},
    "3090ti": {"measured": {"max_readable_ctx": 57344}}}, "recommendation": {"quant": "iq3", "max_stable_ctx": 262144}}


def test_longctx_serves_the_iq3_copy_at_its_measured_262144(tmp_path: Path) -> None:
    (tmp_path / "teacher-ctx.json").write_text(json.dumps(CAPACITY_CTX))
    lc = dict(conf(tmp_path), R2_TOWER_PROFILE="longctx")
    profs, why = R.resolve_profiles(lc)
    assert profs == ["iq3:1x262144", "1x163840", "4x32768"], why
    assert "measured q4 163840, iq3 262144" in why
    assert R.resolve_profiles(dict(lc, R2_LONGCTX_QUANT="q4"))[0] == ["1x163840", "4x32768"]
    assert R.resolve_profiles(dict(lc, R2_LONGCTX_QUANT="iq3"))[0] == ["iq3:1x262144", "4x32768"]
    # capped equal: the configured Q4 teacher is kept (IQ3 only when it reads more)
    assert R.resolve_profiles(dict(lc, R2_LONGCTX_CAP="131072"))[0] == ["1x131072", "4x32768"]
    # the standard profile never changes model
    assert R.resolve_profiles(dict(lc, R2_TOWER_PROFILE="standard"))[0] == ["4x32768"]


def test_generation_names_the_teacher_the_4090_actually_serves(tmp_path: Path) -> None:
    c = conf(tmp_path, "R2_TOWER_HOST=gpu\nR2_TOWER_URL=http://gpu:18094\n")
    q4 = R.remote_overrides(R.served(c, "q4"), t("2026-11-03T06:20:00-05:00"))
    iq3 = R.remote_overrides(R.served(c, "iq3"), t("2026-11-03T06:20:00-05:00"))
    assert q4["GEN_REMOTE_QUANT"] == "UD-Q4_K_XL" and q4["GEN_REMOTE_SHA256"] == c["R2_TOWER_SHA256"]
    assert iq3["GEN_REMOTE_QUANT"] == "IQ3_XXS" and iq3["GEN_REMOTE_SHA256"] == c["R2_IQ3_SHA256"]
    assert "iq3_xxs" in iq3["GEN_REMOTE_NAME"] and iq3["GEN_REMOTE_NAME"] != q4["GEN_REMOTE_NAME"]
    assert R.served(c, None) is c


def test_a_longctx_night_on_the_iq3_copy_generates_as_the_iq3_teacher(tmp_path: Path) -> None:
    (tmp_path / "teacher-ctx.json").write_text(json.dumps(CAPACITY_CTX))
    fake = Fake("2026-11-03T01:12:00-05:00")
    cx = ctx(tmp_path, fake, "R2_TOWER_TASKS_LONGCTX=/t/lc.jsonl\n")
    cx.cmd = "tower-night"
    seen: list[tuple[str, str, str]] = []
    real_start = cx.start_remote_gen
    cx.start_remote_gen = lambda until, tag: (seen.append(  # type: ignore[method-assign]
        (cx.c["R2_TOWER_TASKS"], cx.c["R2_TOWER_QUANT"], cx.c["R2_TOWER_SHA256"])), real_start(until, tag))[1]
    fake.on_sleep.append(lambda f: f.tower.update(jobs_unit="none", r2_units="")
                         if f.now >= t("2026-11-03T06:22:00-05:00") else None)
    assert R.cmd_tower_night(cx) == 0
    assert [a[a.index("--profiles") + 1] for _, a in fake.tower_calls][0] == "iq3:1x262144 1x163840 4x32768"
    assert seen == [("/t/lc.jsonl", "IQ3_XXS", R.R2_DEFAULTS["R2_IQ3_SHA256"])]
    assert cx.c["R2_TOWER_QUANT"] == "UD-Q4_K_XL", "the settings are restored after the night"
    # no IQ3 copy on the 4090: it serves Q4, and generation says Q4
    fake2 = Fake("2026-11-03T01:12:00-05:00")
    fake2.iq3 = False
    cx2 = ctx(tmp_path / "b", fake2, "R2_TOWER_TASKS_LONGCTX=/t/lc.jsonl\n") if (tmp_path / "b").mkdir() is None \
        else None
    assert cx2 is not None
    (tmp_path / "b" / "teacher-ctx.json").write_text(json.dumps(CAPACITY_CTX))
    cx2.cmd = "tower-night"
    seen2: list[str] = []
    real2 = cx2.start_remote_gen
    cx2.start_remote_gen = lambda until, tag: (seen2.append(cx2.c["R2_TOWER_QUANT"]), real2(until, tag))[1]  # type: ignore[method-assign]
    fake2.on_sleep.append(lambda f: f.tower.update(jobs_unit="none", r2_units="")
                          if f.now >= t("2026-11-03T06:22:00-05:00") else None)
    assert R.cmd_tower_night(cx2) == 0
    assert seen2 == ["UD-Q4_K_XL"]
