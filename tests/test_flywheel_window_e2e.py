"""The weekly flywheel end to end: whole simulated Saturday windows, week after week.

``prometheus.jobs.flywheel_weekly.main`` runs with its real runner
(``run_command``: real child processes in their own session, SIGTERM and
SIGKILL to the process group) against stand-in stage executables
(tests/fixtures/flywheel_sim/fake_stages.py) that write what the real ones
write: golden exports and teacher_gen.py-shaped gym traces between windows, a
builder with build.py's data policy, a resumable trainer that checkpoints on
SIGTERM, a quantizer, score.py run/compare, setups.py deadman/apply.

Time is simulated. The job's clock reads a file; a stand-in moves it only by
spending time, then waits until the job's poll has read the new time. So the
window logic acts on the exact simulated minute (the stop time at 05:45, the
SIGTERM, the restore) and four Saturdays take a few real seconds.

The season: Sat 2026-10-31 a normal week trains, quantizes, scores and passes;
Sat 2026-11-07 (the first Saturday after the November DST change) drops
unverified offspring and is cut short mid-train by the window (SIGTERM,
checkpoint); Sat 2026-11-14 resumes that train and the quantizer crashes;
Sat 2026-11-21 is below the train threshold, quantizes and scores against the
previous model. Separately: Max-derived data refused, offspring output refused
at train time. The reviewer's file is checked section by section throughout.

No model, no GPU, no network. Children are ``sys.executable`` scripts that end
by exit code, SIGTERM or SIGKILL (never a crash signal).
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml

import prometheus.jobs.flywheel_weekly as fw

ET = ZoneInfo("America/New_York")
FAKE = Path(__file__).resolve().parent / "fixtures" / "flywheel_sim" / "fake_stages.py"

W1 = datetime(2026, 10, 31, 2, 0, 30, tzinfo=ET)    # EDT, UTC-4
W2 = datetime(2026, 11, 7, 2, 0, 30, tzinfo=ET)     # EST, UTC-5: first Saturday after the change
W3 = datetime(2026, 11, 14, 2, 0, 30, tzinfo=ET)
W4 = datetime(2026, 11, 21, 2, 0, 30, tzinfo=ET)

APPROVED = {"max_data": {"approved": True, "approved_by": "operator",
                         "approved_on": "2020-01-01", "reference": "test decision"}}
BASE = {"task_success": 0.40, "recall_32k": 0.90, "recall_64k": 0.85, "injection": 0.70,
        "sanity": 0.80}
SCORES = {
    "fw-2026-10-31": {"task_success": 0.55, "recall_32k": 0.895, "recall_64k": 0.86,
                      "injection": 0.75, "sanity": 0.80},
    "fw-2026-11-07": {"task_success": 0.63, "recall_32k": 0.88, "recall_64k": 0.875,
                      "injection": 0.75, "sanity": 0.82},
}
OFFSPRING_ID = "distill-9b-fw-2026-10-31-Q4_K_M"


class SimClock:
    """The job's clock: reads the simulated time, and records what it read (and
    how many reads so far) so a stand-in waiting on its own tick knows the job
    has seen it and finished acting on it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.seen = path.with_name(path.name + ".seen")
        self.reads = 0

    def set(self, when: datetime) -> None:
        self.path.write_text(when.isoformat())

    def __call__(self) -> datetime:
        now = datetime.fromisoformat(self.path.read_text().strip())
        self.reads += 1
        tmp = self.seen.with_name(".seen.tmp")
        tmp.write_text(f"{now.isoformat()} {self.reads}")
        os.replace(tmp, self.seen)
        return now.astimezone(timezone.utc)


def _summary(label: str, scores: dict[str, float], cases: int = 4) -> dict[str, Any]:
    def rate(r: float) -> dict[str, Any]:
        return {"k": round(r * cases), "n": cases, "rate": r}

    return {"schema": "distill-score-summary/2", "label": label, "served_model": f"{label}.gguf",
            "sets": {"eval-tools": {"cases": cases, "scored": cases,
                                    "metrics": {"task_success": rate(scores["task_success"])}},
                     "eval-longctx": {"metrics": {"recall_by_depth": {
                         "32k": rate(scores["recall_32k"]), "64k": rate(scores["recall_64k"])}}},
                     "eval-injection": {"metrics": {"resistance": rate(scores["injection"])}},
                     "sanity": {"metrics": {"sanity": rate(scores["sanity"])}}}}


class Sim:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.data = root / "fw"
        self.exports = root / "exports"
        self.gym = root / "gym"
        self.repo = root / "repo"
        self.log_path = root / "sim-calls.jsonl"
        self.clock = SimClock(root / "clock")
        self.clock.set(W1)
        self.served = root / "served"
        self.served.write_text("")
        for rel in ("scripts/distill/build.py", fw.SCORE_SCRIPT, fw.SETUPS_SCRIPT):
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(FAKE, self.repo / rel)
        scores = root / "scores.json"
        scores.write_text(json.dumps(SCORES))
        self.baseline = root / "baseline.summary.json"
        self.baseline.write_text(json.dumps(_summary("base-9b", BASE)))
        cases = root / "cases"
        cases.mkdir()
        (cases / "eval-tools.jsonl").write_text("".join(
            json.dumps({"id": f"case-tools-{i}", "meta": {"task_id": f"held-{i}"}}) + "\n"
            for i in range(3)))
        (cases / "eval-longctx.jsonl").write_text(json.dumps({"id": "case-long-1"}) + "\n")
        self.cases = [cases / "eval-tools.jsonl", cases / "eval-longctx.jsonl"]
        self.ledger = root / "spend-ledger.csv"
        self.spend("2026-10-24T12:00:00+00:00", "phase0", 1, 0.0)
        self.spend("2026-10-31T05:30:00+00:00", "judge", 10, 0.40)     # 01:30 EDT: W1's week
        for key, value in (("FLYWHEEL_SIM_CLOCK", self.clock.path),
                           ("FLYWHEEL_SIM_LOG", self.log_path),
                           ("FLYWHEEL_SIM_SERVED", self.served),
                           ("FLYWHEEL_SIM_SCORES", scores)):
            monkeypatch.setenv(key, str(value))
        monkeypatch.delenv("FLYWHEEL_SIM_CRASH", raising=False)
        monkeypatch.setattr(fw, "POLL_SECONDS", 0.002)
        monkeypatch.setattr(fw, "served_model_id", self._served_model_id)

    def spend(self, ts: str, phase: str, calls: int, usd: float) -> None:
        """A spend-ledger row, as the runbook's judge would append one."""
        fields = ("ts", "phase", "calls", "input_tokens", "output_tokens", "est_usd", "model",
                  "note")
        fresh = not self.ledger.exists()
        with open(self.ledger, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if fresh:
                w.writeheader()
            w.writerow({"ts": ts, "phase": phase, "calls": calls, "input_tokens": calls * 900,
                        "output_tokens": calls * 60, "est_usd": usd, "model": "judge"})

    def _served_model_id(self, base_url: str, timeout: float = 10.0) -> str | None:
        """llama-server's GET /v1/models: the newest quantized model is loaded
        (the setup's job); the scorer reads the same answer from the file."""
        loaded = sorted(p for p in (self.data / "models").glob("*") if any(p.glob("*.gguf")))
        sid = f"models/{loaded[-1].name}/{loaded[-1].name}-Q4_K_M.gguf" if loaded else ""
        self.served.write_text(sid)
        return sid or None

    # -- the week between windows ------------------------------------------------

    def produce(self, kind: str, out: Path, first: int, count: int, **opts: str) -> None:
        argv = [sys.executable, str(FAKE), "traces", "--kind", kind, "--out", str(out),
                "--first", str(first), "--count", str(count)]
        for key, value in opts.items():
            argv += [f"--{key.replace('_', '-')}", value]
        subprocess.run(argv, check=True, timeout=60)

    # -- the config and the run --------------------------------------------------

    def block(self, **over: Any) -> dict[str, Any]:
        block: dict[str, Any] = {
            "enabled": True, "data_dir": str(self.data),
            "harvest": {"golden": [str(self.exports / "golden_traces_*.jsonl")],
                        "extra": [str(self.gym / "*.jsonl")],
                        "teacher_models": ["qwen3.8-max", "qwen3.8-27b*"],
                        "student_models": ["qwen3.5-9b*"]},
            "build": {"args": ["--pack", "pack.json", "--tokenizer-dir", "tok",
                               "--train-sources", "qwen38-27b-gym",
                               "--train-models", "qwen3.8-27b*", "--min-train", "8",
                               "--spot-check-server", "http://127.0.0.1:8099",
                               "--sim-minutes", "20"]},
            "train": {"min_new_traces": 10,
                      "command": ["{python}", str(FAKE), "train", "--data", "{train}",
                                  "--val", "{val}", "--out", "{run_dir}", "--stop-at",
                                  "{stop_at}", "--minutes-per-row", "5", "--save-every", "10"],
                      "resume_args": ["--resume"]},
            "quantize": {"command": ["{python}", str(FAKE), "quantize", "--adapter",
                                     "{adapter_dir}", "--data", "{data_dir}", "--out",
                                     "{out_dir}", "--version", "{adapter_version}"]},
            "score": {"base_url": "http://127.0.0.1:8099",
                      "cases": [str(c) for c in self.cases],
                      "baseline_summary": str(self.baseline)},
            "gpu_setup": "flywheel",
            "spend_ledger": str(self.ledger),
        }
        for key, value in over.items():
            block[key] = ({**block[key], **value} if isinstance(value, dict)
                          and isinstance(block.get(key), dict) else value)
        return block

    def run(self, when: datetime, block: dict[str, Any] | None = None, *extra: str) -> int:
        config = self.root / "prometheus.yaml"
        config.write_text(yaml.safe_dump({"flywheel": block or self.block()}))
        self.clock.set(when)
        return fw.main(["--config", str(config), *extra], clock=self.clock, repo=self.repo)

    # -- what it left ------------------------------------------------------------

    def calls(self) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        return [json.loads(x) for x in self.log_path.read_text().splitlines() if x.strip()]

    def cmds(self, since: int = 0) -> list[str]:
        return [c["cmd"] + (f":{c['setup']}" if c["cmd"] == "apply" else "")
                for c in self.calls()[since:]]

    def report(self) -> dict[str, Any]:
        rep = json.loads((self.data / "reports" / "LATEST.json").read_text())
        assert fw.report_problems(rep) == [], fw.report_problems(rep)
        return rep

    def md(self) -> str:
        return (self.data / "reports" / "LATEST.md").read_text()

    def state(self) -> dict[str, Any]:
        return json.loads((self.data / "state.json").read_text())


def sections(md: str) -> dict[str, str]:
    """LATEST.md split at its "## " headings, which must be the reviewer's, in order."""
    assert fw.markdown_sections(md) == list(fw.REVIEWER_SECTIONS)
    out: dict[str, str] = {}
    name = ""
    for line in md.splitlines():
        if line.startswith("## "):
            name = line[3:]
            out[name] = ""
        elif name:
            out[name] += line + "\n"
    return out


def stage(rep: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in rep["stages"] if s["name"] == name)


def steps(rep: dict[str, Any], by: str) -> list[str]:
    return [s["step"] for s in rep["next_steps"] if s["by"] == by]


@pytest.fixture
def sim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sim:
    return Sim(tmp_path, monkeypatch)


def week_one(sim: Sim, block: dict[str, Any] | None = None) -> int:
    sim.produce("golden", sim.exports / "golden_traces_2026-10-30.jsonl", 0, 6)
    sim.produce("gym", sim.gym / "gym-2026-10-30.jsonl", 0, 14)
    return sim.run(W1, block)


# --------------------------------------------------------------------------- #
# The season
# --------------------------------------------------------------------------- #


def test_four_saturdays_train_pause_resume_crash_and_score(sim: Sim, monkeypatch):
    # ---- Sat 2026-10-31 (EDT): a normal week with training ----------------------
    assert week_one(sim) == fw.EXIT_OK
    assert sim.cmds() == ["build", "deadman", "apply:flywheel", "train", "train-done",
                          "quantize", "score", "compare", "apply:business"]
    deadman = next(c for c in sim.calls() if c["cmd"] == "deadman")
    assert deadman["setup"] == "business" and deadman["at"] == "2026-10-31T06:15-04:00"
    build = next(c for c in sim.calls() if c["cmd"] == "build")
    # the 14 gym traces train under their own source tag; the 6 Max golden are eval-only
    assert (build["train"], build["val"], build["eval_tools"]) == (13, 1, 6)
    assert build["teacher"] == "qwen3.8-27b"

    rep = sim.report()
    assert rep["status"] == "OK" and rep["mode"] == "scheduled" and rep["run_id"] == "2026-10-31"
    assert {s["name"]: s["status"] for s in rep["stages"]} == dict.fromkeys(fw.STAGES, fw.OK)
    # stage times are simulated minutes: the build spent 20, the train 3 (swap) + 13 x 5
    assert stage(rep, "build")["seconds"] == 1200.0
    assert stage(rep, "train")["seconds"] == 68 * 60.0
    assert rep["window"]["start"] == "2026-10-31T02:00-04:00"
    assert rep["finished"] == "2026-10-31T03:54:30-04:00"
    md = sim.md()
    sec = sections(md)
    assert md.startswith("# Flywheel weekly report: 2026-10-31\n\n**FLYWHEEL weekly · OK** · "
                         "scheduled run · window Sat 02:00-06:00 America/New_York")
    assert "Ran Sat 2026-10-31 02:00-03:54 EDT." in md
    assert "- Model: fw-2026-10-31 scored; Phase 6 gate vs baseline (baseline.summary.json): " \
           "PASS; vs baseline (baseline.summary.json): 3 win(s), 0 loss(es)" in sec["At a glance"]
    assert "- Next window: Sat 2026-11-07 02:00-06:00 EST (07:00-11:00 UTC)" in sec["At a glance"]
    assert "| build | OK | 20 min |" in sec["What ran"]
    assert "| train | OK | 1 h 08 min | fw-2026-10-31 finished (1 window(s), 14 rows) |" \
           in sec["What ran"]
    assert sec["Why stages did not run"].strip() == "Every stage ran."
    assert "- New this window: 20 (teacher 20); already harvested 0; unreadable 0" in sec["Traces"]
    assert "- Verified: 20; dropped: none" in sec["Traces"]
    model = sec["Model: wins and losses vs the previous model"]
    assert "Candidate **fw-2026-10-31**: Phase 6 gate vs baseline (baseline.summary.json): " \
           "**PASS**" in model
    assert "Wins (3):" in model and "Losses (0):" in model
    assert "- eval-tools / task_success: 40.0 -> 55.0 (+15.0 pts)" in model
    assert "- Last 7 days (since 2026-10-24T02:00:00-04:00): 11 call(s)" in sec["Spend"]
    assert "- Train data checked: 14 row(s), 0 Max-derived, 0 offspring (0 unverified), 0 " \
           "sharing a task with the score cases" in sec["Guards"]
    assert sec["Issues"].strip() == "None." and sec["Operator notes"].strip() == "None."
    assert steps(rep, "operator") == [
        "Decide whether to promote fw-2026-10-31: it passed the Phase 6 gate vs baseline "
        "(baseline.summary.json). Nothing was swapped; promotion is the operator's call."]
    assert rep["next_steps"][-1] == {
        "by": "job", "step": "Next window: Sat 2026-11-07 02:00-06:00 EST (07:00-11:00 UTC)."}

    # ---- the week: more gym traces, Max exports, and the first offspring --------
    sim.spend("2026-10-31T06:30:00+00:00", "judge", 20, 0.80)          # 02:30 EDT, in W1
    sim.produce("golden", sim.exports / "golden_traces_2026-11-06.jsonl", 100, 3)
    sim.produce("gym", sim.gym / "gym-2026-11-06.jsonl", 100, 50)
    offspring = sim.gym / "offspring-2026-11-06.jsonl"
    sim.produce("offspring", offspring, 200, 3, served=OFFSPRING_ID)                # no verdict
    sim.produce("offspring", offspring, 203, 1, served=OFFSPRING_ID, judged_by=OFFSPRING_ID)
    sim.produce("offspring", offspring, 204, 2, served=OFFSPRING_ID, judged_by="qwen3.8-27b")

    # ---- Sat 2026-11-07 (EST): the timer's 02:00 is 07:00 UTC now ---------------
    before = (sim.md(), len(sim.calls()))
    # 06:00:30 UTC was 02:00:30 last week; this week it is 01:00:30 EST
    assert sim.run(datetime(2026, 11, 7, 6, 0, 30, tzinfo=timezone.utc)) == fw.EXIT_CONFIG
    assert (sim.md(), len(sim.calls())) == before
    n = len(sim.calls())
    assert sim.run(W2) == fw.EXIT_OK
    assert sim.cmds(n) == ["build", "deadman", "apply:flywheel", "train", "train-stopped",
                           "apply:business"]
    train_call = next(c for c in sim.calls()[n:] if c["cmd"] == "train")
    assert train_call["start_step"] == 0 and train_call["steps"] == 56
    assert train_call["stop_at"] == "2026-11-07T10:45:00+00:00"       # 05:45 EST
    # Step 41 ends at 05:48:30, the first poll past the 05:45 stop: SIGTERM there,
    # the checkpoint on that step, out well before the 05:59 SIGKILL.
    stopped = next(c for c in sim.calls()[n:] if c["cmd"] == "train-stopped")
    assert (stopped["at"], stopped["checkpoint"]) == ("2026-11-07T05:48:30-05:00",
                                                      "checkpoint-41")
    checkpoint = "checkpoint-41"

    rep = sim.report()
    assert rep["status"] == "OK" and rep["run_id"] == "2026-11-07"
    assert rep["window"]["start"] == "2026-11-07T02:00-05:00"
    assert rep["window"]["stop_at"] == "2026-11-07T05:45-05:00"
    assert stage(rep, "train")["status"] == fw.PAUSED
    assert stage(rep, "train")["reason"] == (f"stopped for the window at {checkpoint}; resumes "
                                             "next window")
    assert stage(rep, "train")["seconds"] == (3 * 60 + 28) * 60.0     # 02:20:30 to 05:48:30
    for name in ("quantize", "score"):
        assert stage(rep, name)["status"] == fw.SKIPPED
        assert "0 min left before the stop time (05:45)" in stage(rep, name)["reason"]
    sec = sections(sim.md())
    assert "- New this window: 59 (offspring 6, teacher 53)" in sec["Traces"]
    assert "- Verified: 55; dropped: self_verified_offspring 1, unverified_offspring 3" \
           in sec["Traces"]
    assert f"| offspring | {OFFSPRING_ID} | fw-2026-10-31 | 6 |" in sec["Traces"]
    assert "- Offspring traces dropped as unverified this window: 4" in sec["Guards"]
    assert "- **train** PAUSED: stopped for the window at" in sec["Why stages did not run"]
    # the spend week starts at last week's window start, 02:00 EDT, not 168 h ago
    assert rep["spend"]["since"] == "2026-10-31T02:00:00-04:00"
    assert rep["spend"]["week"]["est_usd"] == 0.80 and rep["spend"]["total"]["est_usd"] == 1.20
    assert steps(rep, "job") == [
        f"Resume train fw-2026-11-07 from {checkpoint} in the next window, on the same frozen "
        "data.",
        "Next window: Sat 2026-11-14 02:00-06:00 EST (07:00-11:00 UTC)."]
    pool = [json.loads(x) for x in (sim.data / "pool" / "2026-11-07.jsonl").read_text()
            .splitlines()]
    assert sum(p["_meta"]["origin"] == "offspring" for p in pool) == 2     # the judged ones only

    # ---- Monday: the reviewer's input, refreshed without running anything -------
    state_bytes = (sim.data / "state.json").read_bytes()
    dated = (sim.data / "reports" / "2026-11-07-weekly.md").read_bytes()
    n = len(sim.calls())
    monday = datetime(2026, 11, 9, 7, 30, tzinfo=ET)
    assert sim.run(monday, None, "--report-only", "--note", "Saluki test drive ends today",
                   "--next-step", "Rehearse the GPU swap before Saturday") == fw.EXIT_OK
    assert len(sim.calls()) == n                                          # no child started
    assert (sim.data / "state.json").read_bytes() == state_bytes          # state untouched
    assert (sim.data / "reports" / "2026-11-07-weekly.md").read_bytes() == dated
    refreshed = sim.data / "reports" / "2026-11-09-weekly-report-only.md"
    assert refreshed.read_text() == sim.md()
    rep = sim.report()
    assert (rep["mode"], rep["run_id"], rep["reported_run"]) == ("report-only", "2026-11-07",
                                                                 "2026-11-07")
    assert stage(rep, "train")["status"] == fw.PAUSED
    assert rep["spend"]["since"] == "2026-11-02T07:30:00-05:00"
    assert rep["notes"] == ["Saluki test drive ends today"]
    assert steps(rep, "runbook") == ["Rehearse the GPU swap before Saturday"]
    md = sim.md()
    assert "report only: no stage ran; the 2026-11-07 run's record" in md
    assert "Refreshed Mon 2026-11-09 07:30 EST." in md
    assert "- [runbook] Rehearse the GPU swap before Saturday" in sections(md)["Next steps"]

    # ---- Sat 2026-11-14: below the bar, the paused train resumes; quantize crashes
    sim.produce("gym", sim.gym / "gym-2026-11-13.jsonl", 300, 1)
    monkeypatch.setenv("FLYWHEEL_SIM_CRASH", "quantize")
    n = len(sim.calls())
    assert sim.run(W3) == fw.EXIT_FAILED
    assert sim.cmds(n) == ["build", "deadman", "apply:flywheel", "train", "train-done",
                           "quantize", "apply:business"]
    resumed = next(c for c in sim.calls()[n:] if c["cmd"] == "train")
    assert resumed["resume"] is True and resumed["start_step"] == int(checkpoint.split("-")[1])
    assert resumed["data"] == train_call["data"]                          # the frozen data
    rep = sim.report()
    assert rep["status"] == "FAIL"
    assert stage(rep, "train")["status"] == fw.OK
    assert "fw-2026-11-07 finished (2 window(s)" in stage(rep, "train")["summary"]
    quantize = stage(rep, "quantize")
    assert quantize["status"] == fw.FAILED
    assert "the quantizer exited 1" in quantize["reason"]
    assert "RuntimeError: simulated llama-quantize crash" in quantize["reason"]
    assert stage(rep, "score")["reason"] == "no new quantized model to score"
    sec = sections(sim.md())
    assert sim.md().startswith("# Flywheel weekly report: 2026-11-14\n\n**FLYWHEEL weekly · FAIL**")
    assert "- **quantize** FAILED: the quantizer exited 1" in sec["Why stages did not run"]
    assert any(s.startswith("Look into the quantize failure") for s in steps(rep, "operator"))
    assert steps(rep, "job")[:2] == [
        "Train waits for 9 more verified trace(s) (1 of 10 since the last train).",
        "Quantize fw-2026-11-07 in the next window."]
    # systemd's OnFailure= unit runs next: the run finished, so it changes nothing
    latest = sim.md()
    monkeypatch.delenv("FLYWHEEL_SIM_CRASH")
    n = len(sim.calls())
    assert sim.run(datetime(2026, 11, 14, 4, 0, tzinfo=ET), None, "--recover") == fw.EXIT_OK
    assert len(sim.calls()) == n and sim.md() == latest
    assert sim.state()["gpu"]["applied"] is False

    # ---- Sat 2026-11-21: below the train threshold; quantize again, score -------
    sim.produce("gym", sim.gym / "gym-2026-11-20.jsonl", 400, 2)
    n = len(sim.calls())
    assert sim.run(W4) == fw.EXIT_OK
    assert sim.cmds(n) == ["build", "deadman", "apply:flywheel", "quantize", "score",
                           "compare", "apply:business"]
    rep = sim.report()
    assert rep["status"] == "OK"
    train = stage(rep, "train")
    assert train["status"] == fw.SKIPPED
    assert train["reason"] == ("3 new verified trace(s) since the last train; a train needs 10 "
                               "(flywheel.train.min_new_traces)")
    m = rep["model"]
    assert (m["candidate"], m["verdict"], m["previous"]) == ("fw-2026-11-07", "PASS",
                                                             "fw-2026-10-31")
    assert m["gate_against"] == "baseline (baseline.summary.json)"
    assert [w["metric"] for w in m["wins"]] == [
        "eval-longctx / recall_by_depth @64k", "eval-tools / task_success", "sanity / sanity"]
    assert [x["metric"] for x in m["losses"]] == ["eval-longctx / recall_by_depth @32k"]
    assert m["unchanged"] == 1
    sec = sections(sim.md())
    model = sec["Model: wins and losses vs the previous model"]
    assert "Compared with fw-2026-10-31 (a comparison, not the gate):" in model
    assert "Wins (3):" in model and "Losses (1):" in model
    assert "- eval-longctx / recall_by_depth @32k: 89.5 -> 88.0 (-1.5 pts)" in model
    assert "- **train** SKIPPED: 3 new verified trace(s) since the last train" \
           in sec["Why stages did not run"]
    assert "- Train: fw-2026-11-07 is complete" in sec["At a glance"]
    assert steps(rep, "job") == [
        "Train waits for 7 more verified trace(s) (3 of 10 since the last train).",
        "Next window: Sat 2026-11-28 02:00-06:00 EST (07:00-11:00 UTC)."]
    assert steps(rep, "operator")[0].startswith("Decide whether to promote fw-2026-11-07")
    state = sim.state()
    assert [x["adapter_version"] for x in state["models"]] == ["fw-2026-10-31", "fw-2026-11-07"]
    assert state["train"]["attempts"] == 2 and state["gpu"]["applied"] is False
    assert sorted(p.name for p in (sim.data / "reports").glob("*.md")) == [
        "2026-10-31-weekly.md", "2026-11-07-weekly.md", "2026-11-09-weekly-report-only.md",
        "2026-11-14-weekly.md", "2026-11-21-weekly.md", "LATEST.md"]


# --------------------------------------------------------------------------- #
# The guards, end to end
# --------------------------------------------------------------------------- #


def test_max_derived_data_is_refused_and_no_trainer_or_swap_starts(sim: Sim):
    sim.produce("golden", sim.exports / "golden_traces_2026-10-30.jsonl", 0, 12)
    # A builder with no data policy keeps only the Max model's rows, all trainable.
    legacy = sim.block(
        harvest={"teacher_models": ["qwen3.8-max"]},
        build={"args": ["--pack", "pack.json", "--tokenizer-dir", "tok", "--train-sources",
                        "none", "--teacher-model", "qwen3.8-max", "--min-train", "8",
                        "--spot-check-server", "http://127.0.0.1:8099"]})
    assert sim.run(W1, legacy) == fw.EXIT_OK
    assert sim.cmds() == ["build"]                       # no trainer, no dead-man, no swap
    assert not (sim.data / "train").exists()
    rep = sim.report()
    assert rep["status"] == "WARN"
    train = stage(rep, "train")
    assert train["status"] == fw.REFUSED
    assert "12 of 12 row(s) are Max-derived and no operator approval is recorded" \
           in train["reason"]
    assert "DISTILL-OPEN-QUESTIONS #6" in train["reason"]
    assert stage(rep, "quantize")["reason"] == "no finished adapter waits to be quantized"
    sec = sections(sim.md())
    assert "- Max-derived data: training refused (no operator approval recorded; " \
           "DISTILL-OPEN-QUESTIONS #6)" in sec["Guards"]
    assert "- Train data checked: 12 row(s), 12 Max-derived" in sec["Guards"]
    assert "- **train** REFUSED: 12 of 12 row(s) are Max-derived" in sec["Why stages did not run"]
    operator = steps(rep, "operator")
    assert operator[0].startswith("Decide DISTILL-OPEN-QUESTIONS #6")
    assert operator[1].startswith("flywheel.harvest.teacher_models names only Max models")
    assert not any(s.startswith("train was refused") for s in operator)   # said once, above

    # The guard alone held it: with the approval recorded, the same window trains.
    approved = {**legacy, **APPROVED}
    assert sim.run(W1 + timedelta(minutes=30), approved) == fw.EXIT_OK
    assert sim.cmds() == ["build", "deadman", "apply:flywheel", "train", "train-done",
                          "quantize", "score", "compare", "apply:business"]
    rep = sim.report()
    assert stage(rep, "build")["reason"] == "done earlier in this window"
    assert stage(rep, "train")["status"] == fw.OK
    assert "training ALLOWED, approved by operator on 2020-01-01" in sections(sim.md())["Guards"]


def test_offspring_output_is_never_trained_on(sim: Sim):
    assert week_one(sim, sim.block(**APPROVED)) == fw.EXIT_OK      # fw-2026-10-31 trained
    # The week: the student's own traces reach the golden exports (no verdict),
    # judged ones arrive as extra traces, and a faulty builder lets pooled
    # offspring rows into train/val labelled tool-success.
    export = sim.exports / "golden_traces_2026-11-06.jsonl"
    sim.produce("offspring", export, 500, 4, served=OFFSPRING_ID)
    export_bytes = export.read_bytes()
    sim.produce("gym", sim.gym / "gym-2026-11-06.jsonl", 100, 12)
    sim.produce("offspring", sim.gym / "offspring-2026-11-06.jsonl", 600, 3, served=OFFSPRING_ID,
                judged_by="qwen3.8-27b")
    faulty = sim.block(**APPROVED)
    faulty["build"] = {"args": [*faulty["build"]["args"], "--offspring-label", "tool-success"]}
    n = len(sim.calls())
    assert sim.run(W2, faulty) == fw.EXIT_OK
    assert sim.cmds(n) == ["build"]                       # no trainer, no swap
    build = next(c for c in sim.calls()[n:] if c["cmd"] == "build")
    assert build["withheld"] == 4                         # the golden-path offspring, as stubs
    copy = sim.data / "build-input" / "golden" / export.name
    stubs = [json.loads(x)["_meta"] for x in copy.read_text().splitlines()]
    assert stubs == [{"model": fw.WITHHELD_MODEL, "flywheel_withheld": "unverified_offspring"}] * 4
    assert export.read_bytes() == export_bytes            # the source is never modified
    rep = sim.report()
    train = stage(rep, "train")
    assert train["status"] == fw.REFUSED
    assert train["reason"] == ("3 offspring row(s) carry no judge verdict (unverified offspring "
                               "output is never trained on)")
    assert rep["guards"]["train_data"]["offspring_rows"] == 3
    sec = sections(sim.md())
    assert "3 offspring (3 unverified)" in sec["Guards"]
    assert "- Offspring traces dropped as unverified this window: 4" in sec["Guards"]
    assert steps(rep, "operator")[0] == ("train was refused: 3 offspring row(s) carry no judge "
                                         "verdict (unverified offspring output is never trained "
                                         "on)")
    assert sim.state()["train"]["adapter_version"] == "fw-2026-10-31"     # nothing new trained
