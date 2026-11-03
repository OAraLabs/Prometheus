"""scripts/distill/mini_window/gen_runner.py: teacher generation in a 3090 Ti window, against fakes.

The model port is a real HTTP server on loopback (an ephemeral port) answering with a teacher alias; the
probe guard, teacher_gen.py, spend.py and judge_run.sh are stand-ins (tests/fixtures/distill_gen_runner/)
that log what they were asked and write resumable outputs like the real ones. Nothing touches a GPU, a
systemd manager, a paid API or a real store. The runner itself runs as a subprocess, as under systemd.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "distill_gen_runner"
RUNNER = REPO / "scripts" / "distill" / "mini_window" / "gen_runner.py"
sys.path.insert(0, str(REPO / "scripts" / "distill" / "mini_window"))

import gen_runner as G  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="posix only")

ET = ZoneInfo("America/New_York")


def et_in(**delta: float) -> datetime:
    """Now + *delta* as an ET wall time. In UTC first: ``datetime.now(ET) + timedelta`` is
    wall-clock arithmetic (fold reset), an hour off inside the repeated hour of the fall back."""
    return (datetime.now(timezone.utc) + timedelta(**delta)).astimezone(ET)
IQ3 = "qwen3.8-27b-gsq-rco-iq3_xxs"
BUSINESS = "/models/local.gguf"


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


class Model:
    """The local model port: /v1/models answers with ``alias`` (switchable mid-run)."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = json.dumps({"data": [{"id": outer.alias, "aliases": [outer.alias]}]}
                                  if self.path == "/v1/models" else {"status": "ok"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class Rig:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = root / "state"
        self.state.mkdir()
        self.model = Model(IQ3)
        self.guard_port = _free_port()
        self.runs, self.pool, self.win = root / "runs", root / "pool", root / "window"
        self.win.mkdir()
        code = root / "code"
        dist = code / "scripts" / "distill"
        dist.mkdir(parents=True)
        (code / "src" / "prometheus").mkdir(parents=True)
        (code / "src" / "prometheus" / "__init__.py").write_text("")
        (code / "COMMIT").write_text("c" * 40 + "\n")
        shutil.copy(FIX / "fake_teacher_gen.py", dist / "teacher_gen.py")
        shutil.copy(FIX / "fake_spend.py", dist / "spend.py")
        shutil.copy(FIX / "fake_judge_run.sh", dist / "judge_run.sh")
        shutil.copy(FIX / "fake_taskset_gen.py", dist / "taskset_gen.py")
        shutil.copy(REPO / "scripts" / "distill" / "artifact_policy.py", dist / "artifact_policy.py")
        inp = root / "in"
        inp.mkdir()
        for n in ("tasks.jsonl", "manifest.json", "eval-tools.jsonl", "longctx-tools.jsonl", "pack.json"):
            (inp / n).write_text("{}\n")
        repo = root / "repo"
        repo.mkdir()
        (repo / "README.md").write_text("# readme\n")
        notify = root / "notify.sh"
        notify.write_text('#!/bin/sh\ncat >> "$FAKE_STATE/notify.log"\n')
        notify.chmod(0o755)
        self.proc = root / "proc"
        self.proc.mkdir()
        evals = ",".join(et_in(hours=h).strftime("%H%M") for h in (3, 4))
        self.envfile = root / "prometheus.env"
        self.envfile.write_text("# nothing\n")
        self.envfile.chmod(0o600)
        self.spend_stop = root / "spend-ledger.STOP"
        self.excl = root / "exclude-sessions.txt"
        self.excl.write_text("# the incident session\nbeacon:incident\n")
        self.sbx = root / "sbx"
        self.conf = root / "window.conf"
        self.conf.write_text("\n".join([
            "HOST_KEY=mini", f"WIN_DIR={self.win}", "LOCAL_MODEL_UNIT=local-model.service",
            "PROBE_UNIT=voice.service", "OLLAMA_MODEL=judge:7b", f"NOTIFY={notify}", f"PY={sys.executable}",
            f"MODEL_URL={self.model.url}", f"PROC_ROOT={self.proc}", f"EVAL_TIMES={evals}", "CONTINUOUS_UNTIL=",
            f"GEN_CODE={code}", f"GEN_PY={sys.executable}", f"GEN_RUNS={self.runs}", f"GEN_POOL={self.pool}",
            f"GEN_TASKS={inp / 'tasks.jsonl'}", f"GEN_INJECTION_MANIFEST={inp / 'manifest.json'}",
            f"GEN_EVAL_TOOLS='{inp / 'eval-tools.jsonl'} {inp / 'longctx-tools.jsonl'}'",
            f"GEN_PACK={inp / 'pack.json'}", f"GEN_REPO={repo}", f"GEN_GUARD_PY={FIX / 'fake_probe_guard.py'}",
            f"GEN_GUARD_LISTEN=127.0.0.1:{self.guard_port}", "GEN_NICE_CMD=nice -n 10",
            "GEN_MAX_NICE_CMD=nice -n 15", f"GEN_SPEND_STOP={self.spend_stop}", "GEN_POLL_S=0.1",
            "GEN_CHILD_GRACE_S=3", "GEN_RESTART_WAIT_S=0.3", "GEN_HARVEST_EVERY_MIN=0.01", "GEN_MIN_RUN_S=0.2",
            "GEN_MIN_START_S=0.5", "GEN_GUARD_WAIT_S=10", f"GEN_EXCLUDE_SESSIONS={self.excl}",
            f"GEN_SANDBOX_ROOT={self.sbx}", "",
        ]))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("QWEN_", "PYTHON"))}
        self.env.update(FAKE_STATE=str(self.state), PROMETHEUS_ENV_FILE=str(self.envfile), FAKE_TG_EVERY="0.1")
        self.procs: list[subprocess.Popen] = []

    # running

    def argv(self, cmd: str = "run", execute: bool = True, until_s: float | None = 3.0) -> list[str]:
        a = [sys.executable, "-I", str(RUNNER), "--conf", str(self.conf), cmd]
        if execute:
            a.append("--execute")
        if until_s is not None:
            a += ["--until", et_in(seconds=until_s).isoformat(timespec="seconds")]
        return a

    def run(self, cmd: str = "run", execute: bool = True, until_s: float | None = 3.0,
            env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv(cmd, execute, until_s), env={**self.env, **(env or {})},
                              capture_output=True, text=True, timeout=120)

    def start(self, until_s: float = 60, env: dict[str, str] | None = None) -> subprocess.Popen:
        p = subprocess.Popen(self.argv("run", True, until_s), env={**self.env, **(env or {})},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p

    def with_key(self) -> None:
        self.envfile.write_text("QWEN_PAYG_KEY=sk-not-a-real-key-just-a-test-value\n")

    # reading

    def calls(self, name: str) -> list:
        p = self.state / name
        if not p.exists():
            return []
        return [json.loads(x) if x.startswith("[") else x for x in p.read_text().splitlines()]

    def progress(self) -> dict:
        return json.loads((self.runs / "state" / "progress.json").read_text())

    def pid(self, name: str) -> int:
        return int((self.state / f"{name}-pid").read_text())

    def wait_for(self, cond, timeout: float = 30, what: str = "") -> None:
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}")

    def pool_rows(self, rel: str) -> list[dict]:
        p = self.pool / rel
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def close(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        pids = [self.state / f"{n}-pid" for n in ("local", "max", "judge")]
        pids += list((self.runs / "state").glob("*.pid")) if (self.runs / "state").exists() else []
        for f in pids:
            try:
                pid = int(f.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                continue
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
            if alive(pid) and str(self.root) in cmd or "fake_" in cmd and alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.model.close()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is gone for our purposes
        st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(st) and not st.startswith("Z")
    except OSError:
        return True


def finished(out: Path, name: str) -> list[dict]:
    """The trace rows of finished tasks (a run cut between its two writes leaves rows the ledger lacks)."""
    ok = G.finished_runs(out / f"{name}.tasks.jsonl")
    rows = [json.loads(x) for x in (out / f"{name}.traces.jsonl").read_text().splitlines() if x.endswith("}")]
    return [r for r in rows if r["_meta"]["run_id"] in ok]


@pytest.fixture
def rig(tmp_path: Path):
    r = Rig(tmp_path)
    yield r
    r.close()


def _opt(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


# ── dry run and refusals ─────────────────────────────────────────────────────


def test_dry_run_prints_the_plan_and_starts_nothing(rig: Rig) -> None:
    cp = rig.run(execute=False, until_s=3600)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    out = cp.stdout
    assert f"serves the teacher {IQ3}" in out
    assert f"--listen 127.0.0.1:{rig.guard_port} --upstream {rig.model.url[len('http://'):]}" in out
    assert "--probe-unit voice.service" in out and "--min-tps 15" in out
    assert "--teacher local" in out and "--concurrency 1" in out
    assert f"--probe-guard-url http://127.0.0.1:{rig.guard_port}" in out
    assert "paid branch (no key): SKIP (neither QWEN_PAYG_KEY nor (with GEN_TOKEN_PLAN=on) QWEN_API_KEY" in out
    assert "judge_run.sh --execute --provider payg --budget-usd 25 --blind-budget-usd 5" in out
    assert "--teacher max-payg --budget-usd 100" in out and "max-payg-max-derived" in out
    # each teacher_gen run its own sandbox root (they would wipe each other's workspaces)
    assert f"--sandbox-root {rig.sbx}-local" in out and f"--sandbox-root {rig.sbx}-max-payg" in out
    # the taskset was checked against the excluded sessions (read-only)
    (check,) = rig.calls("ts-calls.log")
    assert check[:2] == ["exclude", "--check"] and _opt(check, "--exclude-sessions") == str(rig.excl)
    assert _opt(check, "--tasks") == str(rig.root / "in" / "tasks.jsonl")
    assert "GEN_TASKS holds no row of an excluded session" in out
    assert not rig.runs.exists() and not rig.pool.exists()
    assert rig.calls("tg-calls.log") == [] and rig.calls("guard-calls.log") == []


@pytest.mark.parametrize("rc, words", [
    (1, "holds 9 rows built from an excluded session's prompts: run taskset_gen.py exclude"),
    (2, "the excluded-session check failed (exit 2): taskset_gen: lcm snapshot not found"),
])
def test_refuses_a_taskset_with_rows_of_an_excluded_session(rig: Rig, rc: int, words: str) -> None:
    cp = rig.run(env={"FAKE_TS_EXCL_RC": str(rc)})
    assert cp.returncode == 2 and words in cp.stdout, cp.stdout[-2000:]
    assert rig.calls("guard-calls.log") == [] and rig.calls("tg-calls.log") == []


def test_refuses_without_the_exclusion_list(rig: Rig) -> None:
    rig.excl.unlink()
    cp = rig.run(execute=False, until_s=3600)
    assert cp.returncode == 2 and "GEN_EXCLUDE_SESSIONS" in cp.stdout and "missing" in cp.stdout
    assert rig.calls("ts-calls.log") == []


def test_refuses_when_the_port_does_not_serve_a_teacher(rig: Rig) -> None:
    rig.model.alias = BUSINESS
    cp = rig.run()
    assert cp.returncode == 2
    assert "not a teacher" in cp.stdout and "no window is open" in cp.stdout
    assert rig.calls("guard-calls.log") == [] and rig.calls("tg-calls.log") == []
    assert "mini gen refused (nothing started)" in (rig.state / "notify.log").read_text()


def test_refuses_without_staged_code(rig: Rig) -> None:
    (rig.root / "code" / "COMMIT").unlink()
    cp = rig.run(execute=False, until_s=3600)
    assert cp.returncode == 2 and "code not staged" in cp.stdout


# ── the local branch ─────────────────────────────────────────────────────────


def test_local_run_until_then_harvest_and_resume(rig: Rig) -> None:
    cp = rig.run(until_s=3)
    assert cp.returncode == 0, cp.stdout[-3000:]
    (guard,) = rig.calls("guard-calls.log")
    assert _opt(guard, "--upstream") == rig.model.url[len("http://"):]
    assert _opt(guard, "--llama-unit") == "distill-teacher.service"
    assert _opt(guard, "--abort-file").endswith("runs/state/guard.abort")
    (local,) = rig.calls("tg-calls.log")
    assert local[:3] == ["run", "--teacher", "local"]
    assert _opt(local, "--concurrency") == "1"
    assert _opt(local, "--probe-guard-url") == f"http://127.0.0.1:{rig.guard_port}"
    assert _opt(local, "--teacher-url") == rig.model.url
    assert _opt(local, "--out") == str(rig.runs / f"local-{IQ3}")
    assert datetime.fromisoformat(_opt(local, "--until")).tzinfo is not None
    assert [x for x in local if x.startswith(str(rig.root / "in"))] and local.count("--eval-tools") == 2
    # everything stopped, the pool holds exactly the finished tasks' rows
    prog = rig.progress()
    assert prog["state"] == "finished" and prog["stop_reason"] == "until"
    assert not (rig.runs / "state" / "guard.pid").exists()
    traces = finished(rig.runs / f"local-{IQ3}", "qwen38-27b-gym")
    rows = rig.pool_rows(f"qwen38-27b-gym.local-{IQ3}.traces.jsonl")
    assert len(rows) == len(traces) > 0 and all(r["_meta"]["max_derived"] is False for r in rows)
    counts = json.loads((rig.pool / "harvest.json").read_text())
    assert counts["rows_by_source"] == {"qwen38-27b-gym": len(rows)}
    assert oct((rig.pool / f"qwen38-27b-gym.local-{IQ3}.traces.jsonl").stat().st_mode & 0o777) == "0o600"
    note = (rig.state / "notify.log").read_text()
    assert "mini gen finished (until)" in note and f"qwen38-27b-gym +{len(rows)}" in note
    # a second window resumes: teacher_gen continues its ledger, the harvest copies nothing twice
    assert rig.run(until_s=3).returncode == 0
    traces = finished(rig.runs / f"local-{IQ3}", "qwen38-27b-gym")
    rows2 = rig.pool_rows(f"qwen38-27b-gym.local-{IQ3}.traces.jsonl")
    assert len(rows2) == len(traces) > len(rows)
    assert len({(r["_meta"]["run_id"], r["_meta"]["step"]) for r in rows2}) == len(rows2)


def test_sigterm_stops_the_children_and_the_guard(rig: Rig) -> None:
    p = rig.start(until_s=60)
    rig.wait_for(lambda: (rig.state / "local-pid").exists(), what="the local teacher_gen")
    local = rig.pid("local")
    guard = int((rig.runs / "state" / "guard.pid").read_text())
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143, p.stdout.read()[-2000:]
    assert time.time() - t0 < 10
    assert not alive(local) and not alive(guard)
    prog = rig.progress()
    assert prog["state"] == "stopped" and prog["stop_reason"] == "sigterm"
    assert prog["local"]["state"] == "stopped"


def test_one_runner_at_a_time(rig: Rig) -> None:
    rig.start(until_s=60)
    rig.wait_for(lambda: (rig.state / "local-pid").exists(), what="the first runner")
    cp = rig.run(until_s=30)
    assert cp.returncode == 2 and "another generation runner holds the lock" in cp.stdout


def test_guard_abort_stops_local_generation_for_the_run(rig: Rig) -> None:
    p = rig.start(until_s=60)
    rig.wait_for(lambda: (rig.state / "local-pid").exists(), what="the local teacher_gen")
    local = rig.pid("local")
    (rig.state / "guard-trip").write_text("")
    assert p.wait(timeout=30) == 0       # local stopped + no paid branch: nothing left to do
    assert not alive(local)
    prog = rig.progress()
    assert prog["local"]["state"] == "stopped" and "abort switch tripped" in prog["local"]["detail"]
    assert prog["stop_reason"] == "nothing left to do"
    assert len(rig.calls("tg-calls.log")) == 1                         # never restarted
    # the next run starts a fresh guard (the earlier abort is kept as .prev)
    (rig.state / "guard-trip").unlink()
    assert rig.run(until_s=3).returncode == 0
    assert (rig.runs / "state" / "guard.prev").exists()


def test_a_nightly_eval_pauses_local_generation_then_it_resumes(rig: Rig) -> None:
    p = rig.start(until_s=60)
    rig.wait_for(lambda: (rig.state / "local-pid").exists(), what="the local teacher_gen")
    first = rig.pid("local")
    ev = rig.proc / "4242"
    ev.mkdir()
    (ev / "cmdline").write_bytes(b"/usr/bin/python3\0scripts/run_nightly_evals.py\0")
    rig.wait_for(lambda: not alive(first), what="the pause")
    rig.wait_for(lambda: rig.progress()["local"]["state"] == "paused", what="paused state")
    time.sleep(0.5)
    assert len(rig.calls("tg-calls.log")) == 1                         # nothing started during the eval
    shutil.rmtree(ev)
    rig.wait_for(lambda: len(rig.calls("tg-calls.log")) == 2, what="the restart")
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)
    assert any("paused: a nightly eval is running" in n for n in rig.progress()["notes"])


def test_local_exit_3_is_restarted_then_given_up(rig: Rig) -> None:
    rig.conf.write_text(rig.conf.read_text() + "GEN_LOCAL_RESTARTS=2\n")
    cp = rig.run(until_s=20, env={"FAKE_TG_FAIL_LOCAL": "10"})
    assert cp.returncode == 1, cp.stdout[-3000:]
    assert len(rig.calls("tg-calls.log")) == 3
    prog = rig.progress()
    assert prog["local"]["state"] == "failed" and prog["local"]["exits"] == [3, 3, 3]


@pytest.mark.parametrize("rc, report", [(1, False), (6, False)])
def test_a_local_crash_is_restarted_never_taken_for_done(rig: Rig, rc: int, report: bool) -> None:
    # exit 1 without a report from that run is Python's own code for an uncaught exception; 6 is
    # teacher_gen's crash exit: neither is "kept nothing"
    rig.conf.write_text(rig.conf.read_text() + "GEN_LOCAL_RESTARTS=2\n")
    cp = rig.run(until_s=20, env={"FAKE_TG_LOCAL_RC": str(rc)})
    assert cp.returncode == 1, cp.stdout[-3000:]
    assert len(rig.calls("tg-calls.log")) == 3
    prog = rig.progress()
    assert prog["local"]["state"] == "failed" and prog["local"]["exits"] == [rc, rc, rc]
    assert any(f"local teacher_gen crashed (exit {rc}" in n for n in prog["notes"])


def test_local_exit_1_with_its_report_is_done(rig: Rig) -> None:
    cp = rig.run(until_s=20, env={"FAKE_TG_LOCAL_RC": "1", "FAKE_TG_LOCAL_REPORT": "1"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert len(rig.calls("tg-calls.log")) == 1
    prog = rig.progress()
    assert prog["local"]["state"] == "done" and "kept nothing" in prog["local"]["detail"]


def test_a_dry_taskset_goes_on_with_the_fallbacks_and_tells_the_operator(rig: Rig) -> None:
    inp = rig.root / "in"
    for n in ("fb1.jsonl", "fb2.jsonl"):
        (inp / n).write_text("{}\n")
    rig.conf.write_text(rig.conf.read_text() + f"GEN_TASKS_FALLBACK='{inp / 'fb1.jsonl'} {inp / 'gone.jsonl'} "
                                               f"{inp / 'fb2.jsonl'}'\n")
    cp = rig.run(until_s=6, env={"FAKE_TG_DRY": "tasks.jsonl,fb1.jsonl"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    used = [_opt(c, "--tasks") for c in rig.calls("tg-calls.log")]
    assert [Path(u).name for u in used] == ["tasks.jsonl", "fb1.jsonl", "fb2.jsonl"]
    # each fallback passed the excluded-session check; the missing one was dropped, said in a note
    checked = [_opt(c, "--tasks") for c in rig.calls("ts-calls.log")]
    assert [Path(x).name for x in checked] == ["tasks.jsonl", "fb1.jsonl", "fb2.jsonl"]
    prog = rig.progress()
    assert prog["tasks"].endswith("fb2.jsonl") and prog["stop_reason"] == "until"
    assert any("fallback taskset dropped" in n and "gone.jsonl missing" in n for n in prog["notes"])
    notes = (rig.state / "notify.log").read_text()
    assert notes.count("has nothing left that fits this teacher") == 2
    assert "going on with the fallback fb1.jsonl" in notes and "going on with the fallback fb2.jsonl" in notes
    assert finished(rig.runs / f"local-{IQ3}", "qwen38-27b-gym")


def test_a_dry_taskset_without_a_fallback_ends_the_branch_and_says_so(rig: Rig) -> None:
    cp = rig.run(until_s=20, env={"FAKE_TG_DRY": "tasks.jsonl"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert len(rig.calls("tg-calls.log")) == 1
    prog = rig.progress()
    assert prog["local"]["state"] == "done" and "no_tasks" in prog["local"]["detail"]
    assert prog["stop_reason"] == "nothing left to do"
    notes = (rig.state / "notify.log").read_text()
    assert "no fallback taskset left" in notes and "early" in notes
    # at most 5 lines per notice (the exhaustion one, then the end of the run)
    assert all(len(block.strip().splitlines()) <= 5 for block in notes.split("mini gen ") if block.strip())

def test_the_window_closing_stops_everything(rig: Rig) -> None:
    p = rig.start(until_s=60)
    rig.wait_for(lambda: (rig.state / "local-pid").exists(), what="the local teacher_gen")
    rig.model.alias = BUSINESS                                           # business restored on the port
    assert p.wait(timeout=30) == 3
    assert rig.progress()["stop_reason"] == "window-closed"
    assert not alive(rig.pid("local"))


def test_the_no_swaps_flag_or_the_gen_stop_file_stop_everything(rig: Rig) -> None:
    p = rig.start(until_s=60)
    rig.wait_for(lambda: (rig.state / "local-pid").exists(), what="the local teacher_gen")
    (rig.win / "NO-SWAPS-TONIGHT").write_text("restore failed\n")
    assert p.wait(timeout=30) == 3 and rig.progress()["stop_reason"] == "no-swaps"
    (rig.win / "NO-SWAPS-TONIGHT").unlink()
    (rig.runs / "STOP").write_text("operator\n")
    cp = rig.run(until_s=5)
    assert cp.returncode == 2 and "runs/STOP is present" in cp.stdout


# ── the paid branch ──────────────────────────────────────────────────────────


def test_paid_branch_judges_first_then_the_max_teacher_and_stops_at_the_spend_stop(rig: Rig) -> None:
    rig.with_key()
    p = rig.start(until_s=60, env={"FAKE_JUDGE_S": "0.5"})
    rig.wait_for(lambda: (rig.state / "max-pid").exists(), what="the Max teacher")
    judge = rig.calls("judge-calls.log")
    assert len(judge) == 1 and judge[0].startswith("--execute --provider payg --budget-usd 25 --blind-budget-usd 5 |")
    assert f"PY={sys.executable}" in judge[0] and f"PYTHONPATH={rig.root / 'code' / 'src'}" in judge[0]
    assert "MODEL=qwen3.8-max" in judge[0]
    tg = rig.calls("tg-calls.log")
    local, paid = tg[0], tg[1]
    assert _opt(local, "--teacher") == "local"
    assert paid[:5] == ["run", "--teacher", "max-payg", "--budget-usd", "100"]
    assert _opt(paid, "--out") == str(rig.runs / "max-payg-max-derived")
    assert _opt(paid, "--judge-url") == rig.model.url                  # the local 27B judges ...
    assert _opt(paid, "--probe-guard-url") == f"http://127.0.0.1:{rig.guard_port}"   # ... through the guard
    assert _opt(local, "--sandbox-root") == f"{rig.sbx}-local" != _opt(paid, "--sandbox-root")
    assert "--teacher-url" not in paid and _opt(paid, "--concurrency") == "2"
    spend = rig.calls("spend-calls.log")                               # the plan, the start, after the judge
    assert len(spend) >= 3 and all(c[0] == "preflight" and _opt(c, "--model") == "qwen3.8-max" for c in spend)
    rig.wait_for(lambda: (rig.runs / "max-payg-max-derived" / "max-payg-gym.tasks.jsonl").exists(), what="Max rows")
    max_pid, local_pid = rig.pid("max"), rig.pid("local")
    rig.spend_stop.write_text("$140 alert\n")
    rig.wait_for(lambda: not alive(max_pid), what="the Max teacher's stop")
    time.sleep(0.5)
    assert alive(local_pid)                                            # $0: the local branch goes on
    assert rig.progress()["paid"]["state"] == "stopped" and "spend STOP" in rig.progress()["paid"]["detail"]
    (rig.runs / "STOP").write_text("operator\n")
    assert p.wait(timeout=30) == 3
    assert len([c for c in rig.calls("tg-calls.log") if "max-payg" in c]) == 1     # never restarted
    rows = rig.pool_rows("max-derived/max-payg-gym.max-payg-max-derived.traces.jsonl")
    assert rows and all(r["_meta"]["max_derived"] is True for r in rows)
    assert not list(rig.pool.glob("max-payg-gym*"))                    # never outside max-derived/
    counts = json.loads((rig.pool / "harvest.json").read_text())["rows_by_source"]
    assert counts["max-payg-gym (max-derived)"] == len(rows) and counts["qwen38-27b-gym"] > 0


def test_a_nightly_eval_pauses_the_max_teacher_whose_judge_is_local(rig: Rig) -> None:
    rig.with_key()
    p = rig.start(until_s=60, env={"FAKE_JUDGE_S": "0.2"})
    rig.wait_for(lambda: (rig.state / "max-pid").exists(), what="the Max teacher")
    first_max, first_local = rig.pid("max"), rig.pid("local")
    ev = rig.proc / "4243"
    ev.mkdir()
    (ev / "cmdline").write_bytes(b"/usr/bin/python3\0scripts/run_nightly_evals.py\0")
    # both users of the teacher's server stop: the local teacher and the Max run's local judge
    rig.wait_for(lambda: not alive(first_max) and not alive(first_local), what="both paused")
    rig.wait_for(lambda: rig.progress()["paid"]["state"] == "paused", what="paid paused")
    assert rig.progress()["local"]["state"] == "paused"
    time.sleep(0.5)
    assert len([c for c in rig.calls("tg-calls.log") if "max-payg" in c]) == 1   # nothing started meanwhile
    shutil.rmtree(ev)
    rig.wait_for(lambda: len([c for c in rig.calls("tg-calls.log") if "max-payg" in c]) == 2, what="Max resumes")
    rig.wait_for(lambda: rig.progress()["paid"]["state"] == "teacher", what="paid running again")
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)
    notes = rig.progress()["notes"]
    assert any("Max teacher paused" in n for n in notes) and any("Max teacher resumes" in n for n in notes)
    assert rig.progress()["paid"]["state"] != "failed"


def test_with_max_judging_itself_the_max_teacher_runs_through_an_eval(rig: Rig) -> None:
    rig.with_key()
    rig.conf.write_text(rig.conf.read_text() + "GEN_MAX_JUDGE=max\n")
    p = rig.start(until_s=60, env={"FAKE_JUDGE_S": "0.2"})
    rig.wait_for(lambda: (rig.state / "max-pid").exists(), what="the Max teacher")
    first_max = rig.pid("max")
    ev = rig.proc / "4244"
    ev.mkdir()
    (ev / "cmdline").write_bytes(b"/usr/bin/python3\0scripts/run_nightly_evals.py\0")
    rig.wait_for(lambda: rig.progress()["local"]["state"] == "paused", what="local paused")
    assert alive(first_max) and rig.progress()["paid"]["state"] == "teacher"
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)


def test_key_present_but_preflight_refused_skips_the_paid_branch(rig: Rig) -> None:
    rig.with_key()
    cp = rig.run(until_s=3, env={"FAKE_SPEND_RC": "2"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert "paid branch (payg): SKIP (spend preflight refused (exit 2): refused: STOP is active" in cp.stdout
    assert rig.calls("judge-calls.log") == [] and len(rig.calls("tg-calls.log")) == 1
    assert rig.progress()["paid"]["state"] == "off"


def test_a_teacher_gen_without_the_max_mode_runs_the_judge_only(rig: Rig) -> None:
    rig.with_key()
    cp = rig.run(until_s=3, env={"FAKE_TG_OLD": "1"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    (local,) = rig.calls("tg-calls.log")
    assert "--teacher" not in local                                    # an older teacher_gen: local only
    assert len(rig.calls("judge-calls.log")) == 1
    prog = rig.progress()
    assert prog["paid"]["state"] == "off" and "no --teacher max-payg" in prog["paid"]["detail"]


@pytest.mark.parametrize("rc, state, words, runner_rc", [
    (4, "done", "spend ledger refused a paid call (phase budget $100", 0),   # its phase budget is spent
    (5, "stopped", "refused (fatal)", 0),                                     # the key / account refused
    (2, "failed", "Max teacher exit 2: see", 1),                              # a real failure stays one
])
def test_the_max_teachers_spend_and_key_exits_are_not_failures(rig: Rig, rc: int, state: str, words: str,
                                                               runner_rc: int) -> None:
    rig.with_key()
    cp = rig.run(until_s=4, env={"FAKE_TG_MAX_RC": str(rc)})
    assert cp.returncode == runner_rc, cp.stdout[-3000:]
    prog = rig.progress()
    assert prog["paid"]["state"] == state and words in prog["paid"]["detail"], prog["paid"]
    assert prog["paid"]["exits"] == [0, rc]                                   # the judge, then the teacher
    assert len([c for c in rig.calls("tg-calls.log") if "max-payg" in c]) == 1     # never restarted
    assert any(words in n for n in prog["notes"])                             # the operator hears of it
    assert prog["local"]["state"] != "failed"


def test_the_judge_failing_fatally_starts_no_max_teacher(rig: Rig) -> None:
    rig.with_key()
    cp = rig.run(until_s=3, env={"FAKE_JUDGE_RC": "5"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert not [c for c in rig.calls("tg-calls.log") if "max-payg" in c]
    assert "stopped fatally" in rig.progress()["paid"]["detail"]


# ── units: the harvest, --until, the key check ──────────────────────────────


def _ledger(d: Path, name: str, events: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.tasks.jsonl", "a") as f:
        f.write("".join(json.dumps(e) + "\n" for e in events))


def _rows(d: Path, name: str, rows: list[dict], torn: str = "") -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.traces.jsonl", "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows) + torn)


def test_harvest_takes_finished_tasks_only_and_routes_max_rows(tmp_path: Path) -> None:
    runs, pool = tmp_path / "runs", tmp_path / "pool"
    policy = REPO / "scripts" / "distill" / "artifact_policy.py"
    loc = runs / "local-iq3"
    clean = {"source": "qwen38-27b-gym", "model": "qwen3.8-27b", "max_derived": False}
    _rows(loc, "g", [{"_meta": dict(clean, run_id="r1")}, {"_meta": dict(clean, run_id="r1")},
                     {"_meta": dict(clean, run_id="r2")},                         # no ledger event: not finished
                     {"_meta": dict(clean, run_id="r3")},                         # held, then released
                     {"_meta": {"source": "qwen38-27b-gym", "run_id": "r4"}},      # no provenance: fail closed
                     {"_meta": dict(clean, run_id="r5", served_model="qwen3.8-max")}],  # names Max
          torn='{"_meta": {"run_id": "r1"')
    _ledger(loc, "g", [{"task_id": "t1", "status": "kept", "run_id": "r1"},
                       {"task_id": "t3", "status": "held", "run_id": "r3"},
                       {"task_id": "t3", "status": "released"},
                       {"task_id": "t4", "status": "kept", "run_id": "r4"},
                       {"task_id": "t5", "status": "kept", "run_id": "r5"},
                       {"task_id": "t6", "status": "rejected", "run_id": "r6"}])
    _rows(loc, "g.relaxed", [{"_meta": dict(clean, run_id="r1")}])               # never harvested
    dry = G.harvest(runs, pool, policy, execute=False, log=lambda s: None)
    assert dry["added"] == {"qwen38-27b-gym": 3, "qwen38-27b-gym (max-derived)": 2} and not pool.exists()
    res = G.harvest(runs, pool, policy, execute=True, log=lambda s: None)
    assert res["added"] == dry["added"] and res["unknown_provenance"] == 1
    clean_rows = [json.loads(x)["_meta"]["run_id"] for x in (pool / "qwen38-27b-gym.local-iq3.traces.jsonl")
                  .read_text().splitlines()]
    assert clean_rows == ["r1", "r1", "r3"]
    maxed = [json.loads(x)["_meta"]["run_id"] for x in (pool / "max-derived" / "qwen38-27b-gym.local-iq3.traces.jsonl")
             .read_text().splitlines()]
    assert sorted(maxed) == ["r4", "r5"]
    # again: nothing new; then a crash between an append and the state write copies nothing twice
    assert G.harvest(runs, pool, policy, execute=True, log=lambda s: None)["added"] == {}
    tr = loc / "g.traces.jsonl"                                       # teacher_gen's resume cuts the torn line
    tr.write_text(tr.read_text()[: tr.read_text().rfind("\n") + 1])
    _rows(loc, "g", [{"_meta": dict(clean, run_id="r7")}])
    _ledger(loc, "g", [{"task_id": "t7", "status": "kept", "run_id": "r7"}])
    with open(pool / "qwen38-27b-gym.local-iq3.traces.jsonl", "a") as f:     # appended, state never written
        f.write(json.dumps({"_meta": dict(clean, run_id="r7")}) + "\n")
    # the unrecorded append is cut back and copied again: once
    assert G.harvest(runs, pool, policy, execute=True, log=lambda s: None)["added"] == {"qwen38-27b-gym": 1}
    summary = json.loads((pool / "harvest.json").read_text())
    assert summary["rows_by_source"] == {"qwen38-27b-gym": 4, "qwen38-27b-gym (max-derived)": 2}
    clean_rows = [json.loads(x)["_meta"]["run_id"] for x in (pool / "qwen38-27b-gym.local-iq3.traces.jsonl")
                  .read_text().splitlines()]
    assert clean_rows == ["r1", "r1", "r3", "r7"]


def test_an_interrupted_harvest_copies_nothing_twice_and_leaves_no_torn_line(tmp_path: Path) -> None:
    runs, pool = tmp_path / "runs", tmp_path / "pool"
    policy = REPO / "scripts" / "distill" / "artifact_policy.py"
    clean = {"source": "qwen38-27b-gym", "model": "qwen3.8-27b", "max_derived": False}
    loc = runs / "local-iq3"
    _rows(loc, "g", [{"_meta": dict(clean, run_id="r1", step=k)} for k in (0, 1)])
    _ledger(loc, "g", [{"task_id": "t1", "status": "kept", "run_id": "r1"}])
    harvest = lambda: G.harvest(runs, pool, policy, execute=True, log=lambda s: None)  # noqa: E731
    # the first harvest records an empty state before its first append, then dies after the append (a new
    # pool file the state never heard of): the next one removes that file and copies its rows once
    assert harvest()["added"] == {"qwen38-27b-gym": 2}
    state = json.loads((pool / ".harvest-state.json").read_text())
    (pool / ".harvest-state.json").write_text(json.dumps({"schema": state["schema"], "files": {}}))
    assert harvest()["added"] == {"qwen38-27b-gym": 2}
    dest = pool / "qwen38-27b-gym.local-iq3.traces.jsonl"
    assert [json.loads(x)["_meta"]["step"] for x in dest.read_text().splitlines()] == [0, 1]
    # a new teacher's first rows (a new pool file), interrupted after the append: copied once
    q4 = runs / "local-q4"
    _rows(q4, "g", [{"_meta": dict(clean, run_id="r2", step=k)} for k in (0, 1)])
    _ledger(q4, "g", [{"task_id": "t2", "status": "kept", "run_id": "r2"}])
    new = pool / "qwen38-27b-gym.local-q4.traces.jsonl"
    new.write_text("".join(json.dumps({"_meta": dict(clean, run_id="r2", step=k)}) + "\n" for k in (0, 1)))
    assert harvest()["added"] == {"qwen38-27b-gym": 2}
    assert [json.loads(x)["_meta"]["run_id"] for x in new.read_text().splitlines()] == ["r2", "r2"]
    # an append killed partway (a torn line at the end of a recorded file): cut back, then appended whole
    _rows(loc, "g", [{"_meta": dict(clean, run_id="r3", step=k)} for k in (0, 1)])
    _ledger(loc, "g", [{"task_id": "t3", "status": "kept", "run_id": "r3"}])
    with open(dest, "a") as f:
        f.write(json.dumps({"_meta": dict(clean, run_id="r3", step=0)}) + "\n" + '{"_meta": {"run_id": "r3", "st')
    assert harvest()["added"] == {"qwen38-27b-gym": 2}
    lines = dest.read_text().splitlines()
    assert [(json.loads(x)["_meta"]["run_id"], json.loads(x)["_meta"]["step"]) for x in lines] == [
        ("r1", 0), ("r1", 1), ("r3", 0), ("r3", 1)]
    assert harvest()["added"] == {}
    assert json.loads((pool / "harvest.json").read_text())["rows_by_source"] == {"qwen38-27b-gym": 6}
    # no state at all (deleted by hand): what the pool files hold counts, nothing is copied again
    (pool / ".harvest-state.json").unlink()
    assert harvest()["added"] == {}
    assert len(dest.read_text().splitlines()) == 4


def _gen(conf_extra: str, now: str, tmp_path: Path) -> G.Gen:
    conf = tmp_path / "w.conf"
    conf.write_text("LOCAL_MODEL_UNIT=m.service\nPROBE_UNIT=p.service\nOLLAMA_MODEL=j:7b\n" + conf_extra)
    t = datetime.fromisoformat(now).replace(tzinfo=ET).timestamp()
    return G.Gen(G.load(conf), False, out=lambda s: None, clock=lambda: t, env={})


def test_until_defaults_follow_the_window(tmp_path: Path) -> None:
    g = _gen("CONTINUOUS_UNTIL=2026-11-01T18:00:00-05:00\n", "2026-10-31T07:30:00", tmp_path)
    assert g.resolve_until(None) == datetime.fromisoformat("2026-11-01T17:50:00-05:00").timestamp()
    g = _gen("CONTINUOUS_UNTIL=2026-11-01T18:00:00-05:00\n", "2026-11-02T01:30:00", tmp_path)
    assert g.resolve_until(None) == datetime.fromisoformat("2026-11-02T06:35:00-05:00").timestamp()
    g = _gen("", "2026-11-02T14:00:00", tmp_path)
    with pytest.raises(G.Skip, match="outside the nightly window"):
        g.resolve_until(None)
    assert g.resolve_until("05:00") == datetime.fromisoformat("2026-11-03T05:00:00-05:00").timestamp()
    with pytest.raises(G.Skip, match="more than 40 h"):
        g.resolve_until("2026-11-05T05:00:00-05:00")
    with pytest.raises(G.Skip, match="less than 600 s"):
        g.resolve_until("14:05")


def test_key_present_checks_presence_only(tmp_path: Path) -> None:
    f = tmp_path / "env"
    f.write_text("QWEN_API_KEY=sk-sp-x\n# QWEN_PAYG_KEY=commented\nQWEN_PAYG_KEY=\n")
    assert not G.key_present("QWEN_PAYG_KEY", {"PROMETHEUS_ENV_FILE": str(f)})
    f.write_text("export QWEN_PAYG_KEY='sk-abc'\n")
    assert G.key_present("QWEN_PAYG_KEY", {"PROMETHEUS_ENV_FILE": str(f)})
    assert G.key_present("QWEN_PAYG_KEY", {"QWEN_PAYG_KEY": "sk-x", "PROMETHEUS_ENV_FILE": str(tmp_path / "none")})
    assert not G.key_present("QWEN_PAYG_KEY", {"PROMETHEUS_ENV_FILE": str(tmp_path / "none")})


def test_eval_pause_covers_the_minutes_around_a_nightly_eval(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    for now, want in (("2026-11-02T02:54:00", None), ("2026-11-02T02:56:00", "starts in 4 min"),
                      ("2026-11-02T03:01:30", "the 0300 nightly eval is due"), ("2026-11-02T03:04:00", None)):
        g = _gen(f"PROC_ROOT={proc}\n", now, tmp_path)
        got = g.eval_pause()
        assert (got is None) if want is None else (want in got), (now, got)
    (proc / "77").mkdir()
    (proc / "77" / "cmdline").write_bytes(b"python3\0scripts/run_nightly_evals.py\0")
    assert _gen(f"PROC_ROOT={proc}\n", "2026-11-02T03:20:00", tmp_path).eval_pause() == "a nightly eval is running"


# ── DST: next occurrences and minutes are computed in UTC ────────────────────

def _utc(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def test_next_wall_time_is_dst_safe() -> None:
    # 2026-11-01: 01:00-02:00 ET happens twice (EDT 05:00-06:00 UTC, then EST 06:00-07:00 UTC)
    second_pass = _utc("2026-11-01T06:30:00")          # 01:30 EST
    assert G.MW.next_wall(second_pass, 1, 45, ET) == _utc("2026-11-01T06:45:00")   # 15 min ahead, not passed
    first_pass = _utc("2026-11-01T05:30:00")           # 01:30 EDT
    assert G.MW.next_wall(first_pass, 1, 15, ET) == _utc("2026-11-01T06:15:00")    # the EST 01:15, 45 min ahead
    assert G.MW.next_wall(first_pass, 1, 45, ET) == _utc("2026-11-01T05:45:00")    # the EDT 01:45 comes first
    assert G.MW.next_wall(second_pass, 6, 35, ET) == _utc("2026-11-01T11:35:00")   # 06:35 EST
    # 2027-03-14: 02:00-03:00 ET does not exist; 03:03 EDT is 5 min after 01:58 EST
    spring = _utc("2027-03-14T06:58:00")
    assert G.MW.next_wall(spring, 3, 3, ET) == _utc("2027-03-14T07:03:00")
    assert G.MW.next_wall(spring, 2, 30, ET) == _utc("2027-03-14T07:30:00")        # a skipped time: fold 0
    assert G.MW.next_wall(_utc("2026-11-02T12:00:00"), 6, 0, ET) == _utc("2026-11-03T11:00:00")


def test_eval_minutes_and_until_across_dst(tmp_path: Path) -> None:
    def gen(at_utc: str, evals: str) -> G.Gen:
        conf = tmp_path / "w.conf"
        conf.write_text("LOCAL_MODEL_UNIT=m.service\nPROBE_UNIT=p.service\nOLLAMA_MODEL=j:7b\n"
                        f"EVAL_TIMES={evals}\n")
        t = _utc(at_utc)
        return G.Gen(G.load(conf), False, out=lambda s: None, clock=lambda: t, env={})

    # spring forward: an eval at 03:03 EDT is 5 minutes after 01:58 EST (wall clock says 65)
    g = gen("2027-03-14T06:58:00", "0303")
    assert round(g.w.minutes_to_next_eval()) == 5
    # fall back: from 01:30 EDT an eval at 02:30 EST is 2 h away (wall clock says 1 h)
    assert round(gen("2026-11-01T05:30:00", "0230").w.minutes_to_next_eval()) == 120
    # --until HH:MM in the repeated hour: the occurrence still ahead, never one that has passed
    g = gen("2026-11-01T06:30:00", "0430")
    g.c["GEN_MIN_RUN_S"] = "60"
    assert g.resolve_until("01:45") == _utc("2026-11-01T06:45:00")  # 01:45 EST, not the passed 01:45 EDT
    assert g.resolve_until("05:30") == _utc("2026-11-01T10:30:00")
    rg = G.RemoteGen(G.load(tmp_path / "w.conf"), False, out=lambda s: None, clock=lambda: _utc("2026-11-01T05:30:00"),
                     env={})
    assert rg.resolve_until("05:30") == _utc("2026-11-01T10:30:00")


# ── a probe trip while generation is paused for a nightly eval (operator ruling 2026-11-03) ──────────


class _FakeProc:
    pid = 4321


class _FakeChild:
    """A child of the runner (the guard, a teacher_gen): running until stopped; nothing is spawned."""

    def __init__(self, name: str) -> None:
        self.name, self.proc, self.log_path, self._rc = name, _FakeProc(), Path("/dev/null"), None

    @property
    def running(self) -> bool:
        return self._rc is None

    def poll(self) -> int | None:
        return self._rc

    def stop(self, grace: float, pid_dir: Path | None) -> int | None:
        if self._rc is None:
            self._rc = 0
        return self._rc


class _Night:
    """A Gen driven tick by tick with a fake clock, a fake guard, a fake probe journal and a switchable
    nightly-eval pause; the local branch only (the paid branch is off)."""

    def __init__(self, tmp_path: Path) -> None:
        conf = tmp_path / "w.conf"
        conf.write_text("LOCAL_MODEL_UNIT=m.service\nPROBE_UNIT=p.service\nOLLAMA_MODEL=j:7b\nPROBE_MIN_TPS=15\n"
                        f"GEN_RUNS={tmp_path / 'runs'}\nGEN_POOL={tmp_path / 'pool'}\n")
        self.t = self.at("02:40:00")
        self.g = g = G.Gen(G.load(conf), False, out=lambda s: None, clock=lambda: self.t, env={})
        g.state_dir.mkdir(parents=True)
        g.alias = IQ3
        g.paid.state = "off"
        self.until = self.at("06:35:00")
        self.eval: str | None = None
        self.journal: list[G.MW.Probe] = []
        self.journal_reads = 0
        self.notified: list[list[str]] = []
        self.guards_started = 0
        self.local_starts = 0
        g.eval_pause = lambda: self.eval  # type: ignore[method-assign]
        g.w.probes = self._probes  # type: ignore[method-assign]
        g.notify = self.notified.append  # type: ignore[method-assign]
        g.start_guard = self._start_guard  # type: ignore[method-assign]
        g.start_child = self._start_child  # type: ignore[method-assign]
        g.guard = _FakeChild("guard")

    @staticmethod
    def at(hms: str) -> float:
        return datetime.fromisoformat(f"2026-11-03T{hms}").replace(tzinfo=ET).timestamp()

    def _probes(self, since_s: int = 900) -> list:
        self.journal_reads += 1
        return [p for p in self.journal if self.t - since_s <= p.t <= self.t]

    def _start_guard(self, until: float) -> None:
        self.guards_started += 1
        self.g.guard = _FakeChild("guard")

    def _start_child(self, name: str, argv: list[str], env: dict[str, str]) -> _FakeChild:
        assert name == "local"
        self.local_starts += 1
        return _FakeChild(name)

    def probe(self, hms: str, tps: float, verdict: str = "OK") -> None:
        self.journal.append(G.MW.Probe(self.at(hms), verdict, tps, 15.0))

    def trip(self, hms: str, tps: float) -> None:
        """A bad probe, and the guard's abort switch as the real guard writes it (the guard stays up)."""
        self.probe(hms, tps, "SLOW")
        self.g.abort_file.write_text(f"2026-11-03T{hms}-05:00 probe at 2026-11-03T{hms}-05:00: {tps:g} tok/s < 15\n")

    def tick(self, hms: str) -> None:
        self.t = self.at(hms)
        self.g.tick_guard(self.until)
        self.g.tick_local(self.until)

    def notes(self, words: str) -> list[str]:
        return [n for n in self.g.notes if words in n]

    def paused_for_the_eval(self) -> None:
        self.tick("02:50:00")
        assert self.g.local.state == "running" and self.local_starts == 1
        self.eval = "a nightly eval is running"
        self.tick("03:00:30")
        assert self.g.local.state == "paused" and not self.g.local.child.running


def test_a_trip_during_the_eval_pause_rearms_and_resumes_after_an_ok_probe(tmp_path: Path) -> None:
    n = _Night(tmp_path)
    n.paused_for_the_eval()
    old_guard = n.g.guard
    n.trip("03:01:58", 7.4)
    n.tick("03:02:20")
    assert n.g.local.state == "paused" and n.g.rearm is not None      # the run goes on
    assert not old_guard.running and not n.g.abort_file.exists()
    assert n.g.abort_file.with_name("guard.abort.forgiven").exists()
    assert len(n.notified) == 1 and "forgiven" in n.notified[0][0] and len(n.notified[0]) == 1
    assert n.notes("tripped while generation was paused for a nightly eval")
    n.tick("03:15:00")                                                  # the eval still runs
    assert n.local_starts == 1 and n.guards_started == 0
    n.eval = None
    n.tick("03:20:00")                                                  # eval over: no probe after it yet
    assert n.g.local.state == "paused" and n.local_starts == 1 and n.guards_started == 0
    assert "first voice-probe result after the nightly eval" in (n.g.gen_pause() or "")
    n.probe("03:21:40", 50.0)
    n.tick("03:21:50")
    assert n.guards_started == 1 and n.g.rearm is None                 # re-armed, then generation resumes
    assert n.g.local.state == "running" and n.local_starts == 2
    assert n.notes("voice probe after the nightly eval OK") and n.notes("local generation resumes")
    assert len(n.notified) == 1


def test_a_rearm_waits_until_the_eval_time_probe_is_out_of_the_new_guards_look_back(tmp_path: Path) -> None:
    n = _Night(tmp_path)
    n.paused_for_the_eval()
    n.trip("03:01:58", 7.4)
    n.tick("03:02:10")
    n.eval = None
    n.tick("03:04:00")                     # a short eval
    n.probe("03:06:30", 48.0)
    n.tick("03:06:40")                     # OK, but 03:01:58 is still inside a fresh guard's 300 s look-back
    assert n.guards_started == 0 and n.g.local.state == "paused" and "re-arms at" in n.g.gen_pause()
    n.tick("03:07:10")
    assert n.guards_started == 1 and n.g.local.state == "running"


def test_a_trip_during_the_eval_pause_then_a_bad_probe_after_it_ends_the_run(tmp_path: Path) -> None:
    n = _Night(tmp_path)
    n.paused_for_the_eval()
    n.trip("03:01:58", 7.4)
    n.tick("03:02:20")
    assert n.g.rearm is not None and len(n.notified) == 1
    n.eval = None
    n.tick("03:20:00")
    n.probe("03:21:40", 9.0, "SLOW")
    n.tick("03:21:50")
    assert n.g.rearm is None and n.g.abort_file.exists()               # the switch stands again
    assert n.g.local.state == "stopped" and "abort switch tripped" in n.g.local.detail
    assert n.guards_started == 0 and n.local_starts == 1
    assert n.notes("the first voice probe after the nightly eval is bad too")
    n.probe("03:26:40", 50.0)
    n.tick("03:26:50")                                                  # never restarted this run
    assert n.g.local.state == "stopped" and n.local_starts == 1 and n.guards_started == 0


def test_a_trip_while_generation_runs_ends_the_run_as_before(tmp_path: Path) -> None:
    n = _Night(tmp_path)
    n.tick("02:50:00")
    child, guard = n.g.local.child, n.g.guard
    n.trip("02:52:00", 7.4)
    n.tick("02:52:20")
    assert n.g.local.state == "stopped" and "abort switch tripped" in n.g.local.detail
    assert not child.running and guard.running                         # the guard goes at shutdown, as before
    assert n.g.abort_file.exists() and n.g.rearm is None and not n.notified
    assert n.notes("not forgiven (generation was not paused for a nightly eval)")
    n.eval = "a nightly eval is running"
    n.tick("03:00:30")
    n.eval = None
    n.probe("03:21:40", 50.0)
    n.tick("03:21:50")                                                  # an eval later changes nothing
    assert n.g.local.state == "stopped" and n.local_starts == 1 and n.guards_started == 0


def test_a_bad_probe_before_the_pause_is_not_forgiven(tmp_path: Path) -> None:
    n = _Night(tmp_path)
    n.tick("02:50:00")
    n.probe("02:54:50", 7.4, "SLOW")       # landed while generation ran; the guard saw it only after the pause
    n.eval = "a nightly eval starts in 5 min"
    n.tick("02:55:00")
    assert n.g.local.state == "paused"
    n.g.abort_file.write_text("probe at 02:54:50: 7.4 tok/s < 15\n")
    n.tick("02:55:20")
    assert n.g.local.state == "stopped" and n.g.rearm is None and not n.notified
    assert n.notes("came before the pause")


def test_no_trip_the_eval_pause_and_resume_are_unchanged(tmp_path: Path) -> None:
    n = _Night(tmp_path)
    n.paused_for_the_eval()
    n.probe("03:01:58", 49.0)
    n.tick("03:10:00")
    n.eval = None
    n.tick("03:20:00")                                                  # resumes at once, no probe wait
    assert n.g.local.state == "running" and n.local_starts == 2
    assert n.guards_started == 0 and n.g.rearm is None and n.journal_reads == 0 and not n.notified
    assert n.notes("local generation paused: a nightly eval is running")
    assert n.notes("local generation resumes (the nightly eval is over)")
    assert not n.notes("probe guard")
