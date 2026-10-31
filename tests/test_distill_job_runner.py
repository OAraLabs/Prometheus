"""Tower job runner (scripts/distill/tower/job_runner.py) against fakes in a temp HOME.

A fake llama-server, score.py, feasibility_27b.py, nvidia-smi, systemctl and systemd-run
stand in for the real ones (tests/fixtures/distill_job_runner/). Nothing touches a GPU,
a systemd manager or the real eval data; every case file here is synthetic.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "distill_job_runner"
RUNNER_SH = REPO / "scripts" / "distill" / "tower" / "job_runner.sh"
RUNNER_PY = REPO / "scripts" / "distill" / "tower" / "job_runner.py"

pytestmark = pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="posix only")

SET_SIZES = {"sanity": 5, "eval-tools": 6, "eval-injection": 4, "eval-longctx": 8}


# ---------------------------------------------------------------------------
# the temp HOME
# ---------------------------------------------------------------------------


def _gguf(arch: str, ctx: int, blocks: int = 32) -> bytes:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    kvs = [
        ("general.architecture", 8, arch), ("general.name", 8, "fake"),
        (f"{arch}.context_length", 4, ctx), (f"{arch}.block_count", 4, blocks),
        (f"{arch}.full_attention_interval", 4, 4), (f"{arch}.attention.head_count_kv", 4, 4),
        (f"{arch}.attention.key_length", 4, 256), (f"{arch}.attention.value_length", 4, 256),
        ("tokenizer.ggml.tokens", 9, [f"t{i}" for i in range(100)]),
        ("tokenizer.ggml.scores", 9, [0.5] * 100),
    ]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for k, t, v in kvs:
        out += s(k) + struct.pack("<I", t)
        if t == 8:
            out += s(v)
        elif t == 4:
            out += struct.pack("<I", v)
        elif isinstance(v[0], str):
            out += struct.pack("<IQ", 8, len(v)) + b"".join(s(x) for x in v)
        else:
            out += struct.pack("<IQ", 6, len(v)) + b"".join(struct.pack("<f", x) for x in v)
    return out + b"\0" * 64


def _free_port() -> int:
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def _script(path: Path, src: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\n" + src.read_text())
    path.chmod(0o755)


def _cases(name: str, n: int) -> str:
    lines = []
    for i in range(n):
        c = {"id": f"{name}-{i}", "set": name, "messages": [{"role": "user", "content": "6*7?"}],
             "expect": {"answer": "42", "match": "exact"}}
        if name == "eval-tools":
            c["tools"] = [{"type": "function", "function": {
                "name": "read_file", "description": "read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                               "required": ["path"]}}}]
            c["expect"] = {"tool": "read_file", "args": {"path": "README.md"}}
        elif name == "eval-injection":
            c["injection"] = {"vector": "memory"}
            c["expect"] = {"forbidden_strings": ["rm -rf"]}
        elif name == "eval-longctx":
            c["depth_bucket"] = ["32k", "64k", "128k", "256k"][i * 4 // n]
            c["expect"] = {"answer": "42", "match": "contains"}
        lines.append(json.dumps(c))
    return "\n".join(lines) + "\n"


class Box:
    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.state = root / "state"
        self.d = self.home / "distill"
        self.port = _free_port()
        self.env = {
            "HOME": str(self.home), "PATH": f"{FIX / 'bin'}:/usr/bin:/bin", "FAKE_STATE": str(self.state),
            "LANG": "C.UTF-8", "DJR_RUNNER_PY": sys.executable, "DJR_ANYTIME": "1",
            "DJR_PORT": str(self.port), "DJR_SCORE_PY": sys.executable, "DJR_TRAIN_PY": sys.executable,
            "DJR_POLL_S": "0.1", "DJR_SLOW_CHECK_S": "0.2", "DJR_SCORE_GRACE_S": "3",
            "DJR_SERVER_GRACE_S": "3", "DJR_FEAS_GRACE_S": "3", "DJR_GPU_RELEASE_S": "5",
            "DJR_SERVER_LOAD_S": "10", "DJR_FEAS_CHILD_S": "1", "DJR_FEAS_MARGIN_S": "1",
            "DJR_FEAS_RESERVE_MIN": "0.1", "DJR_MIN_9B_S": "1", "DJR_MIN_BAKE_S": "1",
            "DJR_MIN_SET_S": "0.5", "DJR_MIN_RUN_S": "5", "DJR_DISK_FLOOR_GIB": "0",
            "DJR_DISK_MARGIN_GIB": "0", "FAKE_SERVER_CTX_LIMITS": "27B=131072", "FAKE_SCORE_DELAY": "0.02",
            "FAKE_FEAS_DELAY": "0.1", "DJR_SETTLE_S": "0.3",
        }
        self.procs: list[subprocess.Popen] = []

    def build(self) -> "Box":
        self.state.mkdir(parents=True)
        lc = self.home / "llama.cpp"
        _script(lc / "build" / "bin" / "llama-server", FIX / "fake_llama_server.py")
        (lc / "src").mkdir(parents=True)
        (lc / "src" / "llama-arch.cpp").write_text('{ LLM_ARCH_QWEN35, "qwen35" },\n')
        (lc / "common").mkdir(parents=True)
        (lc / "common" / "arg.cpp").write_text(" ".join(
            f'"{f}"' for f in ["-c", "-np", "-fa", "-ctk", "-ctv", "--jinja", "-ngl", "-fit", "-b", "-ub",
                               "--host", "--port"]) + " is_truthy(value)\n")
        (lc / ".git").mkdir()
        (lc / ".git" / "HEAD").write_text("9d57ce4560123456789abcdef\n")
        code = self.d / "code" / "scripts" / "distill"
        code.mkdir(parents=True)
        (self.d / "code" / "src").mkdir()
        (self.d / "code" / "COMMIT").write_text("0" * 40 + "\n")
        (code / "score.py").write_text((FIX / "fake_score.py").read_text())
        (code / "feasibility_27b.py").write_text((FIX / "fake_feasibility.py").read_text())
        ev = self.d / "evalsets"
        ev.mkdir(parents=True)
        import hashlib
        sums = []
        for name, n in SET_SIZES.items():
            text = _cases(name, n)
            (ev / f"{name}.jsonl").write_text(text)
            sums.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {name}.jsonl")
        (ev / "SHA256SUMS").write_text("\n".join(sums) + "\n")
        (self.d / "models").mkdir(parents=True)
        (self.d / "models" / "Qwen3.5-9B-Q4_K_M.gguf").write_bytes(_gguf("qwen35", 262144))
        (self.home / "models").mkdir()
        (self.home / "models" / "Qwen3.8-27B-UD-Q4_K_XL.gguf").write_bytes(_gguf("qwen35", 262144, 65))
        bnb = self.d / "models" / "qwen38-27b-bnb4"
        bnb.mkdir()
        (bnb / "config.json").write_text("{}")
        (bnb / "model-00001-of-00004.safetensors").write_bytes(b"\0" * 16)
        inp = self.d / "phase4" / "inputs"
        inp.mkdir(parents=True)
        (inp / "train-lengths.jsonl").write_text(
            "\n".join(json.dumps({"n_tokens": 9000 + i, "truncated": {"original_tokens": 20000}})
                      for i in range(20)) + "\n")
        # the training python: the dry run only asks whether the packages exist (find_spec)
        train_py = self.root / "train-python"
        train_py.write_text('#!/bin/bash\nif [ "$1" = "-I" ] && [ "$2" = "-c" ]; then echo; exit 0; fi\n'
                            f'exec "{sys.executable}" "$@"\n')
        train_py.chmod(0o755)
        self.env["DJR_TRAIN_PY"] = str(train_py)
        notify = self.d / "notify.sh"
        notify.write_text('#!/bin/bash\ncat >> "$FAKE_STATE/notify.log"\n')
        notify.chmod(0o755)
        return self

    # -- running -------------------------------------------------------------

    def args(self, execute: bool, stop_in: float | None, extra: list[str]) -> list[str]:
        a = [str(RUNNER_SH)]
        if execute:
            a.append("--execute")
        if stop_in is not None:
            a += ["--stop-at", str(int(time.time() + stop_in))]
        return a + extra

    def run(self, execute: bool = True, stop_in: float | None = 120, extra: list[str] | None = None,
            env: dict[str, str] | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
        return subprocess.run(self.args(execute, stop_in, extra or []), env={**self.env, **(env or {})},
                              capture_output=True, text=True, timeout=timeout)

    def start(self, stop_in: float = 120, env: dict[str, str] | None = None) -> subprocess.Popen:
        p = subprocess.Popen(self.args(True, stop_in, []), env={**self.env, **(env or {})},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.procs.append(p)
        return p

    # -- reading -------------------------------------------------------------

    def progress(self) -> dict:
        return json.loads((self.d / "logs" / "window" / "progress.json").read_text())

    def step(self, name: str) -> dict:
        return next(s for s in self.progress()["steps"] if s["name"] == name)

    def step_state(self, out: str) -> dict:
        return json.loads((self.d / "phase4" / "results" / out / "step.json").read_text())

    def rows(self, out: str, set_name: str) -> list[dict]:
        p = self.d / "phase4" / "results" / out / f"{set_name}.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

    def log(self, name: str) -> list[str]:
        p = self.state / name
        return p.read_text().splitlines() if p.exists() else []

    def live_servers(self) -> list[int]:
        out = []
        for f in self.state.glob("gpu.*"):
            pid = int(f.name.split(".", 1)[1])
            try:
                os.kill(pid, 0)
                out.append(pid)
            except ProcessLookupError:
                pass
        return out

    def wait_for(self, cond, timeout: float = 30, what: str = "") -> None:
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}")

    def kill_all(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        for pid in self.live_servers():
            os.kill(pid, signal.SIGKILL)
        run = self.d / "run"
        for f in run.glob("*.json") if run.exists() else []:
            try:
                rec = json.loads(f.read_text())
                os.killpg(int(rec["pgid"]), signal.SIGKILL)
            except (ValueError, KeyError, ProcessLookupError, PermissionError):
                pass


@pytest.fixture
def box(tmp_path: Path):
    b = Box(tmp_path).build()
    yield b
    b.kill_all()


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


def test_dry_run_prints_plan_checks_and_writes_nothing(box: Box) -> None:
    cp = box.run(execute=False, stop_in=None)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    out = cp.stdout
    assert "DRY RUN" in out and "all required inputs present" in out
    assert "-c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja -ngl 99" in out
    assert "contexts to try: 262144, 131072, 65536, 32768" in out  # 27B: native, then bucket sizes
    assert "--seqs 16384 --execute" in out and "--seqs 4096 --execute" in out
    assert "glimmer-30b-baseline" in out and "skip: model not downloaded" in out
    assert "[MISSING]" not in out
    assert not (box.d / "logs").exists() and not (box.d / "phase4" / "results").exists()
    assert not (box.d / "run").exists() and not box.log("server-starts.log")


def test_dry_run_flags_a_missing_model_and_a_bad_checksum(box: Box) -> None:
    (box.d / "models" / "Qwen3.5-9B-Q4_K_M.gguf").unlink()
    with open(box.d / "evalsets" / "sanity.jsonl", "a") as f:
        f.write(json.dumps({"id": "extra", "set": "sanity"}) + "\n")
    cp = box.run(execute=False, stop_in=None)
    assert cp.returncode == 1
    assert "[MISSING] base-9B GGUF" in cp.stdout
    assert "[MISSING] eval set sanity" in cp.stdout


# ---------------------------------------------------------------------------
# a whole window
# ---------------------------------------------------------------------------


def test_full_queue_runs_in_order_and_releases_everything(box: Box) -> None:
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    prog = box.progress()
    assert prog["state"] == "finished" and prog["stop_reason"] is None
    st = {s["name"]: s["status"] for s in prog["steps"]}
    assert st == {"base-9b-baseline": "done", "qwen38-27b-feasibility": "done",
                  "qwen38-27b-baseline": "done", "glimmer-30b-baseline": "skipped"}
    # sets ran in the runbook order, one score.py run each
    runs = [x.split()[1:3] for x in box.log("score-calls.log") if x.startswith("run ")]
    assert runs[:4] == [["base-9b", "sanity"], ["base-9b", "eval-tools"], ["base-9b", "eval-injection"],
                        ["base-9b", "eval-longctx"]]
    # 9B at 262144 with the runbook flags; the 27B stepped down from its native 262144 to 131072
    starts = box.log("server-starts.log")
    assert "-c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 --jinja -ngl 99 -fit off -b 2048 -ub 2048" in starts[0]
    assert "--host 127.0.0.1" in starts[0]
    assert [int(x.split()[1]) for x in starts[1:]] == [262144, 131072]
    assert box.step("qwen38-27b-baseline")["ctx"] == 131072
    assert box.step_state("qwen38-27b/baseline")["ctx_chosen"] == 131072
    # the feasibility probe ran each length once, then merged
    assert box.log("feas-calls.log") == ["execute 16384", "execute 12288", "execute 8192", "execute 4096",
                                         "measured 16384,12288,8192,4096"]
    feas = box.step("qwen38-27b-feasibility")
    assert feas["decision"]["verdict"] == "local-viable"
    assert feas["lengths"]["16384"]["fits"] is False and feas["lengths"]["12288"]["fits"] is True
    # every result is in its own dir, with a combined summary
    res = box.d / "phase4" / "results"
    assert (res / "base-9b" / "baseline" / "base-9b.summary.json").is_file()
    assert (res / "qwen38-27b" / "feasibility" / "feasibility-27b.json").is_file()
    assert len(box.rows("base-9b/baseline", "eval-longctx")) == SET_SIZES["eval-longctx"]
    # nothing left running, no pid files, GPU children were scoped as distill-*
    assert box.live_servers() == []
    assert not list((box.d / "run").glob("*.json"))
    assert not (box.d / "run" / "job_runner.pid").exists()
    assert any(u.startswith("distill-llama-base-9b-c262144-") for u in box.log("scopes.log"))
    assert prog["server"]["state"] == "stopped"
    summary = subprocess.run([str(RUNNER_SH), "--summary"], env=box.env, capture_output=True, text=True)
    lines = summary.stdout.strip().splitlines()
    assert 1 <= len(lines) <= 5 and lines[0].startswith("4090 runner finished")
    assert "9B base done c256k: san 5/5" in summary.stdout and "27B feas done" in summary.stdout


def test_a_second_window_skips_finished_steps(box: Box) -> None:
    assert box.run(stop_in=120).returncode == 0
    n_runs = len(box.log("score-calls.log"))
    cp = box.run(stop_in=120)
    assert cp.returncode == 0
    assert len(box.log("score-calls.log")) == n_runs
    assert all(s["detail"] == "done in an earlier window" for s in box.progress()["steps"]
               if s["status"] == "done")


# ---------------------------------------------------------------------------
# stops, kills and resumes
# ---------------------------------------------------------------------------


def _wait_scoring(box: Box, set_name: str, n: int = 1) -> None:
    box.wait_for(lambda: len(box.rows("base-9b/baseline", set_name)) >= n, what=f"{set_name} rows")


def test_sigterm_mid_set_stops_children_then_the_next_window_resumes(box: Box) -> None:
    p = box.start(env={"FAKE_SCORE_DELAY": "0.3"})
    _wait_scoring(box, "eval-tools", 2)
    server_pids = box.live_servers()
    assert len(server_pids) == 1
    t0 = time.time()
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=30)
    assert rc == 143, p.stdout.read()[-2000:]
    assert time.time() - t0 < 15
    assert box.live_servers() == []
    prog = box.progress()
    assert prog["state"] == "stopped" and prog["stop_reason"] == "sigterm"
    st = box.step_state("base-9b/baseline")
    assert st["status"] == "partial"
    assert st["sets"]["sanity"]["status"] == "done"
    assert st["sets"]["eval-tools"]["status"] == "partial"
    done_before = len(box.rows("base-9b/baseline", "eval-tools"))
    assert 2 <= done_before < SET_SIZES["eval-tools"]
    assert [s["status"] for s in prog["steps"][1:]] == ["not-started"] * 3

    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    runs = [x.split()[1:4] for x in box.log("score-calls.log") if x.startswith("run base-9b")]
    assert [r[1] for r in runs].count("sanity") == 1                    # not re-run
    resumed = [r for r in runs if r[1] == "eval-tools"][-1]
    assert int(resumed[2]) == SET_SIZES["eval-tools"] - done_before     # only what was left
    ids = [r["id"] for r in box.rows("base-9b/baseline", "eval-tools") if r["status"] == "ok"]
    assert sorted(set(ids)) == sorted(ids) and len(ids) == SET_SIZES["eval-tools"]
    assert box.step_state("base-9b/baseline")["status"] == "done"


def test_a_killed_runner_leaves_an_orphan_that_the_next_run_stops(box: Box) -> None:
    p = box.start(env={"FAKE_SCORE_DELAY": "0.3"})
    _wait_scoring(box, "sanity", 2)
    orphan = box.live_servers()
    rec = json.loads((box.d / "run" / "server.json").read_text())
    child = json.loads((box.d / "run" / "child.json").read_text())
    p.kill()  # no cleanup at all
    p.wait()
    time.sleep(0.3)
    assert box.live_servers() == orphan  # still holding the "GPU"
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    for pid in (rec["pid"], child["pid"]):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    notes = " ".join(box.progress()["notes"])
    assert "stopping a server left by run" in notes and "stopping a child left by run" in notes
    assert box.step_state("base-9b/baseline")["status"] == "done"
    assert box.live_servers() == []


def test_no_swaps_flag_mid_run_stops_and_blocks_the_next_start(box: Box) -> None:
    p = box.start(env={"FAKE_SCORE_DELAY": "0.3"})
    _wait_scoring(box, "sanity", 1)
    (box.d / "NO-SWAPS").write_text("rehearsal failed\n")
    assert p.wait(timeout=30) == 3
    assert box.live_servers() == []
    assert box.progress()["stop_reason"] == "no-swaps"
    cp = box.run(stop_in=120)
    assert cp.returncode == 2
    assert box.progress()["state"] == "refused" and "NO-SWAPS" in box.progress()["stop_reason"]
    assert len(box.log("server-starts.log")) == 1


def test_production_unit_coming_up_mid_run_yields_the_gpu(box: Box) -> None:
    p = box.start(env={"FAKE_SCORE_DELAY": "0.3"})
    _wait_scoring(box, "sanity", 1)
    (box.state / "prod-active").write_text("")
    assert p.wait(timeout=30) == 3
    assert box.live_servers() == []
    assert box.progress()["stop_reason"] == "production-unit-active"


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_refuses_while_the_production_unit_is_active(box: Box) -> None:
    (box.state / "prod-active").write_text("")
    cp = box.run(stop_in=120)
    assert cp.returncode == 2
    assert "llama-server.service is active" in box.progress()["stop_reason"]
    assert box.log("server-starts.log") == []
    assert "refused" in " ".join(box.log("notify.log"))


def test_refuses_outside_the_window_and_while_the_gpu_is_busy(box: Box) -> None:
    now = time.localtime()
    later = (now.tm_hour + 2) % 24
    env = {"DJR_ANYTIME": "0", "DJR_WINDOW_START": f"{later:02d}:00",
           "DJR_WINDOW_END": f"{(later + 1) % 24:02d}:00" if later != 23 else "23:59"}
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 2 and "outside the window" in box.progress()["stop_reason"]
    cp = box.run(stop_in=120, env={"FAKE_GPU_BASE_MIB": "9000", "DJR_GPU_RELEASE_S": "1"})
    assert cp.returncode == 2 and "GPU holds 9000 MiB" in box.progress()["stop_reason"]
    assert box.log("server-starts.log") == []


def test_one_runner_at_a_time(box: Box) -> None:
    p = box.start(env={"FAKE_SCORE_DELAY": "0.3"})
    _wait_scoring(box, "sanity", 1)
    cp = box.run(stop_in=120)
    assert cp.returncode == 2 and "lock" in cp.stderr
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143


# ---------------------------------------------------------------------------
# time budget and failures
# ---------------------------------------------------------------------------


def test_deadline_keeps_time_for_the_feasibility_probe(box: Box) -> None:
    # the 9B would need ~23 cases x 0.5 s; it must stop ~8 s before stop-at for the probe
    env = {"FAKE_SCORE_DELAY": "0.5", "DJR_FEAS_RESERVE_MIN": "0.14", "DJR_MIN_BAKE_S": "60"}
    cp = box.run(stop_in=16, env=env)
    assert cp.returncode == 0, cp.stdout[-3000:]
    st = {s["name"]: s["status"] for s in box.progress()["steps"]}
    assert st["base-9b-baseline"] == "partial"
    assert st["qwen38-27b-feasibility"] == "done"
    assert st["qwen38-27b-baseline"] == "not-started"
    assert box.progress()["stop_reason"] in (None, "stop-at")
    assert box.live_servers() == []


def test_server_death_fails_the_step_and_the_queue_goes_on(box: Box) -> None:
    # the runner polls /health once a second: the server lives long enough for a few rows
    cp = box.run(stop_in=120, env={"FAKE_SERVER_DIE_AFTER_S": "2.0", "FAKE_SCORE_DELAY": "0.3"})
    assert cp.returncode == 1, cp.stdout[-3000:]
    st = {s["name"]: s["status"] for s in box.progress()["steps"]}
    assert st["base-9b-baseline"] == "failed" and st["qwen38-27b-feasibility"] == "done"
    assert st["qwen38-27b-baseline"] == "failed"
    assert box.step_state("base-9b/baseline")["failures"] == 1
    assert box.step_state("qwen38-27b/baseline")["ctx_bad"] == [131072]
    assert "FAILED" in " ".join(box.log("notify.log"))
    first = box.step_state("qwen38-27b/baseline")
    assert first["ctx_scored"] == 131072 and box.rows("qwen38-27b/baseline", "sanity")
    # next window: the 9B is retried at its fixed context; the 27B steps below the bad one
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert box.step_state("base-9b/baseline")["status"] == "done"
    assert box.step("qwen38-27b-baseline")["ctx"] == 65536
    # like for like: what was scored at 131072 moved aside and every set ran again at 65536
    st = box.step_state("qwen38-27b/baseline")
    assert st["status"] == "done" and st["ctx_scored"] == 65536
    assert [(h["from"], h["to"]) for h in st["ctx_history"]] == [(131072, 65536)]
    archived = Path(st["ctx_history"][0]["archived"])
    assert archived.parent == box.d / "phase4" / "results" / "qwen38-27b" / "baseline"
    assert (archived / "sanity.jsonl").is_file() and not (archived / "step.json").exists()
    assert {v["ctx"] for v in st["sets"].values()} == {65536}
    for name, n in SET_SIZES.items():
        assert len(box.rows("qwen38-27b/baseline", name)) == n
    assert st["summary_info"]["set_ctx"] == [65536]
    assert "NOTE" not in box.step("qwen38-27b-baseline")["detail"]


def test_a_refused_set_fails_the_step(box: Box) -> None:
    cp = box.run(stop_in=120, env={"FAKE_SCORE_EXIT_ON_SET": "eval-tools:2"})
    assert cp.returncode == 1
    b = box.step("base-9b-baseline")
    assert b["status"] == "failed" and "refused eval-tools" in b["detail"]
    assert box.live_servers() == []


def test_feasibility_never_reprobes_a_measured_length(box: Box) -> None:
    assert box.run(stop_in=120).returncode == 0
    (box.d / "phase4" / "results" / "qwen38-27b" / "feasibility" / "step.json").unlink()
    (box.state / "feas-calls.log").unlink()
    assert box.run(stop_in=120).returncode == 0
    assert box.log("feas-calls.log") == ["measured 16384,12288,8192,4096"]


def test_unscoped_mode_still_runs(box: Box) -> None:
    cp = box.run(stop_in=120, env={"DJR_USE_SCOPES": "0"})
    assert cp.returncode == 0
    assert box.log("scopes.log") == []
    assert any("no systemd scopes" in n for n in box.progress()["notes"])


# ---------------------------------------------------------------------------
# the real scorer (HEAD's score.py) behind the runner, against the fake server
# ---------------------------------------------------------------------------


def test_real_scorer_runs_under_the_runner(box: Box, tmp_path: Path) -> None:
    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("project deps not importable")
    code = box.d / "code"
    arch = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "src", "scripts/distill"],
                          capture_output=True)
    if arch.returncode != 0:
        pytest.skip("git archive unavailable")
    import shutil
    shutil.rmtree(code)
    code.mkdir()
    subprocess.run(["tar", "-x", "-C", str(code)], input=arch.stdout, check=True)
    env = {"DJR_LENGTHS": str(tmp_path / "absent.jsonl"), "PYTHONDONTWRITEBYTECODE": "1"}
    cp = box.run(stop_in=240, env=env, timeout=240)
    assert cp.returncode == 0, cp.stdout[-3000:]
    st = {s["name"]: s["status"] for s in box.progress()["steps"]}
    assert st["base-9b-baseline"] == "done" and st["qwen38-27b-feasibility"] == "skipped"
    rows = box.rows("base-9b/baseline", "sanity")
    assert len(rows) == SET_SIZES["sanity"] and all(r["status"] == "ok" for r in rows)
    assert rows[0]["label"] == "base-9b" and rows[0]["server"]["n_ctx"] == 262144
    summ = json.loads((box.d / "phase4" / "results" / "base-9b" / "baseline" /
                       "base-9b.summary.json").read_text())
    assert summ["sets"]["sanity"]["metrics"]["sanity_score"]["rate"] == 1.0
    assert set(summ["sets"]) == set(SET_SIZES)


def test_glimmer_only_when_downloaded_and_supported(box: Box) -> None:
    g = box.d / "models" / "glimmer-30b"
    g.mkdir()
    (g / "Muse-Glimmer-30B-Q4_K_M.gguf").write_bytes(_gguf("museglimmer", 131072))
    (g / ".download-status.json").write_text(json.dumps({"state": "downloading"}))
    assert box.run(stop_in=120).returncode == 0
    assert box.step("glimmer-30b-baseline")["status"] == "skipped"
    assert "download 'downloading'" in box.step("glimmer-30b-baseline")["detail"]
    (g / ".download-status.json").write_text(json.dumps({"state": "done"}))
    assert box.run(stop_in=120).returncode == 0
    gl = box.step("glimmer-30b-baseline")
    assert gl["status"] == "skipped" and "no architecture 'museglimmer'" in gl["detail"]
    (g / "Muse-Glimmer-30B-Q4_K_M.gguf").write_bytes(_gguf("qwen35", 131072))
    assert box.run(stop_in=120).returncode == 0
    gl = box.step("glimmer-30b-baseline")
    assert gl["status"] == "done" and gl["ctx"] == 131072
    assert any("Muse-Glimmer" in x and " 131072 " in x for x in box.log("server-starts.log"))


def test_a_scope_failure_is_not_mistaken_for_a_context_that_does_not_fit(box: Box) -> None:
    cp = box.run(stop_in=120, env={"FAKE_SCOPE_FAIL_PREFIX": "distill-llama-"})
    assert cp.returncode == 1
    for name in ("base-9b-baseline", "qwen38-27b-baseline"):
        s = box.step(name)
        assert s["status"] == "failed" and "systemd-run failed" in s["detail"], s
    assert box.step("qwen38-27b-baseline")["ctx_tried"] == ["262144 error"]  # no step-down
    assert box.log("server-starts.log") == []
    assert box.step("qwen38-27b-feasibility")["status"] == "done"


# ---------------------------------------------------------------------------
# review fixes (2026-10-30): flag default, gates, external stops, summarize stop,
# dry-run requirements, no core dumps
# ---------------------------------------------------------------------------

ONLY_27B = {"DJR_MODEL_9B": "/nonexistent/9b.gguf", "DJR_LENGTHS": "/nonexistent/lengths.jsonl"}


def test_the_window_flag_is_in_the_default_no_swaps_list(box: Box) -> None:
    p = box.start(env={"FAKE_SCORE_DELAY": "0.3"})
    _wait_scoring(box, "sanity", 1)
    (box.d / "window").mkdir()
    (box.d / "window" / "NO-SWAPS-TONIGHT").write_text("rehearsal: restore failed\n")
    assert p.wait(timeout=30) == 3
    assert box.progress()["stop_reason"] == "no-swaps" and box.live_servers() == []
    cp = box.run(stop_in=120)
    assert cp.returncode == 2 and "NO-SWAPS-TONIGHT" in box.progress()["stop_reason"]
    assert len(box.log("server-starts.log")) == 1


def test_gates_refuse_without_a_rehearsal_pass_or_an_armed_dead_man(box: Box) -> None:
    now = int(time.time())
    env = {"DJR_GATES": "1", "DJR_DEADMAN_LATEST": str(now + 3600)}
    last = box.d / "window" / "state" / "rehearsal.last"
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 2 and "no rehearsal PASS" in box.progress()["stop_reason"]
    last.parent.mkdir(parents=True)
    last.write_text("FAIL 2026-10-31 01:04:00 EDT bundle=abc restore failed\n")
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 2 and "FAIL 2026-10-31" in box.progress()["stop_reason"]
    last.write_text("PASS 2026-10-31 01:04:00 EDT bundle=abc down 70s\n")
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 2 and "dead-man restore not armed" in box.progress()["stop_reason"]
    (box.state / "deadman-next").write_text(str(now + 7200))  # due after DJR_DEADMAN_LATEST
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 2 and "not before" in box.progress()["stop_reason"]
    assert box.log("server-starts.log") == []
    (box.state / "deadman-next").write_text(str(now + 1800))
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 0, cp.stdout[-3000:]
    # the dry run shows the gates but does not require them (the window start arms the dead-man later)
    (box.state / "deadman-next").unlink()
    cp = box.run(execute=False, stop_in=None, env=env)
    assert cp.returncode == 0 and "dead-man timer armed (execute gate)" in cp.stdout


def test_a_server_stopped_from_outside_is_a_stop_not_a_crash(box: Box) -> None:
    p = box.start(env={**ONLY_27B, "FAKE_SCORE_DELAY": "0.3"})
    box.wait_for(lambda: len(box.rows("qwen38-27b/baseline", "sanity")) >= 2, what="27B rows")
    (pid,) = box.live_servers()
    os.kill(pid, signal.SIGTERM)  # a restore stopping the runner's distill-llama-* scope
    assert p.wait(timeout=30) == 3, p.stdout.read()[-2000:]
    prog = box.progress()
    assert prog["stop_reason"] == "external-stop"
    st = box.step_state("qwen38-27b/baseline")
    assert st["status"] == "partial" and st["failures"] == 0
    assert "ctx_bad" not in st and st["ctx_chosen"] == 131072
    assert len(box.log("server-starts.log")) == 2  # 262144 (does not fit) + 131072; no respawn
    assert "FAILED" not in " ".join(box.log("notify.log"))
    assert box.live_servers() == []
    # next window resumes at the same context
    cp = box.run(stop_in=120, env=ONLY_27B)
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert box.step_state("qwen38-27b/baseline")["ctx_scored"] == 131072
    assert "ctx_history" not in box.step_state("qwen38-27b/baseline")


def test_a_restore_that_stops_the_scope_then_the_runner_counts_no_failure(box: Box) -> None:
    p = box.start(env={**ONLY_27B, "FAKE_SCORE_DELAY": "0.3", "DJR_SETTLE_S": "3"})
    box.wait_for(lambda: len(box.rows("qwen38-27b/baseline", "sanity")) >= 2, what="27B rows")
    (pid,) = box.live_servers()
    os.kill(pid, signal.SIGTERM)
    time.sleep(0.5)  # setups.py stops distill-llama-* first, then the runner's unit
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143
    st = box.step_state("qwen38-27b/baseline")
    assert box.progress()["stop_reason"] == "sigterm"
    assert st["status"] == "partial" and st["failures"] == 0 and "ctx_bad" not in st
    assert len(box.log("server-starts.log")) == 2


def test_a_probe_stopped_from_outside_starts_no_further_length(box: Box) -> None:
    env = {"DJR_MODEL_9B": "/nonexistent/9b.gguf", "FAKE_FEAS_STOPPED_AT": "12288"}
    cp = box.run(stop_in=120, env=env)
    assert cp.returncode == 3, cp.stdout[-3000:]
    assert box.progress()["stop_reason"] == "external-stop"
    calls = box.log("feas-calls.log")
    assert calls[:2] == ["execute 16384", "execute 12288"]
    assert not any(c.startswith("execute 8192") or c.startswith("execute 4096") for c in calls)
    st = box.step_state("qwen38-27b/feasibility")
    assert st["status"] == "partial" and st["failures"] == 0
    assert st["lengths"]["12288"]["attempts"] == 0 and st["lengths"]["16384"]["measured"] is True
    assert box.step("qwen38-27b-baseline")["status"] == "not-started"
    assert box.log("server-starts.log") == []
    cp = box.run(stop_in=120, env={"DJR_MODEL_9B": "/nonexistent/9b.gguf"})
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert [c for c in box.log("feas-calls.log") if c.startswith("execute")][2:] == [
        "execute 12288", "execute 8192", "execute 4096"]


def test_a_stop_during_the_combined_summarize_is_partial_not_failed(box: Box) -> None:
    p = box.start(env={"FAKE_SUMMARIZE_DELAY": "20"})
    box.wait_for(lambda: (box.state / "summarize-started").exists(), timeout=60, what="summarize")
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143
    st = box.step_state("base-9b/baseline")
    assert st["status"] == "partial" and st["failures"] == 0
    assert all(v["status"] == "done" for v in st["sets"].values())
    assert "summarize stopped" in box.step("base-9b-baseline")["detail"]
    assert "FAILED" not in " ".join(box.log("notify.log"))
    n_runs = len([x for x in box.log("score-calls.log") if x.startswith("run base-9b")])
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert box.step_state("base-9b/baseline")["status"] == "done"
    assert len([x for x in box.log("score-calls.log") if x.startswith("run base-9b")]) == n_runs


def test_dry_run_needs_a_step_inputs_only_while_the_step_has_work(box: Box) -> None:
    import shutil
    shutil.rmtree(box.d / "models" / "qwen38-27b-bnb4")
    cp = box.run(execute=False, stop_in=None)
    assert cp.returncode == 1 and "[MISSING] 27B bnb-4bit model dir" in cp.stdout
    (box.d / "models" / "qwen38-27b-bnb4").mkdir()
    (box.d / "models" / "qwen38-27b-bnb4" / "config.json").write_text("{}")
    (box.d / "models" / "qwen38-27b-bnb4" / "model.safetensors").write_bytes(b"\0")
    assert box.run(stop_in=120).returncode == 0
    # both steps done: their inputs may go (the disk is tight) without blocking later windows
    shutil.rmtree(box.d / "models" / "qwen38-27b-bnb4")
    (box.d / "phase4" / "inputs" / "train-lengths.jsonl").unlink()
    (box.d / "models" / "Qwen3.5-9B-Q4_K_M.gguf").unlink()
    cp = box.run(execute=False, stop_in=None)
    assert cp.returncode == 0, cp.stdout
    assert "[MISSING]" not in cp.stdout
    assert "not needed: qwen38-27b-feasibility is done" in cp.stdout
    assert "not needed: base-9b-baseline is done" in cp.stdout


def test_every_child_runs_with_core_dumps_off(box: Box) -> None:
    assert box.run(stop_in=120).returncode == 0
    lines = box.log("limits.log")
    assert lines and set(lines) == {"core 0 0"}


# ---------------------------------------------------------------------------
# step 0: the teacher quant A/B (DJR_AB=1)
# ---------------------------------------------------------------------------

IQ3_SHA = "f" * 64
Q4_SHA = "e" * 64


def _ab_inputs(box: Box, **env: str) -> None:
    """The A/B's inputs in the temp HOME: a fake driver in code-ab, the taskset, both GGUFs."""
    dist = box.d / "code-ab" / "scripts" / "distill"
    dist.mkdir(parents=True)
    (dist / "teacher_ab_run.sh").write_text(
        f'#!/bin/bash\nexec "{sys.executable}" "{FIX / "fake_teacher_ab_run.py"}" "$@"\n')
    (dist / "teacher_ab.py").write_text("# fake\n")
    (box.d / "code-ab" / "COMMIT").write_text("a" * 40 + "\n")
    (box.d / "phase2b" / "tasksets").mkdir(parents=True)
    (box.d / "phase2b" / "tasksets" / "real-v1-excl.jsonl").write_text('{"id": "ts-1"}\n')
    (box.d / "models" / "teacher").mkdir(parents=True)
    (box.d / "models" / "teacher" / "Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf").write_bytes(_gguf("qwen35", 262144, 65))
    (box.home / "models" / "Qwen3.8-27B-regular-Q4_K_M-bartowski.gguf").write_bytes(_gguf("qwen35", 262144, 64))
    box.env.update({"DJR_AB": "1", "DJR_AB_IQ3_SHA256": IQ3_SHA, "DJR_AB_Q4_SHA256": Q4_SHA,
                    "DJR_AB_MIN_MIN": "0.01", **env})


def _ab_calls(box: Box) -> list[list[str]]:
    return [json.loads(x) for x in box.log("ab-calls.log")]


def _opt(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_ab_off_by_default_leaves_the_queue_as_it_was(box: Box) -> None:
    _ab_inputs(box)
    box.env.pop("DJR_AB")
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert [s["name"] for s in box.progress()["steps"]][0] == "base-9b-baseline"
    assert box.log("ab-calls.log") == []


def test_ab_runs_first_time_boxed_then_the_queue_goes_on(box: Box) -> None:
    _ab_inputs(box, DJR_AB_MAX_MIN="1")
    t0 = time.time()
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    prog = box.progress()
    assert [s["name"] for s in prog["steps"]] == ["teacher-ab", "base-9b-baseline", "qwen38-27b-feasibility",
                                                  "qwen38-27b-baseline", "glimmer-30b-baseline"]
    ab = box.step("teacher-ab")
    assert ab["status"] == "done" and ab["decision"]["verdict"] == "keep IQ3"
    assert ab["detail"] == "Teacher A/B decision: keep IQ3"
    (call,) = _ab_calls(box)
    assert call[0] == "--execute"
    until = datetime.fromisoformat(_opt(call, "--until")).timestamp()
    assert t0 < until <= t0 + 60 + 2                     # the 1-min time box, not stop-at
    assert _opt(call, "--tasks") == str(box.d / "phase2b" / "tasksets" / "real-v1-excl.jsonl")
    assert _opt(call, "--iq3").endswith("/distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf")
    assert _opt(call, "--q4").endswith("/models/Qwen3.8-27B-regular-Q4_K_M-bartowski.gguf")
    assert _opt(call, "--iq3-sha256") == IQ3_SHA and _opt(call, "--q4-sha256") == Q4_SHA
    assert _opt(call, "--port") == str(box.port) and _opt(call, "--code") == str(box.d / "code-ab")
    assert _opt(call, "--py") == sys.executable                          # DJR_SCORE_PY by default
    assert _opt(call, "--notify-cmd") == str(box.d / "notify.sh")
    assert any(u.startswith("distill-teacher-ab-") for u in box.log("scopes.log"))
    # the rest of the queue ran after it, on the runner's own server
    assert box.step("base-9b-baseline")["status"] == "done"
    assert box.step_state("teacher-ab")["decision"]["verdict"] == "keep IQ3"
    summary = subprocess.run([str(RUNNER_SH), "--summary"], env=box.env, capture_output=True, text=True)
    assert "A/B done: keep IQ3" in summary.stdout
    # a later window does not run it again
    assert box.run(stop_in=120).returncode == 0
    assert len(_ab_calls(box)) == 1
    assert box.step("teacher-ab")["detail"] == "done in an earlier window"


def test_ab_cut_by_its_time_box_is_partial_and_resumes(box: Box) -> None:
    _ab_inputs(box, DJR_AB_MAX_MIN="0.05", FAKE_AB_UNTIL_EXIT="1")    # 3 s
    t0 = time.time()
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    ab = box.step("teacher-ab")
    assert ab["status"] == "partial" and "resumes next window" in ab["detail"]
    until = datetime.fromisoformat(_opt(_ab_calls(box)[0], "--until")).timestamp()
    assert until <= t0 + 3 + 1
    assert box.step("base-9b-baseline")["status"] == "done"
    box.env["FAKE_AB_UNTIL_EXIT"] = "0"
    assert box.run(stop_in=120).returncode == 0
    assert len(_ab_calls(box)) == 2 and box.step("teacher-ab")["status"] == "done"


def test_ab_inputs_missing_skip_the_step_not_the_window(box: Box) -> None:
    _ab_inputs(box)
    (box.d / "phase2b" / "tasksets" / "real-v1-excl.jsonl").unlink()
    dry = box.run(execute=False, stop_in=None)
    assert dry.returncode == 0, dry.stdout[-3000:]
    assert "[     --] teacher A/B inputs" in dry.stdout and "real-v1-excl.jsonl" in dry.stdout
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert box.step("teacher-ab")["status"] == "skipped" and "missing" in box.step("teacher-ab")["detail"]
    assert box.log("ab-calls.log") == [] and box.step("base-9b-baseline")["status"] == "done"


def test_ab_driver_failure_fails_the_step_and_notifies(box: Box) -> None:
    _ab_inputs(box, FAKE_AB_RC="3")
    cp = box.run(stop_in=120)
    assert cp.returncode == 1
    ab = box.step("teacher-ab")
    assert ab["status"] == "failed" and "exit 3" in ab["detail"] and "would not start" in ab["detail"]
    assert box.step_state("teacher-ab")["failures"] == 1
    assert "teacher-ab FAILED" in (box.state / "notify.log").read_text()
    assert box.step("base-9b-baseline")["status"] == "done"


def test_ab_invalid_report_is_a_failure(box: Box) -> None:
    _ab_inputs(box, FAKE_AB_RC="1", FAKE_AB_VERDICT="invalid")
    assert box.run(stop_in=120).returncode == 1
    assert box.step("teacher-ab")["status"] == "failed"
    assert box.step("teacher-ab")["decision"]["verdict"] == "invalid"


def test_sigterm_during_the_ab_stops_the_driver(box: Box) -> None:
    _ab_inputs(box, FAKE_AB_DELAY="60")
    p = box.start(stop_in=120)
    box.wait_for(lambda: (box.state / "ab-pid").exists(), what="the A/B driver")
    pid = int((box.state / "ab-pid").read_text())
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    st = box.step_state("teacher-ab")
    assert st["status"] == "partial" and st["failures"] == 0
    assert [s["status"] for s in box.progress()["steps"][1:]] == ["not-started"] * 4


def test_dry_run_shows_the_ab_driver_and_its_own_dry_run(box: Box) -> None:
    _ab_inputs(box)
    cp = box.run(execute=False, stop_in=None)
    assert cp.returncode == 0, cp.stdout[-3000:]
    out = cp.stdout
    assert " 1. teacher-ab" in out and " 2. base-9b-baseline" in out
    assert "--until <time box end>" in out and f"--iq3-sha256 {IQ3_SHA}" in out
    assert "driver dry run (exit 0):" in out and "DRY RUN (nothing is started" in out
    assert "[     ok] teacher A/B inputs" in out
    (call,) = _ab_calls(box)
    assert "--execute" not in call and "--notify-cmd" not in call
    assert not (box.d / "logs").exists() and not (box.d / "teacher-ab").exists()


# ---------------------------------------------------------------------------
# the QLoRA smoke step (train-smoke)
# ---------------------------------------------------------------------------

BASE_REV = "3c78dfb3e32c496a2438fb1790859eff805284cf"


def _smoke_inputs(box: Box, **env: str) -> None:
    """The smoke's inputs in the temp HOME: a fake trainer in code-train/head, data, tokenizer, base."""
    dist = box.d / "code-train" / "head" / "scripts" / "distill"
    dist.mkdir(parents=True)
    (dist / "train_qlora.py").write_text((FIX / "fake_train_qlora.py").read_text())
    (dist / "render.py").write_text("# fake\n")
    (dist / "artifact_policy.py").write_text("# fake\n")
    (box.d / "code-train" / "head" / "COMMIT").write_text("b" * 40 + "\n")
    data = box.d / "phase3" / "build-v3-max-derived"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text('{"id": "r1"}\n{"id": "r2"}\n')
    (data / "val.jsonl").write_text('{"id": "v1"}\n')
    (box.d / "models" / "qwen3.5-9b-tokenizer").mkdir(parents=True)
    (box.d / "models" / "qwen3.5-9b-tokenizer" / "tokenizer.json").write_text("{}")
    base = box.d / "models" / "qwen3.5-9b-bnb4"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"\0" * 16)
    (base / "MANIFEST.json").write_text(json.dumps({"repo": "x/y", "revision": BASE_REV}))
    box.env.update({"DJR_SMOKE": "1", "DJR_SMOKE_MAX_DERIVED_OK": "1", "DJR_SMOKE_MIN_MIN": "0.01",
                    "DJR_SMOKE_GRACE_S": "5", **env})


def _smoke_calls(box: Box) -> list[dict]:
    return [json.loads(x) for x in box.log("smoke-calls.log")]


def test_smoke_off_by_default_leaves_the_queue_as_it_was(box: Box) -> None:
    _smoke_inputs(box)
    box.env.pop("DJR_SMOKE")
    assert box.run(stop_in=120).returncode == 0
    assert [s["name"] for s in box.progress()["steps"]][0] == "base-9b-baseline"
    assert box.log("smoke-calls.log") == []


def test_smoke_runs_first_time_boxed_with_the_approval_then_the_queue_goes_on(box: Box) -> None:
    _smoke_inputs(box, DJR_SMOKE_MAX_MIN="1")
    t0 = time.time()
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    assert [s["name"] for s in box.progress()["steps"]] == [
        "train-smoke", "base-9b-baseline", "qwen38-27b-feasibility", "qwen38-27b-baseline", "glimmer-30b-baseline"]
    sm = box.step("train-smoke")
    assert sm["status"] == "done" and sm["detail"].startswith("smoke-train-pass: 50 steps, loss 1.2500 -> 0.7500")
    assert sm["smoke"]["max_derived"] is True
    (call,) = _smoke_calls(box)
    argv = call["argv"]
    assert call["HF_HUB_OFFLINE"] == "1" and "--smoke" in argv and "--resume" not in argv
    assert "--i-have-operator-approval-for-max-derived-data" in argv
    assert _opt(argv, "--base-revision") == BASE_REV                    # from the base's MANIFEST.json
    assert _opt(argv, "--out") == str(box.d / "phase5" / "runs" / "9b-smoke-max-derived")
    assert _opt(argv, "--data").endswith("/phase3/build-v3-max-derived/train.jsonl")
    assert _opt(argv, "--val").endswith("/phase3/build-v3-max-derived/val.jsonl")
    until = datetime.fromisoformat(_opt(argv, "--stop-at")).timestamp()
    assert t0 < until <= t0 + 60 + 2                     # the 1-min time box, not stop-at
    assert any(u.startswith("distill-train-smoke-") for u in box.log("scopes.log"))
    assert box.step("base-9b-baseline")["status"] == "done"
    summary = subprocess.run([str(RUNNER_SH), "--summary"], env=box.env, capture_output=True, text=True)
    assert "smoke done: 50 steps" in summary.stdout
    assert box.run(stop_in=120).returncode == 0
    assert len(_smoke_calls(box)) == 1 and box.step("train-smoke")["detail"] == "done in an earlier window"


def test_smoke_without_the_approval_setting_omits_the_flag(box: Box) -> None:
    _smoke_inputs(box, DJR_SMOKE_MAX_DERIVED_OK="0", FAKE_SMOKE_MODE="refuse")
    assert box.run(stop_in=120).returncode == 1
    (call,) = _smoke_calls(box)
    assert "--i-have-operator-approval-for-max-derived-data" not in call["argv"]
    sm = box.step("train-smoke")
    assert sm["status"] == "failed" and sm["detail"].startswith("refused: Max-derived data")


def test_smoke_cut_by_its_time_box_is_partial_and_resumes(box: Box) -> None:
    _smoke_inputs(box, DJR_SMOKE_MAX_MIN="0.05", FAKE_SMOKE_MODE="until")    # 3 s
    cp = box.run(stop_in=120)
    assert cp.returncode == 0, cp.stdout[-3000:]
    sm = box.step("train-smoke")
    assert sm["status"] == "partial" and "resumes next window" in sm["detail"]
    assert box.step("base-9b-baseline")["status"] == "done"
    box.env["FAKE_SMOKE_MODE"] = "pass"
    assert box.run(stop_in=120).returncode == 0
    calls = _smoke_calls(box)
    assert len(calls) == 2 and "--resume" in calls[1]["argv"] and box.step("train-smoke")["status"] == "done"


def test_smoke_failure_fails_the_step_notifies_and_the_queue_goes_on(box: Box) -> None:
    _smoke_inputs(box, FAKE_SMOKE_MODE="fail")
    assert box.run(stop_in=120).returncode == 1
    sm = box.step("train-smoke")
    assert sm["status"] == "failed" and "DID NOT DECREASE" in sm["detail"]
    assert box.step_state("base-9b/train-smoke-max-derived")["failures"] == 1
    assert "train-smoke FAILED" in (box.state / "notify.log").read_text()
    assert box.step("base-9b-baseline")["status"] == "done"


def test_smoke_inputs_missing_skip_the_step_not_the_window(box: Box) -> None:
    _smoke_inputs(box)
    (box.d / "models" / "qwen3.5-9b-bnb4" / "model.safetensors").unlink()
    dry = box.run(execute=False, stop_in=None)
    assert dry.returncode == 0, dry.stdout[-3000:]
    assert "[     --] QLoRA smoke inputs" in dry.stdout and "qwen3.5-9b-bnb4" in dry.stdout
    assert box.run(stop_in=120).returncode == 0
    assert box.step("train-smoke")["status"] == "skipped" and box.log("smoke-calls.log") == []


def test_sigterm_during_the_smoke_stops_the_trainer(box: Box) -> None:
    _smoke_inputs(box, FAKE_SMOKE_DELAY="60")
    p = box.start(stop_in=120)
    box.wait_for(lambda: (box.state / "smoke-pid").exists(), what="the trainer")
    pid = int((box.state / "smoke-pid").read_text())
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 143
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    st = box.step_state("base-9b/train-smoke-max-derived")
    assert st["status"] == "partial" and st["failures"] == 0


def test_dry_run_shows_the_smoke_first_and_ab_off(box: Box) -> None:
    _smoke_inputs(box)
    cp = box.run(execute=False, stop_in=None)
    assert cp.returncode == 0, cp.stdout[-3000:]
    out = cp.stdout
    assert " 1. train-smoke" in out and " 2. base-9b-baseline" in out and "teacher-ab" not in out
    assert "[     ok] QLoRA smoke inputs" in out and f"@{BASE_REV[:12]}" in out
    assert "[     ok] Max-derived approval flag passed" in out
    assert "--stop-at <time box end>" in out and "--i-have-operator-approval-for-max-derived-data" in out
    assert box.log("smoke-calls.log") == [] and not (box.d / "phase5").exists()
