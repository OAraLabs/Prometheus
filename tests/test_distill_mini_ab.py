"""scripts/distill/mini_window/mini_ab.py: the teacher quant A/B night on the 3090 Ti host, against fakes.

systemd units, the voice probe's journal, nvidia-smi, systemd-run and the notifier are a fake host driven by
a fake runner; teacher_ab.py and the probe guard are stand-ins that record what they were asked; the clock
only moves when the code sleeps. Nothing touches a real unit, GPU or model.
"""

import json
import re
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "distill" / "mini_window"))

import mini_ab as AB  # noqa: E402
import mini_window as MW  # noqa: E402

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
TEACHER = "distill-teacher.service"
Q4P = "distill-teacher-q4-partial.service"
LOCAL = "llama-mini.service"
GEN = "distill-gen-20261031-070638.service"
IQ3_ALIAS = "qwen3.8-27b-gsq-rco-iq3_xxs"
Q4_ALIAS = "qwen3.8-27b-q4_k_m"
MIB = 1 << 20
# the real Q4_K_M's trunk layer sizes (MiB, from its tensor table), the output matrix 995 MiB
Q4_TRUNK = [306.2, 264.9, 264.9, 241.6] + [264.9, 264.9, 264.9, 241.6] * 3 + [215.1, 215.1, 215.1, 219.7] * 8 \
    + [215.1, 215.1, 215.1, 342.1]
LAYERS = {"trunk": [int(x * MIB) for x in Q4_TRUNK], "output": int(994.6 * MIB), "blocks": 65, "nextn": 1}
START = datetime(2026, 11, 1, 1, 0, tzinfo=ET, fold=0).timestamp()  # 01:00 EDT, the first one


def conf(tmp_path, **extra):
    home = tmp_path / "home"
    win = home / "distill" / "window"
    (win / "state").mkdir(parents=True, exist_ok=True)
    notify = home / "notify.sh"
    notify.write_text("#!/bin/sh\ncat >/dev/null\n")
    notify.chmod(0o755)
    lines = {
        "LOCAL_MODEL_UNIT": LOCAL, "PROBE_UNIT": "probe-host.service", "OLLAMA_MODEL": "qwen2.5:7b-instruct",
        "WIN_DIR": str(win), "LOG_DIR": str(home / "logs"), "NOTIFY": str(notify), "SETUPS_DIR": str(win / "setups"),
        "AB_Q4_ENV": str(win / "state" / "q4-partial.env"), "AB_OUT": str(home / "ab"),
        "AB_CODE": str(home / "code"), "GEN_PY": sys.executable, "EVAL_TIMES": "0300,0600",
        "CONTINUOUS_UNTIL": "2026-11-01T18:00:00-05:00", **extra,
    }
    path = win / "window.conf"
    path.write_text("".join(f"{k}={v}\n" for k, v in lines.items()))
    return AB.load(path)


class Host:
    """systemd user units, the probe journal, the GPU and transient runs, on a fake clock."""

    def __init__(self, t, q4_tps=22.0):
        self.t = t
        self.units = {TEACHER: "active", Q4P: "inactive", LOCAL: "inactive", "ollama.service": "active", GEN: "active"}
        self.q4_tps = q4_tps
        self.iq3_comes_back = True
        self.probe_t0 = t - 40
        self.probes = []
        self.calls = []
        self.notified = []
        self.started_gen = []
        self.eval_windows = []  # (start, end) epochs when the nightly eval process runs
        self.flag_after = None
        self.env_seen = []
        self._tick(t)

    # clock and probes

    def sleep(self, s):
        self._tick(self.t + max(0.0, s))

    def now(self):
        return self.t

    def _tick(self, t):
        k = 0
        while self.probe_t0 + k * 300 <= t:
            at = self.probe_t0 + k * 300
            if not self.probes or at > self.probes[-1][0]:
                self.t = at
                self.probes.append((at, self._line(at)))
            k += 1
        self.t = t

    def serving(self):
        up = [u for u in (TEACHER, Q4P, LOCAL) if self.units[u] == "active"]
        if len(up) != 1:
            return None
        return {TEACHER: IQ3_ALIAS, Q4P: Q4_ALIAS, LOCAL: "/models/saluki.gguf"}[up[0]]

    def _line(self, at):
        stamp = datetime.fromtimestamp(at, ET).isoformat(timespec="microseconds")
        s = self.serving()
        if s is None:
            body = "[DARK] inference_gpu verdict=DARK — 503"
        elif s == Q4_ALIAS and self.q4_tps < 15:
            body = f"[SLOW] inference_gpu verdict=SLOW — {self.q4_tps} tok/s (floor 15)"
        else:
            tps = self.q4_tps if s == Q4_ALIAS else 48.0
            body = f"[OK  ] inference_gpu expected=True verdict=OK — {tps} tok/s (floor 15)"
        return f"{stamp} gpu-host python[1]:   {body}"

    def eval_running(self):
        return any(a <= self.t < b for a, b in self.eval_windows)

    # GPU

    def free(self):
        used = {TEACHER: 12344, Q4P: 12650, LOCAL: 13000}
        return 13265 - sum(used[u] for u in used if self.units[u] == "active")

    # the runner

    def run(self, argv, timeout=60, input=None):
        self.calls.append((self.t, list(argv)))
        ok = subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "systemctl":
            verb = argv[2]
            if verb == "show":
                prop, unit = argv[4], argv[-1]
                if prop == "ActiveState":
                    return subprocess.CompletedProcess(argv, 0, self.units.get(unit, "inactive") + "\n", "")
                if prop == "LoadState":
                    return subprocess.CompletedProcess(argv, 0, "loaded\n" if unit in self.units else "not-found\n", "")
                if prop == "MainPID":
                    return subprocess.CompletedProcess(argv, 0, "4242\n" if self.units[TEACHER] == "active" else "0\n", "")
                return subprocess.CompletedProcess(argv, 0, "\n", "")
            if verb == "list-units":
                glob = argv[-1]
                out = "".join(f"{u} loaded {s} running x\n" for u, s in self.units.items()
                              if re.fullmatch(glob.replace("*", ".*"), u.rsplit(".", 1)[0]) or
                              re.fullmatch(glob.replace("*", ".*"), u))
                return subprocess.CompletedProcess(argv, 0, out, "")
            if verb == "stop":
                self.units[argv[3]] = "inactive"
                return ok
            if verb == "start":
                unit = argv[3]
                if unit == Q4P:
                    self.env_seen.append(Path(self.conf_env).read_text())
                if unit == TEACHER and not self.iq3_comes_back:
                    self.units[unit] = "failed"
                    return subprocess.CompletedProcess(argv, 1, "", "failed")
                for other in (TEACHER, Q4P):
                    if other != unit and unit in (TEACHER, Q4P):
                        self.units[other] = "inactive"
                self.units[unit] = "active"
                return ok
            return ok
        if argv[0] == "journalctl":
            since = int(argv[argv.index("--since") + 1].lstrip("-").rstrip("s"))
            lines = [ln for at, ln in self.probes if at >= self.t - since]
            return subprocess.CompletedProcess(argv, 0, "\n".join(lines) + "\n", "")
        if argv[0] == "nvidia-smi":
            if argv[1].startswith("--query-gpu"):
                return subprocess.CompletedProcess(argv, 0, f"{self.free()}\n", "")
            return subprocess.CompletedProcess(argv, 0, "4242, 12344\n" if self.units[TEACHER] == "active" else "", "")
        if argv[0] == "systemd-run":
            unit = next(a.split("=", 1)[1] for a in argv if a.startswith("--unit="))
            if unit.startswith("distill-gen-"):
                self.started_gen.append(unit)
                self.units[unit + ".service"] = "active"
            return ok
        if argv[0].endswith("notify.sh"):
            self.notified.append(input)
            return ok
        if "setups.py" in " ".join(argv):  # the business restore
            if "--execute" in argv:
                self.units.update({TEACHER: "inactive", Q4P: "inactive", LOCAL: "active"})
            return subprocess.CompletedProcess(argv, 0, "  mini: stop [] ; start [] ; keep []\n", "")
        return ok


class StubAb(AB.Ab):
    """teacher_ab.py stand-in: counts tasks per phase on the fake clock; records each step's URL and server."""

    def __init__(self, c, host, n=30, task_s=(300, 900, 150), fail=None):
        super().__init__(c, Path(c["AB_OUT"]), lambda m: None, host.now, host.sleep)
        self.host = host
        self.n = n
        self.task_s = dict(zip(("arm-A", "arm-B", "judge"), task_s))
        self.done = {"arm-A": 0, "arm-B": 0, "judge": 0}
        self.steps = []
        self.fail = fail or {}
        self.reported = False

    def select(self):
        return {"n": self.n}

    def status(self):
        n = self.n
        return {"selected": n, "arm-A": {"done": self.done["arm-A"], "total": n},
                "arm-B": {"done": self.done["arm-B"], "total": n},
                "judge": {"done": self.done["judge"], "total": self.done["arm-A"] + self.done["arm-B"],
                          "arms_complete": self.done["arm-A"] == n and self.done["arm-B"] == n}}

    def step(self, args, watch):
        kind = f"arm-{args[args.index('--arm') + 1]}" if args[0] == "arm" else "judge"
        url = args[args.index("--url" if args[0] == "arm" else "--judge-url") + 1]
        until = datetime.fromisoformat(args[args.index("--until") + 1]).timestamp()
        self.steps.append({"kind": kind, "url": url, "t0": self.host.t, "until": until,
                           "serving": self.host.serving()})
        if kind in self.fail:
            raise self.fail[kind]
        total = self.n if kind != "judge" else self.done["arm-A"] + self.done["arm-B"]
        while self.done[kind] < total and self.host.t + 300 <= until:
            why = watch()
            if why:
                self.steps[-1]["stopped"] = why
                return 1, {"stop": "watch"}, why
            if self.host.t + self.task_s[kind] > until:
                self.host.sleep(until - self.host.t)
                break
            self.host.sleep(self.task_s[kind])
            self.done[kind] += 1
        self.steps[-1]["t1"] = self.host.t
        left = total - self.done[kind]
        return (0 if not left else 1), {"left": left, "stop": "done" if not left else "until"}, ""

    def report(self):
        self.reported = True
        k = self.done["judge"]
        return (0 if k == 2 * self.n else 1), [f"Teacher A/B: line {i}" for i in range(1, 6)]

    def call(self, *args, timeout=900):
        return 0, "--allow-ctx-difference", ""

    def stop(self):
        pass


class StubGuard(AB.Guard):
    def __init__(self, c, w, deadline):
        super().__init__(c, w, deadline, fake=False)
        self.up = False
        self.starts = []

    def start(self, llama_unit, alias):
        self.starts.append((self.w.now(), llama_unit, alias))
        self.up = True

    def running(self):
        return self.up

    def aborted(self):
        return None

    def stop(self):
        self.up = False

    def stop_recorded(self):
        pass


def night(tmp_path, host=None, *, start=START, n=30, task_s=(300, 900, 150), fail=None, **extra):
    c = conf(tmp_path, **extra)
    host = host or Host(start)
    host.conf_env = c["AB_Q4_ENV"]
    for f in ("AB_Q4_MODEL", "AB_TASKS", "GEN_GUARD_PY"):
        Path(c[f]).parent.mkdir(parents=True, exist_ok=True)
        Path(c[f]).write_text("x")
    (Path(c["AB_CODE"]) / "src").mkdir(parents=True, exist_ok=True)
    (Path(c["AB_CODE"]) / "scripts" / "distill").mkdir(parents=True, exist_ok=True)
    (Path(c["AB_CODE"]) / "scripts" / "distill" / "teacher_ab.py").write_text("")
    runner = Path(c["WIN_DIR"]) / "gen_runner.sh"
    runner.write_text("#!/bin/sh\n")
    runner.chmod(0o755)
    out = []
    w = MW.Window(c, True, "ab-night", runner=host.run, clock=host.now, sleep=host.sleep, out=out.append)
    w.served = lambda: {"health": 200 if host.serving() else 503, "id": host.serving(),  # type: ignore[method-assign]
                        "ids": [host.serving()] if host.serving() else []}
    w.daemon_ok = lambda: (True, "daemon /health 200 ok")  # type: ignore[method-assign]
    w.eval_running = host.eval_running  # type: ignore[method-assign]
    w.business_problems = lambda: [] if host.units[LOCAL] == "active" else ["local down"]  # type: ignore[method-assign]
    w.start_log()
    times = AB.night_times(c, start)
    ab = StubAb(c, host, n=n, task_s=task_s, fail=fail)
    g = StubGuard(c, w, times["hard"])
    nt = AB.Night(w, c, ab, g, times, fake=False)
    nt.layers = LAYERS
    return nt, host, ab, g, out


# ── time ─────────────────────────────────────────────────────────────────────


def test_tonight_falls_back_at_two_and_the_times_are_epochs(tmp_path):
    c = conf(tmp_path)
    t = AB.night_times(c, START)
    assert AB.calendar_utc(START) == "2026-11-01 05:00:00 UTC"  # 01:00 EDT, not the second 01:00 (EST)
    assert AB.calendar_utc(t["hard"]) == "2026-11-01 11:30:00 UTC"  # 06:30 EST: 6.5 h of wall time
    assert [AB.calendar_utc(e) for e in t["evals"]] == ["2026-11-01 08:00:00 UTC", "2026-11-01 11:00:00 UTC"]
    # an eval hold: Q4 must be off 35 min before 03:00 EST and may come back 3 min after (once it is not running)
    e = t["evals"][0]
    assert AB.next_eval_hold(START, t["evals"], 35 * 60) == e - 35 * 60
    assert AB.eval_free_at(e - 30 * 60, t["evals"], 35 * 60, 180, False) == e + 180
    assert AB.eval_free_at(e + 600, t["evals"], 35 * 60, 180, True) == e + 600 + 30
    assert AB.eval_free_at(e + 600, t["evals"], 35 * 60, 180, False) is None


# ── the GPU plan ─────────────────────────────────────────────────────────────


def test_ngl_plan_keeps_600_mib_headroom_and_counts_conservatively(tmp_path):
    c = conf(tmp_path)
    ngl, proj = AB.plan_ngl(LAYERS, 13265, c)  # the IQ3 teacher's 12,344 MiB + the 921 free today
    assert 44 <= ngl <= 47  # the operator's estimate: -ngl 44-46
    assert proj + 600 <= 13265
    bigger, _ = AB.plan_ngl(LAYERS, 13265 + 230, c)
    assert bigger == ngl + 1  # one more ~215 MiB layer
    assert AB.plan_ngl(LAYERS, 20000, c)[0] == 99  # everything fits: -ngl 99, the output on the GPU too
    assert AB.plan_ngl(LAYERS, 3000, c)[0] < int(c["AB_MIN_NGL"])


def _gguf(path, tensors, meta):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str) else struct.pack("<II", 4, v))
    off = 0
    infos = b""
    for name, nbytes in tensors:
        infos += s(name) + struct.pack("<I", 1) + struct.pack("<Q", nbytes) + struct.pack("<IQ", 0, off)
        off += nbytes
    out += infos
    out += b"\0" * ((-len(out)) % 32)
    path.write_bytes(out + b"\1" * off)


def test_gguf_layer_sizes_come_from_the_tensor_table(tmp_path):
    p = tmp_path / "m.gguf"
    _gguf(p, [("token_embd.weight", 64), ("blk.0.attn.weight", 96), ("blk.0.ffn.weight", 32),
              ("blk.1.ffn.weight", 160), ("blk.2.mtp.weight", 32), ("output.weight", 256)],
          {"general.architecture": "qwen35", "qwen35.block_count": 3, "qwen35.nextn_predict_layers": 1})
    got = AB.gguf_layers(p)
    assert got["trunk"] == [128, 160]  # blk.2 is the MTP block: never loaded
    assert got["output"] == 256 and got["blocks"] == 3


# ── the night ────────────────────────────────────────────────────────────────


def _systemctl(host, verb):
    return [(t, a[3]) for t, a in host.calls if a[:3] == ["systemctl", "--user", verb]]


def test_the_whole_night_in_order_around_the_evals(tmp_path):
    nt, host, ab, g, out = night(tmp_path, task_s=(120, 180, 45))  # fast enough to finish in the Q4 phases
    rc = nt.run()
    assert rc == 0, "\n".join(out[-30:])
    stops = _systemctl(host, "stop")
    starts = _systemctl(host, "start")
    # 1. generation paused by name before anything else changed or ran, and how to resume it recorded
    assert stops[0][1] == GEN and stops[0][0] <= ab.steps[0]["t0"]
    state = json.loads((nt.w.state / AB.STATE_NAME).read_text())
    assert state["gen_stopped"][0]["unit"] == GEN
    assert "gen_runner.sh run --execute" in state["gen_resume"] and "--unit=distill-gen-" in state["gen_resume"]
    # 2.-6. every A/B request through the guard; arm A on IQ3, arm B and the judge on Q4
    assert {s["url"] for s in ab.steps} == {"http://127.0.0.1:18593"}
    assert {s["serving"] for s in ab.steps if s["kind"] == "arm-A"} == {IQ3_ALIAS}
    assert {s["serving"] for s in ab.steps if s["kind"] != "arm-A"} == {Q4_ALIAS}
    assert ab.done == {"arm-A": 30, "arm-B": 30, "judge": 60}
    first_b = next(s for s in ab.steps if s["kind"] == "arm-B")
    assert first_b["t0"] > max(s["t1"] for s in ab.steps if s["kind"] == "arm-A")
    # the Q4 unit got the -ngl chosen from the VRAM read right before its start, with 600 MiB headroom
    env = dict(line.split("=") for line in host.env_seen[0].split())
    assert 44 <= int(env["NGL"]) <= 47 and int(env["MIN_FREE_MIB"]) <= 13265
    # no Q4 step within 35 min of an eval or across it, nothing past the hard end
    for s in ab.steps:
        assert s["t1"] <= nt.t["hard"]
        if s["kind"] != "arm-A":
            for e in nt.t["evals"]:
                assert s["t1"] <= e - 35 * 60 or s["t0"] >= e
    # Q4 swapped out before the 03:00 eval and in again after it
    q4_starts = [t for t, u in starts if u == Q4P]
    assert len(q4_starts) == 2 and q4_starts[0] < nt.t["evals"][0] < q4_starts[1]
    # 8. IQ3 back and verified, Q4 stopped, generation restarted, five lines sent
    assert host.units[TEACHER] == "active" and host.units[Q4P] == "inactive"
    assert host.started_gen and host.units[host.started_gen[0] + ".service"] == "active"
    assert ab.reported and len(host.notified[-1].strip().splitlines()) == 5
    assert not g.up


def test_a_slow_voice_probe_on_q4_swaps_back_and_ends_the_night(tmp_path):
    host = Host(START, q4_tps=8.6)  # the partly offloaded Q4 below the probe's 15 tok/s floor
    nt, host, ab, g, out = night(tmp_path, host)
    rc = nt.run()
    assert rc == 1
    assert not [s for s in ab.steps if s["kind"] != "arm-A"], "arm B never ran on a server the probe failed"
    assert nt.q4_refused.startswith("voice probe after the change: SLOW")
    assert host.units[TEACHER] == "active" and host.units[Q4P] == "inactive"
    assert host.started_gen, "generation restarted"
    assert any("no Q4 tonight" in e for e in nt.events)
    assert len(host.notified[-1].strip().splitlines()) == 5


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt("signal 15")])
def test_an_error_or_a_signal_mid_arm_b_still_swaps_back_and_restarts_generation(tmp_path, exc):
    nt, host, ab, g, out = night(tmp_path, fail={"arm-B": exc})
    rc = nt.run()
    assert rc == 1
    assert host.units[TEACHER] == "active" and host.units[Q4P] == "inactive"
    assert host.started_gen
    assert ab.reported
    last_iq3 = max(t for t, u in _systemctl(host, "start") if u == TEACHER)
    assert last_iq3 > max(t for t, u in _systemctl(host, "start") if u == Q4P)


def test_the_hard_end_stops_the_arms_and_still_swaps_back(tmp_path):
    nt, host, ab, g, out = night(tmp_path, task_s=(300, 2400, 600))  # Q4 far too slow to finish tonight
    rc = nt.run()
    assert rc == 0  # ended cleanly by the time box; the report says incomplete
    assert ab.done["arm-B"] < 30
    assert all(s["t1"] <= nt.t["hard"] for s in ab.steps)
    assert host.units[TEACHER] == "active" and host.units[Q4P] == "inactive"
    assert host.t < nt.t["hard"] + 30 * 60
    assert host.started_gen


def test_iq3_that_will_not_come_back_falls_back_to_business_without_generation(tmp_path):
    nt, host, ab, g, out = night(tmp_path, fail={"arm-B": RuntimeError("boom")})
    host.iq3_comes_back = False
    rc = nt.run()
    assert rc == 1
    assert host.units[LOCAL] == "active" and host.units[Q4P] == "inactive"
    assert not host.started_gen, "no generation without the teacher on the port"
    assert nt.w.flag.exists()


def test_preflight_refuses_with_nothing_changed(tmp_path):
    nt, host, ab, g, out = night(tmp_path)
    nt.w.flag.write_text("x\n")
    with pytest.raises(MW.Skip, match="NO-SWAPS"):
        nt.run()
    assert not _systemctl(host, "stop") and not _systemctl(host, "start") and not ab.steps


def test_restore_puts_iq3_back_and_resumes_generation_a_dead_run_paused(tmp_path):
    nt, host, ab, g, out = night(tmp_path)
    host.units.update({TEACHER: "inactive", Q4P: "failed", GEN: "inactive"})
    (nt.w.state / AB.STATE_NAME).write_text(json.dumps({"gen_stopped": [{"unit": GEN}], "finished": False}))
    assert AB.cmd_restore(nt) == 0
    assert host.units[TEACHER] == "active" and host.started_gen
    # nothing to do the second time
    host.calls.clear()
    assert AB.cmd_restore(nt) == 0
    assert not _systemctl(host, "start")


def test_arm_prints_the_timers_in_utc(tmp_path):
    c = conf(tmp_path)
    host = Host(START - 15 * 3600)
    out = []
    w = MW.Window(c, False, "ab-arm", runner=host.run, clock=host.now, sleep=host.sleep, out=out.append)
    assert AB.cmd_arm(w, c, datetime(2026, 11, 1).date(), "0100", False) == 0
    text = "\n".join(out)
    assert "--unit=window-mini-ab-20261101" in text and "--on-calendar=2026-11-01 05:00:00 UTC" in text
    assert "RuntimeMaxSec=25800" in text and "mini_ab_tonight.sh --execute" in text
    assert "--unit=window-mini-ab-backstop-20261101" in text and "2026-11-01 12:25:00 UTC" in text
    assert not any(a[0] == "systemd-run" for _, a in host.calls), "dry run: nothing armed"


def test_the_q4_unit_is_the_teacher_shape_with_partial_offload():
    units = REPO / "scripts" / "distill" / "units"
    q4p = (units / "distill-teacher-q4-partial.service").read_text()
    iq3 = (units / "distill-teacher.service").read_text()

    def flags(text):
        ex = re.search(r"^ExecStart=(.*?)(?=^\S)", text, re.S | re.M).group(1)
        return ex.replace("\\\n", " ").split()

    a, b = flags(iq3), flags(q4p)
    for f in ("--parallel", "--temp", "--top-k", "--top-p", "--min-p", "-ctk", "-ctv", "--ubatch-size",
              "--batch-size", "--reasoning-budget", "--port"):
        assert a[a.index(f) + 1] == b[b.index(f) + 1], f
    assert "--kv-unified" in b and b[b.index("-ngl") + 1] == "${NGL}" and b[b.index("-c") + 1] == "${CTX}"
    assert "Environment=CTX=16384" in q4p and "EnvironmentFile=%h/distill/window/state/q4-partial.env" in q4p
    assert "MIN_FREE_MIB" in q4p and "Conflicts=distill-teacher.service distill-teacher-q4.service" in q4p
    assert not re.search(r"^\[Install\]", q4p, re.M) and "Restart=no" in q4p
    assert "OnFailure=window-mini-ab-restore.service" in q4p


def test_the_restore_handler_unit_and_the_installer():
    units = REPO / "scripts" / "distill" / "units"
    name = "window-mini-ab-restore.service"
    text = (units / name).read_text()
    assert not name.startswith("distill-")  # apply business stops every running distill-* unit
    assert not re.search(r"^\[Install\]", text, re.M) and "/home/" not in text
    assert re.search(r"^Type=oneshot$", text, re.M) and re.search(r"^KillMode=mixed$", text, re.M)
    assert re.search(r"^ExecStart=.*mini_ab\.py --conf %h/distill/window/window\.conf restore --execute$", text, re.M)
    installer = (units / "install_teacher_units.sh").read_text()
    ab_units = re.search(r"^ab_units=\((.+)\)$", installer, re.M).group(1).split()
    assert ab_units == ["distill-teacher-q4-partial.service", name]
    wrapper = (REPO / "scripts" / "distill" / "mini_window" / "mini_ab_tonight.sh").read_text()
    assert "window-mini-ab-20261101" in wrapper and "TimeoutStopSec=20min" in wrapper
