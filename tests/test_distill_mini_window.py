"""scripts/distill/mini_window/mini_window.py: the 3090 Ti window steps (Track B), against fakes.

The model port, Ollama and the daemon are real HTTP servers on loopback (ephemeral ports) whose answers
follow a fake host: systemd units, the dead-man timer, the voice probe's journal and setups.py are all
simulated by a fake runner, and the clock only moves when the code sleeps. Nothing touches a real unit.
"""

import json
import re
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "distill" / "mini_window"))

import mini_window as M  # noqa: E402

ET = ZoneInfo("America/New_York")
LOCAL = "local-model.service"
TEACHER = "distill-teacher.service"
Q4 = "distill-teacher-q4.service"
OLLAMA = "ollama.service"
LOCAL_ID = "/models/local.gguf"
IQ3_ALIAS = "qwen3.8-27b-gsq-rco-iq3_xxs"
Q4_ALIAS = "qwen3.8-27b-q4_k_m"
JUDGE = "judge:7b"
DEADMAN = "distill-deadman-mini"


def epoch(text):
    return datetime.fromisoformat(text).replace(tzinfo=ET).timestamp()


# ── the fake host ────────────────────────────────────────────────────────────


class Host:
    """systemd user units, a dead-man timer, the probe journal and setups.py, on a fake clock."""

    SETUPS = {  # target units in start order, and units the setup stops
        "business": ([LOCAL, OLLAMA], [TEACHER, Q4]),
        "flywheel": ([TEACHER, OLLAMA], [LOCAL, Q4]),
        "flywheel-night-q4": ([Q4], [LOCAL, OLLAMA, TEACHER]),
    }
    ALIAS = {LOCAL: LOCAL_ID, TEACHER: IQ3_ALIAS, Q4: Q4_ALIAS}

    def __init__(self, start, layout="business"):
        self.t = start
        self.units = {u: "inactive" for u in (LOCAL, TEACHER, Q4, OLLAMA)}
        self.load = {u: "loaded" for u in self.units}
        run, _ = self.SETUPS[layout]
        for u in run:
            self.units[u] = "active"
        self.extra = {}            # other services (gen, baseline): name -> state
        self.timers = {}           # timer name -> {"sub": ..., "next": epoch}
        self.ready_at = 0.0        # the model port answers once the clock passes this
        self.load_s = 20.0
        self.probe_t0 = start - 40  # a probe result 40 s before the start, then every 300 s
        self.probes = []
        self.probe_override = None  # force a verdict line
        self.fail_apply = set()     # setups whose --execute fails
        self.ollama_ps = (5828081664, 5828081664)  # the judge's /api/ps (size, size_vram) while Ollama runs
        self.never_ready = set()    # units whose server never answers
        self.gen_runs = []          # transient services started (the generation runner)
        self.fail_gen = False       # its systemd-run fails
        self.calls = []
        self.applied = []
        self.applied_at = []
        self.notified = []
        self.stopped = []
        self._tick_to(start)

    # clock

    def sleep(self, s):
        self._tick_to(self.t + max(0.0, s))

    def _tick_to(self, t):
        k = 0
        while self.probe_t0 + k * 300 <= t:
            at = self.probe_t0 + k * 300
            if at > (self.probes[-1][0] if self.probes else float("-inf")):
                self.t = at
                self.probes.append((at, self._probe_line(at)))
            k += 1
        self.t = t

    def _probe_line(self, at):
        stamp = datetime.fromtimestamp(at, ET).isoformat(timespec="microseconds")
        if self.probe_override:
            body = self.probe_override
        elif self.serving() is not None:
            body = "[OK  ] inference_gpu          expected=True verdict=OK — 45.2 tok/s (floor 15)"
        else:
            body = "[DARK] inference_gpu          expected=True verdict=DARK — http://127.0.0.1:8092/completion -> 503"
        return f"{stamp} host python[1]:   {body}"

    # what the HTTP fakes answer

    def on_port(self):
        return [u for u in (LOCAL, TEACHER, Q4) if self.units[u] == "active"]

    def serving(self):
        up = self.on_port()
        if len(up) != 1 or self.t < self.ready_at or up[0] in self.never_ready:
            return None
        return self.ALIAS[up[0]]

    def ollama_up(self):
        return self.units[OLLAMA] == "active"

    # the runner

    def __call__(self, argv, timeout=60, input=None):
        self.calls.append(list(argv))
        out, rc = self._dispatch(list(argv), input)
        return subprocess.CompletedProcess(argv, rc, out, "")

    def state_of(self, unit):
        if unit in self.units:
            return self.units[unit]
        return self.extra.get(unit, "inactive")

    def _dispatch(self, a, input):
        if a[:3] == ["systemctl", "--user", "show"]:
            prop, unit = a[a.index("-p") + 1], a[-1]
            return self._show(unit, prop), 0
        if a[:3] == ["systemctl", "--user", "list-units"]:
            kind = [x for x in a if x.startswith("--type=")][0].split("=")[1]
            glob = a[-1]
            import fnmatch
            lines = []
            if kind == "timer":
                for name, tm in self.timers.items():
                    if fnmatch.fnmatch(name, glob):
                        lines.append(f"{name} loaded active {tm['sub']} timer")
            else:
                for name, st in {**self.units, **self.extra}.items():
                    if fnmatch.fnmatch(name, glob) and name.endswith(".service"):
                        lines.append(f"{name} loaded {st} {'running' if st == 'active' else 'dead'} svc")
            return "\n".join(lines) + "\n", 0
        if a[:3] == ["systemctl", "--user", "stop"]:
            unit = a[3]
            self.stopped.append(unit)
            if unit.endswith(".timer"):
                self.timers.pop(unit, None)
            elif unit in self.units:
                self.units[unit] = "inactive"
            else:
                self.extra[unit] = "inactive"
            return "", 0
        if a[:3] == ["systemctl", "--user", "reset-failed"]:
            return "", 0
        if a[:2] == ["systemd-run", "--user"]:
            unit = [x for x in a if x.startswith("--unit=")][0].split("=", 1)[1]
            if not any(x.startswith("--on-calendar=") for x in a):  # a transient service: generation
                if self.fail_gen:
                    return "Failed to start transient service unit\n", 1
                self.gen_runs.append(a)
                self.extra[unit + ".service"] = "active"
                return "", 0
            cal = [x for x in a if x.startswith("--on-calendar=")][0].split("=", 1)[1]
            when = datetime.strptime(cal, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=ZoneInfo("UTC")).timestamp()
            self.timers[unit + ".timer"] = {"sub": "waiting", "next": when, "argv": a}
            return "", 0
        if a[0] == "journalctl":
            since = int(a[a.index("--since") + 1].lstrip("-").rstrip("s"))
            return "\n".join(line for t, line in self.probes if self.t - since <= t <= self.t) + "\n", 0
        if a[:2] == ["id", "-un"]:
            return "tester\n", 0
        if a[0] == "loginctl":
            return "yes\n", 0
        if len(a) == 3 and a[1:] == ["-c", "import yaml"]:
            return "", 0
        if len(a) > 1 and a[1].endswith("/setups.py"):
            i = a.index("apply")
            return self._setups(a[i + 1], "--execute" in a)
        if a[-1] == "5" and input is not None:  # notify
            self.notified.append(input)
            return "", 0
        raise AssertionError(f"unexpected command {a}")

    def _show(self, unit, prop):
        if unit.endswith(".timer"):
            tm = self.timers.get(unit)
            if prop == "LoadState":
                return "loaded" if tm else "not-found"
            if prop == "ActiveState":
                return "active" if tm else "inactive"
            if prop == "SubState":
                return tm["sub"] if tm else "dead"
            if prop == "NextElapseUSecRealtime":
                return datetime.fromtimestamp(tm["next"], ET).strftime("%a %Y-%m-%d %H:%M:%S %Z") if tm else ""
        if prop == "LoadState":
            return self.load.get(unit, "not-found") if unit in self.units else "loaded"
        if prop == "ActiveState":
            return self.state_of(unit)
        if prop == "ExecStart" and unit == LOCAL:
            return ("{ path=/opt/llama-server ; argv[]=/opt/llama-server -m " + LOCAL_ID
                    + " --port 8092 -c 65536 ; ignore_errors=no ; start_time=[n/a] }")
        return ""

    def _plan(self, name):
        run, stop = self.SETUPS[name]
        stopping = [u for u in stop if self.units[u] == "active"]
        if name == "business":
            stopping += sorted(u for u, s in self.extra.items() if s == "active" and u.startswith("distill-"))
        else:  # the flywheel setups keep distill-* jobs and other business units they do not name
            stopping += [u for u in (LOCAL, OLLAMA) if self.units[u] == "active" and u not in run + stop]
        starting = [u for u in run if self.units[u] != "active"]
        keep = [u for u in run if self.units[u] == "active"] + ["tts.service"]
        return sorted(stopping), starting, keep

    def _setups(self, name, execute):
        stop, start, keep = self._plan(name)
        text = (f"apply {name} ({'EXECUTE' if execute else 'dry run'}):\n"
                f"  mini: stop {stop or '-'}; start {start or '-'}; keep {keep or '-'}\n")
        if not execute:
            return text, 0
        self.applied.append(name)
        self.applied_at.append(self.t)
        if name in self.fail_apply:
            return text + "setup FAILED\n", 1
        for u in stop:
            if u in self.units:
                self.units[u] = "inactive"
            else:
                self.extra[u] = "inactive"
        for u in start:
            self.units[u] = "active"
        if any(u in (LOCAL, TEACHER, Q4) for u in start):
            self.ready_at = self.t + self.load_s
        return text + f"setup {name} applied and healthy\n", 0


class Server:
    """One loopback HTTP server answering from a callable: path -> (status, json)."""

    def __init__(self, answer):
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                status, body = outer.answer(self.path)
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.answer = answer
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def rig(tmp_path):
    made = []

    def make(start="2026-11-02T01:30:00", layout="business"):
        host = Host(epoch(start), layout)

        def model(path):
            alias = host.serving()
            if alias is None:
                return 503, {"error": "loading"}
            if path == "/health":
                return 200, {"status": "ok"}
            return 200, {"data": [{"id": alias, "aliases": [alias]}]}

        def ollama(path):
            if not host.ollama_up():
                return 503, {}
            if path == "/api/ps":
                size, vram = host.ollama_ps
                return 200, {"models": [{"name": JUDGE, "model": JUDGE, "size": size, "size_vram": vram}]}
            return 200, {"models": [{"name": JUDGE}, {"name": "other:1b"}]}

        servers = [Server(model), Server(ollama), Server(lambda p: (200, {"status": "ok", "service": "x"}))]
        made.extend(servers)
        base = tmp_path / f"host{len(made)}"
        win = base / "window"
        (win / "setups").mkdir(parents=True)
        for n in ("setups.py", "hosts.yaml", "business.yaml", "flywheel.yaml", "flywheel-night-q4.yaml"):
            (win / "setups" / n).write_text("# fake\n")
        notify = base / "notify.sh"
        notify.write_text("#!/bin/sh\ncat >/dev/null\n")
        notify.chmod(0o755)
        proc = base / "proc"
        proc.mkdir()
        conf = base / "window.conf"
        conf.write_text("\n".join([
            "HOST_KEY=mini", f"WIN_DIR={win}", f"SETUPS_DIR={win}/setups", f"LOG_DIR={base}/logs",
            f"NOTIFY={notify}", "PY=/usr/bin/python3",
            f"LOCAL_MODEL_UNIT={LOCAL}", "PROBE_UNIT=voice.service", f"OLLAMA_MODEL={JUDGE}",
            f"MODEL_URL={servers[0].url}", f"OLLAMA_URL={servers[1].url}",
            f"DAEMON_HEALTH_URL={servers[2].url}/health", f"PROC_ROOT={proc}",
            "CONTINUOUS_UNTIL=", "",
        ]))
        host.conf, host.win, host.proc = conf, win, proc
        lines = []
        host.lines = lines

        def run(*argv):
            return M.main(["--conf", str(conf), *argv], runner=host, clock=lambda: host.t,
                          sleep=host.sleep, out=lines.append)

        host.main = run
        return host

    yield make
    for s in made:
        s.close()


def enter_slot(host, at="2026-11-02T02:50:00"):
    """state/slot.active as slot-enter writes it (epoch first) at ``at``."""
    (host.win / "state").mkdir(exist_ok=True)
    (host.win / "state" / "slot.active").write_text(f"{epoch(at):.0f} {at} {IQ3_ALIAS} + Ollama\n")


def forward_applies(host):
    return [a for a in host.calls if len(a) > 1 and a[1].endswith("/setups.py") and "--execute" in a]


def changes(host):
    """The calls that change something on the host."""
    out = []
    for a in host.calls:
        if a[:3] in (["systemctl", "--user", "stop"], ["systemctl", "--user", "start"]) or a[0] == "systemd-run":
            out.append(a)
        elif len(a) > 1 and a[1].endswith("/setups.py") and "--execute" in a and a[0] != "systemd-run":
            out.append(a)
    return out


# ── parsers ──────────────────────────────────────────────────────────────────


def test_parse_probes_reads_verdict_speed_and_floor():
    lines = [
        "2026-10-30T20:55:21.136478-04:00 host python[3]:   [OK  ] inference_gpu          expected=True "
        "verdict=OK — 51.5 tok/s (floor 15)",
        "2026-10-30T21:00:21.4-04:00 host python[3]:   [DARK] inference_gpu   expected=True verdict=DARK — "
        "9.1 tok/s < 15 floor",
        "2026-10-30T21:01:00.0-04:00 host python[3]:   [OK  ] tts   verdict=OK",
        "garbage inference_gpu verdict=OK",
    ]
    ps = M.parse_probes(lines, "inference_gpu")
    assert [(p.verdict, p.tps, p.floor) for p in ps] == [("OK", 51.5, 15.0), ("DARK", 9.1, None)]
    assert ps[0].ok(15) and not ps[1].ok(15)
    assert not M.Probe(0, "OK", 14.0, 15.0).ok(10)          # the line's own floor counts
    assert not M.Probe(0, "OK", None, None).ok(15)          # no speed: not OK


def test_parse_plan_lists_and_problems():
    text = ("apply flywheel (dry run):\n"
            "  mini: stop ['a.service']; start ['b.service', 'c.service']; keep -\n"
            "  mini: pre_apply: daemon_switch.py to-max (disabled)\n"
            "  mini: REFUSED: would stop never-stop unit x.service\n"
            "  other: stop -; start ['z.service']; keep -\n")
    p = M.parse_plan(text, "mini")
    assert p["found"] and p["stop"] == ["a.service"] and p["start"] == ["b.service", "c.service"]
    assert p["keep"] == [] and p["restart"] == []
    assert p["problems"] == ["mini: REFUSED: would stop never-stop unit x.service"]
    p = M.parse_plan("  mini: stop -; start -; keep ['k.service']; restart ['r.service']\n", "mini")
    assert p["restart"] == ["r.service"] and p["stop"] == []


def test_load_conf_expands_home_and_requires_site_names(tmp_path):
    f = tmp_path / "w.conf"
    f.write_text("# c\nLOCAL_MODEL_UNIT=m.service   # trailing comment\nPROBE_UNIT='p.service'\n"
                 "OLLAMA_MODEL=j:7b\nSETUPS_DIR=$HOME/w/setups\nNOTIFY=~/n.sh\n")
    c = M.load_conf(f, home="/h")
    assert c["LOCAL_MODEL_UNIT"] == "m.service" and c["PROBE_UNIT"] == "p.service"
    assert c["SETUPS_DIR"] == "/h/w/setups" and c["NOTIFY"] == "/h/n.sh"
    assert c["TEACHER_ALIAS"] == "qwen3.8-27b-gsq-rco-iq3_xxs"
    f.write_text("LOCAL_MODEL_UNIT=m.service\n")
    with pytest.raises(SystemExit, match="PROBE_UNIT"):
        M.load_conf(f, home="/h")


def test_parse_stamp_handles_dst_and_utc(rig):
    host = rig()
    w = M.Window(M.load_conf(host.conf), False, "t", runner=host, clock=lambda: host.t)
    assert w.parse_stamp("Mon 2026-11-02 07:15:00 EST") == epoch("2026-11-02T07:15:00")
    assert w.parse_stamp("Sat 2026-10-31 07:15:00 EDT") == datetime.fromisoformat(
        "2026-10-31T07:15:00-04:00").timestamp()
    # 01:30 happens twice on 2026-11-01: the abbreviation picks the right one
    assert w.parse_stamp("Sun 2026-11-01 01:30:00 EST") - w.parse_stamp("Sun 2026-11-01 01:30:00 EDT") == 3600
    assert w.parse_stamp("Mon 2026-11-02 12:15:00 UTC") == epoch("2026-11-02T07:15:00")
    assert w.parse_stamp("n/a") is None


# ── rehearsal ────────────────────────────────────────────────────────────────


def test_rehearsal_dry_run_changes_nothing(rig):
    host = rig()
    assert host.main("rehearsal") == 0
    assert changes(host) == [] and host.applied == [] and host.notified == []
    text = "\n".join(host.lines)
    assert "flywheel plan as expected" in text
    assert "[dry run] would run: systemd-run --user --unit=distill-deadman-mini" in text
    assert "[dry run] would run: /usr/bin/python3" in text and "apply flywheel --execute" in text
    assert "apply business --execute" in text and "[dry run] would notify" not in text.split("restore plan")[0]


def test_rehearsal_round_trip_passes_and_cancels_its_dead_man(rig):
    host = rig()
    assert host.main("rehearsal", "--execute") == 0, "\n".join(host.lines)
    assert host.applied == ["flywheel", "business"]
    run = [a for a in host.calls if a[0] == "systemd-run"][0]
    assert run[-4:] == ["--only-host", "mini", "apply", "business"] or run[-5:] == [
        "--only-host", "mini", "apply", "business", "--execute"]
    assert "--execute" in run and "--local" in run
    assert f"{DEADMAN}.timer" not in host.timers           # armed by the rehearsal, cancelled after PASS
    assert host.units[LOCAL] == "active" and host.units[TEACHER] == "inactive"
    assert (host.win / "state" / "rehearsal.last").read_text().startswith("PASS")
    assert len(host.notified) == 1 and host.notified[0].startswith("mini rehearsal PASS")
    assert "45.2 tok/s" in host.notified[0] and not (host.win / M.FLAG_NAME).exists()
    assert f"judge {JUDGE} fully on the GPU (5558 MiB)" in host.notified[0]
    # the forward swap is one attempt (its failure restores at once); the restore keeps setups.py's retries
    fwd, back = forward_applies(host)
    assert fwd[-4:] == ["flywheel", "--execute", "--attempts", "1"] and "--attempts" not in back


def test_rehearsal_reports_a_judge_partly_on_the_cpu(rig):
    host = rig()
    host.ollama_ps = (5828081664, 5300000000)
    assert host.main("rehearsal", "--execute") == 0
    assert "PARTLY ON THE CPU (5054 of 5558 MiB on the GPU)" in host.notified[0]
    assert "WARNING: nightly evals run slower" in host.notified[0]


def test_rehearsal_skips_while_the_baseline_is_armed(rig):
    host = rig()
    host.timers["distill-baseline-a.timer"] = {"sub": "waiting", "next": host.t + 3600}
    assert host.main("rehearsal", "--execute") == 2
    assert changes(host) == []
    assert "Saluki baseline is still armed" in "\n".join(host.lines)
    assert host.notified and "skipped (nothing changed)" in host.notified[0]


def test_rehearsal_skips_outside_hours_and_near_an_eval(rig):
    host = rig(start="2026-11-02T14:00:00")
    assert host.main("rehearsal", "--execute") == 2 and changes(host) == []
    host = rig(start="2026-11-02T02:40:00")
    assert host.main("rehearsal", "--execute") == 2 and changes(host) == []
    assert "nightly eval starts in 20 min" in "\n".join(host.lines)


def test_rehearsal_skips_an_unexpected_plan(rig):
    host = rig()
    host.SETUPS = {**Host.SETUPS, "flywheel": ([TEACHER], [LOCAL, Q4])}   # would also stop Ollama
    assert host.main("rehearsal", "--execute") == 2
    assert host.applied == [] and not [a for a in host.calls if a[0] == "systemd-run"]
    assert "unexpected flywheel plan" in "\n".join(host.lines)


def test_rehearsal_teacher_never_healthy_restores_business(rig):
    host = rig()
    host.never_ready.add(TEACHER)
    assert host.main("rehearsal", "--execute") == 1
    assert host.applied == ["flywheel", "business"]
    assert host.units[LOCAL] == "active" and host.units[TEACHER] == "inactive"
    assert any("business restored" in n for n in host.notified)
    assert f"{DEADMAN}.timer" in host.timers                 # a failed rehearsal leaves the dead-man armed
    assert (host.win / "state" / "rehearsal.last").read_text().startswith("FAIL")


def test_rehearsal_dark_probe_with_the_teacher_restores(rig):
    host = rig()
    real = host._probe_line

    def line(at):
        if host.units[TEACHER] == "active" and host.serving():
            return real(at).replace("verdict=OK — 45.2", "verdict=DARK — 9.0")
        return real(at)

    host._probe_line = line
    assert host.main("rehearsal", "--execute") == 1
    assert host.applied == ["flywheel", "business"] and host.units[LOCAL] == "active"
    assert "voice probe after the change: DARK 9 tok/s" in "\n".join(host.lines)


def test_rehearsal_failed_restore_sets_the_no_swaps_flag(rig):
    host = rig()
    host.fail_apply.add("business")
    assert host.main("rehearsal", "--execute") == 1
    assert host.applied == ["flywheel", "business", "business", "business"]
    assert (host.win / M.FLAG_NAME).exists()
    assert "FAIL" in host.notified[-1] and f"{DEADMAN}.timer" in host.timers
    assert host.main("rehearsal", "--execute") == 2                 # the flag blocks the next swap
    assert "NO-SWAPS-TONIGHT is present" in "\n".join(host.lines)


def deadman_arms(host):
    """The --on-calendar times (epoch) of every dead-man systemd-run, in order."""
    out = []
    for a in host.calls:
        if a[0] == "systemd-run" and any(x.startswith("--on-calendar=") for x in a):
            cal = [x for x in a if x.startswith("--on-calendar=")][0].split("=", 1)[1]
            out.append(datetime.strptime(cal, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=ZoneInfo("UTC")).timestamp())
    return out


@pytest.mark.parametrize("left_at", [6 * 3600, 25 * 60])
def test_rehearsal_replaces_a_dead_man_armed_for_another_time(rig, left_at):
    # 25 min: a failed rehearsal's 45-min dead-man, 20 min later; it would fire mid-swap, without the lock.
    host = rig()
    t0 = host.t
    host.timers[f"{DEADMAN}.timer"] = {"sub": "waiting", "next": t0 + left_at}
    assert host.main("rehearsal", "--execute") == 0, "\n".join(host.lines)
    stop = ["systemctl", "--user", "stop", f"{DEADMAN}.timer"]
    first_run = next(i for i, a in enumerate(host.calls) if a[0] == "systemd-run")
    assert stop in host.calls[:first_run]
    assert deadman_arms(host) == [t0 + 45 * 60]
    assert host.applied == ["flywheel", "business"] and f"{DEADMAN}.timer" not in host.timers
    assert "replacing it" in "\n".join(host.lines)


def test_rehearsal_keeps_a_dead_man_armed_within_the_tolerance(rig):
    host = rig()
    host.timers[f"{DEADMAN}.timer"] = {"sub": "waiting", "next": host.t + 43 * 60}
    assert host.main("rehearsal", "--execute") == 0
    assert deadman_arms(host) == [] and "dead-man already armed" in "\n".join(host.lines)
    assert f"{DEADMAN}.timer" in host.timers           # not this run's: left as it was


def test_interrupt_with_the_swap_applied_restores(rig, monkeypatch):
    host = rig()
    real = M.Window.wait_probe_ok

    def interrupted(self, after, wait_s):
        if host.units[TEACHER] == "active":
            raise KeyboardInterrupt("signal 15")
        return real(self, after, wait_s)

    monkeypatch.setattr(M.Window, "wait_probe_ok", interrupted)
    assert host.main("rehearsal", "--execute") == 1
    assert host.applied == ["flywheel", "business"] and host.units[LOCAL] == "active"


# ── window start / end ──────────────────────────────────────────────────────


def test_window_start_q4_stops_ollama_and_arms_the_0715_dead_man(rig):
    host = rig()
    (host.win / "state").mkdir()
    (host.win / "state" / "rehearsal.last").write_text("PASS 2026-11-02 01:10 EST ok\n")
    assert host.main("window-start", "--execute", "--q4") == 0, "\n".join(host.lines)
    assert host.applied == ["flywheel-night-q4"]
    assert host.units[Q4] == "active" and host.units[OLLAMA] == "inactive" and host.units[LOCAL] == "inactive"
    assert host.timers[f"{DEADMAN}.timer"]["next"] == epoch("2026-11-02T07:15:00")
    assert Q4_ALIAS in host.notified[0]


def test_window_start_q4_refused_in_the_daytime_continuous_window(rig):
    host = rig(start="2026-10-31T14:00:00")
    host.conf.write_text(host.conf.read_text().replace("CONTINUOUS_UNTIL=", "CONTINUOUS_UNTIL=2026-11-01T18:00:00-05:00"))
    (host.win / "state").mkdir()
    (host.win / "state" / "rehearsal.last").write_text("PASS x\n")
    assert host.main("window-start", "--execute", "--q4") == 2 and host.applied == []
    assert host.main("window-start", "--execute") == 0                     # IQ3 in the continuous window
    assert host.timers[f"{DEADMAN}.timer"]["next"] == datetime.fromisoformat(
        "2026-11-01T18:15:00-05:00").timestamp()


def test_window_end_after_a_q4_night_restores_saluki_and_ollama(rig):
    host = rig(start="2026-11-02T06:40:00", layout="flywheel-night-q4")
    host.extra["distill-gen-1.service"] = "active"
    assert host.main("window-end", "--execute") == 0, "\n".join(host.lines)
    assert host.applied == ["business"]
    assert host.units[LOCAL] == "active" and host.units[OLLAMA] == "active" and host.units[Q4] == "inactive"
    assert host.extra["distill-gen-1.service"] == "inactive"
    assert host.notified[0].startswith("mini window end: PASS")
    assert "voice probe OK 45.2 tok/s" in host.notified[0]


def test_window_end_cancels_the_dead_man_after_a_verified_restore(rig):
    host = rig(start="2026-11-02T06:40:00", layout="flywheel-night-q4")
    host.timers[f"{DEADMAN}.timer"] = {"sub": "waiting", "next": epoch("2026-11-02T07:15:00")}
    enter_slot(host, "2026-11-02T05:50:00")
    assert host.main("window-end", "--execute") == 0
    assert f"{DEADMAN}.timer" not in host.timers
    assert not (host.win / "state" / "slot.active").exists()     # no eval slot outlives its window


def test_window_start_replaces_a_leftover_dead_man_in_the_continuous_window(rig):
    # A Q4 night's 07:15 dead-man left armed would restore business 15 min into a reopened IQ3 window.
    host = rig(start="2026-10-31T07:00:00")
    host.conf.write_text(host.conf.read_text().replace("CONTINUOUS_UNTIL=", "CONTINUOUS_UNTIL=2026-11-01T18:00:00-05:00"))
    (host.win / "state").mkdir()
    (host.win / "state" / "rehearsal.last").write_text("PASS x\n")
    host.timers[f"{DEADMAN}.timer"] = {"sub": "waiting", "next": epoch("2026-10-31T07:15:00")}
    assert host.main("window-start", "--execute") == 0, "\n".join(host.lines)
    assert host.timers[f"{DEADMAN}.timer"]["next"] == datetime.fromisoformat("2026-11-01T18:15:00-05:00").timestamp()


def test_window_end_nothing_to_do_when_business_is_healthy(rig):
    host = rig(start="2026-11-02T06:40:00")
    assert host.main("window-end", "--execute") == 0
    assert host.applied == [] and host.notified == []


def test_window_end_failed_restore_flags_and_notifies(rig):
    host = rig(start="2026-11-02T06:40:00", layout="flywheel")
    host.timers[f"{DEADMAN}.timer"] = {"sub": "waiting", "next": epoch("2026-11-02T07:15:00")}
    host.fail_apply.add("business")
    assert host.main("window-end", "--execute") == 1
    assert (host.win / M.FLAG_NAME).exists() and "RESTORE FAILED" in host.notified[-1]
    assert f"{DEADMAN}.timer" in host.timers                 # the dead-man retries


def test_window_end_leaves_a_running_baseline_alone(rig):
    host = rig(start="2026-11-02T06:40:00")
    host.extra["distill-baseline-c.service"] = "active"
    assert host.main("window-end", "--execute") == 2
    assert changes(host) == [] and host.extra["distill-baseline-c.service"] == "active"


# ── eval slots on a Q4 night ────────────────────────────────────────────────


def test_slot_enter_swaps_q4_for_iq3_and_ollama(rig):
    host = rig(start="2026-11-02T02:50:00", layout="flywheel-night-q4")
    host.extra["distill-gen-0120.service"] = "active"
    assert host.main("slot-enter", "--execute") == 0, "\n".join(host.lines)
    assert ["systemctl", "--user", "stop", "distill-gen-0120.service"] in host.calls
    assert host.applied == ["flywheel"]
    assert host.units[TEACHER] == "active" and host.units[OLLAMA] == "active" and host.units[Q4] == "inactive"
    assert host.t < epoch("2026-11-02T03:00:00")             # ready before the 03:00 eval
    assert (host.win / "state" / "slot.active").exists()


def test_slot_enter_is_a_noop_without_the_q4_teacher(rig):
    host = rig(start="2026-11-02T02:50:00", layout="flywheel")
    assert host.main("slot-enter", "--execute") == 0 and changes(host) == []
    host = rig(start="2026-11-02T02:50:00")
    assert host.main("slot-enter", "--execute") == 0 and changes(host) == []


def eval_process(host, until=None):
    """A nightly-eval process in the fake /proc, gone once the clock reaches ``until`` (None: never)."""
    pid = host.proc / "4242"
    pid.mkdir()
    (pid / "cmdline").write_bytes(b"/usr/bin/python3\0scripts/run_nightly_evals.py\0")
    if until is None:
        return
    real_sleep = host.sleep

    def sleep(s):
        real_sleep(s)
        if host.t >= epoch(until) and pid.exists():
            (pid / "cmdline").unlink()
            pid.rmdir()

    host.sleep = sleep


def test_q4_night_slot_enter_then_slot_exit_round_trip(rig):
    host = rig(start="2026-11-02T02:50:00", layout="flywheel-night-q4")
    assert host.main("slot-enter", "--execute") == 0, "\n".join(host.lines)
    entered = float((host.win / "state" / "slot.active").read_text().split()[0])
    assert epoch("2026-11-02T02:50:00") <= entered <= host.t
    host.sleep(epoch("2026-11-02T03:15:00") - host.t)
    assert host.main("slot-exit", "--execute") == 0, "\n".join(host.lines)
    assert host.applied == ["flywheel", "flywheel-night-q4"]
    assert host.units[Q4] == "active" and host.units[OLLAMA] == "inactive" and host.units[TEACHER] == "inactive"
    assert not (host.win / "state" / "slot.active").exists()
    fwd = forward_applies(host)
    assert all(a[-2:] == ["--attempts", "1"] for a in fwd) and len(fwd) == 2


def test_slot_exit_waits_for_the_eval_then_returns_to_q4(rig):
    host = rig(start="2026-11-02T03:15:00", layout="flywheel")
    enter_slot(host)
    eval_process(host, until="2026-11-02T03:21:00")
    assert host.main("slot-exit", "--execute") == 0, "\n".join(host.lines)
    assert host.applied == ["flywheel-night-q4"]
    assert host.units[Q4] == "active" and host.units[OLLAMA] == "inactive" and host.units[TEACHER] == "inactive"
    assert host.applied_at[0] >= epoch("2026-11-02T03:21:00")     # only after the eval process ended
    assert f"Ollama after the eval: {JUDGE} fully on the GPU" in "\n".join(host.lines)


@pytest.mark.parametrize("start, slot", [
    ("2026-11-02T03:15:00", None),                      # a plain IQ3 night: the same layout, no slot-enter
    ("2026-11-01T06:15:00", None),                      # the continuous IQ3 window (Sun morning)
    ("2026-11-02T00:30:00", "2026-11-01T05:50:00"),     # before the window opens; last night's record
    ("2026-11-02T03:15:00", "2026-11-01T02:50:00"),     # a leftover record from last night
])
def test_slot_exit_never_moves_a_plain_iq3_window_to_q4(rig, start, slot):
    host = rig(start=start, layout="flywheel")
    if slot:
        enter_slot(host, slot)
    assert host.main("slot-exit", "--execute") == 0
    assert changes(host) == [] and host.applied == []
    assert host.units[OLLAMA] == "active" and host.units[TEACHER] == "active" and host.units[Q4] == "inactive"
    assert "nothing to do" in "\n".join(host.lines)


def test_slot_exit_gives_up_waiting_for_a_long_eval(rig):
    host = rig(start="2026-11-02T03:15:00", layout="flywheel")
    enter_slot(host)
    eval_process(host)
    assert host.main("slot-exit", "--execute") == 0
    assert host.applied == [] and "staying on" in host.notified[0]
    assert host.t <= epoch("2026-11-02T03:15:00") + float(M.DEFAULTS["EVAL_WAIT_MAX_S"]) + 15


def test_slot_exit_eval_wait_ends_at_the_cutoff(rig):
    # Started 06:15 with the eval running to 06:38: the old wait ran on and swapped to Q4 at ~06:39.
    host = rig(start="2026-11-02T06:15:00", layout="flywheel")
    enter_slot(host, "2026-11-02T05:50:00")
    eval_process(host, until="2026-11-02T06:38:00")
    assert host.main("slot-exit", "--execute") == 0
    assert host.applied == [] and changes(host) == []
    assert host.t <= epoch("2026-11-02T06:20:15") and "still runs at 06:20" in host.notified[0]


def test_slot_exit_rechecks_the_cutoff_right_before_the_swap(rig):
    # The eval ends 06:18:30, but the next voice-probe tick is 06:20:40: past the cutoff, so no swap.
    host = rig(start="2026-11-02T06:15:00", layout="flywheel")
    host.probes.clear()
    host.probe_t0 = epoch("2026-11-02T06:10:40")
    host._tick_to(host.t)
    enter_slot(host, "2026-11-02T05:50:00")
    eval_process(host, until="2026-11-02T06:18:30")
    assert host.main("slot-exit", "--execute") == 0
    assert host.applied == [] and changes(host) == []
    assert "checked again right before the swap" in "\n".join(host.lines)
    assert host.units[TEACHER] == "active" and host.units[OLLAMA] == "active"


def test_slot_exit_after_0620_stays_on_iq3(rig):
    host = rig(start="2026-11-02T06:25:00", layout="flywheel")
    enter_slot(host, "2026-11-02T05:50:00")
    assert host.main("slot-exit", "--execute") == 0 and changes(host) == []
    assert "outside 0100-0620 ET" in "\n".join(host.lines)


def test_slot_exit_falls_back_to_iq3_when_q4_fails(rig):
    host = rig(start="2026-11-02T03:15:00", layout="flywheel")
    enter_slot(host)
    host.fail_apply.add("flywheel-night-q4")
    assert host.main("slot-exit", "--execute") == 1
    assert host.applied == ["flywheel-night-q4", "flywheel"]
    assert host.units[TEACHER] == "active" and host.units[OLLAMA] == "active"
    assert "staying on" in host.notified[-1]
    assert not (host.win / "state" / "slot.active").exists()   # given up on Q4 for the night
    host.sleep(epoch("2026-11-02T06:15:00") - host.t)
    assert host.main("slot-exit", "--execute") == 0 and host.applied == ["flywheel-night-q4", "flywheel"]


def test_slot_exit_does_not_fall_back_after_window_end_restored_business(rig):
    # window-end restored past the lock while the Q4 swap was failing: the fallback `apply flywheel`
    # would stop the local model server again after window-end verified it.
    host = rig(start="2026-11-02T03:15:00", layout="flywheel")
    enter_slot(host)
    real = host._setups

    def setups(name, execute):
        if execute and name == "flywheel-night-q4":
            host.applied.append(name)
            host.units.update({LOCAL: "active", OLLAMA: "active", TEACHER: "inactive", Q4: "inactive"})
            (host.win / "state" / "slot.active").unlink()
            return "setup FAILED\n", 1
        return real(name, execute)

    host._setups = setups
    assert host.main("slot-exit", "--execute") == 1
    assert host.applied == ["flywheel-night-q4"] and host.units[LOCAL] == "active"
    assert "nothing more was changed" in host.notified[-1]


def test_slot_exit_reports_a_judge_that_ran_partly_on_the_cpu(rig):
    host = rig(start="2026-11-02T03:15:00", layout="flywheel")
    enter_slot(host)
    host.ollama_ps = (5828081664, 5500000000)
    assert host.main("slot-exit", "--execute") == 0
    assert any("PARTLY ON THE CPU" in n for n in host.notified) and host.applied == ["flywheel-night-q4"]


# ── a teacher that fails mid-window (OnFailure=) ────────────────────────────


def test_teacher_failed_mid_window_restores_business_and_sets_the_flag(rig):
    host = rig(start="2026-10-31T08:00:00", layout="flywheel")
    host.units[TEACHER] = "failed"                         # crashed; systemd gave up restarting it
    assert host.main("teacher-failed", "--execute") == 0, "\n".join(host.lines)
    assert host.applied == ["business"] and host.units[LOCAL] == "active"
    assert (host.win / M.FLAG_NAME).exists()
    assert host.notified[-1].startswith(f"mini: {TEACHER} FAILED mid-window; business restored")
    assert (host.win / "state" / "window.ended").read_text().startswith("PASS")


@pytest.mark.parametrize("layout, failed", [
    ("flywheel", Q4),           # slot-exit's Q4 swap failed and it fell back: the IQ3 teacher serves
    ("business", TEACHER),      # a window script already restored business
    ("flywheel", None),         # nothing failed
])
def test_teacher_failed_changes_nothing_when_the_port_is_served(rig, layout, failed):
    host = rig(start="2026-11-02T03:20:00", layout=layout)
    if failed:
        host.units[failed] = "failed"
    assert host.main("teacher-failed", "--execute") == 0
    assert changes(host) == [] and host.notified == [] and not (host.win / M.FLAG_NAME).exists()


def test_status_is_read_only(rig):
    host = rig(layout="flywheel")
    assert host.main("status") == 0 and changes(host) == []
    text = "\n".join(host.lines)
    assert IQ3_ALIAS in text and "dead-man: none" in text


# ── the committed unit templates ────────────────────────────────────────────


@pytest.mark.parametrize("name, ctx, alias, model", [
    ("distill-teacher.service", 57344, IQ3_ALIAS, "%h/distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf"),
    ("distill-teacher-q4.service", 24576, Q4_ALIAS, "%h/distill/models/teacher-q4/Qwen3.8-27B-Q4_K_M.gguf"),
])
def test_unit_templates_have_the_local_model_servers_shape(name, ctx, alias, model):
    text = (REPO / "scripts" / "distill" / "units" / name).read_text()
    assert not re.search(r"^\[Install\]", text, re.M)   # never starts at boot
    assert f"Environment=CTX={ctx}\n" in text and ctx % 4096 == 0 and ctx <= 65536
    flags = " ".join(line.strip().rstrip("\\") for line in text.splitlines() if not line.startswith("#"))
    for flag in ("--host 127.0.0.1 --port 8092", "--no-mmproj", f"--alias {alias}", "-ngl 99", "--fit off",
                 "-c ${CTX} --parallel 2 --kv-unified", "--flash-attn on -ctk q8_0 -ctv q8_0", "--jinja",
                 "--batch-size 2048 --ubatch-size 512", "--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0",
                 '--reasoning-budget 2048 --reasoning-budget-message "[REASONING_BUDGET_EXHAUSTED]"',
                 f"-m {model}", "--no-slots"):
        assert flag in flags, flag
    assert re.search(r"^Conflicts=distill-teacher(-q4)?\.service$", text, re.M)
    assert "/home/" not in text
    # a teacher systemd gave up on restores the local model server at once, not at the window end
    assert re.search(r"^OnFailure=window-mini-teacher-failed\.service$", text, re.M)


def test_vram_math_in_the_templates():
    mib = 1 << 20
    kv_per_token = 34816                     # 16 attention layers x 4 KV heads x 256 dims x 2 x q8_0 (34/32 B)
    assert kv_per_token == 16 * 4 * 256 * 2 * 34 // 32
    rs, overhead = 2 * 156893184 / mib, 1191.9
    free_iq3, free_q4 = 13265, 13265 + 5386

    def fits(weights_b, free):
        best = None
        for ctx in range(4096, 65536 + 1, 4096):
            total = weights_b / mib + rs + overhead + ctx * kv_per_token / mib
            if free - total >= 600:
                best = (ctx, round(total, 1), round(free - total, 1))
        return best

    assert fits(9676118016, free_iq3) == (57344, 12623.0, 642.0)
    assert fits(16477259776, free_q4) == (24576, 18021.1, 629.9)


def test_the_teacher_failure_handler_unit():
    name = "window-mini-teacher-failed.service"
    text = (REPO / "scripts" / "distill" / "units" / name).read_text()
    assert not name.startswith("distill-")        # apply business stops every running distill-* unit
    assert not re.search(r"^\[Install\]", text, re.M) and "/home/" not in text
    assert re.search(r"^Type=oneshot$", text, re.M) and re.search(r"^KillMode=mixed$", text, re.M)
    # longer than the lock wait plus a restore: a oneshot's start times out after 90 s by default
    start = int(re.search(r"^TimeoutStartSec=(\d+)min$", text, re.M).group(1))
    assert start * 60 > float(M.DEFAULTS["FAILED_LOCK_WAIT_S"]) + 3 * 600
    assert int(re.search(r"^TimeoutStopSec=(\d+)min$", text, re.M).group(1)) >= 20
    exec_start = re.search(r"^ExecStart=(.+)$", text, re.M).group(1)
    assert exec_start.endswith("mini_window.py --conf %h/distill/window/window.conf teacher-failed --execute")
    installer = (REPO / "scripts" / "distill" / "units" / "install_teacher_units.sh").read_text()
    assert name in re.search(r"^teacher_units=\((.+)\)$", installer, re.M).group(1)


@pytest.mark.parametrize("wrapper", ["mini_rehearsal.sh", "mini_window_start.sh", "mini_window_end.sh",
                                     "eval_slot_enter.sh", "eval_slot_exit.sh"])
def test_wrappers_document_a_stop_timeout_the_restore_fits_in(wrapper):
    # The default TimeoutStopSec (90 s) SIGKILLs the restore a `systemctl stop`'s SIGTERM starts.
    text = (REPO / "scripts" / "distill" / "mini_window" / wrapper).read_text()
    [line] = [x for x in text.splitlines() if "systemd-run --user" in x]
    assert "--unit=window-mini-" in line and "-p TimeoutStopSec=20min -p KillMode=mixed" in line


def test_stage_drops_the_swap_units_from_never_stop_in_the_window_copy_only(tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "distill"))
    import setups as S

    shared = tmp_path / "hosts.yaml"
    shared.write_text(
        "hosts:\n"
        "  tower: {ssh: t, never_stop: [model.service]}\n"
        "  mini:\n"
        "    ssh: m\n"
        "    never_stop: [local-model.service, ollama.service, tts.service]\n"
        "    notify_cmd: /opt/notify\n"
        "never_stop: [site.service]\n"
        "business_units:\n"
        "  mini: [ollama.service]\n")
    swap = [LOCAL, OLLAMA]
    copy = tmp_path / "copy.yaml"
    copy.write_text(M.window_hosts_copy(S, shared, "mini", swap))
    hosts = S.load_hosts(copy)
    assert list(hosts.hosts) == ["mini"] and hosts.business_units == {"mini": (OLLAMA,)}
    g = hosts.guard("mini")
    assert not S.is_never_stop(LOCAL, g) and not S.is_never_stop(OLLAMA, g)
    assert all(S.is_never_stop(u, g) for u in ("tts.service", "site.service", "prometheus.service"))
    # the shared file itself stays guarded: a repo-machine apply still refuses to stop them
    assert all(S.is_never_stop(u, S.load_hosts(shared).guard("mini")) for u in swap)
    # a shared file that no longer guards them is refused (it would let any Mac-side run stop them)
    shared.write_text(shared.read_text().replace("local-model.service, ", ""))
    with pytest.raises(SystemExit, match="must stay never-stop"):
        M.window_hosts_copy(S, shared, "mini", swap)



# ── generation (Track C): window-start and slot-exit start distill-gen-<stamp> ──────────────────


def stage_gen(host):
    runner = host.win / "gen_runner.sh"
    runner.write_text("#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    return runner


def rehearsed(host):
    (host.win / "state").mkdir(exist_ok=True)
    (host.win / "state" / "rehearsal.last").write_text("PASS 2026-11-02 01:10 EST ok\n")


def test_window_start_starts_generation_after_the_verified_swap(rig):
    host = rig()
    runner = stage_gen(host)
    rehearsed(host)
    assert host.main("window-start", "--execute") == 0, "\n".join(host.lines)
    (run,) = host.gen_runs
    unit = [x for x in run if x.startswith("--unit=")][0].split("=", 1)[1]
    assert re.fullmatch(r"distill-gen-\d{8}-\d{6}", unit)              # a restore stops distill-* units
    assert run[-3:] == [str(runner), "run", "--execute"] and "--collect" in run
    assert "KillMode=mixed" in run and "TimeoutStopSec=90" in run
    # only after the swap was verified (the teacher up, the probe OK)
    i_gen = host.calls.index(run)
    i_swap = max(i for i, a in enumerate(host.calls) if len(a) > 1 and a[1].endswith("/setups.py") and "--execute" in a)
    assert i_gen > i_swap and host.units[TEACHER] == "active"
    assert f"generation: {unit} started" in host.notified[0]


def test_window_start_dry_run_only_prints_the_generation_start(rig):
    host = rig()
    stage_gen(host)
    assert host.main("window-start") == 0
    assert host.gen_runs == [] and changes(host) == []
    assert "[dry run] would run: systemd-run --user --unit=distill-gen-" in "\n".join(host.lines)


def test_generation_not_staged_off_or_running_is_not_started(rig):
    host = rig()
    rehearsed(host)
    assert host.main("window-start", "--execute") == 0                  # no runner staged
    assert host.gen_runs == [] and "generation: not staged" in host.notified[0]
    host = rig()
    stage_gen(host)
    rehearsed(host)
    host.conf.write_text(host.conf.read_text() + "GEN_ENABLE=0\n")
    assert host.main("window-start", "--execute") == 0
    assert host.gen_runs == [] and "generation: off (GEN_ENABLE)" in host.notified[0]
    host = rig()
    stage_gen(host)
    rehearsed(host)
    host.extra["distill-gen-20261102-0120.service"] = "active"
    assert host.main("window-start", "--execute") == 0
    assert host.gen_runs == [] and "already running (distill-gen-20261102-0120.service)" in host.notified[0]


def test_a_failed_generation_start_never_fails_the_window(rig):
    host = rig()
    stage_gen(host)
    rehearsed(host)
    host.fail_gen = True
    assert host.main("window-start", "--execute") == 0
    assert host.units[TEACHER] == "active" and host.applied == ["flywheel"]   # the window stays open
    assert "generation: FAILED to start distill-gen-" in host.notified[0]


def test_q4_night_eval_slot_stops_generation_and_slot_exit_restarts_it(rig):
    host = rig(start="2026-11-02T02:50:00", layout="flywheel-night-q4")
    stage_gen(host)
    host.extra["distill-gen-20261102-0120.service"] = "active"
    assert host.main("slot-enter", "--execute") == 0, "\n".join(host.lines)
    assert host.extra["distill-gen-20261102-0120.service"] == "inactive" and host.gen_runs == []
    host.sleep(epoch("2026-11-02T03:15:00") - host.t)
    assert host.main("slot-exit", "--execute") == 0, "\n".join(host.lines)
    assert host.units[Q4] == "active" and len(host.gen_runs) == 1
    assert "generation: distill-gen-" in "\n".join(host.lines)


def test_stage_ships_the_generation_runner(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(M, "load_conf", lambda path, home=None: {**M.DEFAULTS, "LOCAL_MODEL_UNIT": LOCAL,
                                                                 "PROBE_UNIT": "p.service", "OLLAMA_MODEL": JUDGE})
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "distill"))
    import setups as S  # noqa: PLC0415
    monkeypatch.setattr(S, "load_setup", lambda d, n: {"name": n, "hosts": {"mini": {"services": []}}})
    monkeypatch.setattr(S, "dump_setup", lambda s: "x: 1\n")
    monkeypatch.setattr(M, "window_hosts_copy", lambda *a: "hosts: {}\n")
    monkeypatch.setattr(sys, "argv", ["mini_window"])
    conf = tmp_path / "w.conf"
    conf.write_text("x")
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0))
    out = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: out.append(" ".join(map(str, a))))
    assert M.main(["--conf", str(conf), "stage", "--ssh", "host"]) == 0
    text = "\n".join(out)
    assert "gen_runner.py" in text and "gen_runner.sh" in text and calls == []
