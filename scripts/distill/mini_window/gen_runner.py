#!/usr/bin/env python3
"""Teacher generation during a 3090 Ti flywheel window (Track C; DISTILL-OPEN-QUESTIONS #10, Q6b, PAYG).

    gen_runner.py run      [--execute] [--until ISO|HH:MM]   # mini_window.py starts it as distill-gen-<stamp>
    gen_runner.py harvest  [--execute]                       # new verified teacher rows -> the pool
    gen_runner.py status                                     # read-only
    gen_runner.py stage-code --ssh TARGET [--execute]        # (repo machine) HEAD export -> GEN_CODE

``run`` is a dry run unless ``--execute``: the read-only checks run and the plan (every command) is printed.
Runs ON the 3090 Ti host from ``~/distill/window/`` (``gen_runner.sh`` beside ``mini_window.py``), stdlib
only, settings from the window's ``window.conf`` (``GEN_*`` keys, defaults in ``GEN_DEFAULTS``).

What one run does, under one flock (``<runs>/state/gen.lock``), until ``--until``:

a. a second probe-guard instance (``~/distill/phase4/probe_guard.py``, its own CLI) listening on
   ``GEN_GUARD_LISTEN`` (127.0.0.1:18593) in front of the LOCAL teacher on ``MODEL_URL`` (OQ10: the teacher
   holds the daemon's local model port during a window). New completions start only just after a voice-probe
   result; a bad probe trips its abort switch (``<runs>/state/guard.abort``) for the rest of its life.
b. ``teacher_gen.py run --teacher local --tasks <the real-pattern taskset>`` through that guard,
   concurrency 1 (the daemon shares the server), into ``<runs>/local-<served alias>/`` (teacher_gen
   refuses to mix two teachers in one output, so the IQ3 and Q4 teachers each have their own). The
   taskset (``GEN_TASKS``, default ``real-v1-excl.jsonl``) must hold no row built from a prompt of a
   session on ``GEN_EXCLUDE_SESSIONS`` (``taskset_gen.py exclude --check``; OPEN-QUESTIONS #6): a gym
   row has no session id to drop it by later, so the preflight refuses such a taskset. Each teacher_gen
   run has its own ``--sandbox-root`` (``GEN_SANDBOX_ROOT``-local / -max-payg): teacher_gen wipes its
   workers' workspaces at every task start and end, and holds a lock on its root.
c. the Max branch, on one of two routes (keys checked for presence only; no value is read here, and nothing
   here passes a key anywhere):
   * pay-per-token, whenever ``QWEN_PAYG_KEY`` is present (environment or the 0600 env file) AND ``spend.py
     preflight`` passes: ``judge_run.sh --execute --provider payg --budget-usd 25`` (Phase 2 sample on Max,
     the one spend ledger) FIRST, then ``teacher_gen.py run --teacher max-payg --budget-usd 100`` into
     ``<runs>/max-payg-max-derived/``;
   * else the Token Plan (``QWEN_API_KEY``; the operator's informed override of 2026-10-31; ``GEN_TOKEN_PLAN``
     ``off`` disables it) when ``spend.py tokenplan-preflight`` passes: ``judge_run.sh --execute --provider
     token-plan`` (8 workers; its first 50 calls are the trial ``spend.py tokenplan-status`` reports) FIRST,
     then ``teacher_gen.py run --teacher max-token-plan --concurrency 8 --tp-start-concurrency 4`` into
     ``<runs>/max-tokenplan-max-derived/``. $0 per call, no $ budget, no pacing: spend.py's guardrails
     (adaptive concurrency, STOP on an account error or a key change, pause on an exhausted quota).
     ``QWEN_PAYG_KEY`` appearing mid-run stops the Token Plan children at the next poll and the branch starts
     over on the pay-per-token route (the judge resumes from its results). The quota pause
     (``GEN_TP_PAUSE``, written by spend.py; a judge exit 4 or a teacher exit 7) parks the branch in state
     ``quota`` until its reset time, then the branch starts over (the first call is spend.py's one probe).
   Both run concurrently with (b), niced (the API is the bottleneck; the gym's sandbox commands run on this
   CPU). The Max teacher's local judge goes through the same guard. No key, or a refused preflight: logged,
   skipped, never a failure.
d. ``harvest`` every ``GEN_HARVEST_EVERY_MIN`` and at the end: the rows of every finished task (ledger
   event kept or released) in ``<runs>/*/*.traces.jsonl`` not yet copied go to ``<pool>/<source>.<run>.
   traces.jsonl``; Max-derived rows (``artifact_policy.row_verdict``; unknown provenance counts as Max,
   fail closed) go to ``<pool>/max-derived/``. ``<pool>/harvest.json`` holds the counts per source.
   Never the relaxed or held files.

Stops (children by process group, SIGTERM then SIGKILL after ``GEN_CHILD_GRACE_S``; the guard by its PID):
* SIGTERM/SIGINT/SIGHUP (a ``setups.py apply business`` restore stops every distill-* unit; slot-enter
  stops distill-gen-* before a Q4 night's eval), ``--until``;
* the window's NO-SWAPS flag, or ``<runs>/STOP`` (the operator's stop for generation): everything stops;
* the port no longer serving the teacher it served at the start (``GEN_PORT_GONE_CHECKS`` polls in a row):
  the window closed or failed;
* the spend STOP file (``~/distill/spend-ledger.STOP``: the $140 alert, or a Token Plan account error or key
  change): the paid branch stops at once and is not restarted; the local branch goes on ($0). The Max teacher's own exit 4 (the ledger refused a
  reservation: its ``GEN_MAX_BUDGET_USD`` phase budget, a total across runs, or the cap) ends the branch as
  done, noted, never a failure; its exit 5 (the key, host, account or billing refused) stops it, noted;
* the guard's abort switch or the guard gone: the local branch (and the Max run's local judge) stop for
  this run;
* a nightly eval running, one due within ``GEN_EVAL_PAUSE_BEFORE_MIN``, or one whose start time passed
  less than ``GEN_EVAL_PAUSE_AFTER_MIN`` ago (cron starts late): the local branch pauses (teacher_gen
  cancels the task in flight; it reruns later) and resumes once the eval is done; so does the Max
  teacher while its judge is the local teacher (``GEN_MAX_JUDGE=guard``: its judge requests would reach
  the server the eval scores; its paid calls in flight settle first, its tasks in flight rerun). The
  Max judge (``judge_run.sh``) calls only the paid API and runs on. On a Q4 night slot-enter stops this
  unit before the eval instead, and slot-exit starts a new one.

Resume: teacher_gen and judge.py resume from their own ledgers; the harvest from ``<pool>/.harvest-state.json``
(written after every pool file's append; a pool file longer than recorded is cut back to its recorded
size and one the state never recorded is removed, then their rows are copied again, so an interrupted
harvest neither copies a row twice nor leaves a torn line). A local teacher_gen that exits 3 (teacher
unreachable / errors in a row) or crashes (exit 6, or exit 1 without a report from that run: Python's
own code for an uncaught exception) is restarted after ``GEN_RESTART_WAIT_S``, at most
``GEN_LOCAL_RESTARTS`` times; a crashed Max teacher fails the branch.

Outputs: ``<runs>/state/progress.json`` (rewritten on every change), ``<runs>/logs/`` (this runner's log, the
guard's, one log per child), the run directories, ``<pool>``. A notify (<= 5 lines) at the end.

Exit: 0 finished (until, or nothing left to do), 1 a branch failed, 2 refused by a check (nothing started)
or another runner holds the lock, 3 stopped by NO-SWAPS / STOP / the window closing, 143 a signal.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mini_window as MW  # noqa: E402 - the window's own settings loader and helpers, beside this file

GEN_DEFAULTS: dict[str, str] = {
    "GEN_CODE": "~/distill/phase2b/code-head",       # HEAD export: src/, scripts/distill/, COMMIT
    "GEN_PY": "~/prometheus-deploy/.venv/bin/python",  # holds Prometheus's dependencies; src from GEN_CODE
    "GEN_RUNS": "~/distill/phase2b/runs",
    "GEN_POOL": "~/distill/phase2b/pool",
    # the real-pattern taskset with the excluded sessions' rows dropped (taskset_gen.py exclude); the
    # preflight refuses a taskset that still holds one (GEN_EXCLUDE_SESSIONS: the operator's list)
    "GEN_TASKS": "~/distill/phase2b/tasksets/real-v1-excl.jsonl",
    "GEN_EXCLUDE_SESSIONS": "~/distill/policy/exclude-sessions.txt",
    "GEN_EXCLUDE_CHECK_TIMEOUT_S": "900",
    # each teacher_gen run its own sandbox root (<root>-local, <root>-max-payg): a shared one would have the
    # two runs wipe each other's task workspaces. Empty: <tmp>/prometheus-teacher-gen
    "GEN_SANDBOX_ROOT": "",
    "GEN_INJECTION_MANIFEST": "~/distill/phase3/evalsets/eval-injection.manifest.json",
    "GEN_EVAL_TOOLS": "~/distill/phase3/build-v2/eval-tools.jsonl ~/distill/phase3/evalsets/ungated/eval-longctx-tools.jsonl",
    "GEN_PACK": "~/distill/phase3/request_pack.json",
    "GEN_REPO": "~/Prometheus",                       # git-tracked docs for force_compaction padding
    "GEN_GUARD_PY": "~/distill/phase4/probe_guard.py",
    "GEN_GUARD_LISTEN": "127.0.0.1:18593",            # the baseline's guard is on 18592
    "GEN_GUARD_MIN_TPS": "",                          # empty: PROBE_MIN_TPS
    "GEN_GUARD_WAIT_S": "20",
    "GEN_LOCAL_CONCURRENCY": "1",
    "GEN_STOP_HHMM": "0635",                          # a nightly window's generation ends here
    "GEN_END_MARGIN_MIN": "10",                       # before CONTINUOUS_UNTIL in the continuous window
    "GEN_NICE_CMD": "nice -n 10",
    "GEN_MAX_NICE_CMD": "nice -n 15 ionice -c3",
    "GEN_JUDGE_BUDGET_USD": "25",
    "GEN_JUDGE_BLIND_BUDGET_USD": "5",
    "GEN_MAX_BUDGET_USD": "100",
    "GEN_MAX_CONCURRENCY": "2",
    "GEN_MAX_MODEL": "qwen3.8-max",
    "GEN_MAX_JUDGE": "guard",                         # guard: the Max run's judge is the local teacher
    "GEN_PAYG_KEY_NAME": "QWEN_PAYG_KEY",
    "GEN_SPEND_STOP": "~/distill/spend-ledger.STOP",
    # the Token Plan route (operator override 2026-10-31), used only while GEN_PAYG_KEY_NAME is absent
    "GEN_TOKEN_PLAN": "on",                           # off: never the Token Plan
    "GEN_TP_KEY_NAME": "QWEN_API_KEY",
    "GEN_TP_CONCURRENCY": "8",                        # teacher tasks in flight (the adaptive limit's ceiling)
    "GEN_TP_START_CONCURRENCY": "4",                  # the adaptive limit's start
    "GEN_TP_JUDGE_CONCURRENCY": "8",
    "GEN_TP_PAUSE": "~/distill/token-plan.PAUSE",     # spend.py's quota pause (read here, never written)
    # a Token Plan child stopped by the quota with no pause standing (another process's probe in flight):
    # the branch waits this long before it starts over, so it never spins
    "GEN_TP_QUOTA_RECHECK_S": "300",
    "GEN_EVAL_PAUSE_BEFORE_MIN": "5",
    "GEN_EVAL_PAUSE_AFTER_MIN": "3",                  # ... and this long after its start time, until seen
    "GEN_LOCAL_RESTARTS": "3",
    "GEN_RESTART_WAIT_S": "120",
    "GEN_HARVEST_EVERY_MIN": "30",
    "GEN_POLL_S": "5",
    "GEN_CHILD_GRACE_S": "20",
    "GEN_PORT_GONE_CHECKS": "12",                     # polls in a row (~1 min): a hiccup is not a closed window
    "GEN_MAX_HOURS": "40",                            # refuse an --until further out than this
    "GEN_MIN_RUN_S": "600",                           # refuse an --until closer than this
    "GEN_MIN_START_S": "180",                         # no child (re)started closer to --until than this
}
SCHEMA = "distill-gen-runner/1"
POOL_SCHEMA = "distill-gen-pool/1"
FINAL = ("done", "failed", "stopped", "off")
JUDGE_EXIT = {0: "ok", 3: "budget (cap, sub-budget or STOP)", 4: "paused (rate limit)", 5: "fatal",
              6: "items remain", 7: "busy"}
#: teacher_gen.py --teacher max-payg exits: the spend ledger refused a reservation (STOP, the cap,
#: --budget-usd), and the key / host / account refused (before any call or mid-run)
TG_EXIT_SPEND_STOP, TG_EXIT_PAYG_REFUSED = 4, 5
#: teacher_gen.py's exit for an unexpected exception (Python's own exit 1 would read as "kept nothing")
TG_EXIT_CRASH = 6
#: teacher_gen.py --teacher max-token-plan: the Token Plan's quota pause stopped it
TG_EXIT_QUOTA_PAUSED = 7
#: teacher_gen.py's default --name for each branch (the files a run writes are <out>/<name>.*)
LOCAL_NAME, PAID_NAME = "qwen38-27b-gym", "max-payg-gym"
#: the Max branch's two routes: teacher_gen --teacher, its --name, its output directory
ROUTE_PAYG, ROUTE_TP = "payg", "token-plan"
ROUTES = {ROUTE_PAYG: ("max-payg", PAID_NAME, "max-payg-max-derived"),
          ROUTE_TP: ("max-token-plan", "max-tokenplan-gym", "max-tokenplan-max-derived")}


class Skip(Exception):
    """A check refused the run: nothing was started (exit 2)."""


class Fail(Exception):
    """Something broke after the start (exit 1)."""


# ── settings ─────────────────────────────────────────────────────────────────


def load(conf_path: Path) -> dict[str, str]:
    """window.conf (mini_window.load_conf: its defaults, ``~``/``$HOME``) plus the GEN_* defaults."""
    c = MW.load_conf(conf_path)
    home = os.path.expanduser("~")
    for k, v in GEN_DEFAULTS.items():
        if k not in c:
            c[k] = home + v[1:] if v.startswith("~/") else v
    c["GEN_EVAL_TOOLS"] = " ".join(os.path.expanduser(p) for p in c["GEN_EVAL_TOOLS"].split())
    return c


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "unknown"


def host_port(url: str) -> str:
    m = re.match(r"^(?:https?://)?([^/]+)", url.strip())
    return m.group(1) if m else url


def port_free(listen: str) -> bool:
    host, _, port = listen.rpartition(":")
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host or "127.0.0.1", int(port))) != 0


def key_present(name: str, env: dict[str, str]) -> bool:
    """Whether ``name`` is set in the environment or in Prometheus's env file (spend.py's own lookup). Only
    the presence of a non-empty ``NAME=`` line is checked; the value is never kept or shown."""
    if env.get(name, "").strip():
        return True
    if env.get("PROMETHEUS_ENV_FILE"):
        path = Path(env["PROMETHEUS_ENV_FILE"]).expanduser()
    else:
        xdg = env.get("XDG_CONFIG_HOME")
        path = (Path(xdg).expanduser() if xdg else Path.home() / ".config") / "prometheus" / "env"
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=\s*['\"]?\S")
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return any(pat.match(line) for line in fh)
    except OSError:
        return False


# ── children ─────────────────────────────────────────────────────────────────


@dataclass
class Child:
    """One process group this runner started and stops (never anything else)."""

    name: str
    argv: list[str]
    log_path: Path
    env: dict[str, str]
    cwd: Path
    proc: subprocess.Popen[bytes] | None = None
    started_at: float = 0.0

    def start(self, pid_dir: Path | None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as lf:
            self.proc = subprocess.Popen(self.argv, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                         cwd=self.cwd, env=self.env, start_new_session=True)
        self.started_at = time.time()
        if pid_dir is not None:
            (pid_dir / f"{self.name}.pid").write_text(f"{self.proc.pid}\n")

    def poll(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace: float, pid_dir: Path | None) -> int | None:
        p = self.proc
        if p is None:
            return None
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(p.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    p.wait(timeout=10)
            # whatever the leader left in its group (a sandbox command) goes too: only now, right after
            # this stop, can the group id not belong to anyone else
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGKILL)
        if pid_dir is not None:
            with contextlib.suppress(OSError):
                (pid_dir / f"{self.name}.pid").unlink()
        return p.returncode


@dataclass
class Branch:
    state: str = "pending"
    #: the Max branch's route (ROUTE_PAYG / ROUTE_TP) once chosen; "" before
    mode: str = ""
    detail: str = ""
    child: Child | None = None
    restarts: int = 0
    resume_at: float = 0.0
    runs: int = 0
    rcs: list[int | None] = field(default_factory=list)


# ── the runner ───────────────────────────────────────────────────────────────


class Gen:
    def __init__(self, conf: dict[str, str], execute: bool, *, out: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.time, env: dict[str, str] | None = None) -> None:
        self.c = conf
        self.execute = execute
        self.clock = clock
        self.env = dict(os.environ if env is None else env)
        self.tz = ZoneInfo(conf["ET_TZ"])
        self.runs = Path(conf["GEN_RUNS"])
        self.pool = Path(conf["GEN_POOL"])
        self.code = Path(conf["GEN_CODE"])
        self.distill = self.code / "scripts" / "distill"
        self.state_dir = self.runs / "state"
        self.log_dir = self.runs / "logs"
        self.flag = Path(conf["WIN_DIR"]) / MW.FLAG_NAME
        self.gen_stop = self.runs / "STOP"
        self.spend_stop = Path(conf["GEN_SPEND_STOP"])
        self.tp_pause = Path(conf["GEN_TP_PAUSE"])
        self.abort_file = self.state_dir / "guard.abort"
        self.guard_url = "http://" + conf["GEN_GUARD_LISTEN"]
        self._out = out or (lambda s: print(s, flush=True))
        self.log_file: Path | None = None
        self.stop: str | None = None
        self.local = Branch()
        self.paid = Branch()
        self.guard: Child | None = None
        self.alias: str | None = None
        self.help_text = ""
        self.harvested: dict[str, int] = {}
        self.port_misses = 0
        self.notes: list[str] = []
        self.progress: dict[str, Any] = {"schema": SCHEMA, "pid": os.getpid(), "state": "starting",
                                         "started_at": self.iso(self.now())}
        # mini_window's helpers (loopback HTTP without a proxy, the nightly-eval checks), read-only use
        self.w = MW.Window(conf, False, "gen", clock=clock, out=lambda _s: None)

    # basics

    def now(self) -> float:
        return self.clock()

    def iso(self, t: float) -> str:
        return datetime.fromtimestamp(t, self.tz).isoformat(timespec="seconds")

    def log(self, msg: str) -> None:
        line = f"{datetime.fromtimestamp(self.now(), self.tz).strftime('%F %T %Z')} gen: {msg}"
        self._out(line)
        if self.log_file is not None:
            with contextlib.suppress(OSError), open(self.log_file, "a") as f:
                f.write(line + "\n")

    def note(self, msg: str) -> None:
        self.log("NOTE " + msg)
        self.notes.append(f"{datetime.fromtimestamp(self.now(), self.tz).strftime('%H:%M')} {msg}")

    def notify(self, lines: list[str]) -> None:
        msg = "\n".join(x for x in lines if x)[:1500]
        msg = "\n".join(msg.splitlines()[:5]) + "\n"
        if not self.execute:
            self.log("[dry run] would notify: " + " | ".join(msg.strip().splitlines()))
            return
        try:
            cp = subprocess.run([self.c["NOTIFY"], "5"], input=msg, text=True, timeout=60,
                                capture_output=True)
            self.log("notify " + ("sent" if cp.returncode == 0 else f"FAILED (exit {cp.returncode})"))
        except (OSError, subprocess.SubprocessError) as e:
            self.log(f"notify FAILED: {type(e).__name__}")

    def save(self) -> None:
        if not self.execute:
            return
        p = self.progress
        p.update(updated_at=self.iso(self.now()), notes=self.notes[-30:], harvested_this_run=self.harvested,
                 local={"state": self.local.state, "detail": self.local.detail, "runs": self.local.runs,
                        "restarts": self.local.restarts, "exits": self.local.rcs,
                        "out": str(self.local_out()) if self.alias else None},
                 paid={"state": self.paid.state, "detail": self.paid.detail, "exits": self.paid.rcs,
                       "route": self.paid.mode or None},
                 guard={"listen": self.c["GEN_GUARD_LISTEN"], "running": bool(self.guard and self.guard.running),
                        "aborted": self.abort_file.exists()})
        tmp = self.state_dir / f".progress.{os.getpid()}"
        tmp.write_text(json.dumps(p, indent=1) + "\n")
        os.replace(tmp, self.state_dir / "progress.json")

    def sleep(self, s: float) -> None:
        end = time.monotonic() + max(0.0, s)
        while not self.stop and time.monotonic() < end:
            time.sleep(min(0.25, max(0.0, end - time.monotonic())))

    def http_json(self, url: str, timeout: float = 5.0) -> tuple[int, Any]:
        return self.w.http(url, timeout)

    def served_ids(self) -> tuple[int, list[str]]:
        base = self.c["MODEL_URL"].rstrip("/")
        st, body = self.http_json(base + "/v1/models")
        ids: list[str] = []
        if st == 200 and isinstance(body, dict):
            for m in body.get("data") or []:
                if isinstance(m, dict):
                    ids += [str(m.get("id") or "")] + [str(a) for a in m.get("aliases") or []]
        return st, [i for i in ids if i]

    def teacher_alias(self) -> str | None:
        _st, ids = self.served_ids()
        for alias in (self.c["TEACHER_ALIAS"], self.c["TEACHER_Q4_ALIAS"]):
            if alias in ids:
                return alias
        return None

    # time

    def resolve_until(self, text: str | None) -> float:
        now = self.now()
        et_now = datetime.fromtimestamp(now, self.tz)
        if text:
            if re.fullmatch(r"\d{1,2}:\d{2}", text.strip()):
                h, m = (int(x) for x in text.strip().split(":"))
                at = et_now.replace(hour=h, minute=m, second=0, microsecond=0)
                if at.timestamp() <= now:
                    at += timedelta(days=1)
                t = at.timestamp()
            else:
                dt = datetime.fromisoformat(text.strip())
                t = (dt if dt.tzinfo else dt.replace(tzinfo=self.tz)).timestamp()
        else:
            cont = self.c.get("CONTINUOUS_UNTIL", "").strip()
            cu = None
            if cont:
                dt = datetime.fromisoformat(cont)
                cu = (dt if dt.tzinfo else dt.replace(tzinfo=self.tz)).timestamp()
            if cu is not None and now < cu:
                t = cu - 60 * float(self.c["GEN_END_MARGIN_MIN"])
            else:
                h, m = MW.hhmm(self.c["GEN_STOP_HHMM"])
                oh, om = MW.hhmm(self.c["WINDOW_OPEN_HHMM"])
                stop = et_now.replace(hour=h, minute=m, second=0, microsecond=0)
                opened = et_now.replace(hour=oh, minute=om, second=0, microsecond=0)
                if not opened <= et_now < stop:
                    raise Skip(f"outside the nightly window ({self.c['WINDOW_OPEN_HHMM']}-{self.c['GEN_STOP_HHMM']} "
                               f"ET, now {et_now.strftime('%H:%M')}) and not in the continuous one: give --until")
                t = stop.timestamp()
        if t <= now + float(self.c["GEN_MIN_RUN_S"]):
            raise Skip(f"--until {self.iso(t)} is less than {float(self.c['GEN_MIN_RUN_S']):.0f} s away")
        if t > now + 3600 * float(self.c["GEN_MAX_HOURS"]):
            raise Skip(f"--until {self.iso(t)} is more than {self.c['GEN_MAX_HOURS']} h away")
        return t

    def eval_pause(self) -> str | None:
        """Why nothing may use the teacher's server now (a nightly eval running or about to start): the
        local branch, and the Max teacher while its judge is local. None: no eval near."""
        if self.w.eval_running():
            return "a nightly eval is running"
        mins = self.w.minutes_to_next_eval()
        if mins < float(self.c["GEN_EVAL_PAUSE_BEFORE_MIN"]):
            return f"a nightly eval starts in {mins:.0f} min"
        now = self.w.et()
        for t in self.c["EVAL_TIMES"].split(","):
            since = (now - self.w.at_today(t.strip())).total_seconds() / 60
            if 0 <= since < float(self.c["GEN_EVAL_PAUSE_AFTER_MIN"]):
                return f"the {t.strip()} nightly eval is due (cron may start it late)"
        return None

    # commands

    def py_env(self) -> dict[str, str]:
        env = {k: v for k, v in self.env.items() if k.upper() not in ("PYTHONPATH", "HTTP_PROXY", "HTTPS_PROXY",
                                                                        "ALL_PROXY", "PYTHONHOME")}
        env.update(PYTHONPATH=str(self.code / "src"), PYTHONUNBUFFERED="1", NO_PROXY="*")
        return env

    def local_out(self) -> Path:
        return self.runs / f"local-{slug(self.alias or 'teacher')}"

    def paid_out(self, mode: str = ROUTE_PAYG) -> Path:
        return self.runs / ROUTES[mode or ROUTE_PAYG][2]

    def sandbox_root(self, label: str) -> str:
        """One --sandbox-root per branch: teacher_gen wipes ``<root>/w<N>/workspace`` at every task start and
        end, so the local and Max runs (both have a worker w0) must never share one."""
        base = self.c.get("GEN_SANDBOX_ROOT") or os.path.join(tempfile.gettempdir(), "prometheus-teacher-gen")
        return f"{base.rstrip('/')}-{label}"

    def local_judge(self) -> bool:
        """Whether the Max run's judge is the local teacher (through the guard): then it shares the
        teacher's server with the nightly evals too, and pauses for them like the local branch."""
        return self.c["GEN_MAX_JUDGE"] == "guard"

    @staticmethod
    def wrote_report(ch: Child | None, out: Path, name: str) -> bool:
        """Whether teacher_gen rewrote ``<out>/<name>.report.json`` during this child's run: it does at
        the end of every run that returns normally, so an exit 1 without it was a crash, not "kept nothing"."""
        if ch is None:
            return False
        try:
            return (out / f"{name}.report.json").stat().st_mtime >= ch.started_at - 2
        except OSError:
            return False

    def common_args(self, until: float) -> list[str]:
        a = ["--tasks", self.c["GEN_TASKS"], "--until", self.iso(until),
             "--injection-manifest", self.c["GEN_INJECTION_MANIFEST"]]
        for p in self.c["GEN_EVAL_TOOLS"].split():
            a += ["--eval-tools", p]
        if self.c["GEN_PACK"]:
            a += ["--pack", self.c["GEN_PACK"]]
        return a + ["--repo", self.c["GEN_REPO"], "--quiet"]

    def has_teacher_flag(self) -> bool:
        return bool(re.search(r"--teacher \{[^}]*\blocal\b", self.help_text))

    def has_paid_mode(self, mode: str = ROUTE_PAYG) -> bool:
        if mode == ROUTE_TP:
            return (bool(re.search(r"--teacher \{[^}]*max-token-plan", self.help_text))
                    and "--tp-start-concurrency" in self.help_text)
        return bool(re.search(r"--teacher \{[^}]*max-payg", self.help_text)) and "--budget-usd" in self.help_text

    def guard_argv(self, until: float) -> list[str]:
        c = self.c
        unit = c["TEACHER_Q4_UNIT"] if self.alias == c["TEACHER_Q4_ALIAS"] else c["TEACHER_UNIT"]
        return [c["PY"], "-I", c["GEN_GUARD_PY"], "--listen", c["GEN_GUARD_LISTEN"],
                "--upstream", host_port(c["MODEL_URL"]), "--probe-unit", c["PROBE_UNIT"],
                "--probe-name", c["PROBE_NAME"], "--llama-unit", unit,
                "--log", str(self.log_dir / "probe_guard.log"), "--pidfile", str(self.state_dir / "guard.pid"),
                "--deadline", str(int(until - 60)), "--abort-file", str(self.abort_file),
                "--notify-cmd", c["NOTIFY"], "--min-tps", c["GEN_GUARD_MIN_TPS"] or c["PROBE_MIN_TPS"]]

    def local_argv(self, until: float) -> list[str]:
        c = self.c
        return (shlex.split(c["GEN_NICE_CMD"]) + [c["GEN_PY"], str(self.distill / "teacher_gen.py"), "run"]
                + (["--teacher", "local"] if self.has_teacher_flag() else [])
                + ["--out", str(self.local_out()), "--teacher-url", c["MODEL_URL"], "--probe-guard-url",
                   self.guard_url, "--concurrency", c["GEN_LOCAL_CONCURRENCY"],
                   "--sandbox-root", self.sandbox_root("local")] + self.common_args(until))

    def judge_argv(self, mode: str = ROUTE_PAYG) -> list[str]:
        c = self.c
        base = shlex.split(c["GEN_MAX_NICE_CMD"]) + ["/bin/bash", str(self.distill / "judge_run.sh"), "--execute"]
        if mode == ROUTE_TP:
            return base + ["--provider", "token-plan", "--concurrency", c["GEN_TP_JUDGE_CONCURRENCY"]]
        return base + ["--provider", "payg", "--budget-usd", c["GEN_JUDGE_BUDGET_USD"],
                       "--blind-budget-usd", c["GEN_JUDGE_BLIND_BUDGET_USD"]]

    def judge_env(self) -> dict[str, str]:
        return dict(self.py_env(), PY=self.c["GEN_PY"], NOTIFY=self.c["NOTIFY"], MODEL=self.c["GEN_MAX_MODEL"])

    def paid_argv(self, until: float, mode: str = ROUTE_PAYG) -> list[str]:
        c = self.c
        judge = (["--judge-url", c["MODEL_URL"], "--probe-guard-url", self.guard_url]
                 if c["GEN_MAX_JUDGE"] == "guard" else [])
        teacher = ROUTES[mode][0]
        if mode == ROUTE_TP:
            route = ["--teacher", teacher, "--concurrency", c["GEN_TP_CONCURRENCY"],
                     "--tp-start-concurrency", c["GEN_TP_START_CONCURRENCY"]]
        else:
            route = ["--teacher", teacher, "--budget-usd", c["GEN_MAX_BUDGET_USD"], "--concurrency",
                     c["GEN_MAX_CONCURRENCY"]]
        return (shlex.split(c["GEN_MAX_NICE_CMD"]) + [c["GEN_PY"], str(self.distill / "teacher_gen.py"), "run"]
                + route + ["--notify-cmd", c["NOTIFY"], "--out", str(self.paid_out(mode)), "--sandbox-root",
                           self.sandbox_root(teacher)]
                + judge + self.common_args(until))

    def spend_preflight(self, mode: str = ROUTE_PAYG) -> tuple[bool, str]:
        cmd = "tokenplan-preflight" if mode == ROUTE_TP else "preflight"
        argv = [self.c["GEN_PY"], str(self.distill / "spend.py"), cmd, "--notify-cmd", self.c["NOTIFY"],
                "--model", self.c["GEN_MAX_MODEL"]]
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, timeout=120, env=self.py_env())
        except (OSError, subprocess.SubprocessError) as e:
            return False, f"spend.py {cmd} did not run: {type(e).__name__}"
        if cp.returncode == 0:
            return True, f"spend {cmd} ok"
        last = (cp.stderr.strip().splitlines() or cp.stdout.strip().splitlines() or ["?"])[-1]
        return False, f"spend {cmd} refused (exit {cp.returncode}): {last[:240]}"

    def paid_mode(self) -> str:
        """The Max branch's route (guardrail 6: the pay-per-token key always wins), or "" for none."""
        if key_present(self.c["GEN_PAYG_KEY_NAME"], self.env):
            return ROUTE_PAYG
        if self.c["GEN_TOKEN_PLAN"].strip().lower() == "on" and key_present(self.c["GEN_TP_KEY_NAME"], self.env):
            return ROUTE_TP
        return ""

    def pause_until(self) -> float | None:
        """When the Token Plan's quota pause (spend.py's PAUSE file) ends, while it stands; else None. An
        unreadable file counts as one hour from its modification time (spend.py's own rule)."""
        try:
            raw = self.tp_pause.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:
            raw = ""
        try:
            until = float(json.loads(raw)["until"])
        except (ValueError, KeyError, TypeError):
            try:
                until = self.tp_pause.stat().st_mtime + 3600.0
            except OSError:
                return None
        return until if self.now() < until else None

    # checks

    def preflight(self, until: float) -> list[str]:
        c = self.c
        problems: list[str] = []
        if self.flag.exists():
            problems.append(f"{self.flag} is present")
        if self.gen_stop.exists():
            problems.append(f"{self.gen_stop} is present (remove it to allow generation)")
        st, ids = self.served_ids()
        self.alias = self.teacher_alias() if st == 200 else None
        if self.alias is None:
            problems.append(f"{c['MODEL_URL']} serves {ids or ('nothing (HTTP ' + str(st) + ')')}, not a teacher "
                            f"({c['TEACHER_ALIAS']} / {c['TEACHER_Q4_ALIAS']}): no window is open")
        else:
            self.log(f"{c['MODEL_URL']} serves the teacher {self.alias}")
        need = [self.code / "COMMIT"] + [self.distill / n for n in ("teacher_gen.py", "taskset_gen.py", "judge_run.sh",
                                                                    "spend.py", "artifact_policy.py")]
        missing = [str(p) for p in need if not p.exists()]
        if missing:
            problems.append("code not staged (gen_runner.py stage-code): missing " + ", ".join(missing))
        else:
            self.log(f"code {(self.code / 'COMMIT').read_text().strip()[:12]} in {self.code}")
        if not os.access(c["GEN_PY"], os.X_OK):
            problems.append(f"GEN_PY {c['GEN_PY']} is not executable")
        elif not missing:
            try:
                cp = subprocess.run([c["GEN_PY"], "-B", "-c", "import httpx, prometheus; print(prometheus.__file__)"],
                                    capture_output=True, text=True, timeout=120, env=self.py_env())
                where = cp.stdout.strip()
                if cp.returncode != 0 or not os.path.realpath(where).startswith(os.path.realpath(self.code / "src")):
                    problems.append(f"{c['GEN_PY']} does not import Prometheus from {self.code / 'src'} "
                                    f"({where or (cp.stderr.strip().splitlines() or ['?'])[-1][:160]})")
                cp = subprocess.run([c["GEN_PY"], "-B", str(self.distill / "teacher_gen.py"), "run", "--help"],
                                    capture_output=True, text=True, timeout=120, env=self.py_env())
                if cp.returncode != 0:
                    problems.append(f"teacher_gen.py run --help exited {cp.returncode}")
                self.help_text = cp.stdout
            except (OSError, subprocess.SubprocessError) as e:
                problems.append(f"{c['GEN_PY']}: {type(e).__name__}: {e}")
        for key in ("GEN_TASKS", "GEN_INJECTION_MANIFEST", "GEN_PACK", "GEN_GUARD_PY"):
            if c[key] and not Path(c[key]).is_file():
                problems.append(f"{key} {c[key]} missing")
        if not missing and Path(c["GEN_TASKS"]).is_file():
            problem = self.exclusion_problem()
            if problem:
                problems.append(problem)
        for p in c["GEN_EVAL_TOOLS"].split():
            if not Path(p).is_file():
                problems.append(f"GEN_EVAL_TOOLS {p} missing")
        repo = Path(c["GEN_REPO"])
        if not ((repo / "docs").is_dir() or (repo / "README.md").is_file()):
            problems.append(f"GEN_REPO {repo} has no docs/ or README.md (force_compaction pads with real docs)")
        if not port_free(c["GEN_GUARD_LISTEN"]):
            problems.append(f"{c['GEN_GUARD_LISTEN']} is taken (another guard?)")
        if not os.access(c["NOTIFY"], os.X_OK):
            problems.append(f"notify command {c['NOTIFY']} is not executable")
        return problems

    def exclusion_problem(self) -> str | None:
        """Why GEN_TASKS may not feed a teacher (DISTILL-OPEN-QUESTIONS #6: the excluded sessions are in no
        training data, and a gym row carries no session id to drop it by later): ``taskset_gen.py exclude
        --check`` against the operator's list finds a row built from a listed session's prompt, the list is
        missing, or the check could not run. None: clean."""
        c = self.c
        lst = c.get("GEN_EXCLUDE_SESSIONS", "")
        if not lst or not Path(lst).is_file():
            return f"GEN_EXCLUDE_SESSIONS {lst or '(empty)'} missing: the excluded sessions cannot be checked"
        argv = [c["GEN_PY"], "-B", str(self.distill / "taskset_gen.py"), "exclude", "--check",
                "--tasks", c["GEN_TASKS"], "--exclude-sessions", lst]
        try:
            cp = subprocess.run(argv, capture_output=True, text=True, env=self.py_env(),
                                timeout=float(c["GEN_EXCLUDE_CHECK_TIMEOUT_S"]))
        except (OSError, subprocess.SubprocessError) as e:
            return f"the excluded-session check did not run: {type(e).__name__}"
        last = (cp.stderr.strip().splitlines() or cp.stdout.strip().splitlines() or ["?"])[-1][:240]
        if cp.returncode == 0:
            self.log(f"GEN_TASKS holds no row of an excluded session ({Path(c['GEN_TASKS']).name})")
            return None
        if cp.returncode == 1:
            try:
                n = json.loads(cp.stdout.strip().splitlines()[-1]).get("rows_dropped", "?")
            except (ValueError, IndexError, AttributeError):
                n = "?"
            return (f"GEN_TASKS {c['GEN_TASKS']} holds {n} rows built from an excluded session's prompts: "
                    f"run taskset_gen.py exclude and point GEN_TASKS at its output")
        return f"the excluded-session check failed (exit {cp.returncode}): {last}"

    def paid_plan(self, mode: str | None = None) -> tuple[bool, str]:
        """Whether the Max branch may start on *mode* (default: the route the keys pick now; key present +
        the route's spend preflight), and why not."""
        mode = self.paid_mode() if mode is None else mode
        if not mode:
            return False, (f"neither {self.c['GEN_PAYG_KEY_NAME']} nor (with GEN_TOKEN_PLAN=on) "
                           f"{self.c['GEN_TP_KEY_NAME']} is present (environment or env file): Max judge and "
                           "Max teacher skipped")
        if mode == ROUTE_PAYG and not key_present(self.c["GEN_PAYG_KEY_NAME"], self.env):
            return False, f"{self.c['GEN_PAYG_KEY_NAME']} is not present"
        if mode == ROUTE_TP and key_present(self.c["GEN_PAYG_KEY_NAME"], self.env):
            return False, f"{self.c['GEN_PAYG_KEY_NAME']} is present: Token Plan batch use stops"
        if self.spend_stop.exists():
            return False, f"the spend STOP file {self.spend_stop} is present: no Max call"
        if mode == ROUTE_TP:
            until = self.pause_until()
            if until is not None:
                return False, f"the Token Plan quota pause stands until {self.iso(until)}"
        if not (self.distill / "spend.py").exists():
            return False, "spend.py not staged"
        return self.spend_preflight(mode)

    def print_plan(self, until: float) -> None:
        self.log(f"plan, until {self.iso(until)} (local concurrency {self.c['GEN_LOCAL_CONCURRENCY']}):")
        self.log("  a. probe guard: " + shlex.join(self.guard_argv(until)))
        self.log(f"  b. local teacher -> {self.local_out()}: " + shlex.join(self.local_argv(until)))
        route = self.paid_mode() or ROUTE_PAYG
        ok, why = self.paid_plan()
        self.log(f"  c. paid branch ({'route ' + route if self.paid_mode() else 'no key'}): "
                 f"{'GO' if ok else 'SKIP'} ({why})")
        self.log("     1) " + shlex.join(self.judge_argv(route)) + "   [PY, PYTHONPATH, NOTIFY in its env]")
        mode = ("" if self.has_paid_mode(route) or not self.help_text
                else f"   [this teacher_gen has no --teacher {ROUTES[route][0]}: it would be skipped]")
        self.log(f"     2) -> {self.paid_out(route)}: " + shlex.join(self.paid_argv(until, route)) + mode)
        self.log(f"  d. harvest every {self.c['GEN_HARVEST_EVERY_MIN']} min and at the end -> {self.pool}")

    def cmdline(self, pid: int) -> str:
        try:
            return (Path(self.c["PROC_ROOT"]) / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:
            cp = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True)
            return cp.stdout.strip()

    def stop_orphans(self) -> None:
        """Children an earlier runner recorded (``<runs>/state/<name>.pid``) that still run: stopped by their
        recorded PID's group, only when that PID still runs what this runner starts (its command line
        names the guard script with our port, teacher_gen.py or judge_run.sh). Under systemd a unit stop
        takes them all (KillMode=mixed); this covers a runner killed outside it."""
        want = {"guard": ("probe_guard.py", self.c["GEN_GUARD_LISTEN"]), "local": ("teacher_gen.py", "run"),
                "max": ("teacher_gen.py", "--teacher max-"), "judge": ("judge_run.sh", "--execute")}
        for name, sig in want.items():
            f = self.state_dir / f"{name}.pid"
            try:
                pid = int(f.read_text().split()[0])
            except (OSError, ValueError, IndexError):
                continue
            cmd = self.cmdline(pid) if pid > 1 else ""
            if cmd and all(s in cmd for s in sig):
                self.note(f"stopping a {name} an earlier run left (pid {pid})")
                for sgn in (signal.SIGTERM, signal.SIGKILL):
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(pid, sgn)
                    deadline = time.monotonic() + float(self.c["GEN_CHILD_GRACE_S"])
                    while time.monotonic() < deadline and self.cmdline(pid):
                        time.sleep(0.2)
                    if not self.cmdline(pid):
                        break
            with contextlib.suppress(OSError):
                f.unlink()

    # the branches

    def start_child(self, name: str, argv: list[str], env: dict[str, str]) -> Child:
        stamp = datetime.fromtimestamp(self.now(), self.tz).strftime("%Y%m%d")
        ch = Child(name, argv, self.log_dir / f"{name}-{stamp}.log", env, self.runs)
        ch.start(self.state_dir)
        self.log(f"started {name} (pid {ch.proc.pid if ch.proc else '?'}): {shlex.join(argv)}")
        return ch

    def stop_child(self, ch: Child | None, why: str) -> int | None:
        if ch is None or ch.proc is None:
            return None
        was = ch.running
        rc = ch.stop(float(self.c["GEN_CHILD_GRACE_S"]), self.state_dir)
        if was:
            self.log(f"stopped {ch.name} pid {ch.proc.pid} ({why}): exit {rc}")
        return rc

    def start_guard(self, until: float) -> None:
        if self.abort_file.exists():
            self.note(f"an earlier guard's abort switch had tripped ({self.abort_file.read_text().strip()[:120]}); "
                      "starting fresh")
            os.replace(self.abort_file, self.abort_file.with_suffix(".prev"))
        self.guard = self.start_child("guard", self.guard_argv(until), dict(self.env))
        deadline = time.monotonic() + float(self.c["GEN_GUARD_WAIT_S"])
        while time.monotonic() < deadline and not self.stop:
            if not self.guard.running:
                raise Fail(f"the probe guard exited at start (exit {self.guard.poll()}); see {self.guard.log_path}")
            st, body = self.http_json(self.guard_url + "/v1/models")
            ids = [str(m.get("id")) for m in (body or {}).get("data", []) if isinstance(m, dict)] if st == 200 \
                and isinstance(body, dict) else []
            if self.alias in ids:
                self.log(f"probe guard up on {self.c['GEN_GUARD_LISTEN']}: serves {self.alias}")
                return
            time.sleep(0.2)
        raise Fail(f"the probe guard did not answer with {self.alias} within {self.c['GEN_GUARD_WAIT_S']} s")

    def guard_down(self) -> str | None:
        if self.abort_file.exists():
            return "the probe guard's abort switch tripped: " + self.abort_file.read_text().strip()[:160]
        if self.guard is None or not self.guard.running:
            return f"the probe guard is gone (exit {self.guard.poll() if self.guard else '?'})"
        return None

    def tick_local(self, until: float) -> None:
        b = self.local
        if b.state in FINAL:
            return
        down = self.guard_down()
        pause = self.eval_pause()
        if b.state == "running" and b.child is not None:
            rc = b.child.poll()
            if rc is None:
                if down:
                    self.stop_child(b.child, down)
                    b.state, b.detail = "stopped", down
                    self.note(f"local generation stopped for this run: {down}")
                elif pause:
                    self.stop_child(b.child, pause)
                    b.state, b.detail = "paused", pause
                    self.note(f"local generation paused: {pause}")
                return
            b.rcs.append(rc)
            crashed = rc == TG_EXIT_CRASH or (rc == 1 and not self.wrote_report(b.child, self.local_out(),
                                                                                 LOCAL_NAME))
            if crashed:
                self.note(f"local teacher_gen crashed (exit {rc}"
                          f"{'' if rc == TG_EXIT_CRASH else ', no report from this run'}): see {b.child.log_path}")
            if rc in (0, 1) and not crashed:
                b.state = "done"
                b.detail = "teacher_gen finished (exit 0)" if rc == 0 else "teacher_gen kept nothing this run (exit 1)"
                self.log(f"local: {b.detail}")
            elif rc == 2:
                b.state, b.detail = "failed", f"teacher_gen refused (exit 2): see {b.child.log_path}"
                self.note(b.detail)
            elif b.restarts < int(self.c["GEN_LOCAL_RESTARTS"]):
                b.restarts += 1
                b.state = "waiting"
                b.resume_at = time.monotonic() + float(self.c["GEN_RESTART_WAIT_S"])
                b.detail = f"exit {rc}; restart {b.restarts}/{self.c['GEN_LOCAL_RESTARTS']} in " \
                           f"{self.c['GEN_RESTART_WAIT_S']} s"
                self.log(f"local: {b.detail}")
            else:
                b.state, b.detail = "failed", f"exit {rc} after {b.restarts} restarts: see {b.child.log_path}"
                self.note(b.detail)
            self.save()
            return
        if down:
            b.state, b.detail = "stopped", down
            self.note(f"local generation not (re)started: {down}")
            return
        if pause:
            if b.state != "paused":
                b.state, b.detail = "paused", pause
                self.note(f"local generation waits: {pause}")
            return
        if b.state == "waiting" and time.monotonic() < b.resume_at:
            return
        if until - self.now() < float(self.c["GEN_MIN_START_S"]):
            b.state, b.detail = "done", "too close to --until to start again"
            return
        if b.state == "paused":
            self.note("local generation resumes (the nightly eval is over)")
        b.child = self.start_child("local", self.local_argv(until), self.py_env())
        b.state, b.detail = "running", ""
        b.runs += 1
        self.save()

    def tick_paid(self, until: float) -> None:
        b = self.paid
        if b.state in FINAL:
            return
        if b.mode == ROUTE_TP and key_present(self.c["GEN_PAYG_KEY_NAME"], self.env):
            # guardrail 6: the pay-per-token key appeared: Token Plan batch use stops, the branch starts over
            self.stop_child(b.child, "QWEN_PAYG_KEY appeared")
            b.child, b.state, b.mode = None, "pending", ""
            b.detail = f"{self.c['GEN_PAYG_KEY_NAME']} appeared: Token Plan batch use stopped; switching routes"
            self.note(b.detail)
            self.save()
            return
        if b.state == "pending":
            mode = self.paid_mode()
            if mode == ROUTE_TP and self.pause_until() is not None and not self.spend_stop.exists():
                b.mode = mode
                self.park_quota(f"the Token Plan quota pause stands until {self.iso(self.pause_until() or 0)}")
                return
            ok, why = self.paid_plan(mode)
            self.log(f"paid branch ({mode or 'no key'}): {'GO' if ok else 'SKIP'} ({why})")
            if not ok:
                b.state, b.detail = "off", why
                self.save()
                return
            b.mode = mode
            b.child = self.start_child("judge", self.judge_argv(mode), self.judge_env())
            b.state = "judge"
            self.save()
            return
        if self.spend_stop.exists():
            self.stop_child(b.child, "spend STOP")
            b.state, b.detail = "stopped", (f"the spend STOP file {self.spend_stop} appeared ($140 alert, or a Token "
                                            "Plan account error or key change): no more Max calls")
            self.note(b.detail)
            self.save()
            return
        if b.state == "quota":
            if self.pause_until() is not None or self.now() < b.resume_at:
                return
            if until - self.now() < float(self.c["GEN_MIN_START_S"]):
                b.state, b.detail = "done", "the quota pause ended too close to --until"
                self.save()
                return
            self.note("Token Plan quota pause over: the Max branch starts over (one probe call first)")
            b.child, b.state, b.mode = None, "pending", ""
            self.save()
            return
        if b.state == "paused":
            self.resume_paid(until)
            return
        rc = b.child.poll() if b.child else None
        if b.state == "teacher" and rc is None and self.local_judge():
            down = self.guard_down()
            if down:
                self.stop_child(b.child, down)
                b.state, b.detail = "stopped", f"its local judge's guard: {down}"
                self.note(f"Max teacher stopped: {down}")
                self.save()
                return
            pause = self.eval_pause()
            if pause:
                # its judge requests go to the teacher's server the nightly eval scores: stopped like the
                # local branch (the calls in flight settle first; the tasks in flight rerun from the ledger)
                self.stop_child(b.child, pause)
                b.state, b.detail = "paused", pause
                self.note(f"Max teacher paused (its local judge shares the evaluated server): {pause}")
                self.save()
            return
        if rc is None:
            return
        b.rcs.append(rc)
        mode = b.mode or ROUTE_PAYG
        if b.state == "judge":
            self.log(f"judge_run.sh exit {rc} ({JUDGE_EXIT.get(rc, 'see its log')})")
            if mode == ROUTE_TP and self.pause_until() is not None:
                self.park_quota(f"judge exit {rc}; the Token Plan quota pause stands until "
                                f"{self.iso(self.pause_until() or 0)}")
                return
            if rc == 5:
                b.state, b.detail = "done", "the Max judge stopped fatally (key, host, account, billing or auth): no Max teacher"
                self.note(b.detail)
                return
            ok, why = self.paid_plan(mode)
            if not ok:
                b.state, b.detail = "done", f"judge exit {rc}; then {why}"
                self.log(f"paid branch: {b.detail}")
                return
            if not self.has_paid_mode(mode):
                b.state, b.detail = "off", (f"judge exit {rc}; this teacher_gen has no --teacher {ROUTES[mode][0]} "
                                            "yet: Max teacher skipped")
                self.note(b.detail)
                return
            if until - self.now() < float(self.c["GEN_MIN_START_S"]):
                b.state, b.detail = "done", f"judge exit {rc}; too close to --until for the Max teacher"
                return
            pause = self.eval_pause() if self.local_judge() else None
            if pause:
                b.state, b.detail = "paused", f"judge exit {rc}; the Max teacher waits: {pause}"
                self.note(f"Max teacher waits (its local judge shares the evaluated server): {pause}")
                self.save()
                return
            b.child = self.start_child("max", self.paid_argv(until, mode), self.py_env())
            b.state, b.detail = "teacher", f"judge exit {rc}"
            self.save()
            return
        where = f": see {b.child.log_path if b.child else '?'}"
        crashed = rc == TG_EXIT_CRASH or (rc == 1 and not self.wrote_report(b.child, self.paid_out(mode),
                                                                            ROUTES[mode][1]))
        if rc in (0, 1) and not crashed:
            b.state, b.detail = "done", f"Max teacher exit {rc}"
        elif crashed:
            b.state, b.detail = "failed", f"Max teacher crashed (exit {rc}){where}"
            self.note(b.detail)
        elif rc == TG_EXIT_QUOTA_PAUSED and mode == ROUTE_TP:
            self.park_quota(f"Max teacher exit {rc}: the Token Plan quota pause stands"
                            + (f" until {self.iso(self.pause_until() or 0)}" if self.pause_until() else ""))
            return
        elif rc == TG_EXIT_SPEND_STOP:
            # its phase budget (--budget-usd, a total across runs) or the $150 cap is reached, or the
            # STOP file appeared between two polls: a planned end, not a failure; never restarted
            b.state = "done"
            b.detail = (f"Max teacher exit {rc}: the spend ledger refused a paid call (phase budget "
                        f"${self.c['GEN_MAX_BUDGET_USD']}, the cap or STOP){where}") if mode == ROUTE_PAYG else (
                        f"Max teacher exit {rc}: the STOP file, or {self.c['GEN_PAYG_KEY_NAME']} appeared{where}")
            self.note(b.detail)
        elif rc == TG_EXIT_PAYG_REFUSED:
            b.state = "stopped"
            b.detail = f"Max teacher exit {rc}: the key, host, account or billing was refused (fatal){where}"
            self.note(b.detail)
        else:
            b.state, b.detail = "failed", f"Max teacher exit {rc}{where}"
            self.note(b.detail)
        self.log(f"paid branch: {b.detail}")
        self.save()

    def park_quota(self, detail: str) -> None:
        """The Token Plan's quota is exhausted: the Max branch waits (state ``quota``) for spend.py's pause to
        end, then starts over. Not a failure, not a stop (spend.py told the operator once)."""
        b = self.paid
        b.child, b.state, b.detail = None, "quota", detail
        # with a pause standing its end governs; without one (a probe in flight elsewhere) wait a while
        b.resume_at = 0.0 if self.pause_until() is not None else self.now() + float(self.c["GEN_TP_QUOTA_RECHECK_S"])
        self.note(f"Max branch waits: {detail}")
        self.save()

    def resume_paid(self, until: float) -> None:
        """The paid branch's Max teacher, paused for a nightly eval: started again once the pause is over,
        after the same checks as its first start (the guard, the time left, the key, STOP and the spend
        preflight). It resumes from its own ledger."""
        b = self.paid
        down = self.guard_down()
        if down:
            b.state, b.detail = "stopped", f"its local judge's guard: {down}"
            self.note(f"Max teacher not resumed: {down}")
            self.save()
            return
        if self.eval_pause():
            return
        if until - self.now() < float(self.c["GEN_MIN_START_S"]):
            b.state, b.detail = "done", "too close to --until to resume the Max teacher"
            self.save()
            return
        mode = b.mode or ROUTE_PAYG
        if mode == ROUTE_TP and self.pause_until() is not None:
            self.park_quota(f"the Token Plan quota pause stands until {self.iso(self.pause_until() or 0)}")
            return
        ok, why = self.paid_plan(mode)
        if not ok:
            b.state, b.detail = "done", f"Max teacher not resumed: {why}"
            self.note(b.detail)
            self.save()
            return
        self.note("Max teacher resumes (the nightly eval is over)")
        b.child = self.start_child("max", self.paid_argv(until, mode), self.py_env())
        b.state, b.detail = "teacher", "resumed after a nightly eval"
        self.save()

    def stop_reason(self, until: float) -> str | None:
        if self.stop:
            return self.stop
        if self.now() >= until:
            return "until"
        if self.flag.exists():
            return "no-swaps"
        if self.gen_stop.exists():
            return "gen-stop"
        alias = self.teacher_alias()
        if alias != self.alias:
            self.port_misses += 1
            if self.port_misses >= int(self.c["GEN_PORT_GONE_CHECKS"]):
                return "window-closed"
        else:
            self.port_misses = 0
        return None

    def run(self, until: float) -> int:
        """The supervised run (execute only). Returns the exit code."""
        self.progress.update(state="running", until=self.iso(until), teacher=self.alias,
                             code=(self.code / "COMMIT").read_text().strip()[:12])
        self.save()
        harvest_every = 60 * float(self.c["GEN_HARVEST_EVERY_MIN"])
        next_harvest = time.monotonic() + harvest_every
        reason: str | None = None
        try:
            self.start_guard(until)
            while True:
                reason = self.stop_reason(until)
                if reason:
                    break
                self.tick_local(until)
                self.tick_paid(until)
                if time.monotonic() >= next_harvest:
                    self.harvest_now()
                    next_harvest = time.monotonic() + harvest_every
                if self.local.state in FINAL and self.paid.state in FINAL:
                    reason = "nothing left to do"
                    break
                self.sleep(float(self.c["GEN_POLL_S"]))
        except Fail as e:
            self.note(f"FAIL: {e}")
            reason = "failed"
        finally:
            reason = reason or self.stop or "error"
            self.shutdown(reason)
        if reason in ("sigterm",) or (reason or "").startswith("signal"):
            return 143
        if reason in ("no-swaps", "gen-stop", "window-closed"):
            return 3
        if reason == "failed" or self.local.state == "failed" or self.paid.state == "failed":
            return 1
        return 0

    def shutdown(self, reason: str) -> None:
        self.log(f"stopping ({reason})")
        for b in (self.local, self.paid):
            if b.child is not None and b.child.running:
                self.stop_child(b.child, reason)
                b.state, b.detail = "stopped", f"{reason}" + (f"; {b.detail}" if b.detail else "")
        self.stop_child(self.guard, reason)
        added = self.harvest_now()
        state = "finished" if reason in ("until", "nothing left to do") else "stopped"
        if reason == "failed":
            state = "failed"
        self.progress.update(state=state, stop_reason=reason, ended_at=self.iso(self.now()))
        self.save()
        totals = ", ".join(f"{k} {v}" for k, v in sorted((added or {}).get("pool_rows", {}).items())) or "empty"
        new = ", ".join(f"{k} +{v}" for k, v in sorted(self.harvested.items())) or "+0"
        started = self.progress.get("started_at", "")[11:16]
        self.notify([
            f"mini gen {state} ({reason}) {started}-{datetime.fromtimestamp(self.now(), self.tz).strftime('%H:%M')}, "
            f"teacher {self.alias}",
            f"local: {self.local.state} ({self.local.detail or 'ran'}); runs {self.local.runs}, restarts "
            f"{self.local.restarts}"[:300],
            f"Max{' (' + self.paid.mode + ')' if self.paid.mode else ''}: {self.paid.state} "
            f"({self.paid.detail or '-'})"[:300],
            f"harvested this run: {new}; pool: {totals}"[:300],
            ("notes: " + "; ".join(self.notes[-2:]))[:300] if self.notes else "",
        ])

    # harvest

    def harvest_now(self) -> dict[str, Any] | None:
        try:
            res = harvest(self.runs, self.pool, self.distill / "artifact_policy.py", execute=self.execute,
                          log=self.log)
        except Exception as e:  # noqa: BLE001 - a harvest problem must not end generation
            self.note(f"harvest failed: {type(e).__name__}: {e}")
            return None
        for k, v in res["added"].items():
            self.harvested[k] = self.harvested.get(k, 0) + v
        self.save()
        return res


# ── harvest ──────────────────────────────────────────────────────────────────


def _load_policy(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("distill_artifact_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def finished_runs(ledger: Path) -> set[str]:
    """run ids whose rows are final in traces.jsonl: a task kept, or held and then released."""
    ok: set[str] = set()
    held: dict[str, str] = {}
    try:
        text = ledger.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ok
    for line in text.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # a torn last line: that task is not finished
        if not isinstance(ev, dict):
            continue
        st, task, run = ev.get("status"), ev.get("task_id"), ev.get("run_id")
        if st == "kept" and run:
            ok.add(str(run))
        elif st == "held" and run and task:
            held[str(task)] = str(run)
        elif st == "released" and task:
            r = held.pop(str(task), None) or run
            if r:
                ok.add(str(r))
    return ok


def harvest(runs: Path, pool: Path, policy_path: Path, *, execute: bool,
            log: Callable[[str], None] = print) -> dict[str, Any]:
    """Copy the rows of finished tasks from ``runs/*/*.traces.jsonl`` into ``pool`` (see the module doc).
    Returns {"added": {source: n}, "pool_rows": {source: n}, "unknown_provenance": n}."""
    ap = _load_policy(policy_path)
    pool_max = pool / ap.TAG  # "max-derived/": the tag in a parent's name marks every file under it
    state_path = pool / ".harvest-state.json"
    lock_path = pool / ".harvest.lock"
    if execute:
        for d in (pool, pool_max):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
    added: dict[str, int] = {}
    unknown = 0
    with _flock(lock_path if execute else None):
        files, state_ok = _load_harvest_state(state_path)
        on_disk = sorted(str(p.relative_to(pool)) for d in (pool, pool_max) for p in d.glob("*.traces.jsonl"))
        if execute and not state_ok and not on_disk and not state_path.exists():
            # the first harvest: record "nothing yet" before the first append, so a file an interrupted
            # harvest creates is known as unrecorded next time
            _write_json(state_path, {"schema": POOL_SCHEMA, "files": {}})
            state_ok = True
        _recover_pool(pool, pool_max.name, files, on_disk, state_ok, execute=execute, log=log)
        seen: dict[str, set[str]] = {rel: set(rec["runs"]) for rel, rec in files.items()}
        for traces in sorted(runs.glob("*/*.traces.jsonl")):
            label = traces.parent.name
            name = traces.name[: -len(".traces.jsonl")]
            ok = finished_runs(traces.parent / f"{name}.tasks.jsonl")
            if not ok:
                continue
            new: dict[str, list[str]] = {}
            new_runs: dict[str, set[str]] = {}
            meta_of: dict[str, tuple[str, bool]] = {}
            with traces.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.endswith("\n"):
                        continue  # torn: teacher_gen is writing it (or cuts it on resume)
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    meta = row.get("_meta") if isinstance(row, dict) else None
                    run = str((meta or {}).get("run_id") or "")
                    if not run or run not in ok:
                        continue
                    verdict = ap.row_verdict(row)
                    is_max = verdict is not False
                    if verdict is None:
                        unknown += 1
                    source = slug(str((meta or {}).get("source") or "unknown"))
                    rel = (f"{pool_max.name}/" if is_max else "") + f"{source}.{slug(label)}.traces.jsonl"
                    if run in seen.get(rel, set()):
                        continue
                    new.setdefault(rel, []).append(line)
                    new_runs.setdefault(rel, set()).add(run)
                    meta_of[rel] = (source, is_max)
            for rel, lines in new.items():
                src, is_max = meta_of[rel]
                key = f"{src}{' (max-derived)' if is_max else ''}"
                added[key] = added.get(key, 0) + len(lines)
                if not execute:
                    continue
                dest = pool / rel
                fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as out:
                    out.write("".join(lines))
                    out.flush()
                    os.fsync(out.fileno())
                seen.setdefault(rel, set()).update(new_runs[rel])
                files[rel] = {"size": dest.stat().st_size, "runs": sorted(seen[rel]), "source": src,
                              "max_derived": is_max}
                # recorded right after each append: an interruption leaves at most this one file ahead
                # of the state, and the next harvest cuts it back (_recover_pool)
                _write_json(state_path, {"schema": POOL_SCHEMA, "files": files})
        pool_rows: dict[str, int] = {}
        for rel, rec in files.items():
            n = _count_lines(pool / rel)
            key = f"{rec.get('source')}{' (max-derived)' if rec.get('max_derived') else ''}"
            pool_rows[key] = pool_rows.get(key, 0) + n
        if execute:
            _write_json(state_path, {"schema": POOL_SCHEMA, "files": files})
            _write_json(pool / "harvest.json", {
                "schema": POOL_SCHEMA, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rows_by_source": pool_rows, "last_added": added, "unknown_provenance_last": unknown,
                "files": {rel: {"rows": _count_lines(pool / rel), "source": rec.get("source"),
                                "max_derived": rec.get("max_derived")} for rel, rec in sorted(files.items())},
                "note": "max-derived/ holds Max-derived rows (or rows of unknown provenance): train only under "
                        "build.py --allow-max-derived, never published (Q6b)"})
    if added or not execute:
        log(("harvest: " if execute else "[dry run] harvest would add: ")
            + (", ".join(f"{k} +{v}" for k, v in sorted(added.items())) or "nothing new")
            + f"; pool now {', '.join(f'{k} {v}' for k, v in sorted(pool_rows.items())) or 'empty'}"
            + (f"; {unknown} rows of unknown provenance went to {pool_max.name}/" if unknown else ""))
    return {"added": added, "pool_rows": pool_rows, "unknown_provenance": unknown}


def _load_harvest_state(path: Path) -> tuple[dict[str, Any], bool]:
    """(the recorded pool files, whether a state was read at all)."""
    try:
        state = json.loads(path.read_text())
        files = state["files"]
        if not isinstance(files, dict) or not all(isinstance(r, dict) and isinstance(r.get("size"), int)
                                                   and isinstance(r.get("runs"), list) for r in files.values()):
            raise ValueError("bad files")
        return files, True
    except (OSError, ValueError, KeyError, TypeError):
        return {}, False


def _recover_pool(pool: Path, max_dir: str, files: dict[str, Any], on_disk: list[str], state_ok: bool, *,
                  execute: bool, log: Callable[[str], None]) -> None:
    """Make the pool files agree with the state before anything is copied (in place on *files*):

    * longer than recorded (an append the state never recorded, maybe torn): cut back to the recorded
      size; its rows are copied again;
    * never recorded while a state exists (created by an interrupted harvest): removed; copied again;
    * no state at all (deleted by hand) or a file shorter than recorded: what the file holds counts, a
      torn last line cut.

    A dry run changes nothing and plans as if it had (the rows it would copy again are counted)."""
    for rel in sorted(set(on_disk) | set(files)):
        p = pool / rel
        size = p.stat().st_size if p.exists() else 0
        rec = files.get(rel)
        if rec is None and state_ok:
            log(f"harvest: {rel} was never recorded (an interrupted harvest): removed, its rows are copied again")
            if execute:
                p.unlink()
            continue
        if rec is not None and size > rec["size"]:
            log(f"harvest: {rel} holds {size - rec['size']} bytes the state never recorded (an interrupted "
                "harvest): cut back, those rows are copied again")
            if execute:
                os.truncate(p, rec["size"])
            continue
        if rec is None or size < rec["size"]:
            if execute:
                size = _cut_torn_tail(p) if p.exists() else 0
            src = (rec or {}).get("source") or rel.rsplit("/", 1)[-1].split(".", 1)[0]
            is_max = (rec or {}).get("max_derived", rel.startswith(max_dir + "/"))
            files[rel] = {"size": size, "runs": sorted(_runs_in(p)), "source": src, "max_derived": is_max}


def _cut_torn_tail(path: Path) -> int:
    """Cut *path* after its last newline (a torn last line goes); returns the new size."""
    size = path.stat().st_size
    with open(path, "rb+") as fh:
        pos, keep = size, 0
        while pos > 0:
            step = min(1 << 16, pos)
            pos -= step
            fh.seek(pos)
            i = fh.read(step).rfind(b"\n")
            if i >= 0:
                keep = pos + i + 1
                break
        if keep != size:
            fh.truncate(keep)
    return keep


def _runs_in(path: Path) -> set[str]:
    out: set[str] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                with contextlib.suppress(ValueError, AttributeError):
                    run = (json.loads(line).get("_meta") or {}).get("run_id")
                    if run:
                        out.add(str(run))
    except OSError:
        pass
    return out


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


@contextlib.contextmanager
def _flock(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_run(g: Gen, until_arg: str | None) -> int:
    until = g.resolve_until(until_arg)
    if not g.execute:
        problems = g.preflight(until)
        g.print_plan(until)
        if problems:
            raise Skip("; ".join(problems))
        g.log("dry run: nothing started; a real run (--execute) starts the plan above")
        return 0
    lock = os.open(g.state_dir / "gen.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Skip("another generation runner holds the lock") from None
        (g.state_dir / "runner.pid").write_text(f"{os.getpid()}\n")
        g.stop_orphans()
        problems = g.preflight(until)
        g.print_plan(until)
        if problems:
            raise Skip("; ".join(problems))
        return g.run(until)
    finally:
        os.close(lock)


def cmd_status(g: Gen) -> int:
    try:
        print(json.dumps(json.loads((g.state_dir / "progress.json").read_text()), indent=1))
    except (OSError, ValueError):
        print(f"no progress file in {g.state_dir}")
    try:
        print(json.dumps(json.loads((g.pool / "harvest.json").read_text())["rows_by_source"], indent=1))
    except (OSError, ValueError, KeyError):
        print(f"no harvest.json in {g.pool}")
    return 0


def cmd_stage_code(args: argparse.Namespace) -> int:
    """(repo machine) ``git archive HEAD`` of src/, scripts/distill/, docs/, README.md and pyproject.toml into
    ``--dest`` on the host (via ``<dest>.incoming``; the previous copy kept as ``<dest>.prev``), plus COMMIT."""
    repo = HERE.parents[2]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", "src", "scripts/distill"]).returncode
    dest = args.dest.rstrip("/")
    remote = (f"set -e; umask 077; mkdir -p \"$(dirname ~/{dest})\"; rm -rf ~/{dest}.incoming; "
              f"mkdir -m 700 ~/{dest}.incoming; tar -C ~/{dest}.incoming --no-same-owner -xf -; "
              f"echo {head} > ~/{dest}.incoming/COMMIT; rm -rf ~/{dest}.prev; "
              f"if [ -e ~/{dest} ]; then mv ~/{dest} ~/{dest}.prev; fi; mv ~/{dest}.incoming ~/{dest}; "
              f"echo staged $(cat ~/{dest}/COMMIT) in ~/{dest}")
    archive = ["git", "-C", str(repo), "archive", "--format=tar", "HEAD", "src", "scripts/distill", "docs",
               "README.md", "pyproject.toml"]
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh, remote]
    if dirty:
        print("note: uncommitted changes under src/ or scripts/distill/ are NOT staged (HEAD only)")
    print(f"{'' if args.execute else '[dry run] '}{shlex.join(archive)} | {shlex.join(ssh)}")
    if not args.execute:
        print("dry run: nothing copied (pass --execute)")
        return 0
    a = subprocess.Popen(archive, stdout=subprocess.PIPE)
    cp = subprocess.run(ssh, stdin=a.stdout, capture_output=True, text=True, timeout=600)
    if a.stdout:
        a.stdout.close()
    a.wait()
    print((cp.stdout + cp.stderr).strip())
    return 0 if cp.returncode == 0 and a.returncode == 0 else 1


def main(argv: list[str] | None = None, *, out: Callable[[str], None] | None = None,
         clock: Callable[[], float] = time.time, env: dict[str, str] | None = None, signals: bool = False) -> int:
    ap = argparse.ArgumentParser(prog="gen_runner", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--conf", default=os.environ.get("DISTILL_WINDOW_CONF") or str(HERE / "window.conf"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--execute", action="store_true")
    r.add_argument("--until", help="HH:MM ET (next occurrence) or ISO; default: the window's own end")
    h = sub.add_parser("harvest")
    h.add_argument("--execute", action="store_true")
    sub.add_parser("status")
    st = sub.add_parser("stage-code", help="(repo machine) HEAD export to the host's GEN_CODE")
    st.add_argument("--ssh", required=True)
    st.add_argument("--dest", default="distill/phase2b/code-head", help="on the host, relative to its home")
    st.add_argument("--execute", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    if args.cmd == "stage-code":
        return cmd_stage_code(args)
    conf = load(Path(args.conf))
    execute = bool(getattr(args, "execute", False))
    g = Gen(conf, execute, out=out, clock=clock, env=env)
    if args.cmd == "status":
        return cmd_status(g)
    if execute:
        for d in (g.runs, g.state_dir, g.log_dir):
            d.mkdir(parents=True, exist_ok=True)
            os.chmod(d, 0o700)
        g.log_file = g.log_dir / f"gen-{datetime.fromtimestamp(g.now(), g.tz).strftime('%F')}.log"
        g.log(f"start pid {os.getpid()} ({args.cmd})")
    else:
        g.log("dry run: read-only checks run, every action is printed instead (pass --execute)")
    if args.cmd == "harvest":
        policy = g.distill / "artifact_policy.py"
        if not policy.exists():
            g.log(f"no {policy}: stage the code first")
            return 2
        harvest(g.runs, g.pool, policy, execute=execute, log=g.log)
        return 0

    def on_signal(signum: int, _frame: Any) -> None:
        if not g.stop:
            g.stop = "sigterm" if signum == signal.SIGTERM else f"signal-{signum}"

    if signals:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, on_signal)
    try:
        return cmd_run(g, args.until)
    except Skip as e:
        g.log(f"SKIP: {e}")
        if execute:
            g.notify([f"mini gen refused (nothing started): {e}"[:600]])
        return 2
    finally:
        g.log("end")


if __name__ == "__main__":
    sys.exit(main(signals=True))
