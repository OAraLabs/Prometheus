#!/usr/bin/env python3
"""GPU-window steps on the 3090 Ti host, Track B (docs/sprints/DISTILL-OPEN-QUESTIONS.md #10).

    mini_window.py rehearsal     [--execute]   # the first-swap round trip (business -> teacher -> business)
    mini_window.py window-start  [--execute] [--q4]
    mini_window.py window-end    [--execute]   # restore business and verify it
    mini_window.py slot-enter    [--execute]   # 02:50 / 05:50 on a Q4 night: Q4 teacher -> IQ3 teacher + Ollama
    mini_window.py slot-exit     [--execute]   # 03:15 / 06:15: back to the Q4 teacher once the evals are done
    mini_window.py teacher-failed [--execute]  # OnFailure= of the teacher units: restore business, NO-SWAPS
    mini_window.py status                      # read-only
    mini_window.py stage --ssh TARGET [--execute]   # from the repo machine: copy all of this to the host

Every command except ``status`` is a dry run unless ``--execute`` is given: read-only checks run, every
change (a setups.py apply, a dead-man timer, a unit stop, a notification) is printed instead. Runs ON the
host from ``~/distill/window/`` (the shell wrappers next to this file), stdlib only, with settings from
``window.conf`` beside it (host-specific, never committed; see ``window.conf.example``).

Track B: in a window the local teacher (``distill-teacher.service``, Qwen3.8-27B IQ3_XXS) replaces the
daemon's local model server on the same loopback port in the same shape, so the daemon, the voice probe
and the nightly evals keep a local model with no daemon config change and no paid API. Units change only
through ``setups.py apply`` (from the window's own copy in ``SETUPS_DIR``: business, flywheel,
flywheel-night-q4, this host's hosts.yaml), each after a dry run whose plan must be exactly the one
expected (nothing refused, blocked or restarted):

    rehearsal / window-start   apply flywheel           stop the local model unit; start the teacher
    window-start --q4          apply flywheel-night-q4  also stop Ollama; start the Q4 teacher (Q9)
    slot-enter (Q4 night)      apply flywheel           stop the Q4 teacher; start the teacher + Ollama
    slot-exit                  apply flywheel-night-q4  stop the teacher + Ollama; start the Q4 teacher
    window-end / any failure   apply business           stop every distill-* unit; start the local model
                                                        unit (and Ollama)

Safety:
* the dead-man restore (``distill-deadman-<host>``, ``apply business`` from the same copy) is armed
  before any swap: 45 min out for the rehearsal (cancelled again after a PASS; the round trip takes
  ~10-15 min, at most ~30), DEADMAN_HHMM for a nightly window, CONTINUOUS_UNTIL + 15 min before that.
  A timer armed for another time (a leftover) is replaced, and window-end cancels it after a verified
  restore. setups.py itself refuses a swap that stops a business unit while no dead-man is armed;
* the repo machine's hosts.yaml keeps the local model unit and Ollama never-stop; only the window copy
  that ``stage`` writes drops them, so only these scripts can swap them;
* a forward swap is ONE ``setups.py apply --attempts 1``: a teacher that does not come up is restored
  from at once, not after ~17 min of retries with the port dark;
* a teacher that crashes for good mid-window (systemd gives up) starts window-mini-teacher-failed.service
  (OnFailure=), which runs ``teacher-failed``: business restored, NO-SWAPS set, notified;
* a swap starts only right after a voice-probe result (the probe is every ~5 min and alarms on a slow or
  dark port), only in the approved hours, never while the Saluki baseline is armed or running, never
  within EVAL_GAP_MIN of a nightly eval, never with the NO-SWAPS flag present;
* an exception or SIGTERM while a swap is applied restores business before exiting; a failed restore
  writes the NO-SWAPS flag and notifies (the dead-man retries on its own);
* units are stopped only by name through systemctl or setups.py; nothing is killed by pattern.
* The transient unit that runs these scripts must NOT be named distill-*: ``apply business`` stops every
  running distill-* unit, which would stop the script running it. It needs ``-p TimeoutStopSec=20min
  -p KillMode=mixed`` (see the wrappers): the restore that a stop's SIGTERM starts takes minutes, and the
  default 90 s would SIGKILL it half done. A second signal is ignored while that restore runs.

Generation (Track C): once a window-start swap is verified, and again when slot-exit is back on the Q4
teacher, ``gen_runner.sh run --execute`` (GEN_RUNNER; GEN_ENABLE=0 turns it off) starts as the transient
unit ``distill-gen-<stamp>`` (``--collect``, KillMode=mixed, TimeoutStopSec=GEN_STOP_TIMEOUT_S), so every
``apply business`` restore and slot-enter stop it by name. Not when one is already running or the runner is
not staged; a start that fails is reported and never fails the window (gen_runner.py's own checks refuse a
port that does not serve the teacher).

Nightly evals (cron 03:00 and 06:00) are not changed: during a window they score whichever teacher is on
the port (recorded under its alias). On a Q4 night they also need Ollama (the judge) while the Q4 teacher
holds Ollama's VRAM, so the night scheduler runs ``slot-enter`` at 02:50 / 05:50 (Q4 teacher -> IQ3
teacher + Ollama; generation units matching GEN_UNIT_GLOB are stopped first, by name, if still running)
and ``slot-exit`` at 03:15 / 06:15 (waits for the eval process to finish, up to EVAL_WAIT_MAX_S and never
past SLOT_RETURN_LATEST_HHMM, then back to Q4; it acts only on tonight's slot-enter record within the
window's hours, checked again right before the swap, so a plain IQ3 window is never moved to Q4; otherwise
it stays on IQ3 until the window end).

Exit: 0 done (or nothing to do), 1 failed (notified), 2 skipped by a precondition (nothing changed),
64 usage.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import fcntl
import fnmatch
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent

DEFAULTS: dict[str, str] = {
    "HOST_KEY": "mini",
    "ET_TZ": "America/New_York",
    "PY": "/usr/bin/python3",
    "WIN_DIR": "~/distill/window",
    "SETUPS_DIR": "~/distill/window/setups",
    "LOG_DIR": "~/distill/logs/window",
    "NOTIFY": "~/distill/notify.sh",
    "MODEL_URL": "http://127.0.0.1:8092",
    "TEACHER_UNIT": "distill-teacher.service",
    "TEACHER_ALIAS": "qwen3.8-27b-gsq-rco-iq3_xxs",
    "TEACHER_Q4_UNIT": "distill-teacher-q4.service",
    "TEACHER_Q4_ALIAS": "qwen3.8-27b-q4_k_m",
    "OLLAMA_UNIT": "ollama.service",
    "OLLAMA_URL": "http://127.0.0.1:11434",
    "DAEMON_HEALTH_URL": "http://127.0.0.1:8005/health",
    "PROBE_NAME": "inference_gpu",
    "PROBE_MIN_TPS": "15",
    "SETUP_WINDOW": "flywheel",
    "SETUP_Q4": "flywheel-night-q4",
    "SETUP_RESTORE": "business",
    "GEN_UNIT_GLOB": "distill-gen-*",
    "GEN_ENABLE": "1",
    "GEN_RUNNER": "",              # empty: WIN_DIR/gen_runner.sh (staged beside this file)
    "GEN_STOP_TIMEOUT_S": "90",
    "BASELINE_UNIT_GLOB": "distill-baseline-*",
    "EVAL_CMD_MATCH": "scripts/run_nightly_evals.py",
    "EVAL_TIMES": "0300,0600",
    "EVAL_GAP_MIN": "25",
    "WINDOW_OPEN_HHMM": "0100",
    "LATEST_SWAP_HHMM": "0600",
    "CONTINUOUS_UNTIL": "",
    "DEADMAN_HHMM": "0715",
    "DEADMAN_EARLY_TOL_S": "300",
    "REHEARSAL_DEADMAN_MIN": "45",
    "SLOT_RETURN_LATEST_HHMM": "0620",
    "EVAL_WAIT_MAX_S": "1500",
    "FORWARD_APPLY_TIMEOUT_S": "900",
    "FAILED_LOCK_WAIT_S": "1200",
    "TEACHER_WAIT_S": "420",
    "VERIFY_WAIT_S": "240",
    "PROBE_WAIT_S": "660",
    "TICK_WAIT_S": "330",
    "PROC_ROOT": "/proc",
}
# Site-specific names: only window.conf has them.
REQUIRED = ("LOCAL_MODEL_UNIT", "PROBE_UNIT", "OLLAMA_MODEL")
ACTIVE = ("active", "activating", "reloading")
FLAG_NAME = "NO-SWAPS-TONIGHT"

STAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:?\d{2}|Z))\s")
VERDICT_RE = re.compile(r"\bverdict=([A-Za-z_]+)")
TPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*tok/s")
FLOOR_RE = re.compile(r"\(floor (\d+(?:\.\d+)?)\)")
PLAN_RE = re.compile(r"^  (\S+): stop (.+?); start (.+?); keep (.+?)(?:; restart (.+))?$")


class Skip(Exception):
    """A precondition is not met: nothing was changed (exit 2)."""


class Fail(Exception):
    """A step failed (exit 1)."""


# ── settings ─────────────────────────────────────────────────────────────────


def load_conf(path: Path, home: str | None = None) -> dict[str, str]:
    """KEY=VALUE lines (shell-style, so bash can source the same file); ``~`` and ``$HOME`` expand."""
    home = home or os.path.expanduser("~")
    conf = dict(DEFAULTS)
    try:
        text = path.read_text()
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}") from None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, eq, value = line.partition("=")
        if not eq or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise SystemExit(f"{path}:{n}: not KEY=VALUE")
        try:
            words = shlex.split(value, comments=True)
        except ValueError as e:
            raise SystemExit(f"{path}:{n}: {e}") from None
        conf[key] = " ".join(words)
    for k, v in conf.items():
        v = v.replace("${HOME}", home).replace("$HOME", home)
        conf[k] = home + v[1:] if v == "~" or v.startswith("~/") else v
    missing = [k for k in REQUIRED if not conf.get(k)]
    if missing:
        raise SystemExit(f"{path}: set {', '.join(missing)}")
    return conf


def hhmm(text: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d{4}", text):
        raise SystemExit(f"bad HHMM {text!r}")
    return int(text[:2]), int(text[2:])


# ── probe lines ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Probe:
    t: float
    verdict: str
    tps: float | None
    floor: float | None

    def ok(self, min_tps: float) -> bool:
        floor = max(min_tps, self.floor or 0.0)
        return self.verdict.upper() == "OK" and self.tps is not None and self.tps >= floor

    def describe(self) -> str:
        return f"{self.verdict}" + (f" {self.tps:g} tok/s" if self.tps is not None else "")


def line_time(line: str) -> float | None:
    m = STAMP_RE.match(line)
    if not m:
        return None
    text = m.group(1).replace("Z", "+00:00")
    if re.search(r"[+-]\d{4}$", text):
        text = text[:-2] + ":" + text[-2:]
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def parse_probes(lines: list[str], name: str) -> list[Probe]:
    """The voice probe's result lines (``... <name> ... verdict=X — N tok/s (floor F)``), oldest first."""
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    out = []
    for line in lines:
        if not pat.search(line):
            continue
        t = line_time(line)
        v = VERDICT_RE.search(line)
        if t is None or not v:
            continue
        tps = TPS_RE.search(line)
        floor = FLOOR_RE.search(line)
        out.append(Probe(t, v.group(1), float(tps.group(1)) if tps else None,
                         float(floor.group(1)) if floor else None))
    return sorted(out, key=lambda p: p.t)


def parse_plan(text: str, host: str) -> dict[str, Any]:
    """A ``setups.py apply`` dry run's plan for one host: lists, plus every refusal/blocker/note line."""
    plan: dict[str, Any] = {"found": False, "stop": [], "start": [], "keep": [], "restart": [], "problems": []}
    for line in text.splitlines():
        m = PLAN_RE.match(line)
        if m and m.group(1) == host:
            plan["found"] = True
            for key, raw in zip(("stop", "start", "keep", "restart"), m.groups()[1:]):
                plan[key] = [] if raw in (None, "-") else list(ast.literal_eval(raw))
        elif re.match(rf"^  {re.escape(host)}: (REFUSED|BLOCKED|note)\b", line) or line.startswith("REFUSED"):
            plan["problems"].append(line.strip())
    return plan


# ── the window ───────────────────────────────────────────────────────────────


class Window:
    def __init__(self, conf: dict[str, str], execute: bool, name: str,
                 runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 out: Callable[[str], None] | None = None) -> None:
        self.c = conf
        self.execute = execute
        self.name = name
        self._runner = runner or _subprocess_run
        self.clock = clock
        self._sleep = sleep
        self.tz = ZoneInfo(conf["ET_TZ"])
        self.win = Path(conf["WIN_DIR"])
        self.state = self.win / "state"
        self.flag = self.win / FLAG_NAME
        self.swap_pending = False
        self._out = out or (lambda s: print(s, flush=True))
        self.log_file: Path | None = None

    # logging, commands, http

    def log(self, msg: str) -> None:
        line = f"{self.et().strftime('%F %T %Z')} {self.name}: {msg}"
        self._out(line)
        if self.log_file is not None:
            try:
                with open(self.log_file, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def start_log(self) -> None:
        if not self.execute:
            return
        d = Path(self.c["LOG_DIR"])
        d.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)
        for p in (d, self.state):
            with contextlib.suppress(OSError):
                os.chmod(p, 0o700)
        self.log_file = d / f"{self.name}-{self.et().strftime('%F')}.log"
        self.log(f"start (pid {os.getpid()}, log {self.log_file})")

    def run(self, argv: list[str], timeout: float = 60, input: str | None = None) -> subprocess.CompletedProcess[str]:
        return self._runner(argv, timeout=timeout, input=input)

    def act(self, argv: list[str], timeout: float = 120, input: str | None = None) -> subprocess.CompletedProcess[str]:
        """A command that changes something: printed on a dry run, run and logged otherwise."""
        if not self.execute:
            self.log(f"[dry run] would run: {shlex.join(argv)}")
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.log(f"run: {shlex.join(argv)}")
        return self.run(argv, timeout=timeout, input=input)

    def sleep(self, s: float) -> None:
        self._sleep(max(0.0, s))

    def now(self) -> float:
        return self.clock()

    def et(self, t: float | None = None) -> datetime:
        return datetime.fromtimestamp(self.now() if t is None else t, self.tz)

    def http(self, url: str, timeout: float = 5.0) -> tuple[int, Any]:
        try:
            # Loopback only: never through an http_proxy from the environment.
            with _OPENER.open(url, timeout=timeout) as r:
                body = r.read(1 << 20)
                status = r.status
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception:  # noqa: BLE001 - unreachable is a status here, not an error
            return 0, None
        try:
            return status, json.loads(body.decode("utf-8", "replace"))
        except ValueError:
            return status, None

    # reads

    def unit_prop(self, unit: str, prop: str) -> str:
        cp = self.run(["systemctl", "--user", "show", "-p", prop, "--value", unit], timeout=20)
        return cp.stdout.strip() if cp.returncode == 0 else ""

    def parse_stamp(self, text: str) -> float | None:
        """A systemd timestamp (``Sat 2026-10-31 07:15:00 EDT``) as epoch seconds, in this host's zone."""
        m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\S+)$", text.strip())
        if not m:
            return None
        naive = datetime.fromisoformat(m.group(1))
        if m.group(2) in ("UTC", "GMT"):
            return naive.replace(tzinfo=ZoneInfo("UTC")).timestamp()
        for fold in (0, 1):
            dt = naive.replace(tzinfo=self.tz, fold=fold)
            if dt.tzname() == m.group(2):
                return dt.timestamp()
        return None

    def active(self, unit: str) -> bool:
        return self.unit_prop(unit, "ActiveState") in ACTIVE

    def units_matching(self, glob: str, kind: str = "service") -> dict[str, str]:
        """Loaded units of ``kind`` matching ``glob`` -> ACTIVE state (sub-state 'waiting' for a timer)."""
        cp = self.run(["systemctl", "--user", "list-units", "--all", "--plain", "--no-legend", "--no-pager",
                       f"--type={kind}", glob], timeout=20)
        out = {}
        for line in cp.stdout.splitlines():
            parts = line.replace("●", " ").split()
            if len(parts) >= 4 and parts[0].endswith("." + kind) and fnmatch.fnmatch(parts[0], glob + "*"):
                out[parts[0]] = parts[2] if parts[2] != "active" or kind != "timer" else parts[3]
        return out

    def served(self) -> dict[str, Any]:
        """What answers on the shared model port: /health status and the /v1/models id (with aliases)."""
        base = self.c["MODEL_URL"].rstrip("/")
        health, _ = self.http(base + "/health")
        st, body = self.http(base + "/v1/models")
        ids: list[str] = []
        if st == 200 and isinstance(body, dict):
            for m in body.get("data") or []:
                if isinstance(m, dict):
                    ids += [str(m.get("id") or "")] + [str(a) for a in m.get("aliases") or []]
        return {"health": health, "id": ids[0] if ids else None, "ids": [i for i in ids if i]}

    def ollama_has_model(self) -> bool:
        st, body = self.http(self.c["OLLAMA_URL"].rstrip("/") + "/api/tags")
        names = [m.get("name") for m in (body or {}).get("models", [])] if st == 200 and isinstance(body, dict) else []
        return self.c["OLLAMA_MODEL"] in names

    def ollama_fit(self) -> tuple[bool | None, str]:
        """Whether OLLAMA_MODEL is loaded fully on the GPU (/api/ps ``size_vram`` == ``size``); None when it
        is not loaded or Ollama does not answer. Beside the IQ3 teacher a RELOAD of the judge has only a
        few MiB to spare (distill-teacher.service), so a reload may leave layers on the CPU."""
        model = self.c["OLLAMA_MODEL"]
        st, body = self.http(self.c["OLLAMA_URL"].rstrip("/") + "/api/ps")
        if st != 200 or not isinstance(body, dict):
            return None, f"Ollama /api/ps {st or 'unreachable'}"
        for m in body.get("models") or []:
            if isinstance(m, dict) and model in (m.get("name"), m.get("model")):
                try:
                    size, vram = int(m.get("size") or 0), int(m.get("size_vram") or 0)
                except (TypeError, ValueError):
                    return None, f"{model}: unreadable /api/ps sizes"
                mib = 1 << 20
                if size and vram >= size:
                    return True, f"{model} fully on the GPU ({vram / mib:.0f} MiB)"
                return False, f"{model} PARTLY ON THE CPU ({vram / mib:.0f} of {size / mib:.0f} MiB on the GPU)"
        return None, f"{model} not loaded"

    def daemon_ok(self) -> tuple[bool, str]:
        st, body = self.http(self.c["DAEMON_HEALTH_URL"])
        ok = st == 200 and isinstance(body, dict) and body.get("status") == "ok"
        return ok, f"daemon /health {st or 'unreachable'}" + (f" {body.get('status')}" if isinstance(body, dict) else "")

    def business_model_id(self) -> str | None:
        """The model path the local model unit's ExecStart loads (what its /v1/models id is)."""
        text = self.unit_prop(self.c["LOCAL_MODEL_UNIT"], "ExecStart")
        m = re.search(r"argv\[\]=([^;]*)", text)
        toks = (m.group(1) if m else "").split()
        for i, t in enumerate(toks[:-1]):
            if t in ("-m", "--model"):
                return toks[i + 1]
        return None

    def probes(self, since_s: int = 900) -> list[Probe]:
        cp = self.run(["journalctl", "--user", "-u", self.c["PROBE_UNIT"], "--since", f"-{since_s}s",
                       "-o", "short-iso-precise", "-q", "--no-pager"], timeout=30)
        if cp.returncode != 0:
            raise Fail(f"cannot read the probe journal ({self.c['PROBE_UNIT']}): exit {cp.returncode}")
        return parse_probes(cp.stdout.splitlines(), self.c["PROBE_NAME"])

    def eval_running(self) -> bool:
        needle = self.c["EVAL_CMD_MATCH"].encode()
        root = Path(self.c["PROC_ROOT"])
        try:
            entries = [p for p in root.iterdir() if p.name.isdigit()]
        except OSError:
            return False
        for p in entries:
            with contextlib.suppress(OSError):
                if needle in (p / "cmdline").read_bytes().replace(b"\0", b" "):
                    return True
        return False

    # time rules

    def at_today(self, text: str, base: datetime | None = None) -> datetime:
        h, m = hhmm(text)
        base = base or self.et()
        return base.replace(hour=h, minute=m, second=0, microsecond=0)

    def continuous(self) -> bool:
        until = self.c.get("CONTINUOUS_UNTIL", "").strip()
        if not until:
            return False
        dt = datetime.fromisoformat(until)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return self.now() < dt.timestamp()

    def in_night_hours(self) -> bool:
        now = self.et()
        return self.at_today(self.c["WINDOW_OPEN_HHMM"]) <= now < self.at_today(self.c["LATEST_SWAP_HHMM"])

    def in_slot_return_hours(self) -> bool:
        """When slot-exit may still bring the Q4 teacher back: tonight's window open up to the cutoff."""
        now = self.et()
        return self.at_today(self.c["WINDOW_OPEN_HHMM"]) <= now < self.at_today(self.c["SLOT_RETURN_LATEST_HHMM"])

    def slot_entered_at(self) -> float | None:
        """When slot-enter swapped tonight's Q4 teacher out (``state/slot.active``, epoch first), or None: no
        record, an unreadable one, or one from before tonight's window opened (a leftover)."""
        try:
            t = float((self.state / "slot.active").read_text().split()[0])
        except (OSError, IndexError, ValueError):
            return None
        return t if self.at_today(self.c["WINDOW_OPEN_HHMM"]).timestamp() <= t <= self.now() else None

    def clear_slot(self) -> None:
        if self.execute:
            with contextlib.suppress(OSError):
                (self.state / "slot.active").unlink()

    def minutes_to_next_eval(self) -> float:
        now = self.et()
        best = None
        for t in self.c["EVAL_TIMES"].split(","):
            at = self.at_today(t.strip())
            if at <= now:
                at += timedelta(days=1)
            mins = (at - now).total_seconds() / 60
            best = mins if best is None else min(best, mins)
        return best if best is not None else 1e9

    # actions

    def setups_argv(self, *args: str) -> list[str]:
        d = self.c["SETUPS_DIR"]
        return [self.c["PY"], f"{d}/setups.py", "--setups-dir", d, "--hosts-file", f"{d}/hosts.yaml",
                "--local", "--only-host", self.c["HOST_KEY"], *args]

    def plan(self, setup: str) -> dict[str, Any]:
        cp = self.run(self.setups_argv("apply", setup), timeout=600)
        text = (cp.stdout or "") + (cp.stderr or "")
        for line in text.splitlines():
            self.log(f"    {line}")
        plan = parse_plan(text, self.c["HOST_KEY"])
        plan["rc"] = cp.returncode
        return plan

    def expect_plan(self, setup: str, stop: set[str], start: set[str], *, at_most: bool = False,
                    exc: type[Exception] = Skip) -> None:
        """The dry-run plan must be exactly ``stop`` / ``start`` (with ``at_most``: no more than them), with
        nothing refused, blocked or restarted. Otherwise raises ``exc``."""
        p = self.plan(setup)
        why = []
        if p["rc"] != 0:
            why.append(f"dry run exited {p['rc']}")
        if not p["found"]:
            why.append("no plan line")
        for key, want in (("stop", stop), ("start", start)):
            got = p[key]
            if (not set(got) <= want) if at_most else (set(got) != want or len(got) != len(want)):
                why.append(f"{key} {got} (want {'at most ' if at_most else ''}{sorted(want)})")
        if p["restart"]:
            why.append(f"restart {p['restart']}")
        why += p["problems"]
        if why:
            raise exc(f"unexpected {setup} plan: " + "; ".join(why))
        self.log(f"{setup} plan as expected: stop {sorted(p['stop'])}, start {sorted(p['start'])}")

    def apply(self, setup: str) -> int:
        """``setups.py apply <setup> --execute``. A forward swap (anything but the restore setup) makes ONE
        attempt: if its teacher does not come up, the caller restores at once instead of setups.py retrying
        twice more with the port dark (~17 min). The restore keeps setups.py's retries."""
        forward = setup != self.c["SETUP_RESTORE"]
        argv = self.setups_argv("apply", setup, "--execute", *(("--attempts", "1") if forward else ()))
        if not self.execute:
            self.log(f"[dry run] would run: {shlex.join(argv)}")
            return 0
        self.log(f"apply {setup} --execute")
        cp = self.run(argv, timeout=float(self.c["FORWARD_APPLY_TIMEOUT_S"]) if forward else 2400)
        for line in ((cp.stdout or "") + (cp.stderr or "")).splitlines():
            self.log(f"    {line}")
        self.log(f"apply {setup} exited {cp.returncode}")
        return cp.returncode

    def notify(self, text: str) -> None:
        msg = "\n".join(text.strip().splitlines()[:5]) + "\n"
        if not self.execute:
            self.log("[dry run] would notify: " + " | ".join(msg.strip().splitlines()))
            return
        try:
            cp = self.run([self.c["NOTIFY"], "5"], timeout=60, input=msg)
            self.log("notify " + ("sent" if cp.returncode == 0 else f"FAILED (exit {cp.returncode})"))
        except Exception as e:  # noqa: BLE001 - a broken notifier must not hide the result
            self.log(f"notify FAILED: {type(e).__name__}: {e}")

    def write_flag(self, why: str) -> None:
        if not self.execute:
            self.log(f"[dry run] would write {self.flag} ({why})")
            return
        self.flag.write_text(f"{self.et().strftime('%F %T %Z')} {self.name}: {why}\n"
                             "Remove this file to allow swaps again.\n")
        self.log(f"wrote {self.flag}: {why}")

    def record(self, name: str, text: str) -> None:
        if self.execute:
            (self.state / name).write_text(text.rstrip("\n") + "\n")

    @contextlib.contextmanager
    def lock(self, wait_s: float) -> Iterator[bool]:
        """One window script at a time. Yields False if another one still holds it after ``wait_s``."""
        if not self.execute:
            yield True
            return
        fd = os.open(self.state / "window.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            deadline = self.now() + wait_s
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    got = True
                    break
                except OSError:
                    if self.now() >= deadline:
                        got = False
                        break
                    self.sleep(2)
            yield got
        finally:
            os.close(fd)

    # waits

    def wait_tick(self) -> None:
        """Wait until just after a voice-probe result, so a swap has ~4 min before the next one."""
        limit = self.now() + float(self.c["TICK_WAIT_S"])
        if not self.execute:
            ps = self.probes()
            age = self.now() - ps[-1].t if ps else None
            self.log("[dry run] would wait for the next voice-probe result (+8 s) before swapping"
                     + (f" (last one {age:.0f} s ago)" if age is not None else " (none in 15 min)"))
            return
        while True:
            ps = self.probes()
            if not ps or self.now() - ps[-1].t > 660:
                self.log("no voice-probe result in 11 min: nothing to align the swap to; going on")
                return
            age = self.now() - ps[-1].t
            if 8 <= age <= 150:
                self.log(f"swapping {age:.0f} s after a voice-probe result ({ps[-1].describe()})")
                return
            if self.now() >= limit:
                self.log(f"no fresh voice-probe tick within {self.c['TICK_WAIT_S']} s; going on")
                return
            self.sleep(8 - age if age < 8 else 5)

    def wait_served(self, alias: str, wait_s: float) -> float:
        """Until the port answers /health 200 with ``alias`` in /v1/models. Returns the time it did."""
        if not self.execute:
            self.log(f"[dry run] would wait up to {wait_s:.0f} s for {self.c['MODEL_URL']} to serve {alias}")
            return self.now()
        deadline = self.now() + wait_s
        while True:
            s = self.served()
            if s["health"] == 200 and alias in s["ids"]:
                self.log(f"{self.c['MODEL_URL']} healthy, serving {alias}")
                return self.now()
            if self.now() >= deadline:
                raise Fail(f"{self.c['MODEL_URL']} not serving {alias} after {wait_s:.0f} s "
                           f"(health {s['health']}, id {s['id']})")
            self.sleep(5)

    def wait_probe_ok(self, after: float, wait_s: float) -> Probe:
        """The first voice-probe result after ``after`` must be OK and fast enough."""
        if not self.execute:
            self.log(f"[dry run] would wait up to {wait_s:.0f} s for the next voice-probe result to be OK "
                     f"(>= {self.c['PROBE_MIN_TPS']} tok/s)")
            return Probe(self.now(), "OK", None, None)
        deadline = self.now() + wait_s
        while True:
            fresh = [p for p in self.probes(int(self.now() - after) + 120) if p.t > after + 1]
            if fresh:
                p = fresh[0]
                if p.ok(float(self.c["PROBE_MIN_TPS"])):
                    self.log(f"voice probe OK after the change: {p.describe()}")
                    return p
                raise Fail(f"voice probe after the change: {p.describe()}")
            if self.now() >= deadline:
                raise Fail(f"no voice-probe result within {wait_s:.0f} s after the change")
            self.sleep(10)

    # dead-man

    def deadman_unit(self) -> str:
        return f"distill-deadman-{self.c['HOST_KEY']}"

    def deadman_next(self) -> tuple[str, float | None]:
        """(state, next fire epoch). State: running (restoring now), waiting (armed), none (no timer
        loaded), or the timer's ActiveState/SubState when it is loaded but cannot fire (elapsed, failed)."""
        u = self.deadman_unit()
        svc = self.unit_prop(u + ".service", "ActiveState")
        if svc in ("active", "activating", "deactivating", "reloading"):
            return "running", None
        if self.unit_prop(u + ".timer", "LoadState") != "loaded":
            return "none", None
        active = self.unit_prop(u + ".timer", "ActiveState")
        sub = self.unit_prop(u + ".timer", "SubState")
        if active == "active" and sub == "waiting":
            return "waiting", self.parse_stamp(self.unit_prop(u + ".timer", "NextElapseUSecRealtime"))
        return f"{active or '?'}/{sub or '?'}", None

    def ensure_deadman(self, target: float, slack_s: float = 900) -> bool:
        """Armed for ``target``: a waiting timer counts only if it fires from DEADMAN_EARLY_TOL_S before
        ``target`` up to ``slack_s`` after it. One armed for any other time is a leftover (an earlier window
        or a failed rehearsal; callers run after preflight, so no teacher runs) and is replaced: an earlier
        one would restore business in the middle of this swap, a later one would not end it.
        Returns True if this call armed it."""
        u = self.deadman_unit()
        state, nxt = self.deadman_next()
        when = self.et(target).strftime("%F %H:%M %Z")
        if state == "running":
            raise Skip(f"the dead-man restore {u}.service is running now")
        early = target - float(self.c["DEADMAN_EARLY_TOL_S"])
        if state == "waiting" and nxt is not None and max(self.now(), early) < nxt <= target + slack_s:
            self.log(f"dead-man already armed: {u}.timer fires {self.et(nxt).strftime('%F %H:%M %Z')}")
            return False
        if state == "waiting":
            fires = self.et(nxt).strftime("%F %H:%M %Z") if nxt else "?"
            self.log(f"dead-man {u}.timer is armed for {fires}, not {when}: replacing it")
            self.act(["systemctl", "--user", "stop", u + ".timer"], timeout=30)
        else:
            self.log(f"arming the dead-man restore for {when}")
        if state not in ("none", "waiting") and not state.startswith("inactive/"):
            self.act(["systemctl", "--user", "stop", u + ".timer"])  # elapsed or failed: it cannot fire again
        self.act(["systemctl", "--user", "reset-failed", u + ".timer", u + ".service"], timeout=30)
        cal = datetime.fromtimestamp(target, ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S UTC")
        cp = self.act(["systemd-run", "--user", f"--unit={u}", f"--on-calendar={cal}",
                       "--timer-property=AccuracySec=1s",
                       f"--description=distill dead-man: apply {self.c['SETUP_RESTORE']}",
                       *self.setups_argv("apply", self.c["SETUP_RESTORE"], "--execute")], timeout=60)
        if not self.execute:
            return True
        state, nxt = self.deadman_next()
        if cp.returncode != 0 or state != "waiting":
            raise Skip(f"the dead-man timer did not arm (systemd-run exit {cp.returncode}, timer {state})")
        self.record("deadman.armed", f"{u}.timer {cal}")
        self.log(f"dead-man armed: {u}.timer at {cal}")
        return True

    def cancel_deadman(self) -> None:
        u = self.deadman_unit()
        self.act(["systemctl", "--user", "stop", u + ".timer"], timeout=30)
        self.act(["systemctl", "--user", "reset-failed", u + ".timer", u + ".service"], timeout=30)
        if self.execute:
            with contextlib.suppress(OSError):
                (self.state / "deadman.armed").unlink()

    # checks

    def preflight(self, setup: str, unit: str) -> None:
        for f in (Path(self.c["SETUPS_DIR"]) / n for n in ("setups.py", "hosts.yaml", f"{setup}.yaml",
                                                           f"{self.c['SETUP_RESTORE']}.yaml")):
            if not f.is_file():
                raise Skip(f"missing {f} (stage the window files first)")
        if not os.access(self.c["NOTIFY"], os.X_OK):
            raise Skip(f"notify command {self.c['NOTIFY']} is not executable")
        if self.run([self.c["PY"], "-c", "import yaml"], timeout=30).returncode != 0:
            raise Skip(f"{self.c['PY']} cannot import yaml (setups.py needs it)")
        user = self.run(["id", "-un"], timeout=10).stdout.strip()
        linger = self.run(["loginctl", "show-user", user, "-p", "Linger", "--value"], timeout=10).stdout.strip()
        if linger != "yes":
            raise Skip(f"linger is {linger or '?'} for {user}: the dead-man timer would not run without a login")
        if self.unit_prop(unit, "LoadState") != "loaded":
            raise Skip(f"{unit} is not installed (scripts/distill/units/install_teacher_units.sh)")
        if self.flag.exists():
            first = (self.flag.read_text().splitlines() or ["(empty)"])[0]
            raise Skip(f"{self.flag} is present: {first}")
        glob = self.c["BASELINE_UNIT_GLOB"]
        busy = {**{u: s for u, s in self.units_matching(glob, "timer").items() if s in ("waiting", "running")},
                **{u: s for u, s in self.units_matching(glob, "service").items() if s in ACTIVE}}
        if busy:
            raise Skip("the Saluki baseline is still armed or running ("
                       + ", ".join(f"{u} {s}" for u, s in sorted(busy.items())) + "): swap after it ends")
        for u in (self.c["TEACHER_UNIT"], self.c["TEACHER_Q4_UNIT"]):
            if self.active(u):
                raise Skip(f"{u} is already running: a window is open")

    def hours_check(self, q4: bool = False) -> None:
        if q4 and not self.in_night_hours():
            raise Skip(f"the Q4 teacher runs only in the nightly window ({self.c['WINDOW_OPEN_HHMM']}-"
                       f"{self.c['LATEST_SWAP_HHMM']} ET, Q9)")
        if not q4 and not (self.continuous() or self.in_night_hours()):
            raise Skip(f"outside the approved swap hours (now {self.et().strftime('%H:%M %Z')})")
        if self.eval_running():
            raise Skip("a nightly eval is running")
        mins = self.minutes_to_next_eval()
        if mins < float(self.c["EVAL_GAP_MIN"]):
            raise Skip(f"a nightly eval starts in {mins:.0f} min")

    def slot_exit_blocker(self) -> str | None:
        """Why slot-exit must not bring the Q4 teacher back now; None if it may. Checked on entry AND right
        before the swap (after the eval wait and the probe tick): the IQ3 layout alone does not mean a Q4
        night (a plain IQ3 window looks the same), so it needs tonight's slot-enter record and the hours."""
        c = self.c
        q4, iq3, ollama = (self.active(c["TEACHER_Q4_UNIT"]), self.active(c["TEACHER_UNIT"]),
                           self.active(c["OLLAMA_UNIT"]))
        if not (iq3 and ollama and not q4):
            return "not in the eval-slot layout (teacher + Ollama, no Q4 teacher)"
        if self.flag.exists():
            return f"{self.flag} is present"
        if self.slot_entered_at() is None:
            return "no slot-enter record from tonight (state/slot.active): not a Q4 night"
        if not self.in_slot_return_hours():
            return (f"outside {c['WINDOW_OPEN_HHMM']}-{c['SLOT_RETURN_LATEST_HHMM']} ET, when the Q4 teacher may "
                    "come back")
        return None

    def window_closed(self) -> str | None:
        """Whether the window ended while a script waited (window-end restoring past the lock): a fallback
        swap must not stop the local model server again."""
        if self.active(self.c["LOCAL_MODEL_UNIT"]):
            return f"{self.c['LOCAL_MODEL_UNIT']} is running again (the window was closed)"
        if self.flag.exists():
            return f"{self.flag} is present"
        if self.execute and self.slot_entered_at() is None:
            return "state/slot.active is gone (a restore ran)"
        return None

    def business_problems(self) -> list[str]:
        c = self.c
        out = []
        if not self.active(c["LOCAL_MODEL_UNIT"]):
            out.append(f"{c['LOCAL_MODEL_UNIT']} is {self.unit_prop(c['LOCAL_MODEL_UNIT'], 'ActiveState') or '?'}")
        for u in (c["TEACHER_UNIT"], c["TEACHER_Q4_UNIT"]):
            if self.active(u):
                out.append(f"{u} still runs")
        s = self.served()
        want = self.business_model_id()
        if s["health"] != 200:
            out.append(f"{c['MODEL_URL']}/health {s['health'] or 'unreachable'}")
        elif want and want not in s["ids"]:
            out.append(f"{c['MODEL_URL']} serves {s['id']!r}, not {want!r}")
        elif {c["TEACHER_ALIAS"], c["TEACHER_Q4_ALIAS"]} & set(s["ids"]):
            out.append(f"{c['MODEL_URL']} still serves the teacher ({s['id']})")
        ok, detail = self.daemon_ok()
        if not ok:
            out.append(detail)
        if not self.active(c["OLLAMA_UNIT"]):
            out.append(f"{c['OLLAMA_UNIT']} is not running")
        elif not self.ollama_has_model():
            out.append(f"Ollama does not list {c['OLLAMA_MODEL']}")
        return out

    def wait_business(self, wait_s: float) -> list[str]:
        deadline = self.now() + wait_s
        while True:
            problems = self.business_problems()
            if not problems or self.now() >= deadline:
                return problems
            self.sleep(5)

    def restore(self) -> tuple[bool, str]:
        """apply business + verify (local model healthy and serving its model, daemon /health, Ollama
        listing the judge model, the next voice probe OK); up to 3 rounds."""
        setup = self.c["SETUP_RESTORE"]
        if not self.execute:
            self.log(f"restore plan now (dry run of apply {setup}):")
            self.plan(setup)
            self.apply(setup)
            self.log("[dry run] then verify: local model unit active, /health 200 serving its own model, "
                     "daemon /health ok, Ollama lists " + self.c["OLLAMA_MODEL"] + ", next voice probe OK")
            return True, "(dry run)"
        problems: list[str] = []
        for rnd in (1, 2, 3):
            self.log(f"restore round {rnd}/3")
            self.apply(setup)
            t_done = self.now()
            problems = self.wait_business(float(self.c["VERIFY_WAIT_S"]))
            if not problems:
                try:
                    p = self.wait_probe_ok(t_done, float(self.c["PROBE_WAIT_S"]))
                    self.clear_slot()  # no eval slot outlives the window it was in
                    return True, f"business restored (round {rnd}); voice probe {p.describe()}"
                except Fail as e:
                    problems = [str(e)]
            self.log("restore not verified: " + "; ".join(problems))
            if rnd < 3:
                self.sleep(30)
        return False, "restore FAILED: " + "; ".join(problems)[:300]

    # guarded swaps

    @contextlib.contextmanager
    def swapped(self) -> Iterator[None]:
        """While a swap is applied: any exception or SIGTERM restores business before leaving."""
        self.swap_pending = True
        try:
            yield
        except BaseException as e:
            if self.swap_pending and self.execute:
                self.swap_pending = False
                why = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                self.log(f"failed with the swap applied ({why}): restoring business")
                ok, line = self.restore()
                if ok:
                    self.notify(f"mini {self.name} FAILED ({why[:150]}); business restored.\n{line}")
                else:
                    self.write_flag(f"{self.name} failed and the restore failed")
                    self.notify(f"mini RESTORE FAILED after {self.name} failed ({why[:120]}): the daemon's local "
                                f"model may be DOWN.\n{line}\nNO-SWAPS flag set; the dead-man retries.")
            raise
        finally:
            self.swap_pending = False


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _subprocess_run(argv: list[str], timeout: float = 60, input: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, input=input)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(argv, 124, "", f"timeout: {e}")
    except OSError as e:
        return subprocess.CompletedProcess(argv, 127, "", str(e))


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_status(w: Window) -> int:
    c = w.c
    s = w.served()
    w.log(f"port {c['MODEL_URL']}: health {s['health'] or 'unreachable'}, id {s['id']}")
    for u in (c["LOCAL_MODEL_UNIT"], c["TEACHER_UNIT"], c["TEACHER_Q4_UNIT"], c["OLLAMA_UNIT"]):
        w.log(f"  {u}: {w.unit_prop(u, 'ActiveState') or '?'} ({w.unit_prop(u, 'LoadState') or '?'})")
    w.log(f"  Ollama lists {c['OLLAMA_MODEL']}: {w.ollama_has_model()}; loaded: {w.ollama_fit()[1]}")
    w.log(f"  {w.daemon_ok()[1]}")
    ps = w.probes()
    w.log("  last voice probe: " + (f"{ps[-1].describe()} {w.now() - ps[-1].t:.0f} s ago" if ps else "none in 15 min"))
    state, nxt = w.deadman_next()
    w.log(f"  dead-man: {state}" + (f", fires {w.et(nxt).strftime('%F %H:%M %Z')}" if nxt else ""))
    w.log(f"  nightly eval running: {w.eval_running()}; NO-SWAPS flag: {w.flag.exists()}")
    return 0


def cmd_rehearsal(w: Window) -> int:
    """business -> teacher (healthy + next voice probe OK) -> business (verified), with a dead-man armed."""
    c = w.c
    with w.lock(300) as got:
        if not got:
            raise Skip("another window script holds the lock")
        w.preflight(c["SETUP_WINDOW"], c["TEACHER_UNIT"])
        w.hours_check()
        problems = w.business_problems()
        if problems:
            raise Skip("business is not healthy before the swap: " + "; ".join(problems))
        w.expect_plan(c["SETUP_WINDOW"], {c["LOCAL_MODEL_UNIT"]}, {c["TEACHER_UNIT"]})
        mine = w.ensure_deadman(w.now() + 60 * float(c["REHEARSAL_DEADMAN_MIN"]), slack_s=0)
        w.wait_tick()
        with w.swapped():
            rc = w.apply(c["SETUP_WINDOW"])
            if rc != 0:
                raise Fail(f"apply {c['SETUP_WINDOW']} exited {rc}")
            t_up = w.wait_served(c["TEACHER_ALIAS"], float(c["TEACHER_WAIT_S"]))
            p = w.wait_probe_ok(t_up, float(c["PROBE_WAIT_S"]))
            daemon = w.daemon_ok()[1]
            fit, judge = w.ollama_fit()
            teacher_line = (f"teacher {c['TEACHER_ALIAS']} up, voice probe {p.describe()}, {daemon}, judge {judge}"
                            + (" (WARNING: nightly evals run slower)" if fit is False else ""))
            w.wait_tick()
            w.swap_pending = False
            ok, line = w.restore()
        if not ok:
            w.write_flag("rehearsal restore failed")
            w.record("rehearsal.last", f"FAIL {w.et().strftime('%F %T %Z')} {line}")
            w.notify(f"mini rehearsal FAIL: {line}\n{teacher_line}\nNO-SWAPS flag set; the dead-man retries.")
            return 1
        if mine:
            w.cancel_deadman()
        w.record("rehearsal.last", f"PASS {w.et().strftime('%F %T %Z')} {teacher_line}; {line}")
        w.notify(f"mini rehearsal PASS\n{teacher_line}\n{line}")
        return 0


def cmd_window_start(w: Window, q4: bool) -> int:
    c = w.c
    setup, unit, alias = ((c["SETUP_Q4"], c["TEACHER_Q4_UNIT"], c["TEACHER_Q4_ALIAS"]) if q4
                          else (c["SETUP_WINDOW"], c["TEACHER_UNIT"], c["TEACHER_ALIAS"]))
    with w.lock(300) as got:
        if not got:
            raise Skip("another window script holds the lock")
        w.preflight(setup, unit)
        w.hours_check(q4=q4)
        last = (w.state / "rehearsal.last").read_text().split() if (w.state / "rehearsal.last").exists() else []
        if not last or last[0] != "PASS":
            if w.execute:
                raise Skip("no PASSED rehearsal (state/rehearsal.last): run the rehearsal first")
            w.log("[dry run] note: no PASSED rehearsal yet; a real run stops here")
        problems = w.business_problems()
        if problems:
            raise Skip("business is not healthy before the swap: " + "; ".join(problems))
        stop = {c["LOCAL_MODEL_UNIT"]} | ({c["OLLAMA_UNIT"]} if q4 else set())
        w.expect_plan(setup, stop, {unit})
        if w.continuous() and not q4:
            until = datetime.fromisoformat(c["CONTINUOUS_UNTIL"])
            until = until if until.tzinfo else until.replace(tzinfo=w.tz)
            target = until.timestamp() + 900
        else:
            target = w.at_today(c["DEADMAN_HHMM"]).timestamp()
            if target <= w.now():
                target += 86400
        w.ensure_deadman(target)
        w.wait_tick()
        with w.swapped():
            rc = w.apply(setup)
            if rc != 0:
                raise Fail(f"apply {setup} exited {rc}")
            t_up = w.wait_served(alias, float(c["TEACHER_WAIT_S"]))
            p = w.wait_probe_ok(t_up, float(c["PROBE_WAIT_S"]))
            w.swap_pending = False
        judge = "" if q4 else f"; judge {w.ollama_fit()[1]}"
        w.record("window.started", f"{w.et().strftime('%F %T %Z')} {setup} {alias}")
        gen = start_gen(w, "window-start")
        w.notify(f"mini window open: {alias} on the local model port ({setup}); voice probe {p.describe()}{judge}"
                 f"\ngeneration: {gen}")
        return 0


def cmd_window_end(w: Window) -> int:
    c = w.c
    with w.lock(600) as got:
        if not got:
            w.log("lock still held after 600 s: restoring anyway")
        teacher = [u for u in (c["TEACHER_UNIT"], c["TEACHER_Q4_UNIT"]) if w.active(u)]
        baseline = {u: s for u, s in w.units_matching(c["BASELINE_UNIT_GLOB"], "service").items() if s in ACTIVE}
        if baseline and not teacher:
            raise Skip(f"the Saluki baseline is running ({', '.join(sorted(baseline))}) and no teacher is up: "
                       "no window to end")
        if not teacher:
            problems = w.business_problems()
            if not problems:
                w.log("no teacher running and business healthy: nothing to restore")
                return 0
            w.log("no teacher running, but business is not healthy: " + "; ".join(problems))
        w.wait_tick()
        ok, line = w.restore()
        if ok:
            # The window is over and business verified: a dead-man left armed would fire into whatever
            # opens next (the continuous window's, say, 15 min into a reopened one). Not while another
            # script still holds the lock: its swap may still need the dead-man.
            if got:
                w.cancel_deadman()
            w.record("window.ended", f"PASS {w.et().strftime('%F %T %Z')} {line}")
            w.notify(f"mini window end: PASS ({', '.join(teacher) or 'no teacher was up'})\n{line}")
            return 0
        w.write_flag("window-end restore failed")
        w.record("window.ended", f"FAIL {w.et().strftime('%F %T %Z')} {line}")
        w.notify(f"mini window end: RESTORE FAILED, the daemon's local model may be DOWN.\n{line}\n"
                 "NO-SWAPS flag set; the dead-man retries.")
        return 1


def start_gen(w: Window, why: str) -> str:
    """Start the generation runner as ``distill-gen-<stamp>`` (see the module doc). Returns what happened,
    for the log and the notify; never raises: generation is not what the window's swap is judged by."""
    c = w.c
    try:
        if c.get("GEN_ENABLE", "1") != "1":
            return "off (GEN_ENABLE)"
        runner = c.get("GEN_RUNNER") or f"{c['WIN_DIR']}/gen_runner.sh"
        if not os.access(runner, os.X_OK):
            w.log(f"no generation runner at {runner}: the window runs without generation")
            return "not staged"
        running = sorted(u for u, s in w.units_matching(c["GEN_UNIT_GLOB"], "service").items() if s in ACTIVE)
        if running:
            return f"already running ({', '.join(running)})"
        unit = f"distill-gen-{w.et().strftime('%Y%m%d-%H%M%S')}"
        cp = w.act(["systemd-run", "--user", f"--unit={unit}", "--collect", "-p", "KillMode=mixed",
                    "-p", f"TimeoutStopSec={c['GEN_STOP_TIMEOUT_S']}", "-p", f"WorkingDirectory={Path(c['WIN_DIR']).parent}",
                    f"--description=distill generation ({why})", runner, "run", "--execute"], timeout=60)
        if not w.execute:
            return f"would start {unit}"
        if cp.returncode != 0:
            w.log(f"systemd-run {unit} exited {cp.returncode}: {(cp.stderr or cp.stdout).strip()[:200]}")
            return f"FAILED to start {unit} (systemd-run exit {cp.returncode})"
        return f"{unit} started"
    except Exception as e:  # noqa: BLE001 - never fail a verified window over generation
        w.log(f"generation start failed: {type(e).__name__}: {e}")
        return f"FAILED to start ({type(e).__name__})"


def stop_gen_units(w: Window) -> list[str]:
    """Generation units (GEN_UNIT_GLOB) still running: stopped by name, waited for (they resume next run)."""
    running = [u for u, s in w.units_matching(w.c["GEN_UNIT_GLOB"], "service").items() if s in ACTIVE]
    for u in running:
        w.act(["systemctl", "--user", "stop", u], timeout=200)
    if running and w.execute:
        deadline = w.now() + 180
        while any(w.active(u) for u in running):
            if w.now() >= deadline:
                raise Fail("generation still running after its stop: " + ", ".join(running))
            w.sleep(5)
    return running


def cmd_slot_enter(w: Window) -> int:
    """Q4 night, before a nightly eval: Q4 teacher -> IQ3 teacher + Ollama (the judge needs its VRAM)."""
    c = w.c
    with w.lock(300) as got:
        if not got:
            raise Skip("another window script holds the lock")
        q4, iq3, ollama = (w.active(c["TEACHER_Q4_UNIT"]), w.active(c["TEACHER_UNIT"]), w.active(c["OLLAMA_UNIT"]))
        if iq3 and ollama and not q4:
            w.log("already in the eval-slot layout (teacher + Ollama)")
            return 0
        if not q4:
            w.log(f"{c['TEACHER_Q4_UNIT']} is not running: not a Q4 night, nothing to do")
            return 0
        gen = stop_gen_units(w)
        w.expect_plan(c["SETUP_WINDOW"], {c["TEACHER_Q4_UNIT"]}, {c["TEACHER_UNIT"], c["OLLAMA_UNIT"]})
        w.wait_tick()
        with w.swapped():
            rc = w.apply(c["SETUP_WINDOW"])
            if rc != 0:
                raise Fail(f"apply {c['SETUP_WINDOW']} exited {rc}")
            w.wait_served(c["TEACHER_ALIAS"], float(c["TEACHER_WAIT_S"]))
            if w.execute:
                deadline = w.now() + 60
                while not w.ollama_has_model():
                    if w.now() >= deadline:
                        raise Fail(f"Ollama does not list {c['OLLAMA_MODEL']} 60 s after its start")
                    w.sleep(5)
            w.swap_pending = False
        # Epoch first: slot-exit swaps back only on the strength of this record from tonight.
        w.record("slot.active", f"{w.now():.0f} {w.et().strftime('%F %T %Z')} {c['TEACHER_ALIAS']} + Ollama"
                 + (f"; stopped {', '.join(gen)}" if gen else ""))
        w.log("eval-slot layout ready: teacher + Ollama (generation stays paused until slot-exit)")
        return 0


def cmd_slot_exit(w: Window) -> int:
    """After a nightly eval on a Q4 night: IQ3 teacher + Ollama -> Q4 teacher, once the eval is done.

    Only on a Q4 night (tonight's slot-enter record), and only from the window's open up to
    SLOT_RETURN_LATEST_HHMM: the eval wait ends at that cutoff, and every condition is checked again right
    before the swap. Otherwise it stays on the IQ3 teacher + Ollama (window-end restores business)."""
    c = w.c
    stay = f"staying on {c['TEACHER_ALIAS']} + Ollama"
    with w.lock(300) as got:
        if not got:
            raise Skip("another window script holds the lock")
        why = w.slot_exit_blocker()
        if why:
            w.log(f"{why}: nothing to do")
            return 0
        cutoff = w.at_today(c["SLOT_RETURN_LATEST_HHMM"]).timestamp()
        deadline = min(w.now() + float(c["EVAL_WAIT_MAX_S"]), cutoff)
        while w.eval_running():
            if not w.execute:
                w.log("[dry run] a nightly eval is running: a real run waits for it")
                break
            if w.now() >= deadline:
                w.notify(f"mini slot-exit: the nightly eval still runs at {w.et().strftime('%H:%M %Z')}; {stay}")
                return 0
            w.sleep(15)
        fit, judge = w.ollama_fit()
        w.log(f"Ollama after the eval: {judge}")
        if fit is False:
            w.notify(f"mini eval slot: the judge ran beside the IQ3 teacher with {judge}: the evals ran slower "
                     "(distill-teacher.service CTX note)")
        w.expect_plan(c["SETUP_Q4"], {c["TEACHER_UNIT"], c["OLLAMA_UNIT"]}, {c["TEACHER_Q4_UNIT"]})
        w.wait_tick()
        why = w.slot_exit_blocker()
        if why:
            w.log(f"{why} (checked again right before the swap): {stay}")
            return 0
        with w.swapped():
            rc = w.apply(c["SETUP_Q4"])
            if rc == 0:
                try:
                    w.wait_served(c["TEACHER_Q4_ALIAS"], float(c["TEACHER_WAIT_S"]))
                except Fail as e:
                    rc, why = 1, str(e)
            else:
                why = f"apply {c['SETUP_Q4']} exited {rc}"
            if rc != 0:
                closed = w.window_closed()
                if closed:
                    # window-end restored past the lock while this waited: falling back now would stop the
                    # local model server again.
                    w.swap_pending = False
                    w.log(f"{why}; {closed}: no fallback")
                    w.notify(f"mini slot-exit: {why}; {closed}, so nothing more was changed")
                    return 1
                # Back to the safe layout (IQ3 + Ollama) and give up on Q4 for the night. The plan may stop
                # only the Q4 teacher; anything else raises, and swapped() restores business instead.
                w.log(f"{why}: falling back to {c['SETUP_WINDOW']}")
                w.expect_plan(c["SETUP_WINDOW"], {c["TEACHER_Q4_UNIT"]}, {c["TEACHER_UNIT"], c["OLLAMA_UNIT"]},
                              at_most=True, exc=Fail)
                if w.apply(c["SETUP_WINDOW"]) != 0:
                    raise Fail(f"{why}; the fallback to {c['SETUP_WINDOW']} failed too")
                w.wait_served(c["TEACHER_ALIAS"], float(c["TEACHER_WAIT_S"]))
                w.swap_pending = False
                w.clear_slot()
                w.notify(f"mini slot-exit: {why}; {stay} for the rest of the night")
                return 1
            if w.execute and w.active(c["OLLAMA_UNIT"]):
                raise Fail(f"{c['OLLAMA_UNIT']} still runs after {c['SETUP_Q4']}")
            w.swap_pending = False
        w.clear_slot()
        w.log(f"back on {c['TEACHER_Q4_ALIAS']} (Ollama stopped)")
        w.log(f"generation: {start_gen(w, 'slot-exit')}")
        return 0


def cmd_teacher_failed(w: Window) -> int:
    """OnFailure= of both teacher units (window-mini-teacher-failed.service). A teacher that systemd gave
    up on (a crash loop: out of memory beside an image job, a CUDA error; the start limit hit) leaves the
    local model port dark until the window ends, up to ~35 h in the continuous window. So: restore business
    now, set the NO-SWAPS flag and notify. Nothing to do when the other teacher serves the port (slot-exit's
    fallback) or a window script already brought the local model server back."""
    c = w.c
    with w.lock(float(c["FAILED_LOCK_WAIT_S"])) as got:
        if not got:
            w.log("a window script still holds the lock: it restores on its own failure (and the dead-man "
                  "backs it up); not restoring beside it")
            return 2
        teachers = (c["TEACHER_UNIT"], c["TEACHER_Q4_UNIT"])
        failed = [u for u in teachers if w.unit_prop(u, "ActiveState") == "failed"]
        if not failed:
            w.log("no teacher unit is failed: nothing to do")
            return 0
        up = [u for u in teachers if w.active(u)]
        if up:
            w.log(f"{', '.join(failed)} failed, but {', '.join(up)} runs: the window goes on")
            return 0
        if w.active(c["LOCAL_MODEL_UNIT"]):
            w.log(f"{', '.join(failed)} failed, and {c['LOCAL_MODEL_UNIT']} runs already: nothing to restore")
            return 0
        names = ", ".join(failed)
        w.log(f"{names} failed with no teacher up and {c['LOCAL_MODEL_UNIT']} stopped: the port is dark; "
              "restoring business")
        ok, line = w.restore()
        w.write_flag(f"{names} failed mid-window")
        w.record("window.ended", f"{'PASS' if ok else 'FAIL'} {w.et().strftime('%F %T %Z')} {names} failed; {line}")
        if ok:
            w.notify(f"mini: {names} FAILED mid-window; business restored.\n{line}\n"
                     "NO-SWAPS flag set: remove it to allow swaps again.")
            return 0
        w.notify(f"mini: {names} FAILED mid-window and the RESTORE FAILED: the daemon's local model may be DOWN."
                 f"\n{line}\nNO-SWAPS flag set; the dead-man retries.")
        return 1


# ── staging (runs on the repo machine) ──────────────────────────────────────


def window_hosts_copy(S: Any, hosts_path: Path, host: str, swap_units: list[str]) -> str:
    """This host's hosts.yaml for the window copy, with ``swap_units`` (the local model server, Ollama)
    taken out of its never_stop in THIS copy only. The shared file must keep them never-stop, so no
    repo-machine setups.py run (a fleet apply, flywheel_weekly) can stop them at any hour; only the
    window scripts (hours, baseline and eval checks, lock, dead-man) apply from this copy."""
    import tempfile  # noqa: PLC0415 - only the repo machine stages

    import yaml  # noqa: PLC0415

    shared = S.load_hosts(hosts_path)
    kept = [u for u in swap_units if not S.is_never_stop(u, shared.guard(host))]
    if kept:
        raise SystemExit(f"{hosts_path}: {', '.join(kept)} must stay never-stop for {host} there "
                         "(only the window copy drops them)")
    data = yaml.safe_load(S.hosts_file_for(hosts_path, host)) or {}
    hs = data["hosts"][host]
    hs["never_stop"] = [u for u in hs.get("never_stop") or [] if u not in swap_units]
    text = (f"# hosts.yaml for {host}'s GPU-window copy (mini_window.py stage). Never commit.\n"
            f"# {', '.join(swap_units)}: not never-stop in THIS copy only, so just ~/distill/window/ may swap them.\n"
            + yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    with tempfile.TemporaryDirectory(prefix="mini-hosts-") as tmp:
        p = Path(tmp) / "hosts.yaml"
        p.write_text(text)
        guard = S.load_hosts(p).guard(host)
    still = [u for u in swap_units if S.is_never_stop(u, guard)]
    if still:
        raise SystemExit(f"{', '.join(still)} still never-stop in the window copy (a site-wide never_stop entry?)")
    lost = [u for u in shared.guard(host) if u not in swap_units and u not in guard]
    if lost:
        raise SystemExit(f"the window copy lost never-stop entries: {', '.join(lost)}")
    return text


def cmd_stage(args: argparse.Namespace) -> int:
    """Copy the window files to the host: scripts, window.conf, and setups.py with this host's sections only."""
    import tempfile

    repo = HERE.parents[2]
    sys.path.insert(0, str(repo / "scripts" / "distill"))
    import setups as S  # noqa: PLC0415 - only the repo machine stages

    conf_path = Path(args.conf)
    conf = load_conf(conf_path, home="/HOME")  # validates it; paths are the host's
    host = conf["HOST_KEY"]
    sdir = Path(args.setups_dir or repo / "data" / "distill" / "setups")
    names = [conf["SETUP_RESTORE"], conf["SETUP_WINDOW"], conf["SETUP_Q4"]]
    dest = args.dest
    with tempfile.TemporaryDirectory(prefix="mini-window-") as tmp:
        t = Path(tmp)
        (t / "setups").mkdir()
        for n in names:
            s = S.load_setup(sdir, n)
            if host not in s["hosts"]:
                raise SystemExit(f"setup {n} has no {host} section")
            S.write_private(t / "setups" / f"{n}.yaml", S.dump_setup({**s, "hosts": {host: s["hosts"][host]}}))
        S.write_private(t / "setups" / "hosts.yaml",
                        window_hosts_copy(S, sdir / "hosts.yaml", host, [conf["LOCAL_MODEL_UNIT"], conf["OLLAMA_UNIT"]]))
        files = {
            "mini_window.py": HERE / "mini_window.py",
            "mini_rehearsal.sh": HERE / "mini_rehearsal.sh",
            "mini_window_start.sh": HERE / "mini_window_start.sh",
            "mini_window_end.sh": HERE / "mini_window_end.sh",
            "eval_slot_enter.sh": HERE / "eval_slot_enter.sh",
            "eval_slot_exit.sh": HERE / "eval_slot_exit.sh",
            "gen_runner.py": HERE / "gen_runner.py",
            "gen_runner.sh": HERE / "gen_runner.sh",
            "window.conf": conf_path,
            "setups/setups.py": repo / "scripts" / "distill" / "setups.py",
            "setups/daemon_switch.py": repo / "scripts" / "distill" / "daemon_switch.py",
            **{f"setups/{p.name}": p for p in sorted((t / "setups").glob("*.yaml"))},
        }
        (t / "SOURCE").write_text(f"staged from {repo.name} at {datetime.now().isoformat(timespec='seconds')}\n")
        files["SOURCE"] = t / "SOURCE"
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.ssh]
        steps: list[tuple[str, Any]] = [
            ("ssh", f"mkdir -p {dest}/setups {dest}/state && chmod 700 {dest} {dest}/setups {dest}/state"),
        ]
        for rel, src in files.items():
            mode = "0700" if rel.endswith((".sh", "mini_window.py", "gen_runner.py")) else "0600"
            steps.append(("put", (rel, src, mode)))
        steps.append(("ssh", f"ls -la {dest} {dest}/setups"))
        for kind, step in steps:
            if kind == "ssh":
                argv = ssh + [step]
                shown = shlex.join(argv)
            else:
                rel, src, mode = step
                remote = f"cat > {dest}/.{rel.replace('/', '_')}.tmp && chmod {mode} {dest}/.{rel.replace('/', '_')}.tmp " \
                         f"&& mv -f {dest}/.{rel.replace('/', '_')}.tmp {dest}/{rel}"
                argv = ssh + [remote]
                shown = f"{shlex.join(argv)} < {src}"
            if not args.execute:
                print(f"[dry run] {shown}")
                continue
            if kind == "put":
                with open(step[1], "rb") as f:
                    cp = subprocess.run(argv, stdin=f, capture_output=True, timeout=120)
            else:
                cp = subprocess.run(argv, capture_output=True, timeout=120)
            out = (cp.stdout + cp.stderr).decode(errors="replace").strip()
            if out:
                print(out)
            if cp.returncode != 0:
                print(f"FAILED (exit {cp.returncode}): {shown}")
                return 1
            if kind == "put":
                print(f"copied {step[0]}")
    if not args.execute:
        print("dry run: nothing copied (pass --execute); nothing on the host is run, armed or started either way")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None, *, runner: Callable[..., Any] | None = None,
         clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
         out: Callable[[str], None] | None = None, signals: bool = False) -> int:
    ap = argparse.ArgumentParser(prog="mini_window", description=__doc__.split("\n\n")[0])
    ap.add_argument("--conf", default=os.environ.get("DISTILL_WINDOW_CONF") or str(HERE / "window.conf"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rehearsal", "window-end", "slot-enter", "slot-exit", "teacher-failed"):
        sub.add_parser(name).add_argument("--execute", action="store_true", help="really change things")
    ws = sub.add_parser("window-start")
    ws.add_argument("--execute", action="store_true", help="really change things")
    ws.add_argument("--q4", action="store_true", help="flywheel-night-q4 (nightly only; Ollama stopped)")
    sub.add_parser("status")
    st = sub.add_parser("stage", help="(repo machine) copy the window files to the host")
    st.add_argument("--ssh", required=True)
    st.add_argument("--dest", default="distill/window", help="on the host, relative to its home")
    st.add_argument("--setups-dir", default=None)
    st.add_argument("--execute", action="store_true")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 64 if e.code else 0
    if args.cmd == "stage":
        return cmd_stage(args)
    conf = load_conf(Path(args.conf))
    execute = bool(getattr(args, "execute", False))
    w = Window(conf, execute, args.cmd, runner=runner, clock=clock, sleep=sleep, out=out)
    if args.cmd != "status":
        w.start_log()
        if not execute:
            w.log("dry run: read-only checks run, every change is printed instead (pass --execute)")

    terminating: list[int] = []

    def on_term(signum: int, _frame: Any) -> None:
        # Only the first signal interrupts: a second one must not cut short the restore the first started.
        if terminating:
            w.log(f"signal {signum} again: ignored while the restore finishes")
            return
        terminating.append(signum)
        raise KeyboardInterrupt(f"signal {signum}")

    if signals:  # the CLI only: SIGTERM/SIGHUP while a swap is applied restore business first
        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGHUP, on_term)
    try:
        if args.cmd == "status":
            return cmd_status(w)
        if args.cmd == "rehearsal":
            return cmd_rehearsal(w)
        if args.cmd == "window-start":
            return cmd_window_start(w, args.q4)
        if args.cmd == "window-end":
            return cmd_window_end(w)
        if args.cmd == "slot-enter":
            return cmd_slot_enter(w)
        if args.cmd == "teacher-failed":
            return cmd_teacher_failed(w)
        return cmd_slot_exit(w)
    except Skip as e:
        w.log(f"SKIP: {e}")
        if execute and args.cmd in ("rehearsal", "window-start"):
            w.record(f"{args.cmd.replace('-', '_')}.skip", f"SKIP {w.et().strftime('%F %T %Z')} {e}")
            w.notify(f"mini {args.cmd} skipped (nothing changed): {e}")
        return 2
    except (Fail, KeyboardInterrupt) as e:
        w.log(f"FAIL: {e}")
        if execute and args.cmd == "rehearsal":
            w.record("rehearsal.last", f"FAIL {w.et().strftime('%F %T %Z')} {e}")
        return 1
    finally:
        w.log("end")


if __name__ == "__main__":
    sys.exit(main(signals=True))
