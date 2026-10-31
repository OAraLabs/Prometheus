#!/usr/bin/env python3
"""GPU setups: named images of what runs on each GPU host.

    uv run python scripts/distill/setups.py save business
    uv run python scripts/distill/setups.py status
    uv run python scripts/distill/setups.py apply flywheel              # dry run (default)
    uv run python scripts/distill/setups.py apply business --execute
    uv run python scripts/distill/setups.py deadman business --at 2026-10-31T07:15-04:00
    uv run python scripts/distill/setups.py deadman --cancel

A setup is ``data/distill/setups/<name>.yaml`` (gitignored: it holds host
details). Per host it lists the systemd *user* units that own the GPU, with
their ExecStart, unit files, listening ports, VRAM and a health check, plus
``allow_extra`` (patterns of units that may run beside them) and ``stop``
(units this setup may stop although ``business`` does not list them, and
stops whenever they run, even one an ``allow_extra`` pattern matches: two
model servers that take turns on one port, say). The
hosts themselves come from ``data/distill/setups/hosts.yaml``::

    hosts:
      tower: {ssh: user@host-a}
      mini:
        ssh: host-b
        python: /usr/bin/python3         # optional: runs setups.py for the dead-man
        never_stop: [web.service]        # optional: this host only
        notify_cmd: /path/to/notify      # gets a failed restore's summary on stdin;
                                         # `deadman --execute` refuses a host without one
    never_stop: [site-specific.service]  # optional, added to NEVER_STOP on every host
    business_units:                      # optional: units `save business` captures when they
      mini: [model-server.service]       # run, even holding no VRAM at that moment (a
                                         # unit the user manager has not loaded: refused)

Every host is reached over SSH (BatchMode) and inspected by a small read-only
probe that runs there with the host's own python3. ``save`` and ``status`` only
read. ``apply`` prints its plan and changes nothing unless ``--execute`` is
given. ``deadman`` schedules an independent ``systemd-run --user`` timer on
each host that runs ``apply <name> --execute --local`` from a copy of this
script on that host, so a restore never depends on the machine that started it.

Units matching NEVER_STOP (or hosts.yaml ``never_stop``) are never stopped: a
plan that would stop one is refused as a whole, and the execute layer checks
again before every ``systemctl stop``. A GPU process the tool cannot manage
(anything outside a user ``.service`` of this account, except this tool's own
``distill-*.scope`` jobs) blocks the plan before anything runs. Every apply
other than ``business`` also refuses to stop a unit that ``business`` does not
list, so a restore can bring back whatever a swap took down.

Hooks: a host section may list ``pre_apply`` and ``post_apply`` commands, run
on that host before any unit is stopped and after the health wait::

    mini:
      pre_apply:  [daemon_switch.py capture, daemon_switch.py to-max]    # flywheel
      post_apply: [daemon_switch.py restore]                             # business
      # {run: <command>, enabled: false, why: <text>, timeout_s: 300} is the long form

``daemon_switch.py <command>`` runs the sibling script on the host (shipped on
stdin, or the dead-man's copy) as a dry run, with ``--execute`` added only when
the apply executes (a hook passes only the switch's own SWITCH_HOOK_OPTIONS,
spelled out); it moves the Prometheus daemon's primary model to the Max API
and back, restarting the daemon only through the daemon's own
``POST /api/restart`` (no systemd stop). Any other command runs under ``sh -c``
on an executing apply only, after the never-stop guard: a command that could
stop a never-stop unit, stops a whole manager (``isolate``, ``exit``, ...),
runs a command line the guard cannot see (``sh -c``, ``xargs``, ``eval``), or
kills/reboots/escalates, is refused at load and again before it runs. A
failing ``pre_apply`` hook ends the attempt before any unit changes.
``post_apply`` hooks run even when the health wait reported problems (a
restore hook checks its own preconditions). ``enabled: false`` hooks are
listed and never run. A setup that switches a host to Max is applied (and a
dead-man armed) only when ``business`` switches that host back, and executes
only while that host's dead-man timer is armed. The same holds for any apply
other than ``business`` that stops a unit ``business`` lists (only a restore
starts it again). A target unit the user manager has not loaded (not installed,
masked) blocks the plan before anything stops. ``apply --execute`` makes up to
1 + RESTORE_RETRIES attempts; ``--attempts 1`` makes one (a forward swap whose
caller restores on failure), and an attempt stops waiting for health once a
target unit has failed outright.
"""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

# Units that a setup change must never stop, whatever the target says. Matched
# with fnmatch against the unit name. hosts.yaml ``never_stop`` can add to this
# list (site-specific unit names live there, not in the repo); nothing removes
# from it.
NEVER_STOP: tuple[str, ...] = (
    "prometheus.service",
    "prometheus-*.service",
    "cloudflared*",
    "tailscale*",
    "ssh*",
    "sshd*",
    "distill-deadman-*",
    "dbus*",
    "systemd-*",
    "pipewire*",
    "wireplumber*",
)

DEFAULT_SETUPS_DIR = Path(__file__).resolve().parents[2] / "data" / "distill" / "setups"
REMOTE_DIR = ".local/share/distill-setups"  # relative to $HOME, on every host
LAST_APPLY = "last-apply.json"  # in REMOTE_DIR: the outcome of the last --execute apply
DEADMAN_PREFIX = "distill-deadman-"
DEADMAN_PYTHON = "/usr/bin/python3"  # hosts.yaml ``python`` overrides it per host
NAIVE_TZ = "America/New_York"  # the runbook's windows are in ET

# "Restoring always means apply business" (runbook ground rule 4).
RESTORE_SETUP = "business"
# This tool's own jobs. A restore stops every running one, on the GPU or not.
DISTILL_JOBS = "distill-*"

UNIT_RE = re.compile(r"^[A-Za-z0-9@._:-]+\.service$")
# Scopes are stop-only (they cannot be started), and only this tool's own.
SCOPE_RE = re.compile(r"^distill-[A-Za-z0-9@._:-]+\.scope$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HOST_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
ACTIVE_STATES = ("active", "activating", "reloading")

STOP_TIMEOUT_S = 120
DEFAULT_HEALTH_TIMEOUT_S = 300
POLL_S = 5
RESTORE_RETRIES = 2
RETRY_DELAY_S = 60

_sleep = time.sleep
_now = time.monotonic


# ── the probe (runs on each host; read-only) ────────────────────────────────
#
# These functions are shipped to the host verbatim (inspect.getsource) and run
# there with the host's python3, so they use only the stdlib names imported in
# _PROBE_PRELUDE. They are also imported normally here, which is how the tests
# exercise them.


def _run(argv: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout, "err": p.stderr[-500:]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"rc": -1, "out": "", "err": str(e)}


def unit_from_cgroup(text: str, uid: int) -> tuple[str, str | None]:
    """(scope, unit) for a /proc/<pid>/cgroup body.

    scope is "user" for a unit in this account's own user manager
    (user@<uid>.service), "other-user" for a unit in another account's user
    manager, and "system" for everything else (system units, login sessions).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    v2 = [ln for ln in lines if ln.startswith("0::")]
    for line in v2 + lines:
        path = line.split(":", 2)[-1]
        parts = [p for p in path.split("/") if p]
        units = [p for p in parts if p.endswith(".service") or p.endswith(".scope")]
        if not units:
            continue
        unit = units[-1]
        managers = [p for p in parts if re.match(r"^user@\d+\.service$", p)]
        if managers and unit not in managers:
            return ("user" if "user@%d.service" % uid in managers else "other-user"), unit
        return "system", unit
    return "system", None


def parse_ss_listen(text: str) -> list[tuple[int, str]]:
    """(pid, local addr:port) pairs from ``ss -ltnpH`` output."""
    out: list[tuple[int, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "LISTEN":
            continue
        for pid in re.findall(r"pid=(\d+)", line):
            out.append((int(pid), parts[3]))
    return out


def parse_list_units(text: str, suffixes: tuple[str, ...] = (".service", ".scope")) -> dict[str, str]:
    """unit -> ACTIVE state, from ``systemctl list-units --plain --no-legend`` (units with ``suffixes``)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.replace("●", " ").split()
        if len(parts) >= 3 and parts[0].endswith(suffixes):
            out[parts[0]] = parts[2]
    return out


def parse_unit_cat(text: str) -> dict[str, Any]:
    """Unit files and effective ExecStart lines from ``systemctl --user cat``."""
    files: list[str] = []
    execs: list[str] = []
    pending: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if pending is not None:
            if line.endswith("\\"):
                pending += " " + line[:-1].strip()
                continue
            execs.append((pending + " " + line.strip()).strip())
            pending = None
            continue
        if line.startswith("# /"):
            files.append(line[2:].strip())
            continue
        stripped = line.strip()
        if stripped.startswith("ExecStart="):
            value = stripped[len("ExecStart="):]
            if not value:
                execs = []  # an empty ExecStart= in a drop-in resets the list
                continue
            if value.endswith("\\"):
                pending = value[:-1].strip()
                continue
            execs.append(value.strip())
    if pending is not None:
        execs.append(pending.strip())
    return {"files": files, "exec_start": execs}


def redact_unit_text(text: str) -> str:
    """Blank secret-looking values before a unit's text leaves the host.

    Covers NAME=value assignments (quoted or not), ``--flag value`` and
    ``--flag=value`` arguments, short ``-hft``/``-k`` flags, URL userinfo
    (``scheme://user:pass@``) and auth headers (``Authorization: Bearer x``,
    ``X-Api-Key:x``), wherever the name says key, token, secret, password,
    pass, auth or credential.
    """
    red = "<redacted>"
    word = r"(?:key|token|secret|passw|passph|pass(?![a-z0-9_-])|auth|cred)"
    # A value: \"escaped quotes\" (inside an sh -c "..." string, up to the first
    # closing \" and never past the string's own "), "quoted", 'quoted' or bare.
    val = r"(?:\\\"[^\"\n]*\\\"|\\'[^'\n]*\\'|\"[^\"]*\"|'[^']*'|[^\s\"']+)"
    # scheme://user:pass@host -> scheme://<redacted>@host
    text = re.sub(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s@\"']+@", r"\g<1>" + red + "@", text)
    # Authorization: Bearer x / X-Api-Key: x, and -H 'X-Api-Key:x' (curl needs no
    # space after the colon; without one the name must start a word, so a path
    # such as org/tokenizer-model:Q4 is left alone)
    text = re.sub(r"(?i)((?:\b[a-z0-9-]*" + word + r"[a-z0-9-]*:[ \t]+|(?<![^\s\"'=])[a-z0-9-]*"
                  + word + r"[a-z0-9-]*:)(?:(?:bearer|basic|token)[ \t]+)?)" + val,
                  r"\g<1>" + red, text)
    text = re.sub(r"(?i)(\bbearer[ \t]+)" + val, r"\g<1>" + red, text)
    # Environment="API_TOKEN=two words": on an Environment= line the quotes wrap one
    # whole assignment. Elsewhere they may wrap a command (sh -c "TOKEN=x exec srv
    # --port 1"), so there the NAME=value rule below stops at the first space.
    quoted = re.compile(r"(?i)(\"[A-Za-z0-9_]*" + word + r"[A-Za-z0-9_]*=)(?:[^\"\\\n]|\\.)*\"")
    text = re.sub(r"(?m)^[ \t]*Environment=.*$",
                  lambda m: quoted.sub(r"\g<1>" + red + '"', m.group(0)), text)
    # NAME=value, --name=value
    text = re.sub(r"(?i)(\b[A-Za-z0-9_-]*" + word + r"[A-Za-z0-9_-]*=)" + val, r"\g<1>" + red, text)
    # --name value, -hft value, -k value (unless the next word is another flag)
    return re.sub(r"(?i)((?<!\S)(?:--?[a-z0-9_-]*" + word + r"[a-z0-9_-]*|-hft|-k)[ \t]+)(?!-)" + val,
                  r"\g<1>" + red, text)


def guess_health(unit: str, exec_start: list[str], ports: list[str]) -> dict[str, Any]:
    """llama.cpp servers get HTTP /health; everything else is-active."""
    joined = " ".join(exec_start)
    if "llama-server" in joined:
        port = None
        m = re.search(r"--port[= ]+(\d+)", joined)
        if m:
            port = m.group(1)
        addr = None
        for p in ports:
            a, _, pp = p.rpartition(":")
            if port is None or pp == port:
                addr, port = a, pp
                break
        if port is not None:
            host = (addr or "127.0.0.1").strip("[]")
            if host in ("0.0.0.0", "*", "::", ""):
                host = "127.0.0.1"
            if ":" in host:
                host = "[" + host + "]"
            return {"type": "http", "url": "http://%s:%s/health" % (host, port)}
    return {"type": "systemd", "unit": unit}


def check_health(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("type") == "http":
        try:
            with urllib.request.urlopen(spec["url"], timeout=5) as r:
                return {"ok": r.status == 200, "detail": "HTTP %d" % r.status}
        except urllib.error.HTTPError as e:
            return {"ok": False, "detail": "HTTP %d" % e.code}
        except Exception as e:  # noqa: BLE001 - any failure is "unhealthy"
            return {"ok": False, "detail": type(e).__name__}
    state = _run(["systemctl", "--user", "is-active", spec.get("unit", "")])["out"].strip()
    return {"ok": state == "active", "detail": state or "unknown"}


def probe_main(req: dict[str, Any]) -> dict[str, Any]:
    uid = os.getuid()
    gpus_r = _run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
                   "--format=csv,noheader,nounits"])
    if gpus_r["rc"] != 0:
        return {"reachable": True, "ok": False, "error": "nvidia-smi: " + gpus_r["err"].strip()}
    gpus = []
    for line in gpus_r["out"].splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            gpus.append({"index": int(p[0]), "uuid": p[1], "name": ",".join(p[2:-2]),
                         "vram_total_mib": int(p[-2]), "vram_used_mib": int(p[-1])})
    apps_r = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                   "--format=csv,noheader,nounits"])
    procs = []
    for line in apps_r["out"].splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[1].isdigit():
            mem = int(p[-1]) if p[-1].isdigit() else 0
            procs.append({"gpu_uuid": p[0], "pid": int(p[1]), "process": ",".join(p[2:-1]),
                          "vram_mib": mem})
    listen = parse_ss_listen(_run(["ss", "-ltnpH"])["out"])
    cg: dict[int, tuple[str, str | None]] = {}
    for pid in {pr["pid"] for pr in procs} | {pid for pid, _ in listen}:
        try:
            with open("/proc/%d/cgroup" % pid) as f:
                cg[pid] = unit_from_cgroup(f.read(), uid)
        except OSError:
            cg[pid] = ("system", None)
    for pr in procs:
        pr["scope"], pr["unit"] = cg.get(pr["pid"], ("system", None))
    ports: dict[str, list[str]] = {}
    for pid, addr in listen:
        scope, unit = cg.get(pid, ("system", None))
        if unit and addr not in ports.setdefault(unit, []):
            ports[unit].append(addr)
    # This tool's own jobs and dead-man units, whether or not they hold VRAM now.
    listed = parse_list_units(_run(["systemctl", "--user", "list-units", "--plain", "--no-legend",
                                    "--no-pager", "distill-*"])["out"])
    deadman = {u: s for u, s in listed.items() if u.startswith("distill-deadman-")}
    jobs = {u for u, s in listed.items() if u not in deadman and s != "inactive"}
    wanted = {pr["unit"] for pr in procs if pr["scope"] == "user" and pr["unit"]}
    wanted |= set(req.get("units", [])) | jobs
    specs = req.get("health", {})
    units = {}
    for u in sorted(wanted):
        active = _run(["systemctl", "--user", "is-active", u])["out"].strip() or "unknown"
        # "not-found" for a typo or a system unit (is-active says "inactive" for both)
        load = _run(["systemctl", "--user", "show", "-p", "LoadState", "--value", u])["out"].strip()
        cat = _run(["systemctl", "--user", "cat", u])
        text = redact_unit_text(cat["out"]) if cat["rc"] == 0 else ""
        parsed = parse_unit_cat(text)
        spec = specs.get(u) or guess_health(u, parsed["exec_start"], ports.get(u, []))
        if active in ("active", "activating", "reloading"):
            health = check_health(spec)
        else:
            health = {"ok": False, "detail": "unit " + active}
        units[u] = {"active": active, "load": load or "unknown", "files": parsed["files"],
                    "exec_start": parsed["exec_start"], "ports": sorted(ports.get(u, [])),
                    "health_spec": spec, "health": health}
    timers = _run(["systemctl", "--user", "list-timers", "--all", "--no-legend",
                   "--no-pager", "distill-deadman-*"])["out"]
    # A dead-man restore is armed while its timer is active (waiting); an apply that moves the daemon
    # to Max needs one first.
    timer_states = parse_list_units(_run(["systemctl", "--user", "list-units", "--type=timer", "--all", "--plain",
                                          "--no-legend", "--no-pager", "distill-deadman-*"])["out"], (".timer",))
    last = None
    if req.get("last_apply"):
        try:
            with open(os.path.join(os.path.expanduser("~"), req["last_apply"])) as f:
                last = json.loads(f.read())
        except (OSError, ValueError):
            last = None
    return {"reachable": True, "ok": True, "gpus": gpus, "procs": procs, "units": units,
            "deadman_timers": [ln.strip() for ln in timers.splitlines() if ln.strip()],
            "deadman_units": deadman, "deadman_timer_states": timer_states, "last_apply": last}


_PROBE_FUNCS = (_run, unit_from_cgroup, parse_ss_listen, parse_list_units, parse_unit_cat,
                redact_unit_text, guess_health, check_health, probe_main)
_PROBE_PRELUDE = (
    "from __future__ import annotations\n"
    "import json, os, re, subprocess, sys, urllib.error, urllib.request\n"
    "from typing import Any\n"
)


def probe_source() -> str:
    body = "\n\n".join(inspect.getsource(f) for f in _PROBE_FUNCS)
    return (_PROBE_PRELUDE + "\n" + body
            + "\n\nprint(json.dumps(probe_main(json.loads(sys.argv[1]))))\n")


# ── hosts and transport ─────────────────────────────────────────────────────


@dataclass
class Host:
    key: str
    ssh: str | None
    python: str | None = None
    never_stop: tuple[str, ...] = ()
    notify_cmd: str | None = None


@dataclass
class Hosts:
    hosts: dict[str, Host]
    never_stop: tuple[str, ...] = ()
    business_units: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def guard(self, host: str) -> tuple[str, ...]:
        extra = self.hosts[host].never_stop if host in self.hosts else ()
        return NEVER_STOP + self.never_stop + extra


def load_hosts(path: Path, *, required: bool = True) -> Hosts:
    if not path.exists():
        if required:
            raise SystemExit(f"hosts file not found: {path}")
        return Hosts({})
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: not a mapping")
    hosts: dict[str, Host] = {}
    for key, cfg in (data.get("hosts") or {}).items():
        if not HOST_KEY_RE.match(str(key)):
            raise SystemExit(f"bad host key {key!r} (lowercase letters, digits, '-')")
        cfg = cfg or {}
        hosts[key] = Host(key=key, ssh=cfg.get("ssh"), python=cfg.get("python") or None,
                          never_stop=tuple(cfg.get("never_stop") or ()),
                          notify_cmd=cfg.get("notify_cmd") or None)
    business: dict[str, tuple[str, ...]] = {}
    for key, units in (data.get("business_units") or {}).items():
        if not HOST_KEY_RE.match(str(key)):
            raise SystemExit(f"{path}: business_units: bad host key {key!r}")
        try:
            business[key] = tuple(check_unit_name(str(u)) for u in units or ())
        except ValueError as e:
            raise SystemExit(f"{path}: business_units: {e}") from None
    return Hosts(hosts, tuple(data.get("never_stop") or ()), business)


class Runner:
    """Runs argv on a host: over SSH (BatchMode), or here with ``local``."""

    def __init__(self, hosts: Hosts, local: bool = False) -> None:
        self.hosts = hosts
        self.local = local

    def command(self, host: str, argv: list[str]) -> list[str]:
        if self.local:
            return argv
        target = self.hosts.hosts[host].ssh if host in self.hosts.hosts else None
        if not target:
            raise SystemExit(f"no ssh target for host {host!r} in hosts.yaml")
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, shlex.join(argv)]

    def run(self, host: str, argv: list[str], input: str | None = None,
            timeout: int = 90) -> subprocess.CompletedProcess[str]:
        cmd = self.command(host, argv)
        try:
            return subprocess.run(cmd, input=input, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            return subprocess.CompletedProcess(cmd, 124, "", f"timeout: {e}")

    def python(self, host: str) -> str:
        if self.local:
            return sys.executable  # already the interpreter this host runs us with
        h = self.hosts.hosts.get(host)
        return (h.python if h else None) or "python3"


def probe(runner: Runner, host: str, units: list[str] | None = None,
          health: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    req = {"units": sorted(set(units or [])), "health": health or {},
           "last_apply": REMOTE_DIR + "/" + LAST_APPLY}
    for u in req["units"]:
        check_unit_name(u, allow_scope=True)
    cp = runner.run(host, [runner.python(host), "-", json.dumps(req)], input=probe_source())
    if cp.returncode != 0:
        return {"reachable": False, "ok": False,
                "error": (cp.stderr or cp.stdout).strip()[-300:] or f"exit {cp.returncode}"}
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"reachable": True, "ok": False, "error": "unparseable probe output"}


# ── setups: build, validate, load, save ─────────────────────────────────────


def check_unit_name(unit: str, *, allow_scope: bool = False) -> str:
    if UNIT_RE.match(unit) or (allow_scope and SCOPE_RE.match(unit)):
        return unit
    kind = "plain .service or distill-*.scope" if allow_scope else "plain .service"
    raise ValueError(f"not a {kind} unit name: {unit!r}")


def stoppable(unit: str) -> bool:
    """A name this tool may pass to ``systemctl --user stop``."""
    return bool(UNIT_RE.match(unit) or SCOPE_RE.match(unit))


def is_distill_job(unit: str) -> bool:
    return fnmatch.fnmatch(unit, DISTILL_JOBS) and not unit.startswith(DEADMAN_PREFIX)


def is_never_stop(unit: str, guard: tuple[str, ...] = NEVER_STOP) -> bool:
    return any(fnmatch.fnmatch(unit, pat) for pat in guard)


_SHELLS = frozenset({"sh", "bash", "dash", "zsh", "ksh", "ash"})


def guess_model(exec_start: list[str]) -> str | None:
    try:
        toks = shlex.split(" ".join(exec_start))
    except ValueError:
        toks = " ".join(exec_start).split()
    flat: list[str] = []
    shell = False
    for i, t in enumerate(toks):
        # Only a shell's -c string is a command to split again (sh -c "... -m x ...");
        # any other quoted token, such as -m "/models/my model.gguf", is one argument.
        if shell and i and re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", toks[i - 1]):
            try:
                flat += shlex.split(t)
            except ValueError:
                flat += t.split()
        else:
            flat.append(t)
        shell = shell or os.path.basename(t.lstrip("@-+!:")) in _SHELLS
    toks = flat
    for i, t in enumerate(toks[:-1]):
        if t in ("-m", "--model", "-hf", "--hf-repo"):
            return toks[i + 1]
    return None


def host_entry_from_live(live: dict[str, Any], ssh: str | None,
                         extra_units: tuple[str, ...] = ()) -> dict[str, Any]:
    """One host's section of a setup, from a probe result.

    Services are this account's user ``.service`` units that hold VRAM, plus
    ``extra_units`` (hosts.yaml ``business_units``) whether or not they hold
    VRAM now. Every other GPU process is recorded as unmanaged.
    """
    by_unit: dict[str, list[dict[str, Any]]] = {}
    unmanaged = []
    for pr in live.get("procs", []):
        unit = pr.get("unit")
        if pr.get("scope") == "user" and unit and UNIT_RE.match(unit):
            by_unit.setdefault(unit, []).append(pr)
        else:
            unmanaged.append({k: pr.get(k) for k in ("pid", "process", "vram_mib", "scope", "unit")})
    order = sorted(by_unit, key=lambda u: -sum(p["vram_mib"] for p in by_unit[u]))
    order += [u for u in extra_units if u not in by_unit]
    services = []
    for unit in order:
        procs = by_unit.get(unit, [])
        info = live.get("units", {}).get(unit, {})
        execs = info.get("exec_start", [])
        names = sorted({p["process"] for p in procs})
        services.append({
            "unit": unit,
            "vram_mib": sum(p["vram_mib"] for p in procs),
            "gpu_uuids": sorted({p["gpu_uuid"] for p in procs}),
            "processes": names,
            "model": guess_model(execs),
            "exec_start": execs,
            "unit_files": info.get("files", []),
            "ports": info.get("ports", []),
            "health": info.get("health_spec") or {"type": "systemd", "unit": unit},
            "health_timeout_s": DEFAULT_HEALTH_TIMEOUT_S,
            # Ollama holds VRAM only while a model is loaded; after a start it is
            # healthy with no GPU process until the first request. A business_units
            # entry is captured whether or not it is on the GPU, so it is on demand too.
            "gpu_on_demand": unit in extra_units or "ollama" in " ".join(execs + names),
            "healthy_at_save": bool(info.get("health", {}).get("ok")),
        })
    return {
        "ssh": ssh,
        "gpus": live.get("gpus", []),
        "services": services,
        "unmanaged_gpu_processes": unmanaged,
        "allow_extra": [],
        "stop": [],
        "min_free_mib": None,
    }


def build_setup(name: str, live_by_host: dict[str, dict[str, Any]], hosts: Hosts,
                captured_at: str | None = None,
                extra: dict[str, tuple[str, ...]] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "version": SCHEMA_VERSION,
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hosts": {h: host_entry_from_live(live, hosts.hosts[h].ssh if h in hosts.hosts else None,
                                          tuple((extra or {}).get(h, ())))
                  for h, live in live_by_host.items()},
    }


# ── hooks: commands a setup runs on a host around its unit changes ──────────

SWITCH_SCRIPT = "daemon_switch.py"  # a sibling of this file, run on the host
SWITCH_COMMANDS = {"pre_apply": ("capture", "to-max", "status"), "post_apply": ("restore", "status")}
SWITCH_TIMEOUT_S = 1800  # idle wait + daemon restart + model-server wait, with room
# The options a daemon_switch hook may pass, each taking a value, matched exactly (no abbreviation, so
# nothing like --exec reaches the script as --execute). --execute is setups' to add; --offline,
# --notify-cmd and --force are not for hooks.
SWITCH_HOOK_OPTIONS = ("--config", "--pins", "--env-file", "--state-dir", "--api", "--idle-wait", "--boot-wait",
                       "--backend-wait", "--set")
HOOK_TIMEOUT_S = 300
HOOK_PHASES = ("pre_apply", "post_apply")
HOOK_KEYS = frozenset({"run", "enabled", "why", "timeout_s"})
# Commands a hook never runs: killing by pattern or pid, power state, privilege.
_HOOK_FORBIDDEN = frozenset({"kill", "pkill", "killall", "reboot", "shutdown", "poweroff", "halt",
                             "telinit", "init", "sudo", "su", "doas", "pkexec"})
# systemctl/service verbs that stop (or may stop) a unit.
_STOP_VERBS = frozenset({"stop", "restart", "try-restart", "reload-or-restart", "try-reload-or-restart",
                         "condrestart", "force-reload", "kill", "isolate", "mask", "disable", "reboot",
                         "poweroff", "halt", "suspend", "hibernate", "hybrid-sleep", "rescue", "emergency",
                         "exit", "soft-reboot", "default", "clean", "freeze", "switch-root"})


# systemctl verbs that change the whole manager's (or the machine's) state, stopping every unit with no
# unit named: refused outright.
_STATE_VERBS = frozenset({"isolate", "exit", "default", "rescue", "emergency", "soft-reboot", "switch-root",
                          "reboot", "poweroff", "halt", "kexec", "suspend", "hibernate", "hybrid-sleep",
                          "suspend-then-hibernate"})
# Commands that run a command line the guard cannot see (a shell's -c is checked separately).
_INDIRECT = frozenset({"xargs", "eval"})


def hook_guard_problem(run: str, guard: tuple[str, ...] = NEVER_STOP) -> str | None:
    """Why a generic hook command must not run, or None. Catches mistakes; hooks are trusted config."""
    try:
        lex = shlex.shlex(run, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        toks = list(lex)
    except ValueError as e:
        return f"cannot parse it: {e}"
    if "daemon_switch" in run:
        return f"run {SWITCH_SCRIPT} as its own hook form ('{SWITCH_SCRIPT} <command>'), not inside a command"
    words = [os.path.basename(t) for t in toks]
    for w in words:
        if w in _HOOK_FORBIDDEN:
            return f"it runs {w!r}: hooks never kill processes, change power state or escalate"
        if w in _INDIRECT:
            return f"it runs {w!r}, whose command line the never-stop guard cannot check"
    for i, w in enumerate(words):
        if w in _SHELLS and any(t.startswith("-") and not t.startswith("--") and "c" in t[1:] for t in toks[i + 1:]):
            return f"it runs {w} -c, whose command line the never-stop guard cannot check (write it as the hook)"
    if "systemctl" in words:
        state = sorted({t for t in toks if t in _STATE_VERBS})
        if state:
            return f"it runs systemctl {state[0]}, which stops every unit of the manager (never-stop ones too)"
    if ("systemctl" in words or "service" in words) and any(t in _STOP_VERBS for t in toks):
        for t in toks:
            if (t.startswith("-") or t in _STOP_VERBS or os.path.basename(t) in ("systemctl", "service")
                    or not re.search(r"[A-Za-z0-9*?\[$`]", t)):  # options, verbs, ; && | ( ...
                continue
            if any(ch in t for ch in "*?[$`"):
                return f"it stops a unit named by a pattern or variable ({t!r})"
            unit = t if re.search(r"\.[a-z]+$", t) else t + ".service"
            if is_never_stop(unit, guard) or is_never_stop(t, guard):
                return f"it would stop never-stop unit {t}"
    return None


def normalize_hook(entry: Any, phase: str, guard: tuple[str, ...] = NEVER_STOP) -> dict[str, Any]:
    """A hook entry (a command string, or a mapping) as ``{run, enabled, timeout_s[, why]}``."""
    hook = {"run": entry} if isinstance(entry, str) else dict(entry) if isinstance(entry, dict) else None
    if hook is None:
        raise ValueError(f"{phase}: a hook is a command string or a mapping with 'run'")
    unknown = sorted(set(hook) - HOOK_KEYS)
    if unknown:
        raise ValueError(f"{phase}: unknown hook key(s) {', '.join(unknown)}")
    run = hook.get("run")
    if not isinstance(run, str) or not run.strip():
        raise ValueError(f"{phase}: a hook needs a non-empty 'run' command")
    enabled = hook.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{phase}: hook {run!r}: enabled must be true or false")
    try:
        argv = shlex.split(run)
    except ValueError as e:
        raise ValueError(f"{phase}: hook {run!r}: {e}") from None
    if argv[0] == SWITCH_SCRIPT:
        allowed = SWITCH_COMMANDS[phase]
        if len(argv) < 2 or argv[1] not in allowed:
            raise ValueError(f"{phase}: {SWITCH_SCRIPT} hooks here run one of {', '.join(allowed)} "
                             "(to-max before units stop, restore after they start)")
        args, i = argv[2:], 0
        while i < len(args):
            name, eq, _ = args[i].partition("=")
            if name not in SWITCH_HOOK_OPTIONS:
                raise ValueError(f"{phase}: hook {run!r}: {args[i]!r} is not an option a {SWITCH_SCRIPT} hook "
                                 f"passes (only {', '.join(SWITCH_HOOK_OPTIONS)}, spelled out; setups adds "
                                 "--execute itself on an executing apply)")
            if not eq:
                if i + 1 >= len(args) or args[i + 1].startswith("-"):
                    raise ValueError(f"{phase}: hook {run!r}: {name} needs a value")
                i += 1
            i += 1
        default_timeout = SWITCH_TIMEOUT_S
    else:
        problem = hook_guard_problem(run, guard)
        if problem:
            raise ValueError(f"{phase}: hook {run!r} refused: {problem}")
        default_timeout = HOOK_TIMEOUT_S
    timeout = hook.get("timeout_s", default_timeout)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError(f"{phase}: hook {run!r}: timeout_s must be a positive integer")
    out: dict[str, Any] = {"run": run, "enabled": enabled, "timeout_s": timeout}
    if hook.get("why"):
        out["why"] = str(hook["why"])
    return out


def switch_hooks(setup: dict[str, Any], host: str, phase: str) -> list[str]:
    """The daemon_switch commands this setup's ENABLED hooks run on ``host`` in ``phase``."""
    out = []
    for hook in setup["hosts"].get(host, {}).get(phase, []) or []:
        argv = shlex.split(hook["run"]) if hook.get("enabled", True) else []
        if argv and argv[0] == SWITCH_SCRIPT:
            out.append(argv[1])
    return out


def deadman_armed(live: dict[str, Any], host: str) -> bool:
    """Whether the probe saw this host's dead-man timer active (armed, waiting to restore)."""
    return (live.get("deadman_timer_states") or {}).get(f"{DEADMAN_PREFIX}{host}.timer") == "active"


def restore_gaps(switching: list[dict[str, Any]], restore: dict[str, Any] | None,
                 keys: list[str]) -> list[str]:
    """Hosts a setup in ``switching`` moves to Max that the restore setup would not move back."""
    gaps = []
    for h in keys:
        movers = [s["name"] for s in switching if "to-max" in switch_hooks(s, h, "pre_apply")]
        if movers and (restore is None or "restore" not in switch_hooks(restore, h, "post_apply")):
            gaps.append(f"{h}: {', '.join(sorted(movers))} switches the daemon to Max there, but "
                        f"{RESTORE_SETUP} has no post_apply '{SWITCH_SCRIPT} restore' for {h}")
    return gaps


def switch_source() -> str:
    path = Path(__file__).resolve().with_name(SWITCH_SCRIPT)
    try:
        return path.read_text()
    except OSError as e:
        raise RuntimeError(f"{SWITCH_SCRIPT} not found next to setups.py ({path}): {e}") from None


def run_hooks(phase: str, setup: dict[str, Any], keys: list[str], runner: Runner, hosts: Hosts,
              execute: bool) -> list[str]:
    """Run one phase's hooks host by host; returns the problems. A failing pre_apply hook stops the phase."""
    problems: list[str] = []
    for h in keys:
        for hook in setup["hosts"][h].get(phase, []) or []:
            label = f"{h}: {phase} `{hook['run']}`"
            if not hook.get("enabled", True):
                print(f"  {label}: skipped (disabled){': ' + hook['why'] if hook.get('why') else ''}")
                continue
            argv = shlex.split(hook["run"])
            stdin = None
            if argv[0] == SWITCH_SCRIPT:
                cmd = [runner.python(h), "-", *argv[1:]] + (["--execute"] if execute else [])
            else:
                if not execute:
                    print(f"  [dry run] {label}: would run")
                    continue
                problem = hook_guard_problem(hook["run"], hosts.guard(h))
                if problem:
                    problems.append(f"{label} REFUSED: {problem}")
                    log(problems[-1])
                    if phase == "pre_apply":
                        return problems
                    continue
                cmd = ["sh", "-c", hook["run"]]
            log(f"{label}{'' if execute else ' (dry run)'}")
            try:
                if argv[0] == SWITCH_SCRIPT:
                    stdin = switch_source()
                cp = runner.run(h, cmd, input=stdin, timeout=int(hook.get("timeout_s") or HOOK_TIMEOUT_S))
            except Exception as e:  # noqa: BLE001 - a hook that cannot run is a failed hook
                problems.append(f"{label} failed: {type(e).__name__}: {e}")
                log(problems[-1])
                if phase == "pre_apply":
                    return problems
                continue
            out = ((cp.stdout or "") + (cp.stderr or "")).strip()
            for line in out.splitlines():
                print(f"    {line}")
            if cp.returncode != 0:
                tail = " | ".join(out.splitlines()[-2:])[-300:] or "(no output)"
                problems.append(f"{label} failed (exit {cp.returncode}): {tail}")
                log(problems[-1])
                if phase == "pre_apply":
                    return problems
    return problems


def validate_setup(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("hosts"), dict):
        raise ValueError("setup must be a mapping with a 'hosts' mapping")
    if not NAME_RE.match(str(data.get("name", ""))):
        raise ValueError(f"bad setup name {data.get('name')!r}")
    for host, hs in data["hosts"].items():
        if not HOST_KEY_RE.match(str(host)):
            raise ValueError(f"bad host key {host!r}")
        if not isinstance(hs, dict):
            raise ValueError(f"{host}: host section must be a mapping")
        hs.setdefault("services", [])
        hs.setdefault("allow_extra", [])
        hs.setdefault("stop", [])
        hs.setdefault("min_free_mib", None)
        for phase in HOOK_PHASES:
            hooks = hs.get(phase) or []
            if not isinstance(hooks, list):
                raise ValueError(f"{host}: {phase} must be a list of hooks")
            try:
                hs[phase] = [normalize_hook(x, phase) for x in hooks]
            except ValueError as e:
                raise ValueError(f"{host}: {e}") from None
        seen = set()
        for s in hs["services"]:
            check_unit_name(s["unit"])
            if s["unit"] in seen:
                raise ValueError(f"{host}: {s['unit']} listed twice")
            seen.add(s["unit"])
            s.setdefault("health", {"type": "systemd", "unit": s["unit"]})
            s.setdefault("health_timeout_s", DEFAULT_HEALTH_TIMEOUT_S)
            s.setdefault("gpu_on_demand", False)
        for u in hs["stop"]:
            check_unit_name(str(u))
        both = sorted(seen & set(hs["stop"]))
        if both:
            raise ValueError(f"{host}: {', '.join(both)} listed under both services and stop")
    return data


def dump_setup(setup: dict[str, Any]) -> str:
    header = ("# GPU setup (scripts/distill/setups.py). Gitignored: holds host details.\n"
              "# Never commit this file.\n")
    return header + yaml.safe_dump(setup, sort_keys=False, default_flow_style=False, width=120)


def load_setup(setups_dir: Path, name: str) -> dict[str, Any]:
    if not NAME_RE.match(name):
        raise SystemExit(f"bad setup name {name!r}")
    path = setups_dir / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"no such setup: {path}")
    return validate_setup(yaml.safe_load(path.read_text()))


def list_setups(setups_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(setups_dir.glob("*.yaml")):
        if p.stem == "hosts":
            continue
        try:
            out[p.stem] = validate_setup(yaml.safe_load(p.read_text()))
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:
            print(f"warning: skipping {p.name}: {e}", file=sys.stderr)
    return out


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as 0600, also when overwriting a looser file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchmod(fd, 0o600)  # the mode argument above applies only to a new file
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w") as f:
        f.write(text)


def last_apply_path() -> Path:
    return Path.home() / REMOTE_DIR / LAST_APPLY


def read_last_apply() -> dict[str, Any] | None:
    try:
        data = json.loads(last_apply_path().read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── status matching and plans (pure) ────────────────────────────────────────


def gpu_units(live: dict[str, Any]) -> set[str]:
    return {p["unit"] for p in live.get("procs", []) if p.get("scope") == "user" and p.get("unit")}


def _allowed_extra(unit: str | None, patterns: list[str], stop: list[str] | None = None) -> bool:
    """Whether ``unit`` may run beside the target: it matches ``patterns`` and the target does not ``stop`` it."""
    return bool(unit) and unit not in (stop or ()) and any(fnmatch.fnmatch(unit, p) for p in patterns)


def match_setup(setup: dict[str, Any], live_by_host: dict[str, dict[str, Any]]) -> list[str]:
    """Reasons the live state differs from ``setup``; empty means it matches."""
    reasons = []
    for host, hs in setup["hosts"].items():
        live = live_by_host.get(host)
        if not live or not live.get("ok"):
            reasons.append(f"{host}: not reachable")
            continue
        wanted = {s["unit"]: s for s in hs["services"]}
        on_gpu = gpu_units(live)
        for unit, svc in wanted.items():
            state = live.get("units", {}).get(unit, {}).get("active", "unknown")
            if state not in ACTIVE_STATES:
                reasons.append(f"{host}: {unit} is {state}")
            elif unit not in on_gpu and not svc.get("gpu_on_demand"):
                reasons.append(f"{host}: {unit} is not on the GPU")
        for unit in sorted(on_gpu - set(wanted)):
            if not _allowed_extra(unit, hs["allow_extra"], hs.get("stop")):
                reasons.append(f"{host}: extra GPU unit {unit}")
        for p in live.get("procs", []):
            if not (p.get("scope") == "user" and p.get("unit")) and not _allowed_extra(
                    p.get("unit"), hs["allow_extra"]):
                reasons.append(f"{host}: unmanaged GPU process {p['process']} (pid {p['pid']})")
    return reasons


@dataclass
class HostPlan:
    host: str
    stop: list[str] = field(default_factory=list)
    start: list[str] = field(default_factory=list)
    restart: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    uncaptured: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def describe_unmanaged(p: dict[str, Any]) -> str:
    head = f"GPU process {p['process']} (pid {p['pid']}, {p['vram_mib']} MiB)"
    if p.get("scope") == "other-user":
        return f"{head} runs in another account's user manager ({p.get('unit')})"
    if p.get("scope") == "user":
        return (f"{head} is in user unit {p.get('unit')}, which this tool does not stop "
                "(only .service units and distill-*.scope)")
    return f"{head} is not a user unit ({p.get('unit') or 'no unit'})"


def compute_plan(setup: dict[str, Any], live_by_host: dict[str, dict[str, Any]],
                 guards: dict[str, tuple[str, ...]] | None = None,
                 captured: dict[str, set[str]] | None = None,
                 restart_unhealthy: bool = False) -> dict[str, HostPlan]:
    """Per host: which user units to stop, start and restart to reach ``setup``.

    Everything is decided here, before anything runs. Candidates to stop are
    the units the probe reported on (GPU owners, running distill-* jobs, and
    every unit any saved setup names for that host) that are running and not
    in the target. A unit matching the target's ``allow_extra`` is left alone,
    unless the target's ``stop`` names it. A candidate matching the never-stop
    guard lands in ``refused``; with ``captured`` (an apply other than the
    restore setup), a candidate the
    restore setup does not list lands in ``uncaptured`` unless the target's
    ``stop`` names it. Either refuses the whole apply. A GPU process the tool
    cannot stop is a blocker. ``restart_unhealthy`` (a retry) restarts target
    units that run but are unhealthy or missing from the GPU.
    """
    plans = {}
    for host, hs in setup["hosts"].items():
        guard = (guards or {}).get(host, NEVER_STOP)
        plan = HostPlan(host)
        plans[host] = plan
        live = live_by_host.get(host)
        if not live or not live.get("ok"):
            plan.blockers.append("not reachable" + (f": {live.get('error')}" if live else ""))
            continue
        svcs = {s["unit"]: s for s in hs["services"]}
        units = live.get("units", {})
        on_gpu = gpu_units(live)
        running = {u for u, info in units.items() if info.get("active") in ACTIVE_STATES}
        for p in live.get("procs", []):
            unit = p.get("unit")
            if p.get("scope") == "user" and unit and stoppable(unit):
                continue  # a stop candidate (or target) through ``units``
            if _allowed_extra(unit, hs["allow_extra"]):
                continue
            plan.blockers.append(describe_unmanaged(p))
        for unit in sorted(running - set(svcs)):
            if _allowed_extra(unit, hs["allow_extra"], hs.get("stop")) or not stoppable(unit):
                continue  # not stoppable: its GPU processes are blockers above
            if is_never_stop(unit, guard):
                plan.refused.append(unit)
            elif (captured is not None and not is_distill_job(unit)
                  and unit not in captured.get(host, set()) and unit not in hs.get("stop", [])):
                plan.uncaptured.append(unit)
            else:
                plan.stop.append(unit)
        for unit, svc in svcs.items():
            if unit not in running:
                # A unit the user manager has not loaded (not installed, masked) can never start: block
                # before anything is stopped, so a swap does not take the port down for nothing. A swap
                # also blocks on "unknown" (the probe could not read it); a restore still tries.
                load = units.get(unit, {}).get("load")
                if load and load != "loaded" and (load != "unknown" or captured is not None):
                    plan.blockers.append(f"{unit} cannot start: the user manager reports LoadState {load} "
                                         "(not installed?)")
                plan.start.append(unit)
                continue
            sick = (not units.get(unit, {}).get("health", {}).get("ok")
                    or (unit not in on_gpu and not svc.get("gpu_on_demand")))
            if restart_unhealthy and sick and is_never_stop(unit, guard):
                plan.keep.append(unit)
                plan.notes.append(f"{unit} is unhealthy, but it is never-stop: not restarting it")
            elif restart_unhealthy and sick:
                plan.restart.append(unit)
            else:
                plan.keep.append(unit)
    return plans


def captured_units(setup: dict[str, Any], setups: dict[str, dict[str, Any]],
                   keys: list[str]) -> dict[str, set[str]] | None:
    """Per host, the units the restore setup brings back; None when applying it."""
    if setup["name"] == RESTORE_SETUP:
        return None
    restore = setups.get(RESTORE_SETUP) or {"hosts": {}}
    return {h: {s["unit"] for s in restore["hosts"].get(h, {}).get("services", [])} for h in keys}


def known_units(setups: dict[str, dict[str, Any]], host: str) -> list[str]:
    out = set()
    for s in setups.values():
        hs = s["hosts"].get(host, {})
        for svc in hs.get("services", []):
            out.add(svc["unit"])
        out.update(hs.get("stop", []))
    return sorted(out)


def health_specs(setups: dict[str, dict[str, Any]], host: str,
                 prefer: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for s in list(setups.values()) + ([prefer] if prefer else []):
        for svc in s["hosts"].get(host, {}).get("services", []):
            out[svc["unit"]] = svc["health"]
    return out


def verify_host(hs: dict[str, Any], live: dict[str, Any]) -> list[str]:
    """What is still wrong on one host after an apply; empty means done."""
    if not live.get("ok"):
        return [f"not reachable: {live.get('error')}"]
    problems = []
    on_gpu = gpu_units(live)
    for svc in hs["services"]:
        info = live.get("units", {}).get(svc["unit"], {})
        if info.get("active") not in ACTIVE_STATES:
            problems.append(f"{svc['unit']} is {info.get('active', 'unknown')}")
        elif not info.get("health", {}).get("ok"):
            problems.append(f"{svc['unit']} unhealthy ({info.get('health', {}).get('detail')})")
        elif svc["unit"] not in on_gpu and not svc.get("gpu_on_demand"):
            problems.append(f"{svc['unit']} has no GPU process yet")
    wanted = {s["unit"] for s in hs["services"]}
    for unit in sorted(on_gpu - wanted):
        if not _allowed_extra(unit, hs["allow_extra"], hs.get("stop")):
            problems.append(f"{unit} still holds the GPU")
    if hs.get("min_free_mib"):
        free = sum(g["vram_total_mib"] - g["vram_used_mib"] for g in live.get("gpus", []))
        if free < hs["min_free_mib"]:
            problems.append(f"only {free} MiB VRAM free, want {hs['min_free_mib']}")
    return problems


# ── commands ────────────────────────────────────────────────────────────────


def selected_hosts(hosts: Hosts, setup: dict[str, Any] | None, only: str | None) -> list[str]:
    keys = list(setup["hosts"]) if setup else list(hosts.hosts)
    if only:
        if only not in keys:
            raise SystemExit(f"host {only!r} is not in this setup/hosts file")
        keys = [only]
    return keys


def cmd_save(args: argparse.Namespace, hosts: Hosts, runner: Runner) -> int:
    path = args.setups_dir / f"{args.name}.yaml"
    if not NAME_RE.match(args.name) or args.name == "hosts":
        raise SystemExit(f"bad setup name {args.name!r}")
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists; pass --force to overwrite")
    extra = hosts.business_units if args.name == RESTORE_SETUP else {}
    keys = selected_hosts(hosts, None, args.only_host)
    live = {h: probe(runner, h, list(extra.get(h, ()))) for h in keys}
    bad = {h: lv.get("error") for h, lv in live.items() if not lv.get("ok")}
    if bad:
        for h, err in bad.items():
            print(f"{h}: probe failed: {err}", file=sys.stderr)
        return 1
    # A business_units entry is captured only when this account's user manager has
    # it loaded and running. A restore starts every captured unit, so one that is
    # not a user unit (a typo, a system unit) would fail every restore, and one that
    # is stopped would make every restore start it and status never match.
    capture: dict[str, tuple[str, ...]] = {}
    unloaded, skipped = [], []
    for h in keys:
        ok = []
        for u in extra.get(h, ()):
            info = live[h].get("units", {}).get(u, {})
            if info.get("load") != "loaded":
                unloaded.append(f"{h}: business_units entry {u} is not a unit of the SSH "
                                f"user's systemd --user manager (LoadState {info.get('load', 'unknown')})")
            elif info.get("active") not in ACTIVE_STATES:
                skipped.append(f"{h}: business_units entry {u} is {info.get('active', 'unknown')}, "
                               "so it is NOT captured (start it and save again to include it)")
            else:
                ok.append(u)
        capture[h] = tuple(ok)
    if unloaded:
        for x in unloaded:
            print(f"REFUSED: {x}", file=sys.stderr)
        print("fix business_units in hosts.yaml; nothing was saved", file=sys.stderr)
        return 1
    setup = validate_setup(build_setup(args.name, live, hosts, extra=capture))
    # Hooks are hand-written, not captured: an overwrite keeps the old file's
    # (business without its restore hook would leave the daemon on Max).
    kept = []
    if path.exists():
        try:
            old = validate_setup(yaml.safe_load(path.read_text()))
        except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:
            raise SystemExit(f"{path} does not load ({e}); its hooks would be lost. Fix or move it first")
        for h, ohs in old["hosts"].items():
            if h in setup["hosts"]:
                for phase in HOOK_PHASES:
                    if ohs.get(phase):
                        setup["hosts"][h][phase] = ohs[phase]
                        kept.append(f"{h} {phase} ({len(ohs[phase])})")
    write_private(path, dump_setup(setup))
    print(f"saved {path}" + (f"; kept hooks: {', '.join(kept)}" if kept else ""))
    for x in skipped:
        print(f"WARNING: {x}", file=sys.stderr)
    for h, hs in setup["hosts"].items():
        units = ", ".join(f"{s['unit']} ({s['vram_mib']} MiB)" for s in hs["services"]) or "nothing"
        print(f"  {h}: {units}")
        for u in capture.get(h, ()):
            state = live[h].get("units", {}).get(u, {}).get("active", "unknown")
            print(f"  {h}: {u} captured from hosts.yaml business_units (now {state})")
        for u in hs["unmanaged_gpu_processes"]:
            print(f"  {h}: unmanaged GPU process {u['process']} (pid {u['pid']}, {u['unit']})")
    return 0


def describe_last_apply(la: Any) -> str:
    if not isinstance(la, dict):
        return "unreadable"
    text = (f"{la.get('setup')} {str(la.get('result', '?')).upper()} at {la.get('finished_at')} "
            f"({la.get('attempts')} attempt(s), {la.get('where')})")
    problems = la.get("problems") or []
    if la.get("result") != "ok" and problems:
        text += ": " + "; ".join(str(p) for p in problems[:3])
    return text


def cmd_status(args: argparse.Namespace, hosts: Hosts, runner: Runner) -> int:
    setups = list_setups(args.setups_dir)
    live = {}
    for h in selected_hosts(hosts, None, args.only_host):
        live[h] = probe(runner, h, known_units(setups, h), health_specs(setups, h))
    for h, lv in live.items():
        if not lv.get("ok"):
            print(f"{h}: UNREACHABLE ({lv.get('error')})")
            continue
        for g in lv["gpus"]:
            print(f"{h}: GPU{g['index']} {g['name']} {g['vram_used_mib']}/{g['vram_total_mib']} MiB")
        vram = {}
        for p in lv["procs"]:
            key = p["unit"] if p.get("scope") == "user" and p.get("unit") else f"pid {p['pid']}"
            vram[key] = vram.get(key, 0) + p["vram_mib"]
        for unit, info in lv["units"].items():
            if info["active"] not in ACTIVE_STATES and unit not in vram:
                print(f"  {unit:<28} {info['active']}")
                continue
            ok = "healthy" if info["health"]["ok"] else "UNHEALTHY"
            print(f"  {unit:<28} {info['active']:<8} {vram.get(unit, 0):>6} MiB  "
                  f"{ok} ({info['health']['detail']})")
        for p in lv["procs"]:
            if not (p.get("scope") == "user" and p.get("unit")):
                print(f"  unmanaged pid {p['pid']} {p['process']} {p['vram_mib']} MiB")
        for t in lv.get("deadman_timers", []):
            print(f"  dead-man timer: {t}")
        for u, state in sorted((lv.get("deadman_units") or {}).items()):
            print(f"  dead-man unit: {u} {state}")
        if lv.get("last_apply"):
            print(f"  last apply run on {h}: {describe_last_apply(lv['last_apply'])}")
    if not getattr(runner, "local", False):
        mine = read_last_apply()
        if mine:
            print(f"last apply run from this machine: {describe_last_apply(mine)}")
    if not setups:
        print("no saved setups")
        return 0
    matched = []
    for name, s in setups.items():
        scoped = {h: hs for h, hs in s["hosts"].items() if h in live}
        reasons = match_setup({**s, "hosts": scoped}, live)
        if reasons:
            print(f"setup {name}: no match ({'; '.join(reasons[:4])}"
                  f"{'; ...' if len(reasons) > 4 else ''})")
        else:
            sick = [f"{h} {svc['unit']} ({info.get('health', {}).get('detail')})"
                    for h, hs in scoped.items() for svc in hs["services"]
                    for info in [live[h].get("units", {}).get(svc["unit"], {})]
                    if not info.get("health", {}).get("ok")]
            matched.append(name + (" (UNHEALTHY)" if sick else ""))
            print(f"setup {name}: MATCH" + (f", but UNHEALTHY: {'; '.join(sick)}" if sick else ""))
    print(f"active setup: {', '.join(matched) if matched else 'none'}")
    return 0


def print_plan(name: str, plans: dict[str, HostPlan], execute: bool,
               setup: dict[str, Any] | None = None) -> None:
    print(f"apply {name} ({'EXECUTE' if execute else 'dry run'}):")
    for p in plans.values():
        line = f"  {p.host}: stop {p.stop or '-'}; start {p.start or '-'}; keep {p.keep or '-'}"
        print(line + (f"; restart {p.restart}" if p.restart else ""))
        hs = (setup or {}).get("hosts", {}).get(p.host, {})
        for phase in HOOK_PHASES:
            for hook in hs.get(phase, []) or []:
                off = "" if hook.get("enabled", True) else " (disabled)"
                print(f"  {p.host}: {phase}: {hook['run']}{off}")
        for r in p.refused:
            print(f"  {p.host}: REFUSED: would stop never-stop unit {r}")
        for u in p.uncaptured:
            print(f"  {p.host}: REFUSED: would stop {u}, which the {RESTORE_SETUP} setup does not "
                  "list, so a restore would not start it again (name it under this setup's "
                  "'stop' to allow)")
        for n in p.notes:
            print(f"  {p.host}: note: {n}")
        for b in p.blockers:
            print(f"  {p.host}: BLOCKED: {b}")


def systemctl_user(runner: Runner, host: str, verb: str, unit: str,
                   guard: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    if verb not in ("stop", "start", "restart"):
        raise ValueError(f"unsupported systemctl verb {verb!r}")
    check_unit_name(unit, allow_scope=verb == "stop")
    if verb in ("stop", "restart") and is_never_stop(unit, guard):
        raise RuntimeError(f"refusing to {verb} never-stop unit {unit} on {host}")
    return runner.run(host, ["systemctl", "--user", verb, unit], timeout=STOP_TIMEOUT_S + 30)


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}] {msg}", flush=True)


def apply_once(setup: dict[str, Any], keys: list[str], runner: Runner, hosts: Hosts,
               setups: dict[str, dict[str, Any]], execute: bool,
               retry: bool = False) -> tuple[int, list[str]]:
    live = {h: probe(runner, h, known_units(setups, h), health_specs(setups, h, setup))
            for h in keys}
    scoped = {**setup, "hosts": {h: setup["hosts"][h] for h in keys}}
    plans = compute_plan(scoped, live, {h: hosts.guard(h) for h in keys},
                         captured=captured_units(setup, setups, keys), restart_unhealthy=retry)
    print_plan(setup["name"], plans, execute, scoped)
    refusals = [f"{p.host}: would stop never-stop unit {u}" for p in plans.values() for u in p.refused]
    refusals += [f"{p.host}: would stop {u}, which {RESTORE_SETUP} does not list"
                 for p in plans.values() for u in p.uncaptured]
    if refusals:
        return 2, refusals
    if any(p.blockers for p in plans.values()):
        return 1, [f"{p.host}: {b}" for p in plans.values() for b in p.blockers]
    # The daemon goes to Max, or a unit only the restore brings back stops, only with a restore already
    # scheduled that nobody has to remember.
    movers = [h for h in keys if "to-max" in switch_hooks(scoped, h, "pre_apply")]
    captured = captured_units(setup, setups, keys)
    takes_down = {h: sorted(set(plans[h].stop) & captured.get(h, set())) for h in keys} if captured else {}
    why = {h: f"{setup['name']} switches the daemon to Max there" for h in movers}
    for h, units in takes_down.items():
        if units:
            why.setdefault(h, f"{setup['name']} stops {', '.join(units)} there, which only {RESTORE_SETUP} "
                              "starts again")
    unarmed = [h for h in keys if h in why and not deadman_armed(live[h], h)]
    if unarmed:
        gaps = [f"{h}: {why[h]}, but no dead-man restore is armed "
                f"({DEADMAN_PREFIX}{h}.timer is not active): run `setups.py deadman {RESTORE_SETUP} --at "
                "<window end> --execute` first" for h in unarmed]
        if execute:
            return 2, gaps
        for g in gaps:
            print(f"  note: {g} (--execute refuses until then)")
    # Hooks: daemon_switch runs as a dry run here unless executing; other commands only on execute.
    pre = run_hooks("pre_apply", scoped, keys, runner, hosts, execute)
    if not execute:
        problems = pre + run_hooks("post_apply", scoped, keys, runner, hosts, execute)
        return (1 if problems else 0), problems
    if pre:
        log("a pre_apply hook failed: no unit is stopped or started"
            + (f"; the daemon on {', '.join(movers)} may already be on Max (an earlier hook switched it): the "
               f"armed dead-man, or `setups.py apply {RESTORE_SETUP} --execute`, moves it back" if movers else ""))
        return 1, pre
    errors: list[str] = []

    def act(h: str, verb: str, unit: str) -> None:
        # One failing unit must not end the run: the starts and retries still happen.
        log(f"{h}: {verb} {unit}")
        try:
            cp = systemctl_user(runner, h, verb, unit, hosts.guard(h))
        except Exception as e:  # noqa: BLE001 - counted as a failed attempt below
            errors.append(f"{h}: {verb} {unit} failed: {type(e).__name__}: {e}")
            log(errors[-1])
            return
        if cp.returncode != 0:
            log(f"{h}: {verb} {unit} failed: {cp.stderr.strip()[-200:]}")

    for h, p in plans.items():
        for unit in p.stop:
            act(h, "stop", unit)
    deadline = _now() + STOP_TIMEOUT_S
    for h, p in plans.items():
        while p.stop:
            lv = probe(runner, h, p.stop)
            still = [u for u in p.stop if u in gpu_units(lv)
                     or lv.get("units", {}).get(u, {}).get("active") in ACTIVE_STATES]
            if not still or _now() > deadline:
                if still:
                    errors.append(f"{h}: still running after stop: {', '.join(still)}")
                    log(errors[-1])
                break
            _sleep(POLL_S)
    for h, p in plans.items():
        for unit in p.start:
            act(h, "start", unit)
        for unit in p.restart:
            act(h, "restart", unit)
    timeout = max([s.get("health_timeout_s", DEFAULT_HEALTH_TIMEOUT_S)
                   for h in keys for s in setup["hosts"][h]["services"]] or [60])
    deadline = _now() + timeout
    failed_polls = 0
    while True:
        problems = []
        failed = []
        for h in keys:
            hs = setup["hosts"][h]
            lv = probe(runner, h, [s["unit"] for s in hs["services"]],
                       {s["unit"]: s["health"] for s in hs["services"]})
            problems += [f"{h}: {x}" for x in verify_host(hs, lv)]
            failed += [f"{h}: {s['unit']}" for s in hs["services"]
                       if lv.get("units", {}).get(s["unit"], {}).get("active") == "failed"]
        # A target unit that systemd gave up on (failed start, start limit hit) does not come back by
        # itself: stop waiting once two polls in a row see it, so the caller's retry or restore starts now.
        failed_polls = failed_polls + 1 if failed else 0
        if not problems or _now() > deadline:
            break
        if failed_polls >= 2:
            log(f"target unit(s) failed: {', '.join(failed)}; not waiting out the health timeout")
            break
        _sleep(POLL_S)
    # After the units, even when the wait reported problems: a restore hook checks
    # its own preconditions (the model server healthy) before it points anything back.
    problems += errors + run_hooks("post_apply", scoped, keys, runner, hosts, execute)
    return (0 if not problems else 1), problems


def failure_message(name: str, host: str, problems: list[str], attempts: int, result: str) -> str:
    """At most five lines: what failed where, and the first problems."""
    lines = [f"distill setups: apply {name} {result.upper()} on {host} after {attempts} attempt(s)"]
    items = [" ".join(str(p).split()) for p in problems] or ["(no detail)"]
    lines += ["- " + p[:200] for p in items[:3]]
    if len(items) > 3:
        lines.append(f"- and {len(items) - 3} more: run setups.py status")
    return "\n".join(lines) + "\n"


def notify_failure(name: str, keys: list[str], runner: Runner, hosts: Hosts,
                   problems: list[str], attempts: int, result: str) -> None:
    for h in keys:
        host = hosts.hosts.get(h)
        if not host or not host.notify_cmd:
            log(f"{h}: no notify_cmd in hosts.yaml: nobody was notified of this {result} apply")
            continue
        mine = [p for p in problems if p.startswith(h + ":")] or problems
        msg = failure_message(name, h, mine, attempts, result)
        try:
            cp = runner.run(h, ["sh", "-c", host.notify_cmd], input=msg, timeout=60)
            log(f"{h}: notify " + ("sent" if cp.returncode == 0 else f"FAILED (exit {cp.returncode})"))
        except Exception as e:  # noqa: BLE001 - a broken notifier must not hide the result
            log(f"{h}: notify FAILED: {type(e).__name__}: {e}")


def report_apply(name: str, keys: list[str], runner: Runner, hosts: Hosts, rc: int,
                 problems: list[str], attempts: int) -> None:
    """Record an --execute apply's outcome for ``status``; notify when a restore failed."""
    result = {0: "ok", 2: "refused"}.get(rc, "failed")
    record = {"setup": name, "result": result, "rc": rc, "attempts": attempts, "hosts": keys,
              "problems": [" ".join(str(p).split())[:300] for p in problems[:20]],
              "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "where": "on the host (--local)" if getattr(runner, "local", False) else "over ssh"}
    try:
        write_private(last_apply_path(), json.dumps(record, indent=1) + "\n")
    except OSError as e:
        log(f"could not write {last_apply_path()}: {e}")
    if rc != 0 and (name == RESTORE_SETUP or rc == 1):
        notify_failure(name, keys, runner, hosts, problems, attempts, result)


def cmd_apply(args: argparse.Namespace, hosts: Hosts, runner: Runner) -> int:
    setup = load_setup(args.setups_dir, args.name)
    setups = list_setups(args.setups_dir)
    setups[setup["name"]] = setup
    keys = selected_hosts(hosts, setup, args.only_host)
    if setup["name"] != RESTORE_SETUP:
        gaps = restore_gaps([setup], setups.get(RESTORE_SETUP), keys)
        if gaps:
            for g in gaps:
                print(f"REFUSED: {g}")
            print("a restore could not move the daemon back; nothing was changed")
            if args.execute:
                report_apply(args.name, keys, runner, hosts, 2, gaps, 0)
            return 2
    # --attempts 1: a forward swap whose target does not come up should fail now, so its caller restores
    # at once instead of after two more health waits with the port dark. A restore keeps its retries.
    asked = getattr(args, "attempts", None)
    attempts = (asked or 1 + RESTORE_RETRIES) if args.execute else 1
    rc, problems, attempt = 1, [], 0
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            log(f"retrying in {RETRY_DELAY_S}s")
            _sleep(RETRY_DELAY_S)
        try:
            rc, problems = apply_once(setup, keys, runner, hosts, setups, args.execute,
                                      retry=attempt > 1)
        except Exception as e:  # noqa: BLE001 - an executing run must reach retries and report
            if not args.execute:
                raise
            rc, problems = 1, [f"{type(e).__name__}: {e}"]
            log(f"attempt {attempt} raised {problems[0]}")
        if not args.execute:
            for x in problems:
                print(f"  {x}")
            return rc
        if rc == 0:
            log(f"setup {args.name} applied and healthy")
            break
        if rc == 2:
            for x in problems:
                print(f"  {x}")
            log(f"setup {args.name} REFUSED; nothing was changed")
            break
        log(f"attempt {attempt}/{attempts} failed: {'; '.join(problems)}")
    else:
        log(f"setup {args.name} FAILED after {attempts} attempts: {'; '.join(problems)}")
        movers = [h for h in keys if "to-max" in switch_hooks(setup, h, "pre_apply")]
        if movers and setup["name"] != RESTORE_SETUP:
            log(f"the daemon on {', '.join(movers)} may be on Max: the armed dead-man restores it at the window's "
                f"end, or run `setups.py apply {RESTORE_SETUP} --execute` now")
    report_apply(args.name, keys, runner, hosts, rc, problems, attempt)
    return rc


def parse_at(text: str) -> datetime:
    dt = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        from zoneinfo import ZoneInfo
        dt = dt.replace(tzinfo=ZoneInfo(NAIVE_TZ))
    return dt


def on_calendar(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def deadman_commands(name: str, host: str, when: datetime, python: str = DEADMAN_PYTHON
                     ) -> list[tuple[str, Any]]:
    """(kind, step) pairs that arm the dead-man on one host, in order.

    kind "ssh" is a remote shell string. kind "scp" is (role, remote file
    name): the caller copies its local file for that role ("script", "setup",
    "hosts", "switch") to that name in the remote directory. The timer runs
    from those copies with --local, so it needs nothing from this machine (the
    restore setup's ``daemon_switch.py`` hooks run from the "switch" copy).
    kind "check" is a remote shell string too: it runs ``status`` and the
    switch script's ``selftest`` with the same interpreter and copies first,
    so a host that cannot run them (no PyYAML, say) fails while arming instead
    of at restore time.
    """
    unit = DEADMAN_PREFIX + host
    rdir = '"$HOME/' + REMOTE_DIR + '"'
    script = (f"{shlex.quote(python)} {rdir}/setups.py --setups-dir {rdir} "
              f"--hosts-file {rdir}/hosts.yaml --local --only-host {host}")
    run = (f"systemd-run --user --unit={unit} --on-calendar={shlex.quote(on_calendar(when))} "
           f"--timer-property=AccuracySec=1s --description={shlex.quote('distill dead-man: apply ' + name)} "
           f"{script} apply {shlex.quote(name)} --execute")
    return [
        # A failed earlier run stays loaded as failed and would block systemd-run.
        ("ssh", f"systemctl --user reset-failed {unit}.timer {unit}.service 2>/dev/null; true"),
        ("ssh", f"mkdir -p {rdir} && chmod 700 {rdir}"),
        ("scp", ("script", "setups.py")),
        ("scp", ("setup", f"{name}.yaml")),
        ("scp", ("hosts", "hosts.yaml")),
        ("scp", ("switch", SWITCH_SCRIPT)),
        ("check", f"{script} status"),
        ("check", f"{shlex.quote(python)} {rdir}/{SWITCH_SCRIPT} selftest"),
        ("ssh", run),
        ("ssh", f"systemctl --user list-timers --all --no-pager {unit}.timer && "
                f"systemctl --user is-active --quiet {unit}.timer"),
    ]


def cancel_command(host: str) -> str:
    """Stop the host's dead-man timer, then prove it is gone (exit 1 if not)."""
    u = DEADMAN_PREFIX + host
    q = "systemctl --user show -p ActiveState --value"
    return (f"systemctl --user stop {u}.timer 2>/dev/null; "
            f"systemctl --user reset-failed {u}.timer {u}.service 2>/dev/null; "
            f"t=$({q} {u}.timer) && s=$({q} {u}.service) || "
            "{ echo 'cannot query the user manager'; exit 1; }; "
            f'if [ "$t" != inactive ]; then echo "timer {u}.timer is still $t"; exit 1; fi; '
            f'case "$s" in inactive|failed) ;; *) echo "restore {u}.service is $s now '
            '(running; not stopped)"; exit 1;; esac; '
            "echo 'no dead-man timer'")


def _call(argv: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(argv, 124, "", f"timeout: {e}")


def hosts_file_for(path: Path, host: str) -> str:
    """The hosts file cut down to one host (no other host's details leave here)."""
    if not path.exists():
        raise SystemExit(f"hosts file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    out = {k: v for k, v in data.items() if k not in ("hosts", "business_units")}
    out["hosts"] = {host: (data.get("hosts") or {}).get(host) or {}}
    units = (data.get("business_units") or {}).get(host)
    if units:
        out["business_units"] = {host: units}
    return ("# hosts.yaml for one host, copied by setups.py deadman. Never commit.\n"
            + yaml.safe_dump(out, sort_keys=False, default_flow_style=False))


def cmd_deadman(args: argparse.Namespace, hosts: Hosts, runner: Runner) -> int:
    if args.cancel:
        rc = 0
        for h in selected_hosts(hosts, None, args.only_host):
            cmd = cancel_command(h)
            if not args.execute:
                print(f"[dry run] {h}: {cmd}")
                continue
            cp = _call(runner.command(h, ["sh", "-c", cmd]))
            out = " ".join((cp.stdout + cp.stderr).split())
            if cp.returncode == 0:
                print(f"{h}: cancel ok ({out})")
            else:
                print(f"{h}: cancel FAILED (exit {cp.returncode}): {out}")
                rc = 1
        return rc
    if not args.name or not args.at:
        raise SystemExit("deadman needs <name> --at <ISO time> (or --cancel)")
    setup = load_setup(args.setups_dir, args.name)
    when = parse_at(args.at)
    if when <= datetime.now(timezone.utc):
        raise SystemExit(f"--at {when.isoformat()} is in the past")
    keys = selected_hosts(hosts, setup, args.only_host)
    # The dead-man is the restore nobody watches: on a host any saved setup moves to
    # Max, the setup it applies must move the daemon back.
    others = [s for n, s in list_setups(args.setups_dir).items() if n != setup["name"]]
    gaps = restore_gaps(others, setup, keys)
    if gaps:
        raise SystemExit("REFUSED: " + "; ".join(gaps))
    # The dead-man runs when nobody may be watching: a failed restore on a host
    # without notify_cmd would be recorded only on that host.
    silent = [h for h in keys if not (hosts.hosts.get(h) and hosts.hosts[h].notify_cmd)]
    if silent:
        msg = (f"no notify_cmd in hosts.yaml for {', '.join(silent)}: a failed or blocked "
               "restore there would notify nobody")
        if args.execute and not getattr(args, "allow_no_notify", False):
            raise SystemExit(f"REFUSED: {msg}. Set notify_cmd, or pass --allow-no-notify to arm anyway.")
        print(f"WARNING: {msg}")
    rc = 0
    with tempfile.TemporaryDirectory(prefix="distill-deadman-") as tmp:
        for h in keys:
            host = hosts.hosts.get(h)
            target = host.ssh if host else None
            if not target:
                raise SystemExit(f"no ssh target for {h}")
            python = (host.python if host else None) or DEADMAN_PYTHON
            # Each host gets only its own section of the setup and the hosts file.
            own = Path(tmp) / h
            write_private(own / f"{args.name}.yaml",
                          dump_setup({**setup, "hosts": {h: setup["hosts"][h]}}))
            write_private(own / "hosts.yaml", hosts_file_for(args.hosts_file, h))
            files = {"script": str(Path(__file__).resolve()),
                     "setup": str(own / f"{args.name}.yaml"), "hosts": str(own / "hosts.yaml"),
                     "switch": str(Path(__file__).resolve().with_name(SWITCH_SCRIPT))}
            print(f"{h}: dead-man 'apply {args.name}' at {on_calendar(when)} with {python}")
            for kind, step in deadman_commands(args.name, h, when, python):
                if kind == "scp":
                    role, remote = step
                    argv = ["scp", "-q", "-o", "BatchMode=yes", files[role],
                            f"{target}:{REMOTE_DIR}/{remote}"]
                    shown = f"scp <{role}: {Path(files[role]).name} for {h}> {target}:{REMOTE_DIR}/{remote}"
                else:
                    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, step]
                    shown = shlex.join(argv)
                if not args.execute:
                    print(f"  [dry run] {shown}")
                    continue
                cp = _call(argv)
                out = (cp.stdout + cp.stderr).strip()
                if out:
                    print("  " + out.replace("\n", "\n  "))
                if cp.returncode != 0:
                    why = f" ({python} cannot run the copied scripts on {h}; not arming)" if kind == "check" else ""
                    print(f"  FAILED ({kind}, exit {cp.returncode}){why}")
                    rc = 1
                    break
    return rc


def _attempts(text: str) -> int:
    n = int(text)
    if not 1 <= n <= 1 + RESTORE_RETRIES:
        raise argparse.ArgumentTypeError(f"--attempts must be 1..{1 + RESTORE_RETRIES}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="setups", description=__doc__.split("\n\n")[0])
    ap.add_argument("--setups-dir", type=Path, default=DEFAULT_SETUPS_DIR)
    ap.add_argument("--hosts-file", type=Path, default=None,
                    help="default: <setups-dir>/hosts.yaml")
    ap.add_argument("--local", action="store_true",
                    help="run host commands here instead of over SSH (dead-man timers)")
    ap.add_argument("--only-host", default=None, help="limit to one host key")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save", help="capture what runs on each GPU now (read-only)")
    s.add_argument("name")
    s.add_argument("--force", action="store_true", help="overwrite an existing setup")
    sub.add_parser("status", help="which saved setup is live, and is it healthy (read-only)")
    a = sub.add_parser("apply", help="move the hosts to a setup (dry run unless --execute)")
    a.add_argument("name")
    g = a.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True, help="print the plan (default)")
    g.add_argument("--execute", action="store_true", help="really stop/start units")
    a.add_argument("--attempts", type=_attempts, default=None,
                   help=f"--execute attempts (default {1 + RESTORE_RETRIES}; 1 = no retry, for a forward swap "
                        "whose caller restores on failure)")
    d = sub.add_parser("deadman", help="schedule an independent restore timer on each host")
    d.add_argument("name", nargs="?")
    d.add_argument("--at", help="ISO time; naive times are " + NAIVE_TZ)
    d.add_argument("--cancel", action="store_true", help="remove the dead-man timers")
    d.add_argument("--execute", action="store_true", help="really create/cancel (default: print)")
    d.add_argument("--allow-no-notify", action="store_true",
                   help="arm even on a host with no notify_cmd (a failed restore there tells nobody)")
    args = ap.parse_args(argv)
    explicit = args.hosts_file is not None
    if not explicit:
        args.hosts_file = args.setups_dir / "hosts.yaml"
    # Without a hosts file the guard is NEVER_STOP alone. That is acceptable only
    # for a read-only --local run with the default path; a named file must exist,
    # and so must the file of any run that changes units.
    required = not args.local or explicit or bool(getattr(args, "execute", False))
    hosts = load_hosts(args.hosts_file, required=required)
    if args.local:
        if not args.only_host:
            raise SystemExit("--local needs --only-host <this host's key>")
        hosts.hosts.setdefault(args.only_host, Host(args.only_host, None))
    runner = Runner(hosts, local=args.local)
    handler = {"save": cmd_save, "status": cmd_status, "apply": cmd_apply,
               "deadman": cmd_deadman}[args.cmd]
    return handler(args, hosts, runner)


if __name__ == "__main__":
    sys.exit(main())
