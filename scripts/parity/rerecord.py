"""Re-record parity scenarios: the primary live, every other side from a stand-in.

    .venv/bin/python scripts/parity/rerecord.py SCENARIO [SCENARIO ...] \\
        [--live-alt model_switch=http://127.0.0.1:11434] \\
        [--tolerate-c2 repaired_tool_call:alt --tolerate-c2 hosted_route:hosted]

Run from a clone's root on the machine that reaches the primary.

- **The primary's URL** is read in-process from the deploy config (`model.base_url`), so it
  never appears on a command line or in output.
- **Every other side** is answered by `parity.standin.StandIn` from the COMMITTED exchanges
  (`git show HEAD:…`), which the recording then overwrites in the working tree. A side named
  by `--live-alt` goes to that URL instead. `model_switch` needs its alt live for the
  session-title race; see docs on re-recording.
- **A new scenario** has no committed exchanges. `--standin-from NEW=RECORDED` answers its
  other sides' PROBES (the alt's boot-time /api/tags, /api/ps, /api/show) from RECORDED's
  committed exchanges and nothing else: a completion on a stand-in side is refused, since
  none was ever committed for NEW. Without it a new scenario is refused before any attempt.
- **The hosted key.** The harness requires a key for a hosted side. It is given a fake one in
  its own variable (`PARITY_STANDIN_KEY`), which only ever reaches the stand-in; no real key
  is read.

The stop rule, per scenario:
- At most `--attempts` attempts (default 5). A scenario that fails them all stops the run.
- A parity lock held by another run is waited on, never broken, and a lock wait is not an attempt.
- No attempt starts inside `--closed` (default 06:15-10:00, local time).

Each kept sample passed its scenario's `require` rule. Its chat replies are printed for the
recorder to check against the question before committing: faithfulness is the recorder's
job, not `require`'s.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402

from parity import cli  # noqa: E402
from parity.instance import DEFAULT_ROOT  # noqa: E402
from parity.model_server import COMPLETIONS_PATHS, Exchange  # noqa: E402
from parity.runner import build_config, uses_hosted  # noqa: E402
from parity.scenarios import BY_NAME  # noqa: E402
from parity.standin import StandIn, c2_reverted  # noqa: E402

FAKE_KEY_ENV = "PARITY_STANDIN_KEY"
DEPLOY_CONFIG = Path("~/prometheus-deploy/config/prometheus.yaml").expanduser()
LOCK_HELD = "another parity run holds"


class _Tee(io.TextIOBase):
    """stderr, also kept: cli.main reports a held parity lock there, as exit code 2."""

    def __init__(self, stream) -> None:
        self.stream, self.kept = stream, io.StringIO()

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.kept.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream.flush()


def _committed(name: str) -> dict | None:
    """The scenario's committed trace; None when HEAD has none (a new scenario)."""
    proc = subprocess.run(["git", "show", f"HEAD:tests/fixtures/parity/{name}.trace.json"],
                          cwd=cli.SRC_ROOT, capture_output=True, text=True)
    return json.loads(proc.stdout) if proc.returncode == 0 else None


def standin_exchanges(name: str, seed: str | None = None) -> list[Exchange]:
    """What the stand-in answers ``name``'s other sides from.

    A recorded scenario: its own committed exchanges. A new one has none, so it
    answers the PROBES in ``seed``'s committed exchanges (every golden's daemon
    probes the same alt at boot) and nothing else — every completion on a
    stand-in side is refused, because none was ever committed for ``name``."""
    own = _committed(name)
    if own is not None:
        return [Exchange.from_json(d) for d in own["exchanges"]]
    seeded = _committed(seed) if seed else None
    if seeded is None:
        raise ValueError(f"{name} has no committed trace, and no recorded scenario was "
                         f"named to seed its stand-in's probes (--standin-from)")
    return [ex for ex in (Exchange.from_json(d) for d in seeded["exchanges"])
            if not (ex.method == "POST" and ex.path in COMPLETIONS_PATHS)]


def _in_closed_window(spec: str, now: datetime | None = None) -> bool:
    start, end = (int(p.replace(":", "")) for p in spec.split("-"))
    hm = int((now or datetime.now()).strftime("%H%M"))
    return start <= hm < end


def _replies(root: Path, name: str) -> list[str]:
    raw = root.parent / f"{root.name}.recorded-raw" / f"{name}.json"
    try:
        steps = json.loads(raw.read_text())["steps"]
    except (OSError, ValueError, KeyError):
        return []
    return [str(s.get("reply", "")) for s in steps if s.get("op") == "chat"]


def record_one(name: str, primary_url: str, *, live_alt: str | None, tolerate: set[str],
               root: Path, seed: str | None = None) -> tuple[int, bool]:
    """One attempt: (the harness's exit code, 0 = saved; whether the parity lock was held)."""
    exchanges = standin_exchanges(name, seed)
    # Whether a hosted side exists is the scenario's to say (the config this
    # recording builds), not a committed trace's: a new scenario has none.
    hosted = uses_hosted(build_config(cli.SRC_ROOT, BY_NAME[name]))
    labels = ["alt"] + (["hosted"] if hosted else [])
    standin = StandIn(exchanges, labels,
                      tolerate={label: c2_reverted for label in labels if label in tolerate})
    standin.start()
    argv = ["--root", str(root), "record", "--scenario", name,
            "--upstream-primary", primary_url,
            "--upstream-alt", live_alt or standin.url("alt"), "--keep-failed"]
    if hosted:
        os.environ[FAKE_KEY_ENV] = "parity-stand-in-no-real-key"
        argv += ["--upstream-hosted", standin.url("hosted"), "--hosted-key-env", FAKE_KEY_ENV]
    strict = "stand-in, strict" + (f", probes from {seed}" if seed else "")
    sides = {"alt": "live" if live_alt else ("stand-in, C2-tolerant" if "alt" in tolerate
                                             else strict)}
    if hosted:
        sides["hosted"] = "stand-in, C2-tolerant" if "hosted" in tolerate else strict
    print(f"[rerecord] {name}: primary live; {sides}", flush=True)
    tee = _Tee(sys.stderr)
    try:
        with contextlib.redirect_stderr(tee):
            rc = cli.main(argv)
    finally:
        standin.stop()
    print(f"[rerecord] {name}: stand-in answered {standin.summary()}", flush=True)
    return rc, rc == 2 and LOCK_HELD in tee.kept.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--live-alt", action="append", default=[], metavar="SCENARIO=URL")
    ap.add_argument("--tolerate-c2", action="append", default=[], metavar="SCENARIO:SIDE")
    ap.add_argument("--standin-from", action="append", default=[], metavar="NEW=RECORDED",
                    help="a scenario with no committed trace: answer its other sides' probes "
                         "from RECORDED's committed exchanges, and refuse every completion")
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--closed", default="06:15-10:00", help="local HH:MM-HH:MM, no attempt starts inside")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args(argv)

    live = dict(item.split("=", 1) for item in args.live_alt)
    tolerate: dict[str, set[str]] = {}
    for item in args.tolerate_c2:
        scen, side = item.split(":", 1)
        tolerate.setdefault(scen, set()).add(side)
    seeds = dict(item.split("=", 1) for item in args.standin_from)
    # Settled before anything is read or started: a mistake here costs nothing.
    for name in args.scenarios:
        if name not in BY_NAME:
            print(f"[rerecord] {name}: no such scenario", flush=True)
            return 2
        recorded = _committed(name) is not None
        if not recorded and name not in seeds:
            print(f"[rerecord] {name} has no committed trace: name a recorded scenario whose "
                  f"probes its stand-in answers (--standin-from {name}=<recorded>)", flush=True)
            return 2
        if recorded and name in seeds:
            print(f"[rerecord] {name} has a committed trace; --standin-from is only for a "
                  f"new scenario", flush=True)
            return 2
        if name in seeds and _committed(seeds[name]) is None:
            print(f"[rerecord] {seeds[name]} has no committed trace to seed {name}'s "
                  f"stand-in from", flush=True)
            return 2
    primary_url = yaml.safe_load(DEPLOY_CONFIG.read_text())["model"]["base_url"]   # never printed
    if not (isinstance(primary_url, str) and primary_url.startswith("http")):
        print("[rerecord] no primary URL in the deploy config", flush=True)
        return 2

    for name in args.scenarios:
        failures = 0
        while True:
            if _in_closed_window(args.closed):
                print(f"[rerecord] STOP: inside the closed window {args.closed}; "
                      f"{name} and everything after it not recorded", flush=True)
                return 3
            print(f"[rerecord] ===== {name} attempt {failures + 1} "
                  f"{datetime.now():%H:%M:%S}", flush=True)
            rc, lock_held = record_one(name, primary_url, live_alt=live.get(name),
                                       tolerate=tolerate.get(name, set()), root=args.root,
                                       seed=seeds.get(name))
            if lock_held:
                print("[rerecord] the parity lock is held by another run; waiting 60 s "
                      "(not an attempt)", flush=True)
                time.sleep(60)
                continue
            if rc == 0:
                for i, reply in enumerate(_replies(args.root, name), 1):
                    print(f"[rerecord] {name} reply {i}: {reply!r}", flush=True)
                break
            failures += 1
            if failures >= args.attempts:
                print(f"[rerecord] STOP RULE: {name} failed {failures} attempts in a row; "
                      f"nothing after it recorded", flush=True)
                return 1
    print("[rerecord] all scenarios saved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
