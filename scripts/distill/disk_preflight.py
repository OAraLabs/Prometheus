#!/usr/bin/env python3
"""The disk-space preflight every 4090 job runs before it starts (stdlib only).

Operator, Sun 2026-11-01 ~12:00 ET: Phase 6 stopped on the 4090's disk (37 GB free, its check wanted ~60), so
"add a disk-space preflight to every 4090 job so this can't recur silently". Each job declares the GiB it needs
before it starts. Below that the job does not start: one line goes to ``~/distill/logs/disk-preflight.log`` and
the operator gets a Telegram (``~/distill/notify.sh``, stdin, at most 5 lines). It never stops silently.

    disk_preflight.py check --job NAME [--need-gib N] [--item LABEL:GIB[:PATH]]... [--floor-gib F]
                            [--path DIR] [--notify PATH] [--log PATH] [--no-notify] [--detail TEXT]

What a job needs = ``--need-gib`` (its fixed need: logs, a server's scratch) + every ``--item`` still to be
written + ``--floor-gib`` (default 10: the free space kept after the job; post_train.py stops below it).
An item is one output of a stage still to run. When its PATH exists, only the part not yet written counts
(``max(0, GIB - size of PATH)``), so a resumed job does not count again what an earlier attempt already made.
Exit 0: enough room (logged). Exit 28 (ENOSPC): refused (logged and Telegrammed). Exit 2: usage.

Python callers (quantize.py, post_train.py, train_full.py, job_runner.py, capacity.py) use ``preflight()`` or,
when the disk is another host's, ``plan()`` + ``verdict()`` + ``refusal_text()`` with their own probe.
A refusal Telegram for the same job is sent at most once per ``RESEND_S`` (a job that retries every few
minutes does not flood the chat); the log line is written every time.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

GIB = 1024 ** 3
FLOOR_GIB = 10.0  # post_train.py's DISK_FLOOR_GIB: kept free after every job
EXIT_REFUSED = 28  # ENOSPC
NOTIFY_LINES = 5
RESEND_S = 1800
DEFAULT_LOG = "~/distill/logs/disk-preflight.log"
DEFAULT_NOTIFY = "~/distill/notify.sh"


@dataclass(frozen=True)
class Item:
    """One output a stage still has to write: ``gib`` in all; ``path`` is where it lands (absent: all of it)."""
    label: str
    gib: float
    path: Optional[str] = None


@dataclass
class Verdict:
    job: str
    ok: bool
    free_gib: float
    need_gib: float
    lines: List[str] = field(default_factory=list)  # the plan, one line per counted part

    def summary(self) -> str:
        word = "ok" if self.ok else "REFUSED"
        return f"disk {word}: {self.job} needs {self.need_gib:.1f} GiB, {self.free_gib:.1f} GiB free"


def parse_item(text: str) -> Item:
    """``LABEL:GIB`` or ``LABEL:GIB:PATH`` (PATH may hold colons)."""
    parts = text.split(":", 2)
    if len(parts) < 2 or not parts[0].strip():
        raise ValueError(f"--item {text!r}: LABEL:GIB[:PATH]")
    try:
        gib = float(parts[1])
    except ValueError:
        raise ValueError(f"--item {text!r}: GIB is not a number") from None
    if gib < 0:
        raise ValueError(f"--item {text!r}: GIB is negative")
    path = parts[2] if len(parts) == 3 and parts[2] else None
    return Item(parts[0].strip(), gib, path)


def path_bytes(path: str) -> int:
    """A file's size, or the sum over a directory's files (symlinks count as themselves, never followed)."""
    p = Path(os.path.expanduser(path))
    try:
        st = p.lstat()
    except OSError:
        return 0
    if not p.is_dir() or p.is_symlink():
        return st.st_size
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def remaining_gib(item: Item, size_of=path_bytes) -> float:
    if not item.path:
        return item.gib
    have = size_of(item.path) / GIB
    return max(0.0, item.gib - have)


def plan(items: Sequence[Item], *, need_gib: float = 0.0, floor_gib: float = FLOOR_GIB,
         size_of=path_bytes) -> Tuple[float, List[str]]:
    """(GiB needed, the plan's lines): the fixed need, every item's unwritten part, the floor."""
    lines: List[str] = []
    total = 0.0
    if need_gib > 0:
        lines.append(f"fixed {need_gib:.1f}")
        total += need_gib
    for it in items:
        r = remaining_gib(it, size_of)
        if it.path and r < it.gib:
            lines.append(f"{it.label} {r:.1f} of {it.gib:.1f} (present: {it.gib - r:.1f})")
        else:
            lines.append(f"{it.label} {r:.1f}")
        total += r
    if floor_gib > 0:
        lines.append(f"floor {floor_gib:.1f}")
        total += floor_gib
    return total, lines


def free_gib(path: str) -> float:
    st = os.statvfs(os.path.expanduser(path))
    return st.f_bavail * st.f_frsize / GIB


def verdict(job: str, free: float, need: float, lines: Sequence[str] = ()) -> Verdict:
    return Verdict(job=job, ok=free >= need, free_gib=free, need_gib=need, lines=list(lines))


def refusal_text(v: Verdict, *, host: str = "", path: str = "~", detail: str = "") -> str:
    """The operator's Telegram (<= 5 lines): what did not start, why, the plan, what to do."""
    host = host or socket.gethostname().split(".")[0]
    out = [f"{host}: {v.job} NOT started: {v.free_gib:.1f} GiB free under {path}, it needs {v.need_gib:.1f} GiB.",
           ("Plan (GiB): " + ", ".join(v.lines))[:300] if v.lines else "",
           detail[:300] if detail else "",
           "Nothing was changed. Free space there (old checkpoints, staged copies), then rerun or re-arm it."]
    return "\n".join(x for x in out if x)


def log_line(log: str, v: Verdict, extra: str = "") -> None:
    p = Path(os.path.expanduser(log))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job": v.job, "ok": v.ok,
                                 "free_gib": round(v.free_gib, 2), "need_gib": round(v.need_gib, 2),
                                 "plan": v.lines, **({"note": extra} if extra else {})}) + "\n")
    except OSError:
        pass


def _stamp(log: str, job: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in job)[:80]
    return Path(os.path.expanduser(log)).parent / f"disk-preflight.{safe}.sent"


def send(text: str, notify: str, *, job: str = "", log: str = DEFAULT_LOG, resend_s: float = RESEND_S,
         now: Optional[float] = None) -> str:
    """Telegram ``text`` through notify.sh (stdin, <= 5 lines): 'sent', 'failed', 'recent' (sent for this job
    less than ``resend_s`` ago) or 'no-notify' (no such script)."""
    now = time.time() if now is None else now
    stamp = _stamp(log, job) if job else None
    if stamp is not None:
        try:
            if now - stamp.stat().st_mtime < resend_s:
                return "recent"
        except OSError:
            pass
    path = os.path.expanduser(notify)
    if not os.path.isfile(path):
        return "no-notify"
    msg = "\n".join(text.splitlines()[:NOTIFY_LINES]) + "\n"
    try:
        p = subprocess.run([path, str(NOTIFY_LINES)], input=msg, capture_output=True, text=True, timeout=90)
        ok = p.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ok = False
    if ok and stamp is not None:
        try:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
            os.utime(stamp, (now, now))
        except OSError:
            pass
    return "sent" if ok else "failed"


def preflight(job: str, *, need_gib: float = 0.0, items: Sequence[Item] = (), floor_gib: float = FLOOR_GIB,
              path: str = "~", notify: Optional[str] = DEFAULT_NOTIFY, log: str = DEFAULT_LOG,
              detail: str = "", host: str = "") -> Verdict:
    """This host's disk: plan, decide, log; on a refusal also Telegram (unless ``notify`` is None)."""
    need, lines = plan(items, need_gib=need_gib, floor_gib=floor_gib)
    v = verdict(job, free_gib(path), need, lines)
    note = ""
    if not v.ok and notify:
        note = "telegram " + send(refusal_text(v, host=host, path=path, detail=detail), notify, job=job, log=log)
    log_line(log, v, note)
    return v


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="disk_preflight", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="decide, log, and on a refusal Telegram (exit 0 ok, 28 refused)")
    c.add_argument("--job", required=True)
    c.add_argument("--need-gib", type=float, default=0.0, help="the job's fixed need")
    c.add_argument("--item", action="append", default=[], help="LABEL:GIB[:PATH] an output still to write")
    c.add_argument("--floor-gib", type=float, default=FLOOR_GIB, help="kept free after the job")
    c.add_argument("--path", default="~", help="the filesystem the job writes to")
    c.add_argument("--notify", default=DEFAULT_NOTIFY)
    c.add_argument("--no-notify", action="store_true", help="log only (a dry run, a test)")
    c.add_argument("--log", default=DEFAULT_LOG)
    c.add_argument("--detail", default="", help="one more line for the Telegram")
    c.add_argument("--host", default="")
    a = ap.parse_args(argv)
    try:
        items = [parse_item(x) for x in a.item]
    except ValueError as e:
        print(f"disk_preflight: {e}", file=sys.stderr)
        return 2
    if a.need_gib < 0 or a.floor_gib < 0:
        print("disk_preflight: --need-gib and --floor-gib must not be negative", file=sys.stderr)
        return 2
    v = preflight(a.job, need_gib=a.need_gib, items=items, floor_gib=a.floor_gib, path=a.path,
                  notify=None if a.no_notify else a.notify, log=a.log, detail=a.detail, host=a.host)
    print(v.summary() + (" (" + ", ".join(v.lines) + ")" if v.lines else ""))
    return 0 if v.ok else EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
