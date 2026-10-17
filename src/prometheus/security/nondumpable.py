"""nondumpable — keep a same-user process out of the daemon's /proc entries.

WHY THIS EXISTS
---------------
Under the shipped systemd unit the daemon's secrets arrive through
``EnvironmentFile=``, so they sit in the environment block the process STARTED
with — and on Linux that block stays readable at ``/proc/<pid>/environ`` to any
process of the same user. env_scrub removes secret-named variables from the
environment a model-written shell starts with; it does nothing about that shell
reading the daemon's block by path. tasks/worker_auth.py and tasks/child_env.py
already say the same of a worker's parent.

``prctl(PR_SET_DUMPABLE, 0)`` makes the kernel treat this process like a setuid
one for introspection: ``/proc/<pid>/environ``, ``mem``, ``maps``, ``fd`` and
the ``exe``/``cwd`` links become root-owned, and a same-user process gets
EACCES. ``cmdline``, ``stat`` and ``status`` stay world-readable, so ``ps``,
``pgrep`` and the daemon lock's start-time check (gateway/status.py) are
unaffected. The process can still read its own ``/proc/self`` entries except
``environ`` — nothing here reads that.

WHAT IT COSTS — stated so the switch is not over-read
-----------------------------------------------------
* scripts/deploy.sh can no longer read ``/proc/<MainPID>/maps``. B5 asks the
  daemon instead (``runtime`` on ``/api/status``, context/environment.py
  ``runtime_report``), which reads its own maps.
* ``lsof -p``, ``ls /proc/<pid>/fd`` and ``readlink /proc/<pid>/exe`` on the
  daemon need sudo. py-spy and gdb already did under Yama ``ptrace_scope=1``
  (Ubuntu's default); root is unaffected.
* The daemon writes no core dump.

WHAT IT DOES NOT DO
-------------------
* It does not cover children. ``execve`` resets the flag, so a worker, a cron
  job or a tool subprocess started with secrets in its environment is as
  readable as before.
* It is Linux-only. macOS has no equivalent for ``sysctl(KERN_PROCARGS2)``;
  there the daemon loads the env file after exec (daemon.py ``load_env_file``),
  so its starting block holds no keys to begin with.

Source: novel code for Prometheus, 2026-10-17.
License: MIT
"""

from __future__ import annotations

import ctypes
import os
import sys

PR_GET_DUMPABLE = 3
PR_SET_DUMPABLE = 4

#: What :func:`make_nondumpable` did in this process, for the boot log and the
#: ``runtime`` report: "not-attempted", "set", "unsupported" or "failed: …".
STATE = "not-attempted"


def _prctl():
    """libc's prctl, or None. ``CDLL(None)`` is the running program's own
    symbols, which include libc's on glibc and musl alike."""
    try:
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


def make_nondumpable() -> str:
    """Clear this process's dumpable flag (Linux). Returns the new :data:`STATE`.

    Never raises: a daemon that cannot set the flag still boots, and says so.
    """
    global STATE
    if not sys.platform.startswith("linux"):
        STATE = "unsupported"
        return STATE
    prctl = _prctl()
    if prctl is None:
        STATE = "failed: libc prctl not found"
        return STATE
    if prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        STATE = f"failed: prctl errno {err} ({os.strerror(err)})"
        return STATE
    if prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        STATE = "failed: still dumpable after prctl"
        return STATE
    STATE = "set"
    return STATE


def is_dumpable() -> bool | None:
    """The kernel's answer for this process now; None where there is no prctl."""
    if not sys.platform.startswith("linux"):
        return None
    prctl = _prctl()
    if prctl is None:
        return None
    return prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0
