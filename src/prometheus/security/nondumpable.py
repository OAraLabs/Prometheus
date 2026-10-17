"""nondumpable — mark the daemon process non-dumpable (Linux defense-in-depth).

WHAT THIS DOES
--------------
``prctl(PR_SET_DUMPABLE, 0)`` tells the kernel to treat this process like a
setuid one for introspection: another process of the same user can no longer
read its process memory, its memory map, its open-file list or its
``exe``/``cwd`` links — those become root-owned and return EACCES. The
world-readable identity files (command line, scheduling ``stat``, ``status``)
are unchanged, so ``ps``, ``pgrep`` and the daemon lock's start-time check
(gateway/status.py) keep working. The process can still introspect itself.

This is a hardening layer, not a boundary the rest of the system relies on; a
host where it cannot be applied still runs, and run_daemon logs that it did
not.

WHAT IT COSTS — stated so the switch is not over-read
-----------------------------------------------------
* scripts/deploy.sh can no longer read the daemon's memory map directly, so
  its B5 verify asks the running daemon instead (``runtime`` on
  ``/api/status``, context/environment.py ``runtime_report``), which inspects
  itself.
* ``lsof -p``, the process's open-file list and its ``exe`` link now need sudo
  for another user to read. py-spy and gdb already did under Yama
  ``ptrace_scope=1`` (Ubuntu's default); root is unaffected.
* The daemon writes no core dump.

SCOPE
-----
* It applies to the daemon process only. ``execve`` resets the flag, so this
  says nothing about child processes.
* It is Linux-only; on macOS it is a no-op (there is no equivalent control).

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
