#!/usr/bin/env python3
"""The gym's hardened execution profile (OQ14): ``python`` / ``python3`` and
multi-line bash, run ONLY inside a locked-down bwrap jail. OFF by default
(``teacher_gen.py run --hardened-exec on``); nothing else in the gym changes.

Operator (Mon 2026-11-02, OPEN-QUESTIONS #14): allowed only if all of these
hold, with escape tests and an adversarial review before it is enabled. How each
is enforced here (``bwrap_argv`` / ``scope_argv`` / ``HELPER``):

* **no network**: a new network namespace (``--unshare-all`` includes
  ``--unshare-net``): only a downed loopback, no route anywhere, loopback
  services of the host (daemon, model servers, voice) and the tailnet
  unreachable, abstract unix sockets of the host invisible (they are per
  network namespace); no host unix socket is mounted.
* **unprivileged user**: a new user namespace mapped to uid/gid 65534
  (``--uid/--gid``), ``--cap-drop ALL``, no nested user namespaces
  (``--disable-userns``), ``no_new_privs`` (bwrap sets it: a setuid binary
  never elevates), ``--new-session`` (no TIOCSTI into a terminal),
  ``--die-with-parent``, a seccomp filter (``seccomp_program``: keyrings,
  ptrace, bpf, perf, userfaultfd, io_uring, mount / namespace calls, modules
  ... fail with EPERM; a fresh empty session keyring and ``/proc/keys``
  hidden), a new pid namespace (``--unshare-pid`` + ``--proc``:
  no host process is visible or signalable), new ipc/uts/cgroup namespaces.
* **read-only snapshot mounts and a writable tmpfs only**: ``/usr`` (and the
  ``/bin`` ``/lib`` symlinks), a few ``/etc`` loader files and the task's git
  directory are ``--ro-bind``; the task's workspace is ``--ro-bind`` at
  ``SNAPSHOT``; the ONLY writable mounts are tmpfs (``/tmp`` and the workspace
  path, size-capped; ``/`` itself is remounted read-only). No host path is ever
  bound writable. The helper copies the snapshot into the workspace tmpfs,
  runs the command there, and hands the files it changed back through a
  pipe; the gym (trusted code, after the jail is gone) imports
  them with ``apply_export``: regular files and directories only, every path
  checked inside the workspace, never through a symlink, size-capped.
* **no secrets or env tokens inside**: ``--clearenv`` + an explicit allow-list
  (``sandbox_env``); HOME is the workspace; nothing of the real home, ``/run``,
  ``/var``, ``/sys``, ``/home``, ``/root`` or the rest of ``/etc`` exists
  inside; no inherited fd but the export pipe (``close_fds``), which the
  user command never gets.
* **CPU, memory and time limits with a hard kill**: a transient systemd user
  scope per command (``systemd-run --user --scope``: MemoryMax with no swap,
  TasksMax, CPUQuota, RuntimeMaxSec with KillSignal=SIGKILL; the cgroup is
  killed again after every command), RLIMIT_CPU / FSIZE / NOFILE / CORE on the
  jail and RLIMIT_NPROC inside the user namespace, the helper's own wall-clock
  timeout (SIGKILL to the command's process group, then to every other process
  in the pid namespace) and the gym's outer timeout (SIGKILL of the jail, whose
  pid namespace dies with it). The profile fails closed: no working scope, no
  hardened commands (``probe``).

Run ``hardened_exec.py escape-check`` on the tool host for the escape matrix
(network, host paths, writes outside tmpfs, env secrets, privilege, resource
limits, process escape, namespaces). Linux only.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import resource
import shlex
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

PROFILE_VERSION = "hx1"
#: where the task's workspace is visible (read-only) inside the jail
SNAPSHOT = "/.gym-snapshot"
#: the in-namespace identity (nobody / nogroup)
JAIL_UID = 65534
JAIL_GID = 65534
#: the interpreter the helper runs under inside the jail (the host's system python, read-only)
JAIL_PYTHON = "/usr/bin/python3"
JAIL_BASH = "/bin/bash"
#: the tool output read per stream; past it the jail is killed (no unbounded buffering)
OUTPUT_LIMIT = 1 << 20


@dataclass(frozen=True)
class Limits:
    mem_bytes: int = 1 << 30          # cgroup MemoryMax (MemorySwapMax=0): tmpfs pages count here too
    tasks: int = 64                   # cgroup TasksMax (pids.max)
    nproc: int = 64                   # RLIMIT_NPROC inside the user namespace (the helper sets it)
    cpu_quota: str = "100%"           # cgroup CPUQuota: one core
    cpu_seconds: int = 20             # RLIMIT_CPU per process (SIGXCPU, then SIGKILL at the hard limit)
    fsize: int = 64 << 20             # RLIMIT_FSIZE: one file (and the export) at most 64 MiB
    nofile: int = 256                 # RLIMIT_NOFILE
    tmp_bytes: int = 256 << 20        # /tmp tmpfs size
    ws_bytes: int = 512 << 20         # workspace tmpfs size
    export_max: int = 96 << 20        # the gym reads at most this much of the export (one fsize file fits)
    grace: float = 15.0               # copy-in + export, on top of the command's own timeout
    max_entries: int = 200_000        # workspace entries the gym accepts in one export


LIMITS = Limits()

#: Host commands the gym SIMULATES from the task's fixture (teacher_gen.OPS_PROGRAMS minus the
#: harmless ones). Inside the jail they would show the jail's own world (one process, host
#: ``gym``, uid nobody, tmpfs disks), contradicting the fixture, so a routed command that runs one
#: is refused with a hint to run it as its own command (where the simulation answers it).
SIMULATED_IN_SCRIPT = frozenset({"systemctl", "journalctl", "nvidia-smi", "df", "free", "uptime", "ps", "curl",
                                 "lsblk", "mount", "findmnt", "hostname", "uname", "whoami", "id", "nproc",
                                 "lscpu", "sensors", "ss", "pgrep"})
_PY_WORD = re.compile(r"(?:^|/)python(?:3(?:\.\d+)?)?$")
#: words that put the next word in command position
_PREFIX_WORDS = frozenset({"then", "do", "else", "elif", "if", "while", "until", "!", "{", "time", "exec",
                           "nohup", "env", "sudo", "command", "builtin", "xargs", "nice", "setsid", "timeout",
                           "stdbuf", "ionice"})
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


# -- routing -------------------------------------------------------------------------------------


def _strip_heredocs(cmd: str) -> str:
    """The command without heredoc bodies (their lines are data, not commands)."""
    out: list[str] = []
    ends: list[str] = []
    for line in cmd.splitlines():
        if ends:
            if line.strip() == ends[0]:
                ends.pop(0)
            continue
        out.append(line)
        ends += [m.group(2) for m in _HEREDOC.finditer(line)]
    return "\n".join(out)


def _line_words(text: str) -> list[str]:
    lex = shlex.shlex(text, posix=True, punctuation_chars=";&|()<>")
    lex.whitespace = " \t\n"
    lex.whitespace_split = True
    lex.commenters = "#"
    words: list[str] = []
    expect = True
    skip_value = False
    for tok in lex:
        if tok and set(tok) <= set(";&|()"):
            expect = True
            continue
        if tok and set(tok) <= set("<>"):
            skip_value = True
            continue
        if skip_value:
            skip_value = False
            continue
        if not expect:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):
            continue  # an assignment before the command
        if tok.startswith("-") or tok.isdigit():
            continue  # an option or count of a prefix word (timeout 5, nice -n 5)
        words.append(tok)
        expect = tok in _PREFIX_WORDS
    return words


def command_words(cmd: str) -> list[str]:
    """Words in command position (a heuristic: quoting, line continuations, heredoc bodies
    and the usual prefixes are understood; it decides routing and the simulated-command
    hint, never what the jail allows: the jail is the guard). Each line starts a command
    unless a quote or a backslash carries it on."""
    words: list[str] = []
    buf = ""
    for line in _strip_heredocs(cmd).replace("\r", "\n").split("\n"):
        buf = f"{buf}\n{line}" if buf else line
        if buf.endswith("\\") and not buf.endswith("\\\\"):
            buf = buf[:-1] + " "
            continue
        try:
            words += _line_words(buf)
        except ValueError:
            continue  # an open quote: the next line continues it
        buf = ""
    if buf:
        try:
            words += _line_words(buf)
        except ValueError:
            pass
    return words


def wants_hardened(cmd: str) -> bool:
    """A command the hardened profile runs: multi-line, or one that runs python."""
    c = (cmd or "").strip()
    if not c:
        return False
    if "\n" in c or "\r" in c:
        return True
    return any(_PY_WORD.search(w) for w in command_words(c))


def simulated_words(cmd: str) -> list[str]:
    """Simulated host commands a routed command would run inside the jail."""
    return sorted({os.path.basename(w) for w in command_words(cmd)} & SIMULATED_IN_SCRIPT)


# -- the jail ------------------------------------------------------------------------------------


def sandbox_env(root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The ONLY environment inside the jail (``--clearenv`` first)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(root), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": "/tmp",
           "TERM": "dumb", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
    env.update(extra or {})
    return env


def bwrap_argv(root: Path, *, ro: tuple[Path, ...] = (), env: dict[str, str] | None = None,
               bwrap: str | None = None, limits: Limits = LIMITS, seccomp_fd: int | None = None) -> list[str]:
    """The jail. Read-only: /usr, the loader's /etc files, *ro* (the task's git dir) and the
    workspace snapshot at SNAPSHOT. Writable: tmpfs at /tmp and at the workspace path only."""
    root = Path(root)
    argv = [bwrap or shutil.which("bwrap") or "bwrap",
            "--unshare-all", "--unshare-user", "--unshare-net", "--unshare-pid",
            "--disable-userns",
            "--uid", str(JAIL_UID), "--gid", str(JAIL_GID),
            "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--hostname", "gym",
            "--clearenv"]
    for k, v in sorted(sandbox_env(root, env).items()):
        argv += ["--setenv", k, v]
    argv += ["--ro-bind", "/usr", "/usr"]
    for d in ("/bin", "/sbin", "/lib", "/lib64", "/lib32", "/libx32"):
        if os.path.islink(d):
            argv += ["--symlink", os.readlink(d), d]
        elif os.path.isdir(d):
            argv += ["--ro-bind", d, d]
    for f in ("/etc/alternatives", "/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime"):
        argv += ["--ro-bind-try", f, f]
    argv += ["--proc", "/proc", "--dev", "/dev",
             # the host's kernel keyrings are listed there (seccomp refuses reading them)
             "--ro-bind", "/dev/null", "/proc/keys", "--ro-bind", "/dev/null", "/proc/key-users",
             "--size", str(limits.tmp_bytes), "--tmpfs", "/tmp"]
    for extra in ro:  # after the /tmp tmpfs, which would hide a path under /tmp
        argv += ["--ro-bind", str(extra), str(extra)]
    argv += ["--ro-bind", str(root), SNAPSHOT,
             "--size", str(limits.ws_bytes), "--tmpfs", str(root),
             "--remount-ro", "/", "--chdir", "/"]
    if seccomp_fd is not None:
        argv += ["--seccomp", str(seccomp_fd)]
    return argv


def scope_argv(unit: str, runtime_s: float, limits: Limits = LIMITS) -> list[str]:
    """A transient systemd user scope with the cgroup limits (delegated to the user: no root)."""
    return [shutil.which("systemd-run") or "systemd-run", "--user", "--scope", "--quiet", "--collect",
            f"--unit={unit}",
            "-p", f"MemoryMax={limits.mem_bytes}", "-p", "MemorySwapMax=0",
            "-p", f"TasksMax={limits.tasks}", "-p", f"CPUQuota={limits.cpu_quota}",
            "-p", "OOMPolicy=continue",
            "-p", f"RuntimeMaxSec={max(1, int(runtime_s))}", "-p", "KillSignal=SIGKILL",
            "-p", "TimeoutStopSec=1", "--", shutil.which("env") or "/usr/bin/env", "-i", "PATH=/usr/bin:/bin"]


def scope_env() -> dict[str, str]:
    """What systemd-run itself needs to reach the user manager (bwrap clears it all)."""
    env = {"PATH": "/usr/bin:/bin"}
    xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    env["XDG_RUNTIME_DIR"] = xdg
    bus = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
    if bus:
        env["DBUS_SESSION_BUS_ADDRESS"] = bus
    return env


def host_limits(limits: Limits = LIMITS) -> Callable[[], None]:
    def apply() -> None:  # pragma: no cover - runs in the child
        # a new, empty session keyring: the jail never possesses the gym's (seccomp also
        # refuses every keyring call inside)
        try:
            import ctypes
            ctypes.CDLL(None, use_errno=True).syscall(250, 1, None)  # KEYCTL_JOIN_SESSION_KEYRING
        except Exception:  # noqa: BLE001 - best effort; the filter is the guard
            pass
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.fsize, limits.fsize))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.nofile, limits.nofile))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


#: x86_64 syscalls the jail never needs and the kernel's riskier surface: kernel keyrings,
#: ptrace and cross-process memory, bpf / perf / userfaultfd / io_uring, every mount and
#: namespace call, modules, kexec, clocks, swap, reboot, handle-based opens, raw I/O ports.
SECCOMP_DENY_X86_64 = {
    "keyctl": 250, "add_key": 248, "request_key": 249, "ptrace": 101, "process_vm_readv": 310,
    "process_vm_writev": 311, "kcmp": 312, "pidfd_getfd": 438, "bpf": 321, "perf_event_open": 298,
    "userfaultfd": 323, "io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427,
    "mount": 165, "umount2": 166, "pivot_root": 155, "chroot": 161, "unshare": 272, "setns": 308,
    "open_tree": 428, "move_mount": 429, "fsopen": 430, "fsconfig": 431, "fsmount": 432, "fspick": 433,
    "mount_setattr": 442, "kexec_load": 246, "kexec_file_load": 320, "init_module": 175,
    "finit_module": 313, "delete_module": 176, "reboot": 169, "swapon": 167, "swapoff": 168, "acct": 163,
    "settimeofday": 164, "clock_settime": 227, "clock_adjtime": 305, "adjtimex": 159,
    "open_by_handle_at": 304, "name_to_handle_at": 303, "personality": 135, "lookup_dcookie": 212,
    "quotactl": 179, "vhangup": 153, "syslog": 103, "iopl": 172, "ioperm": 173, "modify_ldt": 154,
    "uselib": 134, "sethostname": 170, "setdomainname": 171,
}
_AUDIT_ARCH_X86_64 = 0xC000003E
_RET_ALLOW, _RET_KILL, _RET_EPERM = 0x7FFF0000, 0x80000000, 0x00050000 | 1


def seccomp_program(machine: str | None = None) -> bytes:
    """The jail's seccomp filter (classic BPF for ``bwrap --seccomp``): a wrong
    architecture is killed, the x32 ABI and every ``SECCOMP_DENY_X86_64`` call fail with
    EPERM, everything else is allowed. x86_64 only (the tool host); elsewhere it raises,
    and the profile is unavailable (fails closed)."""
    if (machine or platform.machine()) not in ("x86_64", "AMD64"):
        raise RuntimeError(f"no seccomp filter for {machine or platform.machine()} (x86_64 only)")

    def ins(code: int, jt: int, jf: int, k: int) -> bytes:
        return struct.pack("<HBBI", code, jt, jf, k)

    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    nrs = sorted(set(SECCOMP_DENY_X86_64.values()))
    n = len(nrs)
    prog = [ins(ld, 0, 0, 4), ins(jeq, 1, 0, _AUDIT_ARCH_X86_64), ins(ret, 0, 0, _RET_KILL),
            ins(ld, 0, 0, 0), ins(jge, 0, 1, 0x40000000), ins(ret, 0, 0, _RET_EPERM)]
    for i, nr in enumerate(nrs):
        prog.append(ins(jeq, n - i, 0, nr))
    prog += [ins(ret, 0, 0, _RET_ALLOW), ins(ret, 0, 0, _RET_EPERM)]
    return b"".join(prog)


#: Runs inside the jail as the main process (pid 2; bwrap's init is pid 1): copy the snapshot in,
#: run the command with a timeout, kill every process left in the namespace, then write the
#: workspace's changed regular files and the full entry list (a tar) to the export pipe.
#: Everything in the jail is untrusted, this included: the gym validates the export.
HELPER = r'''
import json, os, signal, stat, subprocess, sys, tarfile, time
snap, ws, cwd, cmd, timeout, nproc, fd = sys.argv[1:8]
timeout, nproc, fd = float(timeout), int(nproc), int(fd)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
cp = subprocess.run(["/usr/bin/cp", "-a", "--no-preserve=ownership", snap + "/.", ws],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
if cp.returncode:
    why = cp.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
    sys.stderr.write("sandbox: could not prepare the workspace (" + why[0][-200:] + ")\n")
    sys.exit(125)
time.sleep(0.02)
with open("/tmp/.gym-mark", "w"):
    pass
mark = os.stat("/tmp/.gym-mark").st_ctime_ns
os.unlink("/tmp/.gym-mark")
if not os.path.isdir(cwd):
    cwd = ws
def limit():
    import resource
    resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
p = subprocess.Popen(["/bin/bash", "--noprofile", "--norc", "-c", cmd], cwd=cwd, stdin=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True, preexec_fn=limit)
timed_out = False
try:
    rc = p.wait(timeout=timeout)
except subprocess.TimeoutExpired:
    timed_out = True
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    p.wait()
    rc = 124
me = os.getpid()
for _ in range(50):
    left = [int(x) for x in os.listdir("/proc") if x.isdigit() and int(x) not in (1, me)]
    if not left:
        break
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.02)
if rc < 0:
    rc = 128 - rc
entries = []
try:
    with os.fdopen(fd, "wb") as out, tarfile.open(fileobj=out, mode="w|", format=tarfile.PAX_FORMAT) as tf:
        for dirpath, dirs, files in os.walk(ws):
            for name in sorted(dirs) + sorted(files):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, ws)
                try:
                    st = os.lstat(full)
                except OSError:
                    continue
                kind = ("d" if stat.S_ISDIR(st.st_mode) else "f" if stat.S_ISREG(st.st_mode)
                        else "l" if stat.S_ISLNK(st.st_mode) else "o")
                entries.append([rel, kind])
                if kind == "f" and st.st_ctime_ns >= mark:
                    ti = tarfile.TarInfo("ws/" + rel)
                    ti.size = st.st_size
                    ti.mode = 0o755 if st.st_mode & 0o100 else 0o644
                    with open(full, "rb") as fh:
                        tf.addfile(ti, fh)
        blob = json.dumps({"entries": entries, "rc": rc, "timed_out": timed_out}).encode()
        ti = tarfile.TarInfo("manifest.json")
        ti.size = len(blob)
        import io
        tf.addfile(ti, io.BytesIO(blob))
except OSError:
    sys.stderr.write("sandbox: the workspace changes were too large to keep\n")
sys.exit(rc if 0 <= rc < 256 else 1)
'''


def helper_argv(root: Path, cwd: Path, cmd: str, timeout: float, fd: int, limits: Limits = LIMITS) -> list[str]:
    return [JAIL_PYTHON, "-I", "-S", "-c", HELPER, SNAPSHOT, str(root), str(cwd), cmd, f"{timeout:g}",
            str(limits.nproc), str(fd)]


# -- importing the jail's changes ----------------------------------------------------------------


@dataclass
class ImportReport:
    ok: bool
    written: int = 0
    deleted: int = 0
    dirs: int = 0
    skipped: int = 0        # symlinks / special files the command made: never kept
    note: str = ""
    rc: int | None = None
    timed_out: bool = False


def _rel_ok(rel: str) -> bool:
    if not rel or "\x00" in rel or rel.startswith("/") or "\\" in rel:
        return False
    parts = PurePosixPath(rel).parts
    return bool(parts) and all(p not in ("", ".", "..") for p in parts) and PurePosixPath(rel).as_posix() == rel


def _host_entries(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for dirpath, dirs, files in os.walk(root):
        for name in dirs + files:
            full = os.path.join(dirpath, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            out[os.path.relpath(full, root)] = ("d" if stat.S_ISDIR(st.st_mode) else "f" if stat.S_ISREG(st.st_mode)
                                                else "l" if stat.S_ISLNK(st.st_mode) else "o")
    return out


def _parent_safe(root: Path, rel: str, make: bool) -> Path | None:
    """The host directory for *rel*'s parent, every component a real directory inside root
    (created when *make*); ``None`` if any component is a symlink or not a directory."""
    cur = root
    for part in PurePosixPath(rel).parts[:-1]:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            if not make:
                return None
            os.mkdir(cur, 0o755)
            continue
        if not stat.S_ISDIR(st.st_mode):
            return None
    return cur


def _remove(path: Path) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode):
        shutil.rmtree(path)
    else:
        os.unlink(path)


def apply_export(data: bytes, root: Path, limits: Limits = LIMITS) -> ImportReport:
    """Apply the jail's export to the host workspace *root* (trusted code; the export is not):
    directories and regular files only, every path inside root and never through a symlink.
    A truncated or malformed export changes nothing."""
    root = Path(root).resolve()
    if len(data) > limits.export_max:
        return ImportReport(False, note="the command's workspace changes were too large to keep")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            members = tf.getmembers()
            if len(members) > limits.max_entries + 1:
                return ImportReport(False, note="too many workspace entries")
            man_m = [m for m in members if m.name == "manifest.json" and m.isreg()]
            if len(man_m) != 1 or man_m[0].size > 32 << 20:
                return ImportReport(False, note="the command's workspace changes were not kept")
            mf = tf.extractfile(man_m[0])
            manifest = json.loads(mf.read() if mf else b"")
            files: dict[str, bytes] = {}
            for m in members:
                if not m.name.startswith("ws/"):
                    continue
                rel = m.name[3:]
                if not m.isreg() or not _rel_ok(rel) or m.size > limits.fsize:
                    continue
                fh = tf.extractfile(m)
                if fh is not None:
                    files[rel] = fh.read()
    except (tarfile.TarError, OSError, EOFError, ValueError, json.JSONDecodeError):
        return ImportReport(False, note="the command's workspace changes were not kept")
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or len(entries) > limits.max_entries:
        return ImportReport(False, note="the command's workspace changes were not kept")
    want: dict[str, str] = {}
    for e in entries:
        if (isinstance(e, list) and len(e) == 2 and isinstance(e[0], str) and e[1] in ("d", "f", "l", "o")
                and _rel_ok(e[0])):
            want[e[0]] = e[1]
    rc = manifest.get("rc")
    rep = ImportReport(True, rc=rc if isinstance(rc, int) else None, timed_out=manifest.get("timed_out") is True)
    have = _host_entries(root)
    # 1. deletions (and type changes), deepest first
    for rel in sorted(have, key=lambda r: r.count("/"), reverse=True):
        kind = have[rel]
        w = want.get(rel)
        if w == kind and kind in ("d", "f"):
            continue
        if w is None or w != kind:
            if _parent_safe(root, rel, make=False) is None:
                continue
            _remove(root / rel)
            rep.deleted += 1
    # 2. directories, shallowest first
    for rel in sorted((r for r, k in want.items() if k == "d"), key=lambda r: r.count("/")):
        parent = _parent_safe(root, rel, make=True)
        if parent is None:
            rep.skipped += 1
            continue
        target = parent / PurePosixPath(rel).name
        try:
            st = os.lstat(target)
            if not stat.S_ISDIR(st.st_mode):
                rep.skipped += 1
            continue
        except FileNotFoundError:
            os.mkdir(target, 0o755)
            rep.dirs += 1
    # 3. regular files the command changed
    for rel, body in sorted(files.items()):
        if want.get(rel) != "f":
            continue
        parent = _parent_safe(root, rel, make=True)
        if parent is None:
            rep.skipped += 1
            continue
        target = parent / PurePosixPath(rel).name
        try:
            st = os.lstat(target)
            if not stat.S_ISREG(st.st_mode):
                rep.skipped += 1
                continue
        except FileNotFoundError:
            pass
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
        try:
            view = memoryview(body)
            while view:
                view = view[os.write(fd, view):]
        finally:
            os.close(fd)
        rep.written += 1
    rep.skipped += sum(1 for k in want.values() if k in ("l", "o"))
    if rep.skipped:
        rep.note = "symbolic links and special files are not kept in this session"
    return rep


# -- running a command ---------------------------------------------------------------------------


@dataclass
class HxResult:
    text: str
    rc: int | None          # None: the gym's outer timeout killed it
    timed_out: bool = False
    imported: ImportReport | None = None
    unit: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _kill_scope(unit: str) -> None:
    try:
        subprocess.run([shutil.which("systemctl") or "systemctl", "--user", "kill", "--signal=SIGKILL",
                        f"{unit}.scope"], env=scope_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass


async def run(cmd: str, root: Path, cwd: Path, *, timeout: float, ro: tuple[Path, ...] = (),
              env: dict[str, str] | None = None, limits: Limits = LIMITS, apply: bool = True) -> HxResult:
    """Run *cmd* (bash) in the hardened jail over a snapshot of *root*, then import the
    workspace changes into *root* (``apply``). The export comes back over a pipe (nothing
    on disk), read at most ``limits.export_max`` bytes. Linux only."""
    root = Path(root).resolve()
    cwd = Path(cwd).resolve()
    unit = f"gym-hx-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    outer = timeout + limits.grace
    prog = seccomp_program()
    rfd, wfd = os.pipe()
    sfd_r, sfd_w = os.pipe()
    try:
        os.write(sfd_w, prog)  # a few hundred bytes: fits the pipe buffer
        os.close(sfd_w)
        argv = (scope_argv(unit, outer + 5, limits)
                + bwrap_argv(root, ro=ro, env=env, limits=limits, seccomp_fd=sfd_r)
                + helper_argv(root, cwd, cmd, timeout, wfd, limits))
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd="/", env=scope_env(), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, preexec_fn=host_limits(limits), pass_fds=(wfd, sfd_r))
    except BaseException:
        os.close(rfd)
        raise
    finally:
        os.close(wfd)
        os.close(sfd_r)
    loop = asyncio.get_running_loop()
    export = asyncio.StreamReader(limit=1 << 20)
    transport, _proto = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(export),
                                                     os.fdopen(rfd, "rb", 0))
    too_big = False

    def kill() -> None:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass

    async def read(stream: Any, cap: int) -> bytes:
        nonlocal too_big
        buf = bytearray()
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return bytes(buf)
            room = cap - len(buf)
            if room > 0:
                buf += chunk[:room]
            if len(chunk) > room:
                if stream is export:
                    too_big = True
                kill()

    try:
        try:
            out, err, data = await asyncio.wait_for(asyncio.gather(
                read(proc.stdout, OUTPUT_LIMIT), read(proc.stderr, OUTPUT_LIMIT),
                read(export, limits.export_max + 1)), outer)
            await proc.wait()
        except asyncio.TimeoutError:
            kill()
            await proc.wait()
            return HxResult("", None, True, None, unit)
        except asyncio.CancelledError:
            kill()
            await proc.wait()
            raise
    finally:
        transport.close()
        await asyncio.to_thread(_kill_scope, unit)
    text = "\n".join(p.decode("utf-8", "replace").rstrip() for p in (out, err) if p).strip()
    if len(text) > 20000:
        text = text[:20000] + "\n...[truncated]..."
    rc = proc.returncode
    if rc == 125 and not data:
        return HxResult(text, rc, False, None, unit)
    imported: ImportReport | None = None
    if apply:
        if too_big:
            imported = ImportReport(False, note="the command's workspace changes were too large to keep")
        elif data:
            imported = await asyncio.to_thread(apply_export, data, root, limits)
        else:
            imported = ImportReport(False, note="the command's workspace changes were not kept")
        if imported.note:
            text = (text + "\n" if text else "") + f"({imported.note})"
    timed_out = bool(imported and imported.timed_out) or rc == 124
    return HxResult(text, rc, timed_out, imported, unit)


# -- probe ---------------------------------------------------------------------------------------


_PROBE: dict[str, tuple[bool, str]] = {}


def probe() -> tuple[bool, str]:
    """Whether this host can run the hardened profile (bwrap with user namespaces and the
    options used, a systemd user scope with the cgroup controllers, a python inside):
    a real command, once. Fails closed."""
    if "r" in _PROBE:
        return _PROBE["r"]
    if not sys.platform.startswith("linux"):
        res = (False, "Linux only")
    elif not (shutil.which("bwrap") and shutil.which("systemd-run")):
        res = (False, "bwrap and systemd-run are required")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="gym-hx-probe-")).resolve()
        try:
            (tmp / "in.txt").write_text("in\n")
            r = asyncio.run(run("cat in.txt && echo ok > out.txt && python3 -c 'print(6*7)'\n"
                                "cat /proc/self/status | grep -c '^CapEff:\\s*0*$'",
                                tmp, tmp, timeout=20))
            good = (r.rc == 0 and r.text.split() == ["in", "42", "1"] and r.imported is not None and r.imported.ok
                    and (tmp / "out.txt").read_text() == "ok\n")
            res = (True, "ok") if good else (False, f"probe failed: rc={r.rc} out={r.text[:200]!r}")
        except Exception as exc:  # noqa: BLE001 - any failure means: not available
            res = (False, f"probe failed: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    _PROBE["r"] = res
    return res


# -- the escape matrix ---------------------------------------------------------------------------


@dataclass
class Attempt:
    area: str
    name: str
    cmd: str
    #: escaped(result, ctx) -> a reason string when the attempt ESCAPED (a failure), else ""
    check: Callable[[HxResult, dict[str, Any]], str]
    timeout: float = 20.0


def _ports_listening() -> list[int]:
    """TCP ports listening on the host's loopback (from /proc/net/tcp; no tool run)."""
    ports: set[int] = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(f).read_text().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) > 3 and parts[3] == "0A":
                ports.add(int(parts[1].rsplit(":", 1)[1], 16))
    return sorted(ports)


def _tailnet_ips() -> list[str]:
    ips: list[str] = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ips
    for m in re.finditer(r"inet (100\.(\d+)\.\d+\.\d+)/", out):
        if 64 <= int(m.group(2)) <= 127:
            ips.append(m.group(1))
    return ips


def _py(code: str) -> str:
    return "python3 -c " + shlex.quote(code)


def attempts(ctx: dict[str, Any]) -> list[Attempt]:
    """Each attempt tries to escape; ``check`` returns why it ESCAPED ("" = contained)."""
    home = ctx["home"]
    uid = ctx["uid"]
    canary = ctx["canary_name"]
    sentinel = ctx["host_sentinel"]
    ports = ctx["ports"][:12]
    tail = ctx["tailnet"][:2]

    def no(marker: str) -> Callable[[HxResult, dict[str, Any]], str]:
        return lambda r, c: f"output has {marker!r}" if marker in r.text else ""

    def connect_all(hosts: list[tuple[str, int]]) -> str:
        return _py("import socket\nfor h,p in %r:\n"
                   "  s=socket.socket(); s.settimeout(2)\n"
                   "  try: s.connect((h,p)); print('CONNECTED',h,p)\n"
                   "  except Exception as e: print('blocked',h,p,type(e).__name__)\n" % (hosts,))

    secrets_paths = [f"{home}/.ssh/id_ed25519", f"{home}/.ssh/id_rsa", f"{home}/.ssh/authorized_keys",
                     f"{home}/.ssh/known_hosts", f"{home}/.config/prometheus/env", f"{home}/.prometheus/config.yaml",
                     f"{home}/distill/secrets", f"{home}/.bash_history", "/etc/shadow", "/etc/passwd",
                     "/etc/sudoers", f"/run/user/{uid}/bus", f"/run/user/{uid}/gnupg", "/var/run/docker.sock",
                     "/run/docker.sock", f"/run/user/{uid}/keyring", "/root"]
    read_code = ("import os\nfor p in %r:\n"
                 "  try:\n"
                 "    if os.path.isdir(p): print('LISTED', p, len(os.listdir(p)))\n"
                 "    else: open(p,'rb').read(1); print('READ', p)\n"
                 "  except Exception as e: print('absent', p, type(e).__name__)\n" % (secrets_paths,))
    out: list[Attempt] = [
        # -- network
        Attempt("network", "tcp 1.1.1.1:443 (python)", connect_all([("1.1.1.1", 443), ("8.8.8.8", 53)]),
                no("CONNECTED")),
        Attempt("network", "curl https://1.1.1.1", "curl -sS -m 3 https://1.1.1.1 && echo REACHED || echo blocked",
                no("REACHED")),
        Attempt("network", f"loopback services {ports}", connect_all([("127.0.0.1", p) for p in ports]
                                                                     + [("::1", p) for p in ports[:4]]),
                no("CONNECTED")),
        Attempt("network", f"tailnet {tail or '(none found)'}",
                connect_all([(ip, p) for ip in tail for p in ([22, 8005] + ports[:4])]) if tail else "echo skip",
                no("CONNECTED")),
        Attempt("network", "DNS lookup", _py("import socket\ntry: print('RESOLVED', socket.gethostbyname('example.com'))\n"
                                            "except Exception as e: print('blocked', type(e).__name__)"),
                no("RESOLVED")),
        Attempt("network", "udp send to 1.1.1.1:53", _py(
            "import socket\ns=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
            "try: s.sendto(b'x',('1.1.1.1',53)); print('SENT')\nexcept Exception as e: print('blocked',type(e).__name__)"),
                no("SENT")),
        Attempt("network", "interfaces / routes", _py(
            "names=[l.split(':')[0].strip() for l in open('/proc/net/dev').read().splitlines()[2:]]\n"
            "routes=len(open('/proc/net/route').read().splitlines())-1\n"
            "print('IFACES', sorted(names), 'ROUTES', routes)"),
                lambda r, c: "" if "IFACES ['lo'] ROUTES 0" in r.text else "a non-loopback interface or a route exists"),
        Attempt("network", "host abstract unix sockets", _py(
            "import socket\nn=0\nfor l in open('/proc/net/unix').read().splitlines()[1:]:\n"
            "  p=l.split()\n  if len(p)>7 and p[7].startswith('@'):\n"
            "    s=socket.socket(socket.AF_UNIX)\n"
            "    try: s.connect('\\0'+p[7][1:]); print('CONNECTED', p[7]); n+=1\n    except Exception: pass\n"
            "print('abstract', n)"),
                no("CONNECTED")),
        # -- host paths
        Attempt("host paths", "read ~/.ssh, ~/.config/prometheus, ~/distill/secrets, /etc/shadow, /run/user",
                _py(read_code), lambda r, c: "a host secret path is readable" if re.search(r"^(READ|LISTED) ", r.text, re.M)
                else ""),
        Attempt("host paths", "shell reads of the same",
                "for p in " + " ".join(shlex.quote(p) for p in secrets_paths) + " " + f"{home}/.ssh/*; do "
                "cat \"$p\" >/dev/null 2>&1 && echo READ-OK \"$p\"; done; ls /home /root /var /run /etc 2>&1 | head -20",
                lambda r, c: "a host secret path is readable" if "READ-OK" in r.text or re.search(
                    r"^(shadow|sudoers|ssh|docker\.sock|user)$", r.text, re.M) else ""),
        Attempt("host paths", "host sentinel file by path", f"cat {sentinel} 2>&1",
                lambda r, c: "read the host sentinel" if c["sentinel_text"] in r.text else ""),
        Attempt("host paths", "/proc/1/environ and host pids", _py(
            "import os\nprint(sorted(int(x) for x in os.listdir('/proc') if x.isdigit()))\n"
            "try: print('ENV1', open('/proc/1/environ','rb').read()[:200])\nexcept Exception as e: print('blocked',type(e).__name__)"),
                lambda r, c: ("host pid visible" if any(int(x) > 50 for x in re.findall(r"\b(\d+)\b", r.text.splitlines()[0] if r.text else ""))
                              else "env secrets in /proc/1/environ" if re.search(r"TOKEN|KEY|SECRET|PASS|TELEGRAM|QWEN", r.text) else "")),
        Attempt("host paths", "/sys and /dev", "ls /sys 2>&1 | head -3; ls /dev",
                lambda r, c: "host devices exposed" if re.search(r"\b(sd[a-z]|nvme\d|nvidia\d|kvm|mem|vsock)\b", r.text) else ""),
        Attempt("host paths", "follow a symlink out of the workspace", f"ln -s {home}/.ssh sshlink && ls sshlink/ 2>&1; "
                f"ln -s {sentinel} s2 && cat s2 2>&1",
                lambda r, c: "symlink reached the host" if c["sentinel_text"] in r.text or "authorized_keys" in r.text else ""),
        # -- writing outside tmpfs
        Attempt("writes", "touch in /, /usr, /etc, the snapshot, the real home, /tmp-on-host",
                f"for p in /x /usr/x /etc/x {SNAPSHOT}/{canary} {home}/{canary} /proc/{canary}; do "
                f"touch $p 2>/dev/null && echo WROTE $p || echo denied $p; done; touch {canary} && echo ws-ok",
                lambda r, c: "a host file appeared" if c["host_canaries"]() else ""),
        Attempt("writes", "python writes outside tmpfs", _py(
            f"import os\nfor p in ['{SNAPSHOT}/{canary}.py', '{home}/{canary}.py', '/usr/lib/{canary}', '/{canary}']:\n"
            "  try: open(p,'w').write('x'); print('WROTE', p)\n  except Exception as e: print('denied', p, type(e).__name__)"),
                lambda r, c: "a host file appeared" if c["host_canaries"]() else ""),
        Attempt("writes", "workspace symlink planted for the import", f"ln -s {home} homelink; mkdir -p d; "
                f"ln -s {home}/.bashrc d/{canary}; echo x > homelink/{canary} 2>&1; echo planted",
                lambda r, c: "the import followed a symlink to the host" if c["host_canaries"]() or (
                    (c["root"] / "homelink").is_symlink()) else ""),
        Attempt("writes", "forged export (path traversal) over the helper's pipe", _py(
            "import os,tarfile,io,json,signal\n"
            "h=None\n"
            "for pid in os.listdir('/proc'):\n"
            "  if pid.isdigit():\n"
            "    try: a=open(f'/proc/{pid}/cmdline','rb').read().split(b'\\0')\n"
            "    except OSError: continue\n"
            f"    if a[0]==b'{JAIL_PYTHON}' and b'{SNAPSHOT}' in a: h=(int(pid), a[-2].decode())\n"
            "b=io.BytesIO(); tf=tarfile.open(fileobj=b,mode='w')\n"
            f"for n in ['ws/../../{canary}', 'ws//{canary}abs', 'ws/../{canary}']:\n"
            "  ti=tarfile.TarInfo(n); ti.size=1; tf.addfile(ti, io.BytesIO(b'x'))\n"
            f"m=json.dumps({{'entries':[['../../{canary}','f'],['/{canary}abs','f'],['../{canary}','f']],'rc':0}}).encode()\n"
            "ti=tarfile.TarInfo('manifest.json'); ti.size=len(m); tf.addfile(ti, io.BytesIO(m)); tf.close()\n"
            "print('helper', h)\n"
            "with open(f'/proc/{h[0]}/fd/{h[1]}','wb') as o: o.write(b.getvalue())\n"
            "print('FORGED'); os.kill(h[0], signal.SIGKILL)\n"),
                lambda r, c: "a forged export wrote outside the workspace" if c["host_canaries"]() else ""),
        # -- env secrets
        Attempt("env", "env has no token/key/secret/pass, QWEN, TELEGRAM",
                "env | grep -i -E 'token|key|secret|pass|qwen|telegram|dbus|ssh_auth|xdg|display|wayland|http' "
                "&& echo LEAK || echo clean; env | cut -d= -f1 | sort | tr '\\n' ' '",
                lambda r, c: "secret-like env var inside" if "LEAK" in r.text else ""),
        Attempt("env", "inherited fds", _py(
            "import os\nfor f in sorted(os.listdir('/proc/self/fd'), key=int):\n"
            "  try: print('FD', f, os.readlink('/proc/self/fd/'+f))\n  except OSError: pass"),
                lambda r, c: "an fd beyond stdin/stdout/stderr leaked in" if [
                    ln for ln in r.text.splitlines() if ln.startswith("FD ") and int(ln.split()[1]) > 2
                    and not re.fullmatch(r"/proc/\d+/fd|/proc/self/fd", ln.split(" ", 2)[2])] else ""),
        # -- privilege
        Attempt("privilege", "id / capabilities / no_new_privs", "id; grep -E '^(Cap(Inh|Prm|Eff|Bnd|Amb)|NoNewPrivs|Seccomp)' "
                "/proc/self/status",
                lambda r, c: ("uid 0 inside" if re.search(r"uid=0\b", r.text) else
                              "a capability is held" if re.search(r"Cap(Prm|Eff|Bnd|Amb):\s*0*[1-9a-f]", r.text) else
                              "no_new_privs is off" if re.search(r"NoNewPrivs:\s*0", r.text) else "")),
        Attempt("privilege", "setuid binaries (sudo, su, passwd, mount)",
                "sudo -n id 2>&1 | head -1; su -c id 2>&1 </dev/null | head -1; mount -t tmpfs none /mnt 2>&1 | head -1; "
                "unshare -r id 2>&1 | head -1; ls -l /usr/bin/sudo",
                lambda r, c: "elevated" if re.search(r"uid=0\b", r.text) else ""),
        Attempt("privilege", "ptrace / kill outside", _py(
            "import os,ctypes,signal\nlibc=ctypes.CDLL(None)\n"
            "r=libc.ptrace(16,1,0,0); print('ptrace1', r)\n"
            "try: os.kill(1, 0); print('signal pid1 ok (bwrap init)')\nexcept Exception as e: print('blocked', e)\n"
            f"try: os.kill({ctx['host_pid']}, signal.SIGTERM); print('KILLED-HOST')\n"
            "except Exception as e: print('blocked host', type(e).__name__)"),
                lambda r, c: "signalled a host process" if not c["host_alive"]() else ""),
        Attempt("privilege", "kernel keyrings / io_uring / bpf / userfaultfd (seccomp)", _py(
            "import ctypes\nlibc=ctypes.CDLL(None, use_errno=True)\n"
            "for name, args in (('keyctl-session', (250, 0, -3, 0)), ('keyctl-user-session', (250, 0, -5, 0)),\n"
            "                   ('io_uring_setup', (425, 8, 0)), ('bpf', (321, 0, 0, 0)), ('userfaultfd', (323, 0)),\n"
            "                   ('perf_event_open', (298, 0, 0, -1, -1, 0)), ('ptrace-traceme', (101, 0, 0, 0)),\n"
            "                   ('unshare-net', (272, 0x40000000)), ('mount', (165, 0, 0, 0, 0, 0))):\n"
            "  r=libc.syscall(*[ctypes.c_long(a) for a in args]); print('SYS', name, r, ctypes.get_errno())\n"
            "try: print('keys', repr(open('/proc/keys').read()[:80]))\nexcept OSError as e: print('keys hidden', e.errno)"),
                lambda r, c: ("a refused call succeeded: " + ", ".join(ok) if (ok := [
                    ln.split()[1] for ln in r.text.splitlines() if ln.startswith("SYS ") and ln.split()[2] != "-1"])
                    else "host keyrings listed" if "keyring" in r.text
                    else "" if r.text.count("SYS ") == 9 else "the probe did not run")),
        # -- resource limits
        Attempt("limits", "fork bomb", ":(){ :|:& };: ; sleep 3; echo after",
                lambda r, c: "bomb processes outlived the jail" if c["scoped"]() else "", timeout=10),
        Attempt("limits", "memory hog (2 GiB)", _py("b=[]\nwhile True: b.append(bytearray(64<<20))"),
                lambda r, c: "not killed" if r.rc == 0 else "", timeout=20),
        Attempt("limits", "tmpfs fill (1 GiB)", "dd if=/dev/zero of=/tmp/fill bs=1M count=1024 2>&1 | tail -1; "
                "dd if=/dev/zero of=fill bs=1M count=1024 2>&1 | tail -1",
                lambda r, c: "a tmpfs took 1 GiB" if "1073741824 bytes" in r.text else "", timeout=20),
        Attempt("limits", "CPU spin (no wall-clock limit of its own)", _py("while True: pass"),
                lambda r, c: "not killed" if r.rc == 0 else "", timeout=60),
        Attempt("limits", "wall clock (sleep 1000)", "sleep 1000; echo WOKE",
                lambda r, c: "not killed" if "WOKE" in r.text else "", timeout=5),
        Attempt("limits", "huge file (FSIZE)", "head -c 200M /dev/zero > big 2>&1; ls -l big | awk '{print $5}'",
                lambda r, c: "file over 64 MiB" if re.search(r"\b(\d{9,})\b", r.text) and int(
                    re.search(r"\b(\d{9,})\b", r.text).group(1)) > (64 << 20) else "", timeout=20),
        # -- process escape
        Attempt("process", "background / nohup / setsid / double fork survive?",
                "nohup sleep 300 >/dev/null 2>&1 & setsid sleep 301 & (sleep 302 &) ; "
                "python3 -c 'import os,time\nif os.fork()==0:\n  os.setsid()\n  if os.fork()==0: time.sleep(303)' ; "
                "echo started",
                lambda r, c: "a child survived the jail" if c["sleepers"]() or c["scoped"]() else "", timeout=10),
        Attempt("namespaces", "host processes / hostname / mounts", "ps -e 2>/dev/null | wc -l; hostname; "
                "cat /proc/self/mountinfo | awk '{print $5, $6}' | head -30",
                lambda r, c: ("a writable mount other than the tmpfs ones" if [
                    ln.split()[0] for ln in r.text.splitlines()[2:] if len(ln.split()) == 2
                    and ln.split()[1].split(",")[0] == "rw" and ln.split()[0] not in ("/tmp", str(c["root"]))
                    and not ln.split()[0].startswith(("/dev", "/proc"))] else
                    "host processes visible" if r.text.split()[0].isdigit() and int(r.text.split()[0]) > 8 else "")),
    ]
    return out


def escape_check(json_out: bool = False) -> int:
    """Run the escape matrix against the real profile on this host. Exit 0 when every
    attempt was contained; 1 when any escaped; 2 when the profile does not work here."""
    ok, detail = probe()
    if not ok:
        print(f"hardened profile unavailable: {detail}", file=sys.stderr)
        return 2
    home = str(Path.home())
    scratch = Path(tempfile.mkdtemp(prefix="gym-hx-escape-")).resolve()
    canary = f"hx-canary-{uuid.uuid4().hex[:10]}"
    sentinel = scratch / "host-sentinel.txt"
    sentinel_text = f"host-sentinel-{uuid.uuid4().hex}"
    sentinel.write_text(sentinel_text)
    root = scratch / "ws"
    victim = subprocess.Popen(["sleep", "600"], start_new_session=True)
    canary_paths = [Path(home) / canary, Path(home) / f"{canary}.py", Path("/") / canary, scratch / canary,
                    Path(f"/{canary}abs")]

    def host_canaries() -> list[str]:
        found = [str(p) for p in canary_paths if p.exists()]
        found += [str(p) for p in Path(home).glob(f"{canary}*")]
        found += [str(p) for p in scratch.glob(f"{canary}*")]
        return found

    def scoped() -> list[str]:
        """Host processes still in a gym-hx scope (nothing may outlive its command)."""
        left = []
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    if "/gym-hx-" in (d / "cgroup").read_text():
                        left.append(d.name)
                except OSError:
                    pass
        return left

    def sleepers() -> list[str]:
        try:
            out = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", r"^sleep 30[0-3]$"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        return out.split()

    ctx: dict[str, Any] = {"home": home, "uid": os.getuid(), "canary_name": canary, "host_sentinel": str(sentinel),
                           "sentinel_text": sentinel_text, "ports": _ports_listening(), "tailnet": _tailnet_ips(),
                           "host_pid": victim.pid, "host_canaries": host_canaries, "sleepers": sleepers, "scoped": scoped,
                           "root": root, "host_alive": lambda: victim.poll() is None}
    results: list[dict[str, Any]] = []
    try:
        for att in attempts(ctx):
            t0 = time.monotonic()
            shutil.rmtree(root, ignore_errors=True)  # every attempt starts from the same small workspace
            root.mkdir()
            (root / "notes.txt").write_text("hello\n")
            try:
                r = asyncio.run(run(att.cmd, root, root, timeout=att.timeout))
                time.sleep(0.3)
                why = att.check(r, ctx)
                if att.area == "process" and not why:
                    time.sleep(1.0)
                    why = att.check(r, ctx)
                rc, text = r.rc, r.text
            except Exception as exc:  # noqa: BLE001
                why, rc, text = "", None, f"(harness error: {exc})"
            results.append({"area": att.area, "attempt": att.name, "escaped": bool(why), "why": why, "rc": rc,
                            "seconds": round(time.monotonic() - t0, 2),
                            "evidence": " | ".join(text.splitlines()[:6])[:300]})
            for p in host_canaries():
                Path(p).unlink(missing_ok=True)
    finally:
        victim.kill()
        victim.wait()
        for pid in sleepers():
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass
        shutil.rmtree(scratch, ignore_errors=True)
    escaped = [r for r in results if r["escaped"]]
    if json_out:
        print(json.dumps({"profile": PROFILE_VERSION, "results": results, "escaped": len(escaped)}, indent=1))
    else:
        for r in results:
            print(f"{'ESCAPED' if r['escaped'] else 'contained':9s} {r['area']:11s} {r['attempt']}"
                  f"  [rc={r['rc']} {r['seconds']}s] {r['why'] or r['evidence'][:140]}")
        print(f"{len(results) - len(escaped)}/{len(results)} contained")
    return 1 if escaped else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="can this host run the hardened profile?")
    e = sub.add_parser("escape-check", help="run the escape matrix against the real profile")
    e.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        ok, detail = probe()
        print(detail)
        return 0 if ok else 2
    return escape_check(a.json)


if __name__ == "__main__":
    sys.exit(main())
