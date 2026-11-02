"""Tests for scripts/distill/hardened_exec.py (OPEN-QUESTIONS #14: python and multi-line bash in
the gym, only inside the hardened profile, OFF by default) and its wiring in teacher_gen.py.

The argument builder, the routing and the export import run everywhere. The real jail (bwrap +
a systemd user scope) runs only where its probe passes (the Linux tool host); elsewhere those
tests skip.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


hx = _load("distill_hardened_exec", "hardened_exec.py")
tg = _load("distill_teacher_gen", "teacher_gen.py")

_real_jail: list[bool] = []


def _jail_works() -> bool:
    if not _real_jail:
        _real_jail.append(sys.platform.startswith("linux") and hx.probe()[0])
    return _real_jail[0]


needs_jail = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the hardened profile is Linux only")


# ── the argument builder ──────────────────────────────────────────────────────


def _pairs(argv: list[str], opt: str, n: int = 1) -> list[tuple[str, ...]]:
    return [tuple(argv[i + 1:i + 1 + n]) for i, a in enumerate(argv) if a == opt]


def test_bwrap_argv_has_every_isolation_option(tmp_path):
    root = tmp_path / "ws"
    gd = tmp_path / "ws.repo-git"
    argv = hx.bwrap_argv(root, ro=(gd,), bwrap="bwrap")
    for opt in ("--unshare-all", "--unshare-user", "--unshare-net", "--unshare-pid", "--disable-userns",
                "--die-with-parent", "--new-session", "--clearenv"):
        assert opt in argv, opt
    assert _pairs(argv, "--uid") == [("65534",)] and _pairs(argv, "--gid") == [("65534",)]
    assert _pairs(argv, "--cap-drop") == [("ALL",)]
    assert _pairs(argv, "--proc") == [("/proc",)]
    # the root of the jail and /dev are read-only; the only writable mounts are the size-capped tmpfs
    assert _pairs(argv, "--remount-ro") == [("/dev",), ("/",)]
    assert _pairs(argv, "--tmpfs") == [("/dev/shm",), ("/tmp",), (str(root),)]
    sizes = [int(argv[i + 1]) for i, a in enumerate(argv) if a == "--size"]
    assert sizes == [hx.LIMITS.shm_bytes, hx.LIMITS.tmp_bytes, hx.LIMITS.ws_bytes]
    for i, a in enumerate(argv):  # each --size caps the --tmpfs right after it
        if a == "--size":
            assert argv[i + 2] == "--tmpfs"
    # /dev/shm is mounted on bwrap's /dev, then /dev is made read-only
    assert argv.index("--dev") < argv.index("/dev/shm") < max(i for i, a in enumerate(argv) if a == "/dev")
    # no host path is ever bound writable (or as a device)
    for opt in ("--bind", "--bind-try", "--dev-bind", "--dev-bind-try", "--bind-fd", "--overlay", "--tmp-overlay"):
        assert opt not in argv, opt
    ro = _pairs(argv, "--ro-bind", 2)
    assert ("/usr", "/usr") in ro and (str(gd), str(gd)) in ro and (str(root), hx.SNAPSHOT) in ro
    # the git dir is mounted after the /tmp tmpfs (which would hide it), the workspace tmpfs after its snapshot
    assert argv.index(str(gd)) > argv.index("/tmp")
    tmpfs_root = max(i for i, a in enumerate(argv) if a == "--tmpfs")
    assert argv.index(hx.SNAPSHOT) < tmpfs_root
    # nothing of the host's home, runtime dirs or secrets is mounted
    home = str(Path.home())
    srcs = [s for s, _d in ro] + [s for (s,) in _pairs(argv, "--ro-bind-try")]
    assert not any(s.startswith((home + "/.", "/run", "/var", "/home/", "/root", "/etc/shadow", "/sys"))
                   for s in srcs if s != str(root) and s != str(gd)), srcs


def _bpf_run(prog: bytes, nr: int, *, arch: int = 0xC000003E, arg0: int = 0) -> int:
    """Evaluate the classic BPF seccomp program on one syscall (the subset it uses)."""
    import struct

    ins = [struct.unpack("<HBBI", prog[i:i + 8]) for i in range(0, len(prog), 8)]
    data = struct.pack("<iIQ6Q", nr, arch, 0, arg0, 0, 0, 0, 0, 0)
    pc, acc = 0, 0
    for _ in range(len(ins) + 1):
        code, jt, jf, k = ins[pc]
        if code == 0x20:
            acc = struct.unpack_from("<I", data, k)[0]
            pc += 1
        elif code == 0x15:
            pc += 1 + (jt if acc == k else jf)
        elif code == 0x35:
            pc += 1 + (jt if acc >= k else jf)
        elif code == 0x06:
            return k
        else:
            raise AssertionError(f"unexpected BPF op {code:#x}")
    raise AssertionError("no return")


ALLOW, KILL, EPERM, EAFNOSUPPORT = 0x7FFF0000, 0x80000000, 0x00050001, 0x00050000 | 97


def test_seccomp_filter_and_keyrings(tmp_path):
    argv = hx.bwrap_argv(tmp_path, seccomp_fd=9)
    assert _pairs(argv, "--seccomp") == [("9",)]
    assert ("/dev/null", "/proc/keys") in _pairs(argv, "--ro-bind", 2)
    assert argv.index("/proc/keys") > argv.index("--proc")
    prog = hx.seccomp_program("x86_64")
    for name, nr in hx.SECCOMP_DENY_X86_64.items():
        assert _bpf_run(prog, nr) == EPERM, name
    for name in ("keyctl", "ptrace", "bpf", "io_uring_setup", "userfaultfd", "mount", "unshare", "setns"):
        assert name in hx.SECCOMP_DENY_X86_64
    for nr in (0, 1, 2, 59, 57, 56, 42, 49, 231):  # read write open execve fork clone connect bind exit_group
        assert _bpf_run(prog, nr) == ALLOW, nr
    assert _bpf_run(prog, 0, arch=0x40000003) == KILL  # i386
    assert _bpf_run(prog, 0x40000000 | 41, arg0=1) == EPERM  # the x32 ABI


@pytest.mark.parametrize("nr", [41, 53])  # socket, socketpair
def test_seccomp_filters_the_socket_family(nr):
    prog = hx.seccomp_program("x86_64")
    for fam in (1, 2, 10):  # AF_UNIX, AF_INET, AF_INET6
        assert _bpf_run(prog, nr, arg0=fam) == ALLOW, fam
    for fam in (16, 38, 17, 40, 0, 3, 4, 5, 9, 29, 44, 45, 46, 255, 0xFFFF):  # netlink alg packet vsock ...
        assert _bpf_run(prog, nr, arg0=fam) == EAFNOSUPPORT, fam
    # the kernel reads the family as an int: only the low 32 bits count, as in the filter
    assert _bpf_run(prog, nr, arg0=(1 << 32) | 38) == EAFNOSUPPORT
    assert _bpf_run(prog, nr, arg0=(7 << 32) | 1) == ALLOW


def test_seccomp_program_is_x86_64_only():
    with pytest.raises(RuntimeError):
        hx.seccomp_program("aarch64")


def test_the_environment_inside_is_an_allow_list(tmp_path, monkeypatch):
    for k in ("TELEGRAM_BOT_TOKEN", "QWEN_PAYG_KEY", "OPENAI_API_KEY", "SSH_AUTH_SOCK", "DBUS_SESSION_BUS_ADDRESS",
              "PROMETHEUS_API_TOKEN", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN"):
        monkeypatch.setenv(k, "secret-value-123")
    root = tmp_path / "ws"
    argv = hx.bwrap_argv(root, env={"GIT_DIR": "/x"})
    setenv = dict(_pairs(argv, "--setenv", 2))
    assert set(setenv) == set(hx.sandbox_env(root)) | {"GIT_DIR"}
    assert setenv["HOME"] == str(root) and setenv["HOME"] != str(Path.home())
    assert "secret-value-123" not in " ".join(argv)
    # systemd-run gets only what it needs to reach the user manager, and execs bwrap with an empty env
    senv = hx.scope_env()
    assert set(senv) <= {"PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"}
    scope = hx.scope_argv("gym-hx-test", 45)
    assert scope[-3:][1:] == ["-i", "PATH=/usr/bin:/bin"] and scope[-3].endswith("env")


def test_scope_argv_sets_the_cgroup_limits():
    lim = hx.Limits()
    props = dict(p.split("=", 1) for p in (a for (a,) in _pairs(hx.scope_argv("u", 42.5, lim), "-p")))
    assert props["MemoryMax"] == str(lim.mem_bytes) and props["MemorySwapMax"] == "0"
    assert props["TasksMax"] == str(lim.tasks) and props["CPUQuota"] == lim.cpu_quota
    assert props["RuntimeMaxSec"] == "42" and props["KillSignal"] == "SIGKILL"
    assert props["OOMPolicy"] == "continue"  # the kernel kills the hog, the helper still reports
    argv = hx.scope_argv("u", 1)
    assert argv[1:4] == ["--user", "--scope", "--quiet"] and "--unit=u" in argv


def test_helper_argv_runs_the_system_python_isolated(tmp_path):
    argv = hx.helper_argv(tmp_path, tmp_path, "echo hi", 30, 7)
    assert argv[:4] == [hx.JAIL_PYTHON, "-I", "-S", "-c"] and argv[5] == hx.SNAPSHOT
    assert argv[-4:] == ["echo hi", "30", str(hx.LIMITS.nproc), "7"]
    compile(hx.HELPER, "<helper>", "exec")


# ── routing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cmd,want", [
    ("python3 -c 'print(1)'", True),
    ("python script.py", True),
    ("/usr/bin/python3 x.py", True),
    ("cat a.json | python3 -m json.tool", True),
    ("env python3 x.py", True),
    ("timeout 5 python3 x.py", True),
    ("FOO=1 python3.12 x.py", True),
    ("ls\nwc -l a", True),
    ("for f in *.md; do\n  wc -l \"$f\"\ndone", True),
    ("echo python3", False),
    ("grep -r python3 .", False),
    ("cat python.txt", False),
    ("ls -la", False),
    ("", False),
])
def test_wants_hardened(cmd, want):
    assert hx.wants_hardened(cmd) is want


def test_simulated_commands_are_found_in_command_position_only():
    assert hx.simulated_words("for s in a b; do\n  systemctl status $s\ndone") == ["systemctl"]
    assert hx.simulated_words("x=$(df -h)\necho $x") == ["df"]  # a substitution runs it too
    assert hx.simulated_words("echo 'ps and id'; python3 -c 'print(type(1), id(2))'") == []
    assert hx.simulated_words("python3 - <<'EOF'\nimport os\nps = 1\nEOF\nnvidia-smi") == ["nvidia-smi"]
    assert hx.simulated_words("cat a | sudo journalctl -u x") == ["journalctl"]


# ── importing the jail's export ───────────────────────────────────────────────


def _export(files: dict[str, bytes], entries: list[list[str]], *, extra: list[tarfile.TarInfo] = (),
            manifest: bool = True) -> bytes:
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, body in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(body)
            tf.addfile(ti, io.BytesIO(body))
        for ti in extra:
            tf.addfile(ti)
        if manifest:
            m = json.dumps({"entries": entries, "rc": 0, "timed_out": False}).encode()
            ti = tarfile.TarInfo("manifest.json")
            ti.size = len(m)
            tf.addfile(ti, io.BytesIO(m))
    return b.getvalue()


def _ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "keep.txt").write_text("keep\n")
    (root / "gone.txt").write_text("gone\n")
    (root / "sub" / "a.txt").write_text("a\n")
    return root.resolve()


def test_apply_export_writes_creates_and_deletes(tmp_path):
    root = _ws(tmp_path)
    data = _export({"ws/new/b.txt": b"b\n", "ws/sub/a.txt": b"A\n"},
                   [["keep.txt", "f"], ["sub", "d"], ["sub/a.txt", "f"], ["new", "d"], ["new/b.txt", "f"],
                    ["empty", "d"]])
    rep = hx.apply_export(data, root)
    assert rep.ok and rep.written == 2 and rep.deleted == 1
    assert (root / "keep.txt").read_text() == "keep\n" and not (root / "gone.txt").exists()
    assert (root / "sub" / "a.txt").read_text() == "A\n" and (root / "new" / "b.txt").read_text() == "b\n"
    assert (root / "empty").is_dir()


def _tree(root: Path) -> dict[str, Any]:
    """Every path under root with its content (files) or kind: a before/after snapshot."""
    out: dict[str, Any] = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        out[rel] = ("l", str(p.readlink())) if p.is_symlink() else p.read_bytes() if p.is_file() else "d"
    return out


def _unchanged(data: bytes, root: Path, limits: Any = None) -> Any:
    before = _tree(root)
    rep = hx.apply_export(data, root, limits or hx.LIMITS)
    assert not rep.ok and rep.note, rep
    assert _tree(root) == before  # nothing deleted, written or staged
    return rep


def _member(name: str, kind: bytes, **kw: Any) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type = kind
    for k, v in kw.items():
        setattr(ti, k, v)
    return ti


_BASE = [["keep.txt", "f"], ["sub", "d"], ["sub/a.txt", "f"]]  # gone.txt is missing: applying deletes it


@pytest.mark.parametrize("forged", [
    _member("ws/evil", tarfile.SYMTYPE, linkname="/etc"),
    _member("ws/hard", tarfile.LNKTYPE, linkname="/etc/passwd"),
    _member("ws/dev", tarfile.CHRTYPE),
    _member("ws/fifo", tarfile.FIFOTYPE),
    _member("ws/cont", tarfile.CONTTYPE),
    _member("ws/gnusparse", tarfile.GNUTYPE_SPARSE),
    _member("other/x", tarfile.REGTYPE),
    _member("ws/../escape.txt", tarfile.REGTYPE),
    _member("ws//abs.txt", tarfile.REGTYPE),
    _member("ws/..", tarfile.DIRTYPE),
], ids=lambda m: m.name + ":" + m.type.decode())
def test_a_forged_member_refuses_the_whole_export(tmp_path, forged):
    root = _ws(tmp_path)
    _unchanged(_export({"ws/keep.txt": b"k2\n"}, _BASE, extra=[forged]), root)
    assert (root / "gone.txt").exists() and (root / "keep.txt").read_text() == "keep\n"


@pytest.mark.parametrize("entries", [
    [*_BASE, ["../escape.txt", "f"]],
    [*_BASE, ["/abs.txt", "f"]],
    [*_BASE, ["x", "q"]],
    [*_BASE, ["sub/a.txt", "f"]],          # listed twice
    [*_BASE, ["nodir/x.txt", "f"]],        # its directory is not listed
    [*_BASE[:1], ["sub/a.txt", "f"]],      # sub itself is not listed
])
def test_a_malformed_manifest_refuses_the_whole_export(tmp_path, entries):
    root = _ws(tmp_path)
    _unchanged(_export({"ws/keep.txt": b"k2\n"}, entries), root)


def test_a_file_member_the_manifest_does_not_list_refuses(tmp_path):
    root = _ws(tmp_path)
    _unchanged(_export({"ws/keep.txt": b"k2\n", "ws/unlisted.txt": b"x"}, _BASE), root)


def test_apply_export_never_writes_through_a_host_symlink(tmp_path):
    root = _ws(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)  # a symlink already in the workspace (e.g. a setup file)
    # forged: a file listed under an entry that is a symlink (the helper never descends into one)
    _unchanged(_export({"ws/link/through.txt": b"x", "ws/keep.txt": b"k2\n"},
                       [*_BASE, ["link", "l"], ["link/through.txt", "f"]]), root)
    rep = hx.apply_export(_export({"ws/keep.txt": b"k2\n"}, [*_BASE, ["link", "l"], ["made", "l"]]), root)
    assert rep.ok and (root / "keep.txt").read_text() == "k2\n" and not (root / "gone.txt").exists()
    assert (root / "link").is_symlink()  # a host symlink is kept as it is, never created
    assert not (root / "made").exists() and rep.note  # the command is told symlinks are not kept
    # the command replaced the symlink with a real directory: the symlink goes, nothing reaches outside
    rep = hx.apply_export(_export({"ws/link/through.txt": b"x"},
                                  [*_BASE, ["link", "d"], ["link/through.txt", "f"]]), root)
    assert rep.ok and not (root / "link").is_symlink() and (root / "link" / "through.txt").read_text() == "x"
    assert list(outside.iterdir()) == []
    assert not [p for p in root.iterdir() if p.name.startswith(".gym-import-")]


def test_apply_export_replaces_files_never_writes_through_a_hard_link(tmp_path):
    root = _ws(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("host\n")
    (root / "keep.txt").unlink()
    (root / "keep.txt").hardlink_to(outside)
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w", format=tarfile.PAX_FORMAT) as tf:
        ti = tarfile.TarInfo("ws/keep.txt")
        ti.size = 4
        tf.addfile(ti, io.BytesIO(b"new\n"))
        tf.addfile(_member("ws/run.sh", tarfile.REGTYPE, size=3, mode=0o755), io.BytesIO(b"x\n\n"))
        m = json.dumps({"entries": [*_BASE, ["run.sh", "f"]], "rc": 0}).encode()
        ti = tarfile.TarInfo("manifest.json")
        ti.size = len(m)
        tf.addfile(ti, io.BytesIO(m))
    rep = hx.apply_export(b.getvalue(), root)
    assert rep.ok and rep.written == 2
    assert (root / "keep.txt").read_text() == "new\n" and outside.read_text() == "host\n"
    assert (root / "keep.txt").stat().st_nlink == 1
    assert (root / "run.sh").stat().st_mode & 0o777 == 0o755


def _sparse_forge(n: int = 4) -> bytes:
    """The adversarial review's forge: PAX members with no data whose sparse map declares 64 MiB."""
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for i in range(n):
            ti = tarfile.TarInfo(f"ws/sparse{i}.bin")
            ti.size = 0
            ti.pax_headers = {"GNU.sparse.map": "0,0", "GNU.sparse.size": str(64 << 20)}
            tf.addfile(ti)
        m = json.dumps({"entries": [*_BASE, *[[f"sparse{i}.bin", "f"] for i in range(n)]], "rc": 0}).encode()
        ti = tarfile.TarInfo("manifest.json")
        ti.size = len(m)
        tf.addfile(ti, io.BytesIO(m))
    return b.getvalue()


def test_the_sparse_member_forge_is_refused_with_nothing_written(tmp_path):
    data = _sparse_forge()
    assert len(data) < 32 << 10
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:  # what tarfile makes of it: a 64 MiB "regular" file
        m = tf.getmember("ws/sparse0.bin")
        assert m.isreg() and m.issparse() and m.size == 64 << 20
    root = _ws(tmp_path)
    rep = _unchanged(data, root)
    assert rep.written == 0
    assert not list(root.glob("sparse*")) and (root / "gone.txt").exists()


def test_a_member_over_the_byte_budget_is_refused(tmp_path):
    root = _ws(tmp_path)
    body = b"y" * 5000
    data = _export({"ws/keep.txt": b"k2\n", "ws/big.bin": body}, [*_BASE, ["big.bin", "f"]])
    rep = _unchanged(data, root, hx.Limits(fsize=4096))
    assert "too large" in rep.note
    assert hx.apply_export(data, root, hx.Limits(fsize=5000)).ok  # at the cap it is kept
    assert (root / "big.bin").read_bytes() == body


def test_a_member_whose_data_is_not_in_the_archive_is_refused(tmp_path):
    root = _ws(tmp_path)
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w", format=tarfile.PAX_FORMAT) as tf:
        ti = tarfile.TarInfo("ws/keep.txt")
        ti.size = 10
        ti.pax_headers = {"size": str(80 << 20)}  # the declared size: far past the archive's end
        tf.addfile(ti, io.BytesIO(b"0123456789"))
    _unchanged(b.getvalue(), root)


def test_the_running_byte_budget_counts_bytes_actually_read():
    budget = [0]
    out = io.BytesIO()
    hx._read_capped(io.BytesIO(b"a" * 100), 100, out, budget, 250)
    hx._read_capped(io.BytesIO(b"b" * 100), 100, out, budget, 250)
    assert budget == [200] and len(out.getvalue()) == 200
    with pytest.raises(hx._Refused, match="too large"):
        hx._read_capped(io.BytesIO(b"c" * 100), 100, None, budget, 250)
    with pytest.raises(hx._Refused):  # more bytes than the member declares
        hx._read_capped(io.BytesIO(b"d" * 50), 10, None, [0], 1 << 20)
    with pytest.raises(hx._Refused):  # fewer
        hx._read_capped(io.BytesIO(b"d" * 5), 10, None, [0], 1 << 20)


def test_too_many_members_is_refused(tmp_path):
    root = _ws(tmp_path)
    files = {f"ws/f{i}.txt": b"x" for i in range(8)}
    data = _export(files, [*_BASE, *[[f"f{i}.txt", "f"] for i in range(8)]])
    rep = _unchanged(data, root, hx.Limits(max_entries=5))
    assert "too many" in rep.note


@pytest.mark.parametrize("bad", [
    "a" * 256,                                   # one component over NAME_MAX
    "/".join(["d" * 200] * 21),                  # the whole path over PATH_MAX
    "bad\ud800name",                             # a lone surrogate: no file can have this name
], ids=["name_max", "path_max", "surrogate"])
@pytest.mark.parametrize("kind", ["d", "f"])
def test_an_unusable_name_leaves_the_target_untouched(tmp_path, bad, kind):
    root = _ws(tmp_path)
    parents = [["/".join(bad.split("/")[:i]), "d"] for i in range(1, bad.count("/") + 1)]
    entries = [*_BASE, ["new.txt", "f"], *parents, [bad, kind]]
    data = _export({"ws/new.txt": b"new\n", "ws/keep.txt": b"k2\n"}, entries)
    _unchanged(data, root)
    assert (root / "gone.txt").exists() and not (root / "new.txt").exists()


def test_an_over_long_member_name_leaves_the_target_untouched(tmp_path):
    root = _ws(tmp_path)
    name = "n" * 300
    _unchanged(_export({f"ws/{name}": b"x", "ws/keep.txt": b"k2\n"}, [*_BASE, [name, "f"]]), root)


def test_rel_ok_counts_the_workspace_path():
    assert hx._rel_ok("a/b.txt") and hx._rel_ok("x" * 255)
    assert not hx._rel_ok("x" * 256)
    assert hx._rel_ok("x" * 200, base=3800) and not hx._rel_ok("x" * 200, base=3900)
    assert hx._rel_ok("caf\udce9")  # a non-UTF-8 byte the jail's file name really had
    for bad in ("", "/a", "a/../b", "./a", "a//b", "a/", "a\x00b", "a\\b", "\ud800"):
        assert not hx._rel_ok(bad), bad


def test_a_broken_export_changes_nothing(tmp_path):
    root = _ws(tmp_path)
    good = _export({"ws/x.txt": b"x"}, [["x.txt", "f"]])
    for data in (good[:700], _export({"ws/x.txt": b"x"}, [], manifest=False), b"not a tar", b"",
                 json.dumps({"entries": []}).encode()):
        _unchanged(data, root)
    rep = _unchanged(good, root, hx.Limits(export_max=100))
    assert "too large" in rep.note


# ── the escape-check's helpers and the probe's verdict ────────────────────────


def _mnt(*rows: str) -> str:
    return "\n".join(["3", "gym", *(f"MNT {r}" for r in rows)])


_GOOD_MOUNTS = ["/ ro tmpfs - rw", "/usr ro ext4 - rw", "/dev ro tmpfs - rw,mode=755",
                "/dev/null rw devtmpfs chr rw", "/dev/pts rw devpts - rw", "/dev/shm rw tmpfs - rw,size=65536k",
                "/proc rw proc - rw", "/tmp rw tmpfs - rw,size=262144k", "/w rw tmpfs - rw,size=524288k"]


def test_mount_escape_allows_only_capped_tmpfs_and_device_nodes():
    assert hx._mount_escape(_mnt(*_GOOD_MOUNTS), "/w") == ""
    bad = {
        "/dev ro tmpfs - rw,mode=755": "/dev rw tmpfs - rw,mode=755",               # /dev writable
        "/dev/shm rw tmpfs - rw,size=65536k": "/dev/shm rw tmpfs - rw",             # uncapped
        "/tmp rw tmpfs - rw,size=262144k": "/tmp rw tmpfs - rw",
        "/usr ro ext4 - rw": "/usr rw ext4 - rw",                                   # a host bind writable
        "/dev/null rw devtmpfs chr rw": "/dev/mqueue rw mqueue - rw",               # not a device node
    }
    for good, worse in bad.items():
        rows = [worse if r == good else r for r in _GOOD_MOUNTS]
        assert hx._mount_escape(_mnt(*rows), "/w"), worse
    assert hx._mount_escape("40\ngym\n" + "\n".join(f"MNT {r}" for r in _GOOD_MOUNTS), "/w")  # host pids
    assert hx._mount_escape("", "/w") == "the probe did not run"


def _probe_result(text: str, scope: dict[str, str] | None = None, ok: bool = True) -> Any:
    lim = hx.LIMITS
    sc = {"memory.max": str(lim.mem_bytes), "memory.swap.max": "0", "pids.max": str(lim.tasks)}
    return hx.HxResult(text, 0, False, hx.ImportReport(ok), "u", {"scope": sc if scope is None else scope})


_PROBE_OUT = ("in\n42\n1\nNoNewPrivs:\t1\nSeccomp:\t2\nIFACES lo\nFAMILY 38 97\nFAMILY 16 97")


def test_probe_verdict_fails_closed(tmp_path):
    (tmp_path / "out.txt").write_text("ok\n")
    assert hx.probe_verdict(_probe_result(_PROBE_OUT), tmp_path) == ""
    for worse in (_PROBE_OUT.replace("Seccomp:\t2", "Seccomp:\t0"),
                  _PROBE_OUT.replace("IFACES lo", "IFACES eth0,lo"),
                  _PROBE_OUT.replace("FAMILY 38 97", "FAMILY 38 open"),
                  _PROBE_OUT.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0"),
                  _PROBE_OUT.replace("\n1\n", "\n0\n", 1)):
        assert hx.probe_verdict(_probe_result(worse), tmp_path), worse
    assert "cgroup" in hx.probe_verdict(_probe_result(_PROBE_OUT, {}), tmp_path)
    assert hx.probe_verdict(_probe_result(_PROBE_OUT, {"memory.max": "max", "memory.swap.max": "0",
                                                       "pids.max": "64"}), tmp_path)
    assert hx.probe_verdict(_probe_result(_PROBE_OUT, ok=False), tmp_path)


def test_escape_check_never_uses_process_patterns_or_real_secrets():
    import inspect

    src = inspect.getsource(hx.escape_check) + inspect.getsource(hx.attempts)
    assert "pgrep" not in src and "pkill" not in src and "killall" not in src
    ctx = {"home": "/scratch/home", "real_home": "/srv/realhome", "uid": 1000, "canary_name": "c",
           "canary_secret": "S", "nonce": "123456789", "host_sentinel": "/scratch/s", "ports": [], "tailnet": [],
           "host_pid": 1}
    for att in hx.attempts(ctx):
        # the real home is only ever tested for existence, never read: no real file under it is named
        for m in re.finditer(r"/srv/realhome[^\s'\"\]]*", att.cmd):
            assert m.group(0) in ("/srv/realhome/.ssh", "/srv/realhome/.config/prometheus",
                                  "/srv/realhome/distill"), (att.name, m.group(0))
    names = [a.name for a in hx.attempts(ctx)]
    assert any("sparse" in n for n in names)


def test_nonce_pids_finds_only_tagged_processes():
    if not Path("/proc/self/cmdline").exists():
        pytest.skip("no /proc here")
    import subprocess as sp

    nonce = "987654321"
    p = sp.Popen([sys.executable, "-c", "import time; time.sleep(30)", f"hx-{nonce}"])
    try:
        for _ in range(100):
            if p.pid in hx._nonce_pids(nonce):
                break
            import time

            time.sleep(0.02)
        assert p.pid in hx._nonce_pids(nonce) and os.getpid() not in hx._nonce_pids(nonce)
    finally:
        p.kill()
        p.wait()


# ── teacher_gen wiring ────────────────────────────────────────────────────────


def _task() -> Any:
    t = tg.GenTask(id="t-00001", kind="agentic", domain="other", family="test", prompt="hello")
    t.validate()
    return t


def _bash(env: Any, cmd: str) -> Any:
    from prometheus.tools.base import ToolExecutionContext

    reg = tg.build_sandbox_registry(env)
    tool = reg.get("bash")
    return asyncio.run(tool.execute(tool.input_model.model_validate({"command": cmd}),
                                    ToolExecutionContext(cwd=env.root, metadata={})))


def test_off_by_default_python_and_multiline_stay_refused(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    env = tg.SandboxEnv(root, _task())
    assert env.hardened is False
    for cmd in ("python3 -c 'print(1)'", "ls\nls"):
        res = _bash(env, cmd)
        assert res.is_error and res.output.startswith("Refused:"), cmd
    assert [tg.refusal_kind(r.detail) for r in env.log] == ["python", "multiline"]
    assert tg.Settings(out=tmp_path).hardened_exec is False
    assert "hardened_exec" not in tg.Settings(out=tmp_path).fingerprint()
    assert tg.Settings(out=tmp_path, hardened_exec=True).fingerprint()["hardened_exec"] == hx.PROFILE_VERSION


def test_cli_flag_defaults_off(tmp_path):
    base = ["run", "--builtin-smoke", "--out", str(tmp_path), "--injection-manifest", "m", "--eval-tools", "e"]
    a = tg.parse_args(base)
    assert a.hardened_exec == "off" and tg.settings_from_args(a).hardened_exec is False
    a = tg.parse_args([*base, "--hardened-exec", "on"])
    assert tg.settings_from_args(a).hardened_exec is True


def test_hardened_env_routes_and_still_refuses_simulated_commands(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    env = tg.SandboxEnv(root, _task(), jail="bwrap", hardened=True)
    seen: list[dict[str, Any]] = []

    async def fake_run(cmd: str, r: Path, cwd: Path, **kw: Any) -> Any:
        seen.append({"cmd": cmd, "root": r, "cwd": cwd, **kw})
        return hx.HxResult("42", 0, False, hx.ImportReport(True))

    monkeypatch.setattr(hx, "run", fake_run)
    res = _bash(env, "python3 -c 'print(6*7)'")
    assert not res.is_error and res.output == "42" and seen[-1]["cmd"] == "python3 -c 'print(6*7)'"
    assert seen[-1]["timeout"] <= 30 and seen[-1]["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env.log[-1].outcome == "ok" and env.log[-1].detail == hx.PROFILE_VERSION
    n = len(seen)
    for cmd in ("for u in a b; do\n systemctl status $u\ndone", "python3 x.py\ncat /proc/meminfo"):
        res = _bash(env, cmd)
        assert res.is_error and res.output.startswith("Refused:"), cmd
    assert len(seen) == n
    # a single-line command without python still goes through the old checks (and their simulation)
    res = _bash(env, "echo $HOME")
    assert res.is_error and "variables" in res.output and len(seen) == n
    assert not _bash(env, "hostname").is_error and len(seen) == n


def test_refusal_kinds():
    assert tg.refusal_kind("multi-line commands are not available in this session") == "multiline"
    assert tg.refusal_kind("`python3` is not available in this session") == "python"
    assert tg.refusal_kind("running `/usr/bin/python3` is not available in this session") == "python"
    assert tg.refusal_kind("'/etc' is outside this session's working directory (/w)") == "outside_workdir"
    assert tg.refusal_kind("command substitution and shell variables are not available here") == "shell_syntax"
    assert tg.refusal_kind("`gh` is not available in this session") == "program"


# ── the real jail (the tool host only) ────────────────────────────────────────


@needs_jail
def test_real_jail_round_trip(tmp_path):
    if not _jail_works():
        pytest.skip(f"hardened profile unavailable here: {hx.probe()[1]}")
    root = (tmp_path / "ws").resolve()
    (root / "d").mkdir(parents=True)
    (root / "in.txt").write_text("in\n")
    (root / "old.txt").write_text("old\n")
    cmd = ("python3 - <<'EOF'\nimport os\nopen('out.txt','w').write(open('in.txt').read().upper())\n"
           "os.makedirs('d/e', exist_ok=True); open('d/e/f.txt','w').write('f')\nos.remove('old.txt')\n"
           "print(os.getuid(), os.environ.get('HOME'))\nEOF\nln -s /etc/passwd pw; echo done")
    r = asyncio.run(hx.run(cmd, root, root, timeout=20))
    assert r.rc == 0 and r.text.splitlines()[:2] == [f"65534 {root}", "done"], r.text
    assert (root / "out.txt").read_text() == "IN\n" and (root / "d/e/f.txt").read_text() == "f"
    assert not (root / "old.txt").exists() and not (root / "pw").exists()


@needs_jail
def test_real_jail_through_the_gym_bash_tool(tmp_path):
    if not _jail_works():
        pytest.skip(f"hardened profile unavailable here: {hx.probe()[1]}")
    root = tmp_path / "ws"
    root.mkdir()
    (root / "data.csv").write_text("a,1\nb,2\n")
    env = tg.SandboxEnv(root, _task(), jail="bwrap", hardened=True)
    res = _bash(env, "python3 - <<'EOF'\nimport csv\nrows = list(csv.reader(open('data.csv')))\n"
                     "open('sum.txt', 'w').write(str(sum(int(r[1]) for r in rows)))\nprint(len(rows))\nEOF\n"
                     "cat sum.txt")
    assert not res.is_error and res.output.split() == ["2", "3"], res.output
    assert (env.root / "sum.txt").read_text() == "3" and env.log[-1].detail == hx.PROFILE_VERSION
    res = _bash(env, "python3 -c 'import sys; sys.exit(3)'")
    assert res.is_error and res.metadata["returncode"] == 3


@needs_jail
def test_real_escape_matrix_is_contained(capsys):
    if not _jail_works():
        pytest.skip(f"hardened profile unavailable here: {hx.probe()[1]}")
    rc = hx.escape_check(json_out=True)
    report = json.loads(capsys.readouterr().out)
    escaped = [r for r in report["results"] if r["escaped"]]
    assert rc == 0 and not escaped, escaped
    assert {r["area"] for r in report["results"]} >= {"network", "host paths", "writes", "env", "privilege",
                                                       "limits", "process", "namespaces"}


@needs_jail
def test_real_jail_refuses_alg_and_netlink_sockets(tmp_path):
    if not _jail_works():
        pytest.skip(f"hardened profile unavailable here: {hx.probe()[1]}")
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    cmd = ("python3 - <<'EOF'\nimport socket\n"
           "for name, fam, kind in (('ALG', 38, socket.SOCK_SEQPACKET), ('NETLINK', 16, socket.SOCK_RAW),\n"
           "                        ('PACKET', 17, socket.SOCK_RAW), ('VSOCK', 40, socket.SOCK_STREAM)):\n"
           "    try:\n        socket.socket(fam, kind).close(); print(name, 'open')\n"
           "    except OSError as e:\n        print(name, e.errno)\n"
           "a, b = socket.socketpair(); a.close(); b.close(); print('UNIX ok')\n"
           "socket.socket(socket.AF_INET).close(); print('INET ok')\nEOF")
    r = asyncio.run(hx.run(cmd, root, root, timeout=20))
    assert r.rc == 0, r.text
    assert r.text.splitlines() == ["ALG 97", "NETLINK 97", "PACKET 97", "VSOCK 97", "UNIX ok", "INET ok"], r.text


@needs_jail
def test_real_jail_dev_is_read_only_and_shm_is_capped(tmp_path):
    if not _jail_works():
        pytest.skip(f"hardened profile unavailable here: {hx.probe()[1]}")
    root = (tmp_path / "ws").resolve()
    root.mkdir()
    # two files under the 64 MiB per-file FSIZE limit, together over the /dev/shm cap
    cmd = ("touch /dev/x 2>/dev/null && echo DEV-WRITABLE; mkdir /dev/d 2>/dev/null && echo DEV-WRITABLE; "
           "echo hi > /dev/null && echo null-ok; "
           "head -c 40M /dev/zero > /dev/shm/a 2>/dev/null; head -c 40M /dev/zero > /dev/shm/b 2>/dev/null; "
           "echo shm $(cat /dev/shm/a /dev/shm/b | wc -c); grep -c 'Seccomp:.2' /proc/self/status | sed 's/^/seccomp /'")
    r = asyncio.run(hx.run(cmd, root, root, timeout=20))
    lines = r.text.splitlines()
    assert "DEV-WRITABLE" not in r.text and "null-ok" in lines and "seccomp 1" in lines, r.text
    shm = int(next(ln.split()[1] for ln in lines if ln.startswith("shm ")))
    assert 40 << 20 <= shm <= hx.LIMITS.shm_bytes, r.text


def test_virtual_files_match_normalised_paths():
    assert tg.virtual_paths_in("cat /proc/meminfo") == ["/proc/meminfo"]
    assert tg.virtual_paths_in("cat /proc//meminfo; head /proc/./loadavg") == ["/proc/loadavg", "/proc/meminfo"]
    assert tg.virtual_paths_in("python3 -c \"open('/proc/self/../uptime')\"") == ["/proc/uptime"]
    assert tg.virtual_paths_in("cat /proc/meminfo2 /etc/hostname.bak") == []
    assert tg.virtual_paths_in("echo hi") == []


def test_probe_fails_closed_off_linux(monkeypatch):
    monkeypatch.setattr(hx, "_PROBE", {})
    monkeypatch.setattr(hx.sys, "platform", "darwin")
    assert hx.probe() == (False, "Linux only")
