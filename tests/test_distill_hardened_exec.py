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
    # the root of the jail is read-only; the only writable mounts are the two tmpfs
    assert _pairs(argv, "--remount-ro") == [("/",)]
    assert _pairs(argv, "--tmpfs") == [("/tmp",), (str(root),)]
    assert len(_pairs(argv, "--size")) == 2
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


def test_seccomp_filter_and_keyrings(tmp_path):
    import struct

    argv = hx.bwrap_argv(tmp_path, seccomp_fd=9)
    assert _pairs(argv, "--seccomp") == [("9",)]
    assert ("/dev/null", "/proc/keys") in _pairs(argv, "--ro-bind", 2)
    assert argv.index("/proc/keys") > argv.index("--proc")
    prog = hx.seccomp_program("x86_64")
    ins = [struct.unpack("<HBBI", prog[i:i + 8]) for i in range(0, len(prog), 8)]
    deny = sorted(set(hx.SECCOMP_DENY_X86_64.values()))
    # arch check first (anything else is killed), then x32, then each denied call jumps to EPERM
    assert ins[0] == (0x20, 0, 0, 4) and ins[1][3] == 0xC000003E and ins[2] == (0x06, 0, 0, 0x80000000)
    assert ins[-2] == (0x06, 0, 0, 0x7FFF0000) and ins[-1] == (0x06, 0, 0, 0x00050001)
    jeqs = ins[6:-2]
    assert [k for _c, _t, _f, k in jeqs] == deny
    assert all(6 + i + 1 + jt == len(ins) - 1 for i, (_c, jt, _f, _k) in enumerate(jeqs))
    for name in ("keyctl", "ptrace", "bpf", "io_uring_setup", "userfaultfd", "mount", "unshare", "setns"):
        assert hx.SECCOMP_DENY_X86_64[name] in deny
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


def test_apply_export_never_leaves_the_workspace(tmp_path):
    root = _ws(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)  # a symlink already in the workspace (e.g. a setup file)
    sym = tarfile.TarInfo("ws/evil")
    sym.type, sym.linkname = tarfile.SYMTYPE, str(outside)
    hard = tarfile.TarInfo("ws/hard")
    hard.type, hard.linkname = tarfile.LNKTYPE, "/etc/passwd"
    dev = tarfile.TarInfo("ws/dev")
    dev.type = tarfile.CHRTYPE
    data = _export({"ws/../escape.txt": b"x", "ws//abs.txt": b"x", f"ws/{outside}/abs2.txt": b"x",
                    "ws/link/through.txt": b"x", "ws/keep.txt": b"k2\n"},
                   [["keep.txt", "f"], ["sub", "d"], ["sub/a.txt", "f"], ["gone.txt", "f"], ["link", "l"],
                    ["link/through.txt", "f"], ["../escape.txt", "f"], ["/abs.txt", "f"], ["evil", "l"]],
                   extra=[sym, hard, dev])
    rep = hx.apply_export(data, root)
    assert rep.ok and (root / "keep.txt").read_text() == "k2\n"
    assert list(outside.iterdir()) == [] and not (tmp_path / "escape.txt").exists()
    assert not (root / "evil").exists() and not (root / "hard").exists() and not (root / "dev").exists()
    assert (root / "link").is_symlink()  # a host symlink is never written through, nor created
    assert rep.note  # the command is told symlinks are not kept


def test_a_broken_export_changes_nothing(tmp_path):
    root = _ws(tmp_path)
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    good = _export({"ws/x.txt": b"x"}, [["x.txt", "f"]])
    for data in (good[:700], _export({"ws/x.txt": b"x"}, [], manifest=False), b"not a tar", b""):
        rep = hx.apply_export(data, root)
        assert not rep.ok and rep.note
        assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before
    rep = hx.apply_export(good, root, hx.Limits(export_max=100))
    assert not rep.ok and "too large" in rep.note


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


def test_probe_fails_closed_off_linux(monkeypatch):
    monkeypatch.setattr(hx, "_PROBE", {})
    monkeypatch.setattr(hx.sys, "platform", "darwin")
    assert hx.probe() == (False, "Linux only")
