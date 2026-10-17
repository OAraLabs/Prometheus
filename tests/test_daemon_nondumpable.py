"""The daemon is non-dumpable, and still reports what it loaded (security/nondumpable.py).

Under the shipped systemd unit the secrets come in through ``EnvironmentFile=``,
so they sit in the block the daemon started with, at ``/proc/<pid>/environ``,
readable by any same-user process — a model's shell included, wherever the
bubblewrap write floor is not wrapping it. ``PR_SET_DUMPABLE=0`` makes that
file (and ``maps``, ``mem``, ``fd``) root-owned.

The child test is the real thing: a separate process clears the flag, and this
process — same user, the way a model's shell would be — tries to read it.
Linux only, and skipped as root, which is exempt by design.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.context import environment
from prometheus.context.environment import _maps_evidence, runtime_report
from prometheus.security import nondumpable

SRC = Path(__file__).resolve().parent.parent / "src"
LINUX = sys.platform.startswith("linux")
MARK = "PROBE_FAKE_" + "VALUE=not-a-secret-42"   # letters only for the pre-commit scanner

_CHILD = """
import json, sys
from prometheus.security.nondumpable import make_nondumpable
from prometheus.context.environment import runtime_report
state = make_nondumpable()
print(json.dumps({"state": state, "runtime": runtime_report()}), flush=True)
sys.stdin.read()
"""


@pytest.fixture
def nondumpable_child():
    """A same-user process that has cleared its dumpable flag; stopped by its own Popen."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p)}
    name, value = MARK.split("=", 1)
    env[name] = value
    proc = subprocess.Popen([sys.executable, "-s", "-c", _CHILD], env=env, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        line = proc.stdout.readline()
        assert line, "the child printed nothing"
        yield proc.pid, json.loads(line)
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


@pytest.mark.skipif(not LINUX, reason="prctl(PR_SET_DUMPABLE) is Linux-only")
@pytest.mark.skipif(LINUX and os.geteuid() == 0, reason="root reads any /proc entry by design")
class TestARealNonDumpableProcess:
    def test_the_flag_is_set(self, nondumpable_child):
        _, out = nondumpable_child
        assert out["state"] == "set"
        assert out["runtime"]["dumpable"] is False

    def test_a_same_user_process_cannot_read_its_environ(self, nondumpable_child):
        """The whole point: the block the service manager filled is out of reach."""
        pid, _ = nondumpable_child
        with pytest.raises(PermissionError):
            Path(f"/proc/{pid}/environ").read_bytes()

    def test_nor_its_maps_which_is_why_deploy_sh_asks_the_daemon(self, nondumpable_child):
        pid, _ = nondumpable_child
        with pytest.raises(PermissionError):
            Path(f"/proc/{pid}/maps").read_text()

    def test_ps_and_the_lock_start_time_still_work(self, nondumpable_child):
        """cmdline and stat stay world-readable: ps, pgrep and the daemon lock's
        start-time check (gateway/status.py) are unaffected."""
        from prometheus.gateway.status import _process_start_time

        pid, _ = nondumpable_child
        assert "make_nondumpable" in Path(f"/proc/{pid}/cmdline").read_text()
        assert _process_start_time(pid) is not None

    def test_it_still_reads_its_own_maps(self, nondumpable_child):
        """runtime_report reads /proc/self/maps from inside; non-dumpable must not
        take that away, or deploy.sh B5 would have nothing to check."""
        _, out = nondumpable_child
        assert out["runtime"]["mapped_from_prefix"] in (True, False)
        assert out["runtime"]["user_site_mapped"] in (True, False)


def test_off_linux_it_says_unsupported_and_does_nothing(monkeypatch):
    monkeypatch.setattr(nondumpable.sys, "platform", "darwin")
    monkeypatch.setattr(nondumpable, "STATE", "not-attempted")
    assert nondumpable.make_nondumpable() == "unsupported"
    assert nondumpable.is_dumpable() is None


def test_a_refused_prctl_is_reported_not_raised(monkeypatch):
    """A daemon that cannot set the flag still boots, and run_daemon says so."""
    monkeypatch.setattr(nondumpable.sys, "platform", "linux")
    monkeypatch.setattr(nondumpable, "_prctl", lambda: (lambda *a: -1))
    monkeypatch.setattr(nondumpable, "STATE", "not-attempted")
    assert nondumpable.make_nondumpable().startswith("failed: prctl errno")
    assert nondumpable.STATE.startswith("failed")


def test_no_libc_prctl_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(nondumpable.sys, "platform", "linux")
    monkeypatch.setattr(nondumpable, "_prctl", lambda: None)
    monkeypatch.setattr(nondumpable, "STATE", "not-attempted")
    assert nondumpable.make_nondumpable() == "failed: libc prctl not found"


def test_main_sets_it_before_anything_else():
    """Before parse_args, before setup mode, before the env file: the secrets are
    already in the starting block when main() is entered."""
    from prometheus import daemon

    src = inspect.getsource(daemon.main)
    assert src.index("make_nondumpable()") < src.index("snapshot_startup_env()")
    assert "_nondumpable.STATE" in inspect.getsource(daemon.run_daemon), (
        "the outcome must reach the boot log")


MAPS = """\
7f0000000000-7f0000001000 r--p 00000000 08:02 1 /home/u/prometheus-venvs/abc123/lib/python3.12/site-packages/pydantic_core/_pydantic_core.so
7f0000002000-7f0000003000 r-xp 00000000 08:02 2 /usr/lib/x86_64-linux-gnu/libc.so.6
7f0000004000-7f0000005000 rw-p 00000000 00:00 0 [heap]
7f0000006000-7f0000007000 rw-p 00000000 00:00 0
7f0000008000-7f0000009000 r--p 00000000 08:02 3 /home/u/dir with space/x.so
"""


def test_maps_evidence_finds_the_venv_and_no_user_site():
    assert _maps_evidence(MAPS, "/home/u/prometheus-venvs/abc123") == (True, False)


def test_maps_evidence_is_a_prefix_match_not_a_substring():
    """abc12 must not match abc123's files: the trailing slash is load-bearing,
    exactly as it was in deploy.sh's grep for "$ROOT/$S/"."""
    assert _maps_evidence(MAPS, "/home/u/prometheus-venvs/abc12") == (False, False)


def test_maps_evidence_sees_a_user_site_mapping():
    text = MAPS + "7f000000a000-7f000000b000 r--p 0 08:02 4 /home/u/.local/lib/python3.12/site-packages/numpy/x.so\n"
    assert _maps_evidence(text, "/home/u/prometheus-venvs/abc123") == (True, True)


def test_runtime_report_shape_and_pid():
    rt = runtime_report()
    assert set(rt) == {"pid", "prefix", "mapped_from_prefix", "user_site_mapped",
                       "user_site_modules", "dumpable"}
    assert rt["pid"] == os.getpid()
    assert rt["prefix"] == os.path.realpath(sys.prefix)
    assert isinstance(rt["user_site_modules"], int)


def test_runtime_report_counts_a_user_site_module(monkeypatch):
    fake = type(sys)("fake_user_site_mod")
    fake.__file__ = "/home/u/.local/lib/python3.12/site-packages/fake_user_site_mod.py"
    before = runtime_report()["user_site_modules"]
    monkeypatch.setitem(sys.modules, "fake_user_site_mod", fake)
    assert runtime_report()["user_site_modules"] == before + 1


def test_runtime_report_without_proc_says_unknown_not_false(monkeypatch):
    """No /proc/self/maps (macOS) is None, never False: deploy.sh B5 must not read
    "nothing mapped from a user site" from a platform that cannot say."""
    real = Path.read_text

    def no_proc(self, *a, **kw):
        if str(self) == "/proc/self/maps":
            raise FileNotFoundError(self)
        return real(self, *a, **kw)

    monkeypatch.setattr(environment.Path, "read_text", no_proc)
    rt = runtime_report()
    assert rt["mapped_from_prefix"] is None and rt["user_site_mapped"] is None


def test_status_serves_it():
    src = Path(__file__).resolve().parent.parent.joinpath("src/prometheus/web/server.py").read_text()
    assert '"runtime": _runtime_report(),' in src
