"""``oara install-service`` — never clobbers, idempotent (Phase 0, item 3).

All writes go to a tmp systemd dir; the injectable runner means systemctl
is NEVER actually invoked — this dev box runs a live prometheus.service.

The macOS half (a LaunchAgent plist instead of a systemd unit) is tested the
same way: the host platform is faked, ``launchctl`` is a recording stand-in,
and HOME / the agents dir / the systemd dir all point into ``tmp_path``. A
test that reached the real ``~/Library/LaunchAgents`` would register a
supervisor on the machine running the suite; ``_host`` below makes that
impossible by default.
"""

from __future__ import annotations

import argparse
import os
import plistlib
from pathlib import Path

import pytest

from prometheus.cli import service
from prometheus.cli.service import (
    DEFAULT_EXEC_START,
    UNIT_NAME,
    UNIT_TEMPLATE,
    add_install_service_subparser,
    install_service,
    render_unit,
    resolve_exec_start,
    run_install_service_command,
)

# The label is restated as a literal on purpose (not imported): another
# component registers a plist with the SAME label, so a drift here must fail
# a test rather than be a quiet edit. It also keeps this module importable
# before cli/launchd.py exists.
LABEL = "com.oaralabs.prometheus.daemon"
FAKE_OARA = "/opt/venv/bin/oara"


class _RunnerSpy:
    def __init__(self, returncode: int = 0):
        self.calls: list[list[str]] = []
        self.returncode = returncode

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(list(cmd))

        class _R:
            returncode = self.returncode
            stdout = ""
            stderr = ""
        return _R()


@pytest.fixture
def systemd_dir(tmp_path) -> Path:
    return tmp_path / "systemd-user"


class _Host:
    """Where every default location resolves to during a test."""

    def __init__(self, tmp_path: Path):
        self.home = tmp_path / "home"
        self.systemd_dir = tmp_path / "guard-systemd"
        self.agents_dir = tmp_path / "LaunchAgents"
        self.log_dir = self.home / "Library" / "Logs" / "Prometheus"

    @property
    def plist(self) -> Path:
        return self.agents_dir / f"{LABEL}.plist"

    @property
    def backup(self) -> Path:
        return self.agents_dir / f"{LABEL}.plist.bak"

    def written(self) -> list[Path]:
        """Every file written anywhere install-service could write."""
        found: list[Path] = []
        for root in (self.agents_dir, self.systemd_dir, self.log_dir):
            if root.exists():
                found.extend(p for p in root.rglob("*") if p.is_file())
        return sorted(found)


@pytest.fixture(autouse=True)
def _host(tmp_path, monkeypatch) -> _Host:
    """Contain every test, and pin the host to Linux unless it asks otherwise.

    The pre-existing tests describe the systemd path and call
    ``install_service`` without a platform; on a Mac they would otherwise take
    the LaunchAgent path (and ``test_env_dir_override`` would write the REAL
    ``~/Library/LaunchAgents``). HOME is redirected so every default location
    — agents dir, log dir, systemd dir — lands in ``tmp_path`` even if a test
    forgets to name one. ``raising=False``: the host-platform hook is what the
    implementation reads, and is absent until it exists.
    """
    host = _Host(tmp_path)
    host.home.mkdir()
    monkeypatch.setenv("HOME", str(host.home))
    monkeypatch.setenv("PROMETHEUS_SYSTEMD_USER_DIR", str(host.systemd_dir))
    monkeypatch.setenv("PROMETHEUS_LAUNCH_AGENTS_DIR", str(host.agents_dir))
    monkeypatch.setattr(service, "_host_platform", lambda: "linux", raising=False)
    return host


@pytest.fixture
def mac(_host, monkeypatch) -> _Host:
    """A macOS host with a resolvable ``oara`` and no systemctl anywhere."""
    monkeypatch.setattr(service, "_host_platform", lambda: "darwin", raising=False)
    monkeypatch.setattr(
        "prometheus.cli.service.shutil.which", lambda name: f"/opt/venv/bin/{name}"
    )
    return _host


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _LaunchctlSpy:
    """Stands in for ``subprocess.run`` and answers the way launchctl does.

    ``print`` exits 0 when ``loaded`` and 113 ("Could not find service")
    otherwise. ``missing`` raises FileNotFoundError on every call, like a host
    with no launchctl; ``missing_after_print`` only after the read-only probe,
    like one that vanishes mid-run. Runs nothing.
    """

    def __init__(self, *, loaded=False, enable_rc=0, bootstrap_rc=0,
                 bootstrap_err="", missing=False, missing_after_print=False):
        self.calls: list[list[str]] = []
        self.loaded = loaded
        self.enable_rc = enable_rc
        self.bootstrap_rc = bootstrap_rc
        self.bootstrap_err = bootstrap_err
        self.missing = missing
        self.missing_after_print = missing_after_print

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(list(cmd))
        sub = cmd[1] if len(cmd) > 1 else ""
        if self.missing or (self.missing_after_print and sub != "print"):
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        if sub == "print":
            return _Result(0 if self.loaded else 113)
        if sub == "enable":
            return _Result(self.enable_rc)
        if sub == "bootstrap":
            return _Result(self.bootstrap_rc, stderr=self.bootstrap_err)
        return _Result(0)

    def changes(self) -> list[list[str]]:
        """Calls that are not the read-only ``print`` probe."""
        return [c for c in self.calls if c[1:2] != ["print"]]


def _gui() -> str:
    return f"gui/{os.getuid()}"


PRINT_ARGV = ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"]
ENABLE_ARGV = ["launchctl", "enable", f"gui/{os.getuid()}/{LABEL}"]


class TestRenderUnit:
    def test_required_directives(self):
        content = render_unit(DEFAULT_EXEC_START)
        assert "After=network.target" in content
        assert "Restart=on-failure" in content
        assert "ExecStart=/usr/bin/env oara daemon" in content
        assert "EnvironmentFile=-%h/.config/prometheus/env" in content
        assert "WantedBy=default.target" in content

    def test_resolves_installed_binary(self, monkeypatch):
        monkeypatch.setattr(
            "prometheus.cli.service.shutil.which",
            lambda name: "/opt/venv/bin/prometheus",
        )
        assert "ExecStart=/opt/venv/bin/prometheus daemon" in render_unit()

    def test_packaging_file_matches_template(self):
        """packaging/prometheus.service must not drift from the template."""
        packaged = (
            Path(__file__).resolve().parents[1] / "packaging" / UNIT_NAME
        ).read_text(encoding="utf-8")
        rendered = UNIT_TEMPLATE.format(exec_start=DEFAULT_EXEC_START)
        # The packaged copy has a leading comment header; the directive
        # body must be identical.
        body = "\n".join(
            line for line in packaged.splitlines() if not line.startswith("#")
        ).strip()
        assert body == "\n".join(
            line for line in rendered.splitlines() if not line.startswith("#")
        ).strip()


class TestInstallService:
    def test_installs_and_enables(self, systemd_dir):
        runner = _RunnerSpy()
        rc = install_service(systemd_dir=systemd_dir, runner=runner)
        assert rc == 0
        unit = systemd_dir / UNIT_NAME
        assert unit.is_file()
        assert "Restart=on-failure" in unit.read_text()
        assert ["systemctl", "--user", "daemon-reload"] in runner.calls
        assert ["systemctl", "--user", "enable", UNIT_NAME] in runner.calls
        # Never started unless --now.
        assert ["systemctl", "--user", "start", UNIT_NAME] not in runner.calls

    def test_refuses_existing_unit_without_force(self, systemd_dir):
        systemd_dir.mkdir(parents=True)
        unit = systemd_dir / UNIT_NAME
        unit.write_text("[Unit]\nDescription=hand-rolled\n", encoding="utf-8")
        runner = _RunnerSpy()
        rc = install_service(systemd_dir=systemd_dir, runner=runner)
        assert rc == 1
        # Untouched, and NO systemctl calls were made.
        assert unit.read_text() == "[Unit]\nDescription=hand-rolled\n"
        assert runner.calls == []

    def test_force_overwrites_with_backup(self, systemd_dir):
        systemd_dir.mkdir(parents=True)
        unit = systemd_dir / UNIT_NAME
        unit.write_text("old contents\n", encoding="utf-8")
        runner = _RunnerSpy()
        rc = install_service(systemd_dir=systemd_dir, force=True, runner=runner)
        assert rc == 0
        assert "Restart=on-failure" in unit.read_text()
        backup = systemd_dir / "prometheus.service.bak"
        assert backup.read_text() == "old contents\n"

    def test_idempotent_when_identical(self, systemd_dir):
        runner = _RunnerSpy()
        assert install_service(systemd_dir=systemd_dir, runner=runner) == 0
        first = (systemd_dir / UNIT_NAME).read_text()
        runner2 = _RunnerSpy()
        assert install_service(systemd_dir=systemd_dir, runner=runner2) == 0
        assert (systemd_dir / UNIT_NAME).read_text() == first
        # Up-to-date short-circuit: no systemctl churn on the rerun.
        assert runner2.calls == []

    def test_now_starts_service(self, systemd_dir):
        runner = _RunnerSpy()
        rc = install_service(systemd_dir=systemd_dir, now=True, runner=runner)
        assert rc == 0
        assert ["systemctl", "--user", "start", UNIT_NAME] in runner.calls

    def test_systemctl_failure_is_nonzero(self, systemd_dir):
        runner = _RunnerSpy(returncode=1)
        rc = install_service(systemd_dir=systemd_dir, runner=runner)
        assert rc == 1

    def test_env_dir_override(self, tmp_path, monkeypatch):
        target = tmp_path / "override"
        monkeypatch.setenv("PROMETHEUS_SYSTEMD_USER_DIR", str(target))
        runner = _RunnerSpy()
        assert install_service(runner=runner) == 0
        assert (target / UNIT_NAME).is_file()


class TestSystemctlAbsent:
    """The same lie, on any host without systemd: the unit is written, nothing
    is enabled, and the command used to say so with exit 0."""

    def test_missing_systemctl_is_not_success(self, systemd_dir, capsys):
        def no_systemctl(cmd, capture_output=True, text=True):
            raise FileNotFoundError(2, "No such file or directory", cmd[0])

        rc = install_service(systemd_dir=systemd_dir, runner=no_systemctl)

        assert rc == 1
        out = capsys.readouterr().out
        # The unit that WAS written stays, and the message says where it is
        # and that nothing is enabled — it must not read like success.
        assert (systemd_dir / UNIT_NAME).is_file()
        assert str(systemd_dir / UNIT_NAME) in out
        assert "systemctl" in out

    def test_resolve_exec_start_falls_back_to_a_path_lookup(self, monkeypatch):
        monkeypatch.setattr("prometheus.cli.service.shutil.which", lambda name: None)
        assert resolve_exec_start() == DEFAULT_EXEC_START


class TestInstallServiceMacOS:
    """On macOS the command writes a LaunchAgent, never a systemd unit."""

    def test_launchctl_absent_is_not_success(self, mac, capsys):
        for now in (False, True):
            spy = _LaunchctlSpy(missing=True)
            rc = install_service(now=now, runner=spy)
            assert rc == 1, f"now={now}"
            assert "launchctl" in capsys.readouterr().out
            # Whatever else happened, no systemd unit appears on a Mac.
            assert not (mac.systemd_dir / UNIT_NAME).exists()

    def test_writes_a_launchagent_and_no_systemd_unit(self, mac):
        spy = _LaunchctlSpy()
        rc = install_service(runner=spy)

        assert rc == 0
        assert mac.plist.is_file()
        assert not (mac.systemd_dir / UNIT_NAME).exists()
        assert not any("systemctl" in c for c in spy.calls)

        parsed = plistlib.loads(mac.plist.read_bytes())
        assert parsed["Label"] == LABEL
        assert parsed["ProgramArguments"] == [FAKE_OARA, "daemon"]
        assert parsed["RunAtLoad"] is True
        assert parsed["KeepAlive"] == {"SuccessfulExit": False}
        assert parsed["ThrottleInterval"] == 10
        # launchd's default PATH is minimal; the binary's own directory leads.
        assert parsed["EnvironmentVariables"]["PATH"] == (
            "/opt/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:"
            "/usr/sbin:/sbin"
        )
        # The daemon rotates its own log under ~/.prometheus/logs; launchd
        # keeps stderr only, so a failure that precedes logging is visible.
        assert parsed["StandardOutPath"] == "/dev/null"
        assert parsed["StandardErrorPath"] == str(mac.log_dir / "launchd.err.log")
        assert mac.log_dir.is_dir()

    def test_default_location_is_the_users_launchagents_dir(
        self, mac, monkeypatch
    ):
        monkeypatch.delenv("PROMETHEUS_LAUNCH_AGENTS_DIR")
        rc = install_service(runner=_LaunchctlSpy())
        assert rc == 0
        # HOME is tmp_path/home here, so this is "~/Library/LaunchAgents".
        assert (mac.home / "Library" / "LaunchAgents" / f"{LABEL}.plist").is_file()

    def test_without_now_only_writes_the_plist(self, mac, capsys):
        spy = _LaunchctlSpy()
        assert install_service(runner=spy) == 0

        # Nothing is enabled, bootstrapped or started: the only launchctl
        # call is the read-only probe for another supervisor.
        assert spy.calls == [PRINT_ARGV]
        out = capsys.readouterr().out
        assert str(mac.plist) in out
        assert "next login" in out

    def test_now_enables_then_bootstraps(self, mac):
        spy = _LaunchctlSpy()
        assert install_service(now=True, runner=spy) == 0

        assert spy.calls == [
            PRINT_ARGV,
            ENABLE_ARGV,
            ["launchctl", "bootstrap", _gui(), str(mac.plist)],
        ]

    def test_bootstrap_failure_is_exit_1_with_launchctls_message(
        self, mac, capsys
    ):
        spy = _LaunchctlSpy(
            bootstrap_rc=5, bootstrap_err="Bootstrap failed: 5: Input/output error"
        )
        rc = install_service(now=True, runner=spy)

        assert rc == 1
        assert "Bootstrap failed: 5: Input/output error" in capsys.readouterr().out

    def test_enable_failure_is_exit_1_and_nothing_is_bootstrapped(self, mac):
        spy = _LaunchctlSpy(enable_rc=1)
        assert install_service(now=True, runner=spy) == 1
        assert not any(c[1] == "bootstrap" for c in spy.calls)

    def test_launchctl_vanishing_after_the_write_leaves_the_plist(
        self, mac, capsys
    ):
        spy = _LaunchctlSpy(missing_after_print=True)
        rc = install_service(now=True, runner=spy)

        assert rc == 1
        assert mac.plist.is_file()
        out = capsys.readouterr().out
        assert str(mac.plist) in out
        assert "in place" in out

    def test_refuses_a_differing_plist_without_force(self, mac):
        mac.agents_dir.mkdir()
        mac.plist.write_bytes(b"hand-rolled\n")
        spy = _LaunchctlSpy()

        rc = install_service(runner=spy)

        assert rc == 1
        assert mac.plist.read_bytes() == b"hand-rolled\n"
        assert not mac.backup.exists()
        assert spy.calls == []

    def test_force_backs_up_then_replaces(self, mac):
        mac.agents_dir.mkdir()
        mac.plist.write_bytes(b"hand-rolled\n")
        spy = _LaunchctlSpy()

        rc = install_service(force=True, runner=spy)

        assert rc == 0
        assert mac.backup.read_bytes() == b"hand-rolled\n"
        assert plistlib.loads(mac.plist.read_bytes())["Label"] == LABEL
        # A plist of ours was already there: nothing to start or enable.
        assert spy.changes() == []

    def test_identical_plist_is_already_up_to_date(self, mac, capsys):
        assert install_service(runner=_LaunchctlSpy()) == 0
        first = mac.plist.read_bytes()
        capsys.readouterr()

        spy = _LaunchctlSpy()
        rc = install_service(runner=spy)

        assert rc == 0
        assert mac.plist.read_bytes() == first
        assert spy.calls == []
        assert "Already installed and up to date" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "force,now", [(False, False), (True, False), (True, True)]
    )
    def test_label_loaded_without_our_plist_is_never_clobbered(
        self, mac, capsys, force, now
    ):
        """The signature of another supervisor (an app's registration, or a
        hand-rolled plist): the label is loaded but no plist of ours exists.
        --force replaces OUR plist only — two supervisors for one daemon is a
        bug, and launchd would refuse the second anyway."""
        spy = _LaunchctlSpy(loaded=True)

        rc = install_service(force=force, now=now, runner=spy)

        assert rc == 1
        assert mac.written() == []
        assert not mac.log_dir.exists()
        assert spy.changes() == []
        out = capsys.readouterr().out
        assert LABEL in out
        assert "supervisor" in out

    def test_systemd_dir_is_not_applicable_on_macos(self, mac, tmp_path, capsys):
        spy = _LaunchctlSpy()
        target = tmp_path / "elsewhere"

        rc = install_service(systemd_dir=target, runner=spy)

        assert rc == 2
        assert "not applicable on macOS" in capsys.readouterr().out
        assert not target.exists()
        assert mac.written() == []
        assert spy.calls == []


class TestPlatformAndDirOptions:
    """``platform=`` and ``launch_agents_dir=`` are the explicit forms of what
    the host and the env var decide by default."""

    def test_platform_argument_overrides_the_host(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "prometheus.cli.service.shutil.which", lambda name: FAKE_OARA
        )
        agents = tmp_path / "explicit-agents"
        # The autouse fixture pinned the HOST to linux; the argument wins.
        rc = install_service(
            platform="darwin", launch_agents_dir=agents, runner=_LaunchctlSpy()
        )
        assert rc == 0
        assert (agents / f"{LABEL}.plist").is_file()

    def test_linux_platform_argument_uses_the_systemd_path(
        self, mac, systemd_dir
    ):
        spy = _LaunchctlSpy()
        rc = install_service(
            platform="linux", systemd_dir=systemd_dir, runner=spy
        )
        assert rc == 0
        assert (systemd_dir / UNIT_NAME).is_file()
        # Nothing went to the macOS locations the host would have used.
        assert mac.written() == []
        assert ["systemctl", "--user", "daemon-reload"] in spy.calls


class TestCliWiring:
    """Through the real argparse subparser, as ``oara install-service`` runs."""

    @staticmethod
    def _parse(*argv: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser(prog="oara")
        add_install_service_subparser(parser.add_subparsers(dest="command"))
        return parser.parse_args(["install-service", *argv])

    def test_launch_agents_dir_flag_selects_the_directory(self, mac, tmp_path):
        target = tmp_path / "flag-agents"
        args = self._parse("--launch-agents-dir", str(target))

        rc = run_install_service_command(args, runner=_LaunchctlSpy())

        assert rc == 0
        assert (target / f"{LABEL}.plist").is_file()
        assert not mac.plist.exists()

    def test_now_flag_reaches_launchctl(self, mac):
        spy = _LaunchctlSpy()
        rc = run_install_service_command(self._parse("--now"), runner=spy)
        assert rc == 0
        assert ENABLE_ARGV in spy.calls

    def test_systemd_dir_flag_on_macos_is_exit_2(self, mac, tmp_path):
        args = self._parse("--systemd-dir", str(tmp_path / "x"))
        spy = _LaunchctlSpy()
        assert run_install_service_command(args, runner=spy) == 2
        assert spy.calls == []

    def test_help_names_both_platforms(self, monkeypatch):
        monkeypatch.setenv("COLUMNS", "300")
        parser = argparse.ArgumentParser(prog="oara")
        add_install_service_subparser(parser.add_subparsers(dest="command"))
        assert "systemd user unit (Linux) or LaunchAgent (macOS)" in (
            parser.format_help()
        )
