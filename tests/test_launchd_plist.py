"""The LaunchAgent plist and the thin launchctl wrappers (``cli/launchd.py``).

Everything here is pure: bytes in, bytes out, and an injected ``runner`` that
records argv. launchctl is NEVER invoked and the real ``~/Library`` is never
touched — this dev box is a Mac with a live daemon.

THE LABEL IS PINNED AS A LITERAL on purpose. Another component registers a
plist with the SAME label, and launchd itself refuses two supervisors for one
label; if the constant drifts, the two stop colliding and a second daemon
starts instead of the second registration failing. Changing it must fail a
test, not be a quiet edit.
"""

from __future__ import annotations

import os
import plistlib
import re

import pytest

from prometheus.cli import launchd
from prometheus.cli.launchd import (
    LABEL,
    bootstrap,
    enable,
    gui_domain,
    is_loaded,
    launch_agents_dir,
    plist_path,
    render_plist,
)
from prometheus.cli.service import UNIT_TEMPLATE

CLI_ARGS = ["/opt/venv/bin/oara", "daemon"]

# The keys BOTH variants always carry. Restated here rather than imported so a
# change to the module's constants has to be made twice, on purpose.
SHARED = {
    "Label": "com.oaralabs.prometheus.daemon",
    "RunAtLoad": True,
    "KeepAlive": {"SuccessfulExit": False},
    "ThrottleInterval": 10,
}


class _Spy:
    """Records argv and answers with a fixed exit code. Runs nothing."""

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


def _parse(blob: bytes) -> dict:
    return plistlib.loads(blob)


class TestLabel:
    def test_label_is_final(self):
        assert LABEL == "com.oaralabs.prometheus.daemon"


class TestRenderPlist:
    def test_cli_variant_parses_with_exact_values(self):
        parsed = _parse(render_plist(CLI_ARGS))
        assert parsed["Label"] == "com.oaralabs.prometheus.daemon"
        assert parsed["ProgramArguments"] == CLI_ARGS
        assert parsed["RunAtLoad"] is True
        assert parsed["KeepAlive"] == {"SuccessfulExit": False}
        assert parsed["KeepAlive"]["SuccessfulExit"] is False
        assert parsed["ThrottleInterval"] == 10
        assert type(parsed["ThrottleInterval"]) is int
        # The CLI variant names no bundle: it is launched by absolute path.
        assert "BundleProgram" not in parsed

    def test_is_xml_not_binary(self):
        blob = render_plist(CLI_ARGS)
        assert isinstance(blob, bytes)
        assert blob.startswith(b"<?xml")
        assert b"<!DOCTYPE plist" in blob

    def test_render_is_deterministic(self):
        assert render_plist(CLI_ARGS) == render_plist(CLI_ARGS)

    def test_bundle_variant_carries_bundle_program_and_arguments(self):
        parsed = _parse(render_plist(
            ["daemon-launcher", "daemon"],
            bundle_program="Contents/MacOS/daemon-launcher",
        ))
        assert parsed["BundleProgram"] == "Contents/MacOS/daemon-launcher"
        assert parsed["ProgramArguments"] == ["daemon-launcher", "daemon"]

    def test_shared_keys_are_identical_across_variants(self):
        """Both variants restate the systemd unit's Restart=on-failure /
        RestartSec=10. One supervisor must never restart faster, or forever,
        than the other: pinned equal to each other AND to the literal, so
        drifting both together fails as well."""
        cli = _parse(render_plist(CLI_ARGS))
        app = _parse(render_plist(
            ["daemon-launcher", "daemon"],
            bundle_program="Contents/MacOS/daemon-launcher",
            associated_bundle_ids=["com.example.app"],
            environment={"PATH": "/usr/bin"},
            working_directory="/tmp",
            stdout_path="/dev/null",
            stderr_path="/tmp/err.log",
        ))
        for key in SHARED:
            assert cli[key] == app[key], key
            assert cli[key] == SHARED[key], key

    def test_throttle_matches_the_systemd_restart_policy(self):
        restart_sec = re.search(r"^RestartSec=(\d+)$", UNIT_TEMPLATE, re.M)
        assert restart_sec is not None
        assert "Restart=on-failure" in UNIT_TEMPLATE
        assert _parse(render_plist(CLI_ARGS))["ThrottleInterval"] == int(
            restart_sec.group(1)
        )

    def test_optional_keys_absent_unless_given(self):
        parsed = _parse(render_plist(CLI_ARGS))
        for key in (
            "AssociatedBundleIdentifiers", "EnvironmentVariables",
            "WorkingDirectory", "StandardOutPath", "StandardErrorPath",
        ):
            assert key not in parsed

    def test_optional_keys_use_launchds_names(self):
        parsed = _parse(render_plist(
            CLI_ARGS,
            associated_bundle_ids=["com.example.app"],
            environment={"PATH": "/usr/bin:/bin"},
            working_directory="/Users/someone",
            stdout_path="/dev/null",
            stderr_path="/Users/someone/Library/Logs/x.log",
        ))
        assert parsed["AssociatedBundleIdentifiers"] == ["com.example.app"]
        assert parsed["EnvironmentVariables"] == {"PATH": "/usr/bin:/bin"}
        assert parsed["WorkingDirectory"] == "/Users/someone"
        assert parsed["StandardOutPath"] == "/dev/null"
        assert parsed["StandardErrorPath"] == "/Users/someone/Library/Logs/x.log"

    def test_no_program_arguments_is_an_error(self):
        with pytest.raises(ValueError):
            render_plist([])


class TestPaths:
    def test_default_agents_dir_is_under_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("PROMETHEUS_LAUNCH_AGENTS_DIR", raising=False)
        assert launch_agents_dir() == tmp_path / "Library" / "LaunchAgents"

    def test_env_overrides_agents_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_LAUNCH_AGENTS_DIR", str(tmp_path / "agents"))
        assert launch_agents_dir() == tmp_path / "agents"

    def test_env_override_expands_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("PROMETHEUS_LAUNCH_AGENTS_DIR", "~/agents")
        assert launch_agents_dir() == tmp_path / "agents"

    def test_plist_path_is_named_for_the_label(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_LAUNCH_AGENTS_DIR", str(tmp_path))
        assert plist_path() == tmp_path / "com.oaralabs.prometheus.daemon.plist"
        other = tmp_path / "elsewhere"
        assert plist_path(other) == other / "com.oaralabs.prometheus.daemon.plist"

    def test_gui_domain(self):
        assert gui_domain(501) == "gui/501"
        assert gui_domain() == f"gui/{os.getuid()}"

    def test_log_dir_is_under_home_library_logs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert launchd.log_dir() == tmp_path / "Library" / "Logs" / "Prometheus"

    def test_daemon_path_puts_the_binary_dir_first(self):
        assert launchd.daemon_path("/opt/venv/bin/oara") == (
            "/opt/venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:"
            "/usr/sbin:/sbin"
        )


class TestLaunchctlWrappers:
    def test_is_loaded_probes_the_exact_service_target(self):
        spy = _Spy(returncode=0)
        assert is_loaded(spy) is True
        assert spy.calls == [
            ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"]
        ]

    def test_is_loaded_false_when_launchd_cannot_find_the_service(self):
        # 113 is launchctl's "Could not find service".
        assert is_loaded(_Spy(returncode=113)) is False

    def test_enable_argv(self):
        spy = _Spy()
        assert enable(spy).returncode == 0
        assert spy.calls == [
            ["launchctl", "enable", f"gui/{os.getuid()}/{LABEL}"]
        ]

    def test_bootstrap_argv(self, tmp_path):
        spy = _Spy(returncode=5)
        plist = tmp_path / f"{LABEL}.plist"
        assert bootstrap(spy, plist).returncode == 5
        assert spy.calls == [
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)]
        ]

