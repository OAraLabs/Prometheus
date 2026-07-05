"""``prometheus install-service`` — never clobbers, idempotent (Phase 0, item 3).

All writes go to a tmp systemd dir; the injectable runner means systemctl
is NEVER actually invoked — this dev box runs a live prometheus.service.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.cli.service import (
    DEFAULT_EXEC_START,
    UNIT_NAME,
    UNIT_TEMPLATE,
    install_service,
    render_unit,
)


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


class TestRenderUnit:
    def test_required_directives(self):
        content = render_unit(DEFAULT_EXEC_START)
        assert "After=network.target" in content
        assert "Restart=on-failure" in content
        assert "ExecStart=/usr/bin/env prometheus daemon" in content
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
