"""Vault format marker — Foundation Spec 1.1.

The marker exists to convert the silent-misread case (a newer vault under an
older binary) into a loud refusal at boot. These tests pin the whole mismatch
table, the corrupt-marker handling under each ``vault.format_check`` mode,
and the two hard rules the module docstring states: adoption is explicit
(never silent), and a parsed marker with the wrong format refuses regardless
of mode.

Also pinned here: the daemon wiring ORDER. The marker check reads the vault
root, the vault root resolver reads PROMETHEUS_VAULT, and PROMETHEUS_VAULT
arrives via the env file — so env-file load must precede root resolution,
which must precede the marker check. That ordering was wrong once already
(env file loaded 15 lines after the roots were pinned, masked by systemd);
the source-order test keeps it from regressing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml

from prometheus.config.vault_marker import (
    MARKER_FILENAME,
    VAULT_FORMAT_CURRENT,
    VaultMarker,
    VaultMarkerCorrupt,
    VaultMarkerError,
    check_vault_marker,
    create_marker,
    marker_path,
    read_marker,
    write_marker,
)


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "brain-vault"
    root.mkdir()
    (root / "BRAIN.md").write_text("# router\n")
    return root


def _valid_marker(fmt: int = VAULT_FORMAT_CURRENT) -> VaultMarker:
    return VaultMarker(
        vault_format=fmt,
        created="2026-08-28T00:00:00Z",
        created_by="prometheus 0.1.0",
        instance_id="550e8400-e29b-41d4-a716-446655440000",
        enrolled_nodes=[],
    )


class TestMarkerIO:
    def test_roundtrip_preserves_fields(self, vault: Path) -> None:
        write_marker(vault, _valid_marker())
        marker = read_marker(vault)
        assert marker is not None
        assert marker.vault_format == VAULT_FORMAT_CURRENT
        assert marker.instance_id == "550e8400-e29b-41d4-a716-446655440000"
        assert marker.created == "2026-08-28T00:00:00Z"
        assert marker.enrolled_nodes == []

    def test_key_order_is_stable(self, vault: Path) -> None:
        # The marker lives in a git repo a human reads diffs of; key order
        # churn would make every rewrite a spurious diff.
        write_marker(vault, _valid_marker())
        text = marker_path(vault).read_text()
        positions = [text.index(k) for k in (
            "vault_format", "created:", "created_by", "instance_id",
            "enrolled_nodes",
        )]
        assert positions == sorted(positions)

    def test_absent_reads_as_none(self, vault: Path) -> None:
        assert read_marker(vault) is None

    def test_atomic_write_leaves_no_tmp(self, vault: Path) -> None:
        write_marker(vault, _valid_marker())
        leftovers = [p.name for p in vault.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    @pytest.mark.parametrize(
        "content",
        [
            ":{[ not yaml",
            "- a\n- list\n",
            "created: 2026-01-01\n",                      # no vault_format
            "vault_format: true\ninstance_id: x\n",       # bool is not a version
            "vault_format: '1'\ninstance_id: x\n",        # string is not a version
            f"vault_format: {VAULT_FORMAT_CURRENT}\n",    # no instance_id
        ],
    )
    def test_invalid_marker_is_corrupt(self, vault: Path, content: str) -> None:
        marker_path(vault).write_text(content)
        with pytest.raises(VaultMarkerCorrupt):
            read_marker(vault)


class TestAdoption:
    def test_create_mints_current_format_and_uuid(self, vault: Path) -> None:
        marker = create_marker(vault, created_by="prometheus test")
        assert marker.vault_format == VAULT_FORMAT_CURRENT
        assert re.fullmatch(r"[0-9a-f-]{36}", marker.instance_id)
        on_disk = read_marker(vault)
        assert on_disk is not None
        assert on_disk.instance_id == marker.instance_id

    def test_instance_id_stable_across_rereads(self, vault: Path) -> None:
        # Acceptance: instance_id present and stable across daemon restarts.
        # A restart is a re-read; nothing on the read path may rewrite it.
        minted = create_marker(vault, created_by="prometheus test")
        for _ in range(3):
            marker = read_marker(vault)
            assert marker is not None
            assert marker.instance_id == minted.instance_id

    def test_refuses_second_adoption(self, vault: Path) -> None:
        create_marker(vault, created_by="prometheus test")
        with pytest.raises(VaultMarkerError, match="already adopted"):
            create_marker(vault, created_by="prometheus test")

    def test_refuses_missing_vault_dir(self, tmp_path: Path) -> None:
        with pytest.raises(VaultMarkerError, match="nothing to adopt"):
            create_marker(tmp_path / "nope", created_by="prometheus test")


class TestStartupGate:
    @pytest.mark.parametrize("mode", ["off", "warn", "refuse"])
    def test_absent_vault_is_never_an_error(self, tmp_path: Path, mode: str) -> None:
        assert check_vault_marker(tmp_path / "nope", mode=mode) is None

    def test_no_marker_off_is_silent(
        self, vault: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert check_vault_marker(vault, mode="off") is None
        assert MARKER_FILENAME not in caplog.text

    def test_no_marker_warn_names_the_adopt_command(
        self, vault: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert check_vault_marker(vault, mode="warn") is None
        assert "prometheus vault adopt" in caplog.text

    def test_no_marker_refuse_refuses(self, vault: Path) -> None:
        with pytest.raises(VaultMarkerError, match="prometheus vault adopt"):
            check_vault_marker(vault, mode="refuse")

    @pytest.mark.parametrize("mode", ["off", "warn", "refuse"])
    def test_matching_format_proceeds(self, vault: Path, mode: str) -> None:
        write_marker(vault, _valid_marker())
        marker = check_vault_marker(vault, mode=mode)
        assert marker is not None
        assert marker.vault_format == VAULT_FORMAT_CURRENT

    @pytest.mark.parametrize("mode", ["off", "warn", "refuse"])
    def test_newer_format_refuses_regardless_of_mode(
        self, vault: Path, mode: str
    ) -> None:
        # THE row that matters (spec 1.1): newer vault, older binary. Not
        # configurable — 'off' silences the legacy check, not this one.
        write_marker(vault, _valid_marker(fmt=VAULT_FORMAT_CURRENT + 1))
        with pytest.raises(VaultMarkerError) as exc:
            check_vault_marker(vault, mode=mode)
        # Loud error naming both versions.
        assert str(VAULT_FORMAT_CURRENT + 1) in str(exc.value)
        assert str(VAULT_FORMAT_CURRENT) in str(exc.value)

    def test_lower_format_with_no_migration_refuses_naming_both(
        self, vault: Path
    ) -> None:
        write_marker(vault, _valid_marker(fmt=0))
        with pytest.raises(VaultMarkerError) as exc:
            check_vault_marker(vault, mode="warn")
        assert "0" in str(exc.value)
        assert str(VAULT_FORMAT_CURRENT) in str(exc.value)

    def test_corrupt_marker_refuse_mode_refuses(self, vault: Path) -> None:
        marker_path(vault).write_text(":{[ not yaml")
        with pytest.raises(VaultMarkerCorrupt):
            check_vault_marker(vault, mode="refuse")

    def test_corrupt_marker_warn_mode_warns_and_continues(
        self, vault: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        marker_path(vault).write_text(":{[ not yaml")
        with caplog.at_level(logging.WARNING):
            assert check_vault_marker(vault, mode="warn") is None
        assert "CORRUPT" in caplog.text

    def test_corrupt_marker_off_mode_is_silent(
        self, vault: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        marker_path(vault).write_text(":{[ not yaml")
        with caplog.at_level(logging.WARNING):
            assert check_vault_marker(vault, mode="off") is None
        assert "CORRUPT" not in caplog.text


class TestDaemonWiring:
    """Source-order assertions on run_daemon — the same style as the vault
    tool's AST guard: cheap, build-failing, and immune to 'works under
    systemd' masking."""

    @pytest.fixture()
    def daemon_source(self) -> str:
        import prometheus.daemon as daemon_mod
        return Path(daemon_mod.__file__).read_text(encoding="utf-8")

    def test_env_file_loads_before_roots_resolve(self, daemon_source: str) -> None:
        # PROMETHEUS_VAULT comes from the env file; resolving the roots
        # first pins the defaults on a bare invocation while systemd pins
        # the configured ones — two boots disagreeing about where the
        # vault is.
        env_load = daemon_source.index("load_env_file()")
        wiki = daemon_source.index("resolve_wiki_root(config)")
        vault = daemon_source.index("resolve_vault_root(config)")
        assert env_load < wiki
        assert env_load < vault

    def test_marker_check_runs_after_root_is_pinned(self, daemon_source: str) -> None:
        pinned = daemon_source.index("set_vault_root(vault_root)")
        checked = daemon_source.index("check_vault_marker(vault_root")
        assert pinned < checked


class TestVaultCli:
    """Drive the real `prometheus vault` dispatch through main()."""

    def _run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
             vault_root: Path, action: str) -> tuple[int, str]:
        import prometheus.__main__ as entry
        cfg = tmp_path / "cli-config.yaml"
        cfg.write_text(yaml.safe_dump({"vault": {"root": str(vault_root)}}))
        monkeypatch.setattr(
            "sys.argv", ["prometheus", "--config", str(cfg), "vault", action]
        )
        with pytest.raises(SystemExit) as exc:
            entry.main()
        return int(exc.value.code or 0), ""

    def test_adopt_then_status_ok(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vault: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        code, _ = self._run(monkeypatch, tmp_path, vault, "adopt")
        assert code == 0
        out = capsys.readouterr().out
        assert "Adopted vault" in out
        assert "instance_id" in out

        code, _ = self._run(monkeypatch, tmp_path, vault, "status")
        assert code == 0
        out = capsys.readouterr().out
        assert f"vault_format {VAULT_FORMAT_CURRENT} — OK" in out

    def test_second_adopt_refuses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vault: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        code, _ = self._run(monkeypatch, tmp_path, vault, "adopt")
        assert code == 0
        capsys.readouterr()
        code, _ = self._run(monkeypatch, tmp_path, vault, "adopt")
        assert code == 1
        assert "Refused" in capsys.readouterr().out

    def test_status_on_unadopted_vault_points_at_adopt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vault: Path,
        capsys: pytest.CaptureFixture,
    ) -> None:
        code, _ = self._run(monkeypatch, tmp_path, vault, "status")
        assert code == 1
        assert "prometheus vault adopt" in capsys.readouterr().out
