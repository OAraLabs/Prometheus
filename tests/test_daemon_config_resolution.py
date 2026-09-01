"""The daemon's own config read: not CWD-relative, not silent, and not alone.

THE DEFECT
----------
``daemon.load_config`` resolved its repo-local candidate as
``Path("config/prometheus.yaml")`` — relative to the PROCESS WORKING
DIRECTORY, not the installed source. The entire system's configuration
therefore depended on where the daemon happened to be launched from.
Measured on one install, one ``cd`` apart::

    cwd=<checkout>    daemon -> loaded      __main__ -> loaded
    cwd=<parent>      daemon -> NOTHING     __main__ -> loaded

Not live — the systemd unit pins ``WorkingDirectory`` AND passes
``--config`` — but "not live" was a property of a unit file, not of the code.

THE SECOND HALF, which is why this file exists rather than a one-line fix.
``web.setup_server.find_config_file`` decides whether the daemon boots for
real or enters setup mode, and it carried its OWN copy of the same
CWD-relative branch. Fixing only ``load_config`` would have made the gate and
the read disagree: with the CWD moved, the gate says "no config, enter setup
mode" about a checkout whose config the read would have found. Both now go
through ``config.defaults.config_search_paths``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.config import defaults  # noqa: E402
from prometheus.config.load import ConfigReadError  # noqa: E402
from prometheus.daemon import load_config  # noqa: E402
from prometheus.web.setup_server import find_config_file  # noqa: E402


@pytest.fixture()
def source_config(tmp_path, monkeypatch):
    """A repo-local config at the INSTALLED SOURCE's location (not the CWD)."""
    cfg = tmp_path / "checkout" / "config" / "prometheus.yaml"
    cfg.parent.mkdir(parents=True)

    def _write(data: dict) -> Path:
        cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", cfg)
        return cfg

    return _write


class TestNotCwdRelative:
    """The headline. Resolution follows the source, never the shell."""

    def test_a_config_under_the_cwd_does_not_hijack_the_read(
        self, tmp_path, monkeypatch, source_config
    ):
        """THE regression test.

        Two files both named ``config/prometheus.yaml``: one at the installed
        source's location, one under the process CWD. The source one must win.
        Against the old body the CWD one wins, because it was literally
        ``Path("config/prometheus.yaml")``.
        """
        source_config({"model": {"model": "from-the-source-tree"}})

        elsewhere = tmp_path / "somewhere-else"
        (elsewhere / "config").mkdir(parents=True)
        (elsewhere / "config" / "prometheus.yaml").write_text(
            "model:\n  model: from-the-cwd\n", encoding="utf-8"
        )
        monkeypatch.chdir(elsewhere)

        assert load_config()["model"]["model"] == "from-the-source-tree"

    def test_the_same_install_resolves_the_same_file_from_any_cwd(
        self, tmp_path, monkeypatch, source_config
    ):
        source_config({"model": {"model": "stable"}})
        seen = []
        for where in ("a", "b", "c"):
            d = tmp_path / where
            d.mkdir()
            monkeypatch.chdir(d)
            seen.append(load_config()["model"]["model"])
        assert seen == ["stable", "stable", "stable"]

    def test_daemon_and_cli_agree_on_the_same_install(
        self, tmp_path, monkeypatch, source_config
    ):
        """``daemon.load_config`` and ``__main__.load_config`` are two readers
        of one file. They disagreed for any CWD outside the checkout."""
        from prometheus.__main__ import load_config as cli_load_config

        source_config({"model": {"model": "one-answer"}})
        monkeypatch.chdir(tmp_path)

        assert (
            load_config()["model"]["model"]
            == cli_load_config()["model"]["model"]
            == "one-answer"
        )


class TestGateAndReadAgree:
    """``find_config_file`` gates the boot; ``load_config`` performs it."""

    def test_both_find_the_source_config_from_a_foreign_cwd(
        self, tmp_path, monkeypatch, source_config
    ):
        """The disagreement fixing only ``load_config`` would have created:
        the gate reporting "no config → setup mode" about a checkout whose
        config the read then finds."""
        cfg = source_config({"model": {"model": "x"}})
        monkeypatch.chdir(tmp_path)

        assert find_config_file(None) == cfg
        assert load_config()["model"]["model"] == "x"

    def test_both_report_absence_together(self, tmp_path, monkeypatch):
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "none.yaml")
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "empty-home"))
        monkeypatch.chdir(tmp_path)

        assert find_config_file(None) is None
        assert load_config() == {}

    def test_an_explicit_path_reaches_both_identically(self, tmp_path):
        cfg = tmp_path / "named.yaml"
        cfg.write_text("model:\n  model: named\n", encoding="utf-8")

        assert find_config_file(str(cfg)) == cfg
        assert load_config(str(cfg))["model"]["model"] == "named"


class TestSetupModeCreatesNoState:
    """``find_config_file`` runs BEFORE the daemon chooses setup mode."""

    def test_the_gate_does_not_create_the_config_dir(self, tmp_path, monkeypatch):
        """Its docstring's oldest constraint, now held by construction.

        It used to be held by NOT calling the helper — which is what made it a
        fifth copy of the search order, CWD branch included.
        ``config_search_paths`` resolves through ``config_dir_path()``, which
        creates nothing; ``get_config_dir()`` keeps the ``mkdir`` for callers
        about to write.
        """
        home = tmp_path / "never-created"
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(home))
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "none.yaml")

        assert find_config_file(None) is None
        assert not home.exists(), "setup mode's gate created ~/.prometheus state"

    def test_search_paths_creates_nothing_either(self, tmp_path, monkeypatch):
        home = tmp_path / "never-created"
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(home))

        defaults.config_search_paths()
        assert not home.exists()

    def test_config_dir_path_asks_get_config_dir_writes(self, tmp_path, monkeypatch):
        """The split, pinned in both directions."""
        from prometheus.config.paths import config_dir_path, get_config_dir

        home = tmp_path / "home"
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(home))

        assert config_dir_path() == home
        assert not home.exists()
        assert get_config_dir() == home
        assert home.is_dir()


class TestAnUnusableConfigRefusesTheBoot:
    """An error state is not answered with defaults."""

    def test_an_empty_config_file_refuses(self, tmp_path, source_config):
        """The state the old body could not express.

        ``yaml.safe_load(fh) or {}`` turned an empty file into ``{}`` with NO
        log line — the whole system on defaults, silently, because "your config
        is empty" and "you have no config" rendered as the same value.
        """
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("", encoding="utf-8")
        with pytest.raises(ConfigReadError, match="malformed"):
            load_config(str(cfg))

    def test_a_comment_only_config_file_refuses(self, tmp_path):
        cfg = tmp_path / "comments.yaml"
        cfg.write_text("# nothing but a comment\n", encoding="utf-8")
        with pytest.raises(ConfigReadError):
            load_config(str(cfg))

    def test_unparseable_yaml_refuses(self, tmp_path):
        """Not a loosening: this already killed the boot, by propagating
        ``yaml.YAMLError`` out of the old body. It now refuses by name."""
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("model: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigReadError):
            load_config(str(cfg))

    def test_a_named_but_missing_config_refuses(self, tmp_path):
        with pytest.raises(ConfigReadError, match="unreadable"):
            load_config(str(tmp_path / "typo.yaml"))

    def test_a_non_mapping_config_refuses(self, tmp_path):
        cfg = tmp_path / "list.yaml"
        cfg.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigReadError):
            load_config(str(cfg))

    def test_absence_still_boots_on_defaults(self, tmp_path, monkeypatch):
        """ABSENT is legitimate and must NOT refuse — a fresh install has no
        config, and ``main()`` routes that to setup mode rather than here."""
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "none.yaml")
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "empty-home"))
        monkeypatch.chdir(tmp_path)

        assert load_config() == {}

    def test_a_good_config_still_loads(self, source_config, tmp_path, monkeypatch):
        source_config({"model": {"model": "m"}, "security": {"permission_mode": "strict"}})
        monkeypatch.chdir(tmp_path)

        config = load_config()
        assert config["model"]["model"] == "m"
        assert config["security"]["permission_mode"] == "strict"

    def test_the_refusal_names_the_file_and_the_consequence(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("", encoding="utf-8")
        with pytest.raises(ConfigReadError) as caught:
            load_config(str(cfg))
        message = str(caught.value)
        assert str(cfg) in message
        assert "Refusing to boot" in message
        # Names what it would otherwise have run on, not merely what broke.
        assert "security" in message
