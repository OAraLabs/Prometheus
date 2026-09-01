"""``config.defaults`` must name a config file that CAN exist.

THE DEFECT
----------
``DEFAULTS_PATH`` applied five ``.parent``s to
``src/prometheus/config/defaults.py``. ``.parent`` #1 is this file's own
``config/`` directory, so five hops landed one directory ABOVE the repo root:
``~/config/prometheus.yaml`` for a checkout at ``~/Prometheus``, and the same
nonexistent path for the ff-only deploy clone. Eight subsystems used it as
their fallback, every one of them wrapped in an ``except OSError``, so the
miss was silent — ``TokenBudget.from_config()`` answered 24000 on a box whose
config says 72000.

WHY A HOP COUNT NEEDS A TEST AT ALL
-----------------------------------
``config/template.py`` and ``cli/doctor.py`` both computed the SAME repo root
from the SAME package with FOUR hops and were right; ``defaults.py`` used five
and was wrong. Nothing could see the disagreement, because a wrong count
produces a valid ``Path`` — it fails only against a filesystem. So these tests
anchor the count on a repo marker (``pyproject.toml``) rather than on anyone's
counting, and pin ``defaults`` and ``template`` to the same root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.config import defaults  # noqa: E402
from prometheus.config.defaults import (  # noqa: E402
    config_search_paths,
    resolve_config_path,
)


# The autouse conftest fixture neutralises REPO_CONFIG_PATH so no test reads the
# developer's own gitignored config. The anchoring tests below are about the REAL
# constant — the actual ``parents[N]`` expression in config/defaults.py — so it is
# snapshotted HERE, at test-module import, which happens at collection before any
# function-scoped fixture runs.
#
# ⚠ Do NOT recompute this from ``defaults.__file__`` with a hop count of its own:
# a guard that restates the expression it is guarding passes for any value of N.
# Verified by re-breaking the source to ``parents[4]`` and watching it fail.
_REAL_REPO_CONFIG_PATH: Path = defaults.REPO_CONFIG_PATH


def _real_repo_config_path() -> Path:
    return _REAL_REPO_CONFIG_PATH


class TestRepoRootAnchoring:
    """The hop count, checked against the filesystem."""

    def test_repo_config_path_sits_under_the_real_repo_root(self):
        """``<repo>/config/prometheus.yaml`` — so ``.parent.parent`` is the repo.

        THE regression test for the original bug: with five hops this asserts
        against ``~/`` (or ``.claude/worktrees/``), where no ``pyproject.toml``
        lives. Anchoring on a tracked marker file means the count is verified,
        not restated.
        """
        repo_root = _real_repo_config_path().parent.parent
        assert (repo_root / "pyproject.toml").is_file(), (
            f"REPO_CONFIG_PATH resolved to {_real_repo_config_path()}, whose "
            f"grandparent {repo_root} is not a Prometheus checkout — the "
            f"``parents[N]`` count in config/defaults.py is off."
        )
        assert (repo_root / "src" / "prometheus" / "config" / "defaults.py").is_file()

    def test_defaults_and_template_agree_on_the_repo_root(self):
        """Two modules in the SAME package, two hop counts. Pin them equal.

        ``config/template.py`` says "repo root is four parents up" in a comment
        and uses ``parents[3]``. ``defaults.py`` used five. A shared assertion
        is what makes the next edit to either one fail loudly.
        """
        from prometheus.config import template

        template_root = Path(template.__file__).resolve().parents[3]
        assert _real_repo_config_path().parent.parent == template_root

    def test_the_checkout_template_lives_beside_the_resolved_config(self):
        """``config/prometheus.yaml.default`` is tracked and sits where the
        resolver expects ``prometheus.yaml`` — the strongest available proof
        that the directory is the right one, since the config itself is
        gitignored and absent from worktrees and CI."""
        assert (_real_repo_config_path().parent / "prometheus.yaml.default").is_file()


class TestSearchOrder:
    """Explicit > repo-local > ``$PROMETHEUS_CONFIG_DIR``."""

    def test_explicit_path_short_circuits_everything(self, tmp_path, monkeypatch):
        explicit = tmp_path / "mine.yaml"
        explicit.write_text("context: {}\n")
        repo_cfg = tmp_path / "repo" / "prometheus.yaml"
        repo_cfg.parent.mkdir()
        repo_cfg.write_text("context: {}\n")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", repo_cfg)

        assert config_search_paths(explicit) == [explicit]
        assert resolve_config_path(explicit) == explicit

    def test_a_missing_explicit_path_does_not_fall_through(self, tmp_path, monkeypatch):
        """A caller that NAMED a file gets that file or nothing.

        ``__main__.load_config`` used to fall through from a missing
        ``--config`` to ``~/.prometheus/prometheus.yaml``, so a typo'd flag
        silently ran against a different config. ``daemon.load_config`` already
        refused to; the two now agree.
        """
        repo_cfg = tmp_path / "repo" / "prometheus.yaml"
        repo_cfg.parent.mkdir()
        repo_cfg.write_text("context: {effective_limit: 999}\n")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", repo_cfg)
        user_cfg = tmp_path / "prom-config" / "prometheus.yaml"
        user_cfg.parent.mkdir(parents=True)
        user_cfg.write_text("context: {effective_limit: 888}\n")
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(user_cfg.parent))

        missing = tmp_path / "typo.yaml"
        assert resolve_config_path(missing) == missing
        assert not resolve_config_path(missing).is_file()

    def test_repo_local_config_wins_over_the_user_config_dir(self, tmp_path, monkeypatch):
        repo_cfg = tmp_path / "repo" / "prometheus.yaml"
        repo_cfg.parent.mkdir()
        repo_cfg.write_text("context: {}\n")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", repo_cfg)
        user_dir = tmp_path / "prom-config"
        user_dir.mkdir()
        (user_dir / "prometheus.yaml").write_text("context: {}\n")
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(user_dir))

        assert resolve_config_path() == repo_cfg

    def test_pip_install_layout_falls_back_to_the_user_config_dir(self, tmp_path, monkeypatch):
        """The layout that made the constant unfixable.

        A wheel packages ``src/prometheus`` only — ``config/`` is not in it
        (``config/template.py`` had to be written for exactly this reason). So
        under site-packages there IS no repo-local candidate and the answer has
        to come from ``$PROMETHEUS_CONFIG_DIR``, which no constant can express.
        """
        monkeypatch.setattr(
            defaults, "REPO_CONFIG_PATH", tmp_path / "site-packages-has-no-repo.yaml"
        )
        user_dir = tmp_path / "prom-config"
        user_dir.mkdir()
        user_cfg = user_dir / "prometheus.yaml"
        user_cfg.write_text("context: {effective_limit: 4242}\n")
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(user_dir))

        assert resolve_config_path() == user_cfg


class TestNeverNoneContract:
    """The contract the eight ``from_config`` fallbacks rest on."""

    def test_returns_the_user_config_candidate_when_nothing_exists(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "nope.yaml")
        user_dir = tmp_path / "prom-config"
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(user_dir))

        resolved = resolve_config_path()
        assert resolved is not None
        assert resolved == user_dir / "prometheus.yaml"
        assert not resolved.is_file()

    def test_the_unfound_path_raises_OSError_on_open_not_TypeError(
        self, tmp_path, monkeypatch
    ):
        """Why never-None is a contract and not a nicety.

        ``skill_creator``, ``skill_refiner``, ``gepa`` and ``nudge`` catch only
        ``(OSError, yaml.YAMLError)``. A ``None`` here becomes ``Path(None)`` ->
        ``TypeError``, which those four do NOT catch — it would propagate out
        of ``SkillRefiner.from_config`` during the daemon's boot. A nonexistent
        Path raises ``FileNotFoundError``, which every caller already handles,
        so "no config anywhere" behaves exactly as it did before the fix.
        """
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "nope.yaml")
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "prom-config"))

        with pytest.raises(OSError):
            open(resolve_config_path())

    def test_no_module_level_constant_reintroduces_the_bug(self):
        """``DEFAULTS_PATH`` is gone on purpose — a constant cannot express a
        search across three install layouts. If it comes back, this fails."""
        assert not hasattr(defaults, "DEFAULTS_PATH")


class TestSearchOrderHasOneImplementation:
    """``doctor`` and ``__main__`` delegate rather than re-deriving."""

    def test_doctor_resolves_through_config_search_paths(self, tmp_path, monkeypatch):
        pytest.importorskip("httpx")
        from prometheus.cli import doctor

        repo_cfg = tmp_path / "repo" / "prometheus.yaml"
        repo_cfg.parent.mkdir()
        repo_cfg.write_text("model: {model: m}\n")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", repo_cfg)

        found, searched = doctor.resolve_config_path()
        assert found == repo_cfg
        assert searched == [repo_cfg]

    def test_doctor_reports_every_candidate_when_nothing_is_found(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("httpx")
        from prometheus.cli import doctor

        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "nope.yaml")
        user_dir = tmp_path / "prom-config"
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(user_dir))

        found, searched = doctor.resolve_config_path()
        assert found is None
        assert searched == [tmp_path / "nope.yaml", user_dir / "prometheus.yaml"]

    def test_main_load_config_reads_the_repo_local_file(self, tmp_path, monkeypatch):
        from prometheus.__main__ import load_config

        repo_cfg = tmp_path / "repo" / "prometheus.yaml"
        repo_cfg.parent.mkdir()
        repo_cfg.write_text("context:\n  effective_limit: 31337\n")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", repo_cfg)

        assert load_config()["context"]["effective_limit"] == 31337
