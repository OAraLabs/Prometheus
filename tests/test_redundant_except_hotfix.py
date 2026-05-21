"""Pre-Phase-2 hotfix regression tests — SPRINT 4 Tier-1.

Verifies the three `except (OSError, Exception)` swallows identified in
``docs/audits/SILENT-FAILURE-AUDIT.md`` now:

1. Catch ``OSError`` (file-not-found / permission-denied) and emit a WARN.
2. Catch ``yaml.YAMLError`` (malformed YAML) and emit a WARN.
3. **Propagate any other exception** (this is the Sprint 4 invariant — the
   audit's whole point is that silent swallows hid the ed8f1a6 bug for an
   unknown duration; the hotfix makes them surface).

Sites:
- ``src/prometheus/learning/skill_creator.py:from_config``
- ``src/prometheus/learning/skill_refiner.py:from_config``
- ``src/prometheus/learning/gepa.py:from_config``
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from prometheus.learning.gepa import GEPAOptimizer
from prometheus.learning.skill_creator import SkillCreator
from prometheus.learning.skill_refiner import SkillRefiner

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def malformed_yaml(tmp_path: Path) -> Path:
    """A YAML file that yaml.safe_load will refuse to parse."""
    path = tmp_path / "broken.yaml"
    # Unbalanced braces — yaml.scanner.ScannerError.
    path.write_text("learning: {curator_enabled: true\n", encoding="utf-8")
    return path


@pytest.fixture
def valid_yaml(tmp_path: Path) -> Path:
    """A minimal valid YAML so the gepa/refiner enable paths are reachable."""
    path = tmp_path / "good.yaml"
    path.write_text(
        "learning:\n"
        "  skill_min_tool_calls: 7\n"
        "  skill_refinement_enabled: true\n"
        "  gepa_enabled: true\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fake_provider() -> MagicMock:
    """A stand-in ModelProvider; none of these tests actually call .stream_message."""
    return MagicMock()


# ---------------------------------------------------------------------------
# SkillCreator.from_config — three scenarios
# ---------------------------------------------------------------------------


class TestSkillCreatorConfigLoad:
    """Verifies the narrowed catch at learning/skill_creator.py:from_config."""

    def test_malformed_yaml_warns_and_uses_defaults(
        self, malformed_yaml: Path, fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="prometheus.learning.skill_creator")
        creator = SkillCreator.from_config(fake_provider, config_path=str(malformed_yaml))
        # Defaults still ship.
        assert isinstance(creator, SkillCreator)
        assert creator._min_tool_calls == 3
        # Warning surfaced.
        assert any(
            "SkillCreator.from_config: failed to load" in rec.message
            for rec in caplog.records
            if rec.levelno == logging.WARNING
        ), caplog.text

    def test_missing_file_warns_and_uses_defaults(
        self, tmp_path: Path, fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="prometheus.learning.skill_creator")
        missing = tmp_path / "no-such-file.yaml"
        creator = SkillCreator.from_config(fake_provider, config_path=str(missing))
        assert creator._min_tool_calls == 3
        assert any("failed to load" in rec.message for rec in caplog.records), caplog.text

    def test_unexpected_exception_propagates(
        self,
        valid_yaml: Path,
        fake_provider: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Sprint 4 invariant — non-OSError / non-YAMLError exceptions surface.

        Before the hotfix the `except (OSError, Exception)` would swallow this
        RuntimeError and silently demote the subsystem to defaults. After the
        hotfix, the exception propagates so the operator sees it on startup.
        """
        def _boom(_):
            raise RuntimeError("synthetic sprint 4 regression")

        # Patch yaml.safe_load AS USED INSIDE skill_creator. The module-local
        # `import yaml` inside from_config means we monkeypatch the top-level
        # module attribute.
        monkeypatch.setattr(yaml, "safe_load", _boom)
        with pytest.raises(RuntimeError, match="synthetic sprint 4 regression"):
            SkillCreator.from_config(fake_provider, config_path=str(valid_yaml))


# ---------------------------------------------------------------------------
# SkillRefiner.from_config — three scenarios
# ---------------------------------------------------------------------------


class TestSkillRefinerConfigLoad:
    """Verifies the narrowed catch at learning/skill_refiner.py:from_config."""

    def test_malformed_yaml_warns_and_returns_none(
        self, malformed_yaml: Path, fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed YAML → warning, then `learning` is empty → returns None
        because ``skill_refinement_enabled`` defaults to False."""
        caplog.set_level(logging.WARNING, logger="prometheus.learning.skill_refiner")
        result = SkillRefiner.from_config(fake_provider, config_path=str(malformed_yaml))
        assert result is None  # disabled fallback
        assert any(
            "SkillRefiner.from_config: failed to load" in rec.message
            for rec in caplog.records
            if rec.levelno == logging.WARNING
        ), caplog.text

    def test_missing_file_warns_and_returns_none(
        self, tmp_path: Path, fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="prometheus.learning.skill_refiner")
        missing = tmp_path / "no-such-file.yaml"
        result = SkillRefiner.from_config(fake_provider, config_path=str(missing))
        assert result is None
        assert any("failed to load" in rec.message for rec in caplog.records), caplog.text

    def test_unexpected_exception_propagates(
        self,
        valid_yaml: Path,
        fake_provider: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(_):
            raise RuntimeError("synthetic sprint 4 regression")

        monkeypatch.setattr(yaml, "safe_load", _boom)
        with pytest.raises(RuntimeError, match="synthetic sprint 4 regression"):
            SkillRefiner.from_config(fake_provider, config_path=str(valid_yaml))


# ---------------------------------------------------------------------------
# GEPAOptimizer.from_config — three scenarios
# ---------------------------------------------------------------------------


class TestGEPAOptimizerConfigLoad:
    """Verifies the narrowed catch at learning/gepa.py:from_config."""

    def test_malformed_yaml_warns_and_returns_none(
        self, malformed_yaml: Path, fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed YAML → warning, then `learning` is empty → returns None
        because ``gepa_enabled`` defaults to False."""
        caplog.set_level(logging.WARNING, logger="prometheus.learning.gepa")
        result = GEPAOptimizer.from_config(fake_provider, config_path=str(malformed_yaml))
        assert result is None  # disabled fallback
        assert any(
            "GEPAOptimizer.from_config: failed to load" in rec.message
            for rec in caplog.records
            if rec.levelno == logging.WARNING
        ), caplog.text

    def test_missing_file_warns_and_returns_none(
        self, tmp_path: Path, fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING, logger="prometheus.learning.gepa")
        missing = tmp_path / "no-such-file.yaml"
        result = GEPAOptimizer.from_config(fake_provider, config_path=str(missing))
        assert result is None
        assert any("failed to load" in rec.message for rec in caplog.records), caplog.text

    def test_unexpected_exception_propagates(
        self,
        valid_yaml: Path,
        fake_provider: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _boom(_):
            raise RuntimeError("synthetic sprint 4 regression")

        monkeypatch.setattr(yaml, "safe_load", _boom)
        with pytest.raises(RuntimeError, match="synthetic sprint 4 regression"):
            GEPAOptimizer.from_config(fake_provider, config_path=str(valid_yaml))
