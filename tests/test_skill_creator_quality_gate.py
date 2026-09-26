"""SkillCreator's write gate: no malformed descriptions, no near-duplicates.

Two things the audit found in the 57 auto skills ever written
(docs/audits/SKILL-USAGE.md §5): two descriptions that were literally the text
``name: …`` (a frontmatter line the loader's tolerant scan kept verbatim), and
duplicates of skills that already existed. The gate:

- rejects a description that is literally ``name: …``, for every writer;
- on the auto path only (``on_collision="skip"``), rejects a skill whose
  ``name + description`` is within cosine ``dedupe_threshold`` (default
  ``DEFAULT_THRESHOLD`` = 0.80) of an existing served skill, using the audit's
  encoder. Deliberate writers — teacher escalation, record-a-skill, an accepted
  draft — keep their content, as they already do on a name collision.

With no encoder installed the near-duplicate check is skipped and says why;
creation is never blocked by a missing optional dependency.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.learning.skill_creator import SkillCreator
from prometheus.skills.similarity import DEFAULT_THRESHOLD, skill_text

GOOD = ("---\nname: release-check\ndescription: Check a release before tagging it\n---\n"
        "# Release check\n## Steps\n1. run the tests\n")
MALFORMED = "---\nname: widget-export\ndescription: name: widget-export\n---\n# Widget export\n"


class _Checker:
    def __init__(self, hit=None, *, available: bool = True, reason: str | None = None) -> None:
        self.hit = hit
        self._available = available
        self.unavailable_reason = reason
        self.calls: list[tuple[str, list]] = []

    @property
    def available(self) -> bool:
        return self._available

    def nearest(self, text, catalog):
        self.calls.append((text, list(catalog)))
        return self.hit if self._available else None


class _Telemetry:
    def __init__(self) -> None:
        self.runs: list[tuple[tuple, dict]] = []

    def record_run(self, *args, **kwargs) -> None:
        self.runs.append((args, kwargs))

    def record_silent_failure(self, *args, **kwargs) -> None:
        pass

    def gate_summaries(self) -> list[dict]:
        return [kw.get("summary") or {} for a, kw in self.runs
                if a[:3] == ("skill_creator", "quality_gate", "skipped")]


CATALOG = [("release-gate", skill_text("release-gate", "Gate a release on its checks"))]


def _creator(tmp_path: Path, checker: _Checker, **kw) -> tuple[SkillCreator, _Telemetry]:
    tel = _Telemetry()
    creator = SkillCreator(MagicMock(), auto_dir=tmp_path, telemetry=tel, similarity=checker,
                           catalog=lambda: CATALOG, **kw)
    return creator, tel


def _persist(creator: SkillCreator, content: str, on_collision: str = "skip"):
    return asyncio.run(creator.persist_skill_content(content, trigger="t", on_collision=on_collision))


def test_the_stated_default_threshold():
    assert DEFAULT_THRESHOLD == 0.80


class TestMalformedDescription:
    def test_a_description_that_is_literally_name_is_rejected(self, tmp_path):
        creator, tel = _creator(tmp_path, _Checker())
        assert _persist(creator, MALFORMED) is None
        assert list(tmp_path.glob("*.md")) == []
        assert tel.gate_summaries() == [{"reason": "malformed_description"}]

    def test_deliberate_writers_are_guarded_too(self, tmp_path):
        creator, _ = _creator(tmp_path, _Checker())
        assert _persist(creator, MALFORMED, on_collision="suffix") is None

    def test_any_case_and_spacing(self, tmp_path):
        creator, _ = _creator(tmp_path, _Checker())
        content = MALFORMED.replace("description: name:", "description: '  Name :")
        assert _persist(creator, content) is None


class TestNearDuplicate:
    def test_a_near_duplicate_is_rejected_on_the_auto_path(self, tmp_path):
        creator, tel = _creator(tmp_path, _Checker((0.91, "release-gate")))
        assert _persist(creator, GOOD) is None
        assert list(tmp_path.glob("*.md")) == []
        [summary] = tel.gate_summaries()
        assert summary == {"reason": "near_duplicate", "nearest": "release-gate",
                           "score": 0.91, "threshold": DEFAULT_THRESHOLD}

    def test_the_threshold_itself_rejects(self, tmp_path):
        creator, _ = _creator(tmp_path, _Checker((DEFAULT_THRESHOLD, "release-gate")))
        assert _persist(creator, GOOD) is None

    def test_just_below_the_threshold_is_written(self, tmp_path):
        creator, _ = _creator(tmp_path, _Checker((DEFAULT_THRESHOLD - 0.01, "release-gate")))
        assert _persist(creator, GOOD) == tmp_path / "release-check.md"

    def test_the_candidate_is_compared_as_name_plus_description(self, tmp_path):
        checker = _Checker((0.1, "release-gate"))
        creator, _ = _creator(tmp_path, checker)
        _persist(creator, GOOD)
        [(text, catalog)] = checker.calls
        assert text == skill_text("release-check", "Check a release before tagging it")
        assert catalog == CATALOG

    def test_a_deliberate_writer_keeps_a_near_duplicate(self, tmp_path):
        checker = _Checker((0.99, "release-gate"))
        creator, _ = _creator(tmp_path, checker)
        assert _persist(creator, GOOD, on_collision="suffix") == tmp_path / "release-check.md"
        assert checker.calls == []

    def test_a_configured_threshold_is_used(self, tmp_path):
        creator, _ = _creator(tmp_path, _Checker((0.85, "release-gate")), dedupe_threshold=0.9)
        assert _persist(creator, GOOD) == tmp_path / "release-check.md"

    def test_no_encoder_means_no_check_and_it_says_why(self, tmp_path, caplog):
        checker = _Checker(available=False, reason="encoder files not found at /x")
        creator, _ = _creator(tmp_path, checker)
        with caplog.at_level(logging.WARNING, logger="prometheus.learning.skill_creator"):
            assert _persist(creator, GOOD) == tmp_path / "release-check.md"
        assert "encoder files not found at /x" in caplog.text


def test_maybe_create_goes_through_the_gate(tmp_path, monkeypatch):
    creator, tel = _creator(tmp_path, _Checker())

    async def fake_call_model(self, prompt):
        return MALFORMED

    monkeypatch.setattr(SkillCreator, "_call_model", fake_call_model)
    trace = [{"tool_name": "bash", "result": "ok", "is_error": False}] * 3
    assert asyncio.run(creator.maybe_create("do it", trace, "done")) is None
    assert tel.gate_summaries() == [{"reason": "malformed_description"}]


def test_from_config_reads_the_threshold(tmp_path):
    import yaml

    cfg = tmp_path / "prometheus.yaml"
    cfg.write_text(yaml.safe_dump({"learning": {"skill_dedupe_threshold": 0.9}}))
    creator = SkillCreator.from_config(MagicMock(), config_path=str(cfg))
    assert creator._dedupe_threshold == 0.9


def test_the_default_catalog_is_every_served_skill(tmp_path, monkeypatch):
    from prometheus.learning import skill_creator as sc
    from prometheus.skills.registry import SkillRegistry
    from prometheus.skills.types import SkillDefinition

    reg = SkillRegistry()
    for name, source in (("commit", "builtin"), ("docker-deploy", "user"), ("x", "auto")):
        reg.register(SkillDefinition(name=name, description=f"{name} desc", content="#",
                                     source=source))
    monkeypatch.setattr(sc, "load_skill_registry", lambda: reg)
    assert sc.served_skill_catalog() == [
        ("commit", skill_text("commit", "commit desc")),
        ("docker-deploy", skill_text("docker-deploy", "docker-deploy desc")),
        ("x", skill_text("x", "x desc")),
    ]
