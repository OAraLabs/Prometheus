"""learning.skill_dedupe_threshold reaches the near-duplicate gate (#591).

The template documents the key (default 0.80), but only
``SkillCreator.from_config`` read it, and nothing calls ``from_config``. The
daemon's ``_wire_skill_creator`` built its SkillCreator without it, so the gate
on the automatic path was always 0.80 whatever the config said.

Three places build a SkillCreator, and each now passes the key:

* the daemon's instance, the one whose automatic path runs the gate (the fix);
* the teacher's own instance, built only when none is handed in;
* the live recorder's own instance, built only on a standalone web launch.

The last two write through ``persist_or_divert`` (refuse mode), which does not
run the gate today. They carry the key so every instance agrees with the config.

Each test runs the gate itself, with a checker that scores 0.60: a near-duplicate
under a configured 0.55, not under the default 0.80.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.learning.skill_creator import SkillCreator
from prometheus.skills.similarity import DEFAULT_THRESHOLD

CONFIGURED = 0.55
SCORE = 0.60
assert CONFIGURED < SCORE < DEFAULT_THRESHOLD


class _HookRecorder:
    """Stands in for AgentLoop: records the post-task hooks registered."""

    def __init__(self) -> None:
        self.hooks: list = []

    def add_post_task_hook(self, hook) -> None:
        self.hooks.append(hook)


class _ScoresSixty:
    """A near-duplicate checker that is always available and scores SCORE."""

    available = True
    unavailable_reason = None

    def nearest(self, text, catalog):
        return SCORE, catalog[0][0]


@pytest.fixture(autouse=True)
def _auto_dir_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ~/.prometheus/skills/auto/ out of the tests."""
    monkeypatch.setattr(
        "prometheus.learning.skill_creator.get_config_dir", lambda: tmp_path
    )


def _gate_rejects(creator: SkillCreator) -> bool:
    """Run the near-duplicate gate against one served skill."""
    creator._similarity = _ScoresSixty()
    creator._catalog = lambda: [("existing-skill", "existing skill: does the thing")]
    return creator._near_duplicate("new-skill", "does the thing again")


# ---------------------------------------------------------------------------
# The daemon's instance: the automatic path, where the gate runs
# ---------------------------------------------------------------------------


def test_the_daemons_skill_creator_gates_at_the_configured_threshold():
    from prometheus.daemon import _wire_skill_creator

    creator = _wire_skill_creator(
        _HookRecorder(), MagicMock(), model_name="m",
        learning_config={"skill_dedupe_threshold": CONFIGURED},
    )
    assert creator is not None
    assert _gate_rejects(creator), (
        "a 0.60 near-duplicate passed the gate: the configured "
        f"{CONFIGURED} never reached the daemon's SkillCreator"
    )


def test_absent_the_daemon_keeps_the_calibrated_default():
    from prometheus.daemon import _wire_skill_creator

    creator = _wire_skill_creator(
        _HookRecorder(), MagicMock(), model_name="m", learning_config={},
    )
    assert creator is not None
    assert not _gate_rejects(creator)


# ---------------------------------------------------------------------------
# The two other instances carry the key too
# ---------------------------------------------------------------------------


def test_the_teachers_own_skill_creator_carries_the_configured_threshold():
    from prometheus.escalation.teacher import TeacherEscalation

    teacher = TeacherEscalation.from_config(
        {"learning": {"skill_dedupe_threshold": CONFIGURED}})
    teacher._provider = MagicMock()  # the persist path makes no model call
    creator = teacher._ensure_skill_creator()
    assert creator is not None
    assert _gate_rejects(creator)


def test_the_live_recorders_own_skill_creator_carries_the_configured_threshold(
    monkeypatch: pytest.MonkeyPatch,
):
    """A standalone web launch: create_app without the daemon's SkillCreator."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app
    from tests.test_api_live_upload import _events, _metadata

    built: list[SkillCreator] = []

    class _Recorded(SkillCreator):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr("prometheus.learning.skill_creator.SkillCreator", _Recorded)
    app = create_app({"learning": {"skill_dedupe_threshold": CONFIGURED,
                                   "live_recorder": {"verify_steps": False}}})
    resp = TestClient(app).post(
        "/api/learning/live-upload",
        data={"events": json.dumps(_events()), "metadata": json.dumps(_metadata())},
    )
    assert resp.status_code == 200, resp.text
    [creator] = built
    assert _gate_rejects(creator)
