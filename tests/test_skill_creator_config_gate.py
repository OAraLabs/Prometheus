"""learning.auto_skill_creation / skill_min_tool_calls must reach the live path.

Live bug (2026-08-03, skillcreator-quality-gate survey §1): daemon.py built
``SkillCreator(provider, model=model_name, telemetry=telemetry)`` directly,
so ``learning.skill_min_tool_calls`` silently rode the hardcoded default (3)
and ``learning.auto_skill_creation`` was read by no code anywhere — the
template promised an off switch the live path did not have.

The daemon now routes construction through ``_wire_skill_creator``, whose
contract these tests pin:

* ``auto_skill_creation: false`` → NO post-task hook, but the instance is
  still built — teacher escalation and record-a-skill trace uploads share
  it, and neither is "auto" creation.
* ``skill_min_tool_calls`` is forwarded from the daemon's loaded config
  dict into the gate that decides whether a trace becomes a skill.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.daemon import _wire_skill_creator
from prometheus.learning.skill_creator import SkillCreator

pytestmark = pytest.mark.integration

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


class _HookRecorder:
    """Stands in for AgentLoop — records what gets registered."""

    def __init__(self) -> None:
        self.hooks: list = []

    def add_post_task_hook(self, hook) -> None:
        self.hooks.append(hook)


@pytest.fixture(autouse=True)
def _auto_dir_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ~/.prometheus/skills/auto/ out of unit tests."""
    monkeypatch.setattr(
        "prometheus.learning.skill_creator.get_config_dir", lambda: tmp_path
    )


@pytest.fixture
def fake_provider() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# The regression: auto_skill_creation: false → no post-task hook
# ---------------------------------------------------------------------------


def test_auto_skill_creation_false_wires_no_hook(fake_provider: MagicMock) -> None:
    loop = _HookRecorder()
    creator = _wire_skill_creator(
        loop,
        fake_provider,
        model_name="gemma4-26b",
        learning_config={"auto_skill_creation": False},
    )
    assert loop.hooks == [], (
        "auto_skill_creation: false must leave the agent loop without a "
        "SkillCreator post-task hook — this was the config-dark bug"
    )
    # The flag governs AUTO creation only; the explicit write paths
    # (teacher escalation, record-a-skill) still receive an instance.
    assert isinstance(creator, SkillCreator)


def test_enabled_by_default_wires_maybe_create(fake_provider: MagicMock) -> None:
    """An empty learning section keeps today's live behaviour: hook on."""
    loop = _HookRecorder()
    creator = _wire_skill_creator(
        loop, fake_provider, model_name="gemma4-26b", learning_config={}
    )
    assert loop.hooks == [creator.maybe_create]
    assert creator._model == "gemma4-26b"


# ---------------------------------------------------------------------------
# skill_min_tool_calls threads through — and actually gates
# ---------------------------------------------------------------------------


async def test_skill_min_tool_calls_reaches_the_gate(fake_provider: MagicMock) -> None:
    loop = _HookRecorder()
    creator = _wire_skill_creator(
        loop,
        fake_provider,
        model_name="gemma4-26b",
        learning_config={"auto_skill_creation": True, "skill_min_tool_calls": 7},
    )
    assert creator._min_tool_calls == 7

    # Behavioural proof, not just attribute plumbing: a 6-call trace clears
    # the old hardcoded default (3) but must be skipped under min=7 —
    # before any model call is attempted.
    trace = [
        {"tool_name": f"tool_{i}", "result": "", "is_error": False}
        for i in range(6)
    ]
    assert await creator.maybe_create("six-call task", trace) is None
    assert fake_provider.method_calls == []


def test_bad_min_tool_calls_degrades_without_hook(
    fake_provider: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """A garbage value keeps the daemon booting — warn, no creator, no hook."""
    caplog.set_level(logging.WARNING, logger="prometheus.daemon")
    loop = _HookRecorder()
    creator = _wire_skill_creator(
        loop,
        fake_provider,
        model_name="gemma4-26b",
        learning_config={"skill_min_tool_calls": "banana"},
    )
    assert creator is None
    assert loop.hooks == []
    assert any(
        "SkillCreator not available" in rec.message for rec in caplog.records
    ), caplog.text


# ---------------------------------------------------------------------------
# Call-site guard — the tested function must be the one run_daemon uses
# ---------------------------------------------------------------------------


def test_run_daemon_routes_through_the_gate() -> None:
    """daemon.py must build SkillCreator via _wire_skill_creator, never
    directly — a direct construction is exactly how both knobs went dark.

    Same AST approach as test_daemon_cloud_iteration_cap.py: assert on the
    source so the guard holds without booting the daemon.
    """
    tree = ast.parse((SRC / "daemon.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_wire_skill_creator"
    ]
    assert calls, "run_daemon no longer calls _wire_skill_creator"
    kwargs = {kw.arg for call in calls for kw in call.keywords}
    assert "learning_config" in kwargs, (
        "_wire_skill_creator must receive the daemon's learning config — "
        "without it both knobs fall back to hardcoded defaults again"
    )

    gate_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_wire_skill_creator"
    )
    inside_gate = {
        id(node) for node in ast.walk(gate_fn) if isinstance(node, ast.Call)
    }
    direct = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "SkillCreator"
        and id(node) not in inside_gate
    ]
    assert not direct, (
        "daemon.py constructs SkillCreator outside _wire_skill_creator "
        "(bypassing the config gate) at line(s): "
        + ", ".join(str(n.lineno) for n in direct)
    )
