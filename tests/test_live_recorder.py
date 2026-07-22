"""Tests for the live recorder pipeline (record-a-skill, DOM path).

Covers the deterministic stages — event processing, action mapping,
synthesis, quality gate — and the LiveRecorderService orchestration with
a real SkillCreator (MagicMock provider; persist makes no model call).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.learning.live_recorder.event_processor import process_events
from prometheus.learning.live_recorder.event_to_actions import events_to_actions
from prometheus.learning.live_recorder.quality_gate import gate_actions
from prometheus.learning.live_recorder.service import LiveRecorderService
from prometheus.learning.live_recorder.synthesizer import build_skill_content
from prometheus.learning.skill_creator import SkillCreator


def _element(**overrides):
    base = {
        "tagName": "INPUT",
        "id": "",
        "className": "",
        "textContent": "",
        "ariaLabel": "",
        "name": "",
        "type": "text",
        "value": "",
        "placeholder": "",
        "cssSelector": "input.field",
        "xpath": "/input",
        "closestLabel": "",
    }
    base.update(overrides)
    return base


def _recording_events():
    """A realistic little trace: navigate, fill two fields, submit."""
    return [
        {
            "timestamp": 1000, "type": "navigation",
            "fromUrl": "https://app.example.com/",
            "toUrl": "https://app.example.com/contacts/new",
            "url": "https://app.example.com/contacts/new",
            "pageTitle": "New Contact",
        },
        {
            "timestamp": 2000, "type": "click",
            "url": "https://app.example.com/contacts/new", "pageTitle": "New Contact",
            "element": _element(cssSelector="#name", id="name", closestLabel="Full name"),
        },
        {
            "timestamp": 2500, "type": "input", "inputValue": "Ada Lovelace",
            "url": "https://app.example.com/contacts/new", "pageTitle": "New Contact",
            "element": _element(cssSelector="#name", id="name", name="name",
                                closestLabel="Full name", value="Ada Lovelace"),
        },
        {
            "timestamp": 4000, "type": "input", "inputValue": "ada@example.com",
            "url": "https://app.example.com/contacts/new", "pageTitle": "New Contact",
            "element": _element(cssSelector="#email", id="email", name="email",
                                type="email", closestLabel="Email", value="ada@example.com"),
        },
        {
            "timestamp": 5000, "type": "scroll",
            "url": "https://app.example.com/contacts/new", "pageTitle": "New Contact",
        },
        {
            "timestamp": 6000, "type": "click",
            "url": "https://app.example.com/contacts/new", "pageTitle": "New Contact",
            "element": _element(tagName="BUTTON", cssSelector="button.save",
                                textContent="Save contact", type=""),
        },
        {
            "timestamp": 6100, "type": "submit",
            "url": "https://app.example.com/contacts/new", "pageTitle": "New Contact",
            "element": _element(tagName="FORM", cssSelector="form"),
        },
    ]


METADATA = {"start_url": "https://app.example.com/contacts/new", "duration_seconds": 6}


# ── event_processor ──────────────────────────────────────────────────


def test_process_events_filters_noise_and_groups():
    result = process_events(_recording_events())
    assert result["success"] is True
    assert result["raw_event_count"] == 7
    # Scroll dropped
    assert result["filtered_event_count"] == 6
    # click#name + input#name grouped into one fill-field action
    types_flat = [e["type"] for g in result["actions"] for e in g["events"]]
    assert "scroll" not in types_flat
    grouped_sizes = [len(g["events"]) for g in result["actions"]]
    assert 2 in grouped_sizes


def test_process_events_rejects_non_list():
    assert process_events({"not": "a list"})["success"] is False


def test_duplicate_rapid_clicks_are_filtered():
    click = {
        "timestamp": 1000, "type": "click", "url": "https://x.test/",
        "element": _element(tagName="BUTTON", cssSelector="button.b", textContent="Go"),
    }
    double = [click, {**click, "timestamp": 1300}]
    result = process_events(double)
    assert result["filtered_event_count"] == 1


def test_attach_screenshots_by_index():
    events = _recording_events()
    events[1]["screenshotIndex"] = 0
    events[5]["screenshotIndex"] = 1
    result = process_events(events, screenshots={0: "/tmp/0.jpg", 1: "/tmp/1.jpg"})
    shots = [g.get("screenshot") for g in result["actions"]]
    assert "/tmp/0.jpg" in shots and "/tmp/1.jpg" in shots


# ── event_to_actions ─────────────────────────────────────────────────


def test_events_to_actions_maps_types_and_parameters():
    actions_data = events_to_actions(process_events(_recording_events()))
    assert actions_data["success"] is True
    types = [a["action_type"] for a in actions_data["actions"]]
    assert "NAVIGATE" in types
    assert "FILL_FIELD" in types  # grouped click+input
    assert "TYPE" in types        # bare email input
    assert "CLICK_BUTTON" in types
    assert "SUBMIT" in types

    params = actions_data["parameters"]
    param_names = {p["name"] for p in params}
    assert "name" in param_names
    assert "email" in param_names


def test_password_values_are_masked():
    events = [{
        "timestamp": 1, "type": "input", "inputValue": "hunter2",
        "url": "https://x.test/login",
        "element": _element(type="password", name="pw", cssSelector="#pw",
                            closestLabel="Password"),
    }]
    actions_data = events_to_actions(process_events(events))
    action = actions_data["actions"][0]
    assert "hunter2" not in str(action)
    assert action["value"] == "••••••"


# ── synthesizer ──────────────────────────────────────────────────────


def test_skill_content_has_prometheus_frontmatter():
    actions_data = events_to_actions(process_events(_recording_events()))
    draft = build_skill_content(actions_data["actions"], actions_data["parameters"], METADATA)

    lines = draft.content.splitlines()
    assert lines[0] == "---"
    assert lines[1].startswith("name: ")
    assert lines[2].startswith("description: ")
    assert "## When to use" in draft.content
    assert "## Steps" in draft.content
    assert "## Notes" in draft.content
    # The name must survive SkillCreator's extraction/slugify
    assert SkillCreator._extract_name(draft.content) == draft.name


def test_skill_content_lists_parameters():
    actions_data = events_to_actions(process_events(_recording_events()))
    draft = build_skill_content(actions_data["actions"], actions_data["parameters"], METADATA)
    assert "## Parameters" in draft.content
    assert "**email**" in draft.content


# ── quality gate ─────────────────────────────────────────────────────


def test_quality_gate_passes_good_recording():
    actions_data = events_to_actions(process_events(_recording_events()))
    gate = gate_actions(actions_data["actions"], actions_data["parameters"])
    assert gate.overall in ("pass", "warn")


def test_quality_gate_fails_trivial_recording():
    events = _recording_events()[:2]  # navigation + one click: below min steps
    actions_data = events_to_actions(process_events(events))
    gate = gate_actions(actions_data["actions"], actions_data["parameters"])
    failed_names = {c.name for c in gate.checks if not c.passed}
    assert "min_viable_steps" in failed_names


def test_quality_gate_flags_consecutive_duplicates():
    action = {
        "action_type": "CLICK_BUTTON", "target": "Save", "description": "Click Save",
        "url": "https://x.test/", "is_parameter": False,
    }
    gate = gate_actions([dict(action) for _ in range(5)], [])
    failed_names = {c.name for c in gate.checks if not c.passed}
    assert "no_consecutive_dupes" in failed_names


# ── service orchestration ────────────────────────────────────────────


@pytest.fixture
def creator(tmp_path: Path) -> SkillCreator:
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    return SkillCreator(MagicMock(), auto_dir=auto_dir)


async def test_service_persists_skill(tmp_path: Path, creator: SkillCreator):
    service = LiveRecorderService(creator, recordings_dir=tmp_path / "recordings")
    result = await service.handle_upload(
        _recording_events(), {"startUrl": METADATA["start_url"], "duration": 6000},
        screenshots=[b"\xff\xd8fakejpeg"],
    )

    assert result["status"] == "created"
    skill_path = Path(result["skill_path"])
    assert skill_path.exists()
    content = skill_path.read_text()
    assert content.startswith("---")
    assert "## Steps" in content

    # Provenance archive
    rec_dir = tmp_path / "recordings" / result["recording_id"]
    assert (rec_dir / "events.json").exists()
    assert (rec_dir / "actions.json").exists()
    assert (rec_dir / "quality_gate.json").exists()
    assert (rec_dir / "screenshots" / "0.jpg").read_bytes() == b"\xff\xd8fakejpeg"


async def test_service_rejects_gate_failure(tmp_path: Path, creator: SkillCreator):
    service = LiveRecorderService(creator, recordings_dir=tmp_path / "recordings")
    # A lone navigation fails two checks (min steps + 100% one action type)
    # which tips the gate from "warn" to "fail".
    result = await service.handle_upload(
        _recording_events()[:1], {"startUrl": "https://x.test/"},
    )
    assert result["status"] == "rejected"
    assert result["reason"] == "quality_gate"
    # Nothing persisted
    assert not list((tmp_path / "auto").glob("*.md"))


async def test_service_reloads_registry_after_persist(tmp_path: Path, creator: SkillCreator):
    registry = MagicMock()
    service = LiveRecorderService(
        creator, skill_registry=registry, recordings_dir=tmp_path / "recordings",
    )
    result = await service.handle_upload(
        _recording_events(), {"startUrl": METADATA["start_url"], "duration": 6000},
    )
    assert result["status"] == "created"
    registry.reload_user_skills.assert_called_once()
