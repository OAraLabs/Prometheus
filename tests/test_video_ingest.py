"""Tests for the video-ingestion pipeline (record-a-skill, vision path).

Deterministic — no network, no ffmpeg, no model. Covers the funnel
adapter, the action extractor's tooltip filter / target normalization /
parameterization, the vision digest's JSON parsing + checkpoint resume,
the model-registry gate against the real config/model_registry.yaml,
keyframe selection (SSIM path skipped when scikit-image is absent; the
no-SSIM fallback is exercised via the import-guard hook), and the
pipeline orchestration with stubbed frame/keyframe/digest stages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import prometheus.learning.video_ingest.pipeline as pipeline_mod
from prometheus.learning.video_ingest import keyframes as keyframes_mod
from prometheus.learning.video_ingest import vision_digest as vision_digest_mod
from prometheus.learning.video_ingest.action_extractor import (
    extract_actions,
    parameterize_actions,
)
from prometheus.learning.video_ingest.funnel import (
    to_funnel_action,
    to_funnel_parameters,
)
from prometheus.learning.video_ingest.ingest import align_narration
from prometheus.learning.video_ingest.pipeline import ingest_video_to_skill
from prometheus.learning.video_ingest.vision_digest import (
    clear_checkpoint,
    digest_keyframes,
    load_checkpoint,
    parse_json_response,
    save_checkpoint,
    validate_vision_model,
)
from prometheus.providers.base import ApiTextDeltaEvent


# ── Helpers ──────────────────────────────────────────────────────────


def _digest(idx, actions, url="https://app.example.com/contacts", narration=""):
    """A canned digest entry in the shape digest_keyframes produces."""
    return {
        "keyframe_index": idx,
        "vision": {
            "actions": actions,
            "context": {
                "application": "ExampleCRM",
                "page_or_view": "Contacts",
                "url": url,
                "page_title": "Contacts",
            },
        },
        "narration": narration,
    }


def _good_digests():
    """Digests producing a varied, gate-passing action sequence."""
    return [
        _digest(0, [{"type": "navigate", "target": "Contacts",
                     "description": "Navigate to the Contacts page"}]),
        _digest(1, [{"type": "type", "target": "Full name",
                     "description": "User types 'Ada Lovelace' into the Full name field"}]),
        _digest(2, [{"type": "type", "target": "Email",
                     "description": "User types 'ada@example.com' into the Email field"}]),
        _digest(3, [{"type": "select", "target": "Status",
                     "description": "Select 'Active' from the Status dropdown"}]),
        _digest(4, [{"type": "click", "target": "Save contact",
                     "description": "Click the 'Save contact' button"}]),
    ]


def _bad_digests():
    """Five identical clicks: fails consecutive-dupes + action-distribution."""
    click = {"type": "click", "target": "Refresh", "description": "Click Refresh"}
    return [_digest(i, [dict(click)]) for i in range(5)]


# ── funnel mapping ───────────────────────────────────────────────────


def test_funnel_maps_each_action_type():
    cases = {
        "click": "CLICK",
        "type": "TYPE",
        "navigate": "NAVIGATE",
        "select": "SELECT",
        "submit": "SUBMIT",
        "toggle": "TOGGLE",
        "teleport": "UNKNOWN",  # anything unrecognized
    }
    for raw, expected in cases.items():
        action = {"type": raw, "target": "Widget", "description": "Do the thing",
                  "context": {}, "keyframe": 3}
        mapped = to_funnel_action(action)
        assert mapped["action_type"] == expected, raw
        assert mapped["target"] == "Widget"
        assert mapped["description"] == "Do the thing"
        assert mapped["timestamp"] == 3


def test_funnel_click_on_button_becomes_click_button():
    by_desc = to_funnel_action(
        {"type": "click", "target": "Save", "description": "Click the Save button"}
    )
    assert by_desc["action_type"] == "CLICK_BUTTON"
    by_target = to_funnel_action(
        {"type": "click", "target": "Submit button", "description": "Click it"}
    )
    assert by_target["action_type"] == "CLICK_BUTTON"
    plain = to_funnel_action(
        {"type": "click", "target": "row 3", "description": "Click the third row"}
    )
    assert plain["action_type"] == "CLICK"


def test_funnel_type_action_carries_parameter_annotations():
    action = {
        "type": "type", "target": "Full name",
        "description": "User types 'Ada' into the Full name field",
        "parameter": "input_0", "original_value": "Ada",
        "context": {"url": "https://x.test/form", "page_title": "New Contact"},
    }
    mapped = to_funnel_action(action)
    assert mapped["action_type"] == "TYPE"
    assert mapped["is_parameter"] is True
    assert mapped["parameter_name"] == "input_0"
    assert mapped["value"] == "Ada"
    assert mapped["field_type"] == "text"
    assert mapped["url"] == "https://x.test/form"
    assert mapped["page_title"] == "New Contact"


def test_funnel_type_without_parameter_is_not_parameterized():
    mapped = to_funnel_action({"type": "type", "target": "Search",
                               "description": "typing in the search field"})
    assert mapped["action_type"] == "TYPE"
    assert mapped["is_parameter"] is False
    assert mapped["parameter_name"] is None


def test_funnel_page_title_falls_back_to_page_or_view():
    mapped = to_funnel_action({
        "type": "click", "target": "x", "description": "Click x",
        "context": {"page_or_view": "Inbox"},
    })
    assert mapped["page_title"] == "Inbox"


def test_funnel_parameters_shape():
    params = to_funnel_parameters([{
        "name": "input_0", "type": "string", "default": "Ada",
        "description": "User types 'Ada'", "source_step": 2,
    }])
    assert params == [{
        "name": "input_0", "type": "text", "default_value": "Ada",
        "description": "User types 'Ada'",
    }]


# ── action_extractor ─────────────────────────────────────────────────


def test_extract_actions_drops_tooltip_markers():
    digests = [
        _digest(0, [
            {"type": "type", "target": "A1",
             "description": "Type '@' then a name to insert a people smart chip"},
            {"type": "type", "target": "A1",
             "description": "User types 'Cat' into cell A1"},
        ]),
    ]
    actions = extract_actions(digests)
    assert len(actions) == 1
    assert "smart chip" not in actions[0]["description"]


def test_extract_actions_normalizes_cell_ref_targets():
    digests = [
        _digest(0, [{"type": "type", "target": "145",
                     "description": "User types '145' into cell B3"}]),
        # No cell ref: form-field target passes through unchanged
        _digest(1, [{"type": "type", "target": "Job Name",
                     "description": "User types 'Plumber' in the Job Name field"}]),
        # CLICK targets are never rewritten
        _digest(2, [{"type": "click", "target": "cell B3",
                     "description": "Click into cell B3"}]),
    ]
    actions = extract_actions(digests)
    assert actions[0]["target"] == "B3"
    assert actions[1]["target"] == "Job Name"
    assert actions[2]["target"] == "cell B3"


def test_extract_actions_numbers_steps_and_keeps_context():
    actions = extract_actions(_good_digests())
    assert [a["step"] for a in actions] == list(range(1, len(actions) + 1))
    assert actions[0]["keyframe"] == 0
    assert actions[0]["context"]["url"] == "https://app.example.com/contacts"


def test_parameterize_actions_extracts_quoted_values():
    actions = extract_actions(_good_digests())
    actions, parameters = parameterize_actions(actions)

    assert [p["name"] for p in parameters] == ["input_0", "input_1"]
    assert parameters[0]["default"] == "Ada Lovelace"
    assert parameters[1]["default"] == "ada@example.com"

    typed = [a for a in actions if a["type"] == "type"]
    assert typed[0]["parameter"] == "input_0"
    assert typed[0]["original_value"] == "Ada Lovelace"
    # Non-TYPE actions never become parameters
    assert all("parameter" not in a for a in actions if a["type"] != "type")


def test_parameterize_actions_skips_type_without_value():
    actions = [{"step": 1, "keyframe": 0, "type": "type", "target": "Search",
                "description": "typing in the search field", "context": {}}]
    actions, parameters = parameterize_actions(actions)
    assert parameters == []
    assert "parameter" not in actions[0]


# ── vision_digest: JSON parsing ──────────────────────────────────────


def test_parse_json_response_strips_json_fence():
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_response_strips_bare_fence():
    assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_response_plain_json():
    assert parse_json_response('  {"a": 1}  ') == {"a": 1}


def test_parse_json_response_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_json_response("I looked at the image and it is nice.")


# ── vision_digest: checkpoint resume ─────────────────────────────────


def test_checkpoint_save_load_round_trip(tmp_path: Path):
    digests = [{"keyframe_index": 0, "vision": {"actions": [], "context": {}}},
               {"keyframe_index": 1, "vision": {"actions": [], "context": {}}}]
    save_checkpoint(tmp_path, {0, 1}, digests)

    checkpoint = load_checkpoint(tmp_path)
    assert checkpoint["completed_frame_ids"] == {0, 1}
    assert checkpoint["digests"] == digests

    clear_checkpoint(tmp_path)
    fresh = load_checkpoint(tmp_path)
    assert fresh["completed_frame_ids"] == set()
    assert fresh["digests"] == []


def test_load_checkpoint_missing_dir_is_fresh(tmp_path: Path):
    checkpoint = load_checkpoint(tmp_path / "nope")
    assert checkpoint == {"completed_frame_ids": set(), "digests": []}


class _FakeVisionProvider:
    """Streams a canned JSON payload for every call (analysis + comparison)."""

    def __init__(self, payload: dict):
        self._text = json.dumps(payload)
        self.calls = 0

    async def stream_message(self, request):
        self.calls += 1
        # First delta split in two to prove concatenation works
        yield ApiTextDeltaEvent(text=self._text[:10])
        yield ApiTextDeltaEvent(text=self._text[10:])


async def test_digest_keyframes_streams_and_resumes(tmp_path: Path, monkeypatch):
    payload = {
        "application": "ExampleCRM",
        "page_or_view": "Contacts",
        "url_if_visible": "https://app.example.com/contacts",
        "description": "The contacts page.",
        "user_action_visible": "clicking the 'Save contact' button",
        "ui_elements": [], "data_on_screen": [], "cursor_context": "",
        "confidence": 0.9,
        # comparison keys (same payload answers both prompts)
        "notable_changes": "Form saved",
        "action_that_caused_change": "clicked the 'Save contact' button",
        "change_type": "button_click", "change_significance": "high",
    }
    provider = _FakeVisionProvider(payload)

    class _StubRegistry:
        @staticmethod
        def create(cfg):
            assert cfg["model"] == "gemma-4"  # cfg handed through unchanged
            return provider

    monkeypatch.setattr(vision_digest_mod, "ProviderRegistry", _StubRegistry)

    kf_paths = []
    for i in range(3):
        p = tmp_path / f"keyframe_{i:04d}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        kf_paths.append(p)
    keyframes = [{"index": i, "path": str(p), "timestamp": i / 2.0}
                 for i, p in enumerate(kf_paths)]

    # Frame 0 already completed — digest must resume, not redo it
    save_checkpoint(tmp_path, {0}, [{
        "keyframe_index": 0,
        "vision": {"actions": [], "context": {}},
        "narration": "",
    }])

    narration = [{"frame_index": 1, "timestamp": 0.5, "narration": "now I save it"}]
    digests = await digest_keyframes(
        keyframes, narration, {"provider": "stub", "model": "gemma-4"}, tmp_path,
        prompt_context="Save a contact",
    )

    assert [d["keyframe_index"] for d in digests] == [0, 1, 2]
    # Frame 0 came from the checkpoint untouched (no actions)
    assert digests[0]["vision"]["actions"] == []
    # Frames 1-2 were digested: derived click action from the comparison/analysis
    assert digests[1]["vision"]["actions"][0]["type"] == "click"
    assert digests[1]["vision"]["context"]["url"] == "https://app.example.com/contacts"
    assert digests[1]["narration"] == "now I save it"
    # 2 pending frames: 2 analysis calls + 2 comparison calls (prev exists)
    assert provider.calls == 4
    # digests.json written, checkpoint cleared after the successful run
    assert json.loads((tmp_path / "digests.json").read_text())
    assert not (tmp_path / "digest_checkpoint.json").exists()


# ── vision_digest: registry gate ─────────────────────────────────────


def test_validate_vision_model_gemma4_supported():
    ok, explanation = validate_vision_model(
        {"provider": "llama_cpp", "model": "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"}
    )
    assert ok is True
    assert "Gemma 4" in explanation
    # setup_hint surfaces (mmproj requirement)
    assert "mmproj" in explanation


def test_validate_vision_model_qwen_rejected():
    ok, explanation = validate_vision_model(
        {"provider": "llama_cpp", "model": "qwen3-32b-instruct"}
    )
    assert ok is False
    assert "vision" in explanation.lower()


def test_validate_vision_model_unknown_is_permissive():
    ok, explanation = validate_vision_model({"model": "totally-unknown-model-9000"})
    assert ok is True
    assert "not in the capability registry" in explanation


def test_validate_vision_model_missing_registry_is_permissive(tmp_path: Path):
    ok, explanation = validate_vision_model(
        {"model": "qwen3-32b"}, registry_path=tmp_path / "absent.yaml",
    )
    assert ok is True
    assert "registry not found" in explanation


# ── ingest: narration alignment ──────────────────────────────────────


def test_align_narration_window():
    transcription = {
        "text": "first then save",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": 9.0, "end": 10.0, "text": "then save"},
        ],
        "language": "en",
    }
    keyframes = [
        {"index": 0, "timestamp": 0.5},
        {"index": 1, "timestamp": 5.0},   # >3s from both segments
        {"index": 2, "timestamp": 8.0},   # within 3s of the second
    ]
    aligned = align_narration(transcription, keyframes)
    assert aligned[0]["narration"] == "first"
    assert aligned[1]["narration"] == ""
    assert aligned[2]["narration"] == "then save"


def test_align_narration_empty_transcription():
    assert align_narration(None, [{"index": 0}]) == []
    assert align_narration({"segments": []}, [{"index": 0}]) == []


# ── keyframes ────────────────────────────────────────────────────────


def _write_frames(session_dir: Path, blobs: list[bytes]) -> Path:
    shots = session_dir / "screenshots"
    shots.mkdir(parents=True)
    for i, blob in enumerate(blobs):
        (shots / f"frame_{i + 1:04d}.png").write_bytes(blob)
    return shots


def test_keyframes_fallback_every_nth_without_ssim(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(keyframes_mod, "_load_ssim_deps", lambda: None)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 6000  # > blank threshold
    _write_frames(tmp_path, [fake_png] * 10)

    out = tmp_path / "out"
    keyframes = keyframes_mod.extract_keyframes(tmp_path, out, max_keyframes=4)

    assert len(keyframes) == 4  # every 3rd of 10 frames
    assert [k["index"] for k in keyframes] == [0, 1, 2, 3]
    assert keyframes[0]["source_frame"] == "frame_0001.png"
    assert keyframes[1]["source_frame"] == "frame_0004.png"
    for k in keyframes:
        assert Path(k["path"]).exists()
    manifest = json.loads((out / "keyframes.json").read_text())
    assert len(manifest) == 4


def test_keyframes_fallback_skips_blank_frames(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(keyframes_mod, "_load_ssim_deps", lambda: None)
    good = b"\x89PNG\r\n\x1a\n" + b"\x00" * 6000
    blank = b"\x89PNG\r\n\x1a\n"  # tiny: below the 5KB blank threshold
    _write_frames(tmp_path, [blank, good, blank, good])

    keyframes = keyframes_mod.extract_keyframes(tmp_path, tmp_path)
    assert [k["source_frame"] for k in keyframes] == ["frame_0002.png", "frame_0004.png"]


def test_keyframes_empty_session(tmp_path: Path):
    assert keyframes_mod.extract_keyframes(tmp_path) == []  # no screenshots dir
    (tmp_path / "screenshots").mkdir()
    assert keyframes_mod.extract_keyframes(tmp_path) == []  # empty dir


def test_keyframes_ssim_dedups_identical_frames(tmp_path: Path):
    pytest.importorskip("skimage")
    np = pytest.importorskip("numpy")
    pil_image = pytest.importorskip("PIL.Image")

    rng = np.random.default_rng(42)
    noise_a = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    noise_b = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

    shots = tmp_path / "screenshots"
    shots.mkdir()
    # Frames 1-5 identical, frame 6 different
    for i in range(1, 6):
        pil_image.fromarray(noise_a).save(shots / f"frame_{i:04d}.png")
    pil_image.fromarray(noise_b).save(shots / "frame_0006.png")

    keyframes = keyframes_mod.extract_keyframes(tmp_path, tmp_path)
    # 3 start-pad frames + the changed frame; identical frames 4-5 deduped
    assert [k["source_frame"] for k in keyframes] == [
        "frame_0001.png", "frame_0002.png", "frame_0003.png", "frame_0006.png",
    ]


# ── pipeline orchestration ───────────────────────────────────────────


VISION_CFG = {"provider": "stub", "model": "gemma-4-27b-it"}


def _stub_stages(monkeypatch, tmp_path: Path, digests: list[dict]):
    """Stub the ffmpeg/SSIM/model stages with canned deterministic results."""
    keyframes = [{"index": i, "source_frame": f"frame_{i + 1:04d}.png",
                  "path": str(tmp_path / f"keyframe_{i:04d}.png"),
                  "forced": False, "timestamp": i / 2.0}
                 for i in range(len(digests))]

    monkeypatch.setattr(pipeline_mod, "extract_frames",
                        lambda video_path, session_dir, fps=2.0: 12)
    monkeypatch.setattr(pipeline_mod, "extract_keyframes",
                        lambda session_dir, output_dir=None, **kw: keyframes)
    monkeypatch.setattr(pipeline_mod, "transcribe",
                        lambda video_path, session_dir, **kw: None)

    seen = {}

    async def fake_digest(kfs, narration, cfg, output_dir, *, prompt_context=""):
        seen["keyframes"] = kfs
        seen["cfg"] = cfg
        seen["prompt_context"] = prompt_context
        return digests

    monkeypatch.setattr(pipeline_mod, "digest_keyframes", fake_digest)
    return seen


async def test_pipeline_produces_skill_draft(tmp_path: Path, monkeypatch):
    seen = _stub_stages(monkeypatch, tmp_path, _good_digests())
    video = tmp_path / "crm_contact_entry.mp4"
    video.write_bytes(b"\x00" * 64)

    result = await ingest_video_to_skill(
        str(video), vision_model_cfg=VISION_CFG,
        work_dir=tmp_path / "session", transcribe_audio=True,
    )

    assert result["status"] == "ok"
    assert result["session_dir"] == str(tmp_path / "session")
    assert seen["cfg"] == VISION_CFG
    assert seen["prompt_context"] == "Crm Contact Entry"  # title from filename

    skill = result["skill"]
    assert skill is not None
    lines = skill["content"].splitlines()
    assert lines[0] == "---"
    assert lines[1].startswith("name: ")
    assert "## Steps" in skill["content"]
    assert skill["step_count"] == 5
    assert skill["parameter_count"] == 2
    # Parameterized TYPE steps surfaced in the draft
    assert "**input_0**" in skill["content"]

    # Gate integration: real gate_actions ran on the funnel-mapped actions
    gate = result["quality_gate"]
    assert gate["overall"] in ("pass", "warn")
    assert {c["name"] for c in gate["checks"]} >= {
        "app_consistency", "action_distribution", "no_consecutive_dupes",
        "min_viable_steps", "parameter_sanity",
    }

    stats = result["digest_stats"]
    assert stats["frames_extracted"] == 12
    assert stats["keyframes"] == 5
    assert stats["actions_extracted"] == 5
    assert stats["parameters"] == 2


async def test_pipeline_rejects_gate_failure(tmp_path: Path, monkeypatch):
    _stub_stages(monkeypatch, tmp_path, _bad_digests())
    video = tmp_path / "noisy_recording.mp4"
    video.write_bytes(b"\x00" * 64)

    result = await ingest_video_to_skill(
        str(video), vision_model_cfg=VISION_CFG,
        work_dir=tmp_path / "session", transcribe_audio=False,
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "quality_gate"
    assert result["skill"] is None
    gate = result["quality_gate"]
    assert gate["overall"] == "fail"
    failed = {c["name"] for c in gate["checks"] if not c["passed"]}
    assert "no_consecutive_dupes" in failed
    assert "action_distribution" in failed


async def test_pipeline_refuses_text_only_model_unless_forced(tmp_path: Path, monkeypatch):
    _stub_stages(monkeypatch, tmp_path, _good_digests())
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00" * 64)
    text_only_cfg = {"provider": "stub", "model": "qwen3-32b-instruct"}

    refused = await ingest_video_to_skill(
        str(video), vision_model_cfg=text_only_cfg,
        work_dir=tmp_path / "s1", transcribe_audio=False,
    )
    assert refused["status"] == "error"
    assert "vision" in refused["error"].lower()
    assert refused["skill"] is None

    forced = await ingest_video_to_skill(
        str(video), vision_model_cfg=text_only_cfg,
        work_dir=tmp_path / "s2", transcribe_audio=False, force=True,
    )
    assert forced["status"] == "ok"


async def test_pipeline_missing_video_is_error(tmp_path: Path):
    result = await ingest_video_to_skill(
        str(tmp_path / "absent.mp4"), vision_model_cfg=VISION_CFG,
        work_dir=tmp_path / "session",
    )
    assert result["status"] == "error"
    assert "not found" in result["error"]


async def test_pipeline_never_persists_to_skills_auto(tmp_path: Path, monkeypatch):
    """The trust-policy invariant: drafts only, nothing written outside the session."""
    _stub_stages(monkeypatch, tmp_path, _good_digests())
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"\x00" * 64)
    work_dir = tmp_path / "session"

    result = await ingest_video_to_skill(
        str(video), vision_model_cfg=VISION_CFG,
        work_dir=work_dir, transcribe_audio=False,
    )
    assert result["status"] == "ok"
    # No SKILL.md landed anywhere — content only travels in the return value
    assert not list(tmp_path.rglob("*.md"))
    assert result["skill"]["content"].startswith("---")
