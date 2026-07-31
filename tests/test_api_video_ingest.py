"""Tests for POST /api/learning/video-ingest (background draft producer)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.learning.skill_creator import SkillCreator  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

SKILL_CONTENT = """---
name: video-derived-workflow
description: A workflow inferred from a screen recording
---

# Video Derived Workflow

## Steps

1. Open the app
2. Fill the form
3. Save the record
"""


def _make_app(tmp_path: Path, *, enabled: bool = True, model: str = "gemma-4"):
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir(exist_ok=True)
    cfg = {
        "learning": {
            "live_recorder": {"verify_steps": False},
            "video_ingest": {
                "enabled": enabled,
                "vision_model": {"provider": "llama_cpp", "model": model} if model else {},
            },
        },
    }
    return create_app(cfg, skill_creator=SkillCreator(MagicMock(), auto_dir=auto_dir))


def test_video_ingest_disabled_by_default(tmp_path: Path):
    app = create_app(
        {}, skill_creator=SkillCreator(MagicMock(), auto_dir=tmp_path)
    )
    resp = TestClient(app).post("/api/learning/video-ingest", json={"source": "x.mp4"})
    assert resp.status_code == 503
    assert "disabled" in resp.json()["error"]


def test_video_ingest_requires_model(tmp_path: Path):
    app = _make_app(tmp_path, model="")
    resp = TestClient(app).post("/api/learning/video-ingest", json={"source": "x.mp4"})
    assert resp.status_code == 503
    assert "vision model" in resp.json()["error"]


def test_video_ingest_requires_source(tmp_path: Path):
    app = _make_app(tmp_path)
    resp = TestClient(app).post("/api/learning/video-ingest", json={})
    assert resp.status_code == 400


def test_video_ingest_files_draft_on_success(tmp_path: Path, monkeypatch):
    async def fake_pipeline(source, *, vision_model_cfg, force=False, **kwargs):
        return {
            "status": "ok",
            "skill": {"name": "video-derived-workflow", "title": "Video Derived Workflow",
                      "description": "d", "content": SKILL_CONTENT,
                      "step_count": 3, "parameter_count": 0},
            "quality_gate": {"overall": "pass"},
            "session_dir": str(tmp_path / "session"),
            "digest_stats": {"keyframes": 5},
        }

    import prometheus.learning.video_ingest.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "ingest_video_to_skill", fake_pipeline)

    app = _make_app(tmp_path)
    with TestClient(app) as tc:  # context manager: lifespan + loop for bg task
        resp = tc.post("/api/learning/video-ingest", json={"source": "demo.mp4"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

        # TestClient runs the app loop between requests; the background
        # task completes before the next request is served.
        drafts = tc.get("/api/learning/skill-drafts").json()["drafts"]

    assert len(drafts) == 1
    draft = drafts[0]
    assert draft["source"] == "video_ingestion"
    assert draft["name"] == "video-derived-workflow"
    assert draft["provenance"]["video_source"] == "demo.mp4"
    # Nothing auto-persisted: the auto dir stays empty until a human accepts
    assert not list((tmp_path / "auto").glob("*.md"))


def test_video_ingest_failure_files_no_draft(tmp_path: Path, monkeypatch):
    async def failing_pipeline(source, *, vision_model_cfg, force=False, **kwargs):
        return {"status": "error", "error": "boom", "skill": None,
                "quality_gate": None, "session_dir": "", "digest_stats": {}}

    import prometheus.learning.video_ingest.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "ingest_video_to_skill", failing_pipeline)

    app = _make_app(tmp_path)
    with TestClient(app) as tc:
        resp = tc.post("/api/learning/video-ingest", json={"source": "demo.mp4"})
        assert resp.status_code == 200
        drafts = tc.get("/api/learning/skill-drafts").json()["drafts"]
    assert drafts == []
