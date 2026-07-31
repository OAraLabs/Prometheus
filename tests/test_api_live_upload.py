"""Tests for POST /api/learning/live-upload (record-a-skill trace upload).

Follows the test_api_documents.py pattern: TestClient over create_app({}),
real multipart round-trip, SkillCreator injected with a MagicMock provider
(the persist path makes no model call). Step verification is off via
config so no provider is constructed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.learning.skill_creator import SkillCreator  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

CFG_NO_VERIFY = {"learning": {"live_recorder": {"verify_steps": False}}}


def _events() -> list[dict]:
    def elem(**kw):
        base = {"tagName": "INPUT", "type": "text", "cssSelector": "#f",
                "closestLabel": "Field", "value": ""}
        base.update(kw)
        return base

    url = "https://crm.example.com/deals/new"
    return [
        {"timestamp": 1000, "type": "navigation", "fromUrl": "https://crm.example.com/",
         "toUrl": url, "url": url, "pageTitle": "New Deal"},
        {"timestamp": 2000, "type": "input", "inputValue": "Acme renewal", "url": url,
         "pageTitle": "New Deal",
         "element": elem(id="title", name="title", cssSelector="#title",
                         closestLabel="Deal title", value="Acme renewal")},
        {"timestamp": 3000, "type": "input", "inputValue": "50000", "url": url,
         "pageTitle": "New Deal",
         "element": elem(id="amount", name="amount", cssSelector="#amount",
                         closestLabel="Amount", value="50000")},
        {"timestamp": 4000, "type": "click", "url": url, "pageTitle": "New Deal",
         "element": elem(tagName="BUTTON", cssSelector="button.create",
                         textContent="Create deal", type="")},
        {"timestamp": 4100, "type": "submit", "url": url, "pageTitle": "New Deal",
         "element": elem(tagName="FORM", cssSelector="form")},
    ]


def _metadata() -> dict:
    return {"extensionVersion": "1.1.0", "startUrl": "https://crm.example.com/deals/new",
            "duration": 4100, "eventCount": 5}


def _make_creator(tmp_path: Path) -> SkillCreator:
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir(exist_ok=True)
    return SkillCreator(MagicMock(), auto_dir=auto_dir)


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(CFG_NO_VERIFY, skill_creator=_make_creator(tmp_path))
    return TestClient(app), tmp_path


def _post_upload(client: TestClient, events=None, metadata=None, files=None):
    data = {
        "events": json.dumps(events if events is not None else _events()),
        "metadata": json.dumps(metadata if metadata is not None else _metadata()),
    }
    return client.post("/api/learning/live-upload", data=data, files=files or {})


def test_upload_creates_skill(client):
    tc, tmp_path = client
    resp = _post_upload(
        tc, files={"screenshot_0": ("0.jpg", b"\xff\xd8jpegbytes", "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "created"
    assert body["step_count"] >= 3
    assert body["quality_gate"]["overall"] in ("pass", "warn")

    skill_file = Path(body["skill_path"])
    assert skill_file.exists()
    assert skill_file.parent == tmp_path / "auto"
    content = skill_file.read_text()
    assert content.startswith("---")
    assert f"name: {body['skill_name']}" in content


def test_upload_rejects_below_gate(client):
    tc, tmp_path = client
    # A lone navigation fails two gate checks -> overall "fail" -> 422
    resp = _post_upload(tc, events=_events()[:1])
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["reason"] == "quality_gate"
    assert not list((tmp_path / "auto").glob("*.md"))


def test_upload_requires_events_and_metadata(client):
    tc, _ = client
    resp = tc.post("/api/learning/live-upload", data={"events": json.dumps(_events())})
    assert resp.status_code == 400


def test_upload_rejects_invalid_json(client):
    tc, _ = client
    resp = tc.post(
        "/api/learning/live-upload",
        data={"events": "not-json", "metadata": "{}"},
    )
    assert resp.status_code == 400


def test_upload_rejects_empty_events(client):
    tc, _ = client
    resp = _post_upload(tc, events=[])
    assert resp.status_code == 400


def test_upload_rejects_non_image_screenshot(client):
    tc, _ = client
    resp = _post_upload(
        tc, files={"screenshot_0": ("0.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 400
    assert "not an image" in resp.json()["error"]


def test_upload_disabled_via_config(tmp_path: Path):
    app = create_app(
        {"learning": {"live_recorder": {"enabled": False}}},
        skill_creator=_make_creator(tmp_path),
    )
    tc = TestClient(app)
    resp = _post_upload(tc)
    assert resp.status_code == 503


def test_upload_body_cap_allows_larger_than_default(client):
    """The live-upload path is exempt from the global 2MB REST cap."""
    tc, _ = client
    big = b"\xff\xd8" + b"x" * (1024 * 1024)  # ~1MB screenshot
    resp = _post_upload(
        tc,
        files={
            "screenshot_0": ("0.jpg", big, "image/jpeg"),
            "screenshot_1": ("1.jpg", big, "image/jpeg"),
            "screenshot_2": ("2.jpg", big, "image/jpeg"),
        },
    )
    # >3MB total body would 413 on any other route; here it processes fine
    assert resp.status_code == 200, resp.text


def test_oversized_screenshot_rejected(client):
    tc, _ = client
    resp = _post_upload(
        tc,
        files={"screenshot_0": ("0.jpg", b"\xff\xd8" + b"x" * (2 * 1024 * 1024 + 1), "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "too large" in resp.json()["error"]
