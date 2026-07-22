"""Tests for the /api/learning/skill-drafts routes (two-tier trust review).

Follows the test_api_live_upload.py pattern: TestClient over create_app
with verify_steps off and a SkillCreator injected with a MagicMock
provider (accept's persist path makes no model call). Drafts are seeded
by driving the store directly against the isolated config dir (the
conftest autouse fixture repoints PROMETHEUS_CONFIG_DIR, so the route's
lazily-built store and the test's store resolve to the same directory).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.paths import get_config_dir  # noqa: E402
from prometheus.learning.skill_creator import SkillCreator  # noqa: E402
from prometheus.learning.skill_drafts import SkillDraftStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

CFG_NO_VERIFY = {"learning": {"live_recorder": {"verify_steps": False}}}

DRAFT_CONTENT = """\
---
name: crm-create-deal
description: Create a new deal in the CRM
---

# CRM - Create Deal

## When to use
When you need to create a new deal record.

## Steps
1. Navigate to https://crm.example.com/deals/new
2. Fill "Deal title"
3. Fill "Amount"
4. Click "Create deal"
"""

EDITED_CONTENT = """\
---
name: crm-create-deal-reviewed
description: Create a new deal in the CRM (human-reviewed)
---

# CRM - Create Deal (Reviewed)

## Steps
1. Navigate to https://crm.example.com/deals/new
2. Fill "Deal title"
3. Click "Create deal"
"""

NO_NAME_CONTENT = """\
---
description: frontmatter without a name field
---

# Broken Draft

## Steps
1. Do a thing
"""


def _make_creator(tmp_path: Path) -> SkillCreator:
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir(exist_ok=True)
    return SkillCreator(MagicMock(), auto_dir=auto_dir)


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(CFG_NO_VERIFY, skill_creator=_make_creator(tmp_path))
    return TestClient(app), tmp_path


def _drafts_dir() -> Path:
    return get_config_dir() / "skills" / "drafts"


def _seed(content: str = DRAFT_CONTENT, source: str = "video_ingestion") -> dict:
    """Seed a draft directly through the store in the isolated config dir."""
    return SkillDraftStore().create(
        content, source=source, provenance={"video": "demo.mp4"}
    )


def test_list_empty(client):
    tc, _ = client
    resp = tc.get("/api/learning/skill-drafts")
    assert resp.status_code == 200
    assert resp.json() == {"drafts": []}


def test_list_and_get(client):
    tc, _ = client
    sidecar = _seed()
    draft_id = sidecar["draft_id"]

    resp = tc.get("/api/learning/skill-drafts")
    assert resp.status_code == 200
    drafts = resp.json()["drafts"]
    assert [d["draft_id"] for d in drafts] == [draft_id]
    assert drafts[0]["title"] == "CRM - Create Deal"

    resp = tc.get(f"/api/learning/skill-drafts/{draft_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["draft"]["draft_id"] == draft_id
    assert body["draft"]["source"] == "video_ingestion"
    assert body["content"] == DRAFT_CONTENT


def test_get_unknown_404(client):
    tc, _ = client
    resp = tc.get("/api/learning/skill-drafts/draft-1-abcd")
    assert resp.status_code == 404
    assert "unknown draft" in resp.json()["error"]


@pytest.mark.parametrize("bad_id", ["draft-1-zzzz;rm", "draft-1-ZZZZ", "not-a-draft"])
def test_invalid_id_400(client, bad_id):
    tc, _ = client
    resp = tc.get(f"/api/learning/skill-drafts/{bad_id}")
    assert resp.status_code == 400
    assert "invalid draft id" in resp.json()["error"]

    resp = tc.post(f"/api/learning/skill-drafts/{bad_id}/accept")
    assert resp.status_code == 400

    resp = tc.post(f"/api/learning/skill-drafts/{bad_id}/reject")
    assert resp.status_code == 400


def test_accept_persists_skill(client):
    tc, tmp_path = client
    draft_id = _seed()["draft_id"]

    resp = tc.post(f"/api/learning/skill-drafts/{draft_id}/accept")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["skill_name"] == "crm-create-deal"

    # Persisted through SkillCreator into the injected auto dir, same slug
    skill_file = Path(body["skill_path"])
    assert skill_file.exists()
    assert skill_file.parent == tmp_path / "auto"
    assert skill_file.name == "crm-create-deal.md"
    assert "name: crm-create-deal" in skill_file.read_text()

    # Draft left the pending listing
    resp = tc.get("/api/learning/skill-drafts")
    assert resp.json()["drafts"] == []
    assert tc.get(f"/api/learning/skill-drafts/{draft_id}").status_code == 404

    # Audit trail: both files archived under .accepted/, never deleted
    accepted = _drafts_dir() / ".accepted"
    assert (accepted / f"{draft_id}.md").is_file()
    assert (accepted / f"{draft_id}.json").is_file()


def test_accept_with_edited_content_persists_the_edit(client):
    tc, tmp_path = client
    draft_id = _seed()["draft_id"]

    resp = tc.post(
        f"/api/learning/skill-drafts/{draft_id}/accept",
        json={"content": EDITED_CONTENT},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skill_name"] == "crm-create-deal-reviewed"

    skill_file = tmp_path / "auto" / "crm-create-deal-reviewed.md"
    assert skill_file.exists()
    # persist_skill_content writes content.strip() + "\n"
    assert skill_file.read_text() == EDITED_CONTENT.strip() + "\n"


def test_accept_with_empty_edited_content_400(client):
    tc, _ = client
    draft_id = _seed()["draft_id"]

    resp = tc.post(
        f"/api/learning/skill-drafts/{draft_id}/accept", json={"content": "   "}
    )
    assert resp.status_code == 400

    # Draft untouched
    resp = tc.get("/api/learning/skill-drafts")
    assert [d["draft_id"] for d in resp.json()["drafts"]] == [draft_id]


def test_accept_broken_frontmatter_422_and_draft_kept(client):
    tc, tmp_path = client
    draft_id = _seed(content=NO_NAME_CONTENT)["draft_id"]

    resp = tc.post(f"/api/learning/skill-drafts/{draft_id}/accept")
    assert resp.status_code == 422
    assert "frontmatter" in resp.json()["error"]

    # Nothing persisted; draft still pending for the reviewer to fix
    assert not list((tmp_path / "auto").glob("*.md"))
    resp = tc.get("/api/learning/skill-drafts")
    assert [d["draft_id"] for d in resp.json()["drafts"]] == [draft_id]


def test_accept_unknown_404(client):
    tc, _ = client
    resp = tc.post("/api/learning/skill-drafts/draft-1-abcd/accept")
    assert resp.status_code == 404


def test_reject(client):
    tc, tmp_path = client
    draft_id = _seed()["draft_id"]

    resp = tc.post(f"/api/learning/skill-drafts/{draft_id}/reject")
    assert resp.status_code == 200
    assert resp.json() == {"status": "rejected", "draft_id": draft_id}

    # Archived, never deleted
    rejected = _drafts_dir() / ".rejected"
    assert (rejected / f"{draft_id}.md").is_file()
    assert (rejected / f"{draft_id}.json").is_file()

    # Gone from the pending listing; nothing entered skills/auto/
    resp = tc.get("/api/learning/skill-drafts")
    assert resp.json()["drafts"] == []
    assert not list((tmp_path / "auto").glob("*.md"))


def test_reject_unknown_404(client):
    tc, _ = client
    resp = tc.post("/api/learning/skill-drafts/draft-1-abcd/reject")
    assert resp.status_code == 404
