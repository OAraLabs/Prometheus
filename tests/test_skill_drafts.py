"""Unit tests for SkillDraftStore (two-tier trust: human review before skills/auto/).

Model-free store: create/list/get round-trip, frontmatter parsing,
strict draft-id validation (path-traversal defense), and the
archive-never-delete lifecycle (.rejected/ and .accepted/).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.learning.skill_drafts import SkillDraftStore

DRAFT_CONTENT = """\
---
name: crm-create-deal
description: Create a new deal in the CRM
---

# CRM - Create Deal

## When to use
When you need to create a new deal record.

## Parameters
- `deal_title` (required)
  The title of the deal.
- `amount` (required)
  The deal amount.

## Steps
1. Navigate to https://crm.example.com/deals/new
2. Fill "Deal title" with `deal_title`
3. Fill "Amount" with `amount`
4. Click "Create deal"

## Notes
- Derived from video ingestion; verify selectors before first run.
"""

INVALID_IDS = [
    "../x",
    "draft-1-zzzz;rm",
    "draft-1-ZZZZ",
    "draft-1-abc",           # only 3 hex chars
    "draft-1-abcde",         # 5 hex chars
    "draft--abcd",           # missing unixtime
    "draft-1-abcd/../../x",  # traversal suffix
    "skill-1-abcd",
    "",
]


@pytest.fixture
def store(tmp_path: Path) -> SkillDraftStore:
    return SkillDraftStore(drafts_dir=tmp_path / "drafts")


def test_create_list_get_roundtrip(store: SkillDraftStore, tmp_path: Path):
    sidecar = store.create(
        DRAFT_CONTENT, source="video_ingestion", provenance={"video": "demo.mp4"}
    )

    draft_id = sidecar["draft_id"]
    assert sidecar["source"] == "video_ingestion"
    assert sidecar["name"] == "crm-create-deal"
    assert sidecar["title"] == "CRM - Create Deal"
    assert sidecar["description"] == "Create a new deal in the CRM"
    assert sidecar["step_count"] == 4
    assert sidecar["parameter_count"] == 2
    assert sidecar["provenance"] == {"video": "demo.mp4"}
    assert "created_at" in sidecar

    # Both files on disk under the drafts dir
    assert (tmp_path / "drafts" / f"{draft_id}.md").is_file()
    assert (tmp_path / "drafts" / f"{draft_id}.json").is_file()

    listed = store.list()
    assert [d["draft_id"] for d in listed] == [draft_id]

    content, got = store.get(draft_id)
    assert content == DRAFT_CONTENT
    assert got == sidecar


def test_draft_id_format(store: SkillDraftStore):
    import re

    sidecar = store.create(DRAFT_CONTENT, source="video_ingestion")
    assert re.fullmatch(r"draft-[0-9]+-[0-9a-f]{4}", sidecar["draft_id"])


def test_list_newest_first(store: SkillDraftStore):
    first = store.create(DRAFT_CONTENT, source="video_ingestion")
    second = store.create(DRAFT_CONTENT, source="video_ingestion")
    listed = store.list()
    assert [d["draft_id"] for d in listed] == [second["draft_id"], first["draft_id"]]


def test_counts_from_provenance_win(store: SkillDraftStore):
    sidecar = store.create(
        DRAFT_CONTENT,
        source="video_ingestion",
        provenance={"step_count": 9, "parameter_count": 3},
    )
    assert sidecar["step_count"] == 9
    assert sidecar["parameter_count"] == 3


def test_description_falls_back_to_first_body_line(store: SkillDraftStore):
    content = "---\nname: no-desc\n---\n\n# No Desc\n\nDoes a thing quickly.\n"
    sidecar = store.create(content, source="video_ingestion")
    assert sidecar["description"] == "Does a thing quickly."
    assert sidecar["title"] == "No Desc"


@pytest.mark.parametrize("bad_id", INVALID_IDS)
def test_invalid_id_rejected_everywhere(store: SkillDraftStore, bad_id: str):
    with pytest.raises(ValueError):
        store.get(bad_id)
    with pytest.raises(ValueError):
        store.reject(bad_id)
    with pytest.raises(ValueError):
        store.remove_accepted(bad_id)


def test_unknown_id_raises_keyerror(store: SkillDraftStore):
    with pytest.raises(KeyError):
        store.get("draft-1-abcd")
    with pytest.raises(KeyError):
        store.reject("draft-1-abcd")
    with pytest.raises(KeyError):
        store.remove_accepted("draft-1-abcd")


def test_reject_moves_to_rejected(store: SkillDraftStore, tmp_path: Path):
    draft_id = store.create(DRAFT_CONTENT, source="video_ingestion")["draft_id"]
    store.reject(draft_id)

    rejected = tmp_path / "drafts" / ".rejected"
    assert (rejected / f"{draft_id}.md").is_file()
    assert (rejected / f"{draft_id}.json").is_file()
    # Never delete: content preserved verbatim
    assert (rejected / f"{draft_id}.md").read_text() == DRAFT_CONTENT

    assert store.list() == []
    with pytest.raises(KeyError):
        store.get(draft_id)


def test_remove_accepted_audit_trail(store: SkillDraftStore, tmp_path: Path):
    draft_id = store.create(DRAFT_CONTENT, source="video_ingestion")["draft_id"]
    store.remove_accepted(draft_id)

    accepted = tmp_path / "drafts" / ".accepted"
    assert (accepted / f"{draft_id}.md").is_file()
    assert (accepted / f"{draft_id}.json").is_file()

    assert store.list() == []
    with pytest.raises(KeyError):
        store.get(draft_id)


def test_default_dir_uses_config_dir(monkeypatch, tmp_path: Path):
    """Default drafts dir is get_config_dir()/skills/drafts (env-repointable)."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    store = SkillDraftStore()
    sidecar = store.create(DRAFT_CONTENT, source="video_ingestion")
    expected = tmp_path / "cfg" / "skills" / "drafts"
    assert (expected / f"{sidecar['draft_id']}.md").is_file()
    assert (expected / f"{sidecar['draft_id']}.json").is_file()
