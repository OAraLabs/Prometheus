"""Pack loader — FOUNDATION Part 2 acceptance.

Fixture packs drive every row of the 2.2 mismatch table (a pack that
declares pack_api 999, one that declares nothing, one that lies in its
provides block), and the two loop-level acceptance items are asserted by
DRIVING run_loop, not by reading a flag: an unpromoted pack skill's body
must never reach the model; after the human-ACCEPT path
(SkillCreator.persist_skill_content — the same validated write every
machine-generated skill uses) the same loop run carries it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from prometheus.learning.skill_drafts import SkillDraftStore
from prometheus.packs.loader import (
    PACK_API_CURRENT,
    get_pack_registry,
    load_packs,
    set_pack_registry,
)

SKILL_BODY_MARKER = "THE-PACK-SKILL-BODY-MARKER"

SKILL_MD = f"""---
name: pack-greeting
description: A greeting workflow shipped by a pack.
---

# Pack Greeting

{SKILL_BODY_MARKER}

## Steps
1. Greet.
"""


@pytest.fixture()
def packs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "packs"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _clear_registry_pin():
    yield
    set_pack_registry(None)


def _write_pack(
    packs_dir: Path,
    dirname: str,
    manifest: dict | str | None,
    skills: dict[str, str] | None = None,
    panels: dict[str, dict | str] | None = None,
    extra_dirs: tuple[str, ...] = (),
) -> Path:
    pack = packs_dir / dirname
    pack.mkdir()
    if manifest is not None:
        text = manifest if isinstance(manifest, str) else yaml.safe_dump(manifest)
        (pack / "pack.yaml").write_text(text)
    for stem, content in (skills or {}).items():
        (pack / "skills").mkdir(exist_ok=True)
        (pack / "skills" / f"{stem}.md").write_text(content)
    for stem, content in (panels or {}).items():
        (pack / "panels").mkdir(exist_ok=True)
        text = content if isinstance(content, str) else yaml.safe_dump(content)
        (pack / "panels" / f"{stem}.yaml").write_text(text)
    for d in extra_dirs:
        (pack / d).mkdir()
    return pack


def _manifest(**overrides) -> dict:
    base = {
        "name": "example-pack",
        "version": "1.0.0",
        "pack_api": PACK_API_CURRENT,
        "provides": {"skills": [], "panels": []},
    }
    base.update(overrides)
    return base


class TestMismatchTable:
    def test_honest_pack_loads(self, packs_dir: Path) -> None:
        _write_pack(
            packs_dir, "example-pack",
            _manifest(provides={"skills": ["pack-greeting"],
                                "panels": ["dash"]}),
            skills={"pack-greeting": SKILL_MD},
            panels={"dash": {"route": "/packs/dash", "title": "Dashboard",
                             "icon": "grid", "component": "dash.js"}},
        )
        registry = load_packs(packs_dir)
        assert [r.name for r in registry.loaded()] == ["example-pack"]
        record = registry.loaded()[0]
        assert record.pack_api == PACK_API_CURRENT
        assert record.skills == ["pack-greeting"]
        assert record.panels[0].route == "/packs/dash"
        assert record.panels[0].title == "Dashboard"

    def test_pack_api_999_refused_naming_both(self, packs_dir: Path) -> None:
        # Acceptance: refused at load, BOTH versions named in the error.
        _write_pack(packs_dir, "future-pack", _manifest(pack_api=999))
        registry = load_packs(packs_dir)
        assert registry.loaded() == []
        detail = registry.refused()[0].detail
        assert "999" in detail
        assert str(PACK_API_CURRENT) in detail

    def test_absent_pack_api_refused(self, packs_dir: Path) -> None:
        manifest = _manifest()
        del manifest["pack_api"]
        _write_pack(packs_dir, "undeclared-pack", manifest)
        registry = load_packs(packs_dir)
        assert "has not agreed" in registry.refused()[0].detail

    def test_lower_pack_api_refused_naming_both(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, "old-pack", _manifest(pack_api=0))
        detail = load_packs(packs_dir).refused()[0].detail
        assert "0" in detail and str(PACK_API_CURRENT) in detail

    def test_no_manifest_refused(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, "bare-dir", None)
        assert "no pack.yaml" in load_packs(packs_dir).refused()[0].detail

    def test_unparseable_manifest_refused(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, "garbled", ":{[ not yaml")
        assert "not a mapping" in load_packs(packs_dir).refused()[0].detail or \
            "unreadable" in load_packs(packs_dir).refused()[0].detail


class TestManifestIntegrity:
    def test_pack_claiming_unshipped_skill_is_refused(self, packs_dir: Path) -> None:
        # Acceptance: a fixture pack that lies.
        _write_pack(
            packs_dir, "liar",
            _manifest(provides={"skills": ["ghost-skill"], "panels": []}),
        )
        detail = load_packs(packs_dir).refused()[0].detail
        assert "ghost-skill" in detail and "does not ship" in detail

    def test_pack_shipping_undeclared_skill_is_refused(self, packs_dir: Path) -> None:
        # Absence in the manifest is not permission.
        _write_pack(
            packs_dir, "smuggler",
            _manifest(provides={"skills": [], "panels": []}),
            skills={"stowaway": SKILL_MD},
        )
        detail = load_packs(packs_dir).refused()[0].detail
        assert "stowaway" in detail and "does not declare" in detail

    def test_panel_without_route_or_title_is_refused(self, packs_dir: Path) -> None:
        _write_pack(
            packs_dir, "half-panel",
            _manifest(provides={"skills": [], "panels": ["p"]}),
            panels={"p": {"title": "No Route"}},
        )
        assert "route" in load_packs(packs_dir).refused()[0].detail

    def test_tools_dir_is_refused_by_name(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, "tooler", _manifest(), extra_dirs=("tools",))
        detail = load_packs(packs_dir).refused()[0].detail
        assert "MCP" in detail

    def test_one_bad_pack_does_not_block_a_good_one(self, packs_dir: Path) -> None:
        _write_pack(packs_dir, "future-pack", _manifest(
            name="future-pack", pack_api=999))
        _write_pack(packs_dir, "good-pack", _manifest(name="good-pack"))
        registry = load_packs(packs_dir)
        assert [r.name for r in registry.loaded()] == ["good-pack"]
        assert len(registry.refused()) == 1


class TestQuarantine:
    def test_pack_skills_land_as_drafts(self, packs_dir: Path, tmp_path: Path) -> None:
        store = SkillDraftStore(tmp_path / "drafts")
        _write_pack(
            packs_dir, "example-pack",
            _manifest(provides={"skills": ["pack-greeting"], "panels": []}),
            skills={"pack-greeting": SKILL_MD},
        )
        registry = load_packs(packs_dir, draft_store=store)
        drafts = store.list()
        assert len(drafts) == 1
        assert drafts[0]["source"] == "pack:example-pack"
        assert drafts[0]["name"] == "pack-greeting"
        assert registry.loaded()[0].quarantined == [drafts[0]["draft_id"]]

    def test_reboot_does_not_requarantine(self, packs_dir: Path, tmp_path: Path) -> None:
        # A human's accept/reject must stick: the same content is drafted
        # once, keyed by hash in packs/_ingested.json.
        store = SkillDraftStore(tmp_path / "drafts")
        _write_pack(
            packs_dir, "example-pack",
            _manifest(provides={"skills": ["pack-greeting"], "panels": []}),
            skills={"pack-greeting": SKILL_MD},
        )
        load_packs(packs_dir, draft_store=store)
        load_packs(packs_dir, draft_store=store)
        assert len(store.list()) == 1


class TestLoopReachability:
    """The two loop-driven acceptance items. The skill tool re-globs disk
    on every call, so directory membership IS the enforcement — and the
    proof is what the model gets to see, not a flag."""

    def _run_loop_asking_for_skill(self) -> str:
        from prometheus.engine.agent_loop import LoopContext, run_loop
        from prometheus.engine.messages import (
            ConversationMessage, TextBlock, ToolUseBlock,
        )
        from prometheus.engine.usage import UsageSnapshot
        from prometheus.providers.base import (
            ApiMessageCompleteEvent, ModelProvider,
        )
        from prometheus.tools.base import ToolRegistry
        from prometheus.tools.builtin.skill import SkillTool

        class _P(ModelProvider):
            def __init__(self) -> None:
                self.requests: list = []

            async def stream_message(self, request):  # noqa: ANN001
                self.requests.append(request)
                if len(self.requests) == 1:
                    content = [ToolUseBlock(
                        id="t1", name="skill",
                        input={"name": "pack-greeting"},
                    )]
                else:
                    content = [TextBlock(text="done")]
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(
                        role="assistant", content=content),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                    stop_reason="stop",
                )

        registry = ToolRegistry()
        registry.register(SkillTool())
        provider = _P()
        ctx = LoopContext(
            provider=provider,
            model="stub",
            system_prompt="",
            max_tokens=128,
            tool_registry=registry,
        )

        async def _drain() -> None:
            async for _ in run_loop(
                ctx, [ConversationMessage.from_user_text("go")]
            ):
                pass

        asyncio.run(_drain())
        assert len(provider.requests) >= 2, "no round followed the tool call"
        # Round 2's request carries the skill tool's RESULT — what the
        # model actually got to read.
        return str(provider.requests[1].messages)

    def test_unpromoted_pack_skill_is_not_reachable(
        self, packs_dir: Path
    ) -> None:
        store = SkillDraftStore()  # real default (isolated config dir)
        _write_pack(
            packs_dir, "example-pack",
            _manifest(provides={"skills": ["pack-greeting"], "panels": []}),
            skills={"pack-greeting": SKILL_MD},
        )
        load_packs(packs_dir, draft_store=store)
        seen_by_model = self._run_loop_asking_for_skill()
        assert SKILL_BODY_MARKER not in seen_by_model, (
            "a quarantined pack skill's body reached the model"
        )

    def test_accepted_pack_skill_is_used_in_a_live_loop(
        self, packs_dir: Path
    ) -> None:
        from prometheus.learning.skill_creator import SkillCreator

        store = SkillDraftStore()
        _write_pack(
            packs_dir, "example-pack",
            _manifest(provides={"skills": ["pack-greeting"], "panels": []}),
            skills={"pack-greeting": SKILL_MD},
        )
        load_packs(packs_dir, draft_store=store)
        draft = store.list()[0]
        content, _sidecar = store.get(draft["draft_id"])
        # The ACCEPT path: the same validated write every machine-generated
        # skill takes (frontmatter-name slugging, traversal confinement,
        # no-overwrite).
        creator = SkillCreator(MagicMock())
        written = asyncio.run(creator.persist_skill_content(
            content, trigger="pack-skill accept (test)",
        ))
        assert written is not None

        seen_by_model = self._run_loop_asking_for_skill()
        assert SKILL_BODY_MARKER in seen_by_model, (
            "an accepted pack skill did not reach the loop"
        )


class TestApiRoute:
    def test_api_packs_reports_wired_and_contents(self, packs_dir: Path) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from prometheus.web.server import create_app

        client = TestClient(create_app({}))
        body = client.get("/api/packs").json()
        assert body == {
            "wired": False, "pack_api": PACK_API_CURRENT, "packs": [],
        }

        _write_pack(packs_dir, "future-pack", _manifest(
            name="future-pack", pack_api=999))
        _write_pack(
            packs_dir, "example-pack",
            _manifest(provides={"skills": [], "panels": ["dash"]}),
            panels={"dash": {"route": "/packs/dash", "title": "Dash"}},
        )
        set_pack_registry(load_packs(packs_dir))
        body = client.get("/api/packs").json()
        assert body["wired"] is True
        states = {p["name"]: p["state"] for p in body["packs"]}
        assert states == {"example-pack": "loaded", "future-pack": "refused"}
        loaded = next(p for p in body["packs"] if p["state"] == "loaded")
        assert loaded["panels"][0]["route"] == "/packs/dash"


class TestDaemonWiring:
    def test_daemon_discovers_packs_before_the_registry_build(self) -> None:
        import prometheus.daemon as daemon_mod

        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        packs = source.index("load_packs(")
        registry = source.index("registry = build_tool_registry(")
        assert packs < registry
