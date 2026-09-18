"""Tests for the Polish-sprint Beacon API additions (WS2).

Covers:
  - /api/activity/recent returns persisted signal-bus events
  - /api/memory/current returns USER.md + MEMORY.md content with usage
  - /api/skills/list returns auto-skill records with state + pinned + mtime
  - /api/skills/{name} returns SKILL.md content
  - /api/skills/{name}/pin POST / DELETE flips the pinned flag
  - WebSocketBridge maps SignalBus events to first-class WS event types
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# fastapi is an optional dep (web extra). Skip the whole module if missing
# so a bare install doesn't fail collection.
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Reroute every config-dir lookup through tmp_path.

    Modules that ``from prometheus.config.paths import get_config_dir`` bind
    the function reference at import time, so we have to monkeypatch each
    bound name individually. Add more lines here as new modules are touched.
    """
    monkeypatch.setattr(
        "prometheus.config.paths.get_config_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "prometheus.memory.hermes_memory_tool.get_config_dir",
        lambda: tmp_path,
    )
    return tmp_path


@pytest.fixture
def client(config_dir):
    """Build a FastAPI app + TestClient using a minimal config."""
    from prometheus.web.server import create_app

    config = {"model": {"model": "test-model", "provider": "test"}}
    app = create_app(config=config)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /api/activity/recent
# ---------------------------------------------------------------------------


class TestActivityRecent:
    def test_returns_empty_when_telemetry_unavailable(self, client, monkeypatch):
        # No telemetry handle wired → endpoint returns [].
        monkeypatch.setattr(
            "prometheus.telemetry.tracker.get_telemetry_handle",
            lambda: None,
        )
        resp = client.get("/api/activity/recent?limit=50")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_passes_through_telemetry_rows(self, client, monkeypatch):
        rows = [
            {
                "id": 1,
                "timestamp": "2026-05-23T12:34:56",
                "signal_type": "skill_created",
                "payload": {"skill_name": "build-pancakes"},
                "source_subsystem": "skill_creator",
            },
        ]
        fake_tel = MagicMock()
        fake_tel.signal_events_since.return_value = rows
        monkeypatch.setattr(
            "prometheus.telemetry.tracker.get_telemetry_handle",
            lambda: fake_tel,
        )
        resp = client.get("/api/activity/recent?limit=10")
        assert resp.status_code == 200
        assert resp.json() == rows
        # Capped to limit (sanity: caller can't ask for unlimited rows).
        fake_tel.signal_events_since.assert_called_once()
        kwargs = fake_tel.signal_events_since.call_args.kwargs
        assert kwargs["limit"] == 10

    def test_limit_capped_at_500(self, client, monkeypatch):
        fake_tel = MagicMock()
        fake_tel.signal_events_since.return_value = []
        monkeypatch.setattr(
            "prometheus.telemetry.tracker.get_telemetry_handle",
            lambda: fake_tel,
        )
        client.get("/api/activity/recent?limit=9999")
        assert fake_tel.signal_events_since.call_args.kwargs["limit"] == 500


# ---------------------------------------------------------------------------
# /api/memory/current
# ---------------------------------------------------------------------------


class TestMemoryCurrent:
    def test_returns_memory_and_user_sections(self, client, config_dir):
        # Pre-create the files the memory store expects.
        from prometheus.memory.hermes_memory_tool import (
            get_memory_store,
            get_user_store,
        )

        memory_store = get_memory_store()
        memory_store.add("first memory")
        memory_store.add("second memory")

        user_store = get_user_store()
        user_store.add("user is a software engineer")

        resp = client.get("/api/memory/current")
        assert resp.status_code == 200
        body = resp.json()

        # Memory section
        mem = body["memory"]
        assert mem["entry_count"] == 2
        assert mem["char_count"] > 0
        assert mem["char_limit"] > 0
        assert "first memory" in mem["content"]
        assert "second memory" in mem["content"]

        # User section
        user = body["user"]
        assert user["entry_count"] == 1
        assert "user is a software engineer" in user["content"]

    def test_returns_empty_when_no_entries(self, client, config_dir):
        # Memory files don't exist yet; store should return empty entries.
        resp = client.get("/api/memory/current")
        assert resp.status_code == 200
        body = resp.json()
        assert body["memory"]["entry_count"] == 0
        assert body["user"]["entry_count"] == 0


# ---------------------------------------------------------------------------
# /api/skills/list, /api/skills/{name}, /api/skills/{name}/pin
# ---------------------------------------------------------------------------


class TestSkillsApi:
    def _setup_skills_dir(self, config_dir, names: list[str]) -> Path:
        auto_dir = config_dir / "skills" / "auto"
        auto_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            (auto_dir / f"{n}.md").write_text(f"# {n}\n\nSkill content for {n}\n")
        return auto_dir

    def test_skills_list_returns_rows(self, client, config_dir):
        self._setup_skills_dir(config_dir, ["alpha", "beta"])
        resp = client.get("/api/skills/list")
        assert resp.status_code == 200
        rows = resp.json()
        names = sorted(r["name"] for r in rows)
        assert names == ["alpha", "beta"]
        for r in rows:
            assert "pinned" in r
            assert "state" in r
            assert "last_modified" in r

    def test_skills_list_empty_when_no_dir(self, client, config_dir):
        # Don't create the dir.
        resp = client.get("/api/skills/list")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_skill_content_returns_file_content(self, client, config_dir):
        self._setup_skills_dir(config_dir, ["alpha"])
        resp = client.get("/api/skills/alpha")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "alpha"
        assert "# alpha" in body["content"]

    def test_skill_content_404_when_missing(self, client, config_dir):
        self._setup_skills_dir(config_dir, ["alpha"])
        resp = client.get("/api/skills/nonexistent")
        assert resp.status_code == 404

    def test_skill_content_400_on_path_traversal(self, client, config_dir):
        # Names that survive URL normalisation but trip the guard.
        # `..foo` contains `..` substring; the FastAPI router will deliver
        # this literal to the handler.
        resp = client.get("/api/skills/..foo")
        assert resp.status_code == 400

    def test_pin_skill_flips_flag(self, client, config_dir):
        self._setup_skills_dir(config_dir, ["alpha"])

        # Initially not pinned.
        rows = client.get("/api/skills/list").json()
        alpha = next(r for r in rows if r["name"] == "alpha")
        assert alpha["pinned"] is False

        # Pin it.
        resp = client.post("/api/skills/alpha/pin")
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True

        # Verify via list.
        rows = client.get("/api/skills/list").json()
        alpha = next(r for r in rows if r["name"] == "alpha")
        assert alpha["pinned"] is True

        # Unpin.
        resp = client.delete("/api/skills/alpha/pin")
        assert resp.status_code == 200
        assert resp.json()["pinned"] is False


# ---------------------------------------------------------------------------
# WebSocketBridge — signal kind → first-class event type mapping
# ---------------------------------------------------------------------------


class TestWebSocketBridgeEventRouting:
    """``_on_signal`` should set ``event["type"]`` based on the signal kind
    so the frontend can route to dedicated UI panes without parsing
    payloads."""

    @pytest.mark.asyncio
    async def test_skill_created_signal_becomes_typed_event(self):
        from prometheus.web.ws_server import WebSocketBridge

        captured: list[dict] = []
        bridge = WebSocketBridge()

        async def fake_broadcast(event):
            captured.append(event)

        bridge.broadcast = fake_broadcast

        signal = MagicMock()
        signal.kind = "skill_created"
        signal.payload = {"skill_name": "build-pancakes"}
        signal.timestamp = 100.0
        signal.source = "skill_creator"

        await bridge._on_signal(signal)
        assert len(captured) == 1
        assert captured[0]["type"] == "skill_created"
        assert captured[0]["payload"] == {"skill_name": "build-pancakes"}

    @pytest.mark.asyncio
    async def test_memory_updated_signal_becomes_typed_event(self):
        from prometheus.web.ws_server import WebSocketBridge

        captured: list[dict] = []
        bridge = WebSocketBridge()
        bridge.broadcast = lambda event: captured.append(event) or self._noop()

        async def fake_broadcast(event):
            captured.append(event)

        bridge.broadcast = fake_broadcast

        signal = MagicMock()
        signal.kind = "memory_updated"
        signal.payload = {"target": "MEMORY.md", "operation": "append"}
        signal.timestamp = 100.0
        signal.source = "memory_tool"

        await bridge._on_signal(signal)
        assert captured[0]["type"] == "memory_updated"

    async def _noop(self):
        pass

    @pytest.mark.asyncio
    async def test_task_lifecycle_signals_become_typed_events(self):
        """task_completed/task_failed are promoted out of the generic sentinel_signal.

        This is the half that was missing (audit P9.7 / Beacon#128): the manager emitted these
        kinds and Beacon listened for exactly these type names, but with no entry in _on_signal's
        promotion chain the frame left as ``sentinel_signal`` with the kind nested in
        ``payload.kind`` — so Beacon's ``maybeNotifyTask`` gate matched nothing and the background
        -task notification + Mission Control ticker had never fired. Each repo's own tests passed
        while the feature did nothing, which is why the promotion itself needs pinning here.

        The assertion that does the work is ``type == kind`` AND that it is NOT the default
        ``sentinel_signal`` — the latter is what fails if this elif is removed, so the test pins the
        fix rather than merely exercising the path.
        """
        from prometheus.web.ws_server import WebSocketBridge

        for kind, status in (("task_completed", "completed"), ("task_failed", "failed")):
            captured: list[dict] = []
            bridge = WebSocketBridge()

            async def fake_broadcast(event):
                captured.append(event)

            bridge.broadcast = fake_broadcast

            signal = MagicMock()
            signal.kind = kind
            signal.payload = {"task_id": "abc123", "status": status, "description": "build the thing"}
            signal.timestamp = 100.0
            signal.source = "task_supervisor"

            await bridge._on_signal(signal)
            assert len(captured) == 1
            assert captured[0]["type"] == kind, f"{kind} must be promoted to a first-class type"
            assert captured[0]["type"] != "sentinel_signal", (
                f"{kind} left as the generic sentinel_signal — Beacon's gate on ev.type would miss it"
            )
            assert captured[0]["payload"]["task_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_unpromoted_signal_still_wraps_as_sentinel_signal(self):
        """The refusal direction: a kind with NO promotion entry still leaves nested, unchanged.

        Without this, a change that promoted EVERYTHING (event["type"] = signal.kind
        unconditionally) would pass the test above while breaking every consumer that legitimately
        reads sentinel_signal — the activity feed's categorize() buckets sentinel_*/dream_*/skill_*
        as 'signal'. One arbitrary unlisted kind pins that the default path is intact.
        """
        from prometheus.web.ws_server import WebSocketBridge

        captured: list[dict] = []
        bridge = WebSocketBridge()

        async def fake_broadcast(event):
            captured.append(event)

        bridge.broadcast = fake_broadcast

        signal = MagicMock()
        signal.kind = "idle_start"  # a real bus kind that is deliberately NOT promoted
        signal.payload = {"foo": "bar"}
        signal.timestamp = 100.0
        signal.source = "sentinel"

        await bridge._on_signal(signal)
        assert captured[0]["type"] == "sentinel_signal"
        assert captured[0]["payload"]["kind"] == "idle_start"
        assert captured[0]["payload"]["payload"] == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_every_coding_stream_kind_is_promoted(self):
        """Every kind the coding tailer can emit leaves as its own frame type, not sentinel_signal.

        The half that was missing (observed 2026-09-18, run coding:ca4c63d84): livestream.py emitted
        five kinds while the promotion named three, so 6 ``coding_tool`` frames and 1
        ``coding_acceptance`` frame arrived at Beacon wrapped as generic ``sentinel_signal`` with the
        real kind nested at ``payload.kind``. Nothing errored and nothing was dropped — every client
        gate keyed on ``ev.type`` simply matched nothing, exactly as #494 describes for
        task_completed/task_failed one screen below.

        Iterating CODING_FRAME_KINDS pins ws_server TO the emitter's declaration. It cannot, on its
        own, notice a kind deleted from that tuple — the loop would just get shorter. That direction
        is covered by ``test_coding_frame_kinds_matches_every_emit_site``, and the two together close
        producer -> wire.
        """
        from prometheus.coding.livestream import CODING_FRAME_KINDS
        from prometheus.web.ws_server import WebSocketBridge

        assert CODING_FRAME_KINDS, "the emitter must declare at least one kind"
        for kind in CODING_FRAME_KINDS:
            captured: list[dict] = []
            bridge = WebSocketBridge()

            async def fake_broadcast(event):
                captured.append(event)

            bridge.broadcast = fake_broadcast

            signal = MagicMock()
            signal.kind = kind
            # An OPAQUE payload on purpose: the promotion is a kind -> type mapping and passes the
            # payload through untouched. Asserting a realistic shape here would invent a contract
            # this function does not have.
            signal.payload = {"session_id": "coding:cafe1234", "opaque": 1}
            signal.timestamp = 100.0
            signal.source = "coding_mode"

            await bridge._on_signal(signal)
            assert len(captured) == 1, f"{kind} produced {len(captured)} frames"
            assert captured[0]["type"] == kind, f"{kind} must be promoted to a first-class type"
            assert captured[0]["type"] != "sentinel_signal", (
                f"{kind} left as generic sentinel_signal — every Beacon gate on ev.type would miss it"
            )
            assert captured[0]["payload"] == {"session_id": "coding:cafe1234", "opaque": 1}

    def test_coding_frame_kinds_matches_every_emit_site(self):
        """CODING_FRAME_KINDS is exactly the set of kinds livestream.py actually emits.

        This is the anchor. Without it the promotion test above is satisfiable by deleting a kind
        from the tuple: the loop shortens, every remaining kind still promotes, green. Parsing the
        module's own ``self._emit(...)`` call sites means the tuple cannot drift from the emitter in
        EITHER direction — a new emit that forgets to declare fails here, and a declaration whose
        emit was removed fails here too.

        AST, not grep: a literal in a comment or docstring must not count as an emit site.
        """
        import ast
        from pathlib import Path

        from prometheus.coding import livestream
        from prometheus.coding.livestream import CODING_FRAME_KINDS

        source = Path(livestream.__file__).read_text(encoding="utf-8")
        emitted: set[str] = set()
        non_literal = 0
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "_emit"):
                continue
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                emitted.add(node.args[0].value)
            else:
                non_literal += 1

        assert emitted, "found no self._emit(...) call sites — the AST walk is broken, not the code"
        assert non_literal == 0, (
            f"{non_literal} _emit call(s) pass a computed kind; this test can no longer see every "
            "emitted kind and must be replaced rather than relaxed"
        )
        assert emitted == set(CODING_FRAME_KINDS), (
            "CODING_FRAME_KINDS has drifted from livestream.py's emit sites.\n"
            f"  emitted but not declared: {sorted(emitted - set(CODING_FRAME_KINDS))}\n"
            f"  declared but not emitted: {sorted(set(CODING_FRAME_KINDS) - emitted)}\n"
            "A kind that is emitted but not declared ships as generic sentinel_signal and every "
            "client gate on ev.type silently misses it."
        )


# ---------------------------------------------------------------------------
# Static frontend mounted by the launcher
# ---------------------------------------------------------------------------


class TestStaticMount:
    def test_index_html_exists_in_package(self):
        from prometheus.web import launcher as _l
        package_static = Path(_l.__file__).resolve().parent / "static" / "index.html"
        assert package_static.exists(), "Beacon frontend index.html must ship with the package"
        content = package_static.read_text(encoding="utf-8")
        # Sanity: shipped file is the Beacon dashboard, not a stub.
        assert "Prometheus Beacon" in content
        assert "Activity Feed" in content
        assert "Memory" in content
        assert "Skills" in content
