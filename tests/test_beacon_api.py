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
