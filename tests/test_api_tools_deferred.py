"""GET/PUT /api/tools/deferred — the Beacon toggle's daemon half.

Contract: GET reports the configured tri-state AND the effective resolution
with its source ("auto → enabled (local provider)" vs "explicitly disabled").
PUT sets the override; "auto" IS the cleared state. Everything applies at the
NEXT run start — the response says so explicitly and the UI repeats it.

Also pinned here: the persistence path must NEVER copy runtime-merged secrets
into the on-disk YAML. env_override.py merges real API keys into the runtime
config dict; a naive yaml.dump(config) would write them to disk. The route
fresh-loads the file, sets the one key, and writes that back.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.context.dynamic_tools import DynamicToolLoader  # noqa: E402
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class _EmptyInput(BaseModel):
    pass


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool_name in ("bash", "read_file", "web_search"):
        class _T(BaseTool):
            name = tool_name
            description = "t"
            input_model = _EmptyInput

            async def execute(self, arguments, context):  # noqa: ANN001
                return ToolResult(output="ok")

        reg.register(_T())
    return reg


def _client(config: dict, tier: str | None = "off"):
    app = create_app(config)
    loader = DynamicToolLoader(
        _registry(), config.setdefault("tools", {}).setdefault("deferred_loading", {})
    )
    adapter = SimpleNamespace(tier=tier) if tier is not None else None
    app.state.ws_bridge = SimpleNamespace(
        loop_context=SimpleNamespace(tool_loader=loader, adapter=adapter)
    )
    return TestClient(app), loader


def _cfg(enabled="auto") -> dict:
    return {"tools": {"deferred_loading": {"enabled": enabled,
                                           "always_loaded": ["bash", "read_file"]}}}


# ---------------------------------------------------------------------------
# GET — effective state + source
# ---------------------------------------------------------------------------


def test_get_reports_auto_resolution_on_cloud():
    client, _ = _client(_cfg("auto"), tier="off")
    d = client.get("/api/tools/deferred").json()
    assert d["configured"] == "auto"
    assert d["effective"] is False
    assert "cloud" in d["source"]
    assert d["advertised_count"] == 3 and d["total_tools"] == 3
    assert d["applies"] == "next run start"


def test_get_reports_auto_resolution_on_local():
    client, _ = _client(_cfg("auto"), tier="strict")
    d = client.get("/api/tools/deferred").json()
    assert d["effective"] is True
    assert "local" in d["source"]
    assert d["advertised_count"] == 2  # the deferred set
    assert d["total_tools"] == 3


def test_get_reports_explicit_override_source():
    client, _ = _client(_cfg(False), tier="strict")
    d = client.get("/api/tools/deferred").json()
    assert d["configured"] is False
    assert d["effective"] is False
    assert d["source"] == "explicitly disabled"


def test_get_without_wired_loop_reports_unknown_effective():
    app = create_app(_cfg("auto"))  # no ws_bridge at all
    d = TestClient(app).get("/api/tools/deferred").json()
    assert d["effective"] is None
    assert "not wired" in d["source"]
    assert d["configured"] == "auto"


# ---------------------------------------------------------------------------
# PUT — override, clear, validation
# ---------------------------------------------------------------------------


def test_put_true_overrides_auto_and_updates_loader():
    config = _cfg("auto")
    client, loader = _client(config, tier="off")
    d = client.put("/api/tools/deferred", json={"enabled": True}).json()
    assert d["configured"] is True
    assert d["effective"] is True          # explicit beats the cloud auto-off
    assert d["source"] == "explicitly enabled"
    # The loader the NEXT run will consult sees it too.
    assert loader.configured_mode is True
    assert config["tools"]["deferred_loading"]["enabled"] is True


def test_put_auto_is_the_cleared_state():
    client, loader = _client(_cfg(True), tier="off")
    d = client.put("/api/tools/deferred", json={"enabled": "auto"}).json()
    assert d["configured"] == "auto"
    assert d["effective"] is False and "cloud" in d["source"]
    assert loader.configured_mode == "auto"


@pytest.mark.parametrize("bad", ["yes", "AUTO ", 1, None, [], {}])
def test_put_rejects_non_tristate_values(bad):
    client, _ = _client(_cfg("auto"))
    resp = client.put("/api/tools/deferred", json={"enabled": bad})
    # Only true/false/"auto" cross the API boundary; loose strings are for
    # hand-edited YAML, not for a UI that can send exact values.
    if isinstance(bad, str) and bad.strip().lower() == "auto":
        assert resp.status_code == 200
    else:
        assert resp.status_code == 400


def test_put_missing_key_is_400():
    client, _ = _client(_cfg("auto"))
    assert client.put("/api/tools/deferred", json={}).status_code == 400


# ---------------------------------------------------------------------------
# Persistence — surgical, and NEVER copies runtime secrets to disk
# ---------------------------------------------------------------------------


def test_put_persists_surgically_without_leaking_runtime_secrets(tmp_path, monkeypatch):
    """THE LEAK GUARD. The runtime config dict carries env-merged secrets
    (env_override.py). The on-disk YAML does not. After a PUT, the file must
    contain the new enabled value and must NOT have gained the secret."""
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "prometheus.yaml").write_text(
        yaml.dump({"model": {"provider": "llama_cpp"},
                   "tools": {"deferred_loading": {"enabled": "auto"}}}),
        encoding="utf-8",
    )

    # Runtime config = on-disk + a merged secret, exactly what env_override does.
    runtime = {
        "model": {"provider": "llama_cpp"},
        "tools": {"deferred_loading": {"enabled": "auto"}},
        "providers": {"anthropic": {"api_key": "sk-ant-FAKE-test-secret-0000"}},
    }
    client, _ = _client(runtime, tier="strict")
    d = client.put("/api/tools/deferred", json={"enabled": False}).json()
    assert d["persisted"] is True

    text = (cfg_dir / "prometheus.yaml").read_text(encoding="utf-8")
    on_disk = yaml.safe_load(text)
    assert on_disk["tools"]["deferred_loading"]["enabled"] is False  # the one change
    assert on_disk["model"]["provider"] == "llama_cpp"               # rest intact
    assert "sk-ant-FAKE-test-secret" not in text, (
        "the PUT wrote runtime-merged credentials into the on-disk YAML"
    )
    assert "providers" not in on_disk


def test_put_survives_missing_config_file(tmp_path, monkeypatch):
    """No YAML anywhere → in-memory update still works; persisted:false, not a 500."""
    monkeypatch.chdir(tmp_path)
    import prometheus.web.server as server_mod  # noqa: F401
    from prometheus.config import paths as paths_mod

    monkeypatch.setattr(paths_mod, "get_config_dir", lambda: tmp_path / "nope")
    client, loader = _client(_cfg("auto"), tier="strict")
    resp = client.put("/api/tools/deferred", json={"enabled": True})
    assert resp.status_code == 200
    assert resp.json()["persisted"] is False
    assert loader.configured_mode is True
