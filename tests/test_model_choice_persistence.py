"""WP-X.7 — every sticky model choice is stored with the session and restored
at boot; the rules each surface and the restore follow.

The end-to-end pin (a real daemon, stopped and booted again) is
tests/test_model_choice_restart.py. These pin the parts:

* the store-backed persister: a picker key in, a row out; clearing deletes;
  an ephemeral session never gets a row;
* the cloud restore (runs early, before any gateway starts): a vetted key is
  restored, a retired one is reported ``not configured``, a missing
  credential ``no credential``, backend keys are left to the backend restore,
  ``router.overrides.enabled: false`` restores nothing, and a choice made
  this boot is never overwritten;
* the backend restore leaves cloud keys alone (they used to be skipped as
  "never probed");
* settling the skips: retired → WARNING + ``silent_failures`` row + stored row
  deleted; no credential → WARNING + row, stored row kept; a down box →
  kept, as before;
* the surfaces hand the router the picker key: REST, /claude and /qwen <model>,
  /<backend>; the OpenAI-compatible surface stores nothing; REST honours
  ``router.overrides.enabled`` like the chat commands.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.router.model_router import (  # noqa: E402
    RESTORE_NO_CREDENTIAL,
    RESTORE_NOT_CONFIGURED,
    ModelRouter,
    RouterConfig,
    restore_backend_overrides,
    restore_cloud_overrides,
    settle_restore,
    store_backed_persister,
)
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

DAEMON = Path(__file__).resolve().parents[1] / "src" / "prometheus" / "daemon.py"
_FAKE_KEY = "not" + "a" + "real" + "key"

CFG = {
    "model": {"model": "qwen3.8-27b", "provider": "llama_cpp", "base_url": "http://a:8080"},
    "backends": {"4090": {"provider": "llama_cpp", "base_url": "http://gpu-box:8080"}},
}


def _router(**cfg) -> ModelRouter:
    return ModelRouter(RouterConfig(**cfg), primary_provider=object(), primary_adapter=object(),
                       primary_model="qwen3.8-27b")


def _store(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


# ── the persister ────────────────────────────────────────────────────────────


def test_the_persister_stores_the_picker_key_and_clears_it(tmp_path):
    store = _store(tmp_path)
    persist = store_backed_persister(store)
    persist("beacon:1", "claude")
    persist("beacon:2", "qwen:qwen3.8-flash")
    assert store.all_session_backends() == {"beacon:1": "claude", "beacon:2": "qwen:qwen3.8-flash"}
    persist("beacon:1", None)
    assert store.all_session_backends() == {"beacon:2": "qwen:qwen3.8-flash"}


def test_an_ephemeral_session_never_gets_a_row(tmp_path):
    from prometheus.config.ephemeral import set_session_ephemeral

    store = _store(tmp_path)
    store.set_session_backend("telegram:7", "gpt")       # chosen before the chat went ephemeral
    set_session_ephemeral("telegram:7", True)
    router = _router()
    router.persist_override = store_backed_persister(store)
    router.set_override("telegram:7", {"provider": "anthropic", "model": "m"}, key="claude")
    assert router.get_override_for_session("telegram:7") is not None   # the choice still applies
    assert store.all_session_backends() == {}      # …nothing is remembered, and the old row is gone
    router.set_override("telegram:8", {"provider": "anthropic", "model": "m"}, key="claude")
    assert store.all_session_backends() == {"telegram:8": "claude"}


def test_a_router_wired_to_the_store_round_trips_a_restart(tmp_path, monkeypatch):
    """Two routers on one store stand in for two daemon processes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    store = _store(tmp_path)
    before = _router()
    before.persist_override = store_backed_persister(store)
    before.set_override("beacon:1", {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
                        key="claude")
    after = _router()
    restored, skipped = restore_cloud_overrides(after, store.all_session_backends(), CFG)
    assert (restored, skipped) == (1, [])
    entry = after.get_override_for_session("beacon:1")
    assert entry is not None and entry.provider_config["provider"] == "anthropic"
    assert entry.key == "claude"


# ── the cloud restore ────────────────────────────────────────────────────────


def test_cloud_restore_restores_vetted_keys_and_reports_the_rest(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("QWEN_API_KEY", _FAKE_KEY)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = _router()
    restored, skipped = restore_cloud_overrides(router, {
        "b:1": "claude",
        "b:2": "qwen:qwen3.8-flash",
        "b:3": "qwen:qwen-retired-long-ago",
        "b:4": "4090",                   # a backend: the late restore's business
        "b:5": "gpt",                    # no OPENAI_API_KEY
    }, CFG)
    assert restored == 2
    assert router.get_override_for_session("b:2").provider_config["model"] == "qwen3.8-flash"
    assert router.get_override_for_session("b:2").key == "qwen:qwen3.8-flash"
    assert {(s, why) for s, _k, why in skipped} == {
        ("b:3", RESTORE_NOT_CONFIGURED), ("b:5", RESTORE_NO_CREDENTIAL)}
    assert router.get_override_for_session("b:4") is None


def test_cloud_restore_restores_nothing_when_overrides_are_disabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    router = _router(overrides_enabled=False)
    restored, _ = restore_cloud_overrides(router, {"b:1": "claude"}, CFG)
    assert restored == 0 and router.get_override_for_session("b:1") is None


def test_a_restore_never_overwrites_a_choice_made_this_boot(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("QWEN_API_KEY", _FAKE_KEY)
    router = _router()
    router.set_override("b:1", {"provider": "qwen", "model": "qwen3.8-max"}, key="qwen")
    restore_cloud_overrides(router, {"b:1": "claude"}, CFG)
    assert router.get_override_for_session("b:1").provider_config["provider"] == "qwen"


def test_the_backend_restore_leaves_cloud_keys_alone():
    reg = SimpleNamespace(status=lambda name: None)
    router = _router()
    restored, skipped = restore_backend_overrides(router, reg, {"b:1": "claude", "b:2": "qwen:qwen3.8-flash"}, CFG)
    assert (restored, skipped) == (0, [])       # not "never probed": not its row at all


# ── settling what was skipped ────────────────────────────────────────────────


def test_settle_deletes_a_retired_choice_and_keeps_a_keyless_one(tmp_path, caplog):
    store = _store(tmp_path)
    store.set_session_backend("b:3", "qwen:qwen-retired-long-ago")
    store.set_session_backend("b:5", "gpt")
    store.set_session_backend("b:6", "4090")
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    with caplog.at_level(logging.INFO, logger="prometheus.router.model_router"):
        settle_restore([
            ("b:3", "qwen:qwen-retired-long-ago", RESTORE_NOT_CONFIGURED),
            ("b:5", "gpt", RESTORE_NO_CREDENTIAL),
            ("b:6", "4090", "connect timeout"),
        ], store, tel)
    assert store.all_session_backends() == {"b:5": "gpt", "b:6": "4090"}
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("b:3" in w and "qwen:qwen-retired-long-ago" in w for w in warnings)
    assert any("b:5" in w and "gpt" in w for w in warnings)
    assert not any("b:6" in w for w in warnings)       # a down box: as before, not a warning
    rows = tel.silent_failures_since(0, subsystem="model_choice")
    by_session = {json.loads(r["context"])["session_id"]: json.loads(r["context"]) for r in rows}
    assert set(by_session) == {"b:3", "b:5"}
    assert by_session["b:3"]["row_deleted"] is True and by_session["b:5"]["row_deleted"] is False


# ── the surfaces ─────────────────────────────────────────────────────────────


def _rest(router: ModelRouter) -> TestClient:
    return TestClient(create_app({"model": {"model": "qwen3.8-27b", "provider": "local"}},
                                 model_router=router))


def test_rest_hands_the_router_the_picker_key():
    saved: list = []
    router = _router()
    router.persist_override = lambda sid, key: saved.append((sid, key))
    c = _rest(router)
    assert c.post("/api/sessions/b:1/model", json={"key": "qwen:qwen3.8-flash"}).status_code == 200
    assert c.post("/api/sessions/b:1/model", json={"key": "local"}).status_code == 200
    assert c.delete("/api/sessions/b:2/model").status_code == 200
    assert saved == [("b:1", "qwen:qwen3.8-flash"), ("b:1", None), ("b:2", None)]


def test_rest_honours_overrides_disabled_like_the_chat_commands():
    router = _router(overrides_enabled=False)
    c = _rest(router)
    r = c.post("/api/sessions/b:1/model", json={"key": "claude"})
    assert r.status_code == 403 and "router.overrides.enabled" in r.json()["error"]
    assert router.get_override_for_session("b:1") is None
    # Going back to the default is never refused.
    assert c.post("/api/sessions/b:1/model", json={"key": "local"}).status_code == 200
    assert c.delete("/api/sessions/b:1/model").status_code == 200


class _Loop:
    def __init__(self, router):
        self._model_router = router


def test_chat_commands_hand_the_router_the_picker_key(monkeypatch):
    from prometheus.gateway.commands import cmd_provider_override

    monkeypatch.setenv("QWEN_API_KEY", _FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", _FAKE_KEY)
    saved: list = []
    router = _router()
    router.persist_override = lambda sid, key: saved.append((sid, key))
    assert cmd_provider_override(_Loop(router), {}, "telegram:1", "claude")[1]
    assert cmd_provider_override(_Loop(router), {}, "telegram:1", "qwen", model="qwen3.8-flash")[1]
    assert saved == [("telegram:1", "claude"), ("telegram:1", "qwen:qwen3.8-flash")]


@pytest.mark.asyncio
async def test_the_backend_command_hands_the_router_its_key(monkeypatch):
    from prometheus.gateway.commands import cmd_backend_override
    from prometheus.providers import backends as backends_mod
    from prometheus.providers.backends import BackendRegistry, BackendStatus

    async def _up(spec, timeout):  # noqa: ANN001
        return BackendStatus(name=spec.name, provider=spec.provider, base_url=spec.base_url,
                             ok=True, model="/models/served.gguf", n_ctx=32768, vision=False)

    reg = BackendRegistry.from_config(CFG, probe=_up)
    await reg.probe_all()
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    saved: list = []
    router = _router()
    router.persist_override = lambda sid, key: saved.append((sid, key))
    text, ok = await cmd_backend_override(_Loop(router), CFG, "telegram:1", "4090")
    assert ok, text
    assert saved == [("telegram:1", "4090")]


def test_the_openai_surface_stores_nothing(tmp_path, monkeypatch):
    """A /v1 request's choice lives for that one request; a row would outlive
    it (and a crash mid-request would leave one to restore)."""
    from prometheus.engine.agent_loop import LoopContext
    from prometheus.engine.messages import ConversationMessage, TextBlock
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
    from prometheus.tools.base import ToolRegistry

    class _Scripted(ModelProvider):
        async def stream_message(self, request):  # noqa: ANN001
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=[TextBlock(text="hi")]),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")

    store = _store(tmp_path)
    router = _router()
    router.persist_override = store_backed_persister(store)
    app = create_app({})
    app.state.ws_bridge = SimpleNamespace(loop_context=LoopContext(
        provider=_Scripted(), model="stub-local", system_prompt="P", max_tokens=16,
        tool_registry=ToolRegistry(), model_router=router,
    ))
    resp = TestClient(app).post("/v1/chat/completions", json={
        "model": "claude", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code in (200, 502), resp.text   # the cloud build may fail; storage is the point
    assert store.all_session_backends() == {}
    assert router._overrides == {}


# ── the daemon wiring ────────────────────────────────────────────────────────


def _run_daemon_body() -> ast.AsyncFunctionDef:
    for node in ast.walk(ast.parse(DAEMON.read_text())):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_daemon":
            return node
    raise AssertionError("run_daemon not found in daemon.py")


def _call_lines(fn: ast.AST, name: str) -> list[int]:
    return [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _adapter_start_lines(fn: ast.AST) -> list[int]:
    return [n.lineno for n in ast.walk(fn)
            if isinstance(n, ast.Await) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "start"
            and isinstance(n.value.func.value, ast.Name)
            and n.value.func.value.id in {"telegram", "slack_adapter", "discord_adapter"}]


def test_cloud_choices_are_restored_before_any_gateway_starts():
    """A Telegram update pending at boot is served the instant the adapter
    starts; a restore after that point serves it on the wrong model."""
    fn = _run_daemon_body()
    cloud = _call_lines(fn, "restore_cloud_overrides")
    persister = _call_lines(fn, "store_backed_persister")
    starts = _adapter_start_lines(fn)
    assert len(cloud) == 1 and len(persister) == 1 and starts
    assert persister[0] < cloud[0] < min(starts)


def test_backend_choices_are_still_restored_where_they_were():
    """Will's amendment (2026-09-30): backend rows keep their late restore."""
    fn = _run_daemon_body()
    backend = _call_lines(fn, "restore_backend_overrides")
    assert len(backend) == 1
    assert backend[0] > max(_adapter_start_lines(fn))
    assert len(_call_lines(fn, "settle_restore")) == 2
