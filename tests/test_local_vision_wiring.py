"""#387 — the local model's vision capability is DETECTED everywhere, asserted nowhere.

The defect: llama.cpp `/props` said `modalities.vision=true`, `LlamaCppProvider.detect_vision`
set `supports_vision=True`, the daemon logged "Vision: enabled (multimodal)" — and three other
places still said no by literal: `provider_class_supports_vision("llama_cpp")` returned False,
the WebSocket gate returned False for the primary ("Phase 1"), and the `local` catalog row
carried `"vision": False`. So Beacon withheld the picture from the one model on the box that
could see it, and the builder would have raised had a block ever got through.

These tests pin the wiring end to end: one attribute (`supports_vision` on the provider
instance) answers the registry, the gate, the catalog and the request builder.
"""

from __future__ import annotations

import sys
import types

# Same circular-import shim the sibling web tests use.
if "prometheus.memory" not in sys.modules:  # pragma: no cover - import plumbing
    try:
        import prometheus.memory  # noqa: F401
    except Exception:
        sys.modules["prometheus.memory"] = types.ModuleType("prometheus.memory")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.messages import ConversationMessage, ImageBlock, TextBlock  # noqa: E402
from prometheus.providers.base import ApiMessageRequest  # noqa: E402
from prometheus.providers.llama_cpp import LlamaCppProvider  # noqa: E402
from prometheus.providers.ollama import OllamaProvider  # noqa: E402
from prometheus.providers.registry import (  # noqa: E402
    provider_class_supports_vision,
    provider_supports_vision,
)
from prometheus.providers.stub import UnsupportedContentBlock  # noqa: E402
from prometheus.router.model_router import ModelRouter, RouterConfig  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402

PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _image_turn() -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[
            TextBlock(text="what is in this screenshot?"),
            ImageBlock(media_type="image/png", data=PNG_B64, source_path="/cache/x.png"),
        ],
    )


def _request() -> ApiMessageRequest:
    return ApiMessageRequest(model="m", messages=[_image_turn()])


# ── the registry: capability is a property of the class, permission of the instance ──


def test_local_provider_classes_can_express_an_image():
    """Both local providers build through the shared OpenAI-shape builder, which
    carries `image_url` — the form llama-server's multimodal endpoint accepts
    (verified live: a 1×1 red PNG → "Red")."""
    assert provider_class_supports_vision("llama_cpp") is True
    assert provider_class_supports_vision("ollama") is True
    assert provider_class_supports_vision("nonesuch") is False


def test_instance_permission_is_the_detected_attribute():
    llama = LlamaCppProvider(base_url="http://x:8080")
    assert provider_supports_vision(llama) is False  # nothing detected yet
    llama.supports_vision = True  # what detect_vision sets on modalities.vision
    assert provider_supports_vision(llama) is True
    assert provider_supports_vision(None) is False
    assert provider_supports_vision(object()) is False


# ── the builders: the detected flag is threaded, not a literal ──


def test_llama_cpp_sends_the_image_when_vision_was_detected():
    p = LlamaCppProvider(base_url="http://x:8080")
    p.supports_vision = True
    payload = p._build_request_payload(_request())
    parts = payload["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in parts), parts
    url = next(part for part in parts if part.get("type") == "image_url")["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_llama_cpp_refuses_the_image_when_nothing_was_detected():
    """No mmproj → the builder raises, naming the block, rather than dropping the
    picture on the way to a text-only model. The gate refuses first in practice;
    this pins the second line of defence."""
    p = LlamaCppProvider(base_url="http://x:8080")
    assert p.supports_vision is False
    with pytest.raises(UnsupportedContentBlock):
        p._build_request_payload(_request())


@pytest.mark.asyncio
async def test_ollama_threads_the_same_flag(monkeypatch):
    """Ollama has no probe, so it stays refusing by default; an operator-set flag
    opens the same path. Captured at the builder call, so no HTTP is involved."""
    from prometheus.providers import ollama as ollama_mod

    seen: dict[str, object] = {}

    def _capture(request, *, allow_images=False):
        seen["allow_images"] = allow_images
        raise RuntimeError("stop here")

    monkeypatch.setattr(ollama_mod, "_build_openai_messages", _capture)
    p = OllamaProvider(base_url="http://x:11434")
    with pytest.raises(RuntimeError):
        async for _ in p._call_once(_request()):
            pass
    assert seen["allow_images"] is False
    p.supports_vision = True
    with pytest.raises(RuntimeError):
        async for _ in p._call_once(_request()):
            pass
    assert seen["allow_images"] is True


# ── the gate: the primary answers from its own detected capability ──


class _Router:
    def __init__(self) -> None:
        self.route_calls = 0

    def get_override_for_session(self, session_id):
        return None

    def route(self, message, context=None):  # pragma: no cover - must not run
        self.route_calls += 1
        raise AssertionError("the capability check must not route")


def _bridge(provider) -> WebSocketBridge:
    ctx = types.SimpleNamespace(model_router=_Router(), provider=provider)
    return WebSocketBridge(session_mgr=None, loop_context=ctx, config={})


def test_gate_opens_for_a_primary_that_detected_vision():
    p = LlamaCppProvider(base_url="http://x:8080")
    p.supports_vision = True
    assert _bridge(p)._turn_supports_vision("beacon:s1") is True


def test_gate_stays_shut_for_a_primary_that_did_not():
    assert _bridge(LlamaCppProvider(base_url="http://x:8080"))._turn_supports_vision("beacon:s1") is False
    assert _bridge(object())._turn_supports_vision("beacon:s1") is False


# ── the catalog: the local row reads the SAME object the gate reads ──


def _router() -> ModelRouter:
    return ModelRouter(
        RouterConfig(),
        primary_provider=object(),
        primary_adapter=object(),
        primary_model="qwen3.8-27b",
    )


def _client_with_primary(provider) -> TestClient:
    cfg = {"model": {"model": "qwen3.8-27b", "provider": "llama_cpp"}}
    client = TestClient(create_app(cfg, model_router=_router()))
    # The daemon's launcher hangs the bridge on app.state; the catalog reads the
    # primary through it, exactly as the upload gate does.
    client.app.state.ws_bridge = types.SimpleNamespace(
        loop_context=types.SimpleNamespace(provider=provider)
    )
    return client


def _local_row(client: TestClient) -> dict:
    body = client.get("/api/models").json()
    return next(m for m in body["models"] if m["key"] == "local")


def test_catalog_local_row_reports_the_detected_capability():
    p = LlamaCppProvider(base_url="http://x:8080")
    p.supports_vision = True
    assert _local_row(_client_with_primary(p))["vision"] is True


def test_catalog_local_row_is_false_when_nothing_was_detected():
    assert _local_row(_client_with_primary(LlamaCppProvider(base_url="http://x:8080")))["vision"] is False


def test_catalog_local_row_is_false_without_a_bridge():
    """A web-only boot has no loop context and therefore no provider to ask.
    Absence is not permission — and it is not an error either."""
    client = TestClient(create_app({"model": {"model": "m", "provider": "llama_cpp"}}, model_router=_router()))
    assert _local_row(client)["vision"] is False


def test_catalog_and_gate_agree_by_construction():
    """The property the fix exists for: flip the one attribute, and both surfaces
    move together. Before, the catalog said False regardless."""
    p = LlamaCppProvider(base_url="http://x:8080")
    for detected in (False, True):
        p.supports_vision = detected
        client = _client_with_primary(p)
        bridge = _bridge(p)
        assert _local_row(client)["vision"] is bridge._turn_supports_vision("beacon:s1") is detected
