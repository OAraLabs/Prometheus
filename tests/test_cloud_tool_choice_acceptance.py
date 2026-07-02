"""IGNITION Piece 2 — cloud tool_choice acceptance: the REAL providers' serialized requests.

Constructs the real ``AnthropicProvider`` / ``OpenAICompatProvider``, drives
``stream_message`` with each tool_choice value, and asserts the exact native
param in the captured request body at the HTTP boundary (the registered
``cloud_http.RecordingAsyncClient``). Not a spy on our code — the serialized
request that would go on the wire.

Mapping under test (our value → native param):
  anthropic:      auto → {"type":"auto"} · required → {"type":"any"} ·
                  {tool:X} → {"type":"tool","name":X}
  openai-compat:  auto → "auto" · required → "required" ·
                  {tool:X} → {"type":"function","function":{"name":X}}
  none (both):    engine sends NO tools → no tools/tool_choice keys at all
  anything else:  ValueError (rider-b fail-loud guard — never degrade, never drop)
"""

from __future__ import annotations

import asyncio
import secrets

import pytest

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.openai_compat import OpenAICompatProvider
from tests.support.cloud_http import RecordingAsyncClient

TOOLS = [
    {"name": "web_search", "description": "search", "input_schema": {"type": "object", "properties": {}}},
    {"name": "read_file", "description": "read", "input_schema": {"type": "object", "properties": {}}},
]


def _request(tool_choice: object = "auto", tools: list | None = TOOLS) -> ApiMessageRequest:
    return ApiMessageRequest(
        model="test-model",
        messages=[ConversationMessage.from_user_text("hi")],
        tools=list(tools) if tools else [],
        tool_choice=tool_choice,
    )


def _drive(provider, request) -> dict:
    """Run stream_message against the recording boundary; return the captured body."""
    RecordingAsyncClient.reset()

    async def _consume():
        async for _ in provider.stream_message(request):
            pass

    asyncio.run(_consume())
    assert RecordingAsyncClient.captured, "provider sent no request"
    return RecordingAsyncClient.captured[-1]["json"]


@pytest.fixture()
def _boundary(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", RecordingAsyncClient)


def _anthropic() -> AnthropicProvider:
    return AnthropicProvider(api_key=secrets.token_hex(8), model="test-model")


def _openai() -> OpenAICompatProvider:
    return OpenAICompatProvider(base_url="http://cloud.test/v1", api_key=secrets.token_hex(8), model="test-model")


@pytest.mark.acceptance(allow_doubles=["cloud_http.RecordingAsyncClient"])
def test_anthropic_native_mapping(_boundary):
    assert _drive(_anthropic(), _request("auto"))["tool_choice"] == {"type": "auto"}
    assert _drive(_anthropic(), _request("required"))["tool_choice"] == {"type": "any"}
    body = _drive(_anthropic(), _request({"tool": "web_search"}))
    assert body["tool_choice"] == {"type": "tool", "name": "web_search"}
    # none = engine sends no tools -> neither key present (native "no tools").
    body = _drive(_anthropic(), _request("none", tools=None))
    assert "tools" not in body and "tool_choice" not in body


@pytest.mark.acceptance(allow_doubles=["cloud_http.RecordingAsyncClient"])
def test_openai_compat_native_mapping(_boundary):
    assert _drive(_openai(), _request("auto"))["tool_choice"] == "auto"
    assert _drive(_openai(), _request("required"))["tool_choice"] == "required"
    body = _drive(_openai(), _request({"tool": "web_search"}))
    assert body["tool_choice"] == {"type": "function", "function": {"name": "web_search"}}
    body = _drive(_openai(), _request("none", tools=None))
    assert "tools" not in body and "tool_choice" not in body


@pytest.mark.acceptance(allow_doubles=["cloud_http.RecordingAsyncClient"])
def test_absent_tool_choice_serializes_like_todays_hardcode(_boundary):
    """Dormant-path (cloud): a request whose tool_choice is the default must
    serialize the tool_choice key exactly as the pre-IGNITION hardcode did."""
    assert _drive(_anthropic(), _request())["tool_choice"] == {"type": "auto"}
    assert _drive(_openai(), _request())["tool_choice"] == "auto"


@pytest.mark.acceptance(allow_doubles=["cloud_http.RecordingAsyncClient"])
def test_unmapped_value_raises_never_degrades(_boundary):
    """Rider-b guard: an unmapped tool_choice RAISES at the provider — it must
    never silently degrade to auto or drop the key (a forced turn quietly
    becoming a normal one is exactly the bug class this sprint kills)."""
    for provider in (_anthropic(), _openai()):
        with pytest.raises(ValueError, match="unmapped tool_choice"):
            _drive(provider, _request("definitely-not-a-directive"))
        # "none" WITH tools present is an engine-contract violation -> loud, not silent.
        with pytest.raises(ValueError, match="unmapped tool_choice"):
            _drive(provider, _request("none"))
