"""The OpenAI-compatible surface: GET /v1/models, POST /v1/chat/completions.

Item 2 of the 2026-09-01 roadmap. Driven through the real app factory, the
real ``run_loop`` and a scripted provider — the same rig the MCP live-loop
test uses — so what is asserted is the wire shape a client sees, not a
mock's opinion of it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.agent_loop import LoopContext  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider  # noqa: E402
from prometheus.tools.base import ToolRegistry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

TOKEN = "compat-test-token"


class _Scripted(ModelProvider):
    """Answers with one text turn; records every request it saw."""

    def __init__(self, reply: str = "hello from the loop") -> None:
        self.reply = reply
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self.reply)]),
            usage=UsageSnapshot(input_tokens=7, output_tokens=3),
            stop_reason="stop",
        )


def _rig(provider: ModelProvider | None = None, *, token: str | None = TOKEN):
    provider = provider or _Scripted()
    app = create_app({"web": {"api_token": token}} if token else {})
    app.state.ws_bridge = SimpleNamespace(loop_context=LoopContext(
        provider=provider, model="stub-local", system_prompt="BASE PROMPT",
        max_tokens=64, tool_registry=ToolRegistry(),
    ))
    return TestClient(app), provider


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _sse(text: str) -> list[dict | str]:
    out: list = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


class TestAuth:
    def test_v1_is_behind_the_same_bearer_as_api(self) -> None:
        """The middleware predicate was `/api/` only; a `/v1` route outside
        it would have shipped unauthenticated."""
        client, _ = _rig()
        assert client.get("/v1/models").status_code == 401
        assert client.post("/v1/chat/completions", json={}).status_code == 401
        assert client.get("/v1/models", headers=_auth()).status_code == 200

    def test_open_daemon_needs_no_bearer(self) -> None:
        client, _ = _rig(token=None)
        assert client.get("/v1/models").status_code == 200


class TestModels:
    def test_list_shape_and_local_default(self) -> None:
        client, _ = _rig()
        body = client.get("/v1/models", headers=_auth()).json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert "local" in ids
        local = next(m for m in body["data"] if m["id"] == "local")
        assert local["object"] == "model"
        assert isinstance(local["created"], int)
        # Only presets with a credential present are listed (local always is).
        assert all(m["id"] == "local" or m["owned_by"] for m in body["data"])


class TestChatCompletions:
    def test_non_streaming_shape(self) -> None:
        client, provider = _rig(_Scripted("hello from the loop"))
        resp = client.post("/v1/chat/completions", headers=_auth(), json={
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        assert body["model"] == "local"
        assert body["choices"] == [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello from the loop"},
            "finish_reason": "stop",
        }]
        assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        assert body["prometheus"]["session_id"].startswith("openai:")
        assert len(provider.requests) == 1

    def test_history_and_system_are_mapped(self) -> None:
        """The client's system text is APPENDED to the daemon's prompt, never
        a replacement; user/assistant history reaches the loop in order."""
        client, provider = _rig()
        resp = client.post("/v1/chat/completions", headers=_auth(), json={
            "messages": [
                {"role": "system", "content": "Answer in French."},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "un"},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
            ],
        })
        assert resp.status_code == 200, resp.text
        req = provider.requests[0]
        assert req.system_prompt.startswith("BASE PROMPT")
        assert "Answer in French." in req.system_prompt
        roles = [m.role for m in req.messages]
        assert roles[-3:] == ["user", "assistant", "user"]
        assert req.messages[-1].content[0].text == "two"

    def test_streaming_sse_shape(self) -> None:
        client, _ = _rig(_Scripted("streamed reply"))
        resp = client.post("/v1/chat/completions", headers=_auth(), json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = _sse(resp.text)
        assert frames[-1] == "[DONE]"
        chunks = [f for f in frames if isinstance(f, dict)]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == "streamed reply"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"]["total_tokens"] == 10

    def test_each_call_is_its_own_session(self) -> None:
        """Stateless like OpenAI: two calls, two sessions, nothing shared."""
        client, _ = _rig()
        a = client.post("/v1/chat/completions", headers=_auth(),
                        json={"messages": [{"role": "user", "content": "a"}]}).json()
        b = client.post("/v1/chat/completions", headers=_auth(),
                        json={"messages": [{"role": "user", "content": "b"}]}).json()
        assert a["prometheus"]["session_id"] != b["prometheus"]["session_id"]


class TestRefusals:
    def _post(self, client, **body):
        return client.post("/v1/chat/completions", headers=_auth(), json=body)

    def test_client_tools_are_refused_not_ignored(self) -> None:
        client, provider = _rig()
        resp = self._post(client, messages=[{"role": "user", "content": "x"}],
                          tools=[{"type": "function", "function": {"name": "f"}}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "tools_unsupported"
        assert provider.requests == []                      # nothing ran

    def test_last_message_must_be_user(self) -> None:
        client, _ = _rig()
        resp = self._post(client, messages=[{"role": "user", "content": "x"},
                                            {"role": "assistant", "content": "y"}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "last_message_not_user"

    def test_unknown_model_is_404_in_the_openai_envelope(self) -> None:
        client, _ = _rig()
        resp = self._post(client, model="gpt-9", messages=[{"role": "user", "content": "x"}])
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert err["code"] == "model_not_found" and err["type"] == "invalid_request_error"

    def test_image_parts_are_refused(self) -> None:
        client, _ = _rig()
        resp = self._post(client, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "unsupported_content"

    def test_empty_messages_and_bad_body(self) -> None:
        client, _ = _rig()
        assert self._post(client, messages=[]).json()["error"]["code"] == "messages_required"
        raw = client.post("/v1/chat/completions", headers=_auth(), content=b"nope")
        assert raw.status_code == 400 and raw.json()["error"]["code"] == "invalid_body"

    def test_loop_failure_is_a_502_with_the_reason(self) -> None:
        class _Boom(ModelProvider):
            async def stream_message(self, request):  # noqa: ANN001
                raise RuntimeError("provider exploded")
                yield  # pragma: no cover

        client, _ = _rig(_Boom())
        resp = self._post(client, messages=[{"role": "user", "content": "x"}])
        assert resp.status_code == 502
        assert resp.json()["error"]["type"] == "server_error"

    def test_no_loop_is_503(self) -> None:
        app = create_app({"web": {"api_token": TOKEN}})
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", headers=_auth(),
                           json={"messages": [{"role": "user", "content": "x"}]})
        assert resp.status_code == 503
