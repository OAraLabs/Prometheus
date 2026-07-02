"""TRIPWIRE Piece 2 — the real-entry-point acceptance base.

Builds the REAL FastAPI app through the production pieces: ``create_app``
(real routes + real bearer-auth middleware, token generated at runtime and
injected via config — never a literal), a real ``SessionManager``, a real
``ModelRouter`` with a real ``RouterConfig``, a real tool registry
(``create_tool_registry``), a real ``LoopContext``, and the REAL
``WebSocketBridge`` wired onto ``app.state.ws_bridge`` exactly like
``web/launcher.py`` does. A POST to ``/api/chat/send`` therefore runs the
production frames end-to-end: REST route → bridge.dispatch_user_message →
_handle_send_message → asyncio.create_task(_run_agent) → run_loop → router →
provider.

The ONLY substitution is the model-provider boundary: ``RecordingProvider``
(ONE registered double, ``real_app.recording_provider``) captures every
outbound ``ApiMessageRequest`` and returns a canned completion. That is the
class of assertion #74 demanded: what actually LEFT the daemon toward the
provider, per turn — not which internal function was called.

Because the recorder is a registered double, acceptance tests using this base
MUST declare it: ``@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])``
— the allowance stays visible and per-test.

The anthropic-override path needs NO production seam: ``set_override`` stores
config only, and ``_route_override`` builds its provider lazily ONLY
``if entry.provider is None`` (model_router.py:544). ``set_override_recorder``
sets the override through the real REST endpoint, then pre-warms that
documented cache slot with a recorder — provider substitution at exactly the
permitted boundary, zero patching of production modules.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from prometheus.__main__ import create_tool_registry
from prometheus.engine.agent_loop import LoopContext
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.session import SessionManager
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.router.model_router import ModelRouter, RouterConfig
from prometheus.web.server import create_app
from prometheus.web.ws_server import WebSocketBridge
from tests.support.doubles import register_double

#: The one boundary double this base introduces — name it in allow_doubles.
BOUNDARY_DOUBLE = "real_app.recording_provider"


@register_double(
    BOUNDARY_DOUBLE,
    replaces="the model-provider HTTP boundary (LlamaCppProvider / AnthropicProvider)",
)
class RecordingProvider(ModelProvider):
    """Records every outbound request verbatim; replies with canned text."""

    def __init__(self, label: str, reply: str | None = None) -> None:
        self.label = label
        self.reply = reply or f"[{label}] acknowledged."
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        msg = ConversationMessage(role="assistant", content=[TextBlock(text=self.reply)])
        yield ApiTextDeltaEvent(text=self.reply)
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


@dataclass
class RealAppHarness:
    client: TestClient
    app: Any
    router: ModelRouter
    session_mgr: SessionManager
    loop_context: LoopContext
    primary: RecordingProvider
    token: str
    agent_state: dict
    recorders: list[RecordingProvider] = field(default_factory=list)

    # ── plumbing ──────────────────────────────────────────────────────────
    def auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _total_requests(self) -> int:
        return sum(len(r.requests) for r in self.recorders)

    # ── the entry point ───────────────────────────────────────────────────
    def send_turn(
        self,
        session_id: str,
        message: str,
        *,
        mode: str | None = None,
        extra_body: dict | None = None,
        timeout: float = 15.0,
    ) -> None:
        """POST /api/chat/send (the real REST entry) and wait for the turn.

        Completion = the boundary saw one more outbound request AND the
        bridge's agent_state went back to idle (set in _run_agent's finally).
        """
        before = self._total_requests()
        body: dict = {"session_id": session_id, "message": message}
        if mode is not None:
            body["mode"] = mode
        if extra_body:
            body.update(extra_body)
        resp = self.client.post("/api/chat/send", json=body, headers=self.auth())
        assert resp.status_code == 200, f"send failed: {resp.status_code} {resp.text}"

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._total_requests() > before and self.agent_state.get("state") == "idle":
                return
            time.sleep(0.02)
        raise AssertionError(
            f"turn did not complete within {timeout}s "
            f"(outbound {self._total_requests()} vs {before}, state={self.agent_state.get('state')!r})"
        )

    # ── the override boundary ─────────────────────────────────────────────
    def set_override_recorder(self, session_id: str, key: str = "claude") -> RecordingProvider:
        """Set a per-session override via the REAL REST endpoint, then pre-warm
        the override entry's documented lazy provider cache with a recorder."""
        resp = self.client.post(
            f"/api/sessions/{session_id}/model", json={"key": key}, headers=self.auth()
        )
        assert resp.status_code == 200, f"override set failed: {resp.status_code} {resp.text}"
        recorder = RecordingProvider(label=f"override:{key}:{session_id}")
        entry = self.router._overrides[session_id]
        entry.provider = recorder  # the lazy slot _route_override would fill
        entry.adapter = self.loop_context.adapter
        self.recorders.append(recorder)
        return recorder

    def get_session_model(self, session_id: str) -> dict:
        resp = self.client.get(f"/api/sessions/{session_id}/model", headers=self.auth())
        assert resp.status_code == 200, resp.text
        return resp.json()


def build_real_app() -> RealAppHarness:
    """Assemble the production wiring (mirrors daemon.py:1189 + launcher.py:60-101)."""
    token = secrets.token_hex(16)  # runtime-generated; injected via config, never a literal
    primary = RecordingProvider(label="primary:local")

    router = ModelRouter(
        RouterConfig(),
        primary_provider=primary,
        primary_adapter=None,
        primary_model="local-test-model",
    )
    tool_registry = create_tool_registry({})
    loop_context = LoopContext(
        provider=primary,
        model="local-test-model",
        system_prompt="You are a test harness assistant.",
        max_tokens=256,
        tool_registry=tool_registry,
        model_router=router,
        session_id="web",  # same namespace the daemon uses for the web bridge
    )

    config: dict = {"web": {"api_token": token}}
    app = create_app(config, session_mgr=None, model_router=router)

    session_mgr = SessionManager()
    agent_state = {"state": "idle"}
    bridge = WebSocketBridge(
        session_mgr=session_mgr,
        loop_context=loop_context,
        agent_state_ref=agent_state,
        api_token=token,
        config=config,
    )
    app.state.ws_bridge = bridge
    app.state.agent_state_ref = agent_state

    client = TestClient(app)
    harness = RealAppHarness(
        client=client,
        app=app,
        router=router,
        session_mgr=session_mgr,
        loop_context=loop_context,
        primary=primary,
        token=token,
        agent_state=agent_state,
        recorders=[primary],
    )
    return harness
