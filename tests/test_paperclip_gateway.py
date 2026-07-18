"""Paperclip gateway — heartbeat protocol driven against a scripted Paperclip API.

Covers the wire-visible contract from the Paperclip agent-developer docs:
identity -> resolve issue -> checkout (409 = never retry) -> agent turn ->
status PATCH / comment -> cost event. The Paperclip server is a scripted
httpx fake (repo pattern, see test_image_dashscope.py); the agent turn is a
recording bridge double.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("httpx")

from prometheus.gateway import paperclip as pc  # noqa: E402
from prometheus.gateway.paperclip import PaperclipGateway, WakeEvent  # noqa: E402
from tests.support.doubles import register_double  # noqa: E402


# --------------------------------------------------------------------------- #
# Scripted Paperclip API fake
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, json_data: Any = None, status_code: int = 200) -> None:
        self._json = json_data if json_data is not None else {}
        self.status_code = status_code

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@register_double(
    "paperclip._FakePaperclipAPI", replaces="httpx.AsyncClient (Paperclip server)"
)
class _FakePaperclipAPI:
    """Records every call and answers from a canned route table."""

    def __init__(self, routes: dict[tuple[str, str], _FakeResponse]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def client_factory(self, **kwargs: Any) -> "_FakePaperclipAPI":
        self.client_kwargs = kwargs
        return self

    async def __aenter__(self) -> "_FakePaperclipAPI":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def _dispatch(self, method: str, url: str, kwargs: dict) -> _FakeResponse:
        self.calls.append((method, url, kwargs))
        try:
            return self.routes[(method, url)]
        except KeyError:  # fail loud — an unscripted call is a contract drift
            raise AssertionError(f"unscripted Paperclip call: {method} {url}")

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        return self._dispatch("GET", url, kwargs)

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        return self._dispatch("POST", url, kwargs)

    async def patch(self, url: str, **kwargs: Any) -> _FakeResponse:
        return self._dispatch("PATCH", url, kwargs)

    def called(self, method: str, url: str) -> list[dict]:
        return [kw for m, u, kw in self.calls if m == method and u == url]


class _Usage:
    input_tokens = 1500
    output_tokens = 300


@register_double(
    "paperclip._RecordingBridge", replaces="prometheus.web.ws_server.WebSocketBridge"
)
class _RecordingBridge:
    def __init__(self, reply: str = "Did the thing.\nSTATUS: done", usage: Any = _Usage(),
                 error: Exception | None = None) -> None:
        self.reply = reply
        self.usage = usage
        self.error = error
        self.turns: list[tuple[str, str]] = []

    async def run_turn_awaited(self, session_id: str, content: str) -> tuple[str, Any]:
        self.turns.append((session_id, content))
        if self.error is not None:
            raise self.error
        return self.reply, self.usage


_CFG = {"enabled": True, "api_url": "http://pc.test:3100", "api_key": "pc-key"}
_DAEMON_CFG = {"model": {"provider": "llama_cpp", "model": "qwen-local"}}


def _routes(overrides: dict[tuple[str, str], _FakeResponse] | None = None) -> dict[tuple[str, str], _FakeResponse]:
    base = {
        ("GET", "/api/agents/me"): _FakeResponse({"id": "agent-1", "companyId": "co-1"}),
        ("POST", "/api/issues/i-1/checkout"): _FakeResponse({}),
        ("GET", "/api/issues/i-1"): _FakeResponse(
            {"id": "i-1", "title": "Ship the widget", "description": "Make it so",
             "status": "todo", "priority": "high"}
        ),
        ("GET", "/api/issues/i-1/comments"): _FakeResponse(
            [{"authorName": "CEO", "body": "please hurry"}]
        ),
        ("PATCH", "/api/issues/i-1"): _FakeResponse({}),
        ("POST", "/api/companies/co-1/cost-events"): _FakeResponse({}),
        ("POST", "/api/issues/i-1/comments"): _FakeResponse({}),
    }
    base.update(overrides or {})
    return base


def _gateway(monkeypatch, api: _FakePaperclipAPI, bridge=None) -> PaperclipGateway:
    gw = PaperclipGateway(_CFG, bridge or _RecordingBridge(), daemon_config=_DAEMON_CFG)
    monkeypatch.setattr(pc.httpx, "AsyncClient", api.client_factory)
    return gw


def _wake(**kw: Any) -> WakeEvent:
    defaults = dict(run_id="r-1", agent_id="agent-1", company_id="co-1", issue_id="i-1")
    defaults.update(kw)
    return WakeEvent(**defaults)


# --------------------------------------------------------------------------- #
# parse_wake / parse_status units
# --------------------------------------------------------------------------- #


def test_parse_wake_extracts_issue_and_reason():
    gw = PaperclipGateway(_CFG, _RecordingBridge())
    wake = gw.parse_wake(
        {"runId": "r-9", "agentId": "a-9", "companyId": "co-9",
         "context": {"taskId": "t-9", "wakeReason": "issue_assigned", "commentId": "c-9"}}
    )
    assert (wake.run_id, wake.agent_id, wake.company_id) == ("r-9", "a-9", "co-9")
    assert (wake.issue_id, wake.wake_reason, wake.comment_id) == ("t-9", "issue_assigned", "c-9")


@pytest.mark.parametrize(
    "body", [{}, {"runId": "r"}, {"agentId": "a"}, {"runId": " ", "agentId": "a"}]
)
def test_parse_wake_rejects_missing_ids(body):
    gw = PaperclipGateway(_CFG, _RecordingBridge())
    with pytest.raises(ValueError):
        gw.parse_wake(body)


def test_parse_status_marker_stripped_and_lowercased():
    status, comment = PaperclipGateway.parse_status("All done here.\nstatus: DONE")
    assert status == "done"
    assert comment == "All done here."


def test_parse_status_defaults_to_in_progress():
    status, comment = PaperclipGateway.parse_status("half way there")
    assert status == "in_progress"
    assert comment == "half way there"


def test_parse_status_last_marker_wins_and_empty_gets_placeholder():
    status, _ = PaperclipGateway.parse_status("STATUS: blocked\nmore\nSTATUS: done")
    assert status == "done"
    status, comment = PaperclipGateway.parse_status("  STATUS: blocked  ")
    assert status == "blocked"
    assert comment  # placeholder, never an empty PATCH comment


def test_api_url_required_fail_loud():
    with pytest.raises(ValueError):
        PaperclipGateway({"enabled": True}, _RecordingBridge())


# --------------------------------------------------------------------------- #
# Heartbeat protocol
# --------------------------------------------------------------------------- #


def test_happy_path_targeted_wake(monkeypatch):
    api = _FakePaperclipAPI(_routes())
    bridge = _RecordingBridge()
    gw = _gateway(monkeypatch, api, bridge)

    result = asyncio.run(gw.run_heartbeat(_wake()))

    assert result == {"outcome": "completed", "issue_id": "i-1", "status": "done",
                      "tokens": 1800}
    # Auth + protocol order
    assert api.client_kwargs["headers"]["authorization"] == "Bearer pc-key"
    assert [(m, u) for m, u, _ in api.calls] == [
        ("GET", "/api/agents/me"),
        ("POST", "/api/issues/i-1/checkout"),
        ("GET", "/api/issues/i-1"),
        ("GET", "/api/issues/i-1/comments"),
        ("PATCH", "/api/issues/i-1"),
        ("POST", "/api/companies/co-1/cost-events"),
    ]
    # Checkout contract: agentId + documented expectedStatuses + run-id header
    checkout = api.called("POST", "/api/issues/i-1/checkout")[0]
    assert checkout["json"] == {
        "agentId": "agent-1",
        "expectedStatuses": ["todo", "backlog", "blocked", "in_review"],
    }
    assert checkout["headers"]["X-Paperclip-Run-Id"] == "r-1"
    # Report: done status + marker-stripped comment, mutating header present
    patch = api.called("PATCH", "/api/issues/i-1")[0]
    assert patch["json"] == {"comment": "Did the thing.", "status": "done"}
    assert patch["headers"]["X-Paperclip-Run-Id"] == "r-1"
    # Cost event carries real tokens + configured model identity
    cost = api.called("POST", "/api/companies/co-1/cost-events")[0]
    assert cost["json"] == {"agentId": "agent-1", "provider": "llama_cpp",
                            "model": "qwen-local", "inputTokens": 1500,
                            "outputTokens": 300}
    # The work ran in the durable per-issue session with issue context present
    assert bridge.turns[0][0] == "paperclip:issue:i-1"
    prompt = bridge.turns[0][1]
    assert "Ship the widget" in prompt and "please hurry" in prompt
    assert "STATUS: done" in prompt  # instructions include the marker contract


def test_checkout_conflict_never_retries(monkeypatch):
    api = _FakePaperclipAPI(
        _routes({("POST", "/api/issues/i-1/checkout"): _FakeResponse({}, status_code=409)})
    )
    bridge = _RecordingBridge()
    gw = _gateway(monkeypatch, api, bridge)

    result = asyncio.run(gw.run_heartbeat(_wake()))

    assert result == {"outcome": "checkout_conflict", "issue_id": "i-1"}
    assert bridge.turns == []  # no work
    assert [(m, u) for m, u, _ in api.calls] == [
        ("GET", "/api/agents/me"),
        ("POST", "/api/issues/i-1/checkout"),
    ]  # and nothing after the 409 — never retried


def test_in_progress_reply_patches_comment_only(monkeypatch):
    api = _FakePaperclipAPI(_routes())
    bridge = _RecordingBridge(reply="JWT signing done, refresh remains.\nSTATUS: in_progress")
    gw = _gateway(monkeypatch, api, bridge)

    result = asyncio.run(gw.run_heartbeat(_wake()))

    assert result["status"] == "in_progress"
    patch = api.called("PATCH", "/api/issues/i-1")[0]
    assert "status" not in patch["json"]  # checkout owns in_progress; comment-only
    assert patch["json"]["comment"] == "JWT signing done, refresh remains."


def test_untargeted_wake_picks_in_progress_from_inbox(monkeypatch):
    inbox = [
        {"id": "i-9", "status": "todo"},
        {"id": "i-1", "status": "in_progress"},
        {"id": "i-3", "status": "blocked"},
    ]
    api = _FakePaperclipAPI(
        _routes({("GET", "/api/companies/co-1/issues"): _FakeResponse(inbox)})
    )
    gw = _gateway(monkeypatch, api)

    result = asyncio.run(gw.run_heartbeat(_wake(issue_id=None)))

    assert result["outcome"] == "completed"
    assert result["issue_id"] == "i-1"  # in_progress beats todo; blocked skipped
    listing = api.called("GET", "/api/companies/co-1/issues")[0]
    assert listing["params"] == {
        "assigneeAgentId": "agent-1",
        "status": "todo,in_progress,in_review,blocked",
    }


def test_untargeted_wake_empty_inbox_is_idle(monkeypatch):
    api = _FakePaperclipAPI(
        _routes({("GET", "/api/companies/co-1/issues"): _FakeResponse([])})
    )
    bridge = _RecordingBridge()
    gw = _gateway(monkeypatch, api, bridge)

    result = asyncio.run(gw.run_heartbeat(_wake(issue_id=None)))

    assert result == {"outcome": "idle"}
    assert bridge.turns == []


def test_agent_error_posts_failure_comment_not_status(monkeypatch):
    api = _FakePaperclipAPI(_routes())
    bridge = _RecordingBridge(error=RuntimeError("model exploded"))
    gw = _gateway(monkeypatch, api, bridge)

    result = asyncio.run(gw.run_heartbeat(_wake()))

    assert result["outcome"] == "agent_error"
    assert "model exploded" in result["error"]
    failure = api.called("POST", "/api/issues/i-1/comments")[0]
    assert "model exploded" in failure["json"]["body"]
    assert api.called("PATCH", "/api/issues/i-1") == []  # no false done/blocked


def test_no_usage_skips_cost_event(monkeypatch):
    api = _FakePaperclipAPI(_routes())
    bridge = _RecordingBridge(usage=None)
    gw = _gateway(monkeypatch, api, bridge)

    result = asyncio.run(gw.run_heartbeat(_wake()))

    assert result["outcome"] == "completed"
    assert result["tokens"] is None
    assert api.called("POST", "/api/companies/co-1/cost-events") == []


def test_paperclip_server_down_returns_error_outcome(monkeypatch):
    api = _FakePaperclipAPI(
        _routes({("GET", "/api/agents/me"): _FakeResponse({}, status_code=500)})
    )
    gw = _gateway(monkeypatch, api)

    result = asyncio.run(gw.run_heartbeat(_wake()))

    assert result["outcome"] == "error"
    assert "500" in result["error"]


# --------------------------------------------------------------------------- #
# Bridge awaited-turn seam (run_turn_awaited)
# --------------------------------------------------------------------------- #


class _FakeSession:
    def __init__(self) -> None:
        self.user_messages: list[str] = []

    def add_user_message(self, content: str) -> int:
        self.user_messages.append(content)
        return len(self.user_messages)

    def last_persisted_row_id(self) -> int:
        return 7

    def get_messages(self) -> list:
        return []

    def persist_loop_result(self, original_len: int) -> None:
        self.persisted_from = original_len


class _FakeSessionMgr:
    def __init__(self) -> None:
        self.session = _FakeSession()
        self.keys: list[str] = []

    def get_or_create(self, session_id: str) -> _FakeSession:
        self.keys.append(session_id)
        return self.session

    def get(self, session_id: str):
        return self.session


# _run_agent dispatches on type(event).__name__, so this class name IS the contract.
class AssistantTextDelta:
    def __init__(self, text: str) -> None:
        self.text = text


@register_double(
    "paperclip._fake_run_loop", replaces="prometheus.engine.agent_loop.run_loop"
)
def _make_fake_run_loop(chunks: list[str], usage: Any, error: Exception | None = None):
    async def fake_run_loop(loop_context, messages, mode="agent", session_id=None, tool_choice=None):
        for chunk in chunks:
            yield AssistantTextDelta(chunk), usage
        if error is not None:
            raise error

    return fake_run_loop


def test_run_turn_awaited_returns_text_and_usage(monkeypatch):
    import prometheus.engine.agent_loop as agent_loop_mod
    from prometheus.web.ws_server import WebSocketBridge

    usage = _Usage()
    monkeypatch.setattr(
        agent_loop_mod, "run_loop", _make_fake_run_loop(["Hello ", "world.\nSTATUS: done"], usage)
    )
    mgr = _FakeSessionMgr()
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=object())

    text, got_usage = asyncio.run(bridge.run_turn_awaited("paperclip:issue:i-1", "do it"))

    assert text == "Hello world.\nSTATUS: done"
    assert got_usage is usage
    assert mgr.keys == ["paperclip:issue:i-1"]
    assert mgr.session.user_messages == ["do it"]
    assert mgr.session.persisted_from == 0  # assistant turn persisted to LCM


def test_run_turn_awaited_raises_on_agent_failure(monkeypatch):
    import prometheus.engine.agent_loop as agent_loop_mod
    from prometheus.web.ws_server import WebSocketBridge

    monkeypatch.setattr(
        agent_loop_mod, "run_loop",
        _make_fake_run_loop(["partial"], None, error=RuntimeError("provider down")),
    )
    bridge = WebSocketBridge(session_mgr=_FakeSessionMgr(), loop_context=object())

    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(bridge.run_turn_awaited("s", "work"))


def test_run_turn_awaited_requires_wiring():
    from prometheus.web.ws_server import WebSocketBridge

    with pytest.raises(RuntimeError, match="not fully wired"):
        asyncio.run(WebSocketBridge().run_turn_awaited("s", "work"))
