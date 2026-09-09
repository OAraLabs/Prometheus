"""The `message` tool's destination perimeter.

`message` is the only tool that POSTs model-authored CONTENT to a model-chosen
URL, which makes it an exfiltration primitive rather than merely an SSRF one.
The rule under test:

    user origin   -> the model may choose the destination
    system origin -> the destination must be OPERATOR-supplied (env var)

Every case drives the REAL ``MessageTool.execute`` with a real
``ToolExecutionContext`` and asserts on the URL that actually reached the HTTP
client — not on the helper in isolation. A refusal is proven by an EMPTY call
log, because "returned an error" and "sent it anyway then reported an error"
are indistinguishable from the return value alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import message as msg
from prometheus.tools.builtin.message import (
    WEBHOOK_URL_ENV,
    MessageInput,
    MessagePlatform,
    MessageTool,
)

MODEL_URL = "https://collector.example.com/drop"
OPERATOR_URL = "https://hooks.example.net/operator-owned"


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"ok": True}


class _FakeAsyncClient:
    def __init__(self, calls: list[tuple[str, dict]]) -> None:
        self._calls = calls

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self._calls.append((url, kwargs))
        return _FakeResponse()


@pytest.fixture()
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Record every POST the tool makes; patched at the module's own httpx."""
    recorded: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        msg.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(recorded)
    )
    return recorded


def _ctx(session_id: str | None, *, with_metadata: bool = True) -> ToolExecutionContext:
    """A context shaped like the agent loop's, which sets session_id itself."""
    if not with_metadata:
        return ToolExecutionContext(cwd=Path.cwd())
    return ToolExecutionContext(cwd=Path.cwd(), metadata={"session_id": session_id})


async def _send(platform: MessagePlatform, recipient: str | None, ctx):
    return await MessageTool().execute(
        MessageInput(platform=platform, content="payload", recipient=recipient), ctx
    )


# ---------------------------------------------------------------------------
# System origin: the model may not choose where data goes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform, env_var",
    [
        (MessagePlatform.webhook, WEBHOOK_URL_ENV),
        (MessagePlatform.discord, "DISCORD_WEBHOOK_URL"),
    ],
)
async def test_system_origin_refuses_a_model_chosen_url(
    platform, env_var, calls, monkeypatch
):
    monkeypatch.delenv(env_var, raising=False)
    result = await _send(platform, MODEL_URL, _ctx("gepa-7f3a-uuid"))

    assert result.is_error
    assert calls == [], "refused send still opened a connection"
    # The refusal must tell the model how to make this legitimate, or it will
    # simply retry the same call.
    assert env_var in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform, env_var",
    [
        (MessagePlatform.webhook, WEBHOOK_URL_ENV),
        (MessagePlatform.discord, "DISCORD_WEBHOOK_URL"),
    ],
)
async def test_system_origin_uses_the_operator_url_and_ignores_the_model(
    platform, env_var, calls, monkeypatch
):
    monkeypatch.setenv(env_var, OPERATOR_URL)
    result = await _send(platform, MODEL_URL, _ctx("cron-nightly-uuid"))

    assert not result.is_error, result.output
    assert [url for url, _ in calls] == [OPERATOR_URL]
    assert MODEL_URL not in [url for url, _ in calls]


@pytest.mark.asyncio
async def test_a_context_without_metadata_is_system_origin(calls, monkeypatch):
    """Jobs and cron build a bare context; absent id must not read as a user."""
    monkeypatch.delenv(WEBHOOK_URL_ENV, raising=False)
    result = await _send(
        MessagePlatform.webhook, MODEL_URL, _ctx(None, with_metadata=False)
    )

    assert result.is_error
    assert calls == []


@pytest.mark.asyncio
async def test_an_unrecognised_session_id_is_system_origin(calls, monkeypatch):
    monkeypatch.delenv(WEBHOOK_URL_ENV, raising=False)
    result = await _send(MessagePlatform.webhook, MODEL_URL, _ctx("smoke-test-42"))

    assert result.is_error
    assert calls == []


# ---------------------------------------------------------------------------
# User origin: unchanged. A human asked and can see the answer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["telegram:12345", "slack:C01", "cli", "web"])
async def test_user_origin_may_choose_any_destination(session_id, calls, monkeypatch):
    """'Post this to my ntfy endpoint' keeps working, including local hosts."""
    monkeypatch.delenv(WEBHOOK_URL_ENV, raising=False)
    result = await _send(MessagePlatform.webhook, MODEL_URL, _ctx(session_id))

    assert not result.is_error, result.output
    assert [url for url, _ in calls] == [MODEL_URL]


@pytest.mark.asyncio
async def test_user_origin_reaches_the_operators_own_local_service(calls, monkeypatch):
    """An address-class guard would have broken this; the origin rule must not."""
    monkeypatch.delenv(WEBHOOK_URL_ENV, raising=False)
    local = "http://127.0.0.1:9999/notify"
    result = await _send(MessagePlatform.webhook, local, _ctx("cli"))

    assert not result.is_error, result.output
    assert [url for url, _ in calls] == [local]


@pytest.mark.asyncio
async def test_user_origin_keeps_the_model_over_env_precedence(calls, monkeypatch):
    """Pre-existing precedence (`recipient or ENV`) is preserved exactly."""
    monkeypatch.setenv(WEBHOOK_URL_ENV, OPERATOR_URL)
    result = await _send(MessagePlatform.webhook, MODEL_URL, _ctx("telegram:1"))

    assert not result.is_error, result.output
    assert [url for url, _ in calls] == [MODEL_URL]


# ---------------------------------------------------------------------------
# Scope: the platforms that POST to hardcoded API hosts are untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_telegram_still_treats_recipient_as_a_chat_id_at_system_origin(
    calls, monkeypatch
):
    """daily_briefing is a cron job on this path — it must not regress."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    result = await _send(MessagePlatform.telegram, "98765", _ctx("cron-briefing"))

    assert not result.is_error, result.output
    assert len(calls) == 1
    assert calls[0][0].startswith("https://api.telegram.org/")
    assert calls[0][1]["json"]["chat_id"] == "98765"


@pytest.mark.asyncio
async def test_slack_bot_path_still_posts_to_slack_at_system_origin(
    calls, monkeypatch
):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    result = await _send(MessagePlatform.slack, "C0123", _ctx("sentinel-uuid"))

    assert not result.is_error, result.output
    assert [url for url, _ in calls] == ["https://slack.com/api/chat.postMessage"]


# ---------------------------------------------------------------------------
# The origin cannot be forged from tool arguments
# ---------------------------------------------------------------------------


def test_origin_is_not_reachable_from_the_input_schema():
    """The loop sets session_id from LoopContext; the model cannot supply one.

    If a session/origin field is ever added to MessageInput, the perimeter
    becomes model-controlled and this test should fail loudly rather than the
    property degrading silently.
    """
    fields = set(MessageInput.model_fields)
    assert fields == {"platform", "content", "recipient"}
