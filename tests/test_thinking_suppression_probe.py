"""The thinking-suppression flag must not be able to silently do nothing.

WHY
---
``model.suppress_thinking`` is a chat-template kwarg. A template that does not
recognise the key ignores it in silence: the server returns 200, the
completion looks ordinary, and the reasoning channel runs anyway — spending
the output budget and returning empty ``content``.

Whether it worked was readable from NOTHING. Not the code (the payload is
built three call-frames away from the config key), not the config (the key is
just ``true``), not the logs (there were none). On 2026-08-17 that gap
produced a confidently wrong root-cause report: a probe of
``{"thinking": false}`` ALONE — a payload the provider never sends — was read
as proof the flag was inert, when the real payload also sends
``enable_thinking`` and does work.

That error is what these tests pin.

THE THREE PROPERTIES
--------------------
1. The probe sends the REAL payload (``_build_request_payload``). A probe
   that hand-writes its body can test something production never sends,
   which is precisely how the wrong report happened.
2. Empty reasoning is not automatically a pass — a control call separates
   "suppression worked" from "this model never reasons".
3. A failed probe is ``unknown``, never ``supported``. Absence of effect must
   not read as effect, and that discipline applies to the instrument too.
"""

from __future__ import annotations

import json

import pytest

from prometheus.providers.llama_cpp import LlamaCppProvider


class _FakeResponse:
    def __init__(self, reasoning: str) -> None:
        self._reasoning = reasoning

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{
                "message": {"content": "ready", "reasoning_content": self._reasoning}
            }]
        }


class _FakeClient:
    """Records every posted body; answers per-call from a script."""

    def __init__(self, script) -> None:
        self._script = script
        self.bodies: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, **kw):
        self.bodies.append(json)
        return self._script(json, len(self.bodies) - 1)


def _provider() -> LlamaCppProvider:
    return LlamaCppProvider(base_url="http://x", suppress_thinking=True)


def _patch(monkeypatch, script) -> _FakeClient:
    client = _FakeClient(script)
    monkeypatch.setattr(
        "prometheus.providers.llama_cpp.httpx.AsyncClient",
        lambda *a, **k: client,
    )
    return client


@pytest.mark.asyncio
async def test_probe_sends_the_payload_production_actually_sends(monkeypatch):
    """PROPERTY 1 — the body comes from the real builder.

    The wrong report of 2026-08-17 came from probing ``{"thinking": false}``
    alone. The provider sends BOTH keys; a probe that does not is testing a
    payload that does not exist.
    """
    client = _patch(monkeypatch, lambda body, i: _FakeResponse("" if i == 0 else "hmm"))
    await _provider().verify_thinking_suppression()

    suppressed_body = client.bodies[0]
    kwargs = suppressed_body.get("chat_template_kwargs") or {}
    assert kwargs.get("thinking") is False, (
        "the probe dropped the gemma-family key; it is no longer testing what "
        "the agent loop sends"
    )
    assert kwargs.get("enable_thinking") is False, (
        "the probe dropped the Qwen-family key — this is the exact omission "
        "that produced the wrong root cause"
    )
    # And the control must genuinely turn suppression OFF, or it is not a
    # control and cannot distinguish 'works' from 'never reasons'.
    assert not (client.bodies[1].get("chat_template_kwargs") or {})


@pytest.mark.asyncio
async def test_suppressed_and_control_reasons_is_supported(monkeypatch):
    _patch(monkeypatch, lambda body, i: _FakeResponse("" if i == 0 else "thinking…"))
    status, detail = await _provider().verify_thinking_suppression()
    assert status == "supported", detail


@pytest.mark.asyncio
async def test_reasoning_despite_suppression_is_unsupported(monkeypatch):
    """The failure the whole probe exists to catch."""
    _patch(monkeypatch, lambda body, i: _FakeResponse("still thinking hard"))
    status, detail = await _provider().verify_thinking_suppression()
    assert status == "unsupported"
    assert "ignores both" in detail


@pytest.mark.asyncio
async def test_a_model_that_never_reasons_is_moot_not_supported(monkeypatch):
    """PROPERTY 2 — empty reasoning alone is not evidence the flag worked.

    Without the control these are the same observation. Reporting this as
    "supported" would claim the flag did something when nothing was there to
    suppress — a small lie that becomes a large one at the next model swap.
    """
    _patch(monkeypatch, lambda body, i: _FakeResponse(""))
    status, detail = await _provider().verify_thinking_suppression()
    assert status == "moot", detail
    assert "not doing anything" in detail


@pytest.mark.asyncio
async def test_a_failed_probe_is_unknown_never_supported(monkeypatch):
    """PROPERTY 3 — the instrument's own silence is not a pass."""
    def _boom(body, i):
        raise RuntimeError("connection refused")

    _patch(monkeypatch, _boom)
    status, detail = await _provider().verify_thinking_suppression()
    assert status == "unknown", (
        f"a probe that could not run reported {status!r} — an unreachable "
        f"server must never be read as a working suppression flag"
    )
    assert "connection refused" in detail


@pytest.mark.asyncio
async def test_probe_does_not_stream_or_carry_a_grammar(monkeypatch):
    """A capability probe must not be shaped by the tool-calling machinery."""
    client = _patch(monkeypatch, lambda body, i: _FakeResponse(""))
    await _provider().verify_thinking_suppression()
    for body in client.bodies:
        assert body.get("stream") is False
        assert "grammar" not in body
        assert "stream_options" not in body


@pytest.mark.asyncio
async def test_probe_is_skipped_when_the_provider_does_not_suppress(monkeypatch):
    """The provider is asked, not the config, and a disabled flag is honest.

    Re-reading ``model.suppress_thinking`` in the daemon would make a second
    reader of one setting, and would have demanded a real template key for a
    knob #246 deliberately left commented-out (absent already means true).
    """
    client = _patch(monkeypatch, lambda body, i: _FakeResponse("thinking"))
    p = LlamaCppProvider(base_url="http://x", suppress_thinking=False)
    status, detail = await p.verify_thinking_suppression()
    assert status == "skipped", detail
    assert not client.bodies, "a skipped probe still called the server"
