"""Provider-level thinking-suppression + empty-content handling.

Branch fix/provider-suppress-thinking. Covers:

  * payload: with suppress_thinking on, chat_template_kwargs contains BOTH
    ``thinking`` and ``enable_thinking`` set to False; off → absent; default
    is on; per-call ``ApiMessageRequest.suppress_thinking`` overrides.
  * fallback: empty ``content`` + non-empty ``reasoning_content`` → returned
    as content AND recorded via ``telemetry.record_silent_failure``.
  * length-no-recovery: empty content + empty reasoning + finish="length"
    → surfaced in the recorded failure's context (budget_exhausted_on_thinking).
"""

from __future__ import annotations

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.llama_cpp import EmptyCompletionError, LlamaCppProvider


def _req(model: str = "gemma", suppress=None) -> ApiMessageRequest:
    return ApiMessageRequest(
        model=model,
        messages=[ConversationMessage.from_user_text("hi")],
        suppress_thinking=suppress,
    )


# --------------------------------------------------------------------------- #
# Payload construction — the chat_template_kwargs injection
# --------------------------------------------------------------------------- #


def test_default_payload_contains_both_keys_false() -> None:
    prov = LlamaCppProvider(base_url="http://x")
    payload = prov._build_request_payload(_req())
    ctk = payload.get("chat_template_kwargs")
    assert ctk == {"thinking": False, "enable_thinking": False}


def test_disabled_at_provider_strips_chat_template_kwargs() -> None:
    prov = LlamaCppProvider(base_url="http://x", suppress_thinking=False)
    payload = prov._build_request_payload(_req())
    assert "chat_template_kwargs" not in payload


def test_per_call_override_false_overrides_provider_on() -> None:
    prov = LlamaCppProvider(base_url="http://x", suppress_thinking=True)
    payload = prov._build_request_payload(_req(suppress=False))
    assert "chat_template_kwargs" not in payload


def test_per_call_override_true_overrides_provider_off() -> None:
    prov = LlamaCppProvider(base_url="http://x", suppress_thinking=False)
    payload = prov._build_request_payload(_req(suppress=True))
    assert payload["chat_template_kwargs"] == {
        "thinking": False, "enable_thinking": False,
    }


def test_default_is_on() -> None:
    # The whole point: callers do nothing and still get suppression.
    assert LlamaCppProvider(base_url="http://x")._suppress_thinking is True


def test_provider_default_used_when_request_field_is_none() -> None:
    prov = LlamaCppProvider(base_url="http://x", suppress_thinking=True)
    payload = prov._build_request_payload(_req(suppress=None))
    assert payload["chat_template_kwargs"]["thinking"] is False
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


# --------------------------------------------------------------------------- #
# Empty-content path — fallback to reasoning_content + telemetry record
# --------------------------------------------------------------------------- #


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_silent_failure(self, subsystem, operation, exc, context=None):  # noqa: ANN001
        self.calls.append(
            {"subsystem": subsystem, "operation": operation,
             "exc_type": type(exc).__name__, "exc_msg": str(exc),
             "context": context or {}}
        )


def _install_telemetry(monkeypatch, tel) -> None:
    # The provider calls get_telemetry_handle() lazily inside _finalize_content,
    # so we patch the lookup it actually performs.
    from prometheus.telemetry import tracker as t

    monkeypatch.setattr(t, "get_telemetry_handle", lambda: tel)


def test_empty_content_with_reasoning_falls_back_and_records(monkeypatch) -> None:
    tel = _RecordingTelemetry()
    _install_telemetry(monkeypatch, tel)
    prov = LlamaCppProvider(base_url="http://x")

    final_text, recorded = prov._finalize_content(
        accumulated_text="",
        accumulated_reasoning="The answer is OK.",
        finish_reason="stop",
        model="gemma",
        output_tokens=42,
    )

    assert final_text == "The answer is OK."   # fallback applied
    assert recorded is True
    assert len(tel.calls) == 1
    c = tel.calls[0]
    assert c["subsystem"] == "llama_cpp_provider"
    assert c["operation"] == "stream_message"
    assert c["exc_type"] == "EmptyCompletionError"
    assert c["context"]["used_reasoning_fallback"] is True
    assert c["context"]["budget_exhausted_on_thinking"] is False
    assert c["context"]["model"] == "gemma"


def test_empty_content_length_with_no_reasoning_marks_budget_exhausted(monkeypatch) -> None:
    tel = _RecordingTelemetry()
    _install_telemetry(monkeypatch, tel)
    prov = LlamaCppProvider(base_url="http://x")

    final_text, recorded = prov._finalize_content(
        accumulated_text="",
        accumulated_reasoning="",
        finish_reason="length",
        model="gemma",
        output_tokens=600,
    )

    assert final_text == ""                    # no recovery available
    assert recorded is True
    assert tel.calls[0]["context"]["budget_exhausted_on_thinking"] is True
    assert tel.calls[0]["context"]["used_reasoning_fallback"] is False


def test_nonempty_content_is_passthrough_no_record(monkeypatch) -> None:
    tel = _RecordingTelemetry()
    _install_telemetry(monkeypatch, tel)
    prov = LlamaCppProvider(base_url="http://x")

    final_text, recorded = prov._finalize_content(
        accumulated_text="hello world",
        accumulated_reasoning="",
        finish_reason="stop",
        model="gemma",
        output_tokens=2,
    )

    assert final_text == "hello world"
    assert recorded is False
    assert tel.calls == []


def test_no_telemetry_handle_does_not_crash(monkeypatch) -> None:
    # Even without a telemetry singleton wired (e.g. fresh test process),
    # the fallback path still recovers content and never raises.
    from prometheus.telemetry import tracker as t

    monkeypatch.setattr(t, "get_telemetry_handle", lambda: None)

    prov = LlamaCppProvider(base_url="http://x")
    final_text, recorded = prov._finalize_content(
        accumulated_text="",
        accumulated_reasoning="recovered",
        finish_reason="stop",
        model="m",
        output_tokens=5,
    )
    assert final_text == "recovered"
    assert recorded is False
