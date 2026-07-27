"""Structured turn-error classification (feat/turn-progress-and-errors).

The WS error frame used to carry only ``str(exc)``, which for the live
2026-07-24 incident read "Client error '400 Bad Request' for url ..." — the
actual cause ("Your credit balance is too low") was in the response body and
the next step was nowhere. These tests pin the mapping from exception to
{kind, provider, status, message, hint}, including the security property that a
URL query string (where Gemini carries its API key) is NEVER echoed.
"""

from __future__ import annotations

import json

from prometheus.api.turn_errors import (
    KIND_AUTH,
    KIND_BAD_REQUEST,
    KIND_BILLING,
    KIND_PROVIDER_ERROR,
    KIND_RATE_LIMIT,
    KIND_TIMEOUT,
    KIND_UNKNOWN,
    KIND_UNREACHABLE,
    classify_turn_error,
)


class _URL:
    def __init__(self, host: str) -> None:
        self.host = host


class _Request:
    def __init__(self, host: str) -> None:
        self.url = _URL(host)


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _HTTPStatusError(Exception):
    """Duck-typed stand-in for httpx.HTTPStatusError."""

    def __init__(self, msg: str, status: int, body: str = "", host: str = "api.anthropic.com"):
        super().__init__(msg)
        self.response = _Response(status, body)
        self.request = _Request(host)


def _anthropic_billing_body() -> str:
    return json.dumps({
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API. "
                       "Please go to Plans & Billing to upgrade or purchase credits.",
        },
    })


# ---------------------------------------------------------------------------
# The live incident: Anthropic 400 + credit-balance body
# ---------------------------------------------------------------------------


def test_anthropic_credit_balance_400_is_billing_not_bad_request():
    exc = _HTTPStatusError(
        "Client error '400 Bad Request' for url 'https://api.anthropic.com/v1/messages'",
        400, _anthropic_billing_body(),
    )
    out = classify_turn_error(exc)
    assert out["kind"] == KIND_BILLING  # NOT bad_request, despite the 400
    assert out["provider"] == "anthropic"
    assert out["status"] == 400
    # The provider's own sentence replaces the useless httpx string.
    assert "credit balance is too low" in out["message"]
    assert "Client error" not in out["message"]
    # And the hint names a concrete next step.
    assert "console.anthropic.com" in out["hint"]
    assert "switch this session's model" in out["hint"]


def test_402_is_billing_even_without_a_body():
    out = classify_turn_error(_HTTPStatusError("payment required", 402, "", "api.openai.com"))
    assert out["kind"] == KIND_BILLING
    assert out["provider"] == "openai"


# ---------------------------------------------------------------------------
# Other HTTP statuses
# ---------------------------------------------------------------------------


def test_401_and_403_are_auth():
    for status in (401, 403):
        out = classify_turn_error(_HTTPStatusError("nope", status, "", "api.x.ai"))
        assert out["kind"] == KIND_AUTH, status
        assert out["provider"] == "xai"
        assert "re-authenticate" in out["hint"]


def test_429_is_rate_limit():
    out = classify_turn_error(_HTTPStatusError("slow down", 429))
    assert out["kind"] == KIND_RATE_LIMIT
    assert "rate-limiting" in out["hint"]


def test_429_with_quota_body_is_billing_not_rate_limit():
    # OpenAI returns 429 for an exhausted balance — the body disambiguates.
    body = json.dumps({"error": {"message": "You exceeded your current quota."}})
    out = classify_turn_error(_HTTPStatusError("429", 429, body, "api.openai.com"))
    assert out["kind"] == KIND_BILLING


def test_5xx_is_provider_error():
    out = classify_turn_error(_HTTPStatusError("boom", 503))
    assert out["kind"] == KIND_PROVIDER_ERROR
    assert "retry" in out["hint"]


def test_other_4xx_is_bad_request():
    out = classify_turn_error(_HTTPStatusError("bad", 422))
    assert out["kind"] == KIND_BAD_REQUEST
    assert out["status"] == 422


# ---------------------------------------------------------------------------
# Non-HTTP failures
# ---------------------------------------------------------------------------


def test_timeout_family_is_timeout():
    class ReadTimeout(Exception):
        pass

    assert classify_turn_error(ReadTimeout("timed out"))["kind"] == KIND_TIMEOUT
    assert classify_turn_error(TimeoutError("timed out"))["kind"] == KIND_TIMEOUT


def test_connection_family_is_unreachable():
    class ConnectError(Exception):
        pass

    out = classify_turn_error(ConnectError("refused"))
    assert out["kind"] == KIND_UNREACHABLE
    assert "backend is running" in out["hint"]
    assert classify_turn_error(ConnectionRefusedError("refused"))["kind"] == KIND_UNREACHABLE


def test_unknown_exception_still_returns_a_usable_payload():
    out = classify_turn_error(TypeError("run_loop() got an unexpected keyword argument"))
    assert out["kind"] == KIND_UNKNOWN
    assert "unexpected keyword" in out["message"]
    assert out["hint"]  # never empty — always tells the human where to look


# ---------------------------------------------------------------------------
# Security + robustness
# ---------------------------------------------------------------------------


def test_never_echoes_a_url_or_query_string():
    # Gemini carries its API key in ?key=... — a leaked URL would leak the
    # credential to every connected client.
    secret = "AIzaSyLEAKED_KEY_MUST_NOT_APPEAR"
    exc = _HTTPStatusError(
        f"Client error '400' for url 'https://generativelanguage.googleapis.com/v1/x?key={secret}'",
        400, "", "generativelanguage.googleapis.com",
    )
    out = classify_turn_error(exc)
    blob = json.dumps(out)
    assert secret not in blob
    assert "key=" not in blob
    assert out["provider"] == "gemini"


def test_message_is_truncated():
    out = classify_turn_error(_HTTPStatusError("x", 400, "E" * 5000))
    assert len(out["message"]) <= 400


def test_classifier_never_raises_on_hostile_input():
    class Hostile(Exception):
        @property
        def response(self):  # noqa: D401 - raises on access
            raise RuntimeError("gotcha")

    out = classify_turn_error(Hostile("bad"))
    assert out["kind"] == KIND_UNKNOWN  # degraded, not exploded
    assert out["hint"]


def test_local_backend_is_named_readably():
    out = classify_turn_error(_HTTPStatusError("nope", 500, "", "127.0.0.1"))
    assert out["provider"] == "the local model backend"
