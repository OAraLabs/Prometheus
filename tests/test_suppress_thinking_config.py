"""Config-layer thinking-suppression override (feat/thinking-config).

The provider-level behaviour is covered by test_llama_cpp_suppress_thinking.py
(constructing LlamaCppProvider directly). This file locks the CONFIG path the
daemon actually uses: ``model.suppress_thinking`` in prometheus.yaml flows
through ``ProviderRegistry.create`` into the provider and shows up — or not —
on the OUTGOING request payload (``chat_template_kwargs.thinking``), the exact
kwargs path the 2026-06-11 bakeoff verified.

Side effects are asserted on the built payload (the actual JSON body sent to
``/v1/chat/completions``), never by counting calls. Contract:

  * key UNSET  → default preserved: thinking suppressed (current behaviour).
  * key true   → thinking suppressed (explicit, same as default).
  * key false  → thinking ENABLED: no chat_template_kwargs suppression keys.
"""

from __future__ import annotations

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.registry import ProviderRegistry


def _payload_for_model_config(model_cfg: dict) -> dict:
    """Build a provider from a ``model:`` config block exactly as the daemon
    does, then return the outgoing /v1/chat/completions payload."""
    provider = ProviderRegistry.create({"provider": "llama_cpp", **model_cfg})
    assert isinstance(provider, LlamaCppProvider)
    request = ApiMessageRequest(
        model=model_cfg.get("model", "gemma4-26b"),
        messages=[ConversationMessage(role="user", content=[TextBlock(text="hi")])],
        max_tokens=16,
    )
    return provider._build_request_payload(request)


# --------------------------------------------------------------------------- #
# The config override, asserted on the outgoing payload
# --------------------------------------------------------------------------- #

def test_key_unset_preserves_current_behaviour_thinking_suppressed() -> None:
    """No ``model.suppress_thinking`` key → identical to today: suppressed."""
    payload = _payload_for_model_config({})
    assert payload["chat_template_kwargs"] == {
        "thinking": False,
        "enable_thinking": False,
    }


def test_key_true_suppresses_thinking() -> None:
    payload = _payload_for_model_config({"suppress_thinking": True})
    assert payload["chat_template_kwargs"] == {
        "thinking": False,
        "enable_thinking": False,
    }


def test_key_false_enables_thinking_no_suppression_kwargs() -> None:
    """The override: ``suppress_thinking: false`` strips the suppression keys
    so the model's chat template runs its thinking channel."""
    payload = _payload_for_model_config({"suppress_thinking": False})
    assert "chat_template_kwargs" not in payload


def test_provider_field_reflects_config_key() -> None:
    """The config key sets the provider's stored default (the value
    ``_effective_suppress`` falls back to when a request doesn't override)."""
    assert ProviderRegistry.create(
        {"provider": "llama_cpp"})._suppress_thinking is True
    assert ProviderRegistry.create(
        {"provider": "llama_cpp", "suppress_thinking": True})._suppress_thinking is True
    assert ProviderRegistry.create(
        {"provider": "llama_cpp", "suppress_thinking": False})._suppress_thinking is False


@pytest.mark.parametrize("flag,expect_suppressed", [(True, True), (False, False)])
def test_config_key_roundtrip(flag: bool, expect_suppressed: bool) -> None:
    payload = _payload_for_model_config({"suppress_thinking": flag})
    has_kwargs = "chat_template_kwargs" in payload
    assert has_kwargs is expect_suppressed
    if has_kwargs:
        assert payload["chat_template_kwargs"]["thinking"] is False
