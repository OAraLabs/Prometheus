"""Regression tests for Bug B1: local backend override at tier 'full' dropping tools.

When a session switches to a configured local LlamaCppProvider whose grammar
was never wired (``grammar=None``), withholding ``tools`` at adapter tier
``"full"`` sent requests with neither ``tools`` nor ``grammar``. With the fix,
tier ``"full"`` withholds ``tools`` ONLY when the provider actually has a
grammar wired.
"""

from __future__ import annotations

from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.llama_cpp import LlamaCppProvider


def test_llama_cpp_provider_has_grammar_property() -> None:
    no_gram = LlamaCppProvider(base_url="http://127.0.0.1:18179")
    with_gram = LlamaCppProvider(base_url="http://127.0.0.1:18179", grammar="root ::= \"ok\"")
    assert no_gram.has_grammar is False
    assert with_gram.has_grammar is True


def test_override_provider_without_grammar_sends_tools_in_payload() -> None:
    provider = LlamaCppProvider(base_url="http://127.0.0.1:18179", grammar=None)
    sample_tools = [
        {
            "name": "read_file",
            "description": "Read a file from workspace.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]
    req = ApiMessageRequest(
        model="local-override",
        messages=[{"role": "user", "content": "Read README.md"}],
        tools=sample_tools,
        max_tokens=256,
    )
    payload = provider._build_request_payload(req)
    assert "tools" in payload and len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert "grammar" not in payload


def test_boot_provider_with_grammar_sends_grammar_when_tools_withheld() -> None:
    provider = LlamaCppProvider(base_url="http://127.0.0.1:18179", grammar='root ::= "ok"')
    req = ApiMessageRequest(
        model="boot-local",
        messages=[{"role": "user", "content": "Reply ok"}],
        tools=[],
        max_tokens=256,
    )
    payload = provider._build_request_payload(req)
    assert "tools" not in payload
    assert payload.get("grammar") == 'root ::= "ok"'
