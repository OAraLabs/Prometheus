"""Regression tests for provider-specific tool-choice limitations."""

import pytest

from prometheus.providers.openai_compat import _native_tool_choice


def test_mimo_accepts_automatic_tool_choice() -> None:
    assert _native_tool_choice("auto", provider_name="mimo") == "auto"
    assert _native_tool_choice(None, provider_name="MiMo") == "auto"


@pytest.mark.parametrize("tool_choice", ["required", {"tool": "web_search"}])
def test_mimo_rejects_forced_tool_choice(tool_choice: object) -> None:
    with pytest.raises(ValueError, match="MiMo only supports tool_choice='auto'"):
        _native_tool_choice(tool_choice, provider_name="mimo")


def test_other_openai_compatible_providers_keep_forced_tool_choice() -> None:
    assert _native_tool_choice("required", provider_name="openai") == "required"
    assert _native_tool_choice(
        {"tool": "web_search"}, provider_name="openai"
    ) == {"type": "function", "function": {"name": "web_search"}}
