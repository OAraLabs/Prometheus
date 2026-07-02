"""Pure unit tests for prometheus.api.tool_choice (IGNITION).

Quarried from draft #73's TestValidation — rewritten to drive the module's
functions DIRECTLY (no conftest fixtures, no doubles, nothing to register:
these are pure functions on plain data, tripwire-irrelevant by construction).
"""

from __future__ import annotations

import pytest

from prometheus.api.tool_choice import (
    AUTO,
    NONE,
    REQUIRED,
    forced_tool_name,
    normalize_tool_choice,
    resolve_mode_to_tool_choice,
)

VALID = frozenset({"web_search", "bash", "read_file"})


class TestNormalize:
    def test_scalars_pass_through(self):
        assert normalize_tool_choice("auto", VALID) == AUTO
        assert normalize_tool_choice("none", VALID) == NONE
        assert normalize_tool_choice("required", VALID) == REQUIRED

    def test_specific_tool_normalizes(self):
        assert normalize_tool_choice({"tool": "web_search"}, VALID) == {"tool": "web_search"}

    def test_unknown_tool_fails_loud(self):
        with pytest.raises(ValueError):
            normalize_tool_choice({"tool": "not_a_tool"}, VALID)

    def test_malformed_shapes_fail_loud(self):
        for bad in ("REQUIRED", "", 7, {"tool": ""}, {"name": "web_search"}, ["required"], {"tool": 3}):
            with pytest.raises(ValueError):
                normalize_tool_choice(bad, VALID)

    def test_no_registry_still_validates_shape(self):
        # valid_tool_names=None (no registry available) → shape-validate only.
        assert normalize_tool_choice({"tool": "anything"}, None) == {"tool": "anything"}
        with pytest.raises(ValueError):
            normalize_tool_choice("bogus", None)


class TestModeSugar:
    def test_agent_resolves_to_auto(self):
        assert resolve_mode_to_tool_choice("agent") == AUTO

    def test_chat_resolves_to_none(self):
        assert resolve_mode_to_tool_choice("chat") == NONE

    def test_unknown_and_absent_resolve_to_auto(self):
        # An unrecognized value must NEVER silently drop tools (byte-identical default).
        assert resolve_mode_to_tool_choice(None) == AUTO
        assert resolve_mode_to_tool_choice("weird") == AUTO


class TestForcedToolName:
    def test_dict_yields_name(self):
        assert forced_tool_name({"tool": "web_search"}) == "web_search"

    def test_scalars_yield_none(self):
        assert forced_tool_name(AUTO) is None
        assert forced_tool_name(REQUIRED) is None
        assert forced_tool_name(NONE) is None
