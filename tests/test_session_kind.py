"""The machine-session namespace — the naming contract probes must honour.

The audit this came from: 213 tombstoned probe sessions purged by hand, of which only the 68
`bakeoff:` ones were nameable. The rest were `desktop:`, `ios:` and `beacon:` — chosen by the
harnesses that made them. These tests pin BOTH halves: what the daemon recognises, and the fact
that an ordinary-looking id is indistinguishable from a real conversation, which is why the
client has to opt in.
"""

from __future__ import annotations

import pytest

from prometheus.memory.session_kind import (
    MACHINE_SESSION_IDS,
    MACHINE_SESSION_PREFIXES,
    is_machine_session,
)


@pytest.mark.parametrize(
    "sid",
    ["bakeoff:t01_rename:r1:1781182970", "coding:x", "eval:y", "gym:z", "smoke:s4-123", "system"],
)
def test_machine_traffic_is_recognised(sid):
    assert is_machine_session(sid) is True


@pytest.mark.parametrize("sid", ["beacon:117082a6", "desktop:abc", "ios:xyz", "telegram:123", "web"])
def test_a_real_conversation_is_not(sid):
    assert is_machine_session(sid) is False


def test_the_ids_that_actually_filled_the_store_are_NOT_recognised():
    """The uncomfortable half, pinned deliberately.

    These are real ids purged in the 2026-09-01 cleanup. The daemon cannot tell them from a user
    chat, so no daemon-side rule can prevent them — only the harness renaming itself can. If this
    ever starts returning True, someone has widened the prefixes to swallow a live namespace and
    real conversations are now classed as disposable.
    """
    for sid in ["desktop:s4-2332364.1786774889602", "ios:approval-gate-1787962456",
                "beacon:2b56a05e0a304599bafc365b69d3b4ef", "walk-wide", "floorcheck"]:
        assert is_machine_session(sid) is False


def test_empty_and_none_are_not_machine():
    assert is_machine_session(None) is False
    assert is_machine_session("") is False
    assert is_machine_session("   ") is False


def test_a_prefix_must_end_with_a_colon():
    """`smokescreen:1` is not `smoke:`. Namespaces are delimited, not substrings."""
    assert is_machine_session("smokescreen:1") is False
    assert is_machine_session("evaluation:1") is False
    assert is_machine_session("systemic") is False


def test_the_contract_is_still_the_one_the_extractor_uses():
    """One definition. A second copy is how the wiki filter and retention drift apart."""
    from prometheus.memory import extractor

    assert extractor._is_machine_session is is_machine_session
    assert extractor._MACHINE_SESSION_PREFIXES is MACHINE_SESSION_PREFIXES
    assert extractor._MACHINE_SESSION_IDS is MACHINE_SESSION_IDS
