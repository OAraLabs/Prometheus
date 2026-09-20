"""Changing the clickable-role set drops stored click grants, loudly.

The consent extent is ``target:app:verb:delivery_mode`` and carries no role
term. So a grant for ``mini:nautilus:click:background`` — "click anything in
Nautilus" — is honoured against whatever the offered set contains when it is
USED, not what it contained when it was GIVEN. Add ``table cell`` and that
grant silently starts meaning "select files", with no new prompt.

Versioning the set into the extent was considered and rejected: the prompt
still reads "click anything", so a suffix fixes the machine half of the consent
and leaves the human half equally vague.

These tests assert the OUTCOME — the grant is gone and the count is visible —
not that a comparison happened.
"""

from __future__ import annotations

import pytest

from prometheus.computer import candidates as C
from prometheus.computer.candidates import role_set_fingerprint
from prometheus.permissions import checker as CK
from prometheus.permissions.checker import (
    COMPUTER_ACTION_KIND,
    Grant,
    _grants_for_current_role_set,
)


def _sec(fingerprint: str | None) -> dict:
    grants = [
        {"kind": COMPUTER_ACTION_KIND, "value": "mini:nautilus:click:background",
         "id": "g-click"},
        {"kind": "path_prefix", "value": "/home/will/projects", "id": "g-path"},
        {"kind": "tool", "tool_name": "bash", "id": "g-tool"},
    ]
    sec: dict = {"grants": grants}
    if fingerprint is not None:
        sec["role_set_fingerprint"] = fingerprint
    return sec


def test_an_unchanged_role_set_keeps_the_grant():
    """The guard must not fire on the case it exists to protect."""
    kept = _grants_for_current_role_set(_sec(role_set_fingerprint()))
    assert any(g.kind == COMPUTER_ACTION_KIND for g in kept)
    assert CK.ROLE_SET_DROPPED_GRANTS == 0


def test_changing_the_set_drops_the_click_grant_and_says_so(monkeypatch):
    """THE OUTCOME TEST. Store a grant, change the set, assert it is gone."""
    before = role_set_fingerprint()
    sec = _sec(before)

    # A real widening: exactly what adding `table cell` would do.
    monkeypatch.setattr(
        C, "_CLICKABLE_ROLES", frozenset({*C._CLICKABLE_ROLES, "table cell"})
    )
    assert role_set_fingerprint() != before, "the fingerprint did not move"

    kept = _grants_for_current_role_set(sec)

    assert not any(g.kind == COMPUTER_ACTION_KIND for g in kept), (
        "the click grant survived a role-set change — it now authorises "
        "clicking elements the operator never saw when granting"
    )
    assert CK.ROLE_SET_DROPPED_GRANTS == 1
    assert "role set changed" in CK.ROLE_SET_DROP_REASON
    assert "Re-grant" in CK.ROLE_SET_DROP_REASON


def test_only_click_grants_are_dropped(monkeypatch):
    """The role set says nothing about paths, commands or tools."""
    sec = _sec(role_set_fingerprint())
    monkeypatch.setattr(
        C, "_CLICKABLE_ROLES", frozenset({*C._CLICKABLE_ROLES, "table cell"})
    )
    kept = {g.kind for g in _grants_for_current_role_set(sec)}
    assert "path_prefix" in kept and "tool" in kept
    assert COMPUTER_ACTION_KIND not in kept


def test_the_count_is_surfaced_on_the_status_block(monkeypatch):
    """A silent drop is the same defect wearing the other hat."""
    from prometheus.computer.status import _role_set_drops

    monkeypatch.setattr(
        C, "_CLICKABLE_ROLES", frozenset({*C._CLICKABLE_ROLES, "table cell"})
    )
    _grants_for_current_role_set(_sec("stale-fingerprint"))

    block = _role_set_drops()
    assert block["count"] == 1, block
    assert "role set changed" in block["detail"]


def test_the_fingerprint_covers_the_editable_roles_too(monkeypatch):
    """An element becomes a candidate through EITHER set."""
    before = role_set_fingerprint()
    monkeypatch.setattr(
        C, "_EDITABLE_ROLES", frozenset({*C._EDITABLE_ROLES, "spin button"})
    )
    assert role_set_fingerprint() != before, (
        "widening _EDITABLE_ROLES left the fingerprint unchanged, so a type "
        "grant would survive a change to what can be typed into"
    )


def test_an_absent_fingerprint_DROPS_and_says_why():
    """Absence is not "unchanged".

    A grant with no fingerprint was made against a role set nobody recorded, so
    its scope is unknowable — and keeping it guesses, in the permissive
    direction. Same object as the three-term grant, whose ruling was
    refuse-don't-pad. Earlier this returned the grants untouched; that was
    convenience, not consent.
    """
    kept = _grants_for_current_role_set(_sec(None))

    assert not any(g.kind == COMPUTER_ACTION_KIND for g in kept), (
        "a click grant with NO role-set fingerprint survived — its scope is "
        "unknowable and it is being honoured against the current set"
    )
    assert CK.ROLE_SET_DROPPED_GRANTS == 1
    assert "NO role-set" in CK.ROLE_SET_DROP_REASON
    assert "unknowable" in CK.ROLE_SET_DROP_REASON
    # path/tool grants are unaffected: the role set says nothing about them
    assert {g.kind for g in kept} == {"path_prefix", "tool"}
