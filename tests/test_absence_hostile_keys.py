"""Absence-hostile config keys — a missing key must not break the product.

THE CLASS (FIRSTLIGHT FL-2u)
---------------------------
80 of the template's 316 keys are absent from the operator's live config and
79 of them are harmless: the reader supplies a sane default. The dangerous
minority are keys whose fallback is **degenerate** — absence produces a
system that silently violates the shipped contract. FL-2u was one:
``tools.deferred_loading.always_loaded`` fell back to ``[]``, which with
deferred loading active means *advertise nothing*, so every install that
upgraded past FL-2 kept an old config and handed the model no callable
tools.

The fix for that class is never "write the missing key into the user's
config" — it is to make absence safe (CROSS-CUTTING §5: a property that
cannot be violated beats a check that must remember to run). This file is
the regression test for that property, and the ratchet that keeps the
category from being forgotten again.

TWO LISTS, ONE DIRECTION
------------------------
* :data:`ABSENCE_SAFE` — keys whose absence is now SAFE. Each entry carries
  a far-side assertion (§2d: build the REAL consumer from a config lacking
  the key and check what it produces, not what the config says).
* :data:`KNOWN_DEGENERATE` — a **shrinking debt list**, same shape and same
  discipline as ``KNOWN_UNREAD`` in ``test_config_drift.py``: keys found to
  be absence-hostile whose fix is out of scope here (both are
  security-adjacent, and the sprint's rules halt on the permission model
  and the security gate). Each entry asserts the defect is STILL THERE, so
  the day someone fixes one this test goes red and tells them to promote
  it. It can only shrink — it is not an allowlist.
"""

from __future__ import annotations

import pytest

from prometheus.config.shipped_defaults import SHIPPED_ALWAYS_LOADED


# ---------------------------------------------------------------------------
# ABSENCE_SAFE — the property must hold
# ---------------------------------------------------------------------------

def _advertised_without_the_key() -> set[str]:
    """What the model is OFFERED when the config has no tools: section."""
    from prometheus.__main__ import create_tool_registry
    from prometheus.context.dynamic_tools import DynamicToolLoader

    # The exact shape daemon.py:355 passes for a config that predates the
    # key: config.get("tools", {}).get("deferred_loading") -> None.
    loader = DynamicToolLoader(create_tool_registry({}), None)
    return {s.get("name") for s in loader.schemas_for_run(True)}


def test_always_loaded_absent_still_advertises_the_shipped_set():
    """FL-2u, far side: no tools: section -> the shipped set, not nothing."""
    advertised = _advertised_without_the_key()
    assert advertised, (
        "a config without tools.deferred_loading advertises NOTHING — FL-2u "
        "regressed; an upgraded install's model cannot call any tool"
    )
    assert advertised == set(SHIPPED_ALWAYS_LOADED), (
        f"absent always_loaded should fall back to the shipped set; got "
        f"{sorted(advertised)}"
    )


def test_an_explicitly_empty_always_loaded_is_still_honoured():
    """The other direction: the fallback must not override an operator who
    deliberately wrote an empty list. Absence and emptiness are different
    statements and only the first is a mistake."""
    from prometheus.__main__ import create_tool_registry
    from prometheus.context.dynamic_tools import DynamicToolLoader

    loader = DynamicToolLoader(
        create_tool_registry({}), {"enabled": "auto", "always_loaded": []}
    )
    assert loader.schemas_for_run(True) == [], (
        "an explicit `always_loaded: []` was overridden by the FL-2u "
        "fallback — the fallback must fire only on ABSENCE"
    )


def test_a_configured_set_still_wins():
    """And a real configured value is untouched by the fallback."""
    from prometheus.__main__ import create_tool_registry
    from prometheus.context.dynamic_tools import DynamicToolLoader

    loader = DynamicToolLoader(
        create_tool_registry({}),
        {"enabled": "auto", "always_loaded": ["bash", "grep"]},
    )
    assert {s.get("name") for s in loader.schemas_for_run(True)} == {
        "bash", "grep"}


# ---------------------------------------------------------------------------
# KNOWN_DEGENERATE — the shrinking debt list
# ---------------------------------------------------------------------------
#
# Found by the FL-2u audit ("a category with one member is usually a category
# nobody looked for"). Both are FAIL-OPEN rather than fail-dead, both are
# security-adjacent, and both are therefore reported rather than fixed here.

KNOWN_DEGENERATE: dict[str, str] = {
    "gateway.media.allowed_{image,audio,document}_types":
        "absent -> [] -> media_guard's `if allowed and ...` (media_guard.py:"
        "210,250,270) SKIPS the MIME check entirely, so an install whose "
        "config predates PR #141 has NO type filtering on the Telegram "
        "surface — the one exposed to the public internet by design. #141 "
        "fixed the shipped TEMPLATE; it did not make absence safe.",
    "security.workspace_root":
        "absent -> None -> SecurityGate._within_workspace (checker.py:459) "
        "returns True unconditionally, so file operations are not confined "
        "at all. The template ships a value; a config that omits the key "
        "silently has no workspace boundary.",
}


def test_media_allowlists_absent_still_skip_the_mime_check():
    """Debt-list pin. When this goes RED, the defect is fixed — move the key
    to ABSENCE_SAFE with a real far-side assertion and delete this."""
    from prometheus.gateway.media_guard import MediaPolicy

    media_cfg: dict = {}  # a config section that predates the keys
    policy = MediaPolicy(
        allowed_image_types=tuple(media_cfg.get("allowed_image_types") or []),
        allowed_audio_types=tuple(media_cfg.get("allowed_audio_types") or []),
        allowed_document_types=tuple(
            media_cfg.get("allowed_document_types") or []),
        max_file_size_mb=media_cfg.get("max_file_size_mb", 20),
    )
    assert policy.allowlist_for("image") == (), (
        "media allowlists are no longer degenerate on absence — promote "
        "them out of KNOWN_DEGENERATE"
    )


def test_workspace_root_absent_still_confines_nothing():
    """Debt-list pin, far side through the real checker."""
    from prometheus.permissions.checker import SecurityGate

    # The exact shape from_config produces for a config lacking the key:
    # sec.get("workspace_root") -> None.
    gate = SecurityGate(workspace_root=None)
    assert gate._within_workspace("/etc/passwd") is True, (
        "workspace_root absence now confines file operations — promote it "
        "out of KNOWN_DEGENERATE"
    )


@pytest.mark.parametrize("key", sorted(KNOWN_DEGENERATE))
def test_every_debt_entry_states_why(key):
    """A debt list without reasons becomes an allowlist."""
    why = KNOWN_DEGENERATE[key]
    assert len(why) > 80 and "absent" in why, (
        f"{key} needs a reason naming what absence produces"
    )
