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

# EMPTY as of 2026-08-12. Both original entries were FIXED and promoted to
# ABSENCE_SAFE below; the list may only shrink, and it reached zero. A new
# entry needs a disposition and a pin asserting the defect is still there.
KNOWN_DEGENERATE: dict[str, str] = {}


@pytest.mark.parametrize("key", sorted(KNOWN_DEGENERATE))
def test_every_debt_entry_states_why(key):
    """A debt list without reasons becomes an allowlist."""
    why = KNOWN_DEGENERATE[key]
    assert len(why) > 80 and "absent" in why, (
        f"{key} needs a reason naming what absence produces"
    )


# ---------------------------------------------------------------------------
# PROMOTED 2026-08-12 — the two fail-open entries, now ABSENCE_SAFE.
#
# Both were worse than FL-2u's fail-DEAD case: a degenerate fallback that
# disables a SECURITY control announces nothing. FL-2u's install was visibly
# useless (no tools); these two look like a working system.
#
# Both directions per §2c — a control needs breach tests AND admission tests,
# or the suite is blind in one direction by construction. The fail-closed
# over-correction (confine everything to nothing / refuse every file) would
# satisfy a breach-only suite perfectly.
# ---------------------------------------------------------------------------

# ── security.workspace_root ──────────────────────────────────────────────────

def _gate_from_yaml(tmp_path, security: dict | None, *, raw: str | None = None):
    """The REAL consumer, from a real config FILE.

    ``SecurityGate.from_config`` takes a PATH, not a dict — passing a dict
    lands in its bare ``except`` and silently yields ``sec = {}``, so a test
    that hands it a dict proves nothing about the key it names. (Written that
    way here first; the explicit-value test is what caught it.)
    """
    import yaml

    from prometheus.permissions.checker import SecurityGate

    path = tmp_path / "prometheus.yaml"
    path.write_text(
        raw if raw is not None else yaml.safe_dump({"security": security or {}}),
        encoding="utf-8",
    )
    return SecurityGate.from_config(path)


def test_workspace_root_absent_still_confines(tmp_path):
    """BREACH direction. Was: absent -> None -> _within_workspace returned
    True for every path, so file operations had no boundary at all."""
    gate = _gate_from_yaml(tmp_path, {"permission_mode": "default"})
    assert gate._within_workspace("/etc/passwd") is False, (
        "a config with no workspace_root confines nothing — the fail-open "
        "default is back"
    )


def test_workspace_root_absent_still_admits_the_workspace(tmp_path):
    """ADMISSION direction. A boundary that refuses its own workspace is a
    broken product, and a breach-only suite would call it a pass.

    The expected path is LITERAL, not derived from SHIPPED_WORKSPACE_ROOT.
    Written the derived way first, and mutation M2 — repointing the constant
    at /nonexistent/nowhere — SURVIVED: the test built its input from the
    value under test, so the two moved together and no input could ever fail
    it. A self-referential assertion is not an assertion. Drift between this
    literal and the constant is caught by the template pin below.
    """
    from pathlib import Path

    gate = _gate_from_yaml(tmp_path, {"permission_mode": "default"})
    inside = str(Path("~/.prometheus/workspace/notes.md").expanduser())
    assert gate._within_workspace(inside) is True, (
        f"{inside} is inside the shipped workspace and was refused — an "
        f"over-correction here makes the agent unable to touch its own files"
    )


def test_explicit_workspace_root_still_wins(tmp_path):
    """The resolver must not override an operator who set the key."""
    from pathlib import Path

    root = tmp_path / "ws"
    root.mkdir()
    gate = _gate_from_yaml(tmp_path, {"workspace_root": str(root)})
    assert gate._within_workspace(str(root / "x")) is True
    assert gate._within_workspace("/etc/passwd") is False


def test_an_unreadable_config_confines_rather_than_opening_up(tmp_path):
    """Bonus hardening, and worth pinning because it is the scarier path.

    ``from_config`` swallows every failure into ``sec = {}`` — so a corrupt,
    truncated or unparseable config used to produce workspace_root=None, i.e.
    NO confinement at all, from a file nobody could read. It now lands on the
    shipped root. A config the system cannot understand must not be the one
    that grants the most access.
    """
    gate = _gate_from_yaml(tmp_path, None, raw="{ this: is: not: valid: yaml")
    assert gate._within_workspace("/etc/passwd") is False


def test_securitygate_none_still_means_no_confinement():
    """The class-level API is UNCHANGED and that is deliberate.

    39 of SecurityGate's 43 construction sites omit workspace_root — tests,
    and callers confined another way. The defect was never "None disables
    confinement"; it was a CONFIG that merely omits the key landing on None.
    Fixing the class instead of the readers would have been a 39-site change
    with a security meaning nobody asked for.
    """
    from prometheus.permissions.checker import SecurityGate

    assert SecurityGate(workspace_root=None)._within_workspace("/etc/passwd") is True


#: The ONE site allowed to read ``workspace_root`` without the resolver, with
#: its reason. Documents carry their own root and the template states they are
#: not limited by this key; substituting the shipped default there 403s every
#: write. A named, asserted-exact exemption — not a list that can grow quietly.
_RAW_WORKSPACE_READERS: dict[str, str] = {
    "prometheus/web/server.py":
        "Documents service — has its own root; the template says documents "
        "are NOT limited by workspace_root. The shipped default confines the "
        "editor to a directory its root is not under.",
}


def test_workspace_root_readers_are_the_resolver_plus_one_named_exemption():
    """Single resolver. Four separate ``sec.get("workspace_root")`` calls is
    the shape §1b warns about — "six edits and a seventh next quarter".

    Enforced in BOTH directions, like ``KNOWN_UNREAD``: an unlisted raw reader
    fails (new drift), and a listed file that no longer has one fails too
    (stale exemption). So the list cannot grow quietly or rot.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    found = {
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p.name != "shipped_defaults.py"
        for line in p.read_text(encoding="utf-8").splitlines()
        if re.search(r'\.get\(\s*["\']workspace_root["\']', line)
    }
    new = sorted(found - set(_RAW_WORKSPACE_READERS))
    assert not new, (
        "read workspace_root through resolve_workspace_root(), not .get() — "
        "a second reader is a second default:\n  " + "\n  ".join(new)
    )
    stale = sorted(set(_RAW_WORKSPACE_READERS) - found)
    assert not stale, (
        "exemption no longer needed — delete it:\n  " + "\n  ".join(stale)
    )


def test_media_allowlists_have_exactly_one_reader():
    """Same guard for the media keys, and it closes a real hole.

    Mutation M8 — reverting the DAEMON's construction line to
    ``list(_media_cfg.get("allowed_image_types") or [])`` — SURVIVED the
    behavioural tests, because they build a policy the way the daemon does
    rather than driving ``run_daemon`` itself. The resolver was covered; the
    CALL SITE was not (§2d: testing the mechanism is not testing the
    outcome). Driving the real daemon here is not viable, so the regression
    path is closed at the source instead — and this test says plainly that
    that is what it proves.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src"
    pattern = re.compile(
        r'\.get\(\s*["\']allowed_(image|audio|document)_types["\']')
    offenders = [
        f"{p.relative_to(src)}:{i}"
        for p in src.rglob("*.py")
        if p.name != "shipped_defaults.py"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "read the media allowlists through resolve_media_allowlist(), not "
        "`.get(...) or []` — that idiom collapses ABSENT into EXPLICITLY "
        "EMPTY, which media_guard reads as 'no restriction':\n  "
        + "\n  ".join(offenders)
    )


def test_the_documents_exemption_is_the_documented_contract():
    """The exemption is only legitimate because the TEMPLATE says so. If that
    sentence ever leaves the template, the exemption needs re-arguing."""
    from pathlib import Path

    import re

    raw = (Path(__file__).resolve().parent.parent
           / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
    # Un-wrap: strip comment markers and collapse whitespace, so the assertion
    # survives a re-flow of the comment. Pinning the exact line break would
    # make this a test of the formatter, not of the claim (§3c).
    flat = re.sub(r"\s+", " ", raw.replace("#", " "))
    assert "documents, and the artifact outbox have their own confinement" in flat, (
        "the template no longer states that documents are exempt from "
        "workspace_root — the web/server.py exemption needs re-arguing"
    )


# ── gateway.media.allowed_*_types ────────────────────────────────────────────

def _policy_from_config_without_the_keys():
    """The REAL consumer, built the way daemon.py builds it for a gateway
    media section that predates the keys."""
    from prometheus.gateway.config import Platform, PlatformConfig
    from prometheus.gateway.media_guard import MediaPolicy

    from prometheus.config.shipped_defaults import resolve_media_allowlist

    media_cfg: dict = {}
    cfg = PlatformConfig(
        platform=Platform.TELEGRAM,
        token="x",
        allowed_image_types=resolve_media_allowlist(
            media_cfg, "allowed_image_types"),
        allowed_audio_types=resolve_media_allowlist(
            media_cfg, "allowed_audio_types"),
        allowed_document_types=resolve_media_allowlist(
            media_cfg, "allowed_document_types"),
    )
    return MediaPolicy(
        allowed_image_types=tuple(cfg.allowed_image_types),
        allowed_audio_types=tuple(cfg.allowed_audio_types),
        allowed_document_types=tuple(cfg.allowed_document_types),
        max_file_size_mb=cfg.max_file_size_mb,
    )


def test_media_allowlists_absent_still_refuse_a_disallowed_type():
    """BREACH direction. Was: absent -> [] -> ``if allowed and ...`` skipped
    the MIME check entirely, so a pre-#141 install filtered nothing."""
    import pytest as _pytest

    from prometheus.gateway.media_guard import MediaRejected, check_declared_mime

    policy = _policy_from_config_without_the_keys()
    with _pytest.raises(MediaRejected) as exc:
        check_declared_mime("image/bmp", "image", policy)
    # §3b: assert guard IDENTITY. "something refused it" is satisfied by any
    # layer; this test is named after the declared-MIME check.
    assert exc.value.guard_name == "media.mime_declared", (
        f"refused by {exc.value.guard_name}, not the declared-MIME check"
    )


def test_media_allowlists_absent_still_admit_a_legitimate_file():
    """ADMISSION direction — §2c, the one PR #140 shipped without.

    Over-refusal looks exactly like the control working: a fail-closed
    over-correction here would make the Telegram surface silently useless
    while every breach test stayed green.
    """
    from prometheus.gateway.media_guard import check_declared_mime, validate_inbound

    policy = _policy_from_config_without_the_keys()
    jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 32
    check_declared_mime("image/jpeg", "image", policy)
    assert validate_inbound(
        data=jpeg, declared_mime="image/jpeg", kind="image", policy=policy
    ) == "image/jpeg"


def test_explicitly_empty_allowlist_still_means_no_restriction():
    """The documented opt-out must survive the fix.

    The template says "Empty list = no restriction for that kind", and an
    operator who wrote ``[]`` meant it. Absence is not that statement — it is
    a config written before the key existed. The whole fix is keeping those
    two apart; collapsing them the OTHER way would be just as wrong.
    """
    from prometheus.config.shipped_defaults import resolve_media_allowlist

    assert resolve_media_allowlist({"allowed_image_types": []},
                                   "allowed_image_types") == []


def test_explicit_media_allowlist_still_wins():
    from prometheus.config.shipped_defaults import resolve_media_allowlist

    assert resolve_media_allowlist(
        {"allowed_image_types": ["image/png"]}, "allowed_image_types",
    ) == ["image/png"]


def test_platformconfig_default_is_not_degenerate():
    """§1b's under-population shape: a construction site that forgets these
    kwargs must not silently produce "no restriction"."""
    from prometheus.config.shipped_defaults import SHIPPED_ALLOWED_IMAGE_TYPES
    from prometheus.gateway.config import Platform, PlatformConfig

    cfg = PlatformConfig(platform=Platform.TELEGRAM, token="x")
    assert cfg.allowed_image_types == list(SHIPPED_ALLOWED_IMAGE_TYPES)


def test_shipped_media_allowlists_equal_the_template():
    """The constant and the documented default cannot drift — same pin as
    SHIPPED_ALWAYS_LOADED's."""
    from pathlib import Path

    import yaml

    from prometheus.config.shipped_defaults import (
        SHIPPED_ALLOWED_AUDIO_TYPES,
        SHIPPED_ALLOWED_DOCUMENT_TYPES,
        SHIPPED_ALLOWED_IMAGE_TYPES,
        SHIPPED_WORKSPACE_ROOT,
    )

    repo = Path(__file__).resolve().parent.parent
    tpl = yaml.safe_load(
        (repo / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    media = tpl["gateway"]["media"]
    assert list(SHIPPED_ALLOWED_IMAGE_TYPES) == media["allowed_image_types"]
    assert list(SHIPPED_ALLOWED_AUDIO_TYPES) == media["allowed_audio_types"]
    assert list(SHIPPED_ALLOWED_DOCUMENT_TYPES) == media["allowed_document_types"]
    assert SHIPPED_WORKSPACE_ROOT == tpl["security"]["workspace_root"]
