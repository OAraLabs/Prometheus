"""The ``site`` consent term — decided while no grant exists (computer-use v1.1 §5.4).

WHY NOW
-------
In a browser, ``mini:firefox:click:background`` covers a bank tab and a docs
tab alike: "Firefox" was consent to every site in it, signed-in sessions
included. Grants match the WHOLE value exactly and ``from_config_dict`` drops
a row with the wrong term count, so a term cannot be added later without
silently orphaning every stored grant — and a grant minted under the old
meaning can be dropped but never narrowed. The term goes in before the door
starts minting answers to "which app may I use?" (Will, W1: agreed).

THE TERM
--------
``target:app:site:verb:delivery``. ``site`` is an origin, ``-`` (POSITIVELY no
web content), or UNKNOWN (``?``). UNKNOWN is gated and shown but never
rememberable — the same mechanism that makes typed text approve-once — and a
stored row carrying it is refused.

``-`` NEEDS POSITIVE EVIDENCE: a walk that is complete by our OWN evidence
(not degraded, not truncated, every node the driver counted was returned —
cua-driver 0.28.2 hard-codes ``elements_complete=False`` on Linux, so the
driver's own flag cannot be used), no web or document-family node anywhere
in it, an app that is not a browser/Electron/WebView host, and a platform
that can flag web content at all. Anything else is UNKNOWN. The failure
direction is over-prompting, never widening.

AND v1.1 OFFERS NO WEB CONTENT: page elements never reach the table.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from prometheus.computer.actions import (
    ACTION_MODELS, ClickInput, TypeTextInput, schema_for,
)
from prometheus.computer.candidates import build_candidates, site_of
from prometheus.computer.chooser import RuleChooser
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import Element, Observation
from prometheus.permissions.approval_queue import (
    PendingAction, derive_grant, prospective_extents,
)
from prometheus.permissions.checker import Grant, PermissionMode, SecurityGate
from prometheus.permissions.computer_extent import (
    COMPUTER_ACTION_KIND, EXTENT_TERMS, computer_extent_for,
)
from prometheus.permissions.computer_schema import (
    SITE_NONE, SITE_UNKNOWN, declared_site_param,
)

ARGS = {
    "target": "box", "app": "gedit", "pid": 1, "window_id": 2,
    "snapshot_id": "s", "element_token": "t", "delivery_mode": "background",
}


def _extent(model=ClickInput, **over):
    extent, unknown = computer_extent_for(
        "computer_x", {**ARGS, **over}, schema=model.model_json_schema())
    assert unknown is None, unknown
    return extent


# ── THE EXTENT ──────────────────────────────────────────────────────────────

def test_the_extent_is_target_app_site_verb_delivery():
    assert EXTENT_TERMS == 5
    assert _extent(site=SITE_NONE).value == "box:gedit:-:click:background"


def test_every_action_model_declares_the_site():
    for verb, model in ACTION_MODELS.items():
        assert declared_site_param(model.model_json_schema()) == "site", verb


@pytest.mark.parametrize("over", [{}, {"site": ""}, {"site": "garbage"},
                                  {"site": "?"}, {"site": "ftp//x"}])
def test_a_missing_or_unrecognised_site_is_unknown(over):
    extent = _extent(**over)
    assert extent.site == SITE_UNKNOWN
    assert extent.value == "box:gedit:?:click:background"


def test_an_unknown_site_is_never_rememberable_and_says_why():
    extent = _extent()
    assert not extent.rememberable
    assert "web" in extent.why_not_rememberable()
    assert "could not be established" in extent.describe()


def test_no_web_content_is_rememberable_and_reads_that_way():
    extent = _extent(site=SITE_NONE)
    assert extent.rememberable
    assert "no web content" in extent.describe()


def test_an_origin_is_encoded_so_it_cannot_forge_terms():
    extent = _extent(site="HTTPS://Bank.Example:8443")
    assert extent.site == "https%3A//bank.example%3A8443"
    assert len(extent.value.split(":")) == EXTENT_TERMS
    assert extent.rememberable
    assert "https://bank.example:8443" in extent.describe()


def test_a_payload_stays_unrememberable_whatever_the_site():
    extent = _extent(TypeTextInput, site=SITE_NONE, text="hello")
    assert not extent.rememberable


# ── STORED GRANTS ───────────────────────────────────────────────────────────

def test_a_four_term_row_predates_the_site_and_is_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        assert Grant.from_config_dict({
            "kind": COMPUTER_ACTION_KIND,
            "value": "box:firefox:click:background", "tool": "computer_click",
        }) is None
    assert "target:app:site:verb:delivery" in caplog.text


def test_a_stored_unknown_site_row_is_refused():
    assert Grant.from_config_dict({
        "kind": COMPUTER_ACTION_KIND,
        "value": "box:firefox:?:click:background", "tool": "computer_click",
    }) is None


def test_a_five_term_row_loads_and_matches_exactly():
    grant = Grant.from_config_dict({
        "kind": COMPUTER_ACTION_KIND,
        "value": "box:gedit:-:click:background", "tool": "computer_click",
    })
    assert grant is not None
    assert grant.matches("computer_click", None, None,
                         "box:gedit:-:click:background")
    assert not grant.matches("computer_click", None, None,
                             "box:gedit:?:click:background")


def test_a_grant_never_matches_an_unknown_site_even_if_one_was_minted():
    """Defence in depth: neither path can mint one, and if something did it
    still could not match."""
    grant = Grant(kind=COMPUTER_ACTION_KIND,
                  value="box:gedit:?:click:background",
                  tool_name="computer_click")
    assert not grant.matches("computer_click", None, None,
                             "box:gedit:?:click:background")


def test_an_unknown_site_offers_no_lasting_scope_on_any_surface():
    action = PendingAction(request_id="r", tool_name="computer_click",
                           description="d", grant_computer_action=_extent())
    assert derive_grant(action, verb="always") is None
    assert prospective_extents(action) == {}


def test_a_stored_grant_describes_its_site():
    grant = Grant(kind=COMPUTER_ACTION_KIND,
                  value="box:gedit:-:click:background",
                  tool_name="computer_click", scope="persistent")
    assert "no web content" in grant.describe()


# ── THE EVIDENCE RULE FOR `-` ───────────────────────────────────────────────

def _obs(app="gedit", elements=None, **over) -> Observation:
    els = elements if elements is not None else (
        Element(0, "tok-save", "push button", "Save"),
        Element(1, "tok-name", "text", "Name"),
    )
    fields = dict(
        target="box", app=app, pid=1, window_id=2, snapshot_id="s1",
        elements=tuple(els), degraded=False, truncated=False,
        elements_complete=False,  # as 0.28.2 reports it on Linux, always
        total_element_count=len(els), returned_element_count=len(els),
        web_content_seen=False,
    )
    fields.update(over)
    return Observation(**fields)


def test_a_complete_plain_window_has_no_web_content():
    assert site_of(_obs(), platform="linux") == SITE_NONE


def test_the_drivers_own_completeness_flag_is_not_relied_on():
    """0.28.2 hard-codes it false on Linux; using it would make every
    extent UNKNOWN and the binding inert."""
    assert site_of(_obs(elements_complete=False), platform="linux") == SITE_NONE


@pytest.mark.parametrize("over", [
    {"degraded": True},
    {"truncated": True},
    {"total_element_count": 900},
    {"total_element_count": None},
    {"returned_element_count": None},
    {"web_content_seen": True},
    {"web_content_seen": None},
])
def test_any_gap_in_the_evidence_is_unknown(over):
    assert site_of(_obs(**over), platform="linux") == SITE_UNKNOWN


def test_an_observation_with_no_evidence_at_all_is_unknown():
    bare = Observation(target="box", app="gedit", pid=1, window_id=2,
                       snapshot_id="s1",
                       elements=(Element(0, "t", "push button", "Save"),))
    assert site_of(bare, platform="linux") == SITE_UNKNOWN


@pytest.mark.parametrize("app", [
    "firefox", "Firefox", "firefox-esr", "Google Chrome", "chromium-browser",
    "brave-browser", "Microsoft Edge", "epiphany", "code", "Code - OSS",
    "Slack", "discord", "obsidian", "Signal", "electron",
])
def test_a_browser_or_web_host_is_never_no_web_content(app):
    """A floor, not a config key: on Linux the web flag only ever arrives as
    true, so a browser that has not exposed its page shows only unflagged
    chrome and would otherwise pass as a plain app."""
    assert site_of(_obs(app=app), platform="linux") == SITE_UNKNOWN


@pytest.mark.parametrize("platform", ["win32", "darwin", "cygwin"])
def test_a_platform_that_cannot_flag_web_content_is_unknown(platform):
    assert site_of(_obs(), platform=platform) == SITE_UNKNOWN


# ── THE TABLE ───────────────────────────────────────────────────────────────

def test_every_row_carries_the_windows_site():
    obs = _obs()
    for cand in build_candidates(obs, text_to_type="x"):
        assert cand.arguments["site"] == site_of(obs)


def test_page_elements_never_reach_the_table():
    """Cua RFC 4268's rule: web content is an untrusted source. It also
    keeps page-authored labels away from the chooser."""
    obs = _obs(app="firefox", web_content_seen=True, elements=(
        Element(0, "tok-back", "push button", "Back"),
        Element(1, "tok-doc", "document web", "Bank — Transfer"),
        Element(2, "tok-pay", "push button", "Pay now", in_web_content=True,
                parent_index=1),
        Element(3, "tok-link", "link", "Home", parent_index=1),
        Element(4, "tok-field", "entry", "Amount", in_web_content=True),
    ))
    rows = build_candidates(obs, text_to_type="500")
    targets = {c.target_description for c in rows}
    assert "push button 'Back'" in targets
    for gone in ("Pay now", "Home", "Amount"):
        assert not any(gone in t for t in targets), (gone, targets)
    assert {c.arguments["site"] for c in rows} == {SITE_UNKNOWN}


def test_no_remembered_grant_can_cover_a_browser_window():
    obs = _obs(app="firefox", web_content_seen=False)
    for cand in build_candidates(obs, text_to_type="x"):
        verb = cand.tool_name.removeprefix("computer_")
        extent, _ = computer_extent_for(cand.tool_name, cand.arguments,
                                        schema=schema_for(verb))
        assert extent is not None and not extent.rememberable, cand


# ── END TO END THROUGH THE REAL GATE ────────────────────────────────────────

def _loop(obs, grants=()):
    prompted: list = []

    async def approver(tool_name, reason, arguments=None):
        prompted.append(reason)
        return True

    loop = ComputerUseLoop(
        driver=FixtureDriver([obs, obs]), chooser=RuleChooser(prefer=("save",)),
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None,
                          grants=list(grants)),
        approve=approver, skip_preconditions=True)
    return loop, prompted


def _run(loop):
    return asyncio.run(loop.step("save", "box", "gedit", 1, 2))


def test_a_site_grant_covers_the_plain_window_it_was_minted_for(monkeypatch):
    monkeypatch.setattr("prometheus.computer.candidates.sys.platform", "linux")
    grant = Grant(kind=COMPUTER_ACTION_KIND,
                  value="box:gedit:-:click:background",
                  tool_name="computer_click")
    loop, prompted = _loop(_obs(), grants=[grant])
    result = _run(loop)
    assert result.ok and result.extent == "box:gedit:-:click:background"
    assert not prompted


def test_the_same_grant_does_not_cover_a_window_without_evidence(monkeypatch):
    monkeypatch.setattr("prometheus.computer.candidates.sys.platform", "linux")
    grant = Grant(kind=COMPUTER_ACTION_KIND,
                  value="box:gedit:-:click:background",
                  tool_name="computer_click")
    loop, prompted = _loop(_obs(truncated=True), grants=[grant])
    result = _run(loop)
    assert prompted, "a site grant covered a window whose site is unknown"
    assert "no lasting grant is offered" in prompted[0]
    assert result.extent == "box:gedit:?:click:background"
