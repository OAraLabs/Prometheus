"""A GTK4 menu button is one row, not two that tie (found on the computer-use rig).

GTK4 exposes a menu button as a "button" with no actions wrapping a "toggle button" of the same
name that carries "click". Both were offered, so "click the Main menu" scored them equally and the
rule chooser abstained: no GTK4 menu button could be clicked by name.
"""

from __future__ import annotations

from prometheus.computer.candidates import build_candidates, build_choice_request
from prometheus.computer.chooser import RuleChooser
from prometheus.computer.types import Element, Observation


def _obs() -> Observation:
    return Observation(
        target="box", app="gnome-text-editor", pid=1, window_id=2, snapshot_id="s1",
        elements=(
            Element(1, "tok-1", "generic", ""),
            Element(8, "tok-wrap", "button", "Main menu", actions=(), parent_index=1),
            Element(9, "tok-toggle", "toggle button", "Main menu", actions=("click",), parent_index=8),
            Element(5, "tok-new", "button", "New tab", actions=("click",), parent_index=1),
            # Same name, not nested: two real buttons stay two rows (and still tie).
            Element(11, "tok-a", "button", "Zoom", parent_index=1),
            Element(12, "tok-b", "button", "Zoom", parent_index=1),
        ),
    )


def test_the_wrapper_is_not_offered_and_its_child_is():
    ids = {c.candidate_id for c in build_candidates(_obs())}
    assert "click-8" not in ids
    assert "click-9" in ids
    assert "click-5" in ids


def test_click_the_main_menu_now_picks_the_one_that_acts():
    rows = build_candidates(_obs())
    choice = RuleChooser().choose(build_choice_request("click the Main menu", _obs(), rows))
    assert choice.candidate_id == "click-9"


def test_two_unrelated_buttons_of_one_name_still_tie():
    rows = build_candidates(_obs())
    assert {"click-11", "click-12"} <= {c.candidate_id for c in rows}
    choice = RuleChooser().choose(build_choice_request("click Zoom", _obs(), rows))
    assert choice.candidate_id == "abstain"
