"""The rule chooser follows the goal's verb, and never guesses between kinds (#667).

THE DEFECT. In the first supervised computer-use test (2026-10-05), the goal
``press Tab in the editor`` clicked the editor's "New tab" button five times.
``RuleChooser`` scored every word of 3+ letters as a SUBSTRING of each
description, so ``Click the button 'New tab'`` matched ``the`` and ``tab``
(2) and ``Press tab`` matched ``press`` and ``tab`` (2). The tie went to the
first row, and click rows are always built first. The goal's verb played no
part.

THE RULES PINNED HERE:

* The goal's leading verb picks the action kind. ``press <key>`` presses a
  key; ``press <anything else>`` presses a button, which is a click;
  ``click``/``tap``/``select`` click; ``type``/``fill``/``set``/``write`` set
  a field. Rows of another kind are not considered, and if no row of the
  right kind matches, the chooser abstains.
* Words match whole, and filler words (``the``, ``in``, ``at``, …) never
  score, so "Click the …" does not match every goal.
* With no recognised verb, a tie between rows of DIFFERENT kinds is not
  broken by list order: the chooser abstains.

The kind comes from the description's own first word ("Click …", "Press …",
"Set …", written by candidates.py), so the chooser still sees only IDs and
descriptions (``Candidate.chooser_view``): no tool name, no arguments.
"""

from __future__ import annotations

import pytest

from prometheus.computer.chooser import RuleChooser
from prometheus.computer.types import CANDIDATE_ABSTAIN, ChoiceRequest

# The rows candidates.py builds, in the order it builds them: clicks and sets
# per element first, then the three key rows.
NEW_TAB = {"id": "click-3", "description": "Click the button 'New tab'"}
OPEN = {"id": "click-1", "description": "Click the button 'Open'"}
SEND = {"id": "click-7", "description": "Click the button 'Send'"}
TEXT = {"id": "set-9", "description": "Set the text area 'Document' to the prepared text, "
                                      "replacing what it holds"}
SEARCH = {"id": "set-4", "description": "Set the text field 'Search' to the prepared text, "
                                        "replacing what it holds"}
KEYS = [
    {"id": "key-return", "description": "Press return"},
    {"id": "key-tab", "description": "Press tab"},
    {"id": "key-escape", "description": "Press escape"},
]


def choose(goal: str, rows: list[dict], chooser: RuleChooser | None = None) -> str:
    req = ChoiceRequest(goal=goal, snapshot_id="s1", candidates=list(rows))
    return (chooser or RuleChooser()).choose(req).candidate_id


class TestTheVerbPicksTheKind:

    def test_press_tab_presses_tab_not_the_new_tab_button(self):
        """The 2026-10-05 regression, row order as the editor produced it."""
        assert choose("press Tab in the editor", [OPEN, NEW_TAB, TEXT, *KEYS]) == "key-tab"

    def test_and_in_any_row_order(self):
        assert choose("press Tab in the editor", [*KEYS, NEW_TAB, OPEN]) == "key-tab"
        assert choose("press Tab in the editor", [NEW_TAB, *reversed(KEYS)]) == "key-tab"

    def test_press_return_and_escape(self):
        assert choose("press Return in the editor", [NEW_TAB, OPEN, *KEYS]) == "key-return"
        assert choose("press escape", [NEW_TAB, *KEYS]) == "key-escape"

    @pytest.mark.parametrize("goal,expected", [
        ("press Enter", "key-return"),
        ("hit Esc", "key-escape"),
        ("hit tab", "key-tab"),
    ])
    def test_key_aliases(self, goal, expected):
        assert choose(goal, [NEW_TAB, *KEYS]) == expected

    def test_press_a_button_is_a_click(self):
        """"press send" means the Send button, as the gate tests have always used it."""
        assert choose("press send", [OPEN, SEND, *KEYS]) == "click-7"

    def test_a_key_with_no_row_abstains_rather_than_clicking(self):
        """space is a key; there is no key-space row; a 'Space' button is not it."""
        space_button = {"id": "click-2", "description": "Click the button 'Space'"}
        assert choose("press space", [space_button, *KEYS]) == CANDIDATE_ABSTAIN

    def test_click_goals_consider_only_clicks(self):
        tab_page = {"id": "click-5", "description": "Click the page tab 'notes'"}
        assert choose("click the notes tab", [*KEYS, tab_page, NEW_TAB]) == "click-5"

    def test_type_goals_consider_only_field_settings(self):
        assert choose("fill search", [OPEN, SEARCH, *KEYS]) == "set-4"
        assert choose("type into the search field", [SEND, SEARCH, *KEYS]) == "set-4"

    def test_a_type_goal_never_clicks(self):
        """The 2026-10-05 type task clicked 'Open' five times. With no matching
        field it abstains: nothing in the window serves the goal."""
        assert choose("type at the end of the file", [OPEN, NEW_TAB, TEXT, *KEYS]) == CANDIDATE_ABSTAIN

    def test_no_row_of_the_kind_abstains(self):
        assert choose("click the Save button", [*KEYS, TEXT]) == CANDIDATE_ABSTAIN


class TestWordsMatchWhole:

    def test_filler_words_never_score(self):
        """Every click row starts 'Click the …'; 'the' alone must match nothing."""
        assert choose("click the thing", [OPEN, NEW_TAB, SEND]) == CANDIDATE_ABSTAIN

    def test_a_word_inside_another_word_does_not_match(self):
        table = {"id": "click-8", "description": "Click the button 'Table'"}
        assert choose("click tab", [table]) == CANDIDATE_ABSTAIN

    def test_quotes_and_case_do_not_hide_a_word(self):
        assert choose("click OPEN", [NEW_TAB, OPEN]) == "click-1"


class TestNoVerbNoGuess:

    def test_a_tie_between_kinds_abstains(self):
        """No recognised verb: 'New tab' (click) and 'Press tab' (key) both
        match 'tab'. List order must not decide which one runs."""
        assert choose("tab", [NEW_TAB, *KEYS]) == CANDIDATE_ABSTAIN
        assert choose("tab", [*KEYS, NEW_TAB]) == CANDIDATE_ABSTAIN

    def test_a_clear_winner_still_wins_without_a_verb(self):
        assert choose("save", [OPEN, {"id": "click-6", "description": "Click the button 'Save'"},
                               *KEYS]) == "click-6"


class TestPreferStillApplies:

    def test_prefer_breaks_nothing_within_the_kind(self):
        """The loop tests drive the chooser with prefer=; it must still choose
        inside the goal's kind."""
        assert choose("press send", [OPEN, SEND, *KEYS], RuleChooser(prefer=("send",))) == "click-7"
        assert choose("fill search", [OPEN, SEARCH, *KEYS],
                      RuleChooser(prefer=("search",))) == "set-4"

    def test_prefer_cannot_pull_in_another_kind(self):
        """prefer('tab') must not make 'press Tab' click the New tab button."""
        assert choose("press Tab in the editor", [NEW_TAB, *KEYS],
                      RuleChooser(prefer=("tab",))) == "key-tab"
