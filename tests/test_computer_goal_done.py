"""A desktop task ends when its goal is done, not at max_steps (#668).

THE DEFECT. In the first supervised computer-use test (2026-10-05), both
tasks that were not refused repeated one click until the step ceiling:
``Click the button 'New tab'`` × 5 and ``Click the button 'Open'`` × 5. The
task loop (``ComputerTaskRunner._run``) has no notion of a finished goal; it
ends only on a limit, a person, a refusal, a failure or the chooser
abstaining, and ``RuleChooser`` ignored the step history it is handed, so it
chose the step it had just taken, again.

THE RULES PINNED HERE:

* A single-action goal ends once its OWN action runs. A goal whose leading verb
  names one action (``press tab``, ``click save``, ``type …``), with no
  sequencing word (``then``, ``twice``, …), is ``done`` after the first executed
  step of that kind (a click does not complete a fill). A key goal needs the
  SAME key it named: ``press tab`` is met by a Tab press, not an Escape press.
  "Met" means the action RAN; its effect is not verified.
* The RULE chooser never repeats the step it just took (the task loop does
  not forbid repeats; a future model chooser may need one). If its best row is
  the last executed step, it abstains (it does not fall back to the next
  row), so the task ends ``done``: nothing more serves the goal.

These run whole tasks through the real ``ComputerTaskRunner`` on the door
tests' rig (fake driver, real gate and approval channel) and read what was
dispatched and how the task ended.
"""

from __future__ import annotations

import pytest

from prometheus.computer.chooser import RuleChooser, goal_key, is_single_action_goal
from prometheus.computer.types import CANDIDATE_ABSTAIN, ChoiceRequest
from tests.test_computer_door import _answer, _bind, _rig, _start, _verbs


@pytest.fixture(autouse=True)
def _linux_site_rule(monkeypatch):
    """Apply the Linux site rule on every runner, as test_computer_door.py does.

    These tasks rely on the binding covering their clicks and key presses,
    which needs site ``-``. On macOS no extent can be ``-`` (v1.1 drives Linux
    only), so a binding covers nothing, every action waits for an approval
    nobody answers, and the task times out: the 10 failures on CI's
    test-macos (3.14). The rig is borrowed from the door tests; its autouse
    fixture is not, so it is repeated here.
    """
    import sys

    from prometheus.computer import candidates

    monkeypatch.setattr(candidates, "_PLATFORMS_THAT_FLAG_WEB", ("linux", sys.platform))


def _with_rule_chooser(rig):
    rig.runner._chooser_factory = RuleChooser
    return rig


# --------------------------------------------------------------------------- #
# Rule 1: a single-action goal ends after its action
# --------------------------------------------------------------------------- #


class TestSingleActionGoals:

    async def test_press_tab_presses_once_and_is_done(self, tmp_path):
        rig = _rig(tmp_path, ["key-tab"] * 5)
        await _bind(rig)
        task = await _start(rig, goal="press tab")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("press_key", "tab")]
        assert (done.outcome, done.steps) == ("done", 1)

    async def test_click_save_clicks_once_and_is_done(self, tmp_path):
        rig = _rig(tmp_path, ["click-0"] * 5)
        await _bind(rig)
        task = await _start(rig, goal="click save")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("click", "tok-save")]
        assert (done.outcome, done.steps) == ("done", 1)

    async def test_the_regression_goal_end_to_end(self, tmp_path):
        """2026-10-05: 'press Tab in the editor' clicked 'New tab' five times.
        With the real rule chooser it presses Tab once and stops."""
        rig = _with_rule_chooser(_rig(tmp_path, []))
        await _bind(rig)
        task = await _start(rig, goal="press Tab in the editor")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("press_key", "tab")]
        assert (done.outcome, done.steps, done.approvals) == ("done", 1, 0)

    async def test_a_key_goal_is_met_only_by_the_key_it_named(self, tmp_path):
        """'press tab' is not met by an Escape press: that ran, the task goes
        on, and the Tab press that follows ends it."""
        rig = _rig(tmp_path, ["key-escape", "key-tab", "key-tab"])
        await _bind(rig)
        task = await _start(rig, goal="press tab")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("press_key", "escape"), ("press_key", "tab")]
        assert (done.outcome, done.steps) == ("done", 2)

    async def test_a_key_alias_names_the_same_key(self, tmp_path):
        """'press Esc' is met by the escape press the table offers."""
        rig = _rig(tmp_path, ["key-escape", "key-escape"])
        await _bind(rig)
        task = await _start(rig, goal="press Esc")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("press_key", "escape")]
        assert (done.outcome, done.steps) == ("done", 1)

    async def test_a_step_of_another_kind_does_not_end_the_goal(self, tmp_path):
        """A click is not the fill a 'fill' goal asked for: the task goes on
        to the field setting (which asks), and ends after THAT."""
        rig = _rig(tmp_path, ["click-0", "set-2", "click-0"])
        await _bind(rig)
        task = await _start(rig, goal="fill the name", text="x")
        await _answer(rig, ["approve"])
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert [v for v, _ in _verbs(rig.driver)] == ["click", "set_value"]
        assert (done.outcome, done.steps) == ("done", 2)

    async def test_a_sequenced_goal_is_not_cut_short(self, tmp_path):
        rig = _rig(tmp_path, ["key-tab", "key-escape"])
        await _bind(rig)
        task = await _start(rig, goal="press tab then press escape")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("press_key", "tab"), ("press_key", "escape")]
        assert done.steps == 2

    async def test_a_goal_with_no_recognised_verb_is_unchanged(self, tmp_path):
        """'save it' is how the door tests drive multi-step runs."""
        rig = _rig(tmp_path, ["key-tab", "key-escape"])
        await _bind(rig)
        task = await _start(rig, goal="save it")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert done.steps == 2

    def test_which_goals_are_single_action(self):
        for goal in ("press tab", "press Tab in the editor", "click save", "hit Esc",
                     "fill search", "type at the end of the file", "please click OK"):
            assert is_single_action_goal(goal), goal
        for goal in ("save", "save it", "press tab then press escape", "press tab twice",
                     "click next again", "click save and send", "", "zzz"):
            assert not is_single_action_goal(goal), goal


    def test_the_key_a_goal_names(self):
        assert goal_key("press Tab in the editor") == "tab"
        assert goal_key("press Enter") == "return"
        assert goal_key("hit Esc") == "escape"
        for goal in ("press send", "click tab", "type tab", "save", ""):
            assert goal_key(goal) is None, goal


# --------------------------------------------------------------------------- #
# Rule 2: the rule chooser never repeats the step it just took
# --------------------------------------------------------------------------- #

SAVE = {"id": "click-0", "description": "Click the button 'Save'"}
SAVE_AS = {"id": "click-4", "description": "Click the button 'Save as'"}


def _choose(goal, rows, history):
    req = ChoiceRequest(goal=goal, snapshot_id="s1", candidates=list(rows), history=list(history))
    return RuleChooser().choose(req).candidate_id


class TestNoRepeat:

    def test_the_step_just_taken_is_not_chosen_again(self):
        assert _choose("save", [SAVE], ["Click the button 'Save'"]) == CANDIDATE_ABSTAIN

    def test_and_the_next_best_row_is_not_taken_instead(self):
        """Falling to the runner-up would be a guess: 'Save as' is not what
        the goal asked for just because 'Save' was already clicked."""
        assert _choose("save", [SAVE, SAVE_AS], ["Click the button 'Save'"]) == CANDIDATE_ABSTAIN

    def test_an_earlier_step_does_not_block_a_row(self):
        assert _choose("save", [SAVE], ["Click the button 'Save'", "Press tab"]) == "click-0"
        assert _choose("save", [SAVE], []) == "click-0"

    async def test_the_task_loop_itself_does_not_block_a_repeat(self, tmp_path):
        """'Never repeat the last step' is the RULE chooser's rule, not the
        loop's: a chooser that repeats (a future model chooser may need to)
        gets its repeat."""
        rig = _rig(tmp_path, ["click-0", "click-0"])
        await _bind(rig)
        task = await _start(rig, goal="save it")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("click", "tok-save"), ("click", "tok-save")]
        assert done.steps == 2

    async def test_a_sequenced_goal_abstains_under_the_rule_chooser(self, tmp_path):
        """The rule chooser cannot sequence: 'Press tab' and 'Press escape'
        tie, and a tie is never broken by position (#672), so nothing runs."""
        rig = _with_rule_chooser(_rig(tmp_path, []))
        await _bind(rig)
        task = await _start(rig, goal="press tab then press escape")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == []
        assert (done.outcome, done.steps) == ("abstained", 0)

    async def test_a_no_verb_goal_ends_done_after_one_click(self, tmp_path):
        rig = _with_rule_chooser(_rig(tmp_path, []))
        await _bind(rig)
        task = await _start(rig, goal="save")
        done = await rig.runner.wait(task.task_id, timeout=10)
        assert _verbs(rig.driver) == [("click", "tok-save")]
        assert (done.outcome, done.steps) == ("done", 1)
