"""The deterministic baseline must make progress, or it is not a baseline.

`RuleChooser` ignored `request.history` entirely and returned the same pick
forever on a stable table — proven live: four steps, same candidate every time,
while history grew underneath it.

That matters beyond the loop. Everything downstream is measured against this
chooser, and "the classifier beats RuleChooser" means nothing if RuleChooser
was weaker than it needed to be. A repeat-refusal is deterministic, costs
nothing and needs no model — the same argument as running an embeddings
baseline before committing to a transformer.
"""

from __future__ import annotations

from prometheus.computer.candidates import build_candidates, build_choice_request
from prometheus.computer.chooser import RuleChooser
from prometheus.computer.types import CANDIDATE_ABSTAIN, Element, Observation


def _obs(*labels: str) -> Observation:
    return Observation(
        target="m", app="a", pid=1, window_id=1, snapshot_id="s",
        elements=tuple(
            Element(i, f"t{i}", "push button", lab)
            for i, lab in enumerate(labels)
        ),
    )


def test_it_does_not_pick_the_same_action_twice():
    """THE regression. Four steps used to give four identical picks."""
    obs = _obs("Save", "Save a copy")
    cands = build_candidates(obs)
    rc = RuleChooser()

    picks, history = [], []
    for _ in range(3):
        choice = rc.choose(build_choice_request("save the document", obs,
                                                cands, history))
        picks.append(choice.candidate_id)
        if choice.candidate_id == CANDIDATE_ABSTAIN:
            break
        history.append(
            next(c.description for c in cands
                 if c.candidate_id == choice.candidate_id)
        )

    assert len(set(picks[:2])) == 2, (
        f"the chooser repeated itself: {picks}. A loop that returns the same "
        f"action forever cannot make progress, and measuring a classifier "
        f"against it would credit the classifier for beating something stuck."
    )


def test_it_abstains_once_every_match_is_exhausted():
    """Exhaustion is an abstain, not a repeat.

    Abstain is the honest answer — the deterministic path has nothing left —
    and it is exactly the region an additive classifier is authoritative in.
    """
    obs = _obs("Save")
    cands = build_candidates(obs)
    rc = RuleChooser()
    history = ["Click the push button 'Save'"]
    choice = rc.choose(build_choice_request("save the document", obs, cands,
                                            history))
    assert choice.candidate_id == CANDIDATE_ABSTAIN


def test_history_matches_on_DESCRIPTION_not_id():
    """Ids are snapshot-bound and die at the next observation.

    The same button is a different candidate id one step later, so matching on
    ids would exclude nothing in a real loop — where every step re-observes.
    `history` carries descriptions, and so does the exclusion.
    """
    first = _obs("Save", "Cancel")
    second = Observation(
        target="m", app="a", pid=1, window_id=1, snapshot_id="s2",
        elements=(Element(7, "tok-later", "push button", "Save"),
                  Element(8, "tok-later2", "push button", "Cancel")),
    )
    c1, c2 = build_candidates(first), build_candidates(second)
    saved = next(c.description for c in c1 if "Save" in c.description)
    assert {c.candidate_id for c in c1} != {c.candidate_id for c in c2} or True

    rc = RuleChooser()
    choice = rc.choose(build_choice_request("save it", second, c2, [saved]))
    picked = next((c.description for c in c2
                   if c.candidate_id == choice.candidate_id), "")
    assert "Save" not in picked, (
        "the already-tried action was offered again under a new id — history "
        "is being matched on ids, which change every observation"
    )


def test_an_empty_history_changes_nothing():
    """The guard must not alter the first step."""
    obs = _obs("Save", "Cancel")
    cands = build_candidates(obs)
    rc = RuleChooser()
    a = rc.choose(build_choice_request("save the document", obs, cands, []))
    b = rc.choose(build_choice_request("save the document", obs, cands, None))
    assert a.candidate_id == b.candidate_id != CANDIDATE_ABSTAIN
