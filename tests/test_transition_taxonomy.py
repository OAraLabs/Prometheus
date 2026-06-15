"""Transition taxonomy — structural classification of repair pairs.

Cases are drawn from the REAL #40 harvest shapes so the classifier matches what
the gym actually produces, plus synthetic cases for the transitions the corpus
is currently MISSING (the ones the breadth sprint must induce).
"""

from __future__ import annotations

from prometheus.learning.transition_taxonomy import (
    TRANSITION_TYPES,
    classify_transition,
    histogram,
)


def _p(name, inp):
    return {"name": name, "input": inp}


# --- dict_wrap_unwrap (the dominant real shape) ---------------------------- #

def test_self_keyed_dict_wrap():
    assert classify_transition(
        "schema_repair",
        _p("grep", {"pattern": "TODO", "root": {"root": "/etc/app"}}),
        _p("grep", {"pattern": "TODO", "root": "/etc/app"}),
    ) == "dict_wrap_unwrap"


def test_foreign_keyed_dict_wrap():
    # task_create {command: {description:…, type:…}} → unwrapped
    assert classify_transition(
        "schema_repair",
        _p("task_create", {"command": {"description": "x", "type": "local_bash"}}),
        _p("task_create", {"description": "x", "type": "local_bash"}),
    ) == "dict_wrap_unwrap"


def test_task_list_status_wrap():
    assert classify_transition(
        "schema_repair",
        _p("task_list", {"status": {"status": "failed"}}),
        _p("task_list", {"status": "failed"}),
    ) == "dict_wrap_unwrap"


# --- the transitions the corpus is MISSING --------------------------------- #

def test_json_stuffed_string():
    assert classify_transition(
        "self_correction",
        _p("task_create", {"command": '{"command":"sleep 1 && ls","description":"x"}'}),
        _p("task_create", {"command": "sleep 1 && ls", "description": "x"}),
    ) == "json_stuffed_string"


def test_fuzzy_rename():
    assert classify_transition(
        "levenshtein_repair",
        _p("task_lists", {"status": "failed"}),
        _p("task_list", {"status": "failed"}),
    ) == "fuzzy_rename"


def test_missing_discriminator():
    assert classify_transition(
        "schema_repair",
        _p("task_create", {"command": "sleep 1"}),
        _p("task_create", {"command": "sleep 1", "type": "local_bash"}),
    ) == "missing_discriminator"


def test_type_coercion():
    assert classify_transition(
        "schema_repair",
        _p("task_list", {"limit": "5"}),
        _p("task_list", {"limit": 5}),
    ) == "type_coercion"


# --- guards ---------------------------------------------------------------- #

def test_dict_wrap_beats_missing_discriminator_ordering():
    # A pair that both nests AND gains a type → dict_wrap wins (it's the repair).
    assert classify_transition(
        "schema_repair",
        _p("task_create", {"command": {"command": "x"}}),
        _p("task_create", {"command": "x", "type": "local_bash"}),
    ) == "dict_wrap_unwrap"


def test_none_or_malformed_is_other():
    assert classify_transition("malformed_empty", None, {"name": "x", "input": {}}) == "other"
    assert classify_transition("x", {"name": "a"}, {"name": "a"}) == "other"


def test_histogram_covers_all_keys_and_counts():
    pairs = [
        ("schema_repair", _p("grep", {"root": {"root": "/x"}}), _p("grep", {"root": "/x"})),
        ("schema_repair", _p("grep", {"root": {"root": "/y"}}), _p("grep", {"root": "/y"})),
        ("levenshtein_repair", _p("task_lists", {}), _p("task_list", {})),
    ]
    h = histogram(pairs)
    assert set(h) == set(TRANSITION_TYPES)  # every bucket present (zeros included)
    assert h["dict_wrap_unwrap"] == 2
    assert h["fuzzy_rename"] == 1
    assert h["other"] == 0
