"""Every advertised tool parameter must carry a description.

WHY
---
A parameter with no description is invisible reasoning surface: the model
sees a name and a type and must guess what the flag does and what the default
means. It does not ask — it works around.

Observed cost, from a real turn on 2026-08-12 (telemetry, 18:23:21–18:23:38).
Asked to update a description across the repo and the site, the agent ran
**seven consecutive greps**, each a casing or stem variation of the same
query::

    SYMBIOTE.*experimental → SYMBIOTE → symbiote → Symbiote → symbi
                           → self-modification → self.*modif

then gave up and re-ran the same search through ``bash grep -in``. ``grep``
has had a ``case_sensitive`` parameter the whole time. It carried **no
description** and defaults to ``True``, so permuting the pattern was a more
obvious move than guessing that an undescribed boolean changes matching.

That turn hit ``max_iterations_hit`` nine seconds after its first edit, having
spent its whole budget exploring, and a second turn reported completion with
three of five surfaces still stale — which is the doc reconciliation PR #175
had to do afterwards.

THE SHAPE
---------
This is §1c's neighbourhood — the advertisement is an interface claim, and the
model is its reader on every call — but the failure is an OMISSION rather than
a contradiction. §1c: the example named a parameter that does not exist. Here:
the parameter exists and says nothing about itself.

Only 7 of 151 parameters were undescribed when this landed — 5% — but they sat
in ``bash``, ``grep``, ``glob``, ``write_file`` and ``edit_file``: the
always-loaded core, the five tools used most. Four of the seven changed
behaviour with a default the model had to guess, and
``edit_file.replace_all=False`` is a correctness question, not merely an
efficiency one.
"""

from __future__ import annotations

import pytest

from prometheus.__main__ import create_tool_registry


def _advertised_parameters():
    """(tool, param, spec) for every parameter the model is shown."""
    for schema in create_tool_registry({}, None).list_schemas():
        props = (schema.get("input_schema") or schema.get("parameters") or {}
                 ).get("properties", {})
        for name, spec in props.items():
            yield schema["name"], name, spec


def test_every_advertised_parameter_has_a_description():
    """No allowlist, deliberately. A parameter that genuinely needs no
    explanation still costs one line to say so, and an exemption list here
    would be the hiding place the guard exists to remove."""
    bare = [
        f"{tool}.{param}"
        for tool, param, spec in _advertised_parameters()
        if not (spec.get("description") or "").strip()
    ]
    assert not bare, (
        f"{len(bare)} advertised parameter(s) have no description. The model "
        f"sees a name and a type and guesses — it burned 7 of 25 tool calls "
        f"permuting pattern casing because grep.case_sensitive said nothing "
        f"about itself:\n  " + "\n  ".join(sorted(bare))
    )


def test_a_description_is_more_than_a_restatement_of_the_name():
    """`limit: "limit"` satisfies the letter of the guard above and teaches
    nothing. Cheap check: the description must say something the parameter
    name does not already say."""
    lazy = []
    for tool, param, spec in _advertised_parameters():
        desc = (spec.get("description") or "").strip().lower().rstrip(".")
        flat = param.replace("_", " ").lower()
        if desc and desc in (flat, param.lower()):
            lazy.append(f"{tool}.{param} -> {desc!r}")
    assert not lazy, (
        "these descriptions only restate the parameter name:\n  "
        + "\n  ".join(sorted(lazy))
    )


@pytest.mark.parametrize("tool,param", [
    ("grep", "case_sensitive"),
    ("edit_file", "replace_all"),
    ("write_file", "create_directories"),
    ("grep", "file_glob"),
])
def test_behaviour_changing_flags_state_their_default(tool, param):
    """The four that cost iterations are the ones where the DEFAULT is the
    surprise. A description that explains the flag but not which way it points
    leaves the model exactly as stuck.
    """
    spec = next(s for t, p, s in _advertised_parameters()
                if t == tool and p == param)
    desc = (spec.get("description") or "").lower()
    assert any(w in desc for w in ("default", "by default", "only", "must")), (
        f"{tool}.{param} does not tell the model what happens when it is "
        f"omitted; that is the half that caused the wasted calls: {desc!r}"
    )
