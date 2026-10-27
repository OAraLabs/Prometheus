"""A field that holds text is never described by that text (found on the computer-use rig).

GTK4 names a text view by its contents. Describing it by its label put the document's text into
the chooser's view, the approval prompt, the persisted action log and every Beacon — including
text a person had typed in an earlier task, which the cockpit promises to keep as a count only.
"""

from __future__ import annotations

from prometheus.computer.candidates import build_candidates, build_choice_request
from prometheus.computer.task import _what
from prometheus.computer.types import Element, Observation

DOC = "Rig note: water the plants on Friday."


def _obs() -> Observation:
    return Observation(
        target="box", app="gnome-text-editor", pid=1, window_id=2, snapshot_id="s1",
        elements=(Element(2, "tok-doc", "document text", DOC),
                  Element(3, "tok-search", "entry", "Search"),
                  Element(4, "tok-open", "button", "Open")),
    )


def test_a_text_box_is_described_by_its_role_alone():
    assert Element(2, "t", "text box", DOC).describe() == "text box"
    assert Element(2, "t", "Document Text", DOC).describe() == "Document Text"
    assert Element(2, "t", "paragraph", DOC).describe() == "paragraph"


def test_a_named_entry_and_a_button_keep_their_names():
    assert Element(3, "t", "entry", "Search").describe() == "entry 'Search'"
    assert Element(4, "t", "button", "Open").describe() == "button 'Open'"


def test_the_document_text_reaches_no_row_no_chooser_view_and_no_prompt():
    rows = build_candidates(_obs(), text_to_type="new text")
    request = build_choice_request("type into the document text", _obs(), rows)
    for row in rows:
        assert DOC not in row.description and DOC not in row.target_description
    assert DOC not in repr(request.candidates)
    prompt = _what("set_value", Element(2, "tok-doc", "text box", DOC), {"text": "new text"})
    assert prompt == "Set the text box to the text below"
    assert any(r.description.startswith("Set the document text to the prepared text") for r in rows)
