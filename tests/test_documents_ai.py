"""Documents suggest-mode parser — deterministic (no model) tests.

The model wiring (generate_suggestions) is exercised live in Phase C; here we
pin the JSON extraction that turns messy local-model output into edit dicts.
"""

from __future__ import annotations

from prometheus.documents.ai import parse_suggestions


def test_plain_json_object():
    text = '{"edits": [{"find": "a", "replace": "b", "reason": "r"}]}'
    edits = parse_suggestions(text)
    assert edits == [{"find": "a", "replace": "b", "reason": "r"}]


def test_fenced_json():
    text = '```json\n{"edits": [{"find": "x", "replace": "y", "reason": "z"}]}\n```'
    assert parse_suggestions(text) == [{"find": "x", "replace": "y", "reason": "z"}]


def test_prose_wrapped_object_is_extracted():
    text = 'Sure! Here are the edits:\n{"edits":[{"find":"cat","replace":"dog","reason":"swap"}]}\nHope that helps.'
    assert parse_suggestions(text) == [{"find": "cat", "replace": "dog", "reason": "swap"}]


def test_edit_without_find_is_dropped():
    text = '{"edits": [{"replace": "b", "reason": "no find"}, {"find": "ok", "replace": "k"}]}'
    edits = parse_suggestions(text)
    assert edits == [{"find": "ok", "replace": "k", "reason": ""}]


def test_garbage_returns_empty():
    assert parse_suggestions("not json at all") == []
    assert parse_suggestions("") == []
    assert parse_suggestions('{"edits": "not a list"}') == []
