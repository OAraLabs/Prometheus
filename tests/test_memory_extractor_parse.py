"""What the memory extractor accepts as a fact — the model's output format is a negotiation.

The extraction prompt asks for one JSON object per line. A 7B model on Ollama does not
reliably comply: it pretty-prints, wraps the payload in a ```json fence, emits a single
array, or prepends a sentence of prose. #798 is what that cost — every batch recorded
"success", every model call looked fine, and the wiki silently got 0 facts, which is
indistinguishable from "nothing worth remembering".

These tests pin the tolerant parser: every shape a small model actually emits must yield
the same facts the documented newline-delimited shape yields, and a reply that is genuinely
prose must still yield nothing (without warning about it).
"""

from __future__ import annotations

import logging

import pytest

from prometheus.memory.extractor import MemoryExtractor

_NDJSON = (
    '{"entity_type": "person", "entity_name": "张三", "relationship": "colleague",'
    ' "fact": "works at UCloud", "confidence": 0.9, "tags": ["work"]}\n'
    '{"entity_type": "tool", "entity_name": "AstraFlow", "relationship": "uses daily",'
    ' "fact": "an AI dev platform", "confidence": 0.95, "tags": ["ai"]}'
)

_PRETTY = """```json
{
  "entity_type": "person",
  "entity_name": "张三",
  "relationship": "colleague",
  "fact": "works at UCloud",
  "confidence": 0.9,
  "tags": ["work"]
}
{
  "entity_type": "tool",
  "entity_name": "AstraFlow",
  "relationship": "uses daily",
  "fact": "an AI dev platform",
  "confidence": 0.95,
  "tags": ["ai"]
}
```"""


def _names(facts):
    return [f["entity_name"] for f in facts]


# ---------------------------------------------------------------------------
# The documented contract still works
# ---------------------------------------------------------------------------

def test_newline_delimited_json_still_parses():
    facts = MemoryExtractor._parse_facts(_NDJSON)
    assert _names(facts) == ["张三", "AstraFlow"]
    assert facts[0]["fact"] == "works at UCloud"
    assert facts[1]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# The shapes small local models actually emit (#798)
# ---------------------------------------------------------------------------

def test_pretty_printed_objects_parse():
    facts = MemoryExtractor._parse_facts(_PRETTY)
    assert _names(facts) == ["张三", "AstraFlow"]


def test_a_single_json_array_parses():
    raw = (
        '[{"entity_name": "张三", "fact": "works at UCloud"},'
        ' {"entity_name": "AstraFlow", "fact": "an AI dev platform"}]'
    )
    assert _names(MemoryExtractor._parse_facts(raw)) == ["张三", "AstraFlow"]


def test_prose_around_the_objects_is_skipped():
    raw = (
        "Sure, here are the facts I found:\n"
        '{"entity_name": "张三", "fact": "works at UCloud"}\n'
        "Let me know if you need more detail.\n"
        '{"entity_name": "AstraFlow", "fact": "an AI dev platform"}\n'
    )
    assert _names(MemoryExtractor._parse_facts(raw)) == ["张三", "AstraFlow"]


def test_mixed_shapes_in_one_reply_all_parse():
    raw = (
        '{"entity_name": "one", "fact": "first"}\n'
        "```json\n"
        '{\n  "entity_name": "two",\n  "fact": "second"\n}\n'
        "```\n"
        '[{"entity_name": "three", "fact": "third"}]\n'
    )
    assert _names(MemoryExtractor._parse_facts(raw)) == ["one", "two", "three"]


def test_nested_values_inside_a_fact_survive():
    raw = '{"entity_name": "AstraFlow", "fact": "a platform", "tags": ["ai", "cloud"], "meta": {"k": 1}}'
    (fact,) = MemoryExtractor._parse_facts(raw)
    assert fact["tags"] == ["ai", "cloud"]
    assert fact["meta"] == {"k": 1}


def test_objects_missing_required_fields_are_skipped():
    raw = (
        '{"entity_type": "tool", "entity_name": "AstraFlow"}\n'
        '{"entity_type": "tool", "fact": "an AI dev platform"}\n'
        '{"entity_name": "张三", "fact": "works at UCloud"}\n'
    )
    assert _names(MemoryExtractor._parse_facts(raw)) == ["张三"]


# ---------------------------------------------------------------------------
# Nothing to extract is still nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "   \n  ", "No facts worth remembering."])
def test_replies_with_no_json_yield_nothing(raw):
    assert MemoryExtractor._parse_facts(raw) == []


def test_a_lone_opening_brace_does_not_hang_or_raise():
    # raw_decode fails at '{' and the scanner must advance, not loop forever.
    assert MemoryExtractor._parse_facts("{ not json at all {{{") == []


# ---------------------------------------------------------------------------
# The silence half of #798: unparseable JSON-looking output is now visible
# ---------------------------------------------------------------------------

def test_unparseable_json_shaped_output_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="prometheus.memory.extractor"):
        facts = MemoryExtractor._parse_facts('{"entity_name": "张三", "fact": ')
    assert facts == []
    assert "no facts parsed" in caplog.text


def test_a_prose_reply_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="prometheus.memory.extractor"):
        assert MemoryExtractor._parse_facts("Nothing worth remembering here.") == []
    assert caplog.text == ""
