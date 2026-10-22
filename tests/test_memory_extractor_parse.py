"""What the memory extractor accepts as a fact, and what it says when it finds none.

The extraction prompt asks for one JSON object per line. A 7B model on Ollama does not
reliably comply: it pretty-prints, wraps the payload in a ```json fence, emits a single
array, or prepends a sentence of prose. #798 is what that cost: every batch recorded
"success", every model call looked fine, and the wiki silently got 0 facts, which is
indistinguishable from "nothing worth remembering".

#831 made the parser tolerant; tests/test_extractor_json_parser.py pins the shapes it
named. These pin the rest of the shapes a small model emits, the scanner's
advance-on-failure behaviour, and the other half of #798: JSON-shaped output that
parses to nothing now logs a WARNING, while a prose or well-formed empty reply stays
quiet.

Carried over from #807 (lijinhongucl-pixel), which found the same bug first.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.memory.extractor import MemoryExtractor

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
# The shapes small local models actually emit (#798)
# ---------------------------------------------------------------------------

def test_pretty_printed_objects_in_one_fence_parse():
    facts = MemoryExtractor._parse_facts(_PRETTY)
    assert _names(facts) == ["张三", "AstraFlow"]


def test_a_single_bare_json_array_parses():
    raw = (
        '[{"entity_name": "张三", "fact": "works at UCloud"},'
        ' {"entity_name": "AstraFlow", "fact": "an AI dev platform"}]'
    )
    assert _names(MemoryExtractor._parse_facts(raw)) == ["张三", "AstraFlow"]


def test_prose_around_and_between_the_objects_is_skipped():
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


def test_each_required_field_is_required_on_its_own():
    raw = (
        '{"entity_type": "tool", "entity_name": "AstraFlow"}\n'
        '{"entity_type": "tool", "fact": "an AI dev platform"}\n'
        '{"entity_name": "张三", "fact": "works at UCloud"}\n'
    )
    assert _names(MemoryExtractor._parse_facts(raw)) == ["张三"]


# ---------------------------------------------------------------------------
# Nothing to extract is still nothing
# ---------------------------------------------------------------------------

def test_a_prose_reply_yields_nothing():
    assert MemoryExtractor._parse_facts("No facts worth remembering.") == []


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


def test_objects_with_the_wrong_keys_warn(caplog):
    # Valid JSON, but not facts: the model ignored the schema. Still a format
    # failure, and the one a whole-document json.loads would otherwise hide.
    with caplog.at_level(logging.WARNING, logger="prometheus.memory.extractor"):
        facts = MemoryExtractor._parse_facts('{"name": "张三", "info": "works at UCloud"}')
    assert facts == []
    assert "no facts parsed" in caplog.text


def test_a_prose_reply_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="prometheus.memory.extractor"):
        assert MemoryExtractor._parse_facts("Nothing worth remembering here.") == []
    assert caplog.text == ""


@pytest.mark.parametrize("raw", ["[]", '{"facts": []}', "```json\n[]\n```"])
def test_a_well_formed_empty_reply_does_not_warn(raw, caplog):
    # An empty array, or #831's {"facts": [...]} envelope with nothing in it,
    # is the model correctly saying "nothing here", not a format failure.
    with caplog.at_level(logging.WARNING, logger="prometheus.memory.extractor"):
        assert MemoryExtractor._parse_facts(raw) == []
    assert caplog.text == ""
