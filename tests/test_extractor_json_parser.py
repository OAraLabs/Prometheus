"""Regression tests for model responses containing multi-line JSON facts."""

from __future__ import annotations

import json

import pytest

from prometheus.memory.extractor import MemoryExtractor


FACT_A = {
    "entity_type": "person",
    "entity_name": "Ada",
    "fact": "Ada prefers Python",
    "confidence": 0.9,
}
FACT_B = {
    "entity_type": "concept",
    "entity_name": "Prometheus",
    "fact": "Prometheus is an AI agent",
    "confidence": 0.8,
}


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            json.dumps(FACT_A, indent=2),
            [FACT_A],
        ),
        (
            "Here are the facts:\n\n```json\n"
            + json.dumps([FACT_A, FACT_B], indent=2)
            + "\n```\n",
            [FACT_A, FACT_B],
        ),
        (
            json.dumps(FACT_A) + "\n" + json.dumps(FACT_B),
            [FACT_A, FACT_B],
        ),
        (
            '{"facts": ' + json.dumps([FACT_A, FACT_B], indent=2) + '}',
            [FACT_A, FACT_B],
        ),
    ],
)
def test_parse_facts_accepts_multiline_fenced_and_wrapped_json(raw, expected):
    assert MemoryExtractor._parse_facts(raw) == expected


def test_parse_facts_ignores_invalid_json_and_non_fact_objects():
    raw = (
        "Not JSON\n"
        '{"note": "missing required fields"}\n'
        '{"entity_name": "Ada", "fact": "Ada likes Python"}\n'
        '{"entity_name": "broken",\n'
    )

    assert MemoryExtractor._parse_facts(raw) == [
        {"entity_name": "Ada", "fact": "Ada likes Python"}
    ]


def test_parse_facts_empty_response_returns_empty_list():
    assert MemoryExtractor._parse_facts("  \n ") == []
