"""Tests for structural entity validation + allow-list (SPRINT MEMORY-1, 2c)."""

from __future__ import annotations

import pytest

from prometheus.memory.entity_validation import (
    classify_entity,
    is_valid_entity,
    load_allowlist,
    quarantine,
)

# Empty allow-list ⇒ pure structural behaviour, independent of any machine's
# ~/.prometheus/entity_allowlist.txt.
EMPTY: set[str] = set()


@pytest.mark.parametrize(
    "name",
    [
        "Will",
        "Dr. Pham",            # internal period must NOT be read as a sentence
        "Mercy Hospital",
        "Salt Lake City",
        "United States of America",
        "claude.ai",           # domain TLD, not a file extension
        "example.ai",
        "AcmeLabs",
        "GPT-4",
        "C#",                  # not shell metachar
        "C++",
        "v0",
        "Kubernetes",
        "Revere Health",
        "en_GB-alan-medium",   # mixed-case ⇒ not flagged as lowercase snake_case
    ],
)
def test_accepts_real_entities(name):
    assert classify_entity(name, allowlist=EMPTY) is None, f"{name!r} should be valid"


@pytest.mark.parametrize(
    "name,reason",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("src/marshmallow/utils.py", "path"),
        ("create a new module src/marshmallow/casing.py", "path"),
        ("utils.py", "filename"),
        ("cell_phone_data_acquisition_blueprint.md", "filename"),
        ("model.gguf", "filename"),
        ("echo $HOME && ls", "code_syntax"),
        ("a | b", "code_syntax"),
        ("timestamp_ms", "code_identifier"),
        ("ensure_text_type", "code_identifier"),
        ("move timedelta_to_microseconds function", "code_identifier"),
        ("fix the bug and update all the callers everywhere", "phrase_too_long"),
    ],
)
def test_rejects_structural_junk(name, reason):
    assert classify_entity(name, allowlist=EMPTY) == reason


def test_allowlist_rescues_structural_junk():
    """A name on the allow-list bypasses the structural gate; off it, it's rejected."""
    # Filename-shaped tool name (the canonical false positive).
    assert classify_entity("llama.cpp", allowlist=EMPTY) == "filename"
    assert classify_entity("llama.cpp", allowlist={"llama.cpp"}) is None
    # snake_case-shaped repo name.
    assert classify_entity("nousresearch_hermes-agent", allowlist=EMPTY) == "code_identifier"
    assert classify_entity("nousresearch_hermes-agent",
                           allowlist={"nousresearch_hermes-agent"}) is None
    # Matching is case-insensitive.
    assert classify_entity("LLaMA.cpp", allowlist={"llama.cpp"}) is None


def test_load_allowlist_from_file(tmp_path):
    f = tmp_path / "entity_allowlist.txt"
    f.write_text(
        "# rescued, real entities that look like junk\n"
        "llama.cpp\n"
        "\n"
        "NousResearch_Hermes-Agent\n",
        encoding="utf-8",
    )
    al = load_allowlist(f)
    assert al == {"llama.cpp", "nousresearch_hermes-agent"}  # lowercased, comment/blank dropped


def test_load_allowlist_absent_is_empty(tmp_path):
    assert load_allowlist(tmp_path / "does-not-exist.txt") == set()


def test_quarantine_writes_inspectable_log(tmp_path):
    log = tmp_path / "q.log"
    quarantine("utils.py", "filename", context="extractor", path=log)
    quarantine("echo $X", "code_syntax", context="wiki_compile", path=log)
    text = log.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert len(lines) == 2
    assert "utils.py" in text and "filename" in text
    assert "code_syntax" in text
    # tab-separated: iso, reason, name, context
    assert lines[0].count("\t") == 3


def test_quarantine_failure_never_raises(tmp_path):
    # Parent path is a file, so the log dir cannot be created — must be
    # swallowed (logged), never raised: quarantining can't break extraction.
    bad = tmp_path / "afile"
    bad.write_text("not a dir")
    quarantine("x", "filename", path=bad / "nested" / "q.log")
