"""The four states are distinguishable, and the loud two reach the ledger.

The guard in test_config_read_honesty_invariant.py pins the SHAPE — no silent
swallow. This file pins the BEHAVIOUR: that each of the four states is actually
told apart, that UNREADABLE and MALFORMED are errors rather than warnings, and
that every message names the path attempted and the value substituted.

Without the second half, a module could satisfy the guard by logging one
indiscriminate line for all four states — which is the original defect wearing
a log statement.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.config.load import (
    ABSENT,
    LOADED,
    MALFORMED,
    UNREADABLE,
    load_config_file,
)

SUBSTITUTING = "the documented defaults"


@pytest.fixture()
def ledger(monkeypatch):
    """Capture record_silent_failure calls without a real telemetry DB."""
    rows: list[tuple] = []

    class _Tel:
        def record_silent_failure(self, subsystem, operation, exc, context):
            rows.append((subsystem, operation, type(exc).__name__, context))

    monkeypatch.setattr(
        "prometheus.telemetry.tracker.get_telemetry_handle", lambda: _Tel()
    )
    return rows


def test_loaded(tmp_path, ledger):
    p = tmp_path / "c.yaml"
    p.write_text("security:\n  denied_commands: [rm -rf /]\n")
    load = load_config_file(p, subsystem="s", substituting=SUBSTITUTING)
    assert load.state == LOADED and load.ok
    assert load.section("security")["denied_commands"] == ["rm -rf /"]
    assert ledger == [], "a successful read must not write a failure row"


def test_absent_when_nothing_was_specified(caplog, ledger):
    with caplog.at_level(logging.INFO):
        load = load_config_file(None, subsystem="s", substituting=SUBSTITUTING)
    assert load.state == ABSENT and not load.ok
    assert SUBSTITUTING in caplog.text
    assert ledger == [], "ABSENT is legitimate — no ledger row"


def test_absent_when_the_default_path_is_missing(tmp_path, caplog, ledger):
    """explicit=False: nothing was asked for, so a missing file is ABSENT.

    This is the state all eight DEFAULTS_PATH consumers were silently in.
    """
    missing = tmp_path / "nope.yaml"
    with caplog.at_level(logging.WARNING):
        load = load_config_file(missing, subsystem="s",
                                substituting=SUBSTITUTING, explicit=False)
    assert load.state == ABSENT
    assert str(missing) in caplog.text and SUBSTITUTING in caplog.text
    assert ledger == []


def test_unreadable_when_a_path_was_specified(tmp_path, caplog, ledger):
    """explicit=True: a caller named a file. Missing is an ERROR, not absence."""
    missing = tmp_path / "nope.yaml"
    with caplog.at_level(logging.ERROR):
        load = load_config_file(missing, subsystem="s", substituting=SUBSTITUTING)
    assert load.state == UNREADABLE
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert str(missing) in caplog.text and SUBSTITUTING in caplog.text
    assert ledger and ledger[0][0] == "s"
    assert ledger[0][3]["state"] == UNREADABLE


def test_the_same_missing_file_is_absent_or_unreadable_by_who_asked(tmp_path):
    """The distinction the old code could not make, stated directly."""
    missing = tmp_path / "nope.yaml"
    asked = load_config_file(missing, subsystem="s", substituting="d")
    unasked = load_config_file(missing, subsystem="s", substituting="d",
                               explicit=False)
    assert asked.state == UNREADABLE
    assert unasked.state == ABSENT
    assert asked.state != unasked.state


def test_malformed_on_an_empty_file(tmp_path, caplog, ledger):
    """THE FOURTH STATE. safe_load returns None; .get would raise."""
    p = tmp_path / "c.yaml"
    p.write_text("# only a comment\n")
    with caplog.at_level(logging.ERROR):
        load = load_config_file(p, subsystem="s", substituting=SUBSTITUTING)
    assert load.state == MALFORMED
    assert "empty" in caplog.text
    assert load.section("security") == {}, "must not raise AttributeError"
    assert ledger and ledger[0][3]["state"] == MALFORMED


def test_malformed_on_bad_yaml(tmp_path, caplog, ledger):
    p = tmp_path / "c.yaml"
    p.write_text("a: [1,\n")
    with caplog.at_level(logging.ERROR):
        load = load_config_file(p, subsystem="s", substituting=SUBSTITUTING)
    assert load.state == MALFORMED
    assert ledger and ledger[0][3]["state"] == MALFORMED


def test_malformed_on_a_non_mapping(tmp_path, ledger):
    p = tmp_path / "c.yaml"
    p.write_text("- one\n- two\n")
    load = load_config_file(p, subsystem="s", substituting=SUBSTITUTING)
    assert load.state == MALFORMED
    assert load.data == {}


def test_partial_is_recorded_not_inferred(tmp_path, caplog):
    p = tmp_path / "c.yaml"
    p.write_text("context:\n  effective_limit: 72000\n")
    load = load_config_file(p, subsystem="s", substituting=SUBSTITUTING)
    with caplog.at_level(logging.INFO):
        assert load.value("context", "effective_limit", 0) == 72000
        assert load.value("context", "tool_result_max", 4000) == 4000
    assert "tool_result_max" in caplog.text and "4000" in caplog.text
    assert "effective_limit" not in caplog.text, "a present key must be quiet"


def test_a_ledger_failure_never_breaks_the_read(tmp_path, monkeypatch):
    """Telemetry is best-effort; config loading must not depend on it."""
    def _boom():
        raise RuntimeError("telemetry down")

    monkeypatch.setattr(
        "prometheus.telemetry.tracker.get_telemetry_handle", _boom
    )
    load = load_config_file(tmp_path / "nope.yaml", subsystem="s",
                            substituting=SUBSTITUTING)
    assert load.state == UNREADABLE


def test_every_state_names_the_path_and_the_substitute(tmp_path, caplog):
    """The standard the learning/ four already met, applied to all four states."""
    good = tmp_path / "ok.yaml"; good.write_text("a: 1\n")
    empty = tmp_path / "empty.yaml"; empty.write_text("\n")
    missing = tmp_path / "nope.yaml"

    for path, explicit in ((missing, True), (missing, False), (empty, True)):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            load_config_file(path, subsystem="probe",
                             substituting="SENTINEL_VALUE", explicit=explicit)
        assert str(path) in caplog.text, f"{path} not named for explicit={explicit}"
        assert "SENTINEL_VALUE" in caplog.text, (
            f"substitute not named for explicit={explicit}"
        )
