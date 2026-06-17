"""Unit tests for the coding mid-run control channel (Loop Manager Sprint 2).

Pure + IO-light. The point of these is the SAFETY properties: fail-safe parsing (a broken
channel never raises), apply-once injection tracking, and the daemon write round-trip. The
loop is untouched here — wiring + its hard gates are tested separately.
"""

from __future__ import annotations

from prometheus.coding.control import (
    EMPTY,
    Injection,
    RunControl,
    control_path,
    parse_control,
    read_state_for_write,
    serialize_control,
    with_injection,
    with_paused,
    write_state,
)


# ── parse_control — FAIL-SAFE ──────────────────────────────────────────────────

def test_parse_none_and_empty_are_dormant():
    assert parse_control(None) == EMPTY
    assert parse_control("") == EMPTY
    assert parse_control("   ") == EMPTY


def test_parse_malformed_json_is_dormant():
    assert parse_control("{not json") == EMPTY
    assert parse_control("[1,2,3]") == EMPTY  # not a dict
    assert parse_control("null") == EMPTY
    assert parse_control("42") == EMPTY


def test_parse_valid_paused_and_injections():
    st = parse_control('{"paused": true, "injections": [{"id": "a", "text": "do X"}]}')
    assert st.paused is True
    assert st.injections == (Injection(id="a", text="do X"),)


def test_parse_drops_malformed_injection_entries():
    st = parse_control(
        '{"paused": false, "injections": ['
        '{"id": "ok", "text": "keep"},'
        '{"id": "blank", "text": "   "},'   # blank text dropped
        '{"id": 5, "text": "bad id"},'      # non-string id dropped
        '{"text": "no id"},'                # missing id dropped
        '"garbage", null]}'                  # non-dict dropped
    )
    assert st.paused is False
    assert st.injections == (Injection(id="ok", text="keep"),)


def test_parse_missing_keys_defaults():
    assert parse_control("{}") == EMPTY
    assert parse_control('{"paused": true}').paused is True
    assert parse_control('{"injections": "not a list"}').injections == ()


# ── round-trip + pure transitions ──────────────────────────────────────────────

def test_serialize_parse_round_trip():
    st = with_injection(with_paused(EMPTY, True), Injection(id="x", text="hello"))
    assert parse_control(serialize_control(st)) == st


def test_with_paused_and_with_injection_are_pure():
    a = with_paused(EMPTY, True)
    assert a.paused is True and EMPTY.paused is False  # original untouched
    b = with_injection(a, Injection(id="1", text="t1"))
    c = with_injection(b, Injection(id="2", text="t2"))
    assert [i.id for i in c.injections] == ["1", "2"]
    assert len(a.injections) == 0  # earlier states untouched (append is non-mutating)


def test_control_path_derivation(tmp_path):
    assert control_path(tmp_path) == tmp_path / "control.json"


# ── RunControl (the run's reader side) ─────────────────────────────────────────

def test_disabled_runcontrol_is_a_noop():
    rc = RunControl(None)
    assert rc.enabled is False
    assert rc.path is None
    assert rc.read() == EMPTY  # never touches disk


def test_runcontrol_missing_file_is_dormant(tmp_path):
    rc = RunControl(tmp_path)
    assert rc.enabled is True
    assert rc.read() == EMPTY  # no control.json yet → dormant, no raise


def test_runcontrol_reads_written_state(tmp_path):
    write_state(control_path(tmp_path), with_paused(EMPTY, True))
    rc = RunControl(tmp_path)
    assert rc.read().paused is True


def test_runcontrol_corrupt_file_is_failsafe(tmp_path):
    control_path(tmp_path).write_text("{ corrupt json", encoding="utf-8")
    rc = RunControl(tmp_path)
    # A broken control file must NOT raise and must read as dormant — the run continues.
    assert rc.read() == EMPTY


def test_take_new_injections_applies_each_once(tmp_path):
    rc = RunControl(tmp_path)
    state = parse_control(
        '{"paused": false, "injections": ['
        '{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]}'
    )
    first = rc.take_new_injections(state)
    assert [i.id for i in first] == ["a", "b"]
    # Same state re-read (injections still in the file) → nothing new (apply-once).
    assert rc.take_new_injections(state) == []
    # A newly-appended injection IS returned, the old ones still suppressed.
    state2 = with_injection(state, Injection(id="c", text="third"))
    assert [i.id for i in rc.take_new_injections(state2)] == ["c"]


# ── daemon writer round-trip ───────────────────────────────────────────────────

def test_daemon_write_modify_round_trip(tmp_path):
    path = control_path(tmp_path)
    # pause → inject → resume, read-modify-write each step (the endpoint pattern).
    write_state(path, with_paused(read_state_for_write(path), True))
    assert read_state_for_write(path).paused is True
    write_state(path, with_injection(read_state_for_write(path), Injection(id="i1", text="steer")))
    st = read_state_for_write(path)
    assert st.paused is True and st.injections == (Injection(id="i1", text="steer"),)
    write_state(path, with_paused(read_state_for_write(path), False))
    st = read_state_for_write(path)
    assert st.paused is False and len(st.injections) == 1  # resume keeps the queue


def test_read_state_for_write_missing_is_empty(tmp_path):
    assert read_state_for_write(control_path(tmp_path)) == EMPTY
