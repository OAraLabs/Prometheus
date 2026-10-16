"""Side calls are unordered and matched by kind; the turn stays strict (parity/sidecalls.py).

The session title is sent once or twice depending on timing, and its telemetry
row lands before or after the turn's own (both shown on unchanged 0.9.12). Pinned
here, against the harness's own matcher and comparer:

* a title before, between or after the turn's requests, and a repeat of it,
  all replay with nothing extra and nothing owed;
* a turn request that changed still fails, and one never made is owed (turn
  requests match exactly as before: tests/test_parity_harness.py pins that an
  exact one matches in any order);
* a side call whose content changed still fails;
* a side call's telemetry row may move or repeat, and any one of a repeated
  call's recorded answers satisfies it; a turn row may not change, and a side
  row the recording never wrote, or a side call never made, still fails.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from parity import compare as cmp  # noqa: E402
from parity.model_server import Exchange, ServerState  # noqa: E402
from parity.runner import RunOutput  # noqa: E402
from parity.scenarios import Evidence, _req_switch  # noqa: E402
from parity.sidecalls import MAIN, SUMMARY, TITLE, TITLE_SYSTEM, request_kind, split_stores  # noqa: E402

PATH = "/v1/chat/completions"
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]


def _main(text: str) -> dict:
    return {"messages": [{"role": "system", "content": "You are Prometheus."}, {"role": "user", "content": text}],
            "tools": TOOLS, "max_tokens": 4096}


def _title(text: str = "Count to three") -> dict:
    return {"messages": [{"role": "system", "content": TITLE_SYSTEM}, {"role": "user", "content": text}],
            "max_tokens": 24}


def _summary(text: str = "Reply with the single word: ready") -> dict:
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 64}


def _state(*requests: dict) -> ServerState:
    st = ServerState(mode="replay")
    for i, req in enumerate(requests):
        st.recorded.append(Exchange(method="POST", path=PATH, status=200, content_type="application/json",
                                    body=json.dumps({"n": i}), request=req, upstream="primary"))
    return st


RECORDED = (_summary(), _main("one"), _title(), _main("two"))


def _replay(st: ServerState, *requests: dict) -> list[tuple[int | None, bool]]:
    return [st.match(PATH, r, "primary") for r in requests]


def test_the_kinds():
    assert request_kind(_main("x")) == MAIN
    assert request_kind(_title()) == TITLE
    assert request_kind(_summary()) == SUMMARY
    assert request_kind({"system": TITLE_SYSTEM, "messages": [{"role": "user", "content": "x"}]}) == TITLE
    # A turn round with its tools withheld is still the turn (it has a system prompt).
    assert request_kind({**_main("x"), "tools": []}) == MAIN


def test_the_titlers_prompt_is_still_the_one_matched():
    src = (REPO / "src" / "prometheus" / "engine" / "session_titles.py").read_text(encoding="utf-8")
    assert f'"{TITLE_SYSTEM}"' in src, "session_titles.py changed its system prompt: update parity/sidecalls.py"


@pytest.mark.parametrize("order", [
    ("summary", "one", "title", "two"),        # as recorded
    ("summary", "one", "two", "title"),        # the title after the second round
    ("title", "summary", "one", "two"),        # before everything
    ("summary", "one", "title", "two", "title"),   # the race: a second, identical title
], ids=["recorded", "title-late", "title-first", "title-twice"])
def test_side_calls_replay_in_any_order_and_any_count(order):
    by = {"summary": _summary(), "one": _main("one"), "two": _main("two"), "title": _title()}
    st = _state(*RECORDED)
    results = _replay(st, *(by[k] for k in order))
    assert all(idx is not None and exact for idx, exact in results), results
    assert st.owed() == []


def test_a_side_call_the_recording_repeated_is_not_owed():
    st = _state(_summary(), _main("one"), _title(), _title(), _main("two"))
    _replay(st, _summary(), _main("one"), _title(), _main("two"))
    assert st.owed() == []


def test_a_changed_turn_request_still_fails():
    st = _state(*RECORDED)
    results = _replay(st, _summary(), _main("one"), _title(), _main("TWO, changed"))
    assert results[-1] == (3, False), "the changed request is diffed against the recorded one"


def test_a_missing_turn_request_is_owed():
    st = _state(*RECORDED)
    _replay(st, _summary(), _main("one"), _title())
    assert st.owed() == [3]


def test_a_changed_side_call_still_fails():
    st = _state(*RECORDED)
    results = _replay(st, _summary(), _main("one"), _title("Something else"), _main("two"))
    assert results[2] == (2, False)


# ---------------------------------------------------------------- stored rows

COLS = ["subsystem", "operation", "input_tokens"]
TURN_ROW = ["agent_loop", "tool_advertisement", None]
TITLE_ROW = ["session_titles", "generate", 82]
DB = "home/.prometheus/telemetry.db"


def _out(rows: list[list]) -> RunOutput:
    return RunOutput(scenario="t", mode="replay", steps=[], stores={
        DB: {"sqlite": {"subsystem_runs": {"columns": COLS, "rows": rows}}}},
        exchanges=[], served=[], unconsumed=[], turns=[], wall_offset_ns=0, rss=[], boot_seconds=None,
        shutdown="clean", daemon_log=Path("/dev/null"))


def _diffs(recorded: list[list], replayed: list[list]) -> dict:
    expected = cmp.expected_from(_out(recorded), Path("/"))
    return cmp.compare(_out(replayed), expected, Path("/")).diffs


def test_a_side_calls_row_may_move_or_repeat():
    first = [["agent_loop", "loop_round", 10], TITLE_ROW, TURN_ROW]
    assert _diffs(first, [["agent_loop", "loop_round", 10], TURN_ROW, TITLE_ROW]) == {}
    assert _diffs(first, [["agent_loop", "loop_round", 10], TITLE_ROW, TURN_ROW, TITLE_ROW]) == {}


def test_a_changed_turn_row_or_side_row_still_fails():
    first = [["agent_loop", "loop_round", 10], TITLE_ROW, TURN_ROW]
    assert _diffs(first, [["agent_loop", "loop_round", 11], TITLE_ROW, TURN_ROW])
    assert _diffs(first, [["agent_loop", "loop_round", 10], ["session_titles", "generate", 99], TURN_ROW])
    # Turn rows keep their order.
    assert _diffs(first, [TURN_ROW, TITLE_ROW, ["agent_loop", "loop_round", 10]])


IDS = ["id", "subsystem", "operation", "input_tokens"]


def _out_ids(rows: list[list]) -> RunOutput:
    out = _out([])
    out.stores = {DB: {"sqlite": {"subsystem_runs": {"columns": IDS, "rows": rows}}}}
    return out


def _diffs_ids(recorded: list[list], replayed: list[list]) -> dict:
    expected = cmp.expected_from(_out_ids(recorded), Path("/"))
    return cmp.compare(_out_ids(replayed), expected, Path("/")).diffs


# Real uuid4 values, minted per run: the normalizer's uuid4 rule turns them into
# <uuid:N> by first appearance, which is what makes a moved row shift the rest.
U1, U2, U3, U4 = (uuid.uuid4().hex for _ in range(4))


def test_ids_after_a_moved_side_row_are_not_a_diff():
    """The normalizer numbers ids in order of appearance: a title row that moves
    renumbers every id after it (seen on the re-recorded compaction golden)."""
    recorded = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 82],
                [U3, "agent_loop", "tool_advertisement", None]]
    replayed = [[U1, "agent_loop", "loop_round", 10], [U3, "agent_loop", "tool_advertisement", None],
                [U2, "session_titles", "generate", 82]]
    assert _diffs_ids(recorded, replayed) == {}


def test_a_repeated_side_row_with_its_own_id_is_not_a_diff():
    """The race writes a second title run with a new id (seen on model_switch)."""
    recorded = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 82],
                [U4, "session_titles", "generate", 82], [U3, "agent_loop", "loop_round", 11]]
    replayed = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 82],
                [U3, "agent_loop", "loop_round", 11]]
    assert _diffs_ids(recorded, replayed) == {}


def test_a_changed_turn_row_with_ids_still_fails():
    recorded = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 82]]
    replayed = [[U1, "agent_loop", "loop_round", 12], [U2, "session_titles", "generate", 82]]
    assert _diffs_ids(recorded, replayed)


def test_a_recording_that_got_two_answers_is_satisfied_by_either():
    """model_switch's live recording asked the title twice and the model answered
    the identical requests differently (2 and 4 output tokens); the replay asks
    once and is answered from one of them."""
    recorded = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 2],
                [U4, "session_titles", "generate", 4], [U3, "agent_loop", "loop_round", 11]]
    for answered in (2, 4):
        replayed = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", answered],
                    [U3, "agent_loop", "loop_round", 11]]
        assert _diffs_ids(recorded, replayed) == {}


def test_a_side_row_the_recording_never_wrote_fails():
    recorded = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 2]]
    replayed = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 7]]
    assert any("never wrote" in ln for ln in _diffs_ids(recorded, replayed).get("telemetry", []))


def test_a_side_call_recorded_but_never_made_fails():
    recorded = [[U1, "agent_loop", "loop_round", 10], [U2, "session_titles", "generate", 2]]
    replayed = [[U1, "agent_loop", "loop_round", 10]]
    assert any("never made" in ln for ln in _diffs_ids(recorded, replayed).get("telemetry", []))


def test_only_the_titlers_subsystem_runs_rows_are_side_rows():
    dump = {"columns": COLS, "rows": [TITLE_ROW, TURN_ROW]}
    main, side = split_stores({DB: {"sqlite": {"tool_calls": dump}},
                               "home/.prometheus/lcm.db": {"sqlite": {"subsystem_runs": dump}}})
    assert side == {}
    assert main[DB]["sqlite"]["tool_calls"]["rows"] == [TITLE_ROW, TURN_ROW]


# ---------------------------------------------------------------- recording


def _evidence(requests: list[dict]) -> Evidence:
    return Evidence(steps=[{"op": "chat", "reply": w} for w in ("one", "two", "three")], stores={},
                    requests=requests, upstream_labels=["primary", "alt"] * (len(requests) // 2 + 1))


def test_a_repeated_title_no_longer_refuses_a_recording():
    reqs = [_main("one"), _title(), _main("two"), _title(), _main("three")]
    assert _req_switch(_evidence(reqs)) == []


def test_a_repeated_turn_request_still_refuses_a_recording():
    reqs = [_main("one"), _main("one"), _main("three")]
    assert any("identical turn request" in why for why in _req_switch(_evidence(reqs)))
