"""The re-record stand-in (scripts/parity/standin.py): strict, with one enumerated tolerance.

A stand-in answers a side of a re-record from the committed exchanges. It must never answer a
request it does not match. The one tolerance is C2 (#593)'s own edit set, on the sides named
for it (repaired_tool_call's alt, hosted_route's hosted): the request is tried with exactly
those edits removed, and served only if that form is byte-identical to a committed request.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from parity.model_server import Exchange  # noqa: E402
from parity.normalize import fingerprint, normalize_request  # noqa: E402
from parity.standin import (  # noqa: E402
    C2_CORE_HEADER,
    C2_SKILL_DESCRIPTION,
    C2_SKILLS_INTRO,
    C2_TOOL_LINE,
    PRE_C2_CORE_HEADER,
    PRE_C2_SKILL_DESCRIPTION,
    StandIn,
    c2_reverted,
)

CORE = "- **commit**: Stage, write, and push a commit."
TOOLS_BEFORE = "- tool_search: Search for tools.\n- memory: Manage memory."
TOOLS_AFTER = f"- tool_search: Search for tools.\n{C2_TOOL_LINE}\n- memory: Manage memory."


def _prompt(*, c2: bool) -> str:
    skills = (f"# Available Skills\n\n{C2_SKILLS_INTRO}\n{C2_CORE_HEADER}\n{CORE}" if c2
              else f"# Available Skills\n\n{PRE_C2_CORE_HEADER}\n{CORE}")
    return f"You are Prometheus.\n\n{TOOLS_AFTER if c2 else TOOLS_BEFORE}\n\n{skills}\n\nBe brief."


def _openai(*, c2: bool, user: str = "read the file 42") -> dict:
    return normalize_request({"model": "alt", "messages": [
        {"role": "system", "content": _prompt(c2=c2)}, {"role": "user", "content": user}]})


def _skill_tool(desc: str) -> dict:
    return {"name": "skill", "description": desc,
            "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}}}


def _anthropic(*, c2: bool) -> dict:
    return normalize_request({
        "model": "hosted", "system": [{"type": "text", "text": _prompt(c2=c2)}],
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": "glob", "description": "Find files.", "input_schema": {}},
                  _skill_tool(C2_SKILL_DESCRIPTION if c2 else PRE_C2_SKILL_DESCRIPTION)]})


# -- the enumerated edits are C2's, byte for byte ----------------------------------------------

def test_the_enumerated_strings_are_the_live_code_s():
    from prometheus.adapter import formatter
    from prometheus.context import prompt_assembler
    from prometheus.tools.builtin.skill import SkillTool

    assert C2_SKILL_DESCRIPTION == SkillTool.description
    src = Path(prompt_assembler.__file__).read_text()
    assert f'lines.append("{C2_CORE_HEADER}")' in src and PRE_C2_CORE_HEADER not in src
    joined = re.sub(r'"\s*\n\s*"', "", src)  # the source splits the sentence over literals
    assert C2_SKILLS_INTRO in joined
    assert "# Available Skills\\n\\n" in src
    assert "f\"- {t['name']}: {t.get('description', '')}\"" in Path(formatter.__file__).read_text()


# -- reverting -----------------------------------------------------------------------------------

def test_reverting_c2_turns_a_c2_request_into_the_committed_one():
    assert fingerprint(_openai(c2=False)) in {fingerprint(v) for v in c2_reverted(_openai(c2=True))}
    assert fingerprint(_anthropic(c2=False)) in {fingerprint(v) for v in c2_reverted(_anthropic(c2=True))}


def test_a_tool_array_that_gained_the_skill_entry_is_reverted_by_dropping_it():
    before = normalize_request({"messages": [{"role": "system", "content": _prompt(c2=False)}],
                                "tools": [{"type": "function", "function": {"name": "glob"}}]})
    after = normalize_request({"messages": [{"role": "system", "content": _prompt(c2=True)}],
                               "tools": [{"type": "function", "function": {"name": "glob"}},
                                         {"type": "function", "function": {
                                             "name": "skill", "description": C2_SKILL_DESCRIPTION}}]})
    assert fingerprint(before) in {fingerprint(v) for v in c2_reverted(after)}


def test_reverting_leaves_everything_else_alone():
    other = _openai(c2=True, user="read the file 43")
    assert fingerprint(_openai(c2=False)) not in {fingerprint(v) for v in c2_reverted(other)}
    # a prompt edited anywhere else is not C2's edit
    edited = json.loads(json.dumps(_openai(c2=True)).replace("Be brief.", "Be very brief."))
    assert fingerprint(_openai(c2=False)) not in {fingerprint(v) for v in c2_reverted(edited)}


# -- the stand-in, end to end over HTTP --------------------------------------------------------

def _exchange(req: dict, upstream: str, reply: str) -> Exchange:
    return Exchange(method="POST", path="/v1/chat/completions", status=200,
                    content_type="application/json", body=json.dumps({"reply": reply}),
                    request=req, upstream=upstream)


def _post(url: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _raw_openai(*, c2: bool, user: str = "read the file 42") -> dict:
    return {"model": "alt", "messages": [{"role": "system", "content": _prompt(c2=c2)},
                                         {"role": "user", "content": user}]}


@pytest.fixture
def standin():
    made = []

    def make(tolerate: bool):
        s = StandIn([_exchange(_openai(c2=False), "alt", "the committed reply")], ["alt"],
                    tolerate={"alt": c2_reverted} if tolerate else None)
        s.start()
        made.append(s)
        return s
    yield make
    for s in made:
        s.stop()


def test_a_strict_stand_in_refuses_a_c2_request(standin):
    s = standin(tolerate=False)
    status, _ = _post(s.url("alt"), _raw_openai(c2=True))
    assert status == 400 and s.summary() == [("alt", False, None)]


def test_a_c2_tolerant_stand_in_serves_the_committed_reply_for_exactly_c2_s_edits(standin):
    s = standin(tolerate=True)
    status, body = _post(s.url("alt"), _raw_openai(c2=True))
    assert status == 200 and "the committed reply" in body
    assert s.summary() == [("alt", True, 0)]


def test_a_c2_tolerant_stand_in_still_refuses_any_other_difference(standin):
    s = standin(tolerate=True)
    status, _ = _post(s.url("alt"), _raw_openai(c2=True, user="read the file 43"))
    assert status == 400 and s.summary() == [("alt", False, None)]


def test_an_unchanged_request_is_served_by_either(standin):
    for tolerant in (False, True):
        s = standin(tolerate=tolerant)
        status, body = _post(s.url("alt"), _raw_openai(c2=False))
        assert status == 200 and "the committed reply" in body


def test_the_tolerance_applies_only_to_the_sides_it_names():
    s = StandIn([_exchange(_openai(c2=False), "alt", "r"), _exchange(_openai(c2=False), "hosted", "r")],
                ["alt", "hosted"], tolerate={"hosted": c2_reverted})
    s.start()
    try:
        assert _post(s.url("alt"), _raw_openai(c2=True))[0] == 400
        assert _post(s.url("hosted"), _raw_openai(c2=True))[0] == 200
    finally:
        s.stop()


def test_the_recorder_never_prints_the_primary_url_or_reads_a_real_key():
    src = (REPO / "scripts" / "parity" / "rerecord.py").read_text()
    assert "ANTHROPIC_API_KEY" not in src and ".bashrc" not in src
    assert "never printed" in src and "print(primary_url" not in src and "{primary_url}" not in src


def test_the_closed_window():
    from datetime import datetime

    from parity.rerecord import _in_closed_window
    assert _in_closed_window("06:15-10:00", datetime(2026, 9, 27, 6, 15))
    assert _in_closed_window("06:15-10:00", datetime(2026, 9, 27, 9, 59))
    assert not _in_closed_window("06:15-10:00", datetime(2026, 9, 27, 6, 14))
    assert not _in_closed_window("06:15-10:00", datetime(2026, 9, 27, 23, 0))


# -- a new scenario: nothing committed of its own --------------------------------------------

def test_a_new_scenario_s_stand_in_answers_the_seed_s_probes_and_no_completion():
    """A scenario recorded for the first time has no committed exchanges. Its stand-in
    answers the alt's boot probes from a recorded golden, and nothing else: a completion
    on a stand-in side was never committed for it, so none may be answered."""
    from parity.model_server import COMPLETIONS_PATHS
    from parity.rerecord import _committed, standin_exchanges

    assert _committed("not_yet_recorded") is None
    own = standin_exchanges("tool_calls")
    assert len(own) == len(_committed("tool_calls")["exchanges"])
    seeded = standin_exchanges("not_yet_recorded", "model_switch")   # its alt DID complete
    assert not [e for e in seeded if e.method == "POST" and e.path in COMPLETIONS_PATHS]
    assert {(e.upstream, e.method, e.path) for e in seeded} >= {
        ("alt", "GET", "/api/tags"), ("alt", "GET", "/api/ps"), ("alt", "POST", "/api/show")}
    with pytest.raises(ValueError, match="standin-from"):
        standin_exchanges("not_yet_recorded")


def test_the_recorder_settles_seeds_before_reading_anything(monkeypatch, capsys):
    """A new scenario without a seed, or a seed for a recorded one, is refused
    before the deploy config is read or a stand-in started."""
    from parity import rerecord

    def unreadable():
        raise AssertionError("the deploy config was read")
    monkeypatch.setattr(rerecord, "DEPLOY_CONFIG", type("P", (), {"read_text": staticmethod(unreadable)}))
    assert rerecord.main(["plain_chat", "--standin-from", "plain_chat=tool_calls"]) == 2
    assert "only for a new scenario" in capsys.readouterr().out
    assert rerecord.main(["no_such_scenario"]) == 2
    real = rerecord._committed
    monkeypatch.setattr(rerecord, "_committed", lambda n: None if n == "plain_chat" else real(n))
    assert rerecord.main(["plain_chat"]) == 2
    assert "--standin-from plain_chat=" in capsys.readouterr().out
    assert rerecord.main(["plain_chat", "--standin-from", "plain_chat=nothing_recorded"]) == 2
    assert "no committed trace to seed" in capsys.readouterr().out
