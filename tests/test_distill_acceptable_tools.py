"""Tests for scripts/distill/acceptable_tools.py, the judge's tool-choice view (judge.py) and the
acceptable-set wrong-tool rate (score.py). Neutral fixture names, no network, no GPU, no model: the judge is a
scripted fake; the outcome check runs /bin/sh in a tmp git fixture with the jail off."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


score = _load("distill_score", "score.py")
judge = _load("distill_judge", "judge.py")
at = _load("distill_acceptable_tools", "acceptable_tools.py")


@pytest.fixture(autouse=True)
def _home_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The spend ledger is ~/distill/spend-ledger.csv: HOME keeps it in tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))


HOME = "/home/tester"
CWD = HOME + "/proj"


def _tool(name: str, props: dict[str, Any], required: list[str], desc: str = "") -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc or f"The {name} tool.",
                                             "parameters": {"type": "object", "required": required,
                                                            "properties": props}}}


BASH = _tool("bash", {"command": {"type": "string"}, "timeout_seconds": {"type": "integer", "default": 30}},
             ["command"], "Run a command.")
READ = _tool("read_file", {"path": {"type": "string", "x-prometheus-path": "file"},
                           "offset": {"type": "integer", "default": 0},
                           "limit": {"type": "integer", "default": 200}}, ["path"], "Read a file.")
EDIT = _tool("edit_file", {"path": {"type": "string"}, "old_str": {"type": "string"},
                           "new_str": {"type": "string"}}, ["path", "old_str", "new_str"], "Edit a file.")
TTS = _tool("tts", {"text": {"type": "string"}}, [], "Speak text aloud.")
TOOLS = [BASH, READ, EDIT, TTS]


def _case(cid: str, tool: str, args: dict[str, Any], *, prompt: str = "check the notes",
          category: str = "coding", match: str = "exact") -> Any:
    raw = {"id": cid, "set": "eval-tools", "tools": TOOLS,
           "messages": [{"role": "system", "content": f"- Working directory: {CWD}"},
                        {"role": "user", "content": prompt}],
           "expect": {"tool": tool, "args": args, "args_match": match},
           "meta": {"category": category}}
    return score.parse_case(raw, "test")


def _call(name: str, args: Any) -> dict[str, Any]:
    return {"id": "c", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _row(c: Any, calls: list[dict[str, Any]] | None = None, label: str = "m") -> dict[str, Any]:
    s = score.RequestSettings().as_dict()
    return {"set": c.set, "id": c.id, "request_sha": score.request_sha(c, s), "status": "ok",
            "outcome": "completed", "message": {"content": "", "tool_calls": calls or []},
            "latency_s": 0.1, "label": label, "settings": s, "served_model": "f.gguf",
            "server": {"n_ctx": 1024, "total_slots": 1}}


def _rubric(rt: int = 2, va: int = 2, so: int = 2, tp: int = 2, nf: int = 2) -> dict[str, int]:
    return {"right_tool": rt, "valid_args": va, "sensible_order": so, "task_progress": tp, "no_fabrication": nf}


def _verdict_row(item_id: str, rubric: dict[str, int], verdict: str = "pass") -> dict[str, Any]:
    return {"id": item_id, "status": "ok", "view": judge.TOOL_CHOICE_VIEW, "rubric_version": judge.RUBRIC_VERSION,
            "rubric": rubric, "verdict": verdict, "model": "judge-27b", "judge_quant": "UD-Q4_K_XL",
            "judge_sha256": "0" * 64}


# ── judge.py: the tool-choice view ─────────────────────────────────────────


def _tc_item(**over: Any) -> dict[str, Any]:
    item = {
        "id": "cand-1", "label": "candidate", "category": "candidate", "stratum_weight": 1.0,
        "source": {"case": "a", "tool": "read_file"},
        "messages": [
            {"role": "system", "content": "Use read_file to read files. " + "x" * 30000},
            {"role": "user", "content": "what does notes.txt say?"},
            {"role": "assistant", "content": "Looking.", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "notes.txt"},
        ],
        "tools": TOOLS, "candidate": {"name": "read_file", "arguments": {"path": "notes.txt"}},
        "outcome": None,
    }
    item.update(over)
    return item


def test_tool_choice_prompt_shows_context_offered_tools_and_the_candidate_last() -> None:
    text, truncated = judge.render_prompt(_tc_item(), view=judge.TOOL_CHOICE_VIEW)
    assert truncated  # the long system prompt was clipped
    assert text.startswith("BEGIN SESSION DATA") and text.endswith("END SESSION DATA")
    assert "- tts: Speak text aloud." in text and "- edit_file: Edit a file." in text
    assert "[tool result: bash]\nnotes.txt" in text
    assert 'call: bash({"command": "ls"})' in text
    assert "Use read_file to read files." in text and "chars cut" in text
    # the candidate's schema and the call come after the session: items of one case share the rest
    assert text.index("## The session so far") < text.index("## Schema of the candidate's tool") \
        < text.index("## The candidate next step")
    assert text.rstrip().split("\n")[-2] == '{"path": "notes.txt"}'
    assert "Tool: read_file" in text
    sysprompt = judge.system_prompt(judge.TOOL_CHOICE_VIEW)
    assert "acceptable next step" in sysprompt and '"right_tool": n' in sysprompt
    assert judge.system_prompt("outcome") != sysprompt


def test_tool_choice_prompt_drops_old_messages_but_keeps_the_system_prompt() -> None:
    msgs = [{"role": "system", "content": "RULES"}] + [
        {"role": "user", "content": f"turn {i} " + "y" * 3000} for i in range(30)]
    text, truncated = judge.render_prompt(_tc_item(messages=msgs), view=judge.TOOL_CHOICE_VIEW,
                                          max_prompt_chars=20000)
    assert truncated and len(text) <= 20000 + 4000
    assert "RULES" in text and "turn 29" in text and "turn 0 " not in text
    assert "[... older messages dropped ...]" in text


def test_a_candidate_tool_the_case_does_not_offer_is_said_so() -> None:
    text, _ = judge.render_prompt(_tc_item(candidate={"name": "shell", "arguments": "not json"}),
                                  view=judge.TOOL_CHOICE_VIEW)
    assert "(this tool is not among the offered tools)" in text and "not json" in text


class _FakeJudge:
    provider = "local"
    judge_key = "none"
    region = ""
    model = "judge-27b"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def budget_describe(self) -> str:
        return "local judge ($0)"

    async def complete(self, system: str, user: str, *, max_tokens: int) -> Any:
        self.calls.append((system, user))
        return judge.CallResult(text=self.reply, input_tokens=10, output_tokens=5, served_model="judge-27b")


def _write_queue(work: Path, items: list[dict[str, Any]], mode: str | None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / judge.QUEUE_FILE).write_text("".join(json.dumps(it) + "\n" for it in items))
    (work / judge.QUEUE_META_FILE).write_text(json.dumps({"mode": mode} if mode else {}))


def test_a_tool_choice_queue_runs_only_in_its_view_on_the_local_judge(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    work = tmp_path / "judge"
    _write_queue(work, [_tc_item()], judge.MODE_TOOL_CHOICE)
    reply = json.dumps({"verdict": "pass", "reasons": ["fine"], "rubric": _rubric()})
    fake = _FakeJudge(reply)
    base = ["--workdir", str(work), "run", "--provider", "local", "--no-live-probe"]
    # the default (outcome) view on a tool-choice queue
    assert judge.main(base, client_factory=lambda: fake) == judge.EXIT_FATAL
    assert "the tool-choice view judges tool-choice queues" in capsys.readouterr().err
    assert judge.main(["--workdir", str(work), "run", "--provider", "max-payg", "--budget-usd", "1",
                       "--view", "tool-choice"], client_factory=lambda: fake) == judge.EXIT_FATAL
    assert "--provider token-plan" in capsys.readouterr().err  # never the pay-per-token key
    assert not fake.calls
    assert judge.main(base + ["--view", "tool-choice"], client_factory=lambda: fake) == judge.EXIT_DONE
    rows = judge.read_jsonl(work / judge.RESULTS_FILE)
    assert [(r["view"], r["phase"], r["verdict"]) for r in rows] == [
        ("tool-choice", judge.PHASE_TOOL_CHOICE, "pass")]
    assert "file" not in (rows[0]["source"] or {})  # build.py never reads these as trace verdicts
    assert fake.calls[0][0] == judge.TOOL_CHOICE_PROMPT and "## The candidate next step" in fake.calls[0][1]
    other = tmp_path / "trace"
    _write_queue(other, [_tc_item()], None)
    assert judge.main(["--workdir", str(other), "run", "--provider", "local", "--no-live-probe",
                       "--view", "tool-choice"], client_factory=lambda: fake) == judge.EXIT_FATAL
    assert "is a trace queue" in capsys.readouterr().err


class _FakeMax(_FakeJudge):
    provider = judge.TOKEN_PLAN_PROVIDER
    judge_key = judge.TOKEN_PLAN_JUDGE_KEY
    model = judge.JUDGE_MODEL

    def budget_describe(self) -> str:
        return "Token Plan ($0 per call)"


def test_max_on_the_token_plan_judges_a_tool_choice_queue_with_the_same_prompt_byte_for_byte(
        tmp_path: Path) -> None:
    items = [_tc_item(), _tc_item(id="neg-1", label="fail", category="control-negative",
                                  candidate={"name": "tts", "arguments": {"text": "hi"}})]
    reply = json.dumps({"verdict": "pass", "reasons": ["fine"], "rubric": _rubric()})
    local, maxj = _FakeJudge(reply), _FakeMax(reply)
    _write_queue(tmp_path / "local", items, judge.MODE_TOOL_CHOICE)
    _write_queue(tmp_path / "max", items, judge.MODE_TOOL_CHOICE)
    assert judge.main(["--workdir", str(tmp_path / "local"), "run", "--provider", "local", "--no-live-probe",
                       "--view", "tool-choice"], client_factory=lambda: local) == judge.EXIT_DONE
    notify = tmp_path / "distill" / "notify.sh"  # HOME is tmp_path: a Token Plan STOP must reach the operator
    notify.parent.mkdir(exist_ok=True)
    notify.write_text("#!/bin/sh\n")
    notify.chmod(0o700)
    assert judge.main(["--workdir", str(tmp_path / "max"), "run", "--provider", "token-plan",
                       "--view", "tool-choice"], client_factory=lambda: maxj) == judge.EXIT_DONE
    assert sorted(maxj.calls) == sorted(local.calls) and len(maxj.calls) == 2  # system + user, byte for byte
    assert all(sysp == judge.TOOL_CHOICE_PROMPT for sysp, _ in maxj.calls)
    rows = judge.read_jsonl(tmp_path / "max" / judge.RESULTS_FILE)
    assert {(r["provider"], r["judge_key"], r["view"], r["phase"]) for r in rows} == {
        (judge.TOKEN_PLAN_PROVIDER, "token-plan", "tool-choice", judge.PHASE_TOOL_CHOICE)}
    assert all("file" not in (r["source"] or {}) for r in rows)


# ── score.py: the acceptable-set wrong-tool rate ───────────────────────────


def _annotation(cases: list[Any], sets: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    entries = {}
    for c in cases:
        e = {"sha": c.sha, "expected": c.expect["tool"], "acceptable": [c.expect["tool"]], "rejected": [],
             "undecided": [], "tools": {}}
        for k, v in sets.get(c.id, {}).items():
            e[k] = e[k] + v
        entries[c.id] = e
    return {"schema": score.ACCEPTABLE_TOOLS_SCHEMA, "set": "eval-tools", "cases": entries, "complete": True}


def _three() -> tuple[list[Any], dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    cases = [_case("a", "bash", {"command": "cat notes.txt"}), _case("b", "bash", {"command": "ls"}),
             _case("c", "bash", {"command": "pwd"}), _case("d", "bash", {"command": "ls -la"})]
    base = {cases[0].key: _row(cases[0], [_call("bash", {"command": "cat notes.txt"})]),
            cases[1].key: _row(cases[1], [_call("read_file", {"path": "x"})]),
            cases[2].key: _row(cases[2], [_call("bash", {"command": "pwd"})]),
            cases[3].key: _row(cases[3], [_call("bash", {"command": "ls -la"})])}
    cand = {cases[0].key: _row(cases[0], [_call("read_file", {"path": "notes.txt"})]),
            cases[1].key: _row(cases[1], [_call("tts", {"text": "hi"})]),
            cases[2].key: _row(cases[2], [_call("edit_file", {"path": "p", "old_str": "a", "new_str": "b"})]),
            cases[3].key: _row(cases[3], [_call("bash", {"command": "ls -la"})])}
    return cases, base, cand


def test_summarize_splits_wrong_tools_by_the_acceptable_set(tmp_path: Path) -> None:
    cases, base, cand = _three()
    ann = _annotation(cases, {"a": {"acceptable": ["read_file"]}, "b": {"rejected": ["tts", "read_file"]},
                              "c": {"undecided": ["edit_file"]}})
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(ann))
    loaded = score.load_acceptable_tools(path)
    assert len(loaded["file_sha256"]) == 64
    sm = score.summarize(cases, cand, label="cand", acceptable=loaded)
    e = sm["sets"]["eval-tools"]
    assert e["metrics"]["wrong_tool_rate"]["k"] == 3
    assert e["metrics"]["wrong_tool_rate_set"]["k"] == 1        # only b's tts is truly wrong
    assert e["wrong_tool_rate_set_upper"]["k"] == 2             # + c's edit_file, not ruled on yet
    assert e["case_tool_outcome_set"] == {"a": "acceptable_tool", "b": "wrong_tool", "c": "undecided_tool",
                                          "d": "success"}
    assert e["acceptable_tools"]["applied"] and e["acceptable_tools"]["wrong_tool_acceptable"] == 1
    assert sum(e["tool_outcomes_set"].values()) == e["scored"]
    md = score.render_summary_markdown(sm)
    assert "wrong_tool_rate_set" in md and "1 acceptable, 1 wrong, 1 undecided" in md
    # without the annotation nothing changes
    plain = score.summarize(cases, cand, label="cand")["sets"]["eval-tools"]
    assert "wrong_tool_rate_set" not in plain["metrics"] and "acceptable_tools" not in plain


def test_an_annotation_for_other_cases_is_not_applied(tmp_path: Path) -> None:
    cases, _base, cand = _three()
    ann = _annotation(cases, {})
    ann["cases"]["a"]["sha"] = "0" * 64
    del ann["cases"]["b"]
    e = score.summarize(cases, cand, label="cand", acceptable=ann)["sets"]["eval-tools"]
    assert e["acceptable_tools"]["applied"] is False and "1 id(s) missing" in e["acceptable_tools"]["problem"]
    assert "wrong_tool_rate_set" not in e["metrics"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other"}))
    with pytest.raises(score.CaseError):
        score.load_acceptable_tools(bad)


def test_compare_pairs_the_acceptable_set_rate(tmp_path: Path) -> None:
    cases, base, cand = _three()
    ann = _annotation(cases, {"a": {"acceptable": ["read_file"]}, "b": {"rejected": ["tts", "read_file"]},
                              "c": {"rejected": ["edit_file"]}})
    b = score.summarize(cases, base, label="base", acceptable=ann)
    c = score.summarize(cases, cand, label="cand", acceptable=ann)
    pairs = score.headline_pairs(b, c)
    assert pairs["wrong_tool_rate"]["diff_pts"] == 50.0      # 1 -> 3 of 4
    assert pairs["wrong_tool_rate_set"]["diff_pts"] == 25.0  # 1 -> 2 of 4
    md = score.render_compare([b, c], score.evaluate_gate(b, c), markdown=True)
    assert "| wrong-tool rate (acceptable set) | 25.0%" in md and "| wrong-tool rate (acceptable set) | 4 |" in md
    # summaries made without the annotation still compare: an n/a row, no paired row
    b0, c0 = score.summarize(cases, base, label="base"), score.summarize(cases, cand, label="cand")
    md0 = score.render_compare([b0, c0], score.evaluate_gate(b0, c0), markdown=True)
    assert "| wrong-tool rate (acceptable set) | n/a |" in md0
    assert "| wrong-tool rate (acceptable set) | 4 |" not in md0


def test_summarize_cli_takes_the_annotation(tmp_path: Path) -> None:
    cases, _base, cand = _three()
    cf = tmp_path / "cases.jsonl"
    cf.write_text("".join(json.dumps(c.raw) + "\n" for c in cases))
    rf = tmp_path / "res.jsonl"
    rf.write_text("".join(json.dumps(r) + "\n" for r in cand.values()))
    af = tmp_path / "ann.json"
    af.write_text(json.dumps(_annotation(cases, {"a": {"acceptable": ["read_file"]}})))
    out = tmp_path / "s.json"
    assert score.main(["summarize", "--results", str(rf), "--cases", str(cf), "--summary-out", str(out),
                       "--markdown-out", str(tmp_path / "s.md"), "--acceptable-tools", str(af)]) == 0
    assert json.loads(out.read_text())["sets"]["eval-tools"]["metrics"]["wrong_tool_rate_set"]["k"] == 0
    assert score.main(["summarize", "--results", str(rf), "--cases", str(cf), "--summary-out", str(out),
                       "--acceptable-tools", str(tmp_path / "missing.json")]) == 2


# ── acceptable_tools.py: candidates, queue, rulings, annotation ────────────


def test_candidates_are_distinct_wrong_tool_first_calls_on_offered_tools() -> None:
    cases, base, cand = _three()
    unknown = {**cand, cases[3].key: _row(cases[3], [_call("shell", {"command": "ls"})])}
    per = {"base": at.first_calls(cases, base), "cand": at.first_calls(cases, unknown),
           "twin": at.first_calls(cases, {cases[0].key: cand[cases[0].key]})}
    got = at.collect_candidates(cases, per)
    assert sorted((k["case"], k["tool"], tuple(k["labels"])) for k in got.values()) == [
        ("a", "read_file", ("cand", "twin")), ("b", "read_file", ("base",)), ("b", "tts", ("cand",)),
        ("c", "edit_file", ("cand",))]
    assert per["cand"]["d"]["outcome"] == "unknown_tool"  # not offered: never a candidate
    k = got[at.call_key("a", "read_file", {"path": "notes.txt"})]
    assert k["arguments"] == {"path": "notes.txt"}


def test_queue_puts_one_call_per_pair_first_with_controls_then_the_rest() -> None:
    cases = [_case(f"c{i}", "bash", {"command": f"ls {i}"}) for i in range(6)] + [
        _case("e", "edit_file", {"path": "p", "old_str": "a", "new_str": "b"}),
        _case("talk", "bash", {"command": "ls"}, prompt="read the transcript aloud")]
    cands = {}
    for i in range(6):
        for lab, path in (("m27b", f"a{i}"), ("m9b", f"b{i}")):
            key = at.call_key(f"c{i}", "read_file", {"path": path})
            cands[key] = {"key": key, "case": f"c{i}", "tool": "read_file", "arguments": {"path": path},
                          "labels": [lab]}
    q = at.build_queue(cases, cands, positives=3, negatives=3, seed=1, reference_label="m27b")
    ids = [it["id"] for it in q]
    assert len(q) == 12 + 6
    lead = [it for it in q[:12] if it["category"] == "candidate"]
    assert len(lead) == 6 and all(it["source"]["labels"] == ["m27b"] for it in lead)  # the 27B's call first
    assert all(it["source"]["labels"] == ["m9b"] for it in q[-6:])
    pos = [it for it in q if it["category"] == "control-positive"]
    assert pos[0]["source"]["case"] == "e" or any(it["source"]["case"] == "e" for it in pos)  # non-bash first
    assert all(it["candidate"] == {"name": c.expect["tool"], "arguments": c.expect["args"]}
               for it in pos for c in cases if c.id == it["source"]["case"])
    neg = [it for it in q if it["category"] == "control-negative"]
    assert len(neg) == 3 and all(it["candidate"]["name"] == "tts" for it in neg)
    assert all(it["source"]["case"] != "talk" for it in neg)  # a case about speech gets no negative
    assert len(set(ids)) == len(ids)
    assert q == at.build_queue(cases, cands, positives=3, negatives=3, seed=1, reference_label="m27b")


@pytest.mark.parametrize("rubric, want", [
    (_rubric(), True),
    (_rubric(va=0, nf=0), True),       # argument faults don't decide tool choice
    (_rubric(rt=1), False),            # doubtful is not acceptable
    (_rubric(so=0), False),
    (_rubric(tp=0), False),
])
def test_tool_acceptable_rule(rubric: dict[str, int], want: bool) -> None:
    assert at.tool_acceptable(rubric) is want


def _setup_annotation(view_ok: bool = True) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cases, base, cand = _three()
    per = {"base": at.first_calls(cases, base), "cand": at.first_calls(cases, cand)}
    cands = at.collect_candidates(cases, per)
    key = {(k["case"], k["tool"]): k["key"] for k in cands.values()}
    outcomes = {k: {"state": "not_run"} for k in cands}
    outcomes[key[("a", "read_file")]] = {"state": "supports"}
    rulings = {f"cand-{key[('b', 'read_file')]}": {"tool_acceptable": True, "call_acceptable": True, "rubric": _rubric()},
               f"cand-{key[('b', 'tts')]}": {"tool_acceptable": False, "call_acceptable": False,
                                             "rubric": _rubric(rt=0)}}
    ann, per_call = at.build_annotation(cases, cands, outcomes, rulings, judge_ok=view_ok, provenance={})
    return cases, per, ann, per_call


def test_annotation_decides_each_tool_and_records_why() -> None:
    cases, per, ann, per_call = _setup_annotation()
    e = ann["cases"]
    assert e["a"]["acceptable"] == ["bash", "read_file"] and e["a"]["tools"]["read_file"]["by"] == ["outcome"]
    assert e["b"]["acceptable"] == ["bash", "read_file"] and e["b"]["rejected"] == ["tts"]
    assert e["b"]["tools"]["read_file"]["by"] == ["judge"] and e["b"]["tools"]["read_file"]["models"] == ["base"]
    assert e["c"]["undecided"] == ["edit_file"] and ann["complete"] is False and ann["undecided_tools"] == 1
    assert e["d"]["acceptable"] == ["bash"] and e["d"]["tools"] == {"bash": {"decision": "reference",
                                                                           "by": ["reference"]}}
    assert ann["schema"] == score.ACCEPTABLE_TOOLS_SCHEMA and all(x["sha"] for x in e.values())
    rep = at.model_report(cases, per["cand"], ann, per_call)
    assert (rep["old_wrong_tool"]["k"], rep["new_wrong_tool"]["k"], rep["new_wrong_tool_upper"]["k"]) == (3, 1, 2)
    assert (rep["wrong_but_acceptable"], rep["truly_wrong"], rep["undecided"]) == (1, 1, 1)
    assert rep["acceptable_by"] == {"outcome": 1} and rep["acceptable_own_call"] == 1
    base = at.model_report(cases, per["base"], ann, per_call)
    assert base["wrong_but_acceptable"] == 1 and base["new_wrong_tool"]["k"] == 0
    d = at.paired_delta(rep, base, "new")
    assert d["n"] == 4 and d["diff_pts"] == 25.0 and (d["base_rate"], d["candidate_rate"]) == (0.0, 25.0)


def test_a_failed_view_check_leaves_judge_only_tools_undecided() -> None:
    _cases, _per, ann, _ = _setup_annotation(view_ok=False)
    e = ann["cases"]
    assert e["a"]["acceptable"] == ["bash", "read_file"]           # the outcome route still admits
    assert e["b"]["undecided"] == ["read_file", "tts"] and e["b"]["rejected"] == []


def test_agreement_check() -> None:
    queue = ([{"id": f"p{i}", "category": "control-positive", "source": {"case": "a"}} for i in range(10)]
             + [{"id": f"n{i}", "category": "control-negative", "source": {"case": "b"}} for i in range(20)]
             + [{"id": "c0", "category": "candidate", "source": {"case": "a"}}])
    rul = {f"p{i}": {"tool_acceptable": i < 9, "call_acceptable": i < 8} for i in range(10)}
    rul.update({f"n{i}": {"tool_acceptable": i == 0, "call_acceptable": False} for i in range(20)})
    a = at.agreement(queue, rul, {"a": "coding", "b": "other"})
    assert a["control-positive"]["agree"]["k"] == 9 and a["control-negative"]["agree"]["k"] == 19
    assert a["check"] == "pass" and a["ruled"] == 30 and a["control-negative"]["misses"] == ["n0"]
    assert a["control-positive"]["by_case_category"] == {"coding": {"k": 9, "n": 10}}
    rul["n1"]["tool_acceptable"] = True
    assert at.agreement(queue, rul, {})["check"] == "fail"     # 18/20 = 90% < 95%
    assert at.agreement(queue, {}, {})["check"] == "incomplete"


def test_judge_rulings_read_the_latest_ok_tool_choice_row() -> None:
    rows = [_verdict_row("x", _rubric(rt=1), "fail"), {"id": "x", "status": "unparseable"},
            _verdict_row("x", _rubric()), {**_verdict_row("y", _rubric()), "view": "outcome"}]
    r = at.judge_rulings(rows)
    assert set(r) == {"x"} and r["x"]["tool_acceptable"] and r["x"]["call_acceptable"]


# ── the outcome check (arg_equiv's execution tier) ─────────────────────────

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def executor(tmp_path: Path) -> Any:
    src = tmp_path / "src"
    src.mkdir()
    (src / "notes.txt").write_text("one\ntwo\nthree\n")
    (src / "other.txt").write_text("zzz\n")
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "first")
    bare = tmp_path / "store.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(bare))
    ex = at.make_outcome_executor(bare, tmp_path / "work", jail="off")
    yield ex
    ex.cleanup()


@needs_git
@pytest.mark.parametrize("ref, cand, want", [
    (("read_file", {"path": "notes.txt"}), ("bash", {"command": "cat notes.txt"}), "supports"),
    # the same text in the fixture, but read_file stops at 200 lines: the padded snapshot tells
    (("bash", {"command": "cat notes.txt"}), ("read_file", {"path": "notes.txt"}), "differs"),
    (("bash", {"command": "sed -n 1,50p notes.txt"}), ("read_file", {"path": "notes.txt"}), "supports"),
    (("bash", {"command": "head -1 notes.txt"}), ("read_file", {"path": f"{CWD}/notes.txt"}), "supports"),
    (("bash", {"command": "cat notes.txt"}), ("bash", {"command": "head -1 notes.txt"}), "differs"),
    (("bash", {"command": "cat notes.txt"}), ("read_file", {"path": "other.txt"}), "differs"),
    (("bash", {"command": "cat notes.txt"}), ("read_file", {"path": "notes.txt", "limit": 2}), "differs"),
    (("bash", {"command": "cat notes.txt"}), ("read_file", {"path": "missing.txt"}), "candidate_not_ok"),
    (("bash", {"command": "cat missing.txt"}), ("read_file", {"path": "notes.txt"}), "reference_not_ok"),
    (("bash", {"command": "cat notes.txt"}), ("edit_file", {"path": "notes.txt"}), "not_run"),
    (("bash", {"command": "echo hi > out.txt; cat out.txt"}), ("bash", {"command": "echo hi"}), "side_effects"),
])
def test_outcome_check(executor: Any, ref: tuple[str, Any], cand: tuple[str, Any], want: str) -> None:
    env = at.EQ().CaseEnv(cwd=CWD, home=HOME)
    got = at.outcome_check(executor, env, ref[0], ref[1], cand[0], cand[1])
    assert got["state"] == want, got
    assert not any("one\ntwo" in v for v in json.dumps(got).split())  # no output text in the verdict


@needs_git
def test_outcome_executor_keeps_text_in_memory_only_and_restores_the_digest(executor: Any) -> None:
    eq = at.EQ()
    real = eq._digest
    env = eq.CaseEnv(cwd=CWD, home=HOME)
    res = executor.run("bash", {"command": "cat notes.txt"}, env)
    assert res["A"].status == "ok" and eq._digest is real
    assert "two" in executor.text("bash", {"command": "cat notes.txt"}, env, "A")
    assert res["A"].output == real(executor.text("bash", {"command": "cat notes.txt"}, env, "A"))


# ── prepare -> finish -> section, end to end ───────────────────────────────


def _write_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path], list[Any]]:
    cases, base, cand = _three()
    cf = tmp_path / "cases.jsonl"
    cf.write_text("".join(json.dumps(c.raw) + "\n" for c in cases))
    paths = {}
    for label, rows in (("base", base), ("cand", cand)):
        p = tmp_path / f"{label}.jsonl"
        p.write_text("".join(json.dumps({**r, "label": label}) + "\n" for r in rows.values()))
        paths[label] = p
    return cf, paths, cases


def test_prepare_finish_and_section_end_to_end(tmp_path: Path) -> None:
    cf, paths, cases = _write_inputs(tmp_path)
    work = tmp_path / "w"
    out = tmp_path / "evalsets" / "eval-tools.acceptable-tools.json"
    out.parent.mkdir()
    argv = ["prepare", "--work", str(work), "--cases", str(cf), "--reference-label", "base",
            "--results", f"base={paths['base']}", "--results", f"cand={paths['cand']}",
            "--pair", "cand:base", "--display", "cand=Student", "--positives", "2", "--negatives", "2",
            "--annotation-out", str(out)]
    assert at.main(argv) == 0
    assert at.main(argv) == 2  # the queue exists: --force to rewrite
    queue = judge.read_jsonl(work / "judge" / judge.QUEUE_FILE)
    meta = json.loads((work / "judge" / judge.QUEUE_META_FILE).read_text())
    assert meta["mode"] == judge.MODE_TOOL_CHOICE and meta["items"] == len(queue) == 4 + 2 + 2
    cand_doc = json.loads((work / at.CANDIDATES_FILE).read_text())
    assert "arguments" not in json.dumps(cand_doc)  # ids, tools and states only
    # nothing judged yet: everything undecided, the view check incomplete
    rep = at.finish(work, date="2026-11-04")
    assert rep["view_check"] == "incomplete" and not rep["annotation_complete"]
    assert rep["models"]["cand"]["new_wrong_tool_upper"]["k"] == 3
    assert not out.exists()  # beside the eval set only once the judge has ruled
    assert "LOWER bound" in (work / at.REPORT_MD).read_text()
    # the judge's rows: every control right, every candidate acceptable except tts
    rows = []
    for it in queue:
        assert it["id"].startswith(("cand-", "pos-", "neg-"))
        ok = it["category"] == "control-positive" or (it["category"] == "candidate"
                                                       and it["candidate"]["name"] != "tts")
        rows.append({**_verdict_row(it["id"], _rubric() if ok else _rubric(rt=0), "pass" if ok else "fail"),
                     "input_tokens": 100, "output_tokens": 10, "elapsed_s": 1.5, "ts": "2026-11-04T06:30:00Z"})
    (work / "judge" / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = at.finish(work, date="2026-11-04")
    assert rep["view_check"] == "pass" and rep["annotation_complete"]
    m = rep["models"]["cand"]
    assert (m["old_wrong_tool"]["k"], m["new_wrong_tool"]["k"], m["wrong_but_acceptable"]) == (3, 1, 2)
    assert rep["pairs"][0]["new"]["diff_pts"] == 25.0
    assert rep["judge_load"]["rows"] == len(queue) and rep["judge_load"]["input_tokens"] == 100 * len(queue)
    ann = score.load_acceptable_tools(out)
    assert ann["cases"]["b"]["rejected"] == ["tts"] and ann["complete"]
    assert json.loads((work / at.ANNOTATION_FILE).read_text())["cases"] == ann["cases"]
    md = (work / at.SECTION_FILE).read_text()
    assert md.startswith("## Acceptable-tool wrong-tool rate (added 2026-11-04)")
    assert "| Student | 4 |" in md and "Result: **pass**" in md and "notes.txt" not in md
    # the scorecard placeholder is replaced, the markers kept
    sc = tmp_path / "scorecard.md"
    sc.write_text(f"# Card\n\n{at.SECTION_BEGIN}\nPENDING\n{at.SECTION_END}\n\ntail\n")
    assert at.main(["section", "--report", str(work / at.REPORT_FILE), "--scorecard", str(sc),
                    "--date", "2026-11-04"]) == 0
    text = sc.read_text()
    assert "PENDING" not in text and text.count(at.SECTION_BEGIN) == 1 and text.endswith("tail\n")
    assert "## Acceptable-tool wrong-tool rate (added 2026-11-04)" in text
    # a results file that changed since prepare is refused
    paths["cand"].write_text(paths["cand"].read_text() + "\n")
    with pytest.raises(SystemExit, match="changed since prepare"):
        at.finish(work)


def _judge_all(work: Path, provider: str, model: str) -> list[dict[str, Any]]:
    """Rows for every queued item: controls right, every candidate acceptable except tts."""
    rows = []
    for it in judge.read_jsonl(work / "judge" / judge.QUEUE_FILE):
        ok = it["category"] == "control-positive" or (it["category"] == "candidate"
                                                       and it["candidate"]["name"] != "tts")
        rows.append({**_verdict_row(it["id"], _rubric() if ok else _rubric(rt=0), "pass" if ok else "fail"),
                     "provider": provider, "model": model, "judge_quant": None, "judge_sha256": None,
                     "input_tokens": 1000, "cached_input_tokens": 400, "output_tokens": 50, "elapsed_s": 2.0,
                     "ts": "2026-11-06T16:00:00Z"})
    return rows


def test_a_max_judged_work_dir_is_tagged_max_derived_and_never_replaces_the_local_copy(tmp_path: Path) -> None:
    cf, paths, _ = _write_inputs(tmp_path)
    work = tmp_path / "w-max"
    evalsets = tmp_path / "evalsets"
    evalsets.mkdir()
    local_copy = evalsets / "eval-tools.acceptable-tools.json"
    local_copy.write_text("{}\n")  # the local judge's copy: never replaced by a max-derived one
    base = ["prepare", "--work", str(work), "--cases", str(cf), "--reference-label", "base",
            "--results", f"base={paths['base']}", "--results", f"cand={paths['cand']}", "--pair", "cand:base",
            "--display", "cand=Student", "--positives", "2", "--negatives", "2"]
    assert at.main(base + ["--annotation-out", str(local_copy)]) == 0
    rows = _judge_all(work, judge.TOKEN_PLAN_PROVIDER, judge.JUDGE_MODEL)
    (work / "judge" / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = at.finish(work, date="2026-11-06")
    assert rep["view_check"] == "pass" and rep["max_derived"] and rep["tags"] == ["max-derived"]
    assert local_copy.read_text() == "{}\n" and rep["annotation_copy"] is None
    for name in (at.ANNOTATION_FILE, at.AGREEMENT_FILE, at.REPORT_FILE):
        doc = json.loads((work / name).read_text())
        assert doc["tags"] == ["max-derived"] and doc["max_derived"] is True, name
    assert rep["judge"] == [{"model": judge.JUDGE_MODEL, "quant": None, "sha256": None,
                             "provider": judge.TOKEN_PLAN_PROVIDER}]
    assert (rep["judge_load"]["mean_input_tokens"], rep["judge_load"]["mean_output_tokens"]) == (1000, 50)
    md = (work / at.SECTION_FILE).read_text()
    assert f"Qwen Max (`{judge.JUDGE_MODEL}`, Token Plan" in md and "4090 Qwen3.8-27B" not in md
    assert "a mean of 1,000 in and 50 out per item" in md and "Load on the shared server" not in md
    assert "`~/distill/w-max/`" in md and "tagged max-derived, private" in md
    # a copy beside the eval set only to a name that says max-derived
    prep = json.loads((work / at.PREPARE_FILE).read_text())
    prep["annotation_out"] = str(evalsets / "eval-tools.acceptable-tools.max-derived.json")
    (work / at.PREPARE_FILE).write_text(json.dumps(prep))
    rep = at.finish(work, date="2026-11-06")
    assert rep["annotation_copy"] == "eval-tools.acceptable-tools.max-derived.json"
    assert score.load_acceptable_tools(evalsets / rep["annotation_copy"])["max_derived"] is True
    # one judge per work dir: a local row among Max's is refused
    extra = {**rows[0], "provider": "local", "model": "judge-27b"}
    with open(work / "judge" / judge.RESULTS_FILE, "a") as fh:
        fh.write(json.dumps(extra) + "\n")
    with pytest.raises(SystemExit, match="more than one judge"):
        at.finish(work)


def test_a_partial_report_prints_no_bound_in_bold(tmp_path: Path) -> None:
    cf, paths, _ = _write_inputs(tmp_path)
    work = tmp_path / "w"
    assert at.main(["prepare", "--work", str(work), "--cases", str(cf), "--results", f"base={paths['base']}",
                    "--results", f"cand={paths['cand']}", "--pair", "cand:base", "--positives", "2",
                    "--negatives", "2"]) == 0
    rep = at.finish(work, date="2026-11-06")  # nothing judged: every candidate tool undecided
    assert not rep["annotation_complete"]
    md = (work / at.REPORT_MD).read_text()
    assert "LOWER bound" in md and "**" not in md


def test_prepare_refuses_a_different_case_set(tmp_path: Path) -> None:
    cf, paths, _ = _write_inputs(tmp_path)
    assert at.main(["prepare", "--work", str(tmp_path / "w"), "--cases", str(cf), "--results",
                    f"base={paths['base']}", "--expect-cases-sha256", "deadbeef"]) == 2


def test_night_refuses_a_server_that_is_not_the_standard_profile(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    cf, paths, _ = _write_inputs(tmp_path)
    work = tmp_path / "w"
    assert at.main(["prepare", "--work", str(work), "--cases", str(cf), "--results", f"base={paths['base']}",
                    "--positives", "1", "--negatives", "0"]) == 0

    class OneSlot:
        slots = 1

        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def preflight(self) -> None:
            pass

    assert "not the standard profile" in at.server_check("http://x", "k", 4, factory=OneSlot)
    monkeypatch.setattr(at, "server_check", lambda *a, **k: "1 slot(s)")
    sent: list[list[str]] = []
    monkeypatch.setattr(at, "notify", lambda cmd, lines, execute: sent.append(lines))
    assert at.main(["night", "--work", str(work), "--judge-url", "http://x", "--judge-key-file", "k",
                    "--until", "04:55"]) == 3
    assert sent and "not run tonight" in sent[0][0]
    monkeypatch.setattr(at, "server_check", lambda *a, **k: None)
    assert at.main(["night", "--work", str(work), "--judge-url", "http://x", "--judge-key-file", "k",
                    "--until", "04:55"]) == 0  # a dry run: prints the judge command, judges nothing
    assert not (work / "judge" / judge.RESULTS_FILE).exists()


def test_night_judges_the_before_queues_first_then_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cf, paths, _ = _write_inputs(tmp_path)
    work = tmp_path / "w"
    assert at.main(["prepare", "--work", str(work), "--cases", str(cf), "--results", f"base={paths['base']}",
                    "--positives", "1", "--negatives", "0"]) == 0
    r2 = tmp_path / "r2"
    _write_queue(r2, [{"id": "r2p-1"}, {"id": "r2p-2"}], judge.MODE_LOCAL_JUDGE)
    monkeypatch.setattr(at, "server_check", lambda *a, **k: None)
    sent: list[list[str]] = []
    monkeypatch.setattr(at, "notify", lambda cmd, lines, execute: sent.append(lines))
    ran: list[list[str]] = []

    def fake_run(argv: list[str], until: Any, env: Any) -> int:
        ran.append(argv)
        wdir = Path(argv[argv.index("--workdir") + 1])
        view = argv[argv.index("--view") + 1]
        q = judge.read_jsonl(wdir / judge.QUEUE_FILE)
        rows = [{**_verdict_row(it["id"], _rubric()), "view": view} for it in q]
        (wdir / judge.RESULTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return 0

    monkeypatch.setattr(at, "run_judge_until", fake_run)
    argv = ["night", "--work", str(work), "--judge-url", "http://x", "--judge-key-file", "k", "--until", "04:50",
            "--before", f"{r2}=03:15", "--concurrency", "2", "--execute"]
    assert at.main(argv) == 0
    assert [Path(a[a.index("--workdir") + 1]) for a in ran] == [r2, (work / "judge").resolve()]
    assert [a[a.index("--view") + 1] for a in ran] == ["outcome", judge.TOOL_CHOICE_VIEW]
    assert all(a[a.index("--concurrency") + 1] == "2" for a in ran)
    assert sent[-1][0].startswith("4090 judge slot: r2 done 2/2") and "acceptable-tools done" in sent[-1][0]
    assert (work / at.REPORT_FILE).exists()
    ran.clear()
    assert at.main(argv) == 0 and not ran  # everything ruled and reported: nothing to do
    with pytest.raises(SystemExit, match="WORKDIR=HH:MM"):
        at.main([*argv[:-1], "--before", "nodeadline"])
