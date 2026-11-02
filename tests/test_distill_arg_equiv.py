"""Tests for scripts/distill/arg_equiv.py (argument equivalence) and the tool
choice metrics score.py reports with it. Neutral fixture names, no network,
no GPU; the execution tests run /bin/sh in a tmp git fixture with the jail off."""

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
eq = _load("distill_arg_equiv", "arg_equiv.py")

HOME = "/home/tester"
CWD = HOME + "/proj"
ENV = eq.CaseEnv(cwd=CWD, home=HOME)

BASH = {"type": "function", "function": {"name": "bash", "description": "Run a command.", "parameters": {
    "type": "object", "required": ["command"],
    "properties": {"command": {"type": "string"},
                   "cwd": {"type": "string", "default": None},
                   "timeout_seconds": {"type": "integer", "default": 30, "minimum": 1}}}}}
READ = {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
    "type": "object", "required": ["path"],
    "properties": {"path": {"type": "string", "x-prometheus-path": "file"},
                   "offset": {"type": "integer", "default": 0},
                   "limit": {"type": "integer", "default": 200}}}}}
GLOB = {"type": "function", "function": {"name": "glob", "description": "Find files.", "parameters": {
    "type": "object", "required": ["pattern"],
    "properties": {"pattern": {"type": "string"},
                   "root": {"type": "string", "default": None, "x-prometheus-path": "dir"}}}}}
WRITE = {"type": "function", "function": {"name": "write_file", "description": "Write a file.", "parameters": {
    "type": "object", "required": ["path", "content"],
    "properties": {"path": {"type": "string", "x-prometheus-path": "file"},
                   "content": {"type": "string"}}}}}
LOOKUP = {"type": "function", "function": {"name": "lookup", "description": "Look a thing up.", "parameters": {
    "type": "object", "properties": {"key": {"type": "string"}, "limit": {"type": "integer"}}}}}
TOOLS = [BASH, READ, GLOB, WRITE, LOOKUP]
SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}


def check(tool: str, expected: dict[str, Any], got: Any, **kw: Any) -> Any:
    return eq.check_call(tool, expected, got, SCHEMAS[tool], ENV, **kw)


def sh(a: str, b: str) -> bool:
    return check("bash", {"command": a}, {"command": b}).equivalent is True


# ── canonical tier ─────────────────────────────────────────────────────────

def test_strict_eq_matches_score_strict_eq():
    samples = [1, 1.0, True, False, 0, "1", None, [1, 2], [1.0, 2], {"a": 1}, {"a": True}, {"a": 1.5}]
    for a in samples:
        for b in samples:
            assert eq.strict_eq(a, b) == score.strict_eq(a, b), (a, b)


def test_exact_is_its_own_tier():
    v = check("bash", {"command": "ls"}, {"command": "ls"})
    assert (v.equivalent, v.tier) == (True, "exact")


@pytest.mark.parametrize("a, b", [
    ("ls -la  /tmp", "ls   -la /tmp"),                 # whitespace between words
    ("cat 'a b.txt'", 'cat "a b.txt"'),                 # quoting style
    ("cat README.md", "cat 'README.md'"),                # quoting a plain word
    ("ls -la", "ls -al"),                                # boolean flag order
    ("ls -l -a src", "ls -al src"),                      # split vs clustered flags
    ("grep -n -i word f.txt", "grep -in word f.txt"),
    ("cat ~/notes.txt", f"cat {HOME}/notes.txt"),        # tilde expansion
    ("ls;", "ls"),                                       # trailing separator
    (f"cd {CWD} && ls", "ls"),                           # cd to the case's own cwd
    ("cd . && ls", "ls"),
    ("cd sub && ls", f"cd {CWD}/sub && ls"),             # cd target resolved
    ("echo a | wc -l", "echo a|wc -l"),
    ("cat x 2>/dev/null", "cat x 2> /dev/null"),
])
def test_shell_identities_are_canonical(a, b):
    v = check("bash", {"command": a}, {"command": b})
    assert (v.equivalent, v.tier) == (True, "canonical"), v


@pytest.mark.parametrize("a, b", [
    ("ls *.py", "ls '*.py'"),            # a glob, quoted, is a literal
    ("echo ~", "echo '~'"),
    ("ls -lt", "ls -tl"),                # not in the boolean set (ls sort flags: last wins)
    ("ls -1l", "ls -l1"),
    ("head -n 5 f", "head -5 f"),        # options with values are never rewritten
    ("a && b", "a ; b"),
    ("a | b", "b | a"),
    ("cd sub && ls", "ls"),
    ("cat x 2>/dev/null", "cat x 2>&1"),
    ("echo $HOME", "echo /home/tester"),  # an expansion: only surrounding whitespace is canonical
    ("grep -e x f", "grep x -e f"),
    ("ls src/", "ls src"),               # trailing slash in a shell word is kept
])
def test_shell_differences_are_never_canonical(a, b):
    v = check("bash", {"command": a}, {"command": b})
    assert v.tier != "canonical" and v.equivalent is not True, v


def test_unlexable_commands_compare_as_trimmed_text():
    assert sh("  echo $X  ", "echo $X")
    assert not sh("echo $X", "echo  $X")
    assert eq.canonical_shell("cat <<EOF\nx\nEOF", ENV)[0][0] == "raw"
    assert eq.canonical_shell("(cd a; ls)", ENV)[0][0] == "raw"
    assert eq.canonical_shell("sleep 1 &", ENV)[0][0] == "raw"


def test_schema_defaults_and_null_optionals():
    v = check("bash", {"command": "ls", "timeout_seconds": 30}, {"command": "ls"})
    assert (v.equivalent, v.tier) == (True, "canonical") and "schema_default" in v.rules
    v = check("bash", {"command": "ls"}, {"command": "ls", "cwd": None})
    assert (v.equivalent, v.tier) == (True, "canonical")
    v = check("lookup", {"key": "k"}, {"key": "k", "limit": None})
    assert v.equivalent is True and "null_optional" in v.rules
    assert check("bash", {"command": "ls", "timeout_seconds": 60}, {"command": "ls"}).equivalent is None
    assert check("lookup", {"key": "k", "limit": 2.0}, {"key": "k", "limit": 2}).equivalent is True
    assert check("lookup", {"key": "k", "limit": 1}, {"key": "k", "limit": True}).equivalent is not True


@pytest.mark.parametrize("a, b, same", [
    ("notes.txt", f"{CWD}/notes.txt", True),
    ("./notes.txt", "notes.txt", True),
    ("~/proj/notes.txt", "notes.txt", True),
    (f"{CWD}//sub/./x.md", f"{CWD}/sub/x.md", True),
    ("sub/../x.md", "x.md", False),       # .. is not collapsed (a symlink can differ)
    ("notes.txt/", "notes.txt", False),   # a file path with a trailing slash is not the file
    ("~other/x", f"{HOME}/x", False),
])
def test_path_arguments(a, b, same):
    assert (check("read_file", {"path": a}, {"path": b}).equivalent is True) is same


def test_directory_paths_drop_the_trailing_slash():
    v = check("glob", {"pattern": "*.py", "root": "src/"}, {"pattern": "*.py", "root": f"{CWD}/src"})
    assert (v.equivalent, v.tier) == (True, "canonical")
    assert check("glob", {"pattern": "./*.py"}, {"pattern": "*.py"}).equivalent is not True


def test_a_case_without_a_working_directory_leaves_paths_alone():
    v = eq.check_call("read_file", {"path": "a"}, {"path": "/x/a"}, SCHEMAS["read_file"], eq.CaseEnv())
    assert v.equivalent is not True


def test_subset_expectations_fill_defaults_only_on_the_call():
    v = check("bash", {"command": "ls -al"}, {"command": "ls -la", "timeout_seconds": 90}, match="subset")
    assert (v.equivalent, v.tier) == (True, "canonical")
    v = check("bash", {"command": "ls", "timeout_seconds": 30}, {"command": "ls"}, match="subset")
    assert v.equivalent is True


def test_args_ignore_is_dropped_on_both_sides():
    v = check("lookup", {"key": "k", "limit": 1}, {"key": "k", "limit": 9}, ignore=("limit",))
    assert v.equivalent is True


# ── tiers past canonical ───────────────────────────────────────────────────

def test_side_effect_tools_are_conservatively_not_equivalent():
    v = check("write_file", {"path": "a", "content": "x"}, {"path": "a", "content": "y"})
    assert (v.equivalent, v.tier) == (False, "conservative")
    v = check("write_file", {"path": "a", "content": "x"}, {"path": f"{CWD}/a", "content": "x"})
    assert (v.equivalent, v.tier) == (True, "canonical")


def test_an_edit_with_wider_context_is_undecidable():
    edit = {"type": "function", "function": {"name": "edit_file", "description": "Edit.", "parameters": {
        "type": "object", "required": ["path", "old_str", "new_str"],
        "properties": {"path": {"type": "string", "x-prometheus-path": "file"}, "old_str": {"type": "string"},
                       "new_str": {"type": "string"}, "replace_all": {"type": "boolean", "default": False}}}}}
    schema = edit["function"]["parameters"]
    narrow = {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"}
    wide = {"path": "a.py", "old_str": "x = 1\ny = 0", "new_str": "x = 2\ny = 0"}
    for a, b in ((narrow, wide), (wide, narrow)):
        v = eq.check_call("edit_file", a, b, schema, ENV)
        assert (v.equivalent, v.tier) == (None, "undecidable"), v
    other = {"path": "a.py", "old_str": "x = 1\ny = 0", "new_str": "x = 2\ny = 9"}
    assert eq.check_call("edit_file", narrow, other, schema, ENV).tier == "conservative"
    assert eq.check_call("edit_file", narrow, {**wide, "path": "b.py"}, schema, ENV).tier == "conservative"


def test_unreadable_arguments_are_not_equivalent():
    v = check("bash", {"command": "ls"}, None)
    assert (v.equivalent, v.tier) == (False, "conservative")


def test_without_an_executor_a_read_only_difference_is_undecidable():
    v = check("bash", {"command": "head -5 f"}, {"command": "sed -n 1,5p f"})
    assert (v.equivalent, v.tier, v.reason) == (None, "undecidable", "no executor")
    v = check("lookup", {"key": "a"}, {"key": "b"}, executor=object())
    assert (v.equivalent, v.reason) == (None, "tool not executable here")


class FakeExecutor:
    def __init__(self, verdict: Any) -> None:
        self.verdict, self.calls = verdict, []

    def compare(self, tool: str, a: dict[str, Any], b: dict[str, Any], env: Any) -> Any:
        self.calls.append((tool, a, b))
        return self.verdict


def test_execution_gets_the_calls_as_written_with_defaults_filled():
    fake = FakeExecutor(eq.Verdict(True, "execution", "same"))
    v = check("bash", {"command": "head -5 f"}, {"command": "sed  -n 1,5p f"}, executor=fake)
    assert (v.equivalent, v.tier) == (True, "execution")
    tool, a, b = fake.calls[0]
    assert tool == "bash" and a == {"command": "head -5 f", "timeout_seconds": 30}
    assert b["command"] == "sed  -n 1,5p f"


def test_case_env_reads_the_system_prompt():
    msgs = [{"role": "system", "content": "# Environment\n- OS: Linux\n- Working directory: /home/tester/proj/\n"},
            {"role": "user", "content": "hi"}]
    assert eq.case_env(msgs) == eq.CaseEnv(cwd="/home/tester/proj", home="/home/tester")
    assert eq.case_env([{"role": "user", "content": "- Working directory: /x"}]) == eq.CaseEnv()
    assert eq.case_env([{"role": "system", "content": "- Working directory: /srv/x"}]).home is None


# ── execution in the gym sandbox ───────────────────────────────────────────

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "docs").mkdir(parents=True)
    (src / "a.txt").write_text("one\ntwo\n")
    (src / "docs" / "guide.md").write_text("# Guide\n\nalpha\nbeta\n")
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "first")
    bare = tmp_path / "store.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(bare))
    return bare


@pytest.fixture
def executor(tmp_path: Path, store: Path) -> Any:
    ex = eq.GymSandboxExecutor(store, tmp_path / "work", jail="off")
    yield ex
    ex.cleanup()


@needs_git
@pytest.mark.parametrize("a, b, want", [
    ("head -2 a.txt", "sed -n 1,2p a.txt", True),
    ("cat a.txt | wc -l", "wc -l < a.txt", True),
    ("ls docs", "ls docs/", True),
    ("head -2 a.txt", "head -3 a.txt", False),          # same on the 2-line file; the perturbed one tells
    ("ls docs | head -1", "ls docs | head -2", False),  # same on a 1-entry dir; pad files tell
    ("head -1 a.txt", "head -2 a.txt", False),
    ("cat ~/missing/x", "cat ~/missing/y", None),       # a path the snapshot lacks proves nothing
    ("grep -c nothing a.txt", "grep -c absent a.txt", None),  # exit 1 both
    ("curl http://127.0.0.1:1/", "cat a.txt", None),    # simulated or refused: never run
    ("date", "date", None),
    ("echo hi > out.txt", "echo hi > out.txt", True),   # same effect on the workspace
    ("echo hi > out.txt", "echo ho > out.txt", False),
    ("true", "true", None),                             # no output and no effect: nothing shown
])
def test_execution_tier(executor, a, b, want):
    v = executor.compare("bash", {"command": a, "timeout_seconds": 30},
                         {"command": b, "timeout_seconds": 30}, ENV)
    assert v.equivalent is want, v
    assert v.tier == ("undecidable" if want is None else "execution")


@needs_git
def test_execution_maps_the_case_home_and_cwd(executor):
    v = executor.compare("bash", {"command": f"cat {CWD}/a.txt"}, {"command": "cat ~/proj/a.txt"}, ENV)
    assert v.equivalent is True
    v = executor.compare("read_file", {"path": f"{CWD}/a.txt", "offset": 0, "limit": 200},
                         {"path": f"{CWD}/a.txt", "offset": 0, "limit": 1}, ENV)
    assert v.equivalent is False
    v = executor.compare("bash", {"command": "ls"}, {"command": "ls"}, eq.CaseEnv(cwd="/srv/x"))
    assert v.equivalent is None and "unmapped" in v.reason


@needs_git
def test_check_call_end_to_end_with_the_executor(executor):
    v = check("bash", {"command": "head -2 a.txt"}, {"command": "sed -n 1,2p a.txt"}, executor=executor)
    assert (v.equivalent, v.tier) == (True, "execution")
    executor.cleanup()
    assert not (executor.work / "A").exists() and not (executor.work / "B").exists()


# ── score.py: tool outcomes, metrics, compare ──────────────────────────────

def _case(cid: str, tool: str | None, args: dict[str, Any] | None = None, **extra: Any) -> Any:
    exp: dict[str, Any] = {"tool": tool}
    if args is not None:
        exp["args"] = args
    raw = {"id": cid, "set": "eval-tools", "tools": [BASH, READ, WRITE],
           "messages": [{"role": "system", "content": f"- Working directory: {CWD}"},
                        {"role": "user", "content": "do it"}], "expect": exp, **extra}
    return score.parse_case(raw, "test")


def _call(name: str, args: Any) -> dict[str, Any]:
    return {"id": "c", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _row(c: Any, calls: list[dict[str, Any]] | None = None, content: str = "", **extra: Any) -> dict[str, Any]:
    s = score.RequestSettings().as_dict()
    return {"set": c.set, "id": c.id, "request_sha": score.request_sha(c, s), "status": "ok",
            "outcome": "completed", "message": {"content": content, "tool_calls": calls or []},
            "latency_s": 0.1, "label": "m", "settings": s, "served_model": "f.gguf",
            "server": {"n_ctx": 1024, "total_slots": 1}, **extra}


def _scenario(label: str, flavour: str) -> dict[str, Any]:
    cases = [_case("ok", "bash", {"command": "ls -la"}), _case("canon", "bash", {"command": "ls -la"}),
             _case("wrong", "bash", {"command": "ls"}), _case("none", "bash", {"command": "ls"}),
             _case("extra", None), _case("unknown", "bash", {"command": "ls"}),
             _case("bad", "read_file", {"path": "a"}), _case("cut", "bash", {"command": "ls"})]
    student = flavour == "student"
    replies = {
        "ok": _row(cases[0], [_call("bash", {"command": "ls -la"})]),
        "canon": _row(cases[1], [_call("bash", {"command": "ls -al" if student else "ls -lt"})]),
        "wrong": _row(cases[2], [_call("read_file" if student else "bash", {"path": "a"} if student
                                       else {"command": "ls"})]),
        "none": _row(cases[3], [] if not student else [_call("bash", {"command": "ls"})]),
        "extra": _row(cases[4], [_call("bash", {"command": "ls"})]),
        "unknown": _row(cases[5], [_call("shell", {"command": "ls"})]),
        "bad": _row(cases[6], [_call("read_file", {"path": 3})]),
        "cut": {**_row(cases[7]), "outcome": "unparseable_emission"},
    }
    rows = {c.key: replies[c.id] for c in cases}
    return score.summarize(cases, rows, label=label)


def test_tool_outcomes_partition_every_case():
    sm = _scenario("base", "base")
    e = sm["sets"]["eval-tools"]
    outs = e["tool_outcomes"]
    assert sum(outs.values()) == e["scored"] == 8
    assert outs == {"success": 2, "wrong_tool": 0, "no_call": 1, "extra_call": 1, "unknown_tool": 1,
                    "failed_call": 2, "no_reply": 1}
    m = e["metrics"]
    total = sum(m[k]["k"] for k in ("task_success", "wrong_tool_rate", "no_call_rate", "extra_call_rate",
                                    "other_miss_rate"))
    assert total == e["scored"]
    assert e["case_tool_outcome"]["cut"] == "no_reply" and "cut" not in e["case_arg_equiv"]


def test_wrong_tool_is_an_offered_tool_only():
    sm = _scenario("student", "student")
    e = sm["sets"]["eval-tools"]
    assert e["tool_outcomes"]["wrong_tool"] == 1 and e["tool_outcomes"]["unknown_tool"] == 1
    assert e["case_tool_outcome"]["wrong"] == "wrong_tool"
    assert e["case_tool_outcome"]["unknown"] == "unknown_tool"


def test_summary_reports_arg_equivalence_beside_exactness():
    e = _scenario("student", "student")["sets"]["eval-tools"]
    m = e["metrics"]
    # named the expected tool: ok, canon, none (a call this time), bad
    assert (m["arg_exactness"]["k"], m["arg_exactness"]["n"]) == (2, 4)
    assert (m["arg_equivalence"]["k"], m["arg_equivalence"]["n"]) == (3, 4)  # + canon (flag order)
    assert e["arg_equivalence_tiers"] == {"exact": 2, "canonical": 1, "execution": 0,
                                          "conservative": 0, "undecidable": 1}
    assert e["arg_equivalence_upper"]["k"] == 4
    assert e["arg_equivalence_method"] == "static"
    assert e["case_arg_equiv"]["canon"] is True


def test_compare_prints_the_headline_rows_and_paired_deltas_without_changing_the_gate():
    base, cand = _scenario("base", "base"), _scenario("student", "student")
    g = score.evaluate_gate(base, cand)
    md = score.render_compare([base, cand], g, markdown=True)
    assert "Headline, eval-tools (information, not gated)" in md
    for name in ("wrong-tool rate", "no call when one was needed", "call when none was needed",
                 "exact arguments", "argument equivalence (proven)"):
        assert f"| {name} |" in md
    pairs = score.headline_pairs(base, cand)
    assert pairs["wrong_tool_rate"]["n"] == 8 and pairs["wrong_tool_rate"]["diff_pts"] == 12.5
    assert pairs["arg_equivalence"]["n"] == 3  # both named the expected tool: ok, canon, bad
    text = score.render_compare([base, cand], g, markdown=False)
    assert "headline, eval-tools (not gated)" in text
    js = score.gate_json([base, cand], g)
    assert js["headline"]["gated"] is False and "wrong_tool_rate" in js["headline"]["paired"]
    assert [c.name for c in g.criteria][0] == "tool-call task success"


def test_compare_still_reads_summaries_without_the_new_fields():
    base, cand = _scenario("base", "base"), _scenario("student", "student")
    for s in (base, cand):
        e = s["sets"]["eval-tools"]
        for k in ("case_tool_outcome", "case_arg_equiv"):
            e.pop(k)
    g = score.evaluate_gate(base, cand)
    md = score.render_compare([base, cand], g, markdown=True)
    assert "| wrong-tool rate | n/a |" in md


# ── rescore + validate ─────────────────────────────────────────────────────

def test_rescore_rates_and_validation():
    cases = [_case("a", "bash", {"command": "ls -la"}), _case("b", "bash", {"command": "head -5 f"}),
             _case("c", "bash", {"command": "ls"})]
    rows = {cases[0].key: _row(cases[0], [_call("bash", {"command": "ls -al"})]),
            cases[1].key: _row(cases[1], [_call("bash", {"command": "sed -n 1,5p f"})]),
            cases[2].key: _row(cases[2], [_call("read_file", {"path": "x"})])}
    rep = {"models": {"m": eq.rescore(cases, rows)}}
    m = rep["models"]["m"]["metrics"]
    assert m["wrong_tool_rate"]["k"] == 1 and m["arg_equivalence"]["k"] == 1
    assert m["arg_equivalence_upper"]["k"] == 2 and m["arg_equivalence_undecidable"] == 1
    labels = {"labels": [{"model": "m", "id": "a", "equivalent": True},
                         {"model": "m", "id": "b", "equivalent": True}]}
    res = eq.validate(rep, labels)
    assert (res["tp"], res["fn"], res["precision"], res["recall"]) == (1, 1, 1.0, 0.5)
    assert res["undecidable_labelled_equivalent"] == 1
    other = {"models": {"m2": eq.rescore(cases, {**rows, cases[0].key: _row(
        cases[0], [_call("bash", {"command": "ls"})])})}}
    p = eq.paired(rep["models"]["m"], other["models"]["m2"], "arg_equiv")
    assert p["n"] == 2 and p["delta_pts"] == 50.0
    assert "| m |" in eq.render(rep | {"pairs": []})
