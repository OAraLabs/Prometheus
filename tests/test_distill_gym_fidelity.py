"""Tests for scripts/distill/gym_fidelity.py (phase-0 read-only measurements).
Neutral fixture names, no network, no GPU, no jail: nothing here runs a command."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

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


gf = _load("distill_gym_fidelity", "gym_fidelity.py")
eq = gf.EQ()
tg = gf.TG()

HOME = "/root"
CWD = HOME + "/proj"
ENV = eq.CaseEnv(cwd=CWD, home=HOME)


def test_classify_bash_grammar() -> None:
    assert gf.classify_bash("ls -la src | head -5", tg) == ("allowed", "ls")
    assert gf.classify_bash("grep -rn foo . 2>/dev/null", tg) == ("allowed", "grep")
    assert gf.classify_bash("git log --oneline -5", tg) == ("git_read", "log")
    assert gf.classify_bash("git -C sub status", tg) == ("git_read", "status")
    assert gf.classify_bash("git push", tg)[0] == "git_refused"
    assert gf.classify_bash("systemctl status x | grep Active", tg) == ("simulated_host", "systemctl")
    assert gf.classify_bash("ls | systemctl status x", tg) == ("refused", "program")
    assert gf.classify_bash("cat /proc/meminfo", tg) == ("virtual_read", "cat")
    assert gf.classify_bash("python3 -c 'print(1)'", tg) == ("unavailable_program", "python")
    assert gf.classify_bash("gh pr list", tg) == ("unavailable_program", "gh")
    assert gf.classify_bash("ls && docker ps", tg) == ("unavailable_program", "docker")
    assert gf.classify_bash("echo $HOME", tg) == ("refused", "shell_syntax")
    assert gf.classify_bash("ls\nls", tg) == ("refused", "multiline")
    assert gf.classify_bash("sleep 1 &", tg)[0] == "refused"
    assert gf.classify_bash("FOO=1 ls", tg) == ("refused", "shell_syntax")


def test_path_classes() -> None:
    def cls(raw: str) -> str:
        return gf.path_class(raw, ENV, eq, tg)[0]

    assert cls("src/a.py") == "cwd"
    assert cls(CWD + "/src") == "cwd"
    assert cls("~/proj/x") == "cwd"
    assert cls("/proc/meminfo") == "virtual"
    assert cls("/tmp/x.log") == "tmp"
    assert cls("~/.prometheus/config.yaml") == "dot_prometheus"
    assert cls("~/other/repo") == "home_other"
    assert cls("/etc/hosts") == "abs_other"
    assert gf.path_class("src/a.py", eq.CaseEnv(), eq, tg)[0] == "unmapped"


def test_reference_paths_and_seen() -> None:
    args = {"command": "cat ~/proj/a.txt /tmp/b --file=/etc/x 2>/dev/null https://h/x", "cwd": "~/proj"}
    assert gf.reference_paths("bash", args, tg) == ["~/proj", "~/proj/a.txt", "/tmp/b", "/etc/x"]
    text = gf.history_text([{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})}}]}])
    _c, norm = gf.path_class("~/proj/a.txt", ENV, eq, tg)
    assert gf.seen_in("~/proj/a.txt", norm, ENV, text)
    assert not gf.seen_in("/tmp/b", "/tmp/b", ENV, text)


def _case(cid: str, tool: str | None, args: Any, *, source: str = "lcm", hard: bool = False,
          tool_result: bool = True) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": f"- Working directory: {CWD}"},
                                  {"role": "user", "content": "do the thing"}]
    if tool_result:
        msgs += [{"role": "assistant", "content": "", "tool_calls": [
                     {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    tools = [{"type": "function", "function": {"name": n, "description": "", "parameters": {"type": "object"}}}
             for n in ("bash", "read_file", "edit_file")]
    return {"id": cid, "set": "eval-tools", "messages": msgs, "tools": tools, "tool_choice": "auto",
            "expect": {"tool": tool, "args": args} if tool else {"tool": None},
            "meta": {"context_source": source, "hard_case": hard, "trace_id": f"golden_traces_1_2.jsonl:{cid[-1]}"}}


def _write_cases(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "cases.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_m1_census(tmp_path: Path) -> None:
    p = _write_cases(tmp_path, [
        _case("c1", "bash", {"command": "ls /tmp/x"}),
        _case("c2", "read_file", {"path": "src/a.py"}, source="stored", tool_result=False),
        _case("c3", "edit_file", {}, hard=True),
        _case("c4", None, None)])
    res = gf.m1(gf.S().load_cases([p]))
    assert res["cases"] == 4 and res["refs_with_args"] == 2
    assert res["tool_has_args"] == {"bash|args": 1, "read_file|args": 1, "edit_file|no_args": 1,
                                    "(no call)|no_args": 1}
    assert res["context_source"] == {"lcm": 3, "stored": 1}
    assert res["hard_case"] == {False: 3, True: 1}
    assert res["env_mapped"] == {True: 4}
    assert res["path_class"] == {"tmp": 1, "cwd": 1}
    assert res["bash_class"] == {"allowed": 1}


def test_m3_tallies() -> None:
    full = {"method": "static+execution", "models": {"a": {"per_case": {
        "x": {"arg_equiv": {"equivalent": None, "tier": "undecidable", "reason": "execution: expected missing_path, got ok"}},
        "y": {"arg_equiv": {"equivalent": True, "tier": "exact"}}, "z": {"outcome": "wrong_tool"}}}}}
    cand = {"calls": [{"case": "x", "tool": "grep", "outcome": {"state": "reference_not_ok", "reason": "reference: nonzero"}},
                      {"case": "x", "tool": "glob", "outcome": {"state": "reference_not_ok", "reason": "reference: nonzero"}},
                      {"case": "y", "tool": "memory", "outcome": {"state": "not_run", "reason": "a tool the gym cannot run"}}],
            "executor": {"runs": 3}}
    rep = {"annotation": {"decisions": {"acceptable": 1, "rejected": 2}, "outcome_states": {"not_run": 1}}}
    res = gf.m3(full, cand, rep)
    assert res["arg_equiv"]["tiers"] == {"undecidable": 1, "exact": 1}
    assert res["outcome_check"]["states"] == {"reference_not_ok": 2, "not_run": 1}
    assert res["outcome_check"]["reference_not_ok_cases"] == 1
    assert res["judge"]["decisions"]["rejected"] == 2


def test_m4_pool(tmp_path: Path) -> None:
    def row(tool: str | None, cr: dict[str, Any] | None, comp: bool = False, area: str = "coding") -> str:
        meta: dict[str, Any] = {"tool_name": tool, "target_area": area, "compacted": comp, "generator": "g1"}
        if cr is not None:
            meta["call_result"] = cr
        return json.dumps({"messages": [], "_meta": meta}) + "\n"

    p = tmp_path / "pool-r2.traces.jsonl"
    p.write_text(row("bash", {"status": "success", "text": "x", "injected": False, "planted": False})
                 + row("bash", {"status": "success", "text": "x", "injected": False, "planted": False}, comp=True)
                 + row("read_file", {"status": "error", "text": "x", "injected": True, "planted": False})
                 + row("grep", {"status": "success", "text": "x", "injected": False, "planted": True})
                 + row(None, None, area="final-answer"))
    res = gf.m4([str(p)])
    assert res["rows"] == 5
    assert res["status"] == {"success": 2, "injected": 1, "planted": 1, "no_call": 1}
    assert res["ok_uncompacted"] == 1
    assert res["ok_uncompacted_executable_today"] == {True: 1}


def test_avail_joins_trace_index(tmp_path: Path) -> None:
    p = _write_cases(tmp_path, [_case("c1", "bash", {"command": "ls"}), _case("c2", "bash", {"command": "ls"})])
    ti = tmp_path / "ti.jsonl"
    ti.write_text(json.dumps({"file": "golden_traces_1_2.jsonl", "ts": 1_700_000_000.0, "rowid": 7}) + "\n")
    snap = tmp_path / "lcm.db"
    snap.write_bytes(b"x")
    res = gf.avail(gf.S().load_cases([p]), str(ti), None, str(snap))
    assert res["trace_index_join"] == {"golden|joined": 1, "golden|not_joined": 1}
    assert res["telemetry_rowid"] == {True: 1}
    assert res["lcm_snapshot"]["exists"] is True and res["lcm_snapshot"]["bytes"] == 1


def test_path_targets_and_first_blocker() -> None:
    def tgt(raw: str) -> tuple[str, str | None]:
        return gf.path_target(raw, ENV, eq, tg)

    assert tgt("~/Prometheus/src/a.py") == ("repo", "src/a.py")
    assert tgt("~/projects/Prometheus-wt/docs") == ("repo", "docs")
    assert tgt("~/projects/elsewhere/x") == ("other_project", None)
    assert tgt("~/notes.txt") == ("home_misc", None)
    assert tgt("~/proj/src") == ("cwd", "src")
    assert gf.ref_target("bash", {"command": "ls ~/Prometheus /tmp/x"}, ENV, eq, tg) == "tmp"
    assert gf.ref_target("bash", {"command": "ls"}, ENV, eq, tg) == "(no absolute path)"
    ok = eq.RunResult("ok", "d", "", empty=False)
    assert gf.first_blocker("refused", "multiline", "A:missing_path", None, "repo") == "grammar:multiline"
    assert gf.first_blocker("unavailable_program", "gh", "A:refused", None, "tmp") == "grammar:program"
    assert gf.first_blocker("git_refused", "push", "A:missing_path", None, "repo") == "grammar:git_write"
    assert gf.first_blocker("allowed", "ls", "A:missing_path", None, "other_project") == "missing_path:other_project"
    assert gf.first_blocker("allowed", "ls", "clean", ok, "repo") == "clean"
    assert gf.first_blocker("simulated_host", "ss", "A:refused", None, "-") == "simulated"
