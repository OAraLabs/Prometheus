"""Tests for scripts/distill/repair_gap.py: eval-tools scored raw and after the
live adapter's repair. Neutral fixture names; no network, no GPU."""

from __future__ import annotations

import importlib.util
import json
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
gap = _load("distill_repair_gap", "repair_gap.py")

pytestmark = pytest.mark.skipif(score._jsonschema is None, reason="needs jsonschema (mcp extra)")

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"},
                           "mode": {"type": "string", "enum": ["w", "a"], "default": "w"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}
BASH_TOOL = {
    "type": "function",
    "function": {"name": "bash", "description": "Run a command.",
                 "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                "required": ["command"]}},
}
WANT = {"path": "a.txt", "content": "hi"}


def _case(cid: str) -> Any:
    return score.parse_case({
        "id": cid, "set": "eval-tools",
        "messages": [{"role": "user", "content": "write the file"}],
        "tools": [WRITE_TOOL, BASH_TOOL],
        "expect": {"tool": "write_file", "args": dict(WANT)},
    }, "test")


def _row(c: Any, calls: list[tuple[str, Any]], content: str = "") -> dict[str, Any]:
    s = score.RequestSettings().as_dict()
    tool_calls = [{"id": f"c{i}", "type": "function",
                   "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                  for i, (n, a) in enumerate(calls)]
    return {"set": c.set, "id": c.id, "case_sha": c.sha, "request_sha": score.request_sha(c, s),
            "status": "ok", "outcome": "completed",
            "message": {"content": content, "tool_calls": tool_calls},
            "latency_s": 0.1, "label": "m", "settings": s, "served_model": "fake.gguf",
            "server": {"n_ctx": 262144, "total_slots": 1}}


def _fixture() -> tuple[list[Any], dict[tuple[str, str], dict[str, Any]]]:
    replies = {
        "clean": [("write_file", WANT)],                          # valid either way
        "null_optional": [("write_file", {**WANT, "mode": None})],  # null_drop fixes it
        "misnamed": [("writefile", WANT)],                         # fuzzy_name fixes it
        "far_name": [("upload_everything", WANT)],                 # refused: stays wrong
        "unreadable": [("write_file", '{"path": "a.txt", ')],      # never reaches the adapter
        "null_required": [("write_file", {"path": "a.txt", "content": None})],  # stays invalid
        "no_call": [],
    }
    cases = [_case(cid) for cid in replies]
    rows = {c.key: _row(c, replies[c.id], "" if replies[c.id] else "I wrote it.") for c in cases}
    return cases, rows


def test_raw_recompute_equals_the_score_summary():
    cases, rows = _fixture()
    summary = score.summarize(cases, rows, label="m")
    rep = gap.repair_gap(cases, rows, summary)
    assert rep["raw_check"]["matches"], rep["raw_check"]
    sm = summary["sets"]["eval-tools"]["metrics"]
    for m in gap.METRICS:
        assert (rep["raw"][m]["k"], rep["raw"][m]["n"]) == (sm[m]["k"], sm[m]["n"])


def test_repairable_calls_flip_to_valid_after_repair():
    cases, rows = _fixture()
    rep = gap.repair_gap(cases, rows, score.summarize(cases, rows, label="m"))
    ch = rep["changed"]
    assert ch["arg_valid"]["fixed"] == ["null_optional"]
    assert ch["tool_name_correct"]["fixed"] == ["misnamed"]
    assert sorted(ch["task_success"]["fixed"]) == ["misnamed", "null_optional"]
    assert all(not v["broken"] for v in ch.values())
    assert rep["raw"]["task_success"]["k"] == 1 and rep["post_repair"]["task_success"]["k"] == 3
    assert rep["repair_kinds"] == {"null_drop": 1, "fuzzy_name": 1}
    assert rep["calls_by_outcome"] == {"unchanged": 2, "repaired": 2, "refused": 1, "unreadable": 1}
    assert rep["cases_with_a_repair"] == ["misnamed", "null_optional"]
    assert rep["gap_pts"]["task_success"] == pytest.approx(100 * 2 / 7, abs=0.01)


def test_repair_calls_one_call():
    c = _case("x")
    calls, _ = score.reply_calls(c, _row(c, [("write_file", {**WANT, "mode": None})])["message"])
    fixed, events = gap.repair_calls(c, calls)
    assert fixed[0]["arguments"] == WANT and calls[0]["arguments"]["mode"] is None
    assert events == [{"outcome": "repaired", "kinds": ["null_drop"], "name_changed": False}]


def test_cli_exit_codes(tmp_path, capsys):
    cases, rows = _fixture()
    cp = tmp_path / "cases.jsonl"
    cp.write_text("".join(json.dumps(c.raw) + "\n" for c in cases), encoding="utf-8")
    rp = tmp_path / "eval-tools.jsonl"
    rp.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")
    summary = score.summarize(cases, rows, label="m")
    sp = tmp_path / "eval-tools.summary.json"
    sp.write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "gap.json"
    assert gap.main(["--results", str(rp), "--cases", str(cp), "--summary", str(sp),
                     "--json-out", str(out)]) == 0
    assert json.loads(out.read_text())["raw_check"]["matches"]
    assert "raw recompute == summary: yes" in capsys.readouterr().out
    summary["sets"]["eval-tools"]["metrics"]["task_success"]["k"] += 1
    sp.write_text(json.dumps(summary), encoding="utf-8")
    assert gap.main(["--results", str(rp), "--cases", str(cp), "--summary", str(sp)]) == 1
