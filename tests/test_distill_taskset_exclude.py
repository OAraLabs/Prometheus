"""scripts/distill/taskset_gen.py exclude: no taskset row may come from a prompt of an excluded session
(DISTILL-OPEN-QUESTIONS #6: the 2026-10-13 incident session is in no training data). Synthetic inputs only.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ts = _load("distill_taskset_gen", "taskset_gen.py")
BAD = "beacon:incident-1013"
GOLDEN_PROMPT = "Pick up where we left off and finish the summary of the backlog."
LCM_PROMPT = "Continue the earlier task about the release checklist now."


def _row(prompt: str, *patterns: str) -> dict[str, Any]:
    srcs = [{"template_id": "tpl-" + ts.h("tpl|" + p), "pattern_hash": p, "origin": "golden",
             "span": [0, len(prompt)]} for p in patterns]
    return {"id": "ts-" + ts.h(ts.norm(prompt)), "taskset": "real-v1", "category": "other",
            "subcategory": "general", "intent": "request", "mode": "agentic", "prompt": prompt, "slots": [],
            "template_id": srcs[0]["template_id"], "source_pattern_hash": patterns[0], "task_group": "tg-" + patterns[0],
            "transform": "recombine" if len(patterns) == 2 else "slot_fill", "sources": srcs,
            "flags": {"suppress_tools": False, "inject_errors": False, "force_compaction": False},
            "provenance": "real-pattern-v1"}


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    traces = tmp_path / "golden_traces_1.jsonl"
    traces.write_text("".join(json.dumps(t) + "\n" for t in [
        {"messages": [{"role": "user", "content": GOLDEN_PROMPT}, {"role": "assistant", "content": "ok"}],
         "_meta": {"session_id": BAD}},
        {"messages": [{"role": "user", "content": "What is on the calendar for Friday?"},
                      {"role": "assistant", "content": "ok"}], "_meta": {"session_id": "beacon:fine"}},
    ]))
    lcm = tmp_path / "lcm.db"
    con = sqlite3.connect(lcm)
    con.execute("CREATE TABLE lcm_messages (session_id TEXT, role TEXT, content TEXT, content_json TEXT, "
                "turn_index INTEGER)")
    con.executemany("INSERT INTO lcm_messages VALUES (?, ?, ?, ?, ?)", [
        (BAD, "user", LCM_PROMPT, None, 3), (BAD, "assistant", "sure", None, 4),
        ("beacon:fine", "user", "Show me the open pull requests.", None, 1)])
    con.commit()
    con.close()
    excl = tmp_path / "exclude-sessions.txt"
    excl.write_text(f"# the Token-Plan incident session\n{BAD}\n")
    clean = ts.h(ts.norm("What is on the calendar for Friday?"))
    rows = [_row("Finish the summary of the backlog for me.", ts.h(ts.norm(GOLDEN_PROMPT))),   # golden source
            _row("What is on the calendar for Friday? Then the release checklist.", clean,     # a recombined
                 ts.h(ts.norm(LCM_PROMPT))),                                                  # partner (lcm)
            _row("What is on the calendar for Monday?", clean)]
    for r in rows:
        assert ts.validate_task(r) == [], ts.validate_task(r)
    out = tmp_path / "tasksets"
    out.mkdir()
    tasks = out / "real-v1.jsonl"
    tasks.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"traces": traces, "lcm": lcm, "excl": excl, "tasks": tasks}


def _args(i: dict[str, Path], *more: str) -> list[str]:
    return ["exclude", "--tasks", str(i["tasks"]), "--exclude-sessions", str(i["excl"]), "--traces",
            str(i["traces"]), "--lcm-snapshot", str(i["lcm"]), *more]


def test_check_finds_rows_of_an_excluded_session_and_exclude_drops_them(inputs: dict[str, Path], capsys) -> None:
    assert ts.main(_args(inputs, "--check")) == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["rows_dropped"] == 2 and summary["task_groups_dropped"] == 2 and summary["rows_out"] == 1
    assert not (inputs["tasks"].parent / "real-v1-excl.jsonl").exists()          # --check writes nothing
    assert ts.main(_args(inputs)) == 0
    capsys.readouterr()
    out = inputs["tasks"].parent / "real-v1-excl.jsonl"
    kept = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["prompt"] for r in kept] == ["What is on the calendar for Monday?"]
    assert all(ts.validate_task(r) == [] for r in kept) and oct(out.stat().st_mode & 0o777) == "0o600"
    record = (inputs["tasks"].parent / "real-v1-excl.exclusion.json").read_text()
    assert BAD not in record and GOLDEN_PROMPT not in record                    # counts and hashes only
    assert json.loads(record)["golden_rows_in_listed_sessions"] == 1
    assert json.loads(record)["lcm_rows_in_listed_sessions"] == 1
    # the filtered taskset passes the check gen_runner.py runs before every window
    assert ts.main(["exclude", "--tasks", str(out), "--exclude-sessions", str(inputs["excl"]), "--traces",
                    str(inputs["traces"]), "--lcm-snapshot", str(inputs["lcm"]), "--check"]) == 0


def test_without_the_snapshot_an_lcm_only_prompt_stays(inputs: dict[str, Path], capsys) -> None:
    # (why the snapshot is read by default: the golden traces alone miss the second row)
    assert ts.main(_args(inputs, "--check")[:-3] + ["--lcm-snapshot", "", "--check"]) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["rows_dropped"] == 1


@pytest.mark.parametrize("change", ["empty-list", "no-traces", "no-snapshot", "live-snapshot", "same-out"])
def test_exclude_refuses_what_it_cannot_check(inputs: dict[str, Path], tmp_path: Path, monkeypatch,
                                              change: str) -> None:
    args = _args(inputs)
    if change == "empty-list":
        inputs["excl"].write_text("# nobody\n\n")
    elif change == "no-traces":
        args[args.index("--traces") + 1] = str(tmp_path / "none-*.jsonl")
    elif change == "no-snapshot":
        args[args.index("--lcm-snapshot") + 1] = str(tmp_path / "missing.db")
    elif change == "live-snapshot":
        monkeypatch.setenv("HOME", str(tmp_path))
        live = tmp_path / ".prometheus" / "lcm.db"
        live.parent.mkdir()
        live.write_bytes(inputs["lcm"].read_bytes())
        args[args.index("--lcm-snapshot") + 1] = str(live)
    else:
        args += ["--out", str(inputs["tasks"])]
    assert ts.main(args) == 2
    assert not (inputs["tasks"].parent / "real-v1-excl.jsonl").exists()
