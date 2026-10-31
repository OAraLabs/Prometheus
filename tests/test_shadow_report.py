"""Tests for scripts/distill/shadow_report.py, the shadow-agreement summarizer.

Rows are written the way the daemon writes them (``engine.shadow.build_row``), so a change to the
row shape that the report does not follow fails here. No model is called: the default judge is
normalized match, and a pluggable judge is loaded from a module written to a tmp path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.shadow import _Outcome, build_row
from prometheus.providers.base import ApiMessageRequest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "distill" / "shadow_report.py"
_spec = importlib.util.spec_from_file_location("distill_shadow_report", _PATH)
assert _spec and _spec.loader
report = importlib.util.module_from_spec(_spec)
sys.modules["distill_shadow_report"] = report
_spec.loader.exec_module(report)

TS = 1_793_000_000.0  # a fixed instant; the date it falls on is read back from the row


def _text(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _call(name: str, **args: Any) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[ToolUseBlock(name=name, input=args)])


def _row(primary: Any, shadow: Any, *, category_tool: str = "edit_file", session: str = "telegram:1",
         primary_error: str | None = None, shadow_error: str | None = None,
         ts: float = TS, model: str = "primary-model") -> dict[str, Any]:
    request = ApiMessageRequest(model=model, messages=[
        ConversationMessage.from_user_text("go"),
        _call(category_tool, path="a.py"),
    ], tools=[{"name": category_tool}])
    return build_row(
        request_id="r" * 32, request=request, session_id=session, round_index=0,
        shadow_model="student-9b", shadow_provider="llama_cpp",
        primary=_Outcome(message=None if primary_error else primary, error=primary_error,
                         latency_ms=1000.0),
        shadow=_Outcome(message=None if shadow_error else shadow, error=shadow_error,
                        latency_ms=400.0),
        ts=ts,
    )


def _write(path: Path, rows: list[dict[str, Any]], *, extra_lines: list[str] = ()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in rows] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _summary(rows, **kw):
    return report.summarize(rows, report.normalized_match, **kw)


# ── per-row verdicts ───────────────────────────────────────────────────────────────────────


def test_a_tool_round_agrees_only_on_the_same_tool_with_equivalent_args():
    same = _row(_call("bash", command="pytest"), _call("bash", command="pytest "))
    other_args = _row(_call("bash", command="pytest"), _call("bash", command="ls"))
    other_tool = _row(_call("bash", command="pytest"), _call("grep", pattern="x"))
    answered = _row(_call("bash", command="pytest"), _text("I would run the tests."))
    verdicts = [report.assess(r, report.normalized_match) for r in (same, other_args, other_tool, answered)]
    assert [v["kind"] for v in verdicts] == ["tool"] * 4
    assert [v["same_tool"] for v in verdicts] == [True, True, False, False]
    assert [v["agree"] for v in verdicts] == [True, False, False, False]


def test_an_answer_round_needs_an_answer_and_the_judge():
    same = _row(_text("It is 42."), _text("it is 42"))
    worded = _row(_text("It is 42."), _text("The value is forty-two."))
    called = _row(_text("It is 42."), _call("bash", command="echo 42"))
    verdicts = [report.assess(r, report.normalized_match) for r in (same, worded, called)]
    assert [v["kind"] for v in verdicts] == ["answer"] * 3
    assert [v["agree"] for v in verdicts] == [True, False, False]


def test_errors_primary_excluded_shadow_counted_unless_asked():
    p_err = _row(None, _text("x"), primary_error="ConnectionError: down")
    s_err = _row(_text("x"), None, shadow_error="timeout after 300s")
    assert report.assess(p_err, report.normalized_match)["comparable"] is False
    counted = report.assess(s_err, report.normalized_match)
    assert counted["comparable"] is True and counted["agree"] is False
    dropped = report.assess(s_err, report.normalized_match, exclude_shadow_errors=True)
    assert dropped["comparable"] is False and dropped["excluded"] == "shadow_error"


# ── summary ────────────────────────────────────────────────────────────────────────────────


def test_categories_at_or_above_the_threshold_are_listed():
    coding = [_row(_call("bash", command="make"), _call("bash", command="make"))] * 40
    memory = ([_row(_call("wiki_search", q="x"), _call("wiki_search", q="x"), category_tool="wiki_search")] * 30
              + [_row(_call("wiki_search", q="x"), _call("wiki_search", q="y"), category_tool="wiki_search")] * 10)
    summary = _summary(coding + memory)
    assert summary["categories"]["coding"]["agreement"]["rate"] == 1.0
    assert summary["categories"]["memory/wiki"]["agreement"]["rate"] == 0.75
    assert summary["equivalent_categories"] == ["coding"]
    assert summary["overall"]["comparable"] == 80


def test_exactly_ninety_five_percent_counts():
    rows = ([_row(_text("yes"), _text("yes"))] * 95 + [_row(_text("yes"), _text("no"))] * 5)
    summary = _summary(rows)
    [cat] = summary["categories"]
    assert summary["categories"][cat]["agreement"]["rate"] == 0.95
    assert summary["equivalent_categories"] == [cat]


def test_too_few_rows_never_qualify():
    rows = [_row(_text("yes"), _text("yes"))] * 10
    summary = _summary(rows, min_rows=30)
    assert summary["equivalent_categories"] == []
    assert "Too few comparable rows" in report.render_markdown(summary)


def test_shadow_errors_lower_agreement_until_excluded():
    rows = [_row(_text("ok"), _text("ok"))] * 30 + [_row(_text("ok"), None, shadow_error="timeout")] * 10
    counted = _summary(rows)
    excluded = _summary(rows, exclude_shadow_errors=True)
    [cat] = counted["categories"]
    assert counted["categories"][cat]["agreement"]["rate"] == 0.75
    assert excluded["categories"][cat]["agreement"]["rate"] == 1.0
    assert excluded["categories"][cat]["shadow_errors"] == 10


def test_a_pluggable_judge_replaces_exact_match(tmp_path, monkeypatch):
    (tmp_path / "lenient_judge.py").write_text(
        "def anything_goes(primary, shadow, row):\n    return bool(primary) and bool(shadow)\n",
        encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    judge, name = report.load_judge("lenient_judge:anything_goes")
    rows = [_row(_text("It is 42."), _text("Forty-two."))] * 30
    assert _summary(rows)["equivalent_categories"] == []
    lenient = report.summarize(rows, judge, judge_name=name)
    assert lenient["judge"] == "lenient_judge:anything_goes"
    assert lenient["equivalent_categories"] != []


@pytest.mark.parametrize("spec", ["no_colon", ":fn", "json:", "json:does_not_exist"])
def test_a_bad_judge_spec_is_refused(spec):
    with pytest.raises((ValueError, AttributeError)):
        report.load_judge(spec)


# ── reading files, CLI ─────────────────────────────────────────────────────────────────────


def test_torn_and_foreign_lines_are_counted_not_fatal(tmp_path):
    rows = [_row(_text("a"), _text("a"))]
    path = _write(tmp_path / "shadow" / "shadow-2026-11-01.jsonl", rows,
                  extra_lines=['{"schema": "other/1"}', '{"torn": '])
    tally: Any = report.Counter()
    got = report.read_rows([path], tally=tally)
    assert len(got) == 1 and tally == {"foreign_rows": 1, "malformed_lines": 1}


def test_date_and_model_filters(tmp_path):
    day = 86_400.0
    rows = [_row(_text("a"), _text("a"), ts=TS), _row(_text("a"), _text("a"), ts=TS + day),
            _row(_text("a"), _text("a"), ts=TS + 2 * day, model="other-primary")]
    path = _write(tmp_path / "rows.jsonl", rows)
    first, second, third = (r["ts"][:10] for r in rows)
    assert len(report.read_rows([path], since=second)) == 2
    assert len(report.read_rows([path], until=first)) == 1
    assert len(report.read_rows([path], primary_model="other-primary")) == 1
    assert third > second > first


def test_cli_writes_markdown_and_json(tmp_path, capsys):
    rows = [_row(_call("bash", command="make"), _call("bash", command="make"))] * 30
    _write(tmp_path / "shadow" / "shadow-2026-11-01.jsonl", rows)
    md_out, json_out = tmp_path / "summary.md", tmp_path / "summary.json"
    code = report.main(["--dir", str(tmp_path / "shadow"), "--markdown-out", str(md_out),
                        "--json-out", str(json_out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "| coding | 30 | 30 | 100.0% (30/30" in printed
    assert "Primary-equivalent" in printed and "coding" in printed.split("Primary-equivalent")[1]
    assert md_out.read_text(encoding="utf-8") == printed
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["schema"] == "shadow-report/1" and data["equivalent_categories"] == ["coding"]


def test_cli_refuses_a_missing_directory_and_a_bad_date(tmp_path, capsys):
    assert report.main(["--dir", str(tmp_path / "nope")]) == 2
    assert report.main(["--dir", str(tmp_path), "--since", "yesterday"]) == 2
    assert "error" in capsys.readouterr().err


def test_an_empty_directory_reports_nothing_yet(tmp_path, capsys):
    (tmp_path / "shadow").mkdir()
    assert report.main(["--dir", str(tmp_path / "shadow")]) == 0
    assert "none yet" in capsys.readouterr().out
