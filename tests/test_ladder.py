"""Model-ladder harness (WP-2.1): suite rules, verdicts, the judge pin, and
what lands in telemetry.db.

The live path needs a served model; everything around it is tested here,
including one end-to-end run through the REAL agent loop with a scripted
provider, which is what proves a run's rows land and join.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.evals.judge import JudgeVerdict
from prometheus.gym.ladder import fixtures as fx
from prometheus.gym.ladder import record as rec
from prometheus.gym.ladder import runner as lr
from prometheus.gym.ladder.selfcheck import selfcheck_task, synthetic_transcript
from prometheus.gym.ladder.suite import LadderTask, load_suite, select_tasks
from prometheus.gym.ladder.verdict import (
    UNSCORED,
    Verdict,
    check_predicates,
    decide,
    run_acceptance,
)
from prometheus.gym.scoring import EMISSION, EXECUTION, RunTranscript, ToolEvent
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


SUITE_META = """
name: t
version: 1
system_prompt: "You are a test."
classes:
  qa: {file: qa.yaml, max_rounds: 3, max_tool_calls: 3, max_tokens: 256, timeout_s: 30,
       success_criterion: "the reply is right"}
"""


def _suite(tmp_path: Path, tasks_yaml: str, head: str = "") -> Path:
    root = tmp_path / "suite"
    root.mkdir(exist_ok=True)
    (root / "suite.yaml").write_text(SUITE_META)
    (root / "qa.yaml").write_text(f"class: qa\n{head}tasks:\n{tasks_yaml}")
    return root


def _task(**kw) -> LadderTask:
    base = dict(id="t1", task_class="qa", prompt="p")
    base.update(kw)
    return LadderTask(**base)


def _transcript(text: str, events: list[ToolEvent] | None = None) -> RunTranscript:
    t = RunTranscript(messages=[], final_text=text)
    t.tool_events = events or []
    return t


@pytest.fixture
def sandbox(tmp_path):
    sb = fx.Sandbox(tmp_path / "sb")
    prev = sb.activate()
    sb.reset()
    yield sb
    fx.Sandbox.restore(prev)


# ---------------------------------------------------------------------------
# Suite rules — what makes a row a verdict
# ---------------------------------------------------------------------------


class TestSuiteRules:

    def test_minimal_suite_loads_with_class_budgets(self, tmp_path):
        root = _suite(tmp_path, """
  - id: a
    prompt: "2+2?"
    score: {expect_text_any: ["4"]}
    reference: {answer: "4"}
""")
        s = load_suite(root)
        (t,) = s.tasks
        assert (t.task_class, t.max_rounds, t.timeout_s) == ("qa", 3, 30.0)
        assert t.verdict_source == "predicates"
        assert len(s.sha256) == 64

    @pytest.mark.parametrize("body, msg", [
        ('score: {expect_text_any: ["4"]}\n    acceptance: "{unittest} test_x"\n'
         '    judge: {rubric: r, reference: x}\n    reference: {answer: "4"}',
         "acceptance and judge"),
        ('score: {expect_text_any: ["4"]}\n    acceptance: "true"\n    reference: {answer: "4"}',
         "must be '{unittest}"),
        ('reference: {answer: "4"}', "no verdict path"),
        ('score: {expect_text_any: ["4"]}', "reference solution"),
        ('score: {expect_nonsense: 1}\n    reference: {answer: "4"}', "unknown score predicate"),
        ('judge: {rubric: r}', "rubric and a reference"),
        ('score: {expect_text_any: ["4"]}\n    reference: {answer: "4"}\n    web: offline',
         "answer_changes_over_time"),
        ('score: {expect_text_any: ["4"]}\n    reference: {answer: "4"}\n    web: offline\n'
         '    answer_changes_over_time: true', "cannot change over time"),
    ])
    def test_refusals(self, tmp_path, body, msg):
        root = _suite(tmp_path, f"  - id: a\n    prompt: q\n    {body}\n")
        with pytest.raises(ValueError, match=msg):
            load_suite(root)

    @pytest.mark.parametrize("extra, msg", [
        ("  long_haul: {status: deferred, max_rounds: 40, max_tool_calls: 60, max_tokens: 4096,"
         " timeout_s: 1500}\n", "success_criterion"),
        ("  long_haul: {status: deferred, file: lh.yaml, max_rounds: 40, max_tool_calls: 60,"
         " max_tokens: 4096, timeout_s: 1500, success_criterion: c}\n", "deferred class has no task file"),
        ("  long_haul: {status: active, max_rounds: 40, max_tool_calls: 60, max_tokens: 4096,"
         " timeout_s: 1500, success_criterion: c}\n", "needs a task file"),
        ("  long_haul: {status: someday, max_rounds: 40, max_tool_calls: 60, max_tokens: 4096,"
         " timeout_s: 1500, success_criterion: c}\n", "status must be"),
    ])
    def test_class_definition_rules(self, tmp_path, extra, msg):
        root = _suite(tmp_path, '  - id: a\n    prompt: q\n    judge: {rubric: r, reference: x}\n')
        (root / "suite.yaml").write_text(SUITE_META + extra)
        with pytest.raises(ValueError, match=msg):
            load_suite(root)

    def test_a_deferred_class_is_defined_with_no_tasks(self, tmp_path):
        root = _suite(tmp_path, '  - id: a\n    prompt: q\n    judge: {rubric: r, reference: x}\n')
        (root / "suite.yaml").write_text(
            SUITE_META + "  long_haul: {status: deferred, max_rounds: 40, max_tool_calls: 60,"
            " max_tokens: 4096, timeout_s: 1500, success_criterion: hidden tests pass}\n")
        s = load_suite(root)
        assert (s.active_classes, s.deferred_classes) == (["qa"], ["long_haul"])
        assert s.classes["long_haul"].success_criterion == "hidden tests pass"
        assert s.by_class()["long_haul"] == []

    def test_duplicate_id_refused(self, tmp_path):
        one = '  - id: a\n    prompt: q\n    judge: {rubric: r, reference: x}\n'
        with pytest.raises(ValueError, match="duplicate"):
            load_suite(_suite(tmp_path, one + one))

    def test_editing_a_class_file_changes_the_sha(self, tmp_path):
        root = _suite(tmp_path, '  - id: a\n    prompt: q\n    judge: {rubric: r, reference: x}\n')
        before = load_suite(root).sha256
        (root / "qa.yaml").write_text((root / "qa.yaml").read_text().replace("q\n", "q2\n"))
        assert load_suite(root).sha256 != before

    def test_shared_fixtures_merge_and_live_web_is_opt_in(self, tmp_path):
        head = ("fixtures:\n  web_pages:\n    - {url: 'https://a.example/x', title: A, body: alpha}\n")
        root = _suite(tmp_path, """
  - id: o1
    prompt: q
    web: offline
    answer_changes_over_time: false
    fixtures:
      web_pages: [{url: 'https://b.example/y', title: B, body: beta}]
    score: {expect_text_any: ["alpha"]}
    reference: {answer: alpha}
  - id: l1
    prompt: q
    web: live
    answer_changes_over_time: true
    judge: {rubric: r, reference: x}
""", head=head)
        s = load_suite(root)
        off = next(t for t in s.tasks if t.id == "o1")
        assert [p["url"] for p in off.fixtures["web_pages"]] == [
            "https://a.example/x", "https://b.example/y"]
        assert [t.id for t in select_tasks(s)] == ["o1"]
        assert [t.id for t in select_tasks(s, include_live_web=True)] == ["o1", "l1"]

    def test_yaml_boolean_id_is_refused_by_name(self, tmp_path):
        root = _suite(tmp_path, "  - id: off\n    prompt: q\n    judge: {rubric: r, reference: x}\n")
        with pytest.raises(ValueError, match="quote it"):
            load_suite(root)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


class TestPredicates:

    def test_text_predicates(self, tmp_path):
        spec = {"expect_text_any": ["714"], "expect_text_all": ["forty"],
                "expect_text_regex": r"\bok\b", "forbid_text": ["741"]}
        assert check_predicates(spec, _transcript("714 ok forty"), tmp_path)[0]
        ok, reasons = check_predicates(spec, _transcript("741 ok forty"), tmp_path)
        assert not ok and any("forbidden" in r for r in reasons)

    @pytest.mark.parametrize("reply, verdict, line", [
        ("Working: 47 × 23 is 1081, not 1071.\nANSWER: 1081", "pass", True),
        ("**ANSWER:** 1,081", "pass", True),
        ("Final answer: $\\boxed{1{,}081}$.", "pass", True),
        ("ANSWER:\n\n1081", "pass", True),                  # value on the next line
        # the right value discussed, a wrong one committed
        ("1081 would be the product of 47 and 23, but\nANSWER: 1071", "fail", True),
        # an earlier draft line does not count — the LAST answer line does
        ("ANSWER: 1081\nOn reflection:\nANSWER: 1071", "fail", True),
        # no answer line, but the bare last line still gives the answer:
        # scored, and the missing line recorded on its own
        ("47 × 23 is\n1081", "pass", False),
        ("Multiplying out.\nSo the answer is 1081.", "pass", False),
        ("Multiplying out.\n1071", "fail", False),           # a bare value of the answer's shape
        # no answer line, answer not isolable: a FORMAT MISS, never "wrong"
        ("47 × 23 = 1081, and 1081 is prime-free of 7.", "format_miss", False),
        ("I multiplied the numbers.", "format_miss", False),
        ("ANSWER:", "format_miss", False),                  # a label with nothing is no line
    ])
    def test_the_answer_line_and_the_format_miss_rule(self, tmp_path, reply, verdict, line):
        task = _task(score={"expect_answer": r"1[,\s]?081", "answer_shape": r"[\d,\s]+"},
                     reference={"answer": "ANSWER: 1081"})
        out = asyncio.run(decide(task, _transcript(reply), tmp_path))
        assert (out.verdict, out.answer_format_ok) == (verdict, line), (reply, out.fail_reasons)
        assert out.success is {"pass": True, "fail": False, "format_miss": None}[verdict]

    PLANET = r"(?:the planet )?(?:mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)"
    NUM = r"-?\d+(?:[,\s]\d{3})*(?:\.\d+)?"
    REGION = r"(?:region\s*=\s*)?[a-z]+-[a-z]+-\d+"
    MUT, MUT_SHAPE = r"1[\s,]+2[\s,]+1[\s,]+3", r"\(?-?\d+(?:[\s,]+-?\d+)*\)?"
    SETTLE, PY_PATH = r"(?:(?:\./|/\S*/)?src/)?billing/settle\.py(?::\d+)?", r"[\w./-]+\.py(?::\d+)?"

    @pytest.mark.parametrize("text, expect, shape, want", [
        # a hedge commits to nothing: two candidate values -> format miss
        ("ANSWER: Saturn or Jupiter", "saturn", PLANET, ("miss", True)),
        ("ANSWER: Saturn (or Jupiter)", "saturn", PLANET, ("miss", True)),
        # a parenthetical may be a qualifier ("(minutes)"): never stripped into a pass
        ("ANSWER: 12,600 (minutes)", r"12[,\s]?600", NUM, ("miss", True)),
        ("ANSWER: 3712 (from base.ini)", "3712", NUM, ("miss", True)),
        # one candidate value of the answer's kind in the committed line decides
        ("ANSWER: The port is 48217.", "48217", NUM, ("pass", True)),
        ("ANSWER: 0.0.0.0", "48217", NUM, ("fail", True)),
        ("ANSWER: 48217, not 8080", "48217", NUM, ("miss", True)),
        # a derivation under a bare label falls back to the reply's last line
        ("**Answer:**\n\n47 × 23 = 940 + 141\n\n1081", r"1[,\s]?081", NUM, ("pass", False)),
        # a list or display math under a bare label is read as one value
        ("ANSWER:\n- a.sql\n- b.sql", r"a\.sql,\s*b\.sql", r"\w+\.sql(?:,\s*\w+\.sql)*", ("pass", True)),
        ("Chain done.\n\n**Final Answer:**\n\\[\n\\boxed{3712}\n\\]", "3712", NUM, ("pass", True)),
        # a reply that ENDS quoting a file is not answering with its last line
        ("Chain:\n```\nregion=cinder-coast-17\nzone=b\n```", "aurora-basin-42",
         r"(?:region\s*=\s*)?[a-z]+-[a-z]+-\d+", ("miss", False)),
        ("Found:\n```\nPG-7351-KX\n```", "PG-7351-KX", r"PG-\d{4}-[A-Z]{2}", ("pass", False)),
        # a sign-off is not an answer when the shape is the answer's kind
        ("Checked.\nLet me know if you need anything else.", "brennwick",
         r"(?:odalys\s+)?(?:brennwick|qarsen|vasquine)", ("miss", False)),
        ("ANSWER: <the number>", r"1[,\s]?081", NUM, ("miss", True)),
        # prose without a line is never FAILED: it may be a procedure, not an answer
        ("An hour has 3,600 seconds, so multiply that by three and a half.",
         r"12[,\s]?600", NUM, ("miss", False)),
        ("The configured port is 48217.", "48217", NUM, ("pass", False)),
        # a clean token committed on the line, of the wrong kind altogether: wrong
        ("ANSWER: 9c41e07", r"[a-z]+-[a-z]+-\d{4}", r"[a-z]+-[a-z]+-\d{4}", ("fail", True)),
        ("ANSWER: It is not 48217", "48217", NUM, ("miss", True)),
        # a list of candidates commits to nothing — with a label or without, in
        # either order: never decided by whichever item happens to come last
        ("ANSWER:\n- cinder-coast-17\n- aurora-basin-42", "aurora-basin-42", REGION, ("miss", True)),
        ("ANSWER:\n1. aurora-basin-42\n2. cinder-coast-17", "aurora-basin-42", REGION, ("miss", True)),
        ("Possible regions:\n- cinder-coast-17\n- aurora-basin-42", "aurora-basin-42", REGION,
         ("miss", False)),
        ("Possible regions:\n- aurora-basin-42\n\n- cinder-coast-17", "aurora-basin-42", REGION,
         ("miss", False)),
        # REAL replies (qwen2.5:7b dry run, 2026-09-25) the reader misread — verbatim
        # but for the sandbox path. A tuple's own brackets are not a qualifier:
        ("So the final answer:\n\nANSWER: (1, 2, 1, 4)", MUT, MUT_SHAPE, ("fail", True)),
        # ...but where the task's shape says a bracketed sequence is a value of its kind (an
        # exact-output question), a printed tuple is a different, wrong output (a real
        # Ornith-1.5-9B reply, 2026-09-26)
        ("`print` displays the tuple: `(1, 2, 1, 3)`.\n\nANSWER: (1, 2, 1, 3)", MUT, MUT_SHAPE,
         ("fail", True)),
        ("ANSWER: (1, 2, 1, 3)", MUT, r"-?\d+(?:[\s,]+-?\d+)*", ("pass", True)),
        # a committed line that denies there is an answer, naming no value of its kind:
        ("It appears that the function `reconcile_ledger_v3` is not defined in any Python file "
         "within this project. \n\nANSWER: /tmp/sb/ws does not contain a definition for "
         "`reconcile_ledger_v3`.", SETTLE, PY_PATH, ("fail", True)),
        ("Given that no matches were found for the 503 status code in `logs/access.log`, it "
         "appears there might be an issue with the log or the search criteria.\n\nANSWER: No "
         "request ID found for HTTP status 503.", r"(?:req-)?f36ea8da", r"(?:req-)?[0-9a-f]{8}",
         ("fail", True)),
        # the answer line written at the end of the last sentence is still the line
        ("The port configured in settings.ini is 48217. ANSWER: 48217", "48217", NUM, ("pass", True)),
        # REAL reply (Bonsai 2 27B smoke, 2026-09-25), verbatim: the value is stated, and the
        # "no" belongs to the NEXT sentence — it negates nothing the reply committed to
        ("The function `reconcile_ledger_v3` is defined in `src/billing/settle.py`. Let me confirm "
         "there's no other definition elsewhere.\n\n", SETTLE, PY_PATH, ("pass", False)),
        # ...while a negation in the value's own sentence still commits to nothing
        ("It is not in src/billing/settle.py.", SETTLE, PY_PATH, ("miss", False)),
    ])
    def test_the_reader_credits_or_fails_only_what_is_unambiguous(self, text, expect, shape, want):
        from prometheus.gym.ladder.verdict import read_answer

        got = read_answer(text, expect, shape)
        assert got[:2] == want, (text, got)

    def test_without_a_shape_a_lineless_wrong_value_is_a_format_miss_not_wrong(self, tmp_path):
        task = _task(score={"expect_answer": r"1[,\s]?081"}, reference={"answer": "ANSWER: 1081"})
        out = asyncio.run(decide(task, _transcript("1071"), tmp_path))
        assert (out.verdict, out.success) == ("format_miss", None)

    def test_answer_format_is_none_where_no_line_was_asked(self, tmp_path):
        other = _task(score={"expect_text_any": ["x"]}, reference={"answer": "x"})
        assert asyncio.run(decide(other, _transcript("x"), tmp_path)).answer_format_ok is None

    def test_answer_shape_needs_an_answer_check(self, tmp_path):
        root = _suite(tmp_path, "  - id: a\n    prompt: q\n    score: {answer_shape: '[0-9]+'}\n"
                                "    reference: {answer: '1'}\n")
        with pytest.raises(ValueError, match="answer_shape only"):
            load_suite(root)

    def test_tool_predicates_respect_the_emission_view(self, tmp_path):
        repaired = ToolEvent(name="Bash", input={}, is_error=False, exec_name="bash",
                             repaired=True)
        t = _transcript("done", [repaired])
        spec = {"expect_tool_any": ["bash"], "expect_tools_all": ["bash"]}
        assert check_predicates(spec, t, tmp_path, view=EXECUTION)[0]
        assert not check_predicates(spec, t, tmp_path, view=EMISSION)[0]

    def test_file_predicates(self, tmp_path):
        (tmp_path / "out.txt").write_text("total 1482\n")
        spec = {"expect_file_regex": {"path": "out.txt", "pattern": r"1482"},
                "expect_file_absent": "gone.txt"}
        assert check_predicates(spec, _transcript(""), tmp_path)[0]
        (tmp_path / "gone.txt").write_text("x")
        assert not check_predicates(spec, _transcript(""), tmp_path)[0]

    def test_cron_predicates_read_the_sandbox_registry(self, sandbox):
        fx.seed_cron_jobs([{"name": "backup", "schedule": "0 3 * * *",
                            "command": "tar -czf b.tgz ."}], sandbox.workspace)
        ok_spec = {"expect_cron_job": {"name": "backup", "schedule_any": ["0 3 * * *"],
                                       "command_contains": "tar"}}
        assert check_predicates(ok_spec, _transcript(""), sandbox.workspace)[0]
        bad = {"expect_cron_job": {"name": "backup", "schedule_any": ["0 2 * * *"]}}
        ok, reasons = check_predicates(bad, _transcript(""), sandbox.workspace)
        assert not ok and "schedule" in reasons[0]
        assert not check_predicates({"forbid_cron_job": "backup"}, _transcript(""),
                                    sandbox.workspace)[0]
        # and it is the SANDBOX's registry, not the machine's
        assert Path(os.environ["PROMETHEUS_DATA_DIR"]) == sandbox.data
        assert (sandbox.data / "cron_jobs.json").exists()


# ---------------------------------------------------------------------------
# Acceptance commands
# ---------------------------------------------------------------------------


OK_TEST = "import unittest\nfrom mod import X\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(X, 1)\n"
BAD_TEST = OK_TEST.replace("X, 1", "X, 2")


class TestAcceptance:
    """The runner owns the verdict: tests must RUN, and nothing the workspace
    holds can turn an unrun or failing suite into exit 0 = pass."""

    def _run(self, tmp_path, files, accept_files=None, command="{unittest} test_mod", **kw):
        ws = tmp_path / "ws"
        ws.mkdir(exist_ok=True)
        fx.write_files(ws, files)
        return run_acceptance(command, accept_files or {}, ws, home=tmp_path, **kw)

    def test_passing_tests_pass_and_are_counted(self, tmp_path):
        r = self._run(tmp_path, {"mod.py": "X = 1\n", "test_mod.py": OK_TEST})
        assert (r["status"], r["passed"], r["tests_run"]) == ("ran", True, 1), r

    def test_failing_tests_fail(self, tmp_path):
        r = self._run(tmp_path, {"mod.py": "X = 1\n", "test_mod.py": BAD_TEST})
        assert (r["passed"], r["failures"]) == (False, 1), r

    def test_no_tests_is_a_fail(self, tmp_path):
        r = self._run(tmp_path, {"test_mod.py": "import unittest\n"})
        assert r["passed"] is False and "no tests ran" in r["error"], r

    def test_exit_zero_at_import_is_a_fail(self, tmp_path):
        # A module-level main() with no __main__ guard.
        r = self._run(tmp_path, {"mod.py": "import sys\nX = 1\nsys.exit(0)\n", "test_mod.py": OK_TEST})
        assert r["passed"] is False, r

    def test_killing_the_interpreter_is_a_fail(self, tmp_path):
        r = self._run(tmp_path, {"mod.py": "import os\nos._exit(0)\n", "test_mod.py": OK_TEST})
        assert r["passed"] is False and "no result" in r["error"], r

    def test_a_workspace_unittest_cannot_replace_the_runner(self, tmp_path):
        r = self._run(tmp_path, {"mod.py": "X = 1\n", "test_mod.py": BAD_TEST,
                                 "unittest.py": "import sys\nsys.exit(0)\n",
                                 "json.py": "raise SystemExit(0)\n"})
        assert r["passed"] is False and r["failures"] == 1, r

    def test_edited_or_shadowed_tests_are_restored(self, tmp_path):
        files = {"mod.py": "X = 1\n", "test_mod.py": OK_TEST.replace("X, 1", "1, 1"),
                 "test_mod/__init__.py": "", "test_mod/test_ok.py": OK_TEST.replace("X, 1", "1, 1")}
        r = self._run(tmp_path, files, accept_files={"test_mod.py": BAD_TEST})
        assert r["passed"] is False and r["failures"] == 1, r

    def test_a_symlinked_or_readonly_test_is_replaced_not_unscored(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        fx.write_files(ws, {"mod.py": "X = 1\n"})
        (ws / "test_mod.py").symlink_to("/etc/hosts")
        r = run_acceptance("{unittest} test_mod", {"test_mod.py": BAD_TEST}, ws, home=tmp_path)
        assert (r["status"], r["passed"]) == ("ran", False), r
        assert not (ws / "test_mod.py").is_symlink()

    def test_a_hang_ran_and_failed(self, tmp_path):
        slow = "import time, unittest\nclass T(unittest.TestCase):\n    def test(self):\n        time.sleep(30)\n"
        r = self._run(tmp_path, {"test_mod.py": slow}, timeout_s=2)
        assert (r["status"], r["passed"]) == ("ran", False) and "timed out" in r["error"], r

    def test_a_free_form_command_is_not_run(self, tmp_path):
        r = self._run(tmp_path, {}, command="true")
        assert (r["status"], r["passed"]) == ("not_run", None), r

    def test_the_check_sees_the_sandbox_stores(self, sandbox):
        probe = ("import os, unittest\nclass T(unittest.TestCase):\n    def test(self):\n"
                 f"        self.assertEqual(os.environ['PROMETHEUS_DATA_DIR'], {str(sandbox.data)!r})\n")
        fx.write_files(sandbox.workspace, {"test_env.py": probe})
        r = run_acceptance("{unittest} test_env", {}, sandbox.workspace, home=sandbox.home)
        assert r["passed"] is True, r

    def test_a_runner_that_cannot_start_is_unscored_never_pass(self, tmp_path, monkeypatch):
        import subprocess

        def boom(*a, **k):
            raise OSError("no interpreter")

        monkeypatch.setattr(subprocess, "run", boom)
        r = self._run(tmp_path, {"test_mod.py": OK_TEST})
        assert (r["status"], r["passed"]) == ("not_run", None)
        task = _task(acceptance="{unittest} test_mod", reference={"files": {}})
        out = asyncio.run(decide(task, _transcript(""), tmp_path / "ws", home=tmp_path))
        assert (out.verdict, out.success) == ("unscored", None)


# ---------------------------------------------------------------------------
# The judge — a judge that cannot answer is not a verdict
# ---------------------------------------------------------------------------


class _Judge:
    def __init__(self, verdict=None, exc=None):
        self.verdict, self.exc, self.calls = verdict, exc, 0

    async def evaluate(self, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.verdict

    def provenance(self):
        return {"base_url": "http://judge", "model": "judge-model", "pinned": True}


JUDGED = dict(judge={"rubric": "r", "reference": "x", "threshold": 0.7})


class TestJudge:

    @pytest.mark.parametrize("judge", [
        _Judge(exc=RuntimeError("503")),
        _Judge(JudgeVerdict(score=0.0, reasoning="Empty response", raw_response="")),
        _Judge(JudgeVerdict(score=1.0, reasoning="Parse fallback: 1", raw_response="1 lol")),
        None,
    ])
    def test_unavailable_judge_is_unscored(self, tmp_path, judge):
        out = asyncio.run(decide(_task(**JUDGED), _transcript("answer"), tmp_path, judge=judge))
        assert (out.verdict, out.success) == ("unscored", None)

    @pytest.mark.parametrize("raw", ['{"reasoning": "fine"}', '{"score": 7, "reasoning": "x"}',
                                     '{"score": true}', '{"score": "0.9"}', '{"score": NaN}'])
    def test_only_a_zero_to_one_score_counts(self, tmp_path, raw):
        j = _Judge(JudgeVerdict(score=1.0, reasoning="x", raw_response=raw))
        out = asyncio.run(decide(_task(**JUDGED), _transcript("a"), tmp_path, judge=j))
        assert (out.verdict, out.success) == ("unscored", None), raw

    def test_score_against_threshold(self, tmp_path):
        lo = _Judge(JudgeVerdict(score=0.5, reasoning="meh", raw_response='{"score":0.5}'))
        hi = _Judge(JudgeVerdict(score=0.9, reasoning="good", raw_response='{"score":0.9}'))
        assert asyncio.run(decide(_task(**JUDGED), _transcript("a"), tmp_path, judge=lo)).verdict == "fail"
        out = asyncio.run(decide(_task(**JUDGED), _transcript("a"), tmp_path, judge=hi))
        assert out.verdict == "pass" and out.judge["provenance"]["pinned"] is True
        # the endpoint does not persist: telemetry.db travels between machines
        assert "base_url" not in out.judge["provenance"]

    def test_judge_not_asked_when_predicates_already_failed(self, tmp_path):
        j = _Judge(JudgeVerdict(score=1.0, reasoning="x", raw_response="{}"))
        task = _task(score={"expect_text_any": ["zebra"]}, reference={"answer": "zebra"}, **JUDGED)
        out = asyncio.run(decide(task, _transcript("horse"), tmp_path, judge=j))
        assert out.verdict == "fail" and j.calls == 0

    def test_crash_is_error_and_a_halt_is_fail(self, tmp_path):
        t = _task(score={"expect_text_any": ["x"]}, reference={"answer": "x"})
        crash = asyncio.run(decide(t, _transcript("x"), tmp_path, harness_error="ConnectError"))
        slow = asyncio.run(decide(t, _transcript("x"), tmp_path, halted="time budget exceeded"))
        assert (crash.verdict, crash.success) == ("error", None)
        assert (slow.verdict, slow.success, slow.fail_reasons) == (
            "fail", False, ["time budget exceeded"])

    def test_emission_view_only_where_a_check_can_tell(self, tmp_path):
        text_only = _task(score={"expect_text_any": ["x"]}, reference={"answer": "x"})
        assert asyncio.run(decide(text_only, _transcript("x"), tmp_path)).emission_pass is None
        repaired = ToolEvent(name="Bash", input={}, is_error=False, exec_name="bash", repaired=True)
        tool_task = _task(score={"expect_tool_any": ["bash"]}, reference={"answer": "x"})
        out = asyncio.run(decide(tool_task, _transcript("x", [repaired]), tmp_path))
        assert (out.verdict, out.emission_pass) == ("pass", False)
        failed = asyncio.run(decide(tool_task, _transcript("x"), tmp_path))
        assert (failed.verdict, failed.emission_pass) == ("fail", False)

    def test_a_directory_where_a_file_should_be_fails_not_crashes(self, tmp_path):
        (tmp_path / "out.txt").mkdir()
        t = _task(score={"expect_file_regex": {"path": "out.txt", "pattern": "x"},
                         "expect_file": "out.txt", "expect_file_contains": "x"},
                  reference={"files": {}})
        out = asyncio.run(decide(t, _transcript(""), tmp_path))
        assert (out.verdict, out.success) == ("fail", False), out


class TestJudgePin:

    def _served(self, monkeypatch, ids):
        async def fake(_url):
            return ids
        monkeypatch.setattr(lr, "served_ids", fake)

    def test_unpinned_refused(self, monkeypatch):
        with pytest.raises(lr.LadderPreflightError, match="pinned"):
            asyncio.run(lr.check_judge_pin(lr.JudgePin("http://j", ""), contestant_model="m",
                                           contestant_served=None))

    def test_self_grading_refused_across_name_forms(self, monkeypatch):
        self._served(monkeypatch, ["/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"])
        with pytest.raises(lr.LadderPreflightError, match="nothing grades itself"):
            asyncio.run(lr.check_judge_pin(
                lr.JudgePin("http://j", "qwen3.8-27b-ud-q4_k_xl.gguf"),
                contestant_model="", contestant_served="Qwen3.8-27B-UD-Q4_K_XL.gguf"))

    def test_pin_must_be_what_the_endpoint_serves(self, monkeypatch):
        # llama-server ignores the request's model field: a pin naming one
        # model while the endpoint serves another would grade with the other.
        self._served(monkeypatch, ["/models/gemma-4-26B-A4B-it-Q4_K_M.gguf"])
        with pytest.raises(lr.LadderPreflightError, match="does not serve"):
            asyncio.run(lr.check_judge_pin(lr.JudgePin("http://j", "qwen2.5:14b-instruct"),
                                           contestant_model="qwen2.5:7b-instruct",
                                           contestant_served=None))

    def test_good_pin_passes(self, monkeypatch):
        self._served(monkeypatch, ["qwen2.5:14b-instruct", "qwen2.5:7b-instruct"])
        asyncio.run(lr.check_judge_pin(lr.JudgePin("http://j", "qwen2.5:14b-instruct"),
                                       contestant_model="qwen2.5:7b-instruct",
                                       contestant_served="qwen2.5:7b-instruct"))


@pytest.mark.parametrize("name, quant", [
    ("/m/Qwen3.8-27B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
    ("gemma-4-26B-A4B-it-Q4_K_M.gguf", "Q4_K_M"),
    ("Qwen3.5-2B-q8_0.gguf", "Q8_0"),
    ("x-IQ4_XS.gguf", "IQ4_XS"),
    ("model-BF16.gguf", "BF16"),
    ("qwen2.5:7b-instruct", None),
])
def test_quant_from_filename(name, quant):
    assert lr.quant_from_filename(name) == quant


# ---------------------------------------------------------------------------
# Offline web + sandbox
# ---------------------------------------------------------------------------


class TestOfflineWeb:

    PAGES = [
        {"url": "https://docs.z.example/releases", "title": "Zephyrine release notes",
         "body": "Zephyrine 4.2 released 2031-03-14."},
        {"url": "https://blog.z.example/x", "title": "Unrelated", "body": "gardening tips"},
    ]

    def test_fetch_and_404(self, tmp_path):
        fetch, search = fx.fixture_web_tools(fx.FixtureWeb(self.PAGES))
        ok = asyncio.run(fetch.execute(fetch.input_model(url="http://docs.z.example/releases/"), None))
        assert not ok.is_error and "2031-03-14" in ok.output and "Status: 200" in ok.output
        miss = asyncio.run(fetch.execute(fetch.input_model(url="https://nope.example/"), None))
        assert miss.is_error and "404" in miss.output

    def test_search_ranks_title_matches_and_keeps_the_real_schema(self, tmp_path):
        from prometheus.tools.builtin.web_search import WebSearchTool

        fetch, search = fx.fixture_web_tools(fx.FixtureWeb(self.PAGES))
        assert search.name == "web_search" and search.input_model is WebSearchTool.input_model
        out = asyncio.run(search.execute(search.input_model(query="zephyrine release"), None))
        assert out.output.splitlines()[1] == "1. Zephyrine release notes"

    def test_registry_is_the_same_fourteen_tools(self, tmp_path):
        reg = fx.build_ladder_registry(tmp_path)
        assert sorted(t.name for t in reg.list_tools()) == sorted(fx.LADDER_TOOLS)

    def test_reset_wipes_the_run_but_keeps_the_gates_audit_dir(self, sandbox):
        (sandbox.workspace / "junk").write_text("x")
        (sandbox.data / "cron_jobs.json").write_text("[]")
        (sandbox.data / "security" / "audit.db").write_text("keep")
        sandbox.reset()
        assert not (sandbox.workspace / "junk").exists()
        assert not (sandbox.data / "cron_jobs.json").exists()
        assert (sandbox.data / "security" / "audit.db").read_text() == "keep"


# ---------------------------------------------------------------------------
# Self-check — the discrimination proof itself must discriminate
# ---------------------------------------------------------------------------


class TestSelfCheck:

    def test_loose_check_is_caught_by_wrong_answers(self, tmp_path):
        task = _task(score={"expect_text_any": ["71"]},
                     reference={"answer": "714", "wrong_answers": ["712"]})
        assert any("wrong answer" in p for p in selfcheck_task(task, tmp_path / "r"))

    def test_broken_reference_is_caught(self, tmp_path):
        task = _task(acceptance="{python} -c 'import mod'", reference={"files": {"other.py": ""}})
        assert any("reference solution FAILS" in p for p in selfcheck_task(task, tmp_path / "r"))

    def test_check_that_passes_on_the_setup_is_caught(self, tmp_path):
        task = _task(setup_files={"a.txt": "x"}, score={"expect_file": "a.txt"},
                     reference={"files": {}})
        assert any("untouched setup" in p for p in selfcheck_task(task, tmp_path / "r"))

    def test_wrong_file_state_is_caught(self, tmp_path):
        # A regex that only asks for "1482" somewhere would accept "11482".
        task = _task(score={"expect_file_regex": {"path": "t.txt", "pattern": "1482"}},
                     reference={"files": {"t.txt": "1482\n"},
                                "wrong_files": [{"t.txt": "11482\n"}]})
        assert any("wrong_files[0]" in p for p in selfcheck_task(task, tmp_path / "r"))
        tight = _task(score={"expect_file_regex": {"path": "t.txt", "pattern": r"\A1482\s*\Z"}},
                      reference={"files": {"t.txt": "1482\n"},
                                 "wrong_files": [{"t.txt": "11482\n"}]})
        assert selfcheck_task(tight, tmp_path / "r2") == []

    def test_wrong_cron_state_is_caught(self, tmp_path):
        loose = _task(score={"expect_cron_job": {"name": "b", "schedule_regex": r"0 \d+ \* \* \*"}},
                      reference={"cron_jobs": [{"name": "b", "schedule": "0 3 * * *", "command": "echo"}],
                                 "wrong_cron_jobs": [[{"name": "b", "schedule": "0 4 * * *",
                                                       "command": "echo"}]]})
        assert any("wrong_cron_jobs[0]" in p for p in selfcheck_task(loose, tmp_path / "r"))

    def test_synthetic_transcript_counts_tools(self):
        t = synthetic_transcript("done", ["read_file", "bash"])
        assert [e.name for e in t.tool_events] == ["read_file", "bash"]
        assert t.final_text == "done"


# ---------------------------------------------------------------------------
# telemetry.db — harvest, record, read back, empty-field check
# ---------------------------------------------------------------------------


def _tel(tmp_path) -> ToolCallTelemetry:
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


class TestTelemetry:

    def test_harvest_joins_by_session_and_attributes_sessionless_rows_by_window(self, tmp_path):
        tel = _tel(tmp_path)
        sid = "ladder:x:t:0:abc"
        t0 = time.time()
        tel.record("m", "bash", True, session_id=sid, repairs=2)
        tel.record("m", "bash", False, error_type="nonzero_exit", session_id=sid)
        tel.record("m", "read_file", False, error_type="validation", session_id=sid, retries=1)
        tel.record("m", "_loop_transition", True, session_id=sid)
        tel.record("m", "write_file", False, error_type="permission_denied")  # no session id
        tel.record("m", "bash", True, session_id="another-run")
        tel.record_run("agent_loop", "loop_round", "success", input_tokens=100,
                       output_tokens=7, session_id=sid)
        tel.record_run("agent_loop", "loop_round", "success", input_tokens=150,
                       output_tokens=9, session_id=sid)
        tel.record_run("agent_loop", "tool_advertisement", "success", session_id=sid)
        m = rec.harvest_run_metrics(tel._conn, sid, window=(t0 - 1, time.time() + 1))
        assert (m["tool_calls"], m["tool_calls_ok"], m["tool_calls_excluded"]) == (4, 1, 2)
        assert m["tool_call_success"] == 0.5          # 1 ok / (4 - 2 non-call failures)
        assert (m["repairs"], m["retries"], m["tool_calls_denied"]) == (2, 1, 1)
        assert m["tool_calls_unattributed"] == 1
        assert (m["rounds"], m["input_tokens"], m["output_tokens"]) == (2, 250, 16)
        assert m["tokens_source"] == "provider"

    def test_unreported_tokens_are_null_not_zero(self, tmp_path):
        tel = _tel(tmp_path)
        tel.record_run("agent_loop", "loop_round", "success", input_tokens=0,
                       output_tokens=0, session_id="s")
        m = rec.harvest_run_metrics(tel._conn, "s")
        assert (m["input_tokens"], m["output_tokens"], m["tokens_source"]) == (None, None, "unreported")
        assert m["tool_call_success"] is None and m["tool_calls"] == 0

    def test_summary_row_lands_and_reads_back(self, tmp_path):
        tel = _tel(tmp_path)
        summary = {"session_id": "s1", "task_class": "qa", "verdict": "pass",
                   "duration_ms": 12.0, "model": "m", "run_label": "L",
                   "input_tokens": 10, "output_tokens": 2, "thinking": None}
        rec.record_summary(tel, summary)
        row = tel._conn.execute(
            "SELECT operation, outcome, input_tokens, model FROM subsystem_runs "
            "WHERE subsystem='model_ladder'").fetchone()
        # tokens live on the loop_round rows; a copy here would double-count
        # in usage_rollup (/api/usage)
        assert row == ("qa", "success", None, "m")
        assert rec.load_rows(tel._conn, "L")[0]["task_class"] == "qa"

    def test_a_row_that_did_not_land_raises(self, tmp_path):
        tel = _tel(tmp_path)
        tel.record_run = lambda *a, **k: None   # record_run swallows its own failures
        with pytest.raises(rec.LadderRecordError):
            rec.record_summary(tel, {"session_id": "s", "task_class": "qa", "verdict": "fail",
                                     "duration_ms": 1.0, "model": "m"})

    @pytest.mark.parametrize("verdict, outcome", [
        ("pass", "success"), ("fail", "failed"), ("format_miss", "partial"),
        ("unscored", "skipped"), ("error", "skipped")])
    def test_outcome_mapping_never_turns_undecided_into_success(self, verdict, outcome):
        assert rec.outcome_for(verdict) == outcome

    def test_empty_field_check(self):
        full = {f: 1 for f in rec.REQUIRED_FIELDS}
        assert rec.check_empty_fields([full, {**full, "tool_call_success": None}]) == []
        rows = [{**full, "input_tokens": None}, {**full, "input_tokens": None}]
        assert rec.check_empty_fields(rows) == ["input_tokens"]
        assert rec.check_empty_fields([{**full, "quantization": "unknown"}]) == ["quantization"]
        assert rec.check_empty_fields([]) == list(rec.REQUIRED_FIELDS)

    def test_report_names_no_hosts(self):
        row = {f: 1 for f in rec.REQUIRED_FIELDS}
        row.update(task_id="t", task_class="qa", verdict="pass", verdict_source="predicates",
                   duration_ms=1000.0, suite="s", suite_sha="0" * 64, run_label="L",
                   provider="ollama", model="m", served_models=[],
                   judge={"provenance": {"base_url": "http://secret-host:8080",
                                         "model": "j", "pinned": True}})
        text = rec.render_report([row], title="x", class_order=["qa"])
        assert "secret-host" not in text and "| qa | 1 | 1 |" in text


# ---------------------------------------------------------------------------
# End to end through the real agent loop
# ---------------------------------------------------------------------------


class _Scripted(ModelProvider):
    """Round 1: write a file. Round 2: say done. Reports usage every round."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.calls += 1
        if self.calls == 1:
            msg = ConversationMessage(role="assistant", content=[ToolUseBlock(
                id="toolu_1", name="write_file",
                input={"path": self.path, "content": "1482\n"})])
        else:
            msg = ConversationMessage(role="assistant", content=[TextBlock(text="Wrote 1482.")])
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(input_tokens=200 + self.calls, output_tokens=5),
            stop_reason="stop")


def test_end_to_end_run_is_recorded_and_joinable(tmp_path):
    from prometheus.__main__ import create_adapter, create_security_gate

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    (suite_root / "suite.yaml").write_text(SUITE_META)
    (suite_root / "qa.yaml").write_text("""class: qa
tasks:
  - id: write-total
    prompt: "Write 1482 to {workspace}/total.txt"
    score:
      expect_file_regex: {path: total.txt, pattern: '^1482$'}
      expect_text_any: ["1482"]
    reference: {files: {total.txt: "1482\\n"}, answer: "1482"}
""")
    suite = load_suite(suite_root)
    sandbox = fx.Sandbox(tmp_path / "sb")
    prev = sandbox.activate()
    try:
        sandbox.reset()
        provider = _Scripted(str(sandbox.workspace / "total.txt"))
        model_cfg = {"provider": "llama_cpp", "model": "qwen2.5-7b-instruct",
                     "grammar_enforcement": False}
        pipeline = {
            "provider": provider,
            "adapter_factory": lambda: create_adapter(model_cfg),
            "security_gate": create_security_gate({"workspace_root": str(sandbox.workspace)}),
            "model_name": "qwen2.5-7b-instruct",
            "model_cfg": model_cfg,
        }
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        static = {"suite": suite.name, "suite_sha": suite.sha256, "run_label": "e2e",
                  "provider": "llama_cpp", "model": "qwen2.5-7b-instruct",
                  "quantization": "Q4_K_M", "adapter_tier": "light",
                  "adapter_strictness": "NONE"}
        row = asyncio.run(lr.run_task(
            suite.tasks[0], suite, pipeline, sandbox=sandbox, tel=tel, judge=None,
            run_label="e2e", run_idx=0, static=static))
    finally:
        fx.Sandbox.restore(prev)

    assert row["verdict"] == "pass", row["fail_reasons"]
    assert (row["rounds"], row["tool_calls"], row["tool_calls_ok"]) == (2, 1, 1)
    assert (row["input_tokens"], row["output_tokens"]) == (403, 10)
    assert row["trace"][0]["tool"] == "write_file"

    conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
    rows = rec.load_rows(conn, "e2e")
    assert len(rows) == 1 and rec.check_empty_fields(rows) == []
    sid = rows[0]["session_id"]
    # the pipeline's own rows carry the same session id — the join works
    n_calls = conn.execute("SELECT COUNT(*) FROM tool_calls WHERE session_id = ? "
                           "AND tool_name = 'write_file'", (sid,)).fetchone()[0]
    n_rounds = conn.execute("SELECT COUNT(*) FROM subsystem_runs WHERE session_id = ? "
                            "AND subsystem = 'agent_loop' AND operation = 'loop_round'",
                            (sid,)).fetchone()[0]
    assert (n_calls, n_rounds) == (1, 2)
    # and the env is back the way it was
    assert os.environ.get("PROMETHEUS_DATA_DIR") != str(sandbox.data)
    assert json.loads(conn.execute(
        "SELECT summary_json FROM subsystem_runs WHERE subsystem='model_ladder'"
    ).fetchone()[0])["verdict_source"] == "predicates"


class _Silent(ModelProvider):
    """Answers with no text at all."""

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="")]),
            usage=UsageSnapshot(input_tokens=50, output_tokens=0), stop_reason="stop")


def test_a_seeded_answer_is_not_the_models_answer(tmp_path):
    """The seed says the code; the model says nothing. Scored over the whole
    message list, the last seeded assistant turn would be read as the model's
    final answer and the run would PASS."""
    from prometheus.__main__ import create_adapter, create_security_gate

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    (suite_root / "suite.yaml").write_text(SUITE_META)
    (suite_root / "qa.yaml").write_text("""class: qa
tasks:
  - id: seeded
    prompt: "What was the door code?"
    seed:
      - {user: "Remember: the door code is 4471."}
      - {assistant_tool_call: {name: read_file, input: {path: "{workspace}/x.txt"}}}
      - {tool_result: {content: "ok", is_error: false}}
      - {assistant_text: "Noted, the door code is 4471."}
    score: {expect_text_any: ["4471"], expect_tool_any: [read_file]}
    reference: {answer: "4471", tools: [read_file]}
""")
    suite = load_suite(suite_root)
    sandbox = fx.Sandbox(tmp_path / "sb")
    prev = sandbox.activate()
    try:
        sandbox.reset()
        model_cfg = {"provider": "llama_cpp", "model": "qwen2.5-7b-instruct",
                     "grammar_enforcement": False}
        pipeline = {"provider": _Silent(), "adapter_factory": lambda: create_adapter(model_cfg),
                    "security_gate": create_security_gate({"workspace_root": str(sandbox.workspace)}),
                    "model_name": "qwen2.5-7b-instruct", "model_cfg": model_cfg}
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        row = asyncio.run(lr.run_task(
            suite.tasks[0], suite, pipeline, sandbox=sandbox, tel=tel, judge=None,
            run_label="seed", run_idx=0, static={"model": "m", "run_label": "seed"}))
    finally:
        fx.Sandbox.restore(prev)
    assert row["verdict"] == "fail", row
    assert row["tools_called"] == []


def test_workspace_expands_inside_seeded_tool_calls(tmp_path):
    seed = [{"assistant_tool_call": {"name": "read_file", "input": {"path": "{workspace}/a"}}}]
    assert lr.expand_deep(seed, tmp_path)[0]["assistant_tool_call"]["input"]["path"] == f"{tmp_path}/a"


class _Looping(ModelProvider):
    """Never stops calling tools."""

    def __init__(self) -> None:
        self.n = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.n += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[ToolUseBlock(
                id=f"toolu_{self.n}", name="glob", input={"pattern": f"*{self.n}"})]),
            usage=UsageSnapshot(input_tokens=60, output_tokens=5), stop_reason="tool_use")


def test_running_out_of_rounds_is_a_fail_not_a_crash(tmp_path):
    """run_loop RAISES when max_turns is exhausted; recorded as a harness
    error that would make a model that never finishes read as 'unscored'."""
    from prometheus.__main__ import create_adapter, create_security_gate

    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    (suite_root / "suite.yaml").write_text(SUITE_META)
    (suite_root / "qa.yaml").write_text("""class: qa
tasks:
  - id: loops
    prompt: "Say hi"
    score: {expect_text_any: ["hi"]}
    reference: {answer: "hi"}
""")
    suite = load_suite(suite_root)
    sandbox = fx.Sandbox(tmp_path / "sb")
    prev = sandbox.activate()
    try:
        sandbox.reset()
        model_cfg = {"provider": "llama_cpp", "model": "qwen2.5-7b-instruct",
                     "grammar_enforcement": False}
        pipeline = {"provider": _Looping(), "adapter_factory": lambda: create_adapter(model_cfg),
                    "security_gate": create_security_gate({"workspace_root": str(sandbox.workspace)}),
                    "model_name": "qwen2.5-7b-instruct", "model_cfg": model_cfg}
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        row = asyncio.run(lr.run_task(
            suite.tasks[0], suite, pipeline, sandbox=sandbox, tel=tel, judge=None,
            run_label="cap", run_idx=0, static={"model": "m", "run_label": "cap"}))
    finally:
        fx.Sandbox.restore(prev)
    assert (row["verdict"], row["success"], row["stopped_by"]) == ("fail", False, "round_cap"), row
    assert row["rounds"] == 3


# ---------------------------------------------------------------------------
# The loop's own halt messages are not the model's answer
# ---------------------------------------------------------------------------


class TestLoopHalts:

    def test_every_halt_template_still_opens_the_way_the_ladder_expects(self):
        """Drift guard: the ladder recognises the loop's halt messages by
        their opening words. If the engine rewords one, this fails instead of
        the ladder silently scoring the new wording as a model answer."""
        src = (Path(lr.__file__).resolve().parents[2] / "engine" / "agent_loop.py").read_text()
        for prefix, _kind in lr.LOOP_HALTS:
            assert prefix.rstrip("( ").rstrip() in src, prefix
        assert lr.REPEAT_BLOCKED_PREFIX.strip() in src

    def test_generated_halt_texts_are_recognised(self):
        from types import SimpleNamespace

        from prometheus.engine import agent_loop as al

        trip = SimpleNamespace(tool_name="read_file", tool_input={"path": "x"}, count=3,
                               empty=False, varied=False)
        texts = {
            "repeat_halt": al._repeat_trip_text(trip),
            "divergence_halt": al._divergence_halt_text(4),
            "boundary_escape": al._boundary_escape_text(["/etc/x"]),
            "tool_call_cap": "Tool iteration limit reached (5/4). Stopping to prevent runaway loops.",
        }
        for kind, text in texts.items():
            msg = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
            assert lr.loop_halt_kind(msg) == kind, text[:60]
        model_says = ConversationMessage(role="assistant", content=[TextBlock(text="The answer is 5.")])
        assert lr.loop_halt_kind(model_says) is None


def _pipeline(provider, sandbox):
    from prometheus.__main__ import create_adapter, create_security_gate

    model_cfg = {"provider": "llama_cpp", "model": "qwen2.5-7b-instruct",
                 "grammar_enforcement": False, "base_url": "http://127.0.0.1:9"}
    return {"provider": provider, "adapter_factory": lambda: create_adapter(model_cfg),
            "security_gate": create_security_gate({"workspace_root": str(sandbox.workspace)}),
            "model_name": "qwen2.5-7b-instruct", "model_cfg": model_cfg}


def _one_task(tmp_path, body: str, budgets: str = ""):
    root = tmp_path / "suite"
    root.mkdir(exist_ok=True)
    meta = SUITE_META
    if budgets:
        meta = meta.replace("max_rounds: 3, max_tool_calls: 3", budgets)
    (root / "suite.yaml").write_text(meta)
    (root / "qa.yaml").write_text("class: qa\ntasks:\n" + body)
    s = load_suite(root)
    return s, s.tasks[0]


def _run(tmp_path, provider, body, budgets="", **kw):
    suite, task = _one_task(tmp_path, body, budgets)
    sandbox = fx.Sandbox(tmp_path / "sb")
    prev = sandbox.activate()
    try:
        sandbox.reset()
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        row = asyncio.run(lr.run_task(
            task, suite, _pipeline(provider, sandbox), sandbox=sandbox, tel=tel, judge=None,
            run_label="t", run_idx=0, static={"model": "m", "run_label": "t"}, **kw))
    finally:
        fx.Sandbox.restore(prev)
    return row


class _ManyCalls(ModelProvider):
    """Writes the right file, then keeps calling tools — several per round."""

    def __init__(self) -> None:
        self.n = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.n += 1
        ws = os.environ["PROMETHEUS_WORKSPACE_DIR"].rsplit("/home/", 1)[0] + "/ws"
        blocks = [ToolUseBlock(id=f"toolu_{self.n}a", name="write_file",
                               input={"path": f"{ws}/out.txt", "content": "3\n"}),
                  ToolUseBlock(id=f"toolu_{self.n}b", name="glob", input={"pattern": f"*{self.n}"})]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=blocks),
            usage=UsageSnapshot(input_tokens=60, output_tokens=5), stop_reason="tool_use")


class _RepeatFailing(ModelProvider):
    """Calls the same failing read forever."""

    def __init__(self) -> None:
        self.n = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.n += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[ToolUseBlock(
                id=f"toolu_{self.n}", name="read_file", input={"path": "/nonexistent/x.txt"})]),
            usage=UsageSnapshot(input_tokens=60, output_tokens=5), stop_reason="tool_use")


class _Hang(ModelProvider):
    async def stream_message(self, request):  # noqa: ANN001
        await asyncio.sleep(60)
        yield  # pragma: no cover


class _Crash(ModelProvider):
    async def stream_message(self, request):  # noqa: ANN001
        raise ConnectionError("connect to http://10.1.2.3:8080/v1/chat/completions refused")
        yield  # pragma: no cover


class _ThinksAloud(ModelProvider):
    """Answers with its unfinished reasoning, as the llama.cpp provider does
    when the whole budget went to thinking — and files the same silent-failure
    row the provider files."""

    async def stream_message(self, request):  # noqa: ANN001
        from prometheus.telemetry.tracker import get_telemetry_handle

        get_telemetry_handle().record_silent_failure(
            "llama_cpp_provider", "stream_message", RuntimeError("empty content"),
            context={"used_reasoning_fallback": True})
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(
                text="Hmm, 7 times 3 is 21, or was it... let me check 21 again")]),
            usage=UsageSnapshot(input_tokens=50, output_tokens=1024), stop_reason="length")


CAP_TASK = """  - id: cap
    prompt: "Write 3 to {workspace}/out.txt, then say 3."
    score:
      expect_file_regex: {path: out.txt, pattern: '^3$'}
      expect_text_regex: '\\b3\\b'
    reference: {files: {out.txt: "3\\n"}, answer: "3"}
"""


class _LongAnswer(ModelProvider):
    """One round: a long working-out, the committed answer on the last line."""

    async def stream_message(self, request):  # noqa: ANN001
        text = "Working it out step by step. " * 20 + "\nANSWER: 1081"
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
            usage=UsageSnapshot(input_tokens=40, output_tokens=150), stop_reason="stop")


ANSWER_TASK = """  - id: mult
    prompt: "What is 47 times 23? End your reply with a final line `ANSWER: <the number>`."
    score:
      expect_answer: '1[,\\s]?081'
      answer_shape: '-?\\d+(?:[,\\s]\\d{3})*'
    reference: {answer: "ANSWER: 1081"}
"""


def test_every_verdict_has_a_progress_line():
    # A format miss crashed the first real run: the progress printer knew only
    # four verdicts. Every verdict the harness can return must print.
    from types import SimpleNamespace

    from prometheus.gym.ladder import verdict as v

    task = SimpleNamespace(task_class="qa", id="t")
    row = {"rounds": 1, "tool_calls_ok": 0, "tool_calls": 0, "repairs": 0,
           "duration_ms": 1200.0, "fail_reasons": ["why"]}
    for verdict in (v.PASS, v.FAIL, v.FORMAT_MISS, v.UNSCORED, v.ERROR):
        line = lr.progress_line(task, {**row, "verdict": verdict})
        assert "qa" in line and "t" in line
        assert ("why" in line) == (verdict != v.PASS)


class TestRunOutcomes:

    def test_the_whole_final_reply_is_recorded_for_audit(self, tmp_path):
        # The reader reads the END of the reply; a hand-check of its verdicts
        # needs what it read, not the first 300 characters.
        row = _run(tmp_path, _LongAnswer(), ANSWER_TASK)
        assert row["verdict"] == "pass", row
        assert len(row["final_text"]) > 300 and row["final_text"].endswith("ANSWER: 1081")
        conn = sqlite3.connect(tmp_path / "telemetry.db")
        try:
            (stored,) = rec.load_rows(conn, "t")
        finally:
            conn.close()
        assert stored["final_text"] == row["final_text"]

    def test_the_tool_call_cap_is_a_fail_and_its_message_is_not_the_answer(self, tmp_path):
        # The cap message "Tool iteration limit reached (4/3)" contains a 3;
        # the file is right. Neither may turn a run that never stopped into a pass.
        row = _run(tmp_path, _ManyCalls(), CAP_TASK,
                   budgets="max_rounds: 10, max_tool_calls: 3")
        assert (row["verdict"], row["stopped_by"]) == ("fail", "tool_call_cap"), row
        assert row["fail_reasons"] == ["tool-call budget exhausted (3 calls)"]
        assert "Tool iteration limit" not in row["final_text_head"]

    def test_a_repeat_halt_is_a_fail_and_blocked_calls_count(self, tmp_path):
        row = _run(tmp_path, _RepeatFailing(), CAP_TASK,
                   budgets="max_rounds: 12, max_tool_calls: 12")
        assert row["verdict"] == "fail", row
        assert row["stopped_by"] in ("repeat_halt", "circuit_breaker"), row["stopped_by"]
        assert row["tool_calls_blocked"] >= 1
        assert row["tool_calls"] >= row["tool_calls_blocked"] and row["tool_calls_ok"] == 0

    def test_a_slow_model_on_a_live_endpoint_is_a_fail(self, tmp_path, monkeypatch):
        async def alive(pipeline, timeout_s=45.0):
            return True, "HTTP 200"

        monkeypatch.setattr(lr, "endpoint_alive", alive)
        row = _run(tmp_path, _Hang(), CAP_TASK, budgets="max_rounds: 3, max_tool_calls: 3")
        # timeout_s from SUITE_META is 30 — patch it down for the test
        assert row["verdict"] == "fail" and row["stopped_by"] == "timeout"

    def test_a_dead_endpoint_is_an_error_and_stops_the_ladder(self, tmp_path, monkeypatch):
        async def dead(pipeline, timeout_s=45.0):
            return False, "ReadTimeout"

        monkeypatch.setattr(lr, "endpoint_alive", dead)
        with pytest.raises(lr.LadderAbort, match="unresponsive"):
            _run(tmp_path, _Hang(), CAP_TASK)
        conn = sqlite3.connect(str(tmp_path / "telemetry.db"))
        (row,) = rec.load_rows(conn, "t")
        assert (row["verdict"], row["success"]) == ("error", None)

    def test_a_provider_crash_is_an_error_and_names_no_host(self, tmp_path):
        row = _run(tmp_path, _Crash(), CAP_TASK)
        assert (row["verdict"], row["success"], row["stopped_by"]) == ("error", None, "error")
        assert "10.1.2.3" not in json.dumps(row) and "<url>" in row["error"]

    def test_a_reasoning_fallback_answer_is_not_an_answer(self, tmp_path):
        body = """  - id: think
    prompt: "What is 7 times 3?"
    score: {expect_text_regex: '\\b21\\b'}
    reference: {answer: "21"}
"""
        row = _run(tmp_path, _ThinksAloud(), body)
        assert (row["verdict"], row["stopped_by"]) == ("fail", "reasoning_fallback"), row


@pytest.fixture(autouse=True)
def _short_timeouts(monkeypatch):
    """Keep hang tests fast: cap every ladder task's wall clock at 2 s."""
    orig = lr.run_task

    async def run_task(task, *a, **kw):
        if task.timeout_s > 2:
            task.timeout_s = 2.0
        return await orig(task, *a, **kw)

    monkeypatch.setattr(lr, "run_task", run_task)


# ---------------------------------------------------------------------------
# Recording and reporting — the mutants the review found surviving
# ---------------------------------------------------------------------------


class TestRecordingDetails:

    def test_sessionless_rows_outside_the_window_are_not_attributed(self, tmp_path):
        tel = _tel(tmp_path)
        tel.record("m", "write_file", False, error_type="permission_denied")
        t0 = time.time() + 5
        m = rec.harvest_run_metrics(tel._conn, "s", window=(t0, t0 + 10))
        assert (m["tool_calls"], m["tool_calls_unattributed"]) == (0, 0)
        m = rec.harvest_run_metrics(tel._conn, "s", window=None)
        assert m["tool_calls_unattributed"] == 0

    def test_two_labels_in_one_db_stay_apart(self, tmp_path):
        tel = _tel(tmp_path)
        for label in ("A", "B", "B"):
            rec.record_summary(tel, {"session_id": f"{label}{time.time()}", "task_class": "qa",
                                     "verdict": "pass", "duration_ms": 1.0, "model": "m",
                                     "run_label": label})
        assert (len(rec.load_rows(tel._conn, "A")), len(rec.load_rows(tel._conn, "B"))) == (1, 2)

    def test_wilson_bounds(self):
        lo, hi = rec.wilson(8, 24)
        assert (round(lo, 3), round(hi, 3)) == (0.180, 0.533)
        assert rec.wilson(0, 0) == (0.0, 0.0)

    def test_report_cells_use_decided_runs_and_telemetry_denominators(self):
        base = {f: 1 for f in rec.REQUIRED_FIELDS}
        base.update(task_class="qa", verdict_source="predicates", duration_ms=1000.0, suite="s",
                    suite_sha="0" * 64, run_label="L", provider="p", model="m", served_models=[],
                    difficulty="easy")
        rows = [
            {**base, "task_id": "a", "verdict": "pass", "tool_calls": 4, "tool_calls_ok": 2,
             "tool_calls_excluded": 1},
            {**base, "task_id": "b", "verdict": "unscored", "tool_calls": 0, "tool_calls_ok": 0,
             "tool_calls_excluded": 0},
            {**base, "task_id": "c", "verdict": "error", "tool_calls": 0, "tool_calls_ok": 0,
             "tool_calls_excluded": 0,
             "fail_reasons": ["ConnectError for url 'http://10.0.0.7:8080/v1/chat'"]},
        ]
        text = rec.render_report(rows, title="x", class_order=["qa"])
        qa_line = next(line for line in text.splitlines() if line.startswith("| qa | 3 |"))
        assert "| 1/1 (" in qa_line          # 1 pass of 1 decided, not of 3 runs
        assert "| 2/3 |" in qa_line          # ok / (calls - excluded)
        assert "10.0.0.7" not in text and "<url>" in text

    def test_format_misses_sit_next_to_accuracy_never_inside_it(self):
        base = {"task_class": "qa", "duration_ms": 1.0, "model": "m", "quantization": "Q4",
                "adapter_tier": "light"}
        small = [
            {**base, "verdict": "pass", "answer_format_ok": True},
            {**base, "verdict": "pass", "answer_format_ok": False},   # credited, no line
            {**base, "verdict": "format_miss", "answer_format_ok": False},
            {**base, "verdict": "fail", "answer_format_ok": True},
        ]
        assert rec.accuracy(small) == (2, 3)          # the format miss is not a wrong answer
        assert rec.format_misses(small) == (2, 4)
        table = rec.render_ladder_table({"small": small}, class_order=["qa"])
        row = next(line for line in table.splitlines() if "`small`" in line)
        assert "| 2/3 (" in row and "| 2/4 |" in row and "2/3 / 2/4" in row
        assert rec.outcome_for("format_miss") == "partial"

    def test_rows_keep_the_machines_node_identity(self, tmp_path, monkeypatch):
        real = tmp_path / "realhome"
        (real / "node").mkdir(parents=True)
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(real))
        monkeypatch.delenv("PROMETHEUS_NODE_DIR", raising=False)
        sb = fx.Sandbox(tmp_path / "sb")
        prev = sb.activate()
        try:
            assert os.environ["PROMETHEUS_NODE_DIR"] == str(real / "node")
            assert os.environ["PROMETHEUS_CONFIG_DIR"] == str(sb.home)
        finally:
            fx.Sandbox.restore(prev)
        assert "PROMETHEUS_NODE_DIR" not in os.environ

    def test_no_identity_is_never_created_in_the_real_home(self, tmp_path, monkeypatch):
        real = tmp_path / "realhome"
        real.mkdir()
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(real))
        monkeypatch.delenv("PROMETHEUS_NODE_DIR", raising=False)
        sb = fx.Sandbox(tmp_path / "sb")
        prev = sb.activate()
        fx.Sandbox.restore(prev)
        assert not (real / "node").exists()


class TestSandboxHygiene:

    def test_reset_clears_what_a_run_locked_down(self, sandbox):
        d = sandbox.workspace / "pkg" / "sub"
        d.mkdir(parents=True)
        (d / "f").write_text("x")
        (sandbox.workspace / "out").symlink_to("/etc")
        os.chmod(d, 0o500)
        os.chmod(d.parent, 0o500)
        sandbox.reset()
        assert list(sandbox.workspace.iterdir()) == []

    def test_one_ladder_run_per_sandbox(self, tmp_path):
        a, b = fx.Sandbox(tmp_path / "sb"), fx.Sandbox(tmp_path / "sb")
        a.lock()
        try:
            with pytest.raises(fx.SandboxError, match="another ladder run"):
                b.lock()
        finally:
            a.unlock()
        b.lock()
        b.unlock()


# ---------------------------------------------------------------------------
# Preflight — the rung is what the endpoint serves, and the CLI refuses
# ---------------------------------------------------------------------------


class TestRungPreflight:

    def _patch(self, monkeypatch, identity):
        monkeypatch.setattr(lr, "preflight_endpoint", lambda config: None)

        async def probe(provider, base_url, model):
            return identity

        monkeypatch.setattr(lr, "probe_identity", probe)

    def _run(self, tmp_path, **kw):
        suite, task = _one_task(tmp_path, '  - id: a\n    prompt: q\n    judge: {rubric: r, reference: x}\n')
        return asyncio.run(lr.run_ladder(
            suite, [task], lr.Contestant(provider="llama_cpp", base_url="http://x", **kw.pop("c", {})),
            run_label="r", judge_pin=None, telemetry_db=tmp_path / "t.db",
            workdir=tmp_path / "sb", progress=False, **kw))

    def test_the_served_model_decides_not_the_operators_flag(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, {"served_model": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
                                  "quantization": "Q4_K_M", "quantization_source": "gguf-filename",
                                  "parameter_size": None})
        with pytest.raises(lr.LadderPreflightError, match="wrong rung"):
            self._run(tmp_path, c={"model": "qwen3-14b"}, rung="r14b",
                      expect_model_match=r"qwen3[-_:]14b", strict_quant=True)

    def test_a_rung_needs_a_probed_quantization(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, {"served_model": "Qwen3-14B.gguf", "quantization": None,
                                  "quantization_source": "unreported", "parameter_size": None})
        with pytest.raises(lr.LadderPreflightError, match="not evidence"):
            self._run(tmp_path, c={"quantization": "Q4_K_M"}, rung="r14b",
                      expect_model_match=r"qwen3[-_:]14b", strict_quant=True)

    def test_a_second_run_on_the_same_db_is_refused(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, {"served_model": "m.gguf", "quantization": "Q8_0",
                                  "quantization_source": "gguf-filename", "parameter_size": None})
        held = lr._lock_db((tmp_path / "t.db").resolve())
        try:
            with pytest.raises(lr.LadderPreflightError, match="another ladder run"):
                self._run(tmp_path)
        finally:
            held.close()


def _cli(*argv: str, cwd=None):
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    env.pop("LADDER_BASE_URL", None)
    env.pop("LADDER_JUDGE_BASE_URL", None)
    return subprocess.run([sys.executable, str(repo / "scripts" / "ladder_run.py"), *argv],
                          capture_output=True, text=True, env=env, cwd=cwd, timeout=120)


class TestCLIRefusals:

    def test_unknown_or_deferred_classes_and_ids_are_refused(self, tmp_path):
        common = ["--base-url", "http://127.0.0.1:9", "--no-judge",
                  "--telemetry-db", str(tmp_path / "t.db")]
        for extra, msg in ((["--classes", "qa,nope"], "unknown class"),
                           (["--classes", "long_haul"], "deferred"),
                           (["--tasks", "qa-arith-mult,no-such-task"], "unknown task")):
            r = _cli(*common, *extra)
            assert r.returncode == 2 and msg in r.stdout, (extra, r.stdout, r.stderr)

    def test_quant_cannot_override_the_rung(self, tmp_path):
        r = _cli("--rung", "r27b", "--quant", "Q8_0", "--telemetry-db", str(tmp_path / "t.db"))
        assert r.returncode == 2 and "pinned to UD-Q4_K_XL" in r.stdout, r.stdout

    def test_report_only_never_overwrites_with_nothing(self, tmp_path):
        out = tmp_path / "report.md"
        out.write_text("the real report")
        r = _cli("--report-only", "--run-label", "nope", "--telemetry-db", str(tmp_path / "t.db"),
                 "--report", str(out))
        assert r.returncode == 1 and out.read_text() == "the real report", r.stdout


# ---------------------------------------------------------------------------
# Tier sweep — the daemon's adapter at a forced tier, counted, never filed
# under a rung
# ---------------------------------------------------------------------------

QWEN_27B = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
BONSAI = "Ternary-Bonsai-2-27B-PQ2_0.gguf"


def _adapter_fp(a):
    return (a.tier, type(a.formatter).__name__, a._base_strictness.value, a.retry.max_retries)


@pytest.mark.parametrize("model", [QWEN_27B, BONSAI])
def test_a_forced_tier_is_the_daemons_own_adapter_for_that_tier(model):
    import prometheus.__main__ as daemon
    from prometheus.gym.ladder.tiers import forced_adapter_factory

    cfg = {"provider": "llama_cpp", "model": model}
    real = daemon._get_adapter_tier
    daemon_pick = daemon.create_adapter(cfg, {})
    # Forcing the daemon's own pick changes nothing.
    assert _adapter_fp(forced_adapter_factory(daemon_pick.tier, cfg, {})()) == _adapter_fp(daemon_pick)
    # 'off' for a local model is exactly what the daemon builds for a cloud provider.
    cloud = daemon.create_adapter({"provider": "openai", "model": model}, {})
    assert _adapter_fp(forced_adapter_factory("off", cfg, {})()) == _adapter_fp(cloud)
    assert {t: _adapter_fp(forced_adapter_factory(t, cfg, {})()) for t in ("light", "full")} == {
        "light": ("light", "QwenFormatter", "NONE", 1),
        "full": ("full", "QwenFormatter", "MEDIUM", 3),
    }
    assert daemon._get_adapter_tier is real, "the daemon's tier decision was not restored"
    with pytest.raises(ValueError):
        forced_adapter_factory("medium", cfg, {})


def test_the_counters_observe_and_never_change_what_the_adapter_returns(tmp_path):
    import copy

    from prometheus.adapter.retry import RetryAction
    from prometheus.gym.ladder.tiers import counts_for_row, forced_adapter_factory, instrument_adapter

    registry = fx.build_ladder_registry(tmp_path)
    cfg = {"provider": "llama_cpp", "model": BONSAI}
    text = 'Reading it.\n{"name": "read_file", "arguments": {"path": "a.txt"}}'

    def calls(blocks):
        return [(b.name, b.input) for b in blocks]

    plain, counted = (forced_adapter_factory("light", cfg, {})() for _ in range(2))
    counts = instrument_adapter(counted)
    assert calls(counted.extract_tool_calls(text, registry)) == calls(plain.extract_tool_calls(text, registry)) != []
    assert counts["calls_from_text"] == 1
    counted.extract_tool_calls("<tool_call>\n<function=read_file>\n</function>\n</tool_call>", registry)
    assert counts["xml_markup_turns"] == 1
    # light allows one retry, then aborts — each decision counted, unchanged
    assert counted.handle_retry("read_file", "bad", registry)[0] == RetryAction.RETRY
    assert counted.handle_retry("read_file", "bad", registry)[0] == RetryAction.ABORT
    assert (counts["adapter_retries"], counts["adapter_aborts"]) == (1, 1)
    # A breaker tier bump works on a copy: same counter dict, new tier seen.
    bumped = copy.copy(counted)
    bumped.tier = "full"
    bumped.extract_tool_calls("no call here", registry)
    assert counts_for_row(counts)["tiers_seen"] == ["full", "light"]

    off = forced_adapter_factory("off", cfg, {})()
    off_counts = instrument_adapter(off)
    assert off.extract_tool_calls(text, registry) == []  # tier off never looks
    assert off_counts["text_calls_missed"] == 1           # ...but light/full would have recovered it


class _TextToolCall(ModelProvider):
    """Round 1: a tool call written as TEXT. Round 2: the answer. Records grammars."""

    def __init__(self) -> None:
        self.calls = 0
        self.grammars: list = ["stale grammar from an earlier run"]

    def set_grammar(self, grammar):  # noqa: ANN001
        self.grammars.append(grammar)

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        ws = os.environ["PROMETHEUS_WORKSPACE_DIR"].rsplit("/home/", 1)[0] + "/ws"
        text = (f'{{"name": "read_file", "arguments": {{"path": "{ws}/n.txt"}}}}'
                if self.calls == 1 else "The file says 41.\nANSWER: 41")
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
            usage=UsageSnapshot(input_tokens=30, output_tokens=12), stop_reason="stop")


TEXT_CALL_TASK = """  - id: tc
    prompt: "Read {workspace}/n.txt. End with `ANSWER: <n>`."
    setup_files: {n.txt: "41\\n"}
    score: {expect_answer: '41', answer_shape: '\\d+'}
    reference: {answer: "ANSWER: 41"}
"""


@pytest.mark.parametrize("tier, verdict, from_text, missed, grammar_is_none", [
    ("off", "format_miss", 0, 1, True),  # the text call is left as the reply: no tool ran
    ("light", "pass", 1, 0, False),  # recovered from the text and run
])
def test_a_forced_tier_run_counts_what_the_adapter_did(tmp_path, tier, verdict, from_text, missed,
                                                        grammar_is_none):
    from prometheus.__main__ import create_security_gate
    from prometheus.gym.ladder.tiers import forced_adapter_factory

    suite, task = _one_task(tmp_path, TEXT_CALL_TASK, "max_rounds: 4, max_tool_calls: 3")
    sandbox = fx.Sandbox(tmp_path / "sb")
    prev = sandbox.activate()
    provider = _TextToolCall()
    try:
        sandbox.reset()
        model_cfg = {"provider": "llama_cpp", "model": BONSAI, "grammar_enforcement": True}
        pipeline = {"provider": provider,
                    "adapter_factory": forced_adapter_factory(tier, model_cfg, {}),
                    "security_gate": create_security_gate({"workspace_root": str(sandbox.workspace)}),
                    "model_name": BONSAI, "model_cfg": model_cfg, "tier_forced": True}
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        row = asyncio.run(lr.run_task(task, suite, pipeline, sandbox=sandbox, tel=tel, judge=None,
                                      run_label="t", run_idx=0, static={"model": BONSAI, "run_label": "t"}))
    finally:
        fx.Sandbox.restore(prev)
    assert row["verdict"] == verdict, row["fail_reasons"]
    assert (row["adapter_tier_start"], row["adapter_tier_end"]) == (tier, tier)
    counts = row["adapter_counts"]
    assert (counts["calls_from_text"], counts["text_calls_missed"]) == (from_text, missed)
    assert row["provider_http_retries"] == 0
    # The grammar is always set for the run — None at off clears a stale one.
    assert (provider.grammars[-1] is None) is grammar_is_none


def _sweep_ladder(tmp_path, monkeypatch, *, force, expect_tier="full", rung="r27b-pq2"):
    from prometheus.__main__ import create_adapter, create_security_gate

    suite, task = _one_task(tmp_path, TEXT_CALL_TASK, "max_rounds: 4, max_tool_calls: 3")
    monkeypatch.setattr(lr, "preflight_endpoint", lambda config: None)

    async def identity(provider, base_url, model):
        return {"served_model": BONSAI, "quantization": "PQ2_0",
                "quantization_source": "gguf-filename", "parameter_size": None}

    async def kv(provider):
        return {"k": None, "v": None, "source": "unreported"}

    async def thinking(provider):
        return {"status": "supported", "detail": "test"}

    def build(config):
        model_cfg = dict(config["model"], grammar_enforcement=True)
        return {"provider": _TextToolCall(), "adapter_factory": lambda: create_adapter(model_cfg, {}),
                "security_gate": create_security_gate({"workspace_root": config["security"]["workspace_root"]}
                                                      if "security" in config else {}),
                "model_name": model_cfg["model"], "model_cfg": model_cfg}

    monkeypatch.setattr(lr, "probe_identity", identity)
    monkeypatch.setattr(lr, "_probe_kv_cache", kv)
    monkeypatch.setattr(lr, "_probe_thinking", thinking)
    monkeypatch.setattr(lr, "build_pipeline", build)
    return asyncio.run(lr.run_ladder(
        suite, [task], lr.Contestant(provider="llama_cpp", base_url="http://x"),
        run_label="sweep-light", judge_pin=None, telemetry_db=tmp_path / "t.db",
        workdir=tmp_path / "sb", rung=rung, expect_model_match="ternary-bonsai-2-27b",
        expect_adapter_tier=expect_tier, strict_quant=True, force_adapter_tier=force, progress=False))


class TestTierSweep:

    def test_sweep_rows_are_filed_under_no_rung(self, tmp_path, monkeypatch):
        (row,) = _sweep_ladder(tmp_path, monkeypatch, force="light")
        assert row["rung"] is None
        assert row["tier_sweep"] == {"of": "r27b-pq2", "forced_tier": "light", "daemon_tier": "full"}
        assert (row["adapter_tier"], row["adapter_tier_forced"], row["adapter_tier_start"]) == (
            "light", True, "light")

    def test_a_sweep_still_runs_the_rungs_checks(self, tmp_path, monkeypatch):
        with pytest.raises(lr.LadderPreflightError, match="sweep OF a rung"):
            _sweep_ladder(tmp_path, monkeypatch, force="light", rung=None)
        # The rung says light, the daemon picks full: refused even though a tier is forced.
        with pytest.raises(lr.LadderPreflightError, match="expects adapter tier"):
            _sweep_ladder(tmp_path, monkeypatch, force="full", expect_tier="light")

    def test_sweep_labels_never_enter_the_rung_table_and_get_their_own_report(self, tmp_path, monkeypatch):
        _sweep_ladder(tmp_path, monkeypatch, force="light")
        db = str(tmp_path / "t.db")
        r = _cli("--compare", "sweep-light", "--telemetry-db", db, "--report", str(tmp_path / "c.md"))
        assert r.returncode == 1 and "tier-sweep runs" in r.stdout, r.stdout
        out = tmp_path / "sweep.md"
        r = _cli("--tier-report", "sweep-light", "--telemetry-db", db, "--report", str(out))
        assert r.returncode == 0, r.stdout + r.stderr
        assert "# Tier sweep — `r27b-pq2`" in out.read_text()
        r = _cli("--force-adapter-tier", "light", "--base-url", "http://127.0.0.1:9", "--no-judge",
                 "--telemetry-db", db)
        assert r.returncode == 2 and "needs --rung" in r.stdout


def _sweep_row(task, tier, verdict, *, bumped=False, retries=0):
    return {"run_label": f"s-{tier}", "task_id": task, "task_class": "single_tool", "verdict": verdict,
            "tier_sweep": {"of": "r27b-pq2", "forced_tier": tier, "daemon_tier": "full"},
            "adapter_tier_start": tier, "adapter_tier_end": "full" if bumped else tier,
            "adapter_counts": {"tiers_seen": [tier, "full"] if bumped else [tier], "adapter_retries": retries,
                               "adapter_aborts": 0, "calls_from_text": 0, "text_calls_missed": 0,
                               "xml_markup_turns": 0},
            "model": BONSAI, "provider": "llama_cpp", "quantization": "PQ2_0", "suite": "ladder-v1",
            "suite_sha": "0" * 64, "tool_calls": 1, "tool_calls_ok": 1, "repairs": 0, "rounds": 2,
            "duration_ms": 1000.0, "stopped_by": "done"}


def test_the_tier_report_leaves_bumped_runs_out_and_pairs_tasks():
    rows = [_sweep_row("a", "off", "fail"), _sweep_row("b", "off", "fail"),
            _sweep_row("a", "light", "pass", retries=1), _sweep_row("b", "light", "pass"),
            _sweep_row("c", "light", "fail", bumped=True)]
    report = rec.render_tier_sweep(rows, class_order=["single_tool"])
    lines = report.splitlines()
    light = next(ln for ln in lines if ln.startswith("| light |"))
    assert light.startswith("| light | 2 | 2/2 ")  # the bumped run is not in the main figures
    assert "| light − off | 2 | +1.000 |" in report
    left_out = next(ln for ln in lines if ln.startswith("| light | 1 |"))  # counted apart
    assert "done 2" in left_out
    assert rec.paired_difference({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 1.0})[:2] == (2, 0.5)


# ---------------------------------------------------------------------------
# What the pre-PR review found: a bump the row missed, rows with no verdict,
# another writer's rows, a judge's URL, a model path, a symlinked sandbox, and
# an abort's exit code
# ---------------------------------------------------------------------------


class _TripThenEmpty(ModelProvider):
    """Round 1: five failing reads — the circuit breaker bumps the tier. Then
    only empty replies, so the adapter is never asked anything at the new tier."""

    def __init__(self) -> None:
        self.calls = 0

    def set_grammar(self, grammar):  # noqa: ANN001
        pass

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        ws = os.environ["PROMETHEUS_WORKSPACE_DIR"].rsplit("/home/", 1)[0] + "/ws"
        content = ([ToolUseBlock(id=f"t{i}", name="read_file", input={"path": f"{ws}/missing{i}.txt"})
                    for i in range(5)] if self.calls == 1 else [TextBlock(text="")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=30, output_tokens=12), stop_reason="stop")


def test_a_breaker_bump_is_seen_even_when_the_adapter_is_never_asked_again(tmp_path):
    # The breaker bumps a COPY of the adapter inside the loop's own copy of the
    # context. The row read the caller's context (light → light), and with no
    # adapter call after the bump the run was filed in tier light's figures.
    from prometheus.__main__ import create_security_gate
    from prometheus.gym.ladder.tiers import forced_adapter_factory

    suite, task = _one_task(tmp_path, TEXT_CALL_TASK, "max_rounds: 6, max_tool_calls: 8")
    sandbox = fx.Sandbox(tmp_path / "sb")
    prev = sandbox.activate()
    try:
        sandbox.reset()
        model_cfg = {"provider": "llama_cpp", "model": BONSAI, "grammar_enforcement": True}
        pipeline = {"provider": _TripThenEmpty(),
                    "adapter_factory": forced_adapter_factory("light", model_cfg, {}),
                    "security_gate": create_security_gate({"workspace_root": str(sandbox.workspace)}),
                    "model_name": BONSAI, "model_cfg": model_cfg, "tier_forced": True}
        tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
        row = asyncio.run(lr.run_task(task, suite, pipeline, sandbox=sandbox, tel=tel, judge=None,
                                      run_label="t", run_idx=0, static={"model": BONSAI, "run_label": "t"}))
    finally:
        fx.Sandbox.restore(prev)
    assert (row["adapter_tier_start"], row["adapter_tier_end"]) == ("light", "full"), row["stopped_by"]
    assert row["adapter_counts"]["tiers_seen"] == ["full", "light"]
    assert rec.tier_bumped(row)


def _summary(tel, label, sid, attribution="time-window"):
    rec.record_summary(tel, {"session_id": sid, "task_class": "single_tool", "verdict": "fail",
                             "duration_ms": 1.0, "model": "m", "run_label": label,
                             "sessionless_attribution": attribution})
    time.sleep(0.01)


def _bump(tel, method="tier_bump:off->light"):
    tel.record_diagnosis("m", "off", "read_file", "wrong_path", False, None, True, method)
    time.sleep(0.01)


def _ladder_run(tel, label, sid, bump=None, *, summary=True, attribution="time-window"):
    """One run as it lands in the database: its first model round, a breaker
    row if the breaker tripped, then (unless interrupted) its summary."""
    tel.record_run("agent_loop", "loop_round", "success", session_id=sid)
    time.sleep(0.01)
    if bump:
        _bump(tel, bump)
    if summary:
        _summary(tel, label, sid, attribution)


def test_the_breakers_own_record_marks_a_bumped_run_in_a_ladder_only_db(tmp_path):
    # A second source for rows recorded before the tier was observed where it
    # is assigned: the breaker's diagnostics row, placed by time — only where
    # the ladder is the database's one writer, and only inside a run's own
    # model rounds (the breaker trips after a round).
    tel = _tel(tmp_path)
    _bump(tel)                                                    # before any run: nobody's
    _ladder_run(tel, "L", "s1")
    _ladder_run(tel, "L", "s2", "tier_bump:off->light")
    _ladder_run(tel, "L", "s3", "already_attempted")
    _ladder_run(tel, "L", "gone", "tier_bump:off->light", summary=False)  # interrupted
    _ladder_run(tel, "L", "s4")
    _ladder_run(tel, "L", "s5", "tier_bump_failed")
    rows = rec.load_rows(tel._conn, "L")
    assert [r["breaker_tier_bumps"] for r in rows] == [[], ["tier_bump:off->light"], [], [], []]
    assert [rec.tier_bumped(r) for r in rows] == [False, True, False, False, False]
    # The live telemetry.db has other writers: nothing is placed by time there.
    live = "off (live telemetry.db)"
    _ladder_run(tel, "live", "s6", attribution=live)
    _ladder_run(tel, "live", "s7", "tier_bump:off->light", attribution=live)
    assert [r["breaker_tier_bumps"] for r in rec.load_rows(tel._conn, "live")] == [None, None]


def test_runs_with_no_verdict_are_left_out_of_a_tiers_figures():
    # An endpoint error is not the adapter's: it must not lower the tier's task
    # success or flip a paired task, and it is counted apart.
    rows = [_sweep_row("a", "off", "pass"), _sweep_row("b", "off", "pass"),
            _sweep_row("a", "light", "pass"), {**_sweep_row("b", "light", "error"), "stopped_by": "error"}]
    report = rec.render_tier_sweep(rows, class_order=["single_tool"])
    lines = report.splitlines()
    assert next(ln for ln in lines if ln.startswith("| light |")).startswith("| light | 1 | 1/1 ")
    assert "| light − off | 1 | +0.000 |" in report
    assert "| light | 0 | 1 | done 1 |" in report


class _AnswersWhileAnotherWriterFallsBack(ModelProvider):
    """Answers right; meanwhile ANOTHER writer to the same database (the live
    daemon, on the live telemetry.db) files a reasoning-fallback row."""

    def __init__(self, db_path) -> None:
        self.other = ToolCallTelemetry(db_path=db_path)

    async def stream_message(self, request):  # noqa: ANN001
        self.other.record_silent_failure(
            "llama_cpp_provider", "stream_message", RuntimeError("empty content"),
            context={"used_reasoning_fallback": True})
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="7 times 3 is 21.")]),
            usage=UsageSnapshot(input_tokens=50, output_tokens=8), stop_reason="stop")


def test_another_writers_reasoning_fallback_is_not_this_runs(tmp_path):
    body = """  - id: think
    prompt: "What is 7 times 3?"
    score: {expect_text_regex: '\\b21\\b'}
    reference: {answer: "21"}
"""
    row = _run(tmp_path, _AnswersWhileAnotherWriterFallsBack(tmp_path / "telemetry.db"), body)
    assert (row["verdict"], row["stopped_by"]) == ("pass", "done"), row


def test_a_judges_error_is_stored_without_its_endpoint(tmp_path, monkeypatch):
    async def judged(*a, **kw):
        return Verdict(UNSCORED, None, None, "judge", ["judge unavailable"], judge={
            "error": "HTTPStatusError: Server error '503' for url "
                     "'http://192.0.2.10:11434/v1/chat/completions'",
            "provenance": {"model": "qwen2.5:14b-instruct", "pinned": True}})

    monkeypatch.setattr(lr, "decide", judged)
    row = _run(tmp_path, _LongAnswer(), ANSWER_TASK)
    stored = sqlite3.connect(str(tmp_path / "telemetry.db")).execute(
        "SELECT summary_json FROM subsystem_runs WHERE subsystem = 'model_ladder'").fetchone()[0]
    assert "192.0.2.10" not in stored and "11434" not in stored
    assert row["judge"]["provenance"] == {"model": "qwen2.5:14b-instruct", "pinned": True}


def test_reports_print_a_model_path_as_its_file_name():
    rows = [{**_sweep_row("a", "light", "pass"), "model": "/home/alice/models/Qwen3.5-9B"}]
    for report in (rec.render_tier_sweep(rows, class_order=["single_tool"]),
                   rec.render_ladder_table({"L": [{**rows[0], "tier_sweep": None}]},
                                           class_order=["single_tool"])):
        assert "/home/alice" not in report and "Qwen3.5-9B" in report


def test_reset_refuses_a_symlinked_home(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("keep me")
    root = tmp_path / "sb"
    root.mkdir()
    (root / "home").symlink_to(victim)
    with pytest.raises(fx.SandboxError, match="symlink"):
        fx.Sandbox(root).reset()
    assert (victim / "notes.txt").read_text() == "keep me"


def test_an_aborted_run_exits_3_even_when_its_rows_are_cut_short(tmp_path, monkeypatch, capsys):
    # The endpoint dies on the first task: one error row with no verdict, then
    # the abort. That is exit 3 (aborted), not 1 (a recording gap).
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "ladder_run_cli", Path(__file__).resolve().parents[1] / "scripts" / "ladder_run.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    db = tmp_path / "t.db"

    async def dead(pipeline, timeout_s=45.0):
        return False, "ReadTimeout"

    # The real dead-endpoint row, recorded by run_task, then filed under the CLI's label.
    monkeypatch.setattr(lr, "endpoint_alive", dead)
    (tmp_path / "first").mkdir()
    with pytest.raises(lr.LadderAbort):
        _run(tmp_path / "first", _Hang(), CAP_TASK)
    (dead_row,) = rec.load_rows(sqlite3.connect(str(tmp_path / "first" / "telemetry.db")), "t")

    async def dies(suite, tasks, contestant, *, run_label, telemetry_db, **kw):
        row = {k: v for k, v in dead_row.items() if k not in ("_columns", "breaker_tier_bumps")}
        static = {"suite": "ladder-v1", "suite_sha": "0" * 64, "provider": "llama_cpp",
                  "run_label": run_label}
        rec.record_summary(ToolCallTelemetry(db_path=telemetry_db), {**row, **static})
        raise lr.LadderAbort("endpoint unresponsive after the time budget (ReadTimeout)")

    monkeypatch.setattr(cli, "run_ladder", dies)
    monkeypatch.setattr(sys, "argv", [
        "ladder_run.py", "--base-url", "http://127.0.0.1:9", "--no-judge", "--tasks",
        "qa-arith-mult", "--run-label", "dead", "--telemetry-db", str(db),
        "--report", str(tmp_path / "r.md")])
    assert cli.main() == 3, capsys.readouterr().out
