"""Gym harness unit tests — manifest rules, taskset freezing, scoring, store, report.

The runner's live path is exercised by actual gym runs (it needs the model);
everything deterministic around it is tested here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.gym.manifest import load_manifest
from prometheus.gym.report import MIN_RUNS_PER_ARM, render_report
from prometheus.gym.runner import apply_variable, build_seed_messages
from prometheus.gym.scoring import RunTranscript, score
from prometheus.gym.store import GymStore
from prometheus.gym.tasks import load_taskset


# ---------------------------------------------------------------------------
# Manifests — the single-variable rule
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


BASE = """
series: s1
experiment: expX
taskset: gym/tasksets/v1.yaml
runs_per_task: 3
"""


class TestManifestRules:

    def test_baseline_none(self, tmp_path):
        m = load_manifest(_write(tmp_path, "m.yaml", BASE + "variable: none\n"))
        assert m.variable_name == "none"
        assert m.runs_per_task == 3

    def test_single_variable_ok(self, tmp_path):
        m = load_manifest(_write(tmp_path, "m.yaml", BASE + (
            "variable:\n"
            "  example_call:\n"
            "    tool: task_create\n"
            "    example: {type: local_agent}\n"
        )))
        assert m.variable_name == "example_call"
        assert m.variable_payload["tool"] == "task_create"

    def test_two_variables_refused(self, tmp_path):
        with pytest.raises(ValueError, match="REFUSED.*exactly one"):
            load_manifest(_write(tmp_path, "m.yaml", BASE + (
                "variable:\n"
                "  example_call: {tool: t, example: {}}\n"
                "  system_prompt: {text: hi}\n"
            )))

    def test_reserved_variable_refused(self, tmp_path):
        with pytest.raises(ValueError, match="reserved but not implemented"):
            load_manifest(_write(tmp_path, "m.yaml", BASE + (
                "variable:\n"
                "  sampling: {temperature: 0.2}\n"
            )))

    def test_unknown_variable_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown variable"):
            load_manifest(_write(tmp_path, "m.yaml", BASE + (
                "variable:\n"
                "  hypnosis: {depth: 3}\n"
            )))

    def test_payload_sanity(self, tmp_path):
        with pytest.raises(ValueError, match="example_call needs"):
            load_manifest(_write(tmp_path, "m.yaml", BASE + (
                "variable:\n"
                "  example_call: {tool: task_create}\n"
            )))


# ---------------------------------------------------------------------------
# Task sets — load + freeze
# ---------------------------------------------------------------------------


class TestTaskSet:

    def test_v1_loads_and_hashes(self):
        ts = load_taskset("gym/tasksets/v1.yaml")
        assert len(ts.tasks) >= 20
        assert len(ts.sha256) == 64
        ids = [t.id for t in ts.tasks]
        assert len(ids) == len(set(ids))
        cats = {t.category for t in ts.tasks}
        assert {"control", "task_create", "argshape", "namespace",
                "resilience", "lcm"} <= cats
        # the collapse-arc replay carries a seed
        replay = next(t for t in ts.tasks if t.id == "resilience_collapse_arc_replay")
        assert replay.seed
        # the task_create tasks stub execution
        for t in ts.tasks:
            if t.category == "task_create":
                assert t.stub_tools == ["task_create"]

    def test_unknown_predicate_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: 1\ntasks:\n"
            "  - id: a\n    category: c\n    prompt: p\n"
            "    score: {expect_vibes: good}\n"
        )
        with pytest.raises(ValueError, match="unknown score predicate"):
            load_taskset(bad)

    def test_editing_changes_hash(self, tmp_path):
        text = (
            "version: 1\ntasks:\n"
            "  - id: a\n    category: c\n    prompt: p\n"
            "    score: {require_graceful: true}\n"
        )
        a = tmp_path / "a.yaml"
        a.write_text(text)
        b = tmp_path / "b.yaml"
        b.write_text(text + "# tweak\n")
        assert load_taskset(a).sha256 != load_taskset(b).sha256


# ---------------------------------------------------------------------------
# Seed reconstruction
# ---------------------------------------------------------------------------


class TestSeedMessages:

    def test_arc_reconstruction_pairs_results_to_calls(self):
        msgs = build_seed_messages([
            {"user": "job is stuck"},
            {"assistant_tool_call": {"name": "bash", "input": {"command": "x"}}},
            {"tool_result": {"content": "Command timed out after 30 seconds", "is_error": True}},
        ])
        assert [m.role for m in msgs] == ["user", "assistant", "user"]
        call = msgs[1].content[0]
        result = msgs[2].content[0]
        assert isinstance(call, ToolUseBlock)
        assert isinstance(result, ToolResultBlock)
        assert result.tool_use_id == call.id
        assert result.is_error

    def test_orphan_tool_result_rejected(self):
        with pytest.raises(ValueError, match="no preceding"):
            build_seed_messages([{"tool_result": {"content": "x"}}])


# ---------------------------------------------------------------------------
# Scoring — deterministic predicates over transcripts
# ---------------------------------------------------------------------------


def _transcript(*blocks_per_msg, dropped=0):
    msgs = []
    for role, blocks in blocks_per_msg:
        msgs.append(ConversationMessage(role=role, content=list(blocks)))
    t = RunTranscript.from_messages(msgs)
    t.dropped_malformed = dropped
    return t


def _call(name, input_, *, ok=True, tid="t1", result="done"):
    return [
        ("assistant", [ToolUseBlock(id=tid, name=name, input=input_)]),
        ("user", [ToolResultBlock(tool_use_id=tid, content=result, is_error=not ok)]),
    ]


class TestScoring:

    def test_expect_tool_success(self, tmp_path):
        t = _transcript(*_call("bash", {"command": "echo hi"}),
                        ("assistant", [TextBlock(text="hi there")]))
        passed, reasons = score({"expect_tool": "bash"}, t, tmp_path)
        assert passed, reasons

    def test_expect_tool_fails_on_error_result(self, tmp_path):
        t = _transcript(*_call("bash", {"command": "x"}, ok=False))
        passed, reasons = score({"expect_tool": "bash"}, t, tmp_path)
        assert not passed
        assert "no successful 'bash'" in reasons[0]

    def test_dict_wrap_caught_by_args_string(self, tmp_path):
        t = _transcript(*_call("task_list", {"status": {"status": None}}))
        passed, reasons = score(
            {"expect_tool": "task_list", "expect_tool_args_string": ["status"]},
            t, tmp_path,
        )
        assert not passed
        assert "should be a string" in reasons[0]

    def test_json_stuffed_prompt_caught(self, tmp_path):
        blob = '{"description":"You are a specialized agent..."}'
        t = _transcript(*_call("task_create", {
            "type": "local_agent", "description": "d", "prompt": blob,
        }))
        passed, reasons = score(
            {"expect_tool": "task_create", "prompt_not_json_blob": True},
            t, tmp_path,
        )
        assert not passed
        assert "JSON blob" in reasons[0]

    def test_plain_prompt_passes_blob_check(self, tmp_path):
        t = _transcript(*_call("task_create", {
            "type": "local_agent", "description": "d",
            "prompt": "Process rows 10-60 adding phone numbers.",
        }))
        passed, reasons = score(
            {
                "expect_tool": "task_create",
                "prompt_not_json_blob": True,
                "expect_tool_args_require": {"type": "local_agent"},
            },
            t, tmp_path,
        )
        assert passed, reasons

    def test_missing_discriminator_caught(self, tmp_path):
        t = _transcript(*_call("task_create", {
            "description": "d", "prompt": "plain text",
        }))
        passed, reasons = score(
            {"expect_tool": "task_create",
             "expect_tool_args_require": {"type": "local_agent"}},
            t, tmp_path,
        )
        assert not passed

    def test_forbid_bash_containing(self, tmp_path):
        t = _transcript(*_call("bash", {"command": "task_list --all"}))
        passed, reasons = score(
            {"forbid_bash_containing": "task_list"}, t, tmp_path
        )
        assert not passed

    def test_breaker_and_malformed_predicates(self, tmp_path):
        t = _transcript(
            ("assistant", [TextBlock(
                text="Circuit breaker tripped: 3 consecutive identical errors"
            )]),
            dropped=3,
        )
        passed, reasons = score(
            {"forbid_breaker_trip": True, "forbid_malformed": True,
             "require_graceful": True},
            t, tmp_path,
        )
        assert not passed
        assert len(reasons) == 3

    def test_file_predicates(self, tmp_path):
        (tmp_path / "out.txt").write_text("gym write ok")
        passed, reasons = score(
            {"expect_file": str(tmp_path / "out.txt"),
             "expect_file_contains": "gym write ok"},
            _transcript(("assistant", [TextBlock(text="created")])),
            tmp_path,
        )
        assert passed, reasons

    def test_orchestrator_feedback_counted(self, tmp_path):
        msgs = [
            ConversationMessage(role="assistant", content=[]),
            ConversationMessage.from_injected(
                "Your previous response contained 1 malformed tool call(s)…",
                provenance="orchestrator", is_trusted=True,
            ),
            ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
        ]
        t = RunTranscript.from_messages(msgs)
        assert t.orchestrator_feedback == 1


# ---------------------------------------------------------------------------
# apply_variable
# ---------------------------------------------------------------------------


class TestApplyVariable:

    def test_example_call_appends_example(self, tmp_path):
        m = load_manifest(_write(tmp_path, "m.yaml", BASE + (
            "variable:\n"
            "  example_call:\n"
            "    tool: task_create\n"
            "    example: {type: local_agent, description: d, prompt: p}\n"
        )))
        out = apply_variable(m, "BASE PROMPT", registry=None)
        assert out.startswith("BASE PROMPT")
        assert '"name": "task_create"' in out
        assert '"type": "local_agent"' in out

    def test_none_is_identity(self, tmp_path):
        m = load_manifest(_write(tmp_path, "m.yaml", BASE + "variable: none\n"))
        assert apply_variable(m, "BASE", registry=None) == "BASE"


# ---------------------------------------------------------------------------
# Store + report
# ---------------------------------------------------------------------------


def _seed_runs(store: GymStore, experiment: str, n_tasks: int, runs: int, pass_mask):
    i = 0
    for t in range(n_tasks):
        for r in range(runs):
            store.record_run(
                series="s1", experiment=experiment, task_id=f"task{t}",
                run_idx=r, model="m", category="control",
                success=int(pass_mask(i)), fail_reasons=None,
                tools_called="[]", latency_ms=10.0, retries=0, repairs=0,
                dropped_malformed=0, feedback_retries=0, breaker_tripped=0,
                error=None, manifest_sha="ms", taskset_sha="ts",
            )
            i += 1


class TestStoreAndReport:

    def test_store_roundtrip_and_replace_key(self, tmp_path):
        store = GymStore(tmp_path / "gym.db")
        _seed_runs(store, "exp0", n_tasks=2, runs=2, pass_mask=lambda i: True)
        assert len(store.runs("s1", "exp0")) == 4
        # re-recording the same (task, run_idx) replaces, not duplicates
        _seed_runs(store, "exp0", n_tasks=2, runs=2, pass_mask=lambda i: False)
        rows = store.runs("s1", "exp0")
        assert len(rows) == 4
        assert all(r["success"] == 0 for r in rows)

    def test_report_insufficient_n_refuses_verdict(self, tmp_path):
        store = GymStore(tmp_path / "gym.db")
        _seed_runs(store, "exp0", n_tasks=2, runs=2, pass_mask=lambda i: True)
        _seed_runs(store, "exp1", n_tasks=2, runs=2, pass_mask=lambda i: i % 2 == 0)
        report = render_report(
            "s1", "exp1",
            store.runs("s1", "exp1"),
            store.runs("s1", "exp0"),
        )
        assert "INSUFFICIENT n" in report
        assert "descriptive only" in report

    def test_report_verdict_above_bar(self, tmp_path):
        store = GymStore(tmp_path / "gym.db")
        runs_per_task = 3
        n_tasks = (MIN_RUNS_PER_ARM // runs_per_task) + 1
        _seed_runs(store, "exp0", n_tasks, runs_per_task, pass_mask=lambda i: i % 2 == 0)
        _seed_runs(store, "exp1", n_tasks, runs_per_task, pass_mask=lambda i: True)
        report = render_report(
            "s1", "exp1",
            store.runs("s1", "exp1"),
            store.runs("s1", "exp0"),
        )
        assert "INSUFFICIENT n" not in report
        assert "Baseline 52% → experiment 100%" in report  # 17/33 vs 33/33

    def test_thin_category_flagged(self, tmp_path):
        store = GymStore(tmp_path / "gym.db")
        _seed_runs(store, "exp0", n_tasks=1, runs=3, pass_mask=lambda i: True)
        report = render_report("s1", "exp0", store.runs("s1", "exp0"))
        assert "⚠️thin" in report
