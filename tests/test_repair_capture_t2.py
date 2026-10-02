"""Adapter repair capture, adapter side (WP-X.54 T-2).

THE ORPHAN
----------
Pair capture's setup call lived only in the daemon's boot and the gym
harvest. A coding run is its own process (`oara code`, launched by the
daemon per task), so the capture singleton there was never configured and
every `capture_pair` returned at its "no store" guard. About 135 "validation
failed, then succeeded" arcs from local-model coding runs in Aug-Sep became
no pair at all (audit Q1). The coding runner now configures capture from the
run's own `training:` block, and honours `capture_enabled: false`.

REPAIR KINDS
------------
The adapter's repairs were known only as free text in `repair_log`. Each
entry now also carries a structured kind (fuzzy_name / json_extract /
type_coerce / strip_params / dict_unwrap / other). The entries are still the
same strings, so what the adapter returns, and everything that joins, counts
or stores `repair_log`, is unchanged. T-3 reads the kinds in the loop.
"""

from __future__ import annotations

import copy
import json
import pickle
import sqlite3
import subprocess
from pathlib import Path
from typing import AsyncIterator

import pytest
from pydantic import BaseModel

from prometheus import __main__ as m
from prometheus.adapter import ModelAdapter
from prometheus.adapter.unwrap import try_unwrap_arguments
from prometheus.adapter.validator import Strictness, ToolCallValidator
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning import pair_capture
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


@pytest.fixture(autouse=True)
def _capture_off_around_each_test():
    """The capture store is a process singleton: start and end every test
    with it unconfigured, so no test passes on another's store."""
    pair_capture.configure({"capture_enabled": False})
    yield
    pair_capture.configure({"capture_enabled": False})


# --------------------------------------------------------------------------- #
# The orphan: a coding run captures pairs
# --------------------------------------------------------------------------- #


class _ScriptedModel(ModelProvider):
    def __init__(self, turns: list[ConversationMessage]) -> None:
        self._turns = turns
        self.calls = 0

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        if self.calls < len(self._turns):
            message = self._turns[self.calls]
        else:
            message = ConversationMessage(
                role="assistant", content=[TextBlock(text="(script exhausted)")]
            )
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=message,
            usage=UsageSnapshot(input_tokens=10, output_tokens=5),
            stop_reason="stop",
        )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README").write_text("x\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=root, check=True,
    )
    return root


def _validation_failure_then_success() -> list[ConversationMessage]:
    return [
        # A name no registered tool is within fuzzy distance of: validation
        # fails, repair refuses, the loop feeds the error back.
        ConversationMessage(role="assistant", content=[ToolUseBlock(
            id="t1", name="run_shell_command", input={"command": "true"},
        )]),
        # The model recovers with the real tool, which succeeds.
        ConversationMessage(role="assistant", content=[ToolUseBlock(
            id="t2", name="code_run", input={"command": "true"},
        )]),
        ConversationMessage(role="assistant", content=[TextBlock(text="Done.")]),
    ]


def _run_code(monkeypatch, tmp_path: Path, training_yaml: str) -> int:
    """Drive `oara code` (the coding subprocess's own entry point) end to end:
    real config load, real adapter at tier light, real sandbox, real loop.
    Only the model is scripted."""
    from prometheus.coding.sandbox import ProcessSandbox

    repo = _repo(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "coding:\n  enabled: true\n"
        "infrastructure:\n  telemetry_enabled: false\n"
        "model:\n  provider: ollama\n  model: scripted\n"
        + training_yaml,
        encoding="utf-8",
    )
    model = _ScriptedModel(_validation_failure_then_success())
    monkeypatch.setattr(m, "create_provider", lambda cfg: (model, "scripted"))
    monkeypatch.setattr(m, "_detect_tool_template_or_none", lambda cfg: None)
    monkeypatch.setattr(
        m, "create_adapter",
        lambda *a, **k: ModelAdapter(tier=ModelAdapter.TIER_LIGHT),
    )
    monkeypatch.setattr(
        "prometheus.coding.sandbox.clone_repo_for_sandbox",
        lambda *a, **k: ProcessSandbox(root=repo),
    )

    class _Args:
        config = str(cfg)
        repo = "unused"
        task_description = "Run true."
        acceptance_command = "true"
        task_id = "t2cap"
        max_rounds = 6
        max_wall_seconds = 120
        sandbox_parent = str(tmp_path / "sb")
        suppress_thinking = False
        control_dir = None

    rc = m.run_coding_task(_Args())
    assert model.calls >= 2, "the scripted run never reached the recovery call"
    return rc


def _pairs(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM training_pairs").fetchall()
    finally:
        conn.close()


class TestCodingRunCapturesPairs:

    def test_validation_failure_then_success_writes_a_retry_success_pair(
        self, monkeypatch, tmp_path
    ):
        db = tmp_path / "training.db"
        rc = _run_code(
            monkeypatch, tmp_path,
            f"training:\n  capture_enabled: true\n  db_path: {db}\n",
        )
        assert rc == 0
        assert db.exists(), "the coding run never opened training.db: capture is not wired"
        rows = [r for r in _pairs(db) if r["pair_source"] == "retry_success"]
        assert len(rows) == 1
        assert json.loads(rows[0]["rejected"])["name"] == "run_shell_command"
        assert json.loads(rows[0]["chosen"]) == {
            "name": "code_run", "input": {"command": "true"},
        }
        assert rows[0]["model_id"] == "scripted"
        assert json.loads(rows[0]["context"])["session_id"] == "coding:t2cap"

    def test_capture_enabled_false_writes_nothing(self, monkeypatch, tmp_path):
        db = tmp_path / "training.db"
        rc = _run_code(
            monkeypatch, tmp_path,
            f"training:\n  capture_enabled: false\n  db_path: {db}\n",
        )
        assert rc == 0
        assert not db.exists()
        assert pair_capture.get_store() is None

    def test_a_session_given_no_training_config_leaves_capture_alone(self, tmp_path):
        """Only the runner that loaded the config configures capture. A
        CodingSession built without one (tests, any in-process caller) must
        not open the operator's real training.db behind their back."""
        import asyncio

        from prometheus.coding.sandbox import ProcessSandbox
        from prometheus.coding.session import CodingSession, CodingTask

        session = CodingSession(
            provider=_ScriptedModel(
                [ConversationMessage(role="assistant", content=[TextBlock(text="ok")])]
            ),
            model="scripted",
            sandbox=ProcessSandbox(root=_repo(tmp_path)),
            task=CodingTask(task_id="t", description="d", acceptance_command="true"),
            max_rounds=2,
        )
        report = asyncio.run(session.run())
        assert report.status == "success"
        assert pair_capture.get_store() is None


# --------------------------------------------------------------------------- #
# Each repair carries a structured kind beside repair_log
# --------------------------------------------------------------------------- #


class _In(BaseModel):
    count: int
    flag: bool = False


class _Tool(BaseTool):
    name = "count_tool"
    description = "counts"
    input_model = _In

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=str(arguments.count))


class _SelfWrapIn(BaseModel):
    status: str | None = None


class _SelfWrapTool(BaseTool):
    name = "status_tool"
    description = "sessions_list shape"
    input_model = _SelfWrapIn

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=str(arguments.status))


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_Tool())
    return reg


def _kinds(log) -> list[str]:
    from prometheus.adapter.validator import repair_kinds
    return repair_kinds(log)


class TestRepairKinds:

    def test_the_vocabulary(self):
        from prometheus.adapter.validator import REPAIR_KINDS
        assert REPAIR_KINDS == (
            "fuzzy_name", "json_extract", "type_coerce", "strip_params",
            "dict_unwrap", "other",
        )

    def test_fuzzy_name(self):
        r = ToolCallValidator(Strictness.NONE).repair(
            "count_tol", {"count": 1}, "", _registry()
        )
        assert r.repaired
        assert _kinds(r.repairs_made) == ["fuzzy_name"]
        assert r.repair_kinds == ["fuzzy_name"]

    def test_json_extract(self):
        r = ToolCallValidator(Strictness.NONE).repair(
            "count_tool", '```json\n{"count": 2}\n```', "", _registry()
        )
        assert r.repaired
        assert r.repair_kinds == ["json_extract"]

    def test_type_coerce_and_strip_params_in_order(self):
        r = ToolCallValidator(Strictness.MEDIUM).repair(
            "count_tool", {"count": "5", "bogus": 1}, "", _registry()
        )
        assert r.repaired and r.tool_input == {"count": 5}
        assert r.repair_kinds == ["type_coerce", "strip_params"]

    def test_several_repairs_on_one_call_keep_one_kind_each(self):
        r = ToolCallValidator(Strictness.MEDIUM).repair(
            "count_tol", '{"count": "7"}', "", _registry()
        )
        assert r.repaired
        assert r.repair_kinds == ["fuzzy_name", "json_extract", "type_coerce"]
        assert len(r.repair_kinds) == len(r.repairs_made)

    def test_dict_unwrap_both_transforms(self):
        tool = _SelfWrapTool()
        unwrapped = try_unwrap_arguments(tool, {"status": {"status": "failed"}})
        assert unwrapped is not None
        _, log = unwrapped
        assert _kinds(log) == ["dict_unwrap"]

        class _PromptIn(BaseModel):
            prompt: str

        class _PromptTool(_SelfWrapTool):
            name = "task_tool"
            input_model = _PromptIn

        promoted = try_unwrap_arguments(_PromptTool(), {"task": {"prompt": "go"}})
        assert promoted == ({"prompt": "go"}, [
            "promoted inner dict wrapped under 'task' to arguments",
        ])
        assert _kinds(promoted[1]) == ["dict_unwrap"]

    def test_an_entry_from_anywhere_else_is_other(self):
        assert _kinds(["a free-text note", ""]) == ["other", "other"]

    def test_the_adapter_returns_what_it_always_returned(self):
        """Same tuple, same strings: the kind rides beside the text."""
        adapter = ModelAdapter(tier=ModelAdapter.TIER_LIGHT)
        name, inp, log = adapter.validate_and_repair(
            "count_tol", {"count": 3}, _registry()
        )
        assert (name, inp) == ("count_tool", {"count": 3})
        assert log == [
            "fuzzy-matched tool name 'count_tol' → 'count_tool' (distance 1)"
        ]
        assert all(isinstance(e, str) for e in log)
        assert "; ".join(log) == log[0]
        assert json.dumps({"repair_log": log}) == json.dumps(
            {"repair_log": [str(e) for e in log]}
        )
        assert _kinds(log) == ["fuzzy_name"]
        # Nothing to repair → nothing logged, as before.
        assert adapter.validate_and_repair("count_tool", {"count": 3}, _registry())[2] == []

    def test_a_kinded_entry_survives_copy_and_pickle(self):
        r = ToolCallValidator(Strictness.NONE).repair(
            "count_tol", {"count": 1}, "", _registry()
        )
        for clone in (
            copy.copy(r.repairs_made),
            copy.deepcopy(r.repairs_made),
            pickle.loads(pickle.dumps(r.repairs_made)),
        ):
            assert clone == r.repairs_made
            assert _kinds(clone) == ["fuzzy_name"]

    def test_an_unknown_kind_is_refused(self):
        from prometheus.adapter.validator import RepairNote
        with pytest.raises(ValueError):
            RepairNote("x", "grammar")
