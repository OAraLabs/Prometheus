"""Repair-pair flywheel — store, loop capture wiring, miner, export.

Acceptance (sprint Workstream A): an induced known-repairable bad call lands
a pair with non-empty context; export produces valid JSONL; a deliberately
broken capture path logs loudly (silent_failures row) and the turn completes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.learning import pair_capture
from prometheus.learning.pair_capture import PairStore
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestPairStore:

    def test_add_stats_roundtrip(self, tmp_path):
        store = PairStore(tmp_path / "t.db")
        ok = store.add_pair(
            pair_source="self_correction",
            model_id="gemma",
            tool_name="task_create",
            context={"kind": "lcm_ref", "session_id": "telegram:1"},
            rejected={"name": "task_create", "input": {"prompt": {"prompt": None}}},
            chosen={"name": "task_create", "input": {"type": "local_agent", "prompt": "do it"}},
        )
        assert ok
        s = store.stats()
        assert s["total"] == 1
        assert s["by_source"] == {"self_correction": 1}
        assert s["by_tool"] == {"task_create": 1}

    def test_dedupe_identical_context_rejected(self, tmp_path):
        store = PairStore(tmp_path / "t.db")
        kwargs = dict(
            pair_source="schema_repair",
            model_id="m",
            tool_name="bash",
            context={"kind": "lcm_ref", "session_id": "s"},
            rejected={"name": "bash", "input": {"command": 5}},
            chosen={"name": "bash", "input": {"command": "5"}},
        )
        assert store.add_pair(**kwargs)
        assert not store.add_pair(**kwargs)  # dedupe hit
        assert store.stats()["total"] == 1

    def test_unknown_source_rejected(self, tmp_path):
        store = PairStore(tmp_path / "t.db")
        with pytest.raises(ValueError, match="unknown pair_source"):
            store.add_pair(
                pair_source="vibes", model_id="m", tool_name="t",
                context=None, rejected=None, chosen={"name": "t", "input": {}},
            )

    def test_cloud_golden_allows_null_rejected(self, tmp_path):
        store = PairStore(tmp_path / "t.db")
        assert store.add_pair(
            pair_source="cloud_golden", model_id="claude", tool_name="bash",
            context={"kind": "lcm_ref"}, rejected=None,
            chosen={"name": "bash", "input": {"command": "ls"}},
        )
        row = store.rows_since()[0]
        assert row["rejected"] is None


# ---------------------------------------------------------------------------
# Loop capture wiring (side effects through _execute_tool_call)
# ---------------------------------------------------------------------------


class _EchoInput(BaseModel):
    text: str


class _EchoTool(BaseTool):
    name = "echo_tool"
    description = "echoes"
    input_model = _EchoInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"echo: {arguments.text}")


class _ModeInput(BaseModel):
    type: str = "a"
    command: str | None = None


class _ModeTool(BaseTool):
    name = "mode_tool"
    description = "mode-discriminated tool (task_create shape)"
    input_model = _ModeInput

    async def execute(self, arguments, context):  # noqa: ANN001
        if arguments.type == "a" and not arguments.command:
            return ToolResult(output="'command' is required for a tasks", is_error=True)
        return ToolResult(output="started")


@pytest.fixture
def capture_env(tmp_path):
    pair_capture.configure({"db_path": str(tmp_path / "training.db")})
    yield pair_capture.get_store()
    pair_capture.configure({"capture_enabled": False})


def _ctx(tmp_path, **kw):
    reg = ToolRegistry()
    reg.register(_EchoTool())
    reg.register(_ModeTool())
    defaults = dict(
        provider=None, model="gemma-test", system_prompt="", max_tokens=256,
        tool_registry=reg, adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT),
        telemetry=ToolCallTelemetry(db_path=tmp_path / "tel.db"),
        session_id="telegram:42",
    )
    defaults.update(kw)
    return LoopContext(**defaults)


class TestLoopCapture:

    def test_levenshtein_repair_produces_pair(self, tmp_path, capture_env):
        ctx = _ctx(tmp_path)
        block = asyncio.run(
            _execute_tool_call(ctx, "echo_tool2", "t1", {"text": "hi"})
        )
        assert not block.is_error
        rows = capture_env.rows_since()
        assert len(rows) == 1
        row = rows[0]
        assert row["pair_source"] == "levenshtein_repair"
        assert json.loads(row["rejected"])["name"] == "echo_tool2"
        assert json.loads(row["chosen"])["name"] == "echo_tool"
        context = json.loads(row["context"])
        assert context["session_id"] == "telegram:42"
        assert context["tool_schema"]["name"] == "echo_tool"  # non-empty context

    def test_self_correction_pair_after_pydantic_failure(self, tmp_path, capture_env):
        ctx = _ctx(tmp_path)
        bad = asyncio.run(
            _execute_tool_call(ctx, "echo_tool", "t1", {"text": {"text": None}})
        )
        assert bad.is_error
        good = asyncio.run(
            _execute_tool_call(ctx, "echo_tool", "t2", {"text": "fixed"})
        )
        assert not good.is_error
        rows = [r for r in capture_env.rows_since()
                if r["pair_source"] == "self_correction"]
        assert len(rows) == 1
        assert json.loads(rows[0]["rejected"])["input"] == {"text": {"text": None}}
        assert json.loads(rows[0]["chosen"])["input"] == {"text": "fixed"}
        assert "error_feedback" in json.loads(rows[0]["meta"])

    def test_mode_misuse_tool_error_pairs_on_recovery(self, tmp_path, capture_env):
        ctx = _ctx(tmp_path)
        bad = asyncio.run(_execute_tool_call(ctx, "mode_tool", "t1", {"type": "a"}))
        assert bad.is_error  # "'command' is required for a tasks"
        good = asyncio.run(
            _execute_tool_call(ctx, "mode_tool", "t2", {"type": "a", "command": "x"})
        )
        assert not good.is_error
        rows = [r for r in capture_env.rows_since()
                if r["pair_source"] == "self_correction"]
        assert len(rows) == 1
        assert json.loads(rows[0]["chosen"])["input"]["command"] == "x"

    def test_unknown_name_recovery_pairs_via_fallback(self, tmp_path, capture_env):
        ctx = _ctx(tmp_path)
        # far from any registered name → repair refused → ValueError path
        bad = asyncio.run(
            _execute_tool_call(ctx, "quantum_zzz_analyzer", "t1", {"x": 1})
        )
        assert bad.is_error
        good = asyncio.run(
            _execute_tool_call(ctx, "echo_tool", "t2", {"text": "recovered"})
        )
        assert not good.is_error
        rows = [r for r in capture_env.rows_since()
                if r["pair_source"] == "retry_success"]
        assert len(rows) == 1
        assert json.loads(rows[0]["rejected"])["name"] == "quantum_zzz_analyzer"
        assert json.loads(rows[0]["chosen"])["name"] == "echo_tool"

    def test_no_pair_when_capture_disabled(self, tmp_path):
        pair_capture.configure({"capture_enabled": False})
        ctx = _ctx(tmp_path)
        block = asyncio.run(
            _execute_tool_call(ctx, "echo_tool2", "t1", {"text": "hi"})
        )
        assert not block.is_error  # repair still works without capture

    def test_broken_capture_is_loud_but_non_blocking(self, tmp_path, capture_env, monkeypatch):
        def boom(self, **kw):  # noqa: ANN001
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(PairStore, "add_pair", boom)
        ctx = _ctx(tmp_path)
        block = asyncio.run(
            _execute_tool_call(ctx, "echo_tool2", "t1", {"text": "hi"})
        )
        # the turn completes …
        assert not block.is_error
        assert "echo: hi" in block.content
        # … and the failure is LOUD: a silent_failures telemetry row exists
        rows = ctx.telemetry._conn.execute(
            "SELECT subsystem, exception_type FROM silent_failures "
            "WHERE subsystem = 'pair_capture'"
        ).fetchall()
        assert rows, "capture failure must land in silent_failures"
        assert rows[0][1] == "RuntimeError"


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------


def _load_miner():
    spec = importlib.util.spec_from_file_location(
        "mine_training_pairs", Path("scripts/mine_training_pairs.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TEL_SCHEMA = """
CREATE TABLE tool_calls (
    id TEXT PRIMARY KEY, timestamp REAL, model TEXT, tool_name TEXT,
    success INTEGER, retries INTEGER DEFAULT 0, latency_ms REAL DEFAULT 0,
    error_type TEXT, error_detail TEXT, raw_model_output TEXT,
    parsed_tool_call TEXT, is_golden INTEGER DEFAULT 0, repairs INTEGER DEFAULT 0
);
"""


class TestMiner:

    def test_mines_calling_errors_not_execution_outcomes(self, tmp_path):
        db = tmp_path / "tel.db"
        conn = sqlite3.connect(db)
        conn.executescript(_TEL_SCHEMA)
        t0 = time.time()

        def row(i, ts, tool, success, etype, edetail, call):
            conn.execute(
                "INSERT INTO tool_calls (id, timestamp, model, tool_name, success,"
                " error_type, error_detail, parsed_tool_call)"
                " VALUES (?, ?, 'gemma', ?, ?, ?, ?, ?)",
                (f"r{i}", ts, tool, success, etype, edetail,
                 json.dumps(call) if call else None),
            )

        # minable: input_validation failure then success on the same tool
        row(1, t0, "task_create", 0, "input_validation",
            "1 validation error for TaskCreateToolInput",
            {"name": "task_create", "input": {"prompt": {"prompt": None}}})
        row(2, t0 + 30, "task_create", 1, None, None,
            {"name": "task_create", "input": {"type": "local_agent", "prompt": "ok"}})
        # minable: mode-misuse tool_error then success
        row(3, t0 + 100, "task_create", 0, "tool_error",
            "'command' is required for local_bash tasks",
            {"name": "task_create", "input": {"description": "d", "prompt": "p"}})
        row(4, t0 + 130, "task_create", 1, None, None,
            {"name": "task_create", "input": {"type": "local_agent", "description": "d", "prompt": "p"}})
        # NOT minable: bash nonzero exit (well-formed call, execution outcome)
        row(5, t0 + 200, "bash", 0, "tool_error", "(no output)",
            {"name": "bash", "input": {"command": "grep nope file"}})
        row(6, t0 + 210, "bash", 1, None, None,
            {"name": "bash", "input": {"command": "ls"}})
        # NOT minable: success outside the window
        row(7, t0 + 300, "glob", 0, "input_validation", "bad",
            {"name": "glob", "input": {"pattern": 5}})
        row(8, t0 + 2000, "glob", 1, None, None,
            {"name": "glob", "input": {"pattern": "*.py"}})
        conn.commit()
        conn.close()

        miner = _load_miner()
        pairs = miner.mine(600.0, db)
        assert len(pairs) == 2
        assert all(p["tool_name"] == "task_create" for p in pairs)
        assert all(p["context"]["kind"] == "telemetry_only" for p in pairs)
        assert pairs[0]["rejected"]["input"] == {"prompt": {"prompt": None}}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:

    def test_export_emits_valid_dpo_jsonl(self, tmp_path):
        db = tmp_path / "training.db"
        store = PairStore(db)
        store.add_pair(
            pair_source="self_correction", model_id="m", tool_name="bash",
            context={"kind": "telemetry_only"},
            rejected={"name": "bash", "input": {"command": 1}},
            chosen={"name": "bash", "input": {"command": "ls"}},
        )
        store.add_pair(
            pair_source="cloud_golden", model_id="claude", tool_name="bash",
            context={"kind": "lcm_ref"}, rejected=None,
            chosen={"name": "bash", "input": {"command": "pwd"}},
        )
        out = tmp_path / "pairs.jsonl"
        proc = subprocess.run(
            [sys.executable, "scripts/export_training_pairs.py",
             "--db", str(db), "--out", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 1  # golden excluded by default
        rec = json.loads(lines[0])
        assert set(rec) == {"prompt", "chosen", "rejected", "meta"}
        assert json.loads(rec["chosen"])["input"] == {"command": "ls"}
        assert "cloud_golden chosen-only rows skipped" in proc.stdout

        proc2 = subprocess.run(
            [sys.executable, "scripts/export_training_pairs.py",
             "--db", str(db), "--out", str(out), "--include-golden"],
            capture_output=True, text=True,
        )
        assert proc2.returncode == 0, proc2.stderr
        lines2 = out.read_text().strip().splitlines()
        assert len(lines2) == 2
        assert any(json.loads(l)["rejected"] is None for l in lines2)


class TestUnwrapCaptureComposition:
    """Phase 4 × Workstream A: an accepted unwrap emits a schema_repair pair."""

    def test_unwrap_emits_schema_repair_pair(self, tmp_path, capture_env):
        class _StatusInput(BaseModel):
            status: str | None = None

        class _StatusTool(BaseTool):
            name = "status_tool"
            description = "sessions_list shape"
            input_model = _StatusInput

            async def execute(self, arguments, context):  # noqa: ANN001
                return ToolResult(output=f"status={arguments.status}")

        reg = ToolRegistry()
        reg.register(_StatusTool())
        ctx = LoopContext(
            provider=None, model="gemma-test", system_prompt="", max_tokens=64,
            tool_registry=reg,
            adapter=ModelAdapter(
                tier=ModelAdapter.TIER_LIGHT,
                unwrap_tools=frozenset({"status_tool"}),
            ),
            telemetry=ToolCallTelemetry(db_path=tmp_path / "tel2.db"),
            session_id="telegram:42",
        )
        block = asyncio.run(_execute_tool_call(
            ctx, "status_tool", "t1", {"status": {"status": "failed"}}
        ))
        assert not block.is_error
        assert "status=failed" in block.content
        rows = [r for r in capture_env.rows_since()
                if r["pair_source"] == "schema_repair"]
        assert len(rows) == 1
        assert json.loads(rows[0]["rejected"])["input"] == {"status": {"status": "failed"}}
        assert json.loads(rows[0]["chosen"])["input"] == {"status": "failed"}
        assert "unwrap_log" in json.loads(rows[0]["meta"])
