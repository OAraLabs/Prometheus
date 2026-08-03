"""`served_model` — what actually served the call, recorded beside what was asked for.

WHY A SECOND COLUMN RATHER THAN A CORRECTION. `model` is the name the CALLER
requested; `served_model` is what the server echoed back as having served the
call. Overwriting one with the other would destroy the only evidence that they
ever disagreed — and their disagreement is the finding: six out-of-daemon
harnesses read `config["model"]` directly and never detect, so `gemma4-26b`
rows kept being written for months after the server moved to Qwen. Recording
both makes that visible in the data instead of requiring an archaeologist.

BOTH DIRECTIONS ARE TESTED, because a column that only ever agrees proves
nothing: there are match cases AND a divergence case where the provider echoes
a different name than was requested.

COST IS DELIBERATELY UNTOUCHED. `CostTracker` keys off the requested string
(`PRICING.get(model)` with a prefix fallback), so a served name that differs
from a pricing key must never reach it. The tool-call `record()` path does not
touch cost at all — that lives in `record_run` — and these tests pin that.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry


REQUESTED = "gemma4-26b"                      # the stale config label
SERVED = "Qwen3.6-27B-UD-Q4_K_XL.gguf"        # what the server actually ran


@pytest.fixture
def tel(tmp_path):
    return ToolCallTelemetry(db_path=tmp_path / "telemetry.db")


def _rows(tel):
    cur = tel._conn.execute("SELECT model, served_model, tool_name FROM tool_calls")
    return [dict(zip(("model", "served_model", "tool_name"), r)) for r in cur.fetchall()]


class TestBothColumnsLand:
    def test_agreement_records_both(self, tel):
        """The ordinary case: the daemon asked for what the server served."""
        tel.record(model=SERVED, tool_name="bash", success=True, served_model=SERVED)
        row = _rows(tel)[0]
        assert row["model"] == SERVED
        assert row["served_model"] == SERVED

    def test_divergence_records_BOTH_distinctly(self, tel):
        """The case this column exists for — a harness's stale config label.

        Neither value is allowed to win: `model` must keep the requested name
        (cost and every historical query depend on it) and `served_model` must
        keep the truth.
        """
        tel.record(model=REQUESTED, tool_name="grep", success=True, served_model=SERVED)
        row = _rows(tel)[0]
        assert row["model"] == REQUESTED, "the requested name must not be overwritten"
        assert row["served_model"] == SERVED, "the served name must be preserved"
        assert row["model"] != row["served_model"]

    def test_absent_served_model_is_null_not_an_error(self, tel):
        """Providers that do not echo a model must still record cleanly."""
        tel.record(model=REQUESTED, tool_name="bash", success=True)
        row = _rows(tel)[0]
        assert row["model"] == REQUESTED
        assert row["served_model"] is None

    def test_divergence_is_queryable(self, tel):
        """The point of two columns: disagreement can be selected for.

        This is the query that would have surfaced the stale-label problem in
        one line instead of a survey.
        """
        tel.record(model=SERVED, tool_name="a", success=True, served_model=SERVED)
        tel.record(model=REQUESTED, tool_name="b", success=True, served_model=SERVED)
        tel.record(model=REQUESTED, tool_name="c", success=True)
        n = tel._conn.execute(
            "SELECT COUNT(*) FROM tool_calls "
            "WHERE served_model IS NOT NULL AND served_model != model"
        ).fetchone()[0]
        assert n == 1


class TestMigration:
    def test_column_is_added_to_a_pre_existing_db(self, tmp_path):
        """An existing telemetry.db must gain the column, not fail to open.

        The live DB has 5,555 rows predating this field; they stay readable
        with served_model NULL.
        """
        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE tool_calls ("
            " id TEXT PRIMARY KEY, timestamp REAL NOT NULL, model TEXT NOT NULL,"
            " tool_name TEXT NOT NULL, success INTEGER NOT NULL,"
            " retries INTEGER NOT NULL DEFAULT 0, latency_ms REAL NOT NULL DEFAULT 0.0,"
            " error_type TEXT, error_detail TEXT)"
        )
        conn.execute(
            "INSERT INTO tool_calls (id,timestamp,model,tool_name,success) "
            "VALUES ('old',1.0,?,'bash',1)", (REQUESTED,))
        conn.commit()
        conn.close()

        tel = ToolCallTelemetry(db_path=db)
        cols = {r[1] for r in tel._conn.execute("PRAGMA table_info(tool_calls)")}
        assert "served_model" in cols

        legacy = tel._conn.execute(
            "SELECT model, served_model FROM tool_calls WHERE id='old'").fetchone()
        assert legacy == (REQUESTED, None), "historical rows survive, unflagged"

        tel.record(model=REQUESTED, tool_name="grep", success=True, served_model=SERVED)
        assert tel._conn.execute(
            "SELECT served_model FROM tool_calls WHERE tool_name='grep'"
        ).fetchone()[0] == SERVED


class TestCostIsUnaffected:
    """The requested string stays the cost key. Divergence must not reach it."""

    def test_tool_call_record_never_touches_the_cost_tracker(self, tel, monkeypatch):
        import prometheus.telemetry.cost as cost_mod

        calls: list[tuple] = []

        class _Handle:
            def record(self, model, i, o):
                calls.append((model, i, o))
                return 0.0

        monkeypatch.setattr(cost_mod, "get_cost_tracker_handle", lambda: _Handle())
        tel.record(model=REQUESTED, tool_name="grep", success=True, served_model=SERVED)
        assert calls == [], "tool-call telemetry must not drive cost accounting"

    def test_cost_lookup_keys_off_the_requested_name(self):
        """Pin the reason served_model is kept away from cost.

        PRICING is keyed by requested model id. A served name (a gguf filename
        locally, or a dated snapshot id from a cloud provider) is not a pricing
        key, so routing it into cost would silently zero or mis-rate calls.
        """
        from prometheus.telemetry.cost import CostTracker

        t = CostTracker()
        assert t.record(SERVED, 1000, 1000) == 0.0, (
            "a gguf filename is not a pricing key"
        )
        assert t.total_cost == 0.0

    def test_served_model_is_not_a_parameter_of_the_cost_path(self):
        """Structural: record_run (which does drive cost) has no served_model.

        Keeps the separation a property of the signature rather than a
        convention someone has to remember.
        """
        import inspect

        assert "served_model" not in inspect.signature(
            ToolCallTelemetry.record_run).parameters
        assert "served_model" in inspect.signature(
            ToolCallTelemetry.record).parameters


# ---------------------------------------------------------------------------
# End to end: provider captures the echo, dispatch lands both columns
# ---------------------------------------------------------------------------


class _FakeResponse:
    status_code = 200

    def __init__(self, lines):
        self._lines = lines
        self.text = ""

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeStreamCM:
    def __init__(self, lines):
        self._lines = lines

    async def __aenter__(self):
        return _FakeResponse(self._lines)

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    """Stands in for httpx.AsyncClient at the llama.cpp HTTP boundary."""

    lines: list[str] = []

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def stream(self, *a: Any, **k: Any):
        return _FakeStreamCM(type(self).lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestProviderCapturesTheEcho:
    """The server's own answer, taken from the response it already sends."""

    def _run(self, lines):
        import asyncio

        import prometheus.providers.llama_cpp as lc
        from prometheus.providers.base import ApiMessageRequest

        orig = lc.httpx.AsyncClient
        _FakeClient.lines = lines
        lc.httpx.AsyncClient = _FakeClient
        try:
            provider = lc.LlamaCppProvider(base_url="http://fake:8080")

            async def _drive():
                events = []
                async for ev in provider.stream_message(
                    ApiMessageRequest(
                        model=REQUESTED,
                        messages=[{"role": "user", "content": "hi"}],
                        system_prompt="", max_tokens=8,
                    )
                ):
                    events.append(ev)
                return events

            return asyncio.run(_drive())
        finally:
            lc.httpx.AsyncClient = orig

    def test_served_model_is_taken_from_the_response(self):
        """Requested REQUESTED, server says SERVED — the event carries SERVED."""
        lines = [
            'data: {"model":"%s","choices":[{"delta":{"content":"ok"},'
            '"finish_reason":null,"index":0}]}' % SERVED,
            'data: {"model":"%s","choices":[{"delta":{},"finish_reason":"stop",'
            '"index":0}],"usage":{"prompt_tokens":3,"completion_tokens":1}}' % SERVED,
            "data: [DONE]",
        ]
        complete = [e for e in self._run(lines) if hasattr(e, "usage")]
        assert complete, "expected a terminal event"
        assert complete[-1].served_model == SERVED

    def test_absent_model_field_leaves_it_none(self):
        """A provider that does not echo must not fabricate a value."""
        lines = [
            'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null,"index":0}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
            "data: [DONE]",
        ]
        complete = [e for e in self._run(lines) if hasattr(e, "usage")]
        assert complete[-1].served_model is None


class TestDispatchLandsBoth:
    """The loop threads it per-turn and both columns reach the row."""

    def _execute(self, tel, served):
        import asyncio

        from prometheus.adapter import ModelAdapter
        from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
        from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
        from pydantic import BaseModel

        class _In(BaseModel):
            pass

        class _Tool(BaseTool):
            name = "noop"
            description = "does nothing"
            input_model = _In

            async def execute(self, arguments, context):
                return ToolResult(output="ok")

        reg = ToolRegistry()
        reg.register(_Tool())
        ctx = LoopContext(
            provider=None, model=REQUESTED, system_prompt="", max_tokens=32,
            tool_registry=reg, adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT),
            telemetry=tel,
        )
        return asyncio.run(
            _execute_tool_call(ctx, "noop", "t1", {}, served_model=served)
        )

    def test_divergent_names_both_reach_the_row(self, tel):
        """context.model is the stale config label; the server served Qwen."""
        block = self._execute(tel, SERVED)
        assert not block.is_error
        row = _rows(tel)[0]
        assert row["model"] == REQUESTED
        assert row["served_model"] == SERVED

    def test_matching_names_both_reach_the_row(self, tel):
        self._execute(tel, REQUESTED)
        row = _rows(tel)[0]
        assert row["model"] == REQUESTED
        assert row["served_model"] == REQUESTED

    def test_no_served_model_still_records(self, tel):
        """Every non-llama.cpp provider today — must not regress the row."""
        self._execute(tel, None)
        row = _rows(tel)[0]
        assert row["model"] == REQUESTED
        assert row["served_model"] is None


class TestLoopThreadsItEndToEnd:
    """The link the direct-dispatch tests cannot see.

    TestDispatchLandsBoth calls _execute_tool_call directly, so it passes even
    if the LOOP stops forwarding the value — the wired-but-untested shape. This
    drives a real turn through AgentLoop with a faked llama.cpp stream that
    emits a tool call, so the whole chain is exercised: response echo ->
    ApiMessageCompleteEvent -> per-turn local -> _dispatch_tool_calls ->
    _execute_tool_call -> telemetry row.
    """

    def test_served_model_reaches_the_row_through_a_real_turn(self, tel, monkeypatch):
        import asyncio

        import httpx
        from pydantic import BaseModel

        from prometheus.adapter import ModelAdapter
        from prometheus.engine.agent_loop import AgentLoop
        from prometheus.providers.llama_cpp import LlamaCppProvider
        from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

        class _In(BaseModel):
            pass

        class _Tool(BaseTool):
            name = "noop"
            description = "does nothing"
            input_model = _In

            async def execute(self, arguments, context):
                return ToolResult(output="ok")

        tool_chunk = (
            'data: {"model":"%s","choices":[{"index":0,"delta":{"tool_calls":'
            '[{"index":0,"id":"c1","function":{"name":"noop","arguments":"{}"}}]},'
            '"finish_reason":null}]}' % SERVED
        )
        stop_chunk = (
            'data: {"model":"%s","choices":[{"index":0,"delta":{},'
            '"finish_reason":"tool_calls"}],'
            '"usage":{"prompt_tokens":5,"completion_tokens":2}}' % SERVED
        )
        done = (
            'data: {"model":"%s","choices":[{"index":0,"delta":{"content":"done"},'
            '"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":6,"completion_tokens":2}}' % SERVED
        )

        # Turn 1 emits the tool call; turn 2 must terminate, or the loop
        # replays the same call until it hits max_turns.
        turns = [
            [tool_chunk, stop_chunk, "data: [DONE]"],
            [done, "data: [DONE]"],
        ]

        class _QueueClient(_FakeClient):
            def stream(self, *a: Any, **k: Any):
                return _FakeStreamCM(turns.pop(0) if turns else [done, "data: [DONE]"])

        monkeypatch.setattr(httpx, "AsyncClient", _QueueClient)

        reg = ToolRegistry()
        reg.register(_Tool())
        loop = AgentLoop(
            provider=LlamaCppProvider(base_url="http://unit.test:1"),
            model=REQUESTED,                       # the stale config label
            tool_registry=reg,
            adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT),
            telemetry=tel,
            max_turns=2,
            max_tool_iterations=2,
        )
        asyncio.run(loop.run_async(system_prompt="", user_message="go"))

        rows = [r for r in _rows(tel) if r["tool_name"] == "noop"]
        assert rows, "the turn should have dispatched the tool call"
        assert rows[0]["model"] == REQUESTED
        assert rows[0]["served_model"] == SERVED, (
            "the loop must forward the per-turn served_model into telemetry"
        )
