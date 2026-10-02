"""Telemetry v2 loop write points (WP-X.54 T-3).

T-1 created the ``turns`` / ``responses`` / ``tool_sets`` tables, seven nullable
``tool_calls`` columns and the queued writer; nothing wrote them. These tests
pin the loop's half, through a real ``run_loop`` against a real
``ToolCallTelemetry`` on a temp file, by the rows actually written:

* one ``turns`` row per ``run_loop`` call, id minted unconditionally, end
  written in ``run_loop``'s ``finally`` with the reason the loop stopped it;
* one ``responses`` row per model round, linked to the round's ``loop_round``
  row, including rounds the guards retried or ended;
* the ``tool_calls`` columns ride on the existing INSERTs, and never on a
  ``_loop_transition`` row beyond turn_id / round_index;
* every free-text v2 column is redacted before it is stored;
* telemetry off writes nothing; the turn never waits for the writer.

Spec section 8's test list, as amended by the audit's rulings
(~/audits/20260930-telemetry-v2-audit.md).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning import pair_capture
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

SESSION = "telegram:4242"
# Letters only, built by concatenation: the shape the redactor matches and the
# repo's secret scanner leaves alone in source.
CANARY = "ghp_" + "Abcdefghij" * 4


@pytest.fixture(autouse=True)
def _capture_off_around_each_test():
    pair_capture.configure({"capture_enabled": False})
    yield
    pair_capture.configure({"capture_enabled": False})


# --------------------------------------------------------------------------- #
# Doubles: only the model is scripted
# --------------------------------------------------------------------------- #


class _In(BaseModel):
    count: int
    note: str = ""


class _CountTool(BaseTool):
    name = "count_tool"
    description = "counts"
    input_model = _In

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"counted {arguments.count}")


class _BrokenTool(BaseTool):
    name = "broken_tool"
    description = "always fails"
    input_model = _In

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"failure number {arguments.count}", is_error=True)


class _Script(ModelProvider):
    """Plays one assistant message per model call; then ends in prose."""

    def __init__(self, turns: list) -> None:
        self.turns = list(turns)
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        item = self.turns[self.calls] if self.calls < len(self.turns) else _prose("done")
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        yield ApiMessageCompleteEvent(
            message=item, usage=UsageSnapshot(input_tokens=11, output_tokens=7),
            stop_reason="stop")


def _prose(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _call(name: str, call_id: str, **inp) -> ConversationMessage:
    return ConversationMessage(role="assistant",
                               content=[ToolUseBlock(id=call_id, name=name, input=inp)])


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_CountTool())
    reg.register(_BrokenTool())
    return reg


def _run(tmp_path: Path, turns: list, *, session: str | None = SESSION, surface: str | None = "telegram",
         telemetry: bool = True, raises: type[BaseException] | None = None, **ctx_kw):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db) if telemetry else None
    ctx = LoopContext(provider=_Script(turns), model="stub-model", system_prompt="", max_tokens=128,
                      tool_registry=_registry(), telemetry=tel, **ctx_kw)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")],
                                session_id=session, surface=surface):
            pass

    if raises is None:
        asyncio.run(drain())
    else:
        with pytest.raises(raises):
            asyncio.run(drain())
    if tel is not None:
        tel.close()  # drains the v2 writer
    return db


def _rows(db: Path, sql: str, *args) -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def _turn(db: Path) -> sqlite3.Row:
    [row] = _rows(db, "SELECT * FROM turns")
    return row


def _responses(db: Path) -> list[sqlite3.Row]:
    return _rows(db, "SELECT * FROM responses ORDER BY id")


def _calls(db: Path) -> list[sqlite3.Row]:
    return _rows(db, "SELECT * FROM tool_calls WHERE tool_name != '_loop_transition' ORDER BY rowid")


def _transitions(db: Path) -> list[sqlite3.Row]:
    return _rows(db, "SELECT * FROM tool_calls WHERE tool_name = '_loop_transition' ORDER BY rowid")


# --------------------------------------------------------------------------- #
# Spec 8: a prose-only turn
# --------------------------------------------------------------------------- #


class TestProseOnlyTurn:

    def test_one_prose_response_and_no_tool_calls(self, tmp_path):
        db = _run(tmp_path, [_prose("The answer is four.")])
        [resp] = _responses(db)
        assert resp["response_kind"] == "prose"
        assert resp["prose"] == "The answer is four."
        assert resp["prose_chars"] == len("The answer is four.")
        assert resp["tool_call_count"] == 0
        assert resp["round_index"] == 0
        assert _rows(db, "SELECT * FROM tool_calls") == [], "a prose-only turn writes no tool_calls row"

    def test_the_turn_row_describes_the_turn(self, tmp_path):
        db = _run(tmp_path, [_prose("ok")])
        turn = _turn(db)
        [resp] = _responses(db)
        assert turn["turn_id"].startswith(f"{SESSION}:")
        assert resp["turn_id"] == turn["turn_id"]
        assert (turn["session_id"], turn["surface"], turn["mode"]) == (SESSION, "telegram", "agent")
        assert turn["started_at"] is not None and turn["ended_at"] >= turn["started_at"]
        assert (turn["rounds"], turn["first_prose_round"], turn["terminal_kind"]) == (1, 0, "prose")
        assert (turn["total_prompt_tokens"], turn["total_completion_tokens"]) == (11, 7)
        assert json.loads(turn["models_used"]) == ["stub-model"]
        assert json.loads(turn["tools_used"]) == []
        assert turn["not_run_calls"] == 0
        assert turn["forced_stop_reason"] is None, "the model chose to stop"
        # T-4 and T-5 own these.
        assert (turn["outcome"], turn["task_class"]) == (None, None)

    def test_the_response_links_to_its_loop_round_row(self, tmp_path):
        db = _run(tmp_path, [_prose("ok")])
        [resp] = _responses(db)
        [run] = _rows(db, "SELECT * FROM subsystem_runs WHERE id = ?", resp["loop_round_id"])
        assert (run["subsystem"], run["operation"], run["round_index"]) == ("agent_loop", "loop_round", 0)
        assert resp["session_id"] == SESSION and resp["surface"] == "telegram"
        assert resp["provider"] is not None and resp["mode"] == "agent"

    def test_the_capture_boundary_is_stamped_once(self, tmp_path):
        db = _run(tmp_path, [_prose("ok")])
        [(first,)] = _rows(db, "SELECT value FROM schema_meta WHERE key = 'telemetry_v2_capture_since'")
        [turn] = _rows(db, "SELECT started_at FROM turns")
        assert float(first) <= turn["started_at"]
        _run(tmp_path, [_prose("again")])
        [(second,)] = _rows(db, "SELECT value FROM schema_meta WHERE key = 'telemetry_v2_capture_since'")
        assert second == first, "a later process must not move the capture boundary"
        assert ToolCallTelemetry(db).telemetry_v2_capture_boundary() == float(first)


# --------------------------------------------------------------------------- #
# Tool rounds: tool_calls columns, tool_sets, transitions
# --------------------------------------------------------------------------- #


class TestToolRounds:

    def test_tool_calls_carry_turn_round_and_call_id(self, tmp_path):
        db = _run(tmp_path, [_call("count_tool", "c1", count=1), _call("count_tool", "c2", count=2),
                             _prose("done")])
        turn_id = _turn(db)["turn_id"]
        calls = _calls(db)
        assert [(c["turn_id"], c["round_index"], c["tool_use_id"]) for c in calls] == [
            (turn_id, 0, "c1"), (turn_id, 1, "c2")]
        assert [c["result_summary"] for c in calls] == ["counted 1", "counted 2"]
        assert [c["retry_index"] for c in calls] == [0, 0]
        assert [c["repair_kind"] for c in calls] == [None, None]

    def test_responses_one_per_round_with_the_tool_set(self, tmp_path):
        db = _run(tmp_path, [_call("count_tool", "c1", count=1), _prose("done")])
        resps = _responses(db)
        assert [(r["round_index"], r["response_kind"], r["tool_call_count"]) for r in resps] == [
            (0, "tool_call", 1), (1, "prose", 0)]
        assert resps[0]["prose"] is None
        [ts] = _rows(db, "SELECT * FROM tool_sets")
        assert {r["tool_set_hash"] for r in resps} == {ts["tool_set_hash"]}
        assert json.loads(ts["tool_names"]) == ["broken_tool", "count_tool"]
        assert ts["tool_count"] == 2
        loop_round_ids = [r["id"] for r in _rows(
            db, "SELECT id FROM subsystem_runs WHERE operation = 'loop_round' ORDER BY timestamp")]
        assert [r["loop_round_id"] for r in resps] == loop_round_ids
        turn = _turn(db)
        assert json.loads(turn["tools_used"]) == ["count_tool"]
        assert (turn["rounds"], turn["terminal_kind"], turn["first_prose_round"]) == (2, "prose", 1)

    def test_a_mixed_response_is_mixed_and_keeps_no_prose(self, tmp_path):
        mixed = ConversationMessage(role="assistant", content=[
            TextBlock(text="Let me count."), ToolUseBlock(id="c1", name="count_tool", input={"count": 1})])
        db = _run(tmp_path, [mixed, _prose("done")])
        first = _responses(db)[0]
        assert (first["response_kind"], first["prose_chars"], first["prose"]) == ("mixed", 13, None)

    def test_transition_rows_get_turn_and_round_and_nothing_else(self, tmp_path):
        db = _run(tmp_path, [_call("count_tool", "c1", count=1), _prose("done")])
        turn_id = _turn(db)["turn_id"]
        [tr] = _transitions(db)
        assert (tr["turn_id"], tr["round_index"]) == (turn_id, 0)
        # Q6: readers exclude transitions by these columns being NULL.
        for col in ("tool_use_id", "repair_kind", "raw_before_repair", "retry_index", "result_summary"):
            assert tr[col] is None, f"{col} filled on a _loop_transition row"
        assert (tr["repairs"], tr["retries"], tr["parsed_tool_call"]) == (0, 0, None)

    def test_retry_index_counts_prior_failures_of_the_same_tool(self, tmp_path):
        db = _run(tmp_path, [_call("broken_tool", "b1", count=1), _call("broken_tool", "b2", count=2),
                             _call("count_tool", "c1", count=3), _prose("done")])
        calls = _calls(db)
        assert [(c["tool_name"], c["retry_index"]) for c in calls] == [
            ("broken_tool", 0), ("broken_tool", 1), ("count_tool", 0)]
        # A failure's output is already in error_detail; the summary is not a second copy.
        assert calls[0]["error_detail"] == "failure number 1"
        assert calls[0]["result_summary"] is None


# --------------------------------------------------------------------------- #
# Exits: forced_stop_reason, and the rounds that skip the seam
# --------------------------------------------------------------------------- #


class TestExits:

    def test_a_circuit_breaker_run_sets_forced_stop_reason(self, tmp_path):
        turns = [_call("broken_tool", f"b{i}", count=i) for i in range(1, 12)]
        db = _run(tmp_path, turns)
        turn = _turn(db)
        assert turn["forced_stop_reason"] == "circuit_breaker_trip"
        assert turn["terminal_kind"] == "tool_call"
        assert turn["first_prose_round"] is None
        assert turn["rounds"] == len(_responses(db))

    def test_the_iteration_cap_counts_the_calls_it_never_ran(self, tmp_path):
        db = _run(tmp_path, [_call("count_tool", "c1", count=1), _call("count_tool", "c2", count=2)],
                  max_tool_iterations=1, max_tool_iterations_cloud=1)
        turn = _turn(db)
        assert turn["forced_stop_reason"] == "max_iterations_hit"
        assert turn["not_run_calls"] == 1
        assert [c["tool_use_id"] for c in _calls(db)] == ["c1"]

    def test_the_repeat_guard_counts_blocked_calls_as_not_run(self, tmp_path):
        same = [_call("broken_tool", f"b{i}", count=1) for i in range(1, 5)]
        db = _run(tmp_path, same + [_prose("giving up")])
        turn = _turn(db)
        executed = len(_calls(db))
        assert executed == 2, "the repeat guard refuses the third identical failing call"
        assert turn["not_run_calls"] == 2

    def test_an_empty_give_up_writes_both_empty_rounds(self, tmp_path):
        db = _run(tmp_path, [_prose(""), _prose("")])
        assert [r["response_kind"] for r in _responses(db)] == ["empty", "empty"]
        turn = _turn(db)
        assert (turn["forced_stop_reason"], turn["terminal_kind"], turn["rounds"]) == (
            "empty_response", "empty", 2)

    def test_an_empty_retry_then_prose(self, tmp_path):
        db = _run(tmp_path, [_prose(""), _prose("there")])
        assert [r["response_kind"] for r in _responses(db)] == ["empty", "prose"]
        assert _turn(db)["forced_stop_reason"] is None

    def test_a_provider_error_is_an_error_response_and_ends_the_turn(self, tmp_path):
        db = _run(tmp_path, [ConnectionError("backend went away")], raises=ConnectionError)
        [resp] = _responses(db)
        assert resp["response_kind"] == "error"
        [run] = _rows(db, "SELECT outcome FROM subsystem_runs WHERE id = ?", resp["loop_round_id"])
        assert run["outcome"] == "failed"
        turn = _turn(db)
        assert (turn["forced_stop_reason"], turn["terminal_kind"]) == ("provider_error", "error")
        assert turn["ended_at"] is not None

    def test_max_turns_exhausted(self, tmp_path):
        db = _run(tmp_path, [_call("count_tool", f"c{i}", count=i) for i in range(5)],
                  max_turns=2, raises=RuntimeError)
        turn = _turn(db)
        assert (turn["forced_stop_reason"], turn["rounds"]) == ("max_turns_exhausted", 2)

    def test_a_cancelled_turn_is_ended_as_cancelled(self, tmp_path):
        db = tmp_path / "telemetry.db"
        tel = ToolCallTelemetry(db)
        ctx = LoopContext(provider=_Script([_call("count_tool", "c1", count=1), _prose("done")]),
                          model="stub-model", system_prompt="", max_tokens=128,
                          tool_registry=_registry(), telemetry=tel)

        async def stop_after_first_tool() -> None:
            from prometheus.engine.stream_events import ToolExecutionCompleted

            gen = run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=SESSION)
            async for event, _ in gen:
                if isinstance(event, ToolExecutionCompleted):
                    break
            await gen.aclose()

        asyncio.run(stop_after_first_tool())
        tel.close()
        assert _turn(db)["forced_stop_reason"] == "cancelled"


# --------------------------------------------------------------------------- #
# Redaction and ephemeral turns
# --------------------------------------------------------------------------- #


class TestRedaction:

    def test_a_canary_in_prose_is_scrubbed_before_insert(self, tmp_path):
        db = _run(tmp_path, [_prose(f"Your token is {CANARY} keep it safe")])
        [resp] = _responses(db)
        assert resp["prose"] is not None and "keep it safe" in resp["prose"]
        assert CANARY not in resp["prose"]

    def test_a_canary_in_raw_before_repair_is_scrubbed_and_the_kind_recorded(self, tmp_path):
        from prometheus.adapter import ModelAdapter

        pair_capture.configure({"capture_enabled": True, "db_path": str(tmp_path / "training.db")})
        db = _run(tmp_path, [_call("count_tol", "r1", count=3, note=CANARY), _prose("done")],
                  adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT))
        [call] = _calls(db)
        assert call["tool_name"] == "count_tool" and call["repairs"] == 1
        assert call["repair_kind"] == "fuzzy_name"
        raw = json.loads(call["raw_before_repair"])
        assert raw["name"] == "count_tol" and raw["input"]["count"] == 3
        assert CANARY not in call["raw_before_repair"]
        assert CANARY not in (call["result_summary"] or "")
        # T-2's handoff: the pair carries the same kind.
        [pair] = _rows(tmp_path / "training.db", "SELECT pair_source, repair_kind FROM training_pairs")
        assert (pair["pair_source"], pair["repair_kind"]) == ("levenshtein_repair", "fuzzy_name")

    def test_an_ephemeral_turn_keeps_its_rows_but_no_content_and_no_session(self, tmp_path):
        from prometheus.config.ephemeral import set_session_ephemeral

        set_session_ephemeral(SESSION, True)
        try:
            db = _run(tmp_path, [_call("count_tool", "c1", count=1), _prose(f"secret {CANARY}")])
        finally:
            set_session_ephemeral(SESSION, False)
        turn = _turn(db)
        assert SESSION not in turn["turn_id"], "the turn id rides on session-less tool_calls rows"
        assert turn["session_id"] == ""
        assert [r["prose"] for r in _responses(db)] == [None, None]
        assert {r["session_id"] for r in _responses(db)} == {""}
        [call] = _calls(db)
        assert call["turn_id"] == turn["turn_id"] and call["result_summary"] is None


# --------------------------------------------------------------------------- #
# Telemetry off; the writer never blocks the turn
# --------------------------------------------------------------------------- #


class TestWriterIsolation:

    def test_telemetry_disabled_writes_nothing_to_the_new_tables(self, tmp_path, monkeypatch):
        from prometheus.telemetry import writer as writer_mod

        started: list[object] = []
        real_init = writer_mod.TelemetryV2Writer.__init__

        def counting_init(self, *a, **kw):  # noqa: ANN001
            started.append(self)
            real_init(self, *a, **kw)

        monkeypatch.setattr(writer_mod.TelemetryV2Writer, "__init__", counting_init)
        # A real database exists in the process; the turn has no telemetry.
        db = tmp_path / "telemetry.db"
        ToolCallTelemetry(db).close()
        _run(tmp_path, [_call("count_tool", "c1", count=1), _prose("done")], telemetry=False)
        assert started == [], "a v2 writer was started with telemetry disabled"
        for table in ("turns", "responses", "tool_sets"):
            assert _rows(db, f"SELECT * FROM {table}") == []
        assert _rows(db, "SELECT * FROM schema_meta WHERE key = 'telemetry_v2_capture_since'") == []

    def test_nothing_on_the_turn_path_awaits_the_writer(self, tmp_path, monkeypatch):
        from prometheus.telemetry import writer as writer_mod

        release = threading.Event()
        real_write_all = writer_mod.TelemetryV2Writer._write_all

        def stalled(self, conn, writes):  # noqa: ANN001
            release.wait(30)
            real_write_all(self, conn, writes)

        monkeypatch.setattr(writer_mod.TelemetryV2Writer, "_write_all", stalled)
        db = tmp_path / "telemetry.db"
        tel = ToolCallTelemetry(db)
        ctx = LoopContext(provider=_Script([_call("count_tool", "c1", count=1), _prose("done")]),
                          model="stub-model", system_prompt="", max_tokens=128,
                          tool_registry=_registry(), telemetry=tel)

        async def drain() -> None:
            async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")],
                                    session_id=SESSION, surface="beacon"):
                pass

        try:
            # The writer thread is wedged; the turn must still finish promptly.
            asyncio.run(asyncio.wait_for(drain(), timeout=10))
            assert _rows(db, "SELECT * FROM responses") == [], "rows landed while the writer was wedged"
            # The synchronous tool_calls INSERT is unaffected and already carries its turn.
            [call] = _calls(db)
            assert call["turn_id"] is not None
        finally:
            release.set()
        assert tel.v2_writer().flush(10)
        assert len(_responses(db)) == 2
        assert _turn(db)["surface"] == "beacon"
        tel.close()


# --------------------------------------------------------------------------- #
# The turn record's prose policy (spec 4.1a option b)
# --------------------------------------------------------------------------- #


class TestProsePolicy:

    def test_only_the_first_and_last_prose_responses_keep_their_text(self):
        from prometheus.engine.agent_loop import _TurnRecord

        class _Writer:
            def __init__(self) -> None:
                self.rows: list[tuple[str, dict]] = []

            def insert(self, table, row, **kw):  # noqa: ANN001
                self.rows.append((table, dict(row)))
                return True

            def upsert(self, table, row, **kw):  # noqa: ANN001
                self.rows.append((table, dict(row)))
                return True

        w = _Writer()
        rec = _TurnRecord(turn_id="s:t", session_id="s", surface=None, mode="agent",
                          coding_run_id=None, ephemeral=False, writer=w)
        for i, text in enumerate(["first", "second", "third", "last"]):
            rec.response(round_index=i, kind="prose", tool_call_count=0, prose=text,
                         model="m", provider="p", adapter_tier=None, ctx_window=None,
                         tool_set_hash=None, forced_tool_choice=None, loop_round_id=None,
                         prompt_tokens=1, completion_tokens=1)
        rec.end()
        responses = [row for table, row in w.rows if table == "responses"]
        assert [(r["round_index"], r["prose"]) for r in responses] == [
            (0, "first"), (1, None), (2, None), (3, "last")]
        assert [r["prose_chars"] for r in responses] == [5, 6, 5, 4]
        assert [r["ts"] for r in responses] == sorted(r["ts"] for r in responses)


# --------------------------------------------------------------------------- #
# Surfaces and coding mode
# --------------------------------------------------------------------------- #


class TestSurfacesAndCoding:

    def test_run_async_passes_its_surface_through(self, tmp_path):
        from prometheus.engine.agent_loop import AgentLoop

        db = tmp_path / "telemetry.db"
        tel = ToolCallTelemetry(db)
        loop = AgentLoop(provider=_Script([_prose("hi")]), model="stub-model", telemetry=tel)
        asyncio.run(loop.run_async("", "hello", session_id="slack:C1:U1", surface="slack"))
        tel.close()
        turn = _turn(db)
        assert (turn["surface"], turn["session_id"]) == ("slack", "slack:C1:U1")

    @pytest.mark.parametrize("path,surface", [
        ("src/prometheus/gateway/telegram.py", "telegram"),
        ("src/prometheus/gateway/slack.py", "slack"),
        ("src/prometheus/gateway/discord.py", "discord"),
        ("src/prometheus/web/server.py", "rest"),
        ("src/prometheus/web/openai_api.py", "rest"),
        ("src/prometheus/web/ws_server.py", "beacon"),
        ("src/prometheus/__main__.py", "cli"),
        ("src/prometheus/coding/session.py", "coding_mode"),
    ])
    def test_every_surface_names_itself_at_every_loop_call(self, path, surface):
        """`surface` cannot be inferred inside the loop (audit, spec corrections):
        each surface passes it. Every run_async / run_loop call in the file names it."""
        import re

        root = Path(__file__).resolve().parents[1]
        src = (root / path).read_text()
        calls = [m.start() for m in re.finditer(r"\b(run_async|run_loop)\(", src)
                 if not src[max(0, m.start() - 4):m.start()].endswith("def ")]
        assert calls, f"no loop call found in {path}"
        for start in calls:
            # The call's own argument list: up to the matching close paren.
            depth, i = 0, src.index("(", start)
            while True:
                depth += {"(": 1, ")": -1}.get(src[i], 0)
                if depth == 0:
                    break
                i += 1
            args = src[start:i]
            assert f'surface="{surface}"' in args, f"{path}: a loop call without surface={surface!r}"

    def test_coding_mode_is_one_turn_per_episode_under_one_run_id(self, tmp_path):
        import subprocess

        from prometheus.coding.sandbox import ProcessSandbox
        from prometheus.coding.session import CodingSession, CodingTask

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "test_x.py").write_text("def test_x():\n    assert False\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "b"],
                       cwd=repo, check=True)
        db = tmp_path / "telemetry.db"
        tel = ToolCallTelemetry(db)
        session = CodingSession(
            provider=_Script([_prose("done"), _prose("done"), _prose("done")]),
            model="stub-model", sandbox=ProcessSandbox(root=repo),
            task=CodingTask(task_id="t-x", description="d", acceptance_command="python3 -m pytest -q"),
            telemetry=tel, max_rounds=2, coding_run_id="t-x-0badcafe",
        )
        report = asyncio.run(session.run())
        tel.close()
        turns = _rows(db, "SELECT * FROM turns ORDER BY started_at")
        assert len(turns) == report.episodes >= 2
        assert {t["coding_run_id"] for t in turns} == {"t-x-0badcafe"}
        assert {t["surface"] for t in turns} == {"coding_mode"}
        assert {t["session_id"] for t in turns} == {"coding:t-x"}
        assert len({t["turn_id"] for t in turns}) == len(turns)
