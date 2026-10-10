"""A null for an optional parameter is "not given", at every strictness.

Observed 2026-10-05 on the mini (v0.9.6, Ollama qwen2.5:7b-instruct, tier
light): the model called read_file twice with
``{"path": "README.md", "limit": null, "offset": 0}``. Both calls failed as
``input_validation`` ("limit: Input should be a valid integer"), and then the
model invented the README's contents in its reply.

The adapter now drops a null that an optional parameter's schema refuses, so
the default applies, and records the repair as ``null_drop``. A null for a
REQUIRED parameter is still refused at every level. At MEDIUM/STRICT the type
coercion used to turn it into the string "None" (``str(None)``) and run it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from prometheus.adapter import ModelAdapter
from prometheus.adapter.validator import Strictness, ToolCallValidator, repair_kinds
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.stream_events import ToolExecutionCompleted
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning import pair_capture
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from prometheus.tools.builtin.file_read import FileReadTool, FileReadToolInput

# The call qwen2.5:7b made, verbatim.
OBSERVED = {"path": "README.md", "limit": None, "offset": 0}


class _FlagIn(BaseModel):
    target: str
    loud: bool = True                      # a default a null used to flip to False
    retries: int | None = 3                # Optional: a null here is a value


class _FlagTool(BaseTool):
    name = "flag_tool"
    description = "flags"
    input_model = _FlagIn

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"{arguments.target} {arguments.loud} {arguments.retries}")


class _OddIn(BaseModel):
    depth: int = 1

    @field_validator("depth", mode="before")
    @classmethod
    def _refuse_oddly(cls, value):  # noqa: ANN001
        if value is None:
            raise TypeError("not a pydantic error")
        return value


class _OddTool(_FlagTool):
    name = "odd_tool"
    input_model = _OddIn


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(FileReadTool())
    reg.register(_FlagTool())
    reg.register(_OddTool())
    return reg


def _adapter(strictness: str) -> ModelAdapter:
    """NONE is what tier light runs (the demo's tier); MEDIUM is tier full's."""
    if strictness == "NONE":
        return ModelAdapter(tier=ModelAdapter.TIER_LIGHT)
    adapter = ModelAdapter(tier=ModelAdapter.TIER_FULL, strictness=strictness)
    assert adapter.validator.strictness == Strictness(strictness)
    return adapter


LEVELS = ("NONE", "MEDIUM", "STRICT")


# --------------------------------------------------------------------------- #
# The adapter, at every strictness
# --------------------------------------------------------------------------- #


class TestOptionalNullTakesTheDefault:

    @pytest.mark.parametrize("level", LEVELS)
    def test_read_file_limit_null_reads_with_the_default(self, level):
        name, inp, log = _adapter(level).validate_and_repair(
            "read_file", dict(OBSERVED), _registry())
        assert (name, inp) == ("read_file", {"path": "README.md", "offset": 0})
        assert FileReadToolInput.model_validate(inp).limit == 200
        assert log == ["dropped null limit: optional, its default applies"]
        assert repair_kinds(log) == ["null_drop"]

    @pytest.mark.parametrize("level", LEVELS)
    def test_each_null_is_one_repair(self, level):
        _, inp, log = _adapter(level).validate_and_repair(
            "read_file", {"path": "a.txt", "limit": None, "offset": None}, _registry())
        assert inp == {"path": "a.txt"}
        assert repair_kinds(log) == ["null_drop", "null_drop"]

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_bool_default_is_kept_not_flipped(self, level):
        _, inp, log = _adapter(level).validate_and_repair(
            "flag_tool", {"target": "x", "loud": None}, _registry())
        assert _FlagIn.model_validate(inp).loud is True
        assert repair_kinds(log) == ["null_drop"]

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_null_an_optional_field_accepts_is_left_alone(self, level):
        _, inp, log = _adapter(level).validate_and_repair(
            "flag_tool", {"target": "x", "retries": None}, _registry())
        assert inp == {"target": "x", "retries": None}
        assert _FlagIn.model_validate(inp).retries is None
        assert log == []

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_call_without_nulls_is_not_a_repair(self, level):
        _, inp, log = _adapter(level).validate_and_repair(
            "read_file", {"path": "README.md", "limit": 10}, _registry())
        assert (inp, log) == ({"path": "README.md", "limit": 10}, [])


class TestRequiredNullIsStillRefused:

    def test_none_passes_it_through_to_the_loops_own_check(self):
        # NONE checks invariants only; the loop's pydantic check refuses it.
        name, inp, log = _adapter("NONE").validate_and_repair(
            "read_file", {"path": None}, _registry())
        assert (inp, log) == ({"path": None}, [])
        with pytest.raises(ValidationError):
            FileReadToolInput.model_validate(inp)

    @pytest.mark.parametrize("level", ("MEDIUM", "STRICT"))
    def test_medium_and_strict_refuse_it_never_the_path_None(self, level):
        with pytest.raises(ValueError, match="could not be repaired"):
            _adapter(level).validate_and_repair("read_file", {"path": None}, _registry())

    @pytest.mark.parametrize("level", ("MEDIUM", "STRICT"))
    def test_an_optional_null_beside_it_does_not_rescue_it(self, level):
        with pytest.raises(ValueError):
            _adapter(level).validate_and_repair(
                "read_file", {"path": None, "limit": None}, _registry())

    @pytest.mark.parametrize("level", LEVELS)
    def test_repair_never_coerces_a_required_null(self, level):
        r = ToolCallValidator(level).repair("read_file", {"path": None}, "", _registry())
        assert not r.repaired
        assert r.tool_input == {"path": None}
        assert r.repairs_made == []


class TestInsideRepair:
    """The pre-pass needs a known tool and a dict; repair runs the step again
    once it has them."""

    def test_after_a_fuzzy_name_at_none(self):
        name, inp, log = _adapter("NONE").validate_and_repair(
            "read_fil", dict(OBSERVED), _registry())
        assert (name, inp) == ("read_file", {"path": "README.md", "offset": 0})
        assert repair_kinds(log) == ["fuzzy_name", "null_drop"]

    def test_in_the_order_applied_at_medium(self):
        r = ToolCallValidator(Strictness.MEDIUM).repair(
            "read_fil", '{"path": 42, "limit": null}', "", _registry())
        assert r.repaired and r.tool_input == {"path": "42"}
        assert r.repair_kinds == ["fuzzy_name", "json_extract", "null_drop", "type_coerce"]

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_tool_validator_raising_something_else_is_left_to_the_loop(self, level):
        v = ToolCallValidator(level)
        assert v.drop_optional_nulls("odd_tool", {"depth": None}, _registry()) == (
            {"depth": None}, [])

    def test_unknown_tool_or_non_dict_is_a_no_op(self):
        v = ToolCallValidator(Strictness.NONE)
        assert v.drop_optional_nulls("nope", {"x": None}, _registry()) == ({"x": None}, [])
        assert v.drop_optional_nulls("read_file", '{"limit": null}', _registry()) == (
            '{"limit": null}', [])


# --------------------------------------------------------------------------- #
# Through the real loop: the demo's call reads the file, and the repair counts
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _capture_off_around_each_test():
    pair_capture.configure({"capture_enabled": False})
    yield
    pair_capture.configure({"capture_enabled": False})


class _Script(ModelProvider):
    def __init__(self, turns: list) -> None:
        self.turns = list(turns)
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        if self.calls < len(self.turns):
            item = self.turns[self.calls]
        else:
            item = ConversationMessage(role="assistant", content=[TextBlock(text="done")])
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=item, usage=UsageSnapshot(input_tokens=11, output_tokens=7),
            stop_reason="stop")


def _call(call_id: str, inp: dict) -> ConversationMessage:
    return ConversationMessage(
        role="assistant", content=[ToolUseBlock(id=call_id, name="read_file", input=inp)])


def _run(tmp_path: Path, inp: dict) -> tuple[Path, list[ToolExecutionCompleted]]:
    (tmp_path / "README.md").write_text("Prometheus parity readme\nsecond line\n")
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=_Script([_call("r1", inp)]), model="qwen2.5:7b-instruct",
                      system_prompt="", max_tokens=128, tool_registry=_registry(),
                      telemetry=tel, cwd=tmp_path,
                      adapter=ModelAdapter(tier=ModelAdapter.TIER_LIGHT))
    done: list[ToolExecutionCompleted] = []

    async def drain() -> None:
        async for event, _ in run_loop(ctx, [ConversationMessage.from_user_text("go")],
                                       session_id="cli:demo"):
            if isinstance(event, ToolExecutionCompleted):
                done.append(event)

    asyncio.run(drain())
    tel.close()
    return db, done


def _calls(db: Path) -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        return con.execute("SELECT * FROM tool_calls WHERE tool_name != '_loop_transition'"
                           " ORDER BY rowid").fetchall()
    finally:
        con.close()


class TestTheDemoThroughTheLoop:

    def test_the_observed_call_reads_the_readme(self, tmp_path):
        pair_capture.configure({"capture_enabled": True, "db_path": str(tmp_path / "training.db")})
        db, [done] = _run(tmp_path, dict(OBSERVED))
        assert not done.is_error
        assert "Prometheus parity readme" in done.output
        [row] = _calls(db)
        assert row["success"] == 1 and row["error_type"] in (None, "")
        assert (row["repairs"], row["repair_kind"]) == (1, "null_drop")
        assert json.loads(row["raw_before_repair"])["input"] == OBSERVED
        # The as-emitted call and its repair are a training pair, like any repair.
        con = sqlite3.connect(tmp_path / "training.db")
        try:
            pairs = con.execute("SELECT pair_source, repair_kind FROM training_pairs").fetchall()
        finally:
            con.close()
        assert pairs == [("schema_repair", "null_drop")]

    def test_a_required_null_still_fails_as_input_validation(self, tmp_path):
        db, [done] = _run(tmp_path, {"path": None, "limit": None})
        assert done.is_error and "path" in done.output
        [row] = _calls(db)
        assert (row["success"], row["error_type"]) == (0, "input_validation")
        # The optional null was dropped and recorded; the required one was not.
        assert row["repair_kind"] == "null_drop"
        assert json.loads(row["raw_before_repair"])["input"] == {"path": None, "limit": None}
