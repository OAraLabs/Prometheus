"""Phase 4 (reshaped): conservative dict-wrap unwrapping + honest mode errors.

Spec mapping: the original 'lowering round-trip' property becomes — for every
builtin tool schema, wrap(valid_input) unwraps back to EXACTLY valid_input
(identity round-trip), and anything the transform can't PROVE right (via
validation) is refused with None rather than approximated.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil

import pytest
from pydantic import BaseModel

from prometheus.adapter import ModelAdapter
from prometheus.adapter.unwrap import try_unwrap_arguments
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult
from prometheus.tools.builtin.task_create import (
    TaskCreateTool,
    _mode_error,
    set_honest_mode_errors,
)


class _StatusInput(BaseModel):
    status: str | None = None


class _StatusTool(BaseTool):
    name = "status_tool"
    description = "sessions_list shape"
    input_model = _StatusInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"status={arguments.status}")


class _MultiInput(BaseModel):
    description: str
    prompt: str | None = None
    type: str = "local_bash"


class _MultiTool(BaseTool):
    name = "multi_tool"
    description = "task_create shape"
    input_model = _MultiInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="ok")


class TestUnwrapTransform:

    def test_param_self_wrap_unwrapped(self):
        # the live sessions_list failure: {"status": {"status": None}}
        out = try_unwrap_arguments(_StatusTool(), {"status": {"status": "failed"}})
        assert out is not None
        unwrapped, log_notes = out
        assert unwrapped == {"status": "failed"}
        assert "self-keyed" in log_notes[0]

    def test_top_level_promote(self):
        # the live task_create failure: {"prompt": {…actual args…}}
        out = try_unwrap_arguments(
            _MultiTool(),
            {"prompt": {"description": "enrich CSV", "prompt": "do rows 10-60"}},
        )
        assert out is not None
        unwrapped, log_notes = out
        assert unwrapped == {"description": "enrich CSV", "prompt": "do rows 10-60"}
        assert "promoted" in log_notes[0]

    def test_never_touches_valid_input(self):
        assert try_unwrap_arguments(_StatusTool(), {"status": "ok"}) is None

    def test_refuses_unfixable_garbage(self):
        # wrapped AND inner is still invalid → refuse, never approximate
        assert try_unwrap_arguments(
            _MultiTool(), {"description": {"description": 123}}
        ) is None or try_unwrap_arguments(
            _MultiTool(), {"description": {"description": 123}}
        )[0]["description"] == 123 and False  # must be None
        assert try_unwrap_arguments(_MultiTool(), {"nope": {"x": 1}}) is None
        assert try_unwrap_arguments(_MultiTool(), "not a dict") is None

    def test_round_trip_identity_across_all_builtin_schemas(self):
        """wrap(valid) → unwrap == identity, for every builtin input model
        with at least one required string param."""
        import prometheus.tools.builtin as B

        checked = 0
        for mod_info in pkgutil.iter_modules(B.__path__):
            try:
                mod = importlib.import_module(
                    f"prometheus.tools.builtin.{mod_info.name}"
                )
                members = inspect.getmembers(mod, inspect.isclass)
            except Exception:
                continue
            for name, obj in members:
                try:
                    if not (
                        issubclass(obj, BaseModel)
                        and obj is not BaseModel
                        and name.endswith(("Input", "ToolInput"))
                        and getattr(obj, "__module__", "") == mod.__name__
                    ):
                        continue
                    schema = obj.model_json_schema()
                    props = schema.get("properties", {})
                    required = schema.get("required", [])
                    # build a minimal valid input of string params
                    valid = {}
                    for r in required:
                        if props.get(r, {}).get("type") == "string":
                            valid[r] = "x"
                    if not valid or len(valid) < len(required):
                        continue  # can't synthesize — skip honestly
                    try:
                        obj.model_validate(valid)
                    except Exception:
                        continue  # Literal/enum params — can't synthesize "x"

                    class _T(BaseTool):
                        name = "rt"
                        description = "round trip"
                        input_model = obj

                        async def execute(self, arguments, context):  # noqa: ANN001
                            return ToolResult(output="")

                    tool = _T()
                    # self-wrap EVERY param, then unwrap
                    wrapped = {k: {k: v} for k, v in valid.items()}
                    out = try_unwrap_arguments(tool, wrapped)
                    if out is None:
                        # acceptable only if the wrapped form already validates
                        # (loose schemas with dict-typed params)
                        try:
                            obj.model_validate(wrapped)
                        except Exception:
                            pytest.fail(f"{name}: wrap not unwrapped and not valid")
                        continue
                    assert out[0] == valid, f"{name}: round-trip not identity"
                    checked += 1
                except Exception as exc:  # pragma: no cover
                    pytest.fail(f"{name}: {exc}")
        assert checked >= 10, f"only {checked} schemas exercised"


class TestLoopIntegration:

    def _ctx(self, unwrap_on: bool):
        reg = ToolRegistry()
        reg.register(_StatusTool())
        adapter = ModelAdapter(
            tier=ModelAdapter.TIER_LIGHT,
            unwrap_tools=frozenset({"status_tool"}) if unwrap_on else frozenset(),
        )
        return LoopContext(
            provider=None, model="m", system_prompt="", max_tokens=64,
            tool_registry=reg, adapter=adapter,
        )

    def test_gated_off_is_inert(self):
        block = asyncio.run(_execute_tool_call(
            self._ctx(False), "status_tool", "t1", {"status": {"status": "failed"}}
        ))
        assert block.is_error
        assert "Invalid input" in block.content

    def test_gated_on_unwraps_and_executes(self):
        block = asyncio.run(_execute_tool_call(
            self._ctx(True), "status_tool", "t1", {"status": {"status": "failed"}}
        ))
        assert not block.is_error
        assert "status=failed" in block.content

    def test_gated_on_still_fails_unfixable(self):
        block = asyncio.run(_execute_tool_call(
            self._ctx(True), "status_tool", "t1", {"status": {"other": 1}}
        ))
        assert block.is_error


class TestHonestModeErrors:

    def teardown_method(self):
        set_honest_mode_errors(False)

    def _args(self, **kw):
        from prometheus.tools.builtin.task_create import TaskCreateToolInput
        return TaskCreateToolInput(description="d", **kw)

    def test_default_message_unchanged(self):
        msg = _mode_error(self._args(prompt="p"), "command", "local_bash")
        assert msg == "'command' is required for local_bash tasks"

    def test_honest_message_names_the_actual_mistake(self):
        set_honest_mode_errors(True)
        msg = _mode_error(self._args(prompt="p"), "command", "local_bash")
        assert "type='local_bash' (the default)" in msg
        assert "You supplied 'prompt'" in msg
        assert "type='local_agent'" in msg
        assert "Valid types:" in msg
        assert '"type": "local_agent"' in msg  # example call present

    def test_honest_message_without_prompt_skips_the_hint(self):
        set_honest_mode_errors(True)
        msg = _mode_error(self._args(), "command", "local_bash")
        assert "You supplied" not in msg
        assert "Valid types:" in msg

    def test_tool_execute_uses_gate(self):
        async def run():
            tool = TaskCreateTool()
            from prometheus.tools.base import ToolExecutionContext
            from pathlib import Path
            return await tool.execute(
                self._args(prompt="p"),
                ToolExecutionContext(cwd=Path("/tmp"), metadata={}),
            )
        set_honest_mode_errors(True)
        result = asyncio.run(run())
        assert result.is_error
        assert "You supplied 'prompt'" in result.output


class TestManifestVariables:

    def test_unwrap_manifest_loads(self):
        from prometheus.gym.manifest import load_manifest
        m = load_manifest("gym/experiments/s1-exp3-unwrap.yaml")
        assert m.variable_name == "adapter_unwrap"
        assert "task_create" in m.variable_payload["tools"]

    def test_honesty_manifest_loads(self):
        from prometheus.gym.manifest import load_manifest
        m = load_manifest("gym/experiments/s1-exp2-error-honesty.yaml")
        assert m.variable_name == "tool_error_honesty"
