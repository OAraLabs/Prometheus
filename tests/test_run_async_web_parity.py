"""``AgentLoop.run_async`` must not hide behaviour from the web bridge.

``tests/test_web_bridge_loop_parity.py`` guards the *field-level* half of the
two-loop asymmetry: every ``LoopContext`` field ``daemon.py`` configures for
the ``AgentLoop(...)`` path must also be configured for the pre-built web
``LoopContext``. That guard is structurally blind to a whole second class of
drift, one level up:

    ``AgentLoop.run_async`` does work AROUND ``run_loop`` — before the loop,
    inside the ``async for`` body, and after it returns. The web bridge
    (``web/ws_server.py:_run_agent``) calls ``run_loop`` DIRECTLY. Anything
    that lives in the wrapper rather than in the loop reaches telegram / CLI /
    bakeoff and no web / Beacon / Bridge turn, and no amount of comparing
    LoopContext kwargs will ever notice.

``PeriodicNudge`` was exactly that. ``daemon.py`` passed ``nudge=nudge`` to
``AgentLoop``, which stored it as ``self._nudge`` and injected it from the
``async for`` body — so with ``learning.nudge_enabled: true`` sitting in the
live config, the self-reflection prompt fired on telegram and never once on
Beacon. It was invisible to the field guard because ``nudge`` was not a
``LoopContext`` field at all.

The fix moved the injection INTO ``run_loop`` behind ``LoopContext.nudge``,
which both shares it with the web path and folds it under the existing
field-level guard. The test below stops the next one: it asserts that
everything ``run_async`` reads off ``self`` is either forwarded into the
``LoopContext`` it builds (→ shared with the web bridge) or explicitly
allowlisted with a reason.

Known blind spot, stated so nobody reads a pass as a proof of parity: the
guard keys on ``self.X`` reads. Behaviour added to ``run_async`` that routes
through neither an instance attribute nor the LoopContext — a hardcoded call,
a module global, a closure — is still invisible to it. It catches the shape
all six known instances took, not every conceivable one.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning.nudge import PeriodicNudge
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

AGENT_LOOP = (
    Path(__file__).resolve().parents[1]
    / "src" / "prometheus" / "engine" / "agent_loop.py"
)

# Attributes ``run_async`` reads off ``self`` that are deliberately NOT shared
# with the web bridge through the LoopContext. Each needs a reason, because
# each one is a feature Beacon does not get.
RUN_ASYNC_ONLY = {
    # Local accumulator, not config: run_async collects a {tool_name, result,
    # is_error} trace off the event stream to feed _post_task_hooks below.
    # ws_server._run_agent builds its own per-event view (the `progress` dict
    # + WS frames) and keeps no trace.
    "_tool_trace",
    # THE REMAINING GAP, deliberately left open — see the PR that added this
    # file. daemon.py registers TWO post-task hooks on the AgentLoop:
    # SkillCreator.maybe_create (learning.auto_skill_creation: true) and
    # SkillRefiner.maybe_refine_recent (learning.skill_refinement_enabled:
    # true). Both are live-on in config and NEITHER fires on a web/Beacon
    # turn. Unlike the nudge this is not a one-line share: each hook is an
    # extra LLM call per turn, so wiring it to the surface that does the most
    # agentic work is a cost/product decision, not a parity bug to fix by
    # reflex. test_the_post_task_hooks_gap_is_real below is the evidence.
    "_post_task_hooks",
}


# ---------------------------------------------------------------------------
# Structural guard
# ---------------------------------------------------------------------------

def _run_async_node() -> ast.FunctionDef:
    tree = ast.parse(AGENT_LOOP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AgentLoop":
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "run_async"
                ):
                    return item
    raise AssertionError("AgentLoop.run_async not found in agent_loop.py")


def _self_attrs_read(fn: ast.AST) -> set[str]:
    """Every ``self.X`` READ inside ``fn``.

    Load context only, so a pure write (``self._tool_trace = []``) does not
    count — but ``self._tool_trace.append(...)`` does, which is the point:
    an attribute run_async *consumes* is behaviour, and behaviour is what the
    web bridge is missing.
    """
    return {
        node.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }


def _loop_context_kwargs(fn: ast.AST) -> dict[str, str]:
    """``{loop_context_field: self_attr}`` for the LoopContext ``fn`` builds."""
    out: dict[str, str] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", None) != "LoopContext":
            continue
        for kw in node.keywords:
            if (
                kw.arg
                and isinstance(kw.value, ast.Attribute)
                and isinstance(kw.value.value, ast.Name)
                and kw.value.value.id == "self"
            ):
                out[kw.arg] = kw.value.attr
    return out


def test_run_async_still_builds_a_loop_context():
    """If run_async stops building its own context this guard is moot and
    should be deleted rather than left passing vacuously."""
    assert _loop_context_kwargs(_run_async_node()), (
        "AgentLoop.run_async no longer constructs a LoopContext from self.*"
    )


def test_nothing_reaches_run_loop_only_through_the_agentloop_wrapper():
    """THE GUARD. Every ``self.X`` run_async reads must be forwarded into the
    LoopContext (→ the web bridge's direct run_loop call gets it too) or be
    allowlisted with a documented reason."""
    fn = _run_async_node()
    forwarded = set(_loop_context_kwargs(fn).values())
    drift = _self_attrs_read(fn) - forwarded - RUN_ASYNC_ONLY

    assert not drift, (
        f"AgentLoop.run_async reads {sorted(drift)} but does not forward "
        f"them into its LoopContext. web/ws_server.py:_run_agent calls "
        f"run_loop DIRECTLY with a pre-built context, so anything reachable "
        f"only through the AgentLoop wrapper silently does nothing on every "
        f"web / Beacon / Bridge turn — and the field-level parity guard in "
        f"test_web_bridge_loop_parity.py cannot see it. Move the behaviour "
        f"into run_loop behind a LoopContext field, or add the attribute to "
        f"RUN_ASYNC_ONLY with a reason."
    )


def test_the_allowlist_is_still_honest():
    """A stale entry would mask real drift."""
    fn = _run_async_node()
    read = _self_attrs_read(fn)
    stale = RUN_ASYNC_ONLY - read
    assert not stale, (
        f"{sorted(stale)} are in RUN_ASYNC_ONLY but run_async no longer "
        f"reads them — remove them from the allowlist"
    )


def test_nudge_is_a_loop_context_field():
    """The nudge moved from ``AgentLoop._nudge`` to ``LoopContext.nudge``
    specifically so the daemon's two constructions can be compared. Losing
    the field would silently re-open the hole AND blind the field guard."""
    fields = {f.name: f for f in dataclasses.fields(LoopContext)}
    assert "nudge" in fields, (
        "LoopContext.nudge is gone — if the nudge went back to being an "
        "AgentLoop-only attribute, web/Beacon turns lost it again"
    )
    assert fields["nudge"].default is None, (
        "the default must stay None so benchmarks/evals/gym keep byte-"
        "identical prompts"
    )


# ---------------------------------------------------------------------------
# Behavioural half: the nudge fires on a BARE run_loop (i.e. the web path)
# ---------------------------------------------------------------------------

class _EchoInput(BaseModel):
    text: str = "hello"


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    input_model = _EchoInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=arguments.text)

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True


class _RecordingProvider(ModelProvider):
    """Replays scripted rounds; records every request it was handed."""

    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self._n = 0
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        events = self._responses[self._n % len(self._responses)]
        self._n += 1
        for event in events:
            yield event

    @property
    def nudged_calls(self) -> list[int]:
        return [
            i for i, r in enumerate(self.requests)
            if "[system-internal]" in (r.system_prompt or "")
        ]


def _tool_round(tool_id: str) -> list:
    msg = ConversationMessage(
        role="assistant",
        content=[ToolUseBlock(id=tool_id, name="echo", input={"text": "hi"})],
    )
    return [
        ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(), stop_reason="tool_calls"
        )
    ]


def _text_round(text: str) -> list:
    msg = ConversationMessage(role="assistant", content=[TextBlock(text=text)])
    return [
        ApiTextDeltaEvent(text=text),
        ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(), stop_reason="stop"
        ),
    ]


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    return reg


def _drive_run_loop(context: LoopContext, messages: list[ConversationMessage]):
    """Exactly what web/ws_server.py:_run_agent does — bare run_loop, no
    AgentLoop wrapper anywhere in the call stack."""

    async def _run():
        async for _event, _usage in run_loop(context, messages):
            pass

    asyncio.run(_run())


def test_the_web_path_gets_the_nudge():
    """The bug: a Beacon turn drives run_loop directly and used to be unable
    to see the nudge at all, whatever learning.nudge_enabled said."""
    provider = _RecordingProvider([
        _tool_round("t1"), _tool_round("t2"), _text_round("done"),
    ])
    context = LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE",
        max_tokens=256,
        tool_registry=_registry(),
        session_id="web",
        nudge=PeriodicNudge(interval=1, enabled=True),
    )
    _drive_run_loop(context, [ConversationMessage.from_user_text("go")])

    assert provider.nudged_calls == [1, 2], (
        "rounds 1 and 2 each complete and arm a nudge for the next call; got "
        f"nudges on calls {provider.nudged_calls} of {len(provider.requests)}"
    )


def test_the_web_path_nudge_never_enters_the_transcript():
    """It must not be a ``messages`` turn. ws_server calls
    ``session.persist_loop_result``, which writes whatever run_loop appended
    to LCM — and ``GET /api/sessions/{id}/messages`` hands it back to Beacon.
    A ``from_user_text`` nudge would render as a chat bubble nobody typed."""
    provider = _RecordingProvider([_tool_round("t1"), _text_round("done")])
    messages = [ConversationMessage.from_user_text("go")]
    context = LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE",
        max_tokens=256,
        tool_registry=_registry(),
        session_id="web",
        nudge=PeriodicNudge(interval=1, enabled=True),
    )
    _drive_run_loop(context, messages)

    assert provider.nudged_calls, "precondition: the nudge must have fired"
    offenders = [m for m in messages if "[system-internal]" in m.text]
    assert not offenders, (
        f"{len(offenders)} nudge turn(s) reached the message list — these "
        f"get persisted to LCM by persist_loop_result and rendered in Beacon"
    )


def test_nudge_off_leaves_the_prompt_byte_identical():
    """Disabled / unwired must cost nothing: benchmarks, evals, coding mode
    and the gym all build a LoopContext without a nudge."""
    for nudge in (None, PeriodicNudge(interval=1, enabled=False)):
        provider = _RecordingProvider([_tool_round("t1"), _text_round("done")])
        context = LoopContext(
            provider=provider,
            model="test",
            system_prompt="BASE",
            max_tokens=256,
            tool_registry=_registry(),
            nudge=nudge,
        )
        _drive_run_loop(context, [ConversationMessage.from_user_text("go")])
        assert [r.system_prompt for r in provider.requests] == ["BASE", "BASE"], (
            f"nudge={nudge!r} changed the prompt"
        )


def test_a_broken_nudge_cannot_break_the_turn():
    """Fail-open, same posture as the recall / steer / verifier channels."""

    class _Exploding:
        def maybe_inject(self, turn_count):  # noqa: ANN001
            raise RuntimeError("boom")

    provider = _RecordingProvider([_tool_round("t1"), _text_round("done")])
    context = LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE",
        max_tokens=256,
        tool_registry=_registry(),
        nudge=_Exploding(),
    )
    _drive_run_loop(context, [ConversationMessage.from_user_text("go")])
    assert len(provider.requests) == 2
    assert provider.nudged_calls == []


# ---------------------------------------------------------------------------
# The gap that is NOT closed here — evidence for the RUN_ASYNC_ONLY entry
# ---------------------------------------------------------------------------

def test_the_post_task_hooks_gap_is_real():
    """SkillCreator/SkillRefiner run ONLY on the AgentLoop path.

    ``daemon.py`` registers both via ``agent_loop.add_post_task_hook`` and
    ``run_async`` fires them after the loop returns, off ``self._tool_trace``.
    A web/Beacon turn goes nowhere near either. Documented, not fixed: each
    hook is an extra LLM call per turn. This test is the receipt — if the
    hooks ever do become reachable from ``run_loop``, it fails and points at
    the RUN_ASYNC_ONLY entry to retire.
    """
    fn = _run_async_node()
    read = _self_attrs_read(fn)
    assert "_post_task_hooks" in read and "_tool_trace" in read, (
        "post-task hooks moved — re-check whether the web path now gets them"
    )

    ws = (
        Path(__file__).resolve().parents[1]
        / "src" / "prometheus" / "web" / "ws_server.py"
    ).read_text(encoding="utf-8")
    assert "post_task_hook" not in ws, (
        "ws_server now mentions post-task hooks — if the web bridge fires "
        "them, drop _post_task_hooks/_tool_trace from RUN_ASYNC_ONLY"
    )

    # And nothing in run_loop can reach them either: they are not on the
    # context the web bridge hands over.
    assert "post_task_hooks" not in {f.name for f in dataclasses.fields(LoopContext)}


@pytest.mark.parametrize("field_name", ["nudge", "compactor", "memory_recall"])
def test_the_shared_channels_stay_optional(field_name):
    """Every behaviour shared through LoopContext defaults to off, so the
    non-daemon callers (gym, evals, coding mode, benchmarks) are untouched."""
    fields = {f.name: f for f in dataclasses.fields(LoopContext)}
    assert fields[field_name].default is None
