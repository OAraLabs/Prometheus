"""Deterministic scoring over a finished run's transcript. No LLM judging.

The transcript is the loop's final ``messages`` list plus run metadata the
runner extracts (drops, breaker state). Every predicate present on the task
must pass (AND).

Series-2 dual scoring
---------------------
Series 1 scored a single outcome and structurally could not credit the
adapter: ``messages`` keep the model's RAW emission ToolUseBlock, while the
adapter repairs/unwraps the input *locally* in ``_execute_tool_call`` before
the tool runs (closeout §3 — exp3's unwrap repairs executed successfully
while the predicate scored the still-wrapped emission). We now score two
independent views over the same run:

- ``emission`` — the model's RAW tool call (what it actually produced). A
  call that needed a repair did NOT satisfy *as emitted*, so it does not
  count as a successful call in this view.
- ``execution`` — the call that ACTUALLY EXECUTED (post adapter repair +
  unwrap). Uses the real ``is_error`` result.

The gap ``execution_pass − emission_pass`` is the adapter's measured value.
The executed call is supplied per ``tool_use_id`` by the gym runner via
``LoopContext.tool_call_observer``; without it (every non-gym caller,
existing tests) the executed view falls back to the raw call and the two
views coincide — i.e. exactly the pre-series-2 behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

EMISSION = "emission"
EXECUTION = "execution"


@dataclass
class ToolEvent:
    name: str                  # raw-emitted tool name
    input: dict[str, Any]      # raw-emitted input
    is_error: bool | None      # None = no result block found (shouldn't happen)
    result_content: str = ""
    # Post-adapter (executed) call. Default to the raw call → when no observer
    # data is supplied the two views are identical (pre-series-2 behavior).
    exec_name: str | None = None
    exec_input: dict[str, Any] | None = None
    repaired: bool = False     # raw call differed from what executed

    def __post_init__(self) -> None:
        if self.exec_name is None:
            self.exec_name = self.name
        if self.exec_input is None:
            self.exec_input = self.input

    def view_name(self, view: str) -> str:
        return self.name if view == EMISSION else (self.exec_name or self.name)

    def view_input(self, view: str) -> dict[str, Any]:
        if view == EMISSION:
            return self.input
        return self.exec_input if self.exec_input is not None else self.input

    def view_ok(self, view: str) -> bool:
        """Did this call succeed *for this view*?

        execution: the real result was non-error.
        emission: the real result was non-error AND the call ran as emitted
        (a repaired call did not satisfy as the model wrote it)."""
        if self.is_error is not False:
            return False
        if view == EMISSION:
            return not self.repaired
        return True


@dataclass
class RunTranscript:
    messages: list[ConversationMessage]
    dropped_malformed: int = 0          # provider-dropped empty envelopes
    orchestrator_feedback: int = 0      # loop's malformed-retry injections
    tool_events: list[ToolEvent] = field(default_factory=list)
    final_text: str = ""
    breaker_tripped: bool = False

    @classmethod
    def from_messages(
        cls,
        messages: list[ConversationMessage],
        observed: dict[str, dict[str, Any]] | None = None,
    ) -> "RunTranscript":
        """Build a transcript. ``observed`` maps tool_use_id → {"raw": {...},
        "executed": {...}} (each {"name","input"}), supplied by the gym
        runner's ``tool_call_observer``. When absent, the executed view
        mirrors the raw emission."""
        observed = observed or {}
        events: list[ToolEvent] = []
        by_id: dict[str, ToolEvent] = {}
        feedback = 0
        final_text = ""
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        ev = ToolEvent(name=block.name, input=dict(block.input), is_error=None)
                        obs = observed.get(block.id)
                        if obs:
                            executed = obs.get("executed") or {}
                            ev.exec_name = executed.get("name", block.name)
                            ev.exec_input = dict(executed.get("input", block.input))
                            raw = obs.get("raw") or {"name": block.name, "input": block.input}
                            ev.repaired = (
                                ev.exec_name != raw.get("name")
                                or ev.exec_input != raw.get("input")
                            )
                        events.append(ev)
                        by_id[block.id] = ev
                    elif isinstance(block, TextBlock) and block.text.strip():
                        final_text = block.text
            else:
                if msg.provenance == "orchestrator":
                    feedback += 1
                for block in msg.content:
                    if isinstance(block, ToolResultBlock) and block.tool_use_id in by_id:
                        by_id[block.tool_use_id].is_error = block.is_error
                        by_id[block.tool_use_id].result_content = block.content

        breaker = "circuit breaker tripped" in final_text.lower()
        return cls(
            messages=messages,
            orchestrator_feedback=feedback,
            tool_events=events,
            final_text=final_text,
            breaker_tripped=breaker,
        )


def _successful_calls(t: RunTranscript, tool: str, view: str) -> list[ToolEvent]:
    return [e for e in t.tool_events if e.view_name(view) == tool and e.view_ok(view)]


def _looks_like_json_blob(value: Any) -> bool:
    if not isinstance(value, str):
        return True  # not even a string — worse than a blob
    s = value.strip()
    if not s.startswith("{"):
        return False
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return s.startswith('{"')  # truncated serialization still counts


def score(
    task_score: dict[str, Any],
    t: RunTranscript,
    workspace: Path,
    view: str = EXECUTION,
) -> tuple[bool, list[str]]:
    """Apply every predicate under one view; return (passed, failure_reasons).

    ``view`` is ``execution`` (post-adapter, the default and the back-compat
    behavior) or ``emission`` (the model's raw call). Arg-shape predicates and
    successful-call matching use the view; side-effect predicates (file, text,
    breaker, malformed, graceful) reflect the actual run in both views — but
    ``expect_tool`` still carries the distinction, since a repaired call is not
    a successful call in the emission view."""
    reasons: list[str] = []
    expect_tool = task_score.get("expect_tool")
    ok_calls = _successful_calls(t, expect_tool, view) if expect_tool else []

    if expect_tool:
        if not ok_calls:
            attempted = [e.view_name(view) for e in t.tool_events]
            reasons.append(
                f"no successful {expect_tool!r} call (tools attempted: {attempted or 'none'})"
            )

    if "expect_tool_args_string" in task_score and ok_calls:
        for param in task_score["expect_tool_args_string"]:
            for call in ok_calls:
                vin = call.view_input(view)
                if param in vin and not isinstance(vin[param], str):
                    reasons.append(
                        f"{expect_tool}.{param} should be a string, got "
                        f"{type(vin[param]).__name__}: {vin[param]!r:.120}"
                    )

    if "expect_tool_args_present" in task_score and ok_calls:
        # Require the param to be actually supplied (key present, value not
        # None/empty) on at least one successful call. Lets us assert a
        # local_bash task supplied `command` WITHOUT pinning type=local_bash
        # (local_bash is the default — demanding the explicit key was the s1
        # predicate artifact, closeout follow-up #14). Presence-only; the
        # value is type-checked separately by expect_tool_args_string.
        for param in task_score["expect_tool_args_present"]:
            if not any(
                param in c.view_input(view)
                and c.view_input(view).get(param) not in (None, "")
                for c in ok_calls
            ):
                reasons.append(
                    f"no successful {expect_tool} call supplied {param!r}"
                )

    if "expect_tool_args_require" in task_score:
        required = task_score["expect_tool_args_require"]
        if ok_calls and not any(
            all(c.view_input(view).get(k) == v for k, v in required.items())
            for c in ok_calls
        ):
            reasons.append(
                f"no {expect_tool} call matched required args {required!r} "
                f"(got: {[c.view_input(view) for c in ok_calls]!r:.300})"
            )

    if task_score.get("prompt_not_json_blob") and ok_calls:
        for call in ok_calls:
            vin = call.view_input(view)
            if "prompt" in vin and _looks_like_json_blob(vin["prompt"]):
                reasons.append(
                    f"{expect_tool}.prompt is a serialized JSON blob, not plain text: "
                    f"{str(vin['prompt'])[:120]!r}"
                )

    if "expect_file" in task_score:
        p = Path(task_score["expect_file"])
        if not p.is_absolute():
            p = workspace / p
        if not p.exists():
            reasons.append(f"expected file {p} to exist")
        elif "expect_file_contains" in task_score:
            content = p.read_text(errors="replace")
            if task_score["expect_file_contains"] not in content:
                reasons.append(
                    f"file {p} lacks {task_score['expect_file_contains']!r} "
                    f"(got: {content[:120]!r})"
                )

    if "expect_text_contains" in task_score:
        want = task_score["expect_text_contains"].lower()
        if want not in t.final_text.lower():
            reasons.append(
                f"final text lacks {task_score['expect_text_contains']!r} "
                f"(got: {t.final_text[:160]!r})"
            )

    if "expect_text_min_len" in task_score:
        if len(t.final_text.strip()) < int(task_score["expect_text_min_len"]):
            reasons.append(
                f"final text too short ({len(t.final_text.strip())} chars "
                f"< {task_score['expect_text_min_len']})"
            )

    if "forbid_bash_containing" in task_score:
        needle = task_score["forbid_bash_containing"]
        for e in t.tool_events:
            if e.view_name(view) == "bash" and needle in str(
                e.view_input(view).get("command", "")
            ):
                reasons.append(f"bash command contained forbidden {needle!r}")

    if task_score.get("forbid_breaker_trip") and t.breaker_tripped:
        reasons.append("circuit breaker tripped")

    if task_score.get("forbid_malformed") and t.dropped_malformed > 0:
        reasons.append(f"{t.dropped_malformed} malformed-empty call(s) dropped")

    if task_score.get("require_graceful"):
        if t.breaker_tripped:
            reasons.append("not graceful: circuit breaker tripped")
        elif not t.final_text.strip():
            reasons.append("not graceful: empty final text")

    return (not reasons, reasons)
