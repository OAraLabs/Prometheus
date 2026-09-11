"""A command hook must not let the payload become shell code.

WHAT HAPPENED
-------------
`HookExecutor._run_command_hook` built the command by string replacement and
handed the result to `bash -lc`::

    command = hook.command.replace("$ARGUMENTS", json.dumps(payload))
    await asyncio.create_subprocess_exec("/bin/bash", "-lc", command, ...)

The payload is model-controlled. A `PRE_TOOL_USE` event carries
``{"tool_name": ..., "tool_input": ...}``, and `tool_input` is whatever the
model chose to pass: a bash command, a file body, or text it copied out of a
page it fetched a moment earlier. Splicing it into a string parsed by bash
turns that data into code.

The loader's own docstring documented ``command: "echo checking $ARGUMENTS"``.
Reproduced against the real executor with that exact example and a `tool_input`
of ``{"command": "$(touch /tmp/PWNED)"}``: **the file was created.** The hook
shell is `bash -lc` with `{**os.environ}`, so the execution carried the
daemon's full environment — `ANTHROPIC_API_KEY`, `PROMETHEUS_API_TOKEN`, all
of it.

THE CATEGORY
------------
This is the third kind: **agent-chosen but remotely steerable.** The agent
picks the tool arguments, so nothing here is "the remote attacker's input" in
the obvious sense — but any argument chosen after reading untrusted content is
attacker-influenced, and a summarise-this-page turn produces exactly that. It
needs perimeter treatment, and the perimeter is *where the data is allowed to
go*, never *how much the agent may attempt*.

THE FIX THESE TESTS PIN
-----------------------
The payload is no longer spliced. It is exported as the `ARGUMENTS`
environment variable, and the operator's command string reaches bash exactly
as written. `$ARGUMENTS` resolves by ordinary parameter expansion, and bash
does not re-evaluate the value of an expanded variable — so command
substitution, backticks, `;`, `&&` and redirections in the payload are all
inert. That is a property of where the data goes, not of how well it was
escaped, which is why there is no character filter or pattern list anywhere in
the fix or in these tests.

`test_the_old_splice_would_have_executed_every_payload` is the mutation check:
it reconstructs the previous behaviour and asserts each payload below really
does execute under it. Without it, a harness that silently ran nothing would
make every assertion here pass green.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.hooks.events import HookEvent  # noqa: E402
from prometheus.hooks.executor import (  # noqa: E402
    HookExecutionContext,
    HookExecutor,
)
from prometheus.hooks.registry import HookRegistry  # noqa: E402
from prometheus.hooks.schemas import CommandHookDefinition  # noqa: E402

# The example the loader documents, and the quoted form it documents now.
DOCUMENTED_OLD = "echo checking $ARGUMENTS"
DOCUMENTED_NOW = 'echo "checking $ARGUMENTS"'

# MEASURED against the pre-fix splice, not assumed. Both spellings, all nine
# shapes, marker-file evidence:
#
#                          unquoted   quoted
#   command_substitution     EXEC      EXEC
#   backticks                EXEC      EXEC
#   quote_breakout           --        EXEC
#   statement_separator      --        --
#   logical_and              --        --
#   pipeline                 --        --
#   redirection              --        --
#   newline                  --        --
#   subshell                 --        --
#
# The six that did not fire were contained by `json.dumps` quoting: the
# payload value sits inside JSON's own double quotes, which bash reads as
# SHELL double quotes, and `;` `&&` `|` `>` are literal there. That is an
# accident of the serializer, not a defense anyone designed — it evaporates if
# the payload is ever spliced somewhere without surrounding quotes, or the
# serializer changes. `$(...)` and backticks are evaluated INSIDE double
# quotes, which is why they fire regardless.
#
# NOTE THE THIRD ROW. `quote_breakout` fires only under the QUOTED spelling:
# the alternating quotes leave the parser outside a string by the time it
# reaches `; touch ...`. Quoting the placeholder did not make the splice safe,
# it made one shape worse — which is the case against fixing this by escaping
# harder, and the case for the payload never entering the command text at all.
EXPLOITED_BOTH_SPELLINGS = ("command_substitution", "backticks")
EXPLOITED_QUOTED_ONLY = ("quote_breakout",)
CONTAINED_BY_JSON_QUOTING = (
    "statement_separator", "logical_and", "pipeline",
    "redirection", "newline", "subshell",
)


def _payloads(marker: Path) -> dict[str, str]:
    """Shapes that turn spliced data into shell code. Each writes `marker`."""
    m = str(marker)
    return {
        "command_substitution": f"$(touch {m})",
        "backticks": f"`touch {m}`",
        "statement_separator": f"x; touch {m}",
        "logical_and": f"x && touch {m}",
        "pipeline": f"x | touch {m}",
        "redirection": f"x > {m}",
        "newline": f"x\ntouch {m}",
        # Breaks out of the operator's own double quotes in DOCUMENTED_NOW.
        "quote_breakout": f'"; touch {m}; echo "',
        "subshell": f"(touch {m})",
    }


def _run(command: str, tool_input_command: str, tmp_path: Path) -> str:
    reg = HookRegistry()
    reg.add(HookEvent.PRE_TOOL_USE,
            CommandHookDefinition(type="command", command=command))
    executor = HookExecutor(
        reg,
        HookExecutionContext(cwd=tmp_path, provider=None, default_model="unused"),
    )
    payload = {
        "tool_name": "bash",
        "tool_input": {"command": tool_input_command},
        "event": HookEvent.PRE_TOOL_USE.value,
    }
    result = asyncio.run(executor.execute(HookEvent.PRE_TOOL_USE, payload))
    return result.results[0].output or ""


# --------------------------------------------------------------------------
# The perimeter
# --------------------------------------------------------------------------
@pytest.mark.parametrize("hook_command", [DOCUMENTED_NOW, DOCUMENTED_OLD],
                         ids=["documented-quoted", "documented-unquoted"])
@pytest.mark.parametrize("shape", sorted(_payloads(Path("/x"))))
def test_payload_shell_metacharacters_do_not_execute(hook_command, shape, tmp_path):
    """No shape executes, under either spelling of the documented example.

    The unquoted spelling is included on purpose: operators who copied the old
    docstring have it in their config right now, and the fix has to hold for
    them without an edit on their side.
    """
    marker = tmp_path / f"EXECUTED_{shape}"
    payload_value = _payloads(marker)[shape]

    _run(hook_command, payload_value, tmp_path)

    assert not marker.exists(), (
        f"payload shape {shape!r} EXECUTED as shell code: {marker} was created. "
        f"The hook command was {hook_command!r} and the tool argument was "
        f"{payload_value!r}."
    )


@pytest.mark.parametrize("shape", sorted(_payloads(Path("/x"))))
def test_the_payload_still_arrives_intact(shape, tmp_path):
    """Neutralised, not mangled — the hook must still SEE what it was sent.

    `unknown` and `zero` are not the same, and neither are "inert" and "empty":
    a fix that stripped metacharacters would pass the test above while quietly
    lying to every hook about what the model did. The old splice actually did
    destroy the payload — command substitution consumed it, and the hook
    received `{command: }`.
    """
    marker = tmp_path / f"EXECUTED_{shape}"
    payload_value = _payloads(marker)[shape]

    output = _run(DOCUMENTED_NOW, payload_value, tmp_path)

    assert output.startswith("checking "), output
    seen = json.loads(output[len("checking "):])
    assert seen["tool_input"]["command"] == payload_value, (
        "the payload reached the hook altered. Expected the exact value the "
        "model passed; got something else — a hook cannot make a correct "
        "decision about an argument it was shown incorrectly."
    )
    assert seen["tool_name"] == "bash"
    assert not marker.exists()


def test_the_command_string_is_not_rewritten(tmp_path):
    """The operator's command reaches bash exactly as written.

    If anything still rewrites the command text, `$ARGUMENTS` inside SINGLE
    quotes would interpolate. It must not: single quotes are how a shell author
    says "literal", and honouring that is what makes the data/code boundary
    real rather than best-effort.
    """
    output = _run("echo 'literal $ARGUMENTS here'", "harmless", tmp_path)
    assert output == "literal $ARGUMENTS here", output


def test_the_payload_is_exported_under_both_names(tmp_path):
    """`PROMETHEUS_HOOK_PAYLOAD` predates this change and must keep working."""
    out = _run('test "$ARGUMENTS" = "$PROMETHEUS_HOOK_PAYLOAD" && echo same',
               "harmless", tmp_path)
    assert out == "same", out

    event = _run('echo "$PROMETHEUS_HOOK_EVENT"', "harmless", tmp_path)
    assert event == HookEvent.PRE_TOOL_USE.value, event


# --------------------------------------------------------------------------
# Mutation check
# --------------------------------------------------------------------------
def _old_splice_executes(hook_command: str, shape: str, tmp_path: Path) -> bool:
    """Replay the removed behaviour verbatim and report whether it fired."""
    marker = tmp_path / f"OLDSPLICE_{shape}"
    # This helper is called twice with the same tmp_path in one test. Without
    # clearing, the second call reads the FIRST call's marker and reports a
    # false positive — which it did, and which briefly looked like a real
    # finding about the unquoted form.
    marker.unlink(missing_ok=True)
    payload = {
        "tool_name": "bash",
        "tool_input": {"command": _payloads(marker)[shape]},
        "event": HookEvent.PRE_TOOL_USE.value,
    }
    # Verbatim the line that was removed from _run_command_hook.
    spliced = hook_command.replace("$ARGUMENTS", json.dumps(payload, ensure_ascii=True))

    async def go():
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash", "-lc", spliced,
            cwd=str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    asyncio.run(go())
    return marker.exists()


@pytest.mark.parametrize("shape", EXPLOITED_BOTH_SPELLINGS)
@pytest.mark.parametrize("hook_command", [DOCUMENTED_OLD, DOCUMENTED_NOW],
                         ids=["documented-unquoted", "documented-quoted"])
def test_the_old_splice_executed_these_under_either_spelling(
    hook_command, shape, tmp_path
):
    """The mutation check. These payloads really did run arbitrary commands.

    This is what keeps the perimeter tests from passing vacuously: if the
    harness stopped exercising anything — a hook that never runs, a marker path
    that cannot be written — every assertion above would still be green and
    this one would not.
    """
    assert _old_splice_executes(hook_command, shape, tmp_path), (
        f"the pre-fix splice did NOT execute {shape!r} under {hook_command!r}. "
        f"This test's premise is stale — if the payload was never dangerous, "
        f"the matching perimeter test is proving nothing."
    )


@pytest.mark.parametrize("shape", EXPLOITED_QUOTED_ONLY)
def test_quoting_the_placeholder_made_one_shape_worse(shape, tmp_path):
    """Quoting `$ARGUMENTS` was not a fix — for this shape it was the opposite.

    Under the quoted example the alternating quotes leave bash's parser outside
    a string by the time it reaches the injected `;`, so the breakout fires
    where the unquoted form contained it. Recorded because "just quote it" is
    the obvious cheap fix, and this is the measurement that rules it out.
    """
    assert _old_splice_executes(DOCUMENTED_NOW, shape, tmp_path), (
        f"{shape!r} no longer fires under the quoted splice — the rationale "
        f"recorded against escaping-based fixes needs re-measuring."
    )
    assert not _old_splice_executes(DOCUMENTED_OLD, shape, tmp_path), (
        f"{shape!r} now also fires under the UNQUOTED splice; the asymmetry "
        f"this test documents has changed."
    )


@pytest.mark.parametrize("shape", CONTAINED_BY_JSON_QUOTING)
@pytest.mark.parametrize("hook_command", [DOCUMENTED_OLD, DOCUMENTED_NOW],
                         ids=["documented-unquoted", "documented-quoted"])
def test_json_quoting_incidentally_contained_these(hook_command, shape, tmp_path):
    """These did NOT fire pre-fix — and that is worth pinning, not celebrating.

    They were contained by `json.dumps` putting the value inside quotes that
    bash reads as shell quotes. Nobody designed that, nothing documents it, and
    it would evaporate the moment a payload were interpolated somewhere without
    surrounding quotes. Pinned so that if this ever changes, it is noticed as a
    change rather than discovered as a breach.
    """
    assert not _old_splice_executes(hook_command, shape, tmp_path), (
        f"{shape!r} now executes under the pre-fix splice where it previously "
        f"did not. The containment was incidental; treat this as evidence the "
        f"serializer or the shape set has moved, and re-derive the table."
    )


def test_prompt_hooks_still_interpolate_textually():
    """The prompt path is unchanged, and that is correct.

    A prompt is model input, where interpolated text stays text. A command
    string is parsed by bash, where it becomes code. Only the second needed
    fixing, and conflating them would be its own bug.
    """
    from prometheus.hooks.executor import _inject_arguments

    payload = {"tool_name": "bash", "tool_input": {"command": "$(touch /tmp/x)"}}
    rendered = _inject_arguments("review: $ARGUMENTS", payload)
    assert rendered.startswith("review: {")
    assert "$(touch /tmp/x)" in rendered
