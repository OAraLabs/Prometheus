"""#457 — two consecutive assistant messages must not reach the provider.

The local backend's chat template rejects the request outright:

    HTTP 400 "Cannot have 2 or more assistant messages at the end of the list."

Captured live 2026-09-01 by the #355 payload diagnostic — an 18-message
healthy interleave (assistant(tool_calls)/tool, repeating) breaking at the
tail with `#19 assistant str(4ch)` followed by `#20 assistant str(25ch)`,
nothing between. The tail is assembled by the loop, not supplied by the
client, so the adjacency is always a construction defect upstream. The issue
names two candidate mechanisms and confirms NEITHER — and the project's own
recurring lesson (§4, and the four defects guessed on 2026-08-28) is that
fixing one unconfirmed mechanism while another produces the same wire shape
is how the bug survives its own fix.

So the invariant is enforced at the ONE seam every model call passes through
(`render_messages_for_model`, the per-call projection), which catches every
mechanism that produces the shape:

  * adjacent text-only assistants are MERGED — the turn survives;
  * the merge is LOUD (WARNING) so the construction site stays findable;
  * assistants carrying tool_uses are NEVER merged — folding one into a
    neighbour would break tool_use/tool_result pairing (#396's invariant
    outranks the template's shape rule), and the adjacency is logged at
    ERROR as the louder bug it is;
  * session/LCM history is untouched — same non-mutating contract as the
    untrusted-banner projection that already lives in this function.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    render_messages_for_model,
)

_LOGGER = "prometheus.engine.messages"


def _a(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _u(text: str) -> ConversationMessage:
    return ConversationMessage.from_user_text(text)


def _a_tool(tool_id: str = "t1") -> ConversationMessage:
    return ConversationMessage(
        role="assistant",
        content=[ToolUseBlock(id=tool_id, name="bash", input={"command": "ls"})],
    )


def _tool_result(tool_id: str = "t1") -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id=tool_id, content="ok", is_error=False)],
    )


# ── the captured shape ───────────────────────────────────────────────────────

def test_the_captured_tail_shape_no_longer_reaches_the_provider(caplog):
    """#19 assistant(4ch) + #20 assistant(25ch) at the end of the list — the
    exact captured adjacency — renders as ONE assistant message."""
    msgs = [
        _u("hi"),
        _a_tool("t1"), _tool_result("t1"),
        _a_tool("t2"), _tool_result("t2"),
        _a("well"),                    # 4 chars
        _a("here is the final answer"),  # 25 chars
    ]
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = render_messages_for_model(msgs)

    assert out[-1].role == "assistant"
    assert out[-2].role != "assistant", (
        "two assistant messages still sit adjacent at the tail — the shape "
        "the local backend's template 400s on"
    )
    assert out[-1].text == "wellhere is the final answer", (
        "the merge must keep both turns' content in order"
    )
    # No adjacency anywhere in the output.
    for i in range(1, len(out)):
        assert not (out[i - 1].role == "assistant" and out[i].role == "assistant")


def test_the_merge_is_loud(caplog):
    """Silent repair is how a construction defect becomes permanent. The
    WARNING is what makes the upstream site findable from the daemon journal."""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        render_messages_for_model([_u("hi"), _a("one"), _a("two")])
    assert any(
        "#457" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_three_in_a_row_collapse_to_one(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        out = render_messages_for_model([_u("hi"), _a("a"), _a("b"), _a("c")])
    assistants = [m for m in out if m.role == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].text == "abc"


# ── what must NOT be merged ──────────────────────────────────────────────────

def test_assistants_carrying_tool_uses_are_never_merged(caplog):
    """Folding a tool_use message into a neighbour breaks tool_use/tool_result
    pairing (#396) — a louder bug than the template 400. Logged at ERROR,
    left structurally alone."""
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        out = render_messages_for_model([_u("hi"), _a_tool("t1"), _a("text")])
    assert len(out) == 3, "the tool-carrying adjacency was merged"
    assert out[1].tool_uses, "the tool_use blocks were lost"
    assert any(
        r.levelno == logging.ERROR and "tool_uses" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_a_healthy_interleave_passes_through_untouched():
    """The normal shape — assistant(tool)/tool-result/assistant(text) — has no
    adjacency and must come back byte-identical in structure."""
    msgs = [
        _u("hi"),
        _a_tool("t1"), _tool_result("t1"),
        _a("answer"),
        _u("more"),
        _a("more answer"),
    ]
    out = render_messages_for_model(msgs)
    assert [(m.role, len(m.content)) for m in out] == \
           [(m.role, len(m.content)) for m in msgs]


# ── the non-mutating contract ────────────────────────────────────────────────

def test_history_is_not_mutated():
    """Session/LCM keep the original turns — the merge lives only in the
    per-call projection, same contract as the untrusted banner."""
    first, second = _a("one"), _a("two")
    msgs = [_u("hi"), first, second]
    out = render_messages_for_model(msgs)

    assert msgs[1].text == "one" and msgs[2].text == "two", (
        "the INPUT messages were edited — history would lose a turn"
    )
    assert first.content == [TextBlock(text="one")]
    assert out[1] is not first, "the merged message must be a copy"


def test_merge_never_upgrades_trust():
    """The untrusted banner does not apply to assistant messages (it fences
    role='user' injections), so the merge's trust contract is the flag itself:
    folding an untrusted block into a message stamped trusted must NOT launder
    it. Weakest trust wins."""
    injected = ConversationMessage(
        role="assistant",
        content=[TextBlock(text="task output")],
        provenance="task_supervisor",
        is_trusted=False,
    )
    out = render_messages_for_model([_u("hi"), _a("one"), injected])
    assert len(out) == 2
    merged = out[1]
    assert "one" in merged.text and "task output" in merged.text
    assert merged.is_trusted is False, (
        "the merge inherited the FIRST message's trusted flag and laundered "
        "the untrusted half — a downstream reader of is_trusted would now "
        "treat machinery output as the operator's own"
    )

    # And the other order — untrusted first, trusted second — stays untrusted.
    out2 = render_messages_for_model([_u("hi"), injected, _a("two")])
    assert out2[1].is_trusted is False


def test_no_adjacency_no_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        render_messages_for_model([_u("hi"), _a("one")])
    assert not any("#457" in r.getMessage() for r in caplog.records), (
        "the invariant fired on a list that never had the defect — cry-wolf "
        "warnings train readers to skip the real one"
    )
