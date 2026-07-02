"""TRIPWIRE Piece 3 — acceptance tests for the three historically-bitten paths.

Every test here enters at the REAL REST surface (POST /api/chat/send through
the production create_app + WebSocketBridge + run_loop + ModelRouter chain,
bearer-auth on) and asserts on the OUTBOUND RECORD at the provider boundary —
what actually left the daemon toward the model — not on internal calls.

The only double is the boundary recorder (tests/support/real_app.py), declared
per-test via allow_doubles so the tripwire stays armed for everything else.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from tests.support.real_app import BOUNDARY_DOUBLE, build_real_app  # noqa: E402


# --------------------------------------------------------------------------- #
# 1 — the #74 regression: per-session model override, interleaved sessions
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_model_override_per_session():
    """#74 regression — every turn's outbound request must carry ITS OWN
    session's routing: session A (overridden to anthropic/claude) hits the
    override provider; session B (local) hits the primary; interleaved A,B,A.

    Sanity contract: this test MUST go red if run_loop reads the routing
    session off the shared ``loop_context`` instead of the per-call
    ``session_id`` parameter (agent_loop.py:665 / the
    ``session_id if session_id is not None else context.session_id`` branch
    at the router.route call). Demonstrated red against a local revert of the
    #74 fix — evidence in the TRIPWIRE PR description; do not re-commit the
    revert.
    """
    h = build_real_app()
    with h.client:
        # Sessions exist implicitly on first dispatch; the override is set via
        # the real REST endpoint first (stores the vetted claude preset).
        override_rec = h.set_override_recorder("session-A", key="claude")

        h.send_turn("session-A", "first question for the cloud override")
        h.send_turn("session-B", "first question for the local primary")
        h.send_turn("session-A", "second question for the cloud override")

        # Outbound records: A's turns ONLY on the override provider, B's ONLY
        # on the primary. (The #74 bug made ALL of these go primary.)
        assert len(override_rec.requests) == 2, (
            f"session-A turns reached the override provider "
            f"{len(override_rec.requests)}x, expected 2 — per-session routing lost"
        )
        assert len(h.primary.requests) == 1, (
            f"primary provider saw {len(h.primary.requests)} turn(s), expected exactly "
            f"session-B's 1 — an A-turn leaked to primary (the #74 failure mode)"
        )
        a_texts = [m.content[0].text for r in override_rec.requests for m in r.messages if m.role == "user"]
        assert any("first question for the cloud" in t for t in a_texts)
        assert any("second question for the cloud" in t for t in a_texts)
        b_texts = [m.content[0].text for r in h.primary.requests for m in r.messages if m.role == "user"]
        assert any("local primary" in t for t in b_texts)

        # And the REST view agrees with the routing that actually happened.
        assert h.get_session_model("session-A")["key"] == "claude"
        assert h.get_session_model("session-B")["key"] == "local"


# --------------------------------------------------------------------------- #
# 2 — the per-message mode channel: agent tools ↔ chat no-tools ↔ agent again
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_mode_channel_per_message():
    """mode is PER-MESSAGE at the engine layer, not session-sticky: an agent
    turn's outbound request carries the tool schema; a chat turn on the SAME
    session carries none (and suppress_tools, so the GBNF grammar dies at
    every tier); the following agent turn restores tools."""
    h = build_real_app()
    with h.client:
        h.send_turn("session-M", "list something", mode="agent")
        h.send_turn("session-M", "just chat with me", mode="chat")
        h.send_turn("session-M", "list something again", mode="agent")

        reqs = h.primary.requests
        assert len(reqs) == 3
        agent1, chat, agent2 = reqs

        assert agent1.tools, "agent turn 1 must carry the real tool schema outbound"
        assert not agent1.suppress_tools

        assert chat.tools == [], "chat turn must carry NO tools outbound"
        assert chat.suppress_tools, (
            "chat turn must set suppress_tools so the provider drops the "
            "tool-calling grammar (structurally tool-free, not just in-practice)"
        )

        assert agent2.tools, "agent turn AFTER a chat turn must restore tools (per-message, not sticky)"
        assert not agent2.suppress_tools


# --------------------------------------------------------------------------- #
# 3 — tool_choice threading to the boundary (the #73 gap, armed as a gate)
# --------------------------------------------------------------------------- #


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING (TRIPWIRE checkpoint 2): tool_choice does NOT survive REST→provider "
        "on main — the channel doesn't exist (ApiMessageRequest has no tool_choice "
        "field; /api/chat/send ignores the key). It is #73's unmerged payload. "
        "strict=True arms the merge gate: when #73 threads it, this XPASSes red and "
        "must be promoted to a plain passing acceptance test."
    ),
)
def test_acceptance_tool_choice_threading():
    """A turn carrying tool_choice=required must show tool_choice present and
    correct ON THE OUTBOUND RECORD. Scope: threading only — enforcement
    semantics (grammar/native param) are #73's own merge gate."""
    h = build_real_app()
    with h.client:
        h.send_turn(
            "session-T",
            "force a search",
            mode="agent",
            extra_body={"tool_choice": "required"},
        )
        req = h.primary.requests[-1]
        assert getattr(req, "tool_choice", None) == "required", (
            f"tool_choice did not reach the provider boundary "
            f"(outbound has {getattr(req, 'tool_choice', '<no field>')!r})"
        )
