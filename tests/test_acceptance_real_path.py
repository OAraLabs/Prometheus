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
def test_acceptance_tool_choice_threading():
    """Every tool_choice value must reach the provider boundary per-call with
    correct shape ON THE OUTBOUND RECORD. Scope: threading — enforcement
    semantics (grammar/native param) are covered by the provider-level
    acceptance tests (IGNITION Pieces 2/3).

    History: this was TRIPWIRE's armed gate — xfail(strict=True) documenting
    that tool_choice did not survive REST→provider on main (no
    ApiMessageRequest field; /api/chat/send ignored the key). IGNITION threaded
    the channel; the XPASS strict-red fired as designed and the test was
    promoted to this plain acceptance test (evidence pair in the IGNITION PR).
    """
    h = build_real_app()
    with h.client:
        # required — the original gate case.
        h.send_turn("session-T", "force a search", mode="agent", extra_body={"tool_choice": "required"})
        assert getattr(h.primary.requests[-1], "tool_choice", None) == "required"

        # {tool: X} — dict shape survives intact (name preserved, not coerced).
        h.send_turn(
            "session-T", "use that one tool", mode="agent",
            extra_body={"tool_choice": {"tool": "web_search"}},
        )
        assert getattr(h.primary.requests[-1], "tool_choice", None) == {"tool": "web_search"}

        # none — resolves like chat mode: no tools offered AND suppress_tools set.
        h.send_turn("session-T", "just talk", mode="agent", extra_body={"tool_choice": "none"})
        none_req = h.primary.requests[-1]
        assert getattr(none_req, "tool_choice", None) == "none"
        assert none_req.tools == [] and none_req.suppress_tools

        # auto (explicit) and absent — both reach the boundary as "auto",
        # with the default path's tools/suppress_tools untouched.
        h.send_turn("session-T", "explicit auto", mode="agent", extra_body={"tool_choice": "auto"})
        assert getattr(h.primary.requests[-1], "tool_choice", None) == "auto"
        h.send_turn("session-T", "absent tool_choice", mode="agent")
        absent_req = h.primary.requests[-1]
        assert getattr(absent_req, "tool_choice", None) == "auto"
        assert not absent_req.suppress_tools


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_tool_choice_unknown_tool_is_400():
    """Fail-loud ingress: a forced tool that isn't registered is a clean 400
    at REST — never silent coercion, never a dispatched turn."""
    h = build_real_app()
    with h.client:
        before = len(h.primary.requests)
        resp = h.client.post(
            "/api/chat/send",
            json={
                "session_id": "session-T",
                "message": "force a ghost",
                "mode": "agent",
                "tool_choice": {"tool": "no_such_tool"},
            },
            headers=h.auth(),
        )
        assert resp.status_code == 400
        assert "no_such_tool" in resp.json().get("error", "")
        assert len(h.primary.requests) == before  # nothing reached the boundary
