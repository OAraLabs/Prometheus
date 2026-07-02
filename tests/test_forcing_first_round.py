"""FIRST-ROUND FORCING — the two pre-toggle follow-ups from IGNITION's live proof.

1. A forced directive (required / {tool:X}) binds ONLY the first substantive
   round, then relaxes to auto — a forced turn concludes in prose instead of
   burning rounds to the iteration cap.
2. Local enforcement: on the forced round, a grammar-capable provider gets
   native tools WITHHELD so its GBNF path enforces deterministically (the live
   llama-server build silently ignores OpenAI-shape function-forcing).
   Plus the fail-loud guard: a forced {tool:X} round answered with a call to a
   DIFFERENT tool raises — Y never masquerades as X.

Acceptance tests enter at the real REST surface (real_app harness); the engine
guard test drives the real run_loop directly. Dormant paths (auto/none) are
asserted unchanged.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")

from tests.support.real_app import BOUNDARY_DOUBLE, RecordingProvider, build_real_app  # noqa: E402


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_forced_round_relaxes_to_auto():
    """required on round 1, auto from round 2 — visible on the outbound records."""
    primary = RecordingProvider(
        label="primary:local",
        script=[("tool", "glob", {"pattern": "*.md"}), ("text", "done — found some files.")],
    )
    h = build_real_app(primary)
    with h.client:
        h.send_turn("session-F", "find markdown files", mode="agent", extra_body={"tool_choice": "required"})
        reqs = primary.requests
        assert len(reqs) == 2, f"expected forced round + relaxed round, got {len(reqs)}"
        assert reqs[0].tool_choice == "required"
        assert reqs[1].tool_choice == "auto", "the force must relax after the first substantive round"
        assert reqs[1].tools, "relaxed round returns to the native tools path"


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_forced_tool_relaxes_and_stays_that_tool_only_round_one():
    primary = RecordingProvider(
        label="primary:local",
        script=[("tool", "glob", {"pattern": "*"}), ("text", "listed.")],
    )
    h = build_real_app(primary)
    with h.client:
        h.send_turn("session-F2", "list things", mode="agent", extra_body={"tool_choice": {"tool": "glob"}})
        reqs = primary.requests
        assert len(reqs) == 2
        assert reqs[0].tool_choice == {"tool": "glob"}
        assert reqs[1].tool_choice == "auto"


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_grammar_capable_provider_gets_tools_withheld_on_forced_round():
    """The engine withholds native tools for the forced round when the provider
    can enforce via grammar — and restores them on the relaxed round."""
    primary = RecordingProvider(
        label="primary:local",
        script=[("tool", "glob", {"pattern": "*"}), ("text", "ok.")],
        force_via_grammar=True,
    )
    h = build_real_app(primary)
    with h.client:
        h.send_turn("session-G", "list things", mode="agent", extra_body={"tool_choice": {"tool": "glob"}})
        reqs = primary.requests
        assert len(reqs) == 2
        assert reqs[0].tools == [], "forced round: native tools withheld so the grammar path fires"
        assert reqs[0].tool_choice == {"tool": "glob"}
        assert reqs[1].tools, "relaxed round: native tools restored"
        assert reqs[1].tool_choice == "auto"


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_acceptance_auto_and_none_are_untouched():
    """Dormant-path gate: auto/none turns see no forcing artifacts."""
    primary = RecordingProvider(label="primary:local", force_via_grammar=True)
    h = build_real_app(primary)
    with h.client:
        h.send_turn("session-D", "hello", mode="agent", extra_body={"tool_choice": "auto"})
        req = primary.requests[-1]
        assert req.tool_choice == "auto" and req.tools, "auto: tools present even on a grammar-capable provider"
        h.send_turn("session-D", "chat", mode="agent", extra_body={"tool_choice": "none"})
        req = primary.requests[-1]
        assert req.tool_choice == "none" and req.tools == [] and req.suppress_tools


def test_engine_guard_wrong_tool_on_forced_round_fails_loud():
    """run_loop raises when a forced {tool:X} round comes back calling Y —
    driven on the real engine (real LoopContext + real tool registry)."""
    from prometheus.__main__ import create_tool_registry
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage

    provider = RecordingProvider(
        label="cloudish",  # native path (no can_force_via_grammar attr)
        script=[("tool", "read_file", {"path": "/tmp/x"})],  # Y, not the forced X
    )
    ctx = LoopContext(
        provider=provider,
        model="m",
        system_prompt="s",
        max_tokens=64,
        tool_registry=create_tool_registry({}),
        session_id="guard-test",
    )

    async def _drive():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")],
                                mode="agent", tool_choice={"tool": "glob"}):
            pass

    with pytest.raises(RuntimeError, match="was not honored"):
        asyncio.run(_drive())
