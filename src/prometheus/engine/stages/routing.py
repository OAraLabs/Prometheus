"""Turn setup, model routing: default hook T6 of the hook contract (Appendix A).

Moved verbatim out of ``_run_loop`` in ``engine/agent_loop.py``, which calls
:func:`route_turn` from exactly where this code sat: once per turn, after the
tool-choice directive (T5) and before the tool advertisement (T7), which must
describe the model routed to here (#462). Not part of "before model call".

Nothing is pluggable here yet. With nothing registered, the stage costs one
plain function call: no await, no payload building, no registry lookup
(contract section 16).

The identity-line rewrite names the provider the RouteDecision carries. The
primary route carries one since WP-X.14 (``ModelRouter._route_primary``);
before that it carried none and every primary turn was told ``(provider:
unknown)`` over a boot line that said ``llama_cpp``. The rewrite itself never
downgrades a known provider (``rewrite_model_identity``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus.context.system_prompt import rewrite_model_identity

if TYPE_CHECKING:
    from prometheus.engine.agent_loop import LoopContext
    from prometheus.engine.messages import ConversationMessage

# The LOOP's logger, not this module's. The daemon's log format prints the
# logger name, "ModelRouter: session=…" is the line an operator greps the
# journal for, and tests capture it under this name. Moving the code must not
# move its log lines.
log = logging.getLogger("prometheus.engine.agent_loop")


def route_turn(
    context: LoopContext,
    messages: list[ConversationMessage],
    *,
    session_id: str | None,
) -> None:
    """Route this turn on the per-run ``context``.

    Swaps in the provider, adapter, model and backend the router decides, and
    rewrites the system prompt's ``- Model:`` line to match. A router that
    raises leaves the turn on the primary and logs a WARNING.
    """
    # Sprint 10 / Phase 2: route the first user message through ModelRouter.
    # The canonical router returns a RouteDecision with pre-instantiated
    # provider + adapter. For the default/primary path the decision's provider
    # is the same instance already on the context (no-op swap). When a rule,
    # smart-routing, override, or escalation branch fires, the swap activates.
    # Phase 3.5: session_id threaded via context dict so the router's
    # per-session override lookup can fire (or, for session_id in (None,
    # "system"), always resolve to primary).
    if context.model_router is not None and messages:
        # M4: route on the MOST RECENT user message, not the first. In a long
        # session the first message is stale — the current turn's request is the
        # latest user turn, which is what smart-routing / task classification
        # should see. (Per-session overrides are keyed on session_id and are
        # unaffected by which message text we pass.)
        latest_user = next(
            (m.text for m in reversed(messages) if m.role == "user" and m.text),
            None,
        )
        if latest_user:
            try:
                # Per-session override is keyed on the ACTUAL turn's session_id,
                # threaded per-call (like `mode`): callers share one LoopContext whose
                # `.session_id` is NOT the live turn's session, so reading it here made
                # REST/WS turns look the override up under a stale id and silently run
                # the primary. Fall back to context.session_id for callers (CLI, coding,
                # gym) that don't pass one.
                # Bound once and reused for BOTH the routing call and the log line
                # below, so the audit record can never name a different session than
                # the one the override was actually looked up under.
                route_session = (
                    session_id if session_id is not None else context.session_id
                )
                decision = context.model_router.route(  # type: ignore[attr-defined]
                    latest_user,
                    context={"session_id": route_session},
                )
                reason_repr = (
                    decision.reason.value
                    if hasattr(decision.reason, "value")
                    else decision.reason
                )
                # INFO, not DEBUG: which model served a turn is the first question asked
                # when a user says "I switched models and it did not take", and at DEBUG
                # the answer does not exist. Checked 2026-08-25 against a daemon up four
                # days: three days of journal held ZERO of these lines, so a session's
                # routing had to be inferred sideways from a provider-build side effect
                # that only fires on the FIRST turn after a switch — the built provider
                # is cached on the override, so every later turn is silent. One line per
                # turn: this block runs once, before the tool-iteration loop.
                #
                # The user's message text is deliberately NOT in it. The old DEBUG line
                # carried a 60-char excerpt of the prompt; promoting that verbatim would
                # write conversation content into the daemon journal at the default
                # level. The routing decision is what needs auditing, not what was said.
                log.info(
                    "ModelRouter: session=%s → %s/%s (%s)",
                    route_session,
                    decision.provider_name,
                    decision.model_name,
                    reason_repr,
                )
                if decision.provider is not None:
                    context.provider = decision.provider
                if decision.adapter is not None:
                    context.adapter = decision.adapter
                if decision.model_name:
                    context.model = decision.model_name
                # Which box (if any) this decision runs on — the primary and cloud
                # overrides leave it None; a `/4090` override names the registry entry.
                context.backend = getattr(decision, "backend", None)
                # Phase 4 fix: after the router swap, rewrite the identity
                # line in the system prompt ("- Model: <name> (provider: <p>)")
                # to match the *active* provider. Without this, a primary-
                # baked system prompt says "Model: gemma4-26b" and Claude/GPT
                # dutifully impersonate the primary when the user asks "what
                # model is this?". The line is emitted by
                # prometheus.context.system_prompt._format_environment_section;
                # we rewrite it in-place rather than rebuilding the whole
                # prompt to avoid pulling environment detection into the hot
                # path of every request.
                if decision.provider_name or decision.model_name:
                    # Shared with the fallback handler rather than copied: a second copy is how
                    # a swapped-in model ends up claiming to be the primary, which is the bug
                    # this rewrite exists to prevent.
                    #
                    # `reason_repr == "primary"` is passed as the ANSWER to "is the serving
                    # model the local backend", which is what the old `!= "primary"` gate
                    # actually meant. See rewrite_model_identity's docstring — the fallback is a
                    # caller where route-reason and local-backend stop agreeing.
                    context.system_prompt = rewrite_model_identity(
                        context.system_prompt,
                        model_name=decision.model_name or "unknown",
                        # Empty for a decision built without one; the rewrite then keeps
                        # the provider the line already names instead of downgrading it.
                        provider_name=decision.provider_name,
                        # A named local backend IS local serving, even though the
                        # route reason is "user override".
                        serving_is_local_backend=(
                            reason_repr == "primary" or context.backend is not None
                        ),
                    )
            except Exception:
                # Phase 4: elevated from DEBUG → WARNING. A silent DEBUG here
                # hid a real production bug (stale-system-prompt identity)
                # from the logs. Any exception in route() means the user's
                # override (or task rule, or escalation) was NOT applied and
                # we silently fell through to primary — that's not something
                # we should discover by reading source code.
                log.warning(
                    "ModelRouter: route() raised — falling back to primary. "
                    "session_id=%r, latest_user=%r",
                    context.session_id,
                    (latest_user or "")[:60],
                    exc_info=True,
                )
