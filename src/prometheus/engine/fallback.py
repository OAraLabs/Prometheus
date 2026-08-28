"""Decide whether a failed turn can be served by a fallback model, and say so out loud.

SPRINT-provider-fallback. The pure half: every rule lives here as a function of plain values, so
the agent loop's job is to gather facts and act on a decision rather than to contain the policy.

Three things shape this module, all of them survey findings rather than assumptions:

1. **A terminal failure is not a retryable one.** `RETRYABLE_STATUS_CODES` already retries 429 and
   the 5xx family with backoff. 401/403 are in no retry set and never should be — retrying an
   expired key is exactly the wrong behaviour. This module governs what happens AFTER not
   retrying, and must never fire on a kind that is still being retried, or it would move traffic
   off a provider that was about to succeed.

2. **Once output has reached the user, there is no starting over.** The same invariant issue #293
   fixed one layer down in the providers' retry loops. A fallback that re-answers over a partial
   reply produces two contradictory answers in one turn, which is worse than a truncated one.

3. **The fallback window is the hazard, not the plumbing.** The local target measured 32768
   n_ctx against a cloud model configured at an order of magnitude more. Silently re-routing a
   turn that does not fit means the answer quietly gets worse with no stated cause, so a turn
   that will not fit is REFUSED with its numbers rather than compacted harder to force it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prometheus.api.turn_errors import KIND_AUTH, KIND_BILLING

# Kinds where trying again with this credential cannot help. Deliberately NOT including
# KIND_RATE_LIMIT (retried with backoff already), KIND_TIMEOUT or KIND_UNREACHABLE (transient by
# nature): degrading on those would abandon a provider that was about to recover.
log = logging.getLogger(__name__)

TERMINAL_KINDS: frozenset[str] = frozenset({KIND_AUTH, KIND_BILLING})


def is_terminal(kind: str | None) -> bool:
    """True when retrying with the same credential cannot help."""
    return kind in TERMINAL_KINDS


@dataclass(frozen=True)
class FallbackDecision:
    """What the loop should do. Exactly one of the three outcomes is true.

    `serve` — run the turn again on `model`, and say so in the reply.
    `refuse` — do not fall back; fail the turn with `message`, which explains why.
    neither  — this failure is not ours; re-raise unchanged.
    """

    serve: bool
    refuse: bool
    model: str | None = None
    provider_name: str | None = None
    message: str = ""

    @property
    def passes_through(self) -> bool:
        return not self.serve and not self.refuse


_PASS = FallbackDecision(serve=False, refuse=False)


def decide(
    *,
    kind: str | None,
    provider_label: str,
    requested_model: str,
    fallback_model: str | None,
    fallback_provider: str | None,
    enabled: bool,
    emitted_output: bool,
    needed_tokens: int,
    fallback_window: int,
    window_is_measured: bool,
) -> FallbackDecision:
    """Decide the outcome of one terminally-failed turn.

    Order matters and is not arbitrary — each gate answers a question the next one assumes.
    """
    # Not our failure. A retryable kind is still being retried a layer down; taking it here would
    # move traffic off a provider that was about to succeed.
    if not is_terminal(kind):
        return _PASS

    if not enabled:
        return _PASS

    # Nothing configured to fall back TO. Passing through keeps the provider's own error, which
    # is more useful than a message about a fallback that does not exist.
    if not fallback_model:
        return _PASS

    # Never fall back to the model that just failed. Without this, a terminal failure on the
    # local model would loop against itself.
    if fallback_model == requested_model:
        return _PASS

    # Output already reached the user (issue #293's invariant, one layer up). A second model
    # answering over a partial reply contradicts what was already read.
    if emitted_output:
        return FallbackDecision(
            serve=False,
            refuse=True,
            message=(
                f"{provider_label} failed partway through this reply ({kind}), and part of the"
                " answer had already been sent. Continuing on a different model would contradict"
                " what you have already read, so this turn stopped here. Retry to start over."
            ),
        )

    # The context cliff. Refuse rather than truncate: silently serving a turn that does not fit
    # makes the answer worse with no stated cause.
    if needed_tokens > fallback_window:
        window_phrase = (
            f"a measured {fallback_window:,}-token window"
            if window_is_measured
            # Cloud APIs do not publish context length, so this side is configuration, not an
            # observation, and the message must not imply otherwise.
            else f"a configured {fallback_window:,}-token window"
        )
        return FallbackDecision(
            serve=False,
            refuse=True,
            message=(
                f"{provider_label} is unavailable ({kind}). This turn needs about"
                f" {needed_tokens:,} tokens of context and the fallback {fallback_model} has"
                f" {window_phrase}, so it cannot serve this conversation. Restore the provider,"
                " or start a narrower session."
            ),
        )

    return FallbackDecision(
        serve=True,
        refuse=False,
        model=fallback_model,
        provider_name=fallback_provider,
        message=(
            f"{provider_label} is unavailable ({kind}) — this reply was served by"
            f" {fallback_model} instead."
        ),
    )


def degraded_notice(decision: FallbackDecision) -> str:
    """The sentence attached to a degraded reply.

    Attached to the REPLY, not logged: a degrade the reader has to go looking for is a silent
    degrade, and the whole point of this sprint is that the substitution is visible where the
    answer is.
    """
    return decision.message


# ── Phase 3/4: the target, and the loop-facing wrapper ───────────────────────────────────────

@dataclass(frozen=True)
class FallbackTarget:
    """Where a degraded turn goes. Resolved once, at loop construction.

    `is_local_backend` is carried explicitly rather than inferred from the provider name, because
    it answers the question the system-prompt identity rewrite actually asks — see
    `context.system_prompt.rewrite_model_identity`. Inferring it at the use site is how the
    degraded model ends up asserting that the local backend is not it, while being it.
    """

    model: str
    provider_name: str
    provider: object  # ModelProvider — constructed by the caller, never by this module
    is_local_backend: bool


def build_fallback_target(
    model_config: dict | None,
    detected_model: str | None = None,
) -> FallbackTarget | None:
    """Construct the fallback target from the LOCAL model configuration, or None.

    Configured, not hardcoded (Phase 4). `model.fallback` may name a different section; absent,
    the local model config is the target, which is what `GET /api/models` already reports as
    `is_default`. Returns None when nothing usable is configured — the caller then passes the
    provider's own error through, which beats a message about a fallback that does not exist.

    `detected_model` is what the caller resolved as the live model name, and it is REQUIRED in
    practice rather than a nicety: `model.model` is blank on the recommended configuration. The
    template says so in its own words — "A HINT, not an assertion. The BACKEND is authoritative
    ... Leave blank unless you need a name for a backend that cannot be asked." Requiring the
    config field therefore returned None on the very setup this daemon ships with, leaving the
    fallback inert for the second time.
    """
    if not model_config:
        return None
    cfg = dict(model_config.get("fallback") or model_config)
    if cfg.get("enabled") is False:
        return None
    provider_name = str(cfg.get("provider") or "llama_cpp")
    # Config first (an explicit `model.fallback.model` is an operator saying so), then what the
    # backend actually reports. Blank config is the NORMAL case here, not a misconfiguration.
    model = str(cfg.get("model") or detected_model or "")
    if not model:
        return None
    try:
        from prometheus.providers.registry import ProviderRegistry

        provider = ProviderRegistry.create(cfg)
    except Exception:  # noqa: BLE001 — an unbuildable fallback is "no fallback", not a crash
        return None
    return FallbackTarget(
        model=model,
        provider_name=provider_name,
        provider=provider,
        # llama_cpp and ollama serve from this box; everything else is a remote API.
        is_local_backend=provider_name in ("llama_cpp", "ollama"),
    )


class FallbackRefused(RuntimeError):
    """The turn cannot be served, and the message says why in the user's terms.

    A distinct type so the layer above can render `str(exc)` directly instead of wrapping it in
    a generic "something went wrong" — the whole point is that the reason reaches the reader.
    """


async def stream_round_with_fallback(
    *,
    envelope,
    provider,
    model: str,
    build_request,
    target: "FallbackTarget | None",
    enabled: bool,
    window_for,
    estimate_tokens,
    on_degrade,
    operation: str,
    round_index: int | None,
    session_id: str | None,
):
    """Stream one round; on a TERMINAL failure, serve it from the fallback instead.

    Wraps the envelope rather than replacing it — the envelope still observes and re-raises, and
    its docstring already says the loop owns recovery. This is that recovery, kept out of the
    loop body so the 4,000-line generator gains one call rather than a nested retry structure.

    Deliberately at most ONE degrade per round. The fallback gets no fallback of its own: a chain
    turns "which model answered?" into an archaeology problem, and the second hop would be
    degrading away from a model that was itself the degrade.

    `build_request(model)` rebuilds the request for whichever model serves, because the model
    name is part of the payload and reusing the failed request would ask the local backend to
    serve `qwen3.8-max`.
    """
    decision: FallbackDecision | None = None
    emitted = False

    try:
        async for event in envelope.stream(
            provider=provider,
            request=build_request(model),
            operation=operation,
            round_index=round_index,
            session_id=session_id,
        ):
            emitted = True
            yield event
        return
    except GeneratorExit:
        # Consumer stopped iterating — protocol signal, not a model failure.
        raise
    except Exception as exc:  # noqa: BLE001 — classified, then acted on or re-raised
        from prometheus.api.turn_errors import classify_turn_error

        detail = classify_turn_error(exc)
        window, measured = window_for(target.model) if target is not None else (0, False)
        needed = estimate_tokens()
        decision_inputs = {"needed": needed, "window": window}
        decision = decide(
            kind=detail.get("kind"),
            # The MODEL name, not classify_turn_error's provider: that field falls back to
            # the literal "the model provider" and is never empty, so `or model` was dead code.
            # And the model is what the user chose and what the message contrasts against —
            # "qwen3.8-max is unavailable ... served by Qwen3.8-27B instead".
            provider_label=model,
            requested_model=model,
            fallback_model=target.model if target is not None else None,
            fallback_provider=target.provider_name if target is not None else None,
            enabled=enabled,
            emitted_output=emitted,
            needed_tokens=needed,
            fallback_window=window,
            window_is_measured=measured,
        )
        if decision.passes_through:
            # A fallback that declines silently is the same failure this sprint removes, one
            # level up: the turn dies and nothing says whether the mechanism considered it.
            log.warning(
                "fallback DECLINED for %s: kind=%r terminal=%s enabled=%s target=%r "
                "emitted=%s needed=%s window=%s",
                model, detail.get("kind"), is_terminal(detail.get("kind")), enabled,
                getattr(target, "model", None), emitted,
                decision_inputs.get("needed"), decision_inputs.get("window"),
            )
            raise
        if decision.refuse:
            raise FallbackRefused(decision.message) from exc

    # Reached only when decision.serve — deliberately OUTSIDE the except block, so a failure on
    # the fallback attempt is reported as itself rather than as "during handling of the above".
    assert decision is not None and target is not None
    on_degrade(decision)

    # The degrade goes into the REPLY, ahead of the fallback's answer. A notice the reader has to
    # go looking for is a silent degrade, and a log line is exactly that.
    #
    # KNOWN GAP: this reaches the live stream but not stored history — history is rebuilt from
    # ApiMessageCompleteEvent, which this delta is not part of. So a re-read of the conversation
    # shows the degraded answer without the notice. Closing that means the notice travelling on
    # the completion event, which is a wider change than this sprint; tracked separately rather
    # than left implicit.
    from prometheus.providers.base import ApiTextDeltaEvent

    yield ApiTextDeltaEvent(text=f"⚠ {decision.message}\n\n")

    async for event in envelope.stream(
        provider=target.provider,
        request=build_request(target.model),
        operation=operation,
        round_index=round_index,
        session_id=session_id,
    ):
        yield event
