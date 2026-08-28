"""The fallback decision — SPRINT-provider-fallback, Phases 1 and 2.

Pure policy, so every rule is pinned here as a function of plain values rather than inferred from
the loop's behaviour.
"""

from __future__ import annotations

import pytest

from prometheus.api.turn_errors import (
    KIND_AUTH,
    KIND_BILLING,
    KIND_RATE_LIMIT,
    KIND_TIMEOUT,
    KIND_UNKNOWN,
    KIND_UNREACHABLE,
)
from prometheus.engine.fallback import TERMINAL_KINDS, decide, is_terminal


def call(**over):
    base = dict(
        kind=KIND_AUTH,
        provider_label="qwen3.8-max",
        requested_model="qwen3.8-max",
        fallback_model="Qwen3.8-27B",
        fallback_provider="llama_cpp",
        enabled=True,
        emitted_output=False,
        needed_tokens=8_000,
        fallback_window=32_768,
        window_is_measured=True,
    )
    base.update(over)
    return decide(**base)


# ── Phase 1: terminal is not retryable ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", [KIND_AUTH, KIND_BILLING])
def test_a_credential_failure_is_terminal(kind):
    assert is_terminal(kind)
    assert call(kind=kind).serve


@pytest.mark.parametrize("kind", [KIND_RATE_LIMIT, KIND_TIMEOUT, KIND_UNREACHABLE, KIND_UNKNOWN, None])
def test_a_recoverable_failure_is_NOT_terminal(kind):
    """The scope guard. 429 is already retried with backoff a layer down; degrading on it would
    move traffic off a provider that was about to succeed."""
    assert not is_terminal(kind)
    assert call(kind=kind).passes_through, f"{kind} must fall through untouched"


def test_rate_limit_is_deliberately_absent_from_the_terminal_set():
    assert KIND_RATE_LIMIT not in TERMINAL_KINDS
    assert TERMINAL_KINDS == frozenset({KIND_AUTH, KIND_BILLING})


# ── the gates that must not fire ─────────────────────────────────────────────────────────────

def test_disabled_falls_through():
    assert call(enabled=False).passes_through


def test_nothing_configured_to_fall_back_TO_falls_through():
    """Passing through keeps the provider's own error, which beats a message about a fallback
    that does not exist."""
    assert call(fallback_model=None).passes_through
    assert call(fallback_model="").passes_through


def test_never_falls_back_to_the_model_that_just_failed():
    """Without this a terminal failure on the local model loops against itself."""
    assert call(requested_model="Qwen3.8-27B", fallback_model="Qwen3.8-27B").passes_through


# ── Phase 2: the two refusals ────────────────────────────────────────────────────────────────

def test_output_already_sent_refuses_rather_than_re_answering(client=None):
    """Issue #293's invariant, one layer up: a second model answering over a partial reply
    contradicts what was already read."""
    d = call(emitted_output=True)
    assert d.refuse and not d.serve
    assert "already been sent" in d.message
    assert "contradict" in d.message


def test_a_turn_too_large_for_the_fallback_is_refused_with_its_numbers():
    d = call(needed_tokens=118_000, fallback_window=32_768)
    assert d.refuse and not d.serve
    assert "118,000" in d.message, "the refusal names what the turn needs"
    assert "32,768" in d.message, "and what the fallback has"
    assert "Qwen3.8-27B" in d.message


def test_the_refusal_says_whether_the_window_was_MEASURED_or_CONFIGURED():
    """The local side is measured (llama.cpp publishes n_ctx); the cloud side is a configured
    floor because cloud APIs do not publish context length. The message must not imply both were
    observed."""
    assert "measured" in call(needed_tokens=99_999, window_is_measured=True).message
    assert "configured" in call(needed_tokens=99_999, window_is_measured=False).message


def test_a_turn_that_fits_is_served_and_says_which_model_served_it():
    d = call(needed_tokens=8_000)
    assert d.serve and not d.refuse
    assert d.model == "Qwen3.8-27B"
    assert d.provider_name == "llama_cpp"
    assert "Qwen3.8-27B" in d.message and "qwen3.8-max" in d.message


def test_exactly_filling_the_window_is_served_not_refused():
    assert call(needed_tokens=32_768, fallback_window=32_768).serve
    assert call(needed_tokens=32_769, fallback_window=32_768).refuse


# ── ordering ─────────────────────────────────────────────────────────────────────────────────

def test_already_sent_output_outranks_the_context_cliff():
    """Both true: the reply is partial AND it would not fit. 'We already sent you half an answer'
    is the more specific truth and the one the reader needs."""
    d = call(emitted_output=True, needed_tokens=999_999)
    assert d.refuse
    assert "already been sent" in d.message
    assert "999,999" not in d.message


def test_a_non_terminal_kind_outranks_everything():
    """Even with output sent and an impossible size, a 429 is not ours to act on."""
    assert call(kind=KIND_RATE_LIMIT, emitted_output=True, needed_tokens=999_999).passes_through


def test_the_three_outcomes_are_mutually_exclusive():
    for d in [call(), call(emitted_output=True), call(kind=KIND_RATE_LIMIT)]:
        assert sum([d.serve, d.refuse, d.passes_through]) == 1
