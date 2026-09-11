"""A model override that cannot be built must not be reported as active.

THE DEFECT
----------
`_route_override` called `ProviderRegistry.create(entry.provider_config)`
unguarded. Selecting a cloud model whose credential is not configured raised
there on EVERY turn. `agent_loop` caught it, logged a warning, and served the
primary — but the router KEPT the entry, so `get_override_for_session` (and the
REST surface that reads it) went on reporting the override as active.

Measured before the fix:

    API reports override active? -> True
    turn 1: route() RAISED ValueError: No API key configured for provider 'anthropic'
    turn 2: route() RAISED ValueError: ...
    turn 3: route() RAISED ValueError: ...
    API still reports override active? -> True

The user was told their override was in effect. The primary answered every
message.

THE FIX
-------
An override that cannot be built is not an active override. The build failure
is logged at ERROR with the reason, the entry is dropped exactly the way
`clear_override` drops one (same backend-only persistence rule — a dropped
override and a user-cleared one must leave the same state behind), and `route`
falls through to normal routing. The turn is served, and by the model the
router will now truthfully report.

Deliberately NOT retried on later turns: the credential is read at build time
from config, so a second attempt fails identically. Re-issuing the override
command after fixing the config is the way back, and it is one command.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.router.model_router import (  # noqa: E402
    ModelRouter,
    RouteReason,
    RouterConfig,
)

CLOUD = {"provider": "anthropic", "model": "claude-opus-5"}
SESSION = "telegram:42"


@pytest.fixture
def router():
    return ModelRouter(
        RouterConfig(),
        primary_provider=object(),
        primary_adapter=object(),
        primary_model="gemma-local",
    )


def _break_provider_creation(monkeypatch, exc: Exception):
    import prometheus.providers.registry as registry

    def _create(cfg):
        raise exc

    monkeypatch.setattr(registry.ProviderRegistry, "create", staticmethod(_create))


def test_an_unbuildable_override_stops_being_reported_as_active(
    router, monkeypatch
):
    """THE defect: reported state must match what actually served."""
    _break_provider_creation(
        monkeypatch, ValueError("No API key configured for provider 'anthropic'")
    )
    router.set_override(SESSION, CLOUD)
    assert router.get_override_for_session(SESSION) is not None

    router.route("hello", context={"session_id": SESSION})

    assert router.get_override_for_session(SESSION) is None, (
        "the router still reports an override that it cannot build and that "
        "did not serve the turn"
    )


def test_the_turn_is_served_by_the_primary_not_an_exception(router, monkeypatch):
    """`route()` raising every turn is what forced the caller to guess."""
    _break_provider_creation(monkeypatch, ValueError("No API key configured"))
    router.set_override(SESSION, CLOUD)

    decision = router.route("hello", context={"session_id": SESSION})

    assert decision.reason is RouteReason.PRIMARY
    assert decision.model_name == "gemma-local"


def test_later_turns_route_cleanly(router, monkeypatch):
    """Three turns, three clean routes — not three raises."""
    _break_provider_creation(monkeypatch, ValueError("No API key configured"))
    router.set_override(SESSION, CLOUD)

    reasons = [
        router.route("hi", context={"session_id": SESSION}).reason
        for _ in range(3)
    ]
    assert reasons == [RouteReason.PRIMARY] * 3


def test_the_failure_is_logged_with_its_reason(router, monkeypatch, caplog):
    """Dropping the override silently would trade one lie for another."""
    _break_provider_creation(
        monkeypatch, ValueError("No API key configured for provider 'anthropic'")
    )
    router.set_override(SESSION, CLOUD)

    with caplog.at_level(logging.ERROR, logger="prometheus.router.model_router"):
        router.route("hello", context={"session_id": SESSION})

    assert "CANNOT BE BUILT" in caplog.text
    assert "No API key configured" in caplog.text, (
        "the log does not say WHY the override could not be built"
    )
    assert "claude-opus-5" in caplog.text and SESSION in caplog.text


def test_a_working_override_still_overrides(router, monkeypatch):
    """The fix must not drop overrides that build fine.

    Without this, dropping unconditionally would satisfy every test above and
    disable the whole override feature.
    """
    import prometheus.providers.registry as registry

    sentinel = object()
    monkeypatch.setattr(
        registry.ProviderRegistry, "create", staticmethod(lambda cfg: sentinel)
    )
    router.set_override(SESSION, CLOUD)

    decision = router.route("hello", context={"session_id": SESSION})

    assert decision.reason is RouteReason.USER_OVERRIDE
    assert decision.model_name == "claude-opus-5"
    assert decision.provider is sentinel
    assert router.get_override_for_session(SESSION) is not None


def test_a_dropped_override_leaves_the_same_state_as_a_cleared_one(
    router, monkeypatch
):
    """A dropped override and a user-cleared one must be indistinguishable.

    Otherwise `/local` and a build failure leave different residue, and the
    next `/claude` behaves differently depending on which happened.
    """
    _break_provider_creation(monkeypatch, ValueError("No API key configured"))

    router.set_override(SESSION, CLOUD)
    router.route("hello", context={"session_id": SESSION})
    after_drop = router.get_override_for_session(SESSION)

    router.set_override(SESSION, CLOUD)
    router.clear_override(SESSION)
    after_clear = router.get_override_for_session(SESSION)

    assert after_drop == after_clear is None
    assert router.has_override is False
