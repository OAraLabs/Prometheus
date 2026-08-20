"""/api/status exposes the ceilings the LOOP is enforcing, not the config's.

The point of the field is the RESOLUTION HOP: config dict ->
resolve_max_tool_iterations() -> LoopContext field. An endpoint that re-resolved
from config would prove the resolver and leave "and the loop got the same
number" as inference. So every test here sets a value on the LIVE LoopContext
that appears in no config anywhere, and asserts the endpoint reports THAT.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from prometheus.config.shipped_defaults import SHIPPED_MAX_TOOL_ITERATIONS  # noqa: E402
from prometheus.daemon import loop_ceiling_divergence  # noqa: E402
from tests.support.real_app import BOUNDARY_DOUBLE, build_real_app  # noqa: E402

# Deliberately distinct, and deliberately NOT the shipped defaults: if the
# endpoint echoed config or the shipped constant, these values could not appear.
LOCAL_OVERRIDE = 137
CLOUD_OVERRIDE = 42


def _status(h) -> dict:
    r = h.client.get("/api/status", headers=h.auth())
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_status_reports_the_live_loop_values_not_the_config():
    """The endpoint must report values held by the LIVE LoopContext.

    137/42 exist in no config file and are not the shipped defaults, so an
    endpoint that re-read config or fell back to SHIPPED_MAX_TOOL_ITERATIONS
    could not produce them. This is the whole claim of the change.
    """
    h = build_real_app()
    h.loop_context.max_tool_iterations = LOCAL_OVERRIDE
    h.loop_context.max_tool_iterations_cloud = CLOUD_OVERRIDE

    with h.client:
        block = _status(h)["iteration_ceilings"]

    assert block["wired"] is True
    assert block["local"] == LOCAL_OVERRIDE, (
        f"expected the LIVE context value {LOCAL_OVERRIDE}, got {block['local']} "
        f"— the endpoint is not reading the loop's object"
    )
    assert block["cloud"] == CLOUD_OVERRIDE
    assert block["local"] != SHIPPED_MAX_TOOL_ITERATIONS, (
        "test is not discriminating: the override equals the shipped default"
    )


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_local_and_cloud_are_not_transposed():
    """Distinct values, asserted per field, so a swapped response is caught."""
    h = build_real_app()
    h.loop_context.max_tool_iterations = LOCAL_OVERRIDE
    h.loop_context.max_tool_iterations_cloud = CLOUD_OVERRIDE

    with h.client:
        block = _status(h)["iteration_ceilings"]

    assert (block["local"], block["cloud"]) == (LOCAL_OVERRIDE, CLOUD_OVERRIDE), (
        f"local/cloud transposed: got local={block['local']} cloud={block['cloud']}"
    )


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_null_cloud_means_cloud_uses_the_local_ceiling():
    """None is a MEANINGFUL state, not a missing one.

    _effective_max_tool_iterations short-circuits when the cloud ceiling is
    None and returns the LOCAL value for every provider regardless of tier
    (agent_loop.py:690-691). Coercing it to an int would report a limit the
    loop does not enforce.
    """
    h = build_real_app()
    h.loop_context.max_tool_iterations = LOCAL_OVERRIDE
    h.loop_context.max_tool_iterations_cloud = None

    with h.client:
        block = _status(h)["iteration_ceilings"]

    assert block["cloud"] is None, f"None was coerced to {block['cloud']!r}"
    assert block["cloud_falls_back_to_local"] is True
    assert block["local"] == LOCAL_OVERRIDE


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_unwired_bridge_reports_unwired_rather_than_guessing():
    """No bridge -> say so. Never invent a plausible number for a dark path."""
    h = build_real_app()
    h.app.state.ws_bridge = None

    with h.client:
        block = _status(h)["iteration_ceilings"]

    assert block == {"wired": False, "local": None, "cloud": None}


@pytest.mark.acceptance(allow_doubles=[BOUNDARY_DOUBLE])
def test_existing_status_fields_are_untouched():
    """Additive only — the field must not displace what /api/status already
    reports, which deploy verification depends on."""
    h = build_real_app()
    with h.client:
        body = _status(h)
    for key in ("state", "running_sha", "tree_head", "stale", "provider", "uptime_seconds"):
        assert key in body, f"/api/status lost {key!r}"


# --------------------------------------------------------------------------- #
# the startup agreement check
# --------------------------------------------------------------------------- #


def test_ceiling_divergence_detects_a_mismatch():
    """The invariant that replaces a comment telling humans to stay in step."""
    class _Loop:
        _max_tool_iterations = 500
        _max_tool_iterations_cloud = 100

    class _Ctx:
        max_tool_iterations = 500
        max_tool_iterations_cloud = 100

    assert loop_ceiling_divergence(_Loop(), _Ctx()) == []

    class _Drifted(_Ctx):
        max_tool_iterations_cloud = 500      # the historical bug's shape
    diverged = loop_ceiling_divergence(_Loop(), _Drifted())
    assert diverged == [("max_tool_iterations_cloud", 100, 500)], diverged


def test_ceiling_divergence_treats_a_renamed_attribute_as_divergence():
    """A renamed private attr must surface as divergence, not AttributeError
    that takes the daemon down at boot."""
    class _Ctx:
        max_tool_iterations = 500
        max_tool_iterations_cloud = 100

    out = loop_ceiling_divergence(object(), _Ctx())
    assert len(out) == 2 and all(a is None for _, a, _ in out), out
