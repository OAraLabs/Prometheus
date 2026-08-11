"""Profiles, WIRED — the far side the selector never had.

Selector survey, 2026-08-11 (audits/20260811T222239Z-selector-survey.md,
target 1): ``filter_tools_by_profile`` had zero callers in src, ``/profile``
on three gateways stored a name nothing read back, ``profiles.default`` fed
a cosmetic UI string, and the builtin lists named tools that have never
existed (``file_read`` for ``read_file``, an ``lsp`` tool that was always a
post-result hook) — wrong from birth and undetectable, because a selector
with no consumer has no far side for a test to stand on.

These tests ARE that far side, in the advertised_names() tradition: they
stand where the consumer stands and assert what it receives.

* the provider request's ``tools`` under a non-full profile (through the
  real run_loop, against the real registry);
* a switch through the shared ActiveProfileState reaching the NEXT run of
  the same long-lived context — the per-call-parameters principle, and the
  property the daemon actually needs (Beacon and /profile switch mid-life);
* the loud unfiltered fallback when a profile filters everything away
  (advertising zero tools is the vault_search failure shape: the model
  concludes the capability does not exist);
* every builtin tool name pinned to the registry the daemon builds, so the
  file_read class of drift can never ship again;
* PUT /api/profiles/active mutating the holder the loops resolve — not a
  display string.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prometheus.__main__ import create_tool_registry
from prometheus.config.profiles import (
    ActiveProfileState,
    AgentProfile,
    ProfileStore,
    _BUILTINS,
)
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.web.server import create_app


class _RecordingProvider(ModelProvider):
    """One text round; records every request — the far side IS the request."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        msg = ConversationMessage(role="assistant", content=[TextBlock(text="ok")])
        yield ApiTextDeltaEvent(text="ok")
        yield ApiMessageCompleteEvent(
            message=msg, usage=UsageSnapshot(), stop_reason="stop"
        )


def _drive(context: LoopContext) -> None:
    async def _run():
        async for _event, _usage in run_loop(
            context, [ConversationMessage.from_user_text("go")]
        ):
            pass

    asyncio.run(_run())


def _advertised(provider: _RecordingProvider, call: int = 0) -> set[str]:
    return {t["name"] for t in provider.requests[call].tools}


def _context(provider, resolver=None, telemetry=None) -> LoopContext:
    return LoopContext(
        provider=provider,
        model="test",
        system_prompt="BASE",
        max_tokens=64,
        tool_registry=create_tool_registry({}),
        session_id="web",
        profile_resolver=resolver,
        telemetry=telemetry,
    )


# ---------------------------------------------------------------------------
# The far side: what the provider request advertises
# ---------------------------------------------------------------------------

def test_the_provider_sees_the_research_catalog_not_the_full_one():
    """Through the real run_loop against the real registry: a research
    profile advertises exactly its (registered) names — a strict subset."""
    research = _BUILTINS["research"]
    provider = _RecordingProvider()
    _drive(_context(provider, resolver=lambda: research))

    advertised = _advertised(provider)
    registry_names = {t.name for t in create_tool_registry({}).list_tools()}
    assert advertised == set(research.tools) & registry_names, (
        f"advertised {sorted(advertised)}"
    )
    assert advertised < registry_names, "research should be a strict subset"
    assert "write_file" not in advertised, "research is a no-mutation profile"


def test_no_resolver_and_full_profile_both_advertise_everything():
    """The negative twin, both flavors: no resolver, and the full profile
    (tools=None) — each advertises the whole registry."""
    registry_names = {t.name for t in create_tool_registry({}).list_tools()}

    bare = _RecordingProvider()
    _drive(_context(bare, resolver=None))
    assert _advertised(bare) == registry_names

    full = _RecordingProvider()
    _drive(_context(full, resolver=lambda: _BUILTINS["full"]))
    assert _advertised(full) == registry_names


def test_a_profile_switch_reaches_the_next_run_without_reconstruction():
    """The property the daemon needs: BOTH its contexts are long-lived, so a
    /profile or Beacon switch must reach the next run through the SAME
    context object. This is why the field is a resolver, not a profile."""
    store = ProfileStore()
    state = ActiveProfileState(store, "research")
    provider = _RecordingProvider()
    context = _context(provider, resolver=state.get)

    _drive(context)
    state.set("minimal")
    _drive(context)

    assert _advertised(provider, 0) == set(_BUILTINS["research"].tools)
    assert _advertised(provider, 1) == set(_BUILTINS["minimal"].tools), (
        "the switch did not reach the second run — resolver is being frozen"
    )


def test_a_profile_that_filters_everything_falls_back_loud(caplog):
    """A profile whose names match nothing is a config error. Advertising
    zero tools would be the vault_search shape (the model concludes the
    capability does not exist), so the loop advertises UNFILTERED and says
    so at ERROR."""
    broken = AgentProfile(name="broken", tools=["no_such_tool_anywhere"])
    provider = _RecordingProvider()
    with caplog.at_level("ERROR"):
        _drive(_context(provider, resolver=lambda: broken))

    registry_names = {t.name for t in create_tool_registry({}).list_tools()}
    assert _advertised(provider) == registry_names, "fallback must be unfiltered"
    assert any("filtered all" in r.message for r in caplog.records), (
        "the fallback must be loud — a silent fallback re-hides the config error"
    )


def test_a_crashing_resolver_does_not_kill_the_run(caplog):
    provider = _RecordingProvider()

    def _boom():
        raise RuntimeError("resolver exploded")

    with caplog.at_level("WARNING"):
        _drive(_context(provider, resolver=_boom))
    assert provider.requests, "the run must survive a resolver failure"
    assert _advertised(provider) == {
        t.name for t in create_tool_registry({}).list_tools()
    }


def test_advertisement_telemetry_records_the_profile():
    """The A/B row states WHICH profile shaped the catalog — 'advertised: 9'
    is uninterpretable without it."""
    telemetry = MagicMock()
    provider = _RecordingProvider()
    _drive(_context(
        provider, resolver=lambda: _BUILTINS["research"], telemetry=telemetry,
    ))
    runs = [
        kw for _, kw in [
            (c.args, c.kwargs) for c in telemetry.record_run.call_args_list
        ]
        if kw.get("operation") == "tool_advertisement"
    ]
    assert runs, "no tool_advertisement telemetry row"
    assert runs[0]["summary"]["profile"] == "research"
    assert runs[0]["summary"]["advertised"] == len(_BUILTINS["research"].tools)


# ---------------------------------------------------------------------------
# The §1d guard: builtin names are registry names
# ---------------------------------------------------------------------------

# Tools that are real but CONDITIONALLY registered — outside
# create_tool_registry, so the guard below must not demand them there. Same
# policy as every carve-out in this suite: a documented reason, and a check
# that fails when the reason stops being true.
KNOWN_CONDITIONAL: dict[str, tuple[str, str]] = {
    # name -> (module, class): run_daemon registers LSPTool only when
    # lsp.enabled (daemon.py ~:418). When unregistered, the name simply drops
    # out of the advertisement intersection — harmless in a profile list.
    "lsp": ("prometheus.tools.builtin.lsp", "LSPTool"),
}


def test_every_builtin_profile_tool_name_exists_in_the_real_registry():
    """The guard that was impossible before wiring: with no far side, the
    original lists said file_read/file_write/file_edit for tools registered
    as read_file/write_file/edit_file, and nothing could notice. Now a
    drifted name fails the build."""
    registry_names = {t.name for t in create_tool_registry({}).list_tools()}
    for profile in _BUILTINS.values():
        if profile.tools is None:
            continue
        missing = set(profile.tools) - registry_names - set(KNOWN_CONDITIONAL)
        assert not missing, (
            f"builtin profile {profile.name!r} names tools that are not in "
            f"the registry the daemon builds: {sorted(missing)} — fix the "
            f"name, or add it to KNOWN_CONDITIONAL with a checkable reason"
        )
    # Oracle sanity — the registry read can tell fake from real.
    assert "definitely_not_a_tool" not in registry_names
    assert "read_file" in registry_names


def test_the_conditional_carveouts_are_still_real():
    """Each KNOWN_CONDITIONAL entry must resolve to an actual tool class
    whose registered name matches — and must not have become unconditional
    (present in the bare registry), which would make the carve-out stale."""
    import importlib

    registry_names = {t.name for t in create_tool_registry({}).list_tools()}
    for name, (module, cls_name) in KNOWN_CONDITIONAL.items():
        cls = getattr(importlib.import_module(module), cls_name)
        assert cls.name == name, (
            f"KNOWN_CONDITIONAL[{name!r}] points at {cls_name}, whose real "
            f"name is {cls.name!r} — the carve-out no longer matches"
        )
        assert name not in registry_names, (
            f"{name!r} is now in the bare registry — remove it from "
            f"KNOWN_CONDITIONAL"
        )


# ---------------------------------------------------------------------------
# ActiveProfileState — one holder, every surface
# ---------------------------------------------------------------------------

def test_active_profile_state_semantics():
    store = ProfileStore()
    state = ActiveProfileState(store, "coder")
    assert state.name == "coder"
    assert state.get().name == "coder"

    assert state.set("nope") is None
    assert state.name == "coder", "an unknown name must not change state"

    assert state.set("minimal").name == "minimal"
    assert state.name == "minimal"


def test_an_unknown_default_falls_back_to_full():
    state = ActiveProfileState(ProfileStore(), "typo-profile")
    assert state.name == "full"
    assert state.get().tools is None


# ---------------------------------------------------------------------------
# The web route mutates the holder the loops resolve
# ---------------------------------------------------------------------------

def test_the_put_route_mutates_the_holder_the_loops_resolve():
    """Before the holder, PUT /api/profiles/active set a cosmetic string that
    nothing downstream read. Now it sets the state run_loop resolves."""
    store = ProfileStore()
    state = ActiveProfileState(store, "full")
    client = TestClient(create_app({}, profile_store=store, profile_state=state))

    resp = client.put("/api/profiles/active", json={"name": "research"})
    assert resp.status_code == 200
    assert state.name == "research", "the route must mutate the shared holder"

    active = [p["name"] for p in client.get("/api/profiles").json() if p["is_active"]]
    assert active == ["research"]

    assert client.put("/api/profiles/active", json={"name": "nope"}).status_code == 404
    assert state.name == "research", "a 404 must leave the holder untouched"


def test_the_status_payload_reports_the_holder():
    store = ProfileStore()
    state = ActiveProfileState(store, "coder")
    client = TestClient(create_app({}, profile_store=store, profile_state=state))
    state.set("minimal")  # a gateway switch, not the web route
    assert client.get("/api/status").json()["profile"] == "minimal"


# ---------------------------------------------------------------------------
# The daemon passes the resolver at both construction sites
# ---------------------------------------------------------------------------

def test_the_daemon_passes_the_resolver_at_both_sites():
    """Named pin on top of the generic two-loop drift guard: profile_resolver
    is a variable kwarg (not config-backed), so the CLI-config invariant
    cannot see it — this can."""
    from tests.test_web_bridge_loop_parity import _kwargs_by_callee

    kwargs = _kwargs_by_callee()
    assert "profile_resolver" in kwargs["AgentLoop"]
    assert "profile_resolver" in kwargs["LoopContext"]
