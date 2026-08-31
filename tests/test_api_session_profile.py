"""Per-session agent profile (branch feat/per-session-profile).

Roadmap item 6, honest version. The original framing — "pick a project, the session gets its
workspace root and instructions" — maps onto nothing: `projects` is the Kanban board's table
(id/name/description/colour) and profiles are a DAEMON-WIDE capability preset. What does exist is
the profile, so this makes the profile per-session instead of inventing two concepts.

Durable on purpose: the model router's per-session overrides live in a dict on the router and
vanish on restart. A setting that silently reverts is the defect class this whole file avoids.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.profiles import ActiveProfileState, get_profile_store  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _Engine:
    def __init__(self, store):
        self.conversation_store = store


@pytest.fixture
def ctx(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    state = ActiveProfileState(get_profile_store(), "full", session_lookup=store.get_session_profile)
    app = create_app({}, lcm_engine=_Engine(store), profile_state=state)
    return TestClient(app), store, state


# ── the binding is durable, and absence means "follow the global" ───────────────────────────

def test_a_session_starts_following_the_global_profile(ctx):
    c, _, state = ctx
    body = c.get("/api/sessions/s1/profile").json()
    assert body["profile"] is None
    assert body["source"] == "global"
    assert body["effective"] == state.name


def test_binding_a_profile_changes_what_a_TURN_resolves(ctx):
    """The load-bearing assertion: a stored binding must reach the agent loop's resolver.

    Without this the feature is inert — the row is written, the client shows it, and every turn
    still runs the global profile. That is the shape that looks delivered from both ends.
    """
    c, _, state = ctx
    assert state.get("s1").name == state.name          # before
    assert c.put("/api/sessions/s1/profile", json={"profile": "coder"}).json()["ok"] is True
    assert state.get("s1").name == "coder"             # after — the resolver honours it
    assert state.get("other").name == state.name       # and no other session moved
    assert state.get().name == state.name              # nor the global default


def test_the_binding_survives_a_new_store_instance(ctx, tmp_path):
    """A restart must not silently revert it — the router's in-memory overrides do."""
    c, _, _ = ctx
    c.put("/api/sessions/s1/profile", json={"profile": "coder"})
    reopened = LCMConversationStore(tmp_path / "lcm.db")
    assert reopened.get_session_profile("s1") == "coder"


def test_blank_clears_the_binding(ctx):
    c, store, state = ctx
    c.put("/api/sessions/s1/profile", json={"profile": "coder"})
    body = c.put("/api/sessions/s1/profile", json={"profile": ""}).json()
    assert body["profile"] is None
    assert c.get("/api/sessions/s1/profile").json()["source"] == "global"
    assert state.get("s1").name == state.name
    # Clearing must REMOVE the row, not store "". Both make the turn resolve globally, so the
    # only place the difference shows is here — and an empty string is a present-but-falsy value
    # that a later reader can mistake for a choice, in a table that would grow a row per clear.
    assert store.get_session_profile("s1") is None, 'blank left a row behind instead of deleting it'
    assert c.get("/api/sessions/s1/profile").json()["profile"] is None


def test_source_distinguishes_chosen_from_inherited(ctx):
    """A client cannot offer "reset to automatic" for a value it cannot tell from a default."""
    c, _, state = ctx
    c.put("/api/sessions/s1/profile", json={"profile": state.name})  # SAME name as the global
    body = c.get("/api/sessions/s1/profile").json()
    assert body["effective"] == state.name
    assert body["source"] == "session", "explicitly choosing the default is still a choice"


# ── validation ──────────────────────────────────────────────────────────────────────────────

def test_an_unknown_profile_is_refused_with_the_valid_names(ctx):
    c, store, _ = ctx
    r = c.put("/api/sessions/s1/profile", json={"profile": "coderr"})
    assert r.status_code == 400
    assert "coder" in r.json()["known"], "the refusal names what WOULD work"
    assert store.get_session_profile("s1") is None, "and nothing was stored"


def test_a_non_string_profile_is_refused(ctx):
    c, _, _ = ctx
    assert c.put("/api/sessions/s1/profile", json={"profile": 7}).status_code == 400


def test_a_dangling_binding_falls_back_and_is_reported(ctx):
    """A profile can be deleted after a session was bound to it."""
    c, store, state = ctx
    store.set_session_profile("s1", "deleted-profile")   # bypass validation, as a rename would
    assert state.get("s1").name == state.name, "the turn falls back rather than advertising nothing"
    body = c.get("/api/sessions/s1/profile").json()
    assert body["dangling"] is True, "and the client is told, rather than shown a silent fallback"
    assert body["source"] == "global"


def test_no_store_degrades_instead_of_500(tmp_path):
    app = create_app({})
    c = TestClient(app)
    assert c.put("/api/sessions/s1/profile", json={"profile": "coder"}).status_code == 503
