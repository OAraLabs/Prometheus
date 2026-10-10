"""Device scoping, unit level — the ownership record and the policy table.

The behavioural tests (test_device_session_scoping_*.py) go through the app and the
bridge. These pin the two pieces they stand on, in isolation: DeviceStore's session
ownership rows (config/device_store.py) and the decisions in web/session_scope.py.
"""

from __future__ import annotations

import sqlite3

import pytest

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity
from prometheus.config.device_store import DeviceStore
from prometheus.web.session_scope import (
    NOBODY,
    OPERATOR,
    Scope,
    SessionAccess,
    event_session_id,
    gateway_of,
    scope_for,
)


def _tables(path) -> set[str]:
    con = sqlite3.connect(str(path))
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


@pytest.fixture
def store(tmp_path) -> DeviceStore:
    return DeviceStore(tmp_path / "devices.db")


# --------------------------------------------------------------------------- #
# DeviceStore: who owns a session
# --------------------------------------------------------------------------- #


def test_the_first_claim_wins_and_is_never_reassigned(store):
    a, b = store.mint("a", "ios")["id"], store.mint("b", "ios")["id"]
    assert store.session_owner("s1") is None

    assert store.claim_session("s1", a) is True
    assert store.session_owner("s1") == a
    assert store.claim_session("s1", a) is True       # idempotent for the owner
    assert store.claim_session("s1", b) is False      # a second device cannot take it
    assert store.session_owner("s1") == a


def test_only_a_live_device_can_claim(store):
    gone = store.mint("gone", "ios")["id"]
    store.revoke(gone)
    assert store.claim_session("s1", gone) is False
    assert store.claim_session("s2", "no-such-device") is False
    assert store.claim_session("", store.mint("c", "ios")["id"]) is False
    assert store.session_owner("s1") is None and store.session_owner("s2") is None


def test_revoking_a_device_does_not_free_its_sessions(store):
    a, b = store.mint("a", "ios")["id"], store.mint("b", "ios")["id"]
    store.claim_session("s1", a)
    store.revoke(a)
    assert store.session_owner("s1") == a
    assert store.claim_session("s1", b) is False


def test_owned_session_ids_are_per_device(store):
    a, b = store.mint("a", "ios")["id"], store.mint("b", "ios")["id"]
    for sid in ("s1", "s2"):
        store.claim_session(sid, a)
    store.claim_session("s3", b)
    assert store.owned_session_ids(a) == {"s1", "s2"}
    assert store.owned_session_ids(b) == {"s3"}
    assert store.owned_session_ids("nobody") == set()


def test_ownership_survives_reopening_the_database(tmp_path):
    first = DeviceStore(tmp_path / "devices.db")
    a = first.mint("a", "ios")["id"]
    first.claim_session("s1", a)
    first.close()
    again = DeviceStore(tmp_path / "devices.db")
    assert again.session_owner("s1") == a


def test_the_ownership_table_appears_only_with_the_first_claim(tmp_path):
    """The parity fixtures record every table in devices.db, so a daemon that never
    scopes a device must not grow one (the computer_devices precedent)."""
    store = DeviceStore(tmp_path / "devices.db")
    a = store.mint("a", "ios")["id"]
    # Reading never creates it …
    assert store.session_owner("s1") is None and store.owned_session_ids(a) == set()
    assert "device_sessions" not in _tables(tmp_path / "devices.db")
    # … a failed claim does not either …
    store.claim_session("s1", "no-such-device")
    assert "device_sessions" not in _tables(tmp_path / "devices.db")
    # … only a real one.
    store.claim_session("s1", a)
    assert "device_sessions" in _tables(tmp_path / "devices.db")


def test_a_fresh_devices_db_has_exactly_the_tables_it_had_before_scoping(tmp_path):
    DeviceStore(tmp_path / "devices.db")
    assert _tables(tmp_path / "devices.db") == {"api_devices", "activity_tokens"}


def test_a_lock_error_is_not_read_as_unowned(tmp_path):
    """session_owner swallows 'no such table' and nothing else: a failing database must
    not turn into 'nobody owns this', which would hand an operator-only session's id to
    the next claimant."""

    class _Locked:
        def execute(self, *a, **kw):
            raise sqlite3.OperationalError("database is locked")

    store = DeviceStore(tmp_path / "devices.db")
    store._conn = _Locked()  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        store.session_owner("s1")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        store.owned_session_ids("d1")


# --------------------------------------------------------------------------- #
# scope_for / gateway_of / event_session_id
# --------------------------------------------------------------------------- #


def test_scope_for_every_caller():
    dev = DeviceIdentity(id="dev1", name="phone", platform="ios")
    assert scope_for(GLOBAL_IDENTITY, auth_required=True) == OPERATOR
    assert scope_for(dev, auth_required=True) == Scope("dev1")
    # Auth off: nobody is restricted, with or without an identity.
    assert scope_for(None, auth_required=False) == OPERATOR
    assert scope_for(dev, auth_required=False) == OPERATOR
    # Auth on and nobody identified: fail closed, never the operator.
    assert scope_for(None, auth_required=True) == NOBODY
    assert not NOBODY.unrestricted and OPERATOR.unrestricted


@pytest.mark.parametrize("sid,expected", [
    ("telegram:123", "telegram"), ("Telegram:9", "telegram"), ("desktop:abc:def", "desktop"),
    ("my-session", ""), ("", ""), (":x", ""),
])
def test_gateway_of(sid, expected):
    assert gateway_of(sid) == expected


@pytest.mark.parametrize("event,expected", [
    ({"type": "chat_delta", "payload": {"session_id": "s1"}}, "s1"),
    ({"type": "approval_pending", "payload": {"session_id": "s2", "request_id": "r"}}, "s2"),
    # the generic wrapper nests the signal's own payload
    ({"type": "sentinel_signal", "payload": {"kind": "k", "payload": {"session_id": "s3"}}}, "s3"),
    ({"type": "dream_start", "payload": {"phase": 1}}, None),
    ({"type": "x", "payload": {"session_id": ""}}, None),
    ({"type": "x", "payload": {"session_id": 7}}, None),
    ({"type": "x", "payload": {"payload": "not a dict"}}, None),
    ({"type": "x", "payload": None}, None),
    ({"type": "x"}, None),
])
def test_event_session_id(event, expected):
    assert event_session_id(event) == expected


# --------------------------------------------------------------------------- #
# SessionAccess: the decision table
# --------------------------------------------------------------------------- #


class _Owners:
    """The two DeviceStore methods the policy calls, over a dict."""

    def __init__(self, owners: dict[str, str]) -> None:
        self.owners = dict(owners)

    def session_owner(self, sid):
        return self.owners.get(sid)

    def claim_session(self, sid, device_id):
        return self.owners.setdefault(sid, device_id) == device_id

    def owned_session_ids(self, device_id):
        return {s for s, d in self.owners.items() if d == device_id}


def _access(owners: dict[str, str], existing: set[str] | None = None):
    store = _Owners(owners)
    return SessionAccess(lambda: store, lambda sid: sid in (existing or set())), store


A, B = Scope("A"), Scope("B")


def test_owns_is_exact_and_never_claims():
    access, store = _access({"s1": "A"})
    assert access.owns(A, "s1") and not access.owns(B, "s1")
    assert not access.owns(A, "unclaimed") and not access.owns(NOBODY, "s1")
    assert access.owns(OPERATOR, "anything-at-all")
    assert store.owners == {"s1": "A"}


def test_may_use_is_pure_and_admit_claims():
    access, store = _access({})
    assert access.may_use(A, "ios:new") is True
    assert store.owners == {}                         # may_use claimed nothing
    assert access.admit(A, "ios:new") is True
    assert store.owners == {"ios:new": "A"}
    assert access.admit(B, "ios:new") is False        # now A's
    assert access.admit(A, "ios:new") is True         # and still A's


@pytest.mark.parametrize("sid", ["telegram:1", "SLACK:C", "discord:2", "cli:3", "api:4", "coding:t", "system"])
def test_a_device_cannot_take_a_daemon_namespace(sid):
    access, store = _access({})
    assert access.admit(A, sid) is False and access.may_use(A, sid) is False
    assert store.owners == {}
    assert access.admit(OPERATOR, sid) is True


def test_a_device_cannot_take_an_id_that_already_exists():
    access, store = _access({}, existing={"desktop:old", "my-chat"})
    assert access.admit(A, "desktop:old") is False
    assert access.admit(A, "my-chat") is False
    assert store.owners == {}
    assert access.admit(A, "desktop:new") is True


def test_mint_skips_the_namespace_and_existence_checks_for_server_made_ids():
    access, store = _access({}, existing={"telegram:uuid"})
    assert access.mint(A, "telegram:uuid") is True    # the server made it; nothing can collide
    assert store.owners == {"telegram:uuid": "A"}
    assert access.mint(B, "telegram:uuid") is False   # but ownership is still exclusive
    assert access.mint(NOBODY, "x") is False and access.mint(OPERATOR, "x") is True


@pytest.mark.parametrize("junk", [7, None, ["x"], {"a": 1}, 1.5])
def test_an_id_that_is_not_a_string_is_nobodys(junk):
    """Session ids arrive as whatever the client's JSON said; the policy must not raise on them."""
    access, store = _access({})
    assert access.owns(A, junk) is False and access.may_use(A, junk) is False
    assert access.admit(A, junk) is False and access.mint(A, junk) is False
    assert store.owners == {}


def test_nobody_and_the_empty_id_get_nothing():
    access, store = _access({})
    for scope in (NOBODY, A):
        assert access.admit(scope, "") is False
    assert access.admit(NOBODY, "ios:x") is False and store.owners == {}


def test_owned_ids_and_frame_visibility():
    access, _ = _access({"s1": "A", "s2": "B"})
    assert access.owned_ids(OPERATOR) is None
    assert access.owned_ids(A) == {"s1"}
    assert access.owned_ids(NOBODY) == set()

    mine = {"type": "chat_delta", "payload": {"session_id": "s1"}}
    theirs = {"type": "chat_delta", "payload": {"session_id": "s2"}}
    nested = {"type": "sentinel_signal", "payload": {"kind": "k", "payload": {"session_id": "s2"}}}
    nobodys = {"type": "chat_delta", "payload": {"session_id": "telegram:1"}}
    daemon = {"type": "dream_start", "payload": {}}
    assert access.frame_visible(A, mine) and access.frame_visible(OPERATOR, theirs)
    assert not access.frame_visible(A, theirs)
    assert not access.frame_visible(A, nested)
    assert not access.frame_visible(A, nobodys)
    assert access.frame_visible(A, daemon) and access.frame_visible(NOBODY, daemon)
    assert not access.frame_visible(NOBODY, mine)
