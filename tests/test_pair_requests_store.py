"""``config/pair_requests.py`` — the pairing request table and its state machine.

A request is a stranger's ask to join, held for five minutes while the owner decides. What is pinned is what
makes that safe to hold:

* the poll secret is never stored (only its SHA-256), and the plaintext token is never stored anywhere: it
  exists in memory for the length of the approve call and leaves sealed to the requester's key;
* every transition is ONE guarded statement, so two decisions at once have exactly one winner, tested with
  two real SQLite connections on one file (the CLI and the daemon are two processes);
* approving mints the device and records the approval in ONE transaction, so there is never a live token
  nobody holds, and a failure while sealing leaves no device behind;
* the device an approval mints is an ordinary, scoped one: never an owner (no ``owner_devices`` row, not
  even the table);
* the table appears on first use, so a box that never pairs a device keeps the exact ``devices.db`` schema
  the parity fixtures record;
* a device nobody collects within five minutes of approval is revoked, through the registry, so its sockets
  close with it.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from prometheus.config import pair_requests as pr
from prometheus.config import pair_seal
from prometheus.config.device_store import DeviceStore, token_digest

INSTANCE_DER = bytes.fromhex(
    "3059301306072a8648ce3d020106082a8648ce3d03010703420004" + "11" * 32 + "22" * 32)


class Clock:
    def __init__(self, now: float = 1_760_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class Requester:
    """The new device: a key pair, the base64url public half, and what it can unseal."""

    def __init__(self) -> None:
        self.private = X25519PrivateKey.generate()
        self.public_b64 = pair_seal.b64url_encode(
            self.private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))

    def unseal(self, request_id: str, sealed: dict) -> dict:
        return pair_seal.unseal_token(private_key=self.private, request_id=request_id, sealed=sealed)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def devices(tmp_path):
    return DeviceStore(tmp_path / "devices.db")


@pytest.fixture
def store(devices, clock):
    return pr.PairRequestStore(devices, settings=pr.PairingSettings(), clock=clock)


def _create(store, *, source="192.0.2.10", name="Jennifer's MacBook", platform="macos", requester=None):
    requester = requester or Requester()
    created = store.create(device_name=name, platform=platform, public_key=requester.public_b64,
                           source=source, instance_public_key_der=INSTANCE_DER)
    return created, requester


def _tables(path) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}


def _db_bytes(path) -> bytes:
    return b"".join(p.read_bytes() for p in (Path(path), Path(f"{path}-journal"), Path(f"{path}-wal")) if p.exists())


# ── settings ─────────────────────────────────────────────────────────────────

def test_the_defaults_are_the_contracts():
    s = pr.PairingSettings()
    assert (s.requests_enabled, s.request_ttl_seconds, s.max_pending, s.max_pending_per_source,
            s.max_requests_per_source_per_hour) == (True, 300, 3, 1, 10)


def test_settings_are_read_from_the_pairing_section_in_one_place():
    s = pr.PairingSettings.from_config({"pairing": {
        "requests_enabled": False, "request_ttl_seconds": 120, "max_pending": 5,
        "max_pending_per_source": 2, "max_requests_per_source_per_hour": 20}})
    assert (s.requests_enabled, s.request_ttl_seconds, s.max_pending, s.max_pending_per_source,
            s.max_requests_per_source_per_hour) == (False, 120, 5, 2, 20)
    assert pr.PairingSettings.from_config(None) == pr.PairingSettings()
    assert pr.PairingSettings.from_config({"pairing": None}) == pr.PairingSettings()


@pytest.mark.parametrize("given, used", [(30, 60), (60, 60), (900, 900), (5000, 900)])
def test_the_ttl_is_clamped_to_60_to_900_seconds_out_loud(given, used, caplog):
    with caplog.at_level("WARNING"):
        s = pr.PairingSettings.from_config({"pairing": {"request_ttl_seconds": given}})
    assert s.request_ttl_seconds == used
    assert (given == used) or any("request_ttl_seconds" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("junk", ["soon", True, 0, -3, 2.5])
def test_an_unusable_count_falls_back_to_the_default_out_loud(junk, caplog):
    with caplog.at_level("WARNING"):
        s = pr.PairingSettings.from_config({"pairing": {"max_pending": junk}})
    assert s.max_pending == 3
    assert any("max_pending" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("junk", ["soon", True, 2.5])
def test_an_unusable_ttl_falls_back_to_300_out_loud(junk, caplog):
    with caplog.at_level("WARNING"):
        s = pr.PairingSettings.from_config({"pairing": {"request_ttl_seconds": junk}})
    assert s.request_ttl_seconds == 300
    assert any("request_ttl_seconds" in r.getMessage() for r in caplog.records)


def test_a_key_left_empty_in_yaml_is_unset_not_an_error(caplog):
    with caplog.at_level("WARNING"):
        s = pr.PairingSettings.from_config({"pairing": {"max_pending": None, "request_ttl_seconds": None}})
    assert s == pr.PairingSettings() and not caplog.records


# ── the table appears on first use ───────────────────────────────────────────

def test_reading_creates_nothing(devices, store, tmp_path):
    before = _tables(tmp_path / "devices.db")
    assert store.pending() == []
    assert store.get("a" * 32) is None
    assert store.sweep().purged == 0
    assert _tables(tmp_path / "devices.db") == before
    assert "pair_requests" not in before


def test_the_first_request_creates_the_table(store, tmp_path):
    _create(store)
    assert "pair_requests" in _tables(tmp_path / "devices.db")


# ── a request ────────────────────────────────────────────────────────────────

def test_a_request_has_a_128_bit_id_a_256_bit_secret_and_the_documented_code(store, clock):
    created, requester = _create(store)
    req = created.request
    assert len(req.id) == 32 and set(req.id) <= set("0123456789abcdef")
    assert len(created.poll_secret) == 43 and pair_seal.b64url_decode(created.poll_secret).__len__() == 32
    assert req.state == "pending" and req.expires_at == clock.now + 300 and req.created_at == clock.now
    assert req.match_code == pair_seal.match_code(
        pair_seal.b64url_decode(requester.public_b64), INSTANCE_DER, req.id)
    assert (req.device_name, req.platform, req.source) == ("Jennifer's MacBook", "macos", "192.0.2.10")


def test_two_requests_never_share_an_id_or_a_secret(store):
    a, _ = _create(store, source="192.0.2.1")
    b, _ = _create(store, source="192.0.2.2")
    assert a.request.id != b.request.id and a.poll_secret != b.poll_secret


def test_the_poll_secret_is_never_stored(store, tmp_path):
    created, _ = _create(store)
    assert created.poll_secret.encode() not in _db_bytes(tmp_path / "devices.db")


def test_the_secret_is_checked_in_constant_time_and_unknown_looks_like_wrong(store, monkeypatch):
    created, _ = _create(store)
    seen = []
    real = pr.hmac.compare_digest
    monkeypatch.setattr(pr.hmac, "compare_digest", lambda a, b: seen.append(1) or real(a, b))
    assert store.verify(created.request.id, created.poll_secret).id == created.request.id
    assert store.verify(created.request.id, "x" * 43) is None
    assert store.verify("f" * 32, created.poll_secret) is None
    assert store.verify("f" * 32, "") is None
    assert len(seen) == 4, "an unknown id still pays for a comparison, so it cannot be told from a wrong secret by time"


# ── limits ───────────────────────────────────────────────────────────────────

def test_one_pending_request_per_source(store):
    _create(store, source="192.0.2.10")
    with pytest.raises(pr.LimitExceeded) as caught:
        _create(store, source="192.0.2.10")
    assert caught.value.reason == "per_source_pending" and caught.value.retry_after >= 1
    _create(store, source="192.0.2.11")   # another source is not affected


def test_three_pending_overall(store):
    for n in range(3):
        _create(store, source=f"192.0.2.{n + 1}")
    with pytest.raises(pr.LimitExceeded) as caught:
        _create(store, source="192.0.2.99")
    assert caught.value.reason == "pending_full"


def test_ten_requests_per_source_per_hour(devices, clock):
    store = pr.PairRequestStore(devices, settings=pr.PairingSettings(max_pending_per_source=50, max_pending=50),
                                clock=clock)
    for _ in range(10):
        created, _ = _create(store, source="192.0.2.10")
        store.cancel(created.request.id)
        clock.now += 60
    with pytest.raises(pr.LimitExceeded) as caught:
        _create(store, source="192.0.2.10")
    assert caught.value.reason == "hourly"
    # The first request (11 minutes ago) ages out of the hour: the hint says when.
    assert 1 <= caught.value.retry_after <= 3600
    clock.now += 3600
    _create(store, source="192.0.2.10")


def test_an_expired_request_frees_its_slot(store, clock):
    _create(store, source="192.0.2.10")
    clock.now += 301
    _create(store, source="192.0.2.10")


def test_a_refused_request_leaves_no_row_behind(store):
    _create(store, source="192.0.2.10")
    with pytest.raises(pr.LimitExceeded):
        _create(store, source="192.0.2.10")
    assert len(store.pending()) == 1


# ── expiry ───────────────────────────────────────────────────────────────────

def test_a_request_lives_exactly_its_ttl(store, clock):
    created, _ = _create(store)
    rid = created.request.id
    clock.now += 299.9
    assert store.get(rid).state == "pending" and [r.id for r in store.pending()] == [rid]
    clock.now += 0.2
    assert store.get(rid).state == "expired" and store.pending() == []


def test_deciding_after_the_ttl_is_expired_not_a_quiet_success(store, clock, devices):
    created, _ = _create(store)
    clock.now += 301
    with pytest.raises(pr.RequestExpired):
        store.approve(created.request.id, name=None, decided_by="beacon:test")
    with pytest.raises(pr.RequestExpired):
        store.deny(created.request.id, decided_by="beacon:test")
    assert devices.list_devices() == []


def test_a_decision_after_the_sweep_recorded_the_expiry_is_still_expired_not_somebody_elses(store, clock):
    """The sweep runs at the start of every request, so it usually records the expiry before the decision
    arrives; the answer must not change from 'too late' (410) to 'someone decided first' (409)."""
    created, _ = _create(store)
    clock.now += 301
    store.sweep()
    assert store.get(created.request.id).state == "expired"
    with pytest.raises(pr.RequestExpired):
        store.approve(created.request.id, name=None, decided_by="beacon:x")
    with pytest.raises(pr.RequestExpired):
        store.deny(created.request.id, decided_by="beacon:x")


def test_deciding_a_request_whose_device_was_revoked_for_going_uncollected_says_it_was_approved(store, clock):
    created, _ = _create(store)
    store.approve(created.request.id, name=None, decided_by="beacon:x")
    clock.now += 301
    store.sweep()
    with pytest.raises(pr.NotPending) as caught:
        store.approve(created.request.id, name=None, decided_by="telegram:1")
    assert caught.value.status == "approved"


def test_the_sweep_records_expiry_and_says_which_requests(store, clock):
    created, _ = _create(store)
    clock.now += 301
    swept = store.sweep()
    assert [r.id for r in swept.expired] == [created.request.id]
    assert store.get(created.request.id).state == "expired"
    assert store.sweep().expired == [], "a request is reported expired once"


# ── approve ──────────────────────────────────────────────────────────────────

def test_approving_mints_an_ordinary_scoped_device_never_an_owner(store, devices, tmp_path):
    created, requester = _create(store, name="Jennifer's MacBook", platform="windows")
    approved = store.approve(created.request.id, name=None, decided_by="beacon:Will's Mac")
    device = next(d for d in devices.list_devices() if d.id == approved.device_id)
    assert device.name == "Jennifer's MacBook"
    assert device.platform == "other", "the registry keeps ios/macos/other; the request row keeps 'windows'"
    assert store.get(created.request.id).platform == "windows"
    assert devices.is_owner(device.id) is False
    assert "owner_devices" not in _tables(tmp_path / "devices.db"), "an approval must not even create the owner table"


def test_an_operator_may_rename_at_approval(store, devices):
    created, _ = _create(store, name="Jennifer's MacBook")
    approved = store.approve(created.request.id, name="Jennifer's MacBook (kitchen)", decided_by="beacon:x")
    assert next(d for d in devices.list_devices() if d.id == approved.device_id).name == "Jennifer's MacBook (kitchen)"


def test_the_token_leaves_sealed_and_authenticates_the_new_device(store, devices):
    created, requester = _create(store)
    approved = store.approve(created.request.id, name=None, decided_by="beacon:x")
    sealed = store.sealed_blob(created.request.id)
    opened = requester.unseal(created.request.id, sealed)
    assert set(opened) == {"token", "device_id", "name"}
    assert opened["device_id"] == approved.device_id
    assert devices.lookup(token_digest(opened["token"])).id == approved.device_id
    assert not hasattr(approved, "token"), "the store's own return value must not carry the plaintext token"


def test_the_plaintext_token_is_never_at_rest(store, devices, tmp_path):
    created, requester = _create(store)
    store.approve(created.request.id, name=None, decided_by="beacon:x")
    token = requester.unseal(created.request.id, store.sealed_blob(created.request.id))["token"]
    assert token.encode() not in _db_bytes(tmp_path / "devices.db")


def test_a_second_decision_loses_and_is_told_who_won(store):
    created, _ = _create(store)
    store.approve(created.request.id, name=None, decided_by="beacon:x")
    with pytest.raises(pr.NotPending) as caught:
        store.approve(created.request.id, name=None, decided_by="telegram:1")
    assert caught.value.status == "approved"
    with pytest.raises(pr.NotPending):
        store.deny(created.request.id, decided_by="telegram:1")


def test_an_unknown_request_is_unknown(store):
    with pytest.raises(pr.UnknownRequest):
        store.approve("f" * 32, name=None, decided_by="beacon:x")
    with pytest.raises(pr.UnknownRequest):
        store.deny("f" * 32, decided_by="beacon:x")


def test_a_failure_while_sealing_leaves_no_device_and_the_request_pending(store, devices, monkeypatch):
    created, _ = _create(store)

    def boom(**kwargs):
        raise RuntimeError("seal failed")

    monkeypatch.setattr(pr.pair_seal, "seal_token", boom)
    with pytest.raises(RuntimeError):
        store.approve(created.request.id, name=None, decided_by="beacon:x")
    assert devices.list_devices() == [], "no live token nobody holds"
    assert store.get(created.request.id).state == "pending"


def test_two_connections_approving_at_once_have_exactly_one_winner(tmp_path, clock):
    """The CLI and the daemon are two processes on one file: the guard is in SQL, not in Python."""
    path = tmp_path / "devices.db"
    first = pr.PairRequestStore(DeviceStore(path), settings=pr.PairingSettings(), clock=clock)
    second = pr.PairRequestStore(DeviceStore(path), settings=pr.PairingSettings(), clock=clock)
    created, _ = _create(first)
    outcomes: list[object] = []
    barrier = threading.Barrier(2)

    def go(store):
        barrier.wait()
        try:
            outcomes.append(store.approve(created.request.id, name=None, decided_by="beacon:x"))
        except pr.PairError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=go, args=(s,)) for s in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    wins = [o for o in outcomes if not isinstance(o, Exception)]
    assert len(wins) == 1 and len(outcomes) == 2, outcomes
    assert len(DeviceStore(path).list_devices()) == 1, "one winner, one device"


# ── deny and cancel ──────────────────────────────────────────────────────────

def test_denying_mints_nothing(store, devices):
    created, _ = _create(store)
    store.deny(created.request.id, decided_by="beacon:x")
    assert store.get(created.request.id).state == "denied"
    assert devices.list_devices() == []
    with pytest.raises(pr.NotPending) as caught:
        store.approve(created.request.id, name=None, decided_by="beacon:x")
    assert caught.value.status == "denied"


def test_the_requester_can_cancel_while_pending_and_only_then(store):
    created, _ = _create(store)
    assert store.cancel(created.request.id).state == "canceled"
    assert store.cancel(created.request.id) is None
    other, _ = _create(store, source="192.0.2.50")
    store.approve(other.request.id, name=None, decided_by="beacon:x")
    assert store.cancel(other.request.id) is None, "cancelling cannot undo an approval"
    assert store.get(other.request.id).state == "approved"


# ── delivery ─────────────────────────────────────────────────────────────────

def test_a_lost_response_is_retry_safe_until_acknowledged(store):
    created, _ = _create(store)
    store.approve(created.request.id, name=None, decided_by="beacon:x")
    first = store.sealed_blob(created.request.id)
    assert store.sealed_blob(created.request.id) == first, "the same sealed blob, not a re-seal"


def test_acknowledging_wipes_the_blob_and_is_once(store, tmp_path):
    created, requester = _create(store)
    store.approve(created.request.id, name=None, decided_by="beacon:x")
    sealed = store.sealed_blob(created.request.id)
    assert store.acknowledge(created.request.id) is True
    assert store.get(created.request.id).state == "delivered"
    assert store.sealed_blob(created.request.id) is None
    assert store.acknowledge(created.request.id) is False
    assert sealed["ciphertext"].encode() not in _db_bytes(tmp_path / "devices.db")


def test_acknowledging_a_request_that_was_not_approved_does_nothing(store):
    created, _ = _create(store)
    assert store.acknowledge(created.request.id) is False
    assert store.get(created.request.id).state == "pending"


def test_the_delivery_window_is_five_minutes_after_approval(store, clock):
    created, _ = _create(store)
    clock.now += 100
    store.approve(created.request.id, name=None, decided_by="beacon:x")
    assert store.get(created.request.id).delivery_deadline == clock.now + 300


def test_a_device_nobody_collects_is_revoked_and_its_sockets_hear_of_it(store, devices, clock):
    created, requester = _create(store)
    approved = store.approve(created.request.id, name=None, decided_by="beacon:x")
    token = requester.unseal(created.request.id, store.sealed_blob(created.request.id))["token"]
    heard: list[list[str]] = []
    devices.add_revoke_listener(heard.append)
    clock.now += 299
    assert store.sweep().uncollected == [] and devices.lookup(token_digest(token)) is not None
    clock.now += 2
    swept = store.sweep()
    assert [(r.id, device_id) for r, device_id in swept.uncollected] == [(created.request.id, approved.device_id)]
    assert devices.lookup(token_digest(token)) is None, "the uncollected token no longer authenticates"
    assert heard == [[approved.device_id]], "through DeviceStore.revoke, so #696's listener closes its sockets"
    assert store.get(created.request.id).state == "uncollected"
    assert store.sealed_blob(created.request.id) is None


def test_a_collected_device_is_not_revoked_by_the_sweep(store, devices, clock):
    created, _ = _create(store)
    approved = store.approve(created.request.id, name=None, decided_by="beacon:x")
    store.acknowledge(created.request.id)
    clock.now += 10_000
    assert store.sweep().uncollected == []
    assert next(d for d in devices.list_devices() if d.id == approved.device_id).revoked_at is None


# ── housekeeping ─────────────────────────────────────────────────────────────

def test_finished_rows_are_deleted_after_seven_days_and_pending_ones_never(store, clock, tmp_path):
    done, _ = _create(store, source="192.0.2.1")
    store.deny(done.request.id, decided_by="beacon:x")
    live, _ = _create(store, source="192.0.2.2")     # never decided: it simply expires
    clock.now += 6 * 86400
    assert store.sweep().purged == 0
    clock.now += 86400 + 1                 # the denied row is now over seven days old
    assert store.sweep().purged == 1
    assert store.get(done.request.id) is None
    assert store.get(live.request.id) is not None


def test_pending_lists_newest_first(store, clock):
    a, _ = _create(store, source="192.0.2.1")
    clock.now += 5
    b, _ = _create(store, source="192.0.2.2")
    assert [r.id for r in store.pending()] == [b.request.id, a.request.id]
