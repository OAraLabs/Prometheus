"""The one-time OWNER code: what makes a remote Beacon the owner's own device, and nothing else can.

Before this, an owner device (operator-equivalent, and "a person" for worker approvals since #744) came only
from same-Mac pairing, so a Beacon on another computer (a MacBook reaching the Mac mini over Tailscale) could hold
the master key or a scoped device token, and neither may approve a worker. ``oara pair --owner``, run on the
machine Prometheus runs on, asks the daemon for a short code; the remote Beacon types it and receives an owner
device token of its own.

This file pins the code itself (``config/owner_codes.py``), with no HTTP:

* 8 ASCII digits, good for 5 minutes, for ONE redemption, and only one code is live at a time;
* 5 wrong tries shut a source out and 25 in all burn the code, so an online guesser gets at most 25 chances in
  10^8 per code the owner asks for, and one neighbour guessing badly cannot burn it for everyone;
* the store keeps a digest, never the code, and nothing it returns or logs carries the code;
* a redemption that fails to mint leaves the code usable (nothing was granted), and a mint happens at most once.
"""

from __future__ import annotations

import logging
import threading

import pytest

from prometheus.config import owner_codes as oc

T0 = 1_760_000_000.0


class Clock:
    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


WALL = 1_760_500_000.0


class Minter:
    """Stands in for ``DeviceStore.mint_owner``: counts calls and returns a mint record."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self) -> dict:
        self.calls += 1
        if self.fail:
            raise RuntimeError("the registry is locked")
        return {"id": f"dev{self.calls:029d}", "name": "Will's MacBook Pro", "platform": "macos",
                "token": f"tok-{self.calls}"}


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def codes(clock) -> oc.OwnerCodes:
    return oc.OwnerCodes(clock=clock, wall=lambda: WALL)


def wrong_for(code: str) -> str:
    return f"{(int(code) + 1) % 10**8:08d}"


# ── the code ─────────────────────────────────────────────────────────────────

def test_a_code_is_eight_ascii_digits_good_for_five_minutes(codes, clock):
    record, code = codes.issue()
    assert len(code) == 8 and code.isascii() and code.isdigit()
    assert record.status == oc.PENDING
    assert record.expires_at == clock.now + oc.TTL_SECONDS == clock.now + 300
    assert record.expires_wall == WALL + 300
    assert len(record.id) == 32 and int(record.id, 16) >= 0


def test_leading_zeros_are_kept(codes, monkeypatch):
    monkeypatch.setattr(oc.secrets, "randbelow", lambda n: 42)
    _record, code = codes.issue()
    assert code == "00000042"


def test_the_code_is_drawn_from_the_whole_space(codes, monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(oc.secrets, "randbelow", lambda n: seen.append(n) or 7)
    codes.issue()
    assert seen == [10**8]


def test_the_store_keeps_no_copy_of_the_code(codes):
    record, code = codes.issue()
    assert code not in repr(record)
    assert code not in repr(vars(codes))


@pytest.mark.parametrize("typed", ["1234-5678", "1234 5678", " 12345678 ", "12-34-56-78"])
def test_what_a_person_types_is_read_without_its_separators(typed):
    assert oc.normalize(typed) == "12345678"


@pytest.mark.parametrize("typed", ["1234567", "123456789", "abcd5678", "１２３４５６７８", "", "1234_5678", None, 12345678])
def test_anything_else_is_not_a_code(typed):
    assert oc.normalize(typed) is None


def test_the_display_form_groups_four_and_four():
    assert oc.format_code("00420042") == "0042-0042"


# ── one redemption ───────────────────────────────────────────────────────────

def test_the_right_code_mints_once_and_records_who(codes):
    record, code = codes.issue()
    mint = Minter()
    minted = codes.redeem(code, mint, source="100.64.0.7")
    assert minted is not None and minted["token"] == "tok-1" and mint.calls == 1
    after = codes.status(record.id)
    assert after.status == oc.REDEEMED
    assert (after.device_id, after.device_name, after.platform, after.source) == (
        minted["id"], "Will's MacBook Pro", "macos", "100.64.0.7")


def test_it_works_once(codes):
    _record, code = codes.issue()
    mint = Minter()
    assert codes.redeem(code, mint, source="a") is not None
    assert codes.redeem(code, mint, source="b") is None
    assert mint.calls == 1


def test_two_redemptions_at_once_mint_one_device(codes):
    _record, code = codes.issue()
    mint = Minter()
    gate = threading.Barrier(8)
    results: list[object] = []

    def go() -> None:
        gate.wait()
        results.append(codes.redeem(code, mint, source="x"))

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mint.calls == 1
    assert sum(r is not None for r in results) == 1


def test_a_mint_that_fails_leaves_the_code_usable_and_counts_no_wrong_try(codes):
    record, code = codes.issue()
    with pytest.raises(RuntimeError):
        codes.redeem(code, Minter(fail=True), source="x")
    assert codes.status(record.id).status == oc.PENDING
    assert codes.status(record.id).wrong_attempts == 0
    assert codes.redeem(code, Minter(), source="x") is not None


def test_no_code_live_is_simply_no(codes):
    mint = Minter()
    assert codes.redeem("12345678", mint, source="x") is None
    assert mint.calls == 0


# ── guessing ─────────────────────────────────────────────────────────────────

def test_a_wrong_code_mints_nothing_and_is_counted(codes):
    record, code = codes.issue()
    mint = Minter()
    assert codes.redeem(wrong_for(code), mint, source="x") is None
    assert mint.calls == 0 and codes.status(record.id).wrong_attempts == 1


def test_four_wrong_tries_still_leave_the_right_code_working_from_that_source(codes):
    _record, code = codes.issue()
    for _ in range(oc.MAX_WRONG_PER_SOURCE - 1):
        codes.redeem(wrong_for(code), Minter(), source="x")
    assert codes.redeem(code, Minter(), source="x") is not None


def test_twenty_four_wrong_tries_in_all_still_leave_it_working(codes):
    _record, code = codes.issue()
    for n in range(oc.MAX_WRONG_ATTEMPTS - 1):
        codes.redeem(wrong_for(code), Minter(), source=f"198.51.100.{n // oc.MAX_WRONG_PER_SOURCE}")
    assert codes.redeem(code, Minter(), source="192.0.2.77") is not None


def test_five_wrong_tries_shut_that_source_out_and_no_one_else(codes, caplog):
    caplog.set_level(logging.INFO)
    record, code = codes.issue()
    for _ in range(oc.MAX_WRONG_PER_SOURCE):
        assert codes.redeem(wrong_for(code), Minter(), source="198.51.100.4") is None
    mint = Minter()
    assert codes.redeem(code, mint, source="198.51.100.4") is None and mint.calls == 0
    assert codes.status(record.id).status == oc.PENDING
    assert codes.redeem(code, Minter(), source="192.0.2.77") is not None
    assert any("198.51.100.4" in r.getMessage() and "shut out" in r.getMessage() for r in caplog.records)


def test_twenty_five_wrong_tries_in_all_burn_the_code(codes, caplog):
    caplog.set_level(logging.INFO)
    record, code = codes.issue()
    assert oc.MAX_WRONG_ATTEMPTS == 25
    for n in range(oc.MAX_WRONG_ATTEMPTS):
        codes.redeem(wrong_for(code), Minter(), source=f"198.51.100.{n // oc.MAX_WRONG_PER_SOURCE}")
    assert codes.status(record.id).status == oc.LOCKED
    mint = Minter()
    assert codes.redeem(code, mint, source="192.0.2.77") is None and mint.calls == 0
    assert any("locked" in r.getMessage() for r in caplog.records)


def test_one_source_cannot_burn_the_code_however_often_it_tries(codes):
    """A shut-out source is refused BEFORE its try is compared, so its tries guess nothing and count nothing."""
    record, code = codes.issue()
    for _ in range(100):
        codes.redeem(wrong_for(code), Minter(), source="198.51.100.4")
    assert codes.status(record.id).status == oc.PENDING
    assert codes.status(record.id).wrong_attempts == oc.MAX_WRONG_PER_SOURCE
    assert codes.redeem(code, Minter(), source="192.0.2.77") is not None


def test_junk_counts_as_a_wrong_try_while_a_code_is_live(codes):
    record, code = codes.issue()
    for junk in ("", "abc", "1234567", None, 5, "１２３４５６７８"):
        codes.redeem(junk, Minter(), source="x")
    assert codes.status(record.id).wrong_attempts == oc.MAX_WRONG_PER_SOURCE    # the sixth was refused unread
    assert codes.redeem(code, Minter(), source="x") is None                      # shut out after five


# ── time, replacement, cancel ────────────────────────────────────────────────

def test_it_expires_after_five_minutes(codes, clock):
    record, code = codes.issue()
    clock.now += oc.TTL_SECONDS
    mint = Minter()
    assert codes.redeem(code, mint, source="x") is None and mint.calls == 0
    assert codes.status(record.id).status == oc.EXPIRED


def test_it_works_until_then(codes, clock):
    _record, code = codes.issue()
    clock.now += oc.TTL_SECONDS - 0.5
    assert codes.redeem(code, Minter(), source="x") is not None


def test_a_new_code_replaces_the_one_before(codes, monkeypatch):
    draws = iter([11111111, 22222222])
    monkeypatch.setattr(oc.secrets, "randbelow", lambda n: next(draws))
    first, first_code = codes.issue()
    _second, second_code = codes.issue()
    assert codes.status(first.id).status == oc.REPLACED
    assert codes.redeem(first_code, Minter(), source="x") is None
    assert codes.redeem(second_code, Minter(), source="x") is not None


def test_a_cancelled_code_stops_working(codes):
    record, code = codes.issue()
    assert codes.cancel(record.id) is True
    assert codes.status(record.id).status == oc.CANCELED
    assert codes.redeem(code, Minter(), source="x") is None


def test_cancel_is_false_for_an_unknown_or_finished_code(codes):
    record, code = codes.issue()
    codes.redeem(code, Minter(), source="x")
    assert codes.cancel(record.id) is False
    assert codes.cancel("0" * 32) is False


def test_the_lifetime_follows_the_monotonic_clock_not_the_wall(clock):
    """An NTP step must not stretch or cut a code's five minutes; the wall clock is only for showing a time."""
    wall = Clock(WALL)
    codes = oc.OwnerCodes(clock=clock, wall=wall)
    record, code = codes.issue()
    wall.now -= 3600                                    # the wall clock steps back an hour
    clock.now += oc.TTL_SECONDS
    assert codes.redeem(code, Minter(), source="x") is None
    assert codes.status(record.id).status == oc.EXPIRED


def test_the_default_clock_is_monotonic():
    import time

    assert oc.OwnerCodes().clock is time.monotonic


def test_the_live_code_is_readable_without_the_code(codes):
    assert codes.live() is None
    record, _code = codes.issue()
    assert codes.live() == record


def test_the_status_of_an_unknown_id_is_none(codes):
    assert codes.status("f" * 32) is None
    assert codes.status("") is None


def test_only_recent_codes_are_remembered(codes):
    first, _ = codes.issue()
    for _ in range(oc.KEEP_FINISHED + 2):
        codes.issue()
    assert codes.status(first.id) is None


# ── nothing secret leaves ────────────────────────────────────────────────────

def test_no_log_line_carries_the_code_or_the_token(codes, caplog):
    caplog.set_level(logging.DEBUG)
    _record, code = codes.issue()
    codes.redeem(wrong_for(code), Minter(), source="x")
    minted = codes.redeem(code, Minter(), source="x")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert code not in text and oc.format_code(code) not in text
    assert minted["token"] not in text
