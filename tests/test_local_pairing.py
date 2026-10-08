"""The one-time same-Mac pairing secret (the module is src/prometheus/config/local_pairing.py).

Prometheus.app's daemon starts under launchd, so nobody sees the six-digit code printed at the terminal.
Beacon, on the same Mac, pairs with a secret in a user-only FILE instead: 32 random bytes in a 0700
directory, a 0600 file, deleted on first use. These tests pin what makes that safe, not just that it works:

* the file's permissions, and that a symlinked or loosened directory/file is refused, not trusted;
* the secret never expires (a person may open Beacon an hour after installing) and survives a daemon
  restart, but is used exactly once, even under a race;
* a wrong, empty or oddly-shaped value never matches, and no value is ever written to a log;
* the six-digit code is never mistaken for the secret.
"""

from __future__ import annotations

import logging
import os
import stat
import threading
from pathlib import Path

import pytest

from prometheus.config import local_pairing as lp


@pytest.fixture
def directory(tmp_path) -> Path:
    return tmp_path / "Application Support" / "Prometheus" / "pairing"


# ── when it applies ──────────────────────────────────────────────────────────

def test_it_is_off_unless_the_install_is_the_app_or_a_directory_is_named(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)
    monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
    assert lp.enabled() is False
    monkeypatch.setenv("PROMETHEUS_INSTALL_KIND", "app")
    assert lp.enabled() is True
    monkeypatch.setenv("PROMETHEUS_INSTALL_KIND", "pip")
    assert lp.enabled() is False
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", "/x")
    assert lp.enabled() is True


def test_the_default_directory_is_application_support_on_a_mac(monkeypatch, tmp_path):
    monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(lp, "_platform", lambda: "darwin")
    assert lp.pairing_dir() == tmp_path / "Library" / "Application Support" / "Prometheus" / "pairing"
    assert lp.secret_path() == lp.pairing_dir() / "pair.secret"


def test_the_directory_can_be_overridden_for_tests(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(tmp_path / "elsewhere"))
    assert lp.pairing_dir() == tmp_path / "elsewhere"


# ── minting ──────────────────────────────────────────────────────────────────

def test_minting_writes_a_private_file_in_a_private_directory(directory):
    secret = lp.mint_secret(directory)
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    path = directory / "pair.secret"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    text = path.read_text()
    assert text == secret + "\n", "one line, exactly the secret"
    assert len(secret) == 43 and set(secret) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"), "32 random bytes, base64url, no padding"


def test_no_other_file_is_left_behind(directory):
    lp.mint_secret(directory)
    assert sorted(p.name for p in directory.iterdir()) == ["pair.secret"]


def test_an_unused_secret_survives_a_restart_and_never_expires(directory):
    first = lp.mint_secret(directory)
    old = os.stat(directory / "pair.secret").st_mtime - 3 * 24 * 3600
    os.utime(directory / "pair.secret", (old, old))                      # three days old
    assert lp.mint_secret(directory) == first, "a person may open Beacon an hour (or a day) after installing"
    assert lp.check_secret(first, directory) is True


def test_a_loosened_directory_is_tightened(directory):
    lp.mint_secret(directory)
    os.chmod(directory, 0o755)
    lp.mint_secret(directory)
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700


def test_a_file_that_others_could_read_is_replaced_not_trusted(directory):
    leaked = lp.mint_secret(directory)
    os.chmod(directory / "pair.secret", 0o644)          # someone may have read it
    fresh = lp.mint_secret(directory)
    assert fresh != leaked
    assert stat.S_IMODE(os.stat(directory / "pair.secret").st_mode) == 0o600
    assert lp.check_secret(leaked, directory) is False


def test_a_symlinked_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(lp.PairingFileError, match="symlink"):
        lp.mint_secret(link)


def test_a_symlinked_secret_file_is_never_followed(directory, tmp_path):
    lp.mint_secret(directory)
    target = tmp_path / "attacker-chosen"
    target.write_text("A" * 43 + "\n")
    os.chmod(target, 0o600)
    (directory / "pair.secret").unlink()
    (directory / "pair.secret").symlink_to(target)
    assert lp.read_secret(directory) is None
    assert lp.check_secret("A" * 43, directory) is False


def test_garbage_in_the_file_is_not_a_secret(directory):
    lp.mint_secret(directory)
    (directory / "pair.secret").write_text("short\n")
    assert lp.read_secret(directory) is None
    again = lp.mint_secret(directory)
    assert len(again) == 43 and lp.check_secret(again, directory)


# ── checking ─────────────────────────────────────────────────────────────────

def test_only_the_exact_secret_matches(directory):
    secret = lp.mint_secret(directory)
    assert lp.check_secret(secret, directory) is True
    # The caller strips what a client sent; the check itself is exact, so an unstripped value never matches.
    assert lp.check_secret(secret + "\n", directory) is False
    for wrong in ("", " ", "0" * 43, secret[:-1], secret + "x", secret.swapcase(), "123456", None):
        assert lp.check_secret(wrong, directory) is False, wrong        # type: ignore[arg-type]


def test_nothing_matches_when_there_is_no_file(directory):
    assert lp.check_secret("A" * 43, directory) is False
    assert lp.read_secret(directory) is None


def test_the_six_digit_code_is_never_the_secret(directory):
    assert lp.is_secret_shaped("123456") is False
    assert lp.is_secret_shaped("000042") is False
    assert lp.is_secret_shaped("1234567") is True            # anything that is not exactly six digits
    assert lp.is_secret_shaped(lp.mint_secret(directory)) is True


# ── using it ─────────────────────────────────────────────────────────────────

def test_a_right_secret_is_consumed_once_and_the_file_is_gone(directory):
    secret = lp.mint_secret(directory)
    assert lp.consume_if_matches(secret, directory) is True
    assert not (directory / "pair.secret").exists()
    assert lp.consume_if_matches(secret, directory) is False


def test_a_wrong_secret_leaves_the_file_for_the_right_one(directory):
    secret = lp.mint_secret(directory)
    assert lp.consume_if_matches("B" * 43, directory) is False
    assert (directory / "pair.secret").exists()
    assert lp.consume_if_matches(secret, directory) is True


def test_a_race_for_one_secret_has_exactly_one_winner(directory):
    for _ in range(25):
        secret = lp.mint_secret(directory)
        barrier = threading.Barrier(8)
        wins: list[bool] = []

        def attempt():
            barrier.wait()
            wins.append(lp.consume_if_matches(secret, directory))

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert wins.count(True) == 1, wins


def test_a_fresh_secret_replaces_the_old_one_atomically(directory):
    """`Prometheus --pair` writes a new secret while the daemon is running; a reader must never see half a file."""
    old = lp.mint_secret(directory)
    new = lp.replace_secret(directory)
    assert new != old and lp.check_secret(new, directory) and not lp.check_secret(old, directory)
    assert sorted(p.name for p in directory.iterdir()) == ["pair.secret"]


# ── nothing is logged ────────────────────────────────────────────────────────

def test_no_secret_is_ever_logged(directory, caplog):
    caplog.set_level(logging.DEBUG)
    secret = lp.mint_secret(directory)
    lp.check_secret("A" * 43, directory)
    lp.check_secret(secret, directory)
    lp.consume_if_matches(secret, directory)
    os.makedirs(directory, exist_ok=True)
    (directory / "pair.secret").write_text("x\n")
    lp.read_secret(directory)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in joined and "A" * 43 not in joined
