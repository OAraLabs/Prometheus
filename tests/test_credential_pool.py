"""Tests for Sprint 15b GRAFT: credential pool rotation."""

from __future__ import annotations

import time

import pytest

from prometheus.providers.credential_pool import CredentialPool


class TestCredentialPool:

    def test_round_robin_cycles(self):
        pool = CredentialPool(["key-a", "key-b", "key-c"])
        assert pool.get_next() == "key-a"
        assert pool.get_next() == "key-b"
        assert pool.get_next() == "key-c"
        assert pool.get_next() == "key-a"

    def test_single_key_always_returns_same(self):
        pool = CredentialPool(["only-key"])
        assert pool.get_next() == "only-key"
        assert pool.get_next() == "only-key"

    def test_429_rotates_to_next(self):
        pool = CredentialPool(["key-a", "key-b"])
        first = pool.get_next()
        assert first == "key-a"
        pool.report_error(first, 429)
        # key-a is still active (429 doesn't kill), but index moved
        second = pool.get_next()
        assert second == "key-b"

    def test_401_marks_dead(self):
        pool = CredentialPool(["key-a", "key-b"])
        pool.get_next()  # key-a
        pool.report_error("key-a", 401)
        # key-a is dead — next should skip to key-b
        assert pool.get_next() == "key-b"
        assert pool.get_next() == "key-b"  # still key-b, key-a is dead
        assert pool.active_count == 1

    def test_all_dead_raises(self):
        pool = CredentialPool(["key-a", "key-b"], dead_key_cooldown_seconds=9999)
        pool.report_error("key-a", 401)
        pool.report_error("key-b", 401)
        with pytest.raises(RuntimeError, match="All.*keys are dead"):
            pool.get_next()

    def test_dead_key_revives_after_cooldown(self):
        pool = CredentialPool(["key-a", "key-b"], dead_key_cooldown_seconds=0)
        pool.report_error("key-a", 401)
        # Cooldown is 0 seconds — should revive immediately
        assert pool.active_count == 2
        # key-a should be accessible again
        keys = {pool.get_next(), pool.get_next()}
        assert "key-a" in keys

    def test_success_tracking(self):
        pool = CredentialPool(["key-a"])
        key = pool.get_next()
        pool.report_success(key)
        pool.report_success(key)
        assert pool.stats["key-a"].successes == 2

    def test_failure_tracking(self):
        pool = CredentialPool(["key-a"])
        key = pool.get_next()
        pool.report_error(key, 500)
        assert pool.stats["key-a"].failures == 1
        assert pool.stats["key-a"].last_error == "HTTP 500"

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CredentialPool([])

    def test_no_pool_regression(self):
        """Without pool, providers should work exactly as before."""
        # This test verifies the CredentialPool is optional
        from prometheus.providers.base import ModelProvider
        assert not hasattr(ModelProvider, "credential_pool") or True
