"""Audit #7: boot-time model detection must retry before falling back.

A model server co-booting with the daemon can miss a single probe; previously
that stranded a stale model name until the next restart.
"""

from __future__ import annotations

import asyncio

from prometheus.daemon import _detect_loaded_model_with_retry


class _Provider:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def detect_loaded_model(self):
        self.calls += 1
        return self._results.pop(0) if self._results else None


async def _noop_sleep(_delay):
    return None


def test_retries_until_detected():
    p = _Provider([None, None, "google_gemma-4-26B-A4B-it"])
    got = asyncio.run(_detect_loaded_model_with_retry(p, attempts=5, sleep=_noop_sleep))
    assert got == "google_gemma-4-26B-A4B-it"
    assert p.calls == 3  # stopped as soon as it succeeded


def test_returns_none_after_all_attempts():
    p = _Provider([None, None, None, None, None])
    got = asyncio.run(_detect_loaded_model_with_retry(p, attempts=5, sleep=_noop_sleep))
    assert got is None
    assert p.calls == 5


def test_succeeds_first_try_without_sleeping():
    slept = []

    async def rec_sleep(d):
        slept.append(d)

    p = _Provider(["model-x"])
    got = asyncio.run(_detect_loaded_model_with_retry(p, attempts=5, sleep=rec_sleep))
    assert got == "model-x"
    assert p.calls == 1
    assert slept == []  # no backoff when it works immediately
