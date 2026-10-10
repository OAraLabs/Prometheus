"""``web/source_limits.py`` — a sliding-window limit keyed by the TCP peer address.

The unauthenticated pairing routes (``GET /api/hello`` first) cannot lean on a token, so they lean on
how often one source may ask. What is pinned is what keeps the limiter from becoming the problem:

* a refusal does NOT spend budget, or a source already over its limit would keep its window full and
  never recover;
* the retry hint is the real time until the oldest hit leaves the window, never a guess;
* sources are independent;
* memory is bounded, because the key is chosen by whoever can reach the port and an unbounded dict
  keyed by it is a slow leak an outsider controls.
"""

from __future__ import annotations

from prometheus.web.source_limits import SourceLimiter


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_it_admits_up_to_the_limit_and_then_refuses():
    clock = _Clock()
    limiter = SourceLimiter(3, 60.0, clock=clock)
    assert [limiter.check("a").allowed for _ in range(5)] == [True, True, True, False, False]


def test_a_refusal_does_not_spend_budget():
    """Refusals are spread across the window: a limiter that counted them would stay full."""
    clock = _Clock()
    limiter = SourceLimiter(2, 60.0, clock=clock)
    assert limiter.check("a").allowed          # t=0, leaves the window at t=60
    clock.now += 1
    assert limiter.check("a").allowed          # t=1
    for _ in range(50):                        # refused at t=2 .. t=51
        clock.now += 1
        assert not limiter.check("a").allowed
    clock.now = 1000.0 + 60.01                 # the first real hit leaves; the refusals are not hits
    assert limiter.check("a").allowed          # one slot is free
    assert not limiter.check("a").allowed      # and only one: the t=1 hit and the new one fill it


def test_the_retry_hint_is_the_time_until_the_oldest_hit_leaves():
    clock = _Clock()
    limiter = SourceLimiter(3, 60.0, clock=clock)
    limiter.check("a")                 # t=0, leaves the window at t=60
    clock.now += 10
    limiter.check("a")
    clock.now += 10
    limiter.check("a")
    clock.now += 10                    # t=30
    refused = limiter.check("a")
    assert not refused.allowed
    assert refused.retry_after == 30
    clock.now += 29.5                  # t=59.5: half a second left, reported as a whole second
    assert limiter.check("a").retry_after == 1


def test_sources_do_not_share_a_budget():
    clock = _Clock()
    limiter = SourceLimiter(1, 60.0, clock=clock)
    assert limiter.check("a").allowed
    assert not limiter.check("a").allowed
    assert limiter.check("b").allowed


def test_memory_is_bounded_by_the_number_of_sources_tracked():
    clock = _Clock()
    limiter = SourceLimiter(1, 60.0, clock=clock, max_sources=4)
    for n in range(50):
        clock.now += 0.001
        limiter.check(f"10.0.0.{n}")
    assert len(limiter) <= 4


def test_an_idle_source_is_forgotten_once_its_window_has_passed():
    clock = _Clock()
    limiter = SourceLimiter(1, 60.0, clock=clock, max_sources=4)
    limiter.check("a")
    clock.now += 61
    for n in range(3):
        limiter.check(f"b{n}")
    assert "a" not in limiter


def test_peek_says_whether_a_source_is_at_its_limit_without_spending_anything():
    """The wrong-secret limit counts only FAILURES, so it must be able to ask before it records."""
    clock = _Clock()
    limiter = SourceLimiter(2, 60.0, clock=clock)
    for _ in range(10):
        assert limiter.peek("a").allowed
    limiter.check("a")
    limiter.check("a")
    for _ in range(3):
        assert not limiter.peek("a").allowed
    assert limiter.peek("a").retry_after == 60
    clock.now += 60.01
    assert limiter.peek("a").allowed


def test_peeking_never_creates_a_source():
    limiter = SourceLimiter(2, 60.0, clock=_Clock())
    assert limiter.peek("never-seen").allowed
    assert "never-seen" not in limiter and len(limiter) == 0
