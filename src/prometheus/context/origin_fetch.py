"""An independent refresh for the one operand `tree_vs_origin` compares against.

WHY THIS EXISTS — the fix inherited the shape of the bug, one level out
------------------------------------------------------------------------
``stale`` compared the running process to the checked-out tree: both operands
from the same clone, so a clone behind ``origin/main`` reported ``stale:
false``. #518 added the missing axis, ``tree_vs_origin``.

But ``origin_main_sha`` reads ``refs/remotes/origin/main`` — a **local cached
copy**, updated only by a fetch — and on the deploy clone **the only thing
that fetched was a deploy**. So the cache refreshed precisely at the moment
the clone became up to date, and the axis could report ``in_sync`` and
nothing else, indefinitely.

Measured on production 2026-09-19: with one merged commit unpulled the block
said ``state: current``; a bare fetch flipped it to ``behind_origin`` with
nothing else changed. ``stale`` compared the tree TO ITSELF;
``tree_vs_origin`` compared it TO A CACHED COPY OF THE OTHER SIDE, which under
that refresh schedule is the same thing wearing a second operand.

BOTH HALVES, OR NEITHER
-----------------------
This module is one half. The other is the staleness threshold in
``deployment_freshness``. Either alone fails the same way:

* a fetch that starts failing **goes silent** — the ref simply stops moving
  and ``in_sync`` keeps being printed;
* a threshold with no fetch is **permanently unknown**, which is noise rather
  than signal and trains an operator to ignore the third answer.

Together a broken fetch degrades to *cannot tell* instead of to false
assurance. That is the whole design and neither half should be landed alone.

⚠ WHY NOT ``origin_ref_age_seconds`` AS THE STALENESS OPERAND
--------------------------------------------------------------
It reads the ref's **reflog**, which records when the ref last MOVED — not
when it was last FETCHED. Verified on the deploy clone: age was 766 s before
a successful fetch and 766 s after, because ``origin/main`` had not changed.

Keying the threshold on it would therefore report ``unknown`` whenever the
repository is merely QUIET, which is most of the time — turning the third
answer into background noise within a day. So this module records its own
success time, and that is the operand.

Ref movement is still *positive evidence* of a fetch (the ref cannot move
without one), so ``deployment_freshness`` takes the FRESHER of the two.

WHAT THE FETCH TOUCHES
----------------------
One remote-tracking ref, by explicit refspec, and nothing else::

    git fetch --no-tags --no-write-fetch-head origin \\
        +refs/heads/main:refs/remotes/origin/main

No ``--prune`` (it could delete refs), no tags, no ``FETCH_HEAD`` write, and
by construction no HEAD update and no working-tree change — a fetch never
touches either, and ``test_the_fetch_touches_the_ref_and_nothing_else``
asserts it against a real repository rather than trusting that sentence.

It never runs in a request path. It is a background task with its own
interval, and the subprocess goes off the event loop
(``asyncio.to_thread``) — the daemon has already paid for that lesson once,
when three synchronous phases in a 30-minute periodic task produced the idle
loop-watchdog stalls fixed in #416.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: How often to refresh the operand. Five minutes matches the cadence the rest
#: of this deployment already runs on (the heartbeat watcher's cron), and the
#: cost is one network round trip that transfers nothing when origin is quiet.
DEFAULT_INTERVAL_SECONDS = 300

#: How old the freshest evidence may be before ``tree_vs_origin`` degrades to
#: ``unknown``. **Four intervals.**
#:
#: The lower bound is "comfortably longer than the interval": at 1x or 2x a
#: single transient network failure flips the block to ``unknown``, and a
#: third answer that cries wolf is one an operator learns to scroll past —
#: which would cost more than the defect it was added to catch. Four intervals
#: survives three consecutive failures.
#:
#: The upper bound is "comfortably shorter than a merge could reach the box
#: unseen". 20 minutes is the widest window in which this block could assert
#: ``in_sync`` on evidence that is actually stale, and the two real
#: merge-to-deploy cycles measured on this box were ~50 minutes and ~5 hours.
#: So the false-assurance window is ~2.5x shorter than the fastest observed
#: opportunity to be wrong, while still absorbing three failed fetches.
DEFAULT_STALE_AFTER_SECONDS = 1200

#: Bound on one fetch. Longer than a slow round trip, far shorter than the
#: interval, so a hung fetch cannot overlap the next one.
FETCH_TIMEOUT_SECONDS = 45

#: Exactly one ref, by explicit refspec.
MAIN_REFSPEC = "+refs/heads/main:refs/remotes/origin/main"


@dataclass
class FetchState:
    """What the fetcher has managed to do. Read by the status block.

    ⚠ ``last_error`` is carried so a FAILING FETCH IS VISIBLE IN THE BLOCK,
    not only in a log. Silence about the refresh is how this whole class of
    defect recurs: the previous version failed by never running at all, and
    nothing anywhere said so.
    """

    enabled: bool = True
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS
    last_attempt_ts: float | None = None
    last_success_ts: float | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    _now: Any = field(default=time.time, repr=False)

    def success_age_seconds(self) -> int | None:
        """Seconds since the last SUCCESSFUL fetch, or None if never."""
        if self.last_success_ts is None:
            return None
        return max(0, int(self._now() - self.last_success_ts))

    def state(self) -> str:
        """``ok`` | ``failing`` | ``never`` | ``disabled``."""
        if not self.enabled:
            return "disabled"
        if self.last_success_ts is None:
            # Distinct from `failing`: nothing has succeeded YET, which on a
            # fresh boot is a normal transient and not the same claim as
            # "it tried and could not".
            return "failing" if self.consecutive_failures else "never"
        if self.consecutive_failures:
            return "failing"
        return "ok"

    def block(self) -> dict[str, Any]:
        """The wire shape. Same conventions as its neighbours on /api/status.

        A string state, a block-level ``detail`` sentence naming the remedy,
        and no operand it cannot stand behind.
        """
        state = self.state()
        return {
            "state": state,
            "last_success_age_seconds": self.success_age_seconds(),
            "consecutive_failures": self.consecutive_failures,
            # Bounded: this is an endpoint, and a git error can be long.
            "last_error": (self.last_error or None) if state != "ok" else None,
            "interval_seconds": self.interval_seconds,
            "stale_after_seconds": self.stale_after_seconds,
            "detail": _FETCH_DETAIL[state],
        }


#: One sentence per state, naming the REMEDY — the convention
#: ``_FRESHNESS_DETAIL`` sets next door, for the reason it gives: the rollup is
#: read by people at 2am, and a state name without an action is a puzzle.
_FETCH_DETAIL: dict[str, str] = {
    "ok": "origin/main is being refreshed independently of deploys.",
    "failing": (
        "The origin refresh is FAILING — tree_vs_origin is comparing against "
        "a cached ref that is no longer being updated, and degrades to "
        "unknown once it passes stale_after_seconds. Check network and git "
        "credentials on this clone."
    ),
    "never": (
        "The origin refresh has not completed a first fetch yet; normal for "
        "the first minutes after a restart."
    ),
    "disabled": (
        "The origin refresh is DISABLED, so tree_vs_origin can only report "
        "what the last deploy happened to cache — set "
        "deployment.origin_fetch.enabled to restore it."
    ),
}


def fetch_origin_main(
    repo_dir: str | Path | None = None,
    timeout: int = FETCH_TIMEOUT_SECONDS,
) -> tuple[bool, str | None]:
    """Update exactly ``refs/remotes/origin/main``. ``(ok, error)``.

    SYNCHRONOUS AND BLOCKING BY DESIGN — callers put it on a worker thread.
    Kept sync so it is testable without an event loop and so the one place
    that must not block (the request path) cannot call it by accident.
    """
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[3]
    cmd = [
        "git", "fetch", "--no-tags", "--no-write-fetch-head",
        "origin", MAIN_REFSPEC,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_dir), capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        # First line only, bounded: an endpoint renders this.
        return False, (detail[0][:200] if detail else f"git exited {proc.returncode}")
    return True, None


class OriginFetcher:
    """Refreshes the cached origin ref on an interval. Never in a request path."""

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
        enabled: bool = True,
        now: Any = time.time,
    ) -> None:
        self._repo_dir = repo_dir
        self._now = now
        self.state = FetchState(
            enabled=enabled,
            interval_seconds=int(interval_seconds),
            stale_after_seconds=int(stale_after_seconds),
            _now=now,
        )

    def fetch_once(self) -> bool:
        """One attempt, recording the outcome. Synchronous."""
        self.state.last_attempt_ts = self._now()
        ok, error = fetch_origin_main(self._repo_dir)
        if ok:
            self.state.last_success_ts = self._now()
            self.state.consecutive_failures = 0
            self.state.last_error = None
            return True
        self.state.consecutive_failures += 1
        self.state.last_error = error
        # WARNING, not debug, and every time rather than once: a refresh that
        # quietly stops is the defect this module exists to close, and a
        # single first-failure log is indistinguishable from a blip when read
        # a week later.
        logger.warning(
            "origin fetch FAILED (%d consecutive): %s",
            self.state.consecutive_failures, error,
        )
        return False

    async def run_forever(self) -> None:
        """Fetch on the interval, off the event loop, forever.

        The subprocess goes through ``asyncio.to_thread`` for the reason #416
        exists: a periodic task doing synchronous work on the loop produced
        the idle loop-watchdog stalls, and a git fetch over the network is
        exactly that shape.
        """
        if not self.state.enabled:
            logger.info("origin fetch disabled by config; tree_vs_origin will "
                        "degrade to unknown once its cached ref goes stale")
            return
        while True:
            try:
                await asyncio.to_thread(self.fetch_once)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a refresh must not kill the daemon
                logger.warning("origin fetch loop error", exc_info=True)
            await asyncio.sleep(self.state.interval_seconds)


def fetcher_from_config(
    config: dict[str, Any] | None, repo_dir: str | Path | None = None
) -> OriginFetcher:
    """Build from ``deployment.origin_fetch``. Enabled by default.

    ON BY DEFAULT DELIBERATELY. The axis it feeds is not optional-correct: a
    ``tree_vs_origin`` without an independent refresh is decorative, and a key
    whose false default silently disables a control is a shape this repo has
    already paid for (``coding.enabled`` gated nothing for months).
    """
    block = ((config or {}).get("deployment") or {}).get("origin_fetch") or {}
    return OriginFetcher(
        repo_dir=repo_dir,
        interval_seconds=int(block.get("interval_seconds", DEFAULT_INTERVAL_SECONDS)),
        stale_after_seconds=int(
            block.get("stale_after_seconds", DEFAULT_STALE_AFTER_SECONDS)),
        enabled=bool(block.get("enabled", True)),
    )
