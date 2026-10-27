"""A pick that ends is logged as ended (found on the computer-use rig, 2026-10-23).

A pick lives in memory, so it ended without a frame when the daemon restarted, when it expired, or
when a client cleared one the daemon no longer held. Beacon, which learns the pick from the log
(live or by backfill), kept showing the app as picked after a restart; its picker opens only when
nothing is picked, so "needs an app" showed no apps, and "Change / clear" changed nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.test_computer_door import (  # noqa: F401 - _linux_site_rule is an autouse fixture
    APP,
    PERSON,
    SESSION,
    _linux_site_rule,
    _rig,
)
from tests.test_computer_livestream import _live

LAUNCHER = Path(__file__).resolve().parent.parent / "src" / "prometheus" / "web" / "launcher.py"


def _states(lv) -> list[tuple[str, str | None]]:
    return [(p.get("session_id"), p.get("state")) for k, p in lv.seen if k == "computer_binding"]


async def _drain(rig) -> None:
    import asyncio

    for _ in range(5):
        await asyncio.sleep(0)
    if rig.runner._background:
        await asyncio.gather(*list(rig.runner._background))


async def test_clearing_a_pick_the_daemon_no_longer_holds_still_says_off(tmp_path):
    rig = _rig(tmp_path, [])
    lv = _live(rig, tmp_path)
    assert rig.runner.unbind(SESSION) is False
    await _drain(rig)
    assert _states(lv) == [(SESSION, "off")]


async def test_an_expired_pick_says_off(tmp_path):
    from prometheus.computer.task import MAX_BINDING_SECONDS

    rig = _rig(tmp_path, [])
    lv = _live(rig, tmp_path)
    now = [1_000_000.0]
    rig.runner._wall = lambda: now[0]
    await rig.runner.bind(SESSION, APP, scope="session", by=PERSON, surface="beacon")
    now[0] += MAX_BINDING_SECONDS + 1
    assert rig.runner.binding_for(SESSION) is None
    await _drain(rig)
    assert _states(lv) == [(SESSION, "on"), (SESSION, "off")]
    off = [p for k, p in lv.seen if k == "computer_binding"][-1]
    assert off["reason"] == "expired"


async def test_at_start_every_pick_the_log_still_shows_on_is_ended(tmp_path):
    from prometheus.computer.livestream import ComputerLiveStream

    rig = _rig(tmp_path, [])
    lv = _live(rig, tmp_path)
    await rig.runner.bind(SESSION, APP, scope="session", by=PERSON, surface="beacon")
    await rig.runner.bind("beacon:other", APP, scope="session", by=PERSON, surface="beacon")
    rig.runner.unbind("beacon:other")
    await _drain(rig)

    # A restart: a new stream over the same durable log, with nothing in memory.
    restarted = ComputerLiveStream(lv.bus, telemetry=lv.tel)
    lv.seen.clear()
    assert await restarted.end_stale_picks() == 1
    assert _states(lv) == [(SESSION, "off")], "only the pick still on is ended"
    assert [p for k, p in lv.seen][-1]["reason"] == "restart"
    assert await restarted.end_stale_picks() == 0, "the log now says off; nothing more to end"


async def test_without_telemetry_the_start_pass_does_nothing(tmp_path):
    from prometheus.computer.livestream import ComputerLiveStream

    rig = _rig(tmp_path, [])
    lv = _live(rig, tmp_path)
    assert await ComputerLiveStream(lv.bus).end_stale_picks() == 0


def test_the_launcher_ends_stale_picks_when_it_builds_the_stream():
    src = LAUNCHER.read_text()
    tree = ast.parse(src)
    builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "ComputerLiveStream"]
    ends = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "end_stale_picks"]
    assert len(builds) == 1 and len(ends) == 1
    assert builds[0].lineno < ends[0].lineno, "the pass runs once the stream exists"
