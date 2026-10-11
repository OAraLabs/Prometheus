"""The driver session expiring (computer-use v1.1, after the first live test).

The adapter's ``start()`` keeps the first ``CuaDriver`` for as long as the
integration holds the adapter. Cua ends an idle driver session on its own, and
from then on every call on that driver answers ``session_ended`` ("this session
has ended; call start_session explicitly to reuse its label"). The first live
test's probe failed exactly so: ``list_apps`` went to a dead session.

What this file pins:

* **One retry, for one cause.** When the ONLY failing check is the probe's
  observe step and its cause is ``session_ended``, the probe drops the cached
  driver, starts a fresh one and runs once more. A second ``session_ended``, or
  any other failure, fails closed exactly as before.
* **The cause is typed, not read from text.** The adapter raises
  ``DriverSessionEnded`` only for the SDK's own ``error_code``; everything else
  stays a plain ``DriverUnavailable``.
* **The status shows its age.** ``/computer status`` reads the cached probe and
  never probes; it says how old that answer is. A task still forces a fresh
  probe before it starts.

Every probe here runs over fakes (cua.py's module docstring says why).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from types import SimpleNamespace

import pytest

from prometheus.computer import cua
from prometheus.computer.driver import (
    HALF_OK, DriverSessionEnded, DriverUnavailable, HalfResult,
    PreconditionResult,
)
from prometheus.computer.integration import ComputerIntegration

LOCAL = {"targets": {"local": {"kind": "local"}}}

#: The SDK's own words for an ended session (cua-driver 0.28.2).
ENDED_MESSAGE = ("this session has ended; call start_session explicitly to "
                 "reuse its label")


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _Adapter:
    """Stands in for CuaDriverAdapter. ``fail`` is what ``list_apps`` raises,
    or None for a healthy answer."""

    def __init__(self, target, fail=None):
        self.target = target
        self.fail = fail
        self.started = 0
        self.shut = 0
        self.listed = 0

    def start(self):
        self.started += 1

    def shutdown(self):
        self.shut += 1

    def list_apps(self):
        self.listed += 1
        if self.fail is not None:
            raise self.fail
        return [SimpleNamespace(name="gedit")]

    def observe(self, *a):  # pragma: no cover - never called by a probe
        raise AssertionError("a probe must not observe a window")

    def act(self, *a):  # pragma: no cover - never called by a probe
        raise AssertionError("a probe must never act")


def _ended():
    return DriverSessionEnded(
        f"list_apps failed: the driver session ended ({ENDED_MESSAGE})")


def _make(failures):
    """An integration whose Nth adapter's ``list_apps`` raises failures[N]
    (None = healthy). Adapters past the end of the list are healthy."""
    made: list[_Adapter] = []

    def factory(target):
        fail = failures[len(made)] if len(made) < len(failures) else None
        made.append(_Adapter(target, fail))
        return made[-1]

    integ = ComputerIntegration.from_config(
        {"computer_use": {"enabled": True, **LOCAL}},
        adapter_factory=factory,
        preconditions=lambda: PreconditionResult(
            act=HalfResult(HALF_OK, "x11-display"),
            observe=HalfResult(HALF_OK, "at-spi-bus")),
        version_reader=lambda: "0.28.2", clock=_Clock(), env={},
    )
    return integ, made


def _probe(integ):
    return asyncio.run(integ.probe(force=True))


def _check(snap, name):
    return next(c for c in snap["checks"] if c["name"] == name)


# ── ONE RETRY, FOR ONE CAUSE ────────────────────────────────────────────────

def test_an_ended_session_is_replaced_and_the_probe_passes_after_one_retry():
    """The live shape: a probe binds a driver, the session idles out, and the
    next forced probe finds ``session_ended``."""
    integ, made = _make([None])
    assert _probe(integ)["state"] == "ready"
    made[0].fail = _ended()

    snap = _probe(integ)

    assert snap["state"] == "ready", snap
    assert len(made) == 2, "not exactly one fresh driver"
    assert made[0].shut == 1, "the dead driver was kept"
    assert made[1].started == 1 and made[1].listed == 1
    assert integ.driver() is made[1]
    assert integ.targets.resolve("local") is made[1]
    assert _check(snap, "observe")["state"] == "ok"
    assert "session" in _check(snap, "observe")["detail"], (
        "a probe that replaced the driver says so")


def test_an_ended_session_on_the_first_probe_is_retried_too():
    integ, made = _make([_ended()])
    snap = _probe(integ)
    assert snap["state"] == "ready", snap
    assert len(made) == 2
    assert made[0].shut == 1
    assert integ.driver() is made[1]


def test_a_second_ended_session_refuses_and_is_not_retried_again():
    integ, made = _make([_ended(), _ended(), _ended()])
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert len(made) == 2, "retried more than once"
    assert [a.shut for a in made] == [1, 1], "a failed driver was kept"
    observe = _check(snap, "observe")
    assert observe["state"] == "down"
    assert "session" in observe["detail"]
    assert integ.driver() is None
    with pytest.raises(DriverUnavailable):
        integ.targets.resolve("local")


@pytest.mark.parametrize("failure", [
    DriverUnavailable("list_apps failed: Tool: tool='list_apps', "
                      "message='denied', error_code='permission_denied'"),
    # The words alone are not the cause: only the adapter's typed exception
    # (from the SDK's own error_code) is.
    DriverUnavailable(f"list_apps failed: {ENDED_MESSAGE} session_ended"),
], ids=["another-driver-error", "text-mentioning-session_ended"])
def test_a_different_observe_error_is_not_retried(failure):
    integ, made = _make([failure, None])
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert len(made) == 1, "a failure that was not session_ended was retried"
    assert made[0].listed == 1 and made[0].shut == 1
    assert integ.driver() is None


def test_an_empty_app_list_is_not_retried():
    integ, made = _make([None])
    _probe(integ)
    made[0].list_apps = lambda: []
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "no applications" in _check(snap, "observe")["detail"]
    assert len(made) == 1


def test_an_ended_session_is_not_retried_when_another_check_also_fails():
    """ONLY the observe step: here the MCP check is fine but the runtime
    refuses to start, so observe is never reached and nothing is retried."""
    integ, made = _make([None])

    def refuse():
        raise DriverUnavailable("no accessibility bus")
    real_factory = integ._adapter_factory

    def factory(target):
        adapter = real_factory(target)
        adapter.start = refuse
        return adapter
    integ._adapter_factory = factory
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert _check(snap, "runtime")["state"] == "down"
    assert len(made) == 1


# ── THE CAUSE IS TYPED: THE ADAPTER CLASSIFIES THE SDK'S ERROR ──────────────

class _ToolError(Exception):
    """``cua_driver.DriverError.Tool``'s shape (0.28.2): ``tool``,
    ``message`` and ``error_code`` attributes, and a str of all three."""

    def __init__(self, tool, message, error_code):
        super().__init__(f"tool={tool!r}, message={message!r}, "
                         f"error_code={error_code!r}")
        self.tool = tool
        self.message = message
        self.error_code = error_code


class _Sdk:
    ListAppsInput = staticmethod(lambda **kw: SimpleNamespace(**kw))


class _DeadDriver:
    def __init__(self, error):
        self.error = error

    def is_available(self):
        return True

    async def list_apps(self, inp):
        raise self.error

    async def shutdown(self):
        pass


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(cua, "_require_sdk", lambda: _Sdk)
    made = []

    def make(error):
        monkeypatch.setattr(_Sdk, "CuaDriver", SimpleNamespace(
            create=lambda: _DeadDriver(error)), raising=False)
        a = cua.CuaDriverAdapter(target="box")
        a.start()
        made.append(a)
        return a

    yield make
    for a in made:
        a.shutdown()


def test_the_adapter_raises_session_ended_for_the_sdks_error_code(adapter):
    a = adapter(_ToolError("list_apps", ENDED_MESSAGE, "session_ended"))
    with pytest.raises(DriverSessionEnded) as caught:
        a.list_apps()
    assert isinstance(caught.value, DriverUnavailable), (
        "every existing refusal path must still catch it")
    assert "session_ended" in str(caught.value)


@pytest.mark.parametrize("error", [
    _ToolError("list_apps", "denied", "permission_denied"),
    _ToolError("list_apps", "session is not available to this transport",
               "session_unavailable"),
    RuntimeError(f"session_ended: {ENDED_MESSAGE}"),
])
def test_any_other_sdk_error_stays_a_plain_driver_unavailable(adapter, error):
    a = adapter(error)
    with pytest.raises(DriverUnavailable) as caught:
        a.list_apps()
    assert not isinstance(caught.value, DriverSessionEnded)


# ── THE STATUS SAYS HOW OLD IT IS; A TASK STILL PROBES FRESH ────────────────

NOW = 1_800_000_000.0


class _CachedIntegration:
    """A probed integration: ``snapshot`` is the cache, ``probe`` counts."""

    enabled = True

    def __init__(self, checked_at):
        self.checked_at = checked_at
        self.probes: list[bool] = []

    def snapshot(self):
        return {"state": "ready", "version": "0.28.2", "checks": [],
                "checked_at": self.checked_at,
                "probed": self.checked_at is not None}

    async def probe(self, *, force=False):
        self.probes.append(force)
        return self.snapshot()

    def driver(self):
        return object()

    @property
    def state(self):
        return "ready"


def _iso(seconds_before_now):
    return dt.datetime.fromtimestamp(
        NOW - seconds_before_now, tz=dt.timezone.utc
    ).isoformat(timespec="seconds")


def _runner(integration):
    from prometheus.computer.task import ComputerTaskRunner

    return ComputerTaskRunner(
        integration=integration, gate=None, channel=None, people=None,
        wall=lambda: NOW)


@pytest.mark.parametrize("age, words", [
    (12, "checked 12s ago"),
    (240, "checked 4m ago"),
    (299, "checked 4m ago"),
    (2 * 3600 + 5, "checked 2h ago"),
    (3 * 86400, "checked 3d ago"),
])
def test_status_text_says_how_old_the_cached_probe_is(age, words):
    integ = _CachedIntegration(_iso(age))
    text = _runner(integ).status_text()
    assert words in text.splitlines()[0], text
    assert integ.probes == [], "a status read ran a probe"


def test_status_text_says_when_nothing_has_been_checked():
    integ = _CachedIntegration(None)
    text = _runner(integ).status_text()
    assert "not checked yet" in text.splitlines()[0], text
    assert "ago" not in text
    assert integ.probes == []


def test_a_task_still_forces_a_fresh_probe():
    """The age is for a person reading the status; a task never trusts the
    cache, however young."""
    integ = _CachedIntegration(_iso(1))
    asyncio.run(_runner(integ)._driver())
    assert integ.probes == [True]
