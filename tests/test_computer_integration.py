"""The Cua driver as an Integration (computer-use v1.1, PR 2; design §5.3).

Decision 4: the driver is an Integration, never a builtin — health KNOWN
BEFORE DISPATCH, refusal like the backend registry, and nothing constructed
while it is off. This file pins:

* **Off means nothing.** ``computer_use.enabled`` defaults to false, only a
  literal ``true`` turns it on (a quoted "false" must not), and while it is off
  no driver is constructed and no probe touches the box.
* **The telemetry floor.** cua-driver phones home by default and 0.28.2
  ignores ``DO_NOT_TRACK``. Both of its own opt-outs are forced to "0" BEFORE
  ``import cua_driver`` — proven in a subprocess against a stand-in package
  that records what it saw at import time.
* **Supported versions are a code constant.** A driver this adapter was not
  validated against reports ``version-mismatch`` and the integration is down.
* **The probe.** Ordered checks, each recorded and never raised: telemetry,
  config, driver version, a declared local target, the substrate halves, the
  runtime starting, the observe half answering (``list_apps``), and no MCP
  server ALSO running ``cua-driver`` (whose raw tools would bypass the
  candidate table). Down, degraded or ready — and TTL-cached, so a status
  read never probes.

The driver leg itself is uncoverable by CI (cua.py's module docstring); every
probe here runs over fakes.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.computer import integration as integ_mod
from prometheus.computer.driver import (
    HALF_OK, HALF_UNAVAILABLE, DriverUnavailable, HalfResult, PreconditionResult,
)
from prometheus.computer.integration import (
    SUPPORTED_DRIVER_VERSIONS, TELEMETRY_FLOOR, ComputerIntegration,
    apply_telemetry_floor, mcp_servers_running_cua_driver,
)

REPO = Path(__file__).resolve().parents[1]
LOCAL = {"targets": {"local": {"kind": "local"}}}


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _Adapter:
    """Stands in for CuaDriverAdapter. Counts what the probe does to it."""

    def __init__(self, target, *, start_error=None, apps=("gedit",),
                 discovery=True):
        self.target = target
        self.start_error = start_error
        self._apps = list(apps)
        self.started = 0
        self.shut = 0
        if not discovery:
            self.list_apps = None  # type: ignore[assignment]

    def start(self):
        self.started += 1
        if self.start_error:
            raise DriverUnavailable(self.start_error)

    def shutdown(self):
        self.shut += 1

    def list_apps(self):
        return [SimpleNamespace(name=a) for a in self._apps]

    def observe(self, *a):  # pragma: no cover - never called by a probe
        raise AssertionError("a probe must not observe a window")

    def act(self, *a):  # pragma: no cover - never called by a probe
        raise AssertionError("a probe must never act")


def _ready_pre():
    return PreconditionResult(act=HalfResult(HALF_OK, "x11-display"),
                              observe=HalfResult(HALF_OK, "at-spi-bus"))


def _make(block=None, *, adapter_kw=None, pre=_ready_pre, version="0.28.2",
          mcp=None, clock=None, env=None):
    made: list[_Adapter] = []

    def factory(target):
        made.append(_Adapter(target, **(adapter_kw or {})))
        return made[-1]

    calls = {"pre": 0}

    def preconditions():
        calls["pre"] += 1
        return pre()

    integ = ComputerIntegration.from_config(
        {"computer_use": {"enabled": True, **LOCAL, **(block or {})}},
        mcp_servers=mcp or {},
        adapter_factory=factory, preconditions=preconditions,
        version_reader=lambda: version, clock=clock or _Clock(),
        env={} if env is None else env,
    )
    return integ, made, calls


def _probe(integ, force=False):
    return asyncio.run(integ.probe(force=force))


def _check(snap, name):
    return next(c for c in snap["checks"] if c["name"] == name)


# ── OFF MEANS NOTHING ───────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, False, "true", "True", "false", 1,
                                   "yes"])
def test_only_a_literal_true_enables_it(value):
    """`resolve_telegram_enabled` uses bool(value), so a quoted "false"
    enables Telegram. This key must not repeat that."""
    block = {} if value is None else {"enabled": value}
    integ = ComputerIntegration.from_config({"computer_use": block})
    assert integ.enabled is False


def test_disabled_constructs_nothing_and_probes_nothing():
    made: list = []
    calls = {"pre": 0}
    integ = ComputerIntegration.from_config(
        {"computer_use": {"enabled": False, **LOCAL}},
        adapter_factory=lambda t: made.append(t),
        preconditions=lambda: calls.__setitem__("pre", calls["pre"] + 1),
        version_reader=lambda: pytest.fail("a disabled probe read the version"))
    snap = _probe(integ, force=True)
    assert snap["state"] == "disabled"
    assert made == [] and calls["pre"] == 0
    assert integ.targets is None
    assert integ.driver() is None


def test_a_missing_block_is_disabled():
    assert ComputerIntegration.from_config({}).enabled is False


# ── THE TELEMETRY FLOOR ─────────────────────────────────────────────────────

def test_the_floor_forces_both_opt_outs_off():
    env = {"CUA_DRIVER_RS_TELEMETRY_ENABLED": "1", "CUA_TELEMETRY_ENABLED": "1"}
    apply_telemetry_floor(env)
    assert env == {"CUA_DRIVER_RS_TELEMETRY_ENABLED": "0",
                   "CUA_TELEMETRY_ENABLED": "0"}
    assert set(TELEMETRY_FLOOR) == set(env)


def test_the_floor_is_set_before_the_sdk_is_imported(tmp_path):
    """A stand-in `cua_driver` records os.environ AT IMPORT TIME. The real
    SDK reads its telemetry setting when it loads, so set-after-import would
    be too late — this is the only ordering that counts."""
    pkg = tmp_path / "cua_driver"
    pkg.mkdir()
    seen = tmp_path / "seen.txt"
    (pkg / "__init__.py").write_text(textwrap.dedent(f"""
        import os
        with open({str(seen)!r}, "w") as fh:
            fh.write(os.environ.get("CUA_DRIVER_RS_TELEMETRY_ENABLED", "unset")
                     + "," + os.environ.get("CUA_TELEMETRY_ENABLED", "unset"))
    """))
    env = {k: v for k, v in os.environ.items()
           if k not in TELEMETRY_FLOOR}
    env.update({"CUA_DRIVER_RS_TELEMETRY_ENABLED": "1",
                "CUA_TELEMETRY_ENABLED": "1",
                "PYTHONPATH": f"{tmp_path}{os.pathsep}{REPO / 'src'}"})
    subprocess.run(
        [sys.executable, "-c",
         "from prometheus.computer.cua import _require_sdk; _require_sdk()"],
        env=env, check=True, cwd=tmp_path, timeout=120)
    assert seen.read_text() == "0,0"


def test_the_probe_reports_telemetry_forced_off():
    integ, _, _ = _make()
    snap = _probe(integ)
    assert snap["telemetry"] == "forced_off"
    assert _check(snap, "telemetry")["state"] == "ok"


# ── SUPPORTED VERSIONS ──────────────────────────────────────────────────────

def test_the_supported_versions_are_the_pin():
    assert SUPPORTED_DRIVER_VERSIONS == frozenset({"0.28.2"})


def test_an_unvalidated_driver_is_a_version_mismatch():
    integ, made, _ = _make(version="0.33.1")
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "version-mismatch" in _check(snap, "driver")["detail"]
    assert made == [], "a runtime was started on an unvalidated driver"


def test_a_missing_driver_is_down_and_says_what_to_install():
    integ, made, _ = _make(version=None)
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "--extra computer" in _check(snap, "driver")["detail"]
    assert made == []


# ── THE PROBE MATRIX ────────────────────────────────────────────────────────

def test_everything_answering_is_ready_and_binds_the_target():
    integ, made, _ = _make()
    snap = _probe(integ)
    assert snap["state"] == "ready", snap
    assert [c["name"] for c in snap["checks"]] == [
        "telemetry", "config", "driver", "target", "substrate", "runtime",
        "observe", "mcp"]
    assert all(c["state"] == "ok" for c in snap["checks"])
    assert snap["version"] == "0.28.2"
    assert snap["execution_mode"] == "embedded"
    assert integ.targets.resolve("local") is made[0]
    assert integ.driver() is made[0]


def test_a_dead_substrate_is_down_before_any_runtime_starts():
    def dead():
        return PreconditionResult(
            act=HalfResult(HALF_UNAVAILABLE, "x11-display", "no DISPLAY"),
            observe=HalfResult(HALF_OK, "at-spi-bus"))
    integ, made, _ = _make(pre=dead)
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "no DISPLAY" in _check(snap, "substrate")["detail"]
    assert made == []


def test_a_runtime_that_will_not_start_is_down_and_unbound():
    integ, made, _ = _make(adapter_kw={"start_error": "no accessibility bus"})
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "no accessibility bus" in _check(snap, "runtime")["detail"]
    assert integ.driver() is None
    with pytest.raises(DriverUnavailable):
        integ.targets.resolve("local")


def test_an_empty_app_list_is_down_not_an_idle_desktop():
    integ, made, _ = _make(adapter_kw={"apps": ()})
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "no applications" in _check(snap, "observe")["detail"]
    assert made[0].shut == 1, "a runtime that failed its probe was kept"
    assert integ.driver() is None


def test_a_driver_without_discovery_is_degraded_not_ready():
    """Until discovery (PR 4) lands, the observe half cannot be established
    by the probe — said, not assumed."""
    integ, _, _ = _make(adapter_kw={"discovery": False})
    snap = _probe(integ)
    assert snap["state"] == "degraded"
    assert _check(snap, "observe")["state"] == "unknown"


def test_no_declared_local_target_is_down():
    integ, made, _ = _make(block={"targets": {}})
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "no local target" in _check(snap, "target")["detail"]
    assert made == []


def test_a_remote_target_is_refused_as_a_config_error():
    integ, made, _ = _make(block={"targets": {
        "local": {"kind": "local"}, "laptop": {"kind": "remote"}}})
    snap = _probe(integ)
    assert any("laptop" in e and "local only" in e
               for e in snap["config_errors"])
    assert _check(snap, "config")["state"] == "down"
    assert snap["state"] == "down"


def test_a_malformed_targets_block_is_a_config_error_not_a_crash():
    integ, _, _ = _make(block={"targets": ["local"]})
    snap = _probe(integ)
    assert snap["config_errors"]
    assert snap["state"] == "down"


@pytest.mark.parametrize("server", [
    {"command": "cua-driver", "args": ["mcp"]},
    {"command": "/usr/local/bin/cua-driver", "args": ["mcp"]},
    {"command": "uvx", "args": ["cua-driver", "mcp"]},
    {"command": "python", "args": ["-m", "cua_driver", "mcp"]},
])
def test_cua_driver_also_configured_as_mcp_is_degraded(server):
    """Its raw `mcp__` tools would bypass the candidate table and the
    extent. They still prompt on every call, so this is degraded, not down."""
    integ, _, _ = _make(mcp={"desktop": server, "fs": {"command": "npx",
                                                        "args": ["fs"]}})
    snap = _probe(integ)
    assert snap["state"] == "degraded"
    assert "cua-driver-also-configured-as-mcp" in _check(snap, "mcp")["detail"]
    assert "desktop" in _check(snap, "mcp")["detail"]
    assert mcp_servers_running_cua_driver(
        {"desktop": server, "fs": {"command": "npx"}}) == ["desktop"]


def test_a_probe_that_raises_is_recorded_never_raised():
    def boom():
        raise RuntimeError("probe exploded")
    integ, _, _ = _make(pre=boom)
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "probe exploded" in str(snap["checks"])


def test_an_unexpected_error_after_start_releases_the_runtime():
    """Whatever breaks mid-probe, a started runtime is not left running
    unbound behind a "down" answer."""
    integ, made, _ = _make()

    def broken():
        raise ValueError("driver returned garbage")
    real_factory = integ._adapter_factory

    def factory(target):
        adapter = real_factory(target)
        adapter.list_apps = broken
        return adapter
    integ._adapter_factory = factory
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "driver returned garbage" in str(snap["checks"])
    assert made[0].started == 1 and made[0].shut == 1
    assert integ.driver() is None


def test_a_hung_probe_is_bounded(monkeypatch):
    import time as _time

    def slow():
        _time.sleep(3)
        return _ready_pre()
    integ, _, _ = _make(pre=slow, block={"probe": {"timeout_s": 0.2}})
    snap = _probe(integ)
    assert snap["state"] == "down"
    assert "timed out" in str(snap["checks"])


# ── TTL, AND A STATUS READ NEVER PROBES ─────────────────────────────────────

def test_the_probe_is_ttl_cached_and_force_reprobes():
    clock = _Clock()
    integ, made, calls = _make(clock=clock)
    _probe(integ)
    _probe(integ)
    assert calls["pre"] == 1
    _probe(integ, force=True)
    assert calls["pre"] == 2
    clock.t += 61
    _probe(integ)
    assert calls["pre"] == 3


def test_snapshot_does_no_io_and_says_when_nothing_was_probed():
    integ, made, calls = _make()
    snap = integ.snapshot()
    assert snap["state"] == "unknown" and snap["probed"] is False
    assert calls["pre"] == 0 and made == []


def test_concurrent_probes_share_one_run():
    integ, _, calls = _make()

    async def both():
        await asyncio.gather(integ.probe(force=True), integ.probe())
    asyncio.run(both())
    assert calls["pre"] == 1


def test_close_shuts_the_runtime_down_and_unbinds():
    integ, made, _ = _make()
    _probe(integ)
    integ.close()
    assert made[0].shut == 1
    assert integ.driver() is None


# ── THE SURFACES ────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


def _client(integ):
    cfg = {"model": {"model": "m", "provider": "llama_cpp"}}
    return TestClient(create_app(cfg, computer_integration=integ))


def test_the_route_probes_through_the_ttl_and_post_forces():
    integ, _, calls = _make()
    client = _client(integ)
    body = client.get("/api/integrations/computer").json()
    assert body["state"] == "ready"
    client.get("/api/integrations/computer")
    assert calls["pre"] == 1
    client.post("/api/integrations/computer/probe")
    assert calls["pre"] == 2


def test_no_integration_is_a_503_not_an_invented_state():
    r = TestClient(create_app({"model": {"model": "m"}})).get(
        "/api/integrations/computer")
    assert r.status_code == 503


def test_a_disabled_integration_answers_disabled():
    integ = ComputerIntegration.from_config({"computer_use": {}})
    assert _client(integ).get("/api/integrations/computer").json()[
        "state"] == "disabled"


def test_status_renders_the_driver_from_cache_without_probing():
    integ, _, calls = _make()
    client = _client(integ)
    client.get("/api/integrations/computer")
    before = calls["pre"]
    computer = client.get("/api/status").json()["computer"]
    assert calls["pre"] == before, "/api/status ran the integration probe"
    assert computer["driver"]["state"] == "ready"
    assert computer["driver"]["telemetry"] == "forced_off"
    assert [t["name"] for t in computer["targets"]] == ["local"]
