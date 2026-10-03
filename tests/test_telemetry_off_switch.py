"""`infrastructure.telemetry_enabled: false` turns telemetry off in the DAEMON too.

The CLI and the coding entry point have honoured the key for a long time
(``__main__.py``: no tracker when it is false). The daemon never read it:
``run_daemon`` built ``ToolCallTelemetry()`` unconditionally, so the off switch
that docs/guide/features.md and the README promise did nothing on the surface
most people run. Found during WP-X.54 T-3.

Ruled 2026-10-02: "off" means NO tracker (``None``), not a tracker that writes
nothing — every consumer already guards ``None``, and a write-nothing tracker
would need a flag check in every writer, forever. What "no tracker" costs is
handled explicitly, and each piece is pinned here:

- the daemon builds no tracker and creates no telemetry.db;
- the consumers that call telemetry without a guard (TelemetryDigest,
  GoldenTraceExporter) and the one that reads telemetry.db by path (GEPA) are
  not built;
- the coding live stream does not tail (its connect would create the file);
- /health and /events say telemetry is off instead of "restart required";
- /api/telemetry says ``enabled: false`` instead of a row of zeros;
- the cloud cost line in /status says "unavailable" instead of $0.00.

The Curator's half (no load data never revives an archived skill) lives in
tests/test_curator_load_based_staleness.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DAEMON = Path(__file__).resolve().parent.parent / "src" / "prometheus" / "daemon.py"


@pytest.fixture(autouse=True)
def _restore_handle():
    """Every test leaves the process-wide handle and the off flag as it found them."""
    from prometheus.telemetry import tracker

    prev_handle = tracker.get_telemetry_handle()
    prev_off = getattr(tracker, "_telemetry_off", False)
    yield
    tracker.set_telemetry_handle(prev_handle)
    tracker._telemetry_off = prev_off


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Point every config-dir lookup at tmp_path so a tracker lands there, not in ~."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# One reading of the key, shared by every entry point
# ---------------------------------------------------------------------------


class TestTheKey:
    @pytest.mark.parametrize("config, expected", [
        ({}, True),
        ({"infrastructure": {}}, True),
        ({"infrastructure": None}, True),
        ({"infrastructure": {"telemetry_enabled": True}}, True),
        ({"infrastructure": {"telemetry_enabled": False}}, False),
    ])
    def test_telemetry_enabled_reads_infrastructure_telemetry_enabled(self, config, expected):
        from prometheus.telemetry.tracker import telemetry_enabled

        assert telemetry_enabled(config) is expected


# ---------------------------------------------------------------------------
# The daemon's tracker
# ---------------------------------------------------------------------------


class TestDaemonTracker:
    def test_off_builds_no_tracker_and_no_database(self, config_dir):
        from prometheus.daemon import build_daemon_telemetry
        from prometheus.telemetry.tracker import get_telemetry_handle, telemetry_is_off

        tel = build_daemon_telemetry({"infrastructure": {"telemetry_enabled": False}})

        assert tel is None
        assert get_telemetry_handle() is None
        assert telemetry_is_off() is True
        assert not (config_dir / "telemetry.db").exists()

    @pytest.mark.parametrize("config", [
        {},
        {"infrastructure": {"telemetry_enabled": True}},
    ])
    def test_on_or_absent_builds_the_tracker_and_registers_it(self, config_dir, config):
        from prometheus.daemon import build_daemon_telemetry
        from prometheus.telemetry.tracker import (
            ToolCallTelemetry,
            get_telemetry_handle,
            telemetry_is_off,
        )

        tel = build_daemon_telemetry(config)
        try:
            assert isinstance(tel, ToolCallTelemetry)
            assert get_telemetry_handle() is tel
            assert telemetry_is_off() is False
            assert tel.db_path == (config_dir / "telemetry.db").resolve()
        finally:
            if tel is not None:
                tel.close()


def _run_daemon() -> ast.AsyncFunctionDef:
    tree = ast.parse(DAEMON.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_daemon":
            return node
    raise AssertionError("run_daemon not found in daemon.py — did the entrypoint move?")


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _callee(n) == name]


def _guarding_tests(fn: ast.AST, target: ast.AST) -> list[str]:
    """Source of every ``if`` test the target sits in the BODY of (not its else)."""
    out: list[str] = []

    def visit(node: ast.AST, guards: list[str]) -> bool:
        if node is target:
            out.extend(guards)
            return True
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for child in node.body:
                if visit(child, guards + [test]):
                    return True
            for child in node.orelse:
                if visit(child, guards):
                    return True
            return visit(node.test, guards)
        return any(visit(child, guards) for child in ast.iter_child_nodes(node))

    visit(fn, [])
    return out


class TestDaemonWiring:
    """Static pins over run_daemon, like tests/test_lcm_wired_before_gateway_start.py."""

    def test_run_daemon_takes_its_tracker_from_the_gated_builder(self):
        fn = _run_daemon()
        assert _calls(fn, "ToolCallTelemetry") == [], (
            "run_daemon constructs ToolCallTelemetry() directly — that ignores "
            "infrastructure.telemetry_enabled; use build_daemon_telemetry(config)"
        )
        assert len(_calls(fn, "build_daemon_telemetry")) == 1

    @pytest.mark.parametrize("consumer", [
        # Calls telemetry.report() unguarded: a traceback every dream cycle.
        "TelemetryDigest",
        # Calls telemetry.export_new_golden_traces() unguarded: a traceback every interval.
        "GoldenTraceExporter",
        # Reads telemetry.db BY PATH when handed None: frozen evidence, behind the switch's back.
        "GEPAOptimizer",
    ])
    def test_consumers_that_need_a_tracker_are_built_only_with_one(self, consumer):
        fn = _run_daemon()
        [call] = _calls(fn, consumer)
        guards = _guarding_tests(fn, call)
        assert any("telemetry is not None" in g for g in guards), (
            f"{consumer}(...) in run_daemon is not under `if ... telemetry is not None`; "
            f"guards found: {guards}"
        )

    def test_the_cloud_cost_tracker_is_told_when_telemetry_is_off(self):
        fn = _run_daemon()
        [call] = _calls(fn, "CostTracker")
        assert "unavailable_reason" in {kw.arg for kw in call.keywords}


# ---------------------------------------------------------------------------
# What the user sees
# ---------------------------------------------------------------------------


def _switch_off() -> None:
    from prometheus.telemetry.tracker import set_telemetry_handle, set_telemetry_off

    set_telemetry_handle(None)
    set_telemetry_off(True)


class TestCommands:
    def test_health_says_telemetry_is_off(self):
        from prometheus.gateway.commands import cmd_health

        _switch_off()
        text = cmd_health()
        assert "telemetry is off" in text
        assert "infrastructure.telemetry_enabled: false" in text
        assert "restart required" not in text

    def test_events_says_telemetry_is_off(self):
        from prometheus.gateway.commands import cmd_events

        _switch_off()
        text = cmd_events("")
        assert "telemetry is off" in text
        assert "infrastructure.telemetry_enabled: false" in text
        assert "restart required" not in text

    def test_an_unwired_handle_still_says_unwired(self):
        """Off is a choice; a missing handle with telemetry ON is still a wiring fault."""
        from prometheus.gateway.commands import cmd_health
        from prometheus.telemetry.tracker import set_telemetry_handle

        set_telemetry_handle(None)
        assert "restart required" in cmd_health()


class TestCost:
    def test_an_unavailable_tracker_says_so_instead_of_zero(self):
        from prometheus.telemetry.cost import CostTracker

        tracker = CostTracker(unavailable_reason="telemetry is off")
        text = tracker.report()
        assert text.startswith("Cost: unavailable")
        assert "telemetry is off" in text
        assert "$0.00" not in text

    def test_status_renders_unavailable(self):
        from unittest.mock import MagicMock

        from prometheus.gateway.commands import cmd_status
        from prometheus.telemetry.cost import CostTracker

        registry = MagicMock()
        registry.list_tools.return_value = []
        text = cmd_status("m", "anthropic", 0.0, registry,
                          CostTracker(unavailable_reason="telemetry is off"))
        assert "Cost: unavailable" in text

    def test_a_normal_tracker_is_unchanged(self):
        from prometheus.telemetry.cost import CostTracker

        assert CostTracker().report() == "Cost: $0.00 (no cloud API usage)"


fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def app_off(config_dir, monkeypatch):
    from prometheus.web.server import create_app

    monkeypatch.setattr("prometheus.config.paths.get_config_dir", lambda: config_dir)
    _switch_off()
    return create_app(config={"model": {"model": "test-model", "provider": "test"}})


class TestWeb:
    def test_api_telemetry_says_disabled_not_zero(self, app_off):
        from fastapi.testclient import TestClient

        body = TestClient(app_off).get("/api/telemetry").json()
        assert body["enabled"] is False

    def test_api_telemetry_says_enabled_with_a_tracker(self, config_dir, monkeypatch):
        from fastapi.testclient import TestClient

        from prometheus.telemetry.tracker import ToolCallTelemetry
        from prometheus.web.server import create_app

        monkeypatch.setattr("prometheus.config.paths.get_config_dir", lambda: config_dir)
        tel = ToolCallTelemetry(db_path=config_dir / "telemetry.db")
        try:
            app = create_app(config={"model": {"model": "m", "provider": "test"}}, telemetry=tel)
            body = TestClient(app).get("/api/telemetry").json()
            assert body["enabled"] is True
            assert "total_calls" in body
        finally:
            tel.close()

    def test_no_coding_live_stream_when_off(self, app_off):
        """No tail when off: its connect would CREATE telemetry.db (at a hard-coded
        ~/.prometheus fallback, since there is no tracker to name the path), and the
        coding subprocess reads the same key, so there would be no rows to tail."""
        assert app_off.state.coding_stream is None
