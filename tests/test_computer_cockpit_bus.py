"""The cockpit has a signal bus when SENTINEL is off (found on the computer-use rig, 2026-10-23).

The cockpit — the app pick, each step, the task's end, thumbnails, and desktop prompts pushed to
Beacon — rides the SignalBus. Only SENTINEL built one, and SENTINEL ships off, so a daemon with
``computer_use.enabled: true`` and SENTINEL off emitted nothing: Beacon picked an app, the daemon
bound it (200), and the pane kept saying "No app picked yet" because the binding frame never came.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from tests.test_computer_door import (  # noqa: F401 - _linux_site_rule is an autouse fixture
    APP,
    PERSON,
    SESSION,
    _linux_site_rule,
    _rig,
)
from tests.test_computer_registration_pin import _enabled, _shipped

DAEMON = Path(__file__).resolve().parent.parent / "src" / "prometheus" / "daemon.py"


def _wiring(config, tmp_path):
    from prometheus.computer.wiring import wire_computer_use
    from prometheus.permissions.audit import AuditLogger
    from prometheus.permissions.checker import PermissionMode, SecurityGate
    from prometheus.tools.base import ToolRegistry

    gate = SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))
    return wire_computer_use(config, gate=gate, registry=ToolRegistry(), device_store=None,
                             telegram_adapter=None, telegram_user_ids=(),
                             adapter_factory=lambda target: None)


def test_off_builds_no_bus(tmp_path):
    wiring = _wiring(_shipped(), tmp_path)
    assert wiring.cockpit_bus() is None


def test_on_builds_a_bus_the_desktop_prompts_ride(tmp_path):
    from prometheus.sentinel.signals import SignalBus

    wiring = _wiring(_enabled(), tmp_path)
    try:
        bus = wiring.cockpit_bus()
        assert isinstance(bus, SignalBus)
        assert wiring.channel.signal_bus is bus, "desktop prompts reach Beacon on the same bus"
    finally:
        wiring.close()


async def test_a_pick_reaches_a_client_through_the_cockpit_bus(tmp_path):
    """The launcher's wiring, given the bus the daemon now builds: the binding frame goes out."""
    from prometheus.computer.livestream import ComputerLiveStream
    from prometheus.computer.wiring import ComputerWiring
    from prometheus.web.ws_server import WebSocketBridge

    rig = _rig(tmp_path, [])
    wiring = ComputerWiring(integration=SimpleNamespace(close=lambda: None),
                            channel=rig.channel, runner=rig.runner)
    bus = wiring.cockpit_bus()

    frames: list[dict] = []

    class _Client:
        async def send(self, raw: str) -> None:
            frames.append(json.loads(raw))

    bridge = WebSocketBridge(loop_context=object())
    bridge._clients.add(_Client())
    bridge.computer_runner = rig.runner
    bus.subscribe("*", bridge._on_signal)
    rig.runner.live = ComputerLiveStream(bus, bridge=bridge)

    await rig.runner.bind(SESSION, APP, scope="session", by=PERSON, surface="beacon")

    bindings = [f for f in frames if f.get("type") == "computer_binding"]
    assert bindings, f"no computer_binding frame; got {[f.get('type') for f in frames]}"
    assert bindings[-1]["payload"]["state"] == "on"


def test_the_daemon_asks_for_the_cockpit_bus_when_sentinel_built_none():
    """Pinned in the source: before the web bridge is launched, a daemon whose SENTINEL built no
    bus asks the computer wiring for one, and only then."""
    tree = ast.parse(DAEMON.read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "cockpit_bus"]
    assert len(calls) == 1, "exactly one place asks for the cockpit bus"
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.If)
              and any(c is calls[0] for c in ast.walk(n))
              and "signal_bus" in ast.unparse(n.test) and "not in dir()" in ast.unparse(n.test)]
    assert guards, "the call must be guarded by `\"signal_bus\" not in dir()`"
    launch = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", None) == "launch_web")
    assert calls[0].lineno < launch.lineno, "the bus must exist before the web bridge is launched"
