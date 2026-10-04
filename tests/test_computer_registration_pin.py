"""The registration pin, widened to EVERY ``computer_*`` path (design §6.1).

"Computer tools are not registered" is a ruling. Until PR 5 one grep pinned
it (``test_computer_status_block::test_the_daemon_registers_none_today``),
and the grep stays. The door adds the first runtime that reaches the driver,
so the pin now also holds:

1. BY EXECUTION AT BOOT — the registry the daemon builds, plus the door's
   wiring (``computer.wiring.wire_computer_use``), with the shipped config
   AND with ``computer_use.enabled: true``. No tool name starts with
   ``computer_``; ``computer.registered`` is 0 either way.
2. THROUGH THE DOOR'S RUNTIME PATHS — a spy on ``ToolRegistry.register``
   stays installed through a whole task (bind → start → steps → stop).
3. BY SCHEMA, NOT ONLY BY NAME — no registered tool declares
   ``x-prometheus-computer-verb``, so a wrapper named ``desktop_task`` cannot
   slip through.
4. BY CALLER — ``ComputerTaskRunner.start`` refuses inside a tool call
   (``test_computer_door.test_a_start_from_inside_a_tool_call_is_refused``).
5. THE SHIPPED DEFAULT IS OFF, and off constructs nothing.

Registering ``computer_task`` as a tool a model can call is L1: a separate
ruling. Until then this file says 0 — with the switch off AND on. That is a
deliberate decision; update this test and say so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prometheus.computer.status import _TOOL_PREFIX, computer_status
from prometheus.permissions.computer_schema import COMPUTER_VERB_KEY

TEMPLATE = (Path(__file__).resolve().parents[1] / "config"
            / "prometheus.yaml.default")


def _shipped() -> dict:
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


def _enabled() -> dict:
    cfg = _shipped()
    cfg["computer_use"] = {
        **(cfg.get("computer_use") or {}),
        "enabled": True,
        "targets": {"local": {"kind": "local"}},
    }
    return cfg


def _computer_named(registry) -> list[str]:
    return [t.name for t in registry.list_tools()
            if str(t.name).startswith(_TOOL_PREFIX)]


def _computer_schemas(registry) -> list[str]:
    hits = []
    for tool in registry.list_tools():
        try:
            schema = tool.to_api_schema()
        except Exception:  # noqa: BLE001 - a tool with no schema declares nothing
            continue
        if COMPUTER_VERB_KEY in repr(schema):
            hits.append(tool.name)
    return hits


class _NoDriverIntegration:
    """``wire_computer_use`` probes nothing here: the pin is about the
    REGISTRY, and a boot probe on a CI box has no display to find."""


@pytest.mark.parametrize("config", [_shipped(), _enabled()],
                         ids=["shipped", "enabled"])
def test_boot_wiring_registers_no_computer_tool(config, tmp_path):
    from prometheus.computer.wiring import wire_computer_use
    from prometheus.daemon import build_tool_registry
    from prometheus.permissions.audit import AuditLogger
    from prometheus.permissions.checker import PermissionMode, SecurityGate

    registry = build_tool_registry(security_cfg=config.get("security") or {})
    gate = SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))
    wiring = wire_computer_use(config, gate=gate, registry=registry,
                               device_store=None, telegram_adapter=None,
                               telegram_user_ids=(), probe_at_boot=False)
    try:
        assert _computer_named(registry) == []
        assert _computer_schemas(registry) == []
        assert computer_status(registry)["registered"] == 0
        if config["computer_use"]["enabled"] is True:
            assert wiring.runner is not None, "on means the door exists"
        else:
            assert wiring.runner is None, "off means no door at all"
            assert wiring.integration.enabled is False
    finally:
        wiring.close()


def test_off_constructs_no_driver_and_starts_no_runtime(tmp_path):
    from prometheus.computer.wiring import wire_computer_use
    from prometheus.permissions.audit import AuditLogger
    from prometheus.permissions.checker import PermissionMode, SecurityGate
    from prometheus.tools.base import ToolRegistry

    built = []
    gate = SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))
    wiring = wire_computer_use(
        _shipped(), gate=gate, registry=ToolRegistry(), device_store=None,
        telegram_adapter=None, telegram_user_ids=(), probe_at_boot=False,
        adapter_factory=lambda target: built.append(target))
    assert wiring.integration.driver() is None
    assert wiring.channel is None and wiring.runner is None
    assert built == [], "a disabled switch must not construct the driver"


async def test_no_registration_through_a_whole_door_task(tmp_path, monkeypatch):
    """A spy on ToolRegistry.register for the length of bind → start →
    steps → stop. Any call fails with the caller's location."""
    import traceback

    from prometheus.tools.base import ToolRegistry
    from tests.test_computer_door import (
        APP,
        PERSON,
        SESSION,
        TYPED,
        _next_pending,
        _rig,
        _start,
    )

    calls: list[str] = []
    original = ToolRegistry.register

    def spy(self, tool, *a, **k):
        calls.append(f"{getattr(tool, 'name', tool)} from "
                     f"{traceback.format_stack(limit=4)[0].strip()}")
        return original(self, tool, *a, **k)

    monkeypatch.setattr(ToolRegistry, "register", spy)
    rig = _rig(tmp_path, ["click-0", "set-2"])
    await rig.runner.bind(SESSION, APP, scope="session", by=PERSON,
                          surface="rest")
    task = await _start(rig, text=TYPED)
    await _next_pending(rig.channel, set())
    rig.runner.stop(task.task_id)
    await rig.runner.wait(task.task_id, timeout=10)
    assert calls == [], calls


def test_the_shipped_default_is_off():
    block = _shipped()["computer_use"]
    assert block["enabled"] is False


def test_only_a_literal_true_switches_it_on():
    from prometheus.computer.integration import computer_use_enabled

    for value, want in ((True, True), ("true", False), ("yes", False),
                        (1, False), (None, False), (False, False)):
        assert computer_use_enabled({"computer_use": {"enabled": value}}) is want
