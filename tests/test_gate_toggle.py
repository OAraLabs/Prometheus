"""Tests for the runtime permission-gate toggle (Telegram /gate).

Covers SecurityGate.set_mode / current_mode and the cmd_gate command layer.
No Telegram adapter, network, or tokens — pure command-layer tests.
"""

from __future__ import annotations

import pytest

from prometheus.gateway.commands import cmd_gate
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.modes import PermissionMode


def _gate(mode: str = "default") -> SecurityGate:
    return SecurityGate(workspace_root="/tmp", mode=mode)


class TestSetMode:
    def test_set_mode_by_string(self):
        gate = _gate()
        assert gate.set_mode("autonomous") == PermissionMode.AUTONOMOUS
        assert gate.current_mode() == PermissionMode.AUTONOMOUS

    def test_set_mode_by_enum(self):
        gate = _gate()
        assert gate.set_mode(PermissionMode.STRICT) == PermissionMode.STRICT

    def test_set_mode_rejects_unknown(self):
        gate = _gate()
        with pytest.raises(ValueError):
            gate.set_mode("nonsense")

    def test_autonomous_mode_suppresses_approval(self):
        gate = _gate()
        # write_file outside workspace -> APPROVE in default mode
        decision = gate.evaluate(
            "write_file", file_path="/opt/outside.txt", origin="system"
        )
        assert decision.action == "APPROVE"
        # toggle off -> same call is now ALLOWed
        gate.set_mode("autonomous")
        decision = gate.evaluate(
            "write_file", file_path="/opt/outside.txt", origin="system"
        )
        assert decision.action == "ALLOW"

    def test_autonomous_mode_still_blocks(self):
        gate = _gate()
        gate.set_mode("autonomous")
        decision = gate.evaluate("bash", command="rm -rf /", origin="user")
        assert decision.action == "DENY"

    def test_toggle_on_restores_default(self):
        gate = _gate()
        gate.set_mode("autonomous")
        gate.set_mode("default")
        assert gate.current_mode() == PermissionMode.DEFAULT
        decision = gate.evaluate(
            "write_file", file_path="/opt/outside.txt", origin="system"
        )
        assert decision.action == "APPROVE"


class TestCmdGate:
    def test_no_gate(self):
        assert "No permission gate" in cmd_gate(None)

    def test_status_default(self):
        text = cmd_gate(_gate())
        assert "default" in text

    def test_off_sets_autonomous(self):
        gate = _gate()
        text = cmd_gate(gate, "off")
        assert "OFF" in text
        assert gate.current_mode() == PermissionMode.AUTONOMOUS

    def test_on_restores_default(self):
        gate = _gate()
        cmd_gate(gate, "off")
        text = cmd_gate(gate, "on")
        assert "ON" in text
        assert gate.current_mode() == PermissionMode.DEFAULT

    def test_strict(self):
        gate = _gate()
        text = cmd_gate(gate, "strict")
        assert gate.current_mode() == PermissionMode.STRICT
        assert "STRICT" in text

    def test_unknown_arg(self):
        gate = _gate()
        text = cmd_gate(gate, "bogus")
        assert "Unknown gate mode" in text
        assert gate.current_mode() == PermissionMode.DEFAULT  # unchanged

    def test_bypass_alias(self):
        gate = _gate()
        cmd_gate(gate, "bypass")
        assert gate.current_mode() == PermissionMode.AUTONOMOUS
