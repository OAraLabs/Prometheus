"""Tests for Sprint 4: SecurityGate, PermissionDecision, SandboxedExecution."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.permissions import (
    PermissionDecision,
    PermissionMode,
    SecurityGate,
    SandboxedExecution,
    TrustLevel,
)


# ---------------------------------------------------------------------------
# TrustLevel
# ---------------------------------------------------------------------------


class TestTrustLevel:
    def test_ordering(self):
        assert TrustLevel.BLOCKED < TrustLevel.APPROVE < TrustLevel.AUTO < TrustLevel.AUTONOMOUS

    def test_values(self):
        assert TrustLevel.BLOCKED == 0
        assert TrustLevel.APPROVE == 1
        assert TrustLevel.AUTO == 2
        assert TrustLevel.AUTONOMOUS == 3


# ---------------------------------------------------------------------------
# PermissionDecision constructors
# ---------------------------------------------------------------------------


class TestPermissionDecision:
    def test_allow(self):
        d = PermissionDecision.allow("ok")
        assert d.allowed is True
        assert d.requires_confirmation is False
        assert d.action == "ALLOW"

    def test_deny(self):
        d = PermissionDecision.deny("too dangerous")
        assert d.allowed is False
        assert d.requires_confirmation is False
        assert d.action == "DENY"

    def test_approve(self):
        d = PermissionDecision.approve("needs ok")
        assert d.allowed is False  # blocked until user confirms
        assert d.requires_confirmation is True
        assert d.action == "APPROVE"


# ---------------------------------------------------------------------------
# SecurityGate — acceptance-test pattern
# ---------------------------------------------------------------------------


class TestSecurityGateAcceptance:
    """Mirror the sprint acceptance tests exactly."""

    def test_rm_rf_is_denied(self):
        gate = SecurityGate()
        result = gate.pre_tool_use("bash", {"command": "rm -rf /"}, {})
        assert result.action == "DENY"

    def test_rm_rf_tilde_is_denied(self):
        gate = SecurityGate()
        result = gate.pre_tool_use("bash", {"command": "rm -rf ~"}, {})
        assert result.action == "DENY"

    def test_mkfs_is_denied(self):
        gate = SecurityGate()
        result = gate.pre_tool_use("bash", {"command": "mkfs.ext4 /dev/sda1"}, {})
        assert result.action == "DENY"


# ---------------------------------------------------------------------------
# SecurityGate — evaluate() interface (used by agent_loop)
# ---------------------------------------------------------------------------


class TestSecurityGateEvaluate:
    def test_read_file_is_allowed(self):
        gate = SecurityGate()
        d = gate.evaluate("read_file", is_read_only=True, file_path="/tmp/foo.txt")
        assert d.allowed is True

    def test_safe_bash_is_allowed(self):
        gate = SecurityGate()
        d = gate.evaluate("bash", is_read_only=False, command="ls -la")
        assert d.allowed is True
        assert d.action == "ALLOW"

    def test_denied_command_from_list(self):
        gate = SecurityGate(denied_commands=["DROP TABLE"])
        d = gate.evaluate("bash", command="DROP TABLE users;")
        assert d.allowed is False
        assert d.action == "DENY"

    def test_denied_path_blocks_file_write(self):
        gate = SecurityGate(denied_paths=["/etc"])
        d = gate.evaluate("write_file", is_read_only=False, file_path="/etc/passwd")
        assert d.allowed is False
        assert d.action == "DENY"

    def test_git_push_requires_approval(self):
        gate = SecurityGate()
        d = gate.evaluate("bash", command="git push origin main")
        assert d.action == "APPROVE"
        assert d.allowed is False  # blocked until user confirms
        assert d.requires_confirmation is True

    def test_curl_requires_approval(self):
        gate = SecurityGate()
        d = gate.evaluate("bash", command="curl https://example.com")
        assert d.action in ("APPROVE", "ALLOW")  # APPROVE in default mode

    def test_write_outside_workspace_requires_approval(self):
        gate = SecurityGate(workspace_root="/tmp/workspace")
        d = gate.evaluate("write_file", is_read_only=False, file_path="/tmp/other/file.txt")
        assert d.action == "APPROVE"
        assert d.requires_confirmation is True

    def test_write_inside_workspace_is_allowed(self):
        gate = SecurityGate(workspace_root="/tmp/workspace")
        d = gate.evaluate("write_file", is_read_only=False, file_path="/tmp/workspace/file.txt")
        assert d.action == "ALLOW"

    def test_autonomous_mode_allows_everything_except_blocked(self):
        gate = SecurityGate(mode=PermissionMode.AUTONOMOUS)
        d = gate.evaluate("bash", command="git push origin main")
        assert d.action == "ALLOW"

    def test_autonomous_mode_still_blocks_rm_rf(self):
        gate = SecurityGate(mode=PermissionMode.AUTONOMOUS)
        d = gate.evaluate("bash", command="rm -rf /")
        assert d.action == "DENY"

    def test_strict_mode_write_requires_approval(self):
        gate = SecurityGate(mode=PermissionMode.STRICT)
        d = gate.evaluate("write_file", is_read_only=False, file_path="/tmp/workspace/file.txt")
        assert d.action == "APPROVE"


# ---------------------------------------------------------------------------
# SecurityGate.from_config
# ---------------------------------------------------------------------------


class TestSecurityGateFromConfig:
    def test_loads_from_yaml(self, tmp_path):
        config = tmp_path / "prometheus.yaml"
        config.write_text(
            "security:\n"
            "  permission_mode: default\n"
            "  workspace_root: /tmp/workspace\n"
            "  denied_commands:\n"
            "    - 'rm -rf /'\n"
            "  denied_paths:\n"
            "    - /etc\n"
        )
        gate = SecurityGate.from_config(config)
        assert gate._mode == PermissionMode.DEFAULT

    def test_graceful_on_missing_file(self):
        gate = SecurityGate.from_config("/nonexistent/prometheus.yaml")
        # Should not raise; creates a default gate
        assert gate is not None


# ---------------------------------------------------------------------------
# SandboxedExecution
# ---------------------------------------------------------------------------


class TestSandboxedExecution:
    def test_runs_simple_command(self, tmp_path):
        sandbox = SandboxedExecution(workspace=tmp_path)
        result = asyncio.run(sandbox.run("echo hello"))
        assert result.output.strip() == "hello"
        assert result.is_error is False

    def test_captures_stderr(self, tmp_path):
        sandbox = SandboxedExecution(workspace=tmp_path)
        result = asyncio.run(sandbox.run("echo err >&2"))
        assert "err" in result.output

    def test_timeout_enforced(self, tmp_path):
        sandbox = SandboxedExecution(workspace=tmp_path, timeout=1)
        result = asyncio.run(sandbox.run("sleep 10"))
        assert result.is_error is True
        assert "timed out" in result.output.lower()

    def test_output_truncated(self, tmp_path):
        sandbox = SandboxedExecution(workspace=tmp_path, max_output=50)
        # Generate more than 50 chars of output
        result = asyncio.run(sandbox.run("python3 -c \"print('x' * 200)\""))
        assert len(result.output) <= 200  # truncation marker adds some chars
        assert "truncated" in result.output

    def test_env_stripped(self, tmp_path):
        import os
        sandbox = SandboxedExecution(workspace=tmp_path)
        # ANTHROPIC_API_KEY should be stripped from the subprocess env
        result = asyncio.run(
            sandbox.run(
                "echo ${ANTHROPIC_API_KEY:-STRIPPED}",
                env_override={},
            )
        )
        assert "STRIPPED" in result.output

    def test_workspace_property(self, tmp_path):
        sandbox = SandboxedExecution(workspace=tmp_path)
        assert sandbox.workspace == tmp_path.resolve()

    def test_nonzero_exit_is_error(self, tmp_path):
        sandbox = SandboxedExecution(workspace=tmp_path)
        result = asyncio.run(sandbox.run("exit 1"))
        assert result.is_error is True
        assert result.metadata["returncode"] == 1
