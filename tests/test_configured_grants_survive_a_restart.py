"""A grant in ``security.grants`` is in force after the daemon restarts.

``security.grants`` is where a remembered approval lives (``/approve always``, ``persist_grant``) and where an
operator writes one by hand. The daemon built its gate with ``create_security_gate``, which never passed ``grants=``;
only ``SecurityGate.from_config`` read them, and the daemon never calls it. So every persisted grant was written to
the file and then ignored at the next boot: the prompt it was meant to silence came back after each restart, and a
hand-written ``kind: tool`` row did nothing at all.

Every existing "survives a reload" test went through ``SecurityGate.from_config``, so none of them saw it. These go
through the daemon's own factory, by outcome: approve, restart, and ask the gate the same question again.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
import yaml

from prometheus.__main__ import create_security_gate
from prometheus.gateway import commands as cmds
from prometheus.permissions.approval_queue import ApprovalQueue
from prometheus.permissions.approver import in_process
from prometheus.permissions.checker import SecurityGate

BY = in_process("test")


@pytest.fixture
def config(tmp_path):
    """A real prometheus.yaml on disk, as the daemon reads it."""
    p = tmp_path / "prometheus.yaml"
    p.write_text(yaml.dump({"security": {"permission_mode": "default"}}))
    return p


def _restart(config):
    """The gate the daemon builds at boot: its file, loaded, through ``create_security_gate``."""
    security = (yaml.safe_load(config.read_text()) or {}).get("security") or {}
    return create_security_gate(security, str(config))


def _write(gate, target, tmp_path):
    """The gate's answer for a write to *target*, with a workspace it lies outside (as the agent loop asks it)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return gate.evaluate("write_file", file_path=str(target), workspace_roots=[workspace], path_is_write=True)


def _write_grants(config, grants):
    data = yaml.safe_load(config.read_text()) or {}
    data.setdefault("security", {})["grants"] = grants
    config.write_text(yaml.dump(data))


async def _approve_always(config, target):
    """The owner answers a write prompt with ``/approve always``; the grant is persisted to the file."""
    queue = ApprovalQueue(security_gate=SecurityGate(), timeout_seconds=5)
    queue._security_gate = SecurityGate(approval_queue=queue, config_path=str(config))
    task = asyncio.create_task(
        queue.request_approval("write_file", "outside", grant_file_path=str(target)))
    for _ in range(100):
        if queue.pending:
            break
        await asyncio.sleep(0.01)
    assert queue.pending, "request_approval did not queue"
    queue._test_task = task
    rid = next(iter(queue.pending))
    await cmds.cmd_approve(queue, f"always {rid}", by=BY)
    return queue._security_gate


# ── an "always" approval ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_always_approval_still_silences_the_prompt_after_a_restart(config, tmp_path):
    target = tmp_path / "outside" / "notes.txt"
    target.parent.mkdir()

    before = _write(_restart(config), target, tmp_path)
    assert before.requires_confirmation, "control: with no grant the write outside the workspace asks"

    await _approve_always(config, target)
    assert (yaml.safe_load(config.read_text())["security"]["grants"]), "control: the grant reached the file"

    after = _write(_restart(config), target, tmp_path)
    assert after.allowed and not after.requires_confirmation, (
        "the grant is in the file but the restarted gate did not load it: the prompt came back")


@pytest.mark.asyncio
async def test_the_grant_covers_only_what_was_approved_after_a_restart(config, tmp_path):
    target = tmp_path / "outside" / "notes.txt"
    target.parent.mkdir()
    await _approve_always(config, target)

    other = tmp_path / "elsewhere" / "other.txt"
    other.parent.mkdir()
    assert _write(_restart(config), other, tmp_path).requires_confirmation


@pytest.mark.asyncio
async def test_a_revoked_grant_stays_gone_after_a_restart(config, tmp_path):
    target = tmp_path / "outside" / "notes.txt"
    target.parent.mkdir()
    gate = await _approve_always(config, target)
    gate.remove_grant(gate.list_grants()[0].grant_id, str(config))

    assert _write(_restart(config), target, tmp_path).requires_confirmation


# ── a grant written by hand ──────────────────────────────────────────────────

def test_a_hand_written_tool_grant_silences_a_third_party_mcp_prompt_at_boot(config):
    tool = "mcp__capline__search_candidates"
    assert _restart(config).evaluate(tool).requires_confirmation, "control: a third-party MCP tool asks"

    _write_grants(config, [{"kind": "tool", "tool": tool, "value": ""}])
    gate = _restart(config)
    assert gate.evaluate(tool).allowed
    assert gate.evaluate("mcp__capline__another_tool").requires_confirmation, "the grant names exactly one tool"


def test_a_malformed_row_is_skipped_and_the_rest_still_load(config):
    tool = "mcp__capline__search_candidates"
    _write_grants(config, ["not a mapping", {"kind": "nonsense"}, {"kind": "tool", "tool": tool, "value": ""}])
    assert _restart(config).evaluate(tool).allowed


# ── one reading of the file, not two ─────────────────────────────────────────

def test_the_daemons_gate_and_from_config_load_the_same_grants(config):
    _write_grants(config, [
        {"kind": "tool", "tool": "mcp__capline__a", "value": ""},
        {"kind": "path_prefix", "tool": "write_file", "value": "/srv/shared"},
    ])
    def covers(gate):  # a row with no id gets a fresh one per load, so compare what each grant covers
        return {(g.kind, g.tool_name, g.value) for g in gate.list_grants()}

    daemon = covers(_restart(config))
    assert daemon == covers(SecurityGate.from_config(str(config))) and len(daemon) == 2


# ── the boot says what came back ─────────────────────────────────────────────

def test_boot_names_each_grant_it_loads(config, caplog):
    _write_grants(config, [{"kind": "tool", "tool": "mcp__capline__a", "value": "", "id": "abc123def456"}])
    with caplog.at_level(logging.INFO):
        _restart(config)
    lines = [r.getMessage() for r in caplog.records if "grant" in r.getMessage().lower()]
    assert lines, "a grant that changes what the gate allows was loaded without a word"
    assert "abc123def456" in lines[0] and "mcp__capline__a" in lines[0]


def test_boot_is_quiet_when_there_are_no_grants(config, caplog):
    with caplog.at_level(logging.INFO):
        _restart(config)
    assert not [r for r in caplog.records if "approval grant" in r.getMessage().lower()]
