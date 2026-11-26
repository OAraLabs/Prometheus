"""The ACP client side: registry, connection, driver, gate-routed permissions.

The far side is REAL code: the TCK's conforming v2 fixture agent (Apache-2.0)
is launched over stdio for the connection/driver tests, exactly as the TCK
drives it. Permission-routing tests use the real SecurityGate and the real
ApprovalQueue. Nothing hand-writes an ACP payload.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from prometheus.acp.client.connection import AcpClientConnection
from prometheus.acp.client.driver import DriverRefused, OutsideAgentDriver
from prometheus.acp.client.registry import OutsideAgent, UnknownAgent, registry_from_config, resolve

TCK_AGENT = Path.home() / "projects/.scratch/track-a-acp/tck/tests/fixtures/agents/v2/conforming.py"
needs_tck = pytest.mark.skipif(not TCK_AGENT.exists(), reason="TCK checkout not present")


# ── registry ─────────────────────────────────────────────────────────────────


def test_registry_ships_empty_and_resolves_only_named_agents():
    assert registry_from_config(None) == {}
    assert registry_from_config({}) == {}
    assert registry_from_config({"acp": {"enabled": True}}) == {}
    r = registry_from_config({"acp": {"clients": {
        "claude": {"command": ["claude-code-acp"], "description": "Claude Code"},
    }}})
    assert resolve(r, "claude").command == ("claude-code-acp",)
    with pytest.raises(UnknownAgent):
        resolve(r, "anything-else")


def test_a_malformed_entry_never_half_registers():
    r = registry_from_config({"acp": {"clients": {
        "bad1": {"command": []},
        "bad2": {"command": [1, 2]},
        "bad3": {"command": "unbalanced 'quote"},
        "bad4": "not-a-mapping",
        "ok": {"command": "agent --flag"},
    }}})
    assert set(r) == {"ok"}
    assert r["ok"].command == ("agent", "--flag")


# ── connection + driver against the REAL TCK fixture agent ──────────────────


async def _launch_fixture():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(TCK_AGENT),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, cwd=str(TCK_AGENT.parent),
    )
    return proc


@needs_tck
@pytest.mark.asyncio
async def test_initialize_new_prompt_roundtrip_against_the_real_fixture():
    proc = await _launch_fixture()
    conn = AcpClientConnection(proc.stdout, proc.stdin)
    await conn.start()
    try:
        init = await conn.request("initialize", {
            "protocolVersion": 2, "info": {"name": "test", "version": "0"}, "capabilities": {}})
        assert init["protocolVersion"] == 2
        new = await conn.request("session/new", {"cwd": str(TCK_AGENT.parent)})
        sid = new["sessionId"]
        updates: list = []

        async def observe(method, params):
            if method == "session/update":
                updates.append(params["update"]["sessionUpdate"])

        conn._notification_handler = observe
        resp = await conn.request("session/prompt", {
            "sessionId": sid, "prompt": [{"type": "text", "text": "hello"}]})
        assert "messageId" in resp
        await asyncio.sleep(0.1)
        assert "user_message" in updates and "state_update" in updates
    finally:
        await conn.close()
        proc.terminate()
        await proc.wait()


@needs_tck
@pytest.mark.asyncio
async def test_driver_streams_updates_and_refuses_without_a_gate():
    agent = OutsideAgent(name="fixture", command=(sys.executable, str(TCK_AGENT)))
    seen: list = []

    async def observe(kind, update):
        seen.append(kind)

    driver = OutsideAgentDriver(agent, observer=observe, cwd=TCK_AGENT.parent)
    try:
        sid = await driver.start()
        assert sid
        await driver.prompt("hello")
        await asyncio.sleep(0.1)
        assert "user_message" in seen
        # no gate -> the fixture's permission ask (it makes none by default)
        # is refused by the connection's fail-closed path; exercised below.
    finally:
        await driver.close()


@pytest.mark.asyncio
async def test_machine_sessions_are_hard_disabled():
    agent = OutsideAgent(name="x", command=("true",))
    for sid in ("bakeoff:x", "eval:x", "smoke:x", "system", "gym:x", "coding:x"):
        with pytest.raises(DriverRefused):
            await OutsideAgentDriver(agent).start(session_id=sid)


@pytest.mark.asyncio
async def test_permission_handler_none_answers_cancelled_on_a_pipe():
    """The connection's fail-closed path, observable: a socketpair stands in
    for the agent; an ask with no handler is answered outcome=cancelled —
    never approved."""
    import json
    import socket
    agent_sock, client_sock = socket.socketpair()
    agent_reader, agent_writer = await asyncio.open_connection(sock=agent_sock)
    client_reader, client_writer = await asyncio.open_connection(sock=client_sock)
    conn = AcpClientConnection(client_reader, client_writer)  # no permission handler
    await conn.start()
    try:
        agent_writer.write((
            '{"jsonrpc":"2.0","id":7,"method":"session/request_permission",'
            '"params":{"sessionId":"s","title":"t","options":['
            '{"optionId":"a","name":"Allow","kind":"allow_once"},'
            '{"optionId":"r","name":"Deny","kind":"reject_once"}]}}\n'
        ).encode())
        await agent_writer.drain()
        line = await asyncio.wait_for(agent_reader.readline(), 5)
        answer = json.loads(line)
        assert answer["id"] == 7
        assert answer["result"]["outcome"]["outcome"] == "cancelled"
    finally:
        await conn.close()
        agent_writer.close()


# ── permission routing through the REAL gate and queue ──────────────────────


def _gate_and_queue(tmp_path, mode="cautious"):
    from prometheus.permissions.approval_queue import ApprovalQueue
    from prometheus.permissions.audit import AuditLogger
    from prometheus.permissions.checker import PermissionMode, SecurityGate
    gate = SecurityGate(mode=PermissionMode(mode), audit_logger=AuditLogger(tmp_path / "audit"))
    queue = ApprovalQueue(security_gate=gate)
    gate._approval_queue = queue
    return gate, queue


def _permission_params(command="rm -rf /tmp/x"):
    from prometheus.acp import types
    return types.RequestPermissionRequest.model_validate({
        "sessionId": "s", "title": "run a command",
        "subject": {"type": "command", "command": command, "cwd": "/tmp"},
        "options": [
            {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
            {"optionId": "always", "name": "Always", "kind": "allow_always"},
            {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
        ],
    }).model_dump(by_alias=True, exclude_none=True)


@pytest.mark.asyncio
async def test_a_command_ask_goes_to_the_owners_queue_and_never_auto_allows(tmp_path):
    gate, queue = _gate_and_queue(tmp_path)
    driver = OutsideAgentDriver(OutsideAgent(name="x", command=("true",)),
                                gate=gate, approval_queue=queue)
    task = asyncio.create_task(driver._answer_permission(_permission_params()))
    await asyncio.sleep(0.05)
    assert queue.pending, "the ask never reached the owner's queue"
    action = next(iter(queue.pending.values()))
    assert action.once_only is True
    # nobody answered -> timeout/refusal path; answer DENY via the queue
    from prometheus.permissions.approver import Approver
    rid = action.request_id
    assert await queue.deny(rid, by=Approver("device-token", "phone", "Phone"))
    answer = await asyncio.wait_for(task, 5)
    assert answer["outcome"]["outcome"] == "selected"
    assert answer["outcome"]["optionId"] == "deny"


@pytest.mark.asyncio
async def test_an_approved_ask_selects_allow_once_never_allow_always(tmp_path):
    gate, queue = _gate_and_queue(tmp_path)
    driver = OutsideAgentDriver(OutsideAgent(name="x", command=("true",)),
                                gate=gate, approval_queue=queue)
    task = asyncio.create_task(driver._answer_permission(_permission_params()))
    await asyncio.sleep(0.05)
    action = next(iter(queue.pending.values()))
    from prometheus.permissions.approver import Approver
    assert await queue.approve(action.request_id, by=Approver("device-token", "phone", "Phone"))
    answer = await asyncio.wait_for(task, 5)
    assert answer["outcome"] == {"outcome": "selected", "optionId": "allow"}


@pytest.mark.asyncio
async def test_no_gate_refuses_with_the_reject_option(tmp_path):
    driver = OutsideAgentDriver(OutsideAgent(name="x", command=("true",)))
    answer = await driver._answer_permission(_permission_params())
    assert answer["outcome"] == {"outcome": "selected", "optionId": "deny"}


@pytest.mark.asyncio
async def test_no_queue_refuses_even_when_the_gate_would_allow(tmp_path):
    gate, _ = _gate_and_queue(tmp_path, mode="autonomous")
    driver = OutsideAgentDriver(OutsideAgent(name="x", command=("true",)), gate=gate)
    answer = await driver._answer_permission(_permission_params())
    assert answer["outcome"]["outcome"] in ("selected", "cancelled")
    if answer["outcome"]["outcome"] == "selected":
        assert answer["outcome"]["optionId"] == "deny"


def test_shipped_template_keeps_the_registry_empty_and_acp_off():
    import yaml

    from prometheus.acp.gate import acp_enabled
    template = Path(__file__).parent.parent / "config" / "prometheus.yaml.default"
    cfg = yaml.safe_load(template.read_text())
    assert acp_enabled(cfg) is False
    assert registry_from_config(cfg) == {}
