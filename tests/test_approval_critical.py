"""Which approval requests must be answered in the app (``critical``), and the push that says so.

The daemon decides; the phone renders. ``critical`` is computed once, in ``ApprovalQueue.serialize_pending``, so
``GET /api/approvals``, the ``approval_pending`` frame and the push cannot disagree. A critical request is pushed
with no ``category`` (no Approve on the notification) and ``critical: true``; every other approval push keeps
``APPROVAL``. ``server_capabilities`` lists ``approval_critical`` so a client knows a missing field means "this
daemon does not say", not "not critical".
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction
from prometheus.permissions.argument_view import arguments_were_cut
from prometheus.permissions.critical import (
    DEFAULT_CRITICAL_COMMANDS,
    DEFAULT_CRITICAL_TOOLS,
    CriticalRules,
    critical_reasons,
    is_critical,
)
from prometheus.push.apns import SendResult
from prometheus.push.dispatcher import PushDispatcher


class _Gate:
    pass


class _Bus:
    def __init__(self) -> None:
        self.signals: list = []

    async def emit(self, signal) -> None:
        self.signals.append(signal)


def _queue(rules: CriticalRules | None = None) -> tuple[ApprovalQueue, _Bus]:
    q = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    bus = _Bus()
    q.signal_bus = bus
    if rules is not None:
        q.critical_rules = rules
    return q, bus


async def _ask(q: ApprovalQueue, tool: str, arguments: dict | None, description: str = "run it", **kw) -> dict:
    before = len(q.signal_bus.signals)
    waiter = asyncio.create_task(q.request_approval(tool, description, arguments=arguments, **kw))
    for _ in range(50):
        await asyncio.sleep(0)
        if len(q.signal_bus.signals) > before:
            break
    payload = q.signal_bus.signals[-1].payload
    waiter.cancel()
    return payload


# ── the rules ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", [
    "rm -rf build", "sudo rm x", "FOO=1 rm x", "/bin/rm x", "ls && rm x", "cat a | xargs echo; rm b",
    "git push", "git push --force origin main", "git -C repo push", "curl -X POST https://example.invalid",
    "echo $(curl https://example.invalid)", "oara pair --owner", "chmod 777 f", "gh pr create",
])
def test_default_commands_that_delete_send_spend_or_change_config_are_critical(command):
    assert CriticalRules().matches("bash", {"command": command}), command


@pytest.mark.parametrize("command", ["ls -la", "git status", "git log --oneline", "cat README.md", "pytest -q",
                                     "echo rm", "grep -r curl src"])
def test_ordinary_commands_are_not(command):
    assert CriticalRules().matches("bash", {"command": command}) == (), command


def test_default_tools_and_config_paths_are_critical(tmp_path):
    rules = CriticalRules()
    for tool in DEFAULT_CRITICAL_TOOLS:
        assert rules.matches(tool, {}), tool
    assert rules.matches("write_file", {"path": "~/.prometheus/prometheus.yaml"})
    assert rules.matches("edit_file", {}, grant_file_path="~/.config/prometheus/x.yaml")
    assert rules.matches("write_file", {"path": str(tmp_path / "notes.md")}) == ()
    assert rules.matches("read_file", {"path": "~/.prometheus/prometheus.yaml"}) == ()


def test_the_owner_lists_replace_the_defaults_and_an_empty_list_means_none():
    rules = CriticalRules.from_config({"push": {"critical_tools": ["web_fetch"], "critical_commands": [],
                                                "critical_paths": []}})
    assert rules.matches("web_fetch", {}) and not rules.matches("message", {})
    assert rules.matches("bash", {"command": "rm -rf /tmp/x"}) == ()
    assert CriticalRules.from_config({}).commands == DEFAULT_CRITICAL_COMMANDS


def test_a_wrong_shape_falls_back_to_the_default_list():
    rules = CriticalRules.from_config({"push": {"critical_commands": "rm"}})
    assert rules.commands == DEFAULT_CRITICAL_COMMANDS


def test_arguments_cut_or_redacted_are_reported():
    assert arguments_were_cut({"command": "x" * 500})
    fake = "sk-" + "q7" * 16                                 # built here so no key-shaped literal is committed
    assert arguments_were_cut({"api" + "_key": fake})
    assert arguments_were_cut({f"k{i}": "v" * 150 for i in range(10)})
    assert not arguments_were_cut({"command": "ls -la"})
    assert not arguments_were_cut(None)


def test_built_in_reasons():
    assert "desktop" in critical_reasons(PendingAction(request_id="a", tool_name="computer", description="d",
                                                       task_id="t1"))
    assert "hold_lift" in critical_reasons(PendingAction(request_id="b", tool_name="hold", description="d",
                                                         hold_device_id="dev1"))
    assert "long_description" in critical_reasons(PendingAction(request_id="c", tool_name="t",
                                                                description="x" * 601))
    assert "long_description" in critical_reasons(PendingAction(request_id="d", tool_name="t",
                                                                description="\n".join("l" * 9)))
    assert not is_critical(PendingAction(request_id="e", tool_name="bash", description="run ls"))


# ── one value on every surface ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_short_bash_ls_is_not_critical_and_truncated_arguments_are():
    q, _ = _queue()
    plain = await _ask(q, "bash", {"command": "ls"})
    assert plain["critical"] is False
    long = await _ask(q, "bash", {"command": "echo " + "y" * 400})
    assert long["critical"] is True


@pytest.mark.asyncio
async def test_the_list_and_the_frame_carry_the_same_value():
    q, _ = _queue()
    frame = await _ask(q, "bash", {"command": "rm -rf build"})
    (listed,) = [q.serialize_pending(a) for a in q.list_pending() if a.request_id == frame["request_id"]]
    assert frame["critical"] is True and listed["critical"] is True


def test_a_hold_lift_serializes_as_critical():
    q, _ = _queue()
    action = PendingAction(request_id="h1", tool_name="lift hold", description="lift the hold", hold_device_id="d1")
    q._register(action)
    assert q.serialize_pending(action)["critical"] is True


# ── the push ─────────────────────────────────────────────────────────────────

class _Store:
    def push_targets(self):
        return [types.SimpleNamespace(id="d1", apns_token="tok", environment="sandbox", bundle_id="",
                                      push_failures=0)]

    def approval_push_targets(self, *, desktop):
        return self.push_targets()


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, **kwargs):
        self.sent.append(kwargs["payload"])
        return SendResult(outcome="ok", status=200)


def _push(payload: dict) -> dict:
    sender = _Sender()
    asyncio.run(PushDispatcher(_Store(), sender, bridge=None).on_signal(
        types.SimpleNamespace(kind="approval_pending", payload=payload)))
    (body,) = sender.sent
    return body


def test_a_critical_push_has_no_category_and_says_critical():
    body = _push({"request_id": "r1", "tool_name": "bash", "description": "rm -rf build", "expires_at": 1.0,
                  "critical": True})
    assert "category" not in body["aps"] and body["critical"] is True


def test_every_other_approval_push_keeps_its_category():
    body = _push({"request_id": "r2", "tool_name": "bash", "description": "ls", "expires_at": 1.0,
                  "critical": False})
    assert body["aps"]["category"] == "APPROVAL" and "critical" not in body


def test_a_payload_without_the_field_is_pushed_as_before():
    body = _push({"request_id": "r3", "tool_name": "bash", "description": "ls", "expires_at": 1.0})
    assert body["aps"]["category"] == "APPROVAL"


def test_the_critical_push_stays_under_the_apns_limit():
    body = _push({"request_id": "r4", "tool_name": "t" * 300, "description": "d" * 5000, "expires_at": 1.0,
                  "critical": True})
    assert len(json.dumps(body).encode()) < 4096


# ── the capability ───────────────────────────────────────────────────────────

def test_server_capabilities_lists_approval_critical():
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge.config = {}
    assert "approval_critical" in bridge._server_capabilities()
