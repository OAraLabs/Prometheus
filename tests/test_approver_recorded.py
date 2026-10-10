"""Every approval answer records WHO gave it. W4, step 1 of 3: RECORD.

``docs/design/approver-credential.md`` (ruled 2026-10-04, B1–B5 as
recommended). An approval used to be answerable with the daemon's API token
on every surface, and no surface recorded who answered: the audit table has
had a ``user_id`` column all along and the resolution row never filled it.

This step records the answerer and REFUSES NOTHING. Warn and enforce come
later, after Beacon desktop enrols as a device. So every test below that
answers with the API token, a device, or no token at all also asserts that
the answer still went through.

The surface tests import nothing new: on a tree without this change they run
and fail on their assertions (the row's ``user_id`` is empty).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
)
from prometheus.permissions.audit import AuditDecision, AuditLogger  # noqa: E402
from prometheus.permissions.checker import (  # noqa: E402
    Grant,
    PermissionMode,
    SecurityGate,
)
from prometheus.web.server import create_app  # noqa: E402

RID = "abc12345"
TOKEN = "global-test-token"


def _gate(tmp_path):
    return SecurityGate(mode=PermissionMode.DEFAULT,
                        audit_logger=AuditLogger(tmp_path / "audit"))


def _queue(tmp_path):
    return ApprovalQueue(security_gate=_gate(tmp_path))


def _pending(queue, rid: str = RID, **kw):
    action = PendingAction(
        request_id=rid,
        tool_name=kw.pop("tool_name", "write_file"),
        description=kw.pop("description", "write a file"),
        **kw,
    )
    queue.pending[rid] = action
    return action


def _answered_by(queue) -> list[str | None]:
    """``user_id`` of every resolution row, oldest first."""
    rows = queue._security_gate._audit.query_recent(limit=50)
    return [r.user_id for r in reversed(rows)
            if r.decision in (AuditDecision.CONFIRM_APPROVED,
                              AuditDecision.CONFIRM_REJECTED)]


class _Bus:
    def __init__(self):
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, signal):
        self.emitted.append((signal.kind, signal.payload))


# --------------------------------------------------------------------------- #
# The core: approve and deny cannot be called without naming who answered
# --------------------------------------------------------------------------- #


class TestTheQueueNamesWhoAnswered:
    @pytest.mark.asyncio
    async def test_approve_requires_an_approver(self, tmp_path):
        queue = _queue(tmp_path)
        action = _pending(queue)
        with pytest.raises(TypeError):
            await queue.approve(RID)
        assert action._result is not ApprovalResult.APPROVED, (
            "a call that names no approver must not resolve the request")

    @pytest.mark.asyncio
    async def test_deny_requires_an_approver(self, tmp_path):
        queue = _queue(tmp_path)
        _pending(queue)
        with pytest.raises(TypeError):
            await queue.deny(RID)

    @pytest.mark.asyncio
    async def test_the_resolution_row_names_the_approver(self, tmp_path):
        from prometheus.permissions.approver import Approver

        queue = _queue(tmp_path)
        _pending(queue)
        _pending(queue, rid="def67890")
        assert await queue.approve(RID, by=Approver("telegram", "456", "will"))
        assert await queue.deny("def67890", by=Approver("device-token", "d1", "iPhone"))
        assert _answered_by(queue) == ["telegram:456", "device-token:d1"]

    @pytest.mark.asyncio
    async def test_the_resolved_signal_names_the_approver(self, tmp_path):
        from prometheus.permissions.approver import Approver

        queue = _queue(tmp_path)
        queue.signal_bus = bus = _Bus()
        _pending(queue)
        _pending(queue, rid="def67890")
        await queue.approve(RID, scope="once", by=Approver("slack", "U1", "will"))
        await queue.deny("def67890", by=Approver("discord", "789", "will"))
        resolved = [p for k, p in bus.emitted if k == "approval_resolved"]
        assert resolved[0]["approved_by"] == {
            "kind": "slack", "id": "U1", "name": "will"}
        assert resolved[1]["denied_by"] == {
            "kind": "discord", "id": "789", "name": "will"}

    @pytest.mark.asyncio
    async def test_a_remembered_grant_names_who_granted_it(self, tmp_path):
        from prometheus.gateway.commands import cmd_approve
        from prometheus.permissions.approver import Approver

        queue = _queue(tmp_path)
        _pending(queue, grant_file_path="/tmp/target.txt")
        who = Approver("device-token", "d1", "iPhone")
        await cmd_approve(queue, f"always {RID}", by=who)
        (grant,) = queue._security_gate.list_grants()
        assert grant.granted_by == who.to_record()
        # And it survives prometheus.yaml: plain data, read back as written.
        row = grant.to_config_dict()
        assert row["granted_by"] == {"kind": "device-token", "id": "d1", "name": "iPhone"}
        assert Grant.from_config_dict(row).granted_by == row["granted_by"]

    def test_a_grant_stored_before_this_change_still_loads(self):
        old = Grant(kind="path_prefix", value="/tmp/x", tool_name="write_file")
        row = old.to_config_dict()
        assert "granted_by" not in row
        assert Grant.from_config_dict(row) is not None


# --------------------------------------------------------------------------- #
# REST: the caller's credential, as the middleware resolved it
# --------------------------------------------------------------------------- #


def _app(queue, monkeypatch, tmp_path, *, token: str | None = TOKEN):
    from prometheus.config.device_store import DeviceStore

    if token:
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", token)
    else:
        monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    store = DeviceStore(tmp_path / "devices.db")
    app = create_app({}, device_store=store)
    app.state.approval_queue = queue
    return TestClient(app), store


class TestRestRecordsTheCredential:
    def test_the_api_token_is_recorded_as_the_api_token_and_still_works(
        self, tmp_path, monkeypatch,
    ):
        queue = _queue(tmp_path)
        _pending(queue)
        client, _ = _app(queue, monkeypatch, tmp_path)
        res = client.post(f"/api/approvals/{RID}/approve", json={"scope": "once"},
                          headers={"Authorization": f"Bearer {TOKEN}"})
        assert res.status_code == 200 and res.json()["ok"] is True, (
            "the record step refuses nothing", res.text)
        assert _answered_by(queue) == ["global-token"]

    def test_a_device_token_is_recorded_as_that_device(self, tmp_path, monkeypatch):
        queue = _queue(tmp_path)
        _pending(queue)
        client, store = _app(queue, monkeypatch, tmp_path)
        device = store.mint("Will's iPhone", "ios")
        # A device answers the tool calls of ITS OWN sessions (web/route_access.py): this one raised the request.
        store.claim_session("web:phone", device["id"])
        queue.pending[RID].session_id = "web:phone"
        res = client.post(f"/api/approvals/{RID}/approve", json={"scope": "once"},
                          headers={"Authorization": f"Bearer {device['token']}"})
        assert res.json()["ok"] is True, res.text
        assert _answered_by(queue) == [f"device-token:{device['id']}"]

    def test_a_deny_with_a_device_token_is_recorded(self, tmp_path, monkeypatch):
        queue = _queue(tmp_path)
        _pending(queue)
        client, store = _app(queue, monkeypatch, tmp_path)
        device = store.mint("desk", "macos")
        store.claim_session("web:desk", device["id"])
        queue.pending[RID].session_id = "web:desk"
        res = client.post(f"/api/approvals/{RID}/deny",
                          headers={"Authorization": f"Bearer {device['token']}"})
        assert res.json()["ok"] is True, res.text
        assert _answered_by(queue) == [f"device-token:{device['id']}"]

    def test_open_mode_is_recorded_as_open_and_still_works(
        self, tmp_path, monkeypatch,
    ):
        queue = _queue(tmp_path)
        _pending(queue)
        client, _ = _app(queue, monkeypatch, tmp_path, token=None)
        res = client.post(f"/api/approvals/{RID}/approve", json={"scope": "once"})
        assert res.json()["ok"] is True, res.text
        assert _answered_by(queue) == ["open"]

    def test_approve_all_records_every_answer(self, tmp_path, monkeypatch):
        queue = _queue(tmp_path)
        _pending(queue)
        _pending(queue, rid="def67890")
        client, _ = _app(queue, monkeypatch, tmp_path)
        res = client.post("/api/approvals/all/approve", json={"scope": "once"},
                          headers={"Authorization": f"Bearer {TOKEN}"})
        assert res.status_code == 200, res.text
        assert _answered_by(queue) == ["global-token", "global-token"]


# --------------------------------------------------------------------------- #
# Chat gateways: the PERSON who sent the command, not just the chat
# --------------------------------------------------------------------------- #


class TestChatGatewaysRecordThePerson:
    @pytest.mark.asyncio
    async def test_telegram_records_the_sender(self, tmp_path):
        from tests.test_gateway_command_pins import (
            _make_adapter,
            _make_context,
            _make_update,
        )

        adapter = _make_adapter()
        adapter._approval_queue = queue = _queue(tmp_path)
        _pending(queue)
        _pending(queue, rid="def67890")
        await adapter._cmd_approve(_make_update(text=f"/approve {RID}"), _make_context())
        await adapter._cmd_deny(_make_update(text="/deny def67890"), _make_context())
        assert _answered_by(queue) == ["telegram:456", "telegram:456"]

    @pytest.mark.asyncio
    async def test_slack_records_the_sender(self, tmp_path):
        from tests.test_gateway_g1 import _make_slack_adapter

        async def _ack(*a, **k):
            return None

        async def _respond(*a, **k):
            return None

        adapter = _make_slack_adapter()
        adapter._approval_queue = queue = _queue(tmp_path)
        _pending(queue)
        _pending(queue, rid="def67890")
        cmd = {"channel_id": "C1", "user_id": "U42", "user_name": "will"}
        await adapter._slash_approve(_ack, {**cmd, "text": RID}, _respond)
        await adapter._slash_deny(_ack, {**cmd, "text": "def67890"}, _respond)
        assert _answered_by(queue) == ["slack:U42", "slack:U42"]

    @pytest.mark.asyncio
    async def test_discord_records_the_sender(self, tmp_path):
        from tests.test_discord import _FakeInteraction, _make_adapter

        adapter = _make_adapter()
        adapter._approval_queue = queue = _queue(tmp_path)
        _pending(queue)
        _pending(queue, rid="def67890")
        interaction = _FakeInteraction()
        interaction.user = SimpleNamespace(id=789, name="will")
        await adapter._app_approve(interaction, RID)
        await adapter._app_deny(interaction, "def67890")
        assert _answered_by(queue) == ["discord:789", "discord:789"]

    @pytest.mark.asyncio
    async def test_a_sender_the_gateway_cannot_name_is_recorded_as_unknown(
        self, tmp_path,
    ):
        """Recorded honestly as unknown, and still answered."""
        from tests.test_discord import _FakeInteraction, _make_adapter

        adapter = _make_adapter()
        adapter._approval_queue = queue = _queue(tmp_path)
        action = _pending(queue)
        await adapter._app_approve(_FakeInteraction(), RID)
        assert action._result is ApprovalResult.APPROVED
        assert _answered_by(queue) == ["discord:unknown"]


def test_no_surface_calls_the_queue_without_naming_the_approver():
    """``by`` is required, so a new surface cannot forget it — but a fake
    queue in a test would hide that. This greps src/ for the call shapes."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "prometheus"
    offenders = []
    for py in src.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for m in re.finditer(
            r"(?:queue\.(?:approve|deny)|cmd_approve|cmd_deny|approve_detail)\(", text,
        ):
            # the call's argument list, up to its matching paren
            depth, i = 1, m.end()
            while depth and i < len(text):
                depth += {"(": 1, ")": -1}.get(text[i], 0)
                i += 1
            args = text[m.end():i]
            if "def " in text[max(0, m.start() - 10):m.start()]:
                continue
            if "by=" not in args:
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{py.relative_to(src)}:{line}")
    assert not offenders, offenders

