"""POST /api/approvals/{id}/approve must report what it DID, not what it said.

THE DEFECT THIS PINS
--------------------
The endpoint derived its only status flag from the prose::

    ok = not text.startswith("No pending request")

That is wrong in three directions at once, and all three were live:

  * an ``always`` whose extent cannot be described approves ONCE and
    remembers nothing (rule 4) — and returned ``ok: true`` with no other
    signal, so even a correctly wired daemon could not tell a client
    "remembered" from "approved once";
  * an ``always`` on a queue with no gate attached returned ``ok: true``
    too, its failure living only in the prose;
  * ``"No pending approval requests."`` does not start with `"No pending
    request"`, so approving against an EMPTY queue returned ``ok: true``
    as well.

WHY BETTER PROSE WAS NOT AN OPTION. Neither Beacon client reads ``message``
on a 200. beacon-desktop's ``resolveApproval`` is ``Promise<void>`` and
parses the body only on the HTTP-error path; beacon-ios decodes ``{ok}``
alone, discards it, and labels the row from the scope the operator ASKED for
("approved — always", unconditionally). So a Beacon user tapping "always"
had no signal whatsoever that nothing was stored.

Every assertion here is on the RESPONSE BODY — the thing a client actually
receives — and ``remembered`` is cross-checked against the gate's real grant
list, never against the message text.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue, PendingAction,
)
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from prometheus.permissions.computer_extent import ComputerExtent  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

RID = "abc12345"


def _gate():
    return SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)


def _wire(gate):
    """A queue that can record grants -- now simply a queue."""
    return ApprovalQueue(security_gate=gate)


def _pending(queue, **kw):
    action = PendingAction(
        request_id=RID,
        tool_name=kw.pop("tool_name", "write_file"),
        description=kw.pop("description", "write a file"),
        **kw,
    )
    queue.pending[action.request_id] = action
    return action


def _client(queue):
    app = create_app({})
    app.state.approval_queue = queue
    return TestClient(app)


def _approve(queue, scope="always", rid=RID):
    return _client(queue).post(
        f"/api/approvals/{rid}/approve", json={"scope": scope}
    ).json()


# ── REMEMBERED: add_grant ran, and the client is told so ────────────────────

class TestRemembered:
    def test_an_always_scope_reports_remembered_with_a_grant_id(self):
        gate = _gate()
        queue = _wire(gate)
        _pending(queue, grant_file_path="/tmp/target.txt")
        body = _approve(queue)
        assert body["ok"] is True
        assert body["remembered"] is True, (
            "add_grant ran and the client was not told; this is the fact the "
            "prose could not carry"
        )
        assert body["grant_id"], "a remembered grant must name its revoke handle"

    def test_the_grant_id_is_the_gate_s_real_grant(self):
        gate = _gate()
        queue = _wire(gate)
        _pending(queue, grant_file_path="/tmp/target.txt")
        body = _approve(queue)
        assert [g.grant_id for g in gate.list_grants()] == [body["grant_id"]], (
            "grant_id must identify the grant that actually exists, not a "
            "value parsed back out of the message"
        )

    def test_remembered_is_about_add_grant_not_about_persistence(self):
        # No config_path, so persist_grant CANNOT write. The grant is still
        # recorded on the gate, so remembered stays true and the message says
        # persistence failed. Conflating the two would report nothing stored.
        gate = _gate()
        queue = _wire(gate)
        _pending(queue, grant_file_path="/tmp/target.txt")
        body = _approve(queue)
        assert body["remembered"] is True
        assert gate.list_grants(), "the grant is on the gate regardless of disk"

    def test_a_rememberable_desktop_action_is_remembered(self):
        gate = _gate()
        queue = _wire(gate)
        _pending(
            queue, tool_name="computer_click",
            grant_computer_action=ComputerExtent(
                target="mini", app="firefox", verb="click", delivery="background"),
        )
        body = _approve(queue)
        assert body["remembered"] is True
        assert body["grant_id"]


# ── NOT REMEMBERED, BUT STILL A SUCCESS ─────────────────────────────────────

class TestApprovedButNothingStored:
    def test_rule_4_reports_not_remembered_and_still_ok(self):
        # No file, no command, no desktop extent: derive_grant returns None,
        # so this approves ONCE and remembers nothing. It SUCCEEDED, so ok
        # stays true — which is precisely why the two facts cannot share one
        # flag.
        gate = _gate()
        queue = _wire(gate)
        _pending(queue, tool_name="some_tool")
        body = _approve(queue)
        assert body["ok"] is True, "the approval succeeded"
        assert body["remembered"] is False, (
            "an always-scope that stored nothing reported success "
            "indistinguishable from one that stored a grant"
        )
        assert body["grant_id"] is None
        assert not gate.list_grants()

    def test_a_payload_bearing_desktop_action_is_not_remembered(self):
        # Unrememberable BY CONSTRUCTION: the extent has no term for the text
        # being typed, so 'type in firefox' must never become a lasting grant.
        gate = _gate()
        queue = _wire(gate)
        _pending(
            queue, tool_name="computer_type_text",
            grant_computer_action=ComputerExtent(
                target="mini", app="firefox", verb="type_text",
                delivery="background", payload_params=("text",)),
        )
        body = _approve(queue)
        assert body["ok"] is True
        assert body["remembered"] is False
        assert not gate.list_grants()

    def test_an_unwired_queue_cannot_be_BUILT_any_more(self):
        """This case used to be reachable, and was the whole defect: a queue
        with no gate answered an always-scope with ok:true, stored nothing,
        and wrote no audit row. The gate is a required constructor argument
        now, so the state that produced it has no constructor."""
        with pytest.raises(TypeError):
            ApprovalQueue()
        with pytest.raises(ValueError, match="requires a SecurityGate"):
            ApprovalQueue(security_gate=None)

    def test_scope_once_is_never_remembered(self):
        gate = _gate()
        queue = _wire(gate)
        _pending(queue, grant_file_path="/tmp/target.txt")
        body = _approve(queue, scope="once")
        assert body["ok"] is True
        assert body["remembered"] is False
        assert not gate.list_grants()


# ── NOT OK: nothing was resolved ────────────────────────────────────────────

class TestNothingResolved:
    def test_an_empty_queue_is_not_ok(self):
        gate = _gate()
        queue = _wire(gate)            # no pending action at all
        body = _approve(queue)
        assert body["ok"] is False, (
            "'No pending approval requests.' does not match the old "
            "'No pending request' prefix, so an empty queue reported success"
        )
        assert body["remembered"] is False

    def test_an_unknown_request_id_is_not_ok(self):
        gate = _gate()
        queue = _wire(gate)
        _pending(queue, grant_file_path="/tmp/target.txt")
        body = _approve(queue, rid="deadbeef")
        assert body["ok"] is False
        assert body["remembered"] is False


# ── STRUCTURAL: the flags must not come back from the prose ─────────────────

def test_the_endpoint_does_not_infer_its_flags_from_the_message():
    """Checked on the AST, so a comment may still QUOTE the old expression.

    A future edit that reintroduces a string test on the message would pass
    every behavioural case above that happens to be phrased the expected way
    — which is exactly how the original survived. Scanning source as text
    would work too, but it would also forbid the comment in ``approve_action``
    that records what the defect WAS, and that comment is worth more than the
    simpler check.
    """
    import ast
    import inspect

    from prometheus.web import server

    tree = ast.parse(inspect.getsource(server))
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.AsyncFunctionDef) and n.name == "approve_action"),
        None,
    )
    assert fn is not None, "approve_action not found — has it been renamed?"

    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr in {
            "startswith", "endswith", "find", "index",
        }:
            raise AssertionError(
                f"approve_action calls .{node.attr}() — the response flags "
                f"must come from ApproveOutcome, which reports what RAN, "
                f"not from the shape of the operator-facing text"
            )
        if isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.In, ast.NotIn)) for op in node.ops
        ):
            raise AssertionError(
                "approve_action uses a substring/membership test; see above"
            )
