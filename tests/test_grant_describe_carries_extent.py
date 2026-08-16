"""Grant.describe() states the extent it CARRIES, never one it re-derives.

WHY THESE ASSERT WHAT THEY ASSERT
---------------------------------
``describe()`` used to recover the extent by calling ``Path(self.value).is_dir()``.
``derive_grant(widen=True)`` sets ``value`` to ``target.parent`` — a directory by
construction — so the stat agreed with reality only when that directory already
existed on disk. Approve ``always here`` for a file in a directory the write was
about to create, and a ``path_prefix`` covering the whole subtree described itself
as "on exactly <dir>": narrow wording, wide grant, which is consent obtained under
a description narrower than the thing consented to.

The existing ``test_extent_does_not_disclose_whether_the_target_exists`` did not
catch it and could not have. It renders the ``always`` verb — the NARROW one,
where ``value`` is the target file and the stat returns False for both an existing
file and a ghost, so the wording matched either way. Its widening twin would also
have passed, because ``tmp_path`` (the parent) exists in both of its branches.
Exposing this needs the PARENT missing, which is what ``_ghost_parent`` builds.

The duration half is the same defect in the same constructor: ``derive_grant``
left ``scope`` at the dataclass default and ``cmd_approve`` patched it on the line
AFTER ``queue.approve()`` had already written the resolution audit row from that
object. So an ``always`` grant was recorded as "until the daemon restarts" — in
the store built to make grants accountable — while the operator-facing string,
composed after the patch, read correctly. The two transient surfaces were right
and the one durable record was wrong, which is why the audit assertions below
read the ROW and not the return value.

Real objects throughout: a real SecurityGate, a real ApprovalQueue, a real
AuditLogger over a real SQLite file. Nothing is doubled.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import pytest
import yaml

from prometheus.gateway import commands as cmds
from prometheus.permissions import approval_queue as aq
from prometheus.permissions.approval_queue import (
    ApprovalQueue, PendingAction, approve_verbs, derive_grant,
    prospective_extents)
from prometheus.permissions.audit import AuditDecision, AuditLogger
from prometheus.permissions.checker import Grant, SecurityGate


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "prometheus.yaml"
    p.write_text(yaml.dump({"security": {"denied_paths": ["/etc"]}}))
    return p


def _queue(config, tmp_path) -> ApprovalQueue:
    """Queue + gate + a REAL audit logger, so resolution rows can be read back."""
    q = ApprovalQueue(timeout_seconds=5)
    gate = SecurityGate(approval_queue=q, config_path=str(config))
    gate._audit = AuditLogger(tmp_path / "audit")
    q._security_gate = gate
    return q


async def _request(queue: ApprovalQueue, **kwargs) -> str:
    task = asyncio.create_task(queue.request_approval("write_file", "outside", **kwargs))
    for _ in range(100):
        if queue.pending:
            break
        await asyncio.sleep(0.01)
    assert queue.pending, "request_approval did not queue"
    queue._test_task = task
    return list(queue.pending.keys())[0]


def _action(path) -> PendingAction:
    return PendingAction(request_id="r1", tool_name="write_file",
                         description="d", grant_file_path=str(path))


def _ghost_parent(tmp_path) -> Path:
    """A target whose PARENT does not exist — the condition that exposed this.

    Not merely a missing file: the parent directory is what widening grants,
    and its absence is what the stat answered False to.
    """
    p = tmp_path / "not-created-yet" / "x.txt"
    assert not p.parent.exists(), "fixture is wrong: the parent must be absent"
    return p


def _audit_reason(queue: ApprovalQueue) -> str:
    rows = queue._security_gate._audit.query_recent(
        limit=1, decision=AuditDecision.CONFIRM_APPROVED)
    assert rows, "no CONFIRM_APPROVED row was written — nothing to read back"
    return rows[0].reason


# ── The extent, both directions ────────────────────────────────────────────

def test_widening_a_missing_parent_still_says_anything_under(tmp_path):
    """THE REGRESSION. Restoring the is_dir() stat turns this red."""
    g = derive_grant(_action(_ghost_parent(tmp_path)), verb="always here")
    assert g is not None
    assert "on anything under" in g.describe(), (
        f"a subtree grant over a not-yet-created directory described itself "
        f"narrowly — the operator would consent to one path and receive the "
        f"whole directory. got: {g.describe()!r}"
    )
    assert "on exactly" not in g.describe()


def test_widening_renders_identically_whether_the_parent_exists(tmp_path):
    """The wording is a property of the GRANT, not of the disk."""
    real = tmp_path / "real-dir" / "x.txt"
    real.parent.mkdir()
    ghost = _ghost_parent(tmp_path)

    def render(p):
        return derive_grant(_action(p), verb="always here").describe().replace(
            str(p.parent), "<DIR>")

    assert render(real) == render(ghost), (
        "the description changed with filesystem state, which means something "
        "is still statting"
    )


def test_narrow_says_on_exactly_even_when_the_target_is_a_directory(tmp_path):
    """widen=False is 'on exactly', unconditionally.

    Guards the OTHER direction: a fix that keyed off is_dir() rather than the
    carried flag would call this one a subtree grant, overstating it.
    """
    a_dir = tmp_path / "iam-a-dir"
    a_dir.mkdir()
    for target in (a_dir, tmp_path / "ghost.txt"):
        g = derive_grant(_action(target), verb="always")
        assert "on exactly" in g.describe(), f"{target}: {g.describe()!r}"
        assert "on anything under" not in g.describe()


# ── describe() must not touch the filesystem AT ALL ────────────────────────

def test_describe_makes_no_filesystem_call():
    """Asserted, not assumed — describe() must not depend on the disk.

    The spy RECORDS and delegates; it does not raise. A raising stub patched
    onto Path breaks pytest's own traceback machinery, which calls
    ``Path.exists()`` while formatting a failure — the mutation run then dies
    with an INTERNALERROR instead of a readable red. And the patches are undone
    in ``finally``, BEFORE the assertion, so no assertion is ever evaluated
    while Path is still instrumented.
    """
    seen: list[str] = []
    watched = ("is_dir", "exists", "is_file", "stat", "lstat", "resolve", "iterdir")
    saved = {n: getattr(Path, n) for n in watched}

    def spy(name, original):
        def wrapper(self, *a, **k):
            seen.append(name)
            return original(self, *a, **k)
        return wrapper

    try:
        for name, original in saved.items():
            setattr(Path, name, spy(name, original))
        for widened in (True, False):
            Grant(kind="path_prefix", value="/some/where",
                  tool_name="write_file", widened=widened).describe()
        Grant(kind="tool", value="", tool_name="bash").describe()
        Grant(kind="command_prefix", value="ls", tool_name="bash").describe()
    finally:
        for name, original in saved.items():
            setattr(Path, name, original)

    assert seen == [], (
        f"describe() reached the filesystem via {sorted(set(seen))} — the extent "
        f"must come from the carried field, not from what happens to be on disk"
    )


# ── Legacy config rows ─────────────────────────────────────────────────────

def test_legacy_config_row_without_widened_describes_as_a_subtree():
    """Rows written before this field existed ARE directory grants.

    Pre-#234 ``derive_grant`` returned ``target.parent`` unconditionally — its
    own comment records that approving one file in $HOME granted the tool
    across all of $HOME. So True is the accurate default, not merely the safe
    one. It is also the safe one, which is why it is not decided by a stat.
    """
    legacy = Grant.from_config_dict({
        "kind": "path_prefix", "value": "/home/u/proj",
        "tool": "write_file", "id": "abc123",
    })
    assert legacy is not None
    assert legacy.widened is True
    assert "on anything under" in legacy.describe()


def test_widened_survives_a_persist_reload_round_trip():
    """Carried through disk, not re-derived on the way back."""
    for widened in (True, False):
        original = Grant(kind="path_prefix", value="/p", tool_name="write_file",
                         widened=widened, scope="persistent")
        reloaded = Grant.from_config_dict(original.to_config_dict())
        assert reloaded.widened is widened
        assert reloaded.describe() == original.describe()


# ── The duration half, read back from the AUDIT ROW ────────────────────────

@pytest.mark.asyncio
async def test_always_is_recorded_as_permanent_in_the_audit_row(config, tmp_path):
    """Read from the ROW, not the return value.

    cmd_approve's operator-facing string is built AFTER the old scope patch and
    was always correct; the audit row is written BEFORE it and was not. A test
    asserting on the return value passes against the bug.
    """
    target = tmp_path / "t.txt"
    target.write_text("x")
    q = _queue(config, tmp_path)
    rid = await _request(q, grant_file_path=str(target))
    await cmds.cmd_approve(q, f"always {rid}")

    reason = _audit_reason(q)
    assert "scope=always" in reason, reason
    assert "permanently, until revoked" in reason, (
        f"an `always` grant was recorded as temporary in the accountability "
        f"store. row: {reason!r}"
    )
    assert "until the daemon restarts" not in reason


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", [v for v in approve_verbs() if v != "once"])
async def test_prompt_string_and_audit_string_are_byte_identical(
    config, tmp_path, verb
):
    """One description, two surfaces — for EVERY grant-creating verb.

    Both call describe(); the point is that they call it on grants built the
    same way. While cmd_approve patched scope after the audit write, these
    diverged for the persistent verbs.
    """
    target = tmp_path / "sub" / "t.txt"
    target.parent.mkdir()
    target.write_text("x")

    action = PendingAction(request_id="p1", tool_name="write_file",
                           description="d", grant_file_path=str(target))
    prompt_string = prospective_extents(action)[verb]

    q = _queue(config, tmp_path)
    rid = await _request(q, grant_file_path=str(target))
    await cmds.cmd_approve(q, f"{verb} {rid}")

    reason = _audit_reason(q)
    assert prompt_string in reason, (
        f"verb {verb!r}: the operator consented to\n  {prompt_string!r}\n"
        f"and the record says\n  {reason!r}"
    )


# ── Construction owns both fields; nobody patches them afterwards ──────────

def test_derive_grant_sets_both_fields_for_every_verb(tmp_path):
    target = tmp_path / "d" / "t.txt"
    target.parent.mkdir()
    expected = {
        "always":              (True, False),
        "always here":         (True, True),
        "until-restart":       (False, False),
        "until-restart here":  (False, True),
    }
    for verb, (persistent, widened) in expected.items():
        g = derive_grant(_action(target), verb=verb)
        assert g.scope == ("persistent" if persistent else "until_restart"), verb
        assert g.widened is widened, verb


def test_no_consent_caller_patches_scope_or_widened():
    """The fields are set at construction, so nothing downstream may reassign.

    Source-level on purpose: a behavioural test cannot see a patch that happens
    to be correct today and drifts tomorrow — which is precisely how the audit
    row went wrong, since the patch existed and simply ran too late.

    SecurityGate.add_grant is exempt and asserted separately: its dedupe merges
    two grants of the SAME identity, which is a documented merge, not a patch.
    """
    # Assignment only. A plain substring test matches `.scope ==` too, which is
    # a COMPARISON — cmd_approve legitimately reads effective.scope to decide
    # whether to persist. The negative lookahead is what separates writing the
    # field from reading it.
    for mod in (aq, cmds):
        src = inspect.getsource(mod)
        for field in ("scope", "widened"):
            hits = re.findall(rf"\.{field}\s*=(?!=)", src)
            assert not hits, (
                f"{mod.__name__} reassigns .{field} after construction "
                f"({len(hits)} site(s)) — derive_grant(verb=…) is the only "
                f"place either is decided"
            )


def test_dedupe_upgrade_never_narrows_the_description(config):
    """add_grant's merge takes the wider wording, never the narrower."""
    gate = SecurityGate(config_path=str(config))
    narrow = gate.add_grant(Grant(kind="path_prefix", value="/tmp/dup",
                                  tool_name="write_file", widened=False))
    assert narrow.widened is False
    merged = gate.add_grant(Grant(kind="path_prefix", value="/tmp/dup",
                                  tool_name="write_file", widened=True))
    assert merged.widened is True, (
        "the merge kept the narrow wording for an entry a widening approval "
        "also reached — matches() is lexical on value, so both admit the same "
        "paths and the description must not understate that"
    )
