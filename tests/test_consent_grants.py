"""SPRINT-CONSENT Phase 5 — consent describes extent, and grants can be revoked.

WHY THESE ASSERT WHAT THEY ASSERT
---------------------------------
The existing ``test_approval_grants.py`` (20 tests) passes with
``persist_grant`` DELETED. Verified as a mutation before this file was
written: making ``persist_grant`` a no-op that returns True left the entire
4,865-test suite green. It asserts on ``gate.list_grants()`` — the in-memory
list — and on ``grants[0].scope``, a label on an object. Nothing anywhere
read the config file back.

So the load-bearing assertions here read the FILE. ``test_persist_grant_...``
below is the one the mutation must kill; if it ever survives that deletion
again, this file has become decoration.

Real objects throughout: a real SecurityGate, a real ApprovalQueue, a real
YAML file on disk. Nothing is doubled.
"""

from __future__ import annotations

import asyncio

import pytest
import yaml

from prometheus.gateway import commands as cmds
from prometheus.permissions.approval_queue import (
    ApprovalQueue, PendingAction, derive_grant, prospective_extents)
from prometheus.permissions.checker import Grant, SecurityGate


@pytest.fixture
def config(tmp_path):
    """A real, minimal prometheus.yaml on disk."""
    p = tmp_path / "prometheus.yaml"
    p.write_text(yaml.dump({"security": {"denied_paths": ["/etc"]}}))
    return p


def _disk_grants(config) -> list:
    return (yaml.safe_load(config.read_text()).get("security") or {}).get("grants") or []


def _queue(config) -> ApprovalQueue:
    q = ApprovalQueue(timeout_seconds=5)
    q._security_gate = SecurityGate(approval_queue=q, config_path=str(config))
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


# ── Test 8 — THE ONE THE MUTATION MUST KILL ────────────────────────────────

@pytest.mark.asyncio
async def test_persist_grant_actually_writes_to_the_config_file(config, tmp_path):
    """Reads the FILE, not list_grants().

    This is the assertion the pre-sprint suite lacked. With ``persist_grant``
    deleted, ``list_grants()`` still shows the grant (add_grant put it there)
    and every old test stayed green — so the disk half was never observed."""
    target = tmp_path / "target.txt"
    target.write_text("x")
    q = _queue(config)
    rid = await _request(q, grant_file_path=str(target))

    assert _disk_grants(config) == [], "config had grants before we made any"
    await cmds.cmd_approve(q, f"always {rid}")

    on_disk = _disk_grants(config)
    assert len(on_disk) == 1, (
        f"persist_grant did not write to {config}. list_grants() is not "
        f"evidence of persistence — that is exactly what this test exists "
        f"to catch. disk={on_disk!r}"
    )
    assert on_disk[0]["value"] == str(target)
    assert on_disk[0]["id"], "no grant_id persisted — revocation has no handle"


# ── Test 1 — the prompt states the extent it will grant ────────────────────

def test_prompt_states_the_extent_not_just_the_request(tmp_path):
    """Consent must describe the GRANT, not the request.

    The defect: the prompt named one file and said "remember permanently",
    while the grant covered that file's whole parent directory."""
    target = tmp_path / "sub" / "f.txt"
    target.parent.mkdir()
    target.write_text("x")
    action = PendingAction(
        request_id="abc12345", tool_name="write_file",
        description=f"write_file targets path outside workspace: {target}",
        grant_file_path=str(target),
    )

    extents = prospective_extents(action)
    assert "always" in extents and "until-restart" in extents
    assert "always here" in extents, "the widening opt-in must be offered explicitly"

    # The narrow default must describe itself as narrow...
    assert str(target) in extents["always"]
    assert "on exactly" in extents["always"]
    # ...and the opt-in must name the DIRECTORY it widens to.
    assert str(target.parent) in extents["always here"]
    assert "anything under" in extents["always here"]
    # Duration is stated too, and differs between the verbs.
    assert "permanently" in extents["always"]
    assert "restarts" in extents["until-restart"]


@pytest.mark.asyncio
async def test_extent_shown_equals_extent_granted(config, tmp_path):
    """Byte-identity between what was described and what was created.

    Both come from one ``derive_grant`` call, so this pins that they cannot
    drift — the §17 under-population failure, aimed at a description."""
    target = tmp_path / "f.txt"
    target.write_text("x")
    q = _queue(config)
    rid = await _request(q, grant_file_path=str(target))
    action = q.pending[rid]
    shown = prospective_extents(action)["always"]

    await cmds.cmd_approve(q, f"always {rid}")
    created = q._security_gate.list_grants()[0]

    assert created.describe() == shown, (
        f"the operator was shown {shown!r} and got {created.describe()!r}"
    )


# ── Test 2 — the default narrows ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_default_grants_the_exact_file_not_its_parent(config, tmp_path):
    """DELIBERATE BEHAVIOUR CHANGE. One approval of one file used to grant
    write_file over that file's entire parent directory."""
    target = tmp_path / "only-this.txt"
    target.write_text("x")
    q = _queue(config)
    rid = await _request(q, grant_file_path=str(target))
    await cmds.cmd_approve(q, f"always {rid}")

    g = q._security_gate.list_grants()[0]
    assert g.value == str(target), f"expected the exact file, got {g.value!r}"
    assert g.value != str(target.parent), "the default widened to the parent again"
    assert _disk_grants(config)[0]["value"] == str(target)


@pytest.mark.asyncio
async def test_here_opts_in_to_the_directory_grant(config, tmp_path):
    """The widening still exists — it is now asked for."""
    target = tmp_path / "sub" / "f.txt"
    target.parent.mkdir()
    target.write_text("x")
    q = _queue(config)
    rid = await _request(q, grant_file_path=str(target))
    await cmds.cmd_approve(q, f"always here {rid}")

    g = q._security_gate.list_grants()[0]
    assert g.value == str(target.parent), f"'here' did not widen; got {g.value!r}"


# ── Test 3 — rule 4 creates nothing ────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_target_creates_no_persistent_grant(config):
    """Asserted by reading the STORE, not the return value.

    Rule 4 produced kind="tool" value="" — the widest grant in the system,
    from the case carrying the least information."""
    q = _queue(config)
    rid = await _request(q)  # no grant_file_path, no grant_command
    text = await cmds.cmd_approve(q, f"always {rid}")

    assert q._security_gate.list_grants() == [], "a grant was created from no target"
    assert _disk_grants(config) == [], "a grant reached the config from no target"
    assert "once" in text.lower(), f"the operator was not told it was once-only: {text!r}"


def test_derive_grant_returns_none_without_a_target():
    action = PendingAction(request_id="x", tool_name="bash", description="strict")
    assert derive_grant(action) is None
    assert prospective_extents(action) == {}, "an undescribable grant was offered"


# ── Test 4 — revocation, both halves, and it stays gone ────────────────────

@pytest.mark.asyncio
async def test_revocation_clears_memory_and_disk_and_survives_reload(config, tmp_path):
    """A revoke that clears only memory is undone by the next restart; one
    that clears only disk is live until then. Both halves, then a
    reload-equivalent proving it stays gone."""
    target = tmp_path / "f.txt"
    target.write_text("x")
    q = _queue(config)
    rid = await _request(q, grant_file_path=str(target))
    await cmds.cmd_approve(q, f"always {rid}")

    gate = q._security_gate
    gid = gate.list_grants()[0].grant_id
    assert len(_disk_grants(config)) == 1

    assert gate.remove_grant(gid) is True
    assert gate.list_grants() == [], "memory half not cleared"
    assert _disk_grants(config) == [], "disk half not cleared — returns on restart"

    reloaded = SecurityGate.from_config(str(config))
    assert reloaded.list_grants() == [], "the grant came back on reload"


def test_remove_grant_reports_a_miss(config):
    gate = SecurityGate(config_path=str(config))
    assert gate.remove_grant("nonexistent") is False


def test_clear_grants_removes_every_scope(config):
    gate = SecurityGate(config_path=str(config))
    gate.add_grant(Grant(kind="path_prefix", value="/a", tool_name="write_file"))
    gate.add_grant(Grant(
        kind="path_prefix", value="/b", tool_name="write_file", scope="persistent"))
    assert gate.clear_grants() == 2
    assert gate.list_grants() == []


# ── Prompt-time path resolution must never break the prompt ────────────────

def test_symlink_loop_does_not_break_the_prompt(tmp_path):
    """A symlink loop used to raise RuntimeError out of derive_grant.

    That raise pre-dates SPRINT-CONSENT, but it only ever broke an APPROVAL
    before. prospective_extents calls derive_grant at PROMPT time, so an
    uncaught raise would mean the operator is never told a permission was
    requested — a silent denial-by-omission on the security surface."""
    (tmp_path / "a").symlink_to(tmp_path / "b")
    (tmp_path / "b").symlink_to(tmp_path / "a")
    action = PendingAction(
        request_id="x", tool_name="write_file", description="d",
        grant_file_path=str(tmp_path / "a"),
    )

    extents = prospective_extents(action)  # must not raise
    assert extents, "a symlink loop silenced the prompt's extent block"
    g = derive_grant(action)
    assert g is not None
    # Fails CLOSED: the unresolved literal is narrower than a resolved path,
    # never wider.
    assert str(tmp_path / "a") in g.value


def test_extent_does_not_disclose_whether_the_target_exists(tmp_path):
    """The prompt renders identically for an existing and a missing file.

    Checked because prospective_extents stats the path before the operator
    has approved anything; if the wording differed, the prompt would leak
    filesystem existence to whoever can trigger a request."""
    real = tmp_path / "real.txt"
    real.write_text("x")

    def render(p):
        a = PendingAction(request_id="x", tool_name="write_file",
                          description="d", grant_file_path=str(p))
        return prospective_extents(a)["always"].replace(str(p), "<PATH>")

    assert render(real) == render(tmp_path / "ghost.txt")


# ── The dedupe upgrade — memory and disk must not disagree ─────────────────

def test_persistent_grant_upgrades_an_existing_until_restart_twin(config):
    """add_grant used to ignore scope, so this second call hit an early
    return: the in-memory entry stayed ``until_restart`` while
    ``persist_grant`` — a separate call in cmd_approve — still wrote it to
    disk. Two stores of one truth, inside the permission system.

    Caught by mutation M-DEDUPE, which survived the first version of this
    file: the behaviour was proven interactively and never asserted."""
    gate = SecurityGate(config_path=str(config))
    first = gate.add_grant(Grant(
        kind="path_prefix", value="/tmp/dup", tool_name="write_file"))
    assert first.scope == "until_restart"
    # Captured as a STRING before the second call. Comparing
    # ``second.grant_id == first.grant_id`` would compare two references to
    # the SAME upgraded object — a self-referential assertion no input can
    # fail (PR #145's M2). Mutation M-DEDUPE-ID survived until this line.
    original_id = str(first.grant_id)

    second = gate.add_grant(Grant(
        kind="path_prefix", value="/tmp/dup", tool_name="write_file",
        scope="persistent"))

    grants = gate.list_grants()
    assert len(grants) == 1, "the upgrade created a duplicate entry"
    assert grants[0].scope == "persistent", (
        "a persistent approval did not upgrade its until_restart twin — "
        "memory now disagrees with what persist_grant writes to disk"
    )
    assert second.grant_id == original_id, (
        f"the upgrade minted a new id ({second.grant_id} != {original_id}); "
        f"the operator may already be holding the old one"
    )


def test_upgrade_does_not_downgrade(config):
    """The reverse must NOT happen: an until_restart approval may not
    silently shorten a permanent grant the operator already consented to."""
    gate = SecurityGate(config_path=str(config))
    gate.add_grant(Grant(kind="path_prefix", value="/tmp/d2",
                         tool_name="write_file", scope="persistent"))
    gate.add_grant(Grant(kind="path_prefix", value="/tmp/d2",
                         tool_name="write_file", scope="until_restart"))
    assert gate.list_grants()[0].scope == "persistent"


# ── Writing a grant must not delete the operator's documentation ───────────

COMMENTED_CONFIG = """\
# Prometheus configuration — reference defaults
# SECRETS never belong in this file.

security:
  # ⚠ THIS IS A SPEED BUMP, NOT CONFINEMENT. It gates write_file and
  # edit_file only. `bash` is checked on its COMMAND STRING.
  workspace_root:
  - ~/projects
  # THE ABOVE DOES NOT COVER bash. denied_paths is only consulted when a
  # call passes a file_path, and bash is handed a command string.
  denied_paths:
  - "/etc"
  - "/*/.ssh"           # ANY home, not just the daemon's
  grants: []

# Assembly-time context compaction (the relief valve).
compaction:
  enabled: true
"""


@pytest.fixture
def commented_config(tmp_path):
    p = tmp_path / "prometheus.yaml"
    p.write_text(COMMENTED_CONFIG)
    return p


def test_persisting_a_grant_preserves_config_comments(commented_config):
    """A grant write must not strip the file's comments.

    THIS IS NOT HYPOTHETICAL. ``_rewrite_config_grants`` used to
    ``yaml.dump`` the whole file, and the live config reached 0 comment
    lines against a shipped template carrying 430 — the blocks explaining
    that ``denied_paths`` does not cover bash, and that this gate is a speed
    bump rather than confinement. The file an operator opens to learn what a
    key means had been emptied of meaning by a routine write.

    The mutation this test must kill: swap the splice back for
    ``yaml.dump(on_disk, ...)``. Every other test in this file survives that
    change, because none of them look at anything but parsed values.
    """
    before = commented_config.read_text()
    assert before.count("#") > 5, "fixture is not actually commented"

    gate = SecurityGate(config_path=str(commented_config))
    # persist_grant, not add_grant: add_grant is the MEMORY half. The disk
    # half is a separate call (see add_grant's own docstring), and the disk
    # half is what deletes comments.
    gate.persist_grant(Grant(kind="path_prefix", value="/tmp/keep",
                             tool_name="write_file", scope="persistent"))

    after = commented_config.read_text()
    assert _disk_grants(commented_config), "grant never reached the file"

    for line in before.splitlines():
        if line.lstrip().startswith("#"):
            assert line in after, (
                f"a grant write deleted the comment {line.strip()!r}. The "
                f"config is the operator's documentation; a permissions "
                f"write must not consume it."
            )


def test_grant_write_changes_nothing_but_grants(commented_config):
    """Everything outside ``security.grants`` survives byte-for-byte.

    Comment survival alone is too weak a claim: a rewrite could keep the
    comments and still reorder keys, restyle quoting, or drop a value.
    """
    before = commented_config.read_text()
    gate = SecurityGate(config_path=str(commented_config))
    grant = Grant(kind="path_prefix", value="/tmp/x",
                  tool_name="write_file", scope="persistent")
    gate.add_grant(grant)
    gate.persist_grant(grant)

    after = commented_config.read_text()
    expected = yaml.safe_load(before)
    expected["security"]["grants"] = _disk_grants(commented_config)
    assert yaml.safe_load(after) == expected, (
        "the grant write altered a key other than security.grants"
    )

    gate.remove_grant(grant.grant_id, config_path=str(commented_config))
    assert commented_config.read_text() == before, (
        "add-then-revoke did not restore the file byte-for-byte; the writer "
        "is leaving formatting drift behind on every approval"
    )
