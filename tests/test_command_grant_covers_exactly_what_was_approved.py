"""A remembered shell-command approval covers exactly the command the owner approved.

The grant's value is the whole approved command (``derive_grant``: "grant that exact command"), but ``Grant.matches``
was a bare ``command.startswith(value)`` behind a guard that only rejects ``; & | ` $( newline``. So a grant for
``echo hi`` also allowed ``echo hi > ~/.zshrc`` and ``echo history``: a longer command, a redirection or an extra
argument, none of which the owner was shown. These go through the gate and through the real approval queue, by
outcome.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.gateway import commands as cmds
from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction, derive_grant, prospective_extents
from prometheus.permissions.approver import in_process
from prometheus.permissions.checker import Grant, SecurityGate

BY = in_process("test")


def _gate(value, tmp_path):
    """``cautious`` asks before every shell command, so a grant is the only thing that lets one through."""
    grant = Grant(kind="command_prefix", value=value, tool_name="bash", scope="persistent")
    return SecurityGate(mode="cautious", audit_logger=None, workspace_root=[str(tmp_path)], grants=[grant])


def _decision(gate, command):
    return gate.evaluate("bash", command=command)


def _asks(gate, command):
    d = _decision(gate, command)
    return d.requires_confirmation and not d.allowed


# ── the approved command, and nothing beyond it ──────────────────────────────

def test_the_approved_command_runs_without_asking(tmp_path):
    assert _decision(_gate("echo hi", tmp_path), "echo hi").allowed


def test_the_command_is_compared_as_written_not_trimmed(tmp_path):
    # Byte for byte. A trimmed comparison let `bash deploy.sh\x0c` ride on a grant for `bash deploy.sh`; see below.
    gate = _gate("echo hi", tmp_path)
    assert _asks(gate, "  echo hi")
    assert _asks(gate, "echo hi  ")
    assert _asks(gate, "echo hi\t")


@pytest.mark.parametrize("blank", ["\r", "\x0b", "\x0c", "\x1c", "\x1f", "\x85", "\xa0", "\u2003", "\u2028", "\u3000"],
                         ids=lambda c: f"U+{ord(c):04X}")
def test_a_control_or_unicode_blank_is_part_of_the_command_not_padding(blank, tmp_path):
    """str.strip() treats these as blank; the shell does not. `bash deploy.sh\\x0c` runs a file named `deploy.sh\\x0c`."""
    gate = _gate("bash deploy.sh", tmp_path)
    assert _asks(gate, "bash deploy.sh" + blank)
    assert _asks(gate, blank + "bash deploy.sh")


@pytest.mark.parametrize("command", [
    "echo hi > ~/.zshrc",          # a redirection the owner was never shown
    "echo hi >> notes.txt",
    "echo hi < /dev/null",
    "echo hi there",               # an extra argument
    "echo hi -n",
    "echo history",                # a longer word that merely begins with it
    "echo hi\tthere",
    "echo hi # > x",
    "echo hi$IFS>x",
], ids=["redirect", "append", "stdin", "extra-arg", "extra-flag", "longer-word", "tab-arg", "comment", "ifs"])
def test_anything_longer_than_the_approved_command_asks_again(command, tmp_path):
    gate = _gate("echo hi", tmp_path)
    assert _asks(gate, command), f"{command!r} ran on the strength of an approval for 'echo hi'"


@pytest.mark.parametrize(("approved", "other"), [
    ("rm notes.txt", "rm notes.txt ~/Documents/report.txt"),
    ("git status", "git statusx"),
    ("git pull", "git pull --force origin main"),
    ("python3 script.py", "python3 script.py --delete-everything"),
    ("ls /tmp", "ls /tmpfoo"),
], ids=["rm", "git-word", "git-flags", "python-args", "path-word"])
def test_the_same_holds_for_commands_that_do_real_work(approved, other, tmp_path):
    gate = _gate(approved, tmp_path)
    assert _decision(gate, approved).allowed
    assert _asks(gate, other)


def test_a_chained_command_is_still_never_remembered(tmp_path):
    # Unchanged: even if the owner approved a chain, it is not a single invocation, so it never matches.
    gate = _gate("ls; ls", tmp_path)
    assert _asks(gate, "ls; ls")


def test_a_row_with_no_command_in_it_covers_nothing(tmp_path):
    # startswith("") is True, so a command_prefix row with an empty value used to allow every single-invocation command.
    gate = _gate("", tmp_path)
    assert _asks(gate, "ls")
    assert _asks(gate, "   ")


def test_another_tool_is_not_covered(tmp_path):
    grant = Grant(kind="command_prefix", value="echo hi", tool_name="bash")
    assert not grant.matches("write_file", None, "echo hi")


# ── through the real approval queue ──────────────────────────────────────────

async def _approve_always(command, tmp_path):
    config = tmp_path / "prometheus.yaml"
    config.write_text("security:\n  permission_mode: cautious\n")
    queue = ApprovalQueue(security_gate=SecurityGate(), timeout_seconds=5)
    gate = SecurityGate(mode="cautious", audit_logger=None, workspace_root=[str(tmp_path)], approval_queue=queue,
                        config_path=str(config))
    queue._security_gate = gate
    task = asyncio.create_task(queue.request_approval("bash", command, grant_command=command))
    for _ in range(100):
        if queue.pending:
            break
        await asyncio.sleep(0.01)
    assert queue.pending, "request_approval did not queue"
    queue._test_task = task
    rid = next(iter(queue.pending))
    await cmds.cmd_approve(queue, f"always {rid}", by=BY)
    return gate


@pytest.mark.asyncio
async def test_an_always_answer_to_a_prompt_covers_only_the_command_it_showed(tmp_path):
    gate = await _approve_always("ls -la /tmp", tmp_path)
    assert len(gate.list_grants()) == 1, "control: the answer was remembered"

    assert _decision(gate, "ls -la /tmp").allowed
    assert _asks(gate, "ls -la /tmp > ~/out.txt")
    assert _asks(gate, "ls -la /tmp/secrets")
    assert _asks(gate, "ls -la /tmp ~/.aws")


# ── a command that could never be remembered is not offered as a lasting grant ─

@pytest.mark.parametrize("command", ["ls | wc -l", "cd x && make", "echo 'a;b'", "echo $(date)", "echo `id`"])
def test_a_chained_command_is_not_offered_as_something_that_will_be_remembered(command):
    # The single-invocation guard means such a grant could never match, so offering "always" promised a remembered
    # approval that silently never applied, and wrote a dead row to the config.
    action = PendingAction(request_id="r1", tool_name="bash", description="d", grant_command=command)
    assert derive_grant(action) is None
    assert prospective_extents(action) == {}


def test_a_plain_command_is_still_offered():
    action = PendingAction(request_id="r1", tool_name="bash", description="d", grant_command="ls -la /tmp")
    assert derive_grant(action) is not None
    assert "always" in prospective_extents(action)


# ── a grant never lifts a block ──────────────────────────────────────────────

def test_an_exact_grant_for_a_denied_command_is_still_denied(tmp_path):
    grant = Grant(kind="command_prefix", value="shutdown now", tool_name="bash", scope="persistent")
    gate = SecurityGate(mode="cautious", audit_logger=None, workspace_root=[str(tmp_path)],
                        denied_commands=["shutdown"], grants=[grant])
    d = _decision(gate, "shutdown now")
    assert not d.allowed and not d.requires_confirmation and d.action == "DENY"


# ── the persisted row, read back by a fresh gate ─────────────────────────────

@pytest.mark.asyncio
async def test_a_persisted_command_grant_covers_only_its_command_after_a_reload(tmp_path):
    await _approve_always("ls -la /tmp", tmp_path)
    reloaded = SecurityGate.from_config(str(tmp_path / "prometheus.yaml"))
    assert len(reloaded.list_grants()) == 1, "control: the row was read back"
    assert _decision(reloaded, "ls -la /tmp").allowed
    assert _asks(reloaded, "ls -la /tmp > ~/out.txt")
    assert _asks(reloaded, "ls -la /tmp\x0c")


def test_two_grants_that_differ_only_by_a_trailing_space_are_two_grants(tmp_path):
    # Dedupe compares the raw value, and matching now does too, so the two cannot disagree.
    gate = SecurityGate(mode="cautious", audit_logger=None, workspace_root=[str(tmp_path)])
    gate.add_grant(Grant(kind="command_prefix", value="echo hi", tool_name="bash", scope="until_restart"))
    gate.add_grant(Grant(kind="command_prefix", value="echo hi ", tool_name="bash", scope="persistent"))
    assert len(gate.list_grants()) == 2
    assert _decision(gate, "echo hi").allowed and _decision(gate, "echo hi ").allowed


# ── what the owner is told ───────────────────────────────────────────────────

def test_the_grant_is_described_as_exactly_that_command(tmp_path):
    text = Grant(kind="command_prefix", value="ls -la /tmp", tool_name="bash", scope="persistent").describe()
    assert "exactly" in text and "ls -la /tmp" in text
    assert "starting with" not in text, "the owner is told a prefix is covered, which is what the bug did"


def test_a_row_read_back_from_the_config_covers_only_its_own_command():
    grant = Grant.from_config_dict({"kind": "command_prefix", "tool": "bash", "value": "git status", "id": "abc123"})
    assert grant is not None
    assert grant.matches("bash", None, "git status")
    assert not grant.matches("bash", None, "git status -s")
    assert not grant.matches("bash", None, "git status > x")
