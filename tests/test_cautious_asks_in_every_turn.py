"""In ``cautious`` mode every shell command the model runs asks, in the owner's own turns too.

``cautious`` promises "EVERY shell command the model runs asks" (permissions/modes.py), and it is the mode a new
Prometheus.app install starts in, where macOS gives the daemon no shell write floor, so the prompt is the protection.
But the rule only applied to system-origin turns (``not is_user``): in a turn the owner started (Telegram, Beacon,
the CLI) the model's ``bash`` ran unasked. A file write the gate held for approval in that same turn could be done
through the shell instead: ``write_file`` waits for the owner, ``echo ... > file`` does not.

In a user-origin turn the owner asked for the turn; the model wrote the command. So cautious asks for both origins.
What still runs unasked in cautious, as before: a call under a remembered grant, a command on the operator's
trusted-command allowlist, and reads.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.permissions.checker import SecurityGate

OWNER_TURN = "telegram:12345"  # a user-origin session: the owner started this turn


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    return root


def _gate(mode, ws, **kw):
    return SecurityGate(mode=mode, audit_logger=None, workspace_root=[str(ws)], **kw)


@pytest.mark.parametrize("command", ["ls", "echo hi > notes.txt", "cat notes.txt", "python3 script.py"])
def test_cautious_asks_before_a_shell_command_in_the_owners_own_turn(ws, command):
    d = _gate("cautious", ws).evaluate("bash", command=command, origin="user")
    assert not d.allowed and d.requires_confirmation, d


def _loop(tmp_path, mode, prompted, *, session_id=OWNER_TURN):
    from prometheus.__main__ import create_tool_registry
    from prometheus.engine.agent_loop import LoopContext

    gate = _gate(mode, tmp_path / "ws")

    async def prompt(tool_name, reason):
        prompted.append((tool_name, reason))
        return False  # the owner says no

    return LoopContext(provider=None, model="t", system_prompt="", max_tokens=512,
                       tool_registry=create_tool_registry({}, gate), permission_checker=gate,
                       permission_prompt=prompt, cwd=tmp_path / "ws", session_id=session_id)


def _call(ctx, tool, args):
    from prometheus.engine.agent_loop import _execute_tool_call

    return asyncio.run(_execute_tool_call(ctx, tool, "t1", args))


def test_a_write_held_for_approval_cannot_go_through_the_shell_instead(tmp_path, ws):
    # The finding: in the owner's turn, write_file waits for the owner; the same write through bash did not.
    target = ws / "held.txt"
    prompted: list = []
    ctx = _loop(tmp_path, "cautious", prompted)
    _call(ctx, "write_file", {"path": str(target), "content": "x\n"})
    assert prompted and not target.exists(), "control: the file write asks in cautious mode"
    prompted.clear()
    _call(ctx, "bash", {"command": f"echo x > {target}"})
    assert prompted and prompted[0][0] == "bash", "the shell ran in the owner's turn without asking"
    assert not target.exists(), "the held write went through the shell"


def test_default_and_strict_owner_turns_are_unchanged(tmp_path, ws):
    # Only cautious promises that every shell command asks; default and strict run an ordinary one unasked.
    marker = ws / "ran"
    for mode in ("default", "strict"):
        prompted: list = []
        _call(_loop(tmp_path, mode, prompted), "bash", {"command": f"touch {marker}"})
        assert not prompted and marker.exists(), mode
        marker.unlink()


def test_a_trusted_command_still_runs_unasked_in_cautious(ws):
    gate = _gate("cautious", ws, allowed_commands=[r"^ls$"])
    assert gate.evaluate("bash", command="ls", origin="user").allowed
    assert not gate.evaluate("bash", command="ls -la", origin="user").allowed


def test_reads_still_do_not_ask_in_cautious(ws):
    assert _gate("cautious", ws).evaluate("read_file", file_path=str(ws / "a.txt"), is_read_only=True,
                                          origin="user").allowed
