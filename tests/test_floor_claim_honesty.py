"""The code must not misdescribe the floor.

Two defects with one thesis. ``SecurityGate.evaluate``'s docstring asserted
that ``denied_paths`` still applies to user-origin bash — it never has, since
1da4dd2 wrote the line — and the file-mutation verifier stat-ed unexpanded
``~`` paths, so a write to ``~/.ssh/x`` produced no mutation record and no
BOUNDARY ESCAPE line. Both make the system look more protected than it is,
which is worse than an undocumented hole: a reader auditing the code gets a
false assurance from the code itself.

Neither fix is a control. The verifier still runs AFTER the write and holds
no content, so it can report but never refuse. This suite exists to keep the
descriptions true, not to claim the hole is closed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prometheus.hooks.file_mutation_verifier import (
    FileMutationVerifier,
    _expand_user,
    _extract_bash_paths,
)
from prometheus.permissions.checker import SecurityGate

pytestmark = pytest.mark.integration


def _decision(gate: SecurityGate, **kw) -> str:
    """ALLOW / APPROVE / DENY as the gate itself labels it.

    Read ``action``, not ``allowed``: an approval-required decision also has
    ``allowed=False``, and collapsing the two reads a prompt as a refusal —
    the exact conflation this sprint exists to remove.
    """
    return str(gate.evaluate(**kw).action)


# ---------------------------------------------------------------------------
# (a) the docstring must match the outcomes
# ---------------------------------------------------------------------------


class TestEvaluateDocstringIsTrue:
    """Bind the prose at checker.py's ``evaluate`` to measured behaviour.

    A docstring is not executable, so nothing else can catch it drifting.
    """

    def test_does_not_claim_denied_paths_applies_to_user_bash(self):
        doc = SecurityGate.evaluate.__doc__ or ""
        # The old text listed denied_paths among the things that "STILL apply"
        # to user-origin bash. Find the sentence and assert it is gone.
        head, _, _ = doc.partition("``denied_paths``")
        still_applies = head[head.find("``\"user\"``"):]
        assert "denied_paths" not in still_applies, (
            "evaluate() again claims denied_paths applies to user-origin "
            "bash. It does not — see the outcome tests below."
        )

    def test_states_the_file_path_precondition(self):
        doc = SecurityGate.evaluate.__doc__ or ""
        assert "file_path" in doc and "does NOT cover bash" in doc, (
            "the docstring must say WHY the floor misses bash — that "
            "_check_denied_path is nested under `if file_path:` — or the "
            "next reader re-derives it from scratch"
        )

    # -- the outcomes the docstring now asserts, measured ------------------
    #
    # These four are deliberately coupled to the prose. When the bash floor
    # is eventually enforced, the ALLOW case flips to DENY, this test goes
    # red, and whoever does that work is forced to update the docstring in
    # the same commit. That coupling is the point, not an accident.

    @pytest.fixture()
    def gate(self, tmp_path: Path) -> SecurityGate:
        return SecurityGate(
            denied_commands=["forbidden-marker"],
            workspace_root=str(tmp_path),
        )

    def test_always_blocked_applies_to_user_bash(self, gate):
        assert _decision(gate, tool_name="bash", command="rm -rf /",
                         origin="user") == "DENY"

    def test_denied_commands_applies_to_user_bash(self, gate):
        assert _decision(gate, tool_name="bash",
                         command="forbidden-marker x", origin="user") == "DENY"

    def test_workspace_gate_applies_as_a_PROMPT_not_a_refusal(self, gate):
        # It "applies", but it asks. Recording the difference because the
        # floor below is the thing that refuses, and conflating the two is
        # how a speed bump gets described as a wall.
        assert _decision(gate, tool_name="write_file",
                         file_path="/home/testuser/elsewhere/x",
                         origin="user") == "APPROVE"

    @pytest.mark.parametrize("tool,path", [
        ("write_file", "/home/testuser/.ssh/x"),
        ("read_file", "/home/testuser/.ssh/private-key"),
        ("write_file", "/home/testuser/.gnupg/x"),
    ])
    def test_floor_REFUSES_for_path_declaring_tools(self, gate, tool, path):
        # Not APPROVE. The floor is the one control in the system that is
        # unconditional, and a prompt would make it conditional on an
        # operator being awake.
        assert _decision(gate, tool_name=tool, file_path=path,
                         origin="user") == "DENY"

    @pytest.mark.parametrize("command", [
        "cat /home/testuser/.gnupg/x",
        "echo x > /home/testuser/.gnupg/x",
    ])
    @pytest.mark.parametrize("origin", ["user", "system"])
    def test_floor_does_NOT_hold_for_bash_either_origin(
        self, gate, command, origin,
    ):
        """The floor never fires for bash. Documenting, not endorsing.

        ``.gnupg`` is the clean probe: no approve-pattern mentions it, so the
        verdict here is the floor's verdict and nothing else. If this ever
        flips to DENY the bash floor has been enforced — update the
        docstring above in the same commit.
        """
        assert _decision(gate, tool_name="bash", command=command,
                         origin=origin) == "ALLOW"

    def test_ssh_is_stopped_at_system_origin_by_a_WORD_not_the_floor(self, gate):
        """Why ``.ssh`` looks protected at system origin, and why it isn't.

        ``\\bssh\\b`` in _APPROVE_BASH_PATTERNS matches the substring in
        ``.ssh`` — a name coincidence aimed at the ssh CLIENT, not a path
        check. It yields APPROVE (never DENY), it does not apply at user
        origin at all, and the ``.gnupg`` control above shows what happens
        to a floor path with no such word: straight through.
        """
        cmd = "cat /home/testuser/.ssh/private-key"
        assert _decision(gate, tool_name="bash", command=cmd,
                         origin="system") == "APPROVE"
        assert _decision(gate, tool_name="bash", command=cmd,
                         origin="user") == "ALLOW"
        # Rename the directory and the "protection" evaporates — proof it is
        # keyed on the word, not on the path.
        assert _decision(gate, tool_name="bash", origin="system",
                         command="cat /home/testuser/.gnupg/secring.gpg") == "ALLOW"


# ---------------------------------------------------------------------------
# (b) the verifier must expand ~ before it stats
# ---------------------------------------------------------------------------


class TestTildeExpansion:
    def test_expands_leading_tilde(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _expand_user("~/.ssh/x") == str(tmp_path / ".ssh/x")

    def test_absolute_path_untouched(self):
        assert _expand_user("/etc/passwd") == "/etc/passwd"

    def test_relative_path_untouched(self):
        # Deliberate: a bash clause can `cd` first, so resolving a relative
        # path against the daemon's cwd would invent a path never used.
        assert _expand_user("notes/x.md") == "notes/x.md"

    def test_extraction_yields_expanded_bash_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        out = _extract_bash_paths("echo W > ~/.ssh/floor-w.txt")
        assert out == [(str(tmp_path / ".ssh/floor-w.txt"), "redirect_write")]

    def test_extraction_yields_expanded_tool_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        v = FileMutationVerifier()
        assert v._paths_for("file_write", {"file_path": "~/notes.md"}) == [
            str(tmp_path / "notes.md")
        ]


class TestTildeWriteIsSeenEndToEnd:
    """Both directions: the tilde write is now seen, the absolute one still is."""

    def _run(self, verifier, command, turn_key):
        verifier.pre_tool_use("bash", {"command": command}, "t1",
                              turn_key=turn_key)
        os.system(command)
        verifier.post_tool_use("bash", {"command": command}, "t1",
                               output="", turn_key=turn_key)

    def test_tilde_write_is_seen(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".ssh").mkdir()
        v = FileMutationVerifier()
        # The command must carry a literal ~ — that is the whole defect. An
        # absolute path here would pass before the fix as well.
        self._run(v, "echo W > ~/.ssh/t.txt", "turn-tilde")

        landed = v.landed_paths(turn_key="turn-tilde")
        assert landed == [str(tmp_path / ".ssh/t.txt")], (
            "a ~-path write must now be reported; before the fix the "
            "unexpanded literal was stat-ed, was absent before AND after, "
            "and produced an empty list"
        )

    def test_absolute_write_still_seen(self, tmp_path):
        v = FileMutationVerifier()
        target = tmp_path / "abs.txt"
        self._run(v, f"echo W > {target}", "turn-abs")
        assert v.landed_paths(turn_key="turn-abs") == [str(target)]

    def test_landed_path_is_what_the_gate_can_match(self, tmp_path, monkeypatch):
        """The reason expansion matters beyond stat().

        ``_boundary_escapes`` feeds landed_paths() to the gate, and the floor
        globs are absolute (``/*/.ssh``). A ``~``-prefixed literal cannot
        match one, so an unexpanded path defeats the check twice over.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / ".ssh").mkdir()
        v = FileMutationVerifier()
        self._run(v, "echo W > ~/.ssh/t.txt", "turn-gate")

        landed = v.landed_paths(turn_key="turn-gate")
        assert landed and not landed[0].startswith("~")
        assert Path(landed[0]).is_absolute()
