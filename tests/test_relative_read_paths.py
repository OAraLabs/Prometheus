"""A relative READ path resolves; a relative WRITE path still prompts.

THE DEFECT (measured on a live daemon, 2026-08-30). ``read_file`` with a
relative target — the shape the model produces constantly, e.g.
``src/prometheus/web/ws_server.py`` — was UNKNOWN, so every such call raised an
approval. That alone would be merely noisy. What made it unanswerable is what
UNKNOWN does downstream: with no resolved target there is no extent to
describe, so ``prospective_extents`` returns ``{}`` and the approval card
offers NO remembered grant. The operator gets "Approve once / Deny" and
nothing else — on Beacon and Telegram at once, forever, with no answer that
stops the asking. Two such approvals were pending simultaneously, both with
zero extents.

THE ASYMMETRY THAT STAYS. The relative→UNKNOWN rule protects WRITES: it must
never be "where the process happens to be running" that decides whether a
write is allowed. A read decides nothing of the sort — the tool resolves the
same relative path against the same cwd whatever the gate concludes, so
refusing to resolve makes the gate rule on a path nobody reads. That is the
identical unsoundness already rejected for ``grep``/``glob`` roots.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.permissions.tool_paths import gate_path_for


@pytest.fixture()
def base(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.py").write_text("x = 1\n")
    return tmp_path


def test_relative_read_resolves_against_base(base: Path):
    path, unknown = gate_path_for(
        "read_file", {"path": "src/thing.py"}, base=base,
    )
    assert unknown is None, "a relative read must no longer be unknown"
    assert path == str((base / "src" / "thing.py").resolve())


def test_relative_write_is_still_unknown(base: Path):
    """The rule the asymmetry exists for. If this ever passes a path, a write's
    permissibility has become a function of the process's cwd again."""
    for tool in ("write_file", "edit_file"):
        path, unknown = gate_path_for(tool, {"path": "src/thing.py"}, base=base)
        assert path is None, f"{tool} must not resolve a relative target"
        assert unknown and "relative path" in unknown


def test_relative_read_without_a_base_stays_unknown():
    """No base means nothing to resolve against — UNKNOWN, never a guess."""
    path, unknown = gate_path_for("read_file", {"path": "src/thing.py"})
    assert path is None
    assert unknown and "relative path" in unknown


def test_absolute_read_is_unchanged(base: Path):
    target = base / "src" / "thing.py"
    path, unknown = gate_path_for("read_file", {"path": str(target)}, base=base)
    assert unknown is None
    assert path == str(target.resolve())


def test_resolution_feeds_the_floor_rather_than_bypassing_it(tmp_path: Path):
    """Resolving is what LETS the floor rule on a relative read: the gate now
    gets a concrete path to compare against denied_paths, where before it got
    None and could only prompt. Traversal is resolved, not preserved."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    workdir = home / "work"
    workdir.mkdir()

    path, unknown = gate_path_for(
        "read_file", {"path": "../.ssh/id_rsa"}, base=workdir,
    )
    assert unknown is None
    assert path == str((home / ".ssh" / "id_rsa").resolve()), (
        "the gate must see the real target, which is what makes a deny possible"
    )


def test_home_relative_read_still_expands(base: Path):
    """~ was already expanded before this change; it must stay that way."""
    path, unknown = gate_path_for("read_file", {"path": "~/anything.txt"}, base=base)
    assert unknown is None
    assert path == str((Path.home() / "anything.txt").resolve())
