"""The download tool's protected-path guard compares like with like (WP-X.23).

WHAT WAS WRONG
--------------
``_resolve_destination`` resolves the destination before checking it, and
compared the result with prefixes it did NOT resolve. Where a protected
prefix is itself reached through a symlink, the two never meet:

* macOS: ``/etc`` is a symlink to ``/private/etc``. ``/etc/passwd`` resolved
  to ``/private/etc/passwd``, which is not under ``Path("/etc")``, so the
  guard never fired on a Mac. The existing ``/etc`` tests failed there, and
  on this Mac ``test_protected_path_rejected`` went on to fetch its URL.
* any host whose ``$HOME`` is reached through a symlink: ``~/.ssh/...``
  resolved into the link's target, never under ``Path.home() / ".ssh"``.

THE PROPERTY
------------
For EVERY protected prefix, a destination under it is refused however it is
spelled: as written, as the resolved path, and through a symlink that points
at it. Cases that cannot differ on a given host (``/sys`` is not a symlink
anywhere this runs) pass on origin/main too. They are here because "every
prefix" is the claim, and a future prefix that IS a symlink somewhere must
not need its own test. The cases that fail on origin/main are the ``/etc``
ones on macOS and the symlinked-HOME one everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.tools.builtin import download_file
from prometheus.tools.builtin.download_file import _resolve_destination

URL = "https://example.com/probe.bin"

# Spelled out here rather than imported, so this file measures the guard on
# any revision, including one that predates the module constant.
PROTECTED_ROOTS = ("/etc", "/sys", "/boot", "/proc", "/dev")


def test_the_guard_still_protects_every_root_this_file_checks():
    """If the module's list shrinks, the tests below stop covering it."""
    assert set(PROTECTED_ROOTS) <= set(getattr(download_file, "_PROTECTED_ROOTS", ()))


@pytest.mark.parametrize("root", PROTECTED_ROOTS)
def test_a_destination_under_each_protected_root_is_refused(root):
    with pytest.raises(ValueError, match="protected path"):
        _resolve_destination(URL, f"{root}/prometheus-probe.bin")


@pytest.mark.parametrize("root", PROTECTED_ROOTS)
def test_the_resolved_spelling_of_each_protected_root_is_refused(root):
    """/private/etc/... on macOS: the path the old check actually compared."""
    resolved = Path(root).resolve() / "prometheus-probe.bin"
    with pytest.raises(ValueError, match="protected path"):
        _resolve_destination(URL, str(resolved))


@pytest.mark.parametrize("root", PROTECTED_ROOTS)
def test_a_symlink_into_each_protected_root_is_refused(root, tmp_path):
    """A writable-looking path whose real target is protected."""
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="protected path"):
        _resolve_destination(URL, str(alias / "prometheus-probe.bin"))


def test_ssh_under_a_symlinked_home_is_refused(tmp_path, monkeypatch):
    """~/.ssh when $HOME is a symlink: fails on origin/main on every OS."""
    real_home = tmp_path / "real-home"
    (real_home / ".ssh").mkdir(parents=True)
    linked_home = tmp_path / "home-link"
    linked_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(linked_home))

    for spelling in ("~/.ssh/authorized_keys",
                     str(linked_home / ".ssh" / "authorized_keys"),
                     str(real_home / ".ssh" / "authorized_keys")):
        with pytest.raises(ValueError, match="protected path"):
            _resolve_destination(URL, spelling)


def test_an_ordinary_destination_is_still_allowed(tmp_path):
    """The guard refuses protected paths, not everything that resolves."""
    dest = tmp_path / "downloads" / "file.bin"
    assert _resolve_destination(URL, str(dest)) == dest.resolve()
