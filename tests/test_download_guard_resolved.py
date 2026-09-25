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

* macOS volumes fold case, and firmlinks give one directory two names:
  ``~/.SSH`` is ``~/.ssh``, ``/private/ETC`` and
  ``/System/Volumes/Data/private/etc`` are ``/private/etc``. ``resolve()``
  follows symlinks but folds neither, so no string comparison catches these;
  the guard also compares by file identity (device, inode), and for a
  protected directory that does not exist yet, by its parent's identity and
  its name folded for case.

WHAT IS MEASURED
----------------
For every protected root, a destination under it is refused when written as
the root, as the resolved root, and through a symlink pointing at it; plus
``~/.ssh`` under a symlinked ``$HOME``, a case-folded ``~/.SSH`` where the
filesystem folds case, ``~/.SSH`` (and ``~/.ſſh``) when ``~/.ssh`` does not
exist yet, and the Data-volume spelling of ``/etc`` where it exists. Other spellings are not claimed. Cases that cannot differ on a given
host (``/sys`` is not a symlink anywhere this runs) pass on origin/main too;
they are here so a root that IS a symlink somewhere needs no new test. The
roots come from this file's list AND the module's, so a root added to the
module is exercised without editing this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.tools.builtin import download_file
from prometheus.tools.builtin.download_file import _resolve_destination

URL = "https://example.com/probe.bin"

# This file's own list (so it measures any revision, including one without the
# module constant) UNION the module's (so a root added there is exercised too).
# A root dropped from the module fails its cases below: it is still listed here.
# A revision without the constant (origin/main) falls back to this file's list;
# a revision that has the new guard but renamed the constant must fail loudly
# here, not quietly shrink to the five roots below.
assert hasattr(download_file, "_PROTECTED_ROOTS") or not hasattr(
    download_file, "_protected_prefixes"), (
    "download_file has _protected_prefixes but no _PROTECTED_ROOTS: update this "
    "file so the roots the module protects are still the roots it tests")
PROTECTED_ROOTS = tuple(sorted(
    {"/etc", "/sys", "/boot", "/proc", "/dev"}
    | set(getattr(download_file, "_PROTECTED_ROOTS", ()))
))


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


def test_a_case_folded_spelling_of_a_not_yet_created_ssh_is_refused(tmp_path, monkeypatch):
    """A fresh account has no ~/.ssh, so there is no directory to compare
    identities with, and ~/.SSH/authorized_keys would CREATE it: on a volume
    that folds case, as the directory sshd reads as ~/.ssh. Refused on every
    OS (on a case-sensitive one ~/.SSH is only a confusing name)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for spelling in ("~/.SSH/authorized_keys", "~/.\u017f\u017fh/authorized_keys"):
        with pytest.raises(ValueError, match="protected path"):
            _resolve_destination(URL, spelling)
    assert not (home / ".ssh").exists()


def test_a_case_folded_spelling_of_ssh_is_refused(tmp_path, monkeypatch):
    """~/.SSH on a volume that folds case is the same directory as ~/.ssh."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    folded = home / ".SSH"
    if not (folded.exists() and folded.samefile(home / ".ssh")):
        pytest.skip("this filesystem is case-sensitive: .SSH is a different "
                    "directory here, not a spelling of .ssh")
    with pytest.raises(ValueError, match="protected path"):
        _resolve_destination(URL, "~/.SSH/authorized_keys")


@pytest.mark.skipif(not Path("/System/Volumes/Data/private/etc").is_dir(),
                    reason="no macOS Data-volume firmlink on this host")
def test_the_data_volume_spelling_of_etc_is_refused():
    """/System/Volumes/Data/private/etc is /private/etc through a firmlink."""
    with pytest.raises(ValueError, match="protected path"):
        _resolve_destination(URL, "/System/Volumes/Data/private/etc/prometheus-probe.bin")


def test_an_ordinary_destination_is_still_allowed(tmp_path):
    """The guard refuses protected paths, not everything that resolves."""
    dest = tmp_path / "downloads" / "file.bin"
    assert _resolve_destination(URL, str(dest)) == dest.resolve()
